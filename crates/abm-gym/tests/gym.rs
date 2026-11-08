use abm_gym::{
    AlmgrenChriss, EpisodeSummary, ExecutionConfig, ExecutionEnv, PairedDiff, Policy, SliceActions,
    Twap, VecEnv, calibrate, paired_difference, run_episodes, shortfall_stats,
};

#[test]
fn vectorised_env_matches_sequential_rollouts() {
    let cfg = ExecutionConfig::default();
    let seeds: Vec<u64> = (0..12).map(|i| 31 + i).collect();
    let par = run_episodes(&cfg, &mut Twap, &seeds, 12);
    for (s, p) in seeds.iter().zip(&par) {
        let mut env = ExecutionEnv::new(cfg.clone());
        env.reset(*s);
        let mut twap = Twap;
        while !env.is_done() {
            let q = twap.act(&[env.observation()], &[env.context()], &[true])[0];
            env.step(q);
        }
        assert_eq!(env.summary(), *p);
    }
    assert_eq!(par, run_episodes(&cfg, &mut Twap, &seeds, 5));
}

#[test]
fn vec_env_reports_done_slots() {
    let cfg = ExecutionConfig {
        quantity: 200,
        ..ExecutionConfig::default()
    };
    let mut env = VecEnv::new(cfg.clone(), 3);
    env.reset(&[1, 2, 3]);
    let out = env.step(&[cfg.quantity, 0, 50]);
    assert_eq!(env.dones(), vec![true, false, false]);
    assert!(out[0].done && !out[1].done);
    let again = env.step(&[50, 50, 50]);
    assert_eq!(again[0].executed, 0);
}

#[test]
fn action_map_brackets_twap_pacing() {
    let a = SliceActions::default();
    let ctx = abm_gym::StepContext { step: 10, n_steps: 30, remaining: 2_000, quantity: 3_000 };
    let q: Vec<u64> = (0..a.len()).map(|i| a.quantity(i, &ctx)).collect();
    assert_eq!(q, vec![0, 50, 100, 150, 200, 300]);
}

#[test]
fn calibration_is_consistent_and_ac_reacts_to_risk_aversion() {
    let cfg = ExecutionConfig::default();
    let seeds: Vec<u64> = (0..24).map(|i| 70_000 + i).collect();
    let cal = calibrate(&cfg, &seeds);
    assert!(cal.sigma > 0.0 && cal.eta > 0.0 && cal.epsilon > 0.0);
    let p = cal.ac_params(0.0);
    assert!(p.eta_tilde(cal.tau) > 0.0, "{cal:?}");
    assert!(cal.decay[0].mean > cal.decay[3].mean, "impact must decay: {:?}", cal.decay);
    let costs: Vec<f64> = cal.temporary.iter().map(|t| t.mean).collect();
    assert!(costs.windows(2).all(|w| w[1] > w[0]), "cost must increase with size: {costs:?}");

    let lambda = p.lambda_for_kappa(3.0 / cfg.horizon_secs, cal.tau);
    let mut ac = AlmgrenChriss::new(cal.ac_params(lambda), cfg.quantity, cfg.horizon_secs, cfg.n_steps);
    let eval: Vec<u64> = (0..96).map(|i| 123_000 + i).collect();
    let r_ac = run_episodes(&cfg, &mut ac, &eval, 48);
    let r_tw = run_episodes(&cfg, &mut Twap, &eval, 48);
    let d = paired_difference(&r_ac, &r_tw);
    assert!(d.mean_bps > 0.0, "front-loading must cost more on average here: {d:?}");
    assert!(shortfall_stats(&r_ac, 0.0).residual_episodes == 0);
}

#[test]
fn self_comparison_is_finite_and_serialisable() {
    let r: Vec<EpisodeSummary> = (0..5)
        .map(|i| EpisodeSummary { seed: i, shortfall_bps: i as f64 * 1.5 - 2.0, ..Default::default() })
        .collect();
    let d = paired_difference(&r, &r);
    assert_eq!((d.n, d.mean_bps, d.t_stat), (5, 0.0, 0.0));
    let back: PairedDiff = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
    assert_eq!(back, d);
}
