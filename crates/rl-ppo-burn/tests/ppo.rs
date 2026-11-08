use abm_gym::{ExecutionConfig, OBS_DIM, SliceActions};
use burn::tensor::{Tensor, TensorData};
use market_agents::Rng;
use rl_ppo_burn::{ModelConfig, PpoConfig, Rollout, Trainer, default_backend, device, gae, sample_row};

#[test]
fn recurrent_step_reproduces_sequence_pass() {
    let dev = device(default_backend());
    dev.seed(3);
    let model = ModelConfig { obs_dim: OBS_DIM, hidden: 16, n_actions: 6 }.init(&dev);
    let (b, l) = (3, 7);
    let mut rng = Rng::new(1);
    let raw: Vec<f32> = (0..b * l * OBS_DIM).map(|_| rng.normal() as f32).collect();
    let seq = Tensor::<3>::from_data(TensorData::new(raw.clone(), [b, l, OBS_DIM]), &dev);
    let (logits, values) = model.forward_sequence(seq);
    let logits = logits.into_data().try_into_vec_as::<f32>().unwrap();
    let values = values.into_data().try_into_vec_as::<f32>().unwrap();
    let mut h = Tensor::<2>::zeros([b, 16], &dev);
    for t in 0..l {
        let mut x = Vec::with_capacity(b * OBS_DIM);
        for i in 0..b {
            let s = (i * l + t) * OBS_DIM;
            x.extend_from_slice(&raw[s..s + OBS_DIM]);
        }
        let x = Tensor::<2>::from_data(TensorData::new(x, [b, OBS_DIM]), &dev);
        let (lg, v, next) = model.step(x, h);
        h = next;
        let lg = lg.into_data().try_into_vec_as::<f32>().unwrap();
        let v = v.into_data().try_into_vec_as::<f32>().unwrap();
        for i in 0..b {
            assert!((v[i] - values[i * l + t]).abs() < 1e-5);
            for a in 0..6 {
                assert!((lg[i * 6 + a] - logits[(i * l + t) * 6 + a]).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn gae_matches_hand_computation() {
    let r = Rollout {
        n: 1,
        len: 4,
        obs: vec![0.0; 4 * OBS_DIM],
        actions: vec![0; 4],
        logp: vec![0.0; 4],
        values: vec![0.5, 0.2, -0.1, 0.0],
        rewards: vec![1.0, -1.0, 2.0, 0.0],
        dones: vec![false, false, true, false],
        mask: vec![1.0, 1.0, 1.0, 0.0],
        summaries: vec![],
    };
    let (g, lam) = (0.9, 0.8);
    let (adv, ret) = gae(&r, g, lam);
    let d2 = 2.0 - (-0.1);
    let d1 = -1.0 + g * -0.1 - 0.2;
    let d0 = 1.0 + g * 0.2 - 0.5;
    let a2 = d2;
    let a1 = d1 + g * lam * a2;
    let a0 = d0 + g * lam * a1;
    for (x, y) in adv.iter().zip([a0, a1, a2, 0.0]) {
        assert!((*x as f64 - y).abs() < 1e-6, "{adv:?}");
    }
    assert!((ret[0] as f64 - (a0 + 0.5)).abs() < 1e-6 && ret[3] == 0.0);
}

#[test]
fn sampling_matches_softmax_and_mode() {
    let logits = [0.0f32, 1.0, 2.0];
    let mut rng = Rng::new(5);
    let mut counts = [0usize; 3];
    for _ in 0..60_000 {
        counts[sample_row(&logits, &mut rng, false).0] += 1;
    }
    let z: f64 = logits.iter().map(|l| (*l as f64).exp()).sum();
    for (c, l) in counts.iter().zip(logits) {
        let p = (l as f64).exp() / z;
        assert!((*c as f64 / 60_000.0 - p).abs() < 0.01);
    }
    let (a, lp) = sample_row(&logits, &mut rng, true);
    assert_eq!(a, 2);
    assert!((lp as f64 - (2.0 - z.ln())).abs() < 1e-6);
}

#[test]
fn training_loop_smoke() {
    let exec = ExecutionConfig {
        quantity: 600,
        horizon_secs: 60.0,
        n_steps: 6,
        warmup_secs: 60.0,
        ..ExecutionConfig::default()
    };
    let ppo = PpoConfig { n_envs: 8, iterations: 2, minibatch_envs: 4, hidden: 16, ..PpoConfig::default() };
    let mut trainer = Trainer::new(ppo, exec, SliceActions::default(), device(default_backend()));
    for _ in 0..2 {
        let log = trainer.iteration();
        assert_eq!(log.episodes, 8);
        assert!(log.mean_shortfall_bps.is_finite());
        assert!(log.update.value_loss.is_finite() && log.update.entropy > 0.0);
    }
}
