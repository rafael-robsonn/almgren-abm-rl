use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use abm_gym::{ExecutionConfig, SliceActions, calibrate, run_episodes};
use burn::module::Module;
use rl_ppo_burn::run::{RunManifest, arg, calibration_seeds, test_seeds, validation_seeds};
use rl_ppo_burn::{PpoConfig, PpoPolicy, Trainer, benchmark, default_backend, device, markdown};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let backend: String = arg(&args, "--backend", default_backend().to_string());
    let out = PathBuf::from(arg(&args, "--out", "runs/ppo".to_string()));
    let kappa_t: f64 = arg(&args, "--kappa-t", 1.0);
    let n_val: usize = arg(&args, "--val-episodes", 256);
    let n_test: usize = arg(&args, "--test-episodes", 1024);
    let eval_every: usize = arg(&args, "--eval-every", 10);
    let mut ppo = PpoConfig::default();
    ppo.iterations = arg(&args, "--iterations", ppo.iterations);
    ppo.n_envs = arg(&args, "--envs", ppo.n_envs);
    ppo.minibatch_envs = arg(&args, "--minibatch", ppo.minibatch_envs);
    ppo.epochs = arg(&args, "--epochs", ppo.epochs);
    ppo.seed = arg(&args, "--seed", ppo.seed);
    ppo.hidden = arg(&args, "--hidden", ppo.hidden);
    ppo.learning_rate = arg(&args, "--lr", ppo.learning_rate);
    ppo.entropy_coef = arg(&args, "--entropy", ppo.entropy_coef);
    fs::create_dir_all(&out).expect("create output directory");

    let t_cal = Instant::now();
    let mut exec = ExecutionConfig::default();
    let cal = calibrate(&exec, &calibration_seeds(64));
    let lambda = cal
        .ac_params(0.0)
        .lambda_for_kappa(kappa_t / exec.horizon_secs, exec.tau());
    exec.risk_aversion = lambda;
    exec.sigma = cal.sigma;
    println!(
        "calibration {:.2}s: sigma={:.4e} $/sqrt(s) epsilon={:.4e} $ eta={:.4e} $s gamma={:.4e} $ | lambda={lambda:.4e} (kappa*T={kappa_t})",
        t_cal.elapsed().as_secs_f64(),
        cal.sigma,
        cal.epsilon,
        cal.eta,
        cal.gamma
    );

    let actions = SliceActions::default();
    let val = validation_seeds(n_val);
    let twap_val = run_episodes(&exec, &mut abm_gym::Twap, &val, 128);
    println!(
        "TWAP validation objective {:.3} bps over {n_val} episodes",
        twap_val.iter().map(|s| s.objective_bps()).sum::<f64>() / twap_val.len() as f64
    );
    let dev = device(&backend);
    let mut trainer = Trainer::new(ppo.clone(), exec.clone(), actions.clone(), dev.clone());
    let mut manifest = RunManifest {
        backend: backend.clone(),
        kappa_t,
        lambda,
        ppo: ppo.clone(),
        exec: exec.clone(),
        actions: actions.clone(),
        model: trainer.model_config,
        calibration: cal.clone(),
        validation_episodes: n_val,
        test_episodes: n_test,
        best_iteration: None,
        best_validation_objective_bps: None,
        train_wall_secs: 0.0,
    };

    let mut csv = fs::File::create(out.join("train.csv")).expect("create train.csv");
    writeln!(
        csv,
        "iteration,episodes,mean_is_bps,std_is_bps,mean_objective_bps,val_objective_bps,mean_multiplier,lr,policy_loss,value_loss,entropy,approx_kl,clip_fraction,epochs,rollout_s,update_s"
    )
    .unwrap();
    let t0 = Instant::now();
    for it in 0..ppo.iterations {
        let log = trainer.iteration();
        let mut val_obj = f64::NAN;
        if it % eval_every == 0 || it + 1 == ppo.iterations {
            let mut pol = PpoPolicy::new(
                trainer.inference_model(),
                trainer.inference_device().clone(),
                actions.clone(),
                true,
                0,
            );
            let r = run_episodes(&exec, &mut pol, &val, 128);
            val_obj = r.iter().map(|s| s.objective_bps()).sum::<f64>() / r.len() as f64;
            if manifest.best_validation_objective_bps.is_none_or(|b| val_obj < b) {
                manifest.best_validation_objective_bps = Some(val_obj);
                manifest.best_iteration = Some(it);
                trainer
                    .inference_model()
                    .save_file(out.join("model.bpk"))
                    .expect("save model");
            }
        }
        let u = &log.update;
        writeln!(
            csv,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3e},{:.5},{:.5},{:.5},{:.5},{:.4},{},{:.3},{:.3}",
            log.iteration,
            log.episodes,
            log.mean_shortfall_bps,
            log.std_shortfall_bps,
            log.mean_objective_bps,
            val_obj,
            log.mean_multiplier,
            log.learning_rate,
            u.policy_loss,
            u.value_loss,
            u.entropy,
            u.approx_kl,
            u.clip_fraction,
            u.epochs,
            log.rollout_secs,
            log.update_secs
        )
        .unwrap();
        println!(
            "it {:>4} | IS {:>7.3} bps (sd {:>6.2}) | val {:>7.3} | mult {:.3} | H {:.3} | KL {:.4} | v_loss {:.4} | {:.2}s+{:.2}s",
            it,
            log.mean_shortfall_bps,
            log.std_shortfall_bps,
            val_obj,
            log.mean_multiplier,
            u.entropy,
            u.approx_kl,
            u.value_loss,
            log.rollout_secs,
            log.update_secs
        );
    }
    manifest.train_wall_secs = t0.elapsed().as_secs_f64();
    fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    println!(
        "training {:.1}s; best validation objective {:.3} bps at iteration {:?}",
        manifest.train_wall_secs,
        manifest.best_validation_objective_bps.unwrap_or(f64::NAN),
        manifest.best_iteration
    );

    let model = manifest.model.init(&dev).load_file(out.join("model.bpk"));
    let mut greedy = PpoPolicy::new(model.clone(), dev.clone(), actions.clone(), true, 1);
    let mut sampled = PpoPolicy::new(model, dev, actions, false, 2);
    let b = benchmark(
        &exec,
        &cal,
        kappa_t,
        &[0.5, 1.0, 2.0, 4.0],
        &test_seeds(n_test),
        128,
        vec![&mut greedy, &mut sampled],
    );
    fs::write(out.join("results.json"), serde_json::to_string_pretty(&b).unwrap()).unwrap();
    let md = markdown(&b);
    fs::write(out.join("results.md"), &md).unwrap();
    println!("{md}");
}
