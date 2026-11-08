use std::fs;
use std::path::PathBuf;

use abm_gym::{FEATURE_NAMES, Observation, Policy, VecEnv};
use burn::module::Module;
use rl_ppo_burn::run::{RunManifest, arg, test_seeds};
use rl_ppo_burn::{PpoPolicy, device};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--run", "runs/ppo".to_string()));
    let n: usize = arg(&args, "--episodes", 1024);
    let manifest: RunManifest = serde_json::from_str(
        &fs::read_to_string(dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    let dev = device(&manifest.backend);
    let model = manifest.model.init(&dev).load_file(dir.join("model.bpk"));
    let mut policy = PpoPolicy::new(model, dev, manifest.actions.clone(), true, 1);

    let mut rows: Vec<(Observation, f64)> = Vec::new();
    for chunk in test_seeds(n).chunks(128) {
        let mut env = VecEnv::new(manifest.exec.clone(), chunk.len());
        let mut obs = env.reset(chunk);
        policy.reset(chunk.len());
        while !env.all_done() {
            let ctx = env.contexts();
            let active: Vec<bool> = env.dones().iter().map(|d| !d).collect();
            let q = policy.act(&obs, &ctx, &active);
            for i in 0..chunk.len() {
                if active[i] && ctx[i].step + 1 < ctx[i].n_steps && ctx[i].remaining > 0 {
                    let pace = ctx[i].remaining as f64 / ctx[i].steps_left() as f64;
                    rows.push((obs[i], q[i] as f64 / pace));
                }
            }
            env.step(&q);
            obs = env.observations();
        }
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let all: Vec<f64> = rows.iter().map(|r| r.1).collect();
    println!(
        "{} decisions (forced last step excluded); mean size / TWAP pace = {:.3}; share of skipped epochs = {:.3}\n",
        rows.len(),
        mean(&all),
        all.iter().filter(|m| **m == 0.0).count() as f64 / all.len() as f64
    );
    println!("| feature | quintile | feature range | mean size / pace | share skipped |");
    println!("|---|---:|---|---:|---:|");
    for f in [2usize, 3, 4, 5, 6, 7, 9] {
        let mut v: Vec<(f32, f64)> = rows.iter().map(|r| (r.0[f], r.1)).collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite feature"));
        let k = v.len() / 5;
        for q in 0..5 {
            let part = &v[q * k..if q == 4 { v.len() } else { (q + 1) * k }];
            let m: Vec<f64> = part.iter().map(|x| x.1).collect();
            println!(
                "| {} | {} | [{:+.3}, {:+.3}] | {:.3} | {:.3} |",
                FEATURE_NAMES[f],
                q + 1,
                part[0].0,
                part[part.len() - 1].0,
                mean(&m),
                m.iter().filter(|x| **x == 0.0).count() as f64 / m.len() as f64
            );
        }
    }
}
