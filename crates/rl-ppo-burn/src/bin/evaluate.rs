use std::fs;
use std::path::PathBuf;

use burn::module::Module;
use rl_ppo_burn::run::{RunManifest, arg, test_seeds};
use rl_ppo_burn::{PpoPolicy, benchmark, device, markdown};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--run", "runs/ppo".to_string()));
    let manifest: RunManifest = serde_json::from_str(
        &fs::read_to_string(dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    let backend: String = arg(&args, "--backend", manifest.backend.clone());
    let n_test: usize = arg(&args, "--test-episodes", manifest.test_episodes);
    let dev = device(&backend);
    let model = manifest.model.init(&dev).load_file(dir.join("model.bpk"));
    let mut greedy = PpoPolicy::new(model.clone(), dev.clone(), manifest.actions.clone(), true, 1);
    let mut sampled = PpoPolicy::new(model, dev, manifest.actions.clone(), false, 2);
    let b = benchmark(
        &manifest.exec,
        &manifest.calibration,
        manifest.kappa_t,
        &[0.5, 1.0, 2.0, 4.0],
        &test_seeds(n_test),
        128,
        vec![&mut greedy, &mut sampled],
    );
    println!("{}", markdown(&b));
}
