use std::fs;
use std::path::Path;

use abm_gym::{Schedule, Twap, paired_difference, run_episodes};
use rl_ppo_burn::Benchmark;
use rl_ppo_burn::run::{RunManifest, test_seeds};

fn main() {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    assert!(!dirs.is_empty(), "usage: seeds <run dir>...");
    let read = |dir: &str, file: &str| {
        fs::read_to_string(Path::new(dir).join(file)).unwrap_or_else(|e| panic!("{dir}/{file}: {e}"))
    };

    println!(
        "| run | backend | seed | best it. | val obj. | E[IS] | Std | IS - IS_TWAP (t) | IS - IS_AC (t) | mean-var. | residual | sampled IS - IS_TWAP (t) | max mean-path dev. | mean path as fixed schedule: IS - IS_TWAP (t) |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let mut deltas = Vec::new();
    let mut sampled = Vec::new();
    let mut schedule_deltas = Vec::new();
    let mut reference: Option<(usize, abm_gym::ExecutionConfig, Vec<abm_gym::EpisodeSummary>)> = None;
    for dir in &dirs {
        let m: RunManifest = serde_json::from_str(&read(dir, "manifest.json")).expect("parse manifest.json");
        let b: Benchmark = serde_json::from_str(&read(dir, "results.json")).expect("parse results.json");
        if reference.is_none() {
            let twap = run_episodes(&m.exec, &mut Twap, &test_seeds(b.episodes), 128);
            reference = Some((b.episodes, m.exec.clone(), twap));
        }
        let (episodes, exec, twap) = reference.as_ref().expect("reference set");
        assert!(
            *episodes == b.episodes && *exec == m.exec,
            "{dir}: test configuration differs from {}",
            dirs[0]
        );
        let row = |key: &str| {
            b.rows
                .iter()
                .find(|r| r.policy.contains(key))
                .unwrap_or_else(|| panic!("{dir}: no `{key}` row"))
        };
        let (base, mode, samp) = (row("TWAP"), row("(mode)"), row("(sampled)"));
        let max_dev = mode
            .holdings_profile
            .iter()
            .zip(&base.holdings_profile)
            .map(|(p, t)| p - t)
            .fold(0.0f64, |a, d| if d.abs() > a.abs() { d } else { a });
        let targets: Vec<f64> = mode.holdings_profile.iter().map(|h| h * exec.quantity as f64).collect();
        let replay = run_episodes(exec, &mut Schedule::new("mean path", targets), &test_seeds(*episodes), 128);
        let sd = paired_difference(&replay, twap);
        deltas.push(mode.vs_twap.mean_bps);
        sampled.push(samp.vs_twap.mean_bps);
        schedule_deltas.push(sd.mean_bps);
        println!(
            "| {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:+.3} ({:+.1}) | {:+.3} ({:+.1}) | {:.3} | {} | {:+.3} ({:+.1}) | {:+.3} | {:+.3} ({:+.1}) |",
            dir,
            m.backend,
            m.ppo.seed,
            m.best_iteration.map_or("-".into(), |i| i.to_string()),
            m.best_validation_objective_bps.unwrap_or(f64::NAN),
            mode.stats.mean_bps,
            mode.stats.std_bps,
            mode.vs_twap.mean_bps,
            mode.vs_twap.t_stat,
            mode.vs_ac.mean_bps,
            mode.vs_ac.t_stat,
            mode.stats.mean_variance_bps,
            mode.stats.residual_episodes,
            samp.vs_twap.mean_bps,
            samp.vs_twap.t_stat,
            max_dev,
            sd.mean_bps,
            sd.t_stat
        );
    }

    let summary = |name: &str, x: &[f64]| {
        let k = x.len() as f64;
        let mean = x.iter().sum::<f64>() / k;
        let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (k - 1.0).max(1.0)).sqrt();
        let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
        println!(
            "{name}: mean {mean:+.3} bps, sd across runs {sd:.3}, se {:.3}, range [{lo:+.3}, {hi:+.3}], negative in {}/{}",
            sd / k.sqrt(),
            x.iter().filter(|v| **v < 0.0).count(),
            x.len()
        );
    };
    println!();
    summary("IS(PPO mode) - IS(TWAP)", &deltas);
    summary("IS(PPO sampled) - IS(TWAP)", &sampled);
    summary("IS(mean path, fixed) - IS(TWAP)", &schedule_deltas);
}
