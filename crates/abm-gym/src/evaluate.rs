use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::env::{EpisodeSummary, ExecutionConfig};
use crate::policy::Policy;
use crate::vec_env::VecEnv;

/// Runs one episode per seed in batches of `batch` parallel environments. Seeds fix the market,
/// so two policies evaluated on the same seeds face common random numbers up to their own impact.
pub fn run_episodes(
    config: &ExecutionConfig,
    policy: &mut dyn Policy,
    seeds: &[u64],
    batch: usize,
) -> Vec<EpisodeSummary> {
    run_episodes_profiled(config, policy, seeds, batch).0
}

/// As [`run_episodes`], also returning the mean fraction of the order still held after each
/// decision step (`profile[0] = 1`, `profile[N] = 0`).
pub fn run_episodes_profiled(
    config: &ExecutionConfig,
    policy: &mut dyn Policy,
    seeds: &[u64],
    batch: usize,
) -> (Vec<EpisodeSummary>, Vec<f64>) {
    let mut out = Vec::with_capacity(seeds.len());
    let mut profile = vec![0.0; config.n_steps + 1];
    for chunk in seeds.chunks(batch.max(1)) {
        let mut env = VecEnv::new(config.clone(), chunk.len());
        let mut obs = env.reset(chunk);
        policy.reset(chunk.len());
        profile[0] += chunk.len() as f64;
        let mut step = 0;
        while !env.all_done() {
            let ctx = env.contexts();
            let active: Vec<bool> = env.dones().iter().map(|d| !d).collect();
            let q = policy.act(&obs, &ctx, &active);
            env.step(&q);
            step += 1;
            profile[step] += env
                .contexts()
                .iter()
                .map(|c| c.remaining as f64 / c.quantity as f64)
                .sum::<f64>();
            obs = env.observations();
        }
        out.extend(env.summaries());
    }
    let n = seeds.len().max(1) as f64;
    profile.iter_mut().for_each(|p| *p /= n);
    (out, profile)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShortfallStats {
    pub n: usize,
    pub mean_bps: f64,
    pub std_bps: f64,
    pub se_bps: f64,
    pub p05_bps: f64,
    pub p50_bps: f64,
    pub p95_bps: f64,
    pub mean_penalty_bps: f64,
    /// `E[IS] + lambda_bps * Var[IS]` with `lambda_bps = lambda * Q * P0 / 1e4`, the
    /// mean-variance criterion expressed in bps of arrival notional.
    pub mean_variance_bps: f64,
    pub residual_episodes: usize,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

pub fn shortfall_stats(results: &[EpisodeSummary], lambda_bps: f64) -> ShortfallStats {
    let n = results.len();
    let xs: Vec<f64> = results.iter().map(|r| r.shortfall_bps).collect();
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let mut sorted = xs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite shortfall"));
    ShortfallStats {
        n,
        mean_bps: mean,
        std_bps: var.sqrt(),
        se_bps: (var / n as f64).sqrt(),
        p05_bps: quantile(&sorted, 0.05),
        p50_bps: quantile(&sorted, 0.5),
        p95_bps: quantile(&sorted, 0.95),
        mean_penalty_bps: results.iter().map(|r| r.penalty_bps).sum::<f64>() / n as f64,
        mean_variance_bps: mean + lambda_bps * var,
        residual_episodes: results.iter().filter(|r| r.residual > 0).count(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairedDiff {
    pub n: usize,
    pub mean_bps: f64,
    pub se_bps: f64,
    pub t_stat: f64,
}

/// Seed-matched difference `IS(a) - IS(b)`; negative values favour `a`. `t_stat` is 0 when the
/// differences have zero dispersion (e.g. a policy compared with itself), so the result is
/// always finite and serialisable.
pub fn paired_difference(a: &[EpisodeSummary], b: &[EpisodeSummary]) -> PairedDiff {
    let bmap: HashMap<u64, f64> = b.iter().map(|r| (r.seed, r.shortfall_bps)).collect();
    let d: Vec<f64> = a
        .iter()
        .filter_map(|r| bmap.get(&r.seed).map(|x| r.shortfall_bps - x))
        .collect();
    let n = d.len();
    let mean = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let se = (var / n as f64).sqrt();
    PairedDiff {
        n,
        mean_bps: mean,
        se_bps: se,
        t_stat: if se > 0.0 { mean / se } else { 0.0 },
    }
}

/// `lambda_bps = lambda * Q * P0 / 1e4`: converts a $-denominated risk aversion into the
/// coefficient multiplying the variance of IS measured in bps.
pub fn lambda_bps(config: &ExecutionConfig) -> f64 {
    config.risk_aversion * config.quantity as f64 * config.market.initial_price as f64 * config.tick() / 1e4
}
