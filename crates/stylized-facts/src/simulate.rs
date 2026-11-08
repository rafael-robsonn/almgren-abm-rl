use market_agents::{Market, MarketConfig};
use matching_engine::secs;
use rayon::prelude::*;

/// Mid-price path of one market sampled every `dt` seconds after `warmup` seconds, carrying the
/// last two-sided mid forward when a side is momentarily empty.
pub fn mid_path(config: &MarketConfig, seed: u64, warmup: f64, horizon: f64, dt: f64) -> Vec<f64> {
    let mut m = Market::new(config.clone(), seed);
    m.run_until(secs(warmup));
    let n = (horizon / dt).floor() as usize;
    let mut out = Vec::with_capacity(n + 1);
    let mut last = m.mid().unwrap_or(config.initial_price as f64);
    out.push(last);
    for i in 1..=n {
        m.run_until(secs(warmup + i as f64 * dt));
        if let Some(x) = m.mid() {
            last = x;
        }
        out.push(last);
    }
    out
}

pub fn mid_paths(config: &MarketConfig, seeds: &[u64], warmup: f64, horizon: f64, dt: f64) -> Vec<Vec<f64>> {
    seeds
        .par_iter()
        .map(|&s| mid_path(config, s, warmup, horizon, dt))
        .collect()
}

/// Log returns at `k * dt` for each path, concatenated without crossing path boundaries.
pub fn pooled_returns(paths: &[Vec<f64>], k: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for p in paths {
        let sub: Vec<f64> = p.iter().step_by(k.max(1)).copied().collect();
        out.extend(crate::stats::log_returns(&sub));
    }
    out
}
