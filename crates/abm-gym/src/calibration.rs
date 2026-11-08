use market_agents::Market;
use matching_engine::{LatencyModel, Report, Request, Side, secs};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::almgren_chriss::AcParams;
use crate::env::ExecutionConfig;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImpactPoint {
    pub x: f64,
    pub mean: f64,
    pub se: f64,
}

/// Almgren-Chriss parameters estimated on the simulator (all prices in $).
/// `sigma^2 = E[(dm)^2] / tau` from mid increments over `tau`;
/// temporary impact from the OLS fit `c(q) = epsilon + (eta / tau) q` of the immediate
/// per-share cost of a sell market order of size `q` against the pre-trade mid;
/// `gamma` is the per-share mid displacement `tau` seconds after a sell of `decay_size`,
/// measured against an unperturbed clone of the same market (common random numbers).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub tau: f64,
    pub n_seeds: usize,
    pub sigma: f64,
    pub epsilon: f64,
    pub eta: f64,
    pub gamma: f64,
    pub temporary_r2: f64,
    pub temporary: Vec<ImpactPoint>,
    pub decay_size: u64,
    pub decay: Vec<ImpactPoint>,
}

impl Calibration {
    pub fn ac_params(&self, lambda: f64) -> AcParams {
        AcParams {
            sigma: self.sigma,
            eta: self.eta,
            gamma: self.gamma,
            epsilon: self.epsilon,
            lambda,
        }
    }
}

const PROBES: [u64; 6] = [25, 50, 100, 150, 200, 300];
const HORIZONS: [f64; 8] = [1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0];

struct SeedSample {
    increments: Vec<f64>,
    costs: Vec<f64>,
    decay: Vec<f64>,
}

fn mid_of(m: &Market, fallback: f64) -> f64 {
    m.mid().unwrap_or(fallback)
}

fn sample(config: &ExecutionConfig, seed: u64, decay_size: u64, sigma_window: f64) -> SeedSample {
    let tick = config.tick();
    let tau = config.tau();
    let mut base = Market::new(config.market.clone(), seed);
    base.run_until(secs(config.warmup_secs));
    let probe = base.register_external(Some(LatencyModel::new(config.latency_base_ns, 0)));
    let t0 = base.now();
    let m0 = mid_of(&base, config.market.initial_price as f64);

    let mut m = base.clone();
    let mut prev = m0;
    let mut increments = Vec::new();
    for k in 1..=(sigma_window / tau) as usize {
        m.run_until(t0 + secs(k as f64 * tau));
        let x = mid_of(&m, prev);
        increments.push((x - prev) * tick);
        prev = x;
    }

    let mut reports = Vec::new();
    let costs = PROBES
        .iter()
        .map(|&q| {
            let mut m = base.clone();
            m.submit(probe, Request::Market { side: Side::Sell, qty: q });
            m.run_until(t0 + secs(1.0));
            reports.clear();
            m.drain_external(&mut reports);
            let (mut px, mut n) = (0.0, 0u64);
            for (_, r) in &reports {
                if let Report::Fill { price, qty, .. } = *r {
                    px += price as f64 * qty as f64;
                    n += qty;
                }
            }
            if n == 0 { f64::NAN } else { (m0 - px / n as f64) * tick }
        })
        .collect();

    let mut a = base.clone();
    let mut b = base;
    a.submit(probe, Request::Market { side: Side::Sell, qty: decay_size });
    let decay = HORIZONS
        .iter()
        .map(|&h| {
            a.run_until(t0 + secs(h));
            b.run_until(t0 + secs(h));
            (mid_of(&b, m0) - mid_of(&a, m0)) * tick / decay_size as f64
        })
        .collect();
    SeedSample {
        increments,
        costs,
        decay,
    }
}

fn mean_se(xs: &[f64]) -> (f64, f64) {
    let v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, (var / n).sqrt())
}

pub fn calibrate(config: &ExecutionConfig, seeds: &[u64]) -> Calibration {
    let tau = config.tau();
    let decay_size = (3 * config.quantity / config.n_steps as u64).max(1);
    let samples: Vec<SeedSample> = seeds
        .par_iter()
        .map(|&s| sample(config, s, decay_size, 1_800.0))
        .collect();

    let incs: Vec<f64> = samples.iter().flat_map(|s| s.increments.iter().copied()).collect();
    let sigma = (incs.iter().map(|x| x * x).sum::<f64>() / incs.len() as f64 / tau).sqrt();

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let temporary: Vec<ImpactPoint> = PROBES
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let c: Vec<f64> = samples.iter().map(|s| s.costs[i]).collect();
            for v in c.iter().filter(|v| v.is_finite()) {
                xs.push(q as f64);
                ys.push(*v);
            }
            let (mean, se) = mean_se(&c);
            ImpactPoint { x: q as f64, mean, se }
        })
        .collect();
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let ss_res: f64 = xs.iter().zip(&ys).map(|(x, y)| (y - intercept - slope * x).powi(2)).sum();
    let ss_tot: f64 = ys.iter().map(|y| (y - my).powi(2)).sum();

    let decay: Vec<ImpactPoint> = HORIZONS
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            let d: Vec<f64> = samples.iter().map(|s| s.decay[i]).collect();
            let (mean, se) = mean_se(&d);
            ImpactPoint { x: h, mean, se }
        })
        .collect();
    let gamma = decay
        .iter()
        .find(|p| (p.x - tau).abs() < 1e-9)
        .map_or(decay[2].mean, |p| p.mean)
        .max(0.0);

    Calibration {
        tau,
        n_seeds: seeds.len(),
        sigma,
        epsilon: intercept.max(0.0),
        eta: slope * tau,
        gamma,
        temporary_r2: 1.0 - ss_res / ss_tot,
        temporary,
        decay_size,
        decay,
    }
}
