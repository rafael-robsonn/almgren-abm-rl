use abm_gym::{
    AcParams, AlmgrenChriss, Calibration, EpisodeSummary, ExecutionConfig, PairedDiff, Policy,
    ShortfallStats, Twap, almgren_chriss, lambda_bps, paired_difference, run_episodes_profiled,
    shortfall_stats,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRow {
    pub policy: String,
    pub stats: ShortfallStats,
    pub vs_twap: PairedDiff,
    pub vs_ac: PairedDiff,
    pub mean_child_orders: f64,
    pub holdings_profile: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Benchmark {
    pub episodes: usize,
    pub kappa_t: f64,
    pub lambda: f64,
    pub lambda_bps: f64,
    pub ac_model_twap_cost_bps: f64,
    pub ac_model_ac_cost_bps: f64,
    pub ac_model_ac_std_bps: f64,
    pub rows: Vec<BenchmarkRow>,
}

/// Evaluates TWAP, the Almgren-Chriss frontier (`kappa T` in `frontier`) and the supplied
/// learned policies on the same seeds. Paired differences are taken against TWAP and against
/// the Almgren-Chriss policy at urgency `kappa_t`, which shares the training objective.
pub fn benchmark(
    exec: &ExecutionConfig,
    cal: &Calibration,
    kappa_t: f64,
    frontier: &[f64],
    seeds: &[u64],
    batch: usize,
    learned: Vec<&mut dyn Policy>,
) -> Benchmark {
    let tau = exec.tau();
    let base = cal.ac_params(0.0);
    let notional = exec.quantity as f64 * exec.market.initial_price as f64 * exec.tick();
    let model_cost = |p: &AcParams| {
        let h = almgren_chriss::holdings(exec.quantity as f64, exec.horizon_secs, exec.n_steps, p);
        (
            almgren_chriss::expected_cost(&h, tau, p) / notional * 1e4,
            almgren_chriss::variance(&h, tau, p).sqrt() / notional * 1e4,
        )
    };

    let mut runs: Vec<(String, Vec<EpisodeSummary>, Vec<f64>)> = Vec::new();
    let (r, p) = run_episodes_profiled(exec, &mut Twap, seeds, batch);
    runs.push(("TWAP (AC, kappa*T=0)".into(), r, p));
    let mut points: Vec<f64> = frontier.to_vec();
    if !points.iter().any(|k| (k - kappa_t).abs() < 1e-12) {
        points.push(kappa_t);
    }
    points.sort_by(|a, b| a.partial_cmp(b).expect("finite urgency"));
    let mut ac_index = 0;
    for k in points {
        let l = base.lambda_for_kappa(k / exec.horizon_secs, tau);
        let mut ac = AlmgrenChriss::new(cal.ac_params(l), exec.quantity, exec.horizon_secs, exec.n_steps);
        let (r, p) = run_episodes_profiled(exec, &mut ac, seeds, batch);
        if (k - kappa_t).abs() < 1e-12 {
            ac_index = runs.len();
        }
        runs.push((format!("AC kappa*T={k}"), r, p));
    }
    for pol in learned {
        let (r, p) = run_episodes_profiled(exec, pol, seeds, batch);
        runs.push((pol.name(), r, p));
    }
    let lb = lambda_bps(exec);
    let twap = runs[0].1.clone();
    let ac = runs[ac_index].1.clone();
    let rows = runs
        .into_iter()
        .map(|(name, r, p)| BenchmarkRow {
            stats: shortfall_stats(&r, lb),
            vs_twap: paired_difference(&r, &twap),
            vs_ac: paired_difference(&r, &ac),
            mean_child_orders: r.iter().map(|s| s.child_orders as f64).sum::<f64>() / r.len() as f64,
            holdings_profile: p,
            policy: name,
        })
        .collect();
    let (twap_cost, _) = model_cost(&base);
    let (ac_cost, ac_std) = model_cost(&cal.ac_params(exec.risk_aversion));
    Benchmark {
        episodes: seeds.len(),
        kappa_t,
        lambda: exec.risk_aversion,
        lambda_bps: lb,
        ac_model_twap_cost_bps: twap_cost,
        ac_model_ac_cost_bps: ac_cost,
        ac_model_ac_std_bps: ac_std,
        rows,
    }
}

fn t(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

pub fn markdown(b: &Benchmark) -> String {
    let mut s = format!(
        "{} episodes on common seeds; lambda = {:.3e} 1/$ (kappa*T = {}), lambda_bps = {:.3e}.\n\n",
        b.episodes, b.lambda, b.kappa_t, b.lambda_bps
    );
    s.push_str("| Policy | E[IS] (bps) | SE | Std[IS] | p5 | p50 | p95 | E[IS] + lambda_bps Var[IS] | IS - IS_TWAP (t) | IS - IS_AC (t) | child orders |\n");
    s.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in &b.rows {
        s.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} | {:.2} | {:.3} | {:+.3} ({:+.1}) | {:+.3} ({:+.1}) | {:.1} |\n",
            r.policy,
            r.stats.mean_bps,
            r.stats.se_bps,
            r.stats.std_bps,
            r.stats.p05_bps,
            r.stats.p50_bps,
            r.stats.p95_bps,
            r.stats.mean_variance_bps,
            r.vs_twap.mean_bps,
            t(r.vs_twap.t_stat),
            r.vs_ac.mean_bps,
            t(r.vs_ac.t_stat),
            r.mean_child_orders,
        ));
    }
    let n = b.rows[0].holdings_profile.len();
    let ks: Vec<usize> = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, n - 1]
        .into_iter()
        .filter(|k| *k < n)
        .collect();
    s.push_str("\nMean fraction of the order held after step k:\n\n| Policy |");
    for k in &ks {
        s.push_str(&format!(" k={k} |"));
    }
    s.push_str("\n|---|");
    s.push_str(&"---:|".repeat(ks.len()));
    s.push('\n');
    for r in &b.rows {
        s.push_str(&format!("| {} |", r.policy));
        for k in &ks {
            s.push_str(&format!(" {:.3} |", r.holdings_profile[*k]));
        }
        s.push('\n');
    }
    s.push_str(&format!(
        "\nAlmgren-Chriss model prediction with calibrated parameters: E[IS] TWAP {:.2} bps; AC(kappa*T={}) {:.2} bps, Std {:.2} bps.\n",
        b.ac_model_twap_cost_bps, b.kappa_t, b.ac_model_ac_cost_bps, b.ac_model_ac_std_bps
    ));
    s
}
