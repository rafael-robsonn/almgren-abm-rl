use std::io::Write;

use market_agents::MarketConfig;
use serde::Serialize;
use stylized_facts::simulate::{mid_paths, pooled_returns};
use stylized_facts::{Criteria, ScaleReport, Verdict, analyse, verdict};

#[derive(Serialize)]
struct Output {
    seeds: Vec<u64>,
    warmup_secs: f64,
    horizon_secs: f64,
    criteria: Criteria,
    scales: Vec<(ScaleReport, Verdict)>,
}

fn arg<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == key)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_seeds: u64 = arg(&args, "--seeds", 32);
    let base: u64 = arg(&args, "--base-seed", 100);
    let horizon: f64 = arg(&args, "--horizon", 20_000.0);
    let warmup: f64 = arg(&args, "--warmup", 600.0);
    let out: String = arg(&args, "--out", String::new());

    let config = MarketConfig::default();
    let seeds: Vec<u64> = (0..n_seeds).map(|i| base + i).collect();
    let t0 = std::time::Instant::now();
    let paths = mid_paths(&config, &seeds, warmup, horizon, 1.0);
    let wall = t0.elapsed().as_secs_f64();
    let criteria = Criteria::default();

    println!(
        "{} paths x {horizon:.0} s (warmup {warmup:.0} s), simulated in {wall:.2} s wall",
        seeds.len()
    );
    println!();
    println!("| dt (s) | n | excess kurtosis | Hill alpha | JB p | rho_r(1) | max_k<=10 abs rho_r(k) | rho_abs(1) | rho_abs(10) | rho_abs(50) | Q_LB(r,10) | Q_LB(abs r,10) | F/N/V |");
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|");
    let mut scales = Vec::new();
    for k in [1usize, 10, 30, 60, 300] {
        let r = pooled_returns(&paths, k);
        let rep = analyse(&r, k as f64, 100, 10);
        let v = verdict(&rep, &criteria);
        let max_abs = rep.acf_returns[..10]
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()));
        let flag = |b: bool| if b { "Y" } else { "n" };
        println!(
            "| {k} | {} | {:.2} | {:.2} | {:.1e} | {:+.3} | {:.3} | {:+.3} | {:+.3} | {:+.3} | {:.1} | {:.1} | {}/{}/{} |",
            rep.n,
            rep.moments.excess_kurtosis,
            rep.hill_alpha,
            rep.jarque_bera.p_value,
            rep.acf_returns[0],
            max_abs,
            rep.acf_abs[0],
            rep.acf_abs[9],
            rep.acf_abs[49],
            rep.ljung_box_returns.statistic,
            rep.ljung_box_abs.statistic,
            flag(v.fat_tails),
            flag(v.no_linear_autocorrelation),
            flag(v.volatility_clustering),
        );
        scales.push((rep, v));
    }
    if !out.is_empty() {
        let o = Output {
            seeds,
            warmup_secs: warmup,
            horizon_secs: horizon,
            criteria,
            scales,
        };
        let mut f = std::fs::File::create(&out).expect("cannot create output file");
        f.write_all(serde_json::to_string_pretty(&o).unwrap().as_bytes())
            .unwrap();
    }
}
