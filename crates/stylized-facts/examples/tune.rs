use market_agents::{Market, MarketConfig};
use matching_engine::{Side, secs};
use stylized_facts::simulate::{mid_paths, pooled_returns};
use stylized_facts::{Criteria, analyse, verdict};

fn set(v: &mut serde_json::Value, path: &str, raw: &str) {
    let mut cur = v;
    for k in path.split('.') {
        cur = cur.get_mut(k).unwrap_or_else(|| panic!("unknown key {path}"));
    }
    *cur = serde_json::from_str(raw).unwrap_or_else(|_| panic!("bad value {raw}"));
}

fn main() {
    let mut v = serde_json::to_value(MarketConfig::default()).unwrap();
    let mut seeds = 8u64;
    let mut horizon = 20_000.0;
    for a in std::env::args().skip(1) {
        let (k, val) = a.split_once('=').unwrap();
        match k {
            "seeds" => seeds = val.parse().unwrap(),
            "horizon" => horizon = val.parse().unwrap(),
            _ => set(&mut v, k, val),
        }
    }
    let cfg: MarketConfig = serde_json::from_value(v).unwrap();

    let mut m = Market::new(cfg.clone(), 1);
    let mut spreads = Vec::new();
    let mut depth = Vec::new();
    let mut top = Vec::new();
    for i in 1..=3_600 {
        m.run_until(secs(600.0 + i as f64));
        if let Some(s) = m.book().spread() {
            spreads.push(s as f64);
        }
        depth.push((m.book().qty_within(Side::Buy, 5) + m.book().qty_within(Side::Sell, 5)) as f64 / 2.0);
        top.push((m.book().best_qty(Side::Buy) + m.book().best_qty(Side::Sell)) as f64 / 2.0);
    }
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    println!(
        "spread {:.2} | depth5/side {:.0} | touch qty {:.0} | msgs/s {:.0} | vol/s {:.1}",
        mean(&spreads),
        mean(&depth),
        mean(&top),
        m.messages() as f64 / 4_200.0,
        m.exchange().stats().volume as f64 / 4_200.0
    );

    let t0 = std::time::Instant::now();
    let seed_list: Vec<u64> = (0..seeds).map(|s| 100 + s).collect();
    let paths = mid_paths(&cfg, &seed_list, 600.0, horizon, 1.0);
    let el = t0.elapsed().as_secs_f64();
    let c = Criteria::default();
    for k in [1usize, 10, 30, 60, 300] {
        let r = pooled_returns(&paths, k);
        let rep = analyse(&r, k as f64, 20, 10);
        let vd = verdict(&rep, &c);
        let fmt = |x: &[f64], idx: &[usize]| idx.iter().map(|i| format!("{:+.3}", x[i - 1])).collect::<Vec<_>>().join(" ");
        println!(
            "dt={k:>3}s n={:>7} zero={:.2} sd={:.2e} ekurt={:>7.2} hill={:>5.2} | acf(r) {} | acf(|r|) {} | LB r {:>8.1} |r| {:>9.1} | {}{}{}",
            rep.n,
            rep.zero_fraction,
            rep.moments.std,
            rep.moments.excess_kurtosis,
            rep.hill_alpha,
            fmt(&rep.acf_returns, &[1, 2, 5, 10]),
            fmt(&rep.acf_abs, &[1, 2, 5, 10, 20]),
            rep.ljung_box_returns.statistic,
            rep.ljung_box_abs.statistic,
            if vd.fat_tails { "F" } else { "-" },
            if vd.no_linear_autocorrelation { "N" } else { "-" },
            if vd.volatility_clustering { "V" } else { "-" },
        );
    }
    println!("wall {el:.1}s for {} sim-seconds", seeds as f64 * (horizon + 600.0));
}
