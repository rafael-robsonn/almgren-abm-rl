use std::time::Instant;

use market_agents::{Market, MarketConfig};
use matching_engine::secs;
use rayon::prelude::*;

fn run(seed: u64, horizon: f64) -> (u64, u64) {
    let mut m = Market::new(MarketConfig::default(), seed);
    m.run_until(secs(horizon));
    (m.messages(), m.wakeups())
}

fn main() {
    let horizon: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200_000.0);
    let t = Instant::now();
    let (msgs, wakes) = run(1, horizon);
    let dt = t.elapsed().as_secs_f64();
    println!(
        "single thread: {horizon:.0} sim-s, {msgs} exchange messages + {wakes} agent epochs in {dt:.3} s -> {:.2} M msg/s, {:.2} M events/s",
        msgs as f64 / dt / 1e6,
        (msgs + wakes) as f64 / dt / 1e6
    );
    for threads in [16usize, 32] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let t = Instant::now();
        let totals: Vec<(u64, u64)> = pool.install(|| {
            (0..threads as u64).into_par_iter().map(|s| run(100 + s, horizon)).collect()
        });
        let dt = t.elapsed().as_secs_f64();
        let msgs: u64 = totals.iter().map(|x| x.0).sum();
        let ev: u64 = totals.iter().map(|x| x.0 + x.1).sum();
        println!(
            "{threads} threads ({threads} independent markets): {:.2} M msg/s aggregate, {:.2} M msg/s per thread, {:.2} M events/s per thread",
            msgs as f64 / dt / 1e6,
            msgs as f64 / dt / 1e6 / threads as f64,
            ev as f64 / dt / 1e6 / threads as f64
        );
    }
}
