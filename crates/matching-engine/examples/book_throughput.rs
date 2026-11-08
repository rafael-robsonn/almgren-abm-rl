use std::time::Instant;

use matching_engine::{Exchange, ExchangeConfig, LatencyModel, OrderBook, OrderId, Price, Request, Side};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5_000_000);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    let mut book = OrderBook::new(1 << 16);
    let mut fills = Vec::with_capacity(64);
    let mut live: Vec<(u32, OrderId)> = Vec::with_capacity(1 << 16);
    let mid: Price = 30_000;
    let start = Instant::now();
    let mut trades = 0usize;
    for i in 0..n {
        fills.clear();
        let r = rng.next_u64();
        let owner = (r & 63) as u32;
        let side = if r & 64 == 0 { Side::Buy } else { Side::Sell };
        let u = (r >> 8) % 100;
        let off = ((r >> 16) % 24) as Price;
        let qty = 1 + (r >> 32) % 20;
        if u < 50 || live.is_empty() {
            let price = match side {
                Side::Buy => mid - 1 - off,
                Side::Sell => mid + 1 + off,
            };
            let (id, out) = book.submit_limit(owner, side, price, qty, i as u64, &mut fills);
            if out.is_ok_and(|o| o.resting > 0) {
                live.push((owner, id));
            }
        } else if u < 65 {
            let _ = book.submit_market(owner, side, qty, i as u64, &mut fills);
        } else if u < 90 {
            let k = (r >> 40) as usize % live.len();
            let (o, id) = live.swap_remove(k);
            let _ = book.cancel(o, id);
        } else {
            let k = (r >> 40) as usize % live.len();
            let (o, id) = live[k];
            let price = match side {
                Side::Buy => mid - 1 - off,
                Side::Sell => mid + 1 + off,
            };
            if book.order(id).is_some_and(|x| x.side == side) {
                let _ = book.modify(o, id, price, qty, i as u64, &mut fills);
            }
        }
        trades += fills.len();
    }
    let dt = start.elapsed().as_secs_f64();
    println!(
        "order book: {n} messages in {dt:.3} s = {:.2} M msg/s ({trades} fills, {} resting)",
        n as f64 / dt / 1e6,
        book.side_orders(Side::Buy) + book.side_orders(Side::Sell)
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
    let mut ex = Exchange::new(
        ExchangeConfig {
            latency: LatencyModel::new(20_000, 30_000),
            ..ExchangeConfig::default()
        },
        3,
    );
    let mut reports = Vec::with_capacity(256);
    let mut live: Vec<(u32, OrderId)> = Vec::with_capacity(1 << 16);
    let start = Instant::now();
    let mut t = 0u64;
    for _ in 0..n {
        reports.clear();
        t += 10_000;
        let r = rng.next_u64();
        let owner = (r & 63) as u32;
        let side = if r & 64 == 0 { Side::Buy } else { Side::Sell };
        let u = (r >> 8) % 100;
        let off = ((r >> 16) % 24) as Price;
        let qty = 1 + (r >> 32) % 20;
        let req = if u < 55 || live.is_empty() {
            let price = match side {
                Side::Buy => mid - 1 - off,
                Side::Sell => mid + 1 + off,
            };
            Request::Limit { side, price, qty }
        } else if u < 70 {
            Request::Market { side, qty }
        } else {
            let k = (r >> 40) as usize % live.len();
            let (o, id) = live.swap_remove(k);
            ex.submit(o, Request::Cancel { target: id }, t);
            ex.process_until(t, &mut reports);
            continue;
        };
        let id = ex.submit(owner, req, t);
        if matches!(req, Request::Limit { .. }) {
            live.push((owner, id));
        }
        ex.process_until(t, &mut reports);
    }
    ex.drain(&mut reports);
    let dt = start.elapsed().as_secs_f64();
    let m = ex.stats().messages;
    println!(
        "exchange (latency queue + reports): {m} messages in {dt:.3} s = {:.2} M msg/s",
        m as f64 / dt / 1e6
    );
}
