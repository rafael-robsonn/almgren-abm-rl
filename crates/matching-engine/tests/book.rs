use std::collections::{BTreeMap, HashMap, VecDeque};

use matching_engine::{
    Exchange, ExchangeConfig, Fill, HistoryConfig, L3Event, LatencyModel, OrderBook, OrderId,
    Price, Qty, RejectReason, Report, Request, Side,
};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn book() -> OrderBook {
    OrderBook::new(4096)
}

#[test]
fn price_time_priority_and_level_walk() {
    let mut b = book();
    let mut fills = Vec::new();
    let (a1, _) = b.submit_limit(1, Side::Sell, 101, 5, 0, &mut fills);
    let (a2, _) = b.submit_limit(2, Side::Sell, 101, 7, 1, &mut fills);
    let (a3, _) = b.submit_limit(3, Side::Sell, 103, 4, 2, &mut fills);
    assert!(fills.is_empty());
    assert_eq!(b.best_ask(), Some(101));
    assert_eq!(b.best_qty(Side::Sell), 12);

    let (_, filled) = b.submit_market(9, Side::Buy, 14, 3, &mut fills);
    assert_eq!(filled, Ok(14));
    let got: Vec<(OrderId, Price, Qty, bool)> = fills
        .iter()
        .map(|f| (f.maker_order, f.price, f.qty, f.maker_done))
        .collect();
    assert_eq!(
        got,
        vec![(a1, 101, 5, true), (a2, 101, 7, true), (a3, 103, 2, false)]
    );
    assert_eq!(b.best_ask(), Some(103));
    assert_eq!(b.order(a3).map(|o| o.qty), Some(2));
    assert!(b.order(a1).is_none());
    b.check_invariants().unwrap();
}

#[test]
fn marketable_limit_rests_remainder_at_limit() {
    let mut b = book();
    let mut fills = Vec::new();
    let _ = b.submit_limit(1, Side::Sell, 100, 3, 0, &mut fills);
    let _ = b.submit_limit(1, Side::Sell, 102, 3, 0, &mut fills);
    let (id, out) = b.submit_limit(2, Side::Buy, 101, 10, 1, &mut fills);
    let out = out.unwrap();
    assert_eq!((out.filled, out.resting), (3, 7));
    assert_eq!(b.best_bid(), Some(101));
    assert_eq!(b.best_ask(), Some(102));
    assert_eq!(b.order(id).unwrap().qty, 7);
    b.check_invariants().unwrap();
}

#[test]
fn cancel_and_stale_handles() {
    let mut b = book();
    let mut fills = Vec::new();
    let (x, _) = b.submit_limit(1, Side::Buy, 99, 5, 0, &mut fills);
    let (y, _) = b.submit_limit(1, Side::Buy, 98, 5, 0, &mut fills);
    assert_eq!(b.cancel(2, x), Err(RejectReason::NotOwner));
    let c = b.cancel(1, x).unwrap();
    assert_eq!((c.price, c.qty), (99, 5));
    assert_eq!(b.best_bid(), Some(98));
    assert_eq!(b.cancel(1, x), Err(RejectReason::UnknownOrder));

    let (z, _) = b.submit_limit(1, Side::Buy, 99, 1, 0, &mut fills);
    assert_ne!(z, x, "a recycled slot must carry a new generation");
    assert_eq!(b.cancel(1, x), Err(RejectReason::UnknownOrder));
    let _ = b.submit_market(3, Side::Sell, 6, 1, &mut fills);
    assert!(!b.is_resting(z) && !b.is_resting(y));
    assert_eq!(b.cancel(1, y), Err(RejectReason::UnknownOrder));
    assert_eq!(b.best_bid(), None);
    b.check_invariants().unwrap();
}

#[test]
fn modify_priority_semantics() {
    let mut b = book();
    let mut fills = Vec::new();
    let (a, _) = b.submit_limit(1, Side::Buy, 50, 10, 0, &mut fills);
    let (c, _) = b.submit_limit(2, Side::Buy, 50, 10, 1, &mut fills);

    let m = b.modify(1, a, 50, 4, 2, &mut fills).unwrap();
    assert!(!m.requeued);
    let q: Vec<OrderId> = b.queue(Side::Buy, 50).iter().map(|o| o.id).collect();
    assert_eq!(q, vec![a, c]);

    let m = b.modify(1, a, 50, 8, 3, &mut fills).unwrap();
    assert!(m.requeued);
    let q: Vec<OrderId> = b.queue(Side::Buy, 50).iter().map(|o| o.id).collect();
    assert_eq!(q, vec![c, a]);

    let _ = b.submit_limit(3, Side::Sell, 52, 5, 4, &mut fills);
    fills.clear();
    let m = b.modify(2, c, 53, 10, 5, &mut fills).unwrap();
    assert_eq!((m.filled, m.resting), (5, 5));
    assert_eq!(fills.len(), 1);
    assert_eq!(fills[0].taker_order, c);
    assert_eq!(b.best_bid(), Some(53));
    assert_eq!(b.best_ask(), None);
    b.check_invariants().unwrap();
}

#[derive(Default)]
struct RefBook {
    bids: BTreeMap<Price, VecDeque<(OrderId, u32, Qty)>>,
    asks: BTreeMap<Price, VecDeque<(OrderId, u32, Qty)>>,
    live: HashMap<OrderId, (Side, Price)>,
}

impl RefBook {
    fn side(&mut self, s: Side) -> &mut BTreeMap<Price, VecDeque<(OrderId, u32, Qty)>> {
        match s {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        }
    }

    fn take(&mut self, side: Side, limit: Option<Price>, mut qty: Qty) -> (Vec<(OrderId, Price, Qty)>, Qty) {
        let RefBook { bids, asks, live } = self;
        let opp = match side {
            Side::Buy => asks,
            Side::Sell => bids,
        };
        let mut out = Vec::new();
        while qty > 0 {
            let best = match side {
                Side::Buy => opp.keys().next().copied(),
                Side::Sell => opp.keys().next_back().copied(),
            };
            let Some(p) = best else { break };
            if let Some(l) = limit
                && ((side == Side::Buy && p > l) || (side == Side::Sell && p < l))
            {
                break;
            }
            let q = opp.get_mut(&p).unwrap();
            while qty > 0 {
                let Some(front) = q.front_mut() else { break };
                let t = front.2.min(qty);
                front.2 -= t;
                qty -= t;
                out.push((front.0, p, t));
                if front.2 == 0 {
                    let id = front.0;
                    q.pop_front();
                    live.remove(&id);
                }
            }
            if q.is_empty() {
                opp.remove(&p);
            }
        }
        (out, qty)
    }

    fn rest(&mut self, id: OrderId, owner: u32, side: Side, price: Price, qty: Qty) {
        self.side(side).entry(price).or_default().push_back((id, owner, qty));
        self.live.insert(id, (side, price));
    }

    fn remove(&mut self, id: OrderId) -> Option<(Side, Price, u32, Qty)> {
        let (side, price) = self.live.remove(&id)?;
        let lvl = self.side(side).get_mut(&price).unwrap();
        let pos = lvl.iter().position(|o| o.0 == id).unwrap();
        let (_, owner, qty) = lvl.remove(pos).unwrap();
        if lvl.is_empty() {
            self.side(side).remove(&price);
        }
        Some((side, price, owner, qty))
    }

    fn l2(&self, side: Side) -> Vec<(Price, Qty)> {
        let m = match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        };
        let it = m.iter().map(|(p, q)| (*p, q.iter().map(|o| o.2).sum()));
        match side {
            Side::Buy => it.rev().collect(),
            Side::Sell => it.collect(),
        }
    }
}

fn fills_key(f: &[Fill]) -> Vec<(OrderId, Price, Qty)> {
    f.iter().map(|f| (f.maker_order, f.price, f.qty)).collect()
}

#[test]
fn randomized_against_reference_model() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let mut b = OrderBook::new(2048);
    let mut r = RefBook::default();
    let mut known: Vec<(OrderId, u32)> = Vec::new();
    let mut fills = Vec::new();
    for step in 0..60_000u64 {
        fills.clear();
        let u = rng.next_u64() % 100;
        let owner = (rng.next_u64() % 5) as u32;
        let side = if rng.next_u64().is_multiple_of(2) { Side::Buy } else { Side::Sell };
        let price = 1000 + (rng.next_u64() % 41) as Price - 20;
        let qty = 1 + rng.next_u64() % 20;
        if u < 55 {
            let (id, out) = b.submit_limit(owner, side, price, qty, step, &mut fills);
            let out = out.unwrap();
            let (rf, rem) = r.take(side, Some(price), qty);
            assert_eq!(fills_key(&fills), rf, "limit fills diverged at {step}");
            assert_eq!(out.resting, rem);
            if rem > 0 {
                r.rest(id, owner, side, price, rem);
                known.push((id, owner));
            }
        } else if u < 70 {
            let q = 1 + rng.next_u64() % 60;
            let (_, filled) = b.submit_market(owner, side, q, step, &mut fills);
            let (rf, rem) = r.take(side, None, q);
            assert_eq!(fills_key(&fills), rf, "market fills diverged at {step}");
            assert_eq!(filled.unwrap(), q - rem);
        } else if u < 85 && !known.is_empty() {
            let k = (rng.next_u64() % known.len() as u64) as usize;
            let (id, own) = known.swap_remove(k);
            let got = b.cancel(own, id);
            match r.remove(id) {
                Some((_, _, _, q)) => assert_eq!(got.unwrap().qty, q),
                None => assert_eq!(got, Err(RejectReason::UnknownOrder)),
            }
        } else if !known.is_empty() {
            let k = (rng.next_u64() % known.len() as u64) as usize;
            let (id, own) = known[k];
            let got = b.modify(own, id, price, qty, step, &mut fills);
            match r.live.get(&id).copied() {
                None => {
                    assert_eq!(got, Err(RejectReason::UnknownOrder));
                    known.swap_remove(k);
                }
                Some((s, p)) => {
                    let lvl = &r.side(s)[&p];
                    let cur = lvl.iter().find(|o| o.0 == id).unwrap().2;
                    let m = got.unwrap();
                    if p == price && qty <= cur {
                        assert!(!m.requeued);
                        let lvl = r.side(s).get_mut(&p).unwrap();
                        lvl.iter_mut().find(|o| o.0 == id).unwrap().2 = qty;
                    } else {
                        assert!(m.requeued);
                        r.remove(id);
                        let (rf, rem) = r.take(s, Some(price), qty);
                        assert_eq!(fills_key(&fills), rf, "modify fills diverged at {step}");
                        if rem > 0 {
                            r.rest(id, own, s, price, rem);
                        } else {
                            known.swap_remove(k);
                        }
                    }
                }
            }
        }
        if step.is_multiple_of(97) {
            b.check_invariants().unwrap();
        }
        for side in [Side::Buy, Side::Sell] {
            let mut lv = Vec::new();
            b.depth(side, usize::MAX, &mut lv);
            assert_eq!(lv, r.l2(side), "{side:?} L2 diverged at step {step}");
        }
    }
    b.check_invariants().unwrap();
}

fn random_flow(ex: &mut Exchange, rng: &mut Xoshiro256PlusPlus, n: usize) -> Vec<(u32, Report)> {
    let mut reports = Vec::new();
    let mut live: Vec<(u32, OrderId)> = Vec::new();
    let mut t = 0;
    for _ in 0..n {
        t += 1 + rng.next_u64() % 1_000_000;
        let agent = (rng.next_u64() % 6) as u32;
        let side = if rng.next_u64().is_multiple_of(2) { Side::Buy } else { Side::Sell };
        let mid = 5000;
        let u = rng.next_u64() % 10;
        let req = match u {
            0..=5 => {
                let off = (rng.next_u64() % 10) as Price;
                let price = match side {
                    Side::Buy => mid - off,
                    Side::Sell => mid + 1 + off,
                };
                Request::Limit { side, price, qty: 1 + rng.next_u64() % 9 }
            }
            6 => Request::Market { side, qty: 1 + rng.next_u64() % 15 },
            7 | 8 if !live.is_empty() => {
                let (a, id) = live[(rng.next_u64() % live.len() as u64) as usize];
                let req = Request::Cancel { target: id };
                ex.submit(a, req, t);
                continue;
            }
            _ if !live.is_empty() => {
                let (a, id) = live[(rng.next_u64() % live.len() as u64) as usize];
                let price = 4995 + (rng.next_u64() % 11) as Price;
                let req = Request::Modify { target: id, price, qty: 1 + rng.next_u64() % 9 };
                ex.submit(a, req, t);
                continue;
            }
            _ => Request::Market { side, qty: 1 },
        };
        let id = ex.submit(agent, req, t);
        if matches!(req, Request::Limit { .. }) {
            live.push((agent, id));
        }
        ex.process_until(t, &mut reports);
    }
    ex.drain(&mut reports);
    reports
}

#[test]
fn l3_replay_reconstructs_book() {
    let cfg = ExchangeConfig {
        n_ticks: 8192,
        latency: LatencyModel::new(50_000, 200_000),
        history: HistoryConfig {
            l3: true,
            l2_interval: Some(10_000_000),
            l2_depth: 5,
        },
    };
    let mut ex = Exchange::new(cfg, 11);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
    random_flow(&mut ex, &mut rng, 20_000);

    let mut orders: HashMap<OrderId, (Side, Price, Qty)> = HashMap::new();
    for e in &ex.history().l3 {
        match *e {
            L3Event::Add { order, side, price, qty, .. } => {
                assert!(orders.insert(order, (side, price, qty)).is_none());
            }
            L3Event::Execute { maker, qty, price, .. } => {
                let o = orders.get_mut(&maker).expect("execution against unknown order");
                assert_eq!(o.1, price);
                o.2 -= qty;
                if o.2 == 0 {
                    orders.remove(&maker);
                }
            }
            L3Event::Cancel { order, qty, .. } => {
                let o = orders.get_mut(&order).expect("cancel of unknown order");
                o.2 -= qty;
                if o.2 == 0 {
                    orders.remove(&order);
                }
            }
        }
    }
    for side in [Side::Buy, Side::Sell] {
        let mut agg: BTreeMap<Price, Qty> = BTreeMap::new();
        for (s, p, q) in orders.values() {
            if *s == side {
                *agg.entry(*p).or_default() += q;
            }
        }
        let rebuilt: Vec<(Price, Qty)> = match side {
            Side::Buy => agg.into_iter().rev().collect(),
            Side::Sell => agg.into_iter().collect(),
        };
        let mut live = Vec::new();
        ex.book().depth(side, usize::MAX, &mut live);
        assert_eq!(rebuilt, live);
    }
    assert!(!ex.history().l2.is_empty());
    let ts: Vec<u64> = ex.history().l2.iter().map(|s| s.ts).collect();
    assert!(ts.windows(2).all(|w| w[1] == w[0] + 10_000_000));
    ex.book().check_invariants().unwrap();
}

#[test]
fn latency_orders_messages_and_preserves_agent_fifo() {
    let mut ex = Exchange::new(ExchangeConfig::default(), 5);
    ex.set_latency(1, LatencyModel::new(1_000, 0));
    ex.set_latency(2, LatencyModel::new(10, 0));
    let ask = ex.submit(1, Request::Limit { side: Side::Sell, price: 100, qty: 5 }, 0);
    let mkt = ex.submit(2, Request::Market { side: Side::Buy, qty: 5 }, 0);
    assert_eq!(ex.submit(2, Request::Cancel { target: mkt }, 0), mkt);
    let mut reports = Vec::new();
    ex.process_until(999, &mut reports);
    assert_eq!(
        reports,
        vec![
            (2, Report::Expired { order: mkt, qty: 5 }),
            (2, Report::Rejected { order: mkt, reason: RejectReason::UnknownOrder }),
        ]
    );
    assert!(ex.book().best_ask().is_none());
    reports.clear();
    ex.process_until(1_000, &mut reports);
    assert_eq!(
        reports,
        vec![(1, Report::Rested { order: ask, side: Side::Sell, price: 100, qty: 5 })]
    );

    ex.set_latency(3, LatencyModel::new(100, 1_000_000));
    let ids: Vec<OrderId> = (0..200)
        .map(|i| ex.submit(3, Request::Limit { side: Side::Sell, price: 200 + i, qty: 1 }, 2_000))
        .collect();
    reports.clear();
    ex.drain(&mut reports);
    let rested: Vec<OrderId> = reports
        .iter()
        .filter_map(|(_, r)| match r {
            Report::Rested { order, .. } => Some(*order),
            _ => None,
        })
        .collect();
    assert_eq!(rested, ids);
}

#[test]
fn exchange_is_deterministic_for_a_seed() {
    let cfg = || ExchangeConfig {
        n_ticks: 8192,
        latency: LatencyModel::new(10_000, 100_000),
        history: HistoryConfig { l3: true, l2_interval: None, l2_depth: 0 },
    };
    let run = |seed| {
        let mut ex = Exchange::new(cfg(), seed);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        let reps = random_flow(&mut ex, &mut rng, 5_000);
        (reps, ex.history().l3.clone())
    };
    assert_eq!(run(1), run(1));
    assert_ne!(run(1).1, run(2).1);
}
