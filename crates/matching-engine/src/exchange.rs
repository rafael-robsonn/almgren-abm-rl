use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::book::OrderBook;
use crate::history::{History, HistoryConfig, L3Event};
use crate::types::{AgentId, Fill, Liquidity, OrderId, Price, Qty, Report, Request, Timestamp};

/// One-way order-entry latency `base + Exp(jitter_mean)` in nanoseconds. Per-agent FIFO is
/// enforced on top of the draw, so messages from one agent never overtake each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencyModel {
    pub base: Timestamp,
    pub jitter_mean: Timestamp,
}

impl LatencyModel {
    pub const ZERO: Self = Self {
        base: 0,
        jitter_mean: 0,
    };

    pub fn new(base: Timestamp, jitter_mean: Timestamp) -> Self {
        Self { base, jitter_mean }
    }

    #[inline]
    fn sample(&self, rng: &mut Xoshiro256PlusPlus) -> Timestamp {
        if self.jitter_mean == 0 {
            return self.base;
        }
        let u = ((rng.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64);
        self.base + (-u.ln() * self.jitter_mean as f64) as Timestamp
    }
}

#[derive(Clone, Copy, Debug)]
struct Inbound {
    arrival: Timestamp,
    seq: u64,
    agent: AgentId,
    order: OrderId,
    request: Request,
}

impl PartialEq for Inbound {
    fn eq(&self, other: &Self) -> bool {
        self.arrival == other.arrival && self.seq == other.seq
    }
}

impl Eq for Inbound {}

impl PartialOrd for Inbound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Inbound {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .arrival
            .cmp(&self.arrival)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

#[derive(Clone, Debug)]
pub struct ExchangeConfig {
    pub n_ticks: usize,
    pub latency: LatencyModel,
    pub history: HistoryConfig,
}

impl Default for ExchangeConfig {
    fn default() -> Self {
        Self {
            n_ticks: 1 << 16,
            latency: LatencyModel::ZERO,
            history: HistoryConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExchangeStats {
    pub messages: u64,
    pub limits: u64,
    pub markets: u64,
    pub cancels: u64,
    pub modifies: u64,
    pub rejects: u64,
    pub trades: u64,
    pub volume: u64,
}

/// Sequenced exchange front-end: order-entry messages are timestamped on submission, delayed by
/// the sender's latency model and applied to the book strictly in `(arrival, sequence)` order.
#[derive(Clone, Debug)]
pub struct Exchange {
    book: OrderBook,
    inbound: BinaryHeap<Inbound>,
    seq: u64,
    default_latency: LatencyModel,
    latency: Vec<Option<LatencyModel>>,
    last_arrival: Vec<Timestamp>,
    rng: Xoshiro256PlusPlus,
    now: Timestamp,
    fills: Vec<Fill>,
    history: History,
    stats: ExchangeStats,
    last_trade: Option<Price>,
}

impl Exchange {
    pub fn new(config: ExchangeConfig, seed: u64) -> Self {
        Self {
            book: OrderBook::new(config.n_ticks),
            inbound: BinaryHeap::with_capacity(256),
            seq: 0,
            default_latency: config.latency,
            latency: Vec::new(),
            last_arrival: Vec::new(),
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            now: 0,
            fills: Vec::with_capacity(64),
            history: History::new(config.history),
            stats: ExchangeStats::default(),
            last_trade: None,
        }
    }

    pub fn set_latency(&mut self, agent: AgentId, model: LatencyModel) {
        let i = agent as usize;
        if self.latency.len() <= i {
            self.latency.resize(i + 1, None);
        }
        self.latency[i] = Some(model);
    }

    pub fn submit(&mut self, agent: AgentId, request: Request, sent_at: Timestamp) -> OrderId {
        let order = match request {
            Request::Limit { .. } | Request::Market { .. } => self.book.allocate(agent),
            Request::Cancel { target } | Request::Modify { target, .. } => target,
        };
        let model = self
            .latency
            .get(agent as usize)
            .copied()
            .flatten()
            .unwrap_or(self.default_latency);
        let i = agent as usize;
        if self.last_arrival.len() <= i {
            self.last_arrival.resize(i + 1, 0);
        }
        let arrival = (sent_at.max(self.now) + model.sample(&mut self.rng)).max(self.last_arrival[i]);
        self.last_arrival[i] = arrival;
        self.seq += 1;
        self.inbound.push(Inbound {
            arrival,
            seq: self.seq,
            agent,
            order,
            request,
        });
        order
    }

    #[inline]
    pub fn next_arrival(&self) -> Option<Timestamp> {
        self.inbound.peek().map(|m| m.arrival)
    }

    pub fn pending(&self) -> usize {
        self.inbound.len()
    }

    pub fn advance_to(&mut self, t: Timestamp) {
        if t > self.now {
            self.now = t;
        }
        self.history.sample_l2(&self.book, self.now);
    }

    pub fn process_until(&mut self, t: Timestamp, reports: &mut Vec<(AgentId, Report)>) {
        while self.next_arrival().is_some_and(|a| a <= t) {
            self.process_next(reports);
        }
        self.advance_to(t);
    }

    pub fn drain(&mut self, reports: &mut Vec<(AgentId, Report)>) {
        while self.process_next(reports) {}
    }

    pub fn process_next(&mut self, reports: &mut Vec<(AgentId, Report)>) -> bool {
        let Some(msg) = self.inbound.pop() else {
            return false;
        };
        self.now = self.now.max(msg.arrival);
        let now = self.now;
        self.history.sample_l2(&self.book, now);
        self.stats.messages += 1;
        self.fills.clear();
        let agent = msg.agent;
        match msg.request {
            Request::Limit { side, price, qty } => {
                self.stats.limits += 1;
                match self.book.limit(msg.order, side, price, qty, now, &mut self.fills) {
                    Ok(out) => {
                        self.emit_fills(reports, out.resting == 0);
                        if out.resting > 0 {
                            self.history.record_l3(L3Event::Add {
                                ts: now,
                                order: msg.order,
                                side,
                                price,
                                qty: out.resting,
                            });
                            reports.push((
                                agent,
                                Report::Rested {
                                    order: msg.order,
                                    side,
                                    price,
                                    qty: out.resting,
                                },
                            ));
                        }
                    }
                    Err(reason) => self.reject(reports, agent, msg.order, reason),
                }
            }
            Request::Market { side, qty } => {
                self.stats.markets += 1;
                match self.book.market(msg.order, side, qty, now, &mut self.fills) {
                    Ok(filled) => {
                        self.emit_fills(reports, filled == qty);
                        if filled < qty {
                            reports.push((
                                agent,
                                Report::Expired {
                                    order: msg.order,
                                    qty: qty - filled,
                                },
                            ));
                        }
                    }
                    Err(reason) => self.reject(reports, agent, msg.order, reason),
                }
            }
            Request::Cancel { target } => {
                self.stats.cancels += 1;
                match self.book.cancel(agent, target) {
                    Ok(c) => {
                        self.history.record_l3(L3Event::Cancel {
                            ts: now,
                            order: target,
                            qty: c.qty,
                        });
                        reports.push((
                            agent,
                            Report::Cancelled {
                                order: target,
                                qty: c.qty,
                            },
                        ));
                    }
                    Err(reason) => self.reject(reports, agent, target, reason),
                }
            }
            Request::Modify { target, price, qty } => {
                self.stats.modifies += 1;
                match self.book.modify(agent, target, price, qty, now, &mut self.fills) {
                    Ok(m) => {
                        if m.requeued {
                            self.history.record_l3(L3Event::Cancel {
                                ts: now,
                                order: target,
                                qty: m.old_qty,
                            });
                            self.emit_fills(reports, m.resting == 0);
                            if m.resting > 0 {
                                self.history.record_l3(L3Event::Add {
                                    ts: now,
                                    order: target,
                                    side: m.side,
                                    price,
                                    qty: m.resting,
                                });
                            }
                        } else if m.old_qty > m.resting {
                            self.history.record_l3(L3Event::Cancel {
                                ts: now,
                                order: target,
                                qty: m.old_qty - m.resting,
                            });
                        }
                        let report = if m.resting == 0 && m.filled == 0 {
                            Report::Cancelled {
                                order: target,
                                qty: m.old_qty,
                            }
                        } else {
                            Report::Modified {
                                order: target,
                                price,
                                qty: m.resting,
                            }
                        };
                        reports.push((agent, report));
                    }
                    Err(reason) => self.reject(reports, agent, target, reason),
                }
            }
        }
        true
    }

    fn reject(
        &mut self,
        reports: &mut Vec<(AgentId, Report)>,
        agent: AgentId,
        order: OrderId,
        reason: crate::types::RejectReason,
    ) {
        self.stats.rejects += 1;
        reports.push((agent, Report::Rejected { order, reason }));
    }

    fn emit_fills(&mut self, reports: &mut Vec<(AgentId, Report)>, taker_complete: bool) {
        let n = self.fills.len();
        for (i, f) in self.fills.iter().enumerate() {
            self.stats.trades += 1;
            self.stats.volume += f.qty;
            self.history.record_l3(L3Event::Execute {
                ts: f.ts,
                maker: f.maker_order,
                taker: f.taker_order,
                taker_side: f.taker_side,
                price: f.price,
                qty: f.qty,
            });
            reports.push((
                f.maker,
                Report::Fill {
                    order: f.maker_order,
                    side: f.taker_side.opposite(),
                    price: f.price,
                    qty: f.qty,
                    liquidity: Liquidity::Maker,
                    done: f.maker_done,
                },
            ));
            reports.push((
                f.taker,
                Report::Fill {
                    order: f.taker_order,
                    side: f.taker_side,
                    price: f.price,
                    qty: f.qty,
                    liquidity: Liquidity::Taker,
                    done: taker_complete && i + 1 == n,
                },
            ));
        }
        if let Some(f) = self.fills.last() {
            self.last_trade = Some(f.price);
        }
    }

    #[inline]
    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    #[inline]
    pub fn now(&self) -> Timestamp {
        self.now
    }

    pub fn stats(&self) -> &ExchangeStats {
        &self.stats
    }

    pub fn history(&self) -> &History {
        &self.history
    }

    pub fn history_mut(&mut self) -> &mut History {
        &mut self.history
    }

    pub fn last_trade(&self) -> Option<Price> {
        self.last_trade
    }

    pub fn traded_volume(&self) -> Qty {
        self.stats.volume
    }
}
