use std::f64::consts::LN_2;

use matching_engine::{OrderId, Price, Qty, Report, Request, Side, Timestamp, secs, to_secs};
use serde::{Deserialize, Serialize};

use super::{Account, Ctx, fill_into};
use crate::rng::{Rng, derive_seed};

/// Avellaneda-Stoikov dealer re-quoting at Poisson epochs of intensity `rate`. With mid `m`,
/// normalised inventory `q = inventory / quote_size` and variance estimate `s2` (ticks^2/s):
/// reservation `r = m - q * gamma * s2 * horizon`, half-spread
/// `delta = gamma * s2 * horizon / 2 + ln(1 + gamma / k) / gamma` capped at `max_half_spread`,
/// quotes `floor(r - delta)` and `ceil(r + delta)`, one side withdrawn at
/// `|inventory| >= max_inventory`. `s2` is the time-decayed mean of `(dm)^2 / dt`, clamped to
/// `[variance_floor, variance_cap]`, with a dealer-specific half-life drawn log-uniformly from
/// `[vol_halflife_min, vol_halflife_max]`; the inventory skew `|r - m|` is capped at `max_skew`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarketMakerParams {
    pub rate: f64,
    pub gamma: f64,
    pub k: f64,
    pub horizon: f64,
    pub quote_size: Qty,
    pub max_inventory: i64,
    pub vol_halflife_min: f64,
    pub vol_halflife_max: f64,
    pub initial_variance: f64,
    pub variance_floor: f64,
    pub variance_cap: f64,
    pub max_half_spread: f64,
    pub max_skew: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Quote {
    id: Option<OrderId>,
    price: Price,
    qty: Qty,
}

#[derive(Clone, Debug)]
pub struct MarketMaker {
    params: MarketMakerParams,
    clock: Rng,
    pub(super) account: Account,
    halflife: f64,
    variance: f64,
    last_mid: f64,
    last_t: Option<Timestamp>,
    bid: Quote,
    ask: Quote,
}

impl MarketMaker {
    pub fn new(params: MarketMakerParams, seed: u64) -> Self {
        let mut init = Rng::new(derive_seed(seed, 2));
        let halflife = init.log_uniform(params.vol_halflife_min, params.vol_halflife_max);
        Self {
            variance: params.initial_variance,
            params,
            clock: Rng::new(derive_seed(seed, 0)),
            account: Account::default(),
            halflife,
            last_mid: 0.0,
            last_t: None,
            bid: Quote::default(),
            ask: Quote::default(),
        }
    }

    pub fn first_wakeup(&mut self) -> Timestamp {
        secs(self.clock.exp(self.params.rate)).max(1)
    }

    pub fn variance(&self) -> f64 {
        self.variance
    }

    pub fn wake(&mut self, ctx: &mut Ctx<'_>) -> Timestamp {
        let mid = ctx.reference_price();
        if let Some(t0) = self.last_t {
            let dt = to_secs(ctx.now - t0);
            if dt > 0.0 {
                let inst = (mid - self.last_mid).powi(2) / dt;
                let a = 1.0 - (-dt * LN_2 / self.halflife).exp();
                self.variance = (self.variance + a * (inst - self.variance))
                    .clamp(self.params.variance_floor, self.params.variance_cap);
            }
        }
        self.last_mid = mid;
        self.last_t = Some(ctx.now);

        let p = &self.params;
        let q = self.account.inventory as f64 / p.quote_size as f64;
        let s2t = self.variance * p.horizon;
        let reservation = mid - (q * p.gamma * s2t).clamp(-p.max_skew, p.max_skew);
        let half =
            (0.5 * p.gamma * s2t + (1.0 + p.gamma / p.k).ln() / p.gamma).min(p.max_half_spread);
        let mut bid = (reservation - half).floor() as Price;
        let mut ask = (reservation + half).ceil() as Price;
        if ask <= bid {
            ask = bid + 1;
        }
        if let Some(a) = ctx.book().best_ask() {
            bid = bid.min(a - 1);
        }
        if let Some(b) = ctx.book().best_bid() {
            ask = ask.max(b + 1);
        }
        let bid = ctx.clamp_price(bid);
        let ask = ctx.clamp_price(ask);
        let want_bid = self.account.inventory < p.max_inventory;
        let want_ask = self.account.inventory > -p.max_inventory;
        self.requote(ctx, Side::Buy, want_bid, bid);
        self.requote(ctx, Side::Sell, want_ask, ask);
        ctx.after(self.clock.exp(self.params.rate))
    }

    fn requote(&mut self, ctx: &mut Ctx<'_>, side: Side, want: bool, price: Price) {
        let size = self.params.quote_size;
        let quote = match side {
            Side::Buy => &mut self.bid,
            Side::Sell => &mut self.ask,
        };
        match (want, quote.id) {
            (true, None) => {
                let id = ctx.submit(Request::Limit {
                    side,
                    price,
                    qty: size,
                });
                *quote = Quote {
                    id: Some(id),
                    price,
                    qty: size,
                };
            }
            (true, Some(id)) => {
                if quote.price != price || quote.qty * 2 < size {
                    ctx.submit(Request::Modify {
                        target: id,
                        price,
                        qty: size,
                    });
                    quote.price = price;
                    quote.qty = size;
                }
            }
            (false, Some(id)) => {
                ctx.submit(Request::Cancel { target: id });
                quote.id = None;
            }
            (false, None) => {}
        }
    }

    fn quote_mut(&mut self, order: OrderId) -> Option<&mut Quote> {
        if self.bid.id == Some(order) {
            Some(&mut self.bid)
        } else if self.ask.id == Some(order) {
            Some(&mut self.ask)
        } else {
            None
        }
    }

    pub fn on_report(&mut self, report: &Report) {
        fill_into(&mut self.account, report);
        let Some(q) = self.quote_mut(report.order()) else {
            return;
        };
        match *report {
            Report::Fill { qty, done, .. } => {
                q.qty = q.qty.saturating_sub(qty);
                if done {
                    q.id = None;
                }
            }
            Report::Rested { price, qty, .. } | Report::Modified { price, qty, .. } => {
                q.price = price;
                q.qty = qty;
                if qty == 0 {
                    q.id = None;
                }
            }
            Report::Cancelled { .. } | Report::Rejected { .. } | Report::Expired { .. } => {
                q.id = None;
            }
        }
    }
}
