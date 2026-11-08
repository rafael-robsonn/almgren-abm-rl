use std::cmp::Reverse;
use std::collections::BinaryHeap;

use matching_engine::{OrderId, Price, Qty, Report, Request, Side, Timestamp, secs};
use serde::{Deserialize, Serialize};

use super::{Account, Ctx, fill_into};
use crate::rng::{Rng, derive_seed};

/// Zero-intelligence trader. Decision epochs form a Poisson process of intensity `rate`; each
/// epoch submits a market order with probability `market_prob`, otherwise a limit order
/// `1 + D` ticks behind the opposite touch with `P(D >= d) = (1 + d / offset_scale)^(-offset_tail)`.
/// Sizes are log-normal. Each resting order carries an exponential lifetime of mean
/// `mean_lifetime` seconds, enforced at the owner's next decision epoch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseParams {
    pub rate: f64,
    pub market_prob: f64,
    pub size_log_mean: f64,
    pub size_log_sd: f64,
    pub max_size: Qty,
    pub offset_scale: f64,
    pub offset_tail: f64,
    pub max_offset: Price,
    pub mean_lifetime: f64,
}

#[derive(Clone, Debug)]
pub struct NoiseTrader {
    params: NoiseParams,
    clock: Rng,
    rng: Rng,
    pub(super) account: Account,
    expiries: BinaryHeap<Reverse<(Timestamp, OrderId)>>,
}

impl NoiseTrader {
    pub fn new(params: NoiseParams, seed: u64) -> Self {
        Self {
            params,
            clock: Rng::new(derive_seed(seed, 0)),
            rng: Rng::new(derive_seed(seed, 1)),
            account: Account::default(),
            expiries: BinaryHeap::new(),
        }
    }

    pub fn first_wakeup(&mut self) -> Timestamp {
        secs(self.clock.exp(self.params.rate)).max(1)
    }

    #[inline]
    fn size(&mut self) -> Qty {
        let s = self
            .rng
            .lognormal(self.params.size_log_mean, self.params.size_log_sd)
            .round() as Qty;
        s.clamp(1, self.params.max_size)
    }

    #[inline]
    fn offset(&mut self) -> Price {
        let u = self.rng.uniform();
        let d = self.params.offset_scale * (u.powf(-1.0 / self.params.offset_tail) - 1.0);
        (d.floor() as Price).min(self.params.max_offset)
    }

    pub fn wake(&mut self, ctx: &mut Ctx<'_>) -> Timestamp {
        while let Some(&Reverse((t, id))) = self.expiries.peek() {
            if t > ctx.now {
                break;
            }
            self.expiries.pop();
            if ctx.book().is_resting(id) {
                ctx.submit(Request::Cancel { target: id });
            }
        }
        let side = if self.rng.bernoulli(0.5) {
            Side::Buy
        } else {
            Side::Sell
        };
        let qty = self.size();
        if self.rng.bernoulli(self.params.market_prob) {
            ctx.submit(Request::Market { side, qty });
        } else {
            let offset = self.offset();
            let reference = ctx.reference_price();
            let price = match side {
                Side::Buy => {
                    let anchor = ctx
                        .book()
                        .best_ask()
                        .unwrap_or((reference + 0.5).ceil() as Price);
                    anchor - 1 - offset
                }
                Side::Sell => {
                    let anchor = ctx
                        .book()
                        .best_bid()
                        .unwrap_or((reference - 0.5).floor() as Price);
                    anchor + 1 + offset
                }
            };
            let price = ctx.clamp_price(price);
            let id = ctx.submit(Request::Limit { side, price, qty });
            let expiry = ctx.after(self.rng.exp(1.0 / self.params.mean_lifetime));
            self.expiries.push(Reverse((expiry, id)));
        }
        ctx.after(self.clock.exp(self.params.rate))
    }

    pub fn on_report(&mut self, report: &Report) {
        fill_into(&mut self.account, report);
    }
}
