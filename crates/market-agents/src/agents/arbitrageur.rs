use matching_engine::{Qty, Report, Request, Side, Timestamp, secs};
use serde::{Deserialize, Serialize};

use super::{Account, Ctx, fill_into};
use crate::rng::{Rng, derive_seed};

/// Informed value trader; the only agent class that reads the latent fundamental `V_t`. Each
/// agent draws an intensity `rate` and a no-trade band `threshold` log-uniformly from the
/// configured ranges. With deviation `d = V_t - mid` it targets
/// `sign(d) * max_position * min(1, max(0, |d| - threshold) / scale)` and trades the gap with
/// market orders of at most `max_order` at Poisson epochs of intensity `rate`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArbitrageParams {
    pub rate_min: f64,
    pub rate_max: f64,
    pub threshold_min: f64,
    pub threshold_max: f64,
    pub scale: f64,
    pub max_position: i64,
    pub min_trade: Qty,
    pub max_order: Qty,
}

#[derive(Clone, Debug)]
pub struct Arbitrageur {
    params: ArbitrageParams,
    clock: Rng,
    pub(super) account: Account,
    rate: f64,
    threshold: f64,
}

impl Arbitrageur {
    pub fn new(params: ArbitrageParams, seed: u64) -> Self {
        let mut init = Rng::new(derive_seed(seed, 2));
        let rate = init.log_uniform(params.rate_min, params.rate_max);
        let threshold = init.log_uniform(params.threshold_min, params.threshold_max);
        Self {
            params,
            clock: Rng::new(derive_seed(seed, 0)),
            account: Account::default(),
            rate,
            threshold,
        }
    }

    pub fn first_wakeup(&mut self) -> Timestamp {
        secs(self.clock.exp(self.rate)).max(1)
    }

    pub fn wake(&mut self, ctx: &mut Ctx<'_>) -> Timestamp {
        let value = ctx.fundamental.value_at(ctx.now);
        let d = value - ctx.reference_price();
        let p = &self.params;
        let excess = ((d.abs() - self.threshold).max(0.0) / p.scale).min(1.0);
        let target = (d.signum() * p.max_position as f64 * excess).round() as i64;
        let gap = target - self.account.inventory;
        if gap.unsigned_abs() >= p.min_trade {
            let side = if gap > 0 { Side::Buy } else { Side::Sell };
            let qty = gap.unsigned_abs().min(p.max_order);
            ctx.submit(Request::Market { side, qty });
        }
        ctx.after(self.clock.exp(self.rate))
    }

    pub fn on_report(&mut self, report: &Report) {
        fill_into(&mut self.account, report);
    }
}
