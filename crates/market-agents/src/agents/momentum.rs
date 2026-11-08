use matching_engine::{Qty, Report, Request, Side, Timestamp, secs, to_secs};
use serde::{Deserialize, Serialize};

use super::{Account, Ctx, fill_into};
use crate::rng::{Rng, derive_seed};

/// Trend follower. At Poisson epochs of intensity `rate` it updates the exponentially weighted
/// sum of mid changes `m <- m * exp(-dt / h) + dmid`, with time constant `h` drawn
/// log-uniformly from `[horizon_min, horizon_max]`, targets the position
/// `max_position * tanh(m / signal_scale)` and closes the gap with a market order of at most
/// `max_order` when it is at least `min_trade`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MomentumParams {
    pub rate: f64,
    pub horizon_min: f64,
    pub horizon_max: f64,
    pub signal_scale: f64,
    pub max_position: i64,
    pub min_trade: Qty,
    pub max_order: Qty,
}

#[derive(Clone, Debug)]
pub struct MomentumTrader {
    params: MomentumParams,
    clock: Rng,
    pub(super) account: Account,
    horizon: f64,
    trend: f64,
    last_mid: f64,
    last_t: Option<Timestamp>,
}

impl MomentumTrader {
    pub fn new(params: MomentumParams, seed: u64) -> Self {
        let mut init = Rng::new(derive_seed(seed, 2));
        let horizon = init.log_uniform(params.horizon_min, params.horizon_max);
        Self {
            params,
            clock: Rng::new(derive_seed(seed, 0)),
            account: Account::default(),
            horizon,
            trend: 0.0,
            last_mid: 0.0,
            last_t: None,
        }
    }

    pub fn first_wakeup(&mut self) -> Timestamp {
        secs(self.clock.exp(self.params.rate)).max(1)
    }

    pub fn wake(&mut self, ctx: &mut Ctx<'_>) -> Timestamp {
        let mid = ctx.reference_price();
        if let Some(t0) = self.last_t {
            let dt = to_secs(ctx.now - t0);
            self.trend = self.trend * (-dt / self.horizon).exp() + (mid - self.last_mid);
        }
        self.last_mid = mid;
        self.last_t = Some(ctx.now);
        let p = &self.params;
        let target = (p.max_position as f64 * (self.trend / p.signal_scale).tanh()).round() as i64;
        let gap = target - self.account.inventory;
        if gap.unsigned_abs() >= p.min_trade {
            let side = if gap > 0 { Side::Buy } else { Side::Sell };
            let qty = gap.unsigned_abs().min(p.max_order);
            ctx.submit(Request::Market { side, qty });
        }
        ctx.after(self.clock.exp(self.params.rate))
    }

    pub fn on_report(&mut self, report: &Report) {
        fill_into(&mut self.account, report);
    }
}
