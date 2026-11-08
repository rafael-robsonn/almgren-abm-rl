mod arbitrageur;
mod market_maker;
mod momentum;
mod noise;

pub use arbitrageur::{ArbitrageParams, Arbitrageur};
pub use market_maker::{MarketMaker, MarketMakerParams};
pub use momentum::{MomentumParams, MomentumTrader};
pub use noise::{NoiseParams, NoiseTrader};

use matching_engine::{
    AgentId, Exchange, OrderBook, OrderId, Price, Qty, Report, Request, Side, Timestamp, secs,
};

use crate::fundamental::Fundamental;

pub struct Ctx<'a> {
    pub now: Timestamp,
    pub id: AgentId,
    pub exchange: &'a mut Exchange,
    pub fundamental: &'a mut Fundamental,
    pub fallback_price: f64,
}

impl Ctx<'_> {
    #[inline]
    pub fn book(&self) -> &OrderBook {
        self.exchange.book()
    }

    #[inline]
    pub fn submit(&mut self, request: Request) -> OrderId {
        self.exchange.submit(self.id, request, self.now)
    }

    /// Mid quote, falling back to the surviving touch, the last trade, then the configured price.
    #[inline]
    pub fn reference_price(&self) -> f64 {
        let b = self.book();
        match (b.best_bid(), b.best_ask()) {
            (Some(x), Some(y)) => 0.5 * (x + y) as f64,
            (Some(x), None) => x as f64 + 0.5,
            (None, Some(y)) => y as f64 - 0.5,
            (None, None) => self
                .exchange
                .last_trade()
                .map_or(self.fallback_price, |p| p as f64),
        }
    }

    #[inline]
    pub fn clamp_price(&self, p: Price) -> Price {
        p.clamp(1, self.book().n_ticks() as Price - 2)
    }

    #[inline]
    pub fn after(&self, seconds: f64) -> Timestamp {
        self.now + secs(seconds).max(1)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Account {
    pub inventory: i64,
    pub cash: f64,
    pub traded: u64,
}

impl Account {
    #[inline]
    pub fn apply(&mut self, side: Side, price: Price, qty: Qty) {
        let q = qty as i64 * side.sign();
        self.inventory += q;
        self.cash -= (q * price) as f64;
        self.traded += qty;
    }

    pub fn mark_to_market(&self, price: f64) -> f64 {
        self.cash + self.inventory as f64 * price
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AgentKind {
    Noise,
    MarketMaker,
    Momentum,
    Arbitrageur,
}

#[derive(Clone, Debug)]
pub enum Agent {
    Noise(NoiseTrader),
    MarketMaker(MarketMaker),
    Momentum(MomentumTrader),
    Arbitrageur(Arbitrageur),
}

impl Agent {
    #[inline]
    pub fn wake(&mut self, ctx: &mut Ctx<'_>) -> Timestamp {
        match self {
            Agent::Noise(a) => a.wake(ctx),
            Agent::MarketMaker(a) => a.wake(ctx),
            Agent::Momentum(a) => a.wake(ctx),
            Agent::Arbitrageur(a) => a.wake(ctx),
        }
    }

    #[inline]
    pub fn first_wakeup(&mut self) -> Timestamp {
        match self {
            Agent::Noise(a) => a.first_wakeup(),
            Agent::MarketMaker(a) => a.first_wakeup(),
            Agent::Momentum(a) => a.first_wakeup(),
            Agent::Arbitrageur(a) => a.first_wakeup(),
        }
    }

    #[inline]
    pub fn on_report(&mut self, report: &Report) {
        match self {
            Agent::Noise(a) => a.on_report(report),
            Agent::MarketMaker(a) => a.on_report(report),
            Agent::Momentum(a) => a.on_report(report),
            Agent::Arbitrageur(a) => a.on_report(report),
        }
    }

    pub fn kind(&self) -> AgentKind {
        match self {
            Agent::Noise(_) => AgentKind::Noise,
            Agent::MarketMaker(_) => AgentKind::MarketMaker,
            Agent::Momentum(_) => AgentKind::Momentum,
            Agent::Arbitrageur(_) => AgentKind::Arbitrageur,
        }
    }

    pub fn account(&self) -> &Account {
        match self {
            Agent::Noise(a) => &a.account,
            Agent::MarketMaker(a) => &a.account,
            Agent::Momentum(a) => &a.account,
            Agent::Arbitrageur(a) => &a.account,
        }
    }
}

#[inline]
pub(crate) fn fill_into(account: &mut Account, report: &Report) {
    if let Report::Fill {
        side, price, qty, ..
    } = *report
    {
        account.apply(side, price, qty);
    }
}
