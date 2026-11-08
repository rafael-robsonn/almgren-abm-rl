use std::cmp::Reverse;
use std::collections::BinaryHeap;

use matching_engine::{
    AgentId, Exchange, ExchangeConfig, History, HistoryConfig, LatencyModel, OrderBook, OrderId,
    Price, Report, Request, Timestamp,
};
use serde::{Deserialize, Serialize};

use crate::agents::{
    Agent, ArbitrageParams, Arbitrageur, Ctx, MarketMaker, MarketMakerParams, MomentumParams,
    MomentumTrader, NoiseParams, NoiseTrader,
};
use crate::fundamental::{Fundamental, OuParams};
use crate::rng::derive_seed;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarketConfig {
    pub tick_size: f64,
    pub n_ticks: usize,
    pub initial_price: Price,
    pub fundamental: OuParams,
    pub n_noise: usize,
    pub noise: NoiseParams,
    pub n_market_makers: usize,
    pub market_maker: MarketMakerParams,
    pub n_momentum: usize,
    pub momentum: MomentumParams,
    pub n_arbitrageurs: usize,
    pub arbitrage: ArbitrageParams,
    pub latency_base_ns: u64,
    pub latency_jitter_ns: u64,
    #[serde(skip)]
    pub history: HistoryConfig,
}

impl Default for MarketConfig {
    fn default() -> Self {
        let initial_price = 10_000;
        Self {
            tick_size: 0.01,
            n_ticks: 1 << 15,
            initial_price,
            fundamental: OuParams {
                mu: initial_price as f64,
                kappa: std::f64::consts::LN_2 / 7_200.0,
                sigma: 1.5,
                dt: 0.1,
            },
            n_noise: 10,
            noise: NoiseParams {
                rate: 1.2,
                market_prob: 0.17,
                size_log_mean: 2.1,
                size_log_sd: 1.1,
                max_size: 200,
                offset_scale: 2.0,
                offset_tail: 1.0,
                max_offset: 40,
                mean_lifetime: 10.0,
            },
            n_market_makers: 4,
            market_maker: MarketMakerParams {
                rate: 4.0,
                gamma: 0.05,
                k: 1.0,
                horizon: 20.0,
                quote_size: 25,
                max_inventory: 400,
                vol_halflife_min: 30.0,
                vol_halflife_max: 300.0,
                initial_variance: 1.0,
                variance_floor: 0.05,
                variance_cap: 25.0,
                max_half_spread: 15.0,
                max_skew: 10.0,
            },
            n_momentum: 6,
            momentum: MomentumParams {
                rate: 0.2,
                horizon_min: 60.0,
                horizon_max: 1_200.0,
                signal_scale: 20.0,
                max_position: 150,
                min_trade: 5,
                max_order: 10,
            },
            n_arbitrageurs: 8,
            arbitrage: ArbitrageParams {
                rate_min: 0.1,
                rate_max: 0.8,
                threshold_min: 3.0,
                threshold_max: 16.0,
                scale: 30.0,
                max_position: 1_500,
                min_trade: 5,
                max_order: 20,
            },
            latency_base_ns: 50_000,
            latency_jitter_ns: 50_000,
            history: HistoryConfig::default(),
        }
    }
}

/// Discrete-event kernel: agent decision epochs and exchange message arrivals are merged in
/// timestamp order (arrivals first on ties). All randomness descends from one seed through
/// per-agent streams, so a run is a deterministic function of `(config, seed)` and of the
/// timing of external submissions.
#[derive(Clone, Debug)]
pub struct Market {
    config: MarketConfig,
    exchange: Exchange,
    agents: Vec<Agent>,
    schedule: BinaryHeap<Reverse<(Timestamp, AgentId)>>,
    fundamental: Fundamental,
    reports: Vec<(AgentId, Report)>,
    external: Vec<(AgentId, Report)>,
    n_external: u32,
    now: Timestamp,
    wakeups: u64,
}

impl Market {
    pub fn new(config: MarketConfig, seed: u64) -> Self {
        let exchange = Exchange::new(
            ExchangeConfig {
                n_ticks: config.n_ticks,
                latency: LatencyModel::new(config.latency_base_ns, config.latency_jitter_ns),
                history: config.history.clone(),
            },
            derive_seed(seed, 1),
        );
        let fundamental = Fundamental::new(
            config.fundamental,
            config.initial_price as f64,
            derive_seed(seed, 2),
        );
        let mut agents = Vec::new();
        let mut stream = 1_000u64;
        let mut next_seed = || {
            stream += 1;
            derive_seed(seed, stream)
        };
        for _ in 0..config.n_noise {
            agents.push(Agent::Noise(NoiseTrader::new(config.noise.clone(), next_seed())));
        }
        for _ in 0..config.n_market_makers {
            agents.push(Agent::MarketMaker(MarketMaker::new(
                config.market_maker.clone(),
                next_seed(),
            )));
        }
        for _ in 0..config.n_momentum {
            agents.push(Agent::Momentum(MomentumTrader::new(
                config.momentum.clone(),
                next_seed(),
            )));
        }
        for _ in 0..config.n_arbitrageurs {
            agents.push(Agent::Arbitrageur(Arbitrageur::new(
                config.arbitrage.clone(),
                next_seed(),
            )));
        }
        let mut schedule = BinaryHeap::with_capacity(agents.len() + 1);
        for (i, a) in agents.iter_mut().enumerate() {
            schedule.push(Reverse((a.first_wakeup(), i as AgentId)));
        }
        Self {
            config,
            exchange,
            agents,
            schedule,
            fundamental,
            reports: Vec::with_capacity(256),
            external: Vec::new(),
            n_external: 0,
            now: 0,
            wakeups: 0,
        }
    }

    pub fn run_until(&mut self, t_end: Timestamp) {
        loop {
            let msg_t = self.exchange.next_arrival().unwrap_or(Timestamp::MAX);
            let wake_t = self.schedule.peek().map_or(Timestamp::MAX, |r| r.0.0);
            if msg_t.min(wake_t) > t_end {
                break;
            }
            if msg_t <= wake_t {
                self.now = msg_t;
                self.exchange.process_next(&mut self.reports);
                let n = self.agents.len();
                for (a, r) in self.reports.drain(..) {
                    match self.agents.get_mut(a as usize) {
                        Some(agent) => agent.on_report(&r),
                        None => {
                            debug_assert!((a as usize) < n + self.n_external as usize);
                            self.external.push((a, r));
                        }
                    }
                }
            } else {
                let Reverse((t, id)) = self.schedule.pop().expect("schedule is non-empty");
                self.now = t;
                let mut ctx = Ctx {
                    now: t,
                    id,
                    exchange: &mut self.exchange,
                    fundamental: &mut self.fundamental,
                    fallback_price: self.config.initial_price as f64,
                };
                let next = self.agents[id as usize].wake(&mut ctx);
                self.schedule.push(Reverse((next, id)));
                self.wakeups += 1;
            }
        }
        self.now = self.now.max(t_end);
        self.exchange.advance_to(t_end);
    }

    pub fn run_for(&mut self, dt: Timestamp) {
        self.run_until(self.now + dt);
    }

    /// Registers an order-entry session that is not driven by the kernel. Its reports are
    /// buffered until drained.
    pub fn register_external(&mut self, latency: Option<LatencyModel>) -> AgentId {
        let id = (self.agents.len() as u32) + self.n_external;
        self.n_external += 1;
        if let Some(l) = latency {
            self.exchange.set_latency(id, l);
        }
        id
    }

    pub fn submit(&mut self, agent: AgentId, request: Request) -> OrderId {
        debug_assert!(agent as usize >= self.agents.len(), "internal agents act through the kernel");
        self.exchange.submit(agent, request, self.now)
    }

    pub fn drain_external(&mut self, out: &mut Vec<(AgentId, Report)>) {
        out.append(&mut self.external);
    }

    #[inline]
    pub fn book(&self) -> &OrderBook {
        self.exchange.book()
    }

    pub fn exchange(&self) -> &Exchange {
        &self.exchange
    }

    pub fn history(&self) -> &History {
        self.exchange.history()
    }

    pub fn history_mut(&mut self) -> &mut History {
        self.exchange.history_mut()
    }

    #[inline]
    pub fn now(&self) -> Timestamp {
        self.now
    }

    pub fn config(&self) -> &MarketConfig {
        &self.config
    }

    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }

    pub fn messages(&self) -> u64 {
        self.exchange.stats().messages
    }

    pub fn wakeups(&self) -> u64 {
        self.wakeups
    }

    /// Latent fundamental at the current time, for diagnostics. Querying does not perturb the
    /// path. Observation builders for external agents must not call this.
    pub fn latent_fundamental(&mut self) -> f64 {
        self.fundamental.value_at(self.now)
    }

    /// Counterfactual intervention on the latent state, used to test that external
    /// observations carry no direct dependence on it.
    pub fn shift_latent_fundamental(&mut self, delta: f64) {
        self.fundamental.value_at(self.now);
        self.fundamental.shift(delta);
    }

    #[inline]
    pub fn mid(&self) -> Option<f64> {
        self.book().mid()
    }
}
