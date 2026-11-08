use market_agents::{Market, MarketConfig};
use matching_engine::{AgentId, LatencyModel, OrderBook, Price, Qty, Report, Request, Side, secs};
use serde::{Deserialize, Serialize};

pub const OBS_DIM: usize = 10;

pub type Observation = [f32; OBS_DIM];

pub const FEATURE_NAMES: [&str; OBS_DIM] = [
    "time_remaining",
    "inventory_remaining",
    "spread",
    "imbalance_l1",
    "imbalance_5ticks",
    "bid_depth_5ticks",
    "mid_drift_since_arrival",
    "mid_change_last_step",
    "realized_vol_last_step",
    "slippage_last_child",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub market: MarketConfig,
    pub quantity: Qty,
    pub horizon_secs: f64,
    pub n_steps: usize,
    pub warmup_secs: f64,
    pub latency_base_ns: u64,
    pub latency_jitter_ns: u64,
    /// Mean-variance coefficient `lambda` (1/$) of the per-step penalty `lambda sigma^2 tau x_k^2`.
    pub risk_aversion: f64,
    /// Volatility (in $/sqrt(s)) used by the risk penalty; set from calibration.
    pub sigma: f64,
    /// Ticks below the last touch at which an unfillable terminal residual is marked.
    pub residual_penalty_ticks: Price,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            market: MarketConfig::default(),
            quantity: 3_000,
            horizon_secs: 300.0,
            n_steps: 30,
            warmup_secs: 300.0,
            latency_base_ns: 50_000,
            latency_jitter_ns: 50_000,
            risk_aversion: 0.0,
            sigma: 0.0,
            residual_penalty_ticks: 50,
        }
    }
}

impl ExecutionConfig {
    pub fn tau(&self) -> f64 {
        self.horizon_secs / self.n_steps as f64
    }

    pub fn tick(&self) -> f64 {
        self.market.tick_size
    }
}

/// Information available to a policy at a decision epoch besides the normalised observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepContext {
    pub step: usize,
    pub n_steps: usize,
    pub remaining: Qty,
    pub quantity: Qty,
}

impl StepContext {
    pub fn steps_left(&self) -> usize {
        self.n_steps - self.step
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StepOutcome {
    pub reward: f64,
    pub cost_bps: f64,
    pub penalty_bps: f64,
    pub executed: Qty,
    pub done: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpisodeSummary {
    pub seed: u64,
    pub arrival_mid: f64,
    pub vwap: f64,
    pub shortfall_bps: f64,
    pub penalty_bps: f64,
    pub residual: Qty,
    pub child_orders: u32,
    pub steps: u32,
}

impl EpisodeSummary {
    pub fn objective_bps(&self) -> f64 {
        self.shortfall_bps + self.penalty_bps
    }
}

#[derive(Clone, Debug, Default)]
struct Trace {
    arrival_mid: f64,
    last_mid: f64,
    step_mids: Vec<f64>,
    last_slippage_ticks: f64,
}

/// Single liquidation episode on a private market instance. Observations are built by
/// [`observe`] from the order book and the agent's own executions only.
#[derive(Clone, Debug)]
pub struct ExecutionEnv {
    config: ExecutionConfig,
    market: Option<Market>,
    agent: AgentId,
    seed: u64,
    step: usize,
    remaining: Qty,
    proceeds: f64,
    executed: Qty,
    shortfall_bps: f64,
    penalty_bps: f64,
    residual: Qty,
    child_orders: u32,
    done: bool,
    trace: Trace,
    reports: Vec<(AgentId, Report)>,
}

impl ExecutionEnv {
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            market: None,
            agent: 0,
            seed: 0,
            step: 0,
            remaining: 0,
            proceeds: 0.0,
            executed: 0,
            shortfall_bps: 0.0,
            penalty_bps: 0.0,
            residual: 0,
            child_orders: 0,
            done: true,
            trace: Trace::default(),
            reports: Vec::new(),
        }
    }

    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    pub fn reset(&mut self, seed: u64) -> Observation {
        let mut market = Market::new(self.config.market.clone(), seed);
        market.run_until(secs(self.config.warmup_secs));
        self.agent = market.register_external(Some(LatencyModel::new(
            self.config.latency_base_ns,
            self.config.latency_jitter_ns,
        )));
        let mid = reference_mid(market.book(), self.config.market.initial_price as f64);
        self.market = Some(market);
        self.seed = seed;
        self.step = 0;
        self.remaining = self.config.quantity;
        self.proceeds = 0.0;
        self.executed = 0;
        self.shortfall_bps = 0.0;
        self.penalty_bps = 0.0;
        self.residual = 0;
        self.child_orders = 0;
        self.done = false;
        self.trace = Trace {
            arrival_mid: mid,
            last_mid: mid,
            step_mids: Vec::with_capacity(16),
            last_slippage_ticks: 0.0,
        };
        self.observation()
    }

    pub fn context(&self) -> StepContext {
        StepContext {
            step: self.step,
            n_steps: self.config.n_steps,
            remaining: self.remaining,
            quantity: self.config.quantity,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn observation(&self) -> Observation {
        let market = self.market.as_ref().expect("reset before observing");
        observe(market.book(), &self.context(), &self.trace)
    }

    /// Sends a sell market order for `min(qty, remaining)` shares (all remaining shares at the
    /// last step) and advances the market by `tau`. With pre-trade mid `m_k`, fills `(p_i, q_i)`
    /// and post-step holdings `x_{k+1}`, the reward in bps of arrival notional `Q P0` is
    /// `[sum_i q_i (p_i - m_k) + x_{k+1} (m_{k+1} - m_k) - lambda sigma^2 tau x_{k+1}^2] / (Q P0) * 1e4`.
    /// Because `m_0 = P0` and `x_N = 0`, the episode sum of rewards equals `-(IS + penalty)`;
    /// the reward-to-go differs from the raw shortfall-to-go by the potential
    /// `x_k (m_k - P0)`, so the optimal policy is unchanged.
    pub fn step(&mut self, qty: Qty) -> StepOutcome {
        if self.done {
            return StepOutcome {
                done: true,
                ..StepOutcome::default()
            };
        }
        let cfg = &self.config;
        let last = self.step + 1 == cfg.n_steps;
        let qty = if last { self.remaining } else { qty.min(self.remaining) };
        let tick = cfg.tick();
        let p0 = self.trace.arrival_mid * tick;
        let notional = cfg.quantity as f64 * p0;
        let tau = cfg.tau();
        let market = self.market.as_mut().expect("reset before stepping");
        let mid_before = reference_mid(market.book(), self.trace.last_mid);
        let t0 = market.now();
        if qty > 0 {
            market.submit(self.agent, Request::Market { side: Side::Sell, qty });
            self.child_orders += 1;
        }
        self.trace.step_mids.clear();
        self.trace.step_mids.push(mid_before);
        let sub = (tau.round() as usize).max(1);
        let mut step_proceeds = 0.0;
        let mut step_exec: Qty = 0;
        let mut worst: Option<Price> = None;
        for i in 1..=sub {
            market.run_until(t0 + secs(tau * i as f64 / sub as f64));
            self.reports.clear();
            market.drain_external(&mut self.reports);
            for (_, r) in &self.reports {
                if let Report::Fill { price, qty, .. } = *r {
                    step_proceeds += price as f64 * qty as f64 * tick;
                    step_exec += qty;
                    worst = Some(worst.map_or(price, |w: Price| w.min(price)));
                }
            }
            let m = reference_mid(market.book(), self.trace.step_mids[i - 1]);
            self.trace.step_mids.push(m);
        }
        self.remaining -= step_exec;
        self.trace.last_slippage_ticks = if step_exec > 0 {
            mid_before - step_proceeds / (step_exec as f64 * tick)
        } else {
            0.0
        };
        self.executed += step_exec;
        let mut sold = step_exec;
        if last && self.remaining > 0 {
            let touch = market
                .book()
                .best_bid()
                .or(worst)
                .unwrap_or(self.trace.arrival_mid as Price);
            let mark = (touch - cfg.residual_penalty_ticks).max(0) as f64 * tick;
            step_proceeds += self.remaining as f64 * mark;
            sold += self.remaining;
            self.residual = self.remaining;
            self.remaining = 0;
        }
        self.proceeds += step_proceeds;
        let mid_after = *self.trace.step_mids.last().expect("non-empty");
        self.trace.last_mid = mid_after;
        let x_next = self.remaining as f64;
        let cost = sold as f64 * p0 - step_proceeds;
        let execution = step_proceeds - sold as f64 * mid_before * tick;
        let carry = x_next * (mid_after - mid_before) * tick;
        let penalty = cfg.risk_aversion * cfg.sigma * cfg.sigma * tau * x_next * x_next;
        let cost_bps = cost / notional * 1e4;
        let penalty_bps = penalty / notional * 1e4;
        self.shortfall_bps += cost_bps;
        self.penalty_bps += penalty_bps;
        self.step += 1;
        self.done = self.step == cfg.n_steps || self.remaining == 0;
        StepOutcome {
            reward: (execution + carry - penalty) / notional * 1e4,
            cost_bps,
            penalty_bps,
            executed: step_exec,
            done: self.done,
        }
    }

    pub fn summary(&self) -> EpisodeSummary {
        let tick = self.config.tick();
        let sold = self.executed + self.residual;
        EpisodeSummary {
            seed: self.seed,
            arrival_mid: self.trace.arrival_mid * tick,
            vwap: if sold > 0 { self.proceeds / sold as f64 } else { 0.0 },
            shortfall_bps: self.shortfall_bps,
            penalty_bps: self.penalty_bps,
            residual: self.residual,
            child_orders: self.child_orders,
            steps: self.step as u32,
        }
    }

    pub fn market(&self) -> Option<&Market> {
        self.market.as_ref()
    }

    pub fn market_mut(&mut self) -> Option<&mut Market> {
        self.market.as_mut()
    }
}

fn reference_mid(book: &OrderBook, fallback: f64) -> f64 {
    book.mid()
        .or_else(|| book.best_bid().map(|b| b as f64 + 0.5))
        .or_else(|| book.best_ask().map(|a| a as f64 - 0.5))
        .unwrap_or(fallback)
}

fn ratio(a: Qty, b: Qty) -> f32 {
    if a + b == 0 {
        0.0
    } else {
        (a as f64 - b as f64) as f32 / (a + b) as f32
    }
}

fn observe(book: &OrderBook, ctx: &StepContext, trace: &Trace) -> Observation {
    let slice = (ctx.quantity as f64 / ctx.n_steps as f64).max(1.0);
    let spread = book.spread().unwrap_or(10) as f64;
    let bid1 = book.best_qty(Side::Buy);
    let ask1 = book.best_qty(Side::Sell);
    let bid5 = book.qty_within(Side::Buy, 4);
    let ask5 = book.qty_within(Side::Sell, 4);
    let mid = reference_mid(book, trace.last_mid);
    let step_change = trace
        .step_mids
        .last()
        .zip(trace.step_mids.first())
        .map_or(0.0, |(b, a)| b - a);
    let rv = if trace.step_mids.len() > 1 {
        (trace
            .step_mids
            .windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f64>()
            / (trace.step_mids.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    let clip = |x: f64, lim: f64| x.clamp(-lim, lim) as f32;
    [
        1.0 - ctx.step as f32 / ctx.n_steps as f32,
        ctx.remaining as f32 / ctx.quantity as f32,
        clip((spread - 1.0) / 4.0, 5.0),
        ratio(bid1, ask1),
        ratio(bid5, ask5),
        clip((1.0 + bid5 as f64 / slice).ln() / 3.0, 5.0),
        clip((mid - trace.arrival_mid) / 20.0, 5.0),
        clip(step_change / 5.0, 5.0),
        clip(rv / 2.0, 5.0),
        clip(trace.last_slippage_ticks / 5.0, 5.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_is_invariant_to_counterfactual_fundamental() {
        let mut env = ExecutionEnv::new(ExecutionConfig::default());
        env.reset(3);
        env.step(100);
        let obs = env.observation();
        for delta in [-400.0, 250.0] {
            let mut twin = env.clone();
            let m = twin.market_mut().unwrap();
            let before = m.latent_fundamental();
            m.shift_latent_fundamental(delta);
            assert!((m.latent_fundamental() - before - delta).abs() < 1e-9);
            assert_eq!(twin.observation(), obs);
        }
        let rebuilt = observe(env.market().unwrap().book(), &env.context(), &env.trace);
        assert_eq!(rebuilt, obs);
    }

    #[test]
    fn rewards_sum_to_negative_shortfall_plus_penalty() {
        let cfg = ExecutionConfig {
            risk_aversion: 1e-4,
            sigma: 0.0115,
            ..ExecutionConfig::default()
        };
        for (seed, pattern) in [(1u64, [100u64, 0, 300]), (2, [0, 0, 0]), (3, [900, 900, 900])] {
            let mut env = ExecutionEnv::new(cfg.clone());
            env.reset(seed);
            let mut total = 0.0;
            let mut k = 0;
            while !env.is_done() {
                total += env.step(pattern[k % 3]).reward;
                k += 1;
            }
            let s = env.summary();
            assert!((total + s.objective_bps()).abs() < 1e-9, "seed {seed}: {total} vs {}", s.objective_bps());
            let notional = cfg.quantity as f64 * s.arrival_mid;
            let implied = (cfg.quantity as f64 * s.arrival_mid - s.vwap * cfg.quantity as f64) / notional * 1e4;
            assert!((implied - s.shortfall_bps).abs() < 1e-9);
        }
    }

    #[test]
    fn last_step_liquidates_everything() {
        let mut env = ExecutionEnv::new(ExecutionConfig::default());
        env.reset(4);
        let mut steps = 0;
        while !env.is_done() {
            env.step(0);
            steps += 1;
        }
        assert_eq!(steps, env.config().n_steps);
        assert_eq!(env.context().remaining, 0);
        assert_eq!(env.summary().child_orders, 1);
    }
}
