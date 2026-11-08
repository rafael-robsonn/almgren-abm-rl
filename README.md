# Agent-based limit order book and recurrent PPO for optimal execution

Rust workspace containing a deterministic discrete-event exchange simulator populated by heterogeneous agents, a statistical validation of the price process it generates, a vectorised execution environment, an Almgren-Chriss baseline calibrated on the simulator, and a recurrent actor-critic trained with PPO in `burn`. Simulation, calibration, training and evaluation run in a single process without a Python boundary.

All numbers below were produced on an AMD Ryzen 9 9950X3D (16 cores, 32 threads), Windows 11, `rustc 1.96.0`, release profile (`lto = "thin"`, `codegen-units = 1`), with the commands listed in [Reproduction](#10-reproduction).

## 1. Workspace

| Crate | Content |
|---|---|
| `matching-engine` | Price-time priority book (limit, market, cancel, modify), sequenced exchange front-end with order-entry latency, L2/L3 history. |
| `market-agents` | Ornstein-Uhlenbeck fundamental, zero-intelligence noise traders, Avellaneda-Stoikov market makers, momentum traders, arbitrageurs; discrete-event kernel. |
| `stylized-facts` | Moments, autocorrelation, Ljung-Box, Jarque-Bera, Hill estimator, acceptance criteria; `validate` binary and emergence tests. |
| `abm-gym` | Liquidation MDP on a private market per episode, `rayon`-parallel vectorised environment, Almgren-Chriss model, calibration by common random numbers, evaluation utilities. |
| `rl-ppo-burn` | GRU actor-critic, recurrent PPO, `train` and `evaluate` binaries, benchmark tables. |

Dependency order: `matching-engine` <- `market-agents` <- {`stylized-facts`, `abm-gym`} <- `rl-ppo-burn`. Randomness comes exclusively from `rand_xoshiro::Xoshiro256PlusPlus` streams derived from a root seed through the SplitMix64 finaliser.

## 2. Matching engine

Prices are integer ticks in `[0, n_ticks)` (default `2^15`, tick `$0.01`). Each side is an array of price levels indexed by tick plus an occupancy bitset of 64-bit words; the best price after a level empties is found with `leading_zeros`/`trailing_zeros` word scans. Orders live in a slab; each level holds an intrusive doubly linked FIFO through the slab. An order handle packs `(slot, generation)`: the generation is incremented when a slot is freed, so stale handles never resolve.

| Operation | Semantics | Cost |
|---|---|---|
| Limit | executes against the opposite side up to the limit price, remainder rests at the tail of its level | O(1) insertion plus O(fills) |
| Market | immediate-or-cancel; unfilled remainder reported as `Expired` | O(fills) |
| Cancel | removes the resting remainder; owner checked | O(1) |
| Modify | sets the new resting quantity. Reduction at an unchanged price keeps queue position; any other change re-queues the order under the same handle and may execute if marketable | O(1) or as Limit |

Best-price maintenance after a level is depleted costs at most `n_ticks / 64` word reads.

`Exchange` wraps the book with order-entry latency. A message sent at `t` by agent `i` arrives at `max(t + base + Exp(jitter_mean), last_arrival_i)`, so messages of one agent never overtake each other; arrivals are applied in `(arrival, sequence)` order. Handles for new orders are allocated at submission, so an agent can cancel or modify an order that is still in flight. Market data is not delayed: agents read the book at their decision time.

History recording is optional. The L3 stream contains `Add`, `Execute` and `Cancel` events (a re-queueing modify is logged as a full `Cancel`, the executions it triggers, then `Add`); L2 snapshots of the top `k` levels are taken on a fixed time grid.

Tests (`crates/matching-engine/tests/book.rs`): a randomised differential test against a `BTreeMap`/`VecDeque` reference book (60,000 mixed operations; fills and the full L2 compared after every operation; structural invariants checked periodically), reconstruction of the live book from the L3 log, per-agent FIFO under random jitter, determinism per seed, and unit cases for priority, partial fills, stale handles and modify semantics.

## 3. Market ecology

The kernel merges agent decision epochs and exchange arrivals in timestamp order (arrivals first on ties). Each agent's Poisson clock draws from its own stream, separate from its decision randomness, so the decision times of a market and of a clone that receives an extra order coincide; this is used for calibration. A run is a deterministic function of `(MarketConfig, seed)` and of the timing of external submissions.

Default population: 10 noise traders, 4 market makers, 6 momentum traders, 8 arbitrageurs. Order-entry latency `50 µs + Exp(50 µs)` for all agents. Initial price `P0 = 10,000` ticks (`$100.00`).

### 3.1 Fundamental value

`dV = kappa (mu - V) dt + sigma_V dW` in ticks, `mu = 10,000`, `kappa = ln 2 / 7,200 s^-1` (half-life 2 h), `sigma_V = 1.5 ticks/sqrt(s)`. The path is sampled on a 0.1 s grid with the exact transition

`V_{t+dt} | V_t ~ N( mu + (V_t - mu) e^{-kappa dt}, sigma_V^2 (1 - e^{-2 kappa dt}) / (2 kappa) )`

and is a function of the seed only. Only arbitrageurs read `V`.

### 3.2 Noise traders

Decision epochs are a Poisson process of intensity 1.2 s^-1 per agent. With probability 0.17 the agent sends a market order, otherwise a limit order; side is buy or sell with probability 1/2. A buy limit is placed at `best_ask - 1 - D`, a sell at `best_bid + 1 + D`, with `P(D >= d) = (1 + d / 2)^{-1}`, `D <= 40`. Size is `round(exp(Z))`, `Z ~ N(2.1, 1.1^2)`, clipped to `[1, 200]` (median 8, mean 15 shares). Each resting order has an `Exp(mean 10 s)` lifetime and is cancelled at the owner's next epoch after expiry.

### 3.3 Market makers

Avellaneda-Stoikov quoting at Poisson epochs of intensity 4 s^-1. With mid `m`, normalised inventory `q = inventory / 25` and variance estimate `s2` (ticks^2/s):

- reservation price `r = m - clip(q gamma s2 H, -10, 10)`
- half-spread `delta = min(gamma s2 H / 2 + ln(1 + gamma / k) / gamma, 15)`
- quotes `floor(r - delta)` and `ceil(r + delta)` of 25 shares, kept passive against the current touch; one side is withdrawn at `|inventory| >= 400`.

`gamma = 0.05`, `k = 1`, `H = 20 s`. `s2` is the time-decayed average of `(dm)^2 / dt` between epochs, clamped to `[0.05, 25]`, with a half-life drawn per dealer log-uniformly from `[30, 300] s`. Quotes are moved with `Modify`. Widening of quotes after realised volatility rises is one of the feedback channels behind volatility clustering.

### 3.4 Momentum traders

At Poisson epochs of intensity 0.2 s^-1: `M <- M exp(-dt / h) + dm`, target position `round(150 tanh(M / 20))`, gap traded with market orders of 5 to 10 shares. The time constant `h` is drawn per agent log-uniformly from `[60, 1200] s`.

### 3.5 Arbitrageurs

With `d = V - m`, target position `sign(d) * 1500 * min(1, max(0, |d| - theta) / 30)`, gap traded with market orders of 5 to 20 shares at Poisson epochs of intensity `rho`. `theta` is drawn per agent log-uniformly from `[3, 16]` ticks and `rho` from `[0.1, 0.8] s^-1`.

### 3.6 Resulting microstructure

Seed 1, 3,600 s after a 600 s warm-up: mean spread 1.32 ticks, mean quantity at the touch 64 shares per side, 370 shares per side within 5 ticks of the touch, 26 order-entry messages and 71 shares traded per simulated second. Over 16 paths of 20,000 s (seeds 100-115) the mid stayed between 9,740 and 10,182 ticks; the largest one-second log return was 0.58%.

Parameters were selected with `crates/stylized-facts/examples/tune.rs` on seeds 100-131 and then checked on four further seed batches not used in the search (Section 4).

## 4. Stylized facts

Procedure (`cargo run --release -p stylized-facts --bin validate`): 32 independent markets, 600 s warm-up, 20,000 s horizon, mid sampled every second (the last two-sided mid is carried forward while a side is empty), log returns at `dt` in {1, 10, 30, 60, 300} s pooled across paths without crossing path boundaries.

Acceptance criteria (`stylized_facts::Criteria::default()`):

- fat tails: excess kurtosis > 1 and Jarque-Bera p < 1e-6;
- absence of linear autocorrelation: `|rho_r(k)| < 0.05` for `k = 1..10`;
- volatility clustering: `rho_|r|(k) > max(0.05, 1.96 / sqrt(n))` for `k = 1..10` and Ljung-Box p < 1e-6 on `|r|`.

The linear-dependence criterion is a bound on economic magnitude. With `n` between 2,000 and 640,000 a Ljung-Box test on `r` rejects white noise for autocorrelations of order 0.02; the statistic is reported for comparison with the one on `|r|`.

Seeds 100-131:

| dt (s) | n | excess kurtosis | Hill alpha (top 2.5%) | rho_r(1) | max_{k<=10} abs rho_r(k) | rho_abs(1) | rho_abs(10) | rho_abs(50) | Q_LB(r, 10) | Q_LB(abs r, 10) | fat tails / no lin. AC / clustering |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 640,000 | 96.74 | 3.07 | +0.045 | 0.045 | +0.316 | +0.209 | +0.189 | 2,622.6 | 363,132.6 | Y / Y / Y |
| 10 | 64,000 | 17.42 | 3.73 | +0.040 | 0.040 | +0.303 | +0.192 | +0.171 | 149.9 | 31,287.5 | Y / Y / Y |
| 30 | 21,312 | 5.10 | 4.50 | -0.024 | 0.024 | +0.250 | +0.181 | +0.127 | 49.2 | 8,649.3 | Y / Y / Y |
| 60 | 10,656 | 1.93 | 5.47 | -0.037 | 0.042 | +0.231 | +0.196 | +0.104 | 39.7 | 4,459.9 | Y / Y / Y |
| 300 | 2,112 | 0.86 | 8.02 | -0.038 | 0.045 | +0.214 | +0.127 | -0.018 | 17.5 | 530.5 | n / Y / Y |

At `dt = 10 s` the autocorrelation of `|r|` at lags 1, 5, 10, 20, 50, 100 is 0.303, 0.215, 0.192, 0.194, 0.171, 0.145, so it is still 0.145 at a lag of 1,000 s. Excess kurtosis decreases monotonically with the sampling interval and the Hill tail index increases (aggregational Gaussianity); at 300 s kurtosis is below the threshold.

Robustness over five disjoint batches of 32 seeds (base seeds 100, 1000, 2000, 3000, 7000; batches other than 100 were not used in the parameter search):

| dt (s) | excess kurtosis (min-max) | max_{k<=10} abs rho_r(k) (min-max) | batches passing all three criteria |
|---:|---:|---:|---:|
| 1 | 67.1-96.7 | 0.009-0.059 | 4 / 5 |
| 10 | 10.4-17.4 | 0.016-0.045 | 5 / 5 |
| 30 | 3.5-5.1 | 0.024-0.042 | 5 / 5 |
| 60 | 1.5-1.9 | 0.041-0.050 | 5 / 5 |
| 300 | 0.4-1.3 | 0.045-0.066 | 0 / 5 |

Observations from the parameter search, on intermediate configurations: removing the momentum traders brought `rho_r(1)` at 30 s from -0.16 to -0.01, i.e. their overshoot, corrected by arbitrageurs, produced reversal; with homogeneous arbitrageurs (single threshold and intensity) a reversal of -0.07 to -0.10 appeared at lag 1 at 60 s, the common mispricing correction time. Drawing thresholds and intensities from wide ranges spreads the correction over time scales and removes the reversal at any single lag.

`crates/stylized-facts/tests/emergence.rs` asserts the three facts at 10 s and 30 s on seeds 7000-7031, the decrease of kurtosis from 10 s to 60 s to 300 s, that the criteria reject all three facts for i.i.d. Gaussian data, and that they accept a GARCH(1,1) control (`alpha = 0.15`, `beta = 0.8`).

## 5. Throughput

| Benchmark | Measure | Result |
|---|---|---|
| `matching-engine/examples/book_throughput` | book only, random flow (50% limit, 15% market, 25% cancel, 10% modify), 5e6 messages | 23.3 M msg/s |
| same | `Exchange` with latency queue and execution reports, 5e6 messages | 12.0 M msg/s |
| `market-agents/examples/throughput` | full ABM, 28 agents, 200,000 simulated seconds, one thread | 7.2 M msg/s; 16.9 M events/s |
| same | 16 independent markets on 16 threads | 4.6 M msg/s per thread (74 M msg/s aggregate) |
| same | 32 independent markets on 32 threads (SMT) | 3.3-3.6 M msg/s per thread (104-115 M msg/s aggregate) |

A message is an order-entry message processed by the matching engine; an event is a message or an agent decision epoch (many epochs submit nothing). The per-core figure with every physical core loaded is 4.6 M messages/s, against a requirement of 0.5 M events/s.

## 6. Execution problem

A seller must liquidate `Q = 3,000` shares within `T = 300 s` in `N = 30` decision steps (`tau = 10 s`). Each episode builds a fresh market from its seed and runs a 300 s warm-up; the agent then trades through an external order-entry session with latency `50 µs + Exp(50 µs)`. `Q` is about 14% of the volume the market trades in `T`.

MDP:

- Decision epochs `t_k = k tau`, `k = 0..N-1`. Holdings `x_0 = Q`, `x_N = 0`.
- Observation `o_k` (10 features, scaled to order one and clipped):

| Feature | Definition |
|---|---|
| `time_remaining` | `1 - k / N` |
| `inventory_remaining` | `x_k / Q` |
| `spread` | `(spread_ticks - 1) / 4` |
| `imbalance_l1` | `(q_bid1 - q_ask1) / (q_bid1 + q_ask1)` at the touch |
| `imbalance_5ticks` | same ratio for quantities within 4 ticks of each touch (5 price levels) |
| `bid_depth_5ticks` | `ln(1 + D_bid5 / (Q / N)) / 3`, `D_bid5` the bid quantity within 4 ticks of the best bid |
| `mid_drift_since_arrival` | `(m_k - P0) / 20 ticks` |
| `mid_change_last_step` | `(m_k - m_{k-1}) / 5 ticks` |
| `realized_vol_last_step` | RMS of one-second mid changes over the last step, `/ 2 ticks` |
| `slippage_last_child` | pre-trade mid minus VWAP of the previous child order, `/ 5 ticks` |

- Action `a in {0..5}`: sell market order of `round(m_a x_k / (N - k))` shares, `m = (0, 0.5, 1, 1.5, 2, 3)`. Multiplier 1 reproduces TWAP pacing on the residual schedule. At `k = N - 1` the whole remainder is sent. A remainder that the book cannot absorb is marked at the best bid minus 50 ticks.
- Transition: the child order reaches the book after its latency, executes against the visible liquidity, and the market evolves for `tau` with the agents reacting to the trade.
- Implementation shortfall: `IS = (Q P0 - sum_i p_i q_i) / (Q P0) * 1e4` bps with `P0` the mid at `t_0`.
- Reward, in bps of `Q P0`:

  `r_k = [ sum_{i in k} q_i (p_i - m_k) + x_{k+1} (m_{k+1} - m_k) - lambda sigma^2 tau x_{k+1}^2 ] / (Q P0) * 1e4`

  the execution cost against the pre-trade mid, the mark-to-market of the inventory carried through the step, and the Almgren-Chriss variance proxy. Because `m_0 = P0` and `x_N = 0`, `sum_k r_k = -(IS + penalty)` exactly (tested). The reward-to-go differs from the raw shortfall-to-go by `x_k (m_k - P0)`, a function of the state, so the reformulation is potential-based shaping and leaves the optimal policy unchanged while removing the price level from the critic's target.
- Objective: minimise `E[IS] + lambda Var[IS]`, with `lambda` set from the calibrated Almgren-Chriss model so that the urgency is `kappa T = 1` (`lambda = 1.276e-5 1/$`). Expressed per bps of IS, `lambda_bps = lambda Q P0 / 1e4 = 3.83e-4`.

No leakage of the fundamental: the observation is computed by a function whose arguments are the order book, the step context and the agent's own fills; the market object, which owns the latent process, is not an argument. `abm-gym/src/env.rs` tests that shifting the latent fundamental of a cloned environment by -400 and +250 ticks leaves the observation bit-identical. Information about `V` reaches the agent only through prices and order flow generated by the arbitrageurs.

`VecEnv` steps `n` environments in lockstep with `rayon`; each environment owns its market and random streams, and a test checks that parallel results equal sequential rollouts for the same seeds.

## 7. Almgren-Chriss baseline

Discrete Almgren-Chriss (2000) model with `S_k = S_{k-1} + sigma sqrt(tau) xi_k - tau g(n_k / tau)`, `g(v) = gamma v`, temporary impact `h(v) = epsilon sgn(v) + eta v`. With `eta_tilde = eta - gamma tau / 2` and `kappa` solving `2 (cosh(kappa tau) - 1) / tau^2 = lambda sigma^2 / eta_tilde`:

- holdings `x_j = X sinh(kappa (T - t_j)) / sinh(kappa T)` (TWAP for `lambda = 0`),
- `E[C] = gamma X^2 / 2 + epsilon sum |n_k| + (eta_tilde / tau) sum n_k^2`,
- `Var[C] = sigma^2 tau sum_{k=1}^{N} x_k^2`.

Unit tests check the discrete Euler-Lagrange equation, that perturbing any interior holding raises `E + lambda Var`, and the `lambda <-> kappa` inversion. The policy tracks the target holdings `x_{k+1}` in integer shares.

Calibration (`abm_gym::calibrate`, 64 seeds, each probe on a clone of the same post-warm-up market):

- `sigma^2 = E[(dm)^2] / tau` from 10 s mid increments over 1,800 s;
- temporary impact: OLS of the immediate per-share cost of one sell market order of size `q` against the pre-trade mid, `c(q) = epsilon + (eta / tau) q`;
- permanent impact `gamma`: per-share mid difference between a clone that sold 300 shares and an unperturbed clone, `tau` seconds later. The probe session has zero latency jitter and consumes no randomness, so both clones see the same agent clocks and decision draws.

| Parameter | Value |
|---|---|
| `sigma` | 0.01228 $/sqrt(s) (1.23 ticks/sqrt(s)) |
| `epsilon` | 0.530 cents/share |
| `eta` | 6.42e-4 $ s/share |
| `gamma` | 9.37e-5 $/share |
| `eta_tilde` | 1.73e-4 $ s/share |
| OLS R^2 (per-episode costs) | 0.35 |

| Child order (shares) | 25 | 50 | 100 | 150 | 200 | 300 |
|---|---:|---:|---:|---:|---:|---:|
| Mean cost vs pre-trade mid (ticks/share) | 0.76 | 0.91 | 1.16 | 1.39 | 1.67 | 2.59 |

| Time after a 300-share sell (s) | 1 | 5 | 10 | 30 | 60 | 120 | 300 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mid displacement vs unperturbed clone (ticks) | 4.56 | 3.52 | 2.81 | 2.03 | 1.84 | 1.86 | 1.14 | 0.93 |

Instantaneous cost is convex in child size and impact is transient with a slowly decaying tail; the model assumes both are linear and permanent. Consequences visible in Section 9: the model predicts `E[IS] = 14.77` bps for TWAP, of which 14.06 bps is the permanent term `gamma X^2 / 2`, against 7.58 bps realised; it predicts a cost increase of 0.033 bps for `kappa T = 2` and 0.175 bps for `kappa T = 4` relative to TWAP, against 0.75 and 4.64 bps realised. The predicted standard deviation for TWAP, 11.97 bps, matches the realised 11.93 bps; for `kappa T = 4` the model predicts 7.00 bps and 10.48 bps is realised.

## 8. Recurrent PPO

Architecture (26,119 parameters): `e_t = tanh(W_e o_t + b_e)` (10 -> 64), `h_t = GRU(e_t, h_{t-1})` (64, `reset_after`), policy logits `W_pi h_t + b_pi` (64 -> 6, initialised `U(-0.01, 0.01)`), value `w_v . h_t + b_v`. Actor and critic share the encoder and the recurrent state.

Training (`rl_ppo_burn::Trainer`):

- Rollout: `n_envs` complete episodes per iteration on fresh seeds, actions sampled from the categorical policy with the GRU state carried across steps. Episodes have a fixed horizon of 30 steps; steps after an early termination are masked.
- Update: the GRU state is recomputed from zero over each whole sequence; a test checks that the step recurrence used in rollouts and the sequence pass used in updates give the same logits and values within 1e-5.
- Advantages: GAE with `gamma = 1`, `lambda = 0.95`, normalised over the batch.
- Loss: `-E[min(rho A, clip(rho, 0.8, 1.2) A)] + 0.5 E[(V - R)^2] - 0.005 E[H(pi)]` over unmasked steps; rewards scaled by 0.1.
- Optimiser: Adam (`eps = 1e-5`), learning rate 1e-3 decayed linearly to 5% of its initial value, gradient clipping by norm 1.0. Clipping in `burn` 0.22 applies to each parameter tensor separately, not to the global norm.
- Epochs and early stopping: 4 epochs, minibatches of 64 sequences, epochs stop when the mean of the `(rho - 1) - ln rho` KL estimator exceeds 0.045.
- Budget: 1,000 iterations x 256 episodes = 256,000 episodes (7.68 M decisions), 226 s wall time on the `flex` CPU backend (0.05 s rollout and 0.16 s update per iteration).

Model selection: every 10 iterations the greedy (mode) policy is evaluated on 512 validation episodes; the checkpoint with the lowest mean of `IS + penalty` is kept. For training seed 7 it is iteration 990, at 6.94 bps; TWAP scores 7.40 bps on the same validation seeds. Calibration, training, validation and test seeds come from disjoint streams; the training seed (`--seed`) sets parameter initialisation, training episodes and action sampling, not the validation or test episodes.

Determinism: two runs with identical arguments and backend produce identical training logs and evaluation tables, also when the run shares the machine with other trainings. The checkpoint files differ only in the randomly generated parameter identifiers stored in the burnpack header; the weights are identical.

Backends, selected at runtime with `--backend`:

| Backend | Cargo feature | Training time per iteration (rollout + update) | Notes |
|---|---|---|---|
| `flex` | default | 0.05 s + 0.16 s | pure-Rust CPU backend |
| `ndarray` | `ndarray` | 0.06 s + 0.25 s | deprecated in `burn` 0.22, kept for comparison |
| `wgpu` | `wgpu` | 0.07 s + 0.50 s (RTX 5080) | the network (26k parameters, minibatches of 64 x 30 steps) is too small to amortise kernel launches |

Cross-backend checks: checkpoints trained on `flex` and on `ndarray` give identical benchmark tables when evaluated on any of the three backends, sampled policy included. With the same seed, `flex` and `ndarray` training logs agree to the printed precision; the first difference is in the fifth decimal of the value loss at iteration 18 and comes from the floating-point summation order. Over 1,000 iterations the two trajectories separate and end in different policies (Section 9.1). `wgpu` draws the initial parameters from the GPU random generator, so the same seed gives a different initial network.

## 9. Results

Test set: 2,048 episodes on seeds disjoint from all others. Every policy is evaluated on the same seeds, so policies face the same market up to the consequences of their own orders, and differences are paired by seed. `lambda` corresponds to `kappa T = 1`; `AC kappa*T=1` is the Almgren-Chriss policy for the training objective. The PPO rows are the run with training seed 7 (`runs/ppo`); Section 9.1 gives the spread over training seeds.

| Policy | E[IS] (bps) | SE | Std[IS] | p5 | p50 | p95 | E[IS] + lambda_bps Var[IS] | IS - IS_TWAP (t) | IS - IS_AC (t) | child orders | episodes with residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TWAP (AC, kappa*T=0) | 7.584 | 0.264 | 11.934 | -13.01 | 7.73 | 26.75 | 7.639 | 0 | -0.096 (-3.0) | 30.0 | 0 |
| AC kappa*T=0.5 | 7.612 | 0.259 | 11.743 | -12.53 | 7.72 | 26.50 | 7.665 | +0.028 (+1.2) | -0.069 (-2.5) | 30.0 | 0 |
| AC kappa*T=1 | 7.681 | 0.249 | 11.267 | -11.62 | 7.83 | 25.54 | 7.729 | +0.096 (+3.0) | 0 | 30.0 | 0 |
| AC kappa*T=2 | 8.330 | 0.228 | 10.333 | -9.12 | 8.16 | 24.92 | 8.371 | +0.746 (+9.6) | +0.650 (+11.1) | 30.0 | 0 |
| AC kappa*T=4 | 12.226 | 0.232 | 10.478 | -3.19 | 11.63 | 29.19 | 12.268 | +4.641 (+25.1) | +4.545 (+26.8) | 30.0 | 0 |
| PPO-GRU, mode | 7.196 | 0.253 | 11.449 | -11.63 | 6.65 | 26.43 | 7.246 | -0.388 (-8.2) | -0.485 (-9.9) | 28.9 | 7 |
| PPO-GRU, sampled | 7.503 | 0.253 | 11.433 | -11.11 | 7.17 | 26.42 | 7.553 | -0.081 (-1.7) | -0.178 (-3.6) | 27.1 | 12 |

Mean fraction of the order held after step `k`:

| Policy | k=3 | k=6 | k=9 | k=12 | k=15 | k=18 | k=21 | k=24 | k=27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TWAP | 0.900 | 0.800 | 0.700 | 0.600 | 0.500 | 0.400 | 0.300 | 0.200 | 0.100 |
| AC kappa*T=1 | 0.873 | 0.756 | 0.645 | 0.542 | 0.443 | 0.350 | 0.259 | 0.171 | 0.085 |
| AC kappa*T=4 | 0.671 | 0.449 | 0.300 | 0.200 | 0.133 | 0.087 | 0.055 | 0.033 | 0.015 |
| PPO-GRU, mode | 0.881 | 0.781 | 0.685 | 0.586 | 0.485 | 0.383 | 0.280 | 0.177 | 0.080 |

Conditional behaviour of the greedy policy (`cargo run --release -p rl-ppo-burn --example inspect`, about 29,600 non-terminal decisions on 1,024 test episodes per run; size expressed as a multiple of the TWAP pace `x_k / (N - k)`). Ranges cover the eight `flex` runs of Section 9.1:

| Conditioning variable | lowest quintile, seed 7 | highest quintile, seed 7 | lowest quintile, 8 seeds | highest quintile, 8 seeds |
|---|---:|---:|---:|---:|
| `bid_depth_5ticks` | 0.69x | 1.65x | 0.47-0.76x | 1.30-1.65x |
| `imbalance_5ticks` | 0.59x | 1.63x | 0.44-0.71x | 1.21-1.63x |
| `mid_change_last_step` | 0.68x | 1.39x | 0.67-0.90x | 0.82-1.39x |
| `mid_drift_since_arrival` | 0.86x | 1.59x | 0.86-1.17x | 1.03-1.59x |
| `spread` (lowest: 1 tick; highest: 2 ticks or wider) | 1.19x | 0.92x | 0.97-1.19x | 0.71-0.99x |

The share of non-terminal epochs in which the policy sells nothing is 3.0% for seed 7 and ranges from 0.0% to 13.4% across seeds.

### 9.1 Variability across training seeds

Eight runs with the arguments of Section 10 and `--seed` in {1, ..., 8}, on the `flex` backend, evaluated on the same 2,048 test episodes (`cargo run --release -p rl-ppo-burn --example seeds`). The last two rows repeat seed 7 on the other backends; they are separate draws, not replications (Section 8). The last column replays each run's mean holdings path as a fixed schedule that ignores market state.

| Seed | Backend | Selected iteration | Validation objective | E[IS] | Std[IS] | IS - IS_TWAP (t) | IS - IS_AC (t) | E[IS] + lambda_bps Var[IS] | Episodes with residual | Sampled: IS - IS_TWAP (t) | Max. deviation of mean path from TWAP | Mean path as fixed schedule: IS - IS_TWAP (t) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `flex` | 999 | 6.950 | 7.224 | 11.978 | -0.360 (-8.7) | -0.457 (-9.1) | 7.279 | 6 | -0.155 (-3.4) | +0.016 | +0.042 (+1.8) |
| 2 | `flex` | 970 | 6.934 | 7.197 | 12.702 | -0.387 (-8.6) | -0.484 (-7.9) | 7.259 | 11 | -0.243 (-5.6) | +0.066 | -0.007 (-0.2) |
| 3 | `flex` | 999 | 6.768 | 7.076 | 12.395 | -0.508 (-9.0) | -0.604 (-9.0) | 7.135 | 20 | -0.288 (-5.2) | +0.044 | -0.014 (-0.5) |
| 4 | `flex` | 750 | 7.064 | 7.328 | 11.953 | -0.256 (-6.9) | -0.353 (-7.7) | 7.383 | 0 | -0.111 (-2.9) | -0.041 | +0.055 (+2.1) |
| 5 | `flex` | 910 | 6.901 | 7.193 | 12.013 | -0.391 (-8.5) | -0.488 (-8.9) | 7.248 | 15 | -0.238 (-5.4) | +0.033 | +0.017 (+0.7) |
| 6 | `flex` | 900 | 6.878 | 7.206 | 12.275 | -0.378 (-8.2) | -0.474 (-8.3) | 7.264 | 22 | -0.092 (-2.0) | +0.027 | -0.010 (-0.4) |
| 7 | `flex` | 990 | 6.941 | 7.196 | 11.449 | -0.388 (-8.2) | -0.485 (-9.9) | 7.246 | 7 | -0.081 (-1.7) | -0.023 | +0.037 (+1.5) |
| 8 | `flex` | 980 | 6.804 | 7.088 | 12.394 | -0.496 (-10.3) | -0.593 (-9.7) | 7.147 | 13 | -0.316 (-6.7) | +0.051 | +0.003 (+0.1) |
| 7 | `ndarray` | 980 | 6.660 | 7.049 | 11.810 | -0.535 (-10.9) | -0.632 (-11.3) | 7.102 | 10 | -0.360 (-7.3) | -0.005 | +0.004 (+0.2) |
| 7 | `wgpu` | 830 | 6.905 | 7.164 | 12.662 | -0.421 (-7.9) | -0.517 (-7.8) | 7.225 | 8 | -0.224 (-4.3) | +0.071 | +0.056 (+1.7) |

TWAP: E[IS] 7.584, Std[IS] 11.934, mean-variance 7.639. Over the eight `flex` seeds, `IS(PPO mode) - IS(TWAP)` has mean -0.396 bps, standard deviation across seeds 0.079 bps and range [-0.508, -0.256]; for the sampled policy the mean is -0.190 bps with range [-0.316, -0.081]. The fixed-schedule replays have mean +0.015 bps and range [-0.014, +0.055]. The Spearman rank correlation between the validation objective and the test difference is 0.83.

Findings:

1. The greedy policy beats TWAP in mean IS for every training seed, by 0.26 to 0.51 bps (3.4-6.7% of TWAP's 7.58 bps, per-run t between -6.9 and -10.3), and beats Almgren-Chriss at the training urgency by 0.35 to 0.60 bps. The standard deviation across seeds (0.079 bps) exceeds the paired standard error of a single run (about 0.05 bps), so one run understates the uncertainty of the improvement. The `ndarray` run with seed 7 (-0.535 bps) lies outside the `flex` range and was a favourable draw.
2. The gain comes from conditioning child size on the book and on recent price moves, not from a different average schedule. In all ten runs the mean holdings path deviates from TWAP by at most 0.07 of the order, and replaying it as a fixed schedule gives between -0.014 and +0.056 bps relative to TWAP (|t| <= 2.1). The direction of conditioning is common to the eight `flex` seeds: larger children into deep and bid-heavy books, smaller ones when the spread is two ticks or wider. After a drop of the mid, when part of the move is the transient impact measured in Section 7, children are smaller in 7 of 8 seeds.
3. In 6 of 8 seeds the policy holds more inventory than TWAP during the middle of the horizon. Std[IS] exceeds TWAP's in 7 of 8 seeds (up to 12.70 against 11.93 bps), and the 95th percentile of IS ranges from 26.43 to 27.98 bps against 26.75. At the training `lambda` the variance term is small (`lambda_bps Var[IS]` near 0.06 bps), and the mean-variance criterion is below TWAP for every seed (7.14 to 7.38 against 7.64 bps).
4. Deferred volume can leave a large final order. In 0 to 22 of 2,048 test episodes per run, the forced liquidation at the last step exceeded the visible bid depth and the remainder was marked at the best bid minus 50 ticks. These episodes are included in all statistics above.
5. Sampling from the policy instead of taking its mode halves the gain (mean -0.190 bps over seeds).
6. On this market Almgren-Chriss front-loading does not pay: realised mean cost grows by 0.75 bps at `kappa T = 2` and 4.64 bps at `kappa T = 4`, far above the model's prediction, while the standard deviation falls by at most 1.6 bps. At the training `lambda`, AC is 0.096 bps worse than TWAP in mean (t = 3.0) and on the mean-variance criterion.
7. Per-episode IS dispersion (about 12 bps) is dominated by the price risk of the inventory, so standard errors of mean IS are 0.23-0.28 bps. Paired differences on common seeds have standard errors near 0.05 bps; this is what resolves differences below 1 bp.

These results hold for this simulator calibration, order size, horizon and action set, and for a budget of 256,000 training episodes per run. They are not a statement about execution on real markets.

## 10. Reproduction

```sh
cargo test --workspace --release
cargo clippy --workspace --all-targets -- -D warnings

cargo run --release -p stylized-facts --bin validate -- --seeds 32 --base-seed 100 --horizon 20000 --out runs/stylized_facts.json
cargo run --release -p matching-engine --example book_throughput
cargo run --release -p market-agents --example throughput

ARGS="--iterations 1000 --envs 256 --minibatch 64 --lr 1e-3 --entropy 0.005 --val-episodes 512 --test-episodes 2048 --eval-every 10"
cargo run --release -p rl-ppo-burn --bin train -- $ARGS --seed 7 --out runs/ppo
cargo run --release -p rl-ppo-burn --bin evaluate -- --run runs/ppo
cargo run --release -p rl-ppo-burn --example inspect -- --run runs/ppo --episodes 1024

for s in 1 2 3 4 5 6 8; do
    cargo run --release -p rl-ppo-burn --bin train -- $ARGS --seed $s --out runs/seeds/seed-$s
done
cargo run --release -p rl-ppo-burn --no-default-features --features ndarray --bin train -- $ARGS --seed 7 --backend ndarray --out runs/seeds/ndarray-seed-7
cargo run --release -p rl-ppo-burn --features wgpu --bin train -- $ARGS --seed 7 --backend wgpu --out runs/seeds/wgpu-seed-7
cargo run --release -p rl-ppo-burn --example seeds -- runs/seeds/seed-1 runs/seeds/seed-2 runs/seeds/seed-3 runs/seeds/seed-4 \
    runs/seeds/seed-5 runs/seeds/seed-6 runs/ppo runs/seeds/seed-8 runs/seeds/ndarray-seed-7 runs/seeds/wgpu-seed-7

cargo run --release -p stylized-facts --example tune -- seeds=32 horizon=20000 arbitrage.rate_max=1.0
```

`train` writes `manifest.json` (configuration, calibration, selected iteration), `train.csv` (per-iteration statistics), `model.bpk`, `results.json` and `results.md` to the output directory. `--backend` selects among the compiled backends: `ndarray` needs the `ndarray` feature, `wgpu` the `wgpu` feature. `seeds` reads `manifest.json` and `results.json` from each run directory, rejects runs evaluated on a different test configuration, and replays each mean holdings path as a fixed schedule on the test episodes. `tune` accepts dotted overrides of any `MarketConfig` field.

## 11. Limitations

- Market data is delivered without latency; only order entry is delayed.
- The learning agent uses market orders only. Passive execution, order placement inside the book and cancellation strategies are not part of the action space.
- Arbitrageurs read the latent value directly. There is no learning or costly information acquisition by other agents.
- Single instrument; no auctions, hidden or iceberg orders, fees, rebates or tick-size regimes.
- The simulated time unit carries about 26 messages per second, orders of magnitude below a liquid equity. Seconds should be read as an event-time scale when comparing with empirical stylized facts. The facts hold at 10-60 s sampling; at 1 s the lag-1 autocorrelation exceeds 0.05 in one of five seed batches, and at 300 s kurtosis falls below the threshold.
- The Almgren-Chriss calibration uses single-order probes. The linear model is misspecified for this book (Section 7), which affects the baseline as much as its parameters do.
- Variability across training seeds is measured with eight seeds for a single hyperparameter configuration. The selected checkpoint lies in the last 3% of the budget in 5 of 8 runs, so the budget of 256,000 episodes may be binding; longer training was not explored.
