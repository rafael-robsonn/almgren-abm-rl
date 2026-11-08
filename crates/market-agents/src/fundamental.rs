use matching_engine::{Timestamp, to_secs};
use serde::{Deserialize, Serialize};

use crate::rng::Rng;

/// Ornstein-Uhlenbeck fundamental `dV = kappa (mu - V) dt + sigma dW` in ticks.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OuParams {
    pub mu: f64,
    pub kappa: f64,
    pub sigma: f64,
    pub dt: f64,
}

impl OuParams {
    /// Stationary standard deviation `sigma / sqrt(2 kappa)`; infinite for `kappa = 0`.
    pub fn stationary_std(&self) -> f64 {
        if self.kappa > 0.0 {
            self.sigma / (2.0 * self.kappa).sqrt()
        } else {
            f64::INFINITY
        }
    }
}

/// Latent value process sampled on a fixed grid of step `dt` with the exact Gaussian transition
/// `V_{t+dt} | V_t ~ N(mu + (V_t - mu) e^{-kappa dt}, sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa))`.
/// The path is a function of the seed only, independent of when it is queried.
#[derive(Clone, Debug)]
pub struct Fundamental {
    params: OuParams,
    value: f64,
    grid_t: Timestamp,
    step: Timestamp,
    decay: f64,
    shock_sd: f64,
    rng: Rng,
}

impl Fundamental {
    pub fn new(params: OuParams, initial: f64, seed: u64) -> Self {
        let decay = (-params.kappa * params.dt).exp();
        let shock_sd = if params.kappa > 0.0 {
            params.sigma * ((1.0 - (-2.0 * params.kappa * params.dt).exp()) / (2.0 * params.kappa)).sqrt()
        } else {
            params.sigma * params.dt.sqrt()
        };
        Self {
            params,
            value: initial,
            grid_t: 0,
            step: matching_engine::secs(params.dt).max(1),
            decay,
            shock_sd,
            rng: Rng::new(seed),
        }
    }

    pub fn params(&self) -> &OuParams {
        &self.params
    }

    /// Value of the piecewise-constant grid path at time `t` (monotone queries only).
    pub fn value_at(&mut self, t: Timestamp) -> f64 {
        while self.grid_t + self.step <= t {
            self.grid_t += self.step;
            self.value = self.params.mu
                + (self.value - self.params.mu) * self.decay
                + self.shock_sd * self.rng.normal();
        }
        self.value
    }

    /// Counterfactual intervention: shifts the current latent value by `delta` ticks.
    pub fn shift(&mut self, delta: f64) {
        self.value += delta;
    }

    pub fn last_value(&self) -> f64 {
        self.value
    }

    pub fn elapsed_secs(&self) -> f64 {
        to_secs(self.grid_t)
    }
}
