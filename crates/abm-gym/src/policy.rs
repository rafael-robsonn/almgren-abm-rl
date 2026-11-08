use matching_engine::Qty;
use serde::{Deserialize, Serialize};

use crate::almgren_chriss::{AcParams, holdings};
use crate::env::{Observation, StepContext};

/// Batched decision rule over a [`crate::VecEnv`]. Implementations must ignore inactive slots.
pub trait Policy {
    fn name(&self) -> String;
    fn reset(&mut self, n_envs: usize);
    fn act(&mut self, obs: &[Observation], ctx: &[StepContext], active: &[bool]) -> Vec<Qty>;
}

/// Equal slices of the remaining inventory over the remaining steps.
#[derive(Clone, Copy, Debug, Default)]
pub struct Twap;

impl Policy for Twap {
    fn name(&self) -> String {
        "TWAP".into()
    }

    fn reset(&mut self, _n_envs: usize) {}

    fn act(&mut self, _obs: &[Observation], ctx: &[StepContext], _active: &[bool]) -> Vec<Qty> {
        ctx.iter()
            .map(|c| (c.remaining as f64 / c.steps_left().max(1) as f64).round() as Qty)
            .collect()
    }
}

/// Static Almgren-Chriss trajectory executed by tracking the target holdings `x_{k+1}`.
#[derive(Clone, Debug)]
pub struct AlmgrenChriss {
    pub params: AcParams,
    targets: Vec<f64>,
}

impl AlmgrenChriss {
    pub fn new(params: AcParams, quantity: Qty, horizon: f64, n_steps: usize) -> Self {
        Self {
            params,
            targets: holdings(quantity as f64, horizon, n_steps, &params),
        }
    }

    pub fn targets(&self) -> &[f64] {
        &self.targets
    }
}

impl Policy for AlmgrenChriss {
    fn name(&self) -> String {
        format!("AC(lambda={:.2e})", self.params.lambda)
    }

    fn reset(&mut self, _n_envs: usize) {}

    fn act(&mut self, _obs: &[Observation], ctx: &[StepContext], _active: &[bool]) -> Vec<Qty> {
        ctx.iter()
            .map(|c| {
                let target = self.targets[(c.step + 1).min(self.targets.len() - 1)].round() as Qty;
                c.remaining.saturating_sub(target)
            })
            .collect()
    }
}

/// Static schedule given by target holdings `x_k` after step `k` (`x_0 = Q`, `x_N = 0`), executed
/// by tracking `x_{k+1}` from the realised inventory; independent of market state.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub label: String,
    targets: Vec<f64>,
}

impl Schedule {
    pub fn new(label: impl Into<String>, targets: Vec<f64>) -> Self {
        assert!(targets.len() >= 2, "a schedule needs at least x_0 and x_N");
        Self { label: label.into(), targets }
    }
}

impl Policy for Schedule {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn reset(&mut self, _n_envs: usize) {}

    fn act(&mut self, _obs: &[Observation], ctx: &[StepContext], _active: &[bool]) -> Vec<Qty> {
        ctx.iter()
            .map(|c| {
                let target = self.targets[(c.step + 1).min(self.targets.len() - 1)].round() as Qty;
                c.remaining.saturating_sub(target)
            })
            .collect()
    }
}

/// Discrete action set for learning agents: action `a` sells
/// `round(multipliers[a] * remaining / steps_left)` shares, so `multiplier = 1` reproduces TWAP
/// pacing on the residual schedule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SliceActions {
    pub multipliers: Vec<f64>,
}

impl Default for SliceActions {
    fn default() -> Self {
        Self {
            multipliers: vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        }
    }
}

impl SliceActions {
    pub fn len(&self) -> usize {
        self.multipliers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.multipliers.is_empty()
    }

    pub fn quantity(&self, action: usize, ctx: &StepContext) -> Qty {
        let pace = ctx.remaining as f64 / ctx.steps_left().max(1) as f64;
        ((self.multipliers[action] * pace).round() as Qty).min(ctx.remaining)
    }
}
