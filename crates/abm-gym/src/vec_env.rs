use matching_engine::Qty;
use rayon::prelude::*;

use crate::env::{EpisodeSummary, ExecutionConfig, ExecutionEnv, Observation, StepContext, StepOutcome};

/// `N` independent execution environments stepped in lockstep on the rayon pool. Each
/// environment owns its market and random streams, so results do not depend on scheduling.
#[derive(Clone, Debug)]
pub struct VecEnv {
    envs: Vec<ExecutionEnv>,
}

impl VecEnv {
    pub fn new(config: ExecutionConfig, n: usize) -> Self {
        Self {
            envs: (0..n).map(|_| ExecutionEnv::new(config.clone())).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    pub fn config(&self) -> &ExecutionConfig {
        self.envs[0].config()
    }

    pub fn reset(&mut self, seeds: &[u64]) -> Vec<Observation> {
        assert_eq!(seeds.len(), self.envs.len());
        self.envs
            .par_iter_mut()
            .zip(seeds.par_iter())
            .map(|(e, &s)| e.reset(s))
            .collect()
    }

    /// Steps every environment that is not done; finished environments return a zero outcome
    /// flagged `done`.
    pub fn step(&mut self, quantities: &[Qty]) -> Vec<StepOutcome> {
        assert_eq!(quantities.len(), self.envs.len());
        self.envs
            .par_iter_mut()
            .zip(quantities.par_iter())
            .map(|(e, &q)| e.step(q))
            .collect()
    }

    pub fn observations(&self) -> Vec<Observation> {
        self.envs.iter().map(|e| e.observation()).collect()
    }

    pub fn contexts(&self) -> Vec<StepContext> {
        self.envs.iter().map(|e| e.context()).collect()
    }

    pub fn dones(&self) -> Vec<bool> {
        self.envs.iter().map(|e| e.is_done()).collect()
    }

    pub fn all_done(&self) -> bool {
        self.envs.iter().all(|e| e.is_done())
    }

    pub fn summaries(&self) -> Vec<EpisodeSummary> {
        self.envs.iter().map(|e| e.summary()).collect()
    }

    pub fn envs(&self) -> &[ExecutionEnv] {
        &self.envs
    }
}
