use std::str::FromStr;

use abm_gym::{Calibration, ExecutionConfig, SliceActions};
use market_agents::derive_seed;
use serde::{Deserialize, Serialize};

use crate::model::ModelConfig;
use crate::ppo::PpoConfig;

pub fn arg<T: FromStr>(args: &[String], key: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == key)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

/// Seed streams for calibration, model selection and the final test are disjoint by
/// construction (distinct stream tags under the same mixing function) and independent of the
/// training seed stream used by [`crate::Trainer`].
pub fn calibration_seeds(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| derive_seed(0xCA11_B8A7, i)).collect()
}

pub fn validation_seeds(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| derive_seed(0x0A11_DA7E, i)).collect()
}

pub fn test_seeds(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| derive_seed(0x7E57_5EED, i)).collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub backend: String,
    pub kappa_t: f64,
    pub lambda: f64,
    pub ppo: PpoConfig,
    pub exec: ExecutionConfig,
    pub actions: SliceActions,
    pub model: ModelConfig,
    pub calibration: Calibration,
    pub validation_episodes: usize,
    pub test_episodes: usize,
    pub best_iteration: Option<usize>,
    pub best_validation_objective_bps: Option<f64>,
    pub train_wall_secs: f64,
}
