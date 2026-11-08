pub mod almgren_chriss;
pub mod calibration;
pub mod env;
pub mod evaluate;
pub mod policy;
pub mod vec_env;

pub use almgren_chriss::AcParams;
pub use calibration::{Calibration, calibrate};
pub use env::{
    EpisodeSummary, ExecutionConfig, ExecutionEnv, FEATURE_NAMES, OBS_DIM, Observation,
    StepContext, StepOutcome,
};
pub use evaluate::{
    PairedDiff, ShortfallStats, lambda_bps, paired_difference, run_episodes, run_episodes_profiled,
    shortfall_stats,
};
pub use policy::{AlmgrenChriss, Policy, Schedule, SliceActions, Twap};
pub use vec_env::VecEnv;
