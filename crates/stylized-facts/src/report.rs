use serde::{Deserialize, Serialize};

use crate::stats::{Moments, TestResult, acf, hill_tail_index, jarque_bera, ljung_box, moments};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleReport {
    pub interval_secs: f64,
    pub n: usize,
    pub zero_fraction: f64,
    pub moments: Moments,
    pub jarque_bera: TestResult,
    pub hill_alpha: f64,
    pub acf_returns: Vec<f64>,
    pub acf_abs: Vec<f64>,
    pub ljung_box_returns: TestResult,
    pub ljung_box_abs: TestResult,
    /// Two-sided 95% band `1.96 / sqrt(n)` for a single autocorrelation under i.i.d. returns.
    pub band: f64,
}

pub fn analyse(returns: &[f64], interval_secs: f64, max_lag: usize, lb_lags: usize) -> ScaleReport {
    let abs: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
    let n = returns.len();
    ScaleReport {
        interval_secs,
        n,
        zero_fraction: returns.iter().filter(|r| **r == 0.0).count() as f64 / n as f64,
        moments: moments(returns),
        jarque_bera: jarque_bera(returns),
        hill_alpha: hill_tail_index(returns, 0.025),
        acf_returns: acf(returns, max_lag),
        acf_abs: acf(&abs, max_lag),
        ljung_box_returns: ljung_box(returns, lb_lags),
        ljung_box_abs: ljung_box(&abs, lb_lags),
        band: 1.96 / (n as f64).sqrt(),
    }
}

/// Acceptance thresholds. Linear dependence is judged by economic size (`|rho_k| <
/// max_abs_acf_returns` for `k <= lags`) because with `n ~ 10^5` microstructure-induced
/// autocorrelations of order `10^-2` are statistically but not economically significant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Criteria {
    pub lags: usize,
    pub min_excess_kurtosis: f64,
    pub max_jarque_bera_p: f64,
    pub max_abs_acf_returns: f64,
    pub min_acf_abs: f64,
    pub max_ljung_box_abs_p: f64,
}

impl Default for Criteria {
    fn default() -> Self {
        Self {
            lags: 10,
            min_excess_kurtosis: 1.0,
            max_jarque_bera_p: 1e-6,
            max_abs_acf_returns: 0.05,
            min_acf_abs: 0.05,
            max_ljung_box_abs_p: 1e-6,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub fat_tails: bool,
    pub no_linear_autocorrelation: bool,
    pub volatility_clustering: bool,
}

impl Verdict {
    pub fn all(&self) -> bool {
        self.fat_tails && self.no_linear_autocorrelation && self.volatility_clustering
    }
}

pub fn verdict(r: &ScaleReport, c: &Criteria) -> Verdict {
    let lags = c.lags.min(r.acf_returns.len());
    let fat_tails = r.moments.excess_kurtosis > c.min_excess_kurtosis
        && r.jarque_bera.p_value < c.max_jarque_bera_p;
    let no_linear_autocorrelation = r.acf_returns[..lags]
        .iter()
        .all(|rho| rho.abs() < c.max_abs_acf_returns);
    let floor = c.min_acf_abs.max(r.band);
    let volatility_clustering = r.acf_abs[..lags].iter().all(|rho| *rho > floor)
        && r.ljung_box_abs.p_value < c.max_ljung_box_abs_p;
    Verdict {
        fat_tails,
        no_linear_autocorrelation,
        volatility_clustering,
    }
}
