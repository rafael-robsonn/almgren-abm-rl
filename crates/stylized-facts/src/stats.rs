use serde::{Deserialize, Serialize};

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64
}

pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
}

/// Sums of non-overlapping blocks of `k` consecutive observations (returns at scale `k`).
pub fn aggregate(x: &[f64], k: usize) -> Vec<f64> {
    x.chunks_exact(k.max(1)).map(|c| c.iter().sum()).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Moments {
    pub n: usize,
    pub mean: f64,
    pub std: f64,
    pub skewness: f64,
    pub excess_kurtosis: f64,
}

/// Sample moments with population normalisation; excess kurtosis is `m4 / m2^2 - 3`
/// (zero for a Gaussian).
pub fn moments(x: &[f64]) -> Moments {
    let n = x.len();
    let m = mean(x);
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for v in x {
        let d = v - m;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    let nf = n as f64;
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    Moments {
        n,
        mean: m,
        std: m2.sqrt(),
        skewness: m3 / m2.powf(1.5),
        excess_kurtosis: m4 / (m2 * m2) - 3.0,
    }
}

/// Biased sample autocorrelation `rho_k = sum_{t} (x_t - m)(x_{t+k} - m) / sum_t (x_t - m)^2`
/// for `k = 1..=max_lag`; element `k - 1` holds lag `k`.
pub fn acf(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let m = mean(x);
    let d: Vec<f64> = x.iter().map(|v| v - m).collect();
    let denom: f64 = d.iter().map(|v| v * v).sum();
    (1..=max_lag)
        .map(|k| {
            if k >= n || denom == 0.0 {
                return 0.0;
            }
            d[..n - k].iter().zip(&d[k..]).map(|(a, b)| a * b).sum::<f64>() / denom
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestResult {
    pub statistic: f64,
    pub dof: f64,
    pub p_value: f64,
}

/// Ljung-Box portmanteau `Q = n (n + 2) sum_{k=1}^{h} rho_k^2 / (n - k)`, asymptotically
/// `chi^2(h)` under the null of no autocorrelation up to lag `h`.
pub fn ljung_box(x: &[f64], lags: usize) -> TestResult {
    let n = x.len() as f64;
    let r = acf(x, lags);
    let q = n * (n + 2.0)
        * r.iter()
            .enumerate()
            .map(|(i, rho)| rho * rho / (n - (i + 1) as f64))
            .sum::<f64>();
    TestResult {
        statistic: q,
        dof: lags as f64,
        p_value: chi2_sf(q, lags as f64),
    }
}

/// Jarque-Bera normality test `JB = n/6 (S^2 + K^2/4)`, asymptotically `chi^2(2)`.
pub fn jarque_bera(x: &[f64]) -> TestResult {
    let m = moments(x);
    let jb = m.n as f64 / 6.0 * (m.skewness.powi(2) + m.excess_kurtosis.powi(2) / 4.0);
    TestResult {
        statistic: jb,
        dof: 2.0,
        p_value: chi2_sf(jb, 2.0),
    }
}

/// Hill estimator of the tail index of `|x|` from the largest `ceil(tail_fraction * n)` order
/// statistics: `alpha = k / sum_{i<k} ln(|x|_(i) / |x|_(k))`. Gaussian data give large values
/// that grow with `n`; power-law tails `P(|X| > u) ~ u^-alpha` give a stable `alpha`.
pub fn hill_tail_index(x: &[f64], tail_fraction: f64) -> f64 {
    let mut a: Vec<f64> = x.iter().map(|v| v.abs()).filter(|v| *v > 0.0).collect();
    if a.len() < 10 {
        return f64::NAN;
    }
    a.sort_by(|p, q| q.partial_cmp(p).expect("finite returns"));
    let k = ((tail_fraction * a.len() as f64).ceil() as usize).clamp(2, a.len() - 1);
    let threshold = a[k];
    let s: f64 = a[..k].iter().map(|v| (v / threshold).ln()).sum();
    k as f64 / s
}

pub fn chi2_sf(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(0.5 * k, 0.5 * x)
}

fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularised upper incomplete gamma `Q(a, x)`.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut sum = 1.0 / a;
    let mut term = sum;
    let mut ap = a;
    for _ in 0..10_000 {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    (sum.ln() - x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_q_cf(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..10_000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi_square_tail_matches_tables() {
        assert!((chi2_sf(3.841_458_8, 1.0) - 0.05).abs() < 1e-6);
        assert!((chi2_sf(18.307_038, 10.0) - 0.05).abs() < 1e-6);
        assert!((chi2_sf(2.0, 2.0) - (-1.0f64).exp()).abs() < 1e-12);
        assert!((chi2_sf(100.0, 50.0) - 3.454_931_382_984_864e-5).abs() < 1e-12);
        assert!((chi2_sf(18.0, 10.0) - 0.054_963_641_495_104_91).abs() < 1e-12);
    }

    #[test]
    fn autocorrelation_of_known_sequences() {
        let alt: Vec<f64> = (0..1000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let r = acf(&alt, 2);
        assert!((r[0] + 0.999).abs() < 1e-3 && (r[1] - 0.998).abs() < 1e-3);
        let m = moments(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert!(m.mean.abs() < 1e-15 && m.skewness.abs() < 1e-15);
        assert!((m.excess_kurtosis - (6.8 / 4.0 - 3.0)).abs() < 1e-12);
    }

    #[test]
    fn hill_recovers_pareto_index() {
        let mut s = 99u64;
        let x: Vec<f64> = (0..200_000)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let u = ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                u.powf(-1.0 / 3.0)
            })
            .collect();
        let a = hill_tail_index(&x, 0.05);
        assert!((a - 3.0).abs() < 0.15, "hill {a}");
    }
}
