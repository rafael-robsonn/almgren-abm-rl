use serde::{Deserialize, Serialize};

/// Almgren-Chriss (2000) discrete-time model for liquidating `X` shares over `N` intervals of
/// length `tau = T / N`. Price dynamics `S_k = S_{k-1} + sigma sqrt(tau) xi_k - tau g(n_k / tau)`
/// with linear permanent impact `g(v) = gamma v` and temporary impact
/// `h(v) = epsilon sgn(v) + eta v`. Units: `sigma` in $/sqrt(s), `gamma` in $/share,
/// `eta` in $ s/share, `epsilon` in $/share, `lambda` in 1/$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcParams {
    pub sigma: f64,
    pub eta: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub lambda: f64,
}

impl AcParams {
    /// `eta_tilde = eta - gamma tau / 2`; must be positive for the problem to be convex.
    pub fn eta_tilde(&self, tau: f64) -> f64 {
        self.eta - 0.5 * self.gamma * tau
    }

    /// Urgency `kappa` solving `2 (cosh(kappa tau) - 1) / tau^2 = lambda sigma^2 / eta_tilde`.
    pub fn kappa(&self, tau: f64) -> f64 {
        let et = self.eta_tilde(tau);
        assert!(et > 0.0, "eta_tilde must be positive (eta={}, gamma={}, tau={tau})", self.eta, self.gamma);
        let kt2 = self.lambda * self.sigma * self.sigma / et;
        (1.0 + 0.5 * kt2 * tau * tau).acosh() / tau
    }

    /// Risk aversion that yields urgency `kappa` for interval length `tau`.
    pub fn lambda_for_kappa(&self, kappa: f64, tau: f64) -> f64 {
        let kt2 = 2.0 * ((kappa * tau).cosh() - 1.0) / (tau * tau);
        kt2 * self.eta_tilde(tau) / (self.sigma * self.sigma)
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }
}

/// Optimal holdings `x_j = X sinh(kappa (T - t_j)) / sinh(kappa T)`, `j = 0..=N`; the
/// `lambda = 0` limit is the linear (TWAP) trajectory `x_j = X (1 - j / N)`.
pub fn holdings(x: f64, horizon: f64, n: usize, p: &AcParams) -> Vec<f64> {
    let tau = horizon / n as f64;
    let kappa = p.kappa(tau);
    (0..=n)
        .map(|j| {
            let t = j as f64 * tau;
            if kappa * horizon < 1e-9 {
                x * (1.0 - j as f64 / n as f64)
            } else {
                x * (kappa * (horizon - t)).sinh() / (kappa * horizon).sinh()
            }
        })
        .collect()
}

/// Trade list `n_k = x_{k-1} - x_k`, `k = 1..=N`.
pub fn trades(holdings: &[f64]) -> Vec<f64> {
    holdings.windows(2).map(|w| w[0] - w[1]).collect()
}

/// `E[C] = gamma X^2 / 2 + epsilon sum |n_k| + (eta_tilde / tau) sum n_k^2`.
pub fn expected_cost(holdings: &[f64], tau: f64, p: &AcParams) -> f64 {
    let x0 = holdings[0];
    let n = trades(holdings);
    0.5 * p.gamma * x0 * x0
        + p.epsilon * n.iter().map(|v| v.abs()).sum::<f64>()
        + p.eta_tilde(tau) / tau * n.iter().map(|v| v * v).sum::<f64>()
}

/// `Var[C] = sigma^2 tau sum_{k=1}^{N} x_k^2`.
pub fn variance(holdings: &[f64], tau: f64, p: &AcParams) -> f64 {
    p.sigma * p.sigma * tau * holdings[1..].iter().map(|v| v * v).sum::<f64>()
}

/// Integer child-order schedule whose cumulative sales track `X - x_k` with rounding error
/// below one share at every step.
pub fn integer_schedule(holdings: &[f64]) -> Vec<u64> {
    let x0 = holdings[0];
    let mut sold = 0u64;
    holdings[1..]
        .iter()
        .map(|x| {
            let target = (x0 - x).round().max(0.0) as u64;
            let q = target.saturating_sub(sold);
            sold += q;
            q
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(lambda: f64) -> AcParams {
        AcParams {
            sigma: 0.015,
            eta: 2e-3,
            gamma: 2e-5,
            epsilon: 0.005,
            lambda,
        }
    }

    #[test]
    fn zero_risk_aversion_is_twap() {
        let h = holdings(3_000.0, 300.0, 30, &params(0.0));
        for (j, x) in h.iter().enumerate() {
            assert!((x - 3_000.0 * (1.0 - j as f64 / 30.0)).abs() < 1e-9);
        }
        assert_eq!(integer_schedule(&h), vec![100; 30]);
    }

    #[test]
    fn trajectory_solves_discrete_euler_lagrange() {
        let p = params(1e-4);
        let tau = 10.0;
        let h = holdings(3_000.0, 300.0, 30, &p);
        let kt2 = p.lambda * p.sigma * p.sigma / p.eta_tilde(tau);
        let kappa = p.kappa(tau);
        assert!((2.0 * ((kappa * tau).cosh() - 1.0) / (tau * tau) - kt2).abs() < 1e-12);
        for j in 1..30 {
            let lhs = (h[j - 1] - 2.0 * h[j] + h[j + 1]) / (tau * tau);
            assert!((lhs - kt2 * h[j]).abs() < 1e-9 * h[0], "j={j}");
        }
        assert!((h[30]).abs() < 1e-9 && (h[0] - 3_000.0).abs() < 1e-9);
        assert!(h.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn trajectory_minimises_mean_variance_objective() {
        let p = params(5e-5);
        let tau = 10.0;
        let h = holdings(3_000.0, 300.0, 30, &p);
        let u = |h: &[f64]| expected_cost(h, tau, &p) + p.lambda * variance(h, tau, &p);
        let base = u(&h);
        for j in 1..30 {
            for d in [-5.0, 5.0] {
                let mut g = h.clone();
                g[j] += d;
                assert!(u(&g) > base, "perturbation at {j} lowered the objective");
            }
        }
        let lam = p.lambda_for_kappa(p.kappa(tau), tau);
        assert!((lam / p.lambda - 1.0).abs() < 1e-9);
    }
}
