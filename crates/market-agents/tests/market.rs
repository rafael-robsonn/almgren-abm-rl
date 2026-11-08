use market_agents::{Fundamental, Market, MarketConfig, OuParams};
use matching_engine::{LatencyModel, Report, Request, Side, secs};
use rayon::prelude::*;

fn trace(m: &mut Market, until: f64) -> Vec<(Option<f64>, u64)> {
    (1..=(until as usize / 10))
        .map(|i| {
            m.run_until(secs(i as f64 * 10.0));
            (m.mid(), m.messages())
        })
        .collect()
}

#[test]
fn a_run_is_a_function_of_config_and_seed() {
    let cfg = MarketConfig::default();
    let a = trace(&mut Market::new(cfg.clone(), 42), 600.0);
    let b = trace(&mut Market::new(cfg.clone(), 42), 600.0);
    let c = trace(&mut Market::new(cfg, 43), 600.0);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn cloning_yields_common_random_numbers() {
    let mut m = Market::new(MarketConfig::default(), 5);
    m.run_until(secs(300.0));
    let mut twin = m.clone();
    assert_eq!(trace(&mut m, 900.0), trace(&mut twin, 900.0));
}

#[test]
fn trades_conserve_inventory_and_cash() {
    let mut m = Market::new(MarketConfig::default(), 9);
    m.run_until(secs(3_000.0));
    let inv: i64 = m.agents().iter().map(|a| a.account().inventory).sum();
    let cash: f64 = m.agents().iter().map(|a| a.account().cash).sum();
    let traded: u64 = m.agents().iter().map(|a| a.account().traded).sum();
    assert_eq!(inv, 0);
    assert_eq!(cash, 0.0);
    assert_eq!(traded, 2 * m.exchange().stats().volume);
    assert!(m.exchange().stats().volume > 0);
    m.book().check_invariants().unwrap();
}

#[test]
fn external_session_receives_its_reports() {
    let mut m = Market::new(MarketConfig::default(), 11);
    m.run_until(secs(120.0));
    let me = m.register_external(Some(LatencyModel::new(1_000, 0)));
    let bid = m.book().best_bid().unwrap();
    let order = m.submit(me, Request::Market { side: Side::Sell, qty: 40 });
    m.run_until(secs(121.0));
    let mut reports = Vec::new();
    m.drain_external(&mut reports);
    assert!(reports.iter().all(|(a, r)| *a == me && r.order() == order));
    let filled: u64 = reports
        .iter()
        .map(|(_, r)| match r {
            Report::Fill { qty, price, .. } => {
                assert!(*price <= bid);
                *qty
            }
            Report::Expired { .. } => 0,
            other => panic!("unexpected report {other:?}"),
        })
        .sum();
    assert_eq!(filled, 40);
}

#[test]
fn fundamental_transition_has_ou_moments() {
    let p = OuParams {
        mu: 100.0,
        kappa: 0.05,
        sigma: 2.0,
        dt: 0.1,
    };
    let t = 20.0;
    let x0 = 130.0;
    let xs: Vec<f64> = (0..4_000u64)
        .into_par_iter()
        .map(|s| Fundamental::new(p, x0, s).value_at(secs(t)))
        .collect();
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let m_th = p.mu + (x0 - p.mu) * (-p.kappa * t).exp();
    let v_th = p.sigma * p.sigma * (1.0 - (-2.0 * p.kappa * t).exp()) / (2.0 * p.kappa);
    assert!((mean - m_th).abs() < 4.0 * (v_th / n).sqrt(), "mean {mean} vs {m_th}");
    assert!((var / v_th - 1.0).abs() < 4.0 * (2.0 / n).sqrt(), "var {var} vs {v_th}");
}
