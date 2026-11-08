use market_agents::MarketConfig;
use stylized_facts::simulate::{mid_paths, pooled_returns};
use stylized_facts::{Criteria, aggregate, analyse, verdict};

fn paths() -> Vec<Vec<f64>> {
    let seeds: Vec<u64> = (0..32).map(|i| 7_000 + i).collect();
    mid_paths(&MarketConfig::default(), &seeds, 600.0, 20_000.0, 1.0)
}

#[test]
fn stylized_facts_emerge_from_the_book() {
    let paths = paths();
    let criteria = Criteria::default();
    for k in [10usize, 30] {
        let r = pooled_returns(&paths, k);
        let rep = analyse(&r, k as f64, 20, 10);
        let v = verdict(&rep, &criteria);
        assert!(
            v.fat_tails,
            "dt={k}s: excess kurtosis {:.2}, JB p {:.1e}",
            rep.moments.excess_kurtosis, rep.jarque_bera.p_value
        );
        assert!(
            v.no_linear_autocorrelation,
            "dt={k}s: acf(r) {:?}",
            &rep.acf_returns[..10]
        );
        assert!(
            v.volatility_clustering,
            "dt={k}s: acf(|r|) {:?}, LB p {:.1e}",
            &rep.acf_abs[..10], rep.ljung_box_abs.p_value
        );
        assert!(
            rep.ljung_box_abs.statistic > 20.0 * rep.ljung_box_returns.statistic,
            "dt={k}s: dependence must live in |r|, not in r"
        );
    }
}

#[test]
fn aggregational_gaussianity() {
    let paths = paths();
    let kurt = |k: usize| analyse(&pooled_returns(&paths, k), k as f64, 5, 5).moments.excess_kurtosis;
    let (k10, k60, k300) = (kurt(10), kurt(60), kurt(300));
    assert!(k10 > k60 && k60 > k300, "kurtosis must decay with scale: {k10:.2} {k60:.2} {k300:.2}");
}

#[test]
fn tests_reject_and_accept_on_synthetic_data() {
    let mut s = 0x1234_5678_u64;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let gauss: Vec<f64> = (0..50_000)
        .map(|_| (-2.0 * u().max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos())
        .collect();
    let rep = analyse(&gauss, 1.0, 10, 10);
    let v = verdict(&rep, &Criteria::default());
    assert!(!v.fat_tails && v.no_linear_autocorrelation && !v.volatility_clustering);
    assert!(rep.ljung_box_returns.p_value > 1e-3);

    let (omega, alpha, beta): (f64, f64, f64) = (0.05, 0.15, 0.8);
    let mut var = omega / (1.0 - alpha - beta);
    let garch: Vec<f64> = gauss
        .iter()
        .map(|z| {
            let r = var.sqrt() * z;
            var = omega + alpha * r * r + beta * var;
            r
        })
        .collect();
    let v = verdict(&analyse(&garch, 1.0, 10, 10), &Criteria::default());
    assert!(v.fat_tails && v.no_linear_autocorrelation && v.volatility_clustering);
    assert_eq!(aggregate(&[1.0, 2.0, 3.0, 4.0, 5.0], 2), vec![3.0, 7.0]);
}
