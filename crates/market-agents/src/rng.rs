use rand_core::{Rng as _, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// SplitMix64 finalizer; a bijection on `u64` used to derive decorrelated child seeds.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic seed for sub-stream `stream` of `seed`.
#[inline]
pub fn derive_seed(seed: u64, stream: u64) -> u64 {
    mix64(seed ^ mix64(stream.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}

#[derive(Clone, Debug)]
pub struct Rng {
    inner: Xoshiro256PlusPlus,
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: Xoshiro256PlusPlus::seed_from_u64(seed),
            spare: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Uniform on the open interval (0, 1) with 53-bit resolution.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    #[inline]
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    #[inline]
    pub fn exp(&mut self, rate: f64) -> f64 {
        -self.uniform().ln() / rate
    }

    #[inline]
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let f = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * f);
                return u * f;
            }
        }
    }

    #[inline]
    pub fn lognormal(&mut self, mu: f64, sigma: f64) -> f64 {
        (mu + sigma * self.normal()).exp()
    }

    #[inline]
    pub fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        (lo.ln() + self.uniform() * (hi.ln() - lo.ln())).exp()
    }
}
