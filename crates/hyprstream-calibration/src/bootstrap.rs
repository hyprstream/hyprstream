//! Deterministic bootstrap confidence intervals for gate statistics.
//!
//! The gate protocol pins **percentile bootstrap CIs at 95% over 1000 resamples**
//! ([`crate::protocol::GATE_BOOTSTRAP_RESAMPLES`], [`crate::protocol::GATE_CI_LEVEL`]).
//! Resampling is seeded and deterministic — a gate report must be reproducible from its
//! recorded seed, so this module carries its own small generator instead of depending on
//! an RNG crate. `SplitMix64` is a fast counter-based generator, adequate for index
//! resampling; it is not cryptographic and is never used for secrets.

use crate::error::MetricError;

/// A percentile bootstrap confidence interval.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConfidenceInterval {
    /// Nominal coverage level (e.g. 0.95).
    pub level: f64,
    /// Lower percentile bound.
    pub low: f64,
    /// Upper percentile bound — gate comparisons that pin a bound condition on this.
    pub high: f64,
}

/// SplitMix64 pseudo-random generator (deterministic, non-cryptographic).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next `u64` in the stream.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        #[allow(clippy::cast_possible_truncation)]
        let n64 = n as u64;
        (self.next_u64() % n64) as usize
    }
}

/// Percentile bootstrap CI for `statistic` over `values`.
///
/// Draws `resamples` resamples of size `values.len()` with replacement using
/// `SplitMix64::new(seed)`, evaluates the statistic on each, and returns the observed
/// statistic together with the `(1 − level)/2` and `1 − (1 − level)/2` percentiles of
/// the resample distribution.
///
/// # Errors
/// [`MetricError::EmptyInput`] on empty values; [`MetricError::InvalidLevel`] for a
/// level outside `(0, 1)`; [`MetricError::InvalidBinCount`] if `resamples` is zero.
pub fn bootstrap_ci(
    values: &[f64],
    statistic: impl Fn(&[f64]) -> f64,
    resamples: usize,
    level: f64,
    seed: u64,
) -> Result<(f64, ConfidenceInterval), MetricError> {
    if values.is_empty() {
        return Err(MetricError::EmptyInput("bootstrap"));
    }
    if !(0.0..1.0).contains(&level) {
        return Err(MetricError::InvalidLevel { value: level });
    }
    if resamples == 0 {
        return Err(MetricError::InvalidBinCount {
            bins: resamples,
            n: values.len(),
        });
    }
    let observed = statistic(values);
    let mut rng = SplitMix64::new(seed);
    let mut dist = Vec::with_capacity(resamples);
    let mut draw = Vec::with_capacity(values.len());
    for _ in 0..resamples {
        draw.clear();
        for _ in 0..values.len() {
            draw.push(values[rng.below(values.len())]);
        }
        dist.push(statistic(&draw));
    }
    dist.sort_by(f64::total_cmp);
    let percentile = |q: f64| -> f64 {
        #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
        let idx = {
            let pos = q * (dist.len() - 1) as f64;
            pos.round().clamp(0.0, (dist.len() - 1) as f64) as usize
        };
        dist[idx]
    };
    let tail = (1.0 - level) / 2.0;
    Ok((
        observed,
        ConfidenceInterval {
            level,
            low: percentile(tail),
            high: percentile(1.0 - tail),
        },
    ))
}
