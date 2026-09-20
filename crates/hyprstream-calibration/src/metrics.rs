//! Calibration and proper-scoring metrics over probability vectors.
//!
//! All functions take parallel slices: one probability distribution per observation plus
//! the realized label index. A `noul` answer is the 2-wide distribution
//! `[P(false), P(true)]`; `choice` and `score` distributions are as wide as their
//! option/level cardinality. Width may vary across observations (score levels are
//! runtime-declared), except where a class-wise metric requires a shared class axis
//! ([`sce`], [`ace`]).
//!
//! - **ECE** ([`ece`]) — top-label, over the pinned equal-mass protocol binning.
//! - **Brier** ([`brier`]) — multiclass Brier score `Σₖ (pₖ − yₖ)²`.
//! - **NLL** ([`nll`]) — negative log-likelihood of the realized label.
//! - **RPS** ([`rps`]) — ranked probability score on the cumulative distribution,
//!   normalized by `K − 1`; the proper ordinal metric for `score` (S6b2).
//! - **SCE / ACE** ([`sce`], [`ace`]) — class-wise calibration error, static
//!   (equal-width) vs adaptive (equal-mass) binning per Nixon et al. 2019; the metrics
//!   that see ordinal tail miscalibration which top-label ECE misses (ORCU,
//!   arXiv:2410.15658).
//! - **Expected value** ([`expected_value`]) — the probability-weighted score readout
//!   `Σₖ k·pₖ`, exposed so re-fits can be validated against weighted-value error
//!   (S6b2 value guardrail: temperature moves the expectation without moving the mode).

use crate::bins::{Bin, Binning};
use crate::error::MetricError;

/// Validate one observation: a non-empty normalized distribution and an in-range label.
fn check_observation(probs: &[f64], label: usize, index: usize) -> Result<(), MetricError> {
    if probs.is_empty() {
        return Err(MetricError::InvalidDistribution {
            index,
            reason: "empty distribution".to_owned(),
        });
    }
    let mut sum = 0.0;
    for &p in probs {
        if !p.is_finite() || p < 0.0 {
            return Err(MetricError::InvalidDistribution {
                index,
                reason: format!("non-finite or negative probability {p}"),
            });
        }
        sum += p;
    }
    // Consumer-side tolerance, matching the jev-1 profile decision D5.
    if (sum - 1.0).abs() > 1e-2 {
        return Err(MetricError::InvalidDistribution {
            index,
            reason: format!("probabilities sum to {sum}, outside the consumer tolerance |Σp − 1| ≤ 1e-2"),
        });
    }
    if label >= probs.len() {
        return Err(MetricError::InvalidLabel {
            index,
            label,
            width: probs.len(),
        });
    }
    Ok(())
}

/// Validate the parallel inputs and return the number of observations.
fn check_inputs(probs: &[&[f64]], labels: &[usize]) -> Result<usize, MetricError> {
    if probs.len() != labels.len() {
        return Err(MetricError::LengthMismatch {
            left_name: "probs",
            left: probs.len(),
            right_name: "labels",
            right: labels.len(),
        });
    }
    if probs.is_empty() {
        return Err(MetricError::EmptyInput("metric inputs"));
    }
    for (i, (&p, &y)) in probs.iter().zip(labels).enumerate() {
        check_observation(p, y, i)?;
    }
    Ok(probs.len())
}

/// Argmax index with earliest-index tie-breaking (jev-1 profile decision D6).
#[must_use]
pub fn argmax(probs: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &p) in probs.iter().enumerate().skip(1) {
        if p > probs[best] {
            best = i;
        }
    }
    best
}

/// Top-label confidence and correctness per observation: `(max_k p_k, argmax == label)`.
fn confidence_pairs(probs: &[&[f64]], labels: &[usize]) -> (Vec<f64>, Vec<bool>) {
    probs
        .iter()
        .zip(labels)
        .map(|(p, &y)| {
            let top = argmax(p);
            (p[top], top == y)
        })
        .unzip()
}

/// Top-label expected calibration error over `binning`.
///
/// `ECE = Σ_b (|B_b| / n) · |acc(B_b) − conf(B_b)|`.
#[must_use]
pub fn ece_from_bins(bins: &[Bin], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = n as f64;
    bins.iter()
        .map(|b| {
            #[allow(clippy::cast_precision_loss)]
            let weight = b.count as f64 / n;
            weight * (b.mean_outcome - b.mean_confidence).abs()
        })
        .sum()
}

/// Top-label ECE under the pinned protocol binning (`n_bins` equal-mass bins).
///
/// # Errors
/// Propagates validation and binning errors.
pub fn ece(probs: &[&[f64]], labels: &[usize], n_bins: usize) -> Result<f64, MetricError> {
    let n = check_inputs(probs, labels)?;
    let (confs, correct) = confidence_pairs(probs, labels);
    let binning = Binning::equal_mass(&confs, n_bins.min(n))?;
    let bins = binning.summarize(&confs, &correct)?;
    Ok(ece_from_bins(&bins, n))
}

/// Reliability curve over the top-label confidence axis.
///
/// # Errors
/// Propagates validation and binning errors.
pub fn reliability_curve(
    probs: &[&[f64]],
    labels: &[usize],
    n_bins: usize,
) -> Result<Vec<Bin>, MetricError> {
    let n = check_inputs(probs, labels)?;
    let (confs, correct) = confidence_pairs(probs, labels);
    let binning = Binning::equal_mass(&confs, n_bins.min(n))?;
    binning.summarize(&confs, &correct)
}

/// Mean multiclass Brier score `Σₖ (pₖ − yₖ)²` (lower is better; 0 is perfect).
///
/// # Errors
/// Propagates validation errors.
pub fn brier(probs: &[&[f64]], labels: &[usize]) -> Result<f64, MetricError> {
    let n = check_inputs(probs, labels)?;
    let total: f64 = probs
        .iter()
        .zip(labels)
        .map(|(p, &y)| {
            p.iter()
                .enumerate()
                .map(|(k, &pk)| {
                    let target = f64::from(k == y);
                    (pk - target).powi(2)
                })
                .sum::<f64>()
        })
        .sum();
    #[allow(clippy::cast_precision_loss)]
    Ok(total / n as f64)
}

/// Mean negative log-likelihood of the realized label.
///
/// # Errors
/// Propagates validation errors. Returns [`MetricError::InvalidDistribution`] if a
/// realized label was assigned exactly zero probability (infinite NLL).
pub fn nll(probs: &[&[f64]], labels: &[usize]) -> Result<f64, MetricError> {
    let n = check_inputs(probs, labels)?;
    let mut total = 0.0;
    for (i, (&p, &y)) in probs.iter().zip(labels).enumerate() {
        let py = p[y];
        if py <= 0.0 {
            return Err(MetricError::InvalidDistribution {
                index: i,
                reason: "realized label has zero probability (infinite NLL)".to_owned(),
            });
        }
        total -= py.ln();
    }
    #[allow(clippy::cast_precision_loss)]
    Ok(total / n as f64)
}

/// Mean ranked probability score over the cumulative distribution, normalized by
/// `K − 1` so scores are comparable across runtime-declared level cardinalities:
/// `RPS = Σₖ₌₀..K₋₂ (Fₖ − 1{y ≤ k})² / (K − 1)` (lower is better; 0 is perfect).
///
/// This is the primary proper scoring rule for the ordinal `score` primitive (S6b2).
/// For width-2 distributions (noul) it equals half the multiclass Brier score.
///
/// # Errors
/// Propagates validation errors.
pub fn rps(probs: &[&[f64]], labels: &[usize]) -> Result<f64, MetricError> {
    let n = check_inputs(probs, labels)?;
    let total: f64 = probs
        .iter()
        .zip(labels)
        .map(|(p, &y)| rps_single(p, y))
        .sum();
    #[allow(clippy::cast_precision_loss)]
    Ok(total / n as f64)
}

/// RPS of a single observation (normalized by `K − 1`; `K = 1` scores 0).
#[must_use]
pub fn rps_single(probs: &[f64], label: usize) -> f64 {
    let k = probs.len();
    if k < 2 {
        return 0.0;
    }
    let mut cdf = 0.0;
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate().take(k - 1) {
        cdf += p;
        let target = f64::from(label <= i);
        acc += (cdf - target).powi(2);
    }
    #[allow(clippy::cast_precision_loss)]
    let denom = (k - 1) as f64;
    acc / denom
}

/// Probability-weighted value of an ordinal distribution over level indices:
/// `E = Σₖ k·pₖ` (level values default to their indices per the jev-1 profile).
///
/// A convenience readout, not a trust object — the served contract is the calibrated
/// distribution plus a conformal interval (S6b2).
#[must_use]
pub fn expected_value(probs: &[f64]) -> f64 {
    probs
        .iter()
        .enumerate()
        .map(|(k, &p)| {
            #[allow(clippy::cast_precision_loss)]
            let v = k as f64;
            v * p
        })
        .sum()
}

/// Class-wise calibration error with a shared binning over each class's confidence axis.
///
/// For every class `k`, treats `p_k` as the confidence and `1{label = k}` as the outcome,
/// bins them, and accumulates the weighted per-bin gap. Returns the unweighted mean over
/// classes. Requires a shared class axis (all distributions the same width).
fn classwise_ce(
    probs: &[&[f64]],
    labels: &[usize],
    n_bins: usize,
    binning: fn(&[f64], usize) -> Result<Binning, MetricError>,
) -> Result<f64, MetricError> {
    let n = check_inputs(probs, labels)?;
    let width = probs[0].len();
    for (i, p) in probs.iter().enumerate() {
        if p.len() != width {
            return Err(MetricError::InvalidDistribution {
                index: i,
                reason: format!(
                    "class-wise metrics require a shared class axis: width {} != {}",
                    p.len(),
                    width
                ),
            });
        }
    }
    let mut total = 0.0;
    for k in 0..width {
        let confs: Vec<f64> = probs.iter().map(|p| p[k]).collect();
        let outcomes: Vec<bool> = labels.iter().map(|&y| y == k).collect();
        let b = binning(&confs, n_bins.min(n))?;
        let bins = b.summarize(&confs, &outcomes)?;
        total += ece_from_bins(&bins, n);
    }
    #[allow(clippy::cast_precision_loss)]
    Ok(total / width as f64)
}

/// Static calibration error — class-wise, equal-width bins (Nixon et al. 2019).
///
/// # Errors
/// Propagates validation and binning errors.
pub fn sce(probs: &[&[f64]], labels: &[usize], n_bins: usize) -> Result<f64, MetricError> {
    classwise_ce(probs, labels, n_bins, |confs, b| {
        let _ = confs;
        Ok(Binning::equal_width(b))
    })
}

/// Adaptive calibration error — class-wise, equal-mass bins (Nixon et al. 2019).
///
/// # Errors
/// Propagates validation and binning errors.
pub fn ace(probs: &[&[f64]], labels: &[usize], n_bins: usize) -> Result<f64, MetricError> {
    classwise_ce(probs, labels, n_bins, Binning::equal_mass)
}
