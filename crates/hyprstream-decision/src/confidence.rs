//! Confidence-from-peakedness: the normative jev-1 derivations over a raw distribution.
//!
//! Raw distributions are the primary contract; confidence is a derived statistic so that
//! ecosystem thresholds port (S6a §2.3 — the formulas are normative from the MIT adapter
//! source and reproduce the docs' non-stale numeric examples exactly; this crate's
//! `tests/jev1_compat.rs` pins those examples). Whether the upstream service computes the
//! identical statistic is a live-check item for P0.5/P0.7 (S6a A1) — it changes nothing
//! here, where confidence is computed from our own distributions.
//!
//! Nothing in this module is stored in the schema: these are pure functions over the
//! probability vectors that live in the Arrow columns.

use thiserror::Error;

use crate::spec::QuestionKind;

/// Producer-side probability-sum tolerance (D5): emitted distributions must satisfy
/// `|Σp − 1| ≤ 1e-6`. Enforced by batch construction.
pub const PRODUCER_SUM_TOLERANCE: f32 = 1e-6;

/// Consumer-side probability-sum tolerance (D5): readers must accept
/// `|Σp − 1| ≤ 1e-2` ("approximately 1" on the wire).
pub const CONSUMER_SUM_TOLERANCE: f32 = 1e-2;

/// Why a probability vector is not a valid distribution under the profile.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DistributionError {
    /// The vector is empty.
    #[error("distribution is empty")]
    Empty,
    /// A component is outside [0, 1] or not finite.
    #[error("component {index} = {value} is not a finite probability in [0, 1]")]
    OutOfRange {
        /// Offending component index.
        index: usize,
        /// Offending value.
        value: f32,
    },
    /// The sum deviates from 1 beyond the applicable tolerance.
    #[error("distribution sums to {sum}, beyond tolerance {tolerance}")]
    SumOutOfTolerance {
        /// The actual sum.
        sum: f32,
        /// The tolerance that was applied.
        tolerance: f32,
    },
}

/// Validate a probability vector against a tolerance. Every component must be a finite
/// probability in [0, 1] and the sum must be within `tolerance` of 1.
pub fn check_distribution(probabilities: &[f32], tolerance: f32) -> Result<(), DistributionError> {
    if probabilities.is_empty() {
        return Err(DistributionError::Empty);
    }
    let mut sum = 0.0f32;
    for (index, &value) in probabilities.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(DistributionError::OutOfRange { index, value });
        }
        sum += value;
    }
    if (sum - 1.0).abs() > tolerance {
        return Err(DistributionError::SumOutOfTolerance { sum, tolerance });
    }
    Ok(())
}

/// Argmax with the profile tie-break (D6): the earliest index in canonical (insertion)
/// order wins. Returns `None` on an empty slice.
pub fn argmax_index(probabilities: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in probabilities.iter().enumerate() {
        match best {
            Some((_, best_value)) if value <= best_value => {}
            _ => best = Some((index, value)),
        }
    }
    best.map(|(index, _)| index)
}

/// Choice confidence: `(p_max − 1/n) / (1 − 1/n)` — the peak probability scaled against
/// the uniform floor. A single-option distribution is certain by construction (1.0).
pub fn choice_confidence(probabilities: &[f32]) -> f32 {
    let n = probabilities.len();
    if n <= 1 {
        return 1.0;
    }
    let p_max = probabilities
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let uniform = 1.0 / n as f32;
    ((p_max - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
}

/// Score confidence: `max(0, 1 − MAD_mode / MAD_uniform)` where
/// `MAD_mode = Σᵢ pᵢ·|i − argmax|` and `MAD_uniform = (Σᵢ |i − (n−1)/2|) / n` — the
/// distribution's mean absolute distance from its mode, normalized by the uniform
/// distribution's mean absolute distance from the center. A single-level distribution is
/// certain by construction (1.0).
pub fn score_confidence(probabilities: &[f32]) -> f32 {
    let n = probabilities.len();
    if n <= 1 {
        return 1.0;
    }
    let Some(mode) = argmax_index(probabilities) else {
        return 1.0;
    };
    let mad_mode: f32 = probabilities
        .iter()
        .enumerate()
        .map(|(index, &p)| p * index.abs_diff(mode) as f32)
        .sum();
    let center = (n - 1) as f32 / 2.0;
    let mad_uniform: f32 = (0..n)
        .map(|index| (index as f32 - center).abs())
        .sum::<f32>()
        / n as f32;
    if mad_uniform == 0.0 {
        return 1.0;
    }
    (1.0 - mad_mode / mad_uniform).max(0.0)
}

/// The normative confidence for any primitive's distribution. Noul has no confidence
/// field upstream by design; as a superset convenience it is computed as the choice
/// formula over the 2-label `[P(false), P(true)]` distribution.
pub fn confidence(kind: QuestionKind, probabilities: &[f32]) -> f32 {
    match kind {
        QuestionKind::Noul | QuestionKind::Choice => choice_confidence(probabilities),
        QuestionKind::Score => score_confidence(probabilities),
        // Reserved v2 kinds carry no distribution contract in v1.
        QuestionKind::Span | QuestionKind::Derived => 0.0,
    }
}

/// The probability-weighted score value `Σᵢ i·pᵢ` ∈ [0, n−1]. A derived convenience —
/// the trust contract remains the distribution itself.
pub fn expected_score(probabilities: &[f32]) -> f32 {
    probabilities
        .iter()
        .enumerate()
        .map(|(index, &p)| index as f32 * p)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_breaks_ties_toward_earliest_index() {
        assert_eq!(argmax_index(&[0.5, 0.5]), Some(0));
        assert_eq!(argmax_index(&[0.1, 0.4, 0.4]), Some(1));
        assert_eq!(argmax_index(&[]), None);
    }

    #[test]
    fn degenerate_distributions_are_certain() {
        assert_eq!(choice_confidence(&[1.0]), 1.0);
        assert_eq!(score_confidence(&[1.0]), 1.0);
        assert_eq!(score_confidence(&[0.0, 0.0, 0.0, 1.0]), 1.0);
    }
}
