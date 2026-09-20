//! Selective prediction: risk–coverage analysis over confidence-thresholded answers.
//!
//! Abstention is a first-class answer value in the jev-1 profile (`null` = abstained),
//! and selective accuracy is the metric that prices it: accuracy among the answers a
//! threshold policy would have *kept*, as a function of coverage. The curve is computed
//! exactly (every distinct threshold), not sampled.

use serde::{Deserialize, Serialize};

use crate::error::MetricError;

/// One point on the risk–coverage curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiskCoverage {
    /// Fraction of observations kept (confidence ≥ threshold at this point).
    pub coverage: f64,
    /// Accuracy among the kept observations (1 − selective risk).
    pub accuracy: f64,
    /// The confidence threshold realizing this operating point.
    pub threshold: f64,
}

/// Exact risk–coverage curve: sort by confidence descending; at each distinct threshold,
/// coverage = kept fraction and accuracy = mean correctness among kept.
///
/// Points are returned from lowest coverage (most selective) to highest (keep all).
///
/// # Errors
/// [`MetricError::LengthMismatch`] / [`MetricError::EmptyInput`] on malformed inputs.
pub fn risk_coverage_curve(
    confidences: &[f64],
    correct: &[bool],
) -> Result<Vec<RiskCoverage>, MetricError> {
    if confidences.len() != correct.len() {
        return Err(MetricError::LengthMismatch {
            left_name: "confidences",
            left: confidences.len(),
            right_name: "correct",
            right: correct.len(),
        });
    }
    if confidences.is_empty() {
        return Err(MetricError::EmptyInput("risk-coverage curve"));
    }
    let mut order: Vec<usize> = (0..confidences.len()).collect();
    order.sort_by(|&a, &b| f64::total_cmp(&confidences[b], &confidences[a]));
    #[allow(clippy::cast_precision_loss)]
    let n = confidences.len() as f64;
    let mut kept_correct = 0usize;
    let mut curve = Vec::with_capacity(confidences.len());
    for (rank, &i) in order.iter().enumerate() {
        kept_correct += usize::from(correct[i]);
        // Emit one point per distinct threshold value.
        let is_last = rank + 1 == order.len();
        if !is_last && confidences[order[rank + 1]] == confidences[i] {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let kept = (rank + 1) as f64;
        curve.push(RiskCoverage {
            coverage: kept / n,
            #[allow(clippy::cast_precision_loss)]
            accuracy: kept_correct as f64 / kept,
            threshold: confidences[i],
        });
    }
    Ok(curve)
}

/// Accuracy at the largest coverage `<= target_coverage` (step semantics: a threshold
/// policy realizes only the curve's own points). Returns `None` if the target is below
/// the smallest realized coverage.
#[must_use]
pub fn accuracy_at_coverage(curve: &[RiskCoverage], target_coverage: f64) -> Option<f64> {
    curve
        .iter()
        .filter(|p| p.coverage <= target_coverage)
        .max_by(|a, b| f64::total_cmp(&a.coverage, &b.coverage))
        .map(|p| p.accuracy)
}

/// Area under the risk–coverage curve (coverage on x, accuracy on y), trapezoidal over
/// the realized points with the left edge extended to coverage 0 at the first point's
/// accuracy. Higher is better; 1.0 is an oracle ranker.
#[must_use]
pub fn selective_auc(curve: &[RiskCoverage]) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    let mut area = curve[0].accuracy * curve[0].coverage;
    for w in curve.windows(2) {
        let (a, b) = (w[0], w[1]);
        area += (b.coverage - a.coverage) * (a.accuracy + b.accuracy) / 2.0;
    }
    area
}
