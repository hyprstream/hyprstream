//! Cross-question / joint coverage.
//!
//! Conformal coverage guarantees are **marginal**: a 90% prediction set per question does
//! not imply 90% of batch rows have every question covered. A decision batch answers many
//! questions against the same state, and downstream consumers (confidence-filtered joins,
//! SQL composition) compose the answers — so the joint number is reported separately from
//! the marginals, alongside the product-of-marginals reference that independence would
//! imply. This module owns that report; it takes per-row coverage indicators produced by
//! any conformal method.

use serde::{Deserialize, Serialize};

use crate::error::MetricError;

/// Joint-coverage report for a batch of multi-question rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointCoverage {
    /// Number of rows.
    pub n: usize,
    /// Number of questions per row.
    pub n_questions: usize,
    /// Per-question marginal coverage.
    pub marginals: Vec<f64>,
    /// Fraction of rows in which **every** question's prediction set covered the true
    /// label.
    pub joint: f64,
    /// Product of the marginals — the joint coverage that question independence would
    /// imply. The gap `joint − independence_reference` measures error correlation
    /// across questions on the same state.
    pub independence_reference: f64,
}

/// Compute joint and marginal coverage from a row-major coverage matrix:
/// `covered[i][q]` is true when row `i`'s prediction set for question `q` covered the
/// realized label.
///
/// # Errors
/// [`MetricError::EmptyInput`] on zero rows or zero questions;
/// [`MetricError::LengthMismatch`] on a ragged matrix.
pub fn joint_coverage(covered: &[Vec<bool>]) -> Result<JointCoverage, MetricError> {
    if covered.is_empty() {
        return Err(MetricError::EmptyInput("joint coverage"));
    }
    let n_questions = covered[0].len();
    if n_questions == 0 {
        return Err(MetricError::EmptyInput("joint coverage (questions)"));
    }
    for row in covered.iter() {
        if row.len() != n_questions {
            return Err(MetricError::LengthMismatch {
                left_name: "covered[0]",
                left: n_questions,
                right_name: "covered[i]",
                right: row.len(),
            });
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let n = covered.len() as f64;
    let mut hits = vec![0usize; n_questions];
    let mut joint_hits = 0usize;
    for row in covered {
        let mut all = true;
        for (q, &c) in row.iter().enumerate() {
            hits[q] += usize::from(c);
            all &= c;
        }
        joint_hits += usize::from(all);
    }
    #[allow(clippy::cast_precision_loss)]
    let marginals: Vec<f64> = hits.iter().map(|&h| h as f64 / n).collect();
    let independence_reference = marginals.iter().product();
    #[allow(clippy::cast_precision_loss)]
    Ok(JointCoverage {
        n: covered.len(),
        n_questions,
        marginals,
        joint: joint_hits as f64 / n,
        independence_reference,
    })
}
