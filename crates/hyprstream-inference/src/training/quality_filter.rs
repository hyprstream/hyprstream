//! Quality Filter Module
//!
//! Provides heuristic-based output filtering using generation metrics.
//!
//! # Important Distinction
//!
//! This module is for **filtering** outputs, NOT for training signals.
//! The metrics (perplexity, entropy, repetition) measure model confidence,
//! NOT correctness. They are useful for:
//!
//! - Filtering low-quality outputs before returning to users
//! - Ranking multiple candidate outputs
//! - Triggering regeneration for concerning outputs
//!
//! They should NOT be used as training rewards because:
//! - Confidence != correctness (a model can be confident and wrong)
//! - Training on confidence creates circular reinforcement
//! - This leads to mode collapse, not improvement
//!
//! For research-valid training, see the `ttt` module (Test-Time Training).

use crate::runtime::generation_metrics::GenerationQualityMetrics;
use serde::{Deserialize, Serialize};

/// Configuration for quality filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFilterConfig {
    /// Minimum confidence threshold (0.0 to 1.0)
    /// Confidence = 1.0 / (1.0 + perplexity)
    /// Higher = stricter filtering
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,

    /// Maximum repetition ratio (0.0 to 1.0)
    /// Lower = stricter filtering for repetitive outputs
    #[serde(default = "default_max_repetition")]
    pub max_repetition: f32,

    /// Maximum entropy variance
    /// Lower = stricter filtering for inconsistent outputs
    #[serde(default = "default_max_entropy_variance")]
    pub max_entropy_variance: f32,

    /// Maximum perplexity threshold
    /// Lower = stricter filtering
    #[serde(default = "default_max_perplexity")]
    pub max_perplexity: f32,

    /// Whether filtering is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_min_confidence() -> f32 {
    0.1 // Very permissive default
}
fn default_max_repetition() -> f32 {
    0.3 // Allow up to 30% repetition
}
fn default_max_entropy_variance() -> f32 {
    2.0 // Matches is_concerning() threshold
}
fn default_max_perplexity() -> f32 {
    50.0 // Matches is_concerning() threshold
}
fn default_enabled() -> bool {
    false // Disabled by default - opt-in feature
}

impl Default for QualityFilterConfig {
    fn default() -> Self {
        Self {
            min_confidence: default_min_confidence(),
            max_repetition: default_max_repetition(),
            max_entropy_variance: default_max_entropy_variance(),
            max_perplexity: default_max_perplexity(),
            enabled: default_enabled(),
        }
    }
}

/// Quality filter for generation outputs
///
/// Filters outputs based on heuristic metrics. This is NOT a training
/// signal - it's a quality gate for user-facing outputs.
///
/// # Example
///
/// ```ignore
/// let filter = QualityFilter::new(QualityFilterConfig::default());
///
/// // After generation
/// let metrics = accumulator.finalize();
///
/// if filter.should_keep(&metrics) {
///     // Return output to user
/// } else {
///     // Regenerate or warn user
/// }
/// ```
#[derive(Debug, Clone)]
pub struct QualityFilter {
    config: QualityFilterConfig,
}

impl QualityFilter {
    /// Create a new quality filter with the given configuration
    pub fn new(config: QualityFilterConfig) -> Self {
        Self { config }
    }

    /// Create a filter with default permissive settings
    pub fn permissive() -> Self {
        Self::new(QualityFilterConfig::default())
    }

    /// Create a filter with strict settings
    pub fn strict() -> Self {
        Self::new(QualityFilterConfig {
            min_confidence: 0.3,
            max_repetition: 0.15,
            max_entropy_variance: 1.0,
            max_perplexity: 20.0,
            enabled: true,
        })
    }

    /// Check if filtering is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Determine if an output should be kept based on quality metrics
    ///
    /// Returns true if the output passes all quality checks.
    pub fn should_keep(&self, metrics: &GenerationQualityMetrics) -> bool {
        if !self.config.enabled {
            return true;
        }

        let confidence = self.compute_confidence(metrics.perplexity);

        confidence >= self.config.min_confidence
            && metrics.repetition_ratio <= self.config.max_repetition
            && metrics.entropy_variance <= self.config.max_entropy_variance
            && metrics.perplexity <= self.config.max_perplexity
    }

    /// Get detailed filter result with reasons
    pub fn evaluate(&self, metrics: &GenerationQualityMetrics) -> FilterResult {
        if !self.config.enabled {
            return FilterResult {
                passed: true,
                confidence: self.compute_confidence(metrics.perplexity),
                reasons: vec![],
            };
        }

        let confidence = self.compute_confidence(metrics.perplexity);
        let mut reasons = Vec::new();

        if confidence < self.config.min_confidence {
            reasons.push(FilterReason::LowConfidence {
                actual: confidence,
                threshold: self.config.min_confidence,
            });
        }

        if metrics.repetition_ratio > self.config.max_repetition {
            reasons.push(FilterReason::HighRepetition {
                actual: metrics.repetition_ratio,
                threshold: self.config.max_repetition,
            });
        }

        if metrics.entropy_variance > self.config.max_entropy_variance {
            reasons.push(FilterReason::HighEntropyVariance {
                actual: metrics.entropy_variance,
                threshold: self.config.max_entropy_variance,
            });
        }

        if metrics.perplexity > self.config.max_perplexity {
            reasons.push(FilterReason::HighPerplexity {
                actual: metrics.perplexity,
                threshold: self.config.max_perplexity,
            });
        }

        FilterResult {
            passed: reasons.is_empty(),
            confidence,
            reasons,
        }
    }

    /// Compute confidence score from perplexity
    ///
    /// Formula: confidence = 1.0 / (1.0 + perplexity)
    /// Range: (0.0, 1.0] where 1.0 is perfect confidence
    fn compute_confidence(&self, perplexity: f32) -> f32 {
        1.0 / (1.0 + perplexity)
    }
}

/// Result of quality filter evaluation
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Whether the output passed filtering
    pub passed: bool,

    /// Computed confidence score
    pub confidence: f32,

    /// Reasons for failing (empty if passed)
    pub reasons: Vec<FilterReason>,
}

/// Reason for filter failure
#[derive(Debug, Clone)]
pub enum FilterReason {
    LowConfidence { actual: f32, threshold: f32 },
    HighRepetition { actual: f32, threshold: f32 },
    HighEntropyVariance { actual: f32, threshold: f32 },
    HighPerplexity { actual: f32, threshold: f32 },
}

impl std::fmt::Display for FilterReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LowConfidence { actual, threshold } => {
                write!(f, "Low confidence: {actual:.3} < {threshold:.3}")
            }
            Self::HighRepetition { actual, threshold } => {
                write!(f, "High repetition: {actual:.3} > {threshold:.3}")
            }
            Self::HighEntropyVariance { actual, threshold } => {
                write!(f, "High entropy variance: {actual:.3} > {threshold:.3}")
            }
            Self::HighPerplexity { actual, threshold } => {
                write!(f, "High perplexity: {actual:.1} > {threshold:.1}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(perplexity: f32, repetition: f32, entropy_var: f32) -> GenerationQualityMetrics {
        GenerationQualityMetrics {
            perplexity,
            avg_entropy: 2.0,
            entropy_variance: entropy_var,
            repetition_ratio: repetition,
            token_count: 100,
        }
    }

    #[test]
    fn test_filter_disabled_passes_all() {
        let filter = QualityFilter::new(QualityFilterConfig {
            enabled: false,
            ..Default::default()
        });

        // Even terrible metrics should pass when disabled
        let bad_metrics = make_metrics(1000.0, 0.9, 10.0);
        assert!(filter.should_keep(&bad_metrics));
    }

    #[test]
    fn test_filter_enabled_rejects_bad() {
        let filter = QualityFilter::strict();

        // High perplexity should fail
        let high_ppl = make_metrics(100.0, 0.1, 0.5);
        assert!(!filter.should_keep(&high_ppl));

        // High repetition should fail
        let high_rep = make_metrics(5.0, 0.5, 0.5);
        assert!(!filter.should_keep(&high_rep));

        // High entropy variance should fail
        let high_var = make_metrics(5.0, 0.1, 5.0);
        assert!(!filter.should_keep(&high_var));
    }

    #[test]
    fn test_filter_accepts_good() {
        let filter = QualityFilter::strict();

        // Good metrics should pass
        // Perplexity 2.0 -> confidence = 1/(1+2) = 0.33 > 0.3 (min_confidence)
        let good = make_metrics(2.0, 0.1, 0.5);
        assert!(filter.should_keep(&good));
    }

    #[test]
    fn test_evaluate_gives_reasons() {
        let filter = QualityFilter::strict();

        let bad = make_metrics(100.0, 0.5, 5.0);
        let result = filter.evaluate(&bad);

        assert!(!result.passed);
        assert!(result.reasons.len() >= 2); // Multiple failures
    }

    #[test]
    fn test_confidence_calculation() {
        let filter = QualityFilter::permissive();

        // perplexity=0 -> confidence=1.0
        let perfect = make_metrics(0.0, 0.0, 0.0);
        let result = filter.evaluate(&perfect);
        assert!((result.confidence - 1.0).abs() < 0.001);

        // perplexity=9 -> confidence=0.1
        let low = make_metrics(9.0, 0.0, 0.0);
        let result = filter.evaluate(&low);
        assert!((result.confidence - 0.1).abs() < 0.001);
    }
}
