//! Error type for metric and conformal computations.

use thiserror::Error;

/// Errors returned by metric, conformal, and protocol computations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MetricError {
    /// An input slice was empty where at least one observation is required.
    #[error("empty input: {0} requires at least one observation")]
    EmptyInput(&'static str),

    /// Parallel input slices had different lengths.
    #[error("length mismatch: {left_name} has {left} rows but {right_name} has {right}")]
    LengthMismatch {
        /// Name of the first input.
        left_name: &'static str,
        /// Length of the first input.
        left: usize,
        /// Name of the second input.
        right_name: &'static str,
        /// Length of the second input.
        right: usize,
    },

    /// A probability vector failed validation.
    #[error("invalid distribution at observation {index}: {reason}")]
    InvalidDistribution {
        /// Row index of the offending observation.
        index: usize,
        /// Why the distribution is invalid.
        reason: String,
    },

    /// A label index was out of range for its distribution.
    #[error("label {label} out of range for distribution of width {width} at observation {index}")]
    InvalidLabel {
        /// Row index of the offending observation.
        index: usize,
        /// The offending label index.
        label: usize,
        /// Distribution width (number of classes/levels).
        width: usize,
    },

    /// A bin count was zero or exceeded the observation count.
    #[error("invalid bin count {bins}: need 1 <= bins <= n ({n})")]
    InvalidBinCount {
        /// Requested number of bins.
        bins: usize,
        /// Number of observations.
        n: usize,
    },

    /// A miscoverage / confidence level outside (0, 1).
    #[error("invalid level {value}: must lie strictly inside (0, 1)")]
    InvalidLevel {
        /// The offending level.
        value: f64,
    },

    /// A shift split was constructed with overlapping fit and evaluation families.
    #[error("shift split violation: families {families:?} appear on both the fit and evaluation side")]
    ShiftSplitOverlap {
        /// The overlapping family identifiers.
        families: Vec<String>,
    },
}
