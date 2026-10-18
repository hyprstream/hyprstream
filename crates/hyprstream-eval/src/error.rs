//! Harness error type.

use hyprstream_calibration::MetricError;
use hyprstream_decision::arrow::{ArrowSchemaError, BatchError};

/// Everything that can go wrong in a harness run, scoring pass, or persistence
/// call. Subjects surface their own failures through the `Subject` variants.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// A subject failed to produce an answer row.
    #[error("subject `{model}` failed on item `{item_id}`: {message}")]
    Subject {
        /// Subject's reported model id.
        model: String,
        /// The item (or row) being answered.
        item_id: String,
        /// What went wrong.
        message: String,
    },
    /// The HTTP subject got a transport or wire-level failure.
    #[error("http subject: {0}")]
    Http(String),
    /// A subject's answer could not be reconciled with the question set
    /// (missing question, kind mismatch, unknown label, bad distribution).
    #[error("invalid answer for question `{question_id}`: {message}")]
    InvalidAnswer {
        /// The question the answer was for.
        question_id: String,
        /// What was wrong with it.
        message: String,
    },
    /// Arrow schema construction failed.
    #[error(transparent)]
    ArrowSchema(#[from] ArrowSchemaError),
    /// Arrow batch assembly failed.
    #[error(transparent)]
    Batch(#[from] BatchError),
    /// Calibration metric computation failed.
    #[error(transparent)]
    Metric(#[from] MetricError),
    /// Persistence through the P0.3 metrics API failed.
    #[error("persistence: {0}")]
    Persist(String),
    /// An eval set/row violated a harness-level rule.
    #[error("invalid eval input: {0}")]
    InvalidInput(String),
}

impl From<tonic::Status> for EvalError {
    fn from(status: tonic::Status) -> Self {
        Self::Persist(format!("{}: {}", status.code(), status.message()))
    }
}
