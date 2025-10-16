//! Custom error types for the storage layer
//!
//! This module provides type-safe error handling patterns using thiserror,
//! improving error diagnostics and enabling better error recovery strategies.

use thiserror::Error;

/// Error types specific to model references
#[derive(Debug, Error)]
pub enum ModelRefError {
    #[error("Invalid model reference format: '{0}'")]
    InvalidFormat(String),

    #[error("Model name is required")]
    MissingModelName,

    #[error("Git reference parsing failed: {0}")]
    GitRefParsing(String),

    #[error("UUID parsing failed: {0}")]
    UuidParsing(#[from] uuid::Error),
}

/// Result type using ModelRefError
pub type ModelRefResult<T> = Result<T, ModelRefError>;
