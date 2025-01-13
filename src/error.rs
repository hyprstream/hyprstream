//! Error types for the Hyprstream service.

use std::error::Error as StdError;
use std::fmt;
use std::result;
use tonic::Status;

/// A specialized Result type for Hyprstream operations.
pub type Result<T> = result::Result<T, Error>;

/// The error type for Hyprstream operations.
#[derive(Debug)]
pub enum Error {
    /// Arrow-related errors
    Arrow(String),
    /// Storage backend errors
    Storage(String),
    /// Configuration errors
    Config(String),
    /// I/O errors
    Io(std::io::Error),
    /// Serialization/deserialization errors
    Serialization(String),
    /// Validation errors
    Validation(String),
    /// Runtime errors
    Runtime(String),
    /// Internal errors
    Internal(String),
    /// Invalid data errors
    InvalidData(String),
    /// Device not available errors
    DeviceNotAvailable(&'static str),
    /// Unsupported backend errors
    UnsupportedBackend(&'static str),
    Metrics(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Arrow(msg) => write!(f, "Arrow error: {}", msg),
            Error::Metrics(msg) => write!(f, "Metrics error: {}", msg),
            Error::Storage(msg) => write!(f, "Storage error: {}", msg),
            Error::Config(msg) => write!(f, "Configuration error: {}", msg),
            Error::Io(err) => write!(f, "I/O error: {}", err),
            Error::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Error::Validation(msg) => write!(f, "Validation error: {}", msg),
            Error::Runtime(msg) => write!(f, "Runtime error: {}", msg),
            Error::Internal(msg) => write!(f, "Internal error: {}", msg),
            Error::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            Error::DeviceNotAvailable(msg) => write!(f, "Device not available: {}", msg),
            Error::UnsupportedBackend(msg) => write!(f, "Unsupported backend: {}", msg),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<arrow::error::ArrowError> for Error {
    fn from(err: arrow::error::ArrowError) -> Self {
        Error::Arrow(err.to_string())
    }
}

impl From<config::ConfigError> for Error {
    fn from(err: config::ConfigError) -> Self {
        Error::Config(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<Error> for Status {
    fn from(err: Error) -> Self {
        match err {
            Error::Arrow(msg) => Status::internal(format!("Arrow error: {}", msg)),
            Error::Storage(msg) => Status::internal(format!("Storage error: {}", msg)),
            Error::Metrics(msg) => Status::internal(format!("Metrics error: {}", msg)),
            Error::Config(msg) => Status::failed_precondition(format!("Config error: {}", msg)),
            Error::Io(err) => Status::internal(format!("I/O error: {}", err)),
            Error::Serialization(msg) => Status::internal(format!("Serialization error: {}", msg)),
            Error::Validation(msg) => Status::invalid_argument(msg),
            Error::Runtime(msg) => Status::internal(format!("Runtime error: {}", msg)),
            Error::Internal(msg) => Status::internal(format!("Internal error: {}", msg)),
            Error::InvalidData(msg) => Status::invalid_argument(msg),
            Error::DeviceNotAvailable(msg) => Status::failed_precondition(msg),
            Error::UnsupportedBackend(msg) => Status::failed_precondition(msg),
        }
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Error::Runtime(status.message().to_string())
    }
}