//! Error handling for XET filter operations
//!
//! Preserves error context across FFI boundary using thread-local storage

use std::cell::RefCell;
use std::fmt;
use thiserror::Error;

thread_local! {
    static LAST_ERROR: RefCell<Option<XetError>> = const { RefCell::new(None) };
}

/// Error type for XET filter operations
#[derive(Debug, Clone, Error)]
#[error("{kind}: {message}")]
pub struct XetError {
    message: String,
    kind: XetErrorKind,
}

/// Categories of XET filter errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XetErrorKind {
    /// XET storage backend not initialized
    StorageNotInitialized,
    /// Failed to upload file to XET CAS
    UploadFailed,
    /// Failed to download file from XET CAS
    DownloadFailed,
    /// Invalid or malformed XET pointer
    InvalidPointer,
    /// I/O or filesystem error
    IoError,
    /// Runtime or initialization error
    RuntimeError,
    /// Invalid configuration
    InvalidConfig,
}

impl fmt::Display for XetErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageNotInitialized => write!(f, "StorageNotInitialized"),
            Self::UploadFailed => write!(f, "UploadFailed"),
            Self::DownloadFailed => write!(f, "DownloadFailed"),
            Self::InvalidPointer => write!(f, "InvalidPointer"),
            Self::IoError => write!(f, "IoError"),
            Self::RuntimeError => write!(f, "RuntimeError"),
            Self::InvalidConfig => write!(f, "InvalidConfig"),
        }
    }
}

impl XetError {
    /// Create a new XET error with the given kind and message
    pub fn new(kind: XetErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Get the error kind
    pub fn kind(&self) -> XetErrorKind {
        self.kind
    }

    /// Get the error message
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Convert this error to an FFI error code
    pub fn to_ffi_code(self) -> libc::c_int {
        to_ffi_error(self)
    }
}

impl From<git2::Error> for XetError {
    fn from(e: git2::Error) -> Self {
        Self::new(XetErrorKind::IoError, format!("Git error: {e}"))
    }
}

/// Set the last error for this thread
pub(crate) fn set_last_error(error: XetError) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(error);
    });
}

/// Get the last error for this thread
pub fn get_last_error() -> Option<XetError> {
    LAST_ERROR.with(|e| e.borrow().clone())
}

/// Clear the last error for this thread
pub fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

/// Convert error to FFI error code and store context
pub(crate) fn to_ffi_error(error: XetError) -> libc::c_int {
    tracing::error!("XET filter error: {}", error);
    set_last_error(error);
    -1
}

/// Result type that converts to FFI error codes
pub type Result<T> = std::result::Result<T, XetError>;

/// Extension trait for converting Results to FFI codes
pub trait FfiResult<T> {
    fn to_ffi_code(self) -> libc::c_int;
}

impl<T> FfiResult<T> for Result<T> {
    fn to_ffi_code(self) -> libc::c_int {
        match self {
            Ok(_) => 0,
            Err(e) => to_ffi_error(e),
        }
    }
}
