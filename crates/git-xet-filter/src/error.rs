//! Error handling for XET filter operations
//!
//! Preserves error context across FFI boundary using thread-local storage

use std::cell::RefCell;
use std::fmt;

thread_local! {
    static LAST_ERROR: RefCell<Option<XetError>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone)]
pub struct XetError {
    pub message: String,
    pub kind: XetErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XetErrorKind {
    StorageNotInitialized,
    UploadFailed,
    DownloadFailed,
    InvalidPointer,
    IoError,
    RuntimeError,
}

impl fmt::Display for XetError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for XetError {}

impl XetError {
    pub fn new(kind: XetErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Convert this error to an FFI error code
    pub fn to_ffi_code(self) -> libc::c_int {
        to_ffi_error(self)
    }
}

impl From<git2::Error> for XetError {
    fn from(e: git2::Error) -> Self {
        Self::new(XetErrorKind::IoError, format!("Git error: {}", e))
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
