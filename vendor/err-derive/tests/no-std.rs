#![cfg(not(feature = "std"))]
extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use err_derive::Error;

#[derive(Debug, Error)]
pub enum FormatError {
    #[error(
        display = "invalid header (expected: {:?}, got: {:?})",
        expected,
        found
    )]
    InvalidHeader { expected: String, found: String },
    #[error(display = "missing attribute: {:?}", _0)]
    MissingAttribute(String),
}

#[derive(Debug, Error)]
pub enum LoadingError {
    #[error(display = "could not decode file")]
    FormatError(#[error(source)] FormatError),
    #[error(display = "could not find file: {:?}", path)]
    NotFound { path: String },
}

/// Verify that our error types implement `From` and `Display`
pub fn test_impl() {
    let loading_err: LoadingError = FormatError::MissingAttribute("attr".to_string()).into();
    let _ = format!("Error: {}", loading_err);
}
