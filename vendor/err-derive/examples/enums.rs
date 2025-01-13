#[cfg(feature = "std")]
use std::error::Error;
#[cfg(not(feature = "std"))]
use std::fmt::Display;
use std::path::PathBuf;

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
    NotFound { path: PathBuf },
}

fn main() {
    let my_error: LoadingError = FormatError::MissingAttribute("some_attr".to_owned()).into();

    print_error(&my_error);
}

#[cfg(feature = "std")]
fn print_error(e: &dyn Error) {
    eprintln!("error: {}", e);
    let mut cause = e.source();
    while let Some(e) = cause {
        eprintln!("caused by: {}", e);
        cause = e.source();
    }
}

#[cfg(not(feature = "std"))]
fn print_error(e: &dyn Display) {
    eprintln!("error: {}", e);
}
