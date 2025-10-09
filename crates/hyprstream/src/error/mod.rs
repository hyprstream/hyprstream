use std::error::Error;
use std::fmt;

// Wrapper for tonic::Status
#[derive(Debug)]
pub struct StatusWrapper(pub tonic::Status);

impl Error for StatusWrapper {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.0)
    }
}

impl fmt::Display for StatusWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

