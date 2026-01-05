use std::error::Error;
use std::fmt;

// Wrapper for tonic::Status
#[derive(Debug)]
pub struct StatusWrapper(pub tonic::Status);

impl Error for StatusWrapper {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl fmt::Display for StatusWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Wrapper for DuckDB errors
#[derive(Debug)]
pub struct DuckDbErrorWrapper(pub duckdb::Error);

impl Error for DuckDbErrorWrapper {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.0)
    }
}

impl fmt::Display for DuckDbErrorWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// Convert DuckDB error wrapper to Status
impl From<DuckDbErrorWrapper> for tonic::Status {
    fn from(err: DuckDbErrorWrapper) -> Self {
        tonic::Status::internal(format!("DuckDB error: {}", err.0))
    }
}

// Convert StatusWrapper to DataFusionError
impl From<StatusWrapper> for datafusion::error::DataFusionError {
    fn from(status: StatusWrapper) -> Self {
        datafusion::error::DataFusionError::External(Box::new(status))
    }
}
