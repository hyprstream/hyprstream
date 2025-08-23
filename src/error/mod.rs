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

// Wrapper for VDB storage errors
#[derive(Debug)]
pub struct VDBErrorWrapper(pub crate::storage::SparseStorageError);

impl Error for VDBErrorWrapper {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.0)
    }
}

impl fmt::Display for VDBErrorWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// Convert VDB error wrapper to Status
impl From<VDBErrorWrapper> for tonic::Status {
    fn from(err: VDBErrorWrapper) -> Self {
        match err.0 {
            crate::storage::SparseStorageError::AdapterNotFound { id } => {
                tonic::Status::not_found(format!("Adapter not found: {}", id))
            }
            crate::storage::SparseStorageError::InvalidCoordinates { reason } => {
                tonic::Status::invalid_argument(format!("Invalid coordinates: {}", reason))
            }
            crate::storage::SparseStorageError::DiskError(ref io_err) => {
                tonic::Status::internal(format!("Disk I/O error: {}", io_err))
            }
            crate::storage::SparseStorageError::CompressionError(ref msg) => {
                tonic::Status::internal(format!("Compression error: {}", msg))
            }
            crate::storage::SparseStorageError::HardwareError(ref msg) => {
                tonic::Status::internal(format!("Hardware acceleration error: {}", msg))
            }
            crate::storage::SparseStorageError::ConcurrencyError { ref id } => {
                tonic::Status::aborted(format!("Concurrent modification detected for adapter: {}", id))
            }
        }
    }
}