//! Error handling extensions for Cap'n Proto operations.

use thiserror::Error;

/// RPC-specific error types.
#[derive(Debug, Error)]
pub enum RpcError {
    /// Cap'n Proto serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Transport error (ZMQ, network, etc.).
    #[error("Transport error: {0}")]
    Transport(String),

    /// Service returned an error response.
    #[error("Service error: {0}")]
    Service(String),

    /// Unexpected response type from service.
    #[error("Unexpected response type")]
    UnexpectedResponse,

    /// Service is unavailable.
    #[error("Service unavailable")]
    Unavailable,
}

impl RpcError {
    /// Create a transport error.
    pub fn transport<S: Into<String>>(msg: S) -> Self {
        Self::Transport(msg.into())
    }

    /// Create a serialization error.
    pub fn serialization<S: Into<String>>(msg: S) -> Self {
        Self::Serialization(msg.into())
    }

    /// Create a service error.
    pub fn service<S: Into<String>>(msg: S) -> Self {
        Self::Service(msg.into())
    }
}

/// Extension trait for adding RPC error context to Results.
///
/// Simplifies the common pattern of mapping errors to RPC errors.
///
/// # Example
///
/// ```ignore
/// use hyprstream_rpc::capnp::CapnpResultExt;
///
/// // Instead of:
/// let value = reader.get_field()
///     .map_err(|e| RpcError::serialization(e.to_string()))?
///     .to_str()
///     .map_err(|e| RpcError::serialization(e.to_string()))?;
///
/// // Write:
/// let value = reader.get_field().rpc_err()?.to_str().rpc_err()?;
/// ```
pub trait CapnpResultExt<T> {
    /// Convert error to RpcError::Serialization.
    fn rpc_err(self) -> Result<T, RpcError>;

    /// Convert error to RpcError::Transport.
    fn transport_err(self) -> Result<T, RpcError>;
}

impl<T, E: std::fmt::Display> CapnpResultExt<T> for Result<T, E> {
    fn rpc_err(self) -> Result<T, RpcError> {
        self.map_err(|e| RpcError::serialization(e.to_string()))
    }

    fn transport_err(self) -> Result<T, RpcError> {
        self.map_err(|e| RpcError::transport(e.to_string()))
    }
}
