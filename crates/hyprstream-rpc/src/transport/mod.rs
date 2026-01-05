//! Transport layer for RPC communication.
//!
//! This module provides:
//! - `Transport` / `AsyncTransport` traits for generic transport abstraction
//! - `TransportConfig` for unified endpoint configuration

mod traits;

use std::path::PathBuf;

pub use traits::{AsyncTransport, Transport};

/// ZMQ endpoint configuration.
///
/// This enum represents the different endpoint types ZMQ can connect to.
/// Network-level routing (SOCKS proxy, WireGuard, etc.) is configured
/// separately at the socket/system level.
///
/// # Examples
///
/// ```
/// use hyprstream_rpc::transport::TransportConfig;
///
/// // In-process endpoint
/// let inproc = TransportConfig::inproc("hyprstream/registry");
///
/// // TCP endpoint (no CURVE)
/// let tcp = TransportConfig::tcp("127.0.0.1:5560", None);
///
/// // TCP endpoint with CURVE authentication
/// let curve_key = [0u8; 32]; // Server's public key
/// let tcp_curve = TransportConfig::tcp("192.168.1.100:5560", Some(curve_key));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportConfig {
    /// In-process endpoint (zero-copy, same process).
    ///
    /// Format: `inproc://hyprstream/service`
    Inproc { endpoint: String },

    /// TCP socket endpoint, optionally with CURVE encryption.
    ///
    /// Format: `tcp://host:port`
    ///
    /// If `curve_pubkey` is provided, CURVE encryption is used.
    /// The key is the server's 32-byte CURVE public key.
    Tcp {
        endpoint: String,
        curve_pubkey: Option<[u8; 32]>,
    },

    /// Unix domain socket (IPC) endpoint.
    ///
    /// Format: `ipc:///path/to/socket`
    Ipc { path: PathBuf },
}

impl TransportConfig {
    /// Create an in-process endpoint configuration.
    pub fn inproc(endpoint: impl Into<String>) -> Self {
        Self::Inproc {
            endpoint: endpoint.into(),
        }
    }

    /// Create a TCP endpoint configuration.
    pub fn tcp(endpoint: impl Into<String>, curve_pubkey: Option<[u8; 32]>) -> Self {
        Self::Tcp {
            endpoint: endpoint.into(),
            curve_pubkey,
        }
    }

    /// Create an IPC (Unix domain socket) endpoint configuration.
    pub fn ipc(path: impl Into<PathBuf>) -> Self {
        Self::Ipc { path: path.into() }
    }

    /// Get the ZMQ endpoint string for this configuration.
    ///
    /// Returns the full ZMQ endpoint URI (e.g., `tcp://127.0.0.1:5560`).
    pub fn zmq_endpoint(&self) -> String {
        match self {
            Self::Inproc { endpoint } => format!("inproc://{}", endpoint),
            Self::Tcp { endpoint, .. } => format!("tcp://{}", endpoint),
            Self::Ipc { path } => format!("ipc://{}", path.display()),
        }
    }

    /// Check if this endpoint uses CURVE encryption.
    pub fn uses_curve(&self) -> bool {
        matches!(self, Self::Tcp { curve_pubkey: Some(_), .. })
    }

    /// Get the CURVE public key if configured.
    pub fn curve_pubkey(&self) -> Option<&[u8; 32]> {
        match self {
            Self::Tcp { curve_pubkey: Some(key), .. } => Some(key),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inproc_endpoint() {
        let config = TransportConfig::inproc("hyprstream/registry");
        assert_eq!(config.zmq_endpoint(), "inproc://hyprstream/registry");
        assert!(!config.uses_curve());
    }

    #[test]
    fn test_tcp_endpoint() {
        let config = TransportConfig::tcp("127.0.0.1:5560", None);
        assert_eq!(config.zmq_endpoint(), "tcp://127.0.0.1:5560");
        assert!(!config.uses_curve());
    }

    #[test]
    fn test_tcp_with_curve() {
        let key = [42u8; 32];
        let config = TransportConfig::tcp("192.168.1.100:5560", Some(key));
        assert_eq!(config.zmq_endpoint(), "tcp://192.168.1.100:5560");
        assert!(config.uses_curve());
        assert_eq!(config.curve_pubkey(), Some(&key));
    }

    #[test]
    fn test_ipc_endpoint() {
        let config = TransportConfig::ipc("/tmp/hyprstream.sock");
        assert_eq!(config.zmq_endpoint(), "ipc:///tmp/hyprstream.sock");
        assert!(!config.uses_curve());
    }
}
