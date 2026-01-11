//! Transport layer for RPC communication.
//!
//! This module provides:
//! - `Transport` / `AsyncTransport` traits for generic transport abstraction
//! - `TransportConfig` for unified endpoint configuration
//! - Systemd socket activation support via `SystemdFd` variant
//! - Raw socket options via `sockopt` submodule

mod traits;
pub mod sockopt;

use std::os::unix::io::RawFd;
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

    /// Systemd socket activation endpoint.
    ///
    /// Used when systemd passes a pre-bound file descriptor to the service.
    /// The `fd` is used for server-side binding (via `ZMQ_USE_FD`),
    /// while `client_path` provides the IPC path for clients to connect.
    SystemdFd {
        /// Pre-bound file descriptor from systemd
        fd: RawFd,
        /// IPC path for client connections
        client_path: PathBuf,
    },
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

    /// Create a systemd socket activation endpoint.
    ///
    /// # Arguments
    ///
    /// * `fd` - Pre-bound file descriptor from systemd
    /// * `client_path` - IPC path for client connections
    pub fn systemd_fd(fd: RawFd, client_path: impl Into<PathBuf>) -> Self {
        Self::SystemdFd {
            fd,
            client_path: client_path.into(),
        }
    }

    /// Parse a ZMQ endpoint string into a TransportConfig.
    ///
    /// Supports:
    /// - `inproc://name` → `TransportConfig::Inproc`
    /// - `tcp://host:port` → `TransportConfig::Tcp` (no CURVE)
    /// - `ipc:///path/to/socket` → `TransportConfig::Ipc`
    ///
    /// # Example
    ///
    /// ```
    /// use hyprstream_rpc::transport::TransportConfig;
    ///
    /// let config = TransportConfig::from_endpoint("inproc://hyprstream/registry");
    /// assert_eq!(config.zmq_endpoint(), "inproc://hyprstream/registry");
    /// ```
    pub fn from_endpoint(endpoint: &str) -> Self {
        if let Some(name) = endpoint.strip_prefix("inproc://") {
            Self::Inproc {
                endpoint: name.to_string(),
            }
        } else if let Some(addr) = endpoint.strip_prefix("tcp://") {
            Self::Tcp {
                endpoint: addr.to_string(),
                curve_pubkey: None,
            }
        } else if let Some(path) = endpoint.strip_prefix("ipc://") {
            Self::Ipc {
                path: PathBuf::from(path),
            }
        } else {
            // Default to inproc if no scheme
            Self::Inproc {
                endpoint: endpoint.to_string(),
            }
        }
    }

    /// Get the ZMQ endpoint string for this configuration.
    ///
    /// Returns the full ZMQ endpoint URI (e.g., `tcp://127.0.0.1:5560`).
    /// For `SystemdFd`, returns the client IPC path.
    pub fn zmq_endpoint(&self) -> String {
        match self {
            Self::Inproc { endpoint } => format!("inproc://{}", endpoint),
            Self::Tcp { endpoint, .. } => format!("tcp://{}", endpoint),
            Self::Ipc { path } => format!("ipc://{}", path.display()),
            Self::SystemdFd { client_path, .. } => format!("ipc://{}", client_path.display()),
        }
    }

    /// Alias for `zmq_endpoint()` for consistency.
    #[inline]
    pub fn to_zmq_string(&self) -> String {
        self.zmq_endpoint()
    }

    /// Bind a ZMQ socket to this endpoint.
    ///
    /// For `SystemdFd`, uses the pre-bound file descriptor via `ZMQ_USE_FD`.
    /// For other variants, binds to the endpoint string.
    ///
    /// # Errors
    ///
    /// Returns an error if binding fails.
    pub fn bind(&self, socket: &mut zmq::Socket) -> anyhow::Result<()> {
        match self {
            Self::SystemdFd { fd, .. } => {
                sockopt::set_use_fd(socket, *fd)?;
                Ok(())
            }
            _ => {
                socket.bind(&self.zmq_endpoint())?;
                Ok(())
            }
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

    /// Check if this is a systemd-activated endpoint.
    pub fn is_systemd_activated(&self) -> bool {
        matches!(self, Self::SystemdFd { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inproc_endpoint() {
        let config = TransportConfig::inproc("hyprstream/registry");
        assert_eq!(config.zmq_endpoint(), "inproc://hyprstream/registry");
        assert_eq!(config.to_zmq_string(), "inproc://hyprstream/registry");
        assert!(!config.uses_curve());
        assert!(!config.is_systemd_activated());
    }

    #[test]
    fn test_tcp_endpoint() {
        let config = TransportConfig::tcp("127.0.0.1:5560", None);
        assert_eq!(config.zmq_endpoint(), "tcp://127.0.0.1:5560");
        assert!(!config.uses_curve());
        assert!(!config.is_systemd_activated());
    }

    #[test]
    fn test_tcp_with_curve() {
        let key = [42u8; 32];
        let config = TransportConfig::tcp("192.168.1.100:5560", Some(key));
        assert_eq!(config.zmq_endpoint(), "tcp://192.168.1.100:5560");
        assert!(config.uses_curve());
        assert_eq!(config.curve_pubkey(), Some(&key));
        assert!(!config.is_systemd_activated());
    }

    #[test]
    fn test_ipc_endpoint() {
        let config = TransportConfig::ipc("/tmp/hyprstream.sock");
        assert_eq!(config.zmq_endpoint(), "ipc:///tmp/hyprstream.sock");
        assert!(!config.uses_curve());
        assert!(!config.is_systemd_activated());
    }

    #[test]
    fn test_systemd_fd_endpoint() {
        let config = TransportConfig::systemd_fd(5, "/run/hyprstream/policy.sock");
        assert_eq!(
            config.zmq_endpoint(),
            "ipc:///run/hyprstream/policy.sock"
        );
        assert!(!config.uses_curve());
        assert!(config.is_systemd_activated());
    }
}
