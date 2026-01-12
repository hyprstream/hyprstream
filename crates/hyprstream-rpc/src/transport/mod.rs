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
/// // IPC endpoint (auto-detects systemd socket activation)
/// let ipc = TransportConfig::ipc("/run/user/1000/hyprstream/service.sock");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportConfig {
    /// In-process endpoint (zero-copy, same process).
    ///
    /// Format: `inproc://hyprstream/service`
    Inproc { endpoint: String },

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
        } else if let Some(path) = endpoint.strip_prefix("ipc://") {
            Self::Ipc { path: PathBuf::from(path) }
        } else {
            // Default to inproc if no scheme
            Self::Inproc {
                endpoint: endpoint.to_string(),
            }
        }
    }

    /// Get the ZMQ endpoint string for this configuration.
    ///
    /// For `SystemdFd`, returns the client IPC path.
    pub fn zmq_endpoint(&self) -> String {
        match self {
            Self::Inproc { endpoint } => format!("inproc://{}", endpoint),
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

    /// Check if this is a systemd-activated endpoint.
    pub fn is_systemd_activated(&self) -> bool {
        matches!(self, Self::SystemdFd { .. })
    }

    /// Check for systemd socket activation and return the file descriptor.
    ///
    /// Returns `Some(fd)` if running under systemd with socket activation,
    /// `None` otherwise.
    fn get_systemd_fd() -> Option<RawFd> {
        const SD_LISTEN_FDS_START: RawFd = 3;

        let listen_fds: i32 = std::env::var("LISTEN_FDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let listen_pid: u32 = std::env::var("LISTEN_PID")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        // Verify PID matches and we have at least 1 FD
        if listen_pid == std::process::id() && listen_fds >= 1 {
            Some(SD_LISTEN_FDS_START)
        } else {
            None
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
        assert_eq!(config.to_zmq_string(), "inproc://hyprstream/registry");
        assert!(!config.is_systemd_activated());
    }

    #[test]
    fn test_ipc_endpoint() {
        let config = TransportConfig::ipc("/tmp/hyprstream.sock");
        assert_eq!(config.zmq_endpoint(), "ipc:///tmp/hyprstream.sock");
        assert!(!config.is_systemd_activated());
    }

    #[test]
    fn test_systemd_fd_endpoint() {
        let config = TransportConfig::systemd_fd(5, "/run/hyprstream/policy.sock");
        assert_eq!(
            config.zmq_endpoint(),
            "ipc:///run/hyprstream/policy.sock"
        );
        assert!(config.is_systemd_activated());
    }
}
