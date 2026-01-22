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

/// CurveZMQ security configuration.
///
/// CurveZMQ provides transport-layer encryption and authentication using
/// elliptic curve cryptography (Curve25519). Requires ZMQ compiled with libsodium.
///
/// # Security Layers
///
/// - **Encryption**: All messages encrypted with ephemeral session keys
/// - **Authentication**: Server verifies client public keys (optional client auth)
/// - **Forward secrecy**: Compromising long-term keys doesn't decrypt past sessions
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurveConfig {
    /// Server public key (required for clients)
    pub server_public_key: Option<[u8; 32]>,

    /// Client secret key (for client authentication)
    pub client_secret_key: Option<[u8; 32]>,

    /// Client public key (for client authentication)
    pub client_public_key: Option<[u8; 32]>,
}

impl CurveConfig {
    /// Create a client configuration (connects to server with server's public key)
    pub fn client(server_public_key: [u8; 32]) -> Self {
        Self {
            server_public_key: Some(server_public_key),
            client_secret_key: None,
            client_public_key: None,
        }
    }

    /// Create a server configuration (accepts any authenticated client)
    pub fn server() -> Self {
        Self {
            server_public_key: None,
            client_secret_key: None,
            client_public_key: None,
        }
    }
}

/// ZMQ endpoint configuration with optional CurveZMQ encryption.
///
/// This enum represents the different endpoint types ZMQ can connect to.
/// Network-level routing (SOCKS proxy, WireGuard, etc.) is configured
/// separately at the socket/system level.
///
/// # Security
///
/// CurveZMQ can be enabled on any transport type via `with_curve()`.
/// This provides transport-layer encryption and authentication.
///
/// # Examples
///
/// ```
/// use hyprstream_rpc::transport::TransportConfig;
///
/// // In-process endpoint (no encryption needed)
/// let inproc = TransportConfig::inproc("hyprstream/registry");
///
/// // IPC endpoint with CurveZMQ encryption
/// let ipc = TransportConfig::ipc("/run/user/1000/hyprstream/service.sock")
///     .with_curve(server_keypair);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportConfig {
    /// Endpoint type (inproc, IPC, systemd FD)
    pub endpoint: EndpointType,

    /// Optional CurveZMQ encryption configuration
    pub curve: Option<CurveConfig>,
}

/// Endpoint type (without encryption config)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointType {
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
        Self {
            endpoint: EndpointType::Inproc {
                endpoint: endpoint.into(),
            },
            curve: None,
        }
    }

    /// Create an IPC (Unix domain socket) endpoint configuration.
    pub fn ipc(path: impl Into<PathBuf>) -> Self {
        Self {
            endpoint: EndpointType::Ipc {
                path: path.into(),
            },
            curve: None,
        }
    }

    /// Create a systemd socket activation endpoint.
    ///
    /// # Arguments
    ///
    /// * `fd` - Pre-bound file descriptor from systemd
    /// * `client_path` - IPC path for client connections
    pub fn systemd_fd(fd: RawFd, client_path: impl Into<PathBuf>) -> Self {
        Self {
            endpoint: EndpointType::SystemdFd {
                fd,
                client_path: client_path.into(),
            },
            curve: None,
        }
    }

    /// Enable CurveZMQ encryption for this transport.
    ///
    /// This provides transport-layer encryption and authentication for any
    /// ZMQ socket type (REQ/REP, PUB/SUB, PUSH/PULL, etc.).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = TransportConfig::ipc("/run/service.sock")
    ///     .with_curve(CurveConfig::server());
    /// ```
    pub fn with_curve(mut self, curve: CurveConfig) -> Self {
        self.curve = Some(curve);
        self
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
        let endpoint_type = if let Some(name) = endpoint.strip_prefix("inproc://") {
            EndpointType::Inproc {
                endpoint: name.to_string(),
            }
        } else if let Some(path) = endpoint.strip_prefix("ipc://") {
            EndpointType::Ipc {
                path: PathBuf::from(path),
            }
        } else {
            // Default to inproc if no scheme
            EndpointType::Inproc {
                endpoint: endpoint.to_string(),
            }
        };

        Self {
            endpoint: endpoint_type,
            curve: None,
        }
    }

    /// Get the ZMQ endpoint string for this configuration.
    ///
    /// For `SystemdFd`, returns the client IPC path.
    pub fn zmq_endpoint(&self) -> String {
        match &self.endpoint {
            EndpointType::Inproc { endpoint } => format!("inproc://{}", endpoint),
            EndpointType::Ipc { path } => format!("ipc://{}", path.display()),
            EndpointType::SystemdFd { client_path, .. } => {
                format!("ipc://{}", client_path.display())
            }
        }
    }

    /// Alias for `zmq_endpoint()` for consistency.
    #[inline]
    pub fn to_zmq_string(&self) -> String {
        self.zmq_endpoint()
    }

    /// Apply CurveZMQ configuration to a socket (client or server).
    ///
    /// This method configures CurveZMQ encryption and authentication on the given socket.
    /// Gracefully degrades if ZMQ wasn't compiled with libsodium support.
    ///
    /// # Arguments
    ///
    /// * `socket` - ZMQ socket to configure
    /// * `is_server` - True for server sockets (bind), false for client sockets (connect)
    ///
    /// # Returns
    ///
    /// `Ok(true)` if CurveZMQ was successfully configured,
    /// `Ok(false)` if CurveZMQ is not supported (graceful degradation),
    /// `Err` if configuration fails.
    pub fn apply_curve(&self, socket: &mut zmq::Socket, is_server: bool) -> anyhow::Result<bool> {
        let Some(ref curve) = self.curve else {
            return Ok(false); // No CurveZMQ requested
        };

        if is_server {
            // Server configuration
            socket
                .set_curve_server(true)
                .map_err(|e| {
                    if e == zmq::Error::ENOTSUP {
                        tracing::warn!("CurveZMQ not supported by ZMQ build - encryption disabled");
                        return anyhow::anyhow!("CurveZMQ not supported");
                    }
                    anyhow::anyhow!("Failed to enable CurveZMQ server: {}", e)
                })?;
            Ok(true)
        } else {
            // Client configuration
            if let Some(server_pubkey) = curve.server_public_key {
                socket
                    .set_curve_serverkey(&server_pubkey)
                    .map_err(|e| anyhow::anyhow!("Failed to set server public key: {}", e))?;

                // Generate ephemeral client keypair if not provided
                if let (Some(client_secret), Some(client_public)) =
                    (curve.client_secret_key, curve.client_public_key)
                {
                    socket
                        .set_curve_secretkey(&client_secret)
                        .map_err(|e| anyhow::anyhow!("Failed to set client secret key: {}", e))?;
                    socket
                        .set_curve_publickey(&client_public)
                        .map_err(|e| anyhow::anyhow!("Failed to set client public key: {}", e))?;
                }

                Ok(true)
            } else {
                Ok(false) // No server public key, skip CurveZMQ
            }
        }
    }

    /// Bind a ZMQ socket to this endpoint with optional CurveZMQ.
    ///
    /// For `SystemdFd`, uses the pre-bound file descriptor via `ZMQ_USE_FD`.
    /// For `Ipc`, creates parent directories before binding.
    /// For other variants, binds to the endpoint string directly.
    ///
    /// # Errors
    ///
    /// Returns an error if binding fails or directory creation fails.
    pub fn bind(&self, socket: &mut zmq::Socket) -> anyhow::Result<()> {
        // Apply CurveZMQ first (before bind)
        self.apply_curve(socket, true)?;

        match &self.endpoint {
            EndpointType::SystemdFd { fd, .. } => {
                sockopt::set_use_fd(socket, *fd)?;
                Ok(())
            }
            EndpointType::Ipc { path } => {
                // Ensure parent directory exists before binding IPC socket
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| anyhow::anyhow!("Failed to create directory {}: {}", parent.display(), e))?;
                }
                socket.bind(&self.zmq_endpoint())?;
                Ok(())
            }
            EndpointType::Inproc { .. } => {
                socket.bind(&self.zmq_endpoint())?;
                Ok(())
            }
        }
    }

    /// Connect a ZMQ socket to this endpoint with optional CurveZMQ.
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails.
    pub fn connect(&self, socket: &mut zmq::Socket) -> anyhow::Result<()> {
        // Apply CurveZMQ first (before connect)
        self.apply_curve(socket, false)?;

        socket.connect(&self.zmq_endpoint())?;
        Ok(())
    }

    /// Check if this is a systemd-activated endpoint.
    pub fn is_systemd_activated(&self) -> bool {
        matches!(&self.endpoint, EndpointType::SystemdFd { .. })
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
