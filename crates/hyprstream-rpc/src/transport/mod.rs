//! Transport layer for RPC communication.
//!
//! This module provides:
//! - `Transport` / `AsyncTransport` traits for generic transport abstraction
//! - `TransportConfig` for unified endpoint configuration
//! - Systemd socket activation support via `SystemdFd` variant
//! - QUIC/WebTransport via `zmtp_quic` (WebTransportServer + process_request; ZMQ removal pending #138)
//! - Raw socket options via `sockopt` submodule

mod traits;
pub mod sockopt;
pub mod zmtp_quic;
#[cfg(not(target_arch = "wasm32"))]
pub mod iroh_substrate;
#[cfg(not(target_arch = "wasm32"))]
pub mod rpc_session;
#[cfg(not(target_arch = "wasm32"))]
pub mod iroh_rpc;
#[cfg(not(target_arch = "wasm32"))]
pub mod quinn_transport;
#[cfg(not(target_arch = "wasm32"))]
pub mod iroh_transport;
#[cfg(not(target_arch = "wasm32"))]
pub mod iroh_moq;
#[cfg(not(target_arch = "wasm32"))]
pub mod in_memory;
#[cfg(not(target_arch = "wasm32"))]
pub mod lazy_quinn;
#[cfg(not(target_arch = "wasm32"))]
pub mod lazy_iroh;
#[cfg(not(target_arch = "wasm32"))]
pub mod uds_session;
#[cfg(not(target_arch = "wasm32"))]
pub mod lazy_uds;
#[cfg(not(target_arch = "wasm32"))]
pub mod uds_server;

use std::net::SocketAddr;
use std::os::unix::io::RawFd;
use std::path::PathBuf;

pub use crate::transport_traits::{PublishSink, Signer, Transport};

/// Socket bind mode for transport configuration.
///
/// Controls whether a socket binds to an endpoint (standalone server)
/// or connects to it (worker behind a load balancer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BindMode {
    /// Socket binds to the endpoint (default, standalone mode).
    #[default]
    Bind,
    /// Socket connects to the endpoint (worker behind ROUTER/DEALER LB).
    Connect,
}

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
/// ```
///
/// ```ignore
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

    /// Socket bind mode (Bind or Connect).
    /// Workers behind a ROUTER/DEALER load balancer use Connect.
    pub bind_mode: BindMode,
}

/// How a QUIC client authenticates the **channel** to the server it dials.
///
/// # This is channel auth, not identity auth
///
/// Peer *identity* ("which node is this") is established at the **application
/// layer** — every response is a signed COSE `SignedEnvelope` verified against
/// the peer's published keys (the DID-doc `#mesh` verification method / JWKS).
/// That works identically on native and in WASM, because it is our code, not the
/// browser's TLS stack. This type only decides how the *TLS channel* is
/// authenticated (confidentiality + anti-active-MITM); it is defence in depth,
/// not the trust root (#185).
///
/// Channel auth is a small lattice of independent requirements, not a fixed set
/// of modes, so it is a validated struct rather than an enum of combinations:
///   - `require_web_pki` — the leaf must chain to a system-trusted CA and match
///     `server_name` (public peers, dialed by hostname).
///   - `accept_cert_hashes` — a **set** of acceptable leaf-cert SHA-256s. A set
///     (not one hash) so cert rotation can overlap (`{current, next}`) and
///     load-balanced endpoints with distinct certs work. The resolver / DID doc
///     supplies these — they are directory-distributed pins, not trust-on-first-
///     use. (For browsers this is the *only* knob the WebTransport API exposes.)
///
/// The verifier ANDs the active requirements. At least one must be active —
/// `{web_pki: false, hashes: []}` is no auth and is rejected at construction.
///
/// RFC 7250 raw-public-key binding (bind the channel to the published key
/// directly, so native QUIC reuses the identity key instead of a cert hash) is
/// tracked in #200; it is channel hygiene, not an identity requirement (identity
/// is app-layer regardless). iroh already does it natively.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuicServerAuth {
    require_web_pki: bool,
    accept_cert_hashes: Vec<[u8; 32]>,
}

impl QuicServerAuth {
    /// WebPKI / CA-chain validation only (public, CA-fronted peer).
    pub fn web_pki() -> Self {
        Self { require_web_pki: true, accept_cert_hashes: Vec::new() }
    }

    /// Pin the leaf cert to one of `hashes` (SHA-256), no CA requirement — the
    /// self-signed internal-mesh model. Errors if `hashes` is empty.
    pub fn pinned(hashes: Vec<[u8; 32]>) -> anyhow::Result<Self> {
        anyhow::ensure!(!hashes.is_empty(), "QuicServerAuth::pinned requires >= 1 cert hash");
        Ok(Self { require_web_pki: false, accept_cert_hashes: hashes })
    }

    /// WebPKI validation **and** the leaf must be one of `hashes` — defence in
    /// depth against CA mis-issuance (HPKP-style). Errors if `hashes` is empty.
    pub fn web_pki_pinned(hashes: Vec<[u8; 32]>) -> anyhow::Result<Self> {
        anyhow::ensure!(!hashes.is_empty(), "QuicServerAuth::web_pki_pinned requires >= 1 cert hash");
        Ok(Self { require_web_pki: true, accept_cert_hashes: hashes })
    }

    /// Whether CA-chain validation is required.
    pub fn require_web_pki(&self) -> bool {
        self.require_web_pki
    }

    /// The set of accepted leaf-cert SHA-256 pins (empty => no pin requirement).
    pub fn accept_cert_hashes(&self) -> &[[u8; 32]] {
        &self.accept_cert_hashes
    }
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

    /// QUIC transport endpoint (ZMTP 3.1 over QUIC).
    ///
    /// Provides TLS 1.3 encryption built into the transport layer,
    /// replacing CurveZMQ. ZMTP handshake uses NULL mechanism since
    /// QUIC already provides wire confidentiality.
    ///
    /// Format: `quic://hostname:port`
    Quic {
        /// Socket address to bind (server) or connect (client)
        addr: SocketAddr,
        /// Server hostname — used as the WebPKI validation name (`WebPki`) and
        /// the TLS SNI. Ignored by `Pinned` (which dials by IP and matches the
        /// cert hash).
        server_name: String,
        /// How the client authenticates this server (WebPKI, cert-hash pin, or
        /// — later — RFC 7250 raw key). Public material only.
        auth: QuicServerAuth,
    },

    /// iroh transport endpoint (RPC over `ALPN_HYPRSTREAM_RPC`).
    ///
    /// The optional NAT-traversing dial. Unlike QUIC's channel-only cert pin,
    /// iroh binds the connection to the peer's `EndpointId` (its Ed25519 public
    /// key), so the transport itself authenticates the peer *identity*. All
    /// fields are serializable primitives so `TransportConfig` stays
    /// `Clone + Eq` and wire-publishable (a DID-doc `service` entry).
    ///
    /// Format: `iroh://{hex node_id}`
    Iroh {
        /// Peer's iroh `EndpointId` = its Ed25519 public key (32 bytes).
        node_id: [u8; 32],
        /// Known direct socket addresses for hole-punching / direct dial.
        direct_addrs: Vec<SocketAddr>,
        /// Optional relay URL for NAT traversal when no direct path exists.
        relay_url: Option<String>,
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
            bind_mode: BindMode::Bind,
        }
    }

    /// Create an IPC (Unix domain socket) endpoint configuration.
    pub fn ipc(path: impl Into<PathBuf>) -> Self {
        Self {
            endpoint: EndpointType::Ipc {
                path: path.into(),
            },
            curve: None,
            bind_mode: BindMode::Bind,
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
            bind_mode: BindMode::Bind,
        }
    }

    /// Create a QUIC transport endpoint.
    ///
    /// QUIC provides TLS 1.3 encryption at the transport layer, so CurveZMQ
    /// is not needed (NULL mechanism is used in ZMTP handshake).
    ///
    /// # Arguments
    ///
    /// * `addr` - Socket address to bind (server) or connect (client)
    /// * `server_name` - Server hostname for TLS certificate validation
    pub fn quic(addr: SocketAddr, server_name: impl Into<String>) -> Self {
        Self {
            endpoint: EndpointType::Quic {
                addr,
                server_name: server_name.into(),
                auth: QuicServerAuth::web_pki(),
            },
            curve: None, // QUIC has TLS 1.3 built-in, no CurveZMQ needed
            bind_mode: BindMode::Bind,
        }
    }

    /// Create a client QUIC endpoint that pins the server's self-signed cert by
    /// its SHA-256 fingerprint. This is the dial target the resolver/`dial()`
    /// produce for an internal-mesh RPC peer (dialed by IP).
    pub fn quic_pinned(addr: SocketAddr, server_name: impl Into<String>, cert_hash: [u8; 32]) -> Self {
        Self {
            endpoint: EndpointType::Quic {
                addr,
                server_name: server_name.into(),
                // In-module: construct directly (a one-element set is always valid).
                auth: QuicServerAuth { require_web_pki: false, accept_cert_hashes: vec![cert_hash] },
            },
            curve: None,
            bind_mode: BindMode::Connect,
        }
    }

    /// Create a client QUIC endpoint with an explicit channel-auth policy
    /// (used by the DID-doc service-entry codec, which may produce any of
    /// WebPKI / Pinned / WebPKI+pin).
    pub fn quic_with_auth(addr: SocketAddr, server_name: impl Into<String>, auth: QuicServerAuth) -> Self {
        Self {
            endpoint: EndpointType::Quic { addr, server_name: server_name.into(), auth },
            curve: None,
            bind_mode: BindMode::Connect,
        }
    }

    /// Create a client iroh endpoint dialing the peer identified by `node_id`
    /// (its `EndpointId` / Ed25519 public key), with optional direct addresses
    /// and relay URL for NAT traversal.
    pub fn iroh(
        node_id: [u8; 32],
        direct_addrs: Vec<SocketAddr>,
        relay_url: Option<String>,
    ) -> Self {
        Self {
            endpoint: EndpointType::Iroh {
                node_id,
                direct_addrs,
                relay_url,
            },
            curve: None,
            bind_mode: BindMode::Connect,
        }
    }

    /// Set bind mode to Connect (for workers behind a load balancer).
    pub fn with_connect_mode(mut self) -> Self {
        self.bind_mode = BindMode::Connect;
        self
    }

    /// Get the bind mode.
    pub fn bind_mode(&self) -> BindMode {
        self.bind_mode
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
                endpoint: name.to_owned(),
            }
        } else if let Some(path) = endpoint.strip_prefix("ipc://") {
            EndpointType::Ipc {
                path: PathBuf::from(path),
            }
        } else if endpoint.contains("://") {
            // Unknown scheme — fail-closed: misrouting to inproc would yield a
            // cryptic "no in-process service registered" error at dial time.
            // Networked transports (quic://, iroh://, tcp://) are obtained via
            // TransportConfig::from_resolver(), not from_endpoint().
            panic!(
                "TransportConfig::from_endpoint: unknown scheme in '{endpoint}'; \
                 use from_resolver() for networked transports"
            );
        } else {
            // Bare name (no scheme) — defaults to inproc
            EndpointType::Inproc {
                endpoint: endpoint.to_owned(),
            }
        };

        Self {
            endpoint: endpoint_type,
            curve: None,
            bind_mode: BindMode::Bind,
        }
    }

    /// Get the ZMQ endpoint string for this configuration.
    ///
    /// For `SystemdFd`, returns the client IPC path.
    /// For `Quic`, returns a descriptive string (not a valid ZMQ endpoint).
    pub fn zmq_endpoint(&self) -> String {
        match &self.endpoint {
            EndpointType::Inproc { endpoint } => format!("inproc://{endpoint}"),
            EndpointType::Ipc { path } => format!("ipc://{}", path.display()),
            EndpointType::SystemdFd { client_path, .. } => {
                format!("ipc://{}", client_path.display())
            }
            EndpointType::Quic { addr, server_name, .. } => {
                format!("quic://{server_name}:{addr}")
            }
            EndpointType::Iroh { node_id, .. } => {
                let hex: String = node_id.iter().map(|b| format!("{b:02x}")).collect();
                format!("iroh://{hex}")
            }
        }
    }

    /// Alias for `zmq_endpoint()` for consistency.
    #[inline]
    pub fn to_zmq_string(&self) -> String {
        self.zmq_endpoint()
    }

    /// Get the WebTransport URL for QUIC endpoints.
    ///
    /// Returns `https://{server_name}:{port}` for QUIC endpoints, `None` otherwise.
    pub fn quic_webtransport_url(&self) -> Option<String> {
        match &self.endpoint {
            EndpointType::Quic { addr, server_name, .. } => {
                Some(format!("https://{}:{}", server_name, addr.port()))
            }
            _ => None,
        }
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
    /// Returns an error if binding fails, directory creation fails,
    /// or the endpoint type is `Quic` (use `QuicRep::bind` instead).
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
            EndpointType::Quic { .. } | EndpointType::Iroh { .. } => {
                anyhow::bail!(
                    "QUIC/iroh endpoints are not ZMQ sockets — they are dialed via \
                     the transport/dial() layer, not bound here"
                )
            }
        }
    }

    /// Connect a ZMQ socket to this endpoint with optional CurveZMQ.
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails or the endpoint type is `Quic`
    /// (use `QuicReq::connect` instead).
    pub fn connect(&self, socket: &mut zmq::Socket) -> anyhow::Result<()> {
        // Apply CurveZMQ first (before connect)
        self.apply_curve(socket, false)?;

        match &self.endpoint {
            EndpointType::Inproc { .. }
            | EndpointType::Ipc { .. }
            | EndpointType::SystemdFd { .. } => {
                socket.connect(&self.zmq_endpoint())?;
                Ok(())
            }
            EndpointType::Quic { .. } | EndpointType::Iroh { .. } => {
                anyhow::bail!(
                    "QUIC/iroh endpoints are not ZMQ sockets — they are dialed via \
                     the transport/dial() layer, not connected here"
                )
            }
        }
    }

    /// Check if this is a systemd-activated endpoint.
    pub fn is_systemd_activated(&self) -> bool {
        matches!(&self.endpoint, EndpointType::SystemdFd { .. })
    }

    /// Check if this is a QUIC endpoint.
    pub fn is_quic(&self) -> bool {
        matches!(&self.endpoint, EndpointType::Quic { .. })
    }

    /// Build an RFC 9728 resource URL from a QUIC endpoint.
    ///
    /// Returns `https://{server_name}/{path}`, matching the format used
    /// by `QuicSharedConfig::for_service()`. The port is omitted because
    /// resource URLs are identity tokens (used as JWT audience), not
    /// connection endpoints — clients discover actual ports via the
    /// endpoint registry.
    ///
    /// Returns `None` if this is not a QUIC endpoint.
    pub fn quic_resource_url(&self, path: &str) -> Option<String> {
        match &self.endpoint {
            EndpointType::Quic { server_name, .. } => {
                Some(format!("https://{}/{}", server_name, path))
            }
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

    #[test]
    fn test_quic_endpoint() {
        use std::net::{IpAddr, Ipv4Addr};

        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 4433);
        let config = TransportConfig::quic(addr, "hyprstream.local");
        assert!(config.zmq_endpoint().starts_with("quic://hyprstream.local:"));
        assert!(config.is_quic());
        assert!(!config.is_systemd_activated());
    }
}
