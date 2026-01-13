//! Event bus endpoint configuration
//!
//! Supports three endpoint modes for EventService sockets:
//! - **Inproc**: In-process transport for monolithic mode (same ZMQ context)
//! - **IPC**: Unix domain sockets for distributed mode (separate processes)
//! - **SystemdFd**: Pre-bound file descriptors from systemd socket activation
//!
//! # Path Resolution
//!
//! IPC socket paths are resolved dynamically based on execution context:
//! - Root/system service: `/run/hyprstream/events/`
//! - User service: `$XDG_RUNTIME_DIR/hyprstream/events/`
//! - Fallback: `/tmp/hyprstream-<uid>/events/`

use std::os::unix::io::RawFd;
use std::path::PathBuf;

use crate::paths;
use hyprstream_rpc::transport::TransportConfig;

// =============================================================================
// Inproc endpoints (monolithic mode)
// =============================================================================

/// Publishers connect here (XSUB binds)
pub const PUB: &str = "inproc://hyprstream/events/pub";

/// Subscribers connect here (XPUB binds)
pub const SUB: &str = "inproc://hyprstream/events/sub";

/// Control socket for graceful shutdown (PAIR)
pub const CTRL: &str = "inproc://hyprstream/events/ctrl";

// =============================================================================
// IPC endpoints (distributed mode)
// =============================================================================

/// Get IPC path for publisher socket
///
/// Path resolved via `paths` module for user/system mode support.
pub fn pub_ipc() -> PathBuf {
    paths::events_pub_socket()
}

/// Get IPC path for subscriber socket
///
/// Path resolved via `paths` module for user/system mode support.
pub fn sub_ipc() -> PathBuf {
    paths::events_sub_socket()
}

/// Get IPC endpoint string for publishers
pub fn pub_ipc_endpoint() -> String {
    format!("ipc://{}", pub_ipc().display())
}

/// Get IPC endpoint string for subscribers
pub fn sub_ipc_endpoint() -> String {
    format!("ipc://{}", sub_ipc().display())
}

// =============================================================================
// TransportConfig factory functions (for ServiceSpawner API)
// =============================================================================

/// Create inproc transport configs (for monolithic mode)
///
/// Returns (pub_transport, sub_transport) for use with `ProxyService`.
pub fn inproc_transports() -> (TransportConfig, TransportConfig) {
    (
        TransportConfig::inproc("hyprstream/events/pub"),
        TransportConfig::inproc("hyprstream/events/sub"),
    )
}

/// Create IPC transport configs (for distributed mode)
///
/// Returns (pub_transport, sub_transport) for use with `ProxyService`.
pub fn ipc_transports() -> (TransportConfig, TransportConfig) {
    (
        TransportConfig::ipc(pub_ipc()),
        TransportConfig::ipc(sub_ipc()),
    )
}

/// Create systemd FD transport configs
///
/// Returns (pub_transport, sub_transport) for use with `ProxyService`.
/// Client connections use the IPC paths (systemd manages the actual FDs).
pub fn systemd_transports(pub_fd: RawFd, sub_fd: RawFd) -> (TransportConfig, TransportConfig) {
    (
        TransportConfig::systemd_fd(pub_fd, pub_ipc()),
        TransportConfig::systemd_fd(sub_fd, sub_ipc()),
    )
}

/// Detect and create transport configs based on mode
///
/// Returns (pub_transport, sub_transport) for use with `ProxyService`.
pub fn detect_transports(mode: EndpointMode) -> (TransportConfig, TransportConfig) {
    match mode {
        EndpointMode::Inproc => inproc_transports(),
        EndpointMode::Ipc => ipc_transports(),
        EndpointMode::Auto => {
            // Check for systemd socket activation
            if let Some((pub_fd, sub_fd)) = get_fds() {
                return systemd_transports(pub_fd, sub_fd);
            }
            // Fall back to inproc for monolithic mode
            inproc_transports()
        }
    }
}

/// Detect systemd socket activation and return raw FDs
///
/// Returns `Some((pub_fd, sub_fd))` if systemd socket activation is detected,
/// `None` otherwise.
pub fn get_fds() -> Option<(RawFd, RawFd)> {
    let listen_fds: i32 = std::env::var("LISTEN_FDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let listen_pid: u32 = std::env::var("LISTEN_PID")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // Verify PID matches current process and we have at least 2 FDs
    if listen_pid == std::process::id() && listen_fds >= 2 {
        let pub_fd = SD_LISTEN_FDS_START;
        let sub_fd = SD_LISTEN_FDS_START + 1;
        Some((pub_fd, sub_fd))
    } else {
        None
    }
}

// =============================================================================
// Systemd FD detection (SD_LISTEN_FDS_START = 3)
// =============================================================================

/// Starting file descriptor number for systemd socket activation
const SD_LISTEN_FDS_START: RawFd = 3;

/// Endpoint detection mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EndpointMode {
    /// Always use inproc (monolithic)
    #[default]
    Inproc,
    /// Use IPC sockets (distributed, no systemd)
    Ipc,
    /// Auto-detect: systemd FDs if available, else inproc
    Auto,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inproc_transports() {
        let (pub_t, sub_t) = inproc_transports();

        assert_eq!(pub_t.zmq_endpoint(), PUB);
        assert_eq!(sub_t.zmq_endpoint(), SUB);
    }

    #[test]
    fn test_ipc_transports() {
        let (pub_t, sub_t) = ipc_transports();

        // Verify paths contain "hyprstream" and "events"
        let pub_str = pub_t.zmq_endpoint();
        let sub_str = sub_t.zmq_endpoint();

        assert!(pub_str.starts_with("ipc://"));
        assert!(sub_str.starts_with("ipc://"));
        assert!(pub_str.contains("hyprstream"));
        assert!(sub_str.contains("hyprstream"));
    }

    #[test]
    fn test_systemd_transports() {
        let (pub_t, sub_t) = systemd_transports(3, 4);

        assert!(pub_t.is_systemd_activated());
        assert!(sub_t.is_systemd_activated());
    }

    #[test]
    fn test_detect_transports_default() {
        // Without LISTEN_FDS set, Auto should fall back to inproc
        let (pub_t, sub_t) = detect_transports(EndpointMode::Auto);
        assert_eq!(pub_t.zmq_endpoint(), PUB);
        assert_eq!(sub_t.zmq_endpoint(), SUB);
    }

    #[test]
    fn test_endpoint_mode_default() {
        assert_eq!(EndpointMode::default(), EndpointMode::Inproc);
    }
}
