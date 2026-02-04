//! Systemd socket activation for HTTP listeners
//!
//! Provides transparent socket activation support - if systemd passes a
//! pre-bound socket via `LISTEN_FDS`, we use it; otherwise we bind normally.
//!
//! # How Socket Activation Works
//!
//! 1. Systemd creates and binds the socket before starting the service
//! 2. Systemd sets `LISTEN_FDS=1` and `LISTEN_PID=<pid>` environment variables
//! 3. The service receives the socket as file descriptor 3 (SD_LISTEN_FDS_START)
//! 4. The service uses the pre-bound socket instead of binding itself
//!
//! # Benefits
//!
//! - Zero-downtime restarts (systemd holds the socket during restart)
//! - Faster startup (socket already bound)
//! - Privilege separation (systemd can bind privileged ports)
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_core::systemd::socket::get_listener;
//!
//! // Get listener - uses systemd socket if available, otherwise binds normally
//! let listener = get_listener("0.0.0.0:3000").await?;
//!
//! // Use with axum
//! axum::serve(listener, app).await?;
//! ```
//!
//! # Systemd Configuration
//!
//! ```ini
//! # hyprstream.socket
//! [Socket]
//! ListenStream=0.0.0.0:3000
//! ReusePort=yes
//!
//! [Install]
//! WantedBy=sockets.target
//! ```

use anyhow::{Context, Result};
use listenfd::ListenFd;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tracing::{debug, info};

/// Get a TCP listener, using systemd socket activation if available
///
/// This function transparently handles socket activation:
/// - If `LISTEN_FDS` is set and valid, uses the systemd-provided socket
/// - Otherwise, binds to the specified address normally
///
/// # Arguments
///
/// * `addr` - Address to bind to if socket activation is not available
///
/// # Returns
///
/// A `TcpListener` ready for accepting connections.
///
/// # Example
///
/// ```ignore
/// let listener = get_listener("0.0.0.0:3000").await?;
/// info!("Server listening on {}", listener.local_addr()?);
/// ```
pub async fn get_listener(addr: &str) -> Result<TcpListener> {
    // Try socket activation first
    if let Some(listener) = try_systemd()? {
        let local_addr = listener
            .local_addr()
            .context("Failed to get local address from systemd socket")?;
        info!(
            "Using systemd socket activation (listening on {})",
            local_addr
        );
        return Ok(listener);
    }

    // Fall back to normal binding
    debug!("Socket activation not available, binding to {}", addr);
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("Failed to bind to {addr}"))?;

    let local_addr = listener.local_addr()?;
    info!("Server listening on {} (direct bind)", local_addr);

    Ok(listener)
}

/// Get a TCP listener for a specific socket address
///
/// Same as `get_listener` but takes a `SocketAddr` directly.
pub async fn get_listener_addr(addr: SocketAddr) -> Result<TcpListener> {
    get_listener(&addr.to_string()).await
}

/// Try to get a listener from systemd socket activation
///
/// Returns `Ok(Some(listener))` if socket activation is available,
/// `Ok(None)` if not running under systemd or no sockets passed,
/// `Err` if socket activation is configured but failed.
fn try_systemd() -> Result<Option<TcpListener>> {
    // Check if we're running under systemd with socket activation
    if !has_socket() {
        debug!("Not running with systemd socket activation");
        return Ok(None);
    }

    let mut listenfd = ListenFd::from_env();

    // Get the first TCP listener (index 0)
    match listenfd.take_tcp_listener(0) {
        Ok(Some(std_listener)) => {
            // Set non-blocking for tokio
            std_listener
                .set_nonblocking(true)
                .context("Failed to set socket non-blocking")?;

            // Convert to tokio TcpListener
            let listener = TcpListener::from_std(std_listener)
                .context("Failed to convert std TcpListener to tokio")?;

            Ok(Some(listener))
        }
        Ok(None) => {
            debug!("No TCP listeners passed by systemd");
            Ok(None)
        }
        Err(e) => {
            // listenfd error - socket activation configured but failed
            Err(anyhow::anyhow!(
                "Failed to get TCP listener from systemd: {}",
                e
            ))
        }
    }
}

/// Check if running with systemd socket activation
///
/// Returns `true` if `LISTEN_FDS` is set and `LISTEN_PID` matches current process.
pub fn has_socket() -> bool {
    let listen_fds: i32 = std::env::var("LISTEN_FDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let listen_pid: u32 = std::env::var("LISTEN_PID")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let current_pid = std::process::id();

    let activated = listen_fds > 0 && listen_pid == current_pid;

    if activated {
        debug!(
            "Socket activation detected: LISTEN_FDS={}, LISTEN_PID={}, current_pid={}",
            listen_fds, listen_pid, current_pid
        );
    }

    activated
}

/// Get the number of file descriptors passed by systemd
///
/// Returns 0 if not running under systemd socket activation.
pub fn listen_fd_count() -> usize {
    if !has_socket() {
        return 0;
    }

    std::env::var("LISTEN_FDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Get all TCP listeners passed by systemd
///
/// Returns a vector of TCP listeners. Useful when systemd passes multiple
/// sockets (e.g., for HTTP and HTTPS on different ports).
///
/// # Example
///
/// ```ignore
/// let listeners = get_all_listeners().await?;
/// for (i, listener) in listeners.iter().enumerate() {
///     info!("Socket {}: {}", i, listener.local_addr()?);
/// }
/// ```
pub fn get_all_tcp_listeners() -> Result<Vec<TcpListener>> {
    if !has_socket() {
        return Ok(Vec::new());
    }

    let mut listenfd = ListenFd::from_env();
    let mut listeners = Vec::new();
    let mut index = 0;

    loop {
        match listenfd.take_tcp_listener(index) {
            Ok(Some(std_listener)) => {
                std_listener
                    .set_nonblocking(true)
                    .with_context(|| format!("Failed to set socket {index} non-blocking"))?;

                let listener = TcpListener::from_std(std_listener)
                    .with_context(|| format!("Failed to convert socket {index} to tokio"))?;

                listeners.push(listener);
                index += 1;
            }
            Ok(None) => break,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to get TCP listener {} from systemd: {}",
                    index,
                    e
                ));
            }
        }
    }

    Ok(listeners)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;

    #[test]
    fn test_has_socket_false() {
        // Should return false when env vars not set
        std::env::remove_var("LISTEN_FDS");
        std::env::remove_var("LISTEN_PID");
        assert!(!has_socket());
    }

    #[test]
    fn test_has_socket_wrong_pid() {
        // Should return false when PID doesn't match
        std::env::set_var("LISTEN_FDS", "1");
        std::env::set_var("LISTEN_PID", "99999999"); // Wrong PID
        assert!(!has_socket());

        // Cleanup
        std::env::remove_var("LISTEN_FDS");
        std::env::remove_var("LISTEN_PID");
    }

    #[test]
    fn test_listen_fd_count_zero() {
        std::env::remove_var("LISTEN_FDS");
        std::env::remove_var("LISTEN_PID");
        assert_eq!(listen_fd_count(), 0);
    }

    #[tokio::test]
    async fn test_get_listener_fallback() {
        // Without socket activation, should bind normally
        std::env::remove_var("LISTEN_FDS");
        std::env::remove_var("LISTEN_PID");

        // Use port 0 to get a random available port
        let listener = get_listener("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        assert!(addr.port() > 0);
    }

    #[test]
    fn test_get_all_tcp_listeners_empty() {
        std::env::remove_var("LISTEN_FDS");
        std::env::remove_var("LISTEN_PID");

        let listeners = get_all_tcp_listeners().unwrap();
        assert!(listeners.is_empty());
    }
}
