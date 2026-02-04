//! Systemd notification support
//!
//! Provides a wrapper for communicating with systemd:
//! - Ready notification (`READY=1`)
//! - Stopping notification (`STOPPING=1`)
//! - Status updates (shown in `systemctl status`)
//! - Watchdog pings (`WATCHDOG=1`)
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_core::systemd::SystemdNotifier;
//!
//! // Signal service is ready
//! SystemdNotifier::ready();
//!
//! // Update status (shown in systemctl status)
//! SystemdNotifier::status("Processing 42 requests");
//!
//! // Signal stopping
//! SystemdNotifier::stopping();
//! ```
//!
//! # Watchdog Integration
//!
//! For services with `WatchdogSec=` configured:
//!
//! ```ignore
//! // Check if watchdog is enabled and get ping interval
//! if let Some(interval) = SystemdNotifier::watchdog_interval() {
//!     // Ping watchdog at the recommended interval (half of WatchdogSec)
//!     loop {
//!         tokio::time::sleep(interval).await;
//!         if is_healthy().await {
//!             SystemdNotifier::watchdog();
//!         }
//!         // If unhealthy, don't ping - systemd will restart us
//!     }
//! }
//! ```

use std::time::Duration;
use tracing::{debug, info, trace};

/// Systemd notification wrapper
///
/// All methods are no-ops if not running under systemd (no NOTIFY_SOCKET).
pub struct SystemdNotifier;

impl SystemdNotifier {
    /// Signal that the service is ready to accept connections
    ///
    /// Sends `READY=1` to systemd. For `Type=notify` services, this must be
    /// called after the service has finished initialization.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // After binding listener and starting server
    /// SystemdNotifier::ready();
    /// ```
    pub fn ready() {
        if Self::is_systemd_managed() {
            info!("Notifying systemd: READY=1");
            let _ = hyprstream_rpc::notify::ready();
        } else {
            debug!("Not running under systemd, skipping READY notification");
        }
    }

    /// Signal that the service is stopping
    ///
    /// Sends `STOPPING=1` to systemd. This should be called when starting
    /// graceful shutdown.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // When shutdown signal received
    /// SystemdNotifier::stopping();
    /// // Then perform graceful shutdown
    /// ```
    pub fn stopping() {
        if Self::is_systemd_managed() {
            info!("Notifying systemd: STOPPING=1");
            let _ = hyprstream_rpc::notify::stopping();
        } else {
            debug!("Not running under systemd, skipping STOPPING notification");
        }
    }

    /// Update the status message shown in `systemctl status`
    ///
    /// Sends `STATUS=<msg>` to systemd. This can be called periodically
    /// to show useful information in the service status.
    ///
    /// # Arguments
    ///
    /// * `msg` - Status message to display
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Show current activity
    /// SystemdNotifier::status("Loaded 3 models, serving requests");
    /// SystemdNotifier::status("Processing 42 active connections");
    /// ```
    pub fn status(msg: &str) {
        if Self::is_systemd_managed() {
            debug!("Notifying systemd: STATUS={}", msg);
            let _ = hyprstream_rpc::notify::status(msg);
        }
    }

    /// Ping the watchdog timer
    ///
    /// Sends `WATCHDOG=1` to systemd. This should ONLY be called when the
    /// service is healthy. If the service becomes unhealthy, stop pinging
    /// and let systemd restart it after `WatchdogSec` timeout.
    ///
    /// # Important
    ///
    /// Only call this when health checks pass. Pinging while unhealthy
    /// defeats the purpose of the watchdog.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if health_check_passed().await {
    ///     SystemdNotifier::watchdog();
    /// }
    /// // If health check fails, don't ping - systemd will restart
    /// ```
    pub fn watchdog() {
        if Self::is_systemd_managed() {
            trace!("Notifying systemd: WATCHDOG=1");
            let _ = hyprstream_rpc::notify::watchdog();
        }
    }

    /// Get the watchdog ping interval
    ///
    /// Returns the recommended interval for pinging the watchdog, which is
    /// half of the `WatchdogSec` value configured in the unit file.
    ///
    /// # Returns
    ///
    /// `Some(duration)` if watchdog is enabled, `None` if not configured
    /// or not running under systemd.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(interval) = SystemdNotifier::watchdog_interval() {
    ///     info!("Watchdog enabled, pinging every {:?}", interval);
    ///     // Start watchdog task
    /// } else {
    ///     info!("Watchdog not configured");
    /// }
    /// ```
    pub fn watchdog_interval() -> Option<Duration> {
        std::env::var("WATCHDOG_USEC")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|usec| {
                // Ping at half the watchdog interval for safety margin
                Duration::from_micros(usec / 2)
            })
    }

    /// Check if running under systemd management
    ///
    /// Returns `true` if `NOTIFY_SOCKET` environment variable is set,
    /// indicating the service was started by systemd with notification
    /// support enabled.
    pub fn is_systemd_managed() -> bool {
        std::env::var("NOTIFY_SOCKET").is_ok()
    }

    /// Get the PID that systemd expects for notifications
    ///
    /// Returns the PID that should be sending notifications. For most
    /// services this is the main PID (returned by `getpid()`).
    pub fn expected_pid() -> Option<u32> {
        std::env::var("WATCHDOG_PID")
            .or_else(|_| std::env::var("MAINPID"))
            .ok()
            .and_then(|v| v.parse().ok())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;

    #[test]
    fn test_is_systemd_managed() {
        // Should return false when not running under systemd
        // (NOTIFY_SOCKET not set in test environment)
        let managed = SystemdNotifier::is_systemd_managed();
        // This may be true or false depending on test environment
        println!("Running under systemd: {}", managed);
    }

    #[test]
    fn test_watchdog_interval_not_set() {
        // Should return None when WATCHDOG_USEC is not set
        std::env::remove_var("WATCHDOG_USEC");
        assert!(SystemdNotifier::watchdog_interval().is_none());
    }

    #[test]
    fn test_ready_no_panic() {
        // Should not panic even when not running under systemd
        SystemdNotifier::ready();
    }

    #[test]
    fn test_status_no_panic() {
        // Should not panic even when not running under systemd
        SystemdNotifier::status("test status");
    }

    #[test]
    fn test_stopping_no_panic() {
        // Should not panic even when not running under systemd
        SystemdNotifier::stopping();
    }

    #[test]
    fn test_watchdog_no_panic() {
        // Should not panic even when not running under systemd
        SystemdNotifier::watchdog();
    }
}
