//! Systemd unit installation for socket activation
//!
//! This module provides automatic installation of systemd user units
//! for the service socket activation pattern.
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream::cli::systemd_setup::ensure_units;
//!
//! if ensure_units() {
//!     // Systemd units are installed and ready
//! }
//! ```

use tracing::info;

/// Services to install units for
const SERVICES: &[&str] = &["registry", "policy", "worker", "event"];

/// Check if systemd is available
pub fn is_systemd_available() -> bool {
    hyprstream_rpc::is_systemd_booted()
}

/// Ensure systemd socket units are installed and enabled (idempotent)
///
/// Returns `true` if sockets are ready for use.
pub fn ensure_units() -> bool {
    if !is_systemd_available() {
        return false;
    }

    // Create a runtime for async ServiceManager operations
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            tracing::warn!("Failed to create runtime for systemd setup: {}", e);
            return false;
        }
    };

    rt.block_on(async {
        let manager = match hyprstream_rpc::detect_service_manager().await {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Failed to detect service manager: {}", e);
                return false;
            }
        };

        // Install and start all services
        for service in SERVICES {
            if let Err(e) = manager.ensure(service).await {
                tracing::warn!("Failed to ensure {} service: {}", service, e);
                // Continue with other services
            }
        }

        info!("Systemd units installed and started");
        true
    })
}
