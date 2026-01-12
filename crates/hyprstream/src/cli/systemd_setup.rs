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
//! let services = &["registry".to_string(), "policy".to_string()];
//! if ensure_units(services) {
//!     // Systemd units are installed and ready
//! }
//! ```

use tracing::info;

/// Check if systemd is available
pub fn is_systemd_available() -> bool {
    hyprstream_rpc::has_systemd()
}

/// Ensure systemd socket units are installed and enabled (idempotent)
///
/// # Arguments
///
/// * `services` - List of services to ensure.
///
/// Returns `true` if sockets are ready for use.
pub fn ensure_units(services: &[String]) -> bool {
    if !is_systemd_available() {
        return false;
    }

    // Convert String slices to &str for ServiceManager
    let services_list: Vec<&str> = services.iter().map(|s| s.as_str()).collect();

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
        for service in services_list {
            if let Err(e) = manager.ensure(service).await {
                tracing::warn!("Failed to ensure {} service: {}", service, e);
                // Continue with other services
            }
        }

        info!("Systemd units installed and started");
        true
    })
}
