//! Systemd integration module
//!
//! Provides integration with systemd for:
//! - Signal handling (SIGTERM, SIGINT, SIGHUP)
//! - Service notification (sd_notify)
//! - Socket activation
//! - Health-based watchdog
//!
//! # Feature Gating
//!
//! Most functionality requires the `systemd` feature flag.
//! Signal handling works on all Unix systems.
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream::systemd::SignalCoordinator;
//!
//! // Create coordinator
//! let coordinator = SignalCoordinator::new();
//!
//! // Get receivers for different signal types
//! let mut shutdown_rx = coordinator.shutdown_receiver();
//! let mut reload_rx = coordinator.reload_receiver();
//!
//! // Spawn signal handler
//! tokio::spawn(coordinator.run());
//!
//! // Wait for shutdown
//! shutdown_rx.recv().await;
//! ```

pub mod signals;

#[cfg(feature = "systemd")]
pub mod notify;

#[cfg(feature = "systemd")]
pub mod health;

#[cfg(feature = "systemd")]
pub mod socket;

// Re-export main types
pub use signals::SignalCoordinator;

#[cfg(feature = "systemd")]
pub use notify::SystemdNotifier;
