//! Unix signal handling for graceful shutdown and configuration reload
//!
//! Provides a signal coordinator that handles:
//! - SIGTERM / SIGINT: Graceful shutdown
//! - SIGHUP: Configuration reload
//!
//! # Architecture
//!
//! The coordinator uses broadcast channels to notify multiple listeners
//! about signal events. This allows different parts of the application
//! to respond to shutdown or reload requests independently.
//!
//! # Example
//!
//! ```ignore
//! use hyprstream::systemd::SignalCoordinator;
//!
//! #[tokio::main]
//! async fn main() {
//!     let coordinator = SignalCoordinator::new();
//!
//!     // Get shutdown receiver before spawning
//!     let mut shutdown_rx = coordinator.shutdown_receiver();
//!
//!     // Spawn signal handler
//!     let signal_handle = tokio::spawn(async move {
//!         coordinator.run().await;
//!     });
//!
//!     // Your application logic here...
//!
//!     // Wait for shutdown signal
//!     let _ = shutdown_rx.recv().await;
//!     println!("Shutting down gracefully...");
//! }
//! ```

use tokio::sync::broadcast;
use tracing::{info, warn};

/// Signal events that can be received
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalEvent {
    /// Shutdown requested (SIGTERM, SIGINT)
    Shutdown,
    /// Configuration reload requested (SIGHUP)
    Reload,
}

/// Coordinates Unix signal handling across the application
///
/// Creates broadcast channels for shutdown and reload events that multiple
/// components can subscribe to.
pub struct SignalCoordinator {
    shutdown_tx: broadcast::Sender<()>,
    reload_tx: broadcast::Sender<()>,
}

impl SignalCoordinator {
    /// Create a new signal coordinator
    ///
    /// The coordinator doesn't start handling signals until `run()` is called.
    pub fn new() -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        let (reload_tx, _) = broadcast::channel(16); // More capacity for reload
        Self {
            shutdown_tx,
            reload_tx,
        }
    }

    /// Get a receiver for shutdown events
    ///
    /// Multiple receivers can be created. Each will receive the shutdown signal.
    pub fn shutdown_receiver(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    /// Get a receiver for reload events
    ///
    /// Multiple receivers can be created. Each will receive reload signals.
    pub fn reload_receiver(&self) -> broadcast::Receiver<()> {
        self.reload_tx.subscribe()
    }

    /// Signal shutdown to all listeners
    ///
    /// This can be called programmatically to trigger shutdown.
    pub fn signal_shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Signal reload to all listeners
    ///
    /// This can be called programmatically to trigger config reload.
    pub fn signal_reload(&self) {
        let _ = self.reload_tx.send(());
    }

    /// Check if any shutdown receivers are active
    pub fn has_shutdown_receivers(&self) -> bool {
        self.shutdown_tx.receiver_count() > 0
    }

    /// Check if any reload receivers are active
    pub fn has_reload_receivers(&self) -> bool {
        self.reload_tx.receiver_count() > 0
    }

    /// Run the signal handler
    ///
    /// This should be spawned as an async task. It will run until a shutdown
    /// signal is received.
    ///
    /// # Signals Handled
    ///
    /// - **SIGTERM**: Initiates graceful shutdown
    /// - **SIGINT**: Initiates graceful shutdown (Ctrl+C)
    /// - **SIGHUP**: Triggers configuration reload
    ///
    /// # Example
    ///
    /// ```ignore
    /// let coordinator = SignalCoordinator::new();
    /// let mut shutdown_rx = coordinator.shutdown_receiver();
    ///
    /// tokio::spawn(async move {
    ///     coordinator.run().await;
    /// });
    /// ```
    #[cfg(unix)]
    pub async fn run(self) {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to register SIGTERM handler: {}", e);
                return;
            }
        };

        let mut sigint = match signal(SignalKind::interrupt()) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to register SIGINT handler: {}", e);
                return;
            }
        };

        let mut sighup = match signal(SignalKind::hangup()) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to register SIGHUP handler: {}", e);
                return;
            }
        };

        info!("Signal coordinator started (SIGTERM, SIGINT, SIGHUP)");

        loop {
            tokio::select! {
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, initiating graceful shutdown");
                    let _ = self.shutdown_tx.send(());
                    break;
                }
                _ = sigint.recv() => {
                    info!("Received SIGINT, initiating graceful shutdown");
                    let _ = self.shutdown_tx.send(());
                    break;
                }
                _ = sighup.recv() => {
                    info!("Received SIGHUP, triggering configuration reload");
                    let _ = self.reload_tx.send(());
                    // Don't break - continue handling signals after reload
                }
            }
        }

        info!("Signal coordinator stopped");
    }

    /// Fallback for non-Unix systems
    #[cfg(not(unix))]
    pub async fn run(self) {
        use tokio::signal;

        info!("Signal coordinator started (Ctrl+C only on this platform)");

        // On non-Unix, only Ctrl+C is available
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("Received Ctrl+C, initiating graceful shutdown");
                let _ = self.shutdown_tx.send(());
            }
            Err(e) => {
                warn!("Failed to listen for Ctrl+C: {}", e);
            }
        }

        info!("Signal coordinator stopped");
    }
}

impl Default for SignalCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a simple shutdown future that completes on SIGTERM/SIGINT
///
/// This is a convenience function for simple use cases where you just
/// need to wait for shutdown.
///
/// # Example
///
/// ```ignore
/// // Start server with graceful shutdown
/// axum::serve(listener, app)
///     .with_graceful_shutdown(shutdown_signal())
///     .await?;
/// ```
#[cfg(unix)]
pub async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigterm = signal(SignalKind::terminate()).expect("Failed to register SIGTERM");
    let mut sigint = signal(SignalKind::interrupt()).expect("Failed to register SIGINT");

    tokio::select! {
        _ = sigterm.recv() => {
            info!("Received SIGTERM");
        }
        _ = sigint.recv() => {
            info!("Received SIGINT");
        }
    }
}

/// Fallback for non-Unix systems
#[cfg(not(unix))]
pub async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl+C");
    info!("Received Ctrl+C");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = SignalCoordinator::new();
        assert!(!coordinator.has_shutdown_receivers());
        assert!(!coordinator.has_reload_receivers());
    }

    #[test]
    fn test_receiver_subscription() {
        let coordinator = SignalCoordinator::new();

        let _shutdown_rx = coordinator.shutdown_receiver();
        assert!(coordinator.has_shutdown_receivers());

        let _reload_rx = coordinator.reload_receiver();
        assert!(coordinator.has_reload_receivers());
    }

    #[test]
    fn test_multiple_receivers() {
        let coordinator = SignalCoordinator::new();

        let _rx1 = coordinator.shutdown_receiver();
        let _rx2 = coordinator.shutdown_receiver();
        let _rx3 = coordinator.shutdown_receiver();

        // All three should be active
        assert!(coordinator.has_shutdown_receivers());
    }

    #[tokio::test]
    async fn test_programmatic_shutdown() {
        let coordinator = SignalCoordinator::new();
        let mut rx = coordinator.shutdown_receiver();

        // Signal shutdown
        coordinator.signal_shutdown();

        // Receiver should get the signal
        let result = rx.recv().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_programmatic_reload() {
        let coordinator = SignalCoordinator::new();
        let mut rx = coordinator.reload_receiver();

        // Signal reload
        coordinator.signal_reload();

        // Receiver should get the signal
        let result = rx.recv().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_event_enum() {
        assert_eq!(SignalEvent::Shutdown, SignalEvent::Shutdown);
        assert_ne!(SignalEvent::Shutdown, SignalEvent::Reload);
    }

    #[test]
    fn test_default_impl() {
        let coordinator = SignalCoordinator::default();
        assert!(!coordinator.has_shutdown_receivers());
    }
}
