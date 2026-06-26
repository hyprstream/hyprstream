//! Service lifecycle management
//!
//! Provides a unified trait for managing hyprstream services across different
//! platforms and init systems.

use anyhow::Result;
use async_trait::async_trait;

use crate::service::spawner::{Spawnable, SpawnedService};

pub mod units;

#[cfg(feature = "systemd")]
pub mod systemd;

pub mod standalone;

/// Which D-Bus / systemd scope to target for service installation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ServiceTarget {
    /// User daemon via the systemd private bus (session-independent, default).
    ///
    /// Connects to `$XDG_RUNTIME_DIR/systemd/private/bus`. Works on headless
    /// machines and survives logout when user linger is enabled.  Correct for
    /// all long-lived public service daemons.
    #[default]
    User,

    /// User daemon via the D-Bus session bus (`$DBUS_SESSION_BUS_ADDRESS`).
    ///
    /// Session-scoped: the unit is managed by the session's systemd user
    /// instance and only while the session is active.  Use only for
    /// session-local services such as the TUI multiplexer.
    UserSession,

    /// System daemon in `/etc/systemd/system/` (requires root).
    System,
}

/// Service lifecycle management trait
///
/// Abstracts over systemd (Linux) vs standalone (process spawn) for service management.
#[async_trait]
pub trait ServiceManager: Send + Sync {
    /// Install unit files (idempotent)
    ///
    /// For systemd, this writes socket/service unit files to ~/.config/systemd/user/
    /// For standalone mode, this is a no-op.
    async fn install(&self, service: &str) -> Result<()>;

    /// Uninstall unit files
    ///
    /// Stops the service, disables units, and removes unit files.
    async fn uninstall(&self, service: &str) -> Result<()>;

    /// Start a service
    async fn start(&self, service: &str) -> Result<()>;

    /// Stop a service
    async fn stop(&self, service: &str) -> Result<()>;

    /// Check if service is running
    async fn is_active(&self, service: &str) -> Result<bool>;

    /// Reload daemon configuration
    ///
    /// For systemd, this calls `daemon-reload`.
    /// For standalone mode, this is a no-op.
    async fn reload(&self) -> Result<()>;

    /// Enable a service for autostart at boot (systemctl enable)
    ///
    /// For systemd, this calls `enable_unit_files` so the unit starts at boot.
    /// For standalone mode, this is a no-op.
    async fn enable(&self, _service: &str) -> Result<()> {
        Ok(())
    }

    /// Ensure service is available (install + start if needed)
    ///
    /// This is the main entry point for CLI commands that need services.
    async fn ensure(&self, service: &str) -> Result<()> {
        self.install(service).await?;
        if !self.is_active(service).await? {
            self.start(service).await?;
        }
        Ok(())
    }

    /// Spawn a service (unified API for inproc and systemd)
    ///
    /// For inproc mode, spawns in current process.
    /// For systemd mode, ensures systemd unit is running.
    async fn spawn(&self, service: Box<dyn Spawnable>) -> Result<SpawnedService>;
}

/// Detect best available service manager (user private bus by default).
pub async fn detect() -> Result<Box<dyn ServiceManager>> {
    detect_with_mode(ServiceTarget::User).await
}

/// Detect best available service manager for the given target scope.
pub async fn detect_with_mode(target: ServiceTarget) -> Result<Box<dyn ServiceManager>> {
    #[cfg(feature = "systemd")]
    {
        if hyprstream_rpc::has_systemd() {
            return match target {
                ServiceTarget::System => Ok(Box::new(systemd::SystemdSystemManager::new().await?)),
                ServiceTarget::UserSession => Ok(Box::new(systemd::SystemdManager::new_session().await?)),
                ServiceTarget::User => Ok(Box::new(systemd::SystemdManager::new().await?)),
            };
        }
    }
    let _ = target;
    Ok(Box::new(standalone::StandaloneManager::new()))
}

// Re-exports
#[cfg(feature = "systemd")]
pub use systemd::{SystemdManager, SystemdSystemManager};
pub use standalone::StandaloneManager;
