//! Service lifecycle management
//!
//! Provides a unified trait for managing hyprstream services across different
//! platforms and init systems.

use anyhow::Result;
use async_trait::async_trait;

pub mod units;

#[cfg(feature = "systemd")]
pub mod systemd;

pub mod standalone;

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
}

/// Detect best available service manager
///
/// Returns `SystemdManager` if systemd is available, otherwise `StandaloneManager`.
pub async fn detect() -> Result<Box<dyn ServiceManager>> {
    #[cfg(feature = "systemd")]
    {
        if crate::has_systemd() {
            return Ok(Box::new(systemd::SystemdManager::new().await?));
        }
    }

    Ok(Box::new(standalone::StandaloneManager::new()))
}

// Re-exports
#[cfg(feature = "systemd")]
pub use systemd::SystemdManager;
pub use standalone::StandaloneManager;
