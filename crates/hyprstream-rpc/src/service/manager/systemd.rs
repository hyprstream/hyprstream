//! Systemd service manager implementation
//!
//! Uses D-Bus to control systemd units via zbus_systemd.

use super::{units, ServiceManager};
use crate::service::spawner::{Spawnable, SpawnedService};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::path::PathBuf;
use tracing::{debug, info};
use zbus_systemd::login1::ManagerProxy as LoginProxy;
use zbus_systemd::systemd1::ManagerProxy;
use zbus_systemd::zbus::Connection;

/// Systemd-based service manager
///
/// Manages services via D-Bus calls to systemd.
pub struct SystemdManager {
    #[allow(dead_code)]
    connection: Connection,
    systemd: ManagerProxy<'static>,
    #[allow(dead_code)]
    login: LoginProxy<'static>,
}

impl SystemdManager {
    /// Create a new SystemdManager
    ///
    /// Connects to the user session D-Bus and enables user lingering
    /// so services persist after logout.
    pub async fn new() -> Result<Self> {
        let connection = Connection::session().await?;
        let systemd = ManagerProxy::new(&connection).await?;
        let login = LoginProxy::new(&connection).await?;

        // Enable linger (services persist after logout)
        let uid = nix::unistd::getuid().as_raw();
        if let Err(e) = login.set_user_linger(uid, true, false).await {
            debug!("Failed to enable user linger (may already be enabled): {}", e);
        } else {
            debug!("User lingering enabled");
        }

        Ok(Self {
            connection,
            systemd,
            login,
        })
    }

    #[allow(dead_code)]
    fn socket_unit(service: &str) -> String {
        format!("hyprstream-{}.socket", service)
    }

    fn service_unit(service: &str) -> String {
        format!("hyprstream-{}.service", service)
    }

    fn units_dir() -> Result<PathBuf> {
        dirs::config_dir()
            .map(|d| d.join("systemd/user"))
            .ok_or_else(|| anyhow!("cannot determine config directory"))
    }
}

#[async_trait]
impl ServiceManager for SystemdManager {
    async fn install(&self, service: &str) -> Result<()> {
        let units_dir = Self::units_dir()?;
        std::fs::create_dir_all(&units_dir)?;

        let service_content = units::service_unit(service)?;
        let service_path = units_dir.join(Self::service_unit(service));

        // Write service unit if changed (idempotent)
        if std::fs::read_to_string(&service_path).ok().as_deref() != Some(&service_content) {
            std::fs::write(&service_path, &service_content)?;
            info!("Installed service unit: {}", service_path.display());
            self.reload().await?;
        }

        Ok(())
    }

    async fn start(&self, service: &str) -> Result<()> {
        self.systemd
            .start_unit(Self::service_unit(service), "replace".into())
            .await?;
        info!("Started service unit: {}", Self::service_unit(service));
        Ok(())
    }

    async fn stop(&self, service: &str) -> Result<()> {
        self.systemd
            .stop_unit(Self::service_unit(service), "replace".into())
            .await?;
        info!("Stopped service unit: {}", Self::service_unit(service));
        Ok(())
    }

    async fn is_active(&self, service: &str) -> Result<bool> {
        // Check if the SERVICE unit is active, not the socket unit
        match self.systemd.get_unit(Self::service_unit(service)).await {
            Ok(unit_path) => {
                // Query the unit's ActiveState property via D-Bus
                let unit = zbus_systemd::systemd1::UnitProxy::builder(&self.systemd.inner().connection())
                    .path(unit_path)?
                    .build()
                    .await?;
                let state = unit.active_state().await?;
                Ok(state == "active")
            }
            Err(_) => Ok(false),
        }
    }

    async fn reload(&self) -> Result<()> {
        self.systemd.reload().await?;
        debug!("Reloaded systemd daemon");
        Ok(())
    }

    async fn uninstall(&self, service: &str) -> Result<()> {
        // Stop the service first
        let _ = self.stop(service).await;

        // Remove service unit file
        let units_dir = Self::units_dir()?;
        let service_path = units_dir.join(Self::service_unit(service));

        if service_path.exists() {
            std::fs::remove_file(&service_path)?;
            info!("Removed: {}", service_path.display());
        }

        self.reload().await
    }

    async fn spawn(&self, service: Box<dyn Spawnable>) -> Result<SpawnedService> {
        // Call existing ensure() to install and start systemd unit
        self.ensure(service.name()).await?;
        Ok(SpawnedService::dummy())
    }
}
