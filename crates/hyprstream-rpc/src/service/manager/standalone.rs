//! Standalone service manager implementation
//!
//! Manages services by spawning processes via ProcessSpawner.
//! Uses systemd-run when available, otherwise direct process spawning.

use super::ServiceManager;
use crate::service::spawner::{ProcessConfig, ProcessSpawner, SpawnedProcess, Spawnable, SpawnedService};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Standalone service manager
///
/// Spawns services via ProcessSpawner, which auto-detects systemd-run availability.
/// Used when systemd socket activation is not available or not desired.
pub struct StandaloneManager {
    spawner: ProcessSpawner,
    processes: Mutex<HashMap<String, SpawnedProcess>>,
}

impl StandaloneManager {
    /// Create a new StandaloneManager
    pub fn new() -> Self {
        Self {
            spawner: ProcessSpawner::new(),
            processes: Mutex::new(HashMap::new()),
        }
    }

    /// Create with explicit ProcessSpawner
    pub fn with_spawner(spawner: ProcessSpawner) -> Self {
        Self {
            spawner,
            processes: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for StandaloneManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ServiceManager for StandaloneManager {
    async fn install(&self, _service: &str) -> Result<()> {
        // No persistent units for standalone mode
        Ok(())
    }

    async fn start(&self, service: &str) -> Result<()> {
        let exe = std::env::current_exe()?;

        let config = ProcessConfig::new(service, exe)
            .args(["service", service, "--ipc"]);

        let process = self.spawner.spawn(config).await?;

        info!(
            "Started service {} (id: {}, backend: {})",
            service,
            process.id(),
            self.spawner.backend_type().backend_name()
        );

        self.processes
            .lock()
            .await
            .insert(service.to_string(), process);

        Ok(())
    }

    async fn stop(&self, service: &str) -> Result<()> {
        if let Some(process) = self.processes.lock().await.remove(service) {
            self.spawner.stop(&process).await?;
            info!("Stopped service {} (id: {})", service, process.id());
        } else {
            debug!("Service {} not running", service);
        }
        Ok(())
    }

    async fn is_active(&self, service: &str) -> Result<bool> {
        let processes = self.processes.lock().await;
        if let Some(process) = processes.get(service) {
            Ok(self.spawner.is_running(process).await?)
        } else {
            Ok(false)
        }
    }

    async fn reload(&self) -> Result<()> {
        // No-op for standalone mode
        Ok(())
    }

    async fn uninstall(&self, service: &str) -> Result<()> {
        self.stop(service).await
    }

    async fn spawn(&self, service: Box<dyn Spawnable>) -> Result<SpawnedService> {
        // For standalone mode, spawn as subprocess
        let exe = std::env::current_exe()?;

        let config = ProcessConfig::new(service.name(), exe)
            .args(["service", service.name(), "--ipc"]);

        let process = self.spawner.spawn(config).await?;

        info!(
            "Spawned service {} (id: {}, backend: {})",
            service.name(),
            process.id(),
            self.spawner.backend_type().backend_name()
        );

        self.processes
            .lock()
            .await
            .insert(service.name().to_string(), process.clone());

        Ok(SpawnedService::subprocess(
            process.id.clone(),
            process,
            crate::paths::service_pid_file(service.name()),
        ))
    }
}

/// Extension trait for ProcessBackend to get display name
trait ProcessBackendExt {
    fn backend_name(&self) -> &'static str;
}

impl ProcessBackendExt for crate::service::spawner::ProcessBackend {
    fn backend_name(&self) -> &'static str {
        match self {
            Self::Standalone => "standalone",
            Self::Systemd { .. } => "systemd-run",
        }
    }
}
