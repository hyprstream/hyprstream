//! Unified process spawner with auto-detection.
//!
//! Provides a single API for spawning processes, automatically selecting
//! the best backend (systemd if available, otherwise standalone).

use std::sync::Arc;

use super::{
    ProcessBackend, ProcessConfig, SpawnedProcess, SpawnerBackend, StandaloneBackend, SystemdBackend,
};
use crate::error::Result;

/// Unified process spawner.
///
/// Automatically detects the best backend (systemd if available, otherwise standalone)
/// and provides a consistent API for spawning and managing processes.
///
/// # Example
///
/// ```ignore
/// use hyprstream_rpc::service::spawner::{ProcessSpawner, ProcessConfig};
///
/// // Auto-detect best backend
/// let spawner = ProcessSpawner::new();
///
/// // Spawn a daemon
/// let config = ProcessConfig::new("nydusd", "/usr/bin/nydusd")
///     .args(["--config", "/etc/nydusd.json"]);
///
/// let process = spawner.spawn(config).await?;
/// println!("Spawned: {}", process.id());
///
/// // Later, stop it
/// spawner.stop(&process).await?;
/// ```
pub struct ProcessSpawner {
    /// The backend implementation.
    backend: Arc<dyn SpawnerBackend>,

    /// The backend type (for reporting).
    backend_type: ProcessBackend,
}

impl ProcessSpawner {
    /// Create a new process spawner with auto-detection.
    ///
    /// Uses systemd if available, otherwise falls back to standalone spawning.
    pub fn new() -> Self {
        if SystemdBackend::is_available() {
            tracing::info!("Using systemd backend for process spawning");
            Self {
                backend: Arc::new(SystemdBackend::new()),
                backend_type: ProcessBackend::Systemd {
                    user_mode: !nix::unistd::geteuid().is_root(),
                },
            }
        } else {
            tracing::info!("Systemd not available, using standalone backend");
            Self {
                backend: Arc::new(StandaloneBackend::new()),
                backend_type: ProcessBackend::Standalone,
            }
        }
    }

    /// Create a spawner with explicit backend selection.
    ///
    /// If systemd is requested but not available, falls back to standalone.
    pub fn with_backend(backend: ProcessBackend) -> Self {
        match backend {
            ProcessBackend::Systemd { user_mode } => {
                if SystemdBackend::is_available() {
                    Self {
                        backend: Arc::new(SystemdBackend::with_user_mode(user_mode)),
                        backend_type: ProcessBackend::Systemd { user_mode },
                    }
                } else {
                    tracing::warn!("Systemd requested but not available, falling back to standalone");
                    Self {
                        backend: Arc::new(StandaloneBackend::new()),
                        backend_type: ProcessBackend::Standalone,
                    }
                }
            }
            ProcessBackend::Standalone => Self {
                backend: Arc::new(StandaloneBackend::new()),
                backend_type: ProcessBackend::Standalone,
            },
        }
    }

    /// Create a spawner that always uses standalone spawning.
    pub fn standalone() -> Self {
        Self {
            backend: Arc::new(StandaloneBackend::new()),
            backend_type: ProcessBackend::Standalone,
        }
    }

    /// Create a spawner that uses systemd if available.
    pub fn systemd() -> Self {
        Self::with_backend(ProcessBackend::Systemd {
            user_mode: !nix::unistd::geteuid().is_root(),
        })
    }

    /// Get the backend type being used.
    pub fn backend_type(&self) -> &ProcessBackend {
        &self.backend_type
    }

    /// Check if systemd is available on this system.
    pub fn is_systemd_available() -> bool {
        SystemdBackend::is_available()
    }

    /// Spawn a daemon process.
    ///
    /// # Arguments
    /// * `config` - Configuration for the daemon
    ///
    /// # Returns
    /// A handle to the spawned process.
    pub async fn spawn(&self, config: ProcessConfig) -> Result<SpawnedProcess> {
        self.backend.spawn(config).await
    }

    /// Stop a spawned process.
    ///
    /// # Arguments
    /// * `process` - The process to stop
    pub async fn stop(&self, process: &SpawnedProcess) -> Result<()> {
        self.backend.stop(process).await
    }

    /// Check if a process is still running.
    ///
    /// # Arguments
    /// * `process` - The process to check
    ///
    /// # Returns
    /// `true` if the process is still running.
    pub async fn is_running(&self, process: &SpawnedProcess) -> Result<bool> {
        self.backend.is_running(process).await
    }
}

impl Default for ProcessSpawner {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ProcessSpawner {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
            backend_type: self.backend_type.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detection() {
        let spawner = ProcessSpawner::new();
        let backend_type = spawner.backend_type();

        // Should be either Standalone or Systemd
        match backend_type {
            ProcessBackend::Standalone => println!("Using standalone backend"),
            ProcessBackend::Systemd { user_mode } => {
                println!("Using systemd backend (user_mode: {})", user_mode)
            }
        }
    }

    #[test]
    fn test_explicit_standalone() {
        let spawner = ProcessSpawner::standalone();
        assert!(matches!(spawner.backend_type(), ProcessBackend::Standalone));
    }

    #[tokio::test]
    async fn test_spawn_and_stop() {
        let spawner = ProcessSpawner::new();

        let config = ProcessConfig::new("test-sleep", "sleep").args(["100"]);

        match spawner.spawn(config).await {
            Ok(process) => {
                // Check it's running
                let running = spawner.is_running(&process).await.unwrap();
                assert!(running, "Process should be running");

                // Stop it
                spawner.stop(&process).await.unwrap();

                // Give it a moment
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;

                // Check it's stopped
                let running = spawner.is_running(&process).await.unwrap();
                assert!(!running, "Process should be stopped");
            }
            Err(e) => {
                eprintln!("Could not spawn test process: {}", e);
            }
        }
    }
}
