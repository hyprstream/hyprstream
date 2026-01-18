//! Worker spawner abstraction for process management
//!
//! This module re-exports the spawner infrastructure from `hyprstream-rpc`
//! and provides compatibility aliases for backward compatibility.
//!
//! # Architecture
//!
//! ```text
//! ProcessSpawner (wrapper with auto-detection)
//!     ├── StandaloneBackend
//!     │   └── tokio::process::Command
//!     │       └── .kill_on_drop(true) for cleanup
//!     │
//!     └── SystemdBackend
//!         └── systemd-run --user (or system)
//!             └── Spawns as transient unit in hyprstream-workers.slice
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_workers::runtime::spawner::{ProcessSpawner, ProcessConfig};
//!
//! // Auto-detect best backend (systemd if available)
//! let spawner = ProcessSpawner::new();
//!
//! // Spawn a daemon
//! let config = ProcessConfig::new("nydusd", "/usr/bin/nydusd")
//!     .args(["--config", "/etc/nydusd.json"])
//!     .working_dir("/var/run/hyprstream");
//!
//! let process = spawner.spawn(config).await?;
//! println!("Spawned: {:?}", process.id());
//!
//! // Later, stop the daemon
//! spawner.stop(&process).await?;
//! ```

// Re-export all spawner types from hyprstream-rpc
pub use hyprstream_rpc::service::spawner::{
    // Core types
    StandaloneBackend,
    ProcessBackend,
    ProcessConfig,
    ProcessKind,
    ProcessSpawner,
    SpawnedProcess,
    SpawnerBackend,
    SystemdBackend,
    // ServiceSpawner types
    ProxyService,
    ServiceKind,
    ServiceMode,
    ServiceSpawner,
    Spawnable,
    SpawnedService,
};

// Compatibility aliases for legacy code
// (These match the original hyprstream-workers naming)

/// Alias for `ProcessConfig` (backward compatibility)
pub type DaemonConfig = ProcessConfig;

/// Alias for `SpawnerBackend` (backward compatibility)
pub trait WorkerSpawner: SpawnerBackend {
    /// Spawn a daemon process (alias for `spawn`)
    fn spawn_daemon(
        &self,
        config: ProcessConfig,
    ) -> impl std::future::Future<Output = hyprstream_rpc::Result<SpawnedProcess>> + Send {
        self.spawn(config)
    }

    /// Check if systemd is available
    fn is_systemd_available(&self) -> bool {
        self.backend_type() == "systemd"
    }

    /// Get the spawner type name
    fn spawner_type(&self) -> &'static str {
        self.backend_type()
    }
}

// Blanket implementation for all SpawnerBackend implementations
impl<T: SpawnerBackend> WorkerSpawner for T {}

/// Create the appropriate spawner based on system capabilities.
///
/// Returns `ProcessSpawner` with auto-detected backend
/// (systemd if available, otherwise direct).
pub fn create_spawner() -> ProcessSpawner {
    ProcessSpawner::new()
}

/// Create a spawner with explicit type selection.
pub fn create_spawner_of_type(use_systemd: bool) -> ProcessSpawner {
    if use_systemd {
        ProcessSpawner::systemd()
    } else {
        ProcessSpawner::standalone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_config_alias() {
        // DaemonConfig should work as alias for ProcessConfig
        let config = DaemonConfig::new("test", "/usr/bin/test")
            .args(["--foo", "bar"])
            .working_dir("/tmp")
            .env("KEY", "VALUE")
            .memory_limit("1G")
            .cpu_quota(200)
            .restart_on_failure();

        assert_eq!(config.name, "test");
        assert_eq!(config.executable.to_str().unwrap(), "/usr/bin/test");
        assert_eq!(config.args, vec!["--foo", "bar"]);
    }

    #[test]
    fn test_create_spawner() {
        let spawner = create_spawner();
        let backend_type = spawner.backend_type();
        // Should be either Standalone or Systemd
        match backend_type {
            ProcessBackend::Standalone => {}
            ProcessBackend::Systemd { .. } => {}
        }
    }

    #[test]
    fn test_spawned_process() {
        let direct = SpawnedProcess::new("test-1", ProcessKind::Direct(1234));
        assert!(direct.is_direct());
        assert!(!direct.is_systemd());
        assert_eq!(direct.pid(), Some(1234));
        assert_eq!(direct.unit_name(), None);

        let systemd = SpawnedProcess::new(
            "test-2",
            ProcessKind::SystemdUnit("test.service".to_string()),
        );
        assert!(!systemd.is_direct());
        assert!(systemd.is_systemd());
        assert_eq!(systemd.pid(), None);
        assert_eq!(systemd.unit_name(), Some("test.service"));
    }
}
