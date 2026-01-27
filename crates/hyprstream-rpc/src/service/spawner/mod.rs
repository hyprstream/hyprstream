//! Process and service spawner abstraction.
//!
//! Provides unified interfaces for spawning:
//! - Raw processes via `ProcessSpawner` (Standalone or Systemd backends)
//! - ZMQ services via `ServiceSpawner` (Tokio, Thread, or Subprocess modes)
//!
//! # Architecture
//!
//! ```text
//! ProcessSpawner (raw process management)
//!     ├── StandaloneBackend (tokio::process::Command)
//!     │   └── .kill_on_drop(true) for cleanup
//!     │
//!     └── SystemdBackend (systemd-run)
//!         └── Transient units in hyprstream-workers.slice
//!
//! ServiceSpawner (ZmqService hosting)
//!     ├── Tokio mode (tokio::spawn)
//!     ├── Thread mode (std::thread + runtime)
//!     └── Subprocess mode (ProcessSpawner)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_rpc::service::spawner::{ProcessSpawner, ProcessConfig};
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

mod process;
mod service;
mod standalone;
mod systemd;

pub use standalone::StandaloneBackend;
pub use process::ProcessSpawner;
pub use service::{
    ProxyService, ServiceKind, ServiceMode,
    ServiceSpawner, Spawnable, SpawnedService,
    InprocManager,
};
pub use systemd::SystemdBackend;

use std::path::PathBuf;

use crate::error::Result;

/// Configuration for spawning a daemon process.
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    /// Human-readable name for the daemon (used in unit names).
    pub name: String,

    /// Path to the executable.
    pub executable: PathBuf,

    /// Command-line arguments.
    pub args: Vec<String>,

    /// Working directory.
    pub working_dir: Option<PathBuf>,

    /// Environment variables (key=value).
    pub env: Vec<(String, String)>,

    /// Memory limit (systemd only, e.g., "1G", "512M").
    pub memory_limit: Option<String>,

    /// CPU quota percentage (systemd only, e.g., 200 for 2 cores).
    pub cpu_quota: Option<u32>,

    /// Custom unit properties (systemd only).
    pub unit_properties: Vec<(String, String)>,

    /// Whether to restart on failure (systemd only).
    pub restart_on_failure: bool,
}

impl ProcessConfig {
    /// Create a new process configuration.
    pub fn new(name: impl Into<String>, executable: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            executable: executable.into(),
            args: Vec::new(),
            working_dir: None,
            env: Vec::new(),
            memory_limit: None,
            cpu_quota: None,
            unit_properties: Vec::new(),
            restart_on_failure: false,
        }
    }

    /// Set command-line arguments.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Set working directory.
    pub fn working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Add an environment variable.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Set memory limit (e.g., "1G", "512M").
    pub fn memory_limit(mut self, limit: impl Into<String>) -> Self {
        self.memory_limit = Some(limit.into());
        self
    }

    /// Set CPU quota percentage (e.g., 200 for 2 cores).
    pub fn cpu_quota(mut self, quota: u32) -> Self {
        self.cpu_quota = Some(quota);
        self
    }

    /// Add a custom systemd unit property.
    pub fn unit_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.unit_properties.push((key.into(), value.into()));
        self
    }

    /// Enable restart on failure.
    pub fn restart_on_failure(mut self) -> Self {
        self.restart_on_failure = true;
        self
    }
}

/// Represents a spawned process.
#[derive(Debug, Clone)]
pub struct SpawnedProcess {
    /// Unique identifier for the process.
    pub id: String,

    /// Process kind (direct or systemd).
    pub kind: ProcessKind,
}

impl SpawnedProcess {
    /// Create a new spawned process reference.
    pub fn new(id: impl Into<String>, kind: ProcessKind) -> Self {
        Self {
            id: id.into(),
            kind,
        }
    }

    /// Get the process ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Check if this is a direct process.
    pub fn is_direct(&self) -> bool {
        matches!(self.kind, ProcessKind::Direct(_))
    }

    /// Check if this is a systemd unit.
    pub fn is_systemd(&self) -> bool {
        matches!(self.kind, ProcessKind::SystemdUnit(_))
    }

    /// Get the PID if this is a direct process.
    pub fn pid(&self) -> Option<u32> {
        match &self.kind {
            ProcessKind::Direct(pid) => Some(*pid),
            ProcessKind::SystemdUnit(_) => None,
        }
    }

    /// Get the unit name if this is a systemd unit.
    pub fn unit_name(&self) -> Option<&str> {
        match &self.kind {
            ProcessKind::Direct(_) => None,
            ProcessKind::SystemdUnit(name) => Some(name),
        }
    }
}

/// Type of spawned process.
#[derive(Debug, Clone)]
pub enum ProcessKind {
    /// Direct process (tracked by PID).
    Direct(u32),

    /// Systemd transient unit (tracked by unit name).
    SystemdUnit(String),
}

/// Backend type for process spawning.
#[derive(Debug, Clone)]
pub enum ProcessBackend {
    /// Standalone process spawning via tokio::process::Command.
    Standalone,

    /// Systemd transient units via systemd-run.
    Systemd {
        /// Whether to use user mode (--user flag).
        user_mode: bool,
    },
}

/// Trait for process spawning backends.
#[async_trait::async_trait]
pub trait SpawnerBackend: Send + Sync {
    /// Spawn a daemon process.
    async fn spawn(&self, config: ProcessConfig) -> Result<SpawnedProcess>;

    /// Stop a spawned process.
    async fn stop(&self, process: &SpawnedProcess) -> Result<()>;

    /// Check if a process is still running.
    async fn is_running(&self, process: &SpawnedProcess) -> Result<bool>;

    /// Get the backend type name.
    fn backend_type(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_config_builder() {
        let config = ProcessConfig::new("test", "/usr/bin/test")
            .args(["--foo", "bar"])
            .working_dir("/tmp")
            .env("KEY", "VALUE")
            .memory_limit("1G")
            .cpu_quota(200)
            .restart_on_failure();

        assert_eq!(config.name, "test");
        assert_eq!(config.executable.to_str().unwrap(), "/usr/bin/test");
        assert_eq!(config.args, vec!["--foo", "bar"]);
        assert_eq!(config.working_dir.unwrap().to_str().unwrap(), "/tmp");
        assert_eq!(config.env, vec![("KEY".to_owned(), "VALUE".to_owned())]);
        assert_eq!(config.memory_limit, Some("1G".to_owned()));
        assert_eq!(config.cpu_quota, Some(200));
        assert!(config.restart_on_failure);
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
            ProcessKind::SystemdUnit("test.service".to_owned()),
        );
        assert!(!systemd.is_direct());
        assert!(systemd.is_systemd());
        assert_eq!(systemd.pid(), None);
        assert_eq!(systemd.unit_name(), Some("test.service"));
    }
}
