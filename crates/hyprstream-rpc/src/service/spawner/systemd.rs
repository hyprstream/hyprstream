//! Systemd spawner using transient units.
//!
//! Spawns daemon processes as systemd transient units via `systemd-run`.
//! Provides better resource isolation, automatic restart, and centralized logging.
//!
//! # Features
//!
//! - Spawns processes in `hyprstream-workers.slice` for resource isolation
//! - Supports memory limits and CPU quotas
//! - Automatic cleanup via `CollectMode=inactive-or-failed`
//! - Works in both user and system mode
//!
//! # Unit Naming
//!
//! Units are named: `hyprstream-{name}-{uuid}.service`
//!
//! Example: `hyprstream-nydusd-abc123.service`

use std::process::Stdio;

use tokio::process::Command;
use uuid::Uuid;

use super::{ProcessConfig, ProcessKind, SpawnedProcess, SpawnerBackend};
use crate::error::{Result, RpcError};

/// Systemd spawner backend using transient units.
///
/// Uses `systemd-run` to spawn processes as transient systemd units.
/// Falls back to user mode (`--user`) if not running as root.
pub struct SystemdBackend {
    /// Whether to use user mode (--user flag).
    user_mode: bool,

    /// Slice for worker processes.
    slice: String,
}

impl SystemdBackend {
    /// Create a new systemd backend.
    ///
    /// Automatically detects whether to use user or system mode based on UID.
    pub fn new() -> Self {
        let user_mode = !nix::unistd::geteuid().is_root();

        Self {
            user_mode,
            slice: "hyprstream-workers.slice".to_owned(),
        }
    }

    /// Create a backend with explicit mode selection.
    pub fn with_user_mode(user_mode: bool) -> Self {
        Self {
            user_mode,
            slice: "hyprstream-workers.slice".to_owned(),
        }
    }

    /// Set the slice for spawned units.
    pub fn with_slice(mut self, slice: impl Into<String>) -> Self {
        self.slice = slice.into();
        self
    }

    /// Check if systemd-run is available.
    pub fn is_available() -> bool {
        // Check if we're running under systemd
        let under_systemd = std::path::Path::new("/run/systemd/system").exists()
            || std::env::var("NOTIFY_SOCKET").is_ok();

        if !under_systemd {
            return false;
        }

        // Check if systemd-run is available
        which::which("systemd-run").is_ok()
    }

    /// Generate a unique unit name.
    fn generate_unit_name(&self, name: &str) -> String {
        let short_uuid = &Uuid::new_v4().to_string()[..8];
        format!("hyprstream-{name}-{short_uuid}.service")
    }

    /// Build systemd-run command.
    fn build_command(&self, config: &ProcessConfig, unit_name: &str) -> Command {
        let mut cmd = Command::new("systemd-run");

        // User mode if not root
        if self.user_mode {
            cmd.arg("--user");
        }

        // Unit configuration
        cmd.arg("--unit").arg(unit_name);
        cmd.arg("--description")
            .arg(format!("Hyprstream worker: {}", config.name));

        // Slice for resource isolation
        cmd.arg("--slice").arg(&self.slice);

        // Automatic cleanup when stopped
        cmd.arg("--property=CollectMode=inactive-or-failed");

        // Service type
        cmd.arg("--property=Type=exec");

        // Memory limit
        if let Some(ref limit) = config.memory_limit {
            cmd.arg(format!("--property=MemoryMax={limit}"));
        }

        // CPU quota
        if let Some(quota) = config.cpu_quota {
            cmd.arg(format!("--property=CPUQuota={quota}%"));
        }

        // Restart on failure
        if config.restart_on_failure {
            cmd.arg("--property=Restart=on-failure");
            cmd.arg("--property=RestartSec=1");
        }

        // Working directory
        if let Some(ref dir) = config.working_dir {
            cmd.arg(format!("--property=WorkingDirectory={}", dir.display()));
        }

        // Environment variables
        for (key, value) in &config.env {
            cmd.arg("--setenv").arg(format!("{key}={value}"));
        }

        // Custom unit properties
        for (key, value) in &config.unit_properties {
            cmd.arg(format!("--property={key}={value}"));
        }

        // The actual command to run
        cmd.arg("--").arg(&config.executable);
        cmd.args(&config.args);

        cmd
    }
}

impl Default for SystemdBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl SpawnerBackend for SystemdBackend {
    async fn spawn(&self, config: ProcessConfig) -> Result<SpawnedProcess> {
        let unit_name = self.generate_unit_name(&config.name);

        tracing::debug!(
            name = %config.name,
            unit = %unit_name,
            user_mode = %self.user_mode,
            "Spawning daemon via systemd-run"
        );

        // Build and run systemd-run command
        let mut cmd = self.build_command(&config, &unit_name);

        // Capture output for debugging
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let output = cmd.output().await.map_err(|e| {
            RpcError::SpawnFailed(format!(
                "failed to run systemd-run for {}: {}",
                config.name, e
            ))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RpcError::SpawnFailed(format!(
                "systemd-run failed for {}: {}",
                config.name, stderr
            )));
        }

        tracing::info!(
            name = %config.name,
            unit = %unit_name,
            "Daemon spawned as systemd transient unit"
        );

        Ok(SpawnedProcess::new(
            unit_name.clone(),
            ProcessKind::SystemdUnit(unit_name),
        ))
    }

    async fn stop(&self, process: &SpawnedProcess) -> Result<()> {
        let unit_name = match &process.kind {
            ProcessKind::SystemdUnit(name) => name,
            ProcessKind::Direct(_) => {
                return Err(RpcError::InvalidOperation(
                    "SystemdBackend cannot stop direct processes".to_owned(),
                ));
            }
        };

        tracing::debug!(
            unit = %unit_name,
            "Stopping systemd unit"
        );

        // Use systemctl stop
        let mut cmd = Command::new("systemctl");

        if self.user_mode {
            cmd.arg("--user");
        }

        cmd.arg("stop").arg(unit_name);

        let output = cmd.output().await.map_err(|e| {
            RpcError::StopFailed(format!(
                "failed to run systemctl stop for {unit_name}: {e}"
            ))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Don't fail if unit doesn't exist (already stopped)
            if !stderr.contains("not loaded") && !stderr.contains("not found") {
                tracing::warn!(
                    unit = %unit_name,
                    error = %stderr,
                    "systemctl stop returned error"
                );
            }
        }

        tracing::info!(
            unit = %unit_name,
            "Systemd unit stopped"
        );

        Ok(())
    }

    async fn is_running(&self, process: &SpawnedProcess) -> Result<bool> {
        let unit_name = match &process.kind {
            ProcessKind::SystemdUnit(name) => name,
            ProcessKind::Direct(_) => {
                return Err(RpcError::InvalidOperation(
                    "SystemdBackend cannot check direct processes".to_owned(),
                ));
            }
        };

        // Use systemctl is-active
        let mut cmd = Command::new("systemctl");

        if self.user_mode {
            cmd.arg("--user");
        }

        cmd.arg("is-active").arg("--quiet").arg(unit_name);

        let status = cmd.status().await.map_err(|e| {
            RpcError::Other(format!(
                "failed to run systemctl is-active for {unit_name}: {e}"
            ))
        })?;

        // Exit code 0 = active, non-zero = inactive/failed
        Ok(status.success())
    }

    fn backend_type(&self) -> &'static str {
        "systemd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        // This will depend on the test environment
        let available = SystemdBackend::is_available();
        println!("Systemd available: {available}");
    }

    #[test]
    fn test_generate_unit_name() {
        let backend = SystemdBackend::new();
        let name = backend.generate_unit_name("nydusd");

        assert!(name.starts_with("hyprstream-nydusd-"));
        assert!(name.ends_with(".service"));
    }

    #[test]
    fn test_user_mode_detection() {
        let backend = SystemdBackend::new();
        let expected_user_mode = !nix::unistd::geteuid().is_root();
        assert_eq!(backend.user_mode, expected_user_mode);
    }

    #[test]
    fn test_build_command() {
        let backend = SystemdBackend::with_user_mode(true);

        let config = ProcessConfig::new("test", "/usr/bin/test")
            .args(["--foo", "bar"])
            .working_dir("/tmp")
            .env("KEY", "VALUE")
            .memory_limit("1G")
            .cpu_quota(200);

        let cmd = backend.build_command(&config, "test-unit.service");

        // We can't easily inspect the Command, but at least verify it builds
        drop(cmd);
    }

    #[tokio::test]
    #[ignore = "requires systemd"]
    async fn test_spawn_and_stop() -> crate::Result<()> {
        if !SystemdBackend::is_available() {
            println!("Systemd not available, skipping test");
            return Ok(());
        }

        let backend = SystemdBackend::new();

        let config = ProcessConfig::new("test-sleep", "sleep").args(["60"]);

        let process = backend.spawn(config).await?;

        assert!(process.is_systemd());
        assert!(process.unit_name().is_some());

        // Check it's running
        let running = backend.is_running(&process).await?;
        assert!(running);

        // Stop it
        backend.stop(&process).await?;

        // Give systemd a moment
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Check it's stopped
        let running = backend.is_running(&process).await?;
        assert!(!running);
        Ok(())
    }
}
