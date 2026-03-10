//! NspawnBackend — systemd-nspawn based sandbox isolation
//!
//! Runs each sandbox as a lightweight container via `systemd-nspawn`,
//! bind-mounting the host root filesystem (`--directory=/`) with ephemeral
//! `/tmp` and `/run`.  Provides network isolation via `--network-veth` and
//! GPU pass-through via device bind-mounts.
//!
//! Service discovery uses host-side IPC paths namespaced by
//! `HYPRSTREAM_INSTANCE` rather than network IP, avoiding timing issues
//! with veth IP assignment.

use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::config::PoolConfig;
use crate::error::{Result, WorkerError};

use super::backend::{SandboxBackend, SandboxHandle};
use super::client::{LinuxContainerResources, PodSandboxConfig};
use super::sandbox::PodSandbox;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the nspawn backend.
#[derive(Debug, Clone)]
pub struct NspawnConfig {
    /// Path to libtorch (bind-mounted read-only into the container).
    pub libtorch_path: PathBuf,
    /// `LD_LIBRARY_PATH` forwarded into the container.
    pub ld_library_path: String,
    /// Run in rootless / user mode (default: true).
    pub user_mode: bool,
    /// Enable `--network-veth` for network isolation.
    pub network_veth: bool,
    /// GPU device paths to bind-mount (auto-detected from `/dev/dri`).
    pub gpu_devices: Vec<PathBuf>,
    /// Readiness poll timeout.
    pub ready_timeout: Duration,
    /// Readiness poll interval.
    pub ready_interval: Duration,
}

impl Default for NspawnConfig {
    fn default() -> Self {
        let libtorch_path = std::env::var("LIBTORCH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/usr/lib/libtorch"));

        let ld_library_path =
            std::env::var("LD_LIBRARY_PATH").unwrap_or_default();

        Self {
            libtorch_path,
            ld_library_path,
            user_mode: !nix::unistd::geteuid().is_root(),
            network_veth: true,
            gpu_devices: detect_gpu_devices(),
            ready_timeout: Duration::from_secs(10),
            ready_interval: Duration::from_millis(100),
        }
    }
}

/// Auto-detect GPU render/card nodes under `/dev/dri`.
fn detect_gpu_devices() -> Vec<PathBuf> {
    let dri = Path::new("/dev/dri");
    if !dri.exists() {
        return Vec::new();
    }
    let mut devices = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dri) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("card") || name.starts_with("renderD") {
                devices.push(entry.path());
            }
        }
    }
    devices
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle
// ─────────────────────────────────────────────────────────────────────────────

/// Backend-specific state stored on each `PodSandbox`.
#[derive(Debug, Clone)]
pub struct NspawnHandle {
    /// Sandbox identifier (matches `PodSandbox::id`).
    pub sandbox_id: String,
    /// Machine name passed to `--machine=`.
    pub machine_name: String,
    /// PID of the nspawn leader process (populated after start).
    pub leader_pid: Option<u32>,
    /// Host-side sandbox directory for bind-mounts.
    pub sandbox_path: PathBuf,
}

impl SandboxHandle for NspawnHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────────────────────────────────────

/// systemd-nspawn sandbox backend.
pub struct NspawnBackend {
    config: NspawnConfig,
}

impl NspawnBackend {
    pub fn new(config: NspawnConfig) -> Self {
        Self { config }
    }

    /// Build the `systemd-nspawn` argument list for a sandbox.
    fn build_nspawn_args(
        &self,
        sandbox: &PodSandbox,
        machine_name: &str,
        annotations: &HashMap<String, String>,
        resources: Option<&LinuxContainerResources>,
    ) -> Vec<String> {
        let mut args = Vec::new();

        // Core flags
        args.push(format!("--machine={machine_name}"));
        args.push("--directory=/".into());
        args.push("--tmpfs=/tmp".into());
        args.push("--tmpfs=/run".into());

        // Bind-mount sandbox runtime directory
        args.push(format!("--bind={}", sandbox.sandbox_path().display()));

        // Bind-mount libtorch read-only
        if self.config.libtorch_path.exists() {
            args.push(format!(
                "--bind-ro={}",
                self.config.libtorch_path.display()
            ));
        }

        // Network isolation
        if self.config.network_veth {
            args.push("--network-veth".into());
        }

        // GPU pass-through when requested via annotation
        let wants_gpu = annotations
            .get("hyprstream.io/gpu")
            .is_some_and(|v| v == "true");
        if wants_gpu {
            for dev in &self.config.gpu_devices {
                args.push(format!("--bind={}", dev.display()));
            }
        }

        // Environment variables
        args.push(format!(
            "--setenv=HYPRSTREAM_INSTANCE={}",
            sandbox.id
        ));
        if !self.config.ld_library_path.is_empty() {
            args.push(format!(
                "--setenv=LD_LIBRARY_PATH={}",
                self.config.ld_library_path
            ));
        }
        args.push(format!(
            "--setenv=LIBTORCH={}",
            self.config.libtorch_path.display()
        ));

        // Resource limits → systemd property flags
        if let Some(res) = resources {
            if res.cpu_quota > 0 {
                args.push(format!(
                    "--property=CPUQuota={}%",
                    res.cpu_quota
                ));
            }
            if res.memory_limit_in_bytes > 0 {
                args.push(format!(
                    "--property=MemoryMax={}",
                    res.memory_limit_in_bytes
                ));
            }
        }

        // Separator + boot command
        args.push("--".into());
        let hyprstream_bin = hyprstream_rpc::paths::executable_path()
            .unwrap_or_else(|_| PathBuf::from("hyprstream"));
        args.push(hyprstream_bin.to_string_lossy().into());
        args.push("service".into());
        args.push("start".into());
        args.push("--all".into());
        args.push("--foreground".into());

        args
    }

    /// Discover leader PID via `machinectl show`.
    async fn discover_leader_pid(machine_name: &str) -> Option<u32> {
        let output = tokio::process::Command::new("machinectl")
            .args(["show", "-p", "Leader", "--value", machine_name])
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout.trim().parse().ok()
    }

    /// Wait for the inner hyprstream to write its discovery socket.
    async fn wait_for_ready(&self, sandbox_id: &str) -> Result<()> {
        let runtime_dir = hyprstream_rpc::paths::runtime_dir();
        // The inner hyprstream uses HYPRSTREAM_INSTANCE, so its sockets
        // land under `instances/{sandbox_id}/` on the host filesystem.
        let discovery_sock = runtime_dir
            .join("instances")
            .join(sandbox_id)
            .join("discovery.sock");

        let deadline =
            tokio::time::Instant::now() + self.config.ready_timeout;

        while tokio::time::Instant::now() < deadline {
            if discovery_sock.exists() {
                debug!(
                    sandbox_id,
                    path = %discovery_sock.display(),
                    "Sandbox ready (discovery socket exists)"
                );
                return Ok(());
            }
            tokio::time::sleep(self.config.ready_interval).await;
        }

        Err(WorkerError::SandboxTimeout {
            operation: format!("nspawn readiness ({sandbox_id})"),
            timeout_secs: self.config.ready_timeout.as_secs(),
        })
    }
}

impl std::fmt::Debug for NspawnBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NspawnBackend")
            .field("config", &self.config)
            .finish()
    }
}

#[async_trait]
impl SandboxBackend for NspawnBackend {
    fn backend_type(&self) -> &'static str {
        "nspawn"
    }

    fn is_available(&self) -> bool {
        which::which("systemd-nspawn").is_ok()
            && which::which("machinectl").is_ok()
    }

    async fn initialize(&self, _config: &PoolConfig) -> Result<()> {
        if !self.is_available() {
            return Err(WorkerError::ConfigError(
                "systemd-nspawn or machinectl not found in PATH".into(),
            ));
        }
        if !self.config.libtorch_path.exists() {
            warn!(
                path = %self.config.libtorch_path.display(),
                "libtorch path does not exist — containers may fail to start"
            );
        }
        Ok(())
    }

    async fn start(
        &self,
        sandbox: &mut PodSandbox,
        config: &PodSandboxConfig,
        pool_config: &PoolConfig,
        annotations: &HashMap<String, String>,
    ) -> Result<Arc<dyn SandboxHandle>> {
        let machine_name = format!("hyprstream-{}", sandbox.id);

        // Ensure sandbox runtime directory exists
        let sandbox_path = pool_config.runtime_dir.join(&sandbox.id);
        tokio::fs::create_dir_all(&sandbox_path).await?;
        sandbox.sandbox_path = sandbox_path.clone();

        // Build argument list
        let nspawn_args = self.build_nspawn_args(
            sandbox,
            &machine_name,
            annotations,
            Some(&config.linux.resources),
        );

        info!(
            sandbox_id = %sandbox.id,
            machine = %machine_name,
            "Starting nspawn sandbox"
        );

        // Spawn detached — nspawn runs as a daemon
        let mut cmd = tokio::process::Command::new("systemd-nspawn");
        cmd.args(&nspawn_args);
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        let _child = cmd.spawn().map_err(|e| {
            WorkerError::VmStartFailed(format!(
                "failed to spawn systemd-nspawn: {e}"
            ))
        })?;

        // Wait for the inner hyprstream to become ready
        self.wait_for_ready(&sandbox.id).await?;

        // Discover leader PID
        let leader_pid =
            Self::discover_leader_pid(&machine_name).await;

        let handle = Arc::new(NspawnHandle {
            sandbox_id: sandbox.id.clone(),
            machine_name: machine_name.clone(),
            leader_pid,
            sandbox_path,
        });

        info!(
            sandbox_id = %sandbox.id,
            machine = %machine_name,
            leader_pid = ?leader_pid,
            "Nspawn sandbox started"
        );

        Ok(handle)
    }

    async fn stop(&self, sandbox: &PodSandbox) -> Result<()> {
        let machine_name = format!("hyprstream-{}", sandbox.id);
        info!(sandbox_id = %sandbox.id, machine = %machine_name, "Stopping nspawn sandbox");

        let status = tokio::process::Command::new("machinectl")
            .args(["stop", &machine_name])
            .status()
            .await
            .map_err(|e| {
                WorkerError::VmStopFailed(format!(
                    "failed to run machinectl stop: {e}"
                ))
            })?;

        if !status.success() {
            warn!(
                sandbox_id = %sandbox.id,
                "machinectl stop returned non-zero (machine may already be stopped)"
            );
        }

        Ok(())
    }

    async fn destroy(&self, sandbox: &PodSandbox) -> Result<()> {
        let machine_name = format!("hyprstream-{}", sandbox.id);
        info!(sandbox_id = %sandbox.id, machine = %machine_name, "Destroying nspawn sandbox");

        // Forcefully terminate
        let _ = tokio::process::Command::new("machinectl")
            .args(["terminate", &machine_name])
            .status()
            .await;

        // Clean up sandbox directory
        let sandbox_path = sandbox.sandbox_path();
        if sandbox_path.exists() {
            if let Err(e) = tokio::fs::remove_dir_all(sandbox_path).await {
                warn!(
                    sandbox_id = %sandbox.id,
                    error = %e,
                    "Failed to remove sandbox directory"
                );
            }
        }

        // Clean up instance runtime directory
        let instance_dir = hyprstream_rpc::paths::runtime_dir()
            .join("instances")
            .join(&sandbox.id);
        if instance_dir.exists() {
            let _ = tokio::fs::remove_dir_all(&instance_dir).await;
        }

        Ok(())
    }

    async fn reset(&self, _sandbox: &mut PodSandbox) -> Result<bool> {
        // Nspawn sandboxes are ephemeral — must be recreated
        Ok(false)
    }

    async fn get_pids(&self, sandbox: &PodSandbox) -> Result<Vec<u32>> {
        let machine_name = format!("hyprstream-{}", sandbox.id);
        match Self::discover_leader_pid(&machine_name).await {
            Some(pid) => Ok(vec![pid]),
            None => Ok(Vec::new()),
        }
    }

    fn supports_exec(&self) -> bool {
        true
    }

    async fn exec_sync(
        &self,
        sandbox: &PodSandbox,
        command: &[String],
        timeout_secs: u64,
    ) -> Result<(i32, Vec<u8>, Vec<u8>)> {
        let machine_name = format!("hyprstream-{}", sandbox.id);

        let mut args = vec![
            "shell".to_owned(),
            machine_name,
            "--".to_owned(),
        ];
        args.extend_from_slice(command);

        let output = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            tokio::process::Command::new("machinectl")
                .args(&args)
                .output(),
        )
        .await
        .map_err(|_| WorkerError::SandboxTimeout {
            operation: format!("exec_sync in {}", sandbox.id),
            timeout_secs,
        })?
        .map_err(|e| {
            WorkerError::ExecFailed(format!(
                "machinectl shell failed: {e}"
            ))
        })?;

        let exit_code = output.status.code().unwrap_or(-1);
        Ok((exit_code, output.stdout, output.stderr))
    }

    async fn update_resources(
        &self,
        sandbox: &PodSandbox,
        resources: &LinuxContainerResources,
    ) -> Result<()> {
        let machine_name = format!("hyprstream-{}", sandbox.id);

        // Use systemctl set-property on the machine's scope unit
        let scope = format!("machine-{machine_name}.scope");

        if resources.cpu_quota > 0 {
            let _ = tokio::process::Command::new("systemctl")
                .args([
                    "set-property",
                    &scope,
                    &format!("CPUQuota={}%", resources.cpu_quota),
                ])
                .status()
                .await;
        }
        if resources.memory_limit_in_bytes > 0 {
            let _ = tokio::process::Command::new("systemctl")
                .args([
                    "set-property",
                    &scope,
                    &format!("MemoryMax={}", resources.memory_limit_in_bytes),
                ])
                .status()
                .await;
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_nspawn_config_default() {
        let config = NspawnConfig::default();
        assert!(config.network_veth);
        assert_eq!(config.ready_timeout, Duration::from_secs(10));
        assert_eq!(config.ready_interval, Duration::from_millis(100));
    }

    #[test]
    fn test_nspawn_handle_as_any() {
        let handle = NspawnHandle {
            sandbox_id: "test-123".into(),
            machine_name: "hyprstream-test-123".into(),
            leader_pid: Some(42),
            sandbox_path: PathBuf::from("/tmp/test"),
        };

        let handle: Arc<dyn SandboxHandle> = Arc::new(handle);
        let downcast = handle.as_any().downcast_ref::<NspawnHandle>();
        assert!(downcast.is_some());
        assert_eq!(downcast.unwrap().sandbox_id, "test-123");
        assert_eq!(downcast.unwrap().leader_pid, Some(42));
    }

    #[test]
    fn test_detect_gpu_devices() {
        // Just verify it doesn't panic — actual devices depend on hardware
        let devices = detect_gpu_devices();
        for d in &devices {
            assert!(
                d.to_string_lossy().contains("card")
                    || d.to_string_lossy().contains("renderD")
            );
        }
    }

    #[test]
    fn test_backend_type() {
        let backend = NspawnBackend::new(NspawnConfig::default());
        assert_eq!(backend.backend_type(), "nspawn");
    }

    #[test]
    fn test_is_available() {
        let backend = NspawnBackend::new(NspawnConfig::default());
        // Just check it runs without panic — availability depends on system
        let _available = backend.is_available();
    }
}
