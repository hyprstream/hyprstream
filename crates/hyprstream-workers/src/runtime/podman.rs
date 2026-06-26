//! PodmanBackend — rootless podman/docker OCI SandboxBackend
//!
//! Runs each sandbox as a rootless OCI container via `podman` (or `docker` as
//! fallback).  No hypervisor, no RAFS — pulls a standard OCI image and keeps
//! it running for `exec_sync` workloads.  Designed for casual/dev installs
//! where Kata + Cloud-Hypervisor toolchain is unavailable.
//!
//! Backend type: `"podman"`.

use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
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
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Locate the container runtime binary: prefer `podman`, fall back to `docker`.
fn find_runtime() -> Option<PathBuf> {
    for name in &["podman", "docker"] {
        if let Ok(path) = which::which(name) {
            return Some(path);
        }
    }
    None
}

/// Container name for a given sandbox id.
fn container_name(sandbox_id: &str) -> String {
    format!("hyprstream-{sandbox_id}")
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the podman backend.
#[derive(Debug, Clone)]
pub struct PodmanConfig {
    /// OCI base image to run.  Overridable per-sandbox via annotations.
    pub base_image: String,
    /// Network mode passed to `--network`.
    pub network_mode: String,
    /// Extra volume mounts (`host:container[:options]`).
    pub extra_mounts: Vec<String>,
    /// Extra env vars forwarded to the container.
    pub extra_env: Vec<(String, String)>,
    /// Stop timeout (seconds) before forceful kill.
    pub stop_timeout: u64,
    /// Exec timeout default.
    pub exec_timeout: Duration,
}

impl Default for PodmanConfig {
    fn default() -> Self {
        Self {
            base_image: std::env::var("HYPRSTREAM_WORKER_IMAGE")
                .unwrap_or_else(|_| "ghcr.io/hyprstream/worker:latest".to_owned()),
            network_mode: "slirp4netns".to_owned(),
            extra_mounts: Vec::new(),
            extra_env: Vec::new(),
            stop_timeout: 10,
            exec_timeout: Duration::from_secs(300),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle
// ─────────────────────────────────────────────────────────────────────────────

/// Backend-specific state stored on each `PodSandbox`.
#[derive(Debug, Clone)]
pub struct PodmanHandle {
    /// Sandbox identifier.
    pub sandbox_id: String,
    /// Container name (`hyprstream-{sandbox_id}`).
    pub container_name: String,
    /// PID of the container init process (populated after start).
    pub init_pid: Option<u32>,
    /// Runtime binary used (podman or docker path).
    pub runtime_bin: PathBuf,
}

impl SandboxHandle for PodmanHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────────────────────────────────────

/// Rootless podman/docker OCI SandboxBackend.
pub struct PodmanBackend {
    config: PodmanConfig,
}

impl PodmanBackend {
    pub fn new(config: PodmanConfig) -> Self {
        Self { config }
    }

    fn runtime_bin(&self) -> Option<PathBuf> {
        find_runtime()
    }

    /// Inspect a running container and return its init PID.
    async fn inspect_pid(runtime: &PathBuf, name: &str) -> Option<u32> {
        let out = tokio::process::Command::new(runtime)
            .args(["inspect", "--format", "{{.State.Pid}}", name])
            .output()
            .await
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let pid_str = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        pid_str.parse().ok().filter(|&p: &u32| p > 0)
    }

    /// Resolve the OCI image: annotation overrides config default.
    fn resolve_image(&self, annotations: &HashMap<String, String>) -> String {
        annotations
            .get("hyprstream.io/worker-image")
            .cloned()
            .unwrap_or_else(|| self.config.base_image.clone())
    }
}

#[async_trait]
impl SandboxBackend for PodmanBackend {
    fn backend_type(&self) -> &'static str {
        "podman"
    }

    fn is_available(&self) -> bool {
        find_runtime().is_some()
    }

    async fn initialize(&self, _config: &PoolConfig) -> Result<()> {
        let bin = self.runtime_bin().ok_or_else(|| {
            WorkerError::VmStartFailed(
                "podman or docker not found in PATH".to_owned(),
            )
        })?;

        let out = tokio::process::Command::new(&bin)
            .args(["info", "--format", "{{.Host.Security.Rootless}}"])
            .output()
            .await
            .map_err(|e| {
                WorkerError::VmStartFailed(format!(
                    "failed to run `{} info`: {e}",
                    bin.display()
                ))
            })?;

        if !out.status.success() {
            return Err(WorkerError::VmStartFailed(format!(
                "`{} info` failed: {}",
                bin.display(),
                String::from_utf8_lossy(&out.stderr)
            )));
        }

        let rootless = String::from_utf8_lossy(&out.stdout).trim() == "true";
        info!(
            runtime = %bin.display(),
            rootless,
            "PodmanBackend initialized"
        );
        Ok(())
    }

    async fn start(
        &self,
        sandbox: &mut PodSandbox,
        _config: &PodSandboxConfig,
        _pool_config: &PoolConfig,
        annotations: &HashMap<String, String>,
    ) -> Result<Arc<dyn SandboxHandle>> {
        let bin = self.runtime_bin().ok_or_else(|| {
            WorkerError::VmStartFailed("no container runtime in PATH".to_owned())
        })?;
        let image = self.resolve_image(annotations);
        let name = container_name(&sandbox.id);

        info!(sandbox_id = %sandbox.id, container = %name, image = %image, "Starting podman sandbox");

        let mut args = vec![
            "run".to_owned(),
            "--detach".to_owned(),
            "--rm".to_owned(),
            format!("--name={name}"),
            format!("--network={}", self.config.network_mode),
            format!("--stop-timeout={}", self.config.stop_timeout),
        ];

        for mount in &self.config.extra_mounts {
            args.push(format!("--volume={mount}"));
        }
        for (k, v) in &self.config.extra_env {
            args.push(format!("--env={k}={v}"));
        }

        args.push(image.clone());
        args.push("sleep".to_owned());
        args.push("infinity".to_owned());

        let status = tokio::process::Command::new(&bin)
            .args(&args)
            .status()
            .await
            .map_err(|e| {
                WorkerError::VmStartFailed(format!("podman run failed: {e}"))
            })?;

        if !status.success() {
            return Err(WorkerError::VmStartFailed(format!(
                "podman run exited non-zero for sandbox {}",
                sandbox.id
            )));
        }

        let init_pid = Self::inspect_pid(&bin, &name).await;
        debug!(sandbox_id = %sandbox.id, pid = ?init_pid, "Container started");

        Ok(Arc::new(PodmanHandle {
            sandbox_id: sandbox.id.clone(),
            container_name: name,
            init_pid,
            runtime_bin: bin,
        }))
    }

    async fn stop(&self, sandbox: &PodSandbox) -> Result<()> {
        let bin = match self.runtime_bin() {
            Some(b) => b,
            None => return Ok(()),
        };
        let name = container_name(&sandbox.id);
        info!(sandbox_id = %sandbox.id, container = %name, "Stopping podman sandbox");

        let status = tokio::process::Command::new(&bin)
            .args(["stop", &name])
            .status()
            .await
            .map_err(|e| {
                WorkerError::VmStopFailed(format!("podman stop failed: {e}"))
            })?;

        if !status.success() {
            warn!(sandbox_id = %sandbox.id, "podman stop returned non-zero (may already be stopped)");
        }
        Ok(())
    }

    async fn destroy(&self, sandbox: &PodSandbox) -> Result<()> {
        let bin = match self.runtime_bin() {
            Some(b) => b,
            None => return Ok(()),
        };
        let name = container_name(&sandbox.id);
        info!(sandbox_id = %sandbox.id, container = %name, "Destroying podman sandbox");

        // Forceful remove (idempotent).
        let _ = tokio::process::Command::new(&bin)
            .args(["rm", "--force", &name])
            .status()
            .await;

        Ok(())
    }

    async fn reset(&self, _sandbox: &mut PodSandbox) -> Result<bool> {
        // Podman containers are ephemeral — recreate on reset.
        Ok(false)
    }

    async fn get_pids(&self, sandbox: &PodSandbox) -> Result<Vec<u32>> {
        let bin = match self.runtime_bin() {
            Some(b) => b,
            None => return Ok(Vec::new()),
        };
        let name = container_name(&sandbox.id);
        match Self::inspect_pid(&bin, &name).await {
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
        let bin = self.runtime_bin().ok_or_else(|| {
            WorkerError::VmStartFailed("no container runtime".to_owned())
        })?;
        let name = container_name(&sandbox.id);

        let mut args = vec!["exec".to_owned(), name, "--".to_owned()];
        args.extend_from_slice(command);

        let output = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            tokio::process::Command::new(&bin).args(&args).output(),
        )
        .await
        .map_err(|_| WorkerError::SandboxTimeout {
            operation: format!("exec_sync in {}", sandbox.id),
            timeout_secs,
        })?
        .map_err(|e| WorkerError::ExecFailed(format!("podman exec failed: {e}")))?;

        let exit_code = output.status.code().unwrap_or(-1);
        Ok((exit_code, output.stdout, output.stderr))
    }

    async fn update_resources(
        &self,
        sandbox: &PodSandbox,
        resources: &LinuxContainerResources,
    ) -> Result<()> {
        let bin = match self.runtime_bin() {
            Some(b) => b,
            None => return Ok(()),
        };
        let name = container_name(&sandbox.id);

        let mut args = vec!["update".to_owned(), name];

        if resources.cpu_quota > 0 && resources.cpu_period > 0 {
            args.push(format!("--cpu-quota={}", resources.cpu_quota));
            args.push(format!("--cpu-period={}", resources.cpu_period));
        }
        if resources.memory_limit_in_bytes > 0 {
            args.push(format!("--memory={}", resources.memory_limit_in_bytes));
        }

        let status = tokio::process::Command::new(&bin)
            .args(&args)
            .status()
            .await
            .map_err(|e| WorkerError::VmStartFailed(format!("podman update failed: {e}")))?;

        if !status.success() {
            warn!(sandbox_id = %sandbox.id, "podman update returned non-zero");
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type() {
        let b = PodmanBackend::new(PodmanConfig::default());
        assert_eq!(b.backend_type(), "podman");
    }

    #[test]
    fn test_is_available_does_not_panic() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let _ = b.is_available();
    }

    #[test]
    fn test_container_name() {
        assert_eq!(container_name("abc-123"), "hyprstream-abc-123");
    }

    #[test]
    fn test_resolve_image_default() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let annotations = HashMap::new();
        let image = b.resolve_image(&annotations);
        assert!(!image.is_empty());
    }

    #[test]
    fn test_resolve_image_annotation_override() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let mut annotations = HashMap::new();
        annotations.insert(
            "hyprstream.io/worker-image".to_owned(),
            "custom/image:v1".to_owned(),
        );
        assert_eq!(b.resolve_image(&annotations), "custom/image:v1");
    }

    #[test]
    fn test_handle_as_any() {
        let h = PodmanHandle {
            sandbox_id: "test-1".into(),
            container_name: "hyprstream-test-1".into(),
            init_pid: Some(99),
            runtime_bin: PathBuf::from("/usr/bin/podman"),
        };
        let h: Arc<dyn SandboxHandle> = Arc::new(h);
        let down = h.as_any().downcast_ref::<PodmanHandle>();
        assert!(down.is_some());
        assert_eq!(down.unwrap().init_pid, Some(99));
    }
}
