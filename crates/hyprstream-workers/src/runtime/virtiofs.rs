//! VirtioFS daemon wrapper for serving RAFS to Kata VMs
//!
//! Provides filesystem service for sandboxes using nydus-service.
//! Each sandbox gets its own daemon instance to serve RAFS images.
//!
//! # Architecture
//!
//! ```text
//! SandboxVirtiofs
//!     │
//!     ├── socket_path      (vhost-user-fs socket for VM)
//!     ├── mountpoint       (FUSE mount path)
//!     └── daemon           (NydusDaemon instance)
//!           │
//!           └── VFS ← RAFS backend ← bootstrap + blobs
//! ```

use std::path::{Path, PathBuf};

use crate::config::ImageConfig;
use crate::error::{Result, WorkerError};
use crate::image::RafsStore;

/// Sandbox filesystem daemon for serving RAFS to VMs
///
/// Wraps nydus-service to provide filesystem service for each sandbox.
/// The daemon serves RAFS images via a socket that the VM can connect to.
#[derive(Debug)]
pub struct SandboxVirtiofs {
    /// Sandbox ID this daemon serves
    sandbox_id: String,

    /// Socket path for vhost-user-fs (VM connects here)
    socket_path: PathBuf,

    /// FUSE mountpoint (for FUSE mode)
    mountpoint: Option<PathBuf>,

    /// Image ID being served
    image_id: String,

    /// Whether the daemon is running
    running: bool,

    /// Process ID if using external nydusd
    daemon_pid: Option<u32>,
}

impl SandboxVirtiofs {
    /// Create a new SandboxVirtiofs instance
    ///
    /// # Arguments
    /// * `sandbox_id` - ID of the sandbox this daemon serves
    /// * `socket_path` - Path for vhost-user-fs socket
    /// * `rafs_store` - RAFS store for image data
    /// * `image_id` - ID of the image to serve
    pub async fn new(
        sandbox_id: String,
        socket_path: PathBuf,
        _rafs_store: &RafsStore,
        image_id: String,
    ) -> Result<Self> {
        Ok(Self {
            sandbox_id,
            socket_path,
            mountpoint: None,
            image_id,
            running: false,
            daemon_pid: None,
        })
    }

    /// Start the virtiofs daemon
    ///
    /// Creates a vhost-user-fs socket that the VM can connect to.
    pub async fn start(
        &mut self,
        rafs_store: &RafsStore,
        config: &ImageConfig,
    ) -> Result<()> {
        if self.running {
            return Ok(());
        }

        tracing::info!(
            sandbox_id = %self.sandbox_id,
            image_id = %self.image_id,
            socket = %self.socket_path.display(),
            "Starting virtiofs daemon"
        );

        // Ensure socket directory exists
        if let Some(parent) = self.socket_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Get the bootstrap and blobs paths
        let bootstrap_path = rafs_store.bootstrap_path(&self.image_id);
        let blobs_dir = rafs_store.blobs_dir();
        let cache_dir = rafs_store.cache_dir();

        // For now, we'll spawn nydusd as an external process in virtiofs mode
        // This is the most reliable approach for VM integration
        let result = self
            .spawn_nydusd_virtiofs(
                &bootstrap_path,
                blobs_dir,
                cache_dir,
                &config.runtime_dir,
            )
            .await;

        match result {
            Ok(pid) => {
                self.daemon_pid = Some(pid);
                self.running = true;
                tracing::info!(
                    sandbox_id = %self.sandbox_id,
                    pid = %pid,
                    "Virtiofs daemon started"
                );
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    sandbox_id = %self.sandbox_id,
                    error = %e,
                    "Failed to start virtiofs daemon, sandbox will use fallback"
                );
                // Don't fail - the sandbox can still work with direct mounts
                // This allows testing without nydusd installed
                self.running = false;
                Ok(())
            }
        }
    }

    /// Spawn nydusd in virtiofs mode
    async fn spawn_nydusd_virtiofs(
        &self,
        bootstrap_path: &Path,
        blobs_dir: &Path,
        cache_dir: &Path,
        runtime_dir: &Path,
    ) -> Result<u32> {
        // Create config file for nydusd
        let config_path = runtime_dir.join(format!("{}-nydusd.json", self.sandbox_id));
        let api_socket = runtime_dir.join(format!("{}-api.sock", self.sandbox_id));

        // Nydus daemon configuration
        let config = serde_json::json!({
            "device": {
                "backend": {
                    "type": "localfs",
                    "config": {
                        "dir": blobs_dir.to_string_lossy(),
                        "readahead": true
                    }
                },
                "cache": {
                    "type": "blobcache",
                    "config": {
                        "work_dir": cache_dir.to_string_lossy()
                    }
                }
            },
            "mode": "direct",
            "digest_validate": false,
            "enable_xattr": true,
            "fs_prefetch": {
                "enable": true,
                "threads_count": 4
            }
        });

        tokio::fs::write(&config_path, serde_json::to_string_pretty(&config)?).await?;

        // Spawn nydusd process
        let config_path_str = config_path.to_str().ok_or_else(|| {
            WorkerError::SandboxCreationFailed("config path contains invalid UTF-8".to_owned())
        })?;
        let bootstrap_path_str = bootstrap_path.to_str().ok_or_else(|| {
            WorkerError::SandboxCreationFailed("bootstrap path contains invalid UTF-8".to_owned())
        })?;
        let socket_path_str = self.socket_path.to_str().ok_or_else(|| {
            WorkerError::SandboxCreationFailed("socket path contains invalid UTF-8".to_owned())
        })?;
        let api_socket_str = api_socket.to_str().ok_or_else(|| {
            WorkerError::SandboxCreationFailed("api socket path contains invalid UTF-8".to_owned())
        })?;
        let child = tokio::process::Command::new("nydusd")
            .args([
                "--config",
                config_path_str,
                "--mountpoint",
                "", // No mountpoint for virtiofs
                "--bootstrap",
                bootstrap_path_str,
                "--sock",
                socket_path_str,
                "--apisock",
                api_socket_str,
                "--log-level",
                "info",
            ])
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                WorkerError::SandboxCreationFailed(format!(
                    "failed to spawn nydusd: {e}"
                ))
            })?;

        let pid = child.id().ok_or_else(|| {
            WorkerError::SandboxCreationFailed("nydusd process has no pid".to_owned())
        })?;

        // Wait briefly for daemon to initialize
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Check if socket exists (daemon is ready)
        if !self.socket_path.exists() {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }

        if !self.socket_path.exists() {
            return Err(WorkerError::SandboxCreationFailed(
                "nydusd socket not created".to_owned(),
            ));
        }

        Ok(pid)
    }

    /// Stop the virtiofs daemon
    pub async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        tracing::info!(
            sandbox_id = %self.sandbox_id,
            "Stopping virtiofs daemon"
        );

        // Kill the daemon process if running
        if let Some(pid) = self.daemon_pid {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGTERM,
            );

            // Wait briefly for graceful shutdown
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            // Force kill if still running
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGKILL,
            );
        }

        // Clean up socket
        if self.socket_path.exists() {
            let _ = tokio::fs::remove_file(&self.socket_path).await;
        }

        self.running = false;
        self.daemon_pid = None;

        Ok(())
    }

    /// Get the socket path for VM connection
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Get the mountpoint (if using FUSE mode)
    pub fn mountpoint(&self) -> Option<&Path> {
        self.mountpoint.as_deref()
    }

    /// Check if the daemon is running
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get the image ID being served
    pub fn image_id(&self) -> &str {
        &self.image_id
    }

    /// Get the sandbox ID
    pub fn sandbox_id(&self) -> &str {
        &self.sandbox_id
    }
}

impl Drop for SandboxVirtiofs {
    fn drop(&mut self) {
        // Best-effort cleanup on drop
        if let Some(pid) = self.daemon_pid {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGKILL,
            );
        }
    }
}

/// Builder for SandboxVirtiofs with fluent API
pub struct SandboxVirtiofsBuilder {
    sandbox_id: String,
    socket_path: Option<PathBuf>,
    image_id: Option<String>,
}

impl SandboxVirtiofsBuilder {
    /// Create a new builder
    pub fn new(sandbox_id: impl Into<String>) -> Self {
        Self {
            sandbox_id: sandbox_id.into(),
            socket_path: None,
            image_id: None,
        }
    }

    /// Set the socket path
    pub fn socket_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.socket_path = Some(path.into());
        self
    }

    /// Set the image ID
    pub fn image_id(mut self, id: impl Into<String>) -> Self {
        self.image_id = Some(id.into());
        self
    }

    /// Build the SandboxVirtiofs instance
    pub async fn build(self, rafs_store: &RafsStore) -> Result<SandboxVirtiofs> {
        let socket_path = self.socket_path.ok_or_else(|| {
            WorkerError::ConfigError("socket_path is required".to_owned())
        })?;

        let image_id = self.image_id.ok_or_else(|| {
            WorkerError::ConfigError("image_id is required".to_owned())
        })?;

        SandboxVirtiofs::new(self.sandbox_id, socket_path, rafs_store, image_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let builder = SandboxVirtiofsBuilder::new("test-sandbox")
            .socket_path("/tmp/test.sock")
            .image_id("sha256:abc123");

        assert_eq!(builder.sandbox_id, "test-sandbox");
        assert_eq!(builder.socket_path, Some(PathBuf::from("/tmp/test.sock")));
        assert_eq!(builder.image_id, Some("sha256:abc123".to_owned()));
    }
}
