//! Sandbox Pool for warm VM management
//!
//! Manages a pool of pre-warmed Kata VMs (PodSandboxes) for fast container startup.
//! Implements warm pool and active sandbox tracking.
//!
//! Uses the Kata Containers `Hypervisor` trait for VM lifecycle management,
//! supporting multiple hypervisors: Cloud Hypervisor, QEMU, Dragonball.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use kata_hypervisor::ch::CloudHypervisor;
#[cfg(feature = "dragonball")]
use kata_hypervisor::dragonball::Dragonball;
use kata_hypervisor::Hypervisor;
use kata_types::config::hypervisor::{Hypervisor as HypervisorConfig, RootlessUser};
use kata_types::rootless;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::{HypervisorType, ImageConfig, PoolConfig};
use crate::error::{Result, WorkerError};
use crate::image::RafsStore;

use super::virtiofs::SandboxVirtiofs;
use super::{PodSandbox, PodSandboxConfig, PodSandboxState};

/// Initialize rootless mode if running as non-root user
fn init_rootless_mode() {
    // Enable rootless mode for non-root users
    // This makes kata_types::build_path() use XDG_RUNTIME_DIR instead of /run/kata
    let is_root = nix::unistd::geteuid().is_root();
    if !is_root {
        rootless::set_rootless(true);
        tracing::debug!("Enabled Kata rootless mode for non-root user");
    }
}

/// Sandbox pool for warm VM management
///
/// Maintains:
/// - Warm pool: Pre-created VMs ready for immediate use
/// - Active sandboxes: VMs currently in use by workloads
pub struct SandboxPool {
    /// Pre-warmed sandboxes ready for use
    warm_pool: Mutex<VecDeque<PodSandbox>>,

    /// Active sandboxes (in use)
    active: Mutex<HashMap<String, PodSandbox>>,

    /// Pool configuration
    config: PoolConfig,

    /// Image configuration (for virtiofs)
    image_config: ImageConfig,

    /// RAFS store for image data
    rafs_store: Arc<RafsStore>,
}

impl SandboxPool {
    /// Create a new sandbox pool with configuration
    pub fn new(config: PoolConfig, image_config: ImageConfig, rafs_store: Arc<RafsStore>) -> Self {
        // Initialize rootless mode before any Kata operations
        init_rootless_mode();

        Self {
            warm_pool: Mutex::new(VecDeque::new()),
            active: Mutex::new(HashMap::new()),
            config,
            image_config,
            rafs_store,
        }
    }

    /// Initialize the warm pool with pre-created sandboxes
    pub async fn initialize(&self) -> Result<()> {
        let mut warm = self.warm_pool.lock().await;

        for _ in 0..self.config.warm_pool_size {
            let sandbox = self.create_warm_sandbox().await?;
            warm.push_back(sandbox);
        }

        tracing::info!(
            "Initialized warm pool with {} sandboxes",
            self.config.warm_pool_size
        );

        Ok(())
    }

    /// Acquire a sandbox from the pool
    ///
    /// Prefers warm sandboxes, creates new if none available.
    pub async fn acquire(&self, config: &PodSandboxConfig) -> Result<String> {
        // Check capacity
        let active_count = self.active.lock().await.len();
        if active_count >= self.config.max_sandboxes {
            return Err(WorkerError::PoolExhausted {
                max: self.config.max_sandboxes,
            });
        }

        // Try to get a warm sandbox
        let mut warm = self.warm_pool.lock().await;
        let sandbox = if let Some(mut sandbox) = warm.pop_front() {
            // Configure the warm sandbox with the provided config
            sandbox.metadata = config.metadata.clone();
            sandbox.labels = config.labels.clone();
            sandbox.annotations = config.annotations.clone();
            sandbox.state = PodSandboxState::SandboxReady;
            sandbox
        } else {
            // Create a new sandbox
            drop(warm); // Release lock before creating
            self.create_sandbox(config).await?
        };

        let sandbox_id = sandbox.id.clone();

        // Move to active
        self.active.lock().await.insert(sandbox_id.clone(), sandbox);

        // Replenish warm pool in background
        self.replenish_warm_pool();

        Ok(sandbox_id)
    }

    /// Release a sandbox back to the pool or destroy it
    pub async fn release(&self, sandbox_id: &str) -> Result<()> {
        let mut active = self.active.lock().await;

        let sandbox = active
            .remove(sandbox_id)
            .ok_or_else(|| WorkerError::SandboxNotFound(sandbox_id.to_owned()))?;

        // Stop the sandbox
        self.stop_sandbox(&sandbox).await?;

        // Decide whether to return to warm pool or destroy
        let warm = self.warm_pool.lock().await;
        if warm.len() < self.config.warm_pool_size {
            drop(warm);
            // Reset and return to warm pool
            let reset_sandbox = self.reset_sandbox(sandbox).await?;
            self.warm_pool.lock().await.push_back(reset_sandbox);
        } else {
            drop(warm);
            // Destroy the sandbox
            self.destroy_sandbox(sandbox).await?;
        }

        Ok(())
    }

    /// Get a sandbox by ID
    pub async fn get(&self, sandbox_id: &str) -> Option<PodSandbox> {
        self.active.lock().await.get(sandbox_id).cloned()
    }

    /// List all active sandboxes
    pub async fn list_active(&self) -> Vec<PodSandbox> {
        self.active.lock().await.values().cloned().collect()
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        let warm_count = self.warm_pool.lock().await.len();
        let active_count = self.active.lock().await.len();

        PoolStats {
            warm_count,
            active_count,
            max_sandboxes: self.config.max_sandboxes,
            warm_pool_target: self.config.warm_pool_size,
        }
    }

    /// Shutdown the pool, destroying all sandboxes
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down sandbox pool");

        // Destroy warm sandboxes
        let mut warm = self.warm_pool.lock().await;
        while let Some(sandbox) = warm.pop_front() {
            if let Err(e) = self.destroy_sandbox(sandbox).await {
                tracing::warn!("Error destroying warm sandbox: {}", e);
            }
        }
        drop(warm);

        // Destroy active sandboxes
        let active: Vec<_> = {
            let mut active = self.active.lock().await;
            active.drain().map(|(_, s)| s).collect()
        };

        for sandbox in active {
            if let Err(e) = self.stop_sandbox(&sandbox).await {
                tracing::warn!("Error stopping sandbox {}: {}", sandbox.id, e);
            }
            if let Err(e) = self.destroy_sandbox(sandbox).await {
                tracing::warn!("Error destroying sandbox: {}", e);
            }
        }

        tracing::info!("Sandbox pool shutdown complete");
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal methods
    // ─────────────────────────────────────────────────────────────────────────

    /// Create a warm sandbox (minimal configuration)
    async fn create_warm_sandbox(&self) -> Result<PodSandbox> {
        let id = format!("warm-{}", Uuid::new_v4());
        let config = PodSandboxConfig::default();
        let sandbox_path = self.config.runtime_dir.join(&id);

        let mut sandbox = PodSandbox::new(id, &config, sandbox_path);

        // Start the VM
        self.start_vm(&mut sandbox).await?;

        Ok(sandbox)
    }

    /// Create a sandbox with full configuration
    async fn create_sandbox(&self, config: &PodSandboxConfig) -> Result<PodSandbox> {
        let id = Uuid::new_v4().to_string();
        let sandbox_path = self.config.runtime_dir.join(&id);

        let mut sandbox = PodSandbox::new(id, config, sandbox_path);

        // Start the VM
        self.start_vm(&mut sandbox).await?;

        sandbox.state = PodSandboxState::SandboxReady;

        Ok(sandbox)
    }

    /// Start a Kata VM for the sandbox using the Hypervisor trait
    async fn start_vm(&self, sandbox: &mut PodSandbox) -> Result<()> {
        tracing::info!(sandbox_id = %sandbox.id, "Starting VM for sandbox");

        // 1. Create runtime directory for this sandbox
        let sandbox_path = sandbox.sandbox_path().clone();
        tokio::fs::create_dir_all(&sandbox_path).await?;

        // 2. Set up paths
        let api_socket = sandbox_path.join("cloud-hypervisor.sock");
        let virtiofs_socket = sandbox_path.join("virtiofs.sock");
        let cloud_init_iso = sandbox_path.join("cloud-init.iso");

        // 3. Generate cloud-init ISO (if we have templates)
        if self.config.cloud_init_dir.exists() {
            self.generate_cloud_init_iso(sandbox, &cloud_init_iso).await?;
        }

        // 4. Start virtiofs daemon if we have an image
        if let Some(ref image_id) = sandbox.image_id {
            let mut virtiofs = SandboxVirtiofs::new(
                sandbox.id.clone(),
                virtiofs_socket.clone(),
                &self.rafs_store,
                image_id.clone(),
            )
            .await?;

            virtiofs.start(&self.rafs_store, &self.image_config).await?;
            sandbox.virtiofs_daemon = Some(Arc::new(virtiofs));
            sandbox.virtiofs_socket = Some(virtiofs_socket.clone());
        }

        // 5. Create and configure the hypervisor using Kata's Hypervisor trait
        let hypervisor = self
            .create_hypervisor(sandbox, &api_socket, &virtiofs_socket, &cloud_init_iso)
            .await?;

        // 6. Prepare the VM (creates configuration)
        // Note: Annotations are passed to prepare_vm, config was set via set_hypervisor_config()
        let annotations = sandbox.annotations.clone();
        hypervisor
            .prepare_vm(&sandbox.id, None, &annotations, None)
            .await
            .map_err(|e| WorkerError::VmStartFailed(format!("failed to prepare VM: {e}")))?;

        // 7. Start the VM
        let timeout_secs = self.config.create_timeout_secs as i32;
        tracing::debug!(sandbox_id = %sandbox.id, timeout_secs, "Starting VM");
        hypervisor
            .start_vm(timeout_secs)
            .await
            .map_err(|e| WorkerError::VmStartFailed(format!("failed to start VM: {e}")))?;

        // 8. Store hypervisor handle and paths
        sandbox.set_hypervisor(hypervisor);
        sandbox.set_api_socket(api_socket);

        // 9. Get PIDs for logging
        let pids = sandbox.get_pids().await;
        tracing::info!(
            sandbox_id = %sandbox.id,
            pids = ?pids,
            "VM started successfully via Kata Hypervisor trait"
        );

        Ok(())
    }

    /// Create a hypervisor instance configured for this sandbox
    ///
    /// Returns the appropriate hypervisor based on `PoolConfig::hypervisor`:
    /// - `CloudHypervisor`: External VMM, requires cloud-hypervisor binary
    /// - `Dragonball`: Built-in VMM (requires dragonball feature)
    async fn create_hypervisor(
        &self,
        sandbox: &PodSandbox,
        api_socket: &Path,
        virtiofs_socket: &Path,
        _cloud_init_iso: &Path,
    ) -> Result<Arc<dyn Hypervisor>> {
        // Build the hypervisor configuration
        let config = self.build_hypervisor_config();

        let hypervisor: Arc<dyn Hypervisor> = match self.config.hypervisor {
            HypervisorType::CloudHypervisor => {
                let ch = CloudHypervisor::new();
                ch.set_hypervisor_config(config).await;
                tracing::debug!(
                    sandbox_id = %sandbox.id,
                    api_socket = %api_socket.display(),
                    virtiofs_socket = %virtiofs_socket.display(),
                    "Created Cloud Hypervisor instance"
                );
                Arc::new(ch)
            }
            #[cfg(feature = "dragonball")]
            HypervisorType::Dragonball => {
                let db = Dragonball::new();
                db.set_hypervisor_config(config).await;
                tracing::debug!(
                    sandbox_id = %sandbox.id,
                    "Created Dragonball hypervisor instance"
                );
                Arc::new(db)
            }
        };

        Ok(hypervisor)
    }

    /// Build a HypervisorConfig from our PoolConfig
    fn build_hypervisor_config(&self) -> HypervisorConfig {
        let mut config = HypervisorConfig::default();

        // VM configuration - use configured path or default based on hypervisor type
        // Note: Kata's canonicalize() requires an absolute path, so we must resolve from PATH
        config.path = if self.config.hypervisor_path.as_os_str().is_empty() {
            // Default path based on hypervisor type - resolve from PATH
            match self.config.hypervisor {
                HypervisorType::CloudHypervisor => {
                    match which::which("cloud-hypervisor") {
                        Ok(path) => path.to_string_lossy().to_string(),
                        Err(e) => {
                            tracing::warn!("Failed to find cloud-hypervisor in PATH: {}", e);
                            // Fallback to name, will fail at Kata's canonicalize() if not found
                            "cloud-hypervisor".to_owned()
                        }
                    }
                }
                #[cfg(feature = "dragonball")]
                HypervisorType::Dragonball => String::new(), // Built-in, no path needed
            }
        } else {
            self.config.hypervisor_path.to_string_lossy().to_string()
        };

        // Boot info (paths need to be converted to String)
        config.boot_info.kernel = self.config.kernel_path.to_string_lossy().to_string();
        config.boot_info.image = self.config.vm_image.to_string_lossy().to_string();

        // Resource limits
        config.cpu_info.default_vcpus = self.config.vm_cpus as f32;
        config.cpu_info.default_maxvcpus = self.config.vm_cpus;
        config.memory_info.default_memory = self.config.vm_memory_mb as u32;

        // Configure rootless user if running in rootless mode
        if rootless::is_rootless() {
            // Get current user info
            let uid = nix::unistd::getuid().as_raw();
            let gid = nix::unistd::getgid().as_raw();
            let username = std::env::var("USER").unwrap_or_else(|_| "user".to_owned());

            // Get supplementary groups
            let groups = nix::unistd::getgroups()
                .map(|gs| gs.into_iter().map(nix::unistd::Gid::as_raw).collect())
                .unwrap_or_else(|_| vec![gid]);

            config.security_info.rootless = true;
            config.security_info.rootless_user = Some(RootlessUser {
                uid,
                gid,
                groups,
                user_name: username,
            });

            tracing::debug!(
                uid = uid,
                gid = gid,
                "Configured rootless user for hypervisor"
            );
        }

        config
    }

    /// Generate cloud-init ISO for the sandbox
    async fn generate_cloud_init_iso(
        &self,
        sandbox: &PodSandbox,
        iso_path: &Path,
    ) -> Result<()> {
        let sandbox_runtime_dir = iso_path.parent().unwrap();

        // Create user-data
        let hostname = if sandbox.metadata.name.is_empty() {
            sandbox.id.clone()
        } else {
            sandbox.metadata.name.clone()
        };

        let user_data = format!(
            r#"#cloud-config
hostname: {}
users:
  - name: root
    lock_passwd: false
write_files:
  - path: /etc/sandbox-id
    content: {}
runcmd:
  - echo "Sandbox {} initialized"
"#,
            hostname, sandbox.id, sandbox.id
        );

        let user_data_path = sandbox_runtime_dir.join("user-data");
        tokio::fs::write(&user_data_path, user_data).await?;

        // Create meta-data
        let meta_data = format!(
            r#"instance-id: {}
local-hostname: {}
"#,
            sandbox.id, hostname
        );

        let meta_data_path = sandbox_runtime_dir.join("meta-data");
        tokio::fs::write(&meta_data_path, meta_data).await?;

        // Generate ISO using genisoimage
        let status = tokio::process::Command::new("genisoimage")
            .args([
                "-output",
                iso_path.to_str().unwrap(),
                "-volid",
                "cidata",
                "-joliet",
                "-rock",
                user_data_path.to_str().unwrap(),
                meta_data_path.to_str().unwrap(),
            ])
            .status()
            .await
            .map_err(|e| WorkerError::CloudInitFailed(format!("failed to run genisoimage: {e}")))?;

        if !status.success() {
            return Err(WorkerError::CloudInitFailed(
                "genisoimage failed".to_owned(),
            ));
        }

        Ok(())
    }

    /// Stop a sandbox's VM using the Hypervisor trait
    async fn stop_sandbox(&self, sandbox: &PodSandbox) -> Result<()> {
        tracing::info!(sandbox_id = %sandbox.id, "Stopping sandbox via Hypervisor trait");

        // Use the Hypervisor trait's stop_vm() method
        if let Some(ref hypervisor) = sandbox.hypervisor {
            // Stop the VM gracefully
            hypervisor
                .stop_vm()
                .await
                .map_err(|e| WorkerError::VmStopFailed(format!("failed to stop VM: {e}")))?;

            tracing::debug!(
                sandbox_id = %sandbox.id,
                "VM stopped via Hypervisor trait"
            );
        } else {
            tracing::warn!(
                sandbox_id = %sandbox.id,
                "No hypervisor handle - sandbox may not have been started"
            );
        }

        // Note: virtiofs_daemon cleanup happens when the Arc is dropped

        tracing::info!(sandbox_id = %sandbox.id, "Sandbox stopped");

        Ok(())
    }

    /// Reset a sandbox for reuse in warm pool
    async fn reset_sandbox(&self, mut sandbox: PodSandbox) -> Result<PodSandbox> {
        tracing::debug!(sandbox_id = %sandbox.id, "Resetting sandbox for warm pool");

        // 1. Clear metadata and labels
        sandbox.metadata = Default::default();
        sandbox.labels.clear();
        sandbox.annotations.clear();
        sandbox.state = PodSandboxState::SandboxNotReady;

        // 2. Clear image-related fields (new sandbox will have different image)
        sandbox.image_id = None;
        sandbox.virtiofs_daemon = None;
        sandbox.virtiofs_socket = None;

        // 3. VM process stays running for warm pool reuse
        // (vm_pid and vm_socket are preserved)

        Ok(sandbox)
    }

    /// Destroy a sandbox completely using the Hypervisor trait
    async fn destroy_sandbox(&self, sandbox: PodSandbox) -> Result<()> {
        tracing::info!(sandbox_id = %sandbox.id, "Destroying sandbox");

        // 1. Ensure VM is stopped and cleaned up via Hypervisor trait
        if let Some(ref hypervisor) = sandbox.hypervisor {
            // First stop if running
            if let Err(e) = hypervisor.stop_vm().await {
                tracing::warn!(
                    sandbox_id = %sandbox.id,
                    error = %e,
                    "Error stopping VM during destroy"
                );
            }

            // Then cleanup resources
            if let Err(e) = hypervisor.cleanup().await {
                tracing::warn!(
                    sandbox_id = %sandbox.id,
                    error = %e,
                    "Error cleaning up hypervisor resources"
                );
            }
        }

        // 2. Clean up runtime directory
        let sandbox_path = sandbox.sandbox_path();
        if sandbox_path.exists() {
            tracing::debug!(
                sandbox_id = %sandbox.id,
                dir = %sandbox_path.display(),
                "Removing sandbox runtime directory"
            );

            // Remove all files in the runtime directory
            if let Err(e) = tokio::fs::remove_dir_all(sandbox_path).await {
                tracing::warn!(
                    sandbox_id = %sandbox.id,
                    error = %e,
                    "Failed to remove sandbox runtime directory"
                );
            }
        }

        // 3. Clean up sockets
        if let Some(ref api_socket) = sandbox.api_socket {
            if api_socket.exists() {
                let _ = tokio::fs::remove_file(api_socket).await;
            }
        }

        if let Some(ref virtiofs_socket) = sandbox.virtiofs_socket {
            if virtiofs_socket.exists() {
                let _ = tokio::fs::remove_file(virtiofs_socket).await;
            }
        }

        tracing::info!(sandbox_id = %sandbox.id, "Sandbox destroyed");

        Ok(())
    }

    /// Replenish warm pool in background
    fn replenish_warm_pool(&self) {
        // TODO: Spawn background task to replenish warm pool
        // For now, this is a no-op
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Number of warm sandboxes ready
    pub warm_count: usize,
    /// Number of active sandboxes in use
    pub active_count: usize,
    /// Maximum allowed sandboxes
    pub max_sandboxes: usize,
    /// Target warm pool size
    pub warm_pool_target: usize,
}

// Note: SandboxPool no longer implements Default since it requires RafsStore
// Use SandboxPool::new() with explicit dependencies

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a test pool with minimal configuration
    async fn create_test_pool(max_sandboxes: usize, warm_pool_size: usize) -> (SandboxPool, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        let pool_config = PoolConfig {
            max_sandboxes,
            warm_pool_size,
            runtime_dir: base_path.join("runtime"),
            ..Default::default()
        };

        let image_config = ImageConfig {
            blobs_dir: base_path.join("blobs"),
            bootstrap_dir: base_path.join("bootstrap"),
            refs_dir: base_path.join("refs"),
            cache_dir: base_path.join("cache"),
            runtime_dir: base_path.join("nydus-runtime"),
            ..Default::default()
        };

        // Create the required directories
        std::fs::create_dir_all(&image_config.blobs_dir).unwrap();
        std::fs::create_dir_all(&image_config.bootstrap_dir).unwrap();
        std::fs::create_dir_all(&image_config.refs_dir).unwrap();
        std::fs::create_dir_all(&image_config.cache_dir).unwrap();
        std::fs::create_dir_all(&pool_config.runtime_dir).unwrap();

        let rafs_store = Arc::new(RafsStore::new(image_config.clone()).unwrap());
        let pool = SandboxPool::new(pool_config, image_config, rafs_store);

        (pool, temp_dir)
    }

    #[tokio::test]
    async fn test_pool_acquire_release() {
        let (pool, _temp_dir) = create_test_pool(5, 0).await;

        // Acquire a sandbox (will fail to start VM since cloud-hypervisor isn't installed,
        // but we can test the pool logic by checking stats)
        let sandbox_config = PodSandboxConfig::default();

        // Note: This will fail in CI since cloud-hypervisor isn't installed
        // We're testing the pool structure, not the actual VM startup
        let result = pool.acquire(&sandbox_config).await;

        // Either it succeeds (cloud-hypervisor available) or fails (not available)
        match result {
            Ok(sandbox_id) => {
                assert!(pool.get(&sandbox_id).await.is_some());
                let stats = pool.stats().await;
                assert_eq!(stats.active_count, 1);

                pool.release(&sandbox_id).await.unwrap();
                assert!(pool.get(&sandbox_id).await.is_none());
            }
            Err(WorkerError::VmStartFailed(_)) => {
                // Expected when cloud-hypervisor is not installed
                let stats = pool.stats().await;
                assert_eq!(stats.active_count, 0);
            }
            Err(e) => panic!("Unexpected error: {e:?}"),
        }
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let (pool, _temp_dir) = create_test_pool(10, 2).await;

        let stats = pool.stats().await;
        assert_eq!(stats.max_sandboxes, 10);
        assert_eq!(stats.warm_pool_target, 2);
        assert_eq!(stats.active_count, 0);
        assert_eq!(stats.warm_count, 0);
    }

    #[tokio::test]
    async fn test_pool_shutdown() {
        let (pool, _temp_dir) = create_test_pool(5, 0).await;

        // Shutdown should succeed even with no sandboxes
        pool.shutdown().await.unwrap();

        let stats = pool.stats().await;
        assert_eq!(stats.active_count, 0);
        assert_eq!(stats.warm_count, 0);
    }
}
