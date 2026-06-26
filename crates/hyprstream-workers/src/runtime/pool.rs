//! Sandbox Pool for warm sandbox management
//!
//! Manages a pool of pre-warmed sandboxes for fast container startup.
//! Delegates sandbox lifecycle to a pluggable `SandboxBackend`.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::PoolConfig;
use crate::error::{Result, WorkerError};

use super::backend::SandboxBackend;
use super::{PodSandbox, PodSandboxConfig, PodSandboxState};

/// Sandbox pool for warm sandbox management
///
/// Maintains:
/// - Warm pool: Pre-created sandboxes ready for immediate use
/// - Active sandboxes: Sandboxes currently in use by workloads
pub struct SandboxPool {
    /// Pre-warmed sandboxes ready for use
    warm_pool: Mutex<VecDeque<PodSandbox>>,

    /// Active sandboxes (in use)
    active: Mutex<HashMap<String, PodSandbox>>,

    /// Pool configuration
    config: PoolConfig,

    /// Pluggable sandbox backend
    backend: Arc<dyn SandboxBackend>,
}

impl SandboxPool {
    /// Create a new sandbox pool with a backend
    pub fn new(config: PoolConfig, backend: Arc<dyn SandboxBackend>) -> Self {
        Self {
            warm_pool: Mutex::new(VecDeque::new()),
            active: Mutex::new(HashMap::new()),
            config,
            backend,
        }
    }

    /// Get a reference to the backend
    pub fn backend(&self) -> &dyn SandboxBackend {
        self.backend.as_ref()
    }

    /// Initialize the warm pool with pre-created sandboxes
    pub async fn initialize(&self) -> Result<()> {
        // Let the backend do one-time setup
        self.backend.initialize(&self.config).await?;

        let mut warm = self.warm_pool.lock().await;

        for _ in 0..self.config.warm_pool_size {
            let sandbox = self.create_warm_sandbox().await?;
            warm.push_back(sandbox);
        }

        tracing::info!(
            backend = self.backend.backend_type(),
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
        self.backend.stop(&sandbox).await?;

        // Decide whether to return to warm pool or destroy
        let warm = self.warm_pool.lock().await;
        if warm.len() < self.config.warm_pool_size {
            drop(warm);
            // Reset and return to warm pool
            let reset_sandbox = self.reset_sandbox(sandbox).await?;
            if let Some(s) = reset_sandbox {
                self.warm_pool.lock().await.push_back(s);
            }
        } else {
            drop(warm);
            // Destroy the sandbox
            self.backend.destroy(&sandbox).await?;
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
            if let Err(e) = self.backend.destroy(&sandbox).await {
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
            if let Err(e) = self.backend.stop(&sandbox).await {
                tracing::warn!("Error stopping sandbox {}: {}", sandbox.id, e);
            }
            if let Err(e) = self.backend.destroy(&sandbox).await {
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

        // Start via backend
        let annotations = HashMap::new();
        let handle = self
            .backend
            .start(&mut sandbox, &config, &self.config, &annotations)
            .await?;
        sandbox.set_backend_handle(handle);

        Ok(sandbox)
    }

    /// Create a sandbox with full configuration
    async fn create_sandbox(&self, config: &PodSandboxConfig) -> Result<PodSandbox> {
        let id = Uuid::new_v4().to_string();
        let sandbox_path = self.config.runtime_dir.join(&id);

        let mut sandbox = PodSandbox::new(id, config, sandbox_path);

        // Convert annotations for backend
        let annotations: HashMap<String, String> = sandbox
            .annotations
            .iter()
            .map(|kv| (kv.key.clone(), kv.value.clone()))
            .collect();

        // Start via backend
        let handle = self
            .backend
            .start(&mut sandbox, config, &self.config, &annotations)
            .await?;
        sandbox.set_backend_handle(handle);

        sandbox.state = PodSandboxState::SandboxReady;

        Ok(sandbox)
    }

    /// Reset a sandbox for reuse in warm pool
    ///
    /// Returns `Some(sandbox)` if the backend supports in-place reset,
    /// `None` if the sandbox is ephemeral and cannot be reused.
    async fn reset_sandbox(&self, mut sandbox: PodSandbox) -> Result<Option<PodSandbox>> {
        tracing::debug!(sandbox_id = %sandbox.id, "Resetting sandbox for warm pool");

        // Clear metadata and labels
        sandbox.metadata = Default::default();
        sandbox.labels.clear();
        sandbox.annotations.clear();
        sandbox.state = PodSandboxState::SandboxNotReady;
        sandbox.image_id = None;

        // Let the backend reset its handle
        let reusable = self.backend.reset(&mut sandbox).await?;
        if reusable {
            Ok(Some(sandbox))
        } else {
            // Backend says sandbox is ephemeral — destroy it
            self.backend.destroy(&sandbox).await?;
            Ok(None)
        }
    }

    /// Replenish warm pool in background
    fn replenish_warm_pool(&self) {
        // TODO: Spawn background task to replenish warm pool
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ImageConfig, PoolConfig};
    use crate::image::RafsStore;
    use crate::runtime::kata_backend::KataBackend;
    use tempfile::TempDir;

    /// Create a test pool with Kata backend and minimal configuration
    async fn create_test_pool(
        max_sandboxes: usize,
        warm_pool_size: usize,
    ) -> Result<(SandboxPool, TempDir)> {
        let temp_dir = TempDir::new()?;
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
        std::fs::create_dir_all(&image_config.blobs_dir)?;
        std::fs::create_dir_all(&image_config.bootstrap_dir)?;
        std::fs::create_dir_all(&image_config.refs_dir)?;
        std::fs::create_dir_all(&image_config.cache_dir)?;
        std::fs::create_dir_all(&pool_config.runtime_dir)?;

        let rafs_store = Arc::new(RafsStore::new(image_config.clone())?);
        let backend: Arc<dyn SandboxBackend> =
            Arc::new(KataBackend::new(image_config, rafs_store));
        let pool = SandboxPool::new(pool_config, backend);

        Ok((pool, temp_dir))
    }

    #[tokio::test]
    async fn test_pool_acquire_release() -> Result<()> {
        let (pool, _temp_dir) = create_test_pool(5, 0).await?;

        let sandbox_config = PodSandboxConfig::default();

        // Note: This will fail in CI since cloud-hypervisor isn't installed
        let result = pool.acquire(&sandbox_config).await;

        match result {
            Ok(sandbox_id) => {
                assert!(pool.get(&sandbox_id).await.is_some());
                let stats = pool.stats().await;
                assert_eq!(stats.active_count, 1);

                pool.release(&sandbox_id).await?;
                assert!(pool.get(&sandbox_id).await.is_none());
            }
            Err(WorkerError::VmStartFailed(_)) => {
                // Expected when cloud-hypervisor is not installed
                let stats = pool.stats().await;
                assert_eq!(stats.active_count, 0);
            }
            Err(e) => panic!("Unexpected error: {e:?}"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_pool_stats() -> Result<()> {
        let (pool, _temp_dir) = create_test_pool(10, 2).await?;

        let stats = pool.stats().await;
        assert_eq!(stats.max_sandboxes, 10);
        assert_eq!(stats.warm_pool_target, 2);
        assert_eq!(stats.active_count, 0);
        assert_eq!(stats.warm_count, 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_pool_shutdown() -> Result<()> {
        let (pool, _temp_dir) = create_test_pool(5, 0).await?;

        // Shutdown should succeed even with no sandboxes
        pool.shutdown().await?;

        let stats = pool.stats().await;
        assert_eq!(stats.active_count, 0);
        assert_eq!(stats.warm_count, 0);
        Ok(())
    }
}
