//! Sandbox Pool for warm sandbox management
//!
//! Manages a pool of pre-warmed sandboxes for fast container startup.
//! Delegates sandbox lifecycle to a pluggable `SandboxBackend`.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::PoolConfig;
use crate::error::{Result, WorkerError};

use super::admission::{self, AdmissionTracker};
use super::backend::SandboxBackend;
use super::client::LinuxContainerResources;
use super::{PodSandbox, PodSandboxConfig, PodSandboxState};
use hyprstream_vfs::Subject;

/// Sandbox pool for warm sandbox management
///
/// Maintains:
/// - Warm pool: Pre-created sandboxes ready for immediate use
/// - Active sandboxes: Sandboxes currently in use by workloads
///
/// # Admission control (#525 P2)
///
/// `acquire` is gated by an [`AdmissionTracker`]: fail-closed identity,
/// per-Subject/per-group quotas, a bounded wait-queue, and resource
/// (cpu/memory/GPU) fit — see `runtime::admission` for the full design and
/// its sign-off-flagged judgment calls.
///
/// # Backend placement (#525 P2, scope item 4)
///
/// A `SandboxPool` is bound to exactly **one** pre-resolved
/// [`SandboxBackend`], selected once at construction via
/// [`super::resolve_backend`] (the fail-closed inventory-registry seam,
/// #507/#516) — never re-selected per request. Routing *individual*
/// `acquire()` calls to different backends (e.g. GPU workloads to `kata`,
/// CPU-only ones to `nspawn`, within the same pool) is **not implemented**:
/// it would require restructuring `SandboxPool` into a router over multiple
/// backend-specific pools, a larger architectural change than this ticket's
/// scope. Flagged as a follow-up, not a silent gap — `acquire` never
/// bypasses the fail-closed backend-resolution seam; it just doesn't have
/// more than one backend to choose among.
pub struct SandboxPool {
    /// Pre-warmed sandboxes ready for use
    warm_pool: Mutex<VecDeque<PodSandbox>>,

    /// Active sandboxes (in use)
    active: Mutex<HashMap<String, PodSandbox>>,

    /// Pool configuration
    config: PoolConfig,

    /// Pluggable sandbox backend
    backend: Arc<dyn SandboxBackend>,

    /// Warm sandboxes currently being created (reserved replenishment slots).
    ///
    /// Counts creations that have been claimed by `replenish_warm_pool` but
    /// have not yet been pushed onto `warm_pool`, so concurrent replenish tasks
    /// can see in-flight work and avoid overshooting `warm_pool_size`.
    in_flight: AtomicUsize,

    /// Admission decision engine (#525 P2): fail-closed identity,
    /// per-Subject/per-group quotas, bounded wait-queue, resource-aware fit.
    admission: AdmissionTracker,
}

impl SandboxPool {
    /// Create a new sandbox pool with a backend
    pub fn new(config: PoolConfig, backend: Arc<dyn SandboxBackend>) -> Self {
        let admission = AdmissionTracker::new(config.admission.clone(), config.max_sandboxes);
        Self {
            warm_pool: Mutex::new(VecDeque::new()),
            active: Mutex::new(HashMap::new()),
            config,
            backend,
            in_flight: AtomicUsize::new(0),
            admission,
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
    ///
    /// # Admission control (#525 P2)
    ///
    /// `subject` must be a verified, non-anonymous [`Subject`] — an
    /// unauthenticated caller is rejected fail-closed before any capacity or
    /// quota bookkeeping happens. Admission (capacity + per-Subject/per-group
    /// quota + cpu/memory/GPU fit) is reserved atomically under one lock
    /// (see `runtime::admission`) before this method ever touches the warm
    /// pool or the backend — never check-then-place.
    ///
    /// A request that only capacity (not quota) blocks may wait, bounded by
    /// `PoolConfig::admission`'s queue settings; see [`WorkerError::QueueFull`]
    /// / [`WorkerError::QueueTimeout`] / [`WorkerError::AdmissionDenied`] for
    /// the distinct failure modes.
    pub async fn acquire(self: &Arc<Self>, subject: &Subject, config: &PodSandboxConfig) -> Result<String> {
        let demand = admission::derive_demand(config);
        let group = admission::derive_group(config);
        let reservation = self.admission.reserve(subject, group.as_deref(), demand).await?;

        let obtained = self.obtain_sandbox(config).await;
        let mut sandbox = match obtained {
            Ok(sandbox) => sandbox,
            Err(e) => {
                // Admission succeeded but no usable sandbox resulted (backend
                // create/reconfigure failure) — give the reservation back
                // rather than leaking capacity/quota.
                self.admission.rollback(&reservation).await;
                return Err(e);
            }
        };

        sandbox.reservation = Some(reservation);
        let sandbox_id = sandbox.id.clone();

        // Move to active
        self.active.lock().await.insert(sandbox_id.clone(), sandbox);

        // Replenish warm pool in background
        Self::replenish_warm_pool(self.clone());

        Ok(sandbox_id)
    }

    /// Pop a warm sandbox (reconfiguring it for `config`, including a resize
    /// via `update_resources` when its previously-applied resources don't
    /// match — the #519 fix) or cold-create one. Does not touch admission
    /// bookkeeping; the caller (`acquire`) owns reserve/rollback around this.
    async fn obtain_sandbox(&self, config: &PodSandboxConfig) -> Result<PodSandbox> {
        let mut warm = self.warm_pool.lock().await;
        if let Some(mut sandbox) = warm.pop_front() {
            drop(warm);
            // Configure the warm sandbox with the provided config
            sandbox.metadata = config.metadata.clone();
            sandbox.labels = config.labels.clone();
            sandbox.annotations = config.annotations.clone();
            sandbox.state = PodSandboxState::SandboxReady;

            // #519 fix: a warm sandbox may have been booted at a different
            // size (or the pool's default) than what THIS caller asked for.
            // Reconfiguring metadata/labels/annotations above must not
            // silently leave a mismatched resource envelope in place.
            let requested = config.linux.resources.clone();
            if requested != LinuxContainerResources::default() && requested != sandbox.applied_resources {
                tracing::debug!(
                    sandbox_id = %sandbox.id,
                    "warm sandbox resource mismatch on reuse — resizing (#519)"
                );
                self.backend.update_resources(&sandbox, &requested).await?;
                sandbox.applied_resources = requested;
            }
            Ok(sandbox)
        } else {
            drop(warm);
            self.create_sandbox(config).await
        }
    }

    /// Release a sandbox back to the pool or destroy it
    pub async fn release(self: &Arc<Self>, sandbox_id: &str) -> Result<()> {
        let mut active = self.active.lock().await;

        let sandbox = active
            .remove(sandbox_id)
            .ok_or_else(|| WorkerError::SandboxNotFound(sandbox_id.to_owned()))?;
        drop(active);

        // Give back the admission reservation (capacity + per-Subject/group
        // quota + resource dimensions) this sandbox was holding, waking any
        // queued waiters. Sandboxes with no reservation (e.g. constructed
        // outside `acquire`) have nothing to give back.
        if let Some(reservation) = sandbox.reservation.clone() {
            self.admission.release(&reservation).await;
        }

        // Stop the sandbox
        self.backend.stop(&sandbox).await?;

        // Decide whether to return to warm pool or destroy
        let warm = self.warm_pool.lock().await;
        if warm.len() < self.config.warm_pool_size {
            drop(warm);
            // Reset and return to warm pool
            let reset_sandbox = self.reset_sandbox(sandbox).await?;
            match reset_sandbox {
                Some(s) => self.warm_pool.lock().await.push_back(s),
                // Backend reported the sandbox as non-reusable and destroyed it,
                // so the warm pool may now be below target — replenish it.
                None => Self::replenish_warm_pool(self.clone()),
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
        // A cold-created sandbox is started with this exact config, so its
        // resource envelope is authoritatively `config.linux.resources`.
        sandbox.applied_resources = config.linux.resources.clone();

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
        // Admission bookkeeping ends with this acquisition; already released
        // above in `release()`. Deliberately NOT resetting `applied_resources`
        // here — see its doc on `PodSandbox` (#519): it must keep reflecting
        // the sandbox's actual physical size for the next `acquire()`'s
        // mismatch check.
        sandbox.reservation = None;

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

    /// Replenish warm pool in background when it drops below target.
    ///
    /// Concurrency-safe: each invocation atomically claims the slots it intends
    /// to fill (the deficit between `warm_pool_size` and the warm count plus any
    /// in-flight creations) while holding the warm-pool lock, then creates that
    /// many sandboxes. Reserving slots up front prevents the TOCTOU race where N
    /// concurrent tasks each observe the same deficit and overshoot the target,
    /// and creating the full deficit (rather than a single sandbox) lets the
    /// pool recover quickly after a burst drains it.
    fn replenish_warm_pool(pool: Arc<Self>) {
        let target = pool.config.warm_pool_size;
        tokio::spawn(async move {
            // Reserve slots while holding the lock so concurrent replenish tasks
            // observe each other's claims via `in_flight`.
            let to_create = {
                let warm = pool.warm_pool.lock().await;
                let in_flight = pool.in_flight.load(Ordering::SeqCst);
                let deficit = target.saturating_sub(warm.len() + in_flight);
                pool.in_flight.fetch_add(deficit, Ordering::SeqCst);
                deficit
            };

            for _ in 0..to_create {
                match pool.create_warm_sandbox().await {
                    Ok(sandbox) => {
                        pool.warm_pool.lock().await.push_back(sandbox);
                        tracing::debug!(target, "Warm pool replenished");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to replenish warm pool");
                    }
                }
                // Release the reserved slot once this creation completes,
                // whether it succeeded or failed.
                pool.in_flight.fetch_sub(1, Ordering::SeqCst);
            }
        });
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

// The pool tests build a KataBackend + RafsStore, so they only run under `kata-vm`.
#[cfg(all(test, feature = "kata-vm"))]
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
    ) -> Result<(Arc<SandboxPool>, TempDir)> {
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
        let pool = Arc::new(SandboxPool::new(pool_config, backend));

        Ok((pool, temp_dir))
    }

    #[tokio::test]
    async fn test_pool_acquire_release() -> Result<()> {
        let (pool, _temp_dir) = create_test_pool(5, 0).await?;

        let sandbox_config = PodSandboxConfig::default();
        let subject = Subject::new("test-user");

        // Note: This will fail in CI since cloud-hypervisor isn't installed
        let result = pool.acquire(&subject, &sandbox_config).await;

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

// #525 P2 — admission/queue/resource-aware placement tests. These use a
// minimal in-memory `FakeBackend` (no kata/nspawn/wasm dependency), so they
// run in the default feature set (unlike the `kata-vm`-gated suite above,
// which needs a real cloud-hypervisor).
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod admission_tests {
    use super::*;
    use crate::runtime::backend::SandboxHandle;
    use crate::runtime::client::KeyValue;
    use parking_lot::Mutex as StdMutex;
    use std::any::Any;
    use std::collections::HashMap as StdHashMap;
    use std::sync::atomic::AtomicUsize as StdAtomicUsize;

    #[derive(Debug)]
    struct FakeHandle;
    impl SandboxHandle for FakeHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Minimal in-memory fake backend. `reset()` returns `true` (reusable,
    /// mirroring Kata) so warm-pool reuse — and the #519 resize path — is
    /// actually exercised. Tracks concurrently-active starts so tests can
    /// assert the admission layer never over-places (the #489/#519 TOCTOU
    /// lesson), and records every `update_resources` call so the #519
    /// regression test can assert a warm-sandbox resize actually happened.
    #[derive(Default)]
    struct FakeBackend {
        current: StdAtomicUsize,
        peak: StdAtomicUsize,
        applied: StdMutex<StdHashMap<String, LinuxContainerResources>>,
    }

    impl FakeBackend {
        fn bump(&self) {
            let cur = self.current.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(cur, Ordering::SeqCst);
        }
    }

    #[async_trait::async_trait]
    impl SandboxBackend for FakeBackend {
        fn backend_type(&self) -> &'static str {
            "fake"
        }
        fn is_available(&self) -> bool {
            true
        }
        async fn initialize(&self, _config: &crate::config::PoolConfig) -> Result<()> {
            Ok(())
        }
        async fn start(
            &self,
            _sandbox: &mut PodSandbox,
            _config: &PodSandboxConfig,
            _pool_config: &crate::config::PoolConfig,
            _annotations: &HashMap<String, String>,
        ) -> Result<Arc<dyn SandboxHandle>> {
            self.bump();
            Ok(Arc::new(FakeHandle))
        }
        async fn stop(&self, _sandbox: &PodSandbox) -> Result<()> {
            self.current.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }
        async fn destroy(&self, _sandbox: &PodSandbox) -> Result<()> {
            Ok(())
        }
        async fn reset(&self, _sandbox: &mut PodSandbox) -> Result<bool> {
            // Reusable (Kata-like) — required to exercise the warm-pool /
            // #519 resize path below.
            Ok(true)
        }
        async fn get_pids(&self, _sandbox: &PodSandbox) -> Result<Vec<u32>> {
            Ok(vec![])
        }
        fn supports_exec(&self) -> bool {
            false
        }
        async fn exec_sync(
            &self,
            _sandbox: &PodSandbox,
            _command: &[String],
            _timeout_secs: u64,
        ) -> Result<(i32, Vec<u8>, Vec<u8>)> {
            Err(WorkerError::ExecFailed("not supported".into()))
        }
        async fn update_resources(
            &self,
            sandbox: &PodSandbox,
            resources: &LinuxContainerResources,
        ) -> Result<()> {
            self.applied.lock().insert(sandbox.id.clone(), resources.clone());
            Ok(())
        }
    }

    fn make_pool_with(max_sandboxes: usize, warm_pool_size: usize) -> (Arc<SandboxPool>, Arc<FakeBackend>) {
        let fake = Arc::new(FakeBackend::default());
        let backend: Arc<dyn SandboxBackend> = fake.clone();
        let config = PoolConfig { max_sandboxes, warm_pool_size, ..Default::default() };
        (Arc::new(SandboxPool::new(config, backend)), fake)
    }

    fn make_pool_with_admission(
        max_sandboxes: usize,
        warm_pool_size: usize,
        admission: crate::runtime::AdmissionConfig,
    ) -> (Arc<SandboxPool>, Arc<FakeBackend>) {
        let fake = Arc::new(FakeBackend::default());
        let backend: Arc<dyn SandboxBackend> = fake.clone();
        let config = PoolConfig { max_sandboxes, warm_pool_size, admission, ..Default::default() };
        (Arc::new(SandboxPool::new(config, backend)), fake)
    }

    // ── Solo/no-regression baseline ──────────────────────────────────────

    #[tokio::test]
    async fn solo_acquire_release_unchanged() {
        let (pool, _fake) = make_pool_with(5, 0);
        let subject = Subject::new("alice");
        let sandbox_config = PodSandboxConfig::default();

        let id = pool.acquire(&subject, &sandbox_config).await.expect("solo acquire must succeed");
        assert!(pool.get(&id).await.is_some());
        assert_eq!(pool.stats().await.active_count, 1);

        pool.release(&id).await.expect("release must succeed");
        assert!(pool.get(&id).await.is_none());
        assert_eq!(pool.stats().await.active_count, 0);
    }

    #[tokio::test]
    async fn solo_capacity_exhausted_without_queue_room_rejects_clearly() {
        let cfg = crate::runtime::AdmissionConfig { queue_capacity: 0, ..Default::default() };
        let (pool, _fake) = make_pool_with_admission(1, 0, cfg);
        let subject = Subject::new("alice");
        let sandbox_config = PodSandboxConfig::default();

        let _id = pool.acquire(&subject, &sandbox_config).await.expect("first acquire admits");
        let err = pool.acquire(&subject, &sandbox_config).await.unwrap_err();
        assert!(matches!(err, WorkerError::QueueFull { .. }), "got: {err:?}");
    }

    // ── Fail-closed identity ─────────────────────────────────────────────

    #[tokio::test]
    async fn anonymous_caller_rejected_fail_closed() {
        let (pool, _fake) = make_pool_with(5, 0);
        let sandbox_config = PodSandboxConfig::default();
        let err = pool.acquire(&Subject::anonymous(), &sandbox_config).await.unwrap_err();
        assert!(matches!(err, WorkerError::Unauthorized { .. }), "got: {err:?}");
        // Must not have consumed any capacity.
        assert_eq!(pool.stats().await.active_count, 0);
    }

    // ── Per-Subject / per-group quota ────────────────────────────────────

    #[tokio::test]
    async fn per_subject_quota_rejects_over_quota_caller() {
        let cfg = crate::runtime::AdmissionConfig { max_per_subject: 1, ..Default::default() };
        let (pool, _fake) = make_pool_with_admission(10, 0, cfg);
        let alice = Subject::new("alice");
        let sandbox_config = PodSandboxConfig::default();

        let _id = pool.acquire(&alice, &sandbox_config).await.expect("first request admitted");
        let err = pool.acquire(&alice, &sandbox_config).await.unwrap_err();
        assert!(matches!(err, WorkerError::AdmissionDenied { .. }), "got: {err:?}");
    }

    // ── GPU-aware placement (fit over the #628 scheduling substrate) ─────

    #[tokio::test]
    async fn gpu_request_placed_only_when_capacity_declared() {
        let cfg = crate::runtime::AdmissionConfig { gpu_total: 0, ..Default::default() };
        let (pool, _fake) = make_pool_with_admission(10, 0, cfg);
        let subject = Subject::new("alice");
        let mut sandbox_config = PodSandboxConfig::default();
        sandbox_config
            .annotations
            .push(KeyValue { key: crate::runtime::ANN_GPU_REQUEST.to_owned(), value: "1".to_owned() });

        // No GPU capacity declared → fail-closed reject, never a silent
        // CPU-only placement.
        let err = pool.acquire(&subject, &sandbox_config).await.unwrap_err();
        assert!(matches!(err, WorkerError::QueueFull { .. } | WorkerError::QueueTimeout { .. }), "got: {err:?}");
    }

    #[tokio::test]
    async fn gpu_request_placed_when_capacity_available() {
        let cfg = crate::runtime::AdmissionConfig { gpu_total: 1, ..Default::default() };
        let (pool, _fake) = make_pool_with_admission(10, 0, cfg);
        let subject = Subject::new("alice");
        let mut sandbox_config = PodSandboxConfig::default();
        sandbox_config
            .annotations
            .push(KeyValue { key: crate::runtime::ANN_GPU_REQUEST.to_owned(), value: "1".to_owned() });

        let id = pool.acquire(&subject, &sandbox_config).await.expect("gpu request should be admitted");
        assert!(pool.get(&id).await.is_some());
    }

    #[tokio::test]
    async fn cpu_and_memory_request_honored_within_capacity() {
        let cfg = crate::runtime::AdmissionConfig {
            cpu_millis_total: Some(1000),
            memory_bytes_total: Some(1024 * 1024 * 1024),
            ..Default::default()
        };
        let (pool, _fake) = make_pool_with_admission(10, 0, cfg);
        let subject = Subject::new("alice");
        let mut sandbox_config = PodSandboxConfig::default();
        sandbox_config.linux.resources.cpu_period = 100_000;
        sandbox_config.linux.resources.cpu_quota = 50_000; // 500m
        sandbox_config.linux.resources.memory_limit_in_bytes = 512 * 1024 * 1024;

        let id = pool.acquire(&subject, &sandbox_config).await.expect("fits within declared capacity");
        assert!(pool.get(&id).await.is_some());
    }

    #[tokio::test]
    async fn memory_request_exceeding_capacity_is_not_silently_placed() {
        let cfg = crate::runtime::AdmissionConfig {
            memory_bytes_total: Some(256 * 1024 * 1024),
            queue_capacity: 0,
            ..Default::default()
        };
        let (pool, _fake) = make_pool_with_admission(10, 0, cfg);
        let subject = Subject::new("alice");
        let mut sandbox_config = PodSandboxConfig::default();
        sandbox_config.linux.resources.memory_limit_in_bytes = 512 * 1024 * 1024; // exceeds declared 256Mi

        let err = pool.acquire(&subject, &sandbox_config).await.unwrap_err();
        assert!(matches!(err, WorkerError::QueueFull { .. }), "got: {err:?}");
    }

    // ── #519 regression: warm-sandbox reuse must resize, never silently
    //    hand out the wrong size ──────────────────────────────────────────

    #[tokio::test]
    async fn warm_sandbox_reuse_resizes_mismatched_resources() {
        let (pool, fake) = make_pool_with(5, 1);
        pool.initialize().await.expect("prewarm one sandbox");
        assert_eq!(pool.stats().await.warm_count, 1);

        let subject = Subject::new("alice");
        let mut sandbox_config = PodSandboxConfig::default();
        sandbox_config.linux.resources.cpu_period = 100_000;
        sandbox_config.linux.resources.cpu_quota = 75_000;
        sandbox_config.linux.resources.memory_limit_in_bytes = 256 * 1024 * 1024;

        // The one warm sandbox was booted with `PodSandboxConfig::default()`
        // (all-zero resources) — this request asks for something different,
        // so acquire() must detect the mismatch and resize it rather than
        // silently handing out the warm sandbox's original (wrong) size.
        let id = pool.acquire(&subject, &sandbox_config).await.expect("warm reuse should succeed");

        let applied = fake.applied.lock();
        let recorded = applied.get(&id).expect("update_resources must have been called for this sandbox");
        assert_eq!(recorded.memory_limit_in_bytes, 256 * 1024 * 1024);
        assert_eq!(recorded.cpu_quota, 75_000);
    }

    #[tokio::test]
    async fn cold_created_sandbox_needs_no_extra_resize_call() {
        // warm_pool_size 0 → always cold-creates. `start()` receives the
        // full config directly, so there should be no separate
        // `update_resources` call layered on top.
        let (pool, fake) = make_pool_with(5, 0);
        let subject = Subject::new("alice");
        let mut sandbox_config = PodSandboxConfig::default();
        sandbox_config.linux.resources.memory_limit_in_bytes = 128 * 1024 * 1024;

        let id = pool.acquire(&subject, &sandbox_config).await.expect("cold create should succeed");
        assert!(fake.applied.lock().get(&id).is_none());
    }

    // ── Concurrency / TOCTOU (#489/#519 lesson): never over-place ─────────

    #[tokio::test]
    async fn concurrent_acquires_never_exceed_capacity() {
        let capacity = 2usize;
        let n = 6usize;
        let cfg = crate::runtime::AdmissionConfig {
            queue_capacity: n,
            queue_timeout_secs: 1,
            ..Default::default()
        };
        let (pool, fake) = make_pool_with_admission(capacity, 0, cfg);

        let mut handles = Vec::new();
        for i in 0..n {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let subject = Subject::new(format!("user-{i}"));
                pool.acquire(&subject, &PodSandboxConfig::default()).await
            }));
        }

        let mut ok = 0usize;
        let mut failed = 0usize;
        for h in handles {
            match h.await.expect("task must not panic") {
                Ok(_) => ok += 1,
                Err(e) => {
                    failed += 1;
                    assert!(
                        matches!(e, WorkerError::QueueFull { .. } | WorkerError::QueueTimeout { .. }),
                        "unexpected error: {e:?}"
                    );
                }
            }
        }

        assert_eq!(ok, capacity, "exactly `capacity` requests must be admitted");
        assert_eq!(failed, n - capacity, "the rest must queue-then-timeout or reject, never silently succeed");
        // The backend itself must never have seen more concurrently-started
        // sandboxes than the declared capacity — the actual TOCTOU check.
        assert!(
            fake.peak.load(Ordering::SeqCst) <= capacity,
            "backend saw more concurrent starts than capacity allows: {}",
            fake.peak.load(Ordering::SeqCst)
        );
        assert_eq!(pool.stats().await.active_count, capacity);
    }

    #[tokio::test]
    async fn queued_request_is_admitted_once_capacity_frees() {
        let cfg = crate::runtime::AdmissionConfig { queue_timeout_secs: 5, ..Default::default() };
        let (pool, _fake) = make_pool_with_admission(1, 0, cfg);
        let subject = Subject::new("alice");
        let sandbox_config = PodSandboxConfig::default();

        let first = pool.acquire(&subject, &sandbox_config).await.expect("first request admitted");

        let pool2 = pool.clone();
        let waiter = tokio::spawn(async move {
            let bob = Subject::new("bob");
            pool2.acquire(&bob, &PodSandboxConfig::default()).await
        });

        // Give the waiter a moment to actually enter the queue, then free
        // capacity by releasing the first sandbox.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        pool.release(&first).await.expect("release must succeed");

        let result = tokio::time::timeout(std::time::Duration::from_secs(2), waiter)
            .await
            .expect("queued waiter should be woken well within 2s")
            .expect("task must not panic");
        assert!(result.is_ok(), "queued request should be admitted once capacity frees: {result:?}");
    }
}
