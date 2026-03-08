//! Per-tenant delta registry with LRU eviction
//!
//! Manages a pool of `TenantDelta` instances, one per tenant/session.
//! Pattern mirrors `KVCacheRegistry` using `DashMap` for concurrent access.

use anyhow::Result;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tch::Device;

use super::tenant_delta::{TenantDelta, TenantDeltaConfig, serialize_state_dict_to_bytes};
use crate::runtime::kv_cache::KVCacheRegistry;
use crate::services::WorktreeClient;
use hyprstream_rpc::Subject;

/// Maximum snapshot size in bytes (512 MB). Deltas exceeding this are not snapshotted on eviction.
const MAX_SNAPSHOT_BYTES: usize = 512 * 1024 * 1024;

/// Default maximum number of tenants before LRU eviction kicks in.
const MAX_TENANTS_DEFAULT: usize = 100;

/// Registry managing per-tenant LoRA deltas for isolated TTT adaptation
///
/// Each `Subject` gets its own isolated `TenantDelta` wrapped in `Arc<Mutex<_>>`.
/// Uses `DashMap` for lock-free access to different tenants.
pub struct DeltaPool {
    /// Active deltas indexed by tenant
    deltas: DashMap<Subject, Arc<Mutex<TenantDelta>>>,
    /// Default configuration for new deltas (behind Mutex for interior mutability via update_config)
    default_config: Mutex<TenantDeltaConfig>,
    /// Module dimensions from the loaded model: module_name -> (in_features, out_features)
    module_dims: HashMap<String, (usize, usize)>,
    /// Per-layer dimension overrides: layer_idx -> module_name -> (in, out)
    /// Used for hybrid architectures (e.g., Qwen3.5) where GDN and full-attn layers
    /// have different dimensions for the same LoRA module name (e.g., "o_proj").
    per_layer_dims: Option<HashMap<usize, HashMap<String, (usize, usize)>>>,
    /// Device for tensor allocation
    device: Device,
    /// Memory budget in bytes (None = unlimited)
    memory_budget_bytes: Option<usize>,
    /// Base weight norms for drift monitoring: module_name -> ||W||
    base_weight_norms: HashMap<String, f64>,
    /// KV cache registry for dependency-aware eviction.
    /// When a delta is evicted or reset, dependent KV caches are invalidated.
    kv_registry: Option<Arc<KVCacheRegistry>>,
    /// Directory for eviction snapshots
    snapshots_dir: PathBuf,
    /// Optional WorktreeClient for worktree-scoped file operations
    fs: Option<WorktreeClient>,
    /// Number of model layers for per-layer delta creation
    num_layers: usize,
    /// Maximum number of tenants before LRU eviction
    max_tenants: usize,
}

impl DeltaPool {
    /// Create a new delta pool
    ///
    /// # Arguments
    /// * `config` - Default configuration for new deltas
    /// * `module_dims` - Module dimensions from model_info
    /// * `device` - Device for tensor allocation
    /// * `kv_registry` - Optional KV cache registry for dependency-aware eviction
    /// * `snapshots_dir` - Directory to write eviction snapshots
    /// * `fs` - Optional WorktreeClient for worktree-scoped file operations
    pub fn new(
        config: TenantDeltaConfig,
        module_dims: HashMap<String, (usize, usize)>,
        device: Device,
        kv_registry: Option<Arc<KVCacheRegistry>>,
        snapshots_dir: PathBuf,
        fs: Option<WorktreeClient>,
        num_layers: usize,
    ) -> Self {
        Self {
            deltas: DashMap::new(),
            default_config: Mutex::new(config),
            module_dims,
            per_layer_dims: None,
            device,
            memory_budget_bytes: None,
            base_weight_norms: HashMap::new(),
            kv_registry,
            snapshots_dir,
            fs,
            num_layers,
            max_tenants: MAX_TENANTS_DEFAULT,
        }
    }

    /// Set per-layer dimension overrides for hybrid architectures (e.g., Qwen3.5).
    pub fn with_per_layer_dims(
        mut self,
        per_layer_dims: HashMap<usize, HashMap<String, (usize, usize)>>,
    ) -> Self {
        self.per_layer_dims = Some(per_layer_dims);
        self
    }

    /// Set memory budget for the pool
    pub fn with_memory_budget(mut self, budget_bytes: usize) -> Self {
        self.memory_budget_bytes = Some(budget_bytes);
        self
    }

    /// Set maximum number of tenants before LRU eviction
    pub fn with_max_tenants(mut self, max_tenants: usize) -> Self {
        self.max_tenants = max_tenants;
        self
    }

    /// Update the default config for new deltas (e.g., when createLora changes target modules)
    ///
    /// Also clears existing deltas since their module sets would be stale.
    pub fn update_config(&self, config: TenantDeltaConfig) {
        // Clear existing deltas — they have the old module set
        self.clear_all();
        *self.default_config.lock() = config;
    }

    /// Set base weight norms for drift monitoring
    pub fn set_base_weight_norms(&mut self, norms: HashMap<String, f64>) {
        self.base_weight_norms = norms;
    }

    /// Set the KV cache registry for dependency-aware eviction
    pub fn set_kv_registry(&mut self, registry: Arc<KVCacheRegistry>) {
        self.kv_registry = Some(registry);
    }

    /// Get or create a delta for the given tenant
    ///
    /// If the delta doesn't exist, creates a new one with the default config.
    pub fn get_or_create(&self, tenant_id: &Subject) -> Result<Arc<Mutex<TenantDelta>>> {
        // Fast path: existing delta
        if let Some(delta) = self.deltas.get(tenant_id) {
            delta.lock().touch();
            return Ok(delta.clone());
        }

        // Slow path: create new delta (lock config briefly to clone it)
        let config = self.default_config.lock().clone();
        let delta = TenantDelta::new_with_per_layer_dims(
            &config,
            &self.module_dims,
            self.device,
            self.num_layers,
            self.per_layer_dims.as_ref(),
        )?;
        let delta = Arc::new(Mutex::new(delta));

        // Insert (handles race condition)
        Ok(self
            .deltas
            .entry(tenant_id.clone())
            .or_insert(delta.clone())
            .clone())
    }

    /// Get an existing delta (returns None if not found)
    pub fn get(&self, tenant_id: &Subject) -> Option<Arc<Mutex<TenantDelta>>> {
        self.deltas.get(tenant_id).map(|d| {
            d.lock().touch();
            d.clone()
        })
    }

    /// Remove a subject's delta and invalidate dependent KV caches
    pub fn remove(&self, subject: &Subject) -> Option<Arc<Mutex<TenantDelta>>> {
        let result = self.deltas.remove(subject).map(|(_, d)| d);
        if result.is_some() {
            if let Some(kv_reg) = &self.kv_registry {
                kv_reg.invalidate_for_tenant(&subject.to_string());
            }
        }
        result
    }

    /// List all subjects in the pool (for debugging)
    pub fn list_subjects(&self) -> Vec<String> {
        self.deltas.iter().map(|e| e.key().to_string()).collect()
    }

    /// Get number of active tenants
    pub fn tenant_count(&self) -> usize {
        self.deltas.len()
    }

    /// Get total memory usage across all deltas in bytes
    pub fn total_memory_usage(&self) -> usize {
        self.deltas
            .iter()
            .map(|entry| entry.value().lock().memory_bytes())
            .sum()
    }

    /// Get a clone of the current default config
    pub fn default_config(&self) -> TenantDeltaConfig {
        self.default_config.lock().clone()
    }

    /// Get the LoRA rank used by this pool's deltas
    pub fn rank(&self) -> usize {
        self.default_config.lock().rank
    }

    /// Get module dimensions reference
    pub fn module_dims(&self) -> &HashMap<String, (usize, usize)> {
        &self.module_dims
    }

    /// Get base weight norms reference
    pub fn base_weight_norms(&self) -> &HashMap<String, f64> {
        &self.base_weight_norms
    }

    /// Get snapshots directory reference
    pub fn snapshots_dir(&self) -> &PathBuf {
        &self.snapshots_dir
    }

    /// Check if eviction is needed (tenant count or memory budget exceeded).
    ///
    /// This is a fast, non-blocking check with no I/O.
    pub fn needs_eviction(&self) -> bool {
        if self.deltas.len() >= self.max_tenants {
            return true;
        }
        if let Some(budget) = self.memory_budget_bytes {
            if self.total_memory_usage() > budget {
                return true;
            }
        }
        false
    }

    /// Ensure the pool has capacity for at least one more tenant.
    ///
    /// Call this from async context **before** `get_or_create()` to avoid
    /// `Handle::block_on()` panics inside `spawn_local` on `current_thread` runtime.
    ///
    /// Evicts LRU tenants (with async snapshot I/O) until capacity is available.
    pub async fn ensure_capacity(&self) -> Result<()> {
        while self.needs_eviction() {
            if self.evict_lru_async().await?.is_none() {
                break; // Pool is empty, nothing left to evict
            }
        }
        Ok(())
    }

    /// Sanitize a subject string for use as a filename.
    fn sanitize_filename(subject: &str) -> String {
        subject
            .replace(['/', '\\', '\0'], "_")
            .replace("..", "_")
            .chars()
            .take(200)
            .collect()
    }

    /// Find the LRU (least recently used) tenant ID.
    ///
    /// Returns `None` if the pool is empty.
    fn find_lru_tenant(&self) -> Option<Subject> {
        let mut oldest_tenant: Option<Subject> = None;
        let mut oldest_time = std::time::Instant::now();

        for entry in self.deltas.iter() {
            let delta = entry.value().lock();
            if delta.last_access < oldest_time {
                oldest_time = delta.last_access;
                oldest_tenant = Some(entry.key().clone());
            }
        }
        oldest_tenant
    }

    /// Async eviction of the least-recently-used tenant delta.
    ///
    /// Before eviction:
    /// 1. Snapshots the delta to a file (if it has accumulated steps) using async I/O
    /// 2. Invalidates dependent KV caches (if kv_registry is set)
    ///
    /// Returns the evicted tenant ID, delta, and optional snapshot path.
    pub async fn evict_lru_async(
        &self,
    ) -> Result<Option<(Subject, Arc<Mutex<TenantDelta>>, Option<PathBuf>)>> {
        if self.deltas.is_empty() {
            return Ok(None);
        }

        let Some(tid) = self.find_lru_tenant() else {
            return Ok(None);
        };

        let Some((id, delta)) = self.deltas.remove(&tid) else {
            return Ok(None);
        };

        let id_filename = Self::sanitize_filename(&id.to_string());

        // Auto-snapshot to file before eviction
        // Extract data under lock, then release before any async I/O
        let snapshot_data = {
            let d = delta.lock();
            if d.accumulated_steps > 0 {
                if self.fs.is_some() {
                    let state_dict = d.extract_state_dict();
                    let memory_bytes = d.memory_bytes();
                    Some((state_dict, memory_bytes))
                } else {
                    tracing::warn!(
                        "FsOps not available — skipping snapshot for delta '{}' during eviction",
                        id
                    );
                    None
                }
            } else {
                None // No accumulated steps, skip snapshot
            }
        }; // lock released here

        let snapshot_path = if let Some((state_dict, memory_bytes)) = snapshot_data {
            let snapshot_file = self
                .snapshots_dir
                .join(format!("{}.safetensors", id_filename));
            // Safety: snapshot_data is Some only when self.fs.is_some() was true above
            let Some(ref fs) = self.fs else { unreachable!() };
            let rel_path = format!("adapters/.snapshots/{}.safetensors", id_filename);

            if memory_bytes > MAX_SNAPSHOT_BYTES {
                tracing::warn!(
                    "Delta '{}' exceeds max snapshot size ({} bytes > {}), skipping snapshot",
                    id, memory_bytes, MAX_SNAPSHOT_BYTES
                );
                None
            } else {
                match async {
                    fs.mkdir_p("adapters/.snapshots").await.map_err(|e| {
                        anyhow::anyhow!("FsOps mkdir failed: {}", e)
                    })?;
                    let bytes = serialize_state_dict_to_bytes(&state_dict)?;
                    fs.write_file_chunked(&rel_path, &bytes).await.map_err(
                        |e| anyhow::anyhow!("FsOps write_file failed: {}", e),
                    )?;
                    Ok::<(), anyhow::Error>(())
                }
                .await
                {
                    Ok(()) => {
                        tracing::info!(
                            "Auto-snapshot delta '{}' before eviction via FsOps: {}",
                            id, rel_path
                        );
                        Some(snapshot_file)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to snapshot delta '{}' via FsOps: {}",
                            id, e
                        );
                        None
                    }
                }
            }
        } else {
            None
        };

        // Invalidate dependent KV caches
        if let Some(kv_reg) = &self.kv_registry {
            let invalidated = kv_reg.invalidate_for_tenant(&id.to_string());
            if invalidated > 0 {
                tracing::info!(
                    "Invalidated {} KV caches on eviction of delta '{}'",
                    invalidated,
                    id
                );
            }
        }

        Ok(Some((id, delta, snapshot_path)))
    }

    /// Evict the least-recently-used tenant delta (sync version).
    ///
    /// **WARNING**: This uses `futures::executor::block_on()` for snapshot I/O.
    /// Only safe to call from non-async contexts (tests, standalone threads).
    /// In async contexts, use `evict_lru_async()` instead.
    pub fn evict_lru(
        &self,
    ) -> Option<(Subject, Arc<Mutex<TenantDelta>>, Option<PathBuf>)> {
        // For sync callers, we can't do async I/O. Remove without snapshot
        // if there's no WorktreeClient, or use block_on only in non-async contexts.
        if self.deltas.is_empty() {
            return None;
        }

        let tid = self.find_lru_tenant()?;
        let (id, delta) = self.deltas.remove(&tid)?;

        let id_filename = Self::sanitize_filename(&id.to_string());

        // Snapshot without async I/O — extract state dict only, skip FsOps writes
        let snapshot_path = {
            let d = delta.lock();
            if d.accumulated_steps > 0 {
                let snapshot_file = self
                    .snapshots_dir
                    .join(format!("{}.safetensors", id_filename));
                // In sync context, we can only write directly to the snapshots_dir
                // (not through WorktreeClient which requires async)
                let state_dict = d.extract_state_dict();
                let memory_bytes = d.memory_bytes();
                drop(d);

                if memory_bytes > MAX_SNAPSHOT_BYTES {
                    tracing::warn!(
                        "Delta '{}' exceeds max snapshot size ({} bytes > {}), skipping snapshot",
                        id, memory_bytes, MAX_SNAPSHOT_BYTES
                    );
                    None
                } else {
                    match serialize_state_dict_to_bytes(&state_dict) {
                        Ok(bytes) => {
                            if let Err(e) = std::fs::create_dir_all(&self.snapshots_dir) {
                                tracing::warn!("Failed to create snapshots dir: {}", e);
                                None
                            } else if let Err(e) = std::fs::write(&snapshot_file, &bytes) {
                                tracing::warn!(
                                    "Failed to write snapshot for delta '{}': {}",
                                    id, e
                                );
                                None
                            } else {
                                tracing::info!(
                                    "Auto-snapshot delta '{}' before eviction: {}",
                                    id,
                                    snapshot_file.display()
                                );
                                Some(snapshot_file)
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to serialize delta '{}' for snapshot: {}",
                                id, e
                            );
                            None
                        }
                    }
                }
            } else {
                None
            }
        };

        // Invalidate dependent KV caches
        if let Some(kv_reg) = &self.kv_registry {
            let invalidated = kv_reg.invalidate_for_tenant(&id.to_string());
            if invalidated > 0 {
                tracing::info!(
                    "Invalidated {} KV caches on eviction of delta '{}'",
                    invalidated,
                    id
                );
            }
        }

        Some((id, delta, snapshot_path))
    }

    /// Async version: evict deltas until total memory is under budget.
    ///
    /// Returns a list of evicted (tenant_id, snapshot_path) pairs for audit/recovery.
    pub async fn evict_to_budget_async(&self) -> Result<Vec<(Subject, Option<PathBuf>)>> {
        let budget = match self.memory_budget_bytes {
            Some(b) => b,
            None => return Ok(Vec::new()),
        };

        let mut evicted = Vec::new();
        while self.total_memory_usage() > budget {
            match self.evict_lru_async().await? {
                Some((tid, _, path)) => evicted.push((tid, path)),
                None => break,
            }
        }
        Ok(evicted)
    }

    /// Evict deltas until total memory is under budget (sync version).
    ///
    /// **WARNING**: Only safe in non-async contexts. See `evict_lru()` docs.
    pub fn evict_to_budget(&self) -> Vec<(Subject, Option<PathBuf>)> {
        let budget = match self.memory_budget_bytes {
            Some(b) => b,
            None => return Vec::new(),
        };

        let mut evicted = Vec::new();
        while self.total_memory_usage() > budget {
            match self.evict_lru() {
                Some((tid, _, path)) => evicted.push((tid, path)),
                None => break,
            }
        }
        evicted
    }

    /// Clear all deltas and invalidate all dependent KV caches
    pub fn clear_all(&self) {
        // Invalidate KV caches for all active subjects
        if let Some(kv_reg) = &self.kv_registry {
            for entry in self.deltas.iter() {
                kv_reg.invalidate_for_tenant(&entry.key().to_string());
            }
        }
        self.deltas.clear();
    }

    /// List all tenant IDs
    pub fn tenant_ids(&self) -> Vec<Subject> {
        self.deltas.iter().map(|e| e.key().clone()).collect()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_module_dims() -> HashMap<String, (usize, usize)> {
        let mut dims = HashMap::new();
        dims.insert("q_proj".to_owned(), (512, 512));
        dims.insert("v_proj".to_owned(), (512, 512));
        dims
    }

    fn test_snapshots_dir() -> PathBuf {
        std::env::temp_dir().join("delta_pool_test_snapshots")
    }

    #[test]
    fn test_delta_pool_creation() {
        let pool = DeltaPool::new(
            TenantDeltaConfig::default(),
            test_module_dims(),
            Device::Cpu,
            None,
            test_snapshots_dir(),
            None,
            2,
        );
        assert_eq!(pool.tenant_count(), 0);
    }

    #[test]
    fn test_get_or_create() {
        let pool = DeltaPool::new(
            TenantDeltaConfig::default(),
            test_module_dims(),
            Device::Cpu,
            None,
            test_snapshots_dir(),
            None,
            2,
        );

        let tid_a = Subject::new("tenant-a");
        let tid_b = Subject::new("tenant-b");

        let delta_a = pool.get_or_create(&tid_a).unwrap();
        let delta_b = pool.get_or_create(&tid_b).unwrap();

        assert_eq!(pool.tenant_count(), 2);

        // Same tenant returns same delta
        let delta_a2 = pool.get_or_create(&tid_a).unwrap();
        assert!(Arc::ptr_eq(&delta_a, &delta_a2));

        // Different tenants are different
        assert!(!Arc::ptr_eq(&delta_a, &delta_b));
    }

    #[test]
    fn test_tenant_isolation() {
        let pool = DeltaPool::new(
            TenantDeltaConfig::default(),
            test_module_dims(),
            Device::Cpu,
            None,
            test_snapshots_dir(),
            None,
            2,
        );

        let tid_a = Subject::new("tenant-a");
        let tid_b = Subject::new("tenant-b");

        let delta_a = pool.get_or_create(&tid_a).unwrap();
        let delta_b = pool.get_or_create(&tid_b).unwrap();

        // Modify A's delta
        {
            let mut a = delta_a.lock();
            a.accumulated_steps = 42;
        }

        // B should be unaffected
        {
            let b = delta_b.lock();
            assert_eq!(b.accumulated_steps, 0);
        }
    }

    #[test]
    fn test_lru_eviction() {
        let pool = DeltaPool::new(
            TenantDeltaConfig::default(),
            test_module_dims(),
            Device::Cpu,
            None,
            test_snapshots_dir(),
            None,
            2,
        );

        let tid_a = Subject::new("tenant-a");
        let tid_b = Subject::new("tenant-b");
        let tid_c = Subject::new("tenant-c");

        let _a = pool.get_or_create(&tid_a).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _b = pool.get_or_create(&tid_b).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _c = pool.get_or_create(&tid_c).unwrap();

        assert_eq!(pool.tenant_count(), 3);

        // Evict LRU should remove tenant-a (oldest)
        let evicted = pool.evict_lru();
        assert!(evicted.is_some());
        let (evicted_id, _, _snapshot_path) = evicted.unwrap();
        assert_eq!(evicted_id, tid_a);
        assert_eq!(pool.tenant_count(), 2);
    }

    #[test]
    fn test_total_memory_usage() {
        let pool = DeltaPool::new(
            TenantDeltaConfig::default(),
            test_module_dims(),
            Device::Cpu,
            None,
            test_snapshots_dir(),
            None,
            2,
        );

        let tid = Subject::new("tenant");
        let _delta = pool.get_or_create(&tid).unwrap();

        let usage = pool.total_memory_usage();
        assert!(usage > 0, "Memory usage should be non-zero with a delta");
    }
}
