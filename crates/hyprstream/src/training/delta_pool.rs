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
use crate::services::FsOps;
use hyprstream_rpc::Subject;

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
    /// Optional FsOps for worktree-scoped file operations
    fs: Option<Arc<dyn FsOps>>,
    /// Number of model layers for per-layer delta creation
    num_layers: usize,
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
    /// * `fs` - Optional FsOps for worktree-scoped file operations
    pub fn new(
        config: TenantDeltaConfig,
        module_dims: HashMap<String, (usize, usize)>,
        device: Device,
        kv_registry: Option<Arc<KVCacheRegistry>>,
        snapshots_dir: PathBuf,
        fs: Option<Arc<dyn FsOps>>,
        num_layers: usize,
    ) -> Self {
        Self {
            deltas: DashMap::new(),
            default_config: Mutex::new(config),
            module_dims,
            device,
            memory_budget_bytes: None,
            base_weight_norms: HashMap::new(),
            kv_registry,
            snapshots_dir,
            fs,
            num_layers,
        }
    }

    /// Set memory budget for the pool
    pub fn with_memory_budget(mut self, budget_bytes: usize) -> Self {
        self.memory_budget_bytes = Some(budget_bytes);
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
        let delta = TenantDelta::new(&config, &self.module_dims, self.device, self.num_layers)?;
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

    /// Evict the least-recently-used tenant delta
    ///
    /// Before eviction:
    /// 1. Snapshots the delta to a file (if it has accumulated steps)
    /// 2. Invalidates dependent KV caches (if kv_registry is set)
    ///
    /// Returns the evicted tenant ID, delta, and optional snapshot path.
    pub fn evict_lru(
        &self,
    ) -> Option<(Subject, Arc<Mutex<TenantDelta>>, Option<PathBuf>)> {
        if self.deltas.is_empty() {
            return None;
        }

        // Find LRU tenant
        let mut oldest_tenant: Option<Subject> = None;
        let mut oldest_time = std::time::Instant::now();

        for entry in self.deltas.iter() {
            let delta = entry.value().lock();
            if delta.last_access < oldest_time {
                oldest_time = delta.last_access;
                oldest_tenant = Some(entry.key().clone());
            }
        }

        oldest_tenant.and_then(|tid| {
            self.deltas.remove(&tid).map(|(id, delta)| {
                let id_filename = id.to_filename();
                // Auto-snapshot to file before eviction
                let snapshot_path = {
                    let d = delta.lock();
                    if d.accumulated_steps > 0 {
                        let snapshot_file = self.snapshots_dir.join(format!("{}.safetensors", id_filename));
                        if let Some(ref fs) = self.fs {
                            // FsOps path: serialize in-memory and write through contained-root
                            let rel_path = format!("adapters/.snapshots/{}.safetensors", id_filename);
                            let result = futures::executor::block_on(async {
                                fs.mkdir("adapters/.snapshots", true).await
                                    .map_err(|e| anyhow::anyhow!("FsOps mkdir failed: {}", e))?;
                                let state_dict = d.extract_state_dict();
                                let bytes = serialize_state_dict_to_bytes(&state_dict)?;
                                fs.write_file(&rel_path, &bytes).await
                                    .map_err(|e| anyhow::anyhow!("FsOps write_file failed: {}", e))?;
                                Ok::<(), anyhow::Error>(())
                            });
                            match result {
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

                (id, delta, snapshot_path)
            })
        })
    }

    /// Evict deltas until total memory is under budget
    ///
    /// Returns a list of evicted (tenant_id, snapshot_path) pairs for audit/recovery.
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

        let tid_a = Subject::Local("tenant-a".into());
        let tid_b = Subject::Local("tenant-b".into());

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

        let tid_a = Subject::Local("tenant-a".into());
        let tid_b = Subject::Local("tenant-b".into());

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

        let tid_a = Subject::Local("tenant-a".into());
        let tid_b = Subject::Local("tenant-b".into());
        let tid_c = Subject::Local("tenant-c".into());

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

        let tid = Subject::Local("tenant".into());
        let _delta = pool.get_or_create(&tid).unwrap();

        let usage = pool.total_memory_usage();
        assert!(usage > 0, "Memory usage should be non-zero with a delta");
    }
}
