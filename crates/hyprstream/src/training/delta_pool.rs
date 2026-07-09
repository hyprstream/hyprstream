//! Per-tenant delta registry with LRU eviction
//!
//! Manages a pool of `TenantDelta` instances, one per tenant/session.
//! Pattern mirrors `KVCacheRegistry` using `DashMap` for concurrent access.
//!
//! # Snapshot durability & lifecycle (#869)
//!
//! Uncommitted per-tenant TTT adaptation is persisted to
//! `adapters/.snapshots/{subject}.safetensors` on two occasions:
//! - **LRU eviction** ([`DeltaPool::evict_lru`] / [`DeltaPool::evict_lru_async`]) — a
//!   warm-cache spill.
//! - **Drain-time export** ([`DeltaPool::export_all`]) — a graceful-shutdown "snapshot
//!   all, now" for pod preStop / scale-in.
//!
//! The read half is [`DeltaPool::get_or_hydrate`], which reloads a snapshot before
//! zero-initializing a fresh delta, restoring weights, effective ranks, Muon momentum,
//! and `accumulated_steps` / `request_count`.
//!
//! **Lifecycle decision — keep, don't delete on load.** A rehydrated snapshot is left
//! on disk after a successful load. Deleting it on load would open a window where a crash
//! between hydrate and the next eviction/export loses every step accumulated since — the
//! exact failure #869 exists to close. The file is instead overwritten by the next
//! eviction/export snapshot of the same subject, so at worst a reader sees slightly stale
//! (never absent) state. These staging snapshots stay in `.snapshots/` and are **never**
//! auto-promoted into a committed `adapters/NN_*.safetensors` adapter — that crossing is
//! the STEP `Promote` boundary and remains an explicit operation.
//!
//! **What a snapshot does NOT preserve** (rehydrated deltas start blank for these; see
//! [`TenantDelta::extract_state_dict`]): the in-memory STEP `Pending` rollback state
//! (`DeltaAdaptationState::Pending`) — a speculative, not-yet-resolved adaptation is
//! folded into the committed weights it was applied on top of, so it survives as
//! committed rather than as a reversible pending. No SSM/optimizer state beyond Muon
//! momentum is carried (there is none in `TenantDelta` today). `avg_loss_improvement`
//! and `last_snapshot_hash` reset.

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tch::{Device, Tensor};

use super::muon::MuonState;
use super::tenant_delta::{
    load_state_dict_from_bytes, serialize_state_dict_to_bytes, TenantDelta, TenantDeltaConfig,
};
use crate::runtime::kv_cache::KVCacheRegistry;
use crate::services::WorktreeClient;
use hyprstream_rpc::Subject;

/// Maximum snapshot size in bytes (512 MB). Deltas exceeding this are not snapshotted on eviction.
const MAX_SNAPSHOT_BYTES: usize = 512 * 1024 * 1024;

/// Default maximum number of tenants before LRU eviction kicks in.
const MAX_TENANTS_DEFAULT: usize = 100;

/// Snapshot-file metadata key: accumulated gradient steps (i64 scalar).
///
/// These `__meta.*` keys ride inside the safetensors snapshot alongside the LoRA
/// A/B / `__effective_rank` / `__muon_momentum` tensors written by
/// [`TenantDelta::extract_state_dict`]. They are injected only at the DeltaPool
/// snapshot boundary (see [`DeltaPool::snapshot_state_dict`]) so the generic
/// in-memory rollback state dict (used by the STEP stage/apply machinery) stays
/// free of pool-lifecycle counters.
const META_ACCUMULATED_STEPS: &str = "__meta.accumulated_steps";
/// Snapshot-file metadata key: served request count (i64 scalar).
const META_REQUEST_COUNT: &str = "__meta.request_count";

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
    /// Optional rank oracle config to attach to new deltas
    rank_oracle_config: Option<super::ttt::RankOracleConfig>,
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
            rank_oracle_config: None,
        }
    }

    /// Set rank oracle config for runtime rank adaptation on new deltas.
    pub fn with_rank_oracle(mut self, config: super::ttt::RankOracleConfig) -> Self {
        self.rank_oracle_config = Some(config);
        self
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

    /// Construct a fresh, zero-initialized delta with the default config.
    ///
    /// This is the pre-#869 status quo: B = zeros, no accumulated steps. Shared by
    /// [`Self::get_or_create`] and [`Self::get_or_hydrate`] (which then overlays a
    /// persisted snapshot on top, when one exists).
    fn create_blank_delta(&self) -> Result<TenantDelta> {
        let config = self.default_config.lock().clone();
        let mut delta = TenantDelta::new_with_per_layer_dims(
            &config,
            &self.module_dims,
            self.device,
            self.num_layers,
            self.per_layer_dims.as_ref(),
        )?;
        // Attach rank oracle if configured
        if let Some(ref oracle_config) = self.rank_oracle_config {
            delta.rank_oracle = Some(super::ttt::RankOracle::new(oracle_config.clone()));
        }
        Ok(delta)
    }

    /// Get or create a delta for the given tenant (synchronous, **no** snapshot rehydration).
    ///
    /// If the delta doesn't exist, creates a new zero-initialized one with the default
    /// config. This path does **not** reload `adapters/.snapshots/{subject}.safetensors` —
    /// it cannot, because snapshot reads require async FsOps. Async callers that want
    /// replica-loss durability (#869) must use [`Self::get_or_hydrate`] instead; this
    /// sync entry point remains for non-async callers (tests, batched-LoRA construction).
    pub fn get_or_create(&self, tenant_id: &Subject) -> Result<Arc<Mutex<TenantDelta>>> {
        // Fast path: existing delta
        if let Some(delta) = self.deltas.get(tenant_id) {
            delta.lock().touch();
            return Ok(delta.clone());
        }

        let delta = Arc::new(Mutex::new(self.create_blank_delta()?));

        // Insert (handles race condition)
        Ok(self
            .deltas
            .entry(tenant_id.clone())
            .or_insert(delta.clone())
            .clone())
    }

    /// Get an existing delta, or create one **rehydrating from a persisted snapshot**
    /// if `adapters/.snapshots/{subject}.safetensors` exists (async sibling of
    /// [`Self::get_or_create`], #869).
    ///
    /// This is the read half that makes eviction/drain snapshots durable across replica
    /// loss and scale-in. On a cold miss it:
    /// 1. builds a blank delta (as `get_or_create` would),
    /// 2. looks for a snapshot (via FsOps when a [`WorktreeClient`] is present, else the
    ///    local `snapshots_dir`),
    /// 3. on hit, restores the LoRA A/B weights, per-key effective ranks, Muon momentum,
    ///    and `accumulated_steps` / `request_count` onto the delta (device-mapped to the
    ///    pool's device),
    /// 4. on a corrupt/unloadable/shape-incompatible snapshot, **falls open** to a blank
    ///    delta (the pre-#869 status quo) and quarantines the bad file as `*.corrupt`.
    ///
    /// The snapshot file is **kept** after a successful load (not deleted) — see the
    /// module note on snapshot lifecycle. It is overwritten by the next eviction/export.
    pub async fn get_or_hydrate(&self, tenant_id: &Subject) -> Result<Arc<Mutex<TenantDelta>>> {
        // Fast path: already resident.
        if let Some(delta) = self.deltas.get(tenant_id) {
            delta.lock().touch();
            return Ok(delta.clone());
        }

        let mut delta = self.create_blank_delta()?;
        let id_filename = Self::sanitize_filename(&tenant_id.to_string());

        match self.read_snapshot_bytes(&id_filename).await {
            Ok(Some(bytes)) => {
                let loaded = load_state_dict_from_bytes(&bytes)
                    .and_then(|state| Self::apply_snapshot(&mut delta, &state));
                match loaded {
                    Ok(()) => {
                        tracing::info!(
                            "Rehydrated delta '{}' from snapshot ({} accumulated steps)",
                            tenant_id,
                            delta.accumulated_steps
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Corrupt/incompatible snapshot for delta '{}': {} — quarantining and starting blank",
                            tenant_id,
                            e
                        );
                        self.quarantine_snapshot(&id_filename).await;
                        // Rebuild pristine: apply_snapshot validates before mutating, but a
                        // rebuild guarantees no partial state leaks through on any error path.
                        delta = self.create_blank_delta()?;
                    }
                }
            }
            Ok(None) => {
                // No snapshot — normal cold start.
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to read snapshot for delta '{}': {} — starting blank",
                    tenant_id,
                    e
                );
            }
        }

        let delta = Arc::new(Mutex::new(delta));
        Ok(self
            .deltas
            .entry(tenant_id.clone())
            .or_insert(delta.clone())
            .clone())
    }

    /// Build the state dict written to a snapshot file: the delta's tensor state
    /// (A/B, effective ranks, Muon momentum) plus pool-lifecycle counters
    /// (`accumulated_steps`, `request_count`) as `__meta.*` scalar tensors.
    fn snapshot_state_dict(delta: &TenantDelta) -> HashMap<String, Tensor> {
        let mut state = delta.extract_state_dict();
        state.insert(
            META_ACCUMULATED_STEPS.to_owned(),
            Tensor::from_slice(&[delta.accumulated_steps as i64]),
        );
        state.insert(
            META_REQUEST_COUNT.to_owned(),
            Tensor::from_slice(&[delta.request_count as i64]),
        );
        state
    }

    /// Restore a loaded snapshot state dict onto a (blank) delta.
    ///
    /// Validates A/B shapes up front — `load_state_dict`'s `copy_` panics on a shape
    /// mismatch, so an incompatible snapshot (e.g. the pool's rank/module config changed
    /// since the snapshot was written) must be rejected as an error here rather than
    /// aborting the process. Also pre-seeds `muon_states` for any momentum buffers in the
    /// snapshot (a blank delta starts with none, so `load_state_dict` would otherwise drop
    /// them), device-mapped to the delta's device.
    fn apply_snapshot(delta: &mut TenantDelta, state: &HashMap<String, Tensor>) -> Result<()> {
        // Shape-compatibility gate (fail before any mutation).
        for key in delta.lora_a.keys() {
            if let Some(src) = state.get(&format!("{}.lora_a", key)) {
                let want = delta.lora_a[key].size();
                if src.size() != want {
                    return Err(anyhow!(
                        "snapshot lora_a '{}' shape {:?} != expected {:?}",
                        key,
                        src.size(),
                        want
                    ));
                }
            }
        }
        for key in delta.lora_b.keys() {
            if let Some(src) = state.get(&format!("{}.lora_b", key)) {
                let want = delta.lora_b[key].size();
                if src.size() != want {
                    return Err(anyhow!(
                        "snapshot lora_b '{}' shape {:?} != expected {:?}",
                        key,
                        src.size(),
                        want
                    ));
                }
            }
        }

        // Pre-seed Muon momentum slots (device-mapped) so load_state_dict restores them.
        let device = delta.device;
        for (name, t) in state.iter() {
            if let Some(key) = name.strip_suffix(".__muon_momentum") {
                delta
                    .muon_states
                    .entry(key.to_owned())
                    .or_insert_with(|| MuonState {
                        momentum_buffer: Some(t.to_device(device)),
                    });
            }
        }

        delta.load_state_dict(state)?;

        if let Some(t) = state.get(META_ACCUMULATED_STEPS) {
            delta.accumulated_steps = t.int64_value(&[]).max(0) as u64;
        }
        if let Some(t) = state.get(META_REQUEST_COUNT) {
            delta.request_count = t.int64_value(&[]).max(0) as u64;
        }
        Ok(())
    }

    /// Read a snapshot's raw bytes, or `Ok(None)` if it does not exist.
    ///
    /// Reads through the [`WorktreeClient`] (9P/FsOps) when present — the same channel
    /// the async eviction/export path writes through — otherwise falls back to the local
    /// `snapshots_dir` (used by the sync eviction path and by tests).
    async fn read_snapshot_bytes(&self, id_filename: &str) -> Result<Option<Vec<u8>>> {
        let rel_path = format!("adapters/.snapshots/{}.safetensors", id_filename);
        if let Some(ref fs) = self.fs {
            let st = fs
                .stat_path(&rel_path)
                .await
                .map_err(|e| anyhow!("FsOps stat failed: {}", e))?;
            if !st.exists {
                return Ok(None);
            }
            let bytes = fs
                .read_file_chunked(&rel_path)
                .await
                .map_err(|e| anyhow!("FsOps read failed: {}", e))?;
            Ok(Some(bytes))
        } else {
            let path = self
                .snapshots_dir
                .join(format!("{}.safetensors", id_filename));
            match std::fs::read(&path) {
                Ok(b) => Ok(Some(b)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(anyhow!("read snapshot {}: {}", path.display(), e)),
            }
        }
    }

    /// Best-effort quarantine of a bad snapshot: move `{id}.safetensors` aside to
    /// `{id}.safetensors.corrupt` so a later hydrate does not retry the same bad file
    /// and an operator can inspect it. Never fails the caller.
    async fn quarantine_snapshot(&self, id_filename: &str) {
        let rel_path = format!("adapters/.snapshots/{}.safetensors", id_filename);
        let corrupt_rel = format!("{}.corrupt", rel_path);
        if let Some(ref fs) = self.fs {
            // No 9P rename op — copy bytes aside, then remove the original.
            match fs.read_file_chunked(&rel_path).await {
                Ok(bytes) => {
                    if let Err(e) = fs.write_file_chunked(&corrupt_rel, &bytes).await {
                        tracing::warn!("Failed to write quarantined snapshot {}: {}", corrupt_rel, e);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read snapshot for quarantine {}: {}", rel_path, e);
                }
            }
            if let Err(e) = fs.remove_path(&rel_path).await {
                tracing::warn!("Failed to remove corrupt snapshot {}: {}", rel_path, e);
            }
        } else {
            let path = self
                .snapshots_dir
                .join(format!("{}.safetensors", id_filename));
            let corrupt = self
                .snapshots_dir
                .join(format!("{}.safetensors.corrupt", id_filename));
            if let Err(e) = std::fs::rename(&path, &corrupt) {
                tracing::warn!("Failed to quarantine corrupt snapshot {}: {}", path.display(), e);
            }
        }
        tracing::warn!("Quarantined corrupt snapshot for '{}' -> {}", id_filename, corrupt_rel);
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
                    let state_dict = Self::snapshot_state_dict(&d);
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
                let state_dict = Self::snapshot_state_dict(&d);
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

    /// Drain-time export: snapshot **every** resident delta with accumulated steps,
    /// without evicting any of them (#869, deliverable 2).
    ///
    /// This is the explicit "snapshot all, now" entry point for a graceful-shutdown /
    /// pod-preStop drain hook. It reuses the eviction snapshot format and path
    /// (`adapters/.snapshots/{subject}.safetensors`) and the same 512 MB cap and
    /// FsOps-required semantics, so a rehydrating replica reads these back through
    /// [`Self::get_or_hydrate`] identically to eviction snapshots.
    ///
    /// Deltas remain live in the pool after export (the process is about to exit, but
    /// the delta state must survive to disk, not be cleared). Returns `(subject,
    /// snapshot_path)` for each resident delta considered — `None` in the path slot
    /// marks a **loss event**: a delta that was skipped (no accumulated steps, over the
    /// size cap, or a write failure). Callers/operators should treat `None` with
    /// `accumulated_steps > 0` as data loss.
    ///
    /// Requires a [`WorktreeClient`]; without one this is a logged no-op (FsOps-absent
    /// skip, matching the eviction path — there is no worktree to write through).
    pub async fn export_all(&self) -> Result<Vec<(Subject, Option<PathBuf>)>> {
        let Some(ref fs) = self.fs else {
            tracing::warn!(
                "export_all: FsOps not available — {} resident delta(s) NOT persisted (loss on shutdown)",
                self.deltas.len()
            );
            return Ok(Vec::new());
        };

        if self.deltas.is_empty() {
            return Ok(Vec::new());
        }

        fs.mkdir_p("adapters/.snapshots")
            .await
            .map_err(|e| anyhow!("FsOps mkdir failed: {}", e))?;

        // Collect ids first so we don't hold DashMap refs across await points.
        let ids: Vec<Subject> = self.deltas.iter().map(|e| e.key().clone()).collect();
        let mut results = Vec::with_capacity(ids.len());

        for id in ids {
            let Some(entry) = self.deltas.get(&id) else {
                continue; // Evicted concurrently.
            };
            let delta = entry.value().clone();
            drop(entry);

            // Extract snapshot data under lock, release before async I/O.
            let snapshot_data = {
                let d = delta.lock();
                if d.accumulated_steps == 0 {
                    None // Nothing to persist.
                } else {
                    let memory_bytes = d.memory_bytes();
                    if memory_bytes > MAX_SNAPSHOT_BYTES {
                        tracing::warn!(
                            "export_all: delta '{}' exceeds max snapshot size ({} > {}), skipping (loss)",
                            id, memory_bytes, MAX_SNAPSHOT_BYTES
                        );
                        None
                    } else {
                        Some(Self::snapshot_state_dict(&d))
                    }
                }
            };

            let path = if let Some(state_dict) = snapshot_data {
                let id_filename = Self::sanitize_filename(&id.to_string());
                let rel_path = format!("adapters/.snapshots/{}.safetensors", id_filename);
                match async {
                    let bytes = serialize_state_dict_to_bytes(&state_dict)?;
                    fs.write_file_chunked(&rel_path, &bytes)
                        .await
                        .map_err(|e| anyhow!("FsOps write_file failed: {}", e))?;
                    Ok::<(), anyhow::Error>(())
                }
                .await
                {
                    Ok(()) => {
                        tracing::info!("export_all: snapshotted delta '{}' -> {}", id, rel_path);
                        Some(
                            self.snapshots_dir
                                .join(format!("{}.safetensors", id_filename)),
                        )
                    }
                    Err(e) => {
                        tracing::warn!("export_all: failed to snapshot delta '{}': {} (loss)", id, e);
                        None
                    }
                }
            } else {
                None
            };

            results.push((id, path));
        }

        Ok(results)
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

    // ===== #869 rehydration + drain-time export =====

    /// Per-test unique snapshots dir under a fresh temp path (isolate file I/O).
    fn unique_snapshots_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("delta_pool_869_{}_{}", tag, nanos))
    }

    fn pool_with_dir(dir: PathBuf) -> DeltaPool {
        DeltaPool::new(
            TenantDeltaConfig::default(),
            test_module_dims(),
            Device::Cpu,
            None,
            dir,
            None, // fs = None -> sync std::fs snapshot path (used by evict_lru + get_or_hydrate)
            2,
        )
    }

    /// snapshot -> evict -> get_or_hydrate rehydrates weights + accumulated_steps.
    #[tokio::test]
    async fn test_hydrate_roundtrip_restores_state() {
        let dir = unique_snapshots_dir("roundtrip");
        let pool = pool_with_dir(dir.clone());
        let tid = Subject::new("tenant-hydrate");

        // Train: mutate B and set accumulated_steps so eviction snapshots it.
        let expected_b: Tensor;
        {
            let delta_arc = pool.get_or_create(&tid).unwrap();
            let mut d = delta_arc.lock();
            let _guard = tch::no_grad_guard(); // B is a grad-requiring leaf var
            let b = d.lora_b.get_mut("0.q_proj").unwrap();
            let _ = b.uniform_(-0.5, 0.5);
            expected_b = b.copy();
            d.accumulated_steps = 17;
            d.request_count = 5;
        }

        // Evict (sync path writes snapshot to `dir`), then hydrate a fresh miss.
        let evicted = pool.evict_lru();
        assert!(evicted.is_some());
        assert!(evicted.unwrap().2.is_some(), "eviction should write a snapshot");
        assert_eq!(pool.tenant_count(), 0);

        let delta_arc = pool.get_or_hydrate(&tid).await.unwrap();
        let d = delta_arc.lock();
        assert_eq!(d.accumulated_steps, 17, "accumulated_steps rehydrated");
        assert_eq!(d.request_count, 5, "request_count rehydrated");
        let max_diff: f64 = (&d.lora_b["0.q_proj"] - &expected_b)
            .abs()
            .max()
            .double_value(&[]);
        assert!(max_diff < 1e-6, "B weights rehydrated (max diff {})", max_diff);

        drop(d);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A corrupt snapshot file is quarantined (.corrupt) and hydrate falls open to blank.
    #[tokio::test]
    async fn test_corrupt_snapshot_quarantined_and_blank() {
        let dir = unique_snapshots_dir("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let pool = pool_with_dir(dir.clone());
        let tid = Subject::new("tenant-corrupt");

        // Write garbage where the snapshot would live.
        let fname = DeltaPool::sanitize_filename(&tid.to_string());
        let snap_path = dir.join(format!("{}.safetensors", fname));
        std::fs::write(&snap_path, b"not a safetensors file").unwrap();

        let delta_arc = pool.get_or_hydrate(&tid).await.unwrap();
        {
            let d = delta_arc.lock();
            // Blank: B is zeros, no accumulated steps.
            assert_eq!(d.accumulated_steps, 0);
            let b_norm: f64 = d.lora_b["0.q_proj"].norm().double_value(&[]);
            assert!(b_norm < 1e-8, "fell open to a blank delta");
        }
        assert!(!snap_path.exists(), "corrupt file moved aside");
        assert!(
            dir.join(format!("{}.safetensors.corrupt", fname)).exists(),
            "corrupt file quarantined"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An incompatible (wrong-shape) snapshot is rejected without panicking.
    #[tokio::test]
    async fn test_incompatible_shape_snapshot_falls_open() {
        let dir = unique_snapshots_dir("incompat");
        std::fs::create_dir_all(&dir).unwrap();
        let pool = pool_with_dir(dir.clone());
        let tid = Subject::new("tenant-incompat");

        // Build a valid safetensors snapshot whose lora_a has the wrong shape.
        let mut state: HashMap<String, Tensor> = HashMap::new();
        state.insert(
            "0.q_proj.lora_a".to_owned(),
            Tensor::zeros([4, 8], (tch::Kind::Float, Device::Cpu)), // expected [8, 512]
        );
        let bytes = serialize_state_dict_to_bytes(&state).unwrap();
        let fname = DeltaPool::sanitize_filename(&tid.to_string());
        std::fs::write(dir.join(format!("{}.safetensors", fname)), &bytes).unwrap();

        // Must not panic; falls open to blank and quarantines.
        let delta_arc = pool.get_or_hydrate(&tid).await.unwrap();
        {
            let d = delta_arc.lock();
            assert_eq!(d.lora_a["0.q_proj"].size(), vec![8, 512]);
        }
        assert!(
            dir.join(format!("{}.safetensors.corrupt", fname)).exists(),
            "incompatible snapshot quarantined"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Hydrating a subject with no snapshot yields a normal blank delta.
    #[tokio::test]
    async fn test_hydrate_no_snapshot_is_blank() {
        let dir = unique_snapshots_dir("nosnap");
        let pool = pool_with_dir(dir.clone());
        let tid = Subject::new("tenant-fresh");

        let delta_arc = pool.get_or_hydrate(&tid).await.unwrap();
        let d = delta_arc.lock();
        assert_eq!(d.accumulated_steps, 0);
        assert_eq!(pool.tenant_count(), 1);
        drop(d);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// export_all with no FsOps is a logged no-op (no worktree to write through).
    #[tokio::test]
    async fn test_export_all_no_fs_is_noop() {
        let dir = unique_snapshots_dir("exportnofs");
        let pool = pool_with_dir(dir.clone());
        let tid = Subject::new("tenant-x");
        {
            let d = pool.get_or_create(&tid).unwrap();
            d.lock().accumulated_steps = 3;
        }
        // fs = None -> export_all returns empty (documented FsOps-absent skip).
        let exported = pool.export_all().await.unwrap();
        assert!(exported.is_empty());
        // Delta stays resident (export never evicts).
        assert_eq!(pool.tenant_count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
