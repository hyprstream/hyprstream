//! Key-Value cache implementation for efficient autoregressive generation
//!
//! This module implements KV caching to avoid recomputing past key and value
//! states during inference, providing 10-50x speedup for long sequences.
//!
//! ## Memory Optimization
//!
//! The cache uses a chunked growth strategy instead of pre-allocating the full
//! max_seq_len. This saves ~4GB+ VRAM for models with large context windows.
//!
//! ## Quantization Support
//!
//! Supports optional quantization via bitsandbytes:
//! - `None`: Full precision (FP16/BF16) - default
//! - `Int8`: 8-bit quantization
//! - `Nf4`: 4-bit NormalFloat (best quality for 4-bit)
//! - `Fp4`: 4-bit FloatingPoint
//!
//! Quantization uses a hybrid approach:
//! - Historical tokens are stored quantized (memory savings)
//! - Recent tokens stay in a full-precision buffer (fast updates)
//! - Dequantized view is cached across layers (avoids repeated dequantization)

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(any(feature = "bnb", test))]
use tch::Device;
use tch::{Kind as DType, Tensor};

use super::KVQuantType;
use super::torch_utils::{estimate_tensor_size_mb, safe_zeros};

// ============================================================================
// Cache Owner and Registry Types (for multi-session support)
// ============================================================================

/// Identifies the owner of a KV cache instance for session isolation.
///
/// Each owner gets their own isolated cache, enabling concurrent inference
/// without interference between different conversations/sessions.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub enum CacheOwner {
    /// Session-based inference - multiple requests share cache for context continuity.
    /// Use this for conversational AI where context should persist across requests.
    Session(String),

    /// Stateless inference - one-off completions where cache is discarded after use.
    /// Use this for standard OpenAI-compatible API requests without session tracking.
    Stateless(u64),

    /// Training validation inference - for evaluating model during training.
    Training { adapter: String, run_id: u64 },
}

impl CacheOwner {
    /// Create a new stateless owner with a random ID
    pub fn new_stateless() -> Self {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        CacheOwner::Stateless(id)
    }

    /// Create a session owner
    pub fn session(session_id: impl Into<String>) -> Self {
        CacheOwner::Session(session_id.into())
    }

    /// Create a training owner
    pub fn training(adapter: impl Into<String>, run_id: u64) -> Self {
        CacheOwner::Training {
            adapter: adapter.into(),
            run_id,
        }
    }
}

/// Configuration for KV cache instances
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Maximum sequence length (context window)
    pub max_seq_len: usize,
    /// Quantization type for memory efficiency
    pub quant_type: KVQuantType,
    /// Whether this cache is exempt from eviction (e.g., for training)
    pub eviction_exempt: bool,
    /// Use paged block storage (PagedAttention-style) instead of contiguous tensors.
    /// Requires a BlockPool to be initialized on the registry.
    pub paged: bool,
}

impl CacheConfig {
    /// Create a new cache configuration
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        Self {
            num_layers,
            max_seq_len,
            quant_type: KVQuantType::None,
            eviction_exempt: false,
            paged: false,
        }
    }

    /// Set quantization type
    pub fn with_quant_type(mut self, quant_type: KVQuantType) -> Self {
        self.quant_type = quant_type;
        self
    }

    /// Mark as eviction exempt
    pub fn with_eviction_exempt(mut self, exempt: bool) -> Self {
        self.eviction_exempt = exempt;
        self
    }

    /// Enable paged block storage (requires BlockPool on the registry)
    pub fn with_paged(mut self, paged: bool) -> Self {
        self.paged = paged;
        self
    }
}

/// Registry managing multiple KV cache instances for concurrent sessions.
///
/// Each `CacheOwner` gets their own isolated `KVCacheManager` wrapped in a Mutex.
/// The registry uses DashMap for lock-free access to different sessions, while
/// each individual cache uses Mutex for thread-safe layer access (required because
/// tch-rs Tensor contains raw pointers that aren't `Sync`).
///
/// # Concurrency Model
///
/// ```text
/// Request A ──► Registry.get_or_create(Session("abc")) ──► Cache A (Mutex)
/// Request B ──► Registry.get_or_create(Session("xyz")) ──► Cache B (Mutex)
///                                                          ↑
///                                                No contention between A and B!
/// ```
pub struct KVCacheRegistry {
    /// Active caches indexed by owner (lock-free access to different owners)
    caches: DashMap<CacheOwner, Arc<Mutex<KVCacheManager>>>,
    /// Default configuration for new caches
    default_config: CacheConfig,
    /// Memory budget in bytes (None = unlimited)
    memory_budget_bytes: Option<usize>,
    /// Maps cache owners to the subject delta that was active when the cache was computed.
    /// When a delta is evicted or reset, dependent caches must be invalidated.
    delta_dependencies: DashMap<CacheOwner, Option<String>>,
    /// Shared block pool for paged KV cache storage (None if paged mode not enabled)
    block_pool: Option<Arc<Mutex<BlockPool>>>,
}

impl KVCacheRegistry {
    /// Create a new KV cache registry
    pub fn new(default_config: CacheConfig, memory_budget: Option<usize>) -> Self {
        tracing::info!(
            "[KVCacheRegistry::new] Creating registry with {} layers, max_seq_len={}, quant={:?}, budget={:?}, paged={}",
            default_config.num_layers,
            default_config.max_seq_len,
            default_config.quant_type,
            memory_budget,
            default_config.paged
        );

        Self {
            caches: DashMap::new(),
            default_config,
            memory_budget_bytes: memory_budget,
            delta_dependencies: DashMap::new(),
            block_pool: None,
        }
    }

    /// Initialize the shared block pool for paged KV cache storage.
    ///
    /// Must be called before any paged caches are created. `num_blocks` determines
    /// the total KV cache capacity across all sessions. `num_kv_heads` and `head_dim`
    /// must match the model's attention configuration.
    pub fn init_block_pool(
        &mut self,
        num_blocks: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: tch::Device,
        dtype: DType,
    ) -> Result<()> {
        let pool = BlockPool::new(num_blocks, num_kv_heads, head_dim, device, dtype)?;
        self.block_pool = Some(Arc::new(Mutex::new(pool)));
        Ok(())
    }

    /// Get the shared block pool (for metrics/debugging).
    pub fn block_pool(&self) -> Option<&Arc<Mutex<BlockPool>>> {
        self.block_pool.as_ref()
    }

    /// Get or create a cache for the given owner.
    ///
    /// If the cache doesn't exist, creates a new one with the default config.
    /// Returns `Arc<Mutex<KVCacheManager>>` - caller must lock() to use.
    pub fn get_or_create(&self, owner: CacheOwner) -> Arc<Mutex<KVCacheManager>> {
        // Try to get existing cache first (fast path)
        if let Some(cache) = self.caches.get(&owner) {
            let mut guard = cache.lock();
            guard.touch();

            // Transparently restore CPU-offloaded caches back to GPU
            if guard.location() == CacheLocation::Cpu {
                // Determine the GPU device from the engine config
                let device = tch::Device::cuda_if_available();
                guard.restore_to_gpu(device);
            }

            drop(guard);
            return cache.clone();
        }

        // Slow path: create new cache (paged or contiguous based on config)
        let cache = if self.default_config.paged {
            if let Some(pool) = &self.block_pool {
                Arc::new(Mutex::new(KVCacheManager::new_paged(
                    self.default_config.num_layers,
                    self.default_config.max_seq_len,
                    pool.clone(),
                )))
            } else {
                tracing::warn!("Paged mode requested but block pool not initialized, falling back to contiguous");
                Arc::new(Mutex::new(KVCacheManager::new(
                    self.default_config.num_layers,
                    self.default_config.max_seq_len,
                    self.default_config.quant_type,
                )))
            }
        } else {
            Arc::new(Mutex::new(KVCacheManager::new(
                self.default_config.num_layers,
                self.default_config.max_seq_len,
                self.default_config.quant_type,
            )))
        };

        // Insert and return (handles race condition - returns existing if another thread inserted)
        self.caches
            .entry(owner)
            .or_insert(cache.clone())
            .clone()
    }

    /// Release a cache when the session is done.
    ///
    /// For `Stateless` owners, this removes the cache immediately.
    /// For `Session` owners, the cache is kept for potential reuse.
    pub fn release(&self, owner: &CacheOwner) {
        match owner {
            CacheOwner::Stateless(_) => {
                // Stateless caches are removed immediately
                self.caches.remove(owner);
                tracing::debug!("Released stateless cache: {:?}", owner);
            }
            CacheOwner::Session(_) | CacheOwner::Training { .. } => {
                // Session and training caches are kept for reuse
                // They will be evicted by LRU if memory pressure occurs
                tracing::debug!("Marked session cache for potential reuse: {:?}", owner);
            }
        }
    }

    /// Evict least-recently-used caches to stay within memory budget.
    pub fn evict_to_budget(&self) {
        let budget = match self.memory_budget_bytes {
            Some(b) => b,
            None => return, // No budget = no eviction
        };

        let current_usage = self.total_memory_usage();
        if current_usage <= budget {
            return;
        }

        // Collect eviction candidates (skip eviction-exempt and training caches)
        let mut candidates: Vec<(CacheOwner, u64, usize)> = self
            .caches
            .iter()
            .filter_map(|entry| {
                let owner = entry.key().clone();

                // Skip training caches
                if matches!(owner, CacheOwner::Training { .. }) {
                    return None;
                }

                let guard = entry.value().lock();
                let last_access = guard.last_access();
                let memory = guard.memory_usage();
                Some((owner, last_access, memory))
            })
            .collect();

        // Sort by last access (oldest first)
        candidates.sort_by_key(|(_, last_access, _)| *last_access);

        // Two-tier eviction: first offload to CPU, then remove if still over budget.
        let mut freed = 0;

        // Tier 1: Offload GPU caches to CPU (frees GPU memory, preserves data)
        for (owner, _, size) in &candidates {
            if current_usage - freed <= budget {
                break;
            }
            if let Some(cache) = self.caches.get(owner) {
                let mut guard = cache.lock();
                if guard.location() == CacheLocation::Gpu {
                    guard.offload_to_cpu();
                    freed += size;
                    tracing::info!("Offloaded cache {:?} to CPU, freed {} GPU bytes", owner, size);
                }
            }
        }

        // Tier 2: If still over budget, remove offloaded caches entirely
        if current_usage - freed > budget {
            for (owner, _, size) in candidates {
                if current_usage - freed <= budget {
                    break;
                }
                if let Some(cache) = self.caches.get(&owner) {
                    let guard = cache.lock();
                    if guard.location() == CacheLocation::Cpu {
                        drop(guard);
                        self.caches.remove(&owner);
                        freed += size;
                        tracing::info!("Evicted CPU cache {:?}, freed {} bytes", owner, size);
                    }
                }
            }
        }
    }

    /// Get total memory usage across all caches in bytes
    pub fn total_memory_usage(&self) -> usize {
        self.caches
            .iter()
            .map(|entry| entry.value().lock().memory_usage())
            .sum()
    }

    /// Get number of active caches
    pub fn cache_count(&self) -> usize {
        self.caches.len()
    }

    /// Clear all caches (useful for cleanup)
    pub fn clear_all(&self) {
        self.caches.clear();
        self.delta_dependencies.clear();
        tracing::info!("Cleared all KV caches from registry");
    }

    /// Register a dependency between a cache owner and a subject's delta.
    ///
    /// When the subject's delta is later evicted or reset, all dependent caches
    /// will be invalidated since they were computed with stale weights.
    pub fn register_delta_dependency(&self, owner: &CacheOwner, tenant_id: Option<String>) {
        self.delta_dependencies
            .insert(owner.clone(), tenant_id);
    }

    /// Invalidate all KV caches that depend on a specific subject's delta.
    ///
    /// Called when a subject's delta is evicted, reset, or significantly modified.
    /// Returns the number of caches invalidated.
    pub fn invalidate_for_tenant(&self, tenant_id: &str) -> usize {
        let mut invalidated = 0;
        let to_remove: Vec<CacheOwner> = self
            .delta_dependencies
            .iter()
            .filter_map(|entry| {
                if entry.value().as_deref() == Some(tenant_id) {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect();

        for owner in to_remove {
            // Clear cached token IDs before removing — ensures prefix matching
            // won't reuse stale KV values if the cache is somehow retained.
            if let Some(cache) = self.caches.get(&owner) {
                let mut guard = cache.lock();
                guard.set_cached_tokens(Vec::new());
                guard.clear_all();
            }
            self.caches.remove(&owner);
            self.delta_dependencies.remove(&owner);
            invalidated += 1;
        }

        if invalidated > 0 {
            tracing::info!(
                "Invalidated {} KV caches dependent on subject delta '{}'",
                invalidated,
                tenant_id
            );
        }

        invalidated
    }
}

// ============================================================================
// Paged Block Pool (PagedAttention-style memory management)
// ============================================================================

/// Number of tokens per block in the paged KV cache.
/// 256 balances fragmentation overhead vs block reuse granularity.
pub const BLOCK_SIZE: usize = 256;

/// Opaque handle to a block in the pool.
pub type BlockId = usize;

/// Shared pool of fixed-size GPU tensor blocks for KV cache storage.
///
/// All KV cache memory is pre-allocated in this pool at startup. Sessions
/// allocate and free blocks from the pool — no per-session `cudaMalloc` calls,
/// zero fragmentation, and instant block reuse across sessions.
///
/// Each block stores `BLOCK_SIZE` tokens of K or V for a single attention layer.
/// Shape: `[1, BLOCK_SIZE, num_kv_heads, head_dim]`.
pub struct BlockPool {
    /// Pre-allocated tensor blocks on GPU (one per block ID).
    /// Each tensor has shape [1, BLOCK_SIZE, num_kv_heads, head_dim].
    blocks: Vec<Tensor>,
    /// Free block IDs available for allocation (LIFO stack for cache locality).
    free_list: Vec<BlockId>,
    /// Block tensor shape: [1, BLOCK_SIZE, num_kv_heads, head_dim]
    block_shape: [i64; 4],
    /// Device the blocks are allocated on
    device: tch::Device,
    /// Data type of the blocks
    dtype: DType,
    /// Total number of blocks in the pool
    total_blocks: usize,
}

impl BlockPool {
    /// Create a new block pool with `num_blocks` pre-allocated blocks.
    ///
    /// All blocks are allocated on `device` with dtype `dtype`.
    /// `num_kv_heads` and `head_dim` define the per-head tensor dimensions.
    pub fn new(
        num_blocks: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: tch::Device,
        dtype: DType,
    ) -> Result<Self> {
        let block_shape = [1, BLOCK_SIZE as i64, num_kv_heads as i64, head_dim as i64];

        tracing::info!(
            "Allocating block pool: {} blocks x {:?} ({:.1} MB total)",
            num_blocks,
            block_shape,
            (num_blocks as f64 * BLOCK_SIZE as f64 * num_kv_heads as f64 * head_dim as f64
                * match dtype { DType::Half | DType::BFloat16 => 2.0, _ => 4.0 })
                / (1024.0 * 1024.0),
        );

        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_list = Vec::with_capacity(num_blocks);

        for id in 0..num_blocks {
            let block = safe_zeros(&block_shape, (dtype, device))?;
            blocks.push(block);
            free_list.push(id);
        }

        Ok(Self {
            blocks,
            free_list,
            block_shape,
            device,
            dtype,
            total_blocks: num_blocks,
        })
    }

    /// Allocate a block from the pool. Returns None if the pool is exhausted.
    pub fn allocate(&mut self) -> Option<BlockId> {
        self.free_list.pop()
    }

    /// Return a block to the pool.
    pub fn free(&mut self, id: BlockId) {
        debug_assert!(id < self.total_blocks, "Invalid block ID: {}", id);
        self.free_list.push(id);
    }

    /// Get a reference to a block's tensor.
    pub fn get_block(&self, id: BlockId) -> &Tensor {
        &self.blocks[id]
    }

    /// Number of blocks currently in use.
    pub fn used_blocks(&self) -> usize {
        self.total_blocks - self.free_list.len()
    }

    /// Number of blocks available for allocation.
    pub fn free_block_count(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of blocks in the pool.
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Memory used by allocated blocks (approximate, in bytes).
    pub fn used_memory(&self) -> usize {
        let elem_size = match self.dtype {
            DType::Half | DType::BFloat16 => 2,
            _ => 4,
        };
        let per_block = BLOCK_SIZE * self.block_shape[2] as usize * self.block_shape[3] as usize * elem_size;
        self.used_blocks() * per_block
    }

    /// Block shape: [1, BLOCK_SIZE, num_kv_heads, head_dim]
    pub fn block_shape(&self) -> &[i64; 4] {
        &self.block_shape
    }

    /// Device the pool is allocated on.
    pub fn device(&self) -> tch::Device {
        self.device
    }

    /// Data type of the pool's tensors.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl std::fmt::Debug for BlockPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockPool")
            .field("total_blocks", &self.total_blocks)
            .field("used_blocks", &self.used_blocks())
            .field("free_blocks", &self.free_block_count())
            .field("block_shape", &self.block_shape)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .finish()
    }
}

// ============================================================================
// Original KV Cache Implementation
// ============================================================================

/// Default chunk size for KV cache growth (in tokens)
const DEFAULT_CHUNK_SIZE: usize = 1024;

/// Default blocksize for bitsandbytes quantization
#[cfg(feature = "bnb")]
const QUANT_BLOCKSIZE: usize = 64;

/// Number of tokens to accumulate before flushing to quantized storage
#[cfg(feature = "bnb")]
const BUFFER_FLUSH_THRESHOLD: usize = 64;

/// Get byte size per element for a dtype
fn dtype_element_size(dtype: DType) -> usize {
    match dtype {
        DType::Half | DType::BFloat16 => 2,
        _ => 4, // Float, Double, Int, etc.
    }
}

/// Quantized tensor storage - stores a contiguous quantized tensor
#[cfg(feature = "bnb")]
#[derive(Debug)]
struct QuantizedTensor {
    /// Quantized data bytes
    data: Vec<u8>,
    /// Quantization state (codebook, absmax, etc.)
    state: bitsandbytes_sys::QuantState,
    /// Original tensor shape for reconstruction
    shape: Vec<i64>,
    /// Original dtype for reconstruction
    dtype: DType,
    /// Target device for reconstruction
    device: Device,
}

#[cfg(feature = "bnb")]
impl QuantizedTensor {
    /// Quantize a tensor using the specified quantization type.
    ///
    /// If the tensor is on a GPU (CUDA/ROCm) and is BF16 or FP16, uses the
    /// GPU-native bitsandbytes kernels directly — no CPU round-trip. For CPU
    /// tensors or f32 dtype, falls back to the CPU quantization path.
    fn from_tensor(tensor: &Tensor, quant_type: KVQuantType) -> Result<Self> {
        let shape = tensor.size();
        let dtype = tensor.kind();
        let device = tensor.device();

        if quant_type == KVQuantType::None {
            return Err(anyhow!("Cannot create QuantizedTensor with KVQuantType::None"));
        }

        // GPU-native path: BF16/FP16 on CUDA/ROCm → quantize directly on GPU
        let use_gpu_path = device != Device::Cpu
            && (dtype == DType::BFloat16 || dtype == DType::Half)
            && quant_type == KVQuantType::Int8;

        if use_gpu_path {
            return Self::from_tensor_gpu(tensor, &shape, dtype, device);
        }

        // CPU fallback path: copy to CPU, convert to f32, quantize
        let cpu_tensor = tensor
            .to_device(Device::Cpu)
            .to_kind(DType::Float)
            .contiguous();
        let numel = cpu_tensor.numel();

        let mut f32_data = vec![0.0f32; numel];
        cpu_tensor
            .f_copy_data(&mut f32_data, numel)
            .map_err(|e| anyhow!("Failed to copy tensor data: {:?}", e))?;

        let (data, state) = match quant_type {
            KVQuantType::Int8 => bitsandbytes_sys::quantize_blockwise_fp32(&f32_data, QUANT_BLOCKSIZE)
                .map_err(|e| anyhow!("Int8 quantization failed: {:?}", e))?,
            KVQuantType::Nf4 => bitsandbytes_sys::quantize_4bit_nf4_fp32(&f32_data, QUANT_BLOCKSIZE)
                .map_err(|e| anyhow!("NF4 quantization failed: {:?}", e))?,
            KVQuantType::Fp4 => bitsandbytes_sys::quantize_4bit_fp4_fp32(&f32_data, QUANT_BLOCKSIZE)
                .map_err(|e| anyhow!("FP4 quantization failed: {:?}", e))?,
            KVQuantType::None => unreachable!(),
        };

        tracing::trace!(
            "Quantized tensor (CPU path): {} elements, shape {:?} -> {} bytes ({:.1}% of f32)",
            numel, shape, data.len(),
            (data.len() as f64 / (numel * 4) as f64) * 100.0
        );

        Ok(Self { data, state, shape, dtype, device })
    }

    /// GPU-native quantization: operates directly on GPU BF16/FP16 tensors.
    ///
    /// Uses bitsandbytes CUDA/HIP kernels. The code/absmax arrays are created
    /// as CPU Vecs (they're small metadata), while the quantization kernel reads
    /// from GPU input and writes to a GPU-resident output tensor. We then copy
    /// the small quantized result back to CPU for storage (the quantized data
    /// is ~8x smaller than the original, so this copy is cheap).
    fn from_tensor_gpu(
        tensor: &Tensor,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let numel = tensor.numel();
        let contiguous = tensor.contiguous();

        // Create quantization state (CPU — small metadata)
        let mut state = bitsandbytes_sys::QuantState::new_8bit(numel, QUANT_BLOCKSIZE);

        // Allocate GPU output tensor for quantized data
        let output_tensor = Tensor::zeros(
            [numel as i64],
            (DType::Uint8, device),
        );

        // Allocate GPU tensors for code and absmax
        let code_tensor = Tensor::from_slice(&state.code).to_device(device);
        let absmax_tensor = Tensor::zeros(
            [state.absmax.len() as i64],
            (DType::Float, device),
        );

        // Call GPU-native kernel
        unsafe {
            let input_ptr = contiguous.data_ptr() as *mut std::ffi::c_void;
            let code_ptr = code_tensor.data_ptr() as *mut f32;
            let absmax_ptr = absmax_tensor.data_ptr() as *mut f32;
            let output_ptr = output_tensor.data_ptr() as *mut u8;

            match dtype {
                DType::BFloat16 => bitsandbytes_sys::quantize_blockwise_bf16_gpu(
                    code_ptr, input_ptr, absmax_ptr, output_ptr,
                    QUANT_BLOCKSIZE as i32, numel as i32,
                ),
                DType::Half => bitsandbytes_sys::quantize_blockwise_fp16_gpu(
                    code_ptr, input_ptr, absmax_ptr, output_ptr,
                    QUANT_BLOCKSIZE as i32, numel as i32,
                ),
                _ => unreachable!("GPU path only for BF16/FP16"),
            }
        }

        // Copy small results back to CPU for storage
        // quantized data: numel bytes (8x smaller than BF16 input)
        // absmax: n_blocks * 4 bytes (tiny)
        let mut data = vec![0u8; numel];
        output_tensor
            .to_device(Device::Cpu)
            .f_copy_data(&mut data, numel)
            .map_err(|e| anyhow!("Failed to copy quantized data: {:?}", e))?;

        let mut absmax = vec![0.0f32; state.absmax.len()];
        absmax_tensor
            .to_device(Device::Cpu)
            .f_copy_data(&mut absmax, state.absmax.len())
            .map_err(|e| anyhow!("Failed to copy absmax: {:?}", e))?;
        state.absmax = absmax;

        tracing::trace!(
            "Quantized tensor (GPU path, {:?}): {} elements -> {} bytes ({:.1}% of original)",
            dtype, numel, data.len(),
            (data.len() as f64 / (numel as f64 * dtype_element_size(dtype) as f64)) * 100.0
        );

        Ok(Self { data, state, shape: shape.to_vec(), dtype, device })
    }

    /// Dequantize back to a tensor.
    ///
    /// If the target device is GPU and dtype is BF16/FP16, uses GPU-native
    /// dequantization to avoid the CPU→f32→GPU round-trip.
    fn to_tensor(&self) -> Result<Tensor> {
        // GPU-native dequant path
        let use_gpu_path = self.device != Device::Cpu
            && (self.dtype == DType::BFloat16 || self.dtype == DType::Half)
            && !self.state.is_4bit;

        if use_gpu_path {
            return self.to_tensor_gpu();
        }

        // CPU fallback path
        let f32_data = if self.state.is_4bit {
            match self.state.quant_type {
                Some(bitsandbytes_sys::QuantType::Nf4) => {
                    bitsandbytes_sys::dequantize_4bit_nf4_fp32(&self.data, &self.state)
                        .map_err(|e| anyhow!("NF4 dequantization failed: {:?}", e))?
                }
                Some(bitsandbytes_sys::QuantType::Fp4) => {
                    bitsandbytes_sys::dequantize_4bit_fp4_fp32(&self.data, &self.state)
                        .map_err(|e| anyhow!("FP4 dequantization failed: {:?}", e))?
                }
                _ => {
                    return Err(anyhow!("Unknown 4-bit quantization type"));
                }
            }
        } else {
            bitsandbytes_sys::dequantize_blockwise_fp32(&self.data, &self.state)
                .map_err(|e| anyhow!("Int8 dequantization failed: {:?}", e))?
        };

        // Create tensor from f32 data
        let cpu_tensor = Tensor::from_slice(&f32_data).reshape(&self.shape);

        // Convert to original dtype and move to original device
        let tensor = cpu_tensor.to_kind(self.dtype).to_device(self.device);

        Ok(tensor)
    }

    /// GPU-native dequantization: produces BF16/FP16 tensor directly on GPU.
    ///
    /// Copies the small quantized data + absmax to GPU, then calls the
    /// bitsandbytes BF16/FP16 dequant kernel. Output is GPU-resident in the
    /// original dtype — no f32 intermediate, no CPU→GPU dtype conversion.
    fn to_tensor_gpu(&self) -> Result<Tensor> {
        let numel = self.state.n_elements;

        // Copy quantized data and absmax to GPU (small: ~numel bytes + n_blocks*4 bytes)
        let quant_tensor = Tensor::from_slice(&self.data).to_device(self.device);
        let code_tensor = Tensor::from_slice(&self.state.code).to_device(self.device);
        let absmax_tensor = Tensor::from_slice(&self.state.absmax).to_device(self.device);

        // Allocate output tensor on GPU in original dtype
        let output_tensor = Tensor::zeros(
            [numel as i64],
            (self.dtype, self.device),
        );

        unsafe {
            let quant_ptr = quant_tensor.data_ptr() as *mut u8;
            let code_ptr = code_tensor.data_ptr() as *mut f32;
            let absmax_ptr = absmax_tensor.data_ptr() as *mut f32;
            let output_ptr = output_tensor.data_ptr() as *mut std::ffi::c_void;

            match self.dtype {
                DType::BFloat16 => bitsandbytes_sys::dequantize_blockwise_bf16_gpu(
                    code_ptr, quant_ptr, absmax_ptr, output_ptr,
                    self.state.blocksize as i32, numel as i32,
                ),
                DType::Half => bitsandbytes_sys::dequantize_blockwise_fp16_gpu(
                    code_ptr, quant_ptr, absmax_ptr, output_ptr,
                    self.state.blocksize as i32, numel as i32,
                ),
                _ => unreachable!("GPU dequant path only for BF16/FP16"),
            }
        }

        // Reshape to original shape
        let result = output_tensor.reshape(&self.shape);

        tracing::trace!(
            "Dequantized tensor (GPU path, {:?}): {} bytes -> {} elements",
            self.dtype, self.data.len(), numel
        );

        Ok(result)
    }

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize {
        self.data.len() + self.state.code.len() * 4 + self.state.absmax.len() * 4
    }
}

/// Storage for KV cache - either full precision tensors or quantized
#[derive(Debug)]
enum KVStorage {
    /// Full precision tensor storage
    FullPrecision {
        keys: Option<Tensor>,
        values: Option<Tensor>,
    },
    /// Quantized storage with buffer for recent tokens (requires `bnb` feature)
    #[cfg(feature = "bnb")]
    Quantized {
        /// Quantized historical keys (contiguous)
        keys_quantized: Option<QuantizedTensor>,
        /// Quantized historical values (contiguous)
        values_quantized: Option<QuantizedTensor>,
        /// Number of tokens in quantized storage
        quantized_len: usize,

        /// Buffer for recent tokens (full precision) - not yet quantized
        keys_buffer: Option<Tensor>,
        /// Buffer for recent values (full precision)
        values_buffer: Option<Tensor>,
        /// Number of tokens in buffer
        buffer_len: usize,

        /// Cached dequantized view (valid for current forward pass)
        /// This avoids re-dequantizing for each layer
        dequant_keys: Option<Tensor>,
        dequant_values: Option<Tensor>,
        /// Position up to which dequant cache is valid
        dequant_valid_len: usize,

        /// Quantization type
        quant_type: KVQuantType,
        /// Template info for tensor creation
        dtype: Option<DType>,
        device: Option<Device>,
    },
    /// Paged block storage — fixed-size blocks from a shared pool.
    ///
    /// Eliminates GPU memory fragmentation by allocating from a pre-allocated
    /// pool of fixed-size blocks. `get()` assembles blocks into a contiguous
    /// view via `torch.cat`. Inspired by PagedAttention (vLLM, SOSP 2023).
    Paged {
        /// Block IDs for key tensors (one per BLOCK_SIZE tokens)
        key_blocks: Vec<BlockId>,
        /// Block IDs for value tensors
        value_blocks: Vec<BlockId>,
        /// Reference to the shared block pool
        pool: Arc<Mutex<BlockPool>>,
        /// Cached contiguous view (invalidated on update)
        cached_keys: Option<Tensor>,
        cached_values: Option<Tensor>,
        /// Position up to which the cached view is valid
        cached_valid_len: usize,
    },
}

/// KV cache for a single attention layer
#[derive(Debug)]
pub struct LayerKVCache {
    /// Storage for keys and values
    storage: KVStorage,
    /// Current sequence position (number of cached tokens)
    pub seq_pos: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Currently allocated capacity (grows in chunks up to max_seq_len)
    allocated_capacity: usize,
    /// Quantization type
    quant_type: KVQuantType,
}

impl LayerKVCache {
    /// Create a new layer KV cache
    #[cfg(feature = "bnb")]
    pub fn new(max_seq_len: usize, quant_type: KVQuantType) -> Self {
        let storage = match quant_type {
            KVQuantType::None => KVStorage::FullPrecision {
                keys: None,
                values: None,
            },
            _ => KVStorage::Quantized {
                keys_quantized: None,
                values_quantized: None,
                quantized_len: 0,
                keys_buffer: None,
                values_buffer: None,
                buffer_len: 0,
                dequant_keys: None,
                dequant_values: None,
                dequant_valid_len: 0,
                quant_type,
                dtype: None,
                device: None,
            },
        };

        Self {
            storage,
            seq_pos: 0,
            max_seq_len,
            allocated_capacity: 0,
            quant_type,
        }
    }

    /// Create a new layer KV cache (without bnb feature - quantization disabled)
    #[cfg(not(feature = "bnb"))]
    pub fn new(max_seq_len: usize, quant_type: KVQuantType) -> Self {
        if quant_type != KVQuantType::None {
            tracing::warn!(
                "KV cache quantization requested ({:?}) but 'bnb' feature not enabled. Falling back to full precision.",
                quant_type
            );
        }

        Self {
            storage: KVStorage::FullPrecision {
                keys: None,
                values: None,
            },
            seq_pos: 0,
            max_seq_len,
            allocated_capacity: 0,
            quant_type: KVQuantType::None,
        }
    }

    /// Create a new layer KV cache using paged block storage.
    pub fn new_paged(max_seq_len: usize, pool: Arc<Mutex<BlockPool>>) -> Self {
        Self {
            storage: KVStorage::Paged {
                key_blocks: Vec::new(),
                value_blocks: Vec::new(),
                pool,
                cached_keys: None,
                cached_values: None,
                cached_valid_len: 0,
            },
            seq_pos: 0,
            max_seq_len,
            allocated_capacity: 0,
            quant_type: KVQuantType::None,
        }
    }

    /// Ensure cache has capacity for required_len tokens (full precision only)
    fn ensure_capacity_fp(&mut self, required_len: usize, template: &Tensor) -> Result<()> {
        if required_len <= self.allocated_capacity {
            return Ok(());
        }

        // Round up to chunk boundary
        let new_capacity =
            required_len.div_ceil(DEFAULT_CHUNK_SIZE) * DEFAULT_CHUNK_SIZE;
        let new_capacity = new_capacity.min(self.max_seq_len);

        let shape = template.size();
        if shape.len() != 4 {
            return Err(anyhow!("Expected 4D tensor, got {:?}", shape));
        }
        let (batch_size, num_heads, head_dim) = (shape[0], shape[2], shape[3]);

        let device = template.device();
        let dtype = template.kind();
        let new_shape = [batch_size, new_capacity as i64, num_heads, head_dim];

        let cache_size_mb = estimate_tensor_size_mb(&new_shape, dtype);
        tracing::debug!(
            "Growing KV cache: {} -> {} tokens ({:.1} MB per tensor)",
            self.allocated_capacity,
            new_capacity,
            cache_size_mb
        );

        let new_keys = safe_zeros(&new_shape, (dtype, device))?;
        let new_values = safe_zeros(&new_shape, (dtype, device))?;

        // Copy existing data if any
        if let KVStorage::FullPrecision {
            keys: Some(old_keys),
            values: Some(old_values),
        } = &self.storage
        {
            if self.seq_pos > 0 {
                new_keys
                    .narrow(1, 0, self.seq_pos as i64)
                    .copy_(&old_keys.narrow(1, 0, self.seq_pos as i64));
                new_values
                    .narrow(1, 0, self.seq_pos as i64)
                    .copy_(&old_values.narrow(1, 0, self.seq_pos as i64));
            }
        }

        self.storage = KVStorage::FullPrecision {
            keys: Some(new_keys),
            values: Some(new_values),
        };
        self.allocated_capacity = new_capacity;

        Ok(())
    }

    /// Flush buffer to quantized storage
    #[cfg(feature = "bnb")]
    fn flush_buffer(&mut self) -> Result<()> {
        if let KVStorage::Quantized {
            keys_quantized,
            values_quantized,
            quantized_len,
            keys_buffer,
            values_buffer,
            buffer_len,
            dequant_keys,
            dequant_values,
            dequant_valid_len,
            quant_type,
            ..
        } = &mut self.storage
        {
            if *buffer_len == 0 {
                return Ok(());
            }

            let kb = keys_buffer
                .as_ref()
                .ok_or_else(|| anyhow!("Buffer keys missing"))?;
            let vb = values_buffer
                .as_ref()
                .ok_or_else(|| anyhow!("Buffer values missing"))?;

            // Get just the valid portion of the buffer
            let keys_to_quantize = kb.narrow(1, 0, *buffer_len as i64);
            let values_to_quantize = vb.narrow(1, 0, *buffer_len as i64);

            if let (Some(kq), Some(vq)) = (keys_quantized.as_ref(), values_quantized.as_ref()) {
                // Merge: dequantize existing + concat buffer + requantize
                let existing_keys = kq.to_tensor()?;
                let existing_values = vq.to_tensor()?;

                let merged_keys = Tensor::cat(&[existing_keys, keys_to_quantize], 1);
                let merged_values = Tensor::cat(&[existing_values, values_to_quantize], 1);

                *keys_quantized = Some(QuantizedTensor::from_tensor(&merged_keys, *quant_type)?);
                *values_quantized =
                    Some(QuantizedTensor::from_tensor(&merged_values, *quant_type)?);

                tracing::debug!(
                    "Flushed buffer: merged {} + {} = {} tokens",
                    *quantized_len,
                    *buffer_len,
                    *quantized_len + *buffer_len
                );
            } else {
                // First flush - just quantize the buffer
                *keys_quantized =
                    Some(QuantizedTensor::from_tensor(&keys_to_quantize, *quant_type)?);
                *values_quantized =
                    Some(QuantizedTensor::from_tensor(&values_to_quantize, *quant_type)?);

                tracing::debug!("Initial quantization: {} tokens", *buffer_len);
            }

            *quantized_len += *buffer_len;
            *buffer_len = 0;

            // Invalidate dequant cache since quantized data changed
            *dequant_keys = None;
            *dequant_values = None;
            *dequant_valid_len = 0;
        }

        Ok(())
    }

    /// Update cache with new keys and values
    pub fn update(
        &mut self,
        new_keys: &Tensor,
        new_values: &Tensor,
        start_pos: usize,
    ) -> Result<()> {
        let k_size = new_keys.size();
        if k_size.len() != 4 {
            return Err(anyhow!("Expected 4D tensor for keys, got {:?}", k_size));
        }
        let seq_len = k_size[1] as usize;

        // Check bounds against hard limit
        let end_pos = start_pos + seq_len;
        if end_pos > self.max_seq_len {
            return Err(anyhow!(
                "Cache overflow: trying to cache {} tokens starting at position {}, but max_seq_len is {}",
                seq_len, start_pos, self.max_seq_len
            ));
        }

        match &mut self.storage {
            KVStorage::FullPrecision { .. } => {
                // Grow cache if needed (chunked allocation)
                self.ensure_capacity_fp(end_pos, new_keys)?;

                // Re-borrow after potential reallocation
                if let KVStorage::FullPrecision {
                    keys: Some(cached_keys),
                    values: Some(cached_values),
                } = &self.storage
                {
                    cached_keys
                        .narrow(1, start_pos as i64, seq_len as i64)
                        .copy_(new_keys);
                    cached_values
                        .narrow(1, start_pos as i64, seq_len as i64)
                        .copy_(new_values);
                }
            }
            #[cfg(feature = "bnb")]
            KVStorage::Quantized {
                keys_quantized,
                values_quantized,
                quantized_len,
                keys_buffer,
                values_buffer,
                buffer_len,
                dequant_keys,
                dequant_values,
                dequant_valid_len,
                dtype,
                device,
                ..
            } => {
                // Store template info on first update
                if dtype.is_none() {
                    *dtype = Some(new_keys.kind());
                    *device = Some(new_keys.device());
                }

                // Invalidate dequant cache
                *dequant_keys = None;
                *dequant_values = None;
                *dequant_valid_len = 0;

                if start_pos == 0 {
                    // Fresh prompt - put everything in buffer, will be quantized on flush
                    *keys_quantized = None;
                    *values_quantized = None;
                    *quantized_len = 0;

                    *keys_buffer = Some(new_keys.shallow_clone());
                    *values_buffer = Some(new_values.shallow_clone());
                    *buffer_len = seq_len;

                    // Flush immediately if large prompt
                    if seq_len >= BUFFER_FLUSH_THRESHOLD {
                        self.flush_buffer()?;
                    }
                } else {
                    // Incremental update - append to buffer
                    let dt = dtype.ok_or_else(|| anyhow!("dtype not set"))?;
                    let dev = device.ok_or_else(|| anyhow!("device not set"))?;

                    // Ensure buffer exists and has capacity
                    let shape = new_keys.size();
                    let (batch_size, num_heads, head_dim) = (shape[0], shape[2], shape[3]);

                    if keys_buffer.is_none() {
                        // Create buffer with capacity for threshold + some extra
                        let buf_capacity = (BUFFER_FLUSH_THRESHOLD + 64) as i64;
                        *keys_buffer = Some(Tensor::zeros(
                            [batch_size, buf_capacity, num_heads, head_dim],
                            (dt, dev),
                        ));
                        *values_buffer = Some(Tensor::zeros(
                            [batch_size, buf_capacity, num_heads, head_dim],
                            (dt, dev),
                        ));
                    }

                    let kb = keys_buffer.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Internal error: keys buffer not initialized")
                    })?;

                    // Check if buffer needs to grow
                    let buf_capacity = kb.size()[1] as usize;
                    if *buffer_len + seq_len > buf_capacity {
                        // Flush current buffer first
                        self.flush_buffer()?;

                        // Re-borrow after flush
                        if let KVStorage::Quantized {
                            keys_buffer,
                            values_buffer,
                            buffer_len,
                            ..
                        } = &mut self.storage
                        {
                            // Create fresh buffer
                            let buf_capacity = (BUFFER_FLUSH_THRESHOLD + 64) as i64;
                            *keys_buffer = Some(Tensor::zeros(
                                [batch_size, buf_capacity, num_heads, head_dim],
                                (dt, dev),
                            ));
                            *values_buffer = Some(Tensor::zeros(
                                [batch_size, buf_capacity, num_heads, head_dim],
                                (dt, dev),
                            ));
                            *buffer_len = 0;
                        }
                    }

                    // Re-borrow for the actual copy
                    if let KVStorage::Quantized {
                        keys_buffer: Some(kb),
                        values_buffer: Some(vb),
                        buffer_len,
                        ..
                    } = &mut self.storage
                    {
                        // Copy new data into buffer
                        kb.narrow(1, *buffer_len as i64, seq_len as i64)
                            .copy_(new_keys);
                        vb.narrow(1, *buffer_len as i64, seq_len as i64)
                            .copy_(new_values);
                        *buffer_len += seq_len;
                    }

                    // Check if we should flush
                    if let KVStorage::Quantized { buffer_len, .. } = &self.storage {
                        if *buffer_len >= BUFFER_FLUSH_THRESHOLD {
                            self.flush_buffer()?;
                        }
                    }
                }
            }
            KVStorage::Paged {
                key_blocks,
                value_blocks,
                pool,
                cached_keys,
                cached_values,
                cached_valid_len,
            } => {
                // Invalidate cached contiguous view
                *cached_keys = None;
                *cached_values = None;
                *cached_valid_len = 0;

                let mut pool_guard = pool.lock();

                // Write K/V into blocks, allocating new blocks as needed
                let mut pos = start_pos;
                let mut token_offset = 0;
                while token_offset < seq_len {
                    let block_idx = pos / BLOCK_SIZE;
                    let offset_in_block = pos % BLOCK_SIZE;
                    let tokens_this_block = (BLOCK_SIZE - offset_in_block).min(seq_len - token_offset);

                    // Ensure we have enough blocks allocated
                    while key_blocks.len() <= block_idx {
                        let kid = pool_guard.allocate()
                            .ok_or_else(|| anyhow!("Block pool exhausted (keys)"))?;
                        let vid = pool_guard.allocate()
                            .ok_or_else(|| anyhow!("Block pool exhausted (values)"))?;
                        key_blocks.push(kid);
                        value_blocks.push(vid);
                    }

                    // Copy K/V slice into the block at the right offset
                    let k_block = pool_guard.get_block(key_blocks[block_idx]);
                    let v_block = pool_guard.get_block(value_blocks[block_idx]);

                    k_block
                        .narrow(1, offset_in_block as i64, tokens_this_block as i64)
                        .copy_(&new_keys.narrow(1, token_offset as i64, tokens_this_block as i64));
                    v_block
                        .narrow(1, offset_in_block as i64, tokens_this_block as i64)
                        .copy_(&new_values.narrow(1, token_offset as i64, tokens_this_block as i64));

                    pos += tokens_this_block;
                    token_offset += tokens_this_block;
                }
            }
        }

        // Update position
        self.seq_pos = end_pos;

        Ok(())
    }

    /// Get cached keys and values up to current position
    pub fn get(&mut self) -> Result<(Tensor, Tensor)> {
        if self.seq_pos == 0 {
            return Err(anyhow!("Cache is empty"));
        }

        match &mut self.storage {
            KVStorage::FullPrecision {
                keys: Some(cached_keys),
                values: Some(cached_values),
            } => {
                // Return views only - the .contiguous() in attention (after transposes)
                // will handle memory layout. Avoids double-copying K/V per layer.
                let keys_slice = cached_keys.narrow(1, 0, self.seq_pos as i64);
                let values_slice = cached_values.narrow(1, 0, self.seq_pos as i64);
                Ok((keys_slice, values_slice))
            }
            KVStorage::FullPrecision { .. } => Err(anyhow!("Keys/values cache not initialized")),
            #[cfg(feature = "bnb")]
            KVStorage::Quantized {
                keys_quantized,
                values_quantized,
                quantized_len,
                keys_buffer,
                values_buffer,
                buffer_len,
                dequant_keys,
                dequant_values,
                dequant_valid_len,
                ..
            } => {
                let total_len = *quantized_len + *buffer_len;

                // Check if we have a valid cached dequantized view
                if let (true, Some(dk), Some(dv)) = (
                    *dequant_valid_len == total_len,
                    dequant_keys.as_ref(),
                    dequant_values.as_ref(),
                ) {
                    return Ok((dk.shallow_clone(), dv.shallow_clone()));
                }

                // Need to build the full view
                let mut key_parts: Vec<Tensor> = Vec::new();
                let mut value_parts: Vec<Tensor> = Vec::new();

                // Part 1: Dequantized historical data
                if let (Some(kq), Some(vq)) = (keys_quantized.as_ref(), values_quantized.as_ref()) {
                    key_parts.push(kq.to_tensor()?);
                    value_parts.push(vq.to_tensor()?);
                }

                // Part 2: Buffer data
                if *buffer_len > 0 {
                    if let Some(kb) = keys_buffer.as_ref() {
                        key_parts.push(kb.narrow(1, 0, *buffer_len as i64));
                    }
                    if let Some(vb) = values_buffer.as_ref() {
                        value_parts.push(vb.narrow(1, 0, *buffer_len as i64));
                    }
                }

                let (keys, values) = if key_parts.is_empty() {
                    return Err(anyhow!("No data in cache"));
                } else if key_parts.len() == 1 {
                    (key_parts.remove(0), value_parts.remove(0))
                } else {
                    (
                        Tensor::cat(&key_parts, 1),
                        Tensor::cat(&value_parts, 1),
                    )
                };

                // Cache the dequantized view
                let keys_clone = keys.shallow_clone();
                let values_clone = values.shallow_clone();

                *dequant_keys = Some(keys_clone);
                *dequant_values = Some(values_clone);
                *dequant_valid_len = total_len;

                Ok((keys, values))
            }
            KVStorage::Paged {
                key_blocks,
                value_blocks,
                pool,
                cached_keys,
                cached_values,
                cached_valid_len,
            } => {
                // Return cached view if still valid
                if *cached_valid_len == self.seq_pos {
                    if let (Some(ck), Some(cv)) = (cached_keys.as_ref(), cached_values.as_ref()) {
                        return Ok((ck.shallow_clone(), cv.shallow_clone()));
                    }
                }

                let pool_guard = pool.lock();
                let num_full_blocks = self.seq_pos / BLOCK_SIZE;
                let remainder = self.seq_pos % BLOCK_SIZE;
                let total_blocks_used = if remainder > 0 { num_full_blocks + 1 } else { num_full_blocks };

                if total_blocks_used == 0 || key_blocks.is_empty() {
                    return Err(anyhow!("No data in paged cache"));
                }

                // Gather block tensors
                let mut key_parts: Vec<Tensor> = Vec::with_capacity(total_blocks_used);
                let mut value_parts: Vec<Tensor> = Vec::with_capacity(total_blocks_used);

                for i in 0..total_blocks_used {
                    let k = pool_guard.get_block(key_blocks[i]);
                    let v = pool_guard.get_block(value_blocks[i]);

                    if i == total_blocks_used - 1 && remainder > 0 {
                        // Last block: narrow to actual token count
                        key_parts.push(k.narrow(1, 0, remainder as i64));
                        value_parts.push(v.narrow(1, 0, remainder as i64));
                    } else {
                        key_parts.push(k.shallow_clone());
                        value_parts.push(v.shallow_clone());
                    }
                }

                let keys = if key_parts.len() == 1 {
                    key_parts.remove(0)
                } else {
                    Tensor::cat(&key_parts, 1)
                };
                let values = if value_parts.len() == 1 {
                    value_parts.remove(0)
                } else {
                    Tensor::cat(&value_parts, 1)
                };

                // Cache the assembled view
                *cached_keys = Some(keys.shallow_clone());
                *cached_values = Some(values.shallow_clone());
                *cached_valid_len = self.seq_pos;

                Ok((keys, values))
            }
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        match &mut self.storage {
            KVStorage::FullPrecision { keys, values } => {
                *keys = None;
                *values = None;
            }
            #[cfg(feature = "bnb")]
            KVStorage::Quantized {
                keys_quantized,
                values_quantized,
                quantized_len,
                keys_buffer,
                values_buffer,
                buffer_len,
                dequant_keys,
                dequant_values,
                dequant_valid_len,
                dtype,
                device,
                ..
            } => {
                *keys_quantized = None;
                *values_quantized = None;
                *quantized_len = 0;
                *keys_buffer = None;
                *values_buffer = None;
                *buffer_len = 0;
                *dequant_keys = None;
                *dequant_values = None;
                *dequant_valid_len = 0;
                *dtype = None;
                *device = None;
            }
            KVStorage::Paged {
                key_blocks,
                value_blocks,
                pool,
                cached_keys,
                cached_values,
                cached_valid_len,
            } => {
                // Return all blocks to the pool
                let mut pool_guard = pool.lock();
                for &id in key_blocks.iter() {
                    pool_guard.free(id);
                }
                for &id in value_blocks.iter() {
                    pool_guard.free(id);
                }
                key_blocks.clear();
                value_blocks.clear();
                *cached_keys = None;
                *cached_values = None;
                *cached_valid_len = 0;
            }
        }
        self.seq_pos = 0;
        self.allocated_capacity = 0;
    }

    /// Truncate cache to keep only tokens at positions 0..pos.
    ///
    /// For FullPrecision and Paged, this adjusts position markers — allocated
    /// capacity/blocks remain and will be reused. For Paged, excess blocks
    /// beyond the last needed block are freed back to the pool.
    ///
    /// For Quantized, this falls back to a full clear since truncation within
    /// quantized blocks is complex.
    pub fn truncate_to(&mut self, pos: usize) {
        if pos >= self.seq_pos {
            return; // Nothing to truncate
        }
        match &mut self.storage {
            KVStorage::FullPrecision { .. } => {
                // Just move the position marker back. Allocated capacity stays.
                self.seq_pos = pos;
            }
            #[cfg(feature = "bnb")]
            KVStorage::Quantized { .. } => {
                // Truncation within quantized storage is complex — clear and recompute.
                self.clear();
            }
            KVStorage::Paged {
                key_blocks,
                value_blocks,
                pool,
                cached_keys,
                cached_values,
                cached_valid_len,
            } => {
                // Free blocks beyond the last needed block
                let blocks_needed = if pos == 0 { 0 } else { (pos - 1) / BLOCK_SIZE + 1 };
                if key_blocks.len() > blocks_needed {
                    let mut pool_guard = pool.lock();
                    for &id in &key_blocks[blocks_needed..] {
                        pool_guard.free(id);
                    }
                    for &id in &value_blocks[blocks_needed..] {
                        pool_guard.free(id);
                    }
                    key_blocks.truncate(blocks_needed);
                    value_blocks.truncate(blocks_needed);
                }
                // Invalidate cached view
                *cached_keys = None;
                *cached_values = None;
                *cached_valid_len = 0;
                self.seq_pos = pos;
            }
        }
    }

    /// Check if cache is initialized
    pub fn is_initialized(&self) -> bool {
        match &self.storage {
            KVStorage::FullPrecision { keys, values } => keys.is_some() && values.is_some(),
            #[cfg(feature = "bnb")]
            KVStorage::Quantized {
                quantized_len,
                buffer_len,
                ..
            } => *quantized_len > 0 || *buffer_len > 0,
            KVStorage::Paged { key_blocks, .. } => !key_blocks.is_empty(),
        }
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        match &self.storage {
            KVStorage::FullPrecision { keys, values } => {
                let mut total = 0;
                if let Some(k) = keys {
                    total += k.numel() * dtype_element_size(k.kind());
                }
                if let Some(v) = values {
                    total += v.numel() * dtype_element_size(v.kind());
                }
                total
            }
            #[cfg(feature = "bnb")]
            KVStorage::Quantized {
                keys_quantized,
                values_quantized,
                keys_buffer,
                values_buffer,
                buffer_len,
                ..
            } => {
                let mut total = 0;

                // Quantized storage
                if let Some(kq) = keys_quantized {
                    total += kq.memory_usage();
                }
                if let Some(vq) = values_quantized {
                    total += vq.memory_usage();
                }

                // Buffer (only count used portion)
                if let Some(kb) = keys_buffer {
                    if *buffer_len > 0 {
                        let shape = kb.size();
                        let per_token = shape[0] * shape[2] * shape[3];
                        total += (*buffer_len as i64 * per_token) as usize
                            * dtype_element_size(kb.kind());
                    }
                }
                if let Some(vb) = values_buffer {
                    if *buffer_len > 0 {
                        let shape = vb.size();
                        let per_token = shape[0] * shape[2] * shape[3];
                        total += (*buffer_len as i64 * per_token) as usize
                            * dtype_element_size(vb.kind());
                    }
                }

                // Note: dequant cache is temporary and not counted
                total
            }
            KVStorage::Paged { key_blocks, pool, .. } => {
                // Count blocks owned by this layer (K + V)
                let pool_guard = pool.lock();
                let elem_size = match pool_guard.dtype() {
                    DType::Half | DType::BFloat16 => 2,
                    _ => 4,
                };
                let shape = pool_guard.block_shape();
                let per_block = BLOCK_SIZE * shape[2] as usize * shape[3] as usize * elem_size;
                key_blocks.len() * 2 * per_block // K + V blocks
            }
        }
    }

    /// Move tensors to a different device (for CPU offload/restore).
    ///
    /// For FullPrecision: moves K/V tensors via `Tensor::to_device()`.
    /// For Paged: no-op (blocks are managed by the pool).
    /// For Quantized: no-op (quantized data is already on CPU).
    pub fn to_device(&mut self, device: tch::Device) {
        match &mut self.storage {
            KVStorage::FullPrecision { keys, values } => {
                if let Some(k) = keys {
                    *k = k.to_device(device);
                }
                if let Some(v) = values {
                    *v = v.to_device(device);
                }
            }
            #[cfg(feature = "bnb")]
            KVStorage::Quantized { .. } => {
                // Quantized data lives on CPU already; dequant cache will be
                // regenerated on next get(). No action needed.
            }
            KVStorage::Paged { .. } => {
                // Paged blocks are managed by the pool. Offloading individual
                // layers is not supported — use block pool eviction instead.
            }
        }
    }

    /// Get the quantization type
    pub fn quant_type(&self) -> KVQuantType {
        self.quant_type
    }
}

// Backwards compatibility: expose keys/values for code that accesses them directly
impl LayerKVCache {
    /// Get keys tensor (only for full precision mode)
    pub fn keys(&self) -> Option<&Tensor> {
        match &self.storage {
            KVStorage::FullPrecision { keys, .. } => keys.as_ref(),
            #[cfg(feature = "bnb")]
            KVStorage::Quantized { .. } => None,
            KVStorage::Paged { .. } => None, // Use get() for paged mode
        }
    }

    /// Get values tensor (only for full precision mode)
    pub fn values(&self) -> Option<&Tensor> {
        match &self.storage {
            KVStorage::FullPrecision { values, .. } => values.as_ref(),
            #[cfg(feature = "bnb")]
            KVStorage::Quantized { .. } => None,
            KVStorage::Paged { .. } => None, // Use get() for paged mode
        }
    }
}

/// Get current timestamp in milliseconds since UNIX_EPOCH
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Where a KV cache's tensors are currently stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLocation {
    /// Tensors are on GPU (active, ready for inference)
    Gpu,
    /// Tensors have been offloaded to CPU RAM (idle, must restore before use)
    Cpu,
}

/// KV cache manager for all layers in a model
pub struct KVCacheManager {
    /// Cache for each layer (lock-free concurrent access)
    layer_caches: DashMap<usize, LayerKVCache>,
    /// Maximum sequence length
    #[allow(dead_code)]
    max_seq_len: usize,
    /// Whether caching is enabled
    enabled: bool,
    /// Quantization type
    quant_type: KVQuantType,
    /// Last access timestamp in milliseconds (for LRU eviction)
    last_access_ms: AtomicU64,
    /// Access count (for metrics)
    access_count: AtomicU64,
    /// Token IDs this cache was computed for (for prefix matching across turns)
    cached_token_ids: Vec<i64>,
    /// Where this cache's tensors currently reside
    location: CacheLocation,
}

impl KVCacheManager {
    /// Create a new KV cache manager
    pub fn new(num_layers: usize, max_seq_len: usize, quant_type: KVQuantType) -> Self {
        tracing::info!(
            "[KVCacheManager::new] Creating cache for {} layers, max_seq_len={}, quant={:?}",
            num_layers,
            max_seq_len,
            quant_type
        );

        let layer_caches = DashMap::new();
        for layer_idx in 0..num_layers {
            layer_caches.insert(layer_idx, LayerKVCache::new(max_seq_len, quant_type));
        }

        Self {
            layer_caches,
            max_seq_len,
            enabled: true,
            quant_type,
            last_access_ms: AtomicU64::new(current_timestamp_ms()),
            access_count: AtomicU64::new(0),
            cached_token_ids: Vec::new(),
            location: CacheLocation::Gpu,
        }
    }

    /// Create a new KV cache manager using paged block storage.
    ///
    /// All layers share the same `BlockPool` for zero-fragmentation memory management.
    pub fn new_paged(num_layers: usize, max_seq_len: usize, pool: Arc<Mutex<BlockPool>>) -> Self {
        tracing::info!(
            "[KVCacheManager::new_paged] Creating paged cache for {} layers, max_seq_len={}",
            num_layers, max_seq_len,
        );

        let layer_caches = DashMap::new();
        for layer_idx in 0..num_layers {
            layer_caches.insert(layer_idx, LayerKVCache::new_paged(max_seq_len, pool.clone()));
        }

        Self {
            layer_caches,
            max_seq_len,
            enabled: true,
            quant_type: KVQuantType::None,
            last_access_ms: AtomicU64::new(current_timestamp_ms()),
            access_count: AtomicU64::new(0),
            cached_token_ids: Vec::new(),
            location: CacheLocation::Gpu,
        }
    }

    /// Get cache for a specific layer with a closure
    pub fn with_layer_cache<F, R>(&self, layer_idx: usize, f: F) -> Option<R>
    where
        F: FnOnce(&mut LayerKVCache) -> R,
    {
        if !self.enabled {
            return None;
        }
        self.layer_caches.get_mut(&layer_idx).map(|mut cache_ref| f(&mut cache_ref))
    }

    /// Check if a layer cache exists (for testing)
    #[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
    fn get_layer_cache(&self, layer_idx: usize) -> Option<()> {
        if !self.enabled {
            return None;
        }
        self.layer_caches.contains_key(&layer_idx).then_some(())
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        for mut cache_ref in self.layer_caches.iter_mut() {
            cache_ref.clear();
        }
    }

    /// Enable or disable caching
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear_all();
        }
    }

    /// Get total memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        if !self.enabled {
            return 0;
        }

        self.layer_caches.iter().map(|c| c.memory_usage()).sum()
    }

    /// Get the quantization type
    pub fn quant_type(&self) -> KVQuantType {
        self.quant_type
    }

    /// Update last access timestamp (for LRU eviction)
    pub fn touch(&self) {
        self.last_access_ms.store(current_timestamp_ms(), Ordering::Relaxed);
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get last access timestamp in milliseconds
    pub fn last_access(&self) -> u64 {
        self.last_access_ms.load(Ordering::Relaxed)
    }

    /// Get access count
    pub fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }

    /// Find the length of the common prefix between cached tokens and new tokens.
    ///
    /// Returns the number of tokens whose KV values can be reused from the cache.
    /// A return value of 0 means no prefix match (full recomputation needed).
    pub fn prefix_match_len(&self, new_tokens: &[i64]) -> usize {
        self.cached_token_ids
            .iter()
            .zip(new_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Record which tokens this cache was computed for.
    ///
    /// Called after generation completes so the next turn can detect prefix overlap.
    pub fn set_cached_tokens(&mut self, tokens: Vec<i64>) {
        self.cached_token_ids = tokens;
    }

    /// Get the cached token IDs (for debugging/metrics).
    pub fn cached_token_count(&self) -> usize {
        self.cached_token_ids.len()
    }

    /// Truncate the cache to a given token position.
    ///
    /// Keeps KV values for tokens 0..pos and discards everything after.
    /// Used when a prefix matches but the suffix has changed (new turn).
    pub fn truncate_to(&self, pos: usize) {
        for mut cache_ref in self.layer_caches.iter_mut() {
            cache_ref.truncate_to(pos);
        }
    }

    /// Current storage location of this cache's tensors.
    pub fn location(&self) -> CacheLocation {
        self.location
    }

    /// Offload all layer caches from GPU to CPU RAM.
    ///
    /// Frees GPU memory while preserving the cached data for later restore.
    /// Only works for FullPrecision storage — Paged caches free blocks to the
    /// pool instead (handled by evict_to_budget), and Quantized data is already
    /// partially on CPU.
    pub fn offload_to_cpu(&mut self) {
        if self.location == CacheLocation::Cpu {
            return; // Already offloaded
        }
        for mut cache_ref in self.layer_caches.iter_mut() {
            cache_ref.to_device(tch::Device::Cpu);
        }
        self.location = CacheLocation::Cpu;
        tracing::debug!(
            "Offloaded KV cache to CPU ({} tokens, {} layers)",
            self.cached_token_ids.len(),
            self.layer_caches.len()
        );
    }

    /// Restore all layer caches from CPU back to GPU.
    ///
    /// Must be called before using the cache for inference.
    pub fn restore_to_gpu(&mut self, device: tch::Device) {
        if self.location == CacheLocation::Gpu {
            return; // Already on GPU
        }
        for mut cache_ref in self.layer_caches.iter_mut() {
            cache_ref.to_device(device);
        }
        self.location = CacheLocation::Gpu;
        tracing::debug!(
            "Restored KV cache to GPU ({} tokens, {} layers)",
            self.cached_token_ids.len(),
            self.layer_caches.len()
        );
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_kv_cache_unquantized() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::Float;

        let mut cache = LayerKVCache::new(100, KVQuantType::None);

        let batch_size = 2;
        let seq_len = 10;
        let num_heads = 8;
        let head_dim = 64;

        let keys = Tensor::randn(
            [batch_size, seq_len as i64, num_heads, head_dim],
            (dtype, device),
        );
        let values = Tensor::randn(
            [batch_size, seq_len as i64, num_heads, head_dim],
            (dtype, device),
        );

        cache.update(&keys, &values, 0)?;
        assert_eq!(cache.seq_pos, 10);

        let (cached_keys, cached_values) = cache.get()?;
        assert_eq!(
            cached_keys.size(),
            vec![batch_size, 10, num_heads, head_dim]
        );
        assert_eq!(
            cached_values.size(),
            vec![batch_size, 10, num_heads, head_dim]
        );

        Ok(())
    }

    #[test]
    #[cfg(feature = "bnb")]
    fn test_layer_kv_cache_quantized_int8() -> Result<()> {
        // Skip if bitsandbytes library is not available (stub mode)
        if !bitsandbytes_sys::is_available() {
            eprintln!("Skipping quantized test: bitsandbytes library not available");
            return Ok(());
        }

        let device = Device::Cpu;
        let dtype = DType::Float;

        let mut cache = LayerKVCache::new(1000, KVQuantType::Int8);

        let batch_size: i64 = 1;
        let seq_len: i64 = 100; // Large enough to trigger flush
        let num_heads: i64 = 2;
        let head_dim: i64 = 64;

        let keys = Tensor::randn([batch_size, seq_len, num_heads, head_dim], (dtype, device));
        let values = Tensor::randn([batch_size, seq_len, num_heads, head_dim], (dtype, device));

        // Initial prompt
        cache.update(&keys, &values, 0)?;
        assert_eq!(cache.seq_pos, seq_len as usize);

        // Get should work
        let (cached_keys, cached_values) = cache.get()?;
        assert_eq!(
            cached_keys.size(),
            vec![batch_size, seq_len, num_heads, head_dim]
        );
        assert_eq!(
            cached_values.size(),
            vec![batch_size, seq_len, num_heads, head_dim]
        );

        // Second get should use cache
        let (cached_keys2, _) = cache.get()?;
        assert_eq!(cached_keys2.size(), cached_keys.size());

        // Add more tokens
        let new_keys = Tensor::randn([batch_size, 1, num_heads, head_dim], (dtype, device));
        let new_values = Tensor::randn([batch_size, 1, num_heads, head_dim], (dtype, device));
        cache.update(&new_keys, &new_values, seq_len as usize)?;

        let (final_keys, _) = cache.get()?;
        assert_eq!(
            final_keys.size(),
            vec![batch_size, seq_len + 1, num_heads, head_dim]
        );

        Ok(())
    }

    #[test]
    fn test_kv_cache_manager() {
        let num_layers = 32;
        let max_seq_len = 2048;

        let mut manager = KVCacheManager::new(num_layers, max_seq_len, KVQuantType::None);

        for layer_idx in 0..num_layers {
            assert!(manager.get_layer_cache(layer_idx).is_some());
        }

        manager.set_enabled(false);
        assert!(manager.get_layer_cache(0).is_none());

        manager.set_enabled(true);
        assert!(manager.get_layer_cache(0).is_some());
    }

    #[test]
    fn test_kv_cache_manager_quantized() {
        let num_layers = 4;
        let max_seq_len = 512;

        let manager = KVCacheManager::new(num_layers, max_seq_len, KVQuantType::Nf4);

        assert_eq!(manager.quant_type(), KVQuantType::Nf4);
    }

    #[test]
    fn test_prefix_match_len() {
        let mut manager = KVCacheManager::new(2, 100, KVQuantType::None);

        // No cached tokens — no match
        assert_eq!(manager.prefix_match_len(&[1, 2, 3]), 0);

        // Set cached tokens
        manager.set_cached_tokens(vec![1, 2, 3, 4, 5]);

        // Full prefix match
        assert_eq!(manager.prefix_match_len(&[1, 2, 3, 4, 5, 6, 7]), 5);

        // Partial prefix match
        assert_eq!(manager.prefix_match_len(&[1, 2, 3, 99, 100]), 3);

        // No match (different first token)
        assert_eq!(manager.prefix_match_len(&[99, 2, 3]), 0);

        // Exact match (entire prompt is cached)
        assert_eq!(manager.prefix_match_len(&[1, 2, 3, 4, 5]), 5);

        // Empty input
        assert_eq!(manager.prefix_match_len(&[]), 0);
    }

    #[test]
    fn test_truncate_to_full_precision() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::Float;
        let mut cache = LayerKVCache::new(100, KVQuantType::None);

        let batch_size = 1;
        let num_heads = 4;
        let head_dim = 8;

        // Write 10 tokens
        let keys = Tensor::ones([batch_size, 10, num_heads, head_dim], (dtype, device));
        let values = Tensor::ones([batch_size, 10, num_heads, head_dim], (dtype, device)) * 2.0;
        cache.update(&keys, &values, 0)?;
        assert_eq!(cache.seq_pos, 10);

        // Truncate to 5
        cache.truncate_to(5);
        assert_eq!(cache.seq_pos, 5);

        // get() should return only 5 tokens
        let (k, v) = cache.get()?;
        assert_eq!(k.size()[1], 5);
        assert_eq!(v.size()[1], 5);

        // Writing new tokens at position 5 should work
        let new_keys = Tensor::ones([batch_size, 3, num_heads, head_dim], (dtype, device)) * 3.0;
        let new_values = Tensor::ones([batch_size, 3, num_heads, head_dim], (dtype, device)) * 4.0;
        cache.update(&new_keys, &new_values, 5)?;
        assert_eq!(cache.seq_pos, 8);

        let (k, _v) = cache.get()?;
        assert_eq!(k.size()[1], 8);

        Ok(())
    }

    #[test]
    fn test_block_pool_allocate_free() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::Float;
        let mut pool = BlockPool::new(4, 2, 8, device, dtype)?;

        assert_eq!(pool.total_blocks(), 4);
        assert_eq!(pool.free_block_count(), 4);
        assert_eq!(pool.used_blocks(), 0);

        // Allocate 3 blocks
        let b0 = pool.allocate().expect("should allocate");
        let b1 = pool.allocate().expect("should allocate");
        let b2 = pool.allocate().expect("should allocate");
        assert_eq!(pool.used_blocks(), 3);
        assert_eq!(pool.free_block_count(), 1);

        // Allocate last block
        let _b3 = pool.allocate().expect("should allocate");
        assert_eq!(pool.free_block_count(), 0);

        // Pool exhausted
        assert!(pool.allocate().is_none());

        // Free a block — now one available
        pool.free(b1);
        assert_eq!(pool.free_block_count(), 1);

        let reused = pool.allocate().expect("should allocate freed block");
        assert_eq!(reused, b1); // LIFO: should get back the same block

        // Free all
        pool.free(b0);
        pool.free(reused);
        pool.free(b2);
        pool.free(_b3);
        assert_eq!(pool.free_block_count(), 4);

        Ok(())
    }

    #[test]
    fn test_paged_kv_cache_update_and_get() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::Float;
        let num_kv_heads = 2;
        let head_dim = 4;

        // Create pool with enough blocks for the test
        let pool = Arc::new(Mutex::new(
            BlockPool::new(8, num_kv_heads, head_dim, device, dtype)?
        ));

        let mut cache = LayerKVCache::new_paged(1024, pool.clone());

        // Write 10 tokens (less than one block of 256)
        let keys = Tensor::ones([1, 10, num_kv_heads as i64, head_dim as i64], (dtype, device));
        let values = Tensor::ones([1, 10, num_kv_heads as i64, head_dim as i64], (dtype, device)) * 2.0;
        cache.update(&keys, &values, 0)?;
        assert_eq!(cache.seq_pos, 10);

        // Get should return 10 tokens
        let (k, v) = cache.get()?;
        assert_eq!(k.size()[1], 10);
        assert_eq!(v.size()[1], 10);

        // Append 5 more tokens
        let more_keys = Tensor::ones([1, 5, num_kv_heads as i64, head_dim as i64], (dtype, device)) * 3.0;
        let more_values = Tensor::ones([1, 5, num_kv_heads as i64, head_dim as i64], (dtype, device)) * 4.0;
        cache.update(&more_keys, &more_values, 10)?;
        assert_eq!(cache.seq_pos, 15);

        let (k, _v) = cache.get()?;
        assert_eq!(k.size()[1], 15);

        // Clear should free blocks back to pool
        let used_before = pool.lock().used_blocks();
        assert!(used_before > 0);
        cache.clear();
        assert_eq!(pool.lock().used_blocks(), used_before - 2); // 2 blocks freed (K + V)
        assert_eq!(cache.seq_pos, 0);

        Ok(())
    }

    #[test]
    fn test_cpu_offload_restore() -> Result<()> {
        let device = Device::Cpu; // Use CPU as "GPU" for test
        let dtype = DType::Float;

        let mut manager = KVCacheManager::new(2, 100, KVQuantType::None);
        assert_eq!(manager.location(), CacheLocation::Gpu);

        // Write some data via with_layer_cache
        manager.with_layer_cache(0, |cache| {
            let keys = Tensor::ones([1, 5, 4, 8], (dtype, device));
            let values = Tensor::ones([1, 5, 4, 8], (dtype, device)) * 2.0;
            cache.update(&keys, &values, 0).unwrap();
        });

        // Offload to CPU (no-op on CPU device, but tests the state machine)
        manager.offload_to_cpu();
        assert_eq!(manager.location(), CacheLocation::Cpu);

        // Restore back
        manager.restore_to_gpu(device);
        assert_eq!(manager.location(), CacheLocation::Gpu);

        // Data should still be accessible
        let result = manager.with_layer_cache(0, |cache| {
            let (k, _v) = cache.get().unwrap();
            k.size()[1]
        });
        assert_eq!(result, Some(5));

        Ok(())
    }

    #[test]
    fn test_eviction_with_budget() {
        let config = CacheConfig::new(2, 100);
        let registry = KVCacheRegistry::new(config, Some(1)); // 1 byte budget = always evict

        // Create two session caches
        let _cache_a = registry.get_or_create(CacheOwner::Session("a".into()));
        let _cache_b = registry.get_or_create(CacheOwner::Session("b".into()));
        assert_eq!(registry.cache_count(), 2);

        // Eviction should try to free caches (offload first, then remove)
        registry.evict_to_budget();

        // With a 1-byte budget and no actual GPU memory, both should be offloaded/removed
        // The exact behavior depends on memory_usage() returning >0 for non-empty caches
    }

    // ========================================================================
    // Multi-turn prefix matching simulation tests
    // ========================================================================

    #[test]
    fn test_prefix_match_multi_turn_simulation() {
        let mut manager = KVCacheManager::new(2, 1024, KVQuantType::None);

        // Simulate tokenized prompts growing across conversation turns.
        // Each turn's prompt includes all prior history + new message.

        // Turn 1: [sys sys sys user1 user1]
        let turn1_tokens = vec![10, 20, 30, 100, 101];
        manager.set_cached_tokens(turn1_tokens.clone());

        // Turn 2: [sys sys sys user1 user1 asst1 asst1 user2]
        // The first 5 tokens match turn 1
        let turn2_tokens = vec![10, 20, 30, 100, 101, 200, 201, 102];
        assert_eq!(manager.prefix_match_len(&turn2_tokens), 5);

        // After turn 2 completes, update cached tokens
        manager.set_cached_tokens(turn2_tokens.clone());

        // Turn 3: [sys sys sys user1 user1 asst1 asst1 user2 asst2 user3]
        // The first 8 tokens match turn 2
        let turn3_tokens = vec![10, 20, 30, 100, 101, 200, 201, 102, 202, 103];
        assert_eq!(manager.prefix_match_len(&turn3_tokens), 8);

        // Verify no match if conversation diverges at the start
        let divergent = vec![99, 20, 30, 100, 101];
        manager.set_cached_tokens(turn3_tokens);
        assert_eq!(manager.prefix_match_len(&divergent), 0);
    }

    #[test]
    fn test_prefix_match_with_tool_call_in_history() {
        let mut manager = KVCacheManager::new(2, 1024, KVQuantType::None);

        // Turn 1: [sys user1_ask_weather]
        let turn1_tokens = vec![10, 100, 101];
        manager.set_cached_tokens(turn1_tokens);

        // Turn 2 includes tool call and response in history:
        // [sys user1_ask_weather asst_toolcall tool_resp user2_followup]
        // Prefix match = 3 (sys + user1 tokens)
        let turn2_tokens = vec![10, 100, 101, 200, 201, 202, 300, 301, 110, 111];
        assert_eq!(manager.prefix_match_len(&turn2_tokens), 3);

        // After turn 2, cache the full sequence
        manager.set_cached_tokens(turn2_tokens.clone());

        // Turn 3: same prefix as turn 2 + new user message
        // [sys user1_ask_weather asst_toolcall tool_resp user2_followup asst2 user3]
        let turn3_tokens = vec![10, 100, 101, 200, 201, 202, 300, 301, 110, 111, 210, 120];
        assert_eq!(manager.prefix_match_len(&turn3_tokens), 10);
    }
}
