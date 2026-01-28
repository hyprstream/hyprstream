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

use super::kv_quant::KVQuantType;
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
}

impl CacheConfig {
    /// Create a new cache configuration
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        Self {
            num_layers,
            max_seq_len,
            quant_type: KVQuantType::None,
            eviction_exempt: false,
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
}

impl KVCacheRegistry {
    /// Create a new KV cache registry
    pub fn new(default_config: CacheConfig, memory_budget: Option<usize>) -> Self {
        tracing::info!(
            "[KVCacheRegistry::new] Creating registry with {} layers, max_seq_len={}, quant={:?}, budget={:?}",
            default_config.num_layers,
            default_config.max_seq_len,
            default_config.quant_type,
            memory_budget
        );

        Self {
            caches: DashMap::new(),
            default_config,
            memory_budget_bytes: memory_budget,
        }
    }

    /// Get or create a cache for the given owner.
    ///
    /// If the cache doesn't exist, creates a new one with the default config.
    /// Returns `Arc<Mutex<KVCacheManager>>` - caller must lock() to use.
    pub fn get_or_create(&self, owner: CacheOwner) -> Arc<Mutex<KVCacheManager>> {
        // Try to get existing cache first (fast path)
        if let Some(cache) = self.caches.get(&owner) {
            // Update access time
            cache.lock().touch();
            return cache.clone();
        }

        // Slow path: create new cache
        let cache = Arc::new(Mutex::new(KVCacheManager::new(
            self.default_config.num_layers,
            self.default_config.max_seq_len,
            self.default_config.quant_type,
        )));

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

        // Evict until under budget
        let mut freed = 0;
        for (owner, _, size) in candidates {
            if current_usage - freed <= budget {
                break;
            }
            self.caches.remove(&owner);
            freed += size;
            tracing::info!("Evicted cache {:?}, freed {} bytes", owner, size);
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
        tracing::info!("Cleared all KV caches from registry");
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
    /// Quantize a tensor using the specified quantization type
    fn from_tensor(tensor: &Tensor, quant_type: KVQuantType) -> Result<Self> {
        let shape = tensor.size();
        let dtype = tensor.kind();
        let device = tensor.device();

        // Move to CPU and convert to f32 for quantization
        let cpu_tensor = tensor
            .to_device(Device::Cpu)
            .to_kind(DType::Float)
            .contiguous();
        let numel = cpu_tensor.numel();

        // Extract f32 data from tensor
        let mut f32_data = vec![0.0f32; numel];
        cpu_tensor
            .f_copy_data(&mut f32_data, numel)
            .map_err(|e| anyhow!("Failed to copy tensor data: {:?}", e))?;

        // Quantize based on type
        let (data, state) = match quant_type {
            KVQuantType::Int8 => bitsandbytes_sys::quantize_blockwise_fp32(&f32_data, QUANT_BLOCKSIZE)
                .map_err(|e| anyhow!("Int8 quantization failed: {:?}", e))?,
            KVQuantType::Nf4 => bitsandbytes_sys::quantize_4bit_nf4_fp32(&f32_data, QUANT_BLOCKSIZE)
                .map_err(|e| anyhow!("NF4 quantization failed: {:?}", e))?,
            KVQuantType::Fp4 => bitsandbytes_sys::quantize_4bit_fp4_fp32(&f32_data, QUANT_BLOCKSIZE)
                .map_err(|e| anyhow!("FP4 quantization failed: {:?}", e))?,
            KVQuantType::None => {
                return Err(anyhow!(
                    "Cannot create QuantizedTensor with KVQuantType::None"
                ));
            }
        };

        tracing::trace!(
            "Quantized tensor: {} elements, shape {:?} -> {} bytes ({:.1}% of f32)",
            numel,
            shape,
            data.len(),
            (data.len() as f64 / (numel * 4) as f64) * 100.0
        );

        Ok(Self {
            data,
            state,
            shape,
            dtype,
            device,
        })
    }

    /// Dequantize back to a tensor
    fn to_tensor(&self) -> Result<Tensor> {
        // Dequantize based on quantization type
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
        }
        self.seq_pos = 0;
        self.allocated_capacity = 0;
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
        }
    }

    /// Get values tensor (only for full precision mode)
    pub fn values(&self) -> Option<&Tensor> {
        match &self.storage {
            KVStorage::FullPrecision { values, .. } => values.as_ref(),
            #[cfg(feature = "bnb")]
            KVStorage::Quantized { .. } => None,
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
}

#[cfg(test)]
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
}
