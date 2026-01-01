//! Llama model implementation with support for Llama 1/2/3 and GQA

use super::{ArchitectureConfig, ModelArchitecture, ModelOperations};
// use super::lora_adapter::ArchitectureAwareLoRAAdapter; // Module removed
use crate::runtime::rope::{RoPE, RoPEManager};
use crate::runtime::tensor_helpers::{
    broadcast_add, broadcast_mul, dims3, dims4, scalar_tensor, square_tensor,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use tch::{nn, Device, Kind as DType, Tensor};

/// Linear projection layer with optional bias
///
/// This is a zero-cost abstraction that encapsulates weight matrices
/// and optional bias vectors for linear transformations.
struct LinearProjection {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearProjection {
    /// Create projection from weight only (no bias)
    #[inline]
    fn new(weight: Tensor) -> Self {
        Self { weight, bias: None }
    }

    /// Create projection with weight and bias
    #[inline]
    fn with_bias(weight: Tensor, bias: Tensor) -> Self {
        Self {
            weight,
            bias: Some(bias),
        }
    }

    /// Apply projection to input: output = input @ weight + bias
    ///
    /// Input shape: [*, in_features]
    /// Weight shape: [in_features, out_features] (already transposed)
    /// Bias shape: [out_features] (broadcasted)
    /// Output shape: [*, out_features]
    #[inline]
    fn apply(&self, input: &Tensor) -> Tensor {
        let output = input.matmul(&self.weight);

        match &self.bias {
            Some(bias) => output + bias,
            None => output,
        }
    }

    /// Get output dimension
    #[inline]
    #[allow(dead_code)]
    fn out_features(&self) -> i64 {
        self.weight.size()[1]
    }
}

unsafe impl Send for LinearProjection {}
unsafe impl Sync for LinearProjection {}

/// Calculate padded vocabulary size to multiple of 64 (for performance optimization)
#[inline]
fn calculate_padded_vocab_size(vocab_size: usize) -> usize {
    vocab_size.div_ceil(64) * 64
}

/// Llama model configuration
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Llama version (1, 2, or 3)
    pub version: u8,
    /// Number of attention heads for queries
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA in Llama 2/3)
    pub num_key_value_heads: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Intermediate size for FFN
    pub intermediate_size: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,
    /// Vocabulary size (may be padded)
    pub vocab_size: usize,
    /// Original vocabulary size (before padding)
    pub original_vocab_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// RoPE theta
    pub rope_theta: f32,
    /// RoPE scaling (for Llama 3)
    pub rope_scaling: Option<RopeScaling>,
    /// Hidden activation function (silu for Llama, gelu_pytorch_tanh for Gemma3)
    pub hidden_activation: String,
    /// Query pre-attention scalar for QK-norm (Gemma3)
    pub query_pre_attn_scalar: Option<f32>,
    /// Use QK-norm (Gemma3)
    pub use_qk_norm: bool,
    /// Scale embeddings by sqrt(hidden_size) (Gemma3)
    pub scale_embeddings: bool,
    /// Layer types for sliding window attention (Gemma3)
    pub layer_types: Vec<String>,
    /// RoPE theta for local attention layers (Gemma3)
    pub rope_local_base_freq: Option<f32>,
}

/// RoPE scaling configuration (Llama 3)
#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub factor: f32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // Llama 2 7B configuration
        let vocab_size = 32000;
        Self {
            version: 2,
            num_attention_heads: 32,
            num_key_value_heads: 32, // No GQA in base 7B
            hidden_size: 4096,
            head_dim: 128,
            intermediate_size: 11008,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            vocab_size,
            original_vocab_size: vocab_size,  // Same as vocab_size initially
            num_hidden_layers: 32,
            rope_theta: 10000.0,
            rope_scaling: None,
            hidden_activation: "silu".to_string(),
            query_pre_attn_scalar: None,
            use_qk_norm: false,
            scale_embeddings: false,
            layer_types: vec![],
            rope_local_base_freq: None,
        }
    }
}

impl LlamaConfig {
    /// Create config for Llama 3 8B
    pub fn llama3_8b() -> Self {
        let vocab_size = 128256;
        Self {
            version: 3,
            num_attention_heads: 32,
            num_key_value_heads: 8, // GQA with 8 KV heads
            hidden_size: 4096,
            head_dim: 128,
            intermediate_size: 14336,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            vocab_size,
            original_vocab_size: vocab_size,
            num_hidden_layers: 32,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                factor: 8.0,
            }),
            hidden_activation: "silu".to_string(),
            query_pre_attn_scalar: None,
            use_qk_norm: false,
            scale_embeddings: false,
            layer_types: vec![],
            rope_local_base_freq: None,
        }
    }

    /// Create config for Llama 3 70B
    pub fn llama3_70b() -> Self {
        let vocab_size = 128256;
        Self {
            version: 3,
            num_attention_heads: 64,
            num_key_value_heads: 8, // GQA with 8 KV heads
            hidden_size: 8192,
            head_dim: 128,
            intermediate_size: 28672,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            vocab_size,
            original_vocab_size: vocab_size,
            num_hidden_layers: 80,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                factor: 8.0,
            }),
            hidden_activation: "silu".to_string(),
            query_pre_attn_scalar: None,
            use_qk_norm: false,
            scale_embeddings: false,
            layer_types: vec![],
            rope_local_base_freq: None,
        }
    }
}

impl ArchitectureConfig for LlamaConfig {
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn rope_theta(&self) -> Option<f32> {
        Some(self.rope_theta)
    }

    fn rope_dim(&self) -> Option<usize> {
        Some(self.head_dim)
    }

    fn layer_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    fn use_rms_norm(&self) -> bool {
        true // All Llama versions use RMSNorm
    }
}

/// Llama model implementation
pub struct LlamaModel {
    config: LlamaConfig,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    dtype: DType,

    // Model weights
    embed_tokens: Option<Tensor>,
    layers: Vec<LlamaLayer>,
    norm: Option<RMSNorm>,
    lm_head: Option<Tensor>,

    /// Pre-transposed lm_head for tied weights (avoids transpose per forward pass)
    /// This is set when lm_head is None but embed_tokens is Some
    lm_head_transposed: Option<Tensor>,

    // KV cache for efficient generation (thread-safe with Mutex)
    kv_cache: Option<std::sync::Arc<std::sync::Mutex<crate::runtime::kv_cache::KVCacheManager>>>,

    // RoPE manager for position encoding (unused, RoPE handled at layer level)
    #[allow(dead_code)]
    rope_manager: std::sync::Arc<std::sync::Mutex<RoPEManager>>,

    // VarStore for training (if model was created with training support)
    vs: Option<nn::VarStore>,
}

// SAFETY: Tch tensors are thread-safe when used correctly
// We ensure no mutable access without proper synchronization
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

/// Single Llama transformer layer
struct LlamaLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

unsafe impl Send for LlamaLayer {}
unsafe impl Sync for LlamaLayer {}

/// Llama attention with optional GQA support
struct LlamaAttention {
    q_proj: LinearProjection,
    k_proj: LinearProjection,
    v_proj: LinearProjection,
    o_proj: LinearProjection,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    #[allow(dead_code)]
    rope_scaling: Option<RopeScaling>,
    // QK-norm weights (Gemma3)
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    #[allow(dead_code)]
    query_pre_attn_scalar: Option<f32>,
    // Sliding window attention (Gemma3)
    sliding_window: Option<usize>,
    layer_type: String, // "local" or "global" for Gemma3
    layer_idx: usize,
}

unsafe impl Send for LlamaAttention {}
unsafe impl Sync for LlamaAttention {}

impl LlamaAttention {
    /// Apply attention with optional GQA and KV caching
    fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: Option<&Tensor>,
        kv_cache: Option<&mut crate::runtime::kv_cache::LayerKVCache>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = dims3(hidden_states)?;

        tracing::trace!("LlamaAttention forward: batch_size={}, seq_len={}, hidden_size={}, start_pos={}, has_cache={}",
                      batch_size, seq_len, hidden_size, start_pos, kv_cache.is_some());

        // Debug: Check shapes and dtypes before projection
        // tracing::debug!("Attention forward - hidden_states shape: {:?}, dtype: {:?}", hidden_states.size(), hidden_states.kind());

        // Reshape hidden_states for matmul: [batch*seq, hidden_size]
        let hidden_states_2d = hidden_states.reshape([batch_size * seq_len, hidden_size]);
        // tracing::debug!("Reshaped hidden_states_2d shape: {:?}, dtype: {:?}", hidden_states_2d.size(), hidden_states_2d.kind());

        // Project to Q, K, V using the projection API (handles biases automatically)
        let q = self.q_proj.apply(&hidden_states_2d);
        let k = self.k_proj.apply(&hidden_states_2d);
        let v = self.v_proj.apply(&hidden_states_2d);

        // tracing::debug!("After projection - Q shape: {:?}, K shape: {:?}, V shape: {:?}",
        //               q.size(), k.size(), v.size());
        // tracing::debug!("Expected reshape: [{}, {}, {}, {}] = {} elements, actual Q elements: {}",
        //               batch_size, seq_len, self.num_heads, self.head_dim,
        //               batch_size * seq_len * self.num_heads * self.head_dim,
        //               q.size().iter().product::<usize>());

        // Determine actual K/V heads from tensor dimensions
        let k_elements = k.size().iter().product::<i64>();
        let _v_elements = v.size().iter().product::<i64>();

        // For K and V, we need to figure out the actual number of heads
        // They might be different from what we detected in config
        let kv_heads = if k_elements
            == batch_size * seq_len * (self.num_kv_heads as i64) * (self.head_dim as i64)
        {
            self.num_kv_heads // Config is correct
        } else if k_elements % (batch_size * seq_len * (self.head_dim as i64)) == 0 {
            // Recalculate based on actual size
            (k_elements / (batch_size * seq_len * (self.head_dim as i64))) as usize
        } else if k_elements % (batch_size * seq_len) == 0 {
            // Try different head_dim
            let kv_dim = (k_elements / (batch_size * seq_len)) as usize;
            if kv_dim == 256 && self.head_dim == 128 {
                2 // 2 heads with 128 dim
            } else if kv_dim == 256 {
                8 // 8 heads with 32 dim
            } else {
                self.num_kv_heads // Fallback to config
            }
        } else {
            self.num_kv_heads // Fallback
        };

        // tracing::debug!("Actual KV heads: {}, config KV heads: {}", kv_heads, self.num_kv_heads);

        // Reshape for attention
        // Q: [batch, seq, num_heads, head_dim]
        let mut q = q.reshape([
            batch_size,
            seq_len,
            self.num_heads as i64,
            self.head_dim as i64,
        ]);
        // K, V: [batch, seq, num_kv_heads, head_dim] - use actual kv_heads
        let mut k = k.reshape([batch_size, seq_len, kv_heads as i64, self.head_dim as i64]);
        let v = v.reshape([batch_size, seq_len, kv_heads as i64, self.head_dim as i64]);

        // Apply QK-norm if configured (Gemma3)
        if let Some(q_norm) = &self.q_norm {
            q = self.apply_qk_norm(&q, q_norm, self.num_heads)?;
        }
        if let Some(k_norm) = &self.k_norm {
            k = self.apply_qk_norm(&k, k_norm, kv_heads)?;
        }

        // Apply RoPE if position_ids provided
        let (q, k) = if let Some(pos_ids) = position_ids {
            (self.apply_rope(&q, pos_ids)?, self.apply_rope(&k, pos_ids)?)
        } else {
            (q, k)
        };

        // Handle KV caching
        let (k_for_attn, v_for_attn) = if let Some(cache) = kv_cache {
            // Update cache with new K and V
            cache.update(&k, &v, start_pos)?;

            // Get full cached K and V for attention computation
            let (cached_k, cached_v) = cache.get()?;
            (cached_k, cached_v)
        } else {
            // No caching, use current K and V directly
            (k.shallow_clone(), v.shallow_clone())
        };

        // Expand K, V for GQA if needed (use cached versions)
        let (k_expanded, v_expanded) = if kv_heads < self.num_heads {
            // tracing::debug!("Expanding KV from {} heads to {} heads", kv_heads, self.num_heads);
            (
                self.expand_kv_for_gqa_with_heads(&k_for_attn, kv_heads)?,
                self.expand_kv_for_gqa_with_heads(&v_for_attn, kv_heads)?,
            )
        } else {
            (k_for_attn, v_for_attn)
        };

        // Manual attention implementation
        // Note: SDPA was tested but:
        // - efficient_attention panics on this ROCm/libtorch version
        // - math_attention is slower than manual (~12 tok/sec vs ~70 tok/sec prefill)
        // See plan file for detailed analysis and future work.

        // Compute attention scores
        let scores = self.compute_attention_scores(&q, &k_expanded)?;

        // V: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        // PERF: .contiguous() required for optimal batched matmul on ROCm/AMD
        let v = v_expanded.transpose(1, 2).contiguous();

        // Apply attention to values: [batch, heads, seq, seq] x [batch, heads, seq, dim] = [batch, heads, seq, dim]
        let attn_output = scores.matmul(&v);

        // Transpose back: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
        // PERF: .contiguous() before reshape ensures optimal memory access
        let attn_output = attn_output.transpose(1, 2).contiguous();

        // Reshape to combine heads: [batch, seq, heads*dim]
        let attn_output =
            attn_output.reshape([batch_size, seq_len, (self.num_heads * self.head_dim) as i64]);

        // Reshape for output projection: [batch*seq, heads*dim]
        let attn_output_2d = attn_output.reshape([
            batch_size * seq_len,
            (self.num_heads * self.head_dim) as i64,
        ]);

        // Apply output projection using the projection API
        let attn_output = self.o_proj.apply(&attn_output_2d);

        // Reshape back to 3D: [batch, seq, hidden_size]
        let attn_output = attn_output.reshape([batch_size, seq_len, hidden_size]);

        Ok(attn_output)
    }

    /// Expand KV tensors for GQA with explicit head count
    fn expand_kv_for_gqa_with_heads(&self, kv: &Tensor, actual_kv_heads: usize) -> Result<Tensor> {
        let (batch_size, seq_len, _detected_heads, head_dim) = dims4(kv)?;
        let repeat_factor = self.num_heads / actual_kv_heads;

        if repeat_factor == 1 {
            return Ok(kv.shallow_clone());
        }

        // tracing::debug!("GQA expansion: {} KV heads -> {} Q heads (repeat {}x)",
        //               actual_kv_heads, self.num_heads, repeat_factor);

        // Expand by repeating KV heads
        Ok(kv
            .unsqueeze(3) // [batch, seq, num_kv_heads, 1, head_dim]
            .expand(
                [
                    batch_size,
                    seq_len,
                    actual_kv_heads as i64,
                    repeat_factor as i64,
                    head_dim,
                ],
                false,
            )
            .reshape([batch_size, seq_len, self.num_heads as i64, head_dim]))
    }

    /// Apply Rotary Position Embeddings with optional scaling
    fn apply_rope(&self, tensor: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Use a thread-local RoPE instance cache for efficiency
        thread_local! {
            static ROPE_CACHE: std::cell::RefCell<std::collections::HashMap<(usize, i64, u32), RoPE>> =
                std::cell::RefCell::new(std::collections::HashMap::new());
        }

        ROPE_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            // Create a unique key for this layer's RoPE configuration
            let key = (
                self.layer_idx,
                self.head_dim as i64,
                (self.rope_theta * 1000.0) as u32, // Convert to integer for hashing
            );

            // Get or create RoPE instance
            if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(key) {
                // Use appropriate base frequency for layer type
                let base = if self.layer_type == "local" {
                    // Local layers in Gemma3 use different base
                    10000.0
                } else {
                    self.rope_theta as f64
                };

                tracing::debug!(
                    "Creating RoPE for layer {} with base={}, head_dim={}",
                    self.layer_idx,
                    base,
                    self.head_dim
                );

                // Create RoPE with the same dtype as the input tensor
                let rope = RoPE::new_with_dtype(
                    self.head_dim as i64,
                    base,
                    8192, // max_seq_len - should come from config
                    tensor.device(),
                    tensor.kind(), // Use same dtype as input tensor
                )
                .map_err(|e| {
                    anyhow!("Failed to create RoPE for layer {}: {}", self.layer_idx, e)
                })?;

                e.insert(rope);
            }

            let rope = cache.get_mut(&key).expect("rope was just inserted");

            // Apply RoPE with the actual position_ids
            rope.forward(tensor, Some(position_ids))
        })
    }

    /// Apply QK-norm (Gemma3)
    fn apply_qk_norm(
        &self,
        tensor: &Tensor,
        norm_weight: &Tensor,
        actual_heads: usize,
    ) -> Result<Tensor> {
        // QK-norm normalizes each head separately
        // tensor shape: [batch, seq, heads, dim]
        // norm_weight shape: [heads * dim] where heads could be Q heads or KV heads
        let (_batch_size, _seq_len, tensor_heads, head_dim) = dims4(tensor)?;

        // Debug logging
        tracing::trace!(
            "apply_qk_norm: tensor heads={}, head_dim={}, actual_heads={}, norm_weight shape={:?}",
            tensor_heads,
            head_dim,
            actual_heads,
            norm_weight.size()
        );

        // First, reshape the norm weights to match [1, 1, heads, dim]
        // The norm weights are typically stored as a flat vector [heads * dim]
        let norm_weight_reshaped = if norm_weight.size().len() == 1 {
            let norm_elements = norm_weight.size()[0];
            // For Gemma3, norm weights are per-head
            // If norm has 256 elements and we have 1 KV head with 256 dim, reshape to [1, 256]
            // If norm has 1024 elements and we have 4 Q heads with 256 dim, reshape to [4, 256]
            if norm_elements == (actual_heads * head_dim as usize) as i64 {
                // Standard case: reshape from [heads * dim] to [1, 1, heads, dim]
                norm_weight
                    .reshape([actual_heads as i64, head_dim])
                    .unsqueeze(0) // Add batch dimension
                    .unsqueeze(0) // Add seq dimension
            } else if norm_elements == head_dim {
                // Special case: norm is per-dimension only (for single head)
                norm_weight
                    .unsqueeze(0) // Add head dimension
                    .unsqueeze(0) // Add batch dimension
                    .unsqueeze(0) // Add seq dimension
            } else {
                return Err(anyhow!(
                    "QK-norm weight size {} doesn't match expected size for {} heads with dim {}",
                    norm_elements,
                    actual_heads,
                    head_dim
                ));
            }
        } else {
            norm_weight.shallow_clone()
        };

        // Apply RMSNorm per head, preserving dtype
        let original_dtype = tensor.kind();
        let x2 = square_tensor(tensor)?;
        let mean = x2.mean_dim(&[-1i64][..], true, original_dtype);
        let eps = 1e-6;
        let rrms = (mean + eps).reciprocal().sqrt();

        // Apply normalization and scale with norm weights
        broadcast_mul(
            &broadcast_mul(tensor, &rrms)?,
            &norm_weight_reshaped,
        )
    }

    /// Compute scaled dot-product attention scores with causal masking
    fn compute_attention_scores(&self, q: &Tensor, k: &Tensor) -> Result<Tensor> {
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Q: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        // PERF: .contiguous() required for optimal batched matmul on ROCm/AMD
        let q = q.transpose(1, 2).contiguous();
        // K: [batch, seq, heads, dim] -> [batch, heads, seq, dim] -> [batch, heads, dim, seq]
        let k = k.transpose(1, 2).transpose(2, 3).contiguous();

        // Compute attention scores: [batch, heads, seq, seq]
        let mut scores = q.matmul(&k) * (scale as f64);

        // Apply causal mask - CRITICAL for autoregressive generation
        // Get sequence lengths from tensor dimensions
        let score_shape = scores.size();
        let q_len = score_shape[2]; // query sequence length
        let k_len = score_shape[3]; // key sequence length

        // Apply causal mask only when processing multiple query tokens (prompt phase)
        // For autoregressive generation (q_len=1), all past positions are valid - no mask needed
        if q_len > 1 {
            // Standard causal mask for multiple queries (e.g., processing prompt)
            // Lower triangular: 1s at and below diagonal, 0s above
            let mask = Tensor::ones([q_len, k_len], (scores.kind(), scores.device())).tril(0);

            // Expand mask to match scores dimensions [batch, heads, q_len, k_len]
            let mask = mask.unsqueeze(0).unsqueeze(0).expand_as(&scores);

            // Apply mask: set future positions (where mask=0) to -inf
            let mask_value = -10000.0f64;
            scores = scores.masked_fill(&mask.eq(0.0), mask_value);
        }
        // When q_len=1 (KV cache autoregressive), skip mask - all past positions valid

        // Apply sliding window mask if configured (Gemma3)
        if let Some(window_size) = self.sliding_window {
            if self.layer_type == "local" {
                scores = self.apply_sliding_window_mask(&scores, window_size)?;
            }
            // Global layers use full attention (no mask)
        }

        // Apply softmax along last dimension, preserving dtype
        Ok(scores.softmax(-1, scores.kind()))
    }

    /// Apply sliding window mask for local attention layers
    /// Optimized to use GPU-native ops instead of O(n²) CPU loop
    fn apply_sliding_window_mask(&self, scores: &Tensor, window_size: usize) -> Result<Tensor> {
        let (batch_size, num_heads, seq_len, _) = dims4(scores)?;
        let device = scores.device();
        let dtype = scores.kind();

        // Create sliding window mask using GPU-native tril/triu operations
        // Valid positions: j <= i (causal) AND i - j < window_size (in window)

        // Causal mask: lower triangular (j <= i)
        let causal = Tensor::ones([seq_len, seq_len], (dtype, device)).tril(0);

        // Window mask: upper triangular starting from -(window_size-1)
        // This keeps positions where j >= i - window_size + 1
        let window = Tensor::ones([seq_len, seq_len], (dtype, device))
            .triu(1 - window_size as i64);

        // Valid = causal AND window (element-wise multiplication of 0/1 masks)
        let valid_mask = &causal * &window;

        // Convert to additive mask: 0 where valid, -10000 where invalid
        // invalid = 1 - valid, then scale by -10000
        let additive_mask = (Tensor::ones([seq_len, seq_len], (dtype, device)) - valid_mask)
            * (-10000.0f64);

        // Broadcast to [batch, heads, seq, seq]
        let mask = additive_mask
            .unsqueeze(0)
            .unsqueeze(0)
            .expand([batch_size, num_heads, seq_len, seq_len], false);

        // Add mask to scores
        broadcast_add(scores, &mask)
    }
}

/// Llama MLP/FFN layer
struct LlamaMLP {
    gate_proj: LinearProjection,
    up_proj: LinearProjection,
    down_proj: LinearProjection,
    activation: String, // Activation function name
}

unsafe impl Send for LlamaMLP {}
unsafe impl Sync for LlamaMLP {}

impl LlamaMLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Get dimensions
        let original_shape = hidden_states.size();
        let (batch_size, seq_len, hidden_size) = if original_shape.len() == 3 {
            (original_shape[0], original_shape[1], original_shape[2])
        } else {
            // Already 2D
            return self.forward_2d(hidden_states);
        };

        // Reshape to 2D for matmul
        let hidden_2d = hidden_states.reshape([batch_size * seq_len, hidden_size]);

        // Apply SwiGLU: down(act(gate(x)) * up(x))
        let gate_pre = self.gate_proj.apply(&hidden_2d);
        let gate = self.apply_activation(&gate_pre)?;
        let up = self.up_proj.apply(&hidden_2d);
        let gated = &gate * &up;
        let output = self.down_proj.apply(&gated);

        // Reshape back to 3D
        let out_size = output.size()[1];
        Ok(output.reshape([batch_size, seq_len, out_size]))
    }

    fn forward_2d(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Apply SwiGLU: down(act(gate(x)) * up(x))
        let gate_pre = self.gate_proj.apply(hidden_states);
        let gate = self.apply_activation(&gate_pre)?;
        let up = self.up_proj.apply(hidden_states);
        Ok(self.down_proj.apply(&(&gate * &up)))
    }

    /// Apply the configured activation function
    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "silu" => Ok(x.silu()),
            "gelu_pytorch_tanh" => self.gelu_pytorch_tanh(x),
            "gelu" => {
                // Standard GELU approximation
                // GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
                // Using tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                self.gelu_pytorch_tanh(x)
            }
            _ => {
                tracing::warn!(
                    "Unknown activation '{}', falling back to SiLU",
                    self.activation
                );
                Ok(x.silu())
            }
        }
    }

    /// GELU activation with PyTorch tanh approximation (for Gemma3)
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    fn gelu_pytorch_tanh(&self, x: &Tensor) -> Result<Tensor> {
        use std::f32::consts::PI;

        // Constants for the approximation
        let sqrt_2_over_pi = (2.0_f32 / PI).sqrt();
        let coeff = 0.044715_f32;

        // x^3
        let x_cubed = x.pow_tensor_scalar(3.0);

        // x + 0.044715 * x^3
        let coeff_tensor = scalar_tensor(coeff, x.device(), x.kind());
        let inner = x + broadcast_mul(&x_cubed, &coeff_tensor)?;

        // sqrt(2/π) * (x + 0.044715 * x^3)
        let sqrt_tensor = scalar_tensor(sqrt_2_over_pi, x.device(), x.kind());
        let scaled = broadcast_mul(&inner, &sqrt_tensor)?;

        // tanh(sqrt(2/π) * (x + 0.044715 * x^3))
        let tanh_result = scaled.tanh();

        // 1 + tanh(...)
        let one_tensor = scalar_tensor(1.0_f32, x.device(), x.kind());
        let one_plus_tanh = broadcast_add(&tanh_result, &one_tensor)?;

        // 0.5 * x * (1 + tanh(...))
        let half_tensor = scalar_tensor(0.5_f32, x.device(), x.kind());
        broadcast_mul(&(x * &one_plus_tanh), &half_tensor)
    }
}

/// RMSNorm implementation for Llama
struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

unsafe impl Send for RMSNorm {}
unsafe impl Sync for RMSNorm {}

impl RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute RMS, preserving the original dtype
        let original_dtype = x.kind();
        let x2 = square_tensor(x)?;
        let mean = x2.mean_dim(&[-1i64][..], true, original_dtype); // Keep original dtype
        let rrms = (mean + self.eps as f64).reciprocal().sqrt();

        // Apply normalization and scaling
        broadcast_mul(&broadcast_mul(x, &rrms)?, &self.weight)
    }
}

impl LlamaModel {
    /// Create Llama model from SafeTensors
    pub fn from_safetensors(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        // Load configuration
        let config_path = path
            .parent()
            .ok_or_else(|| anyhow!("Invalid model path"))?
            .join("config.json");

        let config = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            Self::parse_config(&config_str)?
        } else {
            LlamaConfig::default()
        };

        // Load weights from SafeTensors (simplified)
        let layers = Vec::new();

        Ok(Self {
            config,
            device: *device,
            dtype,
            embed_tokens: None,
            layers,
            norm: None,
            lm_head: None,
            lm_head_transposed: None,
            kv_cache: None, // Cache initialized on first use
            rope_manager: std::sync::Arc::new(std::sync::Mutex::new(RoPEManager::new())),
            vs: None, // No VarStore for safetensors loading
        })
    }

    /// Create Llama model from pre-loaded weight tensors
    pub fn from_weights(
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Parse config from weights if possible, otherwise use defaults
        let config = Self::detect_config_from_weights(weights)?;
        Self::from_weights_with_config(weights, config, device, dtype, crate::runtime::kv_quant::KVQuantType::None)
    }

    /// Create Llama model with explicit config (allows Qwen models to override)
    /// Build model from weights, taking mutable reference to free tensors as they're processed.
    /// This reduces peak memory by ~50% by removing original weight tensors after transposing.
    pub fn from_weights_with_config(
        weights: &mut HashMap<String, Tensor>,
        mut config: LlamaConfig,
        device: &Device,
        dtype: DType,
        kv_quant_type: crate::runtime::kv_quant::KVQuantType,
    ) -> Result<Self> {
        tracing::info!(
            "[from_weights_with_config] Received config.max_position_embeddings = {}, kv_quant = {:?}",
            config.max_position_embeddings,
            kv_quant_type
        );

        // Extract key tensors (with padding for models that need it)
        let embed_tokens = weights
            .get("model.embed_tokens.weight")
            .or_else(|| weights.get("embed_tokens.weight"))
            .map(|w| {
                let vocab_size = w.size()[0] as usize;
                let hidden_size = w.size()[1];

                // Pad vocabulary size to multiple of 64 for models that need it
                let padded_vocab_size = calculate_padded_vocab_size(vocab_size);

                if padded_vocab_size > vocab_size {
                    tracing::info!(
                        "Padding embedding vocab size from {} to {} (multiple of 64)",
                        vocab_size, padded_vocab_size
                    );

                    // Create padded tensor filled with zeros (embeddings can be zero-initialized)
                    let padded = Tensor::zeros(
                        [padded_vocab_size as i64, hidden_size],
                        (w.kind(), w.device())
                    );

                    // Copy original weights to the padded tensor
                    padded.narrow(0, 0, vocab_size as i64).copy_(w);
                    padded
                } else {
                    w.shallow_clone()
                }
            });

        // Update config with padded vocab size if needed
        let original_vocab_size = config.vocab_size;
        let padded_vocab_size = calculate_padded_vocab_size(original_vocab_size);

        // Store original vocab size before padding
        config.original_vocab_size = original_vocab_size;

        if padded_vocab_size != original_vocab_size {
            tracing::info!(
                "Updating config vocab_size from {} to padded size {}",
                original_vocab_size, padded_vocab_size
            );
            config.vocab_size = padded_vocab_size;
        }

        // Handle lm_head - Gemma models use weight tying (lm_head shares weights with embed_tokens)
        // Try to find explicit lm_head first, otherwise we'll use tied weights from embeddings
        let lm_head = weights
            .get("lm_head.weight")
            .or_else(|| weights.get("model.lm_head.weight"))
            .map(|w| {
                // LM head is stored as [vocab_size, hidden_size] in HuggingFace
                let vocab_size = w.size()[0] as usize;
                let hidden_size = w.size()[1];

                // Pad vocabulary size to multiple of 64 (like SGLang does for Qwen)
                // This prevents sampling invalid token IDs
                let padded_vocab_size = calculate_padded_vocab_size(vocab_size);

                let padded_w = if padded_vocab_size > vocab_size {
                    tracing::info!(
                        "Padding lm_head vocab size from {} to {} (multiple of 64)",
                        vocab_size, padded_vocab_size
                    );

                    // Create padded tensor filled with zeros
                    // NOTE: We can't use -1e10 here because this is the weight matrix, not logits
                    // The actual masking needs to happen after the matmul that produces logits
                    let padded = Tensor::zeros(
                        [padded_vocab_size as i64, hidden_size],
                        (w.kind(), w.device())
                    );

                    // Copy original weights to the padded tensor
                    padded.narrow(0, 0, vocab_size as i64).copy_(w);
                    padded
                } else {
                    w.shallow_clone()
                };

                // We need [hidden_size, vocab_size] for matmul
                padded_w.transpose(0, 1).contiguous()
            });

        // Pre-compute transposed lm_head for tied weights case
        // This avoids allocating a new tensor on every forward pass
        let lm_head_transposed = if lm_head.is_none() {
            embed_tokens.as_ref().map(|embed| {
                tracing::info!(
                    "Pre-computing lm_head transpose for tied weights (saves ~740MB per forward)"
                );
                embed.transpose(0, 1).contiguous()
            })
        } else {
            None
        };

        // Extract final layer norm
        let norm = weights
            .get("model.norm.weight")
            .or_else(|| weights.get("norm.weight"))
            .map(|w| RMSNorm {
                weight: w.shallow_clone(),
                eps: config.rms_norm_eps,
            });

        // Build transformer layers incrementally, freeing weight tensors as we go
        // This reduces peak memory by ~50% (7.7GB savings for Qwen3-4B)
        let mut layers = Vec::new();
        for layer_idx in 0..config.num_hidden_layers {
            if let Some(layer) = Self::build_layer(layer_idx, weights, &config, device)? {
                layers.push(layer);
            }
        }

        // Initialize KV cache if we have layers
        let kv_cache = if !layers.is_empty() {
            tracing::info!(
                "[LlamaModel] Creating KV cache: num_layers={}, max_seq_len={}, quant={:?}",
                layers.len(),
                config.max_position_embeddings,
                kv_quant_type
            );
            Some(std::sync::Arc::new(std::sync::Mutex::new(
                crate::runtime::kv_cache::KVCacheManager::new(
                    layers.len(),
                    config.max_position_embeddings,
                    kv_quant_type,
                ),
            )))
        } else {
            None
        };

        Ok(Self {
            config,
            device: *device,
            dtype,
            embed_tokens,
            layers,
            norm,
            lm_head,
            lm_head_transposed,
            kv_cache,
            rope_manager: std::sync::Arc::new(std::sync::Mutex::new(RoPEManager::new())),
            vs: None, // No VarStore for weight loading - weights are stored directly
        })
    }

    /// Detect configuration from weight tensor shapes
    pub fn detect_config_from_weights(weights: &HashMap<String, Tensor>) -> Result<LlamaConfig> {
        // Try to infer config from tensor shapes
        let mut config = LlamaConfig::default();
        tracing::info!(
            "Starting detect_config_from_weights, default rope_theta={}",
            config.rope_theta
        );

        // Get vocab size from embedding
        if let Some(embed) = weights
            .get("model.embed_tokens.weight")
            .or_else(|| weights.get("embed_tokens.weight"))
        {
            let shape = embed.size();
            if shape.len() >= 2 {
                config.vocab_size = shape[0] as usize;
                config.original_vocab_size = shape[0] as usize;  // Initially same
                config.hidden_size = shape[1] as usize;

                // Detect Gemma3 by vocab size (262144) and set specific parameters
                if config.vocab_size == 262144 {
                    config.hidden_activation = "gelu_pytorch_tanh".to_string();
                    config.use_qk_norm = true;
                    config.scale_embeddings = true;
                    config.query_pre_attn_scalar = Some(256.0); // Common value for Gemma3
                    config.rope_local_base_freq = Some(10000.0);
                    config.rope_theta = 1000000.0; // Global attention theta for Gemma3
                    tracing::info!("Detected Gemma3 model (vocab_size=262144), configuring with:");
                    tracing::info!("  - GELU PyTorch tanh activation");
                    tracing::info!("  - QK-norm enabled");
                    tracing::info!("  - Embedding scaling enabled");
                    tracing::info!("  - RoPE theta: global=1000000, local=10000");
                }
            }
        }

        // Count layers
        let layer_count = weights
            .keys()
            .filter(|k| k.contains("layers.") && k.contains(".self_attn.q_proj"))
            .count();
        if layer_count > 0 {
            config.num_hidden_layers = layer_count;
        }

        // Get attention heads from q_proj and k_proj shapes
        if let Some(q_proj) = weights.get("model.layers.0.self_attn.q_proj.weight") {
            let q_shape = q_proj.size();
            if q_shape.len() >= 2 {
                // shape[0] is output dim (num_heads * head_dim)
                // shape[1] is hidden_size
                config.hidden_size = q_shape[1] as usize;
                let q_proj_out_dim = q_shape[0] as usize;

                // Also check k_proj to detect GQA (Grouped Query Attention)
                let k_proj_out_dim =
                    if let Some(k_proj) = weights.get("model.layers.0.self_attn.k_proj.weight") {
                        k_proj.size()[0] as usize
                    } else {
                        q_proj_out_dim // Assume same as Q if K not found
                    };

                tracing::debug!(
                    "Q projection output: {}, K projection output: {}",
                    q_proj_out_dim,
                    k_proj_out_dim
                );

                // Try common head_dim values to find the right configuration
                // Gemma3 uses head_dim=256, so check that first for Gemma models
                let possible_head_dims = if config.vocab_size == 262144 {
                    // Gemma3 detected - prioritize 256 head dim
                    vec![256, 128, 64, 32]
                } else if q_proj_out_dim <= 512 {
                    vec![32, 64, 48, 40, 128] // Smaller models often use smaller head_dim
                } else {
                    vec![128, 64, 80, 96, 160, 256]
                };

                let mut found_config = false;

                for &head_dim in &possible_head_dims {
                    if q_proj_out_dim.is_multiple_of(head_dim) && k_proj_out_dim % head_dim == 0 {
                        config.num_attention_heads = q_proj_out_dim / head_dim;
                        config.num_key_value_heads = k_proj_out_dim / head_dim;
                        config.head_dim = head_dim;
                        found_config = true;
                        tracing::info!("Detected attention config: Q={} heads, KV={} heads, dim={}, total Q={}", 
                                     config.num_attention_heads, config.num_key_value_heads, 
                                     config.head_dim, q_proj_out_dim);
                        break;
                    }
                }

                if !found_config {
                    // Fallback: try to guess based on common patterns
                    // Many small models use head_dim=32 or 64
                    if q_proj_out_dim == 256 && k_proj_out_dim == 256 {
                        // Likely 8 heads with dim 32
                        config.num_attention_heads = 8;
                        config.num_key_value_heads = 8;
                        config.head_dim = 32;
                        tracing::info!("Using common small model config: 8 heads x 32 dim");
                    } else {
                        // Last resort defaults
                        config.num_attention_heads = 8;
                        config.num_key_value_heads = 8;
                        config.head_dim = 64;
                        tracing::warn!("Could not detect attention config from shapes Q:{:?} K:{:?}, using defaults", 
                                     q_shape, k_proj_out_dim);
                    }
                }
            }
        }

        tracing::info!(
            "Finished detect_config_from_weights, final rope_theta={}, vocab_size={}",
            config.rope_theta,
            config.vocab_size
        );
        Ok(config)
    }

    /// Build a single transformer layer from weights
    /// Takes mutable reference to remove processed weights, reducing peak memory
    fn build_layer(
        layer_idx: usize,
        weights: &mut HashMap<String, Tensor>,
        config: &LlamaConfig,
        _device: &Device,
    ) -> Result<Option<LlamaLayer>> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Check if this layer exists (handle both separate and combined projections)
        let has_separate_qkv = weights.contains_key(&format!("{}.self_attn.q_proj.weight", prefix));
        let has_combined_qkv = weights.contains_key(&format!("{}.self_attn.c_attn.weight", prefix));

        if !has_separate_qkv && !has_combined_qkv {
            return Ok(None);
        }

        // Build attention
        let (q_proj, k_proj, v_proj) = if has_separate_qkv {
            // Standard separate Q, K, V projections (Llama/Qwen style)

            // Load Q projection - remove from HashMap to free original tensor after transpose
            let q_weight = weights
                .remove(&format!("{}.self_attn.q_proj.weight", prefix))
                .ok_or_else(|| anyhow!("Missing q_proj weight"))?
                .transpose(0, 1)
                .contiguous();

            let q_proj = if let Some(q_bias) = weights.get(&format!("{}.self_attn.q_proj.bias", prefix)) {
                tracing::debug!("Layer {}: Loading Q bias (Qwen-style)", layer_idx);
                LinearProjection::with_bias(q_weight, q_bias.shallow_clone())
            } else {
                LinearProjection::new(q_weight)
            };

            // Load K projection - remove from HashMap to free original tensor after transpose
            let k_weight = weights
                .remove(&format!("{}.self_attn.k_proj.weight", prefix))
                .ok_or_else(|| anyhow!("Missing k_proj weight"))?
                .transpose(0, 1)
                .contiguous();

            let k_proj = if let Some(k_bias) = weights.get(&format!("{}.self_attn.k_proj.bias", prefix)) {
                tracing::debug!("Layer {}: Loading K bias (Qwen-style)", layer_idx);
                LinearProjection::with_bias(k_weight, k_bias.shallow_clone())
            } else {
                LinearProjection::new(k_weight)
            };

            // Load V projection - remove from HashMap to free original tensor after transpose
            let v_weight = weights
                .remove(&format!("{}.self_attn.v_proj.weight", prefix))
                .ok_or_else(|| anyhow!("Missing v_proj weight"))?
                .transpose(0, 1)
                .contiguous();

            let v_proj = if let Some(v_bias) = weights.get(&format!("{}.self_attn.v_proj.bias", prefix)) {
                tracing::debug!("Layer {}: Loading V bias (Qwen-style)", layer_idx);
                LinearProjection::with_bias(v_weight, v_bias.shallow_clone())
            } else {
                LinearProjection::new(v_weight)
            };

            (q_proj, k_proj, v_proj)
        } else {
            // Combined QKV projection (some Qwen models use c_attn)
            // Remove from HashMap to free original tensor after transpose
            let c_attn_weight = weights
                .remove(&format!("{}.self_attn.c_attn.weight", prefix))
                .ok_or_else(|| anyhow!("Missing c_attn weight"))?
                .transpose(0, 1) // Transpose from [out, in] to [in, out]
                .contiguous();

            // Check for combined bias
            let c_attn_bias = weights.get(&format!("{}.self_attn.c_attn.bias", prefix));

            // Split c_attn into Q, K, V
            // c_attn has shape [hidden_size, 3 * projection_size]
            let dims = c_attn_weight.size();
            let _hidden_size = dims[0];
            let total_proj_size = dims[1];
            let proj_size = total_proj_size / 3;

            let q_weight = c_attn_weight.narrow(1, 0, proj_size);
            let k_weight = c_attn_weight.narrow(1, proj_size, proj_size);
            let v_weight = c_attn_weight.narrow(1, proj_size * 2, proj_size);

            // Split bias if present
            let (q_proj, k_proj, v_proj) = if let Some(bias) = c_attn_bias {
                tracing::debug!("Layer {}: Loading combined QKV bias (Qwen-style)", layer_idx);
                let q_bias = bias.narrow(0, 0, proj_size);
                let k_bias = bias.narrow(0, proj_size, proj_size);
                let v_bias = bias.narrow(0, proj_size * 2, proj_size);

                (
                    LinearProjection::with_bias(q_weight, q_bias),
                    LinearProjection::with_bias(k_weight, k_bias),
                    LinearProjection::with_bias(v_weight, v_bias),
                )
            } else {
                (
                    LinearProjection::new(q_weight),
                    LinearProjection::new(k_weight),
                    LinearProjection::new(v_weight),
                )
            };

            (q_proj, k_proj, v_proj)
        };

        // Check for QK-norm weights (Gemma3)
        let q_norm = weights
            .get(&format!("{}.self_attn.q_norm.weight", prefix))
            .map(|t| t.shallow_clone());
        let k_norm = weights
            .get(&format!("{}.self_attn.k_norm.weight", prefix))
            .map(|t| t.shallow_clone());

        if q_norm.is_some() || k_norm.is_some() {
            tracing::debug!("Layer {} has QK-norm weights", layer_idx);
        }

        // Determine layer type for Gemma3 sliding window attention
        let layer_type = if !config.layer_types.is_empty() && layer_idx < config.layer_types.len() {
            config.layer_types[layer_idx].clone()
        } else if config.layer_types.is_empty() && config.use_qk_norm {
            // Gemma3 pattern: every 6th layer is global, others are local
            if (layer_idx + 1).is_multiple_of(6) {
                "global".to_string()
            } else {
                "local".to_string()
            }
        } else {
            "global".to_string() // Default to global attention
        };

        // Determine sliding window size for Gemma3
        let sliding_window = if config.use_qk_norm && layer_type == "local" {
            Some(512) // Gemma3 uses 512 token sliding window
        } else {
            None
        };

        // Use different RoPE theta for local vs global layers if configured
        let rope_theta = if layer_type == "local" {
            config.rope_local_base_freq.unwrap_or(config.rope_theta)
        } else {
            config.rope_theta
        };

        // Load output projection (typically no bias, but check anyway)
        // Remove from HashMap to free original tensor after transpose
        let o_weight = weights
            .remove(&format!("{}.self_attn.o_proj.weight", prefix))
            .or_else(|| weights.remove(&format!("{}.self_attn.c_proj.weight", prefix))) // Some models use c_proj for output
            .ok_or_else(|| anyhow!("Missing o_proj/c_proj weight"))?
            .transpose(0, 1) // Transpose from [out, in] to [in, out]
            .contiguous();

        let o_proj = if let Some(o_bias) = weights.get(&format!("{}.self_attn.o_proj.bias", prefix))
            .or_else(|| weights.get(&format!("{}.self_attn.c_proj.bias", prefix)))
        {
            tracing::debug!("Layer {}: Loading O bias", layer_idx);
            LinearProjection::with_bias(o_weight, o_bias.shallow_clone())
        } else {
            LinearProjection::new(o_weight)
        };

        let self_attn = LlamaAttention {
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope_theta,
            rope_scaling: config.rope_scaling.clone(),
            q_norm,
            k_norm,
            query_pre_attn_scalar: config.query_pre_attn_scalar,
            sliding_window,
            layer_type,
            layer_idx,
        };

        // Build MLP with optional biases
        // Remove from HashMap to free original tensors after transpose
        let gate_weight = weights
            .remove(&format!("{}.mlp.gate_proj.weight", prefix))
            .ok_or_else(|| anyhow!("Missing gate_proj weight"))?
            .transpose(0, 1) // Transpose from [out, in] to [in, out]
            .contiguous();

        let gate_proj = if let Some(gate_bias) = weights.get(&format!("{}.mlp.gate_proj.bias", prefix)) {
            tracing::debug!("Layer {}: Loading gate_proj bias", layer_idx);
            LinearProjection::with_bias(gate_weight, gate_bias.shallow_clone())
        } else {
            LinearProjection::new(gate_weight)
        };

        let up_weight = weights
            .remove(&format!("{}.mlp.up_proj.weight", prefix))
            .ok_or_else(|| anyhow!("Missing up_proj weight"))?
            .transpose(0, 1) // Transpose from [out, in] to [in, out]
            .contiguous();

        let up_proj = if let Some(up_bias) = weights.get(&format!("{}.mlp.up_proj.bias", prefix)) {
            tracing::debug!("Layer {}: Loading up_proj bias", layer_idx);
            LinearProjection::with_bias(up_weight, up_bias.shallow_clone())
        } else {
            LinearProjection::new(up_weight)
        };

        let down_weight = weights
            .remove(&format!("{}.mlp.down_proj.weight", prefix))
            .ok_or_else(|| anyhow!("Missing down_proj weight"))?
            .transpose(0, 1) // Transpose from [out, in] to [in, out]
            .contiguous();

        let down_proj = if let Some(down_bias) = weights.get(&format!("{}.mlp.down_proj.bias", prefix)) {
            tracing::debug!("Layer {}: Loading down_proj bias", layer_idx);
            LinearProjection::with_bias(down_weight, down_bias.shallow_clone())
        } else {
            LinearProjection::new(down_weight)
        };

        let mlp = LlamaMLP {
            gate_proj,
            up_proj,
            down_proj,
            activation: config.hidden_activation.clone(),
        };

        // Build layer norms
        let input_layernorm = RMSNorm {
            weight: weights
                .get(&format!("{}.input_layernorm.weight", prefix))
                .ok_or_else(|| anyhow!("Missing input_layernorm weight"))?
                .shallow_clone(),
            eps: config.rms_norm_eps,
        };

        let post_attention_layernorm = RMSNorm {
            weight: weights
                .get(&format!("{}.post_attention_layernorm.weight", prefix))
                .ok_or_else(|| anyhow!("Missing post_attention_layernorm weight"))?
                .shallow_clone(),
            eps: config.rms_norm_eps,
        };

        Ok(Some(LlamaLayer {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }))
    }

    /// Parse configuration from JSON
    pub fn parse_config(json_str: &str) -> Result<LlamaConfig> {
        let json: serde_json::Value = serde_json::from_str(json_str)?;

        // Detect version from config
        let version = if json.get("rope_scaling").is_some() {
            3
        } else if json["num_key_value_heads"] != json["num_attention_heads"] {
            2
        } else {
            1
        };

        let mut config = LlamaConfig {
            version,
            num_attention_heads: json["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_key_value_heads: json
                .get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| json["num_attention_heads"].as_u64().unwrap_or(32))
                as usize,
            hidden_size: json["hidden_size"].as_u64().unwrap_or(4096) as usize,
            head_dim: json
                .get("head_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or_else(|| {
                    // If head_dim not specified, calculate from hidden_size / num_attention_heads
                    let heads = json
                        .get("num_attention_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize;
                    let hidden = json
                        .get("hidden_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4096) as usize;
                    // Common head_dim values are 64, 128, 256
                    if hidden.is_multiple_of(heads * 256) {
                        256
                    } else if hidden.is_multiple_of(heads * 128) {
                        128
                    } else if hidden.is_multiple_of(heads * 64) {
                        64
                    } else {
                        128 // Default fallback
                    }
                }),
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(11008) as usize,
            max_position_embeddings: json["max_position_embeddings"].as_u64().unwrap_or(4096)
                as usize,
            rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as usize,
            original_vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as usize,  // Initially same
            num_hidden_layers: json["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            rope_theta: json
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0) as f32,
            rope_scaling: None,
            hidden_activation: json
                .get("hidden_activation")
                .and_then(|v| v.as_str())
                .unwrap_or("silu")
                .to_string(),
            query_pre_attn_scalar: json
                .get("query_pre_attn_scalar")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            use_qk_norm: false, // Will be set based on vocab size or explicit config
            scale_embeddings: false, // Will be set based on vocab size or explicit config
            layer_types: vec![],
            rope_local_base_freq: None,
        };

        // Parse rope scaling if present
        if let Some(rope_scaling) = json.get("rope_scaling") {
            config.rope_scaling = Some(RopeScaling {
                scaling_type: rope_scaling["type"]
                    .as_str()
                    .unwrap_or("linear")
                    .to_string(),
                factor: rope_scaling["factor"].as_f64().unwrap_or(8.0) as f32,
            });
        }

        // Check for Gemma3 specific configurations
        if config.vocab_size == 262144 {
            config.hidden_activation = "gelu_pytorch_tanh".to_string();
            config.use_qk_norm = true;
            config.scale_embeddings = true;
            if config.query_pre_attn_scalar.is_none() {
                config.query_pre_attn_scalar = Some(256.0);
            }
            config.rope_local_base_freq = Some(10000.0);
            config.rope_theta = 1000000.0; // Global attention theta
            tracing::info!("Detected Gemma3 model from config.json, applying Gemma3 settings");
        }

        tracing::info!(
            "Parsed Llama config from JSON: rope_theta={}, vocab_size={}",
            config.rope_theta,
            config.vocab_size
        );
        Ok(config)
    }
}

impl ModelOperations for LlamaModel {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Llama {
            version: self.config.version,
        }
    }

    fn config(&self) -> &dyn ArchitectureConfig {
        &self.config
    }

    fn forward(&self, input: &Tensor, _past_kv: Option<&Tensor>) -> Result<Tensor> {
        // Default forward just calls forward_with_cache with start_pos = 0
        self.forward_with_cache(input, 0)
    }

    fn var_store(&self) -> Option<&nn::VarStore> {
        self.vs.as_ref()
    }

    fn var_store_mut(&mut self) -> Option<&mut nn::VarStore> {
        self.vs.as_mut()
    }

    fn forward_with_cache(&self, input: &Tensor, start_pos: usize) -> Result<Tensor> {
        tracing::trace!("LlamaModel forward_with_cache: input shape={:?}, start_pos={}, config: hidden_size={}, num_layers={}",
                     input.size(), start_pos, self.config.hidden_size, self.layers.len());

        // Input should be token IDs with shape [batch_size, seq_len]
        let mut hidden_states = if let Some(embed) = &self.embed_tokens {
            // Convert token IDs to embeddings
            // The embedding tensor has shape [vocab_size, hidden_size]
            // Input has shape [batch_size, seq_len] with token IDs

            // Get input shape
            let input_shape = input.size();
            // tracing::debug!("Input tensor shape: {:?}, dtype: {:?}", input_shape, input.kind());
            // tracing::debug!("Embedding matrix shape: {:?}, dtype: {:?}", embed.size(), embed.kind());

            let batch_size = input_shape[0];
            let seq_len = if input_shape.len() > 1 {
                input_shape[1]
            } else {
                1
            };

            // Flatten input for embedding lookup (embedding expects 1D tensor)
            let flat_input = input.flatten(0, -1);
            // tracing::debug!("Flattened input shape: {:?}, dtype: {:?}", flat_input.size(), flat_input.kind());

            // Perform embedding lookup using index_select
            let embeddings = embed.index_select(0, &flat_input);

            // Get the actual hidden size from the embedding result
            let emb_dims = embeddings.size();
            let hidden_size = emb_dims[emb_dims.len() - 1]; // Last dimension is hidden size
                                                            // tracing::debug!("Embeddings shape after lookup: {:?}, hidden_size: {}", emb_dims, hidden_size);

            // Reshape back to [batch_size, seq_len, hidden_size]
            let mut embeddings = embeddings.reshape([batch_size, seq_len, hidden_size]);

            // Scale embeddings by sqrt(hidden_size) for Gemma3
            if self.config.scale_embeddings {
                let scale = (hidden_size as f32).sqrt();
                let scale_tensor = Tensor::from_slice(&[scale])
                    .to_kind(embeddings.kind()) // Match embeddings dtype (likely BF16)
                    .to_device(embeddings.device());
                embeddings = broadcast_mul(&embeddings, &scale_tensor)?;
            }

            embeddings
        } else {
            // If no embedding layer, assume input is already embedded
            input.shallow_clone()
        };

        // Apply transformer layers
        tracing::trace!("Total layers to process: {}", self.layers.len());

        // Generate position_ids based on start_pos (for proper KV cache usage)
        let seq_len = hidden_states.size()[1];
        let position_ids = if start_pos == 0 {
            // Processing full prompt - positions from 0 to seq_len
            Tensor::arange(seq_len, (tch::Kind::Int64, hidden_states.device()))
        } else {
            // Processing new tokens - positions from start_pos onwards
            Tensor::arange_start(
                start_pos as i64,
                (start_pos + seq_len as usize) as i64,
                (tch::Kind::Int64, hidden_states.device()),
            )
        };

        // PERF: Lock KV cache ONCE before the layer loop (not 28 times inside!)
        // This reduces lock overhead from O(layers) to O(1) per forward pass.
        let cache_guard = self.kv_cache.as_ref().map(|cache_ref| {
            match cache_ref.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    tracing::warn!("KV cache lock poisoned, recovering");
                    poisoned.into_inner()
                }
            }
        });

        for (idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden_states.shallow_clone();
            // tracing::debug!("Processing layer {}, hidden_states shape: {:?}", idx, hidden_states.size());

            // Self-attention block with optional KV cache
            hidden_states = layer.input_layernorm.forward(&hidden_states)?;

            // Handle KV cache per layer with proper start_pos
            let attn_output = if let Some(ref cache_manager) = cache_guard {
                if let Some(result) = cache_manager.with_layer_cache(idx, |layer_cache| {
                    tracing::trace!(
                        "Layer {}: Using KV cache, cache_pos={}",
                        idx,
                        layer_cache.seq_pos
                    );
                    layer.self_attn.forward(
                        &hidden_states,
                        Some(&position_ids),
                        Some(layer_cache),
                        start_pos,
                    )
                }) {
                    result?
                } else {
                    layer
                        .self_attn
                        .forward(&hidden_states, Some(&position_ids), None, start_pos)?
                }
            } else {
                layer
                    .self_attn
                    .forward(&hidden_states, Some(&position_ids), None, start_pos)?
            };
            hidden_states = residual + attn_output;

            // FFN block
            let residual = hidden_states.shallow_clone();
            hidden_states = layer.post_attention_layernorm.forward(&hidden_states)?;
            let ffn_output = layer.mlp.forward(&hidden_states)?;
            hidden_states = residual + ffn_output;
        }

        // Final layer norm
        if let Some(norm) = &self.norm {
            hidden_states = norm.forward(&hidden_states)?;
        }

        // LM head
        // tracing::debug!("Before LM head: hidden_states shape={:?}", hidden_states.size());
        if let Some(lm_head) = &self.lm_head {
            // LM head weight also needs to be transposed
            // tracing::debug!("Using LM head: shape={:?}", lm_head.size());
            hidden_states = hidden_states.matmul(lm_head);
            // tracing::debug!("After LM head: logits shape={:?}", hidden_states.size());
        } else if let Some(output_proj) = &self.lm_head_transposed {
            // Use pre-computed transposed embedding (tied weights, cached at load time)
            // This saves ~740MB per forward pass by avoiding the transpose allocation
            let hs_shape = hidden_states.size();
            if hs_shape.len() == 3 {
                let (batch_size, seq_len, hidden_size) = (hs_shape[0], hs_shape[1], hs_shape[2]);
                let hidden_2d = hidden_states.reshape([batch_size * seq_len, hidden_size]);
                let logits_2d = hidden_2d.matmul(output_proj);
                hidden_states = logits_2d.reshape([batch_size, seq_len, output_proj.size()[1]]);
            } else {
                hidden_states = hidden_states.matmul(output_proj);
            }
        } else {
            tracing::warn!("No LM head or embedding weights found - returning hidden states!");
        }

        // Mask padded vocabulary entries if vocab was padded
        // This ensures padded tokens can never be sampled
        let original_vocab_size = self.config.original_vocab_size;
        let padded_vocab_size = self.config.vocab_size;
        if padded_vocab_size > original_vocab_size && original_vocab_size > 0 {
            let logits_shape = hidden_states.size();
            let actual_vocab_size = logits_shape[logits_shape.len() - 1] as usize;

            // Only mask if we actually have padded tokens in the output
            if actual_vocab_size == padded_vocab_size {
                // Set logits for padded tokens to -1e10
                // This makes their probability effectively zero after softmax
                let mask_start = original_vocab_size as i64;
                let mask_count = (padded_vocab_size - original_vocab_size) as i64;

                if mask_count > 0 {
                    // Get the last dimension (vocab dimension) and mask the padded portion
                    // We need to use narrow + copy to update the values
                    let mask_values = Tensor::full(
                        [mask_count],
                        -1e10_f64,
                        (hidden_states.kind(), hidden_states.device())
                    );

                    // For a multi-dimensional tensor, we need to handle each batch/sequence position
                    if logits_shape.len() == 3 {
                        // Shape is [batch, seq_len, vocab]
                        let batch_size = logits_shape[0];
                        let seq_len = logits_shape[1];
                        for b in 0..batch_size {
                            for s in 0..seq_len {
                                let slice = hidden_states.select(0, b).select(0, s);
                                slice.narrow(0, mask_start, mask_count).copy_(&mask_values);
                            }
                        }
                    } else if logits_shape.len() == 2 {
                        // Shape is [batch*seq_len, vocab] or [seq_len, vocab]
                        let rows = logits_shape[0];
                        for i in 0..rows {
                            let slice = hidden_states.select(0, i);
                            slice.narrow(0, mask_start, mask_count).copy_(&mask_values);
                        }
                    } else {
                        // 1D tensor [vocab]
                        hidden_states.narrow(0, mask_start, mask_count).copy_(&mask_values);
                    }
                }
            }
        }

        Ok(hidden_states)
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        if let Some(embed) = &self.embed_tokens {
            // Get input shape
            let input_shape = input_ids.size();
            let batch_size = input_shape[0];
            let seq_len = if input_shape.len() > 1 {
                input_shape[1]
            } else {
                1
            };

            // Flatten input for embedding lookup
            let flat_input = input_ids.flatten(0, -1);

            // Perform embedding lookup
            let embeddings = embed.index_select(0, &flat_input);

            // Get hidden size from embedding result
            let emb_dims = embeddings.size();
            let hidden_size = emb_dims[emb_dims.len() - 1];

            // Reshape to [batch_size, seq_len, hidden_size]
            let mut embeddings = embeddings.reshape([batch_size, seq_len, hidden_size]);

            // Scale embeddings if needed (Gemma3)
            if self.config.scale_embeddings {
                let scale = (hidden_size as f32).sqrt();
                let scale_tensor = Tensor::from_slice(&[scale])
                    .to_kind(embeddings.kind())
                    .to_device(embeddings.device());
                embeddings = broadcast_mul(&embeddings, &scale_tensor)?;
            }

            Ok(embeddings)
        } else {
            Err(anyhow!("No embedding layer available in model"))
        }
    }

    fn forward_from_embeddings(&self, embeddings: &Tensor, start_pos: usize) -> Result<Tensor> {
        tracing::trace!(
            "LlamaModel forward_from_embeddings: embeddings shape={:?}, start_pos={}",
            embeddings.size(),
            start_pos
        );

        // Start with pre-computed embeddings instead of embedding token IDs
        let mut hidden_states = embeddings.shallow_clone();

        // Apply transformer layers (same as forward_with_cache)
        let seq_len = hidden_states.size()[1];
        let position_ids = if start_pos == 0 {
            Tensor::arange(seq_len, (tch::Kind::Int64, hidden_states.device()))
        } else {
            Tensor::arange_start(
                start_pos as i64,
                (start_pos + seq_len as usize) as i64,
                (tch::Kind::Int64, hidden_states.device()),
            )
        };

        for (idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden_states.shallow_clone();

            // Self-attention block with optional KV cache
            hidden_states = layer.input_layernorm.forward(&hidden_states)?;

            // Handle KV cache per layer with proper start_pos
            let attn_output = if let Some(cache_ref) = self.kv_cache.as_ref() {
                let cache_manager = cache_ref.lock().expect("Failed to lock KV cache");
                if let Some(result) = cache_manager.with_layer_cache(idx, |layer_cache| {
                    layer.self_attn.forward(
                        &hidden_states,
                        Some(&position_ids),
                        Some(layer_cache),
                        start_pos,
                    )
                }) {
                    result?
                } else {
                    layer
                        .self_attn
                        .forward(&hidden_states, Some(&position_ids), None, start_pos)?
                }
            } else {
                layer
                    .self_attn
                    .forward(&hidden_states, Some(&position_ids), None, start_pos)?
            };
            hidden_states = residual + attn_output;

            // FFN block
            let residual = hidden_states.shallow_clone();
            hidden_states = layer.post_attention_layernorm.forward(&hidden_states)?;
            let ffn_output = layer.mlp.forward(&hidden_states)?;
            hidden_states = residual + ffn_output;
        }

        // Final layer norm
        if let Some(norm) = &self.norm {
            hidden_states = norm.forward(&hidden_states)?;
        }

        // LM head (same logic as forward_with_cache)
        if let Some(lm_head) = &self.lm_head {
            hidden_states = hidden_states.matmul(lm_head);
        } else if let Some(output_proj) = &self.lm_head_transposed {
            // Use pre-computed transposed embedding (tied weights)
            let hs_shape = hidden_states.size();
            if hs_shape.len() == 3 {
                let (batch_size, seq_len, hidden_size) = (hs_shape[0], hs_shape[1], hs_shape[2]);
                let hidden_2d = hidden_states.reshape([batch_size * seq_len, hidden_size]);
                let logits_2d = hidden_2d.matmul(output_proj);
                hidden_states = logits_2d.reshape([batch_size, seq_len, -1]);
            } else {
                hidden_states = hidden_states.matmul(output_proj);
            }
        }

        // Mask padded tokens (same as forward_with_cache)
        let padded_vocab_size = self.config.vocab_size;
        let original_vocab_size = self.config.original_vocab_size;
        if padded_vocab_size > original_vocab_size && original_vocab_size > 0 {
            let logits_shape = hidden_states.size();
            let actual_vocab_size = logits_shape[logits_shape.len() - 1] as usize;

            if actual_vocab_size == padded_vocab_size {
                let mask_start = original_vocab_size as i64;
                let mask_count = (padded_vocab_size - original_vocab_size) as i64;

                if mask_count > 0 {
                    let mask_values = Tensor::full(
                        &[mask_count],
                        -1e10_f64,
                        (hidden_states.kind(), hidden_states.device())
                    );

                    if logits_shape.len() == 3 {
                        let batch_size = logits_shape[0];
                        let seq_len = logits_shape[1];
                        for b in 0..batch_size {
                            for s in 0..seq_len {
                                let slice = hidden_states.select(0, b).select(0, s);
                                slice.narrow(0, mask_start, mask_count).copy_(&mask_values);
                            }
                        }
                    } else if logits_shape.len() == 2 {
                        let rows = logits_shape[0];
                        for i in 0..rows {
                            let slice = hidden_states.select(0, i);
                            slice.narrow(0, mask_start, mask_count).copy_(&mask_values);
                        }
                    } else {
                        hidden_states.narrow(0, mask_start, mask_count).copy_(&mask_values);
                    }
                }
            }
        }

        Ok(hidden_states)
    }

    fn reshape_for_attention(&self, tensor: &Tensor, is_key_value: bool) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = dims3(tensor)?;

        if is_key_value {
            // For K,V: reshape to [batch, seq, num_kv_heads, head_dim]
            Ok(tensor.reshape([
                batch_size,
                seq_len,
                self.config.num_key_value_heads as i64,
                self.config.head_dim as i64,
            ]))
        } else {
            // For Q: reshape to [batch, seq, num_attention_heads, head_dim]
            Ok(tensor.reshape([
                batch_size,
                seq_len,
                self.config.num_attention_heads as i64,
                self.config.head_dim as i64,
            ]))
        }
    }

    fn apply_rope(&self, _tensor: &Tensor, _position_ids: &Tensor) -> Result<Tensor> {
        // RoPE is applied in the attention layers, not at the model level
        // This method should not be called
        Err(anyhow!(
            "RoPE should be applied in attention layers, not at model level"
        ))
    }

    fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        // Llama uses RMSNorm, preserving dtype
        let original_dtype = tensor.kind();
        let x2 = square_tensor(tensor)?;
        let mean = x2.mean_dim(&[-1i64][..], true, original_dtype);
        let rrms = (mean + self.config.rms_norm_eps as f64).reciprocal().sqrt();
        broadcast_mul(tensor, &rrms)
    }

    fn get_attention_mask(&self, seq_len: usize, past_kv_len: usize) -> Result<Tensor> {
        // Create causal attention mask using model dtype
        let total_len = seq_len + past_kv_len;
        let mask = Tensor::ones(
            [seq_len as i64, total_len as i64],
            (self.dtype, self.device),
        );

        // Apply causal masking (simplified)
        Ok(mask)
    }

    fn apply_lora(&mut self, adapter: &crate::lora::torch_adapter::LoRAModel) -> Result<()> {
        // Apply LoRA weights with architecture-specific handling
        // The adapter will handle GQA shape conversions for Llama 2/3
        let _ = adapter;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // ============================================================================
    // KV Cache Management (trait implementation)
    // ============================================================================

    fn clear_kv_cache(&self) {
        if let Some(cache_ref) = self.kv_cache.as_ref() {
            let cache_manager = cache_ref.lock().expect("kv cache mutex poisoned");
            cache_manager.clear_all();
        }
    }

    fn kv_cache_memory_usage(&self) -> usize {
        self.kv_cache
            .as_ref()
            .map(|cache_ref| cache_ref.lock().expect("kv cache mutex poisoned").memory_usage())
            .unwrap_or(0)
    }

    fn set_kv_cache(
        &mut self,
        cache: std::sync::Arc<std::sync::Mutex<crate::runtime::kv_cache::KVCacheManager>>,
    ) {
        self.kv_cache = Some(cache);
    }

    fn get_kv_cache(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<crate::runtime::kv_cache::KVCacheManager>>> {
        self.kv_cache.clone()
    }

    fn take_kv_cache(
        &mut self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<crate::runtime::kv_cache::KVCacheManager>>> {
        self.kv_cache.take()
    }
}

// KV cache methods are now defined in the ModelOperations trait implementation above
