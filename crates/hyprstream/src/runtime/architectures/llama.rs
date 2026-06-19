//! Llama model implementation with support for Llama 1/2/3 and GQA

use super::{ArchitectureConfig, ModelArchitecture, ModelOperations};
// use super::lora_adapter::ArchitectureAwareLoRAAdapter; // Module removed
use crate::runtime::device_pool::LayerDeviceMap;
use crate::runtime::rope::RoPE;
use crate::runtime::tensor_helpers::{
    broadcast_add, broadcast_mul, dims3, dims4, scalar_tensor, square_tensor,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use tch::{nn, Device, Kind as DType, Tensor};

/// Linear projection layer with optional bias and optional FP8 block scale.
///
/// Weights may be stored as FP8 (E4M3 or E5M2) to save VRAM. When a `scale`
/// tensor is present (shape `[out/128, in/128]`), `apply()` performs block-wise
/// dequantization: cast FP8 → BF16 on GPU, expand scale to weight shape, multiply.
/// ROCm 7.1+ dispatches the FP8 cast on-GPU without CPU fallback.
pub(crate) struct LinearProjection {
    pub(crate) weight: Tensor,
    pub(crate) bias: Option<Tensor>,
    /// Block-wise scale for FP8 weights: shape [out/block, in/block], BF16.
    /// None for non-FP8 weights.
    pub(crate) scale: Option<Tensor>,
}

impl LinearProjection {
    /// Create projection from weight only (no bias, no scale).
    #[inline]
    pub(crate) fn new(weight: Tensor) -> Self {
        Self { weight, bias: None, scale: None }
    }

    /// Create projection with weight and bias (no scale).
    #[inline]
    pub(crate) fn with_bias(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias: Some(bias), scale: None }
    }

    /// Create FP8 projection with block-wise scale.
    #[inline]
    pub(crate) fn with_scale(weight: Tensor, scale: Tensor) -> Self {
        Self { weight, bias: None, scale: Some(scale) }
    }

    /// Move all owned tensors to `device` (no-op per tensor already there).
    /// Used by the 2b pipeline to place a layer on its mapped device (#314).
    #[inline]
    pub(crate) fn into_device(self, device: Device) -> Self {
        Self {
            weight: self.weight.to_device(device),
            bias: self.bias.map(|b| b.to_device(device)),
            scale: self.scale.map(|s| s.to_device(device)),
        }
    }

    /// Apply projection to input: output = input @ weight + bias
    ///
    /// Input shape:  [*, in_features]
    /// Weight shape: [in_features, out_features]  (pre-transposed at load time)
    /// Output shape: [*, out_features]
    ///
    /// FP8 weights are dequantized lazily: `LinearProjection::take` captures the
    /// companion `_scale_inv` tensor, and `apply()` expands the block scale and
    /// multiplies on each forward pass (keeping VRAM at FP8 size).
    #[inline]
    pub(crate) fn apply(&self, input: &Tensor) -> Tensor {
        let w = match self.weight.kind() {
            tch::Kind::Float8e4m3fn | tch::Kind::Float8e5m2 => {
                // Cast FP8 → BF16 on GPU (ROCm 7.1+ dispatches on-device, no CPU fallback).
                let w_bf16 = self.weight.to_kind(tch::Kind::BFloat16);
                if let Some(scale) = &self.scale {
                    // scale: [in/128, out/128] — apply block-wise via reshape+expand+reshape.
                    // This avoids repeat_interleave (which fully materializes a copy) by using
                    // a lazy expand view over the 128-element block strides.
                    let ws = w_bf16.size();  // [in, out]
                    let ss = scale.size();   // [in/128, out/128]
                    let block_r = ws[0] / ss[0];
                    let block_c = ws[1] / ss[1];
                    // Block-wise dequantization via 4D broadcast multiply.
                    // The scale is broadcast across block dimensions without materializing
                    // a full [in, out] expanded copy — PyTorch's CUDA kernel reads from
                    // the stride-zero view directly. The multiply output is contiguous,
                    // so the final reshape back to 2D is zero-copy.
                    let w_4d = w_bf16.view([ss[0], block_r, ss[1], block_c]);
                    let s_4d = scale.to_kind(tch::Kind::BFloat16).view([ss[0], 1, ss[1], 1]);
                    (w_4d * s_4d).reshape([ws[0], ws[1]])
                } else {
                    w_bf16
                }
            }
            _ => self.weight.shallow_clone(),
        };
        let output = input.matmul(&w);
        match &self.bias {
            Some(bias) => output + bias,
            None => output,
        }
    }

    /// Remove weight (and its companion `_scale_inv` if present) from `weights`,
    /// transpose from `[out, in]` → `[in, out]`, and build a `LinearProjection`.
    ///
    /// FP8 weights are kept as FP8 in VRAM. The scale (if present) is stored for
    /// use during `apply()`. This keeps memory at FP8 size (e.g. 7.4 GB for 35B),
    /// which is necessary when the BF16 equivalent would exceed VRAM (e.g. 70 GB > 64 GB).
    pub(crate) fn take(weights: &mut HashMap<String, Tensor>, key: &str) -> Result<Self> {
        let weight = weights
            .remove(key)
            .ok_or_else(|| anyhow!("Missing weight tensor: {}", key))?
            .transpose(0, 1)
            .contiguous();
        // Scale stored as [out/128, in/128]; transpose to match [in, out] weight orientation.
        let scale = weights
            .remove(&format!("{key}_scale_inv"))
            .map(|s| s.transpose(0, 1).contiguous());
        Ok(match scale {
            Some(s) => Self::with_scale(weight, s),
            None => Self::new(weight),
        })
    }

    /// Like `take`, but also attaches an optional bias tensor (not transposed).
    /// Tries `bias_key` in `weights`; skips silently if absent.
    pub(crate) fn take_with_optional_bias(
        weights: &mut HashMap<String, Tensor>,
        weight_key: &str,
        bias_key: &str,
    ) -> Result<Self> {
        let mut proj = Self::take(weights, weight_key)?;
        if let Some(bias) = weights.get(bias_key).map(tch::Tensor::shallow_clone) {
            proj.bias = Some(bias);
        }
        Ok(proj)
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
fn calculate_padded_vocab_size(vocab_size: u32) -> u32 {
    vocab_size.div_ceil(64) * 64
}

/// Llama model configuration
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Llama version (1, 2, or 3)
    pub version: u8,
    /// Number of attention heads for queries
    pub num_attention_heads: u32,
    /// Number of key-value heads (for GQA in Llama 2/3)
    pub num_key_value_heads: u32,
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Intermediate size for FFN
    pub intermediate_size: u32,
    /// Maximum position embeddings
    pub max_position_embeddings: u32,
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,
    /// Vocabulary size (may be padded)
    pub vocab_size: u32,
    /// Original vocabulary size (before padding)
    pub original_vocab_size: u32,
    /// Number of hidden layers
    pub num_hidden_layers: u32,
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
            hidden_activation: "silu".to_owned(),
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
                scaling_type: "linear".to_owned(),
                factor: 8.0,
            }),
            hidden_activation: "silu".to_owned(),
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
                scaling_type: "linear".to_owned(),
                factor: 8.0,
            }),
            hidden_activation: "silu".to_owned(),
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
        self.num_attention_heads as usize
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads as usize
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size as usize
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size as usize
    }

    fn head_dim(&self) -> usize {
        self.head_dim as usize
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size as usize
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings as usize
    }

    fn rope_theta(&self) -> Option<f32> {
        Some(self.rope_theta)
    }

    fn rope_dim(&self) -> Option<usize> {
        Some(self.head_dim as usize)
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

    /// Per-global-layer device assignment (#314, 2b pipeline). For the unsplit
    /// fast path this is a single-device map of length `num_hidden_layers`
    /// (`is_single_device() == true`); for a pipeline shard it is the *global*
    /// map (still length `num_hidden_layers`) and `layer_offset` selects the
    /// owned window. Drives both per-layer placement at construction and the lone
    /// boundary `to_device` in `forward_layers`.
    device_map: LayerDeviceMap,

    /// Global index of `self.layers[0]`. `0` for a whole model; `a` for a shard
    /// owning global layers `[a..a+self.layers.len())`. The single place a global
    /// index is needed at runtime is mapping a local KV-cache slot / device
    /// lookup back to its global layer.
    layer_offset: usize,

    // Model weights
    embed_tokens: Option<Tensor>,
    layers: Vec<LlamaLayer>,
    norm: Option<RMSNorm>,
    lm_head: Option<Tensor>,

    /// Pre-transposed lm_head for tied weights (avoids transpose per forward pass)
    /// This is set when lm_head is None but embed_tokens is Some
    lm_head_transposed: Option<Tensor>,

    // KV cache for efficient generation (thread-safe with parking_lot::Mutex)
    kv_cache: Option<std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>>,

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

impl LlamaLayer {
    /// Move every weight in this layer to `device` (#314 pipeline placement).
    /// A no-op per tensor already resident on `device`.
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self {
            self_attn: self.self_attn.into_device(device),
            mlp: self.mlp.into_device(device),
            input_layernorm: self.input_layernorm.into_device(device),
            post_attention_layernorm: self.post_attention_layernorm.into_device(device),
        }
    }
}

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
    /// Move all projection + QK-norm weights to `device` (#314 pipeline placement).
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self {
            q_proj: self.q_proj.into_device(device),
            k_proj: self.k_proj.into_device(device),
            v_proj: self.v_proj.into_device(device),
            o_proj: self.o_proj.into_device(device),
            q_norm: self.q_norm.map(|t| t.to_device(device)),
            k_norm: self.k_norm.map(|t| t.to_device(device)),
            ..self
        }
    }

    /// Apply attention with optional GQA, KV caching, and delta injection
    fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: Option<&Tensor>,
        kv_cache: Option<&mut crate::runtime::kv_cache::LayerKVCache>,
        start_pos: usize,
        delta: Option<&crate::training::TenantDelta>,
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
        let mut q = self.q_proj.apply(&hidden_states_2d);
        let mut k = self.k_proj.apply(&hidden_states_2d);
        let mut v = self.v_proj.apply(&hidden_states_2d);

        // Delta injection — add LoRA corrections to attention projections
        // During inference (no_grad), corrections are detached tensors (no gradient overhead).
        // During training, these participate in the computation graph.
        if let Some(delta) = delta {
            for (name, proj) in [("q_proj", &mut q), ("k_proj", &mut k), ("v_proj", &mut v)] {
                if delta.has_module(name, self.layer_idx) {
                    let correction = delta.forward_2d(&hidden_states_2d, name, self.layer_idx)?;
                    let kind = proj.kind();
                    if start_pos == 0 && self.layer_idx == 0 {
                        tracing::info!("[TTT] Layer 0 {}: correction_norm={:.6}, proj_norm={:.4}, ratio={:.6}",
                            name, correction.norm().double_value(&[]),
                            proj.norm().double_value(&[]),
                            correction.norm().double_value(&[]) / (proj.norm().double_value(&[]) + 1e-10));
                    }
                    *proj = proj.f_add(&correction.to_kind(kind))
                        .map_err(|e| anyhow::anyhow!(
                            "Delta correction shape mismatch at layer {} {}: proj {:?} vs correction {:?}: {}",
                            self.layer_idx, name, proj.size(), correction.size(), e
                        ))?;
                }
            }
        }

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

        // Memory-efficient fused attention. For long prefills it chunks over the
        // query axis so the [q_len, k_len] FP32 score matrix is never materialized
        // at full size — capping peak VRAM at O(CHUNK * k_len) instead of
        // O(q_len * k_len). An 8K-token prompt otherwise balloons the score/softmax
        // tensors past 20 GB and the CUDA caching allocator retains that peak,
        // OOMing the next request. Backend-agnostic (plain matmul/softmax): SDPA's
        // flash/efficient kernels panic on this ROCm/libtorch version and
        // math_attention is slower, so chunking is used instead.
        let attn_output = self.compute_attention(&q, &k_expanded, &v_expanded)?;

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
        let mut attn_output = self.o_proj.apply(&attn_output_2d);

        // Delta injection for o_proj
        if let Some(delta) = delta {
            if delta.has_module("o_proj", self.layer_idx) {
                let correction = delta.forward_2d(&attn_output_2d, "o_proj", self.layer_idx)?;
                let kind = attn_output.kind();
                attn_output += correction.to_kind(kind);
            }
        }

        // Reshape back to 3D: [batch, seq, hidden_size]
        let attn_output = attn_output.reshape([batch_size, seq_len, hidden_size]);

        Ok(attn_output)
    }

    /// Detect the actual number of KV heads from a projected K tensor.
    ///
    /// Mirrors the inline detection in [`LlamaAttention::forward`] so the batched
    /// path stays byte-consistent with the reference. Extracted to keep both paths
    /// in lock-step (DRY).
    fn detect_kv_heads(&self, k_elements: i64, rows: i64) -> usize {
        if k_elements == rows * (self.num_kv_heads as i64) * (self.head_dim as i64) {
            self.num_kv_heads
        } else if k_elements % (rows * (self.head_dim as i64)) == 0 {
            (k_elements / (rows * (self.head_dim as i64))) as usize
        } else if k_elements % rows == 0 {
            let kv_dim = (k_elements / rows) as usize;
            if kv_dim == 256 && self.head_dim == 128 {
                2
            } else if kv_dim == 256 {
                8
            } else {
                self.num_kv_heads
            }
        } else {
            self.num_kv_heads
        }
    }

    /// Batched ragged self-attention for continuous decode (#329, Llama-only).
    ///
    /// Processes `B` sequences in one batched forward. Each row `r` has:
    /// - `hidden_states[r]`: `[q, H]` (all rows share `q`; v1 decode uses `q == 1`),
    /// - its own absolute `start_positions[r]` and per-row `LayerKVCache`.
    ///
    /// Q/K/V are projected on the whole `[B, q, H]` batch (one matmul; same-delta
    /// LoRA injection applies uniformly). RoPE is applied per row with that row's
    /// absolute positions (`apply_rope`'s 1D path is exact; a 2D batched index
    /// would mis-broadcast). KV: each row's new K/V is written to its own cache,
    /// then every row's dense history is read back and **right-padded** to the
    /// batch-max KV length, stacked to `[B, kv_max, heads, dim]`. The supplied
    /// additive `attn_mask` (`[B, 1, q, kv_max]`) masks both the causal structure
    /// and each row's padding, so a single masked matmul is numerically identical
    /// to running each row through [`LlamaAttention::forward`] serially.
    #[allow(clippy::too_many_arguments)]
    fn forward_batched(
        &self,
        hidden_states: &Tensor,
        position_ids: &[Tensor],
        kv_caches: &mut [&mut crate::runtime::kv_cache::LayerKVCache],
        start_positions: &[usize],
        attn_mask: &Tensor,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = dims3(hidden_states)?;
        debug_assert_eq!(batch_size as usize, kv_caches.len());
        debug_assert_eq!(batch_size as usize, position_ids.len());
        debug_assert_eq!(batch_size as usize, start_positions.len());

        // Project Q/K/V on the full batch: [B*q, H] -> projections.
        let hidden_states_2d = hidden_states.reshape([batch_size * seq_len, hidden_size]);
        let mut q = self.q_proj.apply(&hidden_states_2d);
        let mut k = self.k_proj.apply(&hidden_states_2d);
        let mut v = self.v_proj.apply(&hidden_states_2d);

        // Same-delta LoRA injection (scheduler guarantees one delta per batch).
        if let Some(delta) = delta {
            for (name, proj) in [("q_proj", &mut q), ("k_proj", &mut k), ("v_proj", &mut v)] {
                if delta.has_module(name, self.layer_idx) {
                    let correction = delta.forward_2d(&hidden_states_2d, name, self.layer_idx)?;
                    let kind = proj.kind();
                    *proj = proj.f_add(&correction.to_kind(kind)).map_err(|e| {
                        anyhow::anyhow!(
                            "Delta correction shape mismatch at layer {} {}: {}",
                            self.layer_idx, name, e
                        )
                    })?;
                }
            }
        }

        let rows = batch_size * seq_len;
        let kv_heads = self.detect_kv_heads(k.size().iter().product::<i64>(), rows);

        let mut q = q.reshape([batch_size, seq_len, self.num_heads as i64, self.head_dim as i64]);
        let mut k = k.reshape([batch_size, seq_len, kv_heads as i64, self.head_dim as i64]);
        let v = v.reshape([batch_size, seq_len, kv_heads as i64, self.head_dim as i64]);

        if let Some(q_norm) = &self.q_norm {
            q = self.apply_qk_norm(&q, q_norm, self.num_heads)?;
        }
        if let Some(k_norm) = &self.k_norm {
            k = self.apply_qk_norm(&k, k_norm, kv_heads)?;
        }

        // Per-row RoPE with that row's absolute positions, then re-stack.
        let mut q_rows: Vec<Tensor> = Vec::with_capacity(batch_size as usize);
        let mut k_rows: Vec<Tensor> = Vec::with_capacity(batch_size as usize);
        for (r, pos) in position_ids.iter().enumerate() {
            let q_r = q.narrow(0, r as i64, 1); // [1, q, heads, dim]
            let k_r = k.narrow(0, r as i64, 1); // [1, q, kv_heads, dim]
            q_rows.push(self.apply_rope(&q_r, pos)?);
            k_rows.push(self.apply_rope(&k_r, pos)?);
        }
        let q = Tensor::cat(&q_rows, 0);
        let k = Tensor::cat(&k_rows, 0);

        // Update each row's cache and collect its dense KV history.
        let mut row_keys: Vec<Tensor> = Vec::with_capacity(batch_size as usize);
        let mut row_values: Vec<Tensor> = Vec::with_capacity(batch_size as usize);
        let mut kv_lens: Vec<i64> = Vec::with_capacity(batch_size as usize);
        for r in 0..batch_size as usize {
            let k_r = k.narrow(0, r as i64, 1).contiguous();
            let v_r = v.narrow(0, r as i64, 1).contiguous();
            let cache = &mut *kv_caches[r];
            cache.update(&k_r, &v_r, start_positions[r])?;
            let (ck, cv) = cache.get()?; // [1, kv_r, kv_heads, dim]
            kv_lens.push(ck.size()[1]);
            row_keys.push(ck);
            row_values.push(cv);
        }

        // Right-pad every row's KV to the batch-max length so they stack. Padding
        // positions are masked out by `attn_mask`, so their values are irrelevant.
        let kv_max = kv_lens.iter().copied().max().unwrap_or(0);
        let opt = (k.kind(), k.device());
        let pad_row = |t: &Tensor, len: i64| -> Tensor {
            if len == kv_max {
                t.shallow_clone()
            } else {
                // [1, pad_len, kv_heads, dim] zeros appended on the seq axis.
                let mut shape = t.size();
                shape[1] = kv_max - len;
                let pad = Tensor::zeros(&shape[..], opt);
                Tensor::cat(&[t.shallow_clone(), pad], 1)
            }
        };
        let keys_padded: Vec<Tensor> = row_keys
            .iter()
            .zip(&kv_lens)
            .map(|(t, &len)| pad_row(t, len))
            .collect();
        let values_padded: Vec<Tensor> = row_values
            .iter()
            .zip(&kv_lens)
            .map(|(t, &len)| pad_row(t, len))
            .collect();
        let k_stacked = Tensor::cat(&keys_padded, 0); // [B, kv_max, kv_heads, dim]
        let v_stacked = Tensor::cat(&values_padded, 0);

        // GQA expansion on the stacked KV.
        let (k_expanded, v_expanded) = if kv_heads < self.num_heads {
            (
                self.expand_kv_for_gqa_with_heads(&k_stacked, kv_heads)?,
                self.expand_kv_for_gqa_with_heads(&v_stacked, kv_heads)?,
            )
        } else {
            (k_stacked, v_stacked)
        };

        // One masked batched attention for all rows.
        let attn_output = self.compute_attention_masked(&q, &k_expanded, &v_expanded, Some(attn_mask))?;

        let attn_output = attn_output.transpose(1, 2).contiguous();
        let attn_output =
            attn_output.reshape([batch_size, seq_len, (self.num_heads * self.head_dim) as i64]);
        let attn_output_2d =
            attn_output.reshape([batch_size * seq_len, (self.num_heads * self.head_dim) as i64]);
        let mut attn_output = self.o_proj.apply(&attn_output_2d);

        if let Some(delta) = delta {
            if delta.has_module("o_proj", self.layer_idx) {
                let correction = delta.forward_2d(&attn_output_2d, "o_proj", self.layer_idx)?;
                let kind = attn_output.kind();
                attn_output += correction.to_kind(kind);
            }
        }

        Ok(attn_output.reshape([batch_size, seq_len, hidden_size]))
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

            // SAFETY: We just inserted the key above, so it must exist
            let rope = cache.get_mut(&key).ok_or_else(|| {
                anyhow!("Internal error: RoPE cache entry missing for layer {}", self.layer_idx)
            })?;

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

    /// Fused scaled dot-product attention with causal masking.
    ///
    /// Computes `softmax(Q·Kᵀ · scale + mask)·V` in FP32 for numerical stability
    /// (BF16 attention causes progressive precision loss in long sequences,
    /// matching HuggingFace/vLLM practice). Fusing the QK, softmax, and V steps
    /// lets the score matrix be produced and consumed one query-chunk at a time.
    ///
    /// For long prefills (`q_len > CHUNK`) the query axis is processed in chunks
    /// so the `[q_len, k_len]` FP32 score/softmax tensors are never materialized at
    /// full size — peak VRAM is O(CHUNK · k_len) rather than O(q_len · k_len).
    /// The decode path (`q_len == 1`) and short prompts take the single-shot
    /// branch, byte-for-byte equivalent to the previous implementation.
    ///
    /// `q`, `k`, `v` are `[batch, seq, heads, dim]` (K/V already GQA-expanded by
    /// the caller); the result is `[batch, heads, q_len, dim]`.
    fn compute_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        self.compute_attention_masked(q, k, v, None)
    }

    /// Scaled dot-product attention with an explicit additive attention mask.
    ///
    /// When `attn_mask` is `Some`, it is an additive mask broadcastable to the
    /// `[batch, heads, q_len, k_len]` score tensor (`0.0` for attend, a large
    /// negative for mask-out) and *replaces* the implicit `tril` causal mask.
    /// This is the entry the batched/ragged decode path uses to express, in one
    /// matmul, both per-row causality AND per-row KV padding (rows in a batch have
    /// different valid KV lengths). When `attn_mask` is `None` the behavior is
    /// byte-identical to the original inline-`tril` single-sequence path — kept as
    /// the equivalence reference.
    fn compute_attention_masked(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        /// Query-chunk size for long-prefill attention. 1024 keeps the per-chunk
        /// score matrix small while amortizing kernel-launch overhead.
        const CHUNK: i64 = 1024;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let original_kind = q.kind();

        // [batch, seq, heads, dim] -> [batch, heads, seq, dim] (K transposed for matmul)
        // PERF: .contiguous() required for optimal batched matmul on ROCm/AMD
        let q = q.transpose(1, 2).contiguous();
        let k_t = k.transpose(1, 2).transpose(2, 3).contiguous(); // [batch, heads, dim, k_len]
        let v = v.transpose(1, 2).contiguous(); // [batch, heads, k_len, dim]

        // Upcast to FP32 for precise score computation (matches prior behavior).
        let upcast = original_kind != tch::Kind::Float;
        let q = if upcast { q.to_kind(tch::Kind::Float) } else { q };
        let k_t = if upcast { k_t.to_kind(tch::Kind::Float) } else { k_t };
        let v = if upcast { v.to_kind(tch::Kind::Float) } else { v };
        // Match the score dtype (FP32) so broadcast_add doesn't upcast/clash.
        let attn_mask = attn_mask.map(|m| {
            if m.kind() != tch::Kind::Float {
                m.to_kind(tch::Kind::Float)
            } else {
                m.shallow_clone()
            }
        });

        let q_len = q.size()[2];
        let k_len = k_t.size()[3];
        let device = q.device();

        // An explicit mask must go through the single-shot path: it already encodes
        // causality + padding for the whole [q_len, k_len] grid, and chunking would
        // need to re-slice it per chunk. Decode batches have q_len == 1 anyway.
        let single_shot = attn_mask.is_some() || self.sliding_window.is_some() || q_len <= CHUNK;

        let attn_output = if single_shot {
            // [batch, heads, q_len, k_len]
            let mut scores = q.matmul(&k_t) * (scale as f64);

            if let Some(mask) = attn_mask.as_ref() {
                // Explicit additive mask supplied by the batched path; it fully
                // determines which (q, kv) pairs are valid, so skip the inline tril.
                scores = broadcast_add(&scores, mask)?;
            } else if q_len > 1 {
                // Causal mask only when processing multiple query tokens (prompt
                // phase); for q_len == 1 (decode) all past positions are valid.
                let mask = Tensor::ones([q_len, k_len], (tch::Kind::Float, device)).tril(0);
                let mask = mask.unsqueeze(0).unsqueeze(0).expand_as(&scores);
                scores = scores.masked_fill(&mask.eq(0.0), -10000.0f64);
            }

            // Sliding window mask if configured (Gemma3)
            if let Some(window_size) = self.sliding_window {
                if self.layer_type == "local" {
                    scores = self.apply_sliding_window_mask(&scores, window_size)?;
                }
            }

            scores.softmax(-1, tch::Kind::Float).matmul(&v)
        } else {
            // Chunk over the query axis. Per-chunk causal mask `tril(start)` is the
            // exact restriction of the full `tril(0)` mask to rows [start, start+cur),
            // so masking is identical to the single-shot path.
            let mut outputs: Vec<Tensor> = Vec::new();
            let mut start = 0i64;
            while start < q_len {
                let cur = (q_len - start).min(CHUNK);
                let q_chunk = q.narrow(2, start, cur); // [batch, heads, cur, dim]
                let mut scores = q_chunk.matmul(&k_t) * (scale as f64); // [batch, heads, cur, k_len]
                let mask = Tensor::ones([cur, k_len], (tch::Kind::Float, device)).tril(start);
                let mask = mask.unsqueeze(0).unsqueeze(0).expand_as(&scores);
                scores = scores.masked_fill(&mask.eq(0.0), -10000.0f64);
                outputs.push(scores.softmax(-1, tch::Kind::Float).matmul(&v)); // [batch, heads, cur, dim]
                start += cur;
            }
            Tensor::cat(&outputs, 2) // [batch, heads, q_len, dim]
        };

        Ok(if upcast {
            attn_output.to_kind(original_kind)
        } else {
            attn_output
        })
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
pub(crate) struct LlamaMLP {
    pub(crate) gate_proj: LinearProjection,
    pub(crate) up_proj: LinearProjection,
    pub(crate) down_proj: LinearProjection,
    pub(crate) activation: String, // Activation function name
    pub(crate) layer_idx: usize,   // Layer index for per-layer delta lookup
}

unsafe impl Send for LlamaMLP {}
unsafe impl Sync for LlamaMLP {}

impl LlamaMLP {
    /// Move all projection weights to `device` (#314 pipeline placement).
    /// `pub(crate)` so qwen3_5's dense-MLP stage placement can reuse it.
    #[inline]
    pub(crate) fn into_device(self, device: Device) -> Self {
        Self {
            gate_proj: self.gate_proj.into_device(device),
            up_proj: self.up_proj.into_device(device),
            down_proj: self.down_proj.into_device(device),
            activation: self.activation,
            layer_idx: self.layer_idx,
        }
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        // Get dimensions
        let original_shape = hidden_states.size();
        let (batch_size, seq_len, hidden_size) = if original_shape.len() == 3 {
            (original_shape[0], original_shape[1], original_shape[2])
        } else {
            // Already 2D
            return self.forward_2d(hidden_states, delta);
        };

        // Reshape to 2D for matmul
        let hidden_2d = hidden_states.reshape([batch_size * seq_len, hidden_size]);

        // Apply SwiGLU: down(act(gate(x)) * up(x)) with delta corrections
        let mut gate_pre = self.gate_proj.apply(&hidden_2d);
        let mut up = self.up_proj.apply(&hidden_2d);

        if let Some(delta) = delta {
            if delta.has_module("gate_proj", self.layer_idx) {
                let correction = delta.forward_2d(&hidden_2d, "gate_proj", self.layer_idx)?;
                let kind = gate_pre.kind();
                gate_pre += correction.to_kind(kind);
            }
            if delta.has_module("up_proj", self.layer_idx) {
                let correction = delta.forward_2d(&hidden_2d, "up_proj", self.layer_idx)?;
                let kind = up.kind();
                up += correction.to_kind(kind);
            }
        }

        let gate = self.apply_activation(&gate_pre)?;
        let gated = &gate * &up;

        let mut output = self.down_proj.apply(&gated);
        if let Some(delta) = delta {
            if delta.has_module("down_proj", self.layer_idx) {
                let correction = delta.forward_2d(&gated, "down_proj", self.layer_idx)?;
                let kind = output.kind();
                output += correction.to_kind(kind);
            }
        }

        // Reshape back to 3D
        let out_size = output.size()[1];
        Ok(output.reshape([batch_size, seq_len, out_size]))
    }

    fn forward_2d(
        &self,
        hidden_states: &Tensor,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        // Apply SwiGLU: down(act(gate(x)) * up(x))
        let mut gate_pre = self.gate_proj.apply(hidden_states);
        let mut up = self.up_proj.apply(hidden_states);

        if let Some(delta) = delta {
            if delta.has_module("gate_proj", self.layer_idx) {
                let correction = delta.forward_2d(hidden_states, "gate_proj", self.layer_idx)?;
                let kind = gate_pre.kind();
                gate_pre += correction.to_kind(kind);
            }
            if delta.has_module("up_proj", self.layer_idx) {
                let correction = delta.forward_2d(hidden_states, "up_proj", self.layer_idx)?;
                let kind = up.kind();
                up += correction.to_kind(kind);
            }
        }

        let gate = self.apply_activation(&gate_pre)?;
        let gated = &gate * &up;

        let mut output = self.down_proj.apply(&gated);
        if let Some(delta) = delta {
            if delta.has_module("down_proj", self.layer_idx) {
                let correction = delta.forward_2d(&gated, "down_proj", self.layer_idx)?;
                let kind = output.kind();
                output += correction.to_kind(kind);
            }
        }

        Ok(output)
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
pub(crate) struct RMSNorm {
    pub(crate) weight: Tensor,
    pub(crate) eps: f32,
}

unsafe impl Send for RMSNorm {}
unsafe impl Sync for RMSNorm {}

impl RMSNorm {
    /// Move the norm weight to `device` (#314 pipeline placement).
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self { weight: self.weight.to_device(device), eps: self.eps }
    }
}

impl RMSNorm {
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
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

        // Whole-model single-device map (this simplified path has no layers yet;
        // size the map to the declared depth so the invariant holds).
        let device_map = LayerDeviceMap::single(*device, (config.num_hidden_layers as usize).max(1))?;

        Ok(Self {
            config,
            device: *device,
            dtype,
            device_map,
            layer_offset: 0,
            embed_tokens: None,
            layers,
            norm: None,
            lm_head: None,
            lm_head_transposed: None,
            kv_cache: None, // Cache initialized on first use
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
        Self::from_weights_with_config(weights, config, device, dtype, crate::runtime::KVQuantType::None)
    }

    /// Create Llama model with explicit config (allows Qwen models to override)
    /// Build model from weights, taking mutable reference to free tensors as they're processed.
    /// This reduces peak memory by ~50% by removing original weight tensors after transposing.
    pub fn from_weights_with_config(
        weights: &mut HashMap<String, Tensor>,
        mut config: LlamaConfig,
        device: &Device,
        dtype: DType,
        kv_quant_type: crate::runtime::KVQuantType,
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
                let vocab_size = w.size()[0] as u32;
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
                let vocab_size = w.size()[0] as u32;
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
        for layer_idx in 0..config.num_hidden_layers as usize {
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
            Some(std::sync::Arc::new(parking_lot::Mutex::new(
                crate::runtime::kv_cache::KVCacheManager::new(
                    layers.len(),
                    config.max_position_embeddings as usize,
                    kv_quant_type,
                ),
            )))
        } else {
            None
        };

        // Unsplit fast path: every layer on the one device. This keeps the
        // whole-model forward byte-identical — `forward_layers` over this
        // single-device map performs zero cross-device copies (#314).
        let device_map = LayerDeviceMap::single(*device, layers.len().max(1))?;

        Ok(Self {
            config,
            device: *device,
            dtype,
            device_map,
            layer_offset: 0,
            embed_tokens,
            layers,
            norm,
            lm_head,
            lm_head_transposed,
            kv_cache,
            vs: None, // No VarStore for weight loading - weights are stored directly
        })
    }

    /// Build a **single pipeline stage** of a Llama model (#314, 2b layer-split).
    ///
    /// Loads only global decoder layers `[layer_range.start..layer_range.end)`
    /// onto their mapped devices (`devices.device_for(g)`), and gates the
    /// non-layer weights by stage position so middle stages carry none (M-LOAD
    /// seam #1):
    /// - `is_first` (`layer_range.start == 0`)  → keep `embed_tokens`.
    /// - `is_last`  (`layer_range.end == devices.len()`) → keep `norm` + `lm_head`
    ///   (or the tied transpose).
    ///
    /// `weights` must already contain (at least) this stage's tensors; the loader
    /// is responsible for shard selection (a later ticket). Tensors not on the
    /// target device are moved with a single `.to()` at construction — the *only*
    /// placement cost; the forward path then does zero intra-stage copies.
    ///
    /// State sizing follows the **owned** layer count, not the global count
    /// (M-LOAD seam #2): the KV-cache manager and `self.layers` are both sized to
    /// `layer_range.len()`, and the forward loop indexes them locally.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_from_weights_with_config(
        weights: &mut HashMap<String, Tensor>,
        mut config: LlamaConfig,
        devices: &LayerDeviceMap,
        layer_range: std::ops::Range<usize>,
        dtype: DType,
        kv_quant_type: crate::runtime::KVQuantType,
    ) -> Result<Self> {
        let num_global = config.num_hidden_layers as usize;
        if devices.len() != num_global {
            return Err(anyhow!(
                "stage_from_weights: device map covers {} layers but config has {} \
                 (num_hidden_layers)",
                devices.len(),
                num_global
            ));
        }
        if layer_range.end > num_global || layer_range.start >= layer_range.end {
            return Err(anyhow!(
                "stage_from_weights: invalid layer range {:?} for {} layers",
                layer_range,
                num_global
            ));
        }

        let is_first = layer_range.start == 0;
        let is_last = layer_range.end == num_global;
        let layer_offset = layer_range.start;
        // Stage device = device of this stage's first owned layer (its inputs and
        // KV cache live here). `embed_tokens`, when present, lives on the first
        // stage's first device, which is exactly this when is_first.
        let stage_device = devices.device_for(layer_range.start);

        // --- Non-layer weights, gated by stage position (M-LOAD seam #1) ---
        let original_vocab_size = config.vocab_size;
        let padded_vocab_size = calculate_padded_vocab_size(original_vocab_size);
        config.original_vocab_size = original_vocab_size;
        if padded_vocab_size != original_vocab_size {
            config.vocab_size = padded_vocab_size;
        }

        let embed_tokens = if is_first {
            let embed = weights
                .get("model.embed_tokens.weight")
                .or_else(|| weights.get("embed_tokens.weight"))
                .map(|w| Self::pad_embedding_to(w, padded_vocab_size, stage_device))
                .ok_or_else(|| {
                    anyhow!("stage_from_weights: first stage requires model.embed_tokens.weight")
                })?;
            Some(embed)
        } else {
            None
        };

        let (lm_head, norm) = if is_last {
            let lm_head = weights
                .get("lm_head.weight")
                .or_else(|| weights.get("model.lm_head.weight"))
                .map(|w| {
                    Self::pad_embedding_to(w, padded_vocab_size, stage_device)
                        .transpose(0, 1)
                        .contiguous()
                });
            let norm = weights
                .get("model.norm.weight")
                .or_else(|| weights.get("norm.weight"))
                .map(|w| RMSNorm {
                    weight: w.shallow_clone().to_device(stage_device),
                    eps: config.rms_norm_eps,
                });
            if norm.is_none() {
                return Err(anyhow!(
                    "stage_from_weights: last stage requires model.norm.weight"
                ));
            }
            (lm_head, norm)
        } else {
            (None, None)
        };

        // Tied lm_head: only meaningful on the last stage, and only if it also
        // holds embed_tokens (single-stage model). A middle/last stage that does
        // not own the embedding cannot tie — it must ship an explicit lm_head.
        let lm_head_transposed = if is_last && lm_head.is_none() {
            embed_tokens
                .as_ref()
                .map(|embed| embed.transpose(0, 1).contiguous())
        } else {
            None
        };
        if is_last && lm_head.is_none() && lm_head_transposed.is_none() {
            return Err(anyhow!(
                "stage_from_weights: last stage requires lm_head.weight (no tied embedding present)"
            ));
        }

        // --- Owned decoder layers, each on its mapped device (per-layer .to) ---
        let mut layers = Vec::with_capacity(layer_range.len());
        for g in layer_range.clone() {
            let target = devices.device_for(g);
            match Self::build_layer(g, weights, &config, &target)? {
                Some(layer) => layers.push(layer.into_device(target)),
                None => {
                    return Err(anyhow!(
                        "stage_from_weights: expected weights for global layer {g} but none found"
                    ))
                }
            }
        }

        // KV cache sized to the OWNED layer count (M-LOAD seam #2); indexed
        // locally in the forward loop. Lives on the stage device.
        let kv_cache = if layers.is_empty() {
            None
        } else {
            Some(std::sync::Arc::new(parking_lot::Mutex::new(
                crate::runtime::kv_cache::KVCacheManager::new(
                    layers.len(),
                    config.max_position_embeddings as usize,
                    kv_quant_type,
                ),
            )))
        };

        let device_map = devices.clone();

        Ok(Self {
            config,
            device: stage_device,
            dtype,
            device_map,
            layer_offset,
            embed_tokens,
            layers,
            norm,
            lm_head,
            lm_head_transposed,
            kv_cache,
            vs: None,
        })
    }

    /// Pad an embedding/lm_head weight `[vocab, hidden]` up to `padded_vocab` (a
    /// no-op shallow clone when already large enough) and place it on `device`.
    fn pad_embedding_to(w: &Tensor, padded_vocab: u32, device: Device) -> Tensor {
        let vocab_size = w.size()[0] as u32;
        let hidden_size = w.size()[1];
        let placed = if padded_vocab > vocab_size {
            let padded = Tensor::zeros([padded_vocab as i64, hidden_size], (w.kind(), w.device()));
            padded.narrow(0, 0, vocab_size as i64).copy_(w);
            padded
        } else {
            w.shallow_clone()
        };
        placed.to_device(device)
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
                config.vocab_size = shape[0] as u32;
                config.original_vocab_size = shape[0] as u32;  // Initially same
                config.hidden_size = shape[1] as u32;

                // Detect Gemma3 by vocab size (262144) and set specific parameters
                if config.vocab_size == 262144 {
                    config.hidden_activation = "gelu_pytorch_tanh".to_owned();
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
            config.num_hidden_layers = layer_count as u32;
        }

        // Get attention heads from q_proj and k_proj shapes
        if let Some(q_proj) = weights.get("model.layers.0.self_attn.q_proj.weight") {
            let q_shape = q_proj.size();
            if q_shape.len() >= 2 {
                // shape[0] is output dim (num_heads * head_dim)
                // shape[1] is hidden_size
                config.hidden_size = q_shape[1] as u32;
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
                        config.num_attention_heads = (q_proj_out_dim / head_dim) as u32;
                        config.num_key_value_heads = (k_proj_out_dim / head_dim) as u32;
                        config.head_dim = head_dim as u32;
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
        let prefix = format!("model.layers.{layer_idx}");

        // Check if this layer exists (handle both separate and combined projections)
        let has_separate_qkv = weights.contains_key(&format!("{prefix}.self_attn.q_proj.weight"));
        let has_combined_qkv = weights.contains_key(&format!("{prefix}.self_attn.c_attn.weight"));

        if !has_separate_qkv && !has_combined_qkv {
            return Ok(None);
        }

        // Build attention
        let (q_proj, k_proj, v_proj) = if has_separate_qkv {
            // Standard separate Q, K, V projections (Llama/Qwen style)

            let q_proj = LinearProjection::take_with_optional_bias(
                weights,
                &format!("{prefix}.self_attn.q_proj.weight"),
                &format!("{prefix}.self_attn.q_proj.bias"),
            )?;
            let k_proj = LinearProjection::take_with_optional_bias(
                weights,
                &format!("{prefix}.self_attn.k_proj.weight"),
                &format!("{prefix}.self_attn.k_proj.bias"),
            )?;
            let v_proj = LinearProjection::take_with_optional_bias(
                weights,
                &format!("{prefix}.self_attn.v_proj.weight"),
                &format!("{prefix}.self_attn.v_proj.bias"),
            )?;

            (q_proj, k_proj, v_proj)
        } else {
            // Combined QKV projection (some Qwen models use c_attn)
            let c_attn_weight = weights
                .remove(&format!("{prefix}.self_attn.c_attn.weight"))
                .ok_or_else(|| anyhow!("Missing c_attn weight"))?
                .transpose(0, 1)
                .contiguous();

            // c_attn has shape [hidden_size, 3 * projection_size]
            let total_proj_size = c_attn_weight.size()[1];
            let proj_size = total_proj_size / 3;

            let q_weight = c_attn_weight.narrow(1, 0, proj_size);
            let k_weight = c_attn_weight.narrow(1, proj_size, proj_size);
            let v_weight = c_attn_weight.narrow(1, proj_size * 2, proj_size);

            let (q_proj, k_proj, v_proj) = if let Some(bias) = weights.get(&format!("{prefix}.self_attn.c_attn.bias")) {
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
            .get(&format!("{prefix}.self_attn.q_norm.weight"))
            .map(tch::Tensor::shallow_clone);
        let k_norm = weights
            .get(&format!("{prefix}.self_attn.k_norm.weight"))
            .map(tch::Tensor::shallow_clone);

        if q_norm.is_some() || k_norm.is_some() {
            tracing::debug!("Layer {} has QK-norm weights", layer_idx);
        }

        // Determine layer type for Gemma3 sliding window attention
        let layer_type = if !config.layer_types.is_empty() && layer_idx < config.layer_types.len() {
            config.layer_types[layer_idx].clone()
        } else if config.layer_types.is_empty() && config.use_qk_norm {
            // Gemma3 pattern: every 6th layer is global, others are local
            if (layer_idx + 1).is_multiple_of(6) {
                "global".to_owned()
            } else {
                "local".to_owned()
            }
        } else {
            "global".to_owned() // Default to global attention
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
        // Try o_proj first, then c_proj (some Qwen models use c_proj for output)
        let o_proj = if weights.contains_key(&format!("{prefix}.self_attn.o_proj.weight")) {
            LinearProjection::take_with_optional_bias(
                weights,
                &format!("{prefix}.self_attn.o_proj.weight"),
                &format!("{prefix}.self_attn.o_proj.bias"),
            )?
        } else {
            LinearProjection::take_with_optional_bias(
                weights,
                &format!("{prefix}.self_attn.c_proj.weight"),
                &format!("{prefix}.self_attn.c_proj.bias"),
            )?
        };

        let self_attn = LlamaAttention {
            num_heads: config.num_attention_heads as usize,
            num_kv_heads: config.num_key_value_heads as usize,
            head_dim: config.head_dim as usize,
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
        let gate_proj = LinearProjection::take_with_optional_bias(
            weights,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &format!("{prefix}.mlp.gate_proj.bias"),
        )?;
        let up_proj = LinearProjection::take_with_optional_bias(
            weights,
            &format!("{prefix}.mlp.up_proj.weight"),
            &format!("{prefix}.mlp.up_proj.bias"),
        )?;
        let down_proj = LinearProjection::take_with_optional_bias(
            weights,
            &format!("{prefix}.mlp.down_proj.weight"),
            &format!("{prefix}.mlp.down_proj.bias"),
        )?;

        let mlp = LlamaMLP {
            gate_proj,
            up_proj,
            down_proj,
            activation: config.hidden_activation.clone(),
            layer_idx,
        };

        // Build layer norms
        let input_layernorm = RMSNorm {
            weight: weights
                .get(&format!("{prefix}.input_layernorm.weight"))
                .ok_or_else(|| anyhow!("Missing input_layernorm weight"))?
                .shallow_clone(),
            eps: config.rms_norm_eps,
        };

        let post_attention_layernorm = RMSNorm {
            weight: weights
                .get(&format!("{prefix}.post_attention_layernorm.weight"))
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

        // Mandatory architectural dimensions — never silently defaulted (mirrors
        // #315's no-magic-number hardening in model_config.rs). A wrong layer or
        // head count silently truncates / mis-shapes a pipeline split (#314).
        let require_u64 = |field: &str| -> Result<u64> {
            json[field].as_u64().ok_or_else(|| {
                anyhow!(
                    "config.json is missing required field `{field}` \
                     (or it is not a non-negative integer)"
                )
            })
        };
        let num_attention_heads = require_u64("num_attention_heads")? as u32;
        let hidden_size = require_u64("hidden_size")? as u32;
        let num_hidden_layers = require_u64("num_hidden_layers")? as u32;
        if num_attention_heads == 0 || hidden_size == 0 || num_hidden_layers == 0 {
            return Err(anyhow!(
                "config.json has a zero architectural dimension \
                 (num_attention_heads={num_attention_heads}, hidden_size={hidden_size}, \
                 num_hidden_layers={num_hidden_layers})"
            ));
        }

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
            num_attention_heads,
            // num_key_value_heads legitimately defaults to num_attention_heads
            // (MHA) when absent — an HF-spec optional field, not a magic number.
            num_key_value_heads: json
                .get("num_key_value_heads")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(num_attention_heads),
            hidden_size,
            head_dim: json
                .get("head_dim")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or_else(|| {
                    // head_dim is optional in HF configs; derive it from the now-
                    // mandatory hidden_size / num_attention_heads.
                    let heads = num_attention_heads;
                    let hidden = hidden_size;
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
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(11008) as u32,
            max_position_embeddings: json["max_position_embeddings"].as_u64().unwrap_or(4096)
                as u32,
            rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as u32,
            original_vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as u32,  // Initially same
            num_hidden_layers,
            rope_theta: json
                .get("rope_theta")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(10000.0) as f32,
            rope_scaling: None,
            hidden_activation: json
                .get("hidden_activation")
                .and_then(|v| v.as_str())
                .unwrap_or("silu").to_owned(),
            query_pre_attn_scalar: json
                .get("query_pre_attn_scalar")
                .and_then(serde_json::Value::as_f64)
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
                    .unwrap_or("linear").to_owned(),
                factor: rope_scaling["factor"].as_f64().unwrap_or(8.0) as f32,
            });
        }

        // Check for Gemma3 specific configurations
        if config.vocab_size == 262144 {
            config.hidden_activation = "gelu_pytorch_tanh".to_owned();
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

    /// Core forward pass with optional delta injection (internal helper)
    ///
    /// Shared implementation for `forward_with_cache` (delta=None) and
    /// `forward_with_cache_and_delta` (delta=Some). Delta corrections are
    /// injected in each attention layer's Q/V projections before KV cache update.
    fn forward_with_cache_inner(
        &self,
        input: &Tensor,
        start_pos: usize,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        tracing::trace!("LlamaModel forward_with_cache_inner: input shape={:?}, start_pos={}, has_delta={}, config: hidden_size={}, num_layers={}",
                     input.size(), start_pos, delta.is_some(), self.config.hidden_size, self.layers.len());

        // Input should be token IDs with shape [batch_size, seq_len]
        let mut hidden_states = if let Some(embed) = &self.embed_tokens {
            // Convert token IDs to embeddings
            let input_shape = input.size();

            let batch_size = input_shape[0];
            let seq_len = if input_shape.len() > 1 {
                input_shape[1]
            } else {
                1
            };

            // Flatten input for embedding lookup (embedding expects 1D tensor)
            let flat_input = input.flatten(0, -1);

            // Perform embedding lookup using index_select
            let embeddings = embed.index_select(0, &flat_input);

            // Get the actual hidden size from the embedding result
            let emb_dims = embeddings.size();
            let hidden_size = emb_dims[emb_dims.len() - 1];

            // Reshape back to [batch_size, seq_len, hidden_size]
            let mut embeddings = embeddings.reshape([batch_size, seq_len, hidden_size]);

            // Scale embeddings by sqrt(hidden_size) for Gemma3
            if self.config.scale_embeddings {
                let scale = (hidden_size as f32).sqrt();
                let scale_tensor = Tensor::from_slice(&[scale])
                    .to_kind(embeddings.kind())
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
            Tensor::arange(seq_len, (tch::Kind::Int64, hidden_states.device()))
        } else {
            Tensor::arange_start(
                start_pos as i64,
                (start_pos + seq_len as usize) as i64,
                (tch::Kind::Int64, hidden_states.device()),
            )
        };

        // PERF: Lock KV cache ONCE before the layer loop (not 28 times inside!)
        let cache_guard = self.kv_cache.as_ref().map(|cache_ref| cache_ref.lock());

        for (idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden_states.shallow_clone();

            // Self-attention block with optional KV cache and delta injection
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
                        delta,
                    )
                }) {
                    result?
                } else {
                    layer
                        .self_attn
                        .forward(&hidden_states, Some(&position_ids), None, start_pos, delta)?
                }
            } else {
                layer
                    .self_attn
                    .forward(&hidden_states, Some(&position_ids), None, start_pos, delta)?
            };
            hidden_states = residual + attn_output;

            // FFN block with delta injection for MLP projections
            let residual = hidden_states.shallow_clone();
            hidden_states = layer.post_attention_layernorm.forward(&hidden_states)?;
            let ffn_output = layer.mlp.forward(&hidden_states, delta)?;
            hidden_states = residual + ffn_output;
        }

        // Final layer norm
        if let Some(norm) = &self.norm {
            hidden_states = norm.forward(&hidden_states)?;
        }

        // LM head
        if let Some(lm_head) = &self.lm_head {
            hidden_states = hidden_states.matmul(lm_head);
        } else if let Some(output_proj) = &self.lm_head_transposed {
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
        let original_vocab_size = self.config.original_vocab_size;
        let padded_vocab_size = self.config.vocab_size;
        if padded_vocab_size > original_vocab_size && original_vocab_size > 0 {
            let logits_shape = hidden_states.size();
            let actual_vocab_size = logits_shape[logits_shape.len() - 1] as u32;

            if actual_vocab_size == padded_vocab_size {
                let mask_start = original_vocab_size as i64;
                let mask_count = (padded_vocab_size - original_vocab_size) as i64;

                if mask_count > 0 {
                    let mask_values = Tensor::full(
                        [mask_count],
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

    /// Mask padded vocabulary columns to `-1e10` in-place on a logits tensor.
    ///
    /// Shared by the single-sequence and batched forward paths so both produce
    /// identical logits over the padded columns. No-op when the vocab was not
    /// padded. `logits` may be `[B, seq, vocab]` or `[rows, vocab]`.
    fn mask_padded_vocab(&self, logits: &Tensor) {
        let original_vocab_size = self.config.original_vocab_size;
        let padded_vocab_size = self.config.vocab_size;
        if padded_vocab_size <= original_vocab_size || original_vocab_size == 0 {
            return;
        }
        let logits_shape = logits.size();
        let actual_vocab_size = logits_shape[logits_shape.len() - 1] as u32;
        if actual_vocab_size != padded_vocab_size {
            return;
        }
        let mask_start = original_vocab_size as i64;
        let mask_count = (padded_vocab_size - original_vocab_size) as i64;
        if mask_count <= 0 {
            return;
        }
        let mask_values =
            Tensor::full([mask_count], -1e10_f64, (logits.kind(), logits.device()));
        match logits_shape.len() {
            3 => {
                for b in 0..logits_shape[0] {
                    for s in 0..logits_shape[1] {
                        let slice = logits.select(0, b).select(0, s);
                        slice.narrow(0, mask_start, mask_count).copy_(&mask_values);
                    }
                }
            }
            2 => {
                for i in 0..logits_shape[0] {
                    let slice = logits.select(0, i);
                    slice.narrow(0, mask_start, mask_count).copy_(&mask_values);
                }
            }
            _ => {
                logits.narrow(0, mask_start, mask_count).copy_(&mask_values);
            }
        }
    }

    /// Batched ragged forward for continuous decode (#329, Llama-only).
    ///
    /// `sequences[r]` = `(new_token_ids, start_pos, per-sequence KVCacheManager)`.
    /// All rows share the same query length `q = new_token_ids.len()` (v1 decode
    /// drives `q == 1`); each row brings its own absolute `start_pos` and isolated
    /// per-sequence KV cache (tenant isolation). `delta` is the single tenant
    /// delta shared by every row in the batch (the scheduler groups by delta).
    ///
    /// Returns stacked logits `[B, q, vocab]`. The result for row `r` is bit-for-
    /// bit equivalent (within fp tolerance) to `forward_with_cache_and_delta` run
    /// on that sequence alone — this is the #329 PR-1 correctness gate.
    ///
    /// Inherent impl behind the [`ModelOperations::forward_batched`] override.
    fn forward_batched_impl(
        &self,
        sequences: &mut [(Vec<i64>, usize, std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>)],
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        if batch_size == 0 {
            return Err(anyhow!("forward_batched: empty batch"));
        }
        let q_len = sequences[0].0.len();
        if q_len == 0 {
            return Err(anyhow!("forward_batched: empty sequence"));
        }
        if sequences.iter().any(|(toks, ..)| toks.len() != q_len) {
            return Err(anyhow!(
                "forward_batched: ragged query lengths are not supported in v1 (all rows must share q_len)"
            ));
        }
        let device = self.device;

        // Build the [B, q] input id tensor and embed it as one batch.
        let flat_ids: Vec<i64> = sequences.iter().flat_map(|(t, ..)| t.iter().copied()).collect();
        let input_ids = Tensor::from_slice(&flat_ids)
            .reshape([batch_size as i64, q_len as i64])
            .to_device(device);
        let mut hidden_states = self.embed_tokens(&input_ids)?;

        // Per-row absolute position ids [q] for RoPE, and post-update KV lengths.
        let start_positions: Vec<usize> = sequences.iter().map(|&(_, sp, _)| sp).collect();
        let position_ids: Vec<Tensor> = start_positions
            .iter()
            .map(|&sp| {
                Tensor::arange_start(
                    sp as i64,
                    (sp + q_len) as i64,
                    (tch::Kind::Int64, device),
                )
            })
            .collect();
        // After this step each row holds start_pos + q_len cached tokens.
        let kv_lens: Vec<i64> = start_positions.iter().map(|&sp| (sp + q_len) as i64).collect();
        let kv_max = kv_lens.iter().copied().max().unwrap_or(0);

        // Additive attention mask [B, 1, q, kv_max]: row r, query i (absolute pos
        // sp_r + i) attends to kv col j iff j <= sp_r + i AND j < kv_len_r. Padding
        // columns (j >= kv_len_r) and future columns are set to a large negative.
        let neg = -1.0e9_f64;
        let mask = Tensor::zeros(
            [batch_size as i64, 1, q_len as i64, kv_max],
            (tch::Kind::Float, device),
        );
        {
            // Column index row vector [kv_max].
            let cols = Tensor::arange(kv_max, (tch::Kind::Int64, device));
            for r in 0..batch_size {
                let sp = start_positions[r] as i64;
                let kv_len = kv_lens[r];
                for i in 0..q_len as i64 {
                    let allowed_upto = sp + i; // inclusive causal bound
                    // valid = (cols <= allowed_upto) & (cols < kv_len)
                    let valid = cols.le(allowed_upto).logical_and(&cols.lt(kv_len));
                    let row_mask = valid
                        .logical_not()
                        .to_kind(tch::Kind::Float)
                        * neg; // 0 where valid, neg where invalid
                    mask.select(0, r as i64)
                        .select(0, 0)
                        .select(0, i)
                        .copy_(&row_mask);
                }
            }
        }

        // Lock all per-sequence cache managers for the whole layer loop (one lock
        // each, mirroring the single-sequence path's lock-once strategy).
        let guards: Vec<_> = sequences.iter().map(|(_, _, c)| c.lock()).collect();

        let position_refs: Vec<Tensor> = position_ids; // already per-row [q]

        for (idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden_states.shallow_clone();
            hidden_states = layer.input_layernorm.forward(&hidden_states)?;

            // Collect this layer's per-row LayerKVCache mutable refs. DashMap's
            // get_mut borrows the manager, so we briefly take ownership of each
            // layer cache via with_layer_cache-equivalent direct access.
            let attn_output = {
                let mut layer_caches: Vec<
                    dashmap::mapref::one::RefMut<usize, crate::runtime::kv_cache::LayerKVCache>,
                > = Vec::with_capacity(batch_size);
                for g in guards.iter() {
                    let cell = g
                        .layer_cache_ref(idx)
                        .ok_or_else(|| anyhow!("forward_batched: missing layer cache {idx}"))?;
                    layer_caches.push(cell);
                }
                let mut cache_refs: Vec<&mut crate::runtime::kv_cache::LayerKVCache> =
                    layer_caches.iter_mut().map(|c| &mut **c).collect();
                layer.self_attn.forward_batched(
                    &hidden_states,
                    &position_refs,
                    &mut cache_refs,
                    &start_positions,
                    &mask,
                    delta,
                )?
            };
            hidden_states = residual + attn_output;

            let residual = hidden_states.shallow_clone();
            hidden_states = layer.post_attention_layernorm.forward(&hidden_states)?;
            let ffn_output = layer.mlp.forward(&hidden_states, delta)?;
            hidden_states = residual + ffn_output;
        }
        drop(guards);

        hidden_states = self.apply_final_norm(&hidden_states)?;
        let logits = self.lm_head(&hidden_states)?;
        self.mask_padded_vocab(&logits);
        Ok(logits)
    }

    /// Run global decoder layers `[range.start..range.end)` — the 2b pipeline
    /// layer-range runner (#314). See the trait docs for the stage contract.
    ///
    /// Layers are remapped to local `self.layers` indices via `layer_offset`; the
    /// per-layer KV cache is indexed locally (sized to the owned count). The lone
    /// cross-device copy is `hidden.to_device(next)`, inserted only when the next
    /// layer's mapped device differs from where `hidden` currently lives — never
    /// within a stage, never when source == dest (the unsplit map is a no-op).
    fn forward_layers_inner(
        &self,
        hidden: &Tensor,
        range: std::ops::Range<usize>,
        start_pos: usize,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let num_global = self.config.num_hidden_layers as usize;
        if range.end > num_global || range.start >= range.end {
            return Err(anyhow!(
                "forward_layers: invalid global range {range:?} for {num_global} layers"
            ));
        }
        // The owned window must contain the requested range.
        let owned = self.layer_offset..self.layer_offset + self.layers.len();
        if range.start < owned.start || range.end > owned.end {
            return Err(anyhow!(
                "forward_layers: range {range:?} not within this stage's owned layers {owned:?}"
            ));
        }

        let mut hidden_states = hidden.shallow_clone();

        // position_ids recomputed from start_pos + seq (never carried across a
        // boundary). Built on the *input* device (the first owned layer's), so a
        // moved first layer still matches at SDPA.
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
        // position_ids must follow `hidden` across device boundaries; track it.
        let mut position_ids = position_ids;

        let cache_guard = self.kv_cache.as_ref().map(|cache_ref| cache_ref.lock());

        for g in range {
            let local_idx = g - self.layer_offset;
            let layer = &self.layers[local_idx];

            // The single cross-device transfer: only when this layer's device
            // differs from where `hidden` currently is (zero-copy otherwise).
            let target = self.device_map.device_for(g);
            if hidden_states.device() != target {
                hidden_states = hidden_states.to_device(target);
                position_ids = position_ids.to_device(target);
            }

            let residual = hidden_states.shallow_clone();
            hidden_states = layer.input_layernorm.forward(&hidden_states)?;

            let attn_output = if let Some(ref cache_manager) = cache_guard {
                if let Some(result) = cache_manager.with_layer_cache(local_idx, |layer_cache| {
                    layer.self_attn.forward(
                        &hidden_states,
                        Some(&position_ids),
                        Some(layer_cache),
                        start_pos,
                        delta,
                    )
                }) {
                    result?
                } else {
                    layer
                        .self_attn
                        .forward(&hidden_states, Some(&position_ids), None, start_pos, delta)?
                }
            } else {
                layer
                    .self_attn
                    .forward(&hidden_states, Some(&position_ids), None, start_pos, delta)?
            };
            hidden_states = residual + attn_output;

            let residual = hidden_states.shallow_clone();
            hidden_states = layer.post_attention_layernorm.forward(&hidden_states)?;
            let ffn_output = layer.mlp.forward(&hidden_states, delta)?;
            hidden_states = residual + ffn_output;
        }

        Ok(hidden_states)
    }

    /// Training-path sibling of [`Self::forward_layers_inner`] — the cross-device
    /// autograd primitive for TTT-on-split (#316). See the trait
    /// `forward_layers_train` docs for the contract.
    ///
    /// Identical layer loop to the inference runner with three deliberate
    /// differences: **no KV cache** (full causal attention, computed fresh from
    /// the whole context), **`start_pos = 0`** (`position_ids = 0..seq`), and the
    /// autograd graph is left intact (no `no_grad` guard — the caller wraps this
    /// in `with_grad`). The lone cross-device `hidden.to_device(next)` carries the
    /// autograd graph across the boundary, so `loss.backward()` materializes grads
    /// on each parameter's own device.
    fn forward_layers_train_inner(
        &self,
        hidden: &Tensor,
        range: std::ops::Range<usize>,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let num_global = self.config.num_hidden_layers as usize;
        if range.end > num_global || range.start >= range.end {
            return Err(anyhow!(
                "forward_layers_train: invalid global range {range:?} for {num_global} layers"
            ));
        }
        let owned = self.layer_offset..self.layer_offset + self.layers.len();
        if range.start < owned.start || range.end > owned.end {
            return Err(anyhow!(
                "forward_layers_train: range {range:?} not within this stage's owned layers {owned:?}"
            ));
        }

        let mut hidden_states = hidden.shallow_clone();

        // Training path: start_pos is always 0 → position_ids = 0..seq, built on
        // the input device and moved alongside `hidden` at each device boundary.
        let seq_len = hidden_states.size()[1];
        let mut position_ids =
            Tensor::arange(seq_len, (tch::Kind::Int64, hidden_states.device()));

        // No KV cache in the training path: full causal attention is recomputed
        // over the entire context every step (compute_attention applies the tril
        // mask when no cache is present).
        for g in range {
            let local_idx = g - self.layer_offset;
            let layer = &self.layers[local_idx];

            // The single cross-device transfer (autograd-transparent): only when
            // this layer's mapped device differs from where `hidden` currently is.
            let target = self.device_map.device_for(g);
            if hidden_states.device() != target {
                hidden_states = hidden_states.to_device(target);
                position_ids = position_ids.to_device(target);
            }

            let residual = hidden_states.shallow_clone();
            hidden_states = layer.input_layernorm.forward(&hidden_states)?;
            let attn_output = layer.self_attn.forward(
                &hidden_states,
                Some(&position_ids),
                None, // no KV cache (training)
                0,    // start_pos = 0 (training)
                delta,
            )?;
            hidden_states = residual + attn_output;

            let residual = hidden_states.shallow_clone();
            hidden_states = layer.post_attention_layernorm.forward(&hidden_states)?;
            let ffn_output = layer.mlp.forward(&hidden_states, delta)?;
            hidden_states = residual + ffn_output;
        }

        Ok(hidden_states)
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
        self.forward_with_cache_inner(input, start_pos, None)
    }

    fn forward_with_cache_and_delta(
        &self,
        input: &Tensor,
        start_pos: usize,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        self.forward_with_cache_inner(input, start_pos, delta)
    }

    fn forward_batched(
        &self,
        sequences: &mut [(
            Vec<i64>,
            usize,
            std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>,
        )],
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        self.forward_batched_impl(sequences, delta)
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
                let cache_manager = cache_ref.lock();
                if let Some(result) = cache_manager.with_layer_cache(idx, |layer_cache| {
                    layer.self_attn.forward(
                        &hidden_states,
                        Some(&position_ids),
                        Some(layer_cache),
                        start_pos,
                        None,
                    )
                }) {
                    result?
                } else {
                    layer
                        .self_attn
                        .forward(&hidden_states, Some(&position_ids), None, start_pos, None)?
                }
            } else {
                layer
                    .self_attn
                    .forward(&hidden_states, Some(&position_ids), None, start_pos, None)?
            };
            hidden_states = residual + attn_output;

            // FFN block
            let residual = hidden_states.shallow_clone();
            hidden_states = layer.post_attention_layernorm.forward(&hidden_states)?;
            let ffn_output = layer.mlp.forward(&hidden_states, None)?;
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
            let actual_vocab_size = logits_shape[logits_shape.len() - 1] as u32;

            if actual_vocab_size == padded_vocab_size {
                let mask_start = original_vocab_size as i64;
                let mask_count = (padded_vocab_size - original_vocab_size) as i64;

                if mask_count > 0 {
                    let mask_values = Tensor::full(
                        [mask_count],
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

    fn decode_layer(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        _attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        _past_kv: Option<&crate::runtime::kv_cache::LayerKVCache>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let layer = self.layers.get(layer_idx).ok_or_else(|| {
            anyhow!(
                "Layer index {} out of range (model has {} layers)",
                layer_idx,
                self.layers.len()
            )
        })?;

        let residual = hidden_states.shallow_clone();

        // Self-attention block
        let normed = layer.input_layernorm.forward(hidden_states)?;

        // Handle KV cache per layer
        let attn_output = if let Some(cache_ref) = self.kv_cache.as_ref() {
            let cache_manager = cache_ref.lock();
            if let Some(result) = cache_manager.with_layer_cache(layer_idx, |layer_cache| {
                layer.self_attn.forward(
                    &normed,
                    position_ids,
                    Some(layer_cache),
                    0, // start_pos handled by cache
                    None,
                )
            }) {
                result?
            } else {
                layer.self_attn.forward(&normed, position_ids, None, 0, None)?
            }
        } else {
            layer.self_attn.forward(&normed, position_ids, None, 0, None)?
        };

        let hidden_states = residual + attn_output;

        // FFN block
        let residual = hidden_states.shallow_clone();
        let normed = layer.post_attention_layernorm.forward(&hidden_states)?;
        let ffn_output = layer.mlp.forward(&normed, None)?;
        let hidden_states = residual + ffn_output;

        // No separate KV return — cache is managed internally by LlamaAttention
        Ok((hidden_states, None))
    }

    fn decode_layer_with_delta(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        _attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        _past_kv: Option<&crate::runtime::kv_cache::LayerKVCache>,
        delta: &crate::training::TenantDelta,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let layer = self.layers.get(layer_idx).ok_or_else(|| {
            anyhow!(
                "Layer index {} out of range (model has {} layers)",
                layer_idx,
                self.layers.len()
            )
        })?;

        let residual = hidden_states.shallow_clone();

        // Self-attention block with delta injection (no KV cache on training path)
        let normed = layer.input_layernorm.forward(hidden_states)?;
        let attn_output = layer.self_attn.forward(&normed, position_ids, None, 0, Some(delta))?;
        let hidden_states = residual + attn_output;

        // FFN block with delta injection for MLP projections
        let residual = hidden_states.shallow_clone();
        let normed = layer.post_attention_layernorm.forward(&hidden_states)?;
        let ffn_output = layer.mlp.forward(&normed, Some(delta))?;
        let hidden_states = residual + ffn_output;

        Ok((hidden_states, None))
    }

    fn forward_layers(
        &self,
        hidden: &Tensor,
        range: std::ops::Range<usize>,
        start_pos: usize,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        self.forward_layers_inner(hidden, range, start_pos, delta)
    }

    fn forward_layers_train(
        &self,
        hidden: &Tensor,
        range: std::ops::Range<usize>,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        self.forward_layers_train_inner(hidden, range, delta)
    }

    fn apply_final_norm(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if let Some(norm) = &self.norm {
            norm.forward(hidden_states)
        } else {
            Ok(hidden_states.shallow_clone())
        }
    }

    fn lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if let Some(lm_head) = &self.lm_head {
            // Cast weight UP to match hidden_states dtype instead of casting hidden_states DOWN.
            // During training, hidden_states may be f32 for gradient precision — casting it
            // down to bf16 destroys gradients. Casting the (frozen) weight up is a no-op
            // during inference where both are already bf16.
            let lm_head = lm_head.to_kind(hidden_states.kind());
            Ok(hidden_states.f_matmul(&lm_head)
                .map_err(|e| anyhow!("lm_head matmul failed: {}", e))?)
        } else if let Some(output_proj) = &self.lm_head_transposed {
            let output_proj = output_proj.to_kind(hidden_states.kind());
            let hs_shape = hidden_states.size();
            if hs_shape.len() == 3 {
                let (batch_size, seq_len, hidden_size) = (hs_shape[0], hs_shape[1], hs_shape[2]);
                let hidden_2d = hidden_states.reshape([batch_size * seq_len, hidden_size]);
                let logits_2d = hidden_2d.f_matmul(&output_proj)
                    .map_err(|e| anyhow!("lm_head transposed matmul failed: {}", e))?;
                Ok(logits_2d.reshape([batch_size, seq_len, -1]))
            } else {
                Ok(hidden_states.f_matmul(&output_proj)
                    .map_err(|e| anyhow!("lm_head transposed matmul failed: {}", e))?)
            }
        } else {
            Err(anyhow!("No LM head or embedding weights found"))
        }
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers as usize
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // ============================================================================
    // KV Cache Management (trait implementation)
    // ============================================================================

    fn clear_kv_cache(&self) {
        if let Some(cache_ref) = self.kv_cache.as_ref() {
            let cache_manager = cache_ref.lock();
            cache_manager.clear_all();
        }
    }

    fn kv_cache_memory_usage(&self) -> usize {
        self.kv_cache
            .as_ref()
            .map(|cache_ref| cache_ref.lock().memory_usage())
            .unwrap_or(0)
    }

    fn set_kv_cache(
        &mut self,
        cache: std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>,
    ) {
        self.kv_cache = Some(cache);
    }

    fn get_kv_cache(
        &self,
    ) -> Option<std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>> {
        self.kv_cache.clone()
    }

    fn take_kv_cache(
        &mut self,
    ) -> Option<std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>> {
        self.kv_cache.take()
    }
}

// KV cache methods are now defined in the ModelOperations trait implementation above

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod pipeline_tests {
    use super::*;
    use crate::runtime::device_pool::LayerDeviceMap;
    use crate::runtime::KVQuantType;

    const HIDDEN: i64 = 16;
    const HEADS: i64 = 2;
    const HEAD_DIM: i64 = 8; // HEADS * HEAD_DIM == HIDDEN
    const INTER: i64 = 32;
    const VOCAB: i64 = 48; // multiple of 64? no → padded to 64; both paths pad identically
    const LAYERS: usize = 4;

    fn tiny_config() -> LlamaConfig {
        LlamaConfig {
            version: 2,
            num_attention_heads: HEADS as u32,
            num_key_value_heads: HEADS as u32,
            hidden_size: HIDDEN as u32,
            head_dim: HEAD_DIM as u32,
            intermediate_size: INTER as u32,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            vocab_size: VOCAB as u32,
            original_vocab_size: VOCAB as u32,
            num_hidden_layers: LAYERS as u32,
            rope_theta: 10000.0,
            rope_scaling: None,
            hidden_activation: "silu".to_owned(),
            query_pre_attn_scalar: None,
            use_qk_norm: false,
            scale_embeddings: false,
            layer_types: vec![],
            rope_local_base_freq: None,
        }
    }

    /// Deterministic small weight set on CPU/f32 so the two construction paths are
    /// numerically comparable. HF stores projections as `[out, in]`.
    ///
    /// Uses a fixed, RNG-free pattern (a bounded sin of the flat index) rather
    /// than `Tensor::randn`: tests run in parallel and `manual_seed` is global, so
    /// two `randn`-based builds could interleave and diverge. This makes every
    /// call byte-identical regardless of scheduling.
    fn tiny_weights() -> HashMap<String, Tensor> {
        let opt = (DType::Float, Device::Cpu);
        let mut w = HashMap::new();
        let randn = |dims: &[i64]| {
            let n: i64 = dims.iter().product();
            // Deterministic small values in roughly [-0.05, 0.05].
            (Tensor::arange(n, opt) * 0.017).sin().reshape(dims) * 0.05
        };

        w.insert("model.embed_tokens.weight".to_owned(), randn(&[VOCAB, HIDDEN]));
        w.insert("model.norm.weight".to_owned(), Tensor::ones([HIDDEN], opt));
        w.insert("lm_head.weight".to_owned(), randn(&[VOCAB, HIDDEN]));

        for i in 0..LAYERS {
            let p = format!("model.layers.{i}");
            // attention projections: q/k/v are [heads*head_dim, hidden]; o is [hidden, heads*head_dim]
            w.insert(format!("{p}.self_attn.q_proj.weight"), randn(&[HEADS * HEAD_DIM, HIDDEN]));
            w.insert(format!("{p}.self_attn.k_proj.weight"), randn(&[HEADS * HEAD_DIM, HIDDEN]));
            w.insert(format!("{p}.self_attn.v_proj.weight"), randn(&[HEADS * HEAD_DIM, HIDDEN]));
            w.insert(format!("{p}.self_attn.o_proj.weight"), randn(&[HIDDEN, HEADS * HEAD_DIM]));
            // mlp
            w.insert(format!("{p}.mlp.gate_proj.weight"), randn(&[INTER, HIDDEN]));
            w.insert(format!("{p}.mlp.up_proj.weight"), randn(&[INTER, HIDDEN]));
            w.insert(format!("{p}.mlp.down_proj.weight"), randn(&[HIDDEN, INTER]));
            // norms
            w.insert(format!("{p}.input_layernorm.weight"), Tensor::ones([HIDDEN], opt));
            w.insert(format!("{p}.post_attention_layernorm.weight"), Tensor::ones([HIDDEN], opt));
        }
        w
    }

    fn whole_model() -> LlamaModel {
        let mut w = tiny_weights();
        LlamaModel::from_weights_with_config(&mut w, tiny_config(), &Device::Cpu, DType::Float, KVQuantType::None)
            .unwrap()
    }

    /// Orchestrate the pipeline path on a model the way an engine would:
    /// embed → forward_layers(0..N) → final norm → lm_head.
    fn orchestrated_logits(m: &LlamaModel, input: &Tensor) -> Tensor {
        let emb = m.embed_tokens(input).unwrap();
        let h = m.forward_layers(&emb, 0..m.num_layers(), 0, None).unwrap();
        let h = m.apply_final_norm(&h).unwrap();
        m.lm_head(&h).unwrap()
    }

    #[test]
    fn forward_layers_full_range_matches_whole_model_forward() {
        // An all-CPU LayerDeviceMap means forward_layers(0..N) must equal the
        // whole-model single-device forward (the byte-identity equivalence test).
        let input = Tensor::from_slice(&[1i64, 5, 9, 2]).reshape([1, 4]);

        // Reference: whole-model forward (embed→loop→norm→lm_head w/ vocab mask).
        let reference = whole_model();
        let ref_logits = reference.forward_with_cache(&input, 0).unwrap();

        // Pipeline path on a fresh model (independent KV), full range over a
        // single-device map. Compare on the ORIGINAL vocab columns (the whole-
        // model path additionally masks padded columns; lm_head() does not).
        let piped = whole_model();
        let pipe_logits = orchestrated_logits(&piped, &input);

        let orig = reference.config.original_vocab_size as i64;
        let ref_c = ref_logits.narrow(2, 0, orig);
        let pipe_c = pipe_logits.narrow(2, 0, orig);
        let max_diff = (&ref_c - &pipe_c).abs().max().double_value(&[]);
        assert!(
            ref_c.allclose(&pipe_c, 1e-4, 1e-4, false),
            "forward_layers(0..N) over a single-device map must equal whole-model forward \
             (max_diff={max_diff}, ref_shape={:?})",
            ref_c.size()
        );
        // Sanity: the device map for a whole model is single-device (zero-copy).
        assert!(piped.device_map.is_single_device());
        assert_eq!(piped.layer_offset, 0);
    }

    #[test]
    fn staged_split_equals_whole_model() {
        // Build TWO stages over an all-CPU 2-way split and run them as a pipeline:
        // stage0: embed → forward_layers(0..b); stage1: forward_layers(b..N) →
        // norm → lm_head. Result must equal the whole-model forward. This
        // exercises the global↔local remap, is_first/is_last gating, and the
        // (zero, since same-device) boundary copy.
        let input = Tensor::from_slice(&[3i64, 7, 1, 4]).reshape([1, 4]);

        let reference = whole_model();
        let ref_logits = reference.forward_with_cache(&input, 0).unwrap();

        // Single CPU device but split the layer RANGE across two stages.
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let split = LAYERS / 2;

        let mut w0 = tiny_weights();
        let stage0 = LlamaModel::stage_from_weights_with_config(
            &mut w0, tiny_config(), &map, 0..split, DType::Float, KVQuantType::None,
        )
        .unwrap();
        assert_eq!(stage0.layer_offset, 0);
        assert!(stage0.embed_tokens.is_some(), "first stage keeps embed");
        assert!(stage0.norm.is_none() && stage0.lm_head.is_none(), "first stage has no head");

        let mut w1 = tiny_weights();
        let stage1 = LlamaModel::stage_from_weights_with_config(
            &mut w1, tiny_config(), &map, split..LAYERS, DType::Float, KVQuantType::None,
        )
        .unwrap();
        assert_eq!(stage1.layer_offset, split);
        assert!(stage1.embed_tokens.is_none(), "last stage has no embed");
        assert!(stage1.norm.is_some(), "last stage keeps final norm");

        // Drive the pipeline.
        let emb = stage0.embed_tokens(&input).unwrap();
        let h0 = stage0.forward_layers(&emb, 0..split, 0, None).unwrap();
        let h1 = stage1.forward_layers(&h0, split..LAYERS, 0, None).unwrap();
        let h1 = stage1.apply_final_norm(&h1).unwrap();
        let logits = stage1.lm_head(&h1).unwrap();

        let orig = reference.config.original_vocab_size as i64;
        let ref_c = ref_logits.narrow(2, 0, orig);
        let pipe_c = logits.narrow(2, 0, orig);
        // 1e-4: the lm_head() trait casts the weight and uses f_matmul; the whole-
        // model path uses plain matmul — float reassociation, not a logic diff.
        let max_diff = (&ref_c - &pipe_c).abs().max().double_value(&[]);
        assert!(
            ref_c.allclose(&pipe_c, 1e-4, 1e-4, false),
            "two-stage split (same device) must equal whole-model forward (max_diff={max_diff})"
        );
        // The last stage built an explicit lm_head (not the tied transpose).
        assert!(stage1.lm_head_transposed.is_none());
    }

    #[test]
    fn forward_layers_rejects_out_of_window_range() {
        // A stage that owns [2..4) must reject a range outside its window.
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let mut w = tiny_weights();
        let stage = LlamaModel::stage_from_weights_with_config(
            &mut w, tiny_config(), &map, 2..LAYERS, DType::Float, KVQuantType::None,
        )
        .unwrap();
        let emb = Tensor::randn([1, 3, HIDDEN], (DType::Float, Device::Cpu));
        assert!(stage.forward_layers(&emb, 0..2, 0, None).is_err(), "range below window");
        assert!(stage.forward_layers(&emb, 2..LAYERS, 0, None).is_ok(), "owned range ok");
    }

    #[test]
    fn boundary_copy_skipped_when_same_device() {
        // With a single-device map, no layer is a boundary, so forward_layers does
        // zero `to_device` calls. We can only assert the predicate directly here
        // (tch gives no copy counter), which the device-map test also covers.
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        for g in 1..LAYERS {
            assert!(!map.is_boundary(Device::Cpu, g));
        }
    }

    #[test]
    fn stage_loader_rejects_missing_required_weights() {
        // Last stage with no norm weight must error (M-LOAD seam #1 gating).
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let mut w = tiny_weights();
        w.remove("model.norm.weight");
        let result = LlamaModel::stage_from_weights_with_config(
            &mut w, tiny_config(), &map, 0..LAYERS, DType::Float, KVQuantType::None,
        );
        match result {
            Ok(_) => panic!("last stage without norm weight must error"),
            Err(e) => assert!(e.to_string().contains("norm"), "got: {e}"),
        }
    }

    #[test]
    fn parse_config_requires_core_dims() {
        // #315 follow-up: mandatory architectural dims, no silent magic numbers.
        let ok = r#"{"num_attention_heads":2,"hidden_size":16,"num_hidden_layers":4}"#;
        assert!(LlamaModel::parse_config(ok).is_ok());
        let missing = r#"{"hidden_size":16,"num_hidden_layers":4}"#;
        let err = LlamaModel::parse_config(missing).unwrap_err();
        assert!(err.to_string().contains("num_attention_heads"), "got: {err}");
    }

    // ========================================================================
    // #316 — TTT-on-split cross-device autograd equivalence (CPU-verifiable).
    // ========================================================================

    use crate::training::tenant_delta::{TenantDelta, TenantDeltaConfig};

    /// Build a per-layer q_proj/v_proj LoRA delta on CPU with a deterministic,
    /// RNG-free A **and** non-zero B so that gradients w.r.t. BOTH A and B are
    /// non-trivial (B is zero-initialized by default → dL/dA would be zero).
    fn tiny_delta() -> TenantDelta {
        let mut dims = std::collections::HashMap::new();
        dims.insert("q_proj".to_owned(), (HIDDEN as usize, HIDDEN as usize));
        dims.insert("v_proj".to_owned(), (HIDDEN as usize, HIDDEN as usize));
        let cfg = TenantDeltaConfig {
            rank: 2,
            alpha: 2.0,
            target_modules: vec!["q_proj".to_owned(), "v_proj".to_owned()],
            ..Default::default()
        };
        let delta = TenantDelta::new(&cfg, &dims, Device::Cpu, LAYERS).unwrap();
        // Seed deterministic, non-zero A and B in-place (no_grad: weight init, not
        // graph). Seed by a STABLE per-key offset — HashMap iteration order is not
        // deterministic, so two independent `tiny_delta()` builds (whole vs split)
        // must agree key-for-key, not enumeration-index-for-index.
        let key_offset = |key: &str| -> i64 {
            key.bytes().map(|b| b as i64).sum::<i64>() % 17
        };
        let _g = tch::no_grad_guard();
        for (k, a) in delta.lora_a.iter() {
            let n: i64 = a.size().iter().product();
            let vals = (Tensor::arange(n, (DType::Float, Device::Cpu)) + key_offset(k)).sin() * 0.1;
            // copy_ needs &mut; shallow_clone shares storage so the delta param is mutated.
            a.shallow_clone().copy_(&vals.reshape(a.size()));
        }
        for (k, b) in delta.lora_b.iter() {
            let n: i64 = b.size().iter().product();
            let vals = (Tensor::arange(n, (DType::Float, Device::Cpu)) + key_offset(k) + 7).cos() * 0.1;
            b.shallow_clone().copy_(&vals.reshape(b.size()));
        }
        delta
    }

    /// Sum of squared L2 grad norms over all delta params, plus a per-key map.
    fn grad_snapshot(delta: &TenantDelta) -> std::collections::HashMap<String, f64> {
        let mut out = std::collections::HashMap::new();
        for (k, a) in &delta.lora_a {
            assert!(a.grad().defined(), "A grad undefined for {k}");
            out.insert(format!("A:{k}"), a.grad().norm().double_value(&[]));
        }
        for (k, b) in &delta.lora_b {
            assert!(b.grad().defined(), "B grad undefined for {k}");
            out.insert(format!("B:{k}"), b.grad().norm().double_value(&[]));
        }
        out
    }

    /// The core correctness guardrail (#316): a TTT/training step (forward_layers_train
    /// → NTP loss → backward) over a two-stage CPU split must produce the SAME delta
    /// gradients as the whole-model training forward, within ~1e-4. A wrong
    /// cross-device autograd hookup or mis-placed grad would diverge.
    #[test]
    fn ttt_split_autograd_matches_whole_model_grads() {
        use crate::training::pipeline::{compute_ntp_loss_split, TrainStage};

        let input = Tensor::from_slice(&[3i64, 7, 1, 4, 9, 2]).reshape([1, 6]);

        // --- (a) whole-model training forward over a single-device map ---
        let whole = whole_model();
        let whole_delta = tiny_delta();
        let whole_stage = [TrainStage { model: &whole, range: 0..LAYERS }];
        let loss_whole = compute_ntp_loss_split(&whole_stage, &input, Some(&whole_delta)).unwrap();
        loss_whole.backward();
        let whole_grads = grad_snapshot(&whole_delta);
        whole_delta.zero_grad();

        // --- (b) two-stage split over an all-CPU map (exercises the boundary
        // to_device autograd transparency + global↔local remap) ---
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let split = LAYERS / 2;
        let mut w0 = tiny_weights();
        let stage0 = LlamaModel::stage_from_weights_with_config(
            &mut w0, tiny_config(), &map, 0..split, DType::Float, KVQuantType::None,
        ).unwrap();
        let mut w1 = tiny_weights();
        let stage1 = LlamaModel::stage_from_weights_with_config(
            &mut w1, tiny_config(), &map, split..LAYERS, DType::Float, KVQuantType::None,
        ).unwrap();
        let split_delta = tiny_delta();
        let stages = [
            TrainStage { model: &stage0, range: 0..split },
            TrainStage { model: &stage1, range: split..LAYERS },
        ];
        let loss_split = compute_ntp_loss_split(&stages, &input, Some(&split_delta)).unwrap();
        loss_split.backward();
        let split_grads = grad_snapshot(&split_delta);

        // Losses must match (forward equivalence).
        let lw = loss_whole.double_value(&[]);
        let ls = loss_split.double_value(&[]);
        assert!((lw - ls).abs() < 1e-4, "loss diverged: whole={lw} split={ls}");

        // Per-key gradient norms must match within tolerance (backward equivalence).
        assert_eq!(whole_grads.len(), split_grads.len());
        for (k, &gw) in &whole_grads {
            let gs = *split_grads.get(k).unwrap_or_else(|| panic!("missing grad {k}"));
            assert!(
                (gw - gs).abs() <= 1e-4 + 1e-4 * gw.abs(),
                "grad norm diverged for {k}: whole={gw} split={gs}"
            );
            // A non-trivial gradient must actually flow (guards against a silently
            // detached path that would make both sides spuriously equal at 0).
            assert!(gw > 0.0, "grad for {k} is zero — autograd path not exercised");
        }
    }

    // ========================================================================
    // #329 PR-1 — batched ragged decode == serial decode (CPU merge gate).
    // ========================================================================

    use crate::runtime::kv_cache::KVCacheManager;
    use std::sync::Arc;
    use parking_lot::Mutex;

    fn fresh_cache() -> Arc<Mutex<KVCacheManager>> {
        Arc::new(Mutex::new(KVCacheManager::new(LAYERS, 64, KVQuantType::None)))
    }

    /// Decode logits (last position) for one sequence run through the trusted
    /// batch=1 path: prefill the prompt, then one decode step for `next_tok`.
    fn serial_decode_logits(prompt: &[i64], next_tok: i64) -> Tensor {
        let model = whole_model();
        let cache = Arc::new(Mutex::new(KVCacheManager::new(LAYERS, 64, KVQuantType::None)));
        // Inject a fresh isolated cache so the batch=1 path actually caches.
        let model = {
            let mut m = model;
            <LlamaModel as ModelOperations>::set_kv_cache(&mut m, cache);
            m
        };
        // Prefill prompt at pos 0.
        let prompt_t = Tensor::from_slice(prompt).reshape([1, prompt.len() as i64]);
        let _ = model.forward_with_cache(&prompt_t, 0).unwrap();
        // Decode one token at pos = prompt.len().
        let dec = Tensor::from_slice(&[next_tok]).reshape([1, 1]);
        let logits = model.forward_with_cache(&dec, prompt.len()).unwrap();
        // [1, 1, vocab] -> [vocab]
        logits.select(0, 0).select(0, 0)
    }

    #[test]
    fn batched_ragged_decode_matches_serial() {
        // Three sequences with DIFFERENT prompt lengths (ragged KV histories) +
        // one decode token each. The batched decode step (B=3, q=1, ragged kv)
        // must match each sequence's serially-run decode logits.
        let prompts: [&[i64]; 3] = [&[1, 5, 9, 2], &[7, 3], &[4, 8, 6, 1, 0]];
        let next: [i64; 3] = [11, 13, 6];

        // Reference (serial, trusted batch=1 path).
        let ref_logits: Vec<Tensor> = prompts
            .iter()
            .zip(&next)
            .map(|(p, &t)| serial_decode_logits(p, t))
            .collect();

        // Batched path: one model, per-sequence isolated caches.
        let model = whole_model();
        let mut seqs: Vec<(Vec<i64>, usize, Arc<Mutex<KVCacheManager>>)> = prompts
            .iter()
            .map(|_| (Vec::new(), 0usize, fresh_cache()))
            .collect();

        // Prefill each sequence's cache individually (batch=1 calls to the batched
        // path — ragged q across rows is not mixed in v1).
        for (i, p) in prompts.iter().enumerate() {
            let mut one = vec![(p.to_vec(), 0usize, seqs[i].2.clone())];
            let _ = model.forward_batched(&mut one, None).unwrap();
        }

        // One batched decode step over the ragged histories.
        for (i, &t) in next.iter().enumerate() {
            seqs[i].0 = vec![t];
            seqs[i].1 = prompts[i].len();
        }
        let batched = model.forward_batched(&mut seqs, None).unwrap(); // [3, 1, vocab]

        let orig = model.config.original_vocab_size as i64;
        for (i, ref_l) in ref_logits.iter().enumerate() {
            let b_l = batched.select(0, i as i64).select(0, 0); // [vocab]
            let r_c = ref_l.narrow(0, 0, orig);
            let b_c = b_l.narrow(0, 0, orig);
            let max_diff = (&r_c - &b_c).abs().max().double_value(&[]);
            assert!(
                r_c.allclose(&b_c, 1e-4, 1e-4, false),
                "batched decode row {i} diverged from serial (max_diff={max_diff})"
            );
            // Argmax (sampled token) must agree — the user-visible invariant.
            assert_eq!(
                r_c.argmax(0, false).int64_value(&[]),
                b_c.argmax(0, false).int64_value(&[]),
                "row {i} argmax token differs",
            );
        }
    }

    #[test]
    fn batched_single_row_matches_serial_prefill() {
        // A batch of ONE through forward_batched must equal the batch=1 reference
        // (guards the padded-dense KV + explicit-mask path against the inline tril).
        let prompt: [i64; 4] = [1, 5, 9, 2];
        let next = 11i64;
        let ref_l = serial_decode_logits(&prompt, next);

        let model = whole_model();
        let cache = fresh_cache();
        let mut prefill = vec![(prompt.to_vec(), 0usize, cache.clone())];
        let _ = model.forward_batched(&mut prefill, None).unwrap();
        let mut dec = vec![(vec![next], prompt.len(), cache)];
        let logits = model.forward_batched(&mut dec, None).unwrap();
        let b_l = logits.select(0, 0).select(0, 0);

        let orig = model.config.original_vocab_size as i64;
        let r_c = ref_l.narrow(0, 0, orig);
        let b_c = b_l.narrow(0, 0, orig);
        let max_diff = (&r_c - &b_c).abs().max().double_value(&[]);
        assert!(
            r_c.allclose(&b_c, 1e-4, 1e-4, false),
            "single-row batched decode diverged from serial (max_diff={max_diff})"
        );
    }

    /// `forward_layers_train` must reject ranges outside the stage's owned window
    /// (same guardrail as the inference runner).
    #[test]
    fn forward_layers_train_rejects_out_of_window_range() {
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let mut w = tiny_weights();
        let stage = LlamaModel::stage_from_weights_with_config(
            &mut w, tiny_config(), &map, 2..LAYERS, DType::Float, KVQuantType::None,
        ).unwrap();
        let emb = Tensor::randn([1, 3, HIDDEN], (DType::Float, Device::Cpu));
        assert!(stage.forward_layers_train(&emb, 0..2, None).is_err(), "range below window");
        assert!(stage.forward_layers_train(&emb, 2..LAYERS, None).is_ok(), "owned range ok");
    }
}
