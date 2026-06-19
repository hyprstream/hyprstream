//! Architecture-specific model implementations
//!
//! This module provides abstractions and implementations for different
//! transformer architectures (Llama, Gemma, Qwen, etc.) with proper
//! tensor shape handling and LoRA compatibility.

use anyhow::{anyhow, Result};
use tch::{nn, Tensor};

pub mod config;
pub mod gemma;
pub mod janus;
pub mod llama;
pub mod qwen;
pub mod qwen3_5;
pub mod qwen3_5_vision;
pub mod siglip;
// LoRA adapter moved to lora module

pub use config::{ArchitectureConfig, AttentionConfig};
// pub use lora_adapter::ArchitectureAwareLoRAAdapter; // Module removed

/// Supported model architectures
#[derive(Debug, Clone, PartialEq)]
pub enum ModelArchitecture {
    /// Meta's Llama family (1/2/3)
    Llama { version: u8 },
    /// Google's Gemma
    Gemma,
    /// Alibaba's Qwen (including Qwen3 with GQA and sparse attention)
    Qwen {
        version: u8,
        /// Qwen3 MoE variant (30B-A3B, 235B-A22B)
        is_moe: bool,
        /// Context length (32K, 128K, 262K for Qwen3)
        context_length: usize,
    },
    /// Microsoft's Phi
    Phi { version: u8 },
    /// Mistral AI models
    Mistral,
    /// BigCode's Starcoder
    Starcoder,
    /// TII's Falcon
    Falcon,
    /// EleutherAI's GPT-NeoX
    GPTNeoX,
    /// OpenAI GPT-OSS (120B/20B MoE models)
    GPTOSS {
        /// Total parameters in billions (120 or 20)
        total_params_b: u16,
        /// Active parameters in billions (5.1 or 3.6)
        active_params_b: f32,
        /// Number of experts in MoE
        num_experts: usize,
        /// Experts per token
        experts_per_token: usize,
    },
    /// GPT-J (EleutherAI's 6B model)
    GPTJ,
    /// Janus multimodal vision-language model
    Janus {
        /// Base language model architecture (usually Llama)
        base_architecture: Box<ModelArchitecture>,
        /// Type of vision encoder
        vision_encoder: VisionEncoderType,
        /// Whether model supports image generation (VQ-VAE)
        has_generation: bool,
    },
    /// Alibaba Qwen3.5 hybrid GatedDeltaNet/full-attention (dense and MoE variants)
    Qwen3_5,
    /// Custom/unknown architecture
    Custom(String),
}

/// Types of vision encoders for multimodal models
#[derive(Debug, Clone, PartialEq)]
pub enum VisionEncoderType {
    /// SigLIP vision encoder
    SigLIP {
        hidden_size: usize,
        image_size: usize,
        patch_size: usize,
        num_layers: usize,
    },
    /// CLIP vision encoder
    CLIP {
        hidden_size: usize,
        image_size: usize,
        patch_size: usize,
        num_layers: usize,
    },
    /// EVA vision encoder
    EVA {
        hidden_size: usize,
        image_size: usize,
        patch_size: usize,
        num_layers: usize,
    },
}

impl ModelArchitecture {
    /// Get human-readable name
    pub fn name(&self) -> String {
        match self {
            Self::Llama { version } => format!("Llama{version}"),
            Self::Gemma => "Gemma".to_owned(),
            Self::Qwen {
                version,
                is_moe,
                context_length,
            } => {
                if *is_moe {
                    format!("Qwen{}-MoE-{}K", version, context_length / 1000)
                } else {
                    format!("Qwen{}-{}K", version, context_length / 1000)
                }
            }
            Self::Phi { version } => format!("Phi{version}"),
            Self::Mistral => "Mistral".to_owned(),
            Self::Starcoder => "Starcoder".to_owned(),
            Self::Falcon => "Falcon".to_owned(),
            Self::GPTNeoX => "GPT-NeoX".to_owned(),
            Self::GPTOSS { total_params_b, .. } => format!("GPT-OSS-{total_params_b}B"),
            Self::GPTJ => "GPT-J".to_owned(),
            Self::Qwen3_5 => "Qwen3_5".to_owned(),
            Self::Janus { vision_encoder, has_generation, .. } => {
                let encoder_type = match vision_encoder {
                    VisionEncoderType::SigLIP { .. } => "SigLIP",
                    VisionEncoderType::CLIP { .. } => "CLIP",
                    VisionEncoderType::EVA { .. } => "EVA",
                };
                if *has_generation {
                    format!("Janus-{encoder_type}-Gen")
                } else {
                    format!("Janus-{encoder_type}")
                }
            }
            Self::Custom(name) => name.clone(),
        }
    }

    /// Check if architecture supports multi-query attention
    pub fn supports_mqa(&self) -> bool {
        matches!(self, Self::Gemma | Self::Falcon)
    }

    /// Check if architecture supports grouped-query attention
    pub fn supports_gqa(&self) -> bool {
        matches!(
            self,
            Self::Llama { version: 3 } | 
            Self::Mistral | 
            Self::Qwen { version: 3, .. } |  // Qwen3 uses GQA
            Self::GPTOSS { .. } // GPT-OSS uses GQA in MoE
        )
    }

    /// Check if architecture uses Mixture of Experts
    pub fn supports_moe(&self) -> bool {
        matches!(self, Self::Qwen { is_moe: true, .. } | Self::GPTOSS { .. })
    }

    /// Check if architecture supports sparse attention
    pub fn supports_sparse_attention(&self) -> bool {
        matches!(
            self,
            Self::Qwen { version: 3, .. } |  // Qwen3 uses MInference sparse attention
            Self::GPTOSS { .. } // GPT-OSS optimized for long context
        )
    }
}

/// Core trait for architecture-specific model operations
pub trait ModelOperations: Send {
    /// Get the architecture type
    fn architecture(&self) -> ModelArchitecture;

    /// Get architecture configuration
    fn config(&self) -> &dyn ArchitectureConfig;

    /// Get the tokenizer configuration for this model
    ///
    /// Returns a boxed TokenizerConfig trait object that provides
    /// model-specific tokenizer configuration logic.
    fn get_tokenizer_config(&self) -> Box<dyn crate::runtime::tokenizer_config::TokenizerConfig> {
        use crate::runtime::tokenizer_config::{DefaultTokenizerConfig, QwenTokenizerConfig, LlamaTokenizerConfig, GemmaTokenizerConfig};

        // Return appropriate config based on architecture
        match self.architecture() {
            ModelArchitecture::Qwen { .. } | ModelArchitecture::Qwen3_5 => Box::new(QwenTokenizerConfig),
            ModelArchitecture::Llama { .. } => Box::new(LlamaTokenizerConfig),
            ModelArchitecture::Gemma => Box::new(GemmaTokenizerConfig),
            ModelArchitecture::Janus { base_architecture, .. } => {
                // Use the base architecture's tokenizer config
                match base_architecture.as_ref() {
                    ModelArchitecture::Llama { .. } => Box::new(LlamaTokenizerConfig),
                    ModelArchitecture::Qwen { .. } => Box::new(QwenTokenizerConfig),
                    _ => Box::new(DefaultTokenizerConfig),
                }
            }
            _ => Box::new(DefaultTokenizerConfig),
        }
    }

    /// Forward pass through the model
    fn forward(&self, input: &Tensor, past_kv: Option<&Tensor>) -> Result<Tensor>;

    /// Get VarStore for training (if available)
    fn var_store(&self) -> Option<&nn::VarStore> {
        None // Default implementation returns None
    }

    /// Get mutable VarStore for training (if available)
    fn var_store_mut(&mut self) -> Option<&mut nn::VarStore> {
        None // Default implementation returns None
    }

    /// Forward pass with position information for KV caching
    fn forward_with_cache(&self, input: &Tensor, _start_pos: usize) -> Result<Tensor> {
        // Default implementation just calls regular forward
        // Models that support KV caching should override this
        self.forward(input, None)
    }

    /// Forward pass with KV caching and optional per-tenant delta injection
    ///
    /// When `delta` is `Some`, LoRA corrections are injected at each attention layer's
    /// Q/V projections before KV cache update. This is the unified path for both
    /// base model inference (delta=None) and delta-aware inference (delta=Some).
    ///
    /// Default implementation ignores delta and falls back to `forward_with_cache()`.
    /// Only architectures that support delta injection (Llama) need to override this.
    fn forward_with_cache_and_delta(
        &self,
        input: &Tensor,
        start_pos: usize,
        _delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        self.forward_with_cache(input, start_pos)
    }

    /// Batched ragged forward for continuous decode (#329, epic #310).
    ///
    /// Each entry is `(new_token_ids, start_pos, per-sequence KVCacheManager)`;
    /// all rows share the same query length and the single `delta` (the scheduler
    /// groups by tenant delta). Returns stacked logits `[B, q, vocab]`. Per-row
    /// results are equivalent to running each sequence through
    /// `forward_with_cache_and_delta` serially (CPU-verified merge gate).
    ///
    /// Default: unsupported. Only architectures with batched-decode support
    /// (Llama in v1) override this; callers fall back to the batch=1 path.
    fn forward_batched(
        &self,
        _sequences: &mut [(
            Vec<i64>,
            usize,
            std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>,
        )],
        _delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        Err(anyhow!(
            "continuous batching (forward_batched) is not supported for this architecture"
        ))
    }

    /// Get token embeddings for input IDs
    fn embed_tokens(&self, _input_ids: &Tensor) -> Result<Tensor> {
        Err(anyhow!("embed_tokens not implemented for this architecture"))
    }

    /// Forward pass from pre-computed embeddings (for multimodal models)
    ///
    /// This method allows models to start generation from embeddings instead of token IDs.
    /// Useful for multimodal models where vision embeddings are merged with text embeddings.
    ///
    /// # Arguments
    /// * `embeddings` - Pre-computed input embeddings [batch_size, seq_len, hidden_size]
    /// * `start_pos` - Starting position in KV cache (0 for initial forward)
    ///
    /// # Returns
    /// Logits tensor [batch_size, seq_len, vocab_size]
    fn forward_from_embeddings(&self, _embeddings: &Tensor, _start_pos: usize) -> Result<Tensor> {
        Err(anyhow!("forward_from_embeddings not implemented for this architecture"))
    }

    /// Check if this model is multimodal
    fn is_multimodal(&self) -> bool {
        matches!(self.architecture(), ModelArchitecture::Janus { .. })
    }

    /// Prepare inputs with vision embeddings (for multimodal models)
    ///
    /// This method combines text token IDs with vision embeddings for multimodal inference.
    /// Only implemented for multimodal architectures.
    ///
    /// # Arguments
    /// * `input_ids` - Text token IDs [batch_size, seq_len]
    /// * `pixel_values` - Preprocessed images [batch_size, channels, height, width]
    /// * `images_seq_mask` - Where to inject vision embeddings [batch_size, seq_len]
    /// * `images_emb_mask` - Which vision embeddings to use [batch_size, num_patches]
    ///
    /// # Returns
    /// Combined embeddings [batch_size, seq_len, hidden_size]
    fn prepare_multimodal_inputs(
        &self,
        _input_ids: &Tensor,
        _pixel_values: Option<&Tensor>,
        _images_seq_mask: Option<&Tensor>,
        _images_emb_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        Err(anyhow!("prepare_multimodal_inputs not implemented - model is not multimodal"))
    }

    /// Encode images through the vision encoder and return embeddings.
    ///
    /// For multimodal models with a vision encoder (e.g. Janus with SigLIP),
    /// this runs the vision encoder forward pass and returns pooled embeddings.
    ///
    /// # Arguments
    /// * `pixel_values` - Preprocessed images [batch_size, channels, height, width]
    ///
    /// # Returns
    /// Embedding tensor [batch_size, embedding_dim]
    fn encode_vision(&self, _pixel_values: &Tensor) -> Result<Tensor> {
        Err(anyhow!("encode_vision not implemented - model has no vision encoder"))
    }

    /// Decode a single layer (for layer-wise processing)
    fn decode_layer(
        &self,
        _layer_idx: usize,
        _hidden_states: &Tensor,
        _attention_mask: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        _past_kv: Option<&crate::runtime::kv_cache::LayerKVCache>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        Err(anyhow!("decode_layer not implemented for this architecture"))
    }

    /// Decode a single layer with per-tenant delta injection (for TTT training path)
    ///
    /// The delta's A/B matrices are injected after q_proj/v_proj projections inside
    /// the attention layer, creating a differentiable path from loss back to delta
    /// parameters. No KV cache is used (training path).
    ///
    /// Default implementation ignores the delta and falls back to `decode_layer()`.
    /// Only architectures that support delta injection need to override this.
    fn decode_layer_with_delta(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_kv: Option<&crate::runtime::kv_cache::LayerKVCache>,
        _delta: &crate::training::TenantDelta,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        self.decode_layer(layer_idx, hidden_states, attention_mask, position_ids, past_kv)
    }

    /// Run a contiguous range of decoder layers — the 2b intra-host pipeline
    /// (layer-split) primitive (#314).
    ///
    /// This is the missing layer-range runner that complements the already-
    /// exposed `embed_tokens` / `forward_from_embeddings` / `apply_final_norm` /
    /// `lm_head` / `num_layers`. Arch-agnostic orchestration composes them:
    /// - **stage 0** : `embed_tokens` → `forward_layers(0..b)`
    /// - **middle**  : `forward_layers(a..b)`
    /// - **last**    : `forward_layers(a..N)` → `apply_final_norm` → `lm_head`
    ///
    /// `is_first`/`is_last` are *implicit* in `range` (`range.start == 0` /
    /// `range.end == num_layers()`); the runner itself only applies decoder
    /// layers — never embeddings, final norm, or the LM head.
    ///
    /// # Stage-boundary contract
    /// The only state carried across a stage boundary is `hidden` + `start_pos`.
    /// `position_ids` is recomputed inside from `start_pos` + seq. Per-layer KV
    /// cache (and any SSM `conv`/`rec` state) is **stage-local and never
    /// transferred**. `range` is in **global** layer indices; an implementation
    /// that owns a shard remaps to its local `self.layers` via the
    /// `layer_offset` it was constructed with.
    ///
    /// # Device placement
    /// Layer `g` runs on its mapped device; the single cross-device copy is
    /// `hidden.to_device(next)` inserted only at a boundary where the device
    /// actually changes (zero copies within a stage or when source == dest).
    /// The returned tensor lives on the **last owned layer's device**.
    ///
    /// # Arguments
    /// * `hidden` - `[batch, seq, hidden]`; embeddings if `range.start == 0`,
    ///   otherwise the previous stage's output.
    /// * `range` - global layer indices this stage owns, `[a..b)`.
    /// * `start_pos` - KV-cache start position for this forward.
    /// * `delta` - optional per-tenant LoRA delta (delta-aware inference).
    ///
    /// The default implementation errors; only architectures that support the
    /// pipeline split (Llama; Qwen3.5) override it. The single-device whole-model
    /// `forward*` paths are unaffected — this method is purely additive.
    fn forward_layers(
        &self,
        _hidden: &Tensor,
        _range: std::ops::Range<usize>,
        _start_pos: usize,
        _delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        Err(anyhow!(
            "forward_layers (pipeline layer-split) not implemented for this architecture"
        ))
    }

    /// Training-path sibling of [`Self::forward_layers`] — the cross-device
    /// autograd primitive for **TTT-on-split** (#316, M-TRAIN-COUPLING).
    ///
    /// TTT is inference-time training that must traverse the **same** layer
    /// partition inference uses, so the backward pass materializes grads on each
    /// parameter's own device. This runner builds that autograd graph across the
    /// [`crate::runtime::device_pool::LayerDeviceMap`]; the lone stage-boundary
    /// `hidden.to_device(next)` is autograd-transparent (tch `to_device` is
    /// differentiable), so gradients flow back through it to the previous stage's
    /// device.
    ///
    /// # How it differs from the inference [`Self::forward_layers`]
    /// The inference and training paths are kept deliberately separate (as the
    /// whole-model paths already are). The training path:
    /// - uses **no KV cache** — full causal attention over the entire context;
    /// - pins **`start_pos = 0`** — `position_ids` are `0..seq`;
    /// - uses **fresh, call-local recurrent (SSM) state** for hybrid
    ///   architectures — the persistent inference `conv`/`rec` state is never read
    ///   or written, so a TTT step cannot pollute the inference recurrent state
    ///   (and the split is numerically identical to the whole-model training
    ///   forward, since per-layer recurrent state never crosses a layer boundary).
    ///
    /// Everything else — the global↔local layer remap via `layer_offset`, the
    /// single boundary copy, per-layer delta injection keyed by global index —
    /// matches [`Self::forward_layers`]. The arch-agnostic stage orchestration is
    /// identical (`embed_tokens` → `forward_layers_train(0..b)` → … →
    /// `forward_layers_train(a..N)` → `apply_final_norm` → `lm_head`), and the
    /// loss/backward is driven by the caller (e.g. the TTT trainer).
    ///
    /// The default implementation errors; only architectures that support the
    /// pipeline split (Llama; Qwen3.5) override it. The single-device whole-model
    /// training path ([`crate::runtime::TorchEngine::forward_with_delta`]) is
    /// unaffected — this method is purely additive.
    fn forward_layers_train(
        &self,
        _hidden: &Tensor,
        _range: std::ops::Range<usize>,
        _delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        Err(anyhow!(
            "forward_layers_train (pipeline layer-split training) not implemented for this architecture"
        ))
    }

    /// Apply final layer normalization
    fn apply_final_norm(&self, _hidden_states: &Tensor) -> Result<Tensor> {
        Err(anyhow!("apply_final_norm not implemented for this architecture"))
    }

    /// Apply language model head to get logits
    fn lm_head(&self, _hidden_states: &Tensor) -> Result<Tensor> {
        Err(anyhow!("lm_head not implemented for this architecture"))
    }

    /// Get the number of transformer layers
    fn num_layers(&self) -> usize {
        // Default implementation - architectures should override
        32  // Common default
    }

    /// Reshape tensor for attention computation
    fn reshape_for_attention(&self, tensor: &Tensor, is_key_value: bool) -> Result<Tensor>;

    /// Apply RoPE (Rotary Position Embeddings) if supported
    fn apply_rope(&self, tensor: &Tensor, position_ids: &Tensor) -> Result<Tensor>;

    /// Apply architecture-specific normalization (RMSNorm, LayerNorm, etc.)
    fn normalize(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Get attention mask for the architecture
    fn get_attention_mask(&self, seq_len: usize, past_kv_len: usize) -> Result<Tensor>;

    /// Downcast to Any for accessing architecture-specific methods
    fn as_any(&self) -> &dyn std::any::Any;

    // ============================================================================
    // KV Cache Management (for session-based cache isolation)
    // ============================================================================

    /// Clear the KV cache (for new generation)
    fn clear_kv_cache(&self) {
        // Default: no-op for models without KV caching
    }

    /// Get KV cache memory usage in bytes
    fn kv_cache_memory_usage(&self) -> usize {
        0 // Default: no cache
    }

    /// Set an external KV cache (for session-based cache management)
    ///
    /// This allows a KVCacheRegistry to manage caches externally while
    /// the model uses them for inference.
    fn set_kv_cache(
        &mut self,
        _cache: std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>,
    ) {
        // Default: no-op for models without configurable cache
    }

    /// Get the current KV cache (for external management)
    fn get_kv_cache(
        &self,
    ) -> Option<std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>> {
        None // Default: no cache
    }

    /// Take ownership of the KV cache (removes it from the model)
    fn take_kv_cache(
        &mut self,
    ) -> Option<std::sync::Arc<parking_lot::Mutex<crate::runtime::kv_cache::KVCacheManager>>> {
        None // Default: no cache
    }
}

/// Attention mechanism types
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionType {
    /// Multi-Head Attention (standard)
    MHA,
    /// Multi-Query Attention (single KV head)
    MQA,
    /// Grouped-Query Attention (multiple query heads per KV head)
    GQA { num_groups: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_properties() {
        assert!(ModelArchitecture::Gemma.supports_mqa());
        assert!(!ModelArchitecture::Gemma.supports_gqa());
        assert!(ModelArchitecture::Llama { version: 3 }.supports_gqa());
    }

    #[test]
    fn test_architecture_names() {
        assert_eq!(ModelArchitecture::Llama { version: 2 }.name(), "Llama2");
        assert_eq!(ModelArchitecture::Gemma.name(), "Gemma");
        assert_eq!(
            ModelArchitecture::Custom("MyModel".to_owned()).name(),
            "MyModel"
        );
    }
}
