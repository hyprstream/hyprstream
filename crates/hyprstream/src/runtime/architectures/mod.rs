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
            Self::Llama { version } => format!("Llama{}", version),
            Self::Gemma => "Gemma".to_string(),
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
            Self::Phi { version } => format!("Phi{}", version),
            Self::Mistral => "Mistral".to_string(),
            Self::Starcoder => "Starcoder".to_string(),
            Self::Falcon => "Falcon".to_string(),
            Self::GPTNeoX => "GPT-NeoX".to_string(),
            Self::GPTOSS { total_params_b, .. } => format!("GPT-OSS-{}B", total_params_b),
            Self::GPTJ => "GPT-J".to_string(),
            Self::Janus { vision_encoder, has_generation, .. } => {
                let encoder_type = match vision_encoder {
                    VisionEncoderType::SigLIP { .. } => "SigLIP",
                    VisionEncoderType::CLIP { .. } => "CLIP",
                    VisionEncoderType::EVA { .. } => "EVA",
                };
                if *has_generation {
                    format!("Janus-{}-Gen", encoder_type)
                } else {
                    format!("Janus-{}", encoder_type)
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
            ModelArchitecture::Qwen { .. } => Box::new(QwenTokenizerConfig),
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

    /// Forward pass with LoRA integration support (for gradient bridge)
    /// Returns (logits, layer_activations) where layer_activations is for LoRA adapters
    fn forward_with_lora_hooks(
        &self,
        input: &Tensor,
        _lora_model: Option<&crate::lora::torch_adapter::LoRAModel>,
        _training: bool,
    ) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        // Default implementation without LoRA hooks
        let logits = self.forward(input, None)?;
        Ok((logits, Vec::new()))
    }

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

    /// Apply LoRA adapter with architecture-specific handling (for cache/storage only)
    fn apply_lora(&mut self, adapter: &crate::lora::torch_adapter::LoRAModel) -> Result<()>;

    /// Downcast to Any for accessing architecture-specific methods
    fn as_any(&self) -> &dyn std::any::Any;
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
            ModelArchitecture::Custom("MyModel".to_string()).name(),
            "MyModel"
        );
    }
}
