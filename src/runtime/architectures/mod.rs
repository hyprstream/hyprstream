//! Architecture-specific model implementations
//! 
//! This module provides abstractions and implementations for different
//! transformer architectures (Llama, Gemma, Qwen, etc.) with proper
//! tensor shape handling and LoRA compatibility.

use anyhow::{Result, anyhow};
use candle_core::{Device, DType, Tensor};
use candle_core::quantized::gguf_file;
use std::path::Path;

pub mod detector;
pub mod llama;
pub mod gemma;
pub mod config;
pub mod lora_adapter;

pub use detector::ArchitectureDetector;
pub use config::{ArchitectureConfig, AttentionConfig};
pub use lora_adapter::ArchitectureAwareLoRAAdapter;

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
    /// Custom/unknown architecture
    Custom(String),
}

impl ModelArchitecture {
    /// Get human-readable name
    pub fn name(&self) -> String {
        match self {
            Self::Llama { version } => format!("Llama{}", version),
            Self::Gemma => "Gemma".to_string(),
            Self::Qwen { version, is_moe, context_length } => {
                if *is_moe {
                    format!("Qwen{}-MoE-{}K", version, context_length / 1000)
                } else {
                    format!("Qwen{}-{}K", version, context_length / 1000)
                }
            },
            Self::Phi { version } => format!("Phi{}", version),
            Self::Mistral => "Mistral".to_string(),
            Self::Starcoder => "Starcoder".to_string(),
            Self::Falcon => "Falcon".to_string(),
            Self::GPTNeoX => "GPT-NeoX".to_string(),
            Self::GPTOSS { total_params_b, .. } => format!("GPT-OSS-{}B", total_params_b),
            Self::GPTJ => "GPT-J".to_string(),
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
            Self::GPTOSS { .. }  // GPT-OSS uses GQA in MoE
        )
    }
    
    /// Check if architecture uses Mixture of Experts
    pub fn supports_moe(&self) -> bool {
        matches!(
            self,
            Self::Qwen { is_moe: true, .. } |
            Self::GPTOSS { .. }
        )
    }
    
    /// Check if architecture supports sparse attention
    pub fn supports_sparse_attention(&self) -> bool {
        matches!(
            self,
            Self::Qwen { version: 3, .. } |  // Qwen3 uses MInference sparse attention
            Self::GPTOSS { .. }  // GPT-OSS optimized for long context
        )
    }
}

/// Core trait for architecture-specific model operations
pub trait ModelOperations: Send + Sync {
    /// Get the architecture type
    fn architecture(&self) -> ModelArchitecture;
    
    /// Get architecture configuration
    fn config(&self) -> &dyn ArchitectureConfig;
    
    /// Forward pass through the model
    fn forward(&self, input: &Tensor, past_kv: Option<&Tensor>) -> Result<Tensor>;
    
    /// Reshape tensor for attention computation
    fn reshape_for_attention(&self, tensor: &Tensor, is_key_value: bool) -> Result<Tensor>;
    
    /// Apply RoPE (Rotary Position Embeddings) if supported
    fn apply_rope(&self, tensor: &Tensor, position_ids: &Tensor) -> Result<Tensor>;
    
    /// Apply architecture-specific normalization (RMSNorm, LayerNorm, etc.)
    fn normalize(&self, tensor: &Tensor) -> Result<Tensor>;
    
    /// Get attention mask for the architecture
    fn get_attention_mask(&self, seq_len: usize, past_kv_len: usize) -> Result<Tensor>;
    
    /// Apply LoRA adapter with architecture-specific handling
    fn apply_lora(&mut self, adapter: &ArchitectureAwareLoRAAdapter) -> Result<()>;
}

/// Factory for creating architecture-specific models
pub struct ModelFactory;

impl ModelFactory {
    /// Load a model from GGUF file
    pub async fn from_gguf(
        path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        
        let architecture = ArchitectureDetector::detect_from_gguf(&content);
        
        tracing::info!("Detected architecture: {}", architecture.name());
        
        match architecture {
            ModelArchitecture::Llama { .. } => {
                let model = llama::LlamaModel::from_gguf(content, &mut file, device, dtype)?;
                Ok(Box::new(model))
            }
            ModelArchitecture::Gemma => {
                let model = gemma::GemmaModel::from_gguf(content, &mut file, device, dtype)?;
                Ok(Box::new(model))
            }
            _ => {
                tracing::warn!(
                    "Architecture {} not yet implemented, falling back to Llama",
                    architecture.name()
                );
                let model = llama::LlamaModel::from_gguf(content, &mut file, device, dtype)?;
                Ok(Box::new(model))
            }
        }
    }
    
    /// Load a model from SafeTensors file
    pub async fn from_safetensors(
        path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        // Detect architecture from config.json or model card
        let architecture = ArchitectureDetector::detect_from_safetensors(path)?;
        
        match architecture {
            ModelArchitecture::Llama { .. } => {
                let model = llama::LlamaModel::from_safetensors(path, device, dtype)?;
                Ok(Box::new(model))
            }
            ModelArchitecture::Gemma => {
                let model = gemma::GemmaModel::from_safetensors(path, device, dtype)?;
                Ok(Box::new(model))
            }
            _ => {
                Err(anyhow!(
                    "Architecture {} not yet supported for SafeTensors",
                    architecture.name()
                ))
            }
        }
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
        assert_eq!(ModelArchitecture::Custom("MyModel".to_string()).name(), "MyModel");
    }
}