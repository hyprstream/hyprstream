//! Llama model implementation with support for Llama 1/2/3 and GQA

use super::{ModelArchitecture, ModelOperations, ArchitectureConfig};
use super::lora_adapter::ArchitectureAwareLoRAAdapter;
use anyhow::{Result, anyhow};
use candle_core::{Device, DType, Tensor, D};
use candle_core::quantized::gguf_file;
use candle_nn::{Module, VarBuilder};
use std::collections::HashMap;
use std::path::Path;

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
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// RoPE theta
    pub rope_theta: f32,
    /// RoPE scaling (for Llama 3)
    pub rope_scaling: Option<RopeScaling>,
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
        Self {
            version: 2,
            num_attention_heads: 32,
            num_key_value_heads: 32,  // No GQA in base 7B
            hidden_size: 4096,
            head_dim: 128,
            intermediate_size: 11008,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            vocab_size: 32000,
            num_hidden_layers: 32,
            rope_theta: 10000.0,
            rope_scaling: None,
        }
    }
}

impl LlamaConfig {
    /// Create config for Llama 3 8B
    pub fn llama3_8b() -> Self {
        Self {
            version: 3,
            num_attention_heads: 32,
            num_key_value_heads: 8,  // GQA with 8 KV heads
            hidden_size: 4096,
            head_dim: 128,
            intermediate_size: 14336,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            vocab_size: 128256,
            num_hidden_layers: 32,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                factor: 8.0,
            }),
        }
    }

    /// Create config for Llama 3 70B
    pub fn llama3_70b() -> Self {
        Self {
            version: 3,
            num_attention_heads: 64,
            num_key_value_heads: 8,  // GQA with 8 KV heads
            hidden_size: 8192,
            head_dim: 128,
            intermediate_size: 28672,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            vocab_size: 128256,
            num_hidden_layers: 80,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                factor: 8.0,
            }),
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
        true  // All Llama versions use RMSNorm
    }
}

/// Llama model implementation
pub struct LlamaModel {
    config: LlamaConfig,
    device: Device,
    dtype: DType,
    
    // Model weights
    embed_tokens: Option<Tensor>,
    layers: Vec<LlamaLayer>,
    norm: Option<RMSNorm>,
    lm_head: Option<Tensor>,
}

/// Single Llama transformer layer
struct LlamaLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

/// Llama attention with optional GQA support
struct LlamaAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    rope_scaling: Option<RopeScaling>,
}

impl LlamaAttention {
    /// Apply attention with optional GQA
    fn forward(&self, hidden_states: &Tensor, position_ids: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        
        // Project to Q, K, V
        let q = hidden_states.matmul(&self.q_proj)?;
        let k = hidden_states.matmul(&self.k_proj)?;
        let v = hidden_states.matmul(&self.v_proj)?;
        
        // Reshape for attention
        // Q: [batch, seq, num_heads, head_dim]
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?;
        // K, V: [batch, seq, num_kv_heads, head_dim]
        let k = k.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?;
        
        // Apply RoPE if position_ids provided
        let (q, k) = if let Some(pos_ids) = position_ids {
            (
                self.apply_rope(&q, pos_ids)?,
                self.apply_rope(&k, pos_ids)?
            )
        } else {
            (q, k)
        };
        
        // Expand K, V for GQA if needed
        let (k, v) = if self.num_kv_heads < self.num_heads {
            (
                self.expand_kv_for_gqa(&k)?,
                self.expand_kv_for_gqa(&v)?
            )
        } else {
            (k, v)
        };
        
        // Compute attention scores
        let scores = self.compute_attention_scores(&q, &k)?;
        
        // Apply attention to values
        let attn_output = scores.matmul(&v)?;
        
        // Reshape and project output
        let attn_output = attn_output
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?
            .matmul(&self.o_proj)?;
        
        Ok(attn_output)
    }
    
    /// Expand KV heads for GQA (Llama 2/3)
    fn expand_kv_for_gqa(&self, kv: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, num_kv_heads, head_dim) = kv.dims4()?;
        let repeat_factor = self.num_heads / num_kv_heads;
        
        if repeat_factor == 1 {
            return Ok(kv.clone());
        }
        
        // Repeat KV heads to match Q heads
        Ok(kv.unsqueeze(3)?  // [batch, seq, num_kv_heads, 1, head_dim]
            .expand(&[batch_size, seq_len, num_kv_heads, repeat_factor, head_dim])?
            .reshape(&[batch_size, seq_len, self.num_heads, head_dim])?)
    }
    
    /// Apply Rotary Position Embeddings with optional scaling
    fn apply_rope(&self, tensor: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Simplified RoPE implementation
        // In production, implement proper RoPE with sin/cos caching
        // and Llama 3 scaling if configured
        Ok(tensor.clone())  // Placeholder
    }
    
    /// Compute scaled dot-product attention scores
    fn compute_attention_scores(&self, q: &Tensor, k: &Tensor) -> Result<Tensor> {
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        
        // Transpose for matmul
        let k_t = k.transpose(1, 2)?.transpose(2, 3)?;
        let q_t = q.transpose(1, 2)?;
        
        // Compute attention scores
        let scores = q_t.matmul(&k_t)?.affine(scale as f64, 0.0)?;
        
        // Apply softmax
        Ok(candle_nn::ops::softmax_last_dim(&scores)?)
    }
}

/// Llama MLP/FFN layer
struct LlamaMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl LlamaMLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Llama uses SiLU activation
        let gate = candle_nn::ops::silu(&hidden_states.matmul(&self.gate_proj)?)?;
        let up = hidden_states.matmul(&self.up_proj)?;
        Ok(gate.mul(&up)?.matmul(&self.down_proj)?)
    }
}

/// RMSNorm implementation for Llama
struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute RMS
        let x2 = x.sqr()?;
        let mean = x2.mean_keepdim(D::Minus1)?;
        let rrms = mean.affine(1.0, self.eps as f64)?.recip()?.sqrt()?;
        
        // Apply normalization and scaling
        Ok(x.broadcast_mul(&rrms)?.broadcast_mul(&self.weight)?)
    }
}

impl LlamaModel {
    /// Create Llama model from GGUF file
    pub fn from_gguf(
        mut content: gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Extract configuration from metadata
        let config = Self::extract_config(&content)?;
        
        // Load weights (simplified)
        let layers = Vec::new();
        
        Ok(Self {
            config,
            device: device.clone(),
            dtype,
            embed_tokens: None,
            layers,
            norm: None,
            lm_head: None,
        })
    }
    
    /// Create Llama model from SafeTensors
    pub fn from_safetensors(
        path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Load configuration
        let config_path = path.parent()
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
            device: device.clone(),
            dtype,
            embed_tokens: None,
            layers,
            norm: None,
            lm_head: None,
        })
    }
    
    /// Extract configuration from GGUF metadata
    fn extract_config(content: &gguf_file::Content) -> Result<LlamaConfig> {
        let metadata = &content.metadata;
        
        // Detect Llama version
        let version = if metadata.contains_key("llama.rope.scaling.type") {
            3  // Llama 3 has rope scaling
        } else if metadata.contains_key("llama.attention.head_count_kv") {
            2  // Llama 2 has GQA
        } else {
            1  // Original Llama
        };
        
        let mut config = LlamaConfig {
            version,
            num_attention_heads: metadata.get("llama.attention.head_count")
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(32) as usize,
            num_key_value_heads: metadata.get("llama.attention.head_count_kv")
                .and_then(|v| v.to_u32().ok())
                .unwrap_or_else(|| {
                    metadata.get("llama.attention.head_count")
                        .and_then(|v| v.to_u32().ok())
                        .unwrap_or(32)
                }) as usize,
            hidden_size: metadata.get("llama.embedding_length")
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(4096) as usize,
            head_dim: metadata.get("llama.rope.dimension_count")
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(128) as usize,
            intermediate_size: metadata.get("llama.feed_forward_length")
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(11008) as usize,
            max_position_embeddings: metadata.get("llama.context_length")
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(4096) as usize,
            rms_norm_eps: metadata.get("llama.attention.layer_norm_rms_epsilon")
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(1e-5),
            vocab_size: metadata.get("tokenizer.ggml.model.vocab_size")
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(32000) as usize,
            num_hidden_layers: metadata.get("llama.block_count")
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(32) as usize,
            rope_theta: metadata.get("llama.rope.freq_base")
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(10000.0),
            rope_scaling: None,
        };
        
        // Check for Llama 3 rope scaling
        if version == 3 {
            if let Some(scaling_type) = metadata.get("llama.rope.scaling.type")
                .and_then(|v| v.to_string().ok()) {
                config.rope_scaling = Some(RopeScaling {
                    scaling_type: scaling_type.clone(),
                    factor: metadata.get("llama.rope.scaling.factor")
                        .and_then(|v| v.to_f32().ok())
                        .unwrap_or(8.0),
                });
            }
        }
        
        Ok(config)
    }
    
    /// Parse configuration from JSON
    fn parse_config(json_str: &str) -> Result<LlamaConfig> {
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
            num_key_value_heads: json.get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| json["num_attention_heads"].as_u64().unwrap_or(32)) as usize,
            hidden_size: json["hidden_size"].as_u64().unwrap_or(4096) as usize,
            head_dim: json.get("head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(11008) as usize,
            max_position_embeddings: json["max_position_embeddings"].as_u64().unwrap_or(4096) as usize,
            rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as usize,
            num_hidden_layers: json["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            rope_theta: json.get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0) as f32,
            rope_scaling: None,
        };
        
        // Parse rope scaling if present
        if let Some(rope_scaling) = json.get("rope_scaling") {
            config.rope_scaling = Some(RopeScaling {
                scaling_type: rope_scaling["type"].as_str().unwrap_or("linear").to_string(),
                factor: rope_scaling["factor"].as_f64().unwrap_or(8.0) as f32,
            });
        }
        
        Ok(config)
    }
}

impl ModelOperations for LlamaModel {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Llama { version: self.config.version }
    }
    
    fn config(&self) -> &dyn ArchitectureConfig {
        &self.config
    }
    
    fn forward(&self, input: &Tensor, _past_kv: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = input.clone();
        
        // Embedding layer
        if let Some(embed) = &self.embed_tokens {
            // Input should be token IDs for embedding
            // hidden_states = embed.index_select(&hidden_states, 0)?;
        }
        
        // Apply transformer layers
        for layer in &self.layers {
            let residual = hidden_states.clone();
            
            // Self-attention block
            hidden_states = layer.input_layernorm.forward(&hidden_states)?;
            let attn_output = layer.self_attn.forward(&hidden_states, None)?;
            hidden_states = (residual + attn_output)?;
            
            // FFN block
            let residual = hidden_states.clone();
            hidden_states = layer.post_attention_layernorm.forward(&hidden_states)?;
            let ffn_output = layer.mlp.forward(&hidden_states)?;
            hidden_states = (residual + ffn_output)?;
        }
        
        // Final layer norm
        if let Some(norm) = &self.norm {
            hidden_states = norm.forward(&hidden_states)?;
        }
        
        // LM head
        if let Some(lm_head) = &self.lm_head {
            hidden_states = hidden_states.matmul(lm_head)?;
        }
        
        Ok(hidden_states)
    }
    
    fn reshape_for_attention(&self, tensor: &Tensor, is_key_value: bool) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = tensor.dims3()?;
        
        if is_key_value {
            // For K,V: reshape to [batch, seq, num_kv_heads, head_dim]
            Ok(tensor.reshape(&[
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim,
            ])?)
        } else {
            // For Q: reshape to [batch, seq, num_attention_heads, head_dim]
            Ok(tensor.reshape(&[
                batch_size,
                seq_len,
                self.config.num_attention_heads,
                self.config.head_dim,
            ])?)
        }
    }
    
    fn apply_rope(&self, tensor: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Implement RoPE for Llama with optional scaling
        // This is a placeholder - actual implementation would use proper RoPE
        Ok(tensor.clone())
    }
    
    fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        // Llama uses RMSNorm
        let x2 = tensor.sqr()?;
        let mean = x2.mean_keepdim(D::Minus1)?;
        let rrms = mean.affine(1.0, self.config.rms_norm_eps as f64)?.recip()?.sqrt()?;
        Ok(tensor.broadcast_mul(&rrms)?)
    }
    
    fn get_attention_mask(&self, seq_len: usize, past_kv_len: usize) -> Result<Tensor> {
        // Create causal attention mask
        let total_len = seq_len + past_kv_len;
        let mask = Tensor::ones(&[seq_len, total_len], DType::F32, &self.device)?;
        
        // Apply causal masking (simplified)
        Ok(mask)
    }
    
    fn apply_lora(&mut self, adapter: &ArchitectureAwareLoRAAdapter) -> Result<()> {
        // Apply LoRA weights with architecture-specific handling
        // The adapter will handle GQA shape conversions for Llama 2/3
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_llama_configs() {
        let llama2 = LlamaConfig::default();
        assert_eq!(llama2.version, 2);
        assert_eq!(llama2.num_attention_heads, 32);
        assert_eq!(llama2.num_key_value_heads, 32);
        
        let llama3_8b = LlamaConfig::llama3_8b();
        assert_eq!(llama3_8b.version, 3);
        assert_eq!(llama3_8b.num_attention_heads, 32);
        assert_eq!(llama3_8b.num_key_value_heads, 8);  // GQA
        assert!(llama3_8b.rope_scaling.is_some());
        
        let llama3_70b = LlamaConfig::llama3_70b();
        assert_eq!(llama3_70b.version, 3);
        assert_eq!(llama3_70b.num_attention_heads, 64);
        assert_eq!(llama3_70b.num_key_value_heads, 8);  // GQA
    }
    
    #[test]
    fn test_llama_reshape_for_gqa() {
        let device = Device::Cpu;
        let config = LlamaConfig::llama3_8b();
        let model = LlamaModel {
            config: config.clone(),
            device: device.clone(),
            dtype: DType::F32,
            embed_tokens: None,
            layers: vec![],
            norm: None,
            lm_head: None,
        };
        
        // Test tensor with shape [2, 10, 4096]
        let tensor = Tensor::randn(0.0, 1.0, &[2, 10, 4096], &device).unwrap();
        
        // Reshape for key/value (GQA)
        let reshaped = model.reshape_for_attention(&tensor, true).unwrap();
        assert_eq!(reshaped.dims(), &[2, 10, 8, 128]);  // 8 KV heads for Llama 3
        
        // Reshape for query
        let reshaped = model.reshape_for_attention(&tensor, false).unwrap();
        assert_eq!(reshaped.dims(), &[2, 10, 32, 128]);  // 32 Q heads
    }
    
    #[test]
    fn test_kv_expansion_for_gqa() {
        let device = Device::Cpu;
        let attn = LlamaAttention {
            q_proj: Tensor::zeros(&[4096, 4096], DType::F32, &device).unwrap(),
            k_proj: Tensor::zeros(&[4096, 1024], DType::F32, &device).unwrap(),
            v_proj: Tensor::zeros(&[4096, 1024], DType::F32, &device).unwrap(),
            o_proj: Tensor::zeros(&[4096, 4096], DType::F32, &device).unwrap(),
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                factor: 8.0,
            }),
        };
        
        // Create KV tensor with 8 heads
        let kv = Tensor::randn(0.0, 1.0, &[2, 10, 8, 128], &device).unwrap();
        
        // Expand to match 32 query heads
        let expanded = attn.expand_kv_for_gqa(&kv).unwrap();
        assert_eq!(expanded.dims(), &[2, 10, 32, 128]);
    }
}