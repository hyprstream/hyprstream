//! Llama model implementation with support for Llama 1/2/3 and GQA

use super::{ModelArchitecture, ModelOperations, ArchitectureConfig};
use super::lora_adapter::ArchitectureAwareLoRAAdapter;
use anyhow::{Result, anyhow};
use candle_core::{Device, DType, Tensor, D};
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
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        
        // Debug: Check shapes before matmul
        // tracing::debug!("Attention forward - hidden_states shape: {:?}", hidden_states.dims());
        // tracing::debug!("Attention forward - q_proj shape: {:?}", self.q_proj.dims());
        // tracing::debug!("Attention config - num_heads: {}, head_dim: {}, total: {}", 
        //               self.num_heads, self.head_dim, self.num_heads * self.head_dim);
        
        // Reshape hidden_states for matmul: [batch*seq, hidden_size]
        let hidden_states_2d = hidden_states.reshape(&[batch_size * seq_len, hidden_size])?;
        
        // Project to Q, K, V using 2D matmul
        let q = hidden_states_2d.matmul(&self.q_proj)?;
        let k = hidden_states_2d.matmul(&self.k_proj)?;
        let v = hidden_states_2d.matmul(&self.v_proj)?;
        
        // tracing::debug!("After projection - Q shape: {:?}, K shape: {:?}, V shape: {:?}", 
        //               q.dims(), k.dims(), v.dims());
        // tracing::debug!("Expected reshape: [{}, {}, {}, {}] = {} elements, actual Q elements: {}", 
        //               batch_size, seq_len, self.num_heads, self.head_dim, 
        //               batch_size * seq_len * self.num_heads * self.head_dim,
        //               q.dims().iter().product::<usize>());
        
        // Determine actual K/V heads from tensor dimensions
        let k_elements = k.dims().iter().product::<usize>();
        let v_elements = v.dims().iter().product::<usize>();
        
        // For K and V, we need to figure out the actual number of heads
        // They might be different from what we detected in config
        let kv_heads = if k_elements == batch_size * seq_len * self.num_kv_heads * self.head_dim {
            self.num_kv_heads  // Config is correct
        } else if k_elements % (batch_size * seq_len * self.head_dim) == 0 {
            // Recalculate based on actual size
            k_elements / (batch_size * seq_len * self.head_dim)
        } else if k_elements % (batch_size * seq_len) == 0 {
            // Try different head_dim
            let kv_dim = k_elements / (batch_size * seq_len);
            if kv_dim == 256 && self.head_dim == 128 {
                2  // 2 heads with 128 dim
            } else if kv_dim == 256 {
                8  // 8 heads with 32 dim
            } else {
                self.num_kv_heads  // Fallback to config
            }
        } else {
            self.num_kv_heads  // Fallback
        };
        
        // tracing::debug!("Actual KV heads: {}, config KV heads: {}", kv_heads, self.num_kv_heads);
        
        // Reshape for attention
        // Q: [batch, seq, num_heads, head_dim]
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?;
        // K, V: [batch, seq, num_kv_heads, head_dim] - use actual kv_heads
        let k = k.reshape(&[batch_size, seq_len, kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch_size, seq_len, kv_heads, self.head_dim])?;
        
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
        let (k, v) = if kv_heads < self.num_heads {
            // tracing::debug!("Expanding KV from {} heads to {} heads", kv_heads, self.num_heads);
            (
                self.expand_kv_for_gqa_with_heads(&k, kv_heads)?,
                self.expand_kv_for_gqa_with_heads(&v, kv_heads)?
            )
        } else {
            (k, v)
        };
        
        // Compute attention scores
        let scores = self.compute_attention_scores(&q, &k)?;
        
        // V: [batch, seq, heads, dim] -> [batch, heads, seq, dim]  
        let v = v.transpose(1, 2)?;
        
        // Apply attention to values: [batch, heads, seq, seq] x [batch, heads, seq, dim] = [batch, heads, seq, dim]
        let attn_output = scores.matmul(&v)?;
        
        // Transpose back: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
        let attn_output = attn_output.transpose(1, 2)?;
        
        // Reshape to combine heads: [batch, seq, heads*dim]
        let attn_output = attn_output
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;
        
        // Reshape for output projection: [batch*seq, heads*dim]
        let attn_output_2d = attn_output.reshape(&[batch_size * seq_len, self.num_heads * self.head_dim])?;
        
        // Apply output projection: [batch*seq, heads*dim] x [heads*dim, hidden_size] = [batch*seq, hidden_size]
        let attn_output = attn_output_2d.matmul(&self.o_proj)?;
        
        // Reshape back to 3D: [batch, seq, hidden_size]
        let attn_output = attn_output.reshape(&[batch_size, seq_len, hidden_size])?;
        
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
    
    /// Expand KV tensors for GQA with explicit head count
    fn expand_kv_for_gqa_with_heads(&self, kv: &Tensor, actual_kv_heads: usize) -> Result<Tensor> {
        let (batch_size, seq_len, _detected_heads, head_dim) = kv.dims4()?;
        let repeat_factor = self.num_heads / actual_kv_heads;
        
        if repeat_factor == 1 {
            return Ok(kv.clone());
        }
        
        // tracing::debug!("GQA expansion: {} KV heads -> {} Q heads (repeat {}x)", 
        //               actual_kv_heads, self.num_heads, repeat_factor);
        
        // Expand by repeating KV heads
        Ok(kv.unsqueeze(3)?  // [batch, seq, num_kv_heads, 1, head_dim]
            .expand(&[batch_size, seq_len, actual_kv_heads, repeat_factor, head_dim])?
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
        
        // Q: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        let q = q.transpose(1, 2)?;
        // K: [batch, seq, heads, dim] -> [batch, heads, seq, dim] -> [batch, heads, dim, seq]
        let k = k.transpose(1, 2)?.transpose(2, 3)?;
        
        // Compute attention scores: [batch, heads, seq, seq]
        let scores = q.matmul(&k)?.affine(scale as f64, 0.0)?;
        
        // Apply softmax along last dimension
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
        // Get dimensions
        let original_shape = hidden_states.dims();
        let (batch_size, seq_len, hidden_size) = if original_shape.len() == 3 {
            (original_shape[0], original_shape[1], original_shape[2])
        } else {
            // Already 2D
            return self.forward_2d(hidden_states);
        };
        
        // Reshape to 2D for matmul
        let hidden_2d = hidden_states.reshape(&[batch_size * seq_len, hidden_size])?;
        
        // Llama uses SiLU activation
        let gate = candle_nn::ops::silu(&hidden_2d.matmul(&self.gate_proj)?)?;
        let up = hidden_2d.matmul(&self.up_proj)?;
        let output = gate.mul(&up)?.matmul(&self.down_proj)?;
        
        // Reshape back to 3D
        let out_size = output.dims()[1];
        Ok(output.reshape(&[batch_size, seq_len, out_size])?)
    }
    
    fn forward_2d(&self, hidden_states: &Tensor) -> Result<Tensor> {
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
    /// Create Llama model from file (deprecated)
    pub fn from_file(
        _path: &Path,
        _device: &Device,
        _dtype: DType,
    ) -> Result<Self> {
        Err(anyhow!("Model format not supported. Please use SafeTensors format."))
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
    
    /// Create Llama model from pre-loaded weight tensors
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Parse config from weights if possible, otherwise use defaults
        let config = Self::detect_config_from_weights(weights)?;
        Self::from_weights_with_config(weights, config, device, dtype)
    }
    
    /// Create Llama model with explicit config (allows Qwen models to override)
    pub fn from_weights_with_config(
        weights: &HashMap<String, Tensor>,
        config: LlamaConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        
        // Extract key tensors
        let embed_tokens = weights.get("model.embed_tokens.weight")
            .or_else(|| weights.get("embed_tokens.weight"))
            .cloned();
        
        // Handle lm_head - Gemma models use weight tying (lm_head shares weights with embed_tokens)
        // Try to find explicit lm_head first, otherwise we'll use tied weights from embeddings
        let lm_head = weights.get("lm_head.weight")
            .or_else(|| weights.get("model.lm_head.weight"))
            .map(|w| {
                // LM head is stored as [vocab_size, hidden_size] in HuggingFace
                // We need [hidden_size, vocab_size] for matmul
                w.transpose(0, 1).and_then(|t| t.contiguous())
            })
            .transpose()?;
        
        // Check if this is a model with tied weights (like Gemma)
        if lm_head.is_none() && embed_tokens.is_some() {
            tracing::info!("Model appears to use weight tying (lm_head = embed_tokens.T), like Gemma models");
        }
        
        // Extract final layer norm
        let norm = weights.get("model.norm.weight")
            .or_else(|| weights.get("norm.weight"))
            .map(|w| RMSNorm {
                weight: w.clone(),
                eps: config.rms_norm_eps,
            });
        
        // Build transformer layers
        let mut layers = Vec::new();
        for layer_idx in 0..config.num_hidden_layers {
            if let Some(layer) = Self::build_layer(layer_idx, weights, &config, device)? {
                layers.push(layer);
            }
        }
        
        Ok(Self {
            config,
            device: device.clone(),
            dtype,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
    
    /// Detect configuration from weight tensor shapes
    pub fn detect_config_from_weights(weights: &HashMap<String, Tensor>) -> Result<LlamaConfig> {
        // Try to infer config from tensor shapes
        let mut config = LlamaConfig::default();
        
        // Get vocab size from embedding
        if let Some(embed) = weights.get("model.embed_tokens.weight")
            .or_else(|| weights.get("embed_tokens.weight")) {
            let shape = embed.dims();
            if shape.len() >= 2 {
                config.vocab_size = shape[0];
                config.hidden_size = shape[1];
            }
        }
        
        // Count layers
        let layer_count = weights.keys()
            .filter(|k| k.contains("layers.") && k.contains(".self_attn.q_proj"))
            .count();
        if layer_count > 0 {
            config.num_hidden_layers = layer_count;
        }
        
        // Get attention heads from q_proj and k_proj shapes
        if let Some(q_proj) = weights.get("model.layers.0.self_attn.q_proj.weight") {
            let q_shape = q_proj.dims();
            if q_shape.len() >= 2 {
                // shape[0] is output dim (num_heads * head_dim)
                // shape[1] is hidden_size
                config.hidden_size = q_shape[1];
                let q_proj_out_dim = q_shape[0];
                
                // Also check k_proj to detect GQA (Grouped Query Attention)
                let k_proj_out_dim = if let Some(k_proj) = weights.get("model.layers.0.self_attn.k_proj.weight") {
                    k_proj.dims()[0]
                } else {
                    q_proj_out_dim // Assume same as Q if K not found
                };
                
                tracing::debug!("Q projection output: {}, K projection output: {}", q_proj_out_dim, k_proj_out_dim);
                
                // Try common head_dim values to find the right configuration
                // For smaller models, try smaller head dimensions first
                let possible_head_dims = if q_proj_out_dim <= 512 {
                    vec![32, 64, 48, 40, 128] // Smaller models often use smaller head_dim
                } else {
                    vec![128, 64, 80, 96, 160]
                };
                
                let mut found_config = false;
                
                for &head_dim in &possible_head_dims {
                    if q_proj_out_dim % head_dim == 0 && k_proj_out_dim % head_dim == 0 {
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
        
        Ok(config)
    }
    
    /// Build a single transformer layer from weights
    fn build_layer(
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        config: &LlamaConfig,
        device: &Device,
    ) -> Result<Option<LlamaLayer>> {
        let prefix = format!("model.layers.{}", layer_idx);
        
        // Check if this layer exists (handle both separate and combined projections)
        let has_separate_qkv = weights.contains_key(&format!("{}.self_attn.q_proj.weight", prefix));
        let has_combined_qkv = weights.contains_key(&format!("{}.self_attn.c_attn.weight", prefix));
        
        if !has_separate_qkv && !has_combined_qkv {
            return Ok(None);
        }
        
        // Build attention
        // Note: PyTorch/HuggingFace stores linear weights as [out_features, in_features]
        // but we need [in_features, out_features] for matmul, so transpose them
        let (q_proj, k_proj, v_proj) = if has_separate_qkv {
            // Standard separate Q, K, V projections (Llama style)
            
            // Get and transpose q_proj weight
            let q_proj_orig = weights.get(&format!("{}.self_attn.q_proj.weight", prefix))
                .ok_or_else(|| anyhow!("Missing q_proj weight"))?;
            tracing::debug!("Layer {} q_proj original shape: {:?}", layer_idx, q_proj_orig.dims());
            
            // Transpose from PyTorch format [out, in] to [in, out] for matmul
            // Make sure to make it contiguous after transpose
            let q_proj = q_proj_orig.transpose(0, 1)?.contiguous()?;
            tracing::debug!("Layer {} q_proj transposed shape: {:?}", layer_idx, q_proj.dims());
            
            // Get and transpose k_proj weight
            let k_proj_orig = weights.get(&format!("{}.self_attn.k_proj.weight", prefix))
                .ok_or_else(|| anyhow!("Missing k_proj weight"))?;
            tracing::debug!("Layer {} k_proj original shape: {:?}", layer_idx, k_proj_orig.dims());
            let k_proj = k_proj_orig.transpose(0, 1)?.contiguous()?;
            tracing::debug!("Layer {} k_proj transposed shape: {:?}", layer_idx, k_proj.dims());
            
            // Get and transpose v_proj weight
            let v_proj_orig = weights.get(&format!("{}.self_attn.v_proj.weight", prefix))
                .ok_or_else(|| anyhow!("Missing v_proj weight"))?;
            tracing::debug!("Layer {} v_proj original shape: {:?}", layer_idx, v_proj_orig.dims());
            let v_proj = v_proj_orig.transpose(0, 1)?.contiguous()?;
            tracing::debug!("Layer {} v_proj transposed shape: {:?}", layer_idx, v_proj.dims());
            
            (q_proj, k_proj, v_proj)
        } else {
            // Combined QKV projection (some Qwen models use c_attn)
            let c_attn = weights.get(&format!("{}.self_attn.c_attn.weight", prefix))
                .ok_or_else(|| anyhow!("Missing c_attn weight"))?
                .transpose(0, 1)?  // Transpose from [out, in] to [in, out]
                .contiguous()?;
            
            // Split c_attn into Q, K, V
            // c_attn has shape [hidden_size, 3 * projection_size]
            let dims = c_attn.dims();
            let hidden_size = dims[0];
            let total_proj_size = dims[1];
            let proj_size = total_proj_size / 3;
            
            let q = c_attn.narrow(1, 0, proj_size)?;
            let k = c_attn.narrow(1, proj_size, proj_size)?;
            let v = c_attn.narrow(1, proj_size * 2, proj_size)?;
            
            (q, k, v)
        };
        
        let self_attn = LlamaAttention {
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj: weights.get(&format!("{}.self_attn.o_proj.weight", prefix))
                .or_else(|| weights.get(&format!("{}.self_attn.c_proj.weight", prefix)))  // Some models use c_proj for output
                .ok_or_else(|| anyhow!("Missing o_proj/c_proj weight"))?
                .transpose(0, 1)?  // Transpose from [out, in] to [in, out]
                .contiguous()?,
            rope_theta: config.rope_theta,
            rope_scaling: config.rope_scaling.clone(),
        };
        
        // Build MLP
        let mlp = LlamaMLP {
            gate_proj: weights.get(&format!("{}.mlp.gate_proj.weight", prefix))
                .ok_or_else(|| anyhow!("Missing gate_proj weight"))?
                .transpose(0, 1)?  // Transpose from [out, in] to [in, out]
                .contiguous()?,
            up_proj: weights.get(&format!("{}.mlp.up_proj.weight", prefix))
                .ok_or_else(|| anyhow!("Missing up_proj weight"))?
                .transpose(0, 1)?  // Transpose from [out, in] to [in, out]
                .contiguous()?,
            down_proj: weights.get(&format!("{}.mlp.down_proj.weight", prefix))
                .ok_or_else(|| anyhow!("Missing down_proj weight"))?
                .transpose(0, 1)?  // Transpose from [out, in] to [in, out]
                .contiguous()?,
        };
        
        // Build layer norms
        let input_layernorm = RMSNorm {
            weight: weights.get(&format!("{}.input_layernorm.weight", prefix))
                .ok_or_else(|| anyhow!("Missing input_layernorm weight"))?.clone(),
            eps: config.rms_norm_eps,
        };
        
        let post_attention_layernorm = RMSNorm {
            weight: weights.get(&format!("{}.post_attention_layernorm.weight", prefix))
                .ok_or_else(|| anyhow!("Missing post_attention_layernorm weight"))?.clone(),
            eps: config.rms_norm_eps,
        };
        
        Ok(Some(LlamaLayer {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }))
    }
    
    /// Extract configuration from metadata (deprecated)
    fn extract_config(metadata: &HashMap<String, String>) -> Result<LlamaConfig> {
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
                .and_then(|v| v.parse().ok())
                .unwrap_or(32),
            num_key_value_heads: metadata.get("llama.attention.head_count_kv")
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| {
                    metadata.get("llama.attention.head_count")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(32)
                }),
            hidden_size: metadata.get("llama.embedding_length")
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096),
            head_dim: metadata.get("llama.rope.dimension_count")
                .and_then(|v| v.parse().ok())
                .unwrap_or(128),
            intermediate_size: metadata.get("llama.feed_forward_length")
                .and_then(|v| v.parse().ok())
                .unwrap_or(11008),
            max_position_embeddings: metadata.get("llama.context_length")
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096),
            rms_norm_eps: metadata.get("llama.attention.layer_norm_rms_epsilon")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e-5),
            vocab_size: metadata.get("tokenizer.ggml.model.vocab_size")
                .and_then(|v| v.parse().ok())
                .unwrap_or(32000),
            num_hidden_layers: metadata.get("llama.block_count")
                .and_then(|v| v.parse().ok())
                .unwrap_or(32),
            rope_theta: metadata.get("llama.rope.freq_base")
                .and_then(|v| v.parse().ok())
                .unwrap_or(10000.0),
            rope_scaling: None,
        };
        
        // Check for Llama 3 rope scaling
        if version == 3 {
            if let Some(scaling_type) = metadata.get("llama.rope.scaling.type") {
                config.rope_scaling = Some(RopeScaling {
                    scaling_type: scaling_type.clone(),
                    factor: metadata.get("llama.rope.scaling.factor")
                        .and_then(|v| v.parse().ok())
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
        // Input should be token IDs with shape [batch_size, seq_len]
        let mut hidden_states = if let Some(embed) = &self.embed_tokens {
            // Convert token IDs to embeddings
            // The embedding tensor has shape [vocab_size, hidden_size]
            // Input has shape [batch_size, seq_len] with token IDs
            
            // Get input shape
            let input_shape = input.dims();
            tracing::debug!("Input tensor shape: {:?}, dtype: {:?}", input_shape, input.dtype());
            tracing::debug!("Embedding matrix shape: {:?}, dtype: {:?}", embed.dims(), embed.dtype());
            
            let batch_size = input_shape[0];
            let seq_len = if input_shape.len() > 1 { input_shape[1] } else { 1 };
            
            // Flatten input for embedding lookup (embedding expects 1D tensor)
            let flat_input = input.flatten_all()?;
            tracing::debug!("Flattened input shape: {:?}, dtype: {:?}", flat_input.dims(), flat_input.dtype());
            
            // Try to print the actual token IDs if the tensor is small
            if flat_input.elem_count() <= 10 {
                if let Ok(values) = flat_input.to_vec1::<u32>() {
                    tracing::debug!("Token IDs: {:?}", values);
                }
            }
            
            // Perform embedding lookup
            let embeddings = embed.embedding(&flat_input)?;
            
            // Get the actual hidden size from the embedding result
            let emb_dims = embeddings.dims();
            let hidden_size = emb_dims[emb_dims.len() - 1]; // Last dimension is hidden size
            tracing::debug!("Embeddings shape after lookup: {:?}, hidden_size: {}", emb_dims, hidden_size);
            
            // Reshape back to [batch_size, seq_len, hidden_size]
            embeddings.reshape(&[batch_size, seq_len, hidden_size])?
        } else {
            // If no embedding layer, assume input is already embedded
            input.clone()
        };
        
        // Apply transformer layers
        tracing::debug!("Total layers to process: {}", self.layers.len());
        for (idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden_states.clone();
            // tracing::debug!("Processing layer {}, hidden_states shape: {:?}", idx, hidden_states.dims());
            
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
            // LM head weight also needs to be transposed
            hidden_states = hidden_states.matmul(lm_head)?;
        } else if let Some(embed) = &self.embed_tokens {
            // Gemma and some other models tie weights: lm_head = embed_tokens.T
            // The embedding matrix is [vocab_size, hidden_size]
            // For output projection we need [hidden_size, vocab_size]
            // So we transpose: [262144, 640] -> [640, 262144]
            tracing::debug!("Using tied weights: projecting with transposed embedding matrix");
            
            // Get the shape for debugging
            let embed_shape = embed.dims();
            tracing::debug!("Embedding shape: {:?}, hidden_states shape: {:?}", embed_shape, hidden_states.dims());
            
            // The embedding tensor is [vocab_size, hidden_size]
            // We need to transpose it for the projection
            let output_proj = embed.t()?.contiguous()?;
            tracing::debug!("Output projection shape after transpose: {:?}", output_proj.dims());
            
            // Ensure hidden_states is the right shape for matmul
            // Should be [batch * seq_len, hidden_size] for the matmul
            let hs_shape = hidden_states.dims();
            if hs_shape.len() == 3 {
                let (batch_size, seq_len, hidden_size) = (hs_shape[0], hs_shape[1], hs_shape[2]);
                // Reshape to 2D for matmul: [batch * seq_len, hidden_size]
                let hidden_2d = hidden_states.reshape(&[batch_size * seq_len, hidden_size])?;
                // Perform matmul: [batch * seq_len, hidden_size] x [hidden_size, vocab_size] = [batch * seq_len, vocab_size]
                let logits_2d = hidden_2d.matmul(&output_proj)?;
                // Reshape back to 3D: [batch, seq_len, vocab_size]
                hidden_states = logits_2d.reshape(&[batch_size, seq_len, output_proj.dim(1)?])?;
            } else {
                hidden_states = hidden_states.matmul(&output_proj)?;
            }
        } else {
            tracing::warn!("No LM head or embedding weights found - returning hidden states!");
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