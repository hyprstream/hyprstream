//! Fixed RoPE implementation for LlamaAttention
//! This shows how to properly integrate the RoPE module

use anyhow::{Result, anyhow};
use tch::{Device, Tensor};
use crate::runtime::rope::RoPE;

/// Fixed apply_rope method for LlamaAttention
/// This should replace the broken implementation in llama.rs
impl LlamaAttention {
    /// Apply Rotary Position Embeddings using the proper RoPE module
    pub fn apply_rope_fixed(
        &self,
        tensor: &Tensor,
        position_ids: &Tensor,
        rope: &mut RoPE,
    ) -> Result<Tensor> {
        // Debug logging
        tracing::debug!(
            "Applying RoPE: tensor shape={:?}, position_ids shape={:?}, rope_theta={}, layer_type={}",
            tensor.size(),
            position_ids.size(),
            self.rope_theta,
            self.layer_type
        );
        
        // Apply RoPE with proper position_ids
        rope.forward(tensor, Some(position_ids))
    }
    
    /// Create RoPE instance for this attention layer
    pub fn create_rope(&self, device: Device) -> Result<RoPE> {
        // Determine base frequency based on layer type
        let base = if self.layer_type == "local" {
            // Local layers in Gemma3 use different base
            10000.0
        } else {
            // Global layers or standard models
            self.rope_theta as f64
        };
        
        // Create RoPE with appropriate configuration
        RoPE::new(
            self.head_dim as i64,
            base,
            8192, // Max sequence length (should come from config)
            device,
        )
    }
}

/// Alternative: Fixed standalone implementation without the RoPE module
/// (Not recommended, but shows the correct math)
pub fn apply_rope_standalone(
    tensor: &Tensor,
    position_ids: &Tensor,
    rope_theta: f32,
    head_dim: i64,
) -> Result<Tensor> {
    // Get tensor info
    let device = tensor.device();
    let dtype = tensor.kind();
    let tensor_shape = tensor.size();
    
    // tensor shape: [batch, seq, heads, head_dim]
    let batch_size = tensor_shape[0];
    let seq_len = tensor_shape[1];
    let num_heads = tensor_shape[2];
    
    // CRITICAL FIX 1: Actually use the position_ids!
    // position_ids shape: [seq_len] or [batch, seq_len]
    let positions = if position_ids.dim() == 1 {
        // Expand to batch dimension if needed
        position_ids.unsqueeze(0).expand(&[batch_size, seq_len], false)
    } else {
        position_ids.shallow_clone()
    };
    
    // CRITICAL FIX 2: Correct frequency calculation
    // Create inverse frequencies: theta^(-2i/dim) for i in [0, dim/2)
    let half_dim = head_dim / 2;
    let freqs = Tensor::arange(half_dim, (tch::Kind::Float, device))
        / head_dim as f64;
    let inv_freq = (-(freqs * 2.0)).exp() * (rope_theta as f64).ln();
    let inv_freq = inv_freq.exp();
    
    // CRITICAL FIX 3: Use tensor operations for efficiency
    // Compute position * frequency for all positions
    // positions: [batch, seq_len], inv_freq: [head_dim/2]
    // Result: [batch, seq_len, head_dim/2]
    let pos_flat = positions.flatten(0, -1).to_dtype(tch::Kind::Float, false, false);
    let angles = pos_flat.unsqueeze(-1).matmul(&inv_freq.unsqueeze(0));
    
    // Reshape back to [batch, seq_len, head_dim/2]
    let angles = angles.reshape(&[batch_size, seq_len, half_dim]);
    
    // Compute sin and cos
    let sin = angles.sin().to_dtype(dtype, false, false);
    let cos = angles.cos().to_dtype(dtype, false, false);
    
    // Expand sin/cos to match tensor dimensions
    // From [batch, seq, head_dim/2] to [batch, seq, 1, head_dim/2] for broadcasting
    let sin = sin.unsqueeze(2).expand(&[batch_size, seq_len, num_heads, half_dim], false);
    let cos = cos.unsqueeze(2).expand(&[batch_size, seq_len, num_heads, half_dim], false);
    
    // Apply rotary embedding
    // Split tensor into two halves
    let x1 = tensor.narrow(-1, 0, half_dim);
    let x2 = tensor.narrow(-1, half_dim, half_dim);
    
    // Apply rotation formula
    // x_rotated = [x1*cos - x2*sin, x1*sin + x2*cos]
    let x1_rot = &x1 * &cos - &x2 * &sin;
    let x2_rot = &x1 * &sin + &x2 * &cos;
    
    // Concatenate back
    Ok(Tensor::cat(&[x1_rot, x2_rot], -1))
}

/// Create cached RoPE instances for all layers
pub struct RoPECache {
    ropes: std::collections::HashMap<usize, RoPE>,
}

impl RoPECache {
    pub fn new() -> Self {
        Self {
            ropes: std::collections::HashMap::new(),
        }
    }
    
    /// Get or create RoPE for a layer
    pub fn get_rope(
        &mut self,
        layer_idx: usize,
        head_dim: i64,
        rope_theta: f64,
        max_seq_len: i64,
        device: Device,
    ) -> Result<&mut RoPE> {
        if !self.ropes.contains_key(&layer_idx) {
            let rope = RoPE::new(head_dim, rope_theta, max_seq_len, device)?;
            self.ropes.insert(layer_idx, rope);
        }
        Ok(self.ropes.get_mut(&layer_idx).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;
    
    #[test]
    fn test_rope_with_position_ids() {
        let device = Device::Cpu;
        
        // Create test tensor: [batch=2, seq=4, heads=8, dim=64]
        let x = Tensor::randn(&[2, 4, 8, 64], (tch::Kind::Float, device));
        
        // Test with custom position_ids (not sequential)
        let position_ids = Tensor::from_slice(&[0i64, 2, 1, 3]).to_device(device);
        
        // Apply RoPE
        let mut rope = RoPE::new(64, 10000.0, 1024, device).unwrap();
        let result = rope.forward(&x, Some(&position_ids)).unwrap();
        
        // Check shape preserved
        assert_eq!(result.size(), x.size());
        
        // Verify positions were actually used (result should differ from sequential)
        let seq_positions = Tensor::arange(4, (tch::Kind::Int64, device));
        let result_seq = rope.forward(&x, Some(&seq_positions)).unwrap();
        
        // Results should be different if position_ids were used
        let diff = (&result - &result_seq).abs().sum(tch::Kind::Float);
        assert!(diff.double_value(&[]) > 0.01, "Position IDs were not used!");
    }
    
    #[test]
    fn test_rope_with_large_base_frequency() {
        let device = Device::Cpu;

        // Test RoPE with large base frequency (e.g., some Qwen2.5 models use 1,000,000)
        // Note: Always read rope_theta from model config.json - don't assume values
        let mut rope = RoPE::new(128, 1_000_000.0, 8192, device).unwrap();

        // Test tensor
        let x = Tensor::randn(&[1, 10, 4, 128], (tch::Kind::Float, device));
        let result = rope.forward(&x, None).unwrap();

        assert_eq!(result.size(), x.size());
        assert_eq!(rope.base(), 1_000_000.0);
    }
    
    #[test]
    fn test_rope_caching() {
        let device = Device::Cpu;
        let mut cache = RoPECache::new();
        
        // Get RoPE for layer 0
        let rope1 = cache.get_rope(0, 64, 10000.0, 1024, device).unwrap();
        let x = Tensor::randn(&[1, 8, 4, 64], (tch::Kind::Float, device));
        let _ = rope1.forward(&x, None).unwrap();
        
        // Get same RoPE again - should be cached
        let rope2 = cache.get_rope(0, 64, 10000.0, 1024, device).unwrap();
        let _ = rope2.forward(&x, None).unwrap();
        
        // Cache should only have 1 entry
        assert_eq!(cache.ropes.len(), 1);
    }
}