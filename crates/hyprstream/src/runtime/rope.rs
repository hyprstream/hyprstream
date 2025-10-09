//! Rotary Position Embedding (RoPE) implementation for tch-rs
//! 
//! Based on rotary-embedding-tchrs but updated for tch 0.20 compatibility.
//! Original paper: https://arxiv.org/abs/2104.09864

use anyhow::{Result, anyhow};
use tch::{Device, Tensor, Kind};
use std::collections::HashMap;

/// RoPE (Rotary Position Embedding) implementation optimized for tch-rs
pub struct RoPE {
    /// Dimension of the embeddings (typically head_dim)
    dim: i64,
    /// Base frequency for the rotary embeddings (10000 for most models, 1000000 for Qwen3)
    base: f64,
    /// Maximum sequence length to precompute
    max_seq_len: i64,
    /// Device for computation
    device: Device,
    /// Data type for tensors (matches model dtype)
    dtype: Kind,
    /// Cached sin/cos values for efficiency
    cache: Option<(Tensor, Tensor)>,
    /// Cached sequence length to avoid recomputation
    cached_seq_len: i64,
}

impl RoPE {
    /// Create a new RoPE instance
    pub fn new(dim: i64, base: f64, max_seq_len: i64, device: Device) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(anyhow!("RoPE dimension must be even, got {}", dim));
        }
        
        Ok(Self {
            dim,
            base,
            max_seq_len,
            device,
            dtype: Kind::Float,  // Default to Float32 for backward compatibility
            cache: None,
            cached_seq_len: 0,
        })
    }
    
    /// Create a new RoPE instance with explicit dtype
    pub fn new_with_dtype(dim: i64, base: f64, max_seq_len: i64, device: Device, dtype: Kind) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(anyhow!("RoPE dimension must be even, got {}", dim));
        }
        
        Ok(Self {
            dim,
            base,
            max_seq_len,
            device,
            dtype,
            cache: None,
            cached_seq_len: 0,
        })
    }
    
    /// Create RoPE for standard models (base=10000)
    pub fn new_standard(dim: i64, max_seq_len: i64, device: Device) -> Result<Self> {
        Self::new(dim, 10000.0, max_seq_len, device)
    }
    
    /// Create RoPE for Qwen3 models (base=1000000 for long context)
    pub fn new_qwen3(dim: i64, max_seq_len: i64, device: Device) -> Result<Self> {
        Self::new(dim, 1000000.0, max_seq_len, device)
    }
    
    /// Create RoPE for standard models with dtype
    pub fn new_standard_with_dtype(dim: i64, max_seq_len: i64, device: Device, dtype: Kind) -> Result<Self> {
        Self::new_with_dtype(dim, 10000.0, max_seq_len, device, dtype)
    }
    
    /// Create RoPE for Qwen3 models with dtype
    pub fn new_qwen3_with_dtype(dim: i64, max_seq_len: i64, device: Device, dtype: Kind) -> Result<Self> {
        Self::new_with_dtype(dim, 1000000.0, max_seq_len, device, dtype)
    }
    
    /// Generate sin/cos embeddings for the given sequence length
    fn generate_embeddings(&self, seq_len: i64) -> Result<(Tensor, Tensor)> {
        // Create frequency bands: 1 / (base ^ (2i / dim)) for i in [0, dim/2)
        let half_dim = self.dim / 2;
        let inv_freq = {
            let positions = Tensor::arange(half_dim, (self.dtype, self.device));
            let exponents = &positions * 2.0 / (self.dim as f64);
            // Correct: base^(-2i/dim) = 1 / (base^(2i/dim))
            // Create a tensor filled with the base value that matches the shape of exponents
            let base_tensor = Tensor::full(
                &[half_dim], 
                self.base, 
                (self.dtype, self.device)
            );
            // Now both tensors have the same shape [half_dim], so pow will work
            base_tensor.pow(&(-exponents))
        };
        
        // Create position indices [0, 1, 2, ..., seq_len-1]
        let positions = Tensor::arange(seq_len, (self.dtype, self.device));
        
        // Compute angles: position[i] * inv_freq[j] for all i, j
        // Result shape: [seq_len, dim/2]
        let angles = positions.unsqueeze(-1).matmul(&inv_freq.unsqueeze(0));
        
        // Compute sin and cos
        let sin_vals = angles.sin();
        let cos_vals = angles.cos();
        
        Ok((sin_vals, cos_vals))
    }
    
    /// Get cached or compute sin/cos embeddings for the sequence length
    fn get_embeddings(&mut self, seq_len: i64) -> Result<(&Tensor, &Tensor)> {
        // Check if we need to recompute
        if self.cache.is_none() || seq_len > self.cached_seq_len {
            let compute_len = seq_len.max(self.max_seq_len);
            let (sin, cos) = self.generate_embeddings(compute_len)?;
            self.cache = Some((sin, cos));
            self.cached_seq_len = compute_len;
        }
        
        // Return reference to cached values, truncated to requested length
        let (sin, cos) = self.cache.as_ref()
            .ok_or_else(|| anyhow!("RoPE cache not initialized"))?;
        Ok((sin, cos))
    }
    
    /// Apply rotary position embedding to a tensor
    /// 
    /// # Arguments
    /// * `x` - Input tensor with shape [..., seq_len, heads, head_dim]
    /// * `position_ids` - Optional position indices, defaults to [0, 1, 2, ..., seq_len-1]
    /// 
    /// # Returns
    /// Tensor with same shape as input, with RoPE applied
    pub fn forward(&mut self, x: &Tensor, position_ids: Option<&Tensor>) -> Result<Tensor> {
        // Update dtype to match input tensor if needed
        let x_dtype = x.kind();
        if x_dtype != self.dtype {
            self.dtype = x_dtype;
            self.cache = None;  // Invalidate cache since dtype changed
            self.cached_seq_len = 0;
        }
        
        let x_shape = x.size();
        let seq_len = x_shape[x_shape.len() - 3]; // [..., seq_len, heads, head_dim]
        let head_dim = x_shape[x_shape.len() - 1];
        
        if head_dim != self.dim {
            return Err(anyhow!(
                "Input head dimension {} doesn't match RoPE dimension {}", 
                head_dim, self.dim
            ));
        }
        
        // Get sin/cos embeddings
        let (sin_full, cos_full) = self.get_embeddings(seq_len)?;
        
        // Handle custom position_ids or use default sequence
        let (sin, cos) = if let Some(pos_ids) = position_ids {
            // Ensure position_ids are in the correct dtype for indexing
            let pos_ids_int = if pos_ids.kind() != Kind::Int64 {
                pos_ids.to_kind(Kind::Int64)
            } else {
                pos_ids.shallow_clone()
            };
            let sin_selected = sin_full.index_select(0, &pos_ids_int);
            let cos_selected = cos_full.index_select(0, &pos_ids_int);
            (sin_selected, cos_selected)
        } else {
            let sin_truncated = sin_full.narrow(0, 0, seq_len);
            let cos_truncated = cos_full.narrow(0, 0, seq_len);
            (sin_truncated, cos_truncated)
        };
        
        // Apply rotary embedding
        self.apply_rotary_pos_emb(x, &sin, &cos)
    }
    
    /// Core rotary position embedding application
    fn apply_rotary_pos_emb(&self, x: &Tensor, sin: &Tensor, cos: &Tensor) -> Result<Tensor> {
        // Split x into two halves along the last dimension
        let half_dim = self.dim / 2;
        let x1 = x.narrow(-1, 0, half_dim);           // First half
        let x2 = x.narrow(-1, half_dim, half_dim);    // Second half
        
        // Expand sin/cos to match tensor dimensions
        // Input sin/cos shape: [seq_len, dim/2]
        // x1/x2 shape: [..., seq_len, heads, dim/2]
        // Need to add dimensions and broadcast
        
        // First, add dimensions to make sin/cos compatible with x1/x2
        // sin/cos: [seq_len, dim/2] -> [seq_len, 1, dim/2]
        let sin_reshaped = sin.unsqueeze(1); // Add heads dimension
        let cos_reshaped = cos.unsqueeze(1); // Add heads dimension
        
        // Now add batch dimensions if needed
        let x_dims = x1.dim();
        let mut sin_final = sin_reshaped.shallow_clone();
        let mut cos_final = cos_reshaped.shallow_clone();
        
        // If x has more than 3 dimensions, we need to add leading dimensions
        if x_dims > 3 {
            // Add leading dimensions (batch dimensions)
            for _ in 0..(x_dims - 3) {
                sin_final = sin_final.unsqueeze(0);
                cos_final = cos_final.unsqueeze(0);
            }
        }
        
        // Now expand to match x1/x2 shape
        let sin_expanded = sin_final.expand_as(&x1);
        let cos_expanded = cos_final.expand_as(&x1);
        
        // Apply rotation: 
        // x1' = x1 * cos - x2 * sin
        // x2' = x1 * sin + x2 * cos
        let x1_rotated = &x1 * &cos_expanded - &x2 * &sin_expanded;
        let x2_rotated = &x1 * &sin_expanded + &x2 * &cos_expanded;
        
        // Concatenate the rotated halves back together
        Ok(Tensor::cat(&[x1_rotated, x2_rotated], -1))
    }
    
    /// Update the base frequency (useful for different model variants)
    pub fn set_base(&mut self, base: f64) {
        if base != self.base {
            self.base = base;
            self.cache = None; // Invalidate cache
            self.cached_seq_len = 0;
        }
    }
    
    /// Get current dtype
    pub fn dtype(&self) -> Kind {
        self.dtype
    }
    
    /// Get current base frequency
    pub fn base(&self) -> f64 {
        self.base
    }
    
    /// Get embedding dimension
    pub fn dim(&self) -> i64 {
        self.dim
    }
}

/// RoPE cache manager for multiple model layers
pub struct RoPEManager {
    ropes: HashMap<String, RoPE>,
}

impl RoPEManager {
    pub fn new() -> Self {
        Self {
            ropes: HashMap::new(),
        }
    }
    
    /// Get or create a RoPE instance for a specific layer
    pub fn get_rope(&mut self, layer_name: &str, dim: i64, base: f64, max_seq_len: i64, device: Device) -> Result<&mut RoPE> {
        if !self.ropes.contains_key(layer_name) {
            let rope = RoPE::new(dim, base, max_seq_len, device)?;
            self.ropes.insert(layer_name.to_string(), rope);
        }
        self.ropes.get_mut(layer_name)
            .ok_or_else(|| anyhow!("RoPE not found for layer: {}", layer_name))
    }
    
    /// Get or create a RoPE instance with specific dtype
    pub fn get_rope_with_dtype(&mut self, layer_name: &str, dim: i64, base: f64, max_seq_len: i64, device: Device, dtype: Kind) -> Result<&mut RoPE> {
        if !self.ropes.contains_key(layer_name) {
            let rope = RoPE::new_with_dtype(dim, base, max_seq_len, device, dtype)?;
            self.ropes.insert(layer_name.to_string(), rope);
        }
        self.ropes.get_mut(layer_name)
            .ok_or_else(|| anyhow!("RoPE not found for layer: {}", layer_name))
    }
    
    /// Apply RoPE to a tensor using the appropriate layer configuration
    pub fn apply_rope(&mut self, layer_name: &str, x: &Tensor, position_ids: Option<&Tensor>, 
                     dim: i64, base: f64, max_seq_len: i64, device: Device) -> Result<Tensor> {
        let rope = self.get_rope(layer_name, dim, base, max_seq_len, device)?;
        rope.forward(x, position_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rope_basic() {
        let device = Device::Cpu;
        let mut rope = RoPE::new(64, 10000.0, 2048, device).unwrap();
        
        // Create test tensor: [batch=1, seq=8, heads=2, dim=64]
        let x = Tensor::randn(&[1, 8, 2, 64], (Kind::Float, device));
        let result = rope.forward(&x, None).unwrap();
        
        assert_eq!(result.size(), x.size());
    }
    
    #[test]
    fn test_rope_qwen3() {
        let device = Device::Cpu;
        let mut rope = RoPE::new_qwen3(128, 4096, device).unwrap();
        
        // Create test tensor for Qwen3: [batch=1, seq=16, heads=32, dim=128]
        let x = Tensor::randn(&[1, 16, 32, 128], (Kind::Float, device));
        let result = rope.forward(&x, None).unwrap();
        
        assert_eq!(result.size(), x.size());
        assert_eq!(rope.base(), 1000000.0);
    }
    
    #[test]
    fn test_rope_dimension_validation() {
        let device = Device::Cpu;
        
        // Should fail for odd dimensions
        let result = RoPE::new(63, 10000.0, 1024, device);
        assert!(result.is_err());
        
        // Should succeed for even dimensions
        let result = RoPE::new(64, 10000.0, 1024, device);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_rope_shape_broadcasting() {
        let device = Device::Cpu;
        let mut rope = RoPE::new(64, 10000.0, 2048, device).unwrap();
        
        // Test the exact shape from the error: [1, 6, 32, 64]
        let x = Tensor::randn(&[1, 6, 32, 64], (Kind::Float, device));
        let result = rope.forward(&x, None).unwrap();
        
        assert_eq!(result.size(), vec![1, 6, 32, 64]);
        
        // Test with different batch dimensions
        let x2 = Tensor::randn(&[2, 8, 16, 64], (Kind::Float, device));
        let result2 = rope.forward(&x2, None).unwrap();
        assert_eq!(result2.size(), vec![2, 8, 16, 64]);
        
        // Test 3D tensor (no batch dimension)
        let x3 = Tensor::randn(&[10, 4, 64], (Kind::Float, device));
        let result3 = rope.forward(&x3, None).unwrap();
        assert_eq!(result3.size(), vec![10, 4, 64]);
    }
}