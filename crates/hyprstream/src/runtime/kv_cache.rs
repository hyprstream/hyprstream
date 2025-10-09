//! Key-Value cache implementation for efficient autoregressive generation
//!
//! This module implements KV caching to avoid recomputing past key and value
//! states during inference, providing 10-50x speedup for long sequences.

use anyhow::{Result, anyhow};
use tch::{Tensor, Kind as DType};
use std::collections::HashMap;

/// KV cache for a single attention layer
#[derive(Debug)]
pub struct LayerKVCache {
    /// Cached keys with shape [batch, max_seq_len, num_heads, head_dim]
    pub keys: Option<Tensor>,
    /// Cached values with shape [batch, max_seq_len, num_heads, head_dim]  
    pub values: Option<Tensor>,
    /// Current sequence position (number of cached tokens)
    pub seq_pos: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl LayerKVCache {
    /// Create a new layer KV cache
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            keys: None,
            values: None,
            seq_pos: 0,
            max_seq_len,
        }
    }

    /// Update cache with new keys and values
    pub fn update(
        &mut self,
        new_keys: &Tensor,
        new_values: &Tensor,
        start_pos: usize,
    ) -> Result<()> {
        let (batch_size, seq_len, num_heads, head_dim) = {
            let k_size = new_keys.size();
            if k_size.len() != 4 {
                return Err(anyhow!("Expected 4D tensor for keys, got {:?}", k_size));
            }
            (k_size[0], k_size[1] as usize, k_size[2], k_size[3])
        };

        // Initialize cache if needed
        if self.keys.is_none() || self.values.is_none() {
            let device = new_keys.device();
            let dtype = new_keys.kind();
            
            // Allocate cache tensors
            self.keys = Some(Tensor::zeros(
                &[batch_size, self.max_seq_len as i64, num_heads, head_dim],
                (dtype, device),
            ));
            self.values = Some(Tensor::zeros(
                &[batch_size, self.max_seq_len as i64, num_heads, head_dim],
                (dtype, device),
            ));
        }

        // Get mutable references to cache tensors
        let cached_keys = self.keys.as_mut()
            .ok_or_else(|| anyhow!("Keys cache not initialized"))?;
        let cached_values = self.values.as_mut()
            .ok_or_else(|| anyhow!("Values cache not initialized"))?;

        // Check bounds
        if start_pos + seq_len > self.max_seq_len {
            return Err(anyhow!(
                "Cache overflow: trying to cache {} tokens starting at position {}, but max_seq_len is {}",
                seq_len, start_pos, self.max_seq_len
            ));
        }

        // Update cache using narrow and copy
        let end_pos = start_pos + seq_len;
        
        // Update keys: cached_keys[:, start_pos:end_pos] = new_keys
        let _ = cached_keys
            .narrow(1, start_pos as i64, seq_len as i64)
            .copy_(new_keys);
            
        // Update values: cached_values[:, start_pos:end_pos] = new_values  
        let _ = cached_values
            .narrow(1, start_pos as i64, seq_len as i64)
            .copy_(new_values);

        // Update position
        self.seq_pos = end_pos;

        Ok(())
    }

    /// Get cached keys and values up to current position
    pub fn get(&self) -> Result<(Tensor, Tensor)> {
        let cached_keys = self.keys.as_ref()
            .ok_or_else(|| anyhow!("Keys cache not initialized"))?;
        let cached_values = self.values.as_ref()
            .ok_or_else(|| anyhow!("Values cache not initialized"))?;

        if self.seq_pos == 0 {
            return Err(anyhow!("Cache is empty"));
        }

        // Return sliced tensors up to current position
        let keys_slice = cached_keys.narrow(1, 0, self.seq_pos as i64);
        let values_slice = cached_values.narrow(1, 0, self.seq_pos as i64);

        Ok((keys_slice, values_slice))
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
        self.seq_pos = 0;
    }

    /// Check if cache is initialized
    pub fn is_initialized(&self) -> bool {
        self.keys.is_some() && self.values.is_some()
    }
}

/// KV cache manager for all layers in a model
pub struct KVCacheManager {
    /// Cache for each layer
    layer_caches: HashMap<usize, LayerKVCache>,
    /// Maximum sequence length
    #[allow(dead_code)]
    max_seq_len: usize,
    /// Whether caching is enabled
    enabled: bool,
}

impl KVCacheManager {
    /// Create a new KV cache manager
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        let mut layer_caches = HashMap::new();
        for layer_idx in 0..num_layers {
            layer_caches.insert(layer_idx, LayerKVCache::new(max_seq_len));
        }
        
        Self {
            layer_caches,
            max_seq_len,
            enabled: true,
        }
    }

    /// Get cache for a specific layer
    pub fn get_layer_cache(&mut self, layer_idx: usize) -> Option<&mut LayerKVCache> {
        if !self.enabled {
            return None;
        }
        self.layer_caches.get_mut(&layer_idx)
    }

    /// Clear all caches
    pub fn clear_all(&mut self) {
        for cache in self.layer_caches.values_mut() {
            cache.clear();
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

        let mut total = 0;
        for cache in self.layer_caches.values() {
            if let (Some(keys), Some(values)) = (&cache.keys, &cache.values) {
                let element_size = match keys.kind() {
                    DType::Float => 4,
                    DType::Half => 2,
                    DType::BFloat16 => 2,
                    _ => 4,
                };
                
                let key_elements = keys.size().iter().product::<i64>() as usize;
                let value_elements = values.size().iter().product::<i64>() as usize;
                
                total += (key_elements + value_elements) * element_size;
            }
        }
        
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_kv_cache() -> Result<()> {
        use tch::Device;
        let device = Device::Cpu;
        let dtype = DType::Float;
        
        // Create cache
        let mut cache = LayerKVCache::new(100);
        
        // Create test tensors
        let batch_size = 2;
        let seq_len = 10;
        let num_heads = 8;
        let head_dim = 64;
        
        let keys = Tensor::randn(
            &[batch_size, seq_len as i64, num_heads, head_dim],
            (dtype, device),
        );
        let values = Tensor::randn(
            &[batch_size, seq_len as i64, num_heads, head_dim],
            (dtype, device),
        );
        
        // Update cache
        cache.update(&keys, &values, 0)?;
        assert_eq!(cache.seq_pos, 10);
        
        // Get cached values
        let (cached_keys, cached_values) = cache.get()?;
        assert_eq!(cached_keys.size(), vec![batch_size, 10, num_heads, head_dim]);
        assert_eq!(cached_values.size(), vec![batch_size, 10, num_heads, head_dim]);
        
        Ok(())
    }

    #[test]
    fn test_kv_cache_manager() {
        let num_layers = 32;
        let max_seq_len = 2048;
        
        let mut manager = KVCacheManager::new(num_layers, max_seq_len);
        
        // Check all layers are initialized
        for layer_idx in 0..num_layers {
            assert!(manager.get_layer_cache(layer_idx).is_some());
        }
        
        // Test enable/disable
        manager.set_enabled(false);
        assert!(manager.get_layer_cache(0).is_none());
        
        manager.set_enabled(true);
        assert!(manager.get_layer_cache(0).is_some());
    }
}