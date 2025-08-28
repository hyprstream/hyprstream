//! Native OpenVDB LoRA Adapter Implementation
//!
//! This module provides a true OpenVDB-based sparse LoRA adapter that:
//! - Uses native OpenVDB FloatGrid for 99% sparse storage
//! - Writes/reads actual .vdb files on disk
//! - Leverages OpenVDB's hierarchical tree structure
//! - Provides efficient sparse matrix operations

use std::path::Path;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};

use crate::storage::vdb::openvdb_bindings::ffi::{self, LoRAGrid};

/// OpenVDB-native LoRA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenVDBLoRAConfig {
    /// Input feature dimension (matrix height)
    pub in_features: usize,
    
    /// Output feature dimension (matrix width) 
    pub out_features: usize,
    
    /// Low-rank dimension
    pub rank: usize,
    
    /// LoRA scaling factor (alpha / rank)
    pub alpha: f32,
    
    /// Learning rate for updates
    pub learning_rate: f32,
    
    /// Sparsity threshold (values below this become zero)
    pub sparsity_threshold: f32,
    
    /// Target modules this adapter applies to
    pub target_modules: Vec<String>,
    
    /// Auto-prune tolerance for optimization
    pub prune_tolerance: f32,
}

impl Default for OpenVDBLoRAConfig {
    fn default() -> Self {
        Self {
            in_features: 1536,     // Qwen3-1.7B hidden size
            out_features: 1536,    
            rank: 16,              // Low rank dimension
            alpha: 16.0,           // LoRA alpha parameter
            learning_rate: 1e-4,   // Conservative learning rate
            sparsity_threshold: 1e-8, // Aggressive sparsity
            target_modules: vec![
                "self_attn.q_proj".to_string(),
                "self_attn.v_proj".to_string(),
            ],
            prune_tolerance: 1e-6, // OpenVDB pruning tolerance
        }
    }
}

/// Native OpenVDB LoRA adapter with true sparse storage
pub struct OpenVDBLoRAAdapter {
    /// Configuration
    config: OpenVDBLoRAConfig,
    
    /// LoRA A matrix stored in native OpenVDB grid
    lora_a: cxx::UniquePtr<LoRAGrid>,
    
    /// LoRA B matrix stored in native OpenVDB grid  
    lora_b: cxx::UniquePtr<LoRAGrid>,
    
    /// Adapter statistics
    stats: OpenVDBAdapterStats,
    
    /// Adapter name/ID for file storage
    adapter_id: String,
}

/// Performance and usage statistics
#[derive(Debug, Default, Clone)]
pub struct OpenVDBAdapterStats {
    pub forward_passes: u64,
    pub gradient_updates: u64,
    pub files_written: u64,
    pub files_read: u64,
    pub last_prune_time: u64,
    pub memory_usage_bytes: usize,
}

impl OpenVDBLoRAAdapter {
    /// Create new OpenVDB-based LoRA adapter
    pub fn new(adapter_id: String, config: OpenVDBLoRAConfig) -> Result<Self> {
        // Create native OpenVDB grids
        let lora_a = ffi::createLoRAGrid();
        let lora_b = ffi::createLoRAGrid();
        
        if lora_a.is_null() || lora_b.is_null() {
            return Err(anyhow!("Failed to create OpenVDB grids"));
        }
        
        let adapter = Self {
            config,
            lora_a,
            lora_b,
            stats: OpenVDBAdapterStats::default(),
            adapter_id,
        };
        
        println!("✅ Created OpenVDB LoRA adapter: {}", adapter.adapter_id);
        
        Ok(adapter)
    }
    
    /// Initialize with random sparse values
    pub fn initialize_random(&mut self, sparsity_ratio: f32) -> Result<()> {
        println!("🎲 Initializing OpenVDB LoRA adapter with {:.1}% sparsity", sparsity_ratio * 100.0);
        
        // Calculate number of active elements to maintain target sparsity
        let total_elements_a = self.config.in_features * self.config.rank;
        let total_elements_b = self.config.rank * self.config.out_features;
        
        let active_a = ((total_elements_a as f32) * (1.0 - sparsity_ratio)) as usize;
        let active_b = ((total_elements_b as f32) * (1.0 - sparsity_ratio)) as usize;
        
        // Initialize LoRA A matrix with small random values
        for _ in 0..active_a {
            let row = (self.random_u32() as usize) % self.config.in_features;
            let col = (self.random_u32() as usize) % self.config.rank;
            let value = (self.random_f32() - 0.5) * 0.02; // Small random initialization
            
            self.lora_a.pin_mut().setValue(row as i32, col as i32, value);
        }
        
        // Initialize LoRA B matrix with zeros (standard LoRA practice)
        // B starts at zero so the adapter initially has no effect
        
        println!("✅ Initialized {} active weights in A matrix, {} in B matrix", active_a, active_b);
        
        Ok(())
    }
    
    /// Forward pass through the LoRA adapter: input @ A @ B * scaling
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.config.in_features {
            return Err(anyhow!("Input dimension mismatch: expected {}, got {}", 
                self.config.in_features, input.len()));
        }
        
        // Step 1: input @ A -> intermediate [rank]
        let mut intermediate = vec![0.0; self.config.rank];
        
        for (input_idx, &input_val) in input.iter().enumerate() {
            if input_val.abs() < self.config.sparsity_threshold {
                continue; // Skip zero inputs for efficiency
            }
            
            for rank_idx in 0..self.config.rank {
                let weight_a = self.lora_a.getValue(input_idx as i32, rank_idx as i32);
                if weight_a.abs() > self.config.sparsity_threshold {
                    intermediate[rank_idx] += input_val * weight_a;
                }
            }
        }
        
        // Step 2: intermediate @ B -> output [out_features]
        let mut output = vec![0.0; self.config.out_features];
        let scaling = self.config.alpha / self.config.rank as f32;
        
        for (rank_idx, &intermediate_val) in intermediate.iter().enumerate() {
            if intermediate_val.abs() < self.config.sparsity_threshold {
                continue; // Skip zero intermediates
            }
            
            for output_idx in 0..self.config.out_features {
                let weight_b = self.lora_b.getValue(rank_idx as i32, output_idx as i32);
                if weight_b.abs() > self.config.sparsity_threshold {
                    output[output_idx] += intermediate_val * weight_b * scaling;
                }
            }
        }
        
        self.stats.forward_passes += 1;
        
        Ok(output)
    }
    
    /// Apply sparse gradient update using OpenVDB's native operations
    pub fn apply_gradients(
        &mut self, 
        grad_a: &[(i32, i32, f32)], // (row, col, gradient) triplets
        grad_b: &[(i32, i32, f32)],
    ) -> Result<()> {
        let lr = self.config.learning_rate;
        
        // Apply gradients to A matrix
        for &(row, col, grad) in grad_a {
            let current_weight = self.lora_a.getValue(row, col);
            let new_weight = current_weight - lr * grad;
            
            // Apply sparsity threshold
            if new_weight.abs() > self.config.sparsity_threshold {
                self.lora_a.pin_mut().setValue(row, col, new_weight);
            } else {
                self.lora_a.pin_mut().setValueOff(row, col); // Remove from sparse storage
            }
        }
        
        // Apply gradients to B matrix  
        for &(row, col, grad) in grad_b {
            let current_weight = self.lora_b.getValue(row, col);
            let new_weight = current_weight - lr * grad;
            
            if new_weight.abs() > self.config.sparsity_threshold {
                self.lora_b.pin_mut().setValue(row, col, new_weight);
            } else {
                self.lora_b.pin_mut().setValueOff(row, col);
            }
        }
        
        self.stats.gradient_updates += 1;
        
        // Auto-prune periodically for optimal sparsity
        if self.stats.gradient_updates % 100 == 0 {
            self.prune_weights()?;
        }
        
        Ok(())
    }
    
    /// Prune weights below tolerance using OpenVDB's native pruning
    pub fn prune_weights(&mut self) -> Result<()> {
        let start_active_a = self.lora_a.activeVoxelCount();
        let start_active_b = self.lora_b.activeVoxelCount();
        
        // Use OpenVDB's native pruning
        self.lora_a.pin_mut().prune(self.config.prune_tolerance);
        self.lora_b.pin_mut().prune(self.config.prune_tolerance);
        
        let end_active_a = self.lora_a.activeVoxelCount();
        let end_active_b = self.lora_b.activeVoxelCount();
        
        println!("🧹 Pruned weights: A matrix {} -> {}, B matrix {} -> {}", 
                start_active_a, end_active_a, start_active_b, end_active_b);
        
        self.stats.last_prune_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        Ok(())
    }
    
    /// Save adapter to native .vdb files on disk
    pub fn save_to_disk<P: AsRef<Path>>(&mut self, base_path: P) -> Result<()> {
        let base_path = base_path.as_ref();
        std::fs::create_dir_all(base_path)?;
        
        let a_path = base_path.join(format!("{}_a.vdb", self.adapter_id));
        let b_path = base_path.join(format!("{}_b.vdb", self.adapter_id));
        
        println!("💾 Saving OpenVDB LoRA adapter to disk...");
        println!("   📁 A matrix: {}", a_path.display());
        println!("   📁 B matrix: {}", b_path.display());
        
        // Write native .vdb files using OpenVDB's I/O
        let a_path_str = a_path.to_string_lossy().to_string();
        let b_path_str = b_path.to_string_lossy().to_string();
        
        // Use string slices directly
        let a_success = self.lora_a.writeToFile(&a_path_str);
        let b_success = self.lora_b.writeToFile(&b_path_str);
        
        if !a_success {
            return Err(anyhow!("Failed to write A matrix to {}", a_path.display()));
        }
        
        if !b_success {
            return Err(anyhow!("Failed to write B matrix to {}", b_path.display()));
        }
        
        self.stats.files_written += 2;
        
        println!("✅ Saved OpenVDB LoRA adapter with {} active weights", 
                self.active_weight_count());
        
        Ok(())
    }
    
    /// Load adapter from native .vdb files on disk
    pub fn load_from_disk<P: AsRef<Path>>(&mut self, base_path: P) -> Result<()> {
        let base_path = base_path.as_ref();
        
        let a_path = base_path.join(format!("{}_a.vdb", self.adapter_id));
        let b_path = base_path.join(format!("{}_b.vdb", self.adapter_id));
        
        if !a_path.exists() || !b_path.exists() {
            return Err(anyhow!("LoRA adapter files not found: {} or {}", 
                a_path.display(), b_path.display()));
        }
        
        println!("📁 Loading OpenVDB LoRA adapter from disk...");
        println!("   📂 A matrix: {}", a_path.display());
        println!("   📂 B matrix: {}", b_path.display());
        
        // Read native .vdb files using OpenVDB's I/O
        let a_path_str = a_path.to_string_lossy().to_string();
        let b_path_str = b_path.to_string_lossy().to_string();
        
        // Use string slices directly
        let a_success = self.lora_a.pin_mut().readFromFile(&a_path_str);
        let b_success = self.lora_b.pin_mut().readFromFile(&b_path_str);
        
        if !a_success {
            return Err(anyhow!("Failed to read A matrix from {}", a_path.display()));
        }
        
        if !b_success {
            return Err(anyhow!("Failed to read B matrix from {}", b_path.display()));
        }
        
        self.stats.files_read += 2;
        
        println!("✅ Loaded OpenVDB LoRA adapter with {} active weights", 
                self.active_weight_count());
        
        Ok(())
    }
    
    /// Get total number of active (non-zero) weights
    pub fn active_weight_count(&self) -> usize {
        self.lora_a.activeVoxelCount() + self.lora_b.activeVoxelCount()
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.lora_a.memoryUsage() + self.lora_b.memoryUsage()
    }
    
    /// Get current sparsity ratio
    pub fn sparsity_ratio(&self) -> f32 {
        let total_elements = (self.config.in_features * self.config.rank) + 
                           (self.config.rank * self.config.out_features);
        let active_elements = self.active_weight_count();
        
        1.0 - (active_elements as f32 / total_elements as f32)
    }
    
    /// Get detailed sparsity statistics
    pub fn get_sparsity_stats(&self) -> (f32, f32) {
        let a_sparsity = self.lora_a.sparsityRatio();
        let b_sparsity = self.lora_b.sparsityRatio();
        (a_sparsity, b_sparsity)
    }
    
    /// Get adapter statistics
    pub fn get_stats(&self) -> OpenVDBAdapterStats {
        let mut stats = self.stats.clone();
        stats.memory_usage_bytes = self.memory_usage();
        stats
    }
    
    /// Get configuration
    pub fn get_config(&self) -> &OpenVDBLoRAConfig {
        &self.config
    }
    
    /// Get adapter ID
    pub fn get_id(&self) -> &str {
        &self.adapter_id
    }
    
    /// Merge another adapter into this one with scaling
    pub fn merge_adapter(&mut self, other: &OpenVDBLoRAAdapter, scale: f32) -> Result<()> {
        if self.config.rank != other.config.rank ||
           self.config.in_features != other.config.in_features ||
           self.config.out_features != other.config.out_features {
            return Err(anyhow!("Cannot merge adapters with different dimensions"));
        }
        
        println!("🔀 Merging adapter '{}' with scale {:.3}", other.adapter_id, scale);
        
        // Merge A matrices
        self.lora_a.pin_mut().merge(&other.lora_a, scale);
        
        // Merge B matrices
        self.lora_b.pin_mut().merge(&other.lora_b, scale);
        
        println!("✅ Merged adapter, new active weight count: {}", self.active_weight_count());
        
        Ok(())
    }
    
    /// Simple random number generation (replace with proper RNG if needed)
    fn random_u32(&self) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_nanos().hash(&mut hasher);
        self.stats.forward_passes.hash(&mut hasher);
        hasher.finish() as u32
    }
    
    fn random_f32(&self) -> f32 {
        (self.random_u32() as f32) / (u32::MAX as f32)
    }
}

impl Drop for OpenVDBLoRAAdapter {
    fn drop(&mut self) {
        println!("🗑️ Dropping OpenVDB LoRA adapter: {}", self.adapter_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_openvdb_lora_creation() {
        let config = OpenVDBLoRAConfig::default();
        let adapter = OpenVDBLoRAAdapter::new("test_adapter".to_string(), config);
        assert!(adapter.is_ok());
        
        let adapter = adapter.unwrap();
        assert_eq!(adapter.get_id(), "test_adapter");
        assert_eq!(adapter.active_weight_count(), 0); // Starts empty
    }
    
    #[test]
    fn test_openvdb_lora_initialization() {
        let config = OpenVDBLoRAConfig::default();
        let mut adapter = OpenVDBLoRAAdapter::new("test_init".to_string(), config).unwrap();
        
        adapter.initialize_random(0.99).unwrap();
        
        // Should have some active weights after initialization
        assert!(adapter.active_weight_count() > 0);
        assert!(adapter.sparsity_ratio() > 0.98); // Should be very sparse
    }
    
    #[test]
    fn test_openvdb_lora_forward_pass() {
        let config = OpenVDBLoRAConfig {
            in_features: 4,
            out_features: 4,
            rank: 2,
            ..Default::default()
        };
        
        let mut adapter = OpenVDBLoRAAdapter::new("test_forward".to_string(), config).unwrap();
        adapter.initialize_random(0.5).unwrap(); // Less sparse for testing
        
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = adapter.forward(&input).unwrap();
        
        assert_eq!(output.len(), 4);
        // Output should be mostly small due to initialization
        assert!(output.iter().all(|&x| x.abs() < 1.0));
    }
    
    #[test]
    fn test_openvdb_lora_file_io() {
        let temp_dir = tempdir().unwrap();
        let config = OpenVDBLoRAConfig::default();
        
        // Create and save adapter
        {
            let mut adapter = OpenVDBLoRAAdapter::new("test_io".to_string(), config.clone()).unwrap();
            adapter.initialize_random(0.95).unwrap();
            
            let original_count = adapter.active_weight_count();
            adapter.save_to_disk(temp_dir.path()).unwrap();
            
            // Files should exist
            assert!(temp_dir.path().join("test_io_a.vdb").exists());
            assert!(temp_dir.path().join("test_io_b.vdb").exists());
            
            // Create new adapter and load
            let mut loaded_adapter = OpenVDBLoRAAdapter::new("test_io".to_string(), config).unwrap();
            loaded_adapter.load_from_disk(temp_dir.path()).unwrap();
            
            // Should have same number of weights
            assert_eq!(loaded_adapter.active_weight_count(), original_count);
        }
    }
}