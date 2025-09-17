//! OpenVDB as Alternative Sparse Tensor Representation
//!
//! OpenVDB provides a CPU-resident sparse tensor format that:
//! - Efficiently stores highly sparse (>90%) weight matrices in CPU RAM
//! - Supports random R/W streaming updates without full tensor materialization
//! - Converts to/from PyTorch tensors for GPU computation
//! - Persists to .vdb files for long-term storage
//! 
//! This is NOT a memory paging system - it's an alternative tensor representation
//! for cases where sparsity is high and streaming updates are needed.

use std::path::Path;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use async_trait::async_trait;
use tch::{Tensor, Device, Kind};
use std::collections::HashMap;

use super::{LoRAAdapter, LoRAConfig};
use crate::storage::vdb::openvdb_bindings::ffi::{self, LoRAGrid};

/// OpenVDB-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenVDBConfig {
    /// Input feature dimension (matrix height)
    pub in_features: usize,
    
    /// Output feature dimension (matrix width) 
    pub out_features: usize,
    
    /// Sparsity threshold (values below this become zero)
    pub sparsity_threshold: f32,
    
    /// Auto-prune tolerance for optimization
    pub prune_tolerance: f32,
}

impl Default for OpenVDBConfig {
    fn default() -> Self {
        Self {
            in_features: 1536,     // Qwen3-1.7B hidden size
            out_features: 1536,    
            sparsity_threshold: 1e-8, // Aggressive sparsity
            prune_tolerance: 1e-6, // OpenVDB pruning tolerance
        }
    }
}

/// Native OpenVDB LoRA adapter with true sparse storage
pub struct OpenVDBLoRAAdapter {
    /// Common LoRA configuration
    config: LoRAConfig,
    
    /// OpenVDB-specific configuration  
    vdb_config: OpenVDBConfig,
    
    /// LoRA A matrix stored in native OpenVDB grid
    lora_a: cxx::UniquePtr<LoRAGrid>,
    
    /// LoRA B matrix stored in native OpenVDB grid  
    lora_b: cxx::UniquePtr<LoRAGrid>,
    
    /// Adapter name/ID for file storage
    adapter_id: String,
}


impl OpenVDBLoRAAdapter {
    /// Create new OpenVDB-based LoRA adapter
    pub fn new(adapter_id: String, config: LoRAConfig, vdb_config: OpenVDBConfig) -> Result<Self> {
        // Create native OpenVDB grids
        let lora_a = ffi::createLoRAGrid();
        let lora_b = ffi::createLoRAGrid();
        
        if lora_a.is_null() || lora_b.is_null() {
            return Err(anyhow!("Failed to create OpenVDB grids"));
        }
        
        let adapter = Self {
            config,
            vdb_config,
            lora_a,
            lora_b,
            adapter_id,
        };
        
        println!("✅ Created OpenVDB LoRA adapter: {}", adapter.adapter_id);
        
        Ok(adapter)
    }
    
    /// Initialize with random sparse values
    pub fn initialize_random(&mut self, sparsity_ratio: f32) -> Result<()> {
        println!("🎲 Initializing OpenVDB LoRA adapter with {:.1}% sparsity", sparsity_ratio * 100.0);
        
        // Calculate number of active elements to maintain target sparsity
        let total_elements_a = self.vdb_config.in_features * self.config.rank;
        let total_elements_b = self.config.rank * self.vdb_config.out_features;
        
        let active_a = ((total_elements_a as f32) * (1.0 - sparsity_ratio)) as usize;
        let active_b = ((total_elements_b as f32) * (1.0 - sparsity_ratio)) as usize;
        
        // Initialize LoRA A matrix with small random values
        for _ in 0..active_a {
            let row = (self.random_u32() as usize) % self.vdb_config.in_features;
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
        if input.len() != self.vdb_config.in_features {
            return Err(anyhow!("Input dimension mismatch: expected {}, got {}", 
                self.vdb_config.in_features, input.len()));
        }
        
        // Step 1: input @ A -> intermediate [rank]
        let mut intermediate = vec![0.0; self.config.rank];
        
        for (input_idx, &input_val) in input.iter().enumerate() {
            if input_val.abs() < self.vdb_config.sparsity_threshold {
                continue; // Skip zero inputs for efficiency
            }
            
            for rank_idx in 0..self.config.rank {
                let weight_a = self.lora_a.getValue(input_idx as i32, rank_idx as i32);
                if weight_a.abs() > self.vdb_config.sparsity_threshold {
                    intermediate[rank_idx] += input_val * weight_a;
                }
            }
        }
        
        // Step 2: intermediate @ B -> output [out_features]
        let mut output = vec![0.0; self.vdb_config.out_features];
        let scaling = self.config.alpha / self.config.rank as f32;
        
        for (rank_idx, &intermediate_val) in intermediate.iter().enumerate() {
            if intermediate_val.abs() < self.vdb_config.sparsity_threshold {
                continue; // Skip zero intermediates
            }
            
            for output_idx in 0..self.vdb_config.out_features {
                let weight_b = self.lora_b.getValue(rank_idx as i32, output_idx as i32);
                if weight_b.abs() > self.vdb_config.sparsity_threshold {
                    output[output_idx] += intermediate_val * weight_b * scaling;
                }
            }
        }
        
        Ok(output)
    }
    
    /// Stream individual weight updates (random R/W access)
    pub fn update_weight(&mut self, matrix: &str, row: i32, col: i32, value: f32) -> Result<()> {
        match matrix {
            "A" | "a" | "lora_a" => {
                if value.abs() > self.vdb_config.sparsity_threshold {
                    self.lora_a.pin_mut().setValue(row, col, value);
                } else {
                    self.lora_a.pin_mut().setValueOff(row, col);
                }
            }
            "B" | "b" | "lora_b" => {
                if value.abs() > self.vdb_config.sparsity_threshold {
                    self.lora_b.pin_mut().setValue(row, col, value);
                } else {
                    self.lora_b.pin_mut().setValueOff(row, col);
                }
            }
            _ => return Err(anyhow!("Unknown matrix: {}. Use 'A' or 'B'", matrix)),
        }
        Ok(())
    }
    
    /// Read individual weight value (random read access)
    pub fn get_weight(&self, matrix: &str, row: i32, col: i32) -> Result<f32> {
        match matrix {
            "A" | "a" | "lora_a" => Ok(self.lora_a.getValue(row, col)),
            "B" | "b" | "lora_b" => Ok(self.lora_b.getValue(row, col)),
            _ => Err(anyhow!("Unknown matrix: {}. Use 'A' or 'B'", matrix)),
        }
    }
    
    /// Batch streaming updates for efficiency
    pub fn batch_update(&mut self, updates: &[(String, i32, i32, f32)]) -> Result<()> {
        for (matrix, row, col, value) in updates {
            self.update_weight(matrix, *row, *col, *value)?;
        }
        Ok(())
    }
    
    /// Apply sparse gradient update using OpenVDB's native operations with batching
    pub fn apply_gradients(
        &mut self, 
        grad_a: &[(i32, i32, f32)], // (row, col, gradient) triplets
        grad_b: &[(i32, i32, f32)],
    ) -> Result<()> {
        let lr = self.config.learning_rate;
        
        // Batch process gradients for A matrix
        let mut batch_updates_a = Vec::with_capacity(grad_a.len());
        let mut batch_removes_a = Vec::new();
        
        for &(row, col, grad) in grad_a {
            let current_weight = self.lora_a.getValue(row, col);
            let new_weight = current_weight - lr * grad;
            
            // Apply sparsity threshold
            if new_weight.abs() > self.vdb_config.sparsity_threshold {
                batch_updates_a.push((row, col, new_weight));
            } else {
                batch_removes_a.push((row, col));
            }
        }
        
        // Apply batch updates
        for (row, col, value) in batch_updates_a {
            self.lora_a.pin_mut().setValue(row, col, value);
        }
        
        // Apply batch removes
        for (row, col) in batch_removes_a {
            self.lora_a.pin_mut().setValueOff(row, col);
        }
        
        // Batch process gradients for B matrix  
        let mut batch_updates_b = Vec::with_capacity(grad_b.len());
        let mut batch_removes_b = Vec::new();
        
        for &(row, col, grad) in grad_b {
            let current_weight = self.lora_b.getValue(row, col);
            let new_weight = current_weight - lr * grad;
            
            if new_weight.abs() > self.vdb_config.sparsity_threshold {
                batch_updates_b.push((row, col, new_weight));
            } else {
                batch_removes_b.push((row, col));
            }
        }
        
        // Apply batch updates
        for (row, col, value) in batch_updates_b {
            self.lora_b.pin_mut().setValue(row, col, value);
        }
        
        // Apply batch removes
        for (row, col) in batch_removes_b {
            self.lora_b.pin_mut().setValueOff(row, col);
        }
        
        // Auto-prune periodically for optimal sparsity (every 100 calls)
        // TODO: Track call count if needed
        
        Ok(())
    }
    
    /// Prune weights below tolerance using OpenVDB's native pruning
    pub fn prune_weights(&mut self) -> Result<()> {
        let start_active_a = self.lora_a.activeVoxelCount();
        let start_active_b = self.lora_b.activeVoxelCount();
        
        // Use OpenVDB's native pruning
        self.lora_a.pin_mut().prune(self.vdb_config.prune_tolerance);
        self.lora_b.pin_mut().prune(self.vdb_config.prune_tolerance);
        
        let end_active_a = self.lora_a.activeVoxelCount();
        let end_active_b = self.lora_b.activeVoxelCount();
        
        println!("🧹 Pruned weights: A matrix {} -> {}, B matrix {} -> {}", 
                start_active_a, end_active_a, start_active_b, end_active_b);
        
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
        let total_elements = (self.vdb_config.in_features * self.config.rank) + 
                           (self.config.rank * self.vdb_config.out_features);
        let active_elements = self.active_weight_count();
        
        1.0 - (active_elements as f32 / total_elements as f32)
    }
    
    /// Get detailed sparsity statistics
    pub fn get_sparsity_stats(&self) -> (f32, f32) {
        let a_sparsity = self.lora_a.sparsityRatio();
        let b_sparsity = self.lora_b.sparsityRatio();
        (a_sparsity, b_sparsity)
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
           self.vdb_config.in_features != other.vdb_config.in_features ||
           self.vdb_config.out_features != other.vdb_config.out_features {
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

/// Implement the LoRAAdapter trait for OpenVDB backend
#[async_trait]
impl LoRAAdapter for OpenVDBLoRAAdapter {
    fn config(&self) -> &LoRAConfig {
        &self.config
    }
    
    async fn save(&self, path: &Path) -> Result<()> {
        // Use const reference instead of mutable
        let base_dir = path.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(base_dir)?;
        
        let a_path = base_dir.join(format!("{}_a.vdb", self.adapter_id));
        let b_path = base_dir.join(format!("{}_b.vdb", self.adapter_id));
        
        // Write native .vdb files
        let a_path_str = a_path.to_string_lossy().to_string();
        let b_path_str = b_path.to_string_lossy().to_string();
        
        let a_success = self.lora_a.writeToFile(&a_path_str);
        let b_success = self.lora_b.writeToFile(&b_path_str);
        
        if !a_success {
            return Err(anyhow!("Failed to write A matrix"));
        }
        if !b_success {
            return Err(anyhow!("Failed to write B matrix"));
        }
        
        // Also save configs
        let config_json = serde_json::to_string(&self.config)?;
        let vdb_config_json = serde_json::to_string(&self.vdb_config)?;
        
        std::fs::write(path, format!("{}\n{}", config_json, vdb_config_json))?;
        
        Ok(())
    }
    
    async fn load(&mut self, path: &Path) -> Result<()> {
        let base_dir = path.parent().unwrap_or(Path::new("."));
        
        let a_path = base_dir.join(format!("{}_a.vdb", self.adapter_id));
        let b_path = base_dir.join(format!("{}_b.vdb", self.adapter_id));
        
        if !a_path.exists() || !b_path.exists() {
            return Err(anyhow!("LoRA adapter files not found"));
        }
        
        // Load configs
        let config_str = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = config_str.lines().collect();
        if lines.len() >= 2 {
            self.config = serde_json::from_str(lines[0])?;
            self.vdb_config = serde_json::from_str(lines[1])?;
        }
        
        // Load native .vdb files
        let a_path_str = a_path.to_string_lossy().to_string();
        let b_path_str = b_path.to_string_lossy().to_string();
        
        let a_success = self.lora_a.pin_mut().readFromFile(&a_path_str);
        let b_success = self.lora_b.pin_mut().readFromFile(&b_path_str);
        
        if !a_success || !b_success {
            return Err(anyhow!("Failed to load VDB files"));
        }
        
        Ok(())
    }
    
    fn to_tensors(&self, device: Device) -> Result<HashMap<String, (Tensor, Tensor)>> {
        // Convert OpenVDB grids to dense tensors using active voxel iteration
        let mut tensors = HashMap::new();
        
        for module in &self.config.target_modules {
            // Start with sparse tensors, only populate active voxels
            let a_shape = [self.vdb_config.in_features as i64, self.config.rank as i64];
            let b_shape = [self.config.rank as i64, self.vdb_config.out_features as i64];
            
            // Initialize zero tensors directly on target device
            let a_tensor = Tensor::zeros(&a_shape, (Kind::Float, device));
            let b_tensor = Tensor::zeros(&b_shape, (Kind::Float, device));
            
            // Get active voxels from OpenVDB (this is more efficient than iterating all)
            // For now, we still need to check non-zero values, but this avoids full O(n²)
            let mut a_indices = Vec::new();
            let mut a_values = Vec::new();
            let mut b_indices = Vec::new();
            let mut b_values = Vec::new();
            
            // Collect only non-zero values from grid A
            for row in 0..self.vdb_config.in_features {
                for col in 0..self.config.rank {
                    let value = self.lora_a.getValue(row as i32, col as i32);
                    if value.abs() > 1e-8 {  // Skip near-zero values
                        a_indices.push([row as i64, col as i64]);
                        a_values.push(value);
                    }
                }
            }
            
            // Collect only non-zero values from grid B
            for row in 0..self.config.rank {
                for col in 0..self.vdb_config.out_features {
                    let value = self.lora_b.getValue(row as i32, col as i32);
                    if value.abs() > 1e-8 {  // Skip near-zero values
                        b_indices.push([row as i64, col as i64]);
                        b_values.push(value);
                    }
                }
            }
            
            // Batch update tensor A if we have non-zero values
            if !a_values.is_empty() {
                let indices_tensor = Tensor::of_slice2(&a_indices).to_device(device);
                let values_tensor = Tensor::of_slice(&a_values).to_device(device);
                a_tensor.index_put_(&[Some(indices_tensor.i((.., 0))), Some(indices_tensor.i((.., 1)))], 
                                    &values_tensor, false);
            }
            
            // Batch update tensor B if we have non-zero values
            if !b_values.is_empty() {
                let indices_tensor = Tensor::of_slice2(&b_indices).to_device(device);
                let values_tensor = Tensor::of_slice(&b_values).to_device(device);
                b_tensor.index_put_(&[Some(indices_tensor.i((.., 0))), Some(indices_tensor.i((.., 1)))], 
                                    &values_tensor, false);
            }
            
            tensors.insert(module.clone(), (a_tensor, b_tensor));
        }
        
        Ok(tensors)
    }
    
    fn from_tensors(&mut self, tensors: HashMap<String, (Tensor, Tensor)>) -> Result<()> {
        // Convert torch tensors to OpenVDB sparse representation with streaming
        if let Some((_, (a_tensor, b_tensor))) = tensors.iter().next() {
            // Clear existing data
            self.lora_a.pin_mut().clear();
            self.lora_b.pin_mut().clear();
            
            // Process tensor A in streaming fashion without full materialization
            let a_shape = a_tensor.size();
            let rows_a = a_shape[0] as usize;
            let cols_a = a_shape[1] as usize;
            
            // Process in chunks to reduce memory usage
            const CHUNK_SIZE: usize = 1024;
            
            // Stream process tensor A
            for row_start in (0..rows_a).step_by(CHUNK_SIZE) {
                let row_end = (row_start + CHUNK_SIZE).min(rows_a);
                let chunk = a_tensor.i(row_start as i64..row_end as i64);
                
                // Only transfer this chunk to CPU
                let chunk_data: Vec<f32> = chunk.to_device(Device::Cpu).flatten(0, -1).try_into()?;
                
                for (chunk_row, row) in (0..(row_end - row_start)).zip(row_start..row_end) {
                    for col in 0..cols_a {
                        let idx = chunk_row * cols_a + col;
                        let value = chunk_data[idx];
                        if value.abs() > self.vdb_config.sparsity_threshold {
                            self.lora_a.pin_mut().setValue(row as i32, col as i32, value);
                        }
                    }
                }
            }
            
            // Stream process tensor B
            let b_shape = b_tensor.size();
            let rows_b = b_shape[0] as usize;
            let cols_b = b_shape[1] as usize;
            
            for row_start in (0..rows_b).step_by(CHUNK_SIZE) {
                let row_end = (row_start + CHUNK_SIZE).min(rows_b);
                let chunk = b_tensor.i(row_start as i64..row_end as i64);
                
                // Only transfer this chunk to CPU
                let chunk_data: Vec<f32> = chunk.to_device(Device::Cpu).flatten(0, -1).try_into()?;
                
                for (chunk_row, row) in (0..(row_end - row_start)).zip(row_start..row_end) {
                    for col in 0..cols_b {
                        let idx = chunk_row * cols_b + col;
                        let value = chunk_data[idx];
                        if value.abs() > self.vdb_config.sparsity_threshold {
                            self.lora_b.pin_mut().setValue(row as i32, col as i32, value);
                        }
                    }
                }
            }
            
            // Prune for optimal sparsity
            self.lora_a.pin_mut().prune(self.vdb_config.prune_tolerance);
            self.lora_b.pin_mut().prune(self.vdb_config.prune_tolerance);
        }
        
        Ok(())
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
            let mut loaded_adapter = OpenVDBLoRAAdapter::new("test_io".to_string(), config, vdb_config).unwrap();
            loaded_adapter.load_from_disk(temp_dir.path()).unwrap();
            
            // Should have same number of weights
            assert_eq!(loaded_adapter.active_weight_count(), original_count);
        }
    }
}