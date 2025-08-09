//! Rust bindings for OpenVDB sparse storage

use crate::storage::vdb::grid::Coordinate3D;
use std::collections::HashMap;
use anyhow::Result;

#[cxx::bridge(namespace = "hyprstream")]
mod ffi {
    // WeightData struct (must be outside extern block)
    #[derive(Debug, Clone)]
    struct WeightData {
        row: i32,
        col: i32,
        weight: f32,
    }
    
    // Opaque C++ types
    unsafe extern "C++" {
        include!("openvdb_bridge.h");
        
        type LoRAGrid;
        // type ActiveWeightIterator;  // Temporarily disabled
        type SparseBatchOps;
    }
    
    // LoRAGrid methods
    unsafe extern "C++" {
        fn createLoRAGrid() -> UniquePtr<LoRAGrid>;
        
        // Core operations
        fn setValue(self: Pin<&mut LoRAGrid>, row: i32, col: i32, weight: f32);
        fn getValue(self: &LoRAGrid, row: i32, col: i32) -> f32;
        fn isActive(self: &LoRAGrid, row: i32, col: i32) -> bool;
        fn setValueOff(self: Pin<&mut LoRAGrid>, row: i32, col: i32);
        
        // Batch operations (simplified for CXX compatibility)
        fn sparseFill(self: Pin<&mut LoRAGrid>, min_row: i32, min_col: i32, max_row: i32, max_col: i32, value: f32);
        
        // Analysis
        fn activeVoxelCount(self: &LoRAGrid) -> usize;
        fn memoryUsage(self: &LoRAGrid) -> usize;
        fn sparsityRatio(self: &LoRAGrid) -> f32;
        
        // Optimization
        fn prune(self: Pin<&mut LoRAGrid>, tolerance: f32);
        fn merge(self: Pin<&mut LoRAGrid>, other: &LoRAGrid, scale: f32);
        
        // I/O - simplified to avoid string conversion issues for now
        // fn writeToFile(self: &LoRAGrid, filename: &str) -> bool;
        // fn readFromFile(self: Pin<&mut LoRAGrid>, filename: &str) -> bool;
    }
    
    // ActiveWeightIterator methods - temporarily disabled due to CXX struct issues
    // unsafe extern "C++" {
    //     fn createIterator(grid: &LoRAGrid) -> UniquePtr<ActiveWeightIterator>;
    //     fn hasNext(self: &ActiveWeightIterator) -> bool;
    //     fn next(self: Pin<&mut ActiveWeightIterator>) -> WeightData;
    //     fn reset(self: Pin<&mut ActiveWeightIterator>);
    // }
    
    // SparseBatchOps methods - simplified interface for CXX compatibility
    unsafe extern "C++" {
        fn applyDelta(base: Pin<&mut LoRAGrid>, delta: &LoRAGrid, learning_rate: f32);
        fn optimizeSparsity(grid: Pin<&mut LoRAGrid>, sparsity_threshold: f32);
    }
}

pub use ffi::*;

/// Safe Rust wrapper for OpenVDB LoRA storage
pub struct OpenVDBLoRAAdapter {
    grid: cxx::UniquePtr<ffi::LoRAGrid>,
    shape: (usize, usize),
    sparsity_threshold: f32,
}

impl OpenVDBLoRAAdapter {
    /// Create new sparse LoRA adapter
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        let grid = ffi::createLoRAGrid();
        
        Ok(Self {
            grid,
            shape: (rows, cols),
            sparsity_threshold: 1e-8,
        })
    }
    
    /// Update a single weight
    pub fn set_weight(&mut self, row: i32, col: i32, weight: f32) -> Result<()> {
        if self.is_valid_coord(row, col) {
            self.grid.pin_mut().setValue(row, col, weight);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Coordinates out of bounds: ({}, {})", row, col))
        }
    }
    
    /// Get weight value
    pub fn get_weight(&self, row: i32, col: i32) -> Result<f32> {
        if self.is_valid_coord(row, col) {
            Ok(self.grid.getValue(row, col))
        } else {
            Err(anyhow::anyhow!("Coordinates out of bounds: ({}, {})", row, col))
        }
    }
    
    /// Check if coordinate has an active (non-zero) weight
    pub fn is_active(&self, row: i32, col: i32) -> bool {
        self.is_valid_coord(row, col) && self.grid.isActive(row, col)
    }
    
    /// Remove weight (set inactive)
    pub fn remove_weight(&mut self, row: i32, col: i32) -> Result<()> {
        if self.is_valid_coord(row, col) {
            self.grid.pin_mut().setValueOff(row, col);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Coordinates out of bounds: ({}, {})", row, col))
        }
    }
    
    /// Batch update multiple weights efficiently (using individual setValue calls)
    pub fn batch_update(&mut self, updates: &[(i32, i32, f32)]) -> Result<()> {
        // Apply each update individually since CXX doesn't support complex batch types
        for &(row, col, weight) in updates {
            if self.is_valid_coord(row, col) {
                self.grid.pin_mut().setValue(row, col, weight);
            }
        }
        Ok(())
    }
    
    /// Fill a rectangular region with a constant value
    pub fn sparse_fill(&mut self, min_row: i32, min_col: i32, max_row: i32, max_col: i32, value: f32) -> Result<()> {
        if self.is_valid_coord(min_row, min_col) && self.is_valid_coord(max_row, max_col) {
            self.grid.pin_mut().sparseFill(min_row, min_col, max_row, max_col, value);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Invalid fill region"))
        }
    }
    
    /// Get number of active (non-zero) weights
    pub fn active_count(&self) -> usize {
        self.grid.activeVoxelCount()
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.grid.memoryUsage()
    }
    
    /// Get sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    pub fn sparsity_ratio(&self) -> f32 {
        self.grid.sparsityRatio()
    }
    
    /// Optimize sparse representation by removing small values and empty nodes
    pub fn optimize(&mut self) {
        self.grid.pin_mut().prune(self.sparsity_threshold);
    }
    
    /// Merge another adapter with optional scaling
    pub fn merge_with(&mut self, other: &OpenVDBLoRAAdapter, scale: f32) -> Result<()> {
        self.grid.pin_mut().merge(&other.grid, scale);
        Ok(())
    }
    
    /// Save adapter to file (temporarily disabled due to CXX string issues)
    pub fn save(&self, _filename: &str) -> Result<()> {
        // TODO: Re-enable once we resolve CXX string parameter issues
        // if self.grid.writeToFile(filename) {
        //     Ok(())
        // } else {
        //     Err(anyhow::anyhow!("Failed to write LoRA adapter to {}", filename))
        // }
        Err(anyhow::anyhow!("File I/O temporarily disabled"))
    }
    
    /// Load adapter from file (temporarily disabled due to CXX string issues)
    pub fn load(&mut self, _filename: &str) -> Result<()> {
        // TODO: Re-enable once we resolve CXX string parameter issues
        // if self.grid.pin_mut().readFromFile(filename) {
        //     Ok(())
        // } else {
        //     Err(anyhow::anyhow!("Failed to read LoRA adapter from {}", filename))
        // }
        Err(anyhow::anyhow!("File I/O temporarily disabled"))
    }
    
    /// Create iterator over active weights (temporarily disabled)
    pub fn active_weights(&self) -> OpenVDBActiveIterator {
        // TODO: Re-enable once CXX struct issues are resolved
        OpenVDBActiveIterator {
            _placeholder: 0,
        }
    }
    
    /// Get adapter shape
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    
    /// Convert to HashMap (for compatibility with existing code)
    pub fn to_hashmap(&self) -> HashMap<Coordinate3D, f32> {
        let mut map = HashMap::new();
        
        for (row, col, weight) in self.active_weights() {
            let coord = Coordinate3D::new(row, col, 0);
            map.insert(coord, weight);
        }
        
        map
    }
    
    /// Create from HashMap
    pub fn from_hashmap(hashmap: &HashMap<Coordinate3D, f32>, shape: (usize, usize)) -> Result<Self> {
        let mut adapter = Self::new(shape.0, shape.1)?;
        
        let updates: Vec<(i32, i32, f32)> = hashmap
            .iter()
            .map(|(coord, &weight)| (coord.x(), coord.y(), weight))
            .collect();
            
        adapter.batch_update(&updates)?;
        Ok(adapter)
    }
    
    /// Get all active weights (placeholder implementation)
    pub fn get_all_weights(&self) -> HashMap<Coordinate3D, f32> {
        self.to_hashmap()
    }
    
    fn is_valid_coord(&self, row: i32, col: i32) -> bool {
        row >= 0 && col >= 0 && 
        (row as usize) < self.shape.0 && 
        (col as usize) < self.shape.1
    }
    
    /// Sparse matrix multiplication (placeholder)
    pub fn sparse_multiply(&self, _input: &[f32], _output: &mut [f32]) -> Result<()> {
        // TODO: Implement sparse matrix multiplication using OpenVDB
        println!("⚠️ sparse_multiply not yet implemented");
        Ok(())
    }
    
    /// Get active voxel count
    pub fn active_voxel_count(&self) -> u64 {
        // TODO: Get actual count from OpenVDB grid
        0
    }
}

// SAFETY: OpenVDBLoRAAdapter contains C++ OpenVDB data structures which are thread-safe
// when properly accessed. We ensure exclusive access through Rust's borrowing rules.
unsafe impl Send for OpenVDBLoRAAdapter {}
unsafe impl Sync for OpenVDBLoRAAdapter {}

/// Iterator over active (non-zero) weights (temporarily disabled)
pub struct OpenVDBActiveIterator {
    // Placeholder fields
    _placeholder: u8,
}

impl OpenVDBActiveIterator {
    pub fn reset(&mut self) {
        // TODO: Implement when iterator is re-enabled
    }
}

impl Iterator for OpenVDBActiveIterator {
    type Item = (i32, i32, f32);
    
    fn next(&mut self) -> Option<Self::Item> {
        // TODO: Implement when iterator is re-enabled
        None
    }
}

/// Batch operations for multiple LoRA adapters
pub struct OpenVDBBatchOps;

impl OpenVDBBatchOps {
    /// Fuse multiple LoRA adapters with different scales (CPU implementation)
    pub fn fuse_adapters(adapters: &[&OpenVDBLoRAAdapter], scales: &[f32]) -> Result<OpenVDBLoRAAdapter> {
        if adapters.is_empty() || adapters.len() != scales.len() {
            return Err(anyhow::anyhow!("Invalid adapter count or scale mismatch"));
        }
        
        // Determine output shape (use largest dimensions)
        let max_shape = adapters.iter()
            .map(|adapter| adapter.shape())
            .fold((0, 0), |(max_r, max_c), (r, c)| {
                (max_r.max(r), max_c.max(c))
            });
        
        // Create new adapter for fused result
        let mut fused_adapter = OpenVDBLoRAAdapter::new(max_shape.0, max_shape.1)?;
        
        // Manually fuse by iterating over all adapters and their active weights
        for (adapter, &scale) in adapters.iter().zip(scales.iter()) {
            for (row, col, weight) in adapter.active_weights() {
                let scaled_weight = weight * scale;
                let current_weight = fused_adapter.get_weight(row, col).unwrap_or(0.0);
                fused_adapter.set_weight(row, col, current_weight + scaled_weight)?;
            }
        }
        
        Ok(fused_adapter)
    }
    
    /// Apply gradient update: base = base + learning_rate * delta
    pub fn apply_gradient(base: &mut OpenVDBLoRAAdapter, delta: &OpenVDBLoRAAdapter, learning_rate: f32) {
        ffi::applyDelta(base.grid.pin_mut(), &delta.grid, learning_rate);
    }
    
    /// Remove weights below threshold to increase sparsity
    pub fn enforce_sparsity(adapter: &mut OpenVDBLoRAAdapter, threshold: f32) {
        ffi::optimizeSparsity(adapter.grid.pin_mut(), threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_openvdb_basic_operations() -> Result<()> {
        let mut adapter = OpenVDBLoRAAdapter::new(1000, 1000)?;
        
        // Set some weights
        adapter.set_weight(10, 20, 0.5)?;
        adapter.set_weight(100, 200, -0.3)?;
        adapter.set_weight(500, 750, 1.2)?;
        
        // Check values
        assert_eq!(adapter.get_weight(10, 20)?, 0.5);
        assert_eq!(adapter.get_weight(100, 200)?, -0.3);
        assert_eq!(adapter.get_weight(500, 750)?, 1.2);
        assert_eq!(adapter.get_weight(0, 0)?, 0.0);  // Background value
        
        // Check active count
        assert_eq!(adapter.active_count(), 3);
        
        // Check sparsity
        let sparsity = adapter.sparsity_ratio();
        assert!(sparsity > 0.99);  // Should be very sparse
        
        Ok(())
    }
    
    #[test]
    fn test_batch_operations() -> Result<()> {
        let mut adapter = OpenVDBLoRAAdapter::new(100, 100)?;
        
        let updates = vec![
            (10, 10, 0.1),
            (20, 30, 0.2),
            (50, 60, 0.3),
            (80, 90, 0.4),
        ];
        
        adapter.batch_update(&updates)?;
        
        assert_eq!(adapter.active_count(), 4);
        assert_eq!(adapter.get_weight(10, 10)?, 0.1);
        assert_eq!(adapter.get_weight(80, 90)?, 0.4);
        
        Ok(())
    }
    
    #[test]
    fn test_iterator() -> Result<()> {
        let mut adapter = OpenVDBLoRAAdapter::new(50, 50)?;
        
        adapter.set_weight(5, 10, 0.5)?;
        adapter.set_weight(15, 25, 1.0)?;
        adapter.set_weight(35, 40, -0.8)?;
        
        let weights: Vec<(i32, i32, f32)> = adapter.active_weights().collect();
        
        assert_eq!(weights.len(), 3);
        
        // Check that all expected weights are present
        let has_weight = |row, col, expected_weight: f32| {
            weights.iter().any(|(r, c, w)| *r == row && *c == col && (*w - expected_weight).abs() < 1e-6)
        };
        
        assert!(has_weight(5, 10, 0.5));
        assert!(has_weight(15, 25, 1.0));
        assert!(has_weight(35, 40, -0.8));
        
        Ok(())
    }
}