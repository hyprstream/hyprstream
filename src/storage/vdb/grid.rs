//! Sparse 3D grid implementation optimized for 99% sparse neural network weights

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// 3D coordinate for VDB grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Coordinate3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Coordinate3D {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
    
    pub fn x(&self) -> i32 { self.x }
    pub fn y(&self) -> i32 { self.y }
    pub fn z(&self) -> i32 { self.z }
}

/// Sparse VDB-style grid optimized for neural network weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseGrid {
    /// Background value (typically 0.0 for sparse networks)
    background: f32,
    
    /// Active voxels only - 99% compression
    active_voxels: HashMap<Coordinate3D, f32>,
    
    /// Grid metadata
    metadata: GridMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridMetadata {
    /// Bounding box of active region
    min_coord: Coordinate3D,
    max_coord: Coordinate3D,
    
    /// Total voxel count
    total_voxels: usize,
    
    /// Active voxel count
    active_count: usize,
    
    /// Creation timestamp
    created_at: u64,
    
    /// Last modified timestamp
    modified_at: u64,
}

impl SparseGrid {
    /// Create new sparse grid with background value
    pub fn new(background: f32) -> Self {
        Self {
            background,
            active_voxels: HashMap::new(),
            metadata: GridMetadata {
                min_coord: Coordinate3D::new(i32::MAX, i32::MAX, i32::MAX),
                max_coord: Coordinate3D::new(i32::MIN, i32::MIN, i32::MIN),
                total_voxels: 0,
                active_count: 0,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                modified_at: 0,
            },
        }
    }
    
    /// Set value at coordinate (activates voxel if non-background)
    pub fn set_value(&mut self, coord: Coordinate3D, value: f32) {
        if value == self.background {
            self.set_inactive(coord);
        } else {
            self.active_voxels.insert(coord, value);
            self.update_bounds(coord);
            self.metadata.active_count = self.active_voxels.len();
            self.metadata.modified_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
    }
    
    /// Get value at coordinate (returns background if inactive)
    pub fn get_value(&self, coord: Coordinate3D) -> f32 {
        self.active_voxels.get(&coord).copied().unwrap_or(self.background)
    }
    
    /// Deactivate voxel (set to background)
    pub fn set_inactive(&mut self, coord: Coordinate3D) {
        if self.active_voxels.remove(&coord).is_some() {
            self.metadata.active_count = self.active_voxels.len();
            self.metadata.modified_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
    }
    
    /// Check if voxel is active (non-background)
    pub fn is_active(&self, coord: Coordinate3D) -> bool {
        self.active_voxels.contains_key(&coord)
    }
    
    /// Get iterator over active voxels
    pub fn active_voxels(&self) -> impl Iterator<Item = (Coordinate3D, f32)> + '_ {
        self.active_voxels.iter().map(|(&coord, &value)| (coord, value))
    }
    
    /// Get active voxel count
    pub fn active_count(&self) -> usize {
        self.metadata.active_count
    }
    
    /// Get sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    pub fn sparsity(&self) -> f32 {
        if self.metadata.total_voxels == 0 {
            return 1.0;
        }
        1.0 - (self.metadata.active_count as f32 / self.metadata.total_voxels as f32)
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() +
        self.active_voxels.len() * (std::mem::size_of::<Coordinate3D>() + std::mem::size_of::<f32>())
    }
    
    /// Infer original tensor shape from active voxels
    pub fn inferred_shape(&self) -> Vec<usize> {
        if self.active_voxels.is_empty() {
            return vec![0, 0];
        }
        
        let width = (self.metadata.max_coord.x - self.metadata.min_coord.x + 1) as usize;
        let height = (self.metadata.max_coord.y - self.metadata.min_coord.y + 1) as usize;
        
        vec![height, width]
    }
    
    /// Bulk update multiple voxels (optimized for streaming)
    pub fn bulk_update(&mut self, updates: &HashMap<Coordinate3D, f32>) {
        for (&coord, &value) in updates {
            if value == self.background {
                self.active_voxels.remove(&coord);
            } else {
                self.active_voxels.insert(coord, value);
                self.update_bounds(coord);
            }
        }
        
        self.metadata.active_count = self.active_voxels.len();
        self.metadata.modified_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
    
    /// Prune voxels below threshold (maintain sparsity)
    pub fn prune(&mut self, threshold: f32) {
        self.active_voxels.retain(|_, &mut value| value.abs() > threshold);
        self.metadata.active_count = self.active_voxels.len();
        self.metadata.modified_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
    
    /// Get grid statistics
    pub fn stats(&self) -> GridStats {
        GridStats {
            active_voxels: self.metadata.active_count,
            total_voxels: self.metadata.total_voxels,
            sparsity: self.sparsity(),
            memory_bytes: self.memory_usage(),
            bounds: (self.metadata.min_coord, self.metadata.max_coord),
        }
    }
    
    fn update_bounds(&mut self, coord: Coordinate3D) {
        self.metadata.min_coord.x = self.metadata.min_coord.x.min(coord.x);
        self.metadata.min_coord.y = self.metadata.min_coord.y.min(coord.y);
        self.metadata.min_coord.z = self.metadata.min_coord.z.min(coord.z);
        
        self.metadata.max_coord.x = self.metadata.max_coord.x.max(coord.x);
        self.metadata.max_coord.y = self.metadata.max_coord.y.max(coord.y);
        self.metadata.max_coord.z = self.metadata.max_coord.z.max(coord.z);
    }
}

/// Grid statistics for monitoring
#[derive(Debug, Clone)]
pub struct GridStats {
    pub active_voxels: usize,
    pub total_voxels: usize,
    pub sparsity: f32,
    pub memory_bytes: usize,
    pub bounds: (Coordinate3D, Coordinate3D),
}

/// Sparse weights representation optimized for neural networks
#[derive(Debug, Clone)]
pub struct SparseWeights {
    /// Original tensor shape
    pub shape: Vec<usize>,
    
    /// Active weights (linear index -> value)
    active_weights: HashMap<usize, f32>,
}

impl SparseWeights {
    /// Create new sparse weights with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            active_weights: HashMap::new(),
        }
    }
    
    /// Set weight value
    pub fn set(&mut self, index: usize, value: f32) {
        if value.abs() > 1e-6 {
            self.active_weights.insert(index, value);
        } else {
            self.active_weights.remove(&index);
        }
    }
    
    /// Get weight value
    pub fn get(&self, index: usize) -> f32 {
        self.active_weights.get(&index).copied().unwrap_or(0.0)
    }
    
    /// Iterator over active weights
    pub fn active_iter(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.active_weights.iter().map(|(&idx, &val)| (idx, val))
    }
    
    /// Number of active (non-zero) weights
    pub fn active_count(&self) -> usize {
        self.active_weights.len()
    }
    
    /// Total number of weights (dense size)
    pub fn dense_size(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Sparsity ratio
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.active_count() as f32 / self.dense_size() as f32)
    }
    
    /// Memory usage of sparse representation
    pub fn sparse_memory(&self) -> usize {
        self.active_weights.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
    }
    
    /// Memory usage in bytes (total including metadata)
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.sparse_memory()
    }
    
    /// Calculate mean value of active weights
    pub fn mean_value(&self) -> f32 {
        if self.active_weights.is_empty() {
            0.0
        } else {
            let sum: f32 = self.active_weights.values().sum();
            sum / self.active_weights.len() as f32
        }
    }

    /// Get weight value by 3D coordinate (required for temporal streaming)
    pub fn get_coordinate(&self, coord: Coordinate3D) -> f32 {
        let linear_idx = self.coord_to_linear(coord);
        self.get(linear_idx)
    }

    /// Get weight value by 3D coordinate (optional variant)
    pub fn get_coordinate_opt(&self, coord: Coordinate3D) -> Option<f32> {
        let linear_idx = self.coord_to_linear(coord);
        self.active_weights.get(&linear_idx).copied()
    }

    /// Set weight by 3D coordinate
    pub fn set_coordinate(&mut self, coord: Coordinate3D, value: f32) {
        let linear_idx = self.coord_to_linear(coord);
        self.set(linear_idx, value);
    }

    /// Iterator over active weights as 3D coordinates
    pub fn active_coord_iter(&self) -> impl Iterator<Item = (Coordinate3D, f32)> + '_ {
        self.active_weights.iter().map(|(&idx, &val)| {
            let coord = self.linear_to_coord(idx);
            (coord, val)
        })
    }

    /// Convert 3D coordinate to linear index
    fn coord_to_linear(&self, coord: Coordinate3D) -> usize {
        match self.shape.len() {
            2 => {
                let w = self.shape[1];
                (coord.y() as usize) * w + (coord.x() as usize)
            }
            3 => {
                let h = self.shape[1];
                let w = self.shape[2];
                (coord.z() as usize) * h * w + (coord.y() as usize) * w + (coord.x() as usize)
            }
            _ => coord.x() as usize // Fallback for 1D
        }
    }

    /// Convert linear index to 3D coordinate
    pub fn linear_to_coord(&self, index: usize) -> Coordinate3D {
        match self.shape.len() {
            2 => {
                let w = self.shape[1];
                let y = index / w;
                let x = index % w;
                Coordinate3D::new(x as i32, y as i32, 0)
            }
            3 => {
                let h = self.shape[1];
                let w = self.shape[2];
                let z = index / (h * w);
                let remainder = index % (h * w);
                let y = remainder / w;
                let x = remainder % w;
                Coordinate3D::new(x as i32, y as i32, z as i32)
            }
            _ => Coordinate3D::new(index as i32, 0, 0) // Fallback
        }
    }
    
    /// Calculate standard deviation of active weights
    pub fn std_value(&self) -> f32 {
        if self.active_weights.is_empty() {
            0.0
        } else {
            let mean = self.mean_value();
            let variance: f32 = self.active_weights.values()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / self.active_weights.len() as f32;
            variance.sqrt()
        }
    }
    
    /// Get minimum value in active weights
    pub fn min_value(&self) -> f32 {
        self.active_weights.values()
            .copied()
            .fold(f32::INFINITY, f32::min)
    }
    
    /// Get maximum value in active weights
    pub fn max_value(&self) -> f32 {
        self.active_weights.values()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }
    
    /// Get shape as reference (to fix compilation errors)
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_grid_basic() {
        let mut grid = SparseGrid::new(0.0);
        
        // Set some active voxels
        grid.set_value(Coordinate3D::new(0, 0, 0), 1.0);
        grid.set_value(Coordinate3D::new(100, 200, 0), 0.5);
        grid.set_value(Coordinate3D::new(1000, 1000, 0), -0.3);
        
        assert_eq!(grid.active_count(), 3);
        assert_eq!(grid.get_value(Coordinate3D::new(0, 0, 0)), 1.0);
        assert_eq!(grid.get_value(Coordinate3D::new(50, 50, 0)), 0.0); // Background
        
        // Test deactivation
        grid.set_value(Coordinate3D::new(0, 0, 0), 0.0); // Set to background
        assert_eq!(grid.active_count(), 2);
        assert!(!grid.is_active(Coordinate3D::new(0, 0, 0)));
    }
    
    #[test]
    fn test_sparse_weights() {
        let mut weights = SparseWeights::new(vec![1536, 1536]);
        
        // Simulate 99% sparsity (only 1% active)
        let total_weights = 1536 * 1536;
        let active_count = total_weights / 100; // 1%
        
        for i in 0..active_count {
            weights.set(i * 100, 0.01); // Sparse pattern
        }
        
        assert_eq!(weights.active_count(), active_count);
        assert!(weights.sparsity() > 0.98); // Should be > 98% sparse
        
        // Memory efficiency check
        let dense_memory = total_weights * std::mem::size_of::<f32>();
        let sparse_memory = weights.sparse_memory();
        
        assert!(sparse_memory < dense_memory / 10); // At least 10x compression
    }
    
    #[test]
    fn test_bulk_updates() {
        let mut grid = SparseGrid::new(0.0);
        
        // Bulk update simulation (streaming scenario)
        let mut updates = HashMap::new();
        for i in 0..1000 {
            updates.insert(Coordinate3D::new(i, i, 0), i as f32 * 0.001);
        }
        
        grid.bulk_update(&updates);
        assert_eq!(grid.active_count(), 1000);
        
        // Verify random access
        assert_eq!(grid.get_value(Coordinate3D::new(500, 500, 0)), 0.5);
    }
    
    #[test]
    fn test_sparsity_maintenance() {
        let mut grid = SparseGrid::new(0.0);
        
        // Add many small values
        for i in 0..10000 {
            let value = if i % 100 == 0 { 0.1 } else { 0.0001 }; // Mix of significant and tiny values
            grid.set_value(Coordinate3D::new(i, 0, 0), value);
        }
        
        let before_prune = grid.active_count();
        
        // Prune small values
        grid.prune(0.01);
        
        let after_prune = grid.active_count();
        
        assert!(after_prune < before_prune);
        assert_eq!(after_prune, 100); // Only values >= 0.1 should remain
    }
}