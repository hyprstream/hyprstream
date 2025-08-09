//! OpenVDB-backed sparse storage for adaptive layers
//! 
//! This module provides efficient storage and dynamic updates for 99% sparse 
//! adapter weights using OpenVDB's hierarchical grid structure.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::config::HyprConfig;
use std::sync::Arc;
use tokio::sync::RwLock;
use memmap2::MmapOptions;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use serde::{Serialize, Deserialize};

pub mod grid;
pub mod compression;
pub mod adapter_store;
pub mod neuralvdb_codec;
pub mod sparse_storage; // New VDB-first storage interface

// OpenVDB integration (only VDB backend)

pub mod openvdb_bindings;

// Hardware accelerated module requires VDB support

pub mod hardware_accelerated;

pub use grid::*;
pub use compression::*;
pub use adapter_store::*;
pub use sparse_storage::*; // Export new VDB-first interface

// Export OpenVDB bindings when VDB feature enabled

pub use openvdb_bindings::{
    OpenVDBLoRAAdapter, 
    OpenVDBActiveIterator, 
    OpenVDBBatchOps
};

// Export hardware accelerated features when VDB is available

pub use hardware_accelerated::*;

pub use neuralvdb_codec::*;

/// Configuration for VDB storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VDBConfig {
    /// Base storage directory
    pub storage_path: PathBuf,
    /// Compression algorithm (lz4, zstd, none)
    pub compression: String,
    /// Cache size in MB
    pub cache_size_mb: usize,
    /// Background value for sparse grids (typically 0.0)
    pub background_value: f32,
}

impl Default for VDBConfig {
    fn default() -> Self {
        let config = HyprConfig::load().unwrap_or_default();
        
        Self {
            storage_path: config.vdb_storage_dir().clone(),
            compression: "lz4".to_string(),
            cache_size_mb: 1000,
            background_value: 0.0,
        }
    }
}

/// Main VDB storage system for sparse adaptive layers
pub struct VDBStorage {
    /// Configuration
    config: VDBConfig,
    
    /// Active grid collection (in-memory)
    grids: Arc<RwLock<HashMap<String, SparseGrid>>>,
    
    /// Memory-mapped file backend
    mmap_backend: MmapBackend,
    
    /// Compression handler
    compression: CompressionHandler,
    
    /// Statistics
    stats: Arc<RwLock<VDBStats>>,
}

/// Statistics for VDB operations
#[derive(Debug, Default, Clone)]
pub struct VDBStats {
    pub grids_loaded: u64,
    pub grids_saved: u64,
    pub total_voxels: u64,
    pub active_voxels: u64,
    pub compression_ratio: f64,
    pub avg_access_time_us: f64,
}

impl VDBStorage {
    /// Create new VDB storage instance
    pub async fn new(config: VDBConfig) -> io::Result<Self> {
        // Create storage directory
        std::fs::create_dir_all(&config.storage_path)?;
        
        // Initialize components
        let mmap_backend = MmapBackend::new(&config.storage_path)?;
        let compression = CompressionHandler::new(&config.compression)?;
        
        Ok(Self {
            config,
            grids: Arc::new(RwLock::new(HashMap::new())),
            mmap_backend,
            compression,
            stats: Arc::new(RwLock::new(VDBStats::default())),
        })
    }
    
    /// Store sparse adapter weights in VDB grid
    pub async fn store_adapter(
        &self,
        adapter_id: &str,
        weights: &SparseWeights,
    ) -> io::Result<()> {
        let start = std::time::Instant::now();
        
        // Create VDB grid with 99% sparsity optimization
        let grid = self.weights_to_grid(weights).await?;
        
        // Compress grid
        let compressed_data = self.compression.compress(&grid)?;
        
        // Store to memory-mapped backend
        self.mmap_backend.write(adapter_id, &compressed_data).await?;
        
        // Cache in memory
        {
            let mut grids = self.grids.write().await;
            grids.insert(adapter_id.to_string(), grid);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.grids_saved += 1;
            stats.compression_ratio = compressed_data.len() as f64 / weights.dense_size() as f64;
        }
        
        println!("Stored adapter {} in {:.2}ms", adapter_id, start.elapsed().as_millis());
        Ok(())
    }
    
    /// Load sparse adapter weights from VDB
    pub async fn load_adapter(&self, adapter_id: &str) -> io::Result<SparseWeights> {
        let start = std::time::Instant::now();
        
        // Check cache first
        {
            let grids = self.grids.read().await;
            if let Some(grid) = grids.get(adapter_id) {
                let weights = self.grid_to_weights(grid).await?;
                return Ok(weights);
            }
        }
        
        // Load from storage
        let compressed_data = self.mmap_backend.read(adapter_id).await?;
        
        // Decompress grid
        let grid = self.compression.decompress(&compressed_data)?;
        
        // Convert to sparse weights
        let weights = self.grid_to_weights(&grid).await?;
        
        // Cache for future use
        {
            let mut grids = self.grids.write().await;
            grids.insert(adapter_id.to_string(), grid);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.grids_loaded += 1;
            stats.avg_access_time_us = start.elapsed().as_micros() as f64;
        }
        
        println!("Loaded adapter {} in {:.2}ms", adapter_id, start.elapsed().as_millis());
        Ok(weights)
    }
    
    /// Update sparse weights in-place (lock-free for streaming)
    pub async fn update_weights(
        &self,
        adapter_id: &str,
        sparse_updates: &HashMap<Coordinate3D, f32>,
    ) -> io::Result<()> {
        let start = std::time::Instant::now();
        
        // Get grid (create if doesn't exist)
        let mut grids = self.grids.write().await;
        let grid = grids.entry(adapter_id.to_string())
            .or_insert_with(|| SparseGrid::new(self.config.background_value));
        
        // Apply updates directly to VDB grid
        for (coord, value) in sparse_updates {
            if value.abs() > 1e-6 {
                grid.set_value(*coord, *value);
            } else {
                grid.set_inactive(*coord); // Maintain sparsity
            }
        }
        
        println!("Updated {} voxels in {:.2}ms", sparse_updates.len(), start.elapsed().as_millis());
        Ok(())
    }
    
    /// Get storage statistics
    pub async fn get_stats(&self) -> VDBStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    /// Convert sparse weights to VDB grid
    async fn weights_to_grid(&self, weights: &SparseWeights) -> io::Result<SparseGrid> {
        let mut grid = SparseGrid::new(self.config.background_value);
        
        // Iterate over active weights only (99% sparse)
        for (index, value) in weights.active_iter() {
            let coord = self.linear_to_3d(index, &weights.shape);
            grid.set_value(coord, value);
        }
        
        Ok(grid)
    }
    
    /// Convert VDB grid to sparse weights
    async fn grid_to_weights(&self, grid: &SparseGrid) -> io::Result<SparseWeights> {
        let mut weights = SparseWeights::new(grid.inferred_shape());
        
        // Extract active voxels from grid
        for (coord, value) in grid.active_voxels() {
            let linear_index = self.coord_to_linear(coord, &weights.shape);
            weights.set(linear_index, value);
        }
        
        Ok(weights)
    }
    
    /// Convert linear index to 3D coordinate
    fn linear_to_3d(&self, index: usize, shape: &[usize]) -> Coordinate3D {
        // Map 2D weight matrices to 3D VDB space
        match shape.len() {
            2 => {
                let (h, w) = (shape[0], shape[1]);
                let y = index / w;
                let x = index % w;
                Coordinate3D::new(x as i32, y as i32, 0)
            }
            _ => Coordinate3D::new(index as i32, 0, 0) // Fallback for 1D
        }
    }
    
    /// Convert 3D coordinate to linear index
    fn coord_to_linear(&self, coord: Coordinate3D, shape: &[usize]) -> usize {
        match shape.len() {
            2 => {
                let w = shape[1];
                (coord.y() as usize) * w + (coord.x() as usize)
            }
            _ => coord.x() as usize // Fallback
        }
    }
}

/// Memory-mapped file backend for persistence
struct MmapBackend {
    base_path: PathBuf,
}

impl MmapBackend {
    fn new(base_path: &Path) -> io::Result<Self> {
        Ok(Self {
            base_path: base_path.to_path_buf(),
        })
    }
    
    async fn write(&self, key: &str, data: &[u8]) -> io::Result<()> {
        let file_path = self.base_path.join(format!("{}.vdb", key));
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_path)?;
        
        file.write_all(data)?;
        file.flush()?;
        Ok(())
    }
    
    async fn read(&self, key: &str) -> io::Result<Vec<u8>> {
        let file_path = self.base_path.join(format!("{}.vdb", key));
        let file = File::open(file_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(mmap.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_vdb_storage_basic() {
        let config = VDBConfig::default();
        let storage = VDBStorage::new(config).await.unwrap();
        
        // Create test sparse weights (99% sparse)
        let mut weights = SparseWeights::new(vec![1536, 1536]);
        
        // Set only 1% active values
        for i in 0..23593 { // 1% of 1536*1536
            weights.set(i * 100, 0.01); // Sparse pattern
        }
        
        // Store and load
        storage.store_adapter("test_adapter", &weights).await.unwrap();
        let loaded = storage.load_adapter("test_adapter").await.unwrap();
        
        assert_eq!(weights.active_count(), loaded.active_count());
    }
    
    #[tokio::test]
    async fn test_streaming_updates() {
        let config = VDBConfig::default();
        let storage = VDBStorage::new(config).await.unwrap();
        
        // Create initial adapter
        let weights = SparseWeights::new(vec![1536, 1536]);
        storage.store_adapter("streaming_test", &weights).await.unwrap();
        
        // Simulate streaming updates
        let mut updates = HashMap::new();
        updates.insert(Coordinate3D::new(100, 100, 0), 0.5);
        updates.insert(Coordinate3D::new(200, 200, 0), -0.3);
        
        storage.update_weights("streaming_test", &updates).await.unwrap();
        
        // Verify updates applied
        let updated = storage.load_adapter("streaming_test").await.unwrap();
        assert!(updated.active_count() >= 2);
    }
}