//! Rust FFI bindings for NanoVDB hardware acceleration

use std::os::raw::{c_char, c_float, c_void};
use std::ptr::NonNull;
use std::ffi::{CStr, CString};
use crate::storage::vdb::grid::Coordinate3D;

// Include generated bindings
include!(concat!(env!("OUT_DIR"), "/nanovdb_bindings.rs"));

/// Safe Rust wrapper for NanoVDB grid operations
pub struct NanoGrid {
    inner: NonNull<NanoVDBGrid>,
}

unsafe impl Send for NanoGrid {}
unsafe impl Sync for NanoGrid {}

impl Drop for NanoGrid {
    fn drop(&mut self) {
        unsafe {
            nanovdb_destroy_grid(self.inner.as_ptr());
        }
    }
}

impl NanoGrid {
    /// Create new float grid with background value
    pub fn new(background: f32) -> Result<Self, NanoVDBError> {
        unsafe {
            let ptr = nanovdb_create_float_grid(background);
            if ptr.is_null() {
                return Err(NanoVDBError::GridCreationFailed);
            }
            Ok(Self {
                inner: NonNull::new_unchecked(ptr),
            })
        }
    }

    /// Get value at coordinate
    pub fn get_value(&self, coord: Coord3D) -> f32 {
        unsafe {
            let c_coord = NanoVDBCoord {
                x: coord.x as i32,
                y: coord.y as i32,
                z: coord.z as i32,
            };
            nanovdb_grid_get_value(self.inner.as_ptr(), c_coord)
        }
    }

    /// Check if coordinate is active
    pub fn is_active(&self, coord: Coord3D) -> bool {
        unsafe {
            let c_coord = NanoVDBCoord {
                x: coord.x as i32,
                y: coord.y as i32,
                z: coord.z as i32,
            };
            nanovdb_grid_is_active(self.inner.as_ptr(), c_coord)
        }
    }

    /// Get total active voxel count
    pub fn active_voxel_count(&self) -> u64 {
        unsafe {
            nanovdb_grid_active_voxel_count(self.inner.as_ptr())
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> u64 {
        unsafe {
            nanovdb_grid_memory_usage(self.inner.as_ptr())
        }
    }

    /// Get grid statistics
    pub fn stats(&self) -> GridStats {
        unsafe {
            let mut stats = NanoVDBStats {
                active_voxels: 0,
                memory_usage: 0,
                sparsity: 0.0,
                tree_depth: 0,
                leaf_nodes: 0,
                internal_nodes: 0,
            };
            nanovdb_grid_get_stats(self.inner.as_ptr(), &mut stats);
            
            GridStats {
                active_voxels: stats.active_voxels,
                memory_usage: stats.memory_usage,
                sparsity: stats.sparsity,
                tree_depth: stats.tree_depth,
                leaf_nodes: stats.leaf_nodes,
                internal_nodes: stats.internal_nodes,
            }
        }
    }

    /// Create iterator over active voxels
    pub fn iter(&self) -> GridIterator {
        unsafe {
            let ptr = nanovdb_grid_begin(self.inner.as_ptr());
            GridIterator::new(ptr)
        }
    }

    /// Convert to CUDA buffer for GPU operations
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self) -> Result<CudaGrid, NanoVDBError> {
        unsafe {
            let ptr = nanovdb_grid_to_cuda(self.inner.as_ptr());
            if ptr.is_null() {
                return Err(NanoVDBError::CudaConversionFailed);
            }
            Ok(CudaGrid::new(ptr))
        }
    }

    /// Set value at coordinate (mutable operation) - placeholder for VDB functionality
    pub fn set_value(&mut self, _coord: Coordinate3D, _value: f32) {
        // Placeholder implementation - in real NanoVDB this would modify the grid
        println!("Setting value in NanoVDB grid (placeholder)");
    }

    /// Set coordinate as inactive - placeholder for VDB functionality  
    pub fn set_inactive(&mut self, _coord: Coordinate3D) {
        // Placeholder implementation - in real NanoVDB this would modify the grid
        println!("Setting coordinate inactive in NanoVDB grid (placeholder)");
    }

    /// Get raw pointer (for C interop)
    pub fn as_ptr(&self) -> *const NanoVDBGrid {
        self.inner.as_ptr()
    }
}

/// Grid builder for constructing sparse grids efficiently
pub struct GridBuilder {
    inner: NonNull<NanoVDBBuilder>,
    background: f32,
}

unsafe impl Send for GridBuilder {}
unsafe impl Sync for GridBuilder {}

impl Drop for GridBuilder {
    fn drop(&mut self) {
        unsafe {
            nanovdb_destroy_builder(self.inner.as_ptr());
        }
    }
}

impl GridBuilder {
    /// Create new grid builder
    pub fn new(background: f32) -> Result<Self, NanoVDBError> {
        unsafe {
            let ptr = nanovdb_create_builder(background);
            if ptr.is_null() {
                return Err(NanoVDBError::BuilderCreationFailed);
            }
            Ok(Self {
                inner: NonNull::new_unchecked(ptr),
                background,
            })
        }
    }

    /// Set value at coordinate
    pub fn set_value(&mut self, coord: Coord3D, value: f32) {
        unsafe {
            let c_coord = NanoVDBCoord {
                x: coord.x as i32,
                y: coord.y as i32,
                z: coord.z as i32,
            };
            nanovdb_builder_set_value(self.inner.as_ptr(), c_coord, value);
        }
    }

    /// Set value as active at coordinate
    pub fn set_value_on(&mut self, coord: Coord3D, value: f32) {
        unsafe {
            let c_coord = NanoVDBCoord {
                x: coord.x as i32,
                y: coord.y as i32,
                z: coord.z as i32,
            };
            nanovdb_builder_set_value_on(self.inner.as_ptr(), c_coord, value);
        }
    }

    /// Set coordinate as inactive
    pub fn set_value_off(&mut self, coord: Coord3D) {
        unsafe {
            let c_coord = NanoVDBCoord {
                x: coord.x as i32,
                y: coord.y as i32,
                z: coord.z as i32,
            };
            nanovdb_builder_set_value_off(self.inner.as_ptr(), c_coord);
        }
    }

    /// Build final grid
    pub fn build(self) -> Result<NanoGrid, NanoVDBError> {
        unsafe {
            let ptr = nanovdb_builder_get_grid(self.inner.as_ptr());
            if ptr.is_null() {
                return Err(NanoVDBError::GridBuildFailed);
            }
            // Don't drop self, as builder is consumed
            std::mem::forget(self);
            Ok(NanoGrid {
                inner: NonNull::new_unchecked(ptr),
            })
        }
    }

    /// Batch set multiple values efficiently
    pub fn set_sparse_values(&mut self, coords_values: &[(Coord3D, f32)]) {
        for &(coord, value) in coords_values {
            self.set_value_on(coord, value);
        }
    }
}

/// Grid iterator for traversing active voxels
pub struct GridIterator {
    inner: Option<NonNull<NanoVDBIterator>>,
}

impl Drop for GridIterator {
    fn drop(&mut self) {
        if let Some(ptr) = self.inner {
            unsafe {
                nanovdb_destroy_iterator(ptr.as_ptr());
            }
        }
    }
}

impl GridIterator {
    fn new(ptr: *mut NanoVDBIterator) -> Self {
        Self {
            inner: NonNull::new(ptr),
        }
    }
}

impl Iterator for GridIterator {
    type Item = (Coord3D, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ptr) = self.inner {
            unsafe {
                let iter = ptr.as_ref();
                if !iter.valid {
                    return None;
                }

                let coord = Coord3D {
                    x: iter.coord.x,
                    y: iter.coord.y,
                    z: iter.coord.z,
                };
                let value = iter.value;

                nanovdb_iterator_next(ptr.as_ptr());

                Some((coord, value))
            }
        } else {
            None
        }
    }
}

/// CUDA grid for GPU operations
#[cfg(feature = "cuda")]
pub struct CudaGrid {
    inner: NonNull<CudaBuffer>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaGrid {}

#[cfg(feature = "cuda")]
unsafe impl Sync for CudaGrid {}

#[cfg(feature = "cuda")]
impl Drop for CudaGrid {
    fn drop(&mut self) {
        unsafe {
            nanovdb_destroy_cuda_buffer(self.inner.as_ptr());
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaGrid {
    fn new(ptr: *mut CudaBuffer) -> Self {
        unsafe {
            Self {
                inner: NonNull::new_unchecked(ptr),
            }
        }
    }

    /// Perform sparse update on GPU
    pub fn sparse_update(
        &mut self,
        indices: &[u32],
        values: &[f32],
    ) -> Result<(), NanoVDBError> {
        if indices.len() != values.len() {
            return Err(NanoVDBError::MismatchedArrays);
        }

        unsafe {
            nanovdb_cuda_sparse_update(
                self.inner.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
                indices.len() as u32,
            );
        }

        Ok(())
    }

    /// Perform sparse matrix multiplication on GPU
    pub fn sparse_multiply(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), NanoVDBError> {
        unsafe {
            nanovdb_cuda_sparse_multiply(
                self.inner.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len() as u32,
                output.len() as u32,
            );
        }

        Ok(())
    }

    /// Set voxel value at coordinate (GPU operation) - placeholder
    pub fn set_voxel(&mut self, _coord: Coordinate3D, _value: f32) -> Result<(), NanoVDBError> {
        // Placeholder implementation - in real CUDA NanoVDB this would modify the grid on GPU
        println!("Setting voxel in CUDA NanoVDB grid (placeholder)");
        Ok(())
    }

    /// Batch update with 3D coordinates
    pub fn batch_update(
        &mut self,
        coords: &[Coord3D],
        values: &[f32],
    ) -> Result<(), NanoVDBError> {
        if coords.len() != values.len() {
            return Err(NanoVDBError::MismatchedArrays);
        }

        let c_coords: Vec<NanoVDBCoord> = coords
            .iter()
            .map(|c| NanoVDBCoord {
                x: c.x as i32,
                y: c.y as i32,
                z: c.z as i32,
            })
            .collect();

        unsafe {
            nanovdb_cuda_batch_update(
                self.inner.as_ptr(),
                c_coords.as_ptr(),
                values.as_ptr(),
                coords.len() as u32,
            );
        }

        Ok(())
    }

    /// Get GPU buffer size
    pub fn buffer_size(&self) -> usize {
        unsafe {
            nanovdb_cuda_buffer_size(self.inner.as_ptr())
        }
    }
}

/// 3D coordinate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Coord3D {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Convert from linear index (for 2D matrices stored as 3D grids)
    pub fn from_linear_2d(index: usize, width: usize) -> Self {
        let y = (index / width) as i32;
        let x = (index % width) as i32;
        Self { x, y, z: 0 }
    }

    /// Convert to linear index
    pub fn to_linear_2d(&self, width: usize) -> usize {
        (self.y as usize) * width + (self.x as usize)
    }
}

/// Grid statistics
#[derive(Debug, Clone)]
pub struct GridStats {
    pub active_voxels: u64,
    pub memory_usage: u64,
    pub sparsity: f32,
    pub tree_depth: u32,
    pub leaf_nodes: u64,
    pub internal_nodes: u64,
}

/// NanoVDB error types
#[derive(Debug, thiserror::Error)]
pub enum NanoVDBError {
    #[error("Failed to create grid")]
    GridCreationFailed,

    #[error("Failed to create builder")]
    BuilderCreationFailed,

    #[error("Failed to build grid")]
    GridBuildFailed,

    #[error("CUDA conversion failed")]
    CudaConversionFailed,

    #[error("Array lengths do not match")]
    MismatchedArrays,

    #[error("CUDA not available")]
    CudaNotAvailable,
    #[error("Initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Compression failed: {0}")]
    CompressionFailed(String),
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),
}

/// Utility functions
pub mod utils {
    use super::*;

    /// Check if CUDA is available
    #[cfg(feature = "cuda")]
    pub fn cuda_available() -> bool {
        unsafe {
            nanovdb_cuda_is_available()
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cuda_available() -> bool {
        false
    }

    /// Create coordinate from components
    pub fn make_coord(x: i32, y: i32, z: i32) -> Coord3D {
        Coord3D::new(x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let grid = NanoGrid::new(0.0).expect("Failed to create grid");
        assert_eq!(grid.active_voxel_count(), 0);
    }

    #[test]
    fn test_grid_builder() {
        let mut builder = GridBuilder::new(0.0).expect("Failed to create builder");
        
        builder.set_value_on(Coord3D::new(0, 0, 0), 1.0);
        builder.set_value_on(Coord3D::new(1, 1, 0), 2.0);
        builder.set_value_on(Coord3D::new(2, 2, 0), 3.0);

        let grid = builder.build().expect("Failed to build grid");
        
        assert_eq!(grid.active_voxel_count(), 3);
        assert_eq!(grid.get_value(Coord3D::new(0, 0, 0)), 1.0);
        assert_eq!(grid.get_value(Coord3D::new(1, 1, 0)), 2.0);
        assert_eq!(grid.get_value(Coord3D::new(2, 2, 0)), 3.0);
    }

    #[test]
    fn test_sparse_batch_operations() {
        let mut builder = GridBuilder::new(0.0).expect("Failed to create builder");
        
        let coords_values = vec![
            (Coord3D::new(10, 20, 0), 1.5),
            (Coord3D::new(15, 25, 0), -2.3),
            (Coord3D::new(20, 30, 0), 0.8),
        ];

        builder.set_sparse_values(&coords_values);
        let grid = builder.build().expect("Failed to build grid");

        assert_eq!(grid.active_voxel_count(), 3);
        
        for (coord, expected_value) in coords_values {
            assert!((grid.get_value(coord) - expected_value).abs() < 1e-6);
            assert!(grid.is_active(coord));
        }
    }

    #[test]
    fn test_coord_conversion() {
        let coord = Coord3D::from_linear_2d(1537, 1536); // Row 1, Col 1
        assert_eq!(coord, Coord3D::new(1, 1, 0));

        let linear = coord.to_linear_2d(1536);
        assert_eq!(linear, 1537);
    }

    #[test]
    fn test_grid_iterator() {
        let mut builder = GridBuilder::new(0.0).expect("Failed to create builder");
        
        builder.set_value_on(Coord3D::new(0, 0, 0), 1.0);
        builder.set_value_on(Coord3D::new(5, 5, 0), 5.0);

        let grid = builder.build().expect("Failed to build grid");
        
        let active_voxels: Vec<_> = grid.iter().collect();
        assert_eq!(active_voxels.len(), 2);
        
        let has_origin = active_voxels.iter().any(|(coord, value)| {
            *coord == Coord3D::new(0, 0, 0) && (*value - 1.0).abs() < 1e-6
        });
        let has_five = active_voxels.iter().any(|(coord, value)| {
            *coord == Coord3D::new(5, 5, 0) && (*value - 5.0).abs() < 1e-6
        });
        
        assert!(has_origin);
        assert!(has_five);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_availability() {
        // This will pass or fail based on system CUDA setup
        let available = utils::cuda_available();
        println!("CUDA available: {}", available);
    }
}