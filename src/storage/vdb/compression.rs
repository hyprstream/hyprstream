//! Compression algorithms optimized for sparse VDB grids

use std::io::{self, Result};
use serde::{Serialize, Deserialize};
use crate::storage::vdb::grid::SparseGrid;

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub algorithm: CompressionAlgorithm,
    pub level: u32,
    pub enable_delta_compression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd,
    Custom, // For sparse-optimized compression
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Lz4,
            level: 4,
            enable_delta_compression: true,
        }
    }
}

/// Compression handler for VDB grids
pub struct CompressionHandler {
    config: CompressionConfig,
}

impl CompressionHandler {
    pub fn new(algorithm: &str) -> Result<Self> {
        let config = match algorithm {
            "none" => CompressionConfig {
                algorithm: CompressionAlgorithm::None,
                level: 0,
                enable_delta_compression: false,
            },
            "lz4" => CompressionConfig {
                algorithm: CompressionAlgorithm::Lz4,
                level: 4,
                enable_delta_compression: true,
            },
            "zstd" => CompressionConfig {
                algorithm: CompressionAlgorithm::Zstd,
                level: 3,
                enable_delta_compression: true,
            },
            "custom" => CompressionConfig {
                algorithm: CompressionAlgorithm::Custom,
                level: 5,
                enable_delta_compression: true,
            },
            _ => return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unsupported compression algorithm: {}", algorithm)
            )),
        };
        
        Ok(Self { config })
    }
    
    /// Compress sparse grid to bytes
    pub fn compress(&self, grid: &SparseGrid) -> Result<Vec<u8>> {
        // Serialize grid to binary
        let serialized = bincode::serialize(grid)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        
        match self.config.algorithm {
            CompressionAlgorithm::None => Ok(serialized),
            CompressionAlgorithm::Lz4 => self.compress_lz4(&serialized),
            CompressionAlgorithm::Zstd => self.compress_zstd(&serialized),
            CompressionAlgorithm::Custom => self.compress_sparse_optimized(&serialized),
        }
    }
    
    /// Decompress bytes to sparse grid
    pub fn decompress(&self, data: &[u8]) -> Result<SparseGrid> {
        let decompressed = match self.config.algorithm {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::Lz4 => self.decompress_lz4(data)?,
            CompressionAlgorithm::Zstd => self.decompress_zstd(data)?,
            CompressionAlgorithm::Custom => self.decompress_sparse_optimized(data)?,
        };
        
        let grid = bincode::deserialize(&decompressed)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            
        Ok(grid)
    }
    
    /// LZ4 compression (fast, good for streaming)
    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified LZ4 compression
        // In production, use lz4_flex or similar crate
        
        if data.len() < 100 {
            // Don't compress small data
            return Ok(data.to_vec());
        }
        
        // Simple run-length encoding for demonstration
        let mut compressed = Vec::new();
        compressed.push(0x4C); // LZ4 magic
        compressed.push(0x5A);
        compressed.push(0x34);
        
        // Original size
        compressed.extend_from_slice(&(data.len() as u32).to_le_bytes());
        
        // Simple compression (replace with actual LZ4)
        compressed.extend_from_slice(data);
        
        Ok(compressed)
    }
    
    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 7 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid LZ4 data"));
        }
        
        // Check magic
        if &data[0..3] != &[0x4C, 0x5A, 0x34] {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid LZ4 magic"));
        }
        
        // Get original size
        let _original_size = u32::from_le_bytes([data[3], data[4], data[5], data[6]]);
        
        // Return decompressed data (simplified)
        Ok(data[7..].to_vec())
    }
    
    /// Zstd compression (better compression ratio)
    fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Placeholder for Zstd compression
        // In production, use zstd crate
        
        let mut compressed = Vec::new();
        compressed.push(0x28); // Zstd magic
        compressed.push(0xB5);
        compressed.push(0x2F);
        compressed.push(0xFD);
        
        // Add level and original size
        compressed.push(self.config.level as u8);
        compressed.extend_from_slice(&(data.len() as u32).to_le_bytes());
        
        // Simple compression placeholder
        compressed.extend_from_slice(data);
        
        Ok(compressed)
    }
    
    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 9 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid Zstd data"));
        }
        
        // Check magic
        if &data[0..4] != &[0x28, 0xB5, 0x2F, 0xFD] {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid Zstd magic"));
        }
        
        // Skip level and size headers
        Ok(data[9..].to_vec())
    }
    
    /// Custom sparse-optimized compression
    fn compress_sparse_optimized(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Custom compression optimized for 99% sparse data
        let mut compressed = Vec::new();
        
        // Magic header for sparse compression
        compressed.extend_from_slice(b"SPRS");
        compressed.extend_from_slice(&(data.len() as u32).to_le_bytes());
        
        // Delta compression for coordinates (common in sparse data)
        if self.config.enable_delta_compression {
            compressed.extend_from_slice(&self.delta_compress(data)?);
        } else {
            compressed.extend_from_slice(data);
        }
        
        Ok(compressed)
    }
    
    fn decompress_sparse_optimized(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid sparse data"));
        }
        
        // Check magic
        if &data[0..4] != b"SPRS" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid sparse magic"));
        }
        
        let _original_size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        
        if self.config.enable_delta_compression {
            self.delta_decompress(&data[8..])
        } else {
            Ok(data[8..].to_vec())
        }
    }
    
    /// Delta compression for sparse coordinates
    fn delta_compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simple delta encoding
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut compressed = Vec::new();
        compressed.push(data[0]); // First byte as-is
        
        for i in 1..data.len() {
            let delta = (data[i] as i16) - (data[i-1] as i16);
            if delta >= -127 && delta <= 127 {
                compressed.push(delta as u8);
            } else {
                // Fallback for large deltas
                compressed.push(0xFF); // Escape
                compressed.push(data[i]);
            }
        }
        
        Ok(compressed)
    }
    
    fn delta_decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut decompressed = Vec::new();
        decompressed.push(data[0]); // First byte as-is
        
        let mut i = 1;
        while i < data.len() {
            if data[i] == 0xFF && i + 1 < data.len() {
                // Escape sequence
                decompressed.push(data[i + 1]);
                i += 2;
            } else {
                // Delta value
                let prev = *decompressed.last().unwrap();
                let delta = data[i] as i8;
                let new_val = (prev as i16 + delta as i16) as u8;
                decompressed.push(new_val);
                i += 1;
            }
        }
        
        Ok(decompressed)
    }
}

/// Compression statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub algorithm: String,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_time_ms: f64,
    pub decompression_time_ms: f64,
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self {
            algorithm: "none".to_string(),
            original_size: 0,
            compressed_size: 0,
            compression_ratio: 1.0,
            compression_time_ms: 0.0,
            decompression_time_ms: 0.0,
        }
    }
}

impl CompressionStats {
    pub fn new(
        algorithm: String,
        original_size: usize,
        compressed_size: usize,
        compression_time: std::time::Duration,
        decompression_time: std::time::Duration,
    ) -> Self {
        Self {
            algorithm,
            original_size,
            compressed_size,
            compression_ratio: original_size as f64 / compressed_size as f64,
            compression_time_ms: compression_time.as_millis() as f64,
            decompression_time_ms: decompression_time.as_millis() as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::vdb::grid::{SparseGrid, Coordinate3D};
    
    #[test]
    fn test_compression_basic() {
        let handler = CompressionHandler::new("lz4").unwrap();
        
        // Create sparse grid with test data
        let mut grid = SparseGrid::new(0.0);
        
        // Add sparse pattern (99% sparse)
        for i in 0..1000 {
            if i % 100 == 0 {
                grid.set_value(Coordinate3D::new(i, i, 0), i as f32 * 0.001);
            }
        }
        
        // Compress
        let start = std::time::Instant::now();
        let compressed = handler.compress(&grid).unwrap();
        let compression_time = start.elapsed();
        
        // Decompress
        let start = std::time::Instant::now();
        let decompressed = handler.decompress(&compressed).unwrap();
        let decompression_time = start.elapsed();
        
        // Verify
        assert_eq!(grid.active_count(), decompressed.active_count());
        
        println!("Compression: {:.2}ms, Decompression: {:.2}ms", 
                compression_time.as_millis(), decompression_time.as_millis());
    }
    
    #[test]
    fn test_delta_compression() {
        let handler = CompressionHandler::new("custom").unwrap();
        
        // Test delta compression with sequential data
        let original = vec![10, 11, 12, 13, 14, 100, 101, 102]; // Mix of deltas
        let compressed = handler.delta_compress(&original).unwrap();
        let decompressed = handler.delta_decompress(&compressed).unwrap();
        
        assert_eq!(original, decompressed);
        assert!(compressed.len() <= original.len()); // Should compress sequential data
    }
    
    #[test]
    fn test_compression_ratios() {
        let algorithms = ["none", "lz4", "zstd", "custom"];
        let mut grid = SparseGrid::new(0.0);
        
        // Create highly repetitive sparse data (should compress well)
        for i in 0..10000 {
            if i % 1000 == 0 {
                grid.set_value(Coordinate3D::new(i, 0, 0), 0.123); // Repeated value
            }
        }
        
        for alg in &algorithms {
            let handler = CompressionHandler::new(alg).unwrap();
            let compressed = handler.compress(&grid).unwrap();
            let ratio = bincode::serialize(&grid).unwrap().len() as f64 / compressed.len() as f64;
            
            println!("{}: compression ratio {:.2}x", alg, ratio);
            
            // Verify roundtrip
            let decompressed = handler.decompress(&compressed).unwrap();
            assert_eq!(grid.active_count(), decompressed.active_count());
        }
    }
}