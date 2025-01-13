//! GPU memory management and buffer pooling.
//! 
//! This module provides efficient memory management for GPU operations, including:
//! - Smart buffer allocation and reuse
//! - Memory pool management
//! - Automatic garbage collection
//! - Memory defragmentation
//! - Cross-device memory handling

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, Mutex};
use burn::{
    tensor::{
        backend::Backend,
        Tensor,
        Float,
    },
};
use std::time::{Duration, Instant};
use crate::error::Result;

/// Memory buffer status
#[derive(Debug, Clone, Copy, PartialEq)]
enum BufferStatus {
    /// Buffer is free for use
    Free,
    /// Buffer is in use
    InUse,
    /// Buffer is marked for deletion
    PendingDelete,
}

/// Memory buffer metadata
#[derive(Debug)]
struct BufferMetadata {
    /// Buffer size in bytes
    size: usize,
    /// Current status
    status: BufferStatus,
    /// Last access time
    last_access: Instant,
    /// Number of times buffer was reused
    reuse_count: usize,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Initial pool size in bytes
    pub initial_size: usize,
    /// Maximum pool size in bytes
    pub max_size: usize,
    /// Buffer reuse threshold in seconds
    pub reuse_threshold: Duration,
    /// Enable defragmentation
    pub enable_defrag: bool,
    /// Garbage collection interval
    pub gc_interval: Duration,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024 * 1024, // 1GB
            max_size: 8 * 1024 * 1024 * 1024, // 8GB
            reuse_threshold: Duration::from_secs(60),
            enable_defrag: true,
            gc_interval: Duration::from_secs(300),
        }
    }
}

/// GPU memory pool manager
pub struct MemoryPool<B>
where
    B: Backend + 'static,
{
    /// Pool configuration
    config: MemoryConfig,
    /// Available buffers by size
    buffers: Arc<RwLock<HashMap<usize, Vec<(Tensor<B>, BufferMetadata)>>>>,
    /// Total allocated size
    allocated_size: Arc<Mutex<usize>>,
    /// Backend device
    device: B::Device,
}

impl<B: Backend> MemoryPool<B> {
    /// Create a new memory pool
    pub fn new(config: MemoryConfig, device: B::Device) -> Self {
        let pool = Self {
            config,
            buffers: Arc::new(RwLock::new(HashMap::new())),
            allocated_size: Arc::new(Mutex::new(0)),
            device,
        };
        
        // Start garbage collection task
        pool.start_gc();
        
        pool
    }

    /// Allocate a buffer of the specified size
    pub async fn allocate(&self, size: usize) -> Result<Tensor<B>> {
        // Try to reuse existing buffer
        if let Some(tensor) = self.get_free_buffer(size).await? {
            return Ok(tensor);
        }
        
        // Allocate new buffer if under limit
        let total_allocated = *self.allocated_size.lock().await;
        if total_allocated + size <= self.config.max_size {
            let tensor = Tensor::<B>::zeros(&[size], &self.device);
            *self.allocated_size.lock().await += size;
            
            // Track buffer
            let mut buffers = self.buffers.write().await;
            buffers.entry(size).or_default().push((
                tensor.clone(),
                BufferMetadata {
                    size,
                    status: BufferStatus::InUse,
                    last_access: Instant::now(),
                    reuse_count: 0,
                }
            ));
            
            Ok(tensor)
        } else {
            // Try to free memory
            self.collect_garbage().await?;
            
            // Retry allocation
            self.allocate(size).await
        }
    }

    /// Release a buffer back to the pool
    pub async fn release(&self, tensor: Tensor<B>, size: usize) -> Result<()> {
        let mut buffers = self.buffers.write().await;
        
        if let Some(pool_buffers) = buffers.get_mut(&size) {
            for (buf, metadata) in pool_buffers {
                // Compare tensor values element-wise
                if buf.equal(&tensor).into_scalar() {
                    metadata.status = BufferStatus::Free;
                    metadata.last_access = Instant::now();
                    metadata.reuse_count += 1;
                    return Ok(());
                }
            }
        }
        
        // Buffer not found in pool
        Err(crate::error::Error::Validation("Buffer not found in pool".into()))
    }

    /// Get a free buffer of the specified size
    async fn get_free_buffer(&self, size: usize) -> Result<Option<Tensor<B>>> {
        let mut buffers = self.buffers.write().await;
        
        if let Some(pool_buffers) = buffers.get_mut(&size) {
            for (buffer, metadata) in pool_buffers {
                if metadata.status == BufferStatus::Free {
                    metadata.status = BufferStatus::InUse;
                    metadata.last_access = Instant::now();
                    return Ok(Some(buffer.clone()));
                }
            }
        }
        
        Ok(None)
    }

    /// Start garbage collection task
    fn start_gc(&self) {
        let pool = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(pool.config.gc_interval).await;
                if let Err(e) = pool.collect_garbage().await {
                    tracing::error!("Garbage collection error: {}", e);
                }
            }
        });
    }

    /// Collect garbage and free unused buffers
    pub async fn collect_garbage(&self) -> Result<()> {
        let mut buffers = self.buffers.write().await;
        let now = Instant::now();
        let mut freed_size = 0;
        
        // Mark old buffers for deletion
        for pool_buffers in buffers.values_mut() {
            pool_buffers.retain_mut(|(_, metadata)| {
                if metadata.status == BufferStatus::Free 
                    && now.duration_since(metadata.last_access) > self.config.reuse_threshold {
                    metadata.status = BufferStatus::PendingDelete;
                    freed_size += metadata.size;
                    false
                } else {
                    true
                }
            });
        }
        
        // Update allocated size
        if freed_size > 0 {
            *self.allocated_size.lock().await -= freed_size;
        }
        
        Ok(())
    }

    /// Defragment memory pool
    pub async fn defragment(&self) -> Result<()> {
        if !self.config.enable_defrag {
            return Ok(());
        }
        
        let mut buffers = self.buffers.write().await;
        
        // Collect fragmentation info
        let mut fragments = Vec::new();
        for (size, pool_buffers) in buffers.iter() {
            let free_count = pool_buffers.iter()
                .filter(|(_, m)| m.status == BufferStatus::Free)
                .count();
            if free_count > 1 {
                fragments.push((*size, free_count));
            }
        }
        
        // Merge fragments
        for (size, count) in fragments {
            if let Some(pool_buffers) = buffers.get_mut(&size) {
                // Sort by last access time
                pool_buffers.sort_by_key(|(_, m)| m.last_access);
                
                // Merge adjacent free buffers
                let mut i = 0;
                while i < pool_buffers.len() - 1 {
                    if pool_buffers[i].1.status == BufferStatus::Free 
                        && pool_buffers[i + 1].1.status == BufferStatus::Free {
                        // Merge buffers
                        let merged_size = pool_buffers[i].1.size + pool_buffers[i + 1].1.size;
                        let new_tensor = Tensor::<B>::zeros(&[merged_size], &self.device);
                        
                        // Remove old buffers
                        pool_buffers.remove(i + 1);
                        pool_buffers.remove(i);
                        
                        // Add merged buffer
                        pool_buffers.push((
                            new_tensor,
                            BufferMetadata {
                                size: merged_size,
                                status: BufferStatus::Free,
                                last_access: Instant::now(),
                                reuse_count: 0,
                            }
                        ));
                    } else {
                        i += 1;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Get current memory statistics
    pub async fn get_stats(&self) -> MemoryStats {
        let buffers = self.buffers.read().await;
        let allocated = *self.allocated_size.lock().await;
        
        let mut stats = MemoryStats {
            total_allocated: allocated,
            total_used: 0,
            total_free: 0,
            buffer_count: 0,
            fragmentation: 0.0,
        };
        
        for pool_buffers in buffers.values() {
            for (_, metadata) in pool_buffers {
                stats.buffer_count += 1;
                match metadata.status {
                    BufferStatus::InUse => stats.total_used += metadata.size,
                    BufferStatus::Free => stats.total_free += metadata.size,
                    _ => {}
                }
            }
        }
        
        // Calculate fragmentation ratio
        if stats.total_free > 0 {
            stats.fragmentation = buffers.len() as f32 / stats.buffer_count as f32;
        }
        
        stats
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: usize,
    /// Total memory in use
    pub total_used: usize,
    /// Total free memory
    pub total_free: usize,
    /// Number of buffers
    pub buffer_count: usize,
    /// Fragmentation ratio (0-1)
    pub fragmentation: f32,
}

impl<B: Backend> Clone for MemoryPool<B> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            buffers: Arc::clone(&self.buffers),
            allocated_size: Arc::clone(&self.allocated_size),
            device: self.device.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    #[tokio::test]
    async fn test_memory_pool() {
        let device = NdArray::Device::default();
        let config = MemoryConfig::default();
        let pool = MemoryPool::<NdArray>::new(config, device);
        
        // Test allocation
        let buffer = pool.allocate(1024).await.unwrap();
        let stats = pool.get_stats().await;
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.total_used, 1024);
        
        // Test release
        pool.release(buffer, 1024).await.unwrap();
        let stats = pool.get_stats().await;
        assert_eq!(stats.total_free, 1024);
        
        // Test garbage collection
        pool.collect_garbage().await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
        let stats = pool.get_stats().await;
        assert!(stats.total_free <= 1024);
        
        // Test defragmentation
        pool.defragment().await.unwrap();
        let stats = pool.get_stats().await;
        assert!(stats.fragmentation >= 0.0 && stats.fragmentation <= 1.0);
    }
}
