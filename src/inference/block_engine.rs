//! Block-based memory management engine
//! 
//! Manages allocation and tracking of KV cache blocks for sequences

use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;


/// Block allocation status
#[derive(Debug, Clone, PartialEq)]
pub enum AllocStatus {
    /// Can allocate immediately
    Ok,
    /// Can allocate after freeing some blocks
    Later,
    /// Cannot allocate (exceeds capacity)
    Impossible,
}

/// Physical block location
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BlockLocation {
    /// Block is on GPU
    GPU(usize),
    /// Block is on CPU
    CPU(usize),
}

/// Block table mapping logical to physical blocks
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical to physical block mapping
    blocks: Vec<Option<BlockLocation>>,
}

impl BlockTable {
    pub fn new(size: usize) -> Self {
        Self {
            blocks: vec![None; size],
        }
    }
    
    pub fn allocate(&mut self, logical_idx: usize, physical_block: BlockLocation) {
        if logical_idx < self.blocks.len() {
            self.blocks[logical_idx] = Some(physical_block);
        }
    }
    
    pub fn free(&mut self, logical_idx: usize) -> Option<BlockLocation> {
        if logical_idx < self.blocks.len() {
            self.blocks[logical_idx].take()
        } else {
            None
        }
    }
    
    pub fn get(&self, logical_idx: usize) -> Option<&BlockLocation> {
        self.blocks.get(logical_idx).and_then(|b| b.as_ref())
    }
}

/// Allocator for a specific device
pub struct DeviceAllocator {
    /// Device type (GPU/CPU)
    device_type: String,
    /// Total number of blocks
    total_blocks: usize,
    /// Free block indices
    free_blocks: VecDeque<usize>,
    /// Used block indices
    used_blocks: HashSet<usize>,
}

impl DeviceAllocator {
    pub fn new(device_type: String, num_blocks: usize) -> Self {
        Self {
            device_type,
            total_blocks: num_blocks,
            free_blocks: (0..num_blocks).collect(),
            used_blocks: HashSet::new(),
        }
    }
    
    pub fn allocate(&mut self) -> Option<usize> {
        if let Some(block_id) = self.free_blocks.pop_front() {
            self.used_blocks.insert(block_id);
            Some(block_id)
        } else {
            None
        }
    }
    
    pub fn free(&mut self, block_id: usize) {
        if self.used_blocks.remove(&block_id) {
            self.free_blocks.push_back(block_id);
        }
    }
    
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }
    
    pub fn num_used_blocks(&self) -> usize {
        self.used_blocks.len()
    }
}

/// Block engine for memory management
pub struct BlockEngine {
    /// Block size in tokens
    block_size: usize,
    /// GPU block allocator
    gpu_allocator: Arc<RwLock<DeviceAllocator>>,
    /// CPU block allocator
    cpu_allocator: Arc<RwLock<DeviceAllocator>>,
    /// Sequence to block table mapping
    seq_block_tables: Arc<RwLock<HashMap<usize, BlockTable>>>,
    /// Block reference counts for copy-on-write
    block_ref_counts: Arc<RwLock<HashMap<BlockLocation, usize>>>,
}


impl BlockEngine {
    /// Create new block engine
    pub async fn new(
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> Result<Self> {
        Ok(Self {
            block_size,
            gpu_allocator: Arc::new(RwLock::new(
                DeviceAllocator::new("GPU".to_string(), num_gpu_blocks)
            )),
            cpu_allocator: Arc::new(RwLock::new(
                DeviceAllocator::new("CPU".to_string(), num_cpu_blocks)
            )),
            seq_block_tables: Arc::new(RwLock::new(HashMap::new())),
            block_ref_counts: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Check if we can allocate blocks for a sequence
    pub async fn can_allocate(&self, seq_id: usize, num_blocks: usize) -> AllocStatus {
        let gpu_free = {
            let gpu_alloc = self.gpu_allocator.read().await;
            gpu_alloc.num_free_blocks()
        };
        
        if gpu_free >= num_blocks {
            return AllocStatus::Ok;
        }
        
        let cpu_free = {
            let cpu_alloc = self.cpu_allocator.read().await;
            cpu_alloc.num_free_blocks()
        };
        
        let total_free = gpu_free + cpu_free;
        
        if total_free >= num_blocks {
            return AllocStatus::Later; // Need to swap some blocks
        }
        
        
        AllocStatus::Impossible
    }
    
    /// Allocate blocks for a sequence
    pub async fn allocate(&self, seq_id: usize, num_blocks: usize) -> Result<()> {
        let mut block_table = BlockTable::new(num_blocks);
        let mut allocated_blocks = Vec::new();
        
        // Try to allocate on GPU first
        {
            let mut gpu_alloc = self.gpu_allocator.write().await;
            for i in 0..num_blocks {
                if let Some(block_id) = gpu_alloc.allocate() {
                    let location = BlockLocation::GPU(block_id);
                    block_table.allocate(i, location.clone());
                    allocated_blocks.push(location);
                } else {
                    break;
                }
            }
        }
        
        // Allocate remaining on CPU
        let remaining = num_blocks - allocated_blocks.len();
        if remaining > 0 {
            let mut cpu_alloc = self.cpu_allocator.write().await;
            for i in allocated_blocks.len()..num_blocks {
                if let Some(block_id) = cpu_alloc.allocate() {
                    let location = BlockLocation::CPU(block_id);
                    block_table.allocate(i, location.clone());
                    allocated_blocks.push(location);
                } else {
                    break;
                }
            }
        }
        
        // If we can't allocate enough blocks, fail
        if allocated_blocks.len() < num_blocks {
            return Err(anyhow!("Cannot allocate {} blocks, only {} available", num_blocks, allocated_blocks.len()));
        }
        
        // Update reference counts
        {
            let mut ref_counts = self.block_ref_counts.write().await;
            for block in &allocated_blocks {
                *ref_counts.entry(block.clone()).or_insert(0) += 1;
            }
        }
        
        // Store block table for sequence
        {
            let mut tables = self.seq_block_tables.write().await;
            tables.insert(seq_id, block_table);
        }
        
        Ok(())
    }
    
    /// Free blocks for a sequence
    pub async fn free(&self, seq_id: usize) -> Result<()> {
        let block_table = {
            let mut tables = self.seq_block_tables.write().await;
            tables.remove(&seq_id)
        };
        
        if let Some(table) = block_table {
            // Free each block
            for i in 0..table.blocks.len() {
                if let Some(location) = table.blocks[i].as_ref() {
                    self.free_block(location.clone()).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Free a single block
    async fn free_block(&self, location: BlockLocation) -> Result<()> {
        // Decrement reference count
        let should_free = {
            let mut ref_counts = self.block_ref_counts.write().await;
            if let Some(count) = ref_counts.get_mut(&location) {
                *count -= 1;
                if *count == 0 {
                    ref_counts.remove(&location);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        
        // Actually free the block if ref count is 0
        if should_free {
            match location {
                BlockLocation::GPU(block_id) => {
                    let mut gpu_alloc = self.gpu_allocator.write().await;
                    gpu_alloc.free(block_id);
                }
                BlockLocation::CPU(block_id) => {
                    let mut cpu_alloc = self.cpu_allocator.write().await;
                    cpu_alloc.free(block_id);
                }
            }
        }
        
        Ok(())
    }
    
    /// Swap blocks from CPU to GPU
    pub async fn swap_in(&self, seq_id: usize, num_blocks: usize) -> Result<HashMap<usize, usize>> {
        let mut swaps = HashMap::new();
        
        let tables = self.seq_block_tables.read().await;
        if let Some(table) = tables.get(&seq_id) {
            let mut gpu_alloc = self.gpu_allocator.write().await;
            
            for i in 0..num_blocks.min(table.blocks.len()) {
                if let Some(BlockLocation::CPU(cpu_block)) = table.blocks[i].as_ref() {
                    if let Some(gpu_block) = gpu_alloc.allocate() {
                        swaps.insert(*cpu_block, gpu_block);
                    }
                }
            }
        }
        
        Ok(swaps)
    }
    
    /// Swap blocks from GPU to CPU
    pub async fn swap_out(&self, seq_id: usize, num_blocks: usize) -> Result<HashMap<usize, usize>> {
        let mut swaps = HashMap::new();
        
        let tables = self.seq_block_tables.read().await;
        if let Some(table) = tables.get(&seq_id) {
            let mut cpu_alloc = self.cpu_allocator.write().await;
            
            for i in 0..num_blocks.min(table.blocks.len()) {
                if let Some(BlockLocation::GPU(gpu_block)) = table.blocks[i].as_ref() {
                    if let Some(cpu_block) = cpu_alloc.allocate() {
                        swaps.insert(*gpu_block, cpu_block);
                    }
                }
            }
        }
        
        Ok(swaps)
    }
    
    /// Get block table for a sequence
    pub async fn get_block_table(&self, seq_id: usize) -> Option<Vec<Option<BlockLocation>>> {
        let tables = self.seq_block_tables.read().await;
        tables.get(&seq_id).map(|table| table.blocks.clone())
    }
    
    /// Fork block table for copy-on-write
    pub async fn fork(&self, parent_seq: usize, child_seq: usize) -> Result<()> {
        let parent_table = {
            let tables = self.seq_block_tables.read().await;
            tables.get(&parent_seq).cloned()
        };
        
        if let Some(table) = parent_table {
            // Increment reference counts for shared blocks
            {
                let mut ref_counts = self.block_ref_counts.write().await;
                for block in &table.blocks {
                    if let Some(location) = block {
                        *ref_counts.entry(location.clone()).or_insert(0) += 1;
                    }
                }
            }
            
            // Store cloned table for child
            {
                let mut tables = self.seq_block_tables.write().await;
                tables.insert(child_seq, table);
            }
        }
        
        Ok(())
    }
    
    /// Get memory usage statistics
    pub async fn get_stats(&self) -> BlockEngineStats {
        let gpu_stats = {
            let gpu_alloc = self.gpu_allocator.read().await;
            (gpu_alloc.num_used_blocks(), gpu_alloc.num_free_blocks())
        };
        
        let cpu_stats = {
            let cpu_alloc = self.cpu_allocator.read().await;
            (cpu_alloc.num_used_blocks(), cpu_alloc.num_free_blocks())
        };
        
        let num_sequences = {
            let tables = self.seq_block_tables.read().await;
            tables.len()
        };
        
        BlockEngineStats {
            gpu_blocks_used: gpu_stats.0,
            gpu_blocks_free: gpu_stats.1,
            cpu_blocks_used: cpu_stats.0,
            cpu_blocks_free: cpu_stats.1,
            num_sequences,
            block_size: self.block_size,
        }
    }
}

/// Block engine statistics
#[derive(Debug, Clone)]
pub struct BlockEngineStats {
    pub gpu_blocks_used: usize,
    pub gpu_blocks_free: usize,
    pub cpu_blocks_used: usize,
    pub cpu_blocks_free: usize,
    pub num_sequences: usize,
    pub block_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_block_allocation() {
        let engine = BlockEngine::new(16, 10, 10).await.unwrap();
        
        // Test allocation
        let status = engine.can_allocate(0, 5).await;
        assert_eq!(status, AllocStatus::Ok);
        
        engine.allocate(0, 5).await.unwrap();
        
        // Check stats
        let stats = engine.get_stats().await;
        assert_eq!(stats.gpu_blocks_used, 5);
        assert_eq!(stats.num_sequences, 1);
        
        // Free blocks
        engine.free(0).await.unwrap();
        
        let stats = engine.get_stats().await;
        assert_eq!(stats.gpu_blocks_used, 0);
        assert_eq!(stats.num_sequences, 0);
    }
    
    #[tokio::test]
    async fn test_block_overflow_to_cpu() {
        let engine = BlockEngine::new(16, 3, 5).await.unwrap();
        
        // Allocate more than GPU capacity
        engine.allocate(0, 5).await.unwrap();
        
        let stats = engine.get_stats().await;
        assert_eq!(stats.gpu_blocks_used, 3);
        assert_eq!(stats.cpu_blocks_used, 2);
    }
}