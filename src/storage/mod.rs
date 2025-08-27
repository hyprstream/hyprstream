//! VDB-first storage for dynamic sparse weight adjustments
//!
//! This module provides VDB-based storage optimized for:
//! - 99% sparse neural network weights  
//! - Real-time dynamic weight adjustments
//! - Memory-mapped disk persistence
//! - Hardware-accelerated operations
//! - Neural compression (10-100x compression ratios)
//!
//! The architecture is designed for adaptive ML inference systems
//! that require streaming weight updates during inference.

// VDB-first storage for sparse adaptive layers
pub mod vdb;
pub mod paths;
pub mod lora_storage_manager;
pub mod lora_weight_cache;

// Re-export main VDB interfaces
pub use vdb::{
    SparseStorage, VDBSparseStorage, SparseStorageConfig, 
    SparseWeightUpdate, EmbeddingMatch, SparseStorageError,
    AdapterStats, StorageStats, CompactionStats,
    VDBStorage, VDBConfig, AdapterStore,
};

// Re-export LoRA storage manager and cache
pub use lora_storage_manager::{
    LoRAStorageManager, LoRAStorageConfig, 
    LoRAAdapterInfo, LoRAStorageInfo, LoRAFullStats,
};

pub use lora_weight_cache::{
    LoRAWeightCache, LoRAWeightCacheConfig, CachedLoRAAdapter,
    AdapterCacheStats, CacheStats,
};

// View support re-enabled with simplified schema (no arrow dependency)

// XDG-compliant path management (internal use only)
// Note: StoragePaths should only be used by config module
// HfAuth has been moved to crate::auth module
pub use paths::StoragePaths;
