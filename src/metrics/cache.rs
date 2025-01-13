//! Cache management and metrics functionality.

use std::time::Duration;
use crate::error::Result;

/// Trait for cache management and metrics
#[async_trait::async_trait]
pub trait CacheManager: Send + Sync {
    /// Initialize the cache
    async fn init(&mut self) -> Result<()>;
    
    /// Get current cache size in bytes
    async fn get_size(&self) -> Result<u64>;
    
    /// Get cache hit rate
    async fn get_hit_rate(&self) -> Result<f64>;
    
    /// Get cache eviction rate
    async fn get_eviction_rate(&self) -> Result<f64>;
    
    /// Clear the cache
    async fn clear(&mut self) -> Result<()>;
    
    /// Set cache TTL
    async fn set_ttl(&mut self, ttl: Duration) -> Result<()>;
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size_bytes: u64,
    pub items_count: u64,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
}
