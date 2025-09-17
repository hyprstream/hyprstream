//! LoRA Weight Cache - Memory Management for Active Adapters
//!
//! This module provides intelligent caching of LoRA adapter weights with
//! LRU eviction, automatic VDB loading, and performance optimization.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use anyhow::Result;

use crate::api::lora_registry::LoRAId;
// TODO: Replace with generic LoRAAdapter after sparse removal
// use crate::lora::sparse::SparseLoRAAdapter;
use std::marker::PhantomData;
use crate::storage::lora_storage_manager::LoRAStorageManager;

/// Cached LoRA adapter with usage tracking
#[derive(Clone)]
pub struct CachedLoRAAdapter {
    /// The actual adapter
    // TODO: Make this generic over LoRAAdapter trait
    pub adapter: Arc<PhantomData<()>>, // Placeholder after sparse removal
    
    /// Last access time for LRU eviction
    pub last_accessed: Instant,
    
    /// Reference count for active inference
    pub ref_count: usize,
    
    /// Memory usage in bytes
    pub memory_usage: usize,
    
    /// Whether weights have been modified since last save
    pub is_dirty: bool,
    
    /// Performance metrics
    pub cache_stats: AdapterCacheStats,
}

/// Performance statistics for cached adapters
#[derive(Debug, Default, Clone)]
pub struct AdapterCacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub loads_from_vdb: u64,
    pub saves_to_vdb: u64,
    pub evictions: u64,
    pub total_inference_time_ms: u64,
}

/// Configuration for LoRA weight cache
#[derive(Debug, Clone)]
pub struct LoRAWeightCacheConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    
    /// Maximum number of cached adapters
    pub max_adapters: usize,
    
    /// Auto-save dirty adapters after this many updates
    pub auto_save_threshold: usize,
    
    /// Auto-save dirty adapters after this duration
    pub auto_save_interval: Duration,
    
    /// Enable background cleanup task
    pub enable_background_cleanup: bool,
    
    /// Enable preloading of frequently used adapters
    pub enable_preloading: bool,
}

impl Default for LoRAWeightCacheConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            max_adapters: 100,
            auto_save_threshold: 1000,
            auto_save_interval: Duration::from_secs(300), // 5 minutes
            enable_background_cleanup: true,
            enable_preloading: true,
        }
    }
}

/// Intelligent weight cache for LoRA adapters
pub struct LoRAWeightCache {
    /// Cached adapters with LRU tracking
    cache: Arc<RwLock<HashMap<LoRAId, CachedLoRAAdapter>>>,
    
    /// Storage manager for VDB operations
    storage_manager: Arc<LoRAStorageManager>,
    
    /// Configuration
    config: LoRAWeightCacheConfig,
    
    /// Global statistics
    stats: Arc<RwLock<CacheStats>>,
    
    /// Background task handle
    _cleanup_task: Option<tokio::task::JoinHandle<()>>,
}

/// Global cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub total_memory_usage: usize,
    pub active_adapters: usize,
    pub total_cache_hits: u64,
    pub total_cache_misses: u64,
    pub total_evictions: u64,
    pub avg_load_time_ms: f64,
    pub cache_hit_rate: f64,
}

impl LoRAWeightCache {
    /// Create new LoRA weight cache
    pub async fn new(
        storage_manager: Arc<LoRAStorageManager>,
        config: Option<LoRAWeightCacheConfig>,
    ) -> Result<Arc<Self>> {
        let config = config.unwrap_or_default();
        
        let cache_arc = Arc::new(Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            storage_manager,
            config: config.clone(),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            _cleanup_task: None,
        });
        
        // Start background cleanup task if enabled
        if config.enable_background_cleanup {
            let cache_clone = Arc::clone(&cache_arc);
            let _cleanup_task = Some(tokio::spawn(async move {
                cache_clone.background_cleanup_task().await;
            }));
        }
        
        Ok(cache_arc)
    }
    
    /// Get adapter from cache or load from VDB
    // TODO: Fix return type after sparse removal
    pub async fn get_adapter(&self, lora_id: &LoRAId) -> Result<Arc<PhantomData<()>>> {
        let start_time = Instant::now();
        
        // Try cache first
        {
            let mut cache = self.cache.write().await;
            if let Some(cached) = cache.get_mut(lora_id) {
                // Cache hit - update access time and ref count
                cached.last_accessed = Instant::now();
                cached.ref_count += 1;
                cached.cache_stats.cache_hits += 1;
                
                // Update global stats
                let mut stats = self.stats.write().await;
                stats.total_cache_hits += 1;
                
                println!("🎯 Cache hit for LoRA adapter: {}", lora_id);
                return Ok(Arc::clone(&cached.adapter));
            }
        }
        
        // Cache miss - load from VDB
        println!("📁 Loading LoRA adapter from VDB: {}", lora_id);
        let adapter = self.storage_manager.load_adapter_weights(lora_id).await?;
        let memory_usage = adapter.memory_usage().await;
        
        // Create cached entry
        let cached_adapter = CachedLoRAAdapter {
            adapter: Arc::new(adapter),
            last_accessed: Instant::now(),
            ref_count: 1,
            memory_usage,
            is_dirty: false,
            cache_stats: AdapterCacheStats {
                cache_misses: 1,
                loads_from_vdb: 1,
                ..Default::default()
            },
        };
        
        // Check memory limits and evict if needed
        self.ensure_memory_limits().await?;
        
        // Insert into cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(lora_id.clone(), cached_adapter.clone());
        }
        
        // Update global statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_cache_misses += 1;
            stats.total_memory_usage += memory_usage;
            stats.active_adapters += 1;
            stats.avg_load_time_ms = (stats.avg_load_time_ms * (stats.total_cache_misses - 1) as f64 
                + start_time.elapsed().as_millis() as f64) / stats.total_cache_misses as f64;
            stats.cache_hit_rate = stats.total_cache_hits as f64 
                / (stats.total_cache_hits + stats.total_cache_misses) as f64;
        }
        
        println!("📁 Loaded LoRA adapter {} in {:.2}ms", lora_id, start_time.elapsed().as_millis());
        
        Ok(cached_adapter.adapter)
    }
    
    /// Release adapter reference (for resource management)
    pub async fn release_adapter(&self, lora_id: &LoRAId) -> Result<()> {
        let mut cache = self.cache.write().await;
        if let Some(cached) = cache.get_mut(lora_id) {
            cached.ref_count = cached.ref_count.saturating_sub(1);
        }
        Ok(())
    }
    
    /// Mark adapter as dirty (weights modified)
    pub async fn mark_dirty(&self, lora_id: &LoRAId, update_count: usize) -> Result<()> {
        let mut cache = self.cache.write().await;
        if let Some(cached) = cache.get_mut(lora_id) {
            cached.is_dirty = true;
            cached.last_accessed = Instant::now();
            
            // Auto-save if threshold reached
            if update_count >= self.config.auto_save_threshold {
                let adapter = Arc::clone(&cached.adapter);
                let storage_manager = Arc::clone(&self.storage_manager);
                let lora_id = lora_id.clone();
                
                // Spawn background save task
                tokio::spawn(async move {
                    if let Err(e) = storage_manager.save_adapter_weights(&lora_id, &adapter, true).await {
                        eprintln!("Failed to auto-save adapter {}: {}", lora_id, e);
                    }
                });
                
                cached.is_dirty = false;
                cached.cache_stats.saves_to_vdb += 1;
            }
        }
        Ok(())
    }
    
    /// Preload frequently used adapters
    pub async fn preload_adapters(self: &Arc<Self>, lora_ids: &[LoRAId]) -> Result<()> {
        if !self.config.enable_preloading {
            return Ok(());
        }
        
        println!("🚀 Preloading {} LoRA adapters", lora_ids.len());
        
        // Load adapters concurrently
        let mut tasks = Vec::new();
        for lora_id in lora_ids {
            let lora_id = lora_id.clone();
            let cache = Arc::clone(self);
            
            tasks.push(tokio::spawn(async move {
                if let Err(e) = cache.get_adapter(&lora_id).await {
                    eprintln!("Failed to preload adapter {}: {}", lora_id, e);
                }
                cache.release_adapter(&lora_id).await.ok();
            }));
        }
        
        // Wait for all preload tasks
        for task in tasks {
            task.await.ok();
        }
        
        println!("🚀 Preloading complete");
        
        Ok(())
    }
    
    /// Flush all dirty adapters to VDB storage
    pub async fn flush_all_dirty(&self) -> Result<()> {
        let dirty_adapters: Vec<(LoRAId, Arc<PhantomData<()>>)>; // TODO: Fix after sparse removal
        
        // Collect dirty adapters
        {
            let cache = self.cache.read().await;
            dirty_adapters = cache.iter()
                .filter(|(_, cached)| cached.is_dirty)
                .map(|(id, cached)| (id.clone(), Arc::clone(&cached.adapter)))
                .collect();
        }
        
        println!("💾 Flushing {} dirty adapters to VDB", dirty_adapters.len());
        
        // Save all dirty adapters concurrently
        let mut tasks = Vec::new();
        for (lora_id, adapter) in dirty_adapters {
            let storage_manager = Arc::clone(&self.storage_manager);
            
            tasks.push(tokio::spawn(async move {
                if let Err(e) = storage_manager.save_adapter_weights(&lora_id, &adapter, true).await {
                    eprintln!("Failed to save adapter {}: {}", lora_id, e);
                } else {
                    println!("💾 Saved adapter: {}", lora_id);
                }
            }));
        }
        
        // Wait for all saves to complete
        for task in tasks {
            task.await.ok();
        }
        
        // Mark all as clean
        {
            let mut cache = self.cache.write().await;
            for cached in cache.values_mut() {
                if cached.is_dirty {
                    cached.is_dirty = false;
                    cached.cache_stats.saves_to_vdb += 1;
                }
            }
        }
        
        println!("💾 Flush complete");
        
        Ok(())
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    /// Get adapter-specific statistics
    pub async fn get_adapter_stats(&self, lora_id: &LoRAId) -> Option<AdapterCacheStats> {
        let cache = self.cache.read().await;
        cache.get(lora_id).map(|cached| cached.cache_stats.clone())
    }
    
    /// Manually evict adapter from cache
    pub async fn evict_adapter(&self, lora_id: &LoRAId) -> Result<bool> {
        let mut cache = self.cache.write().await;
        
        if let Some(cached) = cache.get(lora_id) {
            // Don't evict if still referenced
            if cached.ref_count > 0 {
                return Ok(false);
            }
            
            // Save if dirty
            if cached.is_dirty {
                let adapter = Arc::clone(&cached.adapter);
                drop(cache); // Release lock for save operation
                
                self.storage_manager.save_adapter_weights(lora_id, &adapter, true).await?;
                
                cache = self.cache.write().await; // Re-acquire lock
            }
        }
        
        if let Some(cached) = cache.remove(lora_id) {
            // Update global stats
            drop(cache);
            let mut stats = self.stats.write().await;
            stats.total_memory_usage = stats.total_memory_usage.saturating_sub(cached.memory_usage);
            stats.active_adapters = stats.active_adapters.saturating_sub(1);
            stats.total_evictions += 1;
            
            println!("🗑️ Evicted LoRA adapter: {}", lora_id);
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Clear entire cache (flush dirty adapters first)
    pub async fn clear(&self) -> Result<()> {
        println!("🧹 Clearing LoRA weight cache");
        
        // Flush all dirty adapters first
        self.flush_all_dirty().await?;
        
        // Clear cache
        {
            let mut cache = self.cache.write().await;
            cache.clear();
        }
        
        // Reset statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_memory_usage = 0;
            stats.active_adapters = 0;
        }
        
        println!("🧹 Cache cleared");
        
        Ok(())
    }
    
    /// Ensure memory limits are respected
    async fn ensure_memory_limits(&self) -> Result<()> {
        let current_memory = {
            let stats = self.stats.read().await;
            stats.total_memory_usage
        };
        
        if current_memory > self.config.max_memory_bytes {
            println!("⚠️ Memory limit exceeded ({} bytes), starting LRU eviction", current_memory);
            self.lru_eviction().await?;
        }
        
        let adapter_count = {
            let cache = self.cache.read().await;
            cache.len()
        };
        
        if adapter_count > self.config.max_adapters {
            println!("⚠️ Adapter count limit exceeded ({}), starting LRU eviction", adapter_count);
            self.lru_eviction().await?;
        }
        
        Ok(())
    }
    
    /// Perform LRU eviction of least recently used adapters
    async fn lru_eviction(&self) -> Result<()> {
        let mut candidates: Vec<(LoRAId, Instant, usize)>;
        
        // Find eviction candidates (not currently referenced)
        {
            let cache = self.cache.read().await;
            candidates = cache.iter()
                .filter(|(_, cached)| cached.ref_count == 0)
                .map(|(id, cached)| (id.clone(), cached.last_accessed, cached.memory_usage))
                .collect();
        }
        
        // Sort by last accessed time (oldest first)
        candidates.sort_by_key(|(_, last_accessed, _)| *last_accessed);
        
        // Evict 25% of unreferenced adapters or until memory/count limits are met
        let target_evictions = (candidates.len() / 4).max(1);
        
        for (lora_id, _, _) in candidates.into_iter().take(target_evictions) {
            self.evict_adapter(&lora_id).await?;
            
            // Check if we've freed enough memory
            let current_memory = {
                let stats = self.stats.read().await;
                stats.total_memory_usage
            };
            
            if current_memory <= self.config.max_memory_bytes {
                break;
            }
        }
        
        Ok(())
    }
    
    /// Background cleanup task
    async fn background_cleanup_task(self: Arc<Self>) {
        let mut interval = tokio::time::interval(self.config.auto_save_interval);
        
        loop {
            interval.tick().await;
            
            // Auto-save dirty adapters
            if let Err(e) = self.auto_save_dirty_adapters().await {
                eprintln!("Background auto-save failed: {}", e);
            }
            
            // Perform maintenance
            if let Err(e) = self.ensure_memory_limits().await {
                eprintln!("Background maintenance failed: {}", e);
            }
        }
    }
    
    /// Auto-save adapters that have been dirty for too long
    async fn auto_save_dirty_adapters(&self) -> Result<()> {
        let dirty_adapters: Vec<(LoRAId, Arc<PhantomData<()>>)>; // TODO: Fix after sparse removal
        
        // Find adapters that need saving
        {
            let cache = self.cache.read().await;
            dirty_adapters = cache.iter()
                .filter(|(_, cached)| {
                    cached.is_dirty && 
                    cached.last_accessed.elapsed() > Duration::from_secs(60) // 1 minute threshold
                })
                .map(|(id, cached)| (id.clone(), Arc::clone(&cached.adapter)))
                .collect();
        }
        
        if !dirty_adapters.is_empty() {
            println!("🕐 Auto-saving {} stale dirty adapters", dirty_adapters.len());
            
            for (lora_id, adapter) in dirty_adapters {
                if let Err(e) = self.storage_manager.save_adapter_weights(&lora_id, &adapter, true).await {
                    eprintln!("Failed to auto-save adapter {}: {}", lora_id, e);
                } else {
                    // Mark as clean
                    let mut cache = self.cache.write().await;
                    if let Some(cached) = cache.get_mut(&lora_id) {
                        cached.is_dirty = false;
                        cached.cache_stats.saves_to_vdb += 1;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// List all cached adapters
    pub async fn list_cached_adapters(&self) -> Vec<LoRAId> {
        let cache = self.cache.read().await;
        cache.keys().cloned().collect()
    }
    
    /// Get memory usage by adapter
    pub async fn get_memory_breakdown(&self) -> HashMap<LoRAId, usize> {
        let cache = self.cache.read().await;
        cache.iter()
            .map(|(id, cached)| (id.clone(), cached.memory_usage))
            .collect()
    }
}

impl Drop for LoRAWeightCache {
    fn drop(&mut self) {
        // Attempt to flush dirty adapters on drop (best effort)
        if let Some(handle) = &mut self._cleanup_task {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::lora_storage_manager::LoRAStorageManager;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_lora_weight_cache_basic() {
        let temp_dir = tempdir().unwrap();
        let registry_dir = temp_dir.path().join("registry");
        let vdb_dir = temp_dir.path().join("vdb");
        
        let storage_manager = Arc::new(
            LoRAStorageManager::new(registry_dir, vdb_dir, None).await.unwrap()
        );
        
        let cache = LoRAWeightCache::new(
            storage_manager.clone(), 
            Some(LoRAWeightCacheConfig {
                max_memory_bytes: 100 * 1024 * 1024, // 100MB
                max_adapters: 10,
                ..Default::default()
            })
        ).await.unwrap();
        
        // Register a test adapter
        let lora_config = crate::api::LoRAConfig::default();
        let lora_id = storage_manager.register_adapter(
            "test_cache".to_string(),
            "test_model".to_string(),
            lora_config,
            None,
        ).await.unwrap();
        
        // Test cache miss (load from VDB)
        let adapter1 = cache.get_adapter(&lora_id).await.unwrap();
        let stats = cache.get_stats().await;
        assert_eq!(stats.total_cache_misses, 1);
        assert_eq!(stats.active_adapters, 1);
        
        // Test cache hit
        let adapter2 = cache.get_adapter(&lora_id).await.unwrap();
        let stats = cache.get_stats().await;
        assert_eq!(stats.total_cache_hits, 1);
        
        // Release references
        cache.release_adapter(&lora_id).await.unwrap();
        cache.release_adapter(&lora_id).await.unwrap();
        
        // Test eviction
        let evicted = cache.evict_adapter(&lora_id).await.unwrap();
        assert!(evicted);
        
        let stats = cache.get_stats().await;
        assert_eq!(stats.active_adapters, 0);
    }
    
    #[tokio::test]
    async fn test_dirty_adapter_auto_save() {
        let temp_dir = tempdir().unwrap();
        let registry_dir = temp_dir.path().join("registry");
        let vdb_dir = temp_dir.path().join("vdb");
        
        let storage_manager = Arc::new(
            LoRAStorageManager::new(registry_dir, vdb_dir, None).await.unwrap()
        );
        
        let cache = LoRAWeightCache::new(
            storage_manager.clone(), 
            Some(LoRAWeightCacheConfig {
                auto_save_threshold: 5, // Save after 5 updates
                ..Default::default()
            })
        ).await.unwrap();
        
        // Register and load adapter
        let lora_config = crate::api::LoRAConfig::default();
        let lora_id = storage_manager.register_adapter(
            "dirty_test".to_string(),
            "test_model".to_string(),
            lora_config,
            None,
        ).await.unwrap();
        
        let _adapter = cache.get_adapter(&lora_id).await.unwrap();
        
        // Mark as dirty multiple times to trigger auto-save
        for i in 1..=10 {
            cache.mark_dirty(&lora_id, i).await.unwrap();
        }
        
        // Check adapter stats
        if let Some(adapter_stats) = cache.get_adapter_stats(&lora_id).await {
            assert!(adapter_stats.saves_to_vdb > 0);
        }
    }
}