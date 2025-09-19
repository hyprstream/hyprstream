//! UUID-based model cache for efficient model reuse across requests

use anyhow::Result;
use lru::LruCache;
use std::sync::Arc;
use std::num::NonZeroUsize;
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;
use tokio::sync::Mutex;
use uuid::Uuid;
use crate::runtime::{TorchEngine, RuntimeConfig, RuntimeEngine};
use tracing::{info, warn, debug};

/// Cached model entry containing the loaded engine
#[derive(Clone)]
pub struct CachedModel {
    /// UUID of the model
    pub model_id: Uuid,
    /// Model name
    pub model_name: String,
    /// Loaded engine with model
    pub engine: Arc<Mutex<TorchEngine>>,
    /// Last access time for debugging
    pub last_accessed: std::time::Instant,
}

/// Global model cache with LRU eviction
pub struct ModelCache {
    /// LRU cache mapping model UUID to loaded engine
    cache: Arc<Mutex<LruCache<Uuid, CachedModel>>>,
    /// Cache for name to UUID mappings to avoid disk scanning
    name_cache: Arc<RwLock<HashMap<String, Uuid>>>,
    /// Maximum cache size
    max_size: usize,
    /// Model storage for resolving paths
    model_storage: Arc<crate::api::model_storage::ModelStorage>,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(
        max_size: usize,
        model_storage: Arc<crate::api::model_storage::ModelStorage>,
    ) -> Self {
        let cache_size = NonZeroUsize::new(max_size).unwrap_or(NonZeroUsize::new(5).unwrap());
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(cache_size))),
            name_cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            model_storage,
        }
    }

    /// Get or load a model by UUID
    pub async fn get_or_load(&self, model_uuid: Uuid, model_name: &str) -> Result<Arc<Mutex<TorchEngine>>> {
        // First check if model is in cache
        {
            let mut cache = self.cache.lock().await;
            if let Some(cached) = cache.get_mut(&model_uuid) {
                info!("Model {} (UUID: {}) found in cache", model_name, model_uuid);
                cached.last_accessed = std::time::Instant::now();
                return Ok(Arc::clone(&cached.engine));
            }
        }

        // Model not in cache, load it
        info!("Model {} (UUID: {}) not in cache, loading...", model_name, model_uuid);
        
        // Get model path from storage with direct loading fallback
        let model_id = crate::api::model_storage::ModelId(model_uuid);
        let metadata = match self.model_storage.get_metadata_by_id(&model_id).await {
            Ok(meta) => meta,
            Err(e) => {
                // UUID not found in storage at all
                warn!("Model not found for UUID: {}. Error: {}", model_uuid, e);
                self.invalidate_uuid_from_name_cache(model_uuid).await;
                return Err(anyhow::anyhow!("Model not found for UUID: {}", model_uuid));
            }
        };
        // Use UUID-based path
        let model_path = self.model_storage.get_models_dir().join(model_id.to_string());

        // Create new engine and load model
        let config = RuntimeConfig::default();
        let mut engine = TorchEngine::new(config)?;
        
        info!("Loading model from: {:?}", model_path);
        engine.load_model(&model_path).await?;
        info!("Successfully loaded model {} into cache", model_name);

        // Create cached entry
        let cached_model = CachedModel {
            model_id: model_uuid,
            model_name: model_name.to_string(),
            engine: Arc::new(Mutex::new(engine)),
            last_accessed: std::time::Instant::now(),
        };

        let engine_ref = Arc::clone(&cached_model.engine);

        // Add to cache (this may evict LRU model)
        {
            let mut cache = self.cache.lock().await;
            if let Some((evicted_id, evicted_model)) = cache.push(model_uuid, cached_model) {
                warn!("Evicted model {} (UUID: {}) from cache", evicted_model.model_name, evicted_id);
            }
        }

        Ok(engine_ref)
    }

    /// Get model by name (resolves name to UUID first)
    pub async fn get_or_load_by_name(&self, model_name: &str) -> Result<Arc<Mutex<TorchEngine>>> {
        // First try to parse as UUID directly
        if let Ok(uuid) = Uuid::parse_str(model_name) {
            return self.get_or_load(uuid, model_name).await;
        }

        // Check name cache first to avoid disk scanning
        {
            let name_cache = self.name_cache.read().await;
            if let Some(&cached_uuid) = name_cache.get(model_name) {
                debug!("Found cached name->UUID mapping for {}: {}", model_name, cached_uuid);
                // Try to load, but handle stale cache entries
                match self.get_or_load(cached_uuid, model_name).await {
                    Ok(engine) => return Ok(engine),
                    Err(e) => {
                        warn!("Cached UUID {} for name '{}' is invalid: {}. Will rescan.", cached_uuid, model_name, e);
                        // Continue to rescan below
                    }
                }
            }
        }

        // Name not in cache or cached UUID was invalid, resolve it (this will scan disk)
        info!("Name '{}' not in cache or stale, scanning disk for UUID mapping", model_name);
        let uuid = self.resolve_model_name_to_uuid(model_name).await?;
        
        // Validate the UUID before caching (will also load from directory if needed)
        let model_id = crate::api::model_storage::ModelId(uuid);
        if let Err(e) = self.model_storage.get_metadata_by_id(&model_id).await {
            return Err(anyhow::anyhow!("Model '{}' resolved to UUID {} but not found: {}", model_name, uuid, e));
        }
        
        // Cache the name->UUID mapping for future requests
        {
            let mut name_cache = self.name_cache.write().await;
            name_cache.insert(model_name.to_string(), uuid);
            debug!("Cached name->UUID mapping: {} -> {}", model_name, uuid);
        }
        
        self.get_or_load(uuid, model_name).await
    }

    /// Resolve model name to UUID (scans disk - should be called rarely)
    async fn resolve_model_name_to_uuid(&self, model_name: &str) -> Result<Uuid> {
        // Try to find model through storage API
        let models = self.model_storage.children().await?;
        
        for (id, metadata) in models {
            // Check against the model's name field
            if metadata.name == model_name {
                return Ok(id.0);
            }
            
            // Check against display_name
            if let Some(ref display_name) = metadata.display_name {
                if display_name == model_name {
                    return Ok(id.0);
                }
            }
            
            // Check directory name (UUID-based path) - exact match only
            let model_path = self.model_storage.get_uuid_model_path(&id);
            if let Some(dir_name) = model_path.file_name() {
                let dir_name_str = dir_name.to_string_lossy();
                if dir_name_str == model_name {
                    return Ok(id.0);
                }
            }
        }
        
        Err(anyhow::anyhow!("Model '{}' not found in storage", model_name))
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.lock().await;
        let name_cache = self.name_cache.read().await;
        CacheStats {
            cached_models: cache.len(),
            cached_names: name_cache.len(),
            max_size: self.max_size,
        }
    }

    /// Clear the cache
    pub async fn clear(&self) {
        let mut cache = self.cache.lock().await;
        cache.clear();
        let mut name_cache = self.name_cache.write().await;
        name_cache.clear();
        info!("Model cache and name cache cleared");
    }
    
    /// Invalidate cache entries for a specific model
    pub async fn invalidate_model(&self, model_uuid: Uuid) {
        // Remove from model cache
        let mut cache = self.cache.lock().await;
        if cache.pop(&model_uuid).is_some() {
            info!("Invalidated model {} from cache", model_uuid);
        }
        
        // Remove associated names from name cache
        self.invalidate_uuid_from_name_cache(model_uuid).await;
    }
    
    /// Remove a UUID from the name cache
    async fn invalidate_uuid_from_name_cache(&self, model_uuid: Uuid) {
        let mut name_cache = self.name_cache.write().await;
        let names_to_remove: Vec<String> = name_cache
            .iter()
            .filter(|(_, &uuid)| uuid == model_uuid)
            .map(|(name, _)| name.clone())
            .collect();
        
        for name in names_to_remove {
            name_cache.remove(&name);
            debug!("Removed name '{}' -> {} from cache", name, model_uuid);
        }
    }

    /// Check if a model is cached by UUID
    pub async fn is_cached(&self, model_uuid: Uuid) -> bool {
        let cache = self.cache.lock().await;
        cache.peek(&model_uuid).is_some()
    }
    
    /// Check if a model is cached by name
    pub async fn is_cached_by_name(&self, model_name: &str) -> bool {
        // First try to parse as UUID directly
        if let Ok(uuid) = Uuid::parse_str(model_name) {
            return self.is_cached(uuid).await;
        }
        
        // Check if we have the name in cache and if that UUID is loaded
        let name_cache = self.name_cache.read().await;
        if let Some(&uuid) = name_cache.get(model_name) {
            drop(name_cache); // Release read lock before calling is_cached
            return self.is_cached(uuid).await;
        }
        
        false
    }
    
    /// Pre-populate name cache by scanning available models once
    pub async fn warm_name_cache(&self) -> Result<()> {
        self.refresh_name_cache().await
    }
    
    /// Refresh the name cache, removing stale entries and adding new ones
    pub async fn refresh_name_cache(&self) -> Result<()> {
        info!("Refreshing name cache by scanning available models...");
        let models = self.model_storage.children().await?;
        
        // Build new cache from scratch to remove stale entries
        let mut new_name_cache = HashMap::new();
        let mut valid_uuids = std::collections::HashSet::new();
        
        for (id, metadata) in models {
            // All models from children() are already validated
            valid_uuids.insert(id.0);
            debug!("Model {} (UUID: {}) is valid", metadata.name, id.0);
            
            // Cache by name field
            new_name_cache.insert(metadata.name.clone(), id.0);
            debug!("Cached name '{}' -> {}", metadata.name, id.0);
            
            // Cache by display_name if present
            if let Some(ref display_name) = metadata.display_name {
                new_name_cache.insert(display_name.clone(), id.0);
                debug!("Cached display_name '{}' -> {}", display_name, id.0);
            }
            
            // Cache by directory name (UUID-based path) - UUID only
            let model_path = self.model_storage.get_uuid_model_path(&id);
            if let Some(dir_name) = model_path.file_name() {
                let dir_name_str = dir_name.to_string_lossy().to_string();
                debug!("Cached UUID dir name '{}' -> {}", dir_name_str, id.0);
                new_name_cache.insert(dir_name_str, id.0);
            }
        }
        
        // Replace the old cache with the new one atomically
        let mut name_cache = self.name_cache.write().await;
        let old_size = name_cache.len();
        *name_cache = new_name_cache;
        let new_size = name_cache.len();
        
        // Also clean up the model cache - remove entries for deleted models
        {
            let mut model_cache = self.cache.lock().await;
            let mut to_remove = Vec::new();
            
            // Collect UUIDs to remove (can't modify while iterating)
            for (uuid, _) in model_cache.iter() {
                if !valid_uuids.contains(uuid) {
                    to_remove.push(*uuid);
                }
            }
            
            // Remove stale entries
            for uuid in to_remove {
                model_cache.pop(&uuid);
                warn!("Removed deleted model {} from cache", uuid);
            }
        }
        
        info!("Name cache refreshed: {} -> {} entries", old_size, new_size);
        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cached_models: usize,
    pub cached_names: usize,
    pub max_size: usize,
}