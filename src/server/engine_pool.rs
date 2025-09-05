//! Engine pool management for concurrent inference

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use crate::runtime::{TorchEngine, RuntimeEngine, RuntimeConfig};
use tracing::{info, warn, error};
use thiserror::Error;

/// Engine pool errors
#[derive(Error, Debug)]
pub enum EnginePoolError {
    
    #[error("Model '{0}' not found")]
    ModelNotFound(String),
    
    #[error("Model '{0}' failed to load: {1}")]
    ModelLoadFailed(String, String),
    
    #[error("No engines available")]
    NoEnginesAvailable,
    
    #[error("Engine pool exhausted")]
    PoolExhausted,
    
    #[error("Internal error: {0}")]
    Internal(String),
}


/// Pool of inference engines
pub struct EnginePool {
    /// Available engines
    available: Arc<Mutex<VecDeque<Arc<Mutex<TorchEngine>>>>>,
    
    /// Semaphore for limiting concurrent access
    semaphore: Arc<Semaphore>,
    
    /// Maximum pool size
    max_size: usize,
    
    /// Current pool size
    current_size: Arc<Mutex<usize>>,
    
    /// Default model path
    default_model: Option<String>,
    
    /// Model storage for resolving model names/UUIDs to paths
    model_storage: Arc<crate::api::model_storage::ModelStorage>,
    
}

impl EnginePool {
    /// Create a new engine pool with model storage for resolution
    pub async fn new(
        max_size: usize, 
        default_model: Option<String>,
        model_storage: Arc<crate::api::model_storage::ModelStorage>,
    ) -> Result<Self> {
        let pool = Self {
            available: Arc::new(Mutex::new(VecDeque::new())),
            semaphore: Arc::new(Semaphore::new(max_size)),
            max_size,
            current_size: Arc::new(Mutex::new(0)),
            default_model,
            model_storage,
        };
        
        // Add initial engine if default model is specified
        if let Some(ref model_identifier) = pool.default_model {
            info!("Adding initial engine with model: {}", model_identifier);
            pool.add().await?;
        }
        
        Ok(pool)
    }
    
    /// Add a new engine to the pool
    async fn add(&self) -> Result<()> {
        let mut current = self.current_size.lock().await;
        if *current >= self.max_size {
            return Ok(());
        }
        
        let config = RuntimeConfig::default();
        let mut engine = TorchEngine::new(config)?;
        
        // Load default model if specified
        if let Some(ref model_identifier) = self.default_model {
            // Try to find the model path
            match self.find_model(model_identifier).await {
                Ok(model_path) => {
                    info!("Loading model from: {:?}", model_path);
                    engine.load_model(&model_path).await?;
                    info!("Successfully loaded model and tokenizer into engine");
                }
                Err(e) => {
                    warn!("Failed to resolve model '{}': {}", model_identifier, e);
                    warn!("Engine created without model. Set HYPRSTREAM_DEFAULT_MODEL to a valid model name, UUID, or path.");
                }
            }
        } else {
            warn!("No default model specified. Engine created without model.");
            warn!("Set HYPRSTREAM_DEFAULT_MODEL environment variable to a model identifier.");
        }
        
        self.available.lock().await.push_back(Arc::new(Mutex::new(engine)));
        *current += 1;
        
        Ok(())
    }
    
    /// Acquire an engine from the pool
    pub async fn acquire(&self) -> Result<EngineGuard> {
        // Try to get an available engine
        let engine = {
            let mut available = self.available.lock().await;
            available.pop_front()
        };
        
        let engine = if let Some(engine) = engine {
            engine
        } else {
            // Create a new engine if under limit
            let current = *self.current_size.lock().await;
            if current < self.max_size {
                self.add().await?;
                self.available.lock().await
                    .pop_front()
                    .ok_or_else(|| anyhow::anyhow!("Failed to create new engine"))?
            } else {
                // Wait for an engine to become available
                warn!("Engine pool exhausted, waiting for available engine");
                let _permit = self.semaphore.acquire().await?;
                loop {
                    if let Some(engine) = self.available.lock().await.pop_front() {
                        break engine;
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
        };
        
        Ok(EngineGuard {
            engine: Some(engine),
            pool: Arc::clone(&self.available),
        })
    }
    
    /// Acquire an engine with the specified model loaded
    /// This will load the model synchronously if not already loaded
    pub async fn acquire_model(&self, model_name: &str) -> Result<EngineGuard, EnginePoolError> {
        // Try to find an engine that already has this model loaded
        let matching_engine = {
            let mut available = self.available.lock().await;
            let mut found_index = None;
            for (index, engine) in available.iter().enumerate() {
                let engine_locked = engine.lock().await;
                if engine_locked.is_loaded() {
                    let current_model = engine_locked.model_info();
                    if self.matches(&current_model.name, model_name) {
                        found_index = Some(index);
                        break;
                    }
                }
            }
            // Remove the matching engine from available pool
            found_index.and_then(|idx| available.remove(idx))
        };
        
        if let Some(engine) = matching_engine {
            // Found an engine with the model already loaded
            info!("Found engine with model {} already loaded", model_name);
            return Ok(EngineGuard {
                engine: Some(engine),
                pool: Arc::clone(&self.available),
            });
        }
        
        // No engine has this model loaded, acquire any engine and load it
        info!("No engine has model {}, acquiring engine to load it", model_name);
        let engine_guard = self.acquire().await
            .map_err(|e| EnginePoolError::Internal(e.to_string()))?;
        let engine = engine_guard.get();
        
        // Double-check this engine doesn't already have the model
        {
            let engine_locked = engine.lock().await;
            let current_model = engine_locked.model_info();
            
            if engine_locked.is_loaded() && self.matches(&current_model.name, model_name) {
                info!("Engine already has model {} loaded", model_name);
                return Ok(engine_guard);
            }
        }
        
        // Find the model path
        let model_path = self.find_model(model_name).await
            .map_err(|_| EnginePoolError::ModelNotFound(model_name.to_string()))?;
        
        // Load the model synchronously (within this request)
        info!("Loading model {} from {:?}", model_name, model_path);
        {
            let mut engine_locked = engine.lock().await;
            engine_locked.load_model(&model_path).await
                .map_err(|e| EnginePoolError::ModelLoadFailed(model_name.to_string(), e.to_string()))?;
            info!("Successfully loaded model {}", model_name);
        }
        
        // Return the engine with the model loaded
        Ok(engine_guard)
    }
    /// Find model path from identifier using model storage
    async fn find_model(&self, model_name: &str) -> Result<std::path::PathBuf> {
        // First try to parse as UUID and resolve via model storage
        if let Ok(uuid) = uuid::Uuid::parse_str(model_name) {
            let model_id = crate::api::model_storage::ModelId(uuid);
            if let Ok(metadata) = self.model_storage.get_metadata_by_id(&model_id).await {
                if let Some(path) = metadata.local_path {
                    return Ok(path);
                }
                return Err(anyhow::anyhow!("Model '{}' found but not cached locally", model_name));
            }
        }
        
        // Try to find by scanning the model list (for name-based lookup)
        let models = self.model_storage.children().await?;
        for (_id, metadata) in models {
            // Check against the model's name field
            if metadata.name == model_name {
                if let Some(ref local_path) = metadata.local_path {
                    return Ok(local_path.clone());
                }
            }
            
            // Check against display_name
            if let Some(ref display_name) = metadata.display_name {
                if display_name == model_name {
                    if let Some(ref local_path) = metadata.local_path {
                        return Ok(local_path.clone());
                    }
                }
            }
            
            // Legacy: Check directory name
            if let Some(ref local_path) = metadata.local_path {
                if let Some(dir_name) = local_path.file_name() {
                    let dir_name_str = dir_name.to_string_lossy();
                    
                    // Exact match on directory name
                    if dir_name_str == model_name {
                        return Ok(local_path.clone());
                    }
                    
                    // Handle "org/model" format by checking if dir contains the model name
                    if model_name.contains('/') {
                        let model_part = model_name.split('/').last().unwrap_or("");
                        if dir_name_str.to_lowercase().contains(&model_part.to_lowercase()) {
                            return Ok(local_path.clone());
                        }
                    }
                }
            }
        }
        
        // If still not found, check if it's a direct path
        let path = std::path::Path::new(model_name);
        if path.exists() && path.is_absolute() {
            return Ok(path.to_path_buf());
        }
        
        Err(anyhow::anyhow!("Model '{}' not found in storage", model_name))
    }
    
    /// Check if two model names match
    fn matches(&self, current: &str, requested: &str) -> bool {
        if current == "unloaded" {
            return false;
        }
        
        // Exact match
        if current == requested {
            return true;
        }
        
        // Handle different name formats:
        // Current might be: "Qwen_Qwen3-4B-Instruct-2507" (directory name with underscore)
        // Requested might be: "Qwen/Qwen3-4B-Instruct-2507" (with slash)
        
        // Normalize both by replacing common separators
        let normalize = |s: &str| -> String {
            s.replace('/', "_").replace('-', "_").to_lowercase()
        };
        
        let current_normalized = normalize(current);
        let requested_normalized = normalize(requested);
        
        // Check exact match after normalization
        if current_normalized == requested_normalized {
            return true;
        }
        
        // Also check if one contains the other (for partial matches)
        // But only the significant part after org separator
        let requested_name = if requested.contains('/') {
            requested.split('/').last().unwrap_or(requested)
        } else {
            requested
        };
        
        current.contains(requested_name) || 
        current_normalized.contains(&normalize(requested_name))
    }
    /// Get pool statistics
    pub async fn stats(&self) -> EnginePoolStats {
        EnginePoolStats {
            total_engines: *self.current_size.lock().await,
            available_engines: self.available.lock().await.len(),
            max_engines: self.max_size,
        }
    }
}

/// Guard for automatic engine return to pool
pub struct EngineGuard {
    engine: Option<Arc<Mutex<TorchEngine>>>,
    pool: Arc<Mutex<VecDeque<Arc<Mutex<TorchEngine>>>>>,
}

impl EngineGuard {
    /// Get the engine
    pub fn get(&self) -> Arc<Mutex<TorchEngine>> {
        self.engine.as_ref()
            .expect("EngineGuard should always have an engine")
            .clone()
    }
    
    /// Take the engine out of the guard (prevents it from being returned to pool on drop)
    pub fn take(&mut self) -> Option<Arc<Mutex<TorchEngine>>> {
        self.engine.take()
    }
}

impl Drop for EngineGuard {
    fn drop(&mut self) {
        // CRITICAL: Must always return engine to pool to prevent resource leak
        if let Some(engine) = self.engine.take() {
            // Try immediate non-blocking return
            if let Ok(mut pool) = self.pool.try_lock() {
                pool.push_back(engine);
                tracing::trace!("Engine returned to pool synchronously");
            } else {
                // Lock is contended, we must ensure the engine is returned
                // Use blocking spawn to ensure the task completes
                let pool = self.pool.clone();
                let engine_id = Arc::as_ptr(&engine) as usize; // For tracking
                
                // Create a detached task that WILL complete
                std::thread::spawn(move || {
                    // Use a blocking runtime handle to ensure completion
                    let handle = tokio::runtime::Handle::current();
                    let result = handle.block_on(async move {
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(30), // Longer timeout
                            pool.lock()
                        ).await {
                            Ok(mut pool_guard) => {
                                pool_guard.push_back(engine);
                                tracing::info!("Engine {:x} returned to pool asynchronously", engine_id);
                                Ok(())
                            }
                            Err(_) => {
                                // CRITICAL: Engine would be leaked here
                                // Log with high severity for monitoring
                                tracing::error!(
                                    "RESOURCE LEAK: Failed to return engine {:x} to pool after 30s timeout. \
                                     This is a critical error that requires investigation.",
                                    engine_id
                                );
                                Err(())
                            }
                        }
                    });
                    
                    if result.is_err() {
                        // Additional alerting could go here
                        eprintln!("CRITICAL: Engine resource leak detected!");
                    }
                });
            }
        }
    }
}

/// Engine pool statistics
#[derive(Debug, Clone)]
pub struct EnginePoolStats {
    pub total_engines: usize,
    pub available_engines: usize,
    pub max_engines: usize,
}