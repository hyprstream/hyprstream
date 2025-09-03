//! Engine pool management for concurrent inference

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, Semaphore};
use crate::runtime::{TorchEngine, RuntimeEngine, RuntimeConfig};
use tracing::{info, warn, error};
use thiserror::Error;

/// Engine pool errors
#[derive(Error, Debug)]
pub enum EnginePoolError {
    #[error("Model '{0}' is currently loading, please retry")]
    ModelLoading(String),
    
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

/// Model loading state
#[derive(Debug, Clone, PartialEq)]
pub enum ModelLoadingState {
    /// No model loaded
    Unloaded,
    /// Model is currently loading
    Loading { model_name: String },
    /// Model loaded and ready
    Loaded { model_name: String },
    /// Model loading failed
    Failed { model_name: String, error: String },
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
    
    /// Track loading state for each model
    loading_states: Arc<RwLock<HashMap<String, ModelLoadingState>>>,
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
            loading_states: Arc::new(RwLock::new(HashMap::new())),
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
                    
                    // Mark the model as loaded in the loading states
                    let actual_model_name = engine.model_info().name.clone();
                    let mut states = self.loading_states.write().await;
                    states.insert(model_identifier.clone(), ModelLoadingState::Loaded { 
                        model_name: actual_model_name.clone() 
                    });
                    // Also store under the actual name if different
                    if actual_model_name != *model_identifier {
                        states.insert(actual_model_name.clone(), ModelLoadingState::Loaded { 
                            model_name: actual_model_name.clone() 
                        });
                    }
                    info!("Marked model as loaded: {} (actual: {})", model_identifier, actual_model_name);
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
    
    /// Acquire an engine with the specified model loaded (non-blocking)
    /// Returns error if model is still loading, client should retry
    pub async fn acquire_model(&self, model_name: &str) -> Result<EngineGuard, EnginePoolError> {
        // Check loading state first
        {
            let states = self.loading_states.read().await;
            if let Some(state) = states.get(model_name) {
                match state {
                    ModelLoadingState::Loading { .. } => {
                        return Err(EnginePoolError::ModelLoading(model_name.to_string()));
                    }
                    ModelLoadingState::Failed { error, .. } => {
                        return Err(EnginePoolError::ModelLoadFailed(model_name.to_string(), error.clone()));
                    }
                    ModelLoadingState::Loaded { .. } => {
                        // Model is loaded, continue to find the engine with it
                    }
                    _ => {}
                }
            }
        }
        
        // Try to find an engine that already has this model loaded
        // Check all available engines first
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
            return Ok(EngineGuard {
                engine: Some(engine),
                pool: Arc::clone(&self.available),
            });
        }
        
        // No engine has this model loaded, acquire any engine and load it
        let mut engine_guard = self.acquire().await
            .map_err(|e| EnginePoolError::Internal(e.to_string()))?;
        let engine = engine_guard.get();
        
        // Double-check this engine doesn't already have the model
        {
            let engine_locked = engine.lock().await;
            let current_model = engine_locked.model_info();
            
            if engine_locked.is_loaded() && self.matches(&current_model.name, model_name) {
                return Ok(engine_guard);
            }
        }
        
        // Model needs loading - take the engine out of the guard for background loading
        // We need to extract the engine from the guard without returning it to the pool
        let engine_for_loading = engine_guard.take()
            .ok_or_else(|| EnginePoolError::Internal("Engine guard has no engine".to_string()))?;
        
        // Start async load with the extracted engine
        self.start_model_loading(model_name.to_string(), engine_for_loading).await
            .map_err(|e| EnginePoolError::Internal(e.to_string()))?;
        
        // The guard is now empty and won't return anything to the pool when dropped
        drop(engine_guard);
        
        // Tell client to retry
        Err(EnginePoolError::ModelLoading(model_name.to_string()))
    }
    
    /// Start loading a model in the background
    async fn start_model_loading(&self, model_name: String, engine: Arc<Mutex<TorchEngine>>) -> Result<()> {
        // Check if already loading
        {
            let mut states = self.loading_states.write().await;
            if let Some(ModelLoadingState::Loading { .. }) = states.get(&model_name) {
                return Ok(()); // Already loading
            }
            states.insert(model_name.clone(), ModelLoadingState::Loading { model_name: model_name.clone() });
        }
        
        // Find model path
        let model_path = match self.find_model(&model_name).await {
            Ok(path) => path,
            Err(_) => {
                let mut states = self.loading_states.write().await;
                let error_msg = format!("Model not found in storage");
                states.insert(model_name.clone(), ModelLoadingState::Failed { 
                    model_name: model_name.clone(), 
                    error: error_msg.clone()
                });
                return Err(EnginePoolError::ModelNotFound(model_name).into());
            }
        };
        
        let loading_states = self.loading_states.clone();
        let model_name_clone = model_name.clone();
        
        // Need a reference to the pool to return the engine after loading
        let available_pool = self.available.clone();
        
        // Spawn background loading task
        tokio::spawn(async move {
            info!("Starting background load of model: {}", model_name_clone);
            
            let result = {
                let mut engine_locked = engine.lock().await;
                // Store the actual loaded model name
                let load_result = engine_locked.load_model(&model_path).await;
                if load_result.is_ok() {
                    // Get the actual model name after loading
                    let actual_name = engine_locked.model_info().name.clone();
                    (load_result, actual_name)
                } else {
                    (load_result, model_name_clone.clone())
                }
            };
            
            // Update state based on result
            let mut states = loading_states.write().await;
            match result.0 {
                Ok(_) => {
                    info!("Successfully loaded model: {} (actual: {})", model_name_clone, result.1);
                    // Mark as loaded and clear loading state
                    states.insert(model_name_clone.clone(), ModelLoadingState::Loaded { 
                        model_name: result.1.clone() 
                    });
                    // Also store under the actual name if different
                    if result.1 != model_name_clone {
                        states.insert(result.1.clone(), ModelLoadingState::Loaded { 
                            model_name: result.1.clone() 
                        });
                    }
                    
                    // CRITICAL: Return the loaded engine to the pool!
                    drop(states); // Release the lock first
                    available_pool.lock().await.push_back(engine);
                    info!("Returned engine with model {} to pool", result.1);
                }
                Err(e) => {
                    error!("Failed to load model '{}': {}", model_name_clone, e);
                    states.insert(model_name_clone.clone(), ModelLoadingState::Failed { 
                        model_name: model_name_clone.clone(), 
                        error: e.to_string() 
                    });
                    
                    // Return the unloaded engine to the pool
                    drop(states);
                    available_pool.lock().await.push_back(engine);
                }
            }
        });
        
        Ok(())
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
        let models = self.model_storage.list_models().await?;
        for (_id, metadata) in models {
            // Check various name formats
            if let Some(ref local_path) = metadata.local_path {
                if let Some(dir_name) = local_path.file_name() {
                    let dir_name_str = dir_name.to_string_lossy();
                    
                    // Exact match
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
    
    /// Get model loading status
    pub async fn model_status(&self, model_name: &str) -> ModelLoadingState {
        let states = self.loading_states.read().await;
        states.get(model_name)
            .cloned()
            .unwrap_or(ModelLoadingState::Unloaded)
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
        // Return engine to pool synchronously to ensure it's returned
        if let Some(engine) = self.engine.take() {
            // Use try_lock to avoid blocking in Drop
            // If we can't get the lock immediately, spawn a task
            if let Ok(mut pool) = self.pool.try_lock() {
                pool.push_back(engine);
            } else {
                // Fallback to async if lock is contended
                let pool = self.pool.clone();
                tokio::spawn(async move {
                    if let Err(e) = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        async {
                            pool.lock().await.push_back(engine);
                        }
                    ).await {
                        tracing::error!("Failed to return engine to pool: timeout after 5s: {}", e);
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