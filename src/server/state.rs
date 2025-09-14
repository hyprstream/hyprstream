//! Server state management

use std::sync::Arc;
use tokio::sync::RwLock;
use crate::{
    api::{
        lora_registry::LoRARegistry,
        model_storage::ModelStorage,
    },
};
use super::model_cache::ModelCache;

/// Training service stub - placeholder until VDB storage integration
pub struct TrainingStub;

impl TrainingStub {
    pub async fn start_auto_training(&self, _lora_id: &str, _config: crate::api::training_service::TrainingConfig) -> Result<(), anyhow::Error> {
        Err(anyhow::anyhow!("Training service not yet implemented"))
    }
    
    pub async fn stop_auto_training(&self, _lora_id: &str) -> Result<(), anyhow::Error> {
        Ok(())
    }
    
    pub async fn get_training_status(&self, _lora_id: &str) -> Result<crate::api::training_service::TrainingStatus, anyhow::Error> {
        Err(anyhow::anyhow!("Training service not yet implemented"))
    }
    
    pub async fn queue_training_sample(&self, _lora_id: &str, _sample: crate::api::TrainingSample) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

/// Shared server state
#[derive(Clone)]
pub struct ServerState {
    /// Model cache for UUID-based model caching with LRU eviction
    pub model_cache: Arc<ModelCache>,
    
    /// Registry of LoRA adapters
    pub lora_registry: Arc<LoRARegistry>,
    
    /// Model storage for managing downloaded models
    pub model_storage: Arc<ModelStorage>,
    
    /// Training service for auto-regressive learning (stub)
    pub training_service: Arc<TrainingStub>,
    
    /// Server configuration
    pub config: Arc<ServerConfig>,
    
    /// Metrics collector
    pub metrics: Arc<Metrics>,
}

/// CORS configuration
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Enable CORS middleware
    pub enabled: bool,
    /// Allowed origins (use ["*"] for all origins - NOT recommended for production)
    pub allowed_origins: Vec<String>,
    /// Allow credentials in CORS requests
    pub allow_credentials: bool,
    /// Max age for preflight cache (in seconds)
    pub max_age: u64,
    /// Allow all headers (permissive mode for development - NOT recommended for production)
    pub permissive_headers: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Default to localhost origins for development
            allowed_origins: vec![
                "http://localhost:3000".to_string(),
                "http://localhost:3001".to_string(),
                "http://127.0.0.1:3000".to_string(),
                "http://127.0.0.1:3001".to_string(),
            ],
            allow_credentials: true,
            max_age: 3600,
            permissive_headers: false, // Secure by default
        }
    }
}

impl CorsConfig {
    /// Create CORS config with dynamic port-based origins
    pub fn with_port(port: u16) -> Self {
        let mut config = Self::default();
        // Add origins for the actual server port as well
        config.allowed_origins.extend(vec![
            format!("http://localhost:{}", port),
            format!("http://127.0.0.1:{}", port),
        ]);
        config
    }
}

/// Default generation parameters
#[derive(Debug, Clone)]
pub struct GenerationDefaults {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub stream_timeout_secs: u64,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 1.0,
            top_p: 1.0,
            repeat_penalty: 1.1,
            stream_timeout_secs: 300,
        }
    }
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Maximum number of models to cache in memory
    pub max_cached_models: usize,
    
    /// Models to preload at startup for faster first request
    pub preload_models: Vec<String>,
    
    /// Enable request logging
    pub enable_logging: bool,
    
    /// Enable metrics collection
    pub enable_metrics: bool,
    
    /// API key for authentication (optional)
    pub api_key: Option<String>,
    
    /// Maximum tokens per request
    pub max_tokens_limit: usize,
    
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    
    /// CORS configuration
    pub cors: CorsConfig,
    
    /// Default generation parameters
    pub generation_defaults: GenerationDefaults,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_cached_models: 5,
            preload_models: Vec::new(),
            enable_logging: true,
            enable_metrics: true,
            api_key: None,
            max_tokens_limit: 4096,
            request_timeout_secs: 300,
            cors: CorsConfig::default(),
            generation_defaults: GenerationDefaults::default(),
        }
    }
}

impl ServerConfig {
    /// Create from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        // Get port early to configure CORS properly
        let port = std::env::var("HYPRSTREAM_SERVER_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(50051);
        
        // Use port-aware CORS config
        config.cors = CorsConfig::with_port(port);
        
        if let Ok(max_cached) = std::env::var("HYPRSTREAM_MAX_CACHED_MODELS") {
            if let Ok(n) = max_cached.parse() {
                config.max_cached_models = n;
            }
        }
        
        // Parse comma-separated list of models to preload
        if let Ok(models) = std::env::var("HYPRSTREAM_PRELOAD_MODELS") {
            config.preload_models = models
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        
        if let Ok(api_key) = std::env::var("HYPRSTREAM_API_KEY") {
            config.api_key = Some(api_key);
        }
        
        // CORS configuration from environment
        if let Ok(cors_enabled) = std::env::var("HYPRSTREAM_CORS_ENABLED") {
            config.cors.enabled = cors_enabled.to_lowercase() != "false";
        }
        
        if let Ok(cors_origins) = std::env::var("HYPRSTREAM_CORS_ORIGINS") {
            // Parse comma-separated list of origins
            config.cors.allowed_origins = cors_origins
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            
            // If wildcard is specified, automatically disable credentials (CORS spec requirement)
            if config.cors.allowed_origins.contains(&"*".to_string()) {
                config.cors.allow_credentials = false;
            }
        }
        
        if let Ok(cors_credentials) = std::env::var("HYPRSTREAM_CORS_CREDENTIALS") {
            // Only allow setting credentials to true if not using wildcard
            if !config.cors.allowed_origins.contains(&"*".to_string()) {
                config.cors.allow_credentials = cors_credentials.to_lowercase() == "true";
            }
        }
        
        if let Ok(permissive) = std::env::var("HYPRSTREAM_CORS_PERMISSIVE_HEADERS") {
            config.cors.permissive_headers = permissive.to_lowercase() == "true";
            if config.cors.permissive_headers {
                eprintln!("⚠️  WARNING: CORS permissive headers mode enabled - not recommended for production");
            }
        }
        
        config
    }
}

/// Metrics collector
pub struct Metrics {
    /// Total requests processed
    pub total_requests: Arc<std::sync::atomic::AtomicU64>,
    
    /// Total tokens generated
    pub total_tokens: Arc<std::sync::atomic::AtomicU64>,
    
    /// Average latency in milliseconds
    pub avg_latency_ms: Arc<RwLock<f64>>,
    
    /// Active requests
    pub active_requests: Arc<std::sync::atomic::AtomicU32>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            total_requests: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            total_tokens: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            avg_latency_ms: Arc::new(RwLock::new(0.0)),
            active_requests: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }
}

impl ServerState {
    /// Create a new server state
    pub async fn new(config: ServerConfig) -> Result<Self, anyhow::Error> {
        // Use proper storage paths via StoragePaths (XDG Base Directory spec)
        let storage_paths = crate::storage::paths::StoragePaths::new()?;
        
        // Allow environment override but use proper XDG paths by default
        let models_dir = if let Ok(dir) = std::env::var("HYPRSTREAM_MODELS_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            storage_paths.models_dir()?
        };
        
        let loras_dir = if let Ok(dir) = std::env::var("HYPRSTREAM_LORA_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            storage_paths.loras_dir()?
        };
        
        tracing::info!("Initializing model storage at: {:?}", models_dir);
        let model_storage = Arc::new(ModelStorage::new(models_dir).await?);
        
        tracing::info!("Initializing LoRA registry at: {:?}", loras_dir);
        let lora_registry = Arc::new(LoRARegistry::new(loras_dir).await?);
        
        let training_service = Arc::new(TrainingStub);
        
        // Initialize model cache
        let model_cache = Arc::new(ModelCache::new(
            config.max_cached_models,
            Arc::clone(&model_storage),
        ));
        
        // Pre-populate name cache to avoid disk scanning on first requests
        if let Err(e) = model_cache.warm_name_cache().await {
            tracing::warn!("Failed to warm name cache: {}. First requests may be slower.", e);
        }
        
        // Preload models for faster first request
        if !config.preload_models.is_empty() {
            tracing::info!("Preloading {} models into cache...", config.preload_models.len());
            for model_name in &config.preload_models {
                tracing::info!("Preloading model: {}", model_name);
                match model_cache.get_or_load_by_name(model_name).await {
                    Ok(_) => tracing::info!("Successfully preloaded model: {}", model_name),
                    Err(e) => tracing::warn!("Failed to preload model '{}': {}", model_name, e),
                }
            }
        }
        
        // Start cache refresher if enabled
        let refresher_config = super::cache_refresher::CacheRefresherConfig::from_env();
        if refresher_config.enabled {
            let refresher = super::cache_refresher::CacheRefresher::new(
                Arc::clone(&model_cache),
                refresher_config.interval_secs,
            );
            let _handle = refresher.start();
            tracing::info!("Started background cache refresher");
        }
        
        // Initialize metrics
        let metrics = Arc::new(Metrics::default());
        
        Ok(Self {
            model_cache,
            lora_registry,
            model_storage,
            training_service,
            config: Arc::new(config),
            metrics,
        })
    }
    
    /// Get server metrics
    pub async fn get_metrics(&self) -> serde_json::Value {
        serde_json::json!({
            "total_requests": self.metrics.total_requests.load(std::sync::atomic::Ordering::Relaxed),
            "total_tokens": self.metrics.total_tokens.load(std::sync::atomic::Ordering::Relaxed),
            "avg_latency_ms": *self.metrics.avg_latency_ms.read().await,
            "active_requests": self.metrics.active_requests.load(std::sync::atomic::Ordering::Relaxed),
        })
    }
}