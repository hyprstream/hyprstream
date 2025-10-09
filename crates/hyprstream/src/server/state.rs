//! Server state management

use std::sync::Arc;
use tokio::sync::RwLock;
use crate::{
    api::{adapter_storage::AdapterStorage, training_service::TrainingService},
    storage::ModelStorage,
};
use super::model_cache::ModelCache;

// Re-export config types so other server modules can access them via server::state
pub use crate::config::{ServerConfig, CorsConfig, GenerationDefaults};

/// Shared server state
#[derive(Clone)]
pub struct ServerState {
    /// Model cache for UUID-based model caching with LRU eviction
    pub model_cache: Arc<ModelCache>,

    /// Adapter storage manager
    pub adapter_storage: Arc<AdapterStorage>,

    /// Model storage for managing downloaded models
    pub model_storage: Arc<ModelStorage>,

    /// Training service for auto-regressive learning
    pub training_service: Arc<TrainingService>,

    /// Server configuration (from unified config system)
    pub config: Arc<ServerConfig>,

    /// Metrics collector
    pub metrics: Arc<Metrics>,
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
        Self::new_with_git2db(config, git2db::config::Git2DBConfig::default()).await
    }

    /// Create a new server state with custom git2db configuration
    pub async fn new_with_git2db(
        config: ServerConfig,
        git2db_config: git2db::config::Git2DBConfig,
    ) -> Result<Self, anyhow::Error> {
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
        let model_storage = Arc::new(ModelStorage::create_with_config(models_dir.clone(), git2db_config).await?);

        // Initialize adapter storage
        tracing::info!("Initializing adapter storage at: {:?}", models_dir);
        let adapter_storage = Arc::new(AdapterStorage::new(models_dir.clone()).await?);

        // Initialize training service with a runtime engine
        use crate::runtime::{RuntimeEngine, TorchEngine, RuntimeConfig};
        let runtime_engine: Arc<dyn RuntimeEngine> = Arc::new(TorchEngine::new(RuntimeConfig::default())?);
        let training_service = Arc::new(TrainingService::new(runtime_engine));

        // Initialize model cache
        let checkout_base = models_dir.join("checkouts");
        std::fs::create_dir_all(&checkout_base)?;

        // Get the registry from model_storage
        let registry = model_storage.registry();

        let model_cache = Arc::new(ModelCache::new(
            config.max_cached_models,
            registry,
            checkout_base,
        )?);

        // Preload models for faster first request
        if !config.preload_models.is_empty() {
            tracing::info!("Preloading {} models", config.preload_models.len());
            for model_name in &config.preload_models {
                tracing::info!("Preloading model: {}", model_name);
                match model_cache.get_or_load(model_name).await {
                    Ok(_) => tracing::info!("Preloaded: {}", model_name),
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
            tracing::info!("Cache refresher started");
        }

        // Initialize metrics
        let metrics = Arc::new(Metrics::default());

        Ok(Self {
            model_cache,
            adapter_storage,
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

// Config types are already imported at the top, no need to re-export here
