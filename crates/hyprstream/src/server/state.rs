//! Server state management

use super::model_cache::ModelCache;
use crate::{
    api::training_service::TrainingService,
    auth::{PolicyManager, TokenManager},
    events::{EventBus, SinkRegistry, SinksConfig},
    storage::ModelStorage,
    training::SelfSupervisedTrainer,
};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export config types so other server modules can access them via server::state
pub use crate::config::{CorsConfig, SamplingParamDefaults, ServerConfig};

// Re-export context storage types when metrics feature is enabled
#[cfg(feature = "metrics")]
pub use hyprstream_metrics::storage::context::{ContextRecord, ContextStore, SearchResult};

/// Shared server state
#[derive(Clone)]
pub struct ServerState {
    /// Model cache for UUID-based model caching with LRU eviction
    pub model_cache: Arc<ModelCache>,

    /// Model storage for managing downloaded models
    pub model_storage: Arc<ModelStorage>,

    /// Training service for auto-regressive learning
    pub training_service: Arc<TrainingService>,

    /// Self-supervised trainers per model (model_ref -> trainer)
    /// Each model with training enabled gets its own trainer instance
    pub trainers: Arc<DashMap<String, Arc<SelfSupervisedTrainer>>>,

    /// Server configuration (from unified config system)
    pub config: Arc<ServerConfig>,

    /// Metrics collector
    pub metrics: Arc<Metrics>,

    /// Policy manager for RBAC/ABAC access control
    pub policy_manager: Arc<PolicyManager>,

    /// Token manager for API key authentication
    pub token_manager: Arc<RwLock<TokenManager>>,

    /// Event bus for pub/sub messaging
    pub event_bus: Arc<EventBus>,

    /// Sink registry for managing event consumers
    pub sink_registry: Arc<SinkRegistry>,

    /// Context store for RAG/CAG (optional, requires metrics feature)
    #[cfg(feature = "metrics")]
    pub context_store: Option<Arc<ContextStore<hyprstream_metrics::storage::duckdb::DuckDbBackend>>>,
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

    /// Create a new server state with a shared registry client
    ///
    /// This is the preferred method when a shared registry client is available.
    /// The client should be obtained from `RegistryZmqClient::new()` in main.rs.
    pub async fn new_with_client(
        config: ServerConfig,
        client: Arc<dyn crate::services::RegistryClient>,
    ) -> Result<Self, anyhow::Error> {
        // Use proper storage paths via StoragePaths (XDG Base Directory spec)
        let storage_paths = crate::storage::paths::StoragePaths::new()?;

        // Allow environment override but use proper XDG paths by default
        let models_dir = if let Ok(dir) = std::env::var("HYPRSTREAM_MODELS_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            storage_paths.models_dir()?
        };

        tracing::info!("Initializing model storage at: {:?}", models_dir);
        let model_storage = Arc::new(ModelStorage::new(client, models_dir));

        Self::create_with_storage(config, model_storage).await
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

        let _loras_dir = if let Ok(dir) = std::env::var("HYPRSTREAM_LORA_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            storage_paths.loras_dir()?
        };

        tracing::info!("Initializing model storage at: {:?}", models_dir);
        let model_storage =
            Arc::new(ModelStorage::create_with_config(models_dir.clone(), git2db_config).await?);

        Self::create_with_storage(config, model_storage).await
    }

    /// Internal helper to create server state with pre-initialized storage
    async fn create_with_storage(
        config: ServerConfig,
        model_storage: Arc<ModelStorage>,
    ) -> Result<Self, anyhow::Error> {
        // Initialize policy manager from .registry/policies/
        let registry_path = model_storage.get_models_dir().join(".registry");
        let policies_dir = registry_path.join("policies");
        tracing::info!("Initializing policy manager from: {:?}", policies_dir);
        let policy_manager = Arc::new(
            PolicyManager::new(&policies_dir)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        "Failed to load policies from {:?}: {}. Using permissive defaults.",
                        policies_dir,
                        e
                    );
                    // Fall back to permissive policy manager synchronously
                    // This is safe since we're already in an async context
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current()
                            .block_on(PolicyManager::permissive())
                            .expect("Failed to create permissive policy manager")
                    })
                }),
        );

        // Initialize token manager from .registry/policies/tokens.csv
        let tokens_path = policies_dir.join("tokens.csv");
        tracing::info!("Initializing token manager from: {:?}", tokens_path);
        let token_manager = Arc::new(RwLock::new(
            TokenManager::new(&tokens_path)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        "Failed to load tokens from {:?}: {}. Starting with empty token store.",
                        tokens_path,
                        e
                    );
                    TokenManager::in_memory()
                }),
        ));

        // Initialize training service
        let training_service = Arc::new(TrainingService::new());

        // Initialize model cache using git2db's worktree management
        // Pass max_context to limit KV cache allocation and reduce GPU memory
        // Pass kv_quant for KV cache quantization (reduces memory by 50-75%)
        let model_cache = Arc::new(ModelCache::new(
            config.max_cached_models,
            model_storage.clone(),
            config.max_context,
            config.kv_quant,
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

        // Initialize metrics
        let metrics = Arc::new(Metrics::default());

        // Initialize self-supervised trainers map (trainers are created lazily per model)
        let trainers = Arc::new(DashMap::new());

        // Initialize event bus for pub/sub messaging
        let event_bus = Arc::new(EventBus::new().await.map_err(|e| {
            anyhow::anyhow!("failed to initialize event bus: {}", e)
        })?);
        tracing::info!("Event bus initialized with endpoints: {:?}", event_bus.endpoints());

        // Initialize sink registry
        let sink_registry = Arc::new(SinkRegistry::new(event_bus.clone()));

        // Load sink configuration from .registry/event_sinks.yaml
        let sinks_config_path = registry_path.join("event_sinks.yaml");
        if sinks_config_path.exists() {
            match SinksConfig::load(&sinks_config_path) {
                Ok(sinks_config) => {
                    tracing::info!(
                        "Loading {} event sinks from {:?}",
                        sinks_config.sinks.len(),
                        sinks_config_path
                    );
                    for sink_config in sinks_config.sinks {
                        if let Err(e) = sink_registry.register(sink_config.clone()).await {
                            tracing::warn!(
                                "Failed to register sink '{}': {}",
                                sink_config.name,
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to load sinks config: {}. Continuing without event sinks.", e);
                }
            }
        }

        Ok(Self {
            model_cache,
            model_storage,
            training_service,
            trainers,
            config: Arc::new(config),
            metrics,
            policy_manager,
            token_manager,
            event_bus,
            sink_registry,
            #[cfg(feature = "metrics")]
            context_store: None, // Initialize via enable_context_store() if needed
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

    /// Enable context storage for RAG/CAG functionality.
    ///
    /// This initializes a context store using DuckDB for embedding storage.
    /// Call this during server initialization if RAG/CAG is needed.
    ///
    /// # Arguments
    /// - `db_path` - Path to DuckDB database (use ":memory:" for in-memory)
    /// - `embedding_dim` - Dimension of embeddings (e.g., 768 for many models)
    ///
    /// # Returns
    /// A mutable reference to the initialized ContextStore
    #[cfg(feature = "metrics")]
    pub async fn enable_context_store(
        &mut self,
        db_path: &str,
        embedding_dim: i32,
    ) -> Result<(), anyhow::Error> {
        use hyprstream_metrics::storage::context::{context_schema, ContextStore};
        use hyprstream_metrics::storage::duckdb::DuckDbBackend;
        use hyprstream_metrics::StorageBackend;

        let backend = Arc::new(DuckDbBackend::new(db_path)?);
        backend.init().await.map_err(|e| anyhow::anyhow!("Failed to init DuckDB: {}", e))?;
        backend.create_table("context", &context_schema(embedding_dim)).await
            .map_err(|e| anyhow::anyhow!("Failed to create context table: {}", e))?;

        let store = ContextStore::new(backend, "context", embedding_dim);
        self.context_store = Some(Arc::new(store));

        tracing::info!(
            "Context store enabled with embedding_dim={} at {}",
            embedding_dim,
            db_path
        );

        Ok(())
    }

    /// Get the context store if initialized
    #[cfg(feature = "metrics")]
    pub fn get_context_store(&self) -> Option<&Arc<ContextStore<hyprstream_metrics::storage::duckdb::DuckDbBackend>>> {
        self.context_store.as_ref()
    }
}

// Config types are already imported at the top, no need to re-export here
