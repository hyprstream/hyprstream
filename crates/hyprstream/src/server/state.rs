//! Server state management

use super::model_cache::ModelCache;
use crate::{
    api::training_service::TrainingService,
    auth::PolicyManager,
    storage::ModelStorage,
    training::SelfSupervisedTrainer,
};
use dashmap::DashMap;
use ed25519_dalek::{SigningKey, VerifyingKey};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export config types so other server modules can access them via server::state
pub use crate::config::{CorsConfig, SamplingParamDefaults, ServerConfig};

// Re-export context storage types
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

    /// Ed25519 signing key for creating JWT tokens
    pub signing_key: Arc<SigningKey>,

    /// Ed25519 verifying key for validating JWT tokens (derived from signing_key)
    pub verifying_key: Arc<VerifyingKey>,

    /// Context store for RAG/CAG (optional)
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

/// Load or generate Ed25519 signing key from .registry/keys/signing.key
async fn load_or_generate_signing_key(keys_dir: &Path) -> Result<SigningKey, anyhow::Error> {
    let key_path = keys_dir.join("signing.key");

    if key_path.exists() {
        // Load existing key
        let key_bytes = tokio::fs::read(&key_path).await?;
        if key_bytes.len() != 32 {
            anyhow::bail!("Invalid signing key file: expected 32 bytes, got {}", key_bytes.len());
        }
        let mut key_array = [0u8; 32];
        key_array.copy_from_slice(&key_bytes);
        let signing_key = SigningKey::from_bytes(&key_array);
        tracing::info!("Loaded signing key from {:?}", key_path);
        Ok(signing_key)
    } else {
        // Generate new key
        tokio::fs::create_dir_all(keys_dir).await?;
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        tokio::fs::write(&key_path, signing_key.to_bytes()).await?;

        // Set restrictive permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&key_path).await?.permissions();
            perms.set_mode(0o600);
            tokio::fs::set_permissions(&key_path, perms).await?;
        }

        tracing::info!("Generated new signing key at {:?}", key_path);
        Ok(signing_key)
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
        // Security: If policy loading fails, the server must NOT start with permissive defaults.
        // This prevents corrupted/missing policy files from silently disabling authorization.
        let registry_path = model_storage.get_models_dir().join(".registry");
        let policies_dir = registry_path.join("policies");
        tracing::info!("Initializing policy manager from: {:?}", policies_dir);
        let policy_manager = Arc::new(
            PolicyManager::new(&policies_dir)
                .await
                .map_err(|e| anyhow::anyhow!(
                    "Failed to load policies from {:?}: {}. Server cannot start without valid policies.",
                    policies_dir,
                    e
                ))?,
        );

        // Load or generate signing key for JWT tokens
        let keys_dir = registry_path.join("keys");
        let signing_key = load_or_generate_signing_key(&keys_dir).await?;
        let verifying_key = signing_key.verifying_key();
        let signing_key = Arc::new(signing_key);
        let verifying_key = Arc::new(verifying_key);

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

        Ok(Self {
            model_cache,
            model_storage,
            training_service,
            trainers,
            config: Arc::new(config),
            metrics,
            policy_manager,
            signing_key,
            verifying_key,
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
    pub async fn enable_context_store(
        &mut self,
        db_path: &str,
        embedding_dim: i32,
    ) -> Result<(), anyhow::Error> {
        use hyprstream_metrics::storage::context::{context_schema, ContextStore};
        use hyprstream_metrics::storage::duckdb::DuckDbBackend;
        use hyprstream_metrics::StorageBackend;

        let backend = Arc::new(DuckDbBackend::new(db_path.to_string(), std::collections::HashMap::new(), None)?);
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
    pub fn get_context_store(&self) -> Option<&Arc<ContextStore<hyprstream_metrics::storage::duckdb::DuckDbBackend>>> {
        self.context_store.as_ref()
    }
}

// Config types are already imported at the top, no need to re-export here
