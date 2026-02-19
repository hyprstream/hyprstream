//! Server state management

use crate::services::{GenRegistryClient, ModelZmqClient, PolicyClient};
use ed25519_dalek::{SigningKey, VerifyingKey};
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export config types so other server modules can access them via server::state
pub use crate::config::{CorsConfig, SamplingParamDefaults, ServerConfig};

// Re-export context storage types
pub use hyprstream_metrics::storage::context::{ContextRecord, ContextStore, SearchResult};

/// Shared server state
#[derive(Clone)]
pub struct ServerState {
    /// Model client for inference operations via ZMQ
    pub model_client: ModelZmqClient,

    /// Policy client for authorization checks via ZMQ
    pub policy_client: PolicyClient,

    /// Registry client for model operations
    pub registry: GenRegistryClient,

    /// Server configuration (from unified config system)
    pub config: Arc<ServerConfig>,

    /// Metrics collector
    pub metrics: Arc<Metrics>,

    /// Ed25519 signing key for creating JWT tokens
    pub signing_key: Arc<SigningKey>,

    /// Ed25519 verifying key for validating JWT tokens (derived from signing_key)
    pub verifying_key: Arc<VerifyingKey>,

    /// Context store for RAG/CAG (optional)
    pub context_store: Option<Arc<ContextStore<hyprstream_metrics::storage::duckdb::DuckDbBackend>>>,

    /// Cached resource URL for WWW-Authenticate headers (avoids per-request config reload)
    pub resource_url: String,
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
    /// Create a new server state with ZMQ clients
    ///
    /// This is the primary method for creating server state with ZMQ-based services.
    /// The model_client and policy_client should be created after starting their
    /// respective services in main.rs.
    pub async fn new(
        config: ServerConfig,
        model_client: ModelZmqClient,
        policy_client: PolicyClient,
        registry: GenRegistryClient,
        signing_key: SigningKey,
        resource_url: String,
    ) -> Result<Self, anyhow::Error> {
        let verifying_key = signing_key.verifying_key();
        let signing_key = Arc::new(signing_key);
        let verifying_key = Arc::new(verifying_key);

        // Preload models for faster first request
        if !config.preload_models.is_empty() {
            tracing::info!("Preloading {} models", config.preload_models.len());
            for model_name in &config.preload_models {
                tracing::info!("Preloading model: {}", model_name);
                match model_client.load(model_name, None).await {
                    Ok(_) => tracing::info!("Preloaded: {}", model_name),
                    Err(e) => tracing::warn!("Failed to preload model '{}': {}", model_name, e),
                }
            }
        }

        // Initialize metrics
        let metrics = Arc::new(Metrics::default());

        Ok(Self {
            model_client,
            policy_client,
            registry,
            config: Arc::new(config),
            metrics,
            signing_key,
            verifying_key,
            context_store: None, // Initialize via enable_context_store() if needed
            resource_url,
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

        let backend = Arc::new(DuckDbBackend::new(db_path.to_owned(), std::collections::HashMap::new(), None)?);
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
