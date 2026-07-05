//! Server state management

use crate::services::{RegistryClient, PolicyClient};
use crate::services::generated::model_client::{ModelClient, LoadModelRequest};
use ed25519_dalek::{SigningKey, VerifyingKey};
use hyprstream_util::TtlCache;
use std::collections::HashMap;
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
    pub model_client: ModelClient,

    /// Policy client for authorization checks via ZMQ
    pub policy_client: PolicyClient,

    /// Registry client for model operations
    pub registry: RegistryClient,

    /// Server configuration (from unified config system)
    pub config: Arc<ServerConfig>,

    /// Metrics collector
    pub metrics: Arc<Metrics>,

    /// Ed25519 signing key for creating JWT tokens
    pub signing_key: Arc<SigningKey>,

    /// Ed25519 verifying key for validating JWT tokens (derived from signing_key).
    ///
    /// This is the cluster CA key. Locally-issued at+JWTs may instead be signed by
    /// a rotation slot (see `published_jwt_verifying_keys`), so token validation
    /// tries this key AND the published rotation keys.
    pub verifying_key: Arc<VerifyingKey>,

    /// Live, shared handle to the node's published Ed25519 rotation-slot verifying
    /// keys — the SAME set the `/oauth/jwks` endpoint publishes. Locally-issued
    /// tokens signed by the OAuth rotation active/lead/drain slots (WIT, S6 grant /
    /// token-exchange, WITs) verify against these in addition to `verifying_key`.
    /// Only node-published public keys are ever admitted here (never token-supplied
    /// keys). Populated + kept current by the key-rotation task.
    pub published_jwt_verifying_keys: crate::auth::key_rotation::PublishedEd25519Keys,

    /// Context store for RAG/CAG (optional)
    pub context_store: Option<Arc<ContextStore<hyprstream_metrics::storage::duckdb::DuckDbBackend>>>,

    /// Cached resource URL for WWW-Authenticate headers (avoids per-request config reload)
    pub resource_url: String,

    /// OAuth issuer URL for local tokens (matches the `iss` claim in locally-issued JWTs).
    pub oauth_issuer_url: String,

    /// Federation key resolver for multi-issuer JWT verification.
    /// Contains trusted issuers (empty if none configured).
    pub federation_resolver: Arc<crate::auth::FederationKeyResolver>,

    /// Shared JTI blocklist for access token revocation (RFC 7009).
    /// Populated by the OAuth revocation endpoint; checked on every request.
    pub jti_blocklist: Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>,

    /// Per-request DPoP JTI dedup cache (RFC 9449 §11.1 replay prevention).
    /// Backed by the shared `TtlCache` with atomic check-and-record
    /// (`insert_if_absent`); TTL = iat + 120s; self-evicting.
    pub dpop_jti_seen: Arc<TtlCache<String, ()>>,

    /// Per-subject request rate limiter (fixed window, 300 req/60s default).
    pub rate_limiter: Arc<crate::server::middleware::RateLimiter>,
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
        model_client: ModelClient,
        policy_client: PolicyClient,
        registry: RegistryClient,
        signing_key: SigningKey,
        jwt_verifying_key: VerifyingKey,
        resource_url: String,
        oauth_issuer_url: String,
        trusted_issuers: &HashMap<String, crate::config::TrustedIssuerConfig>,
        jti_blocklist: Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>,
    ) -> Result<Self, anyhow::Error> {
        let signing_key = Arc::new(signing_key);
        // Use the CA JWT verifying key (not the service's own key) so HTTP Bearer tokens
        // issued by PolicyService can be verified correctly.
        let verifying_key = Arc::new(jwt_verifying_key);

        // Live handle to the node's published Ed25519 rotation-slot verifying keys.
        // Shared with the OAuth key-rotation task so tokens signed by the active
        // rotation slot (not the CA key) still validate here.
        let published_jwt_verifying_keys =
            crate::auth::key_rotation::global_ed25519_verifying_keys();

        // Preload models for faster first request
        if !config.preload_models.is_empty() {
            tracing::info!("Preloading {} models", config.preload_models.len());
            for model_name in &config.preload_models {
                tracing::info!("Preloading model: {}", model_name);
                match model_client.load(&LoadModelRequest {
                    model_ref: model_name.to_owned(),
                    max_context: None,
                    kv_quant: None,
                }).await {
                    Ok(_) => tracing::info!("Preloaded: {}", model_name),
                    Err(e) => tracing::warn!("Failed to preload model '{}': {}", model_name, e),
                }
            }
        }

        // Initialize metrics
        let metrics = Arc::new(Metrics::default());

        // Wire the unified federation:register PolicyService gate. Same
        // gate as CIMD client registration — single atproto-style
        // trust decision applies to both clients and peers.
        let federation_resolver = Arc::new(
            crate::auth::FederationKeyResolver::new(trusted_issuers)
                .with_policy_client(Arc::new(policy_client.clone()))
        );

        Ok(Self {
            model_client,
            policy_client,
            registry,
            config: Arc::new(config),
            metrics,
            signing_key,
            verifying_key,
            published_jwt_verifying_keys,
            context_store: None, // Initialize via enable_context_store() if needed
            resource_url,
            oauth_issuer_url,
            federation_resolver,
            jti_blocklist,
            dpop_jti_seen: Arc::new(TtlCache::new(10_000, 64)),
            rate_limiter: Arc::new(crate::server::middleware::RateLimiter::new(300, 60)),
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
