//! Model service for managing InferenceService instances over ZMQ
//!
//! This service manages the lifecycle of InferenceService instances.
//! It handles model loading, unloading, and routes inference requests
//! to the appropriate InferenceService based on model reference.
//!
//! # Architecture
//!
//! ```text
//! REST API / CLI
//!       │
//!       │ ModelClient (async ZMQ I/O)
//!       ▼
//! ModelService (multi-threaded runtime)
//!       │
//!       ├── LRU cache of loaded models
//!       ├── Spawns InferenceService per model
//!       └── Routes requests to InferenceService
//!             │
//!             │ InferenceClient (async ZMQ I/O)
//!             ▼
//!       InferenceService (dedicated thread per model)
//! ```
//!
//! # Endpoint
//!
//! Uses `registry().endpoint("model", SocketKind::Rep)` for the REP endpoint.
//! Default fallback: `inproc://hyprstream/model`

use async_trait::async_trait;
// GenerationRequest import removed — was only used by deleted ModelZmqClient
use crate::runtime::KVQuantType;
use crate::runtime::RuntimeConfig;
use crate::services::{
    EnvelopeContext,
    NotificationClient, NotificationPublisher, PolicyClient,
};
use crate::services::generated::inference_client::InferenceClient;
use crate::services::RegistryClient;
use crate::services::generated::registry_client::{StageFilesRequest, CommitWithAuthorRequest};
use crate::services::generated::policy_client::PolicyCheck;
use hyprstream_rpc::envelope::RequestIdentity;
use crate::storage::ModelRef;
use anyhow::{anyhow, Result};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::transport::TransportConfig;
use lru::LruCache;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// Default endpoint for the model service
pub const MODEL_ENDPOINT: &str = "inproc://hyprstream/model";

// ============================================================================
// ModelService (server-side)
// ============================================================================


/// Information about a loaded model
pub struct LoadedModel {
    /// Model reference string (e.g., "qwen3-small:main")
    pub model_ref: String,
    /// ZMQ endpoint for this model's InferenceService
    pub endpoint: String,
    /// Handle to stop the InferenceService
    pub service_handle: hyprstream_service::SpawnedService,
    /// Client for communicating with the InferenceService
    pub client: InferenceClient,
    /// When the model was loaded
    pub loaded_at: Instant,
    /// When the model was last used
    pub last_used: Instant,
    /// Online training (TTT) configuration (if enabled)
    pub ttt_config: Option<crate::training::ttt::TTTConfig>,
    /// Generation parameter defaults from model's generation_config.json
    pub generation_defaults: crate::config::SamplingParams,
}

/// How InferenceService instances are spawned
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpawnMode {
    /// Run InferenceService in-process (current behavior)
    #[default]
    InProcess,
    /// Spawn InferenceService as separate process via callback pattern
    Spawned,
}

/// Model service configuration
pub struct ModelServiceConfig {
    /// Maximum number of models to keep loaded
    pub max_models: usize,
    /// Maximum context length for KV cache allocation
    pub max_context: Option<u32>,
    /// KV cache quantization type
    pub kv_quant: KVQuantType,
    /// How to spawn InferenceService instances
    pub spawn_mode: SpawnMode,
    /// Callback endpoint for spawned mode
    pub callback_endpoint: Option<String>,
}

impl Default for ModelServiceConfig {
    fn default() -> Self {
        Self {
            max_models: 5,
            max_context: None,
            kv_quant: KVQuantType::None,
            spawn_mode: SpawnMode::InProcess,
            callback_endpoint: None,
        }
    }
}

/// Inner state for ModelService, behind Arc for continuation capture.
pub struct ModelServiceInner {
    // Business logic
    /// LRU cache of loaded models
    loaded_models: RwLock<LruCache<String, LoadedModel>>,
    /// Models currently being loaded (accepted but not yet in LRU cache)
    pending_loads: Mutex<HashSet<String>>,
    /// Service configuration
    config: ModelServiceConfig,
    /// Ed25519 signing key for creating InferenceClients
    signing_key: SigningKey,
    /// Policy client for authorization checks in InferenceService
    policy_client: PolicyClient,
    /// Notification publisher for model lifecycle events
    notification_publisher: NotificationPublisher,
    /// Registry client for resolving model paths
    registry: RegistryClient,
    /// Callback router for spawned mode (None for in-process)
    #[allow(dead_code)]
    callback_router: Option<crate::services::callback::CallbackRouter>,
    /// Spawned instances by model ref (for spawned mode)
    #[allow(dead_code)]
    spawned_instances: RwLock<HashMap<String, crate::services::callback::Instance>>,
    // Infrastructure (for Spawnable)
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    /// Expected JWT audience for token validation (RFC 8707).
    expected_audience: Option<String>,
    /// Local OAuth issuer URL for distinguishing local vs. federated JWTs.
    local_issuer_url: Option<String>,
    /// Federation key source for verifying externally-issued JWTs.
    federation_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>>,
    /// Persistent 9P synthetic tree for the fs scope (lazily initialized).
    fs_tree: std::sync::OnceLock<crate::services::fs::SyntheticTree>,
}

/// Model service that manages InferenceService lifecycle.
///
/// Wraps `ModelServiceInner` in `Arc` so continuations can capture a cheap
/// clone. All field access is transparent via `Deref`.
///
/// Load requests are handled asynchronously: the request loop returns an
/// immediate "accepted" response and spawns the actual model loading as a
/// `Continuation` (via `spawn_local`), keeping the service responsive for
/// list, health, info, and other requests during long GPU weight transfers.
pub struct ModelService {
    inner: Arc<ModelServiceInner>,
}

impl Clone for ModelService {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl std::ops::Deref for ModelService {
    type Target = ModelServiceInner;
    fn deref(&self) -> &Self::Target { &self.inner }
}

impl ModelService {
    /// Create a new model service with infrastructure
    pub fn new(
        config: ModelServiceConfig,
        signing_key: SigningKey,
        policy_client: PolicyClient,
        registry: RegistryClient,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        // SAFETY: 5 is a valid non-zero value
        const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(5) {
            Some(n) => n,
            None => unreachable!(),
        };
        let cache_size = NonZeroUsize::new(config.max_models).unwrap_or(DEFAULT_CACHE_SIZE);

        let notif_client = NotificationClient::new(signing_key.clone(), RequestIdentity::anonymous());
        let notification_publisher = NotificationPublisher::new(notif_client, signing_key.clone());

        Self { inner: Arc::new(ModelServiceInner {
            loaded_models: RwLock::new(LruCache::new(cache_size)),
            pending_loads: Mutex::new(HashSet::new()),
            config,
            signing_key,
            policy_client,
            notification_publisher,
            registry,
            callback_router: None,
            spawned_instances: RwLock::new(HashMap::new()),
            context,
            transport,
            expected_audience: None,
            local_issuer_url: None,
            federation_key_source: None,
            fs_tree: std::sync::OnceLock::new(),
        })}
    }

    /// Set the expected JWT audience for token validation.
    ///
    /// # Panics
    /// Panics if called after the service has been cloned (Arc refcount > 1).
    /// Must be called during construction, before the service is shared.
    #[allow(clippy::expect_used)]
    pub fn with_expected_audience(mut self, audience: String) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_expected_audience must be called before service is shared")
            .expected_audience = Some(audience);
        self
    }

    /// Set the local OAuth issuer URL for distinguishing local vs. federated JWTs.
    ///
    /// # Panics
    /// Panics if called after the service has been cloned (Arc refcount > 1).
    #[allow(clippy::expect_used)]
    pub fn with_local_issuer_url(mut self, url: String) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_local_issuer_url must be called before service is shared")
            .local_issuer_url = Some(url);
        self
    }

    /// Set the federation key source for verifying externally-issued JWTs.
    ///
    /// # Panics
    /// Panics if called after the service has been cloned (Arc refcount > 1).
    #[allow(clippy::expect_used)]
    pub fn with_federation_key_source(
        mut self,
        src: std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>,
    ) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_federation_key_source must be called before service is shared")
            .federation_key_source = Some(src);
        self
    }

    /// Create a model service with callback router for spawned mode
    pub fn with_callback_router(
        config: ModelServiceConfig,
        signing_key: SigningKey,
        policy_client: PolicyClient,
        registry: RegistryClient,
        callback_router: crate::services::callback::CallbackRouter,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(5) {
            Some(n) => n,
            None => unreachable!(),
        };
        let cache_size = NonZeroUsize::new(config.max_models).unwrap_or(DEFAULT_CACHE_SIZE);

        let notif_client = NotificationClient::new(signing_key.clone(), RequestIdentity::anonymous());
        let notification_publisher = NotificationPublisher::new(notif_client, signing_key.clone());

        Self { inner: Arc::new(ModelServiceInner {
            loaded_models: RwLock::new(LruCache::new(cache_size)),
            pending_loads: Mutex::new(HashSet::new()),
            config,
            signing_key,
            policy_client,
            notification_publisher,
            registry,
            callback_router: Some(callback_router),
            spawned_instances: RwLock::new(HashMap::new()),
            context,
            transport,
            expected_audience: None,
            local_issuer_url: None,
            federation_key_source: None,
            fs_tree: std::sync::OnceLock::new(),
        })}
    }

    /// Derive the deterministic IPC endpoint for a model reference.
    fn inference_endpoint(model_ref_str: &str) -> String {
        let safe_name = model_ref_str.replace([':', '/', '\\'], "-");
        let service_name = format!("inference-{safe_name}");
        registry().endpoint(&service_name, SocketKind::Rep).to_zmq_string()
    }

    /// Load a model by reference with optional per-model config, returns the inference endpoint
    async fn load_model(&self, model_ref_str: &str, max_context: Option<u32>, kv_quant: Option<KVQuantType>) -> Result<String> {
        // Check if already loaded
        {
            let mut cache = self.loaded_models.write().await;
            if let Some(model) = cache.get_mut(model_ref_str) {
                model.last_used = Instant::now();
                debug!("Model {} already loaded at {}", model_ref_str, model.endpoint);
                return Ok(model.endpoint.clone());
            }
        }

        // Atomically check-and-insert into pending_loads (prevents duplicate GPU loads
        // when multiple requests arrive during the ~40s load window).
        // HashSet::insert returns false if the value was already present.
        {
            let mut pending = self.pending_loads.lock().await;
            if !pending.insert(model_ref_str.to_owned()) {
                anyhow::bail!(
                    "Model {} is already being loaded — please retry shortly",
                    model_ref_str
                );
            }
        }

        let result = self.load_model_inner(model_ref_str, max_context, kv_quant).await;

        // Always remove from pending, whether load succeeded or failed
        self.pending_loads.lock().await.remove(model_ref_str);

        if result.is_ok() {
            info!("Model {} loaded successfully", model_ref_str);
        }
        result
    }

    /// Inner model loading logic, called by load_model() which manages pending_loads.
    async fn load_model_inner(&self, model_ref_str: &str, max_context: Option<u32>, kv_quant: Option<KVQuantType>) -> Result<String> {
        // Parse model reference
        let model_ref = ModelRef::parse(model_ref_str)?;

        // Get model path from registry
        let tracked = self.registry.get_by_name(&model_ref.model).await
            .map_err(|e| anyhow!("Model '{}' not found in registry: {}", model_ref.model, e))?;
        let repo_client = self.registry.repo(&tracked.id);

        let branch_name = match &model_ref.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            _ => repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned()),
        };
        let worktrees = repo_client.list_worktrees().await?;
        if !worktrees.iter().any(|wt| wt.branch_name == branch_name) {
            return Err(anyhow!("worktree for {}:{} not found", model_ref.model, branch_name));
        }
        // Derive worktree path locally
        let storage_paths = crate::storage::StoragePaths::new()?;
        let model_path = storage_paths.worktree_path(&model_ref.model, &branch_name)?;

        if !model_path.exists() {
            return Err(anyhow!(
                "Model worktree not found for {}. Please clone the model first.",
                model_ref_str
            ));
        }

        let endpoint = Self::inference_endpoint(model_ref_str);

        info!("Loading model {} at endpoint {}", model_ref_str, endpoint);

        // Create runtime config - use per-model config if provided, otherwise service defaults
        let runtime_config = RuntimeConfig {
            max_context: max_context.or(self.config.max_context),
            kv_quant_type: kv_quant.unwrap_or(self.config.kv_quant),
            ..Default::default()
        };

        // Obtain FsOps from the registry for path-contained adapter I/O
        let fs: Option<crate::services::WorktreeClient> = Some(repo_client.worktree(&branch_name));

        // Start InferenceService for this model via standard Spawnable infrastructure
        let zmq_ctx = Arc::clone(hyprstream_rpc::ZmqService::context(self));
        let spawner = hyprstream_service::ServiceSpawner::threaded();

        let transport = hyprstream_rpc::transport::TransportConfig::from_endpoint(&endpoint);
        let mut service_config = crate::services::InferenceServiceConfig::new(
            &model_path,
            runtime_config,
            self.signing_key.verifying_key(),
            self.signing_key.clone(),
            zmq_ctx,
            transport,
            fs,
        );
        if let Some(ref aud) = self.expected_audience {
            service_config = service_config.with_expected_audience(aud.clone());
        }
        if let Some(ref url) = self.local_issuer_url {
            service_config = service_config.with_local_issuer_url(url.clone());
        }
        if let Some(ref fed) = self.federation_key_source {
            service_config = service_config.with_federation_key_source(fed.clone());
        }
        let service_handle = spawner.spawn(service_config).await
            .map_err(|e| anyhow!("Failed to spawn inference service: {}", e))?;

        // Create client for this service
        let client = InferenceClient::with_endpoint(
            &endpoint,
            self.signing_key.clone(),
            RequestIdentity::anonymous(),
        );

        // Load TTT config from model's config.json (if TTT is enabled)
        let ttt_config = crate::runtime::model_config::ModelConfig::load_training_config(&model_path)
            .and_then(|tc| {
                if tc.is_enabled() && tc.mode == crate::config::TrainingMode::TestTimeTraining {
                    Some(crate::training::ttt::TTTConfig {
                        learning_rate: tc.ttt.learning_rate,
                        gradient_steps: tc.ttt.gradient_steps,
                        max_grad_norm: tc.ttt.max_grad_norm,
                        min_input_length: tc.ttt.min_input_length,
                        max_ttt_context: tc.ttt.max_ttt_context,
                        enabled: true,
                        ..crate::training::ttt::TTTConfig::default()
                    })
                } else {
                    None
                }
            });

        // Load generation parameter defaults from model's generation_config.json
        let generation_defaults = crate::config::SamplingParams::from_model_path(&model_path)
            .await
            .unwrap_or_default();

        // Check if we need to evict
        {
            let mut cache = self.loaded_models.write().await;
            if cache.len() >= self.config.max_models {
                if let Some((evicted_ref, mut evicted)) = cache.pop_lru() {
                    info!("Evicting model {} to load {}", evicted_ref, model_ref_str);
                    // Stop the evicted service in background (fire-and-forget)
                    #[allow(clippy::let_underscore_future)]
                    let _ = tokio::spawn(async move {
                        let _ = evicted.service_handle.stop().await;
                    });
                }
            }

            // Add to cache
            cache.put(
                model_ref_str.to_owned(),
                LoadedModel {
                    model_ref: model_ref_str.to_owned(),
                    endpoint: endpoint.clone(),
                    service_handle,
                    client,
                    loaded_at: Instant::now(),
                    last_used: Instant::now(),
                    ttt_config,
                    generation_defaults,
                },
            );
        }

        Ok(endpoint)
    }

    /// Unload a model
    async fn unload_model(&self, model_ref_str: &str) -> Result<()> {
        let mut cache = self.loaded_models.write().await;
        if let Some((_, mut model)) = cache.pop_entry(model_ref_str) {
            info!("Unloading model {}", model_ref_str);
            let _ = model.service_handle.stop().await;
            let model_name = model_ref_str.split(':').next().unwrap_or(model_ref_str);
            let scope = format!("serve:model:{}", model_name);
            let event = crate::events::EventEnvelope::new(
                crate::events::EventSource::Model,
                scope.clone(),
                crate::events::EventPayload::ModelUnloaded {
                    model_ref: model_ref_str.to_owned(),
                },
            );
            if let Ok(payload) = serde_json::to_vec(&event) {
                let _ = self.notification_publisher.publish(&scope, &payload).await;
            }
            Ok(())
        } else {
            Err(anyhow!("Model {} is not loaded", model_ref_str))
        }
    }

    /// Convert a TTTConfig to a generated OnlineTrainingConfig wire type.
    fn ttt_config_to_wire(cfg: &crate::training::ttt::TTTConfig) -> GenOnlineTrainingConfig {
        GenOnlineTrainingConfig {
            enabled: cfg.enabled,
            learning_rate: cfg.learning_rate,
            gradient_steps: cfg.gradient_steps,
            max_grad_norm: cfg.max_grad_norm,
            min_input_length: cfg.min_input_length,
            max_ttt_context: cfg.max_ttt_context,
        }
    }

    /// Convert SamplingParams to a generated GenerationDefaults wire type.
    fn sampling_params_to_wire(params: &crate::config::SamplingParams) -> GenGenerationDefaults {
        GenGenerationDefaults {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k.map(|v| v as u32),
            max_tokens: params.max_tokens.map(|v| v as u32),
            repeat_penalty: params.repeat_penalty,
            stop_tokens: params.stop_tokens.clone().unwrap_or_default(),
            do_sample: params.do_sample,
        }
    }

    /// Return status entries for all known models (loaded + loading).
    /// Absence from this list means unloaded.
    async fn model_status_all(&self) -> Vec<GenModelStatusEntry> {
        let cache = self.loaded_models.read().await;
        let pending = self.pending_loads.lock().await;
        let mut entries: Vec<GenModelStatusEntry> = cache
            .iter()
            .map(|(_, model)| GenModelStatusEntry {
                model_ref: model.model_ref.clone(),
                status: "loaded".to_owned(),
                endpoint: model.endpoint.clone(),
                loaded_at: model.loaded_at.elapsed().as_millis() as i64,
                last_used: model.last_used.elapsed().as_millis() as i64,
                online_training_config: model.ttt_config.as_ref()
                    .map(Self::ttt_config_to_wire)
                    .unwrap_or_default(),
                generation_defaults: Self::sampling_params_to_wire(&model.generation_defaults),
            })
            .collect();
        for model_ref in pending.iter() {
            if !cache.contains(model_ref) {
                entries.push(GenModelStatusEntry {
                    model_ref: model_ref.clone(),
                    status: "loading".to_owned(),
                    endpoint: String::new(),
                    loaded_at: 0,
                    last_used: 0,
                    online_training_config: GenOnlineTrainingConfig::default(),
                    generation_defaults: GenGenerationDefaults::default(),
                });
            }
        }
        entries
    }

    /// Return status entry for a specific model ref (0 or 1 element).
    async fn model_status_single(&self, model_ref_str: &str) -> Vec<GenModelStatusEntry> {
        let cache = self.loaded_models.read().await;
        if let Some(model) = cache.peek(model_ref_str) {
            return vec![GenModelStatusEntry {
                model_ref: model_ref_str.to_owned(),
                status: "loaded".to_owned(),
                endpoint: model.endpoint.clone(),
                loaded_at: model.loaded_at.elapsed().as_millis() as i64,
                last_used: model.last_used.elapsed().as_millis() as i64,
                online_training_config: model.ttt_config.as_ref()
                    .map(Self::ttt_config_to_wire)
                    .unwrap_or_default(),
                generation_defaults: Self::sampling_params_to_wire(&model.generation_defaults),
            }];
        }
        let pending = self.pending_loads.lock().await;
        if pending.contains(model_ref_str) {
            vec![GenModelStatusEntry {
                model_ref: model_ref_str.to_owned(),
                status: "loading".to_owned(),
                endpoint: String::new(),
                loaded_at: 0,
                last_used: 0,
                online_training_config: GenOnlineTrainingConfig::default(),
                generation_defaults: GenGenerationDefaults::default(),
            }]
        } else {
            vec![]
        }
    }

    /// Get model status
    async fn model_status(&self, model_ref_str: &str) -> ModelStatusResponse {
        let cache = self.loaded_models.read().await;
        if let Some(model) = cache.peek(model_ref_str) {
            ModelStatusResponse {
                loaded: true,
                endpoint: model.endpoint.clone(),
                online_training_config: model.ttt_config.as_ref()
                    .map(Self::ttt_config_to_wire)
                    .unwrap_or_default(),
                ..Default::default()
            }
        } else {
            ModelStatusResponse { loaded: false, ..Default::default() }
        }
    }
    async fn get_inference_client(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<InferenceClient> {
        let _endpoint = self.load_model(model_ref_str, None, None).await?;
        let mut cache = self.loaded_models.write().await;
        let model = cache
            .get_mut(model_ref_str)
            .ok_or_else(|| anyhow!("Model {} not found after loading", model_ref_str))?;
        model.last_used = Instant::now();
        let client = model.client.clone();
        match ctx.claims() {
            Some(claims) => Ok(client.with_claims(claims.clone())),
            None => Ok(client),
        }
    }

    /// Load a LoRA adapter from a file
    async fn load_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext, path: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.load_lora(path).await
    }

    /// Unload the current LoRA adapter
    async fn unload_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.unload_lora().await
    }

    /// Check if a LoRA adapter is loaded
    async fn has_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<bool> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.has_lora().await
    }

    // Training loop control - forward to InferenceService via ZMQ
    async fn writeback_adaptation(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.ttt_writeback().await
    }

    async fn evict_adaptation(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.ttt_evict().await
    }

    async fn zero_delta(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.ttt_zero().await
    }

    async fn get_delta_status_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
    ) -> Result<crate::services::generated::inference_client::DeltaStatusResult> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.get_delta_status().await
    }

    async fn snapshot_delta_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
    ) -> Result<crate::services::generated::inference_client::SnapshotDeltaResult> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.snapshot_delta().await
    }

    async fn export_peft_adapter_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
        data: &ExportPeftRequest,
    ) -> Result<ExportPeftResult> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.export_peft_adapter(data).await
    }

}

// ═══════════════════════════════════════════════════════════════════════════════
// ModelHandler Implementation — generated dispatch for top-level + typed scope traits
// ═══════════════════════════════════════════════════════════════════════════════

use crate::services::generated::model_client::{
    ModelHandler, TttHandler, AdapterHandler, InferHandler,
    dispatch_model, serialize_response, ModelResponseVariant,
    LoadedModelResponse, ErrorInfo, ModelHealthStatus,
    StatusRequest,
    ModelStatusEntry as GenModelStatusEntry, OnlineTrainingConfig as GenOnlineTrainingConfig,
    GenerationDefaults as GenGenerationDefaults,
    // Top-level request types
    LoadModelRequest, UnloadModelRequest,
    // TTT types (names follow inference.capnp via using-import)
    LoraConfig, TrainStepRequest, TrainStepResult,
    DeltaStatusResult,
    SaveAdaptationRequest, SaveAdaptationResult,
    SnapshotDeltaResult, ExportPeftRequest, ExportPeftResult,
    WriteTttConfigRequest,
    // Adapter types
    AdapterInfo, MergeLoraRequest,
    // Infer types (GenerationRequest follows inference.capnp name)
    GenerationRequest, ChatTemplateRequest, ModelStatusResponse,
    EmbedImagesRequest, EmbedImagesResponse,
};
// AdaptationStrategy is now from inference_client (canonical source via using-import).
// model_client types reference it directly — no conversion needed.


#[async_trait::async_trait(?Send)]
impl TttHandler for ModelService {
    async fn handle_init(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &LoraConfig,
    ) -> Result<()> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.create_lora(data).await
    }

    async fn handle_train(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TrainStepRequest,
    ) -> Result<TrainStepResult> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.train_step(data).await
    }

    async fn handle_train_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TrainStepRequest,
    ) -> Result<(crate::services::generated::model_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        let stream_info = client.train_step_stream(data, ctx.ephemeral_pubkey.unwrap_or([0u8; 32])).await?;
        Ok((stream_info, Box::pin(async {})))
    }

    async fn handle_writeback(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.writeback_adaptation(model_ref, ctx).await
    }

    async fn handle_evict(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.evict_adaptation(model_ref, ctx).await
    }

    async fn handle_zero(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.zero_delta(model_ref, ctx).await
    }

    async fn handle_status(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<DeltaStatusResult> {
        self.get_delta_status_forward(model_ref, ctx).await
    }

    async fn handle_save(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &SaveAdaptationRequest,
    ) -> Result<SaveAdaptationResult> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.save_adaptation(data).await
    }

    async fn handle_snapshot(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<SnapshotDeltaResult> {
        self.snapshot_delta_forward(model_ref, ctx).await
    }

    async fn handle_export(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &ExportPeftRequest,
    ) -> Result<ExportPeftResult> {
        self.export_peft_adapter_forward(model_ref, ctx, data).await
    }

    async fn handle_write_ttt_config(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &WriteTttConfigRequest,
    ) -> Result<()> {
        // 1. Parse model ref and resolve worktree path
        let parsed = ModelRef::parse(model_ref)?;
        let tracked = self.registry.get_by_name(&parsed.model).await
            .map_err(|e| anyhow!("Model '{}' not found in registry: {}", parsed.model, e))?;
        let repo_client = self.registry.repo(&tracked.id);

        let branch_name = match &parsed.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            _ => repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned()),
        };

        let storage_paths = crate::storage::StoragePaths::new()?;
        let model_path = storage_paths.worktree_path(&parsed.model, &branch_name)?;

        if !model_path.exists() {
            return Err(anyhow!("Model worktree not found for {}", model_ref));
        }

        // 2. Build HyprstreamTrainingConfig from request
        let ttt_config = crate::config::TTTTrainingConfig {
            learning_rate: if data.learning_rate > 0.0 { data.learning_rate } else { 3e-4 },
            gradient_steps: if data.gradient_steps > 0 { data.gradient_steps } else { 3 },
            max_grad_norm: if data.max_grad_norm > 0.0 { data.max_grad_norm } else { 1.0 },
            min_input_length: if data.min_input_length > 0 { data.min_input_length } else { 32 },
            max_ttt_context: if data.max_ttt_context > 0 { data.max_ttt_context } else { 512 },
            rank_oracle: None,
            gradient_gating: None,
        };

        let training_config = crate::config::HyprstreamTrainingConfig {
            mode: crate::config::TrainingMode::TestTimeTraining,
            ttt: ttt_config,
            lora_rank: if data.lora_rank > 0 { data.lora_rank as usize } else { crate::config::default_lora_rank() },
            lora_alpha: if data.lora_alpha > 0.0 { Some(data.lora_alpha) } else { None },
            target_modules: if data.target_modules.is_empty() {
                crate::config::default_target_modules()
            } else {
                data.target_modules.clone()
            },
            ..Default::default()
        };

        // 3. Write config.json
        crate::runtime::model_config::ModelConfig::save_training_config(&model_path, &training_config)?;

        // 4. Stage and commit via worktree-scoped API
        let wt = repo_client.worktree(&branch_name);
        wt.stage_files(&StageFilesRequest {
            files: vec!["config.json".to_owned()],
        }).await?;
        wt.commit_with_author(&CommitWithAuthorRequest {
            message: "Update hyprstream_training config via RPC".to_owned(),
            author_name: "hyprstream".to_owned(),
            author_email: "noreply@hyprstream.dev".to_owned(),
        }).await?;

        info!("TTT config written for {}", model_ref);

        // 5. Auto-reload if requested and model is loaded
        if data.auto_reload {
            let is_loaded = {
                let cache = self.loaded_models.read().await;
                cache.contains(model_ref)
            };
            if is_loaded {
                info!("Auto-reloading {} after TTT config change", model_ref);
                self.unload_model(model_ref).await?;
                self.load_model(model_ref, None, None).await?;
            }
        }

        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl AdapterHandler for ModelService {
    async fn handle_load(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, value: &str,
    ) -> Result<()> {
        self.load_lora(model_ref, ctx, value).await
    }

    async fn handle_unload(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.unload_lora(model_ref, ctx).await
    }

    async fn handle_status(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<bool> {
        self.has_lora(model_ref, ctx).await
    }

    async fn handle_inspect(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, value: &str,
    ) -> Result<AdapterInfo> {
        // Resolve model_ref to a worktree client (does NOT require model loaded in memory)
        let parsed = ModelRef::parse(model_ref)?;
        let tracked = self.registry.get_by_name(&parsed.model).await
            .map_err(|e| anyhow!("Model '{}' not found in registry: {}", parsed.model, e))?;
        let repo_client = self.registry.repo(&tracked.id);
        let branch_name = match &parsed.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            _ => repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned()),
        };
        let fs = repo_client.worktree(&branch_name);

        // Read adapter_config.json from the adapter directory
        let config_path = format!("{}/adapter_config.json", value);
        let config_bytes = fs.read_file_chunked(&config_path).await
            .map_err(|e| anyhow!("Failed to read {}: {}", config_path, e))?;
        let config_json: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|e| anyhow!("Failed to parse adapter_config.json: {}", e))?;

        // Verify adapter_model.safetensors exists
        let model_path = format!("{}/adapter_model.safetensors", value);
        let stat = fs.stat_path(&model_path).await
            .map_err(|e| anyhow!("Failed to stat {}: {}", model_path, e))?;
        if !stat.exists {
            anyhow::bail!("adapter_model.safetensors not found in {}", value);
        }

        // Extract PEFT fields from config
        let rank = config_json.get("r")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32;
        let lora_alpha = config_json.get("lora_alpha")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        let target_modules = config_json.get("target_modules")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let base_model = config_json.get("base_model_name_or_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();

        // Extract directory name from path
        let name = value.rsplit('/').next().unwrap_or(value).to_owned();

        Ok(AdapterInfo {
            name,
            path: value.to_owned(),
            rank,
            lora_alpha,
            target_modules,
            base_model,
        })
    }

    async fn handle_merge(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &MergeLoraRequest,
    ) -> Result<()> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.merge_lora(data).await
    }
}

#[async_trait::async_trait(?Send)]
impl InferHandler for ModelService {
    async fn handle_generate_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &GenerationRequest,
    ) -> Result<(crate::services::generated::model_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        let stream_info = client.generate_stream(data, ctx.ephemeral_pubkey.unwrap_or([0u8; 32])).await?;
        Ok((stream_info, Box::pin(async {})))
    }

    async fn handle_apply_chat_template(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &ChatTemplateRequest,
    ) -> Result<String> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.apply_chat_template(data).await
    }

    async fn handle_embed(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &EmbedImagesRequest,
    ) -> Result<EmbedImagesResponse> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.embed(data).await
    }

    async fn handle_status(
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<ModelStatusResponse> {
        Ok(self.model_status(model_ref).await)
    }
}

#[async_trait::async_trait(?Send)]
impl ModelHandler for ModelService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        let subject = ctx.subject();
        let allowed = self.policy_client.check(&PolicyCheck { subject: subject.to_string(), domain: "*".to_owned(), resource: resource.to_owned(), operation: operation.to_owned() }).await.unwrap_or_else(|e| {
            warn!("Policy check failed for {} on {}: {} - denying access", subject, resource, e);
            false
        });
        if allowed {
            Ok(())
        } else {
            anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
        }
    }

    async fn handle_load(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &LoadModelRequest,
    ) -> Result<ModelResponseVariant> {
        let max_ctx = data.max_context.filter(|&n| n != 0);
        let kv_q = data.kv_quant.filter(|q| *q != KVQuantType::None);
        let model_ref = &data.model_ref;
        match self.load_model(model_ref, max_ctx, kv_q).await {
            Ok(endpoint) => Ok(ModelResponseVariant::LoadResult(LoadedModelResponse {
                model_ref: model_ref.to_owned(),
                endpoint,
            })),
            Err(e) => Ok(ModelResponseVariant::Error(ErrorInfo {
                message: format!("Failed to load model: {e}"),
                code: "LOAD_FAILED".into(),
                details: String::new(),
            })),
        }
    }

    async fn handle_unload(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &UnloadModelRequest,
    ) -> Result<ModelResponseVariant> {
        let model_ref = &data.model_ref;
        match self.unload_model(model_ref).await {
            Ok(()) => Ok(ModelResponseVariant::UnloadResult),
            Err(e) => Ok(ModelResponseVariant::Error(ErrorInfo {
                message: format!("Failed to unload model: {e}"),
                code: "UNLOAD_FAILED".into(),
                details: String::new(),
            })),
        }
    }

    async fn handle_status(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &StatusRequest,
    ) -> Result<ModelResponseVariant> {
        let entries = if data.model_ref.is_empty() {
            self.model_status_all().await
        } else {
            self.model_status_single(&data.model_ref).await
        };
        Ok(ModelResponseVariant::StatusResult(entries))
    }

    async fn handle_health_check(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
    ) -> Result<ModelResponseVariant> {
        let cache = self.loaded_models.read().await;
                let loaded_count = cache.len() as u32;
                let max_models = self.config.max_models as u32;
                drop(cache);
                Ok(ModelResponseVariant::HealthCheckResult(ModelHealthStatus {
                    status: "healthy".into(),
                    loaded_model_count: loaded_count,
                    max_models,
                    total_memory_bytes: 0,
                }))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9P Filesystem Handler (FsHandler trait)
// ═══════════════════════════════════════════════════════════════════════════════

use crate::services::generated::model_client::{
    NpWalk, NpOpen, NpRead, NpWrite, NpClunk, NpStatReq, NpCreate, NpRemove,
    RWalk, ROpen, RRead, RWrite, RStat,
    Qid as GenQid, NpStat as GenNpStat,
    FsHandler,
};
use crate::services::fs::{SyntheticTree, SyntheticNode, SyntheticQid};
use hyprstream_vfs::DirEntry;

impl ModelService {
    /// Get the persistent synthetic 9P tree (lazily initialized).
    fn fs_tree(&self) -> &SyntheticTree {
        self.inner.fs_tree.get_or_init(|| self.build_fs_tree())
    }

    /// Build a synthetic 9P tree from current model service state.
    fn build_fs_tree(&self) -> SyntheticTree {
        let inner_list = Arc::clone(&self.inner);
        let inner_resolve = Arc::clone(&self.inner);

        SyntheticTree::new(SyntheticNode::DynamicDir {
            list: Box::new(move || {
                let cache = inner_list.loaded_models.blocking_read();
                let pending = inner_list.pending_loads.blocking_lock();
                let mut entries: Vec<DirEntry> = cache
                    .iter()
                    .map(|(name, _)| DirEntry { name: name.clone(), is_dir: true, size: 0, stat: None })
                    .collect();
                for name in pending.iter() {
                    if !cache.contains(name) {
                        entries.push(DirEntry { name: name.clone(), is_dir: true, size: 0, stat: None });
                    }
                }
                entries
            }),
            resolve: Box::new(move |ref_name| {
                let cache = inner_resolve.loaded_models.blocking_read();
                let is_loaded = cache.contains(ref_name);
                let defaults_json = if let Some(model) = cache.peek(ref_name) {
                    serde_json::to_vec_pretty(&model.generation_defaults).unwrap_or_default()
                } else {
                    b"{}".to_vec()
                };
                let status_str = if is_loaded { "loaded" } else { "unloaded" };

                let mut children = std::collections::HashMap::new();
                let status_owned = status_str.to_owned();
                children.insert("status".to_owned(), SyntheticNode::ReadFile(
                    Box::new(move || format!("{status_owned}\n").into_bytes()),
                ));
                children.insert("defaults".to_owned(), SyntheticNode::ReadFile(
                    Box::new(move || defaults_json.clone()),
                ));
                children.insert("ctl".to_owned(), SyntheticNode::CtlFile {
                    handler: Box::new(|data, _subject| {
                        let cmd = String::from_utf8_lossy(data).trim().to_owned();
                        Ok(format!("ctl: {cmd}\n").into_bytes())
                    }),
                });
                Some(SyntheticNode::Dir { children })
            }),
        })
    }

    fn qid_to_gen(qid: &SyntheticQid) -> GenQid {
        GenQid { qtype: qid.qtype, version: qid.version, path: qid.path }
    }
}

#[async_trait::async_trait(?Send)]
impl FsHandler for ModelService {
    async fn handle_walk(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpWalk,
    ) -> Result<RWalk> {
        let tree = self.fs_tree();
        let owner = ctx.subject().to_string();
        let (_fid, qid) = tree.walk(&data.wnames, &owner, Some(data.newfid))
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RWalk { qid: Self::qid_to_gen(&qid) })
    }

    async fn handle_open(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpOpen,
    ) -> Result<ROpen> {
        let tree = self.fs_tree();
        let owner = ctx.subject().to_string();
        // Re-walk to get the fid, then open.
        let (qid, iounit) = tree.open(data.fid, data.mode, &owner)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(ROpen { qid: Self::qid_to_gen(&qid), iounit })
    }

    async fn handle_read(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpRead,
    ) -> Result<RRead> {
        let tree = self.fs_tree();
        let owner = ctx.subject().to_string();
        let bytes = tree.read(data.fid, data.offset, data.count, &owner)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RRead { data: bytes })
    }

    async fn handle_write(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpWrite,
    ) -> Result<RWrite> {
        let tree = self.fs_tree();
        let subject = ctx.subject();
        let owner = subject.to_string();
        let count = tree.write(data.fid, data.offset, &data.data, &owner, &subject)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RWrite { count })
    }

    async fn handle_clunk(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpClunk,
    ) -> Result<()> {
        let tree = self.fs_tree();
        let owner = ctx.subject().to_string();
        tree.clunk(data.fid, &owner);
        Ok(())
    }

    async fn handle_stat(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpStatReq,
    ) -> Result<RStat> {
        let tree = self.fs_tree();
        let owner = ctx.subject().to_string();
        let (qid, name) = tree.stat(data.fid, &owner)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RStat {
            stat: GenNpStat {
                qid: Self::qid_to_gen(&qid),
                mode: if qid.qtype & 0x80 != 0 { 0o040755 } else { 0o100644 },
                atime: 0,
                mtime: 0,
                length: 0,
                name,
                uid: String::new(),
                gid: String::new(),
                muid: String::new(),
            },
        })
    }

    async fn handle_create(&self, _ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, _data: &NpCreate,
    ) -> Result<ROpen> {
        anyhow::bail!("create not supported on model fs")
    }

    async fn handle_remove(&self, _ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, _data: &NpRemove,
    ) -> Result<()> {
        anyhow::bail!("remove not supported on model fs")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Load request interception — parse capnp to detect load before dispatch
// ═══════════════════════════════════════════════════════════════════════════════

/// Parsed load request data extracted from Cap'n Proto payload.
struct ParsedLoadRequest {
    model_ref: String,
    max_context: u32,
    kv_quant: crate::model_capnp::KVQuantType,
}

impl ParsedLoadRequest {
    fn to_load_params(&self) -> (Option<u32>, Option<KVQuantType>) {
        use crate::model_capnp::KVQuantType as CKV;
        let max_ctx = match self.max_context {
            0 => None,
            n => Some(n),
        };
        let kv_q = match self.kv_quant {
            CKV::Int8 => Some(KVQuantType::Int8),
            CKV::Nf4 => Some(KVQuantType::Nf4),
            CKV::Fp4 => Some(KVQuantType::Fp4),
            CKV::None => None,
        };
        (max_ctx, kv_q)
    }
}

impl ModelService {
    /// Try to parse a load request from the raw Cap'n Proto payload.
    /// Returns `None` for all other request variants (list, unload, health, scoped, etc.).
    fn try_parse_load_request(payload: &[u8]) -> Option<(u64, ParsedLoadRequest)> {
        use crate::model_capnp::model_request;
        use crate::model_capnp::KVQuantType as CKV;
        use crate::optional_capnp::option_uint32;
        use crate::model_capnp::option_k_v_quant_type;
        let reader = capnp::serialize::read_message(
            &mut std::io::Cursor::new(payload),
            capnp::message::ReaderOptions::new(),
        ).ok()?;
        let req = reader.get_root::<model_request::Reader>().ok()?;
        let request_id = req.get_id();
        match req.which().ok()? {
            model_request::Which::Load(data) => {
                let data = data.ok()?;
                let model_ref = data.get_model_ref().ok()?.to_str().ok()?.to_owned();
                let max_context = match data.get_max_context().ok()?.which().ok()? {
                    option_uint32::Which::None(()) => 0u32,
                    option_uint32::Which::Some(v) => v,
                };
                let kv_quant = match data.get_kv_quant().ok()?.which().ok()? {
                    option_k_v_quant_type::Which::None(()) => CKV::None,
                    option_k_v_quant_type::Which::Some(v) => v.unwrap_or(CKV::None),
                };
                Some((request_id, ParsedLoadRequest { model_ref, max_context, kv_quant }))
            }
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZmqService Implementation — delegates to generated dispatch_model
// ═══════════════════════════════════════════════════════════════════════════════

#[async_trait(?Send)]
impl crate::services::ZmqService for ModelService {
    async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<crate::services::Continuation>)> {
        debug!(
            "Model request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );

        // Intercept load requests to avoid blocking the request loop.
        // Model loading can take 60s+ (weight transfer to GPU), which would
        // block all other model service requests (list, health, info, etc.).
        // Instead, return an immediate "accepted" response and do the actual
        // load in a Continuation (spawned via spawn_local after the REP is sent).
        if let Some((request_id, load_data)) = Self::try_parse_load_request(payload) {
            // Fast path: if already loaded or already loading, return immediately
            {
                let mut cache = self.loaded_models.write().await;
                if let Some(model) = cache.get_mut(&load_data.model_ref) {
                    model.last_used = Instant::now();
                    let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                        LoadedModelResponse {
                            model_ref: load_data.model_ref.clone(),
                            endpoint: model.endpoint.clone(),
                        },
                    ))?;
                    return Ok((response, None));
                }
            }
            // If already being loaded by another request, return the endpoint
            // without spawning a duplicate continuation
            {
                let pending = self.pending_loads.lock().await;
                if pending.contains(&load_data.model_ref) {
                    debug!("Model {} already being loaded, deduplicating", load_data.model_ref);
                    let endpoint = Self::inference_endpoint(&load_data.model_ref);
                    let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                        LoadedModelResponse {
                            model_ref: load_data.model_ref.clone(),
                            endpoint,
                        },
                    ))?;
                    return Ok((response, None));
                }
            }

            // Slow path: return "accepted" immediately, load in continuation
            let model_ref = load_data.model_ref.clone();
            info!("Load request accepted for {} (async)", model_ref);

            let endpoint = Self::inference_endpoint(&model_ref);
            let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                LoadedModelResponse {
                    model_ref: model_ref.clone(),
                    endpoint,
                },
            ))?;

            let service = self.clone(); // Arc clone — cheap, 'static
            let (load_max_context, load_kv_quant) = load_data.to_load_params();
            let continuation: crate::services::Continuation = Box::pin(async move {
                let model_name = model_ref.split(':').next().unwrap_or(&model_ref);
                let scope = format!("serve:model:{}", model_name);
                match service.load_model(&model_ref, load_max_context, load_kv_quant).await {
                    Ok(endpoint) => {
                        info!("Model {} loaded successfully at {}", model_ref, endpoint);
                        let event = crate::events::EventEnvelope::new(
                            crate::events::EventSource::Model,
                            scope.clone(),
                            crate::events::EventPayload::ModelLoaded {
                                model_ref: model_ref.clone(),
                                endpoint,
                            },
                        );
                        if let Ok(payload) = serde_json::to_vec(&event) {
                            let n = service.notification_publisher.publish(&scope, &payload).await
                                .unwrap_or(0);
                            debug!("Published model.loaded to {} subscriber(s)", n);
                        }
                    }
                    Err(e) => {
                        warn!("Model {} failed to load: {}", model_ref, e);
                        let event = crate::events::EventEnvelope::new(
                            crate::events::EventSource::Model,
                            scope.clone(),
                            crate::events::EventPayload::ModelFailed {
                                model_ref: model_ref.clone(),
                                error: e.to_string(),
                            },
                        );
                        if let Ok(payload) = serde_json::to_vec(&event) {
                            let _ = service.notification_publisher.publish(&scope, &payload).await;
                        }
                    }
                }
            });

            return Ok((response, Some(continuation)));
        }

        dispatch_model(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        "model"
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.expected_audience.as_deref()
    }

    fn local_issuer_url(&self) -> Option<&str> {
        self.inner.local_issuer_url.as_deref()
    }

    fn federation_key_source(
        &self,
    ) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>> {
        self.inner.federation_key_source.clone()
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = ModelResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}

// ============================================================================
// Helper types
// ============================================================================

// ModelStatusEntry, OnlineTrainingConfigInfo, ModelStatusInfo deleted — use generated types directly:
// - GenModelStatusEntry = generated::model_client::ModelStatusEntry
// - GenOnlineTrainingConfig = generated::model_client::OnlineTrainingConfig
// - ModelStatusResponse = generated::model_client::ModelStatusResponse (infer-scoped)

// ModelZmqClient removed — use generated ModelClient directly.
// Convenience methods (load/unload/status/infer_stream) are now inlined at call sites.



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ModelServiceConfig::default();
        assert_eq!(config.max_models, 5);
        assert_eq!(config.max_context, None);
        assert_eq!(config.kv_quant, KVQuantType::None);
    }
}
