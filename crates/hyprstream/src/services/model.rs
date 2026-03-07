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
//!       │ ModelZmqClient (async ZMQ I/O)
//!       ▼
//! ModelService (multi-threaded runtime)
//!       │
//!       ├── LRU cache of loaded models
//!       ├── Spawns InferenceService per model
//!       └── Routes requests to InferenceService
//!             │
//!             │ InferenceZmqClient (async ZMQ I/O)
//!             ▼
//!       InferenceService (dedicated thread per model)
//! ```
//!
//! # Endpoint
//!
//! Uses `registry().endpoint("model", SocketKind::Rep)` for the REP endpoint.
//! Default fallback: `inproc://hyprstream/model`

use async_trait::async_trait;
use crate::api::openai_compat::ChatMessage;
use crate::config::{GenerationRequest, TemplatedPrompt};
use crate::runtime::kv_quant::KVQuantType;
use crate::runtime::RuntimeConfig;
use crate::services::{
    rpc_types::StreamInfo, EnvelopeContext, InferenceZmqClient,
    NotificationClient, NotificationPublisher, PolicyClient,
};
use crate::services::GenRegistryClient;
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

/// Per-model runtime configuration for loading
#[derive(Debug, Clone, Default)]
pub struct ModelLoadConfig {
    /// Maximum context length (None = use service default)
    pub max_context: Option<usize>,
    /// KV cache quantization type (None = use service default)
    pub kv_quant: Option<KVQuantType>,
    /// Number of inference service instances (default: 1).
    /// Number of inference worker instances per model (default: 1).
    ///
    /// When > 1, a ROUTER/DEALER load balancer is spawned and N workers
    /// connect to the DEALER backend for concurrent request handling.
    /// Each worker loads its own copy of the model.
    ///
    /// **WARNING — experimental, stateless inference only:**
    /// Multi-instance mode breaks session affinity (KV cache is per-worker),
    /// pending TTT commit/rollback (per-worker state), and LoRA consistency
    /// (base_delta is per-worker). Use only for throughput scaling on pure
    /// stateless inference. TTT with shared DeltaPool works (pool is Arc-shared),
    /// but commit/rollback and sessions require single-instance mode.
    pub num_inference_instances: Option<usize>,
}

/// Information about a loaded model
pub struct LoadedModel {
    /// Model reference string (e.g., "qwen3-small:main")
    pub model_ref: String,
    /// ZMQ endpoint for this model's InferenceService
    pub endpoint: String,
    /// Handle to stop the InferenceService
    pub service_handle: hyprstream_rpc::service::SpawnedService,
    /// Client for communicating with the InferenceService
    pub client: InferenceZmqClient,
    /// When the model was loaded
    pub loaded_at: Instant,
    /// When the model was last used
    pub last_used: Instant,
    /// Online training (TTT) configuration (if enabled)
    pub ttt_config: Option<crate::training::ttt::TTTConfig>,
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
    pub max_context: Option<usize>,
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
    /// Ed25519 signing key for creating InferenceZmqClients
    signing_key: SigningKey,
    /// Policy client for authorization checks in InferenceService
    policy_client: PolicyClient,
    /// Notification publisher for model lifecycle events
    notification_publisher: NotificationPublisher,
    /// Registry client for resolving model paths
    registry: GenRegistryClient,
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
        registry: GenRegistryClient,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        // SAFETY: 5 is a valid non-zero value
        const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(5) {
            Some(n) => n,
            None => unreachable!(),
        };
        let cache_size = NonZeroUsize::new(config.max_models).unwrap_or(DEFAULT_CACHE_SIZE);

        let notif_client = NotificationClient::new(signing_key.clone(), RequestIdentity::local());
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

    /// Create a model service with callback router for spawned mode
    pub fn with_callback_router(
        config: ModelServiceConfig,
        signing_key: SigningKey,
        policy_client: PolicyClient,
        registry: GenRegistryClient,
        callback_router: crate::services::callback::CallbackRouter,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(5) {
            Some(n) => n,
            None => unreachable!(),
        };
        let cache_size = NonZeroUsize::new(config.max_models).unwrap_or(DEFAULT_CACHE_SIZE);

        let notif_client = NotificationClient::new(signing_key.clone(), RequestIdentity::local());
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
        })}
    }

    /// Load a model by reference with optional per-model config, returns the inference endpoint
    async fn load_model(&self, model_ref_str: &str, config: Option<ModelLoadConfig>) -> Result<String> {
        // Check if already loaded
        {
            let mut cache = self.loaded_models.write().await;
            if let Some(model) = cache.get_mut(model_ref_str) {
                model.last_used = Instant::now();
                debug!("Model {} already loaded at {}", model_ref_str, model.endpoint);
                return Ok(model.endpoint.clone());
            }
        }

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

        // Create unique endpoint for this model using registry
        // Each model gets its own socket: inference-{safe_name}.sock (IPC) or
        // inproc://hyprstream/inference-{safe_name} (inproc)
        let safe_name = model_ref_str.replace([':', '/', '\\'], "-");
        let service_name = format!("inference-{safe_name}");
        let endpoint = registry().endpoint(&service_name, SocketKind::Rep).to_zmq_string();

        info!("Loading model {} at endpoint {}", model_ref_str, endpoint);

        // Create runtime config - use per-model config if provided, otherwise service defaults
        let load_config = config.unwrap_or_default();
        let runtime_config = RuntimeConfig {
            max_context: load_config.max_context.or(self.config.max_context),
            kv_quant_type: load_config.kv_quant.unwrap_or(self.config.kv_quant),
            ..Default::default()
        };

        // Obtain FsOps from the registry for path-contained adapter I/O
        let fs: Option<crate::services::WorktreeClient> = Some(repo_client.worktree(&branch_name));

        // Start InferenceService for this model via standard Spawnable infrastructure
        let num_instances = load_config.num_inference_instances.unwrap_or(1).max(1);
        let zmq_ctx = Arc::clone(hyprstream_rpc::ZmqService::context(self));
        let spawner = hyprstream_rpc::service::spawner::ServiceSpawner::threaded();

        let service_handle = if num_instances > 1 {
            // Multi-instance: spawn ROUTER/DEALER load balancer + N workers.
            // WARNING: experimental — breaks session affinity, pending TTT
            // commit/rollback, and LoRA consistency across workers.
            warn!(
                "Spawning {} inference instances for {} behind load balancer \
                 (experimental: sessions, pending TTT, and LoRA state are per-worker)",
                num_instances, model_ref_str
            );

            let frontend_transport = hyprstream_rpc::transport::TransportConfig::from_endpoint(&endpoint);
            let backend_endpoint = format!("inproc://hyprstream/inference-{safe_name}-backend");
            let backend_transport = hyprstream_rpc::transport::TransportConfig::from_endpoint(&backend_endpoint);

            // Spawn load balancer (ROUTER frontend, DEALER backend)
            let lb = hyprstream_rpc::service::spawner::LoadBalancerService::new(
                format!("inference-{safe_name}-lb"),
                Arc::clone(&zmq_ctx),
                frontend_transport,
                backend_transport.clone(),
            );
            let lb_handle = spawner.spawn(lb).await
                .map_err(|e| anyhow!("Failed to spawn load balancer: {}", e))?;

            // Spawn N worker instances, each connecting to the DEALER backend
            let worker_transport = backend_transport.with_connect_mode();
            let mut worker_handles = Vec::with_capacity(num_instances);
            for idx in 0..num_instances {
                let mut worker_config = crate::services::InferenceServiceConfig::new(
                    &model_path,
                    runtime_config.clone(),
                    self.signing_key.verifying_key(),
                    self.signing_key.clone(),
                    Arc::clone(&zmq_ctx),
                    worker_transport.clone(),
                    fs.clone(),
                );
                if let Some(ref aud) = self.expected_audience {
                    worker_config = worker_config.with_expected_audience(aud.clone());
                }
                let handle = spawner.spawn(worker_config).await
                    .map_err(|e| anyhow!("Failed to spawn inference worker {}: {}", idx, e))?;
                worker_handles.push(handle);
            }

            // Return the LB handle as the primary handle (stopping it stops the frontend)
            // Workers are tracked but not individually managed
            // TODO: Track worker handles for graceful shutdown
            lb_handle
        } else {
            // Single instance: direct bind (zero overhead, current behavior)
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
            spawner.spawn(service_config).await
                .map_err(|e| anyhow!("Failed to spawn inference service: {}", e))?
        };

        // Create client for this service
        let client = InferenceZmqClient::with_endpoint(
            &endpoint,
            self.signing_key.clone(),
            RequestIdentity::local(),
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
                },
            );
        }

        info!("Model {} loaded successfully", model_ref_str);
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

    /// Return status entries for all known models (loaded + loading).
    /// Absence from this list means unloaded.
    async fn model_status_all(&self) -> Vec<ModelStatusEntry> {
        let cache = self.loaded_models.read().await;
        let pending = self.pending_loads.lock().await;
        let mut entries: Vec<ModelStatusEntry> = cache
            .iter()
            .map(|(_, model)| ModelStatusEntry {
                model_ref: model.model_ref.clone(),
                status: "loaded".to_owned(),
                endpoint: model.endpoint.clone(),
                loaded_at: model.loaded_at.elapsed().as_millis() as i64,
                last_used: model.last_used.elapsed().as_millis() as i64,
                online_training_config: model.ttt_config.as_ref().map(OnlineTrainingConfigInfo::from),
            })
            .collect();
        for model_ref in pending.iter() {
            if !cache.contains(model_ref) {
                entries.push(ModelStatusEntry {
                    model_ref: model_ref.clone(),
                    status: "loading".to_owned(),
                    endpoint: String::new(),
                    loaded_at: 0,
                    last_used: 0,
                    online_training_config: None,
                });
            }
        }
        entries
    }

    /// Return status entry for a specific model ref (0 or 1 element).
    async fn model_status_single(&self, model_ref_str: &str) -> Vec<ModelStatusEntry> {
        let cache = self.loaded_models.read().await;
        if let Some(model) = cache.peek(model_ref_str) {
            return vec![ModelStatusEntry {
                model_ref: model_ref_str.to_owned(),
                status: "loaded".to_owned(),
                endpoint: model.endpoint.clone(),
                loaded_at: model.loaded_at.elapsed().as_millis() as i64,
                last_used: model.last_used.elapsed().as_millis() as i64,
                online_training_config: model.ttt_config.as_ref().map(OnlineTrainingConfigInfo::from),
            }];
        }
        let pending = self.pending_loads.lock().await;
        if pending.contains(model_ref_str) {
            vec![ModelStatusEntry {
                model_ref: model_ref_str.to_owned(),
                status: "loading".to_owned(),
                endpoint: String::new(),
                loaded_at: 0,
                last_used: 0,
                online_training_config: None,
            }]
        } else {
            vec![]
        }
    }

    /// Get model status
    async fn model_status(&self, model_ref_str: &str) -> ModelStatusInfo {
        let cache = self.loaded_models.read().await;
        if let Some(model) = cache.peek(model_ref_str) {
            ModelStatusInfo {
                loaded: true,
                endpoint: Some(model.endpoint.clone()),
                online_training_config: model.ttt_config.as_ref().map(OnlineTrainingConfigInfo::from),
            }
        } else {
            ModelStatusInfo {
                loaded: false,
                endpoint: None,
                online_training_config: None,
            }
        }
    }
    async fn get_inference_client(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<InferenceZmqClient> {
        let _endpoint = self.load_model(model_ref_str, None).await?;
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

    /// Apply chat template via the model's InferenceService
    async fn apply_chat_template(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
        messages: Vec<ChatMessage>,
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
    ) -> Result<TemplatedPrompt> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;

        // Convert ChatMessage to the template engine's format
        let template_messages: Vec<crate::runtime::template_engine::ChatMessage> = messages
            .iter()
            .map(|m| crate::runtime::template_engine::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: m.tool_calls.as_ref().map(|tcs| {
                    tcs.iter().map(|tc| serde_json::to_value(tc).unwrap_or_default()).collect()
                }),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();

        // Serialize tools to JSON string for transport over Cap'n Proto
        let tools_json = tools.map(|t| serde_json::to_string(t).unwrap_or_default())
            .unwrap_or_default();

        // Call InferenceService's apply_chat_template
        let prompt_str = client.apply_chat_template_with_tools(
            &template_messages, add_generation_prompt, &tools_json,
        ).await?;

        Ok(TemplatedPrompt::new(prompt_str))
    }

    /// Create a new LoRA adapter on the loaded model
    async fn create_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext, config: crate::training::TenantDeltaConfig) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.create_lora(&config).await
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
    async fn commit_adaptation(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.commit_adaptation().await
    }

    async fn rollback_adaptation(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.rollback_adaptation().await
    }

    #[allow(clippy::too_many_arguments)]
    async fn train_step_stream(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
        input: &str,
        gradient_steps: u32,
        learning_rate: f32,
        auto_commit: bool,
        client_ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamInfo> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.train_step_stream(input, gradient_steps, learning_rate, auto_commit, client_ephemeral_pubkey).await
    }

    async fn reset_delta(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.reset_delta().await
    }

    async fn get_delta_status_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
    ) -> Result<crate::services::generated::inference_client::DeltaStatusResult> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.gen.get_delta_status().await
    }

    async fn snapshot_delta_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
    ) -> Result<crate::services::generated::inference_client::SnapshotDeltaResult> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.gen.snapshot_delta().await
    }

    async fn export_peft_adapter_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
        name: &str,
        commit_message: &str,
    ) -> Result<crate::services::generated::inference_client::ExportPeftResult> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.gen.export_peft_adapter(name, commit_message).await
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
    // Top-level request types
    LoadModelRequest, UnloadModelRequest, KVQuantTypeEnum,
    // TTT types
    InitLoraRequest, TrainStepRequest, TrainStepResponse,
    GetDeltaStatusResponse, ModuleNormRatio,
    SaveAdaptationRequest, SaveAdaptationResponse,
    SnapshotDeltaResponse, TttExportRequest, TttExportResponse,
    WriteTttConfigRequest,
    // Adapter types
    AdapterInfo, AdapterMergeRequest,
    // Infer types
    GenerateRequest, ApplyChatTemplateRequest, ModelStatusResponse, OnlineTrainingConfig,
    EmbedRequest, EmbedResponse,
};
// Conflicting names — use canonical path at usage sites:
//   model_client::LoadedModelInfo, model_client::StreamInfo,
//   model_client::ChatMessage

#[async_trait::async_trait(?Send)]
impl TttHandler for ModelService {
    async fn handle_init(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &InitLoraRequest,
    ) -> Result<()> {
        let config = crate::training::TenantDeltaConfig {
            rank: data.rank as usize,
            alpha: data.alpha,
            dropout: data.dropout,
            target_modules: data.target_modules.clone(),
            learning_rate: data.learning_rate as f64,
            ..crate::training::TenantDeltaConfig::default()
        };
        self.create_lora(model_ref, ctx, config).await
    }

    async fn handle_train(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TrainStepRequest,
    ) -> Result<TrainStepResponse> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        let r = client.gen.train_step(
            &data.input, data.gradient_steps, data.learning_rate, data.auto_commit,
        ).await?;
        Ok(TrainStepResponse {
            avg_loss: r.avg_loss,
            loss_improvement: r.loss_improvement,
            steps_performed: r.steps_performed,
            adaptation_time_ms: r.adaptation_time_ms,
            initial_perplexity: r.initial_perplexity,
            final_perplexity: r.final_perplexity,
            recommendation: r.recommendation,
            committed: r.committed,
            gradient_clipped: r.gradient_clipped,
        })
    }

    async fn handle_train_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TrainStepRequest,
    ) -> Result<(crate::services::generated::model_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let info = self.train_step_stream(
                model_ref, ctx, &data.input, data.gradient_steps, data.learning_rate,
                data.auto_commit, ctx.ephemeral_pubkey,
            ).await?;
        let stream_info = crate::services::generated::model_client::StreamInfo {
            stream_id: info.stream_id,
            endpoint: info.endpoint,
            server_pubkey: info.server_pubkey,
        };
        Ok((stream_info, Box::pin(async {})))
    }

    async fn handle_commit(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.commit_adaptation(model_ref, ctx).await
    }

    async fn handle_rollback(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.rollback_adaptation(model_ref, ctx).await
    }

    async fn handle_reset(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.reset_delta(model_ref, ctx).await
    }

    async fn handle_status(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<GetDeltaStatusResponse> {
        let r = self.get_delta_status_forward(model_ref, ctx).await?;
                Ok(GetDeltaStatusResponse {
                    exists: r.exists,
                    accumulated_steps: r.accumulated_steps,
                    max_accumulated_steps: r.max_accumulated_steps,
                    request_count: r.request_count,
                    avg_loss_improvement: r.avg_loss_improvement,
                    memory_bytes: r.memory_bytes,
                    last_snapshot_hash: r.last_snapshot_hash,
                    delta_norm_ratios: r.delta_norm_ratios.into_iter().map(|d| ModuleNormRatio {
                        module_name: d.module_name,
                        ratio: d.ratio,
                    }).collect(),
                    has_pending: r.has_pending,
        })
    }

    async fn handle_save(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &SaveAdaptationRequest,
    ) -> Result<SaveAdaptationResponse> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        let r = client.gen.save_adaptation(
            &data.name, &data.merge_strategy, data.merge_weight, &data.commit_message,
        ).await?;
        Ok(SaveAdaptationResponse {
            adapter_name: r.adapter_name,
            adapter_path: r.adapter_path,
            content_hash: r.content_hash,
            merge_strategy: r.merge_strategy,
        })
    }

    async fn handle_snapshot(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<SnapshotDeltaResponse> {
        let r = self.snapshot_delta_forward(model_ref, ctx).await?;
        Ok(SnapshotDeltaResponse {
            content_hash: r.content_hash,
            size_bytes: r.size_bytes,
            accumulated_steps: r.accumulated_steps,
            request_count: r.request_count,
        })
    }

    async fn handle_export(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TttExportRequest,
    ) -> Result<TttExportResponse> {
        let r = self.export_peft_adapter_forward(model_ref, ctx, &data.name, &data.commit_message).await?;
        Ok(TttExportResponse {
            adapter_path: r.adapter_path,
            content_hash: r.content_hash,
        })
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
            gradient_steps: if data.gradient_steps > 0 { data.gradient_steps as usize } else { 3 },
            max_grad_norm: if data.max_grad_norm > 0.0 { data.max_grad_norm } else { 1.0 },
            min_input_length: if data.min_input_length > 0 { data.min_input_length as usize } else { 32 },
            max_ttt_context: if data.max_ttt_context > 0 { data.max_ttt_context as usize } else { 512 },
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
        wt.stage_files(&["config.json".to_owned()]).await?;
        wt.commit_with_author(
            "Update hyprstream_training config via RPC",
            "hyprstream",
            "noreply@hyprstream.dev",
        ).await?;

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
                self.load_model(model_ref, None).await?;
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
        model_ref: &str, data: &AdapterMergeRequest,
    ) -> Result<()> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        let weight = if data.weight > 0.0 { data.weight } else { 1.0 };
        let strategy = if data.strategy.is_empty() { "do_merge" } else { &data.strategy };
        client.merge_lora(&data.adapter_name, weight, strategy).await
    }
}

#[async_trait::async_trait(?Send)]
impl InferHandler for ModelService {
    async fn handle_generate_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &GenerateRequest,
    ) -> Result<(crate::services::generated::model_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let request = generate_request_from_data(data);
        let client = self.get_inference_client(model_ref, ctx).await?;
        let info = client.generate_stream(&request, ctx.ephemeral_pubkey).await?;
        let stream_info = crate::services::generated::model_client::StreamInfo {
            stream_id: info.stream_id,
            endpoint: info.endpoint,
            server_pubkey: info.server_pubkey,
        };
        Ok((stream_info, Box::pin(async {})))
    }

    async fn handle_apply_chat_template(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &ApplyChatTemplateRequest,
    ) -> Result<String> {
        let chat_messages: Vec<ChatMessage> = data.messages.iter().map(|m| {
            let tool_calls = if m.tool_calls.is_empty() {
                None
            } else {
                Some(m.tool_calls.iter().map(|tc| crate::api::openai_compat::ToolCall {
                    id: tc.id.clone(),
                    tool_type: tc.call_type.clone(),
                    function: crate::api::openai_compat::ToolCallFunction {
                        name: tc.function_name.clone(),
                        arguments: tc.arguments.clone(),
                    },
                }).collect())
            };
            ChatMessage {
                role: m.role.clone(),
                content: if m.content.is_empty() { None } else { Some(m.content.clone()) },
                function_call: None,
                tool_calls,
                tool_call_id: if m.tool_call_id.is_empty() { None } else { Some(m.tool_call_id.clone()) },
            }
        }).collect();
        // Parse tools from JSON string
        let tools: Option<serde_json::Value> = if data.tools_json.is_empty() {
            None
        } else {
            serde_json::from_str(&data.tools_json).ok()
        };
        let templated = self.apply_chat_template(
            model_ref, ctx, chat_messages, data.add_generation_prompt, tools.as_ref(),
        ).await?;
        Ok(templated.as_str().to_owned())
    }

    async fn handle_embed(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &EmbedRequest,
    ) -> Result<EmbedResponse> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        let embeddings = client.embed(&data.images).await?;
        let dimensions = embeddings.first().map(|v| v.len() as u32).unwrap_or(0);
        Ok(EmbedResponse {
            embeddings,
            dimensions,
        })
    }

    async fn handle_status(
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<ModelStatusResponse> {
        let status = self.model_status(model_ref).await;
                let config = status.online_training_config.map(|c| OnlineTrainingConfig {
                    enabled: c.enabled,
                    learning_rate: c.learning_rate,
                    gradient_steps: c.gradient_steps as u32,
                    max_grad_norm: c.max_grad_norm,
                    min_input_length: c.min_input_length as u32,
                    max_ttt_context: c.max_ttt_context as u32,
                }).unwrap_or_default();
                Ok(ModelStatusResponse {
                    loaded: status.loaded,
                    memory_bytes: 0,
                    session_count: 0,
                    endpoint: status.endpoint.unwrap_or_default(),
                    online_training_config: config,
        })
    }
}

#[async_trait::async_trait(?Send)]
impl ModelHandler for ModelService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        let subject = ctx.subject();
        let allowed = self.policy_client.check(&subject.to_string(), "*", resource, operation).await.unwrap_or_else(|e| {
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
        let max_ctx = match data.max_context {
            0 => None,
            n => Some(n as usize),
        };
        let kv_q = match data.kv_quant {
            KVQuantTypeEnum::Int8 => Some(KVQuantType::Int8),
            KVQuantTypeEnum::Nf4 => Some(KVQuantType::Nf4),
            KVQuantTypeEnum::Fp4 => Some(KVQuantType::Fp4),
            KVQuantTypeEnum::None => None,
        };
        let config = if max_ctx.is_some() || kv_q.is_some() {
            Some(ModelLoadConfig { max_context: max_ctx, kv_quant: kv_q, num_inference_instances: None })
        } else {
            None
        };
        let model_ref = &data.model_ref;
        match self.load_model(model_ref, config).await {
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
        Ok(ModelResponseVariant::StatusResult(entries.into_iter().map(|m| {
            let ttt = m.online_training_config.map(|c| GenOnlineTrainingConfig {
                enabled: c.enabled,
                learning_rate: c.learning_rate,
                gradient_steps: c.gradient_steps as u32,
                max_grad_norm: c.max_grad_norm,
                min_input_length: c.min_input_length as u32,
                max_ttt_context: c.max_ttt_context as u32,
            }).unwrap_or_default();
            GenModelStatusEntry {
                model_ref: m.model_ref,
                status: m.status,
                endpoint: m.endpoint,
                loaded_at: m.loaded_at,
                last_used: m.last_used,
                online_training_config: ttt,
            }
        }).collect()))
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
// Load request interception — parse capnp to detect load before dispatch
// ═══════════════════════════════════════════════════════════════════════════════

/// Parsed load request data extracted from Cap'n Proto payload.
struct ParsedLoadRequest {
    model_ref: String,
    max_context: u32,
    kv_quant: crate::model_capnp::KVQuantType,
}

impl ParsedLoadRequest {
    fn to_config(&self) -> Option<ModelLoadConfig> {
        use crate::model_capnp::KVQuantType as CKV;
        let max_ctx = match self.max_context {
            0 => None,
            n => Some(n as usize),
        };
        let kv_q = match self.kv_quant {
            CKV::Int8 => Some(KVQuantType::Int8),
            CKV::Nf4 => Some(KVQuantType::Nf4),
            CKV::Fp4 => Some(KVQuantType::Fp4),
            CKV::None => None,
        };
        if max_ctx.is_some() || kv_q.is_some() {
            Some(ModelLoadConfig { max_context: max_ctx, kv_quant: kv_q, num_inference_instances: None })
        } else {
            None
        }
    }
}

impl ModelService {
    /// Try to parse a load request from the raw Cap'n Proto payload.
    /// Returns `None` for all other request variants (list, unload, health, scoped, etc.).
    fn try_parse_load_request(payload: &[u8]) -> Option<(u64, ParsedLoadRequest)> {
        use crate::model_capnp::model_request;
        use crate::model_capnp::KVQuantType as CKV;
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
                let max_context = data.get_max_context();
                let kv_quant = data.get_kv_quant().unwrap_or(CKV::None);
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
            // Fast path: if already loaded, return immediately
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

            // Slow path: return "accepted" immediately, load in continuation
            let model_ref = load_data.model_ref.clone();
            info!("Load request accepted for {} (async)", model_ref);

            // Predict the endpoint so we can return it in the response
            let safe_name = model_ref.replace([':', '/', '\\'], "-");
            let service_name = format!("inference-{safe_name}");
            let predicted_endpoint = hyprstream_rpc::registry::global()
                .endpoint(&service_name, SocketKind::Rep)
                .to_zmq_string();

            let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                LoadedModelResponse {
                    model_ref: model_ref.clone(),
                    endpoint: predicted_endpoint,
                },
            ))?;

            // Mark as loading before spawning continuation
            self.pending_loads.lock().await.insert(model_ref.clone());

            let service = self.clone(); // Arc clone — cheap, 'static
            let config = load_data.to_config();
            let continuation: crate::services::Continuation = Box::pin(async move {
                let model_name = model_ref.split(':').next().unwrap_or(&model_ref);
                let scope = format!("serve:model:{}", model_name);
                match service.load_model(&model_ref, config).await {
                    Ok(endpoint) => {
                        service.pending_loads.lock().await.remove(&model_ref);
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
                        service.pending_loads.lock().await.remove(&model_ref);
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

/// Status entry for a single model (loaded or loading).
/// Absence from a status-all response means unloaded.
#[derive(Clone)]
pub struct ModelStatusEntry {
    pub model_ref: String,
    pub status: String,  // "loaded" | "loading"
    pub endpoint: String,
    pub loaded_at: i64,
    pub last_used: i64,
    pub online_training_config: Option<OnlineTrainingConfigInfo>,
}

/// Online training (TTT) configuration information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnlineTrainingConfigInfo {
    pub enabled: bool,
    pub learning_rate: f64,
    pub gradient_steps: usize,
    pub max_grad_norm: f64,
    pub min_input_length: usize,
    pub max_ttt_context: usize,
}

impl From<&crate::training::ttt::TTTConfig> for OnlineTrainingConfigInfo {
    fn from(config: &crate::training::ttt::TTTConfig) -> Self {
        Self {
            enabled: config.enabled,
            learning_rate: config.learning_rate,
            gradient_steps: config.gradient_steps,
            max_grad_norm: config.max_grad_norm,
            min_input_length: config.min_input_length,
            max_ttt_context: config.max_ttt_context,
        }
    }
}

/// Model status information
pub struct ModelStatusInfo {
    pub loaded: bool,
    pub endpoint: Option<String>,
    pub online_training_config: Option<OnlineTrainingConfigInfo>,
}

// ============================================================================
// ModelZmqClient (client-side)
// ============================================================================

/// Wraps a generated `ModelClient`. All methods delegate to the autogenerated
/// typed client which handles transport, serialization, and streaming.
#[derive(Clone)]
pub struct ModelZmqClient {
    /// Generated typed client (handles all transport including streaming via call_with_options)
    pub(crate) gen: crate::services::generated::model_client::ModelClient,
}


impl ModelZmqClient {
    /// Create a new model client (endpoint from registry)
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint("model", SocketKind::Rep).to_zmq_string();
        tracing::debug!("ModelZmqClient connecting to endpoint: {}", endpoint);
        Self::with_endpoint(&endpoint, signing_key, identity)
    }

    /// Create a model client at a specific endpoint
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            gen: crate::services::core::create_service_client(endpoint, signing_key, identity),
        }
    }

    /// Attach claims for e2e verification. All subsequent calls include these claims.
    pub fn with_claims(self, claims: hyprstream_rpc::auth::Claims) -> Self {
        Self { gen: self.gen.with_claims(claims) }
    }

    /// Load a model — delegates to generated client
    pub async fn load(&self, model_ref: &str, config: Option<&ModelLoadConfig>) -> Result<String> {
        let (max_context, kv_quant_str) = match config {
            Some(cfg) => {
                let max_ctx = cfg.max_context.unwrap_or(0) as u32;
                let kv_str = match cfg.kv_quant {
                    Some(KVQuantType::None) | None => "none",
                    Some(KVQuantType::Int8) => "int8",
                    Some(KVQuantType::Nf4) => "nf4",
                    Some(KVQuantType::Fp4) => "fp4",
                };
                (max_ctx, kv_str)
            }
            None => (0, "none"),
        };
        let data = self.gen.load(model_ref, max_context, kv_quant_str).await?;
        Ok(data.endpoint)
    }

    /// Unload a model
    pub async fn unload(&self, model_ref: &str) -> Result<()> {
        self.gen.unload(model_ref).await
    }

    /// Get status of all known models (model_ref = "") or a specific model.
    /// Absence from the result means the model is unloaded.
    pub async fn status(&self, model_ref: &str) -> Result<Vec<ModelStatusEntry>> {
        let data = self.gen.status(model_ref).await?;
        Ok(data.into_iter().map(|m| {
            let ttt = if m.online_training_config.enabled {
                Some(OnlineTrainingConfigInfo {
                    enabled: m.online_training_config.enabled,
                    learning_rate: m.online_training_config.learning_rate,
                    gradient_steps: m.online_training_config.gradient_steps as usize,
                    max_grad_norm: m.online_training_config.max_grad_norm,
                    min_input_length: m.online_training_config.min_input_length as usize,
                    max_ttt_context: m.online_training_config.max_ttt_context as usize,
                })
            } else {
                None
            };
            ModelStatusEntry {
                model_ref: m.model_ref,
                status: m.status,
                endpoint: m.endpoint,
                loaded_at: m.loaded_at,
                last_used: m.last_used,
                online_training_config: ttt,
            }
        }).collect())
    }

    /// Get model status (infer-scoped, for per-model detailed status)
    pub async fn infer_status(&self, model_ref: &str) -> Result<ModelStatusInfo> {
        let data = self.gen.infer(model_ref).status().await?;
        Ok(ModelStatusInfo {
            loaded: data.loaded,
            endpoint: if data.endpoint.is_empty() { None } else { Some(data.endpoint) },
            online_training_config: if data.online_training_config.enabled {
                Some(OnlineTrainingConfigInfo {
                    enabled: data.online_training_config.enabled,
                    learning_rate: data.online_training_config.learning_rate,
                    gradient_steps: data.online_training_config.gradient_steps as usize,
                    max_grad_norm: data.online_training_config.max_grad_norm,
                    min_input_length: data.online_training_config.min_input_length as usize,
                    max_ttt_context: data.online_training_config.max_ttt_context as usize,
                })
            } else {
                None
            },
        })
    }

    /// Start streaming inference with E2E authentication
    pub async fn infer_stream(
        &self,
        model_ref: &str,
        request: &GenerationRequest,
        client_ephemeral_pubkey: [u8; 32],
    ) -> Result<StreamInfo> {
        let images: Vec<Vec<u8>> = Vec::new();
        let info = self.gen.infer(model_ref).generate_stream(
            request.prompt.as_str(),
            request.max_tokens as u32,
            request.temperature,
            request.top_p,
            request.top_k.unwrap_or(0) as u32,
            request.repeat_penalty,
            request.repeat_last_n as u32,
            &request.stop_tokens,
            request.seed.unwrap_or(0),
            &images,
            request.timeout.unwrap_or(0),
            request.ttt_enabled,
            request.ttt_gradient_steps,
            request.ttt_learning_rate,
            request.auto_commit,
            client_ephemeral_pubkey,
        ).await?;
        Ok(StreamInfo {
            stream_id: info.stream_id,
            endpoint: info.endpoint,
            server_pubkey: info.server_pubkey,
        })
    }

    /// Health check
    pub async fn health_check(&self) -> Result<ModelHealthInfo> {
        let data = self.gen.health_check().await?;
        Ok(ModelHealthInfo {
            status: data.status,
            loaded_model_count: data.loaded_model_count,
            max_models: data.max_models,
            total_memory_bytes: data.total_memory_bytes,
        })
    }

    /// Apply chat template — delegates to generated client (infer-scoped)
    pub async fn apply_chat_template(
        &self,
        model_ref: &str,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
    ) -> Result<TemplatedPrompt> {
        let msg_data: Vec<crate::services::generated::model_client::ChatMessage> = messages.iter().map(|m| {
            use crate::services::generated::model_client::{ChatMessage as CapnpMsg, ToolCallData};
            CapnpMsg {
                role: m.role.clone(),
                content: m.content.as_deref().unwrap_or("").to_owned(),
                tool_calls: m.tool_calls.as_ref().map(|tcs| tcs.iter().map(|tc| ToolCallData {
                    id: tc.id.clone(),
                    call_type: tc.tool_type.clone(),
                    function_name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                }).collect()).unwrap_or_default(),
                tool_call_id: m.tool_call_id.as_deref().unwrap_or("").to_owned(),
            }
        }).collect();
        let tools_json = tools.map(|t| serde_json::to_string(t).unwrap_or_default())
            .unwrap_or_default();
        let prompt_str = self.gen.infer(model_ref).apply_chat_template(&msg_data, add_generation_prompt, &tools_json).await?;
        Ok(TemplatedPrompt::new(prompt_str))
    }

    /// Initialize LoRA training infrastructure on a loaded model (ttt-scoped)
    pub async fn create_lora(
        &self,
        model_ref: &str,
        rank: u32,
        alpha: f32,
        dropout: f32,
        target_modules: &[String],
        learning_rate: f32,
    ) -> Result<()> {
        self.gen.ttt(model_ref).init(rank, alpha, dropout, target_modules, learning_rate).await
    }

    /// Load a LoRA adapter into the base_delta register (adapter-scoped)
    pub async fn load_lora(&self, model_ref: &str, path: &str) -> Result<()> {
        self.gen.adapter(model_ref).load(path).await
    }

    /// Save the current LoRA adapter to a file
    /// Note: Legacy — use ttt.save or ttt.export for delta persistence.
    pub async fn save_lora(&self, model_ref: &str, path: &str) -> Result<()> {
        let client = self.gen.infer(model_ref);
        let _result = client.call_method("save_lora", &serde_json::json!({"value": path})).await;
        Ok(())
    }

    /// Unload the current LoRA adapter from the base_delta register (adapter-scoped)
    pub async fn unload_lora(&self, model_ref: &str) -> Result<()> {
        self.gen.adapter(model_ref).unload().await
    }

    /// Check if a LoRA adapter is loaded in the base_delta register (adapter-scoped)
    pub async fn has_lora(&self, model_ref: &str) -> Result<bool> {
        self.gen.adapter(model_ref).status().await
    }

    /// Start streaming training step with E2E authentication
    pub async fn train_step_stream(
        &self,
        model_ref: &str,
        input: &str,
        gradient_steps: u32,
        learning_rate: f32,
        auto_commit: bool,
        client_ephemeral_pubkey: [u8; 32],
    ) -> Result<StreamInfo> {
        let info = self.gen.ttt(model_ref).train_stream(
            input,
            gradient_steps,
            learning_rate,
            auto_commit,
            client_ephemeral_pubkey,
        ).await?;
        Ok(StreamInfo {
            stream_id: info.stream_id,
            endpoint: info.endpoint,
            server_pubkey: info.server_pubkey,
        })
    }

}

/// Health information from the model service
#[derive(Debug, Clone)]
pub struct ModelHealthInfo {
    pub status: String,
    pub loaded_model_count: u32,
    pub max_models: u32,
    pub total_memory_bytes: u64,
}

// ============================================================================
// Serialization helpers
// ============================================================================

/// Convert GenerateRequest (typed, from scope handler) to GenerationRequest
fn generate_request_from_data(data: &GenerateRequest) -> GenerationRequest {
    GenerationRequest {
        prompt: TemplatedPrompt::new(data.prompt.clone()),
        max_tokens: data.max_tokens as usize,
        temperature: data.temperature,
        top_p: data.top_p,
        top_k: if data.top_k > 0 { Some(data.top_k as usize) } else { None },
        repeat_penalty: data.repeat_penalty,
        repeat_last_n: data.repeat_last_n as usize,
        seed: if data.seed > 0 { Some(data.seed) } else { None },
        stop_tokens: data.stop_tokens.clone(),
        timeout: if data.timeout_ms > 0 { Some(data.timeout_ms) } else { None },
        images: Vec::new(),
        collect_metrics: false,
        ttt_enabled: data.ttt_enabled,
        ttt_gradient_steps: data.ttt_gradient_steps,
        ttt_learning_rate: data.ttt_learning_rate,
        auto_commit: data.auto_commit,
    }
}



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
