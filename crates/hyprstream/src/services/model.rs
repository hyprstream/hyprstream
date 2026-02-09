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

use crate::api::openai_compat::ChatMessage;
use crate::config::{GenerationRequest, GenerationResult, TemplatedPrompt};
use crate::model_capnp;
use crate::runtime::kv_quant::KVQuantType;
use crate::runtime::RuntimeConfig;
use crate::services::{
    rpc_types::StreamStartedInfo, CallOptions, EnvelopeContext, InferenceService, InferenceZmqClient,
    PolicyClient,
};
use crate::services::RegistryClient;
use crate::storage::ModelRef;
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::transport::TransportConfig;
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info};

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

/// Model service that manages InferenceService lifecycle
///
/// Runs on multi-threaded runtime. Manages an LRU cache of loaded models,
/// spawning and stopping InferenceService instances as needed.
///
/// Supports two modes:
/// - InProcess: Runs InferenceService in the same process (default)
/// - Spawned: Spawns InferenceService as separate process via callback pattern
pub struct ModelService {
    // Business logic
    /// LRU cache of loaded models
    loaded_models: RwLock<LruCache<String, LoadedModel>>,
    /// Service configuration
    config: ModelServiceConfig,
    /// Ed25519 signing key for creating InferenceZmqClients
    signing_key: SigningKey,
    /// Policy client for authorization checks in InferenceService
    policy_client: PolicyClient,
    /// Registry client for resolving model paths
    registry: Arc<dyn RegistryClient>,
    /// Callback router for spawned mode (None for in-process)
    #[allow(dead_code)]
    callback_router: Option<crate::services::callback::CallbackRouter>,
    /// Spawned instances by model ref (for spawned mode)
    #[allow(dead_code)]
    spawned_instances: RwLock<HashMap<String, crate::services::callback::Instance>>,
    // Infrastructure (for Spawnable)
    context: Arc<zmq::Context>,
    transport: TransportConfig,
}

impl ModelService {
    /// Create a new model service with infrastructure
    pub fn new(
        config: ModelServiceConfig,
        signing_key: SigningKey,
        policy_client: PolicyClient,
        registry: Arc<dyn RegistryClient>,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        // SAFETY: 5 is a valid non-zero value
        const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(5) {
            Some(n) => n,
            None => unreachable!(),
        };
        let cache_size = NonZeroUsize::new(config.max_models).unwrap_or(DEFAULT_CACHE_SIZE);

        Self {
            loaded_models: RwLock::new(LruCache::new(cache_size)),
            config,
            signing_key,
            policy_client,
            registry,
            callback_router: None,
            spawned_instances: RwLock::new(HashMap::new()),
            context,
            transport,
        }
    }

    /// Create a model service with callback router for spawned mode
    pub fn with_callback_router(
        config: ModelServiceConfig,
        signing_key: SigningKey,
        policy_client: PolicyClient,
        registry: Arc<dyn RegistryClient>,
        callback_router: crate::services::callback::CallbackRouter,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(5) {
            Some(n) => n,
            None => unreachable!(),
        };
        let cache_size = NonZeroUsize::new(config.max_models).unwrap_or(DEFAULT_CACHE_SIZE);

        Self {
            loaded_models: RwLock::new(LruCache::new(cache_size)),
            config,
            signing_key,
            policy_client,
            registry,
            callback_router: Some(callback_router),
            spawned_instances: RwLock::new(HashMap::new()),
            context,
            transport,
        }
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
        let model_path = self.registry.get_model_path(&model_ref).await?;

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
        let mut runtime_config = RuntimeConfig::default();
        runtime_config.max_context = load_config.max_context.or(self.config.max_context);
        runtime_config.kv_quant_type = load_config.kv_quant.unwrap_or(self.config.kv_quant);

        // Obtain FsOps from the registry for path-contained adapter I/O
        let branch_name = match &model_ref.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            _ => "main".to_owned(),
        };
        let repo_client = self.registry.repo(&model_ref.model).await
            .map_err(|e| anyhow::anyhow!(
                "Could not get repository client for {}: {} — FsOps required for path containment",
                model_ref_str, e
            ))?;
        let fs: Option<std::sync::Arc<dyn crate::services::FsOps>> = Some(repo_client.worktree(&branch_name));

        // Start InferenceService for this model
        let service_handle = InferenceService::start_at(
            &model_path,
            runtime_config,
            self.signing_key.verifying_key(),
            self.signing_key.clone(),
            self.policy_client.clone(),
            &endpoint,
            fs,
        )
        .await?;

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
            Ok(())
        } else {
            Err(anyhow!("Model {} is not loaded", model_ref_str))
        }
    }

    /// List loaded models
    async fn list_models(&self) -> Vec<LoadedModelInfo> {
        let cache = self.loaded_models.read().await;
        cache
            .iter()
            .map(|(_, model)| LoadedModelInfo {
                model_ref: model.model_ref.clone(),
                endpoint: model.endpoint.clone(),
                loaded_at: model.loaded_at.elapsed().as_millis() as i64,
                last_used: model.last_used.elapsed().as_millis() as i64,
            })
            .collect()
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

    /// Route inference request to the appropriate InferenceService
    async fn infer(&self, model_ref_str: &str, request: GenerationRequest) -> Result<GenerationResult> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.generate(&request).await
    }

    /// Route streaming inference request with E2E authentication.
    ///
    /// The ephemeral pubkey from the client's envelope is passed through to InferenceService.
    async fn infer_stream(
        &self,
        model_ref_str: &str,
        request: GenerationRequest,
        client_ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamStartedInfo> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.generate_stream(&request, client_ephemeral_pubkey).await
    }

    /// Authorize stream subscription via StartStream handshake
    ///
    /// Client must call this after infer_stream() to authorize the SUB subscription.
    /// This sends an AUTHORIZE message to StreamService via InferenceService.
    async fn start_stream(&self, model_ref_str: &str, stream_id: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.start_stream(stream_id).await.map(|_| ())
    }

    /// Helper: Get inference client for a model (ensures loaded, updates last_used)
    async fn get_inference_client(&self, model_ref_str: &str) -> Result<InferenceZmqClient> {
        let _endpoint = self.load_model(model_ref_str, None).await?;
        let mut cache = self.loaded_models.write().await;
        let model = cache
            .get_mut(model_ref_str)
            .ok_or_else(|| anyhow!("Model {} not found after loading", model_ref_str))?;
        model.last_used = Instant::now();
        Ok(model.client.clone())
    }

    /// Apply chat template via the model's InferenceService
    async fn apply_chat_template(
        &self,
        model_ref_str: &str,
        messages: Vec<ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<TemplatedPrompt> {
        let client = self.get_inference_client(model_ref_str).await?;

        // Convert ChatMessage to the template engine's format
        let template_messages: Vec<crate::runtime::template_engine::ChatMessage> = messages
            .iter()
            .map(|m| crate::runtime::template_engine::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone().unwrap_or_default(),
            })
            .collect();

        // Call InferenceService's apply_chat_template
        let prompt_str = client.apply_chat_template(&template_messages, add_generation_prompt).await?;

        Ok(TemplatedPrompt::new(prompt_str))
    }

    /// Create a new LoRA adapter on the loaded model
    async fn create_lora(&self, model_ref_str: &str, config: crate::training::TenantDeltaConfig) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.create_lora(&config).await
    }

    /// Load a LoRA adapter from a file
    async fn load_lora(&self, model_ref_str: &str, path: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.load_lora(path).await
    }

    /// Unload the current LoRA adapter
    async fn unload_lora(&self, model_ref_str: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.unload_lora().await
    }

    /// Check if a LoRA adapter is loaded
    async fn has_lora(&self, model_ref_str: &str) -> Result<bool> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.has_lora().await
    }

    // Training loop control - forward to InferenceService via ZMQ
    async fn commit_adaptation(&self, model_ref_str: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.commit_adaptation().await
    }

    async fn rollback_adaptation(&self, model_ref_str: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.rollback_adaptation().await
    }

    async fn train_step_forward(
        &self,
        model_ref_str: &str,
        train_req: model_capnp::train_step_request::Reader<'_>,
    ) -> Result<crate::services::generated::inference_client::TrainStepResultData> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.gen.train_step(
            train_req.get_input()?.to_str()?,
            train_req.get_gradient_steps(),
            train_req.get_learning_rate(),
            train_req.get_auto_commit(),
        ).await
    }

    /// Route streaming training step request with E2E authentication.
    ///
    /// Similar to infer_stream — forwards to InferenceService with ephemeral pubkey
    /// for DH key exchange. Returns stream info for result subscription.
    async fn train_step_stream(
        &self,
        model_ref_str: &str,
        input: &str,
        gradient_steps: u32,
        learning_rate: f32,
        auto_commit: bool,
        client_ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamStartedInfo> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.train_step_stream(input, gradient_steps, learning_rate, auto_commit, client_ephemeral_pubkey).await
    }

    async fn reset_delta(&self, model_ref_str: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.reset_delta().await
    }

    async fn get_delta_status_forward(
        &self,
        model_ref_str: &str,
    ) -> Result<crate::services::generated::inference_client::DeltaStatusResultData> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.gen.get_delta_status().await
    }

    async fn save_adaptation_forward(
        &self,
        model_ref_str: &str,
        save_req: model_capnp::save_adaptation_request::Reader<'_>,
    ) -> Result<crate::services::generated::inference_client::SaveAdaptationResultData> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.gen.save_adaptation(
            save_req.get_name()?.to_str()?,
            save_req.get_merge_strategy()?.to_str()?,
            save_req.get_merge_weight(),
            save_req.get_commit_message()?.to_str()?,
        ).await
    }

    async fn snapshot_delta_forward(
        &self,
        model_ref_str: &str,
    ) -> Result<crate::services::generated::inference_client::SnapshotDeltaResultData> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.gen.snapshot_delta().await
    }

    // ========================================================================
    // Response builders — top-level
    // ========================================================================

    /// Build a load result response
    fn build_load_result_response(request_id: u64, model_ref: &str, endpoint: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut loaded = response.init_load_result();
            loaded.set_model_ref(model_ref);
            loaded.set_endpoint(endpoint);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build an unload result response
    fn build_unload_result_response(request_id: u64) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            response.set_unload_result(());
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a list result response
    fn build_list_result_response(request_id: u64, models: Vec<LoadedModelInfo>) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let list = response.init_list_result();
            let mut model_list = list.init_models(models.len() as u32);
            for (i, model) in models.iter().enumerate() {
                let mut m = model_list.reborrow().get(i as u32);
                m.set_model_ref(&model.model_ref);
                m.set_endpoint(&model.endpoint);
                m.set_loaded_at(model.loaded_at);
                m.set_last_used(model.last_used);
                m.set_memory_bytes(0); // TODO: track memory usage
                m.set_session_count(0); // TODO: track session count
            }
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a health check result response
    fn build_health_check_result_response(request_id: u64, loaded_count: u32, max_models: u32) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut health = response.init_health_check_result();
            health.set_status("healthy");
            health.set_loaded_model_count(loaded_count);
            health.set_max_models(max_models);
            health.set_total_memory_bytes(0);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a top-level error response
    fn build_error_response(request_id: u64, message_text: &str, code: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut error = response.init_error();
            error.set_message(message_text);
            error.set_code(code);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    // ========================================================================
    // Response builders — TTT scoped
    // ========================================================================

    fn build_ttt_error_response(request_id: u64, message_text: &str, code: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let ttt_resp = response.init_ttt_result();
            let mut error = ttt_resp.init_error();
            error.set_message(message_text);
            error.set_code(code);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_ttt_void_response(request_id: u64, method_name: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut ttt_resp = response.init_ttt_result();
            match method_name {
                "create" => ttt_resp.set_create(()),
                "commit" => ttt_resp.set_commit(()),
                "rollback" => ttt_resp.set_rollback(()),
                "reset" => ttt_resp.set_reset(()),
                _ => return Err(anyhow!("Unknown TTT void method: {}", method_name)),
            }
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_ttt_train_response(
        request_id: u64,
        data: &crate::services::generated::inference_client::TrainStepResultData,
    ) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let ttt_resp = response.init_ttt_result();
            let mut ts = ttt_resp.init_train();
            ts.set_avg_loss(data.avg_loss);
            ts.set_loss_improvement(data.loss_improvement);
            ts.set_steps_performed(data.steps_performed);
            ts.set_adaptation_time_ms(data.adaptation_time_ms);
            ts.set_initial_perplexity(data.initial_perplexity);
            ts.set_final_perplexity(data.final_perplexity);
            ts.set_recommendation(data.recommendation);
            ts.set_committed(data.committed);
            ts.set_gradient_clipped(data.gradient_clipped);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_ttt_train_stream_response(
        request_id: u64,
        stream_id: &str,
        endpoint: &str,
        server_pubkey: &[u8; 32],
    ) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let ttt_resp = response.init_ttt_result();
            let mut stream_info = ttt_resp.init_train_stream();
            stream_info.set_stream_id(stream_id);
            stream_info.set_endpoint(endpoint);
            stream_info.set_server_pubkey(server_pubkey);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_ttt_status_response(
        request_id: u64,
        data: &crate::services::generated::inference_client::DeltaStatusResultData,
    ) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let ttt_resp = response.init_ttt_result();
            let mut ds = ttt_resp.init_status();
            ds.set_exists(data.exists);
            ds.set_accumulated_steps(data.accumulated_steps);
            ds.set_max_accumulated_steps(data.max_accumulated_steps);
            ds.set_request_count(data.request_count);
            ds.set_avg_loss_improvement(data.avg_loss_improvement);
            ds.set_memory_bytes(data.memory_bytes);
            ds.set_last_snapshot_hash(&data.last_snapshot_hash);
            ds.set_has_pending(data.has_pending);
            let mut ratios = ds.init_delta_norm_ratios(data.delta_norm_ratios.len() as u32);
            for (i, r) in data.delta_norm_ratios.iter().enumerate() {
                let mut entry = ratios.reborrow().get(i as u32);
                entry.set_module_name(&r.module_name);
                entry.set_ratio(r.ratio);
            }
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_ttt_save_response(
        request_id: u64,
        data: &crate::services::generated::inference_client::SaveAdaptationResultData,
    ) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let ttt_resp = response.init_ttt_result();
            let mut sa = ttt_resp.init_save();
            sa.set_adapter_name(&data.adapter_name);
            sa.set_adapter_path(&data.adapter_path);
            sa.set_content_hash(&data.content_hash);
            sa.set_merge_strategy(&data.merge_strategy);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_ttt_snapshot_response(
        request_id: u64,
        data: &crate::services::generated::inference_client::SnapshotDeltaResultData,
    ) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let ttt_resp = response.init_ttt_result();
            let mut sd = ttt_resp.init_snapshot();
            sd.set_content_hash(&data.content_hash);
            sd.set_size_bytes(data.size_bytes);
            sd.set_accumulated_steps(data.accumulated_steps);
            sd.set_request_count(data.request_count);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    // ========================================================================
    // Response builders — PEFT scoped
    // ========================================================================

    fn build_peft_error_response(request_id: u64, message_text: &str, code: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let peft_resp = response.init_peft_result();
            let mut error = peft_resp.init_error();
            error.set_message(message_text);
            error.set_code(code);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_peft_void_response(request_id: u64, method_name: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut peft_resp = response.init_peft_result();
            match method_name {
                "load" => peft_resp.set_load(()),
                "unload" => peft_resp.set_unload(()),
                "merge" => peft_resp.set_merge(()),
                _ => return Err(anyhow!("Unknown PEFT void method: {}", method_name)),
            }
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_peft_has_response(request_id: u64, has_lora: bool) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut peft_resp = response.init_peft_result();
            peft_resp.set_has(has_lora);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    // ========================================================================
    // Response builders — Infer scoped
    // ========================================================================

    fn build_infer_error_response(request_id: u64, message_text: &str, code: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let infer_resp = response.init_infer_result();
            let mut error = infer_resp.init_error();
            error.set_message(message_text);
            error.set_code(code);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_infer_generate_response(request_id: u64, result: &GenerationResult) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let infer_resp = response.init_infer_result();
            let mut infer = infer_resp.init_generate();
            infer.set_text(&result.text);
            infer.set_tokens_generated(result.tokens_generated as u32);
            infer.set_finish_reason(&finish_reason_to_str(&result.finish_reason));
            infer.set_generation_time_ms(result.generation_time_ms);
            infer.set_tokens_per_second(result.tokens_per_second);
            infer.set_prefill_tokens(result.prefill_tokens as u32);
            infer.set_prefill_time_ms(result.prefill_time_ms);
            infer.set_prefill_tokens_per_sec(result.prefill_tokens_per_sec);
            infer.set_inference_tokens(result.inference_tokens as u32);
            infer.set_inference_time_ms(result.inference_time_ms);
            infer.set_inference_tokens_per_sec(result.inference_tokens_per_sec);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_infer_generate_stream_response(
        request_id: u64,
        stream_id: &str,
        endpoint: &str,
        server_pubkey: &[u8; 32],
    ) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let infer_resp = response.init_infer_result();
            let mut stream_info = infer_resp.init_generate_stream();
            stream_info.set_stream_id(stream_id);
            stream_info.set_endpoint(endpoint);
            stream_info.set_server_pubkey(server_pubkey);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_infer_start_stream_response(request_id: u64, stream_id: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let infer_resp = response.init_infer_result();
            let mut auth_info = infer_resp.init_start_stream();
            auth_info.set_stream_id(stream_id);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_infer_template_response(request_id: u64, templated_prompt: &TemplatedPrompt) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut infer_resp = response.init_infer_result();
            infer_resp.set_apply_chat_template(templated_prompt.as_str());
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_infer_status_response(request_id: u64, status: ModelStatusInfo) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let infer_resp = response.init_infer_result();
            let mut status_builder = infer_resp.init_status();
            status_builder.set_loaded(status.loaded);
            status_builder.set_memory_bytes(0);
            status_builder.set_session_count(0);
            if let Some(endpoint) = status.endpoint {
                status_builder.set_endpoint(&endpoint);
            }

            // Populate online training config if available
            if let Some(ttt_config) = status.online_training_config {
                let mut config_builder = status_builder.init_online_training_config();
                config_builder.set_enabled(ttt_config.enabled);
                config_builder.set_learning_rate(ttt_config.learning_rate);
                config_builder.set_gradient_steps(ttt_config.gradient_steps as u32);
                config_builder.set_max_grad_norm(ttt_config.max_grad_norm);
                config_builder.set_min_input_length(ttt_config.min_input_length as u32);
                config_builder.set_max_ttt_context(ttt_config.max_ttt_context as u32);
            }
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }
}

/// ZmqService implementation for ModelService
impl crate::services::ZmqService for ModelService {
    fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
        debug!(
            "Model request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );

        // Parse request
        let reader = serialize::read_message(&mut std::io::Cursor::new(payload), ReaderOptions::new())?;
        let req = reader.get_root::<model_capnp::model_request::Reader>()?;
        let request_id = req.get_id();

        // Extract ephemeral pubkey for E2E streaming (needed inside async block)
        let client_ephemeral_pubkey = ctx.ephemeral_pubkey;

        // Handle request in blocking context (we need async operations)
        let result = tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::current();
            handle.block_on(async {
                use model_capnp::model_request::Which;
                match req.which()? {
                    Which::Load(load_req) => {
                        let load = load_req?;
                        let model_ref = load.get_model_ref()?.to_str()?;

                        // Extract optional runtime config
                        let max_context = match load.get_max_context() {
                            0 => None,
                            n => Some(n as usize),
                        };
                        let kv_quant = match load.get_kv_quant()? {
                            model_capnp::KVQuantType::None => None,
                            model_capnp::KVQuantType::Int8 => Some(KVQuantType::Int8),
                            model_capnp::KVQuantType::Nf4 => Some(KVQuantType::Nf4),
                            model_capnp::KVQuantType::Fp4 => Some(KVQuantType::Fp4),
                        };
                        let config = if max_context.is_some() || kv_quant.is_some() {
                            Some(ModelLoadConfig { max_context, kv_quant })
                        } else {
                            None
                        };

                        match self.load_model(model_ref, config).await {
                            Ok(endpoint) => Self::build_load_result_response(request_id, model_ref, &endpoint),
                            Err(e) => Self::build_error_response(
                                request_id,
                                &format!("Failed to load model: {e}"),
                                "LOAD_FAILED",
                            ),
                        }
                    }
                    Which::Unload(unload_req) => {
                        let unload = unload_req?;
                        let model_ref = unload.get_model_ref()?.to_str()?;
                        match self.unload_model(model_ref).await {
                            Ok(()) => Self::build_unload_result_response(request_id),
                            Err(e) => Self::build_error_response(
                                request_id,
                                &format!("Failed to unload model: {e}"),
                                "UNLOAD_FAILED",
                            ),
                        }
                    }
                    Which::List(()) => {
                        let models = self.list_models().await;
                        Self::build_list_result_response(request_id, models)
                    }
                    Which::HealthCheck(()) => {
                        let cache = self.loaded_models.read().await;
                        let loaded_count = cache.len() as u32;
                        let max_models = self.config.max_models as u32;
                        drop(cache);
                        Self::build_health_check_result_response(request_id, loaded_count, max_models)
                    }

                    // ==========================================================
                    // TTT scoped operations
                    // ==========================================================
                    Which::Ttt(ttt_req) => {
                        let ttt = ttt_req?;
                        let model_ref = ttt.get_model_ref()?.to_str()?;

                        use model_capnp::ttt_request::Which as TttWhich;
                        match ttt.which()? {
                            TttWhich::Create(lora_req) => {
                                let lora = lora_req?;
                                let config = crate::training::TenantDeltaConfig {
                                    rank: lora.get_rank() as usize,
                                    alpha: lora.get_alpha(),
                                    dropout: lora.get_dropout(),
                                    target_modules: lora.get_target_modules()?.iter()
                                        .filter_map(|s| s.ok().and_then(|t| t.to_str().ok().map(|s| s.to_owned())))
                                        .collect(),
                                    learning_rate: lora.get_learning_rate() as f64,
                                    ..crate::training::TenantDeltaConfig::default()
                                };
                                match self.create_lora(model_ref, config).await {
                                    Ok(()) => Self::build_ttt_void_response(request_id, "create"),
                                    Err(e) => Self::build_ttt_error_response(
                                        request_id,
                                        &format!("LoRA creation failed: {e}"),
                                        "LORA_CREATE_FAILED",
                                    ),
                                }
                            }
                            TttWhich::Train(req) => {
                                let train_req = req?;
                                match self.train_step_forward(model_ref, train_req).await {
                                    Ok(info) => Self::build_ttt_train_response(request_id, &info),
                                    Err(e) => Self::build_ttt_error_response(
                                        request_id,
                                        &format!("Train step failed: {e}"),
                                        "TRAIN_STEP_FAILED",
                                    ),
                                }
                            }
                            TttWhich::TrainStream(req) => {
                                let train_req = req?;
                                let input = train_req.get_input()?.to_str()?;
                                let gradient_steps = train_req.get_gradient_steps();
                                let learning_rate = train_req.get_learning_rate();
                                let auto_commit = train_req.get_auto_commit();

                                match self.train_step_stream(
                                    model_ref, input, gradient_steps, learning_rate,
                                    auto_commit, client_ephemeral_pubkey,
                                ).await {
                                    Ok(info) => Self::build_ttt_train_stream_response(
                                        request_id,
                                        &info.stream_id,
                                        &info.endpoint,
                                        &info.server_pubkey,
                                    ),
                                    Err(e) => Self::build_ttt_error_response(
                                        request_id,
                                        &format!("Train step stream failed: {e}"),
                                        "TRAIN_STREAM_FAILED",
                                    ),
                                }
                            }
                            TttWhich::Commit(_) => {
                                match self.commit_adaptation(model_ref).await {
                                    Ok(()) => Self::build_ttt_void_response(request_id, "commit"),
                                    Err(e) => Self::build_ttt_error_response(
                                        request_id,
                                        &format!("Commit failed: {e}"),
                                        "COMMIT_FAILED",
                                    ),
                                }
                            }
                            TttWhich::Rollback(_) => {
                                match self.rollback_adaptation(model_ref).await {
                                    Ok(()) => Self::build_ttt_void_response(request_id, "rollback"),
                                    Err(e) => Self::build_ttt_error_response(
                                        request_id,
                                        &format!("Rollback failed: {e}"),
                                        "ROLLBACK_FAILED",
                                    ),
                                }
                            }
                            TttWhich::Reset(_) => {
                                match self.reset_delta(model_ref).await {
                                    Ok(()) => Self::build_ttt_void_response(request_id, "reset"),
                                    Err(e) => Self::build_ttt_error_response(
                                        request_id,
                                        &format!("Reset failed: {e}"),
                                        "RESET_FAILED",
                                    ),
                                }
                            }
                            TttWhich::Status(_) => {
                                match self.get_delta_status_forward(model_ref).await {
                                    Ok(info) => Self::build_ttt_status_response(request_id, &info),
                                    Err(e) => Self::build_ttt_error_response(
                                        request_id,
                                        &format!("Status query failed: {e}"),
                                        "STATUS_FAILED",
                                    ),
                                }
                            }
                            TttWhich::Save(req) => {
                                let save_req = req?;
                                match self.save_adaptation_forward(model_ref, save_req).await {
                                    Ok(info) => Self::build_ttt_save_response(request_id, &info),
                                    Err(e) => Self::build_ttt_error_response(
                                        request_id,
                                        &format!("Save failed: {e}"),
                                        "SAVE_FAILED",
                                    ),
                                }
                            }
                            TttWhich::Snapshot(_) => {
                                match self.snapshot_delta_forward(model_ref).await {
                                    Ok(info) => Self::build_ttt_snapshot_response(request_id, &info),
                                    Err(e) => Self::build_ttt_error_response(
                                        request_id,
                                        &format!("Snapshot failed: {e}"),
                                        "SNAPSHOT_FAILED",
                                    ),
                                }
                            }
                            TttWhich::Export(_export_req) => {
                                // TODO: Implement ttt.export (delta → PEFT adapter directory)
                                Self::build_ttt_error_response(
                                    request_id,
                                    "ttt.export not yet implemented",
                                    "NOT_IMPLEMENTED",
                                )
                            }
                        }
                    }

                    // ==========================================================
                    // PEFT scoped operations
                    // ==========================================================
                    Which::Peft(peft_req) => {
                        let peft = peft_req?;
                        let model_ref = peft.get_model_ref()?.to_str()?;

                        use model_capnp::peft_request::Which as PeftWhich;
                        match peft.which()? {
                            PeftWhich::Load(path_req) => {
                                let path = path_req?.to_str()?;
                                match self.load_lora(model_ref, path).await {
                                    Ok(()) => Self::build_peft_void_response(request_id, "load"),
                                    Err(e) => Self::build_peft_error_response(
                                        request_id,
                                        &format!("PEFT load failed: {e}"),
                                        "PEFT_LOAD_FAILED",
                                    ),
                                }
                            }
                            PeftWhich::Unload(()) => {
                                match self.unload_lora(model_ref).await {
                                    Ok(()) => Self::build_peft_void_response(request_id, "unload"),
                                    Err(e) => Self::build_peft_error_response(
                                        request_id,
                                        &format!("PEFT unload failed: {e}"),
                                        "PEFT_UNLOAD_FAILED",
                                    ),
                                }
                            }
                            PeftWhich::Has(()) => {
                                match self.has_lora(model_ref).await {
                                    Ok(has) => Self::build_peft_has_response(request_id, has),
                                    Err(e) => Self::build_peft_error_response(
                                        request_id,
                                        &format!("PEFT check failed: {e}"),
                                        "PEFT_CHECK_FAILED",
                                    ),
                                }
                            }
                            PeftWhich::Check(_path_req) => {
                                // TODO: Implement peft.check (validate PEFT adapter directory)
                                Self::build_peft_error_response(
                                    request_id,
                                    "peft.check not yet implemented",
                                    "NOT_IMPLEMENTED",
                                )
                            }
                            PeftWhich::Merge(_merge_req) => {
                                // TODO: Implement peft.merge (merge adapter into base weights)
                                Self::build_peft_error_response(
                                    request_id,
                                    "peft.merge not yet implemented",
                                    "NOT_IMPLEMENTED",
                                )
                            }
                        }
                    }

                    // ==========================================================
                    // Infer scoped operations
                    // ==========================================================
                    Which::Infer(infer_req) => {
                        let infer = infer_req?;
                        let model_ref = infer.get_model_ref()?.to_str()?;

                        use model_capnp::infer_request::Which as InferWhich;
                        match infer.which()? {
                            InferWhich::Generate(gen_req) => {
                                let gen = gen_req?;
                                let request = parse_generate_request(gen)?;
                                match self.infer(model_ref, request).await {
                                    Ok(result) => Self::build_infer_generate_response(request_id, &result),
                                    Err(e) => Self::build_infer_error_response(
                                        request_id,
                                        &format!("Inference failed: {e}"),
                                        "INFER_FAILED",
                                    ),
                                }
                            }
                            InferWhich::GenerateStream(gen_req) => {
                                let gen = gen_req?;
                                let request = parse_generate_request(gen)?;
                                match self.infer_stream(model_ref, request, client_ephemeral_pubkey).await {
                                    Ok(info) => Self::build_infer_generate_stream_response(
                                        request_id,
                                        &info.stream_id,
                                        &info.endpoint,
                                        &info.server_pubkey,
                                    ),
                                    Err(e) => Self::build_infer_error_response(
                                        request_id,
                                        &format!("Stream start failed: {e}"),
                                        "STREAM_FAILED",
                                    ),
                                }
                            }
                            InferWhich::StartStream(start_req) => {
                                let start = start_req?;
                                let stream_id = start.get_stream_id()?.to_str()?;
                                match self.start_stream(model_ref, stream_id).await {
                                    Ok(()) => Self::build_infer_start_stream_response(request_id, stream_id),
                                    Err(e) => Self::build_infer_error_response(
                                        request_id,
                                        &format!("Stream authorization failed: {e}"),
                                        "STREAM_AUTH_FAILED",
                                    ),
                                }
                            }
                            InferWhich::ApplyChatTemplate(template_req) => {
                                let template = template_req?;
                                let add_generation_prompt = template.get_add_generation_prompt();

                                // Parse messages from Cap'n Proto
                                let messages_reader = template.get_messages()?;
                                let messages: Vec<ChatMessage> = messages_reader
                                    .iter()
                                    .map(|m| ChatMessage {
                                        role: m.get_role().map(|r| r.to_str().unwrap_or("user").to_owned()).unwrap_or_else(|_| "user".to_owned()),
                                        content: m.get_content().ok().map(|c| c.to_str().unwrap_or("").to_owned()),
                                        function_call: None,
                                    })
                                    .collect();

                                match self.apply_chat_template(model_ref, messages, add_generation_prompt).await {
                                    Ok(templated) => Self::build_infer_template_response(request_id, &templated),
                                    Err(e) => Self::build_infer_error_response(
                                        request_id,
                                        &format!("Template application failed: {e}"),
                                        "TEMPLATE_FAILED",
                                    ),
                                }
                            }
                            InferWhich::Status(()) => {
                                let info = self.model_status(model_ref).await;
                                Self::build_infer_status_response(request_id, info)
                            }
                        }
                    }
                }
            })
        });

        result
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
}

// ============================================================================
// Helper types
// ============================================================================

/// Information about a loaded model (for list response)
#[derive(Clone)]
pub struct LoadedModelInfo {
    pub model_ref: String,
    pub endpoint: String,
    pub loaded_at: i64,
    pub last_used: i64,
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

/// Wraps a generated `ModelClient`. Most methods delegate directly to gen
/// which returns typed results. Streaming methods (`infer_stream`, `start_stream`)
/// use manual request building for custom `CallOptions` support.
#[derive(Clone)]
pub struct ModelZmqClient {
    /// Generated typed client (handles all transport including streaming via call_with_options)
    pub(crate) gen: crate::services::generated::model_client::ModelClient,
}

use crate::services::generated::model_client::{
    InferClient as GenInferClient,
    InferClientResponseVariant,
    TttClient as GenTttClient,
    TttClientResponseVariant,
};

impl ModelZmqClient {
    /// Create a new model client (endpoint from registry)
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint("model", SocketKind::Rep).to_zmq_string();
        tracing::debug!("ModelZmqClient connecting to endpoint: {}", endpoint);
        Self::with_endpoint(&endpoint, signing_key, identity)
    }

    /// Create a model client at a specific endpoint
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        use hyprstream_rpc::service::factory::ServiceClient;
        let server_verifying_key = signing_key.verifying_key();
        let base = crate::services::core::ZmqClientBase::new(
            endpoint, crate::zmq::global_context(), signing_key, server_verifying_key, identity,
        );
        Self {
            gen: crate::services::generated::model_client::ModelClient::from_zmq(base),
        }
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

    /// List loaded models — delegates to generated client
    pub async fn list(&self) -> Result<Vec<LoadedModelInfo>> {
        let data = self.gen.list().await?;
        Ok(data.models.into_iter().map(|m| LoadedModelInfo {
            model_ref: m.model_ref,
            endpoint: m.endpoint,
            loaded_at: m.loaded_at,
            last_used: m.last_used,
        }).collect())
    }

    /// Get model status (infer-scoped)
    pub async fn status(&self, model_ref: &str) -> Result<ModelStatusInfo> {
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

    /// Run inference on a model (infer-scoped)
    pub async fn infer(&self, model_ref: &str, request: &GenerationRequest) -> Result<GenerationResult> {
        // Images are file paths in GenerationRequest but raw bytes in schema — not yet used over wire
        let images: Vec<Vec<u8>> = Vec::new();
        let data = self.gen.infer(model_ref).generate(
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
            false,  // tttEnabled: use server default
            0,      // tttGradientSteps: use server default
            0.0,    // tttLearningRate: use server default
            false,  // autoCommit: default false
        ).await?;
        Ok(GenerationResult {
            text: data.text,
            tokens_generated: data.tokens_generated as usize,
            finish_reason: parse_finish_reason_str(&data.finish_reason),
            generation_time_ms: data.generation_time_ms,
            tokens_per_second: data.tokens_per_second,
            quality_metrics: None,
            prefill_tokens: data.prefill_tokens as usize,
            prefill_time_ms: data.prefill_time_ms,
            prefill_tokens_per_sec: data.prefill_tokens_per_sec,
            inference_tokens: data.inference_tokens as usize,
            inference_time_ms: data.inference_time_ms,
            inference_tokens_per_sec: data.inference_tokens_per_sec,
            ttt_metrics: None,  // TODO: Extract from response when available
        })
    }

    /// Start streaming inference with E2E authentication (manual — needs custom CallOptions)
    pub async fn infer_stream(
        &self,
        model_ref: &str,
        request: &GenerationRequest,
        client_ephemeral_pubkey: [u8; 32],
    ) -> Result<StreamStartedInfo> {
        let request_id = self.gen.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut infer = req.init_infer();
            infer.set_model_ref(model_ref);
            let mut gen = infer.init_generate_stream();
            set_generate_request_fields(&mut gen, request);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let opts = CallOptions::default().ephemeral_pubkey(client_ephemeral_pubkey);
        let response_bytes = self.gen.call_with_options(request_bytes, opts).await?;
        Self::parse_infer_generate_stream_response(&response_bytes)
    }

    /// Authorize a stream subscription (manual — infer-scoped streaming)
    pub async fn start_stream(&self, model_ref: &str, stream_id: &str) -> Result<String> {
        let request_id = self.gen.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut infer = req.init_infer();
            infer.set_model_ref(model_ref);
            let mut start_req = infer.init_start_stream();
            start_req.set_stream_id(stream_id);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.gen.call_with_options(request_bytes, CallOptions::default()).await?;
        Self::parse_infer_start_stream_response(&response_bytes)
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
    ) -> Result<TemplatedPrompt> {
        use crate::services::generated::model_client::ChatMessageData;
        let msg_data: Vec<ChatMessageData> = messages.iter().map(|m| ChatMessageData {
            role: m.role.clone(),
            content: m.content.as_deref().unwrap_or("").to_string(),
        }).collect();
        let prompt_str = self.gen.infer(model_ref).apply_chat_template(&msg_data, add_generation_prompt).await?;
        Ok(TemplatedPrompt::new(prompt_str))
    }

    /// Create a new LoRA adapter on a loaded model (ttt-scoped)
    pub async fn create_lora(
        &self,
        model_ref: &str,
        rank: u32,
        alpha: f32,
        dropout: f32,
        target_modules: &[String],
        learning_rate: f32,
    ) -> Result<()> {
        self.gen.ttt(model_ref).create(rank, alpha, dropout, target_modules, learning_rate).await
    }

    /// Load a LoRA adapter from a file (peft-scoped)
    pub async fn load_lora(&self, model_ref: &str, path: &str) -> Result<()> {
        self.gen.peft(model_ref).load(path).await
    }

    /// Save the current LoRA adapter to a file (peft-scoped)
    /// Note: For backward compat, this delegates to peft.load with save semantics.
    /// Use ttt.save for TTT delta persistence.
    pub async fn save_lora(&self, model_ref: &str, path: &str) -> Result<()> {
        // saveLora was removed from the schema — this is kept for CLI backward compat
        // by forwarding to the inference service's save_lora directly
        let client = self.gen.infer(model_ref);
        // Use call_method for backward compat until the inference schema is updated
        let _result = client.call_method("save_lora", &serde_json::json!({"value": path})).await;
        Ok(())
    }

    /// Unload the current LoRA adapter (peft-scoped)
    pub async fn unload_lora(&self, model_ref: &str) -> Result<()> {
        self.gen.peft(model_ref).unload().await
    }

    /// Check if a LoRA adapter is loaded (peft-scoped)
    pub async fn has_lora(&self, model_ref: &str) -> Result<bool> {
        self.gen.peft(model_ref).has().await
    }

    /// Start streaming training step with E2E authentication (manual — needs custom CallOptions)
    pub async fn train_step_stream(
        &self,
        model_ref: &str,
        input: &str,
        gradient_steps: u32,
        learning_rate: f32,
        auto_commit: bool,
        client_ephemeral_pubkey: [u8; 32],
    ) -> Result<StreamStartedInfo> {
        let request_id = self.gen.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut ttt = req.init_ttt();
            ttt.set_model_ref(model_ref);
            let mut train_req = ttt.init_train_stream();
            train_req.set_input(input);
            train_req.set_gradient_steps(gradient_steps);
            train_req.set_learning_rate(learning_rate);
            train_req.set_auto_commit(auto_commit);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let opts = CallOptions::default().ephemeral_pubkey(client_ephemeral_pubkey);
        let response_bytes = self.gen.call_with_options(request_bytes, opts).await?;
        Self::parse_ttt_train_stream_response(&response_bytes)
    }

    // ========================================================================
    // Streaming response parsers
    // ========================================================================

    /// Parse infer generate stream response
    fn parse_infer_generate_stream_response(bytes: &[u8]) -> Result<StreamStartedInfo> {
        match GenInferClient::parse_scoped_response(bytes)? {
            InferClientResponseVariant::GenerateStream { stream_id, endpoint, server_pubkey } => {
                Ok(StreamStartedInfo {
                    stream_id,
                    endpoint,
                    server_pubkey: server_pubkey.try_into().unwrap_or([0u8; 32]),
                })
            }
            InferClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected infer response type")),
        }
    }

    /// Parse infer start stream response
    fn parse_infer_start_stream_response(bytes: &[u8]) -> Result<String> {
        match GenInferClient::parse_scoped_response(bytes)? {
            InferClientResponseVariant::StartStream { stream_id, .. } => Ok(stream_id),
            InferClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected infer response type")),
        }
    }

    /// Parse ttt train step stream response
    fn parse_ttt_train_stream_response(bytes: &[u8]) -> Result<StreamStartedInfo> {
        match GenTttClient::parse_scoped_response(bytes)? {
            TttClientResponseVariant::TrainStream { stream_id, endpoint, server_pubkey } => {
                Ok(StreamStartedInfo {
                    stream_id,
                    endpoint,
                    server_pubkey: server_pubkey.try_into().unwrap_or([0u8; 32]),
                })
            }
            TttClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected ttt response type")),
        }
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

/// Parse inline GenerateRequest fields from model.capnp into GenerationRequest
fn parse_generate_request(
    reader: model_capnp::generate_request::Reader,
) -> Result<GenerationRequest> {
    Ok(GenerationRequest {
        prompt: TemplatedPrompt::new(reader.get_prompt()?.to_str()?.to_owned()),
        max_tokens: reader.get_max_tokens() as usize,
        temperature: reader.get_temperature(),
        top_p: reader.get_top_p(),
        top_k: if reader.get_top_k() > 0 {
            Some(reader.get_top_k() as usize)
        } else {
            None
        },
        repeat_penalty: reader.get_repeat_penalty(),
        repeat_last_n: reader.get_repeat_last_n() as usize,
        seed: if reader.get_seed() > 0 {
            Some(reader.get_seed() as u32)
        } else {
            None
        },
        stop_tokens: reader
            .get_stop_tokens()?
            .iter()
            .filter_map(|t| t.ok().map(|s| s.to_string().unwrap_or_default()))
            .collect(),
        timeout: if reader.get_timeout_ms() > 0 {
            Some(reader.get_timeout_ms())
        } else {
            None
        },
        images: Vec::new(),
        collect_metrics: false,
    })
}

/// Convert FinishReason to string for InferResult.finishReason field
fn finish_reason_to_str(reason: &crate::config::FinishReason) -> String {
    match reason {
        crate::config::FinishReason::MaxTokens => "max_tokens".to_owned(),
        crate::config::FinishReason::StopToken(t) => format!("stop_token:{}", t),
        crate::config::FinishReason::EndOfSequence => "end_of_sequence".to_owned(),
        crate::config::FinishReason::Error(e) => format!("error:{}", e),
        crate::config::FinishReason::Stop => "stop".to_owned(),
    }
}

/// Set GenerateRequest fields from a GenerationRequest (for manual message building)
fn set_generate_request_fields(
    gen: &mut model_capnp::generate_request::Builder,
    request: &GenerationRequest,
) {
    gen.set_prompt(request.prompt.as_str());
    gen.set_max_tokens(request.max_tokens as u32);
    gen.set_temperature(request.temperature);
    gen.set_top_p(request.top_p);
    gen.set_top_k(request.top_k.unwrap_or(0) as u32);
    gen.set_repeat_penalty(request.repeat_penalty);
    gen.set_repeat_last_n(request.repeat_last_n as u32);
    gen.set_seed(request.seed.unwrap_or(0));
    gen.set_timeout_ms(request.timeout.unwrap_or(0));

    if !request.stop_tokens.is_empty() {
        let mut stop_list = gen.reborrow().init_stop_tokens(request.stop_tokens.len() as u32);
        for (i, token) in request.stop_tokens.iter().enumerate() {
            stop_list.set(i as u32, token);
        }
    }
}

/// Parse a finish reason string back into FinishReason enum
fn parse_finish_reason_str(s: &str) -> crate::config::FinishReason {
    if s.starts_with("stop_token:") {
        crate::config::FinishReason::StopToken(s.strip_prefix("stop_token:").unwrap_or("").to_owned())
    } else if s.starts_with("error:") {
        crate::config::FinishReason::Error(s.strip_prefix("error:").unwrap_or("").to_owned())
    } else {
        match s {
            "max_tokens" => crate::config::FinishReason::MaxTokens,
            "end_of_sequence" => crate::config::FinishReason::EndOfSequence,
            "stop" => crate::config::FinishReason::Stop,
            _ => crate::config::FinishReason::Stop,
        }
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
