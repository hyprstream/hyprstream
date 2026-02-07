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
        let base = registry().endpoint("inference", SocketKind::Rep).to_zmq_string();
        let safe_name = model_ref_str.replace([':', '/', '\\'], "-");
        let endpoint = format!("{base}/{safe_name}");

        info!("Loading model {} at endpoint {}", model_ref_str, endpoint);

        // Create runtime config - use per-model config if provided, otherwise service defaults
        let load_config = config.unwrap_or_default();
        let mut runtime_config = RuntimeConfig::default();
        runtime_config.max_context = load_config.max_context.or(self.config.max_context);
        runtime_config.kv_quant_type = load_config.kv_quant.unwrap_or(self.config.kv_quant);

        // Start InferenceService for this model
        let service_handle = InferenceService::start_at(
            &model_path,
            runtime_config,
            self.signing_key.verifying_key(),
            self.signing_key.clone(),
            self.policy_client.clone(),
            &endpoint,
        )
        .await?;

        // Create client for this service
        let client = InferenceZmqClient::with_endpoint(
            &endpoint,
            self.signing_key.clone(),
            RequestIdentity::local(),
        );

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
            }
        } else {
            ModelStatusInfo {
                loaded: false,
                endpoint: None,
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
    async fn create_lora(&self, model_ref_str: &str, config: crate::lora::LoRAConfig) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.create_lora(&config).await
    }

    /// Load a LoRA adapter from a file
    async fn load_lora(&self, model_ref_str: &str, path: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.load_lora(std::path::Path::new(path)).await
    }

    /// Save the current LoRA adapter to a file
    async fn save_lora(&self, model_ref_str: &str, path: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.save_lora(path).await
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

    /// Build a session status response (nested inside sessionResult)
    fn build_session_status_response(request_id: u64, status: ModelStatusInfo) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let session_resp = response.init_session_result();
            let mut status_builder = session_resp.init_status();
            status_builder.set_loaded(status.loaded);
            status_builder.set_memory_bytes(0);
            status_builder.set_session_count(0);
            if let Some(endpoint) = status.endpoint {
                status_builder.set_endpoint(&endpoint);
            }
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a session infer result response (nested inside sessionResult)
    fn build_session_infer_response(request_id: u64, result: &GenerationResult) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let session_resp = response.init_session_result();
            let mut infer = session_resp.init_infer();
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

    /// Build a session infer stream response (nested inside sessionResult)
    fn build_session_infer_stream_response(
        request_id: u64,
        stream_id: &str,
        endpoint: &str,
        server_pubkey: &[u8; 32],
    ) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let session_resp = response.init_session_result();
            let mut stream_info = session_resp.init_infer_stream();
            stream_info.set_stream_id(stream_id);
            stream_info.set_endpoint(endpoint);
            stream_info.set_server_pubkey(server_pubkey);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a session start stream response (nested inside sessionResult)
    fn build_session_start_stream_response(request_id: u64, stream_id: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let session_resp = response.init_session_result();
            let mut auth_info = session_resp.init_start_stream();
            auth_info.set_stream_id(stream_id);
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

    /// Build an error response
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

    /// Build a session error response (nested inside sessionResult)
    fn build_session_error_response(request_id: u64, message_text: &str, code: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let session_resp = response.init_session_result();
            let mut error = session_resp.init_error();
            error.set_message(message_text);
            error.set_code(code);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a session template result response (nested inside sessionResult)
    fn build_session_template_response(request_id: u64, templated_prompt: &TemplatedPrompt) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut session_resp = response.init_session_result();
            session_resp.set_apply_chat_template(templated_prompt.as_str());
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a session Void response (for createLora, loadLora, saveLora, unloadLora)
    fn build_session_void_response(request_id: u64, method_name: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut session_resp = response.init_session_result();
            match method_name {
                "createLora" => session_resp.set_create_lora(()),
                "loadLora" => session_resp.set_load_lora(()),
                "saveLora" => session_resp.set_save_lora(()),
                "unloadLora" => session_resp.set_unload_lora(()),
                _ => return Err(anyhow!("Unknown void method: {}", method_name)),
            }
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a session hasLora response
    fn build_session_has_lora_response(request_id: u64, has_lora: bool) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut session_resp = response.init_session_result();
            session_resp.set_has_lora(has_lora);
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
            ctx.casbin_subject(),
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
                    Which::Session(session_req) => {
                        let session = session_req?;
                        let model_ref = session.get_model_ref()?.to_str()?;

                        use model_capnp::model_session_request::Which as SessionWhich;
                        match session.which()? {
                            SessionWhich::Status(()) => {
                                let info = self.model_status(model_ref).await;
                                Self::build_session_status_response(request_id, info)
                            }
                            SessionWhich::Infer(infer_req) => {
                                let infer = infer_req?;
                                let request = parse_infer_request(infer)?;
                                match self.infer(model_ref, request).await {
                                    Ok(result) => Self::build_session_infer_response(request_id, &result),
                                    Err(e) => Self::build_session_error_response(
                                        request_id,
                                        &format!("Inference failed: {e}"),
                                        "INFER_FAILED",
                                    ),
                                }
                            }
                            SessionWhich::InferStream(infer_req) => {
                                let infer = infer_req?;
                                let request = parse_infer_request(infer)?;
                                match self.infer_stream(model_ref, request, client_ephemeral_pubkey).await {
                                    Ok(info) => Self::build_session_infer_stream_response(
                                        request_id,
                                        &info.stream_id,
                                        &info.endpoint,
                                        &info.server_pubkey,
                                    ),
                                    Err(e) => Self::build_session_error_response(
                                        request_id,
                                        &format!("Stream start failed: {e}"),
                                        "STREAM_FAILED",
                                    ),
                                }
                            }
                            SessionWhich::StartStream(start_req) => {
                                let start = start_req?;
                                let stream_id = start.get_stream_id()?.to_str()?;
                                match self.start_stream(model_ref, stream_id).await {
                                    Ok(()) => Self::build_session_start_stream_response(request_id, stream_id),
                                    Err(e) => Self::build_session_error_response(
                                        request_id,
                                        &format!("Stream authorization failed: {e}"),
                                        "STREAM_AUTH_FAILED",
                                    ),
                                }
                            }
                            SessionWhich::ApplyChatTemplate(template_req) => {
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
                                    Ok(templated) => Self::build_session_template_response(request_id, &templated),
                                    Err(e) => Self::build_session_error_response(
                                        request_id,
                                        &format!("Template application failed: {e}"),
                                        "TEMPLATE_FAILED",
                                    ),
                                }
                            }
                            SessionWhich::CreateLora(lora_req) => {
                                let lora = lora_req?;
                                let config = crate::lora::LoRAConfig {
                                    rank: lora.get_rank() as usize,
                                    alpha: lora.get_alpha(),
                                    dropout: lora.get_dropout(),
                                    target_modules: lora.get_target_modules()?.iter()
                                        .filter_map(|s| s.ok().and_then(|t| t.to_str().ok().map(|s| s.to_owned())))
                                        .collect(),
                                    learning_rate: lora.get_learning_rate(),
                                };
                                match self.create_lora(model_ref, config).await {
                                    Ok(()) => Self::build_session_void_response(request_id, "createLora"),
                                    Err(e) => Self::build_session_error_response(
                                        request_id,
                                        &format!("LoRA creation failed: {e}"),
                                        "LORA_CREATE_FAILED",
                                    ),
                                }
                            }
                            SessionWhich::LoadLora(path_req) => {
                                let path = path_req?.to_str()?;
                                match self.load_lora(model_ref, path).await {
                                    Ok(()) => Self::build_session_void_response(request_id, "loadLora"),
                                    Err(e) => Self::build_session_error_response(
                                        request_id,
                                        &format!("LoRA load failed: {e}"),
                                        "LORA_LOAD_FAILED",
                                    ),
                                }
                            }
                            SessionWhich::SaveLora(path_req) => {
                                let path = path_req?.to_str()?;
                                match self.save_lora(model_ref, path).await {
                                    Ok(()) => Self::build_session_void_response(request_id, "saveLora"),
                                    Err(e) => Self::build_session_error_response(
                                        request_id,
                                        &format!("LoRA save failed: {e}"),
                                        "LORA_SAVE_FAILED",
                                    ),
                                }
                            }
                            SessionWhich::UnloadLora(()) => {
                                match self.unload_lora(model_ref).await {
                                    Ok(()) => Self::build_session_void_response(request_id, "unloadLora"),
                                    Err(e) => Self::build_session_error_response(
                                        request_id,
                                        &format!("LoRA unload failed: {e}"),
                                        "LORA_UNLOAD_FAILED",
                                    ),
                                }
                            }
                            SessionWhich::HasLora(()) => {
                                match self.has_lora(model_ref).await {
                                    Ok(has_lora) => Self::build_session_has_lora_response(request_id, has_lora),
                                    Err(e) => Self::build_session_error_response(
                                        request_id,
                                        &format!("LoRA check failed: {e}"),
                                        "LORA_CHECK_FAILED",
                                    ),
                                }
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

/// Model status information
pub struct ModelStatusInfo {
    pub loaded: bool,
    pub endpoint: Option<String>,
}

// ============================================================================
// ModelZmqClient (client-side)
// ============================================================================

/// Wraps a generated `ModelClient`. Simple methods delegate to gen;
/// `load`, `list`, `apply_chat_template`, and streaming methods use manual
/// request building for types the generator doesn't fully support.
#[derive(Clone)]
pub struct ModelZmqClient {
    /// Generated typed client (handles all transport including streaming via call_with_options)
    pub(crate) gen: crate::services::generated::model_client::ModelClient,
}

use crate::services::generated::model_client::{
    ModelResponseVariant, ModelSessionClient as GenModelSessionClient,
    ModelSessionClientResponseVariant,
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
        match self.gen.load(model_ref, max_context, kv_quant_str).await? {
            ModelResponseVariant::LoadResult { endpoint, .. } => Ok(endpoint),
            ModelResponseVariant::Error { message, code, .. } => Err(anyhow!("{}: {}", code, message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Unload a model
    pub async fn unload(&self, model_ref: &str) -> Result<()> {
        match self.gen.unload(model_ref).await? {
            ModelResponseVariant::UnloadResult => Ok(()),
            ModelResponseVariant::Error { message, code, .. } => Err(anyhow!("{}: {}", code, message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// List loaded models — delegates to generated client
    pub async fn list(&self) -> Result<Vec<LoadedModelInfo>> {
        match self.gen.list().await? {
            ModelResponseVariant::ListResult { models } => {
                Ok(models.into_iter().map(|m| LoadedModelInfo {
                    model_ref: m.model_ref,
                    endpoint: m.endpoint,
                    loaded_at: m.loaded_at,
                    last_used: m.last_used,
                }).collect())
            }
            ModelResponseVariant::Error { message, code, .. } => Err(anyhow!("{}: {}", code, message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Get model status (session-scoped)
    pub async fn status(&self, model_ref: &str) -> Result<ModelStatusInfo> {
        match self.gen.session(model_ref).status().await? {
            ModelSessionClientResponseVariant::Status { loaded, endpoint, .. } => {
                Ok(ModelStatusInfo {
                    loaded,
                    endpoint: if endpoint.is_empty() { None } else { Some(endpoint) },
                })
            }
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
        }
    }

    /// Run inference on a model (session-scoped)
    pub async fn infer(&self, model_ref: &str, request: &GenerationRequest) -> Result<GenerationResult> {
        // Images are file paths in GenerationRequest but raw bytes in schema — not yet used over wire
        let images: Vec<Vec<u8>> = Vec::new();
        match self.gen.session(model_ref).infer(
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
        ).await? {
            ModelSessionClientResponseVariant::Infer {
                text,
                tokens_generated,
                finish_reason,
                generation_time_ms,
                tokens_per_second,
                prefill_tokens,
                prefill_time_ms,
                prefill_tokens_per_sec,
                inference_tokens,
                inference_time_ms,
                inference_tokens_per_sec,
            } => {
                Ok(GenerationResult {
                    text,
                    tokens_generated: tokens_generated as usize,
                    finish_reason: parse_finish_reason_str(&finish_reason),
                    generation_time_ms,
                    tokens_per_second,
                    quality_metrics: None,
                    prefill_tokens: prefill_tokens as usize,
                    prefill_time_ms,
                    prefill_tokens_per_sec,
                    inference_tokens: inference_tokens as usize,
                    inference_time_ms,
                    inference_tokens_per_sec,
                })
            }
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
        }
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
            let mut session = req.init_session();
            session.set_model_ref(model_ref);
            let mut infer = session.init_infer_stream();
            set_infer_request_fields(&mut infer, request);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let opts = CallOptions::default().ephemeral_pubkey(client_ephemeral_pubkey);
        let response_bytes = self.gen.call_with_options(request_bytes, opts).await?;
        Self::parse_session_infer_stream_response(&response_bytes)
    }

    /// Authorize a stream subscription (manual — session-scoped streaming)
    pub async fn start_stream(&self, model_ref: &str, stream_id: &str) -> Result<String> {
        let request_id = self.gen.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut session = req.init_session();
            session.set_model_ref(model_ref);
            let mut start_req = session.init_start_stream();
            start_req.set_stream_id(stream_id);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.gen.call_with_options(request_bytes, CallOptions::default()).await?;
        Self::parse_session_start_stream_response(&response_bytes)
    }

    /// Health check
    pub async fn health_check(&self) -> Result<ModelHealthInfo> {
        match self.gen.health_check().await? {
            ModelResponseVariant::HealthCheckResult { status, loaded_model_count, max_models, total_memory_bytes } => {
                Ok(ModelHealthInfo { status, loaded_model_count, max_models, total_memory_bytes })
            }
            ModelResponseVariant::Error { message, code, .. } => Err(anyhow!("{}: {}", code, message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Apply chat template — delegates to generated client
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
        match self.gen.session(model_ref).apply_chat_template(&msg_data, add_generation_prompt).await? {
            ModelSessionClientResponseVariant::ApplyChatTemplate(prompt_str) => {
                Ok(TemplatedPrompt::new(prompt_str))
            }
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
        }
    }

    /// Create a new LoRA adapter on a loaded model (session-scoped)
    pub async fn create_lora(
        &self,
        model_ref: &str,
        rank: u32,
        alpha: f32,
        dropout: f32,
        target_modules: &[String],
        learning_rate: f32,
    ) -> Result<()> {
        match self.gen.session(model_ref).create_lora(rank, alpha, dropout, target_modules, learning_rate).await? {
            ModelSessionClientResponseVariant::CreateLora => Ok(()),
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
        }
    }

    /// Load a LoRA adapter from a file (session-scoped)
    pub async fn load_lora(&self, model_ref: &str, path: &str) -> Result<()> {
        match self.gen.session(model_ref).load_lora(path).await? {
            ModelSessionClientResponseVariant::LoadLora => Ok(()),
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
        }
    }

    /// Save the current LoRA adapter to a file (session-scoped)
    pub async fn save_lora(&self, model_ref: &str, path: &str) -> Result<()> {
        match self.gen.session(model_ref).save_lora(path).await? {
            ModelSessionClientResponseVariant::SaveLora => Ok(()),
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
        }
    }

    /// Unload the current LoRA adapter (session-scoped)
    pub async fn unload_lora(&self, model_ref: &str) -> Result<()> {
        match self.gen.session(model_ref).unload_lora().await? {
            ModelSessionClientResponseVariant::UnloadLora => Ok(()),
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
        }
    }

    /// Check if a LoRA adapter is loaded (session-scoped)
    pub async fn has_lora(&self, model_ref: &str) -> Result<bool> {
        match self.gen.session(model_ref).has_lora().await? {
            ModelSessionClientResponseVariant::HasLora(has_lora) => Ok(has_lora),
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
        }
    }

    // ========================================================================
    // Streaming response parsers (kept for manual streaming methods)
    // ========================================================================

    /// Parse session infer stream response — uses generated scoped response parser
    fn parse_session_infer_stream_response(bytes: &[u8]) -> Result<StreamStartedInfo> {
        match GenModelSessionClient::parse_scoped_response(bytes)? {
            ModelSessionClientResponseVariant::InferStream { stream_id, endpoint, server_pubkey } => {
                Ok(StreamStartedInfo {
                    stream_id,
                    endpoint,
                    server_pubkey: server_pubkey.try_into().unwrap_or([0u8; 32]),
                })
            }
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
        }
    }

    /// Parse session start stream response — uses generated scoped response parser
    fn parse_session_start_stream_response(bytes: &[u8]) -> Result<String> {
        match GenModelSessionClient::parse_scoped_response(bytes)? {
            ModelSessionClientResponseVariant::StartStream { stream_id, .. } => Ok(stream_id),
            ModelSessionClientResponseVariant::Error { message, code, .. } => {
                Err(anyhow!("{}: {}", code, message))
            }
            _ => Err(anyhow!("Unexpected session response type")),
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

/// Parse inline InferRequest fields from model.capnp into GenerationRequest
fn parse_infer_request(
    reader: model_capnp::infer_request::Reader,
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

/// Set InferRequest fields from a GenerationRequest (for manual message building)
fn set_infer_request_fields(
    infer: &mut model_capnp::infer_request::Builder,
    request: &GenerationRequest,
) {
    infer.set_prompt(request.prompt.as_str());
    infer.set_max_tokens(request.max_tokens as u32);
    infer.set_temperature(request.temperature);
    infer.set_top_p(request.top_p);
    infer.set_top_k(request.top_k.unwrap_or(0) as u32);
    infer.set_repeat_penalty(request.repeat_penalty);
    infer.set_repeat_last_n(request.repeat_last_n as u32);
    infer.set_seed(request.seed.unwrap_or(0));
    infer.set_timeout_ms(request.timeout.unwrap_or(0));

    if !request.stop_tokens.is_empty() {
        let mut stop_list = infer.reborrow().init_stop_tokens(request.stop_tokens.len() as u32);
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
