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
    PolicyZmqClient, ZmqClient,
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

/// Base endpoint for inference services (per-model endpoints are suffixed)
const INFERENCE_ENDPOINT_BASE: &str = "inproc://hyprstream/inference";

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
    policy_client: PolicyZmqClient,
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
        policy_client: PolicyZmqClient,
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
        policy_client: PolicyZmqClient,
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

        // Create unique endpoint for this model
        let safe_name = model_ref_str.replace([':', '/', '\\'], "-");
        let endpoint = format!("{INFERENCE_ENDPOINT_BASE}/{safe_name}");

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
    async fn infer(&self, model_ref_str: &str, request_bytes: &[u8]) -> Result<Vec<u8>> {
        // Ensure model is loaded (use service defaults)
        let _endpoint = self.load_model(model_ref_str, None).await?;

        // Get client
        let client = {
            let mut cache = self.loaded_models.write().await;
            let model = cache
                .get_mut(model_ref_str)
                .ok_or_else(|| anyhow!("Model {} not found after loading", model_ref_str))?;
            model.last_used = Instant::now();
            model.client.clone()
        };

        // Parse the GenerationRequest
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(request_bytes),
            ReaderOptions::new(),
        )?;
        let gen_req_reader =
            reader.get_root::<crate::inference_capnp::generation_request::Reader>()?;

        let request = parse_generation_request(gen_req_reader)?;

        // Call inference
        let result = client.generate(&request).await?;

        // Serialize result
        serialize_generation_result(&result)
    }

    /// Route streaming inference request with E2E authentication.
    ///
    /// The ephemeral pubkey from the client's envelope is passed through to InferenceService.
    async fn infer_stream(
        &self,
        model_ref_str: &str,
        request_bytes: &[u8],
        client_ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamStartedInfo> {
        // Ensure model is loaded (use service defaults)
        let _endpoint = self.load_model(model_ref_str, None).await?;

        // Get client
        let client = {
            let mut cache = self.loaded_models.write().await;
            let model = cache
                .get_mut(model_ref_str)
                .ok_or_else(|| anyhow!("Model {} not found after loading", model_ref_str))?;
            model.last_used = Instant::now();
            model.client.clone()
        };

        // Parse the GenerationRequest
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(request_bytes),
            ReaderOptions::new(),
        )?;
        let gen_req_reader =
            reader.get_root::<crate::inference_capnp::generation_request::Reader>()?;

        let request = parse_generation_request(gen_req_reader)?;

        // Pass through ephemeral pubkey for E2E authentication
        client.generate_stream(&request, client_ephemeral_pubkey).await
    }

    /// Authorize stream subscription via StartStream handshake
    ///
    /// Client must call this after infer_stream() to authorize the SUB subscription.
    /// This sends an AUTHORIZE message to StreamService via InferenceService.
    async fn start_stream(&self, model_ref_str: &str, stream_id: &str) -> Result<()> {
        // Ensure model is loaded (use service defaults)
        let _endpoint = self.load_model(model_ref_str, None).await?;

        // Get client
        let client = {
            let mut cache = self.loaded_models.write().await;
            let model = cache
                .get_mut(model_ref_str)
                .ok_or_else(|| anyhow!("Model {} not found after loading", model_ref_str))?;
            model.last_used = Instant::now();
            model.client.clone()
        };

        // Authorize stream via InferenceService
        client.start_stream(stream_id).await.map(|_| ())
    }

    /// Apply chat template via the model's InferenceService
    async fn apply_chat_template(
        &self,
        model_ref_str: &str,
        messages: Vec<ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<TemplatedPrompt> {
        // Ensure model is loaded (use service defaults)
        let _endpoint = self.load_model(model_ref_str, None).await?;

        // Get client
        let client = {
            let mut cache = self.loaded_models.write().await;
            let model = cache
                .get_mut(model_ref_str)
                .ok_or_else(|| anyhow!("Model {} not found after loading", model_ref_str))?;
            model.last_used = Instant::now();
            model.client.clone()
        };

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

    /// Build an OK response
    fn build_ok_response(request_id: u64) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            response.set_ok(());
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a loaded response
    fn build_loaded_response(request_id: u64, model_ref: &str, endpoint: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut loaded = response.init_loaded();
            loaded.set_model_ref(model_ref);
            loaded.set_endpoint(endpoint);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a list response
    fn build_list_response(request_id: u64, models: Vec<LoadedModelInfo>) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let list = response.init_list();
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

    /// Build a status response
    fn build_status_response(request_id: u64, status: ModelStatusInfo) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut status_builder = response.init_status();
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

    /// Build an infer result response
    fn build_infer_result_response(request_id: u64, result_bytes: Vec<u8>) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            response.set_infer_result(&result_bytes);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a stream started response (with server pubkey for E2E auth)
    fn build_stream_started_response(
        request_id: u64,
        stream_id: &str,
        endpoint: &str,
        server_pubkey: &[u8; 32],
    ) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut stream_info = response.init_stream_started();
            stream_info.set_stream_id(stream_id);
            stream_info.set_endpoint(endpoint);
            stream_info.set_server_pubkey(server_pubkey);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a stream authorized response
    fn build_stream_authorized_response(request_id: u64, stream_id: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut auth_info = response.init_stream_authorized();
            auth_info.set_stream_id(stream_id);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build a health response
    fn build_health_response(request_id: u64, loaded_count: u32, max_models: u32) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            let mut health = response.init_health();
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

    /// Build a template result response
    fn build_template_result_response(request_id: u64, templated_prompt: &TemplatedPrompt) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<model_capnp::model_response::Builder>();
            response.set_request_id(request_id);
            response.set_template_result(templated_prompt.as_str());
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
                            Ok(endpoint) => Self::build_loaded_response(request_id, model_ref, &endpoint),
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
                            Ok(()) => Self::build_ok_response(request_id),
                            Err(e) => Self::build_error_response(
                                request_id,
                                &format!("Failed to unload model: {e}"),
                                "UNLOAD_FAILED",
                            ),
                        }
                    }
                    Which::List(()) => {
                        let models = self.list_models().await;
                        Self::build_list_response(request_id, models)
                    }
                    Which::Status(status_req) => {
                        let status = status_req?;
                        let model_ref = status.get_model_ref()?.to_str()?;
                        let info = self.model_status(model_ref).await;
                        Self::build_status_response(request_id, info)
                    }
                    Which::Infer(infer_req) => {
                        let infer = infer_req?;
                        let model_ref = infer.get_model_ref()?.to_str()?;
                        let request_data = infer.get_request()?;
                        match self.infer(model_ref, request_data).await {
                            Ok(result_bytes) => Self::build_infer_result_response(request_id, result_bytes),
                            Err(e) => Self::build_error_response(
                                request_id,
                                &format!("Inference failed: {e}"),
                                "INFER_FAILED",
                            ),
                        }
                    }
                    Which::InferStream(infer_req) => {
                        let infer = infer_req?;
                        let model_ref = infer.get_model_ref()?.to_str()?;
                        let request_data = infer.get_request()?;
                        match self.infer_stream(model_ref, request_data, client_ephemeral_pubkey).await {
                            Ok(info) => Self::build_stream_started_response(
                                request_id,
                                &info.stream_id,
                                &info.endpoint,
                                &info.server_pubkey,
                            ),
                            Err(e) => Self::build_error_response(
                                request_id,
                                &format!("Stream start failed: {e}"),
                                "STREAM_FAILED",
                            ),
                        }
                    }
                    Which::StartStream(start_req) => {
                        let start = start_req?;
                        let model_ref = start.get_model_ref()?.to_str()?;
                        let stream_id = start.get_stream_id()?.to_str()?;
                        match self.start_stream(model_ref, stream_id).await {
                            Ok(()) => Self::build_stream_authorized_response(request_id, stream_id),
                            Err(e) => Self::build_error_response(
                                request_id,
                                &format!("Stream authorization failed: {e}"),
                                "STREAM_AUTH_FAILED",
                            ),
                        }
                    }
                    Which::HealthCheck(()) => {
                        let cache = self.loaded_models.read().await;
                        let loaded_count = cache.len() as u32;
                        let max_models = self.config.max_models as u32;
                        drop(cache);
                        Self::build_health_response(request_id, loaded_count, max_models)
                    }
                    Which::ApplyChatTemplate(template_req) => {
                        let template = template_req?;
                        let model_ref = template.get_model_ref()?.to_str()?;
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
                            Ok(templated) => Self::build_template_result_response(request_id, &templated),
                            Err(e) => Self::build_error_response(
                                request_id,
                                &format!("Template application failed: {e}"),
                                "TEMPLATE_FAILED",
                            ),
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

/// Client for model operations over ZMQ
///
/// Provides async methods for loading models, routing inference requests,
/// and managing the model cache.
#[derive(Clone)]
pub struct ModelZmqClient {
    client: Arc<ZmqClient>,
}

impl ModelZmqClient {
    /// Create a new model client (endpoint from registry)
    ///
    /// # Note
    /// Uses the same signing key for both request signing and response verification.
    /// This is appropriate for internal communication where client and server share keys.
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint("model", SocketKind::Rep).to_zmq_string();
        tracing::debug!("ModelZmqClient connecting to endpoint: {}", endpoint);
        let server_verifying_key = signing_key.verifying_key();
        Self {
            client: Arc::new(ZmqClient::new(&endpoint, signing_key, server_verifying_key, identity)),
        }
    }

    /// Create a model client at a specific endpoint
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let server_verifying_key = signing_key.verifying_key();
        Self {
            client: Arc::new(ZmqClient::new(endpoint, signing_key, server_verifying_key, identity)),
        }
    }

    /// Load a model by reference with optional runtime config, returns the inference endpoint
    pub async fn load(&self, model_ref: &str, config: Option<&ModelLoadConfig>) -> Result<String> {
        let request_id = self.client.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut load = req.init_load();
            load.set_model_ref(model_ref);

            // Set optional runtime config
            if let Some(cfg) = config {
                if let Some(max_ctx) = cfg.max_context {
                    load.set_max_context(max_ctx as u32);
                }
                if let Some(kv_quant) = cfg.kv_quant {
                    let capnp_kv_quant = match kv_quant {
                        KVQuantType::None => model_capnp::KVQuantType::None,
                        KVQuantType::Int8 => model_capnp::KVQuantType::Int8,
                        KVQuantType::Nf4 => model_capnp::KVQuantType::Nf4,
                        KVQuantType::Fp4 => model_capnp::KVQuantType::Fp4,
                    };
                    load.set_kv_quant(capnp_kv_quant);
                }
            }
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;
        self.parse_loaded_response(&response_bytes)
    }

    /// Unload a model
    pub async fn unload(&self, model_ref: &str) -> Result<()> {
        let request_id = self.client.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut unload = req.init_unload();
            unload.set_model_ref(model_ref);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;
        self.parse_ok_response(&response_bytes)
    }

    /// List loaded models
    pub async fn list(&self) -> Result<Vec<LoadedModelInfo>> {
        let request_id = self.client.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            req.set_list(());
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;
        self.parse_list_response(&response_bytes)
    }

    /// Get model status
    pub async fn status(&self, model_ref: &str) -> Result<ModelStatusInfo> {
        let request_id = self.client.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut status = req.init_status();
            status.set_model_ref(model_ref);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;
        self.parse_status_response(&response_bytes)
    }

    /// Run inference on a model
    pub async fn infer(&self, model_ref: &str, request: &GenerationRequest) -> Result<GenerationResult> {
        let request_id = self.client.next_id();

        // Serialize the GenerationRequest
        let gen_request_bytes = serialize_generation_request(request)?;

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut infer = req.init_infer();
            infer.set_model_ref(model_ref);
            infer.set_request(&gen_request_bytes);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;
        self.parse_infer_result_response(&response_bytes)
    }

    /// Start streaming inference on a model with E2E authentication.
    ///
    /// # Arguments
    /// * `model_ref` - The model reference (e.g., "qwen3-small:main")
    /// * `request` - Generation parameters
    /// * `client_ephemeral_pubkey` - Client's Ristretto255 ephemeral public key (32 bytes)
    ///
    /// # E2E Authentication
    ///
    /// The client must provide an ephemeral Ristretto255 public key. After receiving
    /// the response, derive stream keys using:
    /// ```ignore
    /// let server_pubkey = RistrettoPublic::from_bytes(&server_pub)?;
    /// let shared = ristretto_dh(&client_secret, &server_pubkey);
    /// let keys = derive_stream_keys(&shared, &client_pub, &server_pub)?;
    /// // keys.topic = DH-derived topic (64 hex chars) to subscribe to
    /// // keys.mac_key = HMAC key for verifying stream chunks
    /// ```
    pub async fn infer_stream(
        &self,
        model_ref: &str,
        request: &GenerationRequest,
        client_ephemeral_pubkey: [u8; 32],
    ) -> Result<StreamStartedInfo> {
        let request_id = self.client.next_id();

        // Serialize the GenerationRequest
        let gen_request_bytes = serialize_generation_request(request)?;

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut infer = req.init_infer_stream();
            infer.set_model_ref(model_ref);
            infer.set_request(&gen_request_bytes);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        // Pass ephemeral pubkey for E2E authenticated streaming
        let opts = CallOptions::default().ephemeral_pubkey(client_ephemeral_pubkey);
        let response_bytes = self.client.call(request_bytes, opts).await?;
        self.parse_stream_started_response(&response_bytes)
    }

    /// Authorize a stream subscription
    ///
    /// After calling `infer_stream`, the client must call this method to authorize
    /// subscription before subscribing to the stream. This routes to InferenceService
    /// which sends an AUTHORIZE message to StreamService.
    ///
    /// # Arguments
    /// * `model_ref` - The model reference (e.g., "qwen3-small:main")
    /// * `stream_id` - The stream ID returned by `infer_stream`
    ///
    /// # Returns
    /// The confirmed stream_id on success.
    pub async fn start_stream(&self, model_ref: &str, stream_id: &str) -> Result<String> {
        let request_id = self.client.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut start_req = req.init_start_stream();
            start_req.set_model_ref(model_ref);
            start_req.set_stream_id(stream_id);
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;
        self.parse_stream_authorized_response(&response_bytes)
    }

    /// Health check
    pub async fn health_check(&self) -> Result<ModelHealthInfo> {
        let request_id = self.client.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            req.set_health_check(());
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;
        self.parse_health_response(&response_bytes)
    }

    /// Apply chat template to messages, returning a TemplatedPrompt
    ///
    /// This routes to the model's InferenceService which has the template engine.
    pub async fn apply_chat_template(
        &self,
        model_ref: &str,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<TemplatedPrompt> {
        let request_id = self.client.next_id();

        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<model_capnp::model_request::Builder>();
            req.set_id(request_id);
            let mut template_req = req.init_apply_chat_template();
            template_req.set_model_ref(model_ref);
            template_req.set_add_generation_prompt(add_generation_prompt);

            // Build messages list
            let mut msg_list = template_req.init_messages(messages.len() as u32);
            for (i, m) in messages.iter().enumerate() {
                let mut msg_builder = msg_list.reborrow().get(i as u32);
                msg_builder.set_role(&m.role);
                msg_builder.set_content(m.content.as_deref().unwrap_or(""));
            }
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;
        self.parse_template_result_response(&response_bytes)
    }

    // ========================================================================
    // Response parsers
    // ========================================================================

    fn parse_loaded_response(&self, bytes: &[u8]) -> Result<String> {
        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let response = reader.get_root::<model_capnp::model_response::Reader>()?;

        use model_capnp::model_response::Which;
        match response.which()? {
            Which::Loaded(loaded_reader) => {
                let loaded = loaded_reader?;
                Ok(loaded.get_endpoint()?.to_str()?.to_owned())
            }
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "{}: {}",
                    error.get_code()?.to_str()?,
                    error.get_message()?.to_str()?
                ))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    fn parse_ok_response(&self, bytes: &[u8]) -> Result<()> {
        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let response = reader.get_root::<model_capnp::model_response::Reader>()?;

        use model_capnp::model_response::Which;
        match response.which()? {
            Which::Ok(()) => Ok(()),
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "{}: {}",
                    error.get_code()?.to_str()?,
                    error.get_message()?.to_str()?
                ))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    fn parse_list_response(&self, bytes: &[u8]) -> Result<Vec<LoadedModelInfo>> {
        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let response = reader.get_root::<model_capnp::model_response::Reader>()?;

        use model_capnp::model_response::Which;
        match response.which()? {
            Which::List(list_reader) => {
                let list = list_reader?;
                let models = list.get_models()?;
                let mut result = Vec::with_capacity(models.len() as usize);
                for model in models.iter() {
                    result.push(LoadedModelInfo {
                        model_ref: model.get_model_ref()?.to_str()?.to_owned(),
                        endpoint: model.get_endpoint()?.to_str()?.to_owned(),
                        loaded_at: model.get_loaded_at(),
                        last_used: model.get_last_used(),
                    });
                }
                Ok(result)
            }
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "{}: {}",
                    error.get_code()?.to_str()?,
                    error.get_message()?.to_str()?
                ))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    fn parse_status_response(&self, bytes: &[u8]) -> Result<ModelStatusInfo> {
        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let response = reader.get_root::<model_capnp::model_response::Reader>()?;

        use model_capnp::model_response::Which;
        match response.which()? {
            Which::Status(status_reader) => {
                let status = status_reader?;
                Ok(ModelStatusInfo {
                    loaded: status.get_loaded(),
                    endpoint: if status.has_endpoint() {
                        Some(status.get_endpoint()?.to_str()?.to_owned())
                    } else {
                        None
                    },
                })
            }
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "{}: {}",
                    error.get_code()?.to_str()?,
                    error.get_message()?.to_str()?
                ))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    fn parse_infer_result_response(&self, bytes: &[u8]) -> Result<GenerationResult> {
        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let response = reader.get_root::<model_capnp::model_response::Reader>()?;

        use model_capnp::model_response::Which;
        match response.which()? {
            Which::InferResult(result_bytes) => {
                let result_data = result_bytes?;
                parse_generation_result(result_data)
            }
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "{}: {}",
                    error.get_code()?.to_str()?,
                    error.get_message()?.to_str()?
                ))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    fn parse_stream_started_response(&self, bytes: &[u8]) -> Result<StreamStartedInfo> {
        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let response = reader.get_root::<model_capnp::model_response::Reader>()?;

        use model_capnp::model_response::Which;
        match response.which()? {
            Which::StreamStarted(stream_reader) => {
                let stream = stream_reader?;
                let server_pubkey_data = stream.get_server_pubkey()?;
                let mut server_pubkey = [0u8; 32];
                if server_pubkey_data.len() == 32 {
                    server_pubkey.copy_from_slice(server_pubkey_data);
                }
                Ok(StreamStartedInfo {
                    stream_id: stream.get_stream_id()?.to_str()?.to_owned(),
                    endpoint: stream.get_endpoint()?.to_str()?.to_owned(),
                    server_pubkey,
                })
            }
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "{}: {}",
                    error.get_code()?.to_str()?,
                    error.get_message()?.to_str()?
                ))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    fn parse_stream_authorized_response(&self, bytes: &[u8]) -> Result<String> {
        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let response = reader.get_root::<model_capnp::model_response::Reader>()?;

        use model_capnp::model_response::Which;
        match response.which()? {
            Which::StreamAuthorized(auth_reader) => {
                let auth = auth_reader?;
                Ok(auth.get_stream_id()?.to_str()?.to_owned())
            }
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "{}: {}",
                    error.get_code()?.to_str()?,
                    error.get_message()?.to_str()?
                ))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    fn parse_health_response(&self, bytes: &[u8]) -> Result<ModelHealthInfo> {
        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let response = reader.get_root::<model_capnp::model_response::Reader>()?;

        use model_capnp::model_response::Which;
        match response.which()? {
            Which::Health(health_reader) => {
                let health = health_reader?;
                Ok(ModelHealthInfo {
                    status: health.get_status()?.to_str()?.to_owned(),
                    loaded_model_count: health.get_loaded_model_count(),
                    max_models: health.get_max_models(),
                    total_memory_bytes: health.get_total_memory_bytes(),
                })
            }
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "{}: {}",
                    error.get_code()?.to_str()?,
                    error.get_message()?.to_str()?
                ))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    fn parse_template_result_response(&self, bytes: &[u8]) -> Result<TemplatedPrompt> {
        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let response = reader.get_root::<model_capnp::model_response::Reader>()?;

        use model_capnp::model_response::Which;
        match response.which()? {
            Which::TemplateResult(result) => {
                let prompt_str = result?.to_str()?.to_owned();
                Ok(TemplatedPrompt::new(prompt_str))
            }
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "{}: {}",
                    error.get_code()?.to_str()?,
                    error.get_message()?.to_str()?
                ))
            }
            _ => Err(anyhow!("Unexpected response type")),
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

/// Parse a GenerationRequest from Cap'n Proto
fn parse_generation_request(
    reader: crate::inference_capnp::generation_request::Reader,
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
            Some(reader.get_seed())
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

/// Serialize a GenerationRequest to Cap'n Proto bytes
fn serialize_generation_request(request: &GenerationRequest) -> Result<Vec<u8>> {
    let mut message = Builder::new_default();
    {
        let mut req = message.init_root::<crate::inference_capnp::generation_request::Builder>();
        req.set_prompt(request.prompt.as_str());
        req.set_max_tokens(request.max_tokens as u32);
        req.set_temperature(request.temperature);
        req.set_top_p(request.top_p);
        req.set_top_k(request.top_k.unwrap_or(0) as u32);
        req.set_repeat_penalty(request.repeat_penalty);
        req.set_repeat_last_n(request.repeat_last_n as u32);
        req.set_seed(request.seed.unwrap_or(0));
        req.set_timeout_ms(request.timeout.unwrap_or(0));

        if !request.stop_tokens.is_empty() {
            let mut stop_list = req.init_stop_tokens(request.stop_tokens.len() as u32);
            for (i, token) in request.stop_tokens.iter().enumerate() {
                stop_list.set(i as u32, token);
            }
        }
    }

    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message)?;
    Ok(bytes)
}

/// Serialize a GenerationResult to Cap'n Proto bytes
fn serialize_generation_result(result: &GenerationResult) -> Result<Vec<u8>> {
    let mut message = Builder::new_default();
    {
        let mut res = message.init_root::<crate::inference_capnp::generation_result::Builder>();
        res.set_text(&result.text);
        res.set_tokens_generated(result.tokens_generated as u32);
        res.set_generation_time_ms(result.generation_time_ms);
        res.set_tokens_per_second(result.tokens_per_second);

        // Set finish reason
        let finish_reason = match &result.finish_reason {
            crate::config::FinishReason::MaxTokens => {
                crate::inference_capnp::FinishReason::MaxTokens
            }
            crate::config::FinishReason::StopToken(_) => {
                crate::inference_capnp::FinishReason::StopToken
            }
            crate::config::FinishReason::EndOfSequence => {
                crate::inference_capnp::FinishReason::EndOfSequence
            }
            crate::config::FinishReason::Error(_) => crate::inference_capnp::FinishReason::Error,
            crate::config::FinishReason::Stop => crate::inference_capnp::FinishReason::Stop,
        };
        res.set_finish_reason(finish_reason);

        // Set prefill/inference metrics BEFORE init_quality_metrics (which consumes the builder)
        res.set_prefill_tokens(result.prefill_tokens as u32);
        res.set_prefill_time_ms(result.prefill_time_ms);
        res.set_prefill_tokens_per_sec(result.prefill_tokens_per_sec);
        res.set_inference_tokens(result.inference_tokens as u32);
        res.set_inference_time_ms(result.inference_time_ms);
        res.set_inference_tokens_per_sec(result.inference_tokens_per_sec);

        // Set quality metrics if present (init_quality_metrics consumes the builder)
        if let Some(qm) = &result.quality_metrics {
            let mut metrics = res.init_quality_metrics();
            metrics.set_perplexity(qm.perplexity);
            metrics.set_avg_entropy(qm.avg_entropy);
            metrics.set_entropy_variance(qm.entropy_variance);
            metrics.set_repetition_ratio(qm.repetition_ratio);
        }
    }

    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message)?;
    Ok(bytes)
}

/// Parse a GenerationResult from Cap'n Proto bytes
fn parse_generation_result(bytes: &[u8]) -> Result<GenerationResult> {
    let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
    let res = reader.get_root::<crate::inference_capnp::generation_result::Reader>()?;

    let finish_reason = match res.get_finish_reason()? {
        crate::inference_capnp::FinishReason::MaxTokens => crate::config::FinishReason::MaxTokens,
        crate::inference_capnp::FinishReason::StopToken => {
            crate::config::FinishReason::StopToken(String::new())
        }
        crate::inference_capnp::FinishReason::EndOfSequence => {
            crate::config::FinishReason::EndOfSequence
        }
        crate::inference_capnp::FinishReason::Error => {
            crate::config::FinishReason::Error(String::new())
        }
        crate::inference_capnp::FinishReason::Stop => crate::config::FinishReason::Stop,
    };

    let quality_metrics = if res.has_quality_metrics() {
        let qm = res.get_quality_metrics()?;
        Some(crate::runtime::generation_metrics::GenerationQualityMetrics {
            perplexity: qm.get_perplexity(),
            avg_entropy: qm.get_avg_entropy(),
            entropy_variance: qm.get_entropy_variance(),
            repetition_ratio: qm.get_repetition_ratio(),
            token_count: res.get_tokens_generated(),
        })
    } else {
        None
    };

    Ok(GenerationResult {
        text: res.get_text()?.to_str()?.to_owned(),
        tokens_generated: res.get_tokens_generated() as usize,
        finish_reason,
        generation_time_ms: res.get_generation_time_ms(),
        tokens_per_second: res.get_tokens_per_second(),
        quality_metrics,
        prefill_tokens: res.get_prefill_tokens() as usize,
        prefill_time_ms: res.get_prefill_time_ms(),
        prefill_tokens_per_sec: res.get_prefill_tokens_per_sec(),
        inference_tokens: res.get_inference_tokens() as usize,
        inference_time_ms: res.get_inference_time_ms(),
        inference_tokens_per_sec: res.get_inference_tokens_per_sec(),
    })
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
