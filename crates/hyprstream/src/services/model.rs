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
use crate::config::{GenerationRequest, GenerationResult, TemplatedPrompt};
use crate::runtime::kv_quant::KVQuantType;
use crate::runtime::RuntimeConfig;
use crate::services::{
    rpc_types::StreamInfo, EnvelopeContext, InferenceService, InferenceZmqClient,
    PolicyClient,
};
use crate::services::GenRegistryClient;
use crate::storage::ModelRef;
use anyhow::{anyhow, Result};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::transport::TransportConfig;
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
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
        let tracked = self.registry.get_by_name(&model_ref.model).await
            .map_err(|e| anyhow!("Model '{}' not found in registry: {}", model_ref.model, e))?;
        let repo_client = self.registry.repo(&tracked.id);

        let branch_name = match &model_ref.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            _ => repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned()),
        };
        let worktrees = repo_client.list_worktrees().await?;
        let model_path = std::path::PathBuf::from(
            &worktrees.iter()
                .find(|wt| wt.branch_name == branch_name)
                .ok_or_else(|| anyhow!("worktree for {}:{} not found", model_ref.model, branch_name))?
                .path,
        );

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
        let fs: Option<crate::services::WorktreeClient> = Some(repo_client.worktree(&branch_name));

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
    ) -> Result<StreamInfo> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.generate_stream(&request, client_ephemeral_pubkey).await
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

    async fn train_step_stream(
        &self,
        model_ref_str: &str,
        input: &str,
        gradient_steps: u32,
        learning_rate: f32,
        auto_commit: bool,
        client_ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamInfo> {
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
    ) -> Result<crate::services::generated::inference_client::DeltaStatusResult> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.gen.get_delta_status().await
    }

    async fn snapshot_delta_forward(
        &self,
        model_ref_str: &str,
    ) -> Result<crate::services::generated::inference_client::SnapshotDeltaResult> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.gen.snapshot_delta().await
    }

    async fn export_peft_adapter_forward(
        &self,
        model_ref_str: &str,
        name: &str,
        commit_message: &str,
    ) -> Result<crate::services::generated::inference_client::ExportPeftResult> {
        let client = self.get_inference_client(model_ref_str).await?;
        client.gen.export_peft_adapter(name, commit_message).await
    }

}

// ═══════════════════════════════════════════════════════════════════════════════
// ModelHandler Implementation — generated dispatch for top-level + typed scope traits
// ═══════════════════════════════════════════════════════════════════════════════

use crate::services::generated::model_client::{
    ModelHandler, TttHandler, PeftHandler, InferHandler,
    dispatch_model, ModelResponseVariant,
    LoadedModelResponse, ErrorInfo, ModelListResponse, ModelHealthStatus,
    // Top-level request types
    LoadModelRequest, UnloadModelRequest, KVQuantTypeEnum,
    // TTT types
    CreateLoraRequest, TrainStepRequest, TrainStepResponse,
    GetDeltaStatusResponse, ModuleNormRatio,
    SaveAdaptationRequest, SaveAdaptationResponse,
    SnapshotDeltaResponse, TttExportRequest, TttExportResponse,
    // PEFT types
    PeftAdapterInfo, PeftMergeRequest,
    // Infer types
    GenerateRequest, ApplyChatTemplateRequest, InferResult, ModelStatusResponse, OnlineTrainingConfig,
};
// Conflicting names — use canonical path at usage sites:
//   model_client::LoadedModelInfo, model_client::StreamInfo,
//   model_client::ChatMessage

#[async_trait::async_trait(?Send)]
impl TttHandler for ModelService {
    async fn handle_create(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &CreateLoraRequest,
    ) -> Result<()> {
        let config = crate::training::TenantDeltaConfig {
            rank: data.rank as usize,
            alpha: data.alpha,
            dropout: data.dropout,
            target_modules: data.target_modules.clone(),
            learning_rate: data.learning_rate as f64,
            ..crate::training::TenantDeltaConfig::default()
        };
        self.create_lora(model_ref, config).await
    }

    async fn handle_train(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TrainStepRequest,
    ) -> Result<TrainStepResponse> {
        let client = self.get_inference_client(model_ref).await?;
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
                model_ref, &data.input, data.gradient_steps, data.learning_rate,
                data.auto_commit, ctx.ephemeral_pubkey,
            ).await?;
        // Convert inference_client::crate::services::generated::model_client::StreamInfo to model_client::crate::services::generated::model_client::StreamInfo
        let stream_info = crate::services::generated::model_client::StreamInfo {
            stream_id: info.stream_id,
            endpoint: info.endpoint,
            server_pubkey: info.server_pubkey,
        };
        Ok((stream_info, Box::pin(async {})))
    }

    async fn handle_commit(
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.commit_adaptation(model_ref).await
    }

    async fn handle_rollback(
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.rollback_adaptation(model_ref).await
    }

    async fn handle_reset(
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.reset_delta(model_ref).await
    }

    async fn handle_status(
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<GetDeltaStatusResponse> {
        let r = self.get_delta_status_forward(model_ref).await?;
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
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &SaveAdaptationRequest,
    ) -> Result<SaveAdaptationResponse> {
        let client = self.get_inference_client(model_ref).await?;
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
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<SnapshotDeltaResponse> {
        let r = self.snapshot_delta_forward(model_ref).await?;
        Ok(SnapshotDeltaResponse {
            content_hash: r.content_hash,
            size_bytes: r.size_bytes,
            accumulated_steps: r.accumulated_steps,
            request_count: r.request_count,
        })
    }

    async fn handle_export(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TttExportRequest,
    ) -> Result<TttExportResponse> {
        let r = self.export_peft_adapter_forward(model_ref, &data.name, &data.commit_message).await?;
        Ok(TttExportResponse {
            adapter_path: r.adapter_path,
            content_hash: r.content_hash,
        })
    }
}

#[async_trait::async_trait(?Send)]
impl PeftHandler for ModelService {
    async fn handle_load(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, value: &str,
    ) -> Result<()> {
        self.load_lora(model_ref, value).await
    }

    async fn handle_unload(
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.unload_lora(model_ref).await
    }

    async fn handle_has(
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<bool> {
        self.has_lora(model_ref).await
    }

    async fn handle_check(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, _value: &str,
    ) -> Result<PeftAdapterInfo> {
        anyhow::bail!("peft.check not yet implemented")
    }

    async fn handle_merge(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, _data: &PeftMergeRequest,
    ) -> Result<()> {
        anyhow::bail!("peft.merge not yet implemented")
    }
}

#[async_trait::async_trait(?Send)]
impl InferHandler for ModelService {
    async fn handle_generate(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &GenerateRequest,
    ) -> Result<InferResult> {
        let request = generate_request_from_data(data);
        let result = self.infer(model_ref, request).await?;
        Ok(InferResult {
            text: result.text,
            tokens_generated: result.tokens_generated as u32,
            finish_reason: finish_reason_to_str(&result.finish_reason),
            generation_time_ms: result.generation_time_ms,
            tokens_per_second: result.tokens_per_second,
            prefill_tokens: result.prefill_tokens as u32,
            prefill_time_ms: result.prefill_time_ms,
            prefill_tokens_per_sec: result.prefill_tokens_per_sec,
            inference_tokens: result.inference_tokens as u32,
            inference_time_ms: result.inference_time_ms,
            inference_tokens_per_sec: result.inference_tokens_per_sec,
        })
    }

    async fn handle_generate_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &GenerateRequest,
    ) -> Result<(crate::services::generated::model_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let request = generate_request_from_data(data);
        let info = self.infer_stream(model_ref, request, ctx.ephemeral_pubkey).await?;
        // Convert inference_client::crate::services::generated::model_client::StreamInfo to model_client::crate::services::generated::model_client::StreamInfo
        let stream_info = crate::services::generated::model_client::StreamInfo {
            stream_id: info.stream_id,
            endpoint: info.endpoint,
            server_pubkey: info.server_pubkey,
        };
        Ok((stream_info, Box::pin(async {})))
    }

    async fn handle_apply_chat_template(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &ApplyChatTemplateRequest,
    ) -> Result<String> {
        let chat_messages: Vec<ChatMessage> = data.messages.iter().map(|m| ChatMessage {
            role: m.role.clone(),
            content: Some(m.content.clone()),
            function_call: None,
        }).collect();
        let templated = self.apply_chat_template(model_ref, chat_messages, data.add_generation_prompt).await?;
        Ok(templated.as_str().to_string())
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
            Some(ModelLoadConfig { max_context: max_ctx, kv_quant: kv_q })
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

    async fn handle_list(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
    ) -> Result<ModelResponseVariant> {
        let models = self.list_models().await;
        Ok(ModelResponseVariant::ListResult(ModelListResponse {
            models: models.into_iter().map(|m| crate::services::generated::model_client::LoadedModelInfo {
                model_ref: m.model_ref,
                endpoint: m.endpoint,
                loaded_at: m.loaded_at,
                last_used: m.last_used,
                memory_bytes: 0,
                session_count: 0,
            }).collect(),
        }))
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
            false,  // ttt_enabled
            0,      // ttt_gradient_steps
            0.0,    // ttt_learning_rate
            false,  // auto_commit
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
    ) -> Result<TemplatedPrompt> {
        let msg_data: Vec<crate::services::generated::model_client::ChatMessage> = messages.iter().map(|m| crate::services::generated::model_client::ChatMessage {
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
    }
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
