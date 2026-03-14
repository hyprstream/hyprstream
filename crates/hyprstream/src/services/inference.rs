//! ZMQ-based inference service for text generation
//!
//! This service wraps TorchEngine and provides a ZMQ interface for inference operations.
//! It uses:
//! - REQ/REP for standard requests (generate, model_info, lora operations, etc.)
//! - PUB (via StreamService) for streaming generation with JWT authorization
//!
//! # Thread Model
//!
//! The service runs on a dedicated thread with its own TorchEngine because:
//! - tch-rs types contain raw pointers (not Send)
//! - GPU operations benefit from thread isolation
//!
//! # Streaming Architecture
//!
//! ```text
//! Client                    InferenceService         StreamService
//!   │                            │                         │
//!   │─ REQ: GenerateStream ─────►│                         │
//!   │◄─ REP: {stream_id,endpoint}│                         │
//!   │                            │                         │
//!   │                            │── PUB: chunks ─────────►│ (validates JWT)
//!   │                            │                         │
//!   │  SUB: stream-{id}|{jwt} ───────────────────────────►│
//!   │◄────────────────── chunks (JWT stripped) ───────────│
//! ```
//!
//! InferenceService publishes to StreamService's XSUB (PUB socket).
//! StreamService validates JWT at subscription and forwards to clients.
//!
//! # Authorization
//!
//! Uses `InferenceHandler::authorize()` via generated dispatch for policy-backed
//! authorization on all requests. The handler delegates to PolicyClient.

use crate::services::PolicyClient;
use crate::config::TrainingMode;
use crate::runtime::GenerationRequest;
use crate::runtime::ModelInfo;
use crate::inference_capnp;
use crate::runtime::kv_cache::CacheOwner;
use crate::runtime::model_config::ModelConfig;
use crate::runtime::{RuntimeConfig, RuntimeEngine, TorchEngine};
use crate::services::rpc_types::StreamInfo;
use crate::services::EnvelopeContext;
use crate::services::WorktreeClient;
use crate::training::{DeltaPool, TenantDeltaConfig, TTTConfig, TestTimeTrainer};
use hyprstream_rpc::Subject;
use crate::training::serialize_state_dict_to_bytes;
use crate::zmq::global_context;
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::StreamChannel;
use capnp::serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::Mutex;
use tokio::runtime::Handle;
use tokenizers::Tokenizer;
use tracing::{debug, error, info, trace, warn};


/// Pending work to be executed after REP response is sent.
///
/// This solves the streaming deadlock where the service waits for subscription
/// before returning the response, but the client can't subscribe without
/// the stream_id from the response.
///
/// Wraps `StreamContext` from hyprstream-rpc with operation-specific data.
enum PendingWork {
    /// Streaming text generation
    Generation {
        /// Stream context with DH-derived keys (from hyprstream-rpc)
        stream_ctx: hyprstream_rpc::StreamContext,
        /// Generation request to execute
        request: GenerationRequest,
        /// Subject identity for tenant-aware TTT (deferred to execute_stream)
        subject: Subject,
        /// Per-request TTT overrides (deferred to execute_stream)
        ttt_overrides: crate::training::ttt::TTTOverrides,
    },
    /// Streaming training step (avoids REQ/REP timeout on backward pass compilation)
    Training {
        /// Stream context with DH-derived keys (from hyprstream-rpc)
        stream_ctx: hyprstream_rpc::StreamContext,
        /// Subject identity for tenant-aware TTT
        subject: Subject,
        /// Text to train on
        input: String,
        /// Number of gradient steps
        gradient_steps: u32,
        /// Learning rate override (0 = use default)
        learning_rate: f32,
        /// How to handle the adaptation result
        adaptation_strategy: crate::training::adaptation_state::AdaptationStrategy,
    },
}


/// Default endpoint for the inference service
pub const INFERENCE_ENDPOINT: &str = "inproc://hyprstream/inference";

// Note: Stream endpoints are now dynamically generated per InferenceService instance
// using UUIDs (e.g., inproc://hyprstream/inference/stream/{uuid}).
// Each service binds to its own unique endpoint to prevent bind conflicts.

/// ZMQ-based inference service
///
/// Wraps TorchEngine and provides a Cap'n Proto interface over ZMQ.
/// Thread-safe via RwLock for multi-threaded access.
///
/// # Security
///
/// All requests must be wrapped in `SignedEnvelope` and are verified before processing.
/// The service logs the identity for audit trails.
///
/// # Streaming Architecture
///
/// Uses a single PUB socket connected to StreamService:
/// - Topic-based multiplexing: `stream-{id}` prefix on each message
/// - StreamService validates JWT at subscription (prevents hijacking)
/// - InferenceService just publishes chunks (no authorization logic)
/// - Clients subscribe via StreamService with JWT tokens
///
/// Inner state of InferenceService, shared via Arc for continuations.
///
/// All methods are defined on this inner type. `InferenceService` wraps it
/// in `Arc` and implements `Deref` so all field/method access is transparent.
/// Streaming continuations clone the `Arc` to own the state across await points.
pub struct InferenceServiceInner {
    engine: parking_lot::RwLock<TorchEngine>,
    /// Model path for checkpoint management
    #[allow(dead_code)] // Future: checkpoint management
    model_path: PathBuf,
    /// Current session ID for events
    session_id: parking_lot::RwLock<Option<String>>,
    /// Runtime handle for async operations (reused instead of creating new runtimes)
    #[allow(dead_code)] // Reserved for future async operations
    runtime_handle: Handle,
    /// Stream channel for streaming generation (connects to StreamService)
    /// Handles DH key exchange, pre-authorization, and publishing.
    stream_channel: Option<StreamChannel>,
    /// Server's Ed25519 verifying key for signature verification
    #[allow(dead_code)] // Reserved for callback-mode signature verification
    server_pubkey: VerifyingKey,
    /// Service signing key for stream registration (generated at init)
    #[allow(dead_code)] // Used by callback mode; standard mode passes key to InferenceZmqAdapter
    signing_key: SigningKey,
    /// Nonce cache for replay protection
    #[allow(dead_code)] // Standard mode uses RequestLoop's nonce cache; kept for callback mode
    nonce_cache: Arc<InMemoryNonceCache>,
    /// Policy client for authorization checks (async via TMQ)
    policy_client: PolicyClient,
    /// Optional TTT trainer (initialized from config.json)
    ttt_trainer: Option<Arc<TestTimeTrainer>>,
    /// Tokenizer for TTT adaptation
    tokenizer: Option<Arc<Tokenizer>>,
    /// Per-tenant delta pool for isolated TTT adaptations
    delta_pool: Option<Arc<DeltaPool>>,
    /// Base LoRA delta loaded from a .safetensors adapter file.
    /// Applied to all tenants (composed with per-tenant delta if both exist).
    base_delta: Mutex<Option<Arc<Mutex<crate::training::TenantDelta>>>>,
    /// Optional WorktreeClient for worktree-scoped file operations.
    /// When present, adapter/snapshot writes use contained-root access.
    fs: Option<WorktreeClient>,
    /// ZMQ context for socket creation
    #[allow(dead_code)] // Standard mode passes context to InferenceZmqAdapter
    zmq_context: Arc<zmq::Context>,
    /// Transport configuration for the service endpoint
    #[allow(dead_code)] // Standard mode passes transport to InferenceZmqAdapter
    transport: hyprstream_rpc::transport::TransportConfig,
    /// LoRA generation counter — incremented on create/load/unload.
    /// Checked before generation to detect LoRA reconfiguration mid-stream.
    lora_generation: Arc<AtomicU64>,
}

/// ZMQ-based inference service
///
/// Wraps `InferenceServiceInner` in `Arc` for continuation-based streaming.
/// All field and method access is transparent via `Deref`.
pub struct InferenceService {
    inner: Arc<InferenceServiceInner>,
}

impl Clone for InferenceService {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl std::ops::Deref for InferenceService {
    type Target = InferenceServiceInner;
    fn deref(&self) -> &Self::Target { &self.inner }
}

// SAFETY: InferenceServiceInner contains tch-rs types (raw pointers) behind
// parking_lot::RwLock/Mutex, providing proper synchronization. This is consistent
// with the codebase pattern for tch-rs wrappers (LlamaModel, TenantDelta,
// TestTimeTrainer, etc.) which all have `unsafe impl Send + Sync`.
unsafe impl Send for InferenceServiceInner {}
unsafe impl Sync for InferenceServiceInner {}

/// Delta status information returned by getDeltaStatus
pub struct DeltaStatusInfo {
    pub exists: bool,
    pub accumulated_steps: u64,
    pub max_accumulated_steps: u64,
    pub request_count: u64,
    pub avg_loss_improvement: f32,
    pub memory_bytes: u64,
    pub last_snapshot_hash: String,
    pub delta_norm_ratios: std::collections::HashMap<String, f64>,
    pub has_pending: bool,
}

/// Save adaptation result information
pub struct SaveAdaptationInfo {
    pub adapter_name: String,
    pub adapter_path: String,
    pub content_hash: String,
    pub merge_strategy: String,
}

/// Snapshot delta result information
pub struct SnapshotDeltaInfo {
    pub content_hash: String,
    pub size_bytes: u64,
    pub accumulated_steps: u64,
    pub request_count: u64,
}

/// Export PEFT adapter result information
pub struct ExportPeftInfo {
    pub adapter_path: String,
    pub content_hash: String,
}

impl InferenceService {
    /// Start inference service in callback mode
    ///
    /// This mode is used when InferenceService is spawned as a separate process.
    /// The service:
    /// 1. Connects DEALER to ModelService's ROUTER (callback endpoint)
    /// 2. Sends Register message with its stream endpoint
    /// 3. Waits for LoadModel command
    /// 4. Loads the model
    /// 5. Enters command loop handling Infer/Shutdown
    ///
    /// # Arguments
    /// * `instance_id` - Unique ID for this instance (e.g., "inference-a1b2c3d4")
    /// * `callback_endpoint` - ModelService's ROUTER endpoint for callbacks
    /// * `config` - Runtime configuration
    /// * `policy_client` - Policy client for authorization
    pub async fn start_with_callback(
        instance_id: String,
        callback_endpoint: String,
        config: RuntimeConfig,
        policy_client: PolicyClient,
    ) -> Result<()> {
        info!(
            "Starting InferenceService {} in callback mode (callback={})",
            instance_id, callback_endpoint
        );

        // Run in current thread (we're likely spawned as a separate process)
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async move {
            Self::run_callback_mode(instance_id, callback_endpoint, config, policy_client).await
        })
    }

    /// Run the callback mode loop
    async fn run_callback_mode(
        instance_id: String,
        callback_endpoint: String,
        config: RuntimeConfig,
        policy_client: PolicyClient,
    ) -> Result<()> {
        let ctx = global_context();

        // Create DEALER socket and connect to callback endpoint
        let dealer = ctx.socket(zmq::DEALER)?;
        dealer.set_identity(instance_id.as_bytes())?;
        dealer.set_rcvtimeo(100)?; // 100ms timeout for polling
        dealer.connect(&callback_endpoint)?;
        info!("Connected DEALER to {}", callback_endpoint);

        // StreamChannel will be created after we have a signing key

        // Get StreamService's Sub endpoint for client subscriptions
        let stream_sub_endpoint = hyprstream_rpc::registry::global()
            .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
            .to_zmq_string();

        // Send Register message (this IS the ready signal)
        let register_msg = Self::build_register(&instance_id, &stream_sub_endpoint)?;
        dealer.send(&register_msg, 0)?;
        info!("Sent Register to callback");

        // Wait for LoadModel command
        let (model_path, model_ref) = Self::wait_for_load_model(&dealer)?;

        // Initialize the engine and load the model
        // SECURITY NOTE: Callback mode uses IPC (unix socket / inproc) between model service
        // and inference worker — both in the same trust domain. Signature verification is
        // skipped because the worker process doesn't have the server's public key.
        // This is acceptable for IPC but must NOT be used for network-facing endpoints.
        if callback_endpoint.starts_with("tcp://") {
            return Err(anyhow!(
                "Callback mode must use IPC transport (ipc:// or inproc://), not TCP. \
                 TCP callback would bypass signature verification."
            ));
        }
        let server_pubkey = VerifyingKey::default();
        // Generate signing key for callback mode (separate process, no shared key access)
        let signing_key = hyprstream_rpc::crypto::signing::generate_signing_keypair().0;
        let nonce_cache = Arc::new(InMemoryNonceCache::new());
        let service = Self::initialize(
            model_path.clone(),
            config,
            server_pubkey,
            signing_key.clone(),
            nonce_cache,
            policy_client,
            None, // Callback mode: no FsOps
        )
        .await?;

        // StreamChannel already created in initialize()

        // Send LoadModelResponse
        let response = Self::build_load_model_response(true, "")?;
        dealer.send(&response, 0)?;
        info!("Model {} loaded, sent response", model_ref);

        // Enter command loop
        Self::callback_command_loop(service, &dealer).await
    }

    /// Wait for LoadModel command from DEALER
    fn wait_for_load_model(dealer: &zmq::Socket) -> Result<(PathBuf, String)> {
        loop {
            match dealer.recv_bytes(0) {
                Ok(data) => {
                    let reader = serialize::read_message(
                        &mut std::io::Cursor::new(&data),
                        ReaderOptions::new(),
                    )?;
                    let cmd = reader.get_root::<crate::model_capnp::inference_command::Reader>()?;

                    use crate::model_capnp::inference_command::Which;
                    match cmd.which()? {
                        Which::LoadModel(load) => {
                            let load = load?;
                            let model_ref = load.get_model_ref()?.to_str()?.to_owned();
                            let model_path = PathBuf::from(load.get_model_path()?.to_str()?);
                            return Ok((model_path, model_ref));
                        }
                        Which::Shutdown(()) => {
                            info!("Received Shutdown before LoadModel, returning");
                            return Err(anyhow!("Shutdown requested before LoadModel"));
                        }
                        Which::Infer(_) => {
                            warn!("Received Infer before LoadModel, ignoring");
                        }
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout, continue waiting
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("DEALER recv error: {}", e));
                }
            }
        }
    }

    /// Callback mode command loop
    async fn callback_command_loop(mut service: Self, dealer: &zmq::Socket) -> Result<()> {
        loop {
            match dealer.recv_bytes(0) {
                Ok(data) => {
                    let (response, continuation) = service.handle_callback_command(&data).await?;
                    dealer.send(&response, 0)?;
                    // Spawn continuation after response (mirrors RequestLoop's spawn_local behavior).
                    // Using spawn_local (not inline await) so the DEALER loop isn't blocked
                    // while streaming completes.
                    if let Some(future) = continuation {
                        tokio::task::spawn_local(future);
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout, continue
                    continue;
                }
                Err(e) => {
                    error!("DEALER recv error: {}", e);
                    return Err(anyhow!("DEALER recv error: {}", e));
                }
            }
        }
    }

    /// Handle a command from the callback channel
    async fn handle_callback_command(&mut self, data: &[u8])
        -> Result<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)>
    {
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(data),
            ReaderOptions::new(),
        )?;
        let cmd = reader.get_root::<crate::model_capnp::inference_command::Reader>()?;

        use crate::model_capnp::inference_command::Which;
        match cmd.which()? {
            Which::LoadModel(_) => {
                // Already loaded, return success
                Ok((Self::build_load_model_response(true, "")?, None))
            }
            Which::Shutdown(()) => {
                info!("Received Shutdown command, returning");
                Err(anyhow!("Shutdown requested"))
            }
            Which::Infer(infer_data) => {
                let infer_data = infer_data?;
                // infer_data contains serialized InferenceRequest
                self.handle_callback_infer(infer_data).await
            }
        }
    }

    /// Run invariant checks before any TTT-scoped operation.
    /// Returns GuardStatus encoding expired, capacity, and generation state.
    /// The result must be passed to delta.adaptation_state.resolve().
    #[allow(dead_code)] // Reserved for dispatch-layer callers; inline guards kept where delta lock is held
    fn ttt_guard(&self, subject: &Subject) -> crate::training::GuardStatus {
        let (expired, at_capacity) = if let Some(pool) = &self.delta_pool {
            if let Some(delta_arc) = pool.get(subject) {
                let delta = delta_arc.lock();
                (delta.adaptation_state.is_expired(), delta.is_at_capacity())
            } else {
                (false, false)
            }
        } else {
            (false, false)
        };
        crate::training::GuardStatus {
            expired,
            at_capacity,
            lora_generation: self.lora_generation.load(std::sync::atomic::Ordering::Acquire),
        }
    }

    /// Handle inference request from callback channel
    async fn handle_callback_infer(&mut self, request_data: &[u8])
        -> Result<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)>
    {
        // Parse InferenceRequest
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(request_data),
            ReaderOptions::new(),
        )?;
        let req = reader.get_root::<inference_capnp::inference_request::Reader>()?;
        let request_id = req.get_id();

        // Create a context for the handler (callback mode uses local identity)
        // Note: Callback mode doesn't use signed envelopes, so we construct context directly
        use hyprstream_rpc::envelope::RequestEnvelope;
        use hyprstream_rpc::crypto::signing::generate_signing_keypair;

        let envelope = RequestEnvelope {
            request_id,
            identity: RequestIdentity::local(),
            payload: vec![],
            ephemeral_pubkey: None,
            nonce: [0u8; 16],
            timestamp: chrono::Utc::now().timestamp_millis(),
            claims: None,
        };

        // Create a minimal signed envelope for context extraction
        let (signing_key, _) = generate_signing_keypair();
        let signed = hyprstream_rpc::envelope::SignedEnvelope::new_signed(envelope, &signing_key);
        let ctx = EnvelopeContext::from_verified(&signed);

        // Dispatch via generated handler — propagate continuation
        dispatch_inference(self, &ctx, request_data).await
    }

    /// Build Register message
    fn build_register(id: &str, stream_endpoint: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut reg = message.init_root::<crate::model_capnp::register::Builder>();
            reg.set_id(id);
            reg.set_stream_endpoint(stream_endpoint);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build LoadModelCommandResponse
    fn build_load_model_response(success: bool, error: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut resp = message.init_root::<crate::model_capnp::load_model_command_response::Builder>();
            resp.set_success(success);
            resp.set_error(error);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Initialize the service
    async fn initialize(
        model_path: PathBuf,
        config: RuntimeConfig,
        server_pubkey: VerifyingKey,
        signing_key: SigningKey,
        nonce_cache: Arc<InMemoryNonceCache>,
        policy_client: PolicyClient,
        fs: Option<WorktreeClient>,
    ) -> Result<Self> {
        // Capture runtime handle for reuse in handlers
        let runtime_handle = Handle::current();

        let mut engine = TorchEngine::new(config.clone())?;
        RuntimeEngine::load_model(&mut engine, &model_path).await?;

        // Initialize KV cache registry
        let model_info = RuntimeEngine::model_info(&engine);
        let num_layers = model_info.num_hidden_layers.unwrap_or(32) as usize;
        let max_seq_len = config.max_context.unwrap_or(model_info.context_length) as usize;
        engine.initialize_kv_registry(num_layers, max_seq_len, config.kv_quant_type, None);

        info!(
            "KV cache registry initialized: {} layers, max_seq_len={}",
            num_layers, max_seq_len
        );

        // Check for training mode in config.json
        let training_config = ModelConfig::load_training_config(&model_path);

        // Only initialize TTT trainer and tokenizer if TTT is enabled
        // This ensures zero memory/compute overhead when not using TTT
        let (ttt_trainer, tokenizer): (Option<Arc<TestTimeTrainer>>, Option<Arc<Tokenizer>>) =
            if let Some(ref tc) = training_config {
                if tc.is_enabled() && tc.mode == TrainingMode::TestTimeTraining {
                    info!(
                        "Test-Time Training enabled: lr={}, steps={}",
                        tc.ttt.learning_rate, tc.ttt.gradient_steps
                    );

                    let ttt_config = TTTConfig {
                        learning_rate: tc.ttt.learning_rate,
                        gradient_steps: tc.ttt.gradient_steps,
                        max_grad_norm: tc.ttt.max_grad_norm,
                        min_input_length: tc.ttt.min_input_length,
                        max_ttt_context: tc.ttt.max_ttt_context,
                        enabled: true,
                        ..TTTConfig::default()
                    };

                    let device = engine.device();
                    let mut trainer = TestTimeTrainer::new(ttt_config, device);
                    // Wire gradient gating config if provided
                    if let Some(ref gating) = tc.ttt.gradient_gating {
                        trainer = trainer.with_gradient_gating(gating.clone());
                    }

                    // Get tokenizer for input tokenization
                    let tokenizer = engine.get_tokenizer().ok().map(Arc::new);
                    (Some(Arc::new(trainer)), tokenizer)
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

        // Initialize delta pool if TTT is enabled
        let delta_pool = if ttt_trainer.is_some() {
            let module_dims = engine.get_lora_module_dims().unwrap_or_default();
            let per_layer_dims = engine.get_per_layer_lora_dims();
            let device = engine.device();

            let base_delta_config = if let Some(ref tc) = training_config {
                let alpha = tc.lora_alpha.unwrap_or(tc.lora_rank as f32);
                TenantDeltaConfig {
                    rank: tc.lora_rank,
                    alpha,
                    target_modules: tc.target_modules.clone(),
                    learning_rate: tc.ttt.learning_rate,
                    ..TenantDeltaConfig::default()
                }
            } else {
                TenantDeltaConfig::default()
            };

            // Try to load TTN layer profile for non-uniform rank allocation
            let model_config_result = crate::runtime::model_config::ModelConfig::load(&model_path, &std::collections::HashMap::new());
            let delta_config = if let Ok(ref mc) = model_config_result {
                match crate::runtime::ttn_profile::get_layer_profile(&model_path, mc, None) {
                    Ok(profile) => {
                        let cfg = TenantDeltaConfig::from_profile(&base_delta_config, &profile);
                        info!(
                            "Delta pool: using TTN profile ({} layer overrides)",
                            cfg.layer_overrides.as_ref().map_or(0, std::collections::HashMap::len)
                        );
                        cfg
                    }
                    Err(e) => {
                        info!("TTN profile unavailable ({}), using uniform config", e);
                        base_delta_config
                    }
                }
            } else {
                base_delta_config
            };

            let kv_reg = engine.kv_registry();
            let snapshots_dir = model_path.join("adapters").join(".snapshots");
            let num_layers = engine.get_num_layers().unwrap_or(32);

            info!(
                "Delta pool initialized: rank={}, alpha={:.1}, modules={:?}, lr={:.1e}",
                delta_config.rank, delta_config.alpha, delta_config.target_modules, delta_config.learning_rate
            );
            let mut pool = DeltaPool::new(delta_config, module_dims, device, kv_reg, snapshots_dir, fs.clone(), num_layers);
            if let Some(pld) = per_layer_dims {
                pool = pool.with_per_layer_dims(pld);
            }
            // Wire rank oracle config if provided in training config
            if let Some(ref tc) = training_config {
                if let Some(ref oracle_config) = tc.ttt.rank_oracle {
                    pool = pool.with_rank_oracle(oracle_config.clone());
                    info!("Rank oracle enabled: interval={}, auto_adapt={}", oracle_config.adaptation_interval, oracle_config.auto_adapt);
                }
            }

            Some(Arc::new(pool))
        } else {
            None
        };

        // Create StreamChannel upfront (lazy socket init on first use)
        let stream_channel = StreamChannel::new(
            Arc::clone(&global_context()),
            signing_key.clone(),
        );

        Ok(InferenceService {
            inner: Arc::new(InferenceServiceInner {
                engine: parking_lot::RwLock::new(engine),
                model_path,
                session_id: parking_lot::RwLock::new(None),
                runtime_handle,
                stream_channel: Some(stream_channel),
                server_pubkey,
                signing_key,
                nonce_cache,
                policy_client,
                ttt_trainer,
                tokenizer,
                delta_pool,
                base_delta: Mutex::new(None),
                fs,
                zmq_context: global_context(),
                transport: hyprstream_rpc::transport::TransportConfig::inproc("inference-unset"),
                lora_generation: Arc::new(AtomicU64::new(0)),
            }),
        })
    }

    /// Resolve the effective delta for a subject: compose base_delta + tenant delta if both exist.
    ///
    /// Returns None if no deltas exist (base model only), which is the common case
    /// and incurs zero overhead.
    fn resolve_delta(
        &self,
        subject: &hyprstream_rpc::Subject,
    ) -> Option<Arc<Mutex<crate::training::TenantDelta>>> {
        let base = self.base_delta.lock().clone();
        let tenant = self.delta_pool.as_ref().and_then(|pool| pool.get(subject));

        match (base, tenant) {
            (Some(base), Some(tenant)) => {
                // Compose: base + tenant corrections
                Some(crate::training::TenantDelta::compose(&base, &tenant))
            }
            (Some(base), None) => Some(base),
            (None, Some(tenant)) => Some(tenant),
            (None, None) => None,
        }
    }

    /// Apply TTT adaptation if enabled (adapts model to input BEFORE generation)
    ///
    /// Returns:
    /// - Ok(Some(result)) if TTT was configured and ran (or was skipped)
    /// - Ok(None) if TTT is not configured
    /// - Err(e) if TTT failed unexpectedly
    ///
    /// Apply TTT adaptation with per-request overrides.
    ///
    /// Uses subject-specific delta pool for isolated per-session adaptation.
    async fn apply_ttt_adaptation_with_overrides(
        &self,
        prompt: &str,
        subject: &Subject,
        overrides: &crate::training::ttt::TTTOverrides,
    ) -> Result<Option<crate::training::ttt::TTTResult>> {
        use anyhow::anyhow;

        let ttt_trainer = match self.ttt_trainer.as_ref() {
            Some(t) => t,
            None => return Ok(None),  // TTT not configured
        };

        let pool = match self.delta_pool.as_ref() {
            Some(p) => p,
            None => return Ok(None),  // No delta pool
        };

        let tokenizer = match self.tokenizer.as_ref() {
            Some(t) => t,
            None => return Ok(None),  // No tokenizer available
        };

        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let input_tokens: Vec<u32> = encoding.get_ids().to_vec();

        // Ensure pool has capacity before creating/accessing delta
        // (must happen before acquiring engine read lock to avoid holding lock across await)
        pool.ensure_capacity().await?;

        let engine = self.engine.read();
        let delta_arc = pool.get_or_create(subject)?;

        // Lock the delta and run adaptation
        let mut delta = delta_arc.lock();

        // Note: capacity check is now handled by guard.at_capacity inside resolve().
        // The guard returns ResolveOutcome::Skipped if the delta is at capacity.

        // Snapshot Muon momentum and effective_ranks for rollback (before adapt_tenant
        // mutates them). adapt_tenant already returns the weight pre_snapshot, but Muon
        // and effective_ranks are mutated in-place during the gradient loop.
        let pre_muon = crate::training::muon::snapshot_muon_states(&delta.muon_states);
        let pre_eff_ranks = delta.effective_ranks.clone();

        let adapt_result = ttt_trainer.adapt_tenant(&engine, &mut delta, &input_tokens, overrides);

        // Early drop: release engine read lock before bookkeeping.
        // Delta has its own mutex; the LoRA generation counter detects reconfiguration.
        drop(engine);

        match adapt_result {
            Ok((mut result, pre_snapshot)) => {
                if !result.skipped {
                    debug!(
                        "TTT (subject {}): steps={}, improvement={:.4}, time={}ms, ppl={:.1}->{:.1}, rec={}",
                        subject,
                        result.steps_performed,
                        result.loss_improvement,
                        result.adaptation_time_ms,
                        result.initial_perplexity,
                        result.final_perplexity,
                        result.recommendation,
                    );
                }

                // Map overrides to AdaptationStrategy and drive the state machine.
                let strategy = overrides.adaptation_strategy.clone();

                // Build guard status for state machine.
                let guard = crate::training::GuardStatus {
                    expired: delta.adaptation_state.is_expired(),
                    at_capacity: delta.is_at_capacity(),
                    lora_generation: self.lora_generation.load(std::sync::atomic::Ordering::Acquire),
                };

                let outcome = delta.adaptation_state.resolve(
                    strategy,
                    &guard,
                    &result,
                    pre_snapshot,
                    pre_muon,
                    pre_eff_ranks,
                );

                match outcome {
                    crate::training::ResolveOutcome::WrittenBack => {
                        delta.accumulated_steps += result.steps_performed as u64;
                        delta.request_count += 1;
                        let n = delta.request_count as f64;
                        delta.avg_loss_improvement = delta.avg_loss_improvement * ((n - 1.0) / n)
                            + result.loss_improvement as f64 / n;
                        result.pending = false;
                        debug!("TTT: auto-committed adaptation for subject {}", subject);
                    }
                    crate::training::ResolveOutcome::Evicted { snapshot, muon, eff_ranks } => {
                        // resolve() transitioned state to Idle and returned the baseline
                        // snapshot (which may be the promoted old-pending baseline in the
                        // stacked case, or the new adaptation's pre-state otherwise).
                        // Restore delta weights, Muon momentum, and effective ranks.
                        let _ = delta.load_state_dict(&snapshot);
                        crate::training::muon::restore_muon_states(&mut delta.muon_states, &muon);
                        delta.effective_ranks = eff_ranks;
                        result.pending = false;
                        debug!("TTT: auto-rolled back adaptation for subject {}", subject);
                    }
                    crate::training::ResolveOutcome::StoredPending => {
                        // Pending state is stored inside delta.adaptation_state.
                        result.pending = true;
                        debug!("TTT: stored pending adaptation for subject {}", subject);
                    }
                    crate::training::ResolveOutcome::Skipped { reason } => {
                        debug!("TTT: skipped for subject {}: {}", subject, reason);
                    }
                }

                Ok(Some(result))
            }
            Err(e) => {
                warn!("TTT adaptation failed for subject {}: {}", subject, e);
                Err(e)
            }
        }
    }

    /// Prepare for streaming generation with DH-based key derivation.
    ///
    /// This is the first phase of streaming that runs BEFORE the REP response is sent.
    /// The actual streaming happens in `execute_stream` which runs AFTER the response.
    ///
    /// Uses `StreamContext::from_dh()` from hyprstream-rpc for DH key exchange:
    /// 1. Server generates ephemeral Ristretto255 keypair
    /// 2. Server computes shared secret: DH(server_secret, client_ephemeral_pubkey)
    /// 3. Both parties derive topic and mac_key from shared secret using HKDF
    ///
    /// # Returns
    ///
    /// (stream_id, server_pubkey, pending) where:
    /// - stream_id: For client display/logging (not used for routing)
    /// - server_pubkey: 32-byte Ristretto255 public key for client to derive same keys
    /// - pending: Stream state including StreamContext with DH-derived keys
    async fn prepare_stream(
        &self,
        request: GenerationRequest,
        client_ephemeral_pubkey: Option<&[u8]>,
        claims: Option<hyprstream_rpc::auth::Claims>,
        expiry_secs: i64,
        subject: &Subject,
        ttt_overrides: crate::training::ttt::TTTOverrides,
    ) -> Result<(String, [u8; 32], PendingWork)> {
        // TTT adaptation is deferred to execute_stream (runs in continuation after REP
        // is sent) to avoid blocking the ZMQ REQ/REP handler with GPU-intensive work.

        // DH key derivation is required - no legacy fallback
        let client_pub_bytes = client_ephemeral_pubkey
            .ok_or_else(|| anyhow!("Streaming requires client ephemeral pubkey for E2E authentication"))?;

        // Use StreamChannel for DH key exchange and pre-authorization
        let stream_channel = self.stream_channel.as_ref()
            .ok_or_else(|| anyhow!("StreamChannel not initialized"))?;

        let stream_ctx = stream_channel
            .prepare_stream_with_claims(client_pub_bytes, expiry_secs, claims)
            .await?;

        debug!(
            stream_id = %stream_ctx.stream_id(),
            topic = %stream_ctx.topic(),
            "Stream prepared via StreamChannel (DH + pre-authorization)"
        );

        let stream_id = stream_ctx.stream_id().to_owned();
        let server_pubkey = *stream_ctx.server_pubkey();

        // Delta lookup deferred to execute_stream (after TTT may modify it)
        let pending = PendingWork::Generation {
            stream_ctx,
            request,
            subject: subject.clone(),
            ttt_overrides,
        };

        Ok((stream_id, server_pubkey, pending))
    }

    /// Execute streaming generation - called AFTER REP response is sent.
    ///
    /// Uses StreamChannel for publishing with DH-derived topic.
    /// The topic is derived from DH shared secret, not guessable from stream_id.
    ///
    /// # Protocol (E2E Authenticated via DH)
    ///
    /// 1. Client calls generate_stream with ephemeral pubkey in envelope
    /// 2. Service generates ephemeral keypair, computes DH shared secret
    /// 3. Both derive topic and mac_key from DH using HKDF
    /// 4. Service returns server_pubkey in response (client derives same keys)
    /// 5. Client subscribes to DH-derived topic (unpredictable, non-colliding)
    /// 6. Service publishes chunks with HMAC chain (verified by client, not StreamService)
    /// 7. StreamService is blind forwarder (no HMAC verification)
    ///
    /// Note: The read lock must be held across await because TextStream<'_> borrows from the engine.
    /// tokio::sync::RwLock allows concurrent readers, but write-lock requests (createLora) will
    /// block until all active streams complete. GPU-intensive TTT is deferred to this continuation
    /// to avoid blocking the ZMQ REQ/REP handler.
    #[allow(clippy::await_holding_lock)]  // engine read-lock must be held while stream borrows it
    async fn execute_stream(&self, pending: PendingWork) {
        use futures::StreamExt;

        // SAFETY: parking_lot::Mutex is safe here because InferenceService runs on
        // current_thread runtime — no task migration, no .await while lock held.
        debug_assert!(
            tokio::runtime::Handle::current().runtime_flavor()
                == tokio::runtime::RuntimeFlavor::CurrentThread,
            "InferenceService must run on current_thread runtime for parking_lot safety"
        );

        let PendingWork::Generation { stream_ctx, request, subject, ttt_overrides } = pending else {
            error!("execute_stream called with non-Generation PendingWork");
            return;
        };
        let stream_ctx = &stream_ctx;

        // Get StreamChannel
        let stream_channel = match &self.stream_channel {
            Some(sc) => sc,
            None => {
                error!("StreamChannel not initialized for streaming");
                return;
            }
        };

        // Snapshot LoRA generation before TTT — used to detect reconfiguration mid-stream
        let lora_gen_before = self.lora_generation.load(Ordering::Acquire);

        // TTT runs HERE (in continuation, after REP sent — ZMQ loop unblocked).
        // If the user explicitly enabled TTT (ttt_enabled=true), failures are stream errors.
        // If TTT is only from server config, failures are non-fatal (warn + continue).
        let ttt_result = match self.apply_ttt_adaptation_with_overrides(
            request.prompt.as_str(), &subject, &ttt_overrides
        ).await {
            Ok(Some(result)) => Some(result),
            Ok(None) => {
                // TTT not available on this server (no trainer/pool/tokenizer configured)
                if ttt_overrides.enabled == Some(true) {
                    // User explicitly requested TTT — this is an error
                    error!(stream_id = %stream_ctx.stream_id(), subject = %subject, "TTT explicitly enabled but not available on server");
                    let sc = stream_channel;
                    if let Err(send_err) = sc.run_stream(stream_ctx, |publisher| async move {
                        (publisher, Err::<(), _>(anyhow::anyhow!("TTT explicitly enabled but not configured on this server")))
                    }).await {
                        error!(stream_id = %stream_ctx.stream_id(), error = %send_err,
                            "Failed to publish TTT error to stream");
                    }
                    return;
                }
                None  // TTT not requested or not configured — proceed without
            }
            Err(e) => {
                if ttt_overrides.enabled == Some(true) {
                    // User explicitly requested TTT — adaptation failure is a stream error
                    error!(stream_id = %stream_ctx.stream_id(), subject = %subject, error = %e, "TTT adaptation failed");
                    let sc = stream_channel;
                    if let Err(send_err) = sc.run_stream(stream_ctx, |publisher| async move {
                        (publisher, Err::<(), _>(anyhow::anyhow!("TTT adaptation failed: {}", e)))
                    }).await {
                        error!(stream_id = %stream_ctx.stream_id(), error = %send_err,
                            "Failed to publish TTT error to stream");
                    }
                    return;
                }
                // TTT running from server config only — warn and continue
                warn!("TTT adaptation failed (server-config), continuing without: {}", e);
                None
            }
        };

        // Re-resolve delta after TTT (may have been updated by adaptation)
        let delta = self.resolve_delta(&subject);

        trace!(
            stream_id = %stream_ctx.stream_id(),
            topic = %stream_ctx.topic(),
            has_delta = delta.is_some(),
            has_ttt = ttt_result.is_some(),
            "Starting E2E authenticated stream via StreamChannel"
        );

        // Check LoRA generation hasn't changed since we started (detects reconfiguration mid-stream)
        let lora_gen_now = self.lora_generation.load(Ordering::Acquire);
        if lora_gen_now != lora_gen_before {
            warn!(
                stream_id = %stream_ctx.stream_id(),
                "LoRA reconfigured during TTT adaptation (gen {} -> {}), aborting stream",
                lora_gen_before, lora_gen_now
            );
            let sc = stream_channel;
            if let Err(e) = sc.run_stream(stream_ctx, |publisher| async move {
                (publisher, Err::<(), _>(anyhow::anyhow!("LoRA adapter was reconfigured during adaptation — retry request")))
            }).await {
                error!(stream_id = %stream_ctx.stream_id(), error = %e, "Failed to publish LoRA error");
            }
            return;
        }

        // Run the stream with StreamChannel's async publisher callback.
        // Engine read-lock held across await: generate_with_delta returns a stream borrowing engine.
        let engine = self.engine.read();
        let stream_result = engine.generate_with_delta(request, delta);

        let result = stream_channel.run_stream(stream_ctx, |mut publisher| async move {
            let result = match stream_result {
                Ok(mut stream) => {
                    let cancel = stream_ctx.cancel_token();
                    loop {
                        let chunk_result = tokio::select! {
                            biased;
                            _ = cancel.cancelled() => {
                                let _ = publisher.publish_error("cancelled").await;
                                return (publisher, Ok(()));
                            }
                            next = stream.next() => match next {
                                Some(r) => r,
                                None => break,
                            },
                        };
                        match chunk_result {
                            Ok(text) => {
                                let rate = stream.stats().inference_tokens_per_sec_ema;
                                if let Err(e) = publisher.publish_data_with_rate(text.as_bytes(), rate).await {
                                    return (publisher, Err(e));
                                }
                            }
                            Err(e) => {
                                let _ = publisher.publish_error(&e.to_string()).await;
                                return (publisher, Ok(()));
                            }
                        }
                    }

                    // Normal completion with stats
                    let stats = stream.stats();
                    let mut complete = crate::services::rpc_types::InferenceComplete::from(&stats);

                    // Attach TTT metrics to completion (from deferred TTT in execute_stream)
                    complete.ttt_metrics = ttt_result.map(std::convert::Into::into);

                    match publisher.complete_ref(&complete.to_bytes()).await {
                        Ok(()) => Ok(()),
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),  // framework sends Error frame automatically
            };
            (publisher, result)
        }).await;

        if let Err(e) = result {
            error!(
                stream_id = %stream_ctx.stream_id(),
                error = %e,
                "Stream execution failed"
            );
        }
    }

    /// Execute streaming training step - called AFTER REP response is sent.
    ///
    /// Runs the training step in the background and publishes results via StreamChannel.
    /// This avoids REQ/REP timeout on long-running training (e.g., backward pass compilation).
    async fn execute_training_stream(&self, pending: PendingWork) {
        let PendingWork::Training { stream_ctx, subject, input, gradient_steps, learning_rate, adaptation_strategy } = pending else {
            error!("execute_training_stream called with non-Training PendingWork");
            return;
        };
        let stream_ctx = &stream_ctx;

        let stream_channel = match &self.stream_channel {
            Some(sc) => sc,
            None => {
                error!("StreamChannel not initialized for training stream");
                return;
            }
        };

        trace!(
            stream_id = %stream_ctx.stream_id(),
            "Starting training stream via StreamChannel"
        );

        let result = stream_channel.run_stream(stream_ctx, |mut publisher| async move {
            if stream_ctx.cancel_token().is_cancelled() {
                let _ = publisher.publish_error("cancelled").await;
                return (publisher, Ok(()));
            }
            let result = match self.handle_train_step(&subject, &input, gradient_steps, learning_rate, adaptation_strategy).await {
                Ok(result) => {
                    // Serialize training result as JSON for the completion payload
                    let payload = serde_json::to_vec(&result)
                        .unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}").into_bytes());
                    publisher.complete_ref(&payload).await
                }
                Err(e) => Err(e),  // framework sends Error frame automatically
            };
            (publisher, result)
        }).await;

        if let Err(e) = result {
            error!(
                stream_id = %stream_ctx.stream_id(),
                error = %e,
                "Training stream execution failed"
            );
        }
    }

    /// Handle model info request
    async fn handle_model_info(&self) -> ModelInfo {
        RuntimeEngine::model_info(&*self.engine.read())
    }

    /// Handle is ready request
    async fn handle_is_ready(&self) -> bool {
        self.engine.read().is_loaded()
    }

    /// Handle apply chat template
    async fn handle_apply_chat_template(
        &self,
        messages: Vec<crate::runtime::template_engine::ChatMessage>,
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        self.engine
            .read()
            .apply_chat_template(&messages, add_generation_prompt, tools)
    }

    /// Handle create LoRA
    async fn handle_create_lora(&self, config: TenantDeltaConfig) -> Result<()> {
        // Propagate target modules to the delta pool so new deltas
        // create A/B matrices for ALL configured modules, not just the default q_proj/v_proj
        if let Some(pool) = &self.delta_pool {
            tracing::info!(
                "[TTT] Updating delta pool config: target_modules={:?}, rank={}, alpha={:.1}, lr={:.1e}",
                config.target_modules, config.rank, config.alpha, config.learning_rate
            );
            pool.update_config(config.clone());
        }
        let result = self.engine.write().create_lora(config);
        if result.is_ok() {
            self.lora_generation.fetch_add(1, Ordering::Release);
        }
        result
    }

    // =========================================================================
    // Generic streaming setup (used by streaming handler variants)
    // =========================================================================

    /// Set up a streaming context for a handler that needs to run work in a continuation.
    ///
    /// Returns `(StreamInfo, StreamContext)` — the caller builds the continuation
    /// and returns `Ok((stream_info, continuation))`.
    async fn setup_stream(
        &self,
        ctx: &EnvelopeContext,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::streaming::StreamContext)> {
        let client_pub_bytes = ctx.ephemeral_pubkey()
            .ok_or_else(|| anyhow!("Streaming requires client ephemeral pubkey for E2E authentication"))?;
        let stream_channel = self.stream_channel.as_ref()
            .ok_or_else(|| anyhow!("StreamChannel not initialized"))?;

        let expiry_secs = ctx.claims()
            .map(|c| c.exp - chrono::Utc::now().timestamp())
            .unwrap_or(300)
            .max(60);
        let claims = ctx.claims().cloned();

        let stream_ctx = stream_channel.prepare_stream_with_claims(client_pub_bytes, expiry_secs, claims).await?;

        let stream_id = stream_ctx.stream_id().to_owned();
        let server_pubkey = *stream_ctx.server_pubkey();
        let stream_sub_endpoint = hyprstream_rpc::registry::global()
            .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
            .to_zmq_string();

        let stream_info = crate::services::generated::inference_client::StreamInfo {
            stream_id,
            endpoint: stream_sub_endpoint,
            server_pubkey,
        };

        Ok((stream_info, stream_ctx))
    }

    // =========================================================================
    // Streaming execution helpers (called from continuations)
    // =========================================================================

    async fn execute_create_lora_stream(
        &self,
        stream_ctx: hyprstream_rpc::streaming::StreamContext,
        config: crate::training::TenantDeltaConfig,
    ) {
        let sc = match &self.stream_channel {
            Some(sc) => sc,
            None => { error!("StreamChannel not initialized"); return; }
        };
        let _ = sc.run_stream(&stream_ctx, |mut publisher| async {
            let result = InferenceService::handle_create_lora(self, config).await;
            match &result {
                Ok(()) => { let _ = publisher.complete_ref(b"{}").await; }
                Err(e) => { let _ = publisher.publish_error(&format!("{:#}", e)).await; }
            }
            (publisher, result.map(|_| ()))
        }).await;
    }

    async fn execute_load_lora_stream(&self, stream_ctx: hyprstream_rpc::streaming::StreamContext, path: String) {
        let sc = match &self.stream_channel {
            Some(sc) => sc,
            None => { error!("StreamChannel not initialized"); return; }
        };
        let _ = sc.run_stream(&stream_ctx, |mut publisher| async {
            let result = self.load_lora(Path::new(&path)).await;
            match &result {
                Ok(()) => { let _ = publisher.complete_ref(b"{}").await; }
                Err(e) => { let _ = publisher.publish_error(&format!("{:#}", e)).await; }
            }
            (publisher, result)
        }).await;
    }

    async fn execute_save_lora_stream(&self, stream_ctx: hyprstream_rpc::streaming::StreamContext, name: String) {
        let sc = match &self.stream_channel {
            Some(sc) => sc,
            None => { error!("StreamChannel not initialized"); return; }
        };
        let _ = sc.run_stream(&stream_ctx, |mut publisher| async {
            let result = self.save_lora(&name).await;
            match &result {
                Ok(()) => { let _ = publisher.complete_ref(b"{}").await; }
                Err(e) => { let _ = publisher.publish_error(&format!("{:#}", e)).await; }
            }
            (publisher, result)
        }).await;
    }

    async fn execute_save_adaptation_stream(
        &self,
        stream_ctx: hyprstream_rpc::streaming::StreamContext,
        subject: Subject,
        name: String,
        merge_strategy: String,
        merge_weight: f32,
    ) {
        let sc = match &self.stream_channel {
            Some(sc) => sc,
            None => { error!("StreamChannel not initialized"); return; }
        };
        let _ = sc.run_stream(&stream_ctx, |mut publisher| async {
            let result = InferenceService::handle_save_adaptation(
                self, &subject, &name, &merge_strategy, merge_weight,
            ).await;
            match &result {
                Ok(info) => {
                    let payload = serde_json::to_vec(&serde_json::json!({
                        "adapter_name": info.adapter_name,
                        "adapter_path": info.adapter_path,
                        "content_hash": info.content_hash,
                        "merge_strategy": info.merge_strategy,
                    })).unwrap_or_default();
                    let _ = publisher.complete_ref(&payload).await;
                }
                Err(e) => { let _ = publisher.publish_error(&format!("{:#}", e)).await; }
            }
            (publisher, result.map(|_| ()))
        }).await;
    }

    async fn execute_snapshot_delta_stream(
        &self,
        stream_ctx: hyprstream_rpc::streaming::StreamContext,
        subject: Subject,
    ) {
        let sc = match &self.stream_channel {
            Some(sc) => sc,
            None => { error!("StreamChannel not initialized"); return; }
        };
        let _ = sc.run_stream(&stream_ctx, |mut publisher| async {
            let result = InferenceService::handle_snapshot_delta(self, &subject).await;
            match &result {
                Ok(info) => {
                    let payload = serde_json::to_vec(&serde_json::json!({
                        "content_hash": info.content_hash,
                        "size_bytes": info.size_bytes,
                        "accumulated_steps": info.accumulated_steps,
                        "request_count": info.request_count,
                    })).unwrap_or_default();
                    let _ = publisher.complete_ref(&payload).await;
                }
                Err(e) => { let _ = publisher.publish_error(&format!("{:#}", e)).await; }
            }
            (publisher, result.map(|_| ()))
        }).await;
    }

    async fn execute_export_peft_adapter_stream(
        &self,
        stream_ctx: hyprstream_rpc::streaming::StreamContext,
        subject: Subject,
        name: String,
        commit_message: String,
    ) {
        let sc = match &self.stream_channel {
            Some(sc) => sc,
            None => { error!("StreamChannel not initialized"); return; }
        };
        let _ = sc.run_stream(&stream_ctx, |mut publisher| async {
            let result = InferenceService::handle_export_peft_adapter(
                self, &subject, &name, &commit_message,
            ).await;
            match &result {
                Ok(info) => {
                    let payload = serde_json::to_vec(&serde_json::json!({
                        "adapter_path": info.adapter_path,
                        "content_hash": info.content_hash,
                    })).unwrap_or_default();
                    let _ = publisher.complete_ref(&payload).await;
                }
                Err(e) => { let _ = publisher.publish_error(&format!("{:#}", e)).await; }
            }
            (publisher, result.map(|_| ()))
        }).await;
    }

    async fn execute_merge_lora_stream(
        &self,
        stream_ctx: hyprstream_rpc::streaming::StreamContext,
        adapter_path: String,
        weight: f32,
        strategy: String,
    ) {
        let sc = match &self.stream_channel {
            Some(sc) => sc,
            None => { error!("StreamChannel not initialized"); return; }
        };
        let _ = sc.run_stream(&stream_ctx, |mut publisher| async {
            let result = self.merge_lora(Path::new(&adapter_path), weight, &strategy).await;
            match &result {
                Ok(()) => { let _ = publisher.complete_ref(b"{}").await; }
                Err(e) => { let _ = publisher.publish_error(&format!("{:#}", e)).await; }
            }
            (publisher, result)
        }).await;
    }

    // =========================================================================
    // Training loop control handlers (tenant-aware TTT)
    // =========================================================================

    /// Write back a pending TTT adaptation (commit path)
    fn handle_writeback(&self, subject: &Subject) -> Result<()> {
        if let Some(pool) = &self.delta_pool {
            if let Some(delta_arc) = pool.get(subject) {
                let mut delta = delta_arc.lock();
                let info = delta.adaptation_state.writeback_stats()?;
                delta.accumulated_steps += info.steps_performed as u64;
                delta.request_count += 1;
                let n = delta.request_count as f64;
                delta.avg_loss_improvement = delta.avg_loss_improvement * ((n - 1.0) / n)
                    + info.loss_improvement as f64 / n;
                debug!("Wrote back adaptation for '{}': steps={}, improvement={:.4}",
                    subject, info.steps_performed, info.loss_improvement);
                Ok(())
            } else {
                Err(anyhow::anyhow!("No delta found for subject '{}'", subject))
            }
        } else {
            Err(anyhow::anyhow!("No delta pool configured"))
        }
    }

    /// Evict a pending TTT adaptation (rollback path)
    fn handle_evict(&self, subject: &Subject) -> Result<()> {
        if let Some(pool) = &self.delta_pool {
            if let Some(delta_arc) = pool.get(subject) {
                let mut delta = delta_arc.lock();
                let snapshot = delta.adaptation_state.evict()?;
                let _ = delta.load_state_dict(&snapshot.pre_snapshot);
                crate::training::muon::restore_muon_states(&mut delta.muon_states, &snapshot.pre_muon);
                delta.effective_ranks = snapshot.pre_eff_ranks;
                debug!("Evicted adaptation for subject '{}'", subject);
                Ok(())
            } else {
                Err(anyhow::anyhow!("No delta found for subject '{}'", subject))
            }
        } else {
            Err(anyhow::anyhow!("No delta pool configured"))
        }
    }

    /// Run pure training steps without generation
    async fn handle_train_step(
        &self,
        subject: &Subject,
        input: &str,
        gradient_steps: u32,
        learning_rate: f32,
        strategy: crate::training::adaptation_state::AdaptationStrategy,
    ) -> Result<crate::training::ttt::TTTResult> {
        let ttt_trainer = self.ttt_trainer.as_ref()
            .ok_or_else(|| anyhow!("TTT not configured"))?;
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow!("No tokenizer available"))?;
        let pool = self.delta_pool.as_ref()
            .ok_or_else(|| anyhow!("Delta pool not initialized"))?;

        // Ensure pool has capacity before creating/accessing delta
        pool.ensure_capacity().await?;
        let delta_arc = pool.get_or_create(subject)?;

        // Phase 1: Sync work under scoped lock — tokenize and snapshot.
        // Always snapshot (even for auto_commit) so auto-rollback can restore state.
        let (input_tokens, pre_snapshot, pre_muon, pre_eff_ranks) = {
            let delta = delta_arc.lock();
            let encoding = tokenizer.encode(input, false)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
            let input_tokens: Vec<u32> = encoding.get_ids().to_vec();
            let pre_snapshot = delta.extract_state_dict();
            let pre_muon = crate::training::muon::snapshot_muon_states(&delta.muon_states);
            let pre_eff_ranks = delta.effective_ranks.clone();
            (input_tokens, pre_snapshot, pre_muon, pre_eff_ranks)
        }; // guard dropped

        let steps = if gradient_steps > 0 { gradient_steps as usize } else { ttt_trainer.config.gradient_steps as usize };
        let lr = if learning_rate > 0.0 { Some(learning_rate as f64) } else { None };

        // Phase 2: Await engine (no lock held — safe for spawn_local concurrency)
        let engine = self.engine.read();

        // Phase 3: Reacquire delta lock for train_step
        let mut delta = delta_arc.lock();
        let mut result = ttt_trainer.train_step(&engine, &mut delta, &input_tokens, steps, lr, None)?;

        // strategy is passed in directly from the caller

        // Build guard status
        let guard = crate::training::GuardStatus {
            expired: delta.adaptation_state.is_expired(),
            at_capacity: delta.is_at_capacity(),
            lora_generation: self.lora_generation.load(std::sync::atomic::Ordering::Acquire),
        };

        let outcome = delta.adaptation_state.resolve(
            strategy,
            &guard,
            &result,
            pre_snapshot,
            pre_muon,
            pre_eff_ranks,
        );

        match outcome {
            crate::training::ResolveOutcome::WrittenBack => {
                delta.accumulated_steps += result.steps_performed as u64;
                delta.request_count += 1;
                let n = delta.request_count as f64;
                delta.avg_loss_improvement = delta.avg_loss_improvement * ((n - 1.0) / n)
                    + result.loss_improvement as f64 / n;
                result.pending = false;
                debug!("TTT: auto-committed train_step adaptation for subject {}", subject);
            }
            crate::training::ResolveOutcome::Evicted { snapshot, muon, eff_ranks } => {
                let _ = delta.load_state_dict(&snapshot);
                crate::training::muon::restore_muon_states(&mut delta.muon_states, &muon);
                delta.effective_ranks = eff_ranks;
                result.pending = false;
                debug!("TTT: auto-rolled back train_step adaptation for subject {} (negative recommendation)", subject);
            }
            crate::training::ResolveOutcome::StoredPending => {
                result.pending = true;
                debug!("TTT: stored pending train_step adaptation for subject {}", subject);
            }
            crate::training::ResolveOutcome::Skipped { reason } => {
                debug!("TTT: train_step skipped for subject {}: {}", subject, reason);
            }
        }

        Ok(result)
    }

    /// Reset a tenant's delta to zeros
    fn handle_reset_delta(&self, subject: &Subject) -> Result<()> {
        if let Some(pool) = &self.delta_pool {
            if let Some(delta_arc) = pool.get(subject) {
                let mut delta = delta_arc.lock();
                delta.reset();
                debug!("Reset delta for subject '{}'", subject);
            } else {
                debug!("No delta to reset for subject '{}'", subject);
            }
        }

        Ok(())
    }

    /// Get status of a tenant's delta
    fn handle_get_delta_status(&self, subject: &Subject) -> Result<DeltaStatusInfo> {
        if let Some(pool) = &self.delta_pool {
            if let Some(delta_arc) = pool.get(subject) {
                let delta = delta_arc.lock();
                let norm_ratios = delta.delta_norm_ratio(pool.base_weight_norms());
                let has_pending = delta.adaptation_state.is_pending();
                return Ok(DeltaStatusInfo {
                    exists: true,
                    accumulated_steps: delta.accumulated_steps,
                    max_accumulated_steps: delta.max_accumulated_steps,
                    request_count: delta.request_count,
                    avg_loss_improvement: delta.avg_loss_improvement as f32,
                    memory_bytes: delta.memory_bytes() as u64,
                    last_snapshot_hash: delta.last_snapshot_hash.clone().unwrap_or_default(),
                    delta_norm_ratios: norm_ratios,
                    has_pending,
                });
            }
        }

        Ok(DeltaStatusInfo {
            exists: false,
            accumulated_steps: 0,
            max_accumulated_steps: 0,
            request_count: 0,
            avg_loss_improvement: 0.0,
            memory_bytes: 0,
            last_snapshot_hash: String::new(),
            delta_norm_ratios: std::collections::HashMap::new(),
            has_pending: false,
        })
    }

    /// Save a tenant's delta as a permanent LoRA adapter
    ///
    /// Supports merge strategies: "replace", "additive", "do_merge" (default).
    /// When an existing adapter exists, the delta is merged using the specified strategy.
    async fn handle_save_adaptation(
        &self,
        subject: &Subject,
        name: &str,
        merge_strategy_name: &str,
        merge_weight: f32,
    ) -> Result<SaveAdaptationInfo> {
        use crate::training::{MergeStrategy, merge_state_dicts};

        let pool = self.delta_pool.as_ref()
            .ok_or_else(|| anyhow!("Delta pool not initialized"))?;

        let delta_arc = pool.get(subject)
            .ok_or_else(|| anyhow!("No delta for subject '{}'", subject))?;
        let new_state_dict = {
            let delta = delta_arc.lock();
            delta.extract_state_dict()
        };

        // Parse merge strategy (default to DO-Merge)
        let strategy_name = if merge_strategy_name.is_empty() { "do_merge" } else { merge_strategy_name };
        let weight = if merge_weight <= 0.0 || merge_weight > 1.0 { 0.3 } else { merge_weight as f64 };
        let strategy = MergeStrategy::from_name(strategy_name, weight)?;

        // Save as adapter file
        let adapter_mgr = crate::storage::AdapterManager::new(&self.model_path);
        let adapter_name = if name.is_empty() {
            format!("ttt_{}", subject)
        } else {
            name.to_owned()
        };

        // Check for existing adapter to merge with (loads via FsOps)
        let existing_adapters = adapter_mgr.list_adapters().unwrap_or_default();
        let existing_state = if let Some(existing) = existing_adapters.iter().find(|a| a.name == adapter_name) {
            let rel_path = format!("adapters/{}", existing.path.file_name()
                .and_then(|f| f.to_str()).unwrap_or(""));
            if let Some(ref fs) = self.fs {
                match fs.read_file_chunked(&rel_path).await {
                    Ok(bytes) => {
                        crate::training::load_state_dict_from_bytes(&bytes)
                            .map_err(|e| {
                                tracing::debug!("Could not load existing adapter '{}': {}", existing.name, e);
                                e
                            })
                            .ok()
                    }
                    Err(e) => {
                        tracing::debug!("Could not read existing adapter '{}': {}", existing.name, e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // Determine adapter filename
        let adapter_filename = if let Some(existing) = existing_adapters.iter().find(|a| a.name == adapter_name) {
            existing.path.file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("adapter.safetensors")
                .to_owned()
        } else {
            let next_index = adapter_mgr.get_next_index().unwrap_or(0);
            format!("{:02}_{}.safetensors", next_index, adapter_name)
        };

        // Apply merge strategy
        let final_state = if let Some(existing) = existing_state {
            merge_state_dicts(&existing, &new_state_dict, &strategy)?
        } else {
            new_state_dict
        };

        // Write adapter file through FsOps (path-contained)
        let fs = self.fs.as_ref()
            .ok_or_else(|| anyhow!("FsOps not available — cannot write without path containment"))?;
        let rel_path = format!("adapters/{}", adapter_filename);
        fs.mkdir_p("adapters").await
            .map_err(|e| anyhow!("FsOps mkdir failed: {}", e))?;
        let bytes = serialize_state_dict_to_bytes(&final_state)?;
        fs.write_file_chunked(&rel_path, &bytes).await
            .map_err(|e| anyhow!("FsOps write_file failed: {}", e))?;
        let result_path = rel_path;

        let actual_strategy = format!("{:?}", strategy).to_lowercase();

        info!(
            "Saved adaptation for subject '{}' as adapter '{}' at {} (strategy: {})",
            subject, adapter_name, result_path, actual_strategy
        );

        Ok(SaveAdaptationInfo {
            adapter_name: adapter_name.clone(),
            adapter_path: result_path,
            content_hash: String::new(),
            merge_strategy: strategy_name.to_owned(),
        })
    }

    /// Snapshot a tenant's delta to a file
    async fn handle_snapshot_delta(&self, subject: &Subject) -> Result<SnapshotDeltaInfo> {
        let pool = self.delta_pool.as_ref()
            .ok_or_else(|| anyhow!("Delta pool not initialized"))?;

        let delta_arc = pool.get(subject)
            .ok_or_else(|| anyhow!("No delta for subject '{}'", subject))?;

        let filename = subject.to_string();
        let state_dict = {
            let delta = delta_arc.lock();
            delta.extract_state_dict()
        };

        // Write snapshot through FsOps (path-contained)
        let fs = self.fs.as_ref()
            .ok_or_else(|| anyhow!("FsOps not available — cannot write without path containment"))?;
        let rel_snapshot = format!("adapters/.snapshots/{}.safetensors", filename);
        fs.mkdir_p("adapters/.snapshots").await
            .map_err(|e| anyhow!("FsOps mkdir failed: {}", e))?;
        let bytes = serialize_state_dict_to_bytes(&state_dict)?;
        let size_bytes = bytes.len() as u64;
        fs.write_file_chunked(&rel_snapshot, &bytes).await
            .map_err(|e| anyhow!("FsOps write_file failed: {}", e))?;
        let path_str = rel_snapshot;

        // Re-acquire lock to update delta state
        let mut delta = delta_arc.lock();
        delta.last_snapshot_hash = Some(path_str.clone());

        Ok(SnapshotDeltaInfo {
            content_hash: path_str,
            size_bytes,
            accumulated_steps: delta.accumulated_steps,
            request_count: delta.request_count,
        })
    }

    /// Export a tenant's delta as a PEFT-compatible adapter directory.
    ///
    /// Creates `adapters/{name}/adapter_model.safetensors` with HuggingFace PEFT naming
    /// and `adapters/{name}/adapter_config.json` with PEFT metadata.
    async fn handle_export_peft_adapter(
        &self,
        subject: &Subject,
        name: &str,
        _commit_message: &str,
    ) -> Result<ExportPeftInfo> {
        let pool = self.delta_pool.as_ref()
            .ok_or_else(|| anyhow!("Delta pool not initialized"))?;

        let delta_arc = pool.get(subject)
            .ok_or_else(|| anyhow!("No delta for subject '{}'", subject))?;

        let adapter_name = if name.is_empty() {
            format!("ttt_{}", subject)
        } else {
            name.to_owned()
        };

        // Extract all data from delta under lock, then drop lock before await
        let (safetensors_bytes, config_bytes) = {
            let delta = delta_arc.lock();

            // Serialize as PEFT-compatible safetensors (with HuggingFace key naming)
            let safetensors_bytes = delta.serialize_to_safetensors_bytes()?;

            // Generate PEFT adapter_config.json
            let adapter_config = serde_json::json!({
                "peft_type": "LORA",
                "auto_mapping": null,
                "base_model_name_or_path": "",
                "bias": "none",
                "fan_in_fan_out": false,
                "inference_mode": true,
                "init_lora_weights": true,
                "layers_to_transform": null,
                "layers_pattern": null,
                "lora_alpha": delta.scaling * delta.rank as f64,
                "lora_dropout": 0.0,
                "modules_to_save": null,
                "r": delta.rank,
                "rank_pattern": {},
                "alpha_pattern": {},
                "revision": null,
                "target_modules": delta.target_modules.clone(),
                "task_type": "CAUSAL_LM"
            });
            let config_bytes = serde_json::to_vec_pretty(&adapter_config)?;
            (safetensors_bytes, config_bytes)
        };

        // Write through FsOps (path-contained)
        let fs = self.fs.as_ref()
            .ok_or_else(|| anyhow!("FsOps not available — cannot write without path containment"))?;

        let dir_path = format!("adapters/{}", adapter_name);
        fs.mkdir_p(&dir_path).await
            .map_err(|e| anyhow!("FsOps mkdir failed: {}", e))?;

        let safetensors_path = format!("{}/adapter_model.safetensors", dir_path);
        fs.write_file_chunked(&safetensors_path, &safetensors_bytes).await
            .map_err(|e| anyhow!("FsOps write adapter_model.safetensors failed: {}", e))?;

        let config_path = format!("{}/adapter_config.json", dir_path);
        fs.write_file_chunked(&config_path, &config_bytes).await
            .map_err(|e| anyhow!("FsOps write adapter_config.json failed: {}", e))?;

        info!(
            "Exported PEFT adapter for subject '{}' as '{}' ({} bytes safetensors)",
            subject, adapter_name, safetensors_bytes.len()
        );

        Ok(ExportPeftInfo {
            adapter_path: dir_path,
            content_hash: String::new(),
        })
    }

    /// Get TTT configuration (for status queries)
    #[allow(dead_code)]
    fn get_ttt_config(&self) -> Option<crate::training::ttt::TTTConfig> {
        self.ttt_trainer.as_ref().map(|trainer| trainer.config.clone())
    }

    /// Handle set session
    async fn handle_set_session(&self, session_id: String) -> Result<()> {
        // Track session ID for events
        *self.session_id.write() = Some(session_id.clone());
        self.engine
            .write()
            .set_session(CacheOwner::Session(session_id))
    }

    /// Handle clear session
    async fn handle_clear_session(&self) {
        *self.session_id.write() = None;
        self.engine.write().clear_kv_cache();
    }

    /// Handle release session
    async fn handle_release_session(&self, session_id: &str) -> Result<()> {
        self.engine
            .write()
            .release_session(&CacheOwner::Session(session_id.to_owned()))
    }

    /// Load a PEFT adapter directory as the base delta.
    ///
    /// `path` is a PEFT adapter directory (e.g. `adapters/my-adapter`) containing
    /// `adapter_model.safetensors`. The loaded adapter is stored as `base_delta`
    /// and applied to all inference requests. If a per-tenant TTT delta also exists,
    /// the two are composed (corrections summed) during inference via `resolve_delta()`.
    pub async fn load_lora(&self, path: &Path) -> Result<()> {
        let device = self.engine.read().device();
        // Read via FsOps (path-contained)
        let fs = self.fs.as_ref()
            .ok_or_else(|| anyhow!("FsOps not available — cannot read without path containment"))?;
        let safetensors_path = format!("{}/adapter_model.safetensors", path.to_string_lossy());
        let bytes = fs.read_file_chunked(&safetensors_path).await
            .map_err(|e| anyhow!("Failed to read LoRA adapter from {}: {}", safetensors_path, e))?;
        let delta = crate::training::TenantDelta::load_from_safetensors_bytes(&bytes, device)?;

        // Warn if the adapter rank differs from the delta pool rank —
        // compose() now handles this via effective-weight decomposition,
        // but it's worth logging for visibility.
        if let Some(pool) = self.delta_pool.as_ref() {
            let pool_rank = pool.rank();
            if delta.rank != pool_rank {
                tracing::warn!(
                    "LoRA adapter rank ({}) differs from delta pool rank ({}). \
                     Composition will use effective-weight decomposition (slower but correct).",
                    delta.rank, pool_rank
                );
            }
        }

        *self.base_delta.lock() = Some(Arc::new(Mutex::new(delta)));
        self.lora_generation.fetch_add(1, Ordering::Release);
        tracing::info!("Loaded LoRA adapter as base delta from {}", path.display());
        Ok(())
    }

    /// Save the current base delta to a safetensors file.
    pub async fn save_lora(&self, path: &str) -> Result<()> {
        let base = self.base_delta.lock().clone();
        if let Some(delta_arc) = base {
            let bytes = {
                let delta = delta_arc.lock();
                delta.serialize_to_safetensors_bytes()?
            };
            // Sanitize name and write via FsOps (path-contained)
            let fs = self.fs.as_ref()
                .ok_or_else(|| anyhow!("FsOps not available — cannot write without path containment"))?;
            let safe_name = sanitize_adapter_name(path)?;
            let rel_path = format!("adapters/{}.safetensors", safe_name);
            fs.mkdir_p("adapters").await
                .map_err(|e| anyhow!("FsOps mkdir failed: {}", e))?;
            fs.write_file_chunked(&rel_path, &bytes).await
                .map_err(|e| anyhow!("FsOps write_file failed: {}", e))?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("No LoRA adapter loaded to save"))
        }
    }

    /// Unload the current base LoRA adapter.
    pub async fn unload_lora(&self) -> Result<()> {
        let mut base = self.base_delta.lock();
        if base.is_some() {
            *base = None;
            self.lora_generation.fetch_add(1, Ordering::Release);
            tracing::info!("Unloaded base LoRA delta");
            Ok(())
        } else {
            Err(anyhow::anyhow!("No LoRA adapter loaded to unload"))
        }
    }

    /// Check if a LoRA adapter (base delta) is loaded.
    pub async fn has_lora(&self) -> Result<bool> {
        Ok(self.base_delta.lock().is_some())
    }

    /// Merge an on-disk PEFT adapter into the currently loaded base_delta.
    ///
    /// Reads the adapter from disk via WorktreeClient, loads it as a TenantDelta,
    /// then merges its corrections into the existing base_delta using the specified
    /// merge strategy. Requires a base_delta to already be loaded.
    pub async fn merge_lora(&self, path: &Path, weight: f32, strategy_name: &str) -> Result<()> {
        let device = self.engine.read().device();

        // Read the adapter from disk via FsOps
        let fs = self.fs.as_ref()
            .ok_or_else(|| anyhow!("FsOps not available — cannot read without path containment"))?;
        let safetensors_path = format!("{}/adapter_model.safetensors", path.to_string_lossy());
        let bytes = fs.read_file_chunked(&safetensors_path).await
            .map_err(|e| anyhow!("Failed to read adapter for merge: {}", e))?;
        let incoming = crate::training::TenantDelta::load_from_safetensors_bytes(&bytes, device)?;

        // Get existing base_delta
        let base_arc = self.base_delta.lock().clone()
            .ok_or_else(|| anyhow!("No adapter loaded in base_delta register. Load one first with adapter.load."))?;

        // Parse merge strategy
        let strategy = crate::training::merge::MergeStrategy::from_name(strategy_name, weight as f64)?;

        // Extract state dicts, merge, and update
        let merged_state = {
            let base = base_arc.lock();
            let existing = base.extract_state_dict();
            let new = incoming.extract_state_dict();
            crate::training::merge::merge_state_dicts(&existing, &new, &strategy)?
        };

        // Load merged weights back into the base_delta
        {
            let mut base = base_arc.lock();
            base.load_state_dict(&merged_state)?;
        }

        tracing::info!(
            "Merged adapter from {} into base_delta (strategy={}, weight={})",
            path.display(), strategy_name, weight
        );
        Ok(())
    }
}

/// Sanitize an adapter name to prevent path traversal.
///
/// Strips path separators, `..`, and file extensions. Returns a safe filename stem.
/// Only allows alphanumeric characters, underscores, and hyphens.
fn sanitize_adapter_name(name: &str) -> Result<String> {
    let stem = std::path::Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name);

    // Reject path traversal
    if stem.contains("..") || stem.contains('/') || stem.contains('\\') || stem.is_empty() {
        return Err(anyhow!("Invalid adapter name: '{}'", name));
    }

    // Only allow alphanumeric, underscore, hyphen
    let safe: String = stem
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect();

    if safe.is_empty() {
        return Err(anyhow!("Adapter name '{}' contains no valid characters", name));
    }

    Ok(safe)
}

// ═══════════════════════════════════════════════════════════════════════════════
// InferenceHandler Implementation — generated dispatch for typed handler trait
// ═══════════════════════════════════════════════════════════════════════════════

use crate::services::generated::inference_client::{
    InferenceHandler, dispatch_inference, serialize_response,
    InferenceResponseVariant, ErrorInfo,
    HealthStatus, DeltaStatusResult, ModuleNormRatio,
    SaveAdaptationResult, SnapshotDeltaResult, ExportPeftResult,
    ChatTemplateRequest, LoraConfig, TrainStepRequest, SaveAdaptationRequest, ExportPeftRequest,
    MergeLoraRequest, EmbedImagesRequest, EmbedImagesResponse,
    AdaptationStrategyEnum,
    GenerationRequest as InfGenerationRequest,
};
use crate::services::generated::policy_client::PolicyCheck;
// Conflicting names — use canonical path at usage sites:
//   inference_client::GenerationResult, inference_client::ModelInfo, inference_client::StreamInfo

/// Map a capnp AdaptationStrategyEnum + optional threshold to the internal AdaptationStrategy.
fn map_adaptation_strategy(
    capnp_strategy: AdaptationStrategyEnum,
    writeback_threshold: f32,
) -> crate::training::adaptation_state::AdaptationStrategy {
    match capnp_strategy {
        AdaptationStrategyEnum::AutoWriteback => {
            crate::training::adaptation_state::AdaptationStrategy::AutoWriteback
        }
        AdaptationStrategyEnum::AutoEvict => {
            crate::training::adaptation_state::AdaptationStrategy::AutoEvict
        }
        AdaptationStrategyEnum::Speculative => {
            crate::training::adaptation_state::AdaptationStrategy::Speculative
        }
        AdaptationStrategyEnum::WritebackIfAbove => {
            crate::training::adaptation_state::AdaptationStrategy::WritebackIfAbove {
                threshold: writeback_threshold,
            }
        }
    }
}

#[async_trait::async_trait(?Send)]
impl InferenceHandler for InferenceService {
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

    async fn handle_generate_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &crate::services::generated::inference_client::GenerationRequest,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let subject = ctx.subject();
        let request = data.clone();

        // Build per-request TTT overrides from RPC fields
        let ttt_overrides = crate::training::ttt::TTTOverrides {
            enabled: if data.ttt_enabled { Some(true) } else { None },
            gradient_steps: data.ttt_gradient_steps,
            learning_rate: data.ttt_learning_rate,
            adaptation_strategy: map_adaptation_strategy(data.adaptation_strategy, data.writeback_threshold),
            max_adaptation_ms: None,
        };

        // Calculate expiry from claims
        let expiry_secs = ctx.claims()
            .map(|c| c.exp - chrono::Utc::now().timestamp())
            .unwrap_or(300)
            .max(60);

        let client_ephemeral_pubkey = ctx.ephemeral_pubkey();
        let claims = ctx.claims().cloned();
        let (stream_id, server_pubkey, pending) = self.prepare_stream(request, client_ephemeral_pubkey, claims, expiry_secs, &subject, ttt_overrides).await?;

        let stream_sub_endpoint = hyprstream_rpc::registry::global()
            .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
            .to_zmq_string();

        let stream_info = crate::services::generated::inference_client::StreamInfo {
            stream_id,
            endpoint: stream_sub_endpoint,
            server_pubkey,
        };

        // Build continuation that executes the stream after REP is sent.
        // With tokio::sync::RwLock, the read guard is Send-safe, so no
        // UnsafeSendFuture wrapper is needed.
        let service = self.clone();
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_stream(pending).await;
        });

        Ok((stream_info, continuation))
    }

    async fn handle_model_info(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let mut info = InferenceService::handle_model_info(self).await;
        info.lora_loaded = self.base_delta.lock().is_some();
        Ok(InferenceResponseVariant::ModelInfoResult(info))
    }

    async fn handle_is_ready(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let ready = InferenceService::handle_is_ready(self).await;
        Ok(InferenceResponseVariant::IsReadyResult(ready))
    }

    async fn handle_get_layer_profile(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        use crate::runtime::ttn_profile;
        use crate::services::generated::inference_client::LayerProfileResult;

        let model_config = crate::runtime::model_config::ModelConfig::load(
            &self.model_path,
            &std::collections::HashMap::new(),
        )?;
        let profile = ttn_profile::get_layer_profile(&self.model_path, &model_config, None)?;
        let json = serde_json::to_string_pretty(&profile)?;

        Ok(InferenceResponseVariant::GetLayerProfileResult(LayerProfileResult { json }))
    }

    async fn handle_apply_chat_template(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &ChatTemplateRequest,
    ) -> Result<InferenceResponseVariant> {
        let chat_messages: Vec<crate::runtime::template_engine::ChatMessage> = data.messages
            .iter()
            .map(|m| {
                let tool_calls = if m.tool_calls.is_empty() {
                    None
                } else {
                    Some(m.tool_calls.clone())
                };
                crate::runtime::template_engine::ChatMessage {
                    role: m.role.clone(),
                    content: if m.content.is_empty() { None } else { Some(m.content.clone()) },
                    tool_calls,
                    tool_call_id: if m.tool_call_id.is_empty() { None } else { Some(m.tool_call_id.clone()) },
                }
            })
            .collect();
        // tools_json is passed as a JSON string via the schema; parse it here
        let tools: Option<serde_json::Value> = data.tools_json.as_deref()
            .filter(|s| !s.is_empty())
            .and_then(|s| serde_json::from_str(s).ok());
        let result = InferenceService::handle_apply_chat_template(
            self, chat_messages, data.add_generation_prompt, tools.as_ref(),
        ).await?;
        Ok(InferenceResponseVariant::ApplyChatTemplateResult(result))
    }

    async fn handle_create_lora(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &LoraConfig,
    ) -> Result<InferenceResponseVariant> {
        let config = crate::training::TenantDeltaConfig {
            rank: data.rank as usize,
            alpha: data.alpha,
            dropout: data.dropout,
            target_modules: data.target_modules.clone(),
            learning_rate: data.learning_rate as f64,
            ..crate::training::TenantDeltaConfig::default()
        };
        InferenceService::handle_create_lora(self, config).await?;
        Ok(InferenceResponseVariant::CreateLoraResult)
    }

    async fn handle_load_lora(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> Result<InferenceResponseVariant> {
        self.load_lora(Path::new(value)).await?;
        Ok(InferenceResponseVariant::LoadLoraResult)
    }

    async fn handle_save_lora(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> Result<InferenceResponseVariant> {
        self.save_lora(value).await?;
        Ok(InferenceResponseVariant::SaveLoraResult)
    }

    async fn handle_unload_lora(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        self.unload_lora().await?;
        Ok(InferenceResponseVariant::UnloadLoraResult)
    }

    async fn handle_has_lora(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let has = self.has_lora().await?;
        Ok(InferenceResponseVariant::HasLoraResult(has))
    }

    async fn handle_set_session(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> Result<InferenceResponseVariant> {
        InferenceService::handle_set_session(self, value.to_owned()).await?;
        Ok(InferenceResponseVariant::SetSessionResult)
    }

    async fn handle_clear_session(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        InferenceService::handle_clear_session(self).await;
        Ok(InferenceResponseVariant::ClearSessionResult)
    }

    async fn handle_release_session(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> Result<InferenceResponseVariant> {
        InferenceService::handle_release_session(self, value).await?;
        Ok(InferenceResponseVariant::ReleaseSessionResult)
    }

    async fn handle_health_check(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let model_loaded = self.engine.read().is_loaded();
        Ok(InferenceResponseVariant::HealthCheckResult(HealthStatus {
            status: if model_loaded { "ok".into() } else { "not_loaded".into() },
            model_loaded,
            kv_cache_usage_percent: 0.0,
            gpu_memory_used_mb: 0,
            gpu_memory_total_mb: 0,
        }))
    }

    async fn handle_shutdown(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        info!("Inference service shutdown requested");
        Ok(InferenceResponseVariant::Success)
    }

    async fn handle_ttt_writeback(&self, ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        InferenceService::handle_writeback(self, &subject)?;
        Ok(InferenceResponseVariant::TttWritebackResult)
    }

    async fn handle_ttt_evict(&self, ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        InferenceService::handle_evict(self, &subject)?;
        Ok(InferenceResponseVariant::TttEvictResult)
    }

    async fn handle_train_step(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        _data: &TrainStepRequest,
    ) -> Result<InferenceResponseVariant> {
        // DEPRECATED: trainStep @17 blocks the REQ/REP loop during forward/backward pass.
        // Use trainStepStream instead, which returns StreamInfo immediately and runs
        // the training in a continuation after the REP is sent.
        Err(anyhow!("trainStep is deprecated — use trainStepStream for non-blocking training"))
    }

    async fn handle_train_step_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &TrainStepRequest,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let subject = ctx.subject();

        // DH key derivation
        let client_pub_bytes = ctx.ephemeral_pubkey()
            .ok_or_else(|| anyhow!("Streaming requires client ephemeral pubkey for E2E authentication"))?;
        let stream_channel = self.stream_channel.as_ref()
            .ok_or_else(|| anyhow!("StreamChannel not initialized"))?;

        let expiry_secs = ctx.claims()
            .map(|c| c.exp - chrono::Utc::now().timestamp())
            .unwrap_or(300)
            .max(60);
        let claims = ctx.claims().cloned();

        let stream_ctx = stream_channel.prepare_stream_with_claims(client_pub_bytes, expiry_secs, claims).await?;

        let stream_id = stream_ctx.stream_id().to_owned();
        let server_pubkey = *stream_ctx.server_pubkey();
        let stream_sub_endpoint = hyprstream_rpc::registry::global()
            .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
            .to_zmq_string();

        let stream_info = crate::services::generated::inference_client::StreamInfo {
            stream_id,
            endpoint: stream_sub_endpoint,
            server_pubkey,
        };

        let adaptation_strategy = map_adaptation_strategy(data.adaptation_strategy, data.writeback_threshold);
        let pending = PendingWork::Training {
            stream_ctx,
            subject,
            input: data.input.clone(),
            gradient_steps: data.gradient_steps.unwrap_or(0),
            learning_rate: data.learning_rate.unwrap_or(0.0),
            adaptation_strategy,
        };

        // Build continuation
        let service = self.clone();
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_training_stream(pending).await;
        });

        Ok((stream_info, continuation))
    }

    async fn handle_ttt_zero(&self, ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        InferenceService::handle_reset_delta(self, &subject)?;
        Ok(InferenceResponseVariant::TttZeroResult)
    }

    async fn handle_get_delta_status(&self, ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        let info = InferenceService::handle_get_delta_status(self, &subject)?;
        Ok(InferenceResponseVariant::GetDeltaStatusResult(DeltaStatusResult {
            exists: info.exists,
            accumulated_steps: info.accumulated_steps,
            max_accumulated_steps: info.max_accumulated_steps,
            request_count: info.request_count,
            avg_loss_improvement: info.avg_loss_improvement,
            memory_bytes: info.memory_bytes,
            last_snapshot_hash: info.last_snapshot_hash,
            delta_norm_ratios: info.delta_norm_ratios.into_iter().map(|(name, ratio)| ModuleNormRatio {
                module_name: name,
                ratio: ratio as f32,
            }).collect(),
            has_pending: info.has_pending,
        }))
    }

    async fn handle_save_adaptation(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &SaveAdaptationRequest,
    ) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        let info = InferenceService::handle_save_adaptation(self, &subject, &data.name, data.merge_strategy.as_deref().unwrap_or(""), data.merge_weight.unwrap_or(0.0)).await?;
        Ok(InferenceResponseVariant::SaveAdaptationResult(SaveAdaptationResult {
            adapter_name: info.adapter_name,
            adapter_path: info.adapter_path,
            content_hash: info.content_hash,
            merge_strategy: info.merge_strategy,
        }))
    }

    async fn handle_snapshot_delta(&self, ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        let info = InferenceService::handle_snapshot_delta(self, &subject).await?;
        Ok(InferenceResponseVariant::SnapshotDeltaResult(SnapshotDeltaResult {
            content_hash: info.content_hash,
            size_bytes: info.size_bytes,
            accumulated_steps: info.accumulated_steps,
            request_count: info.request_count,
        }))
    }

    async fn handle_export_peft_adapter(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &ExportPeftRequest,
    ) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        let info = InferenceService::handle_export_peft_adapter(self, &subject, &data.name, data.commit_message.as_deref().unwrap_or("")).await?;
        Ok(InferenceResponseVariant::ExportPeftAdapterResult(ExportPeftResult {
            adapter_path: info.adapter_path,
            content_hash: info.content_hash,
        }))
    }

    async fn handle_merge_lora(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &MergeLoraRequest,
    ) -> Result<InferenceResponseVariant> {
        let weight = if data.weight > 0.0 { data.weight } else { 1.0 };
        let strategy = if data.strategy.is_empty() { "do_merge" } else { &data.strategy };
        self.merge_lora(Path::new(&data.adapter_path), weight, strategy).await?;
        Ok(InferenceResponseVariant::MergeLoraResult)
    }

    // =========================================================================
    // Streaming handler variants
    //
    // Each returns (StreamInfo, Continuation) immediately. The continuation
    // calls an execute_*_stream method on the cloned service, which accesses
    // self.stream_channel by reference (same pattern as execute_training_stream).
    // =========================================================================

    async fn handle_create_lora_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &LoraConfig,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let (stream_info, stream_ctx) = self.setup_stream(ctx).await?;
        let service = self.clone();
        let config = crate::training::TenantDeltaConfig {
            rank: data.rank as usize,
            alpha: data.alpha,
            dropout: data.dropout,
            target_modules: data.target_modules.clone(),
            learning_rate: data.learning_rate as f64,
            ..crate::training::TenantDeltaConfig::default()
        };
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_create_lora_stream(stream_ctx, config).await;
        });
        Ok((stream_info, continuation))
    }

    async fn handle_load_lora_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        value: &str,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let (stream_info, stream_ctx) = self.setup_stream(ctx).await?;
        let service = self.clone();
        let path = value.to_owned();
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_load_lora_stream(stream_ctx, path).await;
        });
        Ok((stream_info, continuation))
    }

    async fn handle_save_lora_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        value: &str,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let (stream_info, stream_ctx) = self.setup_stream(ctx).await?;
        let service = self.clone();
        let name = value.to_owned();
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_save_lora_stream(stream_ctx, name).await;
        });
        Ok((stream_info, continuation))
    }

    async fn handle_save_adaptation_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &SaveAdaptationRequest,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let (stream_info, stream_ctx) = self.setup_stream(ctx).await?;
        let service = self.clone();
        let subject = ctx.subject();
        let name = data.name.clone();
        let merge_strategy = data.merge_strategy.clone();
        let merge_weight = data.merge_weight;
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_save_adaptation_stream(stream_ctx, subject, name, merge_strategy.unwrap_or_default(), merge_weight.unwrap_or(0.0)).await;
        });
        Ok((stream_info, continuation))
    }

    async fn handle_snapshot_delta_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let (stream_info, stream_ctx) = self.setup_stream(ctx).await?;
        let service = self.clone();
        let subject = ctx.subject();
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_snapshot_delta_stream(stream_ctx, subject).await;
        });
        Ok((stream_info, continuation))
    }

    async fn handle_export_peft_adapter_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &ExportPeftRequest,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let (stream_info, stream_ctx) = self.setup_stream(ctx).await?;
        let service = self.clone();
        let subject = ctx.subject();
        let name = data.name.clone();
        let commit_message = data.commit_message.clone();
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_export_peft_adapter_stream(stream_ctx, subject, name, commit_message.unwrap_or_default()).await;
        });
        Ok((stream_info, continuation))
    }

    async fn handle_merge_lora_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &MergeLoraRequest,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let (stream_info, stream_ctx) = self.setup_stream(ctx).await?;
        let service = self.clone();
        let adapter_path = data.adapter_path.clone();
        let weight = if data.weight > 0.0 { data.weight } else { 1.0 };
        let strategy = if data.strategy.is_empty() { "do_merge".to_owned() } else { data.strategy.clone() };
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_merge_lora_stream(stream_ctx, adapter_path, weight, strategy).await;
        });
        Ok((stream_info, continuation))
    }

    async fn handle_embed(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &EmbedImagesRequest,
    ) -> Result<InferenceResponseVariant> {
        let engine = self.engine.read();
        let embeddings = engine.embed_images(&data.images)?;
        let dimensions = embeddings.first().map(|v| v.len() as u32).unwrap_or(0);
        Ok(InferenceResponseVariant::EmbedResult(EmbedImagesResponse {
            embeddings,
            dimensions,
        }))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZmqService adapter and Spawnable implementation
// ═══════════════════════════════════════════════════════════════════════════════

// verify_claims is now handled by the default ZmqService::verify_claims() implementation
// in hyprstream-rpc (E2E JWT verification for all non-local identities).

/// ZmqService adapter for InferenceService.
///
/// Wraps `InferenceService` to implement `ZmqService` for use with `RequestLoop`.
/// This adapter is created inside `Spawnable::run()` on the service thread —
/// it never crosses thread boundaries.
struct InferenceZmqAdapter {
    service: InferenceService,
    zmq_context: Arc<zmq::Context>,
    transport: hyprstream_rpc::transport::TransportConfig,
    signing_key: SigningKey,
    expected_audience: Option<String>,
    local_issuer_url: Option<String>,
    federation_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>>,
}

#[async_trait::async_trait(?Send)]
impl hyprstream_rpc::service::ZmqService for InferenceZmqAdapter {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> anyhow::Result<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)> {
        dispatch_inference(&self.service, ctx, payload).await
    }

    fn name(&self) -> &str {
        "inference"
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.zmq_context
    }

    fn transport(&self) -> &hyprstream_rpc::transport::TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.expected_audience.as_deref()
    }

    fn local_issuer_url(&self) -> Option<&str> {
        self.local_issuer_url.as_deref()
    }

    fn federation_key_source(
        &self,
    ) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>> {
        self.federation_key_source.clone()
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = InferenceResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_else(|e| {
            error!("Failed to serialize error payload: {}", e);
            vec![]
        })
    }
}

/// Configuration for spawning an InferenceService.
///
/// This struct holds all parameters needed to initialize an InferenceService.
/// It is `Send` and can be moved to a dedicated thread via `Spawnable::run()`.
/// The actual GPU initialization happens on the service thread.
pub struct InferenceServiceConfig {
    model_path: PathBuf,
    config: RuntimeConfig,
    server_pubkey: VerifyingKey,
    signing_key: SigningKey,
    /// PolicyClient is created lazily inside `run()` on the service thread's runtime.
    /// ZMQ sockets must be registered with the correct runtime's reactor — a PolicyClient
    /// created in the caller's runtime will fail with "Tokio context being shutdown" when
    /// used on a different thread's runtime.
    policy_signing_key: SigningKey,
    zmq_context: Arc<zmq::Context>,
    transport: hyprstream_rpc::transport::TransportConfig,
    fs: Option<WorktreeClient>,
    /// Expected audience for JWT validation (resource URL)
    expected_audience: Option<String>,
    /// Local OAuth issuer URL for distinguishing local vs. federated JWTs.
    local_issuer_url: Option<String>,
    /// Federation key source for verifying externally-issued JWTs.
    federation_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>>,
}

impl InferenceServiceConfig {
    /// Create a new InferenceServiceConfig.
    ///
    /// No GPU work is done here — all heavy initialization is deferred
    /// to the service thread via `Spawnable::run()`.
    ///
    /// **Important**: Does NOT take a `PolicyClient`. The policy client is created
    /// lazily inside `run()` on the service thread's own Tokio runtime, because
    /// ZMQ sockets must be registered with the correct runtime's reactor.
    pub fn new(
        model_path: impl AsRef<Path>,
        config: RuntimeConfig,
        server_pubkey: VerifyingKey,
        signing_key: SigningKey,
        zmq_context: Arc<zmq::Context>,
        transport: hyprstream_rpc::transport::TransportConfig,
        fs: Option<WorktreeClient>,
    ) -> Self {
        let policy_signing_key = signing_key.clone();
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            config,
            server_pubkey,
            signing_key,
            policy_signing_key,
            zmq_context,
            transport,
            fs,
            expected_audience: None,
            local_issuer_url: None,
            federation_key_source: None,
        }
    }

    /// Set the expected audience for JWT validation.
    pub fn with_expected_audience(mut self, audience: String) -> Self {
        self.expected_audience = Some(audience);
        self
    }

    /// Set the local OAuth issuer URL for distinguishing local vs. federated JWTs.
    pub fn with_local_issuer_url(mut self, url: String) -> Self {
        self.local_issuer_url = Some(url);
        self
    }

    /// Set the federation key source for verifying externally-issued JWTs.
    pub fn with_federation_key_source(
        mut self,
        src: std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>,
    ) -> Self {
        self.federation_key_source = Some(src);
        self
    }
}

impl hyprstream_service::Spawnable for InferenceServiceConfig {
    fn name(&self) -> &str {
        "inference"
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.zmq_context
    }

    fn registrations(&self) -> Vec<(hyprstream_rpc::registry::SocketKind, hyprstream_rpc::transport::TransportConfig)> {
        vec![(hyprstream_rpc::registry::SocketKind::Rep, self.transport.clone())]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<tokio::sync::Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> hyprstream_rpc::error::Result<()> {
        let transport = self.transport.clone();
        let context = Arc::clone(&self.zmq_context);
        let signing_key = self.signing_key.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("runtime: {e}")))?;

        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async move {
            // Create nonce cache (shared between service and RequestLoop)
            let nonce_cache = Arc::new(InMemoryNonceCache::new());

            // Create PolicyClient HERE, inside the service thread's runtime,
            // so ZMQ sockets are registered with the correct reactor.
            let policy_client = PolicyClient::new(
                self.policy_signing_key,
                hyprstream_rpc::envelope::RequestIdentity::local(),
            );

            // GPU initialization happens HERE, on the service thread
            let service = InferenceService::initialize(
                self.model_path,
                self.config,
                self.server_pubkey,
                self.signing_key.clone(),
                Arc::clone(&nonce_cache),
                policy_client,
                self.fs,
            )
            .await
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("init: {e}")))?;

            // Create ZmqService adapter for RequestLoop
            let adapter = InferenceZmqAdapter {
                service,
                zmq_context: context.clone(),
                transport: transport.clone(),
                signing_key: signing_key.clone(),
                expected_audience: self.expected_audience,
                local_issuer_url: self.local_issuer_url,
                federation_key_source: self.federation_key_source,
            };

            // Use shared nonce cache between service and RequestLoop
            let runner = RequestLoop::new(transport, context, signing_key)
                .with_nonce_cache(nonce_cache);
            let mut handle = runner.run(adapter).await
                .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("loop: {e}")))?;

            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }

            // Notify systemd that service is ready (for Type=notify services)
            let _ = hyprstream_rpc::notify::ready();

            shutdown.notified().await;
            handle.stop().await;
            Ok::<_, hyprstream_rpc::error::RpcError>(())
        })
    }
}

/// Client for the inference service
///
/// # Security
///
/// All requests are wrapped in `SignedEnvelope` for authentication:
/// - Requests are signed with the client's Ed25519 signing key
/// - The service verifies signatures before processing
/// - Identity is included for authorization checks
///
/// Wraps a generated `InferenceClient`. Pure delegation methods are
/// available via `Deref`; this struct adds domain-specific convenience
/// methods with type translation.
#[derive(Clone)]
pub struct InferenceZmqClient {
    /// Generated typed client (handles all transport including streaming via call_with_options)
    pub(crate) gen: crate::services::generated::inference_client::InferenceClient,
}

impl std::ops::Deref for InferenceZmqClient {
    type Target = crate::services::generated::inference_client::InferenceClient;
    fn deref(&self) -> &Self::Target { &self.gen }
}

impl InferenceZmqClient {
    /// Create a new inference client with signing credentials
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self::with_endpoint(INFERENCE_ENDPOINT, signing_key, identity)
    }

    /// Create an inference client connected to a specific endpoint
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            gen: crate::services::core::create_service_client(endpoint, signing_key, identity),
        }
    }

    /// Attach claims for e2e verification. All subsequent calls include these claims.
    pub fn with_claims(self, claims: hyprstream_rpc::auth::Claims) -> Self {
        Self { gen: self.gen.with_claims(claims) }
    }

    /// Health check (discards status detail, returns Ok/Err).
    ///
    /// Named `health_check_ok` to avoid shadowing the generated `health_check()`
    /// method on the Deref target (which returns the full status).
    pub async fn health_check_ok(&self) -> Result<()> {
        let _status = self.gen.health_check().await?;
        Ok(())
    }

    /// Start streaming generation with E2E authentication.
    /// Translates domain `GenerationRequest` to the generated `InfGenerationRequest`.
    pub async fn generate_stream(
        &self,
        request: &GenerationRequest,
        ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamInfo> {
        self.gen.generate_stream(&InfGenerationRequest {
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            repeat_penalty: request.repeat_penalty,
            repeat_last_n: request.repeat_last_n,
            stop_tokens: request.stop_tokens.clone(),
            seed: request.seed,
            images: request.images.clone(),
            timeout_ms: request.timeout_ms,
            ttt_enabled: request.ttt_enabled,
            ttt_gradient_steps: request.ttt_gradient_steps,
            ttt_learning_rate: request.ttt_learning_rate,
            adaptation_strategy: request.adaptation_strategy,
            writeback_threshold: request.writeback_threshold,
        }, ephemeral_pubkey.unwrap_or([0u8; 32])).await.map(Into::into)
    }

    /// Start streaming generation with E2E authenticated handle.
    ///
    /// Returns the canonical `StreamHandle` from hyprstream-rpc which provides:
    /// - `recv_next().await` for async receive with HMAC verification
    /// - `futures::Stream` impl for composability with `StreamExt::next().await`
    ///
    /// Use `StreamChunkMessage::from_stream_payload()` to convert received
    /// `StreamPayload` values to inference-specific message types.
    pub async fn generate_stream_handle(
        &self,
        request: &GenerationRequest,
    ) -> Result<crate::services::rpc_types::StreamHandle> {
        let this = self.clone();
        let req = request.clone();
        crate::services::rpc_types::connect_stream_handle(|pubkey| async move {
            this.generate_stream(&req, Some(pubkey)).await
        }).await
    }

    /// Create a new LoRA adapter. Translates `TenantDeltaConfig` to `LoraConfig`.
    pub async fn create_lora(&self, config: &TenantDeltaConfig) -> Result<()> {
        self.gen.create_lora(&LoraConfig {
            rank: config.rank as u32,
            alpha: config.alpha,
            dropout: config.dropout,
            target_modules: config.target_modules.clone(),
            learning_rate: config.learning_rate as f32,
        }).await
    }

    /// Merge an on-disk adapter into the loaded base_delta.
    pub async fn merge_lora(&self, adapter_path: &str, weight: f32, strategy: &str) -> Result<()> {
        self.gen.merge_lora(&MergeLoraRequest {
            adapter_path: adapter_path.to_owned(),
            weight,
            strategy: strategy.to_owned(),
        }).await
    }

    // Training loop control (TTT operations)

    /// Write back a pending TTT adaptation
    pub async fn writeback_adaptation(&self) -> Result<()> {
        self.gen.ttt_writeback().await
    }

    /// Evict a pending TTT adaptation
    pub async fn evict_adaptation(&self) -> Result<()> {
        self.gen.ttt_evict().await
    }

    /// Zero a tenant's delta accumulator
    pub async fn zero_delta(&self) -> Result<()> {
        self.gen.ttt_zero().await
    }

    /// Start streaming training step with E2E authentication — delegates to generated client.
    ///
    /// `adaptation_strategy` must be one of: "auto_writeback", "auto_evict",
    /// "speculative", "writeback_if_above".
    pub async fn train_step_stream(
        &self,
        input: &str,
        gradient_steps: u32,
        learning_rate: f32,
        adaptation_strategy: &str,
        writeback_threshold: f32,
        ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamInfo> {
        self.gen.train_step_stream(&TrainStepRequest {
            input: input.to_owned(),
            gradient_steps: Some(gradient_steps).filter(|&v| v != 0),
            learning_rate: Some(learning_rate).filter(|&v| v != 0.0),
            adaptation_strategy: adaptation_strategy.to_owned(),
            writeback_threshold,
        }, ephemeral_pubkey.unwrap_or([0u8; 32])).await.map(Into::into)
    }

    /// Compute vision embeddings for images.
    pub async fn embed(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
        let response = self.gen.embed(&EmbedImagesRequest {
            images: images.to_vec(),
        }).await?;
        Ok(response.embeddings)
    }

}

// ============================================================================
// StreamChunkMessage - Client-side stream message handling
// ============================================================================

/// Message received from StreamService during streaming generation.
///
/// Represents one of three possible message types:
/// - Chunk: Text chunk from the model
/// - Complete: Stream finished with generation stats
/// - Error: Generation failed
///
/// Note: Ordering is handled by prevMac in the outer streaming_capnp::StreamBlock wrapper.
/// Authentication via HMAC chain happens at the StreamBlock level.
pub enum StreamChunkMessage {
    Chunk {
        text: String,
    },
    Complete {
        stats: crate::runtime::GenerationStats,
    },
    Error {
        error: String,
    },
}

impl StreamChunkMessage {
    /// Convert a generic StreamPayload to an inference-specific StreamChunkMessage.
    ///
    /// This bridges the generic streaming layer (StreamPayload) to the inference-specific
    /// message types used by consumers.
    pub fn from_stream_payload(payload: crate::services::rpc_types::StreamPayload) -> Self {
        use crate::services::rpc_types::{InferenceStreamPayload, StreamPayloadExt};

        match payload.to_inference() {
            Ok(InferenceStreamPayload::Token(text)) => {
                StreamChunkMessage::Chunk { text }
            }
            Ok(InferenceStreamPayload::Error(message)) => {
                StreamChunkMessage::Error { error: message }
            }
            Ok(InferenceStreamPayload::Complete(stats)) => {
                let quality_metrics = if stats.perplexity.is_some() || stats.avg_entropy.is_some() {
                    Some(crate::runtime::generation_metrics::GenerationQualityMetrics {
                        perplexity: stats.perplexity.unwrap_or(0.0),
                        avg_entropy: stats.avg_entropy.unwrap_or(0.0),
                        ..Default::default()
                    })
                } else {
                    None
                };

                let gen_stats = crate::runtime::GenerationStats {
                    tokens_generated: stats.tokens_generated,
                    generation_time_ms: stats.generation_time_ms,
                    tokens_per_second: stats.tokens_per_second,
                    finish_reason: Some(match stats.finish_reason.as_str() {
                        "length" => crate::config::FinishReason::MaxTokens,
                        "eos" => crate::config::FinishReason::EndOfSequence,
                        "error" => crate::config::FinishReason::Error(String::new()),
                        _ => crate::config::FinishReason::Stop,
                    }),
                    quality_metrics,
                    prefill_tokens: stats.prefill_tokens,
                    prefill_time_ms: stats.prefill_time_ms,
                    prefill_tokens_per_sec: stats.prefill_tokens_per_sec,
                    inference_tokens: stats.inference_tokens,
                    inference_time_ms: stats.inference_time_ms,
                    inference_tokens_per_sec: stats.inference_tokens_per_sec,
                    inference_tokens_per_sec_ema: stats.inference_tokens_per_sec_ema,
                };
                StreamChunkMessage::Complete { stats: gen_stats }
            }
            Err(e) => {
                StreamChunkMessage::Error { error: format!("Failed to parse payload: {e}") }
            }
        }
    }

    /// Returns true if this is the last message (Complete or Error)
    pub fn is_last(&self) -> bool {
        matches!(self, StreamChunkMessage::Complete { .. } | StreamChunkMessage::Error { .. })
    }
}

// StreamHandle consolidated: uses hyprstream_rpc::streaming::StreamHandle (re-exported via rpc_types)
// Use StreamChunkMessage::from_stream_payload() to convert StreamPayload → StreamChunkMessage
