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
use crate::config::{FinishReason, GenerationRequest, GenerationResult, ModelInfo, TrainingMode};
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
use std::time::Instant;
use parking_lot::{Mutex, RwLock};
use tmq::{reply, Multipart};
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
        /// TTT adaptation metrics (if TTT was run in prepare_stream)
        ttt_result: Option<crate::training::ttt::TTTResult>,
        /// Per-tenant delta for delta-aware inference (looked up in prepare_stream)
        delta: Option<Arc<Mutex<crate::training::TenantDelta>>>,
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
        /// Whether to auto-commit if quality gate passes
        auto_commit: bool,
    },
}

/// Wraps a `!Send` future to satisfy the `Send` bound on [`Continuation`].
///
/// # Safety
///
/// This is safe because InferenceService's continuations only run on its
/// dedicated single-threaded tokio runtime (`run_service_loop`). They are
/// never sent between threads. The `Send` bound comes from the [`Continuation`]
/// type alias (`Pin<Box<dyn Future + Send>>`), which exists for services that
/// use `RequestLoop` (which requires `Send`). InferenceService uses a custom loop.
///
/// The specific `!Send` type is `parking_lot::RwLockReadGuard` held across an
/// `.await` in `execute_stream`, protecting the `TorchEngine` during streaming
/// token generation.
struct UnsafeSendFuture(std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>);
unsafe impl Send for UnsafeSendFuture {}
impl std::future::Future for UnsafeSendFuture {
    type Output = ();
    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<()> {
        self.0.as_mut().poll(cx)
    }
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
    engine: RwLock<TorchEngine>,
    /// Model path for checkpoint management
    #[allow(dead_code)] // Future: checkpoint management
    model_path: PathBuf,
    /// Current session ID for events
    session_id: RwLock<Option<String>>,
    /// Runtime handle for async operations (reused instead of creating new runtimes)
    #[allow(dead_code)] // Reserved for future async operations
    runtime_handle: Handle,
    /// Stream channel for streaming generation (connects to StreamService)
    /// Handles DH key exchange, pre-authorization, and publishing.
    stream_channel: Option<StreamChannel>,
    /// Server's Ed25519 verifying key for signature verification
    server_pubkey: VerifyingKey,
    /// Service signing key for stream registration (generated at init)
    signing_key: SigningKey,
    /// Nonce cache for replay protection
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
    /// Pending adaptations awaiting client commit/rollback
    pending_adaptations: Mutex<std::collections::HashMap<Subject, PendingAdaptation>>,
    /// Optional WorktreeClient for worktree-scoped file operations.
    /// When present, adapter/snapshot writes use contained-root access.
    fs: Option<WorktreeClient>,
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

/// A pending TTT adaptation awaiting client commit or rollback
struct PendingAdaptation {
    /// Delta state before adaptation (for rollback)
    pre_adaptation_state: std::collections::HashMap<String, tch::Tensor>,
    /// The TTT result from the adaptation
    ttt_result: crate::training::ttt::TTTResult,
    /// When the adaptation was created
    created_at: Instant,
    /// Auto-rollback after this timeout (default: 30s)
    timeout_ms: u64,
}

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
    /// Start the inference service at a specific endpoint
    pub async fn start_at(
        model_path: impl AsRef<Path>,
        config: RuntimeConfig,
        server_pubkey: VerifyingKey,
        signing_key: SigningKey,
        policy_client: PolicyClient,
        endpoint: &str,
        fs: Option<WorktreeClient>,
    ) -> Result<hyprstream_rpc::service::SpawnedService> {
        let model_path = model_path.as_ref().to_path_buf();
        let endpoint_owned = endpoint.to_owned();
        let nonce_cache = Arc::new(InMemoryNonceCache::new());

        // Use oneshot to get initialization result
        let (init_tx, init_rx) = tokio::sync::oneshot::channel();

        // Spawn service on dedicated thread
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build() {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = init_tx.send(Err(anyhow!("Failed to create service runtime: {}", e)));
                        return;
                    }
                };

            rt.block_on(async move {
                match Self::initialize(model_path, config, server_pubkey, signing_key, nonce_cache, policy_client, fs).await {
                    Ok(service) => {
                        // Pass init_tx to run_service_loop - it signals AFTER socket binding
                        Self::run_service_loop(service, &endpoint_owned, Some(init_tx)).await;
                    }
                    Err(e) => {
                        if init_tx.send(Err(e)).is_err() {
                            tracing::warn!("Failed to send initialization error - receiver dropped");
                        }
                    }
                }
            });
        });

        // Wait for initialization
        init_rx
            .await
            .map_err(|_| anyhow!("Service init channel closed"))??;

        info!("Inference service started at {}", endpoint);

        // Return a dummy handle (the service manages its own lifecycle)
        Ok(hyprstream_rpc::service::SpawnedService::dummy())
    }

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

        rt.block_on(async move {
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
                    let response = service.handle_callback_command(&data).await?;
                    dealer.send(&response, 0)?;
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
    async fn handle_callback_command(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(data),
            ReaderOptions::new(),
        )?;
        let cmd = reader.get_root::<crate::model_capnp::inference_command::Reader>()?;

        use crate::model_capnp::inference_command::Which;
        match cmd.which()? {
            Which::LoadModel(_) => {
                // Already loaded, return success
                Self::build_load_model_response(true, "")
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

    /// Handle inference request from callback channel
    async fn handle_callback_infer(&mut self, request_data: &[u8]) -> Result<Vec<u8>> {
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

        // Dispatch via generated handler
        let (response, _continuation) = dispatch_inference(self, &ctx, request_data).await?;
        Ok(response)
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
        let num_layers = model_info.num_hidden_layers.unwrap_or(32);
        let max_seq_len = config.max_context.unwrap_or(model_info.context_length);
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
                    let trainer = TestTimeTrainer::new(ttt_config, device);

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
            let device = engine.device();

            let delta_config = if let Some(ref tc) = training_config {
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
            let kv_reg = engine.kv_registry();
            let snapshots_dir = model_path.join("adapters").join(".snapshots");
            let num_layers = engine.get_num_layers().unwrap_or(32);

            info!(
                "Delta pool initialized: rank={}, alpha={:.1}, modules={:?}, lr={:.1e}",
                delta_config.rank, delta_config.alpha, delta_config.target_modules, delta_config.learning_rate
            );
            let pool = DeltaPool::new(delta_config, module_dims, device, kv_reg, snapshots_dir, fs.clone(), num_layers);

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
                engine: RwLock::new(engine),
                model_path,
                session_id: RwLock::new(None),
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
                pending_adaptations: Mutex::new(std::collections::HashMap::new()),
                fs,
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

    /// Run the service loop (async with TMQ)
    ///
    /// The `ready_tx` channel signals when sockets are bound and the service is ready.
    /// This ensures callers wait for actual readiness, not just initialization.
    async fn run_service_loop(
        service: Self,
        endpoint: &str,
        ready_tx: Option<tokio::sync::oneshot::Sender<Result<()>>>,
    ) {
        let ctx = global_context();

        // Helper to signal error
        let signal_error = |tx: Option<tokio::sync::oneshot::Sender<Result<()>>>, err: anyhow::Error| {
            if let Some(tx) = tx {
                let _ = tx.send(Err(err));
            }
        };

        // Create REP socket with TMQ for async I/O
        let mut receiver = match reply(&ctx).set_linger(0).bind(endpoint) {
            Ok(r) => r,
            Err(e) => {
                let err = anyhow!("failed to bind REP to {}: {}", endpoint, e);
                error!("{}", err);
                signal_error(ready_tx, err);
                return;
            }
        };

        // StreamChannel already created in initialize()

        let stream_endpoint = hyprstream_rpc::registry::global()
            .endpoint("streams", hyprstream_rpc::registry::SocketKind::Push)
            .to_zmq_string();
        info!("inference service bound to {} (RPC), streaming via {}", endpoint, stream_endpoint);

        // Signal ready - ZMQ connection will establish asynchronously
        // With immediate=false, messages queue until connection is ready
        // execute_stream handles connection errors gracefully
        if let Some(tx) = ready_tx {
            if tx.send(Ok(())).is_err() {
                warn!("Failed to signal service ready - receiver dropped");
            }
        }

        // Main service loop (async with TMQ)
        loop {
            let result = receiver.recv().await;
            let (request_msg, sender) = match result {
                Ok((msg, sender)) => (msg, sender),
                Err(e) => {
                    // recv() consumes the receiver, so on error we must exit
                    // A recv error typically means socket/context problem
                    error!("inference recv error (fatal): {}", e);
                    return;
                }
            };

            // Extract bytes from multipart message
            let request: Vec<u8> = request_msg
                .into_iter()
                .flat_map(|frame| frame.to_vec())
                .collect();

            trace!("inference received request ({} bytes)", request.len());

            // Clean up expired pending adaptations on each request
            service.cleanup_expired_adaptations();

            // Unwrap and verify SignedEnvelope
            let (envelope_ctx, payload) = match hyprstream_rpc::unwrap_envelope(&request, &service.server_pubkey, &*service.nonce_cache) {
                Ok((ctx, payload)) => (ctx, payload),
                Err(e) => {
                    warn!("inference envelope verification failed: {}", e);
                    // Build proper error response (request_id=0 since envelope is invalid)
                    let err_variant = InferenceResponseVariant::Error(ErrorInfo {
                        message: format!("envelope verification failed: {}", e),
                        code: "UNAUTHORIZED".to_owned(),
                        details: String::new(),
                    });
                    let error_payload = serialize_response(0, &err_variant).unwrap_or_default();
                    let msg: Multipart = vec![error_payload].into();
                    receiver = match sender.send(msg).await {
                        Ok(r) => r,
                        Err(e) => {
                            error!("failed to send error response: {}", e);
                            return;
                        }
                    };
                    continue;
                }
            };

            debug!(
                "Inference request from {} (envelope_id={})",
                envelope_ctx.subject(),
                envelope_ctx.request_id
            );

            // Handle request - may return pending work (now async for policy checks)
            // Handle request via generated dispatch
            let request_id = envelope_ctx.request_id;
            let (response_payload, continuation) = match dispatch_inference(&service, &envelope_ctx, &payload).await {
                Ok((resp, cont)) => (resp, cont),
                Err(e) => {
                    error!("inference request handling error: {}", e);
                    let err_variant = InferenceResponseVariant::Error(ErrorInfo {
                        message: e.to_string(),
                        code: "INTERNAL".to_owned(),
                        details: String::new(),
                    });
                    (serialize_response(request_id, &err_variant).unwrap_or_default(), None)
                }
            };

            // Wrap response in signed envelope
            let response_bytes = {
                let signed_response = ResponseEnvelope::new_signed(
                    request_id,
                    response_payload,
                    &service.signing_key,
                );

                let mut message = capnp::message::Builder::new_default();
                let mut builder = message.init_root::<hyprstream_rpc::common_capnp::response_envelope::Builder>();
                signed_response.write_to(&mut builder);

                let mut bytes = Vec::new();
                if let Err(e) = capnp::serialize::write_message(&mut bytes, &message) {
                    error!("Failed to serialize signed response: {}", e);
                    vec![]
                } else {
                    bytes
                }
            };

            // Send signed response via TMQ
            let msg: Multipart = vec![response_bytes].into();
            receiver = match sender.send(msg).await {
                Ok(r) => r,
                Err(e) => {
                    error!("failed to send response: {}", e);
                    return;
                }
            };

            // Execute continuation AFTER REP is sent.
            // This guarantees the client has the StreamInfo (stream_id)
            // before any data flows on the PUB/SUB channel.
            if let Some(future) = continuation {
                future.await;
            }
        }
    }

    /// Apply TTT adaptation if enabled (adapts model to input BEFORE generation)
    ///
    /// Returns:
    /// - Ok(Some(result)) if TTT was configured and ran (or was skipped)
    /// - Ok(None) if TTT is not configured
    /// - Err(e) if TTT failed unexpectedly
    fn apply_ttt_adaptation(&self, prompt: &str, subject: &Subject) -> Result<Option<crate::training::ttt::TTTResult>> {
        self.apply_ttt_adaptation_with_overrides(prompt, subject, &crate::training::ttt::TTTOverrides::default())
    }

    /// Apply TTT adaptation with per-request overrides.
    ///
    /// Uses subject-specific delta pool for isolated per-session adaptation.
    fn apply_ttt_adaptation_with_overrides(
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

        let engine = self.engine.read();

        // Ensure delta exists for this subject
        let delta_arc = pool.get_or_create(subject)?;

        // Lock the delta and run adaptation
        let mut delta = delta_arc.lock();

        match ttt_trainer.adapt_tenant(&engine, &mut delta, &input_tokens, overrides) {
            Ok((result, pre_snapshot)) => {
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

                // Handle auto-commit vs pending
                if overrides.auto_commit && result.recommendation && !result.skipped {
                    debug!("TTT: auto-committed adaptation for subject {}", subject);
                } else if !overrides.auto_commit && !result.skipped && !pre_snapshot.is_empty() {
                    // Store pending adaptation for later commit/rollback
                    let pending = PendingAdaptation {
                        pre_adaptation_state: pre_snapshot,
                        ttt_result: result.clone(),
                        created_at: Instant::now(),
                        timeout_ms: 30_000,
                    };
                    self.pending_adaptations.lock().insert(subject.clone(), pending);
                }

                Ok(Some(result))
            }
            Err(e) => {
                warn!("TTT adaptation failed for subject {}: {}", subject, e);
                Err(e)
            }
        }
    }

    /// Handle non-streaming generation
    fn handle_generate(&self, request: GenerationRequest, subject: &Subject) -> Result<GenerationResult> {
        // Apply TTT adaptation BEFORE generation (if enabled) and capture metrics
        let ttt_result = match self.apply_ttt_adaptation(request.prompt.as_str(), subject) {
            Ok(Some(result)) => Some(result),
            Ok(None) => None,  // TTT not configured/applicable
            Err(e) => {
                // Log error but continue with generation
                warn!("TTT adaptation failed, continuing without: {}", e);
                None
            }
        };

        // Look up tenant's delta and/or base delta for delta-aware inference
        let delta = self.resolve_delta(subject);
        info!("[TTT-DEBUG] handle_generate: subject={}, delta_resolved={}, pool_exists={}, pool_subjects={:?}",
              subject, delta.is_some(), self.delta_pool.is_some(),
              self.delta_pool.as_ref().map(|p| p.list_subjects()));

        let engine = self.engine.read();
        let mut result = futures::executor::block_on(async {
            engine.generate_with_delta_params(request, delta).await
        })?;

        // Attach TTT metrics to response
        result.ttt_metrics = ttt_result.map(std::convert::Into::into);

        Ok(result)
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
    ) -> Result<(String, [u8; 32], PendingWork)> {
        // Apply TTT adaptation BEFORE streaming (capture metrics for completion)
        let ttt_result = match self.apply_ttt_adaptation(request.prompt.as_str(), subject) {
            Ok(Some(result)) => Some(result),
            Ok(None) => None,  // TTT not configured/applicable
            Err(e) => {
                // Log error but continue with streaming
                warn!("TTT adaptation failed, continuing without: {}", e);
                None
            }
        };

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

        // Look up tenant's delta and/or base delta for delta-aware inference
        let delta = self.resolve_delta(subject);

        let pending = PendingWork::Generation {
            stream_ctx,
            request,
            ttt_result,
            delta,
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
    /// This triggers clippy::await_holding_lock, but is necessary for the streaming API.
    #[allow(clippy::await_holding_lock)]
    async fn execute_stream(&self, pending: PendingWork) {
        use futures::StreamExt;

        let PendingWork::Generation { stream_ctx, request, ttt_result, delta } = pending else {
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

        trace!(
            stream_id = %stream_ctx.stream_id(),
            topic = %stream_ctx.topic(),
            has_delta = delta.is_some(),
            "Starting E2E authenticated stream via StreamChannel"
        );

        // Run the stream with StreamChannel's async publisher callback
        let engine = self.engine.read();
        let stream_result = engine.generate_with_delta(request, delta);

        let result = stream_channel.with_publisher(stream_ctx, |mut publisher| async move {
            match stream_result {
                Ok(mut stream) => {
                    let mut had_error = false;
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(text) => {
                                // Get live generation rate from stream stats (EMA for smooth batching)
                                let rate = stream.stats().inference_tokens_per_sec_ema;

                                // Publish with adaptive batching
                                if let Err(e) = publisher.publish_data_with_rate(text.as_bytes(), rate).await {
                                    warn!("Failed to publish stream data: {}", e);
                                    had_error = true;
                                    break;
                                }
                            }
                            Err(e) => {
                                // Publish error and stop
                                if let Err(send_err) = publisher.publish_error(&e.to_string()).await {
                                    error!("Failed to publish stream error: {}", send_err);
                                }
                                had_error = true;
                                break;
                            }
                        }
                    }

                    // Complete the stream if no errors occurred
                    if !had_error {
                        let stats = stream.stats();
                        let mut complete = crate::services::rpc_types::InferenceComplete::from(&stats);

                        // Attach TTT metrics to completion (captured in prepare_stream)
                        complete.ttt_metrics = ttt_result.map(std::convert::Into::into);

                        publisher.complete_ref(&complete.to_bytes()).await?;
                    }
                    Ok(())
                }
                Err(e) => {
                    // Initial error - publish and return
                    publisher.publish_error(&e.to_string()).await?;
                    Err(e)
                }
            }
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
        let PendingWork::Training { stream_ctx, subject, input, gradient_steps, learning_rate, auto_commit } = pending else {
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

        let result = stream_channel.with_publisher(stream_ctx, |mut publisher| async move {
            match self.handle_train_step(&subject, &input, gradient_steps, learning_rate, auto_commit) {
                Ok(result) => {
                    // Serialize training result as JSON for the completion payload
                    let payload = serde_json::to_vec(&result)
                        .unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}").into_bytes());
                    publisher.complete_ref(&payload).await?;
                    Ok(())
                }
                Err(e) => {
                    publisher.publish_error(&e.to_string()).await?;
                    Err(e)
                }
            }
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
    fn handle_model_info(&self) -> ModelInfo {
        RuntimeEngine::model_info(&*self.engine.read())
    }

    /// Handle is ready request
    fn handle_is_ready(&self) -> bool {
        self.engine.read().is_loaded()
    }

    /// Handle apply chat template
    fn handle_apply_chat_template(
        &self,
        messages: Vec<crate::runtime::template_engine::ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.engine
            .read()
            .apply_chat_template(&messages, add_generation_prompt)
    }

    /// Handle create LoRA
    fn handle_create_lora(&self, config: TenantDeltaConfig) -> Result<()> {
        // Propagate target modules to the delta pool so new deltas
        // create A/B matrices for ALL configured modules, not just the default q_proj/v_proj
        if let Some(pool) = &self.delta_pool {
            tracing::info!(
                "[TTT] Updating delta pool config: target_modules={:?}, rank={}, alpha={:.1}, lr={:.1e}",
                config.target_modules, config.rank, config.alpha, config.learning_rate
            );
            pool.update_config(config.clone());
        }
        self.engine.write().create_lora(config)
    }

    // =========================================================================
    // Training loop control handlers (tenant-aware TTT)
    // =========================================================================

    /// Commit a pending TTT adaptation
    fn handle_commit_adaptation(&self, subject: &Subject) -> Result<()> {
        let mut pending = self.pending_adaptations.lock();

        let adaptation = pending.remove(subject)
            .ok_or_else(|| anyhow!("No pending adaptation for subject '{}'", subject))?;

        // Get the subject's delta and update accumulation stats
        if let Some(pool) = &self.delta_pool {
            if let Some(delta_arc) = pool.get(subject) {
                let mut delta = delta_arc.lock();
                delta.accumulated_steps += adaptation.ttt_result.steps_performed as u64;
                delta.request_count += 1;
                let n = delta.request_count as f64;
                delta.avg_loss_improvement = delta.avg_loss_improvement * ((n - 1.0) / n)
                    + adaptation.ttt_result.loss_improvement as f64 / n;
            }
        }

        debug!(
            "Committed adaptation for subject '{}': steps={}, improvement={:.4}",
            subject, adaptation.ttt_result.steps_performed, adaptation.ttt_result.loss_improvement
        );

        Ok(())
    }

    /// Clean up expired pending adaptations (auto-rollback after timeout)
    fn cleanup_expired_adaptations(&self) {
        let mut pending = self.pending_adaptations.lock();
        let expired: Vec<Subject> = pending.iter()
            .filter(|(_, a)| a.created_at.elapsed().as_millis() as u64 > a.timeout_ms)
            .map(|(s, _)| s.clone())
            .collect();

        for subject in expired {
            if let Some(adaptation) = pending.remove(&subject) {
                // Restore delta to pre-adaptation state
                if let Some(pool) = &self.delta_pool {
                    if let Some(delta_arc) = pool.get(&subject) {
                        let mut delta = delta_arc.lock();
                        if let Err(e) = delta.load_state_dict(&adaptation.pre_adaptation_state) {
                            warn!("Failed to rollback expired adaptation for '{}': {}", subject, e);
                        }
                    }
                }
                warn!(
                    "Auto-rolled back expired adaptation for '{}' (timeout {}ms)",
                    subject, adaptation.timeout_ms
                );
            }
        }
    }

    /// Rollback a pending TTT adaptation
    fn handle_rollback_adaptation(&self, subject: &Subject) -> Result<()> {
        let mut pending = self.pending_adaptations.lock();

        let adaptation = pending.remove(subject)
            .ok_or_else(|| anyhow!("No pending adaptation for subject '{}'", subject))?;

        // Restore delta to pre-adaptation state
        if let Some(pool) = &self.delta_pool {
            if let Some(delta_arc) = pool.get(subject) {
                let mut delta = delta_arc.lock();
                delta.load_state_dict(&adaptation.pre_adaptation_state)?;
            }
        }

        debug!("Rolled back adaptation for subject '{}'", subject);
        Ok(())
    }

    /// Run pure training steps without generation
    fn handle_train_step(
        &self,
        subject: &Subject,
        input: &str,
        gradient_steps: u32,
        learning_rate: f32,
        auto_commit: bool,
    ) -> Result<crate::training::ttt::TTTResult> {
        let ttt_trainer = self.ttt_trainer.as_ref()
            .ok_or_else(|| anyhow!("TTT not configured"))?;
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow!("No tokenizer available"))?;
        let pool = self.delta_pool.as_ref()
            .ok_or_else(|| anyhow!("Delta pool not initialized"))?;

        let delta_arc = pool.get_or_create(subject)?;
        let mut delta = delta_arc.lock();

        let encoding = tokenizer.encode(input, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let input_tokens: Vec<u32> = encoding.get_ids().to_vec();

        let steps = if gradient_steps > 0 { gradient_steps as usize } else { ttt_trainer.config.gradient_steps };
        let lr = if learning_rate > 0.0 { Some(learning_rate as f64) } else { None };

        // Snapshot delta state before training (for rollback if not auto-committing)
        let pre_snapshot = if !auto_commit {
            delta.extract_state_dict()
        } else {
            std::collections::HashMap::new()
        };

        let engine = self.engine.read();
        let mut result = ttt_trainer.train_step(&engine, &mut delta, &input_tokens, steps, lr)?;

        // If auto_commit and recommendation is positive, commit immediately.
        // If auto_commit and recommendation is negative, rollback.
        // If not auto_commit, store as pending for client to decide.
        if auto_commit && result.recommendation {
            delta.accumulated_steps += result.steps_performed as u64;
            delta.request_count += 1;
            result.pending = false;
        } else if auto_commit && !result.recommendation {
            // Auto-rollback
            if !pre_snapshot.is_empty() {
                let _ = delta.load_state_dict(&pre_snapshot);
            }
            result.pending = false;
        } else if !auto_commit {
            result.pending = true;
            // Store pending adaptation for later commit/rollback
            if !pre_snapshot.is_empty() {
                let pending = PendingAdaptation {
                    pre_adaptation_state: pre_snapshot,
                    ttt_result: result.clone(),
                    created_at: std::time::Instant::now(),
                    timeout_ms: 30_000,
                };
                self.pending_adaptations.lock().insert(subject.clone(), pending);
            }
        }

        Ok(result)
    }

    /// Reset a tenant's delta to zeros
    fn handle_reset_delta(&self, subject: &Subject) -> Result<()> {
        // Clear any pending adaptation
        self.pending_adaptations.lock().remove(subject);

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
        let has_pending = self.pending_adaptations.lock().contains_key(subject);

        if let Some(pool) = &self.delta_pool {
            if let Some(delta_arc) = pool.get(subject) {
                let delta = delta_arc.lock();
                let norm_ratios = delta.delta_norm_ratio(pool.base_weight_norms());
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

    /// Clean up timed-out pending adaptations
    fn cleanup_pending_adaptations(&self) {
        let mut pending = self.pending_adaptations.lock();
        let now = Instant::now();
        pending.retain(|tenant_id, adaptation| {
            let elapsed = now.duration_since(adaptation.created_at).as_millis() as u64;
            if elapsed > adaptation.timeout_ms {
                // Auto-rollback: restore pre-adaptation state
                if let Some(pool) = &self.delta_pool {
                    if let Some(delta_arc) = pool.get(tenant_id) {
                        let mut delta = delta_arc.lock();
                        let _ = delta.load_state_dict(&adaptation.pre_adaptation_state);
                    }
                }
                debug!("Auto-rolled back timed-out adaptation for subject '{}'", tenant_id);
                false
            } else {
                true
            }
        });
    }

    /// Get TTT configuration (for status queries)
    #[allow(dead_code)]
    fn get_ttt_config(&self) -> Option<crate::training::ttt::TTTConfig> {
        self.ttt_trainer.as_ref().map(|trainer| trainer.config.clone())
    }

    /// Handle set session
    fn handle_set_session(&self, session_id: String) -> Result<()> {
        // Track session ID for events
        *self.session_id.write() = Some(session_id.clone());
        self.engine
            .write()
            .set_session(CacheOwner::Session(session_id))
    }

    /// Handle clear session
    fn handle_clear_session(&self) {
        *self.session_id.write() = None;
        self.engine.write().clear_kv_cache();
    }

    /// Handle release session
    fn handle_release_session(&self, session_id: &str) -> Result<()> {
        self.engine
            .write()
            .release_session(&CacheOwner::Session(session_id.to_owned()))
    }

    /// Load a LoRA adapter from a safetensors file as the base delta.
    ///
    /// The loaded adapter is stored as `base_delta` and applied to all inference
    /// requests. If a per-tenant TTT delta also exists, the two are composed
    /// (corrections summed) during inference via `resolve_delta()`.
    pub async fn load_lora(&self, path: &Path) -> Result<()> {
        let device = self.engine.read().device();
        // Read via FsOps (path-contained)
        let fs = self.fs.as_ref()
            .ok_or_else(|| anyhow!("FsOps not available — cannot read without path containment"))?;
        let rel_path = path.to_string_lossy();
        let bytes = fs.read_file_chunked(&rel_path).await
            .map_err(|e| anyhow!("Failed to read LoRA adapter via FsOps: {}", e))?;
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
    HealthStatus, TrainStepResult, DeltaStatusResult, ModuleNormRatio,
    SaveAdaptationResult, SnapshotDeltaResult, ExportPeftResult,
    OnlineTrainingMetrics, QualityMetrics, FinishReasonEnum,
    ChatTemplateRequest, LoraConfig, TrainStepRequest, SaveAdaptationRequest, ExportPeftRequest,
};
// Conflicting names — use canonical path at usage sites:
//   inference_client::GenerationResult, inference_client::ModelInfo, inference_client::StreamInfo

#[async_trait::async_trait(?Send)]
impl InferenceHandler for InferenceService {
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

    async fn handle_generate(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &crate::services::generated::inference_client::GenerationRequest,
    ) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        let request = GenerationRequest {
            prompt: crate::config::TemplatedPrompt::new(data.prompt.clone()),
            max_tokens: data.max_tokens as usize,
            temperature: data.temperature,
            top_p: data.top_p,
            top_k: if data.top_k == 0 { None } else { Some(data.top_k as usize) },
            repeat_penalty: data.repeat_penalty,
            repeat_last_n: data.repeat_last_n as usize,
            stop_tokens: data.stop_tokens.clone(),
            seed: if data.seed == 0 { None } else { Some(data.seed) },
            images: vec![],
            timeout: if data.timeout_ms == 0 { None } else { Some(data.timeout_ms) },
            collect_metrics: false,
        };
        let result = InferenceService::handle_generate(self, request, &subject)?;
        Ok(InferenceResponseVariant::GenerateResult(generation_result_to_data(&result)))
    }

    async fn handle_generate_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &crate::services::generated::inference_client::GenerationRequest,
    ) -> Result<(crate::services::generated::inference_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let subject = ctx.subject();
        let request = GenerationRequest {
            prompt: crate::config::TemplatedPrompt::new(data.prompt.clone()),
            max_tokens: data.max_tokens as usize,
            temperature: data.temperature,
            top_p: data.top_p,
            top_k: if data.top_k == 0 { None } else { Some(data.top_k as usize) },
            repeat_penalty: data.repeat_penalty,
            repeat_last_n: data.repeat_last_n as usize,
            stop_tokens: data.stop_tokens.clone(),
            seed: if data.seed == 0 { None } else { Some(data.seed) },
            images: vec![],
            timeout: if data.timeout_ms == 0 { None } else { Some(data.timeout_ms) },
            collect_metrics: false,
        };

        // Calculate expiry from claims
        let expiry_secs = ctx.claims()
            .map(|c| c.exp - chrono::Utc::now().timestamp())
            .unwrap_or(300)
            .max(60);

        let client_ephemeral_pubkey = ctx.ephemeral_pubkey();
        let claims = ctx.claims().cloned();
        let (stream_id, server_pubkey, pending) = self.prepare_stream(request, client_ephemeral_pubkey, claims, expiry_secs, &subject).await?;

        let stream_sub_endpoint = hyprstream_rpc::registry::global()
            .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
            .to_zmq_string();

        let stream_info = crate::services::generated::inference_client::StreamInfo {
            stream_id,
            endpoint: stream_sub_endpoint,
            server_pubkey,
        };

        // Build continuation that executes the stream after REP is sent.
        // Uses UnsafeSendFuture because execute_stream holds a parking_lot::RwLockReadGuard
        // (which is !Send) across an .await during streaming token generation.
        // This is safe because the continuation only runs on InferenceService's
        // single-threaded tokio runtime.
        let service = self.clone();
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(UnsafeSendFuture(Box::pin(async move {
            service.execute_stream(pending).await;
        })));

        Ok((stream_info, continuation))
    }

    async fn handle_model_info(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let info = InferenceService::handle_model_info(self);
        let has_lora = self.base_delta.lock().is_some();
        Ok(InferenceResponseVariant::ModelInfoResult(crate::services::generated::inference_client::ModelInfo {
            model_id: info.name,
            architecture: info.architecture,
            vocab_size: info.vocab_size as u32,
            hidden_size: info.hidden_size as u32,
            num_layers: info.num_hidden_layers.unwrap_or(0) as u32,
            num_heads: info.num_attention_heads.unwrap_or(0) as u32,
            max_sequence_length: info.context_length as u32,
            quantization: info.quantization.unwrap_or_default(),
            has_vision: false,
            lora_loaded: has_lora,
        }))
    }

    async fn handle_is_ready(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let ready = InferenceService::handle_is_ready(self);
        Ok(InferenceResponseVariant::IsReadyResult(ready))
    }

    async fn handle_apply_chat_template(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &ChatTemplateRequest,
    ) -> Result<InferenceResponseVariant> {
        let chat_messages: Vec<crate::runtime::template_engine::ChatMessage> = data.messages
            .iter()
            .map(|m| crate::runtime::template_engine::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        let result = InferenceService::handle_apply_chat_template(self, chat_messages, data.add_generation_prompt)?;
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
        InferenceService::handle_create_lora(self, config)?;
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
        InferenceService::handle_set_session(self, value.to_owned())?;
        Ok(InferenceResponseVariant::SetSessionResult)
    }

    async fn handle_clear_session(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        InferenceService::handle_clear_session(self);
        Ok(InferenceResponseVariant::ClearSessionResult)
    }

    async fn handle_release_session(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> Result<InferenceResponseVariant> {
        InferenceService::handle_release_session(self, value)?;
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

    async fn handle_commit_adaptation(&self, ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        self.cleanup_pending_adaptations();
        InferenceService::handle_commit_adaptation(self, &subject)?;
        Ok(InferenceResponseVariant::CommitAdaptationResult)
    }

    async fn handle_rollback_adaptation(&self, ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        InferenceService::handle_rollback_adaptation(self, &subject)?;
        Ok(InferenceResponseVariant::RollbackAdaptationResult)
    }

    async fn handle_train_step(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &TrainStepRequest,
    ) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        let result = InferenceService::handle_train_step(self, &subject, &data.input, data.gradient_steps, data.learning_rate, data.auto_commit)?;
        Ok(InferenceResponseVariant::TrainStepResult(TrainStepResult {
            avg_loss: result.avg_loss,
            loss_improvement: result.loss_improvement,
            steps_performed: result.steps_performed as u32,
            adaptation_time_ms: result.adaptation_time_ms,
            initial_perplexity: result.initial_perplexity,
            final_perplexity: result.final_perplexity,
            recommendation: result.recommendation,
            committed: !result.pending,
            gradient_clipped: result.gradient_clipped,
        }))
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

        let pending = PendingWork::Training {
            stream_ctx,
            subject,
            input: data.input.clone(),
            gradient_steps: data.gradient_steps,
            learning_rate: data.learning_rate,
            auto_commit: data.auto_commit,
        };

        // Build continuation
        let service = self.clone();
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            service.execute_training_stream(pending).await;
        });

        Ok((stream_info, continuation))
    }

    async fn handle_reset_delta(&self, ctx: &EnvelopeContext, _request_id: u64) -> Result<InferenceResponseVariant> {
        let subject = ctx.subject();
        InferenceService::handle_reset_delta(self, &subject)?;
        Ok(InferenceResponseVariant::ResetDeltaResult)
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
        let info = InferenceService::handle_save_adaptation(self, &subject, &data.name, &data.merge_strategy, data.merge_weight).await?;
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
        let info = InferenceService::handle_export_peft_adapter(self, &subject, &data.name, &data.commit_message).await?;
        Ok(InferenceResponseVariant::ExportPeftAdapterResult(ExportPeftResult {
            adapter_path: info.adapter_path,
            content_hash: info.content_hash,
        }))
    }
}

/// Convert a GenerationResult to the generated data type.
fn generation_result_to_data(result: &GenerationResult) -> crate::services::generated::inference_client::GenerationResult {
    let finish_reason = match &result.finish_reason {
        FinishReason::MaxTokens => FinishReasonEnum::MaxTokens,
        FinishReason::StopToken(_) => FinishReasonEnum::StopToken,
        FinishReason::EndOfSequence => FinishReasonEnum::EndOfSequence,
        FinishReason::Error(_) => FinishReasonEnum::Error,
        FinishReason::Stop => FinishReasonEnum::Stop,
    };
    crate::services::generated::inference_client::GenerationResult {
        text: result.text.clone(),
        tokens_generated: result.tokens_generated as u32,
        finish_reason,
        generation_time_ms: result.generation_time_ms,
        tokens_per_second: result.tokens_per_second,
        quality_metrics: QualityMetrics::default(),
        prefill_tokens: result.prefill_tokens as u32,
        prefill_time_ms: result.prefill_time_ms,
        prefill_tokens_per_sec: result.prefill_tokens_per_sec,
        inference_tokens: result.inference_tokens as u32,
        inference_time_ms: result.inference_time_ms,
        inference_tokens_per_sec: result.inference_tokens_per_sec,
        online_training_metrics: result.ttt_metrics.as_ref()
            .map(|m| OnlineTrainingMetrics {
                avg_loss: m.avg_loss,
                loss_improvement: m.loss_improvement,
                steps_performed: m.steps_performed as u32,
                adaptation_time_ms: m.adaptation_time_ms,
                skipped: m.skipped,
                skip_reason: m.skip_reason.clone().unwrap_or_default(),
                avg_grad_norm: m.avg_grad_norm,
                max_grad_norm: m.max_grad_norm,
                gradient_clipped: m.gradient_clipped,
                tokens_used: m.tokens_used as u32,
                tokens_provided: m.tokens_provided as u32,
                was_truncated: m.was_truncated,
                initial_perplexity: m.initial_perplexity,
                final_perplexity: m.final_perplexity,
                recommendation: m.recommendation,
                gated_steps: m.gated_steps as u32,
                pending: m.pending,
            })
            .unwrap_or_default(),
    }
}

// Note: InferenceService does NOT implement ZmqService because it uses a custom
// single-threaded tokio runtime in `run_service_loop` (not RequestLoop).
// Both `run_service_loop` and `handle_callback_infer` call `dispatch_inference` directly.

/// Client for the inference service
///
/// # Security
///
/// All requests are wrapped in `SignedEnvelope` for authentication:
/// - Requests are signed with the client's Ed25519 signing key
/// - The service verifies signatures before processing
/// - Identity is included for authorization checks
///
/// Wraps a generated `InferenceClient`. All methods delegate to the
/// autogenerated typed client.
#[derive(Clone)]
pub struct InferenceZmqClient {
    /// Generated typed client (handles all transport including streaming via call_with_options)
    pub(crate) gen: crate::services::generated::inference_client::InferenceClient,
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

    /// Convert generated FinishReasonEnum to domain FinishReason
    fn parse_finish_reason_enum(
        reason: crate::services::generated::inference_client::FinishReasonEnum,
    ) -> FinishReason {
        use crate::services::generated::inference_client::FinishReasonEnum;
        match reason {
            FinishReasonEnum::MaxTokens => FinishReason::MaxTokens,
            FinishReasonEnum::StopToken => FinishReason::StopToken(String::new()),
            FinishReasonEnum::EndOfSequence => FinishReason::EndOfSequence,
            FinishReasonEnum::Error => FinishReason::Error(String::new()),
            FinishReasonEnum::Stop => FinishReason::Stop,
        }
    }

    /// Generate text (non-streaming) — delegates to generated client
    pub async fn generate(&self, request: &GenerationRequest) -> Result<GenerationResult> {
        let r = self.gen.generate(
            request.prompt.as_str(),
            request.max_tokens as u32,
            request.temperature,
            request.top_p,
            request.top_k.unwrap_or(0) as u32,
            request.repeat_penalty,
            request.repeat_last_n as u32,
            &request.stop_tokens,
            request.seed.unwrap_or(0),
            &[], // images
            request.timeout.unwrap_or(0),
        ).await?;
        Ok(GenerationResult {
            text: r.text,
            tokens_generated: r.tokens_generated as usize,
            finish_reason: Self::parse_finish_reason_enum(r.finish_reason),
            generation_time_ms: r.generation_time_ms,
            tokens_per_second: r.tokens_per_second,
            quality_metrics: None,
            prefill_tokens: r.prefill_tokens as usize,
            prefill_time_ms: r.prefill_time_ms,
            prefill_tokens_per_sec: r.prefill_tokens_per_sec,
            inference_tokens: r.inference_tokens as usize,
            inference_time_ms: r.inference_time_ms,
            inference_tokens_per_sec: r.inference_tokens_per_sec,
            ttt_metrics: None,  // TODO: Extract from response when available
        })
    }

    /// Check if model is ready
    pub async fn is_ready(&self) -> Result<bool> {
        self.gen.is_ready().await
    }

    /// Get model info
    pub async fn model_info(&self) -> Result<ModelInfo> {
        let r = self.gen.model_info().await?;
        Ok(ModelInfo {
            name: r.model_id,
            architecture: r.architecture,
            vocab_size: r.vocab_size as usize,
            hidden_size: r.hidden_size as usize,
            num_hidden_layers: Some(r.num_layers as usize),
            num_attention_heads: Some(r.num_heads as usize),
            num_key_value_heads: None,
            head_dim: None,
            context_length: r.max_sequence_length as usize,
            quantization: Some(r.quantization),
            parameters: 0,
            intermediate_size: None,
        })
    }

    /// Health check
    pub async fn health_check(&self) -> Result<()> {
        let _status = self.gen.health_check().await?;
        Ok(())
    }

    /// Start streaming generation with E2E authentication — delegates to generated client
    pub async fn generate_stream(
        &self,
        request: &GenerationRequest,
        ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamInfo> {
        self.gen.generate_stream(
            request.prompt.as_str(),
            request.max_tokens as u32,
            request.temperature,
            request.top_p,
            request.top_k.unwrap_or(0) as u32,
            request.repeat_penalty,
            request.repeat_last_n as u32,
            &request.stop_tokens,
            request.seed.unwrap_or(0),
            &[], // images
            request.timeout.unwrap_or(0),
            ephemeral_pubkey.unwrap_or([0u8; 32]),
        ).await
    }

    /// Start streaming generation with E2E authenticated handle.
    ///
    /// Returns the canonical `StreamHandle` from hyprstream-rpc which provides:
    /// - `recv_next()` for blocking receive with HMAC verification
    /// - `try_next()` for non-blocking receive
    ///
    /// Use `StreamChunkMessage::from_stream_payload()` to convert received
    /// `StreamPayload` values to inference-specific message types.
    pub async fn generate_stream_handle(
        &self,
        request: &GenerationRequest,
    ) -> Result<crate::services::rpc_types::StreamHandle> {
        use hyprstream_rpc::crypto::generate_ephemeral_keypair;

        let (client_secret, client_pubkey) = generate_ephemeral_keypair();
        let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

        let info = self.generate_stream(request, Some(client_pubkey_bytes)).await?;

        if info.server_pubkey == [0u8; 32] {
            anyhow::bail!(
                "Server did not provide Ristretto255 public key - E2E authentication required"
            );
        }

        // Use the canonical StreamHandle which performs DH key exchange internally
        crate::services::rpc_types::StreamHandle::new(
            &global_context(),
            info.stream_id,
            &info.endpoint,
            &info.server_pubkey,
            &client_secret,
            &client_pubkey_bytes,
        )
    }

    /// Apply chat template — delegates to generated client
    pub async fn apply_chat_template(
        &self,
        messages: &[crate::runtime::template_engine::ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        use crate::services::generated::inference_client::ChatMessage;
        let msg_data: Vec<ChatMessage> = messages.iter().map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        }).collect();
        self.gen.apply_chat_template(&msg_data, add_generation_prompt).await
    }

    /// Create a new LoRA adapter
    pub async fn create_lora(&self, config: &TenantDeltaConfig) -> Result<()> {
        self.gen.create_lora(
            config.rank as u32, config.alpha, config.dropout, &config.target_modules, config.learning_rate as f32,
        ).await
    }

    /// Load a LoRA adapter from a safetensors file (delegates via RPC).
    pub async fn load_lora(&self, path: &str) -> Result<()> {
        self.gen.load_lora(path).await
    }

    /// Save the current LoRA adapter to a safetensors file (delegates via RPC).
    pub async fn save_lora(&self, path: &str) -> Result<()> {
        self.gen.save_lora(path).await
    }

    /// Unload the current LoRA adapter (delegates via RPC).
    pub async fn unload_lora(&self) -> Result<()> {
        self.gen.unload_lora().await
    }

    /// Check if a LoRA adapter is loaded (delegates via RPC).
    pub async fn has_lora(&self) -> Result<bool> {
        self.gen.has_lora().await
    }

    // Training loop control (TTT operations)

    /// Commit a pending TTT adaptation
    pub async fn commit_adaptation(&self) -> Result<()> {
        self.gen.commit_adaptation().await
    }

    /// Rollback a pending TTT adaptation
    pub async fn rollback_adaptation(&self) -> Result<()> {
        self.gen.rollback_adaptation().await
    }

    /// Reset a tenant's delta
    pub async fn reset_delta(&self) -> Result<()> {
        self.gen.reset_delta().await
    }

    /// Start streaming training step with E2E authentication — delegates to generated client
    pub async fn train_step_stream(
        &self,
        input: &str,
        gradient_steps: u32,
        learning_rate: f32,
        auto_commit: bool,
        ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamInfo> {
        self.gen.train_step_stream(
            input,
            gradient_steps,
            learning_rate,
            auto_commit,
            ephemeral_pubkey.unwrap_or([0u8; 32]),
        ).await
    }

    /// Set the current session ID for KV cache management
    pub async fn set_session(&self, session_id: &str) -> Result<()> {
        self.gen.set_session(session_id).await
    }

    /// Clear the current session's KV cache
    pub async fn clear_session(&self) -> Result<()> {
        self.gen.clear_session().await
    }

    /// Release a session's KV cache
    pub async fn release_session(&self, session_id: &str) -> Result<()> {
        self.gen.release_session(session_id).await
    }

    /// Request service shutdown
    pub async fn shutdown(&self) -> Result<()> {
        match self.gen.shutdown().await? {
            InferenceResponseVariant::Success => Ok(()),
            InferenceResponseVariant::Error(ref e) => Err(anyhow!("{}", e.message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
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
