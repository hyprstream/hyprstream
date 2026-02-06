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
//! # Authorization (Future Enhancement)
//!
//! Currently uses manual Casbin checks via `check_auth()`. Can be refactored to use
//! `#[authorize]` macro when moving to typed handlers:
//!
//! ```ignore
//! #[register_scopes]
//! impl InferenceService {
//!     #[authorize(action = "infer", resource = "model", identifier_field = "model")]
//!     fn handle_generate(&self, ctx: &EnvelopeContext, req: GenerateRequest) -> Result<Vec<u8>> {
//!         // JWT validation and scope checks automatically enforced
//!         // ctx.user_claims is guaranteed to exist here
//!     }
//! }
//! ```

use crate::auth::Operation;
use crate::services::PolicyClient;
use crate::config::{FinishReason, GenerationRequest, GenerationResult, ModelInfo, TrainingMode};
use crate::inference_capnp;
use anyhow::bail;
use crate::lora::LoRAConfig;
use crate::runtime::kv_cache::CacheOwner;
use crate::runtime::model_config::ModelConfig;
use crate::runtime::{RuntimeConfig, RuntimeEngine, TorchEngine};
use crate::services::rpc_types::{InferenceResponse, StreamStartedInfo};
use crate::services::{CallOptions, EnvelopeContext};
use crate::training::{TTTConfig, TestTimeTrainer};
use crate::zmq::global_context;
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::StreamChannel;
use capnp::serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use tmq::{reply, Multipart};
use tokio::runtime::Handle;
use tokenizers::Tokenizer;
use tracing::{debug, error, info, trace, warn};

/// Pending stream to be executed after REP response is sent.
///
/// This solves the streaming deadlock where the service waits for subscription
/// before returning the response, but the client can't subscribe without
/// the stream_id from the response.
///
/// Wraps `StreamContext` from hyprstream-rpc with inference-specific data.
struct PendingStream {
    /// Stream context with DH-derived keys (from hyprstream-rpc)
    stream_ctx: hyprstream_rpc::StreamContext,
    /// Generation request to execute
    request: GenerationRequest,
    /// Prompt text for TTT training collection
    prompt: String,
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
pub struct InferenceService {
    engine: RwLock<TorchEngine>,
    /// Model identifier for events
    model_id: String,
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
    ) -> Result<hyprstream_rpc::service::SpawnedService> {
        let model_path = model_path.as_ref().to_path_buf();
        let endpoint_owned = endpoint.to_owned();
        let model_id = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| "unknown".to_owned());
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
                match Self::initialize(model_path, config, model_id, server_pubkey, signing_key, nonce_cache, policy_client).await {
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
        let server_pubkey = VerifyingKey::default(); // Callback mode doesn't need signature verification
        // Generate signing key for callback mode (separate process, no shared key access)
        let signing_key = hyprstream_rpc::crypto::signing::generate_signing_keypair().0;
        let nonce_cache = Arc::new(InMemoryNonceCache::new());
        let mut service = Self::initialize(
            model_path.clone(),
            config,
            model_ref.clone(),
            server_pubkey,
            signing_key.clone(),
            nonce_cache,
            policy_client,
        )
        .await?;

        // Create StreamChannel for streaming operations
        let stream_channel = StreamChannel::new(
            Arc::clone(&ctx),
            signing_key,
        );
        service.stream_channel = Some(stream_channel);

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
                            info!("Received Shutdown before LoadModel, exiting");
                            std::process::exit(0);
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
                info!("Received Shutdown command, exiting");
                std::process::exit(0);
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

        // Reuse existing request handling
        let (response, _pending_stream) = self.handle_request(&ctx, request_data).await?;
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
        model_id: String,
        server_pubkey: VerifyingKey,
        signing_key: SigningKey,
        nonce_cache: Arc<InMemoryNonceCache>,
        policy_client: PolicyClient,
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

        // Use provided signing key for response signing
        Ok(Self {
            engine: RwLock::new(engine),
            model_id,
            model_path,
            session_id: RwLock::new(None),
            runtime_handle,
            stream_channel: None, // Initialized in run_service_loop
            server_pubkey,
            signing_key,
            nonce_cache,
            policy_client,
            ttt_trainer,
            tokenizer,
        })
    }

    /// Run the service loop (async with TMQ)
    ///
    /// The `ready_tx` channel signals when sockets are bound and the service is ready.
    /// This ensures callers wait for actual readiness, not just initialization.
    async fn run_service_loop(
        mut service: Self,
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

        // Create StreamChannel for streaming generation
        // Uses lazy socket initialization - socket is created on first use
        let stream_channel = StreamChannel::new(
            Arc::clone(&ctx),
            service.signing_key.clone(),
        );
        service.stream_channel = Some(stream_channel);

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

            // Unwrap and verify SignedEnvelope
            let (envelope_ctx, payload) = match hyprstream_rpc::unwrap_envelope(&request, &service.server_pubkey, &*service.nonce_cache) {
                Ok((ctx, payload)) => (ctx, payload),
                Err(e) => {
                    warn!("inference envelope verification failed: {}", e);
                    // No valid request_id, send empty response (like RequestLoop does)
                    let msg: Multipart = vec![vec![]].into();
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
                envelope_ctx.casbin_subject(),
                envelope_ctx.request_id
            );

            // Handle request - may return pending stream work (now async for policy checks)
            let request_id = envelope_ctx.request_id;
            let (response_payload, pending_stream) = match service.handle_request(&envelope_ctx, &payload).await {
                Ok((resp, pending)) => (resp, pending),
                Err(e) => {
                    error!("inference request handling error: {}", e);
                    (InferenceResponse::error(request_id, &e.to_string()), None)
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

            // THEN execute any pending stream (after response is sent)
            // This solves the streaming deadlock - client can subscribe after
            // receiving the stream_id in the response
            if let Some(pending) = pending_stream {
                service.execute_stream(pending).await;
            }
        }
    }

    /// Apply TTT adaptation if enabled (adapts model to input BEFORE generation)
    fn apply_ttt_adaptation(&self, prompt: &str) {
        let ttt_trainer = match &self.ttt_trainer {
            Some(t) => t.clone(),
            None => return,
        };

        let tokenizer = match &self.tokenizer {
            Some(t) => t.clone(),
            None => {
                trace!("No tokenizer available for TTT adaptation");
                return;
            }
        };

        // Tokenize prompt for TTT
        let encoding = match tokenizer.encode(prompt, false) {
            Ok(enc) => enc,
            Err(e) => {
                warn!("Failed to tokenize for TTT: {}", e);
                return;
            }
        };

        let input_tokens: Vec<u32> = encoding.get_ids().to_vec();

        // Apply TTT adaptation BEFORE generation
        let engine = self.engine.read();
        match ttt_trainer.adapt(&engine, &input_tokens) {
            Ok(result) => {
                if result.skipped {
                    trace!("TTT skipped: {:?}", result.skip_reason);
                } else {
                    debug!(
                        "TTT adaptation: steps={}, loss_improvement={:.4}, time={}ms",
                        result.steps_performed, result.loss_improvement, result.adaptation_time_ms
                    );
                }
            }
            Err(e) => {
                warn!("TTT adaptation failed: {}", e);
            }
        }
    }

    /// Handle non-streaming generation
    fn handle_generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        // Apply TTT adaptation BEFORE generation (if enabled)
        self.apply_ttt_adaptation(request.prompt.as_str());

        // Use futures::executor::block_on because we're already inside a tokio runtime
        // (run_service_loop runs in rt.block_on), and tokio's block_on can't be nested.
        // futures::executor::block_on works because it's a simple single-threaded executor.
        let engine = self.engine.read();
        futures::executor::block_on(async {
            RuntimeEngine::generate_with_params(&*engine, request).await
        })
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
    ) -> Result<(String, [u8; 32], PendingStream)> {
        let prompt = request.prompt.as_str().to_owned();

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

        let pending = PendingStream {
            stream_ctx,
            request,
            prompt,
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
    async fn execute_stream(&self, pending: PendingStream) {
        use futures::StreamExt;

        let stream_ctx = &pending.stream_ctx;
        let request = pending.request;
        let prompt = pending.prompt;

        // Apply TTT adaptation BEFORE streaming generation (if enabled)
        self.apply_ttt_adaptation(&prompt);

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
            "Starting E2E authenticated stream via StreamChannel"
        );

        // Run the stream with StreamChannel's async publisher callback
        let engine = self.engine.read();
        let stream_result = engine.generate(request);

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
                        let complete = crate::services::rpc_types::InferenceComplete::from(&stats);
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
    fn handle_create_lora(&self, config: LoRAConfig) -> Result<()> {
        self.engine.write().create_lora(config)
    }

    /// Handle load LoRA
    fn handle_load_lora(&self, path: &Path) -> Result<()> {
        // Use futures::executor::block_on to avoid nesting tokio runtimes
        let mut engine = self.engine.write();
        futures::executor::block_on(async { engine.load_lora_from_file(path).await })
    }

    /// Handle save LoRA
    fn handle_save_lora(&self, path: &str) -> Result<()> {
        self.engine.read().save_lora(path)
    }

    /// Handle unload LoRA
    fn handle_unload_lora(&self) -> Result<()> {
        self.engine.write().unload_lora()
    }

    /// Handle has LoRA
    fn handle_has_lora(&self) -> bool {
        self.engine.read().has_lora_model()
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

    /// Parse a generation request from capnp
    fn parse_generation_request(
        &self,
        reader: inference_capnp::generation_request::Reader,
    ) -> Result<GenerationRequest> {
        use crate::config::TemplatedPrompt;
        let prompt = TemplatedPrompt::new(reader.get_prompt()?.to_str()?.to_owned());
        let max_tokens = reader.get_max_tokens() as usize;
        let temperature = reader.get_temperature();
        let top_p = reader.get_top_p();
        let top_k = if reader.get_top_k() == 0 {
            None
        } else {
            Some(reader.get_top_k() as usize)
        };
        let repeat_penalty = reader.get_repeat_penalty();
        let repeat_last_n = reader.get_repeat_last_n() as usize;

        let stop_tokens: Vec<String> = reader
            .get_stop_tokens()?
            .iter()
            .filter_map(|s| s.ok().and_then(|t| t.to_str().ok().map(std::borrow::ToOwned::to_owned)))
            .collect();

        let seed = if reader.get_seed() == 0 {
            None
        } else {
            Some(reader.get_seed())
        };

        // Parse images if present - convert from bytes to paths
        let images: Vec<String> = Vec::new(); // Images are bytes in capnp, but strings in GenerationRequest

        let timeout = if reader.get_timeout_ms() == 0 {
            None
        } else {
            Some(reader.get_timeout_ms())
        };

        Ok(GenerationRequest {
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            repeat_last_n,
            stop_tokens,
            seed,
            images,
            timeout,
            collect_metrics: false, // Default: off for performance
        })
    }
}

impl InferenceService {
    /// Check authorization for an operation.
    ///
    /// Returns the unauthorized response if the check fails, or None if authorized.
    /// Uses PolicyClient for async policy checks via TMQ.
    async fn check_auth(
        &self,
        ctx: &EnvelopeContext,
        request_id: u64,
        resource: &str,
        operation: Operation,
    ) -> Option<Vec<u8>> {
        let subject = ctx.casbin_subject();

        // Async policy check via TMQ
        let allowed = self.policy_client
            .check_policy(&subject, resource, operation)
            .await
            .unwrap_or(false);

        if allowed {
            None // Authorized
        } else {
            debug!(
                "Authorization denied: {} cannot {} on {}",
                subject,
                operation.as_str(),
                resource
            );
            Some(InferenceResponse::unauthorized(
                request_id,
                &subject,
                resource,
                operation.as_str(),
            ))
        }
    }

    /// Handle a capnp request and return a response with optional pending stream.
    ///
    /// Returns (response_bytes, pending_stream) where pending_stream is Some
    /// for streaming requests. The caller should send the response FIRST,
    /// then execute the pending stream.
    async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<PendingStream>)> {
        // Log identity for audit trail
        trace!(
            "Inference request from {} (envelope_id={}, authenticated={})",
            ctx.casbin_subject(),
            ctx.request_id,
            ctx.is_authenticated()
        );

        // Deserialize inner request from payload
        let reader = serialize::read_message(payload, ReaderOptions::new())?;
        let req = reader.get_root::<inference_capnp::inference_request::Reader>()?;

        let request_id = req.get_id();

        use inference_capnp::inference_request::Which;

        // Resource for authorization checks
        let resource = format!("inference:{}", self.model_id);

        match req.which()? {
            Which::Generate(gen_req) => {
                // Authorization: Infer on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Infer).await {
                    return Ok((resp, None));
                }
                let gen_req = gen_req?;
                let request = self.parse_generation_request(gen_req)?;

                match self.handle_generate(request) {
                    Ok(result) => Ok((InferenceResponse::generation_result(request_id, &result), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::GenerateStream(gen_req) => {
                // Authorization: Infer on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Infer).await {
                    return Ok((resp, None));
                }
                let gen_req = gen_req?;
                let request = self.parse_generation_request(gen_req)?;

                // Calculate expiry from claims (for stream pre-authorization)
                let expiry_secs = ctx.claims()
                    .map(|c| c.exp - chrono::Utc::now().timestamp())
                    .unwrap_or(300) // Default: 5 minutes
                    .max(60); // At least 60 seconds

                // Prepare stream with DH key exchange and pre-authorization via StreamChannel
                let client_ephemeral_pubkey = ctx.ephemeral_pubkey();
                let claims = ctx.claims().cloned();
                let (stream_id, server_pubkey, pending) = self.prepare_stream(
                    request,
                    client_ephemeral_pubkey,
                    claims,
                    expiry_secs,
                ).await?;

                // Get StreamService's Sub endpoint (where clients subscribe)
                let stream_sub_endpoint = hyprstream_rpc::registry::global()
                    .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
                    .to_zmq_string();

                // Return response with stream_id AND server_pubkey for client DH
                let response = InferenceResponse::stream_started(
                    request_id,
                    &stream_id,
                    &stream_sub_endpoint,
                    &server_pubkey,
                );
                Ok((response, Some(pending)))
            }

            Which::ModelInfo(()) => {
                // Authorization: Query on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query).await {
                    return Ok((resp, None));
                }
                let info = self.handle_model_info();
                let has_lora = self.engine.read().has_lora_model();
                Ok((InferenceResponse::model_info(request_id, &info, has_lora), None))
            }

            Which::IsReady(()) => {
                // Authorization: Query on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query).await {
                    return Ok((resp, None));
                }
                let ready = self.handle_is_ready();
                Ok((InferenceResponse::ready(request_id, ready), None))
            }

            Which::ApplyChatTemplate(template_req) => {
                // Authorization: Query on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query).await {
                    return Ok((resp, None));
                }
                let template_req = template_req?;
                let messages: Vec<crate::runtime::template_engine::ChatMessage> = template_req
                    .get_messages()?
                    .iter()
                    .filter_map(|m| {
                        Some(crate::runtime::template_engine::ChatMessage {
                            role: m.get_role().ok()?.to_str().ok()?.to_owned(),
                            content: m.get_content().ok()?.to_str().ok()?.to_owned(),
                        })
                    })
                    .collect();

                let add_generation_prompt = template_req.get_add_generation_prompt();

                match self.handle_apply_chat_template(messages, add_generation_prompt) {
                    Ok(result) => Ok((InferenceResponse::template_result(request_id, &result), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::CreateLora(lora_config) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write).await {
                    return Ok((resp, None));
                }
                let lora_config = lora_config?;
                let config = LoRAConfig {
                    rank: lora_config.get_rank() as usize,
                    alpha: lora_config.get_alpha(),
                    dropout: lora_config.get_dropout(),
                    target_modules: lora_config
                        .get_target_modules()?
                        .iter()
                        .filter_map(|s| s.ok().and_then(|t| t.to_str().ok().map(std::borrow::ToOwned::to_owned)))
                        .collect(),
                    learning_rate: 0.0001, // Default learning rate
                };

                match self.handle_create_lora(config) {
                    Ok(()) => Ok((InferenceResponse::success(request_id), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::LoadLora(path) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write).await {
                    return Ok((resp, None));
                }
                let path = path?.to_str()?;
                match self.handle_load_lora(Path::new(path)) {
                    Ok(()) => Ok((InferenceResponse::success(request_id), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::SaveLora(path) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write).await {
                    return Ok((resp, None));
                }
                let path = path?.to_str()?;
                match self.handle_save_lora(path) {
                    Ok(()) => Ok((InferenceResponse::success(request_id), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::UnloadLora(()) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write).await {
                    return Ok((resp, None));
                }
                match self.handle_unload_lora() {
                    Ok(()) => Ok((InferenceResponse::success(request_id), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::HasLora(()) => {
                // Authorization: Query on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query).await {
                    return Ok((resp, None));
                }
                let has_lora = self.handle_has_lora();
                Ok((InferenceResponse::has_lora(request_id, has_lora), None))
            }

            Which::SetSession(session_id) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write).await {
                    return Ok((resp, None));
                }
                let session_id = session_id?.to_str()?.to_owned();
                match self.handle_set_session(session_id) {
                    Ok(()) => Ok((InferenceResponse::success(request_id), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::ClearSession(()) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write).await {
                    return Ok((resp, None));
                }
                self.handle_clear_session();
                Ok((InferenceResponse::success(request_id), None))
            }

            Which::ReleaseSession(session_id) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write).await {
                    return Ok((resp, None));
                }
                let session_id = session_id?.to_str()?;
                match self.handle_release_session(session_id) {
                    Ok(()) => Ok((InferenceResponse::success(request_id), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::HealthCheck(()) => {
                // Health check is public (no authorization required)
                let model_loaded = self.engine.read().is_loaded();
                Ok((InferenceResponse::health(request_id, model_loaded), None))
            }

            Which::Shutdown(()) => {
                // Authorization: Manage on inference (shutdown requires admin)
                if let Some(resp) = self.check_auth(ctx, request_id, "inference", Operation::Manage).await {
                    return Ok((resp, None));
                }
                info!("Inference service shutdown requested");
                Ok((InferenceResponse::success(request_id), None))
            }

            Which::StartStream(start_req) => {
                // Authorization: Infer on inference:{model} (same as generating)
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Infer).await {
                    return Ok((resp, None));
                }

                let start_req = start_req?;
                let stream_id = start_req.get_stream_id()?.to_str()?.to_owned();

                // Validate stream_id format
                if !stream_id.starts_with("stream-") {
                    return Ok((InferenceResponse::error(request_id, "Invalid stream_id format"), None));
                }

                // Get StreamChannel
                let stream_channel = match &self.stream_channel {
                    Some(sc) => sc,
                    None => {
                        error!(stream_id = %stream_id, "StreamChannel not available for stream authorization");
                        return Ok((InferenceResponse::error(request_id, "Stream service not available"), None));
                    }
                };

                // Send AUTHORIZE message to StreamService via StreamChannel
                // Include expiry from claims so StreamService can GC expired entries
                let exp = ctx.claims().map(|c| c.exp).unwrap_or_else(|| {
                    // Default: 5 minutes from now if no claims
                    chrono::Utc::now().timestamp() + 300
                });
                let claims = ctx.claims().cloned();

                if let Err(e) = stream_channel.register_topic(&stream_id, exp, claims).await {
                    error!(stream_id = %stream_id, error = %e, "Failed to send authorize message to StreamService");
                    return Ok((InferenceResponse::error(request_id, &format!("Stream authorization failed: {e}")), None));
                }
                info!(stream_id = %stream_id, exp = exp, "Stream authorized for subscription");

                Ok((InferenceResponse::stream_authorized(request_id, &stream_id), None))
            }
        }
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
/// Wraps a generated `InferenceClient`. Simple methods delegate to gen;
/// `generate`, `apply_chat_template`, and streaming methods use manual
/// request building because the generator doesn't fully support their types.
#[derive(Clone)]
pub struct InferenceZmqClient {
    /// Generated typed client (handles all transport including streaming via call_with_options)
    pub(crate) gen: crate::services::generated::inference_client::InferenceClient,
}

use crate::services::generated::inference_client::InferenceResponseVariant;

impl InferenceZmqClient {
    /// Create a new inference client with signing credentials
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self::with_endpoint(INFERENCE_ENDPOINT, signing_key, identity)
    }

    /// Create an inference client connected to a specific endpoint
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        use hyprstream_rpc::service::factory::ServiceClient;
        let server_verifying_key = signing_key.verifying_key();
        let base = crate::services::core::ZmqClientBase::new(
            endpoint, global_context(), signing_key, server_verifying_key, identity,
        );
        Self {
            gen: crate::services::generated::inference_client::InferenceClient::from_zmq(base),
        }
    }

    /// Convert generated FinishReasonEnum to domain FinishReason
    fn parse_finish_reason_enum(
        reason: &crate::services::generated::inference_client::FinishReasonEnum,
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
        match self.gen.generate(
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
        ).await? {
            InferenceResponseVariant::GenerateResult {
                text, tokens_generated, finish_reason, generation_time_ms,
                tokens_per_second, prefill_tokens, prefill_time_ms,
                prefill_tokens_per_sec, inference_tokens, inference_time_ms,
                inference_tokens_per_sec, ..
            } => Ok(GenerationResult {
                text,
                tokens_generated: tokens_generated as usize,
                finish_reason: Self::parse_finish_reason_enum(&finish_reason),
                generation_time_ms,
                tokens_per_second,
                quality_metrics: None,
                prefill_tokens: prefill_tokens as usize,
                prefill_time_ms,
                prefill_tokens_per_sec,
                inference_tokens: inference_tokens as usize,
                inference_time_ms,
                inference_tokens_per_sec,
            }),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Check if model is ready
    pub async fn is_ready(&self) -> Result<bool> {
        match self.gen.is_ready().await? {
            InferenceResponseVariant::IsReadyResult(ready) => Ok(ready),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Get model info
    pub async fn model_info(&self) -> Result<ModelInfo> {
        match self.gen.model_info().await? {
            InferenceResponseVariant::ModelInfoResult {
                model_id, architecture, vocab_size, hidden_size,
                num_layers, num_heads, max_sequence_length, quantization, ..
            } => Ok(ModelInfo {
                name: model_id,
                architecture,
                vocab_size: vocab_size as usize,
                hidden_size: hidden_size as usize,
                num_hidden_layers: Some(num_layers as usize),
                num_attention_heads: Some(num_heads as usize),
                context_length: max_sequence_length as usize,
                quantization: Some(quantization),
                parameters: 0,
                intermediate_size: None,
            }),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Health check
    pub async fn health_check(&self) -> Result<()> {
        match self.gen.health_check().await? {
            InferenceResponseVariant::HealthCheckResult { .. } => Ok(()),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Start streaming generation with E2E authentication (manual — needs custom CallOptions)
    pub async fn generate_stream(
        &self,
        request: &GenerationRequest,
        ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<StreamStartedInfo> {
        let id = self.gen.next_id();

        let mut message = Builder::new_default();
        let mut req = message.init_root::<inference_capnp::inference_request::Builder>();
        req.set_id(id);

        let mut gen_req = req.init_generate_stream();
        gen_req.set_prompt(request.prompt.as_str());
        gen_req.set_max_tokens(request.max_tokens as u32);
        gen_req.set_temperature(request.temperature);
        gen_req.set_top_p(request.top_p);
        gen_req.set_top_k(request.top_k.unwrap_or(0) as u32);
        gen_req.set_repeat_penalty(request.repeat_penalty);
        gen_req.set_repeat_last_n(request.repeat_last_n as u32);
        gen_req.set_seed(request.seed.unwrap_or(0));
        gen_req.set_timeout_ms(request.timeout.unwrap_or(0));

        if !request.stop_tokens.is_empty() {
            let mut stop_list = gen_req.init_stop_tokens(request.stop_tokens.len() as u32);
            for (i, token) in request.stop_tokens.iter().enumerate() {
                stop_list.set(i as u32, token);
            }
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;

        let opts = match ephemeral_pubkey {
            Some(pk) => CallOptions::default().ephemeral_pubkey(pk),
            None => CallOptions::default(),
        };
        let response = self.gen.call_with_options(bytes, opts).await?;
        Self::parse_stream_started_response(&response)
    }

    /// Authorize a stream subscription (manual — preserves exact request shape)
    pub async fn start_stream(&self, stream_id: &str) -> Result<String> {
        let id = self.gen.next_id();

        let mut message = Builder::new_default();
        let mut req = message.init_root::<inference_capnp::inference_request::Builder>();
        req.set_id(id);

        let mut start_req = req.init_start_stream();
        start_req.set_stream_id(stream_id);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;

        let response = self.gen.call_with_options(bytes, CallOptions::default()).await?;
        Self::parse_stream_authorized_response(&response)
    }

    /// Start streaming generation with E2E authenticated handle.
    pub async fn generate_stream_handle(
        &self,
        request: &GenerationRequest,
        claims: crate::auth::Claims,
    ) -> Result<StreamHandle> {
        use hyprstream_rpc::crypto::{
            derive_stream_keys, generate_ephemeral_keypair, ristretto_dh, RistrettoPublic,
        };

        let (client_secret, client_pubkey) = generate_ephemeral_keypair();
        let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

        let info = self.generate_stream(request, Some(client_pubkey_bytes)).await?;

        let use_e2e = info.server_pubkey != [0u8; 32];

        use crate::zmq::global_context;

        if !use_e2e {
            anyhow::bail!(
                "Server did not provide Ristretto255 public key - E2E authentication required"
            );
        }

        let server_ristretto_pubkey = RistrettoPublic::from_bytes(&info.server_pubkey)
            .ok_or_else(|| anyhow::anyhow!("Invalid server Ristretto255 public key encoding"))?;

        let shared_secret = ristretto_dh(&client_secret, &server_ristretto_pubkey);

        let keys = derive_stream_keys(
            &shared_secret,
            &client_pubkey_bytes,
            &info.server_pubkey,
        )?;

        tracing::info!(
            stream_id = %info.stream_id,
            topic = %keys.topic,
            "Creating E2E authenticated stream handle"
        );

        StreamHandle::new(
            &global_context(),
            info.stream_id,
            &info.endpoint,
            keys.topic,
            *keys.mac_key,
            claims,
            self.gen.signing_key().clone(),
        )
    }

    /// Apply chat template — delegates to generated client
    pub async fn apply_chat_template(
        &self,
        messages: &[crate::runtime::template_engine::ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        use crate::services::generated::inference_client::ChatMessageData;
        let msg_data: Vec<ChatMessageData> = messages.iter().map(|m| ChatMessageData {
            role: m.role.clone(),
            content: m.content.clone(),
        }).collect();
        match self.gen.apply_chat_template(&msg_data, add_generation_prompt).await? {
            InferenceResponseVariant::ApplyChatTemplateResult(text) => Ok(text),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Create a new LoRA adapter
    pub async fn create_lora(&self, config: &LoRAConfig) -> Result<()> {
        match self.gen.create_lora(
            config.rank as u32, config.alpha, config.dropout, &config.target_modules,
        ).await? {
            InferenceResponseVariant::CreateLoraResult => Ok(()),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Load a LoRA adapter from file
    pub async fn load_lora(&self, path: &Path) -> Result<()> {
        match self.gen.load_lora(&path.to_string_lossy()).await? {
            InferenceResponseVariant::LoadLoraResult => Ok(()),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Save the current LoRA adapter to file
    pub async fn save_lora(&self, path: &str) -> Result<()> {
        match self.gen.save_lora(path).await? {
            InferenceResponseVariant::SaveLoraResult => Ok(()),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Unload the current LoRA adapter
    pub async fn unload_lora(&self) -> Result<()> {
        match self.gen.unload_lora().await? {
            InferenceResponseVariant::UnloadLoraResult => Ok(()),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Check if a LoRA adapter is loaded
    pub async fn has_lora(&self) -> Result<bool> {
        match self.gen.has_lora().await? {
            InferenceResponseVariant::HasLoraResult(has_lora) => Ok(has_lora),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Set the current session ID for KV cache management
    pub async fn set_session(&self, session_id: &str) -> Result<()> {
        match self.gen.set_session(session_id).await? {
            InferenceResponseVariant::SetSessionResult => Ok(()),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Clear the current session's KV cache
    pub async fn clear_session(&self) -> Result<()> {
        match self.gen.clear_session().await? {
            InferenceResponseVariant::ClearSessionResult => Ok(()),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Release a session's KV cache
    pub async fn release_session(&self, session_id: &str) -> Result<()> {
        match self.gen.release_session(session_id).await? {
            InferenceResponseVariant::ReleaseSessionResult => Ok(()),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Request service shutdown
    pub async fn shutdown(&self) -> Result<()> {
        match self.gen.shutdown().await? {
            InferenceResponseVariant::Success => Ok(()),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse stream started response — uses generated response parser
    fn parse_stream_started_response(response: &[u8]) -> Result<StreamStartedInfo> {
        use crate::services::generated::inference_client::InferenceClient;
        match InferenceClient::parse_response(response)? {
            InferenceResponseVariant::GenerateStreamResult { stream_id, endpoint, server_pubkey } => {
                Ok(StreamStartedInfo {
                    stream_id,
                    endpoint,
                    server_pubkey: server_pubkey.try_into().unwrap_or([0u8; 32]),
                })
            }
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse stream authorized response — uses generated response parser
    fn parse_stream_authorized_response(response: &[u8]) -> Result<String> {
        use crate::services::generated::inference_client::InferenceClient;
        match InferenceClient::parse_response(response)? {
            InferenceResponseVariant::StartStreamResult { stream_id, .. } => Ok(stream_id),
            InferenceResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
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
    /// Parse an InferencePayload message from StreamService
    ///
    /// Note: This parses the inner inference_capnp::InferencePayload.
    /// HMAC verification and ordering happen at the StreamBlock level
    /// using the outer streaming_capnp::StreamBlock wrapper.
    pub fn from_capnp(payload: inference_capnp::inference_payload::Reader) -> Result<Self> {
        use inference_capnp::inference_payload::Which;

        let stream_id = payload.get_stream_id()?.to_str()?.to_owned();
        tracing::trace!(stream_id = %stream_id, "Parsing inference payload");

        // Parse union variant
        match payload.which()? {
            Which::Token(token) => {
                let token = token?.to_str()?.to_owned();
                Ok(StreamChunkMessage::Chunk { text: token })
            }
            Which::Complete(complete) => {
                let complete = complete?;
                let finish_reason = match complete.get_finish_reason()? {
                    inference_capnp::FinishReason::MaxTokens => FinishReason::MaxTokens,
                    inference_capnp::FinishReason::StopToken => FinishReason::StopToken(String::new()),
                    inference_capnp::FinishReason::EndOfSequence => FinishReason::EndOfSequence,
                    inference_capnp::FinishReason::Error => FinishReason::Error(String::new()),
                    inference_capnp::FinishReason::Stop => FinishReason::Stop,
                };
                let stats = crate::runtime::GenerationStats {
                    tokens_generated: complete.get_tokens_generated() as usize,
                    finish_reason: Some(finish_reason),
                    generation_time_ms: complete.get_generation_time_ms(),
                    tokens_per_second: complete.get_tokens_per_second(),
                    quality_metrics: None,
                    prefill_tokens: 0,
                    prefill_time_ms: 0,
                    prefill_tokens_per_sec: 0.0,
                    inference_tokens: 0,
                    inference_time_ms: 0,
                    inference_tokens_per_sec: 0.0,
                    inference_tokens_per_sec_ema: 0.0,
                };
                Ok(StreamChunkMessage::Complete { stats })
            }
            Which::Error(error) => {
                let error = error?;
                let error_msg = error.get_message()?.to_str()?.to_owned();
                Ok(StreamChunkMessage::Error {
                    error: error_msg,
                })
            }
        }
    }

    /// Returns true if this is the last message (Complete or Error)
    pub fn is_last(&self) -> bool {
        matches!(self, StreamChunkMessage::Complete { .. } | StreamChunkMessage::Error { .. })
    }
}

/// Handle for receiving stream chunks from StreamService with E2E authentication.
///
/// Subscribes to StreamService using DH-derived topic and verifies HMAC chain.
///
/// # E2E Authentication Protocol
///
/// 1. Client generates ephemeral keypair, includes pubkey in request envelope
/// 2. Server generates ephemeral keypair, returns pubkey in StreamInfo
/// 3. Both derive: shared_secret = DH(my_secret, their_pubkey)
/// 4. Both derive: (topic, mac_key) = HKDF(shared_secret, pubkeys)
/// 5. Client subscribes to DH-derived topic (unpredictable, non-colliding)
/// 6. Client verifies HMAC chain on received chunks
pub struct StreamHandle {
    /// ZMQ SUB socket
    subscriber: zmq::Socket,
    /// Stream ID (for display/logging only, not used for routing)
    stream_id: String,
    /// DH-derived topic (64 hex chars) - used for ZMQ subscription
    topic: String,
    /// StreamVerifier for HMAC chain verification (new StreamBlock format)
    verifier: crate::services::rpc_types::StreamVerifier,
    /// Buffer for parsed payloads (StreamBlock can contain multiple)
    pending_payloads: std::collections::VecDeque<crate::services::rpc_types::StreamPayload>,
    /// Whether stream has completed
    stream_completed: bool,
    /// Claims for legacy JWT subscription (backwards compatibility)
    #[allow(dead_code)]
    claims: crate::auth::Claims,
    /// Signing key for legacy JWT subscription
    #[allow(dead_code)]
    signing_key: ed25519_dalek::SigningKey,
}

impl StreamHandle {
    /// Create a new E2E authenticated stream handle.
    ///
    /// # Arguments
    /// * `context` - ZMQ context
    /// * `stream_id` - Stream ID for display/logging
    /// * `endpoint` - StreamService SUB endpoint
    /// * `topic` - DH-derived topic for ZMQ subscription (from Ristretto255 key exchange)
    /// * `mac_key` - DH-derived HMAC key for chain verification
    /// * `claims` - User authorization claims
    /// * `signing_key` - Ed25519 signing key
    pub fn new(
        context: &zmq::Context,
        stream_id: String,
        endpoint: &str,
        topic: String,
        mac_key: [u8; 32],
        claims: crate::auth::Claims,
        signing_key: ed25519_dalek::SigningKey,
    ) -> Result<Self> {
        use crate::services::rpc_types::StreamVerifier;

        // Create SUB socket
        let subscriber = context.socket(zmq::SUB)?;

        // Connect to StreamService endpoint
        subscriber.connect(endpoint)?;

        // Subscribe to DH-derived topic (E2E authentication via Ristretto255)
        subscriber.set_subscribe(topic.as_bytes())?;

        tracing::info!(
            stream_id = %stream_id,
            topic = %topic,
            endpoint = %endpoint,
            "Subscribed to E2E authenticated stream (StreamBlock format)"
        );

        // Create verifier for HMAC chain verification
        let verifier = StreamVerifier::new(mac_key, topic.clone());

        Ok(Self {
            subscriber,
            stream_id,
            topic,
            verifier,
            pending_payloads: std::collections::VecDeque::new(),
            stream_completed: false,
            claims,
            signing_key,
        })
    }

    /// Receive next chunk (blocking) with HMAC verification.
    ///
    /// Returns `None` if the stream has ended (Complete or Error message received).
    ///
    /// # StreamBlock Format
    ///
    /// Messages are received as 3-frame multipart: [topic, capnp StreamBlock, 16-byte mac]
    /// The StreamBlock contains multiple payloads which are buffered and returned one at a time.
    ///
    /// # HMAC Verification
    ///
    /// Each block's HMAC is verified against the chain:
    /// - First block: HMAC(mac_key, topic || capnp)[..16]
    /// - Subsequent: HMAC(mac_key, prev_mac || capnp)[..16]
    ///
    /// If verification fails, returns an error.
    pub fn next_chunk(&mut self) -> Result<Option<StreamChunkMessage>> {
        use crate::services::rpc_types::StreamPayload;

        // Return buffered payloads first
        if let Some(payload) = self.pending_payloads.pop_front() {
            return Ok(Some(self.convert_payload(payload)));
        }

        // Stream completed, no more data
        if self.stream_completed {
            return Ok(None);
        }

        // Receive multipart message from StreamService
        // StreamBlock format: [topic, capnp, 16-byte mac]
        let msg = self.subscriber.recv_multipart(0)?;

        // Validate 3-frame StreamBlock format
        if msg.len() != 3 || msg[2].len() != 16 {
            bail!(
                "Invalid StreamBlock format: expected 3 frames with 16-byte MAC, got {} frames{}",
                msg.len(),
                if msg.len() >= 3 { format!(" with {}-byte MAC", msg[2].len()) } else { String::new() }
            );
        }

        // Verify and parse StreamBlock
        let payloads = self.verifier.verify(&msg)?;

        // Buffer all payloads
        for payload in payloads {
            // Check if this is a completion/error (marks end of stream)
            let is_terminal = matches!(
                payload,
                StreamPayload::Complete { .. } | StreamPayload::Error { .. }
            );
            if is_terminal {
                self.stream_completed = true;
            }
            self.pending_payloads.push_back(payload);
        }

        // Return first payload from buffer
        if let Some(payload) = self.pending_payloads.pop_front() {
            return Ok(Some(self.convert_payload(payload)));
        }

        // Empty block (shouldn't happen but handle gracefully)
        Ok(None)
    }

    /// Convert ParsedStreamPayload to StreamChunkMessage
    fn convert_payload(&self, payload: crate::services::rpc_types::StreamPayload) -> StreamChunkMessage {
        use crate::services::rpc_types::{InferenceStreamPayload, StreamPayloadExt};

        // Convert generic payload to inference-specific
        match payload.to_inference() {
            Ok(InferenceStreamPayload::Token(text)) => {
                StreamChunkMessage::Chunk { text }
            }
            Ok(InferenceStreamPayload::Error(message)) => {
                StreamChunkMessage::Error { error: message }
            }
            Ok(InferenceStreamPayload::Complete(stats)) => {
                // Build quality metrics if present
                let quality_metrics = if stats.perplexity.is_some() || stats.avg_entropy.is_some() {
                    Some(crate::runtime::generation_metrics::GenerationQualityMetrics {
                        perplexity: stats.perplexity.unwrap_or(0.0),
                        avg_entropy: stats.avg_entropy.unwrap_or(0.0),
                        ..Default::default()
                    })
                } else {
                    None
                };

                // Convert to GenerationStats with full metrics
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
                // If parsing fails, return as error
                StreamChunkMessage::Error { error: format!("Failed to parse payload: {e}") }
            }
        }
    }

    /// Get the stream ID (for display/logging)
    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    /// Get the DH-derived topic (for debugging)
    pub fn topic(&self) -> &str {
        &self.topic
    }
}

// Note: Default impl removed - InferenceZmqClient requires signing credentials
