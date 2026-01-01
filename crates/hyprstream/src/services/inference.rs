//! ZMQ-based inference service for text generation
//!
//! This service wraps TorchEngine and provides a ZMQ interface for inference operations.
//! It uses:
//! - REQ/REP for standard requests (generate, model_info, lora operations, etc.)
//! - PUB/SUB for streaming generation (via stream IDs)
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
//! Client                          Service
//!   │                               │
//!   │──── REQ: GenerateStream ─────►│
//!   │◄─── REP: { stream_id } ───────│
//!   │                               │
//!   │  SUB: inproc://inference/stream/{id}
//!   │◄──── PUB: chunk "Hello" ──────│
//!   │◄──── PUB: chunk " world" ─────│
//!   │◄──── PUB: StreamComplete ─────│
//! ```

use crate::auth::Operation;
use crate::services::PolicyZmqClient;
use crate::config::{FinishReason, GenerationRequest, GenerationResult, ModelInfo, TrainingMode};
use crate::inference_capnp;
use crate::lora::LoRAConfig;
use crate::runtime::kv_cache::CacheOwner;
use crate::runtime::model_config::ModelConfig;
use crate::runtime::{RuntimeConfig, RuntimeEngine, TorchEngine};
use crate::services::rpc_types::InferenceResponse;
use crate::services::EnvelopeContext;
use crate::training::{
    CheckpointConfig, CheckpointManager, ReplayBufferConfig, SelfSupervisedConfig,
    SelfSupervisedTrainer,
};
use crate::zmq::global_context;
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::serialize_message;
use capnp::serialize;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::mpsc;
use tokio::runtime::Handle;
use tokenizers::Tokenizer;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

// ============================================================================
// PolicyChecker - Thread-safe wrapper for policy checks from single-threaded runtime
// ============================================================================

/// Request sent to the policy checker thread
struct PolicyCheckRequest {
    subject: String,
    resource: String,
    operation: Operation,
    response_tx: mpsc::Sender<bool>,
}

/// Pending stream to be executed after REP response is sent.
///
/// This solves the streaming deadlock where the service waits for subscription
/// before returning the response, but the client can't subscribe without
/// the stream_id from the response.
struct PendingStream {
    request: GenerationRequest,
    stream_id: String,
    prompt: String, // For training collection (stored as String for simplicity)
}

/// Thread-safe policy checker that runs on a dedicated thread with its own runtime.
///
/// This allows InferenceService (single-threaded runtime) to call async PolicyZmqClient
/// without nesting runtimes. The PolicyChecker thread runs a multi-threaded runtime
/// where TMQ async I/O works properly.
#[derive(Clone)]
pub struct PolicyChecker {
    request_tx: mpsc::Sender<PolicyCheckRequest>,
}

impl PolicyChecker {
    /// Create a new PolicyChecker and start its background thread.
    ///
    /// The background thread runs a multi-threaded tokio runtime and processes
    /// policy check requests via synchronous channels.
    pub fn start(policy_client: PolicyZmqClient) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<PolicyCheckRequest>();

        std::thread::spawn(move || {
            // Create a multi-threaded runtime for this thread
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)  // Single worker is enough for policy checks
                .enable_all()
                .build()
                .expect("Failed to create policy checker runtime");

            rt.block_on(async move {
                // Process requests until the channel is closed
                while let Ok(req) = request_rx.recv() {
                    let allowed = policy_client
                        .check(&req.subject, &req.resource, req.operation)
                        .await
                        .unwrap_or(false);

                    // Send response (ignore errors if receiver dropped)
                    let _ = req.response_tx.send(allowed);
                }
            });
        });

        Self { request_tx }
    }

    /// Check if subject is allowed to perform operation on resource.
    ///
    /// This is synchronous and safe to call from any context (including
    /// inside `block_on`). Internally sends request to background thread.
    pub fn check(&self, subject: &str, resource: &str, operation: Operation) -> bool {
        let (response_tx, response_rx) = mpsc::channel();

        let request = PolicyCheckRequest {
            subject: subject.to_string(),
            resource: resource.to_string(),
            operation,
            response_tx,
        };

        // Send request to background thread
        if self.request_tx.send(request).is_err() {
            warn!("Policy checker thread died, denying request");
            return false;
        }

        // Wait for response (blocking, but background thread does async I/O)
        response_rx.recv().unwrap_or(false)
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
/// Runs on a dedicated thread for thread safety with tch-rs types.
/// Uses RefCell for interior mutability since it runs on a single thread.
///
/// # Security
///
/// All requests must be wrapped in `SignedEnvelope` and are verified before processing.
/// The service logs the identity for audit trails.
///
/// # Streaming Architecture
///
/// Uses a single XPUB socket for all streaming generation:
/// - Topic-based multiplexing: `stream-{id}` prefix on each message
/// - XPUB receives subscription events (knows when clients connect)
/// - No 10ms sleep hack - waits for subscription before streaming
/// - Clients use XSUB to subscribe to their stream topic
pub struct InferenceService {
    engine: RefCell<TorchEngine>,
    stream_id_counter: AtomicU64,
    /// Model identifier for events
    model_id: String,
    /// Model path for checkpoint management
    #[allow(dead_code)] // Future: checkpoint management
    model_path: PathBuf,
    /// Current session ID for events
    session_id: RefCell<Option<String>>,
    /// Runtime handle for async operations (reused instead of creating new runtimes)
    #[allow(dead_code)] // Reserved for future async operations
    runtime_handle: Handle,
    /// Single XPUB socket for all streaming (initialized in run_service_loop)
    xpub_socket: RefCell<Option<zmq::Socket>>,
    /// Server's Ed25519 verifying key for signature verification
    server_pubkey: VerifyingKey,
    /// Nonce cache for replay protection
    nonce_cache: Arc<InMemoryNonceCache>,
    /// Policy checker for authorization (runs on separate thread with its own runtime)
    policy_checker: PolicyChecker,
    /// Optional self-supervised trainer (initialized from config.json)
    trainer: Option<Arc<SelfSupervisedTrainer>>,
    /// Tokenizer for training example collection
    tokenizer: Option<Arc<Tokenizer>>,
    /// Unique stream endpoint for this service instance (each InferenceService has its own)
    stream_endpoint: String,
}

impl InferenceService {
    /// Start the inference service at the default endpoint
    pub async fn start(
        model_path: impl AsRef<Path>,
        config: RuntimeConfig,
        server_pubkey: VerifyingKey,
        policy_client: PolicyZmqClient,
    ) -> Result<crate::services::ServiceHandle> {
        Self::start_at(model_path, config, server_pubkey, policy_client, INFERENCE_ENDPOINT).await
    }

    /// Start the inference service at a specific endpoint
    pub async fn start_at(
        model_path: impl AsRef<Path>,
        config: RuntimeConfig,
        server_pubkey: VerifyingKey,
        policy_client: PolicyZmqClient,
        endpoint: &str,
    ) -> Result<crate::services::ServiceHandle> {
        let model_path = model_path.as_ref().to_path_buf();
        let endpoint_owned = endpoint.to_string();
        let model_id = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());
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
                match Self::initialize(model_path, config, model_id, server_pubkey, nonce_cache, policy_client).await {
                    Ok(service) => {
                        // Pass init_tx to run_service_loop - it signals AFTER socket binding
                        Self::run_service_loop(service, &endpoint_owned, Some(init_tx));
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
        Ok(crate::services::ServiceHandle::dummy())
    }

    /// Initialize the service
    async fn initialize(
        model_path: PathBuf,
        config: RuntimeConfig,
        model_id: String,
        server_pubkey: VerifyingKey,
        nonce_cache: Arc<InMemoryNonceCache>,
        policy_client: PolicyZmqClient,
    ) -> Result<Self> {
        // Capture runtime handle for reuse in handlers
        let runtime_handle = Handle::current();

        // Start PolicyChecker on a dedicated thread with its own runtime.
        // This allows synchronous policy checks from within our single-threaded runtime.
        let policy_checker = PolicyChecker::start(policy_client);

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

        // Only initialize trainer and tokenizer if training is enabled
        // This ensures zero memory/compute overhead when not training
        let (trainer, tokenizer): (Option<Arc<SelfSupervisedTrainer>>, Option<Arc<Tokenizer>>) = if let Some(ref tc) = training_config {
            if tc.is_enabled() && tc.mode == TrainingMode::SelfSupervised {
                info!(
                    "Self-supervised training enabled, target_adapter: {:?}",
                    tc.target_adapter
                );

                let ss_config = SelfSupervisedConfig {
                    learning_rate: tc.learning_rate,
                    batch_size: tc.batch_size,
                    min_buffer_size: tc.min_buffer_size,
                    steps_per_cycle: tc.steps_per_cycle,
                    ..Default::default()
                };
                let buffer_config = ReplayBufferConfig {
                    min_quality_threshold: tc.min_quality_threshold,
                    ..Default::default()
                };

                // Create trainer with optional checkpoint manager
                let mut trainer = SelfSupervisedTrainer::new(ss_config, buffer_config);

                if let Some(ref target_adapter) = tc.target_adapter {
                    // Create checkpoint manager for weight persistence
                    let checkpoint_config = CheckpointConfig {
                        max_checkpoints: 5,
                        git_commit_interval: tc.steps_per_cycle * 10, // Commit every 10 cycles
                        queue_size: 10,
                    };

                    match CheckpointManager::with_config(
                        model_path.clone(),
                        checkpoint_config,
                        None,
                    ) {
                        Ok(mgr) => {
                            let mgr = mgr.with_target_adapter(target_adapter.clone());
                            trainer = trainer.with_checkpoint_manager(mgr);
                            info!(
                                "Checkpoint manager initialized for adapter: {}",
                                target_adapter
                            );
                        }
                        Err(e) => {
                            warn!("Failed to create checkpoint manager: {}", e);
                        }
                    }
                }

                // Get tokenizer only when training is enabled
                let tokenizer = engine.get_tokenizer().ok().map(Arc::new);
                (Some(Arc::new(trainer)), tokenizer)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Generate unique stream endpoint for this service instance
        // Each InferenceService gets its own XPUB endpoint (UUID-based)
        let stream_endpoint = format!("inproc://hyprstream/inference/stream/{}", Uuid::new_v4());

        Ok(Self {
            engine: RefCell::new(engine),
            stream_id_counter: AtomicU64::new(1),
            model_id,
            model_path,
            session_id: RefCell::new(None),
            runtime_handle,
            xpub_socket: RefCell::new(None), // Initialized in run_service_loop
            server_pubkey,
            nonce_cache,
            policy_checker,
            trainer,
            tokenizer,
            stream_endpoint,
        })
    }

    /// Run the service loop (blocking)
    ///
    /// The `ready_tx` channel signals when sockets are bound and the service is ready.
    /// This ensures callers wait for actual readiness, not just initialization.
    fn run_service_loop(
        service: Self,
        endpoint: &str,
        ready_tx: Option<tokio::sync::oneshot::Sender<Result<()>>>,
    ) {
        let ctx = global_context();

        // Helper to signal error and return
        let signal_error = |tx: Option<tokio::sync::oneshot::Sender<Result<()>>>, err: anyhow::Error| {
            if let Some(tx) = tx {
                let _ = tx.send(Err(err));
            }
        };

        // Create REP socket for RPC
        let socket = match ctx.socket(zmq::REP) {
            Ok(s) => s,
            Err(e) => {
                let err = anyhow!("failed to create REP socket: {}", e);
                error!("{}", err);
                signal_error(ready_tx, err);
                return;
            }
        };

        // Set socket options
        if let Err(e) = socket.set_rcvtimeo(100) {
            warn!("failed to set receive timeout: {}", e);
        }

        // Bind to endpoint
        if let Err(e) = socket.bind(endpoint) {
            let err = anyhow!("failed to bind to {}: {}", endpoint, e);
            error!("{}", err);
            signal_error(ready_tx, err);
            return;
        }

        // Create single XPUB socket for all streaming
        // Topic-based multiplexing: clients subscribe to "stream-{id}"
        let xpub_socket = match ctx.socket(zmq::XPUB) {
            Ok(s) => s,
            Err(e) => {
                let err = anyhow!("failed to create XPUB socket: {}", e);
                error!("{}", err);
                signal_error(ready_tx, err);
                return;
            }
        };

        // Enable XPUB_VERBOSE to receive all subscription events
        // (not just unique ones) - useful for debugging
        if let Err(e) = xpub_socket.set_xpub_verbose(true) {
            warn!("failed to set XPUB_VERBOSE: {}", e);
        }

        // Non-blocking mode for checking subscriptions
        if let Err(e) = xpub_socket.set_rcvtimeo(0) {
            warn!("failed to set XPUB receive timeout: {}", e);
        }

        // Bind XPUB to this service's unique streaming endpoint
        if let Err(e) = xpub_socket.bind(&service.stream_endpoint) {
            let err = anyhow!("failed to bind XPUB to {}: {}", service.stream_endpoint, e);
            error!("{}", err);
            signal_error(ready_tx, err);
            return;
        }

        // Store XPUB socket in service for use by handle_generate_stream
        *service.xpub_socket.borrow_mut() = Some(xpub_socket);

        info!("inference service bound to {} (RPC) and {} (streaming)", endpoint, service.stream_endpoint);

        // Signal ready AFTER sockets are bound - this is the correct semantics
        if let Some(tx) = ready_tx {
            if tx.send(Ok(())).is_err() {
                warn!("Failed to signal service ready - receiver dropped");
            }
        }

        // Main service loop
        loop {
            match socket.recv_bytes(0) {
                Ok(request) => {
                    trace!("inference received request ({} bytes)", request.len());

                    // Unwrap and verify SignedEnvelope
                    let (ctx, payload) = match Self::unwrap_envelope(&request, &service.server_pubkey, &*service.nonce_cache) {
                        Ok((ctx, payload)) => (ctx, payload),
                        Err(e) => {
                            warn!("inference envelope verification failed: {}", e);
                            // Send error response
                            let response = InferenceResponse::error(0, "envelope verification failed");
                            if let Err(e) = socket.send(&response, 0) {
                                error!("failed to send error response: {}", e);
                            }
                            continue;
                        }
                    };

                    debug!(
                        "Inference request from {} (envelope_id={})",
                        ctx.casbin_subject(),
                        ctx.request_id
                    );

                    // Handle request - may return pending stream work
                    let (response, pending_stream) = match service.handle_request(&ctx, &payload) {
                        Ok((resp, pending)) => (resp, pending),
                        Err(e) => {
                            error!("inference request handling error: {}", e);
                            (InferenceResponse::error(0, &e.to_string()), None)
                        }
                    };

                    // Send response FIRST (before any streaming)
                    if let Err(e) = socket.send(&response, 0) {
                        error!("failed to send response: {}", e);
                    }

                    // THEN execute any pending stream (after response is sent)
                    // This solves the streaming deadlock - client can subscribe after
                    // receiving the stream_id in the response
                    if let Some(pending) = pending_stream {
                        service.execute_stream(pending);
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout, continue
                    continue;
                }
                Err(e) => {
                    warn!("inference recv error: {}", e);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }

    /// Unwrap and verify a SignedEnvelope from wire bytes.
    fn unwrap_envelope(
        request: &[u8],
        server_pubkey: &VerifyingKey,
        nonce_cache: &dyn NonceCache,
    ) -> Result<(EnvelopeContext, Vec<u8>)> {
        // Deserialize SignedEnvelope from Cap'n Proto
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(request),
            ReaderOptions::default(),
        )?;
        let signed_reader = reader.get_root::<hyprstream_rpc::common_capnp::signed_envelope::Reader>()?;
        let signed = SignedEnvelope::read_from(signed_reader)?;

        // Verify signature and replay protection
        signed.verify(server_pubkey, nonce_cache)?;

        // Extract context and payload
        let ctx = EnvelopeContext::from_verified(&signed);
        let payload = signed.payload().to_vec();

        Ok((ctx, payload))
    }

    /// Generate next stream ID
    fn next_stream_id(&self) -> String {
        let id = self.stream_id_counter.fetch_add(1, Ordering::Relaxed);
        format!("stream-{}", id)
    }

    /// Collect training example after generation if trainer is enabled
    fn collect_training_example(&self, prompt: &str, result: &GenerationResult) {
        let trainer = match &self.trainer {
            Some(t) => t.clone(),
            None => return,
        };

        let tokenizer = match &self.tokenizer {
            Some(t) => t.clone(),
            None => {
                trace!("No tokenizer available for training example collection");
                return;
            }
        };

        let quality_metrics = match &result.quality_metrics {
            Some(qm) => qm.clone(),
            None => {
                trace!("No quality metrics available for training example");
                return;
            }
        };

        // Tokenize prompt and response
        let tokenize = |text: &str| -> Result<Vec<i64>> {
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
            Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
        };

        match (tokenize(prompt), tokenize(&result.text)) {
            (Ok(prompt_tokens), Ok(response_tokens)) => {
                let session_id = self.session_id.borrow().clone();
                let prompt_len = prompt_tokens.len();
                let response_len = response_tokens.len();

                // Add example and potentially train (async)
                // Use futures::executor::block_on to avoid nesting tokio runtimes
                futures::executor::block_on(async {
                    trainer
                        .add_example(prompt_tokens, response_tokens, quality_metrics, session_id)
                        .await;

                    // Trigger training cycle if ready
                    if trainer.ready_to_train().await {
                        info!("Training buffer full, triggering training cycle...");
                        let engine = self.engine.borrow();
                        match trainer.train_cycle(&*engine).await {
                            Ok(cycle_result) => {
                                info!(
                                    "Training cycle complete: {} steps, mean_loss={:.4}, mean_reward={:.3}",
                                    cycle_result.steps, cycle_result.total_loss, cycle_result.mean_reward
                                );
                            }
                            Err(e) => warn!("Training cycle failed: {}", e),
                        }
                    }
                });

                trace!(
                    "Training example collected: prompt_len={}, response_len={}",
                    prompt_len,
                    response_len
                );
            }
            (Err(e), _) | (_, Err(e)) => {
                warn!("Failed to tokenize for training: {}", e);
            }
        }
    }

    /// Handle non-streaming generation
    fn handle_generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        // Save prompt for training collection
        let prompt = request.prompt.clone();

        // Use futures::executor::block_on because we're already inside a tokio runtime
        // (run_service_loop runs in rt.block_on), and tokio's block_on can't be nested.
        // futures::executor::block_on works because it's a simple single-threaded executor.
        let engine = self.engine.borrow();
        let result = futures::executor::block_on(async {
            RuntimeEngine::generate_with_params(&*engine, request).await
        });

        // Collect training example on success
        if let Ok(gen_result) = &result {
            self.collect_training_example(prompt.as_str(), gen_result);
        }

        result
    }

    /// Prepare for streaming generation - returns stream_id immediately.
    ///
    /// This is the first phase of streaming that runs BEFORE the REP response is sent.
    /// The actual streaming happens in `execute_stream` which runs AFTER the response.
    ///
    /// This solves the deadlock where:
    /// - Service waits for subscription before returning response
    /// - Client can't subscribe without the stream_id from response
    fn prepare_stream(&self, request: GenerationRequest) -> (String, PendingStream) {
        let stream_id = self.next_stream_id();
        let prompt = request.prompt.as_str().to_string(); // Convert to String for training

        let pending = PendingStream {
            request,
            stream_id: stream_id.clone(),
            prompt,
        };

        (stream_id, pending)
    }

    /// Execute streaming generation - called AFTER REP response is sent.
    ///
    /// Uses a single XPUB socket with topic-based multiplexing.
    /// The stream_id is used as the topic prefix for all messages.
    ///
    /// # Protocol (with deferred streaming)
    ///
    /// 1. Client calls generate_stream via REQ/REP
    /// 2. Service generates stream_id (prepare_stream)
    /// 3. Service sends REP response (stream_id, endpoint)
    /// 4. Client receives response and subscribes to topic "stream-{id}"
    /// 5. Service waits for subscription (execute_stream - this function)
    /// 6. Service generates tokens and streams via XPUB
    fn execute_stream(&self, pending: PendingStream) {
        use futures::StreamExt;

        let stream_id = pending.stream_id;
        let request = pending.request;
        let prompt = pending.prompt;
        let topic = stream_id.as_bytes().to_vec();

        // Get XPUB socket (created in run_service_loop)
        let xpub_guard = self.xpub_socket.borrow();
        let xpub = match xpub_guard.as_ref() {
            Some(socket) => socket,
            None => {
                error!("XPUB socket not initialized for streaming");
                return;
            }
        };

        // Wait for subscription event from client
        // XPUB receives: 0x01 + topic (subscribe) or 0x00 + topic (unsubscribe)
        // Timeout after 5 seconds if no subscriber
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(5);
        let mut subscribed = false;

        while start.elapsed() < timeout {
            match xpub.recv_bytes(zmq::DONTWAIT) {
                Ok(sub_msg) if !sub_msg.is_empty() => {
                    let is_subscribe = sub_msg[0] == 0x01;
                    let sub_topic = &sub_msg[1..];

                    if is_subscribe && sub_topic == topic.as_slice() {
                        trace!("client subscribed to stream {}", stream_id);
                        subscribed = true;
                        break;
                    }
                    // Other subscription event, continue waiting
                }
                Ok(_) => {} // Empty message, continue
                Err(zmq::Error::EAGAIN) => {
                    // No subscription yet, yield briefly
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(e) => {
                    warn!("XPUB recv error: {}", e);
                    break;
                }
            }
        }

        if !subscribed {
            warn!("no subscriber for stream {} after timeout", stream_id);
            // Continue anyway - client may have subscribed before we started checking
        }

        // Helper to send topic-prefixed message
        let send_with_topic = |data: &[u8]| -> Result<(), zmq::Error> {
            // ZMQ topic filtering uses prefix matching
            // Message format: topic_bytes + data_bytes
            let mut msg = Vec::with_capacity(topic.len() + data.len());
            msg.extend_from_slice(&topic);
            msg.extend_from_slice(data);
            xpub.send(&msg, 0)
        };

        // Run the stream
        let engine = self.engine.borrow();
        let stream_result = engine.generate(request);

        match stream_result {
            Ok(mut stream) => {
                let mut seq_num: u32 = 0;
                let mut accumulated_text = String::new();

                // Use futures::executor::block_on because we're already inside a tokio runtime
                // (run_service_loop runs in rt.block_on), and tokio's block_on can't be nested.
                futures::executor::block_on(async {
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(text) => {
                                // Accumulate text for training
                                accumulated_text.push_str(&text);

                                // Build and send chunk message with topic prefix
                                let chunk_bytes =
                                    InferenceResponse::stream_chunk(&stream_id, seq_num, &text);
                                if let Err(e) = send_with_topic(&chunk_bytes) {
                                    warn!("failed to send stream chunk: {}", e);
                                    break;
                                }
                                seq_num += 1;
                            }
                            Err(e) => {
                                let error_bytes =
                                    InferenceResponse::stream_error(&stream_id, seq_num, &e.to_string());
                                if let Err(send_err) = send_with_topic(&error_bytes) {
                                    tracing::error!("Failed to send stream error: {}", send_err);
                                }
                                break;
                            }
                        }
                    }

                    // Send completion
                    let stats = stream.stats();
                    let complete_bytes = InferenceResponse::stream_complete(&stream_id, seq_num, &stats);
                    if let Err(e) = send_with_topic(&complete_bytes) {
                        tracing::error!("Failed to send stream completion: {}", e);
                    }

                    // Collect training example from streaming generation
                    let gen_result = GenerationResult {
                        text: accumulated_text,
                        tokens_generated: stats.tokens_generated,
                        finish_reason: stats.finish_reason.unwrap_or(FinishReason::Stop),
                        generation_time_ms: stats.generation_time_ms,
                        tokens_per_second: stats.tokens_per_second,
                        quality_metrics: stats.quality_metrics,
                        // Prefill metrics
                        prefill_tokens: stats.prefill_tokens,
                        prefill_time_ms: stats.prefill_time_ms,
                        prefill_tokens_per_sec: stats.prefill_tokens_per_sec,
                        // Inference metrics
                        inference_tokens: stats.inference_tokens,
                        inference_time_ms: stats.inference_time_ms,
                        inference_tokens_per_sec: stats.inference_tokens_per_sec,
                    };
                    self.collect_training_example(&prompt, &gen_result);
                });
            }
            Err(e) => {
                let error_bytes = InferenceResponse::stream_error(&stream_id, 0, &e.to_string());
                if let Err(send_err) = send_with_topic(&error_bytes) {
                    tracing::error!("Failed to send initial stream error: {}", send_err);
                }
            }
        }
    }

    /// Handle model info request
    fn handle_model_info(&self) -> ModelInfo {
        RuntimeEngine::model_info(&*self.engine.borrow())
    }

    /// Handle is ready request
    fn handle_is_ready(&self) -> bool {
        self.engine.borrow().is_loaded()
    }

    /// Handle apply chat template
    fn handle_apply_chat_template(
        &self,
        messages: Vec<crate::runtime::template_engine::ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.engine
            .borrow()
            .apply_chat_template(&messages, add_generation_prompt)
    }

    /// Handle create LoRA
    fn handle_create_lora(&self, config: LoRAConfig) -> Result<()> {
        self.engine.borrow_mut().create_lora(config)
    }

    /// Handle load LoRA
    fn handle_load_lora(&self, path: &Path) -> Result<()> {
        // Use futures::executor::block_on to avoid nesting tokio runtimes
        let mut engine = self.engine.borrow_mut();
        futures::executor::block_on(async { engine.load_lora_from_file(path).await })
    }

    /// Handle save LoRA
    fn handle_save_lora(&self, path: &str) -> Result<()> {
        self.engine.borrow().save_lora(path)
    }

    /// Handle unload LoRA
    fn handle_unload_lora(&self) -> Result<()> {
        self.engine.borrow_mut().unload_lora()
    }

    /// Handle has LoRA
    fn handle_has_lora(&self) -> bool {
        self.engine.borrow().has_lora_model()
    }

    /// Handle set session
    fn handle_set_session(&self, session_id: String) -> Result<()> {
        // Track session ID for events
        *self.session_id.borrow_mut() = Some(session_id.clone());
        self.engine
            .borrow_mut()
            .set_session(CacheOwner::Session(session_id))
    }

    /// Handle clear session
    fn handle_clear_session(&self) {
        *self.session_id.borrow_mut() = None;
        self.engine.borrow_mut().clear_kv_cache();
    }

    /// Handle release session
    fn handle_release_session(&self, session_id: &str) -> Result<()> {
        self.engine
            .borrow_mut()
            .release_session(&CacheOwner::Session(session_id.to_string()))
    }

    /// Parse a generation request from capnp
    fn parse_generation_request(
        &self,
        reader: inference_capnp::generation_request::Reader,
    ) -> Result<GenerationRequest> {
        use crate::config::TemplatedPrompt;
        let prompt = TemplatedPrompt::new(reader.get_prompt()?.to_str()?.to_string());
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
            .filter_map(|s| s.ok().and_then(|t| t.to_str().ok().map(|s| s.to_string())))
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
    /// Uses PolicyChecker which runs on a separate thread with its own runtime.
    /// This is safe to call from any context (including inside block_on).
    fn check_auth(
        &self,
        ctx: &EnvelopeContext,
        request_id: u64,
        resource: &str,
        operation: Operation,
    ) -> Option<Vec<u8>> {
        let subject = ctx.casbin_subject();
        // PolicyChecker runs on a separate thread with its own runtime,
        // so this synchronous call is safe even from within our single-threaded runtime.
        // TMQ async I/O happens on the PolicyChecker thread.
        let allowed = self.policy_checker.check(&subject, resource, operation);

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
    fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<PendingStream>)> {
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
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Infer) {
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
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Infer) {
                    return Ok((resp, None));
                }
                let gen_req = gen_req?;
                let request = self.parse_generation_request(gen_req)?;

                // Prepare stream but don't execute yet - return pending work
                let (stream_id, pending) = self.prepare_stream(request);

                // Return response with stream_id - client will subscribe, then we execute
                let response = InferenceResponse::stream_started(request_id, &stream_id, &self.stream_endpoint);
                Ok((response, Some(pending)))
            }

            Which::ModelInfo(()) => {
                // Authorization: Query on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok((resp, None));
                }
                let info = self.handle_model_info();
                let has_lora = self.engine.borrow().has_lora_model();
                Ok((InferenceResponse::model_info(request_id, &info, has_lora), None))
            }

            Which::IsReady(()) => {
                // Authorization: Query on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok((resp, None));
                }
                let ready = self.handle_is_ready();
                Ok((InferenceResponse::ready(request_id, ready), None))
            }

            Which::ApplyChatTemplate(template_req) => {
                // Authorization: Query on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok((resp, None));
                }
                let template_req = template_req?;
                let messages: Vec<crate::runtime::template_engine::ChatMessage> = template_req
                    .get_messages()?
                    .iter()
                    .filter_map(|m| {
                        Some(crate::runtime::template_engine::ChatMessage {
                            role: m.get_role().ok()?.to_str().ok()?.to_string(),
                            content: m.get_content().ok()?.to_str().ok()?.to_string(),
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
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
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
                        .filter_map(|s| s.ok().and_then(|t| t.to_str().ok().map(|s| s.to_string())))
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
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
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
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
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
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok((resp, None));
                }
                match self.handle_unload_lora() {
                    Ok(()) => Ok((InferenceResponse::success(request_id), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::HasLora(()) => {
                // Authorization: Query on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok((resp, None));
                }
                let has_lora = self.handle_has_lora();
                Ok((InferenceResponse::has_lora(request_id, has_lora), None))
            }

            Which::SetSession(session_id) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok((resp, None));
                }
                let session_id = session_id?.to_str()?.to_string();
                match self.handle_set_session(session_id) {
                    Ok(()) => Ok((InferenceResponse::success(request_id), None)),
                    Err(e) => Ok((InferenceResponse::error(request_id, &e.to_string()), None)),
                }
            }

            Which::ClearSession(()) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok((resp, None));
                }
                self.handle_clear_session();
                Ok((InferenceResponse::success(request_id), None))
            }

            Which::ReleaseSession(session_id) => {
                // Authorization: Write on inference:{model}
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
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
                let model_loaded = self.engine.borrow().is_loaded();
                Ok((InferenceResponse::health(request_id, model_loaded), None))
            }

            Which::Shutdown(()) => {
                // Authorization: Manage on inference (shutdown requires admin)
                if let Some(resp) = self.check_auth(ctx, request_id, "inference", Operation::Manage) {
                    return Ok((resp, None));
                }
                info!("Inference service shutdown requested");
                Ok((InferenceResponse::success(request_id), None))
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
/// Uses `ZmqClient` internally for TMQ-based async transport with auto-signing.
#[derive(Clone)]
pub struct InferenceZmqClient {
    /// Unified ZMQ client (TMQ-based, auto-signing)
    client: std::sync::Arc<crate::services::ZmqClient>,
}

impl InferenceZmqClient {
    /// Create a new inference client with signing credentials
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self::with_endpoint(INFERENCE_ENDPOINT, signing_key, identity)
    }

    /// Create an inference client connected to a specific endpoint
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            client: std::sync::Arc::new(crate::services::ZmqClient::new(endpoint, signing_key, identity)),
        }
    }

    /// Get the next request ID (delegates to inner client)
    fn next_id(&self) -> u64 {
        self.client.next_id()
    }

    /// Convert capnp FinishReason to Rust enum
    fn parse_finish_reason(
        &self,
        reason: inference_capnp::FinishReason,
    ) -> FinishReason {
        match reason {
            inference_capnp::FinishReason::MaxTokens => FinishReason::MaxTokens,
            inference_capnp::FinishReason::StopToken => FinishReason::StopToken(String::new()),
            inference_capnp::FinishReason::EndOfSequence => FinishReason::EndOfSequence,
            inference_capnp::FinishReason::Error => FinishReason::Error(String::new()),
            inference_capnp::FinishReason::Stop => FinishReason::Stop,
        }
    }

    /// Generate text (non-streaming)
    pub async fn generate(&self, request: &GenerationRequest) -> Result<GenerationResult> {
        let id = self.next_id();

        let mut message = Builder::new_default();
        let mut req = message.init_root::<inference_capnp::inference_request::Builder>();
        req.set_id(id);

        let mut gen_req = req.init_generate();
        gen_req.set_prompt(request.prompt.as_str());
        gen_req.set_max_tokens(request.max_tokens as u32);
        gen_req.set_temperature(request.temperature);
        gen_req.set_top_p(request.top_p);
        gen_req.set_top_k(request.top_k.unwrap_or(0) as u32);
        gen_req.set_repeat_penalty(request.repeat_penalty);
        gen_req.set_repeat_last_n(request.repeat_last_n as u32);
        gen_req.set_seed(request.seed.unwrap_or(0));
        gen_req.set_timeout_ms(request.timeout.unwrap_or(0));

        // init_stop_tokens must be called last as it consumes the builder
        if !request.stop_tokens.is_empty() {
            let mut stop_list = gen_req.init_stop_tokens(request.stop_tokens.len() as u32);
            for (i, token) in request.stop_tokens.iter().enumerate() {
                stop_list.set(i as u32, token);
            }
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;

        let response = self.client.call(bytes).await?;
        self.parse_generation_result(&response)
    }

    /// Check if model is ready
    pub async fn is_ready(&self) -> Result<bool> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_is_ready(());
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_ready_response(&response)
    }

    /// Get model info
    pub async fn model_info(&self) -> Result<ModelInfo> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_model_info(());
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_model_info_response(&response)
    }

    /// Health check
    pub async fn health_check(&self) -> Result<()> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_health_check(());
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_health_response(&response)
    }

    /// Start streaming generation
    ///
    /// Returns the stream ID and endpoint for subscribing to chunks.
    pub async fn generate_stream(&self, request: &GenerationRequest) -> Result<(String, String)> {
        let id = self.next_id();

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

        let response = self.client.call(bytes).await?;
        self.parse_stream_started_response(&response)
    }

    /// Apply chat template to messages
    pub async fn apply_chat_template(
        &self,
        messages: &[crate::runtime::template_engine::ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let id = self.next_id();

        let mut message = Builder::new_default();
        let mut req = message.init_root::<inference_capnp::inference_request::Builder>();
        req.set_id(id);

        let mut template_req = req.init_apply_chat_template();
        template_req.set_add_generation_prompt(add_generation_prompt);
        let mut msg_list = template_req.init_messages(messages.len() as u32);
        for (i, msg) in messages.iter().enumerate() {
            let mut m = msg_list.reborrow().get(i as u32);
            m.set_role(&msg.role);
            m.set_content(&msg.content);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;

        let response = self.client.call(bytes).await?;
        self.parse_template_result_response(&response)
    }

    /// Create a new LoRA adapter
    pub async fn create_lora(&self, config: &LoRAConfig) -> Result<()> {
        let id = self.next_id();

        let mut message = Builder::new_default();
        let mut req = message.init_root::<inference_capnp::inference_request::Builder>();
        req.set_id(id);

        let mut lora_config = req.init_create_lora();
        lora_config.set_rank(config.rank as u32);
        lora_config.set_alpha(config.alpha);
        lora_config.set_dropout(config.dropout);
        let mut modules = lora_config.init_target_modules(config.target_modules.len() as u32);
        for (i, module) in config.target_modules.iter().enumerate() {
            modules.set(i as u32, module);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;

        let response = self.client.call(bytes).await?;
        self.parse_success_response(&response)
    }

    /// Load a LoRA adapter from file
    pub async fn load_lora(&self, path: &Path) -> Result<()> {
        let id = self.next_id();
        let path_str = path.to_string_lossy().to_string();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_load_lora(&path_str);
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_success_response(&response)
    }

    /// Save the current LoRA adapter to file
    pub async fn save_lora(&self, path: &str) -> Result<()> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_save_lora(path);
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_success_response(&response)
    }

    /// Unload the current LoRA adapter
    pub async fn unload_lora(&self) -> Result<()> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_unload_lora(());
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_success_response(&response)
    }

    /// Check if a LoRA adapter is loaded
    pub async fn has_lora(&self) -> Result<bool> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_has_lora(());
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_has_lora_response(&response)
    }

    /// Set the current session ID for KV cache management
    pub async fn set_session(&self, session_id: &str) -> Result<()> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_set_session(session_id);
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_success_response(&response)
    }

    /// Clear the current session's KV cache
    pub async fn clear_session(&self) -> Result<()> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_clear_session(());
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_success_response(&response)
    }

    /// Release a session's KV cache
    pub async fn release_session(&self, session_id: &str) -> Result<()> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_release_session(session_id);
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_success_response(&response)
    }

    /// Request service shutdown
    pub async fn shutdown(&self) -> Result<()> {
        let id = self.next_id();
        let bytes = serialize_message(|msg| {
            let mut req = msg.init_root::<inference_capnp::inference_request::Builder>();
            req.set_id(id);
            req.set_shutdown(());
        })?;
        let response = self.client.call(bytes).await?;
        self.parse_success_response(&response)
    }

    /// Parse generation result
    fn parse_generation_result(&self, response: &[u8]) -> Result<GenerationResult> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::GenerationResult(result) => {
                let result = result?;
                Ok(GenerationResult {
                    text: result.get_text()?.to_str()?.to_string(),
                    tokens_generated: result.get_tokens_generated() as usize,
                    finish_reason: self.parse_finish_reason(result.get_finish_reason()?),
                    generation_time_ms: result.get_generation_time_ms(),
                    tokens_per_second: result.get_tokens_per_second(),
                    quality_metrics: None, // TODO: Parse from capnp when schema is updated
                    // Prefill metrics
                    prefill_tokens: result.get_prefill_tokens() as usize,
                    prefill_time_ms: result.get_prefill_time_ms(),
                    prefill_tokens_per_sec: result.get_prefill_tokens_per_sec(),
                    // Inference metrics
                    inference_tokens: result.get_inference_tokens() as usize,
                    inference_time_ms: result.get_inference_time_ms(),
                    inference_tokens_per_sec: result.get_inference_tokens_per_sec(),
                })
            }
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse ready response
    fn parse_ready_response(&self, response: &[u8]) -> Result<bool> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::Ready(ready) => Ok(ready),
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse model info response
    fn parse_model_info_response(&self, response: &[u8]) -> Result<ModelInfo> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::ModelInfo(info) => {
                let info = info?;
                Ok(ModelInfo {
                    name: info.get_model_id()?.to_str()?.to_string(),
                    architecture: info.get_architecture()?.to_str()?.to_string(),
                    vocab_size: info.get_vocab_size() as usize,
                    hidden_size: info.get_hidden_size() as usize,
                    num_hidden_layers: Some(info.get_num_layers() as usize),
                    num_attention_heads: Some(info.get_num_heads() as usize),
                    context_length: info.get_max_sequence_length() as usize,
                    quantization: Some(info.get_quantization()?.to_str()?.to_string()),
                    parameters: 0,
                    intermediate_size: None,
                })
            }
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse health response
    fn parse_health_response(&self, response: &[u8]) -> Result<()> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::Health(_) => Ok(()),
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse stream started response
    fn parse_stream_started_response(&self, response: &[u8]) -> Result<(String, String)> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::StreamStarted(info) => {
                let info = info?;
                let stream_id = info.get_stream_id()?.to_str()?.to_string();
                let endpoint = info.get_endpoint()?.to_str()?.to_string();
                Ok((stream_id, endpoint))
            }
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse template result response
    fn parse_template_result_response(&self, response: &[u8]) -> Result<String> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::TemplateResult(result) => {
                Ok(result?.to_str()?.to_string())
            }
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse success response
    fn parse_success_response(&self, response: &[u8]) -> Result<()> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::Success(()) => Ok(()),
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse has LoRA response
    fn parse_has_lora_response(&self, response: &[u8]) -> Result<bool> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::HasLoraResult(has_lora) => Ok(has_lora),
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }
}

// Note: Default impl removed - InferenceZmqClient requires signing credentials
