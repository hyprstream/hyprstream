//! ZMQ service infrastructure
//!
//! Provides the foundation for REQ/REP services and clients using ZMQ.
//! Uses TMQ for async I/O - services run as async tasks with proper epoll integration.
//!
//! # Envelope-Based Security
//!
//! All requests are wrapped in `SignedEnvelope` for authentication:
//! - `RequestLoop` unwraps and verifies signatures before dispatching
//! - Handlers receive `EnvelopeContext` with verified identity
//! - Services use `ctx.subject()` for policy checks

use crate::prelude::*;
use crate::transport::TransportConfig;
use anyhow::{anyhow, Result};
use zeroize::Zeroizing;
use async_trait::async_trait;
// AtomicU64 and Ordering removed — were unused
use std::rc::Rc;
use std::sync::Arc;
use tmq::{FromZmqSocket, Multipart};
use tokio::sync::Notify;
use tracing::{debug, error, info, warn};

/// Authorization callback for policy checks.
///
/// Parameters: (subject, resource, operation) -> allowed.
/// Services store this and call it from their `authorize()` handler method.
/// The concrete implementation typically wraps `PolicyClient::check_policy()`.
///
/// Returns a boxed future to support async policy checks on single-threaded runtimes.
pub type AuthorizeFn = Arc<dyn Fn(String, String, String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool>> + Send>> + Send + Sync>;

/// Work to execute after the REP response is sent (e.g., stream publishing).
///
/// Built by streaming handlers, spawned by `RequestLoop` after the response
/// is sent to the client. This ensures the client has the `StreamInfo`
/// (with `stream_id`) before any data flows on the PUB/SUB channel.
pub type Continuation = std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>;

/// Context extracted from a verified SignedEnvelope.
///
/// This struct is passed to service handlers after the `RequestLoop`
/// has verified the envelope signature. Handlers use this for:
/// - Authorization checks via `subject()`
/// - Correlation via `request_id`
///
/// # Example
///
/// ```ignore
/// fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<Continuation>)> {
///     // Check authorization
///     let sub = ctx.subject().to_string();
///     if !policy_manager.check(&sub, "Model", "infer") {
///         return Err(anyhow!("unauthorized: {}", sub));
///     }
///     // Process request...
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EnvelopeContext {
    /// Unique request ID for correlation and logging
    pub request_id: u64,

    /// User claims decoded from jwt_token (or legacy claims field).
    /// Populated by `verify_claims()` after JWT signature verification.
    claims: Option<crate::auth::Claims>,

    /// Raw JWT token from the envelope. Server decodes and verifies this.
    /// Preferred over the legacy `claims` field when present.
    jwt_token: Option<String>,

    /// Authorization subject derived from the verified Ed25519 signer key.
    ///
    /// Set by `from_verified_as_system()` (FixedSigner path).
    /// `Anonymous` when AnySigner/WebTransport callers (identity from JWT/trust store).
    key_derived_subject: Subject,

    /// Authorization subject resolved from a verified JWT token.
    ///
    /// Set by `verify_claims()` after it determines whether the token is
    /// local (bare subject: `"alice"`) or federated (`"https://node-a:alice"`).
    /// `None` until `verify_claims()` runs, or when no JWT is present.
    pub(crate) jwt_subject: Option<Subject>,

    /// Ed25519 public key of the envelope signer (RFC 7800 confirmation key).
    /// Cryptographically verified by Ed25519 signature check.
    pub cnf: [u8; 32],

    /// WIMSE wth binding: SHA-256 of the WIT JWT from the envelope (if present).
    /// Populated from `RequestEnvelope.wit_hash` during context construction.
    pub(crate) envelope_wit_hash: Option<[u8; 32]>,

    /// Client's ephemeral DH public key for stream key derivation.
    /// Present on streaming requests; extracted from `RequestEnvelope.client_dh_public`.
    client_dh_public: Option<[u8; 32]>,
}

impl EnvelopeContext {
    /// Create context from a verified SignedEnvelope (AnySigner path).
    ///
    /// `key_derived_subject` is `Anonymous`. Use `from_verified_as_system()` for
    /// FixedSigner/inproc callers.
    ///
    /// `pub(crate)` — external callers should use the named constructors above
    /// to make the trust level explicit.
    pub(crate) fn from_verified(envelope: &SignedEnvelope) -> Self {
        Self {
            request_id: envelope.request_id(),
            claims: None,
            jwt_token: envelope.envelope.jwt_token().map(ToOwned::to_owned),
            key_derived_subject: Subject::anonymous(),
            jwt_subject: None,
            cnf: envelope.cnf,
            envelope_wit_hash: envelope.envelope.wth,
            client_dh_public: envelope.envelope.client_dh_public,
        }
    }

    /// Create context for a FixedSigner (inproc/IPC) caller.
    ///
    /// Sets `key_derived_subject = Subject::new("system")`, so `subject()` always
    /// returns `"system"` for this context regardless of any caller-asserted
    /// authorization field. Used in `process_request` for the ZMQ path.
    pub fn from_verified_as_system(envelope: &SignedEnvelope) -> Self {
        Self {
            request_id: envelope.request_id(),
            claims: None,
            jwt_token: envelope.envelope.jwt_token().map(ToOwned::to_owned),
            key_derived_subject: Subject::new("system"),
            jwt_subject: None,
            cnf: envelope.cnf,
            envelope_wit_hash: envelope.envelope.wth,
            client_dh_public: envelope.envelope.client_dh_public,
        }
    }

    /// Create a service-identity context for internal callbacks that bypass the ZMQ envelope pipeline.
    ///
    /// Used by services that make inproc self-calls without a real `SignedEnvelope`
    /// (e.g., `InferenceService` callback mode). Sets `key_derived_subject = "service:{name}"`
    /// so that `subject()` returns a proper service identity for authorization.
    ///
    /// `cnf` is zeroed because there is no real envelope; the service subject
    /// is asserted directly and is trusted because this constructor is only reachable
    /// from internal code paths that never cross a network boundary.
    pub fn from_callback_service(request_id: u64, service_name: &str) -> Self {
        Self {
            request_id,
            claims: None,
            jwt_token: None,
            key_derived_subject: Subject::new(format!("service:{service_name}")),
            jwt_subject: None,
            cnf: [0u8; 32],
            envelope_wit_hash: None,
            client_dh_public: None,
        }
    }

    /// Get the cryptographically-verified authorization subject.
    ///
    /// Resolution order:
    /// 1. Key-derived subject (from verified Ed25519 signer key via `KeyRegistry`).
    ///    For FixedSigner/inproc callers this is always `Subject::new("system")`.
    /// 2. JWT upgrade: if the envelope carries a verified JWT token, use its subject.
    ///    Federated JWTs (`iss` non-empty) produce `Subject::federated(iss, sub)`.
    /// 3. `Subject::anonymous()` — no verified identity.
    ///
    /// The caller-asserted envelope authorization is not preserved in the
    /// context — only verified state is available to handlers.
    pub fn subject(&self) -> Subject {
        // Prefer key-derived subject (cryptographically proven via signer key)
        if !self.key_derived_subject.is_anonymous() {
            return self.key_derived_subject.clone();
        }
        // JWT upgrade: use the pre-resolved subject set by verify_claims(), which
        // correctly distinguishes local tokens (bare sub) from federated ones
        // (iss:sub format).  Falls back to anonymous if no JWT was verified.
        if let Some(ref s) = self.jwt_subject {
            return s.clone();
        }
        Subject::anonymous()
    }

    /// Get the bare username string.
    pub fn user(&self) -> &str {
        // Resolution order mirrors subject(): prefer key-derived, then JWT, then anonymous.
        if !self.key_derived_subject.is_anonymous() {
            return self.key_derived_subject.name().unwrap_or("anonymous");
        }
        if let Some(ref s) = self.jwt_subject {
            return s.name().unwrap_or("anonymous");
        }
        "anonymous"
    }

    /// Check if the identity is authenticated (not anonymous).
    pub fn is_authenticated(&self) -> bool {
        !self.subject().is_anonymous()
    }

    /// Get user claims (if present, after verify_claims has run).
    pub fn claims(&self) -> Option<&crate::auth::Claims> {
        self.claims.as_ref()
    }

    /// Get the raw JWT token from the envelope (if present).
    pub fn jwt_token(&self) -> Option<&str> {
        self.jwt_token.as_deref()
    }

    /// Check if request has user context
    pub fn has_user_context(&self) -> bool {
        self.claims.is_some()
    }

    /// Get the client's ephemeral DH public key (if present).
    /// Used by streaming handlers to derive shared secrets for HMAC chain keys.
    pub fn ephemeral_pubkey(&self) -> Option<[u8; 32]> {
        self.client_dh_public
    }

}

/// Trait for ZMQ-based REQ/REP services.
///
/// This is the unified trait for services that:
/// 1. Handle requests via `handle_request()`
/// 2. Are automatically spawnable (blanket `impl Spawnable for S: ZmqService`)
///
/// Services include their infrastructure (context, transport, verifying key) so they
/// can be spawned directly without wrapping.
///
/// # Security Model
///
/// The `RequestLoop` unwraps and verifies `SignedEnvelope` before calling handlers.
/// Handlers receive `EnvelopeContext` with verified identity - no need to re-verify.
///
/// # Example
///
/// ```ignore
/// pub struct MyService {
///     // Business logic
///     data: MyData,
///     // Infrastructure (required for spawning)
///     context: Arc<zmq::Context>,
///     transport: TransportConfig,
///     verifying_key: VerifyingKey,
/// }
///
/// impl ZmqService for MyService {
///     fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<Continuation>)> {
///         // ctx.identity is already verified
///         info!("Request from {} (id={})", ctx.subject(), ctx.request_id);
///         Ok((vec![], None))
///     }
///
///     fn name(&self) -> &str { "my-service" }
///     fn context(&self) -> &Arc<zmq::Context> { &self.context }
///     fn transport(&self) -> &TransportConfig { &self.transport }
///     fn verifying_key(&self) -> VerifyingKey { self.verifying_key }
/// }
///
/// // MyService is now automatically Spawnable!
/// manager.spawn(Box::new(my_service)).await?;
/// ```
#[async_trait(?Send)]
pub trait ZmqService: 'static {
    /// Process a request and return a response with optional continuation.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Verified envelope context with identity
    /// * `payload` - Raw inner request bytes (Cap'n Proto encoded)
    ///
    /// Returns `(response_bytes, optional_continuation)`:
    /// - `response_bytes`: Cap'n Proto encoded response sent as REP
    /// - `continuation`: Optional future spawned AFTER the REP is sent
    ///   (used for streaming: ensures client has stream_id before data flows)
    ///
    /// Non-streaming services always return `None` for the continuation.
    async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<Continuation>)>;

    /// Service name (for logging and registry).
    fn name(&self) -> &str;

    /// ZMQ context for socket creation.
    fn context(&self) -> &Arc<zmq::Context>;

    /// Transport configuration (endpoint binding).
    fn transport(&self) -> &TransportConfig;

    /// Ed25519 signing key for signing responses.
    fn signing_key(&self) -> SigningKey;

    /// Ed25519 verifying key for envelope signature verification.
    fn verifying_key(&self) -> VerifyingKey {
        self.signing_key().verifying_key()
    }

    /// JWT key source for token verification.
    ///
    /// Returns the key source used to verify JWT signatures. Services must
    /// provide this to enable JWT verification. Returns `None` by default,
    /// which means JWT verification is skipped (anonymous access only).
    ///
    /// Most services should return a `ClusterKeySource` that trusts the
    /// cluster's CA key. PolicyService may return a `FederatedKeySource`
    /// for cross-cluster token exchange.
    fn jwt_key_source(&self) -> Option<std::sync::Arc<dyn crate::auth::JwtKeySource>> {
        None
    }

    /// Expected audience (resource URL) for JWT validation.
    ///
    /// When `Some`, `verify_claims()` rejects tokens whose `aud` claim doesn't match.
    /// Override this on services that should bind tokens to a specific resource.
    fn expected_audience(&self) -> Option<&str> {
        None
    }

    /// Whether to reject JWTs that lack a `cnf.jwk` key binding.
    ///
    /// When `true`, `verify_claims()` will reject any JWT that does not carry
    /// a `cnf` confirmation key, ensuring every authenticated request is
    /// cryptographically bound to its envelope signer. Default is `false`
    /// for backwards compatibility.
    fn require_cnf_binding(&self) -> bool {
        false
    }

    /// Resolve a signer key to an authorization subject via the trust store.
    ///
    /// Returns `Some(subject)` if the key is cached and not expired.
    /// Default implementation returns `None` (no trust store available).
    fn resolve_key_subject(&self, _signer_pubkey: &[u8; 32]) -> Option<crate::envelope::Subject> {
        None
    }

    /// Cache a verified key→subject binding in the trust store.
    ///
    /// Called after successful JWT verification when the JWT's `pub_key` matches
    /// the envelope signer. Default implementation is a no-op.
    fn cache_key_binding(
        &self,
        _verifying_key: ed25519_dalek::VerifyingKey,
        _subject: &str,
        _jwt: &str,
        _expires_at: i64,
    ) {
    }

    /// E2E JWT verification with unified key source.
    ///
    /// Called by `process_request` after envelope signature verification.
    /// Takes `&mut EnvelopeContext` to store the resolved `jwt_subject` directly,
    /// which correctly distinguishes local tokens (bare `sub`) from federated
    /// ones (`iss:sub` format) using the key source's `local_issuers()`.
    /// Async because federated key resolution may require an HTTP JWKS fetch.
    ///
    /// Prefers `jwt_token` (opaque token string) over legacy `claims` field.
    /// When `jwt_token` is present, the server decodes and verifies it directly.
    /// The legacy `claims` path is kept for backwards compat with older clients.
    async fn verify_claims(&self, ctx: &mut EnvelopeContext) -> anyhow::Result<()> {
        // Prefer jwt_token (new path) over legacy claims
        let token = ctx.jwt_token.clone()
            .or_else(|| ctx.claims().and_then(|c| c.token.clone()));

        let Some(token) = token else {
            // No JWT — try trust store lookup for cached key bindings
            if let Some(subject) = self.resolve_key_subject(&ctx.cnf) {
                ctx.key_derived_subject = subject;
                return Ok(());
            }
            // No JWT and no trust store entry — subject stays anonymous
            return Ok(());
        };

        // Get key source — if not configured, JWT verification is disabled
        let key_source = match self.jwt_key_source() {
            Some(ks) => ks,
            None => {
                tracing::warn!(
                    service = self.name(),
                    "JWT present but jwt_key_source() not configured — rejecting"
                );
                anyhow::bail!("JWT verification not configured for this service");
            }
        };

        // Decode the JWT to get claims for issuer routing
        let unverified = crate::auth::decode_unverified(&token)
            .map_err(|e| anyhow::anyhow!("JWT decode failed: {}", e))?;

        // Extract kid from JOSE header for key selection
        let kid = crate::auth::header_kid(&token)
            .map_err(|e| anyhow::anyhow!("JWT header parse failed: {}", e))?;

        // Check if issuer is trusted
        if !key_source.is_trusted(&unverified.iss) {
            tracing::warn!(
                "JWT from untrusted issuer rejected (iss={})",
                unverified.iss
            );
            anyhow::bail!("JWT issuer not trusted: {}", unverified.iss);
        }

        // Get verifying key from key source
        let verifying_key = key_source.get_key(&unverified.iss, kid.as_deref()).await.map_err(|e| {
            tracing::warn!(
                "JWT key resolution failed for iss={}: {}",
                unverified.iss, e
            );
            anyhow::anyhow!("JWT key resolution failed")
        })?;

        // Verify JWT signature
        let verified = crate::auth::decode_with_key(&token, &verifying_key, self.expected_audience())
            .map_err(|e| {
                tracing::warn!("JWT verification failed: {}", e);
                anyhow::anyhow!("JWT verification failed")
            })?;

        // Store verified claims on context for downstream use
        let local_issuers = key_source.local_issuers();
        let local_issuers_refs: Vec<&str> = local_issuers.iter().map(String::as_str).collect();
        let s = verified.subject(&local_issuers_refs);
        if !s.is_anonymous() {
            ctx.jwt_subject = Some(s);
        }
        ctx.claims = Some(verified.clone());

        // R2: Bind JWT cnf.jwk claim to envelope signer (WIMSE WIT key binding).
        // When a JWT carries a cnf.jwk (Ed25519 pubkey), the envelope signer must
        // match. Prevents a valid JWT holder from signing envelopes with a different
        // key and being attributed the JWT's subject identity.
        if let Some(ref claims) = ctx.claims {
            if let Some(expected) = claims.cnf_key_bytes() {
                use subtle::ConstantTimeEq as _;
                if expected.ct_ne(&ctx.cnf).into() {
                    tracing::warn!("JWT cnf.jwk mismatch: sub={}", claims.sub);
                    anyhow::bail!("JWT cnf.jwk does not match envelope signer");
                }

                // Cache the (key → subject) binding in the trust store.
                let vk = match ed25519_dalek::VerifyingKey::from_bytes(&expected) {
                    Ok(vk) => vk,
                    Err(_) => {
                        tracing::warn!("Invalid Ed25519 verifying key in JWT cnf.jwk");
                        anyhow::bail!("Invalid Ed25519 verifying key in JWT cnf.jwk");
                    }
                };
                let subject_str = claims.subject(&local_issuers_refs);
                if let Some(subject_name) = subject_str.name() {
                    self.cache_key_binding(vk, subject_name, &token, claims.exp);
                    tracing::info!(subject = %subject_name, "Cached key binding in trust store");
                }
            } else if let Some(jkt) = claims.cnf_jkt() {
                // R2b: DPoP path — JWT has cnf.jkt (thumbprint) instead of cnf.jwk.
                // Compute the JWK thumbprint of the envelope signer and compare.
                use subtle::ConstantTimeEq as _;
                let envelope_jkt = crate::auth::jwk_thumbprint(
                    &crate::auth::JwkThumbprintInput::Ed25519 { x: &ctx.cnf },
                );
                if envelope_jkt.as_bytes().ct_ne(jkt.as_bytes()).into() {
                    tracing::warn!("JWT cnf.jkt mismatch: sub={}", claims.sub);
                    anyhow::bail!("JWT cnf.jkt does not match envelope signer");
                }

                let vk = match ed25519_dalek::VerifyingKey::from_bytes(&ctx.cnf) {
                    Ok(vk) => vk,
                    Err(_) => {
                        tracing::warn!("Invalid Ed25519 verifying key in envelope cnf");
                        anyhow::bail!("Invalid Ed25519 verifying key in envelope cnf");
                    }
                };
                let subject_str = claims.subject(&local_issuers_refs);
                if let Some(subject_name) = subject_str.name() {
                    self.cache_key_binding(vk, subject_name, &token, claims.exp);
                    tracing::info!(subject = %subject_name, "Cached key binding from cnf.jkt");
                }
            } else if self.require_cnf_binding() {
                tracing::warn!("JWT missing cnf binding (jwk or jkt): sub={}", claims.sub);
                anyhow::bail!("JWT must include cnf binding for key binding (required by this service)");
            }
        }

        // R3: Verify WIMSE wth binding — envelope proof is committed to a specific WIT.
        // If the envelope carries a witHash, it must match SHA-256(jwtToken we just verified).
        if let Some(ref claimed_hash) = ctx.envelope_wit_hash {
            use sha2::{Digest, Sha256};
            use subtle::ConstantTimeEq as _;
            let expected: [u8; 32] = Sha256::digest(token.as_bytes()).into();
            if expected.ct_ne(claimed_hash.as_slice()).into() {
                tracing::warn!("witHash mismatch — possible rotation-window replay");
                anyhow::bail!("witHash does not match WIT — possible rotation-window replay");
            }
        }

        Ok(())
    }

    /// Build a generic error response payload for unexpected errors.
    ///
    /// Default implementation returns an empty vec (backwards compat).
    /// Generated services should override with schema-correct error payloads
    /// so clients receive proper `Error(ErrorInfo{...})` instead of parse failures.
    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        warn!(
            service = self.name(),
            request_id,
            error,
            "build_error_payload not overridden — sending empty error response"
        );
        vec![]
    }
}

/// REQ/REP message loop for ZmqService.
///
/// Runs the REQ/REP message loop as an async task with TMQ for I/O.
/// Uses proper async/await with epoll integration instead of blocking threads.
///
/// # Envelope Verification
///
/// The loop unwraps and verifies `SignedEnvelope` for every request:
/// 1. Deserialize `SignedEnvelope` from wire bytes
/// 2. Verify Ed25519 signature against server's public key
/// 3. Check nonce not replayed (replay protection)
/// 4. Extract `EnvelopeContext` and dispatch to handler
///
/// Invalid/unsigned requests are rejected with an error response.
///
/// # Usage
///
/// Typically used internally by the blanket `Spawnable` impl for `ZmqService`.
/// Direct usage is for advanced cases only.
/// REQ/REP message loop with ZMQ ROUTER + optional QUIC accept.
///
/// Runs a single service instance handling both ZMQ (via ROUTER socket) and
/// QUIC (via `WebTransportServer`) transports in a single `tokio::select!` loop.
///
/// # Motivation
///
/// Unlike `DualSpawnable` which runs transports on separate threads (requiring
/// `Send + Sync`), `RequestLoop` runs everything on a single `LocalSet`.
/// This allows `!Send` services like `ModelService` (with GPU tensors) to serve
/// both ZMQ and QUIC clients from one thread.
///
/// # ROUTER vs REP
///
/// Uses ROUTER socket instead of REP for ZMQ path:
/// - ROUTER is wire-compatible with existing REQ clients (REQ prepends identity + empty delimiter)
/// - ROUTER doesn't enforce strict send/recv alternation, enabling `select!` with QUIC
/// - Identity frames are stripped before processing and prepended before sending
///
/// # QUIC Path
///
/// Uses `WebTransportServer` for browser clients. WebTransport sessions carry
/// length-prefixed Cap'n Proto payloads (no ZMTP framing needed — both endpoints
/// are controlled code, not libzmq peers).
pub struct RequestLoop {
    /// ZMQ transport configuration
    transport: TransportConfig,
    /// ZMQ context for socket creation
    context: Arc<zmq::Context>,
    /// Server's Ed25519 verifying key
    server_pubkey: VerifyingKey,
    /// Server's Ed25519 signing key
    signing_key: SigningKey,
    /// Nonce cache for replay protection
    nonce_cache: Arc<InMemoryNonceCache>,
    /// Optional QUIC configuration (cert DER + key DER + bind addr)
    quic_config: Option<QuicLoopConfig>,
}

/// QUIC server configuration for `RequestLoop`.
pub struct QuicLoopConfig {
    /// DER-encoded certificate chain (leaf first, then intermediates/CA)
    pub cert_chain: Vec<Vec<u8>>,
    /// DER-encoded private key — zeroed on drop.
    pub key_der: Zeroizing<Vec<u8>>,
    /// Address to bind the WebTransport server
    pub bind_addr: std::net::SocketAddr,
    /// TLS server name (for endpoint discovery registration)
    pub server_name: String,
    /// Pre-serialized RFC 9728 JSON for HTTP/3 `.well-known/oauth-protected-resource`
    pub protected_resource_json: Option<Vec<u8>>,
    /// Callback invoked after QUIC binding succeeds, with (service_name, actual_addr, server_name).
    /// Used to announce endpoints to the DiscoveryService.
    pub on_quic_bound: Option<Box<dyn FnOnce(String, std::net::SocketAddr, String) + Send>>,
}

impl RequestLoop {
    /// Create a new unified request loop with ZMQ transport only.
    pub fn new(transport: TransportConfig, context: Arc<zmq::Context>, signing_key: SigningKey) -> Self {
        Self {
            transport,
            context,
            server_pubkey: signing_key.verifying_key(),
            signing_key,
            nonce_cache: Arc::new(InMemoryNonceCache::new()),
            quic_config: None,
        }
    }

    /// Enable QUIC (WebTransport) alongside ZMQ.
    pub fn with_quic(mut self, config: QuicLoopConfig) -> Self {
        self.quic_config = Some(config);
        self
    }

    /// Create with a shared nonce cache.
    pub fn with_nonce_cache(mut self, nonce_cache: Arc<InMemoryNonceCache>) -> Self {
        self.nonce_cache = nonce_cache;
        self
    }

    /// Run the unified service loop as an async task.
    ///
    /// Spawns on `LocalSet` via `spawn_local` — supports `!Send` services.
    pub async fn run<S: ZmqService>(self, service: S) -> Result<ServiceHandle> {
        let transport = self.transport.clone();
        let context = Arc::clone(&self.context);
        let server_pubkey = self.server_pubkey;
        let signing_key = self.signing_key.clone();
        let nonce_cache = Arc::clone(&self.nonce_cache);
        let quic_config = self.quic_config;
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = Arc::clone(&shutdown);

        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();

        let service = Rc::new(service);

        let handle = tokio::task::spawn_local(async move {
            if let Err(e) = Self::unified_loop(
                transport,
                context,
                service,
                server_pubkey,
                signing_key,
                nonce_cache,
                quic_config,
                shutdown_clone,
                Some(ready_tx),
            )
            .await
            {
                error!("Unified service loop error: {}", e);
            }
        });

        ready_rx
            .await
            .map_err(|_| anyhow!("Unified service task exited before signaling ready"))??;

        Ok(ServiceHandle {
            task: Some(handle),
            shutdown: Some(shutdown),
        })
    }

    /// Main unified select! loop.
    ///
    /// Accepts requests from both ZMQ ROUTER and WebTransport server,
    /// dispatching both through `process_request`.
    #[allow(clippy::too_many_arguments)]
    async fn unified_loop<S: ZmqService>(
        transport: TransportConfig,
        context: Arc<zmq::Context>,
        service: Rc<S>,
        server_pubkey: VerifyingKey,
        signing_key: SigningKey,
        nonce_cache: Arc<InMemoryNonceCache>,
        mut quic_config: Option<QuicLoopConfig>,
        shutdown: Arc<Notify>,
        ready_tx: Option<tokio::sync::oneshot::Sender<Result<()>>>,
    ) -> Result<()> {
        let endpoint = transport.zmq_endpoint();

        // Create raw ZMQ ROUTER socket
        let mut socket = match context.socket(zmq::ROUTER) {
            Ok(s) => s,
            Err(e) => {
                let err = anyhow!("failed to create ROUTER socket: {}", e);
                if let Some(tx) = ready_tx {
                    let _ = tx.send(Err(anyhow!("{}", err)));
                }
                return Err(err);
            }
        };
        socket.set_linger(0).ok();
        // ROUTER_MANDATORY: return errors instead of silently dropping messages
        // when routing identity is unknown
        socket.set_router_mandatory(true).ok();

        // Bind or connect
        let bind_result = match transport.bind_mode() {
            crate::transport::BindMode::Bind => transport.bind(&mut socket),
            crate::transport::BindMode::Connect => transport.connect(&mut socket),
        };
        if let Err(e) = bind_result {
            let err = anyhow!("failed to bind ROUTER to {}: {}", endpoint, e);
            if let Some(tx) = ready_tx {
                let _ = tx.send(Err(anyhow!("{}", err)));
            }
            return Err(err);
        }

        // Convert to TMQ Router for async I/O
        let router = match tmq::Router::from_zmq_socket(socket) {
            Ok(r) => r,
            Err(e) => {
                let err = anyhow!("failed to wrap ROUTER socket in TMQ: {}", e);
                if let Some(tx) = ready_tx {
                    let _ = tx.send(Err(anyhow!("{}", err)));
                }
                return Err(err);
            }
        };

        // Optionally create WebTransport server
        let wt_server = if let Some(ref mut qc) = quic_config {
            match crate::transport::zmtp_quic::WebTransportServer::bind(
                qc.bind_addr,
                qc.cert_chain.clone(),
                (*qc.key_der).clone(),
            ) {
                Ok(mut wts) => {
                    if let Some(ref meta) = qc.protected_resource_json {
                        wts = wts.with_protected_resource_metadata(meta.clone());
                    }
                    let actual_addr = wts.local_addr().unwrap_or(qc.bind_addr);
                    // Cert hash is computed from the leaf cert (first in chain)
                    let cert_hash = crate::transport::zmtp_quic::cert_hash(&qc.cert_chain[0]);
                    info!(
                        "{} QUIC/WebTransport bound to {} (cert hash: {})",
                        service.name(),
                        actual_addr,
                        cert_hash,
                    );
                    // Register QUIC endpoint in the global registry for discovery
                    if let Some(reg) = crate::registry::try_global() {
                        reg.register(
                            service.name(),
                            crate::registry::SocketKind::Quic,
                            crate::transport::TransportConfig::quic(actual_addr, &qc.server_name),
                            None,
                        );
                    }
                    // Announce to DiscoveryService (cross-process)
                    if let Some(cb) = qc.on_quic_bound.take() {
                        cb(service.name().to_owned(), actual_addr, qc.server_name.clone());
                    }
                    Some(wts)
                }
                Err(e) => {
                    warn!("{} failed to bind WebTransport: {}", service.name(), e);
                    None
                }
            }
        } else {
            None
        };

        let mode_str = match transport.bind_mode() {
            crate::transport::BindMode::Bind => "bound to",
            crate::transport::BindMode::Connect => "connected to",
        };
        info!(
            "{} unified service {} {} (ROUTER{})",
            service.name(),
            mode_str,
            endpoint,
            if wt_server.is_some() { " + WebTransport" } else { "" },
        );

        // Signal ready
        if let Some(tx) = ready_tx {
            if tx.send(Ok(())).is_err() {
                warn!("{} unified service ready signal dropped", service.name());
            }
        }

        // If WebTransport is available, spawn its accept loop as a separate local task
        // sharing the same service Rc
        if let Some(wts) = wt_server {
            let svc = Rc::clone(&service);
            let sk = signing_key.clone();
            let nc = Arc::clone(&nonce_cache);
            let sd = Arc::clone(&shutdown);
            tokio::task::spawn_local(async move {
                if let Err(e) = wts
                    .accept_loop_service(svc, server_pubkey, sk, nc, sd)
                    .await
                {
                    error!("WebTransport accept loop error: {}", e);
                }
            });
        }

        // Bounded semaphore for continuations
        const MAX_INFLIGHT_CONTINUATIONS: usize = 16;
        let continuation_semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_INFLIGHT_CONTINUATIONS));

        // Use futures traits for Router async I/O
        use futures::{SinkExt, StreamExt};
        let mut router_stream = router;

        loop {
            tokio::select! {
                biased;

                // Shutdown
                _ = shutdown.notified() => {
                    debug!("{} unified service received shutdown signal", service.name());
                    break;
                }

                // ZMQ ROUTER path
                Some(result) = router_stream.next() => {
                    let msg = match result {
                        Ok(msg) => msg,
                        Err(e) => {
                            error!("{} ROUTER recv error: {}", service.name(), e);
                            continue;
                        }
                    };

                    // Split identity envelope from payload.
                    // ROUTER multipart: [identity, empty_delimiter, ...payload_frames...]
                    // REQ clients automatically prepend identity + empty frame.
                    let frames: Vec<Vec<u8>> = msg.into_iter().map(|f| f.to_vec()).collect();

                    // Find the empty delimiter frame
                    let delimiter_pos = frames.iter().position(Vec::is_empty);
                    let (identity_frames, payload_frames) = match delimiter_pos {
                        Some(pos) => (&frames[..=pos], &frames[pos + 1..]),
                        None => {
                            warn!("{} ROUTER message missing empty delimiter, dropping", service.name());
                            continue;
                        }
                    };

                    // Concatenate payload frames
                    let request: Vec<u8> = payload_frames.iter().flat_map(|f| f.iter().copied()).collect();

                    if request.len() < 8 {
                        warn!("{} received undersized ROUTER message ({} bytes), dropping", service.name(), request.len());
                        // Send error response with identity envelope
                        let error_payload = service.build_error_payload(0, "malformed request: too small");
                        let response_bytes = Self::wrap_response(0, error_payload, &signing_key)?;
                        let mut response_msg = Multipart::default();
                        for frame in identity_frames {
                            response_msg.push_back(frame.as_slice().into());
                        }
                        response_msg.push_back(response_bytes.into());
                        if let Err(e) = router_stream.send(response_msg).await {
                            error!("{} ROUTER send error: {}", service.name(), e);
                        }
                        continue;
                    }

                    // Process through shared envelope pipeline.
                    // AnySigner: per-service keys mean each caller signs with its own key,
                    // not a shared root key. The envelope signature is still verified —
                    // JWT claims + Casbin handle authorization.
                    // subsecond::call wraps the dispatch so handler code can be hot-patched during dev.
                    let (response_bytes, continuation) = match subsecond::call(|| crate::transport::zmtp_quic::process_request(
                        &request,
                        &*service,
                        crate::transport::zmtp_quic::EnvelopeVerification::AnySigner,
                        &signing_key,
                        &nonce_cache,
                    )).await {
                        Ok(result) => result,
                        Err(e) => {
                            error!("{} request processing error: {}", service.name(), e);
                            let error_payload = service.build_error_payload(0, &e.to_string());
                            (Self::wrap_response(0, error_payload, &signing_key)?, None)
                        }
                    };

                    // Prepend identity frames and send response
                    let mut response_msg = Multipart::default();
                    for frame in identity_frames {
                        response_msg.push_back(frame.as_slice().into());
                    }
                    response_msg.push_back(response_bytes.into());

                    if let Err(e) = router_stream.send(response_msg).await {
                        error!("{} ROUTER send error: {}", service.name(), e);
                    }

                    // Spawn continuation after response is sent
                    if let Some(future) = continuation {
                        let sem = continuation_semaphore.clone();
                        tokio::task::spawn_local(async move {
                            let Ok(_permit) = sem.acquire_owned().await else {
                                warn!("continuation semaphore closed");
                                return;
                            };
                            future.await;
                        });
                    }
                }
            }
        }

        // Unregister QUIC endpoint on shutdown
        if quic_config.is_some() {
            if let Some(reg) = crate::registry::try_global() {
                reg.unregister(service.name(), crate::registry::SocketKind::Quic);
            }
        }

        info!("{} unified service stopped", service.name());
        Ok(())
    }

    /// Wrap a response payload in a signed ResponseEnvelope.
    fn wrap_response(
        request_id: u64,
        payload: Vec<u8>,
        signing_key: &SigningKey,
    ) -> Result<Vec<u8>> {
        use crate::envelope::ResponseEnvelope;
        use capnp::message::Builder;
        use capnp::serialize;

        let signed = ResponseEnvelope::new_signed(request_id, payload, signing_key);
        let mut message = Builder::new_default();
        let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
        signed.write_to(&mut builder);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }
}

/// Deprecated alias for `RequestLoop`.
///
/// The old `UnifiedRequestLoop` has been merged into `RequestLoop` which now
/// always uses ROUTER sockets (wire-compatible with REQ clients) and supports
/// optional QUIC via `.with_quic()`.
#[deprecated(note = "Use RequestLoop instead")]
pub type UnifiedRequestLoop = RequestLoop;

/// Handle for a running service
pub struct ServiceHandle {
    task: Option<tokio::task::JoinHandle<()>>,
    /// Shutdown signal using Notify (clean async shutdown)
    shutdown: Option<Arc<Notify>>,
}

impl ServiceHandle {
    /// Create a dummy handle for services that manage their own lifecycle
    pub fn dummy() -> Self {
        Self {
            task: None,
            shutdown: None,
        }
    }

    /// Create a handle from an existing task and shutdown signal.
    ///
    /// Used by ServiceSpawner when spawning Spawnable services.
    pub fn from_task(task: tokio::task::JoinHandle<()>, shutdown: Arc<Notify>) -> Self {
        Self {
            task: Some(task),
            shutdown: Some(shutdown),
        }
    }

    /// Stop the service gracefully
    ///
    /// Idempotent: subsequent calls are no-ops if already stopped.
    pub async fn stop(&mut self) {
        // Signal shutdown via Notify
        if let Some(shutdown) = &self.shutdown {
            shutdown.notify_one();
        }
        // Wait for task to complete
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }

    /// Check if the service is still running
    pub fn is_running(&self) -> bool {
        self.task.as_ref().map(|t| !t.is_finished()).unwrap_or(true)
    }
}
