//! RPC service infrastructure.
//!
//! Provides `RequestService`, `EnvelopeContext`, `ServiceHandle`, and `QuicLoopConfig`.
//!
//! # Envelope-Based Security
//!
//! All requests are wrapped in `SignedEnvelope` for authentication:
//! - `process_request` unwraps and verifies signatures before dispatching
//! - Handlers receive `EnvelopeContext` with verified identity
//! - Services use `ctx.subject()` for policy checks

use crate::prelude::*;
use crate::transport::TransportConfig;
use anyhow::Result;
use zeroize::Zeroizing;
use async_trait::async_trait;
// AtomicU64 and Ordering removed — were unused
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::warn;

/// Authorization callback for policy checks.
///
/// Parameters: (subject, resource, operation) -> allowed.
/// Services store this and call it from their `authorize()` handler method.
/// The concrete implementation typically wraps `PolicyClient::check_policy()`.
///
/// Returns a boxed future to support async policy checks on single-threaded runtimes.
pub type AuthorizeFn = Arc<dyn Fn(String, String, String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool>> + Send>> + Send + Sync>;

/// Decompose a composite JWT alg into its underlying component algs.
///
/// Used by the stripping defense in `verify_claims`: when a JWKS lists
/// multiple algs under one kid, the verifier needs to know that an
/// incoming alg like `ML-DSA-65-Ed25519` covers both `ML-DSA-65` and
/// `EdDSA`, so a composite-signed JWT is sufficient. Single algs return
/// just themselves.
fn composite_alg_components(alg: &str) -> Vec<&'static str> {
    match alg {
        "ML-DSA-65-Ed25519" => vec!["ML-DSA-65", "EdDSA", "ML-DSA-65-Ed25519"],
        "ML-DSA-65" => vec!["ML-DSA-65"],
        "EdDSA" => vec!["EdDSA"],
        "ES256" => vec!["ES256"],
        "RS256" => vec!["RS256"],
        _ => Vec::new(),
    }
}

/// Work to execute after the RPC response is sent (e.g., stream publishing).
pub type Continuation = std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>;

/// Context extracted from a verified SignedEnvelope.
///
/// Passed to service handlers after signature verification. Handlers use this for:
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
    /// authorization field. Used in `process_request` for inproc callers.
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

    /// Create a service-identity context for internal callbacks that bypass the envelope pipeline.
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

/// Trait for RPC services.
///
/// This is the unified trait for services that:
/// 1. Handle requests via `handle_request()`
/// 2. Are automatically spawnable (blanket `impl Spawnable for S: RequestService`)
#[async_trait(?Send)]
pub trait RequestService: 'static {
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

    /// Transport configuration (endpoint binding).
    fn transport(&self) -> &TransportConfig;

    /// Ed25519 signing key for signing responses.
    fn signing_key(&self) -> SigningKey;

    /// ML-DSA-65 signing key for the post-quantum half of the response COSE
    /// composite (#275). When `Some`, `process_request` signs the
    /// `ResponseEnvelope` under the Hybrid policy (EdDSA + ML-DSA-65); when
    /// `None` it signs Classical (EdDSA-only). The matching ML-DSA-65 public key
    /// must be anchored in the client's PQ trust store for Hybrid verification
    /// to succeed (peer attestation). Default `None` (Classical responses).
    ///
    /// Mirrors the request-side signing policy: the server emits the strongest
    /// composite it has keys for; a Classical verifier still accepts it via the
    /// inner EdDSA (skip-unknown interop).
    fn pq_signing_key(&self) -> Option<crate::crypto::pq::MlDsaSigningKey> {
        None
    }

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

    /// Whether to reject JWTs that lack a `cnf` key binding (jwk or jkt).
    ///
    /// When `true`, `verify_claims()` will reject any JWT that does not carry
    /// a `cnf` confirmation key, ensuring every authenticated request is
    /// cryptographically bound to its envelope signer.
    ///
    /// Default is `true` — override to `false` only for services that
    /// intentionally accept unbound JWTs (e.g., legacy compatibility).
    fn require_cnf_binding(&self) -> bool {
        true
    }

    /// JWT ID blocklist for access token revocation.
    ///
    /// When `Some`, `verify_claims()` rejects tokens whose `jti` appears
    /// in the blocklist. Override to provide a shared blocklist instance.
    fn jti_blocklist(&self) -> Option<&dyn crate::auth::JtiBlocklist> {
        None
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

        // Extract algorithm for routing
        let alg = crate::auth::header_alg(&token)
            .map_err(|e| anyhow::anyhow!("JWT header parse failed: {}", e))?
            .unwrap_or_default();

        // ── Stripping defense (composite signature hardening) ───────────────
        //
        // If the JWKS lists this kid with one or more `alg` values, the JWT's
        // header alg MUST exactly match a JWKS alg, AND when the JWKS lists
        // multiple algs for this kid, the JWT MUST use a composite alg that
        // covers all of them. This prevents an attacker from stripping the
        // post-quantum half of a composite signature and presenting only the
        // classical EdDSA half under the composite kid.
        if let Some(ref kid_str) = kid {
            let listed_algs = key_source.kid_algs(kid_str);
            if !listed_algs.is_empty() {
                if !listed_algs.iter().any(|a| a == &alg) {
                    tracing::warn!(
                        "JWT alg={} not listed in JWKS for kid={} (listed: {:?}) — possible stripping attack",
                        alg, kid_str, listed_algs
                    );
                    anyhow::bail!("JWT alg does not match JWKS for kid (stripping defense)");
                }
                // When the JWKS lists multiple algs for the kid (e.g. a
                // composite-signature publication that names both halves),
                // require ALL of them — i.e. require a composite alg whose
                // covered components include every listed alg.
                if listed_algs.len() > 1 {
                    let covered = composite_alg_components(&alg);
                    let all_covered = listed_algs.iter().all(|need| covered.iter().any(|c| c == need) || need == &alg);
                    if !all_covered {
                        tracing::warn!(
                            "JWT alg={} does not cover all JWKS-listed algs {:?} for kid={} — stripping rejected",
                            alg, listed_algs, kid_str
                        );
                        anyhow::bail!("JWT alg does not cover all JWKS-listed algs (stripping defense)");
                    }
                }
            }
        }

        // Route verification by algorithm
        let verified = match alg.as_str() {
            "ML-DSA-65" => {
                let vks = key_source.ml_dsa_verifying_keys();
                if vks.is_empty() {
                    anyhow::bail!("ML-DSA-65 JWT received but no PQ verifying keys available");
                }
                let aud = self.expected_audience();
                let mut last_err = None;
                let mut result = None;
                for vk in &vks {
                    match crate::auth::jwt::decode_ml_dsa_65(&token, vk, aud) {
                        Ok(claims) => { result = Some(claims); break; }
                        Err(e) => { last_err = Some(e); }
                    }
                }
                match result {
                    Some(claims) => claims,
                    None => {
                        tracing::warn!("ML-DSA-65 JWT verification failed (tried {} keys): {:?}", vks.len(), last_err);
                        anyhow::bail!("JWT verification failed");
                    }
                }
            }
            "ML-DSA-65-Ed25519" => {
                let vks = key_source.ml_dsa_verifying_keys();
                if vks.is_empty() {
                    anyhow::bail!("Composite JWT received but no PQ verifying keys available");
                }
                let verifying_key = key_source.get_key(&unverified.iss, kid.as_deref()).await.map_err(|e| {
                    tracing::warn!("JWT key resolution failed for iss={}: {}", unverified.iss, e);
                    anyhow::anyhow!("JWT key resolution failed")
                })?;
                let aud = self.expected_audience();
                let mut last_err = None;
                let mut result = None;
                for vk in &vks {
                    match crate::auth::jwt::decode_composite(&token, vk, &verifying_key, aud) {
                        Ok(claims) => { result = Some(claims); break; }
                        Err(e) => { last_err = Some(e); }
                    }
                }
                match result {
                    Some(claims) => claims,
                    None => {
                        tracing::warn!("Composite ML-DSA-65-Ed25519 JWT verification failed (tried {} keys): {:?}", vks.len(), last_err);
                        anyhow::bail!("JWT verification failed");
                    }
                }
            }
            _ => {
                // EdDSA (default path)
                let verifying_key = key_source.get_key(&unverified.iss, kid.as_deref()).await.map_err(|e| {
                    tracing::warn!("JWT key resolution failed for iss={}: {}", unverified.iss, e);
                    anyhow::anyhow!("JWT key resolution failed")
                })?;
                crate::auth::decode_with_key(&token, &verifying_key, self.expected_audience())
                    .map_err(|e| {
                        tracing::warn!("JWT verification failed: {}", e);
                        anyhow::anyhow!("JWT verification failed")
                    })?
            }
        };

        // Check jti against blocklist (revoked access tokens)
        if let Some(ref jti) = verified.jti {
            if let Some(blocklist) = self.jti_blocklist() {
                if blocklist.is_revoked(jti) {
                    tracing::warn!(jti = %jti, sub = %verified.sub, "Revoked JWT rejected");
                    anyhow::bail!("JWT has been revoked");
                }
            }
        }

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

/// REQ/REP message loop for RequestService.
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
/// QUIC server configuration for the service loop.
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

#[cfg(test)]
mod stripping_defense_tests {
    use super::composite_alg_components;

    /// stripping_composite_covers_components:
    /// `ML-DSA-65-Ed25519` (composite) MUST cover both component algs so a
    /// JWKS that lists both `EdDSA` and `ML-DSA-65` under one kid still
    /// accepts a composite-signed token.
    #[test]
    fn stripping_composite_covers_components() {
        let c = composite_alg_components("ML-DSA-65-Ed25519");
        assert!(c.contains(&"ML-DSA-65"));
        assert!(c.contains(&"EdDSA"));
        assert!(c.contains(&"ML-DSA-65-Ed25519"));
    }

    /// stripping_eddsa_does_not_cover_ml_dsa:
    /// Classical-only EdDSA must NOT cover ML-DSA-65; this is the core
    /// stripping defense — an attacker presenting an EdDSA JWT under a kid
    /// that the JWKS lists with ML-DSA-65 (or composite) must be rejected.
    #[test]
    fn stripping_eddsa_does_not_cover_ml_dsa() {
        let c = composite_alg_components("EdDSA");
        assert!(!c.contains(&"ML-DSA-65"));
        assert!(!c.contains(&"ML-DSA-65-Ed25519"));
    }

    /// stripping_unknown_alg_is_empty:
    /// Unknown algs return an empty component list so the verifier's
    /// "all-covered" check fails closed.
    #[test]
    fn stripping_unknown_alg_is_empty() {
        assert!(composite_alg_components("HS256").is_empty());
        assert!(composite_alg_components("none").is_empty());
    }
}
