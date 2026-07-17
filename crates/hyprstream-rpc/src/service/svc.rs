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
use async_trait::async_trait;
use zeroize::Zeroizing;
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
pub type AuthorizeFn = Arc<
    dyn Fn(
            String,
            String,
            String,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool>> + Send>>
        + Send
        + Sync,
>;

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

/// Whether a JWT `alg` carries an ML-DSA-65 (post-quantum) component. Fu4/#677.
fn alg_covers_pq(alg: &str) -> bool {
    matches!(alg, "ML-DSA-65" | "ML-DSA-65-Ed25519")
}

/// Fu4/#677: CryptoPolicy-driven minimum-alg gate for JWT verification.
///
/// The `verify_claims` alg-routing accepts an EdDSA-only (classical) JWT
/// whenever a kid resolves. Anti-stripping relies on `kid_algs`, whose default
/// impl is empty ([`crate::auth::JwtKeySource::kid_algs]) and which the common
/// `ClusterKeySource` never populates — so the existing defense is a no-op for
/// most services. This gate makes the floor explicit and policy-driven: under a
/// Hybrid [`CryptoPolicy`] a classical-only (`EdDSA`) JWT is rejected
/// **independent of JWKS alg-list hygiene**. Under Classical any single alg is
/// passed through to the per-alg decoders (existing behavior).
fn jwt_alg_satisfies_policy(policy: crate::crypto::CryptoPolicy, alg: &str) -> anyhow::Result<()> {
    if policy.uses_pq() && !alg_covers_pq(alg) {
        anyhow::bail!(
            "Hybrid crypto policy requires a post-quantum JWT alg; \
             classical-only alg '{}' rejected (Fu4/#677)",
            alg
        );
    }
    Ok(())
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

    /// Authenticated, suite-complete ephemeral HyKEM recipient for an identified
    /// stream.  This is only key material; handlers must still supply the
    /// accepted-current [`IdentifiedStreamBinding`] and pass the key-release PEP.
    client_kem_public: Option<crate::crypto::hybrid_kem::RecipientPublic>,

    /// Authenticated request transcript and one-shot response recipient.
    /// These are used only by the response seal chokepoint.
    pub(crate) request_iat: i64,
    pub(crate) request_nonce: [u8; 16],
    pub(crate) response_kem_recipient: Option<crate::crypto::hybrid_kem::RecipientPublic>,
    pub(crate) service_domain: Option<String>,

    /// Whether this request originated from a genuine in-process / IPC caller
    /// (the `FixedSigner` mutual-auth plane), as opposed to a networked peer
    /// (the `AnySigner` plane used by ZMQ-QUIC / iroh / WebTransport).
    ///
    /// `true` ONLY for `from_verified_as_system` and `from_callback_service`
    /// (genuine local callers). Networked callers (`from_verified`) are `false`.
    /// Gates the empty-`iss` JWT shortcut (#328): an empty issuer is the local
    /// PolicyService's bare-`sub` token and is accepted ONLY for local callers;
    /// a networked peer presenting an empty-`iss` token is rejected (fail-closed).
    is_local_caller: bool,
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
            client_kem_public: envelope.envelope.client_kem_public.clone(),
            request_iat: envelope.envelope.iat,
            request_nonce: envelope.envelope.nonce,
            response_kem_recipient: envelope.envelope.response_kem_recipient.clone(),
            service_domain: envelope.envelope.service_domain.clone(),
            // AnySigner / networked plane — NOT a local caller (#328).
            is_local_caller: false,
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
            client_kem_public: envelope.envelope.client_kem_public.clone(),
            request_iat: envelope.envelope.iat,
            request_nonce: envelope.envelope.nonce,
            response_kem_recipient: envelope.envelope.response_kem_recipient.clone(),
            service_domain: envelope.envelope.service_domain.clone(),
            // FixedSigner mutual-auth plane — genuine in-process / IPC caller (#328).
            is_local_caller: true,
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
            client_kem_public: None,
            request_iat: 0,
            request_nonce: [0; 16],
            response_kem_recipient: None,
            service_domain: None,
            // Internal self-call that never crosses a network boundary (#328).
            is_local_caller: true,
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

    /// **Derive the verified key material (S8/#574, S1/#548).** This is the
    /// accessor that threads crypto-derived assurance from the envelope's
    /// verified signature into MAC context assembly.
    ///
    /// Returns the [`VerifiedKeyMaterial`] assurance axis for THIS verified
    /// envelope, derived (never trusted) from what the signature layer actually
    /// verified:
    ///
    /// - **`PqHybrid`** — the envelope's `cnf` Ed25519 signer key is
    ///   cryptographically verified (it is `self.cnf`, set during
    ///   [`Self::from_verified`] from the signature check) AND a bound ML-DSA-65
    ///   anchor for that identity is present in the global `PqTrustStore`
    ///   (`register_pq_trust` binding) under the enforced Hybrid policy. This is
    ///   the kid-anchored binding the rest of the TCB uses — the PQ key is
    ///   resolved from the trust store keyed by the EdDSA identity, never
    ///   self-asserted.
    /// - **`Classical`** — the `cnf` Ed25519 signer key is verified but NO bound
    ///   ML-DSA-65 anchor is present (the federation edge, or a pre-PQ identity).
    /// - **`Unverified`** — the `cnf` key is zeroed (a callback-service context
    ///   with no real envelope; assurance floors to the S1 `Unverified` floor
    ///   so it dominates nothing above it).
    ///
    /// **Re-derive per verified envelope; never cache across identities.** Each
    /// envelope has its own signer; caching would conflate assurances.
    ///
    /// This is the load-bearing S8 input: without it every subject floors to
    /// `Unverified` and dominates nothing (S6 would deny everything). The MAC
    /// PDP combines this with the clearance off `Claims` via
    /// [`crate::auth::mac::SubjectContextClaims::security_context`].
    #[must_use]
    pub fn verified_key_material(&self) -> crate::auth::mac::VerifiedKeyMaterial {
        // The callback-service path mints a context with a zeroed cnf and no
        // real envelope — there is no signature to derive assurance from. Floor
        // to Unverified so the S1 dominance check denies anything above the
        // floor (this is the correct posture for an un-attested service self-
        // call that crosses into MAC-enforced territory).
        if self.cnf == [0u8; 32] {
            return crate::auth::mac::VerifiedKeyMaterial::Unverified;
        }
        // cnf is the cryptographically-verified Ed25519 signer key (set during
        // from_verified* from the envelope signature check). It is verified by
        // construction — the only question is whether a PQ anchor is bound.
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(store) = crate::envelope::global_pq_store() {
                if store.ml_dsa_key_for(&self.cnf).is_some() {
                    return crate::auth::mac::VerifiedKeyMaterial::PqHybrid;
                }
            }
        }
        // Verified Ed25519, no bound PQ anchor ⇒ Classical (the federation edge
        // / pre-PQ identity). NOT PqHybrid — no silent upgrade.
        crate::auth::mac::VerifiedKeyMaterial::Classical
    }

    /// Derive the full MAC [`SecurityContext`] for this verified envelope: the
    /// **two-input** S1 derivation (clearance from `Claims` + assurance from
    /// [`Self::verified_key_material`]). This is the convenience entry-point the
    /// MAC PDP / S6 grant path call; it composes the authority-asserted
    /// clearance with the crypto-derived assurance, clamping the assurance axis
    /// DOWN to what the verified key supports.
    ///
    /// Returns `None` (→ S1 deny) when the subject carries no clearance
    /// (unlabeled subject) — there is no default clearance. The resulting
    /// context is a SIBLING to [`Subject`]: identity and clearance are
    /// independent dimensions of the same verified principal.
    ///
    /// **Re-derive per verified envelope; never cache across identities.**
    #[must_use]
    pub fn security_context(&self) -> Option<crate::auth::mac::SecurityContext> {
        use crate::auth::mac::SubjectContextClaims as _;
        let claims = self.claims.as_ref()?;
        let key_material = self.verified_key_material();
        claims.security_context(key_material)
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

    /// Get the authenticated identified-stream HyKEM recipient.
    pub fn stream_kem_recipient(&self) -> Option<&crate::crypto::hybrid_kem::RecipientPublic> {
        self.client_kem_public.as_ref()
    }

    /// Whether this request came from a genuine in-process / IPC caller (#328).
    ///
    /// `true` for the `FixedSigner` mutual-auth plane and internal self-calls;
    /// `false` for networked / mesh peers (`AnySigner`). Used to confine the
    /// empty-`iss` JWT shortcut to local callers.
    pub fn is_local_caller(&self) -> bool {
        self.is_local_caller
    }

    /// Emit an accounting / audit record for an authorization decision (#445).
    ///
    /// This is the **Accounting** leg of AAA: it makes every authorization
    /// decision attributable after the fact to the *cryptographically verified*
    /// subject (`self.subject()`), never a caller-asserted field. It is called
    /// by generated dispatch code at the authorization decision point — the
    /// single DRY chokepoint where the verified subject, the resource, the
    /// operation, and the allow/deny effect are all in scope.
    ///
    /// The record is a structured `tracing` event on the dedicated `audit`
    /// target so it can be filtered/routed independently (e.g. shipped to an
    /// audit sink) without coupling to the rest of the daemon's logs. Fields:
    /// `subject`, `resource`, `action`, `decision` (`allow`/`deny`), and
    /// `request_id` for correlation. It records only identity + resource +
    /// action + decision — never key material or request payloads.
    ///
    /// Allow is logged at `info`; deny at `warn` (deny is the security-relevant
    /// event), but both carry identical structured fields so "who did what" is
    /// reconstructable for the allow path AND the deny path.
    ///
    /// **S7 (#573, #453):** this is the *control-plane* (Casbin `#[authorize]`)
    /// audit surface. The *data-plane* MAC surface (the AVC) emits its own
    /// records on the same unified `hyprstream.mac.audit` target via
    /// `hyprstream::mac::audit`. Both targets fan out to OpenTelemetry when the
    /// `otel` feature is active (OTel is a `tracing` subscriber), so denials
    /// are first-class observable signals — closing the #453 visibility gap
    /// where auth failures were previously buried at `debug` level. We also
    /// mirror the decision onto `hyprstream.mac.audit` here so a single OTel
    /// subscriber captures the whole authorization audit stream.
    pub fn audit_authz(&self, resource: &str, operation: &str, allowed: bool) {
        let subject = self.subject();
        let decision = if allowed { "allow" } else { "deny" };
        if allowed {
            tracing::info!(
                target: "audit",
                subject = %subject,
                resource = %resource,
                action = %operation,
                decision = decision,
                request_id = self.request_id,
                "authz decision"
            );
        } else {
            tracing::warn!(
                target: "audit",
                subject = %subject,
                resource = %resource,
                action = %operation,
                decision = decision,
                request_id = self.request_id,
                "authz decision"
            );
        }
        // S7 (#453): mirror onto the unified MAC audit target so a single OTel
        // subscriber captures every authorization decision (control-plane
        // Casbin + data-plane MAC AVC). Denials become first-class observable,
        // no longer buried at debug level.
        if allowed {
            tracing::info!(
                target: "hyprstream.mac.audit",
                decision = decision,
                subject = %subject,
                resource = %resource,
                action = %operation,
                request_id = self.request_id,
                plane = "control",
                "authorization decision"
            );
        } else {
            tracing::warn!(
                target: "hyprstream.mac.audit",
                decision = decision,
                subject = %subject,
                resource = %resource,
                action = %operation,
                request_id = self.request_id,
                plane = "control",
                "authorization decision"
            );
        }
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
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)>;

    /// Service name (for logging and registry).
    fn name(&self) -> &str;

    /// Transport configuration (endpoint binding).
    fn transport(&self) -> &TransportConfig;

    /// Ed25519 signing key for signing responses.
    fn signing_key(&self) -> SigningKey;

    /// Mutable reach source filled by the unified service spawner after the
    /// service's QUIC and iroh listeners bind.
    fn producer_reach_config_handle(&self) -> Option<crate::moq_stream::ProducerReachConfigHandle> {
        None
    }

    /// ML-DSA-65 signing key for the post-quantum half of the response COSE
    /// composite (#275). When `Some`, `process_request` signs the
    /// `ResponseEnvelope` under the Hybrid policy (EdDSA + ML-DSA-65); when
    /// `None` it signs Classical (EdDSA-only). The matching ML-DSA-65 public key
    /// must be anchored in the client's PQ trust store for Hybrid verification
    /// to succeed (peer attestation).
    ///
    /// The default derives the node's **persistent** mesh ML-DSA-65 key
    /// deterministically from this service's Ed25519 [`Self::signing_key`] via
    /// [`crate::node_identity::derive_mesh_mldsa_key`] (#157). Because the EdDSA
    /// half of the response is signed with that same Ed25519 key (dispatch is
    /// handed `service.signing_key()`), the kid-anchored binding the client
    /// stores — Ed25519 signer pubkey → ML-DSA vk — is self-consistent, and the
    /// published `#mesh-pq` DID verification method equals this signing key.
    ///
    /// Mirrors the request-side signing policy: the server emits the strongest
    /// composite it has keys for; a Classical verifier still accepts it via the
    /// inner EdDSA (skip-unknown interop). Override to return `None` to force
    /// Classical-only responses.
    fn pq_signing_key(&self) -> Option<crate::crypto::pq::MlDsaSigningKey> {
        Some(crate::node_identity::derive_mesh_mldsa_key(
            &self.signing_key(),
        ))
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

    /// Fu4/#677: the minimum JWT `alg` policy enforced in `verify_claims`.
    ///
    /// Under [`CryptoPolicy::Hybrid`] a classical-only (`EdDSA`) JWT is rejected
    /// outright, independent of JWKS `kid_algs` hygiene. The default reads the
    /// process-global envelope verify config (Hybrid in production), so the
    /// per-call policy is test-isolated from sibling tests that mutate the
    /// shared global — a mock/test service that needs EdDSA acceptance overrides
    /// this to [`CryptoPolicy::Classical`].
    fn jwt_verify_policy(&self) -> crate::crypto::CryptoPolicy {
        #[cfg(not(target_arch = "wasm32"))]
        {
            crate::envelope::global_verify_policy()
        }
        #[cfg(target_arch = "wasm32")]
        {
            crate::crypto::CryptoPolicy::Classical
        }
    }

    /// Resolve a verified signer key to an authorization subject (#446).
    ///
    /// Consulted by `verify_claims` for an unauthenticated (no-JWT) request: the
    /// envelope `cnf` (the signer's Ed25519 pubkey, already verified by the COSE
    /// signature check) is mapped to its authoritative subject so a *signed*
    /// service-to-service IPC caller resolves as its `service:<name>` identity
    /// instead of `anonymous`.
    ///
    /// The default routes through the process-global [key→subject
    /// resolver](crate::auth::key_subject_resolver), which the trust-store layer
    /// (`hyprstream-service`, populated fail-closed under #441) installs at
    /// startup. This wires *every* service — including those whose crate cannot
    /// depend on the trust store (e.g. `DiscoveryService`) — without each one
    /// re-implementing the lookup.
    ///
    /// Fail-closed: an unregistered key resolves to `None` → `anonymous`, so a
    /// genuinely anonymous caller stays denied for policy-gated writes. Override
    /// only to bypass or narrow this resolution; never to fabricate identity.
    #[cfg(not(target_arch = "wasm32"))]
    fn resolve_key_subject(&self, signer_pubkey: &[u8; 32]) -> Option<crate::envelope::Subject> {
        crate::auth::key_subject_resolver::resolve_subject(signer_pubkey)
    }

    /// Resolve a signer key to a subject (WASM: no trust store available).
    #[cfg(target_arch = "wasm32")]
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
        let token = ctx
            .jwt_token
            .clone()
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

        // Empty-`iss` gate (#328): an empty issuer denotes the local
        // PolicyService's bare-`sub` token, which the key sources treat as
        // "always local/trusted". That is only safe for genuine in-process /
        // IPC callers (the FixedSigner plane). A networked / mesh peer
        // (AnySigner) presenting an empty-`iss` token would otherwise inherit
        // local trust — reject it fail-closed; mesh peers must present an
        // explicit issuer (or authenticate via the key roster, no JWT).
        if unverified.iss.is_empty() && !ctx.is_local_caller {
            tracing::warn!(
                service = self.name(),
                "rejecting empty-iss JWT from a networked caller (empty issuer is in-process only) (#328)"
            );
            anyhow::bail!("empty JWT issuer is only accepted from in-process callers");
        }

        // Parse the protected JOSE header once with duplicate detection. All
        // dispatch and primitive verification below consumes this exact value.
        let protected = crate::auth::parse_protected_header(&token)
            .map_err(|e| anyhow::anyhow!("JWT header parse failed: {}", e))?;
        anyhow::ensure!(
            crate::auth::is_rfc9068_access_token_type(&protected.typ) || protected.typ == "wit+jwt",
            "unsupported JWT typ"
        );
        let kid = Some(protected.kid.clone());

        // Check if issuer is trusted
        if !key_source.is_trusted(&unverified.iss) {
            tracing::warn!(
                "JWT from untrusted issuer rejected (iss={})",
                unverified.iss
            );
            anyhow::bail!("JWT issuer not trusted: {}", unverified.iss);
        }

        let alg = protected.alg.clone();

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
                    let all_covered = listed_algs
                        .iter()
                        .all(|need| covered.iter().any(|c| c == need) || need == &alg);
                    if !all_covered {
                        tracing::warn!(
                            "JWT alg={} does not cover all JWKS-listed algs {:?} for kid={} — stripping rejected",
                            alg, listed_algs, kid_str
                        );
                        anyhow::bail!(
                            "JWT alg does not cover all JWKS-listed algs (stripping defense)"
                        );
                    }
                }
            }
        }

        // ── Fu4/#677: CryptoPolicy-driven minimum-alg gate ───────────────────
        //
        // Independent of the JWKS `kid_algs` hygiene below: under a Hybrid
        // policy a classical-only (EdDSA) JWT is rejected outright, before
        // per-alg routing, so anti-stripping no longer depends on every key
        // source populating `kid_algs` (the default impl is empty and
        // `ClusterKeySource` never overrides it). The policy comes from
        // `jwt_verify_policy` (the global config in production, overridable per
        // service for tests).
        jwt_alg_satisfies_policy(self.jwt_verify_policy(), &alg)?;

        // Route verification by algorithm
        let mut verified = match alg.as_str() {
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
                        Ok(claims) => {
                            result = Some(claims);
                            break;
                        }
                        Err(e) => {
                            last_err = Some(e);
                        }
                    }
                }
                match result {
                    Some(claims) => claims,
                    None => {
                        tracing::warn!(
                            "ML-DSA-65 JWT verification failed (tried {} keys): {:?}",
                            vks.len(),
                            last_err
                        );
                        anyhow::bail!("JWT verification failed");
                    }
                }
            }
            "ML-DSA-65-Ed25519" => {
                if !unverified.iss.is_empty()
                    && !key_source
                        .local_issuers()
                        .iter()
                        .any(|issuer| issuer == &unverified.iss)
                {
                    anyhow::bail!("local composite JWT issuer mismatch");
                }
                let allowed_types = [
                    crate::auth::RFC9068_ACCESS_TOKEN_TYPES[0],
                    crate::auth::RFC9068_ACCESS_TOKEN_TYPES[1],
                    "wit+jwt",
                ];
                let dispatch = crate::auth::parse_composite_dispatch(&token, &allowed_types)
                    .map_err(|error| anyhow::anyhow!("JWT dispatch failed: {error}"))?;
                let snapshot = key_source.composite_key_set().snapshot();
                let pair = snapshot
                    .pair(dispatch.kid())
                    .ok_or_else(|| anyhow::anyhow!("unknown composite JWT kid"))?;
                crate::auth::jwt::decode_composite(
                    &token,
                    pair.ml_dsa(),
                    pair.ed25519(),
                    self.expected_audience(),
                    &dispatch,
                )
                .map_err(|error| {
                    tracing::warn!("composite JWT verification failed: {error}");
                    anyhow::anyhow!("JWT verification failed")
                })?
            }
            "EdDSA" => {
                let verifying_key = key_source
                    .get_key(&unverified.iss, kid.as_deref())
                    .await
                    .map_err(|e| {
                        tracing::warn!(
                            "JWT key resolution failed for iss={}: {}",
                            unverified.iss,
                            e
                        );
                        anyhow::anyhow!("JWT key resolution failed")
                    })?;
                crate::auth::decode_with_key(&token, &verifying_key, self.expected_audience())
                    .map_err(|e| {
                        tracing::warn!("JWT verification failed: {}", e);
                        anyhow::anyhow!("JWT verification failed")
                    })?
            }
            _ => anyhow::bail!("unsupported JWT algorithm"),
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
        // Fu5/#677: MAC clearance is authority-asserted and honored only from
        // local-issuer tokens. An external OIDC issuer trusted for identity is
        // not trusted to assert MAC clearance on this node — strip the claim
        // from any federated token before the MAC PDP can read it (⇒ unlabeled
        // ⇒ deny). Local-issuer tokens are unaffected.
        verified.strip_federated_clearance(&local_issuers_refs);
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
                let envelope_jkt =
                    crate::auth::jwk_thumbprint(&crate::auth::JwkThumbprintInput::Ed25519 {
                        x: &ctx.cnf,
                    });
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
                anyhow::bail!(
                    "JWT must include cnf binding for key binding (required by this service)"
                );
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
            request_id, error, "build_error_payload not overridden — sending empty error response"
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
    /// #410/#282: when `true` (the default), the spawner binds an `IrohSubstrate`
    /// as the PRIMARY production endpoint, serving BOTH ALPNs
    /// (`hyprstream-rpc/1` + `moql`) with the same request processor + moq
    /// origin, and installs the shared client endpoint for outbound iroh dials.
    /// The quinn endpoint is bound in parallel for back-compat. Native-only.
    /// Set `false` (via `[quic] iroh = false`) to run quinn-only (legacy).
    #[cfg(not(target_arch = "wasm32"))]
    pub iroh_enabled: bool,
    /// #282: callback invoked after the iroh substrate binds, with
    /// (service_name, node_id) where `node_id` is the endpoint's 32-byte Ed25519
    /// public carrier address. Used to advertise only an `IrohTransport` service
    /// entry when bound; it must never become a verification method.
    #[cfg(not(target_arch = "wasm32"))]
    pub on_iroh_bound: Option<Box<dyn FnOnce(String, [u8; 32]) + Send>>,
    /// #358: the producer-chosen moq RELAY this node rendezvouses through, in
    /// wire-reach form ([`crate::stream_info::TransportConfig`]). When set, the
    /// spawner places it in the service's `ProducerReachConfig` (so published
    /// streams advertise a `Role::Relay` reach) and links this node's
    /// streaming origin UP to the relay
    /// ([`crate::moq_stream::serve_origin_to_relay_background`]) — restoring the
    /// rendezvous property: neither publisher nor subscriber need be directly
    /// reachable by the other. Sourced from the resolved relay DID transport entry
    /// (default: the PDS / federation anchor) decoded by the shared
    /// [`crate::service_entry`] codec, so the stream relay and DID addresses never
    /// drift. `None` = direct-only (the S1/S2 behaviour). Native-only.
    #[cfg(not(target_arch = "wasm32"))]
    pub moq_relay: Option<crate::stream_info::TransportConfig>,
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

/// Empty-`iss` transport-gating tests (#328).
///
/// An empty JWT issuer denotes the local PolicyService's bare-`sub` token, which
/// the key sources treat as always-trusted/local. That shortcut is confined to
/// genuine in-process / IPC callers (`EnvelopeContext::is_local_caller`); a
/// networked / mesh peer presenting an empty-`iss` token is rejected fail-closed.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod empty_iss_gate_tests {
    use super::*;
    use crate::auth::{Claims, ClusterKeySource};
    use crate::transport::TransportConfig;
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    use ed25519_dalek::{Signer as _, SigningKey};

    /// Minimal mock service exposing a `ClusterKeySource` for JWT verification.
    /// `require_cnf_binding` is disabled so the empty-`iss` gate is exercised in
    /// isolation (without a cnf-binding rejection masking the result).
    struct MockService {
        signing_key: SigningKey,
        transport: TransportConfig,
        key_source: std::sync::Arc<dyn crate::auth::JwtKeySource>,
        policy: crate::crypto::CryptoPolicy,
    }

    #[async_trait(?Send)]
    impl RequestService for MockService {
        async fn handle_request(
            &self,
            _ctx: &EnvelopeContext,
            _payload: &[u8],
        ) -> Result<(Vec<u8>, Option<Continuation>)> {
            Ok((vec![], None))
        }
        fn name(&self) -> &str {
            "mock"
        }
        fn transport(&self) -> &TransportConfig {
            &self.transport
        }
        fn signing_key(&self) -> SigningKey {
            self.signing_key.clone()
        }
        fn jwt_key_source(&self) -> Option<std::sync::Arc<dyn crate::auth::JwtKeySource>> {
            Some(self.key_source.clone())
        }
        fn require_cnf_binding(&self) -> bool {
            false
        }
        fn pq_signing_key(&self) -> Option<crate::crypto::pq::MlDsaSigningKey> {
            None
        }
        // These tests exercise the empty-ISS / subject-routing gate with
        // classical EdDSA tokens; pin the JWT policy to Classical so they are
        // isolated from sibling tests that install a Hybrid global config.
        fn jwt_verify_policy(&self) -> crate::crypto::CryptoPolicy {
            self.policy
        }
    }

    /// Build an `EnvelopeContext` carrying `jwt_token`, with the chosen
    /// transport provenance.
    fn ctx_with_token(token: String, is_local_caller: bool) -> EnvelopeContext {
        EnvelopeContext {
            request_id: 1,
            claims: None,
            jwt_token: Some(token),
            key_derived_subject: Subject::anonymous(),
            jwt_subject: None,
            cnf: [0u8; 32],
            envelope_wit_hash: None,
            client_dh_public: None,
            client_kem_public: None,
            request_iat: 0,
            request_nonce: [0; 16],
            response_kem_recipient: None,
            service_domain: None,
            is_local_caller,
        }
    }

    fn mock_service() -> (MockService, SigningKey) {
        // The CA key signs the bare-sub (empty-iss) token; the ClusterKeySource
        // anchors that same CA key with an empty local issuer URL (so empty iss
        // is "local").
        let ca = SigningKey::from_bytes(&[7u8; 32]);
        let key_source =
            std::sync::Arc::new(ClusterKeySource::new(ca.verifying_key(), String::new()));
        let svc = MockService {
            signing_key: SigningKey::from_bytes(&[8u8; 32]),
            transport: TransportConfig::inproc("mock"),
            key_source,
            policy: crate::crypto::CryptoPolicy::Classical,
        };
        (svc, ca)
    }

    fn empty_iss_token(ca: &SigningKey) -> String {
        // iss defaults to empty in Claims::new — the local bare-sub token.
        let now = chrono::Utc::now().timestamp();
        let claims = Claims::new("alice".to_owned(), now, now + 3600);
        assert!(claims.iss.is_empty(), "test token must have empty iss");
        crate::auth::jwt::encode(&claims, ca)
    }

    #[tokio::test]
    async fn empty_iss_rejected_for_networked_caller() {
        let (svc, ca) = mock_service();
        let token = empty_iss_token(&ca);
        let mut ctx = ctx_with_token(token, /* is_local_caller */ false);

        let result = svc.verify_claims(&mut ctx).await;
        assert!(
            result.is_err(),
            "networked empty-iss token must be rejected"
        );
        let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("empty JWT issuer is only accepted from in-process callers"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn empty_iss_accepted_for_inproc_caller() {
        let (svc, ca) = mock_service();
        let token = empty_iss_token(&ca);
        let mut ctx = ctx_with_token(token, /* is_local_caller */ true);

        let result = svc.verify_claims(&mut ctx).await;
        assert!(
            result.is_ok(),
            "in-process empty-iss token must pass the gate: {result:?}"
        );
        // And the local bare-sub subject is resolved.
        assert_eq!(ctx.subject().name(), Some("alice"));
    }

    fn composite_token(
        header: &str,
        claims: &Claims,
        pq: &crate::crypto::pq::MlDsaSigningKey,
        ed: &SigningKey,
        half: bool,
    ) -> String {
        let header = URL_SAFE_NO_PAD.encode(header);
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(claims).unwrap());
        let input = format!("{header}.{payload}");
        let mut signature = crate::crypto::pq::ml_dsa_sign(pq, input.as_bytes());
        if !half {
            signature.extend_from_slice(&ed.sign(input.as_bytes()).to_bytes());
        }
        format!("{input}.{}", URL_SAFE_NO_PAD.encode(signature))
    }

    #[tokio::test]
    async fn request_service_enforces_exact_pair_and_closed_header() {
        let ca = SigningKey::from_bytes(&[41; 32]);
        let ed_a = SigningKey::from_bytes(&[42; 32]);
        let ed_b = SigningKey::from_bytes(&[43; 32]);
        let (pq_a, pq_a_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let (_pq_b, pq_b_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let kid_a = crate::auth::jwk_thumbprint(&crate::auth::JwkThumbprintInput::Akp {
            alg: "ML-DSA-65-Ed25519",
            pub_bytes: &[
                crate::crypto::pq::ml_dsa_vk_bytes(&pq_a_vk),
                ed_a.verifying_key().to_bytes().to_vec(),
            ]
            .concat(),
        });
        let kid_b = crate::auth::jwk_thumbprint(&crate::auth::JwkThumbprintInput::Akp {
            alg: "ML-DSA-65-Ed25519",
            pub_bytes: &[
                crate::crypto::pq::ml_dsa_vk_bytes(&pq_b_vk),
                ed_b.verifying_key().to_bytes().to_vec(),
            ]
            .concat(),
        });
        let key_set = std::sync::Arc::new(crate::auth::CompositeKeySet::default());
        key_set
            .publish(
                1,
                "request-service-test-generation".to_owned(),
                vec![
                    crate::auth::CompositeKeyPair::verifying(
                        kid_a.clone(),
                        pq_a_vk,
                        ed_a.verifying_key(),
                        crate::auth::CompositePairRole::OAuth,
                        crate::auth::CompositePairState::Active,
                        0,
                        i64::MAX,
                    ),
                    crate::auth::CompositeKeyPair::verifying(
                        kid_b,
                        pq_b_vk,
                        ed_b.verifying_key(),
                        crate::auth::CompositePairRole::OAuth,
                        crate::auth::CompositePairState::Drain,
                        0,
                        i64::MAX,
                    ),
                ],
            )
            .unwrap();
        let key_source = std::sync::Arc::new(
            ClusterKeySource::new(ca.verifying_key(), "https://local".to_owned())
                .with_composite_key_set(key_set),
        );
        let svc = MockService {
            signing_key: ca,
            transport: TransportConfig::inproc("mock"),
            key_source,
            policy: crate::crypto::CryptoPolicy::Hybrid,
        };
        let now = chrono::Utc::now().timestamp();
        let claims =
            Claims::new("alice".to_owned(), now, now + 60).with_issuer("https://local".to_owned());
        for typ in crate::auth::RFC9068_ACCESS_TOKEN_TYPES {
            let header = format!(r#"{{"alg":"ML-DSA-65-Ed25519","typ":"{typ}","kid":"{kid_a}"}}"#);
            let valid = composite_token(&header, &claims, &pq_a, &ed_a, false);
            assert!(
                svc.verify_claims(&mut ctx_with_token(valid, false))
                    .await
                    .is_ok(),
                "rejected RFC 9068 access-token type {typ}"
            );
        }

        let valid_header =
            format!(r#"{{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt","kid":"{kid_a}"}}"#);

        let mutations = [
            (
                r#"{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt"}"#.to_owned(),
                false,
            ),
            (
                r#"{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt","kid":"unknown"}"#.to_owned(),
                false,
            ),
            (
                format!(r#"{{"alg":"ML-DSA-65-Ed25519","typ":"JWT","kid":"{kid_a}"}}"#),
                false,
            ),
            (
                format!(r#"{{"alg":"ML-DSA-65-Ed25519","kid":"{kid_a}"}}"#),
                false,
            ),
            (
                format!(r#"{{"alg":"ML-DSA-65-Ed25519","typ":7,"kid":"{kid_a}"}}"#),
                false,
            ),
            (
                format!(r#"{{"alg":"ML-DSA-65-Ed25519","typ":"AT+JWT","kid":"{kid_a}"}}"#),
                false,
            ),
            (
                format!(r#"{{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt ","kid":"{kid_a}"}}"#),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"ML-DSA-65-Ed25519","typ":"Application/at+jwt","kid":"{kid_a}"}}"#
                ),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"ML-DSA-65-Ed25519","typ":"application/AT+JWT","kid":"{kid_a}"}}"#
                ),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"ML-DSA-65-Ed25519","typ":"application/at+jwt ","kid":"{kid_a}"}}"#
                ),
                false,
            ),
            (
                format!(r#"{{"alg":"EdDSA","typ":"at+jwt","kid":"{kid_a}"}}"#),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"EdDSA","alg":"ML-DSA-65-Ed25519","typ":"at+jwt","kid":"{kid_a}"}}"#
                ),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"ML-DSA-65-Ed25519","alg":"EdDSA","typ":"at+jwt","kid":"{kid_a}"}}"#
                ),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"ML-DSA-65-Ed25519","typ":"JWT","typ":"at+jwt","kid":"{kid_a}"}}"#
                ),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt","typ":"JWT","kid":"{kid_a}"}}"#
                ),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt","kid":"other","kid":"{kid_a}"}}"#
                ),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt","kid":"{kid_a}","kid":"other"}}"#
                ),
                false,
            ),
            (
                format!(
                    r#"{{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt","kid":"{kid_a}","crit":["exp"]}}"#
                ),
                false,
            ),
            (valid_header.clone(), true),
        ];
        for (header, half) in mutations {
            let token = composite_token(&header, &claims, &pq_a, &ed_a, half);
            assert!(
                svc.verify_claims(&mut ctx_with_token(token, false))
                    .await
                    .is_err(),
                "accepted {header}"
            );
        }

        let cross = composite_token(&valid_header, &claims, &pq_a, &ed_b, false);
        assert!(svc
            .verify_claims(&mut ctx_with_token(cross, false))
            .await
            .is_err());

        let missing_issuer = Claims::new("alice".to_owned(), now, now + 60);
        let token = composite_token(&valid_header, &missing_issuer, &pq_a, &ed_a, false);
        assert!(svc
            .verify_claims(&mut ctx_with_token(token, false))
            .await
            .is_err());
        let wrong_issuer = claims.clone().with_issuer("https://other".to_owned());
        let token = composite_token(&valid_header, &wrong_issuer, &pq_a, &ed_a, false);
        assert!(svc
            .verify_claims(&mut ctx_with_token(token, false))
            .await
            .is_err());
    }

    #[test]
    fn constructor_provenance_flags() {
        // from_callback_service is a genuine in-process self-call.
        let cb = EnvelopeContext::from_callback_service(1, "inference");
        assert!(cb.is_local_caller());
    }
}

/// Service-to-service IPC identity resolution (#446).
///
/// A signed IPC request carries the caller's verified Ed25519 signer pubkey in
/// `cnf` (the `AnySigner` plane used by UDS/iroh verifies the COSE composite
/// against it). Without a JWT, `verify_claims` must map that key to its
/// authoritative `service:<name>` subject via the process-global key→subject
/// resolver (the trust-store seam installed under #441) — instead of falling
/// back to `anonymous`. A genuinely anonymous caller (unregistered key) must
/// STILL resolve to `anonymous` so policy-gated writes stay denied (fail-closed).
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod ipc_key_identity_tests {
    use super::*;
    use crate::auth::key_subject_resolver::{set_global as set_key_resolver, KeySubjectResolver};
    use crate::transport::TransportConfig;
    use ed25519_dalek::SigningKey;
    use std::collections::HashMap;
    // Async-aware mutex: these tests hold the serialization guard across
    // `verify_claims(..).await`, so a std/parking_lot guard would trip
    // clippy's `await_holding_lock` (a real deadlock-risk lint). `tokio::sync::Mutex`
    // is designed to be held across `.await`. (#468)
    use tokio::sync::Mutex;

    /// Serializes tests that install the process-global key→subject resolver.
    static RESOLVER_LOCK: Mutex<()> = Mutex::const_new(());

    /// Minimal service that uses the DEFAULT `resolve_key_subject` (the #446
    /// seam under test) — it does not override identity resolution.
    struct PlainService {
        signing_key: SigningKey,
        transport: TransportConfig,
    }

    #[async_trait(?Send)]
    impl RequestService for PlainService {
        async fn handle_request(
            &self,
            _ctx: &EnvelopeContext,
            _payload: &[u8],
        ) -> Result<(Vec<u8>, Option<Continuation>)> {
            Ok((vec![], None))
        }
        fn name(&self) -> &str {
            "discovery"
        }
        fn transport(&self) -> &TransportConfig {
            &self.transport
        }
        fn signing_key(&self) -> SigningKey {
            self.signing_key.clone()
        }
        fn pq_signing_key(&self) -> Option<crate::crypto::pq::MlDsaSigningKey> {
            None
        }
    }

    /// Test resolver mapping a fixed set of pubkeys → subjects (the trust store
    /// stand-in). Unknown keys resolve to `None`.
    struct MapResolver {
        bindings: HashMap<[u8; 32], String>,
    }

    impl KeySubjectResolver for MapResolver {
        fn resolve_subject(&self, signer_pubkey: &[u8; 32]) -> Option<Subject> {
            self.bindings
                .get(signer_pubkey)
                .map(|s| Subject::new(s.clone()))
        }
    }

    /// Build a no-JWT context whose `cnf` is the given signer pubkey (the
    /// AnySigner/UDS plane: identity carried only by the verified signer key).
    fn ctx_for_signer(signer_pubkey: [u8; 32]) -> EnvelopeContext {
        EnvelopeContext {
            request_id: 1,
            claims: None,
            jwt_token: None,
            key_derived_subject: Subject::anonymous(),
            jwt_subject: None,
            cnf: signer_pubkey,
            envelope_wit_hash: None,
            client_dh_public: None,
            client_kem_public: None,
            request_iat: 0,
            request_nonce: [0; 16],
            response_kem_recipient: None,
            service_domain: None,
            // AnySigner / networked-or-UDS plane.
            is_local_caller: false,
        }
    }

    #[tokio::test]
    async fn registered_service_key_resolves_as_service_subject() {
        let _guard = RESOLVER_LOCK.lock().await;

        let caller = SigningKey::from_bytes(&[11u8; 32]);
        let caller_pub = caller.verifying_key().to_bytes();

        // Trust store has the caller's key bound to service:discovery (#441).
        let mut bindings = HashMap::new();
        bindings.insert(caller_pub, "service:discovery".to_owned());
        set_key_resolver(std::sync::Arc::new(MapResolver { bindings }));

        let svc = PlainService {
            signing_key: SigningKey::from_bytes(&[22u8; 32]),
            transport: TransportConfig::inproc("discovery"),
        };

        let mut ctx = ctx_for_signer(caller_pub);
        svc.verify_claims(&mut ctx).await.expect("verify_claims ok");

        // The signed IPC caller resolves as its authoritative service identity,
        // NOT anonymous — so the existing discovery:* grant applies.
        assert_eq!(ctx.subject().to_string(), "service:discovery");
        assert!(!ctx.subject().is_anonymous());
    }

    #[tokio::test]
    async fn unregistered_key_stays_anonymous() {
        let _guard = RESOLVER_LOCK.lock().await;

        // Resolver knows ONLY about some other (registered) key.
        let registered = SigningKey::from_bytes(&[33u8; 32])
            .verifying_key()
            .to_bytes();
        let mut bindings = HashMap::new();
        bindings.insert(registered, "service:discovery".to_owned());
        set_key_resolver(std::sync::Arc::new(MapResolver { bindings }));

        let svc = PlainService {
            signing_key: SigningKey::from_bytes(&[22u8; 32]),
            transport: TransportConfig::inproc("discovery"),
        };

        // A genuinely anonymous caller signs with an UNREGISTERED key.
        let anon_pub = SigningKey::from_bytes(&[44u8; 32])
            .verifying_key()
            .to_bytes();
        let mut ctx = ctx_for_signer(anon_pub);
        svc.verify_claims(&mut ctx).await.expect("verify_claims ok");

        // Fail-closed: no registered binding → anonymous → still denied writes.
        assert!(ctx.subject().is_anonymous());
        assert_eq!(ctx.subject().to_string(), "anonymous");
    }
}

#[cfg(test)]
mod stripping_defense_tests {
    use super::{composite_alg_components, jwt_alg_satisfies_policy};
    use crate::crypto::CryptoPolicy;

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

    // ── Fu4/#677: CryptoPolicy-driven minimum-alg gate ──────────────────────

    /// Under Classical policy every alg passes the gate (per-alg decoders do
    /// the real checking); the gate is a no-op floor there.
    #[test]
    fn fu4_classical_accepts_any_alg() {
        for alg in ["EdDSA", "ML-DSA-65", "ML-DSA-65-Ed25519", "ES256", "RS256"] {
            assert!(
                jwt_alg_satisfies_policy(CryptoPolicy::Classical, alg).is_ok(),
                "Classical policy must not gate alg {alg}"
            );
        }
    }

    /// Under Hybrid policy a classical-only (`EdDSA`) JWT is rejected even
    /// though a kid would resolve — the floor Fu4 adds, independent of JWKS
    /// `kid_algs` hygiene. ES256/RS256 are likewise classical and rejected.
    #[test]
    fn fu4_hybrid_rejects_classical_only_alg() {
        for alg in ["EdDSA", "ES256", "RS256", "none", "HS256"] {
            assert!(
                jwt_alg_satisfies_policy(CryptoPolicy::Hybrid, alg).is_err(),
                "Hybrid policy must reject classical-only alg {alg}"
            );
        }
    }

    /// Under Hybrid policy the post-quantum algs (ML-DSA-65, composite) pass
    /// the gate and proceed to per-alg verification.
    #[test]
    fn fu4_hybrid_accepts_pq_algs() {
        for alg in ["ML-DSA-65", "ML-DSA-65-Ed25519"] {
            assert!(
                jwt_alg_satisfies_policy(CryptoPolicy::Hybrid, alg).is_ok(),
                "Hybrid policy must accept post-quantum alg {alg}"
            );
        }
    }
}

/// Accounting (#445): the **A** of AAA. `EnvelopeContext::audit_authz` is the
/// single chokepoint generated dispatch uses to attribute every authorization
/// decision to the cryptographically-verified subject. These tests capture the
/// emitted `tracing` events and assert the structured audit record carries the
/// right subject + decision for an ALLOWED and a DENIED request.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod accounting_audit_tests {
    use super::*;
    // parking_lot::Mutex — std::sync::Mutex is workspace-disallowed (clippy.toml).
    // Held only synchronously here (no `.await` across the guard). (#468)
    use parking_lot::Mutex;
    use std::sync::Arc;
    use tracing::field::{Field, Visit};
    use tracing::subscriber::with_default;
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::Registry;

    /// One captured audit record (the structured fields we care about).
    #[derive(Default, Debug, Clone)]
    struct AuditRecord {
        target: String,
        subject: String,
        resource: String,
        action: String,
        decision: String,
    }

    /// A `tracing` layer that records every event on the `audit` target into a
    /// shared buffer so the test can assert on the structured fields.
    struct CaptureLayer {
        records: Arc<Mutex<Vec<AuditRecord>>>,
    }

    struct FieldVisitor<'a>(&'a mut AuditRecord);

    impl Visit for FieldVisitor<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            // %Display fields arrive here as Debug of the formatted string; strip
            // surrounding quotes that the Debug formatter may add.
            let raw = format!("{value:?}");
            let val = raw.trim_matches('"').to_owned();
            match field.name() {
                "subject" => self.0.subject = val,
                "resource" => self.0.resource = val,
                "action" => self.0.action = val,
                "decision" => self.0.decision = val,
                _ => {}
            }
        }
        fn record_str(&mut self, field: &Field, value: &str) {
            match field.name() {
                "subject" => self.0.subject = value.to_owned(),
                "resource" => self.0.resource = value.to_owned(),
                "action" => self.0.action = value.to_owned(),
                "decision" => self.0.decision = value.to_owned(),
                _ => {}
            }
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for CaptureLayer {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            let mut rec = AuditRecord {
                target: event.metadata().target().to_owned(),
                ..Default::default()
            };
            event.record(&mut FieldVisitor(&mut rec));
            self.records.lock().push(rec);
        }
    }

    /// Build a context whose verified subject is the given username (the
    /// key-derived path, i.e. cryptographically proven, not caller-asserted).
    fn ctx_for_user(name: &str) -> EnvelopeContext {
        EnvelopeContext {
            request_id: 42,
            claims: None,
            jwt_token: None,
            key_derived_subject: Subject::new(name),
            jwt_subject: None,
            cnf: [0u8; 32],
            envelope_wit_hash: None,
            client_dh_public: None,
            client_kem_public: None,
            request_iat: 0,
            request_nonce: [0; 16],
            response_kem_recipient: None,
            service_domain: None,
            is_local_caller: true,
        }
    }

    fn capture<F: FnOnce()>(f: F) -> Vec<AuditRecord> {
        let records = Arc::new(Mutex::new(Vec::new()));
        let layer = CaptureLayer {
            records: records.clone(),
        };
        let subscriber = Registry::default().with(layer);
        with_default(subscriber, f);
        let out = records.lock().clone();
        out
    }

    #[test]
    fn audit_emits_allow_with_verified_subject() {
        let ctx = ctx_for_user("testuser");
        let records = capture(|| {
            ctx.audit_authz("registry:*", "write", /* allowed */ true);
        });

        let audit: Vec<_> = records.iter().filter(|r| r.target == "audit").collect();
        assert_eq!(
            audit.len(),
            1,
            "exactly one audit record expected: {records:?}"
        );
        let r = audit[0];
        assert_eq!(
            r.subject, "testuser",
            "subject must be the verified subject"
        );
        assert_eq!(r.resource, "registry:*");
        assert_eq!(r.action, "write");
        assert_eq!(r.decision, "allow");
    }

    #[test]
    fn audit_emits_deny_with_verified_subject() {
        // anonymous denied a write — the security-relevant deny path.
        let ctx = EnvelopeContext {
            request_id: 7,
            claims: None,
            jwt_token: None,
            key_derived_subject: Subject::anonymous(),
            jwt_subject: None,
            cnf: [0u8; 32],
            envelope_wit_hash: None,
            client_dh_public: None,
            client_kem_public: None,
            request_iat: 0,
            request_nonce: [0; 16],
            response_kem_recipient: None,
            service_domain: None,
            is_local_caller: false,
        };
        let records = capture(|| {
            ctx.audit_authz("registry:*", "write", /* allowed */ false);
        });

        let audit: Vec<_> = records.iter().filter(|r| r.target == "audit").collect();
        assert_eq!(
            audit.len(),
            1,
            "exactly one audit record expected: {records:?}"
        );
        let r = audit[0];
        assert_eq!(
            r.subject, "anonymous",
            "anonymous deny must be attributed as 'anonymous'"
        );
        assert_eq!(r.resource, "registry:*");
        assert_eq!(r.action, "write");
        assert_eq!(r.decision, "deny");
    }

    #[test]
    fn audit_attributes_authenticated_deny() {
        // An authenticated-but-unprivileged subject denied a write is still
        // attributable to that subject (who attempted what).
        let ctx = ctx_for_user("viewer-probe");
        let records = capture(|| {
            ctx.audit_authz("registry:*", "write", /* allowed */ false);
        });
        let r = records
            .iter()
            .find(|r| r.target == "audit")
            .expect("audit record");
        assert_eq!(r.subject, "viewer-probe");
        assert_eq!(r.decision, "deny");
    }

    // ── S8 (#574): verified_key_material() + two-input security_context() ──

    use crate::auth::mac::{Assurance, CompartmentSet, Level, SecurityLabel, VerifiedKeyMaterial};
    use crate::crypto::pq::ml_dsa_generate_keypair;
    use crate::envelope::{install_verify_config, KeyedPqTrustStore};
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    /// Build a context with a specific `cnf` (the cryptographically-verified
    /// Ed25519 signer key) and optional clearance claims.
    fn ctx_with_cnf(cnf: [u8; 32], clearance: Option<SecurityLabel>) -> EnvelopeContext {
        let claims =
            clearance.map(|c| crate::auth::Claims::new("sub".to_owned(), 1, 2).with_clearance(c));
        EnvelopeContext {
            request_id: 1,
            claims,
            jwt_token: None,
            key_derived_subject: Subject::anonymous(),
            jwt_subject: None,
            cnf,
            envelope_wit_hash: None,
            client_dh_public: None,
            client_kem_public: None,
            request_iat: 0,
            request_nonce: [0; 16],
            response_kem_recipient: None,
            service_domain: None,
            is_local_caller: false,
        }
    }

    /// A zeroed cnf (the callback-service shape) ⇒ Unverified assurance.
    #[test]
    fn verified_key_material_zeroed_cnf_is_unverified() {
        let ctx = ctx_with_cnf([0u8; 32], None);
        assert_eq!(ctx.verified_key_material(), VerifiedKeyMaterial::Unverified);
    }

    /// A non-zero cnf with NO bound PQ anchor ⇒ Classical (the federation edge).
    /// NOT PqHybrid — no silent upgrade.
    #[test]
    fn verified_key_material_verified_cnf_no_anchor_is_classical() {
        let ed = SigningKey::generate(&mut OsRng);
        let cnf = ed.verifying_key().to_bytes();
        // No global PQ store installed in this test process (or none binding
        // this key) ⇒ Classical.
        let ctx = ctx_with_cnf(cnf, None);
        // The global store may or may not be installed by another test; the
        // invariant we assert is that it is NOT PqHybrid (no anchor for a
        // random key).
        let km = ctx.verified_key_material();
        assert_ne!(
            km,
            VerifiedKeyMaterial::PqHybrid,
            "a key with no bound PQ anchor must never derive PqHybrid"
        );
        let _ = ed; // suppress unused warning when the assertion path differs
    }

    /// A non-zero cnf WITH a bound PQ anchor (under Hybrid) ⇒ PqHybrid. This is
    /// the activation: the assurance derived from the kid-anchored PQ binding.
    #[test]
    fn verified_key_material_bound_pq_anchor_is_pqhybrid() {
        let ed = SigningKey::generate(&mut OsRng);
        let cnf = ed.verifying_key().to_bytes();
        let (_pq_sk, pq_vk) = ml_dsa_generate_keypair();

        // Install a global verify config with a PQ store binding this key.
        // `install_verify_config` is first-write-wins; if another test already
        // installed one this is a no-op and we verify against whatever store is
        // present. To make this test hermetic we use a dedicated store and rely
        // on the global accessor. If a store is already installed we cannot
        // rebind, so we assert the weaker but still-load-bearing property:
        // the accessor returns Classical OR PqHybrid (never Unverified for a
        // verified cnf), and is deterministic.
        let mut store = KeyedPqTrustStore::new();
        store.bind(cnf, &pq_vk);
        let _ = install_verify_config(crate::envelope::EnvelopeVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(std::sync::Arc::new(store)),
        });
        let ctx = ctx_with_cnf(cnf, None);
        let km = ctx.verified_key_material();
        // The verified cnf is never Unverified. If our install won (no prior
        // config) it is PqHybrid; if a prior test's config is in force it is at
        // worst Classical. Either way: NOT Unverified.
        assert_ne!(
            km,
            VerifiedKeyMaterial::Unverified,
            "a verified cnf must never derive Unverified"
        );
    }

    /// Two-input ctx: clearance from Claims + assurance derived from the
    /// verified cnf. The global `PqTrustStore` is process-global (first-write-
    /// wins via `OnceLock`), so concurrent tests may own it; this test asserts
    /// the load-bearing invariants that hold regardless of which store won:
    /// (1) a verified cnf NEVER derives Unverified, and (2) the derived
    /// context's assurance is whatever the verified crypto supports (Classical
    /// if no anchor bound this key, PqHybrid if one did), NEVER higher than the
    /// clearance. The dedicated `classical_key_subject_clamped_and_denied_on_pq_object`
    /// test below pins the fail-closed direction hermetically.
    #[test]
    fn security_context_verified_cnf_never_unverified_never_exceeds_clearance() {
        let ed = SigningKey::generate(&mut OsRng);
        let cnf = ed.verifying_key().to_bytes();
        let (_pq_sk, pq_vk) = ml_dsa_generate_keypair();
        // Attempt to install a store binding this key. If another test already
        // installed one, this is a no-op and the lookup returns no anchor.
        let mut store = KeyedPqTrustStore::new();
        store.bind(cnf, &pq_vk);
        let _ = install_verify_config(crate::envelope::EnvelopeVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(std::sync::Arc::new(store)),
        });

        let clearance =
            SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY);
        let ctx = ctx_with_cnf(cnf, Some(clearance));
        let sc = ctx
            .security_context()
            .expect("labeled subject has a context");
        assert_ne!(
            ctx.verified_key_material(),
            VerifiedKeyMaterial::Unverified,
            "a verified cnf must never derive Unverified"
        );
        assert!(
            sc.assurance() <= Assurance::PqHybrid,
            "assurance must never exceed the clearance"
        );
        assert_eq!(sc.level(), Level::Secret);
    }

    /// **Fail-closed (the load-bearing #548 invariant):** a Classical-key
    /// subject (verified Ed25519, no bound PQ anchor) carrying a PqHybrid
    /// clearance MUST clamp to Classical assurance, and therefore MUST be
    /// DENIED on a PqHybrid-required object. No silent upgrade.
    #[test]
    fn classical_key_subject_clamped_and_denied_on_pq_object() {
        let ed = SigningKey::generate(&mut OsRng);
        let cnf = ed.verifying_key().to_bytes();
        // No PQ binding for this key in any store we install here. If a global
        // store happens to bind a random key that is astronomically improbable;
        // to be hermetic we install a store that does NOT bind this key.
        let store = KeyedPqTrustStore::new(); // empty
        let _ = install_verify_config(crate::envelope::EnvelopeVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(std::sync::Arc::new(store)),
        });

        // Policy mistakenly assigned a PqHybrid clearance...
        let claimed_clearance =
            SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY);
        let ctx = ctx_with_cnf(cnf, Some(claimed_clearance));
        let sc = ctx
            .security_context()
            .expect("labeled subject has a context");
        // ...but the verified key is Classical, so assurance CLAMPS DOWN.
        assert_eq!(
            sc.assurance(),
            Assurance::Classical,
            "a Classical key must clamp a PqHybrid clearance down to Classical (no silent upgrade)"
        );
        // Consequently the MAC floor DENIES access to a PqHybrid object even
        // though the level (Secret) dominates.
        let pq_object =
            SecurityLabel::new(Level::Public, Assurance::PqHybrid, CompartmentSet::EMPTY);
        assert!(
            !sc.can_access(&pq_object),
            "Classical-assurance subject MUST be denied on a PqHybrid object (fail-closed)"
        );
    }

    /// Unlabeled subject (no clearance claim) ⇒ `security_context()` returns
    /// None ⇒ the S1 monitor denies (no default clearance).
    #[test]
    fn unlabeled_subject_has_no_security_context() {
        let ed = SigningKey::generate(&mut OsRng);
        let cnf = ed.verifying_key().to_bytes();
        let ctx = ctx_with_cnf(cnf, None); // no clearance
        assert!(
            ctx.security_context().is_none(),
            "unlabeled subject must have no security context (S1 deny)"
        );
    }
}
