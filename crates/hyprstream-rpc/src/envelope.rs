//! Request envelope for RPC with Ed25519 signatures and E2E integrity.
//!
//! Every RPC request is wrapped in a `SignedEnvelope` that carries:
//! - A unique request ID for correlation
//! - The serialized inner request payload
//! - Authorization context (local claims, federated JWT, or ID-JAG)
//! - Ed25519 signature over the entire RequestEnvelope
//! - Nonce + timestamp (iat) for replay protection
//!
//! # Two-Layer Security
//!
//! | Layer | Mechanism | Purpose |
//! |-------|-----------|---------|
//! | Transport | CURVE/QUIC-TLS | Encrypts connection, authenticates immediate peer |
//! | Application | Signed envelope | E2E integrity through brokers, authenticates originator |
//!
//! # Envelope Structure
//!
//! ```text
//! SignedEnvelope {
//!     envelope: RequestEnvelope {
//!         request_id, payload, iat, nonce,
//!         authorization, delegation_token, wth
//!     },
//!     sig,  // Ed25519 signature over canonical(envelope)
//!     cnf,  // Ed25519 public key (RFC 7800 confirmation key)
//! }
//! ```

use crate::auth::Claims;
use crate::capnp::{FromCapnp, ToCapnp};
use crate::common_capnp;
use crate::crypto::{SigningKey, VerifyingKey};
use crate::error::{EnvelopeError, EnvelopeResult};
use anyhow::{anyhow, Result};
use ed25519_dalek::Signer;
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};
use subtle::ConstantTimeEq;

// ============================================================================
// Authorization (new envelope auth model)
// ============================================================================

/// Authorization context carried inside a `RequestEnvelope`.
///
/// Replaces the legacy `identity` + `jwt_token` + `claims` fields.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Authorization {
    /// No authorization context.
    #[default]
    None,
    /// Locally-issued verified token claims.
    Local(TokenClaims),
    /// Token from a foreign (federated) issuer.
    Federated(FederatedToken),
    /// Opaque identity JAG string.
    IdJag(String),
}

/// Verified token claims (local issuer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenClaims {
    pub iss: String,
    pub sub: String,
    pub aud: Vec<String>,
    pub exp: i64,
    pub iat: i64,
    pub jti: String,
    pub scope: Vec<crate::auth::scope::Scope>,
    pub cnf_jkt: String,
}

/// Federated token from a foreign issuer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FederatedToken {
    pub raw: String,
    pub claims: TokenClaims,
    pub dpop_proof: Option<String>,
}

// Cap'n Proto impls for new Authorization types

impl ToCapnp for TokenClaims {
    type Builder<'a> = common_capnp::token_claims::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_iss(&self.iss);
        builder.set_sub(&self.sub);
        {
            let mut aud_builder = builder.reborrow().init_aud(self.aud.len() as u32);
            for (i, a) in self.aud.iter().enumerate() {
                aud_builder.set(i as u32, a);
            }
        }
        builder.set_exp(self.exp);
        builder.set_iat(self.iat);
        builder.set_jti(&self.jti);
        {
            let mut scope_builder = builder.reborrow().init_scope(self.scope.len() as u32);
            for (i, s) in self.scope.iter().enumerate() {
                let mut sb = scope_builder.reborrow().get(i as u32);
                s.write_to(&mut sb);
            }
        }
        builder.set_cnf_jkt(&self.cnf_jkt);
    }
}

impl FromCapnp for TokenClaims {
    type Reader<'a> = common_capnp::token_claims::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let aud_reader = reader.get_aud()?;
        let mut aud = Vec::with_capacity(aud_reader.len() as usize);
        for i in 0..aud_reader.len() {
            aud.push(aud_reader.get(i)?.to_str()?.to_owned());
        }

        let scope_reader = reader.get_scope()?;
        let mut scope = Vec::with_capacity(scope_reader.len() as usize);
        for i in 0..scope_reader.len() {
            scope.push(crate::auth::scope::Scope::read_from(scope_reader.get(i))?);
        }

        Ok(Self {
            iss: reader.get_iss()?.to_str()?.to_owned(),
            sub: reader.get_sub()?.to_str()?.to_owned(),
            aud,
            exp: reader.get_exp(),
            iat: reader.get_iat(),
            jti: reader.get_jti()?.to_str()?.to_owned(),
            scope,
            cnf_jkt: reader.get_cnf_jkt()?.to_str()?.to_owned(),
        })
    }
}

impl ToCapnp for FederatedToken {
    type Builder<'a> = common_capnp::federated_token::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_raw(&self.raw);
        self.claims.write_to(&mut builder.reborrow().init_claims());
        if let Some(ref proof) = self.dpop_proof {
            builder.set_dpop_proof(proof);
        }
    }
}

impl FromCapnp for FederatedToken {
    type Reader<'a> = common_capnp::federated_token::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let dpop_proof = if reader.has_dpop_proof() {
            let p = reader.get_dpop_proof()?.to_str()?;
            if p.is_empty() {
                None
            } else {
                Some(p.to_owned())
            }
        } else {
            None
        };
        Ok(Self {
            raw: reader.get_raw()?.to_str()?.to_owned(),
            claims: TokenClaims::read_from(reader.get_claims()?)?,
            dpop_proof,
        })
    }
}

impl ToCapnp for Authorization {
    type Builder<'a> = common_capnp::authorization::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        match self {
            Self::None => builder.set_none(()),
            Self::Local(claims) => {
                claims.write_to(&mut builder.reborrow().init_local());
            }
            Self::Federated(token) => {
                token.write_to(&mut builder.reborrow().init_federated());
            }
            Self::IdJag(jag) => {
                builder.set_id_jag(jag);
            }
        }
    }
}

impl FromCapnp for Authorization {
    type Reader<'a> = common_capnp::authorization::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        use common_capnp::authorization::Which;
        match reader.which()? {
            Which::None(()) => Ok(Self::None),
            Which::Local(r) => Ok(Self::Local(TokenClaims::read_from(r?)?)),
            Which::Federated(r) => Ok(Self::Federated(FederatedToken::read_from(r?)?)),
            Which::IdJag(r) => Ok(Self::IdJag(r?.to_str()?.to_owned())),
        }
    }
}

/// Cap'n Proto reader options with bounded traversal limits to prevent DoS.
fn envelope_reader_options() -> capnp::message::ReaderOptions {
    let mut opts = capnp::message::ReaderOptions::new();
    opts.traversal_limit_in_words(Some(131_072)); // 1 MiB
    opts.nesting_limit(64);
    opts
}

/// Parse exactly one bounded Cap'n Proto message from an already-buffered
/// frame without allocating storage based on its unauthenticated segment
/// table. The no-allocation reader validates declared segment sizes against
/// both the traversal limit and the supplied slice before exposing a root.
fn read_exact_envelope_message(
    bytes: &[u8],
) -> Result<capnp::message::Reader<capnp::serialize::NoAllocSliceSegments<'_>>> {
    let mut remaining = bytes;
    let reader = capnp::serialize::read_message_from_flat_slice_no_alloc(
        &mut remaining,
        envelope_reader_options(),
    )?;
    if !remaining.is_empty() {
        return Err(anyhow!(
            "trailing bytes after Cap'n Proto envelope: {}",
            remaining.len()
        ));
    }
    Ok(reader)
}

// ============================================================================
// Envelope Unwrap Options
// ============================================================================

/// How to verify the envelope signer.
pub enum EnvelopeVerification<'a> {
    /// Require the envelope signer to match this specific verifying key.
    FixedSigner(&'a VerifyingKey),
    /// Accept any valid Ed25519 signer (self-signed).
    AnySigner,
}

/// Fu2/#677: how the verifier treats an identity with **no anchored ML-DSA-65
/// key** (an "unanchored" signer) when verifying under a Hybrid [`CryptoPolicy`].
///
/// The composite signature is Weakly Non-Separable (per-identity): the inner
/// EdDSA is independently verifiable, so whether the outer ML-DSA-65 layer is
/// *required* is a per-identity decision. The audit (Fu2) found the verifier
/// accepted unanchored identities classical-only by default — so "hybrid" was
/// anchor-coverage-dependent, unlike the UCAN verifier which denies on a missing
/// anchor. This enum makes the posture explicit and pins the **default to
/// deny-on-missing-anchor**, with an opt-in per-peer allowlist for legacy
/// classical-only peers (e.g. a pre-PQ federation edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnanchoredHybridPolicy {
    /// Fail closed: an unanchored identity is rejected under Hybrid unless its
    /// Ed25519 verifying key (hex) is present in the supplied allowlist. The
    /// **production default** ([`UnwrapOptions`] defaults to this).
    #[default]
    Deny,
    /// Accept the classical inner EdDSA only — the legacy WNS backcompat floor.
    /// Exposed for low-level callers that explicitly opt into per-identity
    /// classical verification; NOT the production default.
    AllowClassicalFloor,
}

/// Options controlling envelope unwrap, verification, and optional decryption.
///
/// # Crypto policy (fail-closed default)
///
/// `verify_policy` defaults to [`CryptoPolicy::default()`] which is **Hybrid
/// ENFORCED**. Under Hybrid the verifier REQUIRES the outer ML-DSA-65 SNS layer
/// anchored to a [`PqTrustStore`] entry — stripping the outer layer or signing
/// classical-only is rejected. To verify against a kid-anchored PQ key, callers
/// MUST supply `pq_store`; under Hybrid with no resolvable anchor the envelope
/// is **rejected** (fail-closed), never accepted as classical.
///
/// Callers that genuinely need the legacy classical-only path (e.g. WASM in-
/// browser verification without a PQ trust store) must explicitly downgrade via
/// [`UnwrapOptions::classical`].
pub struct UnwrapOptions<'a> {
    /// How to verify the envelope signer.
    pub verification: EnvelopeVerification<'a>,
    /// Nonce cache for replay protection.
    pub nonce_cache: &'a dyn NonceCache,
    /// Server signing key for decrypting encrypted envelopes.
    /// When present and the envelope has `encrypted_envelope`, the server's
    /// Ed25519 key is converted to X25519 for DH decryption.
    pub decryption_key: Option<&'a crate::crypto::SigningKey>,
    /// Require the verified wire envelope to carry a non-empty encrypted
    /// envelope. Enforced after signature verification and before decryption,
    /// the final replay commit, or any claims/application processing.
    pub require_encrypted: bool,
    /// kid-anchored ML-DSA-65 trust store used to resolve the PQ verifying key
    /// for the envelope's EdDSA signer. Required under Hybrid policy.
    pub pq_store: Option<&'a dyn PqTrustStore>,
    /// Verification policy enforced at this site. Defaults to Hybrid (enforced).
    pub verify_policy: crate::crypto::CryptoPolicy,
    /// Fu2/#677: posture for an unanchored signer (no ML-DSA-65 anchor) under
    /// Hybrid. Defaults to [`UnanchoredHybridPolicy::Deny`] (deny-on-missing-
    /// anchor). Under `Deny`, an unanchored identity is accepted classical-only
    /// iff its Ed25519 key hex is in [`Self::unanchored_classical_allowlist`].
    pub unanchored_policy: UnanchoredHybridPolicy,
    /// Fu2/#677: Ed25519 verifying-key hexes permitted to verify classical-only
    /// under Hybrid when [`Self::unanchored_policy`] is `Deny`. Empty by default
    /// (every unanchored identity denied). Anchored identities bypass this list.
    pub unanchored_classical_allowlist: Vec<String>,
}

impl<'a> UnwrapOptions<'a> {
    /// Fixed-signer verification under the default (Hybrid-enforced) policy.
    pub fn fixed_signer(pubkey: &'a VerifyingKey, nonce_cache: &'a dyn NonceCache) -> Self {
        Self {
            verification: EnvelopeVerification::FixedSigner(pubkey),
            nonce_cache,
            decryption_key: None,
            require_encrypted: false,
            pq_store: None,
            verify_policy: crate::crypto::CryptoPolicy::default(),
            unanchored_policy: UnanchoredHybridPolicy::default(),
            unanchored_classical_allowlist: Vec::new(),
        }
    }

    /// Any-signer verification under the default (Hybrid-enforced) policy.
    pub fn any_signer(nonce_cache: &'a dyn NonceCache) -> Self {
        Self {
            verification: EnvelopeVerification::AnySigner,
            nonce_cache,
            decryption_key: None,
            require_encrypted: false,
            pq_store: None,
            verify_policy: crate::crypto::CryptoPolicy::default(),
            unanchored_policy: UnanchoredHybridPolicy::default(),
            unanchored_classical_allowlist: Vec::new(),
        }
    }

    pub fn with_decryption_key(mut self, key: &'a crate::crypto::SigningKey) -> Self {
        self.decryption_key = Some(key);
        self
    }

    /// Require request encryption after authenticating the wire envelope.
    pub fn require_encrypted(mut self, required: bool) -> Self {
        self.require_encrypted = required;
        self
    }

    /// Attach a kid-anchored ML-DSA-65 trust store (required under Hybrid).
    pub fn with_pq_store(mut self, store: &'a dyn PqTrustStore) -> Self {
        self.pq_store = Some(store);
        self
    }

    /// Set the verification policy explicitly.
    pub fn with_verify_policy(mut self, policy: crate::crypto::CryptoPolicy) -> Self {
        self.verify_policy = policy;
        self
    }

    /// Fu2/#677: set the unanchored-signer posture under Hybrid (default `Deny`).
    pub fn with_unanchored_policy(mut self, policy: UnanchoredHybridPolicy) -> Self {
        self.unanchored_policy = policy;
        self
    }

    /// Fu2/#677: set the Ed25519-key-hex allowlist of unanchored identities
    /// accepted classical-only under Hybrid+`Deny`.
    pub fn with_unanchored_allowlist(mut self, allowlist: Vec<String>) -> Self {
        self.unanchored_classical_allowlist = allowlist;
        self
    }

    /// Explicitly downgrade this site to the legacy classical-only verifier.
    ///
    /// Use ONLY for surfaces that cannot carry a PQ trust store (e.g. external
    /// JOSE/classical interop). This accepts a single-EdDSA composite and
    /// ignores any outer ML-DSA layer.
    pub fn classical(mut self) -> Self {
        self.verify_policy = crate::crypto::CryptoPolicy::Classical;
        self.pq_store = None;
        self
    }
}

/// Global request ID counter for unique IDs
static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique request ID
pub fn next_request_id() -> u64 {
    REQUEST_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Get current Unix timestamp in milliseconds.
/// Used for envelope timestamp validation.
pub fn current_timestamp() -> i64 {
    #[cfg(target_arch = "wasm32")]
    {
        // In WASM, SystemTime::now() is not available.
        // Use js_sys::Date::now() which returns milliseconds as f64.
        js_sys::Date::now() as i64
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // SAFETY: Only fails if system time is before Unix epoch (1970)
        // Cap at i64::MAX (won't overflow for ~292 million years)
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0)
    }
}

/// Generate a random 16-byte nonce for replay protection.
pub fn generate_nonce() -> [u8; 16] {
    // CodeQL-recognized CSPRNG: `OsRng.gen()` draws every byte uniformly at
    // random with no initialization literal. The previous `[0u8; 16]` scratch
    // buffer tripped `rust/hard-coded-cryptographic-value` even though
    // `fill_bytes` overwrote it entirely — this construction is equivalent in
    // strength (still OsRng-backed) but literal-free, matching the
    // `crypto::event_crypto::random_nonce` pattern used for AES-GCM nonces.
    use rand::Rng;
    rand::rngs::OsRng.gen()
}

/// Authorization subject for Casbin policy checks and resource isolation.
///
/// A simple newtype over `Option<String>`:
/// - `Some(name)` = authenticated user with bare username
/// - `None` = anonymous (unauthenticated)
///
/// All identity types (Local, ApiToken, Peer, Claims) produce bare usernames.
/// There are no namespace prefixes — a user named "alice" is just "alice"
/// regardless of how they authenticated.
///
/// # Display
///
/// `Subject::new("alice")` displays as `"alice"`, used directly
/// as the Casbin subject string. Anonymous displays as `"anonymous"`.
///
/// # Validation
///
/// `validate()` ensures names contain only `[a-zA-Z0-9._-]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Subject(Option<String>);

impl Subject {
    /// Create an authenticated subject with the given username.
    pub fn new(name: impl Into<String>) -> Self {
        Self(Some(name.into()))
    }

    /// Create an anonymous (unauthenticated) subject.
    pub fn anonymous() -> Self {
        Self(None)
    }

    /// Get the username, if authenticated.
    pub fn name(&self) -> Option<&str> {
        self.0.as_deref()
    }

    /// Check if this subject is anonymous.
    pub fn is_anonymous(&self) -> bool {
        self.0.is_none()
    }

    /// Validate that the name contains only safe characters `[a-zA-Z0-9._-]`.
    ///
    /// This prevents path traversal and other injection attacks when the
    /// subject is used in filesystem paths or policy strings.
    ///
    /// Note: federated subjects (containing `://`) intentionally bypass this
    /// validation — call `validate()` only for local subjects.
    pub fn validate(&self) -> Result<()> {
        if self.is_federated() {
            return Ok(()); // federated subjects are validated at JWT decode time
        }

        let name = match &self.0 {
            Some(n) => n.as_str(),
            None => return Ok(()),
        };

        if name.is_empty() {
            return Err(anyhow!("Subject name must not be empty"));
        }

        for c in name.chars() {
            if !c.is_ascii_alphanumeric() && c != '.' && c != '_' && c != '-' {
                return Err(anyhow!(
                    "Subject name contains invalid character '{}': only [a-zA-Z0-9._-] allowed",
                    c
                ));
            }
        }

        Ok(())
    }

    /// Create a federated subject for a principal from a foreign issuer.
    ///
    /// Format: `"{issuer_url}:{local_sub}"` — for example
    /// `"https://node-a:alice"`. The `://` in the URL makes federated subjects
    /// unambiguously distinguishable from local bare usernames.
    ///
    /// The resulting subject is used directly as the Casbin subject string so
    /// federation-aware policies can match on the issuer prefix.
    pub fn federated(iss: &str, sub: &str) -> Self {
        Subject(Some(format!("{iss}:{sub}")))
    }

    /// Returns `true` if this subject originated from a foreign issuer.
    ///
    /// Federated subjects contain `"://"` (from the issuer URL); local bare
    /// usernames never do.
    pub fn is_federated(&self) -> bool {
        self.0
            .as_deref()
            .map(|s| s.contains("://"))
            .unwrap_or(false)
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(name) => write!(f, "{name}"),
            None => write!(f, "anonymous"),
        }
    }
}

impl FromStr for Subject {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        if s == "anonymous" {
            return Ok(Self::anonymous());
        }

        // Legacy compatibility: strip known prefixes
        if let Some((prefix, name)) = s.split_once(':') {
            if matches!(prefix, "local" | "token" | "peer" | "user") {
                return Ok(Self::new(name));
            }
        }

        Ok(Self::new(s))
    }
}

impl From<&Claims> for Subject {
    fn from(claims: &Claims) -> Self {
        // Phase 7: Federated subjects use "iss:sub" format so that
        // "alice@node-a" and local "alice" never collide in Casbin policy.
        if claims.iss.is_empty() {
            Subject::new(claims.sub.clone())
        } else {
            Subject::federated(&claims.iss, &claims.sub)
        }
    }
}

// Cap'n Proto implementation for Subject (simple text field)
impl ToCapnp for Subject {
    type Builder<'a> = common_capnp::subject::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_name(self.to_string());
    }
}

impl FromCapnp for Subject {
    type Reader<'a> = common_capnp::subject::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let name = reader.get_name()?.to_str()?;
        name.parse()
    }
}

/// Unsigned envelope wrapping an RPC request.
///
/// This struct contains all request metadata and is signed by `SignedEnvelope`.
/// The entire serialized RequestEnvelope is covered by the signature.
///
/// # Replay Protection
///
/// - `nonce`: 16 random bytes, must be unique per request
/// - `iat`: Unix milliseconds, requests older than 5 minutes are rejected
#[derive(Debug, Clone)]
pub struct RequestEnvelope {
    /// Unique request ID for correlation and logging
    pub request_id: u64,

    /// Serialized inner request (e.g., RegistryRequest, InferenceRequest)
    pub payload: Vec<u8>,

    /// Unix timestamp in milliseconds for expiration check
    pub iat: i64,

    /// Random nonce for replay protection (16 bytes)
    pub nonce: [u8; 16],

    /// Authorization context
    pub authorization: Authorization,

    /// Delegation token relayed by a trusted service (e.g., OAI, MCP adapter).
    ///
    /// **Reserved.** Serialized in the envelope and covered by the signature,
    /// but not yet consumed server-side. Intended for bearer delegation flows
    /// where a gateway service forwards a user's JWT on their behalf.
    /// Server-side extraction will be added when the delegation trust model
    /// is finalized.
    pub delegation_token: Option<String>,

    /// SHA-256 hash of the WIT JWT string (WIMSE wth claim).
    /// Binds this proof to a specific Workload Identity Token even when
    /// the JWT is omitted (trust-store cache-hit path).
    pub wth: Option<[u8; 32]>,

    /// Client's ephemeral DH public key for stream key derivation.
    /// Present on streaming requests; the server uses this with its own
    /// ephemeral keypair to derive the shared secret for HMAC chain keys.
    pub client_dh_public: Option<[u8; 32]>,

    /// Fresh, suite-complete HyKEM recipient for identified stream setup.
    ///
    /// This is authenticated inside the request envelope and is distinct from
    /// both the legacy classical DH field and the one-shot unary-response
    /// recipient.  The server must bind it into an [`IdentifiedStreamBinding`]
    /// before deriving or releasing stream epoch keys.
    pub client_kem_public: Option<crate::crypto::hybrid_kem::RecipientPublic>,

    /// Fresh per-call hybrid-KEM recipient for the unary response. This is
    /// distinct from both legacy `client_dh_public` and stream-plane
    /// `clientKemPublic`, and is carried only inside a sealed network request.
    pub response_kem_recipient: Option<crate::crypto::hybrid_kem::RecipientPublic>,

    /// Canonical destination service. On untrusted carriers this is required,
    /// carried only in the signed+sealed request, and checked by dispatch.
    pub service_domain: Option<String>,
}

impl RequestEnvelope {
    /// Create a new request envelope with fresh request ID, nonce, and timestamp.
    pub fn new(payload: Vec<u8>) -> Self {
        Self {
            request_id: next_request_id(),
            payload,
            iat: current_timestamp(),
            nonce: generate_nonce(),
            authorization: Authorization::None,
            delegation_token: None,
            wth: None,
            client_dh_public: None,
            client_kem_public: None,
            response_kem_recipient: None,
            service_domain: None,
        }
    }

    /// Set authorization context.
    pub fn with_authorization(mut self, auth: Authorization) -> Self {
        self.authorization = auth;
        self
    }

    /// Set opaque JWT token as IdJag authorization.
    pub fn with_jwt_token(mut self, token: String) -> Self {
        self.authorization = Authorization::IdJag(token);
        self
    }

    /// Set delegation token for relay by a trusted service.
    pub fn with_delegation_token(mut self, token: String) -> Self {
        self.delegation_token = Some(token);
        self
    }

    /// Bind this proof to a specific WIT by storing SHA-256(jwt).
    /// Call this on cached-identity requests where jwtToken is omitted.
    pub fn with_wth_of(mut self, jwt: &str) -> Self {
        use sha2::{Digest, Sha256};
        self.wth = Some(Sha256::digest(jwt.as_bytes()).into());
        self
    }

    /// Set the client's ephemeral DH public key for stream key derivation.
    pub fn with_client_dh_public(mut self, key: [u8; 32]) -> Self {
        self.client_dh_public = Some(key);
        self
    }

    /// Set the identified stream's ephemeral, pinned-suite HyKEM recipient.
    pub fn with_client_kem_public(
        mut self,
        recipient: crate::crypto::hybrid_kem::RecipientPublic,
    ) -> EnvelopeResult<Self> {
        recipient
            .validate()
            .map_err(|error| EnvelopeError::KeyExchange(error.to_string()))?;
        if recipient.suite_id != crate::stream_epoch::IDENTIFIED_STREAM_SUITE {
            return Err(EnvelopeError::KeyExchange(
                "identified streams require the pinned X25519+ML-KEM-768 suite".into(),
            ));
        }
        self.client_kem_public = Some(recipient);
        Ok(self)
    }

    /// Set the one-shot HyKEM recipient for the corresponding unary response.
    pub fn with_response_kem_recipient(
        mut self,
        recipient: crate::crypto::hybrid_kem::RecipientPublic,
    ) -> Self {
        self.response_kem_recipient = Some(recipient);
        self
    }

    /// Bind the request to one canonical destination service.
    pub fn with_service_domain(mut self, service_domain: impl Into<String>) -> Result<Self> {
        let service_domain = service_domain.into();
        validate_service_domain(&service_domain)?;
        self.service_domain = Some(service_domain);
        Ok(self)
    }

    /// Create an envelope for an anonymous request.
    pub fn anonymous(payload: Vec<u8>) -> Self {
        Self::new(payload)
    }

    /// Extract the JWT string from authorization, if it's an IdJag.
    pub fn jwt_token(&self) -> Option<&str> {
        match &self.authorization {
            Authorization::IdJag(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Serialize this envelope to canonical Cap'n Proto bytes for signing.
    ///
    /// SECURITY: This method MUST produce deterministic output for cryptographic
    /// signatures to work correctly. We use Cap'n Proto's canonical form as
    /// specified in the encoding spec:
    /// https://capnproto.org/encoding.html#canonicalization
    ///
    /// Canonical form guarantees:
    /// - Same logical data → identical bytes (always)
    /// - Preorder encoding (deterministic pointer ordering)
    /// - Single segment (no segment boundary ambiguity)
    /// - Truncated trailing zeros (no padding differences)
    /// - No packing (binary format is deterministic)
    /// - No segment table framing (redundant for signing)
    ///
    /// These bytes are what gets signed in a SignedEnvelope.
    pub fn to_bytes(&self) -> Vec<u8> {
        use capnp::message::Builder;
        use capnp::Word;

        // Build the message.
        let mut message = Builder::new_default();
        {
            let mut builder = message.init_root::<common_capnp::request_envelope::Builder>();
            self.write_to(&mut builder);
        }

        // Canonicalize directly from the builder (#178). Cap'n Proto is
        // zero-copy; the previous implementation defeated that on the signing
        // hot path by serializing to a temp Vec and reparsing it just to obtain
        // a Reader. `Builder::into_reader()` reads the builder's own segments in
        // place — no temp serialize, no reparse. Canonicalization is still
        // REQUIRED: capnp messages aren't canonical by default, and signatures
        // must be over deterministic bytes for cross-platform verification.
        //
        // `into_reader()` uses unlimited traversal, which is correct here: the
        // message is self-produced (not untrusted input), and it removes a
        // latent bug where the old 1-MiB-limited reader silently returned empty
        // bytes — breaking signing — for envelopes larger than 1 MiB.
        let canonical_words = match message.into_reader().canonicalize() {
            Ok(words) => words,
            Err(_e) => {
                #[cfg(not(target_arch = "wasm32"))]
                tracing::error!("Envelope canonicalization failed: {}", _e);
                return Vec::new();
            }
        };

        // Convert Words to bytes (raw segment data, NO stream framing).
        Word::words_to_bytes(&canonical_words).to_vec()
    }
}

/// Signed envelope wrapping a RequestEnvelope with Ed25519 signature.
///
/// The signature covers the **serialized bytes** of the inner `RequestEnvelope`,
/// making the signing scope structurally explicit.
///
/// # Security
///
/// - Signature survives message forwarding (unlike transport-layer auth)
/// - Nonce + timestamp prevent replay attacks
/// - Signer pubkey included for verification without key lookup
///
/// # Example
///
/// ```ignore
/// use hyprstream_rpc::envelope::{RequestEnvelope, SignedEnvelope};
/// use hyprstream_rpc::crypto::SigningKey;
///
/// // Create and sign
/// let envelope = RequestEnvelope::anonymous(payload);
/// let signed = SignedEnvelope::new_signed(envelope, &signing_key);
///
/// // Verify
/// signed.verify(&expected_pubkey, &nonce_cache)?;
/// ```
#[derive(Debug, Clone)]
pub struct SignedEnvelope {
    /// The unsigned envelope (cleartext path, or decrypted from encrypted_envelope)
    pub envelope: RequestEnvelope,

    /// Ed25519 signature (64 bytes)
    pub sig: [u8; 64],

    /// Ed25519 public key of the signer (32 bytes)
    pub cnf: [u8; 32],

    /// AES-256-GCM-SIV ciphertext of serialized RequestEnvelope (None = cleartext mode)
    pub encrypted_envelope: Option<Vec<u8>>,

    /// X25519 ephemeral public key for DH key agreement (present when encrypted)
    pub client_ephemeral_public: Option<[u8; 32]>,

    /// M3 (#152): CBOR-encoded COSE_Sign composite signature (detached).
    ///
    /// Authoritative authentication mechanism. Carries one EdDSA entry
    /// (Classical) or EdDSA + ML-DSA-65 entries (Hybrid) over the canonical
    /// signing-data. The ML-DSA-65 verifying key is NOT embedded; it is
    /// resolved by kid from a trust store (kid-anchored), which fixes the
    /// prior self-certification weakness.
    ///
    /// `sig`/`cnf` remain populated with the raw EdDSA signature + signer
    /// public key for backward compatibility and for the JWT `cnf` key-binding
    /// path, but the COSE composite is what `verify*` enforces.
    pub cose: Vec<u8>,

    /// Runtime crypto policy used when this envelope was signed.
    pub policy: crate::crypto::CryptoPolicy,

    /// ML-KEM-768 ciphertext (1088 bytes, present when hybrid encryption is used)
    pub pq_kem_ciphertext: Option<Vec<u8>>,
}

/// Cap'n Proto file id of `common.capnp` (the envelope schema). Used as the
/// first component of the COSE `external_aad` schema-binding.
pub const ENVELOPE_SCHEMA_ID: u64 = 0xb3e9_f4a1_c7d8_2056;

/// Stable inner-type id for `RequestEnvelope` in the COSE `external_aad`.
/// Distinct from any payload schema id; binds the COSE signature to the
/// envelope structure to prevent schema-confusion / cross-protocol replay.
pub const REQUEST_ENVELOPE_TYPE_ID: u64 = 0x5265_7145_6e76_3031; // "ReqEnv01"

/// Stable inner-type id for `ResponseEnvelope`, distinct from
/// [`REQUEST_ENVELOPE_TYPE_ID`]. The `ResponseEnvelope` COSE composite (#275)
/// binds this type-id into its `external_aad` via
/// [`response_envelope_external_aad`], so a response COSE signature can NEVER
/// verify as a request signature (or vice-versa) — the request↔response domain
/// separation is cryptographically enforced, not merely documented.
pub const RESPONSE_ENVELOPE_TYPE_ID: u64 = 0x5265_7350_456e_7631; // "RsPEnv1"

/// Build the COSE external_aad used for all REQUEST envelope signatures.
pub(crate) fn envelope_external_aad() -> Vec<u8> {
    crate::crypto::cose_sign1::build_external_aad(ENVELOPE_SCHEMA_ID, REQUEST_ENVELOPE_TYPE_ID)
}

/// Build the external_aad for an ENCRYPTED (`#mesh-kem` COSE_Encrypt0) request
/// envelope, binding the outer replay metadata (`request_id`, `iat`, `nonce`)
/// into the AEAD.
///
/// # Why this is security-critical
///
/// The encrypted request path serializes a *redacted* outer `RequestEnvelope`
/// (replay metadata only) beside the ciphertext, and the server runs its replay
/// check on those OUTER fields *before* decapsulating the KEM (a deliberate DoS
/// pre-filter). Those outer fields are not individually signed — the COSE
/// composite signature covers only the ciphertext bytes. Unless they are folded
/// into the AEAD's `external_aad`, a network attacker can capture an encrypted
/// envelope and replay it with a mutated outer `nonce`/`iat`: the signature
/// still validates (ciphertext unchanged), the replay check passes (fresh outer
/// nonce), and a static AAD still decrypts — a full replay bypass.
///
/// Binding them here means any mutation of the outer fields changes the AAD the
/// server recomputes from the received outer envelope, so `open_from_recipient`
/// fails the AEAD tag; an unmutated replay is caught by the nonce cache. The
/// encoding is a self-contained canonical CBOR array with a domain tag so it
/// can never collide with the plain-signature AAD.
///
/// # Fail-closed
///
/// Serialization of this fixed integer/bytes shape into an unbounded `Vec` is
/// infallible in practice, but on the off chance it errors this returns `Err`
/// rather than substituting the weaker plain signing AAD — a silent downgrade
/// there would reintroduce the exact replay bypass this binding closes. Both the
/// seal and open callers propagate the error, so the request simply fails.
pub(crate) fn encrypted_envelope_external_aad(
    request_id: u64,
    iat: i64,
    nonce: &[u8; 16],
) -> EnvelopeResult<Vec<u8>> {
    use ciborium::value::Value as CborValue;
    let value = CborValue::Array(vec![
        CborValue::Integer(ENVELOPE_SCHEMA_ID.into()),
        CborValue::Integer(REQUEST_ENVELOPE_TYPE_ID.into()),
        CborValue::Text("hykem-req-aad-v1".to_owned()),
        CborValue::Integer(request_id.into()),
        CborValue::Integer(iat.into()),
        CborValue::Bytes(nonce.to_vec()),
    ]);
    let mut buf = Vec::new();
    ciborium::ser::into_writer(&value, &mut buf).map_err(|e| {
        EnvelopeError::Encryption(format!("encode replay-bound #mesh-kem AAD: {e}"))
    })?;
    Ok(buf)
}

/// Serialize `envelope` in stream framing and seal it to `recipient` as a
/// `#mesh-kem` COSE_Encrypt0 with the replay-bound external AAD.
///
/// Single source of truth for both client-side sealing paths
/// ([`RpcClientImpl::sign_envelope`](crate::rpc_client) and
/// [`SignedEnvelope::new_signed_encrypted_mesh_kem`]) so the Cap'n Proto
/// framing and the AAD can never drift between them.
pub(crate) fn seal_request_envelope(
    envelope: &RequestEnvelope,
    recipient: &crate::crypto::hybrid_kem::RecipientPublic,
) -> EnvelopeResult<Vec<u8>> {
    // Standard stream framing (NOT the canonical `to_bytes()` signing form,
    // which omits the segment table) so the decrypt side can `read_message`.
    let mut plaintext = Vec::new();
    {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder = message.init_root::<crate::common_capnp::request_envelope::Builder>();
            envelope.write_to(&mut builder);
        }
        capnp::serialize::write_message(&mut plaintext, &message).map_err(|e| {
            EnvelopeError::Encryption(format!("serialize envelope for encryption: {e}"))
        })?;
    }
    let aad = encrypted_envelope_external_aad(envelope.request_id, envelope.iat, &envelope.nonce)?;
    // One-shot: `seal_to_recipient` performs a FRESH encapsulation per call, so
    // a fixed (epoch=0, seq=0) nonce is safe — the content key is unique per call.
    crate::crypto::cose_encrypt::seal_to_recipient(recipient, &plaintext, &aad, 0, 0)
        .map_err(|e| EnvelopeError::Encryption(format!("#mesh-kem envelope seal failed: {e}")))
}

/// Build the COSE external_aad used for all RESPONSE envelope signatures (#275).
///
/// Bound to [`RESPONSE_ENVELOPE_TYPE_ID`] (≠ [`REQUEST_ENVELOPE_TYPE_ID`]), this
/// is the load-bearing domain separation: a COSE composite produced for a
/// response cannot verify against the request AAD, and vice-versa.
fn response_envelope_external_aad() -> Vec<u8> {
    crate::crypto::cose_sign1::build_external_aad(ENVELOPE_SCHEMA_ID, RESPONSE_ENVELOPE_TYPE_ID)
}

const MAX_RESPONSE_KEM_RECIPIENT_BYTES: usize = 2 * 1024;
const MAX_ENCRYPTED_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAX_SERVICE_DOMAIN_BYTES: usize = 128;

/// Validate the one canonical RPC service/destination identifier.
///
/// Domains are compared byte-for-byte and never normalized at a trust
/// boundary, avoiding aliases between the client and dispatcher.
pub fn validate_service_domain(service_domain: &str) -> Result<()> {
    let bytes = service_domain.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_SERVICE_DOMAIN_BYTES {
        anyhow::bail!(
            "service domain must contain 1..={} bytes",
            MAX_SERVICE_DOMAIN_BYTES
        );
    }
    if !bytes[0].is_ascii_lowercase() && !bytes[0].is_ascii_digit() {
        anyhow::bail!("service domain must begin with a lowercase ASCII letter or digit");
    }
    if !bytes.iter().all(|byte| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'.' | b'_' | b'-' | b'/')
    }) {
        anyhow::bail!(
            "service domain must use only lowercase ASCII letters, digits, '.', '_', '-', or '/'"
        );
    }
    Ok(())
}

/// Transcript-bound external AAD for the response HyKEM COSE_Encrypt0.
pub(crate) fn encrypted_response_external_aad(
    request_id: u64,
    request_iat: i64,
    request_nonce: &[u8; 16],
    server_identity: &[u8; 32],
    recipient: &crate::crypto::hybrid_kem::RecipientPublic,
    service_domain: &str,
) -> EnvelopeResult<Vec<u8>> {
    use ciborium::value::Value as CborValue;
    use sha2::{Digest, Sha256};

    let recipient_hash = Sha256::digest(recipient.encode());
    let value = CborValue::Array(vec![
        CborValue::Integer(ENVELOPE_SCHEMA_ID.into()),
        CborValue::Integer(RESPONSE_ENVELOPE_TYPE_ID.into()),
        CborValue::Text("hykem-rpc-response-v1".to_owned()),
        CborValue::Integer(request_id.into()),
        CborValue::Integer(request_iat.into()),
        CborValue::Bytes(request_nonce.to_vec()),
        CborValue::Bytes(server_identity.to_vec()),
        CborValue::Bytes(recipient_hash.to_vec()),
        CborValue::Text(service_domain.to_owned()),
    ]);
    let mut buf = Vec::new();
    ciborium::ser::into_writer(&value, &mut buf)
        .map_err(|e| EnvelopeError::Encryption(format!("encode response HyKEM AAD: {e}")))?;
    Ok(buf)
}

/// Resolves the anchored ML-DSA-65 verifying key for an EdDSA signer identity.
///
/// kid-anchoring: the envelope's EdDSA signer key (`cnf`, 32 bytes) is the
/// identity. A `PqTrustStore` maps that identity to its trusted ML-DSA-65 key.
/// The COSE ML-DSA-65 entry must verify against this anchored key (and its kid
/// must match), so an attacker cannot strip + re-sign with their own PQ key.
pub trait PqTrustStore: Send + Sync {
    /// Return the trusted ML-DSA-65 verifying key bound to the given Ed25519
    /// signer public key, or `None` if no binding is known.
    fn ml_dsa_key_for(
        &self,
        ed25519_pubkey: &[u8; 32],
    ) -> Option<crate::crypto::pq::MlDsaVerifyingKey>;
}

/// In-memory kid-anchored ML-DSA-65 trust store mapping an Ed25519 signer
/// identity to its trusted ML-DSA-65 verifying key.
///
/// This is the authoritative anchor for the mesh/streaming/browser SNS
/// composite. Bindings come from the node's own hybrid identity and from
/// attested peer identities. Entries MUST be established out-of-band (peer
/// attestation / trust-store), never from the self-asserted COSE object.
#[derive(Default)]
pub struct KeyedPqTrustStore {
    bindings: std::collections::HashMap<[u8; 32], Vec<u8>>,
}

impl KeyedPqTrustStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            bindings: std::collections::HashMap::new(),
        }
    }

    /// Bind an Ed25519 signer identity to its trusted ML-DSA-65 verifying key
    /// (stored as raw vk bytes; re-decoded on lookup).
    pub fn bind(
        &mut self,
        ed25519_pubkey: [u8; 32],
        ml_dsa_vk: &crate::crypto::pq::MlDsaVerifyingKey,
    ) {
        self.bindings.insert(
            ed25519_pubkey,
            crate::crypto::pq::ml_dsa_vk_bytes(ml_dsa_vk),
        );
    }

    /// Number of bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether the store has no bindings.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl PqTrustStore for KeyedPqTrustStore {
    fn ml_dsa_key_for(
        &self,
        ed25519_pubkey: &[u8; 32],
    ) -> Option<crate::crypto::pq::MlDsaVerifyingKey> {
        self.bindings
            .get(ed25519_pubkey)
            .and_then(|bytes| crate::crypto::pq::ml_dsa_vk_from_bytes(bytes).ok())
    }
}

// ============================================================================
// Process-global envelope verify configuration (closes the fail-open).
// ============================================================================

/// Process-wide envelope verification configuration shared by all production
/// verify sites (`process_request` / `RequestLoop`, `StreamService` register).
///
/// Holds the enforced [`CryptoPolicy`] and the kid-anchored [`PqTrustStore`].
/// The daemon installs this at startup with `Hybrid` + a real store wired from
/// the node's hybrid identity (see `key_rotation`). When unset (libraries, unit
/// tests that don't opt in), verification defaults to `Classical` so unrelated
/// code paths keep working — production code MUST call [`install_verify_config`].
pub struct EnvelopeVerifyConfig {
    pub policy: crate::crypto::CryptoPolicy,
    pub pq_store: Option<std::sync::Arc<dyn PqTrustStore>>,
}

#[cfg(not(target_arch = "wasm32"))]
static VERIFY_CONFIG: std::sync::OnceLock<EnvelopeVerifyConfig> = std::sync::OnceLock::new();

/// Install the process-global envelope verify configuration. First write wins.
///
/// Returns `Err` if a configuration was already installed.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_verify_config(config: EnvelopeVerifyConfig) -> Result<()> {
    VERIFY_CONFIG
        .set(config)
        .map_err(|_| anyhow!("envelope verify config already installed"))
}

/// Whether a process-global verify configuration has been installed.
#[cfg(not(target_arch = "wasm32"))]
pub fn verify_config_installed() -> bool {
    VERIFY_CONFIG.get().is_some()
}

/// The enforced verify policy.
///
/// When a config has been installed, its policy is returned verbatim.
///
/// When **uninstalled**, the default is **fail-closed**: production builds
/// default to [`CryptoPolicy::Hybrid`] so that any verify site reached before
/// `install_verify_config` (subprocess-spawner services, early init) rejects
/// classical-only envelopes rather than silently accepting EdDSA-only ones
/// (#160 — this previously defaulted `Classical`, re-opening the M3 fail-open).
///
/// Under `cfg(test)` the uninstalled default stays `Classical`: in-process unit
/// tests share one `OnceLock` and rely on per-call `UnwrapOptions` overrides
/// rather than a global install. Integration tests (which compile this crate in
/// non-test mode) must call [`install_verify_config`] explicitly.
#[cfg(not(target_arch = "wasm32"))]
pub fn global_verify_policy() -> crate::crypto::CryptoPolicy {
    if let Some(c) = VERIFY_CONFIG.get() {
        return c.policy;
    }
    #[cfg(test)]
    {
        crate::crypto::CryptoPolicy::Classical
    }
    #[cfg(not(test))]
    {
        // Fail-closed default. Loud (once — this is on the per-request verify
        // path), because reaching a verify site with no installed config in
        // production is a wiring bug (#160).
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "envelope verify config not installed; defaulting to fail-closed Hybrid \
                 policy. Production code MUST call install_verify_config() at startup."
            );
        });
        crate::crypto::CryptoPolicy::Hybrid
    }
}

/// The installed kid-anchored PQ trust store, if any.
#[cfg(not(target_arch = "wasm32"))]
pub fn global_pq_store() -> Option<std::sync::Arc<dyn PqTrustStore>> {
    VERIFY_CONFIG.get().and_then(|c| c.pq_store.clone())
}

/// Apply the process-global verify configuration to an `UnwrapOptions`,
/// unless it has already been explicitly downgraded/overridden.
#[cfg(not(target_arch = "wasm32"))]
pub fn apply_global_verify_config<'a>(
    mut opts: UnwrapOptions<'a>,
    store_holder: &'a Option<std::sync::Arc<dyn PqTrustStore>>,
) -> UnwrapOptions<'a> {
    opts.verify_policy = global_verify_policy();
    opts.pq_store = store_holder.as_deref();
    opts
}

/// Name of the mid-rollout escape-hatch env var shared by the request and
/// response verify paths. Setting it to `classical` downgrades BOTH directions
/// (operators staging the PQ rollout before peer ML-DSA bindings are
/// provisioned). Any other value (or unset) means Hybrid.
pub const ENVELOPE_POLICY_ENV: &str = "HYPRSTREAM_ENVELOPE_POLICY";

/// Parse [`ENVELOPE_POLICY_ENV`] into a [`CryptoPolicy`]. Single source of truth
/// for the escape hatch so the request side (`install_verify_config` in
/// `main.rs`) and the response side share identical semantics. Defaults to
/// fail-closed [`CryptoPolicy::Hybrid`]; only the literal `classical` downgrades.
#[cfg(not(target_arch = "wasm32"))]
pub fn envelope_policy_from_env() -> crate::crypto::CryptoPolicy {
    match std::env::var(ENVELOPE_POLICY_ENV).ok().as_deref() {
        Some("classical") => crate::crypto::CryptoPolicy::Classical,
        Some("hybrid") | None => crate::crypto::CryptoPolicy::Hybrid,
        Some(other) => {
            tracing::warn!("unknown {ENVELOPE_POLICY_ENV}={other:?}, defaulting to Hybrid");
            crate::crypto::CryptoPolicy::Hybrid
        }
    }
}

// ============================================================================
// Process-global RESPONSE-envelope verify configuration (#277).
//
// Symmetric to the request-side `EnvelopeVerifyConfig`/`global_verify_policy`
// above, but for the client-side `ResponseEnvelope` verify path. The daemon /
// native RPC client construction installs this at startup with `Hybrid` + the
// admin-anchored mesh PQ trust store (#157), so server-asserted `StreamInfo`
// (dhPublic anchoring + QoS contract) is PQ-attested in production. When
// uninstalled, the default is fail-closed `Hybrid` (mirroring the request side):
// a classical-only / stripped response is rejected rather than silently
// accepted.
//
// This REUSES the same `CryptoPolicy`, `PqTrustStore`, and `KeyedPqTrustStore`
// machinery as the request side — it is a second arm of the same system, not a
// parallel one.
// ============================================================================

/// Process-wide RESPONSE-envelope verify configuration consulted by clients that
/// did not explicitly set a per-client response policy / PQ store (#277).
pub struct ResponseVerifyConfig {
    pub policy: crate::crypto::CryptoPolicy,
    pub pq_store: Option<std::sync::Arc<dyn PqTrustStore>>,
}

#[cfg(not(target_arch = "wasm32"))]
static RESPONSE_VERIFY_CONFIG: std::sync::OnceLock<ResponseVerifyConfig> =
    std::sync::OnceLock::new();

/// Install the process-global RESPONSE verify configuration. First write wins.
///
/// Returns `Err` if a configuration was already installed.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_response_verify_config(config: ResponseVerifyConfig) -> Result<()> {
    RESPONSE_VERIFY_CONFIG
        .set(config)
        .map_err(|_| anyhow!("response verify config already installed"))
}

/// Whether a process-global RESPONSE verify configuration has been installed.
#[cfg(not(target_arch = "wasm32"))]
pub fn response_verify_config_installed() -> bool {
    RESPONSE_VERIFY_CONFIG.get().is_some()
}

/// The enforced RESPONSE-verify policy (consulted when a client did not set one).
///
/// When a config has been installed, its policy is returned verbatim.
///
/// When **uninstalled**, the default is **fail-closed**, symmetric to the
/// request side ([`global_verify_policy`]): production builds default to
/// [`CryptoPolicy::Hybrid`] so a client that reaches the response-verify path
/// before `install_response_verify_config` rejects classical-only responses
/// rather than silently accepting EdDSA-only ones (#277). The escape hatch
/// ([`ENVELOPE_POLICY_ENV`]`=classical`) is honored even in the uninstalled
/// path so the response side downgrades in parity with the request side.
///
/// Under `cfg(test)` the uninstalled default stays `Classical`: in-process unit
/// tests share one `OnceLock` and rely on explicit per-call policy overrides.
#[cfg(not(target_arch = "wasm32"))]
pub fn global_response_verify_policy() -> crate::crypto::CryptoPolicy {
    if let Some(c) = RESPONSE_VERIFY_CONFIG.get() {
        return c.policy;
    }
    #[cfg(test)]
    {
        crate::crypto::CryptoPolicy::Classical
    }
    #[cfg(not(test))]
    {
        // Honor the escape hatch even before install (parity with request side
        // staging) but otherwise fail closed.
        let policy = envelope_policy_from_env();
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "response verify config not installed; defaulting to fail-closed \
                 {policy:?} policy. Production code MUST call \
                 install_response_verify_config() at startup."
            );
        });
        policy
    }
}

/// The installed RESPONSE-side kid-anchored PQ trust store, if any.
#[cfg(not(target_arch = "wasm32"))]
pub fn global_response_pq_store() -> Option<std::sync::Arc<dyn PqTrustStore>> {
    RESPONSE_VERIFY_CONFIG
        .get()
        .and_then(|c| c.pq_store.clone())
}

/// Replay protection cache interface.
///
/// Implementations should maintain a cache of recently seen nonces
/// to prevent replay attacks.
pub trait NonceCache: Send + Sync {
    /// Check if a nonce has been seen before, and if not, mark it as seen.
    ///
    /// Returns `true` if the nonce is fresh (not seen before).
    /// Returns `false` if the nonce has already been used.
    fn check_and_insert(&self, nonce: &[u8; 16]) -> bool;
}

/// Simple in-memory nonce cache with time-based expiration.
///
/// This implementation uses a `HashMap` with timestamps to track seen nonces.
/// Old entries are cleaned up periodically during insertions.
///
/// # Thread Safety
///
/// Uses `parking_lot::RwLock` for concurrent access (non-poisoning).
/// Suitable for moderate load.
/// For high-throughput scenarios, consider a lock-free or sharded implementation.
///
/// # Memory Usage
///
/// At most `max_entries` nonces are stored. Oldest entries are evicted
/// when the limit is reached.
pub struct InMemoryNonceCache {
    /// Map of nonce -> timestamp when it was first seen
    #[cfg(not(target_arch = "wasm32"))]
    seen: parking_lot::RwLock<std::collections::HashMap<[u8; 16], i64>>,
    #[cfg(target_arch = "wasm32")]
    seen: std::sync::RwLock<std::collections::HashMap<[u8; 16], i64>>,
    /// Maximum age for entries (in milliseconds)
    max_age_ms: i64,
    /// Maximum number of entries before cleanup
    max_entries: usize,
}

impl Default for InMemoryNonceCache {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryNonceCache {
    /// Create a new cache with default settings.
    ///
    /// Defaults:
    /// - Max age: 5 minutes (matches `MAX_TIMESTAMP_AGE_MS`)
    /// - Max entries: 100,000
    pub fn new() -> Self {
        Self {
            seen: Default::default(),
            max_age_ms: MAX_TIMESTAMP_AGE_MS,
            max_entries: 100_000,
        }
    }

    /// Create a new cache with custom settings.
    pub fn with_config(max_age_ms: i64, max_entries: usize) -> Self {
        Self {
            seen: Default::default(),
            max_age_ms,
            max_entries,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_lock(
        &self,
    ) -> parking_lot::RwLockReadGuard<'_, std::collections::HashMap<[u8; 16], i64>> {
        self.seen.read()
    }

    #[cfg(target_arch = "wasm32")]
    fn read_lock(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, std::collections::HashMap<[u8; 16], i64>> {
        self.seen.read().expect("nonce cache lock poisoned")
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_lock(
        &self,
    ) -> parking_lot::RwLockWriteGuard<'_, std::collections::HashMap<[u8; 16], i64>> {
        self.seen.write()
    }

    #[cfg(target_arch = "wasm32")]
    fn write_lock(
        &self,
    ) -> std::sync::RwLockWriteGuard<'_, std::collections::HashMap<[u8; 16], i64>> {
        self.seen.write().expect("nonce cache lock poisoned")
    }

    /// Remove expired entries.
    fn cleanup(&self, now: i64) {
        let mut seen = self.write_lock();

        // Remove entries older than max_age
        seen.retain(|_, timestamp| now - *timestamp <= self.max_age_ms);

        // If still too many entries, remove oldest
        if seen.len() > self.max_entries {
            let mut entries: Vec<_> = seen.iter().map(|(k, v)| (*k, *v)).collect();
            entries.sort_by_key(|(_, ts)| *ts);

            // Keep only the newest max_entries/2 entries
            let to_keep = self.max_entries / 2;
            for (key, _) in entries.iter().take(entries.len().saturating_sub(to_keep)) {
                seen.remove(key);
            }
        }
    }
}

impl NonceCache for InMemoryNonceCache {
    fn check_and_insert(&self, nonce: &[u8; 16]) -> bool {
        let now = current_timestamp();

        // Fast path: check if already seen (read lock)
        {
            let seen = self.read_lock();
            if seen.contains_key(nonce) {
                return false;
            }
        }

        // Slow path: insert (write lock)
        let mut seen = self.write_lock();

        // Double-check after acquiring write lock
        if seen.contains_key(nonce) {
            return false;
        }

        // Insert the nonce
        seen.insert(*nonce, now);

        // Cleanup if we have too many entries
        if seen.len() > self.max_entries {
            drop(seen); // Release lock before cleanup
            self.cleanup(now);
        }

        true
    }
}

/// Maximum age for a valid request timestamp (5 minutes).
pub const MAX_TIMESTAMP_AGE_MS: i64 = 5 * 60 * 1000;

/// Maximum clock skew tolerance (30 seconds into the future).
pub const MAX_CLOCK_SKEW_MS: i64 = 30 * 1000;

impl SignedEnvelope {
    /// Create and sign a new envelope.
    ///
    /// The signature covers the Cap'n Proto serialized bytes of the envelope.
    /// If the authorization is `IdJag` and `wth` is not already set, `wth`
    /// is auto-populated as SHA-256(jwt) per the WIMSE wth binding.
    pub fn new_signed(envelope: RequestEnvelope, signing_key: &SigningKey) -> Self {
        // Default policy is Classical for the bare Ed25519-only constructor so
        // that callers that don't supply a PQ key produce verifiable envelopes.
        Self::new_signed_with_policy(
            envelope,
            signing_key,
            None,
            crate::crypto::CryptoPolicy::Classical,
        )
    }

    /// Create and dual-sign a new envelope with Ed25519 + ML-DSA-65 (Hybrid).
    pub fn new_signed_hybrid(
        envelope: RequestEnvelope,
        signing_key: &SigningKey,
        pq_signing_key: &crate::crypto::pq::MlDsaSigningKey,
    ) -> Self {
        Self::new_signed_with_policy(
            envelope,
            signing_key,
            Some(pq_signing_key),
            crate::crypto::CryptoPolicy::Hybrid,
        )
    }

    /// Create and sign a new envelope under an explicit [`CryptoPolicy`].
    ///
    /// - `Classical`: emits a single-EdDSA COSE composite. `pq_signing_key` is
    ///   ignored.
    /// - `Hybrid`: emits an EdDSA + ML-DSA-65 COSE composite. `pq_signing_key`
    ///   MUST be `Some`; if `None`, falls back to Classical (defensive).
    ///
    /// `sig`/`cnf` are always populated with the raw EdDSA signature + signer
    /// public key for the cnf key-binding path.
    pub fn new_signed_with_policy(
        mut envelope: RequestEnvelope,
        signing_key: &SigningKey,
        pq_signing_key: Option<&crate::crypto::pq::MlDsaSigningKey>,
        policy: crate::crypto::CryptoPolicy,
    ) -> Self {
        if envelope.wth.is_none() {
            if let Some(jwt) = envelope.jwt_token() {
                use sha2::{Digest, Sha256};
                envelope.wth = Some(Sha256::digest(jwt.as_bytes()).into());
            }
        }

        let envelope_bytes = envelope.to_bytes();
        let signature = signing_key.sign(&envelope_bytes);
        // SECURITY: no empty-cose fallback. A COSE build failure is a
        // should-never-happen crypto-encoding error (CBOR-encoding fixed-shape
        // COSE_Sign1 structures over valid keys); fail loud rather than silently
        // emit an empty (and thus potentially fail-open) composite.
        #[allow(clippy::expect_used)]
        let cose = Self::build_cose(signing_key, pq_signing_key, policy, &envelope_bytes)
            .expect("COSE composite signing must not fail for valid keys");

        Self {
            envelope,
            sig: signature.to_bytes(),
            cnf: signing_key.verifying_key().to_bytes(),
            encrypted_envelope: None,
            client_ephemeral_public: None,
            cose,
            policy,
            pq_kem_ciphertext: None,
        }
    }

    /// Build the nested COSE composite signature for the given signing-data per
    /// policy. Returns `Err` on encoding failure — callers MUST NOT substitute
    /// an empty cose (that would fail open at verify time).
    fn build_cose(
        signing_key: &SigningKey,
        pq_signing_key: Option<&crate::crypto::pq::MlDsaSigningKey>,
        policy: crate::crypto::CryptoPolicy,
        signing_data: &[u8],
    ) -> Result<Vec<u8>> {
        let pq = if policy.uses_pq() {
            pq_signing_key
        } else {
            None
        };
        let aad = envelope_external_aad();
        crate::crypto::cose_sign::sign_composite(signing_key, pq, signing_data, &aad)
    }

    /// Create, hybrid-encrypt (HyKEM `#mesh-kem` → COSE_Encrypt0), and dual-sign
    /// (EdDSA + ML-DSA-65) a new envelope — the fail-closed hybrid-PQ path (#555 / S4).
    ///
    /// The serialized `RequestEnvelope` is sealed with
    /// [`crate::crypto::cose_encrypt::seal_to_recipient`] to the server's anchored
    /// `#mesh-kem` [`crate::crypto::hybrid_kem::RecipientPublic`] (resolved
    /// out-of-band via a [`crate::crypto::hybrid_kem::KemTrustStore`], NEVER
    /// self-asserted). The KEM material rides inside the COSE_Encrypt0 (`ek`
    /// header), so the COSE composite signature simply covers the COSE_Encrypt0
    /// bytes.
    ///
    /// This replaces the retired `envelope_crypto` X25519-static scheme, which
    /// reused the Ed25519 *signing* key as a *static* X25519 KEX key (key-reuse +
    /// no forward secrecy + fixed-zero-nonce AEAD). Forward secrecy now comes from
    /// the ephemeral X25519 leg of HyKEM and from rotated `#mesh-kem` prekeys
    /// (S1 `KemPrekey`).
    pub fn new_signed_encrypted_mesh_kem(
        mut envelope: RequestEnvelope,
        signing_key: &SigningKey,
        pq_signing_key: &crate::crypto::pq::MlDsaSigningKey,
        server_kem_public: &crate::crypto::hybrid_kem::RecipientPublic,
    ) -> EnvelopeResult<Self> {
        if envelope.wth.is_none() {
            if let Some(jwt) = envelope.jwt_token() {
                use sha2::{Digest, Sha256};
                envelope.wth = Some(Sha256::digest(jwt.as_bytes()).into());
            }
        }

        // Serialize + seal via the shared helper so the framing and the
        // replay-bound external AAD stay identical to the client path.
        let cose_ct = seal_request_envelope(&envelope, server_kem_public)?;

        // The COSE_Encrypt0 bytes are the signing data — the KEM ciphertexts and
        // suite id are self-described inside the COSE object.
        let signature = signing_key.sign(&cose_ct);
        let cose = Self::build_cose(
            signing_key,
            Some(pq_signing_key),
            crate::crypto::CryptoPolicy::Hybrid,
            &cose_ct,
        )
        .map_err(|e| EnvelopeError::Encryption(format!("COSE composite signing failed: {e}")))?;

        Ok(Self {
            envelope,
            sig: signature.to_bytes(),
            cnf: signing_key.verifying_key().to_bytes(),
            encrypted_envelope: Some(cose_ct),
            // reserved — KEM material lives in the COSE_Encrypt0 `ek` header now.
            client_ephemeral_public: None,
            cose,
            policy: crate::crypto::CryptoPolicy::Hybrid,
            pq_kem_ciphertext: None,
        })
    }

    /// Returns true if this envelope uses the encrypted path.
    pub fn is_encrypted(&self) -> bool {
        self.encrypted_envelope.is_some()
    }

    /// Outer placeholder serialized beside `encrypted_envelope`.
    ///
    /// The outer envelope keeps only non-secret replay metadata until
    /// authenticated decryption succeeds. Payload, JWT authorization,
    /// delegation bearer, WTH, and streaming DH material live only inside the
    /// COSE_Encrypt0 plaintext and replace this placeholder after decrypt.
    fn redacted_encrypted_envelope(&self) -> RequestEnvelope {
        RequestEnvelope {
            request_id: self.envelope.request_id,
            payload: Vec::new(),
            iat: self.envelope.iat,
            nonce: self.envelope.nonce,
            authorization: Authorization::None,
            delegation_token: None,
            wth: None,
            client_dh_public: None,
            client_kem_public: None,
            response_kem_recipient: None,
            service_domain: None,
        }
    }

    /// Verify the signature, then atomically commit replay protection.
    ///
    /// # Verification Steps
    ///
    /// 1. Check timestamp is within the acceptable window without mutation
    /// 2. Verify signer pubkey matches expected key
    /// 3. Verify Ed25519 signature
    /// 4. Atomically check and record the nonce
    ///
    /// # Errors
    ///
    /// - `EnvelopeError::InvalidPublicKey` if signer doesn't match expected
    /// - `EnvelopeError::ReplayAttack` if timestamp or nonce check fails
    /// - `EnvelopeError::InvalidSignature` if signature verification fails
    pub fn verify(
        &self,
        expected_pubkey: &VerifyingKey,
        nonce_cache: &dyn NonceCache,
    ) -> EnvelopeResult<()> {
        self.verify_with(
            expected_pubkey,
            nonce_cache,
            None,
            crate::crypto::CryptoPolicy::Classical,
        )
    }

    /// Verify with an explicit kid-anchored PQ trust store and verify policy.
    ///
    /// - `pq_store`: when `Some`, resolves the trust-anchored ML-DSA-65 key for
    ///   the envelope's EdDSA signer (`cnf`). The COSE ML-DSA-65 entry must
    ///   verify against this key and its kid must match (kid-anchoring) —
    ///   fixing the self-certification weakness.
    /// - `verify_policy`: `Hybrid` is Weakly Non-Separable (per-identity): for a
    ///   signer with an anchored ML-DSA-65 key it ENFORCES the outer layer
    ///   (rejecting stripped-outer, forged, and self-asserted PQ keys); for an
    ///   unanchored signer it falls back to verifying the inner EdDSA (classical
    ///   floor) rather than failing closed. `Classical` verifies only EdDSA and
    ///   SKIPS any PQ entry (RFC 7517 skip-unknown interop).
    pub fn verify_with(
        &self,
        expected_pubkey: &VerifyingKey,
        nonce_cache: &dyn NonceCache,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
    ) -> EnvelopeResult<()> {
        self.validate_timestamp()?;
        // 1. Verify signer matches expected (constant-time comparison)
        if !bool::from(self.cnf.ct_eq(&expected_pubkey.to_bytes())) {
            return Err(EnvelopeError::SignerMismatch {
                expected: hex::encode(expected_pubkey.to_bytes()),
                actual: hex::encode(self.cnf),
            });
        }

        self.verify_cose(expected_pubkey, pq_store, verify_policy)?;
        self.check_replay(nonce_cache)
    }

    /// Verify signature only (skip replay protection).
    ///
    /// Use this for testing or when replay protection is handled elsewhere.
    pub fn verify_signature_only(&self, expected_pubkey: &VerifyingKey) -> EnvelopeResult<()> {
        if !bool::from(self.cnf.ct_eq(&expected_pubkey.to_bytes())) {
            return Err(EnvelopeError::SignerMismatch {
                expected: hex::encode(expected_pubkey.to_bytes()),
                actual: hex::encode(self.cnf),
            });
        }
        self.verify_cose(
            expected_pubkey,
            None,
            crate::crypto::CryptoPolicy::Classical,
        )
    }

    /// Verify signature only under an explicit policy + PQ trust store.
    pub fn verify_signature_only_with(
        &self,
        expected_pubkey: &VerifyingKey,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
    ) -> EnvelopeResult<()> {
        if !bool::from(self.cnf.ct_eq(&expected_pubkey.to_bytes())) {
            return Err(EnvelopeError::SignerMismatch {
                expected: hex::encode(expected_pubkey.to_bytes()),
                actual: hex::encode(self.cnf),
            });
        }
        self.verify_cose(expected_pubkey, pq_store, verify_policy)
    }

    /// Verify against the envelope's own embedded signer pubkey.
    ///
    /// For WebTransport clients that sign with their own keypair rather than
    /// a shared server key. Still checks timestamp and nonce for replay protection.
    pub fn verify_any_signer(&self, nonce_cache: &dyn NonceCache) -> EnvelopeResult<()> {
        self.verify_any_signer_with(nonce_cache, None, crate::crypto::CryptoPolicy::Classical)
    }

    /// `verify_any_signer` with an explicit kid-anchored PQ trust store + policy.
    pub fn verify_any_signer_with(
        &self,
        nonce_cache: &dyn NonceCache,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
    ) -> EnvelopeResult<()> {
        self.validate_timestamp()?;
        let verifying_key =
            VerifyingKey::from_bytes(&self.cnf).map_err(|_| EnvelopeError::InvalidPublicKey {
                expected: 32,
                actual: 0,
            })?;
        self.verify_cose(&verifying_key, pq_store, verify_policy)?;
        self.check_replay(nonce_cache)
    }

    /// Fu2/#677: like [`Self::verify_with`] but with an explicit
    /// [`UnanchoredHybridPolicy`] + classical-floor allowlist governing how an
    /// *unanchored* signer (no ML-DSA-65 binding in `pq_store`) is treated under
    /// Hybrid. This is the production wire-verify path
    /// ([`unwrap_and_verify`]); the default posture is deny-on-missing-anchor.
    pub fn verify_with_unanchored_policy(
        &self,
        expected_pubkey: &VerifyingKey,
        nonce_cache: &dyn NonceCache,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
        unanchored: UnanchoredHybridPolicy,
        allowlist: &[String],
    ) -> EnvelopeResult<()> {
        self.validate_timestamp()?;
        if !bool::from(self.cnf.ct_eq(&expected_pubkey.to_bytes())) {
            return Err(EnvelopeError::SignerMismatch {
                expected: hex::encode(expected_pubkey.to_bytes()),
                actual: hex::encode(self.cnf),
            });
        }
        self.verify_cose_unanchored(
            expected_pubkey,
            pq_store,
            verify_policy,
            unanchored,
            allowlist,
        )?;
        self.check_replay(nonce_cache)
    }

    /// Fu2/#677: any-signer variant of [`Self::verify_with_unanchored_policy`].
    pub fn verify_any_signer_with_unanchored_policy(
        &self,
        nonce_cache: &dyn NonceCache,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
        unanchored: UnanchoredHybridPolicy,
        allowlist: &[String],
    ) -> EnvelopeResult<()> {
        self.validate_timestamp()?;
        let verifying_key =
            VerifyingKey::from_bytes(&self.cnf).map_err(|_| EnvelopeError::InvalidPublicKey {
                expected: 32,
                actual: 0,
            })?;
        self.verify_cose_unanchored(
            &verifying_key,
            pq_store,
            verify_policy,
            unanchored,
            allowlist,
        )?;
        self.check_replay(nonce_cache)
    }

    fn verify_signature_with_unanchored_policy(
        &self,
        expected_pubkey: &VerifyingKey,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
        unanchored: UnanchoredHybridPolicy,
        allowlist: &[String],
    ) -> EnvelopeResult<()> {
        if !bool::from(self.cnf.ct_eq(&expected_pubkey.to_bytes())) {
            return Err(EnvelopeError::SignerMismatch {
                expected: hex::encode(expected_pubkey.to_bytes()),
                actual: hex::encode(self.cnf),
            });
        }
        self.verify_cose_unanchored(
            expected_pubkey,
            pq_store,
            verify_policy,
            unanchored,
            allowlist,
        )
    }

    fn verify_any_signature_with_unanchored_policy(
        &self,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
        unanchored: UnanchoredHybridPolicy,
        allowlist: &[String],
    ) -> EnvelopeResult<()> {
        let verifying_key =
            VerifyingKey::from_bytes(&self.cnf).map_err(|_| EnvelopeError::InvalidPublicKey {
                expected: 32,
                actual: 0,
            })?;
        self.verify_cose_unanchored(
            &verifying_key,
            pq_store,
            verify_policy,
            unanchored,
            allowlist,
        )
    }

    /// Non-mutating timestamp-window validation for early rejection before
    /// signature verification, KEM decapsulation, or decryption.
    fn validate_timestamp(&self) -> EnvelopeResult<()> {
        Self::validate_timestamp_at(self.envelope.iat, current_timestamp())
    }

    /// Validate `iat` against an explicit clock reading.
    ///
    /// Widen both wire-controlled timestamps before subtracting so every `i64`
    /// input, including the two extremes, has defined fail-closed behavior.
    /// Both limits are inclusive: an age exactly equal to the past window or a
    /// future distance exactly equal to the clock-skew allowance is accepted.
    fn validate_timestamp_at(iat: i64, now: i64) -> EnvelopeResult<()> {
        let age = i128::from(now) - i128::from(iat);
        let max_age = i128::from(MAX_TIMESTAMP_AGE_MS);
        if age > max_age {
            return Err(EnvelopeError::ReplayAttack(format!(
                "timestamp too old: {age}ms > {MAX_TIMESTAMP_AGE_MS}ms"
            )));
        }

        let future_distance = i128::from(iat) - i128::from(now);
        let max_future = i128::from(MAX_CLOCK_SKEW_MS);
        if future_distance > max_future {
            let excess = future_distance - max_future;
            return Err(EnvelopeError::ReplayAttack(format!(
                "timestamp in future: {excess}ms beyond clock skew tolerance"
            )));
        }
        Ok(())
    }

    /// Revalidate the timestamp, then atomically check and record the nonce.
    fn check_replay(&self, nonce_cache: &dyn NonceCache) -> EnvelopeResult<()> {
        // Recheck at commit time so work performed near a timestamp-window
        // boundary cannot admit an entry after it expires.
        self.validate_timestamp()?;
        if !nonce_cache.check_and_insert(&self.envelope.nonce) {
            return Err(EnvelopeError::ReplayAttack("nonce already seen".to_owned()));
        }
        Ok(())
    }

    /// Verify the COSE composite signature (authoritative auth check).
    ///
    /// The raw EdDSA `sig`/`cnf` advertisement is NOT trusted on its own; this
    /// re-verifies the EdDSA component inside the COSE composite against
    /// `ed_vk`, and (under Hybrid policy) the kid-anchored ML-DSA-65 component.
    fn verify_cose(
        &self,
        ed_vk: &VerifyingKey,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
    ) -> EnvelopeResult<()> {
        let signing_data = self.signed_bytes();
        let aad = envelope_external_aad();

        // kid-anchor: resolve the trusted ML-DSA-65 key for this EdDSA identity.
        let anchored_pq = pq_store.and_then(|s| s.ml_dsa_key_for(&self.cnf));

        // WNS posture (draft-ietf-pquip-hybrid-signature-spectrums): the composite is
        // Weakly Non-Separable — the inner EdDSA is independently verifiable, so PQ
        // enforcement is applied PER-IDENTITY. Require the ML-DSA-65 outer ONLY for a
        // signer whose PQ key we have anchored out-of-band; for an unanchored signer,
        // fall back to the inner EdDSA (classical floor) rather than failing closed.
        // Safe because ed_vk is derived from this same `cnf` (the PQ-lookup identity ==
        // the EdDSA-verified identity), so an anchored identity cannot be downgraded by
        // spoofing cnf, while an unanchored one is no weaker than classical. PQ is NEVER
        // resolved from the self-asserted COSE entry (that is the self-cert weakness).
        let require_pq = verify_policy.uses_pq() && anchored_pq.is_some();
        #[cfg(not(target_arch = "wasm32"))]
        if verify_policy.uses_pq() && anchored_pq.is_none() {
            tracing::debug!(
                "Hybrid policy active but signer has no anchored ML-DSA-65 key; \
                 verifying classical inner EdDSA (WNS backwards-compat)"
            );
        }

        crate::crypto::cose_sign::verify_composite(
            &self.cose,
            ed_vk,
            anchored_pq.as_ref(),
            &signing_data,
            &aad,
            require_pq,
        )
        .map_err(|e| EnvelopeError::PqSignatureInvalid(e.to_string()))?;

        Ok(())
    }

    /// Fu2/#677: COSE composite verify with an explicit
    /// [`UnanchoredHybridPolicy`] + classical-floor allowlist.
    ///
    /// Mirrors [`Self::verify_cose`] but, under a Hybrid policy, no longer
    /// accepts an unanchored identity classical-only by default: with
    /// [`UnanchoredHybridPolicy::Deny`] (the production default) an unanchored
    /// signer is rejected unless its Ed25519 key hex is in `allowlist`. This
    /// closes the audit's "hybrid is anchor-coverage-dependent" gap — the
    /// default is now deny-on-missing-anchor, matching the UCAN verifier.
    fn verify_cose_unanchored(
        &self,
        ed_vk: &VerifyingKey,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
        unanchored: UnanchoredHybridPolicy,
        allowlist: &[String],
    ) -> EnvelopeResult<()> {
        let signing_data = self.signed_bytes();
        let aad = envelope_external_aad();
        let anchored_pq = pq_store.and_then(|s| s.ml_dsa_key_for(&self.cnf));

        let require_pq = if verify_policy.uses_pq() && anchored_pq.is_none() {
            // Unanchored identity under Hybrid: gate the classical floor.
            match unanchored {
                UnanchoredHybridPolicy::AllowClassicalFloor => {
                    #[cfg(not(target_arch = "wasm32"))]
                    tracing::debug!(
                        "Hybrid policy active but signer has no anchored ML-DSA-65 key; \
                         verifying classical inner EdDSA (explicit AllowClassicalFloor)"
                    );
                    false
                }
                UnanchoredHybridPolicy::Deny => {
                    let signer_hex = hex::encode(ed_vk.to_bytes());
                    if allowlist.iter().any(|a| a == &signer_hex) {
                        // Explicitly-allowlisted legacy peer: classical floor.
                        false
                    } else {
                        return Err(EnvelopeError::PqSignatureInvalid(
                            "Hybrid policy requires an anchored ML-DSA-65 key for this signer; \
                             the identity is unanchored and not on the classical-floor allowlist \
                             (deny-on-missing-anchor, Fu2/#677)"
                                .to_owned(),
                        ));
                    }
                }
            }
        } else {
            // Anchored identity (require the outer under Hybrid) or Classical policy.
            verify_policy.uses_pq() && anchored_pq.is_some()
        };

        crate::crypto::cose_sign::verify_composite(
            &self.cose,
            ed_vk,
            anchored_pq.as_ref(),
            &signing_data,
            &aad,
            require_pq,
        )
        .map_err(|e| EnvelopeError::PqSignatureInvalid(e.to_string()))?;
        Ok(())
    }

    /// Compute the bytes that were signed.
    ///
    /// Encrypted mode: the COSE_Encrypt0 bytes (`encrypted_envelope`) — the KEM
    /// ciphertexts and suite id are self-described inside that object.
    /// Cleartext mode: `canonical(envelope)`.
    fn signed_bytes(&self) -> Vec<u8> {
        match &self.encrypted_envelope {
            Some(ct) => ct.clone(),
            None => self.envelope.to_bytes(),
        }
    }

    /// Decrypt a `#mesh-kem`-encrypted envelope in-place, replacing `envelope`.
    ///
    /// Opens the COSE_Encrypt0 in `encrypted_envelope` with the node's `#mesh-kem`
    /// [`crate::crypto::hybrid_kem::RecipientKeypair`] (deterministically derived
    /// from the node Ed25519 key — see
    /// [`crate::node_identity::derive_mesh_kem_recipient`]). The encrypted bytes
    /// remain for signature verification.
    pub fn decrypt_in_place_mesh_kem(
        &mut self,
        node_kem_keypair: &crate::crypto::hybrid_kem::RecipientKeypair,
    ) -> EnvelopeResult<()> {
        let ct = self
            .encrypted_envelope
            .as_ref()
            .ok_or_else(|| EnvelopeError::Decryption("no encrypted envelope present".into()))?;
        // Recompute the AAD from the OUTER replay metadata that the server will
        // commit only after this method succeeds. This binds those unsigned
        // fields into the AEAD: a network attacker that mutates the outer
        // nonce/iat changes this AAD and the open fails before replay commit.
        let aad = encrypted_envelope_external_aad(
            self.envelope.request_id,
            self.envelope.iat,
            &self.envelope.nonce,
        )?;

        let plaintext =
            crate::crypto::cose_encrypt::open_from_recipient(node_kem_keypair, ct, &aad, 0, 0)
                .map_err(|e| {
                    EnvelopeError::Decryption(format!("#mesh-kem envelope open failed: {e}"))
                })?;

        let reader = read_exact_envelope_message(&plaintext)
            .map_err(|e| EnvelopeError::Decryption(format!("capnp parse after decrypt: {e}")))?;
        let env_reader = reader
            .get_root::<crate::common_capnp::request_envelope::Reader>()
            .map_err(|e| EnvelopeError::Decryption(format!("envelope read after decrypt: {e}")))?;
        let inner = RequestEnvelope::read_from(env_reader).map_err(|e| {
            EnvelopeError::Decryption(format!("envelope decode after decrypt: {e}"))
        })?;

        // Defense in depth: the replay metadata the server verified (outer) MUST
        // equal the authenticated inner metadata, so the request that gets
        // authorized is the same one whose nonce/timestamp will be replay-checked.
        // The AAD binding above guarantees outer == what-was-sealed on success;
        // this additionally rejects a sender that sealed inner fields diverging
        // from the outer envelope it presented for replay admission.
        if inner.request_id != self.envelope.request_id
            || inner.iat != self.envelope.iat
            || inner.nonce != self.envelope.nonce
        {
            return Err(EnvelopeError::Decryption(
                "inner/outer replay metadata mismatch after #mesh-kem decrypt".into(),
            ));
        }
        self.envelope = inner;

        Ok(())
    }

    /// Get the request ID from the inner envelope.
    pub fn request_id(&self) -> u64 {
        self.envelope.request_id
    }

    /// Get the payload from the inner envelope.
    pub fn payload(&self) -> &[u8] {
        &self.envelope.payload
    }

    /// Get the authorization context.
    pub fn authorization(&self) -> &Authorization {
        &self.envelope.authorization
    }
}

impl ToCapnp for RequestEnvelope {
    type Builder<'a> = common_capnp::request_envelope::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_request_id(self.request_id);
        builder.set_payload(&self.payload);
        builder.set_iat(self.iat);
        builder.set_nonce(&self.nonce);
        self.authorization
            .write_to(&mut builder.reborrow().init_authorization());
        if let Some(ref token) = self.delegation_token {
            builder.set_delegation_token(token);
        }
        if let Some(ref hash) = self.wth {
            builder.set_wth(hash);
        }
        if let Some(ref key) = self.client_dh_public {
            builder.set_client_dh_public(key);
        }
        if let Some(ref recipient) = self.client_kem_public {
            builder.set_client_kem_public(&recipient.encode());
        }
        if let Some(ref recipient) = self.response_kem_recipient {
            builder.set_response_kem_recipient(&recipient.encode());
        }
        if let Some(ref service_domain) = self.service_domain {
            builder.set_service_domain(service_domain);
        }
    }
}

impl FromCapnp for RequestEnvelope {
    type Reader<'a> = common_capnp::request_envelope::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let nonce = {
            let data = reader.get_nonce()?;
            if data.len() != 16 {
                return Err(anyhow!(
                    "Invalid nonce length: expected 16, got {}",
                    data.len()
                ));
            }
            let mut arr = [0u8; 16];
            arr.copy_from_slice(data);
            arr
        };

        let authorization = Authorization::read_from(reader.get_authorization()?)?;

        let delegation_token = {
            if reader.has_delegation_token() {
                let t = reader.get_delegation_token()?.to_str()?;
                if t.is_empty() {
                    None
                } else {
                    Some(t.to_owned())
                }
            } else {
                None
            }
        };

        let wth = {
            let data = reader.get_wth()?;
            if data.is_empty() {
                None
            } else if data.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(data);
                Some(arr)
            } else {
                return Err(anyhow!(
                    "Invalid wth length: expected 32, got {}",
                    data.len()
                ));
            }
        };

        let client_dh_public = {
            let data = reader.get_client_dh_public()?;
            if data.is_empty() {
                None
            } else if data.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(data);
                Some(arr)
            } else {
                return Err(anyhow!(
                    "Invalid clientDhPublic length: expected 32, got {}",
                    data.len()
                ));
            }
        };

        let client_kem_public = {
            let data = reader.get_client_kem_public()?;
            if data.is_empty() {
                None
            } else {
                if data.len() > MAX_RESPONSE_KEM_RECIPIENT_BYTES {
                    return Err(anyhow!(
                        "clientKemPublic exceeds {} byte limit",
                        MAX_RESPONSE_KEM_RECIPIENT_BYTES
                    ));
                }
                let recipient = crate::crypto::hybrid_kem::RecipientPublic::decode(data)
                    .map_err(|e| anyhow!("invalid clientKemPublic: {e}"))?;
                if recipient.suite_id != crate::stream_epoch::IDENTIFIED_STREAM_SUITE {
                    return Err(anyhow!(
                        "clientKemPublic does not use the pinned identified-stream suite"
                    ));
                }
                Some(recipient)
            }
        };

        let response_kem_recipient = {
            let data = reader.get_response_kem_recipient()?;
            if data.is_empty() {
                None
            } else {
                if data.len() > MAX_RESPONSE_KEM_RECIPIENT_BYTES {
                    return Err(anyhow!(
                        "responseKemRecipient exceeds {} byte limit",
                        MAX_RESPONSE_KEM_RECIPIENT_BYTES
                    ));
                }
                Some(
                    crate::crypto::hybrid_kem::RecipientPublic::decode(data)
                        .map_err(|e| anyhow!("invalid responseKemRecipient: {e}"))?,
                )
            }
        };

        let service_domain = if reader.has_service_domain() {
            let value = reader.get_service_domain()?.to_str()?.to_owned();
            validate_service_domain(&value)?;
            Some(value)
        } else {
            None
        };

        Ok(Self {
            request_id: reader.get_request_id(),
            payload: reader.get_payload()?.to_vec(),
            iat: reader.get_iat(),
            nonce,
            authorization,
            delegation_token,
            wth,
            client_dh_public,
            client_kem_public,
            response_kem_recipient,
            service_domain,
        })
    }
}

impl ToCapnp for SignedEnvelope {
    type Builder<'a> = common_capnp::signed_envelope::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        if self.encrypted_envelope.is_some() {
            self.redacted_encrypted_envelope()
                .write_to(&mut builder.reborrow().init_envelope());
        } else {
            self.envelope
                .write_to(&mut builder.reborrow().init_envelope());
        }
        builder.set_sig(&self.sig);
        builder.set_cnf(&self.cnf);
        if let Some(ref ct) = self.encrypted_envelope {
            builder.set_encrypted_envelope(ct);
        }
        if let Some(ref eph) = self.client_ephemeral_public {
            builder.set_client_ephemeral_public(eph);
        }
        builder.set_cose(&self.cose);
        if let Some(ref kem_ct) = self.pq_kem_ciphertext {
            builder.set_pq_kem_ciphertext(kem_ct);
        }
    }
}

impl FromCapnp for SignedEnvelope {
    type Reader<'a> = common_capnp::signed_envelope::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let sig = {
            let data = reader.get_sig()?;
            if data.len() != 64 {
                return Err(anyhow!(
                    "Invalid sig length: expected 64, got {}",
                    data.len()
                ));
            }
            let mut arr = [0u8; 64];
            arr.copy_from_slice(data);
            arr
        };

        let cnf = {
            let data = reader.get_cnf()?;
            if data.len() != 32 {
                return Err(anyhow!(
                    "Invalid cnf length: expected 32, got {}",
                    data.len()
                ));
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(data);
            arr
        };

        let encrypted_envelope = {
            let data = reader.get_encrypted_envelope()?;
            if data.is_empty() {
                None
            } else {
                Some(data.to_vec())
            }
        };

        let client_ephemeral_public = {
            let data = reader.get_client_ephemeral_public()?;
            if data.is_empty() {
                None
            } else if data.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(data);
                Some(arr)
            } else {
                return Err(anyhow!(
                    "Invalid clientEphemeralPublic length: expected 32, got {}",
                    data.len()
                ));
            }
        };

        let cose = reader.get_cose()?.to_vec();

        let pq_kem_ciphertext = {
            let data = reader.get_pq_kem_ciphertext()?;
            if data.is_empty() {
                None
            } else {
                Some(data.to_vec())
            }
        };

        // `policy` is a signing-time concept; the verifier supplies the verify
        // policy explicitly, so decode to the default here.
        Ok(Self {
            envelope: RequestEnvelope::read_from(reader.get_envelope()?)?,
            sig,
            cnf,
            encrypted_envelope,
            client_ephemeral_public,
            cose,
            policy: crate::crypto::CryptoPolicy::default(),
            pq_kem_ciphertext,
        })
    }
}

/// Signed response envelope for E2E authenticated responses.
///
/// All RPC responses are signed to prevent MITM attacks on response data
/// (e.g., server's DH public key in StreamInfo).
///
/// # Security (#275 — Hybrid parity with `SignedEnvelope`)
///
/// The signing-data is `request_id || payload`, binding the response to a
/// specific request and ensuring the payload hasn't been tampered with. The
/// authoritative authentication mechanism is the COSE composite [`cose`] —
/// one EdDSA entry (Classical) or EdDSA + ML-DSA-65 entries (Hybrid) — over
/// that signing-data, bound to [`RESPONSE_ENVELOPE_TYPE_ID`] via the COSE
/// `external_aad` so it can NEVER verify as a request signature.
///
/// `sig`/`cnf` remain populated with the raw EdDSA signature + signer public
/// key for backward compatibility and signer-pubkey advertisement, but the
/// COSE composite is what `verify*` enforces.
///
/// [`cose`]: ResponseEnvelope::cose
#[derive(Debug, Clone)]
pub struct ResponseEnvelope {
    /// Request ID this response corresponds to
    pub request_id: u64,

    /// Serialized inner response
    pub payload: Vec<u8>,

    /// HyKEM COSE_Encrypt0 carrying a `ResponsePlaintext`. When present the
    /// outer payload is empty and the hybrid signature covers these bytes.
    pub encrypted_response: Option<Vec<u8>>,

    /// Ed25519 signature (64 bytes) over request_id || payload
    pub sig: [u8; 64],

    /// Ed25519 public key of the signer (32 bytes)
    pub cnf: [u8; 32],

    /// #275: CBOR-encoded nested COSE composite signature (detached).
    ///
    /// Authoritative authentication mechanism, mirroring [`SignedEnvelope::cose`].
    /// Carries one EdDSA entry (Classical) or EdDSA + ML-DSA-65 entries (Hybrid)
    /// over the response signing-data (`request_id || payload`), bound to the
    /// distinct [`RESPONSE_ENVELOPE_TYPE_ID`] domain. The ML-DSA-65 verifying key
    /// is resolved by kid from a [`PqTrustStore`] (kid-anchored), not embedded.
    pub cose: Vec<u8>,

    /// Runtime crypto policy used when this envelope was signed.
    pub policy: crate::crypto::CryptoPolicy,
}

impl ResponseEnvelope {
    /// Response signing-data: `request_id (8 bytes LE) || payload`.
    fn signing_data(request_id: u64, payload: &[u8]) -> Vec<u8> {
        let mut data = Vec::with_capacity(8 + payload.len());
        data.extend_from_slice(&request_id.to_le_bytes());
        data.extend_from_slice(payload);
        data
    }

    /// Signature transcript for a sealed response. The expected service is
    /// retained by the pending call rather than trusted from response bytes.
    fn encrypted_signing_data(
        request_id: u64,
        ciphertext: &[u8],
        service_domain: &str,
    ) -> EnvelopeResult<Vec<u8>> {
        validate_service_domain(service_domain)
            .map_err(|e| EnvelopeError::Encryption(e.to_string()))?;
        let mut data = Vec::with_capacity(40 + service_domain.len() + ciphertext.len());
        data.extend_from_slice(b"hykem-rpc-response-signature-v2\0");
        data.extend_from_slice(&request_id.to_le_bytes());
        data.extend_from_slice(&(service_domain.len() as u16).to_le_bytes());
        data.extend_from_slice(service_domain.as_bytes());
        data.extend_from_slice(ciphertext);
        Ok(data)
    }

    /// Create and sign a new response envelope (Classical, EdDSA-only).
    ///
    /// The bare constructor defaults to Classical so callers that don't supply a
    /// PQ key produce verifiable envelopes, mirroring
    /// [`SignedEnvelope::new_signed`].
    pub fn new_signed(request_id: u64, payload: Vec<u8>, signing_key: &SigningKey) -> Self {
        Self::new_signed_with_policy(
            request_id,
            payload,
            signing_key,
            None,
            crate::crypto::CryptoPolicy::Classical,
        )
    }

    /// Create and dual-sign a response with Ed25519 + ML-DSA-65 (Hybrid).
    pub fn new_signed_hybrid(
        request_id: u64,
        payload: Vec<u8>,
        signing_key: &SigningKey,
        pq_signing_key: &crate::crypto::pq::MlDsaSigningKey,
    ) -> Self {
        Self::new_signed_with_policy(
            request_id,
            payload,
            signing_key,
            Some(pq_signing_key),
            crate::crypto::CryptoPolicy::Hybrid,
        )
    }

    /// Create and sign a response under an explicit [`CryptoPolicy`].
    ///
    /// Mirrors [`SignedEnvelope::new_signed_with_policy`]:
    /// - `Classical`: single-EdDSA COSE composite; `pq_signing_key` ignored.
    /// - `Hybrid`: EdDSA + ML-DSA-65 nested composite; `pq_signing_key` MUST be
    ///   `Some` (if `None`, falls back to Classical, defensive).
    ///
    /// `sig`/`cnf` are always populated with the raw EdDSA signature + signer
    /// public key. The COSE build is **fail-closed**: a COSE encoding error
    /// panics rather than silently emitting an empty (fail-open) composite,
    /// matching the request side.
    pub fn new_signed_with_policy(
        request_id: u64,
        payload: Vec<u8>,
        signing_key: &SigningKey,
        pq_signing_key: Option<&crate::crypto::pq::MlDsaSigningKey>,
        policy: crate::crypto::CryptoPolicy,
    ) -> Self {
        let signing_data = Self::signing_data(request_id, &payload);

        let signature_obj = signing_key.sign(&signing_data);
        let sig: [u8; 64] = signature_obj.to_bytes();
        let cnf: [u8; 32] = signing_key.verifying_key().to_bytes();

        // SECURITY: no empty-cose fallback (mirrors SignedEnvelope). A COSE build
        // failure is a should-never-happen crypto-encoding error; fail loud
        // rather than emit a potentially fail-open empty composite.
        #[allow(clippy::expect_used)]
        let cose = Self::build_cose(signing_key, pq_signing_key, policy, &signing_data)
            .expect("COSE composite signing must not fail for valid keys");

        Self {
            request_id,
            payload,
            encrypted_response: None,
            sig,
            cnf,
            cose,
            policy,
        }
    }

    /// Seal a unary response to the request's one-shot recipient, then
    /// hybrid-sign the ciphertext in the response signature domain.
    pub fn new_signed_encrypted(
        request_id: u64,
        payload: Vec<u8>,
        signing_key: &SigningKey,
        pq_signing_key: &crate::crypto::pq::MlDsaSigningKey,
        recipient: &crate::crypto::hybrid_kem::RecipientPublic,
        request_iat: i64,
        request_nonce: &[u8; 16],
        service_domain: &str,
    ) -> EnvelopeResult<Self> {
        validate_service_domain(service_domain)
            .map_err(|e| EnvelopeError::Encryption(e.to_string()))?;
        let mut plaintext = Vec::new();
        {
            let mut message = capnp::message::Builder::new_default();
            let mut builder =
                message.init_root::<crate::common_capnp::response_plaintext::Builder>();
            builder.set_request_id(request_id);
            builder.set_payload(&payload);
            capnp::serialize::write_message(&mut plaintext, &message).map_err(|e| {
                EnvelopeError::Encryption(format!("serialize response plaintext: {e}"))
            })?;
        }
        let server_identity = signing_key.verifying_key().to_bytes();
        let aad = encrypted_response_external_aad(
            request_id,
            request_iat,
            request_nonce,
            &server_identity,
            recipient,
            service_domain,
        )?;
        let ciphertext =
            crate::crypto::cose_encrypt::seal_to_recipient(recipient, &plaintext, &aad, 0, 0)
                .map_err(|e| {
                    EnvelopeError::Encryption(format!("response HyKEM seal failed: {e}"))
                })?;
        if ciphertext.len() > MAX_ENCRYPTED_RESPONSE_BYTES {
            return Err(EnvelopeError::Encryption(
                "encrypted response exceeds envelope limit".into(),
            ));
        }
        let signing_data = Self::encrypted_signing_data(request_id, &ciphertext, service_domain)?;
        let signature_obj = signing_key.sign(&signing_data);
        let cose = Self::build_cose(
            signing_key,
            Some(pq_signing_key),
            crate::crypto::CryptoPolicy::Hybrid,
            &signing_data,
        )
        .map_err(|e| EnvelopeError::Encryption(format!("sign sealed response: {e}")))?;
        Ok(Self {
            request_id,
            payload: Vec::new(),
            encrypted_response: Some(ciphertext),
            sig: signature_obj.to_bytes(),
            cnf: server_identity,
            cose,
            policy: crate::crypto::CryptoPolicy::Hybrid,
        })
    }

    /// Open a previously verified sealed response with the client's one-shot
    /// recipient and the transcript retained by the pending call.
    pub(crate) fn open_encrypted(
        &self,
        recipient_keypair: &crate::crypto::hybrid_kem::RecipientKeypair,
        recipient_public: &crate::crypto::hybrid_kem::RecipientPublic,
        request_iat: i64,
        request_nonce: &[u8; 16],
        expected_server_identity: &[u8; 32],
        service_domain: &str,
    ) -> Result<Vec<u8>> {
        let ciphertext = self
            .encrypted_response
            .as_deref()
            .ok_or_else(|| anyhow!("cleartext response rejected: sealed response required"))?;
        if &self.cnf != expected_server_identity {
            anyhow::bail!("sealed response server identity does not match pending call");
        }
        let aad = encrypted_response_external_aad(
            self.request_id,
            request_iat,
            request_nonce,
            expected_server_identity,
            recipient_public,
            service_domain,
        )?;
        let plaintext = crate::crypto::cose_encrypt::open_from_recipient(
            recipient_keypair,
            ciphertext,
            &aad,
            0,
            0,
        )
        .map_err(|e| anyhow!("response HyKEM open failed: {e}"))?;
        let reader = read_exact_envelope_message(&plaintext)?;
        let inner = reader.get_root::<crate::common_capnp::response_plaintext::Reader>()?;
        let inner_request_id = inner.get_request_id();
        if inner_request_id != self.request_id {
            anyhow::bail!(
                "inner/outer response request id mismatch: {} != {}",
                inner_request_id,
                self.request_id
            );
        }
        Ok(inner.get_payload()?.to_vec())
    }

    /// Build the nested COSE composite over the response signing-data per policy.
    /// Returns `Err` on encoding failure — callers MUST NOT substitute an empty
    /// cose (that would fail open at verify time). Bound to the RESPONSE domain.
    fn build_cose(
        signing_key: &SigningKey,
        pq_signing_key: Option<&crate::crypto::pq::MlDsaSigningKey>,
        policy: crate::crypto::CryptoPolicy,
        signing_data: &[u8],
    ) -> Result<Vec<u8>> {
        let pq = if policy.uses_pq() {
            pq_signing_key
        } else {
            None
        };
        let aad = response_envelope_external_aad();
        crate::crypto::cose_sign::sign_composite(signing_key, pq, signing_data, &aad)
    }

    /// Verify the response signature (Classical-only, EdDSA via COSE).
    ///
    /// Convenience wrapper mirroring [`SignedEnvelope::verify`]: uses the
    /// `Classical` policy and no PQ trust store. For Hybrid enforcement use
    /// [`Self::verify_with`].
    pub fn verify(&self, expected_pubkey: Option<&VerifyingKey>) -> Result<()> {
        self.verify_with(
            expected_pubkey,
            None,
            crate::crypto::CryptoPolicy::Classical,
        )
    }

    /// Verify with an explicit kid-anchored PQ trust store and verify policy.
    ///
    /// Mirrors [`SignedEnvelope::verify_with`]'s WNS per-identity semantics:
    /// - `pq_store`: when `Some`, resolves the trust-anchored ML-DSA-65 key for
    ///   the response's EdDSA signer (`cnf`); the outer COSE entry must verify
    ///   against it and its kid must match (kid-anchoring).
    /// - `verify_policy = Hybrid` is Weakly Non-Separable (per-identity): for a
    ///   signer with an anchored ML-DSA-65 key it ENFORCES the outer layer — a
    ///   stripped/forged/self-cert outer on an ANCHORED identity is rejected; for
    ///   an unanchored signer it falls back to verifying the inner EdDSA
    ///   (classical floor) rather than failing closed. `Classical` verifies only
    ///   the inner EdDSA and skips any PQ entry.
    pub fn verify_with(
        &self,
        expected_pubkey: Option<&VerifyingKey>,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
    ) -> Result<()> {
        let verifying_key = VerifyingKey::from_bytes(&self.cnf)
            .map_err(|_| anyhow::anyhow!("Invalid signer public key"))?;

        if let Some(expected) = expected_pubkey {
            if !bool::from(verifying_key.to_bytes().ct_eq(&expected.to_bytes())) {
                anyhow::bail!("Response signed by unexpected key");
            }
        }

        self.verify_cose(&verifying_key, pq_store, verify_policy, None)
    }

    /// Verify a sealed response against the destination retained by the
    /// pending call. Generic verification deliberately cannot authenticate an
    /// encrypted response because it has no expected service domain.
    pub(crate) fn verify_encrypted_with_service_domain(
        &self,
        expected_pubkey: &VerifyingKey,
        pq_store: &dyn PqTrustStore,
        service_domain: &str,
    ) -> Result<()> {
        validate_service_domain(service_domain)?;
        if self.encrypted_response.is_none() {
            anyhow::bail!("cleartext response rejected: sealed response required");
        }
        let verifying_key = VerifyingKey::from_bytes(&self.cnf)
            .map_err(|_| anyhow::anyhow!("Invalid signer public key"))?;
        if !bool::from(verifying_key.to_bytes().ct_eq(&expected_pubkey.to_bytes())) {
            anyhow::bail!("Response signed by unexpected key");
        }
        self.verify_cose(
            &verifying_key,
            Some(pq_store),
            crate::crypto::CryptoPolicy::Hybrid,
            Some(service_domain),
        )
    }

    /// Verify the COSE composite signature (authoritative auth check).
    ///
    /// Mirrors [`SignedEnvelope::verify_cose`]: the raw EdDSA `sig`/`cnf` is not
    /// trusted on its own; this re-verifies the EdDSA component inside the COSE
    /// composite and (under Hybrid policy) the kid-anchored ML-DSA-65 component.
    fn verify_cose(
        &self,
        ed_vk: &VerifyingKey,
        pq_store: Option<&dyn PqTrustStore>,
        verify_policy: crate::crypto::CryptoPolicy,
        service_domain: Option<&str>,
    ) -> Result<()> {
        let signing_bytes = self.encrypted_response.as_deref().unwrap_or(&self.payload);
        let signing_data = if self.encrypted_response.is_some() {
            let service_domain = service_domain.ok_or_else(|| {
                anyhow!("encrypted response verification requires the pending service domain")
            })?;
            Self::encrypted_signing_data(self.request_id, signing_bytes, service_domain)?
        } else {
            Self::signing_data(self.request_id, signing_bytes)
        };
        let aad = response_envelope_external_aad();

        // kid-anchor: resolve the trusted ML-DSA-65 key for this EdDSA identity.
        let anchored_pq = pq_store.and_then(|s| s.ml_dsa_key_for(&self.cnf));

        // WNS posture (draft-ietf-pquip-hybrid-signature-spectrums): the composite is
        // Weakly Non-Separable — the inner EdDSA is independently verifiable, so PQ
        // enforcement is applied PER-IDENTITY. Require the ML-DSA-65 outer ONLY for a
        // signer whose PQ key we have anchored out-of-band; for an unanchored signer,
        // fall back to the inner EdDSA (classical floor) rather than failing closed.
        // Safe because ed_vk is derived from this same `cnf` (the PQ-lookup identity ==
        // the EdDSA-verified identity), so an anchored identity cannot be downgraded by
        // spoofing cnf, while an unanchored one is no weaker than classical. PQ is NEVER
        // resolved from the self-asserted COSE entry (that is the self-cert weakness).
        let require_pq = verify_policy.uses_pq() && anchored_pq.is_some();
        #[cfg(not(target_arch = "wasm32"))]
        if verify_policy.uses_pq() && anchored_pq.is_none() {
            tracing::debug!(
                "Hybrid policy active but signer has no anchored ML-DSA-65 key; \
                 verifying classical inner EdDSA (WNS backwards-compat)"
            );
        }

        crate::crypto::cose_sign::verify_composite(
            &self.cose,
            ed_vk,
            anchored_pq.as_ref(),
            &signing_data,
            &aad,
            require_pq,
        )
        .map_err(|e| anyhow::anyhow!("Response signature verification failed: {e}"))?;

        Ok(())
    }

    /// Get the signer's public key.
    pub fn cnf_key(&self) -> Result<VerifyingKey> {
        VerifyingKey::from_bytes(&self.cnf)
            .map_err(|_| anyhow::anyhow!("Invalid signer public key"))
    }
}

impl ToCapnp for ResponseEnvelope {
    type Builder<'a> = common_capnp::response_envelope::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_request_id(self.request_id);
        builder.set_payload(&self.payload);
        if let Some(ref ciphertext) = self.encrypted_response {
            builder.set_encrypted_response(ciphertext);
        }
        builder.set_sig(&self.sig);
        builder.set_cnf(&self.cnf);
        builder.set_cose(&self.cose);
    }
}

impl FromCapnp for ResponseEnvelope {
    type Reader<'a> = common_capnp::response_envelope::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let sig_data = reader.get_sig()?;
        if sig_data.len() != 64 {
            anyhow::bail!("Invalid sig length: {}", sig_data.len());
        }
        let mut sig = [0u8; 64];
        sig.copy_from_slice(sig_data);

        let cnf_data = reader.get_cnf()?;
        if cnf_data.len() != 32 {
            anyhow::bail!("Invalid cnf length: {}", cnf_data.len());
        }
        let mut cnf = [0u8; 32];
        cnf.copy_from_slice(cnf_data);

        let cose = reader.get_cose()?.to_vec();
        let encrypted_data = reader.get_encrypted_response()?;
        let payload_data = reader.get_payload()?;
        if !encrypted_data.is_empty() && !payload_data.is_empty() {
            anyhow::bail!("encrypted response carries a non-empty cleartext payload");
        }
        let payload = payload_data.to_vec();
        let encrypted_response = {
            let data = encrypted_data;
            if data.is_empty() {
                None
            } else {
                if data.len() > MAX_ENCRYPTED_RESPONSE_BYTES {
                    anyhow::bail!("encryptedResponse exceeds envelope limit");
                }
                Some(data.to_vec())
            }
        };

        // `policy` is a signing-time concept; the verifier supplies the verify
        // policy explicitly, so decode to the default here (mirrors SignedEnvelope).
        Ok(Self {
            request_id: reader.get_request_id(),
            payload,
            encrypted_response,
            sig,
            cnf,
            cose,
            policy: crate::crypto::CryptoPolicy::default(),
        })
    }
}

/// Unwrap and verify a SignedEnvelope from wire bytes.
///
/// Dispatches on `opts.verification` to select FixedSigner or AnySigner mode.
/// If `opts.decryption_key` is set and the envelope is encrypted, decrypts
/// the envelope after signature verification.
pub fn unwrap_and_verify(
    request: &[u8],
    opts: &UnwrapOptions<'_>,
) -> Result<(SignedEnvelope, Vec<u8>)> {
    let reader = read_exact_envelope_message(request)?;
    let signed_reader = reader.get_root::<crate::common_capnp::signed_envelope::Reader>()?;
    let mut signed = SignedEnvelope::read_from(signed_reader)?;

    // Reject stale or future requests before signature or KEM work without
    // mutating replay state. The timestamp is revalidated at final commit.
    signed.validate_timestamp()?;

    // Authenticate first without mutating replay state.  The compatibility
    // lifecycle commits the nonce only after canonical COSE parsing, external-
    // AAD authentication, decryption, and inner/outer equality all succeed.
    match &opts.verification {
        EnvelopeVerification::FixedSigner(pubkey) => {
            // Fu2/#677: production wire verify uses the unanchored-aware path —
            // deny-on-missing-anchor by default, with an opt-in classical-floor
            // allowlist for legacy peers.
            signed.verify_signature_with_unanchored_policy(
                pubkey,
                opts.pq_store,
                opts.verify_policy,
                opts.unanchored_policy,
                &opts.unanchored_classical_allowlist,
            )?;
        }
        EnvelopeVerification::AnySigner => {
            signed.verify_any_signature_with_unanchored_policy(
                opts.pq_store,
                opts.verify_policy,
                opts.unanchored_policy,
                &opts.unanchored_classical_allowlist,
            )?;
        }
    }

    // INV-2 receive-side policy (#1042): the carrier requirement is supplied
    // by the accept boundary, never by request bytes. Check it only after the
    // signature above, so unauthenticated input cannot
    // select or exercise a signed policy-error path. A non-empty marker alone
    // is not sufficient: encrypted envelopes continue through canonical KEM
    // open below before any claims or handler sees their payload.
    if opts.require_encrypted && !signed.is_encrypted() {
        return Err(anyhow!(
            "cleartext request envelope rejected: carrier requires #mesh-kem encryption (INV-2)"
        ));
    }

    if signed.is_encrypted() {
        if let Some(decryption_key) = opts.decryption_key {
            // The node's #mesh-kem keypair is deterministically derived from its
            // Ed25519 identity key (S1), so the decryption seam keeps taking the
            // signing key and derives the KEM keypair here.
            let kem_keypair = crate::node_identity::derive_mesh_kem_recipient(decryption_key)
                .map_err(|e| anyhow!("derive #mesh-kem keypair for decryption: {e}"))?;
            signed.decrypt_in_place_mesh_kem(&kem_keypair)?;
        } else {
            return Err(anyhow!(
                "encrypted envelope but no decryption key configured"
            ));
        }
    }

    // This is the single atomic replay commit for the public compatibility
    // lifecycle. Invalid signatures, non-canonical ciphertext, wrong AAD,
    // failed decryption, and inner/outer mismatch cannot consume or evict cache
    // capacity; concurrent identical valid inputs race here and only one wins.
    signed.check_replay(opts.nonce_cache)?;

    let payload = signed.payload().to_vec();
    Ok((signed, payload))
}

/// Unwrap, verify, and build an `EnvelopeContext` from wire bytes.
///
/// Context construction depends on verification mode:
/// - `FixedSigner` → `key_derived_subject = "system"` (inproc/IPC callers)
/// - `AnySigner` → `key_derived_subject = anonymous` (WebTransport, identity from JWT)
#[cfg(not(target_arch = "wasm32"))]
pub fn unwrap_envelope(
    request: &[u8],
    opts: &UnwrapOptions<'_>,
) -> Result<(crate::service::EnvelopeContext, Vec<u8>)> {
    let (signed, payload) = unwrap_and_verify(request, opts)?;

    let ctx = match &opts.verification {
        EnvelopeVerification::FixedSigner(_) => {
            crate::service::EnvelopeContext::from_verified_as_system(&signed)
        }
        EnvelopeVerification::AnySigner => crate::service::EnvelopeContext::from_verified(&signed),
    };

    Ok((ctx, payload))
}

/// Unwrap and verify a ResponseEnvelope from wire bytes (Classical-only).
///
/// Convenience wrapper over [`unwrap_response_with`] using the `Classical`
/// policy and no PQ trust store. Mirrors [`SignedEnvelope::verify`]; for Hybrid
/// enforcement use [`unwrap_response_with`].
///
/// # Arguments
///
/// * `response` - Raw bytes containing a serialized ResponseEnvelope
/// * `expected_pubkey` - Optional expected signer public key
///
/// # Returns
///
/// On success, returns `(request_id, payload)` where:
/// - `request_id` correlates with the original request
/// - `payload` is the inner response bytes
///
/// # Errors
///
/// Returns error if:
/// - Deserialization fails
/// - Signature verification fails
/// - Signer doesn't match expected_pubkey (if provided)
pub fn unwrap_response(
    response: &[u8],
    expected_pubkey: Option<&VerifyingKey>,
) -> Result<(u64, Vec<u8>)> {
    unwrap_response_with(
        response,
        expected_pubkey,
        None,
        crate::crypto::CryptoPolicy::Classical,
    )
}

/// Unwrap and verify a ResponseEnvelope under an explicit kid-anchored PQ trust
/// store + verify policy (#275).
///
/// Enforces the COSE composite with the SAME WNS per-identity semantics as
/// [`unwrap_and_verify`]: under `Hybrid`, an anchored signer must use its
/// ML-DSA-65 key (a stripped-outer, classical-only, self-cert, or forged outer
/// on an ANCHORED identity is REJECTED), while an unanchored signer falls back
/// to the inner EdDSA (classical floor) rather than failing closed.
pub fn unwrap_response_with(
    response: &[u8],
    expected_pubkey: Option<&VerifyingKey>,
    pq_store: Option<&dyn PqTrustStore>,
    verify_policy: crate::crypto::CryptoPolicy,
) -> Result<(u64, Vec<u8>)> {
    let envelope = read_response_envelope(response, false)?;

    if envelope.encrypted_response.is_some() {
        anyhow::bail!("encrypted response requires one-shot pending decapsulation material");
    }

    // Verify the COSE composite under the supplied policy + anchor.
    envelope.verify_with(expected_pubkey, pq_store, verify_policy)?;

    Ok((envelope.request_id, envelope.payload))
}

/// Bounded response parse with an early encrypted-field gate. When encryption
/// is required, the cleartext payload is never copied or surfaced.
pub(crate) fn read_response_envelope(
    response: &[u8],
    require_encrypted: bool,
) -> Result<ResponseEnvelope> {
    let reader = read_exact_envelope_message(response)?;
    let response_reader = reader.get_root::<crate::common_capnp::response_envelope::Reader>()?;
    if require_encrypted && response_reader.get_encrypted_response()?.is_empty() {
        anyhow::bail!("cleartext response rejected before payload use (INV-2)");
    }
    ResponseEnvelope::read_from(response_reader)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::crypto::signing::generate_signing_keypair;
    use parking_lot::Mutex;
    use std::collections::HashSet;

    /// Serializes tests that mutate the shared `HYPRSTREAM_ENVELOPE_POLICY` env
    /// var so they don't race under parallel `cargo test`.
    #[cfg(not(target_arch = "wasm32"))]
    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Simple in-memory nonce cache for testing.
    struct TestNonceCache {
        seen: Mutex<HashSet<[u8; 16]>>,
    }

    impl TestNonceCache {
        fn new() -> Self {
            Self {
                seen: Mutex::new(HashSet::new()),
            }
        }
    }

    impl NonceCache for TestNonceCache {
        fn check_and_insert(&self, nonce: &[u8; 16]) -> bool {
            self.seen.lock().insert(*nonce)
        }
    }

    /// Test envelopes default to Classical signing; the existing tests exercise
    /// the EdDSA path. Hybrid/cross-compat is covered by the dedicated M3 tests
    /// below.
    fn test_new_signed(envelope: RequestEnvelope, signing_key: &SigningKey) -> SignedEnvelope {
        SignedEnvelope::new_signed(envelope, signing_key)
    }

    /// Fresh, OsRng-backed 16-byte nonce for tests.
    ///
    /// Tests never need a *specific* nonce value (they assert on signatures,
    /// AEAD binding, and roundtrip equality — never on the nonce itself), so we
    /// draw from the production CSPRNG instead of seeding envelopes with
    /// hard-coded byte arrays, which trips `rust/hard-coded-cryptographic-value`.
    /// The one place a test needs a *second, distinct* nonce (the replay-binding
    /// tamper) derives it via [`distinct_test_nonce`] rather than a second
    /// literal.
    fn fresh_test_nonce() -> [u8; 16] {
        super::generate_nonce()
    }

    /// Derive a nonce that is guaranteed to differ from `original` in every byte
    /// without introducing another hard-coded cryptographic literal: bitwise
    /// complement (`!b != b` for all bytes), so the replay-binding tamper is
    /// provably distinct yet never touches a literal crypto value.
    fn distinct_test_nonce(original: &[u8; 16]) -> [u8; 16] {
        let mut out = *original;
        for b in out.iter_mut() {
            *b = !*b;
        }
        out
    }

    fn signed_envelope_to_wire(envelope: &SignedEnvelope) -> Vec<u8> {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder = message.init_root::<common_capnp::signed_envelope::Builder>();
            envelope.write_to(&mut builder);
        }
        let mut wire = Vec::new();
        capnp::serialize::write_message(&mut wire, &message).expect("serialize signed envelope");
        wire
    }

    /// In-memory kid-anchored PQ trust store for tests.
    struct TestPqStore {
        bindings: Vec<([u8; 32], crate::crypto::pq::MlDsaVerifyingKey)>,
    }
    impl PqTrustStore for TestPqStore {
        fn ml_dsa_key_for(
            &self,
            ed25519_pubkey: &[u8; 32],
        ) -> Option<crate::crypto::pq::MlDsaVerifyingKey> {
            self.bindings
                .iter()
                .find(|(k, _)| k == ed25519_pubkey)
                .map(|(_, vk)| {
                    crate::crypto::pq::ml_dsa_vk_from_bytes(&crate::crypto::pq::ml_dsa_vk_bytes(vk))
                        .expect("re-decode pq vk")
                })
        }
    }

    #[test]
    fn test_request_envelope() {
        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3]);
        assert_eq!(envelope.payload, vec![1, 2, 3]);
        assert!(envelope.request_id > 0);
        assert!(envelope.iat > 0);
        assert!(envelope.nonce.iter().any(|&b| b != 0)); // Not all zeros
    }

    #[test]
    fn test_request_id_increments() {
        let e1 = RequestEnvelope::anonymous(vec![]);
        let e2 = RequestEnvelope::anonymous(vec![]);
        assert!(e2.request_id > e1.request_id);
    }

    #[test]
    fn test_signed_envelope_sign_verify() -> crate::EnvelopeResult<()> {
        let (signing_key, verifying_key) = generate_signing_keypair();
        let nonce_cache = TestNonceCache::new();

        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3, 4]);
        let signed = test_new_signed(envelope, &signing_key);

        // Verify should succeed
        signed.verify(&verifying_key, &nonce_cache)?;
        Ok(())
    }

    #[test]
    fn test_signed_envelope_wrong_key_fails() {
        let (signing_key, _) = generate_signing_keypair();
        let (_, wrong_verifying_key) = generate_signing_keypair();
        let nonce_cache = TestNonceCache::new();

        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3, 4]);
        let signed = test_new_signed(envelope, &signing_key);

        // Verify with wrong key should fail
        let result = signed.verify(&wrong_verifying_key, &nonce_cache);
        assert!(matches!(result, Err(EnvelopeError::SignerMismatch { .. })));
    }

    #[test]
    fn test_signed_envelope_replay_fails() -> crate::EnvelopeResult<()> {
        let (signing_key, verifying_key) = generate_signing_keypair();
        let nonce_cache = TestNonceCache::new();

        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3, 4]);
        let signed = test_new_signed(envelope, &signing_key);

        // First verify succeeds
        signed.verify(&verifying_key, &nonce_cache)?;

        // Replay (same nonce) should fail
        let result = signed.verify(&verifying_key, &nonce_cache);
        assert!(matches!(result, Err(EnvelopeError::ReplayAttack(_))));
        Ok(())
    }

    #[test]
    fn test_signed_envelope_accessors() {
        let (signing_key, _) = generate_signing_keypair();

        let envelope = RequestEnvelope::anonymous(vec![5, 6, 7]);
        let signed = test_new_signed(envelope, &signing_key);

        assert!(signed.request_id() > 0);
        assert_eq!(signed.payload(), &[5, 6, 7]);
    }

    #[test]
    fn test_capnp_roundtrip_envelope() -> anyhow::Result<()> {
        use capnp::message::Builder;

        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3, 4]);

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<common_capnp::request_envelope::Builder>();
        envelope.write_to(&mut builder);

        let reader = builder.into_reader();
        let decoded = RequestEnvelope::read_from(reader)?;

        assert_eq!(envelope.request_id, decoded.request_id);
        assert_eq!(envelope.payload, decoded.payload);
        assert_eq!(envelope.nonce, decoded.nonce);
        assert_eq!(envelope.iat, decoded.iat);
        assert_eq!(envelope.authorization, decoded.authorization);
        assert_eq!(envelope.delegation_token, decoded.delegation_token);
        assert_eq!(envelope.wth, decoded.wth);
        assert_eq!(envelope.client_dh_public, decoded.client_dh_public);
        Ok(())
    }

    #[test]
    fn test_capnp_roundtrip_signed_envelope() -> Result<(), Box<dyn std::error::Error>> {
        use capnp::message::Builder;

        let (signing_key, verifying_key) = generate_signing_keypair();
        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3]);
        let signed = test_new_signed(envelope, &signing_key);

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<common_capnp::signed_envelope::Builder>();
        signed.write_to(&mut builder);

        let reader = builder.into_reader();
        let decoded = SignedEnvelope::read_from(reader)?;

        assert_eq!(signed.envelope.request_id, decoded.envelope.request_id);
        assert_eq!(signed.envelope.payload, decoded.envelope.payload);
        assert_eq!(signed.sig, decoded.sig);
        assert_eq!(signed.cnf, decoded.cnf);

        // Verify the decoded envelope still has valid signature
        decoded.verify_signature_only(&verifying_key)?;
        Ok(())
    }

    // =========================================================================
    // Subject tests
    // =========================================================================

    #[test]
    fn test_subject_display_roundtrip() {
        let cases = vec![
            Subject::new("alice"),
            Subject::new("bob"),
            Subject::new("gpu-1"),
            Subject::new("charlie"),
            Subject::anonymous(),
        ];

        for subject in cases {
            let s = subject.to_string();
            let parsed: Subject = s.parse().expect("parse subject roundtrip");
            assert_eq!(subject, parsed, "Roundtrip failed for {s}");
        }
    }

    #[test]
    fn test_subject_display_format() {
        assert_eq!(Subject::new("alice").to_string(), "alice");
        assert_eq!(Subject::new("bob").to_string(), "bob");
        assert_eq!(Subject::anonymous().to_string(), "anonymous");
    }

    #[test]
    fn test_subject_legacy_prefix_parsing() {
        assert_eq!(
            "local:alice".parse::<Subject>().expect("parse local:alice"),
            Subject::new("alice")
        );
        assert_eq!(
            "token:bob".parse::<Subject>().expect("parse token:bob"),
            Subject::new("bob")
        );
        assert_eq!(
            "peer:gpu-1".parse::<Subject>().expect("parse peer:gpu-1"),
            Subject::new("gpu-1")
        );
        assert_eq!(
            "user:charlie"
                .parse::<Subject>()
                .expect("parse user:charlie"),
            Subject::new("charlie")
        );
        assert_eq!(
            "alice".parse::<Subject>().expect("parse alice"),
            Subject::new("alice")
        );
        assert_eq!(
            "unknown:foo".parse::<Subject>().expect("parse unknown:foo"),
            Subject::new("unknown:foo")
        );
    }

    #[test]
    fn test_subject_validate() {
        assert!(Subject::new("alice").validate().is_ok());
        assert!(Subject::new("bob_123").validate().is_ok());
        assert!(Subject::new("gpu-server.1").validate().is_ok());
        assert!(Subject::anonymous().validate().is_ok());
        assert!(Subject::new("../evil").validate().is_err());
        assert!(Subject::new("bob/root").validate().is_err());
        assert!(Subject::new("").validate().is_err());
        assert!(Subject::new("alice:bob").validate().is_err());
    }

    #[test]
    fn test_subject_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Subject::new("alice"));
        set.insert(Subject::new("alice")); // duplicate
        set.insert(Subject::new("bob"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_subject_capnp_roundtrip() -> anyhow::Result<()> {
        use capnp::message::Builder;

        let cases = vec![
            Subject::new("alice"),
            Subject::new("bob"),
            Subject::new("gpu-1"),
            Subject::anonymous(),
        ];

        for subject in cases {
            let mut message = Builder::new_default();
            let mut builder = message.init_root::<common_capnp::subject::Builder>();
            subject.write_to(&mut builder);

            let reader = builder.into_reader();
            let decoded = Subject::read_from(reader)?;
            assert_eq!(subject, decoded, "Cap'n Proto roundtrip failed");
        }
        Ok(())
    }

    #[test]
    fn test_subject_from_claims() {
        let claims = crate::auth::Claims::new("charlie".to_owned(), 1000, 2000);
        assert_eq!(Subject::from(&claims), Subject::new("charlie"));
    }

    #[test]
    fn test_subject_federated_format() {
        let s = Subject::federated("https://node-a", "alice");
        assert_eq!(s.to_string(), "https://node-a:alice");
        assert!(s.is_federated());
        assert!(!s.is_anonymous());
    }

    #[test]
    fn test_subject_local_is_not_federated() {
        let s = Subject::new("alice");
        assert!(!s.is_federated());
    }

    #[test]
    fn test_subject_anonymous_is_not_federated() {
        let s = Subject::anonymous();
        assert!(!s.is_federated());
    }

    #[test]
    fn test_subject_federated_two_node_scenario() {
        use crate::auth::{jwt, Claims};
        use ed25519_dalek::SigningKey;

        let key_a = SigningKey::from_bytes(&[0xAAu8; 32]);
        let vk_a = key_a.verifying_key();
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999)
            .with_issuer("https://node-a".to_owned());
        let token = jwt::encode(&claims, &key_a);

        let decoded = jwt::decode_with_key(&token, &vk_a, None)
            .expect("federated token must verify with issuer key");
        assert_eq!(decoded.iss, "https://node-a");
        assert_eq!(decoded.sub, "alice");

        let subject = Subject::federated(&decoded.iss, &decoded.sub);
        assert_eq!(subject.to_string(), "https://node-a:alice");
        assert!(subject.is_federated());

        let local = Subject::new(&decoded.sub);
        assert!(!local.is_federated());
    }

    #[test]
    fn test_authorization_capnp_roundtrip() -> anyhow::Result<()> {
        use capnp::message::Builder;

        // Test None variant
        let auth = Authorization::None;
        let mut msg = Builder::new_default();
        let mut builder = msg.init_root::<common_capnp::authorization::Builder>();
        auth.write_to(&mut builder);
        let reader = builder.into_reader();
        let decoded = Authorization::read_from(reader)?;
        assert_eq!(auth, decoded);

        // Test IdJag variant
        let auth = Authorization::IdJag("test-token".to_owned());
        let mut msg = Builder::new_default();
        let mut builder = msg.init_root::<common_capnp::authorization::Builder>();
        auth.write_to(&mut builder);
        let reader = builder.into_reader();
        let decoded = Authorization::read_from(reader)?;
        assert_eq!(auth, decoded);

        // Test Local variant
        let auth = Authorization::Local(TokenClaims {
            iss: "https://hyprstream.local".to_owned(),
            sub: "alice".to_owned(),
            aud: vec!["inference".to_owned(), "registry".to_owned()],
            exp: 1700000000,
            iat: 1699999000,
            jti: "unique-id-123".to_owned(),
            scope: vec![],
            cnf_jkt: "thumbprint-abc".to_owned(),
        });
        let mut msg = Builder::new_default();
        let mut builder = msg.init_root::<common_capnp::authorization::Builder>();
        auth.write_to(&mut builder);
        let reader = builder.into_reader();
        let decoded = Authorization::read_from(reader)?;
        assert_eq!(auth, decoded);

        // Test Federated variant
        let auth = Authorization::Federated(FederatedToken {
            raw: "eyJ0eXAiOiJKV1QiLCJhbGciOiJFZERTQSJ9.test".to_owned(),
            claims: TokenClaims {
                iss: "https://node-a".to_owned(),
                sub: "bob".to_owned(),
                aud: vec!["https://node-b".to_owned()],
                exp: 1700000000,
                iat: 1699999000,
                jti: "fed-id-456".to_owned(),
                scope: vec![],
                cnf_jkt: "fed-thumbprint".to_owned(),
            },
            dpop_proof: Some("dpop-proof-xyz".to_owned()),
        });
        let mut msg = Builder::new_default();
        let mut builder = msg.init_root::<common_capnp::authorization::Builder>();
        auth.write_to(&mut builder);
        let reader = builder.into_reader();
        let decoded = Authorization::read_from(reader)?;
        assert_eq!(auth, decoded);

        // Test Federated without DPoP proof
        let auth = Authorization::Federated(FederatedToken {
            raw: "raw-jwt".to_owned(),
            claims: TokenClaims {
                iss: "https://node-c".to_owned(),
                sub: "carol".to_owned(),
                aud: vec![],
                exp: 0,
                iat: 0,
                jti: String::new(),
                scope: vec![],
                cnf_jkt: String::new(),
            },
            dpop_proof: None,
        });
        let mut msg = Builder::new_default();
        let mut builder = msg.init_root::<common_capnp::authorization::Builder>();
        auth.write_to(&mut builder);
        let reader = builder.into_reader();
        let decoded = Authorization::read_from(reader)?;
        assert_eq!(auth, decoded);

        Ok(())
    }

    #[test]
    fn test_capnp_roundtrip_envelope_with_authorization() -> anyhow::Result<()> {
        use capnp::message::Builder;

        let envelope = RequestEnvelope {
            request_id: 42,
            payload: vec![1, 2, 3],
            nonce: fresh_test_nonce(),
            iat: 1699999000,
            authorization: Authorization::IdJag("my-jwt-token".to_owned()),
            delegation_token: Some("delegated".to_owned()),
            wth: Some([0xAB; 32]),
            client_dh_public: Some([0xCD; 32]),
            client_kem_public: None,
            response_kem_recipient: None,
            service_domain: None,
        };

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<common_capnp::request_envelope::Builder>();
        envelope.write_to(&mut builder);

        let reader = builder.into_reader();
        let decoded = RequestEnvelope::read_from(reader)?;

        assert_eq!(envelope.request_id, decoded.request_id);
        assert_eq!(envelope.payload, decoded.payload);
        assert_eq!(envelope.nonce, decoded.nonce);
        assert_eq!(envelope.iat, decoded.iat);
        assert_eq!(envelope.authorization, decoded.authorization);
        assert_eq!(envelope.delegation_token, decoded.delegation_token);
        assert_eq!(envelope.wth, decoded.wth);
        assert_eq!(envelope.client_dh_public, decoded.client_dh_public);
        Ok(())
    }

    #[test]
    fn mesh_kem_envelope_seal_open_roundtrip() {
        use crate::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};

        // Node identity: Ed25519 root → #mesh-mldsa sign key + #mesh-kem keypair.
        let (node_sk, node_vk) = generate_signing_keypair();
        let pq_sk = derive_mesh_mldsa_key(&node_sk);
        let kem_kp = derive_mesh_kem_recipient(&node_sk).expect("derive #mesh-kem");
        let kem_pub = kem_kp.public();

        let req_id = 777u64;
        let payload = vec![9u8, 8, 7, 6];
        let envelope = RequestEnvelope {
            request_id: req_id,
            payload: payload.clone(),
            nonce: fresh_test_nonce(),
            iat: current_timestamp(),
            authorization: Authorization::None,
            delegation_token: None,
            wth: None,
            client_dh_public: None,
            client_kem_public: None,
            response_kem_recipient: None,
            service_domain: None,
        };

        // Seal to the node's #mesh-kem public, dual-signed (EdDSA + ML-DSA-65).
        let signed =
            SignedEnvelope::new_signed_encrypted_mesh_kem(envelope, &node_sk, &pq_sk, &kem_pub)
                .expect("seal");

        // Encrypted; plaintext is NOT on the wire; no separate eph/KEM-ct fields.
        assert!(signed.is_encrypted());
        assert!(signed.encrypted_envelope.is_some());
        assert!(signed.client_ephemeral_public.is_none());
        assert!(signed.pq_kem_ciphertext.is_none());

        // COSE composite signature verifies over the COSE_Encrypt0 bytes.
        signed
            .verify_signature_only(&node_vk)
            .expect("signature over COSE_Encrypt0 verifies");

        // Decrypt with the node's #mesh-kem keypair.
        let mut opened = signed.clone();
        opened.decrypt_in_place_mesh_kem(&kem_kp).expect("open");
        assert_eq!(opened.envelope.request_id, req_id);
        assert_eq!(opened.envelope.payload, payload);

        // Fail-closed: a WRONG #mesh-kem keypair must not open.
        let (other_sk, _) = generate_signing_keypair();
        let other_kem = derive_mesh_kem_recipient(&other_sk).unwrap();
        let mut wrong = signed.clone();
        assert!(
            wrong.decrypt_in_place_mesh_kem(&other_kem).is_err(),
            "wrong #mesh-kem key must not open"
        );
    }

    #[test]
    fn mesh_kem_outer_replay_metadata_is_aead_bound() {
        // Regression (S4 replay-binding): the outer replay metadata (nonce/iat/
        // request_id) is unsigned on the wire — the composite signature covers
        // only the ciphertext. It MUST be bound into the COSE_Encrypt0 AEAD so a
        // network attacker cannot mutate the outer nonce/iat to evade the
        // server's replay admission while the signature still verifies.
        use crate::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};

        let (node_sk, node_vk) = generate_signing_keypair();
        let pq_sk = derive_mesh_mldsa_key(&node_sk);
        let kem_kp = derive_mesh_kem_recipient(&node_sk).expect("derive #mesh-kem");
        let kem_pub = kem_kp.public();

        // Bind the original replay nonce so the tamper below can derive a
        // provably-distinct value without a second hard-coded literal.
        let original_nonce = fresh_test_nonce();
        let envelope = RequestEnvelope {
            request_id: 42,
            payload: vec![1, 2, 3],
            nonce: original_nonce,
            iat: current_timestamp(),
            authorization: Authorization::None,
            delegation_token: None,
            wth: None,
            client_dh_public: None,
            client_kem_public: None,
            response_kem_recipient: None,
            service_domain: None,
        };
        let signed =
            SignedEnvelope::new_signed_encrypted_mesh_kem(envelope, &node_sk, &pq_sk, &kem_pub)
                .expect("seal");

        // Attacker mutates ONLY the outer replay metadata (fields carried in the
        // clear beside the ciphertext) — the ciphertext and its composite
        // signature are untouched, so the signature still verifies…
        let mut tampered = signed.clone();
        tampered.envelope.nonce = distinct_test_nonce(&original_nonce);
        tampered.envelope.iat = tampered.envelope.iat.wrapping_add(1);
        tampered
            .verify_signature_only(&node_vk)
            .expect("signature over ciphertext is unaffected by outer-field tampering");

        // …but decryption MUST fail: the AAD the server recomputes from the
        // mutated outer fields no longer matches what was sealed.
        assert!(
            tampered.decrypt_in_place_mesh_kem(&kem_kp).is_err(),
            "mutated outer replay metadata must break #mesh-kem decryption (replay bind)"
        );

        // Sanity: the untouched envelope still opens.
        let mut good = signed.clone();
        good.decrypt_in_place_mesh_kem(&kem_kp)
            .expect("untampered envelope opens");
        assert_eq!(good.envelope.request_id, 42);
    }

    #[test]
    fn compatibility_replay_commits_only_after_full_authentication_and_atomically() {
        use crate::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};

        let (node_sk, node_vk) = generate_signing_keypair();
        let pq_sk = derive_mesh_mldsa_key(&node_sk);
        let kem_pub = derive_mesh_kem_recipient(&node_sk)
            .expect("derive #mesh-kem")
            .public();
        let original_nonce = fresh_test_nonce();
        let original = SignedEnvelope::new_signed_encrypted_mesh_kem(
            RequestEnvelope {
                request_id: 553,
                payload: b"authenticated payload".to_vec(),
                nonce: original_nonce,
                iat: current_timestamp(),
                authorization: Authorization::None,
                delegation_token: None,
                wth: None,
                client_dh_public: None,
                client_kem_public: None,
                response_kem_recipient: None,
                service_domain: None,
            },
            &node_sk,
            &pq_sk,
            &kem_pub,
        )
        .expect("seal original");
        let original_wire = signed_envelope_to_wire(&original);

        // Fill a capacity-one cache with one fully authenticated request.
        let cache = InMemoryNonceCache::with_config(MAX_TIMESTAMP_AGE_MS, 1);
        let options = || {
            UnwrapOptions::fixed_signer(&node_vk, &cache)
                .with_decryption_key(&node_sk)
                .require_encrypted(true)
                .with_verify_policy(crate::crypto::CryptoPolicy::Classical)
        };
        unwrap_and_verify(&original_wire, &options()).expect("original request is accepted");

        // Wire-controlled integer extremes fail in the early non-mutating
        // timestamp precheck and cannot evict the accepted entry from this
        // capacity-one cache. The ciphertext signature is intentionally left
        // valid; changing only outer metadata would otherwise proceed to AEAD.
        for extreme_iat in [i64::MIN, i64::MAX] {
            let mut extreme = original.clone();
            extreme.envelope.nonce = distinct_test_nonce(&original_nonce);
            extreme.envelope.iat = extreme_iat;
            extreme
                .verify_signature_only(&node_vk)
                .expect("ciphertext signature remains valid");
            let error = unwrap_and_verify(&signed_envelope_to_wire(&extreme), &options())
                .expect_err("extreme timestamp is rejected without a panic");
            assert!(error.to_string().contains(if extreme_iat == i64::MIN {
                "timestamp too old"
            } else {
                "timestamp in future"
            }));
            assert!(
                unwrap_and_verify(&original_wire, &options()).is_err(),
                "rejected extreme timestamp must not evict accepted replay state"
            );
        }

        // Authenticated but expired outer metadata is rejected before KEM work
        // and cannot disturb the full capacity-one cache.
        let mut stale = original.clone();
        stale.envelope.nonce = distinct_test_nonce(&original_nonce);
        stale.envelope.iat = current_timestamp() - MAX_TIMESTAMP_AGE_MS - 1;
        stale
            .verify_signature_only(&node_vk)
            .expect("ciphertext signature remains valid");
        let stale_error = unwrap_and_verify(&signed_envelope_to_wire(&stale), &options())
            .expect_err("expired request is rejected");
        assert!(stale_error.to_string().contains("timestamp too old"));
        assert!(unwrap_and_verify(&original_wire, &options()).is_err());

        // A distinct outer nonce plus an invalid signature must not consume or
        // evict cache capacity.
        let mut invalid_signature = original.clone();
        invalid_signature.envelope.nonce = distinct_test_nonce(&original_nonce);
        let last = invalid_signature.cose.len() - 1;
        invalid_signature.cose[last] ^= 1;
        assert!(
            unwrap_and_verify(&signed_envelope_to_wire(&invalid_signature), &options()).is_err()
        );
        assert!(unwrap_and_verify(&original_wire, &options()).is_err());

        // The signature covers the unchanged ciphertext and remains valid, but
        // the distinct outer nonce changes authenticated external AAD. Failed
        // AEAD authentication likewise must not touch replay state.
        let mut invalid_aad = original.clone();
        invalid_aad.envelope.nonce = distinct_test_nonce(&original_nonce);
        invalid_aad
            .verify_signature_only(&node_vk)
            .expect("ciphertext signature remains valid");
        assert!(unwrap_and_verify(&signed_envelope_to_wire(&invalid_aad), &options()).is_err());

        // Both invalid attempts independently left the original accepted nonce
        // resident in the capacity-one cache.
        assert!(unwrap_and_verify(&original_wire, &options()).is_err());

        // Two concurrent, identical, fully valid inputs perform all validation
        // before racing at one atomic check-and-insert; exactly one can commit.
        let concurrent_cache = InMemoryNonceCache::with_config(MAX_TIMESTAMP_AGE_MS, 1);
        let accepted = std::thread::scope(|scope| {
            let attempts: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        let opts = UnwrapOptions::fixed_signer(&node_vk, &concurrent_cache)
                            .with_decryption_key(&node_sk)
                            .require_encrypted(true)
                            .with_verify_policy(crate::crypto::CryptoPolicy::Classical);
                        unwrap_and_verify(&original_wire, &opts).is_ok()
                    })
                })
                .collect();
            attempts
                .into_iter()
                .map(|attempt| attempt.join().expect("verification thread"))
                .filter(|accepted| *accepted)
                .count()
        });
        assert_eq!(accepted, 1);
    }

    #[test]
    fn timestamp_window_boundaries_and_extremes_are_overflow_safe() {
        let now = 0;

        // Both documented limits are inclusive.
        assert!(SignedEnvelope::validate_timestamp_at(-MAX_TIMESTAMP_AGE_MS, now).is_ok());
        assert!(SignedEnvelope::validate_timestamp_at(MAX_CLOCK_SKEW_MS, now).is_ok());

        // Values one millisecond inside either inclusive limit also remain valid.
        assert!(SignedEnvelope::validate_timestamp_at(-MAX_TIMESTAMP_AGE_MS + 1, now).is_ok());
        assert!(SignedEnvelope::validate_timestamp_at(MAX_CLOCK_SKEW_MS - 1, now).is_ok());

        // The immediately adjacent values outside either window fail closed.
        assert!(SignedEnvelope::validate_timestamp_at(-MAX_TIMESTAMP_AGE_MS - 1, now).is_err());
        assert!(SignedEnvelope::validate_timestamp_at(MAX_CLOCK_SKEW_MS + 1, now).is_err());

        // Widening before subtraction defines both ends of the wire domain,
        // even when the clock is itself at the opposite i64 boundary.
        assert!(SignedEnvelope::validate_timestamp_at(i64::MIN, now).is_err());
        assert!(SignedEnvelope::validate_timestamp_at(i64::MAX, now).is_err());
        assert!(SignedEnvelope::validate_timestamp_at(i64::MIN, i64::MAX).is_err());
        assert!(SignedEnvelope::validate_timestamp_at(i64::MAX, i64::MIN).is_err());
    }

    #[test]
    fn ordinary_unwrap_rejects_extreme_timestamps_for_all_compatibility_paths() {
        use crate::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};

        let (node_sk, node_vk) = generate_signing_keypair();
        let pq_sk = derive_mesh_mldsa_key(&node_sk);
        let kem_pub = derive_mesh_kem_recipient(&node_sk)
            .expect("derive #mesh-kem")
            .public();

        let clear = SignedEnvelope::new_signed(
            RequestEnvelope::anonymous(b"clear timestamp boundary".to_vec()),
            &node_sk,
        );
        let encrypted = SignedEnvelope::new_signed_encrypted_mesh_kem(
            RequestEnvelope::anonymous(b"encrypted timestamp boundary".to_vec()),
            &node_sk,
            &pq_sk,
            &kem_pub,
        )
        .expect("seal encrypted boundary fixture");

        for extreme_iat in [i64::MIN, i64::MAX] {
            for fixture in [&clear, &encrypted] {
                let mut extreme = fixture.clone();
                extreme.envelope.iat = extreme_iat;
                let wire = signed_envelope_to_wire(&extreme);

                for any_signer in [false, true] {
                    let cache = TestNonceCache::new();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let opts = if any_signer {
                            UnwrapOptions::any_signer(&cache)
                        } else {
                            UnwrapOptions::fixed_signer(&node_vk, &cache)
                        }
                        .with_verify_policy(crate::crypto::CryptoPolicy::Classical);
                        unwrap_and_verify(&wire, &opts)
                    }));
                    let error = result
                        .expect("wire-controlled timestamp must never panic")
                        .expect_err("extreme timestamp must fail closed");
                    assert!(
                        error.to_string().contains(if extreme_iat == i64::MIN {
                            "timestamp too old"
                        } else {
                            "timestamp in future"
                        }),
                        "timestamp rejection must precede signature or KEM work: {error}"
                    );
                    assert!(
                        cache.seen.lock().is_empty(),
                        "rejected timestamp must not commit replay state"
                    );
                }
            }
        }
    }

    #[test]
    fn test_tampered_authorization_breaks_signature() {
        let (signing_key, verifying_key) = generate_signing_keypair();

        let envelope = RequestEnvelope {
            request_id: 100,
            payload: vec![1, 2, 3],
            nonce: fresh_test_nonce(),
            iat: current_timestamp(),
            authorization: Authorization::None,
            delegation_token: None,
            wth: None,
            client_dh_public: None,
            client_kem_public: None,
            response_kem_recipient: None,
            service_domain: None,
        };

        let mut signed = test_new_signed(envelope, &signing_key);

        // Tamper with the authorization after signing
        signed.envelope.authorization = Authorization::IdJag("evil-token".to_owned());

        // Signature verification must fail
        let result = signed.verify_signature_only(&verifying_key);
        assert!(
            result.is_err(),
            "Tampered authorization must invalidate signature"
        );
    }

    #[test]
    fn test_tampered_payload_breaks_signature() {
        let (signing_key, verifying_key) = generate_signing_keypair();

        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3]);
        let mut signed = test_new_signed(envelope, &signing_key);

        signed.envelope.payload = vec![9, 9, 9];

        let result = signed.verify_signature_only(&verifying_key);
        assert!(
            result.is_err(),
            "Tampered payload must invalidate signature"
        );
    }

    #[test]
    fn test_tampered_wth_breaks_signature() {
        let (signing_key, verifying_key) = generate_signing_keypair();

        let envelope = RequestEnvelope {
            request_id: 100,
            payload: vec![1, 2, 3],
            nonce: fresh_test_nonce(),
            iat: current_timestamp(),
            authorization: Authorization::None,
            delegation_token: None,
            wth: Some([0xAA; 32]),
            client_dh_public: None,
            client_kem_public: None,
            response_kem_recipient: None,
            service_domain: None,
        };

        let mut signed = test_new_signed(envelope, &signing_key);
        signed.envelope.wth = Some([0xBB; 32]);

        let result = signed.verify_signature_only(&verifying_key);
        assert!(result.is_err(), "Tampered wth must invalidate signature");
    }

    // =========================================================================
    // M3 (#152): COSE composite envelope — both schemes + cross-compat + anchor
    // =========================================================================

    use crate::crypto::CryptoPolicy;

    fn pq_store_for(
        ed_pubkey: [u8; 32],
        pq_vk: &crate::crypto::pq::MlDsaVerifyingKey,
    ) -> TestPqStore {
        let vk =
            crate::crypto::pq::ml_dsa_vk_from_bytes(&crate::crypto::pq::ml_dsa_vk_bytes(pq_vk))
                .expect("decode pq vk");
        TestPqStore {
            bindings: vec![(ed_pubkey, vk)],
        }
    }

    /// classical_only: sign + verify in Classical mode.
    #[test]
    fn m3_classical_only() -> crate::EnvelopeResult<()> {
        let (sk, vk) = generate_signing_keypair();
        let cache = TestNonceCache::new();
        let signed = SignedEnvelope::new_signed_with_policy(
            RequestEnvelope::anonymous(vec![1, 2, 3]),
            &sk,
            None,
            CryptoPolicy::Classical,
        );
        signed.verify_with(&vk, &cache, None, CryptoPolicy::Classical)?;
        Ok(())
    }

    /// hybrid: sign + verify composite (both EdDSA + ML-DSA-65 checked).
    #[test]
    fn m3_hybrid_both_checked() -> crate::EnvelopeResult<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();
        let signed = SignedEnvelope::new_signed_hybrid(
            RequestEnvelope::anonymous(vec![4, 5, 6]),
            &sk,
            &pq_sk,
        );
        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        signed.verify_with(&vk, &cache, Some(&store), CryptoPolicy::Hybrid)?;
        Ok(())
    }

    /// cross_pq_signer_classical_verifier: a Hybrid-signed envelope verifies
    /// under a Classical verifier via the EdDSA component + skip-unknown.
    #[test]
    fn m3_cross_hybrid_signer_classical_verifier() -> crate::EnvelopeResult<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, _pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();
        let signed = SignedEnvelope::new_signed_hybrid(
            RequestEnvelope::anonymous(vec![7, 8, 9]),
            &sk,
            &pq_sk,
        );
        // Classical verifier: no PQ store, Classical policy → skips ML-DSA entry.
        signed.verify_with(&vk, &cache, None, CryptoPolicy::Classical)?;
        Ok(())
    }

    /// cross_classical_signer_hybrid_verifier: a Classical-signed item from an
    /// ANCHORED signer under a Hybrid verifier — WNS enforces the outer layer for
    /// anchored identities, so a classical-only (no outer) envelope is rejected.
    #[test]
    fn m3_cross_classical_signer_hybrid_verifier_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let (_pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();
        let signed = SignedEnvelope::new_signed(RequestEnvelope::anonymous(vec![1]), &sk);
        // Signer's PQ key IS anchored → Hybrid must require its outer ML-DSA layer.
        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        let res = signed.verify_with(&vk, &cache, Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "Hybrid policy must reject a classical-only envelope from an anchored signer"
        );
    }

    /// Classical signer accepted by a verifier whose policy permits classical.
    #[test]
    fn m3_classical_signer_classical_policy_accepted() -> crate::EnvelopeResult<()> {
        let (sk, vk) = generate_signing_keypair();
        let cache = TestNonceCache::new();
        let signed = SignedEnvelope::new_signed(RequestEnvelope::anonymous(vec![1]), &sk);
        signed.verify_with(&vk, &cache, None, CryptoPolicy::Classical)?;
        Ok(())
    }

    /// self_cert_fix: an envelope whose ML-DSA key doesn't match the
    /// kid-resolved key is REJECTED (proves the kid-anchoring).
    #[test]
    fn m3_self_cert_fix_wrong_anchored_pq_key_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, _pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        // Attacker/foreign anchored key that does NOT match the signer's PQ key.
        let (_other_sk, other_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();
        let signed = SignedEnvelope::new_signed_hybrid(
            RequestEnvelope::anonymous(vec![1, 2, 3]),
            &sk,
            &pq_sk,
        );
        let store = pq_store_for(vk.to_bytes(), &other_vk);
        let res = signed.verify_with(&vk, &cache, Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "PQ key not matching the kid-anchored key must be rejected (self-cert fix)"
        );
    }

    /// Hybrid policy with NO anchored key falls back to the classical inner
    /// EdDSA floor (WNS per-identity): an unanchored signer is verified via its
    /// EdDSA component rather than failing closed. PQ is never trusted from the
    /// self-asserted COSE entry, so this is no weaker than classical.
    #[test]
    fn m3_hybrid_policy_without_anchor_classical_fallback() -> crate::EnvelopeResult<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, _pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();
        let signed =
            SignedEnvelope::new_signed_hybrid(RequestEnvelope::anonymous(vec![1]), &sk, &pq_sk);
        // Empty store (no anchor) + Hybrid policy → classical inner-EdDSA fallback.
        signed.verify_with(&vk, &cache, None, CryptoPolicy::Hybrid)?;
        Ok(())
    }

    // -- SNS-specific: strip the outer ML-DSA layer at the cose level and prove
    //    Hybrid policy rejects it (the attack the review found accepted before).
    #[test]
    fn m3_sns_strip_outer_mldsa_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();
        let mut signed = SignedEnvelope::new_signed_hybrid(
            RequestEnvelope::anonymous(vec![1, 2, 3]),
            &sk,
            &pq_sk,
        );
        // Strip: re-encode the composite with the outer set to null.
        let (inner, outer) =
            crate::crypto::cose_sign::decode_composite_for_test(&signed.cose).expect("decode");
        assert!(outer.is_some(), "hybrid envelope must have an outer layer");
        signed.cose =
            crate::crypto::cose_sign::encode_composite_for_test(inner, None).expect("encode");

        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        let res = signed.verify_with(&vk, &cache, Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "stripping the outer ML-DSA layer must fail Hybrid policy"
        );
    }

    /// Integration at the `unwrap_and_verify` level (prod-wiring path the review
    /// flagged as missing): sign Hybrid, strip the outer, serialize to wire, and
    /// assert `unwrap_and_verify` REJECTS under a Hybrid deployment.
    #[test]
    fn m3_unwrap_and_verify_strip_rejected_under_hybrid() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();

        // Sign Hybrid, then strip the outer layer.
        let mut signed = SignedEnvelope::new_signed_hybrid(
            RequestEnvelope::anonymous(vec![9, 9, 9]),
            &sk,
            &pq_sk,
        );
        let (inner, _outer) =
            crate::crypto::cose_sign::decode_composite_for_test(&signed.cose).expect("decode");
        signed.cose =
            crate::crypto::cose_sign::encode_composite_for_test(inner, None).expect("encode");

        // Serialize to wire bytes.
        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder = message.init_root::<common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }
        let mut wire = Vec::new();
        capnp::serialize::write_message(&mut wire, &message).expect("serialize");

        // Verify under a Hybrid deployment via unwrap_and_verify.
        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        let opts = UnwrapOptions::fixed_signer(&vk, &cache)
            .with_verify_policy(CryptoPolicy::Hybrid)
            .with_pq_store(&store);
        let res = unwrap_and_verify(&wire, &opts);
        assert!(
            res.is_err(),
            "unwrap_and_verify must reject a stripped Hybrid envelope (prod fail-open closed)"
        );
    }

    /// Integration: a well-formed Hybrid envelope passes unwrap_and_verify under
    /// a Hybrid deployment with the correct anchor.
    #[test]
    fn m3_unwrap_and_verify_hybrid_round_trip() -> crate::EnvelopeResult<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();

        let signed =
            SignedEnvelope::new_signed_hybrid(RequestEnvelope::anonymous(vec![4, 2]), &sk, &pq_sk);
        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder = message.init_root::<common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }
        let mut wire = Vec::new();
        capnp::serialize::write_message(&mut wire, &message).expect("serialize");

        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        let opts = UnwrapOptions::fixed_signer(&vk, &cache)
            .with_verify_policy(CryptoPolicy::Hybrid)
            .with_pq_store(&store);
        let (_signed, payload) = unwrap_and_verify(&wire, &opts)
            .map_err(|e| EnvelopeError::PqSignatureInvalid(e.to_string()))?;
        assert_eq!(payload, vec![4, 2]);
        Ok(())
    }

    // ── Fu2/#677: unanchored identity under Hybrid ──────────────────────────

    /// Helper: a Hybrid-signed envelope on the wire (no PQ anchor installed).
    fn fu2_hybrid_wire_unanchored() -> ((SigningKey, VerifyingKey), Vec<u8>) {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, _pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let signed = SignedEnvelope::new_signed_hybrid(
            RequestEnvelope::anonymous(vec![5, 0, 2]),
            &sk,
            &pq_sk,
        );
        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder = message.init_root::<common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }
        let mut wire = Vec::new();
        capnp::serialize::write_message(&mut wire, &message).expect("serialize");
        ((sk, vk), wire)
    }

    /// Fu2/#677 default: an unanchored identity (no ML-DSA-65 binding) is
    /// DENIED under Hybrid via the production wire-verify path, even though the
    /// inner EdDSA is valid. The previous behavior accepted it (WNS classical
    /// floor); the default is now deny-on-missing-anchor.
    #[test]
    fn fu2_unanchored_denied_by_default_under_hybrid() {
        let ((_sk, vk), wire) = fu2_hybrid_wire_unanchored();
        let cache = TestNonceCache::new();
        // Hybrid + NO pq_store (unanchored) + default Deny posture.
        let opts =
            UnwrapOptions::fixed_signer(&vk, &cache).with_verify_policy(CryptoPolicy::Hybrid);
        let res = unwrap_and_verify(&wire, &opts);
        assert!(
            res.is_err(),
            "an unanchored identity MUST be denied under Hybrid by default (Fu2/#677)"
        );
    }

    /// Fu2/#677 allowlist: the same unanchored identity IS accepted when its
    /// Ed25519 key hex is on the classical-floor allowlist (the legacy-peer
    /// escape hatch).
    #[test]
    fn fu2_unanchored_allowed_via_allowlist() {
        let ((_sk, vk), wire) = fu2_hybrid_wire_unanchored();
        let cache = TestNonceCache::new();
        let opts = UnwrapOptions::fixed_signer(&vk, &cache)
            .with_verify_policy(CryptoPolicy::Hybrid)
            .with_unanchored_allowlist(vec![hex::encode(vk.to_bytes())]);
        let (_signed, payload) = unwrap_and_verify(&wire, &opts)
            .expect("an allowlisted unanchored identity is accepted (classical floor)");
        assert_eq!(payload, vec![5, 0, 2]);
    }

    /// Fu2/#677 explicit opt-in: `AllowClassicalFloor` accepts an unanchored
    /// identity under Hybrid without an allowlist (the legacy WNS posture,
    /// available to low-level callers that want it).
    #[test]
    fn fu2_unanchored_allowed_via_legacy_policy() {
        let ((_sk, vk), wire) = fu2_hybrid_wire_unanchored();
        let cache = TestNonceCache::new();
        let opts = UnwrapOptions::fixed_signer(&vk, &cache)
            .with_verify_policy(CryptoPolicy::Hybrid)
            .with_unanchored_policy(UnanchoredHybridPolicy::AllowClassicalFloor);
        let (_signed, _payload) = unwrap_and_verify(&wire, &opts)
            .expect("AllowClassicalFloor accepts an unanchored identity (legacy posture)");
    }

    // =========================================================================
    // #275: ResponseEnvelope COSE composite (Hybrid parity with SignedEnvelope)
    // =========================================================================

    /// Serialize a ResponseEnvelope to wire bytes (mirrors the prod path).
    fn response_to_wire(env: &ResponseEnvelope) -> Vec<u8> {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder = message.init_root::<common_capnp::response_envelope::Builder>();
            env.write_to(&mut builder);
        }
        let mut wire = Vec::new();
        capnp::serialize::write_message(&mut wire, &message).expect("serialize response");
        wire
    }

    #[test]
    fn exact_flat_parser_rejects_trailing_and_partial_frames() {
        let (sk, _) = generate_signing_keypair();
        let wire = response_to_wire(&ResponseEnvelope::new_signed(1, vec![1, 2, 3], &sk));

        let mut trailing = wire.clone();
        trailing.extend_from_slice(&[0u8; 8]);
        assert!(
            read_exact_envelope_message(&trailing).is_err(),
            "a second/trailing frame must not be ignored"
        );

        let partial = &wire[..wire.len() - 1];
        assert!(
            read_exact_envelope_message(partial).is_err(),
            "a truncated declared segment must be rejected"
        );
    }

    #[test]
    fn exact_flat_parser_rejects_oversized_and_malformed_segment_tables() {
        // One segment declaring 131,073 words: one word over the canonical
        // 1 MiB traversal cap. No body is supplied; rejection must occur from
        // the bounded table parse without allocating the declared body.
        let oversized = [0, 0, 0, 0, 1, 0, 2, 0];
        assert!(read_exact_envelope_message(&oversized).is_err());

        // u32::MAX + 1 wraps to an invalid zero segment count.
        let malformed_count = [0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0];
        assert!(read_exact_envelope_message(&malformed_count).is_err());
    }

    /// Classical round-trip: sign + verify a response (EdDSA-only COSE).
    #[test]
    fn resp_classical_round_trip() -> anyhow::Result<()> {
        let (sk, vk) = generate_signing_keypair();
        let resp = ResponseEnvelope::new_signed(7, vec![1, 2, 3], &sk);
        // Direct verify.
        resp.verify(Some(&vk))?;
        // Via the wire + unwrap_response (Classical).
        let wire = response_to_wire(&resp);
        let (rid, payload) = unwrap_response(&wire, Some(&vk))?;
        assert_eq!(rid, 7);
        assert_eq!(payload, vec![1, 2, 3]);
        Ok(())
    }

    /// Hybrid round-trip: sign + verify a response (EdDSA + ML-DSA-65 anchored).
    #[test]
    fn resp_hybrid_round_trip() -> anyhow::Result<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let resp = ResponseEnvelope::new_signed_hybrid(9, vec![4, 5, 6], &sk, &pq_sk);
        let store = pq_store_for(vk.to_bytes(), &pq_vk);

        resp.verify_with(Some(&vk), Some(&store), CryptoPolicy::Hybrid)?;

        let wire = response_to_wire(&resp);
        let (rid, payload) =
            unwrap_response_with(&wire, Some(&vk), Some(&store), CryptoPolicy::Hybrid)?;
        assert_eq!(rid, 9);
        assert_eq!(payload, vec![4, 5, 6]);
        Ok(())
    }

    // =========================================================================
    // #277: process-global RESPONSE verify config — fail-closed Hybrid default,
    // anchored-key enforcement, and the shared `classical` escape hatch.
    // =========================================================================

    /// The escape-hatch parser is the single source of truth shared by the
    /// request and response sides: unset / `hybrid` => fail-closed Hybrid, the
    /// literal `classical` => downgrade, junk => Hybrid (fail-safe).
    ///
    /// `#[cfg(not(target_arch = "wasm32"))]` because `envelope_policy_from_env`
    /// is native-only (it reads a process env var). Serialized via a mutex with
    /// the other env-mutating test so they don't race on the shared env var.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn resp_escape_hatch_env_parse_parity() {
        let _guard = ENV_TEST_LOCK.lock();
        // SAFETY: single-threaded section guarded by ENV_TEST_LOCK.
        unsafe { std::env::remove_var(ENVELOPE_POLICY_ENV) };
        assert_eq!(
            envelope_policy_from_env(),
            CryptoPolicy::Hybrid,
            "default must be Hybrid"
        );
        unsafe { std::env::set_var(ENVELOPE_POLICY_ENV, "hybrid") };
        assert_eq!(envelope_policy_from_env(), CryptoPolicy::Hybrid);
        unsafe { std::env::set_var(ENVELOPE_POLICY_ENV, "classical") };
        assert_eq!(
            envelope_policy_from_env(),
            CryptoPolicy::Classical,
            "the escape hatch must downgrade the response side in parity with the request side"
        );
        unsafe { std::env::set_var(ENVELOPE_POLICY_ENV, "bogus") };
        assert_eq!(
            envelope_policy_from_env(),
            CryptoPolicy::Hybrid,
            "junk must fail-safe to Hybrid"
        );
        unsafe { std::env::remove_var(ENVELOPE_POLICY_ENV) };
    }

    /// Process-global RESPONSE verify config drives per-identity enforcement
    /// (#277, WNS posture): once a Hybrid config + admin-anchored
    /// `KeyedPqTrustStore` is installed, a Hybrid response whose server ML-DSA key
    /// IS anchored verifies with the outer enforced; an UNanchored signer falls
    /// back to its inner EdDSA (classical floor); but an ANCHORED signer that
    /// sends a classical-only response is rejected (it must use its PQ key — no
    /// downgrade for anchored identities). This is the install/consult path native
    /// RPC clients fall back to when no per-client store was set. Single OnceLock
    /// install per test binary, so this is the one test that installs the config.
    #[test]
    fn resp_global_config_anchored_enforced_unanchored_falls_back() -> anyhow::Result<()> {
        // Anchored signer.
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        // A different signer whose ML-DSA key is NOT bound in the store.
        let (sk2, vk2) = generate_signing_keypair();
        let (pq_sk2, _pq_vk2) = crate::crypto::pq::ml_dsa_generate_keypair();

        // Admin-anchored store (#157 reuse): only the first signer's key is bound.
        let mut keyed = KeyedPqTrustStore::new();
        keyed.bind(vk.to_bytes(), &pq_vk);
        install_response_verify_config(ResponseVerifyConfig {
            policy: CryptoPolicy::Hybrid,
            pq_store: Some(std::sync::Arc::new(keyed)),
        })
        .expect("install response verify config once");

        assert_eq!(global_response_verify_policy(), CryptoPolicy::Hybrid);
        let store = global_response_pq_store().expect("global response store installed");

        // Anchored Hybrid response verifies under the global config.
        let resp = ResponseEnvelope::new_signed_hybrid(11, vec![7, 7], &sk, &pq_sk);
        let wire = response_to_wire(&resp);
        let (rid, payload) = unwrap_response_with(
            &wire,
            Some(&vk),
            Some(store.as_ref()),
            global_response_verify_policy(),
        )?;
        assert_eq!(rid, 11);
        assert_eq!(payload, vec![7, 7]);

        // Unanchored signer's Hybrid response falls back to its inner EdDSA
        // (classical floor) under the WNS posture — no anchored ML-DSA key, so PQ
        // is not enforced for this identity, but the response still verifies.
        let resp2 = ResponseEnvelope::new_signed_hybrid(12, vec![8, 8], &sk2, &pq_sk2);
        let wire2 = response_to_wire(&resp2);
        let (rid2, payload2) = unwrap_response_with(
            &wire2,
            Some(&vk2),
            Some(store.as_ref()),
            global_response_verify_policy(),
        )?;
        assert_eq!(rid2, 12);
        assert_eq!(payload2, vec![8, 8]);

        // A Classical-only (inner-EdDSA-only) response from the ANCHORED signer is
        // rejected: an anchored identity must use its PQ key (the outer is enforced
        // per-identity), so a stripped/absent outer on an anchored signer downgrades.
        let resp_classical = ResponseEnvelope::new_signed(13, vec![9], &sk);
        let wire3 = response_to_wire(&resp_classical);
        let res3 = unwrap_response_with(
            &wire3,
            Some(&vk),
            Some(store.as_ref()),
            global_response_verify_policy(),
        );
        assert!(
            res3.is_err(),
            "classical-only response from an anchored signer must be rejected"
        );
        Ok(())
    }

    /// A Hybrid-signed response verifies under a Classical verifier via its
    /// inner EdDSA (skip-unknown interop) — the default-policy interop path.
    #[test]
    fn resp_hybrid_signed_classical_verifier_accepted() -> anyhow::Result<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, _pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let resp = ResponseEnvelope::new_signed_hybrid(1, vec![1], &sk, &pq_sk);
        resp.verify(Some(&vk))?; // Classical, no store
        Ok(())
    }

    /// Stripping the outer ML-DSA layer under Hybrid policy is rejected.
    #[test]
    fn resp_strip_outer_mldsa_rejected_under_hybrid() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let mut resp = ResponseEnvelope::new_signed_hybrid(2, vec![2, 2], &sk, &pq_sk);
        let (inner, outer) =
            crate::crypto::cose_sign::decode_composite_for_test(&resp.cose).expect("decode");
        assert!(outer.is_some(), "hybrid response must have an outer layer");
        resp.cose =
            crate::crypto::cose_sign::encode_composite_for_test(inner, None).expect("encode");

        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        let res = resp.verify_with(Some(&vk), Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "stripping the outer ML-DSA layer must fail Hybrid policy"
        );
    }

    /// Tampering the inner EdDSA signature invalidates the Hybrid composite.
    #[test]
    fn resp_tamper_inner_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let resp = ResponseEnvelope::new_signed_hybrid(3, vec![3, 3, 3], &sk, &pq_sk);
        let (_inner, outer) =
            crate::crypto::cose_sign::decode_composite_for_test(&resp.cose).expect("decode");
        // Forge a new inner with a different key; keep the original outer.
        let (other_sk, _other_vk) = generate_signing_keypair();
        let other_resp = ResponseEnvelope::new_signed(3, vec![3, 3, 3], &other_sk);
        let (forged_inner, _none) =
            crate::crypto::cose_sign::decode_composite_for_test(&other_resp.cose).expect("decode");
        let mut tampered = resp.clone();
        tampered.cose = crate::crypto::cose_sign::encode_composite_for_test(forged_inner, outer)
            .expect("encode");

        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        let res = tampered.verify_with(Some(&vk), Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "tampering the inner EdDSA must invalidate the response composite"
        );
    }

    /// Tampering the payload invalidates the signature.
    #[test]
    fn resp_tamper_payload_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let mut resp = ResponseEnvelope::new_signed(4, vec![1, 2, 3], &sk);
        resp.payload = vec![9, 9, 9];
        assert!(
            resp.verify(Some(&vk)).is_err(),
            "tampered payload must invalidate the response"
        );
    }

    /// self-cert: anchored ML-DSA key not matching the signer's PQ key → reject.
    #[test]
    fn resp_self_cert_wrong_anchor_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, _pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let (_other_sk, other_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let resp = ResponseEnvelope::new_signed_hybrid(5, vec![5], &sk, &pq_sk);
        let store = pq_store_for(vk.to_bytes(), &other_vk);
        let res = resp.verify_with(Some(&vk), Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "wrong anchored PQ key must be rejected (self-cert fix)"
        );
    }

    /// Hybrid policy with NO anchored key falls back to the inner EdDSA classical
    /// floor (WNS per-identity): an unanchored response signer verifies via its
    /// EdDSA component rather than failing closed. PQ is never trusted from the
    /// self-asserted COSE entry, so this is no weaker than classical.
    #[test]
    fn resp_hybrid_without_anchor_classical_fallback() -> anyhow::Result<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, _pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let resp = ResponseEnvelope::new_signed_hybrid(6, vec![6], &sk, &pq_sk);
        resp.verify_with(Some(&vk), None, CryptoPolicy::Hybrid)?;
        Ok(())
    }

    /// classical-signed response under Hybrid policy → rejected (no downgrade).
    #[test]
    fn resp_classical_signed_hybrid_verifier_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let (_pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let resp = ResponseEnvelope::new_signed(8, vec![8], &sk);
        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        let res = resp.verify_with(Some(&vk), Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "Hybrid policy must reject a classical-only response"
        );
    }

    // =========================================================================
    // WNS posture (draft-ietf-pquip-hybrid-signature-spectrums): per-identity PQ
    // enforcement. Anchored identities cannot downgrade; unanchored identities
    // fall back to the classical inner-EdDSA floor instead of failing closed.
    // Covers BOTH SignedEnvelope and ResponseEnvelope paths.
    // =========================================================================

    /// WNS / SignedEnvelope (1) anchored downgrade rejected: anchor the signer's
    /// correct ML-DSA key, strip the outer layer, verify under Hybrid → REJECTED.
    #[test]
    fn wns_signed_anchored_downgrade_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();
        let mut signed = SignedEnvelope::new_signed_hybrid(
            RequestEnvelope::anonymous(vec![1, 2, 3]),
            &sk,
            &pq_sk,
        );
        let (inner, outer) =
            crate::crypto::cose_sign::decode_composite_for_test(&signed.cose).expect("decode");
        assert!(outer.is_some(), "hybrid envelope must carry an outer layer");
        signed.cose =
            crate::crypto::cose_sign::encode_composite_for_test(inner, None).expect("encode");
        // Signer's PQ key IS anchored → the outer is enforced; stripping it fails.
        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        let res = signed.verify_with(&vk, &cache, Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "anchored identity must not be downgradable by stripping the outer"
        );
    }

    /// WNS / SignedEnvelope (2) unanchored classical fallback: empty store, Hybrid
    /// policy, Hybrid-signed envelope from an UNanchored signer → SUCCEEDS via the
    /// inner EdDSA floor (the deploy-blocker fix).
    #[test]
    fn wns_signed_unanchored_classical_fallback() -> crate::EnvelopeResult<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, _pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let cache = TestNonceCache::new();
        let signed = SignedEnvelope::new_signed_hybrid(
            RequestEnvelope::anonymous(vec![4, 5, 6]),
            &sk,
            &pq_sk,
        );
        let empty = TestPqStore { bindings: vec![] };
        signed.verify_with(&vk, &cache, Some(&empty), CryptoPolicy::Hybrid)?;
        // Also with no store at all (None), mirroring an empty default deployment.
        let cache2 = TestNonceCache::new();
        signed.verify_with(&vk, &cache2, None, CryptoPolicy::Hybrid)?;
        Ok(())
    }

    /// WNS / SignedEnvelope (3) anchored enforced: anchored signer + intact Hybrid
    /// succeeds; anchored signer + classical-only (no outer) is rejected.
    #[test]
    fn wns_signed_anchored_enforced() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let store = pq_store_for(vk.to_bytes(), &pq_vk);

        let cache = TestNonceCache::new();
        let intact =
            SignedEnvelope::new_signed_hybrid(RequestEnvelope::anonymous(vec![7]), &sk, &pq_sk);
        intact
            .verify_with(&vk, &cache, Some(&store), CryptoPolicy::Hybrid)
            .expect("intact hybrid from an anchored signer must verify");

        let cache2 = TestNonceCache::new();
        let classical = SignedEnvelope::new_signed(RequestEnvelope::anonymous(vec![8]), &sk);
        let res = classical.verify_with(&vk, &cache2, Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "anchored signer must use its PQ key — classical-only is rejected"
        );
    }

    /// WNS / ResponseEnvelope (1) anchored downgrade rejected.
    #[test]
    fn wns_response_anchored_downgrade_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let mut resp = ResponseEnvelope::new_signed_hybrid(20, vec![2, 0], &sk, &pq_sk);
        let (inner, outer) =
            crate::crypto::cose_sign::decode_composite_for_test(&resp.cose).expect("decode");
        assert!(outer.is_some(), "hybrid response must carry an outer layer");
        resp.cose =
            crate::crypto::cose_sign::encode_composite_for_test(inner, None).expect("encode");
        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        let res = resp.verify_with(Some(&vk), Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "anchored response identity must not be downgradable"
        );
    }

    /// WNS / ResponseEnvelope (2) unanchored classical fallback (mirrors the
    /// SignedEnvelope case for the response path specifically).
    #[test]
    fn wns_response_unanchored_classical_fallback() -> anyhow::Result<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, _pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let resp = ResponseEnvelope::new_signed_hybrid(21, vec![2, 1], &sk, &pq_sk);
        let empty = TestPqStore { bindings: vec![] };
        resp.verify_with(Some(&vk), Some(&empty), CryptoPolicy::Hybrid)?;
        // And with no store at all.
        resp.verify_with(Some(&vk), None, CryptoPolicy::Hybrid)?;
        Ok(())
    }

    /// WNS / ResponseEnvelope (3) anchored enforced.
    #[test]
    fn wns_response_anchored_enforced() {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let store = pq_store_for(vk.to_bytes(), &pq_vk);

        let intact = ResponseEnvelope::new_signed_hybrid(22, vec![2, 2], &sk, &pq_sk);
        intact
            .verify_with(Some(&vk), Some(&store), CryptoPolicy::Hybrid)
            .expect("intact hybrid response from an anchored signer must verify");

        let classical = ResponseEnvelope::new_signed(23, vec![2, 3], &sk);
        let res = classical.verify_with(Some(&vk), Some(&store), CryptoPolicy::Hybrid);
        assert!(
            res.is_err(),
            "anchored response signer must use its PQ key — classical-only rejected"
        );
    }

    /// Wrong expected signer key → rejected.
    #[test]
    fn resp_wrong_signer_rejected() {
        let (sk, _vk) = generate_signing_keypair();
        let (_other_sk, other_vk) = generate_signing_keypair();
        let resp = ResponseEnvelope::new_signed(10, vec![1], &sk);
        assert!(
            resp.verify(Some(&other_vk)).is_err(),
            "wrong expected signer must be rejected"
        );
    }

    /// Cap'n Proto round-trip preserves the cose field.
    #[test]
    fn resp_capnp_roundtrip_preserves_cose() -> anyhow::Result<()> {
        let (sk, vk) = generate_signing_keypair();
        let (pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let resp = ResponseEnvelope::new_signed_hybrid(11, vec![7, 7], &sk, &pq_sk);
        let wire = response_to_wire(&resp);
        let reader = capnp::serialize::read_message(
            &mut std::io::Cursor::new(&wire),
            envelope_reader_options(),
        )?;
        let r = reader.get_root::<common_capnp::response_envelope::Reader>()?;
        let decoded = ResponseEnvelope::read_from(r)?;
        assert_eq!(
            decoded.cose, resp.cose,
            "cose must survive the capnp round-trip"
        );
        let store = pq_store_for(vk.to_bytes(), &pq_vk);
        decoded.verify_with(Some(&vk), Some(&store), CryptoPolicy::Hybrid)?;
        Ok(())
    }

    // -- Cross-direction replay (the load-bearing domain-separation guard) -----

    /// A REQUEST COSE composite must NOT verify against the RESPONSE AAD, and a
    /// RESPONSE COSE composite must NOT verify against the REQUEST AAD. Distinct
    /// type-ids in `external_aad` enforce this: the bytes are interchangeable but
    /// the AAD binding differs, so each direction's signature is non-fungible.
    #[test]
    fn cross_direction_replay_rejected() {
        let (sk, vk) = generate_signing_keypair();
        let payload = b"shared bytes";

        // Build a REQUEST-domain composite and a RESPONSE-domain composite over
        // the same payload bytes, same key.
        let req_aad = envelope_external_aad();
        let resp_aad = response_envelope_external_aad();
        assert_ne!(
            req_aad, resp_aad,
            "request/response AADs must differ (domain separation)"
        );

        let req_cose = crate::crypto::cose_sign::sign_composite(&sk, None, payload, &req_aad)
            .expect("sign req");
        let resp_cose = crate::crypto::cose_sign::sign_composite(&sk, None, payload, &resp_aad)
            .expect("sign resp");

        // The request signature verifies under the request AAD …
        crate::crypto::cose_sign::verify_composite(&req_cose, &vk, None, payload, &req_aad, false)
            .expect("request verifies under request AAD");
        // … but MUST NOT verify under the response AAD (replay as a response).
        let cross1 = crate::crypto::cose_sign::verify_composite(
            &req_cose, &vk, None, payload, &resp_aad, false,
        );
        assert!(
            cross1.is_err(),
            "a request COSE sig must not verify as a response"
        );

        // And symmetrically for the response signature.
        crate::crypto::cose_sign::verify_composite(
            &resp_cose, &vk, None, payload, &resp_aad, false,
        )
        .expect("response verifies under response AAD");
        let cross2 = crate::crypto::cose_sign::verify_composite(
            &resp_cose, &vk, None, payload, &req_aad, false,
        );
        assert!(
            cross2.is_err(),
            "a response COSE sig must not verify as a request"
        );
    }

    /// End-to-end cross-direction: a `SignedEnvelope`'s COSE must not satisfy a
    /// `ResponseEnvelope::verify_with`, proving the request and response signing
    /// domains are separated at the envelope API level (not just the AAD helper).
    #[test]
    fn cross_direction_request_cose_rejected_as_response() {
        let (sk, vk) = generate_signing_keypair();
        // A request envelope's signing-data is canonical(RequestEnvelope); a
        // response's is request_id||payload. Even constructing a response whose
        // signing-data happens to match, the request-domain AAD differs, so the
        // composite cannot be reused. Here we directly graft a request COSE onto
        // a response and confirm rejection.
        let req = SignedEnvelope::new_signed(RequestEnvelope::anonymous(vec![1, 2, 3]), &sk);
        let mut resp = ResponseEnvelope::new_signed(0, vec![1, 2, 3], &sk);
        resp.cose = req.cose.clone(); // graft the request composite onto the response
        let res = resp.verify(Some(&vk));
        assert!(
            res.is_err(),
            "a request-domain COSE grafted onto a response must be rejected"
        );
    }

    #[test]
    fn sealed_response_roundtrip_is_transcript_bound_and_not_plaintext_on_wire() {
        let (server_sk, server_vk) = generate_signing_keypair();
        let pq_sk = crate::node_identity::derive_mesh_mldsa_key(&server_sk);
        let pq_vk = crate::crypto::pq::ml_dsa_vk_from_bytes(
            &crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk),
        )
        .unwrap();
        let store = pq_store_for(server_vk.to_bytes(), &pq_vk);
        let recipient = crate::crypto::hybrid_kem::generate_recipient(
            crate::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
        )
        .unwrap();
        let public = recipient.public();
        let nonce = fresh_test_nonce();
        let secret_payload = b"response-plaintext-sentinel";
        let response = ResponseEnvelope::new_signed_encrypted(
            77,
            secret_payload.to_vec(),
            &server_sk,
            &pq_sk,
            &public,
            1234,
            &nonce,
            "service-a",
        )
        .unwrap();
        response
            .verify_encrypted_with_service_domain(&server_vk, &store, "service-a")
            .unwrap();
        let wire = response_to_wire(&response);
        assert!(!wire
            .windows(secret_payload.len())
            .any(|w| w == secret_payload));
        let public_unwrap =
            unwrap_response_with(&wire, Some(&server_vk), Some(&store), CryptoPolicy::Hybrid)
                .expect_err("public unwrap has no one-shot decapsulation state");
        assert!(public_unwrap.to_string().contains("decapsulation material"));
        let opened = response
            .open_encrypted(
                &recipient,
                &public,
                1234,
                &nonce,
                &server_vk.to_bytes(),
                "service-a",
            )
            .unwrap();
        assert_eq!(opened, secret_payload);

        let wrong_recipient = crate::crypto::hybrid_kem::generate_recipient(
            crate::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
        )
        .unwrap();
        assert!(response
            .open_encrypted(
                &wrong_recipient,
                &wrong_recipient.public(),
                1234,
                &nonce,
                &server_vk.to_bytes(),
                "service-a",
            )
            .is_err());
        assert!(response
            .open_encrypted(
                &recipient,
                &public,
                1235,
                &nonce,
                &server_vk.to_bytes(),
                "service-a",
            )
            .is_err());
        let other_nonce = distinct_test_nonce(&nonce);
        assert!(response
            .open_encrypted(
                &recipient,
                &public,
                1234,
                &other_nonce,
                &server_vk.to_bytes(),
                "service-a",
            )
            .is_err());
        let (other_server, _) = generate_signing_keypair();
        assert!(response
            .open_encrypted(
                &recipient,
                &public,
                1234,
                &nonce,
                &other_server.verifying_key().to_bytes(),
                "service-a",
            )
            .is_err());
        assert!(response
            .verify_encrypted_with_service_domain(&server_vk, &store, "service-b")
            .is_err());
        assert!(response
            .open_encrypted(
                &recipient,
                &public,
                1234,
                &nonce,
                &server_vk.to_bytes(),
                "service-b",
            )
            .is_err());
    }

    #[test]
    fn sealed_response_rejects_request_id_substitution_and_validly_signed_garbage() {
        let (server_sk, server_vk) = generate_signing_keypair();
        let pq_sk = crate::node_identity::derive_mesh_mldsa_key(&server_sk);
        let pq_vk = crate::crypto::pq::ml_dsa_vk_from_bytes(
            &crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk),
        )
        .unwrap();
        let store = pq_store_for(server_vk.to_bytes(), &pq_vk);
        let recipient = crate::crypto::hybrid_kem::generate_recipient(
            crate::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
        )
        .unwrap();
        let public = recipient.public();
        let nonce = fresh_test_nonce();
        let mut response = ResponseEnvelope::new_signed_encrypted(
            8,
            b"bound".to_vec(),
            &server_sk,
            &pq_sk,
            &public,
            99,
            &nonce,
            "service-a",
        )
        .unwrap();

        response.request_id = 9;
        let data = ResponseEnvelope::encrypted_signing_data(
            response.request_id,
            response.encrypted_response.as_deref().unwrap(),
            "service-a",
        )
        .unwrap();
        response.sig = server_sk.sign(&data).to_bytes();
        response.cose =
            ResponseEnvelope::build_cose(&server_sk, Some(&pq_sk), CryptoPolicy::Hybrid, &data)
                .unwrap();
        response
            .verify_encrypted_with_service_domain(&server_vk, &store, "service-a")
            .unwrap();
        assert!(response
            .open_encrypted(
                &recipient,
                &public,
                99,
                &nonce,
                &server_vk.to_bytes(),
                "service-a",
            )
            .is_err());

        response.encrypted_response = Some(b"not-a-cose-encrypt0".to_vec());
        let data = ResponseEnvelope::encrypted_signing_data(
            9,
            response.encrypted_response.as_deref().unwrap(),
            "service-a",
        )
        .unwrap();
        response.sig = server_sk.sign(&data).to_bytes();
        response.cose =
            ResponseEnvelope::build_cose(&server_sk, Some(&pq_sk), CryptoPolicy::Hybrid, &data)
                .unwrap();
        response
            .verify_encrypted_with_service_domain(&server_vk, &store, "service-a")
            .unwrap();
        assert!(response
            .open_encrypted(
                &recipient,
                &public,
                99,
                &nonce,
                &server_vk.to_bytes(),
                "service-a",
            )
            .is_err());
    }

    #[test]
    fn cleartext_response_is_refused_by_early_network_gate() {
        let (sk, _) = generate_signing_keypair();
        let wire = response_to_wire(&ResponseEnvelope::new_signed(
            1,
            b"must-not-surface".to_vec(),
            &sk,
        ));
        assert!(read_response_envelope(&wire, true).is_err());
    }

    #[test]
    fn response_recipient_and_ciphertext_sizes_are_bounded() {
        let mut request_message = capnp::message::Builder::new_default();
        {
            let mut builder =
                request_message.init_root::<crate::common_capnp::request_envelope::Builder>();
            builder.set_nonce(&fresh_test_nonce());
            builder.set_response_kem_recipient(&vec![0u8; MAX_RESPONSE_KEM_RECIPIENT_BYTES + 1]);
        }
        let request_message_reader = request_message.into_reader();
        let request_reader = request_message_reader
            .get_root::<crate::common_capnp::request_envelope::Reader>()
            .unwrap();
        assert!(RequestEnvelope::read_from(request_reader).is_err());

        let mut response_message = capnp::message::Builder::new_default();
        {
            let mut builder =
                response_message.init_root::<crate::common_capnp::response_envelope::Builder>();
            builder.set_sig(&[0u8; 64]);
            builder.set_cnf(&[0u8; 32]);
            builder.set_encrypted_response(&vec![0u8; MAX_ENCRYPTED_RESPONSE_BYTES + 1]);
        }
        let response_message_reader = response_message.into_reader();
        let response_reader = response_message_reader
            .get_root::<crate::common_capnp::response_envelope::Reader>()
            .unwrap();
        assert!(ResponseEnvelope::read_from(response_reader).is_err());
    }
}
