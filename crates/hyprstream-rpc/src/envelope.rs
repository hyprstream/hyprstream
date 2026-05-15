//! Request envelope for identity-aware RPC with Ed25519 signatures.
//!
//! Every RPC request is wrapped in a `SignedEnvelope` that carries:
//! - A unique request ID for correlation
//! - The identity of the requester (for authorization)
//! - The serialized inner request payload
//! - Ed25519 signature over the entire RequestEnvelope
//! - Nonce + timestamp for replay protection
//!
//! # Two-Layer Security
//!
//! | Layer | Mechanism | Purpose |
//! |-------|-----------|---------|
//! | Transport | CURVE | Encrypts connection, authenticates immediate peer |
//! | Application | Signed envelope | Authenticates request originator, survives forwarding |
//!
//! # Identity → Subject Mapping
//!
//! All identity types produce bare username subjects for Casbin policy checks:
//!
//! | Identity | Subject | Example |
//! |----------|---------|---------|
//! | Local | `<username>` | `"alice"` |
//! | ApiToken | `<username>` | `"bob"` |
//! | Peer | `<name>` | `"gpu-server-1"` |
//! | Anonymous | `anonymous` | `"anonymous"` |
//!
//! # Nested Envelope Structure
//!
//! ```text
//! SignedEnvelope {
//!     envelope: RequestEnvelope {  // This is what gets signed
//!         request_id, identity, payload,
//!         ephemeral_pubkey, nonce, timestamp
//!     },
//!     signature,      // Ed25519(signing_key, serialize(envelope))
//!     signer_pubkey,  // Ed25519 public key
//! }
//! ```
//!
//! The nested structure makes clear exactly what is being signed.

use crate::auth::Claims;
use crate::capnp::{FromCapnp, ToCapnp};
use crate::common_capnp;
use crate::crypto::{SigningKey, VerifyingKey};
use crate::error::{EnvelopeError, EnvelopeResult};
use anyhow::{anyhow, Result};
use ed25519_dalek::{Signature, Signer};
use subtle::ConstantTimeEq;
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

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
            if p.is_empty() { None } else { Some(p.to_owned()) }
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
    use rand::RngCore;
    let mut nonce = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    nonce
}

/// Identity of a request sender.
///
/// DEPRECATED: This type is retained for downstream compilation compatibility
/// only. It is no longer serialized to or deserialized from Cap'n Proto wire
/// format. Authorization decisions use `EnvelopeContext::subject()`, derived
/// from the verified Ed25519 signer key via `KeyRegistry`.
///
/// TODO: remove in Phase 2
#[deprecated(note = "Use Authorization instead — this type is no longer on the wire")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestIdentity {
    /// Authenticated via API token.
    /// User is from the token record.
    ApiToken { user: String, token_name: String },

    /// Authenticated remote peer via CURVE.
    /// User is the peer's registered name.
    Peer {
        name: String,
        curve_key: [u8; 32],
    },

    /// No authentication provided.
    /// User is "anonymous".
    Anonymous,
}

#[allow(deprecated, clippy::derivable_impls)]
impl Default for RequestIdentity {
    fn default() -> Self {
        Self::Anonymous
    }
}

#[allow(deprecated)]
impl RequestIdentity {
    /// Create an API token identity.
    pub fn api_token(user: impl Into<String>, token_name: impl Into<String>) -> Self {
        Self::ApiToken {
            user: user.into(),
            token_name: token_name.into(),
        }
    }

    /// Create a peer identity.
    pub fn peer(name: impl Into<String>, curve_key: [u8; 32]) -> Self {
        Self::Peer {
            name: name.into(),
            curve_key,
        }
    }

    /// Create an anonymous identity.
    pub fn anonymous() -> Self {
        Self::Anonymous
    }

    /// Extract the bare username string.
    ///
    /// This is the canonical form used for logging only.
    /// Authorization decisions must use `EnvelopeContext::subject()`.
    pub fn user(&self) -> &str {
        match self {
            Self::ApiToken { user, .. } => user,
            Self::Peer { name, .. } => name,
            Self::Anonymous => "anonymous",
        }
    }

    /// Check if this is authenticated (not anonymous).
    pub fn is_authenticated(&self) -> bool {
        !matches!(self, Self::Anonymous)
    }
}

#[allow(deprecated)]
impl std::fmt::Display for RequestIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiToken { user, token_name } => write!(f, "token:{user}:{token_name}"),
            Self::Peer { name, .. } => write!(f, "peer:{name}"),
            Self::Anonymous => write!(f, "anonymous"),
        }
    }
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
        self.0.as_deref().map(|s| s.contains("://")).unwrap_or(false)
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

#[allow(deprecated)]
impl From<&RequestIdentity> for Subject {
    fn from(id: &RequestIdentity) -> Self {
        match id {
            RequestIdentity::Anonymous => Subject::anonymous(),
            other => Subject::new(other.user()),
        }
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

// NOTE: ToCapnp/FromCapnp impls for RequestIdentity have been removed.
// The capnp types (RequestIdentity, LocalIdentity, ApiTokenIdentity, PeerIdentity)
// no longer exist in the schema. The Rust enum is kept temporarily for downstream compat.
// TODO: remove in Phase 2

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
    pub delegation_token: Option<String>,

    /// SHA-256 hash of the WIT JWT string (WIMSE wth claim).
    /// Binds this proof to a specific Workload Identity Token even when
    /// the JWT is omitted (trust-store cache-hit path).
    pub wth: Option<[u8; 32]>,
}

impl RequestEnvelope {
    /// Create a new request envelope with fresh request ID, nonce, and timestamp.
    #[allow(deprecated)]
    pub fn new(payload: Vec<u8>) -> Self {
        Self {
            request_id: next_request_id(),
            payload,
            iat: current_timestamp(),
            nonce: generate_nonce(),
            authorization: Authorization::None,
            delegation_token: None,
            wth: None,
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

    /// Create an envelope for an anonymous request.
    pub fn anonymous(payload: Vec<u8>) -> Self {
        Self::new(payload)
    }

    // --- Temporary compatibility shims (TODO: remove in Phase 2) ---

    /// Compatibility shim: create with legacy identity parameter (ignored).
    #[allow(deprecated)]
    pub fn new_compat(_identity: RequestIdentity, payload: Vec<u8>) -> Self {
        Self::new(payload)
    }

    /// Compatibility shim for `with_delegated_bearer`.
    #[deprecated(note = "Use with_delegation_token instead")]
    pub fn with_delegated_bearer(self, bearer: String) -> Self {
        self.with_delegation_token(bearer)
    }

    /// Compatibility shim for `with_wit_hash_of`.
    #[deprecated(note = "Use with_wth_of instead")]
    pub fn with_wit_hash_of(self, jwt: &str) -> Self {
        self.with_wth_of(jwt)
    }

    /// Compatibility shim: set legacy claims as authorization context.
    #[deprecated(note = "Use with_authorization instead")]
    pub fn with_claims(self, _claims: Claims) -> Self {
        // Claims are no longer carried in the envelope — this is a no-op shim.
        self
    }

    /// Compatibility shim: set ephemeral pubkey (no longer in envelope).
    #[deprecated(note = "Ephemeral pubkey is no longer in the envelope")]
    pub fn with_ephemeral_pubkey(self, _pubkey: [u8; 32]) -> Self {
        // Ephemeral pubkey has been removed from the envelope.
        self
    }

    /// Compatibility shim: create with token identity.
    #[allow(deprecated)]
    #[deprecated(note = "Use new() instead — identity is no longer in the envelope")]
    pub fn with_token(_user: impl Into<String>, _token_name: impl Into<String>, payload: Vec<u8>) -> Self {
        Self::new(payload)
    }

    /// Compatibility shim: create with peer identity.
    #[allow(deprecated)]
    #[deprecated(note = "Use new() instead — identity is no longer in the envelope")]
    pub fn with_peer(_name: impl Into<String>, _curve_key: [u8; 32], payload: Vec<u8>) -> Self {
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
        use capnp::serialize;
        use capnp::Word;

        // Step 1: Build the message
        let mut message = Builder::new_default();
        {
            let mut builder = message.init_root::<common_capnp::request_envelope::Builder>();
            self.write_to(&mut builder);
        }

        // Step 2: Serialize to bytes first (to create a reader)
        let mut temp_bytes = Vec::new();
        if let Err(_e) = serialize::write_message(&mut temp_bytes, &message) {
            #[cfg(not(target_arch = "wasm32"))]
            tracing::error!("RequestEnvelope temporary serialization failed: {}", _e);
            return Vec::new();
        }

        // Step 3: Read back to get a Reader
        let reader = match serialize::read_message(
            &mut std::io::Cursor::new(&temp_bytes),
            envelope_reader_options(),
        ) {
            Ok(r) => r,
            Err(_e) => {
                #[cfg(not(target_arch = "wasm32"))]
                tracing::error!("RequestEnvelope reader creation failed: {}", _e);
                return Vec::new();
            }
        };

        // Step 4: CRITICAL - Canonicalize before signing
        // This ensures deterministic serialization as required by Cap'n Proto spec.
        // Non-canonical serialization can produce different bytes for identical data,
        // breaking signature verification across platforms/versions.
        let canonical_words = match reader.canonicalize() {
            Ok(words) => words,
            Err(_e) => {
                #[cfg(not(target_arch = "wasm32"))]
                tracing::error!("Envelope canonicalization failed: {}", _e);
                // Fall back to temp_bytes if canonicalization fails
                return temp_bytes;
            }
        };

        // Step 5: Convert Words to bytes (raw segment data, NO stream framing)
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
    /// The unsigned envelope (this is what gets signed)
    pub envelope: RequestEnvelope,

    /// Ed25519 signature over serialized envelope (64 bytes)
    pub sig: [u8; 64],

    /// Ed25519 public key of the signer (32 bytes)
    pub cnf: [u8; 32],
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
    fn read_lock(&self) -> parking_lot::RwLockReadGuard<'_, std::collections::HashMap<[u8; 16], i64>> {
        self.seen.read()
    }

    #[cfg(target_arch = "wasm32")]
    fn read_lock(&self) -> std::sync::RwLockReadGuard<'_, std::collections::HashMap<[u8; 16], i64>> {
        self.seen.read().expect("nonce cache lock poisoned")
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_lock(&self) -> parking_lot::RwLockWriteGuard<'_, std::collections::HashMap<[u8; 16], i64>> {
        self.seen.write()
    }

    #[cfg(target_arch = "wasm32")]
    fn write_lock(&self) -> std::sync::RwLockWriteGuard<'_, std::collections::HashMap<[u8; 16], i64>> {
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
    pub fn new_signed(mut envelope: RequestEnvelope, signing_key: &SigningKey) -> Self {
        // Auto-populate wth from IdJag JWT when present and not already set
        if envelope.wth.is_none() {
            if let Some(jwt) = envelope.jwt_token() {
                use sha2::{Digest, Sha256};
                envelope.wth = Some(Sha256::digest(jwt.as_bytes()).into());
            }
        }

        // Serialize the envelope to get canonical bytes
        let envelope_bytes = envelope.to_bytes();

        // Sign the serialized envelope
        let signature = signing_key.sign(&envelope_bytes);

        Self {
            envelope,
            sig: signature.to_bytes(),
            cnf: signing_key.verifying_key().to_bytes(),
        }
    }

    /// Verify the signature and check replay protection.
    ///
    /// # Verification Steps
    ///
    /// 1. Verify signer pubkey matches expected key
    /// 2. Check timestamp is within acceptable window
    /// 3. Check nonce hasn't been seen before
    /// 4. Verify Ed25519 signature
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
        // 1. Verify signer matches expected (constant-time comparison)
        if !bool::from(self.cnf.ct_eq(&expected_pubkey.to_bytes())) {
            return Err(EnvelopeError::SignerMismatch {
                expected: hex::encode(expected_pubkey.to_bytes()),
                actual: hex::encode(self.cnf),
            });
        }

        // 2. Check timestamp window
        let now = current_timestamp();

        let age = now - self.envelope.iat;
        if age > MAX_TIMESTAMP_AGE_MS {
            return Err(EnvelopeError::ReplayAttack(format!(
                "timestamp too old: {age}ms > {MAX_TIMESTAMP_AGE_MS}ms"
            )));
        }
        if age < -MAX_CLOCK_SKEW_MS {
            return Err(EnvelopeError::ReplayAttack(format!(
                "timestamp in future: {}ms beyond clock skew tolerance",
                -age - MAX_CLOCK_SKEW_MS
            )));
        }

        // 3. Check nonce not seen before
        if !nonce_cache.check_and_insert(&self.envelope.nonce) {
            return Err(EnvelopeError::ReplayAttack(
                "nonce already seen".to_owned(),
            ));
        }

        // 4. Verify signature (strict: rejects small-order public keys)
        let envelope_bytes = self.envelope.to_bytes();
        let signature = Signature::from_bytes(&self.sig);
        expected_pubkey.verify_strict(&envelope_bytes, &signature)?;

        Ok(())
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

        let envelope_bytes = self.envelope.to_bytes();
        let signature = Signature::from_bytes(&self.sig);
        expected_pubkey.verify_strict(&envelope_bytes, &signature)?;

        Ok(())
    }

    /// Verify against the envelope's own embedded signer pubkey.
    ///
    /// For WebTransport clients that sign with their own keypair rather than
    /// a shared server key. Still checks timestamp and nonce for replay protection.
    pub fn verify_any_signer(
        &self,
        nonce_cache: &dyn NonceCache,
    ) -> EnvelopeResult<()> {
        // 1. Reconstruct the verifying key from the embedded pubkey
        let verifying_key = VerifyingKey::from_bytes(&self.cnf)
            .map_err(|_| EnvelopeError::InvalidPublicKey { expected: 32, actual: 0 })?;

        // 2. Check timestamp window
        let now = current_timestamp();
        let age = now - self.envelope.iat;
        if age > MAX_TIMESTAMP_AGE_MS {
            return Err(EnvelopeError::ReplayAttack(format!(
                "timestamp too old: {age}ms > {MAX_TIMESTAMP_AGE_MS}ms"
            )));
        }
        if age < -MAX_CLOCK_SKEW_MS {
            return Err(EnvelopeError::ReplayAttack(format!(
                "timestamp in future: {}ms beyond clock skew tolerance",
                -age - MAX_CLOCK_SKEW_MS
            )));
        }

        // 3. Check nonce not seen before
        if !nonce_cache.check_and_insert(&self.envelope.nonce) {
            return Err(EnvelopeError::ReplayAttack(
                "nonce already seen".to_owned(),
            ));
        }

        // 4. Verify signature against the embedded public key
        let envelope_bytes = self.envelope.to_bytes();
        let signature = Signature::from_bytes(&self.sig);
        verifying_key.verify_strict(&envelope_bytes, &signature)?;

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

    // --- Temporary compatibility shims (TODO: remove in Phase 2) ---

    /// Compatibility shim: get the authorization subject (always anonymous from envelope).
    pub fn subject(&self) -> Subject {
        Subject::anonymous()
    }

    /// Compatibility shim: ephemeral pubkey is no longer in the envelope.
    #[deprecated(note = "Ephemeral pubkey is no longer in the envelope")]
    pub fn ephemeral_pubkey(&self) -> Option<&[u8; 32]> {
        None
    }

    /// Compatibility shim: access signer_pubkey via the old name.
    #[deprecated(note = "Use .cnf instead")]
    pub fn signer_pubkey(&self) -> [u8; 32] {
        self.cnf
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
                if t.is_empty() { None } else { Some(t.to_owned()) }
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

        Ok(Self {
            request_id: reader.get_request_id(),
            payload: reader.get_payload()?.to_vec(),
            iat: reader.get_iat(),
            nonce,
            authorization,
            delegation_token,
            wth,
        })
    }
}

impl ToCapnp for SignedEnvelope {
    type Builder<'a> = common_capnp::signed_envelope::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        self.envelope
            .write_to(&mut builder.reborrow().init_envelope());
        builder.set_sig(&self.sig);
        builder.set_cnf(&self.cnf);
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

        Ok(Self {
            envelope: RequestEnvelope::read_from(reader.get_envelope()?)?,
            sig,
            cnf,
        })
    }
}

/// Signed response envelope for E2E authenticated responses.
///
/// All RPC responses are signed to prevent MITM attacks on response data
/// (e.g., server's DH public key in StreamInfo).
///
/// # Security
///
/// The signature covers `request_id || payload`, binding the response to
/// a specific request and ensuring the payload hasn't been tampered with.
#[derive(Debug, Clone)]
pub struct ResponseEnvelope {
    /// Request ID this response corresponds to
    pub request_id: u64,

    /// Serialized inner response
    pub payload: Vec<u8>,

    /// Ed25519 signature (64 bytes) over request_id || payload
    pub sig: [u8; 64],

    /// Ed25519 public key of the signer (32 bytes)
    pub cnf: [u8; 32],
}

impl ResponseEnvelope {
    /// Create and sign a new response envelope.
    ///
    /// The signature covers `request_id || payload` to bind the response
    /// to the specific request and prevent tampering.
    pub fn new_signed(request_id: u64, payload: Vec<u8>, signing_key: &SigningKey) -> Self {
        // Build signing data: request_id (8 bytes LE) || payload
        let mut signing_data = Vec::with_capacity(8 + payload.len());
        signing_data.extend_from_slice(&request_id.to_le_bytes());
        signing_data.extend_from_slice(&payload);

        let signature_obj = signing_key.sign(&signing_data);
        let sig: [u8; 64] = signature_obj.to_bytes();
        let cnf: [u8; 32] = signing_key.verifying_key().to_bytes();

        Self {
            request_id,
            payload,
            sig,
            cnf,
        }
    }

    /// Verify the response signature.
    pub fn verify(&self, expected_pubkey: Option<&VerifyingKey>) -> Result<()> {
        let verifying_key = VerifyingKey::from_bytes(&self.cnf)
            .map_err(|_| anyhow::anyhow!("Invalid signer public key"))?;

        if let Some(expected) = expected_pubkey {
            if !bool::from(verifying_key.to_bytes().ct_eq(&expected.to_bytes())) {
                anyhow::bail!("Response signed by unexpected key");
            }
        }

        let mut signing_data = Vec::with_capacity(8 + self.payload.len());
        signing_data.extend_from_slice(&self.request_id.to_le_bytes());
        signing_data.extend_from_slice(&self.payload);

        let signature = ed25519_dalek::Signature::from_bytes(&self.sig);
        verifying_key
            .verify_strict(&signing_data, &signature)
            .map_err(|_| anyhow::anyhow!("Response signature verification failed"))
    }

    /// Get the signer's public key.
    pub fn cnf_key(&self) -> Result<VerifyingKey> {
        VerifyingKey::from_bytes(&self.cnf)
            .map_err(|_| anyhow::anyhow!("Invalid signer public key"))
    }

    /// Compatibility shim for old name.
    #[deprecated(note = "Use cnf_key() instead")]
    pub fn signer_pubkey(&self) -> Result<VerifyingKey> {
        self.cnf_key()
    }
}

impl ToCapnp for ResponseEnvelope {
    type Builder<'a> = common_capnp::response_envelope::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_request_id(self.request_id);
        builder.set_payload(&self.payload);
        builder.set_sig(&self.sig);
        builder.set_cnf(&self.cnf);
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

        Ok(Self {
            request_id: reader.get_request_id(),
            payload: reader.get_payload()?.to_vec(),
            sig,
            cnf,
        })
    }
}

/// Unwrap and verify a SignedEnvelope from wire bytes (core logic, all targets).
///
/// Deserializes, verifies signature and replay protection, then extracts
/// the verified envelope and payload.
pub fn unwrap_and_verify(
    request: &[u8],
    server_pubkey: &VerifyingKey,
    nonce_cache: &dyn NonceCache,
) -> Result<(SignedEnvelope, Vec<u8>)> {
    use capnp::serialize;

    let reader = serialize::read_message(
        &mut std::io::Cursor::new(request),
        envelope_reader_options(),
    )?;
    let signed_reader = reader.get_root::<crate::common_capnp::signed_envelope::Reader>()?;
    let signed = SignedEnvelope::read_from(signed_reader)?;

    signed.verify(server_pubkey, nonce_cache)?;

    let payload = signed.payload().to_vec();
    Ok((signed, payload))
}

/// Unwrap and verify a FixedSigner envelope, returning `EnvelopeContext` + payload.
///
/// Sets `key_derived_subject = Subject::new("system")` on the context, so
/// `EnvelopeContext::subject()` returns `"system"` for any caller that passes
/// signature verification against `server_pubkey`. Use this for inproc/IPC
/// callers that share the node's signing key.
///
/// Native only — `EnvelopeContext` requires the `service` module.
#[cfg(not(target_arch = "wasm32"))]
pub fn unwrap_envelope(
    request: &[u8],
    server_pubkey: &VerifyingKey,
    nonce_cache: &dyn NonceCache,
) -> Result<(crate::service::EnvelopeContext, Vec<u8>)> {
    let (signed, payload) = unwrap_and_verify(request, server_pubkey, nonce_cache)?;
    let ctx = crate::service::EnvelopeContext::from_verified_as_system(&signed);
    Ok((ctx, payload))
}

/// Unwrap and verify a SignedEnvelope from any signer (WebTransport clients).
///
/// Unlike `unwrap_and_verify`, this does NOT require the signer to match a
/// specific server pubkey. Instead, it verifies the signature against the
/// envelope's own embedded public key. Used for browser clients that sign
/// with their own keypair.
pub fn unwrap_and_verify_any_signer(
    request: &[u8],
    nonce_cache: &dyn NonceCache,
) -> Result<(SignedEnvelope, Vec<u8>)> {
    use capnp::serialize;

    let reader = serialize::read_message(
        &mut std::io::Cursor::new(request),
        envelope_reader_options(),
    )?;
    let signed_reader = reader.get_root::<crate::common_capnp::signed_envelope::Reader>()?;
    let signed = SignedEnvelope::read_from(signed_reader)?;

    signed.verify_any_signer(nonce_cache)?;

    let payload = signed.payload().to_vec();
    Ok((signed, payload))
}

/// Unwrap and verify a SignedEnvelope from any signer, returning `EnvelopeContext` + payload.
///
/// WebTransport variant of `unwrap_envelope` — accepts any valid Ed25519 signer.
#[cfg(not(target_arch = "wasm32"))]
pub fn unwrap_envelope_any_signer(
    request: &[u8],
    nonce_cache: &dyn NonceCache,
) -> Result<(crate::service::EnvelopeContext, Vec<u8>)> {
    let (signed, payload) = unwrap_and_verify_any_signer(request, nonce_cache)?;
    let ctx = crate::service::EnvelopeContext::from_verified(&signed);
    Ok((ctx, payload))
}

/// Unwrap and verify a FixedSigner envelope, setting `key_derived_subject = system`.
///
/// Used for inproc/IPC callers who sign with the node key. Sets
/// `Subject::new("system")` as the key-derived subject so that
/// `EnvelopeContext::subject()` returns `"system"` regardless of any
/// caller-asserted `RequestIdentity` field in the envelope.
#[cfg(not(target_arch = "wasm32"))]
pub fn unwrap_envelope_as_system(
    request: &[u8],
    server_pubkey: &VerifyingKey,
    nonce_cache: &dyn NonceCache,
) -> Result<(crate::service::EnvelopeContext, Vec<u8>)> {
    let (signed, payload) = unwrap_and_verify(request, server_pubkey, nonce_cache)?;
    let ctx = crate::service::EnvelopeContext::from_verified_as_system(&signed);
    Ok((ctx, payload))
}

/// Unwrap and verify a ResponseEnvelope from wire bytes.
///
/// Deserializes and verifies signature, then extracts the payload.
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
    use capnp::serialize;

    // Deserialize ResponseEnvelope from Cap'n Proto
    let reader = serialize::read_message(
        &mut std::io::Cursor::new(response),
        envelope_reader_options(),
    )?;
    let response_reader = reader.get_root::<crate::common_capnp::response_envelope::Reader>()?;
    let envelope = ResponseEnvelope::read_from(response_reader)?;

    // Verify signature
    envelope.verify(expected_pubkey)?;

    Ok((envelope.request_id, envelope.payload))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::crypto::signing::generate_signing_keypair;
    use parking_lot::Mutex;
    use std::collections::HashSet;

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

    #[test]
    fn test_api_token_identity() {
        let identity = RequestIdentity::api_token("bob", "ci-token");
        assert_eq!(identity.user(), "bob");
        assert!(identity.is_authenticated());
    }

    #[test]
    fn test_anonymous_identity() {
        let identity = RequestIdentity::anonymous();
        assert_eq!(identity.user(), "anonymous");
        assert!(!identity.is_authenticated());
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
    fn test_identity_display() {
        assert_eq!(
            format!("{}", RequestIdentity::api_token("bob", "ci")),
            "token:bob:ci"
        );
        assert_eq!(format!("{}", RequestIdentity::anonymous()), "anonymous");
    }

    #[test]
    fn test_signed_envelope_sign_verify() -> crate::EnvelopeResult<()> {
        let (signing_key, verifying_key) = generate_signing_keypair();
        let nonce_cache = TestNonceCache::new();

        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3, 4]);
        let signed = SignedEnvelope::new_signed(envelope, &signing_key);

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
        let signed = SignedEnvelope::new_signed(envelope, &signing_key);

        // Verify with wrong key should fail
        let result = signed.verify(&wrong_verifying_key, &nonce_cache);
        assert!(matches!(result, Err(EnvelopeError::SignerMismatch { .. })));
    }

    #[test]
    fn test_signed_envelope_replay_fails() -> crate::EnvelopeResult<()> {
        let (signing_key, verifying_key) = generate_signing_keypair();
        let nonce_cache = TestNonceCache::new();

        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3, 4]);
        let signed = SignedEnvelope::new_signed(envelope, &signing_key);

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
        let signed = SignedEnvelope::new_signed(envelope, &signing_key);

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
        Ok(())
    }

    #[test]
    fn test_capnp_roundtrip_signed_envelope() -> Result<(), Box<dyn std::error::Error>> {
        use capnp::message::Builder;

        let (signing_key, verifying_key) = generate_signing_keypair();
        let envelope = RequestEnvelope::anonymous(vec![1, 2, 3]);
        let signed = SignedEnvelope::new_signed(envelope, &signing_key);

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
    fn test_subject_from_identity() {
        let token = RequestIdentity::ApiToken { user: "bob".into(), token_name: "ci".into() };
        assert_eq!(Subject::from(&token), Subject::new("bob"));

        let peer = RequestIdentity::Peer { name: "gpu-1".into(), curve_key: [0u8; 32] };
        assert_eq!(Subject::from(&peer), Subject::new("gpu-1"));

        assert_eq!(Subject::from(&RequestIdentity::Anonymous), Subject::anonymous());
    }

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
        assert_eq!("local:alice".parse::<Subject>().expect("parse local:alice"), Subject::new("alice"));
        assert_eq!("token:bob".parse::<Subject>().expect("parse token:bob"), Subject::new("bob"));
        assert_eq!("peer:gpu-1".parse::<Subject>().expect("parse peer:gpu-1"), Subject::new("gpu-1"));
        assert_eq!("user:charlie".parse::<Subject>().expect("parse user:charlie"), Subject::new("charlie"));
        assert_eq!("alice".parse::<Subject>().expect("parse alice"), Subject::new("alice"));
        assert_eq!("unknown:foo".parse::<Subject>().expect("parse unknown:foo"), Subject::new("unknown:foo"));
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

        let from_token = Subject::from(&RequestIdentity::ApiToken { user: "alice".into(), token_name: "ci".into() });
        let from_peer = Subject::from(&RequestIdentity::Peer { name: "alice".into(), curve_key: [0u8; 32] });
        assert_eq!(from_token, from_peer);
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
        let claims = crate::auth::Claims::new(
            "charlie".to_owned(),
            1000,
            2000,
        );
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

        Ok(())
    }
}
