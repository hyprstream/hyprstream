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
//! # Identity → Casbin Subject Mapping
//!
//! Each identity type extracts a **namespaced** subject for Casbin policy checks:
//!
//! | Identity | Casbin Subject | Example |
//! |----------|----------------|---------|
//! | Local | `local:<username>` | `"local:alice"` |
//! | ApiToken | `token:<username>` | `"token:bob"` |
//! | Peer | `peer:<name>` | `"peer:gpu-server-1"` |
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
use ed25519_dalek::{Signature, Signer, Verifier};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Global request ID counter for unique IDs
static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique request ID
pub fn next_request_id() -> u64 {
    REQUEST_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Identity of a request sender.
///
/// This enum represents the different ways a client can authenticate.
/// The identity determines the "user" for Casbin policy checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestIdentity {
    /// Local process on the same machine.
    /// User is the OS username (trusted).
    Local { user: String },

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

impl RequestIdentity {
    /// Create a local identity using the current OS username.
    pub fn local() -> Self {
        Self::Local {
            user: whoami::username(),
        }
    }

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

    /// Extract the raw user string (without namespace prefix).
    ///
    /// For Casbin policy checks, use `casbin_subject()` instead which
    /// includes the namespace prefix to prevent collisions.
    pub fn user(&self) -> &str {
        match self {
            // Both Local and ApiToken have a user field
            Self::Local { user } | Self::ApiToken { user, .. } => user,
            Self::Peer { name, .. } => name,
            Self::Anonymous => "anonymous",
        }
    }

    /// Get the namespaced Casbin subject for policy checks.
    ///
    /// Subjects are prefixed to prevent collisions between different
    /// identity types. A peer named "admin" must NOT match policies
    /// for local user "admin".
    ///
    /// # Examples
    ///
    /// ```
    /// use hyprstream_rpc::envelope::RequestIdentity;
    ///
    /// let local = RequestIdentity::Local { user: "alice".into() };
    /// assert_eq!(local.casbin_subject(), "local:alice");
    ///
    /// let token = RequestIdentity::ApiToken {
    ///     user: "bob".into(),
    ///     token_name: "ci".into()
    /// };
    /// assert_eq!(token.casbin_subject(), "token:bob");
    /// ```
    pub fn casbin_subject(&self) -> String {
        match self {
            Self::Local { user } => format!("local:{user}"),
            Self::ApiToken { user, .. } => format!("token:{user}"),
            Self::Peer { name, .. } => format!("peer:{name}"),
            Self::Anonymous => "anonymous".to_owned(),
        }
    }

    /// Check if this is a local (trusted) identity.
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local { .. })
    }

    /// Check if this is authenticated (not anonymous).
    pub fn is_authenticated(&self) -> bool {
        !matches!(self, Self::Anonymous)
    }
}

impl Default for RequestIdentity {
    fn default() -> Self {
        Self::local()
    }
}

impl std::fmt::Display for RequestIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local { user } => write!(f, "local:{user}"),
            Self::ApiToken { user, token_name } => write!(f, "token:{user}:{token_name}"),
            Self::Peer { name, .. } => write!(f, "peer:{name}"),
            Self::Anonymous => write!(f, "anonymous"),
        }
    }
}

// Manual Cap'n Proto implementation for RequestIdentity (union type)
impl ToCapnp for RequestIdentity {
    type Builder<'a> = common_capnp::request_identity::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        match self {
            Self::Local { user } => {
                let mut local = builder.reborrow().init_local();
                local.set_user(user);
            }
            Self::ApiToken { user, token_name } => {
                let mut api_token = builder.reborrow().init_api_token();
                api_token.set_user(user);
                api_token.set_token_name(token_name);
            }
            Self::Peer { name, curve_key } => {
                let mut peer = builder.reborrow().init_peer();
                peer.set_name(name);
                peer.set_curve_key(curve_key);
            }
            Self::Anonymous => {
                builder.set_anonymous(());
            }
        }
    }
}

impl FromCapnp for RequestIdentity {
    type Reader<'a> = common_capnp::request_identity::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        use common_capnp::request_identity::Which;

        match reader.which()? {
            Which::Local(local) => {
                let local = local?;
                Ok(Self::Local {
                    user: local.get_user()?.to_str()?.to_owned(),
                })
            }
            Which::ApiToken(api_token) => {
                let api_token = api_token?;
                Ok(Self::ApiToken {
                    user: api_token.get_user()?.to_str()?.to_owned(),
                    token_name: api_token.get_token_name()?.to_str()?.to_owned(),
                })
            }
            Which::Peer(peer) => {
                let peer = peer?;
                let key_data = peer.get_curve_key()?;
                if key_data.len() != 32 {
                    return Err(anyhow!(
                        "Invalid CURVE key length: expected 32, got {}",
                        key_data.len()
                    ));
                }
                let mut curve_key = [0u8; 32];
                curve_key.copy_from_slice(key_data);
                Ok(Self::Peer {
                    name: peer.get_name()?.to_str()?.to_owned(),
                    curve_key,
                })
            }
            Which::Anonymous(()) => Ok(Self::Anonymous),
        }
    }
}

/// Unsigned envelope wrapping an RPC request with identity context.
///
/// This struct contains all request metadata and is signed by `SignedEnvelope`.
/// The entire serialized RequestEnvelope is covered by the signature.
///
/// # Replay Protection
///
/// - `nonce`: 16 random bytes, must be unique per request
/// - `timestamp`: Unix milliseconds, requests older than 5 minutes are rejected
#[derive(Debug, Clone)]
pub struct RequestEnvelope {
    /// Unique request ID for correlation and logging
    pub request_id: u64,

    /// Identity of the requester
    pub identity: RequestIdentity,

    /// Serialized inner request (e.g., RegistryRequest, InferenceRequest)
    pub payload: Vec<u8>,

    /// Ristretto255/P-256 ephemeral public key for stream HMAC key derivation (optional)
    pub ephemeral_pubkey: Option<[u8; 32]>,

    /// Random nonce for replay protection (16 bytes)
    pub nonce: [u8; 16],

    /// Unix timestamp in milliseconds for expiration check
    pub timestamp: i64,

    /// User authorization claims (protected by envelope signature)
    /// Contains the user's identity, scopes, and expiration.
    /// No separate JWT signature needed - the envelope signature covers this.
    pub claims: Option<Claims>,
}

impl RequestEnvelope {
    /// Create a new request envelope with fresh request ID, nonce, and timestamp.
    pub fn new(identity: RequestIdentity, payload: Vec<u8>) -> Self {
        let mut nonce = [0u8; 16];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce);

        // SAFETY: Only fails if system time is before Unix epoch (1970)
        // Cap at i64::MAX (won't overflow for ~292 million years)
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0);

        Self {
            request_id: next_request_id(),
            identity,
            payload,
            ephemeral_pubkey: None,
            nonce,
            timestamp,
            claims: None,
        }
    }

    /// Set ephemeral public key for streaming requests (DH key exchange).
    pub fn with_ephemeral_pubkey(mut self, pubkey: [u8; 32]) -> Self {
        self.ephemeral_pubkey = Some(pubkey);
        self
    }

    /// Set user authorization claims.
    pub fn with_claims(mut self, claims: Claims) -> Self {
        self.claims = Some(claims);
        self
    }

    /// Create an envelope for a local request.
    pub fn local(payload: Vec<u8>) -> Self {
        Self::new(RequestIdentity::local(), payload)
    }

    /// Create an envelope for an API token authenticated request.
    pub fn with_token(user: impl Into<String>, token_name: impl Into<String>, payload: Vec<u8>) -> Self {
        Self::new(RequestIdentity::api_token(user, token_name), payload)
    }

    /// Create an envelope for a peer authenticated request.
    pub fn with_peer(name: impl Into<String>, curve_key: [u8; 32], payload: Vec<u8>) -> Self {
        Self::new(RequestIdentity::peer(name, curve_key), payload)
    }

    /// Create an envelope for an anonymous request.
    pub fn anonymous(payload: Vec<u8>) -> Self {
        Self::new(RequestIdentity::anonymous(), payload)
    }

    /// Get the raw user string (without namespace prefix).
    pub fn user(&self) -> &str {
        self.identity.user()
    }

    /// Get the namespaced Casbin subject for policy checks.
    pub fn casbin_subject(&self) -> String {
        self.identity.casbin_subject()
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
        if let Err(e) = serialize::write_message(&mut temp_bytes, &message) {
            tracing::error!("RequestEnvelope temporary serialization failed: {}", e);
            return Vec::new();
        }

        // Step 3: Read back to get a Reader
        let reader = match serialize::read_message(
            &mut std::io::Cursor::new(&temp_bytes),
            capnp::message::ReaderOptions::default(),
        ) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("RequestEnvelope reader creation failed: {}", e);
                return Vec::new();
            }
        };

        // Step 4: CRITICAL - Canonicalize before signing
        // This ensures deterministic serialization as required by Cap'n Proto spec.
        // Non-canonical serialization can produce different bytes for identical data,
        // breaking signature verification across platforms/versions.
        let canonical_words = match reader.canonicalize() {
            Ok(words) => words,
            Err(e) => {
                tracing::error!("Envelope canonicalization failed: {}", e);
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
/// let envelope = RequestEnvelope::local(payload);
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
    pub signature: [u8; 64],

    /// Ed25519 public key of the signer (32 bytes)
    pub signer_pubkey: [u8; 32],
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
    seen: parking_lot::RwLock<std::collections::HashMap<[u8; 16], i64>>,
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
            seen: parking_lot::RwLock::new(std::collections::HashMap::new()),
            max_age_ms: MAX_TIMESTAMP_AGE_MS,
            max_entries: 100_000,
        }
    }

    /// Create a new cache with custom settings.
    pub fn with_config(max_age_ms: i64, max_entries: usize) -> Self {
        Self {
            seen: parking_lot::RwLock::new(std::collections::HashMap::new()),
            max_age_ms,
            max_entries,
        }
    }

    /// Remove expired entries.
    fn cleanup(&self, now: i64) {
        let mut seen = self.seen.write();

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
        // SAFETY: Only fails if system time is before Unix epoch (1970)
        // Cap at i64::MAX (won't overflow for ~292 million years)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0);

        // Fast path: check if already seen (read lock)
        {
            let seen = self.seen.read();
            if seen.contains_key(nonce) {
                return false;
            }
        }

        // Slow path: insert (write lock)
        let mut seen = self.seen.write();

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
    pub fn new_signed(envelope: RequestEnvelope, signing_key: &SigningKey) -> Self {
        // Serialize the envelope to get canonical bytes
        let envelope_bytes = envelope.to_bytes();

        // Sign the serialized envelope
        let signature = signing_key.sign(&envelope_bytes);

        Self {
            envelope,
            signature: signature.to_bytes(),
            signer_pubkey: signing_key.verifying_key().to_bytes(),
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
        // 1. Verify signer matches expected
        if self.signer_pubkey != expected_pubkey.to_bytes() {
            return Err(EnvelopeError::SignerMismatch {
                expected: hex::encode(expected_pubkey.to_bytes()),
                actual: hex::encode(self.signer_pubkey),
            });
        }

        // 2. Check timestamp window
        // SAFETY: Only fails if system time is before Unix epoch (1970)
        // Cap at i64::MAX (won't overflow for ~292 million years)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0);

        let age = now - self.envelope.timestamp;
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

        // 4. Verify signature
        let envelope_bytes = self.envelope.to_bytes();
        let signature = Signature::from_bytes(&self.signature);
        expected_pubkey.verify(&envelope_bytes, &signature)?;

        Ok(())
    }

    /// Verify signature only (skip replay protection).
    ///
    /// Use this for testing or when replay protection is handled elsewhere.
    pub fn verify_signature_only(&self, expected_pubkey: &VerifyingKey) -> EnvelopeResult<()> {
        if self.signer_pubkey != expected_pubkey.to_bytes() {
            return Err(EnvelopeError::SignerMismatch {
                expected: hex::encode(expected_pubkey.to_bytes()),
                actual: hex::encode(self.signer_pubkey),
            });
        }

        let envelope_bytes = self.envelope.to_bytes();
        let signature = Signature::from_bytes(&self.signature);
        expected_pubkey.verify(&envelope_bytes, &signature)?;

        Ok(())
    }

    /// Get the request ID from the inner envelope.
    pub fn request_id(&self) -> u64 {
        self.envelope.request_id
    }

    /// Get the identity from the inner envelope.
    pub fn identity(&self) -> &RequestIdentity {
        &self.envelope.identity
    }

    /// Get the Casbin subject for policy checks.
    pub fn casbin_subject(&self) -> String {
        self.envelope.casbin_subject()
    }

    /// Get the payload from the inner envelope.
    pub fn payload(&self) -> &[u8] {
        &self.envelope.payload
    }

    /// Get the ephemeral pubkey for stream HMAC derivation.
    pub fn ephemeral_pubkey(&self) -> Option<&[u8; 32]> {
        self.envelope.ephemeral_pubkey.as_ref()
    }
}

impl ToCapnp for RequestEnvelope {
    type Builder<'a> = common_capnp::request_envelope::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_request_id(self.request_id);
        self.identity
            .write_to(&mut builder.reborrow().init_identity());
        builder.set_payload(&self.payload);
        if let Some(ref pubkey) = self.ephemeral_pubkey {
            builder.set_ephemeral_pubkey(pubkey);
        }
        builder.set_nonce(&self.nonce);
        builder.set_timestamp(self.timestamp);
        if let Some(ref claims) = self.claims {
            claims.write_to(&mut builder.reborrow().init_claims());
        }
    }
}

impl FromCapnp for RequestEnvelope {
    type Reader<'a> = common_capnp::request_envelope::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let ephemeral_pubkey = {
            let data = reader.get_ephemeral_pubkey()?;
            if data.is_empty() {
                None
            } else if data.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(data);
                Some(arr)
            } else {
                return Err(anyhow!(
                    "Invalid ephemeral pubkey length: expected 32, got {}",
                    data.len()
                ));
            }
        };

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

        let claims = {
            if reader.has_claims() {
                Some(Claims::read_from(reader.get_claims()?)?)
            } else {
                None
            }
        };

        Ok(Self {
            request_id: reader.get_request_id(),
            identity: RequestIdentity::read_from(reader.get_identity()?)?,
            payload: reader.get_payload()?.to_vec(),
            ephemeral_pubkey,
            nonce,
            timestamp: reader.get_timestamp(),
            claims,
        })
    }
}

impl ToCapnp for SignedEnvelope {
    type Builder<'a> = common_capnp::signed_envelope::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        self.envelope
            .write_to(&mut builder.reborrow().init_envelope());
        builder.set_signature(&self.signature);
        builder.set_signer_pubkey(&self.signer_pubkey);
    }
}

impl FromCapnp for SignedEnvelope {
    type Reader<'a> = common_capnp::signed_envelope::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let signature = {
            let data = reader.get_signature()?;
            if data.len() != 64 {
                return Err(anyhow!(
                    "Invalid signature length: expected 64, got {}",
                    data.len()
                ));
            }
            let mut arr = [0u8; 64];
            arr.copy_from_slice(data);
            arr
        };

        let signer_pubkey = {
            let data = reader.get_signer_pubkey()?;
            if data.len() != 32 {
                return Err(anyhow!(
                    "Invalid signer pubkey length: expected 32, got {}",
                    data.len()
                ));
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(data);
            arr
        };

        Ok(Self {
            envelope: RequestEnvelope::read_from(reader.get_envelope()?)?,
            signature,
            signer_pubkey,
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
    pub signature: [u8; 64],

    /// Ed25519 public key of the signer (32 bytes)
    pub signer_pubkey: [u8; 32],
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
        let signature: [u8; 64] = signature_obj.to_bytes();
        let signer_pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();

        Self {
            request_id,
            payload,
            signature,
            signer_pubkey,
        }
    }

    /// Verify the response signature.
    ///
    /// # Arguments
    ///
    /// * `expected_pubkey` - Optional expected signer public key. If provided,
    ///   verification fails if the signer doesn't match.
    ///
    /// # Returns
    ///
    /// `Ok(())` if signature is valid, `Err` otherwise.
    pub fn verify(&self, expected_pubkey: Option<&VerifyingKey>) -> Result<()> {
        // Check signer matches expected key if provided
        let verifying_key = VerifyingKey::from_bytes(&self.signer_pubkey)
            .map_err(|_| anyhow::anyhow!("Invalid signer public key"))?;

        if let Some(expected) = expected_pubkey {
            if verifying_key.to_bytes() != expected.to_bytes() {
                anyhow::bail!("Response signed by unexpected key");
            }
        }

        // Reconstruct signing data
        let mut signing_data = Vec::with_capacity(8 + self.payload.len());
        signing_data.extend_from_slice(&self.request_id.to_le_bytes());
        signing_data.extend_from_slice(&self.payload);

        // Verify signature
        let signature = ed25519_dalek::Signature::from_bytes(&self.signature);
        verifying_key
            .verify_strict(&signing_data, &signature)
            .map_err(|_| anyhow::anyhow!("Response signature verification failed"))
    }

    /// Get the signer's public key.
    pub fn signer_pubkey(&self) -> Result<VerifyingKey> {
        VerifyingKey::from_bytes(&self.signer_pubkey)
            .map_err(|_| anyhow::anyhow!("Invalid signer public key"))
    }
}

impl ToCapnp for ResponseEnvelope {
    type Builder<'a> = common_capnp::response_envelope::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_request_id(self.request_id);
        builder.set_payload(&self.payload);
        builder.set_signature(&self.signature);
        builder.set_signer_pubkey(&self.signer_pubkey);
    }
}

impl FromCapnp for ResponseEnvelope {
    type Reader<'a> = common_capnp::response_envelope::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let sig_data = reader.get_signature()?;
        if sig_data.len() != 64 {
            anyhow::bail!("Invalid signature length: {}", sig_data.len());
        }
        let mut signature = [0u8; 64];
        signature.copy_from_slice(sig_data);

        let pubkey_data = reader.get_signer_pubkey()?;
        if pubkey_data.len() != 32 {
            anyhow::bail!("Invalid signer pubkey length: {}", pubkey_data.len());
        }
        let mut signer_pubkey = [0u8; 32];
        signer_pubkey.copy_from_slice(pubkey_data);

        Ok(Self {
            request_id: reader.get_request_id(),
            payload: reader.get_payload()?.to_vec(),
            signature,
            signer_pubkey,
        })
    }
}

/// Unwrap and verify a SignedEnvelope from wire bytes.
///
/// Deserializes, verifies signature and replay protection, then extracts
/// the context and payload.
///
/// # Arguments
///
/// * `request` - Raw bytes containing a serialized SignedEnvelope
/// * `server_pubkey` - Expected Ed25519 public key of the signer
/// * `nonce_cache` - Cache for replay protection
///
/// # Returns
///
/// On success, returns `(EnvelopeContext, payload)` where:
/// - `EnvelopeContext` contains verified request metadata
/// - `payload` is the inner request bytes
///
/// # Errors
///
/// Returns error if:
/// - Deserialization fails
/// - Signature verification fails
/// - Replay attack detected (nonce reused or timestamp expired)
pub fn unwrap_envelope(
    request: &[u8],
    server_pubkey: &VerifyingKey,
    nonce_cache: &dyn NonceCache,
) -> Result<(crate::service::EnvelopeContext, Vec<u8>)> {
    use capnp::serialize;

    // Deserialize SignedEnvelope from Cap'n Proto
    let reader = serialize::read_message(
        &mut std::io::Cursor::new(request),
        capnp::message::ReaderOptions::default(),
    )?;
    let signed_reader = reader.get_root::<crate::common_capnp::signed_envelope::Reader>()?;
    let signed = SignedEnvelope::read_from(signed_reader)?;

    // Verify signature and replay protection
    signed.verify(server_pubkey, nonce_cache)?;

    // Extract context and payload
    let ctx = crate::service::EnvelopeContext::from_verified(&signed);
    let payload = signed.payload().to_vec();

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
        capnp::message::ReaderOptions::default(),
    )?;
    let response_reader = reader.get_root::<crate::common_capnp::response_envelope::Reader>()?;
    let envelope = ResponseEnvelope::read_from(response_reader)?;

    // Verify signature
    envelope.verify(expected_pubkey)?;

    Ok((envelope.request_id, envelope.payload))
}

#[cfg(test)]
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
    fn test_local_identity() {
        let identity = RequestIdentity::local();
        assert!(identity.is_local());
        assert!(identity.is_authenticated());
        assert!(!identity.user().is_empty());
    }

    #[test]
    fn test_api_token_identity() {
        let identity = RequestIdentity::api_token("bob", "ci-token");
        assert_eq!(identity.user(), "bob");
        assert!(identity.is_authenticated());
        assert!(!identity.is_local());
    }

    #[test]
    fn test_peer_identity() {
        let key = [0u8; 32];
        let identity = RequestIdentity::peer("gpu-server-1", key);
        assert_eq!(identity.user(), "gpu-server-1");
        assert!(identity.is_authenticated());
    }

    #[test]
    fn test_anonymous_identity() {
        let identity = RequestIdentity::anonymous();
        assert_eq!(identity.user(), "anonymous");
        assert!(!identity.is_authenticated());
    }

    #[test]
    fn test_casbin_subject_namespacing() {
        // Local identity
        let local = RequestIdentity::Local {
            user: "alice".into(),
        };
        assert_eq!(local.casbin_subject(), "local:alice");

        // API token identity
        let token = RequestIdentity::ApiToken {
            user: "bob".into(),
            token_name: "ci".into(),
        };
        assert_eq!(token.casbin_subject(), "token:bob");

        // Peer identity
        let peer = RequestIdentity::Peer {
            name: "gpu-server-1".into(),
            curve_key: [0u8; 32],
        };
        assert_eq!(peer.casbin_subject(), "peer:gpu-server-1");

        // Anonymous
        let anon = RequestIdentity::Anonymous;
        assert_eq!(anon.casbin_subject(), "anonymous");

        // Verify namespacing prevents collisions
        let admin_local = RequestIdentity::Local {
            user: "admin".into(),
        };
        let admin_peer = RequestIdentity::Peer {
            name: "admin".into(),
            curve_key: [0u8; 32],
        };
        assert_ne!(admin_local.casbin_subject(), admin_peer.casbin_subject());
    }

    #[test]
    fn test_request_envelope() {
        let envelope = RequestEnvelope::local(vec![1, 2, 3]);
        assert!(!envelope.user().is_empty());
        assert_eq!(envelope.payload, vec![1, 2, 3]);
        assert!(envelope.request_id > 0);
        assert!(envelope.timestamp > 0);
        assert!(envelope.nonce.iter().any(|&b| b != 0)); // Not all zeros
    }

    #[test]
    fn test_request_envelope_with_ephemeral_pubkey() {
        let pubkey = [42u8; 32];
        let envelope = RequestEnvelope::local(vec![]).with_ephemeral_pubkey(pubkey);
        assert_eq!(envelope.ephemeral_pubkey, Some(pubkey));
    }

    #[test]
    fn test_request_id_increments() {
        let e1 = RequestEnvelope::local(vec![]);
        let e2 = RequestEnvelope::local(vec![]);
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

        let envelope = RequestEnvelope::local(vec![1, 2, 3, 4]);
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

        let envelope = RequestEnvelope::local(vec![1, 2, 3, 4]);
        let signed = SignedEnvelope::new_signed(envelope, &signing_key);

        // Verify with wrong key should fail
        let result = signed.verify(&wrong_verifying_key, &nonce_cache);
        assert!(matches!(result, Err(EnvelopeError::SignerMismatch { .. })));
    }

    #[test]
    fn test_signed_envelope_replay_fails() -> crate::EnvelopeResult<()> {
        let (signing_key, verifying_key) = generate_signing_keypair();
        let nonce_cache = TestNonceCache::new();

        let envelope = RequestEnvelope::local(vec![1, 2, 3, 4]);
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

        let envelope =
            RequestEnvelope::with_token("alice", "deploy", vec![5, 6, 7]).with_ephemeral_pubkey([99u8; 32]);
        let signed = SignedEnvelope::new_signed(envelope, &signing_key);

        assert!(signed.request_id() > 0);
        assert_eq!(signed.identity().user(), "alice");
        assert_eq!(signed.casbin_subject(), "token:alice");
        assert_eq!(signed.payload(), &[5, 6, 7]);
        assert_eq!(signed.ephemeral_pubkey(), Some(&[99u8; 32]));
    }

    #[test]
    fn test_capnp_roundtrip_local() -> anyhow::Result<()> {
        use capnp::message::Builder;

        let identity = RequestIdentity::local();

        // Serialize
        let mut message = Builder::new_default();
        let mut builder = message.init_root::<common_capnp::request_identity::Builder>();
        identity.write_to(&mut builder);

        // Deserialize
        let reader = builder.into_reader();
        let decoded = RequestIdentity::read_from(reader)?;

        assert_eq!(identity, decoded);
        Ok(())
    }

    #[test]
    fn test_capnp_roundtrip_api_token() -> anyhow::Result<()> {
        use capnp::message::Builder;

        let identity = RequestIdentity::api_token("alice", "prod-key");

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<common_capnp::request_identity::Builder>();
        identity.write_to(&mut builder);

        let reader = builder.into_reader();
        let decoded = RequestIdentity::read_from(reader)?;

        assert_eq!(identity, decoded);
        Ok(())
    }

    #[test]
    fn test_capnp_roundtrip_peer() -> anyhow::Result<()> {
        use capnp::message::Builder;

        let curve_key = [42u8; 32];
        let identity = RequestIdentity::peer("gpu-node-1", curve_key);

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<common_capnp::request_identity::Builder>();
        identity.write_to(&mut builder);

        let reader = builder.into_reader();
        let decoded = RequestIdentity::read_from(reader)?;

        assert_eq!(identity, decoded);
        Ok(())
    }

    #[test]
    fn test_capnp_roundtrip_anonymous() -> anyhow::Result<()> {
        use capnp::message::Builder;

        let identity = RequestIdentity::anonymous();

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<common_capnp::request_identity::Builder>();
        identity.write_to(&mut builder);

        let reader = builder.into_reader();
        let decoded = RequestIdentity::read_from(reader)?;

        assert_eq!(identity, decoded);
        Ok(())
    }

    #[test]
    fn test_capnp_roundtrip_envelope() -> anyhow::Result<()> {
        use capnp::message::Builder;

        let envelope = RequestEnvelope::with_token("bob", "ci-pipeline", vec![1, 2, 3, 4])
            .with_ephemeral_pubkey([77u8; 32]);

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<common_capnp::request_envelope::Builder>();
        envelope.write_to(&mut builder);

        let reader = builder.into_reader();
        let decoded = RequestEnvelope::read_from(reader)?;

        assert_eq!(envelope.request_id, decoded.request_id);
        assert_eq!(envelope.identity, decoded.identity);
        assert_eq!(envelope.payload, decoded.payload);
        assert_eq!(envelope.nonce, decoded.nonce);
        assert_eq!(envelope.timestamp, decoded.timestamp);
        assert_eq!(envelope.ephemeral_pubkey, decoded.ephemeral_pubkey);
        Ok(())
    }

    #[test]
    fn test_capnp_roundtrip_signed_envelope() -> Result<(), Box<dyn std::error::Error>> {
        use capnp::message::Builder;

        let (signing_key, verifying_key) = generate_signing_keypair();
        let envelope = RequestEnvelope::local(vec![1, 2, 3]);
        let signed = SignedEnvelope::new_signed(envelope, &signing_key);

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<common_capnp::signed_envelope::Builder>();
        signed.write_to(&mut builder);

        let reader = builder.into_reader();
        let decoded = SignedEnvelope::read_from(reader)?;

        assert_eq!(signed.envelope.request_id, decoded.envelope.request_id);
        assert_eq!(signed.envelope.payload, decoded.envelope.payload);
        assert_eq!(signed.signature, decoded.signature);
        assert_eq!(signed.signer_pubkey, decoded.signer_pubkey);

        // Verify the decoded envelope still has valid signature
        decoded.verify_signature_only(&verifying_key)?;
        Ok(())
    }
}
