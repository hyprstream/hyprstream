//! Diffie-Hellman key exchange for stream MAC key derivation.
//!
//! Streaming responses use MAC authentication instead of per-token Ed25519
//! signatures for performance. The MAC key is derived from a DH shared secret.
//!
//! # Ristretto255
//!
//! This module uses Ristretto255, a prime-order group built on Curve25519.
//! Unlike X25519, Ristretto255 has no cofactor issues:
//! - No small subgroups or low-order points
//! - Invalid encodings are rejected at decode time
//! - If a point decodes successfully, it's safe to use
//!
//! See: https://ristretto.group/why_ristretto.html
//!
//! # Flow
//!
//! 1. Client generates ephemeral Ristretto keypair
//! 2. Client includes ephemeral public key in signed RequestEnvelope
//! 3. Server computes shared secret: `DH(server_secret, client_ephemeral_pubkey)`
//! 4. Both derive MAC key: `KDF(shared_secret || salt, context="mac")`
//! 5. Server MACs each stream chunk, client verifies
//!
//! # Feature Flags
//!
//! - Default: Blake3 derive_key + Ristretto255 DH
//! - `fips`: HKDF-SHA256 + ECDH P-256 (NIST approved)

use subtle::ConstantTimeEq;
use zeroize::{Zeroize, Zeroizing};

use super::backend::derive_key;
use crate::error::{EnvelopeError, EnvelopeResult};

// ============================================================================
// Stream Key Derivation (E2E Authenticated Streaming)
// ============================================================================

/// Derived keys for E2E authenticated streaming.
///
/// Contains:
/// - `topic`: 64-char hex string derived from DH, used as ZMQ subscription prefix
/// - `mac_key`: 32-byte HMAC key for MAC chain verification
///
/// Both client and server derive identical keys from their DH shared secret.
#[derive(Clone)]
pub struct StreamKeys {
    /// Topic for ZMQ PUB/SUB routing (64 hex chars = 32 bytes encoded).
    pub topic: String,
    /// HMAC key for MAC chain (zeroized on drop).
    pub mac_key: Zeroizing<[u8; 32]>,
}

impl StreamKeys {
    /// Create from raw topic and mac_key bytes.
    pub fn new(topic: String, mac_key: [u8; 32]) -> Self {
        Self {
            topic,
            mac_key: Zeroizing::new(mac_key),
        }
    }

    /// Get the first 16 bytes of topic as bytes (for first block's prevMac).
    ///
    /// In the MAC chain, block 0 uses topic[..16] as prevMac.
    pub fn topic_prefix_bytes(&self) -> [u8; 16] {
        // Decode first 32 hex chars (= 16 bytes)
        let mut prefix = [0u8; 16];
        hex::decode_to_slice(&self.topic[..32], &mut prefix)
            .expect("topic is valid hex");
        prefix
    }
}

/// Check if two public keys are identical (self-connection attack).
fn is_self_connection(client_pub: &[u8; 32], server_pub: &[u8; 32]) -> bool {
    client_pub.ct_eq(server_pub).into()
}

/// Derive stream keys (topic and mac_key) from DH shared secret.
///
/// Uses the crypto backend (Blake3 or HKDF-SHA256) with:
/// - IKM: shared_secret || salt (where salt = XOR of client_pub and server_pub)
/// - Context: "hyprstream stream-keys v1 topic" or "hyprstream stream-keys v1 mac"
///
/// # Arguments
///
/// * `shared_secret` - 32-byte DH shared secret
/// * `client_pub` - Client's ephemeral Ristretto public key (32 bytes)
/// * `server_pub` - Server's ephemeral Ristretto public key (32 bytes)
///
/// # Returns
///
/// `StreamKeys` containing the derived topic (64 hex chars) and mac_key (32 bytes).
///
/// # Errors
///
/// Returns `EnvelopeError::KeyExchange` if client and server keys are identical.
///
/// # Security
///
/// - Ristretto255 eliminates low-order point attacks by construction
/// - XOR salt ensures both parties' keys are bound to the derivation
/// - Self-connection check prevents replay attacks
///
/// # Backend
///
/// - Default: Blake3 `derive_key()` (~10+ GB/s with SIMD)
/// - FIPS mode: HKDF-SHA256 (NIST SP 800-56C)
pub fn derive_stream_keys(
    shared_secret: &[u8; 32],
    client_pub: &[u8; 32],
    server_pub: &[u8; 32],
) -> EnvelopeResult<StreamKeys> {
    // Ristretto255: No low-order point checks needed!
    // Invalid encodings are rejected at decode time, and all valid
    // Ristretto points are in the prime-order subgroup.

    // Security check: reject self-connection
    if is_self_connection(client_pub, server_pub) {
        return Err(EnvelopeError::KeyExchange(
            "client and server keys are identical".into(),
        ));
    }

    // XOR public keys for salt (binds both parties' keys to the derivation)
    let mut salt = [0u8; 32];
    for i in 0..32 {
        salt[i] = client_pub[i] ^ server_pub[i];
    }

    // Build IKM: shared_secret || salt
    let mut ikm = [0u8; 64];
    ikm[..32].copy_from_slice(shared_secret);
    ikm[32..64].copy_from_slice(&salt);

    // Derive topic (32 bytes -> 64 hex chars)
    let topic_bytes = derive_key("hyprstream stream-keys v1 topic", &ikm);
    let topic = hex::encode(topic_bytes);

    // Derive mac_key (32 bytes)
    let mac_key = derive_key("hyprstream stream-keys v1 mac", &ikm);

    Ok(StreamKeys::new(topic, mac_key))
}

/// Shared secret from DH key exchange.
///
/// Wrapped in `Zeroizing` to ensure secure erasure when dropped.
#[derive(Clone)]
pub struct SharedSecret(Zeroizing<[u8; 32]>);

impl SharedSecret {
    /// Create from raw bytes.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(Zeroizing::new(bytes))
    }

    /// Get the raw bytes (for HKDF input).
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Derive a MAC key from this shared secret using the crypto backend.
    ///
    /// # Arguments
    ///
    /// * `request_id` - Used in IKM for domain separation
    /// * `info` - Context info (e.g., "stream-hmac")
    ///
    /// # Backend
    ///
    /// - Default: Blake3 `derive_key()` with fixed context
    /// - FIPS mode: HKDF-SHA256
    pub fn derive_hmac_key(&self, request_id: u64, info: &[u8]) -> [u8; 32] {
        // Fixed context string (not dynamic for security)
        const CONTEXT: &str = "hyprstream hmac v1";

        // Build IKM: shared_secret || request_id || info
        let mut ikm = Vec::with_capacity(40 + info.len());
        ikm.extend_from_slice(self.as_bytes());
        ikm.extend_from_slice(&request_id.to_le_bytes());
        ikm.extend_from_slice(info);

        derive_key(CONTEXT, &ikm)
    }
}

impl Zeroize for SharedSecret {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

/// Trait for Diffie-Hellman key exchange implementations.
///
/// This trait abstracts over Ristretto255 (default) and ECDH P-256 (FIPS mode).
pub trait KeyExchange: Send + Sync {
    /// Secret key type (zeroized on drop).
    type SecretKey: Zeroize + Clone;

    /// Public key type.
    type PublicKey: AsRef<[u8]> + Clone;

    /// Generate a new keypair.
    fn generate_keypair() -> (Self::SecretKey, Self::PublicKey);

    /// Derive shared secret from our secret key and their public key.
    fn derive_shared(
        secret: &Self::SecretKey,
        their_pubkey: &Self::PublicKey,
    ) -> EnvelopeResult<SharedSecret>;

    /// Deserialize a public key from bytes.
    fn pubkey_from_bytes(bytes: &[u8]) -> EnvelopeResult<Self::PublicKey>;

    /// Serialize a public key to bytes.
    fn pubkey_to_bytes(pubkey: &Self::PublicKey) -> Vec<u8>;
}

// ============================================================================
// Ristretto255 Implementation (default)
// ============================================================================

#[cfg(not(feature = "fips"))]
mod ristretto_impl {
    use super::*;
    use curve25519_dalek::{
        constants::RISTRETTO_BASEPOINT_POINT,
        ristretto::{CompressedRistretto, RistrettoPoint},
        scalar::Scalar,
    };

    /// Ristretto255 secret key (scalar).
    ///
    /// Zeroized on drop to prevent key material from lingering in memory.
    #[derive(Clone)]
    pub struct RistrettoSecret(Scalar);

    impl RistrettoSecret {
        /// Get the underlying scalar for DH computation.
        pub fn scalar(&self) -> &Scalar {
            &self.0
        }
    }

    impl Zeroize for RistrettoSecret {
        fn zeroize(&mut self) {
            // Overwrite scalar with zero
            self.0 = Scalar::ZERO;
        }
    }

    impl Drop for RistrettoSecret {
        fn drop(&mut self) {
            self.zeroize();
        }
    }

    /// Ristretto255 public key (compressed point).
    ///
    /// Stores the compressed form for efficient serialization.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct RistrettoPublic {
        point: RistrettoPoint,
        compressed: [u8; 32],
    }

    impl RistrettoPublic {
        /// Create from a RistrettoPoint.
        pub fn from_point(point: RistrettoPoint) -> Self {
            let compressed = point.compress().to_bytes();
            Self { point, compressed }
        }

        /// Get the underlying point for DH computation.
        pub fn point(&self) -> &RistrettoPoint {
            &self.point
        }

        /// Serialize to 32 bytes.
        pub fn to_bytes(&self) -> [u8; 32] {
            self.compressed
        }

        /// Deserialize from 32 bytes.
        ///
        /// Returns `None` if the encoding is invalid.
        /// This is the primary defense against invalid points -
        /// Ristretto255 rejects all non-canonical encodings.
        pub fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
            let compressed = CompressedRistretto::from_slice(bytes).ok()?;
            let point = compressed.decompress()?;
            Some(Self {
                point,
                compressed: *bytes,
            })
        }
    }

    impl AsRef<[u8]> for RistrettoPublic {
        fn as_ref(&self) -> &[u8] {
            &self.compressed
        }
    }

    impl RistrettoPublic {
        /// Deserialize from a byte slice (must be exactly 32 bytes).
        ///
        /// This is a convenience wrapper for streaming that handles &[u8] input.
        pub fn from_slice(bytes: &[u8]) -> Option<Self> {
            if bytes.len() != 32 {
                return None;
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(bytes);
            Self::from_bytes(&arr)
        }
    }

    /// Ristretto255 key exchange implementation.
    pub struct RistrettoKeyExchange;

    impl KeyExchange for RistrettoKeyExchange {
        type SecretKey = RistrettoSecret;
        type PublicKey = RistrettoPublic;

        fn generate_keypair() -> (Self::SecretKey, Self::PublicKey) {
            let secret = Scalar::random(&mut rand::thread_rng());
            let public_point = RISTRETTO_BASEPOINT_POINT * secret;
            (
                RistrettoSecret(secret),
                RistrettoPublic::from_point(public_point),
            )
        }

        fn derive_shared(
            secret: &Self::SecretKey,
            their_pubkey: &Self::PublicKey,
        ) -> EnvelopeResult<SharedSecret> {
            let shared_point = their_pubkey.point() * secret.scalar();
            let shared_bytes = shared_point.compress().to_bytes();
            Ok(SharedSecret::new(shared_bytes))
        }

        fn pubkey_from_bytes(bytes: &[u8]) -> EnvelopeResult<Self::PublicKey> {
            if bytes.len() != 32 {
                return Err(EnvelopeError::InvalidPublicKey {
                    expected: 32,
                    actual: bytes.len(),
                });
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(bytes);
            RistrettoPublic::from_bytes(&arr).ok_or_else(|| {
                EnvelopeError::KeyExchange("invalid ristretto255 point encoding".into())
            })
        }

        fn pubkey_to_bytes(pubkey: &Self::PublicKey) -> Vec<u8> {
            pubkey.to_bytes().to_vec()
        }
    }

    /// Generate an ephemeral Ristretto255 keypair (for one-time use in requests).
    pub fn generate_ephemeral_keypair() -> (RistrettoSecret, RistrettoPublic) {
        RistrettoKeyExchange::generate_keypair()
    }

    /// Perform Ristretto255 Diffie-Hellman and return shared secret bytes.
    pub fn ristretto_dh(secret: &RistrettoSecret, their_public: &RistrettoPublic) -> [u8; 32] {
        let shared_point = their_public.point() * secret.scalar();
        shared_point.compress().to_bytes()
    }
}

#[cfg(not(feature = "fips"))]
pub use ristretto_impl::{
    generate_ephemeral_keypair, ristretto_dh, RistrettoKeyExchange, RistrettoPublic,
    RistrettoSecret,
};

// ============================================================================
// ECDH P-256 Implementation (FIPS mode)
// ============================================================================

#[cfg(feature = "fips")]
mod p256_impl {
    use super::*;
    use p256::{
        elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint},
        EncodedPoint, PublicKey, SecretKey,
    };

    /// P-256 secret key wrapper with Zeroize.
    #[derive(Clone)]
    pub struct P256SecretKey(SecretKey);

    impl Zeroize for P256SecretKey {
        fn zeroize(&mut self) {
            // SecretKey uses NonZeroScalar which is zeroized on drop
        }
    }

    /// P-256 public key wrapper.
    #[derive(Clone)]
    pub struct P256PublicKey(PublicKey);

    impl P256PublicKey {
        /// Deserialize from a byte slice (33-byte compressed or 65-byte uncompressed).
        ///
        /// Note: P-256 keys are NOT compatible with the 32-byte Ristretto format.
        /// This method returns None for 32-byte inputs.
        pub fn from_slice(bytes: &[u8]) -> Option<Self> {
            // P-256 uses SEC1 encoding: 33 bytes compressed, 65 bytes uncompressed
            // 32-byte inputs are invalid for P-256
            let point = EncodedPoint::from_bytes(bytes).ok()?;
            let pubkey: Option<PublicKey> = PublicKey::from_encoded_point(&point).into();
            pubkey.map(P256PublicKey)
        }

        /// Serialize to compressed SEC1 format (33 bytes).
        pub fn to_bytes(&self) -> Vec<u8> {
            self.0.to_encoded_point(true).as_bytes().to_vec()
        }
    }

    impl AsRef<[u8]> for P256PublicKey {
        fn as_ref(&self) -> &[u8] {
            // Note: This returns an empty slice because we can't return a reference
            // to temporary data. Use to_bytes() for serialization instead.
            &[]
        }
    }

    /// ECDH P-256 key exchange implementation.
    pub struct EcdhP256KeyExchange;

    impl KeyExchange for EcdhP256KeyExchange {
        type SecretKey = P256SecretKey;
        type PublicKey = P256PublicKey;

        fn generate_keypair() -> (Self::SecretKey, Self::PublicKey) {
            let secret = SecretKey::random(&mut rand::thread_rng());
            let public = secret.public_key();
            (P256SecretKey(secret), P256PublicKey(public))
        }

        fn derive_shared(
            secret: &Self::SecretKey,
            their_pubkey: &Self::PublicKey,
        ) -> EnvelopeResult<SharedSecret> {
            use p256::ecdh::diffie_hellman;

            let shared = diffie_hellman(secret.0.to_nonzero_scalar(), their_pubkey.0.as_affine());
            let mut bytes = [0u8; 32];
            bytes.copy_from_slice(shared.raw_secret_bytes());
            Ok(SharedSecret::new(bytes))
        }

        fn pubkey_from_bytes(bytes: &[u8]) -> EnvelopeResult<Self::PublicKey> {
            let point =
                EncodedPoint::from_bytes(bytes).map_err(|_| EnvelopeError::InvalidPublicKey {
                    expected: 33, // compressed
                    actual: bytes.len(),
                })?;
            let pubkey: Option<PublicKey> = PublicKey::from_encoded_point(&point).into();
            pubkey
                .map(P256PublicKey)
                .ok_or(EnvelopeError::InvalidPublicKey {
                    expected: 33,
                    actual: bytes.len(),
                })
        }

        fn pubkey_to_bytes(pubkey: &Self::PublicKey) -> Vec<u8> {
            pubkey.0.to_encoded_point(true).as_bytes().to_vec()
        }
    }

    /// Generate an ephemeral P-256 keypair (FIPS mode).
    pub fn generate_ephemeral_keypair() -> (P256SecretKey, P256PublicKey) {
        EcdhP256KeyExchange::generate_keypair()
    }

    /// Perform P-256 ECDH and return shared secret bytes.
    ///
    /// FIPS-mode equivalent of `ristretto_dh()`.
    pub fn p256_dh(secret: &P256SecretKey, their_public: &P256PublicKey) -> [u8; 32] {
        use p256::ecdh::diffie_hellman;

        let shared = diffie_hellman(secret.0.to_nonzero_scalar(), their_public.0.as_affine());
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(shared.raw_secret_bytes());
        bytes
    }
}

#[cfg(feature = "fips")]
pub use p256_impl::{
    generate_ephemeral_keypair, p256_dh, EcdhP256KeyExchange, P256PublicKey, P256SecretKey,
};

// ============================================================================
// Default Key Exchange Type
// ============================================================================

/// Default key exchange algorithm based on feature flags.
#[cfg(not(feature = "fips"))]
pub type DefaultKeyExchange = RistrettoKeyExchange;

#[cfg(feature = "fips")]
pub type DefaultKeyExchange = EcdhP256KeyExchange;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_exchange_roundtrip() {
        // Generate two keypairs
        let (secret_a, public_a) = DefaultKeyExchange::generate_keypair();
        let (secret_b, public_b) = DefaultKeyExchange::generate_keypair();

        // Both sides derive the same shared secret
        let shared_a = DefaultKeyExchange::derive_shared(&secret_a, &public_b).unwrap();
        let shared_b = DefaultKeyExchange::derive_shared(&secret_b, &public_a).unwrap();

        assert_eq!(shared_a.as_bytes(), shared_b.as_bytes());
    }

    #[test]
    fn test_hmac_key_derivation() {
        let (secret_a, public_a) = DefaultKeyExchange::generate_keypair();
        let (secret_b, public_b) = DefaultKeyExchange::generate_keypair();

        let shared_a = DefaultKeyExchange::derive_shared(&secret_a, &public_b).unwrap();
        let shared_b = DefaultKeyExchange::derive_shared(&secret_b, &public_a).unwrap();

        let request_id = 12345u64;
        let hmac_key_a = shared_a.derive_hmac_key(request_id, b"stream-hmac");
        let hmac_key_b = shared_b.derive_hmac_key(request_id, b"stream-hmac");

        assert_eq!(hmac_key_a, hmac_key_b);
    }

    #[test]
    fn test_different_request_ids_different_keys() {
        let (_secret_a, public_a) = DefaultKeyExchange::generate_keypair();
        let (secret_b, _public_b) = DefaultKeyExchange::generate_keypair();

        let shared = DefaultKeyExchange::derive_shared(&secret_b, &public_a).unwrap();

        let key1 = shared.derive_hmac_key(1, b"stream-hmac");
        let key2 = shared.derive_hmac_key(2, b"stream-hmac");

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pubkey_serialization_roundtrip() {
        let (_secret, public) = DefaultKeyExchange::generate_keypair();

        let bytes = DefaultKeyExchange::pubkey_to_bytes(&public);
        let restored = DefaultKeyExchange::pubkey_from_bytes(&bytes).unwrap();

        assert_eq!(
            DefaultKeyExchange::pubkey_to_bytes(&public),
            DefaultKeyExchange::pubkey_to_bytes(&restored)
        );
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_invalid_ristretto_encoding_rejected() {
        // Note: All-zeros is the identity point in Ristretto255 (valid encoding)
        // We need to use truly invalid encodings

        // All 0xFF is not a valid Ristretto encoding (with high probability)
        let garbage = [0xffu8; 32];
        let result = RistrettoPublic::from_bytes(&garbage);
        assert!(result.is_none(), "0xFF bytes should be rejected");

        // A point with the high bit set incorrectly (non-canonical)
        let mut invalid = [0u8; 32];
        invalid[31] = 0x80; // Set high bit - invalid for Ristretto
        let result = RistrettoPublic::from_bytes(&invalid);
        assert!(result.is_none(), "Non-canonical encoding should be rejected");
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_ristretto_dh_function() {
        let (client_secret, client_public) = generate_ephemeral_keypair();
        let (server_secret, server_public) = generate_ephemeral_keypair();

        // Both compute shared secret
        let client_shared = ristretto_dh(&client_secret, &server_public);
        let server_shared = ristretto_dh(&server_secret, &client_public);

        assert_eq!(client_shared, server_shared);
    }

    // =========================================================================
    // Stream key derivation tests
    // =========================================================================

    #[test]
    fn test_derive_stream_keys_deterministic() {
        // Same inputs should produce same outputs
        let shared_secret = [0x42u8; 32];
        let client_pub = [0x01u8; 32];
        let server_pub = [0x02u8; 32];

        let keys1 = derive_stream_keys(&shared_secret, &client_pub, &server_pub).unwrap();
        let keys2 = derive_stream_keys(&shared_secret, &client_pub, &server_pub).unwrap();

        assert_eq!(keys1.topic, keys2.topic);
        assert_eq!(*keys1.mac_key, *keys2.mac_key);
    }

    #[test]
    fn test_derive_stream_keys_symmetric() {
        // Order of pubkeys shouldn't matter (XOR is commutative)
        let shared_secret = [0x42u8; 32];
        let client_pub = [0x01u8; 32];
        let server_pub = [0x02u8; 32];

        let keys1 = derive_stream_keys(&shared_secret, &client_pub, &server_pub).unwrap();
        let keys2 = derive_stream_keys(&shared_secret, &server_pub, &client_pub).unwrap();

        // Both should produce identical keys (XOR is commutative)
        assert_eq!(keys1.topic, keys2.topic);
        assert_eq!(*keys1.mac_key, *keys2.mac_key);
    }

    #[test]
    fn test_derive_stream_keys_different_secrets() {
        let client_pub = [0x01u8; 32];
        let server_pub = [0x02u8; 32];

        let keys1 = derive_stream_keys(&[0x11u8; 32], &client_pub, &server_pub).unwrap();
        let keys2 = derive_stream_keys(&[0x22u8; 32], &client_pub, &server_pub).unwrap();

        assert_ne!(keys1.topic, keys2.topic);
        assert_ne!(*keys1.mac_key, *keys2.mac_key);
    }

    #[test]
    fn test_derive_stream_keys_different_pubkeys() {
        let shared_secret = [0x42u8; 32];

        let keys1 = derive_stream_keys(&shared_secret, &[0x01u8; 32], &[0x02u8; 32]).unwrap();
        let keys2 = derive_stream_keys(&shared_secret, &[0x03u8; 32], &[0x04u8; 32]).unwrap();

        assert_ne!(keys1.topic, keys2.topic);
        assert_ne!(*keys1.mac_key, *keys2.mac_key);
    }

    #[test]
    fn test_derive_stream_keys_topic_is_hex() {
        let shared_secret = [0x42u8; 32];
        let client_pub = [0x01u8; 32];
        let server_pub = [0x02u8; 32];

        let keys = derive_stream_keys(&shared_secret, &client_pub, &server_pub).unwrap();

        // Topic should be 64 hex chars
        assert_eq!(keys.topic.len(), 64);
        assert!(keys.topic.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_derive_stream_keys_topic_prefix() {
        let shared_secret = [0x42u8; 32];
        let client_pub = [0x01u8; 32];
        let server_pub = [0x02u8; 32];

        let keys = derive_stream_keys(&shared_secret, &client_pub, &server_pub).unwrap();
        let prefix = keys.topic_prefix_bytes();

        // Prefix should be first 16 bytes of topic
        assert_eq!(prefix.len(), 16);

        // Verify it matches the topic
        let expected_prefix = hex::decode(&keys.topic[..32]).unwrap();
        assert_eq!(&prefix[..], &expected_prefix[..]);
    }

    #[test]
    fn test_derive_stream_keys_rejects_self_connection() {
        let shared_secret = [0x42u8; 32];
        let same_key = [0x01u8; 32];

        let result = derive_stream_keys(&shared_secret, &same_key, &same_key);
        assert!(
            matches!(result, Err(EnvelopeError::KeyExchange(ref msg)) if msg.contains("identical")),
            "Should reject self-connection"
        );
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_derive_stream_keys_with_real_dh() {
        // Full integration test with actual DH key exchange
        let (client_secret, client_pubkey) = generate_ephemeral_keypair();
        let (server_secret, server_pubkey) = generate_ephemeral_keypair();

        // Both sides compute shared secret
        let client_shared = ristretto_dh(&client_secret, &server_pubkey);
        let server_shared = ristretto_dh(&server_secret, &client_pubkey);

        // Both sides derive stream keys
        let client_keys = derive_stream_keys(
            &client_shared,
            &client_pubkey.to_bytes(),
            &server_pubkey.to_bytes(),
        )
        .unwrap();

        let server_keys = derive_stream_keys(
            &server_shared,
            &client_pubkey.to_bytes(),
            &server_pubkey.to_bytes(),
        )
        .unwrap();

        // Keys should be identical
        assert_eq!(client_keys.topic, server_keys.topic);
        assert_eq!(*client_keys.mac_key, *server_keys.mac_key);
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_stream_keys_e2e_mac_verification() {
        use crate::crypto::hmac::ChainedStreamHmac;

        // Full E2E test: DH → stream keys → MAC chain
        let (client_secret, client_pubkey) = generate_ephemeral_keypair();
        let (server_secret, server_pubkey) = generate_ephemeral_keypair();

        let client_shared = ristretto_dh(&client_secret, &server_pubkey);
        let server_shared = ristretto_dh(&server_secret, &client_pubkey);

        let client_keys = derive_stream_keys(
            &client_shared,
            &client_pubkey.to_bytes(),
            &server_pubkey.to_bytes(),
        )
        .unwrap();

        let server_keys = derive_stream_keys(
            &server_shared,
            &client_pubkey.to_bytes(),
            &server_pubkey.to_bytes(),
        )
        .unwrap();

        // Server produces MAC chain
        // Use topic prefix as initial prev_mac (converted to u64 for request_id)
        let prefix = server_keys.topic_prefix_bytes();
        let request_id = u64::from_le_bytes(prefix[..8].try_into().unwrap());

        let mut producer = ChainedStreamHmac::from_bytes(*server_keys.mac_key, request_id);
        let mac1 = producer.compute_next(b"token 1");
        let mac2 = producer.compute_next(b"token 2");
        let mac3 = producer.compute_next(b"[DONE]");

        // Client verifies MAC chain
        let mut verifier = ChainedStreamHmac::from_bytes(*client_keys.mac_key, request_id);
        verifier.verify_next(b"token 1", &mac1).unwrap();
        verifier.verify_next(b"token 2", &mac2).unwrap();
        verifier.verify_next(b"[DONE]", &mac3).unwrap();
    }
}
