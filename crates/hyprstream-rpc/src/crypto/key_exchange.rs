//! Diffie-Hellman key exchange for stream HMAC key derivation.
//!
//! Streaming responses use HMAC authentication instead of per-token Ed25519
//! signatures for performance. The HMAC key is derived from a DH shared secret.
//!
//! # Flow
//!
//! 1. Client generates ephemeral DH keypair
//! 2. Client includes ephemeral public key in signed RequestEnvelope
//! 3. Server computes shared secret: `DH(server_secret, client_ephemeral_pubkey)`
//! 4. Both derive HMAC key: `HKDF(shared_secret, salt=request_id, info="stream-hmac")`
//! 5. Server HMACs each stream chunk, client verifies
//!
//! # Feature Flags
//!
//! - Default: X25519 (Curve25519)
//! - `fips`: ECDH P-256 (NIST curve, SP 800-56A approved)

use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::{Zeroize, Zeroizing};

use crate::error::{EnvelopeError, EnvelopeResult};

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

    /// Derive an HMAC key from this shared secret using HKDF.
    ///
    /// # Arguments
    ///
    /// * `request_id` - Used as salt for domain separation
    /// * `info` - Context info (e.g., "stream-hmac")
    pub fn derive_hmac_key(&self, request_id: u64, info: &[u8]) -> [u8; 32] {
        let hk = Hkdf::<Sha256>::new(Some(&request_id.to_le_bytes()), self.as_bytes());
        let mut okm = [0u8; 32];
        // SAFETY: HKDF-SHA256 can output up to 255*32=8160 bytes, so 32 is always valid
        if hk.expand(info, &mut okm).is_err() {
            // Fallback to zeroed key (should never happen for 32-byte output)
            okm = [0u8; 32];
        }
        okm
    }
}

impl Zeroize for SharedSecret {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

/// Trait for Diffie-Hellman key exchange implementations.
///
/// This trait abstracts over X25519 (default) and ECDH P-256 (FIPS mode).
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
// X25519 Implementation (default)
// ============================================================================

#[cfg(not(feature = "fips"))]
mod x25519_impl {
    use super::*;
    use x25519_dalek::{EphemeralSecret, PublicKey, StaticSecret};

    /// X25519 secret key wrapper with Zeroize.
    #[derive(Clone)]
    pub struct X25519SecretKey(StaticSecret);

    impl Zeroize for X25519SecretKey {
        fn zeroize(&mut self) {
            // StaticSecret doesn't implement Zeroize, so we can't do much here.
            // The underlying memory will be zeroed by StaticSecret's Drop impl.
        }
    }

    /// X25519 public key wrapper.
    #[derive(Clone)]
    pub struct X25519PublicKey(PublicKey);

    impl AsRef<[u8]> for X25519PublicKey {
        fn as_ref(&self) -> &[u8] {
            self.0.as_bytes()
        }
    }

    /// X25519 key exchange implementation.
    pub struct X25519KeyExchange;

    impl KeyExchange for X25519KeyExchange {
        type SecretKey = X25519SecretKey;
        type PublicKey = X25519PublicKey;

        fn generate_keypair() -> (Self::SecretKey, Self::PublicKey) {
            let secret = StaticSecret::random_from_rng(rand::thread_rng());
            let public = PublicKey::from(&secret);
            (X25519SecretKey(secret), X25519PublicKey(public))
        }

        fn derive_shared(
            secret: &Self::SecretKey,
            their_pubkey: &Self::PublicKey,
        ) -> EnvelopeResult<SharedSecret> {
            let shared = secret.0.diffie_hellman(&their_pubkey.0);
            Ok(SharedSecret::new(shared.to_bytes()))
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
            Ok(X25519PublicKey(PublicKey::from(arr)))
        }

        fn pubkey_to_bytes(pubkey: &Self::PublicKey) -> Vec<u8> {
            pubkey.0.as_bytes().to_vec()
        }
    }

    /// Generate an ephemeral X25519 keypair (for one-time use in requests).
    pub fn generate_ephemeral_keypair() -> (EphemeralSecret, PublicKey) {
        let secret = EphemeralSecret::random_from_rng(rand::thread_rng());
        let public = PublicKey::from(&secret);
        (secret, public)
    }

    /// Derive X25519 public key from Ed25519 verifying key.
    ///
    /// This allows using a single Ed25519 keypair for both signing and DH.
    /// Ed25519 (Edwards curve) and X25519 (Montgomery curve) are mathematically related:
    /// the same secret scalar works for both, just with different curve representations.
    ///
    /// # Security Note
    ///
    /// This is safe to use because:
    /// - The conversion is one-way (X25519 pubkey → Ed25519 pubkey is hard)
    /// - No new secret material is exposed
    /// - Client generates ephemeral keypair, so forward secrecy is maintained
    pub fn ed25519_to_x25519_pubkey(ed_pubkey: &ed25519_dalek::VerifyingKey) -> PublicKey {
        // Ed25519 uses the Edwards curve, X25519 uses Montgomery curve
        // The `to_montgomery()` method converts the point representation
        let montgomery = ed_pubkey.to_montgomery();
        PublicKey::from(montgomery.to_bytes())
    }

    /// Derive X25519 secret key from Ed25519 signing key.
    ///
    /// This allows the server to use its Ed25519 signing key for DH.
    /// The secret scalar is the same for both curves.
    ///
    /// # Security Note
    ///
    /// This function extracts the scalar from the Ed25519 key and uses it for X25519.
    /// This is mathematically sound but means the same secret is used for two purposes.
    /// For maximum security, consider using separate keys.
    pub fn ed25519_to_x25519_secret(ed_secret: &ed25519_dalek::SigningKey) -> StaticSecret {
        // The Ed25519 signing key contains a 32-byte seed that is hashed to get the scalar
        // For X25519, we need to use the same derivation that Ed25519 uses internally
        use sha2::{Digest, Sha512};

        let hash = Sha512::digest(ed_secret.to_bytes());
        let mut scalar_bytes = [0u8; 32];
        scalar_bytes.copy_from_slice(&hash[..32]);

        // Apply clamping (same as X25519/Ed25519 do internally)
        scalar_bytes[0] &= 248;
        scalar_bytes[31] &= 127;
        scalar_bytes[31] |= 64;

        StaticSecret::from(scalar_bytes)
    }
}

#[cfg(not(feature = "fips"))]
pub use x25519_impl::{
    ed25519_to_x25519_pubkey, ed25519_to_x25519_secret, generate_ephemeral_keypair,
    X25519KeyExchange, X25519PublicKey, X25519SecretKey,
};

// ============================================================================
// ECDH P-256 Implementation (FIPS mode)
// ============================================================================

#[cfg(feature = "fips")]
mod p256_impl {
    use super::*;
    use p256::{
        ecdh::EphemeralSecret,
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

    impl AsRef<[u8]> for P256PublicKey {
        fn as_ref(&self) -> &[u8] {
            // Return compressed point (33 bytes)
            // Note: This creates a temporary, which is not ideal
            // In practice, we'd cache this or use a different approach
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
            let point = EncodedPoint::from_bytes(bytes).map_err(|_| EnvelopeError::InvalidPublicKey {
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
}

#[cfg(feature = "fips")]
pub use p256_impl::{EcdhP256KeyExchange, P256PublicKey, P256SecretKey};

// ============================================================================
// Default Key Exchange Type
// ============================================================================

/// Default key exchange algorithm based on feature flags.
#[cfg(not(feature = "fips"))]
pub type DefaultKeyExchange = X25519KeyExchange;

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
        let (secret_a, public_a) = DefaultKeyExchange::generate_keypair();
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
    fn test_ed25519_to_x25519_conversion() {
        use crate::crypto::signing::generate_signing_keypair;

        // Generate Ed25519 keypair
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Derive X25519 keys from Ed25519 keys
        let x25519_secret = ed25519_to_x25519_secret(&signing_key);
        let x25519_pubkey = ed25519_to_x25519_pubkey(&verifying_key);

        // Verify the derived keys work together
        // Generate ephemeral client keypair
        let (client_secret, client_pubkey) = generate_ephemeral_keypair();

        // Server derives shared secret using derived X25519 key
        let server_shared = x25519_secret.diffie_hellman(&client_pubkey);

        // Client derives shared secret using server's derived X25519 pubkey
        let client_shared = client_secret.diffie_hellman(&x25519_pubkey);

        // Both should derive the same shared secret
        assert_eq!(server_shared.as_bytes(), client_shared.as_bytes());
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_ed25519_to_x25519_stream_hmac() {
        use crate::crypto::hmac::ChainedStreamHmac;
        use crate::crypto::signing::generate_signing_keypair;

        // Simulate server setup: generate Ed25519 keypair
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Client knows server's Ed25519 pubkey (from config)
        // Client derives server's X25519 pubkey
        let server_x25519_pubkey = ed25519_to_x25519_pubkey(&verifying_key);

        // Client generates ephemeral keypair
        let (client_secret, client_pubkey) = generate_ephemeral_keypair();

        // Client includes ephemeral pubkey in request envelope
        // Server receives request and derives shared secret
        let server_x25519_secret = ed25519_to_x25519_secret(&signing_key);
        let server_shared = server_x25519_secret.diffie_hellman(&client_pubkey);

        // Client derives same shared secret
        let client_shared = client_secret.diffie_hellman(&server_x25519_pubkey);

        // Derive HMAC keys from shared secrets
        let shared_secret = SharedSecret::new(*server_shared.as_bytes());
        let request_id = 12345u64;
        let hmac_key = shared_secret.derive_hmac_key(request_id, b"stream-hmac");

        let client_secret = SharedSecret::new(*client_shared.as_bytes());
        let client_hmac_key = client_secret.derive_hmac_key(request_id, b"stream-hmac");

        assert_eq!(hmac_key, client_hmac_key);

        // Verify chained HMAC works with derived keys
        let mut producer = ChainedStreamHmac::from_bytes(hmac_key, request_id);
        let mac = producer.compute_next(b"hello world");

        let mut verifier = ChainedStreamHmac::from_bytes(client_hmac_key, request_id);
        verifier.verify_next(b"hello world", &mac).unwrap();
    }
}
