//! Broadcast encryption primitives for the notification service.
//!
//! Provides AES-256-GCM encryption with per-subscriber key wrapping,
//! one-shot MAC authentication, and Ed25519 publisher attestation.
//!
//! # Design
//!
//! - Publisher encrypts payload once with a random `data_key`
//! - Each subscriber gets `data_key` wrapped with their DH-derived `enc_key`
//! - One-shot MAC authenticates ciphertext per subscriber
//! - Ed25519 attestation binds ephemeral DH key to publisher identity
//!
//! # Security Properties
//!
//! - NotificationService (blind relay) never sees plaintext or DH secrets
//! - AAD binding prevents cross-intent and cross-subscriber confusion
//! - Length-prefixed AAD prevents concatenation ambiguity
//! - All nonces are random from OsRng (never derived)

use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use subtle::ConstantTimeEq;
use zeroize::{Zeroize, Zeroizing};

use super::backend::keyed_mac;
use super::key_exchange::derive_notification_keys;
#[cfg(not(feature = "fips"))]
use super::key_exchange::{blinded_dh_raw, reconstruct_blinded_pub_raw, ristretto_dh_raw};
use crate::error::{EnvelopeError, EnvelopeResult};

/// 128-bit pubkey fingerprint (Blake3 truncated).
pub type PubkeyFingerprint = [u8; 16];

/// Compute a 128-bit fingerprint of a public key: `Blake3(pubkey)[..16]`.
pub fn pubkey_fingerprint(pubkey_bytes: &[u8; 32]) -> PubkeyFingerprint {
    let hash = blake3::hash(pubkey_bytes);
    let mut fp = [0u8; 16];
    fp.copy_from_slice(&hash.as_bytes()[..16]);
    fp
}

/// Build length-prefixed AAD for payload encryption.
///
/// Format: `u32_le(len(intent_id)) || intent_id || u32_le(len(scope)) || scope`
///
/// Length prefixing prevents ambiguity where different intent_id/scope splits
/// produce the same concatenated bytes.
pub fn build_payload_aad(intent_id: &str, scope: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(8 + intent_id.len() + scope.len());
    aad.extend_from_slice(&(intent_id.len() as u32).to_le_bytes());
    aad.extend_from_slice(intent_id.as_bytes());
    aad.extend_from_slice(&(scope.len() as u32).to_le_bytes());
    aad.extend_from_slice(scope.as_bytes());
    aad
}

/// Generate a random 12-byte AES-GCM nonce from OsRng.
fn random_nonce() -> [u8; 12] {
    let mut nonce = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    nonce
}

/// Generate a random 32-byte AES-256 key from OsRng.
fn random_data_key() -> Zeroizing<[u8; 32]> {
    let mut key = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut key);
    Zeroizing::new(key)
}

// ============================================================================
// AES-256-GCM Helpers
// ============================================================================

/// Encrypt with AES-256-GCM.
fn aes_gcm_encrypt(key: &[u8; 32], nonce: &[u8; 12], plaintext: &[u8], aad: &[u8]) -> EnvelopeResult<Vec<u8>> {
    let cipher = Aes256Gcm::new(key.into());
    let payload = Payload { msg: plaintext, aad };
    cipher
        .encrypt(Nonce::from_slice(nonce), payload)
        .map_err(|_| EnvelopeError::Encryption("AES-GCM encrypt failed".into()))
}

/// Decrypt with AES-256-GCM.
fn aes_gcm_decrypt(key: &[u8; 32], nonce: &[u8; 12], ciphertext: &[u8], aad: &[u8]) -> EnvelopeResult<Vec<u8>> {
    let cipher = Aes256Gcm::new(key.into());
    let payload = Payload { msg: ciphertext, aad };
    cipher
        .decrypt(Nonce::from_slice(nonce), payload)
        .map_err(|_| EnvelopeError::Decryption("AES-GCM decrypt failed".into()))
}

// ============================================================================
// One-shot MAC
// ============================================================================

/// Compute a one-shot MAC: `keyed_mac(mac_key, ciphertext)`.
pub fn notification_mac(mac_key: &[u8; 32], ciphertext: &[u8]) -> [u8; 32] {
    keyed_mac(mac_key, ciphertext)
}

/// Verify a one-shot MAC in constant time.
pub fn notification_mac_verify(
    mac_key: &[u8; 32],
    ciphertext: &[u8],
    expected_mac: &[u8; 32],
) -> EnvelopeResult<()> {
    let computed = keyed_mac(mac_key, ciphertext);
    if bool::from(computed.ct_eq(expected_mac)) {
        Ok(())
    } else {
        Err(EnvelopeError::MacVerification)
    }
}

// ============================================================================
// Ristretto DH wrappers (delegate to key_exchange.rs)
// ============================================================================

/// Standard DH from raw bytes. Delegates to `key_exchange::ristretto_dh_raw`.
#[cfg(not(feature = "fips"))]
fn ristretto_dh_raw_impl(secret_bytes: &[u8; 32], their_pub_bytes: &[u8; 32]) -> EnvelopeResult<[u8; 32]> {
    ristretto_dh_raw(secret_bytes, their_pub_bytes)
}

#[cfg(feature = "fips")]
fn ristretto_dh_raw_impl(_secret_bytes: &[u8; 32], _their_pub_bytes: &[u8; 32]) -> EnvelopeResult<[u8; 32]> {
    Err(EnvelopeError::KeyExchange("notification broadcast encryption requires Ristretto255 (not available in FIPS mode)".into()))
}

/// Blinding-aware DH from raw bytes. Delegates to `key_exchange::blinded_dh_raw`.
#[cfg(not(feature = "fips"))]
fn blinded_dh_raw_impl(secret_bytes: &[u8; 32], blinding_scalar: &[u8; 32], their_pub: &[u8; 32]) -> EnvelopeResult<[u8; 32]> {
    blinded_dh_raw(secret_bytes, blinding_scalar, their_pub)
}

#[cfg(feature = "fips")]
fn blinded_dh_raw_impl(_secret_bytes: &[u8; 32], _blinding_scalar: &[u8; 32], _their_pub: &[u8; 32]) -> EnvelopeResult<[u8; 32]> {
    Err(EnvelopeError::KeyExchange("notification broadcast requires Ristretto255".into()))
}

/// Reconstruct blinded pubkey from raw bytes. Delegates to `key_exchange::reconstruct_blinded_pub_raw`.
#[cfg(not(feature = "fips"))]
fn reconstruct_blinded_pub_raw_impl(subscriber_pub: &[u8; 32], blinding_scalar: &[u8; 32]) -> EnvelopeResult<[u8; 32]> {
    reconstruct_blinded_pub_raw(subscriber_pub, blinding_scalar)
}

#[cfg(feature = "fips")]
fn reconstruct_blinded_pub_raw_impl(_subscriber_pub: &[u8; 32], _blinding_scalar: &[u8; 32]) -> EnvelopeResult<[u8; 32]> {
    Err(EnvelopeError::KeyExchange("notification broadcast requires Ristretto255".into()))
}

// ============================================================================
// Per-subscriber capsule
// ============================================================================

/// A wrapped data key + MAC for one subscriber.
pub struct RecipientCapsule {
    /// `Blake3(blinded_pubkey)[..16]` for routing.
    pub fingerprint: PubkeyFingerprint,
    /// `AES-GCM(enc_key, key_nonce, data_key, aad=fingerprint)`.
    pub wrapped_key: Vec<u8>,
    /// Random 12-byte nonce for key wrapping.
    pub key_nonce: [u8; 12],
    /// `keyed_mac(mac_key, ciphertext)` — 32 bytes.
    pub mac: [u8; 32],
}

// ============================================================================
// BroadcastEncryptor (publisher side)
// ============================================================================

/// Encrypts a notification payload for multiple subscribers.
///
/// # Usage
///
/// ```ignore
/// let encryptor = BroadcastEncryptor::new(intent_id, scope);
/// let result = encryptor.encrypt(
///     &publisher_secret,
///     &publisher_pubkey_bytes,
///     &blinded_subscriber_pubkeys,  // from publishIntent response
///     payload,
/// )?;
/// // result.ciphertext, result.nonce, result.capsules → DeliverRequest
/// ```
pub struct BroadcastEncryptor {
    intent_id: String,
    scope: String,
}

/// Result of broadcast encryption.
pub struct BroadcastEncrypted {
    /// AES-GCM ciphertext (shared across all recipients).
    pub ciphertext: Vec<u8>,
    /// AES-GCM nonce for payload (12 bytes, random OsRng).
    pub nonce: [u8; 12],
    /// Per-subscriber capsules (wrapped keys + MACs).
    pub capsules: Vec<RecipientCapsule>,
}

impl BroadcastEncryptor {
    /// Create a new encryptor for the given intent.
    pub fn new(intent_id: String, scope: String) -> Self {
        Self { intent_id, scope }
    }

    /// Encrypt payload for multiple subscribers.
    ///
    /// # Arguments
    ///
    /// * `publisher_secret_bytes` - Publisher's Ristretto secret scalar (32 bytes)
    /// * `publisher_pub` - Publisher's Ristretto public key (32 bytes)
    /// * `blinded_subscriber_pubs` - Blinded subscriber public keys from publishIntent (32 bytes each)
    /// * `plaintext` - Payload to encrypt (e.g., serialized SignedNotificationPayload)
    ///
    /// # Returns
    ///
    /// `BroadcastEncrypted` containing shared ciphertext + per-subscriber capsules.
    pub fn encrypt(
        &self,
        publisher_secret_bytes: &[u8; 32],
        publisher_pub: &[u8; 32],
        blinded_subscriber_pubs: &[[u8; 32]],
        plaintext: &[u8],
    ) -> EnvelopeResult<BroadcastEncrypted> {
        // Generate random data key and nonce
        let data_key = random_data_key();
        let nonce = random_nonce();
        let aad = build_payload_aad(&self.intent_id, &self.scope);

        // Encrypt payload with data_key
        let ciphertext = aes_gcm_encrypt(&data_key, &nonce, plaintext, &aad)?;

        // Build per-subscriber capsules
        let mut capsules = Vec::with_capacity(blinded_subscriber_pubs.len());

        for blinded_pub in blinded_subscriber_pubs {
            // DH: shared_secret = publisher_secret * blinded_pubkey
            let shared_secret = ristretto_dh_raw_impl(publisher_secret_bytes, blinded_pub)?;

            // Derive per-subscriber keys
            let keys = derive_notification_keys(&shared_secret, publisher_pub, blinded_pub)?;

            // Fingerprint for routing
            let fingerprint = pubkey_fingerprint(blinded_pub);

            // Wrap data_key: AES-GCM(enc_key, key_nonce, data_key, aad=fingerprint)
            let key_nonce = random_nonce();
            let wrapped_key = aes_gcm_encrypt(&keys.enc_key, &key_nonce, &*data_key, &fingerprint)?;

            // One-shot MAC over ciphertext
            let mac = notification_mac(&keys.mac_key, &ciphertext);

            capsules.push(RecipientCapsule {
                fingerprint,
                wrapped_key,
                key_nonce,
                mac,
            });
        }

        Ok(BroadcastEncrypted {
            ciphertext,
            nonce,
            capsules,
        })
    }

}

// ============================================================================
// BroadcastDecryptor (subscriber side)
// ============================================================================

/// Decrypts a notification payload received via StreamService.
///
/// # Usage
///
/// ```ignore
/// let decryptor = BroadcastDecryptor::new(
///     &subscriber_secret_bytes,
///     &subscriber_pub_bytes,
/// );
/// let plaintext = decryptor.decrypt(
///     &publisher_pub,
///     &blinding_scalar,
///     &wrapped_key,
///     &key_nonce,
///     &ciphertext,
///     &nonce,
///     &mac,
///     intent_id,
///     scope,
/// )?;
/// ```
pub struct BroadcastDecryptor {
    subscriber_secret: [u8; 32],
    subscriber_pub: [u8; 32],
}

impl BroadcastDecryptor {
    /// Create a new decryptor with the subscriber's keypair.
    pub fn new(subscriber_secret: &[u8; 32], subscriber_pub: &[u8; 32]) -> Self {
        Self {
            subscriber_secret: *subscriber_secret,
            subscriber_pub: *subscriber_pub,
        }
    }

    /// Decrypt a notification block.
    ///
    /// Performs blinding-aware DH, MAC verification, key unwrapping, and payload decryption.
    #[allow(clippy::too_many_arguments)]
    pub fn decrypt(
        &self,
        publisher_pub: &[u8; 32],
        blinding_scalar: &[u8; 32],
        wrapped_key: &[u8],
        key_nonce: &[u8; 12],
        ciphertext: &[u8],
        payload_nonce: &[u8; 12],
        mac: &[u8; 32],
        intent_id: &str,
        scope: &str,
    ) -> EnvelopeResult<Vec<u8>> {
        // Step 1: Blinding-aware DH: shared = (s_sub + r) * P_pub
        let shared_secret = blinded_dh_raw_impl(&self.subscriber_secret, blinding_scalar, publisher_pub)?;

        // Step 2: Reconstruct blinded pubkey = P_sub + r * G
        let blinded_pub = reconstruct_blinded_pub_raw_impl(&self.subscriber_pub, blinding_scalar)?;

        // Step 3: Derive keys
        let keys = derive_notification_keys(&shared_secret, publisher_pub, &blinded_pub)?;

        // Step 4: Verify MAC
        notification_mac_verify(&keys.mac_key, ciphertext, mac)?;

        // Step 5: Unwrap data_key
        let fingerprint = pubkey_fingerprint(&blinded_pub);
        let data_key_bytes = aes_gcm_decrypt(&keys.enc_key, key_nonce, wrapped_key, &fingerprint)?;
        if data_key_bytes.len() != 32 {
            return Err(EnvelopeError::Decryption("unwrapped key wrong length".into()));
        }
        let mut data_key = Zeroizing::new([0u8; 32]);
        data_key.copy_from_slice(&data_key_bytes);

        // Step 6: Decrypt payload
        let aad = build_payload_aad(intent_id, scope);
        let plaintext = aes_gcm_decrypt(&data_key, payload_nonce, ciphertext, &aad)?;

        Ok(plaintext)
    }

}

impl Drop for BroadcastDecryptor {
    fn drop(&mut self) {
        self.subscriber_secret.zeroize();
    }
}

// ============================================================================
// Ed25519 Publisher Attestation
// ============================================================================

/// Build the message signed by the publisher's Ed25519 key for attestation.
///
/// Message format: `ephemeral_pubkey || blinded_sub_pubkey || u32_le(len(scope)) || scope || u32_le(len(intentId)) || intentId`
///
/// Fixed-length pubkeys (32 bytes each) are not length-prefixed.
/// Variable-length `scope` and `intentId` are length-prefixed to prevent
/// concatenation ambiguity (same reasoning as `build_payload_aad`).
pub fn build_attestation_message(
    ephemeral_pubkey: &[u8; 32],
    blinded_sub_pubkey: &[u8; 32],
    scope: &str,
    intent_id: &str,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(64 + 8 + scope.len() + intent_id.len());
    msg.extend_from_slice(ephemeral_pubkey);
    msg.extend_from_slice(blinded_sub_pubkey);
    msg.extend_from_slice(&(scope.len() as u32).to_le_bytes());
    msg.extend_from_slice(scope.as_bytes());
    msg.extend_from_slice(&(intent_id.len() as u32).to_le_bytes());
    msg.extend_from_slice(intent_id.as_bytes());
    msg
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[cfg(not(feature = "fips"))]
    use crate::crypto::key_exchange::{generate_ephemeral_keypair, rerandomize_pubkey};

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_broadcast_encrypt_decrypt_roundtrip() {
        let (pub_secret, pub_pubkey) = generate_ephemeral_keypair();
        let (sub_secret, sub_pubkey) = generate_ephemeral_keypair();

        // NS rerandomizes subscriber pubkey
        let (blinded_pub, r_bytes) = rerandomize_pubkey(&sub_pubkey);

        let intent_id = "test-intent-123";
        let scope = "serve:model:qwen3";
        let payload = b"hello, notification world!";

        // Publisher encrypts
        let encryptor = BroadcastEncryptor::new(intent_id.to_owned(), scope.to_owned());
        let encrypted = encryptor
            .encrypt(
                &pub_secret.scalar().to_bytes(),
                &pub_pubkey.to_bytes(),
                &[blinded_pub.to_bytes()],
                payload,
            )
            .unwrap();

        assert_eq!(encrypted.capsules.len(), 1);

        let capsule = &encrypted.capsules[0];

        // Subscriber decrypts
        let decryptor = BroadcastDecryptor::new(
            &sub_secret.scalar().to_bytes(),
            &sub_pubkey.to_bytes(),
        );
        let decrypted = decryptor
            .decrypt(
                &pub_pubkey.to_bytes(),
                &r_bytes,
                &capsule.wrapped_key,
                &capsule.key_nonce,
                &encrypted.ciphertext,
                &encrypted.nonce,
                &capsule.mac,
                intent_id,
                scope,
            )
            .unwrap();

        assert_eq!(decrypted, payload);
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_wrong_intent_id_rejected() {
        let (pub_secret, pub_pubkey) = generate_ephemeral_keypair();
        let (sub_secret, sub_pubkey) = generate_ephemeral_keypair();
        let (blinded_pub, r_bytes) = rerandomize_pubkey(&sub_pubkey);

        let encryptor = BroadcastEncryptor::new("intent-a".to_owned(), "scope".to_owned());
        let encrypted = encryptor
            .encrypt(
                &pub_secret.scalar().to_bytes(),
                &pub_pubkey.to_bytes(),
                &[blinded_pub.to_bytes()],
                b"secret",
            )
            .unwrap();

        let capsule = &encrypted.capsules[0];
        let decryptor = BroadcastDecryptor::new(
            &sub_secret.scalar().to_bytes(),
            &sub_pubkey.to_bytes(),
        );

        // Wrong intent_id → AAD mismatch → AES-GCM decrypt fails
        let result = decryptor.decrypt(
            &pub_pubkey.to_bytes(),
            &r_bytes,
            &capsule.wrapped_key,
            &capsule.key_nonce,
            &encrypted.ciphertext,
            &encrypted.nonce,
            &capsule.mac,
            "intent-WRONG",
            "scope",
        );
        assert!(result.is_err());
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_tampered_ciphertext_rejected() {
        let (pub_secret, pub_pubkey) = generate_ephemeral_keypair();
        let (sub_secret, sub_pubkey) = generate_ephemeral_keypair();
        let (blinded_pub, r_bytes) = rerandomize_pubkey(&sub_pubkey);

        let encryptor = BroadcastEncryptor::new("intent".to_owned(), "scope".to_owned());
        let mut encrypted = encryptor
            .encrypt(
                &pub_secret.scalar().to_bytes(),
                &pub_pubkey.to_bytes(),
                &[blinded_pub.to_bytes()],
                b"secret",
            )
            .unwrap();

        // Tamper with ciphertext
        encrypted.ciphertext[0] ^= 0xff;

        let capsule = &encrypted.capsules[0];
        let decryptor = BroadcastDecryptor::new(
            &sub_secret.scalar().to_bytes(),
            &sub_pubkey.to_bytes(),
        );

        // MAC verification fails
        let result = decryptor.decrypt(
            &pub_pubkey.to_bytes(),
            &r_bytes,
            &capsule.wrapped_key,
            &capsule.key_nonce,
            &encrypted.ciphertext,
            &encrypted.nonce,
            &capsule.mac,
            "intent",
            "scope",
        );
        assert!(result.is_err());
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_multiple_subscribers() {
        let (pub_secret, pub_pubkey) = generate_ephemeral_keypair();
        let (sub1_secret, sub1_pubkey) = generate_ephemeral_keypair();
        let (sub2_secret, sub2_pubkey) = generate_ephemeral_keypair();

        let (blinded1, r1) = rerandomize_pubkey(&sub1_pubkey);
        let (blinded2, r2) = rerandomize_pubkey(&sub2_pubkey);

        let intent_id = "multi-sub-intent";
        let scope = "serve:model:test";
        let payload = b"broadcast to all";

        let encryptor = BroadcastEncryptor::new(intent_id.to_owned(), scope.to_owned());
        let encrypted = encryptor
            .encrypt(
                &pub_secret.scalar().to_bytes(),
                &pub_pubkey.to_bytes(),
                &[blinded1.to_bytes(), blinded2.to_bytes()],
                payload,
            )
            .unwrap();

        assert_eq!(encrypted.capsules.len(), 2);

        // Subscriber 1 decrypts
        let dec1 = BroadcastDecryptor::new(
            &sub1_secret.scalar().to_bytes(),
            &sub1_pubkey.to_bytes(),
        );
        let plain1 = dec1
            .decrypt(
                &pub_pubkey.to_bytes(),
                &r1,
                &encrypted.capsules[0].wrapped_key,
                &encrypted.capsules[0].key_nonce,
                &encrypted.ciphertext,
                &encrypted.nonce,
                &encrypted.capsules[0].mac,
                intent_id,
                scope,
            )
            .unwrap();
        assert_eq!(plain1, payload);

        // Subscriber 2 decrypts
        let dec2 = BroadcastDecryptor::new(
            &sub2_secret.scalar().to_bytes(),
            &sub2_pubkey.to_bytes(),
        );
        let plain2 = dec2
            .decrypt(
                &pub_pubkey.to_bytes(),
                &r2,
                &encrypted.capsules[1].wrapped_key,
                &encrypted.capsules[1].key_nonce,
                &encrypted.ciphertext,
                &encrypted.nonce,
                &encrypted.capsules[1].mac,
                intent_id,
                scope,
            )
            .unwrap();
        assert_eq!(plain2, payload);
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_rerandomization_unlinkability() {
        let (_sub_secret, sub_pubkey) = generate_ephemeral_keypair();

        // Two rerandomizations of the same pubkey should produce different blinded pubkeys
        let (blinded1, _) = rerandomize_pubkey(&sub_pubkey);
        let (blinded2, _) = rerandomize_pubkey(&sub_pubkey);

        assert_ne!(blinded1.to_bytes(), blinded2.to_bytes());
    }

    #[test]
    fn test_length_prefixed_aad_prevents_ambiguity() {
        // "ab" + "cd" vs "abc" + "d" should produce different AAD
        let aad1 = build_payload_aad("ab", "cd");
        let aad2 = build_payload_aad("abc", "d");
        assert_ne!(aad1, aad2);

        // Same inputs should produce same AAD
        let aad3 = build_payload_aad("ab", "cd");
        assert_eq!(aad1, aad3);
    }

    #[cfg(not(feature = "fips"))]
    #[test]
    fn test_fingerprint_distinct() {
        let (_, pub1) = generate_ephemeral_keypair();
        let (_, pub2) = generate_ephemeral_keypair();

        let fp1 = pubkey_fingerprint(&pub1.to_bytes());
        let fp2 = pubkey_fingerprint(&pub2.to_bytes());

        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_one_shot_mac_verify() {
        let key = [0x42u8; 32];
        let data = b"test ciphertext";

        let mac = notification_mac(&key, data);
        assert!(notification_mac_verify(&key, data, &mac).is_ok());

        // Tampered MAC
        let mut bad_mac = mac;
        bad_mac[0] ^= 0xff;
        assert!(notification_mac_verify(&key, data, &bad_mac).is_err());
    }
}
