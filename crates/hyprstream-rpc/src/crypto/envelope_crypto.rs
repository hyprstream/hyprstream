//! Envelope encryption: X25519 static-ephemeral DH + AES-256-GCM-SIV.
//!
//! Provides encrypt-then-sign envelope protection using the server's Ed25519
//! key converted to X25519 via the birational map (`to_montgomery()`).

use crate::error::{EnvelopeError, EnvelopeResult};
use aes_gcm_siv::aead::{Aead, KeyInit};
use aes_gcm_siv::{Aes256GcmSiv, Nonce};
use curve25519_dalek::montgomery::MontgomeryPoint;
use ed25519_dalek::VerifyingKey;
use zeroize::Zeroizing;

const KDF_CONTEXT: &str = "hyprstream-envelope-v1";
const ZERO_NONCE: [u8; 12] = [0u8; 12];

/// Convert an Ed25519 verifying key to its X25519 (Montgomery) form.
pub fn ed25519_to_x25519_public(vk: &VerifyingKey) -> [u8; 32] {
    vk.to_montgomery().to_bytes()
}

/// Convert an Ed25519 signing key's scalar to X25519 secret key bytes (clamped).
pub fn ed25519_to_x25519_secret(sk: &ed25519_dalek::SigningKey) -> Zeroizing<[u8; 32]> {
    let scalar_bytes = sk.to_scalar_bytes();
    let mut clamped = scalar_bytes;
    clamped[0] &= 248;
    clamped[31] &= 127;
    clamped[31] |= 64;
    Zeroizing::new(clamped)
}

/// Perform X25519 Diffie-Hellman: secret * point.
fn x25519_dh(secret: &[u8; 32], public: &[u8; 32]) -> Zeroizing<[u8; 32]> {
    let point = MontgomeryPoint(*public);
    let scalar = curve25519_dalek::scalar::Scalar::from_bytes_mod_order(*secret);
    let shared = scalar * point;
    Zeroizing::new(shared.to_bytes())
}

/// Derive the AES-256-GCM-SIV key from shared secret + context.
fn derive_envelope_key(
    shared_secret: &[u8; 32],
    ephemeral_public: &[u8; 32],
    server_x25519_public: &[u8; 32],
) -> Zeroizing<[u8; 32]> {
    let mut ikm = Vec::with_capacity(96);
    ikm.extend_from_slice(shared_secret);
    ikm.extend_from_slice(ephemeral_public);
    ikm.extend_from_slice(server_x25519_public);
    Zeroizing::new(blake3::derive_key(KDF_CONTEXT, &ikm))
}

/// Client-side: encrypt a serialized RequestEnvelope.
///
/// Returns `(ciphertext, ephemeral_public)`.
pub fn encrypt_envelope(
    plaintext: &[u8],
    server_ed25519_pubkey: &VerifyingKey,
) -> EnvelopeResult<(Vec<u8>, [u8; 32])> {
    let server_x25519 = ed25519_to_x25519_public(server_ed25519_pubkey);

    let mut csprng = rand::rngs::OsRng;
    let eph_secret = ed25519_dalek::SigningKey::generate(&mut csprng);
    let eph_public_ed = eph_secret.verifying_key();
    let eph_x25519_secret = ed25519_to_x25519_secret(&eph_secret);
    let eph_x25519_public = ed25519_to_x25519_public(&eph_public_ed);

    let shared = x25519_dh(&eph_x25519_secret, &server_x25519);
    let key = derive_envelope_key(&shared, &eph_x25519_public, &server_x25519);

    let cipher = Aes256GcmSiv::new((&*key).into());
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&ZERO_NONCE), plaintext)
        .map_err(|_| EnvelopeError::Encryption("AES-256-GCM-SIV encrypt failed".into()))?;

    Ok((ciphertext, eph_x25519_public))
}

/// Server-side: decrypt an encrypted envelope.
///
/// `server_signing_key` is the Ed25519 signing key (converted to X25519 internally).
pub fn decrypt_envelope(
    ciphertext: &[u8],
    client_ephemeral_public: &[u8; 32],
    server_signing_key: &ed25519_dalek::SigningKey,
) -> EnvelopeResult<Vec<u8>> {
    let server_x25519_secret = ed25519_to_x25519_secret(server_signing_key);
    let server_x25519_public = ed25519_to_x25519_public(&server_signing_key.verifying_key());

    let shared = x25519_dh(&server_x25519_secret, client_ephemeral_public);
    let key = derive_envelope_key(&shared, client_ephemeral_public, &server_x25519_public);

    let cipher = Aes256GcmSiv::new((&*key).into());
    cipher
        .decrypt(Nonce::from_slice(&ZERO_NONCE), ciphertext)
        .map_err(|_| EnvelopeError::Decryption("AES-256-GCM-SIV decrypt failed".into()))
}

/// Hybrid envelope encryption: X25519 DH + ML-KEM-768 + AES-256-GCM-SIV.
///
/// Two shared secrets are derived independently and combined via KDF.
/// Returns `(ciphertext, ephemeral_public, kem_ciphertext)`.
pub fn encrypt_envelope_hybrid(
    plaintext: &[u8],
    server_ed25519_pubkey: &VerifyingKey,
    server_kem_ek: &crate::crypto::pq::MlKemEncapsKey,
) -> EnvelopeResult<(Vec<u8>, [u8; 32], Vec<u8>)> {
    let server_x25519 = ed25519_to_x25519_public(server_ed25519_pubkey);

    let mut csprng = rand::rngs::OsRng;
    let eph_secret = ed25519_dalek::SigningKey::generate(&mut csprng);
    let eph_public_ed = eph_secret.verifying_key();
    let eph_x25519_secret = ed25519_to_x25519_secret(&eph_secret);
    let eph_x25519_public = ed25519_to_x25519_public(&eph_public_ed);

    let shared_x25519 = x25519_dh(&eph_x25519_secret, &server_x25519);
    let (kem_ct, shared_kem) = crate::crypto::pq::ml_kem_encapsulate(server_kem_ek);

    let key = derive_hybrid_envelope_key(
        &shared_x25519,
        &shared_kem,
        &eph_x25519_public,
        &server_x25519,
    );

    let cipher = Aes256GcmSiv::new((&*key).into());
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&ZERO_NONCE), plaintext)
        .map_err(|_| EnvelopeError::Encryption("AES-256-GCM-SIV hybrid encrypt failed".into()))?;

    Ok((ciphertext, eph_x25519_public, kem_ct))
}

/// Hybrid envelope decryption: X25519 DH + ML-KEM-768 + AES-256-GCM-SIV.
pub fn decrypt_envelope_hybrid(
    ciphertext: &[u8],
    client_ephemeral_public: &[u8; 32],
    kem_ciphertext: &[u8],
    server_signing_key: &ed25519_dalek::SigningKey,
    server_kem_dk: &crate::crypto::pq::MlKemDecapsKey,
) -> EnvelopeResult<Vec<u8>> {
    let server_x25519_secret = ed25519_to_x25519_secret(server_signing_key);
    let server_x25519_public = ed25519_to_x25519_public(&server_signing_key.verifying_key());

    let shared_x25519 = x25519_dh(&server_x25519_secret, client_ephemeral_public);
    let shared_kem = crate::crypto::pq::ml_kem_decapsulate(server_kem_dk, kem_ciphertext)
        .map_err(|e| EnvelopeError::Decryption(format!("ML-KEM-768 decapsulate failed: {e}")))?;

    let key = derive_hybrid_envelope_key(
        &shared_x25519,
        &shared_kem,
        client_ephemeral_public,
        &server_x25519_public,
    );

    let cipher = Aes256GcmSiv::new((&*key).into());
    cipher
        .decrypt(Nonce::from_slice(&ZERO_NONCE), ciphertext)
        .map_err(|_| EnvelopeError::Decryption("AES-256-GCM-SIV hybrid decrypt failed".into()))
}

/// Derive AES-256 key from combined X25519 + ML-KEM shared secrets.
fn derive_hybrid_envelope_key(
    shared_x25519: &[u8; 32],
    shared_kem: &[u8; 32],
    ephemeral_public: &[u8; 32],
    server_x25519_public: &[u8; 32],
) -> Zeroizing<[u8; 32]> {
    use zeroize::Zeroize;
    let mut ikm = Vec::with_capacity(128);
    ikm.extend_from_slice(shared_x25519);
    ikm.extend_from_slice(shared_kem);
    ikm.extend_from_slice(ephemeral_public);
    ikm.extend_from_slice(server_x25519_public);
    let key = blake3::derive_key("hyprstream-hybrid-envelope-v1", &ikm);
    ikm.zeroize();
    Zeroizing::new(key)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn roundtrip() {
        let server_key = SigningKey::generate(&mut OsRng);
        let plaintext = b"test envelope payload for encryption roundtrip";

        let (ciphertext, eph_pub) =
            encrypt_envelope(plaintext, &server_key.verifying_key()).unwrap();

        assert_ne!(&ciphertext[..], plaintext.as_slice());

        let decrypted = decrypt_envelope(&ciphertext, &eph_pub, &server_key).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn wrong_server_key_fails() {
        let server_key = SigningKey::generate(&mut OsRng);
        let wrong_key = SigningKey::generate(&mut OsRng);
        let plaintext = b"secret data";

        let (ciphertext, eph_pub) =
            encrypt_envelope(plaintext, &server_key.verifying_key()).unwrap();

        let result = decrypt_envelope(&ciphertext, &eph_pub, &wrong_key);
        assert!(result.is_err());
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let server_key = SigningKey::generate(&mut OsRng);
        let plaintext = b"tamper test";

        let (mut ciphertext, eph_pub) =
            encrypt_envelope(plaintext, &server_key.verifying_key()).unwrap();

        ciphertext[0] ^= 0xFF;

        let result = decrypt_envelope(&ciphertext, &eph_pub, &server_key);
        assert!(result.is_err());
    }

    #[test]
    fn different_ephemeral_keys_produce_different_ciphertext() {
        let server_key = SigningKey::generate(&mut OsRng);
        let plaintext = b"same plaintext";

        let (ct1, _) = encrypt_envelope(plaintext, &server_key.verifying_key()).unwrap();
        let (ct2, _) = encrypt_envelope(plaintext, &server_key.verifying_key()).unwrap();

        assert_ne!(ct1, ct2);
    }

    #[test]
    fn hybrid_roundtrip() {
        let server_key = SigningKey::generate(&mut OsRng);
        let (kem_dk, kem_ek) = crate::crypto::pq::ml_kem_generate_keypair();
        let plaintext = b"hybrid envelope payload";

        let (ciphertext, eph_pub, kem_ct) =
            encrypt_envelope_hybrid(plaintext, &server_key.verifying_key(), &kem_ek).unwrap();

        assert_ne!(&ciphertext[..], plaintext.as_slice());
        assert_eq!(kem_ct.len(), 1088);

        let decrypted =
            decrypt_envelope_hybrid(&ciphertext, &eph_pub, &kem_ct, &server_key, &kem_dk)
                .unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn hybrid_wrong_kem_key_fails() {
        let server_key = SigningKey::generate(&mut OsRng);
        let (_kem_dk, kem_ek) = crate::crypto::pq::ml_kem_generate_keypair();
        let (wrong_dk, _) = crate::crypto::pq::ml_kem_generate_keypair();
        let plaintext = b"wrong key test";

        let (ciphertext, eph_pub, kem_ct) =
            encrypt_envelope_hybrid(plaintext, &server_key.verifying_key(), &kem_ek).unwrap();

        let result =
            decrypt_envelope_hybrid(&ciphertext, &eph_pub, &kem_ct, &server_key, &wrong_dk);
        assert!(result.is_err());
    }
}
