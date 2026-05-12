//! Anonymous device enrollment — stateless HMAC challenge and Ed25519 verification.
//!
//! The challenge is `HMAC-SHA256(challenge_key, pubkey)`. It is deterministic:
//! the same pubkey always produces the same challenge. Replay of a valid
//! `(pubkey, signature)` pair is idempotent — it re-enrolls the same device,
//! producing the same `DeviceRecord` upsert. This is analogous to PKCS#10 CSR
//! signing in standard PKI, where the CA validates possession, not freshness.
//!
//! `challenge_key` is derived from the node signing key via HKDF so that:
//! 1. It never overlaps with the JWT signing path.
//! 2. It is safe to use as a challenge oracle (HMAC is secure under chosen-message attacks).

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Derive the challenge key from the node signing key seed.
///
/// Returns the 32-byte HKDF expansion that serves as the HMAC-SHA256 key.
pub fn derive_challenge_key(signing_key_bytes: &[u8; 32]) -> [u8; 32] {
    use hkdf::Hkdf;
    let hk = Hkdf::<Sha256>::new(None, signing_key_bytes);
    let mut okm = [0u8; 32];
    #[allow(clippy::expect_used)]
    hk.expand(b"aegis-device-challenge-v1", &mut okm)
        .expect("HKDF-SHA256 expand to 32 bytes cannot fail");
    okm
}

/// Derive the cookie signing key from the node signing key seed.
///
/// Returns a 32-byte seed suitable for constructing an Ed25519 signing key.
pub fn derive_cookie_key_seed(signing_key_bytes: &[u8; 32]) -> [u8; 32] {
    use hkdf::Hkdf;
    let hk = Hkdf::<Sha256>::new(None, signing_key_bytes);
    let mut okm = [0u8; 32];
    #[allow(clippy::expect_used)]
    hk.expand(b"aegis-device-cookie-v1", &mut okm)
        .expect("HKDF-SHA256 expand to 32 bytes cannot fail");
    okm
}

/// Compute the challenge bytes the client must sign to prove device key possession.
///
/// `HMAC-SHA256(challenge_key, pubkey)` — stateless, deterministic per (key, pubkey) pair.
pub fn device_challenge(challenge_key: &[u8; 32], pubkey: &[u8; 32]) -> [u8; 32] {
    // new_from_slice only fails for keys larger than the block size; a 32-byte key is valid.
    #[allow(clippy::expect_used)]
    let mut mac = HmacSha256::new_from_slice(challenge_key)
        .expect("HMAC-SHA256 accepts 32-byte key");
    mac.update(pubkey);
    mac.finalize().into_bytes().into()
}

/// Verify that `signature` is a valid Ed25519 signature over `HMAC(challenge_key, pubkey)`.
pub fn verify_device_enrollment(
    challenge_key: &[u8; 32],
    pubkey: &[u8; 32],
    signature: &[u8; 64],
) -> bool {
    let challenge = device_challenge(challenge_key, pubkey);
    let Ok(vk) = VerifyingKey::from_bytes(pubkey) else { return false };
    let sig = Signature::from_bytes(signature);
    vk.verify(&challenge, &sig).is_ok()
}

/// Encode a `__Secure-VaultDevice` cookie value.
///
/// Format: `base64url( kid[8 ASCII bytes] || pubkey[32 bytes] || sig[64 bytes] )`
///
/// `kid` is the 8-character hex kid of the cookie signing key.
/// The signed message is `kid_bytes || pubkey`.
pub fn encode_vault_device_cookie(
    cookie_signing_key: &ed25519_dalek::SigningKey,
    kid: &str,
    pubkey: &[u8; 32],
) -> String {
    use ed25519_dalek::Signer;
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};

    assert_eq!(kid.len(), 8, "kid must be exactly 8 ASCII characters");
    let kid_bytes = kid.as_bytes();

    let mut msg = Vec::with_capacity(8 + 32);
    msg.extend_from_slice(kid_bytes);
    msg.extend_from_slice(pubkey);

    let sig = cookie_signing_key.sign(&msg);

    let mut payload = Vec::with_capacity(8 + 32 + 64);
    payload.extend_from_slice(kid_bytes);
    payload.extend_from_slice(pubkey);
    payload.extend_from_slice(&sig.to_bytes());

    URL_SAFE_NO_PAD.encode(&payload)
}

/// Compute the `kid` for a signing key (first 8 hex chars of SHA-256 of verifying key bytes).
pub fn compute_kid(verifying_key_bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(verifying_key_bytes);
    hex::encode(&hash[..8])
}

/// Verify a `__Secure-VaultDevice` cookie and return the device pubkey on success.
///
/// Returns `None` on any malformation or signature failure (including kid mismatch
/// after key rotation — the client should silently re-enroll in that case).
pub fn verify_vault_device_cookie(signing_key_bytes: &[u8; 32], cookie_value: &str) -> Option<[u8; 32]> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use ed25519_dalek::{Signature, Verifier};

    let raw = URL_SAFE_NO_PAD.decode(cookie_value).ok()?;
    if raw.len() != 8 + 32 + 64 {
        return None;
    }
    let kid_bytes = &raw[..8];
    let mut pubkey_arr = [0u8; 32];
    pubkey_arr.copy_from_slice(&raw[8..40]);
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&raw[40..104]);

    let cookie_seed = derive_cookie_key_seed(signing_key_bytes);
    let cookie_sk = ed25519_dalek::SigningKey::from_bytes(&cookie_seed);
    let expected_kid = compute_kid(cookie_sk.verifying_key().as_bytes());

    // kid mismatch → key rotation; caller should signal client to re-enroll
    if kid_bytes != expected_kid.as_bytes() {
        return None;
    }

    let mut msg = Vec::with_capacity(8 + 32);
    msg.extend_from_slice(kid_bytes);
    msg.extend_from_slice(&pubkey_arr);

    let sig = Signature::from_bytes(&sig_arr);
    cookie_sk.verifying_key().verify(&msg, &sig).ok()?;
    Some(pubkey_arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenge_is_deterministic() {
        let key = [0u8; 32];
        let pubkey = [1u8; 32];
        let c1 = device_challenge(&key, &pubkey);
        let c2 = device_challenge(&key, &pubkey);
        assert_eq!(c1, c2);
    }

    #[test]
    fn challenge_differs_by_pubkey() {
        let key = [0u8; 32];
        let c1 = device_challenge(&key, &[1u8; 32]);
        let c2 = device_challenge(&key, &[2u8; 32]);
        assert_ne!(c1, c2);
    }

    #[test]
    fn verify_roundtrip() {
        use ed25519_dalek::{SigningKey, Signer};
        let mut rng = rand::rngs::OsRng;
        let sk = SigningKey::generate(&mut rng);
        let pubkey = sk.verifying_key().to_bytes();
        let challenge_key = derive_challenge_key(&[42u8; 32]);
        let challenge = device_challenge(&challenge_key, &pubkey);
        let sig = sk.sign(&challenge).to_bytes();
        assert!(verify_device_enrollment(&challenge_key, &pubkey, &sig));
    }

    #[test]
    fn verify_bad_sig_fails() {
        let challenge_key = derive_challenge_key(&[0u8; 32]);
        let pubkey = [0u8; 32];
        let bad_sig = [0u8; 64];
        assert!(!verify_device_enrollment(&challenge_key, &pubkey, &bad_sig));
    }
}
