//! Post-quantum hybrid cryptographic primitives (ML-DSA-65 + ML-KEM-768).
//!
//! Always compiled (M3 #152). Whether these are USED is a runtime decision via
//! [`crate::crypto::CryptoPolicy`], not a compile-time cargo feature. Provides:
//! - ML-DSA-65 (FIPS 204) digital signatures
//! - ML-KEM-768 (FIPS 203) key encapsulation

use anyhow::Result;
use ml_dsa::{
    EncodedVerifyingKey, Generate, KeyExport, Keypair, MlDsa65, SignatureEncoding, Signer, Verifier,
};
use ml_kem::{
    kem::{Decapsulate, Encapsulate, Kem, TryKeyInit},
    MlKem768,
};

// ── ML-DSA-65 type aliases ──────────────────────────────────────────────────

pub type MlDsaSigningKey = ml_dsa::SigningKey<MlDsa65>;
pub type MlDsaVerifyingKey = ml_dsa::VerifyingKey<MlDsa65>;
pub type MlDsaSignature = ml_dsa::Signature<MlDsa65>;

// ── ML-KEM-768 type aliases ─────────────────────────────────────────────────

pub type MlKemDecapsKey = ml_kem::DecapsulationKey<MlKem768>;
pub type MlKemEncapsKey = ml_kem::EncapsulationKey<MlKem768>;

// ── ML-DSA-65 operations ────────────────────────────────────────────────────

pub fn ml_dsa_generate_keypair() -> (MlDsaSigningKey, MlDsaVerifyingKey) {
    let sk = MlDsaSigningKey::generate();
    let vk = sk.verifying_key().clone();
    (sk, vk)
}

pub fn ml_dsa_sign(key: &MlDsaSigningKey, message: &[u8]) -> Vec<u8> {
    let sig = key.sign(message);
    sig.to_vec()
}

pub fn ml_dsa_verify(key: &MlDsaVerifyingKey, message: &[u8], sig: &[u8]) -> Result<()> {
    let signature = MlDsaSignature::try_from(sig)
        .map_err(|_| anyhow::anyhow!("invalid ML-DSA-65 signature encoding"))?;
    key.verify(message, &signature)
        .map_err(|_| anyhow::anyhow!("ML-DSA-65 signature verification failed"))
}

pub fn ml_dsa_vk_bytes(key: &MlDsaVerifyingKey) -> Vec<u8> {
    key.to_bytes().to_vec()
}

/// Derive the raw ML-DSA-65 verifying-key bytes (1952 bytes) from a signing key.
///
/// Convenience for callers outside this crate that hold a signing key but do not
/// depend on the `ml_dsa` crate's `Keypair` trait directly.
pub fn ml_dsa_sk_to_vk_bytes(key: &MlDsaSigningKey) -> Vec<u8> {
    ml_dsa_vk_bytes(&key.verifying_key())
}

pub fn ml_dsa_vk_from_bytes(bytes: &[u8]) -> Result<MlDsaVerifyingKey> {
    let encoded = EncodedVerifyingKey::<MlDsa65>::try_from(bytes).map_err(|_| {
        anyhow::anyhow!("invalid ML-DSA-65 verifying key length (expected 1952 bytes)")
    })?;
    Ok(MlDsaVerifyingKey::decode(&encoded))
}

/// Serialize an ML-DSA-65 signing key to its 32-byte seed.
pub fn ml_dsa_sk_to_seed(key: &MlDsaSigningKey) -> [u8; 32] {
    let seed = key.to_seed();
    let mut out = [0u8; 32];
    out.copy_from_slice(seed.as_slice());
    out
}

/// Reconstruct an ML-DSA-65 signing key from its 32-byte seed.
pub fn ml_dsa_sk_from_seed(seed: &[u8; 32]) -> MlDsaSigningKey {
    let arr = ml_dsa::B32::from(*seed);
    MlDsaSigningKey::from_seed(&arr)
}

// ── ML-KEM-768 operations ───────────────────────────────────────────────────

pub fn ml_kem_generate_keypair() -> (MlKemDecapsKey, MlKemEncapsKey) {
    MlKem768::generate_keypair()
}

/// Encapsulate a shared secret. Returns `(ciphertext, shared_secret)`.
pub fn ml_kem_encapsulate(ek: &MlKemEncapsKey) -> (Vec<u8>, [u8; 32]) {
    let (ct, ss): (
        ml_kem::Ciphertext<MlKem768>,
        ml_kem::kem::SharedKey<MlKem768>,
    ) = ek.encapsulate();
    let mut shared = [0u8; 32];
    shared.copy_from_slice(ss.as_slice());
    (ct.as_slice().to_vec(), shared)
}

pub fn ml_kem_decapsulate(dk: &MlKemDecapsKey, ct: &[u8]) -> Result<[u8; 32]> {
    let ss = dk.decapsulate_slice(ct).map_err(|_| {
        anyhow::anyhow!("invalid ML-KEM-768 ciphertext length (expected 1088 bytes)")
    })?;
    let mut shared = [0u8; 32];
    shared.copy_from_slice(ss.as_slice());
    Ok(shared)
}

pub fn ml_kem_ek_bytes(key: &MlKemEncapsKey) -> Vec<u8> {
    ml_kem::kem::KeyExport::to_bytes(key).to_vec()
}

/// Reconstruct an ML-KEM-768 decapsulation key from its 64-byte FIPS 203 seed
/// (`d ‖ z`). Deterministic — the matching encapsulation key is recovered via
/// [`MlKemDecapsKey::encapsulation_key`]. This 64-byte seed is the canonical byte
/// form used by the hybrid-KEM component (#551) and by `#mesh-kem` key derivation
/// (#552), mirroring [`ml_dsa_sk_from_seed`].
pub fn ml_kem_decaps_from_seed(seed: &[u8; 64]) -> MlKemDecapsKey {
    let s = ml_kem::Seed::from(*seed);
    MlKemDecapsKey::from_seed(s)
}

/// Serialize an ML-KEM-768 decapsulation key to its 64-byte seed.
///
/// Returns `None` only if the key was built from the (deprecated) expanded form
/// rather than a seed; keys from [`ml_kem_generate_keypair`] or
/// [`ml_kem_decaps_from_seed`] are always seed-backed.
pub fn ml_kem_dk_to_seed(dk: &MlKemDecapsKey) -> Option<[u8; 64]> {
    dk.to_seed().map(|s| {
        let mut out = [0u8; 64];
        out.copy_from_slice(s.as_ref());
        out
    })
}

/// The encapsulation-key bytes (1184 B) corresponding to a decapsulation key.
pub fn ml_kem_ek_of_dk(dk: &MlKemDecapsKey) -> Vec<u8> {
    ml_kem_ek_bytes(dk.encapsulation_key())
}

pub fn ml_kem_ek_from_bytes(bytes: &[u8]) -> Result<MlKemEncapsKey> {
    let key_array = ml_kem::kem::Key::<MlKemEncapsKey>::try_from(bytes).map_err(|_| {
        anyhow::anyhow!("invalid ML-KEM-768 encapsulation key length (expected 1184 bytes)")
    })?;
    <MlKemEncapsKey as TryKeyInit>::new(&key_array)
        .map_err(|_| anyhow::anyhow!("invalid ML-KEM-768 encapsulation key"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn ml_dsa_sign_verify_roundtrip() {
        let (sk, vk) = ml_dsa_generate_keypair();
        let msg = b"hyprstream post-quantum test";
        let sig = ml_dsa_sign(&sk, msg);
        assert_eq!(sig.len(), 3309);
        ml_dsa_verify(&vk, msg, &sig).unwrap();
    }

    #[test]
    fn ml_dsa_wrong_key_rejects() {
        let (sk, _vk) = ml_dsa_generate_keypair();
        let (_sk2, vk2) = ml_dsa_generate_keypair();
        let msg = b"signed by key 1";
        let sig = ml_dsa_sign(&sk, msg);
        assert!(ml_dsa_verify(&vk2, msg, &sig).is_err());
    }

    #[test]
    fn ml_dsa_tampered_sig_rejects() {
        let (sk, vk) = ml_dsa_generate_keypair();
        let msg = b"original message";
        let sig = ml_dsa_sign(&sk, msg);
        assert!(ml_dsa_verify(&vk, b"tampered message", &sig).is_err());
    }

    #[test]
    fn ml_dsa_vk_serialization_roundtrip() {
        let (_sk, vk) = ml_dsa_generate_keypair();
        let bytes = ml_dsa_vk_bytes(&vk);
        assert_eq!(bytes.len(), 1952);
        let vk2 = ml_dsa_vk_from_bytes(&bytes).unwrap();
        assert_eq!(ml_dsa_vk_bytes(&vk2), bytes);
    }

    #[test]
    fn ml_dsa_seed_roundtrip() {
        let (sk, vk) = ml_dsa_generate_keypair();
        let seed = ml_dsa_sk_to_seed(&sk);
        assert_eq!(seed.len(), 32);
        let sk2 = ml_dsa_sk_from_seed(&seed);
        let vk2 = sk2.verifying_key().clone();
        assert_eq!(ml_dsa_vk_bytes(&vk), ml_dsa_vk_bytes(&vk2));
        // Sign with restored key, verify with original vk
        let msg = b"seed roundtrip test";
        let sig = ml_dsa_sign(&sk2, msg);
        ml_dsa_verify(&vk, msg, &sig).unwrap();
    }

    #[test]
    fn ml_kem_encapsulate_decapsulate_roundtrip() {
        let (dk, ek) = ml_kem_generate_keypair();
        let (ct, k_send) = ml_kem_encapsulate(&ek);
        assert_eq!(ct.len(), 1088);
        let k_recv = ml_kem_decapsulate(&dk, &ct).unwrap();
        assert_eq!(k_send, k_recv);
    }

    #[test]
    fn ml_kem_wrong_key_rejects() {
        let (_dk, ek) = ml_kem_generate_keypair();
        let (dk2, _ek2) = ml_kem_generate_keypair();
        let (ct, k_send) = ml_kem_encapsulate(&ek);
        // ML-KEM decapsulate always succeeds but produces a different shared secret
        let k_recv = ml_kem_decapsulate(&dk2, &ct).unwrap();
        assert_ne!(k_send, k_recv);
    }

    #[test]
    fn ml_kem_ek_serialization_roundtrip() {
        let (_dk, ek) = ml_kem_generate_keypair();
        let bytes = ml_kem_ek_bytes(&ek);
        assert_eq!(bytes.len(), 1184);
        let ek2 = ml_kem_ek_from_bytes(&bytes).unwrap();
        assert_eq!(ml_kem_ek_bytes(&ek2), bytes);
    }
}
