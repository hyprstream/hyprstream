//! Cryptographic primitives for secure RPC communication.
//!
//! This module provides:
//! - Ed25519 digital signatures for request envelope authentication
//! - Diffie-Hellman key exchange (Ristretto255 or ECDH P-256 in FIPS mode)
//! - Chained MACs for streaming response authentication
//! - Backend abstraction for KDF and MAC operations
//!
//! # Chained MAC
//!
//! Streaming responses use chained MACs where each chunk's MAC depends
//! on the previous chunk's MAC. This provides cryptographic ordering
//! without explicit sequence numbers:
//!
//! ```text
//! mac_0 = MAC(key, request_id_bytes || data_0)
//! mac_n = MAC(key, mac_{n-1} || data_n)
//! ```
//!
//! # Feature Flags
//!
//! - Default: Blake3 for KDF and MAC (~10+ GB/s with SIMD)
//! - `fips`: HKDF-SHA256 + HMAC-SHA256 (NIST SP 800-56C, FIPS 198-1)
//!
//! Key exchange:
//! - Default: Ristretto255 (prime-order group, no cofactor issues)
//! - `fips`: ECDH P-256 for FIPS 140-2 compliance

pub mod backend;
pub mod cose_encrypt;
pub mod cose_sign;
pub mod cose_sign1;
pub mod event_crypto;
pub mod group_key;
pub mod hmac;
pub mod hybrid_kem;
pub mod key_exchange;
pub mod broadcast_primitives;
pub mod pq;
pub mod signing;

/// Runtime crypto mode selecting how envelopes/tokens are signed and verified.
///
/// This REPLACES the old compile-time `pq-hybrid` cargo feature. PQ primitives
/// (ML-DSA-65, ML-KEM-768) are always compiled; the policy chooses whether they
/// are used at runtime.
///
/// - `Hybrid` (default): sign with a COSE composite (EdDSA + ML-DSA-65) and
///   REQUIRE both components to verify.
/// - `Classical`: sign with EdDSA only and accept EdDSA-only signatures. A
///   `Classical` verifier still accepts a `Hybrid`-signed item via its EdDSA
///   component (RFC 7517 skip-unknown interop).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CryptoPolicy {
    /// EdDSA only (classical).
    Classical,
    /// EdDSA + ML-DSA-65 composite (post-quantum hybrid). Default.
    #[default]
    Hybrid,
}

impl CryptoPolicy {
    /// Whether this policy emits/requires the post-quantum (ML-DSA-65) component.
    pub fn uses_pq(self) -> bool {
        matches!(self, CryptoPolicy::Hybrid)
    }
}

pub use backend::{derive_key, keyed_mac, keyed_mac_truncated, keyed_mac_truncated_parts};
pub use hmac::StreamHmacState;
pub use key_exchange::{
    derive_notification_keys, derive_stream_keys, DefaultKeyExchange, KeyExchange,
    NotificationKeys, SharedSecret, StreamKeys,
};
pub use signing::{
    generate_signing_keypair, signing_key_from_bytes, verifying_key_from_bytes, SigningKey,
    VerifyingKey,
};

#[cfg(not(feature = "fips"))]
pub use key_exchange::{
    blinded_dh, blinded_dh_raw, generate_ephemeral_keypair, reconstruct_blinded_pub_raw,
    rerandomize_pubkey, ristretto_dh, ristretto_dh_raw, RistrettoKeyExchange, RistrettoPublic,
    RistrettoSecret,
};

#[cfg(feature = "fips")]
pub use key_exchange::{
    generate_ephemeral_keypair, p256_dh, EcdhP256KeyExchange, P256PublicKey, P256SecretKey,
};
