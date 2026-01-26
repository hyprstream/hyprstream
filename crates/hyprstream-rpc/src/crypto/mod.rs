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
pub mod hmac;
pub mod key_exchange;
pub mod signing;

pub use backend::{derive_key, keyed_mac, keyed_mac_truncated};
pub use hmac::{ChainedStreamHmac, HmacKey, StreamHmac};
pub use key_exchange::{derive_stream_keys, DefaultKeyExchange, KeyExchange, SharedSecret, StreamKeys};
pub use signing::{
    generate_signing_keypair, sign_message, signing_key_from_bytes, verify_message,
    verifying_key_from_bytes, SigningKey, VerifyingKey,
};

#[cfg(not(feature = "fips"))]
pub use key_exchange::{
    generate_ephemeral_keypair, ristretto_dh, RistrettoKeyExchange, RistrettoPublic,
    RistrettoSecret,
};

#[cfg(feature = "fips")]
pub use key_exchange::{
    generate_ephemeral_keypair, p256_dh, EcdhP256KeyExchange, P256PublicKey, P256SecretKey,
};
