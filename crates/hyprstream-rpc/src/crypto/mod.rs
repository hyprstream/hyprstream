//! Cryptographic primitives for secure RPC communication.
//!
//! This module provides:
//! - Ed25519 digital signatures for request envelope authentication
//! - Diffie-Hellman key exchange (Ristretto255 or ECDH P-256 in FIPS mode)
//! - Chained HMAC-SHA256 for streaming response authentication
//!
//! # Chained HMAC
//!
//! Streaming responses use chained HMACs where each chunk's HMAC depends
//! on the previous chunk's HMAC. This provides cryptographic ordering
//! without explicit sequence numbers:
//!
//! ```text
//! mac_0 = HMAC(key, request_id_bytes || data_0)
//! mac_n = HMAC(key, mac_{n-1} || data_n)
//! ```
//!
//! # Feature Flags
//!
//! - Default: Uses Ristretto255 for key exchange (prime-order group, no cofactor issues)
//! - `fips`: Uses ECDH P-256 for FIPS 140-2 compliance

pub mod hmac;
pub mod key_exchange;
pub mod signing;

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
pub use key_exchange::EcdhP256KeyExchange;
