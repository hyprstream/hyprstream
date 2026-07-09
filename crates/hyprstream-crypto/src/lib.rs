//! Light, transport-free cryptographic primitives for hyprstream.
//!
//! This crate holds the cleanly-separable generic crypto that used to live in
//! `hyprstream-rpc/src/crypto/`. It carries **no** transport, Cap'n Proto, MoQ,
//! quinn, or iroh dependencies, so crates that need only crypto (`hyprstream-pds`,
//! `hyprstream-discovery`, the wasm surface) can depend on it without pulling the
//! full RPC stack. `hyprstream-rpc` re-exports every module here so existing
//! `hyprstream_rpc::crypto::…` / `hyprstream_rpc::did_key::…` paths keep working.
//!
//! # Modules
//!
//! - [`pq`] — ML-DSA-65 (FIPS 204) signatures and ML-KEM-768 (FIPS 203) KEM.
//! - [`cose_sign`] — COSE composite (EdDSA + ML-DSA-65) sign/verify.
//! - [`cose_sign1`] — COSE_Sign1 (RFC 9052) single-signer envelopes.
//! - [`hmac`] — chained-MAC state for streaming response authentication.
//! - [`backend`] — KDF and MAC backend (Blake3 default, HKDF/HMAC-SHA256 in `fips`).
//! - [`did_key`] — the single canonical `did:key` / Multikey multibase codec and
//!   multicodec constants (`ed25519-pub`, `ml-dsa-65-pub`).
//!
//! # Feature Flags
//!
//! - `fips`: KDF/MAC backend uses HKDF-SHA256 + HMAC-SHA256 (NIST SP 800-56C,
//!   FIPS 198-1) instead of Blake3.

pub mod backend;
pub mod cose_sign;
pub mod cose_sign1;
pub mod did_key;
pub mod hmac;
pub mod pq;

pub use backend::{derive_key, keyed_mac, keyed_mac_truncated, keyed_mac_truncated_parts};
pub use hmac::StreamHmacState;
