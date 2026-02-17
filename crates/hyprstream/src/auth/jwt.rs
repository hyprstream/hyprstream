//! JWT token implementation with Ed25519 (EdDSA) signatures.
//!
//! This module re-exports JWT functionality from hyprstream-rpc.
//! All JWT encoding/decoding logic is in the core hyprstream-rpc crate.

// Re-export everything from hyprstream-rpc
pub use hyprstream_rpc::auth::{
    decode, encode, Claims, JwtError,
};
// Note: decode_unverified is available directly in hyprstream_rpc::auth for tests
// within that crate. Cross-crate cfg(test) re-export doesn't work.
