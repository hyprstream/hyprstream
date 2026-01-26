//! JWT token implementation with Ed25519 (EdDSA) signatures.
//!
//! This module re-exports JWT functionality from hyprstream-rpc.
//! All JWT encoding/decoding logic is in the core hyprstream-rpc crate.

// Re-export everything from hyprstream-rpc
pub use hyprstream_rpc::auth::{
    decode, decode_unverified, encode, has_valid_prefix, is_admin_token, Claims, JwtError,
    ADMIN_TOKEN_PREFIX, TOKEN_PREFIX,
};
