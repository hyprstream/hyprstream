//! JWT authorization types and utilities.
//!
//! This module provides:
//! - Structured scopes for fine-grained authorization
//! - JWT claims with scope validation
//! - JWT encoding/decoding with Ed25519 signatures
//! - Compile-time scope registration via inventory pattern

pub mod claims;
#[cfg(not(target_arch = "wasm32"))]
pub mod federation;
pub mod jwt;
pub mod scope;
#[cfg(not(target_arch = "wasm32"))]
pub mod scope_registry;

pub use claims::{Claims, is_local_iss};
#[cfg(not(target_arch = "wasm32"))]
pub use federation::FederationKeySource;
pub use jwt::{decode, decode_with_key, encode, JwtError};
#[cfg(test)]
pub use jwt::decode_unverified;
pub use scope::Scope;
#[cfg(not(target_arch = "wasm32"))]
pub use scope_registry::ScopeDefinition;
