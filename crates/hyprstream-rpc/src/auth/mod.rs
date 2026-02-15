//! JWT authorization types and utilities.
//!
//! This module provides:
//! - Structured scopes for fine-grained authorization
//! - JWT claims with scope validation
//! - JWT encoding/decoding with Ed25519 signatures
//! - Compile-time scope registration via inventory pattern

pub mod claims;
pub mod jwt;
pub mod scope;
pub mod scope_registry;

pub use claims::Claims;
pub use jwt::{decode, encode, has_valid_prefix, JwtError};
#[cfg(test)]
pub use jwt::decode_unverified;
pub use jwt::TOKEN_PREFIX;
pub use scope::Scope;
pub use scope_registry::ScopeDefinition;
