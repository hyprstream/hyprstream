//! JWT authorization types and utilities.
//!
//! This module provides:
//! - Structured scopes for fine-grained authorization
//! - JWT claims with scope validation
//! - JWT encoding/decoding with Ed25519 signatures
//! - Compile-time scope registration via inventory pattern
//! - JWT key source abstraction for unified key resolution

pub mod claims;
#[cfg(not(target_arch = "wasm32"))]
pub mod federation;
pub mod jti_blocklist;
pub mod jwt;
#[cfg(not(target_arch = "wasm32"))]
pub mod key_source;
#[cfg(not(target_arch = "wasm32"))]
pub mod key_subject_resolver;
pub mod scope;
#[cfg(not(target_arch = "wasm32"))]
pub mod scope_registry;

pub use claims::{Claims, Cnf, CnfJwk, IdTokenClaims, OneOrMany, compute_jkt, is_local_iss};
pub use jti_blocklist::{InMemoryJtiBlocklist, JtiBlocklist};
#[cfg(not(target_arch = "wasm32"))]
pub use federation::FederationKeySource;
pub use jwt::{decode, decode_unverified, decode_with_key, encode, encode_service_jwt, header_alg, header_kid, jwk_thumbprint, JwkThumbprintInput, JwtError};
#[cfg(not(target_arch = "wasm32"))]
pub use key_source::{ClusterKeySource, FederatedKeySource, IssuerResolver, JwksFetcher, JwksKeySource, JwksMode, JwtKeySource};
#[cfg(not(target_arch = "wasm32"))]
pub use key_subject_resolver::{KeySubjectResolver, set_global as set_global_key_subject_resolver};
pub use scope::Scope;
#[cfg(not(target_arch = "wasm32"))]
pub use scope_registry::ScopeDefinition;
