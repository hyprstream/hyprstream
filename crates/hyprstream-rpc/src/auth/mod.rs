//! JWT authorization types and utilities.
//!
//! This module provides:
//! - Structured scopes for fine-grained authorization
//! - JWT claims with scope validation
//! - JWT encoding/decoding with Ed25519 signatures
//! - Compile-time scope registration via inventory pattern
//! - JWT key source abstraction for unified key resolution

/// ATProto perimeter gateway: external-DID → assurance binding (#549, first half).
/// Native-only — depends on the admission gate (`did:web`/`did:key` resolution).
#[cfg(not(target_arch = "wasm32"))]
pub mod atproto_perimeter;
pub mod claims;
#[cfg(not(target_arch = "wasm32"))]
pub mod federation;
pub mod jti_blocklist;
pub mod jwt;
/// Native MAC: security labels, lattice, subject contexts, genesis labeling
/// (S1, #567). Platform-independent (no std-only deps) so it compiles for wasm.
pub mod mac;
#[cfg(not(target_arch = "wasm32"))]
pub mod key_source;
#[cfg(not(target_arch = "wasm32"))]
pub mod key_subject_resolver;
pub mod scope;
#[cfg(not(target_arch = "wasm32"))]
pub mod scope_registry;

#[cfg(not(target_arch = "wasm32"))]
pub use atproto_perimeter::{AtprotoPerimeterGateway, Ed25519Vk, EnrolledPeer, EnrollmentStore, IdentityKeys, IdentityResolver, MlDsaVk};
pub use claims::{Claims, Cnf, CnfJwk, IdTokenClaims, OneOrMany, compute_jkt, is_local_iss};
pub use jti_blocklist::{InMemoryJtiBlocklist, JtiBlocklist};
pub use mac::{
    Assurance, Compartment, CompartmentSet, ContentBoundLabel, GenesisMap, GenesisReport,
    LabelError, LabeledObject, Lattice, LatticeCodecError, LatticeDecodeError, LatticeVersion,
    Level, SecurityContext, SecurityLabel, StaticNodeLabel, SubjectContextClaims,
    VerifiedKeyMaterial, MAX_COMPARTMENTS,
};
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
