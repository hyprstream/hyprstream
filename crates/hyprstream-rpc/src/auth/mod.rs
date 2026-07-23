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
pub mod composite;
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
/// UCAN → Casbin/TE compiler foundation (S5, #571): the UCAN token model,
/// delegation-chain + ceiling/attenuation validation, and the signed
/// `(UCAN, bundle_hash)` approval binding (hybrid EdDSA + ML-DSA-65 COSE).
pub mod ucan;

#[cfg(not(target_arch = "wasm32"))]
pub use atproto_perimeter::{AtprotoPerimeterGateway, EnrolledPeer, EnrollmentStore};
// The identity-resolution contract (#579) lives canonically in `crate::identity`;
// re-exported here so existing `crate::auth::{IdentityResolver, ...}` paths keep working.
pub use crate::identity::{Ed25519Vk, IdentityKeys, IdentityResolver, MlDsaVk};
pub use claims::{ActClaim, Claims, Cnf, CnfJwk, IdTokenClaims, OneOrMany, compute_jkt, is_local_iss};
#[cfg(not(target_arch = "wasm32"))]
pub use composite::{
    global_composite_key_set, CompositeKeyPair, CompositeKeySet, CompositeKeySetSnapshot,
    CompositePairRole, CompositePairState,
};
pub use jti_blocklist::{InMemoryJtiBlocklist, JtiBlocklist};
pub use mac::{
    bind_time_label, import_label, Assurance, Compartment, CompartmentSet, ContentBoundLabel,
    GenesisMap, GenesisReport, LabelError, LabeledObject, Lattice, LatticeCodecError,
    LatticeDecodeError, LatticeVersion, Level, ObjectLabelResolver, ObjectRef, SecurityContext,
    SecurityLabel, StaticNodeLabel, SubjectContextClaims, VerifiedKeyMaterial, MAX_COMPARTMENTS,
};
#[cfg(not(target_arch = "wasm32"))]
pub use federation::FederationKeySource;
pub use jwt::{
    composite_kid, decode, decode_unverified, decode_with_candidates, decode_with_key, encode,
    encode_service_jwt, header_alg, header_kid, is_rfc9068_access_token_type, jwk_thumbprint,
    parse_composite_dispatch, parse_protected_header, CompositeJwtDispatch, JwkThumbprintInput,
    JwtError, ProtectedHeader, RFC9068_ACCESS_TOKEN_TYPES,
};
#[cfg(not(target_arch = "wasm32"))]
pub use key_source::{ClusterKeySource, FederatedKeySource, IssuerResolver, JwksFetcher, JwksKeySource, JwksMode, JwtKeySource};
#[cfg(not(target_arch = "wasm32"))]
pub use key_subject_resolver::{KeySubjectResolver, set_global as set_global_key_subject_resolver};
pub use scope::Scope;
#[cfg(not(target_arch = "wasm32"))]
pub use scope_registry::ScopeDefinition;
pub use ucan::{
    set_attenuates, Ability, ApprovalBinding, ApprovalError, Capability, CaveatValue, Caveats,
    ChainError, Did, Resource, SignedApproval, Ucan, UcanError, UcanPayload, UcanVerifier,
};
