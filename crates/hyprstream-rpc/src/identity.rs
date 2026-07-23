//! Identity provider abstraction for unified key management.
//!
//! Three implementations cover the trust boundaries:
//! - `NodeIdentityProvider` — native server/CLI (file + keyring)
//! - `AegisIdentityProvider` — browser (aegis-vault via JS callback)
//! - `FederatedIdentityProvider` — cross-node (entity statements)
//!
//! HKDF derivation: `HKDF(root_seed, purpose_bytes)` — no prefix.
//! The purpose string IS the domain separator.

use std::fmt;
use std::str::FromStr;

use anyhow::Result;
use async_trait::async_trait;

use crate::auth::mac::Assurance;
use crate::crypto::pq::MlDsaVerifyingKey;
use serde::{Deserialize, Serialize};

use crate::Subject;

/// A W3C Decentralized Identifier (`did:key:…`, `did:web:…`, …), carried on capnp wires as
/// `Text` (W3C DID strings — see <https://www.w3.org/TR/did-core/>) and mapped to this Rust
/// newtype via a field-level `$domainType("hyprstream_rpc::identity::Did")` annotation. The
/// generated ToCapnp/FromCapnp impls write via [`Did::as_str`] and read via [`Did::new`].
///
/// Construction is **lenient** — [`Did::new`] performs no validation, so deserializing a
/// malformed wire value never fails at the boundary. **Strict validation lives at the
/// admission gate** (`crate::admission`); this type is a typed string, not a policy check.
///
/// `did:key` is the self-certifying interop arm (the key *is* the identity; algorithm-agile
/// via multicodec and iroh-convertible), while `did:web` covers operated nodes whose keys
/// live as DID-document verification methods.
///
/// Wire representation is `Text`; serde (de)serializes transparently as the inner string.
/// Cloneable, hashable, orderable (lexicographic on the inner string), and usable as a map
/// key — including ordered maps / `TtlCache<Did, _>` (#524 placement liveness directory).
/// `Default` yields an empty DID (`Did("")`) — the generated capnp data structs that carry a
/// `Did` field derive `Default`, and an absent `Text` field reads back as the empty string.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Did(String);

/// The `did:at9p:` method prefix — the single source of truth for the
/// self-certifying hybrid-PQC capsule identity (#879/#880). A `did:at9p`
/// identifier is this prefix followed by the base32 CIDv1 (`cid512`) of the
/// genesis capsule (#884).
///
/// Shared by the admission gate (`crate::admission`), this method check
/// ([`Did::is_did_at9p`]), and the GATE/capsule code in `hyprstream-pds`
/// (`at9p_gate`) so the literal cannot drift across the three arms (D2/#964).
pub const DID_AT9P_PREFIX: &str = "did:at9p:";

impl Did {
    /// Wrap a DID string verbatim. **No validation** — strict checks belong at the
    /// admission gate (see the type docs).
    pub fn new(did: String) -> Self {
        Did(did)
    }

    /// Borrow the underlying DID string.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Consume the newtype, returning the owned DID string.
    pub fn into_string(self) -> String {
        self.0
    }

    /// Whether this is a `did:key` identifier (the self-certifying interop arm).
    pub fn is_did_key(&self) -> bool {
        // `did_web` is native-only; mirror its canonical check on wasm so the API is
        // uniform across targets. The two implementations are intentionally identical.
        #[cfg(not(target_arch = "wasm32"))]
        {
            crate::did_web::is_did_key(self.0.as_str())
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.0.starts_with("did:key:")
        }
    }

    /// Whether this is a `did:web` identifier (operated nodes resolved via DID document).
    pub fn is_did_web(&self) -> bool {
        self.0.starts_with("did:web:")
    }

    /// Whether this is a `did:at9p` identifier — the PQC-hybrid, self-certifying
    /// identity (`did:at9p:<cid512>`, #879). The authoritative end of the
    /// `alsoKnownAs` aliasing bridge (#896 / D4): when a `did:web` / `did:key`
    /// and a `did:at9p` mutually attest each other, the content-verified at9p
    /// capsule is the stronger identity.
    ///
    /// The capsule GATE pipeline that verifies a `did:at9p` and derives its
    /// `PqHybrid` key material is epic #880 (D2); the method-dispatched resolver
    /// reserves the arm but does not yet resolve it (fails closed until #894).
    pub fn is_did_at9p(&self) -> bool {
        self.0.starts_with(DID_AT9P_PREFIX)
    }

    /// Decode the raw 32-byte Ed25519 public key from a `did:key` identifier.
    ///
    /// Returns `Err` for a non-`did:key` DID, a non-Ed25519 multicodec, or a malformed body.
    pub fn to_ed25519(&self) -> Result<[u8; 32]> {
        crate::did_key::did_key_to_ed25519(self.0.as_str())
    }

    /// Construct a `did:key` (Ed25519) identifier from a raw 32-byte public key.
    pub fn from_ed25519(key: &[u8; 32]) -> Self {
        Did(crate::did_key::ed25519_to_did_key(key))
    }
}

impl fmt::Display for Did {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for Did {
    /// Parsing is infallible — construction is lenient (see [`Did::new`]).
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Did(s.to_owned()))
    }
}

impl From<String> for Did {
    fn from(s: String) -> Self {
        Did(s)
    }
}

impl From<&str> for Did {
    fn from(s: &str) -> Self {
        Did(s.to_owned())
    }
}

/// A purpose-keyed signing identity derived from a root seed.
///
/// Holds a derived Ed25519 keypair for a specific purpose (e.g.,
/// `"hyprstream-rpc-envelope-v1"`). The private key never leaves
/// the identity provider — callers get the pubkey and a sign function.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait SigningIdentity: Send + Sync {
    /// 32-byte Ed25519 public key for this purpose.
    fn pubkey(&self) -> [u8; 32];

    /// Sign canonical bytes. Returns 64-byte Ed25519 signature.
    async fn sign(&self, canonical_bytes: &[u8]) -> Result<[u8; 64]>;
}

/// Unified identity provider — key management + trust resolution.
///
/// Implementations manage key storage, derivation, and peer trust.
/// The `Signer` trait (used by `RpcClientImpl`) adapts over this.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait IdentityProvider: Send + Sync {
    /// Open a purpose-keyed signing identity.
    ///
    /// Derives a unique Ed25519 keypair via `HKDF(root_seed, purpose)`.
    /// Same purpose always produces the same keypair from the same root.
    async fn identity_open(&self, purpose: &str) -> Result<Box<dyn SigningIdentity>>;

    /// Resolve a peer's verified pubkey to a subject (authorization identity).
    ///
    /// Called after Ed25519 signature verification succeeds. Maps the
    /// cryptographically verified signer pubkey to a permission subject.
    fn resolve(&self, pubkey: &[u8; 32]) -> Subject;
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod did_tests {
    use super::*;

    #[test]
    fn new_as_str_into_string_roundtrip() {
        let did = Did::new("did:web:example.com".to_owned());
        assert_eq!(did.as_str(), "did:web:example.com");
        assert_eq!(did.clone().into_string(), "did:web:example.com");
    }

    #[test]
    fn is_did_key_and_is_did_web() {
        let key = Did::new("did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK".to_owned());
        assert!(key.is_did_key());
        assert!(!key.is_did_web());

        let web = Did::new("did:web:example.com".to_owned());
        assert!(web.is_did_web());
        assert!(!web.is_did_key());

        let neither = Did::new("did:plc:abc123".to_owned());
        assert!(!neither.is_did_key());
        assert!(!neither.is_did_web());
        assert!(!neither.is_did_at9p());

        let at9p = Did::new("did:at9p:bafyabcd".to_owned());
        assert!(at9p.is_did_at9p());
        assert!(!at9p.is_did_key());
        assert!(!at9p.is_did_web());
    }

    #[test]
    fn did_key_ed25519_roundtrip() {
        // A fixed 32-byte key → did:key → back to the same bytes.
        let key = [7u8; 32];
        let did = Did::from_ed25519(&key);
        assert!(
            did.is_did_key(),
            "from_ed25519 must produce a did:key: {did}"
        );
        assert_eq!(did.to_ed25519().expect("decode"), key);
    }

    #[test]
    fn to_ed25519_rejects_non_did_key() {
        let web = Did::new("did:web:example.com".to_owned());
        assert!(web.to_ed25519().is_err());
    }

    #[test]
    fn from_str_is_infallible() {
        let did: Did = "did:key:abc".parse().expect("infallible");
        assert_eq!(did.as_str(), "did:key:abc");
    }

    #[test]
    fn display_renders_inner_string() {
        let did = Did::new("did:web:node.example".to_owned());
        assert_eq!(format!("{did}"), "did:web:node.example");
    }

    #[test]
    fn serde_is_transparent_string() {
        let did = Did::new("did:key:z6Mkxyz".to_owned());
        let json = serde_json::to_string(&did).expect("serialize");
        assert_eq!(json, "\"did:key:z6Mkxyz\"");
        let back: Did = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, did);
    }
}

// ============================================================================
// Identity-resolution contract (#579)
//
// The canonical `Did -> verified key material` contract. A method-dispatched
// resolver (did:web VM extractor / did:key single-Ed25519 / did:at9p capsule
// arm, #894) implements `IdentityResolver`; consumers — the ATProto perimeter
// gateway (`auth::atproto_perimeter`) and mesh admission (`admission`) — are
// generic over it and never see key bytes directly. One definition lives here;
// do not re-declare it elsewhere.
// ============================================================================

/// Raw 32-byte Ed25519 verifying key.
pub type Ed25519Vk = [u8; 32];

/// ML-DSA-65 verifying key (FIPS 204), pinned to the always-compiled PQ
/// primitive ([`crate::crypto::pq::MlDsaVerifyingKey`]).
pub type MlDsaVk = MlDsaVerifyingKey;

/// One named verification-key candidate published by an identity.
///
/// A DID document is a key *set*, not a positional one-key container. `id` is
/// the verification-method identifier that names this candidate. A hybrid
/// candidate carries the ML-DSA key explicitly paired with this Ed25519 key;
/// it is never formed by pairing unrelated entries based on document order.
#[derive(Clone)]
pub struct IdentityKeyCandidate {
    /// Stable DID verification-method identifier.
    pub id: String,
    /// The candidate's classical Ed25519 verifying key.
    pub ed25519: Ed25519Vk,
    /// The ML-DSA-65 key explicitly bound to this candidate, when published.
    pub ml_dsa_65: Option<MlDsaVk>,
}

impl IdentityKeyCandidate {
    /// Crypto-derived assurance for this one candidate.
    pub fn assurance(&self) -> Assurance {
        if self.ml_dsa_65.is_some() {
            Assurance::PqHybrid
        } else {
            Assurance::Classical
        }
    }
}

/// The named verification-key candidates a resolver returns for an identity.
///
/// Consumers must select a candidate using a protocol-provided key identifier
/// or verified signer, and must never use document order to choose authority.
/// `did:key` is represented as a one-candidate set because its one-key wire
/// format is intentional and self-certifying.
#[derive(Clone, Default)]
pub struct IdentityKeys {
    pub candidates: Vec<IdentityKeyCandidate>,
}

impl IdentityKeys {
    /// Return the named candidate whose Ed25519 key authenticated the peer.
    pub fn candidate_for_ed25519(&self, key: &Ed25519Vk) -> Option<&IdentityKeyCandidate> {
        let mut matches = self
            .candidates
            .iter()
            .filter(|candidate| &candidate.ed25519 == key);
        let candidate = matches.next()?;
        // Duplicate named candidates for one signer are ambiguous: accepting an
        // arbitrary one could attach a different hybrid binding.
        matches.next().is_none().then_some(candidate)
    }

    /// Construct the intentional single-key `did:key` representation.
    pub fn single_ed25519(id: impl Into<String>, ed25519: Ed25519Vk) -> Self {
        Self {
            candidates: vec![IdentityKeyCandidate {
                id: id.into(),
                ed25519,
                ml_dsa_65: None,
            }],
        }
    }
}

/// Resolves a `Did` to its verified key material.
///
/// **Fail-closed contract:** an implementation MUST return `Err` on any inability
/// to establish key material (network failure, unknown DID, malformed document) —
/// never a default-`Unverified` `Ok`. Consumers treat `Err` as "not enrolled".
pub trait IdentityResolver: Send + Sync {
    /// Resolve `did` to its verified key material, or `Err` (fail-closed).
    fn resolve_identity_keys(&self, did: &Did) -> Result<IdentityKeys>;
}
