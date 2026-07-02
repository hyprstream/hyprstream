//! The typed **UCAN token** model + structural validation + hybrid-COSE
//! signature verification (S5 / #571, milestone 1).
//!
//! We control both ends of this UCAN — issuance and consumption are internal to
//! the MAC epic — so this is a **minimal, typed** model rather than a full
//! JOSE/JWT UCAN. Crucially, UCAN signing is aligned with the project's hybrid
//! post-quantum COSE composite (EdDSA + ML-DSA-65), **not** a JWT `ucan` crate:
//! a UCAN is a CBOR payload signed with [`crate::crypto::cose_sign::sign_composite`]
//! and verified with [`crate::crypto::cose_sign::verify_composite`]
//! (`require_pq = true`). This keeps the whole authority chain on the same
//! HNDL-resistant signature suite the rest of the TCB uses.
//!
//! ## Shape
//!
//! ```text
//! Ucan {
//!   payload: UcanPayload {        // the signed bytes (canonical CBOR)
//!     issuer:   Did,              // did:key of the signer
//!     audience: Did,              // did:key of the delegate
//!     capabilities: [Capability], // the granted authority
//!     not_before / expiration: Option<u64> (unix seconds),
//!     nonce: ...,
//!   },
//!   proofs: [Ucan],               // the delegation chain (delegator UCANs)
//!   signature: Vec<u8>,           // detached hybrid-COSE composite over payload
//! }
//! ```
//!
//! Signature *verification* lives here. Delegation-chain + attenuation
//! validation lives in [`super::chain`] (it is the most security-critical piece
//! and is isolated for that reason).

use super::capability::Capability;
use serde::{Deserialize, Serialize};
use std::fmt;

/// The canonical decentralized identifier — there is exactly **one** `did:key`
/// type and parser in the TCB (#578). S5 milestone 1 originally carried a local
/// `Did(String)` + its own `did_key_to_ed25519` call; milestone 2 (#571) folds it
/// onto [`crate::identity::Did`] so no second DID type or parser survives. The
/// UCAN layer still supports `did:key` (Ed25519) exclusively — the Ed25519 key is
/// resolved on demand via [`crate::identity::Did::to_ed25519`] (which delegates to
/// the SAME `crate::did_key` parser the rest of `hyprstream-rpc` uses).
pub use crate::identity::Did;

/// Domain-separation AAD for UCAN payload signatures (distinct from the
/// compiled-policy, approval, audit, and envelope AADs so a UCAN signature can
/// never be replayed as another message type). Signed and verified by every
/// producer/consumer of UCANs — the test helpers, the chain validator, the
/// production `UcanVerifier` (the HTTP grant path's trust-store-backed
/// verifier), and cross-process consumers. Promoted to a public non-test
/// constant by S8 (#574) so the production verifier shares the exact AAD the
/// signers use (a mismatch silently fails signature verification).
pub const UCAN_AAD: &[u8] = b"hs-mac-ucan-payload-v1";

/// Resolve the Ed25519 verifying-key bytes a UCAN `did:key` issuer/audience
/// encodes, mapped to a fail-closed [`UcanError`]. Thin adapter over the canonical
/// [`Did::to_ed25519`] so the UCAN signature/structure paths keep their typed
/// error without re-implementing any `did:key` parsing (single parser, #578).
fn did_ed25519_key(did: &Did) -> Result<[u8; 32], UcanError> {
    did.to_ed25519().map_err(|e| UcanError::BadDid {
        did: did.as_str().to_owned(),
        reason: e.to_string(),
    })
}

/// The signed body of a UCAN: everything an issuer commits to with their
/// signature. Serialized to canonical CBOR ([`UcanPayload::signing_bytes`]) to
/// produce the exact bytes the hybrid-COSE composite signs/verifies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UcanPayload {
    /// The issuer (signer) DID.
    pub issuer: Did,
    /// The audience (delegate) DID — who this UCAN grants authority to.
    pub audience: Did,
    /// The capabilities granted by this UCAN.
    pub capabilities: Vec<Capability>,
    /// `not before` — earliest unix second at which this UCAN is valid.
    #[serde(default)]
    pub not_before: Option<u64>,
    /// `expiration` — unix second after which this UCAN is invalid. `None` =
    /// non-expiring (discouraged but representable).
    #[serde(default)]
    pub expiration: Option<u64>,
    /// Replay-uniqueness nonce.
    #[serde(default)]
    pub nonce: Vec<u8>,
}

impl UcanPayload {
    /// Canonical CBOR bytes of the payload — the exact bytes that are signed and
    /// verified. Deterministic for a given payload (ciborium emits map keys in
    /// struct-field order; the same payload always yields identical bytes), so a
    /// signature over these bytes is stable across processes.
    ///
    /// Serializing an in-memory payload into a `Vec` is infallible for this
    /// shape; we still avoid `unwrap`/`expect` in non-test code and surface any
    /// (unreachable) encoder error as a fail-closed [`UcanError`].
    pub fn signing_bytes(&self) -> Result<Vec<u8>, UcanError> {
        let mut buf = Vec::new();
        ciborium::ser::into_writer(self, &mut buf).map_err(|e| UcanError::Encode(e.to_string()))?;
        Ok(buf)
    }
}

/// A UCAN: a signed payload plus the proof chain of delegator UCANs.
///
/// The chain is the `proofs` vector — each entry is the *delegator's* UCAN whose
/// audience is this UCAN's issuer and whose capabilities must cover this UCAN's
/// (validated in [`super::chain`]). A root UCAN (self-issued authority) has an
/// empty `proofs` vector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ucan {
    /// The signed claims.
    pub payload: UcanPayload,
    /// The delegation chain: delegator UCANs that authorize this one. Empty for
    /// a root.
    #[serde(default)]
    pub proofs: Vec<Ucan>,
    /// Detached hybrid-COSE composite signature over [`UcanPayload::signing_bytes`].
    pub signature: Vec<u8>,
}

impl Ucan {
    /// Decode a UCAN from canonical CBOR. Pure structural decode — does NOT
    /// verify signatures or attenuation (call [`Ucan::verify_signatures`] then
    /// [`super::chain::validate_chain`]).
    ///
    /// Fail-closed DoS guard: a decoded chain whose nesting meets or exceeds
    /// [`super::chain::MAX_PROOF_DEPTH`] is rejected here, before it is ever
    /// walked, so a maliciously deep `proofs`-within-`proofs` token cannot reach
    /// (or exhaust the stack in) the validator. (The validator independently
    /// re-enforces the same bound during the walk.)
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, UcanError> {
        let ucan: Ucan =
            ciborium::de::from_reader(bytes).map_err(|e| UcanError::Decode(e.to_string()))?;
        // Match the validator's bound exactly: the walk rejects once a link index
        // reaches `MAX_PROOF_DEPTH` (links `0..MAX_PROOF_DEPTH` are allowed), so a
        // `proof_depth` of `MAX_PROOF_DEPTH` or more is over-deep.
        let depth = ucan.proof_depth();
        if depth >= super::chain::MAX_PROOF_DEPTH {
            return Err(UcanError::Malformed(format!(
                "delegation chain nested {depth} deep meets/exceeds maximum proof depth {}",
                super::chain::MAX_PROOF_DEPTH
            )));
        }
        Ok(ucan)
    }

    /// The maximum nesting depth of this UCAN's proof chain: `0` for a root (no
    /// proofs), otherwise `1 + max(proof.proof_depth())`. Used by [`from_cbor`]
    /// as a parse-time DoS bound. Walks the in-memory tree iteratively-bounded by
    /// the already-decoded structure (the decode itself is what we cap).
    ///
    /// [`from_cbor`]: Ucan::from_cbor
    pub fn proof_depth(&self) -> usize {
        self.proofs
            .iter()
            .map(|p| 1 + p.proof_depth())
            .max()
            .unwrap_or(0)
    }

    /// Encode this UCAN to canonical CBOR.
    pub fn to_cbor(&self) -> Result<Vec<u8>, UcanError> {
        let mut buf = Vec::new();
        ciborium::ser::into_writer(self, &mut buf).map_err(|e| UcanError::Encode(e.to_string()))?;
        Ok(buf)
    }

    /// The issuer DID of this UCAN.
    pub fn issuer(&self) -> &Did {
        &self.payload.issuer
    }

    /// The audience DID of this UCAN.
    pub fn audience(&self) -> &Did {
        &self.payload.audience
    }

    /// The capabilities this UCAN grants.
    pub fn capabilities(&self) -> &[Capability] {
        &self.payload.capabilities
    }

    /// The **root** of this delegation chain — the self-issued authority at the
    /// top, the entry with an empty `proofs` vector. For a delegated grant this
    /// is the resource-owning delegator; its [`issuer`] is the source of
    /// authority (#680/#681: the `sub`/delegator principal, distinct from the
    /// leaf [`audience`] = the actor/presenter).
    ///
    /// M1 walks the single-delegator-per-link chain via `proofs[0]` (the
    /// linkage `super::chain::validate` also enforces). Depth is bounded by
    /// [`super::chain::MAX_PROOF_DEPTH`] at decode, so this cannot runaway.
    ///
    /// [`issuer`]: Self::issuer
    /// [`audience`]: Self::audience
    #[must_use]
    pub fn root(&self) -> &Ucan {
        let mut cur = self;
        while let Some(delegator) = cur.proofs.first() {
            cur = delegator;
        }
        cur
    }

    /// The DID of the delegation-chain root's issuer — the ultimate source of
    /// authority (the resource-owning delegator). Convenience over
    /// [`Self::root`]`().issuer()`. See [`Self::root`].
    #[must_use]
    pub fn root_issuer(&self) -> &Did {
        self.root().issuer()
    }

    /// **Structural** validation independent of crypto/attenuation: a UCAN is
    /// structurally well-formed iff both its DIDs resolve to valid `did:key`
    /// Ed25519 identities and (if present) `not_before <= expiration`. Each
    /// proof is recursively checked. Fail-closed on any malformation.
    pub fn validate_structure(&self) -> Result<(), UcanError> {
        // DIDs must be resolvable did:key identifiers.
        let _ = did_ed25519_key(&self.payload.issuer)?;
        let _ = did_ed25519_key(&self.payload.audience)?;
        if let (Some(nbf), Some(exp)) = (self.payload.not_before, self.payload.expiration) {
            if nbf > exp {
                return Err(UcanError::Malformed(format!(
                    "not_before ({nbf}) is after expiration ({exp})"
                )));
            }
        }
        for proof in &self.proofs {
            proof.validate_structure()?;
        }
        Ok(())
    }

    /// Is this UCAN temporally valid at `now` (unix seconds)? `not_before`
    /// inclusive, `expiration` inclusive of `now <= exp`. A `None` bound is
    /// unbounded on that side.
    #[must_use]
    pub fn is_valid_at(&self, now: u64) -> bool {
        if let Some(nbf) = self.payload.not_before {
            if now < nbf {
                return false;
            }
        }
        if let Some(exp) = self.payload.expiration {
            if now > exp {
                return false;
            }
        }
        true
    }

    /// Verify the hybrid-COSE composite signature on THIS UCAN's payload, AND
    /// recursively on every proof in the chain. The issuer's classical key is
    /// taken from its `did:key`; the matching ML-DSA-65 anchor is resolved
    /// through `verifier` (the trust store / key-binding layer — the same
    /// `register_pq_trust` reality `crypto::pq` models). Fail-closed: any link
    /// whose signature does not verify (or whose PQ anchor is missing under the
    /// hybrid policy) rejects the whole UCAN.
    ///
    /// This does NOT check attenuation/delegation linkage — that is
    /// [`super::chain::validate_chain`], deliberately separate.
    pub fn verify_signatures<V: UcanVerifier + ?Sized>(
        &self,
        verifier: &V,
    ) -> Result<(), UcanError> {
        let ed_bytes = did_ed25519_key(&self.payload.issuer)?;
        let payload_bytes = self.payload.signing_bytes()?;
        verifier.verify(
            &self.payload.issuer,
            &ed_bytes,
            &payload_bytes,
            &self.signature,
        )?;
        for proof in &self.proofs {
            proof.verify_signatures(verifier)?;
        }
        Ok(())
    }
}

/// Resolves a UCAN issuer's verified key material and checks a detached
/// hybrid-COSE signature over the payload bytes.
///
/// This is the seam between "what DID/key-binding layer says is anchored" and
/// "what the UCAN verifier trusts". The production implementor resolves the
/// issuer's anchored ML-DSA-65 verifying key (via the `register_pq_trust` /
/// `PqTrustStore` binding referenced in `crypto::cose_sign`) keyed by the
/// issuer's Ed25519 identity and calls
/// [`crate::crypto::cose_sign::verify_composite`] with `require_pq = true`. The
/// classical Ed25519 key is supplied (already decoded from the `did:key`).
pub trait UcanVerifier {
    /// Verify `signature` (a hybrid-COSE composite) over `payload` for the given
    /// `issuer`, whose decoded Ed25519 verifying key bytes are `ed_key`. Returns
    /// `Ok(())` iff the composite verifies under the project's hybrid policy
    /// (both EdDSA and the anchored ML-DSA-65 layer). Fail-closed otherwise.
    fn verify(
        &self,
        issuer: &Did,
        ed_key: &[u8; 32],
        payload: &[u8],
        signature: &[u8],
    ) -> Result<(), UcanError>;
}

// Blanket forwarding impls so a `&dyn UcanVerifier` (or `Box<dyn …>`) can be
// passed anywhere a generic `V: UcanVerifier` is expected — notably S5's
// `chain::validate<V>` and S6's runtime grant path. Without these, the runtime
// path would have to be monomorphized on a concrete verifier type, which fights
// the dynamic trust-store resolution S6 needs.
impl<V: UcanVerifier + ?Sized> UcanVerifier for &V {
    fn verify(
        &self,
        issuer: &Did,
        ed_key: &[u8; 32],
        payload: &[u8],
        signature: &[u8],
    ) -> Result<(), UcanError> {
        (**self).verify(issuer, ed_key, payload, signature)
    }
}

impl<V: UcanVerifier + ?Sized> UcanVerifier for std::boxed::Box<V> {
    fn verify(
        &self,
        issuer: &Did,
        ed_key: &[u8; 32],
        payload: &[u8],
        signature: &[u8],
    ) -> Result<(), UcanError> {
        (**self).verify(issuer, ed_key, payload, signature)
    }
}

/// Errors from UCAN parsing, structure, and signature verification. All map to a
/// fail-closed rejection at the call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UcanError {
    /// CBOR decode failure.
    Decode(String),
    /// CBOR encode failure (effectively unreachable for in-memory shapes).
    Encode(String),
    /// A DID was not a well-formed `did:key` Ed25519 identifier.
    BadDid { did: String, reason: String },
    /// Structurally malformed (e.g. inverted validity window).
    Malformed(String),
    /// A signature did not verify under the hybrid policy.
    BadSignature(String),
}

impl fmt::Display for UcanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UcanError::Decode(e) => write!(f, "UCAN decode failed: {e}"),
            UcanError::Encode(e) => write!(f, "UCAN encode failed: {e}"),
            UcanError::BadDid { did, reason } => write!(f, "invalid did:key {did}: {reason}"),
            UcanError::Malformed(e) => write!(f, "malformed UCAN: {e}"),
            UcanError::BadSignature(e) => write!(f, "UCAN signature verification failed: {e}"),
        }
    }
}

impl std::error::Error for UcanError {}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(super) mod test_support {
    //! Shared test helpers for building and signing UCANs with the real hybrid
    //! COSE path. Visible to sibling test modules (`chain`, `approval`).
    use super::*;
    use crate::auth::ucan::capability::{Ability, Capability, Resource};
    use crate::crypto::cose_sign::{sign_composite, verify_composite};
    use crate::crypto::pq::{ml_dsa_generate_keypair, MlDsaSigningKey, MlDsaVerifyingKey};
    use ed25519_dalek::{SigningKey, VerifyingKey};
    use std::collections::HashMap;

    // The UCAN payload-signature AAD is the public `super::UCAN_AAD` constant
    // (promoted out of `test_support` by S8 so production verifiers share it).

    /// A hybrid keypair (Ed25519 + ML-DSA-65) bound to one DID identity.
    pub struct TestIdentity {
        pub ed_sk: SigningKey,
        pub ed_vk: VerifyingKey,
        pub pq_sk: MlDsaSigningKey,
        pub pq_vk: MlDsaVerifyingKey,
    }

    impl TestIdentity {
        pub fn generate() -> Self {
            use rand::rngs::OsRng;
            let ed_sk = SigningKey::generate(&mut OsRng);
            let ed_vk = ed_sk.verifying_key();
            let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
            Self {
                ed_sk,
                ed_vk,
                pq_sk,
                pq_vk,
            }
        }

        pub fn did(&self) -> Did {
            Did::from_ed25519(&self.ed_vk.to_bytes())
        }
    }

    pub fn cap(resource: &str, ability: &str) -> Capability {
        Capability::new(Resource::new(resource), Ability::new(ability))
    }

    /// Build and hybrid-sign a UCAN from `issuer` to `audience` with `caps` and
    /// the given `proofs`.
    pub fn signed_ucan(
        issuer: &TestIdentity,
        audience: &Did,
        caps: Vec<Capability>,
        proofs: Vec<Ucan>,
    ) -> Ucan {
        let payload = UcanPayload {
            issuer: issuer.did(),
            audience: audience.clone(),
            capabilities: caps,
            not_before: None,
            expiration: Some(9_999_999_999),
            nonce: vec![1, 2, 3],
        };
        let bytes = payload.signing_bytes().unwrap();
        let signature =
            sign_composite(&issuer.ed_sk, Some(&issuer.pq_sk), &bytes, UCAN_AAD).unwrap();
        Ucan {
            payload,
            proofs,
            signature,
        }
    }

    /// A trust store keyed by DID string → anchored ML-DSA-65 verifying key,
    /// implementing [`UcanVerifier`] over the real hybrid composite verify.
    pub struct TestTrustStore {
        pub pq_by_did: HashMap<String, MlDsaVerifyingKey>,
    }

    impl TestTrustStore {
        pub fn new() -> Self {
            Self {
                pq_by_did: HashMap::new(),
            }
        }
        pub fn anchor(&mut self, id: &TestIdentity) {
            self.pq_by_did
                .insert(id.did().into_string(), id.pq_vk.clone());
        }
    }

    impl UcanVerifier for TestTrustStore {
        fn verify(
            &self,
            issuer: &Did,
            ed_key: &[u8; 32],
            payload: &[u8],
            signature: &[u8],
        ) -> Result<(), UcanError> {
            let ed_vk = VerifyingKey::from_bytes(ed_key)
                .map_err(|e| UcanError::BadSignature(e.to_string()))?;
            let pq_vk = self.pq_by_did.get(issuer.as_str());
            verify_composite(signature, &ed_vk, pq_vk, payload, UCAN_AAD, true)
                .map(|_| ())
                .map_err(|e| UcanError::BadSignature(e.to_string()))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn cbor_roundtrip_preserves_ucan() {
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let u = signed_ucan(
            &alice,
            &bob.did(),
            vec![cap("mac://model/*", "infer")],
            vec![],
        );
        let bytes = u.to_cbor().unwrap();
        let back = Ucan::from_cbor(&bytes).unwrap();
        assert_eq!(u, back);
    }

    #[test]
    fn signing_bytes_are_deterministic() {
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let u = signed_ucan(&alice, &bob.did(), vec![cap("mac://x", "y")], vec![]);
        assert_eq!(
            u.payload.signing_bytes().unwrap(),
            u.payload.signing_bytes().unwrap()
        );
    }

    #[test]
    fn validate_structure_accepts_well_formed() {
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let u = signed_ucan(&alice, &bob.did(), vec![cap("mac://x", "y")], vec![]);
        assert!(u.validate_structure().is_ok());
    }

    #[test]
    fn validate_structure_rejects_bad_did() {
        let alice = TestIdentity::generate();
        let mut u = signed_ucan(&alice, &alice.did(), vec![], vec![]);
        u.payload.audience = Did::new("did:web:example.com".to_owned());
        assert!(matches!(
            u.validate_structure(),
            Err(UcanError::BadDid { .. })
        ));
    }

    #[test]
    fn validate_structure_rejects_inverted_window() {
        let alice = TestIdentity::generate();
        let mut u = signed_ucan(&alice, &alice.did(), vec![], vec![]);
        u.payload.not_before = Some(100);
        u.payload.expiration = Some(50);
        assert!(matches!(
            u.validate_structure(),
            Err(UcanError::Malformed(_))
        ));
    }

    #[test]
    fn is_valid_at_respects_window() {
        let alice = TestIdentity::generate();
        let mut u = signed_ucan(&alice, &alice.did(), vec![], vec![]);
        u.payload.not_before = Some(100);
        u.payload.expiration = Some(200);
        assert!(!u.is_valid_at(99));
        assert!(u.is_valid_at(100));
        assert!(u.is_valid_at(200));
        assert!(!u.is_valid_at(201));
    }

    #[test]
    fn verify_signatures_accepts_valid_hybrid_signature() {
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let u = signed_ucan(
            &alice,
            &bob.did(),
            vec![cap("mac://model/*", "infer")],
            vec![],
        );

        let mut store = TestTrustStore::new();
        store.anchor(&alice);
        assert!(u.verify_signatures(&store).is_ok());
    }

    #[test]
    fn verify_signatures_rejects_missing_pq_anchor() {
        // No anchored ML-DSA key → require_pq=true must fail closed.
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let u = signed_ucan(&alice, &bob.did(), vec![cap("mac://x", "y")], vec![]);
        let store = TestTrustStore::new(); // empty: no anchor
        assert!(matches!(
            u.verify_signatures(&store),
            Err(UcanError::BadSignature(_))
        ));
    }

    #[test]
    fn root_walks_to_the_self_issued_delegator() {
        // user (root) ⟶ gateway ⟶ mcp. The delegator (source of authority) is
        // the root's issuer (user); the actor/presenter is the leaf audience (mcp).
        let user = TestIdentity::generate();
        let gateway = TestIdentity::generate();
        let mcp = TestIdentity::generate();

        let root = signed_ucan(&user, &gateway.did(), vec![cap("mac://model/x", "infer")], vec![]);
        let leaf = signed_ucan(
            &gateway,
            &mcp.did(),
            vec![cap("mac://model/x", "infer")],
            vec![root.clone()],
        );

        assert_eq!(leaf.audience(), &mcp.did(), "leaf audience = actor/presenter");
        assert_eq!(leaf.issuer(), &gateway.did(), "leaf issuer = immediate (intermediate) issuer");
        assert_eq!(
            leaf.root_issuer(),
            &user.did(),
            "root_issuer = ultimate delegator, NOT the immediate issuer"
        );
        assert!(leaf.root().proofs.is_empty(), "root has empty proofs");
        // A root UCAN is its own root.
        assert_eq!(root.root_issuer(), &user.did());
    }

    #[test]
    fn verify_signatures_rejects_tampered_payload() {
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let mut u = signed_ucan(
            &alice,
            &bob.did(),
            vec![cap("mac://model/qwen", "infer")],
            vec![],
        );
        // Widen the capability AFTER signing — signature must no longer verify.
        u.payload.capabilities = vec![cap("mac://*", "*")];
        let mut store = TestTrustStore::new();
        store.anchor(&alice);
        assert!(matches!(
            u.verify_signatures(&store),
            Err(UcanError::BadSignature(_))
        ));
    }

    #[test]
    fn verify_signatures_recurses_into_proofs() {
        // root → alice → bob; tamper the embedded root proof, expect rejection.
        let root = TestIdentity::generate();
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let root_ucan = signed_ucan(&root, &alice.did(), vec![cap("mac://*", "*")], vec![]);
        let mut alice_ucan = signed_ucan(
            &alice,
            &bob.did(),
            vec![cap("mac://model/qwen", "infer")],
            vec![root_ucan],
        );

        let mut store = TestTrustStore::new();
        store.anchor(&root);
        store.anchor(&alice);
        assert!(alice_ucan.verify_signatures(&store).is_ok());

        // Corrupt the proof's signature.
        alice_ucan.proofs[0].signature[0] ^= 0xFF;
        assert!(matches!(
            alice_ucan.verify_signatures(&store),
            Err(UcanError::BadSignature(_))
        ));
    }
}
