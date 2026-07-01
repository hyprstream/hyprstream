//! ATProto perimeter gateway — external-DID → assurance binding (#549, first half).
//!
//! This is the **enrollment-time** translation seam between a federated ATProto
//! identity (an external `Did`) and the internal MAC lattice (#548). It "translates,
//! does not tunnel": an external DID is resolved **once, off the auth path**, its
//! verified key material is mapped to an [`Assurance`], and the result is pinned as
//! an [`EnrolledPeer`] carrying a derived [`SecurityContext`]. The per-op reference
//! monitor (#548 S2) then reads the *cached* context and never re-resolves.
//!
//! ## What this module is / is not
//!
//! - **Is:** the `external Did → VerifiedKeyMaterial → SecurityContext` mapping plus
//!   a mutable, admission-gated [`EnrollmentStore`]. It consumes the (mirrored)
//!   #579 resolver and the *existing* epic MAC types ([`Assurance`],
//!   [`SecurityLabel`], [`SecurityContext`], [`VerifiedKeyMaterial`]) — it does
//!   **not** re-declare any of them.
//! - **Is not:** per-op enforcement (#548 S2), the OAuth-scope/lexicon → UCAN+MAC
//!   half (split follow-up), or a concrete `did:plc`/`did:web` resolver (#579).
//!
//! ## Fail-closed
//!
//! Any resolver error, a `None` DID, or a missing verifying key yields `Err` ⇒ the
//! peer is **not** enrolled. There is no default enrollment and no default
//! assurance: an un-resolvable identity dominates nothing above the lattice floor.
//!
//! ## Zero standing privilege (ZSP)
//!
//! The enrollment edge ceiling is `Level::Internal × <crypto-derived assurance> ×
//! EMPTY` compartments. Compartments are **explicit grants made later**, never
//! conferred at enrollment. The clamp-down in
//! [`SecurityContext::from_clearance`] guarantees a classical peer can never obtain
//! a `PqHybrid` context even if the ceiling allowed it.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use parking_lot::RwLock;

use crate::admission::AdmittedIdentity;
use crate::auth::mac::{
    Assurance, CompartmentSet, Level, SecurityContext, SecurityLabel, VerifiedKeyMaterial,
};
use crate::crypto::pq::MlDsaVerifyingKey;
use crate::identity::Did;

/// Raw 32-byte Ed25519 verifying key.
///
/// Mirror alias for the #579 resolver contract. When #579 lands and exports its
/// own key type, this alias swaps to it without touching the gateway logic.
pub type Ed25519Vk = [u8; 32];

/// ML-DSA-65 verifying key (FIPS 204).
///
/// Mirror alias for the #579 resolver contract, pinned to the always-compiled PQ
/// primitive ([`crate::crypto::pq::MlDsaVerifyingKey`]).
pub type MlDsaVk = MlDsaVerifyingKey;

/// The key material a resolver returns for an external identity.
///
/// **Mirror** of the #579 contract; field-for-field what `resolve_identity_keys`
/// must yield. `ml_dsa_65 == None` ⇒ classical-only edge; `Some` ⇒ a hybrid-PQC
/// anchor is present. The `assurance` is **crypto-derived by the resolver**, never
/// self-asserted — the gateway maps it 1:1 into [`VerifiedKeyMaterial`].
#[derive(Clone)]
pub struct IdentityKeys {
    /// The classical Ed25519 verifying key, if the identity published one.
    pub ed25519: Option<Ed25519Vk>,
    /// The bound ML-DSA-65 verifying key, if the identity published a PQ anchor.
    pub ml_dsa_65: Option<MlDsaVk>,
    /// The assurance the resolver established for this identity (crypto-derived).
    pub assurance: Assurance,
}

/// Resolves an external `Did` to its verified key material.
///
/// **Mirror** of the #579 resolver trait — its signature matches #579 exactly so a
/// real `did:plc`/`did:web` resolver (the #579 deliverable) drops in unchanged.
/// Until #579 lands, the gateway is exercised against fixture implementations (as
/// `admission.rs` does with its `DidDocResolve` test doubles).
///
/// **Fail-closed contract:** an implementation MUST return `Err` on any inability
/// to establish key material (network failure, unknown DID, malformed document) —
/// never a default-`Unverified` `Ok`. The gateway treats `Err` as "not enrolled".
pub trait IdentityResolver: Send + Sync {
    /// Resolve `did` to its verified key material, or `Err` (fail-closed).
    fn resolve_identity_keys(&self, did: &Did) -> Result<IdentityKeys>;
}

/// A pinned, enrolled external peer.
///
/// Produced by [`AtprotoPerimeterGateway::enroll`] and cached in the
/// [`EnrollmentStore`]. The reference monitor (#548 S2) reads `context` per-op via
/// [`EnrollmentStore::get`]; it never re-resolves. The `(did, channel_key,
/// assurance)` triple is the deliberate pin — rotation requires re-enrollment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnrolledPeer {
    /// The external identity this peer enrolled under.
    pub did: Did,
    /// The peer's authenticated 32-byte channel/envelope key (the admission `cnf`).
    pub channel_key: [u8; 32],
    /// The crypto-derived assurance pinned at enrollment.
    pub assurance: Assurance,
    /// The derived subject context the monitor compares against object labels.
    pub context: SecurityContext,
    /// The PDS endpoint advertised for this identity, if any (discovery only).
    pub pds_endpoint: Option<String>,
}

/// Map a crypto-derived [`Assurance`] to the MAC seam [`VerifiedKeyMaterial`], 1:1
/// by name. This is a *routing* step — assurance is already established by the
/// resolver; this only re-expresses it in the type the lattice consumes.
fn assurance_to_key_material(assurance: Assurance) -> VerifiedKeyMaterial {
    match assurance {
        Assurance::Unverified => VerifiedKeyMaterial::Unverified,
        Assurance::Classical => VerifiedKeyMaterial::Classical,
        Assurance::PqHybrid => VerifiedKeyMaterial::PqHybrid,
    }
}

/// The perimeter gateway: turns an admission-gated external identity into an
/// enrolled internal peer with a derived MAC context.
///
/// Generic over the resolver so it scaffolds against the #579 mirror and fixtures
/// today, and against the real `did:plc`/`did:web` resolver once #579 merges.
pub struct AtprotoPerimeterGateway<R: IdentityResolver> {
    resolver: R,
}

impl<R: IdentityResolver> AtprotoPerimeterGateway<R> {
    /// Construct a gateway over an [`IdentityResolver`].
    pub fn new(resolver: R) -> Self {
        Self { resolver }
    }

    /// The enrollment edge ceiling: `Level::Internal × <derived assurance> × {}`.
    ///
    /// ZSP: the level is the federation edge default (`Internal`), assurance is the
    /// crypto-derived axis (so [`SecurityContext::from_clearance`]'s clamp-down is a
    /// no-op against an honest ceiling, and a *defensive* upper bound otherwise),
    /// and compartments are **EMPTY** — every need-to-know compartment is an
    /// explicit grant made after enrollment, never conferred here.
    fn edge_ceiling(derived: Assurance) -> SecurityLabel {
        SecurityLabel::new(Level::Internal, derived, CompartmentSet::EMPTY)
    }

    /// Enroll an [`AdmittedIdentity`] (admission-gated) into an [`EnrolledPeer`].
    ///
    /// Steps (all enrollment-time, never per-op):
    /// 1. resolve the admitted DID's verified key material (#579 mirror);
    /// 2. map [`Assurance`] → [`VerifiedKeyMaterial`] 1:1 by name;
    /// 3. clamp the edge ceiling DOWN to the crypto-derived assurance via
    ///    [`SecurityContext::from_clearance`] (the #548 invariant);
    /// 4. pin the result as an [`EnrolledPeer`].
    ///
    /// **Fail-closed:** a `None` admitted DID, any resolver error, or a missing
    /// verifying key for the resolved assurance returns `Err` ⇒ not enrolled.
    pub fn enroll(&self, admitted: &AdmittedIdentity) -> Result<EnrolledPeer> {
        // An admission that matched only via JWKS fallback carries no DID; without a
        // DID there is nothing to resolve → fail-closed.
        let did_str = admitted.did.as_deref().ok_or_else(|| {
            anyhow!("perimeter enroll: admitted identity carries no DID (fail-closed)")
        })?;
        let did = Did::new(did_str.to_owned());

        // Resolve OFF the auth path. Any resolver error is fail-closed.
        let keys = self
            .resolver
            .resolve_identity_keys(&did)
            .map_err(|e| anyhow!("perimeter enroll: DID {did} did not resolve: {e}"))?;

        // Fail-closed on missing verifying key material for the asserted assurance:
        // a `Classical`/`PqHybrid` assurance with no Ed25519 key, or a `PqHybrid`
        // assurance with no ML-DSA-65 anchor, is an inconsistent resolution we will
        // not enroll.
        match keys.assurance {
            Assurance::Unverified => {
                return Err(anyhow!(
                    "perimeter enroll: DID {did} resolved Unverified (no verifiable key) — fail-closed"
                ));
            }
            Assurance::Classical => {
                if keys.ed25519.is_none() {
                    return Err(anyhow!(
                        "perimeter enroll: DID {did} asserts Classical but published no Ed25519 key — fail-closed"
                    ));
                }
            }
            Assurance::PqHybrid => {
                if keys.ed25519.is_none() {
                    return Err(anyhow!(
                        "perimeter enroll: DID {did} asserts PqHybrid but published no Ed25519 key — fail-closed"
                    ));
                }
                if keys.ml_dsa_65.is_none() {
                    return Err(anyhow!(
                        "perimeter enroll: DID {did} asserts PqHybrid but published no ML-DSA-65 anchor — fail-closed"
                    ));
                }
            }
        }

        let key_material = assurance_to_key_material(keys.assurance);
        let derived = key_material.assurance();

        // Edge ceiling × clamp-DOWN to crypto-derived assurance (#548).
        let context = SecurityContext::from_clearance(Self::edge_ceiling(derived), key_material);

        Ok(EnrolledPeer {
            did,
            channel_key: admitted.key,
            assurance: context.assurance(),
            context,
            pds_endpoint: None,
        })
    }
}

/// A mutable, admission-gated store of enrolled external peers.
///
/// **Deliberately separate** from the native, admin-anchored, post-install
/// **immutable** [`crate::envelope::KeyedPqTrustStore`] — that PQ anchor is left
/// untouched. This store holds the *federation perimeter's* enrolled peers and is
/// mutated only through [`AtprotoPerimeterGateway::enroll`]'s admission-gated path:
/// [`insert`](EnrollmentStore::insert) takes an [`EnrolledPeer`], which can only be
/// produced from an [`AdmittedIdentity`].
///
/// [`get`](EnrollmentStore::get) is the read seam the future per-op monitor (#548
/// S2) uses to fetch a peer's cached [`SecurityContext`] without re-resolving.
#[derive(Clone, Default)]
pub struct EnrollmentStore {
    peers: Arc<RwLock<HashMap<Did, EnrolledPeer>>>,
}

impl EnrollmentStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) an enrolled peer, keyed by its DID.
    ///
    /// Only an [`EnrolledPeer`] — which is produced exclusively from an
    /// admission-gated enrollment — can be inserted, so the store stays
    /// admission-gated by construction.
    pub fn insert(&self, peer: EnrolledPeer) {
        self.peers.write().insert(peer.did.clone(), peer);
    }

    /// Look up an enrolled peer's cached state by DID, if present.
    pub fn get(&self, did: &Did) -> Option<EnrolledPeer> {
        self.peers.read().get(did).cloned()
    }

    /// Number of enrolled peers.
    pub fn len(&self) -> usize {
        self.peers.read().len()
    }

    /// Whether no peer is enrolled.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::crypto::pq::ml_dsa_generate_keypair;

    const DID_WEB: &str = "did:web:peer.example";

    fn admitted(did: Option<&str>, key: [u8; 32]) -> AdmittedIdentity {
        AdmittedIdentity {
            origin: "https://peer.example".to_owned(),
            did: did.map(str::to_owned),
            key,
        }
    }

    fn an_ml_dsa_vk() -> MlDsaVk {
        ml_dsa_generate_keypair().1
    }

    /// Fixture resolver returning fixed key material — the #549 stand-in for #579,
    /// mirroring `admission.rs`'s `FixtureDoc`/`FailingResolve` test doubles.
    struct FixtureResolver(IdentityKeys);
    impl IdentityResolver for FixtureResolver {
        fn resolve_identity_keys(&self, _did: &Did) -> Result<IdentityKeys> {
            Ok(self.0.clone())
        }
    }

    /// Fixture resolver that always fails — exercises the fail-closed path.
    struct FailingResolver;
    impl IdentityResolver for FailingResolver {
        fn resolve_identity_keys(&self, did: &Did) -> Result<IdentityKeys> {
            Err(anyhow!("simulated resolution failure for {did}"))
        }
    }

    #[test]
    fn classical_peer_enrolls_at_classical_assurance() {
        let keys = IdentityKeys {
            ed25519: Some([7u8; 32]),
            ml_dsa_65: None,
            assurance: Assurance::Classical,
        };
        let gw = AtprotoPerimeterGateway::new(FixtureResolver(keys));
        let peer = gw
            .enroll(&admitted(Some(DID_WEB), [7u8; 32]))
            .expect("classical must enroll");

        assert_eq!(peer.assurance, Assurance::Classical);
        assert_eq!(peer.context.assurance(), Assurance::Classical);
        // ZSP edge ceiling: Internal level, no compartments.
        assert_eq!(peer.context.level(), Level::Internal);
        assert!(peer.context.compartments().is_empty());
        assert_eq!(peer.channel_key, [7u8; 32]);
        assert_eq!(peer.did.as_str(), DID_WEB);
    }

    #[test]
    fn pqhybrid_peer_enrolls_at_pqhybrid_assurance() {
        let keys = IdentityKeys {
            ed25519: Some([9u8; 32]),
            ml_dsa_65: Some(an_ml_dsa_vk()),
            assurance: Assurance::PqHybrid,
        };
        let gw = AtprotoPerimeterGateway::new(FixtureResolver(keys));
        let peer = gw
            .enroll(&admitted(Some(DID_WEB), [9u8; 32]))
            .expect("pq-hybrid must enroll");

        assert_eq!(peer.assurance, Assurance::PqHybrid);
        assert_eq!(peer.context.assurance(), Assurance::PqHybrid);
    }

    #[test]
    fn unverified_resolution_fails_closed() {
        let keys = IdentityKeys {
            ed25519: None,
            ml_dsa_65: None,
            assurance: Assurance::Unverified,
        };
        let gw = AtprotoPerimeterGateway::new(FixtureResolver(keys));
        assert!(gw.enroll(&admitted(Some(DID_WEB), [0u8; 32])).is_err());
    }

    #[test]
    fn resolver_error_fails_closed() {
        let gw = AtprotoPerimeterGateway::new(FailingResolver);
        assert!(gw.enroll(&admitted(Some(DID_WEB), [1u8; 32])).is_err());
    }

    #[test]
    fn missing_did_fails_closed() {
        // Admission matched via JWKS fallback only — no DID to resolve.
        let keys = IdentityKeys {
            ed25519: Some([2u8; 32]),
            ml_dsa_65: None,
            assurance: Assurance::Classical,
        };
        let gw = AtprotoPerimeterGateway::new(FixtureResolver(keys));
        assert!(gw.enroll(&admitted(None, [2u8; 32])).is_err());
    }

    #[test]
    fn pqhybrid_assertion_without_ml_dsa_fails_closed() {
        // Inconsistent resolution: claims PqHybrid but published no PQ anchor.
        let keys = IdentityKeys {
            ed25519: Some([3u8; 32]),
            ml_dsa_65: None,
            assurance: Assurance::PqHybrid,
        };
        let gw = AtprotoPerimeterGateway::new(FixtureResolver(keys));
        assert!(gw.enroll(&admitted(Some(DID_WEB), [3u8; 32])).is_err());
    }

    #[test]
    fn classical_peer_cannot_obtain_pqhybrid_context() {
        // A classical-key peer, even resolved at the PqHybrid-capable edge, must
        // clamp to Classical and fail to dominate a PqHybrid object.
        let keys = IdentityKeys {
            ed25519: Some([4u8; 32]),
            ml_dsa_65: None,
            assurance: Assurance::Classical,
        };
        let gw = AtprotoPerimeterGateway::new(FixtureResolver(keys));
        let peer = gw.enroll(&admitted(Some(DID_WEB), [4u8; 32])).unwrap();

        let pq_object =
            SecurityLabel::new(Level::Public, Assurance::PqHybrid, CompartmentSet::EMPTY);
        assert!(!peer.context.can_access(&pq_object));
        assert_eq!(peer.context.assurance(), Assurance::Classical);
    }

    #[test]
    fn enrollment_store_insert_get_roundtrip() {
        let keys = IdentityKeys {
            ed25519: Some([5u8; 32]),
            ml_dsa_65: None,
            assurance: Assurance::Classical,
        };
        let gw = AtprotoPerimeterGateway::new(FixtureResolver(keys));
        let peer = gw.enroll(&admitted(Some(DID_WEB), [5u8; 32])).unwrap();

        let store = EnrollmentStore::new();
        assert!(store.is_empty());
        store.insert(peer.clone());
        assert_eq!(store.len(), 1);

        let got = store
            .get(&Did::new(DID_WEB.to_owned()))
            .expect("must round-trip");
        assert_eq!(got, peer);
        assert!(store
            .get(&Did::new("did:web:absent.example".to_owned()))
            .is_none());
    }

    #[test]
    fn native_pq_anchor_is_untouched_by_enrollment() {
        // The perimeter EnrollmentStore is separate from the native, immutable
        // KeyedPqTrustStore. Enrolling peers must not touch the native anchor.
        let native = crate::envelope::KeyedPqTrustStore::new();
        assert!(native.is_empty());

        let keys = IdentityKeys {
            ed25519: Some([6u8; 32]),
            ml_dsa_65: Some(an_ml_dsa_vk()),
            assurance: Assurance::PqHybrid,
        };
        let gw = AtprotoPerimeterGateway::new(FixtureResolver(keys));
        let peer = gw.enroll(&admitted(Some(DID_WEB), [6u8; 32])).unwrap();

        let store = EnrollmentStore::new();
        store.insert(peer);
        assert_eq!(store.len(), 1);

        // Native anchor still empty — no cross-contamination.
        assert!(native.is_empty());
    }
}
