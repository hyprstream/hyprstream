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
//! - **Is:** the `external Did → VerifiedKeyMaterial → SecurityContext` mapping,
//!   mutable admission-gated [`EnrollmentStore`], and exact, DPoP-gated ATProto
//!   OAuth-scope/lexicon → bounded interior UCAN+MAC translation. It consumes the
//!   (mirrored) #579 resolver and the *existing* epic MAC types ([`Assurance`],
//!   [`SecurityLabel`], [`SecurityContext`], [`VerifiedKeyMaterial`]) — it does
//!   **not** re-declare any of them.
//! - **Is not:** per-op enforcement (#548 S2), concrete DPoP proof verification
//!   (performed by the OAuth boundary before translation), or a concrete
//!   `did:plc`/`did:web` resolver (#579).
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
use crate::identity::{Did, IdentityResolver};

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

/// An exact ATProto OAuth scope plus lexicon permission presented at the
/// federation edge.
///
/// Scope strings and lexicon NSIDs are deliberately opaque here: accepting a
/// prefix or wildcard would turn an OAuth permission we do not understand into
/// an interior authority grant. The perimeter therefore maps only exact,
/// administrator-configured pairs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AtprotoPermission {
    /// The OAuth scope asserted by the already-verified ATProto authorization.
    pub scope: String,
    /// The lexicon NSID / operation the caller wants to use under `scope`.
    pub lexicon: String,
}

impl AtprotoPermission {
    /// Build an exact external permission. Empty or whitespace-containing
    /// components are not valid OAuth scope tokens / lexicon NSIDs and are
    /// rejected before they reach the mapping table.
    pub fn new(scope: impl Into<String>, lexicon: impl Into<String>) -> Result<Self> {
        let scope = scope.into();
        let lexicon = lexicon.into();
        if scope.is_empty() || scope.bytes().any(|b| b.is_ascii_whitespace()) {
            return Err(anyhow!(
                "perimeter authorization: invalid ATProto OAuth scope"
            ));
        }
        if lexicon.is_empty() || lexicon.bytes().any(|b| b.is_ascii_whitespace()) {
            return Err(anyhow!(
                "perimeter authorization: invalid ATProto lexicon permission"
            ));
        }
        Ok(Self { scope, lexicon })
    }
}

/// Evidence that the surrounding OAuth resource server verified a DPoP proof
/// and bound this authorization to its JWK thumbprint (`jkt`).
///
/// This module intentionally does not parse or verify DPoP JWTs: that belongs
/// to the OAuth boundary (`services::oauth::dpop`) where the request method,
/// URI, nonce, and replay cache are available. Requiring this value makes the
/// dependency explicit and ensures a translation result is sender-bound rather
/// than usable as a bearer grant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedDpopBinding(String);

impl VerifiedDpopBinding {
    /// Construct from a `jkt` returned by a successful DPoP verifier.
    ///
    /// Callers MUST NOT construct this from an unverified request header; the
    /// verifier is responsible for signature, `htm`/`htu`, nonce, and replay
    /// validation. We reject empty/whitespace values so a missing proof cannot
    /// be represented as a binding.
    pub fn from_verified_jkt(jkt: impl Into<String>) -> Result<Self> {
        let jkt = jkt.into();
        if jkt.is_empty() || jkt.bytes().any(|b| b.is_ascii_whitespace()) {
            return Err(anyhow!(
                "perimeter authorization: missing verified DPoP binding"
            ));
        }
        Ok(Self(jkt))
    }

    /// The verified sender-binding JWK thumbprint for token minting/audit.
    pub fn jkt(&self) -> &str {
        &self.0
    }
}

/// A policy-approved, one-way translation from an exact ATProto permission to
/// an interior UCAN capability and the greatest MAC object label it may target.
///
/// This is configuration supplied by the local authority, never by an external
/// DID. `maximum_label` does not grant clearance: the enrolled peer must already
/// dominate it, so an OAuth permission cannot raise MLS, assurance, or
/// compartment clearance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtprotoPermissionMapping {
    /// The exact interior capability the trusted gateway may place in a new
    /// local UCAN. External ATProto records never carry that UCAN themselves.
    pub capability: crate::auth::ucan::capability::Capability,
    /// The bounded MAC label enforced alongside this translated capability.
    pub maximum_label: SecurityLabel,
}

impl AtprotoPermissionMapping {
    /// Create a mapping after rejecting wildcard interior authority. A broad
    /// external grant must be expanded into individually reviewed exact entries
    /// instead of being smuggled through `*` or a wildcard resource.
    pub fn new(
        capability: crate::auth::ucan::capability::Capability,
        maximum_label: SecurityLabel,
    ) -> Result<Self> {
        let resource = capability.resource.as_str();
        let ability = capability.ability.as_str();
        if !resource.starts_with("mac://")
            || resource.len() == "mac://".len()
            || resource.contains('*')
            || resource.bytes().any(|b| b.is_ascii_whitespace())
            || ability.is_empty()
            || ability.contains('*')
            || ability.bytes().any(|b| b.is_ascii_whitespace())
        {
            return Err(anyhow!(
                "perimeter authorization: mapping must contain an exact non-wildcard interior capability"
            ));
        }
        Ok(Self {
            capability,
            maximum_label,
        })
    }
}

/// A configured, fail-closed ATProto OAuth/lexicon → UCAN+MAC translator.
///
/// It owns no DID resolver and has no network capability. Enrollment resolves
/// and pins the DID before this translator is ever called; authorization reads
/// only [`EnrolledPeer::context`]. This keeps did:web/did:plc/PLC resolution out
/// of the per-operation authorization path by construction.
#[derive(Clone, Default)]
pub struct AtprotoPermissionTranslator {
    mappings: HashMap<AtprotoPermission, AtprotoPermissionMapping>,
}

impl AtprotoPermissionTranslator {
    /// Build from a closed, locally administered mapping table. Duplicate
    /// external permissions are rejected rather than allowing load order to
    /// select an authority unexpectedly.
    pub fn new(
        mappings: impl IntoIterator<Item = (AtprotoPermission, AtprotoPermissionMapping)>,
    ) -> Result<Self> {
        let mut table = HashMap::new();
        for (permission, mapping) in mappings {
            if table.insert(permission.clone(), mapping).is_some() {
                return Err(anyhow!(
                    "perimeter authorization: duplicate mapping for scope {:?}, lexicon {:?}",
                    permission.scope,
                    permission.lexicon
                ));
            }
        }
        Ok(Self { mappings: table })
    }

    /// Translate verified external permissions into narrowly bounded interior
    /// capability requests.
    ///
    /// The OAuth resource server must verify DPoP before constructing
    /// `dpop_binding`; the gateway then requires every permission to have an
    /// exact local mapping and every target label to be dominated by the cached
    /// enrollment context. Any missing mapping, over-clearance request, absent
    /// peer, or malformed binding returns `Err` — there is no fallback,
    /// wildcard, or default capability.
    pub fn translate(
        &self,
        peer: &EnrolledPeer,
        dpop_binding: VerifiedDpopBinding,
        permissions: &[AtprotoPermission],
    ) -> Result<TranslatedAtprotoAuthorization> {
        if permissions.is_empty() {
            return Err(anyhow!(
                "perimeter authorization: no ATProto permissions presented (fail-closed)"
            ));
        }

        let mut capabilities = Vec::with_capacity(permissions.len());
        let mut maximum_labels = Vec::with_capacity(permissions.len());
        for permission in permissions {
            let mapping = self.mappings.get(permission).ok_or_else(|| {
                anyhow!(
                    "perimeter authorization: unmappable ATProto scope {:?} / lexicon {:?} (fail-closed)",
                    permission.scope,
                    permission.lexicon
                )
            })?;

            // A translated OAuth permission is never a clearance grant. The
            // pinned crypto-derived context is the independent MAC floor.
            if !peer.context.can_access(&mapping.maximum_label) {
                return Err(anyhow!(
                    "perimeter authorization: peer {} cannot access mapped label {} (fail-closed)",
                    peer.did,
                    mapping.maximum_label
                ));
            }
            capabilities.push(mapping.capability.clone());
            maximum_labels.push(mapping.maximum_label);
        }

        Ok(TranslatedAtprotoAuthorization {
            did: peer.did.clone(),
            dpop_binding,
            capabilities,
            maximum_labels,
        })
    }
}

/// The bounded result of edge translation, ready for the trusted interior UCAN
/// issuer / MAC compiler. It is deliberately not an external UCAN transport
/// format: UCAN authoring and delegation remain inside HyprStream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslatedAtprotoAuthorization {
    /// The enrolled DID whose cached context bounded the translation.
    pub did: Did,
    /// Sender binding that must be retained by the local token/UCAN issuer.
    pub dpop_binding: VerifiedDpopBinding,
    /// Exact local capabilities to be authored into an interior UCAN.
    pub capabilities: Vec<crate::auth::ucan::capability::Capability>,
    /// Corresponding maximum MAC labels, checked against the cached context.
    pub maximum_labels: Vec<SecurityLabel>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::crypto::pq::ml_dsa_generate_keypair;
    use crate::identity::{IdentityKeys, MlDsaVk};

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

    fn classical_peer() -> EnrolledPeer {
        let keys = IdentityKeys {
            ed25519: Some([11u8; 32]),
            ml_dsa_65: None,
            assurance: Assurance::Classical,
        };
        AtprotoPerimeterGateway::new(FixtureResolver(keys))
            .enroll(&admitted(Some(DID_WEB), [11u8; 32]))
            .unwrap()
    }

    fn mapping(
        ability: &str,
        resource: &str,
        maximum_label: SecurityLabel,
    ) -> AtprotoPermissionMapping {
        use crate::auth::ucan::capability::{Ability, Capability, Resource};
        AtprotoPermissionMapping::new(
            Capability::new(Resource::new(resource), Ability::new(ability)),
            maximum_label,
        )
        .unwrap()
    }

    #[test]
    fn exact_permission_translates_to_bounded_interior_capability() {
        let permission =
            AtprotoPermission::new("atproto.transition:read", "ai.hyprstream.model.get").unwrap();
        let max_label =
            SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY);
        let translator = AtprotoPermissionTranslator::new([(
            permission.clone(),
            mapping("read", "mac://model/qwen3", max_label),
        )])
        .unwrap();

        let result = translator
            .translate(
                &classical_peer(),
                VerifiedDpopBinding::from_verified_jkt("verified-jkt").unwrap(),
                &[permission],
            )
            .unwrap();

        assert_eq!(result.did.as_str(), DID_WEB);
        assert_eq!(result.dpop_binding.jkt(), "verified-jkt");
        assert_eq!(result.capabilities.len(), 1);
        assert_eq!(result.capabilities[0].ability.as_str(), "read");
        assert_eq!(
            result.capabilities[0].resource.as_str(),
            "mac://model/qwen3"
        );
        assert_eq!(result.maximum_labels, vec![max_label]);
    }

    #[test]
    fn unmappable_scope_or_lexicon_fails_closed() {
        let known =
            AtprotoPermission::new("atproto.transition:read", "ai.hyprstream.model.get").unwrap();
        let translator = AtprotoPermissionTranslator::new([(
            known,
            mapping(
                "read",
                "mac://model/qwen3",
                SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY),
            ),
        )])
        .unwrap();
        let unknown =
            AtprotoPermission::new("atproto.transition:write", "ai.hyprstream.model.get").unwrap();

        assert!(translator
            .translate(
                &classical_peer(),
                VerifiedDpopBinding::from_verified_jkt("verified-jkt").unwrap(),
                &[unknown],
            )
            .is_err());
    }

    #[test]
    fn mapped_permission_cannot_raise_cached_assurance_or_clearance() {
        let permission =
            AtprotoPermission::new("atproto.transition:read", "ai.hyprstream.model.get").unwrap();
        let translator = AtprotoPermissionTranslator::new([(
            permission.clone(),
            mapping(
                "read",
                "mac://model/qwen3",
                SecurityLabel::new(Level::Public, Assurance::PqHybrid, CompartmentSet::EMPTY),
            ),
        )])
        .unwrap();

        assert!(translator
            .translate(
                &classical_peer(),
                VerifiedDpopBinding::from_verified_jkt("verified-jkt").unwrap(),
                &[permission],
            )
            .is_err());
    }

    #[test]
    fn translator_rejects_wildcard_capabilities_and_duplicate_mappings() {
        use crate::auth::ucan::capability::{Ability, Capability, Resource};
        let label = SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY);
        assert!(AtprotoPermissionMapping::new(
            Capability::new(Resource::new("mac://model/*"), Ability::new("read")),
            label,
        )
        .is_err());
        assert!(AtprotoPermissionMapping::new(
            Capability::new(Resource::new("mac://model/qwen3"), Ability::new("read/*")),
            label,
        )
        .is_err());

        let permission =
            AtprotoPermission::new("atproto.transition:read", "ai.hyprstream.model.get").unwrap();
        let approved = mapping("read", "mac://model/qwen3", label);
        assert!(AtprotoPermissionTranslator::new([
            (permission.clone(), approved.clone()),
            (permission, approved),
        ])
        .is_err());
    }

    #[test]
    fn empty_permissions_and_missing_dpop_binding_fail_closed() {
        let translator = AtprotoPermissionTranslator::default();
        assert!(VerifiedDpopBinding::from_verified_jkt("").is_err());
        assert!(translator
            .translate(
                &classical_peer(),
                VerifiedDpopBinding::from_verified_jkt("verified-jkt").unwrap(),
                &[],
            )
            .is_err());
    }
}
