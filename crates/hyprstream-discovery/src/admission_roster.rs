//! Derived native admission roster (#1652): deployment service identities
//! resolved from the live accepted-state store at admission time, not from
//! DID-valued process configuration.
//!
//! Formerly `quic.moql_subject_tenants` embedded every deployment DID in
//! config.toml, so each service-identity re-initialization (R1: reviewed
//! store wipe + re-provision, all DIDs change) left every running process
//! pinned to stale DIDs until the boot projector re-wrote the file AND every
//! container restarted. The tenant is a pure roster projection — metal writes
//! `did = "local"` uniformly for all eight services — so it is now DERIVED:
//! DID → accepted state → capsule `#<name>` service entries ∩ the registered
//! factory roster ⇒ tenant [`LOCAL_TENANT`]; everything else ⇒ unresolved,
//! deny. The membership filter is the same rule as the offline provisioner's
//! `validate_service_roster` and the renewal set, so foreign at9p records
//! admitted through the public `ingestAt9pCandidate` RPC (arbitrary `#id`
//! entries that are not deployment identities) never derive a tenant.
//!
//! Resolution runs through the SAME per-call authority read the admission
//! exchange already performs, and the live-session recheck re-runs this
//! resolver, so a roster change is effective without restart — including
//! closing previously admitted sessions when a subject stops resolving.
//!
//! This mirrors the hosted-account tenant index
//! (`HostedAccountStore::resolve_tenant_for_hosted_did`) and the
//! `MoqConnectAuthz` resolver-closure shape: the tenant never comes from the
//! peer, its proof, or a token — only server-side resolution over verified
//! accepted state. With ≤8 roster entries the per-call read the moql authority
//! already does makes an in-memory index unnecessary. Failure inheritance is
//! identical to authentication: a store read failure is indistinguishable
//! from an unknown identity (`None`), and `None` denies.

use anyhow::Result;
use hyprstream_rpc::moq_authz::PeerIdentity;
use hyprstream_rpc::transport::iroh_moq::PeerTenantResolver;
use hyprstream_rpc::transport::moql_admission::{
    AcceptedIdentityState, AcceptedStateAuthority, MoqlAdmissionProof,
};
use std::sync::Arc;

/// The single tenant of the fixed native Event/Streams namespace. The Event
/// tree is the fixed local namespace (`local/events`); every deployment
/// service identity maps to it and no other tenant is derivable.
pub const LOCAL_TENANT: &str = "local";

/// The deployment service name an accepted state proves, if any.
///
/// Capsule `#<name>` service entries ∩ the registered factory roster
/// (`hyprstream_service::get_factory`), which must match exactly ONE service.
/// Zero matches — a foreign record's arbitrary `#id` entries, or a capsule
/// with no service entries — and ambiguous matches both resolve to `None`:
/// deny, never a best-effort pick (the offline roster validator's uniqueness
/// `ensure!`, applied per admission lookup).
pub fn derived_service_name(state: &AcceptedIdentityState) -> Option<&str> {
    let mut matched = None;
    for id in &state.service_ids {
        let Some(name) = id.strip_prefix('#') else {
            continue;
        };
        if hyprstream_service::get_factory(name).is_some() {
            if matched.is_some() {
                return None;
            }
            matched = Some(name);
        }
    }
    matched
}

/// DID → deployment service name through the live authority. A store read
/// failure maps to `None` (the production projection's fail-closed
/// inheritance): no state, no name, no grant.
pub fn authority_service_name(
    authority: &Arc<dyn AcceptedStateAuthority>,
    did: &str,
) -> Option<String> {
    authority
        .accepted_state(did)
        .as_ref()
        .and_then(derived_service_name)
        .map(str::to_owned)
}

/// DID → derived tenant through the live authority: [`LOCAL_TENANT`] iff the
/// subject is a current deployment service identity, `None` otherwise.
pub fn authority_tenant(authority: &Arc<dyn AcceptedStateAuthority>, did: &str) -> Option<String> {
    authority_service_name(authority, did).map(|_| LOCAL_TENANT.to_owned())
}

/// The admission tenant resolver: tenant [`LOCAL_TENANT`] iff the verified
/// subject is a current deployment service identity.
///
/// The peer is only ever the ALREADY-VERIFIED admission subject; the tenant is
/// resolved server-side from live accepted state, never read from the peer or
/// its proof. Re-ridden by the live-session recheck, so roster removal or
/// service-name reassignment closes previously admitted sessions.
pub fn derived_tenant_resolver(authority: Arc<dyn AcceptedStateAuthority>) -> PeerTenantResolver {
    Arc::new(move |peer: &PeerIdentity| {
        peer.subject
            .as_deref()
            .and_then(|did| authority_tenant(&authority, did))
    })
}

/// Spawn-time self-binding (#1652): the authority must resolve the process's
/// OWN admission proof to exactly this service. Refusal propagates and the
/// service never spawns — the fail-closed shape of the former empty-tenant-map
/// refusal, now "the checkpoint store must bind my DID to my name". The store
/// is guaranteed present at spawn (the accepted-state source refuses to open
/// without it), so failure here is a provisioning mismatch, not a degraded
/// mode to run in.
pub fn require_service_self_binding(
    authority: &Arc<dyn AcceptedStateAuthority>,
    service_name: &str,
    proof: &MoqlAdmissionProof,
) -> Result<()> {
    let resolved = authority_service_name(authority, &proof.did);
    anyhow::ensure!(
        resolved.as_deref() == Some(service_name),
        "native {} identity {} does not resolve to accepted deployment service '{}' in the checkpoint store",
        service_name,
        proof.did,
        service_name
    );
    Ok(())
}

/// Client-side membership self-check: the process's own proof must resolve to
/// SOME deployment service. CLI/embedded processes bootstrap with one of the
/// deployment identities (whichever the process was provisioned as), so —
/// like the former `require_event_tenant`, which accepted any local-bound DID
/// — the check is roster membership, not equality with a specific service.
pub fn require_admitted_service_binding(
    authority: &Arc<dyn AcceptedStateAuthority>,
    proof: &MoqlAdmissionProof,
) -> Result<()> {
    anyhow::ensure!(
        authority_service_name(authority, &proof.did).is_some(),
        "Event identity {} is not an accepted deployment service identity in the checkpoint store",
        proof.did
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use super::*;
    use hyprstream_rpc::transport::moql_admission::AcceptedSubjectKey;
    use std::collections::BTreeMap;

    /// The factory inventory (`hyprstream_service::get_factory`) is populated
    /// by `#[service_factory]` submissions in the APP crate, which this crate
    /// cannot link (the app depends on us). This test binary therefore
    /// registers its own admission-only roster: factory functions that are
    /// never called — `derived_service_name` consults only the names. The
    /// submissions are visible to every test in this binary, including the
    /// `moql_checkpoint_admission_tests` fixtures, whose capsule service
    /// entries name "streams" and "reader".
    fn admission_only_test_factory(
        _ctx: &hyprstream_service::ServiceContext,
    ) -> anyhow::Result<Box<dyn hyprstream_service::Spawnable>> {
        anyhow::bail!("admission-only test factory is never spawned")
    }

    inventory::submit! {
        hyprstream_service::ServiceFactory::new("event", admission_only_test_factory)
    }
    inventory::submit! {
        hyprstream_service::ServiceFactory::new("streams", admission_only_test_factory)
    }
    inventory::submit! {
        hyprstream_service::ServiceFactory::new("model", admission_only_test_factory)
    }
    inventory::submit! {
        hyprstream_service::ServiceFactory::new("policy", admission_only_test_factory)
    }
    inventory::submit! {
        hyprstream_service::ServiceFactory::new("reader", admission_only_test_factory)
    }

    fn state(service_ids: &[&str]) -> AcceptedIdentityState {
        AcceptedIdentityState {
            epoch: 1,
            head_digest: [1; 64],
            subject_keys: vec![AcceptedSubjectKey {
                ed25519: [2; 32],
                ml_dsa_65: vec![3; 1952],
            }],
            service_ids: service_ids.iter().map(ToString::to_string).collect(),
            expires_at_unix_ms: None,
        }
    }

    fn peer(did: &str) -> PeerIdentity {
        PeerIdentity::authenticated(did.to_owned())
    }

    #[test]
    fn factory_service_entries_resolve_and_foreign_entries_do_not() {
        // A deployment-shaped capsule: exactly one factory service entry.
        assert_eq!(derived_service_name(&state(&["#event"])), Some("event"));
        assert_eq!(derived_service_name(&state(&["#streams"])), Some("streams"));
        // Foreign public-RPC records carry arbitrary #id entries.
        assert_eq!(derived_service_name(&state(&["#ns"])), None);
        // Unknown factory names never match, with or without foreign siblings.
        assert_eq!(derived_service_name(&state(&["#bogus"])), None);
        assert_eq!(derived_service_name(&state(&["#ns", "#bogus"])), None);
        // No service entries at all.
        assert_eq!(derived_service_name(&state(&[])), None);
        // Ambiguous: a capsule matching two factory services is not
        // deployment-shaped; deny rather than pick.
        assert_eq!(derived_service_name(&state(&["#event", "#streams"])), None);
        // Non-# entries are skipped, not matched.
        assert_eq!(derived_service_name(&state(&["event"])), None);
    }

    #[test]
    fn tenant_resolver_derives_local_only_from_live_state() {
        let did = "did:at9p:abc";
        let foreign = "did:at9p:foreign";
        let states: Arc<parking_lot::RwLock<BTreeMap<String, AcceptedIdentityState>>> =
            Arc::new(parking_lot::RwLock::new(BTreeMap::from([
                (did.to_owned(), state(&["#event"])),
                (foreign.to_owned(), state(&["#ns"])),
            ])));
        let authority: Arc<dyn AcceptedStateAuthority> = {
            let states = Arc::clone(&states);
            Arc::new(move |lookup: &str| states.read().get(lookup).cloned())
        };
        let resolver = derived_tenant_resolver(Arc::clone(&authority));
        assert_eq!(resolver(&peer(did)).as_deref(), Some(LOCAL_TENANT));
        // Foreign record present in the store but not a factory service: deny.
        assert_eq!(resolver(&peer(foreign)), None);
        // Unknown DID: deny.
        assert_eq!(resolver(&peer("did:at9p:unknown")), None);

        // THE CHURN CASE (#1652): the same service name is re-bound to a NEW
        // DID in the live store (as at an R1 store re-init). Admission for the
        // new DID works in the same process, with no restart and no config —
        // impossible under the former DID-keyed config map.
        let successor = "did:at9p:successor";
        states.write().remove(did);
        states.write().insert(successor.to_owned(), state(&["#event"]));
        assert_eq!(resolver(&peer(did)), None, "superseded DID must stop resolving");
        assert_eq!(
            resolver(&peer(successor)).as_deref(),
            Some(LOCAL_TENANT),
            "successor DID must resolve without restart or config change"
        );
    }

    #[test]
    fn store_read_failure_denies_like_unknown_identity() {
        // The production projection maps source errors to None; a resolver
        // over such an authority inherits deny-all, exactly like UnknownIdentity.
        let authority: Arc<dyn AcceptedStateAuthority> =
            Arc::new(|_did: &str| -> Option<AcceptedIdentityState> { None });
        let resolver = derived_tenant_resolver(Arc::clone(&authority));
        assert_eq!(resolver(&peer("did:at9p:anything")), None);
    }

    #[test]
    fn self_binding_requires_the_services_own_name() {
        let did = "did:at9p:streams";
        let authority: Arc<dyn AcceptedStateAuthority> = Arc::new(move |lookup: &str| {
            (lookup == did).then(|| state(&["#streams"]))
        });
        let proof = MoqlAdmissionProof {
            did: did.to_owned(),
            ed25519: ed25519_dalek::SigningKey::from_bytes(&[4; 32]),
            ml_dsa_65: hyprstream_rpc::crypto::pq::ml_dsa_sk_from_seed(&[5; 32]),
            expected_server: hyprstream_rpc::stream_info::MoqlServerIdentity {
                did: did.to_owned(),
                epoch: 1,
                head_digest: vec![1; 64],
                expires_at_unix_ms: i64::MAX,
                ed25519: [2; 32],
                ml_dsa65: vec![3; 1952],
            },
        };
        require_service_self_binding(&authority, "streams", &proof)
            .expect("own service name must bind");
        let mismatch = require_service_self_binding(&authority, "event", &proof)
            .expect_err("a different service name must refuse");
        assert!(
            mismatch.to_string().contains("does not resolve"),
            "unexpected error: {mismatch}"
        );
        // Client-side membership form accepts the deployment identity itself.
        require_admitted_service_binding(&authority, &proof)
            .expect("membership self-check must accept a deployment service DID");
    }
}
