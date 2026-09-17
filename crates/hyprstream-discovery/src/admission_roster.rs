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
//! DID → accepted state → capsule `#<name>` service entry ∩ the registered
//! factory roster, held UNIQUELY across the store ⇒ tenant [`LOCAL_TENANT`];
//! everything else ⇒ unresolved, deny.
//!
//! # Why unique-holder, not name-match alone
//!
//! The store is not a pure deployment roster: the public `ingestAt9pCandidate`
//! RPC admits foreign, self-certifying capsules whose `#id` entries are
//! arbitrary strings, and provisioned and ingested states share one write
//! path — provenance is not distinguishable in the store. A foreign capsule
//! that NAMES a factory service (e.g. `#event`) therefore cannot be told
//! apart from a deployment identity by its content. The derivation closes
//! that self-selection the only way store data allows:
//!
//! - a name admits a tenant only while EXACTLY ONE accepted state in the
//!   whole store derives it — beside the genuine identity a squatter makes
//!   the name ambiguous, and ambiguity denies (fail-closed, loud: the
//!   collision is warned so an operator sees the injected record). This is
//!   the offline roster validator's cross-state uniqueness `ensure!`, and
//!   `project_bootstrap_endpoint`'s ambiguity rule, applied per lookup;
//! - the deriving state must carry the deployment invariants the offline
//!   provisioner guarantees and foreign genesis-only records lack: successor
//!   epoch (`epoch > 0`) and a bounded expiry that has not lapsed (an expired
//!   state — genuine or foreign — never derives and never counts toward
//!   uniqueness).
//!
//! Residual (deliberate, flagged in #1652): if a deployment name has NO
//! genuine accepted state (decommissioned service) and an attacker ingests a
//! capsule claiming it, that name derives for the attacker. Operating with
//! names in `quic.event_publishers` that have no live deployment state is a
//! deployment inconsistency this PR refuses at Event spawn
//! ([`require_publisher_roster`]) and the tenant grant remains bounded to the
//! fixed local namespace. Fully distinguishing provenance needs a
//! provisioner-signed roster the admission path can verify — a maintainer
//! design decision recorded in #1652, not something this change invents.
//!
//! Resolution enumerates the store's verified states through the same
//! verification as the per-DID admission read, behind a TTL cache
//! ([`CachedRoster`]): the live-session recheck runs every 100 ms per session,
//! so the polling path serves a bounded-stale snapshot instead of re-verifying
//! the whole store at 10 Hz per session. Revocation by state REMOVAL stays
//! immediate (the per-DID admission authority recheck is uncached); the cache
//! bounds only how quickly roster-side changes — churn re-binding, name
//! reassignment, squatter arrival — become visible (≤ TTL).
//!
//! This mirrors the hosted-account tenant index
//! (`HostedAccountStore::resolve_tenant_for_hosted_did`) and the
//! `MoqConnectAuthz` resolver-closure shape: the tenant never comes from the
//! peer, its proof, or a token — only server-side resolution over verified
//! accepted state. Failure inheritance is identical to authentication: a
//! store read failure yields an empty roster, and an empty roster denies.

use anyhow::Result;
use hyprstream_rpc::moq_authz::PeerIdentity;
use hyprstream_rpc::transport::iroh_moq::PeerTenantResolver;
use hyprstream_rpc::transport::moql_admission::{AcceptedIdentityState, MoqlAdmissionProof};
use std::sync::Arc;

/// The single tenant of the fixed native Event/Streams namespace. The Event
/// tree is the fixed local namespace (`local/events`); every deployment
/// service identity maps to it and no other tenant is derivable.
pub const LOCAL_TENANT: &str = "local";

/// One projected accepted state, keyed by its DID: the roster universe the
/// derivation enumerates (provisioned deployment identities and foreign
/// ingested records alike).
pub type RosterEntry = (String, AcceptedIdentityState);

/// The deployment-roster side of the accepted-state authority: enumerates
/// every verified accepted state so grant derivation can enforce the
/// cross-store uniqueness a per-DID read cannot see. Implemented by the
/// process's pinned checkpoint source; a blanket `Fn` impl covers fixtures.
pub trait DeploymentRosterSource: Send + Sync {
    fn roster(&self) -> Vec<RosterEntry>;
}

impl<F> DeploymentRosterSource for F
where
    F: Fn() -> Vec<RosterEntry> + Send + Sync,
{
    fn roster(&self) -> Vec<RosterEntry> {
        self()
    }
}

/// TTL-bounded cached roster projection (the
/// `resolve_tenant_for_hosted_did` shape): lookups serve a snapshot for at
/// most [`ROSTER_CACHE_TTL`], then one refresh runs while concurrent lookups
/// keep serving the current snapshot (single-flight under the cache mutex).
///
/// A refresh that fails loads an EMPTY roster — fail-closed, exactly like an
/// uncached read failure — and a previously loaded snapshot is never
/// resurrected after a failure (that would delay revocation beyond the TTL).
/// Spawn-time checks ride the same cache; at process start it is empty, so
/// the first lookup is a fresh load.
pub struct CachedRoster {
    load: Arc<dyn Fn() -> Vec<RosterEntry> + Send + Sync>,
    snapshot: parking_lot::Mutex<Option<(std::time::Instant, Arc<Vec<RosterEntry>>)>>,
    ttl: std::time::Duration,
}

/// How long a roster snapshot is served before the next lookup refreshes it.
/// Bounds the visibility delay of roster-side changes (churn, reassignment,
/// squatter arrival); state removal is caught immediately by the uncached
/// per-DID authority recheck regardless.
pub const ROSTER_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(1);

impl CachedRoster {
    pub fn new(
        load: Arc<dyn Fn() -> Vec<RosterEntry> + Send + Sync>,
        ttl: std::time::Duration,
    ) -> Self {
        Self {
            load,
            snapshot: parking_lot::Mutex::new(None),
            ttl,
        }
    }

    fn current(&self) -> Arc<Vec<RosterEntry>> {
        let mut guard = self.snapshot.lock();
        if let Some((refreshed_at, entries)) = guard.as_ref() {
            if refreshed_at.elapsed() <= self.ttl {
                return Arc::clone(entries);
            }
        }
        // Single-flight: everyone races to this mutex; the loser re-checks
        // the snapshot after acquiring and serves the winner's refresh.
        let entries = Arc::new((self.load)());
        *guard = Some((std::time::Instant::now(), Arc::clone(&entries)));
        entries
    }
}

impl DeploymentRosterSource for CachedRoster {
    fn roster(&self) -> Vec<RosterEntry> {
        self.current().as_ref().clone()
    }
}

/// The deployment service name an accepted state proves, if any.
///
/// Capsule `#<name>` service entries ∩ the registered factory roster
/// (`hyprstream_service::get_factory`), which must match exactly ONE service,
/// and the state must carry the deployment invariants (successor epoch > 0
/// and a bounded expiry — the offline provisioner always stacks a bounded
/// successor, so genesis-only/unbounded states are not deployment-shaped).
/// Zero matches — foreign `#id` entries, no service entries — ambiguous
/// matches, and invariant-violating states all resolve to `None`: deny,
/// never a best-effort pick.
pub fn derived_service_name(state: &AcceptedIdentityState, now_unix_ms: i64) -> Option<&str> {
    // Live expiry is enforced here, not just presence: an expired state is
    // rejected by the admission exchange, and it must not keep participating
    // in name derivation or uniqueness counting either — otherwise an
    // expired foreign claim would make the genuine identity ambiguously
    // denied forever (review round 2).
    if state.epoch == 0
        || state.expires_at_unix_ms.is_none()
        || !state.is_live(now_unix_ms)
    {
        return None;
    }
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

/// Resolve `did` to its deployment service name against the live roster,
/// enforcing unique holdership: the DID's state must derive a name, and no
/// OTHER state may derive the same name. A collision is warned loudly —
/// an ingested record squatting a deployment name is an operator-visible
/// event, not a silent deny.
pub fn roster_service_name(roster: &[RosterEntry], did: &str, now_unix_ms: i64) -> Option<String> {
    let name = derived_service_name(&roster.iter().find(|(d, _)| d == did)?.1, now_unix_ms)?;
    let holders = roster
        .iter()
        .filter(|(_, state)| derived_service_name(state, now_unix_ms) == Some(name))
        .count();
    if holders != 1 {
        tracing::warn!(
            service = name,
            holders,
            "accepted-state store holds multiple identities deriving service; denying until unique"
        );
        return None;
    }
    Some(name.to_owned())
}

/// DID → derived tenant against the live roster: [`LOCAL_TENANT`] iff the
/// subject uniquely proves a deployment service identity.
pub fn roster_tenant(roster: &[RosterEntry], did: &str, now_unix_ms: i64) -> Option<String> {
    roster_service_name(roster, did, now_unix_ms).map(|_| LOCAL_TENANT.to_owned())
}

/// The admission tenant resolver: tenant [`LOCAL_TENANT`] iff the verified
/// subject is a current, uniquely-held deployment service identity.
///
/// The peer is only ever the ALREADY-VERIFIED admission subject; the tenant is
/// resolved server-side from live accepted state, never read from the peer or
/// its proof. Re-ridden by the live-session recheck, so roster removal,
/// service-name reassignment, or a new ambiguous squatter closes previously
/// admitted sessions.
pub fn derived_tenant_resolver(roster: Arc<dyn DeploymentRosterSource>) -> PeerTenantResolver {
    Arc::new(move |peer: &PeerIdentity| {
        peer.subject.as_deref().and_then(|did| {
            roster_tenant(&roster.roster(), did, hyprstream_rpc::envelope::current_timestamp())
        })
    })
}

/// Spawn-time self-binding (#1652): the live roster must resolve the
/// process's OWN admission proof to exactly this service. Refusal propagates
/// and the service never spawns — the fail-closed shape of the former
/// empty-tenant-map refusal, now "the checkpoint store must bind my DID to my
/// name". The store is guaranteed present at spawn (the accepted-state source
/// refuses to open without it), so failure here is a provisioning mismatch,
/// not a degraded mode to run in.
pub fn require_service_self_binding(
    roster: &Arc<dyn DeploymentRosterSource>,
    service_name: &str,
    proof: &MoqlAdmissionProof,
) -> Result<()> {
    let resolved = roster_service_name(&roster.roster(), &proof.did, hyprstream_rpc::envelope::current_timestamp());
    anyhow::ensure!(
        resolved.as_deref() == Some(service_name),
        "native {} identity {} does not uniquely resolve to accepted deployment service '{}' in the checkpoint store",
        service_name,
        proof.did,
        service_name
    );
    Ok(())
}

/// Client-side membership self-check: the process's own proof must resolve to
/// SOME uniquely-held deployment service. CLI/embedded processes bootstrap
/// with one of the deployment identities (whichever the process was
/// provisioned as), so — like the former `require_event_tenant`, which
/// accepted any local-bound DID — the check is roster membership, not
/// equality with a specific service.
pub fn require_admitted_service_binding(
    roster: &Arc<dyn DeploymentRosterSource>,
    proof: &MoqlAdmissionProof,
) -> Result<()> {
    anyhow::ensure!(
        roster_service_name(&roster.roster(), &proof.did, hyprstream_rpc::envelope::current_timestamp())
            .is_some(),
        "Event identity {} is not a uniquely-held accepted deployment service identity in the checkpoint store",
        proof.did
    );
    Ok(())
}

/// Spawn-time Event publisher roster check (#1652): every name in
/// `quic.event_publishers` must be carried by a live accepted state. This
/// catches deployment misconfiguration — a configured publisher name with no
/// state at all (typo, decommissioned service) — by refusing to spawn. It
/// does NOT establish provenance: a sole foreign claimant of an unheld name
/// satisfies it exactly as it satisfies the derivation, which is the
/// maintainer provenance-binding decision recorded in #1652 (see the module
/// doc residual); closing that here is not possible from store data alone.
pub fn require_publisher_roster(
    roster: &Arc<dyn DeploymentRosterSource>,
    publishers: &std::collections::BTreeSet<String>,
) -> Result<()> {
    let live = roster.roster();
    let now = hyprstream_rpc::envelope::current_timestamp();
    for name in publishers {
        anyhow::ensure!(
            live.iter()
                .any(|(_, state)| derived_service_name(state, now) == Some(name.as_str())),
            "quic.event_publishers lists '{}' but no live accepted state uniquely carries that service; refusing to run with a squat-able grant",
            name
        );
    }
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
            service_ids: service_ids.iter().map(|id| (*id).to_owned()).collect(),
            expires_at_unix_ms: Some(i64::MAX),
        }
    }

    fn peer(did: &str) -> PeerIdentity {
        PeerIdentity::authenticated(did.to_owned())
    }

    #[test]
    fn factory_service_entries_resolve_and_foreign_entries_do_not() {
        let now = hyprstream_rpc::envelope::current_timestamp();
        // A deployment-shaped capsule: exactly one factory service entry,
        // successor epoch, bounded unexpired expiry.
        assert_eq!(derived_service_name(&state(&["#event"]), now), Some("event"));
        assert_eq!(derived_service_name(&state(&["#streams"]), now), Some("streams"));
        // Foreign public-RPC records carry arbitrary #id entries.
        assert_eq!(derived_service_name(&state(&["#ns"]), now), None);
        // Unknown factory names never match, with or without foreign siblings.
        assert_eq!(derived_service_name(&state(&["#bogus"]), now), None);
        assert_eq!(derived_service_name(&state(&["#ns", "#bogus"]), now), None);
        // No service entries at all.
        assert_eq!(derived_service_name(&state(&[]), now), None);
        // Ambiguous: a capsule matching two factory services is not
        // deployment-shaped; deny rather than pick.
        assert_eq!(derived_service_name(&state(&["#event", "#streams"]), now), None);
        // Non-# entries are skipped, not matched.
        assert_eq!(derived_service_name(&state(&["event"]), now), None);
        // Deployment invariants: genesis (epoch 0) and unbounded states never
        // derive — foreign genesis-only ingests are not deployment-shaped.
        let mut genesis = state(&["#event"]);
        genesis.epoch = 0;
        assert_eq!(derived_service_name(&genesis, now), None);
        let mut unbounded = state(&["#event"]);
        unbounded.expires_at_unix_ms = None;
        assert_eq!(derived_service_name(&unbounded, now), None);
        // EXPIRED states never derive and never count toward uniqueness
        // (review round 2): an expired foreign claim must not keep the
        // genuine identity ambiguously denied.
        let mut expired = state(&["#event"]);
        expired.expires_at_unix_ms = Some(now - 1);
        assert_eq!(derived_service_name(&expired, now), None);
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
        let roster: Arc<dyn DeploymentRosterSource> = {
            let states = Arc::clone(&states);
            Arc::new(move || states.read().clone().into_iter().collect::<Vec<_>>())
        };
        let resolver = derived_tenant_resolver(Arc::clone(&roster));
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
    fn name_squatting_is_denied_not_escalated() {
        let now = hyprstream_rpc::envelope::current_timestamp();
        // A foreign record claiming a factory service name BESIDE the genuine
        // identity: the name becomes ambiguous and BOTH derive nothing —
        // fail-closed. This is the property a name-only filter lacks.
        let genuine = "did:at9p:genuine-model";
        let squatter = "did:at9p:foreign-model";
        let roster: Vec<RosterEntry> = vec![
            (genuine.to_owned(), state(&["#model"])),
            (squatter.to_owned(), state(&["#model"])),
        ];
        assert_eq!(roster_tenant(&roster, genuine, now), None);
        assert_eq!(roster_tenant(&roster, squatter, now), None);
        // With the squatter gone the genuine identity derives again.
        let clean: Vec<RosterEntry> = vec![(genuine.to_owned(), state(&["#model"]))];
        assert_eq!(roster_tenant(&clean, genuine, now).as_deref(), Some(LOCAL_TENANT));
        // An EXPIRED squatter never counts: the genuine identity keeps
        // deriving while an expired foreign claim sits in the store.
        let mut expired_squat = state(&["#model"]);
        expired_squat.expires_at_unix_ms = Some(now - 1);
        let with_expired: Vec<RosterEntry> = vec![
            (genuine.to_owned(), state(&["#model"])),
            (squatter.to_owned(), expired_squat),
        ];
        assert_eq!(
            roster_tenant(&with_expired, genuine, now).as_deref(),
            Some(LOCAL_TENANT),
            "expired foreign claim must not make the genuine identity ambiguous"
        );
    }

    #[test]
    fn store_read_failure_denies_like_unknown_identity() {
        // The production roster maps source errors to an empty roster; a
        // resolver over such a source inherits deny-all, exactly like
        // UnknownIdentity.
        let roster: Arc<dyn DeploymentRosterSource> = Arc::new(Vec::new);
        let resolver = derived_tenant_resolver(roster);
        assert_eq!(resolver(&peer("did:at9p:anything")), None);
    }

    #[test]
    fn self_binding_requires_the_services_own_name() {
        let did = "did:at9p:streams";
        let roster: Arc<dyn DeploymentRosterSource> =
            Arc::new(move || vec![(did.to_owned(), state(&["#streams"]))]);
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
        require_service_self_binding(&roster, "streams", &proof)
            .expect("own service name must bind");
        let mismatch = require_service_self_binding(&roster, "event", &proof)
            .expect_err("a different service name must refuse");
        assert!(
            mismatch.to_string().contains("does not uniquely resolve"),
            "unexpected error: {mismatch}"
        );
        // Client-side membership form accepts the deployment identity itself.
        require_admitted_service_binding(&roster, &proof)
            .expect("membership self-check must accept a deployment service DID");
    }

    #[test]
    fn cached_roster_serves_within_ttl_and_refreshes_after() {
        let states: Arc<parking_lot::RwLock<Vec<RosterEntry>>> =
            Arc::new(parking_lot::RwLock::new(vec![(
                "did:at9p:event".to_owned(),
                state(&["#event"]),
            )]));
        let load = {
            let states = Arc::clone(&states);
            Arc::new(move || states.read().clone()) as Arc<dyn Fn() -> Vec<RosterEntry> + Send + Sync>
        };
        let now = hyprstream_rpc::envelope::current_timestamp();
        // Zero TTL: every lookup refreshes — churn is visible immediately.
        let hot = CachedRoster::new(Arc::clone(&load), std::time::Duration::ZERO);
        assert_eq!(
            roster_tenant(&hot.roster(), "did:at9p:event", now).as_deref(),
            Some(LOCAL_TENANT)
        );
        states.write().clear();
        assert_eq!(hot.roster().len(), 0, "zero-TTL cache must observe the wipe");
        // Long TTL: the snapshot is served as-is until it expires — the
        // polling recheck path trades bounded staleness for not re-verifying
        // the whole store at 10 Hz per session.
        states.write().push(("did:at9p:event".to_owned(), state(&["#event"])));
        let cold = CachedRoster::new(Arc::clone(&load), std::time::Duration::from_secs(3600));
        assert_eq!(cold.roster().len(), 1);
        states.write().clear();
        assert_eq!(cold.roster().len(), 1, "long-TTL cache must keep serving the snapshot");
    }

    #[test]
    fn publisher_roster_refuses_names_without_a_live_identity() {
        let did = "did:at9p:event";
        let roster: Arc<dyn DeploymentRosterSource> =
            Arc::new(move || vec![(did.to_owned(), state(&["#event"]))]);
        require_publisher_roster(
            &roster,
            &["event".to_owned(), "model".to_owned()].into_iter().collect(),
        )
        .expect_err("a configured publisher with no live state must refuse spawn");
        require_publisher_roster(&roster, &["event".to_owned()].into_iter().collect())
            .expect("a carried publisher name must pass");
    }
}
