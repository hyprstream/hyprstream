//! Discovery service — exposes EndpointRegistry over RPC.
//!
//! Allows remote clients to discover registered services, their endpoints,
//! socket kinds, and schemas via the standard REQ/REP transport.

use async_trait::async_trait;
use hyprstream_rpc::browser_provisioning::{
    BrowserCarrierProfile, BrowserCurrentnessVerifier, BrowserProvisioningDocument,
    BrowserProvisioningMaterial, BrowserProvisioningRequest, BrowserRequestBinding,
    BrowserRouteRole, BrowserTransportSecurity,
};
use hyprstream_rpc::registry::{self, EndpointRegistry, SocketKind};
use hyprstream_rpc::resolver::{Resolver, ResolverProfile, ServiceQuery};
use hyprstream_rpc::rpc_client::{CallOptions, RpcClient};
use hyprstream_rpc::service::{EnvelopeContext, RequestService};
use hyprstream_rpc::stream_consumer::StreamHandle;
use hyprstream_rpc::transport::{EndpointType, TransportConfig};
use hyprstream_rpc::{SigningKey, VerifyingKey};

use crate::generated::discovery_client::{
    dispatch_discovery, serialize_response, AuthMetadata, AuthMetadataList, DiscoveryHandler,
    DiscoveryResponseVariant, EndpointInfo, EntityStatement, EnvelopeKeyset, ErrorInfo,
    GetRecordRequest, IssuerList, NodeLiveness, PingInfo, PlacementCandidate,
    PlacementCandidateSet, QueryCandidatesRequest, RecordCar, RegisterEntityStatementRequest,
    RegisterEnvelopeKeysetRequest, Resource, ServiceAnnouncement, ServiceEndpoints, ServiceList,
    ServiceSummary,
};
use crate::placement_index::PlacementIndex;
use crate::scheduling;

use anyhow::{Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hyprstream_rpc::identity::Did;
use hyprstream_util::ttl_cache::TtlCache;
use parking_lot::RwLock;
use std::collections::{BTreeSet, HashMap};
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, trace};

/// #524 P1 — liveness heartbeat TTL: a node with no live/fresh
/// `reportNodeLiveness` entry is hard-excluded from `queryCandidates`
/// (ratified decision — see the epic's issue comments), never just flagged
/// stale. 45s sits mid-range of the 30-60s window floated in earlier design
/// notes; named here so it's easy to retune.
const LIVENESS_TTL: Duration = Duration::from_secs(45);

/// Liveness cache capacity / inline-reap budget. Generous fleet-scale
/// headroom; eviction is O(log n) and happens inline on every insert/get, so a
/// large bound doesn't cost anything until it's actually full.
const LIVENESS_CACHE_MAX_ENTRIES: usize = 16_384;
const LIVENESS_CACHE_REAP_BUDGET: usize = 32;
/// Bound retry work for an admitted DID whose repo is absent, invalid, or does
/// not yet contain a node record. This prevents heartbeat-rate resolver polls
/// while allowing eventual recovery when a placement record is later published.
const PLACEMENT_INGEST_RETRY_TTL: Duration = Duration::from_secs(300);
const ANNOUNCED_ENDPOINT_TTL: Duration = Duration::from_secs(90);

/// Default bound applied to `queryCandidates` when the caller passes
/// `maxCandidates == 0` (unspecified) — keeps an unscoped query from returning
/// the entire fleet in one response.
const DEFAULT_MAX_CANDIDATES: usize = 100;

/// One node's live allocatable capacity + load, as reported via
/// `reportNodeLiveness`. Stored in a `TtlCache<Did, _>` — absence (never
/// reported, or expired) hard-excludes the node from `queryCandidates`.
#[derive(Clone, Debug)]
struct LiveAllocatable {
    /// resource name -> k8s-quantity, free right now.
    allocatable: Vec<(String, String)>,
    load_fraction: f32,
    /// unix millis of this snapshot.
    last_seen: i64,
}

fn unix_millis_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Private checkpoint-bound projection used only while Discovery validates an
/// announcement against the daemon-owned accepted-state source.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AcceptedStateEvidence {
    service_did: Did,
    digest: [u8; 64],
    epoch: u64,
    expires_at_unix_ms: i64,
    response_ed25519: [u8; 32],
    response_ml_dsa65: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AnchoredKemRecipient {
    key_id: String,
    recipient: hyprstream_rpc::crypto::hybrid_kem::RecipientPublic,
    not_after_unix_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServiceCandidate {
    service_name: String,
    service_did: Did,
    response_verifying_key: [u8; 32],
    response_ml_dsa65: Vec<u8>,
    response_key_id: String,
    request_kem_recipient: Option<AnchoredKemRecipient>,
    transport: TransportConfig,
    capabilities: BTreeSet<String>,
    accepted_state: AcceptedStateEvidence,
    source_signer: [u8; 32],
    expires_at_unix_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CandidateDecision {
    candidate_fingerprint: String,
    accepted: bool,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolutionEvidence {
    accepted_state_digest: [u8; 64],
    accepted_state_epoch: u64,
    selected_fingerprint: String,
    ordered_decisions: Vec<CandidateDecision>,
}

/// Private raw selection result. It can become production authority only via
/// `ResolvedService::mint`, which requires the typed accepted-state witness.
#[derive(Debug, Clone)]
struct SelectedService {
    service_name: String,
    service_did: Did,
    response_verifying_key: VerifyingKey,
    response_ml_dsa65: Vec<u8>,
    response_key_id: String,
    request_kem_recipient: AnchoredKemRecipient,
    transport: TransportConfig,
    expires_at_unix_ms: i64,
    evidence: ResolutionEvidence,
}

impl SelectedService {
    fn same_authority(&self, other: &Self) -> bool {
        self.service_name == other.service_name
            && self.service_did == other.service_did
            && self.response_verifying_key == other.response_verifying_key
            && self.response_ml_dsa65 == other.response_ml_dsa65
            && self.response_key_id == other.response_key_id
            && self.request_kem_recipient == other.request_kem_recipient
            && self.evidence.accepted_state_digest == other.evidence.accepted_state_digest
            && self.evidence.accepted_state_epoch == other.evidence.accepted_state_epoch
    }
    fn service_name(&self) -> &str {
        &self.service_name
    }

    fn service_did(&self) -> &Did {
        &self.service_did
    }

    fn response_verifying_key(&self) -> VerifyingKey {
        self.response_verifying_key
    }

    fn response_ml_dsa65(&self) -> &[u8] {
        &self.response_ml_dsa65
    }

    fn response_key_id(&self) -> &str {
        &self.response_key_id
    }
    fn request_kem_recipient(&self) -> &AnchoredKemRecipient {
        &self.request_kem_recipient
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn evidence(&self) -> &ResolutionEvidence {
        &self.evidence
    }

    fn expires_at_unix_ms(&self) -> i64 {
        self.expires_at_unix_ms
    }

    fn ensure_fresh(&self, now_unix_ms: i64) -> Result<()> {
        anyhow::ensure!(
            now_unix_ms < self.expires_at_unix_ms,
            "resolved service snapshot expired; re-resolution required"
        );
        anyhow::ensure!(
            now_unix_ms < self.request_kem_recipient.not_after_unix_ms,
            "resolved request KEM recipient expired; re-resolution required"
        );
        Ok(())
    }

    fn crypto_stores(
        &self,
    ) -> Result<(
        Arc<dyn hyprstream_rpc::crypto::hybrid_kem::KemTrustStore>,
        Arc<dyn hyprstream_rpc::envelope::PqTrustStore>,
    )> {
        let ed = self.response_verifying_key.to_bytes();
        let mut kem = hyprstream_rpc::crypto::hybrid_kem::KeyedKemTrustStore::new();
        kem.bind(ed, self.request_kem_recipient.recipient.clone());
        let pq_key = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&self.response_ml_dsa65)
            .map_err(|e| anyhow::anyhow!("invalid resolved ML-DSA-65 response key: {e}"))?;
        let mut pq = hyprstream_rpc::envelope::KeyedPqTrustStore::new();
        pq.bind(ed, &pq_key);
        Ok((Arc::new(kem), Arc::new(pq)))
    }
}

fn canonical_service_name(name: &str) -> Result<String> {
    let canonical = name.trim().to_ascii_lowercase();
    anyhow::ensure!(!canonical.is_empty(), "service name is empty");
    anyhow::ensure!(
        canonical
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
        "service name contains non-canonical characters"
    );
    anyhow::ensure!(canonical == name, "service name is not canonical");
    Ok(canonical)
}

fn network_reach(transport: &TransportConfig) -> bool {
    matches!(
        transport.endpoint,
        EndpointType::Quic { .. } | EndpointType::Iroh { .. }
    )
}

fn transport_fingerprint(transport: &TransportConfig) -> String {
    blake3::hash(format!("{transport:?}").as_bytes())
        .to_hex()
        .to_string()
}

fn hash_framed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn candidate_fingerprint(candidate: &ServiceCandidate) -> String {
    let mut h = blake3::Hasher::new();
    hash_framed(&mut h, candidate.service_name.as_bytes());
    hash_framed(&mut h, candidate.service_did.as_str().as_bytes());
    h.update(&candidate.response_verifying_key);
    hash_framed(&mut h, &candidate.response_ml_dsa65);
    hash_framed(&mut h, candidate.response_key_id.as_bytes());
    if let Some(kem) = &candidate.request_kem_recipient {
        hash_framed(&mut h, kem.key_id.as_bytes());
        hash_framed(&mut h, &kem.recipient.encode());
    }
    hash_framed(
        &mut h,
        transport_fingerprint(&candidate.transport).as_bytes(),
    );
    for capability in &candidate.capabilities {
        hash_framed(&mut h, capability.as_bytes());
    }
    h.update(&candidate.accepted_state.digest);
    h.update(&candidate.accepted_state.epoch.to_be_bytes());
    hash_framed(
        &mut h,
        candidate.accepted_state.service_did.as_str().as_bytes(),
    );
    h.update(&candidate.accepted_state.expires_at_unix_ms.to_be_bytes());
    h.update(&candidate.accepted_state.response_ed25519);
    hash_framed(&mut h, &candidate.accepted_state.response_ml_dsa65);
    h.update(&candidate.source_signer);
    h.update(&candidate.expires_at_unix_ms.to_be_bytes());
    if let Some(kem) = &candidate.request_kem_recipient {
        h.update(&kem.not_after_unix_ms.to_be_bytes());
    }
    h.finalize().to_hex().to_string()
}

fn rejection_reason(
    query: &ServiceQuery,
    candidate: &ServiceCandidate,
    now_unix_ms: i64,
) -> Option<&'static str> {
    if canonical_service_name(&candidate.service_name)
        .ok()
        .as_deref()
        != Some(query.service_name.as_str())
    {
        return Some("service-name-mismatch");
    }
    if !candidate.service_did.is_did_at9p()
        || candidate.service_did != candidate.accepted_state.service_did
    {
        return Some("missing-or-mismatched-accepted-identity");
    }
    if candidate.source_signer != candidate.accepted_state.response_ed25519 {
        return Some("announcement-signer-not-current");
    }
    if candidate.response_verifying_key != candidate.accepted_state.response_ed25519
        || candidate.response_ml_dsa65 != candidate.accepted_state.response_ml_dsa65
    {
        return Some("response-key-not-current");
    }
    if !candidate
        .response_key_id
        .starts_with(&format!("{}#", candidate.service_did))
        || hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&candidate.response_ml_dsa65).is_err()
    {
        return Some("missing-or-invalid-response-key-material");
    }
    if !query
        .required_capabilities
        .is_subset(&candidate.capabilities)
    {
        return Some("missing-capability");
    }
    if candidate.expires_at_unix_ms <= now_unix_ms
        || candidate.accepted_state.expires_at_unix_ms <= now_unix_ms
    {
        return Some("expired-candidate-or-state");
    }
    match query.profile {
        ResolverProfile::NetworkDiscovery if !network_reach(&candidate.transport) => {
            return Some("local-reach-for-network-profile");
        }
        ResolverProfile::LocalInproc
            if !matches!(candidate.transport.endpoint, EndpointType::Inproc { .. }) =>
        {
            return Some("non-inproc-reach-for-local-profile");
        }
        ResolverProfile::Ipc
            if !matches!(
                candidate.transport.endpoint,
                EndpointType::Ipc { .. } | EndpointType::SystemdFd { .. }
            ) =>
        {
            return Some("non-ipc-reach-for-ipc-profile");
        }
        _ => {}
    }
    let Some(kem) = &candidate.request_kem_recipient else {
        return Some("missing-request-kem-recipient");
    };
    if !kem
        .key_id
        .starts_with(&format!("{}#", candidate.service_did))
        || kem.not_after_unix_ms <= now_unix_ms
        || kem.recipient.suite_id
            != hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768
        || kem.recipient.eks.len() != kem.recipient.suite_id.components().len()
        || hyprstream_rpc::crypto::hybrid_kem::RecipientPublic::decode(&kem.recipient.encode())
            .is_err()
    {
        return Some("invalid-expired-or-nonhybrid-kem-recipient");
    }
    None
}

fn select_service_candidate(
    query: &ServiceQuery,
    candidates: Vec<ServiceCandidate>,
    now_unix_ms: i64,
) -> Result<SelectedService> {
    let mut decisions = Vec::with_capacity(candidates.len());
    let mut valid = Vec::new();
    for candidate in candidates {
        let fingerprint = candidate_fingerprint(&candidate);
        if let Some(reason) = rejection_reason(query, &candidate, now_unix_ms) {
            decisions.push(CandidateDecision {
                candidate_fingerprint: fingerprint,
                accepted: false,
                reason: reason.to_owned(),
            });
        } else {
            decisions.push(CandidateDecision {
                candidate_fingerprint: fingerprint.clone(),
                accepted: true,
                reason: "validated".to_owned(),
            });
            valid.push((fingerprint, candidate));
        }
    }
    anyhow::ensure!(!valid.is_empty(), "no validated service candidates");
    let authority = &valid[0].1;
    anyhow::ensure!(
        valid.iter().all(|(_, candidate)| {
            candidate.service_did == authority.service_did
                && candidate.response_verifying_key == authority.response_verifying_key
                && candidate.response_ml_dsa65 == authority.response_ml_dsa65
                && candidate.response_key_id == authority.response_key_id
                && candidate.request_kem_recipient == authority.request_kem_recipient
                && candidate.accepted_state == authority.accepted_state
        }),
        "ambiguous validated service authority"
    );
    valid.sort_by(|(af, a), (bf, b)| {
        b.accepted_state
            .epoch
            .cmp(&a.accepted_state.epoch)
            .then_with(|| b.expires_at_unix_ms.cmp(&a.expires_at_unix_ms))
            .then_with(|| a.service_did.cmp(&b.service_did))
            .then_with(|| {
                transport_fingerprint(&a.transport).cmp(&transport_fingerprint(&b.transport))
            })
            .then_with(|| af.cmp(bf))
    });
    valid.dedup_by(|(af, _), (bf, _)| af == bf);
    let (selected_fingerprint, selected) = valid.remove(0);
    decisions.sort_by(|a, b| a.candidate_fingerprint.cmp(&b.candidate_fingerprint));
    let response_verifying_key = VerifyingKey::from_bytes(&selected.response_verifying_key)
        .map_err(|e| anyhow::anyhow!("invalid selected response verifying key: {e}"))?;
    let request_kem_recipient = selected
        .request_kem_recipient
        .ok_or_else(|| anyhow::anyhow!("validated candidate lost request KEM recipient"))?;
    let expires_at_unix_ms = selected
        .expires_at_unix_ms
        .min(selected.accepted_state.expires_at_unix_ms)
        .min(request_kem_recipient.not_after_unix_ms);
    Ok(SelectedService {
        service_name: selected.service_name,
        service_did: selected.service_did,
        response_verifying_key,
        response_ml_dsa65: selected.response_ml_dsa65,
        response_key_id: selected.response_key_id,
        request_kem_recipient,
        transport: selected.transport,
        expires_at_unix_ms,
        evidence: ResolutionEvidence {
            accepted_state_digest: selected.accepted_state.digest,
            accepted_state_epoch: selected.accepted_state.epoch,
            selected_fingerprint,
            ordered_decisions: decisions,
        },
    })
}

fn select_service_candidates(
    query: &ServiceQuery,
    mut candidates: Vec<ServiceCandidate>,
    now_unix_ms: i64,
) -> Result<Vec<SelectedService>> {
    let mut ordered = Vec::new();
    loop {
        let selected = match select_service_candidate(query, candidates.clone(), now_unix_ms) {
            Ok(selected) => selected,
            Err(error) if !ordered.is_empty() && error.to_string().contains("no validated") => {
                break;
            }
            Err(error) => return Err(error),
        };
        let fingerprint = selected.evidence.selected_fingerprint.clone();
        candidates.retain(|candidate| candidate_fingerprint(candidate) != fingerprint);
        ordered.push(selected);
    }
    Ok(ordered)
}

/// Convert the wire `SelectorOp` (generated from `discovery.capnp`) to the
/// shared scheduling-substrate `SelectorOp` (`crate::scheduling`, #628).
/// Deliberately exhaustive (no catch-all) so a future wire variant fails to
/// compile here instead of silently matching nothing.
fn to_scheduling_op(op: crate::generated::discovery_client::SelectorOp) -> scheduling::SelectorOp {
    use crate::generated::discovery_client::SelectorOp as Wire;
    match op {
        Wire::In => scheduling::SelectorOp::In,
        Wire::NotIn => scheduling::SelectorOp::NotIn,
        Wire::Exists => scheduling::SelectorOp::Exists,
        Wire::DoesNotExist => scheduling::SelectorOp::DoesNotExist,
        Wire::Gt => scheduling::SelectorOp::Gt,
        Wire::Lt => scheduling::SelectorOp::Lt,
    }
}

// ============================================================================
// AuthorizationProvider trait (D5)
// ============================================================================

/// Trait for pluggable authorization checks.
///
/// Decouples DiscoveryService from PolicyClient. The main `hyprstream` crate
/// provides a `PolicyAuthProvider` that wraps `PolicyClient`.
#[async_trait(?Send)]
pub trait AuthorizationProvider: Send + Sync {
    /// Check if a subject is authorized for the given operation on a resource.
    async fn check(
        &self,
        subject: &str,
        domain: &str,
        resource: &str,
        operation: &str,
    ) -> Result<bool>;
}

// ============================================================================
// RecordResolver trait (#431)
// ============================================================================

/// A record returned as a CARv1 proof (the in-memory form of `RecordCar`).
///
/// `uri` is the at:// URI this CAR answers; `car` is the CARv1 bytes
/// (roots = [commit CID]; blocks = commit + MST path + record). The caller
/// validates it offline via `hyprstream_pds::car::verify_record_proof` — the
/// DiscoveryService is an untrusted relay, so integrity comes from the signed
/// proof, never from trusting the responder.
#[derive(Clone, Debug)]
pub struct RecordCarData {
    pub uri: String,
    pub car: Vec<u8>,
}

/// Trait for resolving atproto records/repos from the node's local PDS.
///
/// Decouples DiscoveryService from `hyprstream-pds` (a pure crypto/metadata
/// crate with no networking) and from the per-account record stores held by the
/// main `hyprstream` crate. The main crate provides a `PdsRecordResolver` that
/// builds CAR proofs via `hyprstream_pds::car::build_record_proof_car`.
///
/// IMPORTANT: a resolver MUST NOT perform access control — confidentiality is
/// enforced by the DiscoveryService handler's Casbin check on the target
/// DID/collection *before* the resolver is consulted. The resolver only answers
/// "does this record exist, and what is its signed CAR proof".
#[async_trait(?Send)]
pub trait RecordResolver: Send + Sync {
    /// Resolve a single record to a CAR proof. `collection` is the NSID
    /// (e.g. `ai.hyprstream.model`), `rkey` the TID record key. Returns
    /// `Ok(None)` when the DID/collection/rkey does not name a stored record.
    async fn resolve_record(
        &self,
        did: &str,
        collection: &str,
        rkey: &str,
    ) -> Result<Option<RecordCarData>>;

    /// Resolve a full repo CAR for a DID (commit + MST + all records).
    /// Returns `Ok(None)` when no repo is stored for the DID.
    async fn resolve_repo(&self, did: &str) -> Result<Option<RecordCarData>>;

    /// Resolve the DID's `#atproto` P-256 verifying key — the key a repo CAR's
    /// commit is signed by — or `Ok(None)` when this resolver cannot establish
    /// one.
    ///
    /// This is the **signature-verification seam** for verified-by-construction
    /// ingest: [`PlacementIndex::ingest_did`] requires a key and verifies the
    /// ingested repo CAR's commit signature against it *before* any record
    /// enters the index. A failed resolution, including `Ok(None)`, is rejected
    /// at that trust boundary; it never authorizes an unverified fallback.
    /// The default keeps non-participating resolvers source-compatible while
    /// preserving fail-closed ingest.
    async fn resolve_verifying_key(&self, _did: &str) -> Result<Option<p256::ecdsa::VerifyingKey>> {
        Ok(None)
    }

    /// Resolve every bounded `#atproto` verification slot currently published
    /// for `did`. Implementations that only know one trusted active key retain
    /// the fail-closed default through [`RecordResolver::resolve_verifying_key`].
    async fn resolve_verifying_keys(
        &self,
        did: &str,
    ) -> Result<Option<hyprstream_pds::commit::PublishedAtprotoKeys>> {
        Ok(self
            .resolve_verifying_key(did)
            .await?
            .map(hyprstream_pds::commit::PublishedAtprotoKeys::single))
    }
}

// ============================================================================
// DiscoveryService
// ============================================================================

/// Endpoint data stored per announced entry.
#[derive(Clone)]
struct AnnouncedEndpoint {
    /// Socket kind (e.g. "quic", "rep")
    socket_kind: String,
    /// Endpoint string (e.g. "quic://localhost:0.0.0.0:4433")
    endpoint: String,
    /// Service JWT attesting to the service's identity and pubkey
    service_jwt: String,
    service_did: Did,
    capabilities: BTreeSet<String>,
    accepted_state_digest: Vec<u8>,
    accepted_state_epoch: u64,
    response_key_id: String,
    request_kem_key_id: String,
    request_kem_recipient: Vec<u8>,
    expires_at_unix_ms: i64,
    source_signer: [u8; 32],
    /// Last heartbeat timestamp (Instant)
    last_heartbeat: Instant,
}

/// Checkpoint-verifying accepted-current-state read used by production
/// resolution. Implemented by the daemon-owned PDS reader from #1004.
pub(super) trait AcceptedStateSource: Send + Sync {
    fn accepted_state(
        &self,
        did: &str,
    ) -> Result<Option<hyprstream_pds::at9p_duplicity::AcceptedAt9pState>>;
}

/// Opaque production authority minted only while holding the checkpoint/PDS
/// accepted-state object. Raw announcement candidates never have this type.
#[derive(Clone)]
struct ResolvedService {
    selected: SelectedService,
}

impl ResolvedService {
    fn mint(
        selected: SelectedService,
        state: &hyprstream_pds::at9p_duplicity::AcceptedAt9pState,
    ) -> Result<Self> {
        anyhow::ensure!(
            selected.service_did().as_str() == state.did,
            "selected service DID is not the checkpoint-verified subject"
        );
        anyhow::ensure!(
            selected.evidence().accepted_state_epoch == state.epoch,
            "selected service epoch is not checkpoint current"
        );
        anyhow::ensure!(
            selected.evidence().accepted_state_digest == state.head_digest,
            "selected service digest is not checkpoint current"
        );
        let current_key = state
            .current
            .subject_keys
            .iter()
            .find(|key| {
                key.ed25519_pub.as_slice()
                    == selected.response_verifying_key().to_bytes().as_slice()
            })
            .ok_or_else(|| {
                anyhow::anyhow!("selected response key is not one of the accepted current keys")
            })?;
        anyhow::ensure!(
            selected.response_ml_dsa65() == current_key.mldsa65_pub.as_slice(),
            "selected PQ response key is not the atomically-bound accepted current key"
        );
        anyhow::ensure!(
            state
                .current
                .services
                .iter()
                .any(|service| service.id == format!("#{}", selected.service_name())),
            "selected service is not present in accepted current state"
        );
        Ok(Self { selected })
    }
    fn same_authority(&self, other: &Self) -> bool {
        self.selected.same_authority(&other.selected)
    }
}

impl std::ops::Deref for ResolvedService {
    type Target = SelectedService;
    fn deref(&self) -> &Self::Target {
        &self.selected
    }
}

struct CurrentStreamHandle {
    inner: Box<dyn StreamHandle>,
    resolver: Arc<DiscoveryServiceResolver>,
    snapshot: ResolvedService,
    invalidated: bool,
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl StreamHandle for CurrentStreamHandle {
    async fn next_payload(&mut self) -> Result<Option<hyprstream_rpc::streaming::StreamPayload>> {
        anyhow::ensure!(!self.invalidated, "stream snapshot is invalidated");
        if let Err(error) = self.resolver.ensure_current(&self.snapshot).await {
            self.invalidated = true;
            return Err(error);
        }
        self.inner.next_payload().await
    }
    async fn cancel(&mut self) -> Result<()> {
        anyhow::ensure!(!self.invalidated, "stream snapshot is invalidated");
        if let Err(error) = self.resolver.ensure_current(&self.snapshot).await {
            self.invalidated = true;
            return Err(error);
        }
        self.inner.cancel().await
    }
    fn stream_id(&self) -> &str {
        self.inner.stream_id()
    }
    fn is_completed(&self) -> bool {
        self.invalidated || self.inner.is_completed()
    }
}

/// Cloneable production resolver installed after Discovery bootstrap.
struct DiscoveryServiceResolver {
    announced_endpoints: Arc<RwLock<HashMap<String, Vec<AnnouncedEndpoint>>>>,
    accepted_state_source: Arc<dyn AcceptedStateSource>,
    discovery_client: Option<crate::DiscoveryClient>,
}

/// Phase 0.5 Stage D — cached signed OIDF entity statement.
struct CachedEntityStatement {
    /// Signed OpenID Federation 1.0 entity statement (compact JWS).
    jwt: String,
    /// Unix seconds when this was registered (set on push from issuer).
    fetched_at: i64,
}

/// Phase 0.5 Stage D — cached envelope COSE_KeySet.
struct CachedEnvelopeKeyset {
    /// CBOR-encoded COSE_KeySet (RFC 9052 §7).
    cose_keyset_cbor: Vec<u8>,
    /// Unix seconds when this was registered.
    fetched_at: i64,
}

/// Parse an `at://<did>/<collection>/<rkey>` URI into its three components.
///
/// The DID itself may contain `:` (e.g. `did:web:alice.example.com`,
/// `did:plc:abc123`) but no `/`, so splitting the post-`at://` remainder on `/`
/// yields exactly `[did, collection, rkey]`. Returns an error for any other shape.
fn parse_at_uri(uri: &str) -> Result<(String, String, String)> {
    let rest = uri
        .strip_prefix("at://")
        .ok_or_else(|| anyhow::anyhow!("at-uri must start with \"at://\": {uri:?}"))?;
    let mut parts = rest.splitn(3, '/');
    let did = parts.next().filter(|s| !s.is_empty());
    let collection = parts.next().filter(|s| !s.is_empty());
    let rkey = parts.next().filter(|s| !s.is_empty());
    match (did, collection, rkey) {
        (Some(d), Some(c), Some(r)) => Ok((d.to_owned(), c.to_owned(), r.to_owned())),
        _ => anyhow::bail!("at-uri must be at://<did>/<collection>/<rkey>: {uri:?}"),
    }
}

fn unix_seconds_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Discovery service that exposes EndpointRegistry over RPC.
pub struct DiscoveryService {
    /// Timestamp when the service was created
    started_at: Instant,
    /// Ed25519 signing key for envelope signing
    signing_key: Arc<SigningKey>,
    /// JWT verifying key for service JWT verification (derived from root via HKDF)
    jwt_verifying_key: hyprstream_rpc::prelude::VerifyingKey,
    /// JWT key source for client JWT verification (RequestService trait)
    jwt_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>>,
    /// OAuth issuer URL for RFC 9728 metadata (None = not configured)
    oauth_issuer_url: Option<String>,
    /// Expected audience for JWT validation (resource URL)
    expected_audience: Option<String>,
    /// Authorization provider (None = no authorization)
    auth_provider: Option<Box<dyn AuthorizationProvider>>,
    /// Record resolver backing getRecord/getRepo (#431). None = no local PDS,
    /// so getRecord/getRepo report NOT_FOUND for everything.
    record_resolver: Option<Arc<dyn RecordResolver>>,
    accepted_state_source: Option<Arc<dyn AcceptedStateSource>>,
    /// Endpoints announced by other services (cross-process).
    /// Maps service_name → Vec<AnnouncedEndpoint>.
    announced_endpoints: Arc<RwLock<HashMap<String, Vec<AnnouncedEndpoint>>>>,
    /// Phase 0.5 Stage D — cached signed OIDF entity statements per issuer URL.
    /// Pushed by IdPService/OAuth at startup + on every signing-key rotation.
    /// Consumed by FederationKeyResolver before falling back to HTTPS.
    entity_statements: RwLock<HashMap<String, CachedEntityStatement>>,
    /// Phase 0.5 Stage D — cached envelope COSE_KeySets per service did:web.
    /// Pushed by each service at startup + rotation. Consumed by RequestService
    /// receivers verifying COSE_Sign1 envelope signatures.
    envelope_keysets: RwLock<HashMap<String, CachedEnvelopeKeyset>>,
    /// Pre-computed TLS endorsement: Sign(tls_key, ed25519_pubkey || domain).
    /// Empty when TLS endorsement is not available (e.g. self-signed certs).
    tls_endorsement: Vec<u8>,
    /// Domain the TLS endorsement covers (empty when no endorsement).
    tls_domain: String,
    /// #524 P1 — durable placement directory (labels/declared-resources/group
    /// consents), refreshed by polling `record_resolver` lazily for an admitted
    /// heartbeat DID with no ingested `NodeRecord` yet.
    placement_index: PlacementIndex,
    /// Bounded retry gate for first-seen placement repository polls. A DID is
    /// marked before resolver access, so absent/invalid/non-node repos cannot
    /// turn heartbeat frequency into unbounded work.
    placement_ingest_attempts: TtlCache<Did, ()>,
    /// #524 P1 — live allocatable capacity + load per node, TTL'd
    /// (`LIVENESS_TTL`). Backs the hard-exclusion-on-staleness rule in
    /// `queryCandidates`.
    liveness: TtlCache<Did, LiveAllocatable>,
    // Infrastructure (for Spawnable)
    transport: TransportConfig,
}

impl DiscoveryService {
    /// Create a new discovery service with infrastructure.
    ///
    /// `signing_key` is used for envelope signing (should be the per-service key
    /// derived via HKDF). `jwt_verifying_key` is the root-derived JWT verifying
    /// key for verifying service JWTs (derived from root via HKDF "hyprstream-jwt-v1").
    pub fn new(
        signing_key: Arc<SigningKey>,
        jwt_verifying_key: hyprstream_rpc::prelude::VerifyingKey,
        transport: TransportConfig,
    ) -> Self {
        Self {
            started_at: Instant::now(),
            signing_key,
            jwt_verifying_key,
            jwt_key_source: None,
            oauth_issuer_url: None,
            expected_audience: None,
            auth_provider: None,
            record_resolver: None,
            accepted_state_source: None,
            announced_endpoints: Arc::new(RwLock::new(HashMap::new())),
            entity_statements: RwLock::new(HashMap::new()),
            envelope_keysets: RwLock::new(HashMap::new()),
            tls_endorsement: Vec::new(),
            tls_domain: String::new(),
            placement_index: PlacementIndex::new(),
            placement_ingest_attempts: TtlCache::new(
                LIVENESS_CACHE_MAX_ENTRIES,
                LIVENESS_CACHE_REAP_BUDGET,
            ),
            liveness: TtlCache::new(LIVENESS_CACHE_MAX_ENTRIES, LIVENESS_CACHE_REAP_BUDGET),
            transport,
        }
    }

    /// Set the pre-computed TLS endorsement and domain.
    ///
    /// The endorsement is `Sign(tls_private_key, "TLS_ENDORSEMENT_V1" || ed25519_pubkey || domain)`,
    /// computed once at startup by the service factory where TLS key materials are available.
    /// Discovery never touches the TLS private key directly.
    pub fn with_tls_endorsement(mut self, endorsement: Vec<u8>, domain: String) -> Self {
        self.tls_endorsement = endorsement;
        self.tls_domain = domain;
        self
    }

    /// Set the OAuth issuer URL for RFC 9728 auth metadata responses.
    pub fn with_oauth_issuer(mut self, url: String) -> Self {
        self.oauth_issuer_url = Some(url);
        self
    }

    /// Set the expected audience for JWT validation.
    pub fn with_expected_audience(mut self, audience: String) -> Self {
        self.expected_audience = Some(audience);
        self
    }

    /// Set the JWT key source for client JWT verification.
    pub fn with_jwt_key_source(
        mut self,
        src: std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    ) -> Self {
        self.jwt_key_source = Some(src);
        self
    }

    /// Set the authorization provider for access control.
    pub fn with_auth_provider(mut self, provider: Box<dyn AuthorizationProvider>) -> Self {
        self.auth_provider = Some(provider);
        self
    }

    /// Set the record resolver backing getRecord/getRepo (#431).
    pub fn with_record_resolver(mut self, resolver: Arc<dyn RecordResolver>) -> Self {
        self.record_resolver = Some(resolver);
        self
    }

    #[cfg(test)]
    fn with_accepted_state_source(mut self, source: Arc<dyn AcceptedStateSource>) -> Self {
        self.accepted_state_source = Some(source);
        self
    }

    /// Prove that downstream crates cannot access any production authority
    /// construction or installation seam.
    ///
    /// ```compile_fail
    /// use hyprstream_discovery::AcceptedStateSource;
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_discovery::DiscoveryService;
    /// fn inject(service: DiscoveryService) {
    ///     service.with_accepted_state_source(todo!());
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_discovery::DiscoveryService;
    /// fn install_caller_authority(service: &DiscoveryService) {
    ///     service.install_production_resolver().unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_discovery::DiscoveryService;
    /// fn choose_store_and_key(service: &mut DiscoveryService) {
    ///     service.install_checkpointed_pds_resolver(todo!(), todo!()).unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_discovery::CheckpointedPdsAuthority;
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_discovery::DiscoveryService;
    /// fn ambient_production_constructor() {
    ///     let _ = DiscoveryService::new_production(todo!(), todo!(), todo!());
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_service::DiscoveryBootstrapAuthority;
    /// fn forge(path: std::path::PathBuf, key: ed25519_dalek::VerifyingKey) {
    ///     let _ = DiscoveryBootstrapAuthority { store_path: path, acceptance_identity: key };
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn duplicate(authority: hyprstream_service::DiscoveryBootstrapAuthority) {
    ///     let _ = authority.clone();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_discovery::AuthenticatedDiscoveryBootstrap;
    /// fn forge(
    ///     store_path: std::path::PathBuf,
    ///     acceptance_identity: ed25519_dalek::VerifyingKey,
    /// ) -> AuthenticatedDiscoveryBootstrap {
    ///     AuthenticatedDiscoveryBootstrap { seal: (), store_path, acceptance_identity }
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_service::AuthenticatedRegistryDeploymentIdentity;
    /// fn forge(key: ed25519_dalek::VerifyingKey) -> AuthenticatedRegistryDeploymentIdentity {
    ///     AuthenticatedRegistryDeploymentIdentity { verifying_key: key }
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_service::AuthenticatedRegistryDeploymentIdentity;
    /// fn mint(key: ed25519_dalek::VerifyingKey) {
    ///     let _ = AuthenticatedRegistryDeploymentIdentity::mint(key);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn duplicate(identity: hyprstream_service::AuthenticatedRegistryDeploymentIdentity) {
    ///     let _ = identity.clone();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// # use ed25519_dalek::SigningKey;
    /// let signing = SigningKey::from_bytes(&[7; 32]);
    /// let mut context = hyprstream_service::ServiceContext::new(
    ///     signing.clone(), signing.verifying_key(), false, "caller".into()
    /// );
    /// context.authenticate_registry_deployment_credential("caller-issued-jwt")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// ```compile_fail
    /// # use ed25519_dalek::SigningKey;
    /// let signing = SigningKey::from_bytes(&[7; 32]);
    /// let mut context = hyprstream_service::ServiceContext::new(
    ///     signing.clone(), signing.verifying_key(), false, "caller".into()
    /// );
    /// context.authenticate_registry_deployment_credential()?;
    /// context.take_authenticated_registry_identity()?;
    /// let _raw = context.authenticated_registry_identity()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// ```compile_fail
    /// fn old_discovery_chain(
    ///     identity: hyprstream_service::AuthenticatedRegistryDeploymentIdentity,
    ///     client: hyprstream_discovery::DiscoveryClient,
    /// ) {
    ///     let authority = hyprstream_discovery::authenticate_discovery_bootstrap(identity).unwrap();
    ///     hyprstream_discovery::DiscoveryService::bootstrap_authenticated_process(authority, client).unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn extract(verifier: hyprstream_discovery::RegistryDeploymentVerifier) {
    ///     let _: ed25519_dalek::VerifyingKey = verifier.into_verifying_key();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use hyprstream_discovery::RegistryDeploymentVerifier;
    /// fn forge(key: ed25519_dalek::VerifyingKey) -> RegistryDeploymentVerifier {
    ///     RegistryDeploymentVerifier { verifying_key: key }
    /// }
    /// ```
    ///
    /// The complete former public composition chain is unavailable to an
    /// external crate.
    /// ```compile_fail
    /// use hyprstream_discovery::DiscoveryService;
    /// use hyprstream_service::ServiceContext;
    /// fn inject(context: ServiceContext, client: hyprstream_discovery::DiscoveryClient) {
    ///     context.seal_discovery_bootstrap_authority().unwrap();
    ///     let authority = context
    ///         .consume_discovery_bootstrap_authority(|path, key| Ok((path.to_owned(), key)))
    ///         .unwrap();
    ///     DiscoveryService::install_process_bootstrap(&context, client).unwrap();
    ///     drop(authority);
    /// }
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    fn bootstrap_authenticated_process(
        authority: ProcessBootstrapAuthority,
        discovery_client: crate::DiscoveryClient,
    ) -> Result<()> {
        let authority = {
            let mut state = PROCESS_BOOTSTRAP_AUTHORITY.lock();
            match std::mem::replace(&mut *state, ProcessBootstrapAuthorityState::Consumed) {
                ProcessBootstrapAuthorityState::Sealed => {}
                previous => {
                    *state = previous;
                    anyhow::bail!("authenticated Discovery bootstrap is unavailable or consumed");
                }
            }
            authority
        };
        let deployment_identity = match &authority.acceptance_identity {
            ProcessAcceptanceIdentity::Deployment(identity) => Some(identity.clone()),
            #[cfg(test)]
            ProcessAcceptanceIdentity::Test(_) => None,
        };
        let source: Arc<dyn AcceptedStateSource> = match authority.acceptance_identity {
            ProcessAcceptanceIdentity::Deployment(identity) => Arc::new(
                crate::checkpointed_pds::CheckpointedPdsAcceptedStateSource::open(
                    &authority.store_path,
                    identity,
                )?,
            ),
            #[cfg(test)]
            ProcessAcceptanceIdentity::Test(identity) => Arc::new(
                crate::checkpointed_pds::CheckpointedPdsAcceptedStateSource::open_test(
                    &authority.store_path,
                    identity,
                )?,
            ),
        };
        PROCESS_ACCEPTED_STATE_SOURCE
            .set(Arc::clone(&source))
            .map_err(|_| anyhow::anyhow!("process Discovery authority is already consumed"))?;
        if let Some(identity) = deployment_identity {
            PROCESS_REGISTRY_VERIFIER.set(identity).map_err(|_| {
                anyhow::anyhow!("deployment registry verifier is already installed")
            })?;
        }
        PRODUCTION_RESOLVER
            .set(Arc::new(DiscoveryServiceResolver {
                announced_endpoints: Arc::new(RwLock::new(HashMap::new())),
                accepted_state_source: source,
                discovery_client: Some(discovery_client),
            }))
            .map_err(|_| anyhow::anyhow!("production service resolver is already installed"))
    }

    /// Attach the process-pinned source to the Discovery daemon. The source
    /// was fixed and consumed before inventory factories were invoked.
    pub fn attach_process_accepted_state_source(&mut self) -> Result<()> {
        self.accepted_state_source = Some(
            PROCESS_ACCEPTED_STATE_SOURCE
                .get()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("process Discovery authority is not installed"))?,
        );
        Ok(())
    }

    #[cfg(test)]
    fn production_resolver(&self) -> Result<DiscoveryServiceResolver> {
        Ok(DiscoveryServiceResolver {
            announced_endpoints: Arc::clone(&self.announced_endpoints),
            accepted_state_source: self.accepted_state_source.clone().ok_or_else(|| {
                anyhow::anyhow!("Discovery accepted-state source is not installed")
            })?,
            discovery_client: None,
        })
    }

    /// Get the global EndpointRegistry (D9: graceful error, not panic).
    fn reg(&self) -> Result<impl std::ops::Deref<Target = EndpointRegistry> + '_> {
        registry::try_global().ok_or_else(|| anyhow::anyhow!("EndpointRegistry not initialized"))
    }
}

// ============================================================================
// Resolver implementation — DiscoveryService IS the authoritative resolver
// ============================================================================

#[async_trait]
impl Resolver for DiscoveryService {
    async fn resolve(&self, name: &str, kind: SocketKind) -> anyhow::Result<TransportConfig> {
        if let Some(transport) = self.resolve_announced_endpoint(name, kind)? {
            return Ok(transport);
        }

        // Cross-pod QUIC peers must come from fresh service announcements. The
        // registry fallback for QUIC is a non-dialable bootstrap placeholder.
        if kind == SocketKind::Quic {
            anyhow::bail!("no fresh announced QUIC endpoint for service '{name}'");
        }

        self.reg()?.try_endpoint(name, kind)
    }
}

fn accepted_expiry_unix_ms(
    state: &hyprstream_pds::at9p_duplicity::AcceptedAt9pState,
) -> Result<i64> {
    let expiry = state.expires_at.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "genesis-only accepted state has no bounded expiry; refusing production resolution"
        )
    })?;
    Ok(chrono::DateTime::parse_from_rfc3339(expiry)
        .map_err(|e| anyhow::anyhow!("invalid accepted-state expiry: {e}"))?
        .timestamp_millis())
}

impl DiscoveryServiceResolver {
    #[cfg(test)]
    async fn resolve_service(&self, query: ServiceQuery) -> Result<ResolvedService> {
        let selected = select_service_candidate(
            &query,
            self.acquire_candidates(&query).await?,
            unix_millis_now(),
        )?;
        let state = self
            .accepted_state_source
            .accepted_state(selected.service_did().as_str())?
            .ok_or_else(|| anyhow::anyhow!("accepted state disappeared during resolution"))?;
        ResolvedService::mint(selected, &state)
    }

    async fn acquire_candidates(&self, query: &ServiceQuery) -> Result<Vec<ServiceCandidate>> {
        let entries = if let Some(client) = &self.discovery_client {
            client
                .get_endpoints(&query.service_name)
                .await?
                .endpoints
                .into_iter()
                .filter_map(|endpoint| {
                    let source_signer: [u8; 32] = endpoint.source_signer.try_into().ok()?;
                    Some(AnnouncedEndpoint {
                        socket_kind: endpoint.socket_kind,
                        endpoint: endpoint.endpoint,
                        service_jwt: endpoint.service_jwt,
                        service_did: endpoint.service_did,
                        capabilities: endpoint.capabilities.into_iter().collect(),
                        accepted_state_digest: endpoint.accepted_state_digest,
                        accepted_state_epoch: endpoint.accepted_state_epoch,
                        response_key_id: endpoint.response_key_id,
                        request_kem_key_id: endpoint.request_kem_key_id,
                        request_kem_recipient: endpoint.request_kem_recipient,
                        expires_at_unix_ms: endpoint.expires_at_unix_ms,
                        source_signer,
                        last_heartbeat: Instant::now(),
                    })
                })
                .collect()
        } else {
            self.announced_endpoints
                .read()
                .get(&query.service_name)
                .cloned()
                .unwrap_or_default()
        };

        let mut candidates = Vec::new();
        for entry in entries {
            if entry.last_heartbeat.elapsed() > ANNOUNCED_ENDPOINT_TTL
                || entry.service_did.as_str().is_empty()
                || entry.accepted_state_digest.len() != 64
            {
                continue;
            }
            let Some(state) = self
                .accepted_state_source
                .accepted_state(entry.service_did.as_str())?
            else {
                continue;
            };
            if state.did != entry.service_did.as_str()
                || state.epoch != entry.accepted_state_epoch
                || state.head_digest.as_slice() != entry.accepted_state_digest.as_slice()
            {
                continue;
            }
            let Ok(state_expiry) = accepted_expiry_unix_ms(&state) else {
                continue;
            };
            let Some(current_key) = state
                .current
                .subject_keys
                .iter()
                .find(|key| key.ed25519_pub.as_slice() == entry.source_signer)
            else {
                continue;
            };
            let Ok(response_ed25519) = current_key.ed25519_pub.as_slice().try_into() else {
                continue;
            };
            let Ok(recipient) = hyprstream_rpc::crypto::hybrid_kem::RecipientPublic::decode(
                &entry.request_kem_recipient,
            ) else {
                continue;
            };
            let Ok(transport) = announced_endpoint_to_transport(&entry) else {
                continue;
            };
            let mut digest = [0u8; 64];
            digest.copy_from_slice(&entry.accepted_state_digest);
            candidates.push(ServiceCandidate {
                service_name: query.service_name.clone(),
                service_did: entry.service_did,
                response_verifying_key: response_ed25519,
                response_ml_dsa65: current_key.mldsa65_pub.clone(),
                response_key_id: entry.response_key_id,
                request_kem_recipient: Some(AnchoredKemRecipient {
                    key_id: entry.request_kem_key_id,
                    recipient,
                    not_after_unix_ms: entry.expires_at_unix_ms,
                }),
                transport,
                capabilities: entry.capabilities,
                accepted_state: AcceptedStateEvidence {
                    service_did: Did::from(state.did),
                    digest,
                    epoch: state.epoch,
                    expires_at_unix_ms: state_expiry,
                    response_ed25519,
                    response_ml_dsa65: current_key.mldsa65_pub.clone(),
                },
                source_signer: entry.source_signer,
                expires_at_unix_ms: entry.expires_at_unix_ms,
            });
        }
        Ok(candidates)
    }
}

impl DiscoveryServiceResolver {
    async fn resolve_service_candidates(
        &self,
        query: ServiceQuery,
    ) -> Result<Vec<ResolvedService>> {
        let candidates = self.acquire_candidates(&query).await?;
        let selected = select_service_candidates(&query, candidates, unix_millis_now())?;
        let mut resolved = Vec::with_capacity(selected.len());
        for item in selected {
            let state = self
                .accepted_state_source
                .accepted_state(item.service_did().as_str())?
                .ok_or_else(|| anyhow::anyhow!("accepted state disappeared during resolution"))?;
            resolved.push(ResolvedService::mint(item, &state)?);
        }
        Ok(resolved)
    }

    async fn ensure_current(&self, resolved: &ResolvedService) -> Result<()> {
        resolved.ensure_fresh(unix_millis_now())?;
        let state = self
            .accepted_state_source
            .accepted_state(resolved.service_did().as_str())?
            .ok_or_else(|| anyhow::anyhow!("accepted state disappeared; re-resolution required"))?;
        anyhow::ensure!(
            state.epoch == resolved.evidence().accepted_state_epoch
                && state.head_digest == resolved.evidence().accepted_state_digest,
            "accepted state advanced or forked; re-resolution required"
        );
        let state_expiry = accepted_expiry_unix_ms(&state)?;
        anyhow::ensure!(
            unix_millis_now() < state_expiry,
            "accepted state expired; re-resolution required"
        );
        Ok(())
    }

    async fn browser_provisioning(
        &self,
        request: BrowserProvisioningRequest,
    ) -> Result<BrowserProvisioningDocument> {
        request.validate()?;
        match request.carrier_profile {
            BrowserCarrierProfile::OwnedHybridWebTransport => {
                anyhow::ensure!(
                    request.capability == "hyprstream-rpc/1"
                        && request.scope == request.service_name,
                    "owned browser RPC provisioning requires hyprstream-rpc/1 and exact service scope"
                );
            }
            BrowserCarrierProfile::StandardPublicRelay => {
                anyhow::ensure!(
                    request.capability == "hyprstream-moq/1",
                    "standard-public-relay provisioning requires hyprstream-moq/1"
                );
            }
        }

        let query = ServiceQuery::new(
            request.service_name.clone(),
            [request.capability.clone()],
            ResolverProfile::NetworkDiscovery,
            1,
        )?;
        let resolved = self
            .resolve_service_candidates(query)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("resolver returned no validated browser reach"))?;
        self.ensure_current(&resolved).await?;
        let state = self
            .accepted_state_source
            .accepted_state(resolved.service_did().as_str())?
            .ok_or_else(|| {
                anyhow::anyhow!("accepted state disappeared during browser provisioning")
            })?;
        anyhow::ensure!(
            state.epoch == resolved.evidence().accepted_state_epoch
                && state.head_digest == resolved.evidence().accepted_state_digest,
            "accepted state advanced during browser provisioning; re-resolution required"
        );
        let accepted_service = state
            .current
            .services
            .iter()
            .find(|service| service.id == format!("#{}", request.service_name))
            .ok_or_else(|| {
                anyhow::anyhow!("accepted current state does not contain requested service")
            })?;

        let (origin, route, certificate_hashes, route_role, transport_security, encrypted_objects) =
            match request.carrier_profile {
                BrowserCarrierProfile::OwnedHybridWebTransport => {
                    let EndpointType::Quic {
                        addr,
                        server_name,
                        auth,
                    } = resolved.transport().endpoint.clone()
                    else {
                        anyhow::bail!(
                            "browser provisioning requires selected QUIC/WebTransport network reach"
                        );
                    };
                    let authority = format!("{server_name}:{}", addr.port());
                    let origin = format!("https://{authority}/");
                    let route = format!(
                        "https://{authority}{}",
                        hyprstream_rpc::browser_provisioning::BROWSER_RPC_PATH
                    );
                    (
                        origin,
                        route,
                        auth.accept_cert_hashes().to_vec(),
                        BrowserRouteRole::Origin,
                        BrowserTransportSecurity::OwnedHybridRequired,
                        false,
                    )
                }
                BrowserCarrierProfile::StandardPublicRelay => {
                    anyhow::ensure!(
                        accepted_service.endpoint.transport == hyprstream_pds::at9p::Transport::Moq,
                        "accepted service does not select the MoQT carrier"
                    );
                    let relay = accepted_service.endpoint.relay.clone().ok_or_else(|| {
                        anyhow::anyhow!(
                            "accepted service has no signed standard public relay route"
                        )
                    })?;
                    (
                        accepted_service.endpoint.address.clone(),
                        relay,
                        Vec::new(),
                        BrowserRouteRole::Relay,
                        BrowserTransportSecurity::ClassicalUntrusted,
                        true,
                    )
                }
            };

        let now = unix_millis_now();
        let expires_at_unix_ms = resolved
            .expires_at_unix_ms()
            .min(now.saturating_add(30_000));
        anyhow::ensure!(
            expires_at_unix_ms > now,
            "resolved browser provisioning expired before projection"
        );
        Ok(BrowserProvisioningDocument::from_material(
            BrowserProvisioningMaterial {
                service_name: request.service_name,
                service_did: resolved.service_did().as_str().to_owned(),
                service_origin: origin,
                webtransport_url: route,
                capability: request.capability,
                scope: request.scope,
                carrier_profile: request.carrier_profile,
                route_role,
                transport_security,
                response_key_id: resolved.response_key_id().to_owned(),
                response_ed25519: resolved.response_verifying_key().to_bytes(),
                response_ml_dsa65: resolved.response_ml_dsa65().to_vec(),
                request_kem_key_id: resolved.request_kem_recipient().key_id.clone(),
                request_kem_recipient: resolved.request_kem_recipient().recipient.clone(),
                accepted_state_digest: resolved.evidence().accepted_state_digest,
                accepted_state_epoch: resolved.evidence().accepted_state_epoch,
                expires_at_unix_ms,
                certificate_hashes,
                encrypted_objects_required: encrypted_objects,
            },
        ))
    }

    async fn verify_browser_binding(&self, binding: &BrowserRequestBinding) -> Result<()> {
        binding.validate_shape()?;
        let now = unix_millis_now();
        anyhow::ensure!(
            binding.expires_at_unix_ms > now,
            "sealed browser provisioning binding expired before dispatch"
        );
        let current = self
            .browser_provisioning(BrowserProvisioningRequest::new(
                binding.service_name.clone(),
                binding.capability.clone(),
                binding.scope.clone(),
                binding.carrier_profile,
            )?)
            .await?;
        anyhow::ensure!(
            binding.service_did == current.service_did
                && binding.service_origin == current.service_origin
                && binding.webtransport_url == current.webtransport_url
                && binding.certificate_hashes == current.certificate_hashes
                && binding.response_key_id == current.response_key_id
                && binding.request_kem_key_id == current.request_kem_key_id
                && binding.accepted_state_epoch == current.accepted_state_epoch
                && binding.accepted_state_digest == current.accepted_state_digest
                && binding.expires_at_unix_ms <= current.expires_at_unix_ms,
            "accepted-current browser authority changed before dispatch"
        );
        let response_key = URL_SAFE_NO_PAD
            .decode(&current.response_ed25519)
            .context("invalid current response key projection")?;
        let kem = URL_SAFE_NO_PAD
            .decode(&current.request_kem_recipient)
            .context("invalid current request KEM projection")?;
        anyhow::ensure!(
            binding.matches_response_key(&response_key)? && binding.matches_request_kem(&kem)?,
            "accepted-current browser key material changed before dispatch"
        );
        Ok(())
    }
}

struct ProductionBrowserCurrentnessVerifier {
    resolver: Arc<DiscoveryServiceResolver>,
}

#[async_trait]
impl BrowserCurrentnessVerifier for ProductionBrowserCurrentnessVerifier {
    async fn ensure_current(&self, binding: &BrowserRequestBinding) -> Result<()> {
        self.resolver.verify_browser_binding(binding).await
    }
}

static PROCESS_ACCEPTED_STATE_SOURCE: std::sync::OnceLock<Arc<dyn AcceptedStateSource>> =
    std::sync::OnceLock::new();
static PRODUCTION_RESOLVER: std::sync::OnceLock<Arc<DiscoveryServiceResolver>> =
    std::sync::OnceLock::new();
static PROCESS_REGISTRY_VERIFIER: std::sync::OnceLock<RegistryDeploymentVerifier> =
    std::sync::OnceLock::new();

const DEPLOYMENT_CA_ROOT_PATH: &str = "/etc/hyprstream/trust/deployment-ca.ed25519";
const REGISTRY_DEPLOYMENT_CREDENTIAL_PATH: &str =
    "/run/hyprstream/credentials/registry-service.jwt";
const REGISTRY_DEPLOYMENT_CREDENTIAL_PROFILE: &str = "hyprstream.registry-deployment.v1";
const REGISTRY_DEPLOYMENT_CREDENTIAL_AUDIENCE: &str = "urn:hyprstream:service:registry";
const REGISTRY_DEPLOYMENT_CREDENTIAL_MAX_TTL_SECONDS: i64 = 3_600;
const REGISTRY_DEPLOYMENT_CREDENTIAL_CLOCK_SKEW_SECONDS: i64 = 60;

/// Opaque verification-only view of the authenticated deployment registry.
/// It exposes neither the raw key nor an authority replacement operation.
#[derive(Clone)]
pub struct RegistryDeploymentVerifier {
    verifying_key: VerifyingKey,
}

impl RegistryDeploymentVerifier {
    pub fn verify_strict(
        &self,
        message: &[u8],
        signature: &ed25519_dalek::Signature,
    ) -> Result<()> {
        self.verifying_key
            .verify_strict(message, signature)
            .map_err(|error| anyhow::anyhow!("deployment registry signature rejected: {error}"))
    }

    pub fn matches(&self, key: &VerifyingKey) -> bool {
        self.verifying_key == *key
    }
}

/// Read the verification-only identity installed by trusted startup.
pub fn deployment_registry_verifier() -> Result<RegistryDeploymentVerifier> {
    PROCESS_REGISTRY_VERIFIER
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("deployment registry verifier is not installed"))
}

/// Non-cloneable proof privately minted from the fixed CA/JWT pair.
struct AuthenticatedRegistryDeploymentIdentity {
    verifier: RegistryDeploymentVerifier,
}

struct TrustedRegistryDeploymentCredentials {
    ca_verifying_key: VerifyingKey,
    registry_credential: String,
}

/// JSON value decoded while preserving the security property that every object
/// member has exactly one interpretation. `serde_json::Value` alone accepts a
/// duplicate member with last-value-wins semantics, which is unsuitable for a
/// credential profile.
struct UniqueJson(serde_json::Value);

impl<'de> serde::Deserialize<'de> for UniqueJson {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Visitor;

        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueJson;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("JSON without duplicate object members")
            }

            fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Null))
            }

            fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Number(value.into())))
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Number(value.into())))
            }

            fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                serde_json::Number::from_f64(value)
                    .map(serde_json::Value::Number)
                    .map(UniqueJson)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::String(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::String(value)))
            }

            fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                self.visit_unit()
            }

            fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                <UniqueJson as serde::Deserialize>::deserialize(deserializer)
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<UniqueJson>()? {
                    values.push(value.0);
                }
                Ok(UniqueJson(serde_json::Value::Array(values)))
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut values = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, UniqueJson>()? {
                    if values.insert(key.clone(), value.0).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate JSON member {key:?}"
                        )));
                    }
                }
                Ok(UniqueJson(serde_json::Value::Object(values)))
            }
        }

        deserializer.deserialize_any(Visitor)
    }
}

fn exact_json_object<'a>(
    value: &'a serde_json::Value,
    expected: &[&str],
    description: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{description} must be a JSON object"))?;
    for member in object.keys() {
        anyhow::ensure!(
            expected.contains(&member.as_str()),
            "{description} contains unexpected member {member:?}"
        );
    }
    for member in expected {
        anyhow::ensure!(
            object.contains_key(*member),
            "{description} is missing member {member:?}"
        );
    }
    Ok(object)
}

fn exact_string_member<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    member: &str,
    description: &str,
) -> Result<&'a str> {
    object[member]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("{description}.{member} must be a string"))
}

fn exact_i64_member(
    object: &serde_json::Map<String, serde_json::Value>,
    member: &str,
    description: &str,
) -> Result<i64> {
    object[member]
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("{description}.{member} must be an integer"))
}

fn decode_canonical_jwt_segment(segment: &str, description: &str, limit: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(!segment.is_empty(), "{description} is empty");
    anyhow::ensure!(segment.len() <= limit, "{description} is too large");
    let decoded = URL_SAFE_NO_PAD
        .decode(segment)
        .map_err(|error| anyhow::anyhow!("{description} is not unpadded base64url: {error}"))?;
    anyhow::ensure!(
        URL_SAFE_NO_PAD.encode(&decoded) == segment,
        "{description} is not canonical unpadded base64url"
    );
    Ok(decoded)
}

fn parse_unique_jwt_json(
    segment: &str,
    description: &str,
    limit: usize,
) -> Result<serde_json::Value> {
    let bytes = decode_canonical_jwt_segment(segment, description, limit)?;
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes);
    let value = <UniqueJson as serde::Deserialize>::deserialize(&mut deserializer)
        .map_err(|error| anyhow::anyhow!("{description} is malformed: {error}"))?;
    deserializer
        .end()
        .map_err(|error| anyhow::anyhow!("{description} has trailing data: {error}"))?;
    Ok(value.0)
}

fn deployment_domain(ca_verifying_key: &VerifyingKey) -> String {
    hyprstream_rpc::auth::jwk_thumbprint(&hyprstream_rpc::auth::JwkThumbprintInput::Ed25519 {
        x: ca_verifying_key.as_bytes(),
    })
}

fn validate_registry_credential_numeric_dates(
    exp: i64,
    nbf: i64,
    iat: i64,
    now: i64,
) -> Result<()> {
    for (name, value) in [("exp", exp), ("nbf", nbf), ("iat", iat)] {
        anyhow::ensure!(
            value >= 0,
            "credential {name} must not precede the Unix epoch"
        );
    }

    anyhow::ensure!(exp > now, "credential has expired");
    let latest_future_time = now
        .checked_add(REGISTRY_DEPLOYMENT_CREDENTIAL_CLOCK_SKEW_SECONDS)
        .ok_or_else(|| anyhow::anyhow!("credential clock-skew boundary overflow"))?;
    anyhow::ensure!(nbf <= latest_future_time, "credential is not yet valid");
    anyhow::ensure!(
        iat <= latest_future_time,
        "credential was issued in the future"
    );
    anyhow::ensure!(nbf <= exp, "credential nbf is after exp");
    anyhow::ensure!(nbf <= iat, "credential nbf is after iat");
    anyhow::ensure!(iat < exp, "credential iat is not before exp");
    let lifetime = exp
        .checked_sub(iat)
        .ok_or_else(|| anyhow::anyhow!("credential lifetime arithmetic overflow"))?;
    anyhow::ensure!(
        lifetime <= REGISTRY_DEPLOYMENT_CREDENTIAL_MAX_TTL_SECONDS,
        "credential lifetime exceeds the inclusive one-hour profile limit"
    );
    Ok(())
}

fn validate_registry_deployment_credential_profile(
    token: &str,
    ca_verifying_key: &VerifyingKey,
) -> Result<[u8; 32]> {
    let mut segments = token.split('.');
    let protected_segment = segments.next().unwrap_or_default();
    let claims_segment = segments.next().unwrap_or_default();
    let signature_segment = segments.next().unwrap_or_default();
    anyhow::ensure!(
        segments.next().is_none(),
        "credential must contain exactly three segments"
    );
    decode_canonical_jwt_segment(signature_segment, "credential signature", 128)?;

    let protected = parse_unique_jwt_json(protected_segment, "protected header", 4_096)?;
    let protected = exact_json_object(&protected, &["alg", "typ", "kid"], "protected header")?;
    anyhow::ensure!(
        exact_string_member(protected, "alg", "protected header")? == "EdDSA",
        "protected header alg must be exactly EdDSA"
    );
    anyhow::ensure!(
        exact_string_member(protected, "typ", "protected header")? == "wit+jwt",
        "protected header typ must be exactly wit+jwt"
    );
    let domain = deployment_domain(ca_verifying_key);
    anyhow::ensure!(
        exact_string_member(protected, "kid", "protected header")? == domain,
        "protected header kid does not bind the pinned deployment CA"
    );

    let claims = parse_unique_jwt_json(claims_segment, "credential claims", 16_384)?;
    let claims = exact_json_object(
        &claims,
        &[
            "iss",
            "sub",
            "aud",
            "exp",
            "nbf",
            "iat",
            "deployment_domain",
            "profile",
            "cnf",
        ],
        "credential claims",
    )?;
    anyhow::ensure!(
        exact_string_member(claims, "sub", "credential claims")? == "service:registry",
        "credential subject must be exactly service:registry"
    );
    let intended_issuer = format!("urn:hyprstream:deployment:{domain}");
    anyhow::ensure!(
        exact_string_member(claims, "iss", "credential claims")? == intended_issuer,
        "credential issuer does not bind the pinned deployment domain"
    );
    anyhow::ensure!(
        exact_string_member(claims, "aud", "credential claims")?
            == REGISTRY_DEPLOYMENT_CREDENTIAL_AUDIENCE,
        "credential audience must be exactly {REGISTRY_DEPLOYMENT_CREDENTIAL_AUDIENCE}"
    );
    anyhow::ensure!(
        exact_string_member(claims, "deployment_domain", "credential claims")? == domain,
        "credential deployment_domain does not bind the pinned deployment CA"
    );
    anyhow::ensure!(
        exact_string_member(claims, "profile", "credential claims")?
            == REGISTRY_DEPLOYMENT_CREDENTIAL_PROFILE,
        "credential profile must be exactly {REGISTRY_DEPLOYMENT_CREDENTIAL_PROFILE}"
    );

    let now = chrono::Utc::now().timestamp();
    let exp = exact_i64_member(claims, "exp", "credential claims")?;
    let nbf = exact_i64_member(claims, "nbf", "credential claims")?;
    let iat = exact_i64_member(claims, "iat", "credential claims")?;
    validate_registry_credential_numeric_dates(exp, nbf, iat, now)?;

    let cnf = exact_json_object(&claims["cnf"], &["jwk"], "credential claims.cnf")?;
    let jwk = exact_json_object(
        &cnf["jwk"],
        &["kty", "crv", "x"],
        "credential claims.cnf.jwk",
    )?;
    anyhow::ensure!(
        exact_string_member(jwk, "kty", "credential claims.cnf.jwk")? == "OKP",
        "credential cnf.jwk.kty must be exactly OKP"
    );
    anyhow::ensure!(
        exact_string_member(jwk, "crv", "credential claims.cnf.jwk")? == "Ed25519",
        "credential cnf.jwk.crv must be exactly Ed25519"
    );
    let x = exact_string_member(jwk, "x", "credential claims.cnf.jwk")?;
    let key = decode_canonical_jwt_segment(x, "credential cnf.jwk.x", 64)?;
    let key: [u8; 32] = key
        .try_into()
        .map_err(|_| anyhow::anyhow!("credential cnf.jwk.x must decode to exactly 32 bytes"))?;

    // Signature verification is deliberately last: no parsed material can mint
    // authority unless the exact closed profile is authenticated by the fixed CA.
    let verified = hyprstream_rpc::auth::jwt::decode(
        token,
        ca_verifying_key,
        Some(REGISTRY_DEPLOYMENT_CREDENTIAL_AUDIENCE),
    )
    .map_err(|error| anyhow::anyhow!("credential signature rejected: {error}"))?;
    anyhow::ensure!(
        verified.sub == "service:registry" && verified.cnf_key_bytes() == Some(key),
        "verified credential differs from the exact deployment profile"
    );
    Ok(key)
}

#[cfg(unix)]
fn read_os_owned_file(path: &std::path::Path, description: &str) -> Result<Vec<u8>> {
    use std::os::unix::fs::MetadataExt as _;

    anyhow::ensure!(path.is_absolute(), "{description} path must be absolute");
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| anyhow::anyhow!("{description} is unavailable at {path:?}: {error}"))?;
    anyhow::ensure!(
        metadata.file_type().is_file(),
        "{description} is not a regular file"
    );
    anyhow::ensure!(metadata.uid() == 0, "{description} is not owned by root");
    anyhow::ensure!(
        metadata.mode() & 0o022 == 0,
        "{description} is group/world writable"
    );
    for parent in path.ancestors().skip(1) {
        let parent_metadata = std::fs::symlink_metadata(parent).map_err(|error| {
            anyhow::anyhow!("{description} parent {parent:?} is unavailable: {error}")
        })?;
        anyhow::ensure!(
            parent_metadata.file_type().is_dir(),
            "{description} parent {parent:?} is not a directory"
        );
        anyhow::ensure!(
            parent_metadata.uid() == 0 && parent_metadata.mode() & 0o022 == 0,
            "{description} parent {parent:?} is writable or not root-owned"
        );
    }
    std::fs::read(path)
        .map_err(|error| anyhow::anyhow!("failed to read {description} at {path:?}: {error}"))
}

#[cfg(not(unix))]
fn read_os_owned_file(_path: &std::path::Path, description: &str) -> Result<Vec<u8>> {
    anyhow::bail!("{description} requires the OS-owned Unix deployment seam")
}

fn load_registry_deployment_credential() -> Result<String> {
    String::from_utf8(read_os_owned_file(
        std::path::Path::new(REGISTRY_DEPLOYMENT_CREDENTIAL_PATH),
        "registry deployment credential",
    )?)
    .map_err(|error| anyhow::anyhow!("registry deployment credential is not UTF-8: {error}"))
}

fn load_trusted_registry_deployment_credentials() -> Result<TrustedRegistryDeploymentCredentials> {
    let ca_bytes = read_os_owned_file(
        std::path::Path::new(DEPLOYMENT_CA_ROOT_PATH),
        "deployment CA root",
    )?;
    let ca_bytes: [u8; 32] = ca_bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("deployment CA root must be 32 bytes"))?;
    let ca_verifying_key = VerifyingKey::from_bytes(&ca_bytes)
        .map_err(|error| anyhow::anyhow!("deployment CA root is malformed: {error}"))?;
    let registry_credential = load_registry_deployment_credential()?;
    Ok(TrustedRegistryDeploymentCredentials {
        ca_verifying_key,
        registry_credential,
    })
}

fn authenticate_registry_deployment_credentials(
    credentials: TrustedRegistryDeploymentCredentials,
) -> Result<AuthenticatedRegistryDeploymentIdentity> {
    let key_bytes = validate_registry_deployment_credential_profile(
        &credentials.registry_credential,
        &credentials.ca_verifying_key,
    )
    .map_err(|error| anyhow::anyhow!("registry deployment credential rejected: {error}"))?;
    let verifying_key = VerifyingKey::from_bytes(&key_bytes)
        .map_err(|error| anyhow::anyhow!("registry credential key is malformed: {error}"))?;
    Ok(AuthenticatedRegistryDeploymentIdentity {
        verifier: RegistryDeploymentVerifier { verifying_key },
    })
}

struct ProcessBootstrapAuthority {
    store_path: std::path::PathBuf,
    acceptance_identity: ProcessAcceptanceIdentity,
}

enum ProcessAcceptanceIdentity {
    Deployment(RegistryDeploymentVerifier),
    #[cfg(test)]
    Test(VerifyingKey),
}

enum ProcessBootstrapAuthorityState {
    Unsealed,
    Sealed,
    Consumed,
}

static PROCESS_BOOTSTRAP_AUTHORITY: parking_lot::Mutex<ProcessBootstrapAuthorityState> =
    parking_lot::Mutex::new(ProcessBootstrapAuthorityState::Unsealed);

/// Atomically consume the explicitly selected deployment witness and install
/// the process Discovery resolver. Selection never falls back between the
/// OS-owned and DID-anchored providers.
#[cfg(not(target_arch = "wasm32"))]
pub async fn bootstrap_deployment_process(
    signing_key: SigningKey,
    trust_source: crate::DeploymentTrustSource,
) -> Result<()> {
    let discovery_vk = hyprstream_service::global_trust_store()
        .resolve_one("discovery")
        .ok_or_else(|| anyhow::anyhow!("trust store has no authenticated discovery key"))?;
    let (authority, discovery_client) = match trust_source {
        crate::DeploymentTrustSource::OsOwnedFiles => {
            let authority = authenticate_deployment_bootstrap()?;
            let client =
                crate::DiscoveryClient::for_local_bootstrap(signing_key, discovery_vk, None)?;
            (authority, client)
        }
        crate::DeploymentTrustSource::DidAnchored(anchors) => {
            let (authority, transport) = authenticate_did_anchored_bootstrap(&anchors).await?;
            let signer = hyprstream_rpc::signer::LocalSigner::new(signing_key);
            let rpc = hyprstream_rpc::dial::dial(&transport, signer, Some(discovery_vk), None)?;
            let client = crate::DiscoveryClient::new(rpc);
            let health = client
                .ping()
                .await
                .context("DID-anchored Discovery reach failed liveness check")?;
            anyhow::ensure!(
                health.status == "ok",
                "DID-anchored Discovery liveness returned status {:?}",
                health.status
            );
            (authority, client)
        }
    };
    DiscoveryService::bootstrap_authenticated_process(authority, discovery_client)
}

#[cfg(not(target_arch = "wasm32"))]
fn seal_process_bootstrap_authority() -> Result<()> {
    let mut state = PROCESS_BOOTSTRAP_AUTHORITY.lock();
    anyhow::ensure!(
        matches!(*state, ProcessBootstrapAuthorityState::Unsealed),
        "Discovery bootstrap authority is already sealed or consumed"
    );
    *state = ProcessBootstrapAuthorityState::Sealed;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn authenticate_deployment_bootstrap() -> Result<ProcessBootstrapAuthority> {
    seal_process_bootstrap_authority()?;
    let witness = authenticate_registry_deployment_credentials(
        load_trusted_registry_deployment_credentials()?,
    )?;
    let store_path = hyprstream_service::deployment_data_dir()?.join("pds-store");
    Ok(ProcessBootstrapAuthority {
        store_path,
        acceptance_identity: ProcessAcceptanceIdentity::Deployment(witness.verifier),
    })
}

#[cfg(not(target_arch = "wasm32"))]
async fn authenticate_did_anchored_bootstrap(
    anchors: &crate::DidAnchors,
) -> Result<(ProcessBootstrapAuthority, TransportConfig)> {
    seal_process_bootstrap_authority()?;
    let trust = crate::did_anchored::resolve_did_anchored_trust(anchors).await?;
    // #556 / F5: the deployment CA anchors the registry key, which certifies the
    // audit key and accepted-state checkpoints. It must never land classical.
    // Option C sources the CA from the capsule's hybrid subject key, so this is
    // PqHybrid by construction; enforce it as a fail-closed floor regardless.
    anyhow::ensure!(
        trust.assurance == hyprstream_rpc::auth::mac::Assurance::PqHybrid,
        "DID-anchored deployment CA did not land PqHybrid (got {:?}); refusing a classical trust root (#556)",
        trust.assurance
    );
    let witness =
        authenticate_registry_deployment_credentials(TrustedRegistryDeploymentCredentials {
            ca_verifying_key: trust.ca_verifying_key,
            registry_credential: load_registry_deployment_credential()?,
        })?;
    let authority = ProcessBootstrapAuthority {
        store_path: hyprstream_service::deployment_data_dir()?.join("pds-store"),
        acceptance_identity: ProcessAcceptanceIdentity::Deployment(witness.verifier),
    };
    Ok((authority, trust.discovery_transport))
}

#[cfg(test)]
fn authenticate_discovery_bootstrap_identity(
    acceptance_identity: ed25519_dalek::VerifyingKey,
) -> Result<ProcessBootstrapAuthority> {
    let mut state = PROCESS_BOOTSTRAP_AUTHORITY.lock();
    anyhow::ensure!(
        matches!(*state, ProcessBootstrapAuthorityState::Unsealed),
        "Discovery bootstrap authority is already sealed or consumed"
    );
    *state = ProcessBootstrapAuthorityState::Sealed;
    Ok(ProcessBootstrapAuthority {
        store_path: hyprstream_service::deployment_data_dir()?.join("pds-store"),
        acceptance_identity: ProcessAcceptanceIdentity::Test(acceptance_identity),
    })
}

/// Construct an ordinary production RPC client. Production authority cannot
/// be implemented, injected, or replaced by a downstream caller.
///
/// ```compile_fail
/// use hyprstream_rpc::{ResolvedRpcClient, ServiceResolver};
/// ```
///
/// ```compile_fail
/// use hyprstream_discovery::{DiscoveryServiceResolver, ResolvedService};
/// ```
///
/// Raw accepted evidence, candidates, selection, and the selected transport/key
/// bundle are private to Discovery's checkpoint-verifying implementation.
/// ```compile_fail
/// use hyprstream_rpc::{
///     select_service_candidate, AcceptedStateEvidence, AnchoredKemRecipient,
///     SelectedService, ServiceCandidate,
/// };
/// ```
pub fn production_rpc_client(
    service_name: &str,
    signing_key: SigningKey,
    token: Option<String>,
) -> Result<Arc<dyn RpcClient>> {
    let resolver = PRODUCTION_RESOLVER
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("checkpoint-backed production resolver is not installed"))?;
    Ok(Arc::new(ProductionRpcClient::new(
        service_name,
        signing_key,
        token,
        resolver,
    )?))
}

/// Project a browser/WebTransport provisioning document from the installed
/// checkpoint-backed production resolver. No caller can inject candidates,
/// accepted state, keys, reach, or a fallback resolver through this seam.
pub async fn production_browser_provisioning(
    request: BrowserProvisioningRequest,
) -> Result<BrowserProvisioningDocument> {
    let resolver = PRODUCTION_RESOLVER
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("checkpoint-backed production resolver is not installed"))?;
    resolver.browser_provisioning(request).await
}

/// Return the verifier backed by the same opaque production resolver used for
/// projection. It is installed at the server dispatch boundary; no caller can
/// inject accepted state or a fallback trust source.
pub fn production_browser_currentness_verifier() -> Result<Arc<dyn BrowserCurrentnessVerifier>> {
    let resolver = PRODUCTION_RESOLVER
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("checkpoint-backed production resolver is not installed"))?;
    Ok(Arc::new(ProductionBrowserCurrentnessVerifier { resolver }))
}

struct ProductionRpcClient {
    service_name: String,
    signing_key: SigningKey,
    token: Option<String>,
    resolver: Arc<DiscoveryServiceResolver>,
    request_id: std::sync::atomic::AtomicU64,
}

impl ProductionRpcClient {
    fn new(
        service_name: &str,
        signing_key: SigningKey,
        token: Option<String>,
        resolver: Arc<DiscoveryServiceResolver>,
    ) -> Result<Self> {
        anyhow::ensure!(
            !service_name.is_empty()
                && service_name
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
            "service name is not canonical"
        );
        Ok(Self {
            service_name: service_name.to_owned(),
            signing_key,
            token,
            resolver,
            request_id: std::sync::atomic::AtomicU64::new(1),
        })
    }
    async fn snapshots(&self) -> Result<Vec<ResolvedService>> {
        let query = ServiceQuery::network(self.service_name.clone())?;
        let max_attempts = query.max_attempts;
        let snapshots = self.resolver.resolve_service_candidates(query).await?;
        let authority = snapshots
            .first()
            .ok_or_else(|| anyhow::anyhow!("resolver returned no validated alternatives"))?;
        anyhow::ensure!(
            snapshots.iter().all(|item| item.same_authority(authority)),
            "resolver retry set crosses service authority"
        );
        Ok(snapshots.into_iter().take(max_attempts).collect())
    }
    fn client_for(&self, snapshot: &ResolvedService) -> Result<Arc<dyn RpcClient>> {
        let (kem, pq) = snapshot.crypto_stores()?;
        let signer = hyprstream_rpc::signer::LocalSigner::new(self.signing_key.clone());
        hyprstream_rpc::dial::dial_with_crypto_stores(
            snapshot.transport(),
            signer,
            Some(snapshot.response_verifying_key()),
            self.token.clone(),
            Some(kem),
            Some(pq),
        )
    }
    async fn attempt<T, F, Fut>(&self, mut call: F) -> Result<(T, ResolvedService)>
    where
        F: FnMut(Arc<dyn RpcClient>) -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        for refresh in 0..2 {
            let mut last_transport_error = None;
            let mut invalidated = None;
            for snapshot in self.snapshots().await? {
                if let Err(error) = self.resolver.ensure_current(&snapshot).await {
                    invalidated = Some(error);
                    break;
                }
                match call(self.client_for(&snapshot)?).await {
                    Ok(value) => return Ok((value, snapshot)),
                    Err(error)
                        if hyprstream_rpc::transport_traits::is_pre_dispatch_transport_error(
                            &error,
                        ) =>
                    {
                        last_transport_error = Some(error);
                    }
                    Err(error) => return Err(error),
                }
            }
            if let Some(error) = invalidated {
                if refresh == 0 {
                    continue;
                }
                return Err(anyhow::anyhow!(
                    "resolved service remained invalid after re-resolution: {error}"
                ));
            }
            return Err(last_transport_error.unwrap_or_else(|| {
                anyhow::anyhow!("validated same-authority alternatives exhausted")
            }));
        }
        unreachable!("bounded re-resolution loop always returns")
    }
    fn checked_stream(
        &self,
        inner: Box<dyn StreamHandle>,
        snapshot: ResolvedService,
    ) -> Box<dyn StreamHandle> {
        Box::new(CurrentStreamHandle {
            inner,
            resolver: Arc::clone(&self.resolver),
            snapshot,
            invalidated: false,
        })
    }
}

#[async_trait]
impl RpcClient for ProductionRpcClient {
    async fn call(&self, payload: Vec<u8>) -> Result<Vec<u8>> {
        Ok(self
            .attempt(|c| {
                let p = payload.clone();
                async move { c.call(p).await }
            })
            .await?
            .0)
    }
    async fn call_for_service(&self, service: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        Ok(self
            .attempt(|c| {
                let p = payload.clone();
                let s = service.clone();
                async move { c.call_for_service(&s, p).await }
            })
            .await?
            .0)
    }
    async fn call_for_service_with_method(
        &self,
        service: &str,
        method_discriminator: u16,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        Ok(self
            .attempt(|c| {
                let p = payload.clone();
                let s = service.clone();
                async move {
                    c.call_for_service_with_method(&s, method_discriminator, p)
                        .await
                }
            })
            .await?
            .0)
    }

    async fn call_with_options(&self, payload: Vec<u8>, options: CallOptions) -> Result<Vec<u8>> {
        Ok(self
            .attempt(|c| {
                let p = payload.clone();
                let o = options.clone();
                async move { c.call_with_options(p, o).await }
            })
            .await?
            .0)
    }
    async fn call_with_options_for_service(
        &self,
        service: &str,
        payload: Vec<u8>,
        options: CallOptions,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        Ok(self
            .attempt(|c| {
                let p = payload.clone();
                let o = options.clone();
                let s = service.clone();
                async move { c.call_with_options_for_service(&s, p, o).await }
            })
            .await?
            .0)
    }
    async fn call_streaming(&self, payload: Vec<u8>, ephemeral: [u8; 32]) -> Result<Vec<u8>> {
        Ok(self
            .attempt(|c| {
                let p = payload.clone();
                async move { c.call_streaming(p, ephemeral).await }
            })
            .await?
            .0)
    }
    async fn call_streaming_for_service(
        &self,
        service: &str,
        payload: Vec<u8>,
        ephemeral: [u8; 32],
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        Ok(self
            .attempt(|c| {
                let p = payload.clone();
                let s = service.clone();
                async move { c.call_streaming_for_service(&s, p, ephemeral).await }
            })
            .await?
            .0)
    }
    async fn call_streaming_for_service_with_method(
        &self,
        service: &str,
        method_discriminator: u16,
        payload: Vec<u8>,
        ephemeral: [u8; 32],
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        Ok(self
            .attempt(|c| {
                let p = payload.clone();
                let s = service.clone();
                async move {
                    c.call_streaming_for_service_with_method(&s, method_discriminator, p, ephemeral)
                        .await
                }
            })
            .await?
            .0)
    }
    async fn open_stream(&self, payload: Vec<u8>) -> Result<Box<dyn StreamHandle>> {
        let (inner, snapshot) = self
            .attempt(|c| {
                let p = payload.clone();
                async move { c.open_stream(p).await }
            })
            .await?;
        Ok(self.checked_stream(inner, snapshot))
    }
    async fn open_stream_from_info(
        &self,
        info: hyprstream_rpc::stream_info::StreamInfo,
        secret: [u8; 32],
        public: [u8; 32],
    ) -> Result<Box<dyn StreamHandle>> {
        let (inner, snapshot) = self
            .attempt(|c| {
                let i = info.clone();
                async move { c.open_stream_from_info(i, secret, public).await }
            })
            .await?;
        Ok(self.checked_stream(inner, snapshot))
    }
    fn next_id(&self) -> u64 {
        self.request_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

impl DiscoveryService {
    fn resolve_announced_endpoint(
        &self,
        name: &str,
        kind: SocketKind,
    ) -> anyhow::Result<Option<TransportConfig>> {
        let wanted = socket_kind_to_string(kind);
        let announced = self.announced_endpoints.read();
        let Some(endpoints) = announced.get(name) else {
            return Ok(None);
        };

        let Some(endpoint) = endpoints
            .iter()
            .filter(|ep| ep.socket_kind == wanted)
            .find(|ep| ep.last_heartbeat.elapsed() <= ANNOUNCED_ENDPOINT_TTL)
        else {
            return Ok(None);
        };

        Ok(Some(announced_endpoint_to_transport(endpoint)?))
    }
}

fn announced_endpoint_to_transport(
    endpoint: &AnnouncedEndpoint,
) -> anyhow::Result<TransportConfig> {
    match endpoint.socket_kind.as_str() {
        "quic" => parse_announced_quic(&endpoint.endpoint),
        "iroh" => parse_announced_iroh(&endpoint.endpoint),
        _ => Ok(TransportConfig::from_endpoint(&endpoint.endpoint)),
    }
}

fn parse_announced_quic(endpoint: &str) -> anyhow::Result<TransportConfig> {
    let rest = endpoint
        .strip_prefix("quic://")
        .ok_or_else(|| anyhow::anyhow!("announced QUIC endpoint must start with quic://"))?;
    let (reach, encoded_hashes) = rest
        .split_once('#')
        .map_or((rest, None), |(reach, hashes)| (reach, Some(hashes)));
    let (server_name, addr) = reach.split_once(':').ok_or_else(|| {
        anyhow::anyhow!("announced QUIC endpoint must be quic://<server-name>:<socket-addr>")
    })?;
    anyhow::ensure!(
        !server_name.is_empty(),
        "announced QUIC endpoint is missing server name"
    );
    let addr = SocketAddr::from_str(addr)
        .map_err(|e| anyhow::anyhow!("invalid announced QUIC socket address '{addr}': {e}"))?;
    anyhow::ensure!(
        addr.port() != 0,
        "announced QUIC endpoint must not use port 0"
    );
    let transport = if let Some(encoded_hashes) = encoded_hashes {
        let hashes = encoded_hashes
            .split(',')
            .map(|value| {
                URL_SAFE_NO_PAD
                    .decode(value)
                    .context("invalid announced QUIC certificate hash")?
                    .try_into()
                    .map_err(|_| {
                        anyhow::anyhow!("announced QUIC certificate hash must be 32 bytes")
                    })
            })
            .collect::<Result<Vec<[u8; 32]>>>()?;
        anyhow::ensure!(!hashes.is_empty(), "announced QUIC pin set is empty");
        TransportConfig::quic_with_auth(
            addr,
            server_name,
            hyprstream_rpc::transport::QuicServerAuth::pinned(hashes)?,
        )
    } else {
        TransportConfig::quic(addr, server_name).with_connect_mode()
    };
    Ok(transport)
}

fn parse_announced_iroh(endpoint: &str) -> anyhow::Result<TransportConfig> {
    let hex = endpoint
        .strip_prefix("iroh://")
        .ok_or_else(|| anyhow::anyhow!("announced iroh endpoint must start with iroh://"))?;
    anyhow::ensure!(
        hex.len() == 64,
        "announced iroh node id must be 32 bytes of hex"
    );
    let mut node_id = [0u8; 32];
    for (idx, byte) in node_id.iter_mut().enumerate() {
        let start = idx * 2;
        *byte = u8::from_str_radix(&hex[start..start + 2], 16)
            .map_err(|e| anyhow::anyhow!("invalid announced iroh node id hex: {e}"))?;
    }
    Ok(TransportConfig::iroh(node_id, Vec::new(), None))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod resolver_tests {
    use super::*;
    use ed25519_dalek::Signer as _;
    use hyprstream_crypto::pq::ml_dsa_sk_to_vk_bytes;
    use hyprstream_pds::at9p::{
        CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType,
        Transport as At9pTransport,
    };
    use hyprstream_pds::at9p_duplicity::AcceptedAt9pState;
    use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};
    use hyprstream_rpc::transport::{BindMode, EndpointType};

    fn checked_test_time(now: i64, offset: i64) -> i64 {
        now.checked_add(offset).expect("test NumericDate offset")
    }

    fn sign_registry_credential_json(
        ca: &SigningKey,
        protected_json: &str,
        claims_json: &str,
    ) -> String {
        let protected = URL_SAFE_NO_PAD.encode(protected_json.as_bytes());
        let claims = URL_SAFE_NO_PAD.encode(claims_json.as_bytes());
        let signing_input = format!("{protected}.{claims}");
        let signature = ca.sign(signing_input.as_bytes());
        format!(
            "{signing_input}.{}",
            URL_SAFE_NO_PAD.encode(signature.to_bytes())
        )
    }

    fn exact_registry_credential_values(
        ca: &SigningKey,
        registry: &SigningKey,
    ) -> (serde_json::Value, serde_json::Value) {
        let domain = deployment_domain(&ca.verifying_key());
        let now = chrono::Utc::now().timestamp();
        (
            serde_json::json!({"alg":"EdDSA", "typ":"wit+jwt", "kid":domain}),
            serde_json::json!({
                "iss": format!("urn:hyprstream:deployment:{domain}"),
                "sub": "service:registry",
                "aud": REGISTRY_DEPLOYMENT_CREDENTIAL_AUDIENCE,
                "exp": checked_test_time(now, 3600),
                "nbf": now,
                "iat": now,
                "deployment_domain": domain,
                "profile": REGISTRY_DEPLOYMENT_CREDENTIAL_PROFILE,
                "cnf": {"jwk": {
                    "kty": "OKP",
                    "crv": "Ed25519",
                    "x": URL_SAFE_NO_PAD.encode(registry.verifying_key().as_bytes()),
                }},
            }),
        )
    }

    fn exact_registry_credential(ca: &SigningKey, registry: &SigningKey) -> String {
        let (protected, claims) = exact_registry_credential_values(ca, registry);
        sign_registry_credential_json(ca, &protected.to_string(), &claims.to_string())
    }

    fn authenticate_test_registry_credential(
        ca: &SigningKey,
        credential: String,
    ) -> Result<AuthenticatedRegistryDeploymentIdentity> {
        authenticate_registry_deployment_credentials(TrustedRegistryDeploymentCredentials {
            ca_verifying_key: ca.verifying_key(),
            registry_credential: credential,
        })
    }

    fn assert_registry_credential_rejected(
        ca: &SigningKey,
        protected: &serde_json::Value,
        claims: &serde_json::Value,
        case: &str,
    ) {
        let credential =
            sign_registry_credential_json(ca, &protected.to_string(), &claims.to_string());
        assert!(
            authenticate_test_registry_credential(ca, credential).is_err(),
            "invalid deployment credential accepted: {case}"
        );
    }

    #[test]
    fn exact_registry_deployment_credential_installs_the_exact_bound_jwk() {
        let ca = SigningKey::from_bytes(&[0x31; 32]);
        let registry = SigningKey::from_bytes(&[0x32; 32]);
        let witness =
            authenticate_test_registry_credential(&ca, exact_registry_credential(&ca, &registry))
                .expect("exact deployment credential profile");
        assert!(witness.verifier.matches(&registry.verifying_key()));
    }

    #[test]
    fn deployment_credential_numeric_dates_enforce_all_boundaries_and_orderings() {
        let now = 1_000_000;

        for (case, exp, nbf, iat) in [
            (
                "one-hour lifetime endpoint",
                checked_test_time(now, 3600),
                now,
                now,
            ),
            (
                "inside lifetime endpoint",
                checked_test_time(now, 3599),
                now,
                now,
            ),
            (
                "future-skew and lifetime endpoints",
                checked_test_time(now, 3660),
                checked_test_time(now, 60),
                checked_test_time(now, 60),
            ),
        ] {
            validate_registry_credential_numeric_dates(exp, nbf, iat, now)
                .unwrap_or_else(|error| panic!("valid {case} rejected: {error}"));
        }

        for (case, exp, nbf, iat, validation_now) in [
            (
                "expired endpoint",
                now,
                now,
                checked_test_time(now, -1),
                now,
            ),
            (
                "outside lifetime endpoint",
                checked_test_time(now, 3601),
                now,
                now,
                now,
            ),
            (
                "future nbf outside skew",
                checked_test_time(now, 3600),
                checked_test_time(now, 61),
                now,
                now,
            ),
            (
                "future iat outside skew",
                checked_test_time(now, 3661),
                now,
                checked_test_time(now, 61),
                now,
            ),
            (
                "exp equal to iat",
                checked_test_time(now, 1),
                checked_test_time(now, 1),
                checked_test_time(now, 1),
                now,
            ),
            (
                "exp before iat",
                checked_test_time(now, 1),
                now,
                checked_test_time(now, 2),
                now,
            ),
            (
                "nbf after exp",
                checked_test_time(now, 2),
                checked_test_time(now, 3),
                now,
                now,
            ),
            (
                "nbf after iat",
                checked_test_time(now, 2),
                checked_test_time(now, 1),
                now,
                now,
            ),
            ("negative exp", -1, 0, 0, -2),
            ("negative nbf", checked_test_time(now, 1), -1, 0, now),
            ("negative iat", checked_test_time(now, 1), 0, -1, now),
            ("minimum and maximum extremes", i64::MAX, 0, i64::MIN, now),
            ("all minimum extremes", i64::MIN, i64::MIN, i64::MIN, now),
            ("all maximum extremes", i64::MAX, i64::MAX, i64::MAX, now),
            (
                "clock-skew boundary overflow",
                i64::MAX,
                i64::MAX - 1,
                i64::MAX - 1,
                i64::MAX - 1,
            ),
        ] {
            assert!(
                validate_registry_credential_numeric_dates(exp, nbf, iat, validation_now).is_err(),
                "invalid NumericDate profile accepted: {case}"
            );
        }
    }

    #[test]
    fn deployment_credential_rejects_signed_wraparound_lifetime_in_all_profiles() {
        let ca = SigningKey::from_bytes(&[0x6a; 32]);
        let registry = SigningKey::from_bytes(&[0x6b; 32]);
        let (protected, mut claims) = exact_registry_credential_values(&ca, &registry);
        claims["exp"] = serde_json::json!(checked_test_time(
            chrono::Utc::now().timestamp(),
            REGISTRY_DEPLOYMENT_CREDENTIAL_MAX_TTL_SECONDS,
        ));
        claims["nbf"] = serde_json::json!(i64::MIN);
        claims["iat"] = serde_json::json!(i64::MIN);

        assert_registry_credential_rejected(
            &ca,
            &protected,
            &claims,
            "signed exp-minus-iat wraparound",
        );
    }

    #[test]
    fn deployment_credential_rejects_every_protected_and_trust_dimension() {
        let ca = SigningKey::from_bytes(&[0x33; 32]);
        let other_ca = SigningKey::from_bytes(&[0x34; 32]);
        let registry = SigningKey::from_bytes(&[0x35; 32]);
        let (protected, claims) = exact_registry_credential_values(&ca, &registry);

        for (case, member, value) in [
            ("access token type", "typ", serde_json::json!("at+jwt")),
            (
                "application access token type",
                "typ",
                serde_json::json!("application/at+jwt"),
            ),
            ("generic JWT type", "typ", serde_json::json!("JWT")),
            ("wrong-case type", "typ", serde_json::json!("WIT+JWT")),
            ("whitespace type", "typ", serde_json::json!("wit+jwt ")),
            ("alternate algorithm", "alg", serde_json::json!("ES256")),
            ("wrong key id", "kid", serde_json::json!("attacker")),
        ] {
            let mut changed = protected.clone();
            changed[member] = value;
            assert_registry_credential_rejected(&ca, &changed, &claims, case);
        }
        for missing in ["typ", "alg", "kid"] {
            let mut changed = protected.clone();
            changed
                .as_object_mut()
                .expect("header object")
                .remove(missing);
            assert_registry_credential_rejected(&ca, &changed, &claims, missing);
        }
        let mut unknown = protected.clone();
        unknown["crit"] = serde_json::json!([]);
        assert_registry_credential_rejected(&ca, &unknown, &claims, "unknown header member");

        let credential = exact_registry_credential(&ca, &registry);
        assert!(
            authenticate_test_registry_credential(&other_ca, credential).is_err(),
            "CA substitution accepted"
        );
        let signed_by_other = {
            let (other_header, _) = exact_registry_credential_values(&other_ca, &registry);
            sign_registry_credential_json(&other_ca, &other_header.to_string(), &claims.to_string())
        };
        assert!(
            authenticate_test_registry_credential(&ca, signed_by_other).is_err(),
            "signature substitution accepted"
        );
    }

    #[test]
    fn deployment_credential_rejects_every_claim_and_time_dimension() {
        let ca = SigningKey::from_bytes(&[0x36; 32]);
        let registry = SigningKey::from_bytes(&[0x37; 32]);
        let (protected, claims) = exact_registry_credential_values(&ca, &registry);
        let now = chrono::Utc::now().timestamp();

        for (case, member, value) in [
            ("wrong subject", "sub", serde_json::json!("service:model")),
            (
                "wrong issuer",
                "iss",
                serde_json::json!("https://attacker.invalid"),
            ),
            (
                "wrong audience",
                "aud",
                serde_json::json!("service:registry"),
            ),
            (
                "audience array",
                "aud",
                serde_json::json!([REGISTRY_DEPLOYMENT_CREDENTIAL_AUDIENCE]),
            ),
            (
                "wrong deployment domain",
                "deployment_domain",
                serde_json::json!("attacker"),
            ),
            (
                "wrong profile",
                "profile",
                serde_json::json!("hyprstream.registry-deployment.v2"),
            ),
            (
                "expired",
                "exp",
                serde_json::json!(checked_test_time(now, -1)),
            ),
            (
                "future nbf",
                "nbf",
                serde_json::json!(checked_test_time(now, 61)),
            ),
            (
                "future iat",
                "iat",
                serde_json::json!(checked_test_time(now, 61)),
            ),
            (
                "nbf after iat",
                "nbf",
                serde_json::json!(checked_test_time(now, 1)),
            ),
            (
                "excessive lifetime",
                "exp",
                serde_json::json!(checked_test_time(now, 3601)),
            ),
        ] {
            let mut changed = claims.clone();
            changed[member] = value;
            assert_registry_credential_rejected(&ca, &protected, &changed, case);
        }
        for missing in [
            "iss",
            "sub",
            "aud",
            "exp",
            "nbf",
            "iat",
            "deployment_domain",
            "profile",
            "cnf",
        ] {
            let mut changed = claims.clone();
            changed
                .as_object_mut()
                .expect("claims object")
                .remove(missing);
            assert_registry_credential_rejected(&ca, &protected, &changed, missing);
        }
        let mut unknown = claims.clone();
        unknown["scope"] = serde_json::json!("registry");
        assert_registry_credential_rejected(&ca, &protected, &unknown, "unknown claim");
    }

    #[test]
    fn deployment_credential_rejects_every_confirmation_and_jwk_dimension() {
        let ca = SigningKey::from_bytes(&[0x38; 32]);
        let registry = SigningKey::from_bytes(&[0x39; 32]);
        let (protected, claims) = exact_registry_credential_values(&ca, &registry);

        for (case, member, value) in [
            ("RSA declaration", "kty", serde_json::json!("RSA")),
            ("EC declaration", "kty", serde_json::json!("EC")),
            ("P-256 curve", "crv", serde_json::json!("P-256")),
            ("alternate OKP curve", "crv", serde_json::json!("X25519")),
            ("malformed x", "x", serde_json::json!("%%%")),
            (
                "padded x",
                "x",
                serde_json::json!(format!(
                    "{}=",
                    URL_SAFE_NO_PAD.encode(registry.verifying_key().as_bytes())
                )),
            ),
            (
                "short x",
                "x",
                serde_json::json!(URL_SAFE_NO_PAD.encode([0u8; 31])),
            ),
        ] {
            let mut changed = claims.clone();
            changed["cnf"]["jwk"][member] = value;
            assert_registry_credential_rejected(&ca, &protected, &changed, case);
        }
        for missing in ["kty", "crv", "x"] {
            let mut changed = claims.clone();
            changed["cnf"]["jwk"]
                .as_object_mut()
                .expect("jwk object")
                .remove(missing);
            assert_registry_credential_rejected(&ca, &protected, &changed, missing);
        }
        for incompatible in ["alg", "use", "key_ops", "kid"] {
            let mut changed = claims.clone();
            changed["cnf"]["jwk"][incompatible] = serde_json::json!("unexpected");
            assert_registry_credential_rejected(&ca, &protected, &changed, incompatible);
        }
        let mut ambiguous = claims.clone();
        ambiguous["cnf"]["jkt"] = serde_json::json!("thumbprint");
        assert_registry_credential_rejected(&ca, &protected, &ambiguous, "jwk plus jkt");
        let mut alternate = claims.clone();
        alternate["cnf"] = serde_json::json!({"jkt":"thumbprint"});
        assert_registry_credential_rejected(&ca, &protected, &alternate, "jkt only");
    }

    #[test]
    fn deployment_credential_duplicate_members_fail_closed_at_every_level() {
        let ca = SigningKey::from_bytes(&[0x3a; 32]);
        let registry = SigningKey::from_bytes(&[0x3b; 32]);
        let domain = deployment_domain(&ca.verifying_key());
        let (_, claims) = exact_registry_credential_values(&ca, &registry);
        let duplicate_headers = [
            format!(r#"{{"alg":"EdDSA","alg":"ES256","typ":"wit+jwt","kid":"{domain}"}}"#),
            format!(r#"{{"alg":"EdDSA","typ":"wit+jwt","typ":"at+jwt","kid":"{domain}"}}"#),
            format!(r#"{{"alg":"EdDSA","typ":"wit+jwt","kid":"{domain}","kid":"attacker"}}"#),
        ];
        for header in duplicate_headers {
            let token = sign_registry_credential_json(&ca, &header, &claims.to_string());
            assert!(authenticate_test_registry_credential(&ca, token).is_err());
        }

        let now = chrono::Utc::now().timestamp();
        let exp = checked_test_time(now, 3600);
        let x = URL_SAFE_NO_PAD.encode(registry.verifying_key().as_bytes());
        let protected = serde_json::json!({"alg":"EdDSA", "typ":"wit+jwt", "kid":domain});
        for duplicate_claims in [
            format!(
                r#"{{"iss":"urn:hyprstream:deployment:{domain}","sub":"service:registry","sub":"service:model","aud":"{REGISTRY_DEPLOYMENT_CREDENTIAL_AUDIENCE}","exp":{},"nbf":{now},"iat":{now},"deployment_domain":"{domain}","profile":"{REGISTRY_DEPLOYMENT_CREDENTIAL_PROFILE}","cnf":{{"jwk":{{"kty":"OKP","crv":"Ed25519","x":"{x}"}}}}}}"#,
                exp
            ),
            format!(
                r#"{{"iss":"urn:hyprstream:deployment:{domain}","sub":"service:registry","aud":"{REGISTRY_DEPLOYMENT_CREDENTIAL_AUDIENCE}","exp":{},"nbf":{now},"iat":{now},"deployment_domain":"{domain}","profile":"{REGISTRY_DEPLOYMENT_CREDENTIAL_PROFILE}","cnf":{{"jwk":{{"kty":"OKP","crv":"Ed25519","x":"{x}"}},"jwk":{{"kty":"RSA","crv":"P-256","x":"{x}"}}}}}}"#,
                exp
            ),
            format!(
                r#"{{"iss":"urn:hyprstream:deployment:{domain}","sub":"service:registry","aud":"{REGISTRY_DEPLOYMENT_CREDENTIAL_AUDIENCE}","exp":{},"nbf":{now},"iat":{now},"deployment_domain":"{domain}","profile":"{REGISTRY_DEPLOYMENT_CREDENTIAL_PROFILE}","cnf":{{"jwk":{{"kty":"OKP","kty":"RSA","crv":"Ed25519","x":"{x}"}}}}}}"#,
                exp
            ),
        ] {
            let token =
                sign_registry_credential_json(&ca, &protected.to_string(), &duplicate_claims);
            assert!(authenticate_test_registry_credential(&ca, token).is_err());
        }
    }

    #[test]
    fn production_authority_api_inventory_has_no_public_composition_chain() {
        let service_factory = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../hyprstream-service/src/service/factory.rs"
        ));
        for forbidden in [
            "pub fn seal_discovery_bootstrap_authority",
            "pub fn consume_discovery_bootstrap_authority",
            "struct DiscoveryBootstrapAuthority",
            "discovery_authority:",
            "pub fn authenticate_registry_deployment_credential",
            "pub fn take_authenticated_registry_identity",
            "pub fn authenticated_registry_identity",
            "pub struct AuthenticatedRegistryDeploymentIdentity",
            "into_verifying_key",
        ] {
            assert!(
                !service_factory.contains(forbidden),
                "authority seam returned: {forbidden}"
            );
        }
        let discovery = include_str!("service.rs");
        let production = discovery
            .split("mod resolver_tests {")
            .next()
            .expect("production Discovery source");
        assert!(!production.contains("\n    pub fn install_process_bootstrap("));
        assert!(production.contains("struct ProcessBootstrapAuthority {"));
        assert!(production.contains("enum ProcessBootstrapAuthorityState {"));
        let authentication = production
            .split("fn authenticate_deployment_bootstrap()")
            .nth(1)
            .expect("private deployment authentication boundary")
            .split("#[cfg(test)]\nfn authenticate_discovery_bootstrap_identity")
            .next()
            .expect("authentication body");
        assert!(authentication.contains("hyprstream_service::deployment_data_dir()"));
        assert!(authentication.contains("load_trusted_registry_deployment_credentials()"));
        assert!(authentication.contains("authenticate_registry_deployment_credentials("));
        assert!(!authentication.contains("global_trust_store()"));
        assert!(!authentication.contains("resolve_one("));
        assert!(!authentication.contains("FnOnce"));
        assert!(!production.contains("pub fn authenticate_discovery_bootstrap("));
        assert!(!production.contains("pub struct AuthenticatedDiscoveryBootstrap"));
        assert!(!production.contains("pub fn bootstrap_authenticated_process("));
        assert!(production.contains("/etc/hyprstream/trust/deployment-ca.ed25519"));
        assert!(production.contains("/run/hyprstream/credentials/registry-service.jwt"));
        let loader = production
            .split("fn load_trusted_registry_deployment_credentials()")
            .nth(1)
            .expect("fixed deployment loader")
            .split("fn authenticate_registry_deployment_credentials(")
            .next()
            .expect("loader body");
        assert!(!loader.contains("CREDENTIALS_DIRECTORY"));
        assert!(!loader.contains("dirs::config_dir"));
    }

    fn service() -> DiscoveryService {
        let (sk, vk) = hyprstream_rpc::crypto::generate_signing_keypair();
        DiscoveryService::new(Arc::new(sk), vk, TransportConfig::inproc("resolver-test"))
    }

    fn legacy_endpoint(
        socket_kind: &str,
        endpoint: &str,
        last_heartbeat: Instant,
    ) -> AnnouncedEndpoint {
        AnnouncedEndpoint {
            socket_kind: socket_kind.to_owned(),
            endpoint: endpoint.to_owned(),
            service_jwt: "jwt".to_owned(),
            service_did: Did::default(),
            capabilities: BTreeSet::new(),
            accepted_state_digest: Vec::new(),
            accepted_state_epoch: 0,
            response_key_id: String::new(),
            request_kem_key_id: String::new(),
            request_kem_recipient: Vec::new(),
            expires_at_unix_ms: 0,
            source_signer: [0; 32],
            last_heartbeat,
        }
    }

    fn accepted_state(tag: u8) -> (AcceptedAt9pState, SigningKey) {
        let signing = SigningKey::from_bytes(&[tag; 32]);
        let pq_signing = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&signing);
        let keys = HybridKeyPair::new(
            signing.verifying_key().to_bytes().to_vec(),
            ml_dsa_sk_to_vk_bytes(&pq_signing),
        )
        .unwrap_or_else(|e| panic!("test hybrid keys invalid: {e}"));
        let endpoint = ServiceEndpoint::new(At9pTransport::Iroh, "iroh://reach")
            .unwrap_or_else(|e| panic!("test endpoint invalid: {e}"));
        let service = ServiceEntry::new("#model", ServiceType::NinePExport, endpoint)
            .unwrap_or_else(|e| panic!("test service invalid: {e}"));
        let body = CapsuleBody::new(vec![keys], vec![service])
            .unwrap_or_else(|e| panic!("test body invalid: {e}"));
        let genesis = sign_capsule(body.clone(), &signing, &pq_signing)
            .unwrap_or_else(|e| panic!("test genesis signing failed: {e}"));
        let subject = genesis
            .cid512()
            .unwrap_or_else(|e| panic!("test genesis CID failed: {e}"));
        let update = sign_update_record(
            subject,
            1,
            [1; 64],
            body,
            "2099-01-01T00:00:00Z".to_owned(),
            &signing,
            &pq_signing,
        )
        .unwrap_or_else(|e| panic!("test update signing failed: {e}"));
        let bytes = update
            .to_dag_cbor()
            .unwrap_or_else(|e| panic!("test update encoding failed: {e}"));
        let state = AcceptedAt9pState::from_persisted_update(&bytes)
            .unwrap_or_else(|e| panic!("test accepted state invalid: {e}"));
        (state, signing)
    }

    struct MutableAcceptedState(parking_lot::Mutex<Option<AcceptedAt9pState>>);

    impl AcceptedStateSource for MutableAcceptedState {
        fn accepted_state(&self, _did: &str) -> Result<Option<AcceptedAt9pState>> {
            Ok(self.0.lock().clone())
        }
    }

    fn production_fixture(
        local_reach: bool,
    ) -> (DiscoveryServiceResolver, Arc<MutableAcceptedState>) {
        let (state, signing) = accepted_state(11);
        let kem = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(
            hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
        )
        .unwrap_or_else(|e| panic!("test KEM generation failed: {e}"));
        let endpoint = if local_reach {
            "inproc://hyprstream/model".to_owned()
        } else {
            "quic://localhost:127.0.0.1:9".to_owned()
        };
        let announced = Arc::new(RwLock::new(HashMap::from([(
            "model".to_owned(),
            vec![AnnouncedEndpoint {
                socket_kind: if local_reach { "rep" } else { "quic" }.to_owned(),
                endpoint,
                service_jwt: "verified-by-handler".to_owned(),
                service_did: Did::from(state.did.clone()),
                capabilities: ["hyprstream-rpc/1".to_owned(), "hyprstream-moq/1".to_owned()]
                    .into_iter()
                    .collect(),
                accepted_state_digest: state.head_digest.to_vec(),
                accepted_state_epoch: state.epoch,
                response_key_id: format!("{}#response-current", state.did),
                request_kem_key_id: format!("{}#kem-current", state.did),
                request_kem_recipient: kem.public().encode(),
                expires_at_unix_ms: 4_070_908_800_000,
                source_signer: signing.verifying_key().to_bytes(),
                last_heartbeat: Instant::now(),
            }],
        )])));
        let source = Arc::new(MutableAcceptedState(parking_lot::Mutex::new(Some(state))));
        (
            DiscoveryServiceResolver {
                announced_endpoints: announced,
                accepted_state_source: Arc::clone(&source) as Arc<dyn AcceptedStateSource>,
                discovery_client: None,
            },
            source,
        )
    }

    #[tokio::test]
    async fn production_resolver_joins_announcement_to_current_pds_state() {
        let (resolver, _) = production_fixture(false);
        let resolved = resolver
            .resolve_service(ServiceQuery::network("model").expect("query"))
            .await
            .unwrap_or_else(|e| panic!("production candidate rejected: {e}"));
        assert_eq!(resolved.service_name(), "model");
        assert_eq!(resolved.evidence().accepted_state_epoch, 1);
        assert!(resolved.service_did().is_did_at9p());
        assert!(!resolved.request_kem_recipient().recipient.eks.is_empty());
        resolver
            .ensure_current(&resolved)
            .await
            .unwrap_or_else(|e| panic!("unchanged accepted state rejected: {e}"));
    }

    fn owned_browser_request() -> BrowserProvisioningRequest {
        BrowserProvisioningRequest::new(
            "model",
            "hyprstream-rpc/1",
            "model",
            BrowserCarrierProfile::OwnedHybridWebTransport,
        )
        .expect("browser request")
    }

    #[tokio::test]
    async fn production_browser_projection_is_current_and_resolution_bound() {
        let (resolver, _) = production_fixture(false);
        let document = resolver
            .browser_provisioning(owned_browser_request())
            .await
            .unwrap_or_else(|error| panic!("browser projection failed: {error}"))
            .sign_projection(&SigningKey::from_bytes(&[11; 32]))
            .expect("sign accepted projection");
        let json = serde_json::to_vec(&document).expect("serialize provisioning");
        let validated = hyprstream_rpc::browser_provisioning::BrowserProvisioning::from_json(
            &json,
            &owned_browser_request(),
            unix_millis_now(),
        )
        .unwrap_or_else(|error| panic!("browser projection did not validate: {error}"));
        assert_eq!(validated.service_name(), "model");
        assert_eq!(validated.accepted_state_epoch(), 1);
        assert!(validated.service_did().starts_with("did:at9p:"));
    }

    #[tokio::test]
    async fn browser_projection_rejects_state_advance_revocation_and_cross_service() {
        let (resolver, source) = production_fixture(false);
        source.0.lock().as_mut().expect("fixture state").epoch += 1;
        assert!(resolver
            .browser_provisioning(owned_browser_request())
            .await
            .is_err());

        let (resolver, _) = production_fixture(false);
        resolver
            .announced_endpoints
            .write()
            .get_mut("model")
            .and_then(|entries| entries.first_mut())
            .expect("fixture announcement")
            .request_kem_recipient = vec![0x01];
        assert!(resolver
            .browser_provisioning(owned_browser_request())
            .await
            .is_err());

        let (resolver, source) = production_fixture(false);
        source
            .0
            .lock()
            .as_mut()
            .expect("fixture state")
            .current
            .services[0]
            .id = "#policy".to_owned();
        assert!(resolver
            .browser_provisioning(owned_browser_request())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn public_relay_requires_signed_moq_route_and_application_object_profile() {
        let (resolver, source) = production_fixture(false);
        let request = BrowserProvisioningRequest::new(
            "model",
            "hyprstream-moq/1",
            "tenant-a/track-a",
            BrowserCarrierProfile::StandardPublicRelay,
        )
        .expect("relay request");
        assert!(resolver
            .browser_provisioning(request.clone())
            .await
            .is_err());

        {
            let mut state = source.0.lock();
            let endpoint = &mut state.as_mut().expect("fixture state").current.services[0].endpoint;
            endpoint.transport = At9pTransport::Moq;
            endpoint.address = "https://model.example/".to_owned();
            endpoint.relay = Some("https://relay.example/moq".to_owned());
        }
        let document = resolver
            .browser_provisioning(request.clone())
            .await
            .unwrap_or_else(|error| panic!("signed relay projection failed: {error}"))
            .sign_projection(&SigningKey::from_bytes(&[11; 32]))
            .expect("sign accepted relay projection");
        assert_eq!(
            document.transport_security,
            BrowserTransportSecurity::ClassicalUntrusted
        );
        assert!(document.application_hybrid_required);
        assert!(document.encrypted_objects_required);
        let json = serde_json::to_vec(&document).expect("serialize relay provisioning");
        hyprstream_rpc::browser_provisioning::BrowserProvisioning::from_json(
            &json,
            &request,
            unix_millis_now(),
        )
        .expect("validate relay provisioning");
    }

    #[tokio::test]
    async fn post_refetch_pre_dispatch_state_advance_rejects_pinned_binding() {
        let (resolver, source) = production_fixture(false);
        let document = resolver
            .browser_provisioning(owned_browser_request())
            .await
            .expect("pre-seal refetch")
            .sign_projection(&SigningKey::from_bytes(&[11; 32]))
            .expect("sign projection");
        let json = serde_json::to_vec(&document).expect("serialize projection");
        let binding = hyprstream_rpc::browser_provisioning::BrowserProvisioning::from_json(
            &json,
            &owned_browser_request(),
            unix_millis_now(),
        )
        .expect("validate projection")
        .request_binding()
        .expect("bind accepted evidence");

        source.0.lock().as_mut().expect("fixture state").epoch += 1;
        assert!(resolver.verify_browser_binding(&binding).await.is_err());
    }

    #[tokio::test]
    async fn post_refetch_route_and_pin_rotation_rejects_dial_evidence() {
        async fn binding_for(
            resolver: &DiscoveryServiceResolver,
        ) -> hyprstream_rpc::browser_provisioning::BrowserRequestBinding {
            let document = resolver
                .browser_provisioning(owned_browser_request())
                .await
                .expect("pre-seal refetch")
                .sign_projection(&SigningKey::from_bytes(&[11; 32]))
                .expect("sign projection");
            let json = serde_json::to_vec(&document).expect("serialize projection");
            hyprstream_rpc::browser_provisioning::BrowserProvisioning::from_json(
                &json,
                &owned_browser_request(),
                unix_millis_now(),
            )
            .expect("validate projection")
            .request_binding()
            .expect("bind dial evidence")
        }

        let (resolver, _) = production_fixture(false);
        let route_binding = binding_for(&resolver).await;
        resolver
            .announced_endpoints
            .write()
            .get_mut("model")
            .and_then(|entries| entries.first_mut())
            .expect("fixture route")
            .endpoint = "quic://localhost:127.0.0.1:10".to_owned();
        assert!(resolver
            .verify_browser_binding(&route_binding)
            .await
            .is_err());

        let (resolver, _) = production_fixture(false);
        resolver
            .announced_endpoints
            .write()
            .get_mut("model")
            .and_then(|entries| entries.first_mut())
            .expect("fixture pin")
            .endpoint = format!(
            "quic://localhost:127.0.0.1:9#{}",
            URL_SAFE_NO_PAD.encode([0x51; 32])
        );
        let pin_binding = binding_for(&resolver).await;
        resolver
            .announced_endpoints
            .write()
            .get_mut("model")
            .and_then(|entries| entries.first_mut())
            .expect("fixture pin rotation")
            .endpoint = format!(
            "quic://localhost:127.0.0.1:9#{}",
            URL_SAFE_NO_PAD.encode([0x52; 32])
        );
        assert!(resolver.verify_browser_binding(&pin_binding).await.is_err());
    }

    #[tokio::test]
    async fn ordinary_announcement_handler_populates_production_resolver() {
        let (state, service_signing) = accepted_state(12);
        let root = SigningKey::from_bytes(&[0x61; 32]);
        let source = Arc::new(MutableAcceptedState(parking_lot::Mutex::new(Some(
            state.clone(),
        ))));
        let service = DiscoveryService::new(
            Arc::new(root.clone()),
            root.verifying_key(),
            TransportConfig::inproc("announcement-rpc-test"),
        )
        .with_accepted_state_source(source);
        let claims = hyprstream_rpc::auth::Claims::new(
            "service:model".to_owned(),
            chrono::Utc::now().timestamp(),
            chrono::Utc::now().timestamp() + 3_600,
        )
        .with_cnf_jwk(service_signing.verifying_key().as_bytes());
        let jwt = hyprstream_rpc::auth::jwt::encode_service_jwt(&claims, &root);
        let service_pq_signing =
            hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&service_signing);
        let signed = hyprstream_rpc::SignedEnvelope::new_signed_hybrid(
            hyprstream_rpc::RequestEnvelope::anonymous(Vec::new()),
            &service_signing,
            &service_pq_signing,
        );
        let ctx = EnvelopeContext::from_verified_as_system(&signed);
        let kem = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(
            hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
        )
        .expect("test KEM");
        let response = service
            .handle_announce(
                &ctx,
                1,
                &ServiceAnnouncement {
                    service_name: "model".to_owned(),
                    socket_kind: "quic".to_owned(),
                    endpoint: "quic://localhost:127.0.0.1:9".to_owned(),
                    service_jwt: Some(jwt),
                    service_did: Did::from(state.did.clone()),
                    capabilities: vec!["hyprstream-rpc/1".to_owned()],
                    accepted_state_digest: state.head_digest.to_vec(),
                    accepted_state_epoch: state.epoch,
                    response_key_id: format!("{}#response-current", state.did),
                    request_kem_key_id: format!("{}#kem-current", state.did),
                    request_kem_recipient: kem.public().encode(),
                    expires_at_unix_ms: 4_070_908_800_000,
                },
            )
            .await
            .expect("ordinary announcement handler");
        assert!(matches!(response, DiscoveryResponseVariant::AnnounceResult));
        let resolver = service.production_resolver().expect("production resolver");
        let resolved = resolver
            .resolve_service(ServiceQuery::network("model").expect("query"))
            .await
            .expect("ordinary announcement must resolve");
        assert_eq!(resolved.evidence().accepted_state_digest, state.head_digest);
    }

    #[tokio::test]
    async fn accepted_state_advance_between_selection_and_dial_refuses() {
        let (resolver, source) = production_fixture(false);
        let resolved = resolver
            .resolve_service(ServiceQuery::network("model").expect("query"))
            .await
            .unwrap_or_else(|e| panic!("production candidate rejected: {e}"));
        {
            let mut guard = source.0.lock();
            let state = guard.as_mut().expect("fixture state");
            state.epoch += 1;
            state.head_digest = [0x55; 64];
        }
        let error = resolver
            .ensure_current(&resolved)
            .await
            .expect_err("advanced state accepted");
        assert!(error.to_string().contains("advanced or forked"));
    }

    struct CountingStream(Arc<std::sync::atomic::AtomicUsize>);

    #[async_trait]
    impl StreamHandle for CountingStream {
        async fn next_payload(
            &mut self,
        ) -> Result<Option<hyprstream_rpc::streaming::StreamPayload>> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(None)
        }
        async fn cancel(&mut self) -> Result<()> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
        fn stream_id(&self) -> &str {
            "counting"
        }
        fn is_completed(&self) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn live_stream_continuation_fails_closed_after_snapshot_advance() {
        let (resolver, source) = production_fixture(false);
        let resolver = Arc::new(resolver);
        let snapshot = resolver
            .resolve_service_candidates(ServiceQuery::network("model").expect("query"))
            .await
            .expect("snapshot")
            .remove(0);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut stream = CurrentStreamHandle {
            inner: Box::new(CountingStream(Arc::clone(&calls))),
            resolver,
            snapshot,
            invalidated: false,
        };
        {
            let mut guard = source.0.lock();
            let state = guard.as_mut().expect("fixture state");
            state.epoch += 1;
            state.head_digest = [0x77; 64];
        }
        assert!(stream.next_payload().await.is_err());
        assert!(stream.cancel().await.is_err());
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "invalid continuation dispatched an operation"
        );
    }

    #[tokio::test]
    async fn production_network_resolver_never_falls_back_to_local_reach() {
        let (resolver, _) = production_fixture(true);
        assert!(resolver
            .resolve_service(ServiceQuery::network("model").expect("query"))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn generated_client_uses_ordinary_identity_bound_resolver_path() {
        let (resolver, _) = production_fixture(false);
        let entries = resolver
            .announced_endpoints
            .write()
            .remove("model")
            .expect("fixture announcement");
        resolver
            .announced_endpoints
            .write()
            .insert("discovery".to_owned(), entries);
        let resolver = Arc::new(resolver);
        let _ = PRODUCTION_RESOLVER.set(resolver);
        let client_signing = SigningKey::from_bytes(&[0x44; 32]);
        let _client = crate::DiscoveryClient::from_resolver(client_signing, None)
            .unwrap_or_else(|e| panic!("generated resolver path failed: {e}"));
    }

    #[test]
    fn isolated_process_without_bootstrap_fails_closed() {
        const CHILD: &str = "HYPRSTREAM_TEST_RESOLVER_ABSENT_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let signing = SigningKey::from_bytes(&[0x46; 32]);
            let error = match production_rpc_client("model", signing, None) {
                Ok(_) => panic!("isolated process unexpectedly inherited another resolver"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("not installed"));
            return;
        }
        let status =
            std::process::Command::new(std::env::current_exe().expect("discovery test executable"))
                .arg("--exact")
                .arg("service::resolver_tests::isolated_process_without_bootstrap_fails_closed")
                .arg("--nocapture")
                .env(CHILD, "1")
                .status()
                .expect("isolated resolver subprocess");
        assert!(status.success(), "isolated resolver subprocess failed");
    }

    struct NoopBootstrapClient;

    #[async_trait::async_trait]
    impl RpcClient for NoopBootstrapClient {
        async fn call(&self, _: Vec<u8>) -> Result<Vec<u8>> {
            anyhow::bail!("bootstrap transport was unexpectedly dispatched")
        }
        async fn call_for_service(&self, _: &str, _: Vec<u8>) -> Result<Vec<u8>> {
            anyhow::bail!("bootstrap transport was unexpectedly dispatched")
        }
        async fn call_for_service_with_method(
            &self,
            _: &str,
            _: u16,
            _: Vec<u8>,
        ) -> Result<Vec<u8>> {
            anyhow::bail!("noop bootstrap client")
        }
        async fn call_with_options(&self, _: Vec<u8>, _: CallOptions) -> Result<Vec<u8>> {
            anyhow::bail!("bootstrap transport was unexpectedly dispatched")
        }
        async fn call_with_options_for_service(
            &self,
            _: &str,
            _: Vec<u8>,
            _: CallOptions,
        ) -> Result<Vec<u8>> {
            anyhow::bail!("bootstrap transport was unexpectedly dispatched")
        }
        async fn call_streaming(&self, _: Vec<u8>, _: [u8; 32]) -> Result<Vec<u8>> {
            anyhow::bail!("bootstrap transport was unexpectedly dispatched")
        }
        async fn call_streaming_for_service(
            &self,
            _: &str,
            _: Vec<u8>,
            _: [u8; 32],
        ) -> Result<Vec<u8>> {
            anyhow::bail!("bootstrap transport was unexpectedly dispatched")
        }
        async fn call_streaming_for_service_with_method(
            &self,
            _: &str,
            _: u16,
            _: Vec<u8>,
            _: [u8; 32],
        ) -> Result<Vec<u8>> {
            anyhow::bail!("noop bootstrap client")
        }
        async fn open_stream(&self, _: Vec<u8>) -> Result<Box<dyn StreamHandle>> {
            anyhow::bail!("bootstrap transport was unexpectedly dispatched")
        }
        async fn open_stream_from_info(
            &self,
            _: hyprstream_rpc::stream_info::StreamInfo,
            _: [u8; 32],
            _: [u8; 32],
        ) -> Result<Box<dyn StreamHandle>> {
            anyhow::bail!("bootstrap transport was unexpectedly dispatched")
        }
        fn next_id(&self) -> u64 {
            1
        }
    }

    fn authenticated_registry_identity(tag: u8) -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[tag; 32]).verifying_key()
    }

    #[test]
    fn redirected_credential_directories_have_zero_registry_authority() {
        const CHILD: &str = "HYPRSTREAM_TEST_FIXED_REGISTRY_CREDENTIAL_CHILD";
        const ATTACKER_KEY: &str = "HYPRSTREAM_TEST_ATTACKER_REGISTRY_KEY";
        if std::env::var_os(CHILD).is_some() {
            let key_bytes: [u8; 32] =
                hex::decode(std::env::var(ATTACKER_KEY).expect("attacker registry key"))
                    .expect("attacker key hex")
                    .try_into()
                    .expect("attacker key length");
            let attacker = VerifyingKey::from_bytes(&key_bytes).expect("attacker key");
            match load_trusted_registry_deployment_credentials() {
                Ok(credentials) => {
                    let witness = authenticate_registry_deployment_credentials(credentials)
                        .expect("fixed OS-owned credential pair must authenticate");
                    assert!(
                        !witness.verifier.matches(&attacker),
                        "ambient credential pair selected production authority"
                    );
                }
                Err(error) => assert!(
                    error.to_string().contains(DEPLOYMENT_CA_ROOT_PATH)
                        || error
                            .to_string()
                            .contains(REGISTRY_DEPLOYMENT_CREDENTIAL_PATH),
                    "startup did not fail at the fixed OS-owned seam: {error}"
                ),
            }
            return;
        }

        let alternate = tempfile::tempdir().expect("alternate credential directory");
        let ca = SigningKey::from_bytes(&[0x71; 32]);
        let registry = SigningKey::from_bytes(&[0x72; 32]);
        let jwt = exact_registry_credential(&ca, &registry);
        std::fs::write(
            alternate.path().join("ca-pubkey"),
            ca.verifying_key().as_bytes(),
        )
        .expect("alternate CA");
        std::fs::write(alternate.path().join("registry-service-jwt"), jwt)
            .expect("alternate registry JWT");
        let user_config = alternate.path().join("user-config");
        let user_credentials = user_config.join("hyprstream/credentials");
        std::fs::create_dir_all(&user_credentials).expect("user credential fallback");
        std::fs::copy(
            alternate.path().join("ca-pubkey"),
            user_credentials.join("ca-pubkey"),
        )
        .expect("user CA copy");
        std::fs::copy(
            alternate.path().join("registry-service-jwt"),
            user_credentials.join("registry-service-jwt"),
        )
        .expect("user JWT copy");
        let status = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .arg("--exact")
            .arg("service::resolver_tests::redirected_credential_directories_have_zero_registry_authority")
            .arg("--nocapture")
            .env(CHILD, "1")
            .env("CREDENTIALS_DIRECTORY", alternate.path())
            .env("XDG_CONFIG_HOME", &user_config)
            .env(ATTACKER_KEY, hex::encode(registry.verifying_key().as_bytes()))
            .status()
            .expect("redirected credential subprocess");
        assert!(status.success(), "redirected credential subprocess failed");
    }

    #[test]
    fn verified_registry_identity_ignores_policy_and_post_start_environment_mutation() {
        const CHILD: &str = "HYPRSTREAM_TEST_POST_START_REGISTRY_ENV_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(
                std::env::current_exe().expect("discovery test executable"),
            )
            .arg("--exact")
            .arg("service::resolver_tests::verified_registry_identity_ignores_policy_and_post_start_environment_mutation")
            .arg("--nocapture")
            .env(CHILD, "1")
            .status()
            .expect("post-start environment subprocess");
            assert!(status.success(), "post-start environment subprocess failed");
            return;
        }
        let caller = SigningKey::from_bytes(&[0x61; 32]);
        hyprstream_service::global_trust_store().insert(
            caller.verifying_key(),
            hyprstream_service::Attestation {
                scopes: std::iter::once("registry".to_owned()).collect(),
                subject: None,
                jwt: None,
                expires_at: 0,
                attested_by: None,
            },
        );
        let ca = SigningKey::from_bytes(&[0x62; 32]);
        let registry = SigningKey::from_bytes(&[0x63; 32]);
        let witness =
            authenticate_registry_deployment_credentials(TrustedRegistryDeploymentCredentials {
                ca_verifying_key: ca.verifying_key(),
                registry_credential: exact_registry_credential(&ca, &registry),
            })
            .expect("trusted pair");
        std::env::set_var("CREDENTIALS_DIRECTORY", "post-start-attacker-directory");
        std::env::set_var("XDG_CONFIG_HOME", "post-start-attacker-config");
        assert!(witness.verifier.matches(&registry.verifying_key()));
        assert!(!witness.verifier.matches(&caller.verifying_key()));
    }

    #[test]
    fn process_bootstrap_precedes_first_generated_client() {
        const CHILD: &str = "HYPRSTREAM_TEST_RESOLVER_BOOTSTRAP_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let store = hyprstream_service::deployment_data_dir()
                .expect("deployment-owned data root")
                .join("pds-store");
            std::fs::create_dir_all(&store).expect("checkpoint store directory");
            drop(rocksdb::DB::open_default(&store).expect("empty checkpoint store"));
            let signing = SigningKey::from_bytes(&[0x47; 32]);
            hyprstream_service::global_trust_store().insert(
                signing.verifying_key(),
                hyprstream_service::Attestation {
                    scopes: std::iter::once("registry".to_owned()).collect(),
                    subject: None,
                    jwt: None,
                    expires_at: 0,
                    attested_by: None,
                },
            );
            let authority =
                authenticate_discovery_bootstrap_identity(authenticated_registry_identity(0x51))
                    .expect("authenticate process resolver");
            DiscoveryService::bootstrap_authenticated_process(
                authority,
                crate::DiscoveryClient::new(Arc::new(NoopBootstrapClient)),
            )
            .expect("install process resolver");
            assert!(production_rpc_client("model", signing, None).is_ok());
            assert!(
                authenticate_discovery_bootstrap_identity(authenticated_registry_identity(0x52))
                    .is_err()
            );
            return;
        }
        let deployment = tempfile::tempdir().expect("deployment data root");
        let status =
            std::process::Command::new(std::env::current_exe().expect("discovery test executable"))
                .arg("--exact")
                .arg("service::resolver_tests::process_bootstrap_precedes_first_generated_client")
                .arg("--nocapture")
                .env(CHILD, "1")
                .env("XDG_DATA_HOME", deployment.path())
                .status()
                .expect("resolver bootstrap subprocess");
        assert!(status.success(), "resolver bootstrap subprocess failed");
    }

    fn seed_registry_for_bootstrap(tag: u8) {
        let registry = SigningKey::from_bytes(&[tag; 32]);
        hyprstream_service::global_trust_store().insert(
            registry.verifying_key(),
            hyprstream_service::Attestation {
                scopes: std::iter::once("registry".to_owned()).collect(),
                subject: None,
                jwt: None,
                expires_at: 0,
                attested_by: None,
            },
        );
    }

    #[test]
    fn authenticated_process_bootstrap_has_exactly_one_concurrent_consumer() {
        const CHILD: &str = "HYPRSTREAM_TEST_RESOLVER_CONCURRENT_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let store = hyprstream_service::deployment_data_dir()
                .unwrap()
                .join("pds-store");
            std::fs::create_dir_all(&store).unwrap();
            drop(rocksdb::DB::open_default(&store).unwrap());
            seed_registry_for_bootstrap(0x48);
            let barrier = Arc::new(std::sync::Barrier::new(9));
            let attempts: Vec<_> = (0..8)
                .map(|_| {
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        authenticate_discovery_bootstrap_identity(authenticated_registry_identity(
                            0x53,
                        ))
                        .and_then(|authority| {
                            DiscoveryService::bootstrap_authenticated_process(
                                authority,
                                crate::DiscoveryClient::new(Arc::new(NoopBootstrapClient)),
                            )
                        })
                        .is_ok()
                    })
                })
                .collect();
            barrier.wait();
            assert_eq!(
                attempts
                    .into_iter()
                    .map(|attempt| attempt.join().unwrap())
                    .filter(|installed| *installed)
                    .count(),
                1
            );
            return;
        }
        let deployment = tempfile::tempdir().unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("service::resolver_tests::authenticated_process_bootstrap_has_exactly_one_concurrent_consumer")
            .arg("--nocapture")
            .env(CHILD, "1")
            .env("XDG_DATA_HOME", deployment.path())
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn authenticated_process_bootstrap_failure_is_terminal() {
        const CHILD: &str = "HYPRSTREAM_TEST_RESOLVER_FAILURE_CHILD";
        if std::env::var_os(CHILD).is_some() {
            seed_registry_for_bootstrap(0x49);
            let authority =
                authenticate_discovery_bootstrap_identity(authenticated_registry_identity(0x54))
                    .unwrap();
            let first = DiscoveryService::bootstrap_authenticated_process(
                authority,
                crate::DiscoveryClient::new(Arc::new(NoopBootstrapClient)),
            )
            .expect_err("missing checkpoint store unexpectedly installed");
            assert!(first.to_string().contains("failed to open"));
            let store = hyprstream_service::deployment_data_dir()
                .unwrap()
                .join("pds-store");
            std::fs::create_dir_all(&store).unwrap();
            drop(rocksdb::DB::open_default(&store).unwrap());
            let second =
                authenticate_discovery_bootstrap_identity(authenticated_registry_identity(0x55))
                    .err()
                    .expect("failed bootstrap authority was replayed");
            assert!(second.to_string().contains("already sealed or consumed"));
            return;
        }
        let deployment = tempfile::tempdir().unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("service::resolver_tests::authenticated_process_bootstrap_failure_is_terminal")
            .arg("--nocapture")
            .env(CHILD, "1")
            .env("XDG_DATA_HOME", deployment.path())
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[tokio::test]
    async fn stale_or_expired_production_evidence_is_rejected() {
        let (resolver, _) = production_fixture(false);
        resolver
            .announced_endpoints
            .write()
            .get_mut("model")
            .and_then(|entries| entries.first_mut())
            .expect("fixture service")
            .last_heartbeat = Instant::now() - ANNOUNCED_ENDPOINT_TTL - Duration::from_secs(1);
        assert!(resolver
            .resolve_service(ServiceQuery::network("model").expect("query"))
            .await
            .is_err());

        let (resolver, source) = production_fixture(false);
        source.0.lock().as_mut().expect("fixture state").expires_at =
            Some("2000-01-01T00:00:00Z".to_owned());
        assert!(resolver
            .resolve_service(ServiceQuery::network("model").expect("query"))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn malformed_candidate_does_not_poison_valid_alternative() {
        let (resolver, _) = production_fixture(false);
        {
            let mut endpoints = resolver.announced_endpoints.write();
            let entries = endpoints.get_mut("model").expect("fixture service");
            let mut malformed = entries.first().expect("fixture endpoint").clone();
            malformed.request_kem_recipient = vec![0xff];
            malformed.socket_kind = "quic".to_owned();
            malformed.endpoint = "quic://missing-port".to_owned();
            entries.insert(0, malformed);
        }

        let resolved = resolver
            .resolve_service(ServiceQuery::network("model").expect("query"))
            .await
            .unwrap_or_else(|e| panic!("malformed alternative poisoned valid candidate: {e}"));
        assert_eq!(resolved.service_name(), "model");
    }

    #[tokio::test]
    async fn rejected_or_forked_current_state_never_produces_candidate() {
        let (resolver, source) = production_fixture(false);
        *source.0.lock() = None;
        assert!(resolver
            .resolve_service(ServiceQuery::network("model").expect("query"))
            .await
            .is_err());

        let (resolver, _) = production_fixture(false);
        resolver
            .announced_endpoints
            .write()
            .get_mut("model")
            .and_then(|entries| entries.first_mut())
            .expect("fixture service")
            .accepted_state_digest = vec![0x77; 64];
        assert!(resolver
            .resolve_service(ServiceQuery::network("model").expect("query"))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn resolver_uses_fresh_announced_quic_endpoint() {
        let svc = service();
        svc.announced_endpoints.write().insert(
            "model".to_owned(),
            vec![legacy_endpoint(
                "quic",
                "quic://model.hyprstream.svc.cluster.local:10.96.0.42:4433",
                Instant::now(),
            )],
        );

        let transport = match svc.resolve("model", SocketKind::Quic).await {
            Ok(transport) => transport,
            Err(err) => panic!("fresh announced QUIC endpoint must resolve: {err}"),
        };
        assert_eq!(transport.bind_mode(), BindMode::Connect);
        match transport.endpoint {
            EndpointType::Quic {
                addr, server_name, ..
            } => {
                assert_eq!(server_name, "model.hyprstream.svc.cluster.local");
                assert_eq!(addr, SocketAddr::from(([10, 96, 0, 42], 4433)));
            }
            other => panic!("expected announced QUIC endpoint, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn resolver_rejects_missing_announced_quic_endpoint() {
        let err = match service().resolve("model", SocketKind::Quic).await {
            Ok(transport) => panic!("missing announced QUIC endpoint resolved to {transport:?}"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("no fresh announced QUIC endpoint"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn resolver_rejects_stale_announced_quic_endpoint() {
        let svc = service();
        svc.announced_endpoints.write().insert(
            "model".to_owned(),
            vec![legacy_endpoint(
                "quic",
                "quic://model.hyprstream.svc.cluster.local:10.96.0.42:4433",
                Instant::now() - (ANNOUNCED_ENDPOINT_TTL + Duration::from_secs(1)),
            )],
        );

        let err = match svc.resolve("model", SocketKind::Quic).await {
            Ok(transport) => panic!("stale announced QUIC endpoint resolved to {transport:?}"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("no fresh announced QUIC endpoint"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn announced_iroh_endpoint_parses_to_transport() {
        let node_hex = "0707070707070707070707070707070707070707070707070707070707070707";
        let transport = match parse_announced_iroh(&format!("iroh://{node_hex}")) {
            Ok(transport) => transport,
            Err(err) => panic!("valid iroh endpoint must parse: {err}"),
        };
        assert_eq!(transport.bind_mode(), BindMode::Connect);
        match transport.endpoint {
            EndpointType::Iroh {
                node_id,
                direct_addrs,
                relay_url,
            } => {
                assert_eq!(node_id, [7u8; 32]);
                assert!(direct_addrs.is_empty());
                assert!(relay_url.is_none());
            }
            other => panic!("expected iroh endpoint, got {other:?}"),
        }
    }
}

/// Convert a `SocketKind` to a lowercase string for serialization.
fn socket_kind_to_string(kind: SocketKind) -> &'static str {
    match kind {
        SocketKind::Req => "req",
        SocketKind::Rep => "rep",
        SocketKind::Quic => "quic",
    }
}

// ============================================================================
// DiscoveryHandler implementation (generated trait)
// ============================================================================

#[async_trait(?Send)]
impl DiscoveryHandler for DiscoveryService {
    async fn authorize(
        &self,
        ctx: &EnvelopeContext,
        resource: &str,
        operation: &str,
    ) -> Result<()> {
        // Delegate to authorization provider if available
        if let Some(ref auth) = self.auth_provider {
            let subject = ctx.subject().to_string();
            let allowed = auth
                .check(&subject, "*", resource, operation)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!("Discovery auth check failed for {}: {}", subject, e);
                    false
                });
            if allowed {
                Ok(())
            } else {
                anyhow::bail!(
                    "Unauthorized: {} cannot {} on {}",
                    subject,
                    operation,
                    resource
                )
            }
        } else {
            // No auth provider — allow (backward compat for local-only deployments)
            Ok(())
        }
    }

    async fn handle_list_services(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        trace!("Discovery: listing services");

        // Process-local registry
        let reg = self.reg()?;
        let service_names = reg.list_services();
        let mut summaries: Vec<ServiceSummary> = service_names
            .iter()
            .filter_map(|name| {
                reg.service_entry(name).map(|entry| {
                    let socket_kinds: Vec<String> = entry
                        .endpoints
                        .keys()
                        .map(|k| socket_kind_to_string(*k).to_owned())
                        .collect();
                    ServiceSummary {
                        name: entry.name,
                        description: entry.description.unwrap_or_default(),
                        socket_kinds,
                        has_schema: entry.schema.is_some(),
                    }
                })
            })
            .collect();
        drop(reg);

        // Merge announced endpoints from other processes
        let announced = self.announced_endpoints.read();
        let local_names: Vec<String> = summaries.iter().map(|s| s.name.clone()).collect();
        for (name, endpoints) in announced.iter() {
            if local_names.iter().any(|n| n == name) {
                // Service exists locally — add announced socket kinds
                if let Some(summary) = summaries.iter_mut().find(|s| s.name == *name) {
                    for ep in endpoints {
                        if !summary.socket_kinds.contains(&ep.socket_kind) {
                            summary.socket_kinds.push(ep.socket_kind.clone());
                        }
                    }
                }
            } else {
                // Service only known from announcements
                summaries.push(ServiceSummary {
                    name: name.clone(),
                    description: String::new(),
                    socket_kinds: endpoints.iter().map(|e| e.socket_kind.clone()).collect(),
                    has_schema: false,
                });
            }
        }

        Ok(DiscoveryResponseVariant::ListServicesResult(ServiceList {
            services: summaries,
        }))
    }

    async fn handle_get_endpoints(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let service_name = data;
        trace!("Discovery: getting endpoints for '{}'", service_name);

        // Process-local registry
        let reg = self.reg()?;
        let endpoints_map = reg.service_endpoints(service_name);
        drop(reg);

        // Build from local registry (no self-proof — discovery's own JWT not stored locally)
        let mut endpoints: Vec<EndpointInfo> = match endpoints_map {
            Some(map) => map
                .iter()
                .map(|(kind, transport)| EndpointInfo {
                    socket_kind: socket_kind_to_string(*kind).to_owned(),
                    endpoint: transport.endpoint_string(),
                    service_jwt: String::new(),
                    tls_endorsement: self.tls_endorsement.clone(),
                    tls_domain: self.tls_domain.clone(),
                    service_did: Did::default(),
                    capabilities: Vec::new(),
                    accepted_state_digest: Vec::new(),
                    accepted_state_epoch: 0,
                    response_key_id: String::new(),
                    request_kem_key_id: String::new(),
                    request_kem_recipient: Vec::new(),
                    expires_at_unix_ms: 0,
                    source_signer: Vec::new(),
                })
                .collect(),
            None => Vec::new(),
        };

        // Merge announced endpoints from other processes (carry service JWT)
        let announced = self.announced_endpoints.read();
        if let Some(announced_eps) = announced.get(service_name) {
            for ep in announced_eps {
                // Don't duplicate if already present from local registry
                if !endpoints.iter().any(|e| e.socket_kind == ep.socket_kind) {
                    endpoints.push(EndpointInfo {
                        socket_kind: ep.socket_kind.clone(),
                        endpoint: ep.endpoint.clone(),
                        service_jwt: ep.service_jwt.clone(),
                        tls_endorsement: self.tls_endorsement.clone(),
                        tls_domain: self.tls_domain.clone(),
                        service_did: ep.service_did.clone(),
                        capabilities: ep.capabilities.iter().cloned().collect(),
                        accepted_state_digest: ep.accepted_state_digest.clone(),
                        accepted_state_epoch: ep.accepted_state_epoch,
                        response_key_id: ep.response_key_id.clone(),
                        request_kem_key_id: ep.request_kem_key_id.clone(),
                        request_kem_recipient: ep.request_kem_recipient.clone(),
                        expires_at_unix_ms: ep.expires_at_unix_ms,
                        source_signer: ep.source_signer.to_vec(),
                    });
                }
            }
        }

        if endpoints.is_empty() {
            Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("Service '{}' not found", service_name),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            }))
        } else {
            Ok(DiscoveryResponseVariant::GetEndpointsResult(
                ServiceEndpoints { endpoints },
            ))
        }
    }

    async fn handle_get_schema(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let service_name = data;
        trace!("Discovery: getting schema for '{}'", service_name);

        let reg = self.reg()?;
        let schema = reg.service_schema(service_name);
        drop(reg);

        match schema {
            Some(bytes) => Ok(DiscoveryResponseVariant::GetSchemaResult(bytes.to_vec())),
            None => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("No schema registered for service '{}'", service_name),
                code: "NO_SCHEMA".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_ping(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        trace!("Discovery: ping");

        let reg = self.reg()?;
        let service_count = reg.list_services().len() as u32;
        drop(reg);

        let uptime = self.started_at.elapsed().as_secs();

        Ok(DiscoveryResponseVariant::PingResult(PingInfo {
            status: "ok".to_owned(),
            service_count,
            uptime,
        }))
    }

    async fn handle_get_auth_metadata(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let filter = data;
        trace!("Discovery: get auth metadata (filter='{}')", filter);

        let issuer = match &self.oauth_issuer_url {
            Some(url) => url.clone(),
            None => {
                return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                    message: "OAuth issuer URL not configured".to_owned(),
                    code: "NOT_CONFIGURED".to_owned(),
                    details: String::new(),
                }));
            }
        };

        let reg = self.reg()?;
        let service_names = reg.list_services();

        let services: Vec<AuthMetadata> = service_names
            .iter()
            .filter(|name| filter.is_empty() || *name == filter)
            .filter_map(|name| {
                let quic_transport = match reg.try_endpoint(name, SocketKind::Quic) {
                    Ok(transport) => transport,
                    Err(err) => {
                        tracing::debug!(
                            service = %name,
                            error = %err,
                            "skipping auth metadata for service without registered QUIC reach"
                        );
                        return None;
                    }
                };
                let resource = quic_transport.quic_resource_url(name)?;
                Some(AuthMetadata {
                    service_name: name.clone(),
                    resource,
                    authorization_servers: vec![issuer.clone()],
                    scopes_supported: Vec::new(),
                    resource_name: format!("HyprStream {} Service", name),
                })
            })
            .collect();
        drop(reg);

        Ok(DiscoveryResponseVariant::GetAuthMetadataResult(
            AuthMetadataList { services },
        ))
    }

    async fn handle_prepare_stream(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        Ok(DiscoveryResponseVariant::Error(ErrorInfo {
            message: "prepareStream removed — use StreamChannel::prepare_identified_stream for authenticated streaming".to_owned(),
            code: "REMOVED".to_owned(),
            details: String::new(),
        }))
    }

    async fn handle_get_stream(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        Ok(DiscoveryResponseVariant::Error(ErrorInfo {
            message:
                "getStream removed — use StreamChannel::prepare_identified_stream for authenticated streaming"
                    .to_owned(),
            code: "REMOVED".to_owned(),
            details: String::new(),
        }))
    }

    async fn handle_list_streams(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        Ok(DiscoveryResponseVariant::Error(ErrorInfo {
            message: "listStreams removed — use StreamChannel::prepare_identified_stream for authenticated streaming".to_owned(),
            code: "REMOVED".to_owned(),
            details: String::new(),
        }))
    }

    async fn handle_announce(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ServiceAnnouncement,
    ) -> Result<DiscoveryResponseVariant> {
        info!(
            "Discovery: service '{}' announced {} endpoint: {} (from {})",
            data.service_name,
            data.socket_kind,
            data.endpoint,
            ctx.subject()
        );

        let svc_name = data.service_name.clone();
        let sock_kind = data.socket_kind.clone();
        let endpoint = data.endpoint.clone();
        let service_jwt = data.service_jwt.clone().unwrap_or_default();
        let identity_bound = !data.service_did.as_str().is_empty()
            || !data.accepted_state_digest.is_empty()
            || !data.request_kem_recipient.is_empty();
        if identity_bound {
            anyhow::ensure!(
                !service_jwt.is_empty(),
                "identity-bound announcement requires a verified service JWT"
            );
            anyhow::ensure!(
                data.service_did.is_did_at9p()
                    && data.accepted_state_digest.len() == 64
                    && !data.capabilities.is_empty()
                    && data
                        .response_key_id
                        .starts_with(&format!("{}#", data.service_did))
                    && data
                        .request_kem_key_id
                        .starts_with(&format!("{}#", data.service_did))
                    && data.expires_at_unix_ms > unix_millis_now(),
                "identity-bound announcement metadata is incomplete or expired"
            );
            let recipient = hyprstream_rpc::crypto::hybrid_kem::RecipientPublic::decode(
                &data.request_kem_recipient,
            )?;
            anyhow::ensure!(
                recipient.suite_id
                    == hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768
                    && recipient.eks.len() == recipient.suite_id.components().len(),
                "identity-bound announcement requires suite-complete hybrid KEM material"
            );
            match data.socket_kind.as_str() {
                "quic" => {
                    parse_announced_quic(&data.endpoint)?;
                }
                "iroh" => {
                    parse_announced_iroh(&data.endpoint)?;
                }
                _ => {
                    anyhow::bail!("identity-bound network announcement requires QUIC or Iroh reach")
                }
            }
        }

        // R3: Verify service JWT signature + subject matches serviceName.
        // Full JWT verification (not decode_unverified) to prevent forged identities.
        if !service_jwt.is_empty() {
            let verified = hyprstream_rpc::auth::jwt::decode_with_key(
                &service_jwt,
                &self.jwt_verifying_key,
                self.expected_audience.as_deref(),
            )
            .map_err(|e| {
                tracing::warn!("Service JWT verification failed in announce: {}", e);
                anyhow::anyhow!("Invalid service JWT in announce: {}", e)
            })?;
            // Check that sub matches "service:{serviceName}"
            let expected_sub = format!("service:{}", svc_name);
            if verified.sub != expected_sub {
                anyhow::bail!(
                    "Service JWT subject mismatch: expected '{}', got '{}'",
                    expected_sub,
                    verified.sub
                );
            }
            anyhow::ensure!(
                verified.cnf_key_bytes() == Some(ctx.cnf),
                "service JWT confirmation key does not match verified announcement signer"
            );
        }

        let replacement = AnnouncedEndpoint {
            socket_kind: sock_kind.clone(),
            endpoint: endpoint.clone(),
            service_jwt: service_jwt.clone(),
            service_did: data.service_did.clone(),
            capabilities: data.capabilities.iter().cloned().collect(),
            accepted_state_digest: data.accepted_state_digest.clone(),
            accepted_state_epoch: data.accepted_state_epoch,
            response_key_id: data.response_key_id.clone(),
            request_kem_key_id: data.request_kem_key_id.clone(),
            request_kem_recipient: data.request_kem_recipient.clone(),
            expires_at_unix_ms: data.expires_at_unix_ms,
            source_signer: ctx.cnf,
            last_heartbeat: Instant::now(),
        };

        let mut endpoints = self.announced_endpoints.write();
        let entry = endpoints.entry(svc_name).or_default();
        // Replace existing endpoint for the same socket kind, or add new
        if let Some(existing) = entry.iter_mut().find(|e| e.socket_kind == sock_kind) {
            *existing = replacement;
        } else {
            entry.push(replacement);
        }

        Ok(DiscoveryResponseVariant::AnnounceResult)
    }

    // ────────────────────────────────────────────────────────────────────
    // Phase 0.5 Stage D — federation directory
    // ────────────────────────────────────────────────────────────────────

    async fn handle_register_entity_statement(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterEntityStatementRequest,
    ) -> Result<DiscoveryResponseVariant> {
        if data.issuer.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "issuer is required".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        if data.jwt.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "jwt is required".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // NOTE: we intentionally do NOT gate cache writes by the
        // `federation:register` policy here. The load-bearing trust
        // check happens at use time in FederationKeyResolver::get_key,
        // which queries policy before accepting any cached entity
        // statement for verification. Gating writes too would force
        // discovery to call PolicyService on every register, which is
        // a perf hit on a hot path that already requires a passing
        // `authorize()` check on the caller side. Cache may contain
        // statements for issuers no operator currently trusts —
        // they're inert until the resolver's policy check admits them.
        let cached = CachedEntityStatement {
            jwt: data.jwt.clone(),
            fetched_at: unix_seconds_now(),
        };
        let mut map = self.entity_statements.write();
        map.insert(data.issuer.clone(), cached);
        let total = map.len();
        drop(map);

        info!(
            issuer = %data.issuer,
            caller = %ctx.subject(),
            total_cached = total,
            "Discovery: entity statement registered"
        );
        Ok(DiscoveryResponseVariant::RegisterEntityStatementResult)
    }

    async fn handle_get_entity_statement(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let issuer = data;
        let map = self.entity_statements.read();
        match map.get(issuer) {
            Some(cached) => {
                trace!(issuer = %issuer, "Discovery: entity statement cache hit");
                Ok(DiscoveryResponseVariant::GetEntityStatementResult(
                    EntityStatement {
                        issuer: issuer.to_owned(),
                        jwt: cached.jwt.clone(),
                        fetched_at: cached.fetched_at,
                    },
                ))
            }
            None => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("no entity statement cached for issuer: {}", issuer),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_register_envelope_keyset(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterEnvelopeKeysetRequest,
    ) -> Result<DiscoveryResponseVariant> {
        if data.service_did.as_str().is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "serviceDid is required".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        if data.cose_keyset_cbor.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "coseKeysetCbor is required".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        let cached = CachedEnvelopeKeyset {
            cose_keyset_cbor: data.cose_keyset_cbor.clone(),
            fetched_at: unix_seconds_now(),
        };
        let mut map = self.envelope_keysets.write();
        map.insert(data.service_did.as_str().to_owned(), cached);
        let total = map.len();
        drop(map);

        info!(
            service_did = %data.service_did,
            caller = %ctx.subject(),
            total_cached = total,
            "Discovery: envelope keyset registered"
        );
        Ok(DiscoveryResponseVariant::RegisterEnvelopeKeysetResult)
    }

    async fn handle_get_envelope_keyset(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let service_did = data;
        let map = self.envelope_keysets.read();
        match map.get(service_did) {
            Some(cached) => {
                trace!(service_did = %service_did, "Discovery: envelope keyset cache hit");
                Ok(DiscoveryResponseVariant::GetEnvelopeKeysetResult(
                    EnvelopeKeyset {
                        service_did: hyprstream_rpc::identity::Did::new(service_did.to_owned()),
                        cose_keyset_cbor: cached.cose_keyset_cbor.clone(),
                        fetched_at: cached.fetched_at,
                    },
                ))
            }
            None => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("no envelope keyset cached for service: {}", service_did),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_list_known_issuers(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        // Federation topology leak — require authentication. Per
        // Phase 0.5 plan Q10: getEntityStatement/getEnvelopeKeyset are
        // anonymous-readable (public artifacts), but listKnownIssuers
        // enumerates which partners we trust, which is operator-sensitive.
        if let Err(e) = self
            .authorize(ctx, "discovery:federation", "list-known-issuers")
            .await
        {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("unauthorized: {}", e),
                code: "UNAUTHORIZED".to_owned(),
                details: String::new(),
            }));
        }
        let map = self.entity_statements.read();
        let issuers: Vec<String> = map.keys().cloned().collect();
        Ok(DiscoveryResponseVariant::ListKnownIssuersResult(
            IssuerList { issuers },
        ))
    }

    // ────────────────────────────────────────────────────────────────────
    // #431 — federated record lookup as a verifiable CAR proof
    // ────────────────────────────────────────────────────────────────────

    async fn handle_get_record(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetRecordRequest,
    ) -> Result<DiscoveryResponseVariant> {
        // Resolve the three components from either the at:// URI or the structured
        // fields. When `uri` is set it wins (the fields are ignored, per schema).
        let (did, collection, rkey) = if !data.uri.is_empty() {
            match parse_at_uri(&data.uri) {
                Ok(parts) => parts,
                Err(e) => {
                    return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                        message: format!("invalid at-uri: {e}"),
                        code: "INVALID_ARGUMENT".to_owned(),
                        details: String::new(),
                    }));
                }
            }
        } else if !data.did.is_empty() && !data.collection.is_empty() && !data.rkey.is_empty() {
            (data.did.clone(), data.collection.clone(), data.rkey.clone())
        } else {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "getRecord requires either `uri` or all of (did, collection, rkey)"
                    .to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        };

        // ── Access control (CONFIDENTIALITY — the load-bearing check) ──
        // The schema's auto-generated discovery:query gate already ran in dispatch,
        // but that only authorizes "may call getRecord at all". An atproto record's
        // CID is content-derived and predictable, so presenting a valid at:// / CID
        // must NOT by itself grant a read. Independently require the caller to be
        // permitted to read THIS target DID's collection. Integrity (the signed CAR
        // proof, verified offline) does NOT substitute for this check.
        let resource = format!("discovery:record:{did}/{collection}");
        if let Err(e) = self.authorize(ctx, &resource, "query").await {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("unauthorized to read {resource}: {e}"),
                code: "UNAUTHORIZED".to_owned(),
                details: String::new(),
            }));
        }

        let resolver = match &self.record_resolver {
            Some(r) => r,
            None => {
                return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                    message: "no record resolver configured on this node".to_owned(),
                    code: "NOT_FOUND".to_owned(),
                    details: String::new(),
                }));
            }
        };

        match resolver.resolve_record(&did, &collection, &rkey).await {
            Ok(Some(rec)) => Ok(DiscoveryResponseVariant::GetRecordResult(RecordCar {
                uri: rec.uri,
                car: rec.car,
            })),
            Ok(None) => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("no record at at://{did}/{collection}/{rkey}"),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
            Err(e) => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("record resolution failed: {e}"),
                code: "INTERNAL".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_get_repo(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        value: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let did = value;
        if did.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "getRepo requires a non-empty DID".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Access control: a full repo CAR exposes every record in the DID's
        // collections, so gate on the DID itself (broadest read of that repo).
        let resource = format!("discovery:record:{did}");
        if let Err(e) = self.authorize(ctx, &resource, "query").await {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("unauthorized to read repo {did}: {e}"),
                code: "UNAUTHORIZED".to_owned(),
                details: String::new(),
            }));
        }

        let resolver = match &self.record_resolver {
            Some(r) => r,
            None => {
                return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                    message: "no record resolver configured on this node".to_owned(),
                    code: "NOT_FOUND".to_owned(),
                    details: String::new(),
                }));
            }
        };

        match resolver.resolve_repo(did).await {
            Ok(Some(rec)) => Ok(DiscoveryResponseVariant::GetRepoResult(RecordCar {
                uri: rec.uri,
                car: rec.car,
            })),
            Ok(None) => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("no repo stored for DID {did}"),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
            Err(e) => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("repo resolution failed: {e}"),
                code: "INTERNAL".to_owned(),
                details: String::new(),
            })),
        }
    }

    /// #523 P0 / #524 — placement candidate query. The auto-generated dispatch
    /// gate already ran a `discovery:QueryCandidates`/`query` authz check
    /// before this handler was invoked; that only authorizes "may call
    /// queryCandidates at all". Each surviving candidate additionally gets its
    /// own fail-closed `placement:candidate:<did>` check below (a denied node
    /// is silently omitted, not surfaced as an error) — mirroring why
    /// `getRecord`/`getRepo` layer a per-target check on top of the generic gate.
    ///
    /// Pipeline: known node DIDs (placement directory) → hard-exclude any
    /// without a live `reportNodeLiveness` entry (decision: absence/expiry is
    /// exclusion, never a "stale" flag) → sync filter (label selectors AND,
    /// resource requests AND) via `scheduling::filter` → per-candidate async
    /// authz → rank (ascending load, DID tiebreak) via `scheduling::rank` →
    /// bound to `maxCandidates` (or [`DEFAULT_MAX_CANDIDATES`] when 0).
    /// `totalMatching` is the authorized-survivor count *before* the bound.
    async fn handle_query_candidates(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &QueryCandidatesRequest,
    ) -> Result<DiscoveryResponseVariant> {
        struct Candidate {
            did: String,
            record_uri: String,
            load_fraction: f32,
            allocatable: Vec<(String, String)>,
            last_seen: i64,
            labels: Vec<(String, String)>,
        }

        let selectors: Vec<scheduling::LabelSelector> = data
            .selectors
            .iter()
            .map(|s| {
                scheduling::LabelSelector::new(
                    s.key.clone(),
                    to_scheduling_op(s.op),
                    s.values.clone(),
                )
            })
            .collect();
        let resources: Vec<scheduling::ResourceRequest> = data
            .resources
            .iter()
            .map(|r| scheduling::ResourceRequest::new(r.name.clone(), r.min_quantity.clone()))
            .collect();

        // Hard liveness exclusion (decision #1): only nodes with a live,
        // unexpired `reportNodeLiveness` entry become candidates at all.
        let candidates: Vec<Candidate> = self
            .placement_index
            .known_node_dids()
            .into_iter()
            .filter_map(|did| {
                let live = self.liveness.get(&Did::new(did.clone()))?;
                let labels = self.placement_index.effective_labels(&did);
                let record_uri = self.placement_index.record_uri(&did).unwrap_or_default();
                Some(Candidate {
                    did,
                    record_uri,
                    load_fraction: live.load_fraction,
                    allocatable: live.allocatable,
                    last_seen: live.last_seen,
                    labels,
                })
            })
            .collect();

        let predicates: Vec<scheduling::Predicate<Candidate>> = vec![
            Box::new({
                let selectors = selectors.clone();
                move |c: &Candidate| {
                    for sel in &selectors {
                        if !sel.matches(&c.labels) {
                            return Some(scheduling::RejectionReason(format!(
                                "label selector on {:?} did not match",
                                sel.key
                            )));
                        }
                    }
                    None
                }
            }),
            Box::new({
                let resources = resources.clone();
                move |c: &Candidate| {
                    for req in &resources {
                        let satisfied = c
                            .allocatable
                            .iter()
                            .find(|(name, _)| name == &req.name)
                            .is_some_and(|(_, quantity)| req.satisfied_by(quantity));
                        if !satisfied {
                            return Some(scheduling::RejectionReason(format!(
                                "resource {:?} not satisfied",
                                req.name
                            )));
                        }
                    }
                    None
                }
            }),
        ];

        let outcomes = scheduling::filter(&candidates, &predicates);
        let survivors: Vec<&Candidate> = outcomes
            .iter()
            .filter(|o| o.passed())
            .map(|o| o.candidate)
            .collect();

        // Per-candidate fail-closed authz — async, so it runs as its own pass
        // rather than inside a (sync) `scheduling::Predicate` closure. A denied
        // node is silently dropped, never surfaced as an error.
        let mut authorized: Vec<&Candidate> = Vec::with_capacity(survivors.len());
        for c in survivors {
            let resource = format!("placement:candidate:{}", c.did);
            if self.authorize(ctx, &resource, "query").await.is_ok() {
                authorized.push(c);
            }
        }

        // Post-filter, post-authz, pre-bound — so callers can tell truncation
        // apart from "that's really all of them".
        let total_matching = authorized.len() as u32;

        let ranked = scheduling::rank(authorized, |a, b| {
            a.load_fraction
                .partial_cmp(&b.load_fraction)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.did.cmp(&b.did))
        });

        let max = if data.max_candidates == 0 {
            DEFAULT_MAX_CANDIDATES
        } else {
            data.max_candidates as usize
        };
        let candidates_out: Vec<PlacementCandidate> = ranked
            .into_iter()
            .take(max)
            .map(|c| PlacementCandidate {
                node: c.did.clone(),
                record_uri: c.record_uri.clone(),
                load_fraction: c.load_fraction,
                allocatable: c
                    .allocatable
                    .iter()
                    .map(|(name, quantity)| Resource {
                        name: name.clone(),
                        quantity: quantity.clone(),
                    })
                    .collect(),
                last_seen: c.last_seen,
            })
            .collect();

        Ok(DiscoveryResponseVariant::QueryCandidatesResult(
            PlacementCandidateSet {
                candidates: candidates_out,
                total_matching,
            },
        ))
    }

    /// #524 P1 — node liveness heartbeat. The auto-generated dispatch gate
    /// already ran a `discovery:ReportNodeLiveness`/`write` authz check before
    /// this handler was invoked. Before accepting a DID-specific heartbeat,
    /// additionally require write admission for that candidate. This prevents
    /// an authorized caller from using arbitrary heartbeat DIDs to grow the
    /// durable placement directory or trigger unbounded resolver work.
    ///
    /// Re-inserting (heartbeating) an admitted node refreshes its TTL entry. The
    /// first heartbeat seen for a DID this directory hasn't ingested a
    /// `NodeRecord` for yet may lazily trigger a repo poll. The bounded retry
    /// gate records an attempt before resolver access, so absent/invalid/non-
    /// node repos are retried only after [`PLACEMENT_INGEST_RETRY_TTL`]; a
    /// failure there does not fail the admitted heartbeat itself.
    async fn handle_report_node_liveness(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &NodeLiveness,
    ) -> Result<DiscoveryResponseVariant> {
        let node_did = data.node.as_str().to_owned();
        if node_did.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "node is required".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // The generic dispatch gate authorizes calling reportNodeLiveness; it
        // does not authorize claiming liveness for a particular DID. Gate the
        // DID before touching either the bounded volatile cache or the
        // unbounded-by-heartbeat placement snapshot index.
        let resource = format!("placement:candidate:{node_did}");
        if let Err(e) = self.authorize(ctx, &resource, "write").await {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("unauthorized to report liveness for {resource}: {e}"),
                code: "UNAUTHORIZED".to_owned(),
                details: String::new(),
            }));
        }

        let live = LiveAllocatable {
            allocatable: data
                .allocatable
                .iter()
                .map(|r| (r.name.clone(), r.quantity.clone()))
                .collect(),
            load_fraction: data.load_fraction,
            last_seen: if data.ts != 0 {
                data.ts
            } else {
                unix_millis_now()
            },
        };
        self.liveness.insert(data.node.clone(), live, LIVENESS_TTL);

        if self.placement_index.record_uri(&node_did).is_none()
            && self.placement_ingest_attempts.insert_if_absent(
                data.node.clone(),
                (),
                PLACEMENT_INGEST_RETRY_TTL,
            )
        {
            if let Some(resolver) = &self.record_resolver {
                if let Err(e) = self
                    .placement_index
                    .ingest_did(resolver.as_ref(), &node_did)
                    .await
                {
                    tracing::warn!(
                        node = %node_did,
                        error = %e,
                        "placement directory ingestion failed for heartbeating node (liveness still recorded)"
                    );
                }
            }
        }

        Ok(DiscoveryResponseVariant::ReportNodeLivenessResult)
    }
}

// ============================================================================
// RequestService implementation
// ============================================================================

#[async_trait(?Send)]
impl RequestService for DiscoveryService {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<hyprstream_rpc::Continuation>)> {
        trace!(
            "Discovery request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_discovery(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        "discovery"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> hyprstream_rpc::crypto::SigningKey {
        (*self.signing_key).clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.expected_audience.as_deref()
    }

    fn jwt_key_source(&self) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        self.jwt_key_source.clone()
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = DiscoveryResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}

// ============================================================================
// #431 — getRecord / getRepo handler tests (access control + CAR proof)
// ============================================================================

#[cfg(test)]
mod get_record_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;

    use std::collections::BTreeMap;

    use hyprstream_pds::car::{build_record_proof_car, parse_car_v1, verify_record_proof};
    use hyprstream_pds::cid::Cid as PdsCid;
    use hyprstream_pds::commit::{Commit, UnsignedCommit};
    use hyprstream_pds::mst::{Node, Proof};
    use hyprstream_pds::record::{ModelRecord, COLLECTION_NSID};
    use hyprstream_pds::tid::Tid;
    use p256::ecdsa::{SigningKey as P256SigningKey, VerifyingKey as P256VerifyingKey};

    const TEST_DID: &str = "did:web:alice.example.com";

    /// Mock authorization provider: allows exactly the (resource, operation)
    /// pairs in its allow-list, denies everything else. Mirrors the Casbin gate
    /// the production PolicyAuthProvider wraps.
    struct MockAuth {
        allow: Vec<(String, String)>,
    }
    #[async_trait(?Send)]
    impl AuthorizationProvider for MockAuth {
        async fn check(
            &self,
            _subject: &str,
            _domain: &str,
            resource: &str,
            operation: &str,
        ) -> Result<bool> {
            Ok(self
                .allow
                .iter()
                .any(|(r, o)| r == resource && o == operation))
        }
    }

    /// Mock record resolver backed by a real PDS-built CAR proof, so a returned
    /// RecordCar verifies under `verify_record_proof`. `None` records → NOT_FOUND.
    struct MockResolver {
        /// (collection, rkey) → (car bytes, uri). Built once in `build`.
        records: BTreeMap<(String, String), (Vec<u8>, String)>,
        /// The `#atproto` verifying key the test repo was signed with, so
        /// `ingest_did` can verify the commit (fail-closed #918).
        verifying_key: p256::ecdsa::VerifyingKey,
    }

    /// Build a small repo with `n` records and return the resolver plus the
    /// published `#atproto` verifying key, target rkey, target record, and proof
    /// so tests can independently verify the CAR the handler returns.
    fn build_repo(
        n: u64,
    ) -> (
        MockResolver,
        P256VerifyingKey,
        String,      // target rkey
        ModelRecord, // target record
        PdsCid,      // target record CID
        Proof,       // target inclusion proof
    ) {
        let signing_key = P256SigningKey::random(&mut rand::rngs::OsRng);
        let verifying_key = P256VerifyingKey::from(&signing_key);

        let mut records: BTreeMap<Tid, ModelRecord> = BTreeMap::new();
        let mut record_cids: BTreeMap<Tid, PdsCid> = BTreeMap::new();
        for i in 0..n {
            let tid = Tid::from_micros(1_700_000_000_000_000 + i * 1000, i as u16);
            let rec = ModelRecord::new(
                format!("at://{TEST_DID}"),
                format!("bafyreiexampleoid{i:020}"),
                "2026-06-24T00:00:00.000Z",
            )
            .unwrap();
            record_cids.insert(tid, rec.cid());
            records.insert(tid, rec);
        }
        let tree = Node::from_records(COLLECTION_NSID, &record_cids);
        let root = tree.root_cid();
        let unsigned = UnsignedCommit::new(TEST_DID.to_owned(), root, Tid::now(), None);
        let commit = Commit::sign(&unsigned, &signing_key);
        let (_root_data, node_blocks) = tree.to_node_data_with_blocks();

        // Pick a middle record as the target.
        let target_tid = *records.keys().nth((n / 2) as usize).unwrap();
        let target_record = records.get(&target_tid).cloned().unwrap();
        let target_cid = record_cids[&target_tid];
        let target_rkey = target_tid.encode();

        // Build a CAR proof for EVERY record so the resolver can answer any rkey.
        let mut out = BTreeMap::new();
        for (tid, rec) in &records {
            let proof = tree.proof(COLLECTION_NSID, tid).unwrap();
            let car = build_record_proof_car(&commit, &proof, &node_blocks, rec);
            let rkey = tid.encode();
            let uri = format!("at://{TEST_DID}/{COLLECTION_NSID}/{rkey}");
            out.insert((COLLECTION_NSID.to_owned(), rkey), (car, uri));
        }
        let target_proof = tree.proof(COLLECTION_NSID, &target_tid).unwrap();

        (
            MockResolver {
                records: out,
                verifying_key,
            },
            verifying_key,
            target_rkey,
            target_record,
            target_cid,
            target_proof,
        )
    }

    #[async_trait(?Send)]
    impl RecordResolver for MockResolver {
        async fn resolve_record(
            &self,
            _did: &str,
            collection: &str,
            rkey: &str,
        ) -> Result<Option<RecordCarData>> {
            Ok(self
                .records
                .get(&(collection.to_owned(), rkey.to_owned()))
                .map(|(car, uri)| RecordCarData {
                    uri: uri.clone(),
                    car: car.clone(),
                }))
        }
        async fn resolve_repo(&self, _did: &str) -> Result<Option<RecordCarData>> {
            Ok(self
                .records
                .values()
                .next()
                .map(|(car, uri)| RecordCarData {
                    uri: uri.clone(),
                    car: car.clone(),
                }))
        }
        async fn resolve_verifying_key(
            &self,
            _did: &str,
        ) -> Result<Option<p256::ecdsa::VerifyingKey>> {
            Ok(Some(self.verifying_key))
        }
    }

    fn service_with(allow: Vec<(&str, &str)>, resolver: MockResolver) -> DiscoveryService {
        let (sk, vk) = hyprstream_rpc::crypto::generate_signing_keypair();
        let allow = allow
            .into_iter()
            .map(|(r, o)| (r.to_owned(), o.to_owned()))
            .collect();
        DiscoveryService::new(
            Arc::new(sk),
            vk,
            hyprstream_rpc::transport::TransportConfig::inproc("discovery-test"),
        )
        .with_auth_provider(Box::new(MockAuth { allow }))
        .with_record_resolver(Arc::new(resolver))
    }

    fn test_ctx() -> EnvelopeContext {
        // A genuine service-identity caller; subject() → "service:test-caller".
        EnvelopeContext::from_callback_service(1, "test-caller")
    }

    /// (a) An accessible record returns a valid CAR proof that verifies offline.
    #[tokio::test]
    async fn get_record_returns_verifiable_car_when_authorized() {
        let (resolver, vk, rkey, target_record, target_cid, proof) = build_repo(6);
        let resource = format!("discovery:record:{TEST_DID}/{COLLECTION_NSID}");
        let svc = service_with(vec![(&resource, "query")], resolver);

        let req = GetRecordRequest {
            uri: format!("at://{TEST_DID}/{COLLECTION_NSID}/{rkey}"),
            did: String::new(),
            collection: String::new(),
            rkey: String::new(),
        };
        let resp = svc.handle_get_record(&test_ctx(), 1, &req).await.unwrap();
        let car = match resp {
            DiscoveryResponseVariant::GetRecordResult(rc) => rc,
            other => panic!("expected GetRecordResult, got {other:?}"),
        };

        // The CAR parses and contains the target record block.
        let (roots, blocks) = parse_car_v1(&car.car).unwrap();
        assert!(!roots.is_empty(), "CAR must have a root commit CID");
        let block_cids: std::collections::BTreeSet<PdsCid> =
            blocks.iter().map(|(c, _)| *c).collect();
        assert!(
            block_cids.contains(&target_cid),
            "CAR must carry the target record block"
        );

        // The returned proof verifies offline against the published #atproto key.
        let commit = blocks
            .iter()
            .find(|(c, _)| *c == roots[0])
            .map(|(_, b)| Commit::from_dag_cbor(b).unwrap())
            .expect("CAR must contain its commit block");
        verify_record_proof(&commit, &vk, &proof, &target_record)
            .expect("the returned CAR proof must verify offline (D5 untrusted-relay)");
    }

    /// (b) THE LOAD-BEARING TEST: denied when the caller lacks read permission
    /// for the target DID/collection — even though the at:// / CID is valid.
    #[tokio::test]
    async fn get_record_denied_without_read_permission() {
        let (resolver, _vk, rkey, _rec, _cid, _proof) = build_repo(6);
        // Allow-list does NOT include discovery:record:<did>/<collection>.
        let svc = service_with(vec![("discovery:something-else", "query")], resolver);

        let req = GetRecordRequest {
            uri: format!("at://{TEST_DID}/{COLLECTION_NSID}/{rkey}"),
            did: String::new(),
            collection: String::new(),
            rkey: String::new(),
        };
        let resp = svc.handle_get_record(&test_ctx(), 1, &req).await.unwrap();
        match resp {
            DiscoveryResponseVariant::Error(e) => {
                assert_eq!(e.code, "UNAUTHORIZED", "deny must surface as UNAUTHORIZED");
                // CRITICAL: the deny must NOT leak the record bytes.
                assert!(
                    !e.message.is_empty(),
                    "unauthorized error should explain the denied resource"
                );
            }
            DiscoveryResponseVariant::GetRecordResult(_) => {
                panic!("ACCESS-CONTROL FAILURE: a valid at:// returned a record without read permission");
            }
            other => panic!("expected UNAUTHORIZED error, got {other:?}"),
        }
    }

    /// (c) A missing record errors cleanly with NOT_FOUND (authorized caller).
    #[tokio::test]
    async fn get_record_missing_returns_not_found() {
        let (resolver, _vk, _rkey, _rec, _cid, _proof) = build_repo(6);
        let resource = format!("discovery:record:{TEST_DID}/{COLLECTION_NSID}");
        let svc = service_with(vec![(&resource, "query")], resolver);

        let req = GetRecordRequest {
            uri: format!("at://{TEST_DID}/{COLLECTION_NSID}/3zzzzzzzzzzzz"),
            did: String::new(),
            collection: String::new(),
            rkey: String::new(),
        };
        let resp = svc.handle_get_record(&test_ctx(), 1, &req).await.unwrap();
        match resp {
            DiscoveryResponseVariant::Error(e) => {
                assert_eq!(e.code, "NOT_FOUND", "absent record must be NOT_FOUND");
            }
            other => panic!("expected NOT_FOUND error, got {other:?}"),
        }
    }

    /// at:// parser: rejects malformed URIs, accepts the canonical 3-part form.
    #[test]
    fn parse_at_uri_round_trip() {
        let (did, coll, rkey) =
            parse_at_uri("at://did:web:alice.example.com/ai.hyprstream.model/3zztslq4be52u")
                .unwrap();
        assert_eq!(did, "did:web:alice.example.com");
        assert_eq!(coll, "ai.hyprstream.model");
        assert_eq!(rkey, "3zztslq4be52u");

        assert!(parse_at_uri("https://x").is_err());
        assert!(parse_at_uri("at://did:web:x").is_err()); // no collection/rkey
        assert!(parse_at_uri("at://did:web:x/coll").is_err()); // no rkey
    }

    /// getRecord with structured (did, collection, rkey) fields (uri empty).
    #[tokio::test]
    async fn get_record_by_structured_fields() {
        let (resolver, _vk, rkey, _rec, target_cid, _proof) = build_repo(4);
        let resource = format!("discovery:record:{TEST_DID}/{COLLECTION_NSID}");
        let svc = service_with(vec![(&resource, "query")], resolver);

        let req = GetRecordRequest {
            uri: String::new(),
            did: TEST_DID.to_owned(),
            collection: COLLECTION_NSID.to_owned(),
            rkey: rkey.clone(),
        };
        let resp = svc.handle_get_record(&test_ctx(), 1, &req).await.unwrap();
        let car = match resp {
            DiscoveryResponseVariant::GetRecordResult(rc) => rc,
            other => panic!("expected GetRecordResult, got {other:?}"),
        };
        let (_roots, blocks) = parse_car_v1(&car.car).unwrap();
        let block_cids: std::collections::BTreeSet<PdsCid> =
            blocks.iter().map(|(c, _)| *c).collect();
        assert!(block_cids.contains(&target_cid));
    }
}

// ============================================================================
// #524 P1 — queryCandidates / reportNodeLiveness handler tests
// ============================================================================

#[cfg(test)]
mod query_candidates_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;

    use std::collections::BTreeMap;

    use crate::generated::discovery_client::{LabelSelector, ResourceRequest, SelectorOp};
    use hyprstream_pds::car::build_car_v1;
    use hyprstream_pds::cid::Cid;
    use hyprstream_pds::commit::{Commit, UnsignedCommit};
    use hyprstream_pds::mst::Node;
    use hyprstream_pds::placement::node::{self, NodeRecord};
    use hyprstream_pds::tid::Tid;
    use p256::ecdsa::{SigningKey as P256SigningKey, VerifyingKey as P256VerifyingKey};

    /// Allows every (resource, operation) pair.
    struct AllowAll;
    #[async_trait(?Send)]
    impl AuthorizationProvider for AllowAll {
        async fn check(
            &self,
            _subject: &str,
            _domain: &str,
            _resource: &str,
            _operation: &str,
        ) -> Result<bool> {
            Ok(true)
        }
    }

    /// Denies `query` access to exactly one `placement:candidate:<did>`
    /// resource; allows write admission so that test setup can heartbeat it.
    struct DenyNode(String);
    #[async_trait(?Send)]
    impl AuthorizationProvider for DenyNode {
        async fn check(
            &self,
            _subject: &str,
            _domain: &str,
            resource: &str,
            operation: &str,
        ) -> Result<bool> {
            Ok(resource != format!("placement:candidate:{}", self.0) || operation != "query")
        }
    }

    /// Denies `write` admission for exactly one node DID.
    struct DenyNodeLiveness(String);
    #[async_trait(?Send)]
    impl AuthorizationProvider for DenyNodeLiveness {
        async fn check(
            &self,
            _subject: &str,
            _domain: &str,
            resource: &str,
            operation: &str,
        ) -> Result<bool> {
            Ok(resource != format!("placement:candidate:{}", self.0) || operation != "write")
        }
    }

    /// Records repository lookups so admission tests can prove a denied DID
    /// never reaches `RecordResolver::resolve_repo`.
    struct TrackingRepoResolver {
        repos: HashMap<String, Vec<u8>>,
        resolved: parking_lot::Mutex<Vec<String>>,
    }
    #[async_trait(?Send)]
    impl RecordResolver for TrackingRepoResolver {
        async fn resolve_record(
            &self,
            _did: &str,
            _collection: &str,
            _rkey: &str,
        ) -> Result<Option<RecordCarData>> {
            Ok(None)
        }
        async fn resolve_repo(&self, did: &str) -> Result<Option<RecordCarData>> {
            self.resolved.lock().push(did.to_owned());
            Ok(self.repos.get(did).map(|car| RecordCarData {
                uri: format!("at://{did}"),
                car: car.clone(),
            }))
        }
    }

    /// A resolver serving one fixed repo CAR per DID (built with a real
    /// `NodeRecord` encoding — not hand-hacked bytes).
    struct FixedRepoResolver {
        repos: HashMap<String, (Vec<u8>, P256VerifyingKey)>,
    }
    #[async_trait(?Send)]
    impl RecordResolver for FixedRepoResolver {
        async fn resolve_record(
            &self,
            _did: &str,
            _collection: &str,
            _rkey: &str,
        ) -> Result<Option<RecordCarData>> {
            Ok(None)
        }
        async fn resolve_repo(&self, did: &str) -> Result<Option<RecordCarData>> {
            Ok(self.repos.get(did).map(|(car, _vk)| RecordCarData {
                uri: format!("at://{did}"),
                car: car.clone(),
            }))
        }
        async fn resolve_verifying_key(&self, did: &str) -> Result<Option<P256VerifyingKey>> {
            Ok(self.repos.get(did).map(|(_, vk)| *vk))
        }
    }

    /// Build a one-record repo CAR for `did` carrying `rec` as its
    /// `ai.hyprstream.placement.node` record.
    fn node_repo_car(did: &str, rec: &NodeRecord) -> (Vec<u8>, P256VerifyingKey) {
        let signing_key = P256SigningKey::random(&mut rand::rngs::OsRng);
        let verifying_key = P256VerifyingKey::from(&signing_key);
        let key = format!("{}/3a", node::COLLECTION_NSID);
        let record_cid = Cid::from_dag_cbor(&rec.to_dag_cbor());
        let mut keyed: BTreeMap<String, Cid> = BTreeMap::new();
        keyed.insert(key, record_cid);
        let tree = Node::from_keyed_records(&keyed);
        let root = tree.root_cid();
        let (_root_data, node_blocks) = tree.to_node_data_with_blocks();
        let unsigned = UnsignedCommit::new(did.to_owned(), root, Tid::now(), None);
        let commit = Commit::sign(&unsigned, &signing_key);
        let mut blocks: Vec<(Cid, Vec<u8>)> = vec![(commit.cid(), commit.to_dag_cbor())];
        for (cid, data) in node_blocks {
            blocks.push((cid, data.encode()));
        }
        blocks.push((record_cid, rec.to_dag_cbor()));
        (build_car_v1(&[commit.cid()], &blocks), verifying_key)
    }

    fn sample_node_record(did: &str, labels: Vec<(&str, &str)>) -> NodeRecord {
        NodeRecord::new(
            format!("at://{did}"),
            labels
                .into_iter()
                .map(|(k, v)| node::Label {
                    key: k.to_owned(),
                    value: v.to_owned(),
                })
                .collect(),
            vec![],
            vec![],
            "2026-06-23T12:34:56.789Z",
        )
        .unwrap()
    }

    fn service_with(
        auth: Box<dyn AuthorizationProvider>,
        repos: HashMap<String, (Vec<u8>, P256VerifyingKey)>,
    ) -> DiscoveryService {
        let (sk, vk) = hyprstream_rpc::crypto::generate_signing_keypair();
        DiscoveryService::new(
            Arc::new(sk),
            vk,
            hyprstream_rpc::transport::TransportConfig::inproc("query-candidates-test"),
        )
        .with_auth_provider(auth)
        .with_record_resolver(Arc::new(FixedRepoResolver { repos }))
    }

    fn test_ctx() -> EnvelopeContext {
        EnvelopeContext::from_callback_service(1, "test-caller")
    }

    async fn heartbeat(
        svc: &DiscoveryService,
        did: &str,
        allocatable: Vec<(&str, &str)>,
        load_fraction: f32,
    ) {
        let req = NodeLiveness {
            node: Did::new(did.to_owned()),
            allocatable: allocatable
                .into_iter()
                .map(|(n, q)| Resource {
                    name: n.to_owned(),
                    quantity: q.to_owned(),
                })
                .collect(),
            load_fraction,
            ts: 0,
        };
        let resp = svc
            .handle_report_node_liveness(&test_ctx(), 1, &req)
            .await
            .unwrap();
        assert!(matches!(
            resp,
            DiscoveryResponseVariant::ReportNodeLivenessResult
        ));
    }

    /// A denied heartbeat must not create a volatile liveness entry or trigger
    /// a first-seen repository poll, even if the caller passed the generic
    /// reportNodeLiveness dispatch gate.
    #[tokio::test]
    async fn liveness_admission_denial_skips_cache_and_repository_ingest() {
        let did = "did:web:unadmitted-node.example.com";
        let rec = sample_node_record(did, vec![]);
        let (car, _verifying_key) = node_repo_car(did, &rec);
        let resolver = Arc::new(TrackingRepoResolver {
            repos: HashMap::from([(did.to_owned(), car)]),
            resolved: parking_lot::Mutex::new(Vec::new()),
        });
        let (sk, vk) = hyprstream_rpc::crypto::generate_signing_keypair();
        let svc = DiscoveryService::new(
            Arc::new(sk),
            vk,
            hyprstream_rpc::transport::TransportConfig::inproc("liveness-admission-test"),
        )
        .with_auth_provider(Box::new(DenyNodeLiveness(did.to_owned())))
        .with_record_resolver(resolver.clone());
        let req = NodeLiveness {
            node: Did::new(did.to_owned()),
            allocatable: vec![],
            load_fraction: 0.1,
            ts: 0,
        };

        let response = svc
            .handle_report_node_liveness(&test_ctx(), 1, &req)
            .await
            .unwrap();
        assert!(matches!(
            response,
            DiscoveryResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "UNAUTHORIZED"
        ));
        assert!(
            svc.liveness.get(&Did::new(did.to_owned())).is_none(),
            "denied DID must not receive a liveness entry"
        );
        assert!(
            svc.placement_index.record_uri(did).is_none(),
            "denied DID must not receive placement facts"
        );
        assert!(
            resolver.resolved.lock().is_empty(),
            "denied DID must not reach RecordResolver::resolve_repo"
        );
    }

    /// An admitted DID with no repo is polled once per bounded retry window,
    /// not once per heartbeat. Its liveness report still refreshes normally.
    #[tokio::test]
    async fn absent_node_repo_does_not_poll_again_before_retry_window() {
        let did = "did:web:node-without-repo.example.com";
        let resolver = Arc::new(TrackingRepoResolver {
            repos: HashMap::new(),
            resolved: parking_lot::Mutex::new(Vec::new()),
        });
        let (sk, vk) = hyprstream_rpc::crypto::generate_signing_keypair();
        let svc = DiscoveryService::new(
            Arc::new(sk),
            vk,
            hyprstream_rpc::transport::TransportConfig::inproc("placement-retry-test"),
        )
        .with_auth_provider(Box::new(AllowAll))
        .with_record_resolver(resolver.clone());

        heartbeat(&svc, did, vec![], 0.9).await;
        heartbeat(&svc, did, vec![("cpu", "8")], 0.1).await;

        assert_eq!(
            resolver.resolved.lock().as_slice(),
            [did],
            "two heartbeats before retry expiry must produce one repo poll"
        );
        let live = svc
            .liveness
            .get(&Did::new(did.to_owned()))
            .expect("admitted heartbeat must still refresh liveness");
        assert!((live.load_fraction - 0.1).abs() < f32::EPSILON);
        assert_eq!(live.allocatable, vec![("cpu".to_owned(), "8".to_owned())]);
    }

    fn empty_query(max_candidates: u32) -> QueryCandidatesRequest {
        QueryCandidatesRequest {
            selectors: vec![],
            resources: vec![],
            max_candidates,
        }
    }

    fn as_set(resp: DiscoveryResponseVariant) -> PlacementCandidateSet {
        match resp {
            DiscoveryResponseVariant::QueryCandidatesResult(s) => s,
            other => panic!("expected QueryCandidatesResult, got {other:?}"),
        }
    }

    /// A node with an ingested NodeRecord but NO liveness heartbeat must never
    /// appear — absence is hard exclusion, not a "stale" flag (decision #1).
    #[tokio::test]
    async fn liveness_hard_excludes_nodes_never_heartbeated() {
        let did = "did:web:node1.example.com";
        let rec = sample_node_record(did, vec![("zone", "us-east")]);
        let svc = service_with(
            Box::new(AllowAll),
            HashMap::from([(did.to_owned(), node_repo_car(did, &rec))]),
        );

        let set = as_set(
            svc.handle_query_candidates(&test_ctx(), 1, &empty_query(0))
                .await
                .unwrap(),
        );
        assert!(set.candidates.is_empty());
        assert_eq!(set.total_matching, 0);
    }

    #[tokio::test]
    async fn label_selector_matches_and_excludes_non_matching_nodes() {
        let did_a = "did:web:node-a.example.com";
        let did_b = "did:web:node-b.example.com";
        let repos = HashMap::from([
            (
                did_a.to_owned(),
                node_repo_car(did_a, &sample_node_record(did_a, vec![("zone", "us-east")])),
            ),
            (
                did_b.to_owned(),
                node_repo_car(did_b, &sample_node_record(did_b, vec![("zone", "us-west")])),
            ),
        ]);
        let svc = service_with(Box::new(AllowAll), repos);
        heartbeat(&svc, did_a, vec![], 0.1).await;
        heartbeat(&svc, did_b, vec![], 0.1).await;

        let req = QueryCandidatesRequest {
            selectors: vec![LabelSelector {
                key: "zone".to_owned(),
                op: SelectorOp::In,
                values: vec!["us-east".to_owned()],
            }],
            resources: vec![],
            max_candidates: 0,
        };
        let set = as_set(
            svc.handle_query_candidates(&test_ctx(), 1, &req)
                .await
                .unwrap(),
        );
        assert_eq!(set.candidates.len(), 1);
        assert_eq!(set.candidates[0].node, did_a);
        assert_eq!(set.total_matching, 1);
    }

    #[tokio::test]
    async fn resource_request_filters_insufficient_live_capacity() {
        let did = "did:web:node1.example.com";
        let rec = sample_node_record(did, vec![]);
        let svc = service_with(
            Box::new(AllowAll),
            HashMap::from([(did.to_owned(), node_repo_car(did, &rec))]),
        );
        heartbeat(&svc, did, vec![("nvidia.com/gpu", "2")], 0.1).await;

        let req = QueryCandidatesRequest {
            selectors: vec![],
            resources: vec![ResourceRequest {
                name: "nvidia.com/gpu".to_owned(),
                min_quantity: "4".to_owned(),
            }],
            max_candidates: 0,
        };
        let set = as_set(
            svc.handle_query_candidates(&test_ctx(), 1, &req)
                .await
                .unwrap(),
        );
        assert!(
            set.candidates.is_empty(),
            "2 GPUs must not satisfy a request for >=4"
        );

        // A satisfiable request against the same live report succeeds.
        let req_ok = QueryCandidatesRequest {
            selectors: vec![],
            resources: vec![ResourceRequest {
                name: "nvidia.com/gpu".to_owned(),
                min_quantity: "1".to_owned(),
            }],
            max_candidates: 0,
        };
        let set_ok = as_set(
            svc.handle_query_candidates(&test_ctx(), 1, &req_ok)
                .await
                .unwrap(),
        );
        assert_eq!(set_ok.candidates.len(), 1);
    }

    /// THE LOAD-BEARING TEST: a per-candidate authz denial silently drops that
    /// node from the result — it is not surfaced as an RPC error.
    #[tokio::test]
    async fn per_candidate_authz_denial_is_silent_not_an_error() {
        let did_a = "did:web:node-a.example.com";
        let did_b = "did:web:node-b.example.com";
        let repos = HashMap::from([
            (
                did_a.to_owned(),
                node_repo_car(did_a, &sample_node_record(did_a, vec![])),
            ),
            (
                did_b.to_owned(),
                node_repo_car(did_b, &sample_node_record(did_b, vec![])),
            ),
        ]);
        let svc = service_with(Box::new(DenyNode(did_a.to_owned())), repos);
        heartbeat(&svc, did_a, vec![], 0.1).await;
        heartbeat(&svc, did_b, vec![], 0.1).await;

        let resp = svc
            .handle_query_candidates(&test_ctx(), 1, &empty_query(0))
            .await
            .unwrap();
        let set = as_set(resp);
        assert_eq!(set.candidates.len(), 1);
        assert_eq!(set.candidates[0].node, did_b);
        assert_eq!(
            set.total_matching, 1,
            "denied candidate must not inflate totalMatching"
        );
    }

    #[tokio::test]
    async fn bounding_respects_max_candidates_and_total_matching_is_pre_bound() {
        let dids: Vec<String> = (0..5)
            .map(|i| format!("did:web:node{i}.example.com"))
            .collect();
        let mut repos = HashMap::new();
        for d in &dids {
            repos.insert(d.clone(), node_repo_car(d, &sample_node_record(d, vec![])));
        }
        let svc = service_with(Box::new(AllowAll), repos);
        for (i, d) in dids.iter().enumerate() {
            heartbeat(&svc, d, vec![], i as f32 * 0.1).await;
        }

        let set = as_set(
            svc.handle_query_candidates(&test_ctx(), 1, &empty_query(2))
                .await
                .unwrap(),
        );
        assert_eq!(set.candidates.len(), 2, "bounded to maxCandidates");
        assert_eq!(
            set.total_matching, 5,
            "totalMatching counts all authorized survivors, not just the bound"
        );
        // Ranked ascending by load_fraction: node0 (0.0) then node1 (0.1).
        assert_eq!(set.candidates[0].node, dids[0]);
        assert_eq!(set.candidates[1].node, dids[1]);
    }

    #[tokio::test]
    async fn zero_max_candidates_uses_default_bound() {
        let dids: Vec<String> = (0..3)
            .map(|i| format!("did:web:node{i}.example.com"))
            .collect();
        let mut repos = HashMap::new();
        for d in &dids {
            repos.insert(d.clone(), node_repo_car(d, &sample_node_record(d, vec![])));
        }
        let svc = service_with(Box::new(AllowAll), repos);
        for d in &dids {
            heartbeat(&svc, d, vec![], 0.0).await;
        }
        let set = as_set(
            svc.handle_query_candidates(&test_ctx(), 1, &empty_query(0))
                .await
                .unwrap(),
        );
        assert_eq!(
            set.candidates.len(),
            3,
            "under the default bound, all 3 come back"
        );
    }

    #[tokio::test]
    async fn heartbeat_refreshes_ttl_and_liveness_fields() {
        let did = "did:web:node1.example.com";
        let rec = sample_node_record(did, vec![]);
        let svc = service_with(
            Box::new(AllowAll),
            HashMap::from([(did.to_owned(), node_repo_car(did, &rec))]),
        );
        heartbeat(&svc, did, vec![("cpu", "1")], 0.9).await;
        heartbeat(&svc, did, vec![("cpu", "8")], 0.1).await;

        let set = as_set(
            svc.handle_query_candidates(&test_ctx(), 1, &empty_query(0))
                .await
                .unwrap(),
        );
        assert_eq!(set.candidates.len(), 1);
        assert!(
            (set.candidates[0].load_fraction - 0.1).abs() < f32::EPSILON,
            "second heartbeat must win"
        );
        assert_eq!(set.candidates[0].allocatable[0].quantity, "8");
    }
}
