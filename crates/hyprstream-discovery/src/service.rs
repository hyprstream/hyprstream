//! Discovery service — exposes EndpointRegistry over RPC.
//!
//! Allows remote clients to discover registered services, their endpoints,
//! socket kinds, and schemas via the standard REQ/REP transport.

use async_trait::async_trait;
use hyprstream_rpc::registry::{self, EndpointRegistry, SocketKind};
use hyprstream_rpc::resolver::{
    select_service_candidate, AcceptedStateEvidence, AnchoredKemRecipient, ResolvedService,
    Resolver, ServiceCandidate, ServiceQuery, ServiceResolver,
};
use hyprstream_rpc::service::{EnvelopeContext, RequestService};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::SigningKey;

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

use anyhow::Result;
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

    /// Resolve the DID's `#atproto` P-256 verifying key — the key a repo
    /// CAR's commit is signed by — or `Ok(None)` when this resolver cannot
    /// provide one.
    ///
    /// This is the **signature-verification seam** for verified-by-construction
    /// ingest: when a key is returned, [`PlacementIndex::ingest_did`] verifies
    /// the ingested repo CAR's commit signature against it *before* any record
    /// enters the index, and a record whose commit fails verification cannot
    /// produce membership or any other derived fact (see #932). `None` is the
    /// resolver declining to participate — the index then retains the trusted
    /// in-process posture documented in `placement_index` (the day-1 stance;
    /// full key resolution for foreign DIDs is the future federation-hardening
    /// follow-up). A default of `Ok(None)` keeps resolvers that do not
    /// participate unchanged.
    async fn resolve_verifying_key(&self, _did: &str) -> Result<Option<p256::ecdsa::VerifyingKey>> {
        Ok(None)
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
pub trait AcceptedStateSource: Send + Sync {
    fn accepted_state(
        &self,
        did: &str,
    ) -> Result<Option<hyprstream_pds::at9p_duplicity::AcceptedAt9pState>>;
}

/// Cloneable production resolver installed after Discovery bootstrap.
pub struct DiscoveryServiceResolver {
    announced_endpoints: Arc<RwLock<HashMap<String, Vec<AnnouncedEndpoint>>>>,
    accepted_state_source: Arc<dyn AcceptedStateSource>,
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
    /// consents), refreshed by polling `record_resolver` lazily: the first
    /// `reportNodeLiveness` heartbeat seen for a DID with no ingested
    /// `NodeRecord` yet triggers a poll of its repo.
    placement_index: PlacementIndex,
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

    pub fn with_accepted_state_source(mut self, source: Arc<dyn AcceptedStateSource>) -> Self {
        self.accepted_state_source = Some(source);
        self
    }

    /// Build the resolver sharing only owned candidate/state handles. No
    /// Cap'n Proto reader or registry guard crosses the boundary.
    pub fn production_resolver(&self) -> Result<Arc<dyn ServiceResolver>> {
        let source = self
            .accepted_state_source
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Discovery accepted-state source is not installed"))?;
        Ok(Arc::new(DiscoveryServiceResolver {
            announced_endpoints: Arc::clone(&self.announced_endpoints),
            accepted_state_source: source,
        }))
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
    fn acquire_candidates(&self, query: &ServiceQuery) -> Result<Vec<ServiceCandidate>> {
        let announced = self.announced_endpoints.read();
        let entries = announced
            .get(&query.service_name)
            .cloned()
            .unwrap_or_default();
        drop(announced);

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
            let state_expiry = accepted_expiry_unix_ms(&state)?;
            let Some(current_key) = state
                .current
                .subject_keys
                .iter()
                .find(|key| key.ed25519_pub.as_slice() == entry.source_signer)
            else {
                continue;
            };
            let response_ed25519: [u8; 32] =
                current_key.ed25519_pub.as_slice().try_into().map_err(|_| {
                    anyhow::anyhow!("accepted response Ed25519 key has wrong length")
                })?;
            let recipient = hyprstream_rpc::crypto::hybrid_kem::RecipientPublic::decode(
                &entry.request_kem_recipient,
            )?;
            let transport = announced_endpoint_to_transport(&entry)?;
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

#[async_trait]
impl ServiceResolver for DiscoveryServiceResolver {
    async fn resolve_service(&self, query: ServiceQuery) -> Result<ResolvedService> {
        let candidates = self.acquire_candidates(&query)?;
        select_service_candidate(&query, candidates, unix_millis_now())
    }

    async fn ensure_current(&self, resolved: &ResolvedService) -> Result<()> {
        resolved.ensure_fresh(unix_millis_now())?;
        let state = self
            .accepted_state_source
            .accepted_state(resolved.service_did.as_str())?
            .ok_or_else(|| anyhow::anyhow!("accepted state disappeared; re-resolution required"))?;
        anyhow::ensure!(
            state.epoch == resolved.evidence.accepted_state_epoch
                && state.head_digest == resolved.evidence.accepted_state_digest,
            "accepted state advanced or forked; re-resolution required"
        );
        let state_expiry = accepted_expiry_unix_ms(&state)?;
        anyhow::ensure!(
            unix_millis_now() < state_expiry,
            "accepted state expired; re-resolution required"
        );
        Ok(())
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
    let (server_name, addr) = rest.split_once(':').ok_or_else(|| {
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
    Ok(TransportConfig::quic(addr, server_name).with_connect_mode())
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
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes};
    use hyprstream_pds::at9p::{
        CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType,
        Transport as At9pTransport,
    };
    use hyprstream_pds::at9p_duplicity::AcceptedAt9pState;
    use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};
    use hyprstream_rpc::transport::{BindMode, EndpointType};

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
        let (pq_signing, pq_verifying) = ml_dsa_generate_keypair();
        let keys = HybridKeyPair::new(
            signing.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_verifying),
        )
        .unwrap_or_else(|e| panic!("test hybrid keys invalid: {e}"));
        let endpoint = ServiceEndpoint::new(At9pTransport::Iroh, "iroh://reach")
            .unwrap_or_else(|e| panic!("test endpoint invalid: {e}"));
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint)
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
                capabilities: ["hyprstream-rpc/1".to_owned()].into_iter().collect(),
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
        assert_eq!(resolved.service_name, "model");
        assert_eq!(resolved.evidence.accepted_state_epoch, 1);
        assert!(resolved.service_did.is_did_at9p());
        assert!(!resolved.request_kem_recipient.recipient.eks.is_empty());
        resolver
            .ensure_current(&resolved)
            .await
            .unwrap_or_else(|e| panic!("unchanged accepted state rejected: {e}"));
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
        hyprstream_rpc::resolver::set_global_service(resolver);
        let client_signing = SigningKey::from_bytes(&[0x44; 32]);
        let _client = crate::DiscoveryClient::from_resolver(client_signing, None)
            .await
            .unwrap_or_else(|e| panic!("generated resolver path failed: {e}"));
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
            message: "prepareStream removed — use StreamChannel::prepare_stream for authenticated streaming".to_owned(),
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
                "getStream removed — use StreamChannel::prepare_stream for authenticated streaming"
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
            message: "listStreams removed — use StreamChannel::prepare_stream for authenticated streaming".to_owned(),
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
    /// this handler was invoked (mirrors `handle_announce`, which needs no
    /// additional internal check for the same reason).
    ///
    /// Re-inserting (heartbeating) the same node refreshes its TTL entry. The
    /// first heartbeat seen for a DID this directory hasn't ingested a
    /// `NodeRecord` for yet lazily triggers a one-shot poll of that DID's repo
    /// (day-1 ingestion cadence — see the `placement_index` module docs); a
    /// failure there does not fail the heartbeat itself.
    async fn handle_report_node_liveness(
        &self,
        _ctx: &EnvelopeContext,
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

        if self.placement_index.record_uri(&node_did).is_none() {
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
            MockResolver { records: out },
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
    use p256::ecdsa::SigningKey as P256SigningKey;

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

    /// Denies exactly the `placement:candidate:<did>` resource for one DID;
    /// allows everything else (including the dispatch-level auto-gate checks).
    struct DenyNode(String);
    #[async_trait(?Send)]
    impl AuthorizationProvider for DenyNode {
        async fn check(
            &self,
            _subject: &str,
            _domain: &str,
            resource: &str,
            _operation: &str,
        ) -> Result<bool> {
            Ok(resource != format!("placement:candidate:{}", self.0))
        }
    }

    /// A resolver serving one fixed repo CAR per DID (built with a real
    /// `NodeRecord` encoding — not hand-hacked bytes).
    struct FixedRepoResolver {
        repos: HashMap<String, Vec<u8>>,
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
            Ok(self.repos.get(did).map(|car| RecordCarData {
                uri: format!("at://{did}"),
                car: car.clone(),
            }))
        }
    }

    /// Build a one-record repo CAR for `did` carrying `rec` as its
    /// `ai.hyprstream.placement.node` record.
    fn node_repo_car(did: &str, rec: &NodeRecord) -> Vec<u8> {
        let signing_key = P256SigningKey::random(&mut rand::rngs::OsRng);
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
        build_car_v1(&[commit.cid()], &blocks)
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
        repos: HashMap<String, Vec<u8>>,
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
