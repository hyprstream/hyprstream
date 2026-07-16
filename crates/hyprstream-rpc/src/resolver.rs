//! Service endpoint resolution abstraction.
//!
//! Defines the `Resolver` trait for converting service names to transport
//! endpoints. The trait is async to support federation (network-based
//! resolution) in addition to local registry lookups.
//!
//! # Architecture
//!
//! ```text
//! hyprstream-rpc:     trait Resolver + SocketKind + pluggable global
//! hyprstream-discovery: EndpointRegistry implements Resolver (local)
//! future:              DiscoveryClient implements Resolver (remote/federated)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_rpc::{Resolver, SocketKind};
//!
//! async fn connect(resolver: &dyn Resolver) -> Result<()> {
//!     let endpoint = resolver.resolve("inference", SocketKind::Rep).await?;
//!     // Use endpoint...
//!     Ok(())
//! }
//! ```

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::anyhow;
use parking_lot::RwLock;

use crate::identity::Did;
use crate::registry::SocketKind;
use crate::transport::{EndpointType, TransportConfig};
use crate::VerifyingKey;

/// Canonical service-resolution request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceQuery {
    pub service_name: String,
    pub required_capabilities: BTreeSet<String>,
    pub profile: ResolverProfile,
    pub max_attempts: usize,
}

impl ServiceQuery {
    pub fn network(service_name: impl Into<String>) -> anyhow::Result<Self> {
        Self::new(
            service_name,
            ["hyprstream-rpc/1".to_owned()],
            ResolverProfile::NetworkDiscovery,
            3,
        )
    }

    pub fn new(
        service_name: impl Into<String>,
        capabilities: impl IntoIterator<Item = String>,
        profile: ResolverProfile,
        max_attempts: usize,
    ) -> anyhow::Result<Self> {
        let service_name = canonical_service_name(&service_name.into())?;
        anyhow::ensure!(max_attempts > 0, "resolution max_attempts must be non-zero");
        Ok(Self {
            service_name,
            required_capabilities: capabilities.into_iter().collect(),
            profile,
            max_attempts: max_attempts.min(16),
        })
    }
}

/// Current accepted-state projection consumed by service resolution.
///
/// It contains only owned public material. Its producer is responsible for
/// obtaining it through the PDS checkpoint-verifying typed read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedStateEvidence {
    pub service_did: Did,
    pub digest: [u8; 64],
    pub epoch: u64,
    pub expires_at_unix_ms: i64,
    pub response_ed25519: [u8; 32],
    pub response_ml_dsa65: Vec<u8>,
}

/// An anchored, suite-complete request recipient and its current key id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchoredKemRecipient {
    pub key_id: String,
    pub recipient: crate::crypto::hybrid_kem::RecipientPublic,
    pub not_after_unix_ms: i64,
}

/// One owned candidate acquired from Discovery/PDS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceCandidate {
    pub service_name: String,
    pub service_did: Did,
    pub response_verifying_key: [u8; 32],
    pub response_ml_dsa65: Vec<u8>,
    pub response_key_id: String,
    pub request_kem_recipient: Option<AnchoredKemRecipient>,
    pub transport: TransportConfig,
    pub capabilities: BTreeSet<String>,
    pub accepted_state: AcceptedStateEvidence,
    pub source_signer: [u8; 32],
    pub expires_at_unix_ms: i64,
}

/// Non-secret deterministic audit record for filter/rank/select.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateDecision {
    pub candidate_fingerprint: String,
    pub accepted: bool,
    pub reason: String,
}

/// Evidence carried with the selected atomic result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionEvidence {
    pub accepted_state_digest: [u8; 64],
    pub accepted_state_epoch: u64,
    pub selected_fingerprint: String,
    pub ordered_decisions: Vec<CandidateDecision>,
}

/// One identity-bound, owned service resolution result.
#[derive(Debug, Clone)]
pub struct ResolvedService {
    service_name: String,
    service_did: Did,
    response_verifying_key: VerifyingKey,
    response_ml_dsa65: Vec<u8>,
    response_key_id: String,
    request_kem_recipient: AnchoredKemRecipient,
    transport: TransportConfig,
    capabilities: BTreeSet<String>,
    expires_at_unix_ms: i64,
    evidence: ResolutionEvidence,
}

impl ResolvedService {
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    pub fn service_did(&self) -> &Did {
        &self.service_did
    }

    pub fn response_verifying_key(&self) -> VerifyingKey {
        self.response_verifying_key
    }

    pub fn response_key_id(&self) -> &str {
        &self.response_key_id
    }

    pub fn request_kem_recipient(&self) -> &AnchoredKemRecipient {
        &self.request_kem_recipient
    }

    pub fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    pub fn capabilities(&self) -> &BTreeSet<String> {
        &self.capabilities
    }

    pub fn evidence(&self) -> &ResolutionEvidence {
        &self.evidence
    }

    /// Re-check the snapshot clock boundary immediately before dialing/sealing.
    pub fn ensure_fresh(&self, now_unix_ms: i64) -> anyhow::Result<()> {
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

    /// Build the exact per-resolution crypto stores used by the dial path.
    /// No global trust-store lookup occurs here.
    pub fn crypto_stores(
        &self,
    ) -> anyhow::Result<(
        Arc<dyn crate::crypto::hybrid_kem::KemTrustStore>,
        Arc<dyn crate::envelope::PqTrustStore>,
    )> {
        let ed = self.response_verifying_key.to_bytes();
        let mut kem = crate::crypto::hybrid_kem::KeyedKemTrustStore::new();
        kem.bind(ed, self.request_kem_recipient.recipient.clone());
        let pq_key = crate::crypto::pq::ml_dsa_vk_from_bytes(&self.response_ml_dsa65)
            .map_err(|e| anyhow!("invalid resolved ML-DSA-65 response key: {e}"))?;
        let mut pq = crate::envelope::KeyedPqTrustStore::new();
        pq.bind(ed, &pq_key);
        Ok((Arc::new(kem), Arc::new(pq)))
    }
}

/// Identity-bound service resolver used by generated clients.
#[async_trait::async_trait]
pub trait ServiceResolver: Send + Sync {
    async fn resolve_service(&self, query: ServiceQuery) -> anyhow::Result<ResolvedService>;

    /// Consult the authoritative accepted-state source immediately before dial.
    /// Implementations reject an advanced, expired, or missing head.
    async fn ensure_current(&self, resolved: &ResolvedService) -> anyhow::Result<()>;
}

fn canonical_service_name(name: &str) -> anyhow::Result<String> {
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
    let bytes = format!("{transport:?}");
    blake3::hash(bytes.as_bytes()).to_hex().to_string()
}

fn candidate_fingerprint(candidate: &ServiceCandidate) -> String {
    let mut h = blake3::Hasher::new();
    h.update(candidate.service_name.as_bytes());
    h.update(candidate.service_did.as_str().as_bytes());
    h.update(&candidate.response_verifying_key);
    h.update(&candidate.response_ml_dsa65);
    h.update(candidate.response_key_id.as_bytes());
    if let Some(kem) = &candidate.request_kem_recipient {
        h.update(kem.key_id.as_bytes());
        h.update(&kem.recipient.encode());
    }
    h.update(transport_fingerprint(&candidate.transport).as_bytes());
    for capability in &candidate.capabilities {
        h.update(capability.as_bytes());
        h.update(&[0]);
    }
    h.update(&candidate.accepted_state.digest);
    h.update(&candidate.accepted_state.epoch.to_be_bytes());
    h.update(candidate.accepted_state.service_did.as_str().as_bytes());
    h.update(&candidate.accepted_state.expires_at_unix_ms.to_be_bytes());
    h.update(&candidate.accepted_state.response_ed25519);
    h.update(&candidate.accepted_state.response_ml_dsa65);
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
        || crate::crypto::pq::ml_dsa_vk_from_bytes(&candidate.response_ml_dsa65).is_err()
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
        || kem.recipient.suite_id != crate::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768
        || kem.recipient.eks.len() != kem.recipient.suite_id.components().len()
        || crate::crypto::hybrid_kem::RecipientPublic::decode(&kem.recipient.encode()).is_err()
    {
        return Some("invalid-expired-or-nonhybrid-kem-recipient");
    }
    None
}

/// Run a bounded retry over an already validated deterministic order.
/// The callback cannot request or substitute any out-of-plan candidate.
pub async fn retry_validated_candidates<T, F, Fut>(
    ordered: Vec<ResolvedService>,
    max_attempts: usize,
    mut attempt: F,
) -> anyhow::Result<T>
where
    F: FnMut(&ResolvedService) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    anyhow::ensure!(max_attempts > 0, "retry bound must be non-zero");
    let mut last_error = None;
    for resolved in ordered.iter().take(max_attempts.min(16)) {
        match attempt(resolved).await {
            Ok(value) => return Ok(value),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("validated retry set is empty")))
}

/// Filter, canonicalize, rank, and select without insertion-order dependence.
pub fn select_service_candidate(
    query: &ServiceQuery,
    candidates: Vec<ServiceCandidate>,
    now_unix_ms: i64,
) -> anyhow::Result<ResolvedService> {
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
        .map_err(|e| anyhow!("invalid selected response verifying key: {e}"))?;
    let request_kem_recipient = selected
        .request_kem_recipient
        .ok_or_else(|| anyhow!("validated candidate lost request KEM recipient"))?;
    let expires_at_unix_ms = selected
        .expires_at_unix_ms
        .min(selected.accepted_state.expires_at_unix_ms)
        .min(request_kem_recipient.not_after_unix_ms);
    Ok(ResolvedService {
        service_name: selected.service_name,
        service_did: selected.service_did,
        response_verifying_key,
        response_ml_dsa65: selected.response_ml_dsa65,
        response_key_id: selected.response_key_id,
        request_kem_recipient,
        transport: selected.transport,
        capabilities: selected.capabilities,
        expires_at_unix_ms,
        evidence: ResolutionEvidence {
            accepted_state_digest: selected.accepted_state.digest,
            accepted_state_epoch: selected.accepted_state.epoch,
            selected_fingerprint,
            ordered_decisions: decisions,
        },
    })
}

/// Async endpoint resolver.
///
/// Implementations convert a (service_name, socket_kind) pair into a
/// concrete `TransportConfig`. The async signature supports both local
/// (in-memory) and remote (network RPC) resolution.
///
/// # Implementors
///
/// - `EndpointRegistry` — local, in-process resolution (sync internally)
/// - `DiscoveryService` — authoritative resolution after bootstrap
/// - `DiscoveryClient` — remote resolution via DiscoveryService RPC (future)
#[async_trait::async_trait]
pub trait Resolver: Send + Sync {
    /// Resolve a service endpoint.
    ///
    /// Returns the transport configuration for the given service and socket
    /// type. Returns `Err` if the service is unknown and no default can be
    /// generated.
    async fn resolve(&self, name: &str, kind: SocketKind) -> anyhow::Result<TransportConfig>;
}

/// Resolver profile names used at service startup.
///
/// Profiles make locality an explicit deployment decision. Generated clients
/// still ask for `(service_name, socket_kind)`; the installed resolver profile
/// decides whether that name maps to an in-process handle, a same-host Unix
/// socket, or a network-discovered peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolverProfile {
    /// Single-process / hermetic-test profile.
    LocalInproc,
    /// Local IPC profile.
    Ipc,
    /// Multi-host production profile backed by DiscoveryService/PDS records.
    NetworkDiscovery,
}

/// Network profile wrapper for a DiscoveryService/PDS-backed resolver.
///
/// The wrapped resolver owns service discovery. This wrapper enforces the
/// network-profile invariant: same-host endpoints (`inproc`, `ipc`,
/// `systemd-fd`) are not valid routable service reach.
pub struct NetworkDiscoveryResolver {
    inner: Arc<dyn Resolver>,
}

impl NetworkDiscoveryResolver {
    pub fn new(inner: Arc<dyn Resolver>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl Resolver for NetworkDiscoveryResolver {
    async fn resolve(&self, name: &str, kind: SocketKind) -> anyhow::Result<TransportConfig> {
        let transport = self.inner.resolve(name, kind).await?;
        match &transport.endpoint {
            EndpointType::Inproc { .. }
            | EndpointType::Ipc { .. }
            | EndpointType::SystemdFd { .. } => Err(anyhow!(
                "network-discovery resolver returned same-host endpoint for service '{name}' ({kind:?})"
            )),
            EndpointType::Quic { .. } | EndpointType::Iroh { .. } => Ok(transport),
        }
    }
}

// ============================================================================
// Pluggable global resolver (replaceable)
// ============================================================================

static GLOBAL_RESOLVER: RwLock<Option<Arc<dyn Resolver>>> = RwLock::new(None);
static GLOBAL_SERVICE_RESOLVER: RwLock<Option<Arc<dyn ServiceResolver>>> = RwLock::new(None);

/// Install the identity-bound production resolver after explicit Discovery
/// bootstrap. This is intentionally separate from the legacy endpoint-only
/// resolver so `TransportConfig` never becomes an authority object.
pub fn set_global_service(resolver: Arc<dyn ServiceResolver>) {
    *GLOBAL_SERVICE_RESOLVER.write() = Some(resolver);
}

/// Clone the installed identity-bound resolver without retaining a lock guard.
pub fn try_global_service() -> Option<Arc<dyn ServiceResolver>> {
    GLOBAL_SERVICE_RESOLVER.read().clone()
}

/// Set the global resolver.
///
/// Can be called multiple times — each call replaces the previous resolver.
/// During bootstrap, `registry::init()` installs an explicit local resolver
/// profile. A networked deployment can replace that with a
/// [`NetworkDiscoveryResolver`] wrapping DiscoveryService or a DiscoveryClient.
///
/// # Example
///
/// ```ignore
/// // Bootstrap: registry installs itself
/// hyprstream_rpc::registry::init(mode, runtime_dir);
///
/// // After DiscoveryService starts, it replaces the bootstrap resolver:
/// hyprstream_rpc::resolver::set_global(discovery_service.clone());
/// ```
pub fn set_global(resolver: Arc<dyn Resolver>) {
    *GLOBAL_RESOLVER.write() = Some(resolver);
}

/// Get the global resolver (non-panicking).
///
/// Returns `None` before `set_global()` has been called.
pub fn try_global() -> Option<Arc<dyn Resolver>> {
    GLOBAL_RESOLVER.read().clone()
}

/// Get the global resolver.
///
/// # Panics
///
/// Panics if `set_global()` has not been called.
#[deprecated(note = "use try_global() instead for graceful degradation (D9)")]
pub fn global() -> Arc<dyn Resolver> {
    #[allow(clippy::expect_used)] // Intentional panic for programming error
    GLOBAL_RESOLVER
        .read()
        .clone()
        .expect("Global resolver not initialized — call resolver::set_global() first")
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn candidate(tag: u8, transport: TransportConfig) -> ServiceCandidate {
        let ed = crate::crypto::SigningKey::from_bytes(&[tag; 32])
            .verifying_key()
            .to_bytes();
        let (_pq_sk, pq_vk) = crate::crypto::pq::ml_dsa_generate_keypair();
        let pq = crate::crypto::pq::ml_dsa_vk_bytes(&pq_vk);
        let kem = crate::crypto::hybrid_kem::generate_recipient(
            crate::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
        )
        .unwrap_or_else(|e| panic!("test KEM generation failed: {e}"))
        .public();
        let did = Did::from(format!("did:at9p:service-{tag}"));
        ServiceCandidate {
            service_name: "model".to_owned(),
            service_did: did.clone(),
            response_verifying_key: ed,
            response_ml_dsa65: pq.clone(),
            response_key_id: format!("{did}#response-{tag}"),
            request_kem_recipient: Some(AnchoredKemRecipient {
                key_id: format!("{did}#kem-{tag}"),
                recipient: kem,
                not_after_unix_ms: 20_000,
            }),
            transport,
            capabilities: ["hyprstream-rpc/1".to_owned()].into_iter().collect(),
            accepted_state: AcceptedStateEvidence {
                service_did: did,
                digest: [tag; 64],
                epoch: u64::from(tag),
                expires_at_unix_ms: 20_000,
                response_ed25519: ed,
                response_ml_dsa65: pq,
            },
            source_signer: ed,
            expires_at_unix_ms: 20_000,
        }
    }

    fn network_query() -> ServiceQuery {
        ServiceQuery::network("model").unwrap_or_else(|e| panic!("test query must be valid: {e}"))
    }

    #[test]
    fn identity_bound_selection_positive_control() {
        let candidate = candidate(7, TransportConfig::iroh([9; 32], Vec::new(), None));
        let expected_response = candidate.response_verifying_key;
        let selected = select_service_candidate(&network_query(), vec![candidate], 1_000)
            .unwrap_or_else(|e| panic!("valid candidate rejected: {e}"));
        assert_eq!(selected.service_did.as_str(), "did:at9p:service-7");
        assert_eq!(
            selected.response_verifying_key.to_bytes(),
            expected_response
        );
        assert!(selected.request_kem_recipient.key_id.ends_with("#kem-7"));
        assert_eq!(selected.evidence.accepted_state_epoch, 7);
        selected
            .crypto_stores()
            .unwrap_or_else(|e| panic!("selected exact crypto stores invalid: {e}"));
    }

    #[test]
    fn required_properties_fail_before_dial() {
        let query = network_query();
        assert!(select_service_candidate(&query, Vec::new(), 1_000).is_err());

        let base = candidate(3, TransportConfig::iroh([3; 32], Vec::new(), None));
        let mut cases = Vec::new();
        let mut expired = base.clone();
        expired.expires_at_unix_ms = 999;
        cases.push(expired);
        let mut expired_state = base.clone();
        expired_state.accepted_state.expires_at_unix_ms = 999;
        cases.push(expired_state);
        let mut missing_capability = base.clone();
        missing_capability.capabilities.clear();
        cases.push(missing_capability);
        let mut wrong_response = base.clone();
        wrong_response.response_verifying_key = [8; 32];
        cases.push(wrong_response);
        let mut wrong_source = base.clone();
        wrong_source.source_signer = [8; 32];
        cases.push(wrong_source);
        let mut omitted_pq = base.clone();
        omitted_pq.response_ml_dsa65.clear();
        omitted_pq.accepted_state.response_ml_dsa65.clear();
        cases.push(omitted_pq);
        let mut missing_kem = base.clone();
        missing_kem.request_kem_recipient = None;
        cases.push(missing_kem);
        let mut expired_kem = base.clone();
        expired_kem
            .request_kem_recipient
            .as_mut()
            .expect("fixture KEM")
            .not_after_unix_ms = 999;
        cases.push(expired_kem);
        let mut omitted_kem_component = base.clone();
        omitted_kem_component
            .request_kem_recipient
            .as_mut()
            .expect("fixture KEM")
            .recipient
            .eks
            .pop();
        cases.push(omitted_kem_component);
        let mut cross_service_kem = base.clone();
        cross_service_kem
            .request_kem_recipient
            .as_mut()
            .expect("fixture KEM")
            .key_id = "did:at9p:other#kem".to_owned();
        cases.push(cross_service_kem);
        let mut wrong_identity = base.clone();
        wrong_identity.service_did = Did::from("did:at9p:other");
        cases.push(wrong_identity);
        let mut reach_only = base.clone();
        reach_only.service_did = Did::default();
        reach_only.accepted_state.service_did = Did::default();
        cases.push(reach_only);
        let mut local = base;
        local.transport = TransportConfig::inproc("hyprstream/model");
        cases.push(local);

        for invalid in cases {
            assert!(
                select_service_candidate(&query, vec![invalid], 1_000).is_err(),
                "invalid candidate reached dial selection"
            );
        }
    }

    #[test]
    fn candidate_permutations_select_identically() {
        let a = candidate(1, TransportConfig::iroh([1; 32], Vec::new(), None));
        let mut b = a.clone();
        b.transport = TransportConfig::iroh([2; 32], Vec::new(), None);
        let first = select_service_candidate(&network_query(), vec![a.clone(), b.clone()], 1_000)
            .unwrap_or_else(|e| panic!("selection failed: {e}"));
        let second = select_service_candidate(&network_query(), vec![b, a], 1_000)
            .unwrap_or_else(|e| panic!("selection failed: {e}"));
        assert_eq!(first.service_did, second.service_did);
        assert_eq!(first.transport, second.transport);
        assert_eq!(first.response_verifying_key, second.response_verifying_key);
        assert_eq!(first.request_kem_recipient, second.request_kem_recipient);
        assert_eq!(first.evidence, second.evidence);
    }

    #[test]
    fn distinct_valid_identities_are_ambiguous() {
        let a = candidate(1, TransportConfig::iroh([1; 32], Vec::new(), None));
        let b = candidate(2, TransportConfig::iroh([2; 32], Vec::new(), None));
        let error = select_service_candidate(&network_query(), vec![a, b], 1_000)
            .expect_err("distinct service authorities were ranked as interchangeable");
        assert!(error.to_string().contains("ambiguous"));
    }

    #[test]
    fn explicit_local_profiles_accept_only_their_local_reach() {
        let inproc_query = ServiceQuery::new(
            "model",
            ["hyprstream-rpc/1".to_owned()],
            ResolverProfile::LocalInproc,
            1,
        )
        .expect("local query");
        assert!(select_service_candidate(
            &inproc_query,
            vec![candidate(4, TransportConfig::inproc("hyprstream/model"))],
            1_000,
        )
        .is_ok());
        assert!(select_service_candidate(
            &inproc_query,
            vec![candidate(4, TransportConfig::ipc("/tmp/model.sock"))],
            1_000,
        )
        .is_err());

        let ipc_query = ServiceQuery::new(
            "model",
            ["hyprstream-rpc/1".to_owned()],
            ResolverProfile::Ipc,
            1,
        )
        .expect("IPC query");
        assert!(select_service_candidate(
            &ipc_query,
            vec![candidate(5, TransportConfig::ipc("/tmp/model.sock"))],
            1_000,
        )
        .is_ok());
    }

    #[tokio::test]
    async fn retries_are_bounded_to_validated_order() {
        let first = select_service_candidate(
            &network_query(),
            vec![candidate(
                1,
                TransportConfig::iroh([1; 32], Vec::new(), None),
            )],
            1_000,
        )
        .unwrap_or_else(|e| panic!("selection failed: {e}"));
        let second = select_service_candidate(
            &network_query(),
            vec![candidate(
                2,
                TransportConfig::iroh([2; 32], Vec::new(), None),
            )],
            1_000,
        )
        .unwrap_or_else(|e| panic!("selection failed: {e}"));
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let observed = Arc::clone(&seen);
        let result: anyhow::Result<()> =
            retry_validated_candidates(vec![first, second], 1, move |resolved| {
                observed.lock().push(resolved.service_did.clone());
                std::future::ready(Err(anyhow!("sentinel dial failure")))
            })
            .await;
        assert!(result.is_err());
        assert_eq!(seen.lock().len(), 1);
    }

    struct StaticResolver(TransportConfig);

    #[async_trait::async_trait]
    impl Resolver for StaticResolver {
        async fn resolve(&self, _name: &str, _kind: SocketKind) -> anyhow::Result<TransportConfig> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn network_discovery_resolver_rejects_inproc_reach() {
        let resolver = NetworkDiscoveryResolver::new(Arc::new(StaticResolver(
            TransportConfig::inproc("hyprstream/policy"),
        )));

        let err = match resolver.resolve("policy", SocketKind::Rep).await {
            Ok(endpoint) => panic!("network resolver accepted local endpoint: {endpoint:?}"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("same-host endpoint"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn network_discovery_resolver_rejects_same_host_reach() {
        for local in [
            TransportConfig::ipc("/run/hyprstream/policy.sock"),
            TransportConfig::systemd_fd(3, "/run/hyprstream/policy.sock"),
        ] {
            let resolver = NetworkDiscoveryResolver::new(Arc::new(StaticResolver(local)));
            let err = match resolver.resolve("policy", SocketKind::Rep).await {
                Ok(endpoint) => {
                    panic!("network resolver accepted same-host endpoint: {endpoint:?}")
                }
                Err(err) => err,
            };
            assert!(
                err.to_string().contains("same-host endpoint"),
                "unexpected error: {err}"
            );
        }
    }

    #[tokio::test]
    async fn network_discovery_resolver_accepts_quic_reach() {
        let addr = "127.0.0.1:9443"
            .parse()
            .unwrap_or_else(|err| panic!("test socket address must parse: {err}"));
        let quic = TransportConfig::quic(addr, "policy.local");
        let resolver = NetworkDiscoveryResolver::new(Arc::new(StaticResolver(quic.clone())));

        let endpoint = resolver
            .resolve("policy", SocketKind::Rep)
            .await
            .unwrap_or_else(|err| panic!("network resolver rejected QUIC endpoint: {err}"));
        assert_eq!(endpoint.endpoint_string(), quic.endpoint_string());
    }

    #[tokio::test]
    async fn network_discovery_resolver_accepts_iroh_reach() {
        let iroh = TransportConfig::iroh([7; 32], Vec::new(), None);
        let resolver = NetworkDiscoveryResolver::new(Arc::new(StaticResolver(iroh.clone())));

        let endpoint = resolver
            .resolve("policy", SocketKind::Rep)
            .await
            .unwrap_or_else(|err| panic!("network resolver rejected iroh endpoint: {err}"));
        assert_eq!(endpoint, iroh);
    }
}
