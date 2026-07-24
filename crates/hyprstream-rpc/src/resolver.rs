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
use crate::rpc_client::{CallOptions, RpcClient};
use crate::stream_consumer::StreamHandle;
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
pub struct SelectedService {
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

impl SelectedService {
    pub fn same_authority(&self, other: &Self) -> bool {
        self.service_name == other.service_name
            && self.service_did == other.service_did
            && self.response_verifying_key == other.response_verifying_key
            && self.response_ml_dsa65 == other.response_ml_dsa65
            && self.response_key_id == other.response_key_id
            && self.request_kem_recipient == other.request_kem_recipient
            && self.evidence.accepted_state_digest == other.evidence.accepted_state_digest
            && self.evidence.accepted_state_epoch == other.evidence.accepted_state_epoch
    }
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    pub fn service_did(&self) -> &Did {
        &self.service_did
    }

    pub fn response_verifying_key(&self) -> VerifyingKey {
        self.response_verifying_key
    }

    pub fn response_ml_dsa65(&self) -> &[u8] {
        &self.response_ml_dsa65
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

    /// The latest time at which this resolution may be used.
    #[must_use]
    pub fn expires_at_unix_ms(&self) -> i64 {
        self.expires_at_unix_ms
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
    async fn resolve_service(&self, query: ServiceQuery) -> anyhow::Result<SelectedService>;

    /// Resolve the complete deterministic same-authority retry set.  The
    /// default preserves compatibility for resolvers with a single reach.
    async fn resolve_service_candidates(
        &self,
        query: ServiceQuery,
    ) -> anyhow::Result<Vec<SelectedService>> {
        Ok(vec![self.resolve_service(query).await?])
    }

    /// Consult the authoritative accepted-state source immediately before dial.
    /// Implementations reject an advanced, expired, or missing head.
    async fn ensure_current(&self, resolved: &SelectedService) -> anyhow::Result<()>;
}

/// RPC client which retains the identity-bound resolver through every lazy
/// network operation.  It resolves and checks the accepted head immediately
/// before sealing, and constructs the transport and crypto stores from the
/// same snapshot.  A transport failure may advance only through the bounded,
/// deterministic same-authority alternatives returned by that resolver.
pub struct ResolvedRpcClient {
    service_name: String,
    signing_key: crate::crypto::SigningKey,
    token: Option<String>,
    resolver: Arc<dyn ServiceResolver>,
    request_id: std::sync::atomic::AtomicU64,
}

impl ResolvedRpcClient {
    pub fn new(
        service_name: impl Into<String>,
        signing_key: crate::crypto::SigningKey,
        token: Option<String>,
        resolver: Arc<dyn ServiceResolver>,
    ) -> anyhow::Result<Self> {
        let service_name = canonical_service_name(&service_name.into())?;
        Ok(Self {
            service_name,
            signing_key,
            token,
            resolver,
            request_id: std::sync::atomic::AtomicU64::new(1),
        })
    }

    async fn snapshots(&self) -> anyhow::Result<Vec<SelectedService>> {
        let query = ServiceQuery::network(self.service_name.clone())?;
        let max_attempts = query.max_attempts;
        let resolved = self.resolver.resolve_service_candidates(query).await?;
        anyhow::ensure!(
            !resolved.is_empty(),
            "resolver returned no validated alternatives"
        );
        let authority = &resolved[0];
        anyhow::ensure!(
            resolved.iter().all(|item| item.same_authority(authority)),
            "resolver retry set crosses service authority"
        );
        Ok(resolved.into_iter().take(max_attempts).collect())
    }

    fn client_for(&self, snapshot: &SelectedService) -> anyhow::Result<Arc<dyn RpcClient>> {
        let (kem, pq) = snapshot.crypto_stores()?;
        let signer = crate::signer::LocalSigner::new(self.signing_key.clone());
        crate::dial::dial_with_crypto_stores(
            snapshot.transport(),
            signer,
            Some(snapshot.response_verifying_key()),
            self.token.clone(),
            Some(kem),
            Some(pq),
        )
    }

    async fn attempt<T, F, Fut>(&self, mut call: F) -> anyhow::Result<T>
    where
        F: FnMut(Arc<dyn RpcClient>) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<T>>,
    {
        for refresh in 0..2 {
            let mut last_transport_error = None;
            let mut invalidated = None;
            for snapshot in self.snapshots().await? {
                if let Err(error) = self.resolver.ensure_current(&snapshot).await {
                    invalidated = Some(error);
                    break;
                }
                let client = self.client_for(&snapshot)?;
                match call(client).await {
                    Ok(value) => return Ok(value),
                    Err(error)
                        if crate::transport_traits::is_pre_dispatch_transport_error(&error) =>
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
                return Err(anyhow!(
                    "resolved service remained invalid after re-resolution: {error}"
                ));
            }
            return Err(last_transport_error
                .unwrap_or_else(|| anyhow!("validated same-authority alternatives exhausted")));
        }
        unreachable!("bounded re-resolution loop always returns")
    }
}

#[async_trait::async_trait]
impl RpcClient for ResolvedRpcClient {
    async fn call(&self, payload: Vec<u8>) -> anyhow::Result<Vec<u8>> {
        self.attempt(|c| {
            let p = payload.clone();
            async move { c.call(p).await }
        })
        .await
    }
    async fn call_for_service(&self, service: &str, payload: Vec<u8>) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        self.attempt(|c| {
            let p = payload.clone();
            let s = service.clone();
            async move { c.call_for_service(&s, p).await }
        })
        .await
    }
    async fn call_for_service_with_method(
        &self,
        service: &str,
        method_discriminator: u16,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        self.attempt(|c| {
            let p = payload.clone();
            let s = service.clone();
            async move {
                c.call_for_service_with_method(&s, method_discriminator, p)
                    .await
            }
        })
        .await
    }
    async fn call_with_options(
        &self,
        payload: Vec<u8>,
        options: CallOptions,
    ) -> anyhow::Result<Vec<u8>> {
        self.attempt(|c| {
            let p = payload.clone();
            let o = options.clone();
            async move { c.call_with_options(p, o).await }
        })
        .await
    }
    async fn call_with_options_for_service(
        &self,
        service: &str,
        payload: Vec<u8>,
        options: CallOptions,
    ) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        self.attempt(|c| {
            let p = payload.clone();
            let o = options.clone();
            let s = service.clone();
            async move { c.call_with_options_for_service(&s, p, o).await }
        })
        .await
    }
    async fn call_streaming(
        &self,
        payload: Vec<u8>,
        ephemeral: [u8; 32],
    ) -> anyhow::Result<Vec<u8>> {
        self.attempt(|c| {
            let p = payload.clone();
            async move { c.call_streaming(p, ephemeral).await }
        })
        .await
    }
    async fn call_streaming_for_service(
        &self,
        service: &str,
        payload: Vec<u8>,
        ephemeral: [u8; 32],
    ) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        self.attempt(|c| {
            let p = payload.clone();
            let s = service.clone();
            async move { c.call_streaming_for_service(&s, p, ephemeral).await }
        })
        .await
    }
    async fn call_streaming_for_service_with_method(
        &self,
        service: &str,
        method_discriminator: u16,
        payload: Vec<u8>,
        ephemeral: [u8; 32],
    ) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            service == self.service_name,
            "generated service authority mismatch"
        );
        let service = service.to_owned();
        self.attempt(|c| {
            let p = payload.clone();
            let s = service.clone();
            async move {
                c.call_streaming_for_service_with_method(&s, method_discriminator, p, ephemeral).await }
        })
        .await
    }
    async fn open_stream(&self, payload: Vec<u8>) -> anyhow::Result<Box<dyn StreamHandle>> {
        self.attempt(|c| {
            let p = payload.clone();
            async move { c.open_stream(p).await }
        })
        .await
    }
    async fn open_stream_from_info(
        &self,
        info: crate::stream_info::StreamInfo,
        secret: [u8; 32],
        public: [u8; 32],
    ) -> anyhow::Result<Box<dyn StreamHandle>> {
        self.attempt(|c| {
            let i = info.clone();
            async move { c.open_stream_from_info(i, secret, public).await }
        })
        .await
    }
    fn next_id(&self) -> u64 {
        self.request_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
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
    mut ordered: Vec<SelectedService>,
    max_attempts: usize,
    mut attempt: F,
) -> anyhow::Result<T>
where
    F: FnMut(&SelectedService) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    anyhow::ensure!(max_attempts > 0, "retry bound must be non-zero");
    let authority = ordered
        .first()
        .ok_or_else(|| anyhow!("validated retry set is empty"))?;
    anyhow::ensure!(
        ordered.iter().all(|resolved| {
            resolved.service_name == authority.service_name
                && resolved.service_did == authority.service_did
                && resolved.response_verifying_key == authority.response_verifying_key
                && resolved.response_ml_dsa65 == authority.response_ml_dsa65
                && resolved.response_key_id == authority.response_key_id
                && resolved.request_kem_recipient == authority.request_kem_recipient
                && resolved.evidence.accepted_state_digest
                    == authority.evidence.accepted_state_digest
                && resolved.evidence.accepted_state_epoch == authority.evidence.accepted_state_epoch
        }),
        "validated retry set crosses service authority"
    );
    ordered.sort_by(|a, b| {
        b.expires_at_unix_ms
            .cmp(&a.expires_at_unix_ms)
            .then_with(|| {
                transport_fingerprint(&a.transport).cmp(&transport_fingerprint(&b.transport))
            })
            .then_with(|| {
                a.evidence
                    .selected_fingerprint
                    .cmp(&b.evidence.selected_fingerprint)
            })
    });
    ordered.dedup_by(|a, b| a.evidence.selected_fingerprint == b.evidence.selected_fingerprint);
    let mut last_error = None;
    for resolved in ordered.iter().take(max_attempts.min(16)) {
        match attempt(resolved).await {
            Ok(value) => return Ok(value),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("validated retry set exhausted")))
}

/// Filter, canonicalize, rank, and select without insertion-order dependence.
pub fn select_service_candidate(
    query: &ServiceQuery,
    candidates: Vec<ServiceCandidate>,
    now_unix_ms: i64,
) -> anyhow::Result<SelectedService> {
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
    Ok(SelectedService {
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

/// Return every validated alternative in the same deterministic order used by
/// selection. Invalid candidates are discarded independently. Each iteration
/// reuses the authority-ambiguity gate, so the returned set cannot cross a DID,
/// accepted head, response key, or KEM recipient.
pub fn select_service_candidates(
    query: &ServiceQuery,
    mut candidates: Vec<ServiceCandidate>,
    now_unix_ms: i64,
) -> anyhow::Result<Vec<SelectedService>> {
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
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    struct AlwaysAdvancedResolver {
        snapshot: SelectedService,
        resolves: AtomicUsize,
        checks: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ServiceResolver for AlwaysAdvancedResolver {
        async fn resolve_service(&self, _query: ServiceQuery) -> anyhow::Result<SelectedService> {
            self.resolves.fetch_add(1, Ordering::SeqCst);
            Ok(self.snapshot.clone())
        }

        async fn ensure_current(&self, _resolved: &SelectedService) -> anyhow::Result<()> {
            self.checks.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("accepted state advanced")
        }
    }

    #[tokio::test]
    async fn lazy_first_send_re_resolves_once_and_refuses_before_transport() {
        let snapshot = select_service_candidate(
            &network_query(),
            vec![candidate(
                8,
                TransportConfig::iroh([8; 32], Vec::new(), None),
            )],
            1_000,
        )
        .expect("fixture snapshot");
        let resolver = Arc::new(AlwaysAdvancedResolver {
            snapshot,
            resolves: AtomicUsize::new(0),
            checks: AtomicUsize::new(0),
        });
        let signing = crate::crypto::SigningKey::from_bytes(&[0x41; 32]);
        let client = ResolvedRpcClient::new("model", signing, None, resolver.clone())
            .expect("resolved lazy client");
        let error = client
            .call(vec![1, 2, 3])
            .await
            .expect_err("advanced state dialed");
        assert!(
            error.to_string().contains("remained invalid"),
            "unexpected refusal: {error:#}"
        );
        assert_eq!(resolver.resolves.load(Ordering::SeqCst), 2);
        assert_eq!(resolver.checks.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn only_explicit_pre_dispatch_errors_are_retry_safe() {
        let ordinary = anyhow!("service rejected request after dispatch");
        assert!(!crate::transport_traits::is_pre_dispatch_transport_error(
            &ordinary
        ));
        let marked = anyhow::Error::new(crate::transport_traits::PreDispatchTransportError::new(
            anyhow!("dial refused"),
        ));
        assert!(crate::transport_traits::is_pre_dispatch_transport_error(
            &marked
        ));
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
        let mut a = candidate(1, TransportConfig::iroh([1; 32], Vec::new(), None));
        a.capabilities.extend(["x".to_owned(), "y".to_owned()]);
        let mut b = a.clone();
        b.capabilities.remove("x");
        b.capabilities.remove("y");
        b.capabilities.insert("x\0y".to_owned());
        assert_ne!(candidate_fingerprint(&a), candidate_fingerprint(&b));
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
        let first_candidate = candidate(1, TransportConfig::iroh([1; 32], Vec::new(), None));
        let mut second_candidate = first_candidate.clone();
        second_candidate.transport = TransportConfig::iroh([2; 32], Vec::new(), None);
        let first = select_service_candidate(&network_query(), vec![first_candidate], 1_000)
            .unwrap_or_else(|e| panic!("selection failed: {e}"));
        let second = select_service_candidate(&network_query(), vec![second_candidate], 1_000)
            .unwrap_or_else(|e| panic!("selection failed: {e}"));
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let observed = Arc::clone(&seen);
        let result: anyhow::Result<()> =
            retry_validated_candidates(vec![first, second.clone()], 1, move |resolved| {
                observed.lock().push(resolved.service_did.clone());
                std::future::ready(Err(anyhow!("sentinel dial failure")))
            })
            .await;
        assert!(result.is_err());
        assert_eq!(seen.lock().len(), 1);

        let other = select_service_candidate(
            &network_query(),
            vec![candidate(
                2,
                TransportConfig::iroh([3; 32], Vec::new(), None),
            )],
            1_000,
        )
        .unwrap_or_else(|e| panic!("selection failed: {e}"));
        let crossed: anyhow::Result<()> =
            retry_validated_candidates(vec![second, other], 2, |_| std::future::ready(Ok(())))
                .await;
        assert!(crossed.is_err());
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
