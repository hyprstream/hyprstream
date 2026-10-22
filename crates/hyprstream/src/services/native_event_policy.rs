//! Finite internal Event policy. Enrollment grants clearance; accepted state
//! supplies live identity/key/currentness. Neither source alone grants access.
use crate::auth::service_enrollment::ServiceEnrollmentManifest;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hyprstream_discovery::admission_roster::{self, DeploymentRosterSource};
use hyprstream_rpc::auth::mac::{
    Assurance, ClearanceSource, CompartmentSet, Level, MoqEventAction, MoqEventPlane,
    MoqEventPolicyRow, MoqEventPolicyTable, SecurityContext, SecurityLabel, VerifiedKeyMaterial,
    SUPPORTED_TRACK_POLICY_REVISION,
};
use hyprstream_rpc::envelope::Subject;
use hyprstream_rpc::events::{EventAuthz, MacEventAuthz, INTERNAL_EVENT_SOURCES};
use hyprstream_rpc::transport::moql_admission::AcceptedStateAuthority;
use std::sync::Arc;

// Lifecycle producers run in these standalone processes. System consumption
// is the existing native capability probe. Worker/workflow is not in this bus.
pub const INTERNAL_EVENT_SERVICES: &[&str] = &[
    "policy",
    "discovery",
    "event",
    "registry",
    "streams",
    "model",
    "oauth",
    "oai",
];

pub fn internal_event_policy() -> anyhow::Result<MoqEventPolicyTable> {
    let label = SecurityLabel::new(Level::Internal, Assurance::PqHybrid, CompartmentSet::EMPTY);
    MoqEventPolicyTable::build(
        SUPPORTED_TRACK_POLICY_REVISION,
        INTERNAL_EVENT_SOURCES
            .iter()
            .map(|source| MoqEventPolicyRow::new(MoqEventPlane::Event, *source, label))
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(Into::into)
}

pub struct EnrolledEventAuthority {
    pub(crate) manifest: Arc<ServiceEnrollmentManifest>,
    pub(crate) accepted: Arc<dyn AcceptedStateAuthority>,
    pub(crate) roster: Arc<dyn DeploymentRosterSource>,
}

impl EnrolledEventAuthority {
    fn key_is_enrolled(&self, subject: &Subject, ed25519: &[u8; 32]) -> bool {
        let Some((name, _)) = self.binding(subject) else {
            return false;
        };
        URL_SAFE_NO_PAD
            .decode(&self.manifest.services[&name].ed25519_pubkey)
            .ok()
            .is_some_and(|key| key.as_slice() == ed25519)
    }
    /// Both key halves must match one atomic current accepted key pair. Foreign
    /// capsules cannot obtain enrollment clearance by naming a known service.
    pub(crate) fn binding(&self, subject: &Subject) -> Option<(String, SecurityContext)> {
        let did = subject.name()?;
        let now = hyprstream_rpc::envelope::current_timestamp();
        let state = self.accepted.accepted_state(did)?;
        if !state.is_live(now) || state.epoch == 0 || state.expires_at_unix_ms.is_none() {
            return None;
        }
        let snapshot = self.roster.roster();
        // Reject stale cached roster projections rather than using them to
        // extend an old key/head after an uncached authority advance.
        if snapshot.iter().find(|(d, _)| d == did)?.1 != state {
            return None;
        }
        let name = admission_roster::roster_service_name(&snapshot, did, now)?;
        if !INTERNAL_EVENT_SERVICES.contains(&name.as_str()) {
            return None;
        }
        let entry = self.manifest.services.get(&name)?;
        let ed: [u8; 32] = URL_SAFE_NO_PAD
            .decode(&entry.ed25519_pubkey)
            .ok()?
            .try_into()
            .ok()?;
        let pq = URL_SAFE_NO_PAD.decode(entry.ml_dsa_pubkey.as_ref()?).ok()?;
        if pq.len() != 1952 || state.subject_key_for(&ed)?.ml_dsa_65 != pq {
            return None;
        }
        Some((
            name,
            SecurityContext::from_clearance(entry.clearance, VerifiedKeyMaterial::PqHybrid),
        ))
    }
}

#[async_trait::async_trait]
impl ClearanceSource for EnrolledEventAuthority {
    async fn clearance(&self, subject: &Subject) -> Option<SecurityContext> {
        self.binding(subject).map(|(_, clearance)| clearance)
    }
}

pub struct NativeEventAuthz {
    pub(crate) authority: Arc<EnrolledEventAuthority>,
    pub(crate) mac: MacEventAuthz,
}

/// Local publishers bind their installed proof to the same enrolled key as
/// remote admitted sessions. A DID lookup alone never installs identity.
pub fn require_enrolled_event_identity(
    proof: &hyprstream_rpc::transport::moql_admission::MoqlAdmissionProof,
) -> anyhow::Result<()> {
    let authority = EnrolledEventAuthority {
        manifest: crate::auth::service_enrollment::global_service_enrollment()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("native Event requires validated service enrollment"))?,
        accepted: hyprstream_discovery::production_moql_accepted_state_authority()?,
        roster: hyprstream_discovery::production_deployment_roster()?,
    };
    anyhow::ensure!(
        authority.key_is_enrolled(
            &Subject::new(proof.did.clone()),
            &proof.ed25519.verifying_key().to_bytes()
        ),
        "native Event proof key is not its current authority-enrolled service key",
    );
    let (name, _) = authority
        .binding(&Subject::new(proof.did.clone()))
        .ok_or_else(|| anyhow::anyhow!("native Event enrollment became stale"))?;
    let enrolled_pq = URL_SAFE_NO_PAD.decode(
        authority.manifest.services[&name]
            .ml_dsa_pubkey
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("native Event enrollment lacks hybrid key"))?,
    )?;
    anyhow::ensure!(
        enrolled_pq == hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&proof.ml_dsa_65),
        "native Event proof PQ key is not its authority-enrolled key"
    );
    Ok(())
}

#[async_trait::async_trait]
impl EventAuthz for NativeEventAuthz {
    fn admission_is_current(
        &self,
        peer: &hyprstream_rpc::transport::moql_admission::AdmittedMoqPeer,
    ) -> bool {
        let Some(did) = peer.peer.subject.as_ref() else {
            return false;
        };
        let subject = Subject::new(did.clone());
        peer.tenant == admission_roster::LOCAL_TENANT
            && self
                .authority
                .key_is_enrolled(&subject, &peer.subject_ed25519)
            && self
                .authority
                .accepted
                .accepted_state(did)
                .is_some_and(|state| {
                    state.epoch == peer.epoch && state.head_digest == peer.head_digest
                })
    }
    async fn revalidate_publish(&self, subject: &Subject, source: &str) -> bool {
        let Some((name, _)) = self.authority.binding(subject) else {
            return false;
        };
        (source == "system" || (source == "model" && name == "model"))
            && self.mac.revalidate_publish(subject, source).await
    }
    async fn revalidate_subscribe(&self, subject: &Subject, source: &str) -> bool {
        source == "system"
            && self.authority.binding(subject).is_some()
            && self.mac.revalidate_subscribe(subject, source).await
    }
    async fn can_publish(&self, subject: &Subject, source: &str) -> bool {
        if !self.mac.can_publish(subject, source).await {
            return false;
        }
        let Some((name, _)) = self.authority.binding(subject) else {
            return self
                .mac
                .deny_scope(subject, source, MoqEventAction::Publish);
        };
        // The independent live event_publishers allowlist is enforced by the
        // transport before ingress. MAC and source ownership are additional.
        if source == "system" || (source == "model" && name == "model") {
            return true;
        }
        self.mac
            .deny_scope(subject, source, MoqEventAction::Publish)
    }

    async fn can_subscribe(&self, subject: &Subject, source: &str) -> bool {
        if !self.mac.can_subscribe(subject, source).await {
            return false;
        }
        if self.authority.binding(subject).is_some() && source == "system" {
            return true;
        }
        self.mac
            .deny_scope(subject, source, MoqEventAction::Subscribe)
    }
}

/// Shared by both production constructors. The authority manifest was already
/// validated against bootstrap keys at startup; absence fails native startup.
pub async fn production_event_authz(
    signing_key: ed25519_dalek::SigningKey,
    config: &crate::config::HyprConfig,
    audit_stream: &str,
) -> anyhow::Result<Arc<dyn EventAuthz>> {
    let policy = internal_event_policy()?;
    if !config.quic.iroh_required() && !hyprstream_discovery::native_network_required() {
        return Ok(Arc::new(MacEventAuthz::new(
            crate::mac::production_moq_event_pep(signing_key, &config.oauth, audit_stream, policy)
                .await?,
        )));
    }
    let authority = Arc::new(EnrolledEventAuthority {
        manifest: crate::auth::service_enrollment::global_service_enrollment()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("native Event requires validated service enrollment"))?,
        accepted: hyprstream_discovery::production_moql_accepted_state_authority()?,
        roster: hyprstream_discovery::production_deployment_roster()?,
    });
    let pep = crate::mac::production_moq_event_pep_with_clearance(
        signing_key,
        &config.oauth,
        audit_stream,
        policy,
        authority.clone(),
    )
    .await?;
    Ok(Arc::new(NativeEventAuthz {
        authority,
        mac: MacEventAuthz::new(pep),
    }))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
#[path = "native_event_policy_tests.rs"]
mod tests;
