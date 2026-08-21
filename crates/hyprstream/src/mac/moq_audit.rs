//! Signed-WAL adapter for MoQ/event-plane MAC denials.
//!
//! The enforcement point lives in `hyprstream-rpc`, while the authoritative
//! tamper-evident WAL lives in this parent crate. This adapter keeps that crate
//! direction intact and makes every active MoQ denial use the same
//! [`AuditRecord`] / [`AuditSink`] contract as the 9P PEP.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use ed25519_dalek::SigningKey;
use hyprstream_rpc::auth::mac::{
    ClearanceSource, DeclaredTrackPolicyResolver, MacDenyReason, MoqEventAction,
    MoqEventLabelResolver, MoqEventPep, MoqEventPolicyTable, MoqMacAuditReason, MoqMacAuditRecord,
    MoqMacAuditSink, SecurityLabel,
};
use hyprstream_rpc::envelope::Subject;

use crate::mac::audit::{AuditRecord, AuditSink, DecisionReason};
use crate::mac::te::{Action, Decision, ObjectType, ScopeAction, SubjectType};

/// Reserved audit identities for MoQ/event decisions (below the CAS PEP's
/// `u32::MAX - 3` sentinels, and distinct from VFS at `u32::MAX - 2`).
const MOQ_SUBJECT_TYPE: SubjectType = SubjectType(u32::MAX - 4);
const MOQ_OBJECT_TYPE: ObjectType = ObjectType(u32::MAX - 4);

/// Convert RPC-plane denial records into the authoritative MAC audit schema.
pub struct MoqAuditSinkAdapter {
    sink: Arc<dyn AuditSink>,
}

impl MoqAuditSinkAdapter {
    pub fn new(sink: Arc<dyn AuditSink>) -> Self {
        Self { sink }
    }
}

impl MoqMacAuditSink for MoqAuditSinkAdapter {
    fn record_deny(&self, denial: &MoqMacAuditRecord) -> Result<(), String> {
        let policy = crate::mac::compiled_policy();
        let generation = policy.as_ref().map_or(0, |p| p.generation);
        let policy_hash = policy.as_ref().and_then(|p| p.policy_hash().ok());
        let record = AuditRecord {
            seq: 0,
            prev_hash: [0; 32],
            ts_unix_nanos: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos()),
            decision: Decision::Deny,
            generation,
            policy_hash,
            subject_type: MOQ_SUBJECT_TYPE,
            // Missing labels use bottom only as an audit-schema placeholder;
            // the reason remains authoritative and the placeholder is never
            // fed back into the authorization decision.
            subject_clearance: denial
                .subject_clearance
                .unwrap_or_else(SecurityLabel::bottom),
            on_behalf_of: None,
            object_type: MOQ_OBJECT_TYPE,
            object_label: denial.object_label.unwrap_or_else(SecurityLabel::bottom),
            action: audit_action(denial.action),
            reason: audit_reason(denial.reason),
            subject_id: denial.subject.clone(),
            object_id: Some(denial.object.clone()),
        };
        self.sink.record(&record).map_err(|error| error.to_string())
    }
}

/// Construct an active MoQ/event PEP whose denials flow to the canonical MAC
/// audit sink. Production passes a signed [`crate::mac::WalAuditStore`].
pub fn audited_moq_event_pep(
    resolver: Arc<dyn MoqEventLabelResolver>,
    clearance: Arc<dyn ClearanceSource>,
    sink: Arc<dyn AuditSink>,
) -> MoqEventPep {
    MoqEventPep::new(
        resolver,
        clearance,
        Arc::new(MoqAuditSinkAdapter::new(sink)),
    )
}

/// Assemble the production MoQ/event PEP from the declared track-policy
/// table, verified subject contexts, and the mandatory signed audit WAL.
///
/// v16 §10 / #1510: the MoQ/event plane resolves labels from **declared
/// track policy metadata**, never from the VFS/genesis resolver — a
/// track/prefix must not occupy the RPC/VFS service-domain coordinate. The
/// `track_policy` table is the reviewed seam onto the generated dispatch
/// inventory (WS-D / #1505): until that inventory lands, callers pass the
/// empty table and every unlisted track/prefix denies. This assembly has no
/// bootstrap exception and no legacy-resolver path.
pub async fn production_moq_event_pep(
    signing_key: SigningKey,
    oauth: &crate::config::OAuthConfig,
    audit_stream: &str,
    track_policy: MoqEventPolicyTable,
) -> anyhow::Result<MoqEventPep> {
    anyhow::ensure!(
        !audit_stream.is_empty()
            && audit_stream
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'),
        "invalid MoQ MAC audit stream name"
    );
    let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
    let ml_dsa_store = crate::auth::key_rotation::global_ml_dsa_key_store(&secrets_dir, oauth);
    let signer = crate::mac::audit::cose::OwnedCoseAuditSigner::new(
        Arc::new(signing_key),
        ml_dsa_store.active_key().await,
        hyprstream_rpc::envelope::mandatory_envelope_policy(),
    );
    anyhow::ensure!(
        signer.can_sign(),
        "MoQ MAC PEP audit signer unavailable under mandatory Hybrid policy"
    );
    let audit_store = crate::mac::audit::WalAuditStore::open(
        secrets_dir.join("mac-audit").join(audit_stream),
        signer,
    )
    .context("open MoQ MAC audit store")?;
    Ok(audited_moq_event_pep(
        Arc::new(DeclaredTrackPolicyResolver::new(track_policy)),
        Arc::new(VerifiedClaimsMoqClearanceSource),
        Arc::new(audit_store),
    ))
}

/// Event/MoQ subject resolver backed by the same verified-Claims cache as the
/// direct VFS and CAS PEPs.
#[derive(Debug, Default, Clone, Copy)]
pub struct VerifiedClaimsMoqClearanceSource;

impl ClearanceSource for VerifiedClaimsMoqClearanceSource {
    fn clearance(&self, subject: &Subject) -> Option<hyprstream_rpc::auth::mac::SecurityContext> {
        hyprstream_rpc::auth::mac::subject_context(subject, None)
    }
}

const fn audit_action(action: MoqEventAction) -> Action {
    match action {
        MoqEventAction::Publish => Action::from_scope_action(ScopeAction::Publish),
        MoqEventAction::Subscribe | MoqEventAction::JoinDecrypt => {
            Action::from_scope_action(ScopeAction::Subscribe)
        }
    }
}

const fn audit_reason(reason: MoqMacAuditReason) -> DecisionReason {
    match reason {
        MoqMacAuditReason::Mac(MacDenyReason::NoPepInstalled) => DecisionReason::MoqNoPepInstalled,
        MoqMacAuditReason::Mac(MacDenyReason::NoClearance) => DecisionReason::MoqUnlabeledSubject,
        MoqMacAuditReason::Mac(MacDenyReason::UnlabeledObject) => DecisionReason::UnlabeledObject,
        MoqMacAuditReason::Mac(MacDenyReason::FloorDeny) => DecisionReason::FloorDeny,
        MoqMacAuditReason::Mac(MacDenyReason::StaleAuthority) => DecisionReason::MoqStaleAuthority,
        MoqMacAuditReason::UnknownObjectIdentity => DecisionReason::MoqUnknownObjectIdentity,
        MoqMacAuditReason::TrackAdmissionHookUnavailable => {
            DecisionReason::MoqTrackAdmissionHookUnavailable
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mac::{AuditError, AuditSigner, AuditVerifier, WalAuditStore};
    use hyprstream_rpc::auth::mac::{
        Assurance, CompartmentSet, DeclaredTrackPolicyResolver, DenyAllMoqEventResolver, Level,
        MoqEventPolicyRow, MoqEventPolicyTable, SecurityContext, VerifiedKeyMaterial,
    };
    use hyprstream_rpc::envelope::Subject;
    use tempfile::tempdir;

    #[derive(Clone, Copy)]
    struct StubSigner;

    impl AuditSigner for StubSigner {
        fn sign(&self, signing_input: &[u8]) -> Result<Vec<u8>, AuditError> {
            Ok(blake3::hash(signing_input).as_bytes().to_vec())
        }
    }

    impl AuditVerifier for StubSigner {
        fn verify(&self, signing_input: &[u8], signature: &[u8]) -> Result<(), AuditError> {
            if signature == blake3::hash(signing_input).as_bytes() {
                Ok(())
            } else {
                Err(AuditError::Verify("stub signature mismatch".to_owned()))
            }
        }
    }

    struct PublicClearance;

    impl ClearanceSource for PublicClearance {
        fn clearance(&self, _subject: &Subject) -> Option<SecurityContext> {
            Some(SecurityContext::from_clearance(
                SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY),
                VerifiedKeyMaterial::Classical,
            ))
        }
    }

    /// Declared table: `registry` events are public; nothing else is listed.
    fn declared_table() -> MoqEventPolicyTable {
        MoqEventPolicyTable::build(
            1,
            [MoqEventPolicyRow::new(
                hyprstream_rpc::auth::mac::MoqEventPlane::Event,
                "registry",
                SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY),
            )
            .unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn unlisted_moq_prefix_deny_is_durable_in_signed_wal() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalAuditStore::open(dir.path(), StubSigner).unwrap());
        let pep = audited_moq_event_pep(
            Arc::new(DeclaredTrackPolicyResolver::new(declared_table())),
            Arc::new(PublicClearance),
            wal.clone(),
        );

        // `worker` parses as a typed identity but is not in the declared
        // table — an unlisted service denies.
        assert_eq!(
            pep.check_event_prefix(
                &Subject::new("did:web:tenant-a"),
                "worker.sandbox123.started",
                MoqEventAction::Subscribe,
            ),
            hyprstream_rpc::auth::mac::MacDecision::Deny(
                hyprstream_rpc::auth::mac::MacDenyReason::UnlabeledObject
            )
        );

        let records = wal.verify_journal(&StubSigner).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].decision, Decision::Deny);
        assert_eq!(records[0].reason, DecisionReason::UnlabeledObject);
        assert_eq!(records[0].subject_type, SubjectType(u32::MAX - 4));
        assert_eq!(records[0].object_type, ObjectType(u32::MAX - 4));
        assert_eq!(records[0].subject_id.as_deref(), Some("did:web:tenant-a"));
        assert_eq!(
            records[0].object_id.as_deref(),
            Some("worker.sandbox123.started")
        );
        assert_eq!(
            records[0].action,
            Action::from_scope_action(ScopeAction::Subscribe)
        );
    }

    #[test]
    fn unknown_identity_deny_is_durable_with_its_own_reason() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalAuditStore::open(dir.path(), StubSigner).unwrap());
        let pep = audited_moq_event_pep(
            Arc::new(DeclaredTrackPolicyResolver::new(declared_table())),
            Arc::new(PublicClearance),
            wal.clone(),
        );

        // The confidential path's tenant-qualified map key is not an object
        // identity: it denies as an unknown identity, not as an unlisted one.
        assert_eq!(
            pep.check_event_prefix(
                &Subject::new("did:web:tenant-a"),
                "5:tenantworker",
                MoqEventAction::Publish,
            ),
            hyprstream_rpc::auth::mac::MacDecision::Deny(
                hyprstream_rpc::auth::mac::MacDenyReason::UnlabeledObject
            )
        );

        let records = wal.verify_journal(&StubSigner).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].reason, DecisionReason::MoqUnknownObjectIdentity);
        assert_eq!(records[0].object_id.as_deref(), Some("5:tenantworker"));
    }

    #[test]
    fn declared_public_prefix_permits_and_audits_nothing() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalAuditStore::open(dir.path(), StubSigner).unwrap());
        let pep = audited_moq_event_pep(
            Arc::new(DeclaredTrackPolicyResolver::new(declared_table())),
            Arc::new(PublicClearance),
            wal.clone(),
        );

        assert_eq!(
            pep.check_event_prefix(
                &Subject::new("did:web:tenant-a"),
                "registry.repo789.push",
                MoqEventAction::Subscribe,
            ),
            hyprstream_rpc::auth::mac::MacDecision::Permit
        );
        // Only denials flow to the deny sink today (the permit-side WAL is
        // WS-F's class-reserved record).
        assert!(wal.verify_journal(&StubSigner).unwrap().is_empty());
    }

    #[test]
    fn fail_closed_missing_artifact_deny_all_resolver() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalAuditStore::open(dir.path(), StubSigner).unwrap());
        let pep = audited_moq_event_pep(
            Arc::new(DenyAllMoqEventResolver),
            Arc::new(PublicClearance),
            wal.clone(),
        );
        assert_eq!(
            pep.check_event_prefix(
                &Subject::new("did:web:tenant-a"),
                "registry.repo789.push",
                MoqEventAction::Subscribe,
            ),
            hyprstream_rpc::auth::mac::MacDecision::Deny(
                hyprstream_rpc::auth::mac::MacDenyReason::UnlabeledObject
            )
        );
    }
}
