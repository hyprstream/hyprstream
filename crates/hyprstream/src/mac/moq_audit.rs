//! Signed-WAL adapter for MoQ/event-plane MAC denials.
//!
//! The enforcement point lives in `hyprstream-rpc`, while the authoritative
//! tamper-evident WAL lives in this parent crate. This adapter keeps that crate
//! direction intact and makes every active MoQ denial use the same
//! [`AuditRecord`] / [`AuditSink`] contract as the 9P PEP.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use hyprstream_rpc::auth::mac::{
    ClearanceSource, MacDenyReason, MoqEventAction, MoqEventPep, MoqMacAuditReason,
    MoqMacAuditRecord, MoqMacAuditSink, RpcObjectLabelResolver, SecurityLabel,
};

use crate::mac::audit::{AuditRecord, AuditSink, DecisionReason};
use crate::mac::te::{Action, Decision, ObjectType, ScopeAction, SubjectType};

/// Reserved audit identities for MoQ/event decisions.
const MOQ_SUBJECT_TYPE: SubjectType = SubjectType(u32::MAX - 2);
const MOQ_OBJECT_TYPE: ObjectType = ObjectType(u32::MAX - 2);

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
    resolver: Arc<dyn RpcObjectLabelResolver>,
    clearance: Arc<dyn ClearanceSource>,
    sink: Arc<dyn AuditSink>,
) -> MoqEventPep {
    MoqEventPep::new(
        resolver,
        clearance,
        Arc::new(MoqAuditSinkAdapter::new(sink)),
    )
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
        Assurance, CompartmentSet, DenyAllObjectResolver, Level, SecurityContext,
        VerifiedKeyMaterial,
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

    #[test]
    fn cross_tenant_moq_deny_is_durable_in_signed_wal() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalAuditStore::open(dir.path(), StubSigner).unwrap());
        let pep = audited_moq_event_pep(
            Arc::new(DenyAllObjectResolver),
            Arc::new(PublicClearance),
            wal.clone(),
        );

        assert_eq!(
            pep.check(
                &Subject::new("did:web:tenant-a"),
                "tenant-b/events/private",
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
        assert_eq!(
            records[0].subject_id.as_deref(),
            Some("did:web:tenant-a")
        );
        assert_eq!(
            records[0].object_id.as_deref(),
            Some("tenant-b/events/private")
        );
        assert_eq!(
            records[0].action,
            Action::from_scope_action(ScopeAction::Subscribe)
        );
    }
}
