//! Authoritative 9P policy-enforcement point.
//!
//! The translator supplies the verified caller context and walked object
//! reference. This decider resolves the content-truth label, applies the
//! intrinsic lattice dominance rule for read-class operations, fails every
//! write closed pending the IFC write-direction decision, and records every
//! outcome through the existing tamper-evident MAC audit sink.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use hyprstream_9p::{AccessDecider, Action as NinePAction};
use hyprstream_rpc::auth::mac::{ObjectLabelResolver, ObjectRef, SecurityContext, SecurityLabel};
use hyprstream_rpc::SigningKey;

use crate::mac::audit::{AuditRecord, AuditSink, DecisionReason};
use crate::mac::te::{Action, Decision, ObjectType, ScopeAction, SubjectType};

/// Reserved audit identities for 9P PEP decisions. Real compiled-policy type
/// ids grow upward from zero; these sentinels live below the grant-path
/// `u32::MAX` sentinels and cannot collide.
const NINEP_SUBJECT_TYPE: SubjectType = SubjectType(u32::MAX - 1);
const NINEP_OBJECT_TYPE: ObjectType = ObjectType(u32::MAX - 1);

/// Resolver-backed, audited 9P access decider.
pub struct NinePAccessDecider {
    resolver: Arc<dyn ObjectLabelResolver + Send + Sync>,
    sink: Arc<dyn AuditSink>,
}

impl NinePAccessDecider {
    pub fn new(
        resolver: Arc<dyn ObjectLabelResolver + Send + Sync>,
        sink: Arc<dyn AuditSink>,
    ) -> Self {
        Self { resolver, sink }
    }

    fn audit(
        &self,
        ctx: &SecurityContext,
        label: Option<SecurityLabel>,
        action: NinePAction,
        decision: Decision,
        reason: DecisionReason,
    ) -> bool {
        let policy = crate::mac::compiled_policy();
        let generation = policy.as_ref().map_or(0, |p| p.generation);
        let policy_hash = policy.as_ref().and_then(|p| p.policy_hash().ok());
        let record = AuditRecord {
            seq: 0,
            prev_hash: [0; 32],
            ts_unix_nanos: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
            decision,
            generation,
            policy_hash,
            subject_type: NINEP_SUBJECT_TYPE,
            subject_clearance: *ctx.clearance(),
            on_behalf_of: None,
            object_type: NINEP_OBJECT_TYPE,
            // No label exists on an unresolved decision. The bottom value is
            // an audit-schema placeholder only; `UnlabeledObject` is the
            // authoritative reason and the value never enters authorization.
            object_label: label.unwrap_or_else(SecurityLabel::bottom),
            action: audit_action(action),
            reason,
        };

        match self.sink.record(&record) {
            Ok(()) => decision.is_permit(),
            Err(error) => {
                let deny_record = AuditRecord {
                    decision: Decision::Deny,
                    reason: DecisionReason::AuditFailClosed,
                    ..record
                };
                let _ = self.sink.record(&deny_record);
                tracing::error!(
                    target: "hyprstream.mac.audit",
                    %error,
                    reason = DecisionReason::AuditFailClosed.as_str(),
                    "9P decision could not be durably audited; enforcing deny"
                );
                false
            }
        }
    }
}

/// Build the production resolver-backed 9P PEP and its tamper-evident WAL.
///
/// The audit signer is mandatory under the active crypto policy. Callers must
/// propagate an error from this function and refuse to construct a 9P-serving
/// component; substituting [`hyprstream_9p::DenyAllDecider`] would both outage
/// legitimate attaches and bypass the hash-chained MAC audit trail.
pub async fn production_ninep_decider(
    signing_key: SigningKey,
    oauth: &crate::config::OAuthConfig,
    audit_stream: &str,
) -> anyhow::Result<Arc<dyn AccessDecider>> {
    anyhow::ensure!(
        !audit_stream.is_empty()
            && audit_stream
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'),
        "invalid 9P MAC audit stream name"
    );

    let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
    let ml_dsa_store =
        crate::auth::key_rotation::global_ml_dsa_key_store(&secrets_dir, oauth);
    let signer = crate::mac::audit::cose::OwnedCoseAuditSigner::new(
        Arc::new(signing_key),
        ml_dsa_store.active_key().await,
        hyprstream_rpc::envelope::mandatory_envelope_policy(),
    );
    anyhow::ensure!(
        signer.can_sign(),
        "9P MAC PEP audit signer unavailable under mandatory Hybrid policy"
    );

    let audit_store = crate::mac::audit::WalAuditStore::open(
        secrets_dir.join("mac-audit").join(audit_stream),
        signer,
    )
    .context("open 9P MAC audit store")?;
    let resolver = crate::mac::GenesisGate::production().into_resolver();

    Ok(Arc::new(NinePAccessDecider::new(
        Arc::new(resolver),
        Arc::new(audit_store),
    )))
}

impl AccessDecider for NinePAccessDecider {
    fn check(&self, ctx: &SecurityContext, object: ObjectRef<'_>, action: NinePAction) -> bool {
        let Some(label) = self.resolver.resolve(object) else {
            return self.audit(
                ctx,
                None,
                action,
                Decision::Deny,
                DecisionReason::UnlabeledObject,
            );
        };

        if matches!(action, NinePAction::Write) {
            return self.audit(
                ctx,
                Some(label),
                action,
                Decision::Deny,
                DecisionReason::WriteDirectionUndecided,
            );
        }

        let permitted = ctx.can_access(&label);
        self.audit(
            ctx,
            Some(label),
            action,
            if permitted {
                Decision::Permit
            } else {
                Decision::Deny
            },
            if permitted {
                DecisionReason::Permit
            } else {
                DecisionReason::FloorDeny
            },
        )
    }
}

const fn audit_action(action: NinePAction) -> Action {
    match action {
        NinePAction::Write => Action::from_scope_action(ScopeAction::Write),
        NinePAction::Attach
        | NinePAction::Walk
        | NinePAction::Open
        | NinePAction::Read
        | NinePAction::Getattr
        | NinePAction::Readdir => Action::from_scope_action(ScopeAction::Query),
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level, VerifiedKeyMaterial};

    use super::*;
    use crate::mac::audit::{AuditError, AuditRecord};

    struct FixtureResolver {
        public: SecurityLabel,
        secret: SecurityLabel,
    }

    impl ObjectLabelResolver for FixtureResolver {
        fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel> {
            match object {
                ObjectRef::Path(["public"]) => Some(self.public),
                ObjectRef::Path(["secret"]) => Some(self.secret),
                // Existing-but-unlabeled CIDs and unknown paths both resolve
                // to absence, which is denial at the PEP boundary.
                _ => None,
            }
        }
    }

    #[derive(Default)]
    struct SpySink {
        records: Mutex<Vec<AuditRecord>>,
    }

    impl AuditSink for SpySink {
        fn record(&self, record: &AuditRecord) -> Result<(), AuditError> {
            self.records.lock().push(record.clone());
            Ok(())
        }
    }

    fn label(level: Level) -> SecurityLabel {
        SecurityLabel::new(level, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn context(level: Level) -> SecurityContext {
        SecurityContext::from_clearance(label(level), VerifiedKeyMaterial::Classical)
    }

    #[test]
    fn reads_fail_closed_and_every_decision_is_audited() {
        let sink = Arc::new(SpySink::default());
        let decider = NinePAccessDecider::new(
            Arc::new(FixtureResolver {
                public: label(Level::Public),
                secret: label(Level::Secret),
            }),
            sink.clone(),
        );

        assert!(decider.check(
            &context(Level::Secret),
            ObjectRef::Path(&["public"]),
            NinePAction::Read,
        ));
        assert!(!decider.check(
            &context(Level::Public),
            ObjectRef::Path(&["secret"]),
            NinePAction::Read,
        ));
        assert!(!decider.check(
            &context(Level::Secret),
            ObjectRef::Cid(b"unlabeled"),
            NinePAction::Read,
        ));
        assert!(!decider.check(
            &context(Level::Secret),
            ObjectRef::Path(&["does-not-exist"]),
            NinePAction::Read,
        ));

        let records = sink.records.lock();
        assert_eq!(records.len(), 4);
        assert_eq!(records[0].decision, Decision::Permit);
        assert_eq!(records[1].reason, DecisionReason::FloorDeny);
        assert_eq!(records[2].reason, DecisionReason::UnlabeledObject);
        assert_eq!(records[3].reason, DecisionReason::UnlabeledObject);
    }

    #[test]
    fn writes_fail_closed_and_are_audited() {
        let sink = Arc::new(SpySink::default());
        let decider = NinePAccessDecider::new(
            Arc::new(FixtureResolver {
                public: label(Level::Public),
                secret: label(Level::Secret),
            }),
            sink.clone(),
        );

        assert!(!decider.check(
            &context(Level::Secret),
            ObjectRef::Path(&["public"]),
            NinePAction::Write,
        ));
        let records = sink.records.lock();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].reason, DecisionReason::WriteDirectionUndecided);
    }
}
