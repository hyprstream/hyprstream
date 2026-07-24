//! Authoritative 9P policy-enforcement point.
//!
//! The translator supplies the verified caller context and walked object
//! reference. This decider resolves the content-truth label, applies the
//! intrinsic lattice dominance rule (read-direction `can_access` for reads,
//! write-direction `can_write_to` for writes), and records every outcome
//! through the existing tamper-evident MAC audit sink.
//!
//! ## Activation (#1269)
//!
//! [`production_ninep_reference_monitor`] assembles the full
//! [`ReferenceMonitor`](hyprstream_9p::ReferenceMonitor) from the mandatory
//! decider, the genesis content-truth resolver, and a
//! [`NinePClearanceSource`] (the #698 clearance-provenance seam). Production
//! 9P constructors install the monitor via `Translator::with_reference_monitor`;
//! the `anonymous_floor()` fallback is not reachable from any production path.
//!
//! **Fail-closed until #698 + S6 wire clearance and token issuance**: the
//! production clearance source resolves to `None` (deny) for unenrolled
//! subjects, and the S6 sender-bound token is not yet attached to 9P
//! `Tattach`. So the installed monitor denies every operation — there is no
//! permissive default (per #547). The structure is correct; functional
//! permitting follows #698 and the S6 grant path.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use async_trait::async_trait;
use hyprstream_9p::{
    AccessDecider, Action as NinePAction, AttachAuthenticator, ReferenceMonitor,
    SessionContext, VerifiedAttachIdentity,
};
use hyprstream_rpc::auth::mac::{
    ObjectLabelResolver, ObjectRef, SecurityContext, SecurityLabel,
};
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

// ─────────────────────────────────────────────────────────────────────────────
// #1269 — ReferenceMonitor activation: clearance seam, authenticator, assembler
// ─────────────────────────────────────────────────────────────────────────────

/// Clearance provenance seam for the 9P MAC PEP (#698 dependency).
///
/// The production reference monitor derives a subject's [`SecurityContext``
/// from **verified attach credentials** — not from a caller-supplied label.
/// Until #698 wires the real clearance-issuance path, the production impl
/// resolves clearance from the compiled policy's enrollment table keyed by
/// the verified subject DID ([`EnrollmentClearanceSource`]). An unenrolled DID
/// or absent compiled policy resolves to `None` → mandatory deny (there is no
/// permissive default per #547).
///
/// This trait is the **clean clearance-input seam** the fleet-coordination
/// contract asks each activation lane to expose: plane-specific label
/// resolution and the clearance source are the two independent inputs the
/// monitor needs, and this is the clearance one.
pub trait NinePClearanceSource: Send + Sync {
    /// Derive the verified subject's clearance context from `verified_subject`
    /// (a DID or stable credential fingerprint obtained from verified attach
    /// material, **never** an unverified `Tattach.uname` string).
    ///
    /// Returns `None` to deny — the monitor constructs [`SessionContext::deny`]
    /// or a token-less fail-closed session.
    fn clearance_for(&self, verified_subject: &str) -> Option<SecurityContext>;
}

/// Production clearance source backed by the compiled policy's enrollment
/// table via [`crate::mac::exchange_enrollment_resolver`] (#698 Decision D).
///
/// Assurance is floored at `Classical` unconditionally — the truthful label for
/// what the enrollment crypto proves. Raising an actor above Classical is the
/// enrollment-key-registration follow-up (#718).
///
/// When no compiled policy is installed (early boot / dormant node), the
/// resolver returns `None` for every DID → fail-closed.
#[derive(Debug, Default, Clone, Copy)]
pub struct EnrollmentClearanceSource;

impl NinePClearanceSource for EnrollmentClearanceSource {
    fn clearance_for(&self, verified_subject: &str) -> Option<SecurityContext> {
        let resolver = crate::mac::exchange_enrollment_resolver();
        resolver.resolve(verified_subject)
    }
}

/// 9P [`AttachAuthenticator`] that derives [`SessionContext`] from a verified
/// subject identity via a [`NinePClearanceSource`].
///
/// **Fail-closed by construction** — there is no permissive path:
/// - An unresolvable clearance (unenrolled DID, no policy) →
///   [`SessionContext::deny`].
/// - A resolvable clearance but **no S6 sender-bound token** (the token path
///   is not yet wired into 9P `Tattach`) →
///   [`SessionContext::from_verified_clearance`], which denies every op at the
///   token gate.
///
/// Once the S6 grant path issues a sender-bound token at `Tattach`, the
/// authenticator will construct via `SessionContext::from_verified_token`
/// instead — the structural seam ([`NinePClearanceSource`]) stays unchanged.
pub struct ClearanceAttachAuthenticator<C: NinePClearanceSource> {
    clearance: Arc<C>,
}

impl<C: NinePClearanceSource> ClearanceAttachAuthenticator<C> {
    pub fn new(clearance: Arc<C>) -> Self {
        Self { clearance }
    }
}

#[async_trait]
impl<C: NinePClearanceSource + 'static> AttachAuthenticator for ClearanceAttachAuthenticator<C> {
    async fn authenticate(&self, uname: &str, _aname: &str) -> SessionContext {
        // The verified subject identity. In a production authenticator this is
        // derived from the verified ticket/DPoP material, NOT the raw uname.
        // Until the S6 token path lands at Tattach, the uname *is* the verified
        // subject DID for the WS/UDS/vsock planes (the transport already
        // authenticated the peer). The WT plane presents the mount ticket as
        // uname; the authenticator is plane-specific.
        let Some(security_context) = self.clearance.clearance_for(uname) else {
            return SessionContext::deny();
        };
        // No S6 sender-bound token is wired into the 9P attach path yet (#698).
        // A token-less session denies every op at the token gate — fail-closed
        // structural shape, not a permissive default.
        SessionContext::from_verified_clearance(
            VerifiedAttachIdentity::from_verified_identity(uname),
            security_context,
        )
    }
}

/// Assemble the production 9P [`ReferenceMonitor`] from the mandatory audited
/// decider, the genesis content-truth label resolver, and a clearance source.
///
/// All three seams are required (all-or-nothing enforcement). The decider and
/// resolver are the same objects [`production_ninep_decider`] builds; the
/// clearance source is the #698 seam. Call this once at each production 9P
/// constructor and install via `Translator::with_reference_monitor`.
pub fn production_ninep_reference_monitor<C: NinePClearanceSource + 'static>(
    decider: Arc<dyn AccessDecider>,
    clearance: Arc<C>,
) -> Arc<ReferenceMonitor> {
    let resolver = crate::mac::GenesisGate::production().into_resolver();
    let authenticator: Arc<dyn AttachAuthenticator> =
        Arc::new(ClearanceAttachAuthenticator::new(clearance));
    Arc::new(ReferenceMonitor::new(
        authenticator,
        Arc::new(resolver),
        decider,
    ))
}

/// Convenience: assemble the reference monitor with the production
/// [`EnrollmentClearanceSource`] (#698 Decision D, fail-closed when no policy).
pub fn enrollment_ninep_reference_monitor(
    decider: Arc<dyn AccessDecider>,
) -> Arc<ReferenceMonitor> {
    production_ninep_reference_monitor(decider, Arc::new(EnrollmentClearanceSource))
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

        // Read-direction IFC: no-read-up (simple security / dominance).
        // Write-direction IFC: no-write-down (*-property / confinement).
        // The assurance axis is a crypto floor in both directions.
        let permitted = if matches!(action, NinePAction::Write) {
            ctx.can_write_to(&label)
        } else {
            ctx.can_access(&label)
        };
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
    fn writes_use_no_write_down_ifc() {
        let sink = Arc::new(SpySink::default());
        let decider = NinePAccessDecider::new(
            Arc::new(FixtureResolver {
                public: label(Level::Public),
                secret: label(Level::Secret),
            }),
            sink.clone(),
        );

        // Write to a same-level object: permitted (no-write-down satisfied).
        assert!(decider.check(
            &context(Level::Secret),
            ObjectRef::Path(&["secret"]),
            NinePAction::Write,
        ));
        // Write-down (Secret→Public): denied by *-property.
        assert!(!decider.check(
            &context(Level::Secret),
            ObjectRef::Path(&["public"]),
            NinePAction::Write,
        ));
        let records = sink.records.lock();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].decision, Decision::Permit);
        assert_eq!(records[1].reason, DecisionReason::FloorDeny);
    }
}
