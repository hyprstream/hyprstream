//! MoQ / event-plane MAC policy-enforcement point (epic #547, #1271).
//!
//! This plane consumes the shared RPC-PEP contract rather than defining a
//! parallel decision vocabulary. An installed [`MoqEventPep`] returns
//! [`MacDecision`] and resolves labels through [`RpcObjectLabelResolver`].
//! Missing clearance and missing labels deny, as does a failed lattice-floor
//! check. Callers preserve the pre-activation behavior by not installing this
//! PEP; dormant event/MoQ paths remain pass-through.

use std::sync::Arc;

use super::dispatch_pep::{MacDecision, MacDenyReason, RpcObjectLabelResolver};
use super::{SecurityContext, SecurityLabel};
use crate::envelope::Subject;

/// The MoQ/event verb being authorized.
///
/// All actions currently apply the same mandatory label ceiling. The
/// discriminant is retained so future IFC write-direction rules can
/// distinguish publishing from reading without changing the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoqEventAction {
    /// Write an event into a track or prefix.
    Publish,
    /// Subscribe to or read a track.
    Subscribe,
    /// Join an encrypted prefix or decrypt an event.
    JoinDecrypt,
}

/// Resolve a verified event/MoQ subject to its MAC security context.
///
/// The event plane does not have an [`EnvelopeContext`](crate::service::EnvelopeContext),
/// so it derives its subject context from the verified identity available at
/// this boundary. Returning `None` is fail-closed once the PEP is installed.
pub trait ClearanceSource: Send + Sync {
    fn clearance(&self, subject: &Subject) -> Option<SecurityContext>;
}

/// Why a MoQ MAC denial was recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoqMacAuditReason {
    /// A denial returned by the canonical shared MAC contract.
    Mac(MacDenyReason),
    /// An authorizer was installed, but moq-net exposed no per-track callback.
    ///
    /// The transport denies the whole session until #276 lands rather than
    /// serving tracks that bypass the installed policy.
    TrackAdmissionHookUnavailable,
}

/// Plane-neutral denial record handed to the parent crate's MAC audit adapter.
///
/// `hyprstream-rpc` cannot depend on `hyprstream`'s WAL without creating a
/// crate cycle. Active construction therefore requires this record sink; the
/// parent adapter converts it to the canonical signed [`AuditRecord`][1].
///
/// [1]: https://github.com/hyprstream/hyprstream/issues/573
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoqMacAuditRecord {
    /// Independently verified subject, or `None` for anonymous.
    pub subject: Option<String>,
    /// Track or event prefix whose access was denied.
    pub object: String,
    /// Operation that was denied.
    pub action: MoqEventAction,
    /// Resolved subject clearance, when resolution reached that step.
    pub subject_clearance: Option<SecurityLabel>,
    /// Resolved content label, when resolution reached that step.
    pub object_label: Option<SecurityLabel>,
    /// Exact denial cause.
    pub reason: MoqMacAuditReason,
}

/// Required audit destination for an active MoQ/event MAC PEP.
///
/// Production supplies the parent crate's signed WAL adapter. There is
/// deliberately no no-op/default implementation: an installed PEP must always
/// attempt a durable audit write for every denial.
pub trait MoqMacAuditSink: Send + Sync {
    fn record_deny(&self, record: &MoqMacAuditRecord) -> Result<(), String>;
}

/// The installed MoQ/event MAC floor.
///
/// Dormancy is represented by the caller not installing this PEP. Every check
/// on an installed instance is fail-closed.
pub struct MoqEventPep {
    resolver: Arc<dyn RpcObjectLabelResolver>,
    clearance: Arc<dyn ClearanceSource>,
    audit: Arc<dyn MoqMacAuditSink>,
}

impl MoqEventPep {
    /// Construct an active, fail-closed PEP from the canonical object-label
    /// resolver, verified-subject clearance source, and mandatory audit sink.
    pub fn new(
        resolver: Arc<dyn RpcObjectLabelResolver>,
        clearance: Arc<dyn ClearanceSource>,
        audit: Arc<dyn MoqMacAuditSink>,
    ) -> Self {
        Self {
            resolver,
            clearance,
            audit,
        }
    }

    fn audit_deny(
        &self,
        subject: &Subject,
        track_or_prefix: &str,
        action: MoqEventAction,
        subject_clearance: Option<SecurityLabel>,
        object_label: Option<SecurityLabel>,
        reason: MoqMacAuditReason,
    ) {
        let record = MoqMacAuditRecord {
            subject: subject.name().map(str::to_owned),
            object: track_or_prefix.to_owned(),
            action,
            subject_clearance,
            object_label,
            reason,
        };
        if let Err(error) = self.audit.record_deny(&record) {
            tracing::error!(
                target: "hyprstream.mac.audit",
                %error,
                subject = ?record.subject,
                object = %record.object,
                action = ?record.action,
                reason = ?record.reason,
                "MoQ MAC deny could not be durably audited; enforcing deny"
            );
        }
    }

    /// Check the per-track/per-event label ceiling.
    ///
    /// `track_or_prefix` occupies the canonical resolver's service-domain
    /// coordinate. MoQ/event checks have no browser method discriminator, so
    /// the resolver always receives `None`.
    #[must_use]
    pub fn check(
        &self,
        subject: &Subject,
        track_or_prefix: &str,
        action: MoqEventAction,
    ) -> MacDecision {
        let Some(subject_ctx) = self.clearance.clearance(subject) else {
            let reason = MacDenyReason::NoClearance;
            self.audit_deny(
                subject,
                track_or_prefix,
                action,
                None,
                None,
                MoqMacAuditReason::Mac(reason),
            );
            return MacDecision::Deny(reason);
        };
        let Some(object_label) = self.resolver.resolve(track_or_prefix, None) else {
            let reason = MacDenyReason::UnlabeledObject;
            self.audit_deny(
                subject,
                track_or_prefix,
                action,
                Some(*subject_ctx.clearance()),
                None,
                MoqMacAuditReason::Mac(reason),
            );
            return MacDecision::Deny(reason);
        };
        if subject_ctx.can_access(&object_label) {
            MacDecision::Permit
        } else {
            let reason = MacDenyReason::FloorDeny;
            self.audit_deny(
                subject,
                track_or_prefix,
                action,
                Some(*subject_ctx.clearance()),
                Some(object_label),
                MoqMacAuditReason::Mac(reason),
            );
            MacDecision::Deny(reason)
        }
    }

    /// Audit and deny a transport session because no per-track callback exists.
    ///
    /// This is the fail-closed bridge to #276: an installed track authorizer
    /// must never be retained as dead configuration while tracks are served.
    pub fn deny_track_admission_without_hook(&self, subject: &Subject) {
        self.audit_deny(
            subject,
            "<moq-session:track-hook-unavailable>",
            MoqEventAction::Subscribe,
            self.clearance
                .clearance(subject)
                .map(|ctx| *ctx.clearance()),
            None,
            MoqMacAuditReason::TrackAdmissionHookUnavailable,
        );
    }

    pub fn resolver(&self) -> &Arc<dyn RpcObjectLabelResolver> {
        &self.resolver
    }

    pub fn clearance(&self) -> &Arc<dyn ClearanceSource> {
        &self.clearance
    }
}

/// Fail-closed clearance source for an installed event/MoQ PEP.
#[derive(Debug, Clone, Default)]
pub struct DenyAllClearanceSource;

impl ClearanceSource for DenyAllClearanceSource {
    fn clearance(&self, _subject: &Subject) -> Option<SecurityContext> {
        None
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::mac::{
        Assurance, CompartmentSet, DenyAllObjectResolver, Level, VerifiedKeyMaterial,
    };
    use parking_lot::Mutex;

    fn public_label() -> SecurityLabel {
        SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn secret_label() -> SecurityLabel {
        SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY)
    }

    struct StaticResolver {
        secret_track: String,
    }

    impl RpcObjectLabelResolver for StaticResolver {
        fn resolve(&self, service_domain: &str, method: Option<u16>) -> Option<SecurityLabel> {
            assert_eq!(method, None, "event/MoQ checks are not browser RPC");
            if service_domain == self.secret_track {
                Some(secret_label())
            } else {
                Some(public_label())
            }
        }
    }

    struct TieredClearance {
        cleared_did: String,
    }

    impl ClearanceSource for TieredClearance {
        fn clearance(&self, subject: &Subject) -> Option<SecurityContext> {
            if subject.name() == Some(self.cleared_did.as_str()) {
                Some(SecurityContext::from_clearance(
                    secret_label(),
                    VerifiedKeyMaterial::PqHybrid,
                ))
            } else {
                Some(SecurityContext::from_clearance(
                    public_label(),
                    VerifiedKeyMaterial::Classical,
                ))
            }
        }
    }

    #[derive(Default)]
    struct RecordingAudit {
        records: Mutex<Vec<MoqMacAuditRecord>>,
    }

    impl MoqMacAuditSink for RecordingAudit {
        fn record_deny(&self, record: &MoqMacAuditRecord) -> Result<(), String> {
            self.records.lock().push(record.clone());
            Ok(())
        }
    }

    #[test]
    fn installed_pep_audits_every_missing_clearance_deny() {
        let audit = Arc::new(RecordingAudit::default());
        let pep = MoqEventPep::new(
            Arc::new(DenyAllObjectResolver),
            Arc::new(DenyAllClearanceSource),
            audit.clone(),
        );
        for action in [
            MoqEventAction::Publish,
            MoqEventAction::Subscribe,
            MoqEventAction::JoinDecrypt,
        ] {
            assert_eq!(
                pep.check(&Subject::anonymous(), "any", action),
                MacDecision::Deny(MacDenyReason::NoClearance)
            );
        }
        let records = audit.records.lock();
        assert_eq!(records.len(), 3);
        assert!(records
            .iter()
            .all(|record| { record.reason == MoqMacAuditReason::Mac(MacDenyReason::NoClearance) }));
    }

    #[test]
    fn installed_pep_denies_unlabeled_object() {
        let audit = Arc::new(RecordingAudit::default());
        let pep = MoqEventPep::new(
            Arc::new(DenyAllObjectResolver),
            Arc::new(TieredClearance {
                cleared_did: "did:web:cleared".to_owned(),
            }),
            audit.clone(),
        );
        assert_eq!(
            pep.check(
                &Subject::new("did:web:cleared"),
                "missing",
                MoqEventAction::Subscribe,
            ),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );
        let records = audit.records.lock();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].reason,
            MoqMacAuditReason::Mac(MacDenyReason::UnlabeledObject)
        );
    }

    #[test]
    fn installed_pep_enforces_label_ceiling_for_every_action() {
        let audit = Arc::new(RecordingAudit::default());
        let pep = MoqEventPep::new(
            Arc::new(StaticResolver {
                secret_track: "tenant/streams/secret".to_owned(),
            }),
            Arc::new(TieredClearance {
                cleared_did: "did:web:cleared".to_owned(),
            }),
            audit.clone(),
        );
        for action in [
            MoqEventAction::Publish,
            MoqEventAction::Subscribe,
            MoqEventAction::JoinDecrypt,
        ] {
            assert_eq!(
                pep.check(
                    &Subject::new("did:web:public"),
                    "tenant/streams/secret",
                    action,
                ),
                MacDecision::Deny(MacDenyReason::FloorDeny)
            );
            assert_eq!(
                pep.check(
                    &Subject::new("did:web:cleared"),
                    "tenant/streams/secret",
                    action,
                ),
                MacDecision::Permit
            );
        }
        let records = audit.records.lock();
        assert_eq!(records.len(), 3);
        assert!(records
            .iter()
            .all(|record| record.reason == MoqMacAuditReason::Mac(MacDenyReason::FloorDeny)));
    }
}
