//! MoQ / event-plane MAC policy-enforcement point (epic #547, #1271; typed
//! resolver v16 §10 / #1510).
//!
//! This plane consumes the shared RPC-PEP decision contract
//! ([`MacDecision`]) rather than defining a parallel decision vocabulary,
//! and resolves labels through this plane's **own** typed resolver
//! ([`MoqEventLabelResolver`]). A track/prefix string is parsed into the
//! exact typed identity ([`MoqEventObjectRef`]) exactly once at the boundary
//! helper ([`MoqEventPep::check_event_prefix`] /
//! [`MoqEventPep::check_stream_track`]); the resolver never sees a string,
//! so a MoQ/event coordinate can no longer occupy the RPC/VFS resolver's
//! service-domain slot. Unknown or noncanonical coordinates, unlisted
//! services, missing clearance, and missing labels all deny; a failed
//! lattice-floor check denies. Callers preserve the pre-activation behavior
//! by not installing this PEP; dormant event/MoQ paths remain pass-through.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use super::dispatch_pep::{MacDecision, MacDenyReason};
// The typed identity and resolver are consumed only by the native check
// paths; on wasm32 the PEP keeps its historical type-only surface (same gate
// pattern as the dispatch PEP).
#[cfg(not(target_arch = "wasm32"))]
use super::moq_resolve::{MoqEventLabelResolver, MoqEventObjectRef, MoqEventPlane};
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

impl MoqEventAction {
    /// Wire-stable discriminant for decoded ingress.
    pub const fn discriminant(self) -> u16 {
        match self {
            MoqEventAction::Publish => 0,
            MoqEventAction::Subscribe => 1,
            MoqEventAction::JoinDecrypt => 2,
        }
    }

    /// Decode an action discriminant. Unknown values return `None`: an
    /// ingress carrying a verb this PEP does not know denies — it never
    /// falls back to a coarse default action.
    pub const fn from_discriminant(value: u16) -> Option<Self> {
        match value {
            0 => Some(MoqEventAction::Publish),
            1 => Some(MoqEventAction::Subscribe),
            2 => Some(MoqEventAction::JoinDecrypt),
            _ => None,
        }
    }
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
    #[cfg(not(target_arch = "wasm32"))]
    Mac(MacDenyReason),
    /// The boundary coordinate did not decode into a typed identity.
    ///
    /// Malformed grammar, a noncanonical service segment, or an internal
    /// bookkeeping key (e.g. a tenant-qualified map key) passed where an
    /// object identity is required. The shared decision surfaces this as
    /// `Deny(UnlabeledObject)`; this variant keeps the precise cause in the
    /// plane's audit trail.
    UnknownObjectIdentity,
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
    /// Track or event prefix whose access was denied (the raw boundary
    /// coordinate as received — an audit trail, never an authorization
    /// input).
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
    #[cfg(not(target_arch = "wasm32"))]
    resolver: Arc<dyn MoqEventLabelResolver>,
    clearance: Arc<dyn ClearanceSource>,
    audit: Arc<dyn MoqMacAuditSink>,
}

impl MoqEventPep {
    /// Construct an active, fail-closed PEP from this plane's typed
    /// object-label resolver, verified-subject clearance source, and
    /// mandatory audit sink.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        resolver: Arc<dyn MoqEventLabelResolver>,
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
        if let Err(_error) = self.audit.record_deny(&record) {
            #[cfg(not(target_arch = "wasm32"))]
            tracing::error!(
                target: "hyprstream.mac.audit",
                error = %_error,
                subject = ?record.subject,
                object = %record.object,
                action = ?record.action,
                reason = ?record.reason,
                "MoQ MAC deny could not be durably audited; enforcing deny"
            );
        }
    }

    /// Check the per-track/per-event label ceiling for an already-decoded
    /// typed identity.
    ///
    /// This is the typed core: the object is an
    /// [`MoqEventObjectRef`], never a string, so no caller can smuggle
    /// another plane's coordinate into this plane's resolver.
    #[must_use]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn check(
        &self,
        subject: &Subject,
        object: &MoqEventObjectRef,
        action: MoqEventAction,
    ) -> MacDecision {
        let Some(subject_ctx) = self.clearance.clearance(subject) else {
            let reason = MacDenyReason::NoClearance;
            self.audit_deny(
                subject,
                &object.audit_coordinate(),
                action,
                None,
                None,
                MoqMacAuditReason::Mac(reason),
            );
            return MacDecision::Deny(reason);
        };
        let Some(object_label) = self.resolver.resolve(object) else {
            let reason = MacDenyReason::UnlabeledObject;
            self.audit_deny(
                subject,
                &object.audit_coordinate(),
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
                &object.audit_coordinate(),
                action,
                Some(*subject_ctx.clearance()),
                Some(object_label),
                MoqMacAuditReason::Mac(reason),
            );
            MacDecision::Deny(reason)
        }
    }

    /// Boundary check for an event-plane topic prefix.
    ///
    /// Parses the dot grammar exactly once; a coordinate that does not
    /// decode into a typed identity (including the tenant-qualified
    /// internal map keys the confidential path uses, which are not object
    /// identities) denies and is audited as such.
    #[must_use]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn check_event_prefix(
        &self,
        subject: &Subject,
        prefix: &str,
        action: MoqEventAction,
    ) -> MacDecision {
        match MoqEventObjectRef::parse(MoqEventPlane::Event, prefix) {
            Some(object) => self.check(subject, &object, action),
            None => {
                self.audit_deny(
                    subject,
                    prefix,
                    action,
                    None,
                    None,
                    MoqMacAuditReason::UnknownObjectIdentity,
                );
                MacDecision::Deny(MacDenyReason::UnlabeledObject)
            }
        }
    }

    /// Boundary check for a stream-plane track / broadcast name.
    ///
    /// Parses the slash grammar exactly once; malformed or noncanonical
    /// names deny and are audited as unknown identities.
    #[must_use]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn check_stream_track(
        &self,
        subject: &Subject,
        track: &str,
        action: MoqEventAction,
    ) -> MacDecision {
        match MoqEventObjectRef::parse(MoqEventPlane::Stream, track) {
            Some(object) => self.check(subject, &object, action),
            None => {
                self.audit_deny(
                    subject,
                    track,
                    action,
                    None,
                    None,
                    MoqMacAuditReason::UnknownObjectIdentity,
                );
                MacDecision::Deny(MacDenyReason::UnlabeledObject)
            }
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

    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolver(&self) -> &Arc<dyn MoqEventLabelResolver> {
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

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::mac::{
        Assurance, CompartmentSet, DenyAllMoqEventResolver, Level, MoqEventPolicyRow,
        MoqEventPolicyTable, VerifiedKeyMaterial,
    };
    use parking_lot::Mutex;

    fn public_label() -> SecurityLabel {
        SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn secret_label() -> SecurityLabel {
        SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY)
    }

    /// Declared track policy: the event `worker` source is secret, `registry`
    /// and the stream `streams` service are public.
    fn declared_resolver() -> crate::auth::mac::DeclaredTrackPolicyResolver {
        crate::auth::mac::DeclaredTrackPolicyResolver::new(
            MoqEventPolicyTable::build(
                1,
                [
                    MoqEventPolicyRow::new(MoqEventPlane::Event, "worker", secret_label()).unwrap(),
                    MoqEventPolicyRow::new(MoqEventPlane::Event, "registry", public_label())
                        .unwrap(),
                    MoqEventPolicyRow::new(MoqEventPlane::Stream, "streams", public_label())
                        .unwrap(),
                ],
            )
            .unwrap(),
        )
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

    fn pep(resolver: Arc<dyn MoqEventLabelResolver>, audit: &Arc<RecordingAudit>) -> MoqEventPep {
        MoqEventPep::new(
            resolver,
            Arc::new(TieredClearance {
                cleared_did: "did:web:cleared".to_owned(),
            }),
            audit.clone(),
        )
    }

    #[test]
    fn installed_pep_audits_every_missing_clearance_deny() {
        let audit = Arc::new(RecordingAudit::default());
        let pep = MoqEventPep::new(
            Arc::new(DenyAllMoqEventResolver),
            Arc::new(DenyAllClearanceSource),
            audit.clone(),
        );
        for action in [
            MoqEventAction::Publish,
            MoqEventAction::Subscribe,
            MoqEventAction::JoinDecrypt,
        ] {
            assert_eq!(
                pep.check_event_prefix(&Subject::anonymous(), "registry", action),
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
        let pep = pep(Arc::new(DenyAllMoqEventResolver), &audit);
        assert_eq!(
            pep.check_event_prefix(
                &Subject::new("did:web:cleared"),
                "inference",
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
        let pep = pep(Arc::new(declared_resolver()), &audit);
        for action in [
            MoqEventAction::Publish,
            MoqEventAction::Subscribe,
            MoqEventAction::JoinDecrypt,
        ] {
            assert_eq!(
                pep.check_event_prefix(
                    &Subject::new("did:web:public"),
                    "worker.sandbox1.started",
                    action,
                ),
                MacDecision::Deny(MacDenyReason::FloorDeny)
            );
            assert_eq!(
                pep.check_event_prefix(
                    &Subject::new("did:web:cleared"),
                    "worker.sandbox1.started",
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

    #[test]
    fn known_stream_tracks_resolve_and_permit_through_the_typed_boundary() {
        let audit = Arc::new(RecordingAudit::default());
        let pep = pep(Arc::new(declared_resolver()), &audit);
        // Public clearance dominates the declared public `streams` label.
        assert_eq!(
            pep.check_stream_track(
                &Subject::new("did:web:public"),
                "alice/streams/run-1/i0",
                MoqEventAction::Subscribe,
            ),
            MacDecision::Permit
        );
        // The same service on the event plane is a different, unlisted object.
        assert_eq!(
            pep.check_event_prefix(
                &Subject::new("did:web:public"),
                "streams.session.x",
                MoqEventAction::Subscribe,
            ),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );
    }

    #[test]
    fn unknown_or_noncanonical_coordinates_deny_as_unknown_identities() {
        let audit = Arc::new(RecordingAudit::default());
        let pep = pep(Arc::new(declared_resolver()), &audit);
        let subject = Subject::new("did:web:cleared");

        // Malformed event grammar (uppercase service segment, map-key form).
        for bad in ["Worker", "5:acmeworker", "wor/ker"] {
            assert_eq!(
                pep.check_event_prefix(&subject, bad, MoqEventAction::Publish),
                MacDecision::Deny(MacDenyReason::UnlabeledObject),
                "{bad:?} must not decode"
            );
        }
        // Malformed stream grammar (single segment, traversal).
        for bad in ["alice", "alice/../streams/run"] {
            assert_eq!(
                pep.check_stream_track(&subject, bad, MoqEventAction::Subscribe),
                MacDecision::Deny(MacDenyReason::UnlabeledObject),
                "{bad:?} must not decode"
            );
        }

        let records = audit.records.lock();
        assert_eq!(records.len(), 5);
        assert!(records
            .iter()
            .all(|record| record.reason == MoqMacAuditReason::UnknownObjectIdentity));
    }

    #[test]
    fn unknown_plane_and_action_discriminants_have_no_typed_value_to_check() {
        // An ingress that decoded an unknown plane or verb cannot construct
        // the typed inputs of `check`; it denies before the PEP is reached.
        for unknown in [2u16, 3, u16::MAX] {
            assert_eq!(MoqEventPlane::from_discriminant(unknown), None);
        }
        for unknown in [3u16, 4, u16::MAX] {
            assert_eq!(MoqEventAction::from_discriminant(unknown), None);
        }
        // The known set round-trips.
        for action in [
            MoqEventAction::Publish,
            MoqEventAction::Subscribe,
            MoqEventAction::JoinDecrypt,
        ] {
            assert_eq!(
                MoqEventAction::from_discriminant(action.discriminant()),
                Some(action)
            );
        }
        for plane in [MoqEventPlane::Event, MoqEventPlane::Stream] {
            assert_eq!(
                MoqEventPlane::from_discriminant(plane.discriminant()),
                Some(plane)
            );
        }
    }

    #[test]
    fn empty_declared_table_denies_every_parseable_identity() {
        let audit = Arc::new(RecordingAudit::default());
        let pep = pep(
            Arc::new(crate::auth::mac::DeclaredTrackPolicyResolver::new(
                MoqEventPolicyTable::empty(),
            )),
            &audit,
        );
        let subject = Subject::new("did:web:cleared");
        assert_eq!(
            pep.check_event_prefix(&subject, "worker", MoqEventAction::Publish),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );
        assert_eq!(
            pep.check_stream_track(
                &subject,
                "alice/streams/run-1/i0",
                MoqEventAction::Subscribe
            ),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );
    }
}
