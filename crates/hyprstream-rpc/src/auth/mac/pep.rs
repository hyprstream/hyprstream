//! MoQ / event-plane MAC policy-enforcement point (epic #547, #1271).
//!
//! This plane consumes the shared RPC-PEP contract rather than defining a
//! parallel decision vocabulary. An installed [`MoqEventPep`] returns
//! [`MacDecision`] and resolves labels through [`RpcObjectLabelResolver`].
//! Missing clearance and missing labels deny, as does a failed lattice-floor
//! check. Callers preserve the pre-activation behavior by not installing this
//! PEP; dormant event/MoQ paths remain pass-through.

use std::sync::Arc;

use super::dispatch_pep::{
    DenyAllObjectResolver, MacDecision, MacDenyReason, RpcObjectLabelResolver,
};
use super::SecurityContext;
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

/// The installed MoQ/event MAC floor.
///
/// Dormancy is represented by the caller not installing this PEP. Every check
/// on an installed instance is fail-closed.
pub struct MoqEventPep {
    resolver: Arc<dyn RpcObjectLabelResolver>,
    clearance: Arc<dyn ClearanceSource>,
}

impl MoqEventPep {
    /// Construct an active, fail-closed PEP from the canonical object-label
    /// resolver and this plane's verified-subject clearance source.
    pub fn new(
        resolver: Arc<dyn RpcObjectLabelResolver>,
        clearance: Arc<dyn ClearanceSource>,
    ) -> Self {
        Self {
            resolver,
            clearance,
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
        _action: MoqEventAction,
    ) -> MacDecision {
        let Some(subject_ctx) = self.clearance.clearance(subject) else {
            return MacDecision::Deny(MacDenyReason::NoClearance);
        };
        let Some(object_label) = self.resolver.resolve(track_or_prefix, None) else {
            return MacDecision::Deny(MacDenyReason::UnlabeledObject);
        };
        if subject_ctx.can_access(&object_label) {
            MacDecision::Permit
        } else {
            MacDecision::Deny(MacDenyReason::FloorDeny)
        }
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

impl Default for MoqEventPep {
    fn default() -> Self {
        Self::new(
            Arc::new(DenyAllObjectResolver),
            Arc::new(DenyAllClearanceSource),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::mac::{
        Assurance, CompartmentSet, Level, SecurityLabel, VerifiedKeyMaterial,
    };

    fn public_label() -> SecurityLabel {
        SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn secret_label() -> SecurityLabel {
        SecurityLabel::new(
            Level::Secret,
            Assurance::PqHybrid,
            CompartmentSet::EMPTY,
        )
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

    #[test]
    fn installed_default_denies_missing_clearance() {
        let pep = MoqEventPep::default();
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
    }

    #[test]
    fn installed_pep_denies_unlabeled_object() {
        let pep = MoqEventPep::new(
            Arc::new(DenyAllObjectResolver),
            Arc::new(TieredClearance {
                cleared_did: "did:web:cleared".to_owned(),
            }),
        );
        assert_eq!(
            pep.check(
                &Subject::new("did:web:cleared"),
                "missing",
                MoqEventAction::Subscribe,
            ),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );
    }

    #[test]
    fn installed_pep_enforces_label_ceiling_for_every_action() {
        let pep = MoqEventPep::new(
            Arc::new(StaticResolver {
                secret_track: "tenant/streams/secret".to_owned(),
            }),
            Arc::new(TieredClearance {
                cleared_did: "did:web:cleared".to_owned(),
            }),
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
    }
}
