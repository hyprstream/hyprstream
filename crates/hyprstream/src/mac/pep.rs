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
//! decider, the genesis content-truth resolver, and an attach authenticator.
//! Production 9P constructors install the monitor via
//! `Translator::with_reference_monitor`; until the verified attach credential
//! is wired into that seam they explicitly install a deny-only authenticator.
//!
//! **Fail-closed until #698 + S6 wire clearance and token issuance**: the
//! raw `Tattach.uname` is not verified identity material, and the S6
//! sender-bound token is not yet attached to 9P `Tattach`. So the installed
//! monitor denies every operation — there is no permissive default (per #547).
//! Functional permitting requires an authenticator that derives both identity
//! and tenant from verified attach credentials.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use hyprstream_9p::{
    AccessDecider, Action as NinePAction, AnonymousAuthenticator, AttachAuthenticator,
    ReferenceMonitor, ReferenceMonitorDenyReason, SessionContext, VerifiedAttachIdentity,
};
use hyprstream_rpc::auth::mac::{
    MacDecision, MacDenyReason, ObjectLabelResolver, ObjectRef, SecurityContext, SecurityLabel,
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
            subject_id: None,
            object_id: None,
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

/// Constructs a 9P [`SessionContext`] from an identity that an attach
/// credential verifier has already authenticated and tenant-bound.
///
/// **Fail-closed by construction** — there is no permissive path:
/// - An unresolvable clearance (unenrolled DID, no policy) →
///   [`SessionContext::deny`].
/// - A resolvable clearance but **no S6 sender-bound token** (the token path
///   is not yet wired into 9P `Tattach`) →
///   [`SessionContext::from_verified_clearance`], which denies every op at the
///   token gate.
///
/// This type deliberately does **not** implement [`AttachAuthenticator`]:
/// that trait receives raw `Tattach` fields, while this factory accepts only a
/// [`VerifiedAttachIdentity`] produced after credential verification. This
/// prevents raw `uname` from silently becoming a verified identity when S6 is
/// wired.
pub struct VerifiedClearanceSessionFactory<C: NinePClearanceSource> {
    clearance: Arc<C>,
}

impl<C: NinePClearanceSource> VerifiedClearanceSessionFactory<C> {
    pub fn new(clearance: Arc<C>) -> Self {
        Self { clearance }
    }

    /// Derive a token-less, fail-closed session from a credential-verified,
    /// tenant-bound identity.
    pub fn session_for(&self, identity: VerifiedAttachIdentity) -> SessionContext {
        let Some(security_context) = self.clearance.clearance_for(identity.subject()) else {
            return SessionContext::deny();
        };
        SessionContext::from_verified_clearance(identity, security_context)
    }
}

/// Assemble the production 9P [`ReferenceMonitor`] from the mandatory audited
/// decider, the genesis content-truth label resolver, and a clearance source.
///
/// All three seams are required (all-or-nothing enforcement). The caller must
/// supply an authenticator that verifies the attach credential itself; passing
/// [`AnonymousAuthenticator`] explicitly selects the fail-closed state for
/// transports whose verified attach-credential path is not yet wired.
pub fn production_ninep_reference_monitor(
    decider: Arc<dyn AccessDecider>,
    authenticator: Arc<dyn AttachAuthenticator>,
) -> Arc<ReferenceMonitor> {
    let resolver = crate::mac::GenesisGate::production().into_resolver();
    Arc::new(ReferenceMonitor::new(
        authenticator,
        Arc::new(resolver),
        decider,
    ))
}

/// Assemble the production reference monitor in its current fail-closed state.
///
/// The raw 9P attach fields are not credential proof. This helper therefore
/// refuses to derive an identity or tenant from them; a later S6 integration
/// must replace the authenticator with one backed by verified attach material.
pub fn enrollment_ninep_reference_monitor(
    decider: Arc<dyn AccessDecider>,
) -> Arc<ReferenceMonitor> {
    // No production 9P transport currently passes a verified, sender-bound S6
    // credential into the monitor. Raw `Tattach.uname` is untrusted, so the
    // only safe activation state is an explicit deny-only authenticator.
    production_ninep_reference_monitor(decider, Arc::new(AnonymousAuthenticator))
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

    fn audit_denial(
        &self,
        ctx: &SecurityContext,
        _object: ObjectRef<'_>,
        object_label: Option<SecurityLabel>,
        action: NinePAction,
        reason: ReferenceMonitorDenyReason,
    ) {
        let reason = match reason {
            ReferenceMonitorDenyReason::UnlabeledObject => DecisionReason::UnlabeledObject,
            ReferenceMonitorDenyReason::TokenGate => DecisionReason::TokenGate,
            ReferenceMonitorDenyReason::FloorDeny => DecisionReason::FloorDeny,
        };
        let _ = self.audit(ctx, object_label, action, Decision::Deny, reason);
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

// ─────────────────────────────────────────────────────────────────────────────
// VFS-plane PEP — the direct `Namespace` API reference monitor (#1272).
//
// Mirrors `NinePAccessDecider` for the *in-process* direct API: the 9P
// translator decider authorizes wire ops; this decider authorizes the
// `Namespace::cat`/`echo`/`create`/`ls`/`ctl` convenience methods (and the
// `mount`/`bind_mount`/`unmount` mutation surface) once a `NamespacePep` is
// installed. It reuses the SAME `ObjectLabelResolver` (the genesis composite),
// the SAME `AuditSink` (the WAL), and the SAME `can_access` floor — it does
// not reinvent any MAC primitive, only adapts the action surface.
// ─────────────────────────────────────────────────────────────────────────────

use hyprstream_vfs::{NamespaceAccessDecider, NamespaceAction, SubjectContextResolver};

/// Reserved audit subject/object type ids for VFS-PEP decisions (distinct from
/// the 9P sentinels above so audit consumers can distinguish the plane).
const VFS_SUBJECT_TYPE: SubjectType = SubjectType(u32::MAX - 2);
const VFS_OBJECT_TYPE: ObjectType = ObjectType(u32::MAX - 2);

/// Audited `NamespaceAccessDecider` for the direct VFS API (#1272).
///
/// This is the production impl of [`hyprstream_vfs::NamespaceAccessDecider`]:
/// it applies the intrinsic lattice dominance rule for read-class operations,
/// denies write-class pending the VFS IFC write-direction decision, and records
/// every outcome through the MAC audit sink. It
/// receives the attempted subject context and label resolution from
/// [`hyprstream_vfs::NamespacePep`], including `None` for missing clearance or
/// labels, so fail-closed precondition denials cannot bypass the audit WAL.
pub struct VfsAccessDecider {
    sink: Arc<dyn AuditSink>,
}

impl VfsAccessDecider {
    /// Wrap the process-wide MAC audit sink. The sink is mandatory; a decision
    /// that cannot be durably audited is downgraded to Deny (fail-closed),
    /// exactly as the 9P decider does.
    pub fn new(sink: Arc<dyn AuditSink>) -> Self {
        Self { sink }
    }

    fn audit(
        &self,
        ctx: Option<&SecurityContext>,
        label: Option<SecurityLabel>,
        action: NamespaceAction,
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
            subject_type: VFS_SUBJECT_TYPE,
            // Bottom is an audit-schema placeholder when clearance could not
            // be proven. `NoClearance` is authoritative and the placeholder
            // never enters authorization.
            subject_clearance: ctx.map_or_else(SecurityLabel::bottom, |ctx| *ctx.clearance()),
            on_behalf_of: None,
            object_type: VFS_OBJECT_TYPE,
            // Likewise, an unresolved object is represented only in the audit
            // schema; `UnlabeledObject` remains the authoritative reason.
            object_label: label.unwrap_or_else(SecurityLabel::bottom),
            action: vfs_audit_action(action),
            reason,
            subject_id: None,
            object_id: None,
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
                    "VFS PEP decision could not be durably audited; enforcing deny"
                );
                false
            }
        }
    }
}

const fn vfs_audit_action(action: NamespaceAction) -> Action {
    match action {
        NamespaceAction::Read => Action::from_scope_action(ScopeAction::Query),
        // Write/create + namespace mutation (mount/bind/unmount) and raw-handle
        // extraction are all write-capable operations over an object — audited
        // as Write. The deny-on-write-direction pause applies to all of them
        // until IFC lands.
        NamespaceAction::Write
        | NamespaceAction::Create
        | NamespaceAction::Mount
        | NamespaceAction::BindMount
        | NamespaceAction::Unmount
        | NamespaceAction::ResolveHandle => Action::from_scope_action(ScopeAction::Write),
    }
}

impl NamespaceAccessDecider for VfsAccessDecider {
    fn check(
        &self,
        ctx: Option<&SecurityContext>,
        object_label: Option<SecurityLabel>,
        action: NamespaceAction,
    ) -> MacDecision {
        let Some(ctx) = ctx else {
            self.audit(
                None,
                object_label,
                action,
                Decision::Deny,
                DecisionReason::NoClearance,
            );
            return MacDecision::Deny(MacDenyReason::NoClearance);
        };
        let Some(object_label) = object_label else {
            self.audit(
                Some(ctx),
                None,
                action,
                Decision::Deny,
                DecisionReason::UnlabeledObject,
            );
            return MacDecision::Deny(MacDenyReason::UnlabeledObject);
        };

        // Write/create/mutate deny pending the VFS IFC write-direction
        // decision.
        let is_write_class = matches!(
            action,
            NamespaceAction::Write
                | NamespaceAction::Create
                | NamespaceAction::Mount
                | NamespaceAction::BindMount
                | NamespaceAction::Unmount
                | NamespaceAction::ResolveHandle
        );
        if is_write_class {
            self.audit(
                Some(ctx),
                Some(object_label),
                action,
                Decision::Deny,
                DecisionReason::WriteDirectionUndecided,
            );
            return MacDecision::Deny(MacDenyReason::FloorDeny);
        }

        let permitted = ctx.can_access(&object_label);
        let audited_permit = self.audit(
            Some(ctx),
            Some(object_label),
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
        );
        if permitted && audited_permit {
            MacDecision::Permit
        } else {
            MacDecision::Deny(MacDenyReason::FloorDeny)
        }
    }
}

/// Fail-closed `SubjectContextResolver` for the VFS PEP (#698 dependency
/// window).
///
/// The direct `Namespace` API carries only an unauthenticated `Subject` string
/// — it has no verified `EnvelopeContext`/claims the way the RPC dispatch
/// plane does (#1268 owns threading those through). Until production clearance
/// provenance (#698) wires a real resolver (e.g. one keyed on the installed
/// `CompiledPolicy` enrollment table, mirroring
/// [`EnrollmentSubjectContextResolver`]), this stub resolves **no** subject,
/// so an armed VFS PEP denies every op — the correct fail-closed posture
/// (#547: no permissive default).
///
/// Replacing this with a real resolver is the activation B-lane (#1267), not
/// this struct's existence.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyUnenrolledSubjects;

impl SubjectContextResolver for DenyUnenrolledSubjects {
    fn resolve(
        &self,
        _subject: &hyprstream_rpc::Subject,
    ) -> Option<SecurityContext> {
        None
    }
}

/// Assemble the production VFS-plane PEP from the genesis label resolver and
/// the tamper-evident MAC audit sink (#1272).
///
/// This is the `hyprstream`-crate wiring that satisfies the
/// [`hyprstream_vfs::NamespacePep`] contract: the genesis composite
/// `RpcObjectLabelResolver` (carriers (a)+(c), #1228) feeds label resolution, a
/// [`VfsAccessDecider`] over the WAL audit sink evaluates the policy, and —
/// pending #698 — the subject seam is [`DenyUnenrolledSubjects`] (fail-closed:
/// every op denies until clearance provenance is wired). Swapping the subject
/// resolver for a real one is the activation B-lane (#1267).
///
/// Callers propagate the error and refuse to arm a `Namespace` with a
/// permissive fallback; there is no permissive arm path.
pub async fn production_vfs_pep(
    signing_key: SigningKey,
    oauth: &crate::config::OAuthConfig,
    audit_stream: &str,
) -> anyhow::Result<std::sync::Arc<hyprstream_vfs::NamespacePep>> {
    anyhow::ensure!(
        !audit_stream.is_empty()
            && audit_stream
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'),
        "invalid VFS MAC audit stream name"
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
        "VFS MAC PEP audit signer unavailable under mandatory Hybrid policy"
    );

    let audit_store = crate::mac::audit::WalAuditStore::open(
        secrets_dir.join("mac-audit").join(audit_stream),
        signer,
    )
    .context("open VFS MAC audit store")?;
    let resolver = crate::mac::GenesisGate::production().into_resolver();

    Ok(std::sync::Arc::new(hyprstream_vfs::NamespacePep::new(
        std::sync::Arc::new(DenyUnenrolledSubjects),
        std::sync::Arc::new(resolver),
        std::sync::Arc::new(VfsAccessDecider::new(std::sync::Arc::new(audit_store))),
    )))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

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

    struct MonitorResolver {
        public: SecurityLabel,
        secret: SecurityLabel,
    }

    impl ObjectLabelResolver for MonitorResolver {
        fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel> {
            match object {
                ObjectRef::Path(["public"] | ["decider-deny"]) => Some(self.public),
                ObjectRef::Path(["secret"]) => Some(self.secret),
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

    struct FixtureVfsSubjects {
        name: &'static str,
        ctx: SecurityContext,
    }

    impl SubjectContextResolver for FixtureVfsSubjects {
        fn resolve(&self, subject: &hyprstream_rpc::Subject) -> Option<SecurityContext> {
            (subject.name() == Some(self.name)).then(|| self.ctx.clone())
        }
    }

    struct FixtureVfsLabels {
        public: SecurityLabel,
        secret: SecurityLabel,
    }

    impl hyprstream_rpc::auth::mac::RpcObjectLabelResolver for FixtureVfsLabels {
        fn resolve(&self, service_domain: &str, _method: Option<u16>) -> Option<SecurityLabel> {
            match service_domain {
                "/public" => Some(self.public),
                "/secret" => Some(self.secret),
                _ => None,
            }
        }
    }

    fn label(level: Level) -> SecurityLabel {
        SecurityLabel::new(level, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn context(level: Level) -> SecurityContext {
        SecurityContext::from_clearance(label(level), VerifiedKeyMaterial::Classical)
    }

    fn identity(tenant: &str) -> VerifiedAttachIdentity {
        VerifiedAttachIdentity::from_verified_credential("did:key:alice", tenant)
    }

    fn read_token() -> hyprstream_9p::VerifiedTokenScope {
        hyprstream_9p::VerifiedTokenScope::from_verified_token(
            label(Level::Secret),
            Arc::from([NinePAction::Read]),
            Instant::now() + Duration::from_secs(60),
        )
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

    #[test]
    fn every_reference_monitor_deny_stage_is_audited() {
        let sink = Arc::new(SpySink::default());
        let decider: Arc<dyn AccessDecider> = Arc::new(NinePAccessDecider::new(
            Arc::new(FixtureResolver {
                public: label(Level::Public),
                secret: label(Level::Secret),
            }),
            sink.clone(),
        ));
        let monitor = ReferenceMonitor::new(
            Arc::new(AnonymousAuthenticator),
            Arc::new(MonitorResolver {
                public: label(Level::Public),
                secret: label(Level::Secret),
            }),
            decider,
        );
        let permitted = SessionContext::from_verified_token(
            identity("tenant-a"),
            context(Level::Secret),
            read_token(),
        );

        assert!(!monitor.authorize(
            &permitted,
            &["unlabeled".to_owned()],
            NinePAction::Read,
        ));
        assert!(!monitor.authorize(
            &SessionContext::from_verified_clearance(
                identity("tenant-a"),
                context(Level::Secret),
            ),
            &["public".to_owned()],
            NinePAction::Read,
        ));
        assert!(!monitor.authorize(
            &SessionContext::from_verified_token(
                identity("tenant-a"),
                context(Level::Public),
                read_token(),
            ),
            &["secret".to_owned()],
            NinePAction::Read,
        ));
        assert!(!monitor.authorize(
            &permitted,
            &["decider-deny".to_owned()],
            NinePAction::Read,
        ));

        let records = sink.records.lock();
        assert_eq!(records.len(), 4, "one WAL record per denied operation");
        assert_eq!(records[0].reason, DecisionReason::UnlabeledObject);
        assert_eq!(records[1].reason, DecisionReason::TokenGate);
        assert_eq!(records[2].reason, DecisionReason::FloorDeny);
        assert_eq!(records[3].reason, DecisionReason::UnlabeledObject);
        assert!(records
            .iter()
            .all(|record| record.decision == Decision::Deny));
    }

    #[tokio::test]
    async fn raw_attach_uname_never_becomes_verified_identity_or_tenant() {
        let sink = Arc::new(SpySink::default());
        let decider: Arc<dyn AccessDecider> = Arc::new(NinePAccessDecider::new(
            Arc::new(FixtureResolver {
                public: label(Level::Public),
                secret: label(Level::Secret),
            }),
            sink,
        ));
        let monitor = enrollment_ninep_reference_monitor(decider);

        let session = monitor
            .authenticate("did:key:victim", "tenant-victim")
            .await;
        assert!(
            session.verified_attach_identity().is_none(),
            "unverified Tattach fields must not mint identity or tenant"
        );
        assert!(!session.token_authorizes(&label(Level::Public), NinePAction::Read));
    }

    #[test]
    fn vfs_fail_closed_denies_are_audited() {
        let sink = Arc::new(SpySink::default());
        let pep = hyprstream_vfs::NamespacePep::new(
            Arc::new(FixtureVfsSubjects {
                name: "alice",
                ctx: context(Level::Public),
            }),
            Arc::new(FixtureVfsLabels {
                public: label(Level::Public),
                secret: label(Level::Secret),
            }),
            Arc::new(VfsAccessDecider::new(sink.clone())),
        );

        // Missing verified clearance, missing content label, lattice-floor
        // denial, and the write-direction pause must all cross the same audit
        // sink before the VFS reference monitor returns Deny.
        assert_eq!(
            pep.check(
                &hyprstream_rpc::Subject::new("mallory"),
                "/public",
                NamespaceAction::Read,
            ),
            MacDecision::Deny(MacDenyReason::NoClearance)
        );
        assert_eq!(
            pep.check(
                &hyprstream_rpc::Subject::new("alice"),
                "/missing",
                NamespaceAction::Read,
            ),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );
        assert_eq!(
            pep.check(
                &hyprstream_rpc::Subject::new("alice"),
                "/secret",
                NamespaceAction::Read,
            ),
            MacDecision::Deny(MacDenyReason::FloorDeny)
        );
        assert_eq!(
            pep.check(
                &hyprstream_rpc::Subject::new("alice"),
                "/public",
                NamespaceAction::Write,
            ),
            MacDecision::Deny(MacDenyReason::FloorDeny)
        );

        let records = sink.records.lock();
        assert_eq!(records.len(), 4);
        assert!(records.iter().all(|record| record.decision == Decision::Deny));
        assert_eq!(records[0].reason, DecisionReason::NoClearance);
        assert_eq!(records[1].reason, DecisionReason::UnlabeledObject);
        assert_eq!(records[2].reason, DecisionReason::FloorDeny);
        assert_eq!(records[3].reason, DecisionReason::WriteDirectionUndecided);
    }
}
