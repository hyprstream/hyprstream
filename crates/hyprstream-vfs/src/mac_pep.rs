//! **MAC PEP for the direct Namespace API** (#1272, epic #1267 T3).
//!
//! The 9P *translator* already mediates every wire op through
//! `hyprstream_9p::AccessDecider` (implemented by `hyprstream::mac::pep`).
//! But the **direct in-process `Namespace` API** — `cat`/`echo`/`create` and
//! `mount`/`bind_mount`/`unmount` — forwards a plain [`Subject`] to the
//! resolving [`Mount`](crate::Mount) with no mandatory label lookup or
//! reference monitor, bypassing the translator PEP entirely. This module is
//! that missing reference monitor: the contract the `Namespace` invokes per
//! op, plus its fail-closed defaults.
//!
//! ## Dependency-direction (mirrors the 9P split)
//!
//! The *trait* (the contract the low-level `Namespace` consumes) lives here in
//! `hyprstream-vfs`, exactly as `hyprstream_9p::AccessDecider` lives in the 9P
//! crate. The *production audited implementation* — bridging to the real
//! `RpcObjectLabelResolver`, `SecurityContext::can_access`, and the
//! tamper-evident MAC audit sink — lives in `hyprstream::mac::pep`, alongside
//! `NinePAccessDecider`. This keeps the dependency direction one-way
//! (`hyprstream` → `hyprstream-vfs`) and **does not reinvent label resolution,
//! clearance, or `can_access`**; those are S1/S4 primitives re-exported from
//! `hyprstream_rpc::auth::mac`.
//!
//! ## Fail-closed by construction; no permissive mode (#547)
//!
//! Every seam has an explicit deny-all default. A [`NamespacePep`] authorizes
//! an op only when ALL of:
//! 1. the caller's [`Subject`] resolves to a verified [`SecurityContext`]
//!    (clearance provenance — #698; `None` ⇒ deny),
//! 2. the walked object resolves to a [`SecurityLabel`] via the injected
//!    [`RpcObjectLabelResolver`] (`None` ⇒ deny — "no unlabeled-default-allow"),
//! 3. `clearance.can_access(object_label)` (BLP dominance) AND the
//!    [`NamespaceAccessDecider`] permit.
//! There is no permissive path *inside* the PEP.
//!
//! ## Activation posture (dormant structure)
//!
//! A [`Namespace`](crate::Namespace) built with [`Namespace::new`] carries
//! `pep: None` and behaves exactly as before — the un-enforced status quo,
//! matching the rest of MAC enforcement being dormant today (CLAUDE.md "MAC
//! current status"). Installing a PEP via
//! [`Namespace::with_pep`](crate::Namespace::with_pep) arms the fail-closed
//! monitor: every mediated op then requires a proven subject, and the
//! subject-less `mount`/`bind_mount`/`unmount` deny (use the `_as` variants).
//! Flipping the default at construction sites is the separately-gated
//! activation B-lane (#1267), not this PR.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
pub use hyprstream_rpc::auth::mac::{
    MacDecision, MacDenyReason, RpcObjectLabelResolver, SecurityContext, SecurityLabel,
};
#[cfg(not(target_arch = "wasm32"))]
use hyprstream_rpc::Subject;

#[cfg(not(target_arch = "wasm32"))]
use crate::NamespaceError;

/// The direct-`Namespace`-API operation a [`NamespaceAccessDecider`] is asked
/// to authorize.
///
/// This is the VFS-plane action surface — broader than the 9P translator's
/// `hyprstream_9p::Action` because the direct API also mutates the namespace
/// itself (`Mount`/`BindMount`/`Unmount`), ops the wire translator never
/// dispatches. Mirrors [`Namespace`](crate::Namespace)'s convenience methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamespaceAction {
    /// `cat` / `read_one` / `ls` — read-class.
    Read,
    /// `echo` / `ctl` — write-class.
    Write,
    /// `create` — new-object creation.
    Create,
    /// `mount` — install/replace a mount point (namespace mutation).
    Mount,
    /// `bind_mount` — union-bind a target (namespace mutation).
    BindMount,
    /// `unmount` — remove a mount point (namespace mutation).
    Unmount,
    /// Extract raw mount targets from the namespace.
    ///
    /// A returned [`crate::MountTarget`] can invoke the complete `Mount`
    /// surface, including writes, without re-entering [`crate::Namespace`].
    /// This is therefore a privileged, write-capable action rather than a
    /// read-only path lookup.
    ResolveHandle,
}

/// Plane-specific clearance-input seam: resolve a direct-API [`Subject`] to the
/// verified [`SecurityContext`] that authorizes it.
///
/// This is the VFS-plane half of **#698** (production clearance provenance).
/// The companion RPC-dispatch seam (claims → `SecurityContext` threading at
/// dispatch) is owned by **#1268**, which will publish its interface to
/// `.fleet-coord/mac-pep-contract.md`. Until that lands, this trait is the
/// plane-specific resolution point: a `Subject` (the only identity the direct
/// API carries — no verified `EnvelopeContext` crosses it) becomes a clearance
/// only through verified credential material this resolver consults.
///
/// **Returning `None` is the mandatory fail-closed contract** — a subject
/// whose clearance cannot be established from verified credential material
/// MUST NOT be authorized; there is no permissive default (#547).
/// Implementations MUST NOT derive clearance from an unverified,
/// caller-supplied label (D1 — labels in wire schemas are hints; here the
/// `Subject` name is even weaker: an unauthenticated string).
///
/// Async because the production resolver revalidates credential-bearing
/// cached subject contexts against the canonical revocation authority on
/// every read.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
pub trait SubjectContextResolver: Send + Sync {
    /// Resolve `subject` to its verified security context, or `None` if its
    /// clearance cannot be proven — which the PEP treats as deny.
    async fn resolve(&self, subject: &Subject) -> Option<SecurityContext>;
}

/// Fail-closed resolver: no subject resolves to a clearance.
///
/// This is the default a [`NamespacePep`] uses for the subject seam when no
/// production resolver (#698) has been wired — every mediated op denies.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllSubjects;

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
impl SubjectContextResolver for DenyAllSubjects {
    async fn resolve(&self, _subject: &Subject) -> Option<SecurityContext> {
        // No SubjectContextResolver wired (#698 not landed). The authorize()
        // path passes `None` to the authoritative decider, which audits the
        // no-clearance denial before returning it. This low-level crate does
        // not own the audit WAL or a tracing dependency.
        None
    }
}

/// Authorizes one direct-`Namespace` operation against the attempted subject
/// context and object label.
///
/// Label resolution is centralized in [`NamespacePep::authorize`] (the
/// reference monitor), which calls the injected [`RpcObjectLabelResolver`] and
/// hands the result here. Missing clearance and missing labels are deliberately
/// represented as `None` so the authoritative implementation in the
/// `hyprstream` crate can write those fail-closed decisions to the same
/// tamper-evident WAL as policy-floor denials. No deny may return before this
/// audit boundary.
#[cfg(not(target_arch = "wasm32"))]
pub trait NamespaceAccessDecider: Send + Sync {
    /// Return the canonical MAC decision and record every outcome through the
    /// MAC audit sink.
    ///
    /// `None` inputs are mandatory denials: an absent `ctx` means clearance
    /// could not be proven, while an absent `object_label` means the object is
    /// unlabeled. Implementations use bottom labels only as audit-schema
    /// placeholders; placeholders must never participate in authorization.
    fn check(
        &self,
        ctx: Option<&SecurityContext>,
        object_label: Option<SecurityLabel>,
        action: NamespaceAction,
    ) -> MacDecision;
}

/// Fail-closed decider for a [`NamespacePep`] whose production audited decider
/// is unavailable. Every op denies.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllNamespace;

#[cfg(not(target_arch = "wasm32"))]
impl NamespaceAccessDecider for DenyAllNamespace {
    fn check(
        &self,
        _ctx: Option<&SecurityContext>,
        _object_label: Option<SecurityLabel>,
        _action: NamespaceAction,
    ) -> MacDecision {
        MacDecision::Deny(MacDenyReason::NoPepInstalled)
    }
}

/// Fail-closed label resolver: no object resolves. Mirrors
/// `hyprstream_9p::DenyUnlabeledResolver`. Returning `None` is a mandatory
/// deny (design §1 invariant 2) — the PEP refuses to authorize any op whose
/// object has no trusted label.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyUnlabeledResolver;

#[cfg(not(target_arch = "wasm32"))]
impl RpcObjectLabelResolver for DenyUnlabeledResolver {
    fn resolve(&self, _service_domain: &str, _method: Option<u16>) -> Option<SecurityLabel> {
        None
    }
}

/// The mandatory, all-or-nothing reference monitor a
/// [`Namespace`](crate::Namespace) mediates every direct-API op through once
/// installed.
///
/// There is no permissive or partial construction: each of the three seams is
/// required, and the crate's fail-closed defaults ([`DenyAllSubjects`],
/// [`DenyAllNamespace`], and the `None`-returning
/// [`hyprstream_rpc::auth::mac`] `RpcObjectLabelResolver` the `hyprstream` crate
/// supplies) are the explicit choices for a seam the application does not yet
/// populate. Building a `NamespacePep` is therefore always fail-closed;
/// "unenforced" is represented by *not installing* one on the `Namespace`
/// (the dormant status quo), never by a permissive PEP.
#[cfg(not(target_arch = "wasm32"))]
pub struct NamespacePep {
    subjects: Arc<dyn SubjectContextResolver>,
    labels: Arc<dyn RpcObjectLabelResolver>,
    decider: Arc<dyn NamespaceAccessDecider>,
}

#[cfg(not(target_arch = "wasm32"))]
impl std::fmt::Debug for NamespacePep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamespacePep").finish_non_exhaustive()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl NamespacePep {
    /// Assemble the monitor from its three mandatory seams.
    ///
    /// Callers that lack a production seam pass the crate's deny-all default
    /// for it — the resulting PEP denies every op, which is the correct
    /// posture for "structure present, inputs not yet wired" (the #698 /
    /// #1268 dependency window). There is no `new_permissive`.
    pub fn new(
        subjects: Arc<dyn SubjectContextResolver>,
        labels: Arc<dyn RpcObjectLabelResolver>,
        decider: Arc<dyn NamespaceAccessDecider>,
    ) -> Self {
        Self {
            subjects,
            labels,
            decider,
        }
    }

    /// Authorize `action` by `subject` against `object`.
    ///
    /// Fail-closed sequence (any deny ⇒ [`NamespaceError::Denied`]):
    ///   1. resolve `subject` → optional `SecurityContext`,
    ///   2. resolve `object` → `SecurityLabel` via the injected
    ///      [`RpcObjectLabelResolver`] (missing means no unlabeled-default-allow;
    ///      design §1 invariant 2),
    ///   3. `decider.check(ctx, label, action)` — audit every outcome, including
    ///      missing-clearance and missing-label denials, then apply
    ///      `can_access` + TE floor; canonical [`MacDecision::Deny`] ⇒ deny,
    ///   4. permit.
    ///
    /// `object_path` is the canonical normalized VFS path. It occupies the
    /// canonical resolver's `service_domain` slot; VFS has no browser method
    /// discriminator, so `method` is always `None`. The caller never supplies a
    /// label (D1 — caller labels are forbidden).
    pub async fn authorize(
        &self,
        subject: &Subject,
        object_path: &str,
        action: NamespaceAction,
    ) -> Result<(), NamespaceError> {
        match self.check(subject, object_path, action).await {
            MacDecision::Permit => Ok(()),
            MacDecision::Deny(reason) => Err(NamespaceError::Denied(
                action,
                denial_detail(subject, deny_reason_detail(reason)),
            )),
        }
    }

    /// Evaluate one VFS operation using the canonical shared MAC decision
    /// contract. An installed `NamespacePep` is always fail-closed.
    pub async fn check(
        &self,
        subject: &Subject,
        object_path: &str,
        action: NamespaceAction,
    ) -> MacDecision {
        let ctx = self.subjects.resolve(subject).await;
        let label = self.labels.resolve(object_path, None);
        self.decider.check(ctx.as_ref(), label, action)
    }

    /// Record and return the mandatory no-clearance denial for a subject-less
    /// namespace mutation.
    ///
    /// Armed construction APIs have no caller to resolve, but their forced
    /// denial must still cross the same audit boundary as every subjectful
    /// operation. The decider contract requires `ctx=None` to deny and record
    /// the attempt; a non-conforming permit is nevertheless downgraded to
    /// `NoClearance`.
    pub(crate) fn deny_uncredentialed(
        &self,
        object_path: &str,
        action: NamespaceAction,
    ) -> NamespaceError {
        let label = self.labels.resolve(object_path, None);
        let reason = match self.decider.check(None, label, action) {
            MacDecision::Deny(reason) => reason,
            MacDecision::Permit => MacDenyReason::NoClearance,
        };
        NamespaceError::Denied(
            action,
            denial_detail(&Subject::anonymous(), deny_reason_detail(reason)),
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn denial_detail(subject: &Subject, reason: &str) -> String {
    match subject.name() {
        Some(n) => format!("'{n}': {reason}"),
        None => format!("<anonymous>: {reason}"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
const fn deny_reason_detail(reason: MacDenyReason) -> &'static str {
    match reason {
        MacDenyReason::NoPepInstalled => "explicit deny-all PEP installed",
        MacDenyReason::NoClearance => "subject clearance unprovable",
        MacDenyReason::UnlabeledObject => "object unlabeled",
        MacDenyReason::FloorDeny => "MAC floor denied",
        MacDenyReason::StaleAuthority => "stale authority",
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use hyprstream_rpc::auth::mac::{
        Assurance, CompartmentSet, Level, SecurityLabel, VerifiedKeyMaterial,
    };

    fn ctx(level: Level) -> SecurityContext {
        SecurityContext::from_clearance(label(level), VerifiedKeyMaterial::Classical)
    }
    fn label(level: Level) -> SecurityLabel {
        SecurityLabel::new(level, Assurance::Classical, CompartmentSet::EMPTY)
    }
    fn subject(name: &str) -> Subject {
        Subject::new(name)
    }

    /// A resolver that maps one subject name to a fixed clearance; all others
    /// `None` (deny).
    struct OneSubject {
        name: &'static str,
        ctx: SecurityContext,
    }
    #[async_trait::async_trait]
    impl SubjectContextResolver for OneSubject {
        async fn resolve(&self, s: &Subject) -> Option<SecurityContext> {
            (s.name() == Some(self.name)).then(|| self.ctx.clone())
        }
    }

    /// A decider that permits reads at-or-below the subject's level and denies
    /// everything else (write-class = deny, mirroring the IFC write-direction
    /// pause the 9P PEP enforces).
    struct ReadFloorDecider;
    impl NamespaceAccessDecider for ReadFloorDecider {
        fn check(
            &self,
            ctx: Option<&SecurityContext>,
            object_label: Option<SecurityLabel>,
            action: NamespaceAction,
        ) -> MacDecision {
            let Some(ctx) = ctx else {
                return MacDecision::Deny(MacDenyReason::NoClearance);
            };
            let Some(object_label) = object_label else {
                return MacDecision::Deny(MacDenyReason::UnlabeledObject);
            };
            if action != NamespaceAction::Read {
                return MacDecision::Deny(MacDenyReason::FloorDeny);
            }
            if ctx.clearance().can_access(&object_label) {
                MacDecision::Permit
            } else {
                MacDecision::Deny(MacDenyReason::FloorDeny)
            }
        }
    }

    /// A resolver that labels the first path component as a level (test-only).
    struct PathLevelResolver;
    impl RpcObjectLabelResolver for PathLevelResolver {
        fn resolve(&self, service_domain: &str, _method: Option<u16>) -> Option<SecurityLabel> {
            match service_domain {
                "/public" => Some(label(Level::Public)),
                "/secret" => Some(label(Level::Secret)),
                _ => None,
            }
        }
    }

    #[tokio::test]
    async fn deny_all_pep_denies_every_subject() {
        // All three seams deny: subject unresolvable ⇒ DenyAllSubjects ⇒ None
        // ⇒ the PEP returns Err before the decider is even consulted.
        let pep = NamespacePep::new(
            Arc::new(DenyAllSubjects),
            Arc::new(DenyUnlabeledResolver),
            Arc::new(DenyAllNamespace),
        );
        let err = pep
            .authorize(&subject("anyone"), "/x", NamespaceAction::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, NamespaceError::Denied(_, _)));
    }

    #[tokio::test]
    async fn deny_unlabeled_object_denies_even_with_a_valid_subject() {
        // Subject resolves, but the object is unlabeled ⇒ deny (invariant 2).
        let pep = NamespacePep::new(
            Arc::new(OneSubject {
                name: "alice",
                ctx: ctx(Level::Secret),
            }),
            Arc::new(DenyUnlabeledResolver),
            Arc::new(ReadFloorDecider),
        );
        assert!(pep
            .authorize(&subject("alice"), "/secret", NamespaceAction::Read)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn pep_permits_read_when_clearance_dominates_and_denies_otherwise() {
        let pep = NamespacePep::new(
            Arc::new(OneSubject {
                name: "alice",
                ctx: ctx(Level::Secret),
            }),
            Arc::new(PathLevelResolver),
            Arc::new(ReadFloorDecider),
        );

        // alice (Secret) reading /secret ⇒ permit.
        assert!(pep
            .authorize(&subject("alice"), "/secret", NamespaceAction::Read)
            .await
            .is_ok());
        // Write ⇒ deny (write-direction pause).
        assert!(pep
            .authorize(&subject("alice"), "/public", NamespaceAction::Write)
            .await
            .is_err());
        // Unenrolled subject ⇒ deny (fail-closed clearance).
        assert!(pep
            .authorize(&subject("mallory"), "/public", NamespaceAction::Read)
            .await
            .is_err());
    }
}
