//! **The mandatory RPC dispatch PEP** (epic #1267 T3, #1268).
//!
//! This is the single mandatory, unavoidable MAC gate between claims
//! verification and handler invocation in [`crate::service::dispatch::
//! process_request`]. It is the RPC-plane analogue of the 9P
//! `NinePAccessDecider` (`hyprstream/src/mac/pep.rs`): both resolve a
//! trusted object label and check `clearance.can_access(label)` before
//! the data reaches the handler.
//!
//! ## Why this exists
//!
//! `process_request` verifies the caller's identity and JWT claims, then
//! invokes `handle_request` with **no object-label resolution, no lattice
//! floor check, and no `can_access`** (issue #1268). Method-level Casbin
//! (`#[authorize]`) is discretionary — a handler that forgets the attribute
//! is unmediated. This PEP is the **mandatory floor** no handler can bypass.
//!
//! ## The three-step check (per design §10, S1 interface contract)
//!
//! ```text
//! 1. Subject:   ctx.security_context()  →  Option<SecurityContext>
//!                                         None ⇒ Deny(NoClearance)
//! 2. Object:    resolver.resolve(svc, method)  →  Option<SecurityLabel>
//!                                                 None ⇒ Deny(UnlabeledObject)
//! 3. Decision:  ctx.can_access(label)   →  bool
//!                                         false ⇒ Deny(FloorDeny)
//! ```
//!
//! Once a PEP is installed, every `None` or `false` produced by that PEP is a
//! hard deny. There is **no permissive mode inside an active PEP** (epic #547
//! invariant: "no unlabeled-default-allow").
//!
//! ## Clearance provenance (#698 dependency)
//!
//! `EnvelopeContext::security_context()` already composes `Claims ×
//! VerifiedKeyMaterial` via the S1 `SubjectContextClaims` trait — this IS the
//! clearance seam. Until #698 wires the `clearance` field onto `Claims` in
//! production, `security_context()` resolves to `None` and the PEP denies
//! every request — which is correct fail-closed behavior, not a bug.
//!
//! ## Process-global installation
//!
//! Like the PQ trust store and compiled MAC policy, the PEP is installed at
//! startup via [`install_mac_dispatch_pep`]. Until installed, dispatch denies
//! with [`MacDenyReason::NoPepInstalled`]. Once installed, every request is
//! mediated and every installed-PEP denial is authoritative.

use std::sync::Arc;

use super::label::SecurityLabel;
use crate::service::EnvelopeContext;

// ── Decision types ────────────────────────────────────────────────────────

/// The MAC decision returned by the dispatch PEP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacDecision {
    /// Access permitted — proceed to handler.
    Permit,
    /// Access denied — handler MUST NOT be called.
    Deny(MacDenyReason),
}

impl MacDecision {
    /// Returns `true` if the decision is `Permit`.
    #[inline]
    #[must_use]
    pub const fn is_permit(self) -> bool {
        matches!(self, Self::Permit)
    }
}

/// Why a MAC decision denied. Auditable and testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacDenyReason {
    /// The process-global PEP has not been installed, or an operator
    /// deliberately installed [`DenyAllMacPep`] as a fail-closed sentinel.
    NoPepInstalled,
    /// Subject has no derivable clearance (unlabeled subject).
    ///
    /// The verified `Claims` carry no `clearance` field, or no `Claims` are
    /// present. This is the #698 dependency: until production clearance is
    /// wired, this is the expected deny reason for every request.
    NoClearance,
    /// Object has no trusted label (unlabeled object).
    ///
    /// The `RpcObjectLabelResolver` returned `None` for this service+method.
    /// Per D2/D3, objects deny/clamp — never default-allow.
    UnlabeledObject,
    /// Clearance does not dominate the object label (lattice floor deny).
    ///
    /// The subject's clearance exists and the object label exists, but
    /// `SecurityContext::can_access` returned `false`.
    FloorDeny,
    /// Stale continuation authority (streaming re-check failed).
    ///
    /// A streaming continuation was re-checked and the subject's authority
    /// has been revoked or the policy generation has rolled.
    StaleAuthority,
}

// ── Object-label resolution seam ──────────────────────────────────────────

/// Resolve the trusted [`SecurityLabel`] for the concrete object a
/// dispatching RPC acts on.
///
/// `None` ⇒ unlabeled ⇒ the PEP **denies** (D2/D3 — objects deny/clamp,
/// never default-allow). Implementors must NOT manufacture a permissive
/// default.
///
/// `service_domain` is the canonical service name ("model", "registry", …).
/// `method` is the browser method discriminator if available (a `u16`
/// committed in the sealed transcript), else `None` for non-browser RPC.
///
/// Production implementations will map static schema nodes (S3, #569) and
/// content-addressed manifests to labels. The default implementation
/// ([`DenyAllObjectResolver`]) returns `None` for everything — fail-closed.
pub trait RpcObjectLabelResolver: Send + Sync {
    fn resolve(&self, service_domain: &str, method: Option<u16>) -> Option<SecurityLabel>;
}

/// Fail-closed object resolver: every object is unlabeled ⇒ deny.
///
/// This is the default when no production resolver has been wired. It makes
/// the "no permissive default" invariant structural: a node that has not
/// installed a resolver denies every request, regardless of clearance.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllObjectResolver;

impl RpcObjectLabelResolver for DenyAllObjectResolver {
    #[inline]
    fn resolve(&self, _service_domain: &str, _method: Option<u16>) -> Option<SecurityLabel> {
        None
    }
}

// ── The mandatory PEP trait ───────────────────────────────────────────────

/// The mandatory, unavoidable RPC dispatch PEP for native MAC (epic #547,
/// activation-gate T3, #1268).
///
/// Called by [`process_request`](crate::service::dispatch::process_request)
/// after `verify_claims` and **before** `handle_request`. If the PEP returns
/// [`MacDecision::Deny`], the handler is never invoked — the error payload
/// is signed and returned to the caller.
///
/// If no PEP is installed process-globally, the dispatch wrapper denies with
/// [`MacDenyReason::NoPepInstalled`]. Once installed, this check remains
/// mandatory and cannot be bypassed.
pub trait MacDispatchPep: Send + Sync {
    /// Check mandatory MAC access before handler dispatch.
    ///
    /// `ctx` — the verified envelope context. The PEP derives the subject
    /// clearance via [`EnvelopeContext::security_context`] (Claims ×
    /// VerifiedKeyMaterial, S1 invariant). No schema change.
    ///
    /// `service_domain` — the canonical service name being dispatched to.
    /// `method` — the browser method discriminator if sealed in the
    /// transcript, else `None`.
    #[must_use]
    fn check(
        &self,
        ctx: &EnvelopeContext,
        service_domain: &str,
        method: Option<u16>,
    ) -> MacDecision;
}

// ── Default PEP: clearance + object-label + can_access ────────────────────

/// The production-shaped MAC dispatch PEP wiring the three-step check.
///
/// Composes:
/// 1. [`EnvelopeContext::security_context`] (subject clearance — the #698
///    seam; returns `None` until production clearance is wired),
/// 2. an [`RpcObjectLabelResolver`] (trusted object label — stubbed with
///    [`DenyAllObjectResolver`] until S3/#569 schema annotations land),
/// 3. [`SecurityContext::can_access`] (the lattice floor).
///
/// Every missing input denies. There is no constructor that can produce a
/// permissive PEP from these inputs.
pub struct DefaultMacDispatchPep {
    resolver: Box<dyn RpcObjectLabelResolver>,
    activation_controlled: bool,
}

impl DefaultMacDispatchPep {
    /// Construct with a specific object-label resolver.
    pub fn new(resolver: Box<dyn RpcObjectLabelResolver>) -> Self {
        Self {
            resolver,
            activation_controlled: false,
        }
    }

    /// Select subject contexts through the process-global coverage gate.
    ///
    /// Production constructors opt into this explicitly.  Direct unit-test
    /// and embedding constructors retain their historical identity-aware
    /// behavior so the operator gate cannot make unrelated PEP tests
    /// order-dependent.
    pub fn with_activation_control(mut self) -> Self {
        self.activation_controlled = true;
        self
    }

    /// Construct with the fail-closed [`DenyAllObjectResolver`].
    ///
    /// Every object resolves to `None` ⇒ every request denies
    /// `UnlabeledObject`. This is the honest default until a production
    /// resolver is wired.
    pub fn fail_closed() -> Self {
        Self::new(Box::new(DenyAllObjectResolver))
    }
}

impl MacDispatchPep for DefaultMacDispatchPep {
    fn check(
        &self,
        ctx: &EnvelopeContext,
        service_domain: &str,
        method: Option<u16>,
    ) -> MacDecision {
        // Preserve the verified context for the direct VFS/CAS/MoQ PEPs, whose
        // low-level APIs carry Subject but not the full verified envelope.
        super::activation::remember_verified_subject(ctx);

        // 1. Subject clearance selected by the coverage-gated activation
        // control. Floor-only uses anonymous_floor; identity-aware consumes the
        // two-input Claims × VerifiedKeyMaterial derivation.
        let subject_ctx = if self.activation_controlled {
            super::activation::global_mac_activation_control()
                .select_context(ctx.security_context())
        } else {
            ctx.security_context()
        };
        let Some(subject_ctx) = subject_ctx else {
            return MacDecision::Deny(MacDenyReason::NoClearance);
        };

        // 2. Trusted object label (never from a token/UCAN/caveat — design §3).
        let Some(object_label) = self.resolver.resolve(service_domain, method) else {
            return MacDecision::Deny(MacDenyReason::UnlabeledObject);
        };

        // 3. Lattice floor (intrinsic dominance check — no policy argument).
        if subject_ctx.can_access(&object_label) {
            MacDecision::Permit
        } else {
            MacDecision::Deny(MacDenyReason::FloorDeny)
        }
    }
}

/// A PEP that unconditionally denies.
///
/// Install this explicitly when dispatch must fail closed before a real
/// object-label resolver is available. Leaving the global PEP uninstalled is
/// also fail-closed.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllMacPep;

impl MacDispatchPep for DenyAllMacPep {
    #[inline]
    fn check(
        &self,
        _ctx: &EnvelopeContext,
        _service_domain: &str,
        _method: Option<u16>,
    ) -> MacDecision {
        MacDecision::Deny(MacDenyReason::NoPepInstalled)
    }
}

// ── Process-global installation ───────────────────────────────────────────

static GLOBAL_PEP: parking_lot::RwLock<Option<Arc<dyn MacDispatchPep>>> =
    parking_lot::RwLock::new(None);

/// Install the node's MAC dispatch PEP.
///
/// Production calls this exactly once at startup with a
/// [`DefaultMacDispatchPep`] (or a production resolver). Until this is
/// called, [`global_mac_dispatch_pep`] returns `None` and
/// [`process_request`](crate::service::dispatch::process_request) denies.
///
/// Backed by `parking_lot::RwLock` (no poisoning). Production callers SHOULD
/// install exactly once; the security boundary is the PEP trait itself (a PEP
/// that denies is correct regardless of how many times it has been set).
///
/// # Activation gate (epic #1267)
///
/// Identity-aware subject selection remains a deliberate operator choice, but
/// mediation is never off. Before installation the structural sentinel denies;
/// after installation the production PEP remains mandatory.
pub fn install_mac_dispatch_pep(pep: Arc<dyn MacDispatchPep>) {
    *GLOBAL_PEP.write() = Some(pep);
}

/// The installed MAC dispatch PEP, if any.
///
/// `None` on a node that omitted installation. Dispatch treats this as a hard
/// deny; installation is not an enforcement bypass boundary.
#[must_use]
pub fn global_mac_dispatch_pep() -> Option<Arc<dyn MacDispatchPep>> {
    GLOBAL_PEP.read().clone()
}

/// Check the installed dispatch PEP, or deny if installation was omitted.
///
/// Once a PEP is installed, its result is returned unchanged: an installed
/// fail-closed PEP cannot be bypassed by this wrapper.
#[must_use]
pub fn check_dispatch_mac(
    ctx: &EnvelopeContext,
    service_domain: &str,
    method: Option<u16>,
) -> MacDecision {
    match global_mac_dispatch_pep() {
        Some(pep) => pep.check(ctx, service_domain, method),
        None => MacDecision::Deny(MacDenyReason::NoPepInstalled),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::auth::mac::{Assurance, CompartmentSet, Level, SecurityLabel};

    // ── test helpers ──────────────────────────────────────────────────

    fn ctx_with_clearance(label: Option<SecurityLabel>) -> EnvelopeContext {
        // Non-zero cnf so verified_key_material() returns Classical (not
        // Unverified), and the given clearance on Claims.
        EnvelopeContext::for_mac_test([1u8; 32], label)
    }

    fn public_label() -> SecurityLabel {
        SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn secret_label() -> SecurityLabel {
        SecurityLabel::new(Level::Secret, Assurance::Classical, CompartmentSet::EMPTY)
    }

    struct StaticResolver {
        labels: std::collections::HashMap<String, SecurityLabel>,
    }

    impl RpcObjectLabelResolver for StaticResolver {
        fn resolve(&self, service_domain: &str, _method: Option<u16>) -> Option<SecurityLabel> {
            self.labels.get(service_domain).copied()
        }
    }

    // ── DenyAllObjectResolver ─────────────────────────────────────────

    #[test]
    fn deny_all_resolver_returns_none_for_everything() {
        let r = DenyAllObjectResolver;
        assert!(r.resolve("model", None).is_none());
        assert!(r.resolve("anything", Some(42)).is_none());
    }

    // ── DenyAllMacPep ─────────────────────────────────────────────────

    #[test]
    fn deny_all_pep_denies_everything() {
        let pep = DenyAllMacPep;
        let ctx = ctx_with_clearance(Some(secret_label()));
        let decision = pep.check(&ctx, "model", None);
        assert_eq!(decision, MacDecision::Deny(MacDenyReason::NoPepInstalled));
        assert!(!decision.is_permit());
    }

    // ── DefaultMacDispatchPep ─────────────────────────────────────────

    #[test]
    fn default_pep_permits_when_clearance_dominates_object_label() {
        let resolver = Box::new(StaticResolver {
            labels: [("model".to_owned(), public_label())].into(),
        });
        let pep = DefaultMacDispatchPep::new(resolver);
        let ctx = ctx_with_clearance(Some(secret_label())); // Secret ⊒ Public

        let decision = pep.check(&ctx, "model", None);
        assert_eq!(decision, MacDecision::Permit);
    }

    #[test]
    fn default_pep_denies_floor_when_clearance_does_not_dominate() {
        let resolver = Box::new(StaticResolver {
            labels: [("model".to_owned(), secret_label())].into(),
        });
        let pep = DefaultMacDispatchPep::new(resolver);
        let ctx = ctx_with_clearance(Some(public_label())); // Public ⊉ Secret

        let decision = pep.check(&ctx, "model", None);
        assert_eq!(decision, MacDecision::Deny(MacDenyReason::FloorDeny));
    }

    #[test]
    fn default_pep_denies_no_clearance_when_subject_has_none() {
        let resolver = Box::new(StaticResolver {
            labels: [("model".to_owned(), public_label())].into(),
        });
        let pep = DefaultMacDispatchPep::new(resolver);
        let ctx = ctx_with_clearance(None); // unlabeled subject

        let decision = pep.check(&ctx, "model", None);
        assert_eq!(decision, MacDecision::Deny(MacDenyReason::NoClearance));
    }

    #[test]
    fn default_pep_denies_unlabeled_object_when_resolver_returns_none() {
        let pep = DefaultMacDispatchPep::fail_closed();
        let ctx = ctx_with_clearance(Some(secret_label()));

        let decision = pep.check(&ctx, "anything", None);
        assert_eq!(decision, MacDecision::Deny(MacDenyReason::UnlabeledObject));
    }

    #[test]
    fn default_pep_denies_for_unmapped_service() {
        let resolver = Box::new(StaticResolver {
            labels: [("model".to_owned(), public_label())].into(),
        });
        let pep = DefaultMacDispatchPep::new(resolver);
        let ctx = ctx_with_clearance(Some(secret_label()));

        let decision = pep.check(&ctx, "other_service", None);
        assert_eq!(decision, MacDecision::Deny(MacDenyReason::UnlabeledObject));
    }

    // ── Process-global installation ───────────────────────────────────

    /// Reset the global PEP to None for isolated testing of the no-PEP path.
    fn reset_global_pep() {
        *GLOBAL_PEP.write() = None;
    }

    #[test]
    fn check_dispatch_mac_denies_while_pep_is_uninstalled() {
        reset_global_pep();
        assert!(global_mac_dispatch_pep().is_none());

        let ctx = ctx_with_clearance(Some(secret_label()));
        let decision = check_dispatch_mac(&ctx, "model", None);
        assert_eq!(decision, MacDecision::Deny(MacDenyReason::NoPepInstalled));
    }

    #[test]
    fn install_mac_dispatch_pep_is_swappable() {
        reset_global_pep();

        let pep1: Arc<dyn MacDispatchPep> = Arc::new(DefaultMacDispatchPep::fail_closed());
        install_mac_dispatch_pep(pep1);
        assert!(global_mac_dispatch_pep().is_some());

        // Verify the installed PEP actually runs and fails closed.
        let ctx = ctx_with_clearance(Some(secret_label()));
        assert_eq!(
            check_dispatch_mac(&ctx, "model", None),
            MacDecision::Deny(MacDenyReason::UnlabeledObject),
            "installed PEP must fail closed on an unlabeled object"
        );

        // Swap to a deny-all PEP — the RwLock allows replacement.
        let pep2: Arc<dyn MacDispatchPep> = Arc::new(DenyAllMacPep);
        install_mac_dispatch_pep(pep2);
        assert_eq!(
            check_dispatch_mac(&ctx, "model", None),
            MacDecision::Deny(MacDenyReason::NoPepInstalled),
            "DenyAllMacPep must deny after swap"
        );

        reset_global_pep();
    }
    // ── MacDecision helpers ───────────────────────────────────────────

    #[test]
    fn permit_is_permit() {
        assert!(MacDecision::Permit.is_permit());
        assert!(!MacDecision::Deny(MacDenyReason::FloorDeny).is_permit());
    }
}
