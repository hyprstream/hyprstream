//! Attach-time MAC seam for the 9P translator (#568, epic #547).
//!
//! `hyprstream-9p` sits *below* `hyprstream` in the dependency graph (the
//! reverse: `hyprstream` depends on `hyprstream-9p`), so the concrete grant
//! verification (`mac::exchange`), the compiled-policy PDP (`mac::te`), the
//! per-op cache (`mac::avc::CachingAvc`), and the audit trail
//! (`mac::audit::AuditedAvc`) — which all live in the `hyprstream` binary
//! crate — can never be called from here without a dependency cycle.
//!
//! Per the ratified interface policy (CLAUDE.md, "Interface policy — MAC on
//! contracts that don't carry it", rule 2): the fix is dependency inversion,
//! not moving code. This module defines the trait seam; the `hyprstream`
//! crate is expected to provide the concrete implementations and inject them
//! wherever it constructs a [`crate::Translator`] (mirroring the existing
//! `Backend`/`ModelBackend` precedent — the capnp-RPC-backed `Backend` impl
//! already lives in the `hyprstream` crate, not here).
//!
//! ## Fail-closed defaults ship inert, not deny-by-default
//!
//! [`LabeledObject::security_label`] returning `None` means the object is
//! **unlabeled**, and per the S1 invariant (`hyprstream_rpc::auth::mac`,
//! design §1 invariant 2) an unlabeled object MUST deny — there is no
//! "unlabeled ⇒ floor" default at that layer. Every object in the 9P
//! namespace is unlabeled today (genesis labeling for 9P nodes is unstarted,
//! and #699 carrier-b — the content-addressed manifest label field — is still
//! an open design decision). Wiring a real deny-on-unlabeled
//! [`AccessDecider`] into the live dispatch path by default would therefore
//! deny every existing 9P client outright — a severe, silent regression, not
//! a security fix.
//!
//! So the defaults here ([`AnonymousAuthenticator`], [`AllowAllDecider`])
//! preserve **today's actual behavior** (no enforcement) exactly. They exist
//! so the translator has somewhere to hang real enforcement once a caller
//! opts in via [`Translator::with_authenticator`](crate::Translator::with_authenticator)
//! / [`Translator::with_decider`](crate::Translator::with_decider) — which the
//! `hyprstream` crate can do once (a) a standalone "verify a presented S6
//! access token" entry point exists outside the OAuth HTTP handler, and (b)
//! object labels are actually available for the mount being served (genesis
//! labeling, or #699 carrier-b for content-addressed data). Both are called
//! out as follow-ups in the PR body rather than implemented here.

use async_trait::async_trait;
use hyprstream_rpc::auth::mac::{
    Assurance, CompartmentSet, Level, SecurityContext, SecurityLabel, VerifiedKeyMaterial,
};

/// The 9P op an [`AccessDecider`] is being asked to authorize.
///
/// Mirrors the translator's dispatch surface (`walk`/`open`/`read`/`write`/
/// `getattr`/`readdir`) so a real decider can apply per-action policy (e.g.
/// deny `Write` more aggressively than `Read`). Only ops the translator
/// actually dispatches appear here — there is no `Tlcreate` handler, so no
/// `Create` variant (a dead variant would be an unreachable policy surface).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Walk,
    Open,
    Read,
    Write,
    Getattr,
    Readdir,
}

/// Verifies an attach credential (9P `uname`/`aname`) and derives the
/// caller's [`SecurityContext`] for the resulting fid.
///
/// Called exactly once per `Tattach`, mirroring the ratified interface
/// policy's rule 2 (extend at attach time, never per-op): the derived context
/// is cached on the fid (see `FidTable`) and reused for every subsequent op
/// on that fid and its descendants, never re-verified per op.
#[async_trait]
pub trait AttachAuthenticator: Send + Sync {
    /// Verify `uname`/`aname` and return the derived context. Must never
    /// panic or block indefinitely on a malformed/hostile credential — an
    /// invalid credential resolves to the anonymous floor context, exactly
    /// like a missing one (fail-closed by floor, not by rejecting the
    /// attach outright, so unauthenticated legacy clients keep working).
    async fn authenticate(&self, uname: &str, aname: &str) -> SecurityContext;
}

/// The default, no-op authenticator: every attach resolves to the anonymous
/// floor context regardless of `uname`/`aname`. This is exactly today's
/// behavior (the translator performs no identity verification at all), kept
/// as the default so existing callers see no behavior change.
#[derive(Debug, Default, Clone, Copy)]
pub struct AnonymousAuthenticator;

#[async_trait]
impl AttachAuthenticator for AnonymousAuthenticator {
    async fn authenticate(&self, _uname: &str, _aname: &str) -> SecurityContext {
        anonymous_floor()
    }
}

/// The anonymous-floor [`SecurityContext`]: `Level::Public`, no compartments,
/// `VerifiedKeyMaterial::Unverified` (assurance floors at `Unverified`).
pub fn anonymous_floor() -> SecurityContext {
    SecurityContext::from_clearance(
        SecurityLabel::new(Level::Public, Assurance::Unverified, CompartmentSet::EMPTY),
        VerifiedKeyMaterial::Unverified,
    )
}

/// Authorizes one op against a subject's cached [`SecurityContext`].
///
/// `object_label` is `None` when no label source is wired for the mount
/// being served (true for every 9P mount today — see the module-level
/// documentation on why `None` must NOT be treated as "unrestricted" by a
/// real implementation). The default [`AllowAllDecider`] ignores both
/// arguments and always allows, preserving current behavior; a real decider
/// (injected by the `hyprstream` crate, backed by `mac::avc::CachingAvc`)
/// must deny when `object_label` is `None`, per the S1 invariant.
pub trait AccessDecider: Send + Sync {
    /// Returns whether `action` is authorized for `ctx` against `object_label`.
    fn check(
        &self,
        ctx: &SecurityContext,
        object_label: Option<&SecurityLabel>,
        action: Action,
    ) -> bool;
}

/// The default, no-op decider: every op is allowed regardless of context or
/// label. This is exactly today's behavior (the translator performs no
/// per-op authorization at all), kept as the default so existing callers see
/// no behavior change until a real decider is explicitly wired in.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllowAllDecider;

impl AccessDecider for AllowAllDecider {
    fn check(&self, _ctx: &SecurityContext, _object_label: Option<&SecurityLabel>, _action: Action) -> bool {
        true
    }
}

/// Records the outcome of an [`AccessDecider::check`] call for audit.
///
/// A real sink (injected by the `hyprstream` crate, backed by
/// `mac::audit::WalAuditStore`) durably logs the decision; per S7, a
/// decision that cannot be durably audited must downgrade to deny. The
/// default [`NullAuditSink`] does not implement that downgrade (there is
/// nothing to audit against by default) — it is only safe as a default
/// because [`AllowAllDecider`] is also the default, so no security-relevant
/// decision is ever made without an explicit opt-in to both a real decider
/// and a real sink together (see [`AuditedDecider`]).
pub trait AuditSink: Send + Sync {
    /// `allowed` is the decision `AuditedDecider` is about to return.
    fn record(&self, ctx: &SecurityContext, object_label: Option<&SecurityLabel>, action: Action, allowed: bool);
}

/// The default, no-op audit sink: records nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullAuditSink;

impl AuditSink for NullAuditSink {
    fn record(&self, _ctx: &SecurityContext, _object_label: Option<&SecurityLabel>, _action: Action, _allowed: bool) {}
}

/// Wraps an [`AccessDecider`] so every decision is also recorded via an
/// [`AuditSink`], and — mirroring S7's `AuditedAvc` fail-closed contract — a
/// sink that reports it could not durably record the decision causes the
/// wrapped decision to be downgraded to deny.
///
/// [`AuditSink::record`] here is infallible by trait signature (matching the
/// simple sinks available at this layer); a real, fallible durable sink
/// should be adapted to report failure through `allowed` before calling
/// `record`, or a future revision can widen `AuditSink::record` to return a
/// `Result` once a concrete durable sink is wired in from the `hyprstream`
/// crate. Documented here rather than solved speculatively.
pub struct AuditedDecider<D, S> {
    inner: D,
    sink: S,
}

impl<D: AccessDecider, S: AuditSink> AuditedDecider<D, S> {
    pub fn new(inner: D, sink: S) -> Self {
        Self { inner, sink }
    }
}

impl<D: AccessDecider, S: AuditSink> AccessDecider for AuditedDecider<D, S> {
    fn check(&self, ctx: &SecurityContext, object_label: Option<&SecurityLabel>, action: Action) -> bool {
        let allowed = self.inner.check(ctx, object_label, action);
        self.sink.record(ctx, object_label, action, allowed);
        allowed
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn anonymous_authenticator_always_floors() {
        let auth = AnonymousAuthenticator;
        let ctx = auth.authenticate("anyone", "anything").await;
        assert_eq!(ctx.level(), Level::Public);
        assert_eq!(ctx.assurance(), hyprstream_rpc::auth::mac::Assurance::Unverified);
    }

    #[test]
    fn allow_all_decider_ignores_inputs() {
        let decider = AllowAllDecider;
        assert!(decider.check(&anonymous_floor(), None, Action::Write));
        let secret = SecurityLabel::new(
            Level::Secret,
            hyprstream_rpc::auth::mac::Assurance::PqHybrid,
            Default::default(),
        );
        assert!(decider.check(&anonymous_floor(), Some(&secret), Action::Write));
    }

    struct DenyAll;
    impl AccessDecider for DenyAll {
        fn check(&self, _ctx: &SecurityContext, _object_label: Option<&SecurityLabel>, _action: Action) -> bool {
            false
        }
    }

    #[derive(Default)]
    struct CountingSink {
        calls: AtomicUsize,
    }
    impl AuditSink for CountingSink {
        fn record(&self, _ctx: &SecurityContext, _object_label: Option<&SecurityLabel>, _action: Action, _allowed: bool) {
            self.calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn audited_decider_records_and_forwards_decision() {
        let sink = CountingSink::default();
        let decider = AuditedDecider::new(DenyAll, sink);
        assert!(!decider.check(&anonymous_floor(), None, Action::Read));
        assert_eq!(decider.sink.calls.load(Ordering::SeqCst), 1);
    }
}
