//! 9P reference-monitor contract (S2 / #568, epic #547) — **dormant groundwork**.
//!
//! `hyprstream-9p` owns the 9P operation choke point but deliberately does not
//! depend on the application crate that owns MAC policy loading and the AVC.
//! This module is the dependency-inversion boundary: the application verifies a
//! sender-bound access token once at `Tattach`, then the translator applies the
//! label, token, and IFC gates *before* asking an injected local policy monitor
//! (the application's AVC/PDP adapter).
//!
//! The mediation ordering is intentional and non-optional:
//!
//! ```text
//! Tattach: verify sender-bound token -> SessionContext (cached per fid)
//! each op: resolve content-truth label -> token scope -> IFC -> local AVC/PDP
//! ```
//!
//! No token, expired token, unlabeled object, or non-dominating subject reaches
//! the policy monitor, so a permissive policy implementation cannot bypass the
//! mandatory token/IFC floor. Per-op cost is a fid-table lookup plus a local
//! AVC call — never UCAN chain validation or Casbin matching.
//!
//! ## Not active by default
//!
//! This is the S2 machinery, not S2 activation. [`Translator`](crate::Translator)
//! runs **without** a [`ReferenceMonitor`] unless the application installs one
//! via `with_reference_monitor`, and no production construction installs one
//! today — per-op behavior is then exactly what it was before this module
//! existed. Activation is blocked on:
//!
//! - **#698** — no production token-issuance path attaches `Claims.clearance`
//!   yet, so a live monitor would resolve every subject to deny; and
//! - **object labels for the served mount** — the genesis/manifest resolver
//!   (`hyprstream::mac::genesis::CompositeObjectLabelResolver`) exists but is
//!   not yet wired to the 9P export.
//!
//! Do not wire a monitor into the production serve paths from this crate; that
//! wiring lands with the activation change, after #698.
//!
//! ## Mediation is over the *name*, not the *object* (read before relying on it)
//!
//! The monitor is authoritative over the **path** a fid is tagged with, not
//! over the **object** the backend resolves that fid to. [`ReferenceMonitor::
//! authorize`] resolves the label from `ObjectRef::Path(components)` (the
//! cached walked path), while the backend independently resolves the fid to an
//! actual object/handle. Nothing in this contract *binds* the labeled name to
//! the served object: the invariant "the label checked is the label of the
//! object served" holds **only if the label resolver and the backend resolve
//! names identically**. That is a convention, not an enforced invariant — so
//! the honest statement of #568's "complete mediation" claim is:
//!
//! > Every dispatched op is routed through the monitor, **and** the monitor's
//! > label resolver agrees with the backend's name resolution for the served
//! > namespace. The first half is by construction; the second is a precondition.
//!
//! Concretely, this name↔object gap is **not** covered today:
//!
//! - **Partial walks** — the backend reports its exact reached request prefix,
//!   and the translator rejects a result whose QID count disagrees with that
//!   prefix. `MountBackend` emits one QID per bound component; `ModelBackend`
//!   rejects multi-name walks until it can provide that same contract. This
//!   prevents the former one-leaf-QID/deep-fid mismatch, but still relies on
//!   the label resolver and backend agreeing on what a reported path names.
//! - **Path traversal / `..`** — `hyprstream-9p` performs **no** `..`
//!   canonicalization; `wnames` are concatenated verbatim into the cached path.
//!   Not exploitable against `MemoryBackend`/`MountBackend` (they resolve
//!   component-wise from a root), but a future resolver that does not
//!   canonicalize identically to the backend reopens this gap. Any resolver
//!   wired at activation **must** canonicalize or reject `..`, or the label
//!   lookup and the backend resolution can diverge.
//! - **Bind / symlink / mount indirection** — the VFS supports bind mounts and
//!   per-process bind namespaces. If a path component resolves through an
//!   indirection the string-keyed resolver does not model, the label checked is
//!   for the *name* and the bytes served are for the *target*. `ObjectRef::Cid`
//!   exists precisely to close this (content-addressed refs bind the label to
//!   the object, not the name); the 9P seam feeds `ObjectRef::Path`.
//!
//! Closing the name↔object TOCTOU by making the monitor authoritative over the
//! **fid/object** (e.g. resolving labels from the backend's reached object, or
//! via `ObjectRef::Cid`) is **activation-blocking** alongside #698 and tracked
//! with #699. Until then, treat "complete mediation by construction" as the
//! *two-independent-resolutions-agree* claim above, not as object authority.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use hyprstream_rpc::auth::mac::{
    Assurance, CompartmentSet, Level, SecurityContext, SecurityLabel, VerifiedKeyMaterial,
};

pub use hyprstream_rpc::auth::mac::{ObjectLabelResolver, ObjectRef};

/// The 9P operation an [`AccessDecider`] is asked to authorize.
///
/// Mirrors the translator's dispatch surface (`walk`/`open`/`read`/`write`/
/// `getattr`/`readdir`) so a real decider can apply per-action policy (e.g.
/// deny `Write` more aggressively than `Read`). Only ops the translator
/// actually dispatches appear here — there is no `Tlcreate`/`Tremove` handler,
/// so no `Create`/`Remove` variant (a dead variant would be an unreachable
/// policy surface). `Tclunk` is fid disposal, not an object operation, and is
/// deliberately not mediated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Walk,
    Open,
    Read,
    Write,
    Getattr,
    Readdir,
}

/// A verified, sender-bound access-token scope cached for one 9P attach.
///
/// This type intentionally contains only the deny-only token facts needed on
/// the hot path. Signature/expiry/DPoP verification happens in the
/// [`AttachAuthenticator`] before constructing it; UCAN chain validation never
/// runs per operation. The object label is *not* token-supplied: it is passed
/// to [`Self::authorizes`] by the reference monitor after trusted resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedTokenScope {
    label_ceiling: SecurityLabel,
    operations: Arc<[Action]>,
    valid_until: Instant,
}

impl VerifiedTokenScope {
    /// Build the local representation of an access token that has already had
    /// its signature, sender binding, and expiry verified at attach time.
    ///
    /// Callers must not derive these values from a UCAN/caveat or a
    /// caller-supplied label. The monitor independently resolves the object
    /// label from the mounted object's manifest/genesis source.
    pub fn from_verified_token(
        label_ceiling: SecurityLabel,
        operations: impl Into<Arc<[Action]>>,
        valid_until: Instant,
    ) -> Self {
        Self {
            label_ceiling,
            operations: operations.into(),
            valid_until,
        }
    }

    /// The token's label ceiling, for diagnostics and application audit code.
    pub fn label_ceiling(&self) -> SecurityLabel {
        self.label_ceiling
    }

    /// Token gate: unexpired, exact operation membership, and object label no
    /// more restrictive than the verified token's ceiling.
    #[inline]
    pub fn authorizes(&self, object_label: &SecurityLabel, action: Action) -> bool {
        Instant::now() < self.valid_until
            && self.operations.contains(&action)
            && self.label_ceiling.can_access(object_label)
    }
}

/// Attach-scoped state cached onto every fid derived from the attach fid.
///
/// The public constructor is intentionally named for its trust precondition:
/// only an [`AttachAuthenticator`] that has verified the presented token and
/// DPoP sender binding may construct a permitting session. There is no
/// constructor from raw `uname`, `Subject`, claims, paths, or labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionContext {
    attach_identity: VerifiedAttachIdentity,
    security_context: SecurityContext,
    token: Option<VerifiedTokenScope>,
}

/// Stable principal or credential identity returned by an
/// [`AttachAuthenticator`] after verification.
///
/// This is deliberately separate from [`SecurityContext`]: clearance and key
/// assurance are authorization attributes, not a principal identity. The
/// verifier must derive this from the verified subject or stable credential
/// fingerprint, never from an unverified `Tattach.uname` string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedAttachIdentity(Arc<str>);

impl VerifiedAttachIdentity {
    /// Construct an identity obtained from verified credential material.
    pub fn from_verified_identity(identity: impl Into<Arc<str>>) -> Self {
        Self(identity.into())
    }
}

impl SessionContext {
    /// Combine a verified subject context with the scope of the same verified,
    /// sender-bound token.
    pub fn from_verified_token(
        attach_identity: VerifiedAttachIdentity,
        security_context: SecurityContext,
        token: VerifiedTokenScope,
    ) -> Self {
        Self {
            attach_identity,
            security_context,
            token: Some(token),
        }
    }

    /// A deliberately unusable session for an invalid or absent credential.
    /// The anonymous security floor is kept for diagnostics; the missing token
    /// makes every operation deny before the AVC/PDP is reached.
    pub fn deny() -> Self {
        Self {
            attach_identity: VerifiedAttachIdentity::from_verified_identity("denied"),
            security_context: anonymous_floor(),
            token: None,
        }
    }

    /// The context derived at attach from verified identity and key material.
    pub fn security_context(&self) -> &SecurityContext {
        &self.security_context
    }

    /// Apply the mandatory verified-token gate. A missing token is deny.
    #[inline]
    pub fn token_authorizes(&self, object_label: &SecurityLabel, action: Action) -> bool {
        self.token
            .as_ref()
            .is_some_and(|token| token.authorizes(object_label, action))
    }

    /// Identity comparison for re-attach detection, deliberately distinct from
    /// the derived [`PartialEq`].
    ///
    /// Two attaches are the *same* session when their verified identity,
    /// verified subject context, and verified-token scope (label ceiling +
    /// permitted operations) match.
    /// The token's `valid_until` deadline is excluded on purpose: a real
    /// [`AttachAuthenticator`] re-stamps `Instant::now() + ttl` on every
    /// verification of the *same* ticket, so including it would make an
    /// identical, still-valid ticket compare unequal and wrongly reject a
    /// legitimate re-attach/reconnect (F2). The derived `PartialEq` is plain
    /// value equality (useful for diagnostics/tests); *attach identity* — what
    /// `bind_attach_session` checks — is this method.
    pub fn same_attach_identity(&self, other: &SessionContext) -> bool {
        if self.attach_identity != other.attach_identity {
            return false;
        }
        if self.security_context != other.security_context {
            return false;
        }
        match (&self.token, &other.token) {
            // The token's `valid_until` is deliberately ignored: a real
            // authenticator re-stamps it on every verification of the same
            // ticket, so it is not part of attach identity.
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.label_ceiling == b.label_ceiling && a.operations == b.operations
            }
            _ => false,
        }
    }
}

/// Verifies the `Tattach` credential and returns the session cached per fid.
///
/// Implementations verify the S6-minted access token once (hybrid signature,
/// expiry, DPoP binding, and policy generation), derive `SecurityContext` from
/// verified identity/key material, and translate the token's authorized subset
/// to [`VerifiedTokenScope`]. Invalid input returns [`SessionContext::deny`].
#[async_trait]
pub trait AttachAuthenticator: Send + Sync {
    async fn authenticate(&self, uname: &str, aname: &str) -> SessionContext;
}

/// Fail-closed authenticator: every attach resolves to a deny-only session.
/// Used by [`ReferenceMonitor`] constructions that have not been given an
/// application authenticator, and by tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct AnonymousAuthenticator;

#[async_trait]
impl AttachAuthenticator for AnonymousAuthenticator {
    async fn authenticate(&self, _uname: &str, _aname: &str) -> SessionContext {
        SessionContext::deny()
    }
}

/// The anonymous security floor. It is not authorization: a session at this
/// floor still has no [`VerifiedTokenScope`] and is denied.
pub fn anonymous_floor() -> SecurityContext {
    SecurityContext::from_clearance(
        SecurityLabel::new(Level::Public, Assurance::Unverified, CompartmentSet::EMPTY),
        VerifiedKeyMaterial::Unverified,
    )
}

/// Fail-closed label resolver for a mount with no content/static label source.
/// A real export injects the genesis/manifest resolver owned by the
/// application. Returning `None` is a mandatory deny (design §1 invariant 2).
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyUnlabeledResolver;

impl ObjectLabelResolver for DenyUnlabeledResolver {
    fn resolve(&self, _object: ObjectRef<'_>) -> Option<SecurityLabel> {
        None
    }
}

/// The policy decision point behind the mandatory token and IFC gates.
///
/// `check` is called only after the monitor has resolved a non-optional
/// object label, checked the attached token's operation/ceiling/TTL, and
/// enforced `subject.ctx ⊒ object.label`. The concrete application
/// implementation is a local AVC lookup (with a PDP miss); it must not perform
/// UCAN chain validation or Casbin matching per operation.
pub trait AccessDecider: Send + Sync {
    fn check(&self, ctx: &SecurityContext, object_label: &SecurityLabel, action: Action) -> bool;
}

/// The fail-closed default policy monitor. It permits nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllDecider;

impl AccessDecider for DenyAllDecider {
    fn check(
        &self,
        _ctx: &SecurityContext,
        _object_label: &SecurityLabel,
        _action: Action,
    ) -> bool {
        false
    }
}

/// The reference monitor a [`Translator`](crate::Translator) mediates every
/// dispatched op through once installed. Bundles the three injection points so
/// enforcement is all-or-nothing: there is no translator state in which the
/// token gate runs but the label resolver is absent, or vice versa.
///
/// Constructed by the application (`hyprstream` crate) at activation time from
/// its concrete token verifier, genesis/manifest label resolver, and AVC/PDP
/// adapter. Not installed by any production path yet — see the module docs.
pub struct ReferenceMonitor {
    authenticator: Arc<dyn AttachAuthenticator>,
    labels: Arc<dyn ObjectLabelResolver + Send + Sync>,
    decider: Arc<dyn AccessDecider>,
}

impl ReferenceMonitor {
    /// Assemble the monitor from its three mandatory seams. There is no
    /// permissive or partial construction: each argument is required, and the
    /// crate's fail-closed defaults ([`AnonymousAuthenticator`],
    /// [`DenyUnlabeledResolver`], [`DenyAllDecider`]) are the explicit choices
    /// for a seam the application does not supply.
    pub fn new(
        authenticator: Arc<dyn AttachAuthenticator>,
        labels: Arc<dyn ObjectLabelResolver + Send + Sync>,
        decider: Arc<dyn AccessDecider>,
    ) -> Self {
        Self {
            authenticator,
            labels,
            decider,
        }
    }

    /// Verify the `Tattach` credential exactly once for this connection.
    pub async fn authenticate(&self, uname: &str, aname: &str) -> SessionContext {
        self.authenticator.authenticate(uname, aname).await
    }

    /// Complete mediation for one op on `path` under `session`, in the
    /// mandatory order: trusted label resolution → token gate → independent
    /// IFC dominance → local policy monitor. Every step fails closed: an
    /// unresolvable label, missing/expired/insufficient token, non-dominating
    /// subject, or decider denial all return `false`.
    pub fn authorize(&self, session: &SessionContext, path: &[String], action: Action) -> bool {
        let components: Vec<&str> = path.iter().map(String::as_str).collect();
        let Some(object_label) = self.labels.resolve(ObjectRef::Path(&components)) else {
            return false;
        };

        // Token is a deny-only capability gate. Its label ceiling is checked
        // against the trusted object label, never a token/UCAN-provided label.
        if !session.token_authorizes(&object_label, action) {
            return false;
        }

        // IFC dominance is independent of the token and the policy matrix.
        if !session.security_context().can_access(&object_label) {
            return false;
        }

        self.decider
            .check(session.security_context(), &object_label, action)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn label(level: Level) -> SecurityLabel {
        SecurityLabel::new(level, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn ctx(level: Level) -> SecurityContext {
        // Classical key material so the derived assurance dominates the
        // Classical-assurance object labels used across these tests.
        SecurityContext::new(level, CompartmentSet::EMPTY, VerifiedKeyMaterial::Classical)
    }

    fn permit_token(ops: &[Action]) -> VerifiedTokenScope {
        VerifiedTokenScope::from_verified_token(
            label(Level::Secret),
            Arc::from(ops),
            Instant::now() + Duration::from_secs(3600),
        )
    }

    struct StaticLabels(Option<SecurityLabel>);
    impl ObjectLabelResolver for StaticLabels {
        fn resolve(&self, _object: ObjectRef<'_>) -> Option<SecurityLabel> {
            self.0
        }
    }

    struct AllowAll;
    impl AccessDecider for AllowAll {
        fn check(
            &self,
            _ctx: &SecurityContext,
            _object_label: &SecurityLabel,
            _action: Action,
        ) -> bool {
            true
        }
    }

    fn monitor(labels: Option<SecurityLabel>, decider: Arc<dyn AccessDecider>) -> ReferenceMonitor {
        ReferenceMonitor::new(
            Arc::new(AnonymousAuthenticator),
            Arc::new(StaticLabels(labels)),
            decider,
        )
    }

    fn path(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn token_scope_requires_unexpired_op_and_label_ceiling() {
        let scope = VerifiedTokenScope::from_verified_token(
            label(Level::Confidential),
            Arc::from([Action::Read]),
            Instant::now() + Duration::from_secs(10),
        );
        assert!(scope.authorizes(&label(Level::Public), Action::Read));
        assert!(!scope.authorizes(&label(Level::Public), Action::Write));
        assert!(!scope.authorizes(&label(Level::Secret), Action::Read));
    }

    #[test]
    fn expired_or_missing_token_fails_closed() {
        let expired = VerifiedTokenScope::from_verified_token(
            label(Level::Secret),
            Arc::from([Action::Read]),
            Instant::now() - Duration::from_secs(1),
        );
        assert!(!expired.authorizes(&label(Level::Public), Action::Read));
        assert!(!SessionContext::deny().token_authorizes(&label(Level::Public), Action::Read));
    }

    #[test]
    fn authorize_allows_when_all_gates_pass() {
        let monitor = monitor(Some(label(Level::Public)), Arc::new(AllowAll));
        let session =
            SessionContext::from_verified_token(VerifiedAttachIdentity::from_verified_identity("test-subject"), ctx(Level::Secret), permit_token(&[Action::Read]));
        assert!(monitor.authorize(&session, &path(&["a.txt"]), Action::Read));
    }

    #[test]
    fn authorize_denies_unlabeled_before_decider() {
        // Even a permit-everything decider cannot rescue an unlabeled object.
        let monitor = monitor(None, Arc::new(AllowAll));
        let session =
            SessionContext::from_verified_token(VerifiedAttachIdentity::from_verified_identity("test-subject"), ctx(Level::Secret), permit_token(&[Action::Read]));
        assert!(!monitor.authorize(&session, &path(&["a.txt"]), Action::Read));
    }

    #[test]
    fn authorize_denies_missing_or_scope_violating_token() {
        let monitor = monitor(Some(label(Level::Public)), Arc::new(AllowAll));
        assert!(!monitor.authorize(&SessionContext::deny(), &path(&["a.txt"]), Action::Read));
        let read_only =
            SessionContext::from_verified_token(VerifiedAttachIdentity::from_verified_identity("test-subject"), ctx(Level::Secret), permit_token(&[Action::Read]));
        assert!(!monitor.authorize(&read_only, &path(&["a.txt"]), Action::Write));
    }

    #[test]
    fn authorize_enforces_ifc_floor_independent_of_decider() {
        // Permissive policy must not bypass the mandatory IFC floor: a Public
        // subject reading a Secret object denies even with an allow-all
        // decider and a token whose ceiling covers it.
        let monitor = monitor(Some(label(Level::Secret)), Arc::new(AllowAll));
        let session =
            SessionContext::from_verified_token(VerifiedAttachIdentity::from_verified_identity("test-subject"), ctx(Level::Public), permit_token(&[Action::Read]));
        assert!(!monitor.authorize(&session, &path(&["secret.txt"]), Action::Read));
    }

    #[test]
    fn deny_unlabeled_resolver_resolves_nothing() {
        let resolver = DenyUnlabeledResolver;
        assert!(resolver.resolve(ObjectRef::Path(&["anything"])).is_none());
    }
}
