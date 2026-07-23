//! Subscribe-time authorization + per-tenant announce scoping for the moq
//! streaming plane (#276).
//!
//! ## Why this module exists
//!
//! #274 made the moq streaming plane network-reachable (served on
//! `web_transport_quinn` at `/moq`, and over the iroh `moql` ALPN via
//! [`crate::transport::iroh_moq::IrohMoqProtocolHandler`]). That moved the
//! *open-subscribe surface* onto the network. This is **not** a content hole —
//! stream content is already protected by the §7.5 trust model (a 64-hex
//! DH-derived HKDF-of-ECDH capability **topic**, unguessable and learned only
//! from a signed `StreamInfo` over authenticated RPC, plus a per-frame chained
//! HMAC). #276 is **metadata defense-in-depth**:
//!
//! 1. A *connected* peer can announce-enumerate broadcast (topic) names — moq
//!    announces broadcasts by prefix, so a subscriber's `announced()` cursor
//!    sees every broadcast under its allowed prefix. Without scoping, a peer in
//!    tenant `A` could enumerate tenant `B`'s broadcast names.
//! 2. Subscribe-time authorization: a *public* stream stays open (preserving
//!    the working same-tenant subscribe model), but a *private* stream is gated
//!    through the existing policy engine.
//!
//! ## moq-net's authorization primitive
//!
//! `moq_net::Server` exposes **no per-subscribe callback**. Authorization in
//! moq-net is enforced *structurally*: a session can only enumerate/subscribe
//! to broadcasts that are visible through the [`OriginConsumer`] handed to
//! `Server::with_publish`. [`OriginConsumer::scope`] / [`OriginConsumer::with_root`]
//! narrow that visibility by path prefix. So **per-tenant announce scoping** is
//! implemented by handing each accepted session a tenant-scoped consumer
//! (see [`tenant_scoped_consumer`]) — a subscriber that cannot *see* a
//! broadcast cannot subscribe to it either.
//!
//! The **subscribe authorization hook** ([`SubscribeAuthorizer`]) is the
//! pluggable policy layer on top of that structural scoping, for the
//! public-vs-private distinction within a tenant. It is wired where the peer
//! identity needed to evaluate policy is actually available (see the wiring
//! notes on [`SubscribeAuthorizer`]).
//!
//! Track names are hierarchical `{tenant}/{service}/{topic}/{instance}` (the
//! #134 CDN-portability invariant). The first path segment is the tenant.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use moq_net::{OriginConsumer, Path};

/// The hierarchical track-name separator (`{tenant}/{service}/{topic}/{instance}`).
pub const TRACK_NAME_SEP: char = '/';

/// Extract the `{tenant}` segment from a hierarchical track / broadcast name.
///
/// Names are `{tenant}/{service}/{topic}/{instance}`; the tenant is the first
/// path segment. Returns `None` for an empty name or a name whose first segment
/// is empty (e.g. a leading `/`).
///
/// ```
/// use hyprstream_rpc::moq_authz::tenant_of;
/// assert_eq!(tenant_of("alice/streams/run-1/i0"), Some("alice"));
/// assert_eq!(tenant_of("alice"), Some("alice"));
/// assert_eq!(tenant_of(""), None);
/// assert_eq!(tenant_of("/leading"), None);
/// ```
pub fn tenant_of(track_name: &str) -> Option<&str> {
    match track_name.split(TRACK_NAME_SEP).next() {
        Some(t) if !t.is_empty() => Some(t),
        _ => None,
    }
}

/// Build the tenant prefix (`{tenant}/`) used to scope announce enumeration.
///
/// The trailing separator is significant: moq prefix matching is segment-aware,
/// so `"alice/"` matches `alice/...` but NOT a sibling tenant `alicent/...`.
pub fn tenant_prefix(tenant: &str) -> String {
    format!("{tenant}{TRACK_NAME_SEP}")
}

/// Pure per-tenant announce filter — the independently-testable core of the
/// scoping behaviour.
///
/// Given a set of broadcast/track names and a tenant, returns only the names
/// belonging to that tenant (i.e. whose first `{tenant}` segment equals
/// `tenant`). This is what a peer in `tenant` is permitted to enumerate; names
/// from other tenants are filtered out so cross-tenant broadcast-name
/// enumeration is impossible.
///
/// Matching is segment-exact on the tenant (`alice` does not match `alicent`).
pub fn filter_announces_by_tenant<'a, I>(names: I, tenant: &str) -> Vec<&'a str>
where
    I: IntoIterator<Item = &'a str>,
{
    names
        .into_iter()
        .filter(|name| tenant_of(name) == Some(tenant))
        .collect()
}

/// Restrict an [`OriginConsumer`] to a single tenant's broadcasts.
///
/// This is the live, network-side enforcement of per-tenant announce scoping:
/// the returned consumer's `announced()` cursor only ever yields broadcasts
/// under `{tenant}/`, and `subscribe`/`announced_broadcast` for any path outside
/// that prefix returns `None`. Hand the scoped consumer to
/// `moq_net::Server::with_publish` for the accepted session and the peer cannot
/// see — let alone subscribe to — another tenant's broadcasts.
///
/// Returns `None` if the requested prefix is outside the consumer's existing
/// allowed prefixes (the moq-net `scope` contract); callers should treat that
/// as "this tenant has no visible broadcasts" (serve an empty scope) rather
/// than falling back to the unscoped consumer.
#[cfg(not(target_arch = "wasm32"))]
pub fn tenant_scoped_consumer(consumer: &OriginConsumer, tenant: &str) -> Option<OriginConsumer> {
    let prefix = tenant_prefix(tenant);
    consumer.scope(&[Path::new(&prefix)])
}

/// Whether a stream is publicly subscribable or policy-gated.
///
/// Public streams preserve the existing open model (any connected, correctly
/// tenant-scoped peer may subscribe). Private streams are gated through
/// [`SubscribeAuthorizer`] / the policy engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Open subscribe (subject only to tenant scoping).
    Public,
    /// Subscribe must be authorized by policy.
    Private,
}

/// The outcome of a subscribe-authorization check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscribeDecision {
    /// The subscribe is permitted.
    Allow,
    /// The subscribe is denied (fail-closed for private + unauthorized).
    Deny,
}

impl SubscribeDecision {
    /// `true` iff the decision permits the subscribe.
    pub fn is_allowed(self) -> bool {
        matches!(self, SubscribeDecision::Allow)
    }
}

/// Identity of the subscribing peer, as far as the moq accept path can
/// determine it.
///
/// Carrier metadata such as an iroh NodeId is not an authorization identity.
/// Without fresh application proof, peers have `subject == None` and must be
/// treated as unauthenticated.
#[derive(Debug, Clone, Default)]
pub struct PeerIdentity {
    /// Stable independently verified application subject for policy lookups, or
    /// `None` for an unauthenticated/anonymous peer.
    pub subject: Option<String>,
}

impl PeerIdentity {
    /// An anonymous peer (no authenticated identity).
    pub fn anonymous() -> Self {
        Self { subject: None }
    }

    /// A peer authenticated as `subject`.
    pub fn authenticated(subject: impl Into<String>) -> Self {
        Self {
            subject: Some(subject.into()),
        }
    }

    /// `true` iff this peer carries an authenticated subject.
    pub fn is_authenticated(&self) -> bool {
        self.subject.is_some()
    }
}

/// Pluggable subscribe-time authorization hook for the moq accept path.
///
/// Implementors decide, given the (best-effort) peer identity and the
/// hierarchical track name, whether a subscribe is permitted. The intended
/// composition is: **structural tenant scoping** ([`tenant_scoped_consumer`])
/// gates *visibility / cross-tenant enumeration*, and this hook gates
/// *public-vs-private* within what's visible.
///
/// ## Wiring status
///
/// - **iroh `moql` path** ([`crate::transport::iroh_moq::IrohMoqProtocolHandler`]):
///   `remote_id()` is carrier metadata, so the hook receives anonymous until
///   #1027 supplies fresh proof.
/// - **quinn `/moq` path** ([`crate::transport::quinn_transport::QuinnRpcServer`]):
///   as of #1153 the CONNECT is authenticated by
///   [`crate::transport::moq_connect_auth::MoqConnectAuthz`] (bearer JWT,
///   verified before the handshake completes); the tenant is resolved from the
///   *verified* subject and a `tenant_scoped_consumer` is served. With no
///   `MoqConnectAuthz` installed, the peer is anonymous and the hook receives
///   [`PeerIdentity::anonymous`] (single-tenant/open model); a policy-gated
///   authorizer will deny *private* subscribes from anonymous quinn peers
///   (fail-closed) while public broadcasts remain open.
pub trait SubscribeAuthorizer: Send + Sync {
    /// Decide whether `peer` may subscribe to `track_name`.
    fn authorize(&self, peer: &PeerIdentity, track_name: &str) -> SubscribeDecision;
}

/// Classifies a track name as public or private. Returning [`Visibility::Public`]
/// keeps the existing open model; [`Visibility::Private`] routes through the
/// policy gate.
pub type VisibilityFn = Arc<dyn Fn(&str) -> Visibility + Send + Sync>;

/// Evaluates a policy decision for a (subject, tenant, track) tuple. Returns
/// `true` to allow. This is the seam onto the existing Casbin/PolicyService
/// (`PolicyManager::check_with_domain`) without `hyprstream-rpc` depending on
/// the `hyprstream` crate.
pub type PolicyGate = Arc<dyn Fn(&PeerIdentity, &str) -> bool + Send + Sync>;

/// Default authorizer: **public streams are open, private streams are
/// policy-gated, and unknown classification fails closed for safety**.
///
/// - If [`Visibility::Public`] → always [`SubscribeDecision::Allow`] (preserves
///   the working same-tenant open-subscribe model).
/// - If [`Visibility::Private`] → delegate to the [`PolicyGate`]; allow iff the
///   gate returns `true`. With no gate installed, private subscribes are denied
///   (fail-closed) — we never invent fail-dangerous behaviour.
///
/// When constructed with [`DefaultAuthorizer::permissive`], every track is
/// treated as public (the pre-#276 behaviour) — used as the default until a
/// deployment opts into private-stream classification.
#[derive(Clone)]
pub struct DefaultAuthorizer {
    visibility: VisibilityFn,
    policy: Option<PolicyGate>,
}

impl DefaultAuthorizer {
    /// Permissive default: every track is treated as public (open subscribe).
    /// Per-tenant announce scoping still applies independently.
    pub fn permissive() -> Self {
        Self {
            visibility: Arc::new(|_| Visibility::Public),
            policy: None,
        }
    }

    /// Build with a custom visibility classifier and a policy gate for private
    /// streams. Private + (no gate or gate-denied) → denied.
    pub fn new(visibility: VisibilityFn, policy: Option<PolicyGate>) -> Self {
        Self { visibility, policy }
    }

    /// Set/replace the visibility classifier.
    pub fn with_visibility(mut self, visibility: VisibilityFn) -> Self {
        self.visibility = visibility;
        self
    }

    /// Set/replace the policy gate used for private streams.
    pub fn with_policy(mut self, policy: PolicyGate) -> Self {
        self.policy = Some(policy);
        self
    }
}

impl SubscribeAuthorizer for DefaultAuthorizer {
    fn authorize(&self, peer: &PeerIdentity, track_name: &str) -> SubscribeDecision {
        match (self.visibility)(track_name) {
            Visibility::Public => SubscribeDecision::Allow,
            Visibility::Private => match &self.policy {
                // Fail-closed: a private stream with no policy gate is denied.
                None => SubscribeDecision::Deny,
                Some(gate) => {
                    if gate(peer, track_name) {
                        SubscribeDecision::Allow
                    } else {
                        SubscribeDecision::Deny
                    }
                }
            },
        }
    }
}

/// Shared handle to a [`SubscribeAuthorizer`] suitable for stashing on a
/// long-lived server/handler.
pub type SharedSubscribeAuthorizer = Arc<dyn SubscribeAuthorizer>;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tenant_of_extracts_first_segment() {
        assert_eq!(tenant_of("alice/streams/run-1/i0"), Some("alice"));
        assert_eq!(tenant_of("bob/svc/topic/inst"), Some("bob"));
        assert_eq!(tenant_of("solo"), Some("solo"));
        assert_eq!(tenant_of(""), None);
        assert_eq!(tenant_of("/leading"), None);
    }

    #[test]
    fn tenant_prefix_is_segment_significant() {
        assert_eq!(tenant_prefix("alice"), "alice/");
    }

    #[test]
    fn announce_filter_scopes_to_one_tenant() {
        let names = [
            "alice/streams/run-1/i0",
            "alice/streams/run-2/i0",
            "bob/streams/run-9/i0",
            "carol/streams/run-3/i0",
        ];
        let a = filter_announces_by_tenant(names.iter().copied(), "alice");
        assert_eq!(a, vec!["alice/streams/run-1/i0", "alice/streams/run-2/i0"]);
        // Tenant A sees ONLY A's names — not B's or C's.
        assert!(!a.iter().any(|n| n.starts_with("bob/")));
        assert!(!a.iter().any(|n| n.starts_with("carol/")));

        let b = filter_announces_by_tenant(names.iter().copied(), "bob");
        assert_eq!(b, vec!["bob/streams/run-9/i0"]);
    }

    #[test]
    fn announce_filter_is_segment_exact_no_prefix_bleed() {
        // "alice" must NOT match a sibling tenant "alicent".
        let names = ["alice/s/t/i", "alicent/s/t/i"];
        let a = filter_announces_by_tenant(names.iter().copied(), "alice");
        assert_eq!(a, vec!["alice/s/t/i"]);
    }

    #[test]
    fn announce_filter_empty_for_unknown_tenant() {
        let names = ["alice/s/t/i", "bob/s/t/i"];
        assert!(filter_announces_by_tenant(names.iter().copied(), "zzz").is_empty());
    }

    #[test]
    fn public_stream_is_allowed_even_anonymous() {
        let authz = DefaultAuthorizer::permissive();
        let decision = authz.authorize(&PeerIdentity::anonymous(), "alice/s/t/i");
        assert_eq!(decision, SubscribeDecision::Allow);
        assert!(decision.is_allowed());
    }

    #[test]
    fn private_stream_without_gate_is_denied() {
        // All-private classifier, no gate installed → fail-closed deny.
        let authz =
            DefaultAuthorizer::new(Arc::new(|_| Visibility::Private), None);
        let decision = authz.authorize(&PeerIdentity::authenticated("did:key:z6Mk..."), "alice/s/t/i");
        assert_eq!(decision, SubscribeDecision::Deny);
    }

    #[test]
    fn private_stream_authorized_peer_is_allowed() {
        // Private classifier + gate that allows a specific subject.
        let gate: PolicyGate = Arc::new(|peer: &PeerIdentity, _track: &str| {
            peer.subject.as_deref() == Some("alice-node")
        });
        let authz =
            DefaultAuthorizer::new(Arc::new(|_| Visibility::Private), Some(gate));

        let allowed =
            authz.authorize(&PeerIdentity::authenticated("alice-node"), "alice/s/t/i");
        assert_eq!(allowed, SubscribeDecision::Allow);
    }

    #[test]
    fn private_stream_unauthorized_peer_is_rejected() {
        let gate: PolicyGate = Arc::new(|peer: &PeerIdentity, _track: &str| {
            peer.subject.as_deref() == Some("alice-node")
        });
        let authz =
            DefaultAuthorizer::new(Arc::new(|_| Visibility::Private), Some(gate));

        // Different subject → denied.
        let denied =
            authz.authorize(&PeerIdentity::authenticated("mallory-node"), "alice/s/t/i");
        assert_eq!(denied, SubscribeDecision::Deny);

        // Anonymous peer on a private stream → denied (fail-closed).
        let anon = authz.authorize(&PeerIdentity::anonymous(), "alice/s/t/i");
        assert_eq!(anon, SubscribeDecision::Deny);
    }

    #[test]
    fn mixed_visibility_routes_public_open_private_gated() {
        // Public iff under "pub/" prefix; everything else private.
        let visibility: VisibilityFn = Arc::new(|track: &str| {
            if tenant_of(track) == Some("pub") {
                Visibility::Public
            } else {
                Visibility::Private
            }
        });
        let gate: PolicyGate = Arc::new(|peer: &PeerIdentity, _| {
            peer.subject.as_deref() == Some("priv-allowed")
        });
        let authz = DefaultAuthorizer::new(visibility, Some(gate));

        // Public stream: open even to anonymous.
        assert_eq!(
            authz.authorize(&PeerIdentity::anonymous(), "pub/s/t/i"),
            SubscribeDecision::Allow
        );
        // Private stream: authorized subject allowed.
        assert_eq!(
            authz.authorize(&PeerIdentity::authenticated("priv-allowed"), "alice/s/t/i"),
            SubscribeDecision::Allow
        );
        // Private stream: other subject denied.
        assert_eq!(
            authz.authorize(&PeerIdentity::authenticated("nope"), "alice/s/t/i"),
            SubscribeDecision::Deny
        );
    }

    #[test]
    fn peer_identity_constructors() {
        assert!(!PeerIdentity::anonymous().is_authenticated());
        assert!(PeerIdentity::authenticated("x").is_authenticated());
        assert_eq!(
            PeerIdentity::authenticated("x").subject.as_deref(),
            Some("x")
        );
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod scope_tests {
    use super::*;
    use moq_net::Origin;

    /// A tenant-scoped consumer sees ONLY its own tenant's broadcasts on the
    /// announce cursor — the live (network-side) analogue of
    /// [`filter_announces_by_tenant`]. Tenant A must not enumerate B's names.
    #[tokio::test]
    async fn tenant_scoped_consumer_filters_announces() {
        let producer = Origin::random().produce();
        let base = producer.consume();

        // Publish broadcasts for two tenants.
        let _a1 = producer.create_broadcast("alice/streams/run-1/i0").unwrap();
        let _a2 = producer.create_broadcast("alice/streams/run-2/i0").unwrap();
        let _b1 = producer.create_broadcast("bob/streams/run-9/i0").unwrap();

        // Scope a consumer to tenant "alice".
        let mut scoped =
            tenant_scoped_consumer(&base, "alice").expect("alice scope should exist");

        // Drain everything currently announced through the scoped cursor.
        // `OriginAnnounce` is `(path, Some(broadcast))` for an announce and
        // `(path, None)` for an unannounce; we only collect announces.
        let mut seen = Vec::new();
        while let Some((path, active)) = scoped.try_announced() {
            if active.is_some() {
                // path is absolute and always under the scoped prefix.
                seen.push(path.as_str().to_owned());
            }
        }
        seen.sort();

        assert_eq!(
            seen,
            vec![
                "alice/streams/run-1/i0".to_owned(),
                "alice/streams/run-2/i0".to_owned(),
            ]
        );
        // Crucially: bob's broadcast is NOT visible to the alice-scoped consumer.
        assert!(!seen.iter().any(|n| n.starts_with("bob/")));
    }

    /// A scoped consumer cannot subscribe to another tenant's broadcast even by
    /// exact path: `announced_broadcast` for an out-of-scope path returns None.
    #[tokio::test]
    async fn tenant_scoped_consumer_cannot_reach_other_tenant() {
        let producer = Origin::random().produce();
        let base = producer.consume();
        let _b1 = producer.create_broadcast("bob/streams/run-9/i0").unwrap();

        let scoped =
            tenant_scoped_consumer(&base, "alice").expect("alice scope should exist");

        // bob/... is outside alice's scope → not reachable.
        let reached = scoped.announced_broadcast("bob/streams/run-9/i0").await;
        assert!(reached.is_none(), "alice-scoped consumer must not reach bob's broadcast");
    }
}
