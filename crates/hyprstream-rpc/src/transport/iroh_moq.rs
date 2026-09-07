//! Iroh streaming plane — moq-lite (`moql` ALPN) protocol handler.
//!
//! Part of Epic #131 Phase 3 (#134). This module ships the wire layer for
//! the streaming plane:
//!
//! - [`IrohMoqProtocolHandler`] plugs into [`crate::transport::iroh_substrate`]
//!   under the `moql` ALPN, wrapping each accepted iroh `Connection` as a
//!   `web_transport_iroh::Session` and handing it to `moq_net::Server`,
//!   which spawns the moq-lite session machinery internally.
//! - One shared [`OriginShared`] (an `OriginProducer` + matching
//!   `OriginConsumer`) is held by the handler; the *same* origin is used
//!   for **in-process publishing** (callers obtain it via
//!   [`IrohMoqProtocolHandler::origin_producer`] and create broadcasts
//!   directly) **and for external subscribers** — per spike #94's finding,
//!   `moq_net::Server` does exactly this.
//!
//! **Trust model**: §7.5 unchanged. On the wire, subscribers learn the
//! DH-derived (unguessable) Track path out-of-band via authenticated RPC
//! (a signed `StreamInfo`), and the per-Frame payload carries the
//! chained-HMAC envelope. CDN-portability per §10.x of the Federated Agentic
//! Namespaces doc. Admission: without an authenticator the carrier is
//! unauthenticated and the accept path refuses anonymous peers; with a
//! [`crate::transport::moql_admission::MoqlAdmissionAuthenticator`] installed
//! (#1027), each connection must first prove an accepted current
//! Ed25519 + ML-DSA-65 identity inside the carrier, and is then served only
//! its own tenant's scope.
//!
//! **What this module does NOT do**: refactor `StreamService` or
//! `EventService` to use this. That lands in Phase 3 part 2+ along with
//! the §7.5 chained-HMAC tokenstream port.

use std::sync::Arc;

use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use moq_net::{Origin, OriginConsumer, OriginProducer, Server, StatsHandle};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use web_transport_iroh::Session;

use crate::moq_authz::{
    PeerIdentity, SharedSubscribeAuthorizer, SubscribeDecision, is_valid_tenant_segment,
    tenant_prefix,
    tenant_scoped_consumer,
};
use crate::transport::moql_admission::MoqlAdmissionAuthenticator;

/// Resolves the tenant for an independently authenticated application peer.
/// Carrier NodeId is never passed here. Without a fresh-proof seam (#1027)
/// the iroh accept path refuses before invoking this resolver; with a
/// [`MoqlAdmissionAuthenticator`] installed, the resolver runs inside the
/// admission exchange over the *verified* subject.
pub type PeerTenantResolver = Arc<dyn Fn(&PeerIdentity) -> Option<String> + Send + Sync>;

/// Trusted decision for granting an admitted peer ingress into a tenant's
/// shared origin. Admission proves the peer identity and tenant only; it does
/// not imply that the peer may publish. The default is therefore deny.
pub trait IngressAuthorizer: Send + Sync {
    /// Whether this already-admitted `peer` may publish to the supplied tenant.
    fn authorize_ingress(&self, peer: &PeerIdentity, tenant: &str) -> bool;
}

impl<F> IngressAuthorizer for F
where
    F: Fn(&PeerIdentity, &str) -> bool + Send + Sync,
{
    fn authorize_ingress(&self, peer: &PeerIdentity, tenant: &str) -> bool {
        self(peer, tenant)
    }
}

/// Shareable ingress authorization hook supplied by the owning service policy.
pub type SharedIngressAuthorizer = Arc<dyn IngressAuthorizer>;

/// #276 authorization config for a moq accept path: an optional subscribe
/// authorizer and an optional peer→tenant resolver for per-tenant announce
/// scoping. Absence of fresh application/session proof is always fail-closed;
/// these hooks cannot turn an anonymous carrier into an authenticated peer.
///
/// #1027: installing an [`MoqlAdmissionAuthenticator`] changes the posture —
/// the accept path runs the inside-carrier Ed25519 + ML-DSA-65
/// challenge/response before any moq machinery and serves only the admitted
/// peer's tenant scope. Without it, the anonymous fail-closed refusal below is
/// unchanged.
#[derive(Clone, Default)]
pub struct MoqAuthzConfig {
    /// Subscribe-time authorization hook (public-open / private-gated).
    pub authorizer: Option<SharedSubscribeAuthorizer>,
    /// Maps a peer identity to its tenant for per-tenant announce scoping.
    pub tenant_resolver: Option<PeerTenantResolver>,
    /// Explicit trusted producer/relay decision. Its absence is a deliberate
    /// deny: an admitted peer is a read-only subscriber, never a publisher.
    pub ingress_authorizer: Option<SharedIngressAuthorizer>,
    /// #1027 inside-carrier admission authenticator. When set, every accepted
    /// `moql` connection must prove an accepted current Ed25519 + ML-DSA-65
    /// identity before the moq handshake; the carrier NodeId alone is refused.
    pub admission: Option<Arc<MoqlAdmissionAuthenticator>>,
}

impl MoqAuthzConfig {
    /// Resolve the tenant for `peer`, if a resolver is configured.
    pub fn tenant_for(&self, peer: &PeerIdentity) -> Option<String> {
        self.tenant_resolver.as_ref().and_then(|r| r(peer))
    }

    /// Set/replace the subscribe authorizer.
    pub fn with_authorizer(mut self, authorizer: SharedSubscribeAuthorizer) -> Self {
        self.authorizer = Some(authorizer);
        self
    }

    /// Install the service-owned trusted ingress decision. This hook is kept
    /// separate from tenant admission: a subject→tenant row is not a producer
    /// role grant.
    pub fn with_ingress_authorizer(mut self, authorizer: SharedIngressAuthorizer) -> Self {
        self.ingress_authorizer = Some(authorizer);
        self
    }

    /// Install an optional service-owned ingress decision without converting an
    /// absent deployment policy into a grant.
    pub fn with_ingress_authorizer_option(
        mut self,
        authorizer: Option<SharedIngressAuthorizer>,
    ) -> Self {
        self.ingress_authorizer = authorizer;
        self
    }

    /// Whether the admitted peer has an explicit trusted ingress grant.
    /// Missing configuration fails closed to read-only delivery.
    pub fn authorizes_ingress(&self, peer: &PeerIdentity, tenant: &str) -> bool {
        self.ingress_authorizer
            .as_ref()
            .is_some_and(|authorizer| authorizer.authorize_ingress(peer, tenant))
    }

    /// Install the #1027 inside-carrier admission authenticator. The
    /// authenticator carries its own authoritative subject→tenant resolver;
    /// `tenant_resolver` is only consulted for peers authenticated by other
    /// means (none today on this path).
    pub fn with_admission(mut self, admission: Arc<MoqlAdmissionAuthenticator>) -> Self {
        self.admission = Some(admission);
        self
    }

    /// Gate an MoQ session while the transport cannot surface track names.
    ///
    /// Dormant (`None`) preserves the legacy pass-through. Once an authorizer
    /// is installed, its coarse admission decision is mandatory; current
    /// implementations deny until moq-net's #276 per-track callback exists.
    pub fn authorize_without_track_hook(&self, peer: &PeerIdentity) -> SubscribeDecision {
        self.authorizer
            .as_ref()
            .map_or(SubscribeDecision::Allow, |a| {
                a.authorize_without_track_hook(peer)
            })
    }
}

/// Shared `Origin` clone-pair held by the handler.
///
/// - `producer` is what *we* publish into (call `create_broadcast`, `create_track`, etc.).
/// - `consumer` is what *remote subscribers* read from (handed to `Server::with_publish`).
///
/// Both reference the same underlying tree — broadcasts created via
/// `producer` are visible to remote subscribers consuming via `consumer`.
#[derive(Clone)]
pub struct OriginShared {
    producer: OriginProducer,
    consumer: OriginConsumer,
}

impl OriginShared {
    /// Build a fresh origin pair with a random id.
    pub fn new() -> Self {
        let producer = Origin::random().produce();
        let consumer = producer.consume();
        Self { producer, consumer }
    }

    /// Wrap an existing producer/consumer pair (e.g. the process-global moq
    /// origin) so the iroh `moql` accept path serves the SAME broadcasts the
    /// quinn `/moq` path does (#282 parallel bind).
    pub fn from_pair(producer: OriginProducer, consumer: OriginConsumer) -> Self {
        Self { producer, consumer }
    }

    pub fn producer(&self) -> &OriginProducer {
        &self.producer
    }

    pub fn consumer(&self) -> &OriginConsumer {
        &self.consumer
    }
}

impl Default for OriginShared {
    fn default() -> Self {
        Self::new()
    }
}

/// iroh `ProtocolHandler` for the `moql` ALPN. Each accepted connection is
/// wrapped as a `web_transport_iroh::Session` and handed to
/// `moq_net::Server::accept`, which performs the moq handshake and spawns
/// the session machinery.
#[derive(Clone)]
pub struct IrohMoqProtocolHandler {
    inner: Arc<HandlerInner>,
}

struct HandlerInner {
    origin: OriginShared,
    stats: StatsHandle,
    /// #276 subscribe-authz + per-tenant announce scoping config. Defaults to
    /// "off" (open same-tenant subscribe preserved).
    authz: MoqAuthzConfig,
    /// Caps accepted `moql` connections before their admission exchange. The
    /// permit lives until the connection accept task exits.
    connection_limit: Arc<Semaphore>,
    /// Triggered by `ProtocolHandler::shutdown` so accept handlers stop
    /// waiting for `Session::closed()` and exit promptly.
    shutdown: CancellationToken,
}

impl std::fmt::Debug for IrohMoqProtocolHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohMoqProtocolHandler")
            .finish_non_exhaustive()
    }
}

impl IrohMoqProtocolHandler {
    /// Build with a fresh origin pair (use [`Self::origin_producer`] to
    /// publish broadcasts).
    pub fn new() -> Self {
        Self::with_origin(OriginShared::new())
    }

    /// Build with a caller-supplied origin pair. Useful when one origin is
    /// shared across multiple substrates or with an existing service.
    pub fn with_origin(origin: OriginShared) -> Self {
        Self {
            inner: Arc::new(HandlerInner {
                origin,
                stats: StatsHandle::default(),
                authz: MoqAuthzConfig::default(),
                connection_limit: Arc::new(Semaphore::new(
                    super::rpc_session::DEFAULT_CONNECTION_LIMIT,
                )),
                shutdown: CancellationToken::new(),
            }),
        }
    }

    /// Install the #276 subscribe-authz + per-tenant announce-scoping config.
    ///
    /// With an [`MoqlAdmissionAuthenticator`] in the config (#1027), the accept
    /// path runs the inside-carrier challenge/response first and the admitted
    /// peer's tenant scope is served. Without one, no fresh-proof seam supplies
    /// an authenticated [`PeerIdentity`] and the carrier NodeId alone never
    /// reaches the resolver.
    pub fn with_authz(mut self, authz: MoqAuthzConfig) -> Self {
        self.rebuild_inner(|i| i.authz = authz);
        self
    }

    /// Query the installed ingress decision. This is intentionally the same
    /// decision the accept path uses when choosing its scoped writable origin.
    /// It keeps service-spawner wiring testable without exposing handler state.
    pub fn authorizes_ingress(&self, peer: &PeerIdentity, tenant: &str) -> bool {
        self.inner.authz.authorizes_ingress(peer, tenant)
    }

    /// Override the server-wide accepted-connection cap. Connections beyond
    /// the cap are dropped instead of waiting for admission.
    pub fn with_connection_limit(mut self, connection_limit: usize) -> Self {
        self.rebuild_inner(|i| i.connection_limit = Arc::new(Semaphore::new(connection_limit)));
        self
    }

    /// Mutate the inner config, cloning the shared `Arc<HandlerInner>` only when
    /// it is already shared (cloned handler) so builder calls compose without
    /// dropping previously-installed fields (authz / admission).
    fn rebuild_inner(&mut self, f: impl FnOnce(&mut HandlerInner)) {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            f(inner);
        } else {
            let old = &*self.inner;
            let mut cloned = HandlerInner {
                origin: old.origin.clone(),
                stats: old.stats.clone(),
                authz: old.authz.clone(),
                connection_limit: Arc::clone(&old.connection_limit),
                shutdown: old.shutdown.clone(),
            };
            f(&mut cloned);
            self.inner = Arc::new(cloned);
        }
    }

    /// Borrow the shared origin pair.
    pub fn origin(&self) -> &OriginShared {
        &self.inner.origin
    }

    /// Borrow the producer (for in-process publishing).
    pub fn origin_producer(&self) -> &OriginProducer {
        self.inner.origin.producer()
    }

    /// Borrow the consumer (for in-process subscribing — same data as a
    /// remote subscriber sees on the wire).
    pub fn origin_consumer(&self) -> &OriginConsumer {
        self.inner.origin.consumer()
    }
}

impl Default for IrohMoqProtocolHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ProtocolHandler for IrohMoqProtocolHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        // Iroh's Router creates an accept task per carrier connection. Acquire
        // before awaiting the #1027 challenge/response so unauthenticated peers
        // cannot retain unbounded tasks for the admission timeout. Keep this
        // permit through the admitted session's lifetime.
        let _connection_permit = match Arc::clone(&self.inner.connection_limit).try_acquire_owned()
        {
            Ok(permit) => permit,
            Err(_) => {
                tracing::warn!("iroh-moq: connection cap reached, rejecting connection");
                conn.close(0u32.into(), b"moql connection cap reached");
                return Ok(());
            }
        };
        // #1027: with an admission authenticator installed, the peer must
        // complete the fresh inside-carrier Ed25519 + ML-DSA-65
        // challenge/response BEFORE any moq machinery runs. The exchange binds
        // the accepted-state epoch/head and both nonces into the transcript;
        // replay, expiry, rotation, cross-tenant resolution failure, and
        // NodeId-only identity all reject here, fail-closed.
        let admitted = match &self.inner.authz.admission {
            Some(admission) => match admission.accept(&conn).await {
                Ok(admitted) => {
                    tracing::debug!(
                        subject = %admitted.peer.subject.as_deref().unwrap_or("?"),
                        tenant = %admitted.tenant,
                        epoch = admitted.epoch,
                        carrier = %hex::encode(&admitted.carrier_node_id[..4]),
                        "iroh-moq: inside-carrier admission succeeded"
                    );
                    Some(admitted)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "iroh-moq: admission rejected; closing carrier");
                    conn.close(0u32.into(), b"moql admission rejected");
                    return Ok(());
                }
            },
            None => None,
        };
        // Carrier NodeId is transport metadata only. Without #1027 admission
        // proof, authorization sees an anonymous peer.
        let peer = admitted
            .as_ref()
            .map(|a| a.peer.clone())
            .unwrap_or_else(PeerIdentity::anonymous);
        if !self
            .inner
            .authz
            .authorize_without_track_hook(&peer)
            .is_allowed()
        {
            tracing::warn!(
                "iroh-moq: installed track authorizer denied session because #276 callback is unavailable"
            );
            conn.close(0u32.into(), b"per-track authorization unavailable");
            return Ok(());
        }
        if !peer.is_authenticated() {
            tracing::warn!(
                "iroh-moq: refusing anonymous carrier pending verified session proof (#726)"
            );
            conn.close(0u32.into(), b"verified MoQ session proof required");
            return Ok(());
        }

        // Authenticated-session path: hand the admitted peer only a consumer
        // narrowed to its own `{tenant}/` prefix. The tenant comes from the
        // admission exchange (server-resolved from the verified subject); the
        // legacy resolver seam remains for any future proof source. Missing
        // tenant resolution must remain fail-closed; it must never fall back
        // to the process-global consumer.
        let resolved_tenant = match &admitted {
            Some(a) => Some(a.tenant.clone()),
            None => self.inner.authz.tenant_for(&peer),
        };
        let tenant = match resolved_tenant {
            Some(tenant) => tenant,
            None => {
                tracing::warn!("iroh-moq: authenticated peer has no tenant scope; refusing");
                conn.close(0u32.into(), b"tenant scope required");
                return Ok(());
            }
        };
        if !is_valid_tenant_segment(&tenant) {
            tracing::warn!(%tenant, "iroh-moq: invalid tenant scope; refusing");
            conn.close(0u32.into(), b"valid tenant scope required");
            return Ok(());
        }

        // An admitted session is a tenant-scoped *subscriber* by default.
        // Admission proves identity and tenant; it is not a producer/relay
        // capability. Only a separate service-owned ingress decision may hand
        // the peer a scoped writable origin. This keeps an ordinary subscriber
        // from colliding with a producer's broadcast names.
        let ingress_granted = self.inner.authz.authorizes_ingress(&peer, &tenant);
        let server = if ingress_granted {
            let prefix = tenant_prefix(&tenant);
            let path = moq_net::Path::new(&prefix);
            let Some(scoped_origin) = self.inner.origin.producer.scope(&[path]) else {
                tracing::debug!(%tenant, "iroh-moq: tenant has no visible relay scope");
                return Ok(());
            };
            tracing::debug!(
                subject = %peer.subject.as_deref().unwrap_or("?"),
                %tenant,
                "iroh-moq: admitting explicitly authorized ingress"
            );
            Server::new().with_origin(scoped_origin)
        } else {
            let Some(scoped_consumer) = tenant_scoped_consumer(self.inner.origin.consumer(), &tenant) else {
                tracing::debug!(%tenant, "iroh-moq: tenant has no visible subscriber scope");
                return Ok(());
            };
            Server::new().with_publish(scoped_consumer)
        }
        .with_stats(self.inner.stats.clone());
        let session_conn = conn.clone();
        let session = Session::raw(conn);
        let moq_session = server
            .accept(session)
            .await
            .map_err(AcceptError::from_err)?;

        // Admission is stateful rather than a one-time capability. Poll the
        // same daemon-owned authority used by the challenge exchange while the
        // MoQ session is alive; expiry, an accepted-head advance, or a rotated
        // key closes the carrier and drops the scoped session immediately.
        let admission = self.inner.authz.admission.clone();
        let admitted_for_session = admitted.clone();
        let mut currentness = tokio::time::interval(std::time::Duration::from_millis(100));

        // Hold the session alive until either the connection closes or
        // shutdown is requested. Server::accept has already spawned the
        // session's pump tasks; dropping `moq_session` tears them down.
        loop {
            tokio::select! {
                biased;
                _ = self.inner.shutdown.cancelled() => {
                    tracing::debug!("iroh-moq: shutdown signalled, dropping session");
                    break;
                }
                _ = currentness.tick() => {
                    if let (Some(admission), Some(admitted)) =
                        (admission.as_ref(), admitted_for_session.as_ref())
                    {
                        if !admission.is_still_current(admitted) {
                            tracing::warn!(
                                subject = %admitted.peer.subject.as_deref().unwrap_or("?"),
                                "iroh-moq: accepted state changed or expired; closing session"
                            );
                            session_conn.close(0u32.into(), b"moql accepted state no longer current");
                            break;
                        }
                    }
                    // Ingress is live deployment authority, not a one-time
                    // capability: a revoked grant must not keep announcing on
                    // the writable scoped origin this session already holds.
                    if ingress_granted && !self.inner.authz.authorizes_ingress(&peer, &tenant) {
                        tracing::warn!(
                            subject = %peer.subject.as_deref().unwrap_or("?"),
                            %tenant,
                            "iroh-moq: ingress grant revoked; closing session"
                        );
                        session_conn.close(0u32.into(), b"moql ingress grant revoked");
                        break;
                    }
                }
                res = moq_session.closed() => {
                    tracing::debug!(result = ?res, "iroh-moq: session closed");
                    break;
                }
            }
        }
        Ok(())
    }

    async fn shutdown(&self) {
        self.inner.shutdown.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::iroh_substrate::{ALPN_MOQ_LITE, IrohSubstrate, NoopHandler};
    use bytes::Bytes;
    use ed25519_dalek::SigningKey;
    use iroh::{EndpointAddr, TransportAddr};
    use moq_net::{Client, Group, Track};
    use rand::RngCore;

    fn fresh_key() -> [u8; 32] {
        let mut k = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut k);
        k
    }

    fn direct_addr(substrate: &IrohSubstrate) -> EndpointAddr {
        EndpointAddr::from_parts(
            substrate.endpoint_id(),
            substrate
                .endpoint()
                .bound_sockets()
                .into_iter()
                .map(TransportAddr::Ip),
        )
    }

    /// An anonymous carrier cannot subscribe to server broadcasts, publish a
    /// client origin, or reach the tenant resolver. Mutating the carrier NodeId
    /// cannot change that decision because it is never an application proof.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn anonymous_carrier_cannot_publish_subscribe_or_obtain_tenant_scope()
    -> anyhow::Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let resolver_calls = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&resolver_calls);
        let authz = MoqAuthzConfig {
            authorizer: None,
            tenant_resolver: Some(Arc::new(move |_peer| {
                calls.fetch_add(1, Ordering::SeqCst);
                Some("alice".to_owned())
            })),
            ingress_authorizer: None,
            admission: None,
        };
        let handler = IrohMoqProtocolHandler::new().with_authz(authz);
        let producer = handler.origin_producer().clone();
        let server_consumer = handler.origin_consumer().clone();
        let server =
            IrohSubstrate::new_test(fresh_key(), handler, NoopHandler::new("rpc-not-wired"))
                .await?;
        let server_addr = direct_addr(&server);

        // A server-side broadcast would be exposed if anonymous subscribe were
        // still open.
        let mut broadcast = producer
            .create_broadcast("alice/run-1")
            .ok_or_else(|| anyhow::anyhow!("create_broadcast denied"))?;
        let mut track = broadcast.create_track(Track::new("tokens"))?;
        let mut group = track.create_group(Group::from(0u64))?;
        group.write_frame(Bytes::from_static(b"hello-moq"))?;
        drop(group);

        // A client-side origin would be exposed if anonymous publish were open.
        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let conn = client.connect(server_addr, ALPN_MOQ_LITE).await?;
        let session = Session::raw(conn);
        let client_origin: OriginProducer = Origin::random().produce();
        let client_consumer: OriginConsumer = client_origin.consume();
        let mut attacker_broadcast = client_origin
            .create_broadcast("mallory/injected")
            .ok_or_else(|| anyhow::anyhow!("create attacker broadcast"))?;
        let mut attacker_track = attacker_broadcast.create_track(Track::new("tokens"))?;
        let mut attacker_group = attacker_track.create_group(Group::from(0u64))?;
        attacker_group.write_frame(Bytes::from_static(b"attacker-data"))?;
        drop(attacker_group);
        let moq_client = Client::new()
            .with_origin(client_origin.clone())
            .with_consume(client_origin);
        let handshake = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            moq_client.connect(session),
        )
        .await;
        assert!(
            matches!(handshake, Ok(Err(_))),
            "anonymous MoQ handshake must be explicitly rejected, not succeed or time out"
        );

        let server_seen_attacker = tokio::time::timeout(
            std::time::Duration::from_millis(300),
            server_consumer.announced_broadcast("mallory/injected"),
        )
        .await;
        assert!(
            server_seen_attacker.is_err() || server_seen_attacker.ok().flatten().is_none(),
            "anonymous carrier must not publish into the server origin"
        );
        let client_seen_server = tokio::time::timeout(
            std::time::Duration::from_millis(300),
            client_consumer.announced_broadcast("alice/run-1"),
        )
        .await;
        assert!(
            client_seen_server.is_err() || client_seen_server.ok().flatten().is_none(),
            "anonymous carrier must not subscribe to the server origin"
        );
        assert_eq!(
            resolver_calls.load(Ordering::SeqCst),
            0,
            "anonymous carrier must not obtain tenant scope"
        );

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    /// A real mutually admitted producer, relay, and distinct subscriber carry
    /// a frame end-to-end. The subscriber is read-only, while trusted ingress
    /// remains both subject- and tenant-scoped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn trusted_ingress_relays_to_distinct_subscriber_and_scopes_tenants()
    -> anyhow::Result<()> {
        use crate::crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes};
        use crate::stream_info::MoqlServerIdentity;
        use crate::transport::moql_admission::{AcceptedIdentityState, AcceptedSubjectKey, MoqlAdmissionProof, MoqlServerIdentityProof, prove_moql_admission};

        let server_ed = SigningKey::from_bytes(&[0x71; 32]);
        let server_pq = ml_dsa_sk_from_seed(&[0x72; 32]);
        let producer_ed = SigningKey::from_bytes(&[0x73; 32]);
        let producer_pq = ml_dsa_sk_from_seed(&[0x74; 32]);
        let subscriber_ed = SigningKey::from_bytes(&[0x77; 32]);
        let subscriber_pq = ml_dsa_sk_from_seed(&[0x78; 32]);
        let other_ed = SigningKey::from_bytes(&[0x79; 32]);
        let other_pq = ml_dsa_sk_from_seed(&[0x7A; 32]);
        let expiry = crate::envelope::current_timestamp() + 60_000;
        let server_identity = MoqlServerIdentity {
            did: "did:at9p:relay".to_owned(), epoch: 1, head_digest: vec![0x75; 64],
            expires_at_unix_ms: expiry, ed25519: server_ed.verifying_key().to_bytes(),
            ml_dsa65: ml_dsa_sk_to_vk_bytes(&server_pq),
        };
        let server_state = AcceptedIdentityState {
            epoch: 1, head_digest: [0x75; 64], expires_at_unix_ms: Some(expiry),
            subject_keys: vec![AcceptedSubjectKey { ed25519: server_ed.verifying_key().to_bytes(), ml_dsa_65: ml_dsa_sk_to_vk_bytes(&server_pq) }],
        };
        let producer_state = AcceptedIdentityState {
            epoch: 1, head_digest: [0x76; 64], expires_at_unix_ms: Some(expiry),
            subject_keys: vec![AcceptedSubjectKey { ed25519: producer_ed.verifying_key().to_bytes(), ml_dsa_65: ml_dsa_sk_to_vk_bytes(&producer_pq) }],
        };
        let subscriber_state = AcceptedIdentityState {
            epoch: 1, head_digest: [0x77; 64], expires_at_unix_ms: Some(expiry),
            subject_keys: vec![AcceptedSubjectKey { ed25519: subscriber_ed.verifying_key().to_bytes(), ml_dsa_65: ml_dsa_sk_to_vk_bytes(&subscriber_pq) }],
        };
        let other_state = AcceptedIdentityState {
            epoch: 1, head_digest: [0x78; 64], expires_at_unix_ms: Some(expiry),
            subject_keys: vec![AcceptedSubjectKey { ed25519: other_ed.verifying_key().to_bytes(), ml_dsa_65: ml_dsa_sk_to_vk_bytes(&other_pq) }],
        };
        let authority: Arc<dyn crate::transport::moql_admission::AcceptedStateAuthority> = Arc::new(move |did: &str| match did {
            "did:at9p:relay" => Some(server_state.clone()),
            "did:at9p:producer" => Some(producer_state.clone()),
            "did:at9p:subscriber" => Some(subscriber_state.clone()),
            "did:at9p:other-producer" => Some(other_state.clone()),
            _ => None,
        });
        let relay_secret = fresh_key();
        let relay_carrier = *iroh::SecretKey::from_bytes(&relay_secret).public().as_bytes();
        let admission = Arc::new(crate::transport::moql_admission::MoqlAdmissionAuthenticator::new(
            authority,
            Arc::new(|peer| match peer.subject.as_deref() {
                Some("did:at9p:producer") | Some("did:at9p:subscriber") => Some("alice".to_owned()),
                Some("did:at9p:other-producer") => Some("bob".to_owned()),
                _ => None,
            }),
        ).with_server_identity_and_carrier(MoqlServerIdentityProof {
            identity: server_identity.clone(), ed25519: server_ed, ml_dsa_65: server_pq,
        }, relay_carrier));
        let handler = IrohMoqProtocolHandler::new().with_authz(
            MoqAuthzConfig::default()
                .with_admission(admission)
                .with_ingress_authorizer(Arc::new(|peer: &PeerIdentity, tenant: &str| {
                    matches!(peer.subject.as_deref(), Some("did:at9p:producer") | Some("did:at9p:other-producer"))
                        && matches!(tenant, "alice" | "bob")
                })),
        );
        let relay_consumer = handler.origin_consumer().clone();
        let relay = IrohSubstrate::new_test(relay_secret, handler, NoopHandler::new("rpc-not-wired")).await?;
        let producer = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("producer-moq"), NoopHandler::new("producer-rpc")).await?;
        let subscriber = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("subscriber-moq"), NoopHandler::new("subscriber-rpc")).await?;
        let other = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("other-moq"), NoopHandler::new("other-rpc")).await?;

        let producer_conn = producer.connect(direct_addr(&relay), ALPN_MOQ_LITE).await?;
        prove_moql_admission(&producer_conn, &MoqlAdmissionProof {
            did: "did:at9p:producer".to_owned(), ed25519: producer_ed, ml_dsa_65: producer_pq,
            expected_server: server_identity.clone(),
        }, *producer.endpoint_id().as_bytes(), std::time::Duration::from_secs(2)).await?;
        let producer_origin: OriginProducer = Origin::random().produce();
        let producer_session = Client::new().with_origin(producer_origin.clone()).connect(Session::raw(producer_conn)).await?;

        let subscriber_conn = subscriber.connect(direct_addr(&relay), ALPN_MOQ_LITE).await?;
        prove_moql_admission(&subscriber_conn, &MoqlAdmissionProof {
            did: "did:at9p:subscriber".to_owned(), ed25519: subscriber_ed, ml_dsa_65: subscriber_pq,
            expected_server: server_identity.clone(),
        }, *subscriber.endpoint_id().as_bytes(), std::time::Duration::from_secs(2)).await?;
        let subscriber_origin: OriginProducer = Origin::random().produce();
        let subscriber_consumer = subscriber_origin.consume();
        let subscriber_session = Client::new()
            .with_origin(subscriber_origin.clone())
            .with_consume(subscriber_origin.clone())
            .connect(Session::raw(subscriber_conn))
            .await?;

        let broadcast_name = "alice/from-producer";
        let mut producer_broadcast = producer_origin.create_broadcast(broadcast_name).ok_or_else(|| anyhow::anyhow!("create producer broadcast"))?;
        let mut producer_track = producer_broadcast.create_track(Track::new("tokens"))?;
        let mut producer_group = producer_track.create_group(Group::from(0u64))?;
        producer_group.write_frame(Bytes::from_static(b"producer frame"))?;
        drop(producer_group);
        tokio::time::timeout(std::time::Duration::from_secs(2), relay_consumer.announced_broadcast(broadcast_name)).await?.ok_or_else(|| anyhow::anyhow!("relay did not ingest trusted producer origin"))?;

        let broadcast = tokio::time::timeout(std::time::Duration::from_secs(2), subscriber_consumer.announced_broadcast(broadcast_name)).await?.ok_or_else(|| anyhow::anyhow!("distinct subscriber did not receive relay announcement"))?;
        let track = broadcast.subscribe_track(&Track::new("tokens"))?;
        let mut group = tokio::time::timeout(std::time::Duration::from_secs(2), track.get_group(0)).await??.ok_or_else(|| anyhow::anyhow!("subscriber track ended before producer frame"))?;
        assert_eq!(group.read_frame().await?, Some(Bytes::from_static(b"producer frame")));

        // An authenticated subscriber is read-only even when it offers an
        // origin: it cannot inject a same-tenant broadcast into the relay.
        let _subscriber_injection = subscriber_origin
            .create_broadcast("alice/subscriber-injected")
            .ok_or_else(|| anyhow::anyhow!("create subscriber injection"))?;
        assert!(tokio::time::timeout(std::time::Duration::from_millis(300), relay_consumer.announced_broadcast("alice/subscriber-injected")).await.is_err());

        // A separately trusted bob producer can connect, but its scoped origin
        // cannot ingest into alice and its scoped consumer cannot observe alice.
        let other_conn = other.connect(direct_addr(&relay), ALPN_MOQ_LITE).await?;
        prove_moql_admission(&other_conn, &MoqlAdmissionProof {
            did: "did:at9p:other-producer".to_owned(), ed25519: other_ed, ml_dsa_65: other_pq,
            expected_server: server_identity,
        }, *other.endpoint_id().as_bytes(), std::time::Duration::from_secs(2)).await?;
        let other_origin: OriginProducer = Origin::random().produce();
        let other_consumer = other_origin.consume();
        let other_session = Client::new()
            .with_origin(other_origin.clone())
            .with_consume(other_origin.clone())
            .connect(Session::raw(other_conn))
            .await?;
        let _cross_tenant_injection = other_origin
            .create_broadcast("alice/cross-tenant-injected")
            .ok_or_else(|| anyhow::anyhow!("create cross-tenant injection"))?;
        assert!(tokio::time::timeout(std::time::Duration::from_millis(300), relay_consumer.announced_broadcast("alice/cross-tenant-injected")).await.is_err());
        assert!(tokio::time::timeout(std::time::Duration::from_millis(300), other_consumer.announced_broadcast(broadcast_name)).await.is_err());

        drop(other_session);
        drop(subscriber_session);
        drop(producer_session);
        other.shutdown().await?;
        subscriber.shutdown().await?;
        producer.shutdown().await?;
        relay.shutdown().await?;
        Ok(())
    }

    /// Ingress is live deployment authority, not a one-time capability: when
    /// the grant is revoked mid-session, the currentness watchdog closes the
    /// session that already holds the writable scoped origin, so no further
    /// announcement is carried on that connection without any reconnect.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn revoked_ingress_closes_live_producer_session() -> anyhow::Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        use crate::crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes};
        use crate::stream_info::MoqlServerIdentity;
        use crate::transport::moql_admission::{AcceptedIdentityState, AcceptedSubjectKey, MoqlAdmissionProof, MoqlServerIdentityProof, prove_moql_admission};

        let server_ed = SigningKey::from_bytes(&[0x81; 32]);
        let server_pq = ml_dsa_sk_from_seed(&[0x82; 32]);
        let producer_ed = SigningKey::from_bytes(&[0x83; 32]);
        let producer_pq = ml_dsa_sk_from_seed(&[0x84; 32]);
        let expiry = crate::envelope::current_timestamp() + 60_000;
        let server_identity = MoqlServerIdentity {
            did: "did:at9p:relay".to_owned(), epoch: 1, head_digest: vec![0x85; 64],
            expires_at_unix_ms: expiry, ed25519: server_ed.verifying_key().to_bytes(),
            ml_dsa65: ml_dsa_sk_to_vk_bytes(&server_pq),
        };
        let server_state = AcceptedIdentityState {
            epoch: 1, head_digest: [0x85; 64], expires_at_unix_ms: Some(expiry),
            subject_keys: vec![AcceptedSubjectKey { ed25519: server_ed.verifying_key().to_bytes(), ml_dsa_65: ml_dsa_sk_to_vk_bytes(&server_pq) }],
        };
        let producer_state = AcceptedIdentityState {
            epoch: 1, head_digest: [0x86; 64], expires_at_unix_ms: Some(expiry),
            subject_keys: vec![AcceptedSubjectKey { ed25519: producer_ed.verifying_key().to_bytes(), ml_dsa_65: ml_dsa_sk_to_vk_bytes(&producer_pq) }],
        };
        let authority: Arc<dyn crate::transport::moql_admission::AcceptedStateAuthority> = Arc::new(move |did: &str| match did {
            "did:at9p:relay" => Some(server_state.clone()),
            "did:at9p:producer" => Some(producer_state.clone()),
            _ => None,
        });
        let relay_secret = fresh_key();
        let relay_carrier = *iroh::SecretKey::from_bytes(&relay_secret).public().as_bytes();
        let admission = Arc::new(crate::transport::moql_admission::MoqlAdmissionAuthenticator::new(
            authority,
            Arc::new(|peer| (peer.subject.as_deref() == Some("did:at9p:producer")).then(|| "alice".to_owned())),
        ).with_server_identity_and_carrier(MoqlServerIdentityProof {
            identity: server_identity.clone(), ed25519: server_ed, ml_dsa_65: server_pq,
        }, relay_carrier));
        // Mutable policy backing the ingress decision; the test flips
        // `grant` mid-session to model a grant revocation.
        let grant = Arc::new(AtomicBool::new(true));
        let grant_for_authz = Arc::clone(&grant);
        let handler = IrohMoqProtocolHandler::new().with_authz(
            MoqAuthzConfig::default()
                .with_admission(admission)
                .with_ingress_authorizer(Arc::new(move |peer: &PeerIdentity, tenant: &str| {
                    grant_for_authz.load(Ordering::SeqCst)
                        && peer.subject.as_deref() == Some("did:at9p:producer")
                        && tenant == "alice"
                })),
        );
        let relay_consumer = handler.origin_consumer().clone();
        let relay = IrohSubstrate::new_test(relay_secret, handler, NoopHandler::new("rpc-not-wired")).await?;
        let producer = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("producer-moq"), NoopHandler::new("producer-rpc")).await?;

        let producer_conn = producer.connect(direct_addr(&relay), ALPN_MOQ_LITE).await?;
        prove_moql_admission(&producer_conn, &MoqlAdmissionProof {
            did: "did:at9p:producer".to_owned(), ed25519: producer_ed, ml_dsa_65: producer_pq,
            expected_server: server_identity.clone(),
        }, *producer.endpoint_id().as_bytes(), std::time::Duration::from_secs(2)).await?;
        let producer_origin: OriginProducer = Origin::random().produce();
        let producer_session = Client::new().with_origin(producer_origin.clone()).connect(Session::raw(producer_conn)).await?;

        // Granted: the producer's announce is carried end-to-end.
        let mut first = producer_origin
            .create_broadcast("alice/first")
            .ok_or_else(|| anyhow::anyhow!("create first broadcast"))?;
        let mut track = first.create_track(Track::new("tokens"))?;
        let mut group = track.create_group(Group::from(0u64))?;
        group.write_frame(Bytes::from_static(b"first frame"))?;
        drop(group);
        tokio::time::timeout(std::time::Duration::from_secs(2), relay_consumer.announced_broadcast("alice/first"))
            .await?
            .ok_or_else(|| anyhow::anyhow!("relay did not ingest the granted producer origin"))?;

        // Revoke the grant. The writable scoped origin is already in the
        // peer's hands; only the live-session recheck can stop it.
        grant.store(false, Ordering::SeqCst);

        // The same connection's session is closed by the watchdog; the peer
        // cannot carry any further announcement on it. Pinned moq-net
        // (07f558f, rs/moq-net/src/session.rs:81-84) yields
        // `Err(Error::Transport(..))` from `closed()` unconditionally, with
        // the transport close reason inside; the session resolving at all IS
        // the close — that is the success condition here.
        let close = tokio::time::timeout(std::time::Duration::from_secs(5), producer_session.closed())
            .await
            .map_err(|_| anyhow::anyhow!("revoked ingress did not close the live producer session"))?;
        match close {
            Err(moq_net::Error::Transport(_)) => {}
            other => return Err(anyhow::anyhow!("unexpected session close outcome: {other:?}")),
        }
        let _late = producer_origin
            .create_broadcast("alice/after-revoke")
            .ok_or_else(|| anyhow::anyhow!("create post-revocation broadcast"))?;
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), relay_consumer.announced_broadcast("alice/after-revoke")).await.is_err(),
            "a revoked producer must not announce further broadcasts on the same connection"
        );

        drop(first);
        drop(track);
        drop(producer_session);
        producer.shutdown().await?;
        relay.shutdown().await?;
        Ok(())
    }

    /// The connection cap is acquired before #1027 admission awaits the peer's
    /// first stream, is not queued for a saturated peer, and is returned when
    /// the stalled carrier closes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admission_connection_cap_saturates_and_releases() -> anyhow::Result<()> {
        use crate::crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes};
        use crate::stream_info::MoqlServerIdentity;
        use crate::transport::moql_admission::{
            AcceptedIdentityState, AcceptedSubjectKey, MoqlServerIdentityProof,
        };

        let server_ed = SigningKey::from_bytes(&[0xA1; 32]);
        let server_pq = ml_dsa_sk_from_seed(&[0xA2; 32]);
        let server_did = "did:at9p:connection-cap-server".to_owned();
        let server_identity = MoqlServerIdentity {
            did: server_did.clone(),
            epoch: 1,
            head_digest: vec![0xA3; 64],
            expires_at_unix_ms: crate::envelope::current_timestamp() + 60_000,
            ed25519: server_ed.verifying_key().to_bytes(),
            ml_dsa65: ml_dsa_sk_to_vk_bytes(&server_pq),
        };
        let mut head_digest = [0u8; 64];
        head_digest.copy_from_slice(&server_identity.head_digest);
        let accepted_server = AcceptedIdentityState {
            epoch: server_identity.epoch,
            head_digest,
            subject_keys: vec![AcceptedSubjectKey {
                ed25519: server_identity.ed25519,
                ml_dsa_65: server_identity.ml_dsa65.clone(),
            }],
            expires_at_unix_ms: Some(server_identity.expires_at_unix_ms),
        };
        let authority: Arc<dyn crate::transport::moql_admission::AcceptedStateAuthority> =
            Arc::new(move |did: &str| (did == server_did).then(|| accepted_server.clone()));

        let server_secret = fresh_key();
        let server_carrier = *iroh::SecretKey::from_bytes(&server_secret).public().as_bytes();
        let admission = Arc::new(
            crate::transport::moql_admission::MoqlAdmissionAuthenticator::new(
                authority,
                Arc::new(|_peer| None),
            )
            .with_server_identity_and_carrier(MoqlServerIdentityProof {
                identity: server_identity,
                ed25519: server_ed,
                ml_dsa_65: server_pq,
            }, server_carrier),
        );
        let handler = IrohMoqProtocolHandler::new()
            .with_authz(MoqAuthzConfig::default().with_admission(admission))
            .with_connection_limit(1);
        let server = IrohSubstrate::new_test(
            server_secret,
            handler.clone(),
            NoopHandler::new("rpc-not-wired"),
        )
        .await?;
        let server_addr = direct_addr(&server);
        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("client-moq"),
            NoopHandler::new("client-rpc"),
        )
        .await?;

        // This carrier never opens the admission stream, leaving the server in
        // the bounded admission wait while holding its one connection permit.
        let first = client.connect(server_addr.clone(), ALPN_MOQ_LITE).await?;
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while handler.inner.connection_limit.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| {
            anyhow::anyhow!("first stalled admission did not acquire connection permit")
        })?;

        // A second connection reaches iroh's carrier handshake but cannot add a
        // second admission task; the handler rejects it without queueing.
        let second = client.connect(server_addr.clone(), ALPN_MOQ_LITE).await?;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            handler.inner.connection_limit.available_permits(),
            0,
            "saturated admission must retain only the first connection permit"
        );
        second.close(0u32.into(), b"test complete");

        first.close(0u32.into(), b"test complete");
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while handler.inner.connection_limit.available_permits() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| anyhow::anyhow!("connection permit was not released after close"))?;

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }
}
