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
//! **Trust model**: §7.5 unchanged. The transport is unauthenticated;
//! subscribers learn the DH-derived (unguessable) Track path out-of-band
//! via authenticated RPC (a signed `StreamInfo`), and the per-Frame
//! payload carries the chained-HMAC envelope. CDN-portability per §10.x
//! of the Federated Agentic Namespaces doc.
//!
//! **What this module does NOT do**: refactor `StreamService` or
//! `EventService` to use this. That lands in Phase 3 part 2+ along with
//! the §7.5 chained-HMAC tokenstream port.

use std::sync::Arc;

use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use moq_net::{Origin, OriginConsumer, OriginProducer, Server, StatsHandle};
use tokio_util::sync::CancellationToken;
use web_transport_iroh::Session;

use crate::moq_authz::{tenant_scoped_consumer, PeerIdentity, SharedSubscribeAuthorizer};

/// Resolves the tenant a connected peer belongs to, from its (authenticated)
/// identity. Returning `Some(tenant)` scopes that peer's view of announces to
/// `{tenant}/`; returning `None` leaves the peer unscoped (the pre-#276 open
/// model). Wired by the caller, which owns the peer↔tenant policy.
pub type PeerTenantResolver = Arc<dyn Fn(&PeerIdentity) -> Option<String> + Send + Sync>;

/// #276 authorization config for a moq accept path: an optional subscribe
/// authorizer and an optional peer→tenant resolver for per-tenant announce
/// scoping. Both default to "off" so existing open same-tenant subscribe keeps
/// working until a deployment opts in.
#[derive(Clone, Default)]
pub struct MoqAuthzConfig {
    /// Subscribe-time authorization hook (public-open / private-gated).
    pub authorizer: Option<SharedSubscribeAuthorizer>,
    /// Maps a peer identity to its tenant for per-tenant announce scoping.
    pub tenant_resolver: Option<PeerTenantResolver>,
}

impl MoqAuthzConfig {
    /// Resolve the tenant for `peer`, if a resolver is configured.
    pub fn tenant_for(&self, peer: &PeerIdentity) -> Option<String> {
        self.tenant_resolver.as_ref().and_then(|r| r(peer))
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
    /// #137/#282 federation admission hook. `None` = open (pre-#282). When set,
    /// an inbound connection whose `remote_id()` is not admitted is dropped
    /// (fail-closed) before any moq session is served.
    admission: Option<crate::transport::iroh_admission::SharedIrohAdmission>,
    /// Triggered by `ProtocolHandler::shutdown` so accept handlers stop
    /// waiting for `Session::closed()` and exit promptly.
    shutdown: CancellationToken,
}

impl std::fmt::Debug for IrohMoqProtocolHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohMoqProtocolHandler").finish_non_exhaustive()
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
                admission: None,
                shutdown: CancellationToken::new(),
            }),
        }
    }

    /// Install the #137/#282 federation admission hook. When set, an inbound
    /// connection whose authenticated `remote_id()` is not admitted by the gate
    /// is dropped (fail-closed) before any moq session is served.
    pub fn with_admission(
        mut self,
        admission: crate::transport::iroh_admission::SharedIrohAdmission,
    ) -> Self {
        self.rebuild_inner(|i| i.admission = Some(admission));
        self
    }

    /// Install the #276 subscribe-authz + per-tenant announce-scoping config.
    ///
    /// On this (iroh) path the accepted connection is authenticated, so the
    /// `tenant_resolver` receives a real [`PeerIdentity`] (the remote endpoint
    /// id) and per-tenant scoping is enforced live by handing the moq session a
    /// tenant-scoped [`OriginConsumer`].
    pub fn with_authz(mut self, authz: MoqAuthzConfig) -> Self {
        self.rebuild_inner(|i| i.authz = authz);
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
                admission: old.admission.clone(),
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
        // #137/#282 federation admission: the accepted iroh connection is
        // authenticated to the remote endpoint's Ed25519 key (`remote_id()`).
        // Run the gate (origin + key-binding) before serving any broadcast; on
        // rejection, fail-closed by dropping the connection. No hook = open.
        let remote = *conn.remote_id().as_bytes();
        if !crate::transport::iroh_admission::check_admission(
            self.inner.admission.as_ref(),
            &remote,
        )
        .await
        {
            return Ok(());
        }

        // #276: peer identity IS available here — an accepted iroh connection
        // is authenticated by the remote endpoint's public key. Use it to
        // derive the peer's tenant and scope the consumer it's served.
        let peer = PeerIdentity::authenticated(conn.remote_id().to_string());

        // Per-tenant announce scoping (live): if the caller installed a
        // peer→tenant resolver, hand this session a consumer narrowed to its
        // own `{tenant}/` prefix so it cannot enumerate or subscribe to other
        // tenants' broadcasts. No resolver → unscoped (pre-#276 open model).
        let publish_consumer = match self.inner.authz.tenant_for(&peer) {
            Some(tenant) => {
                // `scope` returns None when the tenant prefix is outside the
                // origin's allowed prefixes — serve that peer nothing rather
                // than falling back to the unscoped consumer (fail-closed for
                // cross-tenant enumeration).
                match tenant_scoped_consumer(&self.inner.origin.consumer, &tenant) {
                    Some(scoped) => {
                        tracing::debug!(peer = %peer.subject.as_deref().unwrap_or("?"), %tenant, "iroh-moq: tenant-scoped consumer");
                        scoped
                    }
                    None => {
                        tracing::debug!(%tenant, "iroh-moq: tenant has no visible broadcasts; serving empty scope");
                        // An empty scope: a fresh consumer cursor over a prefix
                        // with no broadcasts. Re-scope to a sentinel under the
                        // tenant so the peer sees nothing it isn't entitled to.
                        // Falling through to a clone would leak cross-tenant
                        // names, so we instead drop the session.
                        return Ok(());
                    }
                }
            }
            None => self.inner.origin.consumer.clone(),
        };

        // NOTE (#276 subscribe-authz seam): `moq_net::Server` exposes no
        // per-subscribe callback, so we cannot gate individual `subscribe_track`
        // calls in-band. The public/private decision is therefore enforced
        // *structurally* by what the served consumer can see (tenant scoping
        // above). The pluggable `SubscribeAuthorizer` is retained as the policy
        // unit a richer moq-net (one that surfaces a subscribe hook) would call;
        // it is exercised by unit tests in `crate::moq_authz`. Until moq-net
        // grows that hook, private-vs-public must be expressed as scoping (e.g.
        // a private broadcast lives under a prefix only entitled peers' tenant
        // scope includes).
        let _authorizer = self.inner.authz.authorizer.as_ref();

        let session = Session::raw(conn);
        let server = Server::new()
            .with_publish(publish_consumer)
            // Subscribe slot is None for v1 — we only serve broadcasts;
            // accepting remote announces (cross-instance fan-out) lands
            // in Phase 3 part N (#142).
            .with_stats(self.inner.stats.clone());
        let moq_session = server
            .accept(session)
            .await
            .map_err(AcceptError::from_err)?;

        // Hold the session alive until either the connection closes or
        // shutdown is requested. Server::accept has already spawned the
        // session's pump tasks; dropping `moq_session` tears them down.
        tokio::select! {
            biased;
            _ = self.inner.shutdown.cancelled() => {
                tracing::debug!("iroh-moq: shutdown signalled, dropping session");
            }
            res = moq_session.closed() => {
                tracing::debug!(result = ?res, "iroh-moq: session closed");
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
    use crate::transport::iroh_substrate::{
        ALPN_MOQ_LITE, IrohSubstrate, NoopHandler,
    };
    use bytes::Bytes;
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

    /// In-process publisher writes one Frame to a Track on a Broadcast;
    /// an external subscriber connects over iroh, navigates the same
    /// broadcast/track path, and reads back the same Frame bytes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moq_publish_subscribe_round_trip() -> anyhow::Result<()> {
        // ─── Server side ──────────────────────────────────────────────
        let handler = IrohMoqProtocolHandler::new();
        let producer = handler.origin_producer().clone();
        let server = IrohSubstrate::new(
            fresh_key(),
            handler,
            NoopHandler::new("rpc-not-wired"),
        )
        .await?;
        let server_addr = direct_addr(&server);

        // Publish a broadcast with one track before the subscriber connects.
        let mut broadcast = producer
            .create_broadcast("alice/run-1")
            .ok_or_else(|| anyhow::anyhow!("create_broadcast denied"))?;
        let mut track = broadcast.create_track(Track::new("tokens"))?;
        let mut group = track.create_group(Group::from(0u64))?;
        group.write_frame(Bytes::from_static(b"hello-moq"))?;
        drop(group);

        // ─── Client side: connect via iroh, run moq Client to negotiate,
        // then subscribe to the known broadcast path. ─────────────────
        let client = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let conn = client.connect(server_addr, ALPN_MOQ_LITE).await?;
        let session = Session::raw(conn);
        let client_origin: OriginProducer = Origin::random().produce();
        let client_consumer: OriginConsumer = client_origin.consume();
        let moq_client = Client::new().with_consume(client_origin);
        let _moq_session = moq_client.connect(session).await?;

        // Subscribe to alice/run-1 and read the first group's frame.
        let broadcast_consumer = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client_consumer.announced_broadcast("alice/run-1"),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("broadcast not announced"))?;
        let mut track_consumer =
            broadcast_consumer.subscribe_track(&Track::new("tokens"))?;
        let mut group_consumer = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            track_consumer.next_group(),
        )
        .await??
        .ok_or_else(|| anyhow::anyhow!("next_group returned None"))?;
        let frame: Bytes = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            group_consumer.read_frame(),
        )
        .await??
        .ok_or_else(|| anyhow::anyhow!("read_frame returned None"))?;
        assert_eq!(&frame[..], b"hello-moq");

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    /// #276 live per-tenant announce scoping over the iroh `moql` path: the
    /// server publishes broadcasts for two tenants but the accept path scopes
    /// every connection to `bob/` (via the tenant resolver). A remote subscriber
    /// must see `bob`'s broadcast and must NOT be able to reach `alice`'s.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn iroh_moq_scopes_announces_to_connection_tenant() -> anyhow::Result<()> {
        // `MoqAuthzConfig` is defined in this module (`super`).
        // Server scoped so EVERY peer is treated as tenant "bob".
        let authz = MoqAuthzConfig {
            authorizer: None,
            tenant_resolver: Some(Arc::new(|_peer| Some("bob".to_owned()))),
        };
        let handler = IrohMoqProtocolHandler::new().with_authz(authz);
        let producer = handler.origin_producer().clone();
        let server = IrohSubstrate::new(fresh_key(), handler, NoopHandler::new("rpc-not-wired")).await?;
        let server_addr = direct_addr(&server);

        // Publish one broadcast per tenant.
        let mut alice_bc = producer
            .create_broadcast("alice/streams/secret/i0")
            .ok_or_else(|| anyhow::anyhow!("create alice broadcast"))?;
        let mut alice_tr = alice_bc.create_track(Track::new("tokens"))?;
        let mut alice_g = alice_tr.create_group(Group::from(0u64))?;
        alice_g.write_frame(Bytes::from_static(b"alice-secret"))?;
        drop(alice_g);

        let mut bob_bc = producer
            .create_broadcast("bob/streams/run-1/i0")
            .ok_or_else(|| anyhow::anyhow!("create bob broadcast"))?;
        let mut bob_tr = bob_bc.create_track(Track::new("tokens"))?;
        let mut bob_g = bob_tr.create_group(Group::from(0u64))?;
        bob_g.write_frame(Bytes::from_static(b"bob-data"))?;
        drop(bob_g);

        // Client connects; the accept path scopes its consumer to bob/.
        let client =
            IrohSubstrate::new(fresh_key(), NoopHandler::new("c-moq"), NoopHandler::new("c-rpc")).await?;
        let conn = client.connect(server_addr, ALPN_MOQ_LITE).await?;
        let session = Session::raw(conn);
        let client_origin: OriginProducer = Origin::random().produce();
        let client_consumer: OriginConsumer = client_origin.consume();
        let moq_client = Client::new().with_consume(client_origin);
        let _moq_session = moq_client.connect(session).await?;

        // bob/ is visible: subscribe and read its frame.
        let bob_consumer = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client_consumer.announced_broadcast("bob/streams/run-1/i0"),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("bob broadcast not announced"))?;
        let mut bob_track = bob_consumer.subscribe_track(&Track::new("tokens"))?;
        let mut bob_group = tokio::time::timeout(std::time::Duration::from_secs(5), bob_track.next_group())
            .await??
            .ok_or_else(|| anyhow::anyhow!("bob next_group None"))?;
        let bob_frame: Bytes = tokio::time::timeout(std::time::Duration::from_secs(5), bob_group.read_frame())
            .await??
            .ok_or_else(|| anyhow::anyhow!("bob read_frame None"))?;
        assert_eq!(&bob_frame[..], b"bob-data");

        // alice/ is NOT visible to this bob-scoped subscriber: the announce
        // must never arrive (timeout → not announced through the scoped cursor).
        let alice_seen = tokio::time::timeout(
            std::time::Duration::from_millis(750),
            client_consumer.announced_broadcast("alice/streams/secret/i0"),
        )
        .await;
        assert!(
            alice_seen.is_err() || alice_seen.ok().flatten().is_none(),
            "bob-scoped subscriber must NOT see alice's broadcast"
        );

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }
}
