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

/// Resolves the tenant for an independently authenticated application peer.
/// Carrier NodeId is never passed here. Until a caller can supply fresh proof,
/// the iroh accept path refuses before invoking this resolver.
pub type PeerTenantResolver = Arc<dyn Fn(&PeerIdentity) -> Option<String> + Send + Sync>;

/// #276 authorization config for a moq accept path: an optional subscribe
/// authorizer and an optional peer→tenant resolver for per-tenant announce
/// scoping. Absence of fresh application/session proof is always fail-closed;
/// these hooks cannot turn an anonymous carrier into an authenticated peer.
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
                shutdown: CancellationToken::new(),
            }),
        }
    }

    /// Install the #276 subscribe-authz + per-tenant announce-scoping config.
    ///
    /// The resolver is usable only after a future fresh-proof seam supplies an
    /// authenticated [`PeerIdentity`]. Carrier EndpointId alone never reaches it.
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
        // Carrier NodeId is transport metadata only. Until #1027 supplies fresh
        // inside-carrier proof, authorization must see an anonymous peer.
        let peer = PeerIdentity::anonymous();
        if !peer.is_authenticated() {
            tracing::warn!(
                "iroh-moq: refusing anonymous carrier pending verified session proof (#1027/#726)"
            );
            conn.close(0u32.into(), b"verified MoQ session proof required");
            return Ok(());
        }

        // Future authenticated-session path: if a fresh-proof integration
        // supplies a verified peer, hand it only a consumer narrowed to its own
        // `{tenant}/` prefix. Missing tenant resolution must remain fail-closed;
        // it must never fall back to the process-global consumer.
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
            None => {
                tracing::warn!("iroh-moq: authenticated peer has no tenant scope; refusing");
                conn.close(0u32.into(), b"tenant scope required");
                return Ok(());
            }
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

    /// An anonymous carrier cannot subscribe to server broadcasts, publish a
    /// client origin, or reach the tenant resolver. Mutating the carrier NodeId
    /// cannot change that decision because it is never an application proof.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn anonymous_carrier_cannot_publish_subscribe_or_obtain_tenant_scope(
    ) -> anyhow::Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let resolver_calls = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&resolver_calls);
        let authz = MoqAuthzConfig {
            authorizer: None,
            tenant_resolver: Some(Arc::new(move |_peer| {
                calls.fetch_add(1, Ordering::SeqCst);
                Some("alice".to_owned())
            })),
        };
        let handler = IrohMoqProtocolHandler::new().with_authz(authz);
        let producer = handler.origin_producer().clone();
        let server_consumer = handler.origin_consumer().clone();
        let server = IrohSubstrate::new_test(
            fresh_key(),
            handler,
            NoopHandler::new("rpc-not-wired"),
        )
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
        let moq_client = Client::new().with_origin(client_origin.clone()).with_consume(client_origin);
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

}
