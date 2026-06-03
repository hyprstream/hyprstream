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
                shutdown: CancellationToken::new(),
            }),
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
        let session = Session::raw(conn);
        let server = Server::new()
            .with_publish(self.inner.origin.consumer.clone())
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
}
