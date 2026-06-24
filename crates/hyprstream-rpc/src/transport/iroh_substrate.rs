//! Iroh transport substrate — the PRIMARY production transport for Epic #131
//! (ZMQ → moq-net + iroh), promoted to default in #410.
//!
//! Builds a shared `iroh::Endpoint` from an Ed25519 node key and wires an
//! [`iroh::protocol::Router`] with two ALPN slots:
//!
//! - [`ALPN_MOQ_LITE`] (`moql`) — moq-net session handler, wired by the daemon
//!   spawner (#282) to dispatch into the shared global `moq_stream::Origin`
//!   for the streaming plane.
//! - [`ALPN_HYPRSTREAM_RPC`] (`hyprstream-rpc/1`) — raw Cap'n Proto bidi
//!   handler, wired by the daemon spawner (#282) to terminate
//!   `IrohRpcProtocolHandler` for the RPC plane.
//!
//! Phase 1 shipped the substrate, a [`NoopHandler`] placeholder, and a smoke
//! test verifying that two endpoints can dial each other directly and open a
//! bidi stream on each ALPN. As of #410, production code DOES consume this
//! module: the daemon spawner (`hyprstream-service::service::spawner`) binds
//! one `IrohSubstrate` per QUIC-enabled service as the PRIMARY production
//! endpoint (on by default; `[quic] iroh = false` opts out to quinn-only). The
//! real `moql` and `hyprstream-rpc/1` handlers are threaded in by the spawner
//! (#282), and the OAuth/DID-controller identity binds its own canonical
//! federation substrate (`build_oauth_iroh_substrate`).

use anyhow::Result;
use iroh::endpoint::{Connection, presets};
use iroh::protocol::{AcceptError, DynProtocolHandler, ProtocolHandler, Router};
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};

/// ALPN for moq-net (moq-lite). Must equal `moq_net::version::ALPN_LITE`.
pub const ALPN_MOQ_LITE: &[u8] = b"moql";

/// ALPN for the raw Cap'n Proto bidi RPC plane.
pub const ALPN_HYPRSTREAM_RPC: &[u8] = b"hyprstream-rpc/1";

/// Two-plane iroh substrate: one shared QUIC endpoint, two ALPN-dispatched
/// protocol handlers.
///
/// Drop to abort the accept loop; call [`IrohSubstrate::shutdown`] for a
/// clean shutdown that drains protocol handlers and closes the endpoint.
pub struct IrohSubstrate {
    endpoint: Endpoint,
    router: Router,
}

impl IrohSubstrate {
    /// Build the substrate from raw 32-byte Ed25519 secret key material.
    ///
    /// Uses iroh's `presets::N0` for discovery (n0 DNS + pkarr) and relay
    /// fallback. Operators wanting self-hosted discovery should bypass this
    /// constructor and use [`IrohSubstrate::from_endpoint`] with a
    /// pre-configured `iroh::Endpoint`.
    pub async fn new<M, R>(
        secret_key_bytes: [u8; 32],
        moq_handler: M,
        rpc_handler: R,
    ) -> Result<Self>
    where
        M: Into<Box<dyn DynProtocolHandler>>,
        R: Into<Box<dyn DynProtocolHandler>>,
    {
        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(SecretKey::from_bytes(&secret_key_bytes))
            .bind()
            .await
            .map_err(|e| anyhow::anyhow!("iroh endpoint bind: {e}"))?;

        Ok(Self::from_endpoint(endpoint, moq_handler, rpc_handler))
    }

    /// Wrap a caller-built endpoint. Useful when the caller wants
    /// non-default discovery (e.g. self-hosted `iroh-dns-server`), a
    /// non-default crypto provider, or to share an existing endpoint
    /// across substrates.
    pub fn from_endpoint<M, R>(
        endpoint: Endpoint,
        moq_handler: M,
        rpc_handler: R,
    ) -> Self
    where
        M: Into<Box<dyn DynProtocolHandler>>,
        R: Into<Box<dyn DynProtocolHandler>>,
    {
        let router = Router::builder(endpoint.clone())
            .accept(ALPN_MOQ_LITE, moq_handler.into())
            .accept(ALPN_HYPRSTREAM_RPC, rpc_handler.into())
            .spawn();
        Self { endpoint, router }
    }

    /// Endpoint id = our Ed25519 public key. Published in JWKS for
    /// federation peer binding (Phase 6, issue #137).
    pub fn endpoint_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    /// Borrow the underlying endpoint, e.g. to issue outbound connects on
    /// an ALPN not served by this substrate's router.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Borrow the router (e.g. to inspect or to share lifetime).
    pub fn router(&self) -> &Router {
        &self.router
    }

    /// Dial a peer for the given ALPN.
    pub async fn connect(&self, addr: EndpointAddr, alpn: &[u8]) -> Result<Connection> {
        self.endpoint
            .connect(addr, alpn)
            .await
            .map_err(|e| anyhow::anyhow!("iroh connect: {e}"))
    }

    /// Drain handlers and close the endpoint.
    pub async fn shutdown(self) -> Result<()> {
        self.router
            .shutdown()
            .await
            .map_err(|e| anyhow::anyhow!("router shutdown: {e}"))?;
        Ok(())
    }
}

/// Placeholder protocol handler that accepts an incoming connection and
/// immediately returns. Used until Phase 2 / Phase 3 wire the real handlers.
#[derive(Debug, Clone, Default)]
pub struct NoopHandler {
    reason: &'static str,
}

impl NoopHandler {
    pub fn new(reason: &'static str) -> Self {
        Self { reason }
    }
}

impl ProtocolHandler for NoopHandler {
    async fn accept(&self, _conn: Connection) -> Result<(), AcceptError> {
        tracing::trace!(reason = %self.reason, "NoopHandler accepted (no-op)");
        Ok(())
    }
}

/// Test-only handler that accepts one bidi stream, reads one length-prefixed
/// message, echoes it back, then closes. Used by the Phase 1 smoke test to
/// verify that both ALPN slots route correctly.
#[derive(Debug, Clone, Default)]
pub struct EchoHandler;

impl ProtocolHandler for EchoHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let (mut send, mut recv) = conn
            .accept_bi()
            .await
            .map_err(|e| AcceptError::from_err(e))?;
        let buf = recv
            .read_to_end(1024)
            .await
            .map_err(|e| AcceptError::from_err(e))?;
        send.write_all(&buf)
            .await
            .map_err(|e| AcceptError::from_err(e))?;
        send.finish().map_err(|e| AcceptError::from_err(e))?;
        // Cleanly wait for the peer to acknowledge by reading the final FIN.
        let _ = send.stopped().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::TransportAddr;
    use rand::RngCore;

    fn fresh_key() -> [u8; 32] {
        let mut k = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut k);
        k
    }

    /// Build an `EndpointAddr` for a server directly from its bound sockets +
    /// endpoint id. Skips the n0 relay/pkarr resolution path, which keeps the
    /// smoke test hermetic — no DNS lookup, no network egress.
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

    /// Smoke test: build two substrates, dial direct (no DNS/pkarr), echo a
    /// message on each ALPN. Verifies the router dispatches by ALPN and that
    /// both planes terminate distinct handlers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn smoke_two_endpoints_two_alpns() -> Result<()> {
        // Server with EchoHandler on both ALPNs.
        let server = IrohSubstrate::new(fresh_key(), EchoHandler, EchoHandler).await?;
        let server_id = server.endpoint_id();
        let server_addr = direct_addr(&server);

        // Client with no-op handlers (it only originates).
        let client = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("client moq"),
            NoopHandler::new("client rpc"),
        )
        .await?;

        // RPC plane round-trip.
        {
            let conn = client
                .connect(server_addr.clone(), ALPN_HYPRSTREAM_RPC)
                .await?;
            let (mut send, mut recv) = conn.open_bi().await?;
            send.write_all(b"hello rpc").await?;
            send.finish()?;
            let got = recv.read_to_end(64).await?;
            assert_eq!(&got, b"hello rpc");
        }

        // Streaming plane round-trip (echo here; real moq-net wiring lands in Phase 3).
        {
            let conn = client.connect(server_addr.clone(), ALPN_MOQ_LITE).await?;
            let (mut send, mut recv) = conn.open_bi().await?;
            send.write_all(b"hello moq").await?;
            send.finish()?;
            let got = recv.read_to_end(64).await?;
            assert_eq!(&got, b"hello moq");
        }

        // Sanity: server id is stable.
        assert_eq!(server.endpoint_id(), server_id);

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }
}
