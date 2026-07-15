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
//! (#282). OAuth binds a reach-only endpoint whose inbound ALPNs remain refused
//! until independently verified application/session proof is wired.
//!
//! # D3 — iroh pkarr is LIVENESS-ONLY, never an authority source (#895)
//!
//! iroh's `presets::N0` discovery publishes classical **pkarr** records —
//! signed-but-not-encrypted packets — onto the open BitTorrent mainline DHT to
//! advertise reach (home relay + direct addresses) for a NodeId. As of #895
//! (epic #880 Track D / D3) this is the contract for pkarr in hyprstream:
//!
//! - **Reach hint, not authority.** A pkarr record is an **unverified dial
//!   candidate**: "NodeId N claims it is reachable at relay R / addr A." It is
//!   signed by N's Ed25519 key, so it is integrity-protected *as a reach
//!   claim* — but that says nothing about *which* federated identity (capsule,
//!   `did:web`, `did:key`) controls N, what its PQ key material is, or whether
//!   it may be admitted. **No identity / trust / admission decision is ever
//!   derived from a pkarr record.** `Connection::remote_id()` authenticates only
//!   the carrier endpoint. Application identity requires independently resolved
//!   current keys plus fresh inside-carrier proof (#1027).
//! - **Two riders, one DHT, one trust posture.** Both iroh pkarr and the at9p
//!   mainline locator (#890 / C2) ride the *same* mainline DHT. Only at9p is
//!   zero-trust: its records are content-addressed (`did:at9p:<cid512>`) and
//!   made trustworthy by the GATE pipeline (canon → hash → composite-sig), so
//!   the DHT is an **untrusted locator** for at9p, never a trust root. pkarr, by
//!   contrast, is classical-only and trusts the DHT to carry the publisher's
//!   reach claim verbatim — fine for liveness, fatal as an authority source.
//! - **Not removed.** pkarr stays — it is genuinely useful for NAT traversal /
//!   liveness ("is this peer dialable right now?"). D3 only strips (and forbids
//!   re-deriving) authority from it; it does not disable publication.
//!
//! Enforcement seams: the dial path (`crate::dial`) resolves *addresses* from a
//! NodeId as an address, never trusting pkarr for identity. Admission does not
//! consume NodeId, `remote_id()`, or pkarr. The wasm pkarr output is typed as an unverified reach hint
//! ([`crate::iroh_peer::PkarrReachHint`]) so it cannot be conflated with
//! verified reach.
//!
//! The #385 metadata-leak side channel (a passive observer can enumerate which
//! NodeIds are reachable and correlate lookup traffic) is **unaffected** by this
//! demotion — pkarr is still public. Its mitigation is oblivious-relay (#361);
//! until then, operators that consider the side channel actionable should run a
//! self-hosted `iroh-dns-server` via [`IrohSubstrate::from_endpoint`].

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
    ///
    /// **pkarr here is liveness-only** — see the module-level "D3" note. The
    /// published record carries reach hints (relay + direct addrs) for this
    /// endpoint's NodeId; it derives **zero** identity/trust authority. The
    /// #385 metadata-leak caveat also applies (mitigated by #361, not by this
    /// demotion).
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
            // Owned mesh policy: require X25519MLKEM768 with no classical
            // fallback. RFC 7250 raw-public-key identity is orthogonal to kx.
            .crypto_provider(crate::transport::pq_provider::internal_mesh_crypto_provider())
            .bind()
            .await
            .map_err(|e| anyhow::anyhow!("iroh endpoint bind: {e}"))?;

        Ok(Self::from_endpoint(endpoint, moq_handler, rpc_handler))
    }

    /// Build a hermetic direct-only substrate for unit tests.
    ///
    /// Production uses [`presets::N0`] for discovery, pkarr publication, and
    /// relay fallback. Unit carrier tests construct their target explicitly
    /// from bound sockets, so those background paths add no coverage and can
    /// race independent Tokio runtimes during parallel suite teardown. The
    /// empty preset keeps the same QUIC crypto provider and real UDP carrier
    /// while disabling discovery and relay workers.
    #[cfg(test)]
    pub(crate) async fn new_test<M, R>(
        secret_key_bytes: [u8; 32],
        moq_handler: M,
        rpc_handler: R,
    ) -> Result<Self>
    where
        M: Into<Box<dyn DynProtocolHandler>>,
        R: Into<Box<dyn DynProtocolHandler>>,
    {
        let endpoint = Endpoint::builder(presets::Empty)
            .secret_key(SecretKey::from_bytes(&secret_key_bytes))
            .crypto_provider(crate::transport::pq_provider::pq_crypto_provider())
            .bind()
            .await
            .map_err(|e| anyhow::anyhow!("test iroh endpoint bind: {e}"))?;

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

    /// Return this endpoint's carrier address for reach advertisement and
    /// diagnostics. It is not a DID/JWKS key or application identity proof.
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

/// Fail-closed protocol handler for an ALPN that is intentionally disabled.
///
/// Unlike [`NoopHandler`], this explicitly closes the carrier connection. It is
/// used where an endpoint may exist for outbound reach but no trustworthy
/// application/session proof is available for inbound requests.
#[derive(Debug, Clone)]
pub struct RefuseHandler {
    reason: &'static str,
}

impl RefuseHandler {
    pub fn new(reason: &'static str) -> Self {
        Self { reason }
    }
}

impl ProtocolHandler for RefuseHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        tracing::warn!(reason = %self.reason, "refusing disabled iroh ALPN");
        conn.close(0u32.into(), self.reason.as_bytes());
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
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use iroh::TransportAddr;
    use noq::crypto::rustls::HandshakeData;
    use rand::RngCore;
    use std::sync::Arc;

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

    fn assert_hybrid_handshake(conn: &Connection, alpn: &[u8]) {
        let data = conn
            .handshake_data()
            .expect("completed iroh connection has handshake data")
            .downcast::<HandshakeData>()
            .expect("iroh uses noq rustls handshake data");
        assert_eq!(data.protocol.as_deref(), Some(alpn));
        assert_eq!(
            data.negotiated_key_exchange_group,
            Some(rustls::NamedGroup::X25519MLKEM768),
            "owned iroh mesh must negotiate X25519MLKEM768"
        );
    }

    /// Smoke test: build two substrates, dial direct (no DNS/pkarr), echo a
    /// message on each ALPN. Verifies the router dispatches by ALPN and that
    /// both planes terminate distinct handlers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn smoke_two_endpoints_two_alpns() -> Result<()> {
        // Server with EchoHandler on both ALPNs.
        let server = IrohSubstrate::new_test(fresh_key(), EchoHandler, EchoHandler).await?;
        let server_id = server.endpoint_id();
        let server_addr = direct_addr(&server);

        // Client with no-op handlers (it only originates).
        let client = IrohSubstrate::new_test(
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
            assert_hybrid_handshake(&conn, ALPN_HYPRSTREAM_RPC);
            let (mut send, mut recv) = conn.open_bi().await?;
            send.write_all(b"hello rpc").await?;
            send.finish()?;
            let got = recv.read_to_end(64).await?;
            assert_eq!(&got, b"hello rpc");
        }

        // Streaming plane round-trip (echo here; real moq-net wiring lands in Phase 3).
        {
            let conn = client.connect(server_addr.clone(), ALPN_MOQ_LITE).await?;
            assert_hybrid_handshake(&conn, ALPN_MOQ_LITE);
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn internal_iroh_mesh_rejects_classical_only_peer() -> Result<()> {
        let server = IrohSubstrate::new(fresh_key(), EchoHandler, EchoHandler).await?;
        let server_addr = direct_addr(&server);
        let classical_client = Endpoint::builder(presets::N0)
            .secret_key(SecretKey::from_bytes(&fresh_key()))
            .crypto_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .bind()
            .await
            .map_err(|e| anyhow::anyhow!("classical mutation endpoint bind: {e}"))?;

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            classical_client.connect(server_addr, ALPN_HYPRSTREAM_RPC),
        )
        .await
        .expect("classical-only iroh mutation handshake must terminate");
        assert!(result.is_err(), "internal iroh mesh accepted classical-only TLS");

        classical_client.close().await;
        server.shutdown().await?;
        Ok(())
    }
}
