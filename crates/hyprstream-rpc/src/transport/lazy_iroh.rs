//! Lazy-connecting iroh RPC transport (optional NAT-traversing dial).
//!
//! Mirrors [`LazyQuinnTransport`](super::lazy_quinn::LazyQuinnTransport): the
//! [`dial`](crate::dial::dial) factory is synchronous, so this holds the dial
//! target (the peer's `EndpointId` + addresses) and connects on the **first
//! `send()`**, caching the session — preserving sync, zero-I/O construction.
//!
//! # The shared client endpoint
//!
//! Unlike quinn (where `connect_pinned_sha256` builds its own one-shot client),
//! iroh dials from a long-lived [`iroh::Endpoint`] that holds the node's
//! identity and bound sockets. There is exactly one client endpoint per
//! process; it is installed once at startup via
//! [`install_iroh_client_endpoint`] (the daemon provisions it during bootstrap),
//! the same install-once pattern as the inproc registry and the envelope verify
//! config. Until it is installed, the iroh dial arm errors loudly rather than
//! silently falling back.
//!
//! # Identity
//!
//! iroh binds the connection to the peer's `EndpointId` (its Ed25519 public
//! key), so the transport authenticates the peer **identity** — stronger than
//! quinn's channel-only cert pin, and it closes the transport half of the
//! cert↔identity gap (#185) for this dial.
//!
//! Re-dial/self-heal semantics match `LazyQuinnTransport`: a per-request timeout
//! keeps the session; a transport-fatal error drops it so the next call
//! re-dials. Refined liveness-based re-dial + backoff is #156's job.

use std::net::SocketAddr;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::transport::iroh_substrate::ALPN_HYPRSTREAM_RPC;
use crate::transport::iroh_transport::{IrohPendingStream, IrohPublishStub, IrohTransport};
use crate::transport_traits::Transport;

/// Default per-request deadline when the caller passes `None`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Ceiling on the lazy connect (iroh dial + handshake). The per-request deadline
/// wraps only the request; without this an unreachable peer would hang the first
/// `send()` indefinitely.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Process-wide client iroh endpoint, installed once at startup.
static IROH_CLIENT_ENDPOINT: OnceLock<iroh::Endpoint> = OnceLock::new();

/// Install the process-global client iroh [`Endpoint`](iroh::Endpoint) used to
/// originate outbound RPC dials. First-write-wins (mirrors
/// `install_verify_config`); returns `Err(endpoint)` if one is already set.
///
/// The daemon calls this once during bootstrap with the shared endpoint (the
/// same one its inbound iroh substrate listens on, so outbound dials reuse the
/// node identity).
pub fn install_iroh_client_endpoint(endpoint: iroh::Endpoint) -> Result<(), iroh::Endpoint> {
    IROH_CLIENT_ENDPOINT.set(endpoint)
}

/// The installed client endpoint, cloned (cheap — iroh `Endpoint` is `Arc`-backed).
fn iroh_client_endpoint() -> Option<iroh::Endpoint> {
    IROH_CLIENT_ENDPOINT.get().cloned()
}

/// An iroh RPC transport that connects on first use and caches the session.
pub struct LazyIrohTransport {
    node_id: [u8; 32],
    direct_addrs: Vec<SocketAddr>,
    relay_url: Option<String>,
    cached: Mutex<Option<IrohTransport>>,
}

impl LazyIrohTransport {
    /// Create a lazy transport for the iroh peer identified by `node_id`. No
    /// connection is made until the first `send()`.
    pub fn new(node_id: [u8; 32], direct_addrs: Vec<SocketAddr>, relay_url: Option<String>) -> Self {
        Self {
            node_id,
            direct_addrs,
            relay_url,
            cached: Mutex::new(None),
        }
    }

    /// Build the iroh `EndpointAddr` for the peer from the stored primitives.
    fn endpoint_addr(&self) -> Result<iroh::EndpointAddr> {
        use iroh::{EndpointAddr, EndpointId, TransportAddr};
        let id = EndpointId::from_bytes(&self.node_id)
            .map_err(|e| anyhow!("invalid iroh node_id: {e}"))?;
        let mut transport_addrs: Vec<TransportAddr> =
            self.direct_addrs.iter().copied().map(TransportAddr::Ip).collect();
        if let Some(url) = &self.relay_url {
            let relay = url
                .parse()
                .map_err(|e| anyhow!("invalid iroh relay_url '{url}': {e}"))?;
            transport_addrs.push(TransportAddr::Relay(relay));
        }
        Ok(EndpointAddr::from_parts(id, transport_addrs))
    }

    /// Return the cached connected transport, dialing once (single-flight under
    /// the lock) if not yet connected.
    async fn connected(&self) -> Result<IrohTransport> {
        let mut guard = self.cached.lock().await;
        if let Some(transport) = guard.as_ref() {
            return Ok(transport.clone());
        }
        let endpoint = iroh_client_endpoint().ok_or_else(|| {
            anyhow!(
                "no iroh client endpoint installed — call \
                 install_iroh_client_endpoint() at startup before dialing iroh peers"
            )
        })?;
        let addr = self.endpoint_addr()?;
        let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(addr, ALPN_HYPRSTREAM_RPC))
            .await
            .map_err(|_| anyhow!("iroh connect timed out after {CONNECT_TIMEOUT:?}"))?
            .map_err(|e| anyhow!("iroh connect: {e}"))?;
        let transport = IrohTransport::new(conn);
        *guard = Some(transport.clone());
        Ok(transport)
    }

    /// Drop the cached session so the next `send()` re-dials.
    async fn invalidate(&self) {
        *self.cached.lock().await = None;
    }
}

#[async_trait]
impl Transport for LazyIrohTransport {
    type Sub = IrohPendingStream;
    type Pub = IrohPublishStub;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        let transport = self.connected().await?;

        // Our deadline is authoritative so a per-request timeout (busy peer —
        // keep the session) is distinguishable from a transport-fatal error
        // (re-dial), matching LazyQuinnTransport.
        let deadline = timeout_ms
            .map(|ms| Duration::from_millis(ms.max(0) as u64))
            .unwrap_or(DEFAULT_TIMEOUT);
        let inner_ceiling_ms = deadline.as_millis().saturating_mul(2).min(i32::MAX as u128) as i32;

        match tokio::time::timeout(deadline, transport.send(payload, Some(inner_ceiling_ms))).await {
            Err(_elapsed) => Err(anyhow!("iroh RPC timeout after {deadline:?}")),
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(e)) => {
                self.invalidate().await;
                Err(e)
            }
        }
    }

    async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
        bail!("lazy iroh RPC transport does not support SUB — streaming is on the moq plane")
    }

    async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
        bail!("lazy iroh RPC transport does not support PUB — streaming is on the moq plane")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::transport::iroh_substrate::{EchoHandler, IrohSubstrate, NoopHandler};

    fn fresh_key() -> [u8; 32] {
        use rand::RngCore;
        let mut k = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut k);
        k
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lazy_iroh_connects_on_first_send_and_caches() {
        // Echo server substrate.
        let server = IrohSubstrate::new(fresh_key(), EchoHandler, EchoHandler).await.unwrap();
        let server_id = server.endpoint_id();
        let direct: Vec<SocketAddr> = server.endpoint().bound_sockets().into_iter().collect();

        // Client substrate (originates only); install its endpoint as the global.
        let client = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("client moq"),
            NoopHandler::new("client rpc"),
        )
        .await
        .unwrap();
        let _ = install_iroh_client_endpoint(client.endpoint().clone());

        let node_id: [u8; 32] = *server_id.as_bytes();
        let t = LazyIrohTransport::new(node_id, direct, None);
        assert!(t.cached.lock().await.is_none(), "no connection before first send");

        // EchoHandler echoes the bytes written on the bidi stream, which is the
        // same shape SessionRpcTransport::send uses (write payload → read to
        // FIN), so the lazy transport round-trips them. (EchoHandler is a
        // one-shot smoke handler — it closes the connection after a single
        // stream, so we don't test session *reuse* here; the multiplexed-stream
        // path is covered by quinn_transport's round-trip test against a real
        // QuinnRpcServer, which IrohTransport shares via SessionRpcTransport.)
        let resp = t.send(b"ping".to_vec(), Some(4_000)).await.unwrap();
        assert_eq!(resp, b"ping");
        assert!(t.cached.lock().await.is_some(), "session cached after first send");

        server.shutdown().await.unwrap();
        // NOTE: do NOT shut down `client` — its endpoint is installed in the
        // process-global IROH_CLIENT_ENDPOINT (install-once), so tearing it down
        // would break the shared dialer for any other test. Drop it; the global
        // Arc clone keeps the endpoint alive (matches the never-reset prod lifecycle).
        drop(client);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wrong_node_id_rejected() {
        // Server we'll dial the *address* of, but with a different identity.
        let server = IrohSubstrate::new(fresh_key(), EchoHandler, EchoHandler).await.unwrap();
        let direct: Vec<SocketAddr> = server.endpoint().bound_sockets().into_iter().collect();

        let client = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("client moq"),
            NoopHandler::new("client rpc"),
        )
        .await
        .unwrap();
        // First-write-wins; may already be installed by another test (same global).
        let _ = install_iroh_client_endpoint(client.endpoint().clone());

        // A *valid* but wrong EndpointId (a third node's), so this tests handshake
        // identity rejection — not a malformed key.
        let other = IrohSubstrate::new(fresh_key(), NoopHandler::new("o1"), NoopHandler::new("o2"))
            .await
            .unwrap();
        let wrong_id: [u8; 32] = *other.endpoint_id().as_bytes();

        let t = LazyIrohTransport::new(wrong_id, direct, None);
        let res = tokio::time::timeout(Duration::from_secs(8), t.send(b"x".to_vec(), Some(3_000)))
            .await
            .expect("send must complete (with an error) — wrong identity should reject, not hang");
        assert!(res.is_err(), "dialing a server's address under a wrong EndpointId must fail");
        assert!(t.cached.lock().await.is_none(), "a failed handshake caches nothing");

        other.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
        // NOTE: do NOT shut down `client` — its endpoint is installed in the
        // process-global IROH_CLIENT_ENDPOINT (install-once), so tearing it down
        // would break the shared dialer for any other test. Drop it; the global
        // Arc clone keeps the endpoint alive (matches the never-reset prod lifecycle).
        drop(client);
    }

    #[tokio::test]
    async fn subscribe_publish_bail() {
        let t = LazyIrohTransport::new([0u8; 32], vec![], None);
        assert!(t.subscribe(b"topic").await.is_err());
        assert!(t.publish(b"topic").await.is_err());
    }
}
