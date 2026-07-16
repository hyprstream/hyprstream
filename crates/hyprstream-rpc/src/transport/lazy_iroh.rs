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
//! iroh dials from a long-lived [`iroh::Endpoint`] that holds its transport
//! secret and bound sockets. There is exactly one client endpoint per
//! process; it is installed once at startup via
//! [`install_iroh_client_endpoint`] (the daemon provisions it during bootstrap),
//! the same install-once pattern as the inproc registry and the envelope verify
//! config. Until it is installed, the iroh dial arm errors loudly rather than
//! silently falling back.
//!
//! # Target integrity
//!
//! iroh binds the connection to the peer's `EndpointId` (its Ed25519 public
//! key), proving the dial reached the configured carrier address. This is path
//! integrity only. It does not authenticate a DID, admission subject, assurance
//! level, response key, or authorization identity (#1031).
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

use crate::transport::backoff::LazyState;
use crate::transport::iroh_substrate::{ALPN_HYPRSTREAM_RPC, OwnedIrohClientEndpoint};
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

/// Install the process-global client iroh endpoint used to
/// originate outbound RPC dials. First-write-wins (mirrors
/// `install_verify_config`); returns `Err(endpoint)` if one is already set.
///
/// The daemon calls this once during bootstrap with the shared endpoint (the
/// same one its inbound iroh substrate listens on, so outbound dials reuse the
/// node identity). The capability can only be obtained from
/// [`crate::transport::iroh_substrate::IrohSubstrate::owned_client_endpoint`],
/// which proves the endpoint was bound with the exact hybrid-only provider.
///
/// Arbitrary already-bound endpoints are rejected by the type system:
///
/// ```compile_fail
/// # let endpoint: iroh::Endpoint = todo!();
/// hyprstream_rpc::transport::lazy_iroh::install_iroh_client_endpoint(endpoint);
/// ```
///
/// The opaque capability itself also cannot be forged from a raw endpoint:
///
/// ```compile_fail
/// # let endpoint: iroh::Endpoint = todo!();
/// let _owned =
///     hyprstream_rpc::transport::iroh_substrate::OwnedIrohClientEndpoint(endpoint);
/// ```
pub fn install_iroh_client_endpoint(
    endpoint: OwnedIrohClientEndpoint,
) -> Result<(), OwnedIrohClientEndpoint> {
    endpoint.install_into(&IROH_CLIENT_ENDPOINT)
}

/// The installed client endpoint, cloned (cheap — iroh `Endpoint` is `Arc`-backed).
///
/// Exposed crate-internally so the streaming-plane dialer
/// ([`crate::dial::dial_stream`]'s iroh arm, #282) can reuse the same install-once
/// endpoint the RPC plane dials from.
pub(crate) fn iroh_client_endpoint() -> Option<iroh::Endpoint> {
    IROH_CLIENT_ENDPOINT.get().cloned()
}

/// An iroh RPC transport that connects on first use and caches the session.
pub struct LazyIrohTransport {
    node_id: [u8; 32],
    direct_addrs: Vec<SocketAddr>,
    relay_url: Option<String>,
    /// Cached session + reconnect backoff (#156).
    state: Mutex<LazyState<IrohTransport>>,
    /// Test-only dial endpoint override.
    ///
    /// Production always dials from the process-global install-once endpoint
    /// ([`iroh_client_endpoint`]); there is exactly one tokio runtime for the
    /// whole process, so that endpoint lives for the process lifetime.
    ///
    /// Under `cargo test`, however, every `#[tokio::test]` spins up and tears
    /// down its OWN runtime. iroh's `Endpoint` spawns its magicsock/router
    /// background tasks on the runtime active at `bind()`, so the install-once
    /// global endpoint dies with whichever test's runtime installed it. A later
    /// test that reuses the (now dead) global hits iroh's
    /// `RemoteStateActorStopped` → "Internal consistency error". Injecting a
    /// per-test endpoint that lives on the *current* test's runtime makes the
    /// dialing tests deterministic regardless of order (and lets them avoid
    /// polluting the global at all).
    #[cfg(test)]
    endpoint_override: Option<iroh::Endpoint>,
}

impl LazyIrohTransport {
    /// Create a lazy transport for the iroh peer identified by `node_id`. No
    /// connection is made until the first `send()`.
    pub fn new(node_id: [u8; 32], direct_addrs: Vec<SocketAddr>, relay_url: Option<String>) -> Self {
        Self {
            node_id,
            direct_addrs,
            relay_url,
            state: Mutex::new(LazyState::default()),
            #[cfg(test)]
            endpoint_override: None,
        }
    }

    /// Test-only constructor that dials from a caller-supplied endpoint instead
    /// of the process-global one. See [`Self::endpoint_override`] for why tests
    /// need this (per-test tokio runtimes vs. an install-once global endpoint).
    #[cfg(test)]
    pub(crate) fn new_with_endpoint(
        node_id: [u8; 32],
        direct_addrs: Vec<SocketAddr>,
        relay_url: Option<String>,
        endpoint: iroh::Endpoint,
    ) -> Self {
        Self {
            node_id,
            direct_addrs,
            relay_url,
            state: Mutex::new(LazyState::default()),
            endpoint_override: Some(endpoint),
        }
    }

    /// The endpoint to dial from: the injected per-test endpoint when present,
    /// otherwise the process-global install-once client endpoint. In production
    /// builds this is exactly [`iroh_client_endpoint`].
    fn dial_endpoint(&self) -> Option<iroh::Endpoint> {
        #[cfg(test)]
        if let Some(ep) = self.endpoint_override.clone() {
            return Some(ep);
        }
        iroh_client_endpoint()
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
    /// the lock) if not yet connected. Backs off after consecutive failures (#156).
    async fn connected(&self) -> Result<IrohTransport> {
        let mut guard = self.state.lock().await;
        if let Some(transport) = guard.cached.as_ref() {
            return Ok(transport.clone());
        }
        if let Some(remaining) = guard.backoff.cooldown_remaining() {
            return Err(anyhow!(
                "iroh peer is in reconnect backoff — retry in {remaining:.1?}"
            ));
        }
        let endpoint = self.dial_endpoint().ok_or_else(|| {
            anyhow!(
                "no iroh client endpoint installed — call \
                 install_iroh_client_endpoint() at startup before dialing iroh peers"
            )
        })?;
        let addr = self.endpoint_addr()?;
        match tokio::time::timeout(
            CONNECT_TIMEOUT,
            endpoint.connect(addr, ALPN_HYPRSTREAM_RPC),
        )
        .await
        {
            Err(_) => {
                guard.backoff.record_failure();
                Err(anyhow!("iroh connect timed out after {CONNECT_TIMEOUT:?}"))
            }
            Ok(Err(e)) => {
                guard.backoff.record_failure();
                Err(anyhow!("iroh connect: {e}"))
            }
            Ok(Ok(conn)) => {
                guard.backoff.record_success();
                let transport = IrohTransport::new(conn);
                guard.cached = Some(transport.clone());
                Ok(transport)
            }
        }
    }

    /// Drop the cached session and record a failure so the next `send()`
    /// re-dials after the backoff cooldown (#156).
    async fn invalidate(&self) {
        let mut guard = self.state.lock().await;
        guard.cached = None;
        guard.backoff.record_failure();
    }
}

#[async_trait]
impl Transport for LazyIrohTransport {
    type Sub = IrohPendingStream;
    type Pub = IrohPublishStub;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        let transport = self.connected().await.map_err(|error| {
            crate::transport_traits::PreDispatchTransportError::new(error)
        })?;

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

    fn forbids_cleartext_envelope(&self) -> bool {
        true
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
        let server = IrohSubstrate::new_test(fresh_key(), EchoHandler, EchoHandler).await.unwrap();
        let server_id = server.endpoint_id();
        let direct: Vec<SocketAddr> = server.endpoint().bound_sockets().into_iter().collect();

        // Client substrate (originates only). Dial from THIS test's endpoint via
        // `new_with_endpoint` rather than the process-global install-once one: the
        // global is bound to whichever test's tokio runtime installed it and dies
        // with that runtime, which made this test flake with iroh's "Internal
        // consistency error" (RemoteStateActorStopped) when it ran after another
        // iroh test. A per-test endpoint on the current runtime is deterministic.
        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("client moq"),
            NoopHandler::new("client rpc"),
        )
        .await
        .unwrap();

        let node_id: [u8; 32] = *server_id.as_bytes();
        let t = LazyIrohTransport::new_with_endpoint(node_id, direct, None, client.endpoint().clone());
        assert!(t.state.lock().await.cached.is_none(), "no connection before first send");

        // EchoHandler echoes the bytes written on the bidi stream, which is the
        // same shape SessionRpcTransport::send uses (write payload → read to
        // FIN), so the lazy transport round-trips them. (EchoHandler is a
        // one-shot smoke handler — it closes the connection after a single
        // stream, so we don't test session *reuse* here; the multiplexed-stream
        // path is covered by quinn_transport's round-trip test against a real
        // QuinnRpcServer, which IrohTransport shares via SessionRpcTransport.)
        let resp = t.send(b"ping".to_vec(), Some(4_000)).await.unwrap();
        assert_eq!(resp, b"ping");
        assert!(t.state.lock().await.cached.is_some(), "session cached after first send");

        // Release the cached connection and endpoint clone before draining both
        // routers. Abruptly dropping the client substrate while `t` still held a
        // live connection left noq driver tasks racing runtime teardown, which
        // intermittently triggered its GSO-batch assertion in the full suite.
        drop(t);
        client.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wrong_node_id_rejected() {
        // Server we'll dial the *address* of, but with a different identity.
        let server = IrohSubstrate::new_test(fresh_key(), EchoHandler, EchoHandler).await.unwrap();
        let direct: Vec<SocketAddr> = server.endpoint().bound_sockets().into_iter().collect();

        // Dial from this test's own endpoint (see `lazy_iroh_connects_*` above and
        // `endpoint_override`'s docs): the install-once global is bound to another
        // test's runtime and would flake with "Internal consistency error".
        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("client moq"),
            NoopHandler::new("client rpc"),
        )
        .await
        .unwrap();

        // A *valid* but wrong EndpointId (a third node's), so this tests handshake
        // identity rejection — not a malformed key.
        let other = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("o1"), NoopHandler::new("o2"))
            .await
            .unwrap();
        let wrong_id: [u8; 32] = *other.endpoint_id().as_bytes();

        let t = LazyIrohTransport::new_with_endpoint(wrong_id, direct, None, client.endpoint().clone());
        let res = tokio::time::timeout(Duration::from_secs(8), t.send(b"x".to_vec(), Some(3_000)))
            .await
            .expect("send must complete (with an error) — wrong identity should reject, not hang");
        assert!(res.is_err(), "dialing a server's address under a wrong EndpointId must fail");
        assert!(t.state.lock().await.cached.is_none(), "a failed handshake caches nothing");

        // As above, release the transport's endpoint clone before cleanly
        // draining every substrate owned by this test.
        drop(t);
        client.shutdown().await.unwrap();
        other.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn subscribe_publish_bail() {
        let t = LazyIrohTransport::new([0u8; 32], vec![], None);
        assert!(t.subscribe(b"topic").await.is_err());
        assert!(t.publish(b"topic").await.is_err());
    }
}
