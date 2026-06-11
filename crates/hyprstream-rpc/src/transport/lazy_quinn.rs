//! Lazy-connecting quinn RPC transport.
//!
//! The [`dial`](crate::dial::dial) factory is synchronous and does no I/O, but a
//! networked QUIC peer must be connected before the first request. This wrapper
//! bridges that: it holds the dial target (`addr` + `server_name` + the
//! [`QuicServerAuth`](crate::transport::QuicServerAuth) policy) and connects on
//! the **first `send()`**, caching the established session for subsequent calls
//! — exactly the lazy-connect model `ZmqConnection` uses, so sync construction +
//! zero I/O at dial time is preserved (A1 spike). The connect path (WebPKI vs
//! cert-hash pin) is selected by the auth policy.
//!
//! # Re-dial / self-heal
//!
//! On a `send()` error the cached session is dropped, so the next call re-dials
//! — like `ZmqConnection`, which a dead cached session would otherwise not
//! recover from. This is deliberately **coarse**: it re-dials on *any* failure,
//! including a per-request timeout on an otherwise-live connection. Refined,
//! liveness-based re-dial with backoff is the connection-manager's job (#156);
//! this wrapper only needs to not strand a dead session.
//!
//! Concurrent first-sends are **single-flighted**: the dial happens under the
//! cache lock, so N racing callers dial once and the rest reuse the session.
//! The connected handle is cloned out from under the lock (both `QuinnTransport`
//! and its `Session` are cheap `Clone`), so `send()` itself never holds the
//! lock — only the dial does.

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::transport::quinn_transport::{
    connect_pinned_sha256, connect_webpki, QuinnPendingStream, QuinnPublishStub, QuinnTransport,
};
use crate::transport::QuicServerAuth;
use crate::transport_traits::Transport;

/// Default per-request deadline when the caller passes `None`, mirroring
/// `SessionRpcTransport`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Ceiling on the lazy connect itself (TLS handshake + dial). The per-request
/// deadline wraps only the *request*; without this a dead/misrouted address (or
/// a stalled handshake) would hang the first `send()` indefinitely.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// A quinn RPC transport that connects on first use and caches the session.
pub struct LazyQuinnTransport {
    addr: SocketAddr,
    server_name: String,
    auth: QuicServerAuth,
    /// `None` until the first successful dial, and reset to `None` after a
    /// `send()` failure so the next call re-dials.
    cached: Mutex<Option<QuinnTransport>>,
}

impl LazyQuinnTransport {
    /// Create a lazy transport for a QUIC peer, authenticated per `auth`
    /// (WebPKI, cert-hash pin, or — later — RFC 7250 raw key). No connection is
    /// made until the first `send()`.
    pub fn new(addr: SocketAddr, server_name: impl Into<String>, auth: QuicServerAuth) -> Self {
        Self {
            addr,
            server_name: server_name.into(),
            auth,
            cached: Mutex::new(None),
        }
    }

    /// Return the cached connected transport, dialing once (under the lock —
    /// single-flight) if not yet connected. The connect path is chosen by the
    /// server-auth policy.
    async fn connected(&self) -> Result<QuinnTransport> {
        let mut guard = self.cached.lock().await;
        if let Some(transport) = guard.as_ref() {
            return Ok(transport.clone());
        }
        let connect = async {
            match &self.auth {
                QuicServerAuth::WebPki => connect_webpki(&self.server_name, self.addr.port()).await,
                QuicServerAuth::Pinned(hash) => connect_pinned_sha256(self.addr, *hash).await,
                QuicServerAuth::RawPublicKey(_) => bail!(
                    "RFC 7250 raw-public-key QUIC auth is not yet implemented (#200) — use iroh \
                     for identity-bound transport, or QuicServerAuth::Pinned / WebPki"
                ),
            }
        };
        let session = tokio::time::timeout(CONNECT_TIMEOUT, connect)
            .await
            .map_err(|_| anyhow!("quinn connect timed out after {CONNECT_TIMEOUT:?}"))??;
        let transport = QuinnTransport::new(session);
        *guard = Some(transport.clone());
        Ok(transport)
    }

    /// Drop the cached session so the next `send()` re-dials.
    async fn invalidate(&self) {
        *self.cached.lock().await = None;
    }
}

#[async_trait]
impl Transport for LazyQuinnTransport {
    type Sub = QuinnPendingStream;
    type Pub = QuinnPublishStub;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        let transport = self.connected().await?;

        // Make *our* deadline authoritative so we can tell a per-request timeout
        // (the connection is probably fine — keep it) apart from a transport-fatal
        // error (re-dial). The inner transport gets a generous ceiling that fires
        // only if ours somehow doesn't.
        let deadline = timeout_ms
            .map(|ms| Duration::from_millis(ms.max(0) as u64))
            .unwrap_or(DEFAULT_TIMEOUT);
        let inner_ceiling_ms = deadline
            .as_millis()
            .saturating_mul(2)
            .min(i32::MAX as u128) as i32;

        match tokio::time::timeout(deadline, transport.send(payload, Some(inner_ceiling_ms))).await {
            // Our deadline fired: a slow/stalled request, not necessarily a dead
            // connection. Keep the cached session — re-dialing on every timeout
            // would thrash a merely-busy peer. (#156 owns liveness-based re-dial.)
            Err(_elapsed) => Err(anyhow!("quinn RPC timeout after {deadline:?}")),
            Ok(Ok(resp)) => Ok(resp),
            // Transport-fatal (open_bi/write/read failed): the session is likely
            // dead — drop it so the next call re-dials, like ZmqConnection.
            Ok(Err(e)) => {
                self.invalidate().await;
                Err(e)
            }
        }
    }

    async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
        bail!("lazy quinn RPC transport does not support SUB — streaming is on the moq plane")
    }

    async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
        bail!("lazy quinn RPC transport does not support PUB — streaming is on the moq plane")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::transport::quinn_transport::QuinnRpcServer;
    use bytes::Bytes;
    use ed25519_dalek::SigningKey;
    use rand::RngCore;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    fn fresh_signing_key() -> SigningKey {
        let mut k = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut k);
        SigningKey::from_bytes(&k)
    }

    fn sha256_32(bytes: &[u8]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = [0u8; 32];
        h.copy_from_slice(&Sha256::digest(bytes));
        h
    }

    /// Hermetic loopback WebTransport echo server. Returns the bound addr, the
    /// server cert's SHA-256 pin, and a shutdown token. (Mirrors the harness in
    /// `quinn_transport::tests`; `build_server` there is test-private.)
    fn spawn_echo_server() -> (SocketAddr, [u8; 32], CancellationToken) {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let cert_key =
            rcgen::generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap();
        let cert_der = cert_key.cert.der().to_vec();
        let key_der = cert_key.key_pair.serialize_der();
        let chain = vec![rustls::pki_types::CertificateDer::from(cert_der.clone())];
        let key = rustls::pki_types::PrivateKeyDer::Pkcs8(
            rustls::pki_types::PrivatePkcs8KeyDer::from(key_der),
        );
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let server = web_transport_quinn::ServerBuilder::new()
            .with_addr(addr)
            .with_certificate(chain, key)
            .unwrap();
        let bound = server.local_addr().unwrap();

        let processor =
            crate::transport::rpc_session::from_fn(|req: Bytes| async move { Ok(req) });
        let rpc_server = QuinnRpcServer::new(server, processor, fresh_signing_key());
        let shutdown = rpc_server.shutdown_token();
        tokio::spawn(rpc_server.run());

        (bound, sha256_32(&cert_der), shutdown)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lazy_connects_on_first_send_and_caches() {
        let (addr, pin, shutdown) = spawn_echo_server();
        let t = LazyQuinnTransport::new(addr, "localhost", QuicServerAuth::Pinned(pin));

        assert!(t.cached.lock().await.is_none(), "no connection before first send");

        let resp = t.send(b"hello".to_vec(), Some(5_000)).await.unwrap();
        assert_eq!(resp, b"hello");
        assert!(t.cached.lock().await.is_some(), "session cached after first send");

        let resp2 = t.send(b"again".to_vec(), Some(5_000)).await.unwrap();
        assert_eq!(resp2, b"again");

        shutdown.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wrong_cert_pin_does_not_connect() {
        let (addr, _pin, shutdown) = spawn_echo_server();
        let t = LazyQuinnTransport::new(addr, "localhost", QuicServerAuth::Pinned([0u8; 32])); // wrong fingerprint
        // The TLS handshake must *promptly reject* a wrong pin: the send must
        // COMPLETE with an error well within the hang-guard. If the guard fires
        // (i.e. send hung), the test fails — so a hang can't masquerade as a pass.
        let res = tokio::time::timeout(Duration::from_secs(8), t.send(b"x".to_vec(), Some(3_000)))
            .await
            .expect("send must complete (with an error) — a wrong pin should reject, not hang");
        assert!(res.is_err(), "a wrong cert pin must make send fail");
        assert!(t.cached.lock().await.is_none(), "failed dial leaves nothing cached");
        shutdown.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dead_address_send_fails_fast_not_hang() {
        // No server at 127.0.0.1:1 → the lazy connect must fail (or hit
        // CONNECT_TIMEOUT) so `send()` returns an error rather than hanging
        // forever. The guard (> CONNECT_TIMEOUT) fails the test if it hangs.
        let t = LazyQuinnTransport::new(
            "127.0.0.1:1".parse().unwrap(),
            "localhost",
            QuicServerAuth::Pinned([0u8; 32]),
        );
        let res = tokio::time::timeout(Duration::from_secs(13), t.send(b"x".to_vec(), Some(2_000)))
            .await
            .expect("send must complete (with an error) within the connect timeout, not hang");
        assert!(res.is_err(), "a dead address must yield an error, not a connection");
        assert!(t.cached.lock().await.is_none(), "a failed connect caches nothing");
    }

    #[tokio::test]
    async fn subscribe_publish_bail() {
        let t = LazyQuinnTransport::new(
            "127.0.0.1:1".parse().unwrap(),
            "localhost",
            QuicServerAuth::Pinned([0u8; 32]),
        );
        assert!(t.subscribe(b"topic").await.is_err());
        assert!(t.publish(b"topic").await.is_err());
    }
}
