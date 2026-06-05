//! Quinn WebTransport RPC plane — moq M1 (#151).
//!
//! A second backend for the transport-generic RPC core (see
//! [`super::rpc_session`]), proving the RPC plane is transport-pluggable: the
//! exact same Cap'n Proto bidi wire protocol, DoS bounds, graceful-drain, and
//! error-envelope semantics run over quinn's [`web_transport_quinn::Session`]
//! instead of iroh's.
//!
//! - [`QuinnRpcServer`] — accepts WebTransport sessions on a quinn endpoint and
//!   runs [`serve_rpc_connection`] per session.
//! - [`QuinnTransport`] — client transport wrapping
//!   [`SessionRpcTransport<web_transport_quinn::Session>`].
//!
//! Unlike the iroh substrate (which terminates raw QUIC bidi streams under a
//! custom ALPN), web-transport-quinn speaks the full WebTransport handshake
//! (HTTP/3 CONNECT, ALPN `h3`). The wire framing *inside* each bidi stream is
//! identical, so the generic core is reused unchanged.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::transport::rpc_session::{
    DEFAULT_STREAM_LIMIT, IrohRequestProcessor, RpcPendingStream, RpcPublishStub,
    SessionRpcTransport, serve_rpc_connection,
};
use crate::transport_traits::Transport;

/// A quinn-backed WebTransport server for the RPC plane.
///
/// Accepts WebTransport sessions and spawns [`serve_rpc_connection`] for each.
/// Caps concurrent streams *server-wide* via a single shared [`Semaphore`] (DoS
/// bound) passed into every connection, matching the iroh handler's
/// shared-semaphore drain. [`QuinnRpcServer::shutdown`] cancels the accept loop
/// and then drains all in-flight streams before returning, so detached
/// `handle_stream` tasks finish writing their responses rather than being
/// abandoned mid-flight.
pub struct QuinnRpcServer {
    server: web_transport_quinn::Server,
    processor: Arc<dyn IrohRequestProcessor>,
    signing_key: SigningKey,
    /// Server-wide concurrent-stream cap. Shared across all connections; one
    /// permit is held per in-flight bidi stream. `shutdown` drains it via
    /// `acquire_many` to wait for every in-flight stream to complete.
    stream_limit: Arc<Semaphore>,
    stream_limit_capacity: u32,
    /// Per-stream request-read timeout (#159 slowloris bound).
    read_timeout: Duration,
    shutdown: CancellationToken,
}

impl QuinnRpcServer {
    /// Build a server from a quinn endpoint configured for WebTransport (ALPN
    /// `h3`). Use [`web_transport_quinn::ServerBuilder`] to construct one.
    pub fn new<P: IrohRequestProcessor>(
        server: web_transport_quinn::Server,
        processor: P,
        signing_key: SigningKey,
    ) -> Self {
        Self::with_capacity(server, Arc::new(processor), signing_key, DEFAULT_STREAM_LIMIT)
    }

    /// Build a server with an explicit server-wide concurrent-stream cap.
    pub fn with_capacity(
        server: web_transport_quinn::Server,
        processor: Arc<dyn IrohRequestProcessor>,
        signing_key: SigningKey,
        stream_limit: usize,
    ) -> Self {
        Self {
            server,
            processor,
            signing_key,
            stream_limit: Arc::new(Semaphore::new(stream_limit)),
            stream_limit_capacity: u32::try_from(stream_limit).unwrap_or(u32::MAX),
            read_timeout: super::rpc_session::REQUEST_READ_TIMEOUT,
            shutdown: CancellationToken::new(),
        }
    }

    /// Override the server-wide concurrent-stream cap (builder style).
    pub fn with_stream_limit(mut self, limit: usize) -> Self {
        self.stream_limit = Arc::new(Semaphore::new(limit));
        self.stream_limit_capacity = u32::try_from(limit).unwrap_or(u32::MAX);
        self
    }

    /// Override the per-stream request-read timeout (#159). Primarily for
    /// tests; production uses [`super::rpc_session::REQUEST_READ_TIMEOUT`].
    pub fn with_read_timeout(mut self, read_timeout: Duration) -> Self {
        self.read_timeout = read_timeout;
        self
    }

    /// A handle that, when cancelled, stops the accept loop and per-connection
    /// serve loops.
    ///
    /// Note: cancelling this token alone does **not** drain in-flight streams.
    /// To wait for in-flight responses to complete, hold a clone of the server
    /// and call [`QuinnRpcServer::shutdown`] instead (or use it via a shared
    /// handle). The token is exposed primarily for tests and for callers that
    /// want to stop accepting without a graceful drain.
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// Graceful shutdown mirroring [`super::iroh_rpc::IrohRpcProtocolHandler::shutdown`]:
    /// cancel the accept loop, then wait for every in-flight stream to release
    /// its permit (`acquire_many(capacity)`), then `forget()` + `close()` the
    /// semaphore so any post-shutdown accept sees a closed semaphore and exits.
    ///
    /// Takes the shared `Semaphore` + token by reference, so it can be called
    /// through a clone of the relevant handles while [`QuinnRpcServer::run`]
    /// owns `self`. See the `quinn_shutdown_drains_in_flight` test.
    pub async fn shutdown(stream_limit: &Arc<Semaphore>, capacity: u32, token: &CancellationToken) {
        // Stop accepting new streams (level-triggered).
        token.cancel();
        // Wait for all in-flight streams to release their permits, but bound
        // the wait (#159): a wedged processor/transport must not hang shutdown
        // forever. On timeout we close() and proceed — remaining tasks are torn
        // down when the connection drops.
        match tokio::time::timeout(
            super::rpc_session::DRAIN_TIMEOUT,
            stream_limit.acquire_many(capacity),
        )
        .await
        {
            Ok(Ok(permits)) => {
                permits.forget();
                stream_limit.close();
            }
            Ok(Err(_)) => {
                // Already closed; nothing to drain.
            }
            Err(_) => {
                tracing::warn!(
                    timeout = ?super::rpc_session::DRAIN_TIMEOUT,
                    "quinn-rpc: drain timed out, forcing teardown"
                );
                stream_limit.close();
            }
        }
    }

    /// The shared concurrent-stream semaphore. Pair with [`QuinnRpcServer::capacity`]
    /// and [`QuinnRpcServer::shutdown_token`] to perform a graceful
    /// [`QuinnRpcServer::shutdown`] while [`QuinnRpcServer::run`] owns `self`.
    pub fn stream_limit(&self) -> Arc<Semaphore> {
        Arc::clone(&self.stream_limit)
    }

    /// The configured server-wide concurrent-stream capacity.
    pub fn capacity(&self) -> u32 {
        self.stream_limit_capacity
    }

    /// Run the accept loop until `shutdown` is cancelled. Each accepted session
    /// is served on its own task by the transport-generic core, sharing the
    /// server-wide [`Semaphore`] so [`QuinnRpcServer::shutdown`] can drain every
    /// in-flight stream regardless of which connection it belongs to.
    pub async fn run(mut self) -> Result<()> {
        loop {
            tokio::select! {
                biased;
                _ = self.shutdown.cancelled() => {
                    tracing::debug!("quinn-rpc: accept loop cancelled");
                    return Ok(());
                }
                request = self.server.accept() => {
                    let Some(request) = request else {
                        tracing::debug!("quinn-rpc: server endpoint closed");
                        return Ok(());
                    };
                    let session = match request.ok().await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!(error = ?e, "quinn-rpc: handshake failed");
                            continue;
                        }
                    };
                    let processor = Arc::clone(&self.processor);
                    let signing_key = self.signing_key.clone();
                    let stream_limit = Arc::clone(&self.stream_limit);
                    let read_timeout = self.read_timeout;
                    let shutdown = self.shutdown.clone();
                    tokio::spawn(async move {
                        if let Err(e) = serve_rpc_connection(
                            session,
                            processor,
                            signing_key,
                            stream_limit,
                            read_timeout,
                            shutdown,
                        )
                        .await
                        {
                            tracing::debug!(error = ?e, "quinn-rpc: connection serve ended");
                        }
                    });
                }
            }
        }
    }
}

/// Wraps a quinn [`web_transport_quinn::Session`] as a [`Transport`] for RPC
/// plane traffic. Delegates to the transport-generic [`SessionRpcTransport`].
#[derive(Clone)]
pub struct QuinnTransport {
    inner: SessionRpcTransport<web_transport_quinn::Session>,
}

impl QuinnTransport {
    /// Build from an already-established WebTransport session.
    pub fn new(session: web_transport_quinn::Session) -> Self {
        Self {
            inner: SessionRpcTransport::new(session),
        }
    }
}

/// Re-export the generic stubs under quinn-flavoured names.
pub type QuinnPendingStream = RpcPendingStream;
pub type QuinnPublishStub = RpcPublishStub;

#[async_trait]
impl Transport for QuinnTransport {
    type Sub = QuinnPendingStream;
    type Pub = QuinnPublishStub;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        self.inner.send(payload, timeout_ms).await
    }

    async fn subscribe(&self, topic: &[u8]) -> Result<Self::Sub> {
        self.inner.subscribe(topic).await
    }

    async fn publish(&self, topic: &[u8]) -> Result<Self::Pub> {
        self.inner.publish(topic).await
    }
}

/// Helper to build a quinn WebTransport client session against a self-signed
/// server, pinning the server cert by its sha256 fingerprint. Hermetic: dials
/// an IP-literal URL so no DNS lookup or network egress occurs.
pub async fn connect_pinned(
    addr: std::net::SocketAddr,
    cert_der: &[u8],
) -> Result<web_transport_quinn::Session> {
    let client = web_transport_quinn::ClientBuilder::new()
        .with_server_certificate_hashes(vec![sha256(cert_der)])
        .map_err(|e| anyhow!("quinn client build: {e}"))?;
    let url = url::Url::parse(&format!("https://{addr}/"))
        .map_err(|e| anyhow!("quinn url: {e}"))?;
    client
        .connect(url)
        .await
        .map_err(|e| anyhow!("quinn connect: {e}"))
}

fn sha256(bytes: &[u8]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use rand::RngCore;
    use std::time::Duration;

    fn fresh_signing_key() -> SigningKey {
        let mut k = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut k);
        SigningKey::from_bytes(&k)
    }

    /// Build a hermetic quinn WebTransport server bound to loopback with a
    /// self-signed cert. Returns (server, bound_addr, cert_der).
    fn build_server() -> Result<(web_transport_quinn::Server, std::net::SocketAddr, Vec<u8>)> {
        let cert_key = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()])?;
        let cert_der = cert_key.cert.der().to_vec();
        let key_der = cert_key.key_pair.serialize_der();

        let chain = vec![rustls::pki_types::CertificateDer::from(cert_der.clone())];
        let key = rustls::pki_types::PrivateKeyDer::Pkcs8(
            rustls::pki_types::PrivatePkcs8KeyDer::from(key_der),
        );

        let addr: std::net::SocketAddr = "127.0.0.1:0".parse()?;
        let server = web_transport_quinn::ServerBuilder::new()
            .with_addr(addr)
            .with_certificate(chain, key)
            .map_err(|e| anyhow!("quinn server build: {e}"))?;
        let bound = server.local_addr()?;
        Ok((server, bound, cert_der))
    }

    /// Round-trip: client sends a request, a closure-processor echoes it back
    /// with a 0xCD marker prefix, client asserts on the response.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn quinn_rpc_round_trip() -> Result<()> {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let processor = crate::transport::rpc_session::from_fn(|req: Bytes| async move {
            let mut out = Vec::with_capacity(1 + req.len());
            out.push(0xCD);
            out.extend_from_slice(&req);
            Ok(Bytes::from(out))
        });

        let (server, addr, cert_der) = build_server()?;
        let rpc_server = QuinnRpcServer::new(server, processor, fresh_signing_key());
        let shutdown = rpc_server.shutdown_token();
        let server_task = tokio::spawn(rpc_server.run());

        let session = connect_pinned(addr, &cert_der).await?;
        let client = QuinnTransport::new(session);

        let resp = client.send(b"ping".to_vec(), Some(5_000)).await?;
        assert_eq!(&resp[..], b"\xCDping");

        shutdown.cancel();
        let _ = server_task.await;
        Ok(())
    }

    /// Graceful shutdown drains in-flight requests: a request in-progress when
    /// `shutdown()` starts still completes with its real response rather than
    /// being abandoned mid-flight. Mirrors `iroh_rpc::rpc_shutdown_drains_in_flight`.
    ///
    /// Uses a `Notify` to synchronise "processor has entered the long sleep"
    /// with "now safe to shut down" — no wall-clock sleep races.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn quinn_shutdown_drains_in_flight() -> Result<()> {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let entered = Arc::new(tokio::sync::Notify::new());
        let entered_c = Arc::clone(&entered);
        let processor = crate::transport::rpc_session::from_fn(move |_req: Bytes| {
            let entered = Arc::clone(&entered_c);
            async move {
                entered.notify_one();
                tokio::time::sleep(Duration::from_millis(300)).await;
                Ok(Bytes::from_static(b"drained-ok"))
            }
        });

        let (server, addr, cert_der) = build_server()?;
        let rpc_server = QuinnRpcServer::new(server, processor, fresh_signing_key());
        // Grab the shared drain handles before `run()` consumes the server.
        let stream_limit = rpc_server.stream_limit();
        let capacity = rpc_server.capacity();
        let token = rpc_server.shutdown_token();
        let server_task = tokio::spawn(rpc_server.run());

        let session = connect_pinned(addr, &cert_der).await?;
        let client = QuinnTransport::new(session);

        // Fire the slow request on its own task.
        let req_task = {
            let client = client.clone();
            tokio::spawn(async move { client.send(b"slow".to_vec(), Some(10_000)).await })
        };

        // Synchronise: wait until the processor has entered the sleep, then
        // drain-shutdown. No wall-clock guesses.
        entered.notified().await;
        QuinnRpcServer::shutdown(&stream_limit, capacity, &token).await;

        // The in-flight request must still complete with its real response.
        let resp = req_task.await??;
        assert_eq!(&resp[..], b"drained-ok");

        let _ = server_task.await;
        Ok(())
    }
}
