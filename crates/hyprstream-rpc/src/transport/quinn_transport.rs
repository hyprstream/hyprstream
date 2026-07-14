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

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::transport::rpc_session::{
    DEFAULT_STREAM_LIMIT, IrohRequestProcessor, RpcPendingStream, RpcPublishStub,
    SessionRpcTransport, serve_rpc_connection,
};
use crate::transport_traits::Transport;

/// A WebTransport bidi stream carrying a 9P2000.L session, erased to a byte
/// stream. The H1b `/9p` arm hands one of these to the injected
/// [`NinePWtHandler`]; the handler runs the transport-agnostic
/// `Translator::serve_connection` core over it (the same core H1a's WebSocket
/// pump feeds). Blanket-implemented for any `AsyncRead + AsyncWrite` byte stream
/// (a joined WT send/recv pair).
pub trait NinePWtStream:
    tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + Unpin + 'static
{
}
impl<T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + Unpin + 'static> NinePWtStream for T {}

/// Serves one 9P session over a WebTransport bidi stream (H1b / #765).
///
/// This is the injection seam that keeps the transport crate free of the 9P
/// core: the concrete impl lives in the `hyprstream` crate (it owns the 9P
/// `Translator`, the Subject-scoped export mount, and the mount-ticket
/// validator), and is registered here via [`QuinnRpcServer::with_ninep_handler`]
/// or the process-global [`set_global_ninep_handler`] — mirroring how the moq
/// consumer is threaded onto the same endpoint. The mount ticket is validated
/// at `Tattach.uname` inside the handler (the cert-pinned session carries no URL
/// query), so this trait exposes only the raw byte stream.
#[async_trait]
pub trait NinePWtHandler: Send + Sync {
    /// Drive the 9P session on `stream` to completion (returns on EOF, the peer
    /// resetting the stream, or a fatal 9P framing error). The handler owns
    /// session teardown.
    async fn serve(&self, stream: Box<dyn NinePWtStream>);
}

/// Process-global 9P WebTransport handler, populated by the `hyprstream` crate
/// at server startup (where `ServerState` — hence the export mount + ticket
/// validator — is available). The per-service `QuinnRpcServer` build site lives
/// in `hyprstream-service`, which cannot depend on `hyprstream`; a global here
/// is the same seam the moq origin uses to cross that boundary. First write
/// wins; later writes are ignored.
static GLOBAL_NINEP_HANDLER: std::sync::OnceLock<Arc<dyn NinePWtHandler>> =
    std::sync::OnceLock::new();

/// Register the process-global 9P WebTransport handler (idempotent, first-wins).
/// Returns `true` if this call installed it, `false` if one was already set.
pub fn set_global_ninep_handler(handler: Arc<dyn NinePWtHandler>) -> bool {
    GLOBAL_NINEP_HANDLER.set(handler).is_ok()
}

/// The process-global 9P WebTransport handler, if one has been registered.
pub fn global_ninep_handler() -> Option<Arc<dyn NinePWtHandler>> {
    GLOBAL_NINEP_HANDLER.get().cloned()
}

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
    /// Cap on concurrent accepted connections (#162). One permit per live
    /// connection; connections beyond the cap are rejected, not queued.
    connection_limit: Arc<Semaphore>,
    /// Per-stream request-read timeout (#159 slowloris bound).
    read_timeout: Duration,
    /// Optional moq streaming-plane consumer (#274). When set, sessions whose
    /// WebTransport CONNECT URL path is [`crate::dial::MOQ_PATH`] are handed to
    /// `moq_net::Server::accept` instead of the RPC core. `None` = RPC only.
    moq_consumer: Option<moq_net::OriginConsumer>,
    /// Optional moq RELAY origin (#358). When set, the `/moq` plane runs in
    /// *relay* (bidirectional) mode: `moq_net::Server::with_origin` ingests a
    /// connected producer's announced broadcasts into this origin AND re-serves
    /// them to subscribers by track name — so a publisher and a subscriber
    /// rendezvous through this node without either dialing the other. Takes
    /// precedence over [`Self::moq_consumer`] (serve-only). The relay carries the
    /// AEAD-sealed, chained-HMAC frames opaquely and holds no stream keys.
    moq_relay_origin: Option<moq_net::OriginProducer>,
    /// #276 subscribe-authz + per-tenant announce-scoping config for the `/moq`
    /// plane. Defaults to "off" (open subscribe preserved).
    moq_authz: crate::transport::iroh_moq::MoqAuthzConfig,
    /// Optional 9P export handler (H1b / #765). When set, sessions whose
    /// WebTransport CONNECT URL path is [`crate::dial::NINEP_PATH`] are handed to
    /// it instead of the RPC core. Defaults to [`global_ninep_handler`] at
    /// construction; `None` = 9P plane off (unknown-path sessions fall through to
    /// the RPC core, matching pre-H1b behaviour).
    ninep_handler: Option<Arc<dyn NinePWtHandler>>,
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
            connection_limit: Arc::new(Semaphore::new(
                super::rpc_session::DEFAULT_CONNECTION_LIMIT,
            )),
            read_timeout: super::rpc_session::REQUEST_READ_TIMEOUT,
            moq_consumer: None,
            moq_relay_origin: None,
            moq_authz: crate::transport::iroh_moq::MoqAuthzConfig::default(),
            // Pick up the process-global 9P handler if the `hyprstream` crate
            // registered one; a builder call can still override it.
            ninep_handler: global_ninep_handler(),
            shutdown: CancellationToken::new(),
        }
    }

    /// Serve the moq streaming plane on the same endpoint (#274).
    ///
    /// Sessions whose WebTransport CONNECT URL path is [`crate::dial::MOQ_PATH`]
    /// are handed to `moq_net::Server::accept` against `consumer`; all other
    /// paths route to the RPC core. Mirrors
    /// [`crate::transport::iroh_moq::IrohMoqProtocolHandler`], which serves moq
    /// over a per-connection iroh ALPN — here the multiplex is by URL path.
    pub fn with_moq_consumer(mut self, consumer: moq_net::OriginConsumer) -> Self {
        self.moq_consumer = Some(consumer);
        self
    }

    /// Run the `/moq` plane as a RELAY (bidirectional) over `origin` (#358).
    ///
    /// In relay mode each `/moq` session is served via
    /// `moq_net::Server::with_origin`, so a connected producer's announced
    /// broadcasts are ingested into `origin` and re-served to every subscriber by
    /// track name. This is the rendezvous endpoint a producer advertises as a
    /// `Role::Relay` reach: a publisher links UP to it
    /// ([`crate::moq_stream::serve_origin_to_relay_background`]) and a subscriber
    /// dials it for the SAME `broadcastPath` — neither dials the other.
    ///
    /// The relay is **blind by construction**: frames are AEAD-sealed and
    /// chained-HMAC'd at the source; this node holds no `enc_key` / `mac_key` and
    /// forwards opaque `Bytes`. Takes precedence over [`Self::with_moq_consumer`].
    pub fn with_moq_relay(mut self, origin: moq_net::OriginProducer) -> Self {
        self.moq_relay_origin = Some(origin);
        self
    }

    /// Install the #276 subscribe-authz + per-tenant announce-scoping config for
    /// the `/moq` plane.
    ///
    /// **Identity caveat (documented seam):** the WebTransport `/moq` CONNECT is
    /// *not* mutually authenticated — the server presents a pinned cert, but the
    /// client is anonymous at the TLS layer. So the `tenant_resolver` is invoked
    /// with [`crate::moq_authz::PeerIdentity::anonymous`]; per-tenant scoping on
    /// this path is only effective if the resolver returns a tenant from
    /// non-identity context (e.g. a single-tenant endpoint), and a policy-gated
    /// authorizer will deny *private* subscribes here (fail-closed) while public
    /// broadcasts stay open. Cross-tenant enumeration defense for authenticated
    /// peers lives on the iroh `moql` path.
    pub fn with_moq_authz(
        mut self,
        authz: crate::transport::iroh_moq::MoqAuthzConfig,
    ) -> Self {
        self.moq_authz = authz;
        self
    }

    /// Serve the 9P export plane on the same endpoint (H1b / #765).
    ///
    /// Sessions whose WebTransport CONNECT URL path is [`crate::dial::NINEP_PATH`]
    /// are handed to `handler` (which runs the 9P `Translator::serve_connection`
    /// core over the WT bidi stream); all other paths route to the RPC core (or
    /// the moq plane for `/moq`). Overrides any process-global handler picked up
    /// at construction.
    pub fn with_ninep_handler(mut self, handler: Arc<dyn NinePWtHandler>) -> Self {
        self.ninep_handler = Some(handler);
        self
    }

    /// Override the concurrent-connection cap (#162, builder style).
    pub fn with_connection_limit(mut self, limit: usize) -> Self {
        self.connection_limit = Arc::new(Semaphore::new(limit));
        self
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

    /// Apply all tunables from an [`super::rpc_session::RpcConfig`] in one call (#197).
    pub fn with_rpc_config(self, cfg: &super::rpc_session::RpcConfig) -> Self {
        self.with_stream_limit(cfg.stream_limit)
            .with_connection_limit(cfg.connection_limit)
            .with_read_timeout(cfg.request_read_timeout)
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

                    // Connection cap (#162): take a permit before doing any
                    // per-connection work. try_acquire → reject (drop) when at
                    // cap rather than queueing, so a flood can't build backlog.
                    let conn_permit = match Arc::clone(&self.connection_limit).try_acquire_owned() {
                        Ok(p) => p,
                        Err(_) => {
                            tracing::warn!("quinn-rpc: connection cap reached, rejecting connection");
                            drop(request);
                            continue;
                        }
                    };

                    let processor = Arc::clone(&self.processor);
                    let signing_key = self.signing_key.clone();
                    let stream_limit = Arc::clone(&self.stream_limit);
                    let read_timeout = self.read_timeout;
                    let shutdown = self.shutdown.clone();
                    let moq_consumer = self.moq_consumer.clone();
                    let moq_relay_origin = self.moq_relay_origin.clone();
                    let moq_authz = self.moq_authz.clone();
                    let ninep_handler = self.ninep_handler.clone();
                    tokio::spawn(async move {
                        let _conn_permit = conn_permit; // released when this connection ends
                        // Multiplex by WebTransport CONNECT URL path (#274, H1b): read
                        // it BEFORE `request.ok()` consumes the request. `/moq`
                        // → moq streaming plane; `/9p` → 9P export plane (#765);
                        // any other path → RPC core.
                        // (`Request` Derefs to `ConnectRequest`, exposing `url`.)
                        let is_moq = request.url.path() == crate::dial::MOQ_PATH;
                        let is_ninep = request.url.path() == crate::dial::NINEP_PATH;
                        // Resolve the handshake INSIDE the task, bounded (#162),
                        // so a slow/stalled CONNECT never blocks the accept loop.
                        let session = match tokio::time::timeout(
                            super::rpc_session::HANDSHAKE_TIMEOUT,
                            request.ok(),
                        )
                        .await
                        {
                            Ok(Ok(s)) => s,
                            Ok(Err(e)) => {
                                tracing::warn!(error = ?e, "quinn-rpc: handshake failed");
                                return;
                            }
                            Err(_) => {
                                tracing::warn!(
                                    timeout = ?super::rpc_session::HANDSHAKE_TIMEOUT,
                                    "quinn-rpc: handshake timed out"
                                );
                                return;
                            }
                        };

                        if is_moq {
                            // Relay (bidirectional) mode takes precedence (#358):
                            // ingest a producer's announced broadcasts into the
                            // shared origin AND re-serve them to subscribers by
                            // track name, so publisher and subscriber rendezvous
                            // here without either dialing the other. The relay
                            // holds no stream keys — frames pass through opaquely.
                            if let Some(relay_origin) = moq_relay_origin {
                                let server = moq_net::Server::new().with_origin(relay_origin);
                                match server.accept(session).await {
                                    Ok(moq_session) => {
                                        tokio::select! {
                                            biased;
                                            _ = shutdown.cancelled() => {}
                                            res = moq_session.closed() => {
                                                tracing::debug!(result = ?res, "quinn-moq-relay: session closed");
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::debug!(error = ?e, "quinn-moq-relay: accept failed");
                                    }
                                }
                                return;
                            }
                            match moq_consumer {
                                Some(consumer) => {
                                    // #276: the `/moq` WebTransport CONNECT is NOT
                                    // mutually authenticated, so no peer identity
                                    // is available — treat the peer as anonymous.
                                    let peer = crate::moq_authz::PeerIdentity::anonymous();
                                    // Per-tenant announce scoping: only effective
                                    // if the resolver yields a tenant from
                                    // non-identity context (e.g. a single-tenant
                                    // endpoint). With no resolver, serve unscoped
                                    // (preserves the pre-#276 open model).
                                    // TODO(#276): when the `/moq` plane gains a
                                    // mutually-authenticated peer identity (client
                                    // cert / app-level token), derive the tenant
                                    // from it here instead of `anonymous()` so
                                    // cross-tenant enumeration defense applies to
                                    // network subscribers, not just iroh peers.
                                    let publish_consumer = match moq_authz.tenant_for(&peer) {
                                        Some(tenant) => {
                                            match crate::moq_authz::tenant_scoped_consumer(&consumer, &tenant) {
                                                Some(scoped) => scoped,
                                                None => {
                                                    tracing::debug!(%tenant, "quinn-moq: tenant has no visible broadcasts; dropping session");
                                                    return;
                                                }
                                            }
                                        }
                                        None => consumer,
                                    };
                                    // See iroh_moq::accept for why the authorizer
                                    // is not enforced per-track here (moq-net has
                                    // no subscribe callback); structural scoping
                                    // is the live enforcement.
                                    let _authorizer = moq_authz.authorizer.as_ref();
                                    // Serve moq over this session, mirroring
                                    // IrohMoqProtocolHandler: publish the shared
                                    // origin consumer to remote subscribers.
                                    let server = moq_net::Server::new().with_publish(publish_consumer);
                                    match server.accept(session).await {
                                        Ok(moq_session) => {
                                            // Hold the session until it closes or
                                            // shutdown is requested.
                                            tokio::select! {
                                                biased;
                                                _ = shutdown.cancelled() => {}
                                                res = moq_session.closed() => {
                                                    tracing::debug!(result = ?res, "quinn-moq: session closed");
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            tracing::debug!(error = ?e, "quinn-moq: accept failed");
                                        }
                                    }
                                }
                                None => {
                                    tracing::warn!(
                                        "quinn-rpc: /moq requested but no moq consumer configured"
                                    );
                                }
                            }
                            return;
                        }

                        if is_ninep {
                            // 9P export plane (H1b / #765): the QUIC path-mux
                            // sibling of H1a's axum `/9p` WebSocket. Serve the
                            // SAME `Translator::serve_connection` core (via the
                            // injected handler) over a WT bidi stream.
                            let Some(handler) = ninep_handler else {
                                tracing::warn!(
                                    "quinn-rpc: /9p requested but no 9P handler configured"
                                );
                                return;
                            };
                            // A stream permit bounds concurrency server-wide and
                            // lets `shutdown` drain the in-flight session (same
                            // DoS/drain contract as the RPC streams). If the
                            // semaphore is closed (shutdown drained it), exit.
                            let _permit = match Arc::clone(&stream_limit).acquire_owned().await {
                                Ok(p) => p,
                                Err(_) => return,
                            };
                            // Accept the single client-opened bidi stream (bounded
                            // so a session that never opens one can't pin the
                            // permit). `RecvStream`/`SendStream` impl tokio
                            // `AsyncRead`/`AsyncWrite`, so joining them yields the
                            // byte stream the 9P core serves directly — the WT
                            // analogue of H1a's ws↔9p pump, tolerant of arbitrary
                            // chunking.
                            let accept = tokio::time::timeout(
                                super::rpc_session::HANDSHAKE_TIMEOUT,
                                session.accept_bi(),
                            )
                            .await;
                            let (send, recv) = match accept {
                                Ok(Ok(pair)) => pair,
                                Ok(Err(e)) => {
                                    tracing::debug!(error = ?e, "quinn-9p: accept_bi failed");
                                    return;
                                }
                                Err(_) => {
                                    tracing::warn!("quinn-9p: client opened no bidi stream in time");
                                    return;
                                }
                            };
                            let stream: Box<dyn NinePWtStream> =
                                Box::new(tokio::io::join(recv, send));
                            // Liveness (H1b, mirrors H1a's server-owned teardown):
                            // the handler's serve loop returns on stream EOF / a
                            // fatal 9P error; a dead peer is torn down by the
                            // cert-pinned QUIC session's idle timeout (transport-
                            // owned, the WT/reach layer) which resets the stream →
                            // EOF. We also stop on server shutdown and on the WT
                            // session closing, whichever comes first.
                            tokio::select! {
                                biased;
                                _ = shutdown.cancelled() => {}
                                _ = handler.serve(stream) => {}
                                err = session.closed() => {
                                    tracing::debug!(?err, "quinn-9p: WT session closed");
                                }
                            }
                            return;
                        }

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

    fn forbids_cleartext_envelope(&self) -> bool {
        true
    }

    async fn subscribe(&self, topic: &[u8]) -> Result<Self::Sub> {
        self.inner.subscribe(topic).await
    }

    async fn publish(&self, topic: &[u8]) -> Result<Self::Pub> {
        self.inner.publish(topic).await
    }
}

/// Build a quinn WebTransport client session against a server with a CA-issued
/// certificate, validated via the system root store and the DNS `server_name`
/// (`QuicServerAuth::web_pki`). Dials `https://{server_name}:{port}/`.
pub async fn connect_webpki(
    server_name: &str,
    port: u16,
) -> Result<web_transport_quinn::Session> {
    let client = web_transport_quinn::ClientBuilder::new()
        .with_system_roots()
        .map_err(|e| anyhow!("quinn client (system roots): {e}"))?;
    let url = url::Url::parse(&format!("https://{server_name}:{port}/"))
        .map_err(|e| anyhow!("quinn url: {e}"))?;
    client
        .connect(url)
        .await
        .map_err(|e| anyhow!("quinn connect (webpki): {e}"))
}

/// Build a quinn WebTransport client session against a self-signed server,
/// accepting the leaf cert if its **SHA-256 fingerprint** is any of
/// `cert_hashes` (`QuicServerAuth::accept_cert_hashes` — a set so rotation can
/// overlap). Hermetic: dials an IP-literal URL so no DNS lookup or network
/// egress occurs.
pub async fn connect_pinned_hashes(
    addr: std::net::SocketAddr,
    cert_hashes: &[[u8; 32]],
) -> Result<web_transport_quinn::Session> {
    let hashes: Vec<Vec<u8>> = cert_hashes.iter().map(|h| h.to_vec()).collect();
    let client = web_transport_quinn::ClientBuilder::new()
        .with_server_certificate_hashes(hashes)
        .map_err(|e| anyhow!("quinn client build: {e}"))?;
    let url = url::Url::parse(&format!("https://{addr}/"))
        .map_err(|e| anyhow!("quinn url: {e}"))?;
    client
        .connect(url)
        .await
        .map_err(|e| anyhow!("quinn connect: {e}"))
}

/// Verify the peer's leaf-cert SHA-256 is one of `accept` — the post-connect pin
/// check for the WebPKI+pin mode (CA validation runs in the handshake; this adds
/// the pin on top). Call immediately after connect, before any RPC. The cert
/// fingerprint is public, so a plain comparison is fine (no secret to leak).
pub fn verify_peer_cert_pinned(
    session: &web_transport_quinn::Session,
    accept: &[[u8; 32]],
) -> Result<()> {
    // `Session` derefs to `quinn::Connection`, exposing `peer_identity()`.
    let identity = session
        .peer_identity()
        .ok_or_else(|| anyhow!("peer presented no certificate"))?;
    let certs = identity
        .downcast_ref::<Vec<rustls::pki_types::CertificateDer<'static>>>()
        .ok_or_else(|| anyhow!("unexpected peer-identity type (not an X.509 chain)"))?;
    let leaf = certs
        .first()
        .ok_or_else(|| anyhow!("peer cert chain is empty"))?;
    let mut leaf_hash = [0u8; 32];
    leaf_hash.copy_from_slice(&sha256(leaf.as_ref()));
    if accept.contains(&leaf_hash) {
        Ok(())
    } else {
        bail!("peer leaf-cert SHA-256 is not in the configured pin set")
    }
}

/// Build a quinn WebTransport client session against a self-signed server at an
/// explicit URL **path** (e.g. `/moq` for the moq plane vs `/` for RPC),
/// accepting the leaf cert if its SHA-256 is any of `cert_hashes`. Hermetic:
/// dials an IP-literal URL so no DNS lookup occurs. Mirrors
/// [`connect_pinned_hashes`] but lets the caller select the path the server
/// path-dispatches on (#274).
pub async fn connect_pinned_hashes_path(
    addr: std::net::SocketAddr,
    cert_hashes: &[[u8; 32]],
    path: &str,
) -> Result<web_transport_quinn::Session> {
    let hashes: Vec<Vec<u8>> = cert_hashes.iter().map(|h| h.to_vec()).collect();
    let client = web_transport_quinn::ClientBuilder::new()
        .with_server_certificate_hashes(hashes)
        .map_err(|e| anyhow!("quinn client build: {e}"))?;
    let url = url::Url::parse(&format!("https://{addr}{path}"))
        .map_err(|e| anyhow!("quinn url: {e}"))?;
    client
        .connect(url)
        .await
        .map_err(|e| anyhow!("quinn connect: {e}"))
}

/// WebPKI variant of [`connect_pinned_hashes_path`]: validate via the system
/// root store + DNS `server_name`, dialing the given URL `path` (#274).
pub async fn connect_webpki_path(
    server_name: &str,
    port: u16,
    path: &str,
) -> Result<web_transport_quinn::Session> {
    let client = web_transport_quinn::ClientBuilder::new()
        .with_system_roots()
        .map_err(|e| anyhow!("quinn client (system roots): {e}"))?;
    let url = url::Url::parse(&format!("https://{server_name}:{port}{path}"))
        .map_err(|e| anyhow!("quinn url: {e}"))?;
    client
        .connect(url)
        .await
        .map_err(|e| anyhow!("quinn connect (webpki): {e}"))
}

/// Convenience: pin by the full server cert DER (hashes it for you). Used by
/// tests and callers that hold the cert rather than its fingerprint.
pub async fn connect_pinned(
    addr: std::net::SocketAddr,
    cert_der: &[u8],
) -> Result<web_transport_quinn::Session> {
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&sha256(cert_der));
    connect_pinned_hashes(addr, &[hash]).await
}

fn sha256(bytes: &[u8]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).to_vec()
}

/// SHA-256 fingerprint of a DER-encoded certificate, as 32 raw bytes — the
/// cert-hash pin used in `QuicServerAuth`, the DID-doc `#quic` entry, and the
/// `StreamInfo.reach` Quic option (#274). The fingerprint is public material.
pub fn cert_sha256(cert_der: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(&sha256(cert_der));
    out
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
        crate::transport::pq_provider::install_pq_crypto_provider();

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

    /// Isolated test of the WebPKI+pin post-connect check: against the hermetic
    /// self-signed server, the leaf hash must match the pin set (right => Ok,
    /// wrong => Err, empty set => Err = fail-closed).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn verify_peer_cert_pinned_accepts_right_rejects_wrong() -> Result<()> {
        crate::transport::pq_provider::install_pq_crypto_provider();
        let processor =
            crate::transport::rpc_session::from_fn(|req: Bytes| async move { Ok(req) });
        let (server, addr, cert_der) = build_server()?;
        let rpc_server = QuinnRpcServer::new(server, processor, fresh_signing_key());
        let shutdown = rpc_server.shutdown_token();
        let server_task = tokio::spawn(rpc_server.run());

        let session = connect_pinned(addr, &cert_der).await?;
        let mut right = [0u8; 32];
        right.copy_from_slice(&sha256(&cert_der));

        assert!(verify_peer_cert_pinned(&session, &[right]).is_ok(), "matching leaf hash must pass");
        assert!(verify_peer_cert_pinned(&session, &[[0u8; 32]]).is_err(), "wrong hash must reject");
        assert!(verify_peer_cert_pinned(&session, &[]).is_err(), "empty pin set must reject (fail-closed)");

        shutdown.cancel();
        let _ = server_task.await;
        Ok(())
    }

    /// #162: with a connection cap of 1, a second concurrent connection is
    /// rejected while the first is still live. The first keeps working.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn quinn_connection_cap_rejects_excess() -> Result<()> {
        crate::transport::pq_provider::install_pq_crypto_provider();

        let processor =
            crate::transport::rpc_session::from_fn(|req: Bytes| async move { Ok(req) });

        let (server, addr, cert_der) = build_server()?;
        let rpc_server =
            QuinnRpcServer::new(server, processor, fresh_signing_key()).with_connection_limit(1);
        let shutdown = rpc_server.shutdown_token();
        let server_task = tokio::spawn(rpc_server.run());

        // First connection takes the only permit and works.
        let session1 = connect_pinned(addr, &cert_der).await?;
        let client1 = QuinnTransport::new(session1);
        let resp1 = client1.send(b"one".to_vec(), Some(5_000)).await?;
        assert_eq!(&resp1[..], b"one");

        // Second connection: the server rejects it (drops the request before
        // accepting), so a request on it must fail rather than be served.
        // connect_pinned itself may or may not error depending on timing; the
        // load-bearing assertion is that no request succeeds on conn #2.
        let second = async {
            let session2 = connect_pinned(addr, &cert_der).await?;
            let client2 = QuinnTransport::new(session2);
            client2.send(b"two".to_vec(), Some(2_000)).await
        }
        .await;
        assert!(
            second.is_err(),
            "second connection over the cap must not be served, got {second:?}"
        );

        // First connection still works after the rejection.
        let resp1b = client1.send(b"again".to_vec(), Some(5_000)).await?;
        assert_eq!(&resp1b[..], b"again");

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
        crate::transport::pq_provider::install_pq_crypto_provider();

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
