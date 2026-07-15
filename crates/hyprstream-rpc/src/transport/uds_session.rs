//! `UdsSession` — one [`web_transport_trait::Session`] over a Unix-domain socket.
//!
//! Epic #131 / moq M1. Same-host processes keep an `ipc` transport after the ZMQ
//! removal, but re-platformed off ZMQ framing onto the *same* generic plane that
//! quinn and iroh use: a [`web_transport_trait::Session`]. Because both the RPC
//! plane ([`super::rpc_session::SessionRpcTransport`] /
//! [`super::rpc_session::serve_rpc_connection`]) and the streaming plane
//! (`moq_net::{Client, Server}`) are generic over `Session`, **one** `UdsSession`
//! impl carries *both* over a single `UnixStream` — exactly as the QUIC/iroh
//! sessions do.
//!
//! # Why a multiplexer
//!
//! `web_transport_trait::Session` is a *stream-multiplexed* abstraction
//! (`open_bi`/`open_uni`/`accept_bi`/`accept_uni`). QUIC and iroh provide that
//! natively; a `UnixStream` is a single ordered byte-stream. So `UdsSession`
//! carries a [`yamux`] multiplexer — flow-controlled, head-of-line-correct
//! concurrent substreams over the one socket. moq-lite needs exactly this: one
//! bidi SETUP control stream plus one uni stream per Group. It never uses
//! datagrams, so those trait methods are stubbed (`max_datagram_size() == 0`).
//!
//! # Two wire conventions (UDS has neither ALPN nor a uni/bidi distinction)
//!
//! 1. **Plane byte** ([`PLANE_RPC`] / [`PLANE_MOQ`]) — written once on the raw
//!    `UnixStream` *before* yamux wraps it. Replaces ALPN: the listener reads it
//!    to route the connection to the RPC dispatcher vs `moq_net::Server::accept`.
//! 2. **Substream tag** ([`TAG_BIDI`] / [`TAG_UNI`]) — yamux substreams are all
//!    bidirectional, so the opener prefixes each new substream with one tag byte
//!    and the acceptor reads it to route the substream to `accept_bi` vs
//!    `accept_uni`. A uni opener simply never reads its half.
//!
//! # Driver
//!
//! yamux's `Connection` is single-owner and poll-driven (not `Clone`), but
//! `Session: Clone` and is called concurrently. So one **driver task** owns the
//! `Connection` and services `open_*`/inbound-stream work over channels; cloned
//! `UdsSession` handles all talk to it. This mirrors `libp2p-yamux`.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use futures::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::sync::{mpsc, oneshot, Mutex, Notify};
use tokio_util::compat::TokioAsyncReadCompatExt;
use web_transport_trait::Session;

/// Connection-plane selector written as the first byte on the raw socket. The
/// RPC dispatcher and the moq server are distinct planes (as ALPN distinguishes
/// them on QUIC); one `UnixStream` serves exactly one.
pub const PLANE_RPC: u8 = 0x01;
/// See [`PLANE_RPC`]. Routes the connection to `moq_net::Server::accept`.
pub const PLANE_MOQ: u8 = 0x02;

/// Substream-kind tag: the opener intends a bidirectional stream.
const TAG_BIDI: u8 = 0x01;
/// Substream-kind tag: the opener intends a unidirectional stream (writer only).
const TAG_UNI: u8 = 0x02;

/// Bound on queued accepted substreams and pending opens before backpressure.
const ACCEPT_CHANNEL_BOUND: usize = 64;
const CMD_CHANNEL_BOUND: usize = 64;

/// Hard cap on concurrent yamux substreams per connection. Bounds the inbound
/// classification fan-out (one task + one stream slot each) so a hostile local
/// peer cannot open an unbounded number of substreams. yamux enforces this at
/// the protocol layer (excess opens are refused), giving the transport a DoS
/// bound independent of the downstream rpc_session semaphore.
const MAX_CONCURRENT_STREAMS: usize = 256;

/// Per-substream deadline for reading the 1-byte UNI/BIDI tag. Without it, a
/// peer that opens a substream and never sends the tag pins a classification
/// task and a yamux stream slot forever — a pre-dispatch slowloris that evades
/// rpc_session's `REQUEST_READ_TIMEOUT` (which only starts after classification).
const TAG_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// The concrete yamux substream type once wrapped over a compat `UnixStream`.
type YamuxStream = yamux::Stream;

// ============================================================================
// Error
// ============================================================================

/// Error type for [`UdsSession`] and its streams.
#[derive(Debug, Clone)]
pub struct UdsError(String);

impl UdsError {
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
    fn closed() -> Self {
        Self("uds session closed".to_owned())
    }
    fn io(e: std::io::Error) -> Self {
        Self(format!("uds io: {e}"))
    }
    fn conn(e: yamux::ConnectionError) -> Self {
        Self(format!("uds yamux: {e}"))
    }
}

impl std::fmt::Display for UdsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UdsError {}

impl web_transport_trait::Error for UdsError {
    fn session_error(&self) -> Option<(u32, String)> {
        Some((0, self.0.clone()))
    }
}

// ============================================================================
// Streams
// ============================================================================

/// Outbound half of a UDS substream. Wraps the yamux write half; `finish`
/// half-closes (sends FIN) so the peer's read sees EOF.
pub struct UdsSendStream {
    // `None` after finish()/reset() so further writes error.
    inner: Option<WriteHalf<YamuxStream>>,
}

impl UdsSendStream {
    fn new(wh: WriteHalf<YamuxStream>) -> Self {
        Self { inner: Some(wh) }
    }
}

impl web_transport_trait::SendStream for UdsSendStream {
    type Error = UdsError;

    async fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        let wh = self
            .inner
            .as_mut()
            .ok_or_else(|| UdsError::new("write after finish/reset"))?;
        wh.write(buf).await.map_err(UdsError::io)
    }

    fn set_priority(&mut self, _order: u8) {
        // yamux has no per-stream priority; no-op.
    }

    fn finish(&mut self) -> Result<(), Self::Error> {
        // `finish` is sync but yamux close is async; send the FIN on a detached
        // task. Subsequent writes error (inner is now None). The driver pumps the
        // connection so the FIN reaches the peer.
        if let Some(mut wh) = self.inner.take() {
            tokio::spawn(async move {
                if let Err(e) = wh.close().await {
                    tracing::debug!(error = %e, "uds: send-stream FIN flush failed");
                }
            });
        }
        Ok(())
    }

    fn reset(&mut self, _code: u32) {
        // Drop the write half without a graceful FIN; yamux signals the peer.
        self.inner = None;
    }

    async fn closed(&mut self) -> Result<(), Self::Error> {
        // `closed` must DETECT closure, never CAUSE it. After finish()/reset()
        // the FIN/RST was already initiated, so report closed. While the stream
        // is still open we have no portable way to observe a peer STOP_SENDING
        // through yamux's WriteHalf, so we pend — returning early here would make
        // moq's `select! { biased; _ = closed() => cancel }` abort a live group
        // right after its header, dropping every frame.
        if self.inner.is_some() {
            std::future::pending::<()>().await;
        }
        Ok(())
    }
}

/// Inbound half of a UDS substream. Wraps the yamux read half.
pub struct UdsRecvStream {
    inner: ReadHalf<YamuxStream>,
}

impl UdsRecvStream {
    fn new(rh: ReadHalf<YamuxStream>) -> Self {
        Self { inner: rh }
    }
}

impl web_transport_trait::RecvStream for UdsRecvStream {
    type Error = UdsError;

    async fn read(&mut self, dst: &mut [u8]) -> Result<Option<usize>, Self::Error> {
        match self.inner.read(dst).await.map_err(UdsError::io)? {
            0 => Ok(None), // EOF (FIN)
            n => Ok(Some(n)),
        }
    }

    fn stop(&mut self, _code: u32) {
        // Best-effort: the read half closes on drop of this `UdsRecvStream`,
        // which yamux surfaces to the peer. No separate sync close path.
    }

    async fn closed(&mut self) -> Result<(), Self::Error> {
        // Must DETECT closure without CONSUMING data — moq selects on
        // `recv.closed()` concurrently with reads, so draining here would steal
        // frame bytes. We have no portable yamux peek for a peer FIN/RST on the
        // ReadHalf, so we pend; the stream closes for real when this
        // `UdsRecvStream` drops. (A `read()` returning `None` is the actual
        // EOF signal moq relies on.)
        std::future::pending::<()>().await;
        Ok(())
    }
}

// ============================================================================
// Driver
// ============================================================================

/// Command from a [`UdsSession`] handle to its driver task.
enum DriverCmd {
    /// Open a new outbound yamux substream; the tag byte is written by the
    /// `open_*` method after it receives the stream.
    OpenOutbound(oneshot::Sender<Result<YamuxStream, yamux::ConnectionError>>),
    /// Begin a graceful connection close.
    Close,
}

/// Read the 1-byte substream tag and route the substream to the matching accept
/// queue. Runs as its own task so the tag read is driven by the still-pumping
/// connection driver (it never blocks the driver loop).
async fn classify_inbound(
    mut stream: YamuxStream,
    bi_tx: mpsc::Sender<(UdsSendStream, UdsRecvStream)>,
    uni_tx: mpsc::Sender<UdsRecvStream>,
) {
    let mut tag = [0u8; 1];
    // Bound the tag read so a peer that opens a substream and never tags it
    // cannot pin this task + a yamux stream slot indefinitely.
    match tokio::time::timeout(TAG_READ_TIMEOUT, stream.read_exact(&mut tag)).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::debug!(error = %e, "uds: inbound substream closed before tagging");
            return;
        }
        Err(_) => {
            tracing::warn!(
                timeout = ?TAG_READ_TIMEOUT,
                "uds: inbound substream did not send its tag in time, dropping"
            );
            return;
        }
    }
    match tag[0] {
        TAG_BIDI => {
            let (rh, wh) = stream.split();
            let _ = bi_tx.send((UdsSendStream::new(wh), UdsRecvStream::new(rh))).await;
        }
        TAG_UNI => {
            let (rh, _wh) = stream.split();
            // Uni: peer only writes; we keep the read half, drop the write half.
            let _ = uni_tx.send(UdsRecvStream::new(rh)).await;
        }
        unknown => {
            tracing::warn!(tag = unknown, "uds: unknown substream tag, dropping");
        }
    }
}

/// The connection driver: owns the yamux `Connection` and pumps it forever,
/// servicing outbound-open commands and routing inbound substreams, until the
/// connection ends or all handles drop. A single `poll_fn` keeps exclusive
/// `&mut conn` per poll, avoiding the borrow dance of `select!`.
async fn drive_connection<T>(
    socket: T,
    mode: yamux::Mode,
    mut cmd_rx: mpsc::Receiver<DriverCmd>,
    bi_tx: mpsc::Sender<(UdsSendStream, UdsRecvStream)>,
    uni_tx: mpsc::Sender<UdsRecvStream>,
    is_closed: Arc<AtomicBool>,
    closed_notify: Arc<Notify>,
) where
    T: futures::AsyncRead + futures::AsyncWrite + Unpin + Send + 'static,
{
    let mut cfg = yamux::Config::default();
    // Cap concurrent substreams (DoS bound on inbound classification fan-out).
    cfg.set_max_num_streams(MAX_CONCURRENT_STREAMS);
    let mut conn = yamux::Connection::new(socket, cfg, mode);
    let mut pending_opens: VecDeque<oneshot::Sender<Result<YamuxStream, yamux::ConnectionError>>> =
        VecDeque::new();
    let mut closing = false;

    std::future::poll_fn(|cx: &mut Context<'_>| {
        // 1. Ingest outbound-open / close commands (non-blocking drain).
        if !closing {
            loop {
                match cmd_rx.poll_recv(cx) {
                    Poll::Ready(Some(DriverCmd::OpenOutbound(reply))) => {
                        pending_opens.push_back(reply);
                    }
                    // Explicit close, or all `UdsSession` handles dropped
                    // (nothing more will be requested) — begin close either way.
                    Poll::Ready(Some(DriverCmd::Close)) | Poll::Ready(None) => {
                        closing = true;
                        break;
                    }
                    Poll::Pending => break,
                }
            }
        }

        // 2. Service queued outbound opens (skip while closing).
        if !closing {
            while !pending_opens.is_empty() {
                match conn.poll_new_outbound(cx) {
                    Poll::Ready(result) => {
                        let is_err = result.is_err();
                        if let Some(reply) = pending_opens.pop_front() {
                            let _ = reply.send(result);
                        }
                        if is_err {
                            closing = true;
                            break;
                        }
                    }
                    Poll::Pending => break,
                }
            }
        }

        // 3. Closing path: drive the close handshake to completion.
        if closing {
            return match conn.poll_close(cx) {
                Poll::Ready(_) => Poll::Ready(()),
                Poll::Pending => Poll::Pending,
            };
        }

        // 4. Drive inbound: each accepted substream is classified on its own task.
        loop {
            match conn.poll_next_inbound(cx) {
                Poll::Ready(Some(Ok(stream))) => {
                    tokio::spawn(classify_inbound(stream, bi_tx.clone(), uni_tx.clone()));
                }
                Poll::Ready(Some(Err(_))) | Poll::Ready(None) => return Poll::Ready(()),
                Poll::Pending => return Poll::Pending,
            }
        }
    })
    .await;

    // Connection ended: wake `closed()` waiters; dropping bi_tx/uni_tx closes the
    // accept channels so pending `accept_*` calls observe closure.
    is_closed.store(true, Ordering::Release);
    closed_notify.notify_waiters();
}

// ============================================================================
// Session
// ============================================================================

struct Inner {
    cmd_tx: mpsc::Sender<DriverCmd>,
    accept_bi_rx: Mutex<mpsc::Receiver<(UdsSendStream, UdsRecvStream)>>,
    accept_uni_rx: Mutex<mpsc::Receiver<UdsRecvStream>>,
    is_closed: Arc<AtomicBool>,
    closed_notify: Arc<Notify>,
}

/// A [`web_transport_trait::Session`] over a Unix-domain socket, multiplexed by
/// yamux. Clone-cheap (shares one driver via `Arc`). Carries both the RPC and
/// moq planes — see the module docs.
#[derive(Clone)]
pub struct UdsSession {
    inner: Arc<Inner>,
}

impl UdsSession {
    /// Wrap an established, plane-selected socket and spawn its driver task.
    /// `socket` must already have had the [`PLANE_RPC`]/[`PLANE_MOQ`] byte
    /// consumed (see [`connect_uds`] / [`accept_uds`]).
    fn spawn<T>(socket: T, mode: yamux::Mode) -> Self
    where
        T: futures::AsyncRead + futures::AsyncWrite + Unpin + Send + 'static,
    {
        let (cmd_tx, cmd_rx) = mpsc::channel(CMD_CHANNEL_BOUND);
        let (bi_tx, bi_rx) = mpsc::channel(ACCEPT_CHANNEL_BOUND);
        let (uni_tx, uni_rx) = mpsc::channel(ACCEPT_CHANNEL_BOUND);
        let is_closed = Arc::new(AtomicBool::new(false));
        let closed_notify = Arc::new(Notify::new());

        tokio::spawn(drive_connection(
            socket,
            mode,
            cmd_rx,
            bi_tx,
            uni_tx,
            Arc::clone(&is_closed),
            Arc::clone(&closed_notify),
        ));

        Self {
            inner: Arc::new(Inner {
                cmd_tx,
                accept_bi_rx: Mutex::new(bi_rx),
                accept_uni_rx: Mutex::new(uni_rx),
                is_closed,
                closed_notify,
            }),
        }
    }

    /// Request a fresh outbound yamux substream from the driver.
    async fn open_outbound(&self) -> Result<YamuxStream, UdsError> {
        let (tx, rx) = oneshot::channel();
        self.inner
            .cmd_tx
            .send(DriverCmd::OpenOutbound(tx))
            .await
            .map_err(|_| UdsError::closed())?;
        rx.await.map_err(|_| UdsError::closed())?.map_err(UdsError::conn)
    }
}

impl Session for UdsSession {
    type SendStream = UdsSendStream;
    type RecvStream = UdsRecvStream;
    type Error = UdsError;

    async fn accept_uni(&self) -> Result<Self::RecvStream, Self::Error> {
        let mut rx = self.inner.accept_uni_rx.lock().await;
        rx.recv().await.ok_or_else(UdsError::closed)
    }

    async fn accept_bi(&self) -> Result<(Self::SendStream, Self::RecvStream), Self::Error> {
        let mut rx = self.inner.accept_bi_rx.lock().await;
        rx.recv().await.ok_or_else(UdsError::closed)
    }

    async fn open_bi(&self) -> Result<(Self::SendStream, Self::RecvStream), Self::Error> {
        let mut stream = self.open_outbound().await?;
        // Tag + flush so the acceptor can classify before any plane data.
        stream.write_all(&[TAG_BIDI]).await.map_err(UdsError::io)?;
        stream.flush().await.map_err(UdsError::io)?;
        let (rh, wh) = stream.split();
        Ok((UdsSendStream::new(wh), UdsRecvStream::new(rh)))
    }

    async fn open_uni(&self) -> Result<Self::SendStream, Self::Error> {
        let mut stream = self.open_outbound().await?;
        stream.write_all(&[TAG_UNI]).await.map_err(UdsError::io)?;
        stream.flush().await.map_err(UdsError::io)?;
        let (_rh, wh) = stream.split();
        Ok(UdsSendStream::new(wh))
    }

    fn send_datagram(&self, _payload: Bytes) -> Result<(), Self::Error> {
        Err(UdsError::new("UDS transport does not support datagrams"))
    }

    async fn recv_datagram(&self) -> Result<Bytes, Self::Error> {
        // UDS has no datagram channel; moq-lite never calls this. Never resolves.
        std::future::pending().await
    }

    fn max_datagram_size(&self) -> usize {
        0
    }

    fn protocol(&self) -> Option<&str> {
        None
    }

    fn close(&self, _code: u32, _reason: &str) {
        // Best-effort: signal the driver. If the channel is full/closed the
        // connection is already going down.
        let _ = self.inner.cmd_tx.try_send(DriverCmd::Close);
    }

    async fn closed(&self) -> Self::Error {
        loop {
            if self.inner.is_closed.load(Ordering::Acquire) {
                return UdsError::closed();
            }
            let notified = self.inner.closed_notify.notified();
            // Re-check after arming to avoid missing a wake between the load and
            // the await.
            if self.inner.is_closed.load(Ordering::Acquire) {
                return UdsError::closed();
            }
            notified.await;
        }
    }
}

// ============================================================================
// Connect / accept helpers
// ============================================================================

/// Connect to a UDS endpoint, select `plane`, and return a multiplexed session.
///
/// Writes the [`PLANE_RPC`]/[`PLANE_MOQ`] byte on the raw socket, then wraps it
/// in a client-mode yamux session.
pub async fn connect_uds(path: impl AsRef<Path>, plane: u8) -> Result<UdsSession> {
    let mut stream = tokio::net::UnixStream::connect(path.as_ref()).await?;
    tokio::io::AsyncWriteExt::write_all(&mut stream, &[plane]).await?;
    Ok(UdsSession::spawn(stream.compat(), yamux::Mode::Client))
}

/// Accept side: read the plane byte from a freshly-accepted `UnixStream`, then
/// wrap it in a server-mode yamux session. The caller routes by `plane`
/// ([`PLANE_RPC`] → RPC dispatcher, [`PLANE_MOQ`] → `moq_net::Server::accept`).
pub async fn accept_uds(mut stream: tokio::net::UnixStream) -> Result<(u8, UdsSession)> {
    let mut plane = [0u8; 1];
    tokio::io::AsyncReadExt::read_exact(&mut stream, &mut plane).await?;
    // Fail closed on an unrecognized plane selector rather than handing an
    // ambiguous connection to a caller that would have to guess how to route it.
    if !matches!(plane[0], PLANE_RPC | PLANE_MOQ) {
        anyhow::bail!("uds: unknown plane selector byte 0x{:02x}", plane[0]);
    }
    Ok((plane[0], UdsSession::spawn(stream.compat(), yamux::Mode::Server)))
}

// Compile-time assertion that our types satisfy the native `MaybeSend` bound
// (= `Send`) the moq/rpc planes require.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<UdsSession>();
    assert_send::<UdsSendStream>();
    assert_send::<UdsRecvStream>();
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use web_transport_trait::{RecvStream as _, SendStream as _};

    /// Spawn a connected client/server `UdsSession` pair over an in-process
    /// socketpair (no filesystem path needed), with the given plane byte already
    /// consumed.
    async fn session_pair() -> (UdsSession, UdsSession) {
        let (a, b) = tokio::net::UnixStream::pair().unwrap();
        let client = UdsSession::spawn(a.compat(), yamux::Mode::Client);
        let server = UdsSession::spawn(b.compat(), yamux::Mode::Server);
        (client, server)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bidi_round_trip() {
        let (client, server) = session_pair().await;

        let srv = tokio::spawn(async move {
            let (mut send, mut recv) = server.accept_bi().await.unwrap();
            let mut buf = vec![0u8; 5];
            let n = recv.read(&mut buf).await.unwrap().unwrap();
            buf.truncate(n);
            send.write_all(&buf).await.unwrap(); // echo
            send.finish().unwrap();
            // keep the session alive until the client has read the echo
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        });

        let (mut send, mut recv) = client.open_bi().await.unwrap();
        send.write_all(b"hello").await.unwrap();
        send.finish().unwrap();
        let mut got = Vec::new();
        let mut scratch = [0u8; 64];
        while let Some(n) = recv.read(&mut scratch).await.unwrap() {
            got.extend_from_slice(&scratch[..n]);
        }
        assert_eq!(&got, b"hello");
        srv.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn many_concurrent_uni_streams() {
        let (client, server) = session_pair().await;
        const N: usize = 8;

        let srv = tokio::spawn(async move {
            let mut totals = Vec::new();
            for _ in 0..N {
                let mut recv = server.accept_uni().await.unwrap();
                let data = recv.read_all().await.unwrap();
                totals.push(data.to_vec());
            }
            totals
        });

        // Open N uni streams concurrently, each carrying its index byte.
        let mut handles = Vec::new();
        for i in 0..N {
            let c = client.clone();
            handles.push(tokio::spawn(async move {
                let mut send = c.open_uni().await.unwrap();
                send.write_all(&[i as u8; 4]).await.unwrap();
                send.finish().unwrap();
                // Hold the stream briefly so FIN flushes.
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let mut got = srv.await.unwrap();
        got.sort();
        let mut want: Vec<Vec<u8>> = (0..N).map(|i| vec![i as u8; 4]).collect();
        want.sort();
        assert_eq!(got, want);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn uni_stream_server_to_client() {
        // moq publishes group frames as server->client uni streams, so verify
        // that direction explicitly (the other uni test is client->server).
        let (client, server) = session_pair().await;

        let cli = tokio::spawn(async move {
            let mut recv = client.accept_uni().await.unwrap();
            recv.read_all().await.unwrap()
        });

        let mut send = server.open_uni().await.unwrap();
        send.write_all(b"srv->cli").await.unwrap();
        send.finish().unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        let got = cli.await.unwrap();
        assert_eq!(&got[..], b"srv->cli");
        // keep server alive until client read
        drop(server);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn datagrams_unsupported() {
        let (client, _server) = session_pair().await;
        assert_eq!(client.max_datagram_size(), 0);
        assert!(client.send_datagram(Bytes::from_static(b"x")).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn closed_fires_when_peer_drops() {
        let (client, server) = session_pair().await;
        drop(server);
        // The driver should observe the broken connection and fire `closed`.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), client.closed())
            .await
            .expect("closed() must resolve after peer drop");
    }

    // ── Integration: the real RPC plane rides UdsSession unchanged ──────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_plane_round_trip_over_uds() {
        use crate::transport::rpc_session::{
            from_fn, serve_rpc_connection, SessionRpcTransport, DEFAULT_STREAM_LIMIT,
            REQUEST_READ_TIMEOUT,
        };
        use crate::transport_traits::Transport;
        use std::sync::Arc;
        use tokio::sync::Semaphore;
        use tokio_util::sync::CancellationToken;

        let (client_session, server_session) = session_pair().await;

        let (sk, _vk) = crate::generate_signing_keypair();
        let processor = Arc::new(from_fn(|req: Bytes| async move { Ok(req) })); // echo
        let limit = Arc::new(Semaphore::new(DEFAULT_STREAM_LIMIT));
        let shutdown = CancellationToken::new();

        let srv = tokio::spawn(serve_rpc_connection(
            server_session,
            processor,
            sk,
            limit,
            REQUEST_READ_TIMEOUT,
            shutdown.clone(),
            crate::transport::carrier::CarrierContext::explicit_trusted_local(),
        ));

        // Two sequential calls prove the session multiplexes fresh bidi streams.
        let client = SessionRpcTransport::new(client_session);
        let r1 = client.send(b"ping".to_vec(), Some(4_000)).await.unwrap();
        assert_eq!(r1, b"ping");
        let r2 = client.send(b"pong".to_vec(), Some(4_000)).await.unwrap();
        assert_eq!(r2, b"pong");

        shutdown.cancel();
        let _ = srv.await;
    }

    // ── Integration: the real moq streaming plane rides UdsSession unchanged ──
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moq_plane_publish_subscribe_over_uds() {
        use moq_net::{Client, Group, Origin, Server, Track};
        use std::time::Duration;

        let (client_session, server_session) = session_pair().await;

        // Publisher origin: publish a fully-formed broadcast/track/group (one
        // frame, group closed) before the subscriber connects, so the served
        // group is complete — matches the known-good iroh_moq round-trip.
        let producer = Origin::random().produce();
        let consumer = producer.consume();
        let mut broadcast = producer.create_broadcast("alice/run-1").unwrap();
        let mut track = broadcast.create_track(Track::new("tokens")).unwrap();
        let mut group = track.create_group(Group::from(0u64)).unwrap();
        group.write_frame(Bytes::from_static(b"hello-uds-moq")).unwrap();
        drop(group);

        // The moq handshake is bidirectional (client opens the SETUP stream, the
        // server accepts it), so server.accept and client.connect must run
        // concurrently — over iroh/quinn the server side lives in a spawned
        // accept loop; here we spawn it explicitly.
        let server = Server::new().with_publish(consumer);
        let server_task = tokio::spawn(async move { server.accept(server_session).await });

        // Client: run moq Client over the UDS session, subscribe, read the frame.
        let client_origin = Origin::random().produce();
        let client_consumer = client_origin.consume();
        let moq_client = Client::new().with_consume(client_origin);
        let _moq_session = tokio::time::timeout(Duration::from_secs(8), moq_client.connect(client_session))
            .await
            .expect("client.connect hung")
            .unwrap();
        let _moq_server = tokio::time::timeout(Duration::from_secs(8), server_task)
            .await
            .expect("server.accept hung")
            .unwrap()
            .unwrap();

        let bc = tokio::time::timeout(
            Duration::from_secs(5),
            client_consumer.announced_broadcast("alice/run-1"),
        )
        .await
        .unwrap()
        .unwrap();
        let mut tc = bc.subscribe_track(&Track::new("tokens")).unwrap();

        let mut gc = tokio::time::timeout(Duration::from_secs(5), tc.next_group())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let frame = tokio::time::timeout(Duration::from_secs(5), gc.read_frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(&frame[..], b"hello-uds-moq");

        // Keep broadcast/track alive until the read completes.
        drop(track);
        drop(broadcast);
    }

    // ── connect_uds / accept_uds plane handshake over a real listener ────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plane_handshake_over_listener() {
        let dir = std::env::temp_dir().join(format!("uds-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("plane.sock");
        let _ = std::fs::remove_file(&path);
        let listener = tokio::net::UnixListener::bind(&path).unwrap();

        let srv = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (plane, session) = accept_uds(stream).await.unwrap();
            assert_eq!(plane, PLANE_RPC);
            // echo one bidi stream
            let (mut send, mut recv) = session.accept_bi().await.unwrap();
            let data = recv.read_all().await.unwrap();
            send.write_all(&data).await.unwrap();
            send.finish().unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        });

        let client = connect_uds(&path, PLANE_RPC).await.unwrap();
        let (mut send, mut recv) = client.open_bi().await.unwrap();
        send.write_all(b"plane-ok").await.unwrap();
        send.finish().unwrap();
        let got = recv.read_all().await.unwrap();
        assert_eq!(&got[..], b"plane-ok");

        srv.await.unwrap();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn accept_uds_rejects_unknown_plane() {
        let (a, b) = tokio::net::UnixStream::pair().unwrap();
        // Client writes a bogus plane selector.
        let cli = tokio::spawn(async move {
            let mut a = a;
            tokio::io::AsyncWriteExt::write_all(&mut a, &[0xFF]).await.unwrap();
            // Hold the socket open so the server's read sees the byte, not EOF.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        });
        let res = accept_uds(b).await;
        assert!(res.is_err(), "unknown plane selector must be rejected");
        cli.await.unwrap();
    }
}
