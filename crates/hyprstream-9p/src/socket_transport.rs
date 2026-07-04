//! Socket-backed [`P9Transport`] — a native 9P transport over a byte stream.
//!
//! This is the **client-side** counterpart to the server-side [`Translator`]
//! accept loop: where the translator *reads* length-prefixed 9P frames off a
//! `tokio::net` stream and *dispatches* them to a [`Backend`], `SocketTransport`
//! *carries* complete 9P frames between a [`P9Client`] and a remote 9P2000.L
//! server over the same wire (a Unix domain socket or TCP). It is the inverse of
//! the DMA transport ([`crate::dma`]): identical [`P9Transport`] contract, but a
//! real OS socket instead of a SharedArrayBuffer ring, so it carries across
//! process (and host) boundaries.
//!
//! ## Framing contract
//!
//! [`P9Client`] hands `send` a complete 9P message whose first four bytes are the
//! self-counting `size[4]` field, and expects `recv` to return exactly one such
//! complete frame. `recv` therefore reads the 4-byte length prefix first, then
//! the `size - 4` remaining body bytes, handling partial reads via
//! [`read_exact`](tokio::io::AsyncReadExt::read_exact) — matching the framing the
//! [`Translator`] itself uses on the server side.
//!
//! ## Concurrency
//!
//! The read and write halves each sit behind their own async `Mutex`, so the
//! transport is `Send + Sync`. Whole request/response pairs are serialized one
//! level up (in [`Remote9pMount`](crate::remote_mount::Remote9pMount), which
//! holds the [`P9Client`] behind a single mutex), which is what keeps
//! interleaved T-messages from racing on one connection.
//!
//! [`Translator`]: crate::translator::Translator
//! [`Backend`]: crate::backend::Backend
//! [`P9Client`]: crate::client::P9Client

use std::path::Path;

use anyhow::{Context, Result};
use tokio::io::{split, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::{TcpStream, UnixStream};
use tokio::sync::Mutex;

use crate::client::P9Transport;

/// A [`P9Transport`] backed by an `AsyncRead + AsyncWrite` byte stream.
///
/// Construct it directly from any stream via [`SocketTransport::new`], or use the
/// [`connect_uds`](SocketTransport::connect_uds) /
/// [`connect_tcp`](SocketTransport::connect_tcp) helpers for the common Unix and
/// TCP cases.
pub struct SocketTransport<S> {
    rx: Mutex<ReadHalf<S>>,
    tx: Mutex<WriteHalf<S>>,
}

impl<S> SocketTransport<S>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    /// Wrap an already-connected byte stream as a 9P transport.
    pub fn new(stream: S) -> Self {
        let (rx, tx) = split(stream);
        Self {
            rx: Mutex::new(rx),
            tx: Mutex::new(tx),
        }
    }
}

impl SocketTransport<UnixStream> {
    /// Dial a 9P server listening on a Unix domain socket at `path`.
    pub async fn connect_uds(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let stream = UnixStream::connect(path)
            .await
            .with_context(|| format!("connect 9P UDS at {path:?}"))?;
        Ok(Self::new(stream))
    }
}

impl SocketTransport<TcpStream> {
    /// Dial a 9P server listening on a TCP `addr`.
    pub async fn connect_tcp(addr: impl tokio::net::ToSocketAddrs) -> Result<Self> {
        let stream = TcpStream::connect(addr)
            .await
            .context("connect 9P TCP endpoint")?;
        // 9P is request/response with small messages; disable Nagle to avoid
        // coalescing latency. Best-effort — failure here is non-fatal.
        let _ = stream.set_nodelay(true);
        Ok(Self::new(stream))
    }
}

#[async_trait::async_trait]
impl<S> P9Transport for SocketTransport<S>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    async fn send(&self, data: &[u8]) -> Result<()> {
        let mut tx = self.tx.lock().await;
        tx.write_all(data).await.context("9P socket send")?;
        tx.flush().await.context("9P socket flush")?;
        Ok(())
    }

    async fn recv(&self) -> Result<Vec<u8>> {
        let mut rx = self.rx.lock().await;

        // Read the 4-byte self-counting size prefix first...
        let mut len_buf = [0u8; 4];
        rx.read_exact(&mut len_buf)
            .await
            .context("9P socket recv: size prefix")?;
        let total = u32::from_le_bytes(len_buf) as usize;

        // A valid 9P2000.L message is at least size[4] + type[1] + tag[2].
        if total < 7 {
            anyhow::bail!("invalid 9P frame size: {total}");
        }

        // ...then the remaining `total - 4` body bytes. `read_exact` loops over
        // partial reads until the full frame has arrived.
        let mut buf = vec![0u8; total];
        buf[..4].copy_from_slice(&len_buf);
        rx.read_exact(&mut buf[4..])
            .await
            .context("9P socket recv: message body")?;
        Ok(buf)
    }
}
