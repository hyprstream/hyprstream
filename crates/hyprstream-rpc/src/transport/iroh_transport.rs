//! Iroh client-side transport — Phase 2 part 2 of Epic #131 (#133).
//!
//! Implements [`Transport`] over an iroh [`Connection`] so existing
//! [`RpcClientImpl<S, T>`] consumers can ride the `hyprstream-rpc/1` ALPN
//! with zero changes to envelope signing, JWT handling, or response parsing.
//!
//! Each `send()` opens a fresh bidi stream on the connection, writes the
//! length-prefixed `SignedEnvelope` bytes, and reads the response — matching
//! the [`IrohRpcProtocolHandler`] (`transport::iroh_rpc`) accept loop on the
//! server side.
//!
//! **Out of scope** (Phase 3): the [`Transport::subscribe`] and
//! [`Transport::publish`] methods are streaming-plane concerns. They return
//! errors here; subscribers go through `moq-net` on the `moql` ALPN instead.
//!
//! [`IrohRpcProtocolHandler`]: super::iroh_rpc::IrohRpcProtocolHandler

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use iroh::endpoint::Connection;

use crate::transport::iroh_rpc::MAX_FRAME_BYTES;
use crate::transport_traits::{PublishSink, Transport};

/// Default per-call timeout when the caller passes `None`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Wraps an iroh [`Connection`] as a [`Transport`] for RPC plane traffic.
///
/// Clone-cheap — the connection itself is reference-counted internally by
/// iroh, and we share it via `Arc` so the same `IrohTransport` can be
/// installed in multiple `RpcClientImpl`s if a service holds parallel
/// per-tenant clients.
#[derive(Clone)]
pub struct IrohTransport {
    conn: Arc<Connection>,
}

impl IrohTransport {
    /// Build from an already-established iroh connection on
    /// [`super::iroh_substrate::ALPN_HYPRSTREAM_RPC`].
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Arc::new(conn),
        }
    }

    /// Borrow the underlying iroh connection (e.g. for inspection).
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
}

/// Stub stream type for [`Transport::Sub`]. Always pending — the iroh RPC
/// plane does not carry SUB-style topic streams; those live on the `moql`
/// ALPN under Phase 3.
pub type IrohPendingStream = futures::stream::Pending<Result<Vec<Vec<u8>>>>;

/// Stub publish sink for [`Transport::Pub`]. Errors on any `send_frames`
/// call; iroh-rpc is request-response only.
pub struct IrohPublishStub;

#[async_trait]
impl PublishSink for IrohPublishStub {
    async fn send_frames(&self, _frames: &[&[u8]]) -> Result<()> {
        bail!("iroh RPC plane does not support PUB/PUSH — use moq-net on the `moql` ALPN")
    }
}

#[async_trait]
impl Transport for IrohTransport {
    type Sub = IrohPendingStream;
    type Pub = IrohPublishStub;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        let timeout = timeout_ms
            .map(|ms| Duration::from_millis(ms.max(0) as u64))
            .unwrap_or(DEFAULT_TIMEOUT);
        let fut = async {
            let (mut send, mut recv) = self
                .conn
                .open_bi()
                .await
                .map_err(|e| anyhow!("iroh open_bi: {e}"))?;
            send.write_all(&payload)
                .await
                .map_err(|e| anyhow!("iroh write_all: {e}"))?;
            send.finish().map_err(|e| anyhow!("iroh finish: {e}"))?;
            let buf = recv
                .read_to_end(MAX_FRAME_BYTES)
                .await
                .map_err(|e| anyhow!("iroh read_to_end: {e}"))?;
            Ok(buf)
        };
        tokio::time::timeout(timeout, fut)
            .await
            .map_err(|_| anyhow!("iroh RPC timeout after {timeout:?}"))?
    }

    async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
        bail!("iroh RPC plane does not support SUB — use moq-net on the `moql` ALPN")
    }

    async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
        bail!("iroh RPC plane does not support PUB — use moq-net on the `moql` ALPN")
    }
}
