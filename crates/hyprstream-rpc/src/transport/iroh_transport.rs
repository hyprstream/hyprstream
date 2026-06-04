//! Iroh client-side transport — Phase 2 part 2 of Epic #131 (#133).
//!
//! Thin wrapper over the transport-generic [`SessionRpcTransport`] (see
//! [`super::rpc_session`]) specialised to iroh's [`web_transport_iroh::Session`]
//! so existing [`RpcClientImpl<S, T>`] consumers can ride the
//! `hyprstream-rpc/1` ALPN with zero changes to envelope signing, JWT handling,
//! or response parsing.
//!
//! Each `send()` opens a fresh bidi stream on the connection, writes the
//! `SignedEnvelope` bytes, and reads the response (bounded) — matching the
//! [`IrohRpcProtocolHandler`] (`transport::iroh_rpc`) accept loop on the
//! server side.
//!
//! **Out of scope** (Phase 3): the [`Transport::subscribe`] and
//! [`Transport::publish`] methods are streaming-plane concerns. They return
//! errors here; subscribers go through `moq-net` on the `moql` ALPN instead.
//!
//! [`IrohRpcProtocolHandler`]: super::iroh_rpc::IrohRpcProtocolHandler

use anyhow::Result;
use async_trait::async_trait;
use iroh::endpoint::Connection;

use crate::transport::rpc_session::{
    RpcPendingStream, RpcPublishStub, SessionRpcTransport,
};
use crate::transport_traits::Transport;

/// Wraps an iroh [`Connection`] as a [`Transport`] for RPC plane traffic.
///
/// Clone-cheap — delegates to the transport-generic [`SessionRpcTransport`]
/// over iroh's `web-transport` session, which is itself reference-counted.
#[derive(Clone)]
pub struct IrohTransport {
    // NOTE: removed unused connection() accessor in M1.
    inner: SessionRpcTransport<web_transport_iroh::Session>,
}

impl IrohTransport {
    /// Build from an already-established iroh connection on
    /// [`super::iroh_substrate::ALPN_HYPRSTREAM_RPC`].
    pub fn new(conn: Connection) -> Self {
        Self {
            inner: SessionRpcTransport::new(web_transport_iroh::Session::raw(conn)),
        }
    }
}

/// Re-export the generic pending-stream stub under the historical name.
pub type IrohPendingStream = RpcPendingStream;

/// Re-export the generic publish stub under the historical name.
pub type IrohPublishStub = RpcPublishStub;

#[async_trait]
impl Transport for IrohTransport {
    type Sub = IrohPendingStream;
    type Pub = IrohPublishStub;

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
