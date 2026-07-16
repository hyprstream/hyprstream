//! In-process RPC transport for co-located services.
//!
//! Dials a service running in the *same process* without a network socket.
//! This is the in-memory dial target the [`crate::dial`] factory selects for
//! `inproc://` endpoints — the replacement for ZMQ's `inproc://` transport.
//!
//! # No second security model
//!
//! [`InMemoryTransport::send`] forwards the canonical request bytes to an
//! [`IrohRequestProcessor`] (in production a
//! [`LocalServiceBridge`](crate::transport::iroh_rpc::LocalServiceBridge),
//! which isolates `!Send` `tch`-holding services on a dedicated `LocalSet`
//! thread and exposes only a `Send` `Bytes`-in/`Bytes`-out surface) and returns
//! the signed response bytes. The envelope is signed and verified by the exact
//! same `process_request` path a networked transport uses — co-located calls
//! skip *serialization to a socket*, never authentication or Casbin authz. This
//! is behaviourally identical to ZMQ's `inproc://` (serialize once, no socket),
//! so there is no distinct trust path to review.
//!
//! Per the A1 addressing spike, every in-process call goes through the bridge
//! (one channel hop — the same cost as inproc-ZMQ today). A zero-hop tier that
//! calls `process_request` directly is inexpressible while `RequestService` is
//! unconditionally `?Send`; it is a deliberate later follow-up, not part of this
//! change.
//!
//! # RPC plane only
//!
//! Like [`SessionRpcTransport`](crate::transport::rpc_session::SessionRpcTransport),
//! this transport is request-response only; `subscribe`/`publish` bail. The
//! streaming plane (moq-net) carries SUB/PUB, and in-process streaming is
//! deferred with the streaming substrate (#134).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use bytes::Bytes;

use crate::transport::rpc_session::{IrohRequestProcessor, RpcPendingStream, RpcPublishStub};
use crate::transport_traits::Transport;

/// Default ceiling on a single in-process request, mirroring the networked
/// RPC transport. A wedged co-located handler should surface as a timeout
/// rather than hanging the caller indefinitely.
///
/// Note this deadline also covers *enqueue* latency: `LocalServiceBridge`
/// forwards over a bounded mpsc, so a saturated bridge can make `send()` block
/// on the channel and eventually trip this timeout. The error therefore
/// conflates "handler wedged" with "bridge backpressure" — acceptable for the
/// transport surface, but not a substitute for a real backpressure signal.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// In-process RPC transport over an [`IrohRequestProcessor`].
///
/// Cheaply cloneable: holds an `Arc` to the shared processor.
#[derive(Clone)]
pub struct InMemoryTransport {
    processor: Arc<dyn IrohRequestProcessor>,
}

impl InMemoryTransport {
    /// Dial a co-located service by its already-resolved request processor.
    pub fn new(processor: Arc<dyn IrohRequestProcessor>) -> Self {
        Self { processor }
    }
}

#[async_trait]
impl Transport for InMemoryTransport {
    type Sub = RpcPendingStream;
    type Pub = RpcPublishStub;

    /// Same-process, zero-copy carrier: never leaves the address space, so
    /// cleartext envelopes are permitted (explicit opt-out of the fail-closed
    /// default).
    fn forbids_cleartext_envelope(&self) -> bool {
        false
    }

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        let timeout = timeout_ms
            .map(|ms| Duration::from_millis(ms.max(0) as u64))
            .unwrap_or(DEFAULT_TIMEOUT);
        let processor = Arc::clone(&self.processor);
        let fut = async move { processor.process(Bytes::from(payload)).await };
        let resp = tokio::time::timeout(timeout, fut)
            .await
            .map_err(|_| anyhow!("in-process RPC timeout after {timeout:?}"))??;
        Ok(resp.to_vec())
    }

    async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
        bail!("in-process RPC transport does not support SUB — streaming moves to moq-net (#134)")
    }

    async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
        bail!("in-process RPC transport does not support PUB — streaming moves to moq-net (#134)")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::transport::rpc_session::from_fn;

    fn echo_processor() -> Arc<dyn IrohRequestProcessor> {
        Arc::new(from_fn(|req: Bytes| async move { Ok(req) }))
    }

    #[tokio::test]
    async fn send_roundtrips_through_processor() {
        let t = InMemoryTransport::new(echo_processor());
        let resp = t.send(b"ping".to_vec(), None).await.unwrap();
        assert_eq!(resp, b"ping");
    }

    #[tokio::test]
    async fn subscribe_and_publish_bail() {
        let t = InMemoryTransport::new(echo_processor());
        assert!(t.subscribe(b"topic").await.is_err());
        assert!(t.publish(b"topic").await.is_err());
    }

    #[tokio::test]
    async fn slow_processor_times_out() {
        let slow = Arc::new(from_fn(|req: Bytes| async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok(req)
        }));
        let t = InMemoryTransport::new(slow);
        let err = t.send(b"x".to_vec(), Some(20)).await;
        assert!(err.is_err(), "a handler exceeding the deadline must time out");
    }

    #[tokio::test]
    async fn processor_error_propagates() {
        let failing = Arc::new(from_fn(|_req: Bytes| async move {
            Err(anyhow!("handler boom"))
        }));
        let t = InMemoryTransport::new(failing);
        let err = t.send(b"x".to_vec(), None).await;
        assert!(err.is_err());
    }
}
