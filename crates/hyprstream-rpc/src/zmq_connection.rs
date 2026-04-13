//! ZMQ transport implementation.
//!
//! `ZmqConnection` implements `Transport` for native ZMQ sockets.
//! Extracted from `ZmqClient` — same resilience features (REQ_RELAXED,
//! REQ_CORRELATE, auto-reconnect with exponential backoff).

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::Stream;
use tmq::{AsZmqSocket, FromZmqSocket, Multipart, RequestSender, SocketExt};
use tracing::{debug, trace, warn};

use crate::transport_traits::{PublishSink, Transport};

/// ZMQ REQ/REP transport with auto-reconnect and resilience.
///
/// Handles socket creation, connection management, and send/recv.
/// Signing and envelope logic live in `RpcClient<S, T>`, not here.
pub struct ZmqConnection {
    endpoint: String,
    context: Arc<zmq::Context>,
    sender: tokio::sync::Mutex<Option<RequestSender>>,
}

// ZmqConnection is Send+Sync: endpoint is String, context is Arc, sender is tokio::Mutex
// (all Send+Sync). The zmq::Socket inside RequestSender is only accessed under the Mutex.
unsafe impl Send for ZmqConnection {}
unsafe impl Sync for ZmqConnection {}

impl ZmqConnection {
    /// Create a new ZMQ connection to the given endpoint.
    ///
    /// Socket is lazily initialized on first `send()`.
    pub fn new(endpoint: &str, context: Arc<zmq::Context>) -> Self {
        Self {
            endpoint: endpoint.to_owned(),
            context,
            sender: tokio::sync::Mutex::new(None),
        }
    }

    /// Get the endpoint this connection targets.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Get the ZMQ context.
    pub fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    /// Create and configure a new REQ socket with resilience options.
    fn create_socket(&self) -> Result<zmq::Socket> {
        let socket = self.context.socket(zmq::REQ)
            .map_err(|e| anyhow!("Failed to create REQ socket: {}", e))?;

        // REQ_RELAXED: allows sending new request after timeout
        socket.set_req_relaxed(true)
            .map_err(|e| anyhow!("Failed to set REQ_RELAXED: {}", e))?;

        // REQ_CORRELATE: match replies to requests (required with REQ_RELAXED)
        socket.set_req_correlate(true)
            .map_err(|e| anyhow!("Failed to set REQ_CORRELATE: {}", e))?;

        // Auto-reconnect with exponential backoff (100ms to 5s)
        socket.set_reconnect_ivl(100)
            .map_err(|e| anyhow!("Failed to set reconnect interval: {}", e))?;
        socket.set_reconnect_ivl_max(5000)
            .map_err(|e| anyhow!("Failed to set max reconnect interval: {}", e))?;

        // Don't block on close
        socket.set_linger(0)
            .map_err(|e| anyhow!("Failed to set linger: {}", e))?;

        socket.connect(&self.endpoint)
            .map_err(|e| anyhow!("Failed to connect to {}: {}", self.endpoint, e))?;

        debug!("ZmqConnection connected to {} with REQ_RELAXED+REQ_CORRELATE", self.endpoint);
        Ok(socket)
    }

    async fn get_or_create_sender(&self) -> Result<RequestSender> {
        let mut guard = self.sender.lock().await;
        if let Some(sender) = guard.take() {
            return Ok(sender);
        }
        let socket = self.create_socket()?;
        let sender = RequestSender::from_zmq_socket(socket)
            .map_err(|e| anyhow!("Failed to wrap socket in TMQ: {}", e))?;
        Ok(sender)
    }

    async fn store_sender(&self, sender: RequestSender) {
        let mut guard = self.sender.lock().await;
        *guard = Some(sender);
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Transport for ZmqConnection {
    type Sub = ZmqSubscriber;
    type Pub = ZmqPublisher;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        let timeout = timeout_ms.unwrap_or(30_000);

        trace!(
            "ZmqConnection sending {} bytes to {} (timeout: {}ms)",
            payload.len(), self.endpoint, timeout
        );

        let sender = self.get_or_create_sender().await?;
        let receiver = sender
            .send(Multipart::from(vec![payload]))
            .await
            .map_err(|e| anyhow!("Failed to send request to {}: {}", self.endpoint, e))?;

        let timeout_ms = u64::try_from(timeout.max(0)).unwrap_or(0);
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            receiver.recv(),
        ).await;

        match result {
            Ok(Ok((response, sender))) => {
                self.store_sender(sender).await;
                let wire_bytes: Vec<u8> = response
                    .into_iter()
                    .flat_map(|frame| frame.to_vec())
                    .collect();
                trace!("ZmqConnection received {} bytes from {}", wire_bytes.len(), self.endpoint);
                Ok(wire_bytes)
            }
            Ok(Err(e)) => {
                warn!("ZmqConnection recv error from {}: {}", self.endpoint, e);
                Err(anyhow!("Failed to receive response from {}: {}", self.endpoint, e))
            }
            Err(_) => {
                warn!("ZmqConnection timeout after {}ms waiting for {}", timeout, self.endpoint);
                Err(anyhow!("Request to {} timed out after {}ms", self.endpoint, timeout))
            }
        }
    }

    async fn subscribe(&self, topic: &[u8]) -> Result<ZmqSubscriber> {
        let sub_endpoint = {
            use crate::registry::{global as endpoint_registry, SocketKind};
            endpoint_registry()
                .endpoint("streams", SocketKind::Sub)
                .to_zmq_string()
        };

        let subscriber = tmq::subscribe::subscribe(&self.context)
            .connect(&sub_endpoint)
            .map_err(|e| anyhow!("SUB connect to {}: {}", sub_endpoint, e))?
            .subscribe(topic)
            .map_err(|e| anyhow!("SUB subscribe to topic: {}", e))?;

        subscriber.get_socket().set_linger(0)
            .map_err(|e| anyhow!("Failed to set linger on SUB: {}", e))?;

        Ok(ZmqSubscriber(subscriber))
    }

    async fn publish(&self, _topic: &[u8]) -> Result<ZmqPublisher> {
        let push_endpoint = {
            use crate::registry::{global as endpoint_registry, SocketKind};
            endpoint_registry()
                .endpoint("streams", SocketKind::Push)
                .to_zmq_string()
        };

        let socket = self.context.socket(zmq::PUSH)
            .map_err(|e| anyhow!("Failed to create PUSH socket: {}", e))?;
        socket.connect(&push_endpoint)
            .map_err(|e| anyhow!("PUSH connect to {}: {}", push_endpoint, e))?;

        Ok(ZmqPublisher(socket))
    }
}

// ============================================================================
// ZmqSubscriber — wraps tmq::Subscribe, yields Vec<Vec<u8>> frames
// ============================================================================

/// Newtype wrapping `tmq::subscribe::Subscribe` to implement
/// `futures::Stream<Item = Result<Vec<Vec<u8>>>>`.
///
/// Converts `tmq::Multipart` → `Vec<Vec<u8>>` so both `ZmqSubscriber`
/// and `WtSubscriber` share the same stream item type.
pub struct ZmqSubscriber(pub tmq::subscribe::Subscribe);

impl Stream for ZmqSubscriber {
    type Item = Result<Vec<Vec<u8>>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.0).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e.into()))),
            Poll::Ready(Some(Ok(multipart))) => {
                let frames: Vec<Vec<u8>> = multipart.into_iter().map(|m| m.to_vec()).collect();
                Poll::Ready(Some(Ok(frames)))
            }
        }
    }
}

// ============================================================================
// ZmqPublisher — wraps zmq::Socket (PUSH) for ctrl channel
// ============================================================================

/// ZMQ PUSH socket for sending control messages (cancel, etc.).
///
/// Uses sync `zmq::Socket::send` with `DONTWAIT` — fire-and-forget.
pub struct ZmqPublisher(zmq::Socket);

// SAFETY: ZmqPublisher is only used on the thread that created it.
// zmq::Socket is !Send because libzmq sockets aren't thread-safe,
// but we never move it across threads.
unsafe impl Send for ZmqPublisher {}
unsafe impl Sync for ZmqPublisher {}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl PublishSink for ZmqPublisher {
    async fn send_frames(&self, frames: &[&[u8]]) -> Result<()> {
        for (i, frame) in frames.iter().enumerate() {
            let flags = if i < frames.len() - 1 {
                zmq::SNDMORE | zmq::DONTWAIT
            } else {
                zmq::DONTWAIT
            };
            self.0.send(*frame, flags)
                .map_err(|e| anyhow!("PUSH send failed: {}", e))?;
        }
        Ok(())
    }
}
