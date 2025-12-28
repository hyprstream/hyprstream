//! ZMQ transport implementation.
//!
//! Provides both synchronous (`ZmqTransport`) and asynchronous (`AsyncServiceClient`)
//! transports for ZMQ-based RPC communication.
//!
//! The async transport uses TMQ for proper epoll-based async I/O without
//! blocking threads or busy-wait loops.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, trace};

use super::traits::{AsyncTransport, Transport};

/// Synchronous ZMQ REQ socket transport.
///
/// Wraps a ZMQ REQ socket for synchronous request/response communication.
/// Thread-safe through internal locking.
///
/// This is used by blocking service threads (e.g., inference service).
/// For async code, use `AsyncServiceClient` instead.
pub struct ZmqTransport {
    socket: Arc<Mutex<zmq::Socket>>,
    endpoint: String,
}

impl ZmqTransport {
    /// Create a new ZMQ transport connected to the given endpoint.
    pub fn new(endpoint: &str) -> Result<Self> {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::REQ)?;
        socket.connect(endpoint)?;

        // Set reasonable timeouts
        socket.set_rcvtimeo(30_000)?; // 30 seconds
        socket.set_sndtimeo(5_000)?; // 5 seconds

        debug!("ZMQ transport connected to {}", endpoint);

        Ok(Self {
            socket: Arc::new(Mutex::new(socket)),
            endpoint: endpoint.to_string(),
        })
    }

    /// Get the endpoint this transport is connected to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

impl Transport for ZmqTransport {
    fn call(&self, request: Vec<u8>) -> Result<Vec<u8>> {
        let socket = self.socket.blocking_lock();

        trace!("Sending {} bytes to {}", request.len(), self.endpoint);
        socket.send(&request, 0)?;

        let response = socket.recv_bytes(0)?;
        trace!("Received {} bytes from {}", response.len(), self.endpoint);

        Ok(response)
    }

    fn is_connected(&self) -> bool {
        true
    }
}

/// Async ZMQ client for service communication.
///
/// Uses TMQ for proper async I/O with epoll integration.
/// Each call creates a fresh REQ socket for isolation.
///
/// # Architecture
///
/// TMQ provides true async ZMQ by:
/// 1. Using `AsyncFd` to register socket FD with tokio's reactor
/// 2. Using epoll to wait for socket readiness
/// 3. Returning `Poll::Pending` on EAGAIN (no busy-wait)
///
/// This replaces the previous `spawn_blocking` approach that created
/// new sockets per call without proper async integration.
pub struct AsyncServiceClient {
    endpoint: String,
    context: tmq::Context,
    #[allow(dead_code)]
    timeout: Duration,
}

impl AsyncServiceClient {
    /// Create a new async service client.
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            context: tmq::Context::new(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Create with custom timeout.
    pub fn with_timeout(endpoint: &str, timeout: Duration) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            context: tmq::Context::new(),
            timeout,
        }
    }

    /// Get the endpoint.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[async_trait]
impl AsyncTransport for AsyncServiceClient {
    async fn call(&self, request: Vec<u8>) -> Result<Vec<u8>> {
        // Create REQ socket using TMQ for proper async I/O
        let socket = tmq::request(&self.context)
            .connect(&self.endpoint)
            .map_err(|e| anyhow!("Failed to connect to {}: {}", self.endpoint, e))?;

        trace!("TMQ async client sending {} bytes to {}", request.len(), self.endpoint);

        // TMQ REQ pattern: send returns RequestReceiver, which we use to recv
        // Multipart::from(vec![request]) converts Vec<u8> -> zmq::Message -> Multipart
        let receiver = socket
            .send(tmq::Multipart::from(vec![request]))
            .await
            .map_err(|e| anyhow!("Failed to send request: {}", e))?;

        // Receive response
        let (response, _sender) = receiver
            .recv()
            .await
            .map_err(|e| anyhow!("Failed to receive response: {}", e))?;

        // Extract bytes from multipart message (typically single frame for RPC)
        let bytes: Vec<u8> = response
            .into_iter()
            .flat_map(|frame| frame.to_vec())
            .collect();

        trace!("TMQ async client received {} bytes", bytes.len());

        Ok(bytes)
    }

    fn is_connected(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_client_creation() {
        let client = AsyncServiceClient::new("inproc://test");
        assert_eq!(client.endpoint(), "inproc://test");
    }

    #[test]
    fn test_async_client_with_timeout() {
        let client = AsyncServiceClient::with_timeout(
            "inproc://test",
            Duration::from_secs(60),
        );
        assert_eq!(client.endpoint(), "inproc://test");
    }
}
