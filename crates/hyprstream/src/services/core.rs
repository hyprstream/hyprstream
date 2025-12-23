//! Core service infrastructure for ZMQ-based services
//!
//! Provides the foundation for REQ/REP services and clients using ZMQ.
//! Services run on dedicated threads due to ZMQ socket thread locality requirements.

use crate::zmq::global_context;
use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

/// Trait for ZMQ-based services
///
/// Implement this trait to create a service that handles REQ/REP requests.
/// The service runs on a dedicated thread and processes requests synchronously.
pub trait ZmqService: Send + 'static {
    /// Process a request and return a response
    ///
    /// The request is raw bytes (Cap'n Proto encoded).
    /// Returns raw bytes for the response (Cap'n Proto encoded).
    fn handle_request(&self, request: &[u8]) -> Result<Vec<u8>>;

    /// Get the name of this service (for logging)
    fn name(&self) -> &str;
}

/// REQ/REP service runner
///
/// Runs a ZmqService on a dedicated thread, handling incoming requests
/// and sending responses.
pub struct ServiceRunner {
    endpoint: String,
}

impl ServiceRunner {
    /// Create a new service runner bound to the given endpoint
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
        }
    }

    /// Run the service on a dedicated blocking thread
    ///
    /// This spawns a blocking task that:
    /// 1. Creates a REP socket on the current thread
    /// 2. Binds to the endpoint
    /// 3. Loops receiving requests and sending responses
    ///
    /// Returns a handle that can be used to stop the service.
    pub fn run<S: ZmqService>(self, service: S) -> ServiceHandle {
        let endpoint = self.endpoint.clone();
        // Use AtomicBool for shutdown signaling instead of mpsc channel
        // This avoids the nested runtime anti-pattern
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let handle = tokio::task::spawn_blocking(move || {
            Self::service_loop(endpoint, service, shutdown_clone)
        });

        ServiceHandle {
            task: Some(handle),
            shutdown: Some(shutdown),
        }
    }

    /// Internal service loop
    fn service_loop<S: ZmqService>(endpoint: String, service: S, shutdown: Arc<AtomicBool>) {
        let ctx = global_context();

        // Create REP socket
        let socket = match ctx.socket(zmq::REP) {
            Ok(s) => s,
            Err(e) => {
                error!("failed to create REP socket for {}: {}", service.name(), e);
                return;
            }
        };

        // Set socket options
        if let Err(e) = socket.set_rcvtimeo(100) {
            // 100ms timeout for checking shutdown
            warn!("failed to set receive timeout: {}", e);
        }

        // Set LINGER to 0 for immediate close without blocking
        if let Err(e) = socket.set_linger(0) {
            warn!("failed to set socket linger: {}", e);
        }

        // Bind to endpoint
        if let Err(e) = socket.bind(&endpoint) {
            error!(
                "failed to bind {} to {}: {}",
                service.name(),
                endpoint,
                e
            );
            return;
        }

        info!("{} service bound to {}", service.name(), endpoint);

        // Main service loop
        // No nested runtime needed - AtomicBool check is lock-free
        loop {
            // Check for shutdown signal (lock-free atomic read)
            if shutdown.load(Ordering::Relaxed) {
                debug!("{} service received shutdown signal", service.name());
                break;
            }

            // Try to receive a request (with timeout)
            match socket.recv_bytes(0) {
                Ok(request) => {
                    trace!(
                        "{} received request ({} bytes)",
                        service.name(),
                        request.len()
                    );

                    // Process the request
                    let response = match service.handle_request(&request) {
                        Ok(resp) => resp,
                        Err(e) => {
                            error!("{} request handling error: {}", service.name(), e);
                            // Send error response (empty for now, will be capnp error later)
                            vec![]
                        }
                    };

                    // Send response
                    if let Err(e) = socket.send(&response, 0) {
                        error!("{} failed to send response: {}", service.name(), e);
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout, just continue to check shutdown
                    continue;
                }
                Err(e) => {
                    warn!("{} recv error: {}", service.name(), e);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }

        info!("{} service stopped", service.name());
    }
}

/// Handle for a running service
pub struct ServiceHandle {
    task: Option<tokio::task::JoinHandle<()>>,
    /// Shutdown signal using AtomicBool (avoids nested runtime anti-pattern)
    shutdown: Option<Arc<AtomicBool>>,
}

impl ServiceHandle {
    /// Create a dummy handle for services that manage their own lifecycle
    pub fn dummy() -> Self {
        Self {
            task: None,
            shutdown: None,
        }
    }

    /// Stop the service gracefully
    pub async fn stop(self) {
        // Signal shutdown via atomic flag (no async needed)
        if let Some(shutdown) = &self.shutdown {
            shutdown.store(true, Ordering::Relaxed);
        }
        // Wait for task to complete
        if let Some(task) = self.task {
            let _ = task.await;
        }
    }

    /// Check if the service is still running
    pub fn is_running(&self) -> bool {
        self.task.as_ref().map(|t| !t.is_finished()).unwrap_or(true)
    }
}

/// Client for REQ/REP services
///
/// Note: ServiceClient is not thread-safe because ZMQ sockets are not thread-safe.
/// Create a new client for each thread/task that needs to communicate with the service.
pub struct ServiceClient {
    socket: zmq::Socket,
    endpoint: String,
}

impl ServiceClient {
    /// Connect to a service at the given endpoint
    pub fn connect(endpoint: &str) -> Result<Self> {
        let ctx = global_context();

        let socket = ctx
            .socket(zmq::REQ)
            .map_err(|e| anyhow!("failed to create REQ socket: {}", e))?;

        socket
            .connect(endpoint)
            .map_err(|e| anyhow!("failed to connect to {}: {}", endpoint, e))?;

        // Set LINGER to 0 for immediate close without blocking
        socket
            .set_linger(0)
            .map_err(|e| anyhow!("failed to set socket linger: {}", e))?;

        debug!("ServiceClient connected to {}", endpoint);

        Ok(Self {
            socket,
            endpoint: endpoint.to_string(),
        })
    }

    /// Send a request and wait for a response
    ///
    /// This is a blocking call.
    pub fn call(&self, request: &[u8]) -> Result<Vec<u8>> {
        self.socket
            .send(request, 0)
            .map_err(|e| anyhow!("failed to send request: {}", e))?;

        let response = self
            .socket
            .recv_bytes(0)
            .map_err(|e| anyhow!("failed to receive response: {}", e))?;

        trace!(
            "ServiceClient {} call completed ({} bytes response)",
            self.endpoint,
            response.len()
        );

        Ok(response)
    }

    /// Send a request and wait for a response with timeout
    pub fn call_with_timeout(&self, request: &[u8], timeout_ms: i32) -> Result<Vec<u8>> {
        // Set receive timeout
        self.socket
            .set_rcvtimeo(timeout_ms)
            .map_err(|e| anyhow!("failed to set timeout: {}", e))?;

        self.socket
            .send(request, 0)
            .map_err(|e| anyhow!("failed to send request: {}", e))?;

        let response = self
            .socket
            .recv_bytes(0)
            .map_err(|e| anyhow!("failed to receive response (timeout={}ms): {}", timeout_ms, e))?;

        // Reset timeout
        let _ = self.socket.set_rcvtimeo(-1);

        Ok(response)
    }
}

/// Async wrapper for ServiceClient
///
/// This wraps the blocking ServiceClient for use in async contexts.
/// Each call spawns a blocking task.
pub struct AsyncServiceClient {
    endpoint: String,
}

impl AsyncServiceClient {
    /// Create a new async client for the given endpoint
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
        }
    }

    /// Get the endpoint this client is connected to
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Send a request and wait for a response asynchronously
    pub async fn call(&self, request: Vec<u8>) -> Result<Vec<u8>> {
        let endpoint = self.endpoint.clone();

        tokio::task::spawn_blocking(move || {
            let client = ServiceClient::connect(&endpoint)?;
            client.call(&request)
        })
        .await
        .map_err(|e| anyhow!("task join error: {}", e))?
    }

    /// Send a request with timeout asynchronously
    pub async fn call_with_timeout(&self, request: Vec<u8>, timeout_ms: i32) -> Result<Vec<u8>> {
        let endpoint = self.endpoint.clone();

        tokio::task::spawn_blocking(move || {
            let client = ServiceClient::connect(&endpoint)?;
            client.call_with_timeout(&request, timeout_ms)
        })
        .await
        .map_err(|e| anyhow!("task join error: {}", e))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoService;

    impl ZmqService for EchoService {
        fn handle_request(&self, request: &[u8]) -> Result<Vec<u8>> {
            Ok(request.to_vec())
        }

        fn name(&self) -> &str {
            "echo"
        }
    }

    #[tokio::test]
    async fn test_service_runner() {
        // Use a unique endpoint for this test
        let endpoint = "inproc://test-echo-service";

        // Start the service
        let runner = ServiceRunner::new(endpoint);
        let handle = runner.run(EchoService);

        // Give the service time to bind
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Test the service
        let response = tokio::task::spawn_blocking(move || {
            let client = ServiceClient::connect(endpoint).unwrap();
            client.call(b"hello")
        })
        .await
        .unwrap();

        assert_eq!(response.unwrap(), b"hello");

        // Stop the service
        handle.stop().await;
    }

    #[tokio::test]
    async fn test_async_client() {
        let endpoint = "inproc://test-async-echo";

        // Start the service
        let runner = ServiceRunner::new(endpoint);
        let handle = runner.run(EchoService);

        // Give the service time to bind
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Test async client
        let client = AsyncServiceClient::new(endpoint);
        let response = client.call(b"async hello".to_vec()).await.unwrap();
        assert_eq!(response, b"async hello");

        // Stop the service
        handle.stop().await;
    }
}
