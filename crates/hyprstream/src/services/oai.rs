//! OAIService - OpenAI-compatible HTTP API with ZMQ control channel
//!
//! Dual-protocol service:
//! - HTTP server for OpenAI API requests (data plane)
//! - ZMQ REQ/REP for health, metrics, shutdown (control plane)
//!
//! # Architecture
//!
//! ```text
//! HTTP Clients ──► HTTP Server (Axum) ──► OAIService
//!                        │
//!                        ├──► ModelZmqClient ──► ModelService
//!                        └──► PolicyClient ──► PolicyService
//!
//! Control ──► ZMQ REP Socket ──► OAIService (health, metrics)
//! ```
//!
//! # Usage
//!
//! OAIService is typically started via the factory system:
//!
//! ```ignore
//! // In config.toml
//! [services]
//! startup = ["event", "registry", "policy", "model", "oai"]
//!
//! [oai]
//! host = "0.0.0.0"
//! port = 8080
//! ```

use crate::config::OAIConfig;
use crate::server::{create_app, state::ServerState};
use anyhow::Result;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::service::spawner::Spawnable;
use hyprstream_rpc::transport::TransportConfig;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tracing::{error, info};

/// Service name for registry and logging
pub const SERVICE_NAME: &str = "oai";

/// OAIService - OpenAI-compatible HTTP API with ZMQ control channel
///
/// This service provides:
/// - HTTP server with OpenAI-compatible API endpoints
/// - ZMQ REQ/REP control channel for health checks and metrics
///
/// The HTTP server handles all inference requests via ModelZmqClient,
/// which communicates with ModelService over ZMQ.
pub struct OAIService {
    /// OAI-specific configuration (host, port, TLS)
    config: OAIConfig,

    /// Shared server state containing clients and metrics
    server_state: ServerState,

    /// ZMQ context for control socket
    context: Arc<zmq::Context>,

    /// Transport configuration for ZMQ control channel
    control_transport: TransportConfig,

    /// Verifying key for envelope verification
    #[allow(dead_code)]
    verifying_key: VerifyingKey,
}

impl OAIService {
    /// Create a new OAIService
    ///
    /// # Arguments
    ///
    /// * `config` - OAI configuration (host, port, TLS settings)
    /// * `server_state` - Shared state with ZMQ clients and metrics
    /// * `context` - ZMQ context for control socket
    /// * `control_transport` - Transport for ZMQ control channel
    /// * `verifying_key` - Key for verifying signed envelopes
    pub fn new(
        config: OAIConfig,
        server_state: ServerState,
        context: Arc<zmq::Context>,
        control_transport: TransportConfig,
        verifying_key: VerifyingKey,
    ) -> Self {
        Self {
            config,
            server_state,
            context,
            control_transport,
            verifying_key,
        }
    }

    /// Get the HTTP bind address
    pub fn http_addr(&self) -> Result<SocketAddr> {
        let addr_str = format!("{}:{}", self.config.host, self.config.port);
        addr_str.parse().map_err(|e| anyhow::anyhow!("Invalid address: {}", e))
    }
}

impl Spawnable for OAIService {
    fn name(&self) -> &str {
        SERVICE_NAME
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        // Register control channel endpoint
        vec![(SocketKind::Rep, self.control_transport.clone())]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<(), hyprstream_rpc::error::RpcError> {
        // Create multi-threaded runtime for HTTP server
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("runtime: {e}")))?;

        rt.block_on(async move {
            // Parse HTTP bind address
            let addr = self.http_addr().map_err(|e| {
                hyprstream_rpc::error::RpcError::SpawnFailed(format!("Invalid HTTP address: {e}"))
            })?;

            // Create HTTP server
            let app = create_app(self.server_state.clone());

            // Bind HTTP listener
            let listener = TcpListener::bind(addr).await.map_err(|e| {
                hyprstream_rpc::error::RpcError::SpawnFailed(format!("HTTP bind failed: {e}"))
            })?;

            info!(
                "OAIService HTTP server listening on http://{}",
                addr
            );
            info!(
                "OpenAI-compatible API available at http://{}/oai/v1",
                addr
            );

            // Signal ready after HTTP socket is bound
            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }

            // Notify systemd that service is ready
            let _ = hyprstream_rpc::notify::ready();

            // Create shutdown signal for axum
            let shutdown_clone = shutdown.clone();

            // Run HTTP server with graceful shutdown
            let server_result = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    shutdown_clone.notified().await;
                    info!("OAIService received shutdown signal");
                })
                .await;

            if let Err(e) = server_result {
                error!("OAIService HTTP server error: {}", e);
            }

            info!("OAIService stopped");
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_name() {
        assert_eq!(SERVICE_NAME, "oai");
    }
}
