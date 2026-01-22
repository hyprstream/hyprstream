//! FlightService - Arrow Flight SQL with ZMQ control channel
//!
//! Dual-protocol service:
//! - gRPC server for Flight SQL queries (data plane)
//! - ZMQ REQ/REP for health, metrics, shutdown (control plane)
//!
//! # Architecture
//!
//! ```text
//! Flight SQL Clients ──► gRPC Server ──► FlightService
//!                             │
//!                             └──► DuckDB / Dataset Backend
//!
//! Control ──► ZMQ REP Socket ──► FlightService (health, metrics)
//! ```
//!
//! # Usage
//!
//! FlightService is typically started via the factory system:
//!
//! ```ignore
//! // In config.toml
//! [services]
//! startup = ["event", "registry", "policy", "model", "flight"]
//!
//! [flight]
//! host = "0.0.0.0"
//! port = 50051
//! default_dataset = "my-metrics"
//! ```

use crate::config::FlightConfig as HyprFlightConfig;
use anyhow::Result;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::service::spawner::Spawnable;
use hyprstream_rpc::transport::TransportConfig;
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::{error, info};

/// Service name for registry and logging
pub const SERVICE_NAME: &str = "flight";

/// FlightService - Arrow Flight SQL with ZMQ control channel
///
/// This service provides:
/// - gRPC server with Flight SQL protocol for dataset queries
/// - ZMQ REQ/REP control channel for health checks
///
/// The Flight server uses hyprstream-flight crate for the actual
/// Flight SQL implementation with DuckDB backend.
pub struct FlightService {
    /// Flight configuration (host, port, dataset)
    config: HyprFlightConfig,

    /// Optional registry client for dataset lookup
    registry_client: Option<Arc<dyn hyprstream_metrics::RegistryClient>>,

    /// ZMQ context for control socket
    context: Arc<zmq::Context>,

    /// Transport configuration for ZMQ control channel
    control_transport: TransportConfig,

    /// Verifying key for envelope verification
    #[allow(dead_code)]
    verifying_key: VerifyingKey,
}

impl FlightService {
    /// Create a new FlightService
    ///
    /// # Arguments
    ///
    /// * `config` - Flight configuration (host, port, dataset)
    /// * `registry_client` - Optional registry client for dataset lookup
    /// * `context` - ZMQ context for control socket
    /// * `control_transport` - Transport for ZMQ control channel
    /// * `verifying_key` - Key for verifying signed envelopes
    pub fn new(
        config: HyprFlightConfig,
        registry_client: Option<Arc<dyn hyprstream_metrics::RegistryClient>>,
        context: Arc<zmq::Context>,
        control_transport: TransportConfig,
        verifying_key: VerifyingKey,
    ) -> Self {
        Self {
            config,
            registry_client,
            context,
            control_transport,
            verifying_key,
        }
    }
}

impl Spawnable for FlightService {
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
        // Create multi-threaded runtime for gRPC server
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("runtime: {}", e)))?;

        rt.block_on(async move {
            // Build hyprstream_flight config from our config
            let flight_config = hyprstream_flight::FlightConfig::default()
                .with_host(&self.config.host)
                .with_port(self.config.port);

            // Get dataset name (empty string for in-memory mode)
            let dataset_name = self.config.default_dataset.as_deref().unwrap_or("");

            info!(
                "FlightService starting on {}:{} (dataset: {})",
                self.config.host,
                self.config.port,
                if dataset_name.is_empty() { "<in-memory>" } else { dataset_name }
            );

            // Signal ready before starting server
            // Note: The Flight server doesn't expose a "bound" callback,
            // so we signal ready optimistically
            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }

            // Notify systemd that service is ready
            let _ = hyprstream_rpc::notify::ready();

            // Run Flight server with shutdown handling
            // start_flight_server blocks, so we need to select! with shutdown
            tokio::select! {
                biased;

                _ = shutdown.notified() => {
                    info!("FlightService received shutdown signal");
                    // Note: Tonic server doesn't have graceful shutdown in this version
                    // It will be terminated when the runtime drops
                }

                result = hyprstream_flight::start_flight_server(
                    self.registry_client.clone(),
                    dataset_name,
                    flight_config,
                ) => {
                    match result {
                        Ok(()) => info!("FlightService stopped normally"),
                        Err(e) => error!("FlightService error: {}", e),
                    }
                }
            }

            info!("FlightService stopped");
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_name() {
        assert_eq!(SERVICE_NAME, "flight");
    }
}
