//! Arrow Flight SQL server and client for hyprstream-metrics
//!
//! This crate provides:
//! - **Flight SQL server** - embedded in `hyprstream serve` for dataset queries
//! - **Flight SQL client** - used by `hyprstream flight` CLI command
//!
//! # Architecture
//!
//! ```text
//! hyprstream serve                    hyprstream flight
//!     │                                    │
//!     └── start_flight_server()            └── FlightClient::connect()
//!              │                                    │
//!              ▼                                    ▼
//!         FlightSqlServer ◄──────────────── Flight SQL Protocol
//! ```

pub mod cli;
pub mod client;
pub mod service;

// Re-export main types
pub use client::FlightClient;
pub use service::FlightSqlServer;

use hyprstream_metrics::RegistryClient;
use hyprstream_metrics::storage::{duckdb::DuckDbBackend, StorageBackendType};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tonic::transport::{Certificate, Identity, Server, ServerTlsConfig};
use tracing::{debug, error, info};

/// Configuration for the Flight SQL server
#[derive(Debug, Clone)]
pub struct FlightConfig {
    /// Host address to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// TLS certificate (PEM)
    pub tls_cert: Option<Vec<u8>>,
    /// TLS private key (PEM)
    pub tls_key: Option<Vec<u8>>,
    /// TLS CA certificate for client auth (PEM)
    pub tls_ca: Option<Vec<u8>>,
}

impl Default for FlightConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 50051,
            tls_cert: None,
            tls_key: None,
            tls_ca: None,
        }
    }
}

impl FlightConfig {
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_tls(mut self, cert: Vec<u8>, key: Vec<u8>) -> Self {
        self.tls_cert = Some(cert);
        self.tls_key = Some(key);
        self
    }

    pub fn with_client_ca(mut self, ca: Vec<u8>) -> Self {
        self.tls_ca = Some(ca);
        self
    }

    /// Get the socket address
    pub fn addr(&self) -> Result<SocketAddr, std::net::AddrParseError> {
        format!("{}:{}", self.host, self.port).parse()
    }
}

/// Start the Flight SQL server
///
/// This function is called by `hyprstream serve` to embed the Flight SQL server
/// alongside the HTTP API.
///
/// # Arguments
///
/// * `client` - Optional registry client to look up dataset
/// * `dataset_name` - Name of the dataset in the registry (required if client provided)
/// * `config` - Flight server configuration
///
/// If `client` is `None`, starts with an in-memory database. Otherwise looks up
/// the dataset in the registry and uses file-backed storage if available.
///
/// # Example
///
/// ```rust,ignore
/// let client = LocalService::start(&base_dir).await?;
/// let config = FlightConfig::default().with_port(50051);
///
/// // Start with dataset from registry
/// tokio::spawn(async move {
///     start_flight_server(Arc::new(client), "my-metrics", config).await
/// });
///
/// // Or start with in-memory database
/// start_flight_server(None, "", config).await?;
/// ```
pub async fn start_flight_server(
    client: Option<Arc<dyn RegistryClient>>,
    dataset_name: &str,
    config: FlightConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Determine connection string based on whether we have a registry client
    let connection_string = if let Some(ref registry_client) = client {
        // Look up dataset in registry
        let tracked = registry_client
            .get_by_name(dataset_name)
            .await
            .map_err(|e| format!("Failed to look up dataset: {e}"))?
            .ok_or_else(|| format!("Dataset '{dataset_name}' not found in registry"))?;

        let dataset_path = PathBuf::from(&tracked.worktree_path);

        info!(
            dataset = %dataset_name,
            path = %dataset_path.display(),
            "Starting Flight SQL server for dataset"
        );

        // Use file-backed if database file exists, otherwise in-memory
        let db_path = dataset_path.join("data.duckdb");
        if db_path.exists() {
            db_path.to_string_lossy().to_string()
        } else {
            ":memory:".to_owned()
        }
    } else {
        info!("Starting Flight SQL server with in-memory database");
        ":memory:".to_owned()
    };

    // Create DuckDB backend
    let backend = DuckDbBackend::new(connection_string, HashMap::new(), None)
        .map_err(|e| format!("Failed to create DuckDB backend: {e}"))?;

    let addr = config.addr()?;

    // Create Flight SQL server
    let flight_service = FlightSqlServer::new(StorageBackendType::DuckDb(backend))
        .await
        .map_err(|e| format!("Failed to create Flight SQL server: {e}"))?
        .into_service();

    let mut server = Server::builder();

    // Configure TLS if provided
    if let (Some(cert), Some(key)) = (&config.tls_cert, &config.tls_key) {
        let identity = Identity::from_pem(cert, key);
        let mut tls_config = ServerTlsConfig::new().identity(identity);

        if let Some(ca) = &config.tls_ca {
            tls_config = tls_config.client_ca_root(Certificate::from_pem(ca));
        }

        server = server.tls_config(tls_config)?;
        debug!("TLS enabled for Flight SQL server");
    }

    info!(address = %addr, "Flight SQL server listening");

    server
        .add_service(flight_service)
        .serve(addr)
        .await
        .map_err(|e| {
            error!(error = %e, "Flight SQL server failed");
            Box::new(e) as Box<dyn std::error::Error + Send + Sync>
        })?;

    Ok(())
}
