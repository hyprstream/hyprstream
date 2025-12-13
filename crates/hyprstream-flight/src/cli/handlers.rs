use hyprstream_metrics::storage::{StorageBackendType, duckdb::DuckDbBackend};
use crate::service::FlightSqlServer;
use config::{Config, File};
use std::{
    collections::HashMap,
    error::Error as StdError,
    fmt,
    net::SocketAddr,
    path::PathBuf,
};
use tonic::transport::{Certificate, Identity, Server, ServerTlsConfig};
use tracing::{debug, error, info};

#[derive(Debug)]
pub enum ConnectionError {
    Timeout(String),
    Other(Box<dyn StdError>),
}

impl fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConnectionError::Timeout(msg) => write!(f, "Connection timeout: {}", msg),
            ConnectionError::Other(e) => write!(f, "Connection error: {}", e),
        }
    }
}

impl StdError for ConnectionError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            ConnectionError::Timeout(_) => None,
            ConnectionError::Other(e) => Some(e.as_ref()),
        }
    }
}

pub async fn handle_server(
    config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}",
        config.get_string("host")?.as_str(),
        config.get_string("port")?.as_str()
    ).parse()?;

    let backend = match config.get_string("storage.type")?.as_str() {
        "duckdb" => {
            let conn_str = config.get_string("storage.connection")?.to_string();
            let backend = DuckDbBackend::new(conn_str, HashMap::new(), None)?;
            StorageBackendType::DuckDb(backend)
        }
        _ => return Err("Unsupported storage type (only duckdb supported currently)".into()),
    };

    let service = FlightSqlServer::new(backend).into_service();
    let mut server = Server::builder();

    // Configure TLS if enabled
    if config.get_bool("tls.enabled").unwrap_or(false) {
        let cert = match config.get::<Vec<u8>>("tls.cert_data") {
            Ok(data) => data,
            Err(_) => {
                let path = config.get_string("tls.cert_path")
                    .map_err(|_| "TLS certificate not found")?;
                std::fs::read(path)
                    .map_err(|_| "Failed to read TLS certificate")?
            }
        };
        let key = match config.get::<Vec<u8>>("tls.key_data") {
            Ok(data) => data,
            Err(_) => {
                let path = config.get_string("tls.key_path")
                    .map_err(|_| "TLS key not found")?;
                std::fs::read(path)
                    .map_err(|_| "Failed to read TLS key")?
            }
        };
        let identity = Identity::from_pem(&cert, &key);

        let mut tls_config = ServerTlsConfig::new().identity(identity);

        if let Some(ca) = config.get::<Vec<u8>>("tls.ca_data").ok()
            .or_else(|| config.get_string("tls.ca_path").ok()
                .and_then(|p| if p.is_empty() { None } else { Some(p) })
                .and_then(|p| std::fs::read(p).ok())) {
            tls_config = tls_config.client_ca_root(Certificate::from_pem(&ca));
        }

        server = server.tls_config(tls_config)?;
    }

    info!(address = %addr, "Starting Flight SQL server");
    debug!(
        tls_enabled = config.get_bool("tls.enabled").unwrap_or(false),
        storage_type = config.get_string("storage.type").unwrap_or_default(),
        "Server configuration"
    );

    server
        .add_service(service)
        .serve(addr)
        .await
        .map_err(|e| {
            error!(error = %e, "Server failed to start");
            e
        })?;

    Ok(())
}

pub fn handle_config(config_path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = Config::builder()
        .set_default("host", "127.0.0.1")?
        .set_default("port", "50051")?
        .set_default("storage.type", "duckdb")?
        .set_default("storage.connection", ":memory:")?;

    // Load config file if provided
    if let Some(path) = config_path {
        builder = builder.add_source(File::from(path));
    }

    let settings = builder.build()?;
    info!("Configuration loaded successfully");
    debug!(settings = ?settings, "Current configuration settings");
    Ok(())
}
