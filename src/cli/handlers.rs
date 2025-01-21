use crate::{
    service::FlightSqlServer,
    storage::{StorageBackendType, adbc::AdbcBackend, duckdb::DuckDbBackend},
    config::{get_tls_config, set_tls_data},
};
use std::error::Error as StdError;
use std::fmt;
use tracing::{debug, error, info};

use arrow_flight::flight_service_client::FlightServiceClient as FlightSqlClient;
use ::config::{Config, File};
use tonic::transport::{Certificate, ClientTlsConfig, Identity, Server, ServerTlsConfig};
use futures::StreamExt;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    net::SocketAddr,
};
use bytes::Bytes;

#[derive(Debug)]
enum ConnectionError {
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

pub async fn execute_sql(
    addr: Option<SocketAddr>,
    sql: String,
    config: Option<&Config>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 50051)));
    let scheme = if config.map(|c| c.get_bool("tls.enabled").unwrap_or(false)).unwrap_or(false) { "https" } else { "http" };
    let channel = {
        debug!(
            scheme = scheme,
            ip = %addr.ip(),
            port = %addr.port(),
            tls_enabled = config.map(|c| c.get_bool("tls.enabled").unwrap_or(false)).unwrap_or(false),
            "Connecting to endpoint"
        );
        
        let mut endpoint = tonic::transport::Channel::from_shared(format!("{}://{}:{}", scheme, addr.ip(), addr.port()))?
            .timeout(std::time::Duration::from_secs(60))  // Increase timeout for TLS handshake
            .tcp_keepalive(Some(std::time::Duration::from_secs(5)))
            .connect_timeout(std::time::Duration::from_secs(30));  // Separate connect timeout

        debug!("Channel endpoint created with 60s timeout and TCP keepalive");

        if let Some(config) = config {
            if let Some((identity, ca_cert)) = get_tls_config(config) {
                debug!("Configuring TLS client with identity and CA cert");
                debug!("Creating client TLS config with identity");
                
                let mut tls = ClientTlsConfig::new()
                    .domain_name("localhost")
                    .identity(identity);
                
                if let Some(ca) = ca_cert {
                    debug!("Adding CA certificate to TLS config");
                    tls = tls.ca_certificate(ca);
                }

                debug!("Client TLS config created with SNI name 'localhost'");
                debug!("Attempting TLS connection with server...");
                debug!("Applying TLS configuration to endpoint");
                
                endpoint = endpoint.tls_config(tls)?;
            } else {
                debug!("No TLS configuration found in config");
            }
        } else {
            debug!("No config provided for TLS");
        }

        debug!("Attempting to connect to endpoint...");
        
        match endpoint.connect().await {
            Ok(chan) => {
                debug!("Successfully connected to endpoint");
                Ok(chan)
            }
            Err(e) => {
                let err_str = e.to_string();
                error!(error = %err_str, "Connection error occurred");
                if let Some(source) = e.source() {
                    error!(source = %source, "Error source");
                }
                
                let conn_error = if err_str.contains("transport error") ||
                   err_str.contains("deadline has elapsed") ||
                   err_str.contains("connection refused") {
                    ConnectionError::Timeout(err_str)
                } else {
                    ConnectionError::Other(Box::new(e))
                };
                Err(Box::new(conn_error))
            }
        }
    }.map_err(|e| {
        let err_str = e.to_string();
        if err_str.contains("transport error") ||
           err_str.contains("deadline has elapsed") ||
           err_str.contains("connection refused") {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "Connection timed out"
            )) as Box<dyn std::error::Error>
        } else {
            Box::new(e) as Box<dyn std::error::Error>
        }
    })?;

    let mut client = FlightSqlClient::new(channel);
    
    // Create JSON command structure
    let json = if sql.trim().to_uppercase().starts_with("CREATE ") ||
                 sql.trim().to_uppercase().starts_with("DROP ") ||
                 sql.trim().to_uppercase().starts_with("ALTER ") {
        serde_json::json!({
            "type": "sql.execute",
            "data": sql
        })
    } else {
        serde_json::json!({
            "type": "sql.query",
            "data": sql
        })
    };

    let action = arrow_flight::Action {
        r#type: "CommandStatementQuery".to_string(),
        body: Bytes::from(serde_json::to_vec(&json)?),
    };

    // Execute the action and get results
    let result = client.do_action(tonic::Request::new(action)).await;
    debug!(result = ?result, "SQL execution result");

    match result {
        Ok(mut response) => {
            // For queries that return data, fetch results using do_get
            if !sql.trim().to_uppercase().starts_with("CREATE ") &&
               !sql.trim().to_uppercase().starts_with("DROP ") &&
               !sql.trim().to_uppercase().starts_with("ALTER ") {
                let result = response.get_mut().next().await
                    .ok_or_else(|| std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "No ticket received for query"
                    ))??;

                let ticket = arrow_flight::Ticket {
                    ticket: result.body,
                };

                let mut stream = client.do_get(tonic::Request::new(ticket)).await?.into_inner();
                
                // Process results
                while let Some(data) = stream.message().await? {
                    if verbose {
                        debug!(data_size = data.data_body.len(), "Received flight data");
                    }
                }
            }
            
            debug!("SQL execution completed successfully");
            Ok(())
        }
        Err(status) => {
            let err_str = status.to_string();
            error!(error = %err_str, "SQL execution failed");
            
            if err_str.contains("transport error") ||
               err_str.contains("deadline has elapsed") ||
               err_str.contains("connection refused") {
                let timeout_err = std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "Connection timed out"
                );
                error!(error = %timeout_err, "Connection timeout occurred");
                Err(Box::new(timeout_err))
            } else {
                error!(status = %status, "SQL execution error");
                Err(Box::new(status))
            }
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
        "adbc" => {
            let driver = config.get_string("storage.driver")?.to_string();
            let conn_str = config.get_string("storage.connection")?.to_string();
            let backend = AdbcBackend::new(&driver, Some(&conn_str), None)?;
            StorageBackendType::Adbc(backend)
        }
        _ => return Err("Unsupported storage type".into()),
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

pub async fn handle_sql(
    host: Option<String>,
    query: &str,
    tls_cert: Option<&Path>,
    tls_key: Option<&Path>,
    tls_ca: Option<&Path>,
    _tls_skip_verify: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = host.unwrap_or_else(|| "localhost:50051".to_string());

    // Create Config with TLS settings if certificates are provided
    let config = match (tls_cert, tls_key) {
        (Some(cert_path), Some(key_path)) => {
            let cert = tokio::fs::read(cert_path).await?;
            let key = tokio::fs::read(key_path).await?;
            let ca = if let Some(ca_path) = tls_ca {
                Some(tokio::fs::read(ca_path).await?)
            } else {
                None
            };

            let config = set_tls_data(
                Config::builder(),
                &cert,
                &key,
                ca.as_deref(),
            )?
            .build()?;

            Some(config)
        }
        _ => None,
    };

    // Parse address and execute SQL
    let addr_parts: Vec<&str> = addr.split(':').collect();
    if addr_parts.len() != 2 {
        return Err("Invalid address format. Expected host:port".into());
    }

    let socket_addr = SocketAddr::new(
        addr_parts[0].parse()?,
        addr_parts[1].parse()?
    );

    // Execute SQL and process results
    execute_sql(
        Some(socket_addr),
        query.to_string(),
        config.as_ref(),
        verbose,
    ).await?;

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
