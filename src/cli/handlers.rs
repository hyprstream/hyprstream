use crate::{
    config::{get_tls_config, set_tls_data},
    storage::{StorageBackendType, adbc::AdbcBackend, duckdb::DuckDbBackend},
    service::FlightSqlServer,
};
use adbc_core::{
    driver_manager::ManagedDriver,
    options::{OptionDatabase, OptionValue},
    Connection, Driver, Database, Statement,
};
use arrow::{
    record_batch::RecordBatchReader,
    util::pretty,
};
use ::config::{Config, File};
use std::{
    collections::HashMap,
    error::Error as StdError,
    fmt,
    net::SocketAddr,
    path::{Path, PathBuf},
};
use tonic::transport::{Certificate, Identity, Server, ServerTlsConfig};
use tracing::{debug, error, info};

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
    
    // Create ADBC driver and database
    let mut driver = ManagedDriver::load_dynamic_from_filename(
        "/home/birdetta/.local/share/mamba/lib/libadbc_driver_flightsql.so", //adbc-driver-flightsql",
        None,
        adbc_core::options::AdbcVersion::V100,
    )?;

    // Create database with options
    // Build URI with TLS if enabled
    let uri = if let Some(config) = config {
        if config.get_bool("tls.enabled").unwrap_or(false) {
            format!("grpc+tls://{}:{}", addr.ip(), addr.port())
        } else {
            format!("grpc://{}:{}", addr.ip(), addr.port())
        }
    } else {
        format!("grpc://{}:{}", addr.ip(), addr.port())
    };

    let mut options = vec![(
        OptionDatabase::Uri,
        OptionValue::String(uri),
    )];

    let mut database = driver.new_database_with_opts(options)?;
    
    if verbose {
        debug!(uri = ?format!("flight-sql://{}:{}", addr.ip(), addr.port()), "Connecting to database");
    }
    
    // Create connection with options
    let conn_options = vec![];  // Add connection-specific options if needed
    let mut conn = database.new_connection_with_opts(conn_options)?;
    
    // Create and prepare statement
    let mut stmt = conn.new_statement()?;
    stmt.set_sql_query(&sql)?;
    
    // Execute statement and get results
    let mut reader = stmt.execute()?;
    
    if verbose {
        debug!(schema = ?reader.schema().as_ref(), "Query schema");
    }
    
    // Process results in streaming fashion
    let mut batches = Vec::new();
    while let Some(result) = reader.next() {
        match result {
            Ok(batch) => {
                if verbose {
                    debug!(rows = batch.num_rows(), "Received batch");
                }
                batches.push(batch);
            }
            Err(e) => {
                error!("Error reading batch: {}", e);
                return Err(Box::new(e));
            }
        }
    }
    
    if !batches.is_empty() {
        let table = pretty::pretty_format_batches(&batches)?;
        info!("\n{}", table);
    }
    
    Ok(())
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
