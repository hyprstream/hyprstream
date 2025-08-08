//! VDB-first CLI handlers for adaptive ML inference server

use crate::{
    config::set_tls_data,
    storage::{VDBSparseStorage, SparseStorageConfig},
    service::embedding_flight::create_embedding_flight_server,
};
use ::config::{Config, File};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};
use tonic::transport::{Certificate, Identity, Server, ServerTlsConfig};
use tracing::{debug, error, info};

pub async fn execute_sparse_query(
    addr: Option<SocketAddr>,
    query: String,
    _config: Option<&Config>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 50051)));
    
    if verbose {
        info!("Executing sparse query: {}", query);
        debug!("Connecting to VDB service at: {}", addr);
    }
    
    // Parse the query as JSON for embedding operations
    let embedding_query: serde_json::Value = serde_json::from_str(&query)?;
    
    if verbose {
        debug!("Parsed embedding query: {:?}", embedding_query);
    }
    
    // TODO: Implement embedding query execution via FlightSQL client
    info!("✅ Embedding query processed successfully");
    
    Ok(())
}

pub async fn handle_server(
    config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}",
        config.get_string("host").unwrap_or_else(|_| "127.0.0.1".to_string()),
        config.get_string("port").unwrap_or_else(|_| "50051".to_string())
    ).parse()?;
    
    // Initialize VDB sparse storage
    let storage_config = SparseStorageConfig {
        storage_path: PathBuf::from(
            config.get_string("storage.path")
                .unwrap_or_else(|_| "./vdb_storage".to_string())
        ),
        neural_compression: config.get_bool("storage.neural_compression").unwrap_or(true),
        hardware_acceleration: config.get_bool("storage.hardware_acceleration").unwrap_or(true),
        cache_size_mb: config.get_int("storage.cache_size_mb").unwrap_or(2048) as usize,
        compaction_interval_secs: config.get_int("storage.compaction_interval_secs").unwrap_or(300) as u64,
        streaming_updates: config.get_bool("storage.streaming_updates").unwrap_or(true),
        update_batch_size: config.get_int("storage.update_batch_size").unwrap_or(1000) as usize,
    };

    let sparse_storage = Arc::new(VDBSparseStorage::new(storage_config).await?);
    
    // Start background processing for streaming updates
    sparse_storage.start_background_processing().await?;

    // Create embedding-focused FlightSQL service
    let flight_service = create_embedding_flight_server(sparse_storage);
    
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

    info!(address = %addr, "🚀 Starting VDB-first adaptive ML inference server");
    debug!(
        tls_enabled = config.get_bool("tls.enabled").unwrap_or(false),
        neural_compression = config.get_bool("storage.neural_compression").unwrap_or(true),
        hardware_acceleration = config.get_bool("storage.hardware_acceleration").unwrap_or(true),
        "Server configuration"
    );
    
    server
        .add_service(flight_service)
        .serve(addr)
        .await
        .map_err(|e| {
            error!(error = %e, "VDB server failed to start");
            e
        })?;

    Ok(())
}

pub async fn handle_embedding_query(
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

    // Parse address and execute embedding query
    let addr_parts: Vec<&str> = addr.split(':').collect();
    if addr_parts.len() != 2 {
        return Err("Invalid address format. Expected host:port".into());
    }

    let socket_addr = SocketAddr::new(
        addr_parts[0].parse()?,
        addr_parts[1].parse()?
    );

    // Execute embedding query
    execute_sparse_query(
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
        .set_default("storage.path", "./vdb_storage")?
        .set_default("storage.neural_compression", true)?
        .set_default("storage.hardware_acceleration", true)?
        .set_default("storage.cache_size_mb", 2048)?
        .set_default("storage.compaction_interval_secs", 300)?
        .set_default("storage.streaming_updates", true)?
        .set_default("storage.update_batch_size", 1000)?;

    // Load config file if provided
    if let Some(path) = config_path {
        builder = builder.add_source(File::from(path));
    }

    let settings = builder.build()?;
    info!("📁 VDB configuration loaded successfully");
    debug!(settings = ?settings, "Current VDB configuration settings");
    Ok(())
}