//! Hyprstream server binary.
//!
//! This binary provides the main entry point for the Hyprstream service.
//! It handles:
//! - Configuration loading and validation
//! - Storage backend initialization
//! - Flight SQL service setup
//! - Server startup and shutdown
//!
//! The server can be configured through:
//! - Command line arguments
//! - Environment variables
//! - Configuration files (TOML)
//!
//! # Usage
//!
//! ```bash
//! # Run with default configuration
//! hyprstream
//!
//! # Run with custom configuration file
//! hyprstream --config /path/to/config.toml
//!
//! # Run with environment variables
//! HYPRSTREAM_SERVER_HOST=0.0.0.0 HYPRSTREAM_SERVER_PORT=50051 hyprstream
//! ```
//!
//! # Storage Backends
//!
//! The server supports multiple storage backends:
//! - `duckdb`: Embedded database for high-performance local storage
//! - `adbc`: Arrow Database Connectivity for external databases
//! - `cached`: Two-tier storage with configurable caching
//!
//! # Example Configuration
//!
//! ```toml
//! [server]
//! host = "127.0.0.1"
//! port = 50051
//!
//! [storage]
//! backend = "cached"
//!
//! [cache]
//! backend = "duckdb"
//! duration_secs = 3600
//! ```

use hyprstream_core::config::Settings;
use hyprstream_core::service::FlightServiceImpl;
use hyprstream_core::storage::{adbc::AdbcBackend, cached::CachedStorageBackend, duckdb::DuckDbBackend, StorageBackend};
use std::sync::Arc;
use tonic::transport::Server;

/// Main entry point for the Hyprstream server.
///
/// This function:
/// 1. Initializes logging
/// 2. Loads configuration
/// 3. Sets up the appropriate storage backend
/// 4. Starts the Flight SQL service
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Load configuration
    let settings = Settings::new()?;

    // Create storage backend based on configuration
    let backend = match settings.storage.backend.as_str() {
        "duckdb" => {
            let backend = DuckDbBackend::new(&settings.duckdb.connection_string)?;
            Arc::new(backend) as Arc<dyn StorageBackend>
        }
        "adbc" => {
            let backend = AdbcBackend::new(&settings.adbc)?;
            Arc::new(backend) as Arc<dyn StorageBackend>
        }
        "cached" => {
            let cache: Arc<dyn StorageBackend> = match settings.cache.backend.as_str() {
                "duckdb" => Arc::new(DuckDbBackend::new(":memory:")?),
                "adbc" => Arc::new(AdbcBackend::new(&settings.adbc)?),
                _ => return Err("Invalid cache backend type".into()),
            };
            let store: Arc<dyn StorageBackend> = match settings.storage.backend.as_str() {
                "duckdb" => Arc::new(DuckDbBackend::new(&settings.duckdb.connection_string)?),
                "adbc" => Arc::new(AdbcBackend::new(&settings.adbc)?),
                _ => return Err("Invalid storage backend type".into()),
            };
            Arc::new(CachedStorageBackend::new(
                cache,
                store,
                settings.cache.duration_secs,
            ))
        }
        _ => return Err("Invalid storage backend type".into()),
    };

    // Initialize storage backend
    backend.init().await?;

    // Create and start the Flight SQL service
    let addr = format!("{}:{}", settings.server.host, settings.server.port)
        .parse()
        .unwrap();
    let service = FlightServiceImpl::new(backend);

    println!("Starting server on {}", addr);
    Server::builder()
        .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(
            service,
        ))
        .serve(addr)
        .await?;

    Ok(())
} 