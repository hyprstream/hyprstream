//! Hyprstream server binary.
//!
//! This binary provides the main entry point for the Hyprstream service, a next-generation application
//! for real-time data ingestion, windowed aggregation, caching, and serving.
//!
//! # Features
//!
//! - **Data Ingestion**: Ingest data efficiently using Arrow Flight
//! - **Intelligent Caching**: High-performance caching with DuckDB
//! - **Real-time Aggregation**: Dynamic metrics and time-windowed aggregates
//! - **ADBC Integration**: Seamless connection to external databases
//!
//! # Configuration
//!
//! Configuration can be provided through multiple sources, in order of precedence:
//!
//! 1. Command-line arguments (highest precedence)
//! 2. Environment variables (prefixed with `HYPRSTREAM_`)
//! 3. User-specified configuration file (via `--config`)
//! 4. System-wide configuration (`/etc/hyprstream/config.toml`)
//! 5. Default configuration (embedded in binary)
//!
//! ## Command-line Options
//!
//! ```text
//! Options:
//!   -c, --config <FILE>             Path to configuration file
//!       --host <HOST>               Server host address [env: HYPRSTREAM_SERVER_HOST]
//!       --port <PORT>               Server port [env: HYPRSTREAM_SERVER_PORT]
//!       --storage-backend <TYPE>    Storage backend type [env: HYPRSTREAM_STORAGE_BACKEND]
//!       --cache-backend <TYPE>      Cache backend type [env: HYPRSTREAM_CACHE_BACKEND]
//!       --cache-duration <SECS>     Cache duration in seconds [env: HYPRSTREAM_CACHE_DURATION]
//!       --duckdb-connection <STR>   DuckDB connection string [env: HYPRSTREAM_DUCKDB_CONNECTION]
//!       --driver-path <PATH>        ADBC driver path [env: HYPRSTREAM_ADBC_DRIVER_PATH]
//!       --db-url <URL>              Database URL [env: HYPRSTREAM_ADBC_URL]
//!       --db-user <USER>            Database username [env: HYPRSTREAM_ADBC_USERNAME]
//!       --db-name <NAME>            Database name [env: HYPRSTREAM_ADBC_DATABASE]
//! ```
//!
//! ## Configuration File Format (TOML)
//!
//! ```toml
//! # Server Configuration
//! [server]
//! host = "127.0.0.1"     # Server host address
//! port = 50051           # Server port number
//!
//! # Storage Configuration
//! [storage]
//! backend = "cached"     # Options: "duckdb", "adbc", "cached"
//!
//! # Cache Configuration
//! [cache]
//! backend = "duckdb"     # Options: "duckdb", "adbc"
//! duration_secs = 3600   # Cache entry lifetime
//!
//! # DuckDB Configuration
//! [duckdb]
//! connection_string = ":memory:"  # Use ":memory:" for in-memory or file path
//!
//! # ADBC Configuration
//! [adbc]
//! driver_path = "/path/to/driver.so"  # ADBC driver library path
//! url = "postgresql://localhost:5432"  # Database connection URL
//! username = "user"                    # Database username
//! password = ""                        # Database password
//! database = "dbname"                  # Database name
//!
//! # Optional: Connection pool settings
//! [adbc.pool]
//! max_connections = 10        # Maximum pool connections
//! min_connections = 1         # Minimum pool connections
//! acquire_timeout_secs = 30   # Connection acquisition timeout
//! ```
//!
//! ## Storage Backends
//!
//! ### DuckDB Backend
//!
//! The DuckDB backend provides high-performance embedded storage:
//!
//! - Use `:memory:` for in-memory database (fastest, non-persistent)
//! - Use a file path for persistent storage
//! - Ideal for caching and local storage
//!
//! ### ADBC Backend
//!
//! The ADBC backend enables connection to external databases:
//!
//! - Supports any ADBC-compliant database
//! - Configurable connection pooling
//! - Requires appropriate ADBC driver
//!
//! ### Cached Backend
//!
//! The cached backend implements a two-tier storage system:
//!
//! - Fast in-memory cache (using DuckDB or ADBC)
//! - Persistent backing store (using DuckDB or ADBC)
//! - Configurable cache duration
//! - Automatic cache management
//!
//! # Examples
//!
//! ## Basic Usage
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
//! ## Advanced Configuration
//!
//! ```bash
//! # Run with cached storage and DuckDB cache
//! hyprstream \
//!   --storage-backend cached \
//!   --cache-backend duckdb \
//!   --cache-duration 3600 \
//!   --duckdb-connection ":memory:"
//!
//! # Run with ADBC backend
//! hyprstream \
//!   --storage-backend adbc \
//!   --driver-path /usr/local/lib/libadbc_driver_postgresql.so \
//!   --db-url postgresql://localhost:5432 \
//!   --db-user postgres \
//!   --db-name metrics
//! ```
//!
//! For more examples and detailed API documentation, visit the
//! [Hyprstream documentation](https://docs.rs/hyprstream).

use hyprstream_core::config::Settings;
use hyprstream_core::service::FlightServiceImpl;
use hyprstream_core::storage::{
    adbc::AdbcBackend, cached::CachedStorageBackend, duckdb::DuckDbBackend, StorageBackend,
};
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
    let backend = match settings.engine.engine.as_str() {
        "duckdb" => {
            let backend = DuckDbBackend::new(&settings.engine.connection)?;
            Arc::new(backend) as Arc<dyn StorageBackend>
        }
        "adbc" => {
            let backend = AdbcBackend::new_with_options(
                &settings.engine.connection,
                &settings.engine.options,
            )?;
            Arc::new(backend) as Arc<dyn StorageBackend>
        }
        _ => return Err("Invalid engine type".into()),
    };

    // Add caching if enabled
    let backend = if settings.cache.enabled {
        let cache: Arc<dyn StorageBackend> = match settings.cache.engine.as_str() {
            "duckdb" => Arc::new(DuckDbBackend::new(&settings.cache.connection)?),
            "adbc" => Arc::new(AdbcBackend::new_with_options(
                &settings.cache.connection,
                &settings.cache.options,
            )?),
            _ => return Err("Invalid cache engine type".into()),
        };
        Arc::new(CachedStorageBackend::new(
            cache,
            backend,
            settings.cache.max_duration_secs,
        ))
    } else {
        backend
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
        .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
