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
//!       --engine <TYPE>             Primary storage engine type [env: HYPRSTREAM_ENGINE]
//!       --engine-connection <STR>   Engine connection string [env: HYPRSTREAM_ENGINE_CONNECTION]
//!       --engine-options <KEY=VAL>  Engine options (can be specified multiple times) [env: HYPRSTREAM_ENGINE_OPTIONS]
//!       --enable-cache              Enable caching [env: HYPRSTREAM_ENABLE_CACHE]
//!       --cache-engine <TYPE>       Cache engine type [env: HYPRSTREAM_CACHE_ENGINE]
//!       --cache-connection <STR>    Cache connection string [env: HYPRSTREAM_CACHE_CONNECTION]
//!       --cache-options <KEY=VAL>   Cache options (can be specified multiple times) [env: HYPRSTREAM_CACHE_OPTIONS]
//!       --cache-max-duration <SECS> Cache max duration in seconds [env: HYPRSTREAM_CACHE_MAX_DURATION]
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
//! # Engine Configuration
//! [engine]
//! engine = "duckdb"      # Options: "duckdb", "adbc"
//! connection = ":memory:"
//! options = { }          # Engine-specific options
//!
//! # Cache Configuration
//! [cache]
//! enabled = true         # Enable caching
//! engine = "duckdb"      # Options: "duckdb", "adbc"
//! connection = ":memory:"
//! max_duration_secs = 3600
//! options = { }          # Cache-specific options
//! ```
//!
//! ## Storage Backends
//!
//! ### DuckDB Backend
//!
//! The DuckDB backend provides high-performance embedded storage:
//!
//! ```toml
//! [engine]
//! engine = "duckdb"
//! connection = ":memory:"  # Use ":memory:" for in-memory or file path
//! options = {
//!     threads = "4",      # Number of threads (optional)
//!     read_only = "false" # Read-only mode (optional)
//! }
//! ```
//!
//! ### ADBC Backend
//!
//! The ADBC backend enables connection to external databases:
//!
//! ```toml
//! [engine]
//! engine = "adbc"
//! connection = "postgresql://localhost:5432"
//! options = {
//!     driver_path = "/usr/local/lib/libadbc_driver_postgresql.so",  # Required
//!     username = "postgres",                                        # Optional
//!     password = "secret",                                         # Optional
//!     database = "metrics",                                        # Optional
//!     pool_max = "10",                                            # Optional
//!     pool_min = "1",                                             # Optional
//!     connect_timeout = "30"                                      # Optional
//! }
//! ```
//!
//! ### Cached Backend
//!
//! The cached backend implements a two-tier storage system:
//!
//! ```toml
//! [engine]
//! engine = "duckdb"
//! connection = "data.db"
//! options = { }
//!
//! [cache]
//! enabled = true
//! engine = "duckdb"
//! connection = ":memory:"
//! max_duration_secs = 3600
//! options = {
//!     threads = "2"
//! }
//! ```
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
//! # Run with DuckDB storage and caching
//! hyprstream \
//!   --engine duckdb \
//!   --engine-connection ":memory:" \
//!   --engine-options threads=4 \
//!   --enable-cache \
//!   --cache-engine duckdb \
//!   --cache-connection ":memory:" \
//!   --cache-max-duration 3600
//!
//! # Run with ADBC PostgreSQL backend
//! hyprstream \
//!   --engine adbc \
//!   --engine-connection "postgresql://localhost:5432" \
//!   --engine-options driver_path=/usr/local/lib/libadbc_driver_postgresql.so \
//!   --engine-options username=postgres \
//!   --engine-options database=metrics \
//!   --engine-options pool_max=10
//!
//! # Run with ADBC backend and DuckDB cache
//! hyprstream \
//!   --engine adbc \
//!   --engine-connection "postgresql://localhost:5432" \
//!   --engine-options driver_path=/usr/local/lib/libadbc_driver_postgresql.so \
//!   --engine-options username=postgres \
//!   --enable-cache \
//!   --cache-engine duckdb \
//!   --cache-connection ":memory:" \
//!   --cache-options threads=2 \
//!   --cache-max-duration 3600
//! ```
//!
//! For more examples and detailed API documentation, visit the
//! [Hyprstream documentation](https://docs.rs/hyprstream).

use clap::Parser;
use hyprstream_core::{
    config::{CliArgs, Settings},
    service::FlightServiceImpl,
    storage::{
        adbc::AdbcBackend,
        duckdb::DuckDbBackend,
        StorageBackend,
    },
};
use std::sync::Arc;
use tonic::transport::Server;
use arrow_flight::flight_service_server::FlightServiceServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let cli_args = CliArgs::parse();
    
    // Initialize settings with CLI args
    let settings = Settings::new(cli_args)?;

    // Create the engine backend
    let engine_backend: Arc<dyn StorageBackend> = match settings.engine.engine.as_str() {
        "adbc" => Arc::new(
            AdbcBackend::new_with_options(
                &settings.engine.connection,
                &settings.engine.options,
                settings.engine.credentials.as_ref(),
            )?
        ),
        "duckdb" => Arc::new(DuckDbBackend::new_with_options(
            &settings.engine.connection,
            &settings.engine.options,
            None,
        )?),
        engine => return Err(format!("Unsupported engine backend: {}", engine).into()),
    };

    // Create the cache backend if configured
    let cache_backend = if settings.cache.enabled {
        match settings.cache.engine.as_str() {
            "adbc" => Some(Arc::new(
                AdbcBackend::new_with_options(
                    &settings.cache.connection,
                    &settings.cache.options,
                    settings.cache.credentials.as_ref(),
                )?
            ) as Arc<dyn StorageBackend>),
            "duckdb" => Some(Arc::new(DuckDbBackend::new_with_options(
                &settings.cache.connection,
                &settings.cache.options,
                None,
            )?) as Arc<dyn StorageBackend>),
            engine => return Err(format!("Unsupported cache backend: {}", engine).into()),
        }
    } else {
        None
    };

    // Initialize the backends
    engine_backend.init().await?;
    if let Some(ref cache) = cache_backend {
        cache.init().await?;
    }

    // Create and start the service
    let addr = format!("{}:{}", settings.server.host, settings.server.port).parse()?;
    let service = match cache_backend {
        Some(cache) => FlightServiceImpl::new_with_cache(engine_backend, cache),
        None => FlightServiceImpl::new(engine_backend),
    };
    let service = FlightServiceServer::new(service);

    println!("Starting server on {}", addr);

    Server::builder()
        .add_service(service)
        .serve(addr)
        .await?;

    Ok(())
}
