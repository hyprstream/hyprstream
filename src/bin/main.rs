use hyprstream::config::Settings;
use hyprstream::service::FlightServiceImpl;
use hyprstream::storage::{adbc::AdbcBackend, cached::CachedStorageBackend, duckdb::DuckDbBackend, StorageBackend};
use std::sync::Arc;
use tonic::transport::Server;

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