use crate::storage::StorageBackend;
use arrow_flight::flight_service_server::FlightServiceServer;
use std::sync::Arc;
use tonic::transport::Server;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod metrics;
mod service;
mod storage;

fn create_storage_backend(settings: &config::Settings) -> Result<Arc<dyn StorageBackend>, Box<dyn std::error::Error>> {
    match settings.storage.backend.as_str() {
        "duckdb" => Ok(Arc::new(storage::duckdb::DuckDbBackend::new()) as Arc<dyn StorageBackend>),
        "adbc" => Ok(Arc::new(
            storage::adbc::AdbcBackend::new(&settings.adbc)
                .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)))?,
        ) as Arc<dyn StorageBackend>),
        "cached" => {
            let cache: Arc<dyn StorageBackend> = match settings.cache.backend.as_str() {
                "duckdb" => Arc::new(storage::duckdb::DuckDbBackend::new()),
                "adbc" => Arc::new(
                    storage::adbc::AdbcBackend::new(&settings.adbc)
                        .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)))?,
                ),
                _ => return Err("Invalid cache backend type".into()),
            };

            let backing_store: Arc<dyn StorageBackend> = Arc::new(
                storage::adbc::AdbcBackend::new(&settings.adbc)
                    .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)))?,
            );

            Ok(Arc::new(storage::cached::CachedStorageBackend::new(
                cache,
                backing_store,
                settings.cache.duration_secs,
            )))
        }
        _ => Err("Invalid storage backend type".into()),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let settings = config::Settings::new()?;
    let addr = format!("{}:{}", settings.server.host, settings.server.port).parse()?;

    // Create and initialize the configured storage backend
    let backend = create_storage_backend(&settings)?;
    backend.init().await?;

    let service = service::FlightServiceImpl::new(backend);

    println!("Flight server listening on {}", addr);
    Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
