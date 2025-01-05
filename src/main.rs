use crate::storage::StorageBackend;
use arrow_flight::flight_service_server::FlightServiceServer;
use std::sync::Arc;
use tonic::transport::Server;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod metrics;
mod service;
mod storage;

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

    // Initialize backends
    let duckdb = Arc::new(storage::duckdb::DuckDbBackend::new());

    // Create ADBC backend with configuration
    let adbc = Arc::new(
        storage::adbc::AdbcBackend::new(&settings.adbc)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)))?,
    );

    // Create cached backend using DuckDB as cache and ADBC as backing store
    /*let backend = Arc::new(storage::cached::CachedStorageBackend::new(
        duckdb,                       // Use DuckDB as the cache
        adbc,                         // Use ADBC as the backing store
        settings.cache.duration_secs, // Cache duration from config
    ));*/
    let backend = duckdb;

    // Initialize the backend
    backend.init().await?;

    let service = service::FlightServiceImpl::new(backend);

    println!("Flight server listening on {}", addr);
    Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
