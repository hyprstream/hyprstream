use anyhow::{Result, Context};
use std::sync::Arc;
use std::fs::File;
use daemonize::Daemonize;
use crate::{
    config::Settings,
    models::{storage::TimeSeriesModelStorage, ModelStorage},
    storage::{
        StorageBackendType,
        StorageBackend,
        adbc::AdbcBackend,
        duckdb::DuckDbBackend,
    },
    service::FlightSqlService,
};
use tonic::transport::Server;

pub async fn run_server(detach: bool, settings: Settings) -> Result<()> {
    if detach {
        // Set up daemon logging
        let stdout = File::create("/tmp/hyprstream.out")
            .context("Failed to create daemon stdout file")?;
        let stderr = File::create("/tmp/hyprstream.err")
            .context("Failed to create daemon stderr file")?;

        // Configure daemon
        let daemonize = Daemonize::new()
            .pid_file("/tmp/hyprstream.pid")
            .chown_pid_file(true)
            .working_directory("/tmp")
            .stdout(stdout)
            .stderr(stderr);

        // Start daemon
        tracing::info!("Starting server in detached mode");
        daemonize.start()
            .context("Failed to start daemon")?;
    }

    // Create storage backend
    let engine_backend: Arc<StorageBackendType> = Arc::new(match settings.engine.engine.as_str() {
        "adbc" => StorageBackendType::Adbc(
            AdbcBackend::new_with_options(
                &settings.engine.connection,
                &settings.engine.options,
                settings.engine.credentials.as_ref(),
            ).context("Failed to create ADBC backend")?
        ),
        "duckdb" => StorageBackendType::DuckDb(
            DuckDbBackend::new_with_options(
                &settings.engine.connection,
                &settings.engine.options,
                settings.engine.credentials.as_ref(),
            ).context("Failed to create DuckDB backend")?
        ),
        _ => anyhow::bail!("Unsupported engine type"),
    });

    // Initialize backend
    engine_backend.init().await.context("Failed to initialize storage backend")?;

    // Create model storage
    let model_storage = Box::new(TimeSeriesModelStorage::new(engine_backend.clone()));
    model_storage.init().await.context("Failed to initialize model storage")?;

    // Create the service
    let service = FlightSqlService::new(engine_backend, model_storage);

    // Start the server
    let addr = format!("{}:{}", settings.server.host, settings.server.port).parse()
        .context("Invalid listen address")?;

    tracing::warn!("This is a pre-release alpha for preview purposes only.");
    tracing::info!("Starting server on {}", addr);

    // Run the server (it's already detached if detach was true)
    Server::builder()
        .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
        .serve(addr)
        .await
        .context("Server error")?;

    Ok(())
}