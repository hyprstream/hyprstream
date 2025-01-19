use anyhow::{Result, Context};
use std::sync::Arc;
use daemonize::Daemonize;
use tracing_subscriber::{fmt, EnvFilter};
use tracing_log::LogTracer;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
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
    // Set up logging before anything else
    LogTracer::init().context("Failed to initialize log tracer")?;

    // Set up file appender for logs
    let file_appender = RollingFileAppender::new(
        Rotation::NEVER,
        &settings.server.working_dir,
        "hyprstream.log",
    );
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    // Initialize tracing subscriber with both console and file outputs
    fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::new(&settings.server.log_level))
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_target(false)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .with_level(true)
        .with_target(true)
        .compact()
        .init();

    if detach {
        // Configure daemon
        let daemonize = Daemonize::new()
            .pid_file(&settings.server.pid_file)
            .chown_pid_file(true)
            .working_directory(&settings.server.working_dir);

        // Start daemon
        tracing::info!("Starting server in detached mode");
        tracing::info!("PID file: {}", settings.server.pid_file);
        tracing::info!("Working directory: {}", settings.server.working_dir);
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