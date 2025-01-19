use crate::{
    cli::commands::config::Credentials,
    config::Settings,
    service::FlightSqlService,
    storage::{adbc::AdbcBackend, duckdb::DuckDbBackend, StorageBackend, StorageBackendType},
};
use anyhow::{Context, Result};
use arrow_flight::flight_service_client::FlightServiceClient;
use daemonize::Daemonize;
use std::collections::HashMap;
use tonic::transport::{Server, Uri};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_log::LogTracer;
use tracing_subscriber::{fmt, EnvFilter};
use arrow_flight::Action;

pub async fn execute_sql(host: Option<String>, query: String) -> Result<()> {
    // Use default host:port if not provided
    let addr = host.unwrap_or_else(|| "127.0.0.1:50051".to_string());
    // Ensure URL has http:// scheme
    let uri_str = if !addr.starts_with("http://") && !addr.starts_with("https://") {
        format!("https://{}", addr)
    } else {
        addr
    };
    let uri: Uri = uri_str.parse().context("Invalid server address")?;

    // Connect to the server
    let mut client = FlightServiceClient::connect(uri)
        .await
        .context("Failed to connect to server")?;

    // Create the SQL action
    let action = Action {
        r#type: "sql.query".to_string(),
        body: query.into_bytes().into(),
    };

    // Execute the query
    let response = client
        .do_action(tonic::Request::new(action))
        .await
        .context("Failed to execute query")?;

    // Stream and print results
    let mut stream = response.into_inner();
    while let Some(result) = stream.message().await? {
        if !result.body.is_empty() {
            // Try to parse as JSON for better formatting
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&result.body) {
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                // Fallback to string output
                println!("{}", String::from_utf8_lossy(&result.body));
            }
        }
    }

    Ok(())
}

pub async fn run_server(detach: bool, settings: Settings) -> Result<()> {
    // Set up logging before anything else
    LogTracer::init().context("Failed to initialize log tracer")?;

    // Set up file appender for logs
    let file_appender = RollingFileAppender::new(
        Rotation::NEVER,
        settings.server.working_dir.as_deref().unwrap_or("/tmp"),
        "hyprstream.log",
    );
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    // Initialize tracing subscriber with both console and file outputs
    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::new(
            settings.server.log_level.as_deref().unwrap_or("info"),
        ))
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
        .finish();

    // Try to set the subscriber, but don't panic if it fails (e.g., if already set)
    if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
        eprintln!("Warning: Could not set global tracing subscriber: {}", e);
    }

    if detach {
        // Configure daemon
        let daemonize = Daemonize::new()
            .pid_file(
                settings
                    .server
                    .pid_file
                    .as_deref()
                    .unwrap_or("/tmp/hyprstream.pid"),
            )
            .chown_pid_file(true)
            .working_directory(settings.server.working_dir.as_deref().unwrap_or("/tmp"));

        // Start daemon
        tracing::info!("Starting server in detached mode");
        tracing::info!("PID file: {:?}", settings.server.pid_file);
        tracing::info!("Working directory: {:?}", settings.server.working_dir);
        daemonize.start().context("Failed to start daemon")?;
    }

    // Convert engine options from Vec<String> to HashMap
    let engine_options: HashMap<String, String> = settings
        .engine
        .engine_options
        .as_ref()
        .map(|opts| {
            opts.iter()
                .filter_map(|opt| {
                    let parts: Vec<&str> = opt.split('=').collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    // Create credentials if username and password are provided
    let engine_credentials = if let (Some(username), Some(password)) = (
        settings.engine.engine_username.as_ref(),
        settings.engine.engine_password.as_ref(),
    ) {
        Some(Credentials {
            username: username.clone(),
            password: password.clone(),
        })
    } else {
        None
    };

    // Create storage backend
    let engine_backend = match settings.engine.engine_type.as_deref().unwrap_or("duckdb") {
        "adbc" => StorageBackendType::Adbc(
            AdbcBackend::new_with_options(
                settings
                    .engine
                    .engine_connection
                    .as_deref()
                    .unwrap_or(":memory:"),
                &engine_options,
                engine_credentials.as_ref(),
            )
            .context("Failed to create ADBC backend")?,
        ),
        "duckdb" => StorageBackendType::DuckDb(
            DuckDbBackend::new_with_options(
                settings
                    .engine
                    .engine_connection
                    .as_deref()
                    .unwrap_or(":memory:"),
                &engine_options,
                engine_credentials.as_ref(),
            )
            .context("Failed to create DuckDB backend")?,
        ),
        engine_type => anyhow::bail!("Unsupported engine type: {}", engine_type),
    };

    // Initialize backend
    match &engine_backend {
        StorageBackendType::Adbc(backend) => backend.init().await,
        StorageBackendType::DuckDb(backend) => backend.init().await,
    }
    .context("Failed to initialize storage backend")?;

    // Create the service
    let service = FlightSqlService::new(engine_backend);

    // Start the server
    let addr = format!(
        "{}:{}",
        settings.server.host.as_deref().unwrap_or("127.0.0.1"),
        settings.server.port.unwrap_or(50051)
    )
    .parse()
    .context("Invalid listen address")?;

    tracing::warn!("This is a pre-release alpha for preview purposes only.");
    tracing::info!("Starting server on {}", addr);

    // Run the server (it's already detached if detach was true)
    match Server::builder()
        .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
        .serve(addr)
        .await
    {
        Ok(_) => Ok(()),
        Err(e) => {
            if e.to_string().contains("Address already in use") {
                anyhow::bail!(
                    "Port {} is already in use. Try using a different port with --port",
                    settings.server.port.unwrap_or(50051)
                );
            } else {
                Err(e).context("Server error")?
            }
        }
    }
}
