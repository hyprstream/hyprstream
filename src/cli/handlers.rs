use crate::{
    cli::commands::config::Credentials,
    config::Settings,
    service::FlightSqlService,
    storage::{adbc::AdbcBackend, duckdb::DuckDbBackend, StorageBackend, StorageBackendType},
};
use anyhow::{Context, Result};
use arrow_flight::{flight_service_client::FlightServiceClient, Action};
use daemonize::Daemonize;
use rustls::{
    pki_types::{CertificateDer, PrivateKeyDer},
    RootCertStore,
};
use std::{collections::HashMap, fs::File, io::BufReader, path::Path, time::Duration};
use tonic::transport::{
    Certificate as TonicCertificate, ClientTlsConfig, Identity, Server, ServerTlsConfig,
};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_log::LogTracer;
use tracing_subscriber::{fmt, EnvFilter};

fn load_certificate(path: &Path) -> Result<Vec<CertificateDer<'static>>> {
    let file = File::open(path).context("Failed to open certificate file")?;
    let mut reader = BufReader::new(file);
    let certs = rustls_pemfile::certs(&mut reader)
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("Failed to parse certificate")?;
    Ok(certs)
}

fn load_private_key(path: &Path) -> Result<PrivateKeyDer<'static>> {
    let file = File::open(path).context("Failed to open private key file")?;
    let mut reader = BufReader::new(file);
    let key = rustls_pemfile::pkcs8_private_keys(&mut reader)
        .next()
        .ok_or_else(|| anyhow::anyhow!("No private key found"))?
        .context("Failed to parse private key")?;
    Ok(PrivateKeyDer::Pkcs8(key))
}

fn configure_server_tls(
    cert_path: Option<&Path>,
    key_path: Option<&Path>,
    client_ca_path: Option<&Path>,
) -> Result<Option<ServerTlsConfig>> {
    match (cert_path, key_path) {
        (Some(cert_path), Some(key_path)) => {
            let mut config = ServerTlsConfig::new();
            config = config.identity(Identity::from_pem(
                std::fs::read(cert_path)?,
                std::fs::read(key_path)?,
            ));

            // Configure client authentication if CA cert is provided
            if let Some(ca_path) = client_ca_path {
                let mut root_store = RootCertStore::empty();
                for cert in load_certificate(ca_path)? {
                    root_store
                        .add(cert)
                        .context("Failed to add CA certificate")?;
                }
                config = config.client_ca_root(TonicCertificate::from_pem(std::fs::read(ca_path)?));
            }

            Ok(Some(config))
        }
        (None, None) => Ok(None),
        _ => anyhow::bail!("Both certificate and private key must be provided for TLS"),
    }
}

fn configure_client_tls(
    cert_path: Option<&Path>,
    key_path: Option<&Path>,
    ca_path: Option<&Path>,
    skip_verify: bool,
) -> Result<Option<ClientTlsConfig>> {
    let mut config = if skip_verify {
        ClientTlsConfig::new().domain_name("localhost")
    } else {
        ClientTlsConfig::new()
    };

    if let Some(ca_path) = ca_path {
        config = config.ca_certificate(TonicCertificate::from_pem(std::fs::read(ca_path)?));
    }

    match (cert_path, key_path) {
        (Some(cert_path), Some(key_path)) => {
            config = config.identity(Identity::from_pem(
                std::fs::read(cert_path)?,
                std::fs::read(key_path)?,
            ));
        }
        (None, None) => {}
        _ => anyhow::bail!("Both certificate and private key must be provided for mTLS"),
    }

    Ok(Some(config))
}

pub async fn execute_sql(
    host: Option<String>,
    query: String,
    tls_cert: Option<&Path>,
    tls_key: Option<&Path>,
    tls_ca: Option<&Path>,
    tls_skip_verify: bool,
    verbose: bool,
) -> Result<()> {
    // Set up logging
    let level = if verbose { "debug" } else { "info" };
    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::new(level))
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_level(true)
        .compact()
        .finish();

    // Try to set the subscriber, but don't panic if it fails (e.g., if already set)
    if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
        eprintln!("Warning: Could not set global tracing subscriber: {}", e);
    }

    // Install the default crypto provider (ignore result as it may already be installed)
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Use default host:port if not provided
    let addr = host.unwrap_or_else(|| "127.0.0.1:50051".to_string());
    tracing::info!("Connecting to {}...", addr);

    // Ensure URL has correct scheme based on TLS configuration
    let use_tls = tls_cert.is_some() || tls_key.is_some() || tls_ca.is_some();
    let uri_str = if !addr.starts_with("http://") && !addr.starts_with("https://") {
        format!(
            "{}://{}",
            if use_tls { "https" } else { "http" },
            addr.clone()
        )
    } else {
        addr.clone()
    };

    if verbose {
        tracing::debug!("Using URL {}", uri_str);
        tracing::debug!("TLS enabled: {}", use_tls);
        tracing::debug!("Connection timeout: 5s");
        tracing::debug!("Query timeout: 30s");
    }

    tracing::info!("Connecting to {}...", addr);

    // Configure TLS if enabled
    let mut endpoint =
        tonic::transport::Channel::from_shared(uri_str.clone())?.timeout(Duration::from_secs(5));

    if let Some(tls_config) = configure_client_tls(tls_cert, tls_key, tls_ca, tls_skip_verify)? {
        endpoint = endpoint.tls_config(tls_config)?;
    }

    // Connect to the server with timeout
    let channel = match tokio::time::timeout(Duration::from_secs(5), endpoint.connect()).await {
        Ok(Ok(channel)) => channel,
        Ok(Err(e)) => {
            return Err(anyhow::anyhow!(
                "Failed to connect: {}\nPossible reasons:\n- Server is not running (try 'hyprstream server')\n- Wrong host or port\n- Firewall blocking connection",
                e
            ));
        }
        Err(_) => {
            return Err(anyhow::anyhow!(
                "Connection timed out after 5 seconds\nPossible reasons:\n- Server is not responding\n- Network issues\n- Firewall blocking connection"
            ));
        }
    };

    let mut client = FlightServiceClient::new(channel);

    // Create the SQL action
    let action = Action {
        r#type: "sql.query".to_string(),
        body: query.clone().into_bytes().into(),
    };

    if verbose {
        tracing::debug!("Executing query: {}", query);
    }
    tracing::info!("Executing query...");

    // Execute the query with timeout
    let response = match tokio::time::timeout(
        Duration::from_secs(30),
        client.do_action(tonic::Request::new(action)),
    )
    .await
    {
        Ok(Ok(response)) => response,
        Ok(Err(e)) => {
            return Err(anyhow::anyhow!(
                "Query failed: {}\nPossible reasons:\n- SQL syntax error\n- Table does not exist\n- Insufficient permissions",
                e
            ));
        }
        Err(_) => {
            return Err(anyhow::anyhow!(
                "Query timed out after 30 seconds\nThe query might be too complex or the server might be overloaded"
            ));
        }
    };

    // Stream and print results
    let mut stream = response.into_inner();
    let mut row_count = 0;
    while let Some(result) = stream.message().await? {
        if !result.body.is_empty() {
            // Try to parse as JSON for better formatting
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&result.body) {
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                // Fallback to string output
                println!("{}", String::from_utf8_lossy(&result.body));
            }
            row_count += 1;
        }
    }

    if row_count == 0 {
        tracing::info!("Query completed successfully (0 rows)");
    } else {
        tracing::info!("Query completed successfully ({} rows)", row_count);
    }

    Ok(())
}

pub async fn run_server(detach: bool, settings: Settings) -> Result<()> {
    // Install the default crypto provider (ignore result as it may already be installed)
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Set up signal handling
    #[cfg(unix)]
    fn setup_signals() -> Result<()> {
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        
        tokio::spawn({
            async move {
                let sigterm = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                    Ok(signal) => signal,
                    Err(e) => {
                        tracing::error!("Failed to create SIGTERM handler: {}", e);
                        if let Err(e) = shutdown_tx.send(()) {
                            tracing::error!("Additionally failed to send shutdown signal: {:?}", e);
                        }
                        return;
                    }
                };

                let sigint = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
                    Ok(signal) => signal,
                    Err(e) => {
                        tracing::error!("Failed to create SIGINT handler: {}", e);
                        if let Err(e) = shutdown_tx.send(()) {
                            tracing::error!("Additionally failed to send shutdown signal: {:?}", e);
                        }
                        return;
                    }
                };

                let (mut sigterm, mut sigint) = (sigterm, sigint);

                tokio::select! {
                    _ = sigterm.recv() => {
                        tracing::info!("Received SIGTERM signal, initiating graceful shutdown");
                    }
                    _ = sigint.recv() => {
                        tracing::info!("Received SIGINT signal, initiating graceful shutdown");
                    }
                }

                if let Err(e) = shutdown_tx.send(()) {
                    tracing::error!("Failed to send shutdown signal: {:?}", e);
                    std::process::exit(1);
                }
            }
        });

        Ok(())
    }

    setup_signals()?;

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
        
        // Create log directory if it doesn't exist
        let log_dir = settings
            .server
            .working_dir
            .as_deref()
            .unwrap_or("/tmp")
            .to_string();
        std::fs::create_dir_all(&log_dir).context("Failed to create log directory")?;

        // Create file descriptors for stdout/stderr with proper paths
        let stdout_path = std::path::Path::new(&log_dir).join("hyprstream.out");
        let stderr_path = std::path::Path::new(&log_dir).join("hyprstream.err");

        let stdout = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&stdout_path)
            .with_context(|| format!("Failed to create stdout file at {:?}", stdout_path))?;

        let stderr = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&stderr_path)
            .with_context(|| format!("Failed to create stderr file at {:?}", stderr_path))?;

        tracing::info!("Configured daemon output:");
        tracing::info!("  stdout: {:?}", stdout_path);
        tracing::info!("  stderr: {:?}", stderr_path);
        
        // Configure daemon with proper file descriptors
        daemonize.stdout(stdout).stderr(stderr).start().context("Failed to start daemon")?;
        
        // Logging configuration persists after daemonization
        tracing::info!("Successfully daemonized server process");
    }

    // Convert engine options from Vec<String> to HashMap
    let engine_options: HashMap<String, String> = settings
        .engine
        .engine_options
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

    // Configure server with TLS if certificates are provided
    let mut server = Server::builder();
    if let Some(tls_config) = configure_server_tls(
        settings.server.tls_cert.as_deref(),
        settings.server.tls_key.as_deref(),
        settings.server.tls_client_ca.as_deref(),
    )? {
        server = server.tls_config(tls_config)?;
    }

    // Run the server with enhanced error handling and graceful shutdown
    tracing::info!("Binding server to {}", addr);
    
    let server_future = server
        .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
        .serve_with_shutdown(addr, async {
            let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
            let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                .expect("failed to install SIGINT handler");

            tokio::select! {
                _ = sigterm.recv() => {
                    tracing::info!("Received SIGTERM signal, initiating graceful shutdown");
                }
                _ = sigint.recv() => {
                    tracing::info!("Received SIGINT signal, initiating graceful shutdown");
                }
            }
        });

    // Add timeout for initial binding
    match tokio::time::timeout(std::time::Duration::from_secs(5), server_future).await {
        Ok(result) => match result {
            Ok(_) => {
                tracing::info!("Server shutdown completed successfully");
                if detach {
                    tracing::info!("Daemon process terminated cleanly");
                }
                Ok(())
            }
            Err(e) => {
                let err_msg = e.to_string();
                let host = settings.server.host.as_deref().unwrap_or("127.0.0.1");
                let port = settings.server.port.unwrap_or(50051);

                // Log detailed error information
                tracing::error!("Server error details:");
                tracing::error!("  Message: {}", err_msg);
                tracing::error!("  Binding: {}:{}", host, port);
                if detach {
                    tracing::error!("  Mode: Daemon process");
                }

                if err_msg.contains("Address already in use") {
                    anyhow::bail!(
                        "Port {} is already in use on {}. Check if another instance is running or try a different port",
                        port,
                        host
                    )
                }

                Err(anyhow::anyhow!(e))
                    .with_context(|| format!("Failed to start server on {}:{}", host, port))
            }
        },
        Err(_) => {
            let port = settings.server.port.unwrap_or(50051);
            let host = settings.server.host.as_deref().unwrap_or("127.0.0.1");
            
            tracing::error!(
                "Server failed to bind to {}:{} within timeout period",
                host,
                port
            );
            
            anyhow::bail!(
                "Server failed to bind within 5 seconds.\nPossible issues:\n\
                 - Port {} is blocked or requires elevated permissions\n\
                 - Network interface {} is not available\n\
                 - System resources are constrained\n\
                 Try using a different port/host or check system permissions",
                port,
                host
            )
        }
    }
}
