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
use std::{
    collections::HashMap,
    fs::File,
    io::BufReader,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    path::Path,
    time::Duration,
};
use tokio::sync::broadcast;
use tonic::transport::{
    Certificate as TonicCertificate, ClientTlsConfig, Identity, Server, ServerTlsConfig,
};
use tracing_log::LogTracer;
use tracing_subscriber::{fmt, EnvFilter};
fn parse_server_addr(settings: &Settings) -> Result<SocketAddr> {
    let port = settings.server.port.unwrap_or(50051);
    
    match settings.server.host.as_deref() {
        None | Some("0.0.0.0") => Ok(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port)),
        Some("::") => Ok(SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port)),
        Some(host) => {
            host.parse::<SocketAddr>()
                .or_else(|_| {
                    // Try parsing as IPv6
                    host.parse::<Ipv6Addr>()
                        .map(|addr| SocketAddr::new(IpAddr::V6(addr), port))
                        .or_else(|_| {
                            // Try parsing as IPv4
                            host.parse::<Ipv4Addr>()
                                .map(|addr| SocketAddr::new(IpAddr::V4(addr), port))
                                .context("Invalid address format")
                        })
                })
        }
    }
}

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
    addr: Option<SocketAddr>,
    query: String,
    tls_cert: Option<&Path>,
    tls_key: Option<&Path>,
    tls_ca: Option<&Path>,
    tls_skip_verify: bool,
    verbose: bool,
) -> Result<()> {
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

    if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
        eprintln!("Warning: Could not set global tracing subscriber: {}", e);
    }

    let _ = rustls::crypto::ring::default_provider().install_default();

    let addr = addr.unwrap_or_else(|| {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 50051)
    });
    tracing::info!("Connecting to {}...", addr);

    let use_tls = tls_cert.is_some() || tls_key.is_some() || tls_ca.is_some();
    let uri_str = format!(
        "{}://{}",
        if use_tls { "https" } else { "http" },
        addr
    );

    if verbose {
        tracing::debug!("Using URL {}", uri_str);
        tracing::debug!("TLS enabled: {}", use_tls);
        tracing::debug!("Connection timeout: 5s");
        tracing::debug!("Query timeout: 30s");
    }

    let mut endpoint =
        tonic::transport::Channel::from_shared(uri_str.clone())?.timeout(Duration::from_secs(5));

    if let Some(tls_config) = configure_client_tls(tls_cert, tls_key, tls_ca, tls_skip_verify)? {
        endpoint = endpoint.tls_config(tls_config)?;
    }

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

    let action = Action {
        r#type: "sql.query".to_string(),
        body: query.clone().into_bytes().into(),
    };

    if verbose {
        tracing::debug!("Executing query: {}", query);
    }
    tracing::info!("Executing query...");

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

    let mut stream = response.into_inner();
    let mut row_count = 0;
    while let Some(result) = stream.message().await? {
        if !result.body.is_empty() {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&result.body) {
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
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
    let _ = rustls::crypto::ring::default_provider().install_default();

    if detach {
        let daemonize = Daemonize::new()
            .pid_file(settings.server.pid_file.as_deref().unwrap_or("/tmp/hyprstream.pid"))
            .chown_pid_file(true);

        daemonize.start().context("Failed to start daemon")?;
    }

    LogTracer::init().context("Failed to initialize log tracer")?;

    let log_level = settings.server.log_level.as_deref().unwrap_or("info");
    let env_filter = EnvFilter::new(log_level);
    
    // Configure stdout/stderr handling
    let (non_blocking_stdout, _guard_stdout) = tracing_appender::non_blocking(std::io::stdout());

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(env_filter)
        .with_writer(non_blocking_stdout)
        .with_ansi(true)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_level(true)
        .compact()
        .finish();

    if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
        eprintln!("Warning: Could not set global tracing subscriber: {}", e);
    }

    let (shutdown_tx, _) = broadcast::channel(1);

    // Set up cross-platform signal handling
    let shutdown_tx_ctrlc = shutdown_tx.clone();
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("Failed to install Ctrl+C handler: {}", e);
            return;
        }
        tracing::info!("Received Ctrl+C signal");
        let _ = shutdown_tx_ctrlc.send(());
    });

    #[cfg(unix)]
    {
        let shutdown_tx_unix = shutdown_tx.clone();
        tokio::spawn(async move {
            let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
            
            if sigterm.recv().await.is_some() {
                tracing::info!("Received SIGTERM signal");
                let _ = shutdown_tx_unix.send(());
            }
        });
    }

    let engine_options: HashMap<String, String> = settings
        .engine
        .engine_options
        .clone()
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

    match &engine_backend {
        StorageBackendType::Adbc(backend) => backend.init().await,
        StorageBackendType::DuckDb(backend) => backend.init().await,
    }
    .context("Failed to initialize storage backend")?;

    let service = FlightSqlService::new(engine_backend);
    let addr = parse_server_addr(&settings)
        .context("Failed to parse server address")?;

    tracing::warn!("This is a pre-release alpha for preview purposes only.");
    tracing::info!("Starting server on {}", addr);

    let mut server = Server::builder();
    if let Some(tls_config) = configure_server_tls(
        settings.server.tls_cert.as_deref(),
        settings.server.tls_key.as_deref(),
        settings.server.tls_client_ca.as_deref(),
    )? {
        server = server.tls_config(tls_config)?;
    }

    let mut shutdown_rx = shutdown_tx.subscribe();
    let server_future = server
        .add_service(service.into_server())
        .serve_with_shutdown(addr, async move {
            let _ = shutdown_rx.recv().await;
            tracing::info!("Initiating graceful shutdown");
        });

    match server_future.await {
        Ok(_) => {
            tracing::info!("Server shutdown completed successfully");
            if detach {
                tracing::info!("Daemon process terminated cleanly");
            }
            Ok(())
        }
        Err(e) => {
            let err_msg = e.to_string();
            tracing::error!("Server error: {}", err_msg);

            if err_msg.contains("Address already in use") {
                anyhow::bail!(
                    "Address {} is already in use. Check if another instance is running",
                    addr
                )
            }

            Err(anyhow::anyhow!(e))
                .with_context(|| format!("Failed to start server on {}", addr))
        }
    }
}
