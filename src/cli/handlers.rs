use crate::{
    service::FlightSqlServer,
    storage::{StorageBackendType, adbc::AdbcBackend, duckdb::DuckDbBackend},
    config::{get_tls_config, set_tls_data},
};
use arrow_flight::flight_service_client::{FlightServiceClient, FlightServiceClient as FlightSqlClient};
use ::config::{Config, File};
use tonic::transport::{Certificate, ClientTlsConfig, Identity, Server, ServerTlsConfig};
use prost::Message;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    net::SocketAddr,
};
use bytes::Bytes;

pub async fn execute_sql(
    addr: Option<SocketAddr>,
    sql: String,
    config: Option<&Config>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 50051)));
    let scheme = if config.map(|c| c.get_bool("tls.enabled").unwrap_or(false)).unwrap_or(false) { "https" } else { "http" };
    let channel = {
        let mut endpoint = tonic::transport::Channel::from_shared(format!("{}://{}:{}", scheme, addr.ip(), addr.port()))?
            .timeout(std::time::Duration::from_secs(5));

        if let Some(config) = config {
            if let Some((identity, ca_cert)) = get_tls_config(config) {
                let mut tls = ClientTlsConfig::new()
                    .domain_name("localhost")
                    .identity(identity);
                
                if let Some(ca) = ca_cert {
                    tls = tls.ca_certificate(ca);
                }

                endpoint = endpoint.tls_config(tls)?;
            }
        }

        endpoint.connect().await
    }.map_err(|e| {
        let err_str = e.to_string();
        if err_str.contains("transport error") ||
           err_str.contains("deadline has elapsed") ||
           err_str.contains("connection refused") {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "Connection timed out"
            )) as Box<dyn std::error::Error>
        } else {
            Box::new(e) as Box<dyn std::error::Error>
        }
    })?;

    let mut client = FlightSqlClient::new(channel);
    
    // Create JSON command structure
    let json = if sql.trim().to_uppercase().starts_with("CREATE ") ||
                 sql.trim().to_uppercase().starts_with("DROP ") ||
                 sql.trim().to_uppercase().starts_with("ALTER ") {
        serde_json::json!({
            "type": "sql.execute",
            "data": sql
        })
    } else {
        serde_json::json!({
            "type": "sql.query",
            "data": sql
        })
    };

    let action = arrow_flight::Action {
        r#type: "CommandStatementQuery".to_string(),
        body: Bytes::from(serde_json::to_vec(&json)?),
    };

    let result = client.do_action(tonic::Request::new(action)).await;

    if verbose {
        println!("SQL execution result: {:?}", result);
    }

    match result {
        Ok(_) => Ok(()),
        Err(status) => {
            let err_str = status.to_string();
            if err_str.contains("transport error") ||
               err_str.contains("deadline has elapsed") ||
               err_str.contains("connection refused") {
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "Connection timed out"
                )))
            } else {
                Err(Box::new(status))
            }
        }
    }
}

pub async fn handle_server(
    config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}",
        config.get_string("host")?.as_str(),
        config.get_string("port")?.as_str()
    ).parse()?;
    
    let backend = match config.get_string("storage.type")?.as_str() {
        "duckdb" => {
            let conn_str = config.get_string("storage.connection")?.to_string();
            let backend = DuckDbBackend::new(conn_str, HashMap::new(), None)?;
            StorageBackendType::DuckDb(backend)
        }
        "adbc" => {
            let driver = config.get_string("storage.driver")?.to_string();
            let conn_str = config.get_string("storage.connection")?.to_string();
            let backend = AdbcBackend::new(&driver, Some(&conn_str), None)?;
            StorageBackendType::Adbc(backend)
        }
        _ => return Err("Unsupported storage type".into()),
    };

    let service = FlightSqlServer::new(backend).into_service();
    let mut server = Server::builder();

    // Configure TLS if enabled
    if config.get_bool("tls.enabled").unwrap_or(false) {
        let cert = match config.get::<Vec<u8>>("tls.cert_data") {
            Ok(data) => data,
            Err(_) => {
                let path = config.get_string("tls.cert_path")
                    .map_err(|_| "TLS certificate not found")?;
                std::fs::read(path)
                    .map_err(|_| "Failed to read TLS certificate")?
            }
        };
        let key = match config.get::<Vec<u8>>("tls.key_data") {
            Ok(data) => data,
            Err(_) => {
                let path = config.get_string("tls.key_path")
                    .map_err(|_| "TLS key not found")?;
                std::fs::read(path)
                    .map_err(|_| "Failed to read TLS key")?
            }
        };
        let identity = Identity::from_pem(&cert, &key);

        let mut tls_config = ServerTlsConfig::new().identity(identity);

        if let Some(ca) = config.get::<Vec<u8>>("tls.ca_data").ok()
            .or_else(|| config.get_string("tls.ca_path").ok()
                .and_then(|p| if p.is_empty() { None } else { Some(p) })
                .and_then(|p| std::fs::read(p).ok())) {
            tls_config = tls_config.client_ca_root(Certificate::from_pem(&ca));
        }

        server = server.tls_config(tls_config)?;
    }

    println!("Starting server on {}", addr);
    server
        .add_service(service)
        .serve(addr)
        .await?;

    Ok(())
}

pub async fn handle_sql(
    host: Option<String>,
    query: &str,
    tls_cert: Option<&Path>,
    tls_key: Option<&Path>,
    tls_ca: Option<&Path>,
    _tls_skip_verify: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = host.unwrap_or_else(|| "localhost:50051".to_string());

    // Create Config with TLS settings if certificates are provided
    let config = match (tls_cert, tls_key) {
        (Some(cert_path), Some(key_path)) => {
            let cert = tokio::fs::read(cert_path).await?;
            let key = tokio::fs::read(key_path).await?;
            let ca = if let Some(ca_path) = tls_ca {
                Some(tokio::fs::read(ca_path).await?)
            } else {
                None
            };

            let config = set_tls_data(
                Config::builder(),
                &cert,
                &key,
                ca.as_deref(),
            )?
            .build()?;

            Some(config)
        }
        _ => None,
    };

    // Parse address and execute SQL
    let addr_parts: Vec<&str> = addr.split(':').collect();
    if addr_parts.len() != 2 {
        return Err("Invalid address format. Expected host:port".into());
    }

    let socket_addr = SocketAddr::new(
        addr_parts[0].parse()?,
        addr_parts[1].parse()?
    );

    execute_sql(
        Some(socket_addr),
        query.to_string(),
        config.as_ref(),
        verbose,
    ).await?;

    let channel = {
        let addr_parts: Vec<&str> = addr.split(':').collect();
        if addr_parts.len() != 2 {
            return Err("Invalid address format. Expected host:port".into());
        }

        let scheme = if config.as_ref().map(|c| c.get_bool("tls.enabled").unwrap_or(false)).unwrap_or(false) {
            "https"
        } else {
            "http"
        };

        let mut endpoint = tonic::transport::Channel::from_shared(format!("{}://{}:{}", scheme, addr_parts[0], addr_parts[1]))?
            .timeout(std::time::Duration::from_secs(5));

        if let Some(config) = config {
            if let Some((identity, ca_cert)) = get_tls_config(&config) {
                let mut tls = ClientTlsConfig::new()
                    .domain_name("localhost")
                    .identity(identity);
                
                if let Some(ca) = ca_cert {
                    tls = tls.ca_certificate(ca);
                }

                endpoint = endpoint.tls_config(tls)?;
            }
        }

        endpoint.connect().await?
    };
    let mut client = FlightServiceClient::new(channel);
    
    let command = arrow_flight::sql::CommandStatementQuery {
        query: query.to_string(),
        transaction_id: None,
    };
    
    let mut buf = Vec::new();
    command.encode(&mut buf)?;
    let request = arrow_flight::Ticket {
        ticket: buf.into(),
    };
    
    let mut stream = client.do_get(request).await?.into_inner();
    
    while let Some(flight_data) = stream.message().await? {
        if verbose {
            println!("Received data: {:?}", flight_data);
        }
    }

    Ok(())
}

pub fn handle_config(config_path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = Config::builder()
        .set_default("host", "127.0.0.1")?
        .set_default("port", "50051")?
        .set_default("storage.type", "duckdb")?
        .set_default("storage.connection", ":memory:")?;

    // Load config file if provided
    if let Some(path) = config_path {
        builder = builder.add_source(File::from(path));
    }

    let settings = builder.build()?;
    println!("{:#?}", settings);
    Ok(())
}
