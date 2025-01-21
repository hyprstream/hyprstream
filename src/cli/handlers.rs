use crate::service::FlightSqlServer;
use crate::storage::StorageBackendType;
use crate::storage::adbc::AdbcBackend;
use crate::storage::duckdb::DuckDbBackend;
use arrow_flight::flight_service_client::FlightServiceClient;
use config::Config;
use prost::Message;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tonic::transport::{Certificate, ClientTlsConfig, Identity, Server};
use std::net::SocketAddr;
use arrow_flight::flight_service_client::FlightServiceClient as FlightSqlClient;
use bytes::Bytes;

pub async fn execute_sql(
    addr: Option<SocketAddr>,
    sql: String,
    tls_cert: Option<&Path>,
    tls_key: Option<&Path>,
    tls_ca: Option<&Path>,
    _tls_skip_verify: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 50051)));
    let endpoint = tonic::transport::Channel::from_shared(format!("http://{}:{}", addr.ip(), addr.port()))?;

    let channel = if let (Some(cert_path), Some(key_path), Some(ca_path)) = (tls_cert, tls_key, tls_ca) {
        let cert = std::fs::read(cert_path)?;
        let key = std::fs::read(key_path)?;
        let ca = std::fs::read(ca_path)?;

        let tls = ClientTlsConfig::new()
            .identity(Identity::from_pem(cert, key))
            .ca_certificate(Certificate::from_pem(ca))
            .domain_name(addr.ip().to_string());

        endpoint.tls_config(tls)?.connect().await?
    } else {
        endpoint.connect().await?
    };

    let mut client = FlightSqlClient::new(channel);
    let action = arrow_flight::Action {
        r#type: "CommandStatementUpdate".to_string(),
        body: Bytes::from(sql.into_bytes()),
    };
    let result = client.do_action(tonic::Request::new(action)).await;

    if verbose {
        println!("SQL execution result: {:?}", result);
    }

    result?;
    Ok(())
}

pub async fn handle_server(config: Config) -> Result<(), Box<dyn std::error::Error>> {
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

    println!("Starting server on {}", addr);
    Server::builder()
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
    tls_skip_verify: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = host.unwrap_or_else(|| "localhost:50051".to_string());
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))?;

    let endpoint = if let (Some(cert_path), Some(key_path)) = (tls_cert, tls_key) {
        let cert = tokio::fs::read(cert_path).await?;
        let key = tokio::fs::read(key_path).await?;
        let identity = Identity::from_pem(cert, key);

        let mut tls = ClientTlsConfig::new().identity(identity);

        if let Some(ca_path) = tls_ca {
            let ca = tokio::fs::read(ca_path).await?;
            tls = tls.ca_certificate(Certificate::from_pem(ca));
        }

        if tls_skip_verify {
            tls = tls.domain_name("localhost");
        }

        endpoint.tls_config(tls)?
    } else {
        endpoint
    };

    let channel = endpoint.connect().await?;
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
        builder = builder.add_source(config::File::from(path));
    }

    let settings = builder.build()?;
    println!("{:#?}", settings);
    Ok(())
}
