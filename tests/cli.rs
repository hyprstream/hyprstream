use hyprstream_core::{
    cli::{commands::sql::SqlCommand, handlers::execute_sql},
    config::{self, set_tls_data, get_tls_config},
    service::FlightSqlServer,
    storage::{duckdb::DuckDbBackend, StorageBackendType},
};
use ::config::Config;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::error::Error;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tonic::transport::Server;

use std::fs;
use std::path::PathBuf;

struct TestTlsPaths {
    cert_path: PathBuf,
    key_path: PathBuf,
    ca_path: PathBuf,
}

fn get_test_tls_paths() -> TestTlsPaths {
    TestTlsPaths {
        cert_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config").join("test.crt"),
        key_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config").join("test.key"),
        ca_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config").join("test.crt"),
    }
}

async fn start_test_server() -> (tokio::task::JoinHandle<()>, std::net::SocketAddr) {
    // Install the default crypto provider
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Start a test server
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Create a test database
    let backend = StorageBackendType::DuckDb(DuckDbBackend::new_in_memory().unwrap());
    let service = FlightSqlServer::new(backend);

    // Run the server in the background
    let server_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    (server_handle, addr)
}

async fn start_tls_test_server() -> (tokio::task::JoinHandle<()>, std::net::SocketAddr) {
    // Install the default crypto provider
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Start a test server with TLS
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Create a test database
    let backend = StorageBackendType::DuckDb(DuckDbBackend::new_in_memory().unwrap());
    let service = FlightSqlServer::new(backend);

    // Create config with TLS settings
    let tls_paths = get_test_tls_paths();
    let config = Config::builder()
        .set_override("tls.enabled", true).unwrap()
        .set_override("tls.cert_path", tls_paths.cert_path.to_string_lossy().to_string()).unwrap()
        .set_override("tls.key_path", tls_paths.key_path.to_string_lossy().to_string()).unwrap()
        .set_override("tls.ca_path", tls_paths.ca_path.to_string_lossy().to_string()).unwrap()
        .build()
        .unwrap();

    // Get TLS config from Config
    let (identity, ca_cert) = get_tls_config(&config).unwrap();

    // Run the server in the background with TLS
    let server_handle = tokio::spawn(async move {
        println!("Starting TLS server...");
        println!("Setting up server TLS config...");
        
        // Create server TLS config with more permissive settings
        let mut tls_config = tonic::transport::ServerTlsConfig::new()
            .identity(identity)
            .client_auth_optional(true);  // Allow both TLS auth and non-auth clients

        // Add CA cert if present
        if let Some(ca) = ca_cert {
            println!("Adding CA certificate to server TLS config");
            tls_config = tls_config.client_ca_root(ca);
        }

        println!("Created server TLS config with client auth optional");
        println!("Waiting for TLS connections...");

        let server = Server::builder()
            .tls_config(tls_config)
            .unwrap()
            .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

        println!("Server built with TLS config, starting to serve...");
        
        match server.await {
            Ok(_) => println!("Server finished successfully"),
            Err(e) => {
                println!("Server error: {}", e);
                if let Some(source) = e.source() {
                    println!("Error source: {:?}", source);
                }
            }
        }
    });

    // Give the server more time to start and initialize TLS
    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

    // Print server address and status for debugging
    println!("TLS test server started on {}", addr);
    println!("Waiting for server to be ready...");

    // Try to connect to verify server is listening
    let socket = tokio::net::TcpStream::connect(addr).await;
    match socket {
        Ok(_) => println!("Server is accepting connections"),
        Err(e) => println!("Server connection test failed: {}", e),
    }

    (server_handle, addr)
}

#[tokio::test]
async fn test_sql_command_basic() {
    let (server_handle, addr) = start_test_server().await;

    // Test basic SQL query
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), addr.port())),
        "CREATE TABLE test (id INTEGER);".to_string(),
        None,
        false,
    )
    .await;
    assert!(result.is_ok(), "Basic SQL query failed: {:?}", result.err());

    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_sql_command_tls() {
    // Start TLS server
    let (server_handle, addr) = start_tls_test_server().await;

    // Create config with TLS settings using paths
    let tls_paths = get_test_tls_paths();
    let config = Config::builder()
        .set_override("tls.enabled", true).unwrap()
        .set_override("tls.cert_path", tls_paths.cert_path.to_string_lossy().to_string()).unwrap()
        .set_override("tls.key_path", tls_paths.key_path.to_string_lossy().to_string()).unwrap()
        .set_override("tls.ca_path", tls_paths.ca_path.to_string_lossy().to_string()).unwrap()
        .build()
        .unwrap();

    // Print TLS configuration for debugging
    println!("Testing TLS connection to {}", addr);
    println!("TLS enabled: {}", config.get_bool("tls.enabled").unwrap_or(false));
    println!("Certificate path: {}", tls_paths.cert_path.display());
    println!("Key path: {}", tls_paths.key_path.display());

    // Test TLS connection
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), addr.port())),
        "CREATE TABLE test (id INTEGER);".to_string(),
        Some(&config),
        true, // Enable verbose mode for more debug output
    )
    .await;
    assert!(result.is_ok(), "TLS connection failed: {:?}", result.err());

    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_sql_command_timeout() {
    // Start test server
    let (server_handle, addr) = start_test_server().await;

    // Test connection timeout (wrong port)
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), addr.port() + 1)), // Wrong port
        "SELECT 1;".to_string(),
        None,
        false,
    )
    .await;
    assert!(result.is_err(), "Expected timeout error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("timed out") ||
        err.contains("connection refused") ||
        err.contains("Timeout expired") ||
        err.contains("status: Cancelled message: \"Timeout expired\""),
        "Expected timeout error but got: {}",
        err
    );

    // Test query timeout
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), addr.port())),
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t WHERE n < 1000000) SELECT COUNT(*) FROM t;".to_string(), // Query that will take too long
        None,
        false,
    )
    .await;
    assert!(result.is_err(), "Expected query timeout");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("timed out"),
        "Expected timeout error but got: {}",
        err
    );

    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_sql_command_args() {
    // Create temporary files with test certificates
    // let temp_dir = TempDir::new().unwrap();
    let tls_paths = get_test_tls_paths();
    // Test command line argument parsing
    let cmd = SqlCommand {
        host: Some("localhost:8080".to_string()),
        query: "SELECT 1".to_string(),
        tls_cert: Some(tls_paths.cert_path.clone()),
        tls_key: Some(tls_paths.key_path.clone()),
        tls_ca: Some(tls_paths.ca_path.clone()),
        tls_skip_verify: false,
        verbose: true,
        help: None,
    };

    // Verify command line arguments
    assert_eq!(cmd.host.as_deref(), Some("localhost:8080"));
    assert_eq!(cmd.query, "SELECT 1");
    assert!(cmd.tls_cert.is_some());
    assert!(cmd.tls_key.is_some());
    assert!(cmd.tls_ca.is_some());
    assert!(!cmd.tls_skip_verify);
    assert!(cmd.verbose);

    // Test that we can create a Config with TLS settings from command args
    let config = Config::builder()
        .set_override("tls.enabled", true).unwrap()
        .set_override("tls.cert_path", tls_paths.cert_path.to_string_lossy().to_string()).unwrap()
        .set_override("tls.key_path", tls_paths.key_path.to_string_lossy().to_string()).unwrap()
        .set_override("tls.ca_path", tls_paths.ca_path.to_string_lossy().to_string()).unwrap()
        .build()
        .unwrap();

    // Verify we can get TLS config from the Config
    let (identity, ca_cert) = get_tls_config(&config).unwrap();
    assert!(ca_cert.is_some());

    // Test that we can use the Config with execute_sql
    let result = execute_sql(
        Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080)),
        cmd.query.clone(),
        Some(&config),
        cmd.verbose,
    ).await;

    // The connection will fail (no server) but we just want to verify the Config works
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Connection timed out"));
}
