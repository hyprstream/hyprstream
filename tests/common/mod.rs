use hyprstream_core::{
    service::FlightSqlServer,
    storage::{
        duckdb::DuckDbBackend,
        StorageBackendType,
    },
};
use std::collections::HashMap;
use std::error::Error;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use tokio::net::TcpListener;
use tonic::transport::Server;

pub struct TestServer {
    pub handle: tokio::task::JoinHandle<()>,
    pub addr: std::net::SocketAddr,
}

pub async fn start_test_server(use_cache: bool) -> TestServer {
    // Create the backend
    let backend = if use_cache {
        let mut options = HashMap::new();
        options.insert("engine".to_string(), "cached".to_string());
        options.insert("store_engine".to_string(), "duckdb".to_string());
        options.insert("max_duration_secs".to_string(), "3600".to_string());
        
        StorageBackendType::new_with_options(":memory:", &options, None).unwrap()
    } else {
        StorageBackendType::DuckDb(DuckDbBackend::new_in_memory().unwrap())
    };

    let service = FlightSqlServer::new(backend);

    // Create and bind to a TCP listener
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);
    let listener = TcpListener::bind(addr).await.unwrap();
    let bound_addr = listener.local_addr().unwrap();

    // Create channels for server readiness and bound port
    let (tx, rx) = tokio::sync::oneshot::channel();
    
    println!("Starting test server on {}", bound_addr);

    // Spawn the server task
    let server_handle = tokio::spawn(async move {
        // Run the server in the background
        let server = Server::builder()
            .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

        // Signal that we're ready to accept connections
        let _ = tx.send(());

        if let Err(e) = server.await {
            println!("Server error: {}", e);
            if let Some(source) = e.source() {
                println!("Error source: {:?}", source);
            }
        }
    });

    // Wait for the server to be ready
    rx.await.expect("Server startup signal not received");
    
    // Give the server a moment to fully initialize
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    TestServer {
        handle: server_handle,
        addr: bound_addr,
    }
}

pub fn get_test_endpoint(addr: SocketAddr) -> String {
    format!("http://{}:{}", addr.ip(), addr.port())
}