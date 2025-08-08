//! Common test utilities for VDB-first adaptive ML inference server

use hyprstream_core::{
    service::EmbeddingFlightService,
    storage::{VDBSparseStorage, SparseStorageConfig},
};
use std::error::Error;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use tokio::net::TcpListener;
use tonic::transport::Server;

pub struct TestServer {
    pub handle: tokio::task::JoinHandle<()>,
    pub addr: std::net::SocketAddr,
}

pub async fn start_test_vdb_server() -> Result<TestServer, Box<dyn Error>> {
    // Find available port
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    drop(listener);

    // Initialize VDB sparse storage for testing
    let storage_config = SparseStorageConfig {
        storage_path: tempfile::tempdir()?.path().to_path_buf(),
        neural_compression: false, // Disable for tests
        hardware_acceleration: false, // Disable for tests
        cache_size_mb: 128, // Small cache for tests
        compaction_interval_secs: 3600,
        streaming_updates: true,
        update_batch_size: 100,
    };

    let sparse_storage = Arc::new(VDBSparseStorage::new(storage_config).await?);

    // Create embedding-focused FlightSQL service
    let flight_service = hyprstream_core::service::embedding_flight::create_embedding_flight_server(sparse_storage);

    let handle = tokio::spawn(async move {
        if let Err(e) = Server::builder()
            .add_service(flight_service)
            .serve(addr)
            .await
        {
            eprintln!("Test server failed: {}", e);
        }
    });

    // Give server time to start
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    Ok(TestServer { handle, addr })
}

/// Create a test VDB sparse storage instance
pub async fn create_test_vdb_storage() -> Result<Arc<VDBSparseStorage>, Box<dyn Error>> {
    let storage_config = SparseStorageConfig {
        storage_path: tempfile::tempdir()?.path().to_path_buf(),
        neural_compression: false,
        hardware_acceleration: false,
        cache_size_mb: 64,
        compaction_interval_secs: 3600,
        streaming_updates: true,
        update_batch_size: 50,
    };

    let sparse_storage = Arc::new(VDBSparseStorage::new(storage_config).await?);
    Ok(sparse_storage)
}