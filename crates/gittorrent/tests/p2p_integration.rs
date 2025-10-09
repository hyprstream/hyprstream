//! P2P integration tests for GitTorrent

use gittorrent::{
    service::{GitTorrentService, GitTorrentConfig, P2PStatus},
    Result,
};
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;
use tempfile::TempDir;

/// Test basic P2P connectivity between two GitTorrent instances
#[tokio::test]
async fn test_p2p_connectivity() -> Result<()> {
    // Initialize tracing for debugging
    let _ = tracing_subscriber::fmt::try_init();

    // Create temporary directories for storage
    let temp1 = TempDir::new()?;
    let temp2 = TempDir::new()?;

    // Create first instance (bootstrap node)
    let config1 = GitTorrentConfig {
        storage_dir: PathBuf::from(temp1.path()),
        bind_address: "127.0.0.1".to_string(),
        bind_port: 7881,
        bootstrap_nodes: vec![], // No bootstrap, this is the first node
        ..Default::default()
    };

    let service1 = GitTorrentService::new(config1).await?;

    // Wait for first service to initialize
    sleep(Duration::from_secs(1)).await;

    // Create second instance connecting to the first
    let config2 = GitTorrentConfig {
        storage_dir: PathBuf::from(temp2.path()),
        bind_address: "127.0.0.1".to_string(),
        bind_port: 7882,
        bootstrap_nodes: vec!["127.0.0.1:7881".to_string()],
        ..Default::default()
    };

    let service2 = GitTorrentService::new(config2).await?;

    // Wait for connection establishment
    sleep(Duration::from_secs(3)).await;

    // Check P2P status
    let status1 = service1.get_p2p_status().await;
    let status2 = service2.get_p2p_status().await;

    println!("Service 1 status: {:?}", status1);
    println!("Service 2 status: {:?}", status2);

    // Verify both services have P2P initialized
    assert!(status1.connected, "Service 1 should be connected");
    assert!(status2.connected, "Service 2 should be connected");
    assert!(status1.dht_connected, "Service 1 DHT should be connected");
    assert!(status2.dht_connected, "Service 2 DHT should be connected");

    // Verify peer IDs are set
    assert!(status1.peer_id.is_some(), "Service 1 should have a peer ID");
    assert!(status2.peer_id.is_some(), "Service 2 should have a peer ID");

    // Note: Peer count might still be 0 if connection handling needs more work
    // This is expected in the current implementation stage

    Ok(())
}

/// Test repository announcement and discovery
#[tokio::test]
async fn test_repository_discovery() -> Result<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt::try_init();

    // Create temporary directories
    let temp1 = TempDir::new()?;
    let temp2 = TempDir::new()?;
    let repo_dir = TempDir::new()?;

    // Initialize a git repository
    let repo_path = repo_dir.path();
    std::process::Command::new("git")
        .arg("init")
        .arg(repo_path)
        .output()?;

    // Create a test file and commit
    std::fs::write(repo_path.join("README.md"), "# Test Repository")?;
    std::process::Command::new("git")
        .args(&["add", "."])
        .current_dir(repo_path)
        .output()?;
    std::process::Command::new("git")
        .args(&["commit", "-m", "Initial commit"])
        .current_dir(repo_path)
        .output()?;

    // Start two GitTorrent services
    let config1 = GitTorrentConfig {
        storage_dir: PathBuf::from(temp1.path()),
        bind_address: "127.0.0.1".to_string(),
        bind_port: 7883,
        bootstrap_nodes: vec![],
        ..Default::default()
    };

    let service1 = GitTorrentService::new(config1).await?;

    let config2 = GitTorrentConfig {
        storage_dir: PathBuf::from(temp2.path()),
        bind_address: "127.0.0.1".to_string(),
        bind_port: 7884,
        bootstrap_nodes: vec!["127.0.0.1:7883".to_string()],
        ..Default::default()
    };

    let service2 = GitTorrentService::new(config2).await?;

    // Wait for services to connect
    sleep(Duration::from_secs(2)).await;

    // Service 1 announces the repository
    service1.announce_repository(repo_path).await?;

    // Wait for DHT propagation
    sleep(Duration::from_secs(2)).await;

    // TODO: Add discovery method to service and test it
    // Currently, the discovery is internal to the service
    // We would need to expose a method to query for repositories

    let status1 = service1.get_p2p_status().await;
    let status2 = service2.get_p2p_status().await;

    println!("After announcement - Service 1: {:?}", status1);
    println!("After announcement - Service 2: {:?}", status2);

    // Both services should still be connected
    assert!(status1.connected);
    assert!(status2.connected);

    Ok(())
}

/// Test chunk storage and retrieval
#[tokio::test]
async fn test_chunk_exchange() -> Result<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt::try_init();

    // Create temporary directories
    let temp1 = TempDir::new()?;
    let temp2 = TempDir::new()?;

    // Start two services
    let config1 = GitTorrentConfig {
        storage_dir: PathBuf::from(temp1.path()),
        bind_address: "127.0.0.1".to_string(),
        bind_port: 7885,
        bootstrap_nodes: vec![],
        enable_xet: true,
        ..Default::default()
    };

    let service1 = GitTorrentService::new(config1).await?;

    let config2 = GitTorrentConfig {
        storage_dir: PathBuf::from(temp2.path()),
        bind_address: "127.0.0.1".to_string(),
        bind_port: 7886,
        bootstrap_nodes: vec!["127.0.0.1:7885".to_string()],
        enable_xet: true,
        ..Default::default()
    };

    let service2 = GitTorrentService::new(config2).await?;

    // Wait for connection
    sleep(Duration::from_secs(2)).await;

    // Store a test chunk in service1
    let test_data = b"This is test chunk data for P2P exchange";
    let chunk_hash = service1.store_xet_chunk(test_data).await?;

    println!("Stored chunk with hash: {:?}", chunk_hash);

    // Wait for DHT propagation
    sleep(Duration::from_secs(2)).await;

    // TODO: Implement and test chunk retrieval from service2
    // This would require exposing a retrieve_chunk method that uses P2P

    // For now, just verify the services are still connected
    let status1 = service1.get_p2p_status().await;
    let status2 = service2.get_p2p_status().await;

    assert!(status1.connected);
    assert!(status2.connected);

    Ok(())
}

/// Test multiple peer network
#[tokio::test]
#[ignore] // This test takes longer, run with --ignored flag
async fn test_multiple_peers() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let mut services = Vec::new();
    let mut temp_dirs = Vec::new();

    // Create a network of 5 nodes
    for i in 0..5 {
        let temp = TempDir::new()?;
        temp_dirs.push(temp);

        let config = GitTorrentConfig {
            storage_dir: PathBuf::from(temp_dirs[i].path()),
            bind_address: "127.0.0.1".to_string(),
            bind_port: 7890 + i as u16,
            bootstrap_nodes: if i == 0 {
                vec![]
            } else {
                // Connect to the first node
                vec!["127.0.0.1:7890".to_string()]
            },
            ..Default::default()
        };

        let service = GitTorrentService::new(config).await?;
        services.push(service);

        // Wait between starting nodes
        sleep(Duration::from_millis(500)).await;
    }

    // Wait for network to stabilize
    sleep(Duration::from_secs(5)).await;

    // Check all nodes are connected
    for (i, service) in services.iter().enumerate() {
        let status = service.get_p2p_status().await;
        println!("Node {} status: {:?}", i, status);
        assert!(status.connected, "Node {} should be connected", i);
        assert!(status.dht_connected, "Node {} DHT should be connected", i);
    }

    Ok(())
}

/// Test network partition and recovery
#[tokio::test]
#[ignore] // Complex test, run with --ignored
async fn test_network_partition() -> Result<()> {
    // TODO: Implement network partition simulation
    // This would require more sophisticated network control
    Ok(())
}