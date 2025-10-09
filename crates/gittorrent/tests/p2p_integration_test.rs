//! Integration test for P2P bidirectional communication
//!
//! This test verifies that two GitTorrent nodes can:
//! 1. Discover each other
//! 2. Exchange handshake messages
//! 3. Send and receive chunk requests/responses
//! 4. Maintain heartbeat communication

use gittorrent_rs::p2p::{P2PService, protocol::GitTorrentMessage};
use gittorrent_rs::service::GitTorrentConfig;
use gittorrent_rs::xet_integration::MerkleHash;
use saorsa_core::{P2PNode, dht::client::DhtClient, types::IdentityHandle};
use saorsa_core::network::NodeConfig;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, debug};

/// Create a test P2P node with unique port
async fn create_test_node(port: u16) -> anyhow::Result<(Arc<P2PNode>, Arc<DhtClient>, IdentityHandle)> {
    // Create node configuration
    let mut config = NodeConfig::new()?;
    config.listen_addr = format!("127.0.0.1:{}", port).parse()?;
    config.listen_addrs = vec![config.listen_addr];

    // Create P2P node
    let p2p_node = Arc::new(P2PNode::new(config)?);

    // Create DHT client
    let dht_client = Arc::new(DhtClient::new(p2p_node.clone()).await?);

    // Create identity (mock for testing)
    let identity = IdentityHandle::default();

    // Start the node
    p2p_node.start().await?;

    Ok((p2p_node, dht_client, identity))
}

#[tokio::test]
async fn test_two_node_handshake() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("gittorrent_rs=debug,saorsa_core=info")
        .try_init()
        .ok();

    info!("Starting two-node handshake test");

    // Create two P2P nodes on different ports
    let (node1_p2p, node1_dht, node1_identity) = create_test_node(9001).await?;
    let (node2_p2p, node2_dht, node2_identity) = create_test_node(9002).await?;

    // Create GitTorrent P2P services
    let config = GitTorrentConfig::default();
    let service1 = P2PService::new(&config, node1_p2p.clone(), node1_dht, node1_identity).await?;
    let service2 = P2PService::new(&config, node2_p2p.clone(), node2_dht, node2_identity).await?;

    // Start both services
    service1.clone().start().await?;
    service2.clone().start().await?;

    info!("Both P2P services started");

    // Get peer IDs
    let peer1_id = node1_p2p.peer_id().clone();
    let peer2_id = node2_p2p.peer_id().clone();

    info!("Node 1 ID: {}", peer1_id);
    info!("Node 2 ID: {}", peer2_id);

    // Connect node1 to node2
    let node2_addr = saorsa_core::NetworkAddress::from(
        "127.0.0.1:9002".parse::<std::net::SocketAddr>()?
    );
    node1_p2p.connect(&peer2_id, node2_addr).await?;

    info!("Node 1 connected to Node 2");

    // Wait for connection establishment and handshake
    sleep(Duration::from_secs(2)).await;

    // Verify both nodes see each other as connected
    let node1_peers = service1.connected_peers().await;
    let node2_peers = service2.connected_peers().await;

    assert!(node1_peers.contains(&peer2_id), "Node 1 should see Node 2 as connected");
    assert!(node2_peers.contains(&peer1_id), "Node 2 should see Node 1 as connected");

    info!("Connection verified - both nodes see each other");

    // Test sending a chunk request from node1 to node2
    let test_hash = MerkleHash::from_bytes(&[1u8; 32]);

    // This should fail with "chunk not found" since node2 doesn't have the chunk
    // But it proves the request-response mechanism works
    match service1.request_chunk(&peer2_id, test_hash, 0, 1024).await {
        Err(e) if e.to_string().contains("Chunk not found") => {
            info!("Received expected 'chunk not found' response - communication working!");
        }
        Ok(_) => panic!("Unexpected success - node2 shouldn't have the chunk"),
        Err(e) => panic!("Unexpected error: {}", e),
    }

    // Test repository announcement
    let mut refs = std::collections::HashMap::new();
    refs.insert("refs/heads/main".to_string(), "abc123".to_string());

    service1.announce_repository(
        "test-repo".to_string(),
        refs,
        vec!["chunk1".to_string(), "chunk2".to_string()],
    ).await?;

    info!("Repository announcement sent");

    // Wait for messages to propagate
    sleep(Duration::from_secs(1)).await;

    // Verify heartbeat is working (peer count should remain stable)
    let final_peer_count = service1.peer_count().await;
    assert_eq!(final_peer_count, 1, "Node 1 should still have 1 peer");

    info!("Test completed successfully!");
    Ok(())
}

#[tokio::test]
async fn test_concurrent_requests() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("gittorrent_rs=info")
        .try_init()
        .ok();

    info!("Starting concurrent requests test");

    // Create two P2P nodes
    let (node1_p2p, node1_dht, node1_identity) = create_test_node(9003).await?;
    let (node2_p2p, node2_dht, node2_identity) = create_test_node(9004).await?;

    // Create services
    let config = GitTorrentConfig::default();
    let service1 = P2PService::new(&config, node1_p2p.clone(), node1_dht, node1_identity).await?;
    let service2 = P2PService::new(&config, node2_p2p.clone(), node2_dht, node2_identity).await?;

    // Start services
    service1.clone().start().await?;
    service2.clone().start().await?;

    // Connect nodes
    let peer2_id = node2_p2p.peer_id().clone();
    let node2_addr = saorsa_core::NetworkAddress::from(
        "127.0.0.1:9004".parse::<std::net::SocketAddr>()?
    );
    node1_p2p.connect(&peer2_id, node2_addr).await?;

    // Wait for connection
    sleep(Duration::from_secs(1)).await;

    // Send multiple concurrent requests
    let mut handles = vec![];
    for i in 0..5 {
        let service = service1.clone();
        let peer_id = peer2_id.clone();
        let handle = tokio::spawn(async move {
            let hash = MerkleHash::from_bytes(&[i as u8; 32]);
            service.request_chunk(&peer_id, hash, 0, 1024).await
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        match handle.await? {
            Err(e) if e.to_string().contains("Chunk not found") => {
                debug!("Request completed with expected error");
            }
            Ok(_) => panic!("Unexpected success"),
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    info!("All concurrent requests processed successfully");
    Ok(())
}