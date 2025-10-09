//! Example demonstrating P2P handshake between two GitTorrent nodes
//!
//! This example shows how the transport layer enables bidirectional communication:
//! 1. Two nodes start up and discover each other
//! 2. They exchange handshake messages
//! 3. One node requests a chunk from another
//! 4. The second node responds with "chunk not found"

use gittorrent_rs::p2p::P2PService;
use gittorrent_rs::service::GitTorrentConfig;
use gittorrent_rs::xet_integration::MerkleHash;
use saorsa_core::{P2PNode, dht::client::DhtClient, types::IdentityHandle};
use saorsa_core::network::NodeConfig;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, error};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("gittorrent_rs=debug,saorsa_core=info")
        .init();

    info!("Starting P2P handshake example");

    // Create first node
    info!("Creating Node 1 on port 8001...");
    let mut config1 = NodeConfig::new()?;
    config1.listen_addr = "127.0.0.1:8001".parse()?;
    config1.listen_addrs = vec![config1.listen_addr];

    let node1 = Arc::new(P2PNode::new(config1)?);
    let dht1 = Arc::new(DhtClient::new(node1.clone()).await?);
    let identity1 = IdentityHandle::default();

    // Create second node
    info!("Creating Node 2 on port 8002...");
    let mut config2 = NodeConfig::new()?;
    config2.listen_addr = "127.0.0.1:8002".parse()?;
    config2.listen_addrs = vec![config2.listen_addr];

    let node2 = Arc::new(P2PNode::new(config2)?);
    let dht2 = Arc::new(DhtClient::new(node2.clone()).await?);
    let identity2 = IdentityHandle::default();

    // Start both nodes
    info!("Starting P2P nodes...");
    node1.start().await?;
    node2.start().await?;

    // Create GitTorrent P2P services with transport layer
    info!("Creating P2P services with transport layer...");
    let config = GitTorrentConfig::default();
    let service1 = P2PService::new(&config, node1.clone(), dht1, identity1).await?;
    let service2 = P2PService::new(&config, node2.clone(), dht2, identity2).await?;

    // Start services
    service1.clone().start().await?;
    service2.clone().start().await?;

    info!("P2P services started successfully");

    // Get peer IDs
    let peer1_id = node1.peer_id().clone();
    let peer2_id = node2.peer_id().clone();

    info!("Node 1 peer ID: {}", peer1_id);
    info!("Node 2 peer ID: {}", peer2_id);

    // Subscribe to events
    let mut events1 = service1.subscribe();
    let mut events2 = service2.subscribe();

    // Monitor events in background
    tokio::spawn(async move {
        while let Ok(event) = events1.recv().await {
            info!("Node 1 event: {:?}", event);
        }
    });

    tokio::spawn(async move {
        while let Ok(event) = events2.recv().await {
            info!("Node 2 event: {:?}", event);
        }
    });

    // Connect node1 to node2
    info!("Connecting Node 1 to Node 2...");
    let addr2 = saorsa_core::NetworkAddress::from("127.0.0.1:8002".parse::<std::net::SocketAddr>()?);
    node1.connect(&peer2_id, addr2).await?;

    // Wait for connection and handshake exchange
    info!("Waiting for connection establishment and handshake exchange...");
    sleep(Duration::from_secs(3)).await;

    // Check connection status
    let peers1 = service1.connected_peers().await;
    let peers2 = service2.connected_peers().await;

    info!("Node 1 sees {} peers: {:?}", peers1.len(), peers1);
    info!("Node 2 sees {} peers: {:?}", peers2.len(), peers2);

    if peers1.is_empty() || peers2.is_empty() {
        error!("Connection failed - nodes don't see each other");
        return Ok(());
    }

    info!("Connection established! Nodes can see each other.");

    // Test chunk request/response
    info!("Testing chunk request/response mechanism...");
    let test_hash = MerkleHash::from_bytes(&[42u8; 32]);

    info!("Node 1 requesting chunk from Node 2...");
    match service1.request_chunk(&peer2_id, test_hash, 0, 1024).await {
        Ok(data) => {
            info!("Received chunk data: {} bytes", data.len());
        }
        Err(e) if e.to_string().contains("Chunk not found") => {
            info!("Received expected 'chunk not found' response - bidirectional communication working!");
        }
        Err(e) => {
            error!("Request failed: {}", e);
        }
    }

    // Test repository announcement
    info!("Testing repository announcement...");
    let mut refs = std::collections::HashMap::new();
    refs.insert("refs/heads/main".to_string(), "deadbeef".to_string());

    service1.announce_repository(
        "example-repo".to_string(),
        refs,
        vec!["chunk1".to_string(), "chunk2".to_string()],
    ).await?;

    info!("Repository announced from Node 1");

    // Keep running for a bit to observe heartbeats
    info!("Observing heartbeat messages for 5 seconds...");
    sleep(Duration::from_secs(5)).await;

    // Final status
    let final_peers1 = service1.peer_count().await;
    let final_peers2 = service2.peer_count().await;

    info!("Final status:");
    info!("  Node 1: {} connected peers", final_peers1);
    info!("  Node 2: {} connected peers", final_peers2);

    info!("Example completed successfully!");
    info!("Bidirectional P2P communication is working!");

    Ok(())
}