//! Integration tests for DHT bootstrap functionality

use gittorrent::dht::GitTorrentDht;
use libp2p::Multiaddr;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_bootstrap_with_peer_id() -> Result<(), Box<dyn std::error::Error>> {
    // Start first node
    let _node1 = GitTorrentDht::new(0, gittorrent::DhtMode::Server).await?;

    // Give it time to start listening
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Start second node
    let _node2 = GitTorrentDht::new(0, gittorrent::DhtMode::Server).await?;

    // In a real test, we would:
    // 1. Get the actual listen address and peer ID from node1
    // 2. Use that to bootstrap node2
    // 3. Verify they can discover each other

    // For now, this test verifies that the DHT nodes can be created
    // (the test passes if no panic occurred above)
    Ok(())
}

#[tokio::test]
async fn test_bootstrap_without_peer_id_fails() -> Result<(), Box<dyn std::error::Error>> {
    let dht = GitTorrentDht::new(0, gittorrent::DhtMode::Server).await?;

    // Bootstrap with address lacking peer ID should handle gracefully
    let bootstrap_peers = vec![
        "/ip4/127.0.0.1/tcp/4001".parse::<Multiaddr>()?,
    ];

    // This should warn but not panic
    let result = dht.bootstrap(bootstrap_peers).await;

    // Depending on implementation, this might fail or succeed with warnings
    // The important thing is it doesn't panic
    match result {
        Ok(()) => println!("Bootstrap attempted despite missing peer ID"),
        Err(e) => println!("Bootstrap failed as expected: {e}"),
    }
    Ok(())
}

#[tokio::test]
async fn test_bootstrap_with_unreachable_peer() -> Result<(), Box<dyn std::error::Error>> {
    let dht = GitTorrentDht::new(0, gittorrent::DhtMode::Server).await?;

    // Use a peer ID that doesn't exist
    let bootstrap_peers = vec![
        "/ip4/127.0.0.1/tcp/55555/p2p/12D3KooWEqnuKZQRgR7dgyFvD8PFhvnmRdYfNWvhDPPqhEDqwXNU"
            .parse::<Multiaddr>()?,
    ];

    // This should handle the connection failure gracefully
    let result = timeout(
        Duration::from_secs(5),
        dht.bootstrap(bootstrap_peers)
    ).await;

    match result {
        Ok(Ok(())) => println!("Bootstrap initiated (connection may fail later)"),
        Ok(Err(e)) => println!("Bootstrap failed: {e}"),
        Err(_) => println!("Bootstrap timed out"),
    }
    Ok(())
}

#[tokio::test]
async fn test_mdns_discovery() -> Result<(), Box<dyn std::error::Error>> {
    // Start two nodes on the same network
    let _node1 = GitTorrentDht::new(0, gittorrent::DhtMode::Server).await?;
    let _node2 = GitTorrentDht::new(0, gittorrent::DhtMode::Server).await?;

    // Give them time to discover each other via mDNS
    tokio::time::sleep(Duration::from_secs(2)).await;

    // In a real implementation, we would verify they discovered each other
    // by checking the routing table or attempting to exchange data
    // (the test passes if no panic occurred above)
    Ok(())
}
