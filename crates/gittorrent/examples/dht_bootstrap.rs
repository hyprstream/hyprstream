//! Example demonstrating proper Kademlia DHT bootstrap in libp2p
//!
//! This example shows how to correctly bootstrap a Kademlia DHT node by:
//! 1. Adding peer addresses to the routing table
//! 2. Establishing connections
//! 3. Calling bootstrap() once peers are known
//!
//! Run with: cargo run --example dht_bootstrap

use gittorrent::dht::GitTorrentDht;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("gittorrent=debug,libp2p=info")
        .init();

    println!("Starting GitTorrent DHT Bootstrap Example");
    println!("==========================================\n");

    // Create DHT instance on a random port (0) in Server mode
    let dht = GitTorrentDht::new(0, gittorrent::DhtMode::Server).await?;

    // Example bootstrap addresses with peer IDs
    // Format: /ip4/<ip>/tcp/<port>/p2p/<peer_id>
    let bootstrap_peers = vec![
        // Example 1: Local peer (if you have another node running)
        // "/ip4/127.0.0.1/tcp/4001/p2p/12D3KooWExample...",

        // Example 2: Using the helper function if you know the peer ID
        // GitTorrentDht::create_bootstrap_addr(
        //     "127.0.0.1",
        //     4001,
        //     &PeerId::from_str("12D3KooWExample...")?,
        // ),

        // For testing, you can run two instances of this example
        // and use the peer ID from the first in the second
    ];

    if !bootstrap_peers.is_empty() {
        println!("Bootstrapping with {} peers...", bootstrap_peers.len());

        match dht.bootstrap(bootstrap_peers).await {
            Ok(()) => {
                println!("✓ Bootstrap initiated successfully!");
                println!("  The DHT will now:");
                println!("  1. Connect to the specified peers");
                println!("  2. Populate its routing table via Identify protocol");
                println!("  3. Perform iterative lookups to discover more peers");
            }
            Err(e) => {
                println!("✗ Bootstrap failed: {e}");
                println!("  Make sure:");
                println!("  - The bootstrap addresses include peer IDs (/p2p/<id>)");
                println!("  - The target peers are running and reachable");
                println!("  - The network configuration allows connections");
            }
        }
    } else {
        println!("No bootstrap peers configured.");
        println!("The node will:");
        println!("  - Accept incoming connections");
        println!("  - Discover local peers via mDNS");
        println!("  - Build its routing table as peers connect");
    }

    // Keep the node running
    println!("\n📡 DHT node is running. Press Ctrl-C to exit.\n");

    // In a real application, you would perform DHT operations here
    tokio::signal::ctrl_c().await?;

    println!("\nShutting down...");
    Ok(())
}