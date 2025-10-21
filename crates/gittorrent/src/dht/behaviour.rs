//! libp2p behaviour for GitTorrent DHT

use libp2p::{
    kad,
    identify,
    mdns,
    swarm::NetworkBehaviour,
    PeerId,
};
use crate::dht::storage::GitObjectStore;
use crate::error::Result;

/// Combined behaviour for GitTorrent DHT
#[derive(NetworkBehaviour)]
pub struct GitTorrentBehaviour {
    pub kademlia: kad::Behaviour<GitObjectStore>,
    pub mdns: mdns::tokio::Behaviour,
    pub identify: identify::Behaviour,
}

impl GitTorrentBehaviour {
    /// Create a new GitTorrent behaviour
    pub fn new_with_keypair(keypair: &libp2p::identity::Keypair) -> Result<Self> {
        let local_peer_id = PeerId::from(keypair.public());
        // Create custom Git object store
        let store = GitObjectStore::new();

        // Configure Kademlia
        let mut kad_config = kad::Config::default();
        kad_config.set_query_timeout(std::time::Duration::from_secs(60));
        kad_config.set_replication_factor(std::num::NonZeroUsize::new(3).unwrap());

        // Create Kademlia DHT
        let kademlia = kad::Behaviour::with_config(local_peer_id, store, kad_config);

        // Create mDNS for local discovery
        let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)?;

        // Create Identify protocol
        let identify = identify::Behaviour::new(identify::Config::new(
            "/gittorrent/1.0.0".to_string(),
            keypair.public()
        ));

        Ok(Self {
            kademlia,
            mdns,
            identify,
        })
    }

    /// Add a known peer to the routing table
    pub fn add_address(&mut self, peer_id: PeerId, address: libp2p::Multiaddr) {
        self.kademlia.add_address(&peer_id, address);
    }

    /// Get routing table information
    pub fn routing_table_info(&mut self) -> Vec<PeerId> {
        self.kademlia.kbuckets().map(|bucket| bucket.iter().map(|entry| *entry.node.key.preimage()).collect::<Vec<_>>()).flatten().collect()
    }
}