//! Example implementation of libp2p Kademlia DHT for GitTorrent
//!
//! This example demonstrates how to implement a complete libp2p-based DHT service
//! that can replace the current BEP 44 implementation while maintaining all
//! GitTorrent functionality.

use anyhow::Result;
use libp2p::{
    kad::{
        record::{store::MemoryStore, Key, Record},
        GetRecordOk, Kademlia, KademliaConfig, KademliaEvent, PeerRecord, PutRecordOk,
        QueryResult, Quorum,
    },
    identify,
    mdns,
    request_response::{self, ProtocolSupport},
    swarm::{NetworkBehaviour, SwarmBuilder, SwarmEvent},
    tcp, noise, yamux,
    identity::Keypair,
    Multiaddr, PeerId, Transport,
};
use multihash::{Code, MultihashDigest};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, warn};

// Re-use types from the main codebase
use ed25519_dalek::{SigningKey, Signature, Signer, Verifier, VerifyingKey};

/// Custom key type for Git SHA256 objects in the DHT
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GitObjectKey {
    inner: Key,
    git_oid: String,
}

impl GitObjectKey {
    /// Create a DHT key from a Git object ID (SHA256)
    pub fn from_git_oid(oid: &str) -> Result<Self> {
        // Validate the OID format
        if oid.len() != 64 || !oid.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(anyhow::anyhow!("Invalid Git SHA256 OID: {}", oid));
        }

        // Git OIDs are already SHA256 hashes, convert to bytes
        let oid_bytes = hex::decode(oid)?;

        // Create a multihash with SHA256 code
        let mh = Code::Sha2_256.digest(&oid_bytes);
        let key = Key::from(mh.to_bytes());

        Ok(GitObjectKey {
            inner: key,
            git_oid: oid.to_string(),
        })
    }

    /// Get the underlying Kademlia key
    pub fn as_kad_key(&self) -> &Key {
        &self.inner
    }

    /// Get the original Git OID
    pub fn git_oid(&self) -> &str {
        &self.git_oid
    }
}

/// Mutable record with Ed25519 signatures (BEP 44 compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutableRecord {
    /// Ed25519 public key (32 bytes)
    pub public_key: Vec<u8>,
    /// Sequence number for versioning
    pub sequence: u64,
    /// The actual data value
    pub value: Vec<u8>,
    /// Ed25519 signature (64 bytes)
    pub signature: Vec<u8>,
    /// Unix timestamp when this record expires
    pub expires: Option<u64>,
    /// Salt for key derivation (optional)
    pub salt: Option<Vec<u8>>,
}

impl MutableRecord {
    /// Create a new signed mutable record
    pub fn new(
        signing_key: &SigningKey,
        value: Vec<u8>,
        sequence: u64,
        salt: Option<Vec<u8>>,
    ) -> Result<Self> {
        let verifying_key = signing_key.verifying_key();
        let public_key = verifying_key.as_bytes().to_vec();

        // Create the signature payload (BEP 44 compatible format)
        let sig_payload = Self::create_signature_payload(&value, sequence, &salt);
        let signature = signing_key.sign(&sig_payload).to_bytes().to_vec();

        // Set expiration to 24 hours from now
        let expires = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_secs()
                + 86400,
        );

        Ok(MutableRecord {
            public_key,
            sequence,
            value,
            signature,
            expires,
            salt,
        })
    }

    /// Create the signature payload
    fn create_signature_payload(value: &[u8], sequence: u64, salt: &Option<Vec<u8>>) -> Vec<u8> {
        let mut payload = Vec::new();

        // Add salt if present
        if let Some(salt) = salt {
            payload.extend_from_slice(salt);
        }

        // Add sequence number (big-endian)
        payload.extend_from_slice(&sequence.to_be_bytes());

        // Add the value
        payload.extend_from_slice(value);

        payload
    }

    /// Verify the signature of this record
    pub fn verify(&self) -> Result<bool> {
        if self.public_key.len() != 32 {
            return Ok(false);
        }
        if self.signature.len() != 64 {
            return Ok(false);
        }

        let verifying_key = VerifyingKey::from_bytes(
            self.public_key.as_slice().try_into()?
        )?;

        let sig_payload = Self::create_signature_payload(&self.value, self.sequence, &self.salt);
        let signature = Signature::from_bytes(
            self.signature.as_slice().try_into()?
        );

        Ok(verifying_key.verify(&sig_payload, &signature).is_ok())
    }

    /// Get the DHT key for this mutable record
    pub fn get_key(&self) -> Key {
        // Use SHA256 of public key + salt (if present)
        let mut hasher = Sha256::new();
        hasher.update(&self.public_key);
        if let Some(salt) = &self.salt {
            hasher.update(salt);
        }
        let hash = hasher.finalize();

        let mh = Code::Sha2_256.digest(&hash);
        Key::from(mh.to_bytes())
    }

    /// Convert to a Kademlia record
    pub fn to_kad_record(&self) -> Result<Record> {
        let key = self.get_key();
        let value = bincode::serialize(self)?;

        Ok(Record {
            key,
            value,
            publisher: None,
            expires: self.expires.map(|t| {
                std::time::Instant::now() + Duration::from_secs(t - current_unix_timestamp())
            }),
        })
    }

    /// Check if this record is newer than another
    pub fn is_newer_than(&self, other: &MutableRecord) -> bool {
        self.sequence > other.sequence
    }
}

/// Protocol for requesting Git objects directly from peers
#[derive(Debug, Clone)]
pub struct GitObjectProtocol;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitObjectRequest {
    pub oid: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitObjectResponse {
    pub oid: String,
    pub data: Option<Vec<u8>>,
}

impl request_response::Codec for GitObjectProtocol {
    type Protocol = &'static str;
    type Request = GitObjectRequest;
    type Response = GitObjectResponse;

    fn read_request<'a, T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
    ) -> std::io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send + 'a,
    {
        // Implementation would deserialize the request
        todo!()
    }

    fn read_response<'a, T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
    ) -> std::io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send + 'a,
    {
        // Implementation would deserialize the response
        todo!()
    }

    fn write_request<'a, T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> std::io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send + 'a,
    {
        // Implementation would serialize the request
        todo!()
    }

    fn write_response<'a, T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> std::io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send + 'a,
    {
        // Implementation would serialize the response
        todo!()
    }
}

/// Combined network behaviour for GitTorrent
#[derive(NetworkBehaviour)]
pub struct GitTorrentBehaviour {
    kademlia: Kademlia<MemoryStore>,
    identify: identify::Behaviour,
    mdns: mdns::tokio::Behaviour,
    request_response: request_response::cbor::Behaviour<GitObjectRequest, GitObjectResponse>,
}

/// Main DHT service for GitTorrent using libp2p
pub struct GitTorrentDHT {
    /// The libp2p swarm
    swarm: libp2p::Swarm<GitTorrentBehaviour>,
    /// Our peer ID
    local_peer_id: PeerId,
    /// Manager for mutable record sequences
    sequence_tracker: Arc<Mutex<HashMap<Vec<u8>, u64>>>,
    /// Local cache of Git objects
    object_cache: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    /// Event receiver for DHT events
    event_rx: mpsc::UnboundedReceiver<DHTEvent>,
    /// Event sender
    event_tx: mpsc::UnboundedSender<DHTEvent>,
}

/// Events emitted by the DHT
#[derive(Debug, Clone)]
pub enum DHTEvent {
    /// Successfully stored a record
    RecordStored { key: Vec<u8> },
    /// Retrieved a record from the DHT
    RecordRetrieved { key: Vec<u8>, value: Vec<u8> },
    /// Peer discovered through mDNS
    PeerDiscovered { peer_id: PeerId, addresses: Vec<Multiaddr> },
    /// Connected to a peer
    PeerConnected { peer_id: PeerId },
    /// Disconnected from a peer
    PeerDisconnected { peer_id: PeerId },
}

impl GitTorrentDHT {
    /// Create a new GitTorrent DHT service
    pub async fn new(keypair: Keypair, listen_addr: Multiaddr) -> Result<Self> {
        let local_peer_id = PeerId::from(keypair.public());
        info!("Starting GitTorrent DHT with peer ID: {}", local_peer_id);

        // Configure Kademlia
        let mut kad_config = KademliaConfig::default();
        kad_config.set_query_timeout(Duration::from_secs(60));
        kad_config.set_replication_factor(20.try_into()?);
        kad_config.set_parallelism(3.try_into()?);
        kad_config.set_record_ttl(Some(Duration::from_secs(86400))); // 24 hours

        // Create Kademlia with memory store
        let store = MemoryStore::new(local_peer_id);
        let mut kademlia = Kademlia::with_config(local_peer_id, store, kad_config);

        // Set Kademlia to server mode (can store records for others)
        kademlia.set_mode(Some(libp2p::kad::Mode::Server));

        // Create identify behaviour
        let identify = identify::Behaviour::new(
            identify::Config::new("/gittorrent/1.0.0".to_string(), keypair.public())
                .with_push_listen_addr_updates(true),
        );

        // Create mDNS for local peer discovery
        let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)?;

        // Create request-response behaviour for direct Git object transfers
        let request_response = request_response::cbor::Behaviour::new(
            [(
                libp2p::StreamProtocol::new("/gittorrent/git-objects/1.0.0"),
                ProtocolSupport::Full,
            )],
            request_response::Config::default(),
        );

        // Combine all behaviours
        let behaviour = GitTorrentBehaviour {
            kademlia,
            identify,
            mdns,
            request_response,
        };

        // Build the transport
        let transport = tcp::tokio::Transport::new(tcp::Config::default())
            .upgrade(libp2p::core::upgrade::Version::V1)
            .authenticate(noise::Config::new(&keypair)?)
            .multiplex(yamux::Config::default())
            .boxed();

        // Create the swarm
        let mut swarm = SwarmBuilder::with_tokio_executor(transport, behaviour, local_peer_id).build();

        // Listen on the specified address
        swarm.listen_on(listen_addr.clone())?;
        info!("DHT listening on: {}", listen_addr);

        // Create event channel
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Ok(GitTorrentDHT {
            swarm,
            local_peer_id,
            sequence_tracker: Arc::new(Mutex::new(HashMap::new())),
            object_cache: Arc::new(Mutex::new(HashMap::new())),
            event_rx,
            event_tx,
        })
    }

    /// Bootstrap the DHT with known peers
    pub async fn bootstrap(&mut self, bootstrap_peers: Vec<(PeerId, Multiaddr)>) -> Result<()> {
        for (peer_id, addr) in bootstrap_peers {
            self.swarm
                .behaviour_mut()
                .kademlia
                .add_address(&peer_id, addr.clone());
            info!("Added bootstrap peer: {} at {}", peer_id, addr);
        }

        // Start the bootstrap process
        if let Err(e) = self.swarm.behaviour_mut().kademlia.bootstrap() {
            warn!("Bootstrap failed: {}", e);
        }

        Ok(())
    }

    /// Store a Git object in the DHT
    pub async fn put_git_object(&mut self, oid: &str, data: Vec<u8>) -> Result<()> {
        info!("Storing Git object: {}", oid);

        // Create the key
        let key = GitObjectKey::from_git_oid(oid)?;

        // Create the record
        let record = Record {
            key: key.as_kad_key().clone(),
            value: data.clone(),
            publisher: Some(self.local_peer_id),
            expires: None, // Git objects are immutable, no expiry
        };

        // Store in DHT
        self.swarm
            .behaviour_mut()
            .kademlia
            .put_record(record, Quorum::One)?;

        // Cache locally
        self.object_cache.lock().await.insert(oid.to_string(), data);

        Ok(())
    }

    /// Retrieve a Git object from the DHT
    pub async fn get_git_object(&mut self, oid: &str) -> Result<Option<Vec<u8>>> {
        info!("Retrieving Git object: {}", oid);

        // Check local cache first
        if let Some(data) = self.object_cache.lock().await.get(oid) {
            debug!("Git object {} found in local cache", oid);
            return Ok(Some(data.clone()));
        }

        // Create the key
        let key = GitObjectKey::from_git_oid(oid)?;

        // Query the DHT
        let query_id = self
            .swarm
            .behaviour_mut()
            .kademlia
            .get_record(key.as_kad_key().clone());

        debug!("Started DHT query {:?} for object {}", query_id, oid);

        // In production, this would be event-driven
        // For this example, we'll return None and handle asynchronously
        Ok(None)
    }

    /// Store a mutable record (e.g., repository refs)
    pub async fn put_mutable_record(
        &mut self,
        signing_key: &SigningKey,
        value: Vec<u8>,
        salt: Option<Vec<u8>>,
    ) -> Result<()> {
        let public_key = signing_key.verifying_key().as_bytes().to_vec();

        // Get the next sequence number
        let mut tracker = self.sequence_tracker.lock().await;
        let sequence = tracker
            .entry(public_key.clone())
            .and_modify(|s| *s += 1)
            .or_insert(0);
        let sequence = *sequence;

        // Create the mutable record
        let mutable_record = MutableRecord::new(signing_key, value, sequence, salt)?;

        // Verify the signature
        if !mutable_record.verify()? {
            return Err(anyhow::anyhow!("Failed to verify own signature"));
        }

        // Convert to Kademlia record
        let kad_record = mutable_record.to_kad_record()?;

        info!(
            "Storing mutable record with sequence {} at key: {:?}",
            sequence,
            hex::encode(&kad_record.key.to_vec())
        );

        // Store in DHT
        self.swarm
            .behaviour_mut()
            .kademlia
            .put_record(kad_record, Quorum::Majority)?;

        Ok(())
    }

    /// Get a mutable record from the DHT
    pub async fn get_mutable_record(
        &mut self,
        public_key: &[u8],
        salt: Option<Vec<u8>>,
    ) -> Result<Option<MutableRecord>> {
        // Calculate the key
        let mut hasher = Sha256::new();
        hasher.update(public_key);
        if let Some(salt) = &salt {
            hasher.update(salt);
        }
        let hash = hasher.finalize();

        let mh = Code::Sha2_256.digest(&hash);
        let key = Key::from(mh.to_bytes());

        info!(
            "Retrieving mutable record at key: {:?}",
            hex::encode(&key.to_vec())
        );

        // Query the DHT
        let query_id = self.swarm.behaviour_mut().kademlia.get_record(key);

        debug!("Started DHT query {:?} for mutable record", query_id);

        // In production, this would be event-driven
        Ok(None)
    }

    /// Run the main event loop
    pub async fn run(&mut self) -> Result<()> {
        loop {
            tokio::select! {
                event = self.swarm.select_next_some() => {
                    self.handle_swarm_event(event).await?;
                }
                Some(event) = self.event_rx.recv() => {
                    debug!("Processing DHT event: {:?}", event);
                }
            }
        }
    }

    /// Handle swarm events
    async fn handle_swarm_event(
        &mut self,
        event: SwarmEvent<GitTorrentBehaviourEvent>,
    ) -> Result<()> {
        match event {
            SwarmEvent::Behaviour(GitTorrentBehaviourEvent::Kademlia(kad_event)) => {
                self.handle_kademlia_event(kad_event).await?;
            }
            SwarmEvent::Behaviour(GitTorrentBehaviourEvent::Identify(identify_event)) => {
                self.handle_identify_event(identify_event).await?;
            }
            SwarmEvent::Behaviour(GitTorrentBehaviourEvent::Mdns(mdns_event)) => {
                self.handle_mdns_event(mdns_event).await?;
            }
            SwarmEvent::NewListenAddr { address, .. } => {
                info!("Listening on: {}", address);
            }
            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                info!("Connected to peer: {}", peer_id);
                self.event_tx.send(DHTEvent::PeerConnected { peer_id })?;
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                info!("Disconnected from peer: {}", peer_id);
                self.event_tx.send(DHTEvent::PeerDisconnected { peer_id })?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Handle Kademlia events
    async fn handle_kademlia_event(&mut self, event: KademliaEvent) -> Result<()> {
        match event {
            KademliaEvent::OutboundQueryProgressed { result, .. } => match result {
                QueryResult::GetRecord(Ok(GetRecordOk::FoundRecord(peer_record))) => {
                    match peer_record {
                        PeerRecord {
                            record: Record { key, value, .. },
                            ..
                        } => {
                            info!("Retrieved record from DHT: key={:?}", hex::encode(&key.to_vec()));

                            // Try to deserialize as mutable record
                            if let Ok(mutable_record) = bincode::deserialize::<MutableRecord>(&value)
                            {
                                if mutable_record.verify()? {
                                    info!("Valid mutable record with sequence {}", mutable_record.sequence);
                                } else {
                                    warn!("Invalid signature on mutable record");
                                }
                            }

                            self.event_tx.send(DHTEvent::RecordRetrieved {
                                key: key.to_vec(),
                                value,
                            })?;
                        }
                    }
                }
                QueryResult::PutRecord(Ok(PutRecordOk { key })) => {
                    info!("Successfully stored record: key={:?}", hex::encode(&key.to_vec()));
                    self.event_tx.send(DHTEvent::RecordStored { key: key.to_vec() })?;
                }
                QueryResult::GetRecord(Err(e)) => {
                    warn!("Failed to get record: {:?}", e);
                }
                QueryResult::PutRecord(Err(e)) => {
                    warn!("Failed to store record: {:?}", e);
                }
                _ => {}
            },
            KademliaEvent::RoutingUpdated { peer, .. } => {
                debug!("Routing table updated for peer: {}", peer);
            }
            _ => {}
        }
        Ok(())
    }

    /// Handle identify events
    async fn handle_identify_event(&mut self, event: identify::Event) -> Result<()> {
        match event {
            identify::Event::Received { peer_id, info } => {
                debug!(
                    "Identified peer {} with {} addresses",
                    peer_id,
                    info.listen_addrs.len()
                );

                // Add addresses to Kademlia
                for addr in info.listen_addrs {
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Handle mDNS events
    async fn handle_mdns_event(&mut self, event: mdns::Event) -> Result<()> {
        match event {
            mdns::Event::Discovered(peers) => {
                for (peer_id, addr) in peers {
                    info!("Discovered peer via mDNS: {} at {}", peer_id, addr);
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());

                    // Try to dial the peer
                    if let Err(e) = self.swarm.dial(addr.clone()) {
                        warn!("Failed to dial {}: {}", addr, e);
                    }
                }
            }
            mdns::Event::Expired(peers) => {
                for (peer_id, addr) in peers {
                    debug!("mDNS peer expired: {} at {}", peer_id, addr);
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .remove_address(&peer_id, &addr);
                }
            }
        }
        Ok(())
    }

    /// Get statistics about the DHT
    pub fn stats(&self) -> DHTStats {
        let kademlia = &self.swarm.behaviour().kademlia;

        DHTStats {
            num_peers: self.swarm.connected_peers().count(),
            num_pending_queries: kademlia.queries().count(),
            routing_table_size: 0, // Would need to iterate through buckets
        }
    }
}

/// DHT statistics
#[derive(Debug, Clone)]
pub struct DHTStats {
    pub num_peers: usize,
    pub num_pending_queries: usize,
    pub routing_table_size: usize,
}

/// Helper function to get current Unix timestamp
fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Generate or load keypair
    let keypair = Keypair::generate_ed25519();
    let peer_id = PeerId::from(keypair.public());
    info!("Local peer ID: {}", peer_id);

    // Create DHT service
    let listen_addr: Multiaddr = "/ip4/0.0.0.0/tcp/0".parse()?;
    let mut dht = GitTorrentDHT::new(keypair.clone(), listen_addr).await?;

    // Example bootstrap nodes (would be configured in production)
    let bootstrap_nodes = vec![
        // Add your bootstrap nodes here
        // (peer_id, multiaddr)
    ];

    if !bootstrap_nodes.is_empty() {
        dht.bootstrap(bootstrap_nodes).await?;
    }

    // Example: Store a Git object
    let example_oid = "a" .repeat(64); // Example SHA256
    let example_data = b"example git object data".to_vec();
    dht.put_git_object(&example_oid, example_data).await?;

    // Example: Store a mutable record
    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let refs_data = b"refs/heads/main:abc123".to_vec();
    dht.put_mutable_record(&signing_key, refs_data, None).await?;

    // Run the DHT service
    info!("GitTorrent DHT service running...");
    dht.run().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_object_key() {
        let oid = "a".repeat(64);
        let key = GitObjectKey::from_git_oid(&oid).unwrap();
        assert_eq!(key.git_oid(), &oid);
    }

    #[test]
    fn test_mutable_record_signature() {
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let value = b"test value".to_vec();
        let sequence = 42;

        let record = MutableRecord::new(&signing_key, value, sequence, None).unwrap();
        assert!(record.verify().unwrap());

        // Tamper with the value
        let mut tampered = record.clone();
        tampered.value = b"tampered".to_vec();
        assert!(!tampered.verify().unwrap());
    }

    #[test]
    fn test_mutable_record_sequence() {
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let value = b"test".to_vec();

        let record1 = MutableRecord::new(&signing_key, value.clone(), 1, None).unwrap();
        let record2 = MutableRecord::new(&signing_key, value, 2, None).unwrap();

        assert!(record2.is_newer_than(&record1));
        assert!(!record1.is_newer_than(&record2));
    }
}