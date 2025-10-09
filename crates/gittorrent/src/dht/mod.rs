//! DHT implementation using libp2p Kademlia for SHA256 Git objects

use libp2p::{
    kad::{self, QueryResult, Record, RecordKey},
    swarm::{Swarm, SwarmEvent},
    PeerId, Multiaddr,
};
use multihash::Multihash;
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};
use futures::stream::StreamExt;
use crate::{error::Result, types::Sha256Hash};

pub mod behaviour;
pub mod storage;

pub use behaviour::GitTorrentBehaviour;
pub use storage::GitObjectStore;

/// Git SHA256 hash as a DHT key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GitObjectKey(pub Sha256Hash);

impl GitObjectKey {
    /// Create a new Git object key from SHA256 hash
    pub fn new(hash: Sha256Hash) -> Self {
        Self(hash)
    }

    /// Convert to libp2p RecordKey using multihash
    pub fn to_record_key(&self) -> RecordKey {
        // Create multihash from SHA2-256 (code 0x12)
        let multihash: Multihash<64> = Multihash::wrap(0x12, &self.0.to_bytes()).expect("32-byte hash should fit");
        RecordKey::new(&multihash.to_bytes())
    }

    /// Get the SHA256 hash
    pub fn hash(&self) -> &Sha256Hash {
        &self.0
    }
}

/// Git object record in the DHT
#[derive(Debug, Clone)]
pub struct GitObjectRecord {
    pub key: GitObjectKey,
    pub data: Vec<u8>,
    pub provider: Option<PeerId>,
}

impl GitObjectRecord {
    pub fn new(key: GitObjectKey, data: Vec<u8>) -> Self {
        Self {
            key,
            data,
            provider: None,
        }
    }

    /// Convert to libp2p Record
    pub fn to_record(&self) -> Record {
        Record {
            key: self.key.to_record_key(),
            value: self.data.clone(),
            publisher: self.provider,
            expires: None,
        }
    }
}

/// DHT command for async operations
#[derive(Debug)]
pub enum DhtCommand {
    /// Store a Git object in the DHT
    PutObject {
        record: GitObjectRecord,
        response: oneshot::Sender<Result<()>>,
    },
    /// Retrieve a Git object from the DHT
    GetObject {
        key: GitObjectKey,
        response: oneshot::Sender<Result<Option<GitObjectRecord>>>,
    },
    /// Announce as provider for a Git object
    Provide {
        key: GitObjectKey,
        response: oneshot::Sender<Result<()>>,
    },
    /// Find providers for a Git object
    GetProviders {
        key: GitObjectKey,
        response: oneshot::Sender<Result<Vec<PeerId>>>,
    },
    /// Bootstrap the DHT
    Bootstrap {
        peers: Vec<Multiaddr>,
        response: oneshot::Sender<Result<()>>,
    },
}

/// Pending query tracking
#[derive(Debug)]
enum PendingQuery {
    GetObject {
        key: GitObjectKey,
        response: oneshot::Sender<Result<Option<GitObjectRecord>>>,
    },
    GetProviders {
        key: GitObjectKey,
        response: oneshot::Sender<Result<Vec<PeerId>>>,
    },
}

/// Main DHT service using libp2p Kademlia
pub struct GitTorrentDht {
    command_tx: mpsc::UnboundedSender<DhtCommand>,
    _task_handle: tokio::task::JoinHandle<()>,
}

impl GitTorrentDht {
    /// Create a new DHT service
    pub async fn new(p2p_port: u16) -> Result<Self> {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel();

        // Create swarm with SwarmBuilder
        let mut swarm = libp2p::SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                libp2p::tcp::Config::default().nodelay(true),
                libp2p::noise::Config::new,
                libp2p::yamux::Config::default,
            )?
            .with_behaviour(|key| GitTorrentBehaviour::new_with_keypair(key).unwrap())?
            .build();

        let local_peer_id = *swarm.local_peer_id();
        tracing::info!("Starting GitTorrent DHT with peer ID: {}", local_peer_id);

        // Listen on all interfaces with specified port (0 = random)
        let listen_addr = format!("/ip4/0.0.0.0/tcp/{}", p2p_port);
        swarm.listen_on(listen_addr.parse()?)?;

        // Spawn background task
        let task_handle = tokio::spawn(async move {
            let mut pending_queries = HashMap::new();

            loop {
                tokio::select! {
                    // Handle swarm events
                    event = swarm.select_next_some() => {
                        // Special handling for events that need swarm access
                        match &event {
                            SwarmEvent::Behaviour(crate::dht::behaviour::GitTorrentBehaviourEvent::Identify(
                                libp2p::identify::Event::Received { peer_id, info, .. }
                            )) => {
                                tracing::info!("Identified peer {} with protocols: {:?}", peer_id, info.protocols);
                                // Add the peer's listen addresses to Kademlia routing table
                                for addr in &info.listen_addrs {
                                    tracing::debug!("Adding address {} for peer {} to Kademlia", addr, peer_id);
                                    swarm.behaviour_mut().kademlia.add_address(peer_id, addr.clone());
                                }
                            }
                            SwarmEvent::Behaviour(crate::dht::behaviour::GitTorrentBehaviourEvent::Mdns(
                                libp2p::mdns::Event::Discovered(peers)
                            )) => {
                                // Add mDNS discovered peers to Kademlia
                                for (peer_id, addr) in peers {
                                    tracing::info!("Adding mDNS discovered peer {} at {} to Kademlia", peer_id, addr);
                                    swarm.behaviour_mut().kademlia.add_address(peer_id, addr.clone());
                                }
                            }
                            _ => {}
                        }
                        Self::handle_swarm_event(event, &mut pending_queries).await;
                    }

                    // Handle commands
                    Some(command) = command_rx.recv() => {
                        Self::handle_command(&mut swarm, command, &mut pending_queries).await;
                    }
                }
            }
        });

        Ok(Self {
            command_tx,
            _task_handle: task_handle,
        })
    }

    /// Store a Git object in the DHT
    pub async fn put_object(&self, record: GitObjectRecord) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.command_tx.send(DhtCommand::PutObject { record, response: tx })?;
        rx.await?
    }

    /// Retrieve a Git object from the DHT
    pub async fn get_object(&self, key: GitObjectKey) -> Result<Option<GitObjectRecord>> {
        let (tx, rx) = oneshot::channel();
        self.command_tx.send(DhtCommand::GetObject { key, response: tx })?;
        rx.await?
    }

    /// Announce as provider for a Git object
    pub async fn provide(&self, key: GitObjectKey) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.command_tx.send(DhtCommand::Provide { key, response: tx })?;
        rx.await?
    }

    /// Find providers for a Git object
    pub async fn get_providers(&self, key: GitObjectKey) -> Result<Vec<PeerId>> {
        let (tx, rx) = oneshot::channel();
        self.command_tx.send(DhtCommand::GetProviders { key, response: tx })?;
        rx.await?
    }

    /// Bootstrap the DHT with known peers
    ///
    /// Note: Bootstrap addresses should include the peer ID in the format:
    /// `/ip4/127.0.0.1/tcp/4001/p2p/12D3KooWExample...`
    ///
    /// Without peer IDs, the bootstrap will attempt to dial but cannot add peers
    /// to the routing table until the Identify protocol completes.
    pub async fn bootstrap(&self, peers: Vec<Multiaddr>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.command_tx.send(DhtCommand::Bootstrap { peers, response: tx })?;
        rx.await?
    }

    /// Helper to create a proper bootstrap address with peer ID
    pub fn create_bootstrap_addr(ip: &str, port: u16, peer_id: &PeerId) -> Multiaddr {
        format!("/ip4/{}/tcp/{}/p2p/{}", ip, port, peer_id).parse().unwrap()
    }

    async fn handle_swarm_event(
        event: SwarmEvent<crate::dht::behaviour::GitTorrentBehaviourEvent>,
        pending_queries: &mut HashMap<libp2p::kad::QueryId, PendingQuery>,
    ) {
        use crate::dht::behaviour::GitTorrentBehaviourEvent;

        match event {
            SwarmEvent::Behaviour(GitTorrentBehaviourEvent::Kademlia(kad_event)) => {
                match kad_event {
                    kad::Event::OutboundQueryProgressed { id, result, .. } => {
                        if let Some(pending) = pending_queries.remove(&id) {
                            match (result, pending) {
                                (QueryResult::GetRecord(Ok(record)), PendingQuery::GetObject { key, response }) => {
                                    // Handle GetRecordOk enum variants
                                    match record {
                                        libp2p::kad::GetRecordOk::FoundRecord(peer_record) => {
                                            let git_record = GitObjectRecord {
                                                key,
                                                data: peer_record.record.value,
                                                provider: peer_record.record.publisher,
                                            };
                                            let _ = response.send(Ok(Some(git_record)));
                                        },
                                        _ => {
                                            let _ = response.send(Ok(None));
                                        }
                                    }
                                }
                                (QueryResult::GetRecord(Err(e)), PendingQuery::GetObject { response, .. }) => {
                                    let _ = response.send(Err(crate::Error::Dht(format!("Get record failed: {:?}", e))));
                                }
                                (QueryResult::GetProviders(Ok(providers)), PendingQuery::GetProviders { response, .. }) => {
                                    // Handle GetProvidersOk enum variants
                                    match providers {
                                        libp2p::kad::GetProvidersOk::FoundProviders { providers, .. } => {
                                            let _ = response.send(Ok(providers.into_iter().collect()));
                                        },
                                        _ => {
                                            let _ = response.send(Ok(vec![]));
                                        }
                                    }
                                }
                                (QueryResult::GetProviders(Err(e)), PendingQuery::GetProviders { response, .. }) => {
                                    let _ = response.send(Err(crate::Error::Dht(format!("Get providers failed: {:?}", e))));
                                }
                                _ => {
                                    tracing::warn!("Mismatched query result and pending query type");
                                }
                            }
                        }
                    }
                    _ => {
                        tracing::debug!("Other Kademlia event: {:?}", kad_event);
                    }
                }
            }
            SwarmEvent::Behaviour(GitTorrentBehaviourEvent::Mdns(mdns_event)) => {
                match mdns_event {
                    libp2p::mdns::Event::Discovered(peers) => {
                        for (peer_id, addr) in peers {
                            tracing::info!("Discovered peer via mDNS: {} at {}", peer_id, addr);
                            // mDNS discovered peers are automatically added to Kademlia by libp2p
                        }
                    }
                    libp2p::mdns::Event::Expired(peers) => {
                        for (peer_id, _addr) in peers {
                            tracing::debug!("mDNS peer expired: {}", peer_id);
                        }
                    }
                }
            }
            SwarmEvent::Behaviour(GitTorrentBehaviourEvent::Identify(identify_event)) => {
                match identify_event {
                    libp2p::identify::Event::Received { peer_id, .. } => {
                        // Main handling is done in the event loop before this function
                        tracing::debug!("Processed identify event for peer {}", peer_id);
                    }
                    libp2p::identify::Event::Sent { peer_id, .. } => {
                        tracing::debug!("Sent identify info to peer {}", peer_id);
                    }
                    libp2p::identify::Event::Pushed { peer_id, .. } => {
                        tracing::debug!("Pushed identify info to peer {}", peer_id);
                    }
                    libp2p::identify::Event::Error { peer_id, error, .. } => {
                        tracing::warn!("Identify error with peer {}: {}", peer_id, error);
                    }
                }
            }
            SwarmEvent::NewListenAddr { address, .. } => {
                tracing::info!("DHT listening on: {}", address);
            }
            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                tracing::debug!("Connected to peer: {}", peer_id);
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                tracing::debug!("Disconnected from peer: {}", peer_id);
            }
            _ => {}
        }
    }

    async fn handle_command(
        swarm: &mut Swarm<GitTorrentBehaviour>,
        command: DhtCommand,
        pending_queries: &mut HashMap<libp2p::kad::QueryId, PendingQuery>,
    ) {
        match command {
            DhtCommand::PutObject { record, response } => {
                let result = swarm.behaviour_mut().kademlia.put_record(
                    record.to_record(),
                    libp2p::kad::Quorum::One,
                );

                let result = match result {
                    Ok(_) => Ok(()),
                    Err(e) => Err(crate::Error::Dht(format!("Failed to put record: {}", e))),
                };

                let _ = response.send(result);
            }

            DhtCommand::GetObject { key, response } => {
                let query_id = swarm.behaviour_mut().kademlia.get_record(key.to_record_key());
                // Store the pending query for later completion
                pending_queries.insert(query_id, PendingQuery::GetObject { key, response });
            }

            DhtCommand::Provide { key, response } => {
                let result = swarm.behaviour_mut().kademlia.start_providing(key.to_record_key());

                let result = match result {
                    Ok(_) => Ok(()),
                    Err(e) => Err(crate::Error::Dht(format!("Failed to provide: {}", e))),
                };

                let _ = response.send(result);
            }

            DhtCommand::GetProviders { key, response } => {
                let query_id = swarm.behaviour_mut().kademlia.get_providers(key.to_record_key());
                // Store the pending query for later completion
                pending_queries.insert(query_id, PendingQuery::GetProviders { key, response });
            }

            DhtCommand::Bootstrap { peers, response } => {
                // First, add bootstrap peers to Kademlia's routing table
                // This requires extracting the peer ID from the multiaddress
                let mut added_peers = 0;
                for addr in peers {
                    // Try to extract peer ID from the multiaddress
                    // Expected format: /ip4/.../tcp/.../p2p/<peer_id>
                    if let Some(peer_id) = addr.iter().find_map(|proto| {
                        if let libp2p::multiaddr::Protocol::P2p(peer_id) = proto {
                            Some(peer_id)
                        } else {
                            None
                        }
                    }) {
                        // Add the peer to Kademlia's routing table
                        // First, create address without the p2p component for adding to routing table
                        let mut addr_without_p2p = addr.clone();
                        addr_without_p2p.pop(); // Remove the p2p component

                        swarm.behaviour_mut().kademlia.add_address(&peer_id, addr_without_p2p.clone());
                        added_peers += 1;

                        // Now dial the peer to establish connection
                        if let Err(e) = swarm.dial(addr.clone()) {
                            tracing::warn!("Failed to dial bootstrap peer {}: {}", addr, e);
                        } else {
                            tracing::info!("Dialing bootstrap peer: {} ({})", peer_id, addr);
                        }
                    } else {
                        // If no peer ID in address, try to dial anyway (might work with identify)
                        tracing::warn!("Bootstrap address {} doesn't contain peer ID, attempting dial anyway", addr);
                        if let Err(e) = swarm.dial(addr.clone()) {
                            tracing::warn!("Failed to dial bootstrap peer {}: {}", addr, e);
                        }
                    }
                }

                // Only attempt bootstrap if we added at least one peer
                let result = if added_peers > 0 {
                    match swarm.behaviour_mut().kademlia.bootstrap() {
                        Ok(query_id) => {
                            tracing::info!("Bootstrap started with query ID: {:?}, added {} peers", query_id, added_peers);
                            Ok(())
                        },
                        Err(e) => {
                            tracing::error!("Failed to bootstrap: {}", e);
                            Err(crate::Error::Dht(format!("Failed to bootstrap: {}", e)))
                        }
                    }
                } else {
                    tracing::warn!("No valid bootstrap peers with peer IDs found");
                    Err(crate::Error::Dht("No valid bootstrap peers found".to_string()))
                };

                let _ = response.send(result);
            }
        }
    }
}