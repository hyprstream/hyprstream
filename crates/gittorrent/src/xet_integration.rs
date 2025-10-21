//! XET + Saorsa-Core integration for distributed Git storage
//!
//! This module provides the core integration between XET's chunking system
//! and saorsa-core's P2P distributed hash table for decentralized Git hosting.

use crate::{Result, Error};
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// XET MerkleHash representation (SHA-256 based)
/// This is a simplified version that will be replaced with actual XET imports
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MerkleHash {
    pub hash: [u8; 32],
}

impl MerkleHash {
    pub fn from_hex(hex: &str) -> Result<Self> {
        if hex.len() != 64 {
            return Err(Error::other("Invalid MerkleHash hex length"));
        }

        let mut hash = [0u8; 32];
        for i in 0..32 {
            let byte_hex = &hex[i*2..i*2+2];
            hash[i] = u8::from_str_radix(byte_hex, 16)
                .map_err(|_| Error::other("Invalid hex character in MerkleHash"))?;
        }

        Ok(MerkleHash { hash })
    }

    pub fn to_hex(&self) -> String {
        hex::encode(self.hash)
    }
}

/// Saorsa-Core content address (BLAKE3 based)
/// This mirrors the structure in saorsa-core but adapted for our needs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentAddress {
    /// BLAKE3 hash of the content
    pub root_hash: [u8; 32],
    /// Individual chunk hashes (for multi-chunk content)
    pub chunk_hashes: Vec<[u8; 32]>,
    /// Total content size in bytes
    pub total_size: u64,
    /// Number of chunks
    pub chunk_count: u32,
}

impl ContentAddress {
    /// Create a new content address from data bytes
    pub fn new(data: &[u8]) -> Self {
        // For now, use a simple hash. In production, this would use BLAKE3
        let hash = crate::crypto::hash::sha256(data);
        let blake3_hash = hash;

        Self {
            root_hash: blake3_hash,
            chunk_count: 1,
            chunk_hashes: vec![blake3_hash],
            total_size: data.len() as u64,
        }
    }
}

/// Chunk metadata stored in the DHT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Content address of the chunk
    pub content_addr: ContentAddress,
    /// Size of the chunk in bytes
    pub size: u64,
    /// List of nodes that have this chunk
    pub available_nodes: Vec<String>, // Node IDs
    /// Number of replicas in the network
    pub replication_count: u32,
    /// Last time this chunk was verified
    pub last_verified: SystemTime,
}

/// Repository metadata for GitTorrent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryMetadata {
    /// Repository hash (derived from repository URL or content)
    pub repo_hash: String,
    /// List of XET chunks in this repository
    pub xet_chunks: Vec<MerkleHash>,
    /// Git references (branch -> commit hash)
    pub git_refs: HashMap<String, String>,
    /// LFS objects managed by XET
    pub lfs_objects: Vec<String>,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

/// Bidirectional mapping between XET hashes and Saorsa-Core content addresses
pub struct HashBridge {
    /// XET SHA-256 → Saorsa BLAKE3 mapping
    xet_to_saorsa: RwLock<HashMap<MerkleHash, ContentAddress>>,
    /// Reverse mapping for lookups
    saorsa_to_xet: RwLock<HashMap<ContentAddress, MerkleHash>>,
}

impl HashBridge {
    /// Create a new hash bridge
    pub fn new() -> Self {
        Self {
            xet_to_saorsa: RwLock::new(HashMap::new()),
            saorsa_to_xet: RwLock::new(HashMap::new()),
        }
    }

    /// Convert XET chunk hash to saorsa-core content address
    pub async fn xet_to_saorsa(&self, merkle_hash: &MerkleHash) -> Option<ContentAddress> {
        let map = self.xet_to_saorsa.read().await;
        map.get(merkle_hash).cloned()
    }

    /// Convert saorsa-core address to XET hash for retrieval
    pub async fn saorsa_to_xet(&self, content_addr: &ContentAddress) -> Option<MerkleHash> {
        let map = self.saorsa_to_xet.read().await;
        map.get(content_addr).cloned()
    }

    /// Register new mapping when storing XET chunk in saorsa-core
    pub async fn register_mapping(&self, xet_hash: MerkleHash, saorsa_addr: ContentAddress) {
        let mut xet_map = self.xet_to_saorsa.write().await;
        let mut saorsa_map = self.saorsa_to_xet.write().await;

        xet_map.insert(xet_hash.clone(), saorsa_addr.clone());
        saorsa_map.insert(saorsa_addr, xet_hash);
    }

    /// Remove mapping (for cleanup)
    pub async fn remove_mapping(&self, xet_hash: &MerkleHash) {
        let mut xet_map = self.xet_to_saorsa.write().await;
        let mut saorsa_map = self.saorsa_to_xet.write().await;

        if let Some(saorsa_addr) = xet_map.remove(xet_hash) {
            saorsa_map.remove(&saorsa_addr);
        }
    }
}

impl Default for HashBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for chunk storage abstraction
/// This will be implemented by saorsa-core extensions
#[async_trait::async_trait]
pub trait ChunkStorage: Send + Sync {
    /// Store chunk data and return its content address
    async fn store(&self, data: &[u8]) -> Result<ContentAddress>;

    /// Retrieve chunk data by content address
    async fn retrieve(&self, addr: &ContentAddress) -> Result<Vec<u8>>;

    /// Check if chunk exists locally
    async fn exists(&self, addr: &ContentAddress) -> Result<bool>;

    /// Get chunk metadata
    async fn metadata(&self, addr: &ContentAddress) -> Result<ChunkMetadata>;
}

/// Local chunk storage implementation (placeholder)
pub struct LocalChunkStorage {
    storage_dir: std::path::PathBuf,
}

impl LocalChunkStorage {
    pub fn new(storage_dir: std::path::PathBuf) -> Self {
        Self { storage_dir }
    }
}

#[async_trait::async_trait]
impl ChunkStorage for LocalChunkStorage {
    async fn store(&self, data: &[u8]) -> Result<ContentAddress> {
        let content_addr = ContentAddress::new(data);

        // Create storage path based on content address
        let chunk_dir = self.storage_dir.join("chunks");
        tokio::fs::create_dir_all(&chunk_dir).await?;

        let chunk_path = chunk_dir.join(hex::encode(content_addr.root_hash));
        tokio::fs::write(chunk_path, data).await?;

        Ok(content_addr)
    }

    async fn retrieve(&self, addr: &ContentAddress) -> Result<Vec<u8>> {
        let chunk_path = self.storage_dir
            .join("chunks")
            .join(hex::encode(addr.root_hash));

        tokio::fs::read(chunk_path).await
            .map_err(Error::from)
    }

    async fn exists(&self, addr: &ContentAddress) -> Result<bool> {
        let chunk_path = self.storage_dir
            .join("chunks")
            .join(hex::encode(addr.root_hash));

        Ok(chunk_path.exists())
    }

    async fn metadata(&self, addr: &ContentAddress) -> Result<ChunkMetadata> {
        let chunk_path = self.storage_dir
            .join("chunks")
            .join(hex::encode(addr.root_hash));

        let metadata = tokio::fs::metadata(chunk_path).await
            .map_err(Error::from)?;

        Ok(ChunkMetadata {
            content_addr: addr.clone(),
            size: metadata.len(),
            available_nodes: vec!["local".to_string()],
            replication_count: 1,
            last_verified: SystemTime::now(),
        })
    }
}

/// Configuration for GitTorrent XET integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XetIntegrationConfig {
    /// Local storage directory for chunks
    pub storage_dir: std::path::PathBuf,
    /// Replication factor for chunks
    pub replication_factor: usize,
    /// Maximum chunk size (XET uses ~16KB)
    pub max_chunk_size: usize,
    /// DHT bootstrap nodes
    pub bootstrap_nodes: Vec<String>,
    /// XET CAS endpoint (fallback)
    pub xet_endpoint: Option<String>,
    /// Enable P2P chunk distribution
    pub enable_p2p: bool,
}

impl Default for XetIntegrationConfig {
    fn default() -> Self {
        Self {
            storage_dir: dirs::cache_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("gittorrent"),
            replication_factor: 3,
            max_chunk_size: 16 * 1024, // 16KB like XET
            bootstrap_nodes: vec![],
            xet_endpoint: Some("https://cas.xet.dev".to_string()),
            enable_p2p: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_merkle_hash_conversion() {
        let hex = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";
        let hash = MerkleHash::from_hex(hex).unwrap();
        assert_eq!(hash.to_hex(), hex);
    }

    #[test]
    fn test_content_address_creation() {
        let data = b"hello world";
        let addr = ContentAddress::new(data);
        assert_eq!(addr.total_size, 11);
        assert_eq!(addr.chunk_count, 1);
        assert_eq!(addr.chunk_hashes.len(), 1);
    }

    #[tokio::test]
    async fn test_hash_bridge() {
        let bridge = HashBridge::new();
        let xet_hash = MerkleHash::from_hex(
            "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        ).unwrap();
        let content_addr = ContentAddress::new(b"test data");

        // Test mapping registration
        bridge.register_mapping(xet_hash.clone(), content_addr.clone()).await;

        // Test forward lookup
        let result = bridge.xet_to_saorsa(&xet_hash).await;
        assert_eq!(result, Some(content_addr.clone()));

        // Test reverse lookup
        let result = bridge.saorsa_to_xet(&content_addr).await;
        assert_eq!(result, Some(xet_hash));
    }

    #[tokio::test]
    async fn test_local_chunk_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalChunkStorage::new(temp_dir.path().to_path_buf());

        let test_data = b"test chunk data";

        // Store chunk
        let addr = storage.store(test_data).await.unwrap();

        // Check existence
        assert!(storage.exists(&addr).await.unwrap());

        // Retrieve chunk
        let retrieved = storage.retrieve(&addr).await.unwrap();
        assert_eq!(retrieved, test_data);

        // Get metadata
        let metadata = storage.metadata(&addr).await.unwrap();
        assert_eq!(metadata.size, test_data.len() as u64);
    }
}