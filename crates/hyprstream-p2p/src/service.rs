//! GitTorrent Service - Core P2P Git Repository Service
//!
//! This module implements the main GitTorrent service that orchestrates:
//! - Git repository operations using git2
//! - Content-addressed object storage/transfer via iroh-blobs + the mainline
//!   locator (the libp2p Kademlia DHT was retired in F3, #901)
//! - CID-based content addressing and distribution

use crate::crypto::hash::sha256_git;
use crate::{ContentCid, Error, GitTorrentUrl, Result, Sha256Hash};

use crate::blobs::IrohBlobStore;
use crate::locator::{Cid512, MainlineLocator, PeerContact, CID512_LEN};
use async_trait::async_trait;
use iroh_blobs::Hash as Blake3Hash;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for GitTorrent service
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GitTorrentConfig {
    /// Storage directory for repositories
    pub storage_dir: PathBuf,
    /// Bootstrap nodes for P2P network
    pub bootstrap_nodes: Vec<String>,
    /// Enable automatic object discovery
    pub auto_discovery: bool,
    /// Local bind address for daemon
    pub bind_address: String,
    /// Local bind port for daemon
    pub bind_port: u16,
    /// Port advertised to mainline as this node's iroh-blobs object-service
    /// port (0 = let remote nodes infer via BEP5 implied_port).
    pub p2p_port: u16,
}

impl Default for GitTorrentConfig {
    fn default() -> Self {
        Self {
            storage_dir: dirs::cache_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("gittorrent"),
            bootstrap_nodes: vec![
                // No default bootstrap nodes - user must specify them
                // or run in standalone mode for the first network node
            ],
            auto_discovery: true,
            bind_address: "127.0.0.1".to_owned(),
            bind_port: 8080,
            p2p_port: 0, // 0 = let remote nodes infer the port (BEP5 implied_port)
        }
    }
}

impl GitTorrentConfig {
    /// Get the configuration directory
    fn config_dir() -> Result<PathBuf> {
        dirs::config_dir()
            .map(|dir| dir.join("gittorrent"))
            .ok_or_else(|| Error::other("Unable to determine config directory"))
    }

    /// Create a configuration builder with the standard priority stack
    ///
    /// Returns a `config::ConfigBuilder` that developers can extend with additional sources
    /// before building the final configuration.
    ///
    /// **Default Priority Order (highest to lowest):**
    /// 1. Environment variables (GITTORRENT_* prefix)
    /// 2. Config file (~/.config/gittorrent/git-remote-gittorrent.*)
    /// 3. Default values
    ///
    /// # Basic Usage
    /// ```ignore
    /// use hyprstream_p2p::service::GitTorrentConfig;
    ///
    /// // Simple case - use defaults
    /// let config: GitTorrentConfig = GitTorrentConfig::builder()?
    ///     .build()?
    ///     .try_deserialize()?;
    /// ```
    ///
    /// # Extended Usage - Add Custom Sources
    /// ```ignore
    /// use hyprstream_p2p::service::GitTorrentConfig;
    /// use config::File;
    ///
    /// // Add custom config source with highest priority
    /// let config: GitTorrentConfig = GitTorrentConfig::builder()?
    ///     .add_source(File::with_name("/custom/config"))  // Highest priority
    ///     .build()?
    ///     .try_deserialize()?;
    /// ```
    ///
    /// # Programmatic Overrides
    /// ```ignore
    /// use hyprstream_p2p::service::GitTorrentConfig;
    ///
    /// // Override specific values programmatically
    /// let config: GitTorrentConfig = GitTorrentConfig::builder()?
    ///     .set_override("p2p_port", 4001)?  // Highest priority
    ///     .build()?
    ///     .try_deserialize()?;
    /// ```
    ///
    /// # Supported Environment Variables
    /// - `GITTORRENT_P2P_PORT`: port advertised to the mainline locator
    /// - `GITTORRENT_BOOTSTRAP_NODES`: Comma-separated list of bootstrap nodes
    /// - `GITTORRENT_STORAGE_DIR`: Storage directory for repositories
    /// - `GITTORRENT_BIND_ADDRESS`: Local bind address for daemon
    /// - `GITTORRENT_BIND_PORT`: Local bind port for daemon
    /// - `GITTORRENT_AUTO_DISCOVERY`: Enable/disable automatic discovery (true/false)
    pub fn builder() -> Result<config::ConfigBuilder<config::builder::DefaultState>> {
        let config_dir = Self::config_dir()?;
        let config_file = config_dir.join("git-remote-gittorrent");

        Ok(config::Config::builder()
            // Start with defaults (lowest priority)
            .add_source(config::Config::try_from(&Self::default())?)
            // Layer config file (optional, medium priority)
            .add_source(config::File::from(config_file).required(false))
            // Layer environment variables (high priority)
            .add_source(
                config::Environment::with_prefix("GITTORRENT")
                    .separator("_")
                    .list_separator(","),
            ))
    }

    /// Convenience method to load configuration with defaults
    ///
    /// This is equivalent to `GitTorrentConfig::builder()?.build()?.try_deserialize()?`
    ///
    /// # Basic Usage
    /// ```ignore
    /// use hyprstream_p2p::service::GitTorrentConfig;
    ///
    /// let config = GitTorrentConfig::load()?;
    /// ```
    pub fn load() -> Result<Self> {
        let config = Self::builder()?
            .build()
            .map_err(|e| Error::other(format!("Failed to build config: {e}")))?;

        config
            .try_deserialize()
            .map_err(|e| Error::other(format!("Failed to deserialize config: {e}")))
    }
}

/// Statistics about the GitTorrent service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStats {
    /// Number of known repositories
    pub repository_count: usize,
    /// Number of stored Git objects
    pub object_count: usize,
    /// Total bytes stored
    pub total_bytes: u64,
    /// Number of connected peers
    pub peer_count: usize,
}

/// Repository metadata for object-based storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryMetadata {
    /// Repository name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Remote URL (if any)
    pub remote_url: Option<String>,
    /// List of branches and their HEAD commits
    pub branches: HashMap<String, Sha256Hash>,

    // Object-based fields
    /// All object hashes in this repository
    pub object_hashes: Vec<Sha256Hash>,
    /// Root commit objects for each branch
    pub root_commits: HashMap<String, Sha256Hash>,
    /// Repository format version (for compatibility)
    pub format_version: u32,
    /// Object count by type
    pub object_stats: ObjectStats,

    /// Total size in bytes
    pub size_bytes: u64,
    /// Last updated timestamp
    pub last_updated: u64,
    /// Repository publisher/owner (optional)
    pub publisher: Option<String>,
    /// Git references information
    pub refs: Vec<String>,
    /// LFS chunks if any (for future LFS support)
    pub lfs_chunks: Vec<String>,
}

/// The bytes of an object fetched by CID, paired with their commit
/// provenance so callers can decide whether persistence is safe.
///
/// `committed == true` means the bytes are already durable in the local
/// validated store (the in-memory cache or the blob plane) or were
/// self-verified against the CID and committed during this fetch — safe
/// for the caller to treat as validated.
///
/// `committed == false` means the bytes were returned by a remote
/// provider under a CID that cannot self-verify them (e.g. an XET Merkle
/// CID, which addresses the reconstruction DAG rather than the raw file).
/// The service deliberately does NOT commit such bytes (it cannot prove
/// they match the CID); the caller MUST validate them end-to-end (e.g.
/// pointer size + SHA-256) before persisting or treating them as
/// committed. Conflating this case with `committed == true` lets
/// unvalidated provider bytes leak past a caller's persistence gate.
#[derive(Debug, Clone)]
pub struct ObjectFetch {
    /// The fetched object bytes.
    pub data: Vec<u8>,
    /// Whether the bytes are committed to the local validated store.
    pub committed: bool,
}

/// Statistics about objects in a repository
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ObjectStats {
    pub commit_count: usize,
    pub tree_count: usize,
    pub blob_count: usize,
    pub tag_count: usize,
}

/// Main GitTorrent service
pub struct GitTorrentService {
    /// iroh-blobs/mainline object plane backing the `put_object`/`get_object`
    /// facade (F1–F3, epic #880 Track F).
    object_plane: ObjectPlane,
    /// Local repository metadata
    repositories: Arc<RwLock<HashMap<String, RepositoryMetadata>>>,
    /// Object cache keyed by the complete content CID.
    object_cache: Arc<RwLock<HashMap<ContentCid, Vec<u8>>>>,
}

struct ObjectPlane {
    blobs: Arc<IrohBlobStore>,
    locator: Arc<dyn ObjectLocator>,
    fetcher: Arc<dyn RemoteBlobFetcher>,
    announce_port: Option<u16>,
}

#[async_trait]
trait ObjectLocator: Send + Sync {
    async fn announce(&self, cid: &Cid512, port: Option<u16>) -> Result<()>;
    async fn providers(&self, cid: &Cid512) -> Result<Vec<PeerContact>>;
}

#[async_trait]
impl ObjectLocator for MainlineLocator {
    async fn announce(&self, cid: &Cid512, port: Option<u16>) -> Result<()> {
        MainlineLocator::announce(self, cid, port).await
    }

    async fn providers(&self, cid: &Cid512) -> Result<Vec<PeerContact>> {
        MainlineLocator::providers(self, cid).await
    }
}

#[async_trait]
trait RemoteBlobFetcher: Send + Sync {
    async fn fetch(&self, cid: &ContentCid, providers: Vec<PeerContact>)
        -> Result<Option<Vec<u8>>>;
}

#[derive(Debug, Default)]
struct IrohMainlineBlobFetcher;

#[cfg(test)]
#[derive(Debug, Default)]
struct NoopObjectLocator;

#[cfg(test)]
#[async_trait]
impl ObjectLocator for NoopObjectLocator {
    async fn announce(&self, _cid: &Cid512, _port: Option<u16>) -> Result<()> {
        Ok(())
    }

    async fn providers(&self, _cid: &Cid512) -> Result<Vec<PeerContact>> {
        Ok(Vec::new())
    }
}

#[async_trait]
impl RemoteBlobFetcher for IrohMainlineBlobFetcher {
    async fn fetch(
        &self,
        cid: &ContentCid,
        providers: Vec<PeerContact>,
    ) -> Result<Option<Vec<u8>>> {
        // C2 currently exposes BEP5 socket contacts. The iroh-blobs downloader
        // needs authenticated iroh EndpointIds, so this stays the single adapter
        // point until the locator carries endpoint identity metadata.
        if !providers.is_empty() {
            tracing::debug!(
                "found {} mainline providers for content CID {cid}, but endpoint-id fetch is not wired yet",
                providers.len(),
            );
        }
        Ok(None)
    }
}

fn locator_cid_from_blake3(hash: Blake3Hash) -> Cid512 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"hyprstream.gittorrent.iroh-blobs.locator.v1");
    hasher.update(hash.as_bytes());

    let mut cid = [0u8; CID512_LEN];
    hasher.finalize_xof().fill(&mut cid);
    Cid512::from_bytes(cid)
}

fn locator_cid_from_content(cid: &ContentCid) -> Cid512 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"hyprstream.gittorrent.content-cid.locator.v1");
    // Hash the complete canonical CID, including its codec and multihash
    // algorithm. Equal-width digests from different domains cannot alias.
    hasher.update(cid.as_str().as_bytes());

    let mut cid = [0u8; CID512_LEN];
    hasher.finalize_xof().fill(&mut cid);
    Cid512::from_bytes(cid)
}

impl GitTorrentService {
    /// Create a new GitTorrent service
    pub async fn new(config: GitTorrentConfig) -> Result<Self> {
        // Ensure storage directory exists
        tokio::fs::create_dir_all(&config.storage_dir).await?;

        // iroh-blobs + mainline object plane (the sole backend after F3).
        // Tests use an in-memory store + a no-op locator so they never touch
        // the network or the filesystem outside `storage_dir`.
        let object_plane = ObjectPlane {
            #[cfg(not(test))]
            blobs: Arc::new(IrohBlobStore::open_fs(config.storage_dir.join("iroh-blobs")).await?),
            #[cfg(test)]
            blobs: Arc::new(IrohBlobStore::new_memory()),
            #[cfg(not(test))]
            locator: Arc::new(MainlineLocator::new()?),
            #[cfg(test)]
            locator: Arc::new(NoopObjectLocator),
            fetcher: Arc::new(IrohMainlineBlobFetcher),
            announce_port: config.auto_discovery.then_some(config.p2p_port),
        };

        // `bootstrap_nodes` configured a libp2p Kademlia bootstrap that no
        // longer exists; the mainline locator bootstraps itself against the
        // default BEP5 routers, so there is nothing to dial here.
        if !config.bootstrap_nodes.is_empty() {
            tracing::info!(
                "{} bootstrap node(s) configured; mainline locator self-bootstraps against default BEP5 routers",
                config.bootstrap_nodes.len()
            );
        }

        Ok(Self {
            object_plane,
            repositories: Arc::new(RwLock::new(HashMap::new())),
            object_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    #[cfg(any(test, feature = "testing"))]
    async fn new_with_object_plane(
        config: GitTorrentConfig,
        locator: Arc<dyn ObjectLocator>,
        fetcher: Arc<dyn RemoteBlobFetcher>,
    ) -> Result<Self> {
        tokio::fs::create_dir_all(&config.storage_dir).await?;
        let object_plane = ObjectPlane {
            blobs: Arc::new(IrohBlobStore::open_fs(config.storage_dir.join("iroh-blobs")).await?),
            locator,
            fetcher,
            announce_port: config.auto_discovery.then_some(config.p2p_port),
        };

        Ok(Self {
            object_plane,
            repositories: Arc::new(RwLock::new(HashMap::new())),
            object_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Add a repository to the service
    pub async fn add_repository(&self, name: String, path: &Path) -> Result<()> {
        // Scan repository for SHA256 objects
        let metadata = self.scan_repository(path).await?;

        // Store repository metadata
        let mut repos = self.repositories.write().await;
        repos.insert(name.clone(), metadata);

        tracing::info!("Added repository: {}", name);
        Ok(())
    }

    /// Get a Git object by SHA256 hash
    pub async fn get_object(&self, hash: &Sha256Hash) -> Result<Option<Vec<u8>>> {
        let cid = ContentCid::git_sha256(hash)?;
        self.get_object_by_cid(&cid).await
    }

    /// Get an object by its canonical content CID.
    ///
    /// Discards commit provenance. Callers that persist fetched bytes
    /// (e.g. the XET smudge path, which must validate before persisting)
    /// must use [`get_object_by_cid_with_provenance`] instead; treating a
    /// provider-fetched, non-self-verifying result as already-validated
    /// bypasses the caller's persistence gate.
    pub async fn get_object_by_cid(&self, cid: &ContentCid) -> Result<Option<Vec<u8>>> {
        Ok(self
            .get_object_by_cid_with_provenance(cid)
            .await?
            .map(|fetch| fetch.data))
    }

    /// Get an object by its canonical content CID, along with its commit
    /// provenance.
    ///
    /// See [`ObjectFetch`] for the meaning of `committed`. The provenance
    /// is structural: it is derived from which store produced the bytes
    /// and whether the CID can self-verify them, never from a caller
    /// assertion. A non-`committed` result must not be persisted without
    /// end-to-end validation.
    pub async fn get_object_by_cid_with_provenance(
        &self,
        cid: &ContentCid,
    ) -> Result<Option<ObjectFetch>> {
        // Check local cache first. The cache only ever holds committed
        // bytes (see `put_object_by_cid` and the self-verifying provider
        // path below), so a hit is committed by construction.
        {
            let cache = self.object_cache.read().await;
            if let Some(data) = cache.get(cid) {
                return Ok(Some(ObjectFetch {
                    data: data.clone(),
                    committed: true,
                }));
            }
        }

        if let Some(data) = self.object_plane.blobs.get_object_by_cid(cid).await? {
            let mut cache = self.object_cache.write().await;
            cache.insert(cid.clone(), data.clone());
            return Ok(Some(ObjectFetch {
                data,
                committed: true,
            }));
        }

        let rendezvous = locator_cid_from_content(cid);
        let providers = self.object_plane.locator.providers(&rendezvous).await?;
        if providers.is_empty() {
            return Ok(None);
        }

        if let Some(data) = self.object_plane.fetcher.fetch(cid, providers).await? {
            // A raw Git CID self-verifies inside `put_object_by_cid`, so
            // provider bytes can be committed here. A XET Merkle CID addresses
            // the reconstruction DAG rather than the raw file, so this layer
            // cannot verify the fetched bytes: return them uncommitted and
            // leave persistence to the caller, which must validate them
            // (pointer size + SHA-256) before calling `put_object_by_cid`.
            let decoded = cid.decoded()?;
            let self_verifying = matches!(
                (decoded.codec, decoded.multihash.algo),
                (
                    hyprstream_rpc::cid::Codec::GitRaw,
                    hyprstream_rpc::cid::HashAlgo::Sha2_256
                )
            );
            if !self_verifying {
                return Ok(Some(ObjectFetch {
                    data,
                    committed: false,
                }));
            }
            self.object_plane
                .blobs
                .put_object_by_cid(cid.clone(), data.clone())
                .await?;
            let mut cache = self.object_cache.write().await;
            cache.insert(cid.clone(), data.clone());
            return Ok(Some(ObjectFetch {
                data,
                committed: true,
            }));
        }

        Ok(None)
    }

    /// Store a Git object, returning its SHA256 hash (the consumer-facing key).
    pub async fn put_object(&self, data: Vec<u8>) -> Result<Sha256Hash> {
        let hash = sha256_git(&data)?;
        let cid = ContentCid::git_sha256(&hash)?;
        self.put_object_by_cid(cid, data).await?;
        Ok(hash)
    }

    /// Store an object under a canonical content CID.
    pub async fn put_object_by_cid(&self, cid: ContentCid, data: Vec<u8>) -> Result<()> {
        self.object_plane
            .blobs
            .put_object_by_cid(cid.clone(), data.clone())
            .await?;
        let blake3 = self
            .object_plane
            .blobs
            .blake3_of_cid(&cid)
            .await
            .ok_or_else(|| Error::not_found(format!("missing blake3 index for {cid}")))?;
        let blob_rendezvous = locator_cid_from_blake3(blake3);

        if let Some(port) = self.object_plane.announce_port {
            let content_rendezvous = locator_cid_from_content(&cid);
            self.object_plane
                .locator
                .announce(&blob_rendezvous, Some(port))
                .await?;
            self.object_plane
                .locator
                .announce(&content_rendezvous, Some(port))
                .await?;
        }

        let mut cache = self.object_cache.write().await;
        cache.insert(cid.clone(), data);

        tracing::debug!("Stored object with CID: {cid}");
        Ok(())
    }

    /// Store all objects from a repository individually through the object
    /// facade (iroh-blobs + mainline announce).
    pub async fn store_repository_objects(&self, repo_path: &Path) -> Result<Vec<Sha256Hash>> {
        use crate::git::objects::extract_objects;

        let objects = extract_objects(repo_path).await?;
        let mut stored_hashes = Vec::new();

        tracing::info!("Storing {} objects from repository", objects.len());

        for obj in objects {
            self.put_object(obj.data).await?;
            stored_hashes.push(obj.hash);
        }

        tracing::info!("Successfully stored {} objects", stored_hashes.len());
        Ok(stored_hashes)
    }

    /// Store objects for a specific commit and its history through the object
    /// facade.
    pub async fn store_commit_objects(
        &self,
        repo_path: &Path,
        commit_hash: &str,
    ) -> Result<Vec<Sha256Hash>> {
        use crate::git::objects::extract_commit_objects;

        let objects = extract_commit_objects(repo_path, commit_hash).await?;
        let mut stored_hashes = Vec::new();

        tracing::info!(
            "Storing {} objects for commit {}",
            objects.len(),
            commit_hash
        );

        for obj in objects {
            self.put_object(obj.data).await?;
            stored_hashes.push(obj.hash);
        }

        Ok(stored_hashes)
    }

    /// Reconstruct repository from individual objects
    pub async fn reconstruct_repository(
        &self,
        object_hashes: &[Sha256Hash],
        target_path: &Path,
    ) -> Result<()> {
        use crate::git::objects::{write_objects, GitObject, GitObjectType};

        // Create repository directory
        tokio::fs::create_dir_all(target_path).await?;

        // Initialize git repository
        let _repo = git2::Repository::init(target_path)?;

        // Collect all objects
        let mut git_objects = Vec::new();
        let mut missing_objects = Vec::new();

        for hash in object_hashes {
            if let Some(data) = self.get_object(hash).await? {
                // Parse object type from git format
                if let Some(null_pos) = data.iter().position(|&b| b == 0) {
                    let header = &data[..null_pos];
                    if let Ok(header_str) = std::str::from_utf8(header) {
                        let parts: Vec<&str> = header_str.split(' ').collect();
                        if parts.len() == 2 {
                            let obj_type = match parts[0] {
                                "commit" => GitObjectType::Commit,
                                "tree" => GitObjectType::Tree,
                                "tag" => GitObjectType::Tag,
                                // "blob" or unknown types default to Blob
                                _ => GitObjectType::Blob,
                            };

                            let size = parts[1].parse().unwrap_or(0);

                            git_objects.push(GitObject {
                                hash: hash.clone(),
                                object_type: obj_type,
                                data,
                                size,
                            });
                        }
                    }
                }
            } else {
                missing_objects.push(hash);
            }
        }

        if !missing_objects.is_empty() {
            tracing::warn!(
                "Missing {} objects during reconstruction",
                missing_objects.len()
            );
            // The object facade (`get_object`) already consults the local
            // iroh-blobs store plus the mainline locator before reporting an
            // object missing, so there is no separate peer-fetch path here.
        }

        // Write objects to repository
        write_objects(target_path, &git_objects).await?;

        tracing::info!(
            "Reconstructed repository with {} objects at {:?}",
            git_objects.len(),
            target_path
        );
        Ok(())
    }

    /// Get objects for a specific commit (recursive)
    pub async fn get_commit_objects(&self, commit_hash: &Sha256Hash) -> Result<Vec<Vec<u8>>> {
        use crate::types::GitHash;

        let mut objects = Vec::new();
        let mut visited = std::collections::HashSet::new();

        // Convert Sha256Hash to GitHash for internal use
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&commit_hash.to_bytes());
        let initial_hash = GitHash::Sha256(bytes);
        let mut to_fetch = vec![initial_hash];

        while let Some(hash) = to_fetch.pop() {
            if visited.contains(&hash) {
                continue;
            }
            visited.insert(hash.clone());

            // Convert GitHash back to Sha256Hash for the object facade.
            let sha256_hash = match &hash {
                GitHash::Sha256(b) => Sha256Hash::from_bytes(b)?,
                GitHash::Sha1(_) => {
                    // For SHA1 hashes, we need to use the DHT key derivation
                    // For now, skip SHA1 objects in this legacy function
                    continue;
                }
            };

            if let Some(data) = self.get_object(&sha256_hash).await? {
                objects.push(data.clone());

                // Parse object to find referenced objects (returns GitHash).
                // The object we just fetched was keyed by `hash`, whose variant
                // tells us the repository object format for binary tree entries.
                let referenced = self.parse_object_references(&data, hash.object_format())?;
                to_fetch.extend(referenced);
            }
        }

        Ok(objects)
    }

    /// Parse an object to find hash references to other objects.
    ///
    /// `format` is the repository's object format (from the fetched object's
    /// own hash), used to slice binary tree entries at the correct OID width.
    fn parse_object_references(
        &self,
        object_data: &[u8],
        format: crate::types::ObjectFormat,
    ) -> Result<Vec<crate::types::GitHash>> {
        use crate::git::objects::parse_object_references;
        parse_object_references(object_data, format)
    }

    /// Clone a repository from a GitTorrent URL
    pub async fn clone_repository(&self, url: &GitTorrentUrl, local_path: &Path) -> Result<()> {
        match url {
            GitTorrentUrl::Commit { hash } | GitTorrentUrl::CommitWithRefs { hash } => {
                tracing::info!("Cloning repository with commit hash {}", hash);

                // Try to find the repository metadata object
                if let Some(metadata_data) = self.get_object(hash).await? {
                    // Parse repository metadata (implement based on your format)
                    let metadata: RepositoryMetadata = serde_json::from_slice(&metadata_data)?;

                    // Create local repository
                    self.create_local_repository(local_path, &metadata).await?;

                    tracing::info!("Successfully cloned repository to {:?}", local_path);
                } else {
                    return Err(Error::not_found(format!(
                        "Repository metadata not found for hash {hash}"
                    )));
                }
            }
            GitTorrentUrl::GitServer { server: _, repo: _ } => {
                return Err(Error::other("Git server URLs not yet supported"));
            }
            GitTorrentUrl::Username { username: _ } => {
                return Err(Error::other("Username URLs not yet supported"));
            }
        }

        Ok(())
    }

    /// Get service statistics
    pub async fn stats(&self) -> ServiceStats {
        let repos = self.repositories.read().await;
        let cache = self.object_cache.read().await;

        ServiceStats {
            repository_count: repos.len(),
            object_count: cache.len(),
            total_bytes: cache.values().map(|data| data.len() as u64).sum(),
            peer_count: 0, // TODO: Get from DHT
        }
    }

    /// List all repositories
    pub async fn list_repositories(&self) -> Vec<(String, RepositoryMetadata)> {
        let repos = self.repositories.read().await;
        repos
            .iter()
            .map(|(name, meta)| (name.clone(), meta.clone()))
            .collect()
    }

    /// Announce repository to P2P network (wrapper for store_repository_objects)
    pub async fn announce_repository(&self, repo_path: &Path) -> Result<()> {
        self.store_repository_objects(repo_path).await?;
        tracing::info!("Announced repository at {:?} to P2P network", repo_path);
        Ok(())
    }

    /// Query for repository metadata by identifier
    pub async fn query_repository(&self, identifier: &str) -> Result<Option<RepositoryMetadata>> {
        // Try to parse identifier as SHA256 hash and look up repository metadata
        if let Ok(hash) = Sha256Hash::new(identifier) {
            if let Some(data) = self.get_object(&hash).await? {
                // Try to parse as repository metadata JSON
                if let Ok(metadata) = serde_json::from_slice::<RepositoryMetadata>(&data) {
                    return Ok(Some(metadata));
                }
            }
        }

        // Also check local repositories by name
        let repos = self.repositories.read().await;
        if let Some(metadata) = repos.get(identifier) {
            return Ok(Some(metadata.clone()));
        }

        Ok(None)
    }

    /// Scan a repository and extract metadata with object information
    async fn scan_repository(&self, repo_path: &Path) -> Result<RepositoryMetadata> {
        use crate::git::objects::{extract_objects, GitObjectType};

        // Open repository in spawn_blocking (git2::Repository is !Send)
        let path = repo_path.to_path_buf();
        let (branches, root_commits) = crate::git::with_repo_blocking(path, |repo| {
            let mut branches = HashMap::new();
            let mut root_commits = HashMap::new();

            let refs = repo.references()?;
            for reference in refs {
                let reference = reference?;
                if let Some(name) = reference.shorthand() {
                    if let Some(target) = reference.target() {
                        let hash_str = format!("{:0>64}", target.to_string());
                        if let Ok(sha256) = Sha256Hash::new(hash_str) {
                            branches.insert(name.to_owned(), sha256.clone());

                            if reference.is_branch() {
                                root_commits.insert(name.to_owned(), sha256);
                            }
                        }
                    }
                }
            }

            Ok((branches, root_commits))
        })
        .await?;

        // Extract all objects to get complete metadata (async)
        let objects = extract_objects(repo_path).await?;
        let mut object_hashes = Vec::new();
        let mut object_stats = ObjectStats::default();
        let mut total_size = 0u64;

        for obj in &objects {
            object_hashes.push(obj.hash.clone());
            total_size += obj.size as u64;

            // Count objects by type
            match obj.object_type {
                GitObjectType::Commit => object_stats.commit_count += 1,
                GitObjectType::Tree => object_stats.tree_count += 1,
                GitObjectType::Blob => object_stats.blob_count += 1,
                GitObjectType::Tag => object_stats.tag_count += 1,
            }
        }

        let repo_name = repo_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_owned();

        // Collect ref names
        let ref_names: Vec<String> = branches.keys().cloned().collect();

        Ok(RepositoryMetadata {
            name: repo_name,
            description: None,
            remote_url: None,
            branches,
            object_hashes,
            root_commits,
            format_version: 1, // Current format version
            object_stats,
            size_bytes: total_size,
            // SAFETY: Only fails if system time is before Unix epoch (1970)
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            publisher: None, // Could be set from git config in future
            refs: ref_names,
            lfs_chunks: vec![], // No LFS support yet
        })
    }

    /// Create a local repository from metadata using object reconstruction
    async fn create_local_repository(
        &self,
        local_path: &Path,
        metadata: &RepositoryMetadata,
    ) -> Result<()> {
        // Reconstruct repository from individual objects
        self.reconstruct_repository(&metadata.object_hashes, local_path)
            .await?;

        // Set up refs in spawn_blocking (git2::Repository is !Send)
        let path = local_path.to_path_buf();
        let branches = metadata.branches.clone();
        let object_count = metadata.object_hashes.len();
        crate::git::with_repo_blocking(path, move |repo| {
            for (branch_name, commit_hash) in &branches {
                if let Ok(oid) = git2::Oid::from_str(commit_hash.as_str()) {
                    let ref_name = format!("refs/heads/{branch_name}");
                    if let Err(e) = repo.reference(&ref_name, oid, false, "Initial clone") {
                        tracing::warn!("Failed to create ref {}: {}", ref_name, e);
                    }
                }
            }

            if let Some(main_commit) = branches.get("main").or_else(|| branches.get("master")) {
                if let Ok(oid) = git2::Oid::from_str(main_commit.as_str()) {
                    if let Err(e) = repo.set_head_detached(oid) {
                        tracing::warn!("Failed to set HEAD: {}", e);
                    }
                }
            }

            Ok(())
        })
        .await?;

        tracing::info!(
            "Created local repository at {:?} with {} objects",
            local_path,
            object_count
        );
        Ok(())
    }
}

/// Test infrastructure for downstream crates that need to exercise
/// `GitTorrentService` against an in-memory, deterministic object plane.
///
/// Compiled under `cfg(test)` and the `testing` cargo feature. The mock
/// store and fetcher simulate a remote provider advertising bytes; the
/// service is the real production type, so provenance and persistence
/// flow through the same code paths as in deployment.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use super::*;
    use std::collections::HashMap as StdHashMap;
    use tokio::sync::Mutex;

    /// In-memory `ObjectLocator` that hands back whatever providers were
    /// registered for a rendezvous CID. Never touches the network.
    #[derive(Default)]
    pub struct MockObjectLocator {
        announcements: Mutex<Vec<(Cid512, Option<u16>)>>,
        providers: Mutex<StdHashMap<Cid512, Vec<PeerContact>>>,
    }

    impl MockObjectLocator {
        /// Register a provider for a rendezvous CID, mirroring a BEP5
        /// `get_peers` hit.
        pub async fn add_provider(&self, cid: Cid512, provider: PeerContact) {
            self.providers
                .lock()
                .await
                .entry(cid)
                .or_default()
                .push(provider);
        }

        /// Convenience wrapper: register a provider for a content CID,
        /// mapping it to its rendezvous CID the way the real mainline
        /// locator would. Hides the private content→rendezvous mapping
        /// from downstream test code.
        pub async fn add_provider_for_content(
            &self,
            content_cid: &ContentCid,
            provider: PeerContact,
        ) {
            self.add_provider(locator_cid_from_content(content_cid), provider)
                .await;
        }

        /// Announcements recorded by `announce`, in insertion order.
        pub async fn announcements(&self) -> Vec<(Cid512, Option<u16>)> {
            self.announcements.lock().await.clone()
        }
    }

    #[async_trait]
    impl ObjectLocator for MockObjectLocator {
        async fn announce(&self, cid: &Cid512, port: Option<u16>) -> Result<()> {
            self.announcements.lock().await.push((*cid, port));
            Ok(())
        }

        async fn providers(&self, cid: &Cid512) -> Result<Vec<PeerContact>> {
            Ok(self
                .providers
                .lock()
                .await
                .get(cid)
                .cloned()
                .unwrap_or_default())
        }
    }

    /// In-memory `RemoteBlobFetcher` keyed by content CID. Simulates a
    /// remote provider returning bytes (which may or may not match the
    /// CID — the service cannot tell for non-self-verifying CIDs).
    #[derive(Default)]
    pub struct MockBlobFetcher {
        objects: Mutex<StdHashMap<ContentCid, Vec<u8>>>,
        calls: Mutex<usize>,
    }

    impl MockBlobFetcher {
        /// Insert a Git object keyed by its SHA-256 hash (self-verifying).
        pub async fn insert(&self, hash: Sha256Hash, data: Vec<u8>) -> Result<()> {
            let cid = ContentCid::git_sha256(&hash)?;
            self.objects.lock().await.insert(cid, data);
            Ok(())
        }

        /// Insert raw bytes keyed by an arbitrary CID. Used to simulate a
        /// provider returning bytes under a non-self-verifying CID (e.g.
        /// an XET Merkle CID), which is the case the service must NOT
        /// commit before caller validation.
        pub async fn insert_raw(&self, cid: ContentCid, data: Vec<u8>) {
            self.objects.lock().await.insert(cid, data);
        }

        /// Number of `fetch` invocations observed.
        pub async fn calls(&self) -> usize {
            *self.calls.lock().await
        }
    }

    #[async_trait]
    impl RemoteBlobFetcher for MockBlobFetcher {
        async fn fetch(
            &self,
            cid: &ContentCid,
            _providers: Vec<PeerContact>,
        ) -> Result<Option<Vec<u8>>> {
            *self.calls.lock().await += 1;
            Ok(self.objects.lock().await.get(cid).cloned())
        }
    }

    /// Build a `GitTorrentConfig` whose storage lives under `path` and
    /// whose auto-discovery is off, so tests never reach the network.
    pub fn test_config(path: &std::path::Path) -> GitTorrentConfig {
        GitTorrentConfig {
            storage_dir: path.to_path_buf(),
            bootstrap_nodes: vec![],
            auto_discovery: false,
            ..Default::default()
        }
    }

    /// Construct a real `GitTorrentService` backed by the supplied mocks.
    ///
    /// The private locator/fetcher trait bounds stay internal to this
    /// module so downstream callers configure the service purely through
    /// the public mock methods.
    pub async fn new_service_with_mocks(
        config: GitTorrentConfig,
        locator: Arc<MockObjectLocator>,
        fetcher: Arc<MockBlobFetcher>,
    ) -> Result<GitTorrentService> {
        GitTorrentService::new_with_object_plane(config, locator, fetcher).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::testing::{
        new_service_with_mocks, test_config, MockBlobFetcher, MockObjectLocator,
    };
    use std::net::{Ipv4Addr, SocketAddrV4};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_service_creation() -> crate::error::Result<()> {
        let temp_dir = TempDir::new()?;
        let config = test_config(temp_dir.path());

        let service = GitTorrentService::new(config).await?;
        let stats = service.stats().await;

        assert_eq!(stats.repository_count, 0);
        assert_eq!(stats.object_count, 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_object_storage() -> crate::error::Result<()> {
        let temp_dir = TempDir::new()?;
        let config = test_config(temp_dir.path());

        let service = GitTorrentService::new(config).await?;

        let test_data = b"hello world".to_vec();
        let hash = service.put_object(test_data.clone()).await?;

        let retrieved = service.get_object(&hash).await?;
        let retrieved = retrieved.ok_or_else(|| crate::Error::not_found("Object not found"))?;
        assert_eq!(retrieved, test_data);
        Ok(())
    }

    #[test]
    fn test_content_rendezvous_preserves_cid_domain() -> crate::error::Result<()> {
        let digest = [0x5a; 32];
        let git = ContentCid::git_sha256(&Sha256Hash::from_bytes(&digest)?)?;
        let xet = ContentCid::xet_merkle(&digest)?;

        assert_ne!(git, xet);
        assert_ne!(
            locator_cid_from_content(&git),
            locator_cid_from_content(&xet),
            "rendezvous derivation must include CID codec and hash algorithm"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_object_storage_announces_blake3_and_content_cids() -> crate::error::Result<()> {
        let temp_dir = TempDir::new()?;
        let locator = Arc::new(MockObjectLocator::default());
        let fetcher = Arc::new(MockBlobFetcher::default());
        let config = GitTorrentConfig {
            auto_discovery: true,
            p2p_port: 4242,
            ..test_config(temp_dir.path())
        };

        let service =
            GitTorrentService::new_with_object_plane(config, locator.clone(), fetcher).await?;

        let data = b"announced object".to_vec();
        let hash = service.put_object(data.clone()).await?;
        let content_cid = ContentCid::git_sha256(&hash)?;
        let blake3 = service
            .object_plane
            .blobs
            .blake3_of(&hash)
            .await
            .ok_or_else(|| Error::not_found("missing blake3 index"))?;
        let announcements = locator.announcements().await;

        assert!(announcements.contains(&(locator_cid_from_blake3(blake3), Some(4242))));
        assert!(announcements.contains(&(locator_cid_from_content(&content_cid), Some(4242))));
        assert_eq!(
            service.get_object(&hash).await?.as_deref(),
            Some(data.as_slice())
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_get_object_fetches_from_provider_and_stores_locally() -> crate::error::Result<()>
    {
        let temp_dir = TempDir::new()?;
        let locator = Arc::new(MockObjectLocator::default());
        let fetcher = Arc::new(MockBlobFetcher::default());
        let service = GitTorrentService::new_with_object_plane(
            test_config(temp_dir.path()),
            locator.clone(),
            fetcher.clone(),
        )
        .await?;

        let data = b"remote object".to_vec();
        let hash = sha256_git(&data)?;
        let content_cid = ContentCid::git_sha256(&hash)?;
        locator
            .add_provider(
                locator_cid_from_content(&content_cid),
                PeerContact::untrusted(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 6881)),
            )
            .await;
        fetcher.insert(hash.clone(), data.clone()).await?;

        assert_eq!(
            service.get_object(&hash).await?.as_deref(),
            Some(data.as_slice())
        );
        assert_eq!(fetcher.calls().await, 1);

        assert_eq!(
            service.get_object(&hash).await?.as_deref(),
            Some(data.as_slice())
        );
        assert_eq!(fetcher.calls().await, 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_get_object_returns_remote_xet_bytes_uncommitted() -> crate::error::Result<()> {
        let temp_dir = TempDir::new()?;
        let locator = Arc::new(MockObjectLocator::default());
        let fetcher = Arc::new(MockBlobFetcher::default());
        let service = new_service_with_mocks(
            test_config(temp_dir.path()),
            locator.clone(),
            fetcher.clone(),
        )
        .await?;

        let data = b"xet reconstruction bytes".to_vec();
        let cid = ContentCid::xet_merkle(&[0x7f; 32])?;
        locator
            .add_provider(
                locator_cid_from_content(&cid),
                PeerContact::untrusted(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 6881)),
            )
            .await;
        fetcher.insert_raw(cid.clone(), data.clone()).await;

        assert_eq!(
            service.get_object_by_cid(&cid).await?.as_deref(),
            Some(data.as_slice())
        );
        // A Merkle CID cannot verify raw bytes, so provider bytes must not be
        // committed to the local store or cache before the caller validates.
        assert!(service
            .object_plane
            .blobs
            .get_object_by_cid(&cid)
            .await?
            .is_none());
        assert_eq!(
            service.get_object_by_cid(&cid).await?.as_deref(),
            Some(data.as_slice())
        );
        assert_eq!(fetcher.calls().await, 2, "uncommitted bytes are re-fetched");
        Ok(())
    }

    #[tokio::test]
    async fn test_get_object_by_cid_provenance_distinguishes_committed_from_provider_xet_bytes(
    ) -> crate::error::Result<()> {
        // Regression for #1115: provider-fetched XET (non-self-verifying)
        // bytes must surface as `committed == false` so the caller's
        // persistence gate fires. A cache/blob hit and a self-verifying
        // provider fetch must surface as `committed == true`.
        let temp_dir = TempDir::new()?;

        // --- Provider-fetched XET Merkle bytes: NOT committed. ---
        let locator = Arc::new(MockObjectLocator::default());
        let fetcher = Arc::new(MockBlobFetcher::default());
        let service = new_service_with_mocks(
            test_config(temp_dir.path()),
            locator.clone(),
            fetcher.clone(),
        )
        .await?;
        let xet_data = b"xet reconstruction bytes".to_vec();
        let xet_cid = ContentCid::xet_merkle(&[0x7f; 32])?;
        locator
            .add_provider(
                locator_cid_from_content(&xet_cid),
                PeerContact::untrusted(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 6881)),
            )
            .await;
        fetcher.insert_raw(xet_cid.clone(), xet_data.clone()).await;

        let fetch = service
            .get_object_by_cid_with_provenance(&xet_cid)
            .await?
            .ok_or_else(|| crate::Error::not_found("provider did not return the XET object"))?;
        assert_eq!(fetch.data, xet_data);
        assert!(
            !fetch.committed,
            "provider-fetched XET bytes must not be reported committed"
        );

        // --- Self-verifying Git CID provider fetch: committed. ---
        let git_data = b"self-verifying git object".to_vec();
        let git_hash = sha256_git(&git_data)?;
        let git_cid = ContentCid::git_sha256(&git_hash)?;
        locator
            .add_provider(
                locator_cid_from_content(&git_cid),
                PeerContact::untrusted(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 6882)),
            )
            .await;
        fetcher.insert(git_hash, git_data.clone()).await?;
        let fetch = service
            .get_object_by_cid_with_provenance(&git_cid)
            .await?
            .ok_or_else(|| crate::Error::not_found("provider did not return the Git object"))?;
        assert_eq!(fetch.data, git_data);
        assert!(
            fetch.committed,
            "self-verifying provider bytes are committed during fetch"
        );

        // --- Local cache hit (the prior line cached it): committed. ---
        let fetch = service
            .get_object_by_cid_with_provenance(&git_cid)
            .await?
            .ok_or_else(|| crate::Error::not_found("object was not cached"))?;
        assert!(fetch.committed, "cache hits are committed");
        // No additional provider fetch for the cached path.
        assert_eq!(fetcher.calls().await, 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_get_object_rejects_provider_bytes_with_wrong_sha256() -> crate::error::Result<()>
    {
        let temp_dir = TempDir::new()?;
        let locator = Arc::new(MockObjectLocator::default());
        let fetcher = Arc::new(MockBlobFetcher::default());
        let service = GitTorrentService::new_with_object_plane(
            test_config(temp_dir.path()),
            locator.clone(),
            fetcher.clone(),
        )
        .await?;

        let requested_hash = sha256_git(b"wanted")?;
        let content_cid = ContentCid::git_sha256(&requested_hash)?;
        locator
            .add_provider(
                locator_cid_from_content(&content_cid),
                PeerContact::untrusted(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 6881)),
            )
            .await;
        fetcher
            .insert(requested_hash.clone(), b"poisoned".to_vec())
            .await?;

        assert!(service.get_object(&requested_hash).await.is_err());
        assert!(service
            .object_plane
            .blobs
            .get_object(&requested_hash)
            .await?
            .is_none());
        Ok(())
    }
}
