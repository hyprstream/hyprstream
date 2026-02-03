//! GitTorrent Service - Core P2P Git Repository Service
//!
//! This module implements the main GitTorrent service that orchestrates:
//! - Git repository operations using git2
//! - P2P networking for repository discovery and sharing via libp2p
//! - SHA256-based content addressing and distribution

use crate::{Result, Error, GitTorrentUrl, Sha256Hash};
use crate::dht::{GitTorrentDht, GitObjectKey, GitObjectRecord};
use crate::crypto::hash::{sha256_git, verify_sha256};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use libp2p::Multiaddr;

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
    /// P2P/DHT listen port (0 = random port)
    pub p2p_port: u16,
    /// DHT mode (Client or Server)
    pub dht_mode: crate::dht::DhtMode,
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
            p2p_port: 0, // 0 = random port
            dht_mode: crate::dht::DhtMode::Server, // Server mode by default
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
    /// use gittorrent::service::GitTorrentConfig;
    ///
    /// // Simple case - use defaults
    /// let config: GitTorrentConfig = GitTorrentConfig::builder()?
    ///     .build()?
    ///     .try_deserialize()?;
    /// ```
    ///
    /// # Extended Usage - Add Custom Sources
    /// ```ignore
    /// use gittorrent::service::GitTorrentConfig;
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
    /// use gittorrent::service::GitTorrentConfig;
    ///
    /// // Override specific values programmatically
    /// let config: GitTorrentConfig = GitTorrentConfig::builder()?
    ///     .set_override("p2p_port", 4001)?  // Highest priority
    ///     .build()?
    ///     .try_deserialize()?;
    /// ```
    ///
    /// # Supported Environment Variables
    /// - `GITTORRENT_P2P_PORT`: P2P/DHT listen port
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
                    .list_separator(",")
            ))
    }

    /// Convenience method to load configuration with defaults
    ///
    /// This is equivalent to `GitTorrentConfig::builder()?.build()?.try_deserialize()?`
    ///
    /// # Basic Usage
    /// ```ignore
    /// use gittorrent::service::GitTorrentConfig;
    ///
    /// let config = GitTorrentConfig::load()?;
    /// ```
    pub fn load() -> Result<Self> {
        let config = Self::builder()?
            .build()
            .map_err(|e| Error::other(format!("Failed to build config: {e}")))?;

        config.try_deserialize()
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
    /// DHT service for P2P networking
    dht: Arc<GitTorrentDht>,
    /// Local repository metadata
    repositories: Arc<RwLock<HashMap<String, RepositoryMetadata>>>,
    /// Git object cache
    object_cache: Arc<RwLock<HashMap<Sha256Hash, Vec<u8>>>>,
}

impl GitTorrentService {
    /// Create a new GitTorrent service
    pub async fn new(config: GitTorrentConfig) -> Result<Self> {
        // Ensure storage directory exists
        tokio::fs::create_dir_all(&config.storage_dir).await?;

        // Initialize DHT with configured port
        let dht = Arc::new(GitTorrentDht::new(config.p2p_port, config.dht_mode).await?);

        // Bootstrap DHT with known peers
        let bootstrap_addrs: Vec<Multiaddr> = config
            .bootstrap_nodes
            .iter()
            .filter_map(|addr| addr.parse().ok())
            .collect();

        if !bootstrap_addrs.is_empty() {
            if let Err(e) = dht.bootstrap(bootstrap_addrs).await {
                tracing::warn!("Bootstrap failed: {}. Running in standalone mode.", e);
                tracing::info!("This node will start its own network. Other nodes can connect to:");
                tracing::info!("  Listen addresses will be shown below");
            } else {
                tracing::info!("Successfully bootstrapped to P2P network");
            }
        } else {
            tracing::info!("No bootstrap nodes configured - running in standalone mode");
            tracing::info!("This node will start its own network. Other nodes can connect using --bootstrap with this node's addresses");
        }

        Ok(Self {
            dht,
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
        // Check local cache first
        {
            let cache = self.object_cache.read().await;
            if let Some(data) = cache.get(hash) {
                return Ok(Some(data.clone()));
            }
        }

        // Try to fetch from DHT
        let key = GitObjectKey::from_sha256(hash);
        if let Some(record) = self.dht.get_object(key).await? {
            // Verify the object hash
            if verify_sha256(&record.data, hash)? {
                // Cache the object
                let mut cache = self.object_cache.write().await;
                cache.insert(hash.clone(), record.data.clone());
                return Ok(Some(record.data));
            } else {
                tracing::warn!("Object hash mismatch for {}", hash);
            }
        }

        Ok(None)
    }

    /// Store a Git object in the DHT
    pub async fn put_object(&self, data: Vec<u8>) -> Result<Sha256Hash> {
        // Calculate SHA256 hash
        let hash = sha256_git(&data)?;

        // Store in DHT
        let key = GitObjectKey::from_sha256(&hash);
        let record = GitObjectRecord::new(key.clone(), data.clone());
        self.dht.put_object(record).await?;

        // Announce as provider
        self.dht.provide(key).await?;

        // Cache locally
        let mut cache = self.object_cache.write().await;
        cache.insert(hash.clone(), data);

        tracing::debug!("Stored object with hash: {}", hash);
        Ok(hash)
    }

    /// Find providers for a Git object
    pub async fn find_providers(&self, hash: &Sha256Hash) -> Result<Vec<libp2p::PeerId>> {
        let key = GitObjectKey::from_sha256(hash);
        self.dht.get_providers(key).await
    }

    /// Store all objects from a repository individually
    pub async fn store_repository_objects(&self, repo_path: &Path) -> Result<Vec<Sha256Hash>> {
        use crate::git::objects::extract_objects;

        let objects = extract_objects(repo_path).await?;
        let mut stored_hashes = Vec::new();

        tracing::info!("Storing {} objects from repository", objects.len());

        for obj in objects {
            // Store object in DHT
            let key = GitObjectKey::from_sha256(&obj.hash);
            let record = GitObjectRecord::new(key.clone(), obj.data.clone());
            self.dht.put_object(record).await?;

            // Announce as provider
            self.dht.provide(key).await?;

            // Cache locally
            {
                let mut cache = self.object_cache.write().await;
                cache.insert(obj.hash.clone(), obj.data);
            }

            stored_hashes.push(obj.hash);
        }

        tracing::info!("Successfully stored {} objects", stored_hashes.len());
        Ok(stored_hashes)
    }

    /// Store objects for a specific commit and its history
    pub async fn store_commit_objects(&self, repo_path: &Path, commit_hash: &str) -> Result<Vec<Sha256Hash>> {
        use crate::git::objects::extract_commit_objects;

        let objects = extract_commit_objects(repo_path, commit_hash).await?;
        let mut stored_hashes = Vec::new();

        tracing::info!("Storing {} objects for commit {}", objects.len(), commit_hash);

        for obj in objects {
            // Store object in DHT
            let key = GitObjectKey::from_sha256(&obj.hash);
            let record = GitObjectRecord::new(key.clone(), obj.data.clone());
            self.dht.put_object(record).await?;

            // Announce as provider
            self.dht.provide(key).await?;

            // Cache locally
            {
                let mut cache = self.object_cache.write().await;
                cache.insert(obj.hash.clone(), obj.data);
            }

            stored_hashes.push(obj.hash);
        }

        Ok(stored_hashes)
    }

    /// Reconstruct repository from individual objects
    pub async fn reconstruct_repository(&self,
        object_hashes: &[Sha256Hash],
        target_path: &Path
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
                                "blob" => GitObjectType::Blob,
                                "tag" => GitObjectType::Tag,
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
            tracing::warn!("Missing {} objects during reconstruction", missing_objects.len());
            // Try to fetch missing objects from DHT
            for hash in missing_objects {
                self.fetch_object_from_peers(hash).await?;
            }
        }

        // Write objects to repository
        write_objects(target_path, &git_objects).await?;

        tracing::info!("Reconstructed repository with {} objects at {:?}", git_objects.len(), target_path);
        Ok(())
    }

    /// Fetch object from DHT peers
    async fn fetch_object_from_peers(&self, hash: &Sha256Hash) -> Result<()> {
        // Find providers for this object
        let providers = self.find_providers(hash).await?;

        if providers.is_empty() {
            return Err(Error::not_found(format!("No providers found for object {hash}")));
        }

        tracing::debug!("Found {} providers for object {}", providers.len(), hash);

        // Try to get the object from DHT again (providers might have updated)
        if let Some(data) = self.get_object(hash).await? {
            let mut cache = self.object_cache.write().await;
            cache.insert(hash.clone(), data);
            return Ok(());
        }

        Err(Error::not_found(format!("Failed to fetch object {hash} from peers")))
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

            // Convert GitHash back to Sha256Hash for get_object (which uses DHT)
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

                // Parse object to find referenced objects (returns GitHash)
                let referenced = self.parse_object_references(&data)?;
                to_fetch.extend(referenced);
            }
        }

        Ok(objects)
    }

    /// Parse an object to find hash references to other objects
    fn parse_object_references(&self, object_data: &[u8]) -> Result<Vec<crate::types::GitHash>> {
        use crate::git::objects::parse_object_references;
        parse_object_references(object_data)
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
                    return Err(Error::not_found(format!("Repository metadata not found for hash {hash}")));
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
        repos.iter().map(|(name, meta)| (name.clone(), meta.clone())).collect()
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

        // Open repository with git2
        let repo = git2::Repository::open(repo_path)?;

        let mut branches = HashMap::new();
        let mut root_commits = HashMap::new();

        // Get all references
        let refs = repo.references()?;
        for reference in refs {
            let reference = reference?;
            if let Some(name) = reference.shorthand() {
                if let Some(target) = reference.target() {
                    // Convert git2::Oid to SHA256 (this would need actual SHA256 support in git2)
                    let hash_str = format!("{:0>64}", target.to_string());
                    if let Ok(sha256) = Sha256Hash::new(hash_str) {
                        branches.insert(name.to_owned(), sha256.clone());

                        // If this is a branch (not a tag), add to root commits
                        if reference.is_branch() {
                            root_commits.insert(name.to_owned(), sha256);
                        }
                    }
                }
            }
        }

        // Extract all objects to get complete metadata
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

        let repo_name = repo_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown").to_owned();

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
    async fn create_local_repository(&self, local_path: &Path, metadata: &RepositoryMetadata) -> Result<()> {
        // Reconstruct repository from individual objects
        self.reconstruct_repository(&metadata.object_hashes, local_path).await?;

        // Open the reconstructed repository to set up refs
        let repo = git2::Repository::open(local_path)?;

        // Create refs for branches
        for (branch_name, commit_hash) in &metadata.branches {
            // Convert hash back to git2::Oid
            if let Ok(oid) = git2::Oid::from_str(commit_hash.as_str()) {
                let ref_name = format!("refs/heads/{branch_name}");

                // Create the reference
                if let Err(e) = repo.reference(&ref_name, oid, false, "Initial clone") {
                    tracing::warn!("Failed to create ref {}: {}", ref_name, e);
                }
            }
        }

        // Set HEAD to the default branch
        if let Some(main_commit) = metadata.branches.get("main").or_else(|| metadata.branches.get("master")) {
            if let Ok(oid) = git2::Oid::from_str(main_commit.as_str()) {
                if let Err(e) = repo.set_head_detached(oid) {
                    tracing::warn!("Failed to set HEAD: {}", e);
                }
            }
        }

        tracing::info!("Created local repository at {:?} with {} objects",
                      local_path, metadata.object_hashes.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_service_creation() -> crate::error::Result<()> {
        let temp_dir = TempDir::new()?;
        let config = GitTorrentConfig {
            storage_dir: temp_dir.path().to_path_buf(),
            bootstrap_nodes: vec![],
            ..Default::default()
        };

        let service = GitTorrentService::new(config).await?;
        let stats = service.stats().await;

        assert_eq!(stats.repository_count, 0);
        assert_eq!(stats.object_count, 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_object_storage() -> crate::error::Result<()> {
        let temp_dir = TempDir::new()?;
        let config = GitTorrentConfig {
            storage_dir: temp_dir.path().to_path_buf(),
            bootstrap_nodes: vec![],
            ..Default::default()
        };

        let service = GitTorrentService::new(config).await?;

        let test_data = b"hello world".to_vec();
        let hash = service.put_object(test_data.clone()).await?;

        let retrieved = service.get_object(&hash).await?;
        let retrieved = retrieved.ok_or_else(|| crate::Error::not_found("Object not found"))?;
        assert_eq!(retrieved, test_data);
        Ok(())
    }
}