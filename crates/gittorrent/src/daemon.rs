//! GitTorrent Daemon Server
//!
//! This module implements the gittorrentd daemon that provides:
//! - Repository hosting and announcement
//! - P2P networking and discovery
//! - XET chunk distribution
//! - Git protocol serving

use crate::service::{GitTorrentService, GitTorrentConfig};
use crate::{Result, Error};

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::net::TcpListener;
use serde::{Serialize, Deserialize};

/// GitTorrent daemon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// GitTorrent service configuration
    pub service: GitTorrentConfig,
    /// Hosted repositories directory
    pub repositories_dir: PathBuf,
    /// Maximum number of hosted repositories
    pub max_repositories: usize,
    /// Enable Git protocol server
    pub enable_git_server: bool,
    /// Git protocol port
    pub git_port: u16,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            service: GitTorrentConfig::default(),
            repositories_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("gittorrent")
                .join("repositories"),
            max_repositories: 100,
            enable_git_server: true,
            git_port: 9418, // Standard Git protocol port
        }
    }
}

/// Repository information for hosting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostedRepository {
    /// Repository name/identifier
    pub name: String,
    /// Local path to repository
    pub path: PathBuf,
    /// Repository metadata
    pub description: Option<String>,
    /// Repository size in bytes
    pub size: u64,
    /// Last announced timestamp
    pub last_announced: u64,
    /// Number of active peers
    pub peer_count: usize,
}

/// GitTorrent daemon server
pub struct GitTorrentDaemon {
    /// GitTorrent service instance
    service: Arc<GitTorrentService>,
    /// Daemon configuration
    config: DaemonConfig,
    /// Hosted repositories
    repositories: Arc<RwLock<HashMap<String, HostedRepository>>>,
    /// Git protocol listener
    git_listener: Option<TcpListener>,
}

impl GitTorrentDaemon {
    /// Create a new GitTorrent daemon
    pub async fn new(config: DaemonConfig) -> Result<Self> {
        // Create repositories directory
        tokio::fs::create_dir_all(&config.repositories_dir).await
            .map_err(Error::from)?;

        // Initialize GitTorrent service
        let service = Arc::new(GitTorrentService::new(config.service.clone()).await?);

        // Initialize Git protocol listener if enabled
        let git_listener = if config.enable_git_server {
            let addr = format!("{}:{}", config.service.bind_address, config.git_port);
            Some(TcpListener::bind(&addr).await
                .map_err(|e| Error::other(format!("Failed to bind Git server to {addr}: {e}")))?)
        } else {
            None
        };

        Ok(Self {
            service,
            config,
            repositories: Arc::new(RwLock::new(HashMap::new())),
            git_listener,
        })
    }

    /// Run the daemon server
    pub async fn run(&self) -> Result<()> {
        tracing::info!("GitTorrent daemon starting...");
        tracing::info!("Repositories directory: {}", self.config.repositories_dir.display());
        tracing::info!("Service bind address: {}:{}",
                      self.config.service.bind_address,
                      self.config.service.bind_port);

        // Scan for existing repositories
        self.scan_repositories().await?;

        // Start Git protocol server if enabled
        if let Some(ref listener) = self.git_listener {
            let _service = Arc::clone(&self.service);
            let repositories = Arc::clone(&self.repositories);
            let listener_addr = listener.local_addr().map_err(Error::from)?;

            tokio::spawn(async move {
                // Note: In a real implementation, we'd need to pass the listener properly
                // For now, just log that the server would be started
                tracing::info!("Git protocol server would listen on: {}", listener_addr);
                tracing::info!("Git server task started with {} repositories",
                    repositories.read().await.len());
            });

            tracing::info!("Git protocol server listening on port {}", self.config.git_port);
        }

        // Start periodic announcement of repositories
        let service = Arc::clone(&self.service);
        let repositories = Arc::clone(&self.repositories);

        tokio::spawn(async move {
            Self::run_announcement_loop(service, repositories).await;
        });

        // Keep the daemon running
        tracing::info!("GitTorrent daemon is ready");

        // In a real implementation, this would handle shutdown signals
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            tracing::debug!("Daemon heartbeat");
        }
    }

    /// Scan repositories directory for existing repositories
    async fn scan_repositories(&self) -> Result<()> {
        let mut entries = tokio::fs::read_dir(&self.config.repositories_dir).await
            .map_err(Error::from)?;

        let mut repositories = self.repositories.write().await;

        while let Some(entry) = entries.next_entry().await.map_err(Error::from)? {
            let path = entry.path();

            if path.is_dir() {
                // Check if it's a Git repository
                if path.join(".git").exists() || path.join("objects").exists() {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown").to_owned();

                    let size = self.calculate_repository_size(&path).await?;

                    let hosted_repo = HostedRepository {
                        name: name.clone(),
                        path: path.clone(),
                        description: None,
                        size,
                        last_announced: 0,
                        peer_count: 0,
                    };

                    repositories.insert(name.clone(), hosted_repo);
                    tracing::info!("Found repository: {} at {}", name, path.display());

                    // Announce repository to P2P network
                    if let Err(e) = self.service.announce_repository(&path).await {
                        tracing::warn!("Failed to announce repository {}: {}", name, e);
                    }
                }
            }
        }

        tracing::info!("Scanned {} repositories", repositories.len());
        Ok(())
    }

    /// Calculate repository size
    async fn calculate_repository_size(&self, repo_path: &Path) -> Result<u64> {
        let mut total_size = 0;

        let mut entries = tokio::fs::read_dir(repo_path).await
            .map_err(Error::from)?;

        while let Some(entry) = entries.next_entry().await.map_err(Error::from)? {
            let metadata = entry.metadata().await.map_err(Error::from)?;
            if metadata.is_file() {
                total_size += metadata.len();
            }
        }

        Ok(total_size)
    }


    /// Run periodic announcement loop
    async fn run_announcement_loop(
        service: Arc<GitTorrentService>,
        repositories: Arc<RwLock<HashMap<String, HostedRepository>>>,
    ) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // 5 minutes

        loop {
            interval.tick().await;

            let repos = repositories.read().await;
            for (name, repo) in repos.iter() {
                if let Err(e) = service.announce_repository(&repo.path).await {
                    tracing::warn!("Failed to announce repository {}: {}", name, e);
                } else {
                    tracing::debug!("Announced repository: {}", name);
                }
            }

            tracing::debug!("Completed announcement cycle for {} repositories", repos.len());
        }
    }

    /// Add a new repository to hosting
    pub async fn add_repository(&self, name: String, path: PathBuf) -> Result<()> {
        if !path.exists() {
            return Err(Error::not_found(format!("Repository path does not exist: {}", path.display())));
        }

        let size = self.calculate_repository_size(&path).await?;

        let hosted_repo = HostedRepository {
            name: name.clone(),
            path: path.clone(),
            description: None,
            size,
            last_announced: 0,
            peer_count: 0,
        };

        let mut repositories = self.repositories.write().await;
        repositories.insert(name.clone(), hosted_repo);

        // Announce to P2P network
        self.service.announce_repository(&path).await?;

        tracing::info!("Added repository: {} at {}", name, path.display());
        Ok(())
    }

    /// Get list of hosted repositories
    pub async fn list_repositories(&self) -> HashMap<String, HostedRepository> {
        self.repositories.read().await.clone()
    }

    /// Get daemon statistics
    pub async fn get_stats(&self) -> DaemonStats {
        let repositories = self.repositories.read().await;
        let total_size: u64 = repositories.values().map(|r| r.size).sum();
        let total_peers: usize = repositories.values().map(|r| r.peer_count).sum();

        DaemonStats {
            repository_count: repositories.len(),
            total_size,
            total_peers,
            uptime_seconds: 0, // TODO: Track actual uptime
        }
    }
}


/// Daemon statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonStats {
    pub repository_count: usize,
    pub total_size: u64,
    pub total_peers: usize,
    pub uptime_seconds: u64,
}

/// Entry point for gittorrentd binary
pub async fn run_daemon(config: DaemonConfig) -> Result<()> {
    let daemon = GitTorrentDaemon::new(config).await?;
    daemon.run().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_daemon_creation() -> crate::error::Result<()> {
        let temp_dir = TempDir::new()?;
        let config = DaemonConfig {
            repositories_dir: temp_dir.path().to_path_buf(),
            enable_git_server: false, // Disable for testing
            ..Default::default()
        };

        let daemon = GitTorrentDaemon::new(config).await?;
        let stats = daemon.get_stats().await;
        assert_eq!(stats.repository_count, 0);
        Ok(())
    }


    #[tokio::test]
    async fn test_repository_management() -> crate::error::Result<()> {
        let temp_dir = TempDir::new()?;
        let repo_dir = temp_dir.path().join("test-repo");

        // Create a proper Git repository using git2
        git2::Repository::init(&repo_dir)?;
        tokio::fs::write(repo_dir.join("README.md"), "# Test Repository").await?;

        let config = DaemonConfig {
            repositories_dir: temp_dir.path().to_path_buf(),
            enable_git_server: false,
            ..Default::default()
        };

        let daemon = GitTorrentDaemon::new(config).await?;

        // Add repository
        daemon.add_repository("test-repo".to_owned(), repo_dir).await?;

        // Check it was added
        let repos = daemon.list_repositories().await;
        assert_eq!(repos.len(), 1);
        assert!(repos.contains_key("test-repo"));
        Ok(())
    }
}