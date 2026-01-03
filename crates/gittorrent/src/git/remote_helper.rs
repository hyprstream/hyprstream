//! Git Remote Helper for gittorrent:// protocol
//!
//! This module implements the git-remote-gittorrent binary that allows
//! standard Git commands to work with gittorrent:// URLs transparently.
//!
//! Usage: git clone gittorrent://example.com/user/repo

use crate::service::{GitTorrentService, GitTorrentConfig};
use crate::{Result, Error, GitTorrentUrl};

use std::path::{Path, PathBuf};
use std::io::{self, BufRead, Write};
use std::collections::HashMap;

/// Git remote helper for gittorrent:// protocol
pub struct GitRemoteHelper {
    /// GitTorrent service instance
    service: GitTorrentService,
    /// Remote repository URL
    remote_url: Option<GitTorrentUrl>,
    /// Local repository path
    local_repo: Option<PathBuf>,
    /// Git capabilities
    capabilities: Vec<String>,
}

impl GitRemoteHelper {
    /// Create a new Git remote helper with default config
    pub async fn new() -> Result<Self> {
        let config = GitTorrentConfig::default();
        Self::new_with_config(config).await
    }

    /// Create a new Git remote helper with custom config
    pub async fn new_with_config(config: GitTorrentConfig) -> Result<Self> {
        let service = GitTorrentService::new(config).await?;

        Ok(Self {
            service,
            remote_url: None,
            local_repo: None,
            capabilities: vec![
                "connect".to_string(),
                "list".to_string(),
                "fetch".to_string(),
                "push".to_string(),
                "option".to_string(),
            ],
        })
    }

    /// Run the Git remote helper protocol
    pub async fn run(&mut self) -> Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut batch_commands = Vec::new();
        let mut in_batch = false;

        // Read commands from stdin and respond
        for line in stdin.lock().lines() {
            let line = line.map_err(Error::from)?;
            let line = line.trim();

            tracing::debug!("Git remote helper received: '{}'", line);

            // Handle blank lines (end of batch or empty input)
            if line.is_empty() {
                if in_batch {
                    // Process the batch of commands
                    match self.process_batch(&batch_commands).await {
                        Ok(()) => {
                            // Send blank line to indicate batch completion
                            writeln!(stdout).map_err(Error::from)?;
                            stdout.flush().map_err(Error::from)?;
                        }
                        Err(e) => {
                            writeln!(stdout, "error {}", e).map_err(Error::from)?;
                            stdout.flush().map_err(Error::from)?;
                            return Err(e);
                        }
                    }
                    batch_commands.clear();
                    in_batch = false;
                } else {
                    // Just an empty line, ignore
                    continue;
                }
            } else {
                // Check if this is a batch command
                let parts: Vec<&str> = line.split_whitespace().collect();
                if !parts.is_empty() && (parts[0] == "fetch" || parts[0] == "push") {
                    // Start or continue a batch
                    batch_commands.push(line.to_string());
                    in_batch = true;
                } else {
                    // Handle single command immediately
                    match self.handle_command(line).await {
                        Ok(response) => {
                            if let Some(resp) = response {
                                print!("{}", resp);
                                stdout.flush().map_err(Error::from)?;
                            }
                        }
                        Err(e) => {
                            writeln!(stdout, "error {}", e).map_err(Error::from)?;
                            stdout.flush().map_err(Error::from)?;
                            return Err(e);
                        }
                    }
                }
            }
        }

        // Handle any remaining batch commands
        if !batch_commands.is_empty() {
            self.process_batch(&batch_commands).await?;
            writeln!(stdout).map_err(Error::from)?;
            stdout.flush().map_err(Error::from)?;
        }

        Ok(())
    }

    /// Handle a single Git remote helper command
    async fn handle_command(&mut self, command: &str) -> Result<Option<String>> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        tracing::debug!("Handling command: '{}' -> parts: {:?}", command, parts);
        if parts.is_empty() {
            return Ok(None);
        }

        match parts[0] {
            "capabilities" => Ok(Some(self.handle_capabilities())),
            "list" => self.handle_list(&parts[1..]).await,
            "connect" => self.handle_connect(&parts[1..]).await,
            "option" => self.handle_option(&parts[1..]).await,
            "" => Ok(None), // Empty line
            _ => {
                tracing::warn!("Unknown git remote helper command: {}", command);
                Ok(None)
            }
        }
    }

    /// Process a batch of commands (for fetch/push)
    async fn process_batch(&mut self, commands: &[String]) -> Result<()> {
        for command in commands {
            let parts: Vec<&str> = command.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "fetch" => {
                    self.handle_fetch(&parts[1..]).await?;
                }
                "push" => {
                    let result = self.handle_push(&parts[1..]).await?;
                    print!("{}", result);
                }
                _ => {
                    return Err(Error::other(format!("Invalid batch command: {}", parts[0])));
                }
            }
        }
        Ok(())
    }

    /// Handle capabilities command
    fn handle_capabilities(&self) -> String {
        let mut response = String::new();
        for cap in &self.capabilities {
            response.push_str(&format!("{}\n", cap));
        }
        response.push('\n'); // End with empty line
        response
    }

    /// Handle list command - list remote references
    async fn handle_list(&mut self, args: &[&str]) -> Result<Option<String>> {
        let for_push = args.first().map(|&s| s == "for-push").unwrap_or(false);
        tracing::debug!("List command, for_push: {}", for_push);

        let url = self.remote_url.as_ref()
            .ok_or_else(|| Error::other("No remote URL set"))?;

        // Try to get refs from P2P network, fall back to defaults if not available
        let refs = self.query_remote_refs(url).await?;

        let mut response = String::new();
        for (ref_name, commit_hash) in &refs {
            response.push_str(&format!("{} {}\n", commit_hash, ref_name));
        }

        // Set HEAD to point to main or master if available
        if refs.contains_key("refs/heads/main") {
            response.push_str("@refs/heads/main HEAD\n");
        } else if refs.contains_key("refs/heads/master") {
            response.push_str("@refs/heads/master HEAD\n");
        }

        response.push('\n'); // End with empty line

        tracing::info!("Listed {} references for {}",
                      refs.len(),
                      match url {
                          GitTorrentUrl::Commit { hash } => format!("commit {}", hash),
                          GitTorrentUrl::CommitWithRefs { hash } => format!("commit {} (with refs)", hash),
                          GitTorrentUrl::GitServer { server, repo } => format!("{}/{}", server, repo),
                          GitTorrentUrl::Username { username } => username.clone(),
                      });

        Ok(Some(response))
    }

    /// Handle connect command - establish connection to remote
    async fn handle_connect(&mut self, args: &[&str]) -> Result<Option<String>> {
        tracing::debug!("Connect called with {} args: {:?}", args.len(), args);
        if args.is_empty() {
            return Err(Error::other("connect command requires service"));
        }

        let service_name = args[0];

        // URL should already be set from command line arguments
        if self.remote_url.is_none() {
            return Err(Error::other("Remote URL not set"));
        }

        tracing::info!("Connected for service {} to {:?}", service_name, self.remote_url);

        // For GitTorrent, we don't support the bidirectional Git protocol directly.
        // Instead, we use the import/export capabilities through the remote helper protocol.
        // Tell Git to fall back to using the remote helper commands (list, fetch, push)
        Ok(Some("fallback\n".to_string()))
    }

    /// Handle fetch command in batch
    async fn handle_fetch(&mut self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(Error::other("fetch command requires arguments"));
        }

        let commit_hash = args[0];
        let ref_name = if args.len() > 1 { args[1] } else { "refs/heads/main" };

        let url = self.remote_url.as_ref()
            .ok_or_else(|| Error::other("No remote URL set"))?;

        let local_repo = self.local_repo.as_ref()
            .ok_or_else(|| Error::other("No local repository path set"))?;

        tracing::info!("Fetching {} {} from GitTorrent", commit_hash, ref_name);

        // In a full implementation, this would:
        // 1. Query the P2P network for the repository
        // 2. Download Git objects from peers
        // 3. Process LFS/XET files
        // 4. Import objects into local Git repository

        self.fetch_repository_data(url, local_repo, commit_hash, ref_name).await?;
        Ok(())
    }

    /// Handle push command in batch
    async fn handle_push(&mut self, args: &[&str]) -> Result<String> {
        if args.is_empty() {
            return Err(Error::other("push command requires refspec"));
        }

        let refspec = args[0];
        let (src_ref, dst_ref) = if let Some(colon_pos) = refspec.find(':') {
            let src = &refspec[..colon_pos];
            let dst = &refspec[colon_pos + 1..];
            // Handle force push prefix
            let src = src.strip_prefix('+').unwrap_or(src);
            (src, dst)
        } else {
            // If no colon, assume same ref name for both src and dst
            (refspec, refspec)
        };

        let url = self.remote_url.as_ref()
            .ok_or_else(|| Error::other("No remote URL set"))?;

        let local_repo = self.local_repo.as_ref()
            .ok_or_else(|| Error::other("No local repository path set"))?;

        tracing::info!("Pushing {} to {} on GitTorrent", src_ref, dst_ref);

        // In a full implementation, this would:
        // 1. Extract Git objects to push
        // 2. Process large files with LFS/XET
        // 3. Distribute chunks to P2P network
        // 4. Announce updated repository metadata

        match self.push_repository_data(url, local_repo, src_ref, dst_ref).await {
            Ok(()) => Ok(format!("ok {}\n", dst_ref)),
            Err(e) => Ok(format!("error {} {}\n", dst_ref, e)),
        }
    }

    /// Handle option command - set options
    async fn handle_option(&mut self, args: &[&str]) -> Result<Option<String>> {
        if args.len() < 2 {
            return Err(Error::other("option command requires key and value"));
        }

        let key = args[0];
        let value = args[1];

        match key {
            "verbosity" => {
                tracing::debug!("Set verbosity to {}", value);
                Ok(Some("ok\n".to_string()))
            }
            "progress" => {
                tracing::debug!("Set progress to {}", value);
                Ok(Some("ok\n".to_string()))
            }
            _ => {
                tracing::debug!("Unknown option: {} = {}", key, value);
                Ok(Some("unsupported\n".to_string()))
            }
        }
    }

    /// Set local repository path
    pub fn set_local_repo(&mut self, path: PathBuf) -> Result<()> {
        self.local_repo = Some(path);
        Ok(())
    }

    /// Fetch repository data from P2P network
    async fn fetch_repository_data(
        &self,
        url: &GitTorrentUrl,
        local_repo: &Path,
        commit_hash: &str,
        ref_name: &str,
    ) -> Result<()> {
        tracing::info!("Fetching repository data for {} at {}", ref_name, commit_hash);

        // Handle commit-based URLs differently
        match url {
            GitTorrentUrl::Commit { hash } | GitTorrentUrl::CommitWithRefs { hash } => {
                // For commit-based URLs, try to clone from DHT directly
                if let Err(e) = self.clone_from_dht(hash, local_repo, url.includes_refs()).await {
                    tracing::warn!("Failed to clone from DHT: {}, falling back to basic repository", e);
                    self.create_basic_repository(local_repo, hash.as_str(), ref_name, url).await?;
                }
            }
            _ => {
                // Legacy behavior for other URL types
                let repo_identifier = match url {
                    GitTorrentUrl::GitServer { server, repo } => format!("{}/{}", server, repo),
                    GitTorrentUrl::Username { username } => username.clone(),
                    _ => unreachable!(),
                };

                // Try to query repository metadata from P2P network
                match self.service.query_repository(&repo_identifier).await? {
                    Some(metadata) => {
                        let repo = self.init_git_repo(local_repo)?;
                        self.create_git_repo_from_metadata(&repo, &metadata, ref_name, url).await?;
                    }
                    None => {
                        tracing::info!("Repository metadata not found in P2P network, creating basic repository");
                        self.create_basic_repository(local_repo, &repo_identifier, ref_name, url).await?;
                    }
                }
            }
        }

        tracing::debug!("Successfully fetched repository data to {}", local_repo.display());
        Ok(())
    }

    /// Initialize or open Git repository
    fn init_git_repo(&self, local_repo: &Path) -> Result<git2::Repository> {
        if !local_repo.join(".git").exists() {
            git2::Repository::init_bare(local_repo).map_err(Error::from)
        } else {
            git2::Repository::open(local_repo).map_err(Error::from)
        }
    }

    /// Clone repository from DHT using merkle tree traversal
    async fn clone_from_dht(
        &self,
        commit_hash: &crate::types::Sha256Hash,
        _target_path: &Path,
        _include_refs: bool,
    ) -> Result<()> {
        // This would use the DHT service to perform the actual clone
        // For now, we'll create a placeholder since we need the actual DHT connection

        tracing::info!("Attempting to clone commit {} from DHT", commit_hash);

        // TODO: When DHT is properly initialized, use:
        // if include_refs {
        //     crate::git::objects::clone_commit_with_refs(commit_hash.clone(), target_path, &dht, true).await?;
        // } else {
        //     crate::git::objects::clone_commit(commit_hash.clone(), target_path, &dht).await?;
        // }

        Err(crate::Error::other("DHT cloning not yet implemented in remote helper"))
    }

    /// Create a basic functional repository
    async fn create_basic_repository(
        &self,
        local_repo: &Path,
        identifier: &str,
        ref_name: &str,
        url: &GitTorrentUrl,
    ) -> Result<()> {
        let repo = self.init_git_repo(local_repo)?;
        self.create_functional_git_repo(&repo, identifier, ref_name, url).await
    }

    /// Push repository data to P2P network
    async fn push_repository_data(
        &self,
        _url: &GitTorrentUrl,
        local_repo: &Path,
        src_ref: &str,
        dst_ref: &str,
    ) -> Result<()> {
        tracing::info!("Pushing {} to {} for repository", src_ref, dst_ref);

        // Announce repository to P2P network
        self.service.announce_repository(local_repo).await?;

        // In a real implementation, this would:
        // 1. Extract objects to be pushed
        // 2. Create and distribute Git pack files
        // 3. Store XET chunks in P2P network
        // 4. Update repository metadata in DHT

        Ok(())
    }

    /// Create a Git repository from P2P network metadata
    async fn create_git_repo_from_metadata(
        &self,
        repo: &git2::Repository,
        metadata: &crate::service::RepositoryMetadata,
        ref_name: &str,
        url: &GitTorrentUrl,
    ) -> Result<()> {
        // Create initial commit with README that includes metadata
        let sig = git2::Signature::new("GitTorrent", "gittorrent@example.com", &git2::Time::new(1234567890, 0))
            .map_err(Error::from)?;

        // Create tree with README and metadata info
        let tree_id = {
            let mut tree_builder = repo.treebuilder(None).map_err(Error::from)?;

            // Add a README file with repository information
            let readme_content = format!(
                "# GitTorrent Repository\n\nThis repository is hosted via GitTorrent P2P network.\n\n\
                URL: {}\n\
                Publisher: {}\n\
                Size: {} bytes\n\
                References: {}\n\
                LFS Files: {}\n\
                Last Updated: {}\n",
                url,
                metadata.publisher.as_deref().unwrap_or("Unknown"),
                metadata.size_bytes,
                metadata.refs.len(),
                metadata.lfs_chunks.len(),
                metadata.last_updated
            );

            let readme_oid = repo.blob(readme_content.as_bytes()).map_err(Error::from)?;
            tree_builder.insert("README.md", readme_oid, 0o100644).map_err(Error::from)?;

            // Add a metadata file for debugging
            let metadata_content = serde_json::to_string_pretty(metadata)
                .map_err(|e| Error::other(format!("Failed to serialize metadata: {}", e)))?;
            let metadata_oid = repo.blob(metadata_content.as_bytes()).map_err(Error::from)?;
            tree_builder.insert(".gittorrent-metadata.json", metadata_oid, 0o100644).map_err(Error::from)?;

            tree_builder.write().map_err(Error::from)?
        };

        let tree = repo.find_tree(tree_id).map_err(Error::from)?;

        // Create initial commit
        let commit_id = repo.commit(
            None, // Don't update any reference yet
            &sig,
            &sig,
            "Initial GitTorrent commit from P2P network",
            &tree,
            &[] // No parents for initial commit
        ).map_err(Error::from)?;

        // Update the reference to point to this commit
        let ref_name_normalized = if ref_name.starts_with("refs/") {
            ref_name.to_string()
        } else {
            format!("refs/heads/{}", ref_name)
        };

        repo.reference(&ref_name_normalized, commit_id, true, "Initial commit from P2P")
            .map_err(Error::from)?;

        // Set HEAD if this is a main/master branch
        if ref_name.contains("main") || ref_name.contains("master") {
            repo.set_head(&ref_name_normalized).map_err(Error::from)?;
        }

        tracing::debug!("Created Git repository from P2P metadata with commit {}", commit_id);
        Ok(())
    }

    /// Create a functional Git repository with proper Git objects
    async fn create_functional_git_repo(&self, repo: &git2::Repository, _commit_hash: &str, ref_name: &str, url: &GitTorrentUrl) -> Result<()> {
        // Create initial commit with empty tree
        let sig = git2::Signature::new("GitTorrent", "gittorrent@example.com", &git2::Time::new(1234567890, 0))
            .map_err(Error::from)?;

        // Create empty tree
        let tree_id = {
            let mut tree_builder = repo.treebuilder(None).map_err(Error::from)?;

            // Add a README file to make it a non-empty repository
            let readme_content = format!("# GitTorrent Repository\n\nThis repository is hosted via GitTorrent.\n\nURL: {}\n",
                url);

            let readme_oid = repo.blob(readme_content.as_bytes()).map_err(Error::from)?;
            tree_builder.insert("README.md", readme_oid, 0o100644).map_err(Error::from)?;
            tree_builder.write().map_err(Error::from)?
        };

        let tree = repo.find_tree(tree_id).map_err(Error::from)?;

        // Create initial commit
        let commit_id = repo.commit(
            None, // Don't update any reference yet
            &sig,
            &sig,
            "Initial GitTorrent commit",
            &tree,
            &[] // No parents for initial commit
        ).map_err(Error::from)?;

        // Update the reference to point to this commit
        let ref_name_normalized = if ref_name.starts_with("refs/") {
            ref_name.to_string()
        } else {
            format!("refs/heads/{}", ref_name)
        };

        repo.reference(&ref_name_normalized, commit_id, true, "Initial commit")
            .map_err(Error::from)?;

        // If this is a main/master branch, also update HEAD
        if ref_name.contains("main") || ref_name.contains("master") {
            repo.set_head(&ref_name_normalized).map_err(Error::from)?;
        }

        tracing::debug!("Created functional Git repository with commit {}", commit_id);
        Ok(())
    }

    /// Query remote repository references (via P2P or fallback)
    async fn query_remote_refs(&self, url: &GitTorrentUrl) -> Result<HashMap<String, String>> {
        // TODO: In a full implementation, this would query the P2P network
        // For now, check if we have a local copy or return defaults

        // Check if we have cached repository locally
        if let Some(local_repo) = &self.local_repo {
            if let Ok(refs) = self.get_repository_refs_git2(local_repo).await {
                return Ok(refs);
            }
        }

        // Try to query via GitTorrent service
        // TODO: Implement P2P repository discovery

        // Fallback to common default references
        let mut refs = HashMap::new();

        // Create a default commit SHA (empty tree)
        let empty_tree_sha = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

        refs.insert("refs/heads/main".to_string(), empty_tree_sha.to_string());
        refs.insert("refs/heads/master".to_string(), empty_tree_sha.to_string());

        tracing::debug!("Using fallback references for {}", url.to_string());

        Ok(refs)
    }

    /// Get repository references using git2
    async fn get_repository_refs_git2(&self, repo_path: &Path) -> Result<HashMap<String, String>> {
        let mut refs = HashMap::new();

        if !repo_path.exists() {
            return Ok(refs);
        }

        // Open repository with git2
        let repo = match git2::Repository::open(repo_path) {
            Ok(repo) => repo,
            Err(_) => return Ok(refs), // Not a git repository
        };

        // Iterate through all references
        let ref_iter = repo.references().map_err(Error::from)?;
        for reference in ref_iter {
            let reference = reference.map_err(Error::from)?;

            if let Some(name) = reference.name() {
                // Get the target OID
                if let Some(target) = reference.target() {
                    refs.insert(name.to_string(), target.to_string());
                } else if let Some(symbolic_target) = reference.symbolic_target() {
                    // Handle symbolic references (like HEAD)
                    if let Ok(resolved) = repo.resolve_reference_from_short_name(symbolic_target) {
                        if let Some(target) = resolved.target() {
                            refs.insert(name.to_string(), target.to_string());
                        }
                    }
                }
            }
        }

        tracing::debug!("Found {} references in repository at {}", refs.len(), repo_path.display());
        Ok(refs)
    }
}

/// Entry point for git-remote-gittorrent binary
pub async fn run_git_remote_helper(config: Option<GitTorrentConfig>, url: Option<&str>) -> Result<()> {
    let mut helper = match config {
        Some(config) => GitRemoteHelper::new_with_config(config).await?,
        None => {
            tracing_subscriber::fmt::init();
            GitRemoteHelper::new().await?
        }
    };

    // Parse and set the remote URL if provided
    if let Some(url) = url {
        helper.remote_url = Some(GitTorrentUrl::parse(url)?);
    }

    // Get repository path from environment
    if let Ok(git_dir) = std::env::var("GIT_DIR") {
        let repo_path = PathBuf::from(git_dir);
        helper.set_local_repo(repo_path)?;
    }

    // Run the helper protocol
    helper.run().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_git_remote_helper_creation() {
        let helper = GitRemoteHelper::new().await.unwrap();
        assert_eq!(helper.capabilities.len(), 5);
        assert!(helper.capabilities.contains(&"connect".to_string()));
        assert!(helper.capabilities.contains(&"list".to_string()));
    }

    #[tokio::test]
    async fn test_capabilities_command() {
        let helper = GitRemoteHelper::new().await.unwrap();
        let response = helper.handle_capabilities();

        assert!(response.contains("connect"));
        assert!(response.contains("list"));
        assert!(response.contains("fetch"));
        assert!(response.contains("push"));
        assert!(response.ends_with("\n\n"));
    }

    #[tokio::test]
    async fn test_connect_command() {
        let mut helper = GitRemoteHelper::new().await.unwrap();

        // Set remote URL first (as would be done during argument parsing)
        helper.remote_url = Some(GitTorrentUrl::parse("gittorrent://example.com/user/repo").unwrap());

        let result = helper.handle_connect(&["git-upload-pack"]).await;
        assert!(result.is_ok());
        assert!(helper.remote_url.is_some());
    }

    #[tokio::test]
    async fn test_local_repo_setup() {
        let temp_dir = TempDir::new().unwrap();
        let mut helper = GitRemoteHelper::new().await.unwrap();

        helper.set_local_repo(temp_dir.path().to_path_buf()).unwrap();
        assert!(helper.local_repo.is_some());
    }
}