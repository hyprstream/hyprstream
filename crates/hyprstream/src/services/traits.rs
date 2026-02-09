//! Service trait definitions for registry operations.
//!
//! These traits define the contract for registry client implementations.
//! Implementations can use ZMQ/Cap'n Proto (RegistryZmqClient) or other transports.

use async_trait::async_trait;
use git2db::{Git2DBError, GitRef, RepoId, RepositoryStatus, TrackedRepository};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

use crate::storage::ModelRef;

// Re-export RemoteInfo from rpc_types for public API
pub use super::rpc_types::RemoteInfo;

/// File change type for detailed status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileChangeType {
    /// File added to index
    Added,
    /// File modified
    Modified,
    /// File deleted
    Deleted,
    /// File renamed
    Renamed,
    /// Untracked file
    Untracked,
    /// File type changed
    TypeChanged,
    /// Conflicted (during merge)
    Conflicted,
}

impl FileChangeType {
    /// Get the single-character status indicator (like git status --short).
    pub fn as_char(&self) -> char {
        match self {
            Self::Added => 'A',
            Self::Modified => 'M',
            Self::Deleted => 'D',
            Self::Renamed => 'R',
            Self::Untracked => '?',
            Self::TypeChanged => 'T',
            Self::Conflicted => 'U',
        }
    }
}

/// A file with its change status.
#[derive(Debug, Clone)]
pub struct FileStatus {
    /// Relative path to the file
    pub path: String,
    /// Status in the index (staging area)
    pub index_status: Option<FileChangeType>,
    /// Status in the working tree
    pub worktree_status: Option<FileChangeType>,
}

impl FileStatus {
    /// Format as git status --short style (e.g., "M ", " M", "MM", "??")
    pub fn format_short(&self) -> String {
        let index = self.index_status.map(|s| s.as_char()).unwrap_or(' ');
        let worktree = self.worktree_status.map(|s| s.as_char()).unwrap_or(' ');
        format!("{}{}", index, worktree)
    }
}

/// Detailed repository status with file-level change information.
#[derive(Debug, Clone)]
pub struct DetailedStatus {
    /// Current branch name (if any)
    pub branch: Option<String>,
    /// HEAD commit OID
    pub head: Option<String>,
    /// Whether there's a merge in progress
    pub merge_in_progress: bool,
    /// Whether there's a rebase in progress
    pub rebase_in_progress: bool,
    /// Files with changes (both staged and unstaged)
    pub files: Vec<FileStatus>,
    /// Number of commits ahead of upstream
    pub ahead: usize,
    /// Number of commits behind upstream
    pub behind: usize,
}

impl DetailedStatus {
    /// Get only staged files.
    pub fn staged_files(&self) -> impl Iterator<Item = &FileStatus> {
        self.files.iter().filter(|f| f.index_status.is_some())
    }

    /// Get only unstaged files (modified in working tree).
    pub fn unstaged_files(&self) -> impl Iterator<Item = &FileStatus> {
        self.files.iter().filter(|f| {
            f.worktree_status.is_some() && f.index_status.is_none()
        })
    }

    /// Get only untracked files.
    pub fn untracked_files(&self) -> impl Iterator<Item = &FileStatus> {
        self.files.iter().filter(|f| {
            matches!(f.worktree_status, Some(FileChangeType::Untracked))
        })
    }

    /// Check if working tree is clean.
    pub fn is_clean(&self) -> bool {
        self.files.is_empty()
    }

    /// Check if there are conflicts.
    pub fn has_conflicts(&self) -> bool {
        self.files.iter().any(|f| {
            matches!(f.index_status, Some(FileChangeType::Conflicted))
                || matches!(f.worktree_status, Some(FileChangeType::Conflicted))
        })
    }
}

/// Worktree information returned by list_worktrees.
#[derive(Debug, Clone)]
pub struct WorktreeInfo {
    /// Path to the worktree
    pub path: PathBuf,
    /// Branch name (if associated with a branch)
    pub branch: Option<String>,
    /// Storage driver name
    pub driver: String,
    /// Whether the worktree has uncommitted changes
    pub is_dirty: bool,
}

/// Model information for listing (combines repo + worktree info).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Display name in "model:branch" format
    pub display_name: String,
    /// Model name (repository name)
    pub model: String,
    /// Branch name
    pub branch: String,
    /// Path to the worktree
    pub path: PathBuf,
    /// Whether the worktree has uncommitted changes
    pub is_dirty: bool,
    /// Storage driver name
    pub driver: String,
}

/// Options for cloning a repository.
#[derive(Debug, Clone, Default)]
pub struct CloneOptions {
    /// Clone only the most recent commits (shallow clone).
    /// If `depth` is 0, defaults to depth=1.
    pub shallow: bool,
    /// Number of commits to fetch (0 = full history unless `shallow` is true).
    pub depth: u32,
    /// Branch to clone (None = default branch).
    pub branch: Option<String>,
}

impl CloneOptions {
    /// Create options for a full clone (all history).
    pub fn full() -> Self {
        Self {
            shallow: false,
            depth: 0,
            branch: None,
        }
    }

    /// Create options for a shallow clone with specified depth.
    pub fn shallow(depth: u32) -> Self {
        Self {
            shallow: true,
            depth: if depth == 0 { 1 } else { depth },
            branch: None,
        }
    }

    /// Set the branch to clone.
    pub fn with_branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = Some(branch.into());
        self
    }
}

/// Service error type for registry operations.
///
/// Split into:
/// - Registry: Underlying git2db errors
/// - Transport: Service communication errors (ZMQ, Cap'n Proto, etc.)
/// - Unavailable: Service not ready
#[derive(Debug, Error)]
pub enum RegistryServiceError {
    /// Registry operation failed (wraps underlying Git2DBError).
    #[error("Registry operation failed: {0}")]
    Registry(#[from] Git2DBError),

    /// Service communication failed (ZMQ, channel, network, etc.).
    #[error("Service communication failed: {0}")]
    Transport(String),

    /// Service is unavailable (not started, shutdown, etc.).
    #[error("Service unavailable")]
    Unavailable,

    /// Resource not found (model, worktree, branch, etc.).
    #[error("Not found: {0}")]
    NotFound(String),
}

impl RegistryServiceError {
    /// Create a transport error.
    pub fn transport<S: Into<String>>(msg: S) -> Self {
        Self::Transport(msg.into())
    }
}

/// Filesystem service error type.
#[derive(Debug, Error)]
pub enum FsServiceError {
    /// Bad file descriptor.
    #[error("Bad file descriptor: {0}")]
    BadFd(u32),
    /// Path or file not found.
    #[error("Not found: {0}")]
    NotFound(String),
    /// Permission denied (FD not owned by caller, or access denied).
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    /// Underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Path escaped containment root (symlink or traversal attack).
    #[error("Path containment violation: {0}")]
    PathEscape(String),
    /// Resource limit exceeded (too many FDs, IO size too large).
    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),
    /// Transport / communication error.
    #[error("Transport error: {0}")]
    Transport(String),
    /// Service is unavailable.
    #[error("Service unavailable")]
    Unavailable,
}

// Note: FsServiceError derives thiserror::Error which provides std::error::Error.
// anyhow's blanket `From<E: std::error::Error>` handles the conversion automatically.

/// File stat information.
#[derive(Debug, Clone)]
pub struct FsStatInfo {
    pub exists: bool,
    pub is_dir: bool,
    pub size: u64,
    pub modified_at: i64,
}

/// Directory entry information.
#[derive(Debug, Clone)]
pub struct FsDirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

/// Seek direction for filesystem operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekWhence {
    /// From beginning of file (SEEK_SET).
    Set,
    /// From current position (SEEK_CUR).
    Cur,
    /// From end of file (SEEK_END).
    End,
}

/// Maximum I/O size per read/write operation (16 MiB).
pub const MAX_FS_IO_SIZE: u64 = 16 * 1024 * 1024;
/// Maximum open file descriptors per client.
pub const MAX_FDS_PER_CLIENT: u32 = 64;
/// Maximum open file descriptors globally.
pub const MAX_FDS_GLOBAL: u32 = 4096;

/// POSIX-like filesystem operations trait (async, for remote callers).
///
/// Server-side handlers are synchronous (matching existing pattern).
/// This trait is implemented by the generated FsClient over ZMQ.
#[async_trait]
pub trait FsOps: Send + Sync {
    /// Open a file, returning a file descriptor.
    async fn open(
        &self,
        path: &str,
        write: bool,
        create: bool,
        truncate: bool,
    ) -> Result<u32, FsServiceError>;

    /// Close a file descriptor.
    async fn close(&self, fd: u32) -> Result<(), FsServiceError>;

    /// Read up to `len` bytes from a file descriptor.
    async fn read(&self, fd: u32, len: u64) -> Result<Vec<u8>, FsServiceError>;

    /// Write bytes to a file descriptor.
    async fn write(&self, fd: u32, data: &[u8]) -> Result<u64, FsServiceError>;

    /// Read at a specific offset without changing the file position.
    async fn pread(&self, fd: u32, offset: u64, len: u64) -> Result<Vec<u8>, FsServiceError>;

    /// Write at a specific offset without changing the file position.
    async fn pwrite(&self, fd: u32, offset: u64, data: &[u8]) -> Result<u64, FsServiceError>;

    /// Seek to a position in the file.
    async fn seek(&self, fd: u32, offset: i64, whence: SeekWhence) -> Result<u64, FsServiceError>;

    /// Truncate a file to the given length.
    async fn truncate(&self, fd: u32, len: u64) -> Result<(), FsServiceError>;

    /// Sync file data (and optionally metadata) to disk.
    async fn fsync(&self, fd: u32, data_only: bool) -> Result<(), FsServiceError>;

    /// Stat a path.
    async fn stat(&self, path: &str) -> Result<FsStatInfo, FsServiceError>;

    /// Create a directory (optionally recursive).
    async fn mkdir(&self, path: &str, recursive: bool) -> Result<(), FsServiceError>;

    /// Remove a file.
    async fn remove(&self, path: &str) -> Result<(), FsServiceError>;

    /// Remove an empty directory.
    async fn rmdir(&self, path: &str) -> Result<(), FsServiceError>;

    /// Rename a file or directory.
    async fn rename(&self, src: &str, dst: &str) -> Result<(), FsServiceError>;

    /// Copy a file (uses reflink/COW when possible).
    async fn copy(&self, src: &str, dst: &str) -> Result<(), FsServiceError>;

    /// List directory entries.
    async fn list_dir(&self, path: &str) -> Result<Vec<FsDirEntry>, FsServiceError>;

    // =========================================================================
    // Convenience methods (default implementations using primitives above)
    // =========================================================================

    /// Convenience: read entire file as bytes.
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, FsServiceError> {
        let fd = self.open(path, false, false, false).await?;
        let info = self.stat(path).await?;
        let data = self.read(fd, info.size).await?;
        self.close(fd).await?;
        Ok(data)
    }

    /// Convenience: read entire file as UTF-8 string.
    async fn read_to_string(&self, path: &str) -> Result<String, FsServiceError> {
        let data = self.read_file(path).await?;
        String::from_utf8(data).map_err(|e| {
            FsServiceError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })
    }

    /// Convenience: write entire file atomically (create+truncate+write+fsync+close).
    async fn write_file(&self, path: &str, data: &[u8]) -> Result<(), FsServiceError> {
        let fd = self.open(path, true, true, true).await?;
        self.write(fd, data).await?;
        self.fsync(fd, false).await?;
        self.close(fd).await?;
        Ok(())
    }

    /// Convenience: check if path exists.
    async fn exists(&self, path: &str) -> Result<bool, FsServiceError> {
        match self.stat(path).await {
            Ok(info) => Ok(info.exists),
            Err(FsServiceError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// Transport-agnostic registry client trait.
///
/// Implementations can be:
/// - `RegistryZmqClient` (ZMQ + Cap'n Proto)
/// - Direct (in-process, for testing)
///
/// All methods return owned data to avoid lifetime issues across transports.
#[async_trait]
pub trait RegistryClient: Send + Sync {
    // === Discovery (read-heavy, consider using cached_list for performance) ===

    /// List all tracked repositories.
    ///
    /// Goes through the service channel to get fresh data.
    /// For read-heavy workloads, prefer `cached_list()` when available.
    async fn list(&self) -> Result<Vec<TrackedRepository>, RegistryServiceError>;

    /// Get repository by ID.
    async fn get(&self, id: &RepoId) -> Result<Option<TrackedRepository>, RegistryServiceError>;

    /// Get repository by name.
    async fn get_by_name(&self, name: &str)
        -> Result<Option<TrackedRepository>, RegistryServiceError>;

    /// Fast path: get cached list (bypasses channel if available).
    ///
    /// Returns `None` if caching is not supported by this client.
    /// Default implementation returns `None`.
    fn cached_list(&self) -> Option<Vec<TrackedRepository>> {
        None
    }

    // === Convenience Helpers ===

    /// Check if a repository exists by name (sync, cache-only).
    ///
    /// Returns `true` if the repository exists in the cache, `false` otherwise.
    /// If no cache is available, returns `false` (use `exists_async` for reliable check).
    fn exists(&self, name: &str) -> bool {
        if let Some(repos) = self.cached_list() {
            repos.iter().any(|t| t.name.as_ref() == Some(&name.to_owned()))
        } else {
            false
        }
    }

    /// Check if a repository exists by name (async, reliable).
    ///
    /// Queries the service if not in cache. Use this when you need a definitive answer.
    async fn exists_async(&self, name: &str) -> bool {
        self.get_by_name(name).await.ok().flatten().is_some()
    }

    // === Mutation (always through channel) ===

    /// Clone a repository from URL.
    ///
    /// # Arguments
    /// * `url` - Repository URL to clone
    /// * `name` - Optional name for the repository (defaults to URL-derived name)
    /// * `options` - Clone options (shallow, depth, branch)
    async fn clone_repo(
        &self,
        url: &str,
        name: Option<&str>,
        options: &CloneOptions,
    ) -> Result<RepoId, RegistryServiceError>;

    /// Clone a repository with streaming progress.
    ///
    /// Returns StreamStartedInfo containing:
    /// - stream_id: For client display/logging
    /// - endpoint: StreamService SUB endpoint
    /// - server_pubkey: Server's ephemeral Ristretto255 public key for DH
    ///
    /// # Arguments
    /// * `url` - Repository URL to clone
    /// * `name` - Optional name for the repository
    /// * `options` - Clone options (shallow, depth, branch)
    /// * `ephemeral_pubkey` - Client's ephemeral Ristretto255 public key for DH
    async fn clone_stream(
        &self,
        url: &str,
        name: Option<&str>,
        options: &CloneOptions,
        ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<crate::services::rpc_types::StreamStartedInfo, RegistryServiceError>;

    /// Register an existing repository.
    ///
    /// # Arguments
    /// * `id` - Repository ID to assign
    /// * `name` - Optional name for the repository
    /// * `path` - Path to the existing repository
    async fn register(
        &self,
        id: &RepoId,
        name: Option<&str>,
        path: &Path,
    ) -> Result<(), RegistryServiceError>;

    /// Upsert: update if exists, create if not.
    ///
    /// This is useful for ensuring a repository exists without checking first.
    async fn upsert(&self, name: &str, url: &str) -> Result<RepoId, RegistryServiceError>;

    /// Remove a repository from the registry.
    async fn remove(&self, id: &RepoId) -> Result<(), RegistryServiceError>;

    // === Health ===

    /// Check service health (for testing/monitoring).
    ///
    /// Returns `Ok(())` if the service is healthy, error otherwise.
    async fn health_check(&self) -> Result<(), RegistryServiceError>;

    // === Repository Client Access ===

    /// Get a scoped repository client by name.
    ///
    /// Returns a client that provides repository-level operations (worktrees, branches, etc.).
    async fn repo(&self, name: &str) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError>;

    /// Get a scoped repository client by ID.
    async fn repo_by_id(
        &self,
        id: &RepoId,
    ) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError>;

    // === Convenience Methods ===

    /// Get the path for a model worktree.
    ///
    /// Convenience method that combines `repo()` and `worktree_path()`.
    /// If branch is None, uses the default branch.
    ///
    /// # Arguments
    /// * `model` - Model (repository) name
    /// * `branch` - Branch name, or None for default branch
    ///
    /// # Returns
    /// Path to the worktree, or error if not found
    async fn model_path(
        &self,
        model: &str,
        branch: Option<&str>,
    ) -> Result<PathBuf, RegistryServiceError> {
        let repo_client = self.repo(model).await?;

        // Get branch name (use default if not specified)
        let branch_name = match branch {
            Some(b) => b.to_owned(),
            None => repo_client.default_branch().await?,
        };

        // Get worktree path
        repo_client
            .worktree_path(&branch_name)
            .await?
            .ok_or_else(|| RegistryServiceError::NotFound(format!(
                "worktree for {}:{} not found",
                model, branch_name
            )))
    }

    /// Get model path from a ModelRef.
    ///
    /// This is a convenience method that extracts model name and branch from
    /// a ModelRef and calls model_path().
    async fn get_model_path(&self, model_ref: &ModelRef) -> Result<PathBuf, RegistryServiceError> {
        let branch = match &model_ref.git_ref {
            GitRef::Branch(name) => Some(name.as_str()),
            GitRef::DefaultBranch => None,
            other => {
                tracing::warn!(
                    "Model reference specifies non-branch git ref {:?}, using default branch",
                    other
                );
                None
            }
        };
        self.model_path(&model_ref.model, branch).await
    }

    /// List all models (worktrees across all repositories).
    ///
    /// Returns a list of all available model:branch combinations.
    /// This is the replacement for ModelStorage::list_models().
    async fn list_models(&self) -> Result<Vec<ModelInfo>, RegistryServiceError> {
        let repos = self.list().await?;
        let mut models = Vec::new();

        for repo in repos {
            if let Some(name) = &repo.name {
                // Get repo client to list worktrees
                match self.repo(name).await {
                    Ok(repo_client) => {
                        match repo_client.list_worktrees().await {
                            Ok(worktrees) => {
                                for wt in worktrees {
                                    if let Some(branch) = wt.branch {
                                        models.push(ModelInfo {
                                            display_name: format!("{}:{}", name, branch),
                                            model: name.clone(),
                                            branch: branch.clone(),
                                            path: wt.path,
                                            is_dirty: wt.is_dirty,
                                            driver: wt.driver,
                                        });
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Failed to list worktrees for {}: {}", name, e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to get repo client for {}: {}", name, e);
                    }
                }
            }
        }

        Ok(models)
    }
}

/// Scoped repository operations (mirrors RepositoryHandle).
///
/// This trait provides repository-level operations that work over the service layer.
/// Implementations send requests to the service, which uses RepositoryHandle internally.
#[async_trait]
pub trait RepositoryClient: Send + Sync {
    /// Repository name
    fn name(&self) -> &str;

    /// Repository ID
    fn id(&self) -> &RepoId;

    // === Worktree Operations ===

    /// Create a new worktree for a branch.
    ///
    /// This properly handles LFS/XET file smudging via the service.
    async fn create_worktree(
        &self,
        path: &Path,
        branch: &str,
    ) -> Result<PathBuf, RegistryServiceError>;

    /// List all worktrees.
    async fn list_worktrees(&self) -> Result<Vec<WorktreeInfo>, RegistryServiceError>;

    /// Get the path for a worktree.
    async fn worktree_path(&self, branch: &str) -> Result<Option<PathBuf>, RegistryServiceError>;

    /// Ensure a worktree exists for a branch, creating it if necessary.
    ///
    /// This is a convenience method that:
    /// 1. Checks if a worktree already exists for the branch
    /// 2. Creates one at the server's default location if not
    /// 3. Returns the path to the worktree
    ///
    /// The server determines where worktrees are stored.
    async fn ensure_worktree(&self, branch: &str) -> Result<PathBuf, RegistryServiceError> {
        // Check if worktree already exists
        if let Some(path) = self.worktree_path(branch).await? {
            return Ok(path);
        }

        // Get the bare repo path from list_worktrees or similar
        // The server will compute the default worktree path
        let worktrees = self.list_worktrees().await?;
        let base_path = if let Some(wt) = worktrees.first() {
            // Derive worktrees directory from existing worktree
            wt.path.parent()
                .ok_or_else(|| RegistryServiceError::Transport("Cannot determine worktrees directory".to_owned()))?
                .to_path_buf()
        } else {
            // No worktrees exist - need server to tell us the path
            // For now, return an error - the server should have at least main worktree
            return Err(RegistryServiceError::Transport(
                "No existing worktrees to derive path from. Use create_worktree with explicit path.".to_owned()
            ));
        };

        let worktree_path = base_path.join(branch);
        self.create_worktree(&worktree_path, branch).await
    }

    // === Branch Operations ===

    /// Create a new branch.
    async fn create_branch(
        &self,
        name: &str,
        from: Option<&str>,
    ) -> Result<(), RegistryServiceError>;

    /// Checkout a branch or ref.
    async fn checkout(&self, ref_spec: &str) -> Result<(), RegistryServiceError>;

    /// Get the default branch name.
    async fn default_branch(&self) -> Result<String, RegistryServiceError>;

    /// List all branches.
    async fn list_branches(&self) -> Result<Vec<String>, RegistryServiceError>;

    /// Merge a branch or reference into the current HEAD.
    ///
    /// # Arguments
    /// * `source` - The branch or ref to merge (e.g., "feature-branch", "refs/tags/v1.0")
    /// * `message` - Optional commit message (defaults to "Merge branch 'source'")
    ///
    /// Returns the merge commit OID as a string.
    async fn merge(&self, source: &str, message: Option<&str>) -> Result<String, RegistryServiceError>;

    /// Remove a worktree.
    async fn remove_worktree(&self, path: &Path) -> Result<(), RegistryServiceError>;

    // === Staging/Commit Operations ===

    /// Stage all changes in the repository.
    async fn stage_all(&self) -> Result<(), RegistryServiceError>;

    /// Stage specific files.
    async fn stage_files(&self, files: &[&str]) -> Result<(), RegistryServiceError>;

    /// Commit staged changes with a message.
    ///
    /// Returns the commit OID as a string.
    async fn commit(&self, message: &str) -> Result<String, RegistryServiceError>;

    /// Get repository status (branch, dirty files, etc.)
    async fn status(&self) -> Result<RepositoryStatus, RegistryServiceError>;

    // === Reference Operations ===

    /// Get the HEAD commit OID.
    async fn get_head(&self) -> Result<String, RegistryServiceError>;

    /// Get the OID for a named reference.
    async fn get_ref(&self, ref_name: &str) -> Result<String, RegistryServiceError>;

    /// Update repository from remote (fetch).
    ///
    /// # Arguments
    /// * `refspec` - Optional refspec to fetch (e.g., "refs/heads/main")
    async fn update(&self, refspec: Option<&str>) -> Result<(), RegistryServiceError>;

    // === Remote Operations ===

    /// List all remotes.
    async fn list_remotes(&self) -> Result<Vec<RemoteInfo>, RegistryServiceError>;

    /// Add a new remote.
    async fn add_remote(&self, name: &str, url: &str) -> Result<(), RegistryServiceError>;

    /// Remove a remote.
    async fn remove_remote(&self, name: &str) -> Result<(), RegistryServiceError>;

    /// Change a remote's URL.
    async fn set_remote_url(&self, name: &str, url: &str) -> Result<(), RegistryServiceError>;

    /// Rename a remote.
    async fn rename_remote(
        &self,
        old_name: &str,
        new_name: &str,
    ) -> Result<(), RegistryServiceError>;

    // === Push Operations ===

    /// Push to a remote.
    ///
    /// # Arguments
    /// * `remote` - Remote name (e.g., "origin")
    /// * `refspec` - Refspec to push (e.g., "refs/heads/main:refs/heads/main")
    /// * `force` - Force push (overwrite remote refs)
    async fn push(
        &self,
        remote: &str,
        refspec: &str,
        force: bool,
    ) -> Result<(), RegistryServiceError>;

    // === Advanced Commit Operations ===

    /// Amend the most recent commit.
    ///
    /// Updates the previous commit with staged changes and/or a new message.
    ///
    /// # Arguments
    /// * `message` - New commit message (required)
    ///
    /// Returns the new commit OID as a string.
    async fn amend_commit(&self, message: &str) -> Result<String, RegistryServiceError>;

    /// Commit with a custom author.
    ///
    /// # Arguments
    /// * `message` - Commit message
    /// * `author_name` - Author name
    /// * `author_email` - Author email
    ///
    /// Returns the commit OID as a string.
    async fn commit_with_author(
        &self,
        message: &str,
        author_name: &str,
        author_email: &str,
    ) -> Result<String, RegistryServiceError>;

    /// Stage all files including untracked.
    ///
    /// This is equivalent to `git add -A`.
    async fn stage_all_including_untracked(&self) -> Result<(), RegistryServiceError>;

    // === Merge Conflict Resolution ===

    /// Abort a merge in progress.
    ///
    /// Resets HEAD to ORIG_HEAD and cleans up merge state.
    async fn abort_merge(&self) -> Result<(), RegistryServiceError>;

    /// Continue a merge after conflict resolution.
    ///
    /// Checks that all conflicts are resolved and creates the merge commit.
    ///
    /// # Arguments
    /// * `message` - Optional commit message (defaults to auto-generated message)
    ///
    /// Returns the merge commit OID as a string.
    async fn continue_merge(&self, message: Option<&str>) -> Result<String, RegistryServiceError>;

    /// Quit a merge in progress, keeping working tree changes.
    ///
    /// Removes merge state files (MERGE_HEAD, etc.) but keeps working tree intact.
    async fn quit_merge(&self) -> Result<(), RegistryServiceError>;

    // === Tags ===

    /// List all tags.
    async fn list_tags(&self) -> Result<Vec<String>, RegistryServiceError>;

    /// Create a lightweight tag.
    async fn create_tag(&self, name: &str, target: Option<&str>) -> Result<(), RegistryServiceError>;

    /// Delete a tag.
    async fn delete_tag(&self, name: &str) -> Result<(), RegistryServiceError>;

    // === Detailed Status ===

    /// Get detailed repository status with file-level change information.
    ///
    /// This provides more detailed information than `status()`, including
    /// per-file change types (Added, Modified, Deleted, etc.).
    async fn detailed_status(&self) -> Result<DetailedStatus, RegistryServiceError>;

    // === Filesystem Operations ===

    /// Get a POSIX filesystem client scoped to a worktree.
    ///
    /// Returns a client providing path-contained file operations.
    /// All paths are relative to the worktree root.
    ///
    /// # Example
    /// ```ignore
    /// let wt = repo_client.worktree("main");
    /// let fd = wt.open("config.json", false, false, false).await?;
    /// let data = wt.read(fd, 4096).await?;
    /// wt.close(fd).await?;
    /// ```
    fn worktree(&self, name: &str) -> Arc<dyn FsOps>;
}

