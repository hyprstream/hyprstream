//! Core driver trait and types

use crate::errors::{Git2DBError, Git2DBResult};
use crate::references::GitRef;
use crate::references::IntoGitRef;
use crate::repository_handle::RepositoryStatus;
use async_trait::async_trait;
use git2::{Oid, Signature};
use std::fmt;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use inventory;


/// Storage driver selection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageDriver {
    Overlay2,
    Reflink,
    Vfs,
}

impl fmt::Display for StorageDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overlay2 => write!(f, "overlay2"),
            Self::Reflink => write!(f, "reflink"),
            Self::Vfs => write!(f, "vfs"),
        }
    }
}

impl std::str::FromStr for StorageDriver {
    type Err = DriverError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "overlay2" | "overlayfs" => Ok(Self::Overlay2),
            "reflink" => Ok(Self::Reflink),
            "vfs" | "none" => Ok(Self::Vfs),
            _ => Err(DriverError::UnknownDriver(s.to_string())),
        }
    }
}

/// Options for creating a worktree with a driver
#[derive(Debug, Clone)]
pub struct DriverOpts {
    /// Base repository path (lower layer for CoW drivers)
    pub base_repo: PathBuf,

    /// Target worktree path (where to mount/create)
    pub worktree_path: PathBuf,

    /// Git ref specification (branch, commit SHA, tag, HEAD~3, etc.)
    /// Examples: "main", "a1b2c3d4", "v1.0.0", "HEAD~3", "origin/main"
    pub ref_spec: String,
}


/// Driver-specific errors
#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    #[error("Unknown driver: {0}")]
    UnknownDriver(String),

    #[error("Driver not available: {0}")]
    NotAvailable(String),

    #[error("Driver operation failed: {0}")]
    OperationFailed(String),

    #[error("Git2DB error: {0}")]
    Git2DB(#[from] Git2DBError),
}

/// Driver factory for plugin discovery
///
/// This struct represents a factory for creating storage drivers that can be
/// dynamically discovered at runtime using the inventory crate. It separates
/// const-compatible registration from runtime driver creation.
#[derive(Clone)]
pub struct DriverFactory {
    /// Driver name for identification
    name: &'static str,

    /// Factory function to create the driver (returns Box for const compatibility)
    factory: fn() -> Box<dyn Driver>,
}

impl DriverFactory {
    /// Create a new driver factory with const-compatible registration
    pub const fn new(name: &'static str, factory: fn() -> Box<dyn Driver>) -> Self {
        Self {
            name,
            factory,
        }
    }

    /// Get the driver instance (wrapped in Arc for sharing)
    pub fn get_driver(&self) -> Arc<dyn Driver> {
        // Create Box<dyn Driver> and then convert to Arc<dyn Driver>
        // The Box can be safely coerced to the trait object
        let boxed: Box<dyn Driver> = (self.factory)();
        Arc::from(boxed)
    }

    /// Get the driver name
    pub fn name(&self) -> &'static str {
        self.name
    }
}

// Inventory registration for driver factories
inventory::collect!(DriverFactory);

/// Storage driver trait (Docker's graphdriver interface)
///
/// All storage drivers must implement this trait to provide worktree
/// creation with optional storage optimization.
#[async_trait]
pub trait Driver: Send + Sync {
    /// Get driver name (e.g., "overlay2", "vfs")
    fn name(&self) -> &'static str;

    /// Check if this driver is available on the current system
    fn is_available(&self) -> bool;

    /// Create a worktree using this driver
    ///
    /// This creates a git worktree, potentially with storage optimization
    /// layers underneath (overlayfs, reflinks, etc.).
    async fn create_worktree(&self, opts: &DriverOpts) -> Git2DBResult<WorktreeHandle>;

    /// Get all worktrees managed by this driver for the given repository
    ///
    /// Returns all existing worktrees for this repository that were created
    /// by this storage driver, maintaining storage optimization context.
    async fn get_worktrees(&self, base_repo: &Path) -> Git2DBResult<Vec<WorktreeHandle>>;

    /// Get specific worktree by branch name (if it exists)
    ///
    /// Returns the worktree handle if it exists and is managed by this driver,
    /// or None if the worktree doesn't exist or isn't managed by this driver.
    async fn get_worktree(&self, base_repo: &Path, branch: &str) -> Git2DBResult<Option<WorktreeHandle>>;
}

/// Async cleanup function type
pub type AsyncCleanupFn = Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = Git2DBResult<()>> + Send>> + Send>;

/// Handle to a created worktree
///
/// Provides access to the worktree and handles cleanup on drop.
/// Implements git repository operations and worktree management.
pub struct WorktreeHandle {
    /// Path to the worktree
    pub path: PathBuf,

    /// Driver that created this worktree
    pub driver_name: String,

    /// Async cleanup function, wrapped in Arc<Mutex> for Sync
    cleanup: Option<Arc<Mutex<Option<AsyncCleanupFn>>>>,

    /// Cached repository instance (to avoid reopening)
    repo: Option<git2::Repository>,
}

impl WorktreeHandle {
    /// Create a new worktree handle
    pub fn new(path: PathBuf, driver_name: String) -> Self {
        Self {
            path,
            driver_name,
            cleanup: None,
            repo: None,
        }
    }

    /// Create with async cleanup function
    pub fn with_cleanup<F, Fut>(
        path: PathBuf,
        driver_name: String,
        cleanup: F
    ) -> Self
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Git2DBResult<()>> + Send + 'static,
    {
        let cleanup_fn: AsyncCleanupFn = Box::new(move || Box::pin(cleanup()));
        Self {
            path,
            driver_name,
            cleanup: Some(Arc::new(Mutex::new(Some(cleanup_fn)))),
            repo: None,
        }
    }

    /// Open the worktree as a git repository
    pub fn open_repository(&self) -> Git2DBResult<git2::Repository> {
        git2::Repository::open(&self.path).map_err(|e| Git2DBError::internal(format!(
            "Failed to open worktree at {}: {}", self.path.display(), e
        )))
    }

    /// Get the driver name
    pub fn driver_name(&self) -> &str {
        &self.driver_name
    }

    /// Get the worktree path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the git repository with lazy loading and caching
    pub fn get_repository(&mut self) -> Git2DBResult<&mut git2::Repository> {
        if self.repo.is_none() {
            self.repo = Some(self.open_repository()?);
        }
        self.repo
            .as_mut()
            .ok_or_else(|| Git2DBError::internal("Repository not initialized"))
    }

    /// Get the current HEAD reference
    pub fn head(&mut self) -> Git2DBResult<git2::Reference<'_>> {
        let repo = self.get_repository()?;
        let head = repo.head()?;
        Ok(head)
    }

    /// Get the current commit
    pub fn head_commit(&mut self) -> Git2DBResult<git2::Commit<'_>> {
        let repo = self.get_repository()?;
        let head = repo.head()?;
        let commit = head.peel_to_commit()?;
        Ok(commit)
    }

    /// Check if the worktree is valid and accessible
    pub fn is_valid(&self) -> bool {
        self.path.exists() &&
        self.path.join(".git").exists() &&
        self.open_repository().is_ok()
    }

    /// Check if the worktree has uncommitted changes
    pub fn is_dirty(&mut self) -> Git2DBResult<bool> {
        let repo = self.get_repository()?;
        let mut opts = git2::StatusOptions::new();
        // Show working directory and staging area
        opts.include_untracked(true);
        opts.include_unmodified(true);
        let statuses = repo.statuses(Some(&mut opts))?;
        Ok(!statuses.is_empty())
    }

    /// Get basic worktree metadata
    pub fn metadata(&mut self) -> WorktreeMetadata {
        WorktreeMetadata {
            path: self.path.clone(),
            driver_name: self.driver_name.clone(),
            is_valid: self.is_valid(),
            is_dirty: self.is_dirty().unwrap_or(false),
            created_at: chrono::Utc::now(), // This should be actual creation time
        }
    }

    /// Async cleanup method
    pub async fn cleanup(&mut self) -> Git2DBResult<()> {
        if let Some(cleanup_arc) = self.cleanup.take() {
            let cleanup_fn = {
                if let Ok(mut cleanup_guard) = cleanup_arc.lock() {
                    cleanup_guard.take()
                } else {
                    None
                }
            };

            if let Some(cleanup_fn) = cleanup_fn {
                cleanup_fn().await?;
            }
        }
        Ok(())
    }

    /// Commit staged changes with default signature
    ///
    /// Similar to `git commit -m "<message>"`
    /// This operation is performed on the worktree's checked-out files.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// // Stage changes
    /// worktree.staging().add("README.md")?;
    ///
    /// // Commit changes
    /// let oid = worktree.commit("Update docs").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn commit(&mut self, message: &str) -> Git2DBResult<Oid> {
        // Get default signature from the global GitManager
        let git_manager = crate::manager::GitManager::global();
        let sig = git_manager.create_signature(None, None)?;

        self.commit_as(&sig, message).await
    }

    /// Commit staged changes with a specific author/committer
    ///
    /// Similar to `git commit -m "<message>" --author="<name> <email>"`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// use git2::Signature;
    ///
    /// // Stage changes
    /// worktree.staging().add("README.md")?;
    ///
    /// // Create custom signature
    /// let sig = Signature::now("Alice", "alice@example.com")?;
    ///
    /// // Commit as specific user
    /// let oid = worktree.commit_as(&sig, "Update docs").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn commit_as(&mut self, signature: &Signature<'_>, message: &str) -> Git2DBResult<Oid> {
        let repo = self.open_repository()?;

        // Get the current index
        let mut index = repo
            .index()
            .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?;

        // Write index to tree
        let tree_oid = index
            .write_tree()
            .map_err(|e| Git2DBError::internal(format!("Failed to write tree: {}", e)))?;

        let tree = repo
            .find_tree(tree_oid)
            .map_err(|e| Git2DBError::internal(format!("Failed to find tree: {}", e)))?;

        // Get parent commit (if any)
        let parent_commits = if let Ok(head) = repo.head() {
            vec![head.peel_to_commit().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD commit: {}", e))
            })?]
        } else {
            vec![]
        };

        let parent_refs: Vec<&git2::Commit> = parent_commits.iter().collect();

        // Create the commit
        let commit_oid = repo
            .commit(
                Some("HEAD"),
                signature,
                signature,
                message,
                &tree,
                &parent_refs,
            )
            .map_err(|e| Git2DBError::internal(format!("Failed to create commit: {}", e)))?;

        Ok(commit_oid)
    }

    /// Amend the last commit
    ///
    /// Similar to `git commit --amend`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// // Stage additional changes
    /// worktree.staging().add("forgotten_file.rs")?;
    ///
    /// // Amend previous commit
    /// let oid = worktree.amend(Some("Updated commit message")).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn amend(&mut self, message: Option<&str>) -> Git2DBResult<Oid> {
        let repo = self.open_repository()?;

        // Get HEAD commit
        let head = repo
            .head()
            .map_err(|e| Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e)))?;

        let head_commit = head.peel_to_commit().map_err(|e| {
            Git2DBError::reference("HEAD", format!("Failed to get HEAD commit: {}", e))
        })?;

        // Get current index
        let mut index = repo
            .index()
            .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?;

        // Write index to tree
        let tree_oid = index
            .write_tree()
            .map_err(|e| Git2DBError::internal(format!("Failed to write tree: {}", e)))?;

        let tree = repo
            .find_tree(tree_oid)
            .map_err(|e| Git2DBError::internal(format!("Failed to find tree: {}", e)))?;

        let commit_message = message.unwrap_or(head_commit.message().unwrap_or(""));

        // Create amended commit
        let commit_oid = repo
            .commit(
                Some("HEAD"),
                &head_commit.author(),
                &head_commit.committer(),
                commit_message,
                &tree,
                &[&head_commit],
            )
            .map_err(|e| Git2DBError::internal(format!("Failed to amend commit: {}", e)))?;

        Ok(commit_oid)
    }

    /// Merge a branch into the current worktree
    ///
    /// Similar to `git merge <branch>` with various merge options
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// // Merge a feature branch
    /// let merge_oid = worktree.merge("feature-branch", false, false).await?;
    /// println!("Merged commit: {}", merge_oid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn merge(
        &mut self,
        branch: &str,
        ff_only: bool,
        no_ff: bool,
    ) -> Git2DBResult<Oid> {
        let repo = self.open_repository()?;

        // Resolve branch reference
        let branch_ref = repo
            .find_reference(&format!("refs/heads/{}", branch))
            .or_else(|_| repo.find_reference(&format!("refs/remotes/origin/{}", branch)))
            .or_else(|_| repo.find_reference(branch))
            .map_err(|e| {
                Git2DBError::reference(branch, format!("Branch not found: {}", e))
            })?;

        let branch_commit = branch_ref.peel_to_commit().map_err(|e| {
            Git2DBError::reference(branch, format!("Failed to resolve commit: {}", e))
        })?;

        let annotated_commit = repo.find_annotated_commit(branch_commit.id()).map_err(|e| {
            Git2DBError::internal(format!("Failed to create annotated commit: {}", e))
        })?;

        // Perform merge analysis
        let (merge_analysis, _) = repo.merge_analysis(&[&annotated_commit]).map_err(|e| {
            Git2DBError::internal(format!("Merge analysis failed: {}", e))
        })?;

        // Already up-to-date
        if merge_analysis.is_up_to_date() {
            return Ok(branch_commit.id());
        }

        // Check fast-forward constraints
        if ff_only && !merge_analysis.is_fast_forward() {
            return Err(Git2DBError::merge_conflict(
                "Cannot fast-forward - branches have diverged",
            ));
        }

        // Fast-forward merge (if possible and not --no-ff)
        if merge_analysis.is_fast_forward() && !no_ff {
            let mut head_ref = repo.head().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
            })?;

            head_ref
                .set_target(branch_commit.id(), "Fast-forward merge")
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to update HEAD: {}", e))
                })?;

            repo.checkout_head(Some(
                git2::build::CheckoutBuilder::default()
                    .force()
            ))
            .map_err(|e| {
                Git2DBError::internal(format!("Checkout failed: {}", e))
            })?;

            return Ok(branch_commit.id());
        }

        // Regular merge (create merge commit)
        repo.merge(&[&annotated_commit], None, None)
            .map_err(|e| Git2DBError::internal(format!("Merge failed: {}", e)))?;

        // Check for conflicts
        if repo
            .index()
            .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?
            .has_conflicts()
        {
            return Err(Git2DBError::merge_conflict(
                "Merge conflicts detected - please resolve manually",
            ));
        }

        // Create merge commit
        let sig = crate::manager::GitManager::global().create_signature(None, None)?;

        let tree_id = repo
            .index()
            .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?
            .write_tree()
            .map_err(|e| Git2DBError::internal(format!("Failed to write tree: {}", e)))?;

        let tree = repo
            .find_tree(tree_id)
            .map_err(|e| Git2DBError::internal(format!("Failed to find tree: {}", e)))?;

        let parent = repo
            .head()
            .map_err(|e| Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e)))?
            .peel_to_commit()
            .map_err(|e| {
                Git2DBError::internal(format!("Failed to resolve HEAD commit: {}", e))
            })?;

        let merge_oid = repo
            .commit(
                Some("HEAD"),
                &sig,
                &sig,
                &format!("Merge branch '{}'", branch),
                &tree,
                &[&parent, &branch_commit],
            )
            .map_err(|e| Git2DBError::internal(format!("Failed to create merge commit: {}", e)))?;

        repo.cleanup_state()
            .map_err(|e| Git2DBError::internal(format!("Failed to cleanup merge state: {}", e)))?;

        Ok(merge_oid)
    }

    /// Get the current status of the worktree
    ///
    /// Returns branch information, modified files, repository state, and ahead/behind counts
    /// compared to the upstream tracking branch (if configured).
    ///
    /// The `ahead` and `behind` fields indicate how many commits the local branch is ahead
    /// or behind its upstream tracking branch. If no upstream is configured, both will be 0.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// let status = worktree.status().await?;
    /// println!("Current branch: {:?}", status.branch);
    /// println!("Modified files: {:?}", status.modified_files);
    /// println!("Is clean: {}", status.is_clean);
    /// println!("Ahead: {} commits", status.ahead);
    /// println!("Behind: {} commits", status.behind);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn status(&mut self) -> Git2DBResult<RepositoryStatus> {
        let repo = self.open_repository()?;

        let head = repo.head().ok();
        let branch = head
            .as_ref()
            .and_then(|h| h.shorthand())
            .map(|s| s.to_string());
        let head_oid = head.as_ref().and_then(|h| h.target());

        let statuses = repo.statuses(None).map_err(|e| {
            Git2DBError::internal(format!("Failed to get repository status: {}", e))
        })?;

        let is_clean = statuses.is_empty();
        let modified_files: Vec<PathBuf> = statuses
            .iter()
            .filter_map(|entry| entry.path().map(PathBuf::from))
            .collect();

        // Calculate ahead/behind counts if we have a branch with upstream
        let (ahead, behind) = if let (Some(branch_name), Some(local_oid)) = (&branch, head_oid) {
            // Try to find the local branch to get its upstream
            if let Ok(local_branch) = repo.find_branch(branch_name, git2::BranchType::Local) {
                // Get the upstream tracking branch
                if let Ok(upstream_branch) = local_branch.upstream() {
                    if let Some(upstream_oid) = upstream_branch.get().target() {
                        // Calculate ahead/behind using git2's graph_ahead_behind
                        match repo.graph_ahead_behind(local_oid, upstream_oid) {
                            Ok((a, b)) => (a, b),
                            Err(_) => {
                                // If calculation fails (e.g., no common ancestor), return zeros
                                (0, 0)
                            }
                        }
                    } else {
                        (0, 0)
                    }
                } else {
                    // No upstream tracking branch configured
                    (0, 0)
                }
            } else {
                // Couldn't find the branch (shouldn't happen, but handle gracefully)
                (0, 0)
            }
        } else {
            // No branch or no HEAD OID (detached HEAD or empty repo)
            (0, 0)
        };

        Ok(RepositoryStatus {
            branch,
            head: head_oid,
            ahead,
            behind,
            is_clean,
            modified_files,
        })
    }

    /// Checkout a specific reference (branch, tag, or commit)
    ///
    /// Accepts strings, GitRef enums, or direct Oids for type safety.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// // String reference
    /// worktree.checkout("main").await?;
    ///
    /// // Explicit GitRef
    /// worktree.checkout(git2db::GitRef::Branch("develop".into())).await?;
    ///
    /// // Direct Oid (type-safe)
    /// let oid = git2::Oid::from_str("abc123...")?;
    /// worktree.checkout(oid).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn checkout(&mut self, reference: impl IntoGitRef) -> Git2DBResult<()> {
        let repo = self.open_repository()?;
        let git_ref = reference.into_git_ref();

        // Resolve the reference to an OID
        let oid = match git_ref {
            GitRef::DefaultBranch => {
                let head = repo.head().map_err(|e| {
                    Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
                })?;
                head.target().ok_or_else(|| {
                    Git2DBError::reference("HEAD", "HEAD is not a direct reference")
                })?
            }
            GitRef::Branch(ref branch_name) => {
                let branch = repo
                    .find_branch(branch_name, git2::BranchType::Local)
                    .or_else(|_| repo.find_branch(branch_name, git2::BranchType::Remote))
                    .map_err(|e| {
                        Git2DBError::reference(branch_name, format!("Branch not found: {}", e))
                    })?;
                branch
                    .get()
                    .target()
                    .ok_or_else(|| Git2DBError::reference(branch_name, "Branch has no target"))?
            }
            GitRef::Tag(ref tag_name) => {
                let reference = repo
                    .find_reference(&format!("refs/tags/{}", tag_name))
                    .map_err(|e| {
                        Git2DBError::reference(tag_name, format!("Tag not found: {}", e))
                    })?;
                reference
                    .target()
                    .ok_or_else(|| Git2DBError::reference(tag_name, "Tag has no target"))?
            }
            GitRef::Commit(oid) => oid,
            GitRef::Revspec(ref spec) => {
                let obj = repo.revparse_single(spec).map_err(|e| {
                    Git2DBError::reference(spec, format!("Failed to resolve revspec: {}", e))
                })?;
                obj.id()
            }
        };

        // Checkout the commit
        let commit = repo
            .find_commit(oid)
            .map_err(|e| Git2DBError::internal(format!("Failed to find commit: {}", e)))?;

        repo.checkout_tree(commit.as_object(), None)
            .map_err(|e| Git2DBError::internal(format!("Failed to checkout tree: {}", e)))?;

        // Update HEAD
        repo.set_head_detached(oid)
            .map_err(|e| Git2DBError::internal(format!("Failed to update HEAD: {}", e)))?;

        Ok(())
    }

    /// Fetch from a remote
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// // Fetch from default remote (origin)
    /// worktree.fetch(None).await?;
    ///
    /// // Fetch from specific remote
    /// worktree.fetch(Some("upstream")).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn fetch(&mut self, remote: Option<&str>) -> Git2DBResult<()> {
        let repo = self.open_repository()?;
        let remote_name = remote.unwrap_or("origin");

        let mut remote_obj = repo
            .find_remote(remote_name)
            .map_err(|e| Git2DBError::reference(remote_name, format!("Remote not found: {}", e)))?;

        remote_obj.fetch(&[] as &[&str], None, None).map_err(|e| {
            Git2DBError::network(format!("Failed to fetch from {}: {}", remote_name, e))
        })?;

        Ok(())
    }

    /// Pull updates from remote (fetch + merge)
    ///
    /// Updates the current branch with changes from the remote tracking branch.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// // Pull from default remote (origin)
    /// worktree.pull(None).await?;
    ///
    /// // Pull from specific remote
    /// worktree.pull(Some("upstream")).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn pull(&mut self, remote: Option<&str>) -> Git2DBResult<()> {
        // First fetch from remote
        self.fetch(remote).await?;

        let repo = self.open_repository()?;
        let remote_name = remote.unwrap_or("origin");

        // Get current branch
        let head = repo.head().map_err(|e| {
            Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
        })?;
        let current_branch = head.shorthand().ok_or_else(|| {
            Git2DBError::reference("HEAD", "HEAD is detached")
        })?;

        // Get tracking branch (e.g., origin/main)
        let tracking_ref = format!("refs/remotes/{}/{}", remote_name, current_branch);
        let tracking_commit = repo
            .find_reference(&tracking_ref)
            .and_then(|r| r.peel_to_commit())
            .map_err(|e| Git2DBError::internal(format!("Failed to find tracking branch {}: {}", tracking_ref, e)))?;

        let annotated_commit = repo.find_annotated_commit(tracking_commit.id())?;

        // Check if we can fast-forward
        let (merge_analysis, _) = repo.merge_analysis(&[&annotated_commit])?;

        if merge_analysis.is_up_to_date() {
            return Ok(());
        }

        if merge_analysis.is_fast_forward() {
            // Fast-forward
            let commit_id = annotated_commit.id();
            let commit = repo.find_commit(commit_id).map_err(|e| Git2DBError::internal(format!("Failed to find commit: {}", e)))?;
            repo.checkout_tree(&commit.as_object(), None)
                .map_err(|e| Git2DBError::internal(format!("Failed to checkout tree: {}", e)))?;

            let mut head_ref = repo.head().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
            })?;

            head_ref
                .set_target(tracking_commit.id(), "Fast-forward pull")
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to update HEAD: {}", e))
                })?;
        } else {
            // Merge (create merge commit)
            repo.merge(&[&annotated_commit], None, None)
                .map_err(|e| Git2DBError::internal(format!("Merge failed: {}", e)))?;

            // Check for conflicts
            if repo
                .index()
                .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?
                .has_conflicts()
            {
                return Err(Git2DBError::merge_conflict(
                    "Pull conflicts detected - please resolve manually",
                ));
            }

            // Create merge commit
            let sig = crate::manager::GitManager::global().create_signature(None, None)?;

            let tree_id = repo
                .index()
                .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?
                .write_tree()
                .map_err(|e| Git2DBError::internal(format!("Failed to write tree: {}", e)))?;

            let tree = repo
                .find_tree(tree_id)
                .map_err(|e| Git2DBError::internal(format!("Failed to find tree: {}", e)))?;

            let parent = repo
                .head()
                .map_err(|e| Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e)))?
                .peel_to_commit()
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to resolve HEAD commit: {}", e))
                })?;

            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                &format!("Pull branch '{}' from {}", current_branch, remote_name),
                &tree,
                &[&parent, &tracking_commit],
            )
            .map_err(|e| Git2DBError::internal(format!("Failed to create merge commit: {}", e)))?;
        }

        repo.cleanup_state()
            .map_err(|e| Git2DBError::internal(format!("Failed to cleanup merge state: {}", e)))?;

        Ok(())
    }

    /// Push changes to remote
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// // Push to default remote (origin)
    /// worktree.push(None).await?;
    ///
    /// // Push to specific remote
    /// worktree.push(Some("upstream")).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn push(&mut self, remote: Option<&str>) -> Git2DBResult<()> {
        let repo = self.open_repository()?;
        let remote_name = remote.unwrap_or("origin");

        let mut remote_obj = repo
            .find_remote(remote_name)
            .map_err(|e| Git2DBError::reference(remote_name, format!("Remote not found: {}", e)))?;

        remote_obj.push(&[] as &[&str], None).map_err(|e| {
            Git2DBError::network(format!("Failed to push to {}: {}", remote_name, e))
        })?;

        Ok(())
    }

    /// Update to latest from tracking remote
    ///
    /// Equivalent to `git pull --ff-only` - only fast-forwards, no merges
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(worktree: git2db::storage::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
    /// // Update from default remote (origin)
    /// worktree.update().await?;
    ///
    /// // Update from specific remote
    /// worktree.update_with_remote("upstream").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn update(&mut self) -> Git2DBResult<()> {
        self.fetch(None).await?;

        let repo = self.open_repository()?;

        // Get current branch
        let head = repo.head().map_err(|e| {
            Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
        })?;
        let current_branch = head.shorthand().ok_or_else(|| {
            Git2DBError::reference("HEAD", "HEAD is detached")
        })?;

        // Get tracking branch
        let tracking_ref = format!("refs/remotes/origin/{}", current_branch);
        let tracking_commit = repo
            .find_reference(&tracking_ref)
            .and_then(|r| r.peel_to_commit())
            .map_err(|e| Git2DBError::internal(format!("Failed to find tracking branch {}: {}", tracking_ref, e)))?;

        let annotated_commit = repo.find_annotated_commit(tracking_commit.id())?;

        // Check if we can fast-forward
        let (merge_analysis, _) = repo.merge_analysis(&[&annotated_commit])?;

        if merge_analysis.is_up_to_date() {
            return Ok(());
        }

        if merge_analysis.is_fast_forward() {
            // Fast-forward only
            let commit_id = annotated_commit.id();
            let commit = repo.find_commit(commit_id).map_err(|e| Git2DBError::internal(format!("Failed to find commit: {}", e)))?;
            repo.checkout_tree(&commit.as_object(), None)
                .map_err(|e| Git2DBError::internal(format!("Failed to checkout tree: {}", e)))?;

            let mut head_ref = repo.head().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
            })?;

            head_ref
                .set_target(tracking_commit.id(), "Fast-forward update")
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to update HEAD: {}", e))
                })?;
        } else {
            return Err(Git2DBError::merge_conflict(
                "Cannot update - branches have diverged. Use `pull()` to merge.",
            ));
        }

        repo.cleanup_state()
            .map_err(|e| Git2DBError::internal(format!("Failed to cleanup merge state: {}", e)))?;

        Ok(())
    }

    /// Update from a specific remote
    pub async fn update_with_remote(&mut self, remote: &str) -> Git2DBResult<()> {
        self.fetch(Some(remote)).await?;

        let repo = self.open_repository()?;

        // Get current branch
        let head = repo.head().map_err(|e| {
            Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
        })?;
        let current_branch = head.shorthand().ok_or_else(|| {
            Git2DBError::reference("HEAD", "HEAD is detached")
        })?;

        // Get tracking branch
        let tracking_ref = format!("refs/remotes/{}/{}", remote, current_branch);
        let tracking_commit = repo
            .find_reference(&tracking_ref)
            .and_then(|r| r.peel_to_commit())
            .map_err(|e| Git2DBError::internal(format!("Failed to find tracking branch {}: {}", tracking_ref, e)))?;

        let annotated_commit = repo.find_annotated_commit(tracking_commit.id())?;

        // Check if we can fast-forward
        let (merge_analysis, _) = repo.merge_analysis(&[&annotated_commit])?;

        if merge_analysis.is_up_to_date() {
            return Ok(());
        }

        if merge_analysis.is_fast_forward() {
            // Fast-forward only
            let commit_id = annotated_commit.id();
            let commit = repo.find_commit(commit_id).map_err(|e| Git2DBError::internal(format!("Failed to find commit: {}", e)))?;
            repo.checkout_tree(&commit.as_object(), None)
                .map_err(|e| Git2DBError::internal(format!("Failed to checkout tree: {}", e)))?;

            let mut head_ref = repo.head().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
            })?;

            head_ref
                .set_target(tracking_commit.id(), &format!("Fast-forward update from {}", remote))
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to update HEAD: {}", e))
                })?;
        } else {
            return Err(Git2DBError::merge_conflict(
                &format!("Cannot update from {} - branches have diverged. Use `pull()` to merge.", remote),
            ));
        }

        repo.cleanup_state()
            .map_err(|e| Git2DBError::internal(format!("Failed to cleanup merge state: {}", e)))?;

        Ok(())
    }
}

/// Worktree metadata for inspection and debugging
#[derive(Debug, Clone)]
pub struct WorktreeMetadata {
    pub path: PathBuf,
    pub driver_name: String,
    pub is_valid: bool,
    pub is_dirty: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}


impl Drop for WorktreeHandle {
    fn drop(&mut self) {
        if self.cleanup.is_some() {
            tracing::warn!(
                "WorktreeHandle at {} dropped without calling cleanup() - \
                 resources may leak (overlayfs mounts, temporary directories)",
                self.path.display()
            );
        }
    }
}

impl fmt::Debug for WorktreeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorktreeHandle")
            .field("path", &self.path)
            .field("driver_name", &self.driver_name)
            .finish()
    }
}
