//! Reflink driver (cross-platform CoW)
//!
//! Implements the reflink storage driver using filesystem reflinks,
//! providing space-efficient worktrees through copy-on-write semantics.
//!
//! Supports:
//! - Linux: XFS, Btrfs, bcachefs
//! - macOS: APFS
//! - Windows: ReFS, Dev Drive

use super::driver::{Driver, DriverOpts, WorktreeHandle, DriverFactory};
use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use std::path::Path;
use tracing::{info, debug, warn};

#[cfg(feature = "reflink")]
use reflink_copy::reflink;

#[cfg(feature = "reflink")]
inventory::submit!(DriverFactory::new(
    "reflink",
    || Box::new(ReflinkDriver::new())
));

/// Configuration for reflink driver
#[derive(Debug, Clone)]
pub struct ReflinkConfig {
    /// Minimum file size to reflink (smaller files are skipped)
    pub min_size_bytes: u64,
}

impl Default for ReflinkConfig {
    fn default() -> Self {
        Self {
            min_size_bytes: 0, // Reflink all files by default
        }
    }
}

/// Reflink storage driver with cross-platform CoW support
///
/// This driver creates git worktrees and then replaces working files
/// with reflinks from the base repository, providing space savings
/// through copy-on-write semantics.
///
/// Unlike overlay2, this driver has no special cleanup requirements -
/// worktrees can be removed with a simple `rm -rf`.
pub struct ReflinkDriver {
    config: ReflinkConfig,
    /// Cached availability check result
    available: bool,
}

impl ReflinkDriver {
    /// Create with default configuration
    fn new() -> Self {
        let available = Self::check_availability();
        Self {
            config: ReflinkConfig::default(),
            available,
        }
    }

    /// Create with custom configuration
    fn with_config(config: ReflinkConfig) -> Self {
        let available = Self::check_availability();
        Self { config, available }
    }

    /// Check if reflinks are supported on the system
    ///
    /// Performs an actual reflink test in the temp directory to verify support.
    fn check_availability() -> bool {
        #[cfg(feature = "reflink")]
        {
            let temp_dir = std::env::temp_dir();

            // Use predictable names for temp files (no sensitive data)
            let test_src = temp_dir.join(".git2db_reflink_test_src");
            let test_dst = temp_dir.join(".git2db_reflink_test_dst");

            // Clean up any leftover test files
            let _ = std::fs::remove_file(&test_src);
            let _ = std::fs::remove_file(&test_dst);

            // Create simple test file (no sensitive data)
            if std::fs::write(&test_src, "reflink_test").is_err() {
                return false;
            }

            // Try strict reflink (not reflink_or_copy)
            let result = reflink(&test_src, &test_dst).is_ok();

            // Cleanup
            let _ = std::fs::remove_file(&test_src);
            let _ = std::fs::remove_file(&test_dst);

            if result {
                info!("Reflink support detected in temp directory");
            } else {
                debug!("Reflink not available on temp filesystem");
            }

            result
        }

        #[cfg(not(feature = "reflink"))]
        false
    }
}

impl Default for ReflinkDriver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Driver for ReflinkDriver {
    fn name(&self) -> &'static str {
        "reflink"
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "reflink")]
        {
            self.available
        }

        #[cfg(not(feature = "reflink"))]
        false
    }

    
    #[cfg(feature = "reflink")]
    async fn create_worktree(&self, opts: &DriverOpts) -> Git2DBResult<WorktreeHandle> {
        // Validate inputs
        if !opts.base_repo.exists() {
            return Err(Git2DBError::invalid_path(
                opts.base_repo.clone(),
                "Base repository does not exist",
            ));
        }

        info!(
            "Creating reflink worktree: base={}, path={}, ref={}",
            opts.base_repo.display(),
            opts.worktree_path.display(),
            opts.ref_spec
        );

        let worktree_path = opts.worktree_path.clone();

        // Step 1: Create git worktree with rollback
        match self.create_git_worktree(&opts.base_repo, &worktree_path, &opts.ref_spec).await {
            Ok(_) => {
                // Step 2: Replace working files with reflinks from base
                match self.reflink_working_files(&opts.base_repo, &worktree_path).await {
                    Ok(_) => {
                        info!(
                            "Successfully created reflink worktree at {}",
                            worktree_path.display()
                        );

                        // Return handle (no special cleanup needed - just rm like vfs)
                        Ok(WorktreeHandle::new(worktree_path, "reflink".to_string()))
                    }
                    Err(e) => {
                        // SECURITY: Clean up partially created worktree on reflink failure
                        let _ = tokio::fs::remove_dir_all(&worktree_path).await;
                        info!("Cleaned up partially created worktree due to reflink failure");
                        Err(e)
                    }
                }
            }
            Err(e) => {
                // Worktree creation failed, no cleanup needed as directory was never created
                Err(e)
            }
        }
    }

    #[cfg(not(feature = "reflink"))]
    async fn create_worktree(&self, _opts: &DriverOpts) -> Git2DBResult<WorktreeHandle> {
        Err(Git2DBError::internal(
            "reflink driver requires 'reflink' feature to be enabled",
        ))
    }

    async fn get_worktrees(&self, base_repo: &Path) -> Git2DBResult<Vec<WorktreeHandle>> {
        let worktrees_dir = base_repo.parent()
            .ok_or_else(|| Git2DBError::invalid_path(base_repo.to_path_buf(), "Invalid base repository path"))?
            .join("worktrees");

        let mut worktrees = Vec::new();

        if !worktrees_dir.exists() {
            return Ok(worktrees);
        }

        // Read worktrees directory
        for entry in std::fs::read_dir(&worktrees_dir)? {
            let entry = entry?;
            let worktree_path = entry.path();

            // Skip non-directories and git's internal directories
            if !worktree_path.is_dir() || worktree_path.file_name().map_or(false, |name| {
                name.to_string_lossy().starts_with(".git")
            }) {
                continue;
            }

            // For Reflink driver, any directory with a .git inside is a valid worktree
            if worktree_path.join(".git").exists() {
                worktrees.push(WorktreeHandle::new(worktree_path, "reflink".to_string()));
            }
        }

        Ok(worktrees)
    }

    async fn get_worktree(&self, base_repo: &Path, branch: &str) -> Git2DBResult<Option<WorktreeHandle>> {
        let worktree_path = base_repo.parent()
            .ok_or_else(|| Git2DBError::invalid_path(base_repo.to_path_buf(), "Invalid base repository path"))?
            .join("worktrees")
            .join(branch);

        if worktree_path.exists() && worktree_path.join(".git").exists() {
            Ok(Some(WorktreeHandle::new(worktree_path, "reflink".to_string())))
        } else {
            Ok(None)
        }
    }
}

#[cfg(feature = "reflink")]
impl ReflinkDriver {
    /// Create git worktree using libgit2
    async fn create_git_worktree(
        &self,
        base_repo: &Path,
        worktree_path: &Path,
        ref_spec: &str,
    ) -> Git2DBResult<()> {
        // Ensure parent directory exists
        if let Some(parent) = worktree_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                Git2DBError::internal(format!("Failed to create parent directory: {}", e))
            })?;
        }

        // Open the base repository
        let repo = git2::Repository::open(base_repo)
            .map_err(|e| Git2DBError::internal(format!("Failed to open repository: {}", e)))?;

        // Resolve ref_spec to a commit
        let object = repo.revparse_single(ref_spec).map_err(|e| {
            Git2DBError::internal(format!("Failed to resolve ref '{}': {}", ref_spec, e))
        })?;

        let commit = object.peel_to_commit().map_err(|e| {
            Git2DBError::internal(format!(
                "Ref '{}' does not point to a commit: {}",
                ref_spec, e
            ))
        })?;

        // Check if this is a branch
        let branch_ref_name = format!("refs/heads/{}", ref_spec);
        let is_branch = repo.find_reference(&branch_ref_name).is_ok();

        // Create worktree name
        let worktree_name = worktree_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                Git2DBError::invalid_path(worktree_path.to_path_buf(), "Invalid worktree path")
            })?;

        // Create worktree
        if is_branch {
            let reference = repo.find_reference(&branch_ref_name)?;
            repo.worktree(
                worktree_name,
                worktree_path,
                Some(git2::WorktreeAddOptions::new().reference(Some(&reference))),
            )
            .map_err(|e| Git2DBError::internal(format!("Failed to create worktree: {}", e)))?;

            info!(
                "Created git worktree at {} for branch '{}' (commit: {})",
                worktree_path.display(),
                ref_spec,
                commit.id()
            );
        } else {
            repo.worktree(worktree_name, worktree_path, None)
                .map_err(|e| Git2DBError::internal(format!("Failed to create worktree: {}", e)))?;

            let wt_repo = git2::Repository::open(worktree_path)?;
            wt_repo.set_head_detached(commit.id()).map_err(|e| {
                Git2DBError::internal(format!("Failed to set detached HEAD: {}", e))
            })?;

            wt_repo
                .checkout_head(Some(git2::build::CheckoutBuilder::default().force()))
                .map_err(|e| Git2DBError::internal(format!("Failed to checkout HEAD: {}", e)))?;

            info!(
                "Created git worktree at {} for ref '{}' (detached HEAD at {})",
                worktree_path.display(),
                ref_spec,
                commit.id()
            );
        }

        Ok(())
    }

    /// Replace working files with reflinks from base repository
    ///
    /// This walks the worktree and replaces each file with a reflink
    /// from the corresponding file in the base repository.
    async fn reflink_working_files(
        &self,
        base_repo: &Path,
        worktree_path: &Path,
    ) -> Git2DBResult<()> {
        use tokio::task::spawn_blocking;

        // SECURITY: Check if tokio runtime is available before spawning blocking task
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(Git2DBError::internal(
                "Cannot spawn blocking task - tokio runtime not available (possibly shutting down)",
            ));
        }

        let base = base_repo.to_path_buf();
        let worktree = worktree_path.to_path_buf();
        let min_size = self.config.min_size_bytes;

        spawn_blocking(move || Self::reflink_directory(&base, &worktree, min_size))
            .await
            .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))?
    }

    /// Recursively reflink files in directory
    fn reflink_directory(base: &Path, target: &Path, min_size: u64) -> Git2DBResult<()> {
        let mut reflinked_count = 0u64;
        let mut reflinked_bytes = 0u64;
        let mut skipped_count = 0u64;

        for entry in walkdir::WalkDir::new(target)
            .into_iter()
            .filter_entry(|e| {
                // Skip .git directory
                e.file_name() != ".git"
            })
        {
            let entry = entry.map_err(|e| {
                Git2DBError::internal(format!("Failed to walk directory: {}", e))
            })?;

            if !entry.file_type().is_file() {
                continue;
            }

            // SECURITY: Use safe-path crate to prevent path traversal
            let rel_path = entry.path().strip_prefix(target).map_err(|e| {
                Git2DBError::internal(format!("Failed to get relative path: {}", e))
            })?;

            // SECURITY: Validate path components to prevent traversal attacks
            if rel_path.components().any(|c| c.as_os_str() == "..") {
                warn!("Path traversal attempt detected: {:?}", entry.path());
                return Err(Git2DBError::invalid_path(
                    entry.path().to_path_buf(),
                    "Path traversal attempt detected",
                ));
            }

            // Construct base file path (same relative path but from base directory)
            let base_file = base.join(rel_path);
            let target_file = entry.path();

            // Check if base file exists
            if !base_file.exists() {
                skipped_count += 1;
                continue;
            }

            // Check file size
            let metadata = std::fs::metadata(&base_file).map_err(|e| {
                Git2DBError::internal(format!(
                    "Failed to get file metadata for {}: {}",
                    base_file.display(),
                    e
                ))
            })?;

            if metadata.len() < min_size {
                skipped_count += 1;
                continue;
            }

            // Remove target file
            std::fs::remove_file(target_file).map_err(|e| {
                Git2DBError::internal(format!(
                    "Failed to remove file {}: {}",
                    target_file.display(),
                    e
                ))
            })?;

            // Reflink from base (strict - no fallback to copy)
            reflink(&base_file, target_file).map_err(|e| {
                Git2DBError::internal(format!(
                    "Reflink failed for {} -> {}: {}",
                    base_file.display(),
                    target_file.display(),
                    e
                ))
            })?;

            reflinked_count += 1;
            reflinked_bytes += metadata.len();
        }

        info!(
            "Reflink complete: {} files ({} bytes) reflinked, {} files skipped",
            reflinked_count, reflinked_bytes, skipped_count
        );

        Ok(())
    }

    async fn get_worktrees(&self, base_repo: &Path) -> Git2DBResult<Vec<WorktreeHandle>> {
        let worktrees_dir = base_repo.parent()
            .ok_or_else(|| Git2DBError::invalid_path(base_repo.to_path_buf(), "Invalid base repository path"))?
            .join("worktrees");

        let mut worktrees = Vec::new();

        if !worktrees_dir.exists() {
            return Ok(worktrees);
        }

        // Read worktrees directory
        for entry in std::fs::read_dir(&worktrees_dir)? {
            let entry = entry?;
            let worktree_path = entry.path();

            // Skip non-directories and git's internal directories
            if !worktree_path.is_dir() || worktree_path.file_name().map_or(false, |name| {
                name.to_string_lossy().starts_with(".git")
            }) {
                continue;
            }

            // For Reflink driver, any directory with a .git inside is a valid worktree
            if worktree_path.join(".git").exists() {
                worktrees.push(WorktreeHandle::new(worktree_path, "reflink".to_string()));
            }
        }

        Ok(worktrees)
    }

    async fn get_worktree(&self, base_repo: &Path, branch: &str) -> Git2DBResult<Option<WorktreeHandle>> {
        let worktree_path = base_repo.parent()
            .ok_or_else(|| Git2DBError::invalid_path(base_repo.to_path_buf(), "Invalid base repository path"))?
            .join("worktrees")
            .join(branch);

        if worktree_path.exists() && worktree_path.join(".git").exists() {
            Ok(Some(WorktreeHandle::new(worktree_path, "reflink".to_string())))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_name() {
        let driver = ReflinkDriver::new();
        assert_eq!(driver.name(), "reflink");
    }

    #[test]
    fn test_default_config() {
        let config = ReflinkConfig::default();
        assert_eq!(config.min_size_bytes, 0);
    }
}
