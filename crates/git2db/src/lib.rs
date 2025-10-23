//! Git2DB - Git-native database operations using libgit2
//!
//! This library provides a high-level interface for using Git repositories
//! as databases, with submodules as the primary storage mechanism for artifacts.
//!
//! Now enhanced with hyprstream-inspired patterns for better thread safety,
//! UUID-based model management, and robust registry operations.

pub mod auth;
pub mod callback_config;
pub mod clone_options;
pub mod config;
pub mod errors;
pub mod transport;
pub mod transport_registry;
pub mod utils;

// Enhanced modules with git-native patterns
pub mod branch;
pub mod clone_builder;
pub mod manager;
pub mod references;
pub mod registration_builder;
pub mod registry;
pub mod remote;
pub mod repo_accessor;
pub mod repository;
pub mod repository_handle;
pub mod stage;
pub mod transaction;
pub mod worktree;

// Storage drivers (Docker's graphdriver pattern)
pub mod storage;

// Optional gittorrent integration
#[cfg(feature = "gittorrent-transport")]
pub mod gittorrent_integration;

// Optional XET large file storage integration
#[cfg(feature = "xet-storage")]
pub mod xet_filter;

// Public XET module with high-level API
#[cfg(feature = "xet-storage")]
pub mod xet {
    //! XET large file storage integration
    //!
    //! Provides high-level XET storage operations for model files and large data.

    pub use git_xet_filter::config::XetConfig;
    pub use git_xet_filter::storage::{StorageBackend, XetStorage};
    pub use crate::xet_filter::{initialize, is_initialized, last_error, clear_last_error};
}

// Re-export main types
pub use config::{Git2DBConfig, GitSignature, WorktreeConfig};
pub use errors::{Git2DBError, Git2DBResult};

// Re-export git2 types for cleaner API boundary
// Consumers should use git2db::Oid and git2db::Repository instead of git2::Oid and git2::Repository
// This maintains a clean abstraction layer while allowing low-level access when needed
pub use git2::{Oid, Repository};

// Re-export XetConfig only when xet-storage feature is enabled
// This prevents API confusion - if you can import XetConfig, XET functionality is available
#[cfg(feature = "xet-storage")]
pub use config::XetConfig;

// Enhanced exports with git-native API (v2)
pub use branch::{Branch, BranchKind, BranchManager};
pub use clone_builder::CloneBuilder;
pub use manager::GitManager;
pub use references::{GitRef, IntoGitRef};
pub use registration_builder::RegistrationBuilder;
pub use registry::{Git2DB, RemoteConfig, RepoId, TrackedRepository};
pub use remote::{Remote, RemoteManager};
pub use repository_handle::{RepositoryHandle, RepositoryStatus};
pub use stage::{StageManager, StagedFile};
pub use transaction::{IsolationMode, TransactionHandle};
pub use transport::TransportFactory;

// Worktree exports
pub use worktree::{WorktreeHandle, WorktreeMetadata};

// Overlayfs exports (feature-gated)
#[cfg(feature = "overlayfs")]
pub use worktree::overlayfs_available;

// Storage driver exports
pub use storage::{Driver, DriverRegistry, StorageDriver};

/// Version of git2db library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize global Git2DB with default config (hyprstream pattern)
pub fn init() {
    let _ = manager::GitManager::global();
}

/// Initialize global Git2DB with custom configuration
///
/// This must be called before any operations that use the global manager.
/// Returns Ok(true) if initialization succeeded, Ok(false) if already initialized.
///
/// # Example
///
/// ```rust,no_run
/// use git2db::{init_with_config, config::Git2DBConfig};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = Git2DBConfig::default();
/// init_with_config(config)?;
/// # Ok(())
/// # }
/// ```
pub fn init_with_config(config: Git2DBConfig) -> Result<(), errors::Git2DBError> {
    manager::GitManager::init_with_config(config)
}

/// Get the global GitManager instance
pub fn manager() -> &'static GitManager {
    GitManager::global()
}

/// Create a new registry at the specified path
pub async fn create_registry<P: AsRef<std::path::Path>>(base_dir: P) -> Git2DBResult<Git2DB> {
    Git2DB::open(base_dir).await
}

/// Quick operations using global manager
pub mod ops {
    use super::*;
    use std::path::Path;

    /// Clone a repository
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Simple clone with defaults
    /// git2db::ops::clone("https://github.com/user/repo.git", "./repo", None).await?;
    ///
    /// // Clone with custom options
    /// use git2db::clone_options::CloneOptions;
    /// use git2db::callback_config::{CallbackConfigBuilder, ProgressConfig};
    /// use git2db::auth::AuthStrategy;
    ///
    /// let options = CloneOptions::builder()
    ///     .callback_config(
    ///         CallbackConfigBuilder::new()
    ///             .auth(AuthStrategy::SshAgent { username: Some("git".to_string()) })
    ///             .progress(ProgressConfig::Stdout)
    ///             .build()
    ///     )
    ///     .shallow(true)
    ///     .depth(1)
    ///     .build();
    ///
    /// git2db::ops::clone("https://github.com/user/repo.git", "./repo", Some(options)).await?;
    /// ```
    pub async fn clone(
        url: &str,
        target: &Path,
        options: Option<clone_options::CloneOptions>,
    ) -> Git2DBResult<git2::Repository> {
        manager().clone_repository(url, target, options).await
    }

    /// Open a repository
    pub fn open<P: AsRef<Path>>(path: P) -> Git2DBResult<git2::Repository> {
        manager().get_repository(path)?.open()
    }

    /// Create a worktree (async)
    pub async fn create_worktree(
        base_repo: &Path,
        worktree: &Path,
        branch: &str,
    ) -> Git2DBResult<Box<dyn crate::worktree::WorktreeHandle>> {
        manager().create_worktree(base_repo, worktree, branch).await
    }

    /// Remove a worktree
    pub fn remove_worktree(base_repo: &Path, worktree_name: &str) -> Git2DBResult<()> {
        manager().remove_worktree(base_repo, worktree_name, None)
    }
}
