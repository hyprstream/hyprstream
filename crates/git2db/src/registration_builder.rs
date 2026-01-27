//! Fluent builder API for registering repositories and worktrees
//!
//! Provides a git-native interface for registration with flexible configuration options

use crate::errors::{Git2DBError, Git2DBResult};
use crate::references::GitRef;
use crate::registry::{Git2DB, RemoteConfig, RepoId};
use std::collections::HashMap;
use std::path::PathBuf;

/// Builder for registering repositories and worktrees
///
/// Provides a chainable interface for configuring registration operations.
/// This is the recommended way to register repos, worktrees, and adapters.
///
/// # Examples
///
/// ```rust,no_run
/// # async fn example(registry: &mut git2db::Git2DB) -> Result<(), Box<dyn std::error::Error>> {
/// use git2db::{RepoId, GitRef};
/// use std::path::PathBuf;
///
/// // Register a worktree (adapter)
/// let adapter_id = RepoId::new();
/// registry.register(adapter_id)
///     .name("my-adapter")
///     .worktree_path(PathBuf::from("/models/working/adapter-uuid"))
///     .tracking_ref(GitRef::Branch("adapter/uuid".to_owned()))
///     .metadata("type", "adapter")
///     .metadata("base_model", "llama-uuid")
///     .exec()
///     .await?;
///
/// // Register a cloned repository
/// let model_id = RepoId::new();
/// registry.register(model_id)
///     .name("llama-3")
///     .url("https://github.com/meta/llama")
///     .worktree_path(PathBuf::from("/models/llama-uuid"))
///     .remote("origin", "https://github.com/meta/llama")
///     .exec()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct RegistrationBuilder<'a> {
    registry: &'a mut Git2DB,
    repo_id: RepoId,
    name: Option<String>,
    url: String,
    worktree_path: Option<PathBuf>,
    tracking_ref: GitRef,
    remotes: Vec<RemoteConfig>,
    metadata: HashMap<String, String>,
}

impl<'a> RegistrationBuilder<'a> {
    /// Create a new registration builder
    pub(crate) fn new(registry: &'a mut Git2DB, repo_id: RepoId) -> Self {
        Self {
            registry,
            repo_id,
            name: None,
            url: String::new(),
            worktree_path: None,
            tracking_ref: GitRef::DefaultBranch,
            remotes: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the repository name
    ///
    /// This is a human-friendly name for looking up the repository.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the remote URL
    ///
    /// For local worktrees/adapters, this can be an empty string.
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Set the worktree path
    ///
    /// This is the actual filesystem location of the repository or worktree.
    /// **REQUIRED** - must be called before exec().
    ///
    /// The path must exist before registration.
    pub fn worktree_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.worktree_path = Some(path.into());
        self
    }

    /// Set the tracking ref
    ///
    /// Specifies which branch, tag, or commit this repository tracks.
    /// Defaults to `GitRef::DefaultBranch` if not specified.
    pub fn tracking_ref(mut self, git_ref: GitRef) -> Self {
        self.tracking_ref = git_ref;
        self
    }

    /// Add a remote
    ///
    /// Can be called multiple times to add multiple remotes.
    pub fn remote(mut self, name: impl Into<String>, url: impl Into<String>) -> Self {
        self.remotes.push(RemoteConfig {
            name: name.into(),
            url: url.into(),
            fetch_refs: vec![],
        });
        self
    }

    /// Add metadata key-value pair
    ///
    /// Metadata is stored in the registry and can be used for domain-specific
    /// information like adapter type, base model ID, etc.
    ///
    /// Can be called multiple times to add multiple metadata entries.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Execute the registration
    ///
    /// Validates configuration and registers the repository/worktree in the registry.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `worktree_path` was not specified
    /// - The worktree path does not exist
    /// - The repository ID is already registered
    /// - Git operations fail (submodule creation, commit, etc.)
    pub async fn exec(self) -> Git2DBResult<()> {
        // Validate required fields
        let worktree_path = self.worktree_path.ok_or_else(|| {
            Git2DBError::configuration(
                "worktree_path is required for registration. Call .worktree_path() before .exec()",
            )
        })?;

        if !worktree_path.exists() {
            return Err(Git2DBError::invalid_repository(
                self.repo_id.to_string(),
                format!("Worktree path does not exist: {}", worktree_path.display()),
            ));
        }

        // Call the internal registration method
        self.registry
            .register_repository_internal(
                self.repo_id,
                self.name,
                self.url,
                worktree_path,
                self.tracking_ref,
                self.remotes,
                self.metadata,
            )
            .await
    }
}
