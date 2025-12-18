//! Unified git2db registry implementation
//!
//! Combines artifact management with hyprstream's UUID-based model patterns

use anyhow::Result;
use chrono::Utc;
use git2::{IndexAddOption, Repository};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;

use crate::config::Git2DBConfig;
use crate::errors::{Git2DBError, Git2DBResult};
use crate::manager::GitManager;
use crate::references::GitRef;
use crate::repository_handle::RepositoryHandle;
use crate::storage::{Driver, StorageDriver};
use tracing::{debug, info, trace, warn};

/// Repository identifier using UUID
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct RepoId(pub Uuid);

impl RepoId {
    /// Create a new repository ID with a random UUID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from an existing UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Result<Self> {
        Ok(Self(Uuid::parse_str(s)?))
    }

    /// Get the inner UUID
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl std::fmt::Display for RepoId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for RepoId {
    fn default() -> Self {
        Self::new()
    }
}

/// Remote configuration for a tracked repository
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteConfig {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub fetch_refs: Vec<String>,
}

impl RemoteConfig {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            fetch_refs: vec![],
        }
    }
}

/// A tracked repository in the registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedRepository {
    pub id: RepoId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub url: String,
    pub worktree_path: PathBuf,
    pub tracking_ref: GitRef,
    #[serde(default)]
    pub remotes: Vec<RemoteConfig>,
    pub registered_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_oid: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl TrackedRepository {
    /// Get the UUID of this repository
    pub fn uuid(&self) -> Uuid {
        self.id.0
    }

    /// Get the repository name or UUID as fallback
    pub fn display_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| self.id.to_string())
    }
}

/// Registry metadata stored in registry.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    pub version: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub repositories: HashMap<Uuid, TrackedRepository>,
}

impl Default for RegistryMetadata {
    fn default() -> Self {
        Self {
            version: "2.0.0".to_string(),
            repositories: HashMap::new(),
        }
    }
}

/// Git-based model registry using submodules (unified implementation)
pub struct Git2DB {
    /// Path to the .registry Git repository
    registry_path: PathBuf,
    /// Base models directory
    base_dir: PathBuf,
    /// Registry metadata
    metadata: RegistryMetadata,
    /// Git manager for operations
    git_manager: &'static GitManager,
    /// Pre-loaded storage driver for this registry (loaded once)
    storage_driver: std::sync::Arc<dyn Driver>,
}

impl Git2DB {
    /// Get the pre-loaded storage driver for this registry
    pub fn storage_driver(&self) -> &std::sync::Arc<dyn Driver> {
        &self.storage_driver
    }

    /// Get the registry version
    ///
    /// Returns the version string from the registry metadata.
    pub fn version(&self) -> &str {
        &self.metadata.version
    }
}

impl Git2DB {
    /// Initialize or open the registry
    pub async fn open<P: AsRef<Path>>(base_dir: P) -> Git2DBResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        let registry_path = base_dir.join(".registry");
        let git_manager = GitManager::global();

        // Create or open registry repo
        if registry_path.exists() {
            // Verify we can open it
            let _repo = git_manager.get_repository(&registry_path).map_err(|e| {
                Git2DBError::repository(
                    &registry_path,
                    format!("Failed to open registry repository: {}", e),
                )
            })?;
        } else {
            // Initialize new registry
            info!("Initializing new model registry at {:?}", registry_path);

            fs::create_dir_all(&registry_path).await.map_err(|e| {
                Git2DBError::repository(
                    &registry_path,
                    format!("Failed to create directory: {}", e),
                )
            })?;

            let repo = Repository::init(&registry_path).map_err(|e| {
                Git2DBError::repository(&registry_path, format!("Failed to init repository: {}", e))
            })?;

            // Create directory structure for v2
            fs::create_dir_all(registry_path.join("repos"))
                .await
                .map_err(|e| {
                    Git2DBError::repository(
                        &registry_path,
                        format!("Failed to create repos directory: {}", e),
                    )
                })?;
            fs::create_dir_all(registry_path.join(".worktrees"))
                .await
                .map_err(|e| {
                    Git2DBError::repository(
                        &registry_path,
                        format!("Failed to create .worktrees directory: {}", e),
                    )
                })?;

            // Create initial registry.json
            let metadata = RegistryMetadata::default();
            let json = serde_json::to_string_pretty(&metadata).map_err(|e| {
                Git2DBError::internal(format!("Failed to serialize metadata: {}", e))
            })?;
            fs::write(registry_path.join("registry.json"), json)
                .await
                .map_err(|e| {
                    Git2DBError::repository(
                        &registry_path,
                        format!("Failed to write registry.json: {}", e),
                    )
                })?;

            // Initial commit using standardized signature
            let sig = git_manager
                .create_signature(None, None)
                .map_err(|e| Git2DBError::internal(format!("Failed to create signature: {}", e)))?;

            let tree_id = {
                let mut index = repo.index().map_err(|e| {
                    Git2DBError::repository(&registry_path, format!("Failed to get index: {}", e))
                })?;
                index
                    .add_all(["*"].iter(), IndexAddOption::DEFAULT, None)
                    .map_err(|e| {
                        Git2DBError::repository(
                            &registry_path,
                            format!("Failed to stage files: {}", e),
                        )
                    })?;
                index.write().map_err(|e| {
                    Git2DBError::repository(&registry_path, format!("Failed to write index: {}", e))
                })?;
                index.write_tree().map_err(|e| {
                    Git2DBError::repository(&registry_path, format!("Failed to write tree: {}", e))
                })?
            };

            let tree = repo.find_tree(tree_id).map_err(|e| {
                Git2DBError::repository(&registry_path, format!("Failed to find tree: {}", e))
            })?;

            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                "Initialize model registry",
                &tree,
                &[],
            )
            .map_err(|e| {
                Git2DBError::repository(
                    &registry_path,
                    format!("Failed to create initial commit: {}", e),
                )
            })?;
        }

        // Load metadata
        let metadata_path = registry_path.join("registry.json");
        let metadata = if metadata_path.exists() {
            let content = fs::read_to_string(&metadata_path).await.map_err(|e| {
                Git2DBError::repository(
                    &registry_path,
                    format!("Failed to read registry.json: {}", e),
                )
            })?;
            serde_json::from_str(&content).map_err(|e| {
                Git2DBError::repository(
                    &registry_path,
                    format!("Failed to parse registry.json: {}", e),
                )
            })?
        } else {
            RegistryMetadata::default()
        };

        // Load storage driver once based on fresh configuration
        let config = Git2DBConfig::load()
            .map_err(|e| Git2DBError::internal(format!("Failed to load configuration: {}", e)))?;
        // Debug logging for configuration
        info!("Worktree driver configuration loaded: '{}'", config.worktree.driver);

        let driver_config = config.worktree.driver
            .parse::<StorageDriver>()
            .map_err(|e| Git2DBError::internal(format!("Invalid storage driver config '{}': {}", config.worktree.driver, e)))?;

        info!("Parsed driver configuration: {:?}", driver_config);

        let storage_driver = git_manager
            .driver_registry()
            .get_driver(driver_config)
            .map_err(|e| Git2DBError::internal(format!("Failed to load storage driver: {}", e)))?;

        // Validate that the selected driver is available
        if !storage_driver.is_available() {
            return Err(Git2DBError::internal(format!(
                "Selected storage driver '{}' is not available on this system",
                storage_driver.name()
            )));
        }

        let mut registry = Self {
            registry_path,
            base_dir,
            metadata,
            git_manager,
            storage_driver,
        };

        // Check for and recover from uncommitted changes (e.g., from interrupted operations)
        if let Err(e) = registry.recover_uncommitted_changes() {
            warn!("Failed to recover uncommitted changes: {}", e);
        }

        // Initialize submodules if configured (equivalent to git clone --recurse-submodules)
        let config = git_manager.config();
        if config.repository.auto_init_submodules {
            if let Err(e) = registry.init_submodules().await {
                warn!("Failed to initialize submodules: {}", e);
            }
        }

        // Initialize XET filter if configured
        #[cfg(feature = "xet-storage")]
        {
            let config = git_manager.config();
            if !config.xet.endpoint.is_empty() {
                // Note: XET filter initialization will now properly fail with a clear error
                // if called from within an existing runtime context
                if let Err(e) = crate::xet_filter::initialize(config.xet.clone()).await {
                    warn!(
                        "Failed to initialize XET filter: {}. Falling back to Git LFS.",
                        e
                    );
                }
            }
        }

        Ok(registry)
    }

    /// Open the repository on-demand
    fn open_repo(&self) -> Git2DBResult<Repository> {
        let repo_cache = self
            .git_manager
            .get_repository(&self.registry_path)
            .map_err(|e| {
                Git2DBError::repository(
                    &self.registry_path,
                    format!("Failed to open registry repository: {}", e),
                )
            })?;
        repo_cache.open()
    }

    /// Add a repository to the registry
    ///
    /// Clones the repository and registers it with the given name.
    /// Names must be unique - attempting to add a repository with a duplicate name will fail.
    ///
    /// # Arguments
    /// * `name` - Unique name for the repository
    /// * `url` - Git URL to clone from
    ///
    /// # Returns
    /// The UUID (RepoId) of the newly added repository
    ///
    /// # Examples
    /// ```rust,ignore
    /// let id = registry.add_repository("myrepo", "https://github.com/user/repo.git").await?;
    /// ```
    pub async fn add_repository(&mut self, name: &str, url: &str) -> Git2DBResult<RepoId> {
        let repo_id = RepoId::new();
        self.add_repository_with_id(repo_id.clone(), name, url)
            .await?;
        Ok(repo_id)
    }

    /// Add a repository with a specific ID (used by transactions)
    ///
    /// This is an internal method used by the transaction system to ensure
    /// IDs are consistent between transaction creation and commit.
    pub(crate) async fn add_repository_with_id(
        &mut self,
        repo_id: RepoId,
        name: &str,
        url: &str,
    ) -> Git2DBResult<()> {
        // Path to the actual repository
        let repo_path = self.base_dir.join(repo_id.0.to_string());

        // Clone the repository
        self.git_manager
            .clone_repository(url, &repo_path, None)
            .await
            .map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to clone repository: {}", e))
            })?;

        // Register the repository using the builder API
        let repo_path = self.base_dir.join(repo_id.0.to_string());
        self.register(repo_id.clone())
            .name(name)
            .url(url)
            .worktree_path(repo_path)
            .exec()
            .await?;

        Ok(())
    }

    /// Register a repository that was cloned to a UUID directory
    ///
    /// **DEPRECATED:** Use `registry.register(repo_id)` builder instead.
    ///
    /// This method assumes the repository is at `base_dir/{uuid}/` which doesn't
    /// work for worktrees at custom paths. Use the builder for flexibility:
    ///
    /// ```rust,ignore
    /// registry.register(repo_id)
    ///     .name("my-repo")
    ///     .worktree_path(actual_path)
    ///     .exec().await?;
    /// ```
    #[deprecated(
        since = "2.1.0",
        note = "Use `registry.register(repo_id)` builder instead"
    )]
    pub async fn register_repository(
        &mut self,
        repo_id: &RepoId,
        name: Option<String>,
        url: String,
    ) -> Git2DBResult<()> {
        // Check if UUID is already registered
        if self.metadata.repositories.contains_key(&repo_id.0) {
            warn!(
                "Repository with UUID {} is already registered, skipping",
                repo_id.0
            );
            return Ok(());
        }

        // Path to the actual repository
        let repo_path = self.base_dir.join(repo_id.0.to_string());
        if !repo_path.exists() {
            return Err(Git2DBError::invalid_repository(
                repo_id.to_string(),
                "Repository directory does not exist",
            ));
        }

        // Add as submodule (relative path from .registry)
        // For register_repository, the repo is always at <uuid> location
        let relative_path = format!("../{}", repo_id.0);
        let submodule_path = format!("repos/{}", repo_id.0);

        // Check if submodule already exists
        let repo = self.open_repo()?;
        if repo.find_submodule(&submodule_path).is_ok() {
            warn!("Submodule {} already exists, skipping", submodule_path);
        } else {
            // Add submodule
            repo.submodule(
                &relative_path,
                Path::new(&submodule_path),
                true, // use gitlink
            )
            .map_err(|e| {
                Git2DBError::submodule(
                    repo_id.to_string(),
                    format!("Failed to add submodule: {}", e),
                )
            })?;
        }

        // Update metadata
        self.metadata.repositories.insert(
            repo_id.0,
            TrackedRepository {
                id: repo_id.clone(),
                name,
                url: url.clone(),
                worktree_path: repo_path.clone(),
                tracking_ref: GitRef::DefaultBranch,
                remotes: vec![RemoteConfig::new("origin", url)],
                registered_at: Utc::now().timestamp(),
                current_oid: None,
                metadata: HashMap::new(),
            },
        );

        // Save metadata and commit
        self.save_metadata().await?;
        self.commit_changes(&format!("Register repository: {}", repo_id.0))?;

        info!("Registered repository with UUID {}", repo_id.0);
        Ok(())
    }

    /// Internal registration method with full configuration
    ///
    /// Used by both CloneBuilder and RegistrationBuilder.
    /// External callers should use `registry.register(repo_id)` builder.
    pub(crate) async fn register_repository_internal(
        &mut self,
        repo_id: RepoId,
        name: Option<String>,
        url: String,
        worktree_path: PathBuf,
        tracking_ref: GitRef,
        remotes: Vec<RemoteConfig>,
        metadata: std::collections::HashMap<String, String>,
    ) -> Git2DBResult<()> {
        // Check if UUID is already registered
        if self.metadata.repositories.contains_key(&repo_id.0) {
            warn!(
                "Repository with UUID {} is already registered, skipping",
                repo_id.0
            );
            return Ok(());
        }

        // Verify path exists
        if !worktree_path.exists() {
            return Err(Git2DBError::invalid_repository(
                repo_id.to_string(),
                "Repository directory does not exist",
            ));
        }

        // Add as submodule (relative path from .registry)
        // Use the actual worktree path to calculate the relative path
        let worktree_name = worktree_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| Git2DBError::internal("Invalid worktree path"))?;
        let relative_path = format!("../{}", worktree_name);
        let submodule_path = format!("repos/{}", repo_id.0);

        // Check if submodule already exists
        let repo = self.open_repo()?;
        if repo.find_submodule(&submodule_path).is_ok() {
            warn!("Submodule {} already exists, skipping", submodule_path);
        } else {
            // Add submodule
            repo.submodule(
                &relative_path,
                Path::new(&submodule_path),
                true, // use gitlink
            )
            .map_err(|e| {
                Git2DBError::submodule(
                    repo_id.to_string(),
                    format!("Failed to add submodule: {}", e),
                )
            })?;
        }

        // Update metadata with full configuration
        self.metadata.repositories.insert(
            repo_id.0,
            TrackedRepository {
                id: repo_id.clone(),
                name,
                url,
                worktree_path,
                tracking_ref,
                remotes,
                registered_at: Utc::now().timestamp(),
                current_oid: None,
                metadata,
            },
        );

        // Save metadata and commit
        self.save_metadata().await?;
        self.commit_changes(&format!("Register repository: {}", repo_id.0))?;

        info!(
            "Registered repository with UUID {} (full config)",
            repo_id.0
        );
        Ok(())
    }

    /// Remove a repository from the registry with complete cleanup
    ///
    /// This performs proper git submodule cleanup including:
    /// - Removes from .gitmodules
    /// - Removes from .git/config
    /// - Removes .git/modules/{id}/ directory
    /// - Removes working directory
    /// - Removes from registry metadata
    ///
    /// Similar to `git submodule deinit --force <name> && git rm --force <name>`
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(registry: &mut git2db::Git2DB, id: &git2db::RepoId) -> Result<(), Box<dyn std::error::Error>> {
    /// registry.remove_repository(id).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn remove_repository(&mut self, id: &RepoId) -> Git2DBResult<()> {
        // Get repository info before removal
        let repo_info = self
            .get_by_id(id)
            .ok_or_else(|| {
                Git2DBError::invalid_repository(id.to_string(), "Repository not found")
            })?
            .clone();

        info!("Removing repository {}", id);

        let submodule_path = format!("repos/{}", id.0);
        let worktree_path = repo_info.worktree_path.clone();

        // Open registry repository
        let repo = self.open_repo()?;

        // 1. Remove submodule from git (handles .gitmodules, .git/config, .git/modules cleanup)
        if let Ok(mut submodule) = repo.find_submodule(&submodule_path) {
            debug!("Removing submodule {}", submodule_path);

            // Deinitialize submodule (removes from .git/config)
            match submodule.init(false) {
                Ok(_) => {}
                Err(e) => warn!(
                    "Failed to deinit submodule (may already be deinitialized): {}",
                    e
                ),
            }

            // Remove submodule (removes from .gitmodules and working directory)
            // Note: libgit2 doesn't have a direct remove_submodule, so we do manual cleanup

            // Remove .git/modules/{id} directory
            let modules_dir = self
                .registry_path
                .join(".git")
                .join("modules")
                .join(id.to_string());
            if modules_dir.exists() {
                debug!("Removing git modules directory: {:?}", modules_dir);
                fs::remove_dir_all(&modules_dir).await.map_err(|e| {
                    Git2DBError::internal(format!("Failed to remove .git/modules/{}: {}", id, e))
                })?;
            }

            // Remove from .gitmodules manually (libgit2 submodule API is limited)
            let gitmodules_path = self.registry_path.join(".gitmodules");
            if gitmodules_path.exists() {
                let content = fs::read_to_string(&gitmodules_path).await.map_err(|e| {
                    Git2DBError::internal(format!("Failed to read .gitmodules: {}", e))
                })?;

                // Filter out the entire [submodule "repos/{id}"] section
                let mut new_content = String::new();
                let mut skip_section = false;
                let section_marker = format!("[submodule \"repos/{}\"", id.0);

                for line in content.lines() {
                    // Check if this is the start of the submodule section we want to remove
                    if line.starts_with(&section_marker) {
                        skip_section = true;
                        continue;
                    }
                    // Check if we've reached a new section
                    if line.starts_with('[') && skip_section {
                        skip_section = false;
                    }
                    // Only include lines not in the skipped section
                    if !skip_section {
                        new_content.push_str(line);
                        new_content.push('\n');
                    }
                }

                // Remove trailing newline if content is now empty
                let final_content = new_content.trim_end();

                if final_content.is_empty() {
                    // Remove .gitmodules file if it's now empty
                    fs::remove_file(&gitmodules_path).await.map_err(|e| {
                        Git2DBError::internal(format!("Failed to remove empty .gitmodules: {}", e))
                    })?;
                } else {
                    fs::write(&gitmodules_path, final_content)
                        .await
                        .map_err(|e| {
                            Git2DBError::internal(format!("Failed to update .gitmodules: {}", e))
                        })?;
                }
            }

            // Remove submodule working directory from registry
            let submodule_work_path = self.registry_path.join(&submodule_path);
            if submodule_work_path.exists() {
                debug!(
                    "Removing submodule working directory: {:?}",
                    submodule_work_path
                );
                fs::remove_dir_all(&submodule_work_path)
                    .await
                    .map_err(|e| {
                        Git2DBError::internal(format!(
                            "Failed to remove submodule directory: {}",
                            e
                        ))
                    })?;
            }
        } else {
            warn!(
                "Submodule {} not found in registry repo, continuing cleanup",
                submodule_path
            );
        }

        // 2. Remove actual repository working directory
        if worktree_path.exists() {
            debug!("Removing repository working directory: {:?}", worktree_path);
            fs::remove_dir_all(&worktree_path).await.map_err(|e| {
                Git2DBError::repository(
                    &worktree_path,
                    format!("Failed to remove repository directory: {}", e),
                )
            })?;
        }

        // 3. Remove from metadata
        self.metadata.repositories.remove(&id.0);

        // 4. Save metadata and commit
        self.save_metadata().await?;
        self.commit_changes(&format!("Remove repository: {}", id.0))?;

        info!("Successfully removed repository {}", id);
        Ok(())
    }

    /// Upsert a repository by name (add if missing, return existing if present)
    ///
    /// This operation embodies git's philosophy that "something is there but maybe you just don't know it yet".
    /// If a repository with the given name exists, returns its ID. Otherwise, clones and registers it.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(registry: &mut git2db::Git2DB) -> Result<(), Box<dyn std::error::Error>> {
    /// // First call clones the repository
    /// let id1 = registry.upsert_repository("my-repo", "https://github.com/user/repo.git").await?;
    ///
    /// // Second call returns the same ID without cloning
    /// let id2 = registry.upsert_repository("my-repo", "https://github.com/user/repo.git").await?;
    /// assert_eq!(id1, id2);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn upsert_repository(&mut self, name: &str, url: &str) -> Git2DBResult<RepoId> {
        // Check if repository with this name already exists
        if let Some(existing) = self.get_by_name(name) {
            info!(
                "Repository '{}' already exists with ID {}",
                name, existing.id
            );

            // Optionally update URL if different
            if existing.url != url {
                warn!(
                    "Repository '{}' exists with different URL. Existing: {}, Requested: {}",
                    name, existing.url, url
                );
                // For now, just return existing ID. In the future, could update remotes.
            }

            return Ok(existing.id.clone());
        }

        // Repository doesn't exist, add it
        info!("Repository '{}' not found, adding from {}", name, url);
        self.add_repository(name, url).await
    }

    /// Update a repository's configuration
    ///
    /// Allows updating URL, tracking ref, or remotes for an existing repository.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Update the primary URL
    /// registry.update_repository(id, Some("https://new-host.com/repo.git".to_string())).await?;
    ///
    /// // Refresh without changing URL
    /// registry.update_repository(id, None).await?;
    /// ```
    pub async fn update_repository(
        &mut self,
        id: &RepoId,
        new_url: Option<String>,
    ) -> Git2DBResult<()> {
        // Get mutable reference to repository metadata
        let repo = self.metadata.repositories.get_mut(&id.0).ok_or_else(|| {
            Git2DBError::invalid_repository(id.to_string(), "Repository not found")
        })?;

        // Update URL if provided
        if let Some(url) = new_url {
            info!("Updating repository {} URL: {} -> {}", id, repo.url, url);
            repo.url = url.clone();

            // Update origin remote config
            if let Some(origin) = repo.remotes.iter_mut().find(|r| r.name == "origin") {
                origin.url = url;
            }
        }

        // Save and commit changes
        self.save_metadata().await?;
        self.commit_changes(&format!("Update repository: {}", id.0))?;

        Ok(())
    }

    /// Create a CloneBuilder for fluent repository cloning
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(registry: &mut git2db::Git2DB) -> Result<(), Box<dyn std::error::Error>> {
    /// let id = registry.clone("https://github.com/user/repo.git")
    ///     .name("my-repo")
    ///     .branch("main")
    ///     .remote("backup", "https://backup.com/repo.git")
    ///     .exec()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    /// Clone a repository using fluent builder API
    ///
    /// Returns a builder for configuring clone operations.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let id = registry.clone("https://github.com/user/repo.git")
    ///     .name("my-repo")
    ///     .branch("develop")
    ///     .exec()
    ///     .await?;
    /// ```
    pub fn clone<'a>(
        &'a mut self,
        url: impl Into<String>,
    ) -> crate::clone_builder::CloneBuilder<'a> {
        crate::clone_builder::CloneBuilder::new(self, url.into())
    }

    /// Register an existing repository or worktree using fluent builder API
    ///
    /// Returns a builder for configuring registration parameters.
    /// Use this to register repos/worktrees at custom paths.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Register a worktree (adapter)
    /// registry.register(adapter_id)
    ///     .name("my-adapter")
    ///     .worktree_path(worktree_path)
    ///     .tracking_ref(GitRef::Branch("adapter/uuid".to_string()))
    ///     .metadata("type", "adapter")
    ///     .exec()
    ///     .await?;
    /// ```
    pub fn register<'a>(
        &'a mut self,
        repo_id: RepoId,
    ) -> crate::registration_builder::RegistrationBuilder<'a> {
        crate::registration_builder::RegistrationBuilder::new(self, repo_id)
    }

    /// Get the base directory where repositories are stored
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// List all repositories in the registry
    pub fn list(&self) -> impl Iterator<Item = &TrackedRepository> {
        self.metadata.repositories.values()
    }

    /// Get repository by name
    pub fn get_by_name(&self, name: &str) -> Option<&TrackedRepository> {
        self.metadata
            .repositories
            .values()
            .find(|repo| repo.name.as_ref().map(|n| n == name).unwrap_or(false))
    }

    /// Get repository by ID
    pub fn get_by_id(&self, id: &RepoId) -> Option<&TrackedRepository> {
        self.metadata.repositories.get(&id.0)
    }

    /// Get repository worktree path
    pub fn get_worktree_path(&self, id: &RepoId) -> Option<PathBuf> {
        self.get_by_id(id).map(|repo| repo.worktree_path.clone())
    }

    /// Get a handle to a repository by ID
    pub fn repo(&self, id: &RepoId) -> Git2DBResult<RepositoryHandle<'_>> {
        if self.get_by_id(id).is_none() {
            return Err(Git2DBError::invalid_repository(
                id.to_string(),
                "Repository not found",
            ));
        }
        Ok(RepositoryHandle::new(self, id.clone()))
    }

    /// Get a handle to a repository by name
    pub fn repo_by_name(&self, name: &str) -> Git2DBResult<RepositoryHandle<'_>> {
        let repo = self
            .get_by_name(name)
            .ok_or_else(|| Git2DBError::invalid_repository(name, "Repository not found"))?;
        Ok(RepositoryHandle::new(self, repo.id.clone()))
    }

    /// Get registry path
    pub fn registry_path(&self) -> &Path {
        &self.registry_path
    }

  
    /// Initialize and update all submodules
    ///
    /// Equivalent to:
    /// ```bash
    /// git submodule init
    /// git submodule update --init
    /// ```
    ///
    /// This ensures all tracked repositories (stored as submodules) are cloned
    /// and checked out to the correct commit.
    async fn init_submodules(&self) -> Git2DBResult<()> {
        let registry_path = self.registry_path.clone();

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&registry_path).map_err(|e| {
                Git2DBError::repository(&registry_path, format!("Failed to open registry: {}", e))
            })?;

            // Get all submodules
            let submodules = repo.submodules().map_err(|e| {
                Git2DBError::submodule("*", format!("Failed to list submodules: {}", e))
            })?;

            if submodules.is_empty() {
                debug!("No submodules found in registry");
                return Ok(());
            }

            info!("Initializing {} submodule(s)", submodules.len());

            for mut submodule in submodules {
                let name = submodule.name().unwrap_or("unknown").to_string();

                // Check if submodule is checked out by trying to open it
                let needs_init = submodule.open().is_err();

                if needs_init {
                    debug!("Initializing submodule: {}", name);

                    // Initialize (adds to .git/config)
                    if let Err(e) = submodule.init(false) {
                        warn!("Failed to init submodule '{}': {}", name, e);
                        continue;
                    }

                    // Update (clone + checkout)
                    let mut update_opts = git2::SubmoduleUpdateOptions::new();
                    if let Err(e) = submodule.update(true, Some(&mut update_opts)) {
                        warn!("Failed to update submodule '{}': {}", name, e);
                        continue;
                    }

                    info!("Initialized submodule: {}", name);
                } else {
                    trace!("Submodule '{}' already initialized", name);
                }
            }

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))?
    }

    /// Save metadata to disk
    async fn save_metadata(&self) -> Git2DBResult<()> {
        let metadata_path = self.registry_path.join("registry.json");

        let json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| Git2DBError::internal(format!("Failed to serialize metadata: {}", e)))?;
        fs::write(metadata_path, json).await.map_err(|e| {
            Git2DBError::repository(
                &self.registry_path,
                format!("Failed to write registry.json: {}", e),
            )
        })?;

        Ok(())
    }

    /// Commit changes to registry
    ///
    /// This method stages and commits all changes. If the commit fails, staged changes
    /// remain in the index and will be recovered on the next registry open.
    fn commit_changes(&self, message: &str) -> Git2DBResult<()> {
        let repo = self.open_repo()?;

        // Create signature early to fail fast if this is misconfigured
        let sig = self.git_manager.create_signature(None, None).map_err(|e| {
            Git2DBError::internal(format!(
                "Failed to create git signature (check git config): {}",
                e
            ))
        })?;

        // Stage all changes
        let mut index = repo.index().map_err(|e| {
            Git2DBError::repository(&self.registry_path, format!("Failed to get index: {}", e))
        })?;

        index
            .add_all(["*"].iter(), IndexAddOption::DEFAULT, None)
            .map_err(|e| {
                Git2DBError::repository(
                    &self.registry_path,
                    format!("Failed to stage changes: {}", e),
                )
            })?;
        index.write().map_err(|e| {
            Git2DBError::repository(&self.registry_path, format!("Failed to write index: {}", e))
        })?;

        let tree_id = index.write_tree().map_err(|e| {
            Git2DBError::repository(&self.registry_path, format!("Failed to write tree: {}", e))
        })?;
        let tree = repo.find_tree(tree_id).map_err(|e| {
            Git2DBError::repository(&self.registry_path, format!("Failed to find tree: {}", e))
        })?;

        // Get parent commit with proper error handling
        let parent_commit = match repo.head() {
            Ok(head) => Some(head.peel_to_commit().map_err(|e| {
                Git2DBError::repository(
                    &self.registry_path,
                    format!("Failed to peel HEAD to commit: {}", e),
                )
            })?),
            Err(e) if e.code() == git2::ErrorCode::UnbornBranch => {
                // No commits yet, this is fine
                None
            }
            Err(e) => {
                return Err(Git2DBError::repository(
                    &self.registry_path,
                    format!("Failed to get HEAD: {}", e),
                ));
            }
        };

        let parent_commits: Vec<&git2::Commit> =
            parent_commit.as_ref().map(|c| vec![c]).unwrap_or_default();

        // Create the commit
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            message,
            &tree,
            &parent_commits,
        ).map_err(|e| Git2DBError::repository(&self.registry_path,
            format!("Failed to create commit '{}': {}. Staged changes remain in index and will be recovered on next open.", message, e)))?;

        debug!("Committed to registry: {}", message);
        Ok(())
    }

    /// Recover from uncommitted changes left by interrupted operations
    ///
    /// This detects if there are staged changes without a corresponding commit
    /// and either commits them or resets the index depending on the situation.
    fn recover_uncommitted_changes(&mut self) -> Git2DBResult<()> {
        let repo = self.open_repo()?;

        // Check if there are staged changes
        let statuses = repo.statuses(None).map_err(|e| {
            Git2DBError::repository(&self.registry_path, format!("Failed to get status: {}", e))
        })?;

        let has_staged_changes = statuses.iter().any(|s| {
            s.status().intersects(
                git2::Status::INDEX_NEW
                    | git2::Status::INDEX_MODIFIED
                    | git2::Status::INDEX_DELETED,
            )
        });

        if !has_staged_changes {
            return Ok(()); // Nothing to recover
        }

        info!("Detecting staged but uncommitted changes in registry, attempting recovery...");

        // Try to commit the staged changes with a recovery message
        let sig = self.git_manager.create_signature(None, None).map_err(|e| {
            Git2DBError::internal(format!("Failed to create signature for recovery: {}", e))
        })?;

        let mut index = repo.index().map_err(|e| {
            Git2DBError::repository(&self.registry_path, format!("Failed to get index: {}", e))
        })?;

        let tree_id = index.write_tree().map_err(|e| {
            Git2DBError::repository(
                &self.registry_path,
                format!("Failed to write tree during recovery: {}", e),
            )
        })?;
        let tree = repo.find_tree(tree_id).map_err(|e| {
            Git2DBError::repository(
                &self.registry_path,
                format!("Failed to find tree during recovery: {}", e),
            )
        })?;

        let parent_commit = match repo.head() {
            Ok(head) => Some(head.peel_to_commit().map_err(|e| {
                Git2DBError::repository(
                    &self.registry_path,
                    format!("Failed to peel HEAD during recovery: {}", e),
                )
            })?),
            Err(_) => None,
        };

        let parent_commits: Vec<&git2::Commit> =
            parent_commit.as_ref().map(|c| vec![c]).unwrap_or_default();

        // Commit the recovered changes
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "[git2db recovery] Commit staged changes from interrupted operation",
            &tree,
            &parent_commits,
        )
        .map_err(|e| {
            Git2DBError::repository(
                &self.registry_path,
                format!("Failed to commit during recovery: {}", e),
            )
        })?;

        info!("Successfully recovered uncommitted changes in registry");

        // Reload metadata to reflect recovered state
        let metadata_path = self.registry_path.join("registry.json");
        if metadata_path.exists() {
            let content = std::fs::read_to_string(&metadata_path).map_err(|e| {
                Git2DBError::repository(
                    &self.registry_path,
                    format!("Failed to read registry.json during recovery: {}", e),
                )
            })?;
            self.metadata = serde_json::from_str(&content).map_err(|e| {
                Git2DBError::repository(
                    &self.registry_path,
                    format!("Failed to parse registry.json during recovery: {}", e),
                )
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_recovery_from_uncommitted_changes() {
        // Create a temporary directory for testing
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().to_path_buf();

        // Create initial registry
        let mut registry = Git2DB::open(&base_dir).await.unwrap();

        // Simulate interrupted operation by manually staging changes without committing
        {
            let repo = registry.open_repo().unwrap();
            let test_file = registry.registry_path().join("test_uncommitted.txt");
            std::fs::write(&test_file, "uncommitted data").unwrap();

            let mut index = repo.index().unwrap();
            index.add_path(Path::new("test_uncommitted.txt")).unwrap();
            index.write().unwrap();
        }

        // Verify there are staged changes
        {
            let repo = registry.open_repo().unwrap();
            let statuses = repo.statuses(None).unwrap();
            let has_staged = statuses
                .iter()
                .any(|s| s.status().intersects(git2::Status::INDEX_NEW));
            assert!(has_staged, "Should have staged changes before recovery");
        }

        // Call recovery mechanism
        registry.recover_uncommitted_changes().unwrap();

        // Verify staged changes were committed
        {
            let repo = registry.open_repo().unwrap();
            let statuses = repo.statuses(None).unwrap();
            let has_staged = statuses.iter().any(|s| {
                s.status()
                    .intersects(git2::Status::INDEX_NEW | git2::Status::INDEX_MODIFIED)
            });
            assert!(!has_staged, "Should have no staged changes after recovery");

            // Verify commit was created
            let head = repo.head().unwrap();
            let commit = head.peel_to_commit().unwrap();
            assert!(
                commit.message().unwrap().contains("recovery"),
                "Recovery commit should exist"
            );
        }
    }

    #[tokio::test]
    async fn test_recovery_does_nothing_when_no_staged_changes() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().to_path_buf();

        let mut registry = Git2DB::open(&base_dir).await.unwrap();

        // Get initial commit count
        let initial_commit_count = {
            let repo = registry.open_repo().unwrap();
            repo.revwalk().unwrap().count()
        };

        // Call recovery when there are no staged changes
        registry.recover_uncommitted_changes().unwrap();

        // Verify no new commit was created
        let final_commit_count = {
            let repo = registry.open_repo().unwrap();
            repo.revwalk().unwrap().count()
        };

        assert_eq!(
            initial_commit_count, final_commit_count,
            "Recovery should not create commit when no staged changes"
        );
    }

    #[tokio::test]
    async fn test_automatic_recovery_on_open() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().to_path_buf();

        // Create registry and simulate interrupted operation
        {
            let registry = Git2DB::open(&base_dir).await.unwrap();
            let repo = registry.open_repo().unwrap();
            let test_file = registry.registry_path().join("auto_recovery_test.txt");
            std::fs::write(&test_file, "test data").unwrap();

            let mut index = repo.index().unwrap();
            index.add_path(Path::new("auto_recovery_test.txt")).unwrap();
            index.write().unwrap();
        }

        // Reopen registry - should automatically recover
        let registry = Git2DB::open(&base_dir).await.unwrap();

        // Verify recovery happened
        let repo = registry.open_repo().unwrap();
        let statuses = repo.statuses(None).unwrap();
        let has_staged = statuses.iter().any(|s| {
            s.status()
                .intersects(git2::Status::INDEX_NEW | git2::Status::INDEX_MODIFIED)
        });
        assert!(
            !has_staged,
            "Automatic recovery should have committed staged changes"
        );
    }
}
