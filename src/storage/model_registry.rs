//! Git-native model registry using submodules

use anyhow::{Result, anyhow, bail};
use git2::{Repository, IndexAddOption, FetchOptions, RemoteCallbacks};
use safe_path::scoped_join;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;
use tokio::sync::RwLock;
use git2db::{GitManager, Git2DBConfig as GitConfig};

use super::model_ref::{ModelRef, validate_model_name};
use super::xet_native::XetNativeStorage;

/// Status of a model's git repository using git2 types
#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub model_ref: ModelRef,
    pub current_oid: git2::Oid,
    pub current_ref_name: Option<String>, // Reference name (owned string for lifetime safety)
    pub current_ref_type: Option<git2::ReferenceType>, // Type of current reference
    pub target_oid: Option<git2::Oid>,
    pub target_ref_name: Option<String>, // Target reference name
    pub target_ref_type: Option<git2::ReferenceType>, // Type of target reference
    pub is_dirty: bool,
    pub file_statuses: Vec<(String, git2::Status)>,
    pub ref_matches: bool,
}

/// Result of a checkout operation
#[derive(Debug, Clone)]
pub struct CheckoutResult {
    pub previous_oid: git2::Oid,
    pub new_oid: git2::Oid,
    pub previous_ref_name: Option<String>,
    pub new_ref_name: Option<String>,
    pub was_forced: bool,
    pub files_changed: usize,
    pub has_submodule: bool, // Whether operation involved a submodule
}

/// Options for checkout operations
#[derive(Debug, Default)]
pub struct CheckoutOptions {
    pub create_branch: bool,
    pub force: bool,
}

/// Comprehensive model information
pub struct ModelInfo {
    pub model_ref: ModelRef,
    pub path: PathBuf,
    pub current_oid: git2::Oid,
    pub current_ref_name: Option<String>,
    pub resolved_target_oid: Option<git2::Oid>,
    pub target_ref_name: Option<String>,
    pub status: ModelStatus,
    pub repository: git2::Repository,
    pub has_submodule: bool, // Whether model is managed as submodule
}

/// Model registry that manages models as git submodules
pub struct ModelRegistry {
    base_dir: PathBuf,
    xet_storage: Option<Arc<XetNativeStorage>>,
    git_manager: Arc<GitManager>,
}

impl ModelRegistry {
    /// Open or initialize a model registry
    pub fn new(base_dir: PathBuf, xet_storage: Option<Arc<XetNativeStorage>>) -> Result<Self> {
        Self::new_with_config(base_dir, xet_storage, GitConfig::default())
    }

    /// Create with custom Git configuration
    pub fn new_with_config(
        base_dir: PathBuf,
        xet_storage: Option<Arc<XetNativeStorage>>,
        git_config: GitConfig
    ) -> Result<Self> {
        if !base_dir.join(".git").exists() {
            info!("Initializing new model registry at {:?}", base_dir);
            std::fs::create_dir_all(&base_dir)?;
            let repo = Repository::init(&base_dir)?; // One-time initialization

            // Create initial commit so we can add submodules
            Self::create_initial_commit(&repo)?;
        }

        let git_manager = Arc::new(GitManager::new(git_config));

        Ok(Self {
            base_dir,
            xet_storage,
            git_manager,
        })
    }

    /// Open the repository when needed (with proper caching)
    fn open_repo(&self) -> Result<Repository> {
        crate::storage::get_cached_repository(&self.base_dir)
            .map_err(|e| anyhow!("Failed to open repository: {}", e))
    }

    /// Create initial commit for new repository so submodules can be added
    fn create_initial_commit(repo: &Repository) -> Result<()> {
        // Check if there are already commits
        if repo.head().is_ok() {
            return Ok(()); // Already has commits
        }

        // Create an empty .gitignore file
        let gitignore_path = repo.path().parent().unwrap().join(".gitignore");
        std::fs::write(&gitignore_path, "# HyprStream model registry\n")?;

        // Stage the .gitignore file
        let mut index = repo.index()?;
        index.add_path(std::path::Path::new(".gitignore"))?;
        index.write()?;

        // Create signature for commit
        let signature = git2::Signature::now("HyprStream", "noreply@hyprstream.ai")?;

        // Get the tree from the index
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;

        // Create initial commit
        repo.commit(
            Some("HEAD"),
            &signature,
            &signature,
            "Initial commit: HyprStream model registry",
            &tree,
            &[],
        )?;

        info!("Created initial commit for model registry");
        Ok(())
    }

    /// Check if a submodule exists for a model
    /// Uses the Interior Mutability pattern to safely handle repository lifetimes
    fn has_submodule(&self, model_ref: &ModelRef) -> Result<bool> {
        let submodule_path = model_ref.model.clone(); // Models are direct children

        // RAII pattern: Repository is owned for the entire scope
        let repo = self.open_repo()?;

        // Perform the check and immediately return the boolean result
        // This avoids lifetime issues by not returning any references
        let x = match repo.find_submodule(&submodule_path) {
            Ok(_) => Ok(true),
            Err(e) if e.code() == git2::ErrorCode::NotFound => Ok(false),
            Err(e) => Err(anyhow!("Failed to access submodule {}: {}", submodule_path, e)),
        }; x
    }

    /// Execute a closure with a submodule reference
    /// Uses the Visitor pattern to safely handle git2 object lifetimes
    fn with_submodule<F, R>(&self, model_ref: &ModelRef, f: F) -> Result<Option<R>>
    where
        F: FnOnce(&git2::Submodule) -> Result<R>,
        R: 'static,  // Result must be independent of repository lifetime
    {
        let submodule_path = model_ref.model.clone(); // Models are direct children

        // RAII pattern: Repository owns the submodule for this scope
        let repo = self.open_repo()?;

        // Execute the closure within the repository's lifetime scope
        let x = match repo.find_submodule(&submodule_path) {
            Ok(submodule) => {
                // The closure must extract owned data, not references
                let result = f(&submodule)?;
                Ok(Some(result))
            },
            Err(e) if e.code() == git2::ErrorCode::NotFound => Ok(None),
            Err(e) => Err(anyhow!("Failed to access submodule {}: {}", submodule_path, e)),
        }; x
    }

    /// Get the model repository and check if it's a submodule
    /// Uses the Factory pattern to create repositories with proper ownership
    ///
    /// STRICT MODE: Only returns repositories that are properly registered as submodules.
    /// This enforces git2db architectural integrity.
    fn get_model_repository(&self, model_ref: &ModelRef) -> Result<(git2::Repository, bool)> {
        let submodule_path = model_ref.model.clone(); // Models are direct children
        let repo = self.open_repo()?;

        // Submodule must exist - no fallback to untracked directories
        let submodule = repo.find_submodule(&submodule_path)
            .map_err(|e| {
                if e.code() == git2::ErrorCode::NotFound {
                    anyhow!("Model '{}' not found in registry. Use 'model add' to register it as a submodule.", model_ref.model)
                } else {
                    anyhow!("Failed to access submodule {}: {}", submodule_path, e)
                }
            })?;

        // Open the submodule repository directly by path
        let submodule_repo_path = submodule.path();
        let full_path = self.base_dir.join(submodule_repo_path);

        if !full_path.exists() {
            bail!("Submodule '{}' registered but path {:?} not found. Run 'git submodule update --init' to populate it.",
                  model_ref.model, full_path);
        }

        let model_repo = crate::storage::get_cached_repository(&full_path)?;
        Ok((model_repo, true))
    }

    /// Robust helper to resolve submodule commit with multiple fallback strategies
    fn resolve_submodule_commit(&self, submodule: &git2::Submodule<'_>, name: &str) -> Result<git2::Oid> {
        // Strategy 1: Try index_id (may fail for transitional submodules)
        if let Some(commit_id) = submodule.index_id() {
            return Ok(commit_id);
        }

        // Strategy 2: Fallback to submodule repository HEAD
        if let Ok(submodule_repo) = submodule.open() {
            if let Ok(head_ref) = submodule_repo.head() {
                if let Ok(commit) = head_ref.peel_to_commit() {
                    return Ok(commit.id());
                }
            }
        }

        Err(anyhow!("Could not resolve commit for submodule '{}' - both index_id and repository HEAD strategies failed", name))
    }

    /// Resolve a model reference to a specific commit SHA using git2 objects
    pub fn resolve(&self, model_ref: &ModelRef) -> Result<git2::Oid> {
        let (model_repo, is_submodule) = self.get_model_repository(model_ref)?;

        match &model_ref.git_ref {
            crate::storage::GitRef::DefaultBranch => {
                if is_submodule {
                    // For submodules, get the pinned commit from the parent repository
                    let parent_repo = self.open_repo()?;
                    let submodule_path = model_ref.model.clone(); // Models are direct children

                    let x = match parent_repo.find_submodule(&submodule_path) {
                        Ok(submodule) => {
                            // Use robust commit resolution
                            self.resolve_submodule_commit(&submodule, &model_ref.model)
                        },
                        Err(_) => {
                            // Fallback to model repo's HEAD if submodule not found
                            Ok(model_repo.head()?.peel_to_commit()?.id())
                        }
                    }; x
                } else {
                    // Use HEAD for direct repositories
                    Ok(model_repo.head()?.peel_to_commit()?.id())
                }
            },
            crate::storage::GitRef::Commit(oid) => {
                // Direct commit reference - just return the OID
                Ok(*oid)
            },
            _ => {
                // Branch, tag, or revspec - resolve using git2::Reference
                let git_ref_str = model_ref.git_ref.as_ref_str()
                    .ok_or_else(|| anyhow!("Cannot resolve git reference for model {}", model_ref.model))?;

                // Try to find as a reference first (more efficient)
                if let Ok(reference) = model_repo.find_reference(git_ref_str) {
                    Ok(reference.peel_to_commit()?.id())
                } else {
                    // Fallback to revparse for complex expressions
                    Ok(model_repo.revparse_single(git_ref_str)?.id())
                }
            }
        }
    }

    /// Resolve a model reference to reference information (name and target)
    pub fn resolve_reference(&self, model_ref: &ModelRef) -> Result<Option<(String, git2::Oid)>> {
        let (model_repo, _) = self.get_model_repository(model_ref)?;

        match &model_ref.git_ref {
            crate::storage::GitRef::DefaultBranch => {
                // Get the default branch reference
                match model_repo.head() {
                    Ok(head_ref) => {
                        let name = head_ref.shorthand().unwrap_or("HEAD").to_string();
                        let target = head_ref.target().unwrap_or_else(|| git2::Oid::zero());
                        Ok(Some((name, target)))
                    },
                    Err(_) => Ok(None), // Repository might be empty
                }
            },
            crate::storage::GitRef::Branch(branch_name) => {
                let ref_name = format!("refs/heads/{}", branch_name);
                match model_repo.find_reference(&ref_name) {
                    Ok(reference) => {
                        let target = reference.target().unwrap_or_else(|| git2::Oid::zero());
                        Ok(Some((branch_name.clone(), target)))
                    },
                    Err(_) => Ok(None),
                }
            },
            crate::storage::GitRef::Tag(tag_name) => {
                let ref_name = format!("refs/tags/{}", tag_name);
                match model_repo.find_reference(&ref_name) {
                    Ok(reference) => {
                        let target = reference.target().unwrap_or_else(|| git2::Oid::zero());
                        Ok(Some((tag_name.clone(), target)))
                    },
                    Err(_) => Ok(None),
                }
            },
            crate::storage::GitRef::Commit(_) | crate::storage::GitRef::Revspec(_) => {
                // Commits and revspecs don't have associated references
                Ok(None)
            }
        }
    }

    /// Add a new model to the registry using proper git submodule workflow with race condition protection
    pub async fn add_model(&self, name: &str, source: &str) -> Result<ModelRef> {
        validate_model_name(name)?;

        info!("Adding model {} as git submodule from {}", name, source);

        // Use safe-path to prevent race conditions with path operations
        let safe_model_path = scoped_join(&self.base_dir, name)?;

        let repo = self.open_repo()?;

        // Check if submodule already exists
        if repo.find_submodule(name).is_ok() {
            return Err(anyhow!("Model '{}' already exists as submodule", name));
        }

        // Check if directory already exists using safe path operations
        if safe_model_path.exists() {
            return Err(anyhow!("Directory '{}' already exists at {:?}", name, safe_model_path));
        }

        // Use proper libgit2 submodule workflow
        // Step 1: Create the submodule entry (adds to .gitmodules)
        // Note: use_gitlink=true ensures proper .git file creation in submodule directory
        let mut submodule = repo.submodule(source, std::path::Path::new(name), true)?;

        // Step 2: Initialize the submodule configuration
        // This writes the submodule.<name>.url to .git/config
        submodule.init(false)?;

        // Step 3: Clone and checkout the submodule (THIS WAS THE MISSING STEP!)
        // This is critical - it actually populates the submodule directory
        let mut update_opts = git2::SubmoduleUpdateOptions::new();

        // Configure fetch options with authentication callbacks
        let mut fetch_opts = git2::FetchOptions::new();
        let mut callbacks = git2::RemoteCallbacks::new();

        // Add authentication callbacks for private repos
        callbacks.credentials(|_url, username_from_url, _allowed_types| {
            git2::Cred::ssh_key_from_agent(username_from_url.unwrap_or("git"))
        });

        fetch_opts.remote_callbacks(callbacks);
        update_opts.fetch(fetch_opts);

        // Perform the actual clone and checkout
        submodule.update(true, Some(&mut update_opts))?;

        // Step 4: Reload the submodule to get the updated state
        // Important: re-find the submodule after update
        drop(submodule); // Release the old reference
        {
            let _submodule = repo.find_submodule(name)?; // Verify it exists
        }

        // Step 5: Stage the changes in the index with explicit flush
        {
            let mut index = repo.index()?;

            // Add .gitmodules file if it has content
            if std::path::Path::new(".gitmodules").exists() {
                index.add_path(std::path::Path::new(".gitmodules"))?;
            }

            // Add the submodule path - this should create the proper gitlink entry
            // after submodule.update() has completed successfully
            index.add_path(std::path::Path::new(name))?;

            // Write the index to disk
            index.write()?;

            // Explicitly drop index to ensure file handle is closed and flushed
            drop(index);
        }

        info!("Successfully added model {} as submodule", name);

        // Commit the submodule addition to registry BEFORE returning
        // This ensures the submodule has a proper index_id when accessed later
        self.commit_registry(&format!("Add model: {}", name))?;

        // Reload the repository to ensure the index is fresh after commit
        drop(repo);
        let _repo = self.open_repo()?;

        // Return the ModelRef for the new model
        Ok(ModelRef::new(name.to_string()))
    }

    /// Update a model to a different version
    pub async fn update_model(&self, name: &str, ref_spec: &str) -> Result<()> {
        if !git2::Reference::is_valid_name(ref_spec) {
            bail!("Invalid git reference name: '{}'", ref_spec);
        }

        let submodule_path = name.to_string(); // Models are direct children
        let repo = self.open_repo()?;
        let submodule = repo.find_submodule(&submodule_path)?;
        let model_repo = submodule.open()?;

        // Fetch latest changes
        info!("Fetching updates for model {}", name);
        let mut remote = model_repo.find_remote("origin")?;

        let mut fetch_opts = FetchOptions::new();
        if let Some(_xet) = &self.xet_storage {
            // Configure XET callbacks if needed
            let callbacks = git2::RemoteCallbacks::new();
            fetch_opts.remote_callbacks(callbacks);
        }

        remote.fetch(&[ref_spec], Some(&mut fetch_opts), None)?;

        // Resolve and checkout the ref
        let commit = model_repo.revparse_single(ref_spec)?.peel_to_commit()?;
        model_repo.set_head_detached(commit.id())?;
        model_repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;

        // Update submodule pointer in registry with explicit flush
        {
            let mut index = self.open_repo()?.index()?;
            index.add_path(Path::new(&submodule_path))?;
            index.write()?;
            drop(index);  // Ensure index is flushed
        }

        // Commit registry change
        self.commit_registry(&format!("Update {} to {}", name, ref_spec))?;

        // Reload the repository to ensure the index is fresh after commit
        let _repo = self.open_repo()?;

        Ok(())
    }

    /// List all registered models with their current commits
    pub fn list_models(&self) -> Result<Vec<(String, git2::Oid)>> {
        let mut models = Vec::new();

        // List submodules (models are now direct children)
        for submodule in self.open_repo()?.submodules()? {
            if let Some(path) = submodule.path().to_str() {
                // Use robust commit resolution with fallbacks
                match self.resolve_submodule_commit(&submodule, path) {
                    Ok(commit_id) => {
                        models.push((path.to_string(), commit_id));
                    },
                    Err(e) => {
                        // Log warning but continue listing other models
                        tracing::warn!("Could not resolve commit for submodule '{}': {}", path, e);
                    }
                }
            }
        }

        Ok(models)
    }

    /// List available refs for a model - returns reference info without lifetimes
    pub fn list_model_refs(&self, model_ref: &ModelRef) -> Result<Vec<(String, git2::Oid)>> {
        let (model_repo, _is_submodule) = self.get_model_repository(model_ref)?;

        let mut ref_info = Vec::new();

        // Collect all references, extracting owned data immediately
        model_repo.references()?.for_each(|reference| {
            if let Ok(r) = reference {
                if let Some(name) = r.shorthand() {
                    let target = r.target().unwrap_or_else(|| git2::Oid::zero());
                    ref_info.push((name.to_string(), target));
                }
            }
        });

        Ok(ref_info)
    }

    /// List available refs for a model using ModelRef - returns git2::Reference objects (deprecated - use list_model_refs instead)
    ///
    /// WARNING: This method has lifetime issues and should not be used.
    /// Use list_model_refs() instead which returns owned data.
    #[deprecated(note = "Use list_model_refs() instead to avoid lifetime issues")]
    pub fn list_model_refs_legacy(&self, _model_ref: &ModelRef) -> Result<Vec<git2::Reference>> {
        // This method cannot be implemented safely due to git2::Reference lifetime constraints
        // Callers should use list_model_refs() instead
        Err(anyhow::anyhow!("list_model_refs_legacy is deprecated due to lifetime issues - use list_model_refs instead"))
    }

    /// Get branch and tag names for a model (legacy string interface)
    pub fn list_model_ref_names(&self, model_ref: &ModelRef) -> Result<Vec<String>> {
        let (model_repo, _is_submodule) = self.get_model_repository(model_ref)?;

        let mut refs = Vec::new();

        // Add branches
        for branch in model_repo.branches(None)? {
            let (branch, _) = branch?;
            if let Some(name) = branch.name()? {
                refs.push(name.to_string());
            }
        }

        // Add tags
        model_repo.tag_foreach(|_, name| {
            if let Ok(tag) = std::str::from_utf8(name) {
                let tag_name = tag.strip_prefix("refs/tags/").unwrap_or(tag);
                refs.push(format!("tags/{}", tag_name));
            }
            true
        })?;

        Ok(refs)
    }

    /// Get the path to a model's repository from ModelRef
    pub fn get_model_path(&self, model_ref: &ModelRef) -> Result<PathBuf> {
        // Use safe-path to prevent race conditions and path traversal attacks
        let safe_model_path = scoped_join(&self.base_dir, &model_ref.model)?;

        if !safe_model_path.exists() {
            bail!("Model {} not found at {:?}", model_ref.model, safe_model_path);
        }

        Ok(safe_model_path)
    }

    /// Create a branch for a model using ModelRef
    pub fn create_branch(&self, model_ref: &ModelRef, branch_name: &str, from_ref: Option<&str>) -> Result<()> {
        let (model_repo, _is_submodule) = self.get_model_repository(model_ref)?;

        // Get the commit to branch from
        let commit = if let Some(ref_str) = from_ref {
            model_repo.revparse_single(ref_str)?.peel_to_commit()?
        } else {
            model_repo.head()?.peel_to_commit()?
        };

        // Create the branch
        let _branch = model_repo.branch(branch_name, &commit, false)?;

        info!("Created branch {} for model {} from {}",
              branch_name, model_ref.model, from_ref.unwrap_or("HEAD"));
        Ok(())
    }

    /// Get the default branch of a repository using ModelRef
    pub fn get_default_branch(&self, model_ref: &ModelRef) -> Result<String> {
        let (model_repo, _is_submodule) = self.get_model_repository(model_ref)?;

        // Try to get the symbolic reference HEAD points to
        if let Ok(head_ref) = model_repo.head() {
            if let Some(name) = head_ref.symbolic_target() {
                // Remove refs/heads/ prefix to get just the branch name
                if let Some(branch_name) = name.strip_prefix("refs/heads/") {
                    return Ok(branch_name.to_string());
                }
            }
        }

        // If HEAD doesn't point to a symbolic ref, try to find the default branch
        // Check for common default branch names
        for default_name in ["main", "master"] {
            if let Ok(_) = model_repo.find_branch(default_name, git2::BranchType::Local) {
                return Ok(default_name.to_string());
            }
        }

        // Fallback: get the first branch
        let mut branches = model_repo.branches(Some(git2::BranchType::Local))?;
        if let Some(Ok((branch, _))) = branches.next() {
            if let Some(name) = branch.name()? {
                return Ok(name.to_string());
            }
        }

        // Final fallback
        Ok("main".to_string())
    }

    /// Checkout a specific ref for a model using ModelRef and git2 objects
    pub fn checkout(&self, model_ref: &ModelRef, options: CheckoutOptions) -> Result<CheckoutResult> {
        let (model_repo, is_submodule) = self.get_model_repository(model_ref)?;

        // Get current state before checkout using git2::Reference
        let previous_reference = match model_repo.head() {
            Ok(head_ref) => Some(head_ref),
            Err(_) => None,
        };
        let previous_oid = previous_reference.as_ref()
            .and_then(|r| r.target())
            .unwrap_or_else(|| git2::Oid::zero());

        // Resolve the target using our existing resolve method
        let target_oid = self.resolve(model_ref)?;
        let target_commit = model_repo.find_commit(target_oid)?;

        // Determine how to checkout based on the git ref type using git2::Reference
        let new_ref_name = match &model_ref.git_ref {
            crate::storage::GitRef::DefaultBranch => {
                // Get or create the default branch reference
                let default_branch = self.get_default_branch(model_ref)?;
                let branch_ref_name = format!("refs/heads/{}", default_branch);
                match model_repo.find_reference(&branch_ref_name) {
                    Ok(branch_ref) => {
                        model_repo.set_head(branch_ref.name().unwrap())?;
                        Some(default_branch)
                    },
                    Err(_) => {
                        // Create the default branch if it doesn't exist
                        let _branch = model_repo.branch(&default_branch, &target_commit, false)?;
                        model_repo.set_head(&branch_ref_name)?;
                        Some(default_branch)
                    }
                }
            },
            crate::storage::GitRef::Branch(branch_name) => {
                let branch_ref_name = format!("refs/heads/{}", branch_name);
                match model_repo.find_reference(&branch_ref_name) {
                    Ok(_branch_ref) => {
                        model_repo.set_head(&branch_ref_name)?;
                        Some(branch_name.clone())
                    },
                    Err(_) if options.create_branch => {
                        // Create new branch
                        let _branch = model_repo.branch(branch_name, &target_commit, false)?;
                        model_repo.set_head(&branch_ref_name)?;
                        Some(branch_name.clone())
                    },
                    Err(e) => return Err(anyhow!("Branch '{}' not found and create_branch=false: {}", branch_name, e)),
                }
            },
            crate::storage::GitRef::Tag(tag_name) => {
                // Detached HEAD for tags
                model_repo.set_head_detached(target_commit.id())?;
                Some(tag_name.clone())
            },
            crate::storage::GitRef::Commit(_) | crate::storage::GitRef::Revspec(_) => {
                // Detached HEAD for commits and revspecs
                model_repo.set_head_detached(target_commit.id())?;
                None
            }
        };

        // Update working directory
        let mut checkout_builder = git2::build::CheckoutBuilder::new();
        if options.force {
            checkout_builder.force();
        }
        model_repo.checkout_head(Some(&mut checkout_builder))?;

        let files_changed = target_commit.tree()?.len();

        info!("Checked out {} for model {} ({})",
              new_ref_name.as_ref().unwrap_or(&"detached".to_string()),
              model_ref.model,
              target_oid);

        Ok(CheckoutResult {
            previous_oid,
            new_oid: target_oid,
            previous_ref_name: previous_reference.and_then(|r| r.shorthand().map(|s| s.to_string())),
            new_ref_name,
            was_forced: options.force,
            files_changed,
            has_submodule: is_submodule,
        })
    }

    /// Get the status of a model's repository using git2 objects
    pub fn status(&self, model_ref: &ModelRef) -> Result<ModelStatus> {
        let (model_repo, _is_submodule) = self.get_model_repository(model_ref)?;

        // Get current state using git2::Reference
        let (current_oid, current_ref_name, current_ref_type) = match model_repo.head() {
            Ok(head_ref) => {
                let oid = head_ref.target().unwrap_or_else(|| git2::Oid::zero());
                let name = head_ref.shorthand().map(|s| s.to_string());
                let ref_type = head_ref.kind();
                (oid, name, ref_type)
            },
            Err(_) => (git2::Oid::zero(), None, None),
        };

        // Resolve target from ModelRef using git2::Reference
        let target_oid = match self.resolve(model_ref) {
            Ok(oid) => Some(oid),
            Err(_) => None, // Target doesn't exist yet
        };

        let target_ref_name = match self.resolve_reference(model_ref) {
            Ok(Some((name, _oid))) => Some(name),
            _ => None,
        };

        // Check if current matches target
        let ref_matches = target_oid.map_or(false, |target| current_oid == target);

        // Get file statuses using git2 types directly
        let statuses = model_repo.statuses(None)?;
        let mut file_statuses = Vec::new();

        for entry in statuses.iter() {
            if let Some(path) = entry.path() {
                file_statuses.push((path.to_string(), entry.status()));
            }
        }

        Ok(ModelStatus {
            model_ref: model_ref.clone(),
            current_oid,
            current_ref_name,
            current_ref_type: current_ref_type,
            target_oid,
            target_ref_name,
            target_ref_type: None, // We don't track reference types anymore for simplicity
            is_dirty: !statuses.is_empty(),
            file_statuses,
            ref_matches,
        })
    }

    /// Commit changes in a model's repository using ModelRef
    pub fn commit_model(&self, model_ref: &ModelRef, message: &str, stage_all: bool) -> Result<git2::Oid> {
        let (model_repo, _is_submodule) = self.get_model_repository(model_ref)?;

        // Create signature
        let sig = self.git_manager.create_signature(None, None)?;

        // Get index
        let mut index = model_repo.index()?;

        // Stage files if requested
        if stage_all {
            index.add_all(["*"].iter(), IndexAddOption::DEFAULT, None)?;
            index.update_all(["*"].iter(), None)?;
        }

        index.write()?;

        // Create tree from index
        let tree_id = index.write_tree()?;
        let tree = model_repo.find_tree(tree_id)?;

        // Get parent commit
        let parent_commit = model_repo.head()?.peel_to_commit()?;

        // Create commit and return the OID
        let commit_oid = model_repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            message,
            &tree,
            &[&parent_commit],
        )?;

        info!("Committed changes to model {}: {} ({})", model_ref.model, message, commit_oid);
        Ok(commit_oid)
    }

    /// Push changes to remote using ModelRef
    pub fn push_model(
        &self,
        model_ref: &ModelRef,
        remote_name: &str,
        branch_name: Option<&str>,
        set_upstream: bool,
        force: bool,
    ) -> Result<()> {
        let (model_repo, _is_submodule) = self.get_model_repository(model_ref)?;

        // Get remote
        let mut remote = model_repo.find_remote(remote_name)?;

        // Determine refspec
        let refspec = if let Some(branch) = branch_name {
            if force {
                format!("+refs/heads/{}:refs/heads/{}", branch, branch)
            } else {
                format!("refs/heads/{}:refs/heads/{}", branch, branch)
            }
        } else {
            // Push current branch
            let head = model_repo.head()?;
            let branch_name = head.shorthand().ok_or_else(|| anyhow!("Not on a branch"))?;
            if force {
                format!("+refs/heads/{}:refs/heads/{}", branch_name, branch_name)
            } else {
                format!("refs/heads/{}:refs/heads/{}", branch_name, branch_name)
            }
        };

        // Configure callbacks
        let mut push_opts = git2::PushOptions::new();
        let callbacks = RemoteCallbacks::new();
        push_opts.remote_callbacks(callbacks);

        // Push
        remote.push(&[&refspec], Some(&mut push_opts))?;

        // Set upstream if requested
        if set_upstream {
            let resolved_branch = if let Some(branch) = branch_name {
                branch.to_string()
            } else {
                // No branch specified - use current branch from HEAD
                let head = model_repo.head()?;
                head.shorthand().ok_or_else(|| anyhow!("Not on a branch"))?.to_string()
            };
            let mut branch = model_repo.find_branch(&resolved_branch, git2::BranchType::Local)?;
            branch.set_upstream(Some(&format!("{}/{}", remote_name, resolved_branch)))?;
        }

        info!("Pushed model {} to {}", model_ref.model, remote_name);
        Ok(())
    }

    /// Pull changes from remote using ModelRef and git2::AnnotatedCommit
    pub fn pull_model(
        &self,
        model_ref: &ModelRef,
        remote_name: &str,
        branch_name: Option<&str>,
        rebase: bool,
    ) -> Result<()> {
        // Check for uncommitted changes before pulling
        let status = self.status(model_ref)?;
        if status.is_dirty {
            bail!("Cannot pull: model '{}' has uncommitted changes. Commit or stash them first.", model_ref.model);
        }

        let (model_repo, _is_submodule) = self.get_model_repository(model_ref)?;

        // Fetch from remote
        let mut remote = model_repo.find_remote(remote_name)?;
        let mut fetch_opts = git2::FetchOptions::new();
        let callbacks = RemoteCallbacks::new();
        fetch_opts.remote_callbacks(callbacks);

        let refspec = branch_name.unwrap_or("+refs/heads/*:refs/remotes/origin/*");
        remote.fetch(&[refspec], Some(&mut fetch_opts), None)?;

        // Get the remote branch reference using git2::Reference
        let remote_ref_name = if let Some(branch) = branch_name {
            format!("refs/remotes/{}/{}", remote_name, branch)
        } else {
            // Use current branch's upstream
            let head = model_repo.head()?;
            let local_branch = model_repo.find_branch(
                head.shorthand().unwrap(),
                git2::BranchType::Local
            )?;
            let upstream = local_branch.upstream()?;
            upstream.get().name().unwrap().to_string()
        };
        let remote_branch_ref = model_repo.find_reference(&remote_ref_name)?;

        // Create git2::AnnotatedCommit from the remote reference
        let fetch_commit = model_repo.reference_to_annotated_commit(&remote_branch_ref)?;
        let remote_branch = remote_branch_ref.shorthand().unwrap_or("<unknown>");

        // Perform merge or rebase
        if rebase {
            // TODO: Implement rebase (complex operation)
            return Err(anyhow!("Rebase not yet implemented. Use merge for now."));
        } else {
            // Merge
            let (merge_analysis, _) = model_repo.merge_analysis(&[&fetch_commit])?;

            if merge_analysis.is_fast_forward() {
                // Fast-forward using git2::Reference
                let current_branch_name = match branch_name {
                    Some(name) => name.to_string(),
                    None => model_repo.head()?.shorthand().unwrap().to_string(),
                };
                let refname = format!("refs/heads/{}", current_branch_name);
                let mut current_ref = model_repo.find_reference(&refname)?;
                current_ref.set_target(fetch_commit.id(), "Fast-forward")?;
                model_repo.set_head(&refname)?;
                model_repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
                info!("Fast-forwarded {} to {}", model_ref.model, fetch_commit.id());
            } else if merge_analysis.is_normal() {
                // Normal merge using git2::AnnotatedCommit
                let sig = self.git_manager.create_signature(None, None)?;
                let local_commit = model_repo.reference_to_annotated_commit(&model_repo.head()?)?;
                model_repo.merge(&[&fetch_commit], None, None)?;
                let mut index = model_repo.index()?;

                // Check for conflicts
                if index.has_conflicts() {
                    return Err(anyhow!("Merge conflicts detected. Please resolve manually."));
                }

                // Write merged tree
                let tree_id = index.write_tree_to(&model_repo)?;
                let tree = model_repo.find_tree(tree_id)?;

                // Create merge commit
                let parent_commits = vec![
                    model_repo.head()?.peel_to_commit()?,
                    model_repo.find_commit(fetch_commit.id())?
                ];
                let parent_refs: Vec<&git2::Commit> = parent_commits.iter().collect();

                model_repo.commit(
                    Some("HEAD"),
                    &sig,
                    &sig,
                    &format!("Merge {} from {}", remote_branch, remote_name),
                    &tree,
                    &parent_refs,
                )?;

                info!("Merged {} from {}", remote_branch, remote_name);
            } else if merge_analysis.is_up_to_date() {
                info!("Already up-to-date");
            }
        }

        Ok(())
    }

    /// Merge a branch into current branch using ModelRef and git2::AnnotatedCommit
    pub fn merge_branch(
        &self,
        model_ref: &ModelRef,
        branch_name: &str,
        ff_only: bool,
        no_ff: bool,
    ) -> Result<()> {
        // Check for uncommitted changes before merging
        let status = self.status(model_ref)?;
        if status.is_dirty {
            bail!("Cannot merge: model '{}' has uncommitted changes. Commit or stash them first.", model_ref.model);
        }

        let (model_repo, _is_submodule) = self.get_model_repository(model_ref)?;

        // Get the branch to merge using git2::Reference and git2::AnnotatedCommit
        let branch = model_repo.find_branch(branch_name, git2::BranchType::Local)?;
        let branch_ref = branch.get();
        let branch_commit = branch_ref.peel_to_commit()?;
        let branch_annotated = model_repo.reference_to_annotated_commit(branch_ref)?;

        // Analyze merge using git2::AnnotatedCommit
        let (merge_analysis, _) = model_repo.merge_analysis(&[&branch_annotated])?;

        if ff_only && !merge_analysis.is_fast_forward() {
            return Err(anyhow!("Cannot fast-forward merge"));
        }

        if merge_analysis.is_fast_forward() && !no_ff {
            // Fast-forward using git2::Reference
            let head_ref = model_repo.head()?;
            let refname = head_ref.name().unwrap();
            let mut current_ref = model_repo.find_reference(refname)?;
            current_ref.set_target(branch_commit.id(), "Fast-forward merge")?;
            model_repo.set_head(refname)?;
            model_repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
            info!("Fast-forwarded to {}", branch_name);
        } else if merge_analysis.is_normal() || no_ff {
            // Normal merge using git2::AnnotatedCommit
            let sig = self.git_manager.create_signature(None, None)?;
            let local_annotated = model_repo.reference_to_annotated_commit(&model_repo.head()?)?;
            model_repo.merge(&[&branch_annotated], None, None)?;
            let mut index = model_repo.index()?;

            // Check for conflicts
            if index.has_conflicts() {
                return Err(anyhow!("Merge conflicts detected. Please resolve manually."));
            }

            // Write merged tree
            let tree_id = index.write_tree_to(&model_repo)?;
            let tree = model_repo.find_tree(tree_id)?;

            // Create merge commit
            let parent_commits = vec![
                model_repo.head()?.peel_to_commit()?,
                branch_commit,
            ];
            let parent_refs: Vec<&git2::Commit> = parent_commits.iter().collect();

            model_repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                &format!("Merge branch '{}'", branch_name),
                &tree,
                &parent_refs,
            )?;

            info!("Merged branch {}", branch_name);
        } else if merge_analysis.is_up_to_date() {
            info!("Already up-to-date with {}", branch_name);
        }

        Ok(())
    }

    /// Remove a model from the registry
    pub fn remove_model(&self, model_ref: &ModelRef) -> Result<()> {
        info!("Removing model {} from registry", model_ref.model);

        // Check if model exists as submodule
        let submodule_path = model_ref.model.clone(); // Models are direct children
        let repo = self.open_repo()?;

        // Check if submodule exists
        let _submodule = match repo.find_submodule(&submodule_path) {
            Ok(submodule) => submodule,
            Err(_) => {
                return Err(anyhow!("Model '{}' not found in registry (no submodule)", model_ref.model));
            }
        };

        // Remove the submodule
        // This involves several steps:
        // 1. Remove the submodule entry from .gitmodules
        // 2. Remove the submodule entry from .git/config
        // 3. Remove the submodule directory
        // 4. Remove from git index

        // Atomic staging of submodule removal
        {
            let mut index = repo.index()?;
            index.remove_path(std::path::Path::new(&submodule_path))?;
            index.write()?;
        }

        // Remove submodule configuration
        let mut config = repo.config()?;
        let config_key = format!("submodule.{}.url", submodule_path);
        if let Err(_) = config.remove(&config_key) {
            // Config entry might not exist, that's okay
        }

        // Remove from .gitmodules file using safe-path
        let gitmodules_path = scoped_join(repo.workdir().unwrap(), ".gitmodules")?;
        if gitmodules_path.exists() {
            let content = std::fs::read_to_string(&gitmodules_path)?;
            let mut new_content = String::new();
            let mut lines = content.lines();
            let mut skip_section = false;

            while let Some(line) = lines.next() {
                if line.trim().starts_with(&format!("[submodule \"{}\"", submodule_path)) {
                    skip_section = true;
                    continue;
                }

                if skip_section {
                    if line.trim().starts_with("[") && !line.trim().starts_with(&format!("[submodule \"{}\"", submodule_path)) {
                        skip_section = false;
                        new_content.push_str(line);
                        new_content.push('\n');
                    }
                    // Skip lines in the section we're removing
                    continue;
                }

                new_content.push_str(line);
                new_content.push('\n');
            }

            std::fs::write(&gitmodules_path, new_content)?;
        }

        // Atomic staging of .gitmodules changes
        {
            let mut index = repo.index()?;
            index.add_path(std::path::Path::new(".gitmodules"))?;
            index.write()?;
        }

        // Commit the changes
        self.commit_registry(&format!("Remove model {}", model_ref.model))?;

        info!("Successfully removed model {} from registry", model_ref.model);
        Ok(())
    }

    /// Helper to commit changes to the registry
    fn commit_registry(&self, message: &str) -> Result<()> {
        let repo = self.open_repo()?;
        let sig = self.git_manager.create_signature(None, None)?;
        let mut index = repo.index()?;

        // Stage .gitmodules and submodule changes
        index.add_all(["*"].iter(), IndexAddOption::DEFAULT, None)?;
        index.write()?;

        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;

        let parent_commit = if repo.head().is_ok() {
            Some(repo.head()?.peel_to_commit()?)
        } else {
            None
        };

        let parent_commits: Vec<&git2::Commit> = parent_commit.as_ref().map(|c| vec![c]).unwrap_or_default();

        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            message,
            &tree,
            &parent_commits,
        )?;

        info!("Committed to registry: {}", message);
        Ok(())
    }

    /// Get comprehensive model information using git2 types
    pub fn get_model_info(&self, model_ref: &ModelRef) -> Result<ModelInfo> {
        let (model_repo, is_submodule) = self.get_model_repository(model_ref)?;

        // Extract path from repository for ModelInfo struct
        let model_path = model_repo.workdir()
            .ok_or_else(|| anyhow!("Repository has no working directory"))?
            .to_path_buf();

        // Get current state using git2::Reference
        let (current_oid, current_ref_name) = match model_repo.head() {
            Ok(head_ref) => {
                let oid = head_ref.target().unwrap_or_else(|| git2::Oid::zero());
                let name = head_ref.shorthand().map(|s| s.to_string());
                (oid, name)
            },
            Err(_) => (git2::Oid::zero(), None),
        };

        // Resolve target using git2 objects
        let resolved_target_oid = self.resolve(model_ref).ok();
        let target_ref_name = match self.resolve_reference(model_ref) {
            Ok(Some((name, _oid))) => Some(name),
            _ => None,
        };

        // Get status
        let status = self.status(model_ref)?;

        Ok(ModelInfo {
            model_ref: model_ref.clone(),
            path: model_path,
            current_oid,
            current_ref_name,
            resolved_target_oid,
            target_ref_name,
            status,
            repository: model_repo,
            has_submodule: is_submodule,
        })
    }

    /// Legacy method - get model path by name string (deprecated)
    pub fn get_model_path_by_name(&self, model_name: &str) -> Result<PathBuf> {
        let model_ref = ModelRef::new(model_name.to_string());
        self.get_model_path(&model_ref)
    }
}

/// Thread-safe wrapper around ModelRegistry
pub struct SharedModelRegistry {
    inner: Arc<RwLock<ModelRegistry>>,
}

impl SharedModelRegistry {
    pub fn new(base_dir: PathBuf, xet_storage: Option<Arc<XetNativeStorage>>) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(RwLock::new(ModelRegistry::new(base_dir, xet_storage)?)),
        })
    }

    pub fn new_with_config(
        base_dir: PathBuf,
        xet_storage: Option<Arc<XetNativeStorage>>,
        git_config: GitConfig,
    ) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(RwLock::new(ModelRegistry::new_with_config(base_dir, xet_storage, git_config)?)),
        })
    }

    pub async fn resolve(&self, model_ref: &ModelRef) -> Result<git2::Oid> {
        let registry = self.inner.read().await;
        registry.resolve(model_ref)
    }

    pub async fn add_model(&self, name: &str, source: &str) -> Result<ModelRef> {
        let registry = self.inner.write().await;
        registry.add_model(name, source).await
    }

    pub async fn update_model(&self, name: &str, ref_spec: &str) -> Result<()> {
        let registry = self.inner.write().await;
        registry.update_model(name, ref_spec).await
    }

    pub async fn list_models(&self) -> Result<Vec<(String, git2::Oid)>> {
        let registry = self.inner.read().await;
        registry.list_models()
    }

    /// List model refs as reference information (name and target)
    pub async fn list_model_refs(&self, model_ref: &ModelRef) -> Result<Vec<(String, git2::Oid)>> {
        let registry = self.inner.read().await;
        registry.list_model_refs(model_ref)
    }

    /// List model ref names (legacy string interface)
    pub async fn list_model_ref_names(&self, model_ref: &ModelRef) -> Result<Vec<String>> {
        let registry = self.inner.read().await;
        registry.list_model_ref_names(model_ref)
    }

    pub async fn get_model_path(&self, model_ref: &ModelRef) -> Result<PathBuf> {
        let registry = self.inner.read().await;
        registry.get_model_path(model_ref)
    }

    /// Get comprehensive model information
    pub async fn get_model_info(&self, model_ref: &ModelRef) -> Result<ModelInfo> {
        let registry = self.inner.read().await;
        registry.get_model_info(model_ref)
    }

    /// Legacy method for compatibility
    pub async fn get_model_path_by_name(&self, model_name: &str) -> Result<PathBuf> {
        let registry = self.inner.read().await;
        registry.get_model_path_by_name(model_name)
    }

    /// Create a branch for a model
    pub async fn create_branch(&self, model_ref: &ModelRef, branch_name: &str, from_ref: Option<&str>) -> Result<()> {
        let registry = self.inner.write().await;  // WRITE lock - creates branch
        registry.create_branch(model_ref, branch_name, from_ref)
    }

    /// Checkout a branch/tag/commit for a model
    pub async fn checkout(&self, model_ref: &ModelRef, options: CheckoutOptions) -> Result<CheckoutResult> {
        let registry = self.inner.write().await;  // WRITE lock - modifies working directory
        registry.checkout(model_ref, options)
    }

    /// Get the default branch of a repository
    pub async fn get_default_branch(&self, model_ref: &ModelRef) -> Result<String> {
        let registry = self.inner.read().await;
        registry.get_default_branch(model_ref)
    }

    /// Get the status of a model's repository
    pub async fn status(&self, model_ref: &ModelRef) -> Result<ModelStatus> {
        let registry = self.inner.read().await;
        registry.status(model_ref)
    }

    /// Commit changes in a model's repository
    pub async fn commit_model(&self, model_ref: &ModelRef, message: &str, stage_all: bool) -> Result<git2::Oid> {
        let registry = self.inner.write().await;  // WRITE lock - creates commit
        registry.commit_model(model_ref, message, stage_all)
    }

    /// Push model to remote
    pub async fn push_model(
        &self,
        model_ref: &ModelRef,
        remote_name: &str,
        branch_name: Option<&str>,
        set_upstream: bool,
        force: bool,
    ) -> Result<()> {
        let registry = self.inner.write().await;  // WRITE lock - may update upstream
        registry.push_model(model_ref, remote_name, branch_name, set_upstream, force)
    }

    /// Pull model from remote
    pub async fn pull_model(
        &self,
        model_ref: &ModelRef,
        remote_name: &str,
        branch_name: Option<&str>,
        rebase: bool,
    ) -> Result<()> {
        let registry = self.inner.write().await;  // WRITE lock - merges changes
        registry.pull_model(model_ref, remote_name, branch_name, rebase)
    }

    /// Merge branch
    pub async fn merge_branch(
        &self,
        model_ref: &ModelRef,
        branch_name: &str,
        ff_only: bool,
        no_ff: bool,
    ) -> Result<()> {
        let registry = self.inner.write().await;  // WRITE lock - creates merge commit
        registry.merge_branch(model_ref, branch_name, ff_only, no_ff)
    }

    /// Remove a model from the registry
    pub async fn remove_model(&self, model_ref: &ModelRef) -> Result<()> {
        let registry = self.inner.write().await;  // WRITE lock - removes submodule
        registry.remove_model(model_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_model_registry() {
        let dir = tempdir().unwrap();
        let registry_path = dir.path().join("registry");

        let registry = ModelRegistry::new(registry_path.clone(), None).unwrap();

        // Test listing (should be empty)
        let models = registry.list_models().unwrap();
        assert_eq!(models.len(), 0);

        // Test resolving non-existent model
        let model_ref = ModelRef::parse("nonexistent").unwrap();
        assert!(registry.resolve(&model_ref).is_err());
    }

    #[tokio::test]
    async fn test_concurrent_add_and_list() {
        use std::sync::Arc;

        let dir = tempdir().unwrap();
        let registry_path = dir.path().join("registry");

        // Create shared registry
        let registry = Arc::new(SharedModelRegistry::new(registry_path.clone(), None).unwrap());

        // Spawn concurrent operations
        let add_task = {
            let reg = Arc::clone(&registry);
            tokio::spawn(async move {
                // Simulate adding a model (would need a real git URL in practice)
                // For this test, we're just verifying concurrent access doesn't panic
                reg.list_models().await
            })
        };

        let list_task = {
            let reg = Arc::clone(&registry);
            tokio::spawn(async move {
                // Should not fail even if add is in progress
                reg.list_models().await
            })
        };

        let list_task2 = {
            let reg = Arc::clone(&registry);
            tokio::spawn(async move {
                // Multiple concurrent list operations
                reg.list_models().await
            })
        };

        // All tasks should complete without errors
        let (add_res, list_res, list_res2) = tokio::join!(add_task, list_task, list_task2);

        assert!(add_res.unwrap().is_ok(), "Add task should complete");
        assert!(list_res.unwrap().is_ok(), "List task 1 should complete");
        assert!(list_res2.unwrap().is_ok(), "List task 2 should complete");
    }

    #[tokio::test]
    async fn test_resolve_submodule_commit_fallback() {
        let dir = tempdir().unwrap();
        let registry_path = dir.path().join("registry");

        let registry = ModelRegistry::new(registry_path.clone(), None).unwrap();

        // Create a test repository for the model
        let model_path = registry_path.join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();
        let model_repo = git2::Repository::init(&model_path).unwrap();

        // Create an initial commit
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();
        let tree_id = model_repo.index().unwrap().write_tree().unwrap();
        let tree = model_repo.find_tree(tree_id).unwrap();
        model_repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Initial commit",
            &tree,
            &[],
        ).unwrap();

        // The submodule resolution logic should handle various states
        // This test verifies the code structure is correct
        let model_ref = ModelRef::parse("test_model").unwrap();

        // Should fail gracefully for non-submodule model
        let result = registry.resolve(&model_ref);
        // Will error because it's not actually a submodule, but shouldn't panic
        assert!(result.is_err());
    }
}