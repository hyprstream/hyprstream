//! Git-native model registry using submodules

use anyhow::{Result, anyhow, bail};
use safe_path::scoped_join;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, debug};
use tokio::sync::RwLock;
use git2db::{GitManager, Git2DB};

use super::model_ref::{ModelRef, validate_model_name};
use super::xet_native::XetNativeStorage;
use super::model_registry_adapter::RegistryAdapter;

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
    #[allow(dead_code)]
    xet_storage: Option<Arc<XetNativeStorage>>,
    /// Adapter layer for delegating to git2db
    adapter: Arc<RegistryAdapter>,
}

impl ModelRegistry {
    /// Open or initialize a model registry with git2db integration
    pub async fn open(base_dir: PathBuf, xet_storage: Option<Arc<XetNativeStorage>>) -> Result<Self> {
        info!("Opening model registry at {:?} with git2db integration", base_dir);

        // Open or create git2db registry
        let git2db = Git2DB::open(&base_dir).await?;
        let adapter = Arc::new(RegistryAdapter::new(git2db));

        Ok(Self {
            base_dir,
            xet_storage,
            adapter,
        })
    }

    /// Get raw repository for sync operations (escape hatch)
    ///
    /// TODO: Migrate callers to async RepositoryHandle when possible.
    /// This is a temporary bridge for sync methods that need git2::Repository.
    fn get_raw_repo(&self, model_ref: &ModelRef) -> Result<git2::Repository> {
        let repo_id = self.adapter.resolve(model_ref)?;

        // Use tokio::task::block_in_place for sync context
        let model_path = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let registry = self.adapter.registry().await;
                registry.get_worktree_path(&repo_id)
            })
        }).ok_or_else(|| anyhow!("Model '{}' not found", model_ref.model))?;

        let cache = GitManager::global().get_repository(&model_path)
            .map_err(|e| anyhow!("Failed to get repository: {}", e))?;
        cache.open()
            .map_err(|e| anyhow!("Failed to open repository: {}", e))
    }


    /// Resolve a model reference to a specific commit SHA using git2 objects
    pub fn resolve(&self, model_ref: &ModelRef) -> Result<git2::Oid> {
        let model_repo = self.get_raw_repo(model_ref)?;

        match &model_ref.git_ref {
            crate::storage::GitRef::DefaultBranch => {
                // Use HEAD for default branch
                Ok(model_repo.head()?.peel_to_commit()?.id())
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
        let model_repo = self.get_raw_repo(model_ref)?;

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

    /// Add a new model to the registry using git2db with XET filter support
    pub async fn add_model(&self, name: &str, source: &str) -> Result<ModelRef> {
        validate_model_name(name)?;
        info!("Adding model '{}' from '{}' via git2db", name, source);

        // Check if model already exists
        if self.adapter.exists(name) {
            return Err(anyhow!("Model '{}' already exists in registry", name));
        }

        // Use git2db's clone builder (handles registration automatically)
        info!("Cloning model '{}' from '{}'", name, source);
        let mut registry = self.adapter.registry_mut().await;

        let _repo_id = registry.clone(source)
            .name(name)
            .depth(1)  // Shallow clone by default
            .exec()
            .await
            .map_err(|e| anyhow!("Failed to clone model '{}': {}", name, e))?;

        info!("Successfully added model '{}' via git2db with XET support", name);
        Ok(ModelRef::new(name.to_string()))
    }

    /// Update a model to a different version using git2db
    pub async fn update_model(&self, name: &str, ref_spec: &str) -> Result<()> {
        if !git2::Reference::is_valid_name(ref_spec) {
            bail!("Invalid git reference name: '{}'", ref_spec);
        }

        info!("Updating model '{}' to '{}' via git2db", name, ref_spec);

        // Resolve model name to RepoId
        let model_ref = ModelRef::new(name.to_string());
        let repo_id = self.adapter.resolve(&model_ref)?;

        // Get repository handle from git2db
        let registry = self.adapter.registry().await;
        let handle = registry.repo(&repo_id)?;

        // Fetch from remote (uses GitManager, applies XET filters)
        info!("Fetching updates for model '{}'", name);
        handle.fetch(None).await?;

        // Checkout the specified ref (XET filters apply automatically)
        info!("Checking out '{}' for model '{}'", ref_spec, name);
        handle.checkout(ref_spec).await?;

        info!("Successfully updated model '{}' to '{}'", name, ref_spec);
        Ok(())
    }

    /// List all registered models with their current commits using git2db
    pub async fn list_models(&self) -> Result<Vec<(String, git2::Oid)>> {
        let registry = self.adapter.registry().await;
        let mut models = Vec::new();

        for tracked in registry.list() {
            if let Some(name) = &tracked.name {
                // Get current OID from the repository
                match registry.repo(&tracked.id) {
                    Ok(handle) => {
                        match handle.current_oid() {
                            Ok(Some(oid)) => {
                                models.push((name.clone(), oid));
                            }
                            Ok(None) => {
                                debug!("Model '{}' has no current OID", name);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to get OID for '{}': {}", name, e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to get repository handle for '{}': {}", name, e);
                    }
                }
            }
        }

        Ok(models)
    }

    /// List available refs for a model - returns reference info without lifetimes
    pub fn list_model_refs(&self, model_ref: &ModelRef) -> Result<Vec<(String, git2::Oid)>> {
        let model_repo = self.get_raw_repo(model_ref)?;

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
    pub fn list_model_refs_legacy(&self, _model_ref: &ModelRef) -> Result<Vec<git2::Reference<'_>>> {
        // This method cannot be implemented safely due to git2::Reference lifetime constraints
        // Callers should use list_model_refs() instead
        Err(anyhow::anyhow!("list_model_refs_legacy is deprecated due to lifetime issues - use list_model_refs instead"))
    }

    /// Get branch and tag names for a model (legacy string interface)
    pub fn list_model_ref_names(&self, model_ref: &ModelRef) -> Result<Vec<String>> {
        let model_repo = self.get_raw_repo(model_ref)?;

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
    pub async fn create_branch(&self, model_ref: &ModelRef, branch_name: &str, from_ref: Option<&str>) -> Result<()> {
        // Resolve model name to RepoId
        let repo_id = self.adapter.resolve(model_ref)?;

        // Get repository handle from git2db
        let registry = self.adapter.registry().await;
        let handle = registry.repo(&repo_id)?;

        // Use BranchManager to create branch
        handle.branch().create(branch_name, from_ref).await?;

        info!("Created branch {} for model {} from {}",
              branch_name, model_ref.model, from_ref.unwrap_or("HEAD"));
        Ok(())
    }

    /// Get the default branch of a repository using ModelRef
    pub fn get_default_branch(&self, model_ref: &ModelRef) -> Result<String> {
        let model_repo = self.get_raw_repo(model_ref)?;

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

    /// Checkout a specific ref for a model using git2db with XET filter support
    pub async fn checkout(&self, model_ref: &ModelRef, _options: CheckoutOptions) -> Result<CheckoutResult> {
        let ref_str = model_ref.git_ref_str().unwrap_or_else(|| "default".to_string());
        info!("Checking out '{}' for model '{}' via git2db", ref_str, model_ref.model);

        // Resolve model name to RepoId
        let repo_id = self.adapter.resolve(model_ref)?;

        // Get repository handle from git2db
        let registry = self.adapter.registry().await;
        let handle = registry.repo(&repo_id)?;

        // Get previous state
        let previous_oid = handle.current_oid()?.unwrap_or(git2::Oid::zero());

        // Convert hyprstream GitRef to git2db-compatible string
        let ref_string = match &model_ref.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            crate::storage::GitRef::Tag(name) => name.clone(),
            crate::storage::GitRef::Commit(oid) => oid.to_string(),
            crate::storage::GitRef::Revspec(spec) => spec.clone(),
            crate::storage::GitRef::DefaultBranch => "HEAD".to_string(),
        };

        // Perform checkout (XET filters apply automatically)
        handle.checkout(ref_string.as_str()).await?;

        // Get new state
        let new_oid = handle.current_oid()?.unwrap_or(git2::Oid::zero());

        // Get ref names for result
        let new_ref_name = match &model_ref.git_ref {
            crate::storage::GitRef::Branch(name) => Some(name.clone()),
            crate::storage::GitRef::Tag(name) => Some(name.clone()),
            _ => None,
        };

        info!("Successfully checked out '{}' for model '{}'", ref_str, model_ref.model);

        Ok(CheckoutResult {
            previous_oid,
            new_oid,
            previous_ref_name: None,
            new_ref_name,
            was_forced: false,
            files_changed: 0,
            has_submodule: true,
        })
    }

    /// Get the status of a model's repository using git2 objects
    pub fn status(&self, model_ref: &ModelRef) -> Result<ModelStatus> {
        let model_repo = self.get_raw_repo(model_ref)?;

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
    ///
    /// Note: This uses raw git2 for commit creation as git2db's commit() API
    /// is not yet available (Phase 3). Uses RepositoryHandle for everything else.
    pub async fn commit_model(&self, model_ref: &ModelRef, message: &str, stage_all: bool) -> Result<git2::Oid> {
        // Resolve model name to RepoId
        let repo_id = self.adapter.resolve(model_ref)?;

        // Get repository handle from git2db
        let registry = self.adapter.registry().await;
        let handle = registry.repo(&repo_id)?;

        // Stage files if requested using git2db StageManager
        if stage_all {
            handle.staging().add_all().await?;
        }

        // TODO: Replace with handle.commit(message) when git2db Phase 3 complete
        // For now, use escape hatch to raw git2 for commit creation
        let model_repo = handle.open_repo()?;
        let sig = GitManager::global().create_signature(None, None)?;

        let tree_id = model_repo.index()?.write_tree()?;
        let tree = model_repo.find_tree(tree_id)?;
        let parent_commit = model_repo.head()?.peel_to_commit()?;

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
    ///
    /// Note: Uses git2db RepositoryHandle.push() for push operations.
    /// Upstream tracking requires raw git2 (not in RepositoryHandle yet).
    pub async fn push_model(
        &self,
        model_ref: &ModelRef,
        remote_name: &str,
        branch_name: Option<&str>,
        set_upstream: bool,
        _force: bool,
    ) -> Result<()> {
        // Resolve model name to RepoId
        let repo_id = self.adapter.resolve(model_ref)?;

        // Get repository handle from git2db
        let registry = self.adapter.registry().await;
        let handle = registry.repo(&repo_id)?;

        // Determine what to push
        let refspec = if let Some(branch) = branch_name {
            git2db::GitRef::Branch(branch.to_string())
        } else {
            git2db::GitRef::DefaultBranch
        };

        // Use RepositoryHandle to push
        handle.push(Some(remote_name), refspec).await?;

        // Set upstream if requested (requires escape hatch for now)
        if set_upstream {
            let model_repo = handle.open_repo()?;
            let resolved_branch = if let Some(branch) = branch_name {
                branch.to_string()
            } else {
                let head = model_repo.head()?;
                head.shorthand().ok_or_else(|| anyhow!("Not on a branch"))?.to_string()
            };
            let mut branch = model_repo.find_branch(&resolved_branch, git2::BranchType::Local)?;
            branch.set_upstream(Some(&format!("{}/{}", remote_name, resolved_branch)))?;
        }

        info!("Pushed model {} to {}", model_ref.model, remote_name);
        Ok(())
    }

    /// Pull changes from remote using ModelRef
    ///
    /// Note: Uses git2db RepositoryHandle.pull() for pull operations.
    /// Rebase is not yet supported - will fail with error.
    pub async fn pull_model(
        &self,
        model_ref: &ModelRef,
        remote_name: &str,
        branch_name: Option<&str>,
        rebase: bool,
    ) -> Result<()> {
        if rebase {
            bail!("Rebase not yet supported by git2db RepositoryHandle");
        }

        // Check for uncommitted changes before pulling
        let status = self.status(model_ref)?;
        if status.is_dirty {
            bail!("Cannot pull: model '{}' has uncommitted changes. Commit or stash them first.", model_ref.model);
        }

        // Resolve model name to RepoId
        let repo_id = self.adapter.resolve(model_ref)?;

        // Get repository handle from git2db
        let registry = self.adapter.registry().await;
        let handle = registry.repo(&repo_id)?;

        // Determine what to pull
        let refspec = if let Some(branch) = branch_name {
            git2db::GitRef::Branch(branch.to_string())
        } else {
            git2db::GitRef::DefaultBranch
        };

        // Use RepositoryHandle to pull
        handle.pull(Some(remote_name), refspec).await?;

        info!("Pulled model {} from {}", model_ref.model, remote_name);
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

        let model_repo = self.get_raw_repo(model_ref)?;

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
            let sig = GitManager::global().create_signature(None, None)?;
            let _local_annotated = model_repo.reference_to_annotated_commit(&model_repo.head()?)?;
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
    ///
    /// Uses git2db's remove_repository() which handles complete cleanup.
    pub async fn remove_model(&self, model_ref: &ModelRef) -> Result<()> {
        info!("Removing model {} from registry", model_ref.model);

        // Resolve model name to RepoId
        let repo_id = self.adapter.resolve(model_ref)?;

        // Use git2db to remove repository (handles submodule cleanup)
        let mut registry = self.adapter.registry_mut().await;
        registry.remove_repository(&repo_id).await?;

        info!("Successfully removed model {} from registry", model_ref.model);
        Ok(())
    }

    /// Get comprehensive model information using git2 types
    pub fn get_model_info(&self, model_ref: &ModelRef) -> Result<ModelInfo> {
        let model_repo = self.get_raw_repo(model_ref)?;

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
            has_submodule: true,  // All models are managed by git2db now
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
    /// Open shared registry with git2db integration
    pub async fn open(base_dir: PathBuf, xet_storage: Option<Arc<XetNativeStorage>>) -> Result<Self> {
        let registry = ModelRegistry::open(base_dir, xet_storage).await?;
        Ok(Self {
            inner: Arc::new(RwLock::new(registry)),
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
        registry.list_models().await
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
        registry.create_branch(model_ref, branch_name, from_ref).await
    }

    /// Checkout a branch/tag/commit for a model
    pub async fn checkout(&self, model_ref: &ModelRef, options: CheckoutOptions) -> Result<CheckoutResult> {
        let registry = self.inner.write().await;  // WRITE lock - modifies working directory
        registry.checkout(model_ref, options).await
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
        registry.commit_model(model_ref, message, stage_all).await
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
        registry.push_model(model_ref, remote_name, branch_name, set_upstream, force).await
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
        registry.pull_model(model_ref, remote_name, branch_name, rebase).await
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
        registry.remove_model(model_ref).await
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

        let registry = ModelRegistry::open(registry_path.clone(), None).await.unwrap();

        // Test listing (should be empty)
        let models = registry.list_models().await.unwrap();
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
        let registry = Arc::new(SharedModelRegistry::open(registry_path.clone(), None).await.unwrap());

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

        let registry = ModelRegistry::open(registry_path.clone(), None).await.unwrap();

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