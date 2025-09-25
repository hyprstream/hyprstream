//! Git-native model registry using submodules

use anyhow::{Result, anyhow, bail};
use git2::{Repository, IndexAddOption, FetchOptions, RemoteCallbacks, Signature, build::RepoBuilder};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;
use tokio::sync::RwLock;
use crate::git::{GitManager, GitConfig};

use super::model_ref::{ModelRef, validate_model_name};
use super::xet_native::XetNativeStorage;

/// Status of a model's git repository
#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub model_name: String,
    pub current_ref: String,
    pub is_dirty: bool,
    pub modified_files: Vec<String>,
    pub new_files: Vec<String>,
    pub deleted_files: Vec<String>,
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
            Repository::init(&base_dir)?; // One-time initialization
        }

        let git_manager = Arc::new(GitManager::new(git_config));

        Ok(Self {
            base_dir,
            xet_storage,
            git_manager,
        })
    }

    /// Open the repository when needed (with caching)
    fn open_repo(&self) -> Result<Repository> {
        self.git_manager.get_repository(&self.base_dir)
            .map_err(|e| anyhow!("Failed to open repository: {}", e))
    }

    /// Resolve a model reference to a specific commit SHA
    pub fn resolve(&self, model_ref: &ModelRef) -> Result<git2::Oid> {
        let submodule_path = format!("models/{}", model_ref.model);
        let repo = self.open_repo()?;

        // Check if this is a submodule
        let submodule_result = repo.find_submodule(&submodule_path);

        match submodule_result {
            Ok(submodule) => {
                if let Some(ref git_ref) = model_ref.git_ref {
                    // User specified a ref - resolve in model's repo
                    let model_repo = submodule.open()?;
                    let obj = model_repo.revparse_single(git_ref)?;
                    Ok(obj.id())
                } else {
                    // Use registry's pinned commit
                    let index_id = submodule.index_id()
                        .ok_or_else(|| anyhow!("Model {} not initialized", model_ref.model))?;
                    Ok(index_id)
                }
            }
            Err(_) => {
                // Not a submodule - check if it's a direct path (for development)
                let model_path = self.base_dir.parent()
                    .ok_or_else(|| anyhow!("Invalid base directory"))?
                    .join("models")
                    .join(&model_ref.model);

                if model_path.exists() {
                    let model_repo = self.git_manager.get_repository(&model_path)?;
                    if let Some(ref git_ref) = model_ref.git_ref {
                        Ok(model_repo.revparse_single(git_ref)?.id())
                    } else {
                        Ok(model_repo.head()?.peel_to_commit()?.id())
                    }
                } else {
                    bail!("Model '{}' not found", model_ref.model)
                }
            }
        }
    }

    /// Add a new model to the registry
    pub async fn add_model(&self, name: &str, source: &str) -> Result<()> {
        validate_model_name(name)?;

        let submodule_path = format!("models/{}", name);
        let model_path = self.base_dir.parent()
            .ok_or_else(|| anyhow!("Invalid base directory"))?
            .join("models")
            .join(name);

        // Clone the model repository
        info!("Cloning model {} from {}", name, source);

        let mut builder = RepoBuilder::new();

        // Configure XET callbacks if available
        if let Some(_xet) = &self.xet_storage {
            // XET will handle LFS pointers automatically
            // Configure callbacks if needed
            let callbacks = git2::RemoteCallbacks::new();
            let mut fetch_options = git2::FetchOptions::new();
            fetch_options.remote_callbacks(callbacks);
            builder.fetch_options(fetch_options);
        }

        let _model_repo = builder.clone(source, &model_path)?;

        // Add as submodule to registry
        info!("Adding {} as submodule to registry", name);
        let repo = self.open_repo()?;
        repo.submodule(
            &model_path.to_str().unwrap(),
            Path::new(&submodule_path),
            false
        )?;

        // Commit to registry
        self.commit_registry(&format!("Add model: {}", name))?;

        Ok(())
    }

    /// Update a model to a different version
    pub async fn update_model(&self, name: &str, ref_spec: &str) -> Result<()> {
        if !git2::Reference::is_valid_name(ref_spec) {
            bail!("Invalid git reference name: '{}'", ref_spec);
        }

        let submodule_path = format!("models/{}", name);
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

        // Update submodule pointer in registry
        let mut index = self.open_repo()?.index()?;
        index.add_path(Path::new(&submodule_path))?;
        index.write()?;

        // Commit registry change
        self.commit_registry(&format!("Update {} to {}", name, ref_spec))?;

        Ok(())
    }

    /// List all registered models with their current commits
    pub fn list_models(&self) -> Result<Vec<(String, git2::Oid)>> {
        let mut models = Vec::new();

        // List submodules
        for submodule in self.open_repo()?.submodules()? {
            if let Some(path) = submodule.path().to_str() {
                if let Some(name) = path.strip_prefix("models/") {
                    if let Some(commit) = submodule.index_id() {
                        models.push((name.to_string(), commit));
                    }
                }
            }
        }

        Ok(models)
    }

    /// List available refs for a model (branches and tags)
    pub fn list_model_refs(&self, model_name: &str) -> Result<Vec<String>> {
        let submodule_path = format!("models/{}", model_name);
        let repo = self.open_repo()?;
        let submodule = repo.find_submodule(&submodule_path)?;
        let model_repo = submodule.open()?;

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

    /// Get the path to a model's repository
    pub fn get_model_path(&self, model_name: &str) -> Result<PathBuf> {
        // base_dir is registry dir, parent is already models dir
        let model_path = self.base_dir.parent()
            .ok_or_else(|| anyhow!("Invalid base directory"))?
            .join(model_name);

        if !model_path.exists() {
            bail!("Model {} not found at {:?}", model_name, model_path);
        }

        Ok(model_path)
    }

    /// Create a branch for a model
    pub fn create_branch(&self, model_name: &str, branch_name: &str, from_ref: Option<&str>) -> Result<()> {
        let model_path = self.get_model_path(model_name)?;
        let model_repo = self.git_manager.get_repository(&model_path)?;

        // Get the commit to branch from
        let commit = if let Some(ref_str) = from_ref {
            model_repo.revparse_single(ref_str)?.peel_to_commit()?
        } else {
            model_repo.head()?.peel_to_commit()?
        };

        // Create the branch
        model_repo.branch(branch_name, &commit, false)?;

        info!("Created branch {} for model {} from {}",
              branch_name, model_name, from_ref.unwrap_or("HEAD"));
        Ok(())
    }

    /// Checkout a specific branch/tag/commit for a model
    pub fn checkout(&self, model_name: &str, ref_spec: Option<&str>, create_branch: bool) -> Result<()> {
        let model_path = self.get_model_path(model_name)?;
        let model_repo = self.git_manager.get_repository(&model_path)?;

        // Parse the ref_spec
        let target = if let Some(ref_str) = ref_spec {
            if create_branch {
                // Create branch and checkout
                let commit = model_repo.head()?.peel_to_commit()?;
                model_repo.branch(ref_str, &commit, false)?;
                format!("refs/heads/{}", ref_str)
            } else {
                // Try to resolve as branch, tag, or commit
                if let Ok(branch) = model_repo.find_branch(ref_str, git2::BranchType::Local) {
                    branch.get().name().unwrap().to_string()
                } else if let Ok(_) = model_repo.revparse_single(&format!("refs/tags/{}", ref_str)) {
                    format!("refs/tags/{}", ref_str)
                } else {
                    // Try as commit SHA
                    ref_str.to_string()
                }
            }
        } else {
            return Err(anyhow!("No ref specified for checkout"));
        };

        // Checkout the target
        let obj = model_repo.revparse_single(&target)?;

        if let Ok(commit) = obj.clone().into_commit() {
            // Detached HEAD checkout for commits
            model_repo.set_head_detached(commit.id())?;
        } else {
            // Branch/tag checkout
            model_repo.set_head(&target)?;
        }

        // Update working directory
        model_repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;

        info!("Checked out {} for model {}", target, model_name);
        Ok(())
    }

    /// Get the status of a model's repository
    pub fn status(&self, model_name: &str) -> Result<ModelStatus> {
        let model_path = self.get_model_path(model_name)?;
        let model_repo = self.git_manager.get_repository(&model_path)?;

        // Get current branch/ref
        let head = model_repo.head()?;
        let current_ref = if head.is_branch() {
            head.shorthand().unwrap_or("unknown").to_string()
        } else {
            format!("detached at {}", head.target().unwrap().to_string().chars().take(7).collect::<String>())
        };

        // Get status of working directory
        let statuses = model_repo.statuses(None)?;
        let mut modified_files = Vec::new();
        let mut new_files = Vec::new();
        let mut deleted_files = Vec::new();

        for entry in statuses.iter() {
            let path = entry.path().unwrap_or("unknown").to_string();
            let status = entry.status();

            if status.contains(git2::Status::WT_NEW) {
                new_files.push(path);
            } else if status.contains(git2::Status::WT_MODIFIED) {
                modified_files.push(path);
            } else if status.contains(git2::Status::WT_DELETED) {
                deleted_files.push(path);
            }
        }

        Ok(ModelStatus {
            model_name: model_name.to_string(),
            current_ref,
            is_dirty: !statuses.is_empty(),
            modified_files,
            new_files,
            deleted_files,
        })
    }

    /// Commit changes in a model's repository
    pub fn commit_model(&self, model_name: &str, message: &str, stage_all: bool) -> Result<()> {
        let model_path = self.get_model_path(model_name)?;
        let model_repo = self.git_manager.get_repository(&model_path)?;

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

        // Create commit
        model_repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            message,
            &tree,
            &[&parent_commit],
        )?;

        info!("Committed changes to model {}: {}", model_name, message);
        Ok(())
    }

    /// Push changes to remote
    pub fn push_model(
        &self,
        model_name: &str,
        remote_name: &str,
        branch_name: Option<&str>,
        set_upstream: bool,
        force: bool,
    ) -> Result<()> {
        let model_path = self.get_model_path(model_name)?;
        let model_repo = self.git_manager.get_repository(&model_path)?;

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
            let resolved_branch = branch_name.unwrap_or("main");
            let mut branch = model_repo.find_branch(resolved_branch, git2::BranchType::Local)?;
            branch.set_upstream(Some(&format!("{}/{}", remote_name, resolved_branch)))?;
        }

        info!("Pushed model {} to {}", model_name, remote_name);
        Ok(())
    }

    /// Pull changes from remote
    pub fn pull_model(
        &self,
        model_name: &str,
        remote_name: &str,
        branch_name: Option<&str>,
        rebase: bool,
    ) -> Result<()> {
        // Check for uncommitted changes before pulling
        let status = self.status(model_name)?;
        if status.is_dirty {
            bail!("Cannot pull: model '{}' has uncommitted changes. Commit or stash them first.", model_name);
        }

        let model_path = self.get_model_path(model_name)?;
        let model_repo = self.git_manager.get_repository(&model_path)?;

        // Fetch from remote
        let mut remote = model_repo.find_remote(remote_name)?;
        let mut fetch_opts = git2::FetchOptions::new();
        let callbacks = RemoteCallbacks::new();
        fetch_opts.remote_callbacks(callbacks);

        let refspec = branch_name.unwrap_or("+refs/heads/*:refs/remotes/origin/*");
        remote.fetch(&[refspec], Some(&mut fetch_opts), None)?;

        // Get the remote branch reference
        let remote_branch = if let Some(branch) = branch_name {
            format!("{}/{}", remote_name, branch)
        } else {
            // Use current branch's upstream
            let head = model_repo.head()?;
            let local_branch = model_repo.find_branch(
                head.shorthand().unwrap(),
                git2::BranchType::Local
            )?;
            let upstream = local_branch.upstream()?;
            upstream.name()?.unwrap().to_string()
        };

        // Get commits
        let fetch_commit = model_repo.reference_to_annotated_commit(
            &model_repo.find_reference(&format!("refs/remotes/{}", remote_branch))?
        )?;

        // Perform merge or rebase
        if rebase {
            // TODO: Implement rebase (complex operation)
            return Err(anyhow!("Rebase not yet implemented. Use merge for now."));
        } else {
            // Merge
            let (merge_analysis, _) = model_repo.merge_analysis(&[&fetch_commit])?;

            if merge_analysis.is_fast_forward() {
                // Fast-forward
                let refname = format!("refs/heads/{}",
                    branch_name.unwrap_or(model_repo.head()?.shorthand().unwrap()));
                let mut reference = model_repo.find_reference(&refname)?;
                reference.set_target(fetch_commit.id(), "Fast-forward")?;
                model_repo.set_head(&refname)?;
                model_repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
                info!("Fast-forwarded {} to {}", model_name, fetch_commit.id());
            } else if merge_analysis.is_normal() {
                // Normal merge
                let sig = Signature::now("hyprstream", "hyprstream@local")?;
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

    /// Merge a branch into current branch
    pub fn merge_branch(
        &self,
        model_name: &str,
        branch_name: &str,
        ff_only: bool,
        no_ff: bool,
    ) -> Result<()> {
        // Check for uncommitted changes before merging
        let status = self.status(model_name)?;
        if status.is_dirty {
            bail!("Cannot merge: model '{}' has uncommitted changes. Commit or stash them first.", model_name);
        }

        let model_path = self.get_model_path(model_name)?;
        let model_repo = self.git_manager.get_repository(&model_path)?;

        // Get the branch to merge
        let branch = model_repo.find_branch(branch_name, git2::BranchType::Local)?;
        let branch_commit = branch.get().peel_to_commit()?;
        let branch_oid = model_repo.reference_to_annotated_commit(branch.get())?;

        // Analyze merge
        let (merge_analysis, _) = model_repo.merge_analysis(&[&branch_oid])?;

        if ff_only && !merge_analysis.is_fast_forward() {
            return Err(anyhow!("Cannot fast-forward merge"));
        }

        if merge_analysis.is_fast_forward() && !no_ff {
            // Fast-forward
            let head_ref = model_repo.head()?;
            let refname = head_ref.name().unwrap();
            let mut reference = model_repo.find_reference(refname)?;
            reference.set_target(branch_commit.id(), "Fast-forward merge")?;
            model_repo.set_head(refname)?;
            model_repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
            info!("Fast-forwarded to {}", branch_name);
        } else if merge_analysis.is_normal() || no_ff {
            // Normal merge or forced merge commit
            let sig = Signature::now("hyprstream", "hyprstream@local")?;
            let local_commit = model_repo.reference_to_annotated_commit(&model_repo.head()?)?;
            model_repo.merge(&[&branch_oid], None, None)?;
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

    pub async fn resolve(&self, model_ref: &ModelRef) -> Result<git2::Oid> {
        let registry = self.inner.read().await;
        registry.resolve(model_ref)
    }

    pub async fn add_model(&self, name: &str, source: &str) -> Result<()> {
        let registry = self.inner.read().await;
        registry.add_model(name, source).await
    }

    pub async fn update_model(&self, name: &str, ref_spec: &str) -> Result<()> {
        let registry = self.inner.read().await;
        registry.update_model(name, ref_spec).await
    }

    pub async fn list_models(&self) -> Result<Vec<(String, git2::Oid)>> {
        let registry = self.inner.read().await;
        registry.list_models()
    }

    pub async fn list_model_refs(&self, model_name: &str) -> Result<Vec<String>> {
        let registry = self.inner.read().await;
        registry.list_model_refs(model_name)
    }

    pub async fn get_model_path(&self, model_name: &str) -> Result<PathBuf> {
        let registry = self.inner.read().await;
        registry.get_model_path(model_name)
    }

    /// Create a branch for a model
    pub async fn create_branch(&self, model_name: &str, branch_name: &str, from_ref: Option<&str>) -> Result<()> {
        let registry = self.inner.read().await;
        registry.create_branch(model_name, branch_name, from_ref)
    }

    /// Checkout a branch/tag/commit for a model
    pub async fn checkout(&self, model_name: &str, ref_spec: Option<&str>, create_branch: bool) -> Result<()> {
        let registry = self.inner.read().await;
        registry.checkout(model_name, ref_spec, create_branch)
    }

    /// Get the status of a model's repository
    pub async fn status(&self, model_name: &str) -> Result<ModelStatus> {
        let registry = self.inner.read().await;
        registry.status(model_name)
    }

    /// Commit changes in a model's repository
    pub async fn commit_model(&self, model_name: &str, message: &str, stage_all: bool) -> Result<()> {
        let registry = self.inner.read().await;
        registry.commit_model(model_name, message, stage_all)
    }

    /// Push model to remote
    pub async fn push_model(
        &self,
        model_name: &str,
        remote_name: &str,
        branch_name: Option<&str>,
        set_upstream: bool,
        force: bool,
    ) -> Result<()> {
        let registry = self.inner.read().await;
        registry.push_model(model_name, remote_name, branch_name, set_upstream, force)
    }

    /// Pull model from remote
    pub async fn pull_model(
        &self,
        model_name: &str,
        remote_name: &str,
        branch_name: Option<&str>,
        rebase: bool,
    ) -> Result<()> {
        let registry = self.inner.read().await;
        registry.pull_model(model_name, remote_name, branch_name, rebase)
    }

    /// Merge branch
    pub async fn merge_branch(
        &self,
        model_name: &str,
        branch_name: &str,
        ff_only: bool,
        no_ff: bool,
    ) -> Result<()> {
        let registry = self.inner.read().await;
        registry.merge_branch(model_name, branch_name, ff_only, no_ff)
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
}