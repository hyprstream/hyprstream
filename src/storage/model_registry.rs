//! Git-native model registry using submodules

use anyhow::{Result, anyhow, bail};
use git2::{Repository, Signature, IndexAddOption, FetchOptions, build::RepoBuilder};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn, debug};
use tokio::sync::RwLock;
use crate::git::{GitManager, GitConfig};

use super::model_ref::{ModelRef, validate_model_name};
use super::xet_native::XetNativeStorage;

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
            let mut callbacks = git2::RemoteCallbacks::new();
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
        let model_path = self.base_dir.parent()
            .ok_or_else(|| anyhow!("Invalid base directory"))?
            .join("models")
            .join(model_name);

        if !model_path.exists() {
            bail!("Model {} not found at {:?}", model_name, model_path);
        }

        Ok(model_path)
    }

    /// Create a branch for a model
    pub fn create_branch(&self, model_name: &str, branch_name: &str, from_ref: &str) -> Result<()> {
        if !git2::Reference::is_valid_name(branch_name) {
            bail!("Invalid git reference name: '{}'", branch_name);
        }
        if !git2::Reference::is_valid_name(from_ref) {
            bail!("Invalid git reference name: '{}'", from_ref);
        }

        let submodule_path = format!("models/{}", model_name);
        let repo = self.open_repo()?;
        let submodule = repo.find_submodule(&submodule_path)?;
        let model_repo = submodule.open()?;

        let commit = model_repo.revparse_single(from_ref)?.peel_to_commit()?;
        model_repo.branch(branch_name, &commit, false)?;

        info!("Created branch {} for model {} from {}", branch_name, model_name, from_ref);
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