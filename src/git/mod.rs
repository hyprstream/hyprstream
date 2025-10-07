//! Git operations for model version control
//!
//! This module contains hyprstream-specific git extensions.
//! Core git functionality is provided by git2db.

pub mod branch_manager;

pub use branch_manager::{BranchManager, BranchInfo};

// Re-export commonly used git2db types
pub use git2db::{Git2DB as GitModelRegistry, TrackedRepository};

// Compatibility alias for internal use
pub type RegisteredModel = TrackedRepository;

// Hyprstream-specific types and traits
use anyhow::Result;
use git2::{Repository, Signature, FetchOptions, RemoteCallbacks};
use git2::build::CheckoutBuilder;
use std::path::Path;
use std::sync::Arc;
use git2db::{GitManager, Git2DBError as GitError};

/// Progress trait for git operations
pub trait GitProgress: Send + Sync {
    fn on_progress(&self, progress: &GitProgressInfo);
    fn on_error(&self, error: &GitError);
    fn is_cancelled(&self) -> bool;
}

/// Progress info for git operations
#[derive(Debug, Clone)]
pub struct GitProgressInfo {
    pub operation: String,
    pub current: usize,
    pub total: usize,
    pub bytes_received: usize,
    pub bytes_total: usize,
    pub elapsed: std::time::Duration,
}

/// Git operations trait for Repository extensions
pub trait GitOperations {
    fn fetch_with_progress(&self, progress: Option<Arc<dyn GitProgress>>) -> Result<()>;
    fn create_branch(&self, name: &str, target: Option<&str>) -> Result<()>;
    fn checkout_branch(&self, name: &str) -> Result<()>;
    fn commit_changes(&self, message: &str, signature: Option<&Signature>) -> Result<()>;
}

impl GitOperations for Repository {
    fn fetch_with_progress(&self, _progress: Option<Arc<dyn GitProgress>>) -> Result<()> {
        let mut remote = self.find_remote("origin")?;
        let mut fetch_opts = FetchOptions::new();
        let callbacks = RemoteCallbacks::new();
        fetch_opts.remote_callbacks(callbacks);
        remote.fetch(&[] as &[&str], Some(&mut fetch_opts), None)?;
        Ok(())
    }

    fn create_branch(&self, name: &str, target: Option<&str>) -> Result<()> {
        let commit = if let Some(target_ref) = target {
            self.revparse_single(target_ref)?.peel_to_commit()?
        } else {
            self.head()?.peel_to_commit()?
        };
        self.branch(name, &commit, false)?;
        Ok(())
    }

    fn checkout_branch(&self, name: &str) -> Result<()> {
        let obj = self.revparse_single(name)?;
        self.checkout_tree(&obj, Some(CheckoutBuilder::default().force()))?;
        self.set_head(&format!("refs/heads/{}", name))?;
        Ok(())
    }

    fn commit_changes(&self, message: &str, signature: Option<&Signature>) -> Result<()> {
        let sig = match signature {
            Some(s) => git2::Signature::now(
                s.name().unwrap_or("unknown"),
                s.email().unwrap_or("unknown@local")
            )?,
            None => GitManager::global().create_signature(None, None)?,
        };

        let mut index = self.index()?;
        index.add_all(["."].iter(), git2::IndexAddOption::DEFAULT, None)?;
        index.write()?;

        let tree_id = index.write_tree()?;
        let tree = self.find_tree(tree_id)?;
        let parent = self.head()?.peel_to_commit()?;

        self.commit(
            Some("HEAD"),
            &sig,
            &sig,
            message,
            &tree,
            &[&parent],
        )?;

        Ok(())
    }
}

/// Shareable model reference for P2P model sharing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShareableModelRef {
    pub name: String,
    pub uuid: uuid::Uuid,
    pub git_commit: String,
    pub source: Option<String>,
}

/// Get a repository handle using the global GitManager
pub fn get_repository<P: AsRef<Path>>(path: P) -> Result<Repository> {
    GitManager::global().get_repository(path)
        .map_err(|e| anyhow::anyhow!("Failed to get repository: {}", e))
}

/// Clone a repository using the global GitManager
pub async fn clone_repository<P: AsRef<Path>>(
    url: &str,
    path: P,
    options: Option<git2db::clone_options::CloneOptions>,
    _progress: Option<Arc<dyn GitProgress>>,
) -> Result<Repository> {
    // Note: Progress callback not yet fully supported
    GitManager::global()
        .clone_repository(url, path.as_ref(), options)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to clone repository: {}", e))
}

/// Create a signature using the global GitManager
pub fn create_signature(name: Option<&str>, email: Option<&str>) -> Result<Signature<'static>> {
    GitManager::global()
        .create_signature(name, email)
        .map_err(|e| anyhow::anyhow!("Failed to create signature: {}", e))
}

/// Create a standard hyprstream signature
pub fn create_hyprstream_signature() -> Result<Signature<'static>> {
    create_signature(None, None)
}
