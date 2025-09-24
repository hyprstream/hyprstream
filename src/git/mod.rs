//! Git operations for model version control using git2

pub mod registry;
pub mod branch_manager;
pub mod manager;

pub use registry::{GitModelRegistry, ShareableModelRef, ModelType as RegistryModelType};
pub use branch_manager::{BranchManager, BranchInfo};
pub use manager::{GitManager, GitConfig, GitProgress, GitProgressInfo, GitError, CloneOptions, GitOperations};

// Convenience functions for global GitManager access
use anyhow::Result;
use std::path::Path;
use git2::{Repository, Signature};

/// Get a repository handle using the global GitManager
pub fn get_repository<P: AsRef<Path>>(path: P) -> Result<Repository> {
    GitManager::global().get_repository(path)
}

/// Clone a repository using the global GitManager
pub async fn clone_repository<P: AsRef<Path>>(
    url: &str,
    path: P,
    options: Option<CloneOptions>,
    progress: Option<std::sync::Arc<dyn GitProgress>>,
) -> Result<Repository> {
    let options = options.unwrap_or_default();
    GitManager::global().clone_repository(url, path, options, progress).await
}

/// Create a signature using the global GitManager
pub fn create_signature(name: Option<&str>, email: Option<&str>) -> Result<Signature<'static>> {
    GitManager::global().create_signature(name, email)
}

/// Create a standard hyprstream signature
pub fn create_hyprstream_signature() -> Result<Signature<'static>> {
    GitManager::global().create_signature(None, None)
}