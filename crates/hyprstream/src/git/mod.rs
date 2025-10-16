//! Git operations for model version control
//!
//! This module contains hyprstream-specific git extensions.
//! Core git functionality is provided by git2db.

pub mod helpers;

// Re-export commonly used git2db types
pub use git2db::{Git2DB as GitModelRegistry, TrackedRepository};

// Compatibility alias for internal use
pub type RegisteredModel = TrackedRepository;

// Hyprstream-specific types and traits
use anyhow::Result;
use git2::Signature;
use git2db::{Git2DBError as GitError, GitManager, Repository};
use std::path::Path;
use std::sync::Arc;

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

/// Get a repository handle using the global GitManager
pub fn get_repository<P: AsRef<Path>>(path: P) -> Result<Repository> {
    let cache = GitManager::global()
        .get_repository(path)
        .map_err(|e| anyhow::anyhow!("Failed to get repository: {}", e))?;
    cache
        .open()
        .map_err(|e| anyhow::anyhow!("Failed to open repository: {}", e))
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
