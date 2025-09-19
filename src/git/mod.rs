//! Git operations for model version control using git2

pub mod registry;
pub mod branch_manager;

pub use registry::{GitModelRegistry, ShareableModelRef, ModelType as RegistryModelType};
pub use branch_manager::{BranchManager, BranchInfo};

use anyhow::{Result, Context, bail};
use git2::{Repository, Signature};
use safe_path::scoped_join;
use std::path::{Path, PathBuf};

/// Git operations for model repositories
pub struct GitOps {
    base_dir: PathBuf,
}

impl GitOps {
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }
    
    /// Safely join and validate a path within base_dir
    fn safe_path(&self, path: &Path) -> Result<PathBuf> {
        // Use safe_path crate to prevent traversal
        scoped_join(&self.base_dir, path)
            .map_err(|e| anyhow::anyhow!("Path traversal attempt detected: {}", e))
    }
    
    /// Open a repository with path validation
    pub fn open(&self, path: &Path) -> Result<Repository> {
        let safe_path = if path.is_absolute() {
            // If absolute, verify it's within base_dir
            if !path.starts_with(&self.base_dir) {
                bail!("Repository path outside model storage");
            }
            path.to_path_buf()
        } else {
            // If relative, safely join with base_dir
            self.safe_path(path)?
        };
        
        Repository::open(&safe_path)
            .context("Failed to open repository")
    }
    
    /// Create a worktree for an adapter using git2
    pub fn create_worktree(
        &self,
        base_repo_path: &Path,
        worktree_path: &Path,
        branch_name: &str,
    ) -> Result<()> {
        // Validate paths
        let base_canonical = if base_repo_path.is_absolute() {
            if !base_repo_path.starts_with(&self.base_dir) {
                bail!("Base repository path outside model storage");
            }
            base_repo_path.to_path_buf()
        } else {
            self.safe_path(base_repo_path)?
        };
        
        let worktree_canonical = if worktree_path.is_absolute() {
            if !worktree_path.starts_with(&self.base_dir) {
                bail!("Worktree path outside model storage");
            }
            worktree_path.to_path_buf()
        } else {
            self.safe_path(worktree_path)?
        };
        
        // Open the base repository
        let repo = Repository::open(&base_canonical)?;
        
        // Create a new branch for the worktree
        let head = repo.head()?;
        let commit = head.peel_to_commit()?;
        repo.branch(branch_name, &commit, false)?;
        
        // Add the worktree
        let options = git2::WorktreeAddOptions::new();
        repo.worktree(
            branch_name,
            &worktree_canonical,
            Some(&options),
        )?;
        
        // Checkout the branch in the worktree
        let worktree_repo = Repository::open(&worktree_canonical)?;
        let branch_ref = format!("refs/heads/{}", branch_name);
        worktree_repo.set_head(&branch_ref)?;
        worktree_repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
        
        Ok(())
    }
    
    /// Remove a worktree
    pub fn remove_worktree(
        &self,
        base_repo_path: &Path,
        worktree_name: &str,
    ) -> Result<()> {
        let base_canonical = if base_repo_path.is_absolute() {
            if !base_repo_path.starts_with(&self.base_dir) {
                bail!("Base repository path outside model storage");
            }
            base_repo_path.to_path_buf()
        } else {
            self.safe_path(base_repo_path)?
        };
        
        // Open the repository
        let repo = Repository::open(&base_canonical)?;
        
        // Find and prune the worktree
        if let Ok(worktree) = repo.find_worktree(worktree_name) {
            worktree.prune(Some(git2::WorktreePruneOptions::new().working_tree(true)))?;
        }
        
        Ok(())
    }
    
    /// Commit changes in a repository
    pub fn commit(&self, repo: &Repository, message: &str) -> Result<()> {
        let mut index = repo.index()?;
        index.add_all(["."].iter(), git2::IndexAddOption::DEFAULT, None)?;
        index.write()?;
        
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        
        let sig = Signature::now("hyprstream", "hyprstream@local")?;
        
        // Get HEAD commit as parent
        let parent = repo.head()?.peel_to_commit()?;
        
        repo.commit(
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