//! Branch management with UUID-based branches and human-friendly tags
//!
//! This module provides a unified approach to branch management where:
//! - Branches use UUIDs for stability and uniqueness
//! - Git tags provide human-friendly names
//! - Branch descriptions store metadata

use anyhow::{Result, Context, bail};
use git2::{Repository, BranchType, Oid};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;
use git2db::{GitManager, Git2DBConfig as GitConfig};
use super::{GitOperations, get_repository};

/// Information about a branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    pub uuid: Uuid,
    pub branch_name: String,
    pub branch_type: String,
    pub human_name: Option<String>,
    pub created_at: i64,
    pub base_commit: String,
}

/// Manages branches with UUID naming and human tags
pub struct BranchManager {
    repo_path: PathBuf,
}

impl BranchManager {
    /// Create a new branch manager for a repository
    pub fn new(repo_path: impl AsRef<Path>) -> Result<Self> {
        let repo_path = repo_path.as_ref().to_path_buf();

        // Verify repository exists using global GitManager
        let _repo = GitManager::global().get_repository(&repo_path)
            .context("Failed to open repository")?;

        Ok(Self { repo_path })
    }

    /// Create with custom Git configuration (deprecated - use GitManager::global())
    #[deprecated(note = "Git configuration is now global via GitManager::global()")]
    pub fn new_with_config(repo_path: impl AsRef<Path>, _git_config: GitConfig) -> Result<Self> {
        Self::new(repo_path)
    }

    /// Get repository handle with caching
    fn get_repo(&self) -> Result<Repository> {
        let cache = GitManager::global().get_repository(&self.repo_path)
            .map_err(|e| anyhow::anyhow!("Failed to get repository: {}", e))?;
        cache.open()
            .map_err(|e| anyhow::anyhow!("Failed to open repository: {}", e))
    }
    
    /// Create a new UUID-based branch
    pub fn create_branch(
        &self,
        branch_type: &str,
        human_name: Option<&str>,
        base_commit: Option<&str>,
    ) -> Result<BranchInfo> {
        // Generate UUID for branch
        let uuid = Uuid::new_v4();
        let branch_name = format!("{}/{}", branch_type, uuid);
        
        let repo = self.get_repo()?;

        // Get base commit (default to HEAD)
        let commit = if let Some(commit_ref) = base_commit {
            if let Ok(oid) = Oid::from_str(commit_ref) {
                repo.find_commit(oid)?
            } else {
                // Try as reference (branch/tag name)
                {
                    let annotated = repo.reference_to_annotated_commit(
                        &repo.find_reference(commit_ref)?
                    )?;
                    repo.find_commit(annotated.id())?
                }
            }
        } else {
            repo.head()?.peel_to_commit()?
        };

        // Create branch using GitOperations trait
        repo.create_branch(&branch_name, Some(&commit.id().to_string()))?;
        
        // Create tag for human name if provided
        if let Some(name) = human_name {
            self.create_tag(name, &branch_name)?;
            
            // Store metadata in branch config
            let config_key = format!("branch.{}.description",
                branch_name.replace('/', "."));
            repo.config()?.set_str(&config_key, name)?;
        }
        
        let info = BranchInfo {
            uuid,
            branch_name: branch_name.clone(),
            branch_type: branch_type.to_string(),
            human_name: human_name.map(String::from),
            created_at: chrono::Utc::now().timestamp(),
            base_commit: commit.id().to_string(),
        };
        
        // Store extended metadata in refs/notes/branches
        self.store_branch_metadata(&branch_name, &info)?;
        
        Ok(info)
    }
    
    /// Create a tag pointing to a branch or commit
    pub fn create_tag(&self, tag_name: &str, target: &str) -> Result<()> {
        let safe_name = Self::sanitize_name(tag_name);
        
        // Get repository handle
        let repo = self.get_repo()?;

        // Find target object
        let target_obj = if target.contains('/') {
            // It's a branch reference
            let branch = repo.find_branch(target, BranchType::Local)
                .with_context(|| format!("Branch {} not found", target))?;
            branch.get().peel_to_commit()?.into_object()
        } else if target.len() == 40 {
            // Looks like a commit SHA
            let oid = Oid::from_str(target)?;
            repo.find_commit(oid)?.into_object()
        } else {
            // Try as tag or other reference
            repo.revparse_single(target)?
        };

        // Create lightweight tag
        repo.tag_lightweight(&safe_name, &target_obj, true)?;
        
        tracing::info!("Created tag '{}' -> {}", safe_name, target);
        
        Ok(())
    }
    
    /// Update or create a user-friendly name for a branch
    pub fn set_branch_name(&self, branch: &str, name: &str) -> Result<()> {
        let repo = self.get_repo()?;

        // Verify branch exists
        let _ = repo.find_branch(branch, BranchType::Local)?;

        // Create/update tag
        self.create_tag(name, branch)?;

        // Update branch description
        let config_key = format!("branch.{}.description", branch.replace('/', "."));
        repo.config()?.set_str(&config_key, name)?;

        Ok(())
    }
    
    /// Resolve a human name to a UUID branch
    pub fn resolve_name(&self, name: &str) -> Result<String> {
        // First, check if it's already a UUID branch
        if name.contains('/') && Uuid::parse_str(name.split('/').last().unwrap_or("")).is_ok() {
            return Ok(name.to_string());
        }
        
        // Try to find tag
        let repo = self.get_repo()?;
        let tag_ref = format!("refs/tags/{}", Self::sanitize_name(name));
        if let Ok(reference) = repo.find_reference(&tag_ref) {
            // Get the branch that the tag points to
            if let Ok(target) = reference.peel_to_commit() {
                // Find which branch contains this commit
                for branch_result in repo.branches(Some(BranchType::Local))? {
                    let (branch, _) = branch_result?;
                    if let Ok(branch_commit) = branch.get().peel_to_commit() {
                        if branch_commit.id() == target.id() {
                            if let Some(name) = branch.name()? {
                                return Ok(name.to_string());
                            }
                        }
                    }
                }
            }
        }

        // Search branch descriptions
        for branch_result in repo.branches(Some(BranchType::Local))? {
            let (branch, _) = branch_result?;
            if let Some(branch_name) = branch.name()? {
                let config_key = format!("branch.{}.description", 
                    branch_name.replace('/', "."));
                if let Ok(desc) = repo.config()?.get_string(&config_key) {
                    if desc == name {
                        return Ok(branch_name.to_string());
                    }
                }
            }
        }
        
        bail!("Could not resolve '{}' to a branch", name)
    }
    
    /// List all branches with their metadata
    pub fn list_branches(&self) -> Result<Vec<BranchInfo>> {
        let repo = self.get_repo()?;
        let mut branches = Vec::new();

        for branch_result in repo.branches(Some(BranchType::Local))? {
            let (branch, _) = branch_result?;
            let branch_name = branch.name()?.unwrap_or("").to_string();
            
            // Skip non-UUID branches (like main)
            if !branch_name.contains('/') {
                continue;
            }
            
            // Try to load stored metadata
            if let Ok(info) = self.load_branch_metadata(&branch_name) {
                branches.push(info);
            } else {
                // Construct from available info
                let parts: Vec<&str> = branch_name.split('/').collect();
                if parts.len() == 2 {
                    if let Ok(uuid) = Uuid::parse_str(parts[1]) {
                        // Get human name from description
                        let config_key = format!("branch.{}.description", 
                            branch_name.replace('/', "."));
                        let human_name = repo.config()
                            .ok()
                            .and_then(|c| c.get_string(&config_key).ok());
                        
                        branches.push(BranchInfo {
                            uuid,
                            branch_name: branch_name.clone(),
                            branch_type: parts[0].to_string(),
                            human_name,
                            created_at: 0,
                            base_commit: branch.get().peel_to_commit()
                                .map(|c| c.id().to_string())
                                .unwrap_or_default(),
                        });
                    }
                }
            }
        }
        
        branches.sort_by_key(|b| b.created_at);
        Ok(branches)
    }
    
    /// Delete a branch and its associated tags
    pub fn delete_branch(&mut self, identifier: &str) -> Result<()> {
        let repo = self.get_repo()?;
        let branch_name = self.resolve_name(identifier)?;

        // Get branch info before deletion
        let info = self.load_branch_metadata(&branch_name).ok();

        // Delete the branch
        let mut branch = repo.find_branch(&branch_name, BranchType::Local)?;
        branch.delete()?;
        
        // Delete associated tags
        if let Some(info) = info {
            if let Some(human_name) = info.human_name {
                let tag_name = Self::sanitize_name(&human_name);
                let _ = repo.tag_delete(&tag_name);
            }
        }
        
        // Clean up metadata
        self.delete_branch_metadata(&branch_name)?;
        
        Ok(())
    }
    
    /// Create a worktree for a branch
    pub fn create_worktree(
        &self,
        branch: &str,
        worktree_path: Option<PathBuf>,
    ) -> Result<PathBuf> {
        let branch_name = self.resolve_name(branch)?;
        
        // Generate worktree path if not provided
        let wt_path = worktree_path.unwrap_or_else(|| {
            let wt_uuid = Uuid::new_v4();
            self.repo_path.parent()
                .unwrap_or(Path::new("."))
                .join("working")
                .join(wt_uuid.to_string())
        });
        
        // Ensure parent directory exists
        if let Some(parent) = wt_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Create worktree
        let wt_name = wt_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("worktree");
        
        let repo = self.get_repo()?;
        let options = git2::WorktreeAddOptions::new();
        repo.worktree(
            wt_name,
            &wt_path,
            Some(&options),
        )?;
        
        // Checkout the branch in the worktree
        let wt_repo = get_repository(&wt_path)?;
        wt_repo.set_head(&format!("refs/heads/{}", branch_name))?;
        wt_repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
        
        Ok(wt_path)
    }
    
    /// Sanitize a name for use as a Git tag
    fn sanitize_name(name: &str) -> String {
        name.chars()
            .map(|c| match c {
                ' ' | '\t' | '\n' => '-',
                '/' | '\\' | ':' | '?' | '*' | '"' | '<' | '>' | '|' | '[' | ']' => '_',
                _ if c.is_control() => '_',
                _ => c,
            })
            .collect::<String>()
            .trim_matches(|c| c == '-' || c == '_' || c == '.')
            .to_lowercase()
    }
    
    /// Store branch metadata in Git notes
    fn store_branch_metadata(&self, branch_name: &str, info: &BranchInfo) -> Result<()> {
        let json = serde_json::to_string(info)?;
        let sig = GitManager::global().create_signature(Some("branch-manager"), Some("branch@hyprstream"))?;
        let repo = self.get_repo()?;

        // Store as Git note
        let branch = repo.find_branch(branch_name, BranchType::Local)?;
        let commit = branch.get().peel_to_commit()?;

        repo.note(
            &sig,
            &sig,
            Some("refs/notes/branches"),
            commit.id(),
            &json,
            true,
        )?;

        Ok(())
    }

    /// Load branch metadata from Git notes
    fn load_branch_metadata(&self, branch_name: &str) -> Result<BranchInfo> {
        let repo = self.get_repo()?;
        let branch = repo.find_branch(branch_name, BranchType::Local)?;
        let commit = branch.get().peel_to_commit()?;

        let note = repo.find_note(Some("refs/notes/branches"), commit.id())?;
        let json = note.message()
            .ok_or_else(|| anyhow::anyhow!("Empty note"))?;

        Ok(serde_json::from_str(json)?)
    }

    /// Delete branch metadata
    fn delete_branch_metadata(&self, branch_name: &str) -> Result<()> {
        let repo = self.get_repo()?;
        if let Ok(branch) = repo.find_branch(branch_name, BranchType::Local) {
            if let Ok(commit) = branch.get().peel_to_commit() {
                let sig = GitManager::global().create_signature(Some("branch-manager"), Some("branch@hyprstream"))?;
                let _ = repo.note_delete(
                    commit.id(),
                    Some("refs/notes/branches"),
                    &sig,
                    &sig,
                );
            }
        }
        Ok(())
    }
}