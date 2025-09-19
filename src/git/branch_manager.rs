//! Branch management with UUID-based branches and human-friendly tags
//!
//! This module provides a unified approach to branch management where:
//! - Branches use UUIDs for stability and uniqueness
//! - Git tags provide human-friendly names
//! - Branch descriptions store metadata

use anyhow::{Result, Context, bail};
use git2::{Repository, BranchType, Oid, Signature};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

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
    repo: Repository,
    repo_path: PathBuf,
}

impl BranchManager {
    /// Create a new branch manager for a repository
    pub fn new(repo_path: impl AsRef<Path>) -> Result<Self> {
        let repo_path = repo_path.as_ref().to_path_buf();
        let repo = Repository::open(&repo_path)
            .context("Failed to open repository")?;
        
        Ok(Self { repo, repo_path })
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
        
        // Get base commit (default to HEAD)
        let commit = if let Some(commit_ref) = base_commit {
            if let Ok(oid) = Oid::from_str(commit_ref) {
                self.repo.find_commit(oid)?
            } else {
                // Try as reference (branch/tag name)
                {
                    let annotated = self.repo.reference_to_annotated_commit(
                        &self.repo.find_reference(commit_ref)?
                    )?;
                    self.repo.find_commit(annotated.id())?
                }
            }
        } else {
            self.repo.head()?.peel_to_commit()?
        };
        
        // Create branch
        self.repo.branch(&branch_name, &commit, false)
            .context("Failed to create branch")?;
        
        // Create tag for human name if provided
        if let Some(name) = human_name {
            self.create_tag(name, &branch_name)?;
            
            // Store metadata in branch config
            let config_key = format!("branch.{}.description", 
                branch_name.replace('/', "."));
            self.repo.config()?.set_str(&config_key, name)?;
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
        
        // Find target object
        let target_obj = if target.contains('/') {
            // It's a branch reference
            let branch = self.repo.find_branch(target, BranchType::Local)
                .with_context(|| format!("Branch {} not found", target))?;
            branch.get().peel_to_commit()?.into_object()
        } else if target.len() == 40 {
            // Looks like a commit SHA
            let oid = Oid::from_str(target)?;
            self.repo.find_commit(oid)?.into_object()
        } else {
            // Try as tag or other reference
            self.repo.revparse_single(target)?
        };
        
        // Create lightweight tag
        self.repo.tag_lightweight(&safe_name, &target_obj, true)?;
        
        tracing::info!("Created tag '{}' -> {}", safe_name, target);
        
        Ok(())
    }
    
    /// Update or create a user-friendly name for a branch
    pub fn set_branch_name(&self, branch: &str, name: &str) -> Result<()> {
        // Verify branch exists
        let _ = self.repo.find_branch(branch, BranchType::Local)?;
        
        // Create/update tag
        self.create_tag(name, branch)?;
        
        // Update branch description
        let config_key = format!("branch.{}.description", branch.replace('/', "."));
        self.repo.config()?.set_str(&config_key, name)?;
        
        Ok(())
    }
    
    /// Resolve a human name to a UUID branch
    pub fn resolve_name(&self, name: &str) -> Result<String> {
        // First, check if it's already a UUID branch
        if name.contains('/') && Uuid::parse_str(name.split('/').last().unwrap_or("")).is_ok() {
            return Ok(name.to_string());
        }
        
        // Try to find tag
        let tag_ref = format!("refs/tags/{}", Self::sanitize_name(name));
        if let Ok(reference) = self.repo.find_reference(&tag_ref) {
            // Get the branch that the tag points to
            if let Ok(target) = reference.peel_to_commit() {
                // Find which branch contains this commit
                for branch_result in self.repo.branches(Some(BranchType::Local))? {
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
        for branch_result in self.repo.branches(Some(BranchType::Local))? {
            let (branch, _) = branch_result?;
            if let Some(branch_name) = branch.name()? {
                let config_key = format!("branch.{}.description", 
                    branch_name.replace('/', "."));
                if let Ok(desc) = self.repo.config()?.get_string(&config_key) {
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
        let mut branches = Vec::new();
        
        for branch_result in self.repo.branches(Some(BranchType::Local))? {
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
                        let human_name = self.repo.config()
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
        let branch_name = self.resolve_name(identifier)?;
        
        // Get branch info before deletion
        let info = self.load_branch_metadata(&branch_name).ok();
        
        // Delete the branch
        let mut branch = self.repo.find_branch(&branch_name, BranchType::Local)?;
        branch.delete()?;
        
        // Delete associated tags
        if let Some(info) = info {
            if let Some(human_name) = info.human_name {
                let tag_name = Self::sanitize_name(&human_name);
                let _ = self.repo.tag_delete(&tag_name);
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
        
        let options = git2::WorktreeAddOptions::new();
        self.repo.worktree(
            wt_name,
            &wt_path,
            Some(&options),
        )?;
        
        // Checkout the branch in the worktree
        let wt_repo = Repository::open(&wt_path)?;
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
        let sig = Signature::now("branch-manager", "branch@hyprstream")?;
        
        // Store as Git note
        let branch = self.repo.find_branch(branch_name, BranchType::Local)?;
        let commit = branch.get().peel_to_commit()?;
        
        self.repo.note(
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
        let branch = self.repo.find_branch(branch_name, BranchType::Local)?;
        let commit = branch.get().peel_to_commit()?;
        
        let note = self.repo.find_note(Some("refs/notes/branches"), commit.id())?;
        let json = note.message()
            .ok_or_else(|| anyhow::anyhow!("Empty note"))?;
        
        Ok(serde_json::from_str(json)?)
    }
    
    /// Delete branch metadata
    fn delete_branch_metadata(&self, branch_name: &str) -> Result<()> {
        if let Ok(branch) = self.repo.find_branch(branch_name, BranchType::Local) {
            if let Ok(commit) = branch.get().peel_to_commit() {
                let sig = Signature::now("branch-manager", "branch@hyprstream")?;
                let _ = self.repo.note_delete(
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