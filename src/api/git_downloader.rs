//! Git-based model downloader using git2 crate

use anyhow::{Result, bail};
use git2::{Repository, FetchOptions, build::RepoBuilder};
use std::path::{Path, PathBuf};
use crate::api::model_storage::ModelId;

/// Git-based model source using git2
pub struct GitModelSource {
    cache_dir: PathBuf,
}

impl GitModelSource {
    /// Create a new Git model source
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }
    
    /// Clone a model repository using git2
    pub fn clone_model(
        &self,
        repo_url: &str,
    ) -> Result<(ModelId, PathBuf)> {
        let git_url = repo_url;

        // Always generate new model ID for clones
        let model_id = ModelId::new();
        let model_path = self.cache_dir.join(model_id.0.to_string());
        
        // Check if already cloned
        if model_path.join(".git").exists() {
            println!("📦 Model already cloned at: {}", model_path.display());
            return Ok((model_id, model_path));
        }
        
        println!("📥 Cloning model from: {}", git_url);
        
        // Clone with git2
        let mut builder = RepoBuilder::new();
        
        // Configure fetch options to use system git config
        let mut fetch_opts = FetchOptions::new();
        
        // Use git2's credential callback which respects system config
        fetch_opts.remote_callbacks(Self::get_callbacks());
        builder.fetch_options(fetch_opts);
        
        // Clone the repository
        match builder.clone(&git_url, &model_path) {
            Ok(_) => {
                println!("✅ Model cloned successfully");
                Ok((model_id, model_path))
            }
            Err(e) => {
                // Provide helpful error messages
                if e.message().contains("authentication") || e.message().contains("Authentication") {
                    bail!("Git authentication failed. Please configure authentication:\n\
                           \n\
                           For SSH URLs (git@...):\n\
                           • Use ssh-add to add your SSH key to the agent\n\
                           • Or ensure ~/.ssh/id_rsa or ~/.ssh/id_ed25519 exists\n\
                           \n\
                           For HTTPS URLs:\n\
                           • Configure git credential helper: git config --global credential.helper store\n\
                           • For HuggingFace: export HF_TOKEN=your_token\n\
                           • For GitHub: export GITHUB_TOKEN=your_token\n\
                           \n\
                           Error: {}", e);
                } else if e.message().contains("not found") || e.message().contains("repository") {
                    bail!("Repository not found or you don't have access: {}", git_url);
                } else {
                    bail!("Git clone failed: {}", e);
                }
            }
        }
    }
    
    /// Clone with specific branch or tag
    pub fn clone_ref(
        &self,
        repo_url: &str,
        git_ref: &str,
    ) -> Result<(ModelId, PathBuf)> {
        let git_url = repo_url;

        // Always generate new model ID
        let model_id = ModelId::new();
        let model_path = self.cache_dir.join(model_id.0.to_string());
        
        if model_path.join(".git").exists() {
            println!("📦 Model already cloned, checking out ref: {}", git_ref);
            
            // Open existing repo and checkout ref
            let repo = Repository::open(&model_path)?;
            
            // Fetch the ref first
            let mut remote = repo.find_remote("origin")?;
            remote.fetch(&[git_ref], Some(&mut FetchOptions::new()), None)?;
            
            // Checkout the ref
            let reference = repo.find_reference(&format!("refs/remotes/origin/{}", git_ref))
                .or_else(|_| repo.find_reference(&format!("refs/tags/{}", git_ref)))
                .or_else(|_| repo.find_reference(git_ref))?;
            
            let commit = reference.peel_to_commit()?;
            repo.checkout_tree(commit.as_object(), None)?;
            repo.set_head_detached(commit.id())?;
            
            return Ok((model_id, model_path));
        }
        
        println!("🎯 Cloning specific ref: {}", git_ref);
        
        let mut builder = RepoBuilder::new();
        builder.branch(git_ref);
        
        // Configure fetch options
        let mut fetch_opts = FetchOptions::new();
        fetch_opts.remote_callbacks(Self::get_callbacks());
        builder.fetch_options(fetch_opts);
        
        match builder.clone(&git_url, &model_path) {
            Ok(_) => {
                println!("✅ Model cloned successfully at ref: {}", git_ref);
                Ok((model_id, model_path))
            }
            Err(e) => {
                bail!("Git clone of ref {} failed: {}", git_ref, e);
            }
        }
    }
    
    /// Update a cloned model repository
    pub fn update_model(&self, model_path: &Path) -> Result<()> {
        println!("🔄 Updating model repository...");
        
        let repo = Repository::open(model_path)?;
        
        // Fetch from origin
        let mut remote = repo.find_remote("origin")?;
        let mut fetch_opts = FetchOptions::new();
        fetch_opts.remote_callbacks(Self::get_callbacks());
        
        remote.fetch(&["refs/heads/*:refs/remotes/origin/*"], Some(&mut fetch_opts), None)?;
        
        // Fast-forward merge if possible
        let fetch_head = repo.find_reference("FETCH_HEAD")?;
        let fetch_commit = fetch_head.peel_to_commit()?;
        let annotated_commit = repo.find_annotated_commit(fetch_commit.id())?;
        let analysis = repo.merge_analysis(&[&annotated_commit])?;
        
        if analysis.0.is_fast_forward() {
            // Fast-forward
            let mut reference = repo.head()?;
            reference.set_target(fetch_commit.id(), "Fast-forward")?;
            repo.set_head(reference.name().unwrap())?;
            repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
            println!("✅ Model updated successfully");
        } else if analysis.0.is_up_to_date() {
            println!("✅ Model is already up to date");
        } else {
            bail!("Cannot fast-forward merge. Manual intervention required.");
        }
        
        Ok(())
    }
    
    /// Get git2 callbacks that respect system configuration
    fn get_callbacks() -> git2::RemoteCallbacks<'static> {
        let mut callbacks = git2::RemoteCallbacks::new();
        
        // Try to use SSH agent for SSH URLs
        callbacks.credentials(|_url, username_from_url, allowed_types| {
            // First try SSH key from agent
            if allowed_types.contains(git2::CredentialType::SSH_KEY) {
                match git2::Cred::ssh_key_from_agent(username_from_url.unwrap_or("git")) {
                    Ok(cred) => return Ok(cred),
                    Err(_) => {} // Continue to next method
                }
            }

            // Try default SSH key locations
            if allowed_types.contains(git2::CredentialType::SSH_KEY) {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                let ssh_dir = PathBuf::from(home).join(".ssh");

                match git2::Cred::ssh_key(
                    username_from_url.unwrap_or("git"),
                    None,
                    &ssh_dir.join("id_rsa"),
                    None,
                ) {
                    Ok(cred) => return Ok(cred),
                    Err(_) => {} // Continue to next method
                }

                match git2::Cred::ssh_key(
                    username_from_url.unwrap_or("git"),
                    None,
                    &ssh_dir.join("id_ed25519"),
                    None,
                ) {
                    Ok(cred) => return Ok(cred),
                    Err(_) => {} // Continue to next method
                }
            }

            // Try HF_TOKEN for HuggingFace repos
            if allowed_types.contains(git2::CredentialType::USER_PASS_PLAINTEXT) {
                if let Ok(token) = std::env::var("HF_TOKEN") {
                    match git2::Cred::userpass_plaintext("token", &token) {
                        Ok(cred) => return Ok(cred),
                        Err(_) => {} // Continue to next method
                    }
                }
                if let Ok(token) = std::env::var("GITHUB_TOKEN") {
                    match git2::Cred::userpass_plaintext("token", &token) {
                        Ok(cred) => return Ok(cred),
                        Err(_) => {} // Continue to next method
                    }
                }
            }

            // Fall back to git credential helper
            if allowed_types.contains(git2::CredentialType::DEFAULT) {
                match git2::Cred::default() {
                    Ok(cred) => return Ok(cred),
                    Err(_) => {} // Continue to final fallback
                }
            }

            // Final fallback - try default again or return error
            match git2::Cred::default() {
                Ok(cred) => Ok(cred),
                Err(e) => Err(e),
            }
        });
        
        callbacks
    }
}