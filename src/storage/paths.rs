//! Storage path management using XDG Base Directory specification
//!
//! This module provides standardized paths for storing models, LoRAs, and other data
//! following the XDG Base Directory specification.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use xdg::BaseDirectories;

/// Application name for XDG directories
const APP_NAME: &str = "hyprstream";

/// Storage manager for organizing models, LoRAs, and other data
pub struct StoragePaths {
    base_dirs: BaseDirectories,
}

impl StoragePaths {
    /// Create a new storage paths manager
    pub fn new() -> Result<Self> {
        let base_dirs = BaseDirectories::with_prefix(APP_NAME)
            .map_err(|e| anyhow!("Failed to create XDG base directories: {}", e))?;
        
        Ok(Self { base_dirs })
    }

    /// Get the models directory path
    pub fn models_dir(&self) -> Result<PathBuf> {
        let models_dir = self.base_dirs.get_data_home().join("models");
        self.ensure_dir_exists(&models_dir)?;
        Ok(models_dir)
    }

    /// Get the LoRAs directory path
    pub fn loras_dir(&self) -> Result<PathBuf> {
        let loras_dir = self.base_dirs.get_data_home().join("loras");
        self.ensure_dir_exists(&loras_dir)?;
        Ok(loras_dir)
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> Result<PathBuf> {
        let cache_dir = self.base_dirs.get_cache_home();
        self.ensure_dir_exists(&cache_dir)?;
        Ok(cache_dir)
    }

    /// Get the config directory path
    pub fn config_dir(&self) -> Result<PathBuf> {
        let config_dir = self.base_dirs.get_config_home();
        self.ensure_dir_exists(&config_dir)?;
        Ok(config_dir)
    }

    /// Get the path for HuggingFace authentication token
    pub fn hf_token_path(&self) -> Result<PathBuf> {
        Ok(self.config_dir()?.join("hf_token"))
    }

    /// Get a specific model path by name
    pub fn model_path(&self, model_name: &str) -> Result<PathBuf> {
        let sanitized_name = sanitize_filename(model_name);
        Ok(self.models_dir()?.join(sanitized_name))
    }

    /// Get a specific LoRA path by name
    pub fn lora_path(&self, lora_name: &str) -> Result<PathBuf> {
        let sanitized_name = sanitize_filename(lora_name);
        Ok(self.loras_dir()?.join(sanitized_name))
    }

    /// Get the temporary download directory
    pub fn temp_download_dir(&self) -> Result<PathBuf> {
        let temp_dir = self.cache_dir()?.join("downloads");
        self.ensure_dir_exists(&temp_dir)?;
        Ok(temp_dir)
    }

    /// List all models in the models directory
    pub fn list_local_models(&self) -> Result<Vec<String>> {
        let models_dir = self.models_dir()?;
        let mut models = Vec::new();

        if models_dir.exists() {
            for entry in std::fs::read_dir(&models_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    if let Some(name) = entry.file_name().to_str() {
                        models.push(name.to_string());
                    }
                }
            }
        }

        models.sort();
        Ok(models)
    }

    /// List all LoRAs in the LoRAs directory
    pub fn list_local_loras(&self) -> Result<Vec<String>> {
        let loras_dir = self.loras_dir()?;
        let mut loras = Vec::new();

        if loras_dir.exists() {
            for entry in std::fs::read_dir(&loras_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    if let Some(name) = entry.file_name().to_str() {
                        loras.push(name.to_string());
                    }
                }
            }
        }

        loras.sort();
        Ok(loras)
    }

    /// Ensure a directory exists, creating it if necessary
    fn ensure_dir_exists(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            std::fs::create_dir_all(path)
                .map_err(|e| anyhow!("Failed to create directory {}: {}", path.display(), e))?;
        }
        Ok(())
    }
}

impl Default for StoragePaths {
    fn default() -> Self {
        Self::new().expect("Failed to initialize storage paths")
    }
}

/// Sanitize a filename to be safe for filesystem storage
fn sanitize_filename(name: &str) -> String {
    name.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_")
        .replace(' ', "_")
}

/// HuggingFace authentication management
pub struct HfAuth {
    storage: StoragePaths,
}

impl HfAuth {
    /// Create a new HuggingFace authentication manager
    pub fn new() -> Result<Self> {
        Ok(Self {
            storage: StoragePaths::new()?,
        })
    }

    /// Set the HuggingFace authentication token
    pub async fn set_token(&self, token: &str) -> Result<()> {
        let token_path = self.storage.hf_token_path()?;
        tokio::fs::write(&token_path, token.trim()).await
            .map_err(|e| anyhow!("Failed to write HF token to {}: {}", token_path.display(), e))?;
        
        // Set restrictive permissions (readable only by owner)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&token_path).await?.permissions();
            perms.set_mode(0o600); // rw-------
            tokio::fs::set_permissions(&token_path, perms).await?;
        }

        println!("✅ HuggingFace token saved to: {}", token_path.display());
        Ok(())
    }

    /// Get the HuggingFace authentication token
    pub async fn get_token(&self) -> Result<Option<String>> {
        let token_path = self.storage.hf_token_path()?;
        
        if !token_path.exists() {
            return Ok(None);
        }

        let token = tokio::fs::read_to_string(&token_path).await
            .map_err(|e| anyhow!("Failed to read HF token from {}: {}", token_path.display(), e))?;
        
        Ok(Some(token.trim().to_string()))
    }

    /// Check if authentication token is configured
    pub async fn is_authenticated(&self) -> bool {
        self.get_token().await.map_or(false, |token| token.is_some())
    }

    /// Remove the authentication token
    pub async fn logout(&self) -> Result<()> {
        let token_path = self.storage.hf_token_path()?;
        
        if token_path.exists() {
            tokio::fs::remove_file(&token_path).await
                .map_err(|e| anyhow!("Failed to remove HF token: {}", e))?;
            println!("✅ Logged out from HuggingFace");
        } else {
            println!("ℹ️  No HuggingFace token found");
        }
        
        Ok(())
    }
}

impl Default for HfAuth {
    fn default() -> Self {
        Self::new().expect("Failed to initialize HF auth")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_storage_paths() {
        let storage = StoragePaths::new().unwrap();
        
        // Test directory creation
        let models_dir = storage.models_dir().unwrap();
        let loras_dir = storage.loras_dir().unwrap();
        
        assert!(models_dir.exists());
        assert!(loras_dir.exists());
        assert!(models_dir != loras_dir);
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("hello/world"), "hello_world");
        assert_eq!(sanitize_filename("model:v1.0"), "model_v1.0");
        assert_eq!(sanitize_filename("test file.gguf"), "test_file.gguf");
    }
}