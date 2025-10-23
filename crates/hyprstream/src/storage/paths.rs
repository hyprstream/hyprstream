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

#[cfg(test)]
mod tests {
    use super::*;

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
}
