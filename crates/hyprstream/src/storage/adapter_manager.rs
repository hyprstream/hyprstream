//! Adapter management for LoRA models
//!
//! This module handles the storage, discovery, and composition of LoRA adapters
//! within the git-versioned model repositories.
//!
//! Two modes of operation:
//! - **Direct** (sync): Uses `PathBuf` and `std::fs` for standalone/test use.
//! - **FsOps** (async): Uses `WorktreeClient` for worktree-scoped, path-contained access.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::services::WorktreeClient;

/// Information about a discovered adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterInfo {
    /// Full filename (e.g., "00_base.safetensors")
    pub filename: String,
    /// Numeric index parsed from filename
    pub index: u32,
    /// Base name without index (e.g., "base")
    pub name: String,
    /// Full path to the adapter file
    pub path: PathBuf,
    /// Size in bytes
    pub size: u64,
    /// Associated config file if exists
    pub config_path: Option<PathBuf>,
}

/// Adapter configuration stored alongside the weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub rank: u32,
    pub alpha: f32,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub created_at: String,
    pub model_ref: String,
    pub training_data: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            learning_rate: 1e-4,
            batch_size: 4,
            epochs: 10,
            created_at: chrono::Utc::now().to_rfc3339(),
            model_ref: String::new(),
            training_data: None,
            metadata: HashMap::new(),
        }
    }
}

/// Manages LoRA adapters within a model's git repository
pub struct AdapterManager {
    pub adapters_dir: PathBuf,
    /// Optional FsOps for worktree-scoped file operations.
    /// When present, async methods use contained-root access.
    /// When absent, sync methods use direct filesystem access.
    fs: Option<WorktreeClient>,
}

impl AdapterManager {
    /// Create a new adapter manager for a model (direct filesystem access)
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        let adapters_dir = model_path.as_ref().join("adapters");
        Self {
            adapters_dir,
            fs: None,
        }
    }

    /// Create a new adapter manager with WorktreeClient for worktree-scoped access
    pub fn with_fs(fs: WorktreeClient) -> Self {
        // adapters_dir is kept for backward compat but not used when fs is Some
        Self {
            adapters_dir: PathBuf::from("adapters"),
            fs: Some(fs),
        }
    }

    /// Returns true if this manager is using FsOps for file operations
    pub fn has_fs(&self) -> bool {
        self.fs.is_some()
    }

    // =========================================================================
    // Sync methods (direct filesystem access — backward compatible)
    // =========================================================================

    /// Ensure the adapters directory exists
    pub fn ensure_adapters_dir(&self) -> Result<()> {
        if !self.adapters_dir.exists() {
            std::fs::create_dir_all(&self.adapters_dir).with_context(|| {
                format!(
                    "Failed to create adapters directory: {:?}",
                    self.adapters_dir
                )
            })?;
        }
        Ok(())
    }

    /// List all adapters in the model, sorted by index
    pub fn list_adapters(&self) -> Result<Vec<AdapterInfo>> {
        let mut adapters = Vec::new();

        if !self.adapters_dir.exists() {
            return Ok(adapters);
        }

        for entry in std::fs::read_dir(&self.adapters_dir)? {
            let entry = entry?;
            let path = entry.path();
            let filename = entry.file_name();

            if let Some(name_str) = filename.to_str() {
                // Only process .safetensors files
                if !name_str.ends_with(".safetensors") {
                    continue;
                }

                // Parse index and name from filename like "00_base.safetensors"
                let base_name = name_str.trim_end_matches(".safetensors");
                let (index, name) = if let Some(underscore_pos) = base_name.find('_') {
                    let index_str = &base_name[..underscore_pos];
                    let name_str = &base_name[underscore_pos + 1..];

                    if let Ok(idx) = index_str.parse::<u32>() {
                        (idx, name_str.to_owned())
                    } else {
                        // If parsing fails, treat whole name as adapter name with index 999
                        (999, base_name.to_owned())
                    }
                } else {
                    // No index prefix, assign high index
                    (999, base_name.to_owned())
                };

                let metadata = std::fs::metadata(&path)?;
                let config_path = self.adapters_dir.join(format!("{base_name}.config.json"));

                adapters.push(AdapterInfo {
                    filename: name_str.to_owned(),
                    index,
                    name,
                    path,
                    size: metadata.len(),
                    config_path: if config_path.exists() {
                        Some(config_path)
                    } else {
                        None
                    },
                });
            }
        }

        // Sort by index
        adapters.sort_by_key(|a| a.index);
        Ok(adapters)
    }

    /// Get the next available index for a new adapter
    pub fn get_next_index(&self) -> Result<u32> {
        let adapters = self.list_adapters()?;
        if adapters.is_empty() {
            Ok(0)
        } else {
            Ok(adapters.last().map(|a| a.index + 1).unwrap_or(0))
        }
    }

    /// Create an indexed adapter filename
    pub fn create_indexed_name(&self, name: &str, index: Option<u32>) -> Result<String> {
        let idx = if let Some(i) = index {
            i
        } else {
            self.get_next_index()?
        };
        Ok(format!("{idx:02}_{name}"))
    }

    /// Initialize adapter with proper implementation
    pub fn initialize_adapter(
        &self,
        name: &str,
        index: Option<u32>,
        config: AdapterConfig,
    ) -> Result<std::path::PathBuf> {
        self.ensure_adapters_dir()?;

        let idx = if let Some(i) = index {
            i
        } else {
            self.get_next_index()?
        };

        let adapter_name = format!("{idx:02}_{name}");
        let adapter_path = self
            .adapters_dir
            .join(format!("{adapter_name}.safetensors"));

        // Save configuration file
        let config_path = self
            .adapters_dir
            .join(format!("{adapter_name}.config.json"));
        let config_json = serde_json::to_string_pretty(&config)
            .with_context(|| "Failed to serialize adapter config")?;
        std::fs::write(&config_path, config_json)
            .with_context(|| format!("Failed to write adapter config: {config_path:?}"))?;

        // Create empty adapter file (weights will be added during training)
        std::fs::File::create(&adapter_path)
            .with_context(|| format!("Failed to create adapter file: {adapter_path:?}"))?;

        Ok(adapter_path)
    }

    /// Load all adapter paths sorted by index
    pub fn get_adapter_paths(&self) -> Result<Vec<PathBuf>> {
        let adapters = self.list_adapters()?;
        Ok(adapters.into_iter().map(|a| a.path).collect())
    }

    /// Load configuration for a specific adapter
    pub fn load_adapter_config(&self, adapter_name: &str) -> Result<AdapterConfig> {
        let config_path = self
            .adapters_dir
            .join(format!("{adapter_name}.config.json"));
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read adapter config: {config_path:?}"))?;
        let config: AdapterConfig = serde_json::from_str(&config_str)?;
        Ok(config)
    }

    /// Check if any adapters exist for this model
    pub fn has_adapters(&self) -> bool {
        self.adapters_dir.exists() && !self.list_adapters().unwrap_or_default().is_empty()
    }

    /// Get total size of all adapters
    pub fn get_total_size(&self) -> Result<u64> {
        let adapters = self.list_adapters()?;
        Ok(adapters.iter().map(|a| a.size).sum())
    }

    /// Remove an adapter by name or index
    pub fn remove_adapter(&self, identifier: &str) -> Result<()> {
        let adapters = self.list_adapters()?;

        // Try to parse as index first
        let to_remove = if let Ok(idx) = identifier.parse::<u32>() {
            adapters.iter().find(|a| a.index == idx)
        } else {
            // Otherwise match by name
            adapters
                .iter()
                .find(|a| a.name == identifier || a.filename.starts_with(identifier))
        };

        if let Some(adapter) = to_remove {
            // Remove adapter file
            if adapter.path.exists() {
                std::fs::remove_file(&adapter.path)?;
            }

            // Remove config if exists
            if let Some(config_path) = &adapter.config_path {
                if config_path.exists() {
                    std::fs::remove_file(config_path)?;
                }
            }

            Ok(())
        } else {
            anyhow::bail!("Adapter '{}' not found", identifier)
        }
    }

    // =========================================================================
    // Async methods (FsOps — worktree-scoped, path-contained access)
    // =========================================================================

    /// Ensure the adapters directory exists (async, via FsOps)
    pub async fn ensure_adapters_dir_async(&self) -> Result<()> {
        if let Some(fs) = &self.fs {
            fs.mkdir_p("adapters").await?;
            Ok(())
        } else {
            self.ensure_adapters_dir()
        }
    }

    /// List all adapters via FsOps, sorted by index
    pub async fn list_adapters_async(&self) -> Result<Vec<AdapterInfo>> {
        let fs = match &self.fs {
            Some(fs) => fs,
            None => return self.list_adapters(),
        };

        let mut adapters = Vec::new();

        // Check if adapters dir exists
        if !fs.stat_path("adapters").await.map(|s| s.exists).unwrap_or(false) {
            return Ok(adapters);
        }

        let entries = fs.list_dir_path("adapters").await?;

        for entry in entries {
            let name_str = &entry.name;

            // Only process .safetensors files
            if !name_str.ends_with(".safetensors") {
                continue;
            }

            // Parse index and name from filename like "00_base.safetensors"
            let base_name = name_str.trim_end_matches(".safetensors");
            let (index, name) = if let Some(underscore_pos) = base_name.find('_') {
                let index_str = &base_name[..underscore_pos];
                let adapter_name = &base_name[underscore_pos + 1..];

                if let Ok(idx) = index_str.parse::<u32>() {
                    (idx, adapter_name.to_owned())
                } else {
                    (999, base_name.to_owned())
                }
            } else {
                (999, base_name.to_owned())
            };

            let rel_config = format!("adapters/{base_name}.config.json");
            let config_exists = fs.stat_path(&rel_config).await.map(|s| s.exists).unwrap_or(false);

            adapters.push(AdapterInfo {
                filename: name_str.clone(),
                index,
                name,
                // For FsOps, path is relative within worktree
                path: PathBuf::from(format!("adapters/{name_str}")),
                size: entry.size,
                config_path: if config_exists {
                    Some(PathBuf::from(rel_config))
                } else {
                    None
                },
            });
        }

        adapters.sort_by_key(|a| a.index);
        Ok(adapters)
    }

    /// Get the next available index (async, via FsOps)
    pub async fn get_next_index_async(&self) -> Result<u32> {
        let adapters = self.list_adapters_async().await?;
        if adapters.is_empty() {
            Ok(0)
        } else {
            Ok(adapters.last().map(|a| a.index + 1).unwrap_or(0))
        }
    }

    /// Create an indexed adapter filename (async, via FsOps)
    pub async fn create_indexed_name_async(&self, name: &str, index: Option<u32>) -> Result<String> {
        let idx = if let Some(i) = index {
            i
        } else {
            self.get_next_index_async().await?
        };
        Ok(format!("{idx:02}_{name}"))
    }

    /// Initialize adapter via FsOps (async)
    ///
    /// Returns the relative path within the worktree (e.g., "adapters/00_base.safetensors").
    pub async fn initialize_adapter_async(
        &self,
        name: &str,
        index: Option<u32>,
        config: AdapterConfig,
    ) -> Result<String> {
        let fs = match &self.fs {
            Some(fs) => fs,
            None => {
                let path = self.initialize_adapter(name, index, config)?;
                return Ok(path.to_string_lossy().to_string());
            }
        };

        fs.mkdir_p("adapters").await?;

        let idx = if let Some(i) = index {
            i
        } else {
            self.get_next_index_async().await?
        };

        let adapter_name = format!("{idx:02}_{name}");

        // Save configuration file
        let rel_config = format!("adapters/{adapter_name}.config.json");
        let config_json = serde_json::to_string_pretty(&config)
            .with_context(|| "Failed to serialize adapter config")?;
        fs.write_file_chunked(&rel_config, config_json.as_bytes()).await?;

        // Create empty adapter file
        let rel_adapter = format!("adapters/{adapter_name}.safetensors");
        fs.write_file_chunked(&rel_adapter, &[]).await?;

        Ok(rel_adapter)
    }

    /// Load configuration for a specific adapter (async, via FsOps)
    pub async fn load_adapter_config_async(&self, adapter_name: &str) -> Result<AdapterConfig> {
        let fs = match &self.fs {
            Some(fs) => fs,
            None => return self.load_adapter_config(adapter_name),
        };

        let rel_config = format!("adapters/{adapter_name}.config.json");
        let config_str = String::from_utf8(fs.read_file_chunked(&rel_config).await?)?;
        let config: AdapterConfig = serde_json::from_str(&config_str)?;
        Ok(config)
    }

    /// Check if any adapters exist (async, via FsOps)
    pub async fn has_adapters_async(&self) -> Result<bool> {
        let fs = match &self.fs {
            Some(fs) => fs,
            None => return Ok(self.has_adapters()),
        };

        if !fs.stat_path("adapters").await.map(|s| s.exists).unwrap_or(false) {
            return Ok(false);
        }

        let adapters = self.list_adapters_async().await?;
        Ok(!adapters.is_empty())
    }

    /// Remove an adapter by name or index (async, via FsOps)
    pub async fn remove_adapter_async(&self, identifier: &str) -> Result<()> {
        let fs = match &self.fs {
            Some(fs) => fs,
            None => return self.remove_adapter(identifier),
        };

        let adapters = self.list_adapters_async().await?;

        let to_remove = if let Ok(idx) = identifier.parse::<u32>() {
            adapters.iter().find(|a| a.index == idx)
        } else {
            adapters
                .iter()
                .find(|a| a.name == identifier || a.filename.starts_with(identifier))
        };

        if let Some(adapter) = to_remove {
            // Remove adapter file (path is relative for FsOps)
            let rel_path = adapter.path.to_string_lossy();
            fs.remove_path(&rel_path).await?;

            // Remove config if exists
            if let Some(config_path) = &adapter.config_path {
                let rel_config = config_path.to_string_lossy();
                if fs.stat_path(&rel_config).await.map(|s| s.exists).unwrap_or(false) {
                    fs.remove_path(&rel_config).await?;
                }
            }

            Ok(())
        } else {
            anyhow::bail!("Adapter '{}' not found", identifier)
        }
    }

    /// Write adapter config file (async, via FsOps)
    pub async fn write_config_async(&self, adapter_name: &str, config: &AdapterConfig) -> Result<()> {
        let fs = match &self.fs {
            Some(fs) => fs,
            None => {
                let config_path = self
                    .adapters_dir
                    .join(format!("{adapter_name}.config.json"));
                let config_json = serde_json::to_string_pretty(config)?;
                std::fs::write(&config_path, config_json)?;
                return Ok(());
            }
        };

        let rel_config = format!("adapters/{adapter_name}.config.json");
        let config_json = serde_json::to_string_pretty(config)?;
        fs.write_file_chunked(&rel_config, config_json.as_bytes()).await?;
        Ok(())
    }

    /// Write adapter weights data (async, via FsOps)
    pub async fn write_adapter_data_async(&self, adapter_name: &str, data: &[u8]) -> Result<()> {
        let fs = match &self.fs {
            Some(fs) => fs,
            None => {
                let adapter_path = self
                    .adapters_dir
                    .join(format!("{adapter_name}.safetensors"));
                std::fs::write(&adapter_path, data)?;
                return Ok(());
            }
        };

        let rel_path = format!("adapters/{adapter_name}.safetensors");
        fs.write_file_chunked(&rel_path, data).await?;
        Ok(())
    }

    /// Read adapter weights data (async, via FsOps)
    pub async fn read_adapter_data_async(&self, adapter_name: &str) -> Result<Vec<u8>> {
        let fs = match &self.fs {
            Some(fs) => fs,
            None => {
                let adapter_path = self
                    .adapters_dir
                    .join(format!("{adapter_name}.safetensors"));
                return Ok(std::fs::read(&adapter_path)?);
            }
        };

        let rel_path = format!("adapters/{adapter_name}.safetensors");
        fs.read_file_chunked(&rel_path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_adapter_indexing() -> Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("test_model");
        std::fs::create_dir_all(&model_path)?;

        let manager = AdapterManager::new(&model_path);
        manager.ensure_adapters_dir()?;

        // Test getting next index on empty directory
        assert_eq!(manager.get_next_index()?, 0);

        // Create some test adapters
        let adapters_dir = model_path.join("adapters");
        std::fs::write(adapters_dir.join("00_base.safetensors"), b"")?;
        std::fs::write(adapters_dir.join("01_custom.safetensors"), b"")?;

        // Test getting next index with existing adapters
        assert_eq!(manager.get_next_index()?, 2);

        // Test listing adapters
        let adapters = manager.list_adapters()?;
        assert_eq!(adapters.len(), 2);
        assert_eq!(adapters[0].index, 0);
        assert_eq!(adapters[0].name, "base");
        assert_eq!(adapters[1].index, 1);
        assert_eq!(adapters[1].name, "custom");
        Ok(())
    }

    #[test]
    fn test_adapter_initialization() -> Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("test_model");
        std::fs::create_dir_all(&model_path)?;

        let manager = AdapterManager::new(&model_path);

        let mut config = AdapterConfig::default();
        config.model_ref = "test_model".to_owned();

        // Initialize adapter with auto-index
        let adapter_path = manager
            .initialize_adapter("test", None, config.clone())?;
        assert!(adapter_path.exists());
        let path_str = adapter_path.to_str().ok_or_else(|| anyhow::anyhow!("path not valid utf-8"))?;
        assert!(path_str.contains("00_test"));

        // Initialize adapter with specific index
        let adapter_path = manager
            .initialize_adapter("custom", Some(5), config)?;
        assert!(adapter_path.exists());
        let path_str = adapter_path.to_str().ok_or_else(|| anyhow::anyhow!("path not valid utf-8"))?;
        assert!(path_str.contains("05_custom"));

        // Check that configs were created
        let adapters_dir = model_path.join("adapters");
        assert!(adapters_dir.join("00_test.config.json").exists());
        assert!(adapters_dir.join("05_custom.config.json").exists());

        // Verify adapter files were created
        assert!(adapters_dir.join("00_test.safetensors").exists());
        assert!(adapters_dir.join("05_custom.safetensors").exists());
        Ok(())
    }
}
