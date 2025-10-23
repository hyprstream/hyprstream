//! Adapter management for LoRA models
//!
//! This module handles the storage, discovery, and composition of LoRA adapters
//! within the git-versioned model repositories.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
}

impl AdapterManager {
    /// Create a new adapter manager for a model
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        let adapters_dir = model_path.as_ref().join("adapters");
        Self { adapters_dir }
    }

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
                        (idx, name_str.to_string())
                    } else {
                        // If parsing fails, treat whole name as adapter name with index 999
                        (999, base_name.to_string())
                    }
                } else {
                    // No index prefix, assign high index
                    (999, base_name.to_string())
                };

                let metadata = std::fs::metadata(&path)?;
                let config_path = self.adapters_dir.join(format!("{}.config.json", base_name));

                adapters.push(AdapterInfo {
                    filename: name_str.to_string(),
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
        Ok(format!("{:02}_{}", idx, name))
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

        let adapter_name = format!("{:02}_{}", idx, name);
        let adapter_path = self
            .adapters_dir
            .join(format!("{}.safetensors", adapter_name));

        // Save configuration file
        let config_path = self
            .adapters_dir
            .join(format!("{}.config.json", adapter_name));
        let config_json = serde_json::to_string_pretty(&config)
            .with_context(|| "Failed to serialize adapter config")?;
        std::fs::write(&config_path, config_json)
            .with_context(|| format!("Failed to write adapter config: {:?}", config_path))?;

        // Create empty adapter file (weights will be added during training)
        std::fs::File::create(&adapter_path)
            .with_context(|| format!("Failed to create adapter file: {:?}", adapter_path))?;

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
            .join(format!("{}.config.json", adapter_name));
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read adapter config: {:?}", config_path))?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_adapter_indexing() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();

        let manager = AdapterManager::new(&model_path);
        manager.ensure_adapters_dir().unwrap();

        // Test getting next index on empty directory
        assert_eq!(manager.get_next_index().unwrap(), 0);

        // Create some test adapters
        let adapters_dir = model_path.join("adapters");
        std::fs::write(adapters_dir.join("00_base.safetensors"), b"").unwrap();
        std::fs::write(adapters_dir.join("01_custom.safetensors"), b"").unwrap();

        // Test getting next index with existing adapters
        assert_eq!(manager.get_next_index().unwrap(), 2);

        // Test listing adapters
        let adapters = manager.list_adapters().unwrap();
        assert_eq!(adapters.len(), 2);
        assert_eq!(adapters[0].index, 0);
        assert_eq!(adapters[0].name, "base");
        assert_eq!(adapters[1].index, 1);
        assert_eq!(adapters[1].name, "custom");
    }

    #[test]
    fn test_adapter_initialization() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();

        let manager = AdapterManager::new(&model_path);

        let mut config = AdapterConfig::default();
        config.model_ref = "test_model".to_string();

        // Initialize adapter with auto-index
        let adapter_path = manager
            .initialize_adapter("test", None, config.clone())
            .unwrap();
        assert!(adapter_path.exists());
        assert!(adapter_path.to_str().unwrap().contains("00_test"));

        // Initialize adapter with specific index
        let adapter_path = manager
            .initialize_adapter("custom", Some(5), config)
            .unwrap();
        assert!(adapter_path.exists());
        assert!(adapter_path.to_str().unwrap().contains("05_custom"));

        // Check that configs were created
        let adapters_dir = model_path.join("adapters");
        assert!(adapters_dir.join("00_test.config.json").exists());
        assert!(adapters_dir.join("05_custom.config.json").exists());

        // Verify adapter files were created
        assert!(adapters_dir.join("00_test.safetensors").exists());
        assert!(adapters_dir.join("05_custom.safetensors").exists());
    }
}
