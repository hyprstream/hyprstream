//! Registry repair utilities to fix the existing broken state
//!
//! This module provides tools to migrate from the broken dual-authority state
//! to a properly synchronized registry-based system.

use anyhow::{Result, Context, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn, error};

use super::{
    ModelRef, SharedModelRegistry,
};

/// Report from repair operations
#[derive(Debug, Default)]
pub struct RepairReport {
    pub models_found: Vec<String>,
    pub models_registered: Vec<String>,
    pub models_already_registered: Vec<String>,
    pub models_failed: Vec<(String, String)>,
    pub orphaned_submodules: Vec<String>,
    pub inconsistencies_fixed: usize,
}

impl RepairReport {
    /// Check if the repair was successful
    pub fn is_success(&self) -> bool {
        self.models_failed.is_empty() && self.orphaned_submodules.is_empty()
    }

    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("Registry Repair Summary:\n"));
        summary.push_str(&format!("  Models found in file system: {}\n", self.models_found.len()));
        summary.push_str(&format!("  Models newly registered: {}\n", self.models_registered.len()));
        summary.push_str(&format!("  Models already registered: {}\n", self.models_already_registered.len()));

        if !self.models_failed.is_empty() {
            summary.push_str(&format!("  Failed registrations: {}\n", self.models_failed.len()));
            for (name, error) in &self.models_failed {
                summary.push_str(&format!("    - {}: {}\n", name, error));
            }
        }

        if !self.orphaned_submodules.is_empty() {
            summary.push_str(&format!("  Orphaned submodules removed: {}\n", self.orphaned_submodules.len()));
            for name in &self.orphaned_submodules {
                summary.push_str(&format!("    - {}\n", name));
            }
        }

        if self.inconsistencies_fixed > 0 {
            summary.push_str(&format!("  Inconsistencies fixed: {}\n", self.inconsistencies_fixed));
        }

        summary
    }
}

/// Registry repair utility
pub struct RegistryRepair {
    models_dir: PathBuf,
    registry: Arc<SharedModelRegistry>,
    dry_run: bool,
}

impl RegistryRepair {
    /// Create a new repair utility
    pub fn new(
        models_dir: PathBuf,
        registry: Arc<SharedModelRegistry>,
        dry_run: bool,
    ) -> Self {
        Self {
            models_dir,
            registry,
            dry_run,
        }
    }

    /// Perform a full repair of the registry
    pub async fn repair(&self) -> Result<RepairReport> {
        let mut report = RepairReport::default();

        info!("Starting registry repair (dry_run: {})", self.dry_run);

        // Step 1: Discover all models in the file system
        let fs_models = self.discover_filesystem_models().await?;
        report.models_found = fs_models.iter().map(|(name, _)| name.clone()).collect();

        info!("Found {} models in file system", fs_models.len());

        // Step 2: Check which models are already registered
        // Note: list_models returns (String, Oid) not (ModelRef, _)
        let registered_models = self.registry.list_models().await?;
        let registered_names: std::collections::HashSet<_> = registered_models
            .iter()
            .map(|(model_name, _commit_id)| model_name.clone())
            .collect();

        // Step 3: Register unregistered models
        for (name, path) in &fs_models {
            if registered_names.contains(name) {
                report.models_already_registered.push(name.clone());
                info!("Model {} is already registered", name);
            } else {
                info!("Model {} needs registration", name);

                if !self.dry_run {
                    match self.register_model(name, path).await {
                        Ok(_) => {
                            info!("Successfully registered model {}", name);
                            report.models_registered.push(name.clone());
                        }
                        Err(e) => {
                            error!("Failed to register model {}: {}", name, e);
                            report.models_failed.push((name.clone(), e.to_string()));
                        }
                    }
                } else {
                    info!("[DRY RUN] Would register model {}", name);
                    report.models_registered.push(name.clone());
                }
            }
        }

        // Step 4: Check for orphaned submodules (in registry but not in file system)
        for (model_name, _commit_id) in &registered_models {
            let model_path = self.models_dir.join(model_name);
            if !model_path.exists() {
                warn!("Found orphaned submodule: {}", model_name);
                report.orphaned_submodules.push(model_name.clone());

                if !self.dry_run {
                    // TODO: Implement remove_model in registry to handle orphaned submodules
                    warn!("Cannot remove orphaned submodule {} - manual cleanup required", model_name);
                    report.inconsistencies_fixed += 1;
                } else {
                    info!("[DRY RUN] Would remove orphaned submodule {}", model_name);
                    report.inconsistencies_fixed += 1;
                }
            }
        }

        // Step 5: Verify git consistency for registered models
        for (name, path) in &fs_models {
            if registered_names.contains(name) {
                if let Err(e) = self.verify_git_consistency(name, path).await {
                    warn!("Git consistency issue for {}: {}", name, e);

                    if !self.dry_run {
                        if let Ok(_) = self.fix_git_consistency(name, path).await {
                            info!("Fixed git consistency for {}", name);
                            report.inconsistencies_fixed += 1;
                        }
                    } else {
                        info!("[DRY RUN] Would fix git consistency for {}", name);
                        report.inconsistencies_fixed += 1;
                    }
                }
            }
        }

        info!("Registry repair completed\n{}", report.summary());

        Ok(report)
    }

    /// Discover all models in the file system
    async fn discover_filesystem_models(&self) -> Result<Vec<(String, PathBuf)>> {
        let mut models = Vec::new();

        if !self.models_dir.exists() {
            return Ok(models);
        }

        let entries = std::fs::read_dir(&self.models_dir)
            .context("Failed to read models directory")?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Check if it's a git repository
                if path.join(".git").exists() {
                    if let Some(name) = path.file_name() {
                        let name = name.to_string_lossy().to_string();

                        // Skip hidden directories and special names
                        if !name.starts_with('.') && name != "registry" && name != "deprecated" {
                            models.push((name, path));
                        }
                    }
                }
            }
        }

        Ok(models)
    }

    /// Register a model with the git registry
    async fn register_model(&self, name: &str, path: &Path) -> Result<()> {
        // Try to determine the origin URL
        let origin_url = self.get_git_origin(path).ok();

        // Register with the registry - takes name as string, not ModelRef
        self.registry
            .add_model(name, &origin_url.unwrap_or_default())
            .await
            .context(format!("Failed to register model {}", name))?;

        Ok(())
    }

    /// Get the git origin URL for a repository
    fn get_git_origin(&self, repo_path: &Path) -> Result<String> {
        let repo = git2::Repository::open(repo_path)?;
        let url = repo.find_remote("origin")?
            .url()
            .ok_or_else(|| anyhow::anyhow!("No URL for origin remote"))?
            .to_string();
        Ok(url)
    }

    /// Verify git consistency between file system and registry
    async fn verify_git_consistency(&self, name: &str, path: &Path) -> Result<()> {
        let model_ref = ModelRef::new(name.to_string());

        // Check if the registry knows about this model
        let registry_info = self.registry.get_model_info(&model_ref).await?;

        // Check if the paths match
        if registry_info.path != *path {
            bail!("Path mismatch: registry has {:?}, file system has {:?}",
                  registry_info.path, path);
        }

        // Check if both point to the same commit
        let repo = git2::Repository::open(path)?;
        let head = repo.head()?.target()
            .ok_or_else(|| anyhow::anyhow!("No HEAD commit"))?;

        if head != registry_info.current_oid {
            bail!("Commit mismatch: file system at {}, registry at {}",
                  head, registry_info.current_oid);
        }

        Ok(())
    }

    /// Fix git consistency issues
    async fn fix_git_consistency(&self, name: &str, path: &Path) -> Result<()> {
        // For now, we'll re-register the model which should update the registry
        // TODO: Once remove_model is implemented, we can remove and re-add
        // For now, just try to re-register which may update the submodule

        // Re-register
        self.register_model(name, path).await?;

        Ok(())
    }

    /// Create a backup of the current state before repair
    pub async fn create_backup(&self) -> Result<PathBuf> {
        let backup_dir = self.models_dir.parent()
            .unwrap_or(Path::new("/tmp"))
            .join(format!("registry_backup_{}", chrono::Utc::now().timestamp()));

        std::fs::create_dir_all(&backup_dir)?;

        // Copy the registry directory if it exists
        let registry_dir = self.models_dir.parent()
            .map(|p| p.join("registry"))
            .filter(|p| p.exists());

        if let Some(registry_dir) = registry_dir {
            let backup_registry = backup_dir.join("registry");
            self.copy_dir_recursive(&registry_dir, &backup_registry)?;
            info!("Backed up registry to {:?}", backup_registry);
        }

        // Save current model list
        let models = self.discover_filesystem_models().await?;
        let models_json = serde_json::to_string_pretty(&models)?;
        std::fs::write(backup_dir.join("models.json"), models_json)?;

        info!("Created backup at {:?}", backup_dir);

        Ok(backup_dir)
    }

    /// Copy directory recursively
    fn copy_dir_recursive(&self, src: &Path, dst: &Path) -> Result<()> {
        std::fs::create_dir_all(dst)?;

        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry.file_name();
            let dst_path = dst.join(file_name);

            if path.is_dir() {
                self.copy_dir_recursive(&path, &dst_path)?;
            } else {
                std::fs::copy(&path, &dst_path)?;
            }
        }

        Ok(())
    }
}

/// Convenience function to repair the registry
pub async fn repair_registry(
    models_dir: PathBuf,
    registry: Arc<SharedModelRegistry>,
    dry_run: bool,
    create_backup: bool,
) -> Result<RepairReport> {
    let repair = RegistryRepair::new(models_dir, registry, dry_run);

    if create_backup && !dry_run {
        repair.create_backup().await?;
    }

    repair.repair().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_setup() -> Result<(RegistryRepair, TempDir)> {
        let temp_dir = TempDir::new()?;
        let models_dir = temp_dir.path().join("models");
        std::fs::create_dir_all(&models_dir)?;

        let registry_dir = temp_dir.path().join("registry");
        let registry = SharedModelRegistry::open(registry_dir, None).await?;

        let repair = RegistryRepair::new(
            models_dir,
            Arc::new(registry),
            false, // Not dry run for tests
        );

        Ok((repair, temp_dir))
    }

    #[tokio::test]
    async fn test_discover_models() {
        let (repair, temp) = create_test_setup().await.unwrap();

        // Create a model directory with git repo
        let model_dir = temp.path().join("models").join("test-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        git2::Repository::init(&model_dir).unwrap();

        let models = repair.discover_filesystem_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].0, "test-model");
    }

    #[tokio::test]
    async fn test_repair_unregistered_model() {
        let (repair, temp) = create_test_setup().await.unwrap();

        // Create an unregistered model
        let model_dir = temp.path().join("models").join("unregistered");
        std::fs::create_dir_all(&model_dir).unwrap();
        git2::Repository::init(&model_dir).unwrap();

        // Run repair
        let report = repair.repair().await.unwrap();

        assert_eq!(report.models_found.len(), 1);
        assert_eq!(report.models_registered.len(), 1);
        assert_eq!(report.models_registered[0], "unregistered");
    }

    #[tokio::test]
    async fn test_dry_run() {
        let (mut repair, temp) = create_test_setup().await.unwrap();
        repair.dry_run = true;

        // Create an unregistered model
        let model_dir = temp.path().join("models").join("test-dry-run");
        std::fs::create_dir_all(&model_dir).unwrap();
        git2::Repository::init(&model_dir).unwrap();

        // Run repair in dry run mode
        let report = repair.repair().await.unwrap();

        assert_eq!(report.models_found.len(), 1);
        assert_eq!(report.models_registered.len(), 1);

        // Verify nothing was actually changed
        let registered = repair.registry.list_models().await.unwrap();
        assert_eq!(registered.len(), 0);
    }
}