//! Local model storage and metadata management

use crate::api::model_management::ModelUri;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Model metadata stored locally
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub uri: ModelUri,
    pub size_bytes: u64,
    pub files: Vec<String>,
    pub model_type: String,
    pub architecture: Option<String>,
    pub created_at: i64,
    pub last_accessed: i64,
    pub parameters: Option<u64>,
    pub tokenizer_type: Option<String>,
    pub tags: Vec<String>,
}

/// Local model storage manager
pub struct ModelStorage {
    /// Base directory for model storage
    base_dir: PathBuf,
    
    /// Metadata cache file
    metadata_file: PathBuf,
    
    /// In-memory metadata cache
    metadata_cache: tokio::sync::RwLock<HashMap<String, ModelMetadata>>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_models: usize,
    pub total_size_bytes: u64,
    pub models_by_registry: HashMap<String, usize>,
}

/// Cleanup result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupResult {
    pub models_removed: usize,
    pub bytes_freed: u64,
    pub models_remaining: usize,
}

impl ModelStorage {
    /// Create new model storage manager
    pub async fn new(base_dir: PathBuf) -> Result<Self> {
        // Ensure base directory exists
        fs::create_dir_all(&base_dir).await?;
        
        let metadata_file = base_dir.join("metadata.json");
        let storage = Self {
            base_dir,
            metadata_file,
            metadata_cache: tokio::sync::RwLock::new(HashMap::new()),
        };
        
        // Load existing metadata
        storage.load_metadata().await?;
        
        Ok(storage)
    }
    
    /// Load metadata from disk
    async fn load_metadata(&self) -> Result<()> {
        if !self.metadata_file.exists() {
            return Ok(());
        }
        
        let content = fs::read_to_string(&self.metadata_file).await?;
        let metadata_map: HashMap<String, ModelMetadata> = serde_json::from_str(&content)?;
        
        let mut cache = self.metadata_cache.write().await;
        *cache = metadata_map;
        
        Ok(())
    }
    
    /// Save metadata to disk
    async fn save_metadata(&self) -> Result<()> {
        let cache = self.metadata_cache.read().await;
        let content = serde_json::to_string_pretty(&*cache)?;
        fs::write(&self.metadata_file, content).await?;
        Ok(())
    }
    
    /// Store metadata for a model
    pub async fn store_metadata(&self, model_uri: &ModelUri, metadata: ModelMetadata) -> Result<()> {
        let mut cache = self.metadata_cache.write().await;
        cache.insert(model_uri.uri.clone(), metadata);
        drop(cache);
        
        self.save_metadata().await?;
        Ok(())
    }
    
    /// Get metadata for a model
    pub async fn get_metadata(&self, model_uri: &ModelUri) -> Result<ModelMetadata> {
        let cache = self.metadata_cache.read().await;
        cache.get(&model_uri.uri)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Model metadata not found: {}", model_uri.uri))
    }
    
    /// Remove metadata for a model
    pub async fn remove_metadata(&self, model_uri: &ModelUri) -> Result<()> {
        let mut cache = self.metadata_cache.write().await;
        cache.remove(&model_uri.uri);
        drop(cache);
        
        self.save_metadata().await?;
        Ok(())
    }
    
    /// List all locally cached models
    pub async fn list_local_models(&self) -> Result<Vec<(ModelUri, ModelMetadata)>> {
        let cache = self.metadata_cache.read().await;
        let mut models = Vec::new();
        
        for (uri_str, metadata) in cache.iter() {
            if let Ok(uri) = ModelUri::parse(uri_str) {
                // Check if model files still exist
                let model_path = uri.local_path(&self.base_dir);
                if model_path.exists() {
                    models.push((uri, metadata.clone()));
                }
            }
        }
        
        Ok(models)
    }
    
    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> Result<CacheStats> {
        let cache = self.metadata_cache.read().await;
        let mut total_size = 0u64;
        let mut registry_counts = HashMap::new();
        
        for metadata in cache.values() {
            // Check if files still exist
            let model_path = metadata.uri.local_path(&self.base_dir);
            if model_path.exists() {
                total_size += metadata.size_bytes;
                *registry_counts.entry(metadata.uri.registry.clone()).or_insert(0) += 1;
            }
        }
        
        Ok(CacheStats {
            total_models: registry_counts.values().sum(),
            total_size_bytes: total_size,
            models_by_registry: registry_counts,
        })
    }
    
    /// Clean up old models to free space
    pub async fn cleanup_old_models(&self, target_size_bytes: u64) -> Result<CleanupResult> {
        let cache_stats = self.get_cache_stats().await?;
        
        if cache_stats.total_size_bytes <= target_size_bytes {
            return Ok(CleanupResult {
                models_removed: 0,
                bytes_freed: 0,
                models_remaining: cache_stats.total_models,
            });
        }
        
        // Get all models sorted by last accessed time (oldest first)
        let mut models = self.list_local_models().await?;
        models.sort_by_key(|(_, metadata)| metadata.last_accessed);
        
        let mut bytes_freed = 0u64;
        let mut models_removed = 0;
        let bytes_to_free = cache_stats.total_size_bytes - target_size_bytes;
        
        for (uri, metadata) in models {
            if bytes_freed >= bytes_to_free {
                break;
            }
            
            // Remove model files
            let model_path = uri.local_path(&self.base_dir);
            if model_path.exists() {
                match fs::remove_dir_all(&model_path).await {
                    Ok(()) => {
                        bytes_freed += metadata.size_bytes;
                        models_removed += 1;
                        
                        // Remove from metadata cache
                        self.remove_metadata(&uri).await?;
                        
                        println!("🗑️ Cleaned up model: {} ({} MB)", 
                                uri.uri, metadata.size_bytes / (1024 * 1024));
                    }
                    Err(e) => {
                        eprintln!("⚠️ Failed to remove {}: {}", model_path.display(), e);
                    }
                }
            }
        }
        
        Ok(CleanupResult {
            models_removed,
            bytes_freed,
            models_remaining: cache_stats.total_models - models_removed,
        })
    }
    
    /// Update last accessed time for a model
    pub async fn update_access_time(&self, model_uri: &ModelUri) -> Result<()> {
        let mut cache = self.metadata_cache.write().await;
        if let Some(metadata) = cache.get_mut(&model_uri.uri) {
            metadata.last_accessed = chrono::Utc::now().timestamp();
            drop(cache);
            self.save_metadata().await?;
        }
        Ok(())
    }
    
    /// Get total size of a model directory
    pub async fn calculate_model_size(&self, model_path: &Path) -> Result<u64> {
        self.calculate_model_size_recursive(model_path).await
    }
    
    /// Recursive helper method for calculate_model_size
    fn calculate_model_size_recursive<'a>(&'a self, model_path: &'a Path) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64>> + Send + '_>> {
        Box::pin(async move {
            let mut total_size = 0u64;
            
            if !model_path.exists() {
                return Ok(0);
            }
            
            let mut entries = tokio::fs::read_dir(model_path).await?;
            while let Some(entry) = entries.next_entry().await? {
                let metadata = entry.metadata().await?;
                if metadata.is_file() {
                    total_size += metadata.len();
                } else if metadata.is_dir() {
                    total_size += self.calculate_model_size_recursive(&entry.path()).await?;
                }
            }
            
            Ok(total_size)
        })
    }
    
    /// Verify model integrity
    pub async fn verify_model(&self, model_uri: &ModelUri) -> Result<bool> {
        let metadata = self.get_metadata(model_uri).await?;
        let model_path = model_uri.local_path(&self.base_dir);
        
        if !model_path.exists() {
            return Ok(false);
        }
        
        // Check if all expected files exist
        for filename in &metadata.files {
            let file_path = model_path.join(filename);
            if !file_path.exists() {
                return Ok(false);
            }
        }
        
        // Optionally verify file sizes match
        let actual_size = self.calculate_model_size(&model_path).await?;
        if actual_size != metadata.size_bytes {
            println!("⚠️ Size mismatch for {}: expected {}, found {}", 
                    model_uri.uri, metadata.size_bytes, actual_size);
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Get base directory path
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
    
    /// Migrate metadata format (for future compatibility)
    pub async fn migrate_metadata(&self, _from_version: u32, _to_version: u32) -> Result<()> {
        // Placeholder for future metadata format migrations
        Ok(())
    }
}

/// Model file with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelFile {
    pub filename: String,
    pub size_bytes: u64,
    pub checksum: Option<String>,
    pub created_at: i64,
}

/// Model validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub missing_files: Vec<String>,
    pub extra_files: Vec<String>,
    pub size_mismatches: Vec<SizeMismatch>,
}

/// Size mismatch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeMismatch {
    pub filename: String,
    pub expected_size: u64,
    pub actual_size: u64,
}

use chrono;