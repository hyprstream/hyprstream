//! Local model storage and metadata management

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use url::Url;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fmt;
use tokio::fs;
use uuid::Uuid;

/// Model URI representation for different registries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelUri {
    /// Registry type (hf, custom, etc.)
    pub registry: String,
    
    /// Organization/namespace  
    pub org: String,
    
    /// Model name
    pub name: String,
    
    /// Optional revision/tag
    pub revision: Option<String>,
    
    /// Full URI string
    pub uri: String,
}

impl ModelUri {
    /// Parse model URI from string
    pub fn parse(uri: &str) -> Result<Self> {
        let url = Url::parse(uri)
            .map_err(|e| anyhow!("Invalid URI format: {}", e))?;
        
        let registry = url.scheme().to_string();
        let host = url.host_str().unwrap_or("");
        let path = url.path().trim_start_matches('/');
        
        let (org, name, revision) = match registry.as_str() {
            "hf" => {
                let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
                
                let (org, name_part) = if parts.len() >= 2 {
                    (parts[0], parts[1]) 
                } else if !host.is_empty() && parts.len() >= 1 {
                    (host, parts[0])
                } else {
                    return Err(anyhow!("HuggingFace URI must have org/name format"));
                };
                
                let (name, revision) = if name_part.contains('@') {
                    let parts: Vec<&str> = name_part.split('@').collect();
                    (parts[0], Some(parts[1].to_string()))
                } else {
                    (name_part, None)
                };
                
                (org.to_string(), name.to_string(), revision)
            }
            _ => {
                return Err(anyhow!("Unsupported registry: {}", registry));
            }
        };
        
        Ok(ModelUri {
            registry,
            org,
            name,
            revision,
            uri: uri.to_string(),
        })
    }
    
}

impl fmt::Display for ModelUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.uri)
    }
}

/// UUID-based model identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelId(pub Uuid);

/// UUID-based composed model identifier (base model + LoRA stack)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ComposedModelId(pub Uuid);

/// Model metadata file stored in each model directory
/// This allows safe UUID-based storage with human-readable metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadataFile {
    pub model_id: ModelId,
    pub name: String,
    pub display_name: String,
    pub source_uri: String,
    pub architecture: Option<String>,
    pub parameters: Option<u64>,
    pub created_at: i64,
    pub last_accessed: i64,
}

impl ModelId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_content_hash(name: &str, architecture: &str, parameters: Option<u64>) -> Self {
        let content = format!("{}:{}:{}", name, architecture, parameters.unwrap_or(0));
        // Use a simple hash of the content to create a reproducible UUID
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Create UUID from hash
        let bytes = hash.to_le_bytes();
        let mut uuid_bytes = [0u8; 16];
        uuid_bytes[..8].copy_from_slice(&bytes);
        uuid_bytes[8..].copy_from_slice(&bytes);
        
        Self(Uuid::from_bytes(uuid_bytes))
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for ModelId {
    type Err = uuid::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

impl ComposedModelId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for ComposedModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for ComposedModelId {
    type Err = uuid::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

/// External source reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSource {
    pub source_type: SourceType,
    pub identifier: String,
    pub revision: Option<String>,
    pub download_url: Option<String>,
    pub last_verified: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceType {
    HuggingFace,
    LocalPath,
    HttpUrl,
    Custom(String),
}

// Re-export ModelFile and FileType from model_downloader
pub use crate::api::model_downloader::{ModelFile, FileType};

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyModelMetadata {
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

/// UUID-based model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: ModelId,
    pub name: String,
    pub display_name: Option<String>,
    pub architecture: String,
    pub parameters: Option<u64>,
    pub model_type: String,
    pub tokenizer_type: Option<String>,
    
    // File information
    pub size_bytes: u64,
    pub files: Vec<ModelFile>,
    
    // Multiple external sources for same model
    pub external_sources: Vec<ExternalSource>,
    
    // Storage information
    pub local_path: Option<PathBuf>,
    pub is_cached: bool,
    
    // Metadata
    pub tags: Vec<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    
    // Timestamps
    pub created_at: i64,
    pub last_accessed: i64,
    pub last_updated: i64,
}

/// Composed model metadata (base model + LoRA stack)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedModelMetadata {
    pub composed_id: ComposedModelId,
    pub name: String,
    pub base_model_id: ModelId,
    pub lora_stack: Vec<String>, // LoRA IDs in application order
    pub created_at: i64,
    pub last_used: i64,
}

/// Local model storage manager
pub struct ModelStorage {
    /// Base directory for model storage
    base_dir: PathBuf,
    
    /// Metadata cache file
    metadata_file: PathBuf,
    
    /// In-memory metadata cache (keyed by ModelId UUID)
    metadata_cache: tokio::sync::RwLock<HashMap<ModelId, ModelMetadata>>,
    
    /// URI to UUID mapping cache (for backward compatibility)
    uri_to_uuid: tokio::sync::RwLock<HashMap<String, ModelId>>,
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
    /// Get the path for a UUID-based model directory with validation
    pub fn get_uuid_model_path(&self, model_id: &ModelId) -> PathBuf {
        // SECURITY: Validate UUID format to prevent injection
        let uuid_str = model_id.to_string();
        
        // UUID v4 format: 8-4-4-4-12 hex characters with hyphens
        let uuid_regex = regex::Regex::new(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$").unwrap();
        if !uuid_regex.is_match(&uuid_str) {
            // Return a safe fallback path if validation fails
            tracing::error!("Invalid UUID format detected: {}", uuid_str);
            return self.base_dir.join("invalid_uuid");
        }
        
        self.base_dir.join(uuid_str)
    }
    
    /// Save model metadata file in the model directory
    pub async fn save_model_metadata_file(&self, model_id: &ModelId, metadata: &ModelMetadataFile) -> Result<()> {
        let model_path = self.get_uuid_model_path(model_id);
        fs::create_dir_all(&model_path).await?;
        
        let metadata_file = model_path.join("model.json");
        let content = serde_json::to_string_pretty(metadata)?;
        fs::write(&metadata_file, content).await?;
        
        tracing::info!("Saved model metadata for {} to {:?}", model_id, metadata_file);
        Ok(())
    }
    
    /// Load model metadata file from a model directory with size limits
    pub async fn load_model_metadata_file(&self, model_id: &ModelId) -> Result<ModelMetadataFile> {
        let model_path = self.get_uuid_model_path(model_id);
        let metadata_file = model_path.join("model.json");
        
        // SECURITY: Check file size before reading to prevent OOM
        const MAX_METADATA_SIZE: u64 = 1024 * 1024; // 1MB max for metadata
        let file_meta = fs::metadata(&metadata_file).await?;
        if file_meta.len() > MAX_METADATA_SIZE {
            return Err(anyhow::anyhow!("Metadata file too large: {} bytes", file_meta.len()));
        }
        
        let content = fs::read_to_string(&metadata_file).await?;
        let metadata: ModelMetadataFile = serde_json::from_str(&content)?;
        
        Ok(metadata)
    }
    
    /// Check if a UUID-based model exists
    pub fn uuid_model_exists(&self, model_id: &ModelId) -> bool {
        let model_path = self.get_uuid_model_path(model_id);
        model_path.join("model.json").exists()
    }
    
    /// Create new model storage manager
    pub async fn new(base_dir: PathBuf) -> Result<Self> {
        // Ensure base directory exists
        fs::create_dir_all(&base_dir).await?;
        
        let metadata_file = base_dir.join("metadata.json");
        let storage = Self {
            base_dir,
            metadata_file,
            metadata_cache: tokio::sync::RwLock::new(HashMap::new()),
            uri_to_uuid: tokio::sync::RwLock::new(HashMap::new()),
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
        
        // Try to load new UUID-based format first
        if let Ok(uuid_metadata_map) = serde_json::from_str::<HashMap<ModelId, ModelMetadata>>(&content) {
            let mut cache = self.metadata_cache.write().await;
            let mut uri_map = self.uri_to_uuid.write().await;
            
            // Build URI to UUID mapping
            for (model_id, metadata) in &uuid_metadata_map {
                // Generate a URI key from external sources for backward compatibility
                if let Some(external_source) = metadata.external_sources.first() {
                    let uri_key = match external_source.source_type {
                        SourceType::HuggingFace => format!(
                            "hf://{}", 
                            external_source.identifier
                        ),
                        SourceType::LocalPath => format!(
                            "local://{}", 
                            external_source.identifier
                        ),
                        SourceType::HttpUrl => external_source.identifier.clone(),
                        SourceType::Custom(ref name) => format!(
                            "{}://{}", 
                            name, 
                            external_source.identifier
                        ),
                    };
                    uri_map.insert(uri_key, model_id.clone());
                }
            }
            
            *cache = uuid_metadata_map;
            return Ok(());
        }
        
        // Fall back to legacy URI-based format and migrate
        if let Ok(legacy_metadata_map) = serde_json::from_str::<HashMap<String, ModelMetadata>>(&content) {
            let mut cache = self.metadata_cache.write().await;
            let mut uri_map = self.uri_to_uuid.write().await;
            
            // Migrate legacy format to UUID-based format
            for (uri, metadata) in legacy_metadata_map {
                cache.insert(metadata.model_id.clone(), metadata.clone());
                uri_map.insert(uri, metadata.model_id.clone());
            }
            
            drop(cache);
            drop(uri_map);
            
            // Save migrated data in new format
            self.save_metadata().await?;
        }
        
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
        let mut uri_map = self.uri_to_uuid.write().await;
        
        // Get the model ID before moving metadata
        let model_id = metadata.model_id.clone();
        
        // Store with UUID as primary key
        cache.insert(model_id.clone(), metadata);
        // Maintain URI mapping for backward compatibility
        uri_map.insert(model_uri.uri.clone(), model_id);
        
        drop(cache);
        drop(uri_map);
        
        self.save_metadata().await?;
        Ok(())
    }
    
    /// Get metadata for a model by URI
    pub async fn get_metadata(&self, model_uri: &ModelUri) -> Result<ModelMetadata> {
        let uri_map = self.uri_to_uuid.read().await;
        let model_id = uri_map.get(&model_uri.uri)
            .ok_or_else(|| anyhow::anyhow!("Model URI not found: {}", model_uri.uri))?;
        
        let cache = self.metadata_cache.read().await;
        cache.get(model_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Model metadata not found for UUID: {}", model_id))
    }
    
    /// Get metadata for a model by UUID
    pub async fn get_metadata_by_id(&self, model_id: &ModelId) -> Result<ModelMetadata> {
        let cache = self.metadata_cache.read().await;
        cache.get(model_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Model metadata not found for UUID: {}", model_id))
    }
    
    /// Remove metadata for a model by URI
    pub async fn remove_metadata(&self, model_uri: &ModelUri) -> Result<()> {
        let mut uri_map = self.uri_to_uuid.write().await;
        if let Some(model_id) = uri_map.remove(&model_uri.uri) {
            drop(uri_map);
            
            let mut cache = self.metadata_cache.write().await;
            cache.remove(&model_id);
            drop(cache);
            
            self.save_metadata().await?;
        }
        Ok(())
    }
    
    /// Remove metadata for a model by UUID
    pub async fn remove_metadata_by_id(&self, model_id: &ModelId) -> Result<()> {
        let mut cache = self.metadata_cache.write().await;
        cache.remove(model_id);
        drop(cache);
        
        // Also remove from URI mapping
        let mut uri_map = self.uri_to_uuid.write().await;
        uri_map.retain(|_, id| id != model_id);
        drop(uri_map);
        
        self.save_metadata().await?;
        Ok(())
    }
    
    /// List all locally cached models
    pub async fn list_local_models(&self) -> Result<Vec<(ModelUri, ModelMetadata)>> {
        let cache = self.metadata_cache.read().await;
        let mut models = Vec::new();
        
        for (_model_id, metadata) in cache.iter() {
            // Generate ModelUri from external sources for backward compatibility
            if let Some(external_source) = metadata.external_sources.first() {
                let uri_str = match &external_source.source_type {
                    SourceType::HuggingFace => format!(
                        "hf://{}", 
                        external_source.identifier
                    ),
                    SourceType::LocalPath => format!(
                        "local://{}", 
                        external_source.identifier
                    ),
                    SourceType::HttpUrl => external_source.identifier.clone(),
                    SourceType::Custom(name) => format!(
                        "{}://{}", 
                        name, 
                        external_source.identifier
                    ),
                };
                
                if let Ok(uri) = ModelUri::parse(&uri_str) {
                    // Check if model files still exist
                    let model_exists = if let Some(local_path) = &metadata.local_path {
                        local_path.exists()
                    } else {
                        // UUID models should have local_path set
                        false
                    };
                    
                    if model_exists {
                        models.push((uri, metadata.clone()));
                    }
                }
            }
        }
        
        Ok(models)
    }
    
    /// List all models by UUID (new primary interface)
    pub async fn list_all(&self) -> Result<Vec<(ModelId, ModelMetadata)>> {
        let cache = self.metadata_cache.read().await;
        let mut models = Vec::new();
        
        for (model_id, metadata) in cache.iter() {
            // Check if model files still exist
            let model_exists = if let Some(local_path) = &metadata.local_path {
                local_path.exists()
            } else if let Some(external_source) = metadata.external_sources.first() {
                // Try to construct path from external source
                match &external_source.source_type {
                    SourceType::LocalPath => {
                        std::path::Path::new(&external_source.identifier).exists()
                    },
                    _ => true, // For remote sources, assume they exist
                }
            } else {
                false
            };
            
            if model_exists {
                models.push((model_id.clone(), metadata.clone()));
            }
        }
        
        Ok(models)
    }
    
    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> Result<CacheStats> {
        let cache = self.metadata_cache.read().await;
        let mut total_size = 0u64;
        let mut registry_counts = HashMap::new();
        
        for (_model_id, metadata) in cache.iter() {
            // Check if files still exist
            let model_exists = if let Some(local_path) = &metadata.local_path {
                local_path.exists()
            } else {
                false
            };
            
            if model_exists {
                total_size += metadata.size_bytes;
                // Extract registry from external sources
                if let Some(external_source) = metadata.external_sources.first() {
                    let registry = match &external_source.source_type {
                        SourceType::HuggingFace => "hf".to_string(),
                        SourceType::LocalPath => "local".to_string(),
                        SourceType::HttpUrl => "http".to_string(),
                        SourceType::Custom(name) => name.clone(),
                    };
                    *registry_counts.entry(registry).or_insert(0) += 1;
                }
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
            
            // Remove model files from UUID directory if we have the metadata
            if let Some(ref local_path) = metadata.local_path {
                match fs::remove_dir_all(&local_path).await {
                    Ok(()) => {
                        bytes_freed += metadata.size_bytes;
                        models_removed += 1;
                        
                        // Remove from metadata cache
                        self.remove_metadata(&uri).await?;
                        
                        println!("🗑️ Cleaned up model: {} ({} MB)", 
                                uri.uri, metadata.size_bytes / (1024 * 1024));
                    }
                    Err(e) => {
                        eprintln!("⚠️ Failed to remove model: {}", e);
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
    
    /// Update last accessed time for a model by URI
    pub async fn update_access_time(&self, model_uri: &ModelUri) -> Result<()> {
        // First get the UUID from URI mapping
        let uri_map = self.uri_to_uuid.read().await;
        if let Some(model_id) = uri_map.get(&model_uri.uri) {
            let model_id = model_id.clone();
            drop(uri_map);
            
            let mut cache = self.metadata_cache.write().await;
            if let Some(metadata) = cache.get_mut(&model_id) {
                metadata.last_accessed = chrono::Utc::now().timestamp();
                drop(cache);
                self.save_metadata().await?;
            }
        }
        Ok(())
    }
    
    /// Update last accessed time for a model by UUID
    pub async fn update_access_time_by_id(&self, model_id: &ModelId) -> Result<()> {
        let mut cache = self.metadata_cache.write().await;
        if let Some(metadata) = cache.get_mut(model_id) {
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
        
        // Get the model path from metadata
        let model_path = metadata.local_path
            .ok_or_else(|| anyhow::anyhow!("Model does not have a local path"))?;
        
        if !model_path.exists() {
            return Ok(false);
        }
        
        // Check if all expected files exist
        for model_file in &metadata.files {
            let file_path = model_path.join(&model_file.filename);
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
    
    /// List all models in storage (alias for list_local_models for API compatibility)
    pub async fn children(&self) -> Result<Vec<(ModelId, ModelMetadata)>> {
        // First, load metadata from cache
        let _ = self.load_metadata().await; // Refresh cache from disk
        
        let mut models = Vec::new();
        let metadata = self.metadata_cache.read().await;
        
        tracing::info!("Listing models from directory: {:?}", self.base_dir);
        tracing::info!("Found {} models in metadata cache", metadata.len());
        
        // Add models from metadata cache
        for (id, meta) in metadata.iter() {
            tracing::debug!("Model from cache: {} -> {}", id, meta.name);
            models.push((id.clone(), meta.clone()));
        }
        
        // Also scan the models directory for any models not in metadata
        match tokio::fs::read_dir(&self.base_dir).await {
            Ok(mut entries) => {
                tracing::info!("Scanning models directory for additional models...");
                let mut scanned_count = 0;
                
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    scanned_count += 1;
                    
                    if path.is_dir() {
                        let dir_name = path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown");
                        
                        tracing::debug!("Checking directory: {}", dir_name);
                        
                        // First check if this is a UUID directory with model.json
                        let model_json_path = path.join("model.json");
                        if model_json_path.exists() {
                            // SECURITY: Check file size before reading
                            const MAX_METADATA_SIZE: u64 = 1024 * 1024; // 1MB max
                            if let Ok(file_meta) = tokio::fs::metadata(&model_json_path).await {
                                if file_meta.len() > MAX_METADATA_SIZE {
                                    tracing::warn!("Skipping oversized metadata file: {:?}", model_json_path);
                                    continue;
                                }
                            }
                            
                            // Try to load the model metadata file
                            if let Ok(content) = tokio::fs::read_to_string(&model_json_path).await {
                                if let Ok(model_meta) = serde_json::from_str::<crate::api::model_storage::ModelMetadataFile>(&content) {
                                    tracing::info!("Found UUID model: {} ({})", model_meta.model_id, model_meta.display_name);
                                    
                                    // Check if already listed
                                    let already_listed = models.iter().any(|(id, _)| id == &model_meta.model_id);
                                    
                                    if !already_listed {
                                        // Convert ModelMetadataFile to ModelMetadata
                                        let metadata = ModelMetadata {
                                            model_id: model_meta.model_id.clone(),
                                            name: model_meta.name.clone(),
                                            display_name: Some(model_meta.display_name.clone()),
                                            architecture: model_meta.architecture.unwrap_or_else(|| "unknown".to_string()),
                                            parameters: model_meta.parameters,
                                            model_type: "language_model".to_string(),
                                            tokenizer_type: None,
                                            size_bytes: 0, // Could calculate if needed
                                            files: vec![],
                                            external_sources: vec![],
                                            local_path: Some(path.clone()),
                                            is_cached: true,
                                            tags: vec![],
                                            description: None,
                                            license: None,
                                            created_at: model_meta.created_at,
                                            last_accessed: model_meta.last_accessed,
                                            last_updated: model_meta.created_at,
                                        };
                                        
                                        models.push((model_meta.model_id, metadata));
                                    }
                                    continue; // Skip legacy check for UUID models
                                }
                            }
                        }
                        
                    }
                }
                
                tracing::info!("Scanned {} entries in models directory", scanned_count);
            }
            Err(e) => {
                tracing::warn!("Failed to read models directory {:?}: {}", self.base_dir, e);
            }
        }
        
        tracing::info!("Total models found: {}", models.len());
        Ok(models)
    }
    
    /// Check if a directory contains model files
    async fn has_model_files(path: &Path) -> bool {
        // Check for common model file patterns
        let patterns = ["*.safetensors", "*.bin", "*.gguf", "*.ggml", "config.json", "tokenizer.json"];
        
        for pattern in &patterns {
            if pattern.contains('*') {
                // For wildcards, check if any file matches
                if let Ok(mut entries) = tokio::fs::read_dir(path).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        let file_name = entry.file_name();
                        let file_str = file_name.to_string_lossy();
                        if pattern == &"*.safetensors" && file_str.ends_with(".safetensors") {
                            return true;
                        } else if pattern == &"*.bin" && file_str.ends_with(".bin") {
                            return true;
                        } else if pattern == &"*.gguf" && file_str.ends_with(".gguf") {
                            return true;
                        } else if pattern == &"*.ggml" && file_str.ends_with(".ggml") {
                            return true;
                        }
                    }
                }
            } else {
                // For exact filenames
                let file_path = path.join(pattern);
                if file_path.exists() {
                    return true;
                }
            }
        }
        false
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