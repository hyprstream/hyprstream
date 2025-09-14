//! Local model storage and metadata management

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use url::Url;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fmt;
use tokio::fs;
use uuid::Uuid;
use json_threat_protection as jtp;

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
        let uuid_str = model_id.0.to_string();
        
        // Validate UUID format for security
        match Uuid::parse_str(&uuid_str) {
            Ok(_) => self.base_dir.join(uuid_str),
            Err(e) => {
                tracing::error!("UUID validation failed for {}: {}", uuid_str, e);
                self.base_dir.join("invalid_uuid")
            }
        }
    }
    
    /// Get the data directory for model files (weights, configs, etc.)
    pub fn get_uuid_model_data_path(&self, model_id: &ModelId) -> PathBuf {
        self.get_uuid_model_path(model_id).join("data")
    }
    
    /// Save model metadata file in the model directory
    pub async fn save_model_metadata_file(&self, model_id: &ModelId, metadata: &ModelMetadataFile) -> Result<()> {
        let model_path = self.get_uuid_model_path(model_id);
        fs::create_dir_all(&model_path).await?;
        
        let metadata_file = model_path.join("model.json");
        
        // Serialize and validate JSON
        let content = serde_json::to_string_pretty(metadata)?;
        jtp::from_str(&content)
            .with_max_depth(10)
            .with_max_string_length(10000)
            .validate()?;
        
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
        
        // Validate JSON before deserializing
        jtp::from_str(&content)
            .with_max_depth(10)
            .with_max_string_length(10000)
            .with_max_array_entries(1000)
            .validate()?;
        
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
    
    /// Load metadata from disk (cache only, not authoritative)
    async fn load_metadata(&self) -> Result<()> {
        if !self.metadata_file.exists() {
            tracing::debug!("No metadata cache file found, will scan directories");
            let _ = self.children().await?;
            return Ok(());
        }
        
        let content = fs::read_to_string(&self.metadata_file).await?;
        
        // Validate JSON before deserializing
        jtp::from_str(&content)
            .with_max_depth(15)
            .with_max_string_length(50000)
            .with_max_array_entries(10000)
            .validate()?;
        
        if let Ok(uuid_metadata_map) = serde_json::from_str::<HashMap<ModelId, ModelMetadata>>(&content) {
            let mut cache = self.metadata_cache.write().await;
            let mut uri_map = self.uri_to_uuid.write().await;
            
            // Validate each cached entry against actual directories
            for (model_id, metadata) in uuid_metadata_map {
                let model_path = self.get_uuid_model_path(&model_id);
                let model_json_path = model_path.join("model.json");
                
                // Only include in cache if the directory and model.json exist
                if model_json_path.exists() {
                    // Verify UUID matches
                    if let Ok(metadata_file) = self.load_model_metadata_file(&model_id).await {
                        if metadata_file.model_id == model_id {
                            // Valid entry, add to cache
                            cache.insert(model_id.clone(), metadata.clone());
                            
                            // Build URI mapping
                            if let Some(external_source) = metadata.external_sources.first() {
                                let uri_key = match external_source.source_type {
                                    SourceType::HuggingFace => format!("hf://{}", external_source.identifier),
                                    SourceType::LocalPath => format!("local://{}", external_source.identifier),
                                    SourceType::HttpUrl => external_source.identifier.clone(),
                                    SourceType::Custom(ref name) => format!("{}://{}", name, external_source.identifier),
                                };
                                uri_map.insert(uri_key, model_id.clone());
                            }
                        } else {
                            tracing::warn!("UUID mismatch in cache for {}, skipping", model_id);
                        }
                    }
                } else {
                    tracing::debug!("Model {} in cache but not on disk, skipping", model_id);
                }
            }
            
            tracing::info!("Loaded {} valid models from metadata cache", cache.len());
        } else {
            tracing::warn!("Failed to parse metadata cache, will rebuild from directories");
            // Rebuild cache from directories
            let _ = self.children().await?;
        }
        
        Ok(())
    }
    
    /// Save metadata to disk
    async fn save_metadata(&self) -> Result<()> {
        let cache = self.metadata_cache.read().await;
        
        // Serialize and validate JSON
        let content = serde_json::to_string_pretty(&*cache)?;
        jtp::from_str(&content)
            .with_max_depth(15)
            .with_max_string_length(50000)
            .validate()?;
        
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
        // Remove the actual model directory
        let model_path = self.get_uuid_model_path(model_id);
        
        if model_path.exists() {
            tracing::info!("Removing model directory: {:?}", model_path);
            fs::remove_dir_all(&model_path).await?;
        }
        
        // Remove from cache
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
    
    /// Get metadata by ID with direct directory loading fallback
    pub async fn get_metadata_by_id_or_load(&self, model_id: &ModelId) -> Result<ModelMetadata> {
        // First try cache
        {
            let cache = self.metadata_cache.read().await;
            if let Some(metadata) = cache.get(model_id) {
                return Ok(metadata.clone());
            }
        }
        
        // Not in cache, try to load from directory
        let model_path = self.get_uuid_model_path(model_id);
        let model_json_path = model_path.join("model.json");
        
        if !model_json_path.exists() {
            return Err(anyhow!("Model {} not found", model_id));
        }
        
        // Load metadata file
        let metadata_file = self.load_model_metadata_file(model_id).await?;
        
        // Validate UUID matches
        if metadata_file.model_id != *model_id {
            return Err(anyhow!(
                "UUID mismatch: expected {}, found {} in model.json",
                model_id, metadata_file.model_id
            ));
        }
        
        // Build full metadata
        let data_path = model_path.join("data");
        let size_bytes = if data_path.exists() {
            self.calculate_model_size(&data_path).await.unwrap_or(0)
        } else {
            0
        };
        
        let metadata = ModelMetadata {
            model_id: model_id.clone(),
            name: metadata_file.name.clone(),
            display_name: Some(metadata_file.display_name.clone()),
            architecture: metadata_file.architecture.unwrap_or_else(|| "unknown".to_string()),
            parameters: metadata_file.parameters,
            model_type: "language_model".to_string(),
            tokenizer_type: None,
            size_bytes,
            files: vec![],
            external_sources: if metadata_file.source_uri.starts_with("hf://") {
                vec![ExternalSource {
                    source_type: SourceType::HuggingFace,
                    identifier: metadata_file.source_uri.trim_start_matches("hf://").to_string(),
                    revision: None,
                    download_url: None,
                    last_verified: metadata_file.created_at,
                }]
            } else {
                vec![]
            },
            local_path: Some(data_path),
            is_cached: true,
            tags: vec![],
            description: None,
            license: None,
            created_at: metadata_file.created_at,
            last_accessed: metadata_file.last_accessed,
            last_updated: metadata_file.created_at,
        };
        
        // Update cache
        {
            let mut cache = self.metadata_cache.write().await;
            cache.insert(model_id.clone(), metadata.clone());
        }
        
        Ok(metadata)
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
    
    /// Repair metadata by scanning directories and fixing inconsistencies
    pub async fn repair_metadata(&self) -> Result<()> {
        tracing::info!("Starting metadata repair...");
        
        let mut repaired_count = 0;
        let mut removed_count = 0;
        
        // Scan all directories
        match tokio::fs::read_dir(&self.base_dir).await {
            Ok(mut entries) => {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    
                    if !path.is_dir() {
                        continue;
                    }
                    
                    let dir_name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");
                    
                    // Check if this is a UUID directory
                    if let Ok(dir_uuid) = Uuid::parse_str(dir_name) {
                        let model_id = ModelId(dir_uuid);
                        let model_json_path = path.join("model.json");
                        
                        if model_json_path.exists() {
                            // Load and verify model.json
                            match self.load_model_metadata_file(&model_id).await {
                                Ok(metadata_file) => {
                                    // Check for UUID mismatch
                                    if metadata_file.model_id.0 != dir_uuid {
                                        tracing::warn!(
                                            "UUID mismatch: directory {} contains metadata for model {}",
                                            dir_uuid, metadata_file.model_id
                                        );
                                        
                                        // Fix the metadata file
                                        let fixed_metadata = ModelMetadataFile {
                                            model_id: model_id.clone(),
                                            ..metadata_file
                                        };
                                        
                                        self.save_model_metadata_file(&model_id, &fixed_metadata).await?;
                                        tracing::info!("Repaired metadata for model {}", dir_uuid);
                                        repaired_count += 1;
                                    }
                                    
                                    // Check for data directory and migrate if needed
                                    let data_path = path.join("data");
                                    if !data_path.exists() {
                                        // Look for model files in the root directory
                                        let has_model_files = Self::has_model_files(&path).await;
                                        
                                        if has_model_files {
                                            tracing::info!("Migrating model files to data/ subdirectory for {}", dir_uuid);
                                            
                                            // Create data directory
                                            fs::create_dir_all(&data_path).await?;
                                            
                                            // Move model files to data directory
                                            let mut dir_entries = fs::read_dir(&path).await?;
                                            while let Some(entry) = dir_entries.next_entry().await? {
                                                let file_name = entry.file_name();
                                                let file_str = file_name.to_string_lossy();
                                                
                                                // Skip model.json and directories
                                                if file_str == "model.json" || entry.file_type().await?.is_dir() {
                                                    continue;
                                                }
                                                
                                                // Move model files (safetensors, bin, gguf, config.json, etc.)
                                                if file_str.ends_with(".safetensors") ||
                                                   file_str.ends_with(".bin") ||
                                                   file_str.ends_with(".gguf") ||
                                                   file_str.ends_with(".ggml") ||
                                                   file_str == "config.json" ||
                                                   file_str == "tokenizer.json" ||
                                                   file_str == "tokenizer_config.json" ||
                                                   file_str == "special_tokens_map.json" ||
                                                   file_str == "vocab.json" ||
                                                   file_str == "merges.txt" {
                                                    let src = entry.path();
                                                    let dst = data_path.join(&file_name);
                                                    
                                                    match fs::rename(&src, &dst).await {
                                                        Ok(_) => {
                                                            tracing::debug!("Moved {} to data/", file_str);
                                                        }
                                                        Err(e) => {
                                                            tracing::error!("Failed to move {}: {}", file_str, e);
                                                        }
                                                    }
                                                }
                                            }
                                            
                                            repaired_count += 1;
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Failed to load model.json for {}: {}", dir_uuid, e);
                                }
                            }
                        } else {
                            // Directory exists but no model.json - might be orphaned
                            tracing::warn!("Directory {} has no model.json, might be orphaned", dir_uuid);
                            
                            // Check if it has any model files
                            if !Self::has_model_files(&path).await {
                                tracing::info!("Removing empty directory {}", dir_uuid);
                                match fs::remove_dir_all(&path).await {
                                    Ok(_) => removed_count += 1,
                                    Err(e) => tracing::error!("Failed to remove directory: {}", e),
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                return Err(anyhow!("Failed to scan models directory: {}", e));
            }
        }
        
        // Rebuild cache from corrected directories
        let models = self.children().await?;
        
        tracing::info!(
            "Metadata repair complete: {} repaired, {} removed, {} total models",
            repaired_count, removed_count, models.len()
        );
        
        Ok(())
    }
    
    /// List all models in storage (directories are the source of truth)
    pub async fn children(&self) -> Result<Vec<(ModelId, ModelMetadata)>> {
        let mut models = Vec::new();
        let mut found_uuids = std::collections::HashSet::new();
        
        tracing::info!("Listing models from directory: {:?}", self.base_dir);
        
        // Scan the models directory - this is our single source of truth
        match tokio::fs::read_dir(&self.base_dir).await {
            Ok(mut entries) => {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    
                    if !path.is_dir() {
                        continue;
                    }
                    
                    let dir_name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");
                    
                    // Skip non-UUID directories
                    if let Ok(dir_uuid) = Uuid::parse_str(dir_name) {
                        let model_id = ModelId(dir_uuid);
                        let model_json_path = path.join("model.json");
                        
                        // Check if this is a valid model directory
                        if model_json_path.exists() {
                            // Load model metadata from model.json
                            if let Ok(metadata_file) = self.load_model_metadata_file(&model_id).await {
                                // Validate that the UUID in metadata matches the directory name
                                if metadata_file.model_id.0 != dir_uuid {
                                    tracing::warn!(
                                        "UUID mismatch: directory {} contains metadata for model {}",
                                        dir_uuid, metadata_file.model_id
                                    );
                                    // Skip this corrupted entry
                                    continue;
                                }
                                
                                // Check if data directory exists
                                let data_path = path.join("data");
                                let has_data = data_path.exists();
                                
                                // Calculate actual size if data exists
                                let size_bytes = if has_data {
                                    self.calculate_model_size(&data_path).await.unwrap_or(0)
                                } else {
                                    0
                                };
                                
                                // Convert to full metadata
                                let metadata = ModelMetadata {
                                    model_id: model_id.clone(),
                                    name: metadata_file.name.clone(),
                                    display_name: Some(metadata_file.display_name.clone()),
                                    architecture: metadata_file.architecture.unwrap_or_else(|| "unknown".to_string()),
                                    parameters: metadata_file.parameters,
                                    model_type: "language_model".to_string(),
                                    tokenizer_type: None,
                                    size_bytes,
                                    files: vec![], // Will be populated from data directory if needed
                                    external_sources: if metadata_file.source_uri.starts_with("hf://") {
                                        vec![ExternalSource {
                                            source_type: SourceType::HuggingFace,
                                            identifier: metadata_file.source_uri.trim_start_matches("hf://").to_string(),
                                            revision: None,
                                            download_url: None,
                                            last_verified: metadata_file.created_at,
                                        }]
                                    } else {
                                        vec![]
                                    },
                                    local_path: Some(data_path),
                                    is_cached: has_data,
                                    tags: vec![],
                                    description: None,
                                    license: None,
                                    created_at: metadata_file.created_at,
                                    last_accessed: metadata_file.last_accessed,
                                    last_updated: metadata_file.created_at,
                                };
                                
                                models.push((model_id, metadata));
                                found_uuids.insert(dir_uuid);
                                
                                tracing::debug!("Found valid model: {} ({})", dir_uuid, metadata_file.display_name);
                            } else {
                                tracing::warn!("Failed to load model.json for directory: {}", dir_uuid);
                            }
                        } else {
                            tracing::debug!("Directory {} has no model.json, skipping", dir_uuid);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to read models directory {:?}: {}", self.base_dir, e);
                return Err(anyhow!("Failed to list models: {}", e));
            }
        }
        
        // Update the metadata cache with what we found (cache-only, not authoritative)
        {
            let mut cache = self.metadata_cache.write().await;
            cache.clear();
            for (id, metadata) in &models {
                cache.insert(id.clone(), metadata.clone());
            }
        }
        
        // Save the updated cache
        self.save_metadata().await?;
        
        tracing::info!("Found {} valid models", models.len());
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio;

    /// Helper to create a test storage instance
    async fn create_test_storage() -> (ModelStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = ModelStorage::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();
        (storage, temp_dir)
    }

    /// Helper to create a valid model metadata file
    fn create_test_metadata() -> ModelMetadataFile {
        ModelMetadataFile {
            model_id: ModelId::new(),
            name: "test-model".to_string(),
            display_name: "Test Model".to_string(),
            source_uri: "hf://test/model".to_string(),
            architecture: Some("transformer".to_string()),
            parameters: Some(1_000_000),
            created_at: chrono::Utc::now().timestamp(),
            last_accessed: chrono::Utc::now().timestamp(),
        }
    }

    #[tokio::test]
    async fn test_save_and_load_model_metadata() {
        let (storage, _temp_dir) = create_test_storage().await;
        let metadata = create_test_metadata();
        let model_id = metadata.model_id.clone();

        // Save metadata
        storage.save_model_metadata_file(&model_id, &metadata)
            .await
            .expect("Failed to save metadata");

        // Load metadata
        let loaded = storage.load_model_metadata_file(&model_id)
            .await
            .expect("Failed to load metadata");

        assert_eq!(loaded.model_id, metadata.model_id);
        assert_eq!(loaded.name, metadata.name);
        assert_eq!(loaded.display_name, metadata.display_name);
    }

    #[tokio::test]
    async fn test_json_validation_max_depth() {
        // Create deeply nested JSON that should be rejected
        let mut deep_json = String::from("{");
        for i in 0..20 {
            deep_json.push_str(&format!("\"level{}\":{{", i));
        }
        deep_json.push_str("\"data\":\"test\"");
        for _ in 0..20 {
            deep_json.push_str("}");
        }
        deep_json.push_str("}");

        // Try to validate with json_threat_protection
        let result = jtp::from_str(&deep_json)
            .with_max_depth(10)
            .validate();

        assert!(result.is_err(), "Should reject deeply nested JSON");
    }

    #[tokio::test]
    async fn test_json_validation_max_string_length() {
        let long_string = "x".repeat(100_000);
        let json = format!("{{\"data\":\"{}\"}}", long_string);

        // Should reject strings that are too long
        let result = jtp::from_str(&json)
            .with_max_string_length(10_000)
            .validate();

        assert!(result.is_err(), "Should reject JSON with overly long strings");
    }

    #[tokio::test]
    async fn test_json_validation_max_array_entries() {
        let large_array: Vec<i32> = (0..10_000).collect();
        let json = serde_json::to_string(&large_array).unwrap();

        // Should reject arrays that are too large
        let result = jtp::from_str(&json)
            .with_max_array_entries(1_000)
            .validate();

        assert!(result.is_err(), "Should reject JSON with too many array entries");
    }

    #[tokio::test]
    async fn test_uuid_validation() {
        let (storage, _temp_dir) = create_test_storage().await;
        
        // Test various UUID versions are all accepted
        
        // Valid UUID v4 (random)
        let valid_id = ModelId::new();
        let valid_path = storage.get_uuid_model_path(&valid_id);
        assert!(valid_path.to_string_lossy().contains(&valid_id.0.to_string()));
        assert!(!valid_path.to_string_lossy().contains("invalid_uuid"));

        // Valid UUID v3 format (MD5 hash based)
        let uuid_v3 = Uuid::parse_str("f9651c03-3f15-38ab-f965-1c033f1538ab").unwrap();
        let v3_id = ModelId(uuid_v3);
        let v3_path = storage.get_uuid_model_path(&v3_id);
        assert!(v3_path.to_string_lossy().contains(&uuid_v3.to_string()));
        assert!(!v3_path.to_string_lossy().contains("invalid_uuid"));
        
        // Nil UUID (all zeros) should also be valid
        let nil_id = ModelId(Uuid::nil());
        let nil_path = storage.get_uuid_model_path(&nil_id);
        assert!(nil_path.to_string_lossy().contains(&Uuid::nil().to_string()));
        assert!(!nil_path.to_string_lossy().contains("invalid_uuid"));
        
        // Max UUID (all ones) should be valid
        let max_uuid = Uuid::parse_str("ffffffff-ffff-ffff-ffff-ffffffffffff").unwrap();
        let max_id = ModelId(max_uuid);
        let max_path = storage.get_uuid_model_path(&max_id);
        assert!(max_path.to_string_lossy().contains(&max_uuid.to_string()));
        assert!(!max_path.to_string_lossy().contains("invalid_uuid"));
    }

    #[tokio::test]
    async fn test_children_single_source_of_truth() {
        let (storage, temp_dir) = create_test_storage().await;
        
        // Create a model directory with model.json
        let model_id = ModelId::new();
        let model_dir = temp_dir.path().join(model_id.0.to_string());
        tokio::fs::create_dir_all(&model_dir).await.unwrap();
        
        let metadata = ModelMetadataFile {
            model_id: model_id.clone(),
            name: "test-model".to_string(),
            display_name: "Test Model".to_string(),
            source_uri: "hf://test/model".to_string(),
            architecture: Some("transformer".to_string()),
            parameters: Some(1_000_000),
            created_at: chrono::Utc::now().timestamp(),
            last_accessed: chrono::Utc::now().timestamp(),
        };
        
        // Save model.json in the directory
        let model_json_path = model_dir.join("model.json");
        let content = serde_json::to_string_pretty(&metadata).unwrap();
        tokio::fs::write(&model_json_path, content).await.unwrap();
        
        // List models - should find the model
        let models = storage.children().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].0, model_id);
        assert_eq!(models[0].1.name, "test-model");
    }

    #[tokio::test]
    async fn test_uuid_mismatch_detection() {
        let (storage, temp_dir) = create_test_storage().await;
        
        // Create a model directory with one UUID
        let dir_uuid = ModelId::new();
        let model_dir = temp_dir.path().join(dir_uuid.0.to_string());
        tokio::fs::create_dir_all(&model_dir).await.unwrap();
        
        // But save metadata with a different UUID
        let wrong_uuid = ModelId::new();
        let metadata = ModelMetadataFile {
            model_id: wrong_uuid, // Wrong UUID!
            name: "test-model".to_string(),
            display_name: "Test Model".to_string(),
            source_uri: "hf://test/model".to_string(),
            architecture: Some("transformer".to_string()),
            parameters: Some(1_000_000),
            created_at: chrono::Utc::now().timestamp(),
            last_accessed: chrono::Utc::now().timestamp(),
        };
        
        // Save model.json with wrong UUID
        let model_json_path = model_dir.join("model.json");
        let content = serde_json::to_string_pretty(&metadata).unwrap();
        tokio::fs::write(&model_json_path, content).await.unwrap();
        
        // List models - should detect mismatch and skip this model
        let models = storage.children().await.unwrap();
        assert_eq!(models.len(), 0, "Should skip models with UUID mismatch");
    }

    #[tokio::test]
    async fn test_data_subdirectory_structure() {
        let (storage, _temp_dir) = create_test_storage().await;
        
        let model_id = ModelId::new();
        let model_path = storage.get_uuid_model_path(&model_id);
        let data_path = storage.get_uuid_model_data_path(&model_id);
        
        assert_eq!(data_path, model_path.join("data"));
    }

    #[tokio::test]
    async fn test_metadata_repair() {
        let (storage, temp_dir) = create_test_storage().await;
        
        // Create a model directory with mismatched UUID
        let dir_uuid = ModelId::new();
        let model_dir = temp_dir.path().join(dir_uuid.0.to_string());
        tokio::fs::create_dir_all(&model_dir).await.unwrap();
        
        // Save metadata with wrong UUID
        let wrong_uuid = ModelId::new();
        let metadata = ModelMetadataFile {
            model_id: wrong_uuid,
            name: "test-model".to_string(),
            display_name: "Test Model".to_string(),
            source_uri: "hf://test/model".to_string(),
            architecture: Some("transformer".to_string()),
            parameters: Some(1_000_000),
            created_at: chrono::Utc::now().timestamp(),
            last_accessed: chrono::Utc::now().timestamp(),
        };
        
        let model_json_path = model_dir.join("model.json");
        let content = serde_json::to_string_pretty(&metadata).unwrap();
        tokio::fs::write(&model_json_path, content).await.unwrap();
        
        // Run repair
        storage.repair_metadata().await.unwrap();
        
        // Load the repaired metadata
        let repaired = storage.load_model_metadata_file(&dir_uuid).await.unwrap();
        assert_eq!(repaired.model_id, dir_uuid, "UUID should be fixed after repair");
    }

    #[tokio::test]
    async fn test_direct_loading_without_cache() {
        let (storage, temp_dir) = create_test_storage().await;
        
        // Create a model directory
        let model_id = ModelId::new();
        let model_dir = temp_dir.path().join(model_id.0.to_string());
        let data_dir = model_dir.join("data");
        tokio::fs::create_dir_all(&data_dir).await.unwrap();
        
        // Save model.json
        let metadata_file = ModelMetadataFile {
            model_id: model_id.clone(),
            name: "test-model".to_string(),
            display_name: "Test Model".to_string(),
            source_uri: "hf://test/model".to_string(),
            architecture: Some("transformer".to_string()),
            parameters: Some(1_000_000),
            created_at: chrono::Utc::now().timestamp(),
            last_accessed: chrono::Utc::now().timestamp(),
        };
        
        let model_json_path = model_dir.join("model.json");
        let content = serde_json::to_string_pretty(&metadata_file).unwrap();
        tokio::fs::write(&model_json_path, content).await.unwrap();
        
        // Load directly without going through cache
        let loaded = storage.get_metadata_by_id_or_load(&model_id).await.unwrap();
        assert_eq!(loaded.model_id, model_id);
        assert_eq!(loaded.name, "test-model");
    }
}