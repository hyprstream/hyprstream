//! Model management API with URI-based registry support

use axum::{
    Router,
    routing::{get, post, delete},
    extract::{State, Json, Query},
    response::Json as JsonResponse,
    http::StatusCode,
};
use std::collections::HashMap;
use std::path::{Path as StdPath, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use url::Url;

use crate::api::model_registry::{ModelRegistry, ModelRegistryType};
use crate::api::model_storage::{ModelStorage, ModelMetadata, ModelId, ExternalSource, SourceType, ModelFile, FileType};
use crate::config::HyprConfig;

/// Model management service state
#[derive(Clone)]
pub struct ModelManagementState {
    /// Local model storage
    storage: Arc<ModelStorage>,
    
    /// Registry clients (HuggingFace, etc.)
    registries: Arc<RwLock<HashMap<ModelRegistryType, Box<dyn ModelRegistry + Send + Sync>>>>,
    
    
    /// Configuration
    config: ModelManagementConfig,
}

/// Type alias for backwards compatibility
pub type ModelManager = ModelManagementState;

impl ModelManagementState {
    /// Create new model management state
    pub async fn new() -> Result<Self> {
        let config = ModelManagementConfig::default();
        let storage = Arc::new(ModelStorage::new(config.models_dir.clone()).await?);
        let registries = Arc::new(RwLock::new(HashMap::new()));
        
        Ok(Self {
            storage,
            registries,
            config,
        })
    }
    
    /// List cached models (compatibility method)
    pub async fn list_cached_models(&self) -> Result<Vec<(ModelUri, ModelMetadata)>> {
        self.storage.list_local_models().await
    }
    
    /// Get cache statistics (compatibility method)  
    pub async fn get_cache_stats(&self) -> Result<crate::api::model_storage::CacheStats> {
        self.storage.get_cache_stats().await
    }
    
    /// Remove model metadata
    pub async fn remove_metadata(&self, model_uri: &ModelUri) -> Result<()> {
        self.storage.remove_metadata(model_uri).await
    }
    
    /// Remove model metadata by ID
    pub async fn remove_metadata_by_id(&self, model_id: &crate::api::model_storage::ModelId) -> Result<()> {
        self.storage.remove_metadata_by_id(model_id).await
    }
}

/// Configuration for model management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManagementConfig {
    /// Base directory for model storage
    pub models_dir: PathBuf,
    
    /// Maximum concurrent downloads
    pub max_concurrent_downloads: usize,
    
    /// Cache size limit in GB
    pub cache_size_gb: u64,
    
    /// Auto-cleanup old models
    pub auto_cleanup: bool,
    
    /// Hugging Face token (optional)
    pub hf_token: Option<String>,
}

impl Default for ModelManagementConfig {
    fn default() -> Self {
        let config = HyprConfig::load().unwrap_or_default();
        
        Self {
            models_dir: config.models_dir().clone(),
            max_concurrent_downloads: 3,
            cache_size_gb: 100,
            auto_cleanup: true,
            hf_token: None,
        }
    }
}

/// Model URI for registry-agnostic model identification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelUri {
    /// Registry type (hf, ollama, custom, etc.)
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
    /// Formats:
    /// - hf://microsoft/DialoGPT-medium
    /// - hf://microsoft/DialoGPT-medium@main
    /// - ollama://llama2:7b
    /// - custom://company.com/model-name
    pub fn parse(uri: &str) -> Result<Self> {
        let url = Url::parse(uri)
            .map_err(|e| anyhow::anyhow!("Invalid URI format: {}", e))?;
        
        let registry = url.scheme().to_string();
        let host = url.host_str().unwrap_or("");
        let path = url.path().trim_start_matches('/');
        
        // Debug logging to understand parsing
        eprintln!("DEBUG: Parsing URI: {}", uri);
        eprintln!("DEBUG: scheme={}, host={:?}, path={:?}", registry, host, path);
        
        let (org, name, revision) = match registry.as_str() {
            "hf" => {
                // hf://microsoft/DialoGPT-medium@main
                let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
                eprintln!("DEBUG: parts={:?}, len={}", parts, parts.len());
                
                let (org, name_part) = if parts.len() >= 2 {
                    // Standard case: hf://org/name
                    (parts[0], parts[1])
                } else if !host.is_empty() && parts.len() >= 1 {
                    // Host-as-org case: hf://org/name where org becomes host
                    eprintln!("DEBUG: Using host-as-org parsing");
                    (host, parts[0])
                } else if !host.is_empty() && parts.is_empty() {
                    // Case where everything after hf:// is treated as host
                    eprintln!("DEBUG: Host contains full path, trying to split");
                    let host_parts: Vec<&str> = host.split('/').collect();
                    if host_parts.len() >= 2 {
                        (host_parts[0], host_parts[1])
                    } else {
                        return Err(anyhow::anyhow!("HuggingFace URI must have org/name format. Got host={:?}, path={:?}", host, path));
                    }
                } else {
                    return Err(anyhow::anyhow!("HuggingFace URI must have org/name format. Got host={:?}, path={:?}, parts={:?}", host, path, parts));
                };
                
                // Handle revision parsing
                if let Some(at_pos) = name_part.find('@') {
                    let (name, rev) = name_part.split_at(at_pos);
                    (org.to_string(), name.to_string(), Some(rev[1..].to_string()))
                } else {
                    (org.to_string(), name_part.to_string(), None)
                }
            }
            "ollama" => {
                // ollama://llama2:7b
                let model_spec = if host.is_empty() { path } else { host };
                if let Some(colon_pos) = model_spec.find(':') {
                    let (name, tag) = model_spec.split_at(colon_pos);
                    ("ollama".to_string(), name.to_string(), Some(tag[1..].to_string()))
                } else {
                    ("ollama".to_string(), model_spec.to_string(), None)
                }
            }
            _ => {
                // custom://company.com/model-name
                let org = host.to_string();
                let name = path.to_string();
                (org, name, None)
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
    
    /// Get local storage path for this model
    pub fn local_path(&self, base_dir: &StdPath) -> PathBuf {
        let mut path = base_dir.join(&self.registry).join(&self.org).join(&self.name);
        if let Some(rev) = &self.revision {
            path = path.join(rev);
        }
        path
    }
}

/// Request to pull a model
#[derive(Debug, Deserialize)]
pub struct PullModelRequest {
    /// Model URI (e.g., "hf://microsoft/DialoGPT-medium")
    pub uri: String,
    
    /// Force re-download even if cached
    #[serde(default)]
    pub force: bool,
    
    /// Include specific files (optional)
    pub files: Option<Vec<String>>,
}

/// Response from pulling a model
#[derive(Debug, Serialize)]
pub struct PullModelResponse {
    pub uri: String,
    pub status: String,
    pub local_path: String,
    pub size_bytes: u64,
    pub files_downloaded: Vec<String>,
    pub download_time_ms: u64,
}

/// Model listing query parameters
#[derive(Debug, Deserialize)]
pub struct ListModelsQuery {
    /// Filter by registry type
    pub registry: Option<String>,
    
    /// Search query
    pub search: Option<String>,
    
    /// Include remote models (not just local)
    #[serde(default)]
    pub include_remote: bool,
}

/// Model information
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub uri: String,
    pub local_path: Option<String>,
    pub size_bytes: Option<u64>,
    pub files: Vec<String>,
    pub metadata: ModelMetadata,
    pub is_cached: bool,
    pub last_accessed: Option<i64>,
}

/// Model removal request
#[derive(Debug, Deserialize)]
pub struct RemoveModelRequest {
    /// Keep metadata but remove files
    #[serde(default)]
    pub keep_metadata: bool,
}

/// Create model management router
pub fn create_model_router(state: ModelManagementState) -> Router {
    Router::new()
        // Model operations
        .route("/v1/models/pull", post(pull_model))
        .route("/v1/models/list", get(list_models))
        .route("/v1/models/info", get(get_model_info))
        .route("/v1/models/remove", delete(remove_model))
        
        // Registry operations
        .route("/v1/models/search", get(search_models))
        .route("/v1/models/registries", get(list_registries))
        
        // Cache management
        .route("/v1/models/cache/status", get(cache_status))
        .route("/v1/models/cache/cleanup", post(cleanup_cache))
        
        .with_state(state)
}

/// Pull model from registry
async fn pull_model(
    State(state): State<ModelManagementState>,
    Json(request): Json<PullModelRequest>,
) -> Result<JsonResponse<PullModelResponse>, StatusCode> {
    let start = std::time::Instant::now();
    
    // Parse model URI
    let model_uri = ModelUri::parse(&request.uri)
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    
    println!("📥 Pulling model: {}", model_uri.uri);
    
    // Check if already cached and not forcing
    let local_path = model_uri.local_path(&state.config.models_dir);
    if !request.force && local_path.exists() {
        let metadata = state.storage.get_metadata(&model_uri).await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
        return Ok(JsonResponse(PullModelResponse {
            uri: request.uri,
            status: "cached".to_string(),
            local_path: local_path.to_string_lossy().to_string(),
            size_bytes: metadata.size_bytes,
            files_downloaded: metadata.files.iter().map(|f| f.filename.clone()).collect(),
            download_time_ms: 0,
        }));
    }
    
    // Get appropriate registry client
    let registries = state.registries.read().await;
    let registry_type = ModelRegistryType::from_string(&model_uri.registry);
    let registry = registries.get(&registry_type)
        .ok_or(StatusCode::BAD_REQUEST)?;
    
    // Download model
    let download_result = registry.download_model(
        &model_uri,
        &local_path,
        request.files.as_deref(),
    ).await.map_err(|e| {
        eprintln!("Download failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    // Generate UUID for the model
    let model_id = ModelId::from_content_hash(
        &model_uri.name,
        &download_result.architecture.clone().unwrap_or_else(|| "unknown".to_string()),
        download_result.parameters,
    );
    
    // Create external source
    let external_source = ExternalSource {
        source_type: match model_uri.registry.as_str() {
            "hf" | "huggingface" => SourceType::HuggingFace,
            "ollama" => SourceType::Ollama,
            _ => SourceType::Custom(model_uri.registry.clone()),
        },
        identifier: format!("{}/{}", model_uri.org, model_uri.name),
        revision: model_uri.revision.clone(),
        download_url: None,
        last_verified: chrono::Utc::now().timestamp(),
    };
    
    // Convert files to ModelFile structs
    let model_files: Vec<ModelFile> = download_result.files.iter().map(|filename| ModelFile {
        filename: filename.clone(),
        size_bytes: 0, // TODO: Get individual file sizes
        checksum: None,
        file_type: if filename.ends_with(".gguf") {
            FileType::Model
        } else if filename.contains("tokenizer") {
            FileType::Tokenizer
        } else if filename.contains("config") {
            FileType::Config
        } else if filename.contains("README") {
            FileType::Readme
        } else {
            FileType::Other
        },
    }).collect();
    
    // Store metadata with UUID system
    let metadata = ModelMetadata {
        model_id,
        name: model_uri.name.clone(),
        display_name: None,
        architecture: download_result.architecture.unwrap_or_else(|| "unknown".to_string()),
        parameters: download_result.parameters,
        model_type: download_result.model_type,
        tokenizer_type: download_result.tokenizer_type,
        size_bytes: download_result.size_bytes,
        files: model_files,
        external_sources: vec![external_source],
        local_path: Some(local_path.to_path_buf()),
        is_cached: true,
        tags: download_result.tags,
        description: None,
        license: None,
        created_at: chrono::Utc::now().timestamp(),
        last_accessed: chrono::Utc::now().timestamp(),
        last_updated: chrono::Utc::now().timestamp(),
    };
    
    state.storage.store_metadata(&model_uri, metadata).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let download_time = start.elapsed().as_millis() as u64;
    
    println!("✅ Model pulled successfully in {}ms: {}", download_time, model_uri.uri);
    
    Ok(JsonResponse(PullModelResponse {
        uri: request.uri,
        status: "downloaded".to_string(),
        local_path: local_path.to_string_lossy().to_string(),
        size_bytes: download_result.size_bytes,
        files_downloaded: download_result.files,
        download_time_ms: download_time,
    }))
}

/// List available models
async fn list_models(
    State(state): State<ModelManagementState>,
    Query(query): Query<ListModelsQuery>,
) -> Result<JsonResponse<Vec<ModelInfo>>, StatusCode> {
    let mut models = Vec::new();
    
    // Get local models
    let local_models = state.storage.list_local_models().await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    for (uri, metadata) in local_models {
        if let Some(registry_filter) = &query.registry {
            if uri.registry != *registry_filter {
                continue;
            }
        }
        
        if let Some(search) = &query.search {
            if !uri.name.contains(search) && !uri.org.contains(search) {
                continue;
            }
        }
        
        let local_path = uri.local_path(&state.config.models_dir);
        models.push(ModelInfo {
            uri: uri.uri.clone(),
            local_path: Some(local_path.to_string_lossy().to_string()),
            size_bytes: Some(metadata.size_bytes),
            files: metadata.files.iter().map(|f| f.filename.clone()).collect(),
            metadata: metadata.clone(),
            is_cached: true,
            last_accessed: Some(chrono::Utc::now().timestamp()),
        });
    }
    
    // Include remote models if requested
    if query.include_remote {
        if let Some(registry_filter) = &query.registry {
            let registries = state.registries.read().await;
            let registry_type = ModelRegistryType::from_string(registry_filter);
            
            if let Some(registry) = registries.get(&registry_type) {
                if let Ok(remote_models) = registry.search_models(
                    query.search.as_deref(),
                    Some(20)
                ).await {
                    for remote_model in remote_models {
                        // Skip if already in local list
                        if models.iter().any(|m| m.uri == remote_model.uri) {
                            continue;
                        }
                        
                        models.push(ModelInfo {
                            uri: remote_model.uri,
                            local_path: None,
                            size_bytes: remote_model.size_bytes,
                            files: remote_model.files,
                            metadata: remote_model.metadata,
                            is_cached: false,
                            last_accessed: None,
                        });
                    }
                }
            }
        }
    }
    
    Ok(JsonResponse(models))
}

/// Get detailed model information
async fn get_model_info(
    State(state): State<ModelManagementState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<JsonResponse<ModelInfo>, StatusCode> {
    let uri_str = params.get("uri")
        .ok_or(StatusCode::BAD_REQUEST)?;
    
    let model_uri = ModelUri::parse(uri_str)
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    
    // Check local storage first
    if let Ok(metadata) = state.storage.get_metadata(&model_uri).await {
        let local_path = model_uri.local_path(&state.config.models_dir);
        
        return Ok(JsonResponse(ModelInfo {
            uri: model_uri.uri,
            local_path: Some(local_path.to_string_lossy().to_string()),
            size_bytes: Some(metadata.size_bytes),
            files: metadata.files.iter().map(|f| f.filename.clone()).collect(),
            metadata: metadata.clone(),
            is_cached: true,
            last_accessed: Some(chrono::Utc::now().timestamp()),
        }));
    }
    
    // Query remote registry
    let registries = state.registries.read().await;
    let registry_type = ModelRegistryType::from_string(&model_uri.registry);
    let registry = registries.get(&registry_type)
        .ok_or(StatusCode::NOT_FOUND)?;
    
    let remote_info = registry.get_model_info(&model_uri).await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    
    Ok(JsonResponse(ModelInfo {
        uri: model_uri.uri,
        local_path: None,
        size_bytes: remote_info.size_bytes,
        files: remote_info.files,
        metadata: remote_info.metadata,
        is_cached: false,
        last_accessed: None,
    }))
}

/// Remove model from local storage
async fn remove_model(
    State(state): State<ModelManagementState>,
    Query(params): Query<HashMap<String, String>>,
    Json(request): Json<RemoveModelRequest>,
) -> Result<JsonResponse<serde_json::Value>, StatusCode> {
    let uri_str = params.get("uri")
        .ok_or(StatusCode::BAD_REQUEST)?;
    
    let model_uri = ModelUri::parse(uri_str)
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    
    println!("🗑️ Removing model: {}", model_uri.uri);
    
    // Remove files
    let local_path = model_uri.local_path(&state.config.models_dir);
    if local_path.exists() {
        tokio::fs::remove_dir_all(&local_path).await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    
    // Remove metadata if not keeping
    if !request.keep_metadata {
        state.storage.remove_metadata(&model_uri).await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    
    Ok(JsonResponse(serde_json::json!({
        "status": "removed",
        "uri": model_uri.uri,
        "kept_metadata": request.keep_metadata
    })))
}

/// Search models across registries
async fn search_models(
    State(state): State<ModelManagementState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<JsonResponse<Vec<ModelInfo>>, StatusCode> {
    let query = params.get("q")
        .ok_or(StatusCode::BAD_REQUEST)?;
    
    let registry_filter = params.get("registry");
    let limit: usize = params.get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    
    let mut all_results = Vec::new();
    let registries = state.registries.read().await;
    
    for (registry_type, registry) in registries.iter() {
        if let Some(filter) = registry_filter {
            if registry_type.to_string() != *filter {
                continue;
            }
        }
        
        if let Ok(results) = registry.search_models(Some(query), Some(limit)).await {
            for result in results {
                all_results.push(ModelInfo {
                    uri: result.uri,
                    local_path: None,
                    size_bytes: result.size_bytes,
                    files: result.files,
                    metadata: result.metadata,
                    is_cached: false,
                    last_accessed: None,
                });
            }
        }
    }
    
    // Limit total results
    all_results.truncate(limit);
    
    Ok(JsonResponse(all_results))
}

/// List available registries
async fn list_registries(
    State(state): State<ModelManagementState>,
) -> Result<JsonResponse<Vec<serde_json::Value>>, StatusCode> {
    let registries = state.registries.read().await;
    let registry_list: Vec<_> = registries.keys()
        .map(|r| serde_json::json!({
            "name": r.to_string(),
            "type": r.to_string(),
            "available": true
        }))
        .collect();
    
    Ok(JsonResponse(registry_list))
}

/// Get cache status
async fn cache_status(
    State(state): State<ModelManagementState>,
) -> Result<JsonResponse<serde_json::Value>, StatusCode> {
    let stats = state.storage.get_cache_stats().await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(JsonResponse(serde_json::json!({
        "total_models": stats.total_models,
        "total_size_bytes": stats.total_size_bytes,
        "total_size_gb": stats.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        "cache_limit_gb": state.config.cache_size_gb,
        "usage_percent": (stats.total_size_bytes as f64 / (state.config.cache_size_gb as f64 * 1024.0 * 1024.0 * 1024.0)) * 100.0,
        "models_by_registry": stats.models_by_registry,
    })))
}

/// Cleanup old cached models
async fn cleanup_cache(
    State(state): State<ModelManagementState>,
) -> Result<JsonResponse<serde_json::Value>, StatusCode> {
    let cleanup_result = state.storage.cleanup_old_models(
        state.config.cache_size_gb * 1024 * 1024 * 1024, // Convert to bytes
    ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(JsonResponse(serde_json::json!({
        "status": "completed",
        "models_removed": cleanup_result.models_removed,
        "bytes_freed": cleanup_result.bytes_freed,
        "models_remaining": cleanup_result.models_remaining,
    })))
}

use chrono;