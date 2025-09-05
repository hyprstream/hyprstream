//! Model registry abstraction for different model sources

use crate::api::model_storage::ModelUri;
use crate::api::model_storage::ModelMetadata;

use std::path::Path;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Types of model registries
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelRegistryType {
    HuggingFace,
    Custom(String),
}

impl ModelRegistryType {
    pub fn from_string(s: &str) -> Self {
        match s {
            "hf" | "huggingface" => Self::HuggingFace,
            _ => Self::Custom(s.to_string()),
        }
    }
    
    pub fn to_string(&self) -> String {
        match self {
            Self::HuggingFace => "hf".to_string(),
            Self::Custom(name) => name.clone(),
        }
    }
}

/// Result from downloading a model
#[derive(Debug, Clone)]
pub struct DownloadResult {
    pub size_bytes: u64,
    pub files: Vec<String>,
    pub model_type: String,
    pub architecture: Option<String>,
    pub parameters: Option<u64>,
    pub tokenizer_type: Option<String>,
    pub tags: Vec<String>,
}

/// Model search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub uri: String,
    pub size_bytes: Option<u64>,
    pub files: Vec<String>,
    pub metadata: ModelMetadata,
}

/// Model info from registry
#[derive(Debug, Clone)]
pub struct RegistryModelInfo {
    pub size_bytes: Option<u64>,
    pub files: Vec<String>,
    pub metadata: ModelMetadata,
}

/// Model info for search results and API responses
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub id: String,
    pub description: String,
    pub size_bytes: Option<u64>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Vec<String>,
    pub architecture: Option<String>,
    pub task: Option<String>,
    pub library_name: Option<String>,
    pub created_at: i64,
    pub last_modified: Option<String>,
}

/// Abstract model registry trait
#[async_trait]
pub trait ModelRegistry {
    /// Download a model to local storage
    async fn download_model(
        &self,
        model_uri: &ModelUri,
        local_path: &Path,
        files: Option<&[String]>,
    ) -> Result<DownloadResult>;
    
    /// Get information about a model without downloading
    async fn get_model_info(&self, model_uri: &ModelUri) -> Result<RegistryModelInfo>;
    
    /// Search for models in the registry
    async fn search_models(
        &self,
        query: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<SearchResult>>;
    
    /// Check if a model exists in the registry
    async fn model_exists(&self, model_uri: &ModelUri) -> Result<bool>;
    
    /// Get download URL for a specific file
    async fn get_download_url(
        &self,
        model_uri: &ModelUri,
        filename: &str,
    ) -> Result<String>;
    
    /// List files in a model repository
    async fn list_model_files(&self, model_uri: &ModelUri) -> Result<Vec<ModelFileInfo>>;
}

/// Information about a file in a model repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFileInfo {
    pub filename: String,
    pub size_bytes: Option<u64>,
    pub sha: Option<String>,
    pub lfs: bool, // Git LFS file
}

/// Download progress callback
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Registry configuration
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// API token for authentication
    pub token: Option<String>,
    
    /// Base URL for API requests
    pub base_url: String,
    
    /// Timeout for requests in seconds
    pub timeout_secs: u64,
    
    /// Maximum number of retries
    pub max_retries: u32,
    
    /// User agent string
    pub user_agent: String,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            token: None,
            base_url: "https://huggingface.co".to_string(),
            timeout_secs: 300, // 5 minutes
            max_retries: 3,
            user_agent: "hyprstream/0.1.0".to_string(),
        }
    }
}

/// Error types for model registry operations
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Authentication failed")]
    AuthenticationFailed,
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Invalid model URI: {0}")]
    InvalidUri(String),
    
    #[error("Download failed: {0}")]
    DownloadFailed(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("HTTP error: {0}")]
    HttpError(String),
}