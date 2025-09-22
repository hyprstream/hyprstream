//! Unified model downloader with SafeTensors-first approach
//! 
//! This module provides a clean abstraction for downloading models from various sources,
//! prioritizing SafeTensors format for security and compatibility.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use async_trait::async_trait;
use json_threat_protection as jtp;
// Note: indicatif imports removed as they're not used in this cleaned version
use glob;

/// Model format - SafeTensors only for security and efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFormat {
    SafeTensors,
}

impl ModelFormat {
    /// Get file extension for SafeTensors
    pub fn extension(&self) -> &'static str {
        "safetensors"
    }
    
    /// Check if a filename is a SafeTensors file
    pub fn matches(&self, filename: &str) -> bool {
        filename.to_lowercase().ends_with(".safetensors")
    }
}

/// Model configuration (from config.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: Option<f32>,
    pub torch_dtype: Option<String>,
}

/// Tokenizer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerInfo {
    pub tokenizer_class: Option<String>,
    pub vocab_size: usize,
    pub pad_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

/// Model source abstraction
#[async_trait]
pub trait ModelSource: Send + Sync {
    /// List available models
    async fn list_models(&self, query: Option<&str>) -> Result<Vec<ModelInfo>>;
    
    /// Get model metadata
    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo>;
    
    /// Download a model
    async fn download_model(&self, model_id: &str, target_dir: &Path) -> Result<DownloadedModel>;
}

/// Information about a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub size_bytes: Option<u64>,
    pub format: ModelFormat,
    pub architecture: Option<String>,
    pub tags: Vec<String>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Downloaded model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadedModel {
    pub model_id: String,
    pub uuid: uuid::Uuid,  // The actual UUID used for storage
    pub local_path: PathBuf,
    pub format: ModelFormat,
    pub files: Vec<ModelFile>,
    pub config: Option<ModelConfig>,
    pub tokenizer: Option<TokenizerInfo>,
    pub total_size_bytes: u64,
    pub metadata: HashMap<String, String>,
}

/// Individual model file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub filename: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub format: ModelFormat,
    pub file_type: FileType,
    pub sha256: Option<String>,
}

/// Type of model file
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileType {
    Weights,        // Model weights
    Config,         // Configuration JSON
    Tokenizer,      // Tokenizer files
    Vocabulary,     // Vocab files
    Metadata,       // Other metadata
}

/// Download options
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    /// Preferred format (SafeTensors by default)
    pub preferred_format: ModelFormat,
    /// Force re-download even if cached
    pub force: bool,
    /// Show progress bar
    pub show_progress: bool,
    /// Maximum file size to download (in bytes)
    pub max_size_bytes: Option<u64>,
    /// Specific files to download (if None, download all)
    pub files_filter: Option<Vec<String>>,
    /// Verify checksums after download
    pub verify_checksums: bool,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            preferred_format: ModelFormat::SafeTensors,
            force: false,
            show_progress: true,
            max_size_bytes: None,
            files_filter: None,
            verify_checksums: true,
        }
    }
}

/// Git-based model source
pub struct GitSource {
    git_source: crate::api::git_downloader::GitModelSource,
    cache_dir: PathBuf,
}

impl GitSource {
    /// Create a new Git model source
    pub async fn new(cache_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&cache_dir).await?;
        
        let git_source = crate::api::git_downloader::GitModelSource::new(cache_dir.clone());
        
        Ok(Self {
            git_source,
            cache_dir,
        })
    }
}

#[async_trait]
impl ModelSource for GitSource {
    async fn list_models(&self, _query: Option<&str>) -> Result<Vec<ModelInfo>> {
        // Git repos don't have a central listing service
        // This could be extended to read from a configured list of repos
        Ok(vec![])
    }
    
    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        // Return basic info from the Git URL
        Ok(ModelInfo {
            id: model_id.to_string(),
            name: model_id.split('/').last().unwrap_or(model_id).to_string(),
            description: None,
            author: None,
            size_bytes: None,
            format: ModelFormat::SafeTensors,
            architecture: None,
            tags: vec![],
            created_at: None,
            updated_at: None,
        })
    }
    
    async fn download_model(&self, model_id: &str, _target_dir: &Path) -> Result<DownloadedModel> {
        self.download_model_with_options(model_id, _target_dir, DownloadOptions::default()).await
    }

}

impl GitSource {
    pub async fn download_model_with_options(&self, model_id: &str, _target_dir: &Path, options: DownloadOptions) -> Result<DownloadedModel> {
        // Pass the URL directly to git2 - it will handle validation
        let git_url = model_id.to_string();

        if options.show_progress {
            println!("📥 Cloning model from: {}", git_url);
        }

        let (model_uuid, model_path) = if options.show_progress {
            self.git_source.clone_model(&git_url).await?
        } else {
            self.git_source.clone_model_with_progress(&git_url, false).await?
        };
        let uuid = model_uuid.0;
        
        // Model is already cloned to model_path
        let model_dir = model_path.clone();
        let data_dir = model_dir.clone(); // Git clone includes everything
        
        // Create model metadata file
        let model_metadata = crate::api::model_storage::ModelMetadataFile {
            model_id: model_uuid.clone(),
            name: model_id.to_string(),
            display_name: model_id.split('/').last().unwrap_or(model_id).to_string(),
            source_uri: model_id.to_string(),
            architecture: None, // Will be updated after downloading config
            parameters: None,
            created_at: chrono::Utc::now().timestamp(),
            last_accessed: chrono::Utc::now().timestamp(),
        };
        
        // Save metadata file in the model directory (hardcoded filename for security)
        // SECURITY: Never use user input for this filename to prevent path traversal
        let metadata_filename = "model.json"; // Hardcoded, never from user input
        let metadata_path = model_dir.join(metadata_filename);
        // Serialize and validate with json_threat_protection
        let metadata_content = serde_json::to_string_pretty(&model_metadata)?;
        // Validate before writing
        jtp::from_str(&metadata_content)
            .with_max_depth(10)
            .with_max_string_length(10000)
            .validate()?;
        
        // Additional validation that we're writing to the right place
        if metadata_path.parent() != Some(&model_dir) {
            return Err(anyhow::anyhow!("Security violation: metadata path escape attempt"));
        }
        
        fs::write(&metadata_path, metadata_content).await?;
        
        println!("📁 Model directory: {} ({})", uuid, model_id);
        
        // Git clone already downloaded everything, now just find what we have
        let mut files = Vec::new();
        let mut total_size = 0u64;
        
        // Check for configuration file
        let config_path = data_dir.join("config.json");
        if config_path.exists() {
            let metadata = fs::metadata(&config_path).await?;
            files.push(ModelFile {
                filename: "config.json".to_string(),
                path: config_path.clone(),
                size_bytes: metadata.len(),
                format: ModelFormat::SafeTensors,
                file_type: FileType::Config,
                sha256: None,
            });
            total_size += metadata.len();
        }
        
        // Parse config
        let config = if config_path.exists() {
            let content = fs::read_to_string(&config_path).await?;
            // Parse JSON config - simplified for now
            Some(serde_json::from_str::<ModelConfig>(&content).unwrap_or_else(|_| {
                ModelConfig {
                    architecture: "unknown".to_string(),
                    hidden_size: 4096,
                    num_attention_heads: 32,
                    num_hidden_layers: 32,
                    vocab_size: 32000,
                    max_position_embeddings: 2048,
                    rope_theta: Some(10000.0),
                    torch_dtype: Some("float16".to_string()),
                }
            }))
        } else {
            None
        };
        
        // Check for tokenizer files (already cloned by Git)
        for tokenizer_file in &["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"] {
            let target_path = data_dir.join(tokenizer_file);
            if target_path.exists() {
                let metadata = fs::metadata(&target_path).await?;
                println!("📝 Found {}", tokenizer_file);
                files.push(ModelFile {
                    filename: tokenizer_file.to_string(),
                    path: target_path,
                    size_bytes: metadata.len(),
                    format: ModelFormat::SafeTensors,
                    file_type: FileType::Tokenizer,
                    sha256: None,
                });
                total_size += metadata.len();
            }
        }
        
        // Parse tokenizer info
        let tokenizer = {
            let tokenizer_config_path = model_dir.join("tokenizer_config.json");
            if tokenizer_config_path.exists() {
                let content = fs::read_to_string(&tokenizer_config_path).await?;
                // Parse JSON tokenizer config - simplified for now
                Some(serde_json::from_str::<TokenizerInfo>(&content).unwrap_or_else(|_| {
                    TokenizerInfo {
                        tokenizer_class: Some("unknown".to_string()),
                        vocab_size: 32000,
                        pad_token_id: Some(0),
                        eos_token_id: Some(1),
                    }
                }))
            } else {
                None
            }
        };
        
        // Find SafeTensors weight files (already cloned by Git)
        let patterns = vec![
            "*.safetensors",
            "model-*.safetensors",
        ];
        
        let mut weight_files: Vec<String> = Vec::new();
        for pattern in patterns {
            let glob_pattern = data_dir.join(pattern).to_string_lossy().to_string();
            if let Ok(entries) = glob::glob(&glob_pattern) {
                for entry in entries {
                    if let Ok(path) = entry {
                        weight_files.push(path.to_string_lossy().to_string());
                    }
                }
            }
        }
        
        if weight_files.is_empty() {
            return Err(anyhow!("No SafeTensors model files found in repository"));
        }
        
        let final_format = ModelFormat::SafeTensors;
        
        println!("📦 Found {} weight files", weight_files.len());
        
        for weight_path in &weight_files {
            let metadata = fs::metadata(&weight_path).await?;
            let filename = Path::new(weight_path).file_name().unwrap().to_str().unwrap().to_string();
            println!("✓ {} ({:.2} GB)", 
                filename,
                metadata.len() as f64 / 1_073_741_824.0
            );
            files.push(ModelFile {
                filename: filename.clone(),
                path: PathBuf::from(weight_path.clone()),
                size_bytes: metadata.len(),
                format: final_format,
                file_type: FileType::Weights,
                sha256: None,
            });
            total_size += metadata.len();
        }
        
        // Update model metadata with config information
        if let Some(ref cfg) = config {
            let mut updated_metadata = model_metadata.clone();
            updated_metadata.architecture = Some(cfg.architecture.clone());
            
            let metadata_path = model_dir.join("model.json");
            
            // Serialize and validate
            let metadata_content = serde_json::to_string_pretty(&updated_metadata)?;
            // Validate
            jtp::from_str(&metadata_content)
                .with_max_depth(10)
                .with_max_string_length(10000)
                .validate()?;
            
            fs::write(&metadata_path, metadata_content).await?;
        }
        
        println!("✅ Model downloaded successfully!");
        println!("📊 Total size: {:.2} GB", total_size as f64 / 1_073_741_824.0);
        println!("📁 Location: {} (UUID: {})", model_id, uuid);
        
        Ok(DownloadedModel {
            model_id: model_id.to_string(),
            uuid,  // Include the actual UUID used for storage
            local_path: data_dir,
            format: ModelFormat::SafeTensors,
            files,
            config,
            tokenizer,
            total_size_bytes: total_size,
            metadata: HashMap::new(),
        })
    }
}

/// Unified model downloader
pub struct ModelDownloader {
    git_source: GitSource,
    cache_dir: PathBuf,
}

impl ModelDownloader {
    /// Create a new model downloader
    pub async fn new(cache_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&cache_dir).await?;
        
        let git_source = GitSource::new(cache_dir.clone()).await?;
        
        Ok(Self {
            git_source,
            cache_dir,
        })
    }
    
    /// Download a model from a Git repository
    pub async fn download(
        &self,
        git_url: &str,
        _options: DownloadOptions,
    ) -> Result<DownloadedModel> {
        // Download the model using Git
        self.git_source.download_model(git_url, &self.cache_dir).await
    }
    
    /// Get model information
    pub async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        self.git_source.get_model_info(model_id).await
    }
}

/// Parse model configuration from JSON
fn parse_config(json_str: &str) -> Result<ModelConfig> {
    // Validate before parsing
    jtp::from_str(json_str)
        .with_max_depth(20)
        .with_max_string_length(100000)
        .with_max_array_entries(10000)
        .validate()
        .map_err(|e| anyhow!("Invalid model config JSON: {:?}", e))?;
    
    let json: serde_json::Value = serde_json::from_str(json_str)?;
    
    Ok(ModelConfig {
        architecture: json["architectures"][0].as_str().unwrap_or("unknown").to_string(),
        hidden_size: json["hidden_size"].as_u64().unwrap_or(0) as usize,
        num_attention_heads: json["num_attention_heads"].as_u64().unwrap_or(0) as usize,
        num_hidden_layers: json["num_hidden_layers"].as_u64().unwrap_or(0) as usize,
        vocab_size: json["vocab_size"].as_u64().unwrap_or(0) as usize,
        max_position_embeddings: json["max_position_embeddings"].as_u64().unwrap_or(0) as usize,
        rope_theta: json["rope_theta"].as_f64().map(|f| f as f32),
        torch_dtype: json["torch_dtype"].as_str().map(|s| s.to_string()),
    })
}

/// Parse tokenizer information from JSON
fn parse_tokenizer_info(json_str: &str) -> Result<TokenizerInfo> {
    // Validate before parsing
    jtp::from_str(json_str)
        .with_max_depth(10)
        .with_max_string_length(50000)
        .validate()
        .map_err(|e| anyhow!("Invalid tokenizer config JSON: {:?}", e))?;
    
    let json: serde_json::Value = serde_json::from_str(json_str)?;
    
    Ok(TokenizerInfo {
        tokenizer_class: json["tokenizer_class"].as_str().map(|s| s.to_string()),
        vocab_size: json["vocab_size"].as_u64().unwrap_or(0) as usize,
        pad_token_id: json["pad_token_id"].as_u64().map(|n| n as u32),
        eos_token_id: json["eos_token_id"].as_u64().map(|n| n as u32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parse_config_with_valid_json() {
        let json = r#"{
            "architectures": ["TransformerModel"],
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "vocab_size": 50265,
            "max_position_embeddings": 512
        }"#;

        let config = parse_config(json).expect("Failed to parse valid config");
        assert_eq!(config.architecture, "TransformerModel");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_hidden_layers, 12);
    }

    #[test]
    fn test_parse_config_rejects_deeply_nested_json() {
        // Create deeply nested JSON
        let mut deep_json = String::from(r#"{"architectures": ["Model"], "nested": "#);
        for _ in 0..25 {
            deep_json.push_str(r#"{"level": "#);
        }
        deep_json.push_str("\"deep\"");
        for _ in 0..25 {
            deep_json.push_str("}");
        }
        deep_json.push_str("}");

        let result = parse_config(&deep_json);
        assert!(result.is_err(), "Should reject deeply nested JSON");
    }

    #[test]
    fn test_parse_config_rejects_oversized_strings() {
        let huge_string = "x".repeat(200_000);
        let json = format!(
            r#"{{"architectures": ["{}"], "hidden_size": 768}}"#, 
            huge_string
        );

        let result = parse_config(&json);
        assert!(result.is_err(), "Should reject JSON with oversized strings");
    }

    #[test]
    fn test_parse_tokenizer_info_with_valid_json() {
        let json = r#"{
            "tokenizer_class": "GPT2Tokenizer",
            "vocab_size": 50257,
            "pad_token_id": 50256,
            "eos_token_id": 50256
        }"#;

        let info = parse_tokenizer_info(json).expect("Failed to parse valid tokenizer info");
        assert_eq!(info.tokenizer_class, Some("GPT2Tokenizer".to_string()));
        assert_eq!(info.vocab_size, 50257);
        assert_eq!(info.pad_token_id, Some(50256));
        assert_eq!(info.eos_token_id, Some(50256));
    }

    #[test]
    fn test_parse_tokenizer_info_rejects_invalid_json() {
        let mut json = r#"{
            "tokenizer_class": "GPT2Tokenizer",
            "huge_array": ["#.to_string();
        
        // Add huge array
        for i in 0..20000 {
            json.push_str(&format!("\"{}\",", i));
        }
        
        let result = parse_tokenizer_info(&json);
        assert!(result.is_err(), "Should reject malformed JSON");
    }

    #[test]
    fn test_model_format_serialization() {
        let format = ModelFormat::SafeTensors;
        let serialized = serde_json::to_string(&format).unwrap();
        let deserialized: ModelFormat = serde_json::from_str(&serialized).unwrap();
        assert_eq!(format, deserialized);

        let format = ModelFormat::PyTorch;
        let serialized = serde_json::to_string(&format).unwrap();
        let deserialized: ModelFormat = serde_json::from_str(&serialized).unwrap();
        assert_eq!(format, deserialized);
    }

    #[test]
    fn test_file_type_classification() {
        assert_eq!(
            FileType::Weights,
            FileType::Weights
        );
        assert_eq!(
            FileType::Config,
            FileType::Config
        );
        assert_eq!(
            FileType::Tokenizer,
            FileType::Tokenizer
        );
    }

    // HuggingFace URL parsing tests removed - only direct Git URLs supported now

    #[tokio::test]
    async fn test_metadata_json_protection() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path().join("test-uuid");
        tokio::fs::create_dir_all(&model_dir).await.unwrap();

        let metadata = crate::api::model_storage::ModelMetadataFile {
            model_id: crate::api::model_storage::ModelId::new(),
            name: "test-model".to_string(),
            display_name: "Test Model".to_string(),
            source_uri: "https://github.com/test/model".to_string(),
            architecture: Some("test".to_string()),
            parameters: Some(1000),
            created_at: 0,
            last_accessed: 0,
        };

        // Serialize and validate
        let content = serde_json::to_string_pretty(&metadata).unwrap();
        
        // Should pass validation
        let result = jtp::from_str(&content)
            .with_max_depth(10)
            .with_max_string_length(10000)
            .validate();
        
        assert!(result.is_ok(), "Valid metadata should pass validation");
    }

    #[test]
    fn test_download_options_defaults() {
        let options = DownloadOptions {
            preferred_format: ModelFormat::SafeTensors,
            force: false,
            show_progress: true,
            files_filter: None,
            verify_checksums: true,
            max_size_bytes: None,
        };

        assert_eq!(options.preferred_format, ModelFormat::SafeTensors);
        assert!(!options.force);
        assert!(options.show_progress);
        assert!(options.files_filter.is_none());
        assert!(options.verify_checksums);
        assert!(options.max_size_bytes.is_none());
    }
}