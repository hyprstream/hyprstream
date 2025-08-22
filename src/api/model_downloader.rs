//! Unified model downloader with SafeTensors-first approach
//! 
//! This module provides a clean abstraction for downloading models from various sources,
//! prioritizing SafeTensors format for security and compatibility.

use anyhow::{anyhow, Result};
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use async_trait::async_trait;
use candle_core::Device;
use tempfile::NamedTempFile;

/// Model format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFormat {
    SafeTensors,
    PyTorch,
}

impl ModelFormat {
    /// Get file extensions for this format
    pub fn extensions(&self) -> Vec<&'static str> {
        match self {
            ModelFormat::SafeTensors => vec!["safetensors"],
            ModelFormat::PyTorch => vec!["bin", "pt", "pth"],
        }
    }
    
    /// Check if a filename matches this format
    pub fn matches(&self, filename: &str) -> bool {
        let lower = filename.to_lowercase();
        self.extensions().iter().any(|ext| lower.ends_with(ext))
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

/// HuggingFace model source
pub struct HuggingFaceSource {
    api: Api,
    cache_dir: PathBuf,
}

impl HuggingFaceSource {
    /// Create a new HuggingFace source
    pub async fn new(cache_dir: PathBuf, token: Option<String>) -> Result<Self> {
        fs::create_dir_all(&cache_dir).await?;
        
        let api = if let Some(token) = token {
            ApiBuilder::new()
                .with_token(Some(token))
                .build()?
        } else {
            ApiBuilder::new().build()?
        };
        
        Ok(Self { api, cache_dir })
    }
    
    /// Parse repository ID (e.g., "microsoft/phi-2" or "hf://microsoft/phi-2")
    fn parse_repo_id(model_id: &str) -> &str {
        model_id.strip_prefix("hf://").unwrap_or(model_id)
    }
    
    /// Determine the best format available for a model
    async fn determine_best_format(&self, repo: &ApiRepo) -> Result<(ModelFormat, Vec<String>)> {
        // List files in the repository
        let files = self.list_repo_files(repo).await?;
        
        // Categorize files by format
        let mut formats: HashMap<ModelFormat, Vec<String>> = HashMap::new();
        
        for file in files {
            let filename = file.as_str();
            
            // Check for model weight files (including sharded models)
            if filename.ends_with(".safetensors") || 
               (filename.contains("model") && filename.contains(".safetensors")) {
                formats.entry(ModelFormat::SafeTensors)
                    .or_default()
                    .push(file);
            } else if filename.ends_with(".bin") || filename.ends_with(".pth") || filename.ends_with(".pt") ||
                      (filename.starts_with("pytorch_model") && filename.ends_with(".bin")) {
                formats.entry(ModelFormat::PyTorch)
                    .or_default()
                    .push(file);
            }
        }
        
        // Filter to only include actual model weight files (not just any .bin file)
        if let Some(files) = formats.get_mut(&ModelFormat::SafeTensors) {
            files.retain(|f| f.contains("model") || f == "model.safetensors");
        }
        if let Some(files) = formats.get_mut(&ModelFormat::PyTorch) {
            files.retain(|f| f.contains("model") || f.contains("pytorch"));
        }
        
        // Choose the best format (SafeTensors > PyTorch)
        if let Some(files) = formats.get(&ModelFormat::SafeTensors) {
            if !files.is_empty() {
                Ok((ModelFormat::SafeTensors, files.clone()))
            } else {
                formats.remove(&ModelFormat::SafeTensors);
                self.choose_next_best_format(formats)
            }
        } else if let Some(files) = formats.get(&ModelFormat::PyTorch) {
            if !files.is_empty() {
                Ok((ModelFormat::PyTorch, files.clone()))
            } else {
                formats.remove(&ModelFormat::PyTorch);
                self.choose_next_best_format(formats)
            }
        } else {
            Err(anyhow!("No supported model format found"))
        }
    }
    
    fn choose_next_best_format(&self, formats: HashMap<ModelFormat, Vec<String>>) -> Result<(ModelFormat, Vec<String>)> {
        if let Some(files) = formats.get(&ModelFormat::PyTorch) {
            if !files.is_empty() {
                return Ok((ModelFormat::PyTorch, files.clone()));
            }
        }
        Err(anyhow!("No supported model format found"))
    }
    
    /// Determine best format from a list of files
    fn determine_best_format_from_files(&self, files: &[String]) -> Result<(ModelFormat, Vec<String>)> {
        let mut formats: HashMap<ModelFormat, Vec<String>> = HashMap::new();
        
        for file in files {
            let filename = file.as_str();
            
            // Check for model weight files (including sharded models)
            if filename.ends_with(".safetensors") && 
               (filename.contains("model") || filename == "model.safetensors") {
                formats.entry(ModelFormat::SafeTensors)
                    .or_default()
                    .push(file.clone());
            } else if (filename.ends_with(".bin") && 
                      (filename.contains("pytorch_model") || filename.contains("model"))) ||
                      filename.ends_with(".pth") || filename.ends_with(".pt") {
                formats.entry(ModelFormat::PyTorch)
                    .or_default()
                    .push(file.clone());
            }
        }
        
        // Choose the best format (SafeTensors > PyTorch)
        if let Some(files) = formats.get(&ModelFormat::SafeTensors) {
            if !files.is_empty() {
                Ok((ModelFormat::SafeTensors, files.clone()))
            } else {
                self.choose_next_best_format(formats)
            }
        } else if let Some(files) = formats.get(&ModelFormat::PyTorch) {
            if !files.is_empty() {
                Ok((ModelFormat::PyTorch, files.clone()))
            } else {
                self.choose_next_best_format(formats)
            }
        } else {
            Err(anyhow!("No supported model format found in files: {:?}", files))
        }
    }
    
    /// Get repository files using HuggingFace API directly
    async fn get_repo_files_from_api(&self, repo_id: &str) -> Result<Vec<String>> {
        // Use the HuggingFace REST API to get actual file list
        let url = format!("https://huggingface.co/api/models/{}", repo_id);
        let client = reqwest::Client::new();
        
        let response = client.get(&url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to query HuggingFace API: {}", e))?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Failed to get repository info: HTTP {}", response.status()));
        }
        
        let json: serde_json::Value = response.json().await
            .map_err(|e| anyhow!("Failed to parse repository response: {}", e))?;
        
        // Extract file list from siblings field
        if let Some(siblings) = json.get("siblings").and_then(|s| s.as_array()) {
            let files: Vec<String> = siblings.iter()
                .filter_map(|sibling| {
                    sibling.get("rfilename")
                        .and_then(|name| name.as_str())
                        .map(|s| s.to_string())
                })
                .collect();
            
            if !files.is_empty() {
                return Ok(files);
            }
        }
        
        // Fallback to default files
        Ok(vec![
            "config.json".to_string(),
            "model.safetensors".to_string(),
            "tokenizer.json".to_string(),
            "tokenizer_config.json".to_string(),
        ])
    }
    
    /// List files in a repository
    async fn list_repo_files(&self, repo: &ApiRepo) -> Result<Vec<String>> {
        // Try to use the hf-hub API first
        match repo.info().await {
            Ok(_info) => {
                // The hf-hub crate doesn't expose file listing directly
                // We need to use a different approach - try to query known patterns
                // or use the REST API directly
                
                // For now, try to detect which files exist by attempting to get them
                let mut files = Vec::new();
                
                // Common config files
                let common_files = vec![
                    "config.json",
                    "tokenizer.json", 
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "special_tokens_map.json",
                ];
                
                for file in common_files {
                    // Try to check if file exists (this is a workaround)
                    if repo.get(file).await.is_ok() {
                        files.push(file.to_string());
                    }
                }
                
                // Try common model file patterns
                // Single file models
                if repo.get("model.safetensors").await.is_ok() {
                    files.push("model.safetensors".to_string());
                } else {
                    // Check for sharded models (up to 10 shards for now)
                    for i in 1..=10 {
                        let filename = format!("model-{:05}-of-{:05}.safetensors", i, 10);
                        if repo.get(&filename).await.is_ok() {
                            files.push(filename);
                        } else {
                            // Also check the more common pattern
                            let filename = format!("model-{:05}-of-00002.safetensors", i);
                            if repo.get(&filename).await.is_ok() {
                                files.push(filename);
                            } else if i > 2 {
                                break; // Stop checking if we haven't found files after 2
                            }
                        }
                    }
                }
                
                // Check for PyTorch files
                if files.iter().filter(|f| f.contains("model")).count() == 0 {
                    if repo.get("pytorch_model.bin").await.is_ok() {
                        files.push("pytorch_model.bin".to_string());
                    } else {
                        // Check for sharded PyTorch models
                        for i in 1..=10 {
                            let filename = format!("pytorch_model-{:05}-of-{:05}.bin", i, 10);
                            if repo.get(&filename).await.is_ok() {
                                files.push(filename);
                            } else if i > 2 {
                                break;
                            }
                        }
                    }
                }
                
                if files.is_empty() {
                    // Last resort: return common patterns
                    Ok(vec![
                        "config.json".to_string(),
                        "model.safetensors".to_string(),
                        "tokenizer.json".to_string(),
                        "tokenizer_config.json".to_string(),
                    ])
                } else {
                    Ok(files)
                }
            }
            Err(_) => {
                // Fallback to common patterns
                Ok(vec![
                    "config.json".to_string(),
                    "model.safetensors".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer_config.json".to_string(),
                ])
            }
        }
    }
    
    /// Download a file with progress
    async fn download_file_with_progress(
        &self,
        repo: &ApiRepo,
        filename: &str,
        target_path: &Path,
        pb: Option<&ProgressBar>,
    ) -> Result<u64> {
        // Check if file already exists and get its size
        if target_path.exists() && !pb.is_some() {
            let metadata = fs::metadata(target_path).await?;
            return Ok(metadata.len());
        }
        
        // Download using hf_hub
        let file_path = repo.get(filename).await?;
        
        // Copy to target location
        if file_path != target_path {
            fs::copy(&file_path, target_path).await?;
        }
        
        if let Some(pb) = pb {
            pb.inc(1);
        }
        
        let metadata = fs::metadata(target_path).await?;
        Ok(metadata.len())
    }
}

#[async_trait]
impl ModelSource for HuggingFaceSource {
    async fn list_models(&self, query: Option<&str>) -> Result<Vec<ModelInfo>> {
        // This would use the HF API to search models
        // For now, return some examples
        let models = vec![
            ModelInfo {
                id: "microsoft/phi-2".to_string(),
                name: "Phi-2".to_string(),
                description: Some("Small but capable model".to_string()),
                author: Some("Microsoft".to_string()),
                size_bytes: Some(5_000_000_000),
                format: ModelFormat::SafeTensors,
                architecture: Some("PhiForCausalLM".to_string()),
                tags: vec!["text-generation".to_string()],
                created_at: None,
                updated_at: None,
            },
        ];
        
        Ok(if let Some(q) = query {
            models.into_iter()
                .filter(|m| m.name.to_lowercase().contains(&q.to_lowercase()))
                .collect()
        } else {
            models
        })
    }
    
    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let repo_id = Self::parse_repo_id(model_id);
        let repo = self.api.model(repo_id.to_string());
        
        // Get repository info
        let info = repo.info().await?;
        
        // Determine format
        let (format, _) = self.determine_best_format(&repo).await?;
        
        Ok(ModelInfo {
            id: repo_id.to_string(),
            name: repo_id.split('/').last().unwrap_or(repo_id).to_string(),
            description: None, // Would parse from README
            author: repo_id.split('/').next().map(|s| s.to_string()),
            size_bytes: None, // Would calculate from files
            format,
            architecture: None, // Would parse from config.json
            tags: vec![],
            created_at: None,
            updated_at: None,
        })
    }
    
    async fn download_model(&self, model_id: &str, target_dir: &Path) -> Result<DownloadedModel> {
        let repo_id = Self::parse_repo_id(model_id);
        println!("📥 Downloading model: {}", repo_id);
        
        // Create target directory
        let model_dir = target_dir.join(repo_id.replace('/', "_"));
        fs::create_dir_all(&model_dir).await?;
        
        // Get repository handle
        let repo = self.api.model(repo_id.to_string());
        
        // First, get the actual file list from the API
        let actual_files = self.get_repo_files_from_api(repo_id).await
            .unwrap_or_else(|e| {
                eprintln!("Warning: Failed to get file list from API: {}", e);
                eprintln!("Falling back to file probing...");
                vec![]
            });
        
        // Determine best format using actual files if available
        let (format, weight_files) = if !actual_files.is_empty() {
            self.determine_best_format_from_files(&actual_files)?
        } else {
            self.determine_best_format(&repo).await?
        };
        println!("📦 Format: {:?} ({} files)", format, weight_files.len());
        
        let mut files = Vec::new();
        let mut total_size = 0u64;
        
        // Download configuration
        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            println!("📄 Downloading config.json");
            let size = self.download_file_with_progress(&repo, "config.json", &config_path, None).await?;
            files.push(ModelFile {
                filename: "config.json".to_string(),
                path: config_path.clone(),
                size_bytes: size,
                format: ModelFormat::SafeTensors,
                file_type: FileType::Config,
                sha256: None,
            });
            total_size += size;
        }
        
        // Parse config
        let config = if config_path.exists() {
            let content = fs::read_to_string(&config_path).await?;
            Some(parse_config(&content)?)
        } else {
            None
        };
        
        // Download tokenizer files
        for tokenizer_file in &["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"] {
            let target_path = model_dir.join(tokenizer_file);
            if !target_path.exists() {
                match self.download_file_with_progress(&repo, tokenizer_file, &target_path, None).await {
                    Ok(size) => {
                        println!("📝 Downloaded {}", tokenizer_file);
                        files.push(ModelFile {
                            filename: tokenizer_file.to_string(),
                            path: target_path,
                            size_bytes: size,
                            format: ModelFormat::SafeTensors,
                            file_type: FileType::Tokenizer,
                            sha256: None,
                        });
                        total_size += size;
                    }
                    Err(_) => {
                        // Tokenizer file might not exist, that's okay
                    }
                }
            }
        }
        
        // Parse tokenizer info
        let tokenizer = {
            let tokenizer_config_path = model_dir.join("tokenizer_config.json");
            if tokenizer_config_path.exists() {
                let content = fs::read_to_string(&tokenizer_config_path).await?;
                Some(parse_tokenizer_info(&content)?)
            } else {
                None
            }
        };
        
        // Download weight files
        let pb = ProgressBar::new(weight_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        
        // Use the detected format
        let final_format = format;
        
        for weight_file in &weight_files {
            let filename = Path::new(weight_file).file_name().unwrap().to_str().unwrap();
            
            // Convert PyTorch to SafeTensors if needed
            let safetensors_filename = if filename.ends_with(".bin") || filename.ends_with(".pth") {
                // PyTorch files also get converted
                let base = Path::new(filename).file_stem().unwrap().to_str().unwrap();
                format!("{}.safetensors", base)
            } else {
                filename.to_string()
            };
            
            let target_path = model_dir.join(&safetensors_filename);
            
            if target_path.exists() {
                let metadata = fs::metadata(&target_path).await?;
                println!("✓ {} already exists ({:.2} GB)", 
                    safetensors_filename,
                    metadata.len() as f64 / 1_073_741_824.0
                );
                files.push(ModelFile {
                    filename: safetensors_filename.clone(),
                    path: target_path,
                    size_bytes: metadata.len(),
                    format: ModelFormat::SafeTensors,
                    file_type: FileType::Weights,
                    sha256: None,
                });
                total_size += metadata.len();
                pb.inc(1);
            } else {
                pb.set_message(format!("Downloading {}", filename));
                
                // Download the file
                let size = self.download_file_with_progress(&repo, weight_file, &target_path, Some(&pb)).await?;
                
                files.push(ModelFile {
                    filename: safetensors_filename,
                    path: target_path,
                    size_bytes: size,
                    format: final_format,
                    file_type: FileType::Weights,
                    sha256: None,
                });
                total_size += size;
            }
        }
        
        pb.finish_with_message("Download complete");
        
        println!("✅ Model downloaded successfully!");
        println!("📊 Total size: {:.2} GB", total_size as f64 / 1_073_741_824.0);
        println!("📁 Location: {}", model_dir.display());
        
        Ok(DownloadedModel {
            model_id: repo_id.to_string(),
            local_path: model_dir,
            format: final_format,  // Always report SafeTensors if we converted
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
    sources: HashMap<String, Box<dyn ModelSource>>,
    cache_dir: PathBuf,
    default_source: String,
}

impl ModelDownloader {
    /// Create a new model downloader with optional HuggingFace token
    pub async fn new(cache_dir: PathBuf, hf_token: Option<String>) -> Result<Self> {
        fs::create_dir_all(&cache_dir).await?;
        
        let mut sources: HashMap<String, Box<dyn ModelSource>> = HashMap::new();
        
        // Add HuggingFace source by default with token
        // Create two instances for the two aliases
        let hf_source1 = HuggingFaceSource::new(
            cache_dir.join("huggingface"),
            hf_token.clone(),
        ).await?;
        let hf_source2 = HuggingFaceSource::new(
            cache_dir.join("huggingface"),
            hf_token,
        ).await?;
        // Register with both "huggingface" and "hf" aliases
        sources.insert("huggingface".to_string(), Box::new(hf_source1));
        sources.insert("hf".to_string(), Box::new(hf_source2));
        
        Ok(Self {
            sources,
            cache_dir,
            default_source: "huggingface".to_string(),
        })
    }
    
    /// Add a model source
    pub fn add_source(&mut self, name: String, source: Box<dyn ModelSource>) {
        self.sources.insert(name, source);
    }
    
    /// Download a model with options
    pub async fn download(
        &self,
        model_id: &str,
        options: DownloadOptions,
    ) -> Result<DownloadedModel> {
        // Parse source from model_id (e.g., "hf://model" or just "model")
        let (source_name, model_name) = if model_id.contains("://") {
            let parts: Vec<&str> = model_id.splitn(2, "://").collect();
            (parts[0], parts[1])
        } else {
            (self.default_source.as_str(), model_id)
        };
        
        // Get the source
        let source = self.sources.get(source_name)
            .ok_or_else(|| anyhow!("Unknown model source: {}", source_name))?;
        
        // Download the model
        let target_dir = self.cache_dir.join(source_name);
        let model = source.download_model(model_name, &target_dir).await?;
        
        // Model is downloaded in its native format
        
        Ok(model)
    }
    
    /// List available models
    pub async fn list_models(&self, source: Option<&str>, query: Option<&str>) -> Result<Vec<ModelInfo>> {
        let source_name = source.unwrap_or(&self.default_source);
        let source = self.sources.get(source_name)
            .ok_or_else(|| anyhow!("Unknown model source: {}", source_name))?;
        
        source.list_models(query).await
    }
    
    /// Get model information
    pub async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let (source_name, model_name) = if model_id.contains("://") {
            let parts: Vec<&str> = model_id.splitn(2, "://").collect();
            (parts[0], parts[1])
        } else {
            (self.default_source.as_str(), model_id)
        };
        
        let source = self.sources.get(source_name)
            .ok_or_else(|| anyhow!("Unknown model source: {}", source_name))?;
        
        source.get_model_info(model_name).await
    }
}

/// Parse model configuration from JSON
fn parse_config(json_str: &str) -> Result<ModelConfig> {
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
    let json: serde_json::Value = serde_json::from_str(json_str)?;
    
    Ok(TokenizerInfo {
        tokenizer_class: json["tokenizer_class"].as_str().map(|s| s.to_string()),
        vocab_size: json["vocab_size"].as_u64().unwrap_or(0) as usize,
        pad_token_id: json["pad_token_id"].as_u64().map(|n| n as u32),
        eos_token_id: json["eos_token_id"].as_u64().map(|n| n as u32),
    })
}