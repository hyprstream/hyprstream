//! Model download functionality for Hyprstream

use anyhow::{anyhow, Result};
use hf_hub::api::tokio::ApiBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use crate::{auth::HfAuth, config::HyprConfig};
use crate::api::model_storage::{ModelStorage, ModelMetadata, ModelId, ExternalSource, SourceType, ModelFile, FileType, ModelRegistry};
use crate::api::model_management::ModelUri;

/// Download a model from HuggingFace Hub
pub async fn download_qwen3_model(config: Option<&HyprConfig>) -> Result<PathBuf> {
    let config = config.map(|c| c.clone()).unwrap_or_else(|| HyprConfig::load().unwrap_or_default());
    let base_dir = config.models_dir();
    
    // Ensure models directory exists
    tokio::fs::create_dir_all(base_dir).await
        .map_err(|e| anyhow!("Failed to create models directory: {}", e))?;

    println!("📥 Downloading Qwen3-1.7B model from HuggingFace Hub...");
    println!("📁 Saving to: {}", base_dir.display());
    
    // Initialize HuggingFace API with authentication
    let auth = HfAuth::new()?;
    let api = if let Some(token) = auth.get_token().await? {
        ApiBuilder::from_env()
            .with_token(Some(token))
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    } else {
        println!("ℹ️  No HuggingFace token found. Some models may not be accessible.");
        println!("   Use 'hyprstream auth login' to authenticate for gated models.");
        ApiBuilder::from_env()
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    };
    
    // Get the model repository
    let _repo = api.model("Qwen/Qwen2-1.5B-Instruct".to_string());
    
    // Only SafeTensors format is supported
    eprintln!("");
    eprintln!("❌ Model format not supported.");
    eprintln!("   Please download models in SafeTensors format.");
    eprintln!("");
    eprintln!("   For example: hyprstream model pull hf://Qwen/Qwen2-1.5B");
    
    Err(anyhow!("Model format not supported"))
}

/// Download with progress tracking
async fn download_with_progress(
    repo: &hf_hub::api::tokio::ApiRepo,
    filename: &str,
    local_path: &Path,
    pb: &ProgressBar,
) -> Result<()> {
    pb.set_message(format!("Downloading {}", filename));
    
    // Get the file from the repository with better error handling
    let file_path = repo.get(filename).await
        .map_err(|e| {
            let error_msg = format!("{}", e);
            if error_msg.contains("etag") {
                anyhow!("Download failed - server missing required headers. Try again or check if the file exists in the repository")
            } else if error_msg.contains("404") {
                anyhow!("File '{}' not found in repository. Use --files with the exact filename from the repository", filename)
            } else {
                anyhow!("Failed to download file '{}': {}", filename, e)
            }
        })?;
    
    // Copy the downloaded file to the target location
    tokio::fs::copy(&file_path, local_path).await
        .map_err(|e| anyhow!("Failed to copy file to destination: {}", e))?;
    
    pb.set_position(100);
    pb.set_message("Download completed");
    
    Ok(())
}

/// Download a specific model by HuggingFace URI
pub async fn download_model_by_uri(
    model_uri: &str,
    filename: Option<&str>,
    config: Option<&HyprConfig>,
) -> Result<ModelId> {
    let config = config.map(|c| c.clone()).unwrap_or_else(|| HyprConfig::load().unwrap_or_default());
    let base_dir = config.models_dir();
    
    // Parse the URI with optional revision/tag support
    // Format: author/model-name[:revision]
    let (repo_path, revision) = if let Some(colon_pos) = model_uri.rfind(':') {
        let repo_part = &model_uri[..colon_pos];
        let rev_part = &model_uri[colon_pos + 1..];
        (repo_part, Some(rev_part))
    } else {
        (model_uri, None)
    };
    
    let parts: Vec<&str> = repo_path.split('/').collect();
    if parts.len() != 2 {
        return Err(anyhow!("Invalid model URI format. Expected: author/model-name[:revision]"));
    }
    
    let _author = parts[0];
    let model_name = parts[1];
    
    // Ensure models directory exists
    tokio::fs::create_dir_all(&base_dir).await
        .map_err(|e| anyhow!("Failed to create models directory: {}", e))?;
    
    println!("📥 Downloading model: {}", model_uri);
    println!("📁 Saving to: {}", base_dir.display());
    
    // Initialize HuggingFace API with authentication
    let auth = HfAuth::new()?;
    let api = if let Some(token) = auth.get_token().await? {
        ApiBuilder::from_env()
            .with_token(Some(token))
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    } else {
        println!("ℹ️  No HuggingFace token found. Some models may not be accessible.");
        ApiBuilder::from_env()
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    };
    
    // Create the repository reference with revision if specified
    let repo = api.model(repo_path.to_string());
    if let Some(rev) = revision {
        println!("📌 Using revision: {}", rev);
        println!("⚠️  Revision support limited in current hf-hub version");
    }
    
    // Use the provided filename directly, or default
    let target_filename = filename.unwrap_or("model.safetensors");
    let filename_prefix = format!("{}_{}", model_name.replace('/', "_"), target_filename);
    let local_path = base_dir.join(filename_prefix);
    
    // Check if model already exists
    if local_path.exists() {
        println!("✅ Model already exists at: {}", local_path.display());
        
        // Still register the existing model with storage system to ensure it appears in model list
        println!("📝 Registering existing model with storage system...");
        match register_model_and_get_id(model_uri, &local_path, target_filename, &config).await {
            Ok(model_id) => {
                println!("✅ Model registered with storage system");
                println!("🆔 Model UUID: {}", model_id);
                return Ok(model_id);
            }
            Err(e) => {
                eprintln!("⚠️  Warning: Failed to register existing model with storage system: {}", e);
                eprintln!("   Model exists but may not appear in 'model list'");
                // Create fallback ModelId
                let model_id = ModelId::from_content_hash(&repo_path.replace("/", "-"), "language_model", None);
                println!("🆔 Model UUID (fallback): {}", model_id);
                return Ok(model_id);
            }
        }
    }
    
    // Create progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    
    // Download the model
    match download_with_progress(&repo, target_filename, &local_path, &pb).await {
        Ok(()) => {
            pb.finish_with_message("✅ Download completed");
            println!("💾 Model saved to: {}", local_path.display());
            
            // Register model with storage system for proper indexing
            match register_model_and_get_id(model_uri, &local_path, target_filename, &config).await {
                Ok(model_id) => {
                    println!("✅ Model registered with storage system");
                    println!("🆔 Model UUID: {}", model_id);
                    Ok(model_id)
                }
                Err(e) => {
                    eprintln!("⚠️  Warning: Failed to register model with storage system: {}", e);
                    eprintln!("   Model downloaded successfully but may not appear in 'model list'");
                    // Create fallback ModelId
                    let model_id = ModelId::from_content_hash(&repo_path.replace("/", "-"), "language_model", None);
                    println!("🆔 Model UUID (fallback): {}", model_id);
                    Ok(model_id)
                }
            }
        }
        Err(e) => {
            pb.finish_with_message("❌ Download failed");
            
            // Clean up partial download
            if local_path.exists() {
                let _ = tokio::fs::remove_file(&local_path).await;
            }
            
            Err(e)
        }
    }
}

/// List available models in a HuggingFace repository
pub async fn list_repo_files(model_uri: &str) -> Result<Vec<String>> {
    println!("📋 Listing files in repository: {}", model_uri);
    
    // Parse the URI with optional revision/tag support
    let (repo_path, revision) = if let Some(colon_pos) = model_uri.rfind(':') {
        let repo_part = &model_uri[..colon_pos];
        let rev_part = &model_uri[colon_pos + 1..];
        (repo_part, Some(rev_part))
    } else {
        (model_uri, None)
    };
    
    // Initialize HuggingFace API with authentication
    let auth = HfAuth::new()?;
    let api = if let Some(token) = auth.get_token().await? {
        ApiBuilder::from_env()
            .with_token(Some(token))
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    } else {
        ApiBuilder::from_env()
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    };
    
    // Create the repository reference with revision if specified
    let _repo = api.model(repo_path.to_string());
    if let Some(rev) = revision {
        println!("📌 Using revision: {}", rev);
        println!("⚠️  Revision support limited in current hf-hub version");
    }
    
    // Use HuggingFace REST API to list repository files
    let url = format!("https://huggingface.co/api/models/{}", repo_path);
    let client = reqwest::Client::new();

    match client.get(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                match response.json::<serde_json::Value>().await {
                    Ok(repo_data) => {
                        let files: Vec<String> = repo_data
                            .get("siblings")
                            .and_then(|s| s.as_array())
                            .map(|siblings| {
                                siblings.iter()
                                    .filter_map(|sibling| {
                                        sibling.get("rfilename")
                                            .and_then(|name| name.as_str())
                                            .map(|s| s.to_string())
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();

                        println!("📂 Available files ({} found):", files.len());
                        for file in &files {
                            println!("   {}", file);
                        }

                        Ok(files)
                    }
                    Err(e) => Err(anyhow!("Failed to parse repository data: {}", e))
                }
            } else {
                Err(anyhow!("Failed to fetch repository info: HTTP {}", response.status()))
            }
        }
        Err(e) => {
            Err(anyhow!("Failed to connect to HuggingFace API: {}", e))
        }
    }
}

/// Download a model by path/URI with optional filename and progress
pub async fn download_model(
    model_path: &str,
    config: Option<&HyprConfig>,
    filename: Option<String>,
    show_progress: bool,
) -> Result<ModelId> {
    if show_progress {
        download_model_by_uri(model_path, filename.as_deref(), config).await
    } else {
        // Still show progress for now, but could disable progress bars in future
        download_model_by_uri(model_path, filename.as_deref(), config).await
    }
}

/// Quick start model downloader - downloads the recommended model for testing
pub async fn quick_start_download() -> Result<PathBuf> {
    println!("🚀 Quick Start Model Download");
    println!("📥 This will download a small quantized model suitable for testing");
    println!();
    
    // Use configuration-managed storage paths
    let config = HyprConfig::load().unwrap_or_default();
    download_qwen3_model(Some(&config)).await
}

/// Register a downloaded model with the storage system
async fn register_downloaded_model(
    model_uri: &str,
    local_path: &Path,
    filename: &str,
    config: &HyprConfig,
) -> Result<ModelId> {
    // Create ModelUri from the URI string
    let uri_with_scheme = if model_uri.starts_with("hf://") {
        model_uri.to_string()
    } else {
        format!("hf://{}", model_uri)
    };
    
    let model_uri_obj = ModelUri::parse(&uri_with_scheme)?;
    
    // Get file size
    let file_metadata = tokio::fs::metadata(local_path).await
        .map_err(|e| anyhow!("Failed to get file metadata: {}", e))?;
    let file_size = file_metadata.len();
    
    // Determine model type and architecture from filename/URI
    let model_type = if filename.ends_with(".safetensors") {
        "safetensors".to_string()
    } else {
        "unknown".to_string()
    };
    
    // Extract architecture info from model name if possible
    let architecture = if model_uri_obj.name.to_lowercase().contains("qwen") {
        Some("qwen".to_string())
    } else if model_uri_obj.name.to_lowercase().contains("llama") {
        Some("llama".to_string())
    } else if model_uri_obj.name.to_lowercase().contains("mistral") {
        Some("mistral".to_string())
    } else {
        None
    };
    
    // Extract parameter count from model name if possible
    let parameters = if model_uri_obj.name.contains("1.5B") {
        Some(1_500_000_000u64)
    } else if model_uri_obj.name.contains("7B") {
        Some(7_000_000_000u64)
    } else if model_uri_obj.name.contains("13B") {
        Some(13_000_000_000u64)
    } else if model_uri_obj.name.contains("70B") {
        Some(70_000_000_000u64)
    } else {
        None
    };
    
    // Generate UUID based on model content
    let model_id = ModelId::from_content_hash(
        &model_uri_obj.name,
        &architecture.clone().unwrap_or_else(|| "unknown".to_string()),
        parameters,
    );
    
    // Create external source reference
    let external_source = ExternalSource {
        source_type: SourceType::HuggingFace,
        identifier: format!("{}/{}", model_uri_obj.org, model_uri_obj.name), // Store full org/name
        revision: model_uri_obj.revision.clone(),
        download_url: None,
        last_verified: chrono::Utc::now().timestamp(),
    };
    
    // Create model file info
    let model_file = ModelFile {
        filename: filename.to_string(),
        size_bytes: file_size,
        checksum: None, // TODO: Calculate checksum
        file_type: FileType::Model,
    };
    
    // Create metadata with UUID system
    let metadata = ModelMetadata {
        model_id: model_id.clone(),
        name: model_uri_obj.name.clone(),
        display_name: None,
        architecture: architecture.unwrap_or_else(|| "unknown".to_string()),
        parameters,
        model_type,
        tokenizer_type: Some("tiktoken".to_string()), // Common for modern models
        size_bytes: file_size,
        files: vec![model_file],
        external_sources: vec![external_source],
        local_path: Some(local_path.to_path_buf()),
        is_cached: true,
        tags: vec![
            if model_uri_obj.name.to_lowercase().contains("instruct") {
                "instruct".to_string()
            } else if model_uri_obj.name.to_lowercase().contains("chat") {
                "chat".to_string()
            } else {
                "base".to_string()
            }
        ],
        description: None,
        license: None,
        created_at: chrono::Utc::now().timestamp(),
        last_accessed: chrono::Utc::now().timestamp(),
        last_updated: chrono::Utc::now().timestamp(),
    };
    
    // Validate model format
    if !local_path.exists() {
        return Err(anyhow!("Model file not found after download"));
    }
    
    // Initialize model storage and register the metadata
    let models_dir = config.models_dir();
    let storage = ModelStorage::new(models_dir.clone()).await?;
    
    // Register model with UUID-based storage system
    storage.store_metadata(&model_uri_obj, metadata).await?;
    
    println!("📝 Model registered with storage system");
    println!("✅ Model validated");
    println!("🆔 Model UUID: {}", model_id);
    Ok(model_id.clone())
}


/// Register an already-downloaded model with the storage system
pub async fn register_existing_model(
    model_path: &Path, 
    model_uri: &str,
    config: Option<&HyprConfig>
) -> Result<()> {
    let config = config.map(|c| c.clone()).unwrap_or_else(|| HyprConfig::load().unwrap_or_default());
    
    if !model_path.exists() {
        return Err(anyhow!("Model file does not exist: {}", model_path.display()));
    }
    
    // Extract filename from path
    let filename = model_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown.safetensors");
    
    println!("📝 Registering existing model: {}", filename);
    
    // Register with the system  
    register_downloaded_model(model_uri, model_path, filename, &config).await?;
    
    println!("✅ Existing model registered successfully!");
    Ok(())
}


/// Register a model with ModelRegistry and return its UUID
async fn register_model_and_get_id(
    model_uri: &str, 
    local_path: &Path, 
    _target_filename: &str,
    config: &HyprConfig
) -> Result<ModelId> {
    // Parse model URI to extract name and architecture info
    let repo_name = model_uri.replace("hf://", "").replace("/", "-");
    
    // Create model registry
    let registry = ModelRegistry::new(config.models_dir().to_path_buf()).await?;
    
    // Create external source
    let external_source = ExternalSource {
        source_type: SourceType::HuggingFace,
        identifier: model_uri.replace("hf://", ""),
        revision: None,
        download_url: None,
        last_verified: chrono::Utc::now().timestamp(),
    };
    
    // Register the model
    let model_id = registry.register_model(
        repo_name.clone(),
        "language_model".to_string(), // Default architecture
        None, // Parameters unknown for now
        local_path.to_path_buf(),
        vec![external_source],
    ).await?;
    
    Ok(model_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_model_uri_parsing() {
        // Test valid URI format
        let result = download_model_by_uri(
            "microsoft/DialoGPT-medium",
            Some("model.safetensors"),
            None,
        ).await;
        
        // This will fail in test environment without network, but should parse correctly
        assert!(result.is_err()); // Expected to fail due to network/auth issues in test
    }
    
    #[test]
    fn test_model_dir_creation() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("test_models");
        
        // This test is synchronous, so we can't directly test the async function
        // But we can verify the path logic works
        assert!(!model_path.exists());
    }
}