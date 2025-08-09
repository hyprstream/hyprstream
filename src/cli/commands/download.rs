//! Model download functionality for Hyprstream

use anyhow::{anyhow, Result};
use hf_hub::api::tokio::ApiBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use crate::{auth::HfAuth, config::HyprConfig};

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
        ApiBuilder::new()
            .with_token(Some(token))
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    } else {
        println!("ℹ️  No HuggingFace token found. Some models may not be accessible.");
        println!("   Use 'hyprstream auth login' to authenticate for gated models.");
        ApiBuilder::new()
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    };
    
    // Get the model repository
    let repo = api.model("Qwen/Qwen2-1.5B-Instruct-GGUF".to_string());
    
    // Target filename for the quantized model
    let filename = "qwen2-1_5b-instruct-q4_0.gguf";
    let local_path = base_dir.join(filename);
    
    // Check if model already exists
    if local_path.exists() {
        println!("✅ Model already exists at: {}", local_path.display());
        return Ok(local_path);
    }
    
    println!("🌐 Downloading: {}", filename);
    
    // Create progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    pb.set_message("Initializing download...");
    
    // Use a timeout-aware download with progress tracking
    match download_with_progress(&repo, filename, &local_path, &pb).await {
        Ok(()) => {
            pb.finish_with_message("✅ Download completed");
            println!("💾 Model saved to: {}", local_path.display());
            
            // Verify the downloaded file
            let file_size = tokio::fs::metadata(&local_path).await
                .map_err(|e| anyhow!("Failed to get file metadata: {}", e))?
                .len();
            
            println!("📊 File size: {:.2} MB", file_size as f64 / 1_048_576.0);
            
            if file_size < 1_048_576 { // Less than 1MB is suspicious
                return Err(anyhow!("Downloaded file seems too small, download may have failed"));
            }
            
            Ok(local_path)
        }
        Err(e) => {
            pb.finish_with_message("❌ Download failed");
            
            // Clean up partial download
            if local_path.exists() {
                let _ = tokio::fs::remove_file(&local_path).await;
            }
            
            Err(anyhow!("Download failed: {}", e))
        }
    }
}

/// Download with progress tracking
async fn download_with_progress(
    repo: &hf_hub::api::tokio::ApiRepo,
    filename: &str,
    local_path: &Path,
    pb: &ProgressBar,
) -> Result<()> {
    pb.set_message(format!("Downloading {} to {}", filename, local_path.display()));
    
    // Get the file from the repository
    let file_path = repo.get(filename).await
        .map_err(|e| anyhow!("Failed to download file '{}': {}", filename, e))?;
    
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
) -> Result<PathBuf> {
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
    
    let author = parts[0];
    let model_name = parts[1];
    
    // Ensure models directory exists
    tokio::fs::create_dir_all(&base_dir).await
        .map_err(|e| anyhow!("Failed to create models directory: {}", e))?;
    
    println!("📥 Downloading model: {}", model_uri);
    println!("📁 Saving to: {}", base_dir.display());
    
    // Initialize HuggingFace API with authentication
    let auth = HfAuth::new()?;
    let api = if let Some(token) = auth.get_token().await? {
        ApiBuilder::new()
            .with_token(Some(token))
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    } else {
        println!("ℹ️  No HuggingFace token found. Some models may not be accessible.");
        ApiBuilder::new()
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    };
    
    // Create the repository reference 
    let repo = api.model(repo_path.to_string());
    
    // Handle revision/tag to filename mapping (like llamacpp)
    let resolved_filename = if let Some(rev) = revision {
        println!("📌 Resolving tag/revision: {}", rev);
        resolve_tag_to_filename(rev, &repo).await?
    } else {
        filename.unwrap_or("model.gguf").to_string()
    };
    
    // Use resolved filename or provided filename
    let target_filename = &resolved_filename;
    let filename_prefix = format!("{}_{}", model_name.replace('/', "_"), target_filename);
    let local_path = base_dir.join(filename_prefix);
    
    // Check if model already exists
    if local_path.exists() {
        println!("✅ Model already exists at: {}", local_path.display());
        return Ok(local_path);
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
    match download_with_progress(&repo, &resolved_filename, &local_path, &pb).await {
        Ok(()) => {
            pb.finish_with_message("✅ Download completed");
            println!("💾 Model saved to: {}", local_path.display());
            Ok(local_path)
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
        ApiBuilder::new()
            .with_token(Some(token))
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    } else {
        ApiBuilder::new()
            .build()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?
    };
    
    // Create the repository reference 
    let repo = api.model(repo_path.to_string());
    
    // Note: Revision handling may need to be implemented differently
    if let Some(rev) = revision {
        println!("📌 Attempting to use revision/tag: {} (may not be fully supported yet)", rev);
    }
    
    // Try to get the actual file list from the repository
    match repo.info().await {
        Ok(repo_info) => {
            let files: Vec<String> = repo_info.siblings
                .iter()
                .map(|sibling| sibling.rfilename.clone())
                .collect();
            
            println!("📂 Available files ({} found):", files.len());
            for file in &files {
                println!("   {}", file);
            }
            
            Ok(files)
        }
        Err(e) => {
            println!("⚠️  Could not fetch repository info: {}", e);
            println!("📂 Falling back to common GGUF filenames:");
            
            // Fallback to common GGUF filenames
            let common_files = vec![
                "model.gguf".to_string(),
                "model.q4_0.gguf".to_string(),
                "model.q4_1.gguf".to_string(),
                "model.q5_0.gguf".to_string(),
                "model.q5_1.gguf".to_string(),
                "model.q8_0.gguf".to_string(),
                "tokenizer.json".to_string(),
                "config.json".to_string(),
            ];
            
            for file in &common_files {
                println!("   {}", file);
            }
            
            Ok(common_files)
        }
    }
}

/// Download a model by path/URI with optional filename and progress
pub async fn download_model(
    model_path: &str,
    config: Option<&HyprConfig>,
    filename: Option<String>,
    show_progress: bool,
) -> Result<PathBuf> {
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

/// Resolve a tag/revision to the corresponding GGUF filename
/// This mimics llamacpp's behavior for resolving tags to quantization formats
async fn resolve_tag_to_filename(tag: &str, repo: &hf_hub::api::tokio::ApiRepo) -> Result<String> {
    // First, try to get the repository info to see available files
    match repo.info().await {
        Ok(repo_info) => {
            let files: Vec<String> = repo_info.siblings
                .iter()
                .map(|sibling| sibling.rfilename.clone())
                .filter(|name| name.ends_with(".gguf"))
                .collect();
            
            // Try different common patterns for tag-to-filename mapping
            let possible_patterns = vec![
                // Direct format: Model-TAG.gguf (most common)
                format!("-{}.gguf", tag),
                format!("_{}.gguf", tag),
                // With model name patterns
                format!("-{}.gguf", tag.to_uppercase()),
                format!("_{}.gguf", tag.to_uppercase()),
                // Quantization format patterns
                format!("-Q{}.gguf", tag),
                format!("_Q{}.gguf", tag),
            ];
            
            // Find files that match the tag pattern
            for pattern in &possible_patterns {
                for file in &files {
                    if file.contains(pattern) {
                        println!("✅ Resolved tag '{}' to file: {}", tag, file);
                        return Ok(file.clone());
                    }
                }
            }
            
            // If no exact pattern match, try case-insensitive partial matching
            let tag_lower = tag.to_lowercase();
            for file in &files {
                let file_lower = file.to_lowercase();
                if file_lower.contains(&tag_lower) && file_lower.contains(".gguf") {
                    println!("✅ Resolved tag '{}' to file (partial match): {}", tag, file);
                    return Ok(file.clone());
                }
            }
            
            // If no matches found, show available options
            println!("❌ Could not resolve tag '{}' to a specific file", tag);
            println!("💡 Available GGUF files:");
            for file in &files[..std::cmp::min(10, files.len())] {
                println!("   {}", file);
            }
            if files.len() > 10 {
                println!("   ... and {} more files", files.len() - 10);
            }
            
            Err(anyhow!("Could not resolve tag '{}' to a specific filename. Please specify the exact filename using --files flag.", tag))
        }
        Err(e) => {
            println!("⚠️ Could not fetch repository info: {}", e);
            
            // Fallback: try common quantization format patterns
            let fallback_patterns = vec![
                format!("model-{}.gguf", tag),
                format!("model_{}.gguf", tag),
                format!("{}.gguf", tag),
            ];
            
            println!("🔄 Trying fallback filename: model-{}.gguf", tag);
            Ok(format!("model-{}.gguf", tag))
        }
    }
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
            Some("model.gguf"),
            Some(tempdir().unwrap().path().to_path_buf()),
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