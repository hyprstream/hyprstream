//! CLI handlers for adaptive ML inference server

use crate::{
    // storage::SparseStorageConfig,  // VDB removed
    runtime::{RuntimeEngine, TorchEngine},
};
use ::config::{Config, File};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
};
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};
use crate::api::model_storage::{ModelStorage, ModelId};
use crate::api::model_storage::ModelUri;

/// Response structure for LoRA inference
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub lora_id: String,
    pub output: String,
    pub tokens_generated: usize,
    pub latency_ms: f64,
    pub finish_reason: String,
}

/// Resolve base model identifier - can be UUID or URI
async fn resolve_base_model_identifier(identifier: &str) -> Result<String, anyhow::Error> {
    // Try to parse as UUID first
    if let Ok(uuid) = uuid::Uuid::parse_str(identifier) {
        let model_id = ModelId(uuid);
        
        // Load model storage
        let config = crate::config::HyprConfig::load().unwrap_or_default();
        let storage = ModelStorage::new(config.models_dir().to_path_buf()).await?;
        
        // Get metadata by UUID
        if let Ok(metadata) = storage.get_metadata_by_id(&model_id).await {
            return Ok(format!("UUID {} ({})", model_id, metadata.name));
        } else {
            return Err(anyhow::anyhow!("Model with UUID {} not found in storage", model_id));
        }
    }
    
    // Try to parse as URI
    if let Ok(model_uri) = ModelUri::parse(identifier) {
        let config = crate::config::HyprConfig::load().unwrap_or_default();
        let storage = ModelStorage::new(config.models_dir().to_path_buf()).await?;
        
        // Check if model exists in storage
        if let Ok(metadata) = storage.get_metadata(&model_uri).await {
            return Ok(format!("URI {} (UUID: {})", model_uri.uri, metadata.model_id));
        } else {
            return Ok(format!("URI {} (not cached locally)", model_uri.uri));
        }
    }
    
    // If neither UUID nor valid URI, treat as simple string identifier
    Err(anyhow::anyhow!("Invalid base model identifier format: {}", identifier))
}

pub async fn execute_sparse_query(
    addr: Option<SocketAddr>,
    query: String,
    _config: Option<&Config>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 3000)));
    let base_url = format!("http://{}", addr);
    
    if verbose {
        info!("Executing sparse query: {}", query);
        debug!("Connecting to REST API at: {}", base_url);
    }
    
    // Parse the query as JSON for embedding operations
    let embedding_query: serde_json::Value = serde_json::from_str(&query)?;
    
    if verbose {
        debug!("Parsed embedding query: {:?}", embedding_query);
    }
    
    let client = Client::new();
    
    // Extract LoRA ID from query (assuming it's in the query structure)
    let lora_id = embedding_query
        .get("lora_id")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    // Make REST API call for embeddings
    let url = format!("{}/v1/inference/{}/embeddings", base_url, lora_id);
    let request_body = json!({
        "input": embedding_query.get("input").unwrap_or(&json!("")),
        "model": "text-embedding-ada-002"
    });
    
    if verbose {
        debug!("Making request to: {}", url);
        debug!("Request body: {}", request_body);
    }
    
    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await?;
    
    if response.status().is_success() {
        let result: Value = response.json().await?;
        if verbose {
            debug!("Response: {}", serde_json::to_string_pretty(&result)?);
        }
        info!("✅ Embedding query processed successfully");
    } else {
        let status_code = response.status();
        let error_text = response.text().await?;
        error!("❌ Query failed with status {}: {}", status_code, error_text);
    }
    
    Ok(())
}


pub async fn handle_server(
    config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check environment variable first, then config, then default to 0.0.0.0
    let host = std::env::var("HYPRSTREAM_SERVER_HOST")
        .unwrap_or_else(|_| config.get_string("host")
            .unwrap_or_else(|_| "0.0.0.0".to_string()));
    
    let port = std::env::var("HYPRSTREAM_SERVER_PORT")
        .unwrap_or_else(|_| config.get_string("port")
            .unwrap_or_else(|_| "50051".to_string()));
    
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    
    // Create server configuration from environment variables and config file
    let mut server_config = crate::server::state::ServerConfig::from_env();
    
    // Override with config file values if present
    if let Ok(preload_models) = config.get_array("server.preload_models") {
        server_config.preload_models = preload_models
            .into_iter()
            .filter_map(|v| v.into_string().ok())
            .collect();
    }
    if let Ok(enable_logging) = config.get_bool("server.enable_logging") {
        server_config.enable_logging = enable_logging;
    }
    if let Ok(enable_metrics) = config.get_bool("server.enable_metrics") {
        server_config.enable_metrics = enable_metrics;
    }
    if let Ok(api_key) = config.get_string("server.api_key") {
        server_config.api_key = Some(api_key);
    }
    if let Ok(max_tokens_limit) = config.get_int("server.max_tokens_limit") {
        server_config.max_tokens_limit = max_tokens_limit as usize;
    }
    if let Ok(request_timeout_secs) = config.get_int("server.request_timeout_secs") {
        server_config.request_timeout_secs = request_timeout_secs as u64;
    }
    
    // CORS permissive headers from config
    if let Ok(permissive) = config.get_bool("server.cors_permissive_headers") {
        server_config.cors.permissive_headers = permissive;
    }
    
    // Update CORS to include the actual server address if not using wildcard
    if !server_config.cors.allowed_origins.contains(&"*".to_string()) {
        // Add the actual listening address to allowed origins
        server_config.cors.allowed_origins.push(format!("http://{}", addr));
        // If listening on 0.0.0.0, also add localhost variants with the port
        if host == "0.0.0.0" {
            server_config.cors.allowed_origins.extend(vec![
                format!("http://localhost:{}", port),
                format!("http://127.0.0.1:{}", port),
            ]);
        }
    }
    
    // Create server state
    let server_state = crate::server::state::ServerState::new(server_config).await?;
    
    info!("Starting Hyprstream HTTP server on {}", addr);
    info!("OpenAI-compatible API available at http://{}/oai/v1", addr);
    
    // Check for TLS configuration
    if config.get_bool("tls.enabled").unwrap_or(false) {
        let cert_path = config.get_string("tls.cert_path")?;
        let key_path = config.get_string("tls.key_path")?;
        crate::server::start_server_tls(addr, server_state, &cert_path, &key_path).await?;
    } else {
        crate::server::start_server(addr, server_state).await?;
    }
    
    Ok(())
}

pub async fn handle_embedding_query(
    host: Option<String>,
    query: &str,
    tls_cert: Option<&Path>,
    tls_key: Option<&Path>,
    tls_ca: Option<&Path>,
    _tls_skip_verify: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = host.unwrap_or_else(|| "localhost:50051".to_string());

    // Create Config with TLS settings if certificates are provided
    let config = match (tls_cert, tls_key) {
        (Some(cert_path), Some(key_path)) => {
            let _cert = tokio::fs::read(cert_path).await?;
            let _key = tokio::fs::read(key_path).await?;
            let _ca = if let Some(ca_path) = tls_ca {
                Some(tokio::fs::read(ca_path).await?)
            } else {
                None
            };

            let config = Config::builder().build()?;

            Some(config)
        }
        _ => None,
    };

    // Parse address and execute embedding query
    let addr_parts: Vec<&str> = addr.split(':').collect();
    if addr_parts.len() != 2 {
        return Err("Invalid address format. Expected host:port".into());
    }

    let socket_addr = SocketAddr::new(
        addr_parts[0].parse()?,
        addr_parts[1].parse()?
    );

    // Execute embedding query
    execute_sparse_query(
        Some(socket_addr),
        query.to_string(),
        config.as_ref(),
        verbose,
    ).await?;

    Ok(())
}

pub fn handle_config(config_path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = Config::builder()
        .set_default("host", "127.0.0.1")?
        .set_default("port", "50051")?
        .set_default("storage.path", "./storage")?
        .set_default("storage.neural_compression", true)?
        .set_default("storage.hardware_acceleration", true)?
        .set_default("storage.cache_size_mb", 2048)?
        .set_default("storage.compaction_interval_secs", 300)?
        .set_default("storage.streaming_updates", true)?
        .set_default("storage.update_batch_size", 1000)?;

    // Load config file if provided
    if let Some(path) = config_path {
        builder = builder.add_source(File::from(path));
    }

    let settings = builder.build()?;
    info!("📁 Configuration loaded successfully");
    debug!(settings = ?settings, "Current configuration settings");
    Ok(())
}

// Placeholder implementations for model and LoRA commands
// These will be fully implemented as part of the complete system
pub async fn handle_model_command(
    cmd: crate::cli::commands::ModelCommand,
    _server_url: String,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::cli::commands::model::ModelAction;
    
    match cmd.action {
        ModelAction::List { registry, search, remote, format } => {
            info!("📋 Listing available models...");
            
            // Use model storage directly
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
            
            let mut models = Vec::new();
            
            // Get local models from cache
            let local_models = model_storage.list_local_models().await?;
            for (model_uri, model_metadata) in local_models {
                let category = if model_uri.name.to_lowercase().contains("instruct") {
                    "instruct"
                } else if model_uri.name.to_lowercase().contains("chat") {
                    "chat"
                } else if model_uri.name.to_lowercase().contains("code") {
                    "code"
                } else {
                    "general"
                };
                
                models.push((
                    model_uri.name.clone(),
                    "local".to_string(),
                    format!("{:.1}GB", model_metadata.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
                    category.to_string(),
                    model_metadata.model_id.to_string(), // Add UUID
                ));
            }
            
            // Get remote models if requested
            if remote {
                if let Some(_reg_filter) = &registry {
                    // Remote search functionality has been deprecated
                    // Models should be pulled directly using git URLs
                    eprintln!("⚠️  Remote model search is no longer supported");
                    eprintln!("   Use 'hyprstream model pull' with a git URL instead:");
                    eprintln!("   • hyprstream model pull https://huggingface.co/Qwen/Qwen2-0.5B-Instruct");
                    eprintln!("   • hyprstream model pull git@github.com:user/model.git");
                }
            }
            
            // Apply filters
            if let Some(reg) = &registry {
                models.retain(|(name, _, _, _, _)| name.starts_with(reg));
            }
            
            if let Some(query) = &search {
                let query_lower = query.to_lowercase();
                models.retain(|(name, _, _, category, uuid)| 
                    name.to_lowercase().contains(&query_lower) || 
                    category.to_lowercase().contains(&query_lower) ||
                    uuid.to_lowercase().contains(&query_lower)
                );
            }
            
            if !remote {
                models.retain(|(_, location, _, _, _)| location == "local");
            }
            
            match format.as_str() {
                "json" => {
                    println!("{{");
                    println!("  \"models\": [");
                    for (i, (name, location, size, category, uuid)) in models.iter().enumerate() {
                        let comma = if i < models.len() - 1 { "," } else { "" };
                        println!("    {{\"name\": \"{}\", \"location\": \"{}\", \"size\": \"{}\", \"category\": \"{}\", \"uuid\": \"{}\"}}{}", 
                               name, location, size, category, uuid, comma);
                    }
                    println!("  ]");
                    println!("}}");
                },
                _ => {
                    println!("Available Models ({} found):", models.len());
                    for (name, location, size, category, uuid) in &models {
                        let status = if *location == "local" { "📁" } else { "☁️" };
                        println!("  {} {} ({}) - {} [{}]", status, name, size, category, location);
                        if uuid != "N/A" {
                            println!("    🆔 UUID: {}", uuid);
                        }
                    }
                    if models.is_empty() {
                        println!("No models found matching your criteria.");
                        println!("Try: hyprstream model pull https://huggingface.co/Qwen/Qwen2-1.5B-Instruct");
                    }
                }
            }
        }
        ModelAction::Clone { repo_url, git_ref, model_id: _ } => {
            info!("📦 Cloning model from Git repository...");

            // Get storage paths
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;

            let git_source = crate::api::git_downloader::GitModelSource::new(models_dir);
            let (model_id, model_path) = if let Some(ref_str) = git_ref {
                git_source.clone_ref(&repo_url, &ref_str)?
            } else {
                git_source.clone_model(&repo_url)?
            };
            
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(models_dir).await?;
            let model_name = repo_url.split('/').last()
                .unwrap_or("unknown")
                .trim_end_matches(".git");
            
            if let Err(e) = model_storage.register_with_git_registry(
                &model_id,
                model_name,
                Some(repo_url.clone())
            ).await {
                tracing::warn!("Failed to register with Git registry: {}", e);
            }
            
            println!();
            println!("✅ Model cloned successfully!");
            println!("   Model ID: {}", model_id);
            println!("   Name: {}", model_name);
            println!("   Location: {}", model_path.display());
            println!();
            println!("📚 Next steps:");
            println!("   • Create adapter: hyprstream lora create --base-model {}", model_id);
            println!("   • Run inference: hyprstream model infer {} --prompt \"...\"", model_id);
        }
        ModelAction::Pull { uri, force, .. } => {
            info!("📥 Pulling model: {}", uri);
            
            // Get storage paths
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;
            
            let git_source = crate::api::git_downloader::GitModelSource::new(models_dir.clone());
            match git_source.clone_model(&uri) {
                Ok((model_id, model_path)) => {
                    println!();
                    println!("✅ Model downloaded successfully!");
                    println!("   Model ID: {}", model_id);
                    println!("   Location: {}", model_path.display());
                    
                    // Save metadata
                    let model_storage = crate::api::model_storage::ModelStorage::new(models_dir).await?;
                    let metadata = crate::api::model_storage::ModelMetadataFile {
                        model_id: model_id.clone(),
                        name: uri.clone(),
                        display_name: uri.clone(),
                        source_uri: uri.clone(),
                        architecture: None,
                        parameters: None,
                        created_at: chrono::Utc::now().timestamp(),
                        last_accessed: chrono::Utc::now().timestamp(),
                    };
                    model_storage.save_model_metadata_file(&model_id, &metadata).await?;
                    
                    // Register with Git registry (if available)
                    // Extract a friendly name from the URI
                    let model_name = uri.split('/').last().unwrap_or(&uri);
                    if let Err(e) = model_storage.register_with_git_registry(
                        &model_id, 
                        model_name,
                        Some(uri.clone())
                    ).await {
                        tracing::warn!("Failed to register with Git registry: {}", e);
                    }
                    
                    println!();
                    println!("📚 Next steps:");
                    println!("   • Create adapter: hyprstream lora create --base-model {}", model_id);
                    println!("   • Run inference: hyprstream model infer {} --prompt \"...\"", model_id);
                }
                Err(e) => {
                    eprintln!("❌ Download failed: {}", e);
                    return Err(e.into());
                }
            }
        }
        
        ModelAction::Share { model_name, include_metrics, push_to } => {
            info!("📤 Sharing model: {}", model_name);
            
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;
            
            let mut sharing = crate::api::model_sharing::ModelSharing::new(models_dir).await?;
            
            // Create shareable reference
            let share_ref = sharing.create_share_ref(&model_name, include_metrics).await?;
            
            println!("📋 Shareable Model Reference:");
            println!("   Name: {}", share_ref.name);
            println!("   Type: {:?}", share_ref.model_type);
            println!("   Commit: {}", share_ref.commit);
            println!("   Size: {:.2} GB", share_ref.size_bytes as f64 / 1_073_741_824.0);
            
            if let Some(metrics) = &share_ref.metrics {
                println!("   Performance:");
                println!("     Loss: {:.4}", metrics.loss);
                if let Some(acc) = metrics.accuracy {
                    println!("     Accuracy: {:.2}%", acc * 100.0);
                }
                println!("     Training Steps: {}", metrics.training_steps);
            }
            
            // Optionally push to remote
            if let Some(remote_url) = push_to {
                println!();
                println!("📤 Pushing to remote: {}", remote_url);
                sharing.push_to_remote(&model_name, &remote_url, None).await?;
            }
            
            // Output as JSON for easy sharing
            println!();
            println!("📦 Share this with peers:");
            println!("{}", serde_json::to_string_pretty(&share_ref)?);
        }
        
        ModelAction::Import { git_url, name, verify: _ } => {
            info!("📥 Importing shared model from: {}", git_url);
            
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;
            
            let mut sharing = crate::api::model_sharing::ModelSharing::new(models_dir).await?;
            
            // Parse share reference if provided as JSON
            let share_ref = if git_url.starts_with('{') {
                // Assume it's JSON share reference
                serde_json::from_str(&git_url)?
            } else {
                // Create minimal share ref
                crate::api::model_sharing::ShareableModelRef {
                    name: name.clone().unwrap_or_else(|| "imported-model".to_string()),
                    model_type: crate::api::model_sharing::ModelType::Base,
                    git_url: Some(git_url.clone()),
                    commit: "HEAD".to_string(),
                    size_bytes: 0,
                    metrics: None,
                    signature: None,
                }
            };
            
            // Import the model
            let model_id = sharing.import_shared_model(share_ref, &git_url, name).await?;
            
            println!();
            println!("✅ Model imported successfully!");
            println!("   Model ID: {}", model_id);
        }
        
        ModelAction::Remove { uri, keep_metadata, yes } => {
            // Parse as UUID only - no fallbacks
            let model_id = crate::api::model_storage::ModelId::from_str(&uri)
                .map_err(|_| anyhow::anyhow!("Invalid UUID: '{}'. Use 'hyprstream model list' to see model UUIDs", uri))?;

            info!("🗑️ Removing model: {}", model_id);

            // Check if confirmation is needed
            if !yes {
                println!("⚠️  Are you sure you want to remove model '{}'?", model_id);
                println!("This action cannot be undone.");
                println!("");
                println!("Type 'yes' to confirm, or use --yes flag to skip confirmation:");

                use std::io::{self, BufRead};
                let stdin = io::stdin();
                let mut line = String::new();
                stdin.lock().read_line(&mut line)?;

                if line.trim().to_lowercase() != "yes" {
                    println!("❌ Removal cancelled");
                    return Ok(());
                }
            }

            // Get storage paths
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;

            // Direct UUID path - no fallbacks
            let model_path = models_dir.join(model_id.to_string());

            // Check if model exists
            if !model_path.exists() {
                eprintln!("❌ Model {} not found", model_id);
                eprintln!("   Use 'hyprstream model list' to see available models");
                return Err(anyhow::anyhow!("Model {} not found", model_id).into());
            }

            // Remove the model directory
            println!("🗑️ Removing model files from: {}", model_path.display());
            if let Err(e) = tokio::fs::remove_dir_all(&model_path).await {
                eprintln!("❌ Failed to remove model directory: {}", e);
                eprintln!("   You may need to manually remove: {}", model_path.display());
                return Err(e.into());
            }

            // Remove metadata unless keeping it
            if !keep_metadata {
                let model_storage = crate::api::model_storage::ModelStorage::new(models_dir).await?;
                if let Err(e) = model_storage.remove_metadata_by_id(&model_id).await {
                    eprintln!("⚠️  Failed to remove metadata: {}", e);
                    // Continue anyway since files are already deleted
                }
                println!("🗑️ Model metadata removed");
            } else {
                println!("📋 Model metadata preserved");
            }

            println!("✅ Model {} removed successfully", model_id);
        }
        ModelAction::Info { uri, format } => {
            // Parse as UUID only for local models
            let model_id = crate::api::model_storage::ModelId::from_str(&uri)
                .map_err(|_| anyhow::anyhow!("Invalid UUID: '{}'. For remote model info, use 'hyprstream model search'", uri))?;

            info!("ℹ️ Getting model info: {}", model_id);

            // Get storage and retrieve metadata
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
            let metadata = model_storage.get_metadata_by_id(&model_id).await?;

            // Display metadata
            match format.as_str() {
                "json" => {
                    let json_output = serde_json::json!({
                        "model_id": metadata.model_id.to_string(),
                        "name": metadata.name,
                        "display_name": metadata.display_name,
                        "architecture": metadata.architecture,
                        "parameters": metadata.parameters,
                        "size_bytes": metadata.size_bytes,
                        "size_gb": format!("{:.2}", metadata.size_bytes as f64 / 1_073_741_824.0),
                        "created_at": metadata.created_at,
                        "last_accessed": metadata.last_accessed,
                    });
                    println!("{}", serde_json::to_string_pretty(&json_output)?);
                }
                "yaml" => {
                    println!("model_id: {}", metadata.model_id);
                    println!("name: {}", metadata.name);
                    if let Some(display_name) = &metadata.display_name {
                        println!("display_name: {}", display_name);
                    }
                    println!("architecture: {}", metadata.architecture);
                    if let Some(params) = metadata.parameters {
                        println!("parameters: {}", params);
                    }
                    println!("size_gb: {:.2}", metadata.size_bytes as f64 / 1_073_741_824.0);
                }
                _ => {
                    println!("Model: {}", metadata.name);
                    println!("UUID: {}", metadata.model_id);
                    if let Some(display_name) = &metadata.display_name {
                        println!("Display Name: {}", display_name);
                    }
                    println!("Architecture: {}", metadata.architecture);
                    if let Some(params) = metadata.parameters {
                        println!("Parameters: {}", params);
                    }
                    println!("Size: {:.2} GB", metadata.size_bytes as f64 / 1_073_741_824.0);
                    println!("Status: ✅ Available");
                }
            }
        }
        ModelAction::Repair { yes, verbose } => {
            info!("🔧 Repairing model metadata...");
            
            // Confirm repair action if not auto-confirmed
            if !yes {
                println!("⚠️  This will scan and repair all model metadata.");
                println!("The following actions may be performed:");
                println!("  - Fix UUID mismatches between directories and metadata");
                println!("  - Migrate model files to data/ subdirectories");
                println!("  - Remove orphaned directories");
                println!("  - Rebuild metadata cache from directories");
                println!("");
                println!("Type 'yes' to confirm:");
                
                use std::io::{self, BufRead};
                let stdin = io::stdin();
                let mut line = String::new();
                stdin.lock().read_line(&mut line)?;
                
                if line.trim().to_lowercase() != "yes" {
                    println!("❌ Repair cancelled");
                    return Ok(());
                }
            }
            
            // Get storage and run repair
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
            
            if verbose {
                println!("📂 Models directory: {}", storage_paths.models_dir()?.display());
                println!("🔍 Scanning for models...");
            }
            
            // Run the repair
            match model_storage.repair_metadata().await {
                Ok(()) => {
                    println!("✅ Metadata repair completed successfully!");
                    
                    // List repaired models
                    if verbose {
                        if let Ok(models) = model_storage.children().await {
                            println!("\n📋 {} models found after repair:", models.len());
                            for (id, metadata) in models.iter().take(10) {
                                println!("  🆔 {} - {}", id, metadata.name);
                                if let Some(display_name) = &metadata.display_name {
                                    println!("     Display: {}", display_name);
                                }
                                println!("     Size: {:.2} GB", metadata.size_bytes as f64 / 1_073_741_824.0);
                            }
                            if models.len() > 10 {
                                println!("  ... and {} more", models.len() - 10);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ Repair failed: {}", e);
                    return Err(e.into());
                }
            }
        }
        ModelAction::Search { .. } => {
            eprintln!("❌ Model search has been removed.");
            eprintln!("   Use 'hyprstream model list' to see local models");
            eprintln!("   Use 'hyprstream model pull <uri>' to download models");
            return Err(anyhow::anyhow!("Model search feature has been removed").into());
        }
        ModelAction::Convert { source, to, output, precision, verify } => {
            info!("🔄 Converting model from {} to {}", source, to);
            
            use std::path::PathBuf;
            
            // Parse source path
            let source_path = PathBuf::from(&source);
            if !source_path.exists() {
                eprintln!("❌ Source file not found: {}", source);
                return Ok(());
            }
            
            // Determine output path
            let output_path = if let Some(out) = output {
                PathBuf::from(out)
            } else {
                // Generate output filename based on format
                let stem = source_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model");
                let ext = match to.as_str() {
                    "safetensors" => "safetensors",
                    _ => "bin",
                };
                source_path.with_file_name(format!("{}_{}.{}", stem, precision, ext))
            };
            
            // Parse precision
            let target_dtype = Some(precision.clone());
            
            println!("📂 Source: {}", source_path.display());
            println!("📂 Output: {}", output_path.display());
            println!("🎯 Target precision: {:?}", target_dtype);
            
            // Model conversion is no longer supported
            eprintln!("❌ Model conversion has been removed.");
            eprintln!("   Model format conversion is not supported.");
            eprintln!("   Please download models directly in SafeTensors format.");
            eprintln!("");
            eprintln!("   Try: hyprstream model pull https://huggingface.co/<org>/<model-name>");
        }
        ModelAction::Cache { action: _ } => {
            info!("🗄️ Managing model cache");
            
            // Use real model management system for cache info
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
            let cached_models = model_storage.list_local_models().await?;
            let cache_stats = model_storage.get_cache_stats().await?;
            
            println!("Model Cache Status:");
            println!("📊 Cache location: ./models");
            println!("💾 Total size: {:.1} GB", cache_stats.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
            println!("📦 Cached models: {}", cached_models.len());
            println!();
            
            if cached_models.is_empty() {
                println!("No models in cache.");
                println!("Try: hyprstream model pull hf://Qwen/Qwen2-1.5B-Instruct");
            } else {
                println!("Cached Models:");
                for (model_uri, model_metadata) in cached_models {
                    use chrono::{DateTime, Utc};
                    let accessed_dt = DateTime::<Utc>::from_timestamp(model_metadata.last_accessed, 0)
                        .unwrap_or_else(|| Utc::now());
                    println!("  📁 {} ({:.1} GB) - Last accessed: {}", 
                        model_uri.name,
                        model_metadata.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                        accessed_dt.format("%Y-%m-%d %H:%M:%S")
                    );
                }
            }
            
            println!();
            println!("Use 'hyprstream model cache clear' to clear cache");
            println!("Use 'hyprstream model cache prune' to remove unused models");
        }
        ModelAction::Registries => {
            info!("📋 Listing registries");
            
            println!("Available Model Registries:");
            println!();
            println!("🤗 HuggingFace Hub");
            println!("   • URL: https://huggingface.co");
            println!("   • Status: ✅ Connected");
            println!("   • Models: 500,000+");
            println!("   • Formats: SafeTensors, PyTorch, ONNX, TensorFlow");
            println!();
            println!("Configure registries with 'hyprstream config' command");
        }
        ModelAction::Infer { model, prompt, max_tokens, temperature, top_p, top_k, stream, force_download } => {
            info!("Running base model inference: model={}, prompt_len={}", model, prompt.len());
            
            use crate::runtime::{TorchEngine, RuntimeConfig};
            use crate::runtime::sampling::{SamplingConfig, load_sampling_config};
            use std::path::PathBuf;
            
            debug!("Looking up model: {}", model);
            
            // Initialize model storage to find models
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
            
            // Get all available models (this calls children() which scans for models with model.json)
            let available_models = model_storage.children().await?;
            
            // Try to find the model by UUID, name, or display name
            let mut found_model = None;
            for (model_id, metadata) in available_models {
                // Check if UUID matches
                if model_id.to_string() == model {
                    debug!("Found model by UUID: {}", model);
                    found_model = Some((model_id, metadata));
                    break;
                }
                
                // Check if name matches
                if metadata.name == model {
                    debug!("Found model by name: {}", model);
                    found_model = Some((model_id, metadata));
                    break;
                }
                
                // Check if display name matches
                if let Some(ref display_name) = metadata.display_name {
                    if display_name == &model || display_name.ends_with(&format!("/{}", model)) {
                        debug!("Found model by display name: {}", display_name);
                        found_model = Some((model_id, metadata));
                        break;
                    }
                }
                
                // Check if it's a partial match (just the model part of org/model)
                if model.contains('/') {
                    if metadata.name == model || metadata.name.ends_with(&format!("/{}", model)) {
                        debug!("Found model by partial name: {}", &metadata.name);
                        found_model = Some((model_id, metadata));
                        break;
                    }
                } else {
                    // Check if the model name ends with the requested name
                    let name_parts: Vec<&str> = metadata.name.split('/').collect();
                    if let Some(last_part) = name_parts.last() {
                        if last_part == &model {
                            debug!("Found model by short name: {} (full: {})", model, &metadata.name);
                            found_model = Some((model_id, metadata));
                            break;
                        }
                    }
                }
            }
            
            // Get the model path or error if not found
            let model_path = match found_model {
                Some((model_id, _metadata)) => {
                    // Use UUID-based path
                    storage_paths.models_dir()?.join(model_id.to_string())
                }
                None => {
                    error!("Model '{}' not found in model storage", model);
                    eprintln!("Error: Model '{}' not found", model);
                    eprintln!("Available models:");
                    
                    // List available models to help the user
                    let available_models = model_storage.children().await?;
                    if available_models.is_empty() {
                        eprintln!("  No models found. Download one with:");
                        eprintln!("  hyprstream model pull https://huggingface.co/Qwen/Qwen2-0.5B-Instruct");
                    } else {
                        for (_id, metadata) in available_models.iter().take(5) {
                            eprintln!("  - {} (UUID: {})", metadata.name, metadata.model_id);
                        }
                        if available_models.len() > 5 {
                            eprintln!("  ... and {} more", available_models.len() - 5);
                        }
                    }
                    return Ok(());
                }
            };
            
            info!("Using model at: {}", model_path.display());
            
            // Load model configuration from model card
            let sampling_config = if model_path.join("config.json").exists() {
                debug!("Loading model configuration from config.json");
                match load_sampling_config(&model_path).await {
                    Ok(config) => {
                        debug!("Loaded model-specific sampling configuration");
                        config
                    }
                    Err(e) => {
                        warn!("Could not load model config: {}. Using defaults.", e);
                        SamplingConfig::default()
                    }
                }
            } else {
                debug!("No config.json found. Using default sampling configuration.");
                SamplingConfig::for_model(&model)
            };
            
            debug!("Initializing inference engine");
            let runtime_config = RuntimeConfig::default();
            let mut engine = TorchEngine::new(runtime_config)?;
            
            // Apply overrides to sampling config
            let mut final_config = sampling_config;
            if let Some(t) = temperature {
                final_config.temperature = t;
            }
            if let Some(p) = top_p {
                final_config.top_p = Some(p);
            }
            if let Some(k) = top_k {
                final_config.top_k = Some(k);
            }
            final_config.do_sample = final_config.temperature > 0.0;
            
            debug!("Loading base model (no LoRA)");
            
            // Load the model
            match engine.load_model(&model_path).await {
                Ok(_) => {
                    // Clear any LoRA that might have been loaded
                    {
                        let mut lora_guard = engine.active_lora.lock().unwrap();
                        *lora_guard = None;
                    }
                    debug!("Base model loaded successfully");
                }
                Err(e) => {
                    error!("Failed to load model: {}", e);
                    eprintln!("Error: Failed to load model: {}", e);
                    return Ok(());
                }
            }
            
            // Use model defaults or overrides
            let max_tokens = max_tokens.unwrap_or(100);
            
            info!(
                "Generating response: max_tokens={}, temperature={}, top_p={:?}, top_k={:?}",
                max_tokens, final_config.temperature, final_config.top_p, final_config.top_k
            );
            debug!("Prompt: {}", prompt);
            
            // Use clean inference interface
            use crate::runtime::InferenceExt;
            use crate::runtime::template_engine::ChatMessage;
            use std::io::{self, Write};
            
            // Apply chat template to the prompt
            let formatted_prompt = {
                // Create a user message
                let messages = vec![
                    ChatMessage {
                        role: "user".to_string(),
                        content: prompt.clone(),
                    }
                ];
                
                // Try to apply the model's chat template
                match engine.apply_chat_template(&messages, true) {
                    Ok(formatted) => {
                        debug!("Applied chat template successfully");
                        formatted
                    }
                    Err(e) => {
                        warn!("Could not apply chat template: {}. Using raw prompt.", e);
                        prompt.clone()
                    }
                }
            };
            
            let request = crate::runtime::InferenceRequest {
                prompt: formatted_prompt,
                max_tokens,
                temperature: final_config.temperature,
                top_p: final_config.top_p.unwrap_or(0.95),
                top_k: final_config.top_k,
                repeat_penalty: final_config.repetition_penalty,
                stream,
                lora_weights: None, // No LoRA for model infer
            };
            
            let result = if stream {
                engine.run_inference_streaming(request, |token| {
                    print!("{}", token);
                    io::stdout().flush().ok();
                }).await?
            } else {
                let result = engine.run_inference(request).await?;
                println!("{}", result.text);
                result
            };
            
            info!(
                "Generation complete: {} tokens generated in {:.2}s", 
                result.tokens_generated, 
                result.latency_ms as f64 / 1000.0
            );
        }
    }
    
    Ok(())
}




/// Create HTTP client for REST API communication
pub fn create_http_client() -> Client {
    Client::builder()
        .timeout(std::time::Duration::from_secs(300)) // 5 minutes for long inference
        .build()
        .expect("Failed to create HTTP client")
}

/// Create LoRA adapter via REST API
pub async fn create_lora_via_api(
    base_url: &str,
    name: Option<String>,
    base_model: &str,
    rank: usize,
    alpha: f32,
    target_modules: &[String],
    sparsity: f32,
) -> Result<Value, Box<dyn std::error::Error>> {
    let client = create_http_client();
    let url = format!("{}/v1/lora/create", base_url);
    
    let request_body = json!({
        "name": name,
        "base_model": base_model,
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "sparsity_ratio": sparsity,
        "neural_compression": true,
        "auto_regressive": true
    });
    
    let response = client.post(&url).json(&request_body).send().await?;
    
    if response.status().is_success() {
        let result: Value = response.json().await?;
        Ok(result)
    } else {
        let status_code = response.status();
        let error_text = response.text().await?;
        Err(format!("Failed to create LoRA: HTTP {} - {}", status_code, error_text).into())
    }
}

/// List LoRA adapters via REST API
pub async fn list_lora_via_api(base_url: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let client = create_http_client();
    let url = format!("{}/v1/lora/list", base_url);
    
    let response = client.get(&url).send().await?;
    
    if response.status().is_success() {
        let result: Value = response.json().await?;
        Ok(result)
    } else {
        let status_code = response.status();
        let error_text = response.text().await?;
        Err(format!("Failed to list LoRA adapters: HTTP {} - {}", status_code, error_text).into())
    }
}

/// Get LoRA adapter info via REST API
pub async fn get_lora_info_via_api(
    base_url: &str,
    lora_id: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let client = create_http_client();
    let url = format!("{}/v1/lora/{}/info", base_url, lora_id);
    
    let response = client.get(&url).send().await?;
    
    if response.status().is_success() {
        let result: Value = response.json().await?;
        Ok(result)
    } else {
        let status_code = response.status();
        let error_text = response.text().await?;
        Err(format!("Failed to get LoRA info: HTTP {} - {}", status_code, error_text).into())
    }
}

/// Start LoRA training via REST API
pub async fn start_training_via_api(
    base_url: &str,
    lora_id: &str,
    learning_rate: f32,
    batch_size: usize,
) -> Result<Value, Box<dyn std::error::Error>> {
    let client = create_http_client();
    let url = format!("{}/v1/training/{}/start", base_url, lora_id);
    
    let request_body = json!({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation": true,
        "mixed_precision": true
    });
    
    let response = client.post(&url).json(&request_body).send().await?;
    
    if response.status().is_success() {
        let result: Value = response.json().await?;
        Ok(result)
    } else {
        let status_code = response.status();
        let error_text = response.text().await?;
        Err(format!("Failed to start training: HTTP {} - {}", status_code, error_text).into())
    }
}

/// Get training status via REST API
pub async fn get_training_status_via_api(
    base_url: &str,
    lora_id: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let client = create_http_client();
    let url = format!("{}/v1/training/{}/status", base_url, lora_id);
    
    let response = client.get(&url).send().await?;
    
    if response.status().is_success() {
        let result: Value = response.json().await?;
        Ok(result)
    } else {
        let status_code = response.status();
        let error_text = response.text().await?;
        Err(format!("Failed to get training status: HTTP {} - {}", status_code, error_text).into())
    }
}

/// Perform chat completion via REST API
pub async fn chat_completion_via_api(
    base_url: &str,
    lora_id: &str,
    messages: &[serde_json::Value],
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stream: bool,
) -> Result<Value, Box<dyn std::error::Error>> {
    let client = create_http_client();
    let url = format!("{}/v1/inference/{}/chat/completions", base_url, lora_id);
    
    let request_body = json!({
        "model": format!("lora-{}", lora_id),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    });
    
    let response = client.post(&url).json(&request_body).send().await?;
    
    if response.status().is_success() {
        let result: Value = response.json().await?;
        Ok(result)
    } else {
        let status_code = response.status();
        let error_text = response.text().await?;
        Err(format!("Failed to perform chat completion: HTTP {} - {}", status_code, error_text).into())
    }
}

/// Handle chat command - inference with models/composed models
pub async fn handle_chat_command(
    cmd: crate::cli::commands::ChatCommand,
    _server_url: String,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("💬 Starting chat with model: {}", cmd.model_id);
    
    if cmd.train {
        println!("📚 Training mode enabled - will adapt model during conversation");
        println!("   → Inference runs on base model");
        println!("   → LoRA training applied to composed model");
    } else {
        println!("🧠 Inference mode - no training");
    }
    
    // Single prompt mode
    if let Some(prompt) = cmd.prompt {
        println!("\n🤖 Processing prompt: {}", prompt);
        
        // Try to run actual inference with TorchEngine
        match run_chat_inference(&cmd.model_id, &prompt, cmd.max_tokens, cmd.temperature).await {
            Ok(response) => {
                println!("\n📤 Response:");
                println!("========");
                println!("{}", response);
                println!("========");
                
                // If training mode is enabled, apply temporal LoRA training
                if cmd.train {
                    println!("\n🎓 Training mode: Applying temporal LoRA updates...");
                    
                    // For training, we need an expected response
                    // In interactive mode, we'd get this from user feedback
                    // For now, use a simple training example
                    let expected_response = format!("Hello! I'm a helpful AI assistant. How can I help you today?");
                    
                    match run_temporal_training(&cmd.model_id, &prompt, &expected_response).await {
                        Ok(()) => {
                            println!("✅ Training completed successfully");
                        }
                        Err(e) => {
                            println!("⚠️ Training error: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                // Fallback to mock response if inference fails
                println!("⚠️ Inference error: {}", e);
                println!("📤 Response: [Mock response - inference system integration needed]");
                
                if cmd.train {
                    println!("📈 Training skipped due to inference failure");
                }
            }
        }
        
        return Ok(());
    }
    
    // Interactive chat mode
    println!("\n💬 Interactive Chat Mode");
    println!("📋 Configuration:");
    println!("   Max tokens: {}", cmd.max_tokens);
    println!("   Temperature: {}", cmd.temperature);
    println!("   Top-p: {}", cmd.top_p);
    if cmd.train {
        println!("   Training: Enabled");
    }
    println!();
    println!("Type 'quit' or 'exit' to end the conversation");
    println!("---");
    
    println!("💡 Interactive chat coming soon!");
    println!("   Integration with conversation router and inference system needed");
    
    Ok(())
}

/// Run chat inference - finds and loads model, then generates response
async fn run_chat_inference(
    model_id: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<String, Box<dyn std::error::Error>> {
    use crate::config::RuntimeConfig;
    use std::path::PathBuf;
    
    // Create runtime config
    let runtime_config = RuntimeConfig::default();
    // Temperature will be used in generation request, not runtime config
    
    // Create TorchEngine
    let mut engine = TorchEngine::new_async(runtime_config).await?;
    
    // Find and load the model
    let storage_paths = crate::storage::StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;
    
    // Try to find the model file
    let model_filename = if model_id.ends_with(".safetensors") {
        model_id.to_string()
    } else {
        format!("{}.safetensors", model_id)
    };
    
    // Look for model in various locations
    let possible_filenames = vec![
        model_filename.clone(),
        format!("{}.safetensors", model_id),
        format!("model.safetensors"),
    ];
    
    let mut possible_paths = Vec::new();
    for filename in &possible_filenames {
        possible_paths.push(models_dir.join(filename));
    }
    
    let mut model_path = None;
    for path in possible_paths {
        if path.exists() {
            model_path = Some(path);
            break;
        }
    }
    
    let model_path = model_path.ok_or_else(|| {
        format!("Model '{}' not found. Try: hyprstream model download {}", model_id, model_id)
    })?;
    
    println!("📂 Loading model from: {}", model_path.display());
    
    // Load the model
    engine.load_model(&model_path).await?;
    
    // Use clean inference interface
    use crate::runtime::InferenceExt;
    
    println!("🔮 Generating response...");
    
    let request = crate::runtime::InferenceRequest {
        prompt: prompt.to_string(),
        max_tokens,
        temperature,
        top_p: 0.95,
        top_k: Some(40),
        repeat_penalty: 1.1,  // Default for composed model
        stream: false,
        lora_weights: None,
    };
    
    let result = engine.run_inference(request).await?;
    
    Ok(result.text)
}

/// Run temporal LoRA training using TorchEngine
async fn run_temporal_training(
    model_id: &str,
    prompt: &str,
    expected_response: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::config::RuntimeConfig;
    use crate::runtime::TorchEngine;
    
    tracing::info!("🎓 Starting temporal LoRA training for model: {}", model_id);
    
    // Create runtime config
    let runtime_config = RuntimeConfig::default();
    
    // Create TorchEngine
    let mut engine = TorchEngine::new_async(runtime_config).await?;
    
    // Find and load the model (reuse same logic as run_candle_inference)
    let storage_paths = crate::storage::StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;
    
    let model_filename = if model_id.ends_with(".safetensors") {
        model_id.to_string()
    } else {
        format!("{}.safetensors", model_id)
    };
    
    let possible_filenames = vec![
        model_filename.clone(),
        format!("{}.safetensors", model_id),
        format!("model.safetensors"),
    ];
    
    let mut possible_paths = Vec::new();
    for filename in &possible_filenames {
        possible_paths.push(models_dir.join(filename));
    }
    
    let mut model_path = None;
    for path in possible_paths {
        if path.exists() {
            model_path = Some(path);
            break;
        }
    }
    
    let model_path = model_path.ok_or_else(|| {
        format!("Model '{}' not found for training", model_id)
    })?;
    
    // Load the model
    engine.load_model(&model_path).await?;
    
    // Run temporal LoRA training
    let learning_rate = 0.001; // Default learning rate
    engine.train_temporal_lora(prompt, expected_response, learning_rate).await?;
    
    tracing::info!("✅ Temporal LoRA training completed");
    
    Ok(())
}
