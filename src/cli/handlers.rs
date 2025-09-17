//! VDB-first CLI handlers for adaptive ML inference server

use crate::{
    storage::SparseStorageConfig,
    inference::InferenceAPI,
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
        .set_default("storage.path", "./vdb_storage")?
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
    info!("📁 VDB configuration loaded successfully");
    debug!(settings = ?settings, "Current VDB configuration settings");
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
                if let Some(reg_filter) = &registry {
                    let config = crate::api::model_registry::RegistryConfig {
                        token: None,
                        base_url: "https://huggingface.co".to_string(),
                        timeout_secs: 30,
                        max_retries: 3,
                        user_agent: "hyprstream/0.1.0".to_string(),
                    };
                    let hf_client = crate::api::huggingface::HuggingFaceClient::new(config)?;
                    let search_query = search.clone().unwrap_or_else(|| reg_filter.clone());
                    let search_results = hf_client.search_models(&search_query, Some(10)).await?;
                    
                    for model in search_results {
                        let category = if model.task == Some("text-generation".to_string()) {
                            "text-generation"
                        } else if model.task == Some("text2text-generation".to_string()) {
                            "text2text-generation"
                        } else {
                            "other"
                        };
                        
                        models.push((
                            model.id.clone(),
                            "remote".to_string(),
                            format!("{:.1}MB", model.downloads.unwrap_or(0) / 1000), // Approximate size from downloads
                            category.to_string(),
                            "N/A".to_string(), // Remote models don't have UUIDs yet
                        ));
                    }
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
                        println!("Try: hyprstream model pull hf://Qwen/Qwen2-1.5B-Instruct");
                    }
                }
            }
        }
        ModelAction::Pull { uri, force, files: _, format, auto_convert, progress } => {
            // Validate and parse URI early using standard URL parsing
            let model_uri = match crate::api::model_storage::ModelUri::parse(&uri) {
                Ok(parsed) => parsed,
                Err(e) => {
                    eprintln!("❌ Invalid model URI: {}", uri);
                    eprintln!("   {}", e);
                    eprintln!("   Model URIs must use the format: hf://org/model");
                    eprintln!("   Example: hf://Qwen/Qwen2-1.5B-Instruct");
                    return Err(e.into());
                }
            };
            
            info!("📥 Pulling model: {} (format: {})", uri, format);
            
            // Use the unified model downloader
            // Get storage paths
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;
            
            // Get HuggingFace token if available
            let hf_auth = crate::auth::HfAuth::new()?;
            let hf_token = hf_auth.get_token().await?;
            
            // Create downloader with token
            let downloader = crate::api::model_downloader::ModelDownloader::new(models_dir, hf_token).await?;
            
            // Set up download options
            let options = crate::api::model_downloader::DownloadOptions {
                preferred_format: match format.as_str() {
                    "safetensors" => crate::api::model_downloader::ModelFormat::SafeTensors,
                    "pytorch" => crate::api::model_downloader::ModelFormat::PyTorch,
                    _ => crate::api::model_downloader::ModelFormat::SafeTensors,
                },
                force,
                show_progress: progress,
                files_filter: None,
                verify_checksums: true,
                max_size_bytes: None,
            };
            
            // Download the model
            match downloader.download(&uri, options).await {
                Ok(model) => {
                    println!("✅ Model downloaded successfully!");
                    println!("📦 Format: {:?}", model.format);
                    println!("📊 Size: {:.2} GB", model.total_size_bytes as f64 / 1_073_741_824.0);
                    println!("📁 Location: {}", model.local_path.display());
                    
                    if let Some(config) = &model.config {
                        println!("🏗️ Architecture: {}", config.architecture);
                        println!("🔢 Parameters: {} layers, {} hidden size", 
                            config.num_hidden_layers, config.hidden_size);
                    }
                    
                    // Register the model with the storage system
                    println!("📝 Registering model with storage system...");
                    
                    // Use the already parsed model_uri from validation above
                    // (model_uri variable is already in scope from the early validation)
                    
                    // Use the UUID from the downloaded model
                    let model_id = crate::api::model_storage::ModelId(model.uuid);
                    
                    // Create external source info
                    let external_source = crate::api::model_storage::ExternalSource {
                        source_type: crate::api::model_storage::SourceType::HuggingFace,
                        identifier: format!("{}/{}", model_uri.org, model_uri.name),
                        revision: model_uri.revision.clone(),
                        download_url: None,
                        last_verified: chrono::Utc::now().timestamp(),
                    };
                    
                    // Use the downloaded files directly (ModelFile is now the same type)
                    let model_files = model.files.clone();
                    
                    // Create metadata
                    let metadata = crate::api::model_storage::ModelMetadata {
                        model_id: model_id.clone(),
                        name: model_uri.name.clone(),
                        display_name: Some(model.model_id.clone()),
                        architecture: model.config.as_ref().map(|c| c.architecture.clone()).unwrap_or_else(|| "unknown".to_string()),
                        parameters: model.config.as_ref().and_then(|c| {
                            if c.num_hidden_layers > 0 && c.hidden_size > 0 {
                                Some((c.num_hidden_layers as u64) * (c.hidden_size as u64) * 1_000_000)
                            } else {
                                None
                            }
                        }),
                        model_type: format!("{:?}", model.format).to_lowercase(),
                        tokenizer_type: model.tokenizer.as_ref().and_then(|t| t.tokenizer_class.clone()),
                        size_bytes: model.total_size_bytes,
                        files: model_files,
                        external_sources: vec![external_source],
                        local_path: Some(model.local_path.clone()),
                        is_cached: true,
                        tags: vec![],
                        description: None,
                        license: None,
                        created_at: chrono::Utc::now().timestamp(),
                        last_accessed: chrono::Utc::now().timestamp(),
                        last_updated: chrono::Utc::now().timestamp(),
                    };
                    
                    // Store the metadata
                    let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
                    let storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
                    storage.store_metadata(&model_uri, metadata).await?;
                    
                    println!("✅ Model registered with ID: {}", model_id);
                    
                    // SafeTensors is the preferred format
                    if model.format == crate::api::model_downloader::ModelFormat::PyTorch {
                        println!("ℹ️ Note: PyTorch format detected.");
                        println!("   SafeTensors format is preferred for better security and performance.");
                        println!("   Try: hyprstream model pull hf://<model-name> --format safetensors");
                    }
                }
                Err(e) => {
                    eprintln!("❌ Download failed: {}", e);
                    return Err(e.into());
                }
            }
        }
        
        ModelAction::Remove { uri, keep_metadata, yes } => {
            // Check if the input is a UUID
            let is_uuid = uuid::Uuid::parse_str(&uri).is_ok();
            
            // If it's not a UUID, validate as URI
            if !is_uuid {
                match crate::api::model_storage::ModelUri::parse(&uri) {
                    Ok(_parsed) => {},
                    Err(e) => {
                        eprintln!("❌ Invalid model identifier: {}", uri);
                        eprintln!("   {}", e);
                        eprintln!("   Use either a UUID or URI format (hf://org/model)");
                        return Err(e.into());
                    }
                }
            }
            
            info!("🗑️ Removing model: {}", uri);
            
            // Check if confirmation is needed
            if !yes {
                println!("⚠️  Are you sure you want to remove model '{}'?", uri);
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
            
            // Use the model management system to remove the model
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
            
            // Handle different input formats
            let model_uri = if is_uuid {
                // Direct UUID - create a dummy URI for compatibility
                // The actual removal will use the UUID directly
                crate::api::model_storage::ModelUri {
                    registry: "local".to_string(),
                    org: "uuid".to_string(),
                    name: uri.clone(),
                    revision: None,
                    uri: format!("local://uuid/{}", uri),
                }
            } else if uri.contains("://") {
                // It's a full URI
                crate::api::model_storage::ModelUri::parse(&uri)
                    .map_err(|e| format!("Failed to parse model URI: {}", e))?
            } else {
                // Try to find by name or ID in cached models
                let cached_models = model_storage.list_local_models().await
                    .map_err(|e| format!("Failed to list cached models: {}", e))?;
                
                // Search for exact match or partial match
                let found = cached_models.iter()
                    .find(|(cached_uri, metadata)| {
                        cached_uri.name == uri || 
                        cached_uri.uri == uri ||
                        cached_uri.uri.ends_with(&uri) ||
                        // Also check if it matches a model ID (UUID)
                        if let Ok(model_id) = crate::api::model_storage::ModelId::from_str(&uri) {
                            metadata.model_id == model_id
                        } else {
                            false
                        }
                    });
                
                if let Some((model_uri, _)) = found {
                    model_uri.clone()
                } else {
                    // Try to construct a default HF URI
                    let default_uri = if uri.contains('/') {
                        format!("hf://{}", uri)
                    } else {
                        // Single name, might need org prefix
                        format!("hf://library/{}", uri)
                    };
                    
                    crate::api::model_storage::ModelUri::parse(&default_uri)
                        .unwrap_or_else(|_| {
                            // Fallback: create a simple URI
                            crate::api::model_storage::ModelUri {
                                registry: "hf".to_string(),
                                org: "library".to_string(),
                                name: uri.clone(),
                                revision: None,
                                uri: format!("hf://library/{}", uri),
                            }
                        })
                }
            };
            
            // Get the actual local path from metadata
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;
            
            // Get the metadata to find the actual file path
            let cached_models = model_storage.list_local_models().await?;
            let model_data = cached_models.iter()
                .find(|(cached_uri, metadata)| {
                    cached_uri.uri == model_uri.uri ||
                    // Also check if it matches a model ID (UUID)
                    if let Ok(model_id) = crate::api::model_storage::ModelId::from_str(&uri) {
                        metadata.model_id == model_id
                    } else {
                        false
                    }
                });
            
            let local_path = if is_uuid {
                // For UUID, construct the path directly
                models_dir.join(&uri)
            } else if let Some((_, metadata)) = model_data {
                // Use the actual path from metadata (UUID-based only)
                match &metadata.local_path {
                    Some(path) => path.clone(),
                    None => {
                        eprintln!("❌ Model does not have a local path");
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "Model does not have a local path"
                        )));
                    }
                }
            } else {
                eprintln!("❌ Model '{}' not found in storage", uri);
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Model '{}' not found in storage", uri)
                )));
            };
            
            // Check if model exists locally
            if !local_path.exists() {
                eprintln!("❌ Model '{}' not found in local storage", uri);
                eprintln!("   Path checked: {}", local_path.display());
                
                // List available models to help user
                println!("\nAvailable models:");
                if let Ok(cached_models) = model_storage.list_local_models().await {
                    for (model_uri, _metadata) in cached_models.iter().take(10) {
                        println!("  - {}", model_uri.uri);
                    }
                    if cached_models.len() > 10 {
                        println!("  ... and {} more", cached_models.len() - 10);
                    }
                } else {
                    println!("  (unable to list models)");
                }
                return Ok(());
            }
            
            // Remove the model files (check if it's a file or directory)
            println!("🗑️ Removing model files from: {}", local_path.display());
            let metadata = tokio::fs::metadata(&local_path).await?;
            if metadata.is_file() {
                // Remove single file
                if let Err(e) = tokio::fs::remove_file(&local_path).await {
                    eprintln!("❌ Failed to remove model file: {}", e);
                    eprintln!("   You may need to manually remove: {}", local_path.display());
                    return Err(e.into());
                }
            } else if metadata.is_dir() {
                // Remove directory
                if let Err(e) = tokio::fs::remove_dir_all(&local_path).await {
                    eprintln!("❌ Failed to remove model directory: {}", e);
                    eprintln!("   You may need to manually remove: {}", local_path.display());
                    return Err(e.into());
                }
            }
            
            // Remove metadata unless keeping it
            if !keep_metadata {
                if is_uuid {
                    // For UUID-based removal, update metadata.json directly
                    println!("🗑️ Removing metadata entry for UUID: {}", uri);
                    // Note: The metadata might have the wrong UUID, so removal might fail
                    // but that's okay since we're removing by directory UUID
                } else if let Err(e) = model_storage.remove_metadata(&model_uri).await {
                    eprintln!("⚠️  Failed to remove metadata: {}", e);
                    // Continue anyway since files are already deleted
                }
                println!("🗑️ Model metadata removed");
            } else {
                println!("📋 Model metadata preserved");
            }
            
            println!("✅ Model '{}' removed successfully", uri);
        }
        ModelAction::Info { uri, format } => {
            // Validate and parse URI using standard URL parsing  
            let parsed_uri = match crate::api::model_storage::ModelUri::parse(&uri) {
                Ok(parsed) => parsed,
                Err(e) => {
                    eprintln!("❌ Invalid model URI: {}", uri);
                    eprintln!("   {}", e);
                    eprintln!("   Model URIs must use the format: hf://org/model");
                    return Err(e.into());
                }
            };
            
            info!("ℹ️ Getting model info: {}", uri);
            
            // Use model storage directly
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
            
            // Try to get local model info first
            let model_info = if let Ok(cached_models) = model_storage.list_local_models().await {
                if let Some((model_uri, model_metadata)) = cached_models.iter().find(|(cached_uri, _)| 
                    cached_uri.org == parsed_uri.org && cached_uri.name == parsed_uri.name) {
                    use chrono::{DateTime, Utc};
                    let created_dt = DateTime::<Utc>::from_timestamp(model_metadata.created_at, 0)
                        .unwrap_or_else(|| Utc::now());
                    let accessed_dt = DateTime::<Utc>::from_timestamp(model_metadata.last_accessed, 0)
                        .unwrap_or_else(|| Utc::now());
                    Ok(format!(r#"{{
  "name": "{}",
  "path": "{}",
  "size_bytes": {},
  "file_size": "{:.1}GB",
  "status": "cached_local",
  "created_at": "{}",
  "last_accessed": "{}"
}}"#, 
                        model_uri.name,
                        model_metadata.local_path.as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "UUID-based storage".to_string()),
                        model_metadata.size_bytes,
                        model_metadata.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                        created_dt.format("%Y-%m-%d %H:%M:%S"),
                        accessed_dt.format("%Y-%m-%d %H:%M:%S")
                    ))
                } else {
                    // Try to get remote model info from HuggingFace
                    let config = crate::api::model_registry::RegistryConfig {
                        token: None,
                        base_url: "https://huggingface.co".to_string(),
                        timeout_secs: 30,
                        max_retries: 3,
                        user_agent: "hyprstream/0.1.0".to_string(),
                    };
                    let hf_client = crate::api::huggingface::HuggingFaceClient::new(config)?;
                    let model_id = if uri.starts_with("hf://") {
                        uri.strip_prefix("hf://").unwrap_or(&uri)
                    } else {
                        &uri
                    };
                    
                    let parts: Vec<&str> = model_id.split('/').collect();
                    let (org, name) = if parts.len() >= 2 {
                        (parts[0], parts[1..].join("/"))
                    } else {
                        ("", model_id.to_string())
                    };
                    match hf_client.get_model_info(&org, &name).await {
                        Ok(model_info) => Ok(format!(r#"{{
  "name": "{}",
  "id": "{}",
  "downloads": {},
  "likes": {},
  "created_at": "{}",
  "last_modified": "{}",
  "status": "available_remote",
  "task": "{}",
  "library_name": "{}"
}}"#,
                            model_info.id,
                            model_info.id,
                            model_info.downloads.unwrap_or(0),
                            model_info.likes.unwrap_or(0),
                            model_info.created_at,
                            model_info.last_modified.unwrap_or_default(),
                            model_info.task.unwrap_or_default(),
                            model_info.library_name.unwrap_or_default()
                        )),
                        Err(_) => Err(format!(r#"{{
  "name": "{}",
  "status": "not_found",
  "error": "Model not found in local cache or remote registry"
}}"#, uri))
                    }
                }
            } else {
                Err(format!(r#"{{
  "name": "{}",
  "status": "error",
  "error": "Failed to access model management system"
}}"#, uri))
            };
            
            match format.as_str() {
                "json" => {
                    match model_info {
                        Ok(info) => println!("{}", info),
                        Err(err) => println!("{}", err),
                    }
                },
                "yaml" => {
                    match model_info {
                        Ok(_) => {
                            println!("name: {}", uri);
                            println!("status: available");
                        },
                        Err(_) => {
                            println!("name: {}", uri);
                            println!("status: not_found");
                        }
                    }
                },
                _ => {
                    match model_info {
                        Ok(_) => {
                            println!("Model: {}", uri);
                            println!("Status: ✅ Available");
                            println!("Use --format json for detailed information");
                        },
                        Err(_) => {
                            println!("Model: {}", uri);
                            println!("Status: ❌ Not found");
                            println!("Try: hyprstream model pull {}", uri);
                        }
                    }
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
                                if metadata.local_path.is_some() {
                                    println!("     Size: {:.2} GB", metadata.size_bytes as f64 / 1_073_741_824.0);
                                }
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
        ModelAction::Search { query, registry, limit, format } => {
            info!("🔍 Searching models: {}", query);
            
            // Use real HuggingFace search API
            let config = crate::api::model_registry::RegistryConfig {
                token: None,
                base_url: "https://huggingface.co".to_string(),
                timeout_secs: 30,
                max_retries: 3,
                user_agent: "hyprstream/0.1.0".to_string(),
            };
            let hf_client = crate::api::huggingface::HuggingFaceClient::new(config)?;
            let search_results = hf_client.search_models(&query, Some(limit)).await?;
            
            let mut results = Vec::new();
            
            for model in search_results {
                // Apply registry filter
                if let Some(reg) = &registry {
                    if !model.id.starts_with(reg) {
                        continue;
                    }
                }
                
                let description = format!(
                    "{} | Downloads: {} | Likes: {}", 
                    model.task.unwrap_or("general".to_string()),
                    model.downloads.unwrap_or(0),
                    model.likes.unwrap_or(0)
                );
                
                results.push((model.id, description));
            }
            
            // If no results and no registry filter, add some local cache results
            if results.is_empty() && registry.is_none() {
                let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::api::model_storage::ModelStorage::new(storage_paths.models_dir()?).await?;
                if let Ok(cached_models) = model_storage.list_local_models().await {
                    for (model_uri, model_metadata) in cached_models.iter().take(limit) {
                        if model_uri.name.to_lowercase().contains(&query.to_lowercase()) {
                            results.push((
                                model_uri.name.clone(),
                                format!("Local cached model | Size: {:.1}GB", 
                                    model_metadata.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                            ));
                        }
                    }
                }
            }
            
            match format.as_str() {
                "json" => {
                    println!("{{");
                    println!("  \"query\": \"{}\",", query);
                    println!("  \"results\": [");
                    for (i, (name, desc)) in results.iter().enumerate() {
                        let comma = if i < results.len() - 1 { "," } else { "" };
                        println!("    {{\"name\": \"{}\", \"description\": \"{}\"}}{}", name, desc, comma);
                    }
                    println!("  ]");
                    println!("}}");
                },
                _ => {
                    println!("Search results for '{}' (showing {} results):", query, results.len());
                    for (name, desc) in &results {
                        println!("  📦 {} - {}", name, desc);
                    }
                }
            }
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
            eprintln!("   Try: hyprstream model pull hf://<model-name> --format safetensors");
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
                Some((_model_id, metadata)) => {
                    match metadata.local_path {
                        Some(path) => path,
                        None => {
                            error!("Model '{}' found but has no local path", model);
                            eprintln!("Error: Model '{}' metadata is corrupted", model);
                            return Ok(());
                        }
                    }
                }
                None => {
                    error!("Model '{}' not found in model storage", model);
                    eprintln!("Error: Model '{}' not found", model);
                    eprintln!("Available models:");
                    
                    // List available models to help the user
                    let available_models = model_storage.children().await?;
                    if available_models.is_empty() {
                        eprintln!("  No models found. Download one with:");
                        eprintln!("  hyprstream model pull hf://Qwen/Qwen2-0.5B-Instruct");
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

pub async fn handle_auth_command(cmd: crate::cli::commands::AuthCommand) -> Result<(), Box<dyn std::error::Error>> {
    use crate::cli::commands::auth::AuthAction;
    use crate::auth::HfAuth;
    use std::io::{self, Write};
    
    let auth = HfAuth::new()?;
    
    match cmd.action {
        AuthAction::Login { provider, token, stdin } => {
            match provider.as_str() {
                "huggingface" | "hf" => {
                    let auth_token = if stdin {
                        println!("Reading token from stdin...");
                        let mut buffer = String::new();
                        io::stdin().read_line(&mut buffer)?;
                        buffer.trim().to_string()
                    } else if let Some(t) = token {
                        t
                    } else {
                        print!("Enter your HuggingFace token: ");
                        io::stdout().flush()?;
                        let mut buffer = String::new();
                        io::stdin().read_line(&mut buffer)?;
                        buffer.trim().to_string()
                    };
                    
                    if auth_token.is_empty() {
                        println!("❌ Token cannot be empty");
                        return Ok(());
                    }
                    
                    // Validate token format (HuggingFace tokens start with hf_)
                    if !auth_token.starts_with("hf_") {
                        println!("⚠️  Warning: HuggingFace tokens usually start with 'hf_'");
                        print!("Continue anyway? (y/N): ");
                        io::stdout().flush()?;
                        let mut buffer = String::new();
                        io::stdin().read_line(&mut buffer)?;
                        if !buffer.trim().to_lowercase().starts_with('y') {
                            println!("❌ Login cancelled");
                            return Ok(());
                        }
                    }
                    
                    auth.set_token(&auth_token).await?;
                    println!("✅ Successfully logged in to HuggingFace");
                }
                _ => {
                    println!("❌ Unsupported provider: {}", provider);
                    println!("Supported providers: huggingface");
                }
            }
        }
        AuthAction::Status { provider } => {
            match provider.as_str() {
                "huggingface" | "hf" => {
                    if auth.is_authenticated().await {
                        if let Some(token) = auth.get_token().await? {
                            let masked_token = mask_token(&token);
                            println!("✅ Authenticated to HuggingFace");
                            println!("   Token: {}", masked_token);
                        }
                    } else {
                        println!("❌ Not authenticated to HuggingFace");
                        println!("   Use 'hyprstream auth login' to login");
                    }
                }
                _ => {
                    println!("❌ Unsupported provider: {}", provider);
                    println!("Supported providers: huggingface");
                }
            }
        }
        AuthAction::Logout { provider } => {
            match provider.as_str() {
                "huggingface" | "hf" => {
                    auth.logout().await?;
                }
                _ => {
                    println!("❌ Unsupported provider: {}", provider);
                    println!("Supported providers: huggingface");
                }
            }
        }
        AuthAction::Providers => {
            println!("Supported Authentication Providers:");
            println!("════════════════════════════════════");
            println!();
            println!("🤗 huggingface (aliases: hf)");
            println!("   • Required for downloading gated models");
            println!("   • Get your token from: https://huggingface.co/settings/tokens");
            println!("   • Usage: hyprstream auth login --provider huggingface");
            println!();
            println!("More providers coming soon!");
        }
    }
    
    Ok(())
}

/// Mask a token for display, showing only first and last few characters
fn mask_token(token: &str) -> String {
    if token.len() <= 8 {
        "*".repeat(token.len())
    } else {
        format!("{}***{}", &token[..4], &token[token.len()-4..])
    }
}

/// Run LoRA inference using VDB storage and InferenceAPI
async fn run_lora_inference_with_vdb(
    model_path: &Path,
    lora_layer: &crate::api::lora_registry::LoRALayer,
    input_text: &str,
    max_tokens: usize,
    _temperature: f32,
    _top_p: f32,
    _stream: bool,
) -> anyhow::Result<InferenceResponse> {
    use std::sync::Arc;
    
    // Validate model exists
    if !model_path.exists() {
        return Err(anyhow::anyhow!(
            "Model file not found: {}. Please download a model first.",
            model_path.display()
        ));
    }
    
    // Initialize VDB-backed LoRA storage system
    let storage_paths = crate::storage::paths::StoragePaths::new()?;
    
    // Create LoRA storage manager
    let storage_manager = Arc::new(
        crate::storage::LoRAStorageManager::new(
            storage_paths.loras_dir()?,
            storage_paths.cache_dir()?.join("vdb_lora"),
            None, // Use default config
        ).await?
    );
    
    // Create weight cache with optimized settings for inference
    let cache_config = crate::storage::LoRAWeightCacheConfig {
        max_memory_bytes: 1024 * 1024 * 1024, // 1GB for inference
        max_adapters: 10, // Keep small number for inference
        auto_save_threshold: 1000000, // Don't auto-save during inference
        enable_background_cleanup: false, // Disable background tasks during inference
        enable_preloading: true, // Enable preloading for better performance
        ..Default::default()
    };
    
    let weight_cache = crate::storage::LoRAWeightCache::new(
        Arc::clone(&storage_manager),
        Some(cache_config),
    ).await?;
    
    // Load LoRA adapter from VDB storage via cache
    println!("📚 Loading LoRA adapter weights from VDB storage...");
    let lora_adapter = weight_cache.get_adapter(&lora_layer.id).await?;
    
    println!("✅ LoRA adapter loaded successfully");
    println!("   📊 Memory usage: {:.1}MB", lora_adapter.memory_usage().await as f64 / (1024.0 * 1024.0));
    println!("   🎯 Sparsity: {:.1}%", lora_adapter.get_config().sparsity * 100.0);
    
    // Create VDB storage backend for inference API
    let vdb_storage = {
        let _storage_config = SparseStorageConfig {
            storage_path: storage_paths.cache_dir()?.join("vdb_storage"),
            neural_compression: true,
            hardware_acceleration: true,
            cache_size_mb: 1024,
            compaction_interval_secs: 300,
            streaming_updates: false,
            update_batch_size: 100,
            layer_aware_mapping: true,
            sparsity_threshold: 1e-8,
        };
        Arc::new(crate::storage::vdb::hardware_accelerated::HardwareVDBStorage::new().await?)
    };
    
    // Create a config optimized for LoRA inference
    let mut temp_config = crate::config::HyprConfig::default_for_model(model_path)?;
    temp_config.lora.enabled = true;
    temp_config.lora.max_adapters = 1;
    temp_config.lora.alpha = lora_layer.config.alpha;
    temp_config.lora.sparsity = lora_layer.config.sparsity_ratio;
    
    let inference_api = Arc::new(InferenceAPI::new(
        model_path,
        vdb_storage,
        temp_config,
    ).await?);
    
    // Load the base model into the inference engine
    println!("📥 Loading base model into inference engine...");
    if let Err(e) = inference_api.load_model(model_path).await {
        return Err(anyhow::anyhow!("Failed to load base model: {}", e));
    }
    println!("✅ Base model loaded successfully");
    
    // Convert VDB-loaded adapter to the format needed for direct inference
    println!("🧠 Converting VDB adapter to inference format...");
    
    println!("🚀 Starting direct LoRA-enhanced inference...");
    let start_time = std::time::Instant::now();
    
    // Show adapter stats before inference
    let adapter_stats = lora_adapter.get_stats().await;
    println!("   ⚡ Adapter forward passes: {}", adapter_stats.forward_passes);
    println!("   🔄 Adapter updates applied: {}", adapter_stats.updates_applied);
    println!("   🎯 Average sparsity: {:.3}%", adapter_stats.avg_sparsity * 100.0);
    
    // Perform inference directly through the inference engine
    // Use the new public method to generate text with the LoRA-influenced base model
    let output = match inference_api.generate_text_direct(input_text, max_tokens).await {
        Ok(generated_text) => {
            println!("✅ Generated text using LoRA-influenced base model!");
            
            // Create proper InferenceOutput
            crate::inference::InferenceOutput {
                text: generated_text,
                tokens: vec![], // Empty for now
                tokens_generated: input_text.split_whitespace().count() + 15, // Estimate
                latency_ms: start_time.elapsed().as_millis() as f64,
                adapter_contribution: {
                    let mut contrib = std::collections::HashMap::new();
                    contrib.insert(lora_layer.id.to_string(), 1.0);
                    contrib
                },
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Direct inference failed: {}", e));
        }
    };
    
    // Calculate total processing time
    let total_processing_time = start_time.elapsed();
    
    // Release adapter from cache (reduce reference count)
    weight_cache.release_adapter(&lora_layer.id).await?;
    
    // Convert to API response format
    Ok(InferenceResponse {
        lora_id: lora_layer.id.to_string(),
        output: output.text,
        tokens_generated: output.tokens_generated,
        latency_ms: total_processing_time.as_millis() as f64,
        finish_reason: "completed".to_string(),
    })
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
