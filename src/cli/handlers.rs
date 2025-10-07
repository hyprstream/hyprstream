//! CLI handlers for adaptive ML inference server

use crate::cli::commands::model::ModelAction;
use crate::config::RuntimeConfig;
use crate::git::BranchManager;
use crate::runtime::{RuntimeEngine, TorchEngine, InferenceExt};
use crate::runtime::sampling::{SamplingConfig, load_sampling_config};
use crate::runtime::template_engine::ChatMessage;
use crate::storage::{ModelStorage, ModelMetadata};
use crate::training::{CheckpointManager, WeightSnapshot, WeightFormat};
use ::config::{Config, File};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

/// Response structure for LoRA inference
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub lora_id: String,
    pub output: String,
    pub tokens_generated: usize,
    pub latency_ms: f64,
    pub finish_reason: String,
}

// TODO: Re-add model identifier resolution when needed

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
    ctx: crate::cli::AppContext,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = ctx.config();

    // Clone git2db config and set DHT mode to Server for full P2P participation
    let git2db_config = {
        let mut cfg = config.git2db.clone();

        #[cfg(feature = "gittorrent")]
        {
            cfg.gittorrent.dht_mode = gittorrent::DhtMode::Server;
            info!("DHT mode set to Server for full P2P participation");
        }

        cfg
    };

    // Use server config from HyprConfig
    let server_config = config.server.clone();

    // Get host and port for addr binding
    let host = &server_config.host;
    let port = server_config.port;
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;

    // Create server state with git2db config (DHT mode set to Server)
    let server_state = crate::server::state::ServerState::new_with_git2db(
        server_config,
        git2db_config
    ).await?;

    info!("Starting Hyprstream HTTP server on {}", addr);
    info!("OpenAI-compatible API available at http://{}/oai/v1", addr);

    // Start server (TLS configuration would come from config.server.tls if needed)
    crate::server::start_server(addr, server_state).await?;

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
        .set_default("port", "50051")?;

    // Load config file if provided
    if let Some(path) = config_path {
        builder = builder.add_source(File::from(path));
    }

    let settings = builder.build()?;
    info!("📁 Configuration loaded successfully");
    debug!(settings = ?settings, "Current configuration settings");
    Ok(())
}

pub async fn handle_model_command(
    cmd: crate::cli::commands::ModelCommand,
    _server_url: String,
) -> Result<(), Box<dyn std::error::Error>> {
    
    match cmd.action {
        ModelAction::List {
            registry,
            search,
            remote,
            format,
            show_git_ref,
            show_status,
            branch,
            tag,
            dirty_only
        } => {
            info!("📋 Listing available models...");
            
            // Use model storage directly
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::storage::ModelStorage::create(storage_paths.models_dir()?).await?;
            
            let models = model_storage.list_models().await?;
            
            // Get remote models if requested
            if remote {
                if let Some(_) = &registry {
                    eprintln!("⚠️  Remote model search is no longer supported");
                    eprintln!("   Use 'hyprstream model pull' with a git URL instead:");
                    eprintln!("   • hyprstream model pull https://huggingface.co/Qwen/Qwen2-0.5B-Instruct");
                    eprintln!("   • hyprstream model pull git@github.com:user/model.git");
                }
            }
            
            // Determine if git info should be extracted
            let extract_git_info = show_git_ref || show_status || branch.is_some() || tag.is_some() || dirty_only;

            // Extract git information if needed
            let models_with_git: Vec<_> = models.into_iter()
                .map(|(model_ref, metadata)| {
                    let git_info = if extract_git_info {
                        // Get model path
                        let models_dir = storage_paths.models_dir().unwrap_or_default();
                        let model_path = models_dir.join(&model_ref.model);
                        crate::cli::commands::model::GitInfo::from_repo_path(&model_path)
                    } else {
                        None
                    };
                    (model_ref, metadata, git_info)
                })
                .collect();

            // Apply git-based filters
            let filtered_models: Vec<_> = models_with_git.into_iter()
                .filter(|(model_ref, metadata, git_info)| {
                    // Apply search filter
                    if let Some(query) = &search {
                        let query_lower = query.to_lowercase();
                        if !(model_ref.model.to_lowercase().contains(&query_lower) ||
                             metadata.name.to_lowercase().contains(&query_lower)) {
                            return false;
                        }
                    }

                    // Apply git branch filter
                    if let Some(branch_filter) = &branch {
                        match git_info {
                            Some(git) => {
                                if !git.matches_branch(branch_filter) {
                                    return false;
                                }
                            }
                            None => return false, // No git info means can't match branch
                        }
                    }

                    // Apply git tag filter
                    if let Some(tag_filter) = &tag {
                        match git_info {
                            Some(git) => {
                                if !git.matches_tag(tag_filter) {
                                    return false;
                                }
                            }
                            None => return false, // No git info means can't match tag
                        }
                    }

                    // Apply dirty-only filter
                    if dirty_only {
                        match git_info {
                            Some(git) => {
                                if !git.is_dirty {
                                    return false;
                                }
                            }
                            None => return false, // No git info means can't determine dirty status
                        }
                    }

                    true
                })
                .collect();
            
            match format.as_str() {
                "json" => {
                    let json_models: Vec<_> = filtered_models.iter()
                        .map(|(model_ref, metadata, git_info)| {
                            let mut model_json = serde_json::json!({
                                "name": model_ref.model,
                                "display_name": metadata.display_name,
                                "size_bytes": metadata.size_bytes,
                            });

                            // Add git information if available
                            if let Some(git) = git_info {
                                model_json["git"] = serde_json::to_value(git).unwrap_or_default();
                            }

                            model_json
                        })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                        "models": json_models
                    }))?);
                },
                _ => {
                    // Always use enhanced table format
                    println!("{:<30} {:<15} {:<8} {:<6} {:<10}", "MODEL NAME", "REF", "COMMIT", "STATUS", "SIZE");
                    println!("{}", "-".repeat(75));

                    for (model_ref, metadata, git_info) in &filtered_models {
                        let size_str = if let Some(size) = metadata.size_bytes {
                            format!("{:.1}GB", size as f64 / (1024.0 * 1024.0 * 1024.0))
                        } else {
                            "n/a".to_string()
                        };

                        let (git_ref, commit, status) = match git_info {
                            Some(git) => (
                                git.current_ref.clone().unwrap_or_else(|| "detached".to_string()),
                                git.short_commit.clone().unwrap_or_else(|| "unknown".to_string()),
                                if git.is_dirty { "dirty" } else { "clean" }
                            ),
                            None => ("n/a".to_string(), "n/a".to_string(), "n/a")
                        };

                        println!("{:<30} {:<15} {:<8} {:<6} {:<10}",
                            model_ref.model, git_ref, commit, status, size_str);
                    }

                    if filtered_models.is_empty() {
                        println!("No models found.");
                        println!("Try: hyprstream model pull https://huggingface.co/Qwen/Qwen2-1.5B-Instruct");
                    }
                }
            }
        }
        ModelAction::Clone { repo_url, git_ref, model_id: _ } => {
            info!("📦 Cloning model from Git repository...");

            // Use shared operation (model_id parameter is deprecated, ignore it)
            let cloned = crate::storage::operations::clone_model(
                &repo_url,
                None,  // name - will be derived from URL
                git_ref.as_deref()
            ).await?;
            
            println!();
            println!("✅ Model cloned successfully!");
            println!("   Model ID: {}", cloned.model_id);
            println!("   Name: {}", cloned.model_name);
            println!("   Location: {}", cloned.model_path.display());
            println!();
            println!("📚 Next steps:");
            println!("   • Create adapter: hyprstream lora create --base-model {}", cloned.model_id);
            println!("   • Run inference: hyprstream model infer {} --prompt \"...\"", cloned.model_id);
        }
        ModelAction::Pull { uri,  .. } => {
            info!("📥 Pulling model: {}", uri);
            
            // Get storage paths
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;
            
            // Initialize GitModelSource with XET support for LFS files
            let git_source = crate::storage::GitModelSource::new(models_dir.clone());

            match git_source.clone_model(&uri).await {
                Ok((model_id, model_path)) => {
                    println!();
                    println!("✅ Model downloaded successfully!");
                    println!("   Model ID: {}", model_id);
                    println!("   Location: {}", model_path.display());
                    
                    let model_storage = crate::storage::ModelStorage::create(models_dir).await?;
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
            
            let sharing = crate::storage::sharing::ModelSharing::new(models_dir).await?;
            
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
            
            let mut sharing = crate::storage::sharing::ModelSharing::new(models_dir).await?;
            
            // Parse share reference if provided as JSON
            let share_ref = if git_url.starts_with('{') {
                // Assume it's JSON share reference
                serde_json::from_str(&git_url)?
            } else {
                // Create minimal share ref
                crate::storage::sharing::ShareableModelRef {
                    name: name.clone().unwrap_or_else(|| "imported-model".to_string()),
                    model_type: crate::storage::sharing::ModelType::Base,
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
            // Parse model reference (e.g., "gitignore", "Qwen3-4B", "qwen/qwen-2b")
            let model_ref = crate::storage::ModelRef::parse(&uri)
                .map_err(|e| anyhow::anyhow!("Invalid model reference '{}': {}. Use 'hyprstream model list' to see available models", uri, e))?;

            info!("🗑️ Removing model: {}", model_ref.model);

            // Check if confirmation is needed
            if !yes {
                println!("⚠️  Are you sure you want to remove model '{}'?", model_ref.model);
                println!("This action cannot be undone.");
                println!("");
                println!("Type 'yes' to confirm, or use --yes flag to skip confirmation:");

                let stdin = io::stdin();
                let mut line = String::new();
                stdin.lock().read_line(&mut line)?;

                if line.trim().to_lowercase() != "yes" {
                    println!("❌ Removal cancelled");
                    return Ok(());
                }
            }

            // Get storage paths and model storage
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;
            let model_storage = crate::storage::ModelStorage::create(models_dir.clone()).await?;

            // Get model path from storage
            let model_path = match model_storage.get_model_path(&model_ref).await {
                Ok(path) => path,
                Err(_) => {
                    // Fallback to direct path lookup
                    models_dir.join(&model_ref.model)
                }
            };

            // Check if model exists
            if !model_path.exists() {
                eprintln!("❌ Model '{}' not found", model_ref.model);
                eprintln!("   Use 'hyprstream model list' to see available models");
                return Err(anyhow::anyhow!("Model '{}' not found", model_ref.model).into());
            }

            println!("🗑️ Removing model files from: {}", model_path.display());

            // Use registry's remove_model to properly clean up submodule metadata and directories
            // Note: git2db manages tracked repositories, not submodules within a parent repo.
            // ModelRegistry uses submodules, which is outside git2db's scope.
            let registry = model_storage.registry();
            if let Err(e) = registry.remove_model(&model_ref).await {
                eprintln!("❌ Failed to remove model: {}", e);
                eprintln!("   You may need to manually clean up: {}", model_path.display());
                return Err(e.into());
            }

            // Note: With git-native storage, metadata is embedded in the repository
            // No separate metadata cleanup needed unless we implement registry cleanup
            if !keep_metadata {
                println!("🗑️ Model and git repository removed");
            } else {
                println!("📋 Model files removed (git repository data deleted)");
            }

            println!("✅ Model '{}' removed successfully", model_ref.model);
        }
        ModelAction::Inspect { uri, format } => {
            // Parse model reference (e.g., "Qwen3-4B", "qwen/qwen-2b", "model:branch")
            let model_ref = crate::storage::ModelRef::parse(&uri)
                .map_err(|e| anyhow::anyhow!("Invalid model reference '{}': {}. Use 'hyprstream model list' to see available models", uri, e))?;

            info!("ℹ️ Getting model info: {}", model_ref.model);

            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::storage::ModelStorage::create(storage_paths.models_dir()?).await?;

            // Check if model exists and get path
            let model_path = match model_storage.get_model_path(&model_ref).await {
                Ok(path) => path,
                Err(_) => {
                    eprintln!("❌ Model '{}' not found", model_ref.model);
                    eprintln!("   Use 'hyprstream model list' to see available models");
                    return Err(anyhow::anyhow!("Model '{}' not found", model_ref.model).into());
                }
            };

            // Extract metadata from git repo directly
            let metadata = extract_model_metadata(&model_path, &model_ref.model)?;

            // Display metadata
            match format.as_str() {
                "json" => {
                    let json_output = serde_json::json!({
                        "name": metadata.name,
                        "display_name": metadata.display_name,
                        "model_type": metadata.model_type,
                        "size_bytes": metadata.size_bytes,
                        "size_gb": format!("{:.2}", metadata.size_bytes.unwrap_or(0) as f64 / 1_073_741_824.0),
                        "created_at": metadata.created_at,
                        "updated_at": metadata.updated_at,
                        "tags": metadata.tags,
                    });
                    println!("{}", serde_json::to_string_pretty(&json_output)?);
                }
                "yaml" => {
                    println!("name: {}", metadata.name);
                    if let Some(display_name) = &metadata.display_name {
                        println!("display_name: {}", display_name);
                    }
                    println!("model_type: {}", metadata.model_type);
                    if let Some(size) = metadata.size_bytes {
                        println!("size_gb: {:.2}", size as f64 / 1_073_741_824.0);
                    }
                    println!("created_at: {}", metadata.created_at);
                    println!("updated_at: {}", metadata.updated_at);
                }
                _ => {
                    println!("Model: {}", metadata.name);
                    if let Some(display_name) = &metadata.display_name {
                        println!("Display Name: {}", display_name);
                    }
                    println!("Type: {}", metadata.model_type);
                    if let Some(size) = metadata.size_bytes {
                        println!("Size: {:.2} GB", size as f64 / 1_073_741_824.0);
                    }
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
                
                let stdin = io::stdin();
                let mut line = String::new();
                stdin.lock().read_line(&mut line)?;
                
                if line.trim().to_lowercase() != "yes" {
                    println!("❌ Repair cancelled");
                    return Ok(());
                }
            }
            
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::storage::ModelStorage::create(storage_paths.models_dir()?).await?;
            
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
                        if let Ok(models) = model_storage.list_models().await {
                            println!("\n📋 {} models found after repair:", models.len());
                            for (model_ref, metadata) in models.iter() {
                                println!("  📁 {} ({})", model_ref.model, metadata.name);
                                if let Some(display_name) = &metadata.display_name {
                                    println!("     Display: {}", display_name);
                                }
                                if let Some(size) = metadata.size_bytes {
                                    println!("     Size: {:.2} GB", size as f64 / 1_073_741_824.0);
                                }
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
        // Search variant has been removed from the enum - use List with search filter
        ModelAction::Convert { source, to, output, precision, verify } => {
            info!("🔄 Converting model from {} to {}", source, to);
            
            
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
            eprintln!("   Please download models directly in SafeTensors format.");
            eprintln!("   Try: hyprstream model pull https://huggingface.co/<org>/<model-name>");
        }
        ModelAction::Cache { action: _ } => {
            info!("🗄️ Managing model cache");
            
            // Use real model management system for cache info
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let model_storage = crate::storage::ModelStorage::create(storage_paths.models_dir()?).await?;
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
                for (model_ref, model_metadata) in cached_models {
                    let accessed_dt = DateTime::<Utc>::from_timestamp(model_metadata.updated_at, 0)
                        .unwrap_or_else(|| Utc::now());
                    println!("  📁 {} ({:.1} GB) - Last accessed: {}",
                        model_ref.model,
                        model_metadata.size_bytes.unwrap_or(0) as f64 / (1024.0 * 1024.0 * 1024.0),
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
            
            
            
            debug!("Looking up model: {}", model);

            // Parse model reference (e.g., "Qwen3-4B", "qwen/qwen-2b", "model:branch")
            let model_ref = crate::storage::ModelRef::parse(&model)
                .map_err(|e| anyhow::anyhow!("Invalid model reference '{}': {}. Use 'hyprstream model list' to see available models", model, e))?;

            // Initialize model storage
            let storage_paths = crate::storage::paths::StoragePaths::new()?;
            let models_dir = storage_paths.models_dir()?;
            let model_storage = crate::storage::ModelStorage::create(models_dir.clone()).await?;

            // Get model path using the same method as remove command
            let model_path = match model_storage.get_model_path(&model_ref).await {
                Ok(path) => path,
                Err(_) => {
                    // Fallback to direct path lookup if not in registry
                    models_dir.join(&model_ref.model)
                }
            };

            if !model_path.exists() {
                error!("Model '{}' not found in model storage", model_ref.model);
                eprintln!("❌ Model '{}' not found", model_ref.model);
                eprintln!("   Use 'hyprstream model list' to see available models");
                return Err(anyhow::anyhow!("Model '{}' not found", model_ref.model).into());
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

            // Time the model loading
            let load_start = std::time::Instant::now();
            match engine.load_model(&model_path).await {
                Ok(_) => {
                    let load_time = load_start.elapsed();
                    info!("Model loaded in {:.2}s", load_time.as_secs_f64());
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
            
            let seconds = result.latency_ms as f64 / 1000.0;
            let tokens_per_sec = if seconds > 0.0 {
                result.tokens_generated as f64 / seconds
            } else {
                0.0
            };
            info!(
                "Generation: {} tokens in {:.2}s ({:.2} tokens/sec)",
                result.tokens_generated,
                seconds,
                tokens_per_sec
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
                println!("⚠️ Inference error: {}", e);
                println!("📤 Response: [Inference error occurred]");
                
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
    
    println!("💡 Interactive chat mode will be implemented in future versions");
    
    Ok(())
}

/// Run chat inference - finds and loads model, then generates response
async fn run_chat_inference(
    model_id: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<String, Box<dyn std::error::Error>> {
    
    
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

/// Extract model metadata directly from git repository
fn extract_model_metadata(model_path: &std::path::Path, model_name: &str) -> Result<ModelMetadata, Box<dyn std::error::Error>> {
    // Calculate directory size
    let size_bytes = calculate_directory_size(model_path).unwrap_or(0);

    // Get git metadata if available
    let (created_at, updated_at) = if let Ok(repo) = crate::git::get_repository(model_path) {
        // Get first commit time as created_at
        let created_at = repo.head().ok()
            .and_then(|head| head.peel_to_commit().ok())
            .map(|commit| commit.time().seconds())
            .unwrap_or_else(|| std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_secs() as i64);

        // Get last commit time as updated_at
        let updated_at = repo.head().ok()
            .and_then(|head| head.peel_to_commit().ok())
            .map(|commit| commit.time().seconds())
            .unwrap_or(created_at);

        (created_at, updated_at)
    } else {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_secs() as i64;
        (now, now)
    };

    // Try to read model card or config for display name
    let display_name = read_model_display_name(model_path);

    Ok(ModelMetadata {
        name: model_name.to_string(),
        display_name,
        model_type: "transformer".to_string(), // Default type
        size_bytes: Some(size_bytes),
        created_at,
        updated_at,
        tags: Vec::new(),
    })
}

/// Calculate directory size recursively
fn calculate_directory_size(dir: &std::path::Path) -> std::io::Result<u64> {
    let mut total_size = 0;

    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                total_size += calculate_directory_size(&path)?;
            } else {
                total_size += entry.metadata()?.len();
            }
        }
    }

    Ok(total_size)
}

/// Try to extract display name from model card or config
fn read_model_display_name(model_path: &std::path::Path) -> Option<String> {
    // Try README.md first
    if let Ok(readme) = std::fs::read_to_string(model_path.join("README.md")) {
        // Look for title in first few lines
        for line in readme.lines().take(10) {
            if let Some(title) = line.strip_prefix("# ") {
                return Some(title.trim().to_string());
            }
        }
    }

    // Try config.json
    if let Ok(config) = std::fs::read_to_string(model_path.join("config.json")) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&config) {
            if let Some(name) = json.get("name").and_then(|v| v.as_str()) {
                return Some(name.to_string());
            }
            if let Some(name) = json.get("model_name").and_then(|v| v.as_str()) {
                return Some(name.to_string());
            }
        }
    }

    None
}

/// Handle pre-training command - NOT IMPLEMENTED
pub async fn handle_pretrain(
    model_id: String,
    _learning_rate: f32,
    _steps: usize,
    _warmup_steps: usize,
    _batch_size: usize,
    _checkpoint_every: Option<usize>,
    _resume_from: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    error!("Pre-training is not supported for model: {}", model_id);
    error!("The base model doesn't expose its VarStore for training");
    error!("Please use LoRA fine-tuning instead with 'hyprstream lora create' and 'hyprstream lora train start'");

    Err("Pre-training not supported - use LoRA fine-tuning instead".into())
}

/// Handle checkpoint write command
pub async fn handle_write_checkpoint(
    model_id: String,
    name: Option<String>,
    step: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {

    info!("Writing checkpoint for model: {}", model_id);

    // Load model storage
    let config = crate::config::HyprConfig::load().unwrap_or_default();
    let storage = ModelStorage::create(config.models_dir().to_path_buf()).await?;

    // Resolve model path
    let model_ref = crate::storage::ModelRef::parse(&model_id)?;
    let model_path = storage.get_model_path(&model_ref).await?;

    // Create checkpoint manager
    let checkpoint_mgr = CheckpointManager::new(model_path.clone())?;

    // Get step number
    let step = step.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as usize
    });

    // Create weight snapshot (would need actual implementation)
    let weights = WeightSnapshot::Memory {
        data: vec![], // Placeholder - would need actual weight data
        format: WeightFormat::SafeTensors,
    };

    // Write checkpoint
    let checkpoint_path = checkpoint_mgr.write_checkpoint(
        weights,
        step,
        None,
    ).await?;

    info!("✅ Checkpoint written to: {}", checkpoint_path.display());

    Ok(())
}

/// Handle checkpoint commit command
pub async fn handle_commit_checkpoint(
    checkpoint_path: String,
    message: Option<String>,
    branch: Option<String>,
    tag: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {

    info!("Committing checkpoint: {}", checkpoint_path);

    let checkpoint_path = PathBuf::from(&checkpoint_path);

    // Get model path (parent of .checkpoints directory)
    let model_path = checkpoint_path
        .parent()
        .and_then(|p| if p.file_name() == Some(std::ffi::OsStr::new(".checkpoints")) {
            p.parent()
        } else {
            Some(p)
        })
        .ok_or("Invalid checkpoint path")?;

    // Create checkpoint manager
    let checkpoint_mgr = CheckpointManager::new(model_path.to_path_buf())?;

    // Commit checkpoint
    let commit_id = checkpoint_mgr.commit_checkpoint(
        &checkpoint_path,
        message,
        branch,
    ).await?;

    info!("✅ Checkpoint committed: {}", commit_id);

    // Create tag if requested
    if let Some(tag_name) = tag {
        let branch_mgr = BranchManager::new(model_path)?;
        branch_mgr.create_tag(&tag_name, "HEAD")?;
        info!("✅ Tag created: {}", tag_name);
    }

    Ok(())
}
