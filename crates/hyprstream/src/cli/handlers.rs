//! CLI handlers for adaptive ML inference server

use crate::storage::ModelStorage;
use crate::training::{CheckpointManager, WeightFormat, WeightSnapshot};
use ::config::{Config, File};
use reqwest::Client;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info};

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

    let response = client.post(&url).json(&request_body).send().await?;

    if response.status().is_success() {
        let result: Value = response.json().await?;
        if verbose {
            debug!("Response: {}", serde_json::to_string_pretty(&result)?);
        }
        info!("✅ Embedding query processed successfully");
    } else {
        let status_code = response.status();
        let error_text = response.text().await?;
        error!(
            "❌ Query failed with status {}: {}",
            status_code, error_text
        );
    }

    Ok(())
}

pub async fn handle_server(ctx: crate::cli::AppContext) -> Result<(), Box<dyn std::error::Error>> {
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
    let server_state =
        crate::server::state::ServerState::new_with_git2db(server_config, git2db_config).await?;

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

    let socket_addr = SocketAddr::new(addr_parts[0].parse()?, addr_parts[1].parse()?);

    // Execute embedding query
    execute_sparse_query(
        Some(socket_addr),
        query.to_string(),
        config.as_ref(),
        verbose,
    )
    .await?;

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
        Err(format!(
            "Failed to create LoRA: HTTP {} - {}",
            status_code, error_text
        )
        .into())
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
        Err(format!(
            "Failed to list LoRA adapters: HTTP {} - {}",
            status_code, error_text
        )
        .into())
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
        Err(format!(
            "Failed to get LoRA info: HTTP {} - {}",
            status_code, error_text
        )
        .into())
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
        Err(format!(
            "Failed to start training: HTTP {} - {}",
            status_code, error_text
        )
        .into())
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
        Err(format!(
            "Failed to get training status: HTTP {} - {}",
            status_code, error_text
        )
        .into())
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
        Err(format!(
            "Failed to perform chat completion: HTTP {} - {}",
            status_code, error_text
        )
        .into())
    }
}

/// Handle chat command - inference with models/composed models

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
    _name: Option<String>,
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
    let checkpoint_path = checkpoint_mgr.write_checkpoint(weights, step, None).await?;

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
        .and_then(|p| {
            if p.file_name() == Some(std::ffi::OsStr::new(".checkpoints")) {
                p.parent()
            } else {
                Some(p)
            }
        })
        .ok_or("Invalid checkpoint path")?;

    // Create checkpoint manager
    let checkpoint_mgr = CheckpointManager::new(model_path.to_path_buf())?;

    // Commit checkpoint
    let commit_id = checkpoint_mgr
        .commit_checkpoint(&checkpoint_path, message, branch)
        .await?;

    info!("✅ Checkpoint committed: {}", commit_id);

    // Create tag if requested
    if let Some(tag_name) = tag {
        crate::git::helpers::create_tag(model_path, &tag_name)?;
        info!("✅ Tag created: {}", tag_name);
    }

    Ok(())
}
