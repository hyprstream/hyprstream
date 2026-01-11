//! Training command handlers
//!
//! Handlers for the `training` subcommand:
//! - `training init` - Initialize adapter for training
//! - `training infer` - Inference with TTT (dirty writes)
//! - `training batch` - Batch training with checkpoints
//! - `training checkpoint` - Commit dirty adapter changes

use anyhow::{bail, Result};
use tracing::{info, warn};

use crate::cli::commands::KVQuantArg;
use crate::config::{HyprstreamTrainingConfig, TrainingMode, TTTTrainingConfig};
use crate::runtime::model_config::ModelConfig;
use crate::runtime::template_engine::ChatMessage;
use crate::services::{InferenceZmqClient, PolicyZmqClient, INFERENCE_ENDPOINT};
use crate::storage::{ModelRef, ModelStorage};
use hyprstream_rpc::{RequestIdentity, SigningKey, VerifyingKey};
use std::path::PathBuf;

/// Handle `training init` command
///
/// Initializes a LoRA adapter for training. Optionally creates a new branch/worktree.
pub async fn handle_training_init(
    storage: &ModelStorage,
    model_ref_str: &str,
    branch_name: Option<String>,
    adapter_name: Option<String>,
    index: Option<u32>,
    rank: u32,
    alpha: u32,
    mode: &str,
    learning_rate: f32,
) -> Result<()> {
    let model_ref = ModelRef::parse(model_ref_str)?;

    // If branch specified, create isolated training environment
    if let Some(new_branch) = branch_name {
        info!(
            "Creating isolated training environment: branch {} from {}",
            new_branch,
            model_ref.git_ref.display_name()
        );

        // Create branch from model_ref's git_ref
        let from_ref = model_ref.git_ref.to_ref_string();
        storage
            .create_branch(&model_ref, &new_branch, from_ref.as_deref())
            .await?;
        println!(
            "✓ Created branch {} from {}",
            new_branch,
            model_ref.git_ref.display_name()
        );

        // Create worktree for new branch
        let worktree_path = storage.create_worktree(&model_ref, &new_branch).await?;
        println!("✓ Created worktree at {}", worktree_path.display());

        // Initialize adapter in worktree
        init_adapter_at_path(
            &worktree_path,
            adapter_name.as_deref(),
            index,
            rank,
            alpha,
            learning_rate,
            mode,
            &format!("{}:{}", model_ref.model, new_branch),
        )
        .await?;

        println!("\n✓ Isolated training environment ready!");
        println!("\n→ Next steps:");
        println!(
            "  hyprstream training infer {}:{} --prompt \"...\"",
            model_ref.model, new_branch
        );
        println!(
            "  hyprstream training checkpoint {}:{}",
            model_ref.model, new_branch
        );

        return Ok(());
    }

    // Train on existing worktree
    let branch_name = match &model_ref.git_ref {
        git2db::GitRef::Branch(name) => name.clone(),
        git2db::GitRef::DefaultBranch => {
            let base_ref = ModelRef::new(model_ref.model.clone());
            storage.get_default_branch(&base_ref).await?
        }
        _ => {
            bail!(
                "Training requires a branch reference. Use model:branch format (e.g., {}:main)",
                model_ref.model
            );
        }
    };

    let model_path = storage
        .get_worktree_path(&model_ref, &branch_name)
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "Worktree '{}' does not exist for model '{}'. Create with:\n  \
                 hyprstream training init {} --branch {}\nError: {}",
                branch_name,
                model_ref.model,
                model_ref.model,
                branch_name,
                e
            )
        })?;

    init_adapter_at_path(
        &model_path,
        adapter_name.as_deref(),
        index,
        rank,
        alpha,
        learning_rate,
        mode,
        &model_ref.to_string(),
    )
    .await?;

    println!("\n✓ Training initialization complete!");
    println!("\n→ Next steps:");
    println!(
        "  hyprstream training infer {} --prompt \"...\"",
        model_ref
    );
    println!("  hyprstream training checkpoint {}", model_ref);

    Ok(())
}

/// Initialize adapter at a specific path
async fn init_adapter_at_path(
    model_path: &std::path::Path,
    adapter_name: Option<&str>,
    index: Option<u32>,
    rank: u32,
    alpha: u32,
    learning_rate: f32,
    mode: &str,
    model_ref_str: &str,
) -> Result<()> {
    use crate::runtime::{RuntimeEngine, TorchEngine};
    use crate::storage::AdapterManager;

    let adapter_manager = AdapterManager::new(model_path);
    adapter_manager.ensure_adapters_dir()?;

    let adapter_base_name = adapter_name.unwrap_or("default");
    let indexed_adapter_name = adapter_manager.create_indexed_name(adapter_base_name, index)?;

    println!("\n→ Initializing adapter: {}", indexed_adapter_name);

    // Create adapter configuration
    let adapter_config = crate::storage::AdapterConfig {
        rank,
        alpha: alpha as f32,
        learning_rate: learning_rate as f64,
        batch_size: 4,
        epochs: 10,
        model_ref: model_ref_str.to_string(),
        training_data: None,
        ..Default::default()
    };

    // Load model to get proper dimensions
    info!("Loading model to determine LoRA structure");
    let config = crate::config::RuntimeConfig::default();
    let mut engine = TorchEngine::new(config)?;
    RuntimeEngine::load_model(&mut engine, model_path).await?;

    // Create LoRA configuration
    let lora_config = crate::lora::LoRAConfig {
        rank: rank as usize,
        alpha: alpha as f32,
        dropout: 0.1,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "down_proj".to_string(),
        ],
        learning_rate,
    };

    engine.create_lora(lora_config)?;

    // Save adapter weights
    let adapter_path = adapter_manager
        .adapters_dir
        .join(format!("{}.safetensors", indexed_adapter_name));
    let adapter_path_str = adapter_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("adapter path is not valid UTF-8"))?;
    engine.save_lora_weights(adapter_path_str)?;

    // Save adapter config
    let config_path = adapter_manager
        .adapters_dir
        .join(format!("{}.config.json", indexed_adapter_name));
    let config_json = serde_json::to_string_pretty(&adapter_config)?;
    std::fs::write(&config_path, config_json)?;

    println!(
        "✓ Created adapter: adapters/{}.safetensors",
        indexed_adapter_name
    );

    // Save training mode config
    save_training_config(model_path, mode, &indexed_adapter_name, learning_rate)?;

    Ok(())
}

/// Save training configuration to model's config.json
fn save_training_config(
    model_path: &std::path::Path,
    mode_str: &str,
    target_adapter: &str,
    learning_rate: f32,
) -> Result<()> {
    let mode = match mode_str.to_lowercase().as_str() {
        "ttt" | "test_time_training" | "test-time-training" => TrainingMode::TestTimeTraining,
        "supervised" => TrainingMode::Supervised,
        "disabled" | "off" | "none" => TrainingMode::Disabled,
        _ => {
            warn!("Unknown training mode '{}', defaulting to ttt", mode_str);
            TrainingMode::TestTimeTraining
        }
    };

    let training_config = HyprstreamTrainingConfig {
        mode: mode.clone(),
        target_adapter: Some(target_adapter.to_string()),
        learning_rate: learning_rate as f64,
        batch_size: 4,
        steps_per_cycle: 10,
        min_quality_threshold: 0.3,
        train_base_model: false,
        ttt: TTTTrainingConfig {
            learning_rate: learning_rate as f64,
            gradient_steps: 3,
            max_grad_norm: 1.0,
            min_input_length: 32,
            max_ttt_context: 512,
        },
    };

    ModelConfig::save_training_config(model_path, &training_config)?;

    match mode {
        TrainingMode::Disabled => println!("✓ Training mode: disabled"),
        TrainingMode::TestTimeTraining => {
            println!("✓ Training mode: TTT (test-time-training)");
            println!(
                "  → {} gradient steps, lr={}",
                training_config.ttt.gradient_steps, training_config.ttt.learning_rate
            );
        }
        TrainingMode::Supervised => println!("✓ Training mode: supervised"),
    }

    Ok(())
}

/// Handle `training infer` command
///
/// Runs inference with TTT enabled, making dirty writes to adapter weights.
pub async fn handle_training_infer(
    storage: &ModelStorage,
    model_ref_str: &str,
    prompt: &str,
    image_path: Option<String>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    stream: bool,
    max_context: Option<usize>,
    kv_quant: KVQuantArg,
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
) -> Result<()> {
    use crate::runtime::RuntimeConfig;
    use crate::services::InferenceService;

    info!(
        "Training inference: model={}, prompt_len={}",
        model_ref_str,
        prompt.len()
    );

    // Parse model reference and get path
    let model_ref = ModelRef::parse(model_ref_str)?;
    let storage_paths = crate::storage::StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;
    let model_path = match storage.get_model_path(&model_ref).await {
        Ok(path) => path,
        Err(_) => models_dir.join(&model_ref.model),
    };

    if !model_path.exists() {
        bail!("Model '{}' not found. Use 'hyprstream list' to see available models", model_ref.model);
    }

    // Ensure TTT is enabled in config
    ensure_ttt_enabled(&model_path)?;

    info!("Using model at: {} (TTT enabled)", model_path.display());

    // Create policy client
    let policy_client = PolicyZmqClient::new(signing_key.clone(), RequestIdentity::local());

    // Configure runtime
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.max_context = max_context;
    runtime_config.kv_quant_type = kv_quant.into();

    // Start InferenceService with TTT enabled
    let mut service_handle = InferenceService::start_at(
        &model_path,
        runtime_config,
        verifying_key,
        policy_client,
        INFERENCE_ENDPOINT,
    )
    .await?;

    // Create client for service communication
    let client = InferenceZmqClient::new(signing_key, RequestIdentity::local());

    // Apply chat template
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
    }];

    let formatted_prompt = match client.apply_chat_template(&messages, true).await {
        Ok(formatted) => formatted,
        Err(e) => {
            warn!("Could not apply chat template: {}. Using raw prompt.", e);
            prompt.to_string()
        }
    };

    // Build generation request
    let sampling_params = crate::config::SamplingParams::from_model_path(&model_path)
        .await
        .unwrap_or_default();
    let mut request_builder = crate::runtime::GenerationRequest::builder(formatted_prompt)
        .apply_config(&sampling_params);

    if let Some(t) = temperature {
        request_builder = request_builder.temperature(t);
    }
    if let Some(p) = top_p {
        request_builder = request_builder.top_p(p);
    }
    if let Some(k) = top_k {
        request_builder = request_builder.top_k(Some(k));
    }
    if let Some(r) = repeat_penalty {
        request_builder = request_builder.repeat_penalty(r);
    }
    if let Some(m) = max_tokens {
        request_builder = request_builder.max_tokens(m);
    }

    // Add image path if provided (for multimodal models)
    if let Some(img_path) = image_path {
        info!("Using image: {}", img_path);
        request_builder = request_builder.image_path(std::path::PathBuf::from(img_path));
    }

    let request = request_builder.build();

    // Generate with TTT
    // Note: For training inference, we use non-streaming mode to simplify
    // and ensure TTT adaptation is applied fully before output
    if stream {
        warn!("Streaming mode not fully supported for training infer, using non-streaming");
    }

    let response = client.generate(&request).await?;
    println!("{}", response.text);

    println!("\n[TTT applied - adapter weights modified]");
    println!("Run 'hyprstream training checkpoint {}' to commit changes", model_ref);

    // Stop service to properly release resources
    service_handle.stop().await;

    Ok(())
}

/// Ensure TTT is enabled in model config
fn ensure_ttt_enabled(model_path: &std::path::Path) -> Result<()> {
    let mut config = ModelConfig::load_training_config(model_path).unwrap_or_else(|| {
        HyprstreamTrainingConfig {
            mode: TrainingMode::TestTimeTraining,
            ..Default::default()
        }
    });

    if config.mode != TrainingMode::TestTimeTraining {
        config.mode = TrainingMode::TestTimeTraining;
        ModelConfig::save_training_config(model_path, &config)?;
        info!("Enabled TTT mode in config");
    }

    Ok(())
}

/// Handle `training batch` command
///
/// Batch training: starts InferenceService once and processes all files with TTT.
/// Each file's content is sent as a generation request, triggering TTT adaptation.
pub async fn handle_training_batch(
    storage: &ModelStorage,
    model_ref_str: &str,
    input_files: Vec<String>,
    input_dir: Option<PathBuf>,
    pattern: &str,
    format: &str,
    max_tokens: usize,
    chunk_size: usize,
    skip: usize,
    limit: Option<usize>,
    progress_interval: usize,
    checkpoint_interval: usize,
    test_set: Option<String>,
    max_context: Option<usize>,
    kv_quant: KVQuantArg,
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
) -> Result<()> {
    use crate::runtime::RuntimeConfig;
    use crate::services::{InferenceService, INFERENCE_ENDPOINT};
    use crate::storage::AdapterManager;
    use glob::glob;

    info!(
        "Batch training: model={}, format={}",
        model_ref_str, format
    );

    let model_ref = ModelRef::parse(model_ref_str)?;
    let storage_paths = crate::storage::StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;

    let model_path = match storage.get_model_path(&model_ref).await {
        Ok(path) => path,
        Err(_) => {
            let base_path = models_dir.join(&model_ref.model);
            if !base_path.exists() {
                bail!("Model '{}' not found", model_ref.model);
            }
            let branch_name = model_ref.git_ref.to_string();
            bail!(
                "Worktree '{}' not found for model {}.\n\
                 Run: hyprstream training init {} --branch {}",
                branch_name,
                model_ref.model,
                model_ref.model,
                branch_name
            );
        }
    };

    // Collect input files
    let mut files: Vec<PathBuf> = Vec::new();

    for file_pattern in &input_files {
        for entry in glob(file_pattern)? {
            if let Ok(path) = entry {
                if path.is_file() {
                    files.push(path);
                }
            }
        }
    }

    if let Some(dir) = input_dir {
        if !dir.exists() {
            bail!("Input directory does not exist: {}", dir.display());
        }
        let full_pattern = dir.join(pattern);
        for entry in glob(&full_pattern.to_string_lossy())? {
            if let Ok(path) = entry {
                if path.is_file() {
                    files.push(path);
                }
            }
        }
    }

    if files.is_empty() {
        bail!("No files found matching input patterns");
    }

    files.sort();
    let files: Vec<_> = files
        .into_iter()
        .skip(skip)
        .take(limit.unwrap_or(usize::MAX))
        .collect();

    let total_files = files.len();
    println!("Found {} files to process", total_files);

    // Ensure TTT is enabled in model config
    ensure_ttt_enabled(&model_path)?;

    // Load test set (for future validation)
    let test_files: Vec<PathBuf> = if let Some(ref pattern) = test_set {
        glob(pattern)?
            .filter_map(|e| e.ok())
            .filter(|p| p.is_file())
            .collect()
    } else {
        vec![]
    };

    if !test_files.is_empty() {
        println!("Loaded {} test files for validation", test_files.len());
    }

    // Start InferenceService with TTT enabled
    println!("Starting InferenceService for {}...", model_ref_str);
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.max_context = max_context;
    runtime_config.kv_quant_type = kv_quant.into();

    let policy_client = PolicyZmqClient::new(signing_key.clone(), RequestIdentity::local());
    let mut service_handle = InferenceService::start_at(
        &model_path,
        runtime_config,
        verifying_key,
        policy_client,
        INFERENCE_ENDPOINT,
    )
    .await?;

    let client = InferenceZmqClient::new(signing_key, RequestIdentity::local());

    // Get adapter info for checkpoint saves
    let adapter_manager = AdapterManager::new(&model_path);
    let adapters = adapter_manager.list_adapters().unwrap_or_default();

    println!("\nStarting batch training...\n");

    let mut processed = 0;
    let mut total_input_tokens = 0usize;
    let mut total_output_tokens = 0usize;
    let start_time = std::time::Instant::now();

    for file_path in &files {
        // Read file content
        let content = std::fs::read_to_string(file_path)?;
        let chunk: String = content.chars().take(chunk_size).collect();

        if chunk.is_empty() {
            continue;
        }

        // Build generation request - TTT is applied by InferenceService before generation
        let prompt = match format {
            "jsonl" => {
                // Parse JSONL format - expect {"prompt": "...", "completion": "..."} or similar
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&chunk) {
                    parsed
                        .get("prompt")
                        .or_else(|| parsed.get("text"))
                        .or_else(|| parsed.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or(&chunk)
                        .to_string()
                } else {
                    chunk.clone()
                }
            }
            _ => chunk.clone(), // "text" format - use as-is
        };

        let request = crate::runtime::GenerationRequest::builder(prompt)
            .max_tokens(max_tokens)
            .temperature(0.7)
            .build();

        // Send generation request - InferenceService applies TTT during this call
        match client.generate(&request).await {
            Ok(response) => {
                // Track tokens from response
                total_input_tokens += response.prefill_tokens;
                total_output_tokens += response.tokens_generated;
            }
            Err(e) => {
                warn!("Generation failed for {}: {}", file_path.display(), e);
            }
        }

        processed += 1;

        // Progress update
        if processed % progress_interval == 0 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let rate = processed as f32 / elapsed;
            let tokens_per_sec = (total_input_tokens + total_output_tokens) as f32 / elapsed;
            println!(
                "[{}/{}] {:.1} files/sec, {} input tokens, {} output tokens ({:.0} tok/s)",
                processed, total_files, rate, total_input_tokens, total_output_tokens, tokens_per_sec
            );
        }

        // Checkpoint - save adapter weights
        if processed % checkpoint_interval == 0 {
            println!("\n→ Checkpoint at {} files...", processed);
            // Request the service to save weights (if supported), or save manually
            if let Some(adapter) = adapters.first() {
                println!("  Adapter: {}", adapter.name);
            }
            println!("  ✓ Weights updated (will be saved on service stop)");
        }
    }

    // Stop service - this saves final adapter weights
    println!("\nStopping InferenceService and saving weights...");
    service_handle.stop().await;

    let elapsed = start_time.elapsed();
    let total_tokens = total_input_tokens + total_output_tokens;
    let tokens_per_sec = total_tokens as f32 / elapsed.as_secs_f32();
    println!("\n✓ Batch training complete!");
    println!("  Processed: {} files", processed);
    println!("  Input tokens: {}", total_input_tokens);
    println!("  Output tokens: {}", total_output_tokens);
    println!("  Total tokens: {} ({:.0} tok/s)", total_tokens, tokens_per_sec);
    println!("  Time: {:.1}s", elapsed.as_secs_f32());
    println!("\n→ Run 'hyprstream training checkpoint {}' to commit", model_ref);

    Ok(())
}

/// Handle `training checkpoint` command
///
/// Commits dirty adapter changes to git.
pub async fn handle_training_checkpoint(
    storage: &ModelStorage,
    model_ref_str: &str,
    message: Option<String>,
    push: bool,
    remote: &str,
) -> Result<()> {
    use crate::storage::AdapterManager;

    let model_ref = ModelRef::parse(model_ref_str)?;
    let model_path = storage.get_model_path(&model_ref).await?;

    let adapter_manager = AdapterManager::new(&model_path);
    let adapters = adapter_manager.list_adapters()?;

    if adapters.is_empty() {
        bail!("No adapters found in {}", model_path.display());
    }

    // Check for dirty files
    let repo = git2db::GitManager::global()
        .get_repository(&model_path)?
        .open()?;

    let statuses = repo.statuses(None)?;
    let dirty_adapters: Vec<_> = statuses
        .iter()
        .filter(|s| {
            s.path()
                .map(|p| p.contains("adapters/") && p.ends_with(".safetensors"))
                .unwrap_or(false)
        })
        .collect();

    if dirty_adapters.is_empty() {
        println!("No dirty adapter files to commit.");
        return Ok(());
    }

    println!("Found {} modified adapter files:", dirty_adapters.len());
    for status in &dirty_adapters {
        if let Some(path) = status.path() {
            println!("  {}", path);
        }
    }

    // Stage adapter files
    let mut index = repo.index()?;
    for adapter in &adapters {
        let relative_path = adapter
            .path
            .strip_prefix(&model_path)
            .unwrap_or(&adapter.path);
        index.add_path(relative_path)?;

        // Also stage config if exists
        let config_path = adapter.path.with_extension("config.json");
        if config_path.exists() {
            let relative_config = config_path
                .strip_prefix(&model_path)
                .unwrap_or(&config_path);
            index.add_path(relative_config)?;
        }
    }
    index.write()?;

    // Create commit
    let commit_message = message.unwrap_or_else(|| {
        format!(
            "Training checkpoint: {} adapters updated\n\n\
             🤖 Generated with hyprstream training",
            adapters.len()
        )
    });

    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;
    let head = repo.head()?.peel_to_commit()?;
    let sig = repo.signature()?;

    let commit_id = repo.commit(Some("HEAD"), &sig, &sig, &commit_message, &tree, &[&head])?;

    println!("\n✓ Created commit: {}", &commit_id.to_string()[..8]);

    // Push if requested
    if push {
        println!("Pushing to {}...", remote);
        let mut remote_obj = repo.find_remote(remote)?;
        let refspec = format!(
            "refs/heads/{}:refs/heads/{}",
            model_ref.git_ref.to_string(),
            model_ref.git_ref.to_string()
        );
        remote_obj.push(&[&refspec], None)?;
        println!("✓ Pushed to {}", remote);
    }

    Ok(())
}
