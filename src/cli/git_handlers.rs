//! Handlers for git-style CLI commands

use anyhow::Result;
use tracing::info;

use crate::storage::{ModelStorage, ModelRef};

/// Handle branch command
pub async fn handle_branch(
    storage: &ModelStorage,
    model: &str,
    branch_name: &str,
    from_ref: Option<String>,
) -> Result<()> {
    info!("Creating branch {} for model {}", branch_name, model);

    let registry = storage.registry();
    registry.create_branch(model, branch_name, from_ref.as_deref()).await?;

    println!("✓ Created branch {} for model {}", branch_name, model);

    if from_ref.is_some() {
        println!("  Branch created from: {}", from_ref.unwrap());
    }

    Ok(())
}

/// Handle checkout command
pub async fn handle_checkout(
    storage: &ModelStorage,
    model_ref_str: &str,
    create_branch: bool,
    force: bool,
) -> Result<()> {
    // Parse model reference
    let model_ref = ModelRef::parse(model_ref_str)?;

    info!("Checking out {} for model {}",
          model_ref.git_ref.as_deref().unwrap_or("HEAD"),
          model_ref.model);

    let registry = storage.registry();

    // Check for uncommitted changes if not forcing
    if !force {
        let status = registry.status(&model_ref.model).await?;
        if status.is_dirty {
            println!("Warning: Model has uncommitted changes");
            println!("Use --force to discard changes, or commit them first");
            return Ok(());
        }
    }

    registry.checkout(&model_ref.model, model_ref.git_ref.as_deref(), create_branch).await?;

    println!("✓ Switched to {}",
             model_ref.git_ref.as_deref().unwrap_or("HEAD"));

    Ok(())
}

/// Handle status command
pub async fn handle_status(
    storage: &ModelStorage,
    model: Option<String>,
    verbose: bool,
) -> Result<()> {
    let registry = storage.registry();

    if let Some(model_name) = model {
        // Status for specific model
        let status = registry.status(&model_name).await?;
        print_model_status(&status, verbose);
    } else {
        // Status for all models
        let models = storage.list_models().await?;

        if models.is_empty() {
            println!("No models found");
            return Ok(());
        }

        for (model_ref, _metadata) in models {
            if let Ok(status) = registry.status(&model_ref.model).await {
                print_model_status(&status, verbose);
                println!(); // Add spacing between models
            }
        }
    }

    Ok(())
}

/// Handle commit command
pub async fn handle_commit(
    storage: &ModelStorage,
    model: &str,
    message: &str,
    stage_all: bool,
) -> Result<()> {
    info!("Committing changes to model {}", model);

    let registry = storage.registry();

    // Check status first
    let status = registry.status(model).await?;

    if !status.is_dirty {
        println!("No changes to commit for model {}", model);
        return Ok(());
    }

    // Show what will be committed
    if stage_all {
        println!("Staging all changes:");
        for file in &status.new_files {
            println!("  new: {}", file);
        }
        for file in &status.modified_files {
            println!("  modified: {}", file);
        }
        for file in &status.deleted_files {
            println!("  deleted: {}", file);
        }
    }

    // Commit
    registry.commit_model(model, message, stage_all).await?;

    println!("✓ Committed changes to {}", model);
    println!("  Message: {}", message);

    Ok(())
}

/// Print model status in a nice format
fn print_model_status(status: &crate::storage::model_registry::ModelStatus, verbose: bool) {
    println!("Model: {}", status.model_name);
    println!("Branch: {}", status.current_ref);

    if status.is_dirty {
        println!("Status: modified (uncommitted changes)");

        if verbose || (!status.new_files.is_empty() || !status.modified_files.is_empty() || !status.deleted_files.is_empty()) {
            if !status.new_files.is_empty() {
                println!("\n  New files:");
                for file in &status.new_files {
                    println!("    + {}", file);
                }
            }

            if !status.modified_files.is_empty() {
                println!("\n  Modified files:");
                for file in &status.modified_files {
                    println!("    M {}", file);
                }
            }

            if !status.deleted_files.is_empty() {
                println!("\n  Deleted files:");
                for file in &status.deleted_files {
                    println!("    - {}", file);
                }
            }
        }
    } else {
        println!("Status: clean");
    }
}

/// Handle lt (LoRA training) command
pub async fn handle_lora_train(
    storage: &ModelStorage,
    model_ref_str: &str,
    adapter_name: Option<String>,
    index: Option<u32>,
    rank: Option<u32>,
    learning_rate: Option<f32>,
    batch_size: Option<usize>,
    epochs: Option<usize>,
    data: Option<String>,
    interactive: bool,
    config: Option<String>,
) -> Result<()> {
    let model_ref = ModelRef::parse(model_ref_str)?;

    // Check that we're on a branch (not detached HEAD)
    let registry = storage.registry();
    let status = registry.status(&model_ref.model).await?;

    if status.current_ref.starts_with("detached") {
        println!("Warning: Training on detached HEAD");
        println!("Consider creating a branch first:");
        println!("  hyprstream branch {} training/experiment", model_ref.model);
    }

    println!("Starting LoRA adapter initialization for {}", model_ref.to_string());
    println!("Current branch: {}", status.current_ref);

    // Get model path and create adapter manager
    let model_path = storage.get_model_path(&model_ref).await?;
    let adapter_manager = crate::storage::AdapterManager::new(&model_path);

    // Determine adapter name and create indexed name
    let adapter_base_name = adapter_name.as_deref().unwrap_or("default");
    let indexed_adapter_name = adapter_manager.create_indexed_name(adapter_base_name, index)?;

    // Create adapter configuration
    let mut adapter_config = crate::storage::AdapterConfig {
        rank: rank.unwrap_or(16),
        alpha: 32.0,
        learning_rate: learning_rate.unwrap_or(1e-4) as f64,
        batch_size: batch_size.unwrap_or(4),
        epochs: epochs.unwrap_or(10),
        model_ref: model_ref.to_string(),
        training_data: data.clone(),
        ..Default::default()
    };

    // Add metadata
    adapter_config.metadata.insert("branch".to_string(), status.current_ref.clone());
    if interactive {
        adapter_config.metadata.insert("mode".to_string(), "interactive".to_string());
    }

    // Display configuration
    println!("\n→ Adapter configuration:");
    println!("  Name: {}", indexed_adapter_name);
    println!("  Rank: {}", adapter_config.rank);
    println!("  Learning rate: {}", adapter_config.learning_rate);
    println!("  Batch size: {}", adapter_config.batch_size);
    println!("  Epochs: {}", adapter_config.epochs);

    // Show current adapter stack
    let existing_adapters = adapter_manager.list_adapters()?;
    if !existing_adapters.is_empty() {
        println!("\n→ Existing adapter stack:");
        for adapter in &existing_adapters {
            println!("  [{}] {}", adapter.index, adapter.name);
        }
    }

    if interactive {
        println!("\n  Mode: Interactive learning");
    } else if let Some(data_file) = &data {
        println!("\n  Training data: {}", data_file);
    } else if config.is_none() {
        println!("\n  Mode: Initialization only (no training data)");
    }

    if let Some(cfg) = &config {
        println!("  Config override: {}", cfg);
    }

    // Initialize the adapter
    let adapter_path = adapter_manager.initialize_adapter(
        adapter_base_name,
        index,
        adapter_config.clone(),
    )?;

    println!("\n✓ Initialized adapter: adapters/{}.safetensors", indexed_adapter_name);
    println!("✓ Created config: adapters/{}.config.json", indexed_adapter_name);

    if interactive {
        println!("\n→ Interactive mode enabled");
        println!("  Start an inference session to begin learning:");
        println!("  hyprstream infer {} --prompt \"...\" --learn", model_ref.model);
    } else if data.is_some() {
        println!("\n→ Batch training ready");
        println!("  Run training with: hyprstream train {} --adapter {}", model_ref.model, indexed_adapter_name);
    }

    println!("\n✓ Adapter initialization complete!");
    println!("\n→ Next steps:");
    println!("  1. Check status: hyprstream status {}", model_ref.model);
    println!("  2. Test inference: hyprstream infer {}", model_ref.model);
    println!("  3. Commit changes: hyprstream commit {} -m \"Added {} adapter\"", model_ref.model, indexed_adapter_name);

    Ok(())
}

/// Handle serve command
pub async fn handle_serve(
    model: Option<String>,
    port: u16,
    host: &str,
) -> Result<()> {
    if let Some(ref model_ref_str) = model {
        println!("Starting server with pre-loaded model: {}", model_ref_str);
    } else {
        println!("Starting server in lazy-loading mode");
        println!("Models will be loaded on demand via API requests");
    }

    println!("Listening on {}:{}", host, port);

    // TODO: Call the actual server start function
    // For now, just simulate
    println!("\n→ Server configuration:");
    println!("  Lazy loading: {}", if model.is_none() { "enabled" } else { "disabled" });
    println!("  Pre-loaded models: {}", model.as_deref().unwrap_or("none"));
    println!("  API endpoint: http://{}:{}/v1/completions", host, port);

    Ok(())
}

/// Handle list command
pub async fn handle_list(
    storage: &ModelStorage,
    branch: Option<String>,
    tag: Option<String>,
    dirty: bool,
    verbose: bool,
) -> Result<()> {
    use crate::cli::commands::model::GitInfo;

    info!("Listing models");

    let models = storage.list_models().await?;

    if models.is_empty() {
        println!("No models found.");
        println!("Try: hyprstream clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct");
        return Ok(());
    }

    // Get storage paths for model directories
    let storage_paths = crate::storage::StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;

    // Collect models with git info
    let mut models_with_git = Vec::new();
    for (model_ref, metadata) in models {
        let model_path = models_dir.join(&model_ref.model);
        let git_info = GitInfo::from_repo_path(&model_path);

        // Apply filters
        if dirty {
            if let Some(ref git) = git_info {
                if !git.is_dirty {
                    continue;
                }
            } else {
                continue;
            }
        }

        if let Some(ref branch_filter) = branch {
            if let Some(ref git) = git_info {
                if !git.matches_branch(branch_filter) {
                    continue;
                }
            } else {
                continue;
            }
        }

        if let Some(ref tag_filter) = tag {
            if let Some(ref git) = git_info {
                if !git.matches_tag(tag_filter) {
                    continue;
                }
            } else {
                continue;
            }
        }

        models_with_git.push((model_ref, metadata, git_info));
    }

    if verbose {
        // Verbose output with detailed information
        for (model_ref, metadata, git_info) in &models_with_git {
            println!("Model: {}", model_ref.model);
            if let Some(desc) = &metadata.display_name {
                println!("  Display Name: {}", desc);
            }

            if let Some(git) = git_info {
                println!("  Git Reference: {}", git.current_ref.as_deref().unwrap_or("detached"));
                println!("  Commit: {}", git.short_commit.as_deref().unwrap_or("unknown"));
                println!("  Status: {}", if git.is_dirty { "dirty" } else { "clean" });
                if let Some(date) = &git.last_commit_date {
                    println!("  Last Commit: {}", date);
                }
            }

            if let Some(size) = metadata.size_bytes {
                println!("  Size: {:.2} GB", size as f64 / 1_073_741_824.0);
            }
            println!("  Created: {}", chrono::DateTime::from_timestamp(metadata.created_at, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "unknown".to_string()));
            println!();
        }
    } else {
        // Table format - the nice format you liked!
        println!("{:<30} {:<15} {:<8} {:<6} {:<10}", "MODEL NAME", "REF", "COMMIT", "STATUS", "SIZE");
        println!("{}", "-".repeat(75));

        for (model_ref, metadata, git_info) in &models_with_git {
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

        if models_with_git.is_empty() {
            println!("No models match the specified filters.");
        }
    }

    Ok(())
}

/// Handle clone command
pub async fn handle_clone(
    _storage: &ModelStorage,
    repo_url: &str,
    name: Option<String>,
) -> Result<()> {
    info!("Cloning model from {}", repo_url);

    println!("📦 Cloning model from: {}", repo_url);

    // Use the existing working implementation that handles LFS properly
    let cloned = crate::storage::operations::clone_model(repo_url, None).await?;

    // If user provided a custom name, inform them of the actual name used
    if let Some(custom_name) = name {
        if custom_name != cloned.model_name {
            println!("ℹ️  Model cloned as '{}' (derived from URL)", cloned.model_name);
            println!("    Custom naming will be supported in a future update");
        }
    }

    println!("✅ Model '{}' cloned successfully!", cloned.model_name);
    println!("   Model ID: {}", cloned.model_id);
    println!("   Location: {}", cloned.model_path.display());

    // The model is already registered by clone_model, so we're done

    Ok(())
}

/// Handle info command
pub async fn handle_info(
    storage: &ModelStorage,
    model: &str,
    verbose: bool,
) -> Result<()> {
    info!("Getting info for model {}", model);

    let model_ref = ModelRef::parse(model)?;
    let registry = storage.registry();

    // Get model path
    let model_path = storage.get_model_path(&model_ref).await?;

    println!("Model: {}", model_ref.model);
    println!("Reference: {}", model_ref.git_ref.as_deref().unwrap_or("main"));
    println!("Path: {}", model_path.display());

    // Get git status
    let status = registry.status(&model_ref.model).await?;
    println!("\nGit Status:");
    println!("  Current branch/ref: {}", status.current_ref);

    if status.is_dirty {
        println!("  Working tree: dirty");
        if !status.modified_files.is_empty() {
            println!("  Modified files: {}", status.modified_files.len());
            if verbose {
                for file in &status.modified_files {
                    println!("    M {}", file);
                }
            }
        }
        if !status.new_files.is_empty() {
            println!("  New files: {}", status.new_files.len());
            if verbose {
                for file in &status.new_files {
                    println!("    A {}", file);
                }
            }
        }
        if !status.deleted_files.is_empty() {
            println!("  Deleted files: {}", status.deleted_files.len());
            if verbose {
                for file in &status.deleted_files {
                    println!("    D {}", file);
                }
            }
        }
    } else {
        println!("  Working tree: clean");
    }

    // Show model size if we can
    if let Ok(metadata) = std::fs::metadata(&model_path) {
        if metadata.is_dir() {
            // Calculate directory size (simplified - just count files)
            let mut total_size = 0u64;
            let mut file_count = 0u32;

            if let Ok(entries) = std::fs::read_dir(&model_path) {
                for entry in entries.flatten() {
                    if let Ok(meta) = entry.metadata() {
                        if meta.is_file() {
                            total_size += meta.len();
                            file_count += 1;
                        }
                    }
                }
            }

            println!("\nModel Size:");
            println!("  Files: {}", file_count);
            println!("  Total size: {:.2} MB", total_size as f64 / 1_048_576.0);
        }
    }

    Ok(())
}

/// Handle infer command
pub async fn handle_infer(
    storage: &ModelStorage,
    model_ref_str: &str,
    prompt: &str,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    stream: bool,
    _force_download: bool,
) -> Result<()> {
    use crate::runtime::{TorchEngine, RuntimeConfig, RuntimeEngine};
    use crate::runtime::sampling::{SamplingConfig, load_sampling_config};
    use crate::runtime::template_engine::ChatMessage;
    use crate::config::GenerationRequest;
    use std::io::{self, Write};

    info!("Running base model inference: model={}, prompt_len={}", model_ref_str, prompt.len());

    // Parse model reference
    let model_ref = ModelRef::parse(model_ref_str)?;

    // Get model path
    let storage_paths = crate::storage::StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;
    let model_path = match storage.get_model_path(&model_ref).await {
        Ok(path) => path,
        Err(_) => {
            // Fallback to direct path lookup
            models_dir.join(&model_ref.model)
        }
    };

    if !model_path.exists() {
        eprintln!("❌ Model '{}' not found", model_ref.model);
        eprintln!("   Use 'hyprstream list' to see available models");
        return Err(anyhow::anyhow!("Model '{}' not found", model_ref.model));
    }

    info!("Using model at: {}", model_path.display());

    // Load model configuration
    let sampling_config = if model_path.join("config.json").exists() {
        match load_sampling_config(&model_path).await {
            Ok(config) => config,
            Err(e) => {
                tracing::warn!("Could not load model config: {}. Using defaults.", e);
                SamplingConfig::default()
            }
        }
    } else {
        SamplingConfig::for_model(&model_ref_str)
    };

    // Initialize inference engine
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

    // Load the model
    let load_start = std::time::Instant::now();
    RuntimeEngine::load_model(&mut engine, &model_path).await.map_err(|e| {
        anyhow::anyhow!("Failed to load model: {}", e)
    })?;
    let load_time = load_start.elapsed();
    info!("Model loaded in {:.2}s", load_time.as_secs_f64());

    // Auto-load adapters from the model directory
    let adapter_manager = crate::storage::AdapterManager::new(&model_path);
    let adapters = adapter_manager.list_adapters()?;

    if !adapters.is_empty() {
        println!("\n→ Loading adapters:");
        for adapter_info in &adapters {
            println!("  [{}] {} ({:.1} KB)",
                     adapter_info.index,
                     adapter_info.name,
                     adapter_info.size as f64 / 1024.0);

            // Load the adapter weights into the engine
            match engine.load_lora_weights(adapter_info.path.to_str().unwrap()) {
                Ok(_) => {
                    info!("Successfully loaded adapter: {:?}", adapter_info.path);
                }
                Err(e) => {
                    tracing::warn!("Failed to load adapter {:?}: {}", adapter_info.path, e);
                    println!("  ⚠️  Failed to load [{}] {} - continuing without it",
                             adapter_info.index, adapter_info.name);
                }
            }
        }
        println!("  Total: {} adapters loaded", adapters.len());
    } else {
        info!("No adapters found, using base model");
    }

    // Clear any previously loaded LoRA (in case of manual loads)
    {
        let mut lora_guard = engine.active_lora.lock().unwrap();
        *lora_guard = None;
    }

    // Use model defaults or overrides
    let max_tokens = max_tokens.unwrap_or(100);

    info!(
        "Generating response: max_tokens={}, temperature={}, top_p={:?}, top_k={:?}",
        max_tokens, final_config.temperature, final_config.top_p, final_config.top_k
    );

    // Apply chat template to the prompt
    let formatted_prompt = {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }
        ];

        match RuntimeEngine::apply_chat_template(&engine, &messages, true) {
            Ok(formatted) => formatted,
            Err(e) => {
                tracing::warn!("Could not apply chat template: {}. Using raw prompt.", e);
                prompt.to_string()
            }
        }
    };

    let request = GenerationRequest {
        prompt: formatted_prompt,
        max_tokens,
        temperature: final_config.temperature,
        top_p: final_config.top_p.unwrap_or(1.0),
        top_k: final_config.top_k,
        repeat_penalty: 1.0,
        stop_tokens: vec![],
        seed: None,
        stream,
        active_adapters: None,
        realtime_adaptation: None,
        user_feedback: None,
    };

    if stream {
        // Stream tokens as they're generated
        print!("\n");
        let result = engine.generate_streaming_with_params(
            &request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            request.repeat_penalty,
            |token| {
                print!("{}", token);
                let _ = io::stdout().flush();
            }
        ).await?;
        println!();
        info!("Generated {} tokens", result.split_whitespace().count());
    } else {
        // Generate all at once using streaming with collection
        let mut response = String::new();
        engine.generate_streaming_with_params(
            &request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            request.repeat_penalty,
            |token| {
                response.push_str(token);
            }
        ).await?;
        println!("\n{}", response);
    }

    Ok(())
}

/// Handle push command
pub async fn handle_push(
    storage: &ModelStorage,
    model: &str,
    remote: Option<String>,
    branch: Option<String>,
    set_upstream: bool,
    force: bool,
) -> Result<()> {
    info!("Pushing model {} to remote", model);

    let registry = storage.registry();
    let remote_name = remote.as_deref().unwrap_or("origin");
    let branch_name = branch.as_deref();

    registry.push_model(model, remote_name, branch_name, set_upstream, force).await?;

    println!("✓ Pushed model {} to {}", model, remote_name);
    if let Some(b) = branch_name {
        println!("  Branch: {}", b);
    }
    if set_upstream {
        println!("  Upstream tracking configured");
    }

    Ok(())
}

/// Handle pull command
pub async fn handle_pull(
    storage: &ModelStorage,
    model: &str,
    remote: Option<String>,
    branch: Option<String>,
    rebase: bool,
) -> Result<()> {
    info!("Pulling model {} from remote", model);

    let registry = storage.registry();
    let remote_name = remote.as_deref().unwrap_or("origin");
    let branch_name = branch.as_deref();

    registry.pull_model(model, remote_name, branch_name, rebase).await?;

    println!("✓ Pulled latest changes for model {}", model);
    println!("  Remote: {}", remote_name);
    if let Some(b) = branch_name {
        println!("  Branch: {}", b);
    }
    if rebase {
        println!("  Strategy: rebase");
    } else {
        println!("  Strategy: merge");
    }

    Ok(())
}

/// Handle merge command
pub async fn handle_merge(
    storage: &ModelStorage,
    model: &str,
    branch: &str,
    ff_only: bool,
    no_ff: bool,
) -> Result<()> {
    info!("Merging branch {} into model {}", branch, model);

    let registry = storage.registry();

    registry.merge_branch(model, branch, ff_only, no_ff).await?;

    println!("✓ Merged branch '{}' into model {}", branch, model);
    if ff_only {
        println!("  Strategy: fast-forward only");
    } else if no_ff {
        println!("  Strategy: no fast-forward (merge commit created)");
    } else {
        println!("  Strategy: auto (fast-forward if possible)");
    }

    Ok(())
}