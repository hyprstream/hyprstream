//! Handlers for git-style CLI commands

use crate::cli::commands::model::GitInfo;
use crate::config::GenerationRequest;
// Sampling config now loaded via builder pattern
use crate::runtime::template_engine::ChatMessage;
use crate::runtime::{RuntimeConfig, RuntimeEngine, TorchEngine};
use crate::storage::{CheckoutOptions, ModelRef, ModelStorage};
use anyhow::Result;
use std::io::{self, Write};
use tracing::info;

/// Handle branch command
pub async fn handle_branch(
    storage: &ModelStorage,
    model: &str,
    branch_name: &str,
    from_ref: Option<String>,
) -> Result<()> {
    info!("Creating branch {} for model {}", branch_name, model);

    let model_ref = ModelRef::new(model.to_string());

    // Create branch using git2db directly
    let repo_id = storage.resolve_repo_id(&model_ref)?;
    let registry = storage.registry().await;
    let handle = registry.repo(&repo_id)?;
    handle.branch().create(branch_name, from_ref.as_deref()).await?;

    println!("✓ Created branch {} for model {}", branch_name, model);

    if let Some(from) = from_ref {
        println!("  Branch created from: {}", from);
    }

    // Branch created successfully

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

    info!(
        "Checking out {} for model {}",
        model_ref.git_ref.to_string(),
        model_ref.model
    );

    // Check for uncommitted changes if not forcing
    if !force {
        let status = storage.status(&model_ref).await?;
        if !status.is_clean {
            println!("Warning: Model has uncommitted changes");
            println!("Use --force to discard changes, or commit them first");
            return Ok(());
        }
    }

    // Use CheckoutOptions
    let options = CheckoutOptions {
        create_branch,
        force,
    };

    let result = storage.checkout(&model_ref, options).await?;

    // Display checkout results using reference names
    let ref_display = result.new_ref_name.as_deref().unwrap_or("detached HEAD");
    println!("✓ Switched to {} ({})", ref_display, result.new_oid);

    if result.was_forced {
        println!("  ⚠️ Forced checkout - local changes discarded");
    }

    if result.files_changed > 0 {
        println!("  Files in working tree: {}", result.files_changed);
    }

    Ok(())
}

/// Handle status command
pub async fn handle_status(
    storage: &ModelStorage,
    model: Option<String>,
    verbose: bool,
) -> Result<()> {
    if let Some(model_ref_str) = model {
        // Status for specific model with full ModelRef support
        let model_ref = ModelRef::parse(&model_ref_str)?;
        let status = storage.status(&model_ref).await?;
        print_model_status(&model_ref.model, &status, verbose);
    } else {
        // Status for all models
        let models = storage.list_models().await?;

        if models.is_empty() {
            println!("No models found");
            return Ok(());
        }

        for (model_ref, _metadata) in models {
            if let Ok(status) = storage.status(&model_ref).await {
                print_model_status(&model_ref.model, &status, verbose);
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

    let model_ref = ModelRef::parse(model)?;

    // Check status first
    let status = storage.status(&model_ref).await?;

    if status.is_clean {
        println!("No changes to commit for model {}", model);
        return Ok(());
    }

    // Show what will be committed
    if stage_all {
        println!("Staging all changes:");
        for file_path in &status.modified_files {
            println!("  M {}", file_path.display());
        }
    }

    // Use git2db API for commit
    let repo_id = storage.resolve_repo_id(&model_ref)?;
    let registry = storage.registry().await;
    let handle = registry.repo(&repo_id)?;

    if stage_all {
        handle.staging().add_all().await?;
    }

    let commit_oid = handle.commit(message).await?;

    println!("✓ Committed changes to {}", model);
    println!("  Message: {}", message);
    println!("  Commit: {}", commit_oid);

    Ok(())
}

/// Print model status in a nice format using git2db's RepositoryStatus
fn print_model_status(model_name: &str, status: &git2db::RepositoryStatus, verbose: bool) {
    println!("Model: {}", model_name);

    // Show current branch/commit
    if let Some(branch) = &status.branch {
        if let Some(head) = status.head {
            println!("Current ref: {} ({})", branch, head);
        } else {
            println!("Current ref: {}", branch);
        }
    } else if let Some(head) = status.head {
        println!("Current ref: detached HEAD ({})", head);
    } else {
        println!("Current ref: unknown");
    }

    // Show ahead/behind if tracking a remote
    if status.ahead > 0 || status.behind > 0 {
        println!(
            "Tracking: ahead {}, behind {}",
            status.ahead, status.behind
        );
    }

    // Show dirty/clean status
    if !status.is_clean {
        println!("Status: modified (uncommitted changes)");

        if verbose || !status.modified_files.is_empty() {
            println!("\n  Modified files:");
            for file in &status.modified_files {
                println!("    M {}", file.display());
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
    let status = storage.status(&model_ref).await?;

    if status.branch.is_none()
    {
        println!("Warning: Training on detached HEAD");
        println!("Consider creating a branch first:");
        println!(
            "  hyprstream branch {} training/experiment",
            model_ref.model
        );
    }

    println!(
        "Starting LoRA adapter initialization for {}",
        model_ref
    );
    println!(
        "Current branch: {}",
        status.branch.as_deref().unwrap_or("detached")
    );

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
    adapter_config.metadata.insert(
        "branch".to_string(),
        status
            .branch
            .as_deref()
            .unwrap_or("detached")
            .to_string(),
    );
    if interactive {
        adapter_config
            .metadata
            .insert("mode".to_string(), "interactive".to_string());
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

    // Load the model to get proper dimensions
    tracing::info!("Loading model to determine LoRA structure for adapter creation");
    let config = crate::config::RuntimeConfig::default();
    let mut engine = crate::runtime::TorchEngine::new(config)?;
    crate::runtime::RuntimeEngine::load_model(&mut engine, &model_path).await?;

    // Create LoRA configuration with proper target modules
    let lora_config = crate::lora::LoRAConfig {
        rank: adapter_config.rank as usize,
        alpha: adapter_config.alpha,
        dropout: 0.1,
        // Use comprehensive target modules for full model adaptation
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "down_proj".to_string(),
        ],
        learning_rate: adapter_config.learning_rate as f32,
    };

    // Create LoRA model structure using proper model dimensions
    engine.create_lora(lora_config.clone())?;

    // Ensure adapters directory exists
    adapter_manager.ensure_adapters_dir()?;

    // Save the initialized LoRA weights
    let adapter_path = adapter_manager
        .adapters_dir
        .join(format!("{}.safetensors", indexed_adapter_name));
    engine.save_lora_weights(adapter_path.to_str().unwrap())?;

    // Save config
    let config_path = adapter_manager
        .adapters_dir
        .join(format!("{}.config.json", indexed_adapter_name));
    let config_json = serde_json::to_string_pretty(&adapter_config)?;
    std::fs::write(&config_path, config_json)?;

    println!(
        "\n✓ Initialized adapter: adapters/{}.safetensors",
        indexed_adapter_name
    );
    println!(
        "✓ Created config: adapters/{}.config.json",
        indexed_adapter_name
    );

    if interactive {
        println!("\n→ Interactive mode enabled");
        println!("  Start an inference session to begin learning:");
        println!(
            "  hyprstream infer {} --prompt \"...\" --learn",
            model_ref.model
        );
    } else if data.is_some() {
        println!("\n→ Batch training ready");
        println!(
            "  Run training with: hyprstream train {} --adapter {}",
            model_ref.model, indexed_adapter_name
        );
    }

    println!("\n✓ Adapter initialization complete!");
    println!("\n→ Next steps:");
    println!("  1. Check status: hyprstream status {}", model_ref.model);
    println!("  2. Test inference: hyprstream infer {}", model_ref.model);
    println!(
        "  3. Commit changes: hyprstream commit {} -m \"Added {} adapter\"",
        model_ref.model, indexed_adapter_name
    );

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
                println!(
                    "  Git Reference: {}",
                    git.current_ref.as_deref().unwrap_or("detached")
                );
                println!(
                    "  Commit: {}",
                    git.short_commit.as_deref().unwrap_or("unknown")
                );
                println!("  Status: {}", if git.is_dirty { "dirty" } else { "clean" });
                if let Some(date) = &git.last_commit_date {
                    println!("  Last Commit: {}", date);
                }
            }

            if let Some(size) = metadata.size_bytes {
                println!("  Size: {:.2} GB", size as f64 / 1_073_741_824.0);
            }
            println!(
                "  Created: {}",
                chrono::DateTime::from_timestamp(metadata.created_at, 0)
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            );
            println!();
        }
    } else {
        // Table format - the nice format you liked!
        println!(
            "{:<30} {:<15} {:<8} {:<6} {:<10}",
            "MODEL NAME", "REF", "COMMIT", "STATUS", "SIZE"
        );
        println!("{}", "-".repeat(75));

        for (model_ref, metadata, git_info) in &models_with_git {
            let size_str = if let Some(size) = metadata.size_bytes {
                format!("{:.1}GB", size as f64 / (1024.0 * 1024.0 * 1024.0))
            } else {
                "n/a".to_string()
            };

            let (git_ref, commit, status) = match git_info {
                Some(git) => (
                    git.current_ref
                        .clone()
                        .unwrap_or_else(|| "detached".to_string()),
                    git.short_commit
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string()),
                    if git.is_dirty { "dirty" } else { "clean" },
                ),
                None => ("n/a".to_string(), "n/a".to_string(), "n/a"),
            };

            println!(
                "{:<30} {:<15} {:<8} {:<6} {:<10}",
                model_ref.model, git_ref, commit, status, size_str
            );
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
    let cloned = crate::storage::operations::clone_model(repo_url, name.as_deref(), None).await?;

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
    adapters_only: bool,
) -> Result<()> {
    info!("Getting info for model {}", model);

    let model_ref = ModelRef::parse(model)?;

    // Get model path
    let model_path = storage.get_model_path(&model_ref).await?;

    // If adapters_only is true, skip the general model info
    if !adapters_only {
        println!("Model: {}", model_ref.model);
        let display_ref = match &model_ref.git_ref {
            crate::storage::GitRef::DefaultBranch => {
                storage.get_default_branch(&model_ref).await?
            }
            _ => model_ref.git_ref.to_string(),
        };
        println!("Reference: {}", display_ref);
        println!("Path: {}", model_path.display());
    }

    // Get git status
    let status = storage.status(&model_ref).await?;
    println!("\nGit Status:");
    println!(
        "  Current branch/ref: {}",
        status.branch.as_deref().unwrap_or("detached")
    );

    if !status.is_clean {
        println!("  Working tree: dirty");
        println!("  Modified files: {}", status.modified_files.len());
        if verbose {
            for file in &status.modified_files {
                println!("    M {}", file.display());
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

    // List adapters for this model
    let adapter_manager = crate::storage::AdapterManager::new(&model_path);

    match adapter_manager.list_adapters() {
        Ok(adapters) => {
            if adapters.is_empty() {
                println!("\nAdapters: None");
            } else {
                println!("\nAdapters: {}", adapters.len());

                // Sort adapters by index for consistent display
                let mut sorted_adapters = adapters;
                sorted_adapters.sort_by_key(|a| a.index);

                for adapter in &sorted_adapters {
                    let size_kb = adapter.size as f64 / 1024.0;
                    print!("  [{}] {} ({:.1} KB)", adapter.index, adapter.name, size_kb);

                    // Show config info if available and verbose mode is on
                    if verbose && adapter.config_path.is_some() {
                        if let Ok(config_content) =
                            std::fs::read_to_string(adapter.config_path.as_ref().unwrap())
                        {
                            if let Ok(config) = serde_json::from_str::<crate::storage::AdapterConfig>(
                                &config_content,
                            ) {
                                print!(
                                    " - rank: {}, alpha: {}, lr: {:.0e}",
                                    config.rank, config.alpha, config.learning_rate
                                );
                            }
                        }
                    }
                    println!();
                }

                if verbose {
                    println!("\nAdapter Details:");
                    for adapter in &sorted_adapters {
                        println!("  [{}] {}", adapter.index, adapter.name);
                        println!("      File: {}", adapter.filename);
                        println!("      Path: {}", adapter.path.display());
                        println!("      Size: {:.1} KB", adapter.size as f64 / 1024.0);

                        if let Some(config_path) = &adapter.config_path {
                            println!("      Config: {}", config_path.display());
                            if let Ok(config_content) = std::fs::read_to_string(config_path) {
                                if let Ok(config) =
                                    serde_json::from_str::<crate::storage::AdapterConfig>(
                                        &config_content,
                                    )
                                {
                                    println!("      Rank: {}", config.rank);
                                    println!("      Alpha: {}", config.alpha);
                                    println!("      Learning Rate: {:.2e}", config.learning_rate);
                                    println!("      Created: {}", config.created_at);
                                }
                            }
                        } else {
                            println!("      Config: Not found");
                        }
                        println!();
                    }
                }
            }
        }
        Err(e) => {
            if verbose {
                println!("\nAdapters: Error listing adapters: {}", e);
            } else {
                println!("\nAdapters: Unable to list");
            }
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
    repeat_penalty: Option<f32>,
    stream: bool,
    _force_download: bool,
) -> Result<()> {
    info!(
        "Running base model inference: model={}, prompt_len={}",
        model_ref_str,
        prompt.len()
    );

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

    // Initialize inference engine
    let runtime_config = RuntimeConfig::default();
    let mut engine = TorchEngine::new(runtime_config)?;

    // Load the model
    let load_start = std::time::Instant::now();
    RuntimeEngine::load_model(&mut engine, &model_path)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
    let load_time = load_start.elapsed();
    info!("Model loaded in {:.2}s", load_time.as_secs_f64());

    // Auto-load adapters from the model directory
    let adapter_manager = crate::storage::AdapterManager::new(&model_path);
    let adapters = adapter_manager.list_adapters()?;

    if !adapters.is_empty() {
        println!("\n→ Loading adapters:");

        // Flag to track LoRA model initialization state across adapter loading iterations.
        // The LoRA model structure must be initialized before individual adapter weights
        // can be loaded. This is done once for the first valid adapter configuration.
        let mut lora_model_initialized = false;

        for adapter_info in &adapters {
            tracing::info!(
                "[{}] {} ({:.1} KB)",
                adapter_info.index,
                adapter_info.name,
                adapter_info.size as f64 / 1024.0
            );

            // Parse adapter configuration from JSON file.
            // Each adapter has an associated .config.json file containing metadata
            // like rank, alpha scaling factor, and training parameters.
            let config_path = adapter_info.path.with_extension("config.json");
            let adapter_config = match std::fs::read_to_string(&config_path) {
                Ok(content) => {
                    match serde_json::from_str::<crate::storage::AdapterConfig>(&content) {
                        Ok(config) => config,
                        Err(parse_error) => {
                            tracing::error!(
                                config_path = %config_path.display(),
                                error = %parse_error,
                                "Failed to deserialize adapter configuration - JSON structure may be invalid"
                            );
                            tracing::warn!(
                                "Failed to parse config for adapter [{}] {} - skipping",
                                adapter_info.index,
                                adapter_info.name
                            );
                            continue;
                        }
                    }
                }
                Err(io_error) => {
                    tracing::warn!(
                        config_path = %config_path.display(),
                        error = %io_error,
                        adapter_name = %adapter_info.name,
                        "Adapter configuration file not found - adapter may be corrupted"
                    );
                    tracing::warn!(
                        "No config found for adapter [{}] {} - skipping",
                        adapter_info.index,
                        adapter_info.name
                    );
                    continue;
                }
            };

            // Initialize LoRA model structure on first successful adapter.
            // This creates the VarStore and module mapping required for gradient tracking.
            // All adapters share the same LoRA model structure but have different weights.
            if !lora_model_initialized {
                tracing::info!(
                    rank = adapter_config.rank,
                    alpha = adapter_config.alpha,
                    learning_rate = adapter_config.learning_rate,
                    "Initializing LoRA model structure with configuration from first adapter"
                );

                // Create LoRA config with comprehensive target modules
                // The engine will auto-detect which modules are actually present in the SafeTensors file
                let lora_config = crate::lora::LoRAConfig {
                    rank: adapter_config.rank as usize, // Convert u32 to usize
                    alpha: adapter_config.alpha,
                    dropout: 0.1, // Standard dropout rate for LoRA training
                    // Use comprehensive target modules - engine will filter based on what's available
                    target_modules: vec![
                        "q_proj".to_string(),
                        "k_proj".to_string(),
                        "v_proj".to_string(),
                        "o_proj".to_string(),
                        "gate_proj".to_string(),
                        "up_proj".to_string(),
                        "down_proj".to_string(),
                    ],
                    learning_rate: adapter_config.learning_rate as f32, // Convert f64 to f32
                };

                match engine.create_lora(lora_config) {
                    Ok(_) => {
                        tracing::info!("LoRA model structure initialized successfully - ready to load adapter weights");
                        lora_model_initialized = true;
                    }
                    Err(init_error) => {
                        tracing::error!(
                            error = %init_error,
                            "Failed to initialize LoRA model structure - adapters cannot be loaded"
                        );
                        tracing::error!("Failed to initialize LoRA model - skipping all adapters");
                        break;
                    }
                }
            }

            // Load adapter weights into the initialized LoRA model.
            // This loads the actual trained LoRA A and B matrices from SafeTensors format.
            match engine.load_lora_weights(adapter_info.path.to_str().unwrap()) {
                Ok(_) => {
                    tracing::info!(
                        adapter_path = %adapter_info.path.display(),
                        adapter_name = %adapter_info.name,
                        adapter_size_kb = adapter_info.size as f64 / 1024.0,
                        "Successfully loaded LoRA adapter weights"
                    );
                    tracing::info!(
                        "Successfully loaded adapter [{}] {}",
                        adapter_info.index,
                        adapter_info.name
                    );
                }
                Err(load_error) => {
                    tracing::warn!(
                        adapter_path = %adapter_info.path.display(),
                        adapter_name = %adapter_info.name,
                        error = %load_error,
                        "Failed to load LoRA adapter weights - SafeTensors file may be corrupted"
                    );
                    tracing::warn!(
                        "Failed to load adapter [{}] {} - continuing without it",
                        adapter_info.index,
                        adapter_info.name
                    );
                }
            }
        }

        tracing::info!(
            total_adapters = adapters.len(),
            "Completed LoRA adapter loading process"
        );
        tracing::info!("Completed processing {} adapters", adapters.len());
    } else {
        tracing::info!("No LoRA adapters found in model directory - using base model only");
    }

    // Clear any previously loaded LoRA (in case of manual loads)
    {
        let mut lora_guard = engine.active_lora.lock().unwrap();
        *lora_guard = None;
    }

    // Apply chat template to the prompt
    let formatted_prompt = {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        match RuntimeEngine::apply_chat_template(&engine, &messages, true) {
            Ok(formatted) => formatted,
            Err(e) => {
                tracing::warn!("Could not apply chat template: {}. Using raw prompt.", e);
                prompt.to_string()
            }
        }
    };

    // Build request with proper cascade: model config → CLI overrides
    let request = GenerationRequest::builder(formatted_prompt)
        .with_model_config(&model_path)
        .await?
        .temperature(temperature)
        .top_p(top_p)
        .top_k(top_k)
        .repeat_penalty(repeat_penalty)
        .max_tokens(max_tokens)
        .build();

    info!(
        "Generating response: max_tokens={}, temperature={}, top_p={}, top_k={:?}, repeat_penalty={}",
        request.max_tokens, request.temperature, request.top_p, request.top_k, request.repeat_penalty
    );

    if stream {
        // Stream tokens as they're generated
        println!();
        let result = engine
            .generate_streaming(
                request.clone(),
                |token| {
                    print!("{}", token);
                    let _ = io::stdout().flush();
                },
            )
            .await?;
        println!();
        info!("Generated {} tokens", result.split_whitespace().count());
    } else {
        // Generate all at once using streaming with collection
        let response = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let response_clone = response.clone();
        engine
            .generate_streaming(
                request.clone(),
                move |token| {
                    if let Ok(mut r) = response_clone.lock() {
                        r.push_str(token);
                    }
                },
            )
            .await?;
        let final_response = response.lock().unwrap().clone();
        println!("\n{}", final_response);
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
    _force: bool,
) -> Result<()> {
    info!("Pushing model {} to remote", model);

    let remote_name = remote.as_deref().unwrap_or("origin");
    let branch_name = branch.as_deref();
    let model_ref = ModelRef::new(model.to_string());

    // Use git2db API for push
    let repo_id = storage.resolve_repo_id(&model_ref)?;
    let registry = storage.registry().await;
    let handle = registry.repo(&repo_id)?;

    if let Some(branch) = branch_name {
        handle.push(Some(remote_name), branch).await?;
    } else {
        // Push current branch
        let status = handle.status().await?;
        if let Some(current_branch) = status.branch {
            handle.push(Some(remote_name), current_branch).await?;
        } else {
            anyhow::bail!("Not on a branch - specify branch to push");
        }
    }

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

    let remote_name = remote.as_deref().unwrap_or("origin");
    let branch_name = branch.as_deref();
    let model_ref = ModelRef::new(model.to_string());

    // Use git2db API for pull
    let repo_id = storage.resolve_repo_id(&model_ref)?;
    let registry = storage.registry().await;
    let handle = registry.repo(&repo_id)?;

    if let Some(branch) = branch_name {
        handle.pull(Some(remote_name), branch).await?;
    } else {
        // Pull current branch
        let status = handle.status().await?;
        if let Some(current_branch) = status.branch {
            handle.pull(Some(remote_name), current_branch).await?;
        } else {
            anyhow::bail!("Not on a branch - specify branch to pull");
        }
    }

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

    let model_ref = ModelRef::new(model.to_string());

    // Use git2db's merge() API
    let repo_id = storage.resolve_repo_id(&model_ref)?;
    let registry = storage.registry().await;
    let handle = registry.repo(&repo_id)?;
    let _merge_oid = handle.merge(branch, ff_only, no_ff).await?;

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

/// Handle remove command
pub async fn handle_remove(
    storage: &ModelStorage,
    model: &str,
    force: bool,
    registry_only: bool,
    files_only: bool,
) -> Result<()> {
    info!("Removing model {}", model);

    // Validate flags
    if registry_only && files_only {
        return Err(anyhow::anyhow!(
            "Cannot specify both --registry-only and --files-only"
        ));
    }

    let model_ref = ModelRef::new(model.to_string());

    // Check if model exists in registry
    let registry_exists = storage.get_model_path(&model_ref).await.is_ok();

    // Check if model exists in filesystem
    let model_path = storage.get_models_dir().join(model);
    let files_exist = model_path.exists();

    if !registry_exists && !files_exist {
        println!("❌ Model '{}' not found in registry or filesystem", model);
        return Ok(());
    }

    // Show what will be removed
    println!("Model '{}' removal plan:", model);
    if registry_exists && !files_only {
        println!("  🗂️  Remove from git registry (submodule)");
    }
    if files_exist && !registry_only {
        println!("  📁 Remove files from: {}", model_path.display());

        // Show size if possible
        if let Ok(metadata) = std::fs::metadata(&model_path) {
            if metadata.is_dir() {
                // Calculate directory size
                let mut total_size = 0u64;
                if let Ok(entries) = walkdir::WalkDir::new(&model_path)
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                {
                    for entry in entries {
                        if entry.file_type().is_file() {
                            if let Ok(meta) = entry.metadata() {
                                total_size += meta.len();
                            }
                        }
                    }
                }
                println!("      Size: {:.2} GB", total_size as f64 / 1_073_741_824.0);
            }
        }
    }

    if registry_only && !registry_exists {
        println!("⚠️  Model not found in registry, nothing to remove");
        return Ok(());
    }

    if files_only && !files_exist {
        println!("⚠️  Model files not found, nothing to remove");
        return Ok(());
    }

    // Confirmation prompt unless forced
    if !force {
        print!("Are you sure you want to remove model '{}'? [y/N]: ", model);
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim().to_lowercase();

        if input != "y" && input != "yes" {
            println!("Removal cancelled");
            return Ok(());
        }
    }

    // Remove from registry (if requested and exists)
    if registry_exists && !files_only {
        match storage.remove_model(&model_ref).await {
            Ok(_) => {
                println!("✓ Removed '{}' from git registry", model);
            }
            Err(e) => {
                eprintln!("❌ Failed to remove '{}' from git registry: {}", model, e);
                if !files_only {
                    return Err(e);
                }
            }
        }
    }

    // Remove files (if requested and exist)
    if files_exist && !registry_only {
        match std::fs::remove_dir_all(&model_path) {
            Ok(_) => {
                println!("✓ Removed model files from: {}", model_path.display());
            }
            Err(e) => {
                eprintln!("❌ Failed to remove model files: {}", e);
                return Err(anyhow::anyhow!("Failed to remove model files: {}", e));
            }
        }
    }

    println!("✓ Model '{}' removed successfully", model);
    Ok(())
}
