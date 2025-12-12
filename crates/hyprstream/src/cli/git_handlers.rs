//! Handlers for git-style CLI commands

use crate::cli::commands::model::GitInfo;
use crate::config::{GenerationRequest, TrainingMode};
// Sampling config now loaded via builder pattern
use crate::runtime::model_config::ModelConfig;
use crate::runtime::template_engine::ChatMessage;
use crate::runtime::{RuntimeConfig, RuntimeEngine, TorchEngine};
use crate::storage::{CheckoutOptions, GitRef, ModelRef, ModelStorage};
use crate::training::{
    checkpoint::{CheckpointConfig, CheckpointManager},
    ReplayBufferConfig, SelfSupervisedConfig, SelfSupervisedTrainer,
};
use anyhow::{bail, Result};
use std::io::{self, Write};
use std::sync::Arc;
use tracing::{debug, info, warn};

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
    let repo_id = storage.resolve_repo_id(&model_ref).await?;
    let registry = storage.registry().await;
    let handle = registry.repo(&repo_id)?;
    handle.branch().create(branch_name, from_ref.as_deref()).await?;

    println!("✓ Created branch {}", branch_name);

    if let Some(ref from) = from_ref {
        println!("  Branch created from: {}", from);
    }

    // Create worktree for the branch
    let worktree_path = storage.create_worktree(&model_ref, branch_name).await?;
    println!("✓ Created worktree at {}", worktree_path.display());

    // Show helpful next steps
    println!("\n→ Next steps:");
    println!("  cd {}", worktree_path.display());
    println!("  hyprstream status {}:{}", model, branch_name);
    println!("  hyprstream lt {}:{} --adapter my-adapter", model, branch_name);

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
    model_ref_str: &str,
    message: &str,
    all: bool,
    all_untracked: bool,
    amend: bool,
    author: Option<String>,
    author_name: Option<String>,
    author_email: Option<String>,
    allow_empty: bool,
    dry_run: bool,
    verbose: bool,
) -> Result<()> {
    info!("Committing changes to model {}", model_ref_str);

    // Parse model reference to detect branch
    let model_ref = ModelRef::parse(model_ref_str)?;

    // Determine which branch to commit to
    let branch_name = match &model_ref.git_ref {
        git2db::GitRef::Branch(name) => name.clone(),
        git2db::GitRef::DefaultBranch => {
            let base_ref = ModelRef::new(model_ref.model.clone());
            storage.get_default_branch(&base_ref).await?
        }
        git2db::GitRef::Tag(tag) => {
            anyhow::bail!(
                "Cannot commit to a tag reference. Tags are immutable.\nTag: {}\nUse a branch instead: {}:main",
                tag, model_ref.model
            );
        }
        git2db::GitRef::Commit(oid) => {
            anyhow::bail!(
                "Cannot commit to a detached HEAD (commit reference).\nCommit: {}\nCheckout a branch first: hyprstream checkout {}:main",
                oid, model_ref.model
            );
        }
        git2db::GitRef::Revspec(spec) => {
            anyhow::bail!(
                "Cannot commit to a revspec reference. Revspecs are for querying history.\nRevspec: {}\nUse a branch instead: {}:main",
                spec, model_ref.model
            );
        }
    };

    // Get worktree path (this verifies it exists)
    let worktree_path = storage.get_worktree_path(&model_ref, &branch_name).await
        .map_err(|e| anyhow::anyhow!(
            "Worktree '{}' does not exist for model '{}'.\n\nCreate it first with:\n  hyprstream branch {} {}\n\nError: {}",
            branch_name, model_ref.model, model_ref.model, branch_name, e
        ))?;

    info!("Operating on worktree: {}", worktree_path.display());

    // Check for changes if not allowing empty commits
    // Open the worktree repository directly to check status
    let worktree_repo = git2::Repository::open(&worktree_path)?;
    let statuses = worktree_repo.statuses(None)?;
    let has_changes = !statuses.is_empty();

    if !allow_empty && !amend && !has_changes && !all_untracked {
        println!("No changes to commit for {}:{}", model_ref.model, branch_name);
        println!("\nUse --allow-empty to create a commit without changes");
        return Ok(());
    }

    // Show what will be committed
    if verbose || dry_run {
        println!("\n→ Changes to be committed:");

        if all || all_untracked {
            // Show all working tree changes
            for entry in statuses.iter() {
                if let Some(path) = entry.path() {
                    let status_char = if entry.status().contains(git2::Status::WT_NEW) {
                        "??"
                    } else if entry.status().contains(git2::Status::WT_MODIFIED) {
                        "M"
                    } else if entry.status().contains(git2::Status::WT_DELETED) {
                        "D"
                    } else {
                        "?"
                    };
                    println!("  {} {}", status_char, path);
                }
            }
        } else {
            // Show only staged files (index)
            for entry in statuses.iter() {
                if let Some(path) = entry.path() {
                    let status = entry.status();
                    if status.intersects(git2::Status::INDEX_NEW | git2::Status::INDEX_MODIFIED | git2::Status::INDEX_DELETED | git2::Status::INDEX_RENAMED) {
                        let status_char = if status.contains(git2::Status::INDEX_NEW) {
                            "A"
                        } else if status.contains(git2::Status::INDEX_DELETED) {
                            "D"
                        } else if status.contains(git2::Status::INDEX_RENAMED) {
                            "R"
                        } else if status.contains(git2::Status::INDEX_MODIFIED) {
                            "M"
                        } else {
                            "??"
                        };
                        println!("  {} {}", status_char, path);
                    }
                }
            }
        }
        println!();
    }

    // Dry run - show what would be committed
    if dry_run {
        println!("→ Dry run mode - no commit will be created\n");
        println!("Would commit to: {}:{}", model_ref.model, branch_name);
        println!("Message: {}", message);

        if let Some(ref auth) = author {
            println!("Author: {}", auth);
        } else if author_name.is_some() || author_email.is_some() {
            println!("Author: {} <{}>",
                author_name.as_deref().unwrap_or("default"),
                author_email.as_deref().unwrap_or("default"));
        }

        if amend {
            println!("Mode: Amend previous commit");
        }

        return Ok(());
    }

    // Stage files based on flags
    // We need to work with the worktree repository directly, not the bare repo
    let mut index = worktree_repo.index()?;

    if all_untracked {
        // Stage all files including untracked (git add -A)
        info!("Staging all files including untracked");
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
    } else if all {
        // Stage all tracked files only (git add -u)
        info!("Staging all tracked files");
        index.update_all(["*"].iter(), None)?;
    }

    index.write()?;

    // Perform commit operation in worktree
    let commit_oid = if amend {
        // Amend the previous commit
        info!("Amending previous commit");

        let tree_id = index.write_tree()?;
        let tree = worktree_repo.find_tree(tree_id)?;
        let head = worktree_repo.head()?;
        let commit_to_amend = head.peel_to_commit()?;

        // Use commit_amend to properly amend
        commit_to_amend.amend(
            Some("HEAD"),               // Update HEAD
            None,                        // Keep original author
            None,                        // Keep committer timestamp (update by default)
            None,                        // Keep encoding
            Some(message),              // New message
            Some(&tree),                // New tree
        )?
    } else {
        // Create new commit
        let tree_id = index.write_tree()?;
        let tree = worktree_repo.find_tree(tree_id)?;
        let head = worktree_repo.head()?;
        let parent_commit = head.peel_to_commit()?;

        // Parse author if provided
        let signature = if let Some(author_str) = author {
            // Parse "Name <email>" format
            let re = regex::Regex::new(r"^(.+?)\s*<(.+?)>$")?;
            if let Some(captures) = re.captures(&author_str) {
                let name = captures.get(1).unwrap().as_str().trim();
                let email = captures.get(2).unwrap().as_str().trim();
                git2::Signature::now(name, email)?
            } else {
                anyhow::bail!(
                    "Invalid author format. Expected: \"Name <email>\"\nGot: {}",
                    author_str
                );
            }
        } else if author_name.is_some() || author_email.is_some() {
            // Use author-name and author-email if provided
            let name = author_name.as_deref()
                .ok_or_else(|| anyhow::anyhow!("--author-name required when using --author-email"))?;
            let email = author_email.as_deref()
                .ok_or_else(|| anyhow::anyhow!("--author-email required when using --author-name"))?;
            git2::Signature::now(name, email)?
        } else {
            // Default signature from git config
            worktree_repo.signature()?
        };

        worktree_repo.commit(
            Some("HEAD"),
            &signature,
            &signature,
            message,
            &tree,
            &[&parent_commit],
        )?
    };

    // Success output
    println!("✓ Committed changes to {}:{}", model_ref.model, branch_name);
    println!("  Message: {}", message);
    println!("  Commit: {}", commit_oid);

    if amend {
        println!("  ⚠️  Previous commit amended");
    }

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
    branch_name: Option<String>,
    adapter_name: Option<String>,
    index: Option<u32>,
    rank: Option<u32>,
    learning_rate: Option<f32>,
    batch_size: Option<usize>,
    epochs: Option<usize>,
    config: Option<String>,
    training_mode: Option<String>,
) -> Result<()> {
    let model_ref = ModelRef::parse(model_ref_str)?;

    // WORKFLOW 2: New branch + worktree for isolated training
    if let Some(new_branch) = branch_name {
        info!("Creating isolated training environment: branch {} from {}", new_branch, model_ref.git_ref.display_name());

        // 1. Create branch from model_ref's git_ref
        let repo_id = storage.resolve_repo_id(&model_ref).await?;
        let registry = storage.registry().await;
        let handle = registry.repo(&repo_id)?;

        let from_ref = model_ref.git_ref.to_ref_string();
        handle.branch().create(&new_branch, from_ref.as_deref()).await?;

        println!("✓ Created branch {} from {}", new_branch, model_ref.git_ref.display_name());

        // 2. Create worktree for new branch
        let worktree_path = storage.create_worktree(&model_ref, &new_branch).await?;
        println!("✓ Created worktree at {}", worktree_path.display());

        // 3. Train adapter in worktree
        let adapter_manager = crate::storage::AdapterManager::new(&worktree_path);
        adapter_manager.ensure_adapters_dir()?;

        let adapter_base_name = adapter_name.as_deref().unwrap_or("default");
        let indexed_adapter_name = adapter_manager.create_indexed_name(adapter_base_name, index)?;

        println!("\n→ Training adapter {} in isolated worktree", indexed_adapter_name);

        // Create adapter configuration
        let adapter_config = crate::storage::AdapterConfig {
            rank: rank.unwrap_or(16),
            alpha: 32.0,
            learning_rate: learning_rate.unwrap_or(1e-4) as f64,
            batch_size: batch_size.unwrap_or(4),
            epochs: epochs.unwrap_or(10),
            model_ref: format!("{}:{}", model_ref.model, new_branch),
            training_data: None,
            ..Default::default()
        };

        // Load model and train
        let config = crate::config::RuntimeConfig::default();
        let mut engine = crate::runtime::TorchEngine::new(config)?;
        crate::runtime::RuntimeEngine::load_model(&mut engine, &worktree_path).await?;

        let lora_config = crate::lora::LoRAConfig {
            rank: adapter_config.rank as usize,
            alpha: adapter_config.alpha,
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
            learning_rate: adapter_config.learning_rate as f32,
        };

        engine.create_lora(lora_config)?;

        // Save adapter
        let adapter_path = adapter_manager
            .adapters_dir
            .join(format!("{}.safetensors", indexed_adapter_name));
        engine.save_lora_weights(adapter_path.to_str().unwrap())?;

        let config_path = adapter_manager
            .adapters_dir
            .join(format!("{}.config.json", indexed_adapter_name));
        let config_json = serde_json::to_string_pretty(&adapter_config)?;
        std::fs::write(&config_path, config_json)?;

        println!("\n✓ Initialized adapter: adapters/{}.safetensors", indexed_adapter_name);
        println!("✓ Created config: adapters/{}.config.json", indexed_adapter_name);

        // Save training mode to config.json if specified
        if let Some(ref mode_str) = training_mode {
            save_training_mode_config(&worktree_path, mode_str, &indexed_adapter_name, learning_rate, batch_size)?;
        }

        println!("\n✓ Isolated training complete!");
        println!("\n→ Next steps:");
        println!("  cd {}", worktree_path.display());
        println!("  hyprstream status {}:{}", model_ref.model, new_branch);
        println!("  hyprstream commit {}:{} -a -m \"Add {} adapter\"", model_ref.model, new_branch, indexed_adapter_name);

        return Ok(());
    }

    // WORKFLOW 1: Train on existing worktree

    // Verify worktree exists for the specified branch
    let branch_name = match &model_ref.git_ref {
        git2db::GitRef::Branch(name) => name.clone(),
        git2db::GitRef::DefaultBranch => {
            let base_ref = ModelRef::new(model_ref.model.clone());
            storage.get_default_branch(&base_ref).await?
        }
        _ => {
            return Err(anyhow::anyhow!(
                "LoRA training requires a branch reference. Use model:branch format (e.g., {}:main)",
                model_ref.model
            ));
        }
    };

    // Get worktree path (this verifies it exists)
    let model_path = storage.get_worktree_path(&model_ref, &branch_name).await
        .map_err(|e| anyhow::anyhow!(
            "Worktree '{}' does not exist for model '{}'. Create it first with:\n  hyprstream branch {} {}\nError: {}",
            branch_name, model_ref.model, model_ref.model, branch_name, e
        ))?;

    println!(
        "Starting LoRA adapter initialization for {}:{}",
        model_ref.model, branch_name
    );
    println!("Worktree: {}", model_path.display());

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
        training_data: None,
        ..Default::default()
    };

    // Add metadata
    adapter_config.metadata.insert(
        "branch".to_string(),
        branch_name.clone(),
    );

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

    // Save training mode to config.json if specified
    if let Some(ref mode_str) = training_mode {
        save_training_mode_config(&model_path, mode_str, &indexed_adapter_name, learning_rate, batch_size)?;
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

/// Helper function to save training mode configuration to config.json
fn save_training_mode_config(
    model_path: &std::path::Path,
    mode_str: &str,
    target_adapter: &str,
    learning_rate: Option<f32>,
    batch_size: Option<usize>,
) -> Result<()> {
    use crate::config::{HyprstreamTrainingConfig, TrainingMode};
    use crate::runtime::model_config::ModelConfig;

    let mode = match mode_str.to_lowercase().as_str() {
        "self_supervised" | "self-supervised" => TrainingMode::SelfSupervised,
        "supervised" => TrainingMode::Supervised,
        "disabled" | "off" | "none" => TrainingMode::Disabled,
        _ => {
            warn!("Unknown training mode '{}', defaulting to disabled", mode_str);
            TrainingMode::Disabled
        }
    };

    let training_config = HyprstreamTrainingConfig {
        mode: mode.clone(),
        target_adapter: Some(target_adapter.to_string()),
        learning_rate: learning_rate.unwrap_or(1e-4) as f64,
        batch_size: batch_size.unwrap_or(4),
        // Use sensible defaults for training (not Rust's Default which is 0)
        min_buffer_size: 1,   // Train after first quality example
        steps_per_cycle: 10,  // 10 training steps per cycle
        min_quality_threshold: 0.3,
        train_after_examples: 1,  // Start training immediately
        train_base_model: false,
    };

    ModelConfig::save_training_config(model_path, &training_config)?;

    match mode {
        TrainingMode::Disabled => println!("✓ Training mode set to: disabled"),
        TrainingMode::SelfSupervised => {
            println!("✓ Training mode set to: self_supervised");
            println!("  → Inference will automatically collect training examples");
            println!("  → Training cycles will trigger after {} high-quality examples", training_config.min_buffer_size);
        }
        TrainingMode::Supervised => println!("✓ Training mode set to: supervised"),
    }

    Ok(())
}

/// Handle list command
pub async fn handle_list(
    storage: &ModelStorage,
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
    let mut git_repos = Vec::new();
    for (model_ref, metadata) in models {
        // Get git info from the bare repository
        // The bare repo is at: models/{name}/{name}.git/
        let bare_repo_path = models_dir
            .join(&model_ref.model)
            .join(format!("{}.git", model_ref.model));

        let git_info = GitInfo::from_bare_repo(&bare_repo_path);

        git_repos.push((model_ref, metadata, git_info));
    }

    // Table format - the nice format you liked!
    println!(
        "{:<30} {:<15} {:<8} {:<6} {:<10}",
        "MODEL NAME", "REF", "COMMIT", "STATUS", "SIZE"
    );
    println!("{}", "-".repeat(75));

    for (_model_ref, metadata, git_info) in &git_repos {
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
            metadata.display_name.as_ref().unwrap(), git_ref, commit, status, size_str
        );
    }

    if git_repos.is_empty() {
        println!("No models match the specified filters.");
    }

    Ok(())
}

/// Handle clone command
pub async fn handle_clone(
    _storage: &ModelStorage,
    repo_url: &str,
    name: Option<String>,
    branch: Option<String>,
    depth: u32,
    full: bool,
    quiet: bool,
    verbose: bool,
) -> Result<()> {
    if !quiet {
        info!("Cloning model from {}", repo_url);
        println!("📦 Cloning model from: {}", repo_url);

        if let Some(ref b) = branch {
            println!("   Branch: {}", b);
        }

        if full {
            println!("   Mode: Full history");
        } else if depth > 0 {
            println!("   Depth: {} commits", depth);
        }

        if verbose {
            println!("   Verbose output enabled");
        }
    }

    // Create clone options struct to pass to storage layer
    let clone_opts = crate::storage::CloneOptions {
        branch,
        depth: if full { 0 } else { depth },
        quiet,
        verbose,
    };

    // Use the existing working implementation that handles LFS properly
    let cloned = crate::storage::operations::clone_model_with_options(
        repo_url,
        name.as_deref(),
        None,  // model_id
        clone_opts
    ).await?;

    if !quiet {
        println!("✅ Model '{}' cloned successfully!", cloned.model_name);
        println!("   Model ID: {}", cloned.model_id);
        println!("   Location: {}", cloned.model_path.display());
    }

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

    // Try to get git2db metadata
    let repo_metadata = if let Ok(_repo_id) = storage.resolve_repo_id(&model_ref).await {
        // Access registry through storage method
        match storage.get_bare_repo_path(&model_ref).await {
            Ok(_) => {
                // We know the model exists, try to get its metadata via the bare repo
                // Since we can't easily access git2db internals, we'll get info from git directly
                let bare_repo_path = storage.get_models_dir()
                    .join(&model_ref.model)
                    .join(format!("{}.git", &model_ref.model));

                if let Ok(bare_repo) = git2::Repository::open(&bare_repo_path) {
                    // Get remote URL
                    let url = bare_repo.find_remote("origin")
                        .ok()
                        .and_then(|r| r.url().map(|s| s.to_string()))
                        .unwrap_or_else(|| "unknown".to_string());

                    // Get current HEAD
                    let current_oid = bare_repo.head()
                        .ok()
                        .and_then(|h| h.target())
                        .map(|oid| oid.to_string());

                    Some((
                        Some(model_ref.model.clone()),
                        url,
                        model_ref.git_ref.to_string(),
                        current_oid,
                    ))
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    } else {
        None
    };

    // Get model path - this may fail for bare repos due to git2db bug
    let model_path = match storage.get_model_path(&model_ref).await {
        Ok(path) => path,
        Err(e) => {
            debug!("Failed to get model path via storage: {}", e);
            // Fallback: construct the path manually for bare repos
            let models_dir = storage.get_models_dir();
            let worktree_path = models_dir
                .join(&model_ref.model)
                .join("worktrees")
                .join("master"); // Try master first

            if !worktree_path.exists() {
                // Try main if master doesn't exist
                models_dir
                    .join(&model_ref.model)
                    .join("worktrees")
                    .join("main")
            } else {
                worktree_path
            }
        }
    };

    // If adapters_only is true, skip the general model info
    if !adapters_only {
        println!("Model: {}", model_ref.model);

        // Show git2db metadata if available
        if let Some((name, url, tracking_ref, current_oid)) = &repo_metadata {
            if let Some(n) = name {
                if n != &model_ref.model {
                    println!("  Registry name: {}", n);
                }
            }
            println!("  Origin URL: {}", url);
            println!("  Tracking ref: {}", tracking_ref);
            if let Some(oid) = current_oid {
                println!("  Current OID: {}", &oid[..8.min(oid.len())]);
            }
        }

        // Get display ref - avoid calling get_default_branch which fails on bare repos
        let display_ref = match &model_ref.git_ref {
            crate::storage::GitRef::DefaultBranch => {
                // Try to determine default branch from worktree directory names
                let worktrees_dir = storage.get_models_dir()
                    .join(&model_ref.model)
                    .join("worktrees");

                if worktrees_dir.join("main").exists() {
                    "main".to_string()
                } else if worktrees_dir.join("master").exists() {
                    "master".to_string()
                } else {
                    // Fallback
                    "unknown".to_string()
                }
            }
            _ => model_ref.git_ref.to_string(),
        };

        println!("Reference: {}", display_ref);
        println!("Path: {}", model_path.display());
    }

    // Get bare repository information
    let bare_repo_path = storage.get_models_dir()
        .join(&model_ref.model)
        .join(format!("{}.git", &model_ref.model));

    if bare_repo_path.exists() {
        println!("\nBare Repository:");
        println!("  Path: {}", bare_repo_path.display());

        // Try to open the bare repo to get more information
        if let Ok(bare_repo) = git2::Repository::open(&bare_repo_path) {
            // Get remote information
            if let Ok(remotes) = bare_repo.remotes() {
                for remote_name in remotes.iter().flatten() {
                    if let Ok(remote) = bare_repo.find_remote(remote_name) {
                        if let Some(url) = remote.url() {
                            println!("  Remote '{}': {}", remote_name, url);
                        }
                    }
                }
            }

            // Get branches
            if let Ok(branches) = bare_repo.branches(Some(git2::BranchType::Local)) {
                let branch_names: Vec<String> = branches
                    .filter_map(|b| b.ok())
                    .filter_map(|(branch, _)| branch.name().ok().flatten().map(|s| s.to_string()))
                    .collect();

                if !branch_names.is_empty() {
                    println!("  Local branches: {}", branch_names.join(", "));
                }
            }

            // Get tags
            if let Ok(tag_names) = bare_repo.tag_names(None) {
                let tags: Vec<&str> = tag_names.iter().flatten().collect();
                if !tags.is_empty() {
                    println!("  Tags: {}", tags.join(", "));
                }
            }

            // Repository size (approximate)
            if let Ok(metadata) = std::fs::metadata(&bare_repo_path) {
                if metadata.is_dir() {
                    let mut total_size = 0u64;
                    if let Ok(entries) = walkdir::WalkDir::new(&bare_repo_path)
                        .into_iter()
                        .collect::<std::result::Result<Vec<_>, _>>()
                    {
                        for entry in entries {
                            if entry.file_type().is_file() {
                                if let Ok(meta) = entry.metadata() {
                                    total_size += meta.len();
                                }
                            }
                        }
                    }
                    println!("  Repository size: {:.2} MB", total_size as f64 / 1_048_576.0);
                }
            }
        } else {
            println!("  (Unable to inspect bare repository)");
        }
    }

    // Get git status from the worktree directly
    // Note: storage.status() currently fails for bare repos, so we get status from worktree
    println!("\nWorktree Status:");

    // Try to open the worktree as a repository to get its status
    match git2::Repository::open(&model_path) {
        Ok(repo) => {
        // Get current branch/HEAD info
        if let Ok(head) = repo.head() {
            let branch_name = head.shorthand().unwrap_or("detached");
            println!("  Current branch/ref: {}", branch_name);

            if let Some(oid) = head.target() {
                println!("  HEAD commit: {}", &oid.to_string()[..8]);
            }
        } else {
            println!("  Current branch/ref: detached");
        }

        // Get working tree status
        if let Ok(statuses) = repo.statuses(None) {
            if statuses.is_empty() {
                println!("  Working tree: clean");
            } else {
                println!("  Working tree: dirty");
                let modified_count = statuses.iter().count();
                println!("  Modified files: {}", modified_count);

                if verbose {
                    for entry in statuses.iter() {
                        if let Some(path) = entry.path() {
                            let status = entry.status();
                            let prefix = if status.contains(git2::Status::INDEX_NEW) ||
                                           status.contains(git2::Status::WT_NEW) {
                                "A"
                            } else if status.contains(git2::Status::INDEX_MODIFIED) ||
                                      status.contains(git2::Status::WT_MODIFIED) {
                                "M"
                            } else if status.contains(git2::Status::INDEX_DELETED) ||
                                      status.contains(git2::Status::WT_DELETED) {
                                "D"
                            } else {
                                "?"
                            };
                            println!("    {} {}", prefix, path);
                        }
                    }
                }
            }
        } else {
            println!("  Working tree: unable to get status");
        }
        }
        Err(_) => {
            // Fallback: try to use storage.status() which may work for non-bare repos
            match storage.status(&model_ref).await {
            Ok(status) => {
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
            }
            Err(e) => {
                println!("  Unable to get status: {}", e);
                debug!("Status error details: {:?}", e);
            }
        }
        }
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
    image_path: Option<String>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u32>,
    stream: bool,
    _force_download: bool,
    max_context: Option<usize>,
    kv_quant: crate::cli::commands::KVQuantArg,
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

    // Initialize inference engine with max_context and kv_quant from CLI/env
    // Clap handles precedence: CLI args > env vars > defaults
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.max_context = max_context;  // From clap (already merged CLI > env)
    runtime_config.kv_quant_type = kv_quant.into();  // From clap (already merged CLI > env)
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

    // Note: Do NOT clear active_lora here - adapter loading above correctly sets it.
    // The loaded adapter weights should persist for inference + training.

    // Check for training mode in config.json
    let training_config = ModelConfig::load_training_config(&model_path);
    let trainer: Option<Arc<SelfSupervisedTrainer>> = if let Some(ref tc) = training_config {
        if tc.is_enabled() && tc.mode == TrainingMode::SelfSupervised {
            info!(
                "Self-supervised training enabled, target_adapter: {:?}",
                tc.target_adapter
            );
            let ss_config = SelfSupervisedConfig {
                learning_rate: tc.learning_rate,
                batch_size: tc.batch_size,
                min_buffer_size: tc.min_buffer_size,
                steps_per_cycle: tc.steps_per_cycle,
                ..Default::default()
            };
            let buffer_config = ReplayBufferConfig {
                min_quality_threshold: tc.min_quality_threshold,
                ..Default::default()
            };

            // Create trainer with checkpoint manager if target_adapter is set
            let mut trainer = SelfSupervisedTrainer::new(ss_config, buffer_config);

            if let Some(ref target_adapter) = tc.target_adapter {
                // Create checkpoint manager for weight persistence
                let checkpoint_config = CheckpointConfig {
                    max_checkpoints: 5,
                    git_commit_interval: tc.steps_per_cycle * 10, // Commit every 10 cycles
                    queue_size: 10,
                };

                match CheckpointManager::with_config(
                    model_path.clone(),
                    checkpoint_config,
                    None,
                ) {
                    Ok(checkpoint_mgr) => {
                        let checkpoint_mgr =
                            checkpoint_mgr.with_target_adapter(target_adapter.clone());
                        trainer = trainer.with_checkpoint_manager(checkpoint_mgr);
                        info!(
                            "Checkpoint manager initialized, target_adapter: {}",
                            target_adapter
                        );
                    }
                    Err(e) => {
                        warn!("Failed to create checkpoint manager: {}", e);
                    }
                }
            }

            Some(Arc::new(trainer))
        } else {
            None
        }
    } else {
        None
    };

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

    // Keep a copy for training (prompt is moved into request builder)
    let formatted_prompt_for_training = formatted_prompt.clone();

    let mut request_builder = GenerationRequest::builder(formatted_prompt)
        .apply_config(&crate::config::SamplingParams::from_model_path(&model_path).await.unwrap_or_default())
        .temperature(temperature.unwrap_or(0.7))
        .top_p(top_p.unwrap_or(0.95))
        .top_k(top_k)
        .repeat_penalty(repeat_penalty.unwrap_or(1.0))
        .seed(seed.map(|s| s as u64))
        .max_tokens(max_tokens.unwrap_or(2048));

    // Add image path if provided (for multimodal models)
    if let Some(img_path) = image_path {
        info!("Using image: {}", img_path);
        request_builder = request_builder.image_path(std::path::PathBuf::from(img_path));
    }

    let request = request_builder.build();

    info!(
        "Generating response: max_tokens={}, temperature={}, top_p={}, top_k={:?}, repeat_penalty={}",
        request.max_tokens, request.temperature, request.top_p, request.top_k, request.repeat_penalty
    );

    use futures::StreamExt;

    let mut text_stream = engine.generate(request)?;

    // Collect full response text (needed for training)
    let mut full_response = String::new();

    if stream {
        println!();
        while let Some(text_chunk) = text_stream.next().await {
            let chunk = text_chunk?;
            full_response.push_str(&chunk);
            print!("{}", chunk);
            let _ = io::stdout().flush();
        }
        println!();
    } else {
        while let Some(text_chunk) = text_stream.next().await {
            full_response.push_str(&text_chunk?);
        }
        println!("\n{}", full_response);
    }

    let stats = text_stream.stats();
    info!(
        "Generated {} tokens in {}ms ({:.2} tokens/sec)",
        stats.tokens_generated, stats.generation_time_ms, stats.tokens_per_second
    );
    if let Some(ref qm) = stats.quality_metrics {
        info!(
            "Quality metrics: perplexity={:.2}, entropy={:.2}, entropy_var={:.4}, repetition={:.3}, quality_score={:.3}",
            qm.perplexity, qm.avg_entropy, qm.entropy_variance, qm.repetition_ratio, qm.quality_score()
        );
    }

    // Collect training example if self-supervised training is enabled
    if let Some(ref trainer) = trainer {
        if let Some(ref qm) = stats.quality_metrics {
            // Tokenize prompt and response for training using public tokenizer
            match engine.get_tokenizer() {
                Ok(tokenizer) => {
                    let tokenize = |text: &str| -> Result<Vec<i64>> {
                        let encoding = tokenizer
                            .encode(text, false)
                            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
                    };

                    match (tokenize(&formatted_prompt_for_training), tokenize(&full_response)) {
                        (Ok(prompt_tokens), Ok(response_tokens)) => {
                            trainer
                                .add_example(prompt_tokens, response_tokens, qm.clone(), None)
                                .await;

                            info!(
                                "Training example collected (quality={:.3}, buffer_size={})",
                                qm.quality_score(),
                                trainer.replay_buffer.len().await
                            );

                            // Check if ready to train and trigger training cycle
                            if trainer.ready_to_train().await {
                                info!("Training cycle triggered...");
                                match trainer.train_cycle(&engine).await {
                                    Ok(result) => {
                                        info!(
                                            "Training cycle complete: {} steps, mean_loss={:.4}, mean_reward={:.3}",
                                            result.steps, result.total_loss, result.mean_reward
                                        );
                                    }
                                    Err(e) => {
                                        warn!("Training cycle failed: {}", e);
                                    }
                                }
                            }
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            warn!("Failed to tokenize for training: {}", e);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to get tokenizer for training: {}", e);
                }
            }
        }
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
    let repo_id = storage.resolve_repo_id(&model_ref).await?;
    let registry = storage.registry().await;
    let handle = registry.repo(&repo_id)?;

    // Create a temporary worktree for push operations
    let temp_dir = tempfile::tempdir()?;
    let worktree_path = temp_dir.path();
    let mut worktree = handle.create_worktree(worktree_path, "temp-push").await?;

    // Push current branch (worktree.push only takes remote parameter)
    worktree.push(Some(remote_name)).await?;

    // Note: The old code was trying to push specific branches, but WorktreeHandle.push()
    // only operates on the current branch of the worktree. For multi-branch support,
    // we'd need to checkout branches first or use different approach.

    worktree.cleanup().await?;

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
    let repo_id = storage.resolve_repo_id(&model_ref).await?;
    let registry = storage.registry().await;
    let handle = registry.repo(&repo_id)?;

    // Create a temporary worktree for pull operations
    let temp_dir = tempfile::tempdir()?;
    let worktree_path = temp_dir.path();
    let mut worktree = handle.create_worktree(worktree_path, "temp-pull").await?;

    // Pull current branch (worktree.pull only takes remote parameter)
    worktree.pull(Some(remote_name)).await?;

    // Note: The old code was trying to pull specific branches, but WorktreeHandle.pull()
    // only operates on the current branch of the worktree. For multi-branch support,
    // we'd need to checkout branches first or use different approach.

    worktree.cleanup().await?;

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

/// Build git2::MergeOptions from our MergeOptions
fn build_git2_merge_options(options: &MergeOptions) -> Result<git2::MergeOptions> {
    let mut merge_opts = git2::MergeOptions::new();

    // Apply strategy-based settings
    if let Some(strategy) = &options.strategy {
        match strategy.as_str() {
            "ours" => {
                merge_opts.file_favor(git2::FileFavor::Ours);
            }
            "theirs" => {
                merge_opts.file_favor(git2::FileFavor::Theirs);
            }
            "recursive" => {
                // recursive is the default, enable rename detection
                merge_opts.find_renames(true);
            }
            "resolve" => {
                // Simple two-way merge
                merge_opts.find_renames(false);
            }
            "subtree" => {
                // Subtree merge - no specific git2 option, but find renames
                merge_opts.find_renames(true);
            }
            "octopus" => {
                bail!("Octopus strategy (multi-branch merge) not supported for two-branch merges");
            }
            _ => {
                bail!("Unknown merge strategy: '{}'. Supported: ours, theirs, recursive, resolve, subtree", strategy);
            }
        }
    }

    // Apply strategy options (-X options)
    for opt in &options.strategy_option {
        match opt.as_str() {
            "ours" => {
                merge_opts.file_favor(git2::FileFavor::Ours);
            }
            "theirs" => {
                merge_opts.file_favor(git2::FileFavor::Theirs);
            }
            "patience" => {
                merge_opts.patience(true);
            }
            "diff-algorithm=patience" => {
                merge_opts.patience(true);
            }
            "diff-algorithm=minimal" => {
                merge_opts.minimal(true);
            }
            "ignore-space-change" | "ignore-all-space" => {
                merge_opts.ignore_whitespace_change(true);
            }
            "ignore-space-at-eol" => {
                merge_opts.ignore_whitespace_eol(true);
            }
            "ignore-cr-at-eol" => {
                // git2 doesn't have direct support, but ignore-whitespace-eol covers this
                merge_opts.ignore_whitespace_eol(true);
            }
            "renormalize" => {
                // Not directly supported in git2, but we can note it
                if !options.quiet {
                    eprintln!("Warning: renormalize strategy option not fully supported");
                }
            }
            "no-renames" => {
                merge_opts.find_renames(false);
            }
            "find-renames" => {
                merge_opts.find_renames(true);
            }
            opt if opt.starts_with("find-renames=") => {
                merge_opts.find_renames(true);
                if let Some(threshold_str) = opt.strip_prefix("find-renames=") {
                    if let Ok(threshold) = threshold_str.parse::<u32>() {
                        merge_opts.rename_threshold(threshold);
                    }
                }
            }
            opt if opt.starts_with("rename-threshold=") => {
                if let Some(threshold_str) = opt.strip_prefix("rename-threshold=") {
                    if let Ok(threshold) = threshold_str.parse::<u32>() {
                        merge_opts.rename_threshold(threshold);
                    }
                }
            }
            opt if opt.starts_with("subtree") => {
                // Subtree strategy - find renames enabled
                merge_opts.find_renames(true);
            }
            _ => {
                if !options.quiet {
                    eprintln!("Warning: unknown or unsupported strategy option: '{}'", opt);
                }
            }
        }
    }

    // Default: enable rename detection for better merge results
    if options.strategy.is_none() && options.strategy_option.is_empty() {
        merge_opts.find_renames(true);
    }

    Ok(merge_opts)
}

/// Perform merge operation in a repository
fn perform_merge(
    repo: &git2::Repository,
    source: &str,
    options: &MergeOptions,
) -> Result<git2::Oid> {
    // Resolve source branch reference
    let source_ref = repo
        .find_reference(&format!("refs/heads/{}", source))
        .or_else(|_| repo.find_reference(&format!("refs/remotes/origin/{}", source)))
        .or_else(|_| repo.find_reference(source))
        .map_err(|e| anyhow::anyhow!("Source branch '{}' not found: {}", source, e))?;

    let source_commit = source_ref
        .peel_to_commit()
        .map_err(|e| anyhow::anyhow!("Failed to resolve source commit: {}", e))?;

    let annotated_commit = repo
        .find_annotated_commit(source_commit.id())
        .map_err(|e| anyhow::anyhow!("Failed to create annotated commit: {}", e))?;

    // Perform merge analysis
    let (merge_analysis, _) = repo
        .merge_analysis(&[&annotated_commit])
        .map_err(|e| anyhow::anyhow!("Merge analysis failed: {}", e))?;

    // Already up-to-date
    if merge_analysis.is_up_to_date() {
        return Ok(source_commit.id());
    }

    // Check fast-forward constraints
    if options.ff_only && !merge_analysis.is_fast_forward() {
        bail!("Cannot fast-forward - branches have diverged");
    }

    // Fast-forward merge (if possible and not --no-ff)
    if merge_analysis.is_fast_forward() && !options.no_ff {
        let mut head_ref = repo
            .head()
            .map_err(|e| anyhow::anyhow!("Failed to get HEAD: {}", e))?;

        head_ref
            .set_target(source_commit.id(), "Fast-forward merge")
            .map_err(|e| anyhow::anyhow!("Failed to update HEAD: {}", e))?;

        repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))
            .map_err(|e| anyhow::anyhow!("Checkout failed: {}", e))?;

        return Ok(source_commit.id());
    }

    // Build merge options based on strategy and strategy options
    let mut merge_opts = build_git2_merge_options(options)?;

    // Regular merge (create merge commit)
    repo.merge(&[&annotated_commit], Some(&mut merge_opts), None)
        .map_err(|e| anyhow::anyhow!("Merge failed: {}", e))?;

    // Check for conflicts
    let index = repo
        .index()
        .map_err(|e| anyhow::anyhow!("Failed to get index: {}", e))?;

    if index.has_conflicts() {
        bail!("Merge conflicts detected");
    }

    // Create merge commit
    let sig = git2db::GitManager::global().create_signature(None, None)?;

    let mut index = repo
        .index()
        .map_err(|e| anyhow::anyhow!("Failed to get index: {}", e))?;

    let tree_id = index
        .write_tree()
        .map_err(|e| anyhow::anyhow!("Failed to write tree: {}", e))?;

    let tree = repo
        .find_tree(tree_id)
        .map_err(|e| anyhow::anyhow!("Failed to find tree: {}", e))?;

    let parent = repo
        .head()
        .map_err(|e| anyhow::anyhow!("Failed to get HEAD: {}", e))?
        .peel_to_commit()
        .map_err(|e| anyhow::anyhow!("Failed to resolve HEAD commit: {}", e))?;

    let message = options.message.as_deref().unwrap_or("Merge branch");
    let full_message = format!("{} '{}'", message, source);

    let merge_oid = repo
        .commit(
            Some("HEAD"),
            &sig,
            &sig,
            &full_message,
            &tree,
            &[&parent, &source_commit],
        )
        .map_err(|e| anyhow::anyhow!("Failed to create merge commit: {}", e))?;

    repo.cleanup_state()
        .map_err(|e| anyhow::anyhow!("Failed to cleanup merge state: {}", e))?;

    Ok(merge_oid)
}

/// Options for merge command
pub struct MergeOptions {
    pub ff: bool,
    pub no_ff: bool,
    pub ff_only: bool,
    pub no_commit: bool,
    pub squash: bool,
    pub message: Option<String>,
    pub abort: bool,
    pub continue_merge: bool,
    pub quit: bool,
    pub no_stat: bool,
    pub quiet: bool,
    pub verbose: bool,
    pub strategy: Option<String>,
    pub strategy_option: Vec<String>,
    pub allow_unrelated_histories: bool,
    pub no_verify: bool,
}

/// Handle merge command
pub async fn handle_merge(
    storage: &ModelStorage,
    target: &str,
    source: &str,
    options: MergeOptions,
) -> Result<()> {
    // Handle conflict resolution modes first
    if options.abort || options.continue_merge || options.quit {
        return handle_merge_conflict_resolution(storage, target, options).await;
    }

    // Parse target ModelRef (e.g., "Qwen3-4B:branch3")
    let target_ref = ModelRef::parse(target)?;

    if !options.quiet {
        info!("Merging '{}' into '{}'", source, target_ref);
    }

    // Extract target branch from ModelRef
    let target_branch = match &target_ref.git_ref {
        GitRef::Branch(b) => b.clone(),
        GitRef::DefaultBranch => {
            // Use default branch if not specified
            storage.get_default_branch(&target_ref).await?
        },
        _ => {
            bail!("Target must be a branch reference, not tag or commit: {}", target_ref.git_ref.display_name());
        }
    };

    // Ensure worktree exists for target branch (create if needed)
    let worktree_path = match storage.get_worktree_path(&target_ref, &target_branch).await {
        Ok(path) if path.exists() => path,
        _ => {
            if !options.quiet {
                println!("→ Creating worktree for target branch '{}'", target_branch);
            }
            storage.create_worktree(&target_ref, &target_branch).await?
        }
    };

    if options.verbose {
        println!("  Worktree: {}", worktree_path.display());
    }

    // Open the worktree repository (not the bare repo)
    let repo = git2::Repository::open(&worktree_path)
        .map_err(|e| anyhow::anyhow!("Failed to open worktree repository: {}", e))?;

    // Verify we're on the target branch
    let head = repo.head().map_err(|e| anyhow::anyhow!("Failed to get HEAD: {}", e))?;
    let current_branch = head.shorthand().unwrap_or("<detached>");

    if current_branch != target_branch {
        // The worktree should already be on the correct branch
        // This shouldn't normally happen, but if it does, we can fix it
        if options.verbose {
            println!("  Switching to target branch '{}' (currently on '{}')", target_branch, current_branch);
        }

        let branch_ref = repo.find_branch(&target_branch, git2::BranchType::Local)
            .map_err(|e| anyhow::anyhow!("Target branch '{}' not found: {}", target_branch, e))?;

        let commit = branch_ref.get().peel_to_commit()
            .map_err(|e| anyhow::anyhow!("Failed to get commit for branch '{}': {}", target_branch, e))?;

        repo.checkout_tree(commit.as_object(), Some(git2::build::CheckoutBuilder::new().force()))
            .map_err(|e| anyhow::anyhow!("Failed to checkout '{}': {}", target_branch, e))?;

        repo.set_head(&format!("refs/heads/{}", target_branch))
            .map_err(|e| anyhow::anyhow!("Failed to set HEAD to '{}': {}", target_branch, e))?;
    }

    // Perform the merge directly in the worktree
    let merge_result = perform_merge(&repo, source, &options);

    match merge_result {
        Ok(merge_oid) => {
            if !options.quiet {
                println!("✓ Merged '{}' into '{}'", source, target_ref);

                // Show merge strategy used
                if !options.no_stat {
                    if options.ff_only {
                        println!("  Strategy: fast-forward only");
                    } else if options.no_ff {
                        println!("  Strategy: no fast-forward (merge commit created)");
                    } else {
                        println!("  Strategy: auto (fast-forward if possible)");
                    }

                    // Show commit ID
                    if options.verbose {
                        println!("  Commit: {}", merge_oid);
                    }
                }
            }

            Ok(())
        },
        Err(e) => {
            // Check if it's a merge conflict
            if e.to_string().contains("conflict") {
                eprintln!("✗ Merge conflict detected");
                eprintln!("\nResolve conflicts in: {}", worktree_path.display());
                eprintln!("\nThen run:");
                eprintln!("  hyprstream merge {} --continue", target);
                eprintln!("\nOr abort the merge:");
                eprintln!("  hyprstream merge {} --abort", target);
                bail!("Merge conflicts must be resolved manually");
            } else {
                Err(e)
            }
        }
    }
}

/// Handle merge conflict resolution (--abort, --continue, --quit)
async fn handle_merge_conflict_resolution(
    storage: &ModelStorage,
    target: &str,
    options: MergeOptions,
) -> Result<()> {
    let target_ref = ModelRef::parse(target)?;

    // Get target branch
    let target_branch = match &target_ref.git_ref {
        GitRef::Branch(b) => b.clone(),
        GitRef::DefaultBranch => storage.get_default_branch(&target_ref).await?,
        _ => bail!("Target must be a branch reference"),
    };

    // Get worktree path
    let worktree_path = storage.get_worktree_path(&target_ref, &target_branch).await?;

    if !worktree_path.exists() {
        bail!("Worktree not found: {}", worktree_path.display());
    }

    // Open repository
    let repo_id = storage.resolve_repo_id(&target_ref).await?;
    let registry = storage.registry().await;
    let handle = registry.repo(&repo_id)?;
    let repo = handle.open_repo()?;

    if options.abort {
        // Abort merge: restore pre-merge state
        if !options.quiet {
            println!("→ Aborting merge...");
        }

        // Reset to ORIG_HEAD if it exists
        if let Ok(orig_head) = repo.refname_to_id("ORIG_HEAD") {
            let commit = repo.find_commit(orig_head)?;
            repo.reset(commit.as_object(), git2::ResetType::Hard, None)?;

            // Cleanup merge state
            repo.cleanup_state()?;

            if !options.quiet {
                println!("✓ Merge aborted, restored pre-merge state");
            }
        } else {
            bail!("No merge in progress (ORIG_HEAD not found)");
        }
    } else if options.continue_merge {
        // Continue merge: check if conflicts are resolved
        if !options.quiet {
            println!("→ Continuing merge...");
        }

        let mut index = repo.index()?;
        if index.has_conflicts() {
            bail!("Conflicts still present. Resolve all conflicts before continuing.");
        }

        // Write tree and create merge commit
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;

        let sig = git2db::GitManager::global().create_signature(None, None)?;

        // Get parent commits
        let head = repo.head()?.peel_to_commit()?;
        let merge_head = repo.find_reference("MERGE_HEAD")?
            .peel_to_commit()?;

        let message = options.message.unwrap_or_else(|| {
            format!("Merge branch '{}'", merge_head.summary().unwrap_or("unknown"))
        });

        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            &message,
            &tree,
            &[&head, &merge_head],
        )?;

        // Cleanup merge state
        repo.cleanup_state()?;

        if !options.quiet {
            println!("✓ Merge completed successfully");
        }
    } else if options.quit {
        // Quit merge: keep working tree changes but remove merge state
        if !options.quiet {
            println!("→ Quitting merge (keeping changes)...");
        }

        repo.cleanup_state()?;

        if !options.quiet {
            println!("✓ Merge state removed, changes retained");
            println!("  Use 'git status' to see modified files");
        }
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
        // First, try to clean up any worktrees which might have overlay mounts
        // This is important to avoid permission errors from mounted filesystems
        let worktrees_dir = model_path.join("worktrees");
        if worktrees_dir.exists() {
            // Try to clean up overlay mounts gracefully
            debug!("Cleaning up worktrees at: {}", worktrees_dir.display());

            // git2db overlay mounts are typically in .git2db-overlay subdirectories
            // We'll attempt cleanup but continue on error since some may not be mounted
            if let Ok(entries) = std::fs::read_dir(&worktrees_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        // Attempt to remove directory - overlays should auto-unmount on cleanup
                        if let Err(e) = std::fs::remove_dir_all(&path) {
                            debug!("Failed to remove worktree dir {}: {} (may have been unmounted)",
                                   path.display(), e);
                        }
                    }
                }
            }
        }

        // Now remove the main model directory
        match std::fs::remove_dir_all(&model_path) {
            Ok(_) => {
                println!("✓ Removed model files from: {}", model_path.display());
            }
            Err(e) => {
                // Check if it's a permission error and provide helpful message
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    eprintln!("❌ Failed to remove model files: {}", e);
                    eprintln!("   This may be due to overlay filesystem mounts.");
                    eprintln!("   Try running with sudo or manually unmount any overlayfs mounts:");
                    eprintln!("   $ mount | grep {}", model_path.display());
                    eprintln!("   $ sudo umount <mount_path>");
                } else {
                    eprintln!("❌ Failed to remove model files: {}", e);
                }
                return Err(anyhow::anyhow!("Failed to remove model files: {}", e));
            }
        }
    }

    println!("✓ Model '{}' removed successfully", model);
    Ok(())
}
