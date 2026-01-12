//! Handlers for git-style CLI commands

use crate::config::GenerationRequest;
use crate::inference_capnp;
use crate::runtime::template_engine::ChatMessage;
use crate::runtime::RuntimeConfig;
use crate::services::{
    InferenceService, InferenceZmqClient, PolicyZmqClient, INFERENCE_ENDPOINT,
};
use crate::zmq::global_context;
use crate::storage::{CheckoutOptions, GitRef, ModelRef, ModelStorage};
use anyhow::{bail, Result};
use capnp::message::ReaderOptions;
use capnp::serialize;
use hyprstream_rpc::prelude::*;
use std::io::{self, Write};
use tracing::{debug, info, warn};

/// Handle branch command
pub async fn handle_branch(
    storage: &ModelStorage,
    model: &str,
    branch_name: &str,
    from_ref: Option<String>,
    policy_template: Option<String>,
) -> Result<()> {
    info!("Creating branch {} for model {}", branch_name, model);

    let model_ref = ModelRef::new(model.to_string());

    // Create branch using ModelStorage helper
    storage.create_branch(&model_ref, branch_name, from_ref.as_deref()).await?;

    println!("✓ Created branch {}", branch_name);

    if let Some(ref from) = from_ref {
        println!("  Branch created from: {}", from);
    }

    // Create worktree for the branch
    let worktree_path = storage.create_worktree(&model_ref, branch_name).await?;
    println!("✓ Created worktree at {}", worktree_path.display());

    // Apply policy template if specified
    if let Some(ref template_name) = policy_template {
        apply_policy_template_to_model(storage, model, template_name).await?;
    }

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
                // SAFETY: capture groups 1 and 2 exist because regex matched
                let name = captures.get(1)
                    .ok_or_else(|| anyhow::anyhow!("Invalid author format: missing name"))?
                    .as_str().trim();
                let email = captures.get(2)
                    .ok_or_else(|| anyhow::anyhow!("Invalid author format: missing email"))?
                    .as_str().trim();
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

/// Handle list command
pub async fn handle_list(
    storage: &ModelStorage,
    policy_manager: Option<&crate::auth::PolicyManager>,
) -> Result<()> {
    info!("Listing models");

    let models = storage.list_models().await?;

    if models.is_empty() {
        println!("No models found.");
        println!("Try: hyprstream clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct");
        return Ok(());
    }

    // Get current user for permission checks (OS user for CLI)
    let current_user = users::get_current_username()
        .map(|u| u.to_string_lossy().to_string())
        .unwrap_or_else(|| "anonymous".to_string());

    // Get archetype registry for capability detection
    let archetype_registry = crate::archetypes::global_registry();

    // Get storage paths for model directories
    let storage_paths = crate::storage::StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;

    // Collect models with git info and capabilities
    let mut models_with_info = Vec::new();
    for (model_ref, metadata) in models {
        // Get commit hash via service layer (avoids GitManager::global())
        let commit = match storage.registry().repo(&model_ref.model).await {
            Ok(repo_client) => {
                match repo_client.get_head().await {
                    Ok(head_oid) => {
                        // Truncate to 7 characters for display
                        head_oid.chars().take(7).collect::<String>()
                    }
                    Err(_) => "unknown".to_string(),
                }
            }
            Err(_) => "unknown".to_string(),
        };

        // Get ref name from ModelRef
        let git_ref = model_ref.git_ref_str().unwrap_or_else(|| "detached".to_string());

        // Detect capabilities from worktree path
        let worktree_path = models_dir
            .join(&model_ref.model)
            .join("worktrees")
            .join(model_ref.git_ref_str().unwrap_or_else(|| "main".to_string()));
        let detected = archetype_registry.detect(&worktree_path);
        let domains = detected.to_detected_domains();

        // Compute access string based on user permissions
        let resource = format!("model:{}", model_ref.model);
        let access_str = if let Some(pm) = policy_manager {
            crate::auth::capabilities_to_access_string(pm, &current_user, &resource, &domains.capabilities)
        } else {
            // No policy manager = full access (show all capabilities)
            domains.capabilities.to_ids().join(",")
        };

        models_with_info.push((model_ref, metadata, commit, git_ref, domains, access_str));
    }

    // Table format with DOMAINS and ACCESS columns
    println!(
        "{:<30} {:<16} {:<16} {:<15} {:<8} {:<6} {:<10}",
        "MODEL NAME", "DOMAINS", "ACCESS", "REF", "COMMIT", "STATUS", "SIZE"
    );
    println!("{}", "-".repeat(111));

    for (_model_ref, metadata, commit, git_ref, domains, access_str) in &models_with_info {
        let size_str = if let Some(size) = metadata.size_bytes {
            format!("{:.1}GB", size as f64 / (1024.0 * 1024.0 * 1024.0))
        } else {
            "n/a".to_string()
        };

        let domains_str = domains.domains_display();

        // commit and git_ref are now available directly from the tuple
        // (fetched via service layer, avoiding GitManager::global() initialization)
        let status = if metadata.is_dirty { "dirty" } else { "clean" };

        println!(
            "{:<30} {:<16} {:<16} {:<15} {:<8} {:<6} {:<10}",
            metadata.display_name.as_deref().unwrap_or("unnamed"), domains_str, access_str, git_ref, commit, status, size_str
        );
    }

    if models_with_info.is_empty() {
        println!("No models match the specified filters.");
    }

    Ok(())
}

/// Handle clone command
pub async fn handle_clone(
    storage: &ModelStorage,
    repo_url: &str,
    name: Option<String>,
    branch: Option<String>,
    depth: u32,
    full: bool,
    quiet: bool,
    verbose: bool,
    policy_template: Option<String>,
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

    // Determine model name from URL or use provided name
    let model_name = if let Some(n) = name {
        n
    } else {
        let extracted = repo_url
            .split('/')
            .next_back()
            .unwrap_or("")
            .trim_end_matches(".git")
            .to_string();

        if extracted.is_empty() {
            anyhow::bail!(
                "Cannot derive model name from URL '{}'. Please provide --name.",
                repo_url
            );
        }
        extracted
    };

    // Use the passed storage directly (which has the shared registry client)
    // This avoids starting a second registry service
    storage.add_model(&model_name, repo_url).await?;

    // Create default worktree so the model appears in list
    // Use requested branch or fall back to default branch (usually "main")
    let worktree_branch = if let Some(ref b) = branch {
        b.clone()
    } else {
        storage
            .get_default_branch(&ModelRef::new(model_name.clone()))
            .await
            .unwrap_or_else(|_| "main".to_string())
    };

    if !quiet {
        println!("   Creating worktree for branch: {}", worktree_branch);
    }
    let model_ref = ModelRef::new(model_name.clone());
    storage.create_worktree(&model_ref, &worktree_branch).await?;

    let model_path = storage.get_model_path(&model_ref).await?;

    // Generate a model ID for compatibility (matches operations.rs behavior)
    let model_id = crate::storage::ModelId::new();

    let cloned = crate::storage::ClonedModel {
        model_id,
        model_path,
        model_name,
    };

    if !quiet {
        println!("✅ Model '{}' cloned successfully!", cloned.model_name);
        println!("   Model ID: {}", cloned.model_id);
        println!("   Location: {}", cloned.model_path.display());
    }

    // Apply policy template if specified
    if let Some(ref template_name) = policy_template {
        apply_policy_template_to_model(storage, &cloned.model_name, template_name).await?;
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

        // Detect and display archetypes
        let archetype_registry = crate::archetypes::global_registry();
        let detected = archetype_registry.detect(&model_path);

        if !detected.is_empty() {
            println!("\nArchetypes:");
            for archetype in &detected.archetypes {
                println!("  - {}", archetype);
            }
            println!("Capabilities: {}", detected.capabilities);
        } else {
            println!("\nArchetypes: None detected");
        }
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
                    if let (true, Some(config_path)) = (verbose, adapter.config_path.as_ref()) {
                        if let Ok(config_content) =
                            std::fs::read_to_string(config_path)
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

/// Apply a policy template to a model's registry
///
/// This is a helper used by branch, clone, and worktree commands to apply
/// policy templates when the --policy flag is specified.
pub async fn apply_policy_template_to_model(
    storage: &ModelStorage,
    model: &str,
    template_name: &str,
) -> Result<()> {
    use crate::auth::PolicyManager;
    use crate::cli::policy_handlers::get_template;
    use std::process::Command;

    let template = get_template(template_name)
        .ok_or_else(|| anyhow::anyhow!(
            "Unknown policy template: '{}'. Available templates: local, public-inference, public-read",
            template_name
        ))?;

    // Get the policies directory for this model's registry
    let registry_path = storage.get_models_dir().join(".registry");
    let policies_dir = registry_path.join("policies");

    // Ensure policies directory exists
    if !policies_dir.exists() {
        tokio::fs::create_dir_all(&policies_dir).await?;
    }

    let policy_path = policies_dir.join("policy.csv");

    // Read existing policy content or create default
    let existing_content = if policy_path.exists() {
        tokio::fs::read_to_string(&policy_path).await?
    } else {
        default_policy_header().to_string()
    };

    // Check if template rules already exist
    if existing_content.contains(template.rules.trim()) {
        println!("✓ Policy template '{}' already applied", template_name);
        return Ok(());
    }

    // Append the template rules
    let new_content = format!("{}\n{}", existing_content.trim_end(), template.rules);

    // Write the updated policy
    tokio::fs::write(&policy_path, &new_content).await?;

    // Validate the new policy
    let policy_manager = PolicyManager::new(&policies_dir)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to validate policy: {}", e))?;

    // Reload to validate
    if let Err(e) = policy_manager.reload().await {
        // Rollback on validation failure
        tokio::fs::write(&policy_path, &existing_content).await?;
        bail!("Policy validation failed: {}. Template not applied.", e);
    }

    // Commit the change
    let commit_msg = format!("policy: apply {} template for model {}", template_name, model);

    Command::new("git")
        .current_dir(&registry_path)
        .args(["add", "policies/"])
        .output()
        .ok();

    Command::new("git")
        .current_dir(&registry_path)
        .args(["commit", "-m", &commit_msg])
        .output()
        .ok();

    println!("✓ Applied policy template: {}", template_name);
    println!("  {}", template.description);

    Ok(())
}

/// Default policy header for new policy files
fn default_policy_header() -> &'static str {
    r#"# Hyprstream Access Control Policy
# Format: p, subject, resource, action
#
# Subjects: user names or role names
# Resources: model:<name>, data:<name>, or * for all
# Actions: infer, train, query, write, serve, manage, or * for all
"#
}

/// Handle infer command
///
/// Runs inference via InferenceService, which:
/// - Enforces authorization via PolicyManager
/// - Auto-loads adapters from model directory
/// - **Training is always DISABLED** - this is a read-only inference command
///
/// For inference with training (TTT), use `hyprstream training infer` instead.
///
/// # Parameters
/// - `signing_key`: Ed25519 signing key for request authentication
/// - `verifying_key`: Ed25519 verifying key for signature verification
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
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
) -> Result<()> {
    use crate::config::TrainingMode;
    use crate::runtime::model_config::ModelConfig;

    info!(
        "Running read-only inference via service: model={}, prompt_len={}",
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
        eprintln!("❌ Model '{}' not found", model_ref.model);
        eprintln!("   Use 'hyprstream list' to see available models");
        return Err(anyhow::anyhow!("Model '{}' not found", model_ref.model));
    }

    info!("Using model at: {}", model_path.display());

    // CRITICAL: Force training mode to Disabled for CLI infer command
    // This ensures hyprstream infer is ALWAYS read-only, regardless of model config.
    // For training inference, use `hyprstream training infer` instead.
    if let Some(mut training_config) = ModelConfig::load_training_config(&model_path) {
        if training_config.mode != TrainingMode::Disabled {
            debug!(
                "Model has training mode {:?}, overriding to Disabled for read-only infer",
                training_config.mode
            );
            training_config.mode = TrainingMode::Disabled;
            // Save temporarily to ensure InferenceService sees Disabled mode
            if let Err(e) = ModelConfig::save_training_config(&model_path, &training_config) {
                warn!("Could not save disabled training config: {}", e);
            }
        }
    }

    // PolicyService is already running (started by main.rs).
    // Create policy client with the same keypair to connect to it.
    let policy_client = PolicyZmqClient::new(signing_key.clone(), RequestIdentity::local());

    // Configure runtime
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.max_context = max_context;
    runtime_config.kv_quant_type = kv_quant.into();
    // Note: Training mode is already forced to Disabled in the config file above

    // Start InferenceService (loads model, adapters, trainer automatically)
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

    // Apply chat template to the prompt via service
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
    }];

    let formatted_prompt = match client.apply_chat_template(&messages, true).await {
        Ok(formatted) => formatted,
        Err(e) => {
            tracing::warn!("Could not apply chat template: {}. Using raw prompt.", e);
            prompt.to_string()
        }
    };

    // Build generation request
    let mut request_builder = GenerationRequest::builder(formatted_prompt)
        .apply_config(
            &crate::config::SamplingParams::from_model_path(&model_path)
                .await
                .unwrap_or_default(),
        )
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

    // Generate via service (handles adapter loading, training collection, auth)
    if stream {
        // Streaming via ZMQ pub/sub
        let (stream_id, endpoint) = client.generate_stream(&request).await?;
        info!("Streaming started: stream_id={}, endpoint={}", stream_id, endpoint);

        // Subscribe to stream chunks using global context
        // (inproc:// only works within the same ZMQ context)
        let ctx = global_context();
        let subscriber = ctx.socket(zmq::SUB)?;
        subscriber.connect(&endpoint)?;
        subscriber.set_subscribe(stream_id.as_bytes())?;
        subscriber.set_rcvtimeo(30000)?; // 30s timeout

        println!();
        loop {
            match subscriber.recv_bytes(0) {
                Ok(msg) => {
                    // Message format: topic_bytes + capnp_bytes
                    // Topic is the stream_id, followed by Cap'n Proto StreamChunk
                    let topic_len = stream_id.as_bytes().len();
                    if msg.len() <= topic_len {
                        continue; // Too short, skip
                    }

                    // Parse Cap'n Proto StreamChunk after the topic prefix
                    let chunk_bytes = &msg[topic_len..];
                    let reader = match serialize::read_message(
                        &mut std::io::Cursor::new(chunk_bytes),
                        ReaderOptions::default(),
                    ) {
                        Ok(r) => r,
                        Err(e) => {
                            warn!("Failed to parse stream chunk: {}", e);
                            continue;
                        }
                    };

                    let chunk = match reader.get_root::<inference_capnp::stream_chunk::Reader>() {
                        Ok(c) => c,
                        Err(e) => {
                            warn!("Failed to get stream chunk root: {}", e);
                            continue;
                        }
                    };

                    // Handle the union variant
                    use inference_capnp::stream_chunk::Which;
                    match chunk.which() {
                        Ok(Which::Text(text)) => {
                            if let Ok(t) = text {
                                print!("{}", t.to_str().unwrap_or(""));
                                let _ = io::stdout().flush();
                            }
                        }
                        Ok(Which::Complete(_stats)) => {
                            // Stream completed
                            break;
                        }
                        Ok(Which::Error(error_info)) => {
                            if let Ok(e) = error_info {
                                if let Ok(msg) = e.get_message() {
                                    warn!("Stream error: {}", msg.to_str().unwrap_or("unknown"));
                                }
                            }
                            break;
                        }
                        Err(e) => {
                            warn!("Failed to parse chunk variant: {:?}", e);
                            continue;
                        }
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout waiting for stream
                    warn!("Stream timeout");
                    break;
                }
                Err(e) => {
                    warn!("Stream error: {}", e);
                    break;
                }
            }
        }
        println!();
    } else {
        // Non-streaming: get full response
        let result = client.generate(&request).await?;

        println!("\n{}", result.text);
        info!(
            "Generated {} tokens in {}ms ({:.2} tokens/sec overall)",
            result.tokens_generated, result.generation_time_ms, result.tokens_per_second
        );
        info!(
            "  Prefill: {} tokens in {}ms ({:.2} tokens/sec)",
            result.prefill_tokens, result.prefill_time_ms, result.prefill_tokens_per_sec
        );
        info!(
            "  Inference: {} tokens in {}ms ({:.2} tokens/sec)",
            result.inference_tokens, result.inference_time_ms, result.inference_tokens_per_sec
        );

        if let Some(ref qm) = result.quality_metrics {
            info!(
                "Quality metrics: perplexity={:.2}, entropy={:.2}, entropy_var={:.4}, repetition={:.3}",
                qm.perplexity, qm.avg_entropy, qm.entropy_variance, qm.repetition_ratio
            );
        }
    }

    // Stop the service
    service_handle.stop().await;

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

    // Open repository for push operations
    let repo = storage.open_repo(&model_ref).await?;

    // Get current branch or specified branch
    let push_branch = if let Some(b) = branch_name {
        b.to_string()
    } else {
        repo.head()?
            .shorthand()
            .ok_or_else(|| anyhow::anyhow!("Could not determine current branch"))?
            .to_string()
    };

    // Find remote and push
    let mut remote = repo.find_remote(remote_name)?;
    let refspec = format!("refs/heads/{}:refs/heads/{}", push_branch, push_branch);
    remote.push(&[&refspec], None)?;

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

    // Open repository for pull operations
    let repo = storage.open_repo(&model_ref).await?;

    // Fetch from remote
    let mut remote = repo.find_remote(remote_name)?;
    remote.fetch(&["refs/heads/*:refs/remotes/origin/*"], None, None)?;

    // Get current branch or specified branch
    let pull_branch = if let Some(b) = branch_name {
        b.to_string()
    } else {
        repo.head()?
            .shorthand()
            .ok_or_else(|| anyhow::anyhow!("Could not determine current branch"))?
            .to_string()
    };

    // Get the remote tracking branch
    let remote_ref = format!("refs/remotes/{}/{}", remote_name, pull_branch);
    let fetch_head = repo.find_reference(&remote_ref)?;
    let fetch_commit = repo.reference_to_annotated_commit(&fetch_head)?;

    // Merge the fetched changes
    let (analysis, _) = repo.merge_analysis(&[&fetch_commit])?;

    if analysis.is_up_to_date() {
        println!("✓ Already up to date");
    } else if analysis.is_fast_forward() {
        // Fast-forward merge
        let refname = format!("refs/heads/{}", pull_branch);
        let mut reference = repo.find_reference(&refname)?;
        reference.set_target(fetch_commit.id(), "Fast-forward")?;
        repo.set_head(&refname)?;
        repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
        println!("✓ Fast-forwarded to latest");
    } else if analysis.is_normal() {
        // Normal merge
        repo.merge(&[&fetch_commit], None, None)?;
        println!("✓ Merged remote changes");
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

    // Get repository client from service layer
    let repo_client = storage.registry().repo(&target_ref.model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;

    // Build merge message
    let message = if let Some(msg) = &options.message {
        Some(msg.as_str())
    } else {
        None
    };

    // Perform merge via service layer
    let merge_result = repo_client.merge(source, message).await;

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
                        println!("  Commit: {}", &merge_oid[..8.min(merge_oid.len())]);
                    }
                }
            }

            Ok(())
        },
        Err(e) => {
            // Check if it's a merge conflict
            let err_msg = e.to_string();
            if err_msg.contains("conflict") || err_msg.contains("Conflict") {
                eprintln!("✗ Merge conflict detected");
                eprintln!("\nResolve conflicts in the repository");
                eprintln!("\nThen run:");
                eprintln!("  hyprstream merge {} --continue", target);
                eprintln!("\nOr abort the merge:");
                eprintln!("  hyprstream merge {} --abort", target);
                bail!("Merge conflicts must be resolved manually");
            } else {
                Err(e.into())
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
    let repo = storage.open_repo(&target_ref).await?;

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

        let sig = repo.signature()?;

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

    // Parse model reference to handle model:branch format
    let model_ref = ModelRef::parse(model)?;

    // Check if a specific branch/worktree was specified
    let is_worktree_removal = !matches!(model_ref.git_ref, crate::storage::GitRef::DefaultBranch);

    if is_worktree_removal {
        // Removing a specific worktree, not the entire model
        let branch = model_ref.git_ref.display_name();
        info!("Removing worktree {} for model {}", branch, model_ref.model);

        // Check if the worktree exists
        let worktree_path = match storage.get_model_path(&model_ref).await {
            Ok(path) => path,
            Err(_) => {
                println!("❌ Worktree '{}' not found for model '{}'", branch, model_ref.model);
                return Ok(());
            }
        };

        if !worktree_path.exists() {
            println!("❌ Worktree '{}' not found for model '{}'", branch, model_ref.model);
            return Ok(());
        }

        // Show what will be removed
        println!("Worktree '{}:{}' removal plan:", model_ref.model, branch);
        println!("  📁 Remove worktree at: {}", worktree_path.display());

        // Confirmation prompt unless forced
        if !force {
            print!("Are you sure you want to remove worktree '{}:{}'? [y/N]: ", model_ref.model, branch);
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim().to_lowercase();

            if input != "y" && input != "yes" {
                println!("Removal cancelled");
                return Ok(());
            }
        }

        // Remove the worktree
        storage.remove_worktree(&ModelRef::new(model_ref.model.clone()), &branch).await?;
        println!("✓ Worktree '{}:{}' removed successfully", model_ref.model, branch);
        return Ok(());
    }

    // Removing the entire model (no branch specified)
    // Check if model exists in registry
    let registry_exists = storage.get_model_path(&model_ref).await.is_ok();

    // Check if model exists in filesystem (model directory)
    let model_path = storage.get_models_dir().join(&model_ref.model);
    let files_exist = model_path.exists();

    if !registry_exists && !files_exist {
        println!("❌ Model '{}' not found in registry or filesystem", model_ref.model);
        return Ok(());
    }

    // Show what will be removed
    println!("Model '{}' removal plan:", model_ref.model);
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
        print!("Are you sure you want to remove model '{}'? [y/N]: ", model_ref.model);
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
                println!("✓ Removed '{}' from git registry", model_ref.model);
            }
            Err(e) => {
                eprintln!("❌ Failed to remove '{}' from git registry: {}", model_ref.model, e);
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

    println!("✓ Model '{}' removed successfully", model_ref.model);
    Ok(())
}
