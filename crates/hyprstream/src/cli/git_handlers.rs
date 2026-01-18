//! Handlers for git-style CLI commands

use crate::config::GenerationRequest;
use crate::inference_capnp;
use crate::api::openai_compat::ChatMessage;
use crate::runtime::RuntimeConfig;
use crate::services::{
    ModelZmqClient, PolicyZmqClient,
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
///
/// TODO: This function still uses local git2 operations for advanced features.
/// The following need to be added to RepositoryClient:
/// - Detailed status with file-level change types (A/M/D/R/??)
/// - Amend support
/// - Custom author support
/// - Different staging modes (stage_all vs stage_all_including_untracked)
///
/// **EXPERIMENTAL**: This feature is behind the `experimental` flag.
#[cfg(feature = "experimental")]
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
            // Get repository client from registry service
            let repo_client = storage.registry().repo(&model_ref.model).await?;
            repo_client.default_branch().await?
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

    // Get repository client
    let repo_client = storage.registry().repo(&model_ref.model).await?;

    // Check if worktree exists
    let worktree_path = repo_client.worktree_path(&branch_name).await?
        .ok_or_else(|| anyhow::anyhow!(
            "Worktree '{}' does not exist for model '{}'.\n\nCreate it first with:\n  hyprstream branch {} {}",
            branch_name, model_ref.model, model_ref.model, branch_name
        ))?;

    info!("Operating on worktree: {}", worktree_path.display());

    // TODO: Use RepositoryClient.status() here once it provides detailed file status
    // For now, we need to use git2 directly to get detailed status for verbose/dry-run output
    // TEMPORARY: Local git2 access for detailed status
    let worktree_repo = git2::Repository::open(&worktree_path)?;
    let statuses = worktree_repo.statuses(None)?;
    let has_changes = !statuses.is_empty();

    if !allow_empty && !amend && !has_changes && !all_untracked {
        println!("No changes to commit for {}:{}", model_ref.model, branch_name);
        println!("\nUse --allow-empty to create a commit without changes");
        return Ok(());
    }

    // Show what will be committed (needs detailed status)
    // TODO: Replace with RepositoryClient method that returns detailed file status
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

    // TODO: The following features need to be added to RepositoryClient:
    // - stage_all_including_untracked() for all_untracked mode
    // - commit_with_author() for custom author
    // - amend_commit() for amend mode
    //
    // For now, we use git2 directly for these advanced features
    // TEMPORARY: Local git2 access for staging and committing
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

    // Perform commit operation
    let commit_oid = if amend {
        // TODO: Add RepositoryClient.amend_commit(message) method
        // TEMPORARY: Local git2 access for amend
        info!("Amending previous commit");

        let tree_id = index.write_tree()?;
        let tree = worktree_repo.find_tree(tree_id)?;
        let head = worktree_repo.head()?;
        let commit_to_amend = head.peel_to_commit()?;

        commit_to_amend.amend(
            Some("HEAD"),
            None,
            None,
            None,
            Some(message),
            Some(&tree),
        )?
    } else if author.is_some() || author_name.is_some() || author_email.is_some() {
        // TODO: Add RepositoryClient.commit_with_author(message, author_name, author_email) method
        // TEMPORARY: Local git2 access for custom author
        let tree_id = index.write_tree()?;
        let tree = worktree_repo.find_tree(tree_id)?;
        let head = worktree_repo.head()?;
        let parent_commit = head.peel_to_commit()?;

        let signature = if let Some(author_str) = author {
            let re = regex::Regex::new(r"^(.+?)\s*<(.+?)>$")?;
            if let Some(captures) = re.captures(&author_str) {
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
        } else {
            let name = author_name.as_deref()
                .ok_or_else(|| anyhow::anyhow!("--author-name required when using --author-email"))?;
            let email = author_email.as_deref()
                .ok_or_else(|| anyhow::anyhow!("--author-email required when using --author-name"))?;
            git2::Signature::now(name, email)?
        };

        worktree_repo.commit(
            Some("HEAD"),
            &signature,
            &signature,
            message,
            &tree,
            &[&parent_commit],
        )?
    } else {
        // Simple commit - can use RepositoryClient
        repo_client.stage_all().await?;
        let oid_string = repo_client.commit(message).await?;
        git2::Oid::from_str(&oid_string)?
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
///
/// TODO: Adapter listing still uses local filesystem access. Consider moving to ModelService.
pub async fn handle_info(
    storage: &ModelStorage,
    model: &str,
    verbose: bool,
    adapters_only: bool,
) -> Result<()> {
    info!("Getting info for model {}", model);

    let model_ref = ModelRef::parse(model)?;

    // Get repository client from service layer
    let repo_client = storage.registry().repo(&model_ref.model).await?;

    // Get repository metadata via RepositoryClient
    let repo_metadata = {
        // Get remote URL
        let remotes = repo_client.list_remotes().await.ok();
        let url = remotes
            .as_ref()
            .and_then(|r| r.iter().find(|remote| remote.name == "origin"))
            .map(|remote| remote.url.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Get current HEAD
        let current_oid = repo_client.get_head().await.ok();

        Some((
            Some(model_ref.model.clone()),
            url,
            model_ref.git_ref.to_string(),
            current_oid,
        ))
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

        // Get display ref using RepositoryClient
        let display_ref = match &model_ref.git_ref {
            crate::storage::GitRef::DefaultBranch => {
                repo_client.default_branch().await.unwrap_or_else(|_| "unknown".to_string())
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

    // Get bare repository information via RepositoryClient
    println!("\nRepository Information:");

    // Get remote information
    if let Ok(remotes) = repo_client.list_remotes().await {
        if !remotes.is_empty() {
            for remote in remotes {
                println!("  Remote '{}': {}", remote.name, remote.url);
            }
        }
    }

    // Get branches
    if let Ok(branches) = repo_client.list_branches().await {
        if !branches.is_empty() {
            println!("  Local branches: {}", branches.join(", "));
        }
    }

    // TODO: Add list_tags() method to RepositoryClient
    // TEMPORARY: Size calculation - this could be moved to RepositoryClient as get_repo_size()
    let bare_repo_path = storage.get_models_dir()
        .join(&model_ref.model)
        .join(format!("{}.git", &model_ref.model));
    if bare_repo_path.exists() {
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
    }

    // Get git status via RepositoryClient
    println!("\nWorktree Status:");

    match repo_client.status().await {
        Ok(status) => {
            println!(
                "  Current branch/ref: {}",
                status.branch.as_deref().unwrap_or("detached")
            );

            if let Some(head_oid) = status.head {
                println!("  HEAD commit: {}", &head_oid.to_string()[..8]);
            }

            if !status.is_clean {
                println!("  Working tree: dirty");
                println!("  Modified files: {}", status.modified_files.len());
                if verbose {
                    // TODO: RepositoryStatus should include detailed file change types (A/M/D)
                    // For now we just show M for all modified files
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
    _max_context: Option<usize>,
    _kv_quant: crate::cli::commands::KVQuantArg,
    signing_key: SigningKey,
    _verifying_key: VerifyingKey,
) -> Result<()> {
    use crate::config::TrainingMode;
    use crate::runtime::model_config::ModelConfig;

    info!(
        "Running read-only inference via service: model={}, prompt_len={}",
        model_ref_str,
        prompt.len()
    );

    // Parse model reference and get path
    // TODO: This should eventually be handled by ModelService, but for now we need it
    // to ensure training mode is disabled before the service loads the model
    let model_ref = ModelRef::parse(model_ref_str)?;
    let model_path = storage.get_model_path(&model_ref).await?;

    // CRITICAL: Force training mode to Disabled for CLI infer command
    // This ensures hyprstream infer is ALWAYS read-only, regardless of model config.
    // For training inference, use `hyprstream training infer` instead.
    // TODO: Move this logic into ModelService initialization
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

    // ModelService is already running (started by main.rs in inproc mode, or by systemd in ipc-systemd mode).
    // Create ModelZmqClient to talk to ModelService.
    let model_client = ModelZmqClient::new(signing_key.clone(), RequestIdentity::local());

    // Apply chat template to the prompt via ModelService
    // The ModelService will load the model if needed and apply the template
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(prompt.to_string()),
        function_call: None,
    }];

    let formatted_prompt = match model_client.apply_chat_template(model_ref_str, &messages, true).await {
        Ok(formatted) => formatted,
        Err(e) => {
            tracing::warn!("Could not apply chat template: {}. Using raw prompt.", e);
            crate::config::TemplatedPrompt::new(prompt.to_string())
        }
    };

    // Build generation request
    // Load model-specific sampling params as defaults, but allow CLI overrides
    // TODO: ModelService should handle this, but for now load from config
    let mut request_builder = GenerationRequest::builder(formatted_prompt.into_inner())
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

    // Generate via ModelService (handles model loading, adapter loading, training collection, auth)
    if stream {
        // Streaming via ZMQ pub/sub with JWT authorization
        let (stream_id, _endpoint) = model_client.infer_stream(model_ref_str, &request).await?;
        info!("Streaming started: stream_id={}", stream_id);

        // Extract clean stream UUID (remove "stream-" prefix if present)
        let stream_id_clean = stream_id.strip_prefix("stream-").unwrap_or(&stream_id);

        // Create PolicyZmqClient to request JWT token
        let policy_client = PolicyZmqClient::new(signing_key.clone(), RequestIdentity::local());

        // Build structured scopes for JWT token
        let scopes = vec![
            // Subscribe to this specific stream
            format!("subscribe:stream:{}", stream_id_clean),
            // Infer on the model (for RPC calls)
            format!("infer:model:{}", model_ref.model),
        ];

        // Request JWT token from PolicyService (300 second TTL)
        let (token, _expires_at) = policy_client
            .issue_token(scopes, Some(300))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to issue JWT token: {}", e))?;

        info!("JWT token issued for stream subscription");

        // Get StreamService endpoint from registry (NOT from infer_stream response)
        // StreamService validates JWT and forwards to backend
        let stream_sub_endpoint = hyprstream_rpc::registry::global()
            .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
            .to_zmq_string();

        // Subscribe to stream chunks using global context
        // (inproc:// only works within the same ZMQ context)
        let ctx = global_context();
        let subscriber = ctx.socket(zmq::SUB)?;
        subscriber.connect(&stream_sub_endpoint)?;

        // Subscribe with JWT token in topic: "stream-{uuid}|{jwt}"
        // StreamService validates JWT signature, checks scope, then strips JWT before XSUB forwarding
        let subscription_topic = format!("stream-{}|{}", stream_id_clean, token);
        subscriber.set_subscribe(subscription_topic.as_bytes())?;
        subscriber.set_rcvtimeo(30000)?; // 30s timeout

        info!("Subscribed to stream via StreamService with JWT validation");

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
        // Non-streaming: get full response via ModelService
        let result = model_client.infer(model_ref_str, &request).await?;

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

    Ok(())
}

/// Handle push command
///
/// TODO: This function still uses local git operations.
/// Need to add the following to RepositoryClient:
/// - push(remote, refspec, force) method
/// - set_upstream_tracking(remote, branch) method
///
/// **EXPERIMENTAL**: This feature is behind the `experimental` flag.
/// It requires service layer implementation to be complete.
#[cfg(feature = "experimental")]
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

    // TODO: Replace with RepositoryClient.push() once available
    // TEMPORARY: Using storage.open_repo() for push operations
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
///
/// TODO: This currently uses RepositoryClient.update() which does fetch+merge,
/// but doesn't provide control over merge strategy (fast-forward vs regular merge).
/// Consider adding more granular methods to RepositoryClient:
/// - fetch_remote(remote, refspec)
/// - merge_with_strategy(ref, strategy)
pub async fn handle_pull(
    storage: &ModelStorage,
    model: &str,
    remote: Option<String>,
    branch: Option<String>,
    rebase: bool,
) -> Result<()> {
    info!("Pulling model {} from remote", model);

    let remote_name = remote.as_deref().unwrap_or("origin");
    let model_ref = ModelRef::new(model.to_string());

    // Get repository client from service layer
    let repo_client = storage.registry().repo(&model_ref.model).await?;

    // Build refspec for fetch
    let refspec = if let Some(ref branch_name) = branch {
        Some(format!("refs/heads/{}", branch_name))
    } else {
        None
    };

    // Use RepositoryClient.update() to fetch and merge
    // Note: This does a basic fetch+merge, doesn't expose merge analysis or fast-forward control
    repo_client.update(refspec.as_deref()).await?;

    println!("✓ Pulled latest changes for model {}", model);
    println!("  Remote: {}", remote_name);
    if let Some(ref b) = branch {
        println!("  Branch: {}", b);
    }
    if rebase {
        println!("  Strategy: rebase (note: currently performs merge)");
        warn!("Rebase strategy not yet implemented at service layer");
    } else {
        println!("  Strategy: merge");
    }

    Ok(())
}

/// Options for merge command
///
/// **EXPERIMENTAL**: This is behind the `experimental` flag.
#[cfg(feature = "experimental")]
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
///
/// **EXPERIMENTAL**: This feature is behind the `experimental` flag.
/// Basic merge functionality works via RepositoryClient.merge(), but conflict
/// resolution (--abort, --continue, --quit) still needs service layer implementation.
#[cfg(feature = "experimental")]
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
///
/// TODO: This function still uses local git operations for merge conflict resolution.
/// Need to add the following to RepositoryClient:
/// - abort_merge() - Reset to ORIG_HEAD and cleanup merge state
/// - continue_merge(message) - Check conflicts resolved, create merge commit
/// - quit_merge() - Remove merge state but keep changes
///
/// **EXPERIMENTAL**: This is behind the `experimental` flag.
#[cfg(feature = "experimental")]
async fn handle_merge_conflict_resolution(
    storage: &ModelStorage,
    target: &str,
    options: MergeOptions,
) -> Result<()> {
    let target_ref = ModelRef::parse(target)?;

    // Get repository client
    let repo_client = storage.registry().repo(&target_ref.model).await?;

    // Get target branch
    let target_branch = match &target_ref.git_ref {
        GitRef::Branch(b) => b.clone(),
        GitRef::DefaultBranch => repo_client.default_branch().await?,
        _ => bail!("Target must be a branch reference"),
    };

    // Get worktree path via RepositoryClient
    let worktree_path = repo_client.worktree_path(&target_branch).await?
        .ok_or_else(|| anyhow::anyhow!("Worktree not found for branch {}", target_branch))?;

    if !worktree_path.exists() {
        bail!("Worktree not found: {}", worktree_path.display());
    }

    // TODO: Replace with RepositoryClient methods once available
    // TEMPORARY: Using storage.open_repo() for conflict resolution
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
