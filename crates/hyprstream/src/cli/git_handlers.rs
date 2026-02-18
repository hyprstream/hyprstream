//! Handlers for git-style CLI commands
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use crate::config::GenerationRequest;
use crate::api::openai_compat::ChatMessage;
use crate::services::{ModelZmqClient, GenRegistryClient};
#[cfg(feature = "experimental")]
use crate::services::generated::registry_client::FileChangeTypeEnum;
use crate::zmq::global_context;
use crate::storage::ModelRef;
#[cfg(feature = "experimental")]
use crate::storage::GitRef;
use anyhow::{bail, Result};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::streaming::{StreamHandle, StreamPayload};
use hyprstream_rpc::crypto::generate_ephemeral_keypair;
use std::io::{self, Write};
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Handle branch command
pub async fn handle_branch(
    registry: &GenRegistryClient,
    model: &str,
    branch_name: &str,
    from_ref: Option<String>,
    policy_template: Option<String>,
) -> Result<()> {
    info!("Creating branch {} for model {}", branch_name, model);

    let tracked = registry.get_by_name(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    // Create branch via service
    repo_client.create_branch(branch_name, from_ref.as_deref().unwrap_or("")).await
        .map_err(|e| anyhow::anyhow!("Failed to create branch '{}': {}", branch_name, e))?;

    println!("✓ Created branch {branch_name}");

    if let Some(ref from) = from_ref {
        println!("  Branch created from: {from}");
    }

    // Create worktree for the branch via service
    let worktree_path = repo_client.create_worktree("", branch_name, false).await
        .map_err(|e| anyhow::anyhow!("Failed to create worktree: {}", e))?;
    println!("✓ Created worktree at {}", worktree_path);

    // Apply policy template if specified
    if let Some(ref template_name) = policy_template {
        apply_policy_template_to_model(registry, model, template_name).await?;
    }

    // Show helpful next steps
    println!("\n→ Next steps:");
    println!("  cd {}", worktree_path);
    println!("  hyprstream status {model}:{branch_name}");
    println!("  hyprstream lt {model}:{branch_name} --adapter my-adapter");

    Ok(())
}

/// Handle checkout command
pub async fn handle_checkout(
    registry: &GenRegistryClient,
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

    // Get repository client
    let tracked = registry.get_by_name(&model_ref.model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    // Check for uncommitted changes if not forcing
    if !force {
        let status = repo_client.status().await
            .map_err(|e| anyhow::anyhow!("Failed to get status: {}", e))?;
        if !status.is_clean {
            println!("Warning: Model has uncommitted changes");
            println!("Use --force to discard changes, or commit them first");
            return Ok(());
        }
    }

    // Get previous HEAD for result display
    let previous_oid = repo_client.get_head().await.unwrap_or_else(|_| "unknown".to_owned());

    // Convert GitRef to string for checkout
    let ref_spec = match &model_ref.git_ref {
        crate::storage::GitRef::Branch(name) => name.clone(),
        crate::storage::GitRef::Tag(name) => format!("refs/tags/{name}"),
        crate::storage::GitRef::Commit(oid) => oid.to_string(),
        crate::storage::GitRef::DefaultBranch => "HEAD".to_owned(),
        crate::storage::GitRef::Revspec(spec) => spec.clone(),
    };

    // If create_branch, create branch first
    if create_branch {
        if let crate::storage::GitRef::Branch(name) = &model_ref.git_ref {
            repo_client.create_branch(name, &previous_oid).await
                .map_err(|e| anyhow::anyhow!("Failed to create branch: {}", e))?;
        }
    }

    // Checkout via service layer
    repo_client.checkout(&ref_spec, false).await
        .map_err(|e| anyhow::anyhow!("Failed to checkout '{}': {}", ref_spec, e))?;

    // Get new HEAD
    let new_oid = repo_client.get_head().await.unwrap_or_else(|_| "unknown".to_owned());

    // Display checkout results
    let ref_display = model_ref.git_ref.to_string();
    println!("✓ Switched to {} ({})", ref_display, &new_oid[..8.min(new_oid.len())]);

    if force {
        println!("  ⚠️ Forced checkout - local changes discarded");
    }

    Ok(())
}

/// Handle status command
pub async fn handle_status(
    registry: &GenRegistryClient,
    model: Option<String>,
    verbose: bool,
) -> Result<()> {
    if let Some(model_ref_str) = model {
        // Status for specific model with full ModelRef support
        let model_ref = ModelRef::parse(&model_ref_str)?;
        let tracked = registry.get_by_name(&model_ref.model).await
            .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
        let repo_client = registry.repo(&tracked.id);
        let status = repo_client.status().await
            .map_err(|e| anyhow::anyhow!("Failed to get status: {}", e))?;
        print_model_status(&model_ref.model, &status, verbose);
    } else {
        // Status for all models - get list from registry
        let repos = registry.list().await
            .map_err(|e| anyhow::anyhow!("Failed to list repositories: {}", e))?;

        if repos.is_empty() {
            println!("No models found");
            return Ok(());
        }

        for tracked in repos {
            if !tracked.name.is_empty() {
                let name = &tracked.name;
                let repo_client = registry.repo(&tracked.id);
                if let Ok(status) = repo_client.status().await {
                    print_model_status(name, &status, verbose);
                    println!(); // Add spacing between models
                }
            }
        }
    }

    Ok(())
}

/// Handle commit command
///
/// Git status --short style single-character indicator.
#[cfg(feature = "experimental")]
fn status_char(s: FileChangeTypeEnum) -> char {
    match s {
        FileChangeTypeEnum::None => ' ',
        FileChangeTypeEnum::Added => 'A',
        FileChangeTypeEnum::Modified => 'M',
        FileChangeTypeEnum::Deleted => 'D',
        FileChangeTypeEnum::Renamed => 'R',
        FileChangeTypeEnum::Untracked => '?',
        FileChangeTypeEnum::TypeChanged => 'T',
        FileChangeTypeEnum::Conflicted => 'U',
    }
}

/// **EXPERIMENTAL**: This feature is behind the `experimental` flag.
#[cfg(feature = "experimental")]
pub async fn handle_commit(
    registry: &GenRegistryClient,
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

    // Get repository client
    let tracked = registry.get_by_name(&model_ref.model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    // Determine which branch to commit to
    let branch_name = match &model_ref.git_ref {
        crate::storage::GitRef::Branch(name) => name.clone(),
        crate::storage::GitRef::DefaultBranch => {
            repo_client.get_head().await
                .map_err(|e| anyhow::anyhow!("Failed to get default branch: {}", e))?
        }
        crate::storage::GitRef::Tag(tag) => {
            anyhow::bail!(
                "Cannot commit to a tag reference. Tags are immutable.\nTag: {}\nUse a branch instead: {}:main",
                tag, model_ref.model
            );
        }
        crate::storage::GitRef::Commit(oid) => {
            anyhow::bail!(
                "Cannot commit to a detached HEAD (commit reference).\nCommit: {}\nCheckout a branch first: hyprstream checkout {}:main",
                oid, model_ref.model
            );
        }
        crate::storage::GitRef::Revspec(spec) => {
            anyhow::bail!(
                "Cannot commit to a revspec reference. Revspecs are for querying history.\nRevspec: {}\nUse a branch instead: {}:main",
                spec, model_ref.model
            );
        }
    };

    // Check if worktree exists
    let wts = repo_client.list_worktrees().await?;
    let _worktree_path = wts.iter().find(|wt| wt.branch_name == branch_name).map(|wt| wt.path.clone())
        .ok_or_else(|| anyhow::anyhow!(
            "Worktree '{}' does not exist for model '{}'.\n\nCreate it first with:\n  hyprstream branch {} {}",
            branch_name, model_ref.model, model_ref.model, branch_name
        ))?;

    // Use RepositoryClient.detailed_status() for file-level change information
    let detailed_status = repo_client.detailed_status().await?;
    let has_changes = !detailed_status.files.is_empty();

    if !allow_empty && !amend && !has_changes && !all_untracked {
        println!("No changes to commit for {}:{}", model_ref.model, branch_name);
        println!("\nUse --allow-empty to create a commit without changes");
        return Ok(());
    }

    // Show what will be committed using detailed status
    if verbose || dry_run {
        println!("\n→ Changes to be committed:");

        if all || all_untracked {
            // Show all working tree changes
            for file in &detailed_status.files {
                println!("  {}{} {}", status_char(file.index_status), status_char(file.worktree_status), file.path);
            }
        } else {
            // Show only staged files (index)
            for file in detailed_status.files.iter().filter(|f| !matches!(f.index_status, FileChangeTypeEnum::None)) {
                println!("  {}  {}", status_char(file.index_status), file.path);
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

    // Stage files based on mode
    if all_untracked {
        // Stage all files including untracked (git add -A)
        info!("Staging all files including untracked");
        repo_client.stage_all_including_untracked().await?;
    } else if all {
        // Stage all tracked files only (git add -u)
        info!("Staging all tracked files");
        repo_client.stage_all().await?;
    }

    // Perform commit operation
    let commit_oid = if amend {
        info!("Amending previous commit");
        repo_client.amend_commit(message).await?
    } else if author.is_some() || author_name.is_some() || author_email.is_some() {
        // Parse author information
        let (name, email) = if let Some(author_str) = author {
            let re = regex::Regex::new(r"^(.+?)\s*<(.+?)>$")?;
            if let Some(captures) = re.captures(&author_str) {
                let name = captures.get(1)
                    .ok_or_else(|| anyhow::anyhow!("Invalid author format: missing name"))?
                    .as_str().trim().to_owned();
                let email = captures.get(2)
                    .ok_or_else(|| anyhow::anyhow!("Invalid author format: missing email"))?
                    .as_str().trim().to_owned();
                (name, email)
            } else {
                anyhow::bail!(
                    "Invalid author format. Expected: \"Name <email>\"\nGot: {}",
                    author_str
                );
            }
        } else {
            let name = author_name
                .ok_or_else(|| anyhow::anyhow!("--author-name required when using --author-email"))?;
            let email = author_email
                .ok_or_else(|| anyhow::anyhow!("--author-email required when using --author-name"))?;
            (name, email)
        };

        repo_client.commit_with_author(message, &name, &email).await?
    } else {
        // Simple commit
        repo_client.stage_all().await?;
        repo_client.commit(message).await?
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

/// Print model status in a nice format using generated RepositoryStatus
fn print_model_status(model_name: &str, status: &crate::services::GenRepositoryStatus, verbose: bool) {
    println!("Model: {model_name}");

    // Show current branch/commit
    if !status.branch.is_empty() {
        if !status.head_oid.is_empty() {
            println!("Current ref: {} ({})", status.branch, status.head_oid);
        } else {
            println!("Current ref: {}", status.branch);
        }
    } else if !status.head_oid.is_empty() {
        println!("Current ref: detached HEAD ({})", status.head_oid);
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
                println!("    M {}", file);
            }
        }
    } else {
        println!("Status: clean");
    }
}

/// Handle list command
pub async fn handle_list(
    registry: &GenRegistryClient,
    policy_manager: Option<&crate::auth::PolicyManager>,
) -> Result<()> {
    info!("Listing models");

    // Get list of tracked repositories from service
    let repos = registry.list().await
        .map_err(|e| anyhow::anyhow!("Failed to list repositories: {}", e))?;

    if repos.is_empty() {
        println!("No models found.");
        println!("Try: hyprstream clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct");
        return Ok(());
    }

    // Get current user for permission checks (OS user for CLI)
    let current_user = hyprstream_rpc::envelope::RequestIdentity::local().user().to_owned();

    // Get archetype registry for capability detection
    let archetype_registry = crate::archetypes::global_registry();

    // Collect models with git info and capabilities
    struct ModelInfo {
        display_name: String,
        git_ref: String,
        commit: String,
        is_dirty: bool,
        size_bytes: Option<u64>,
        domains: crate::archetypes::DetectedDomains,
        access_str: String,
    }

    let mut models_with_info = Vec::new();

    for tracked in repos {
        if !tracked.name.is_empty() {
            let name = &tracked.name;
            let repo_client = registry.repo(&tracked.id);

            // Get worktrees for this model
            let worktrees = match repo_client.list_worktrees().await {
                Ok(wts) => wts,
                Err(e) => {
                    warn!("Failed to list worktrees for '{}': {}", name, e);
                    continue;
                }
            };

            // Get commit hash via service layer
            let commit = match repo_client.get_head().await {
                Ok(head_oid) => head_oid.chars().take(7).collect::<String>(),
                Err(_) => "unknown".to_owned(),
            };

            for wt in worktrees {
                let branch_name = if wt.branch_name.is_empty() { "detached".to_owned() } else { wt.branch_name.clone() };
                let display_name = format!("{}:{}", name, branch_name);

                // Detect capabilities from worktree path (from service)
                let wt_path = std::path::Path::new(&wt.path);
                let detected = archetype_registry.detect(wt_path);
                let domains = detected.to_detected_domains();

                // Compute access string based on user permissions
                let resource = format!("model:{}", name);
                let access_str = if let Some(pm) = policy_manager {
                    crate::auth::capabilities_to_access_string(pm, &current_user, &resource, &domains.capabilities)
                } else {
                    // No policy manager = full access (show all capabilities)
                    domains.capabilities.to_ids().join(",")
                };

                models_with_info.push(ModelInfo {
                    display_name,
                    git_ref: branch_name,
                    commit: commit.clone(),
                    is_dirty: wt.is_dirty,
                    size_bytes: None, // Could be computed if needed
                    domains,
                    access_str,
                });
            }
        }
    }

    if models_with_info.is_empty() {
        println!("No models found.");
        println!("Try: hyprstream clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct");
        return Ok(());
    }

    // Table format with DOMAINS and ACCESS columns
    println!(
        "{:<30} {:<16} {:<16} {:<15} {:<8} {:<6} {:<10}",
        "MODEL NAME", "DOMAINS", "ACCESS", "REF", "COMMIT", "STATUS", "SIZE"
    );
    println!("{}", "-".repeat(111));

    for info in &models_with_info {
        let size_str = if let Some(size) = info.size_bytes {
            format!("{:.1}GB", size as f64 / (1024.0 * 1024.0 * 1024.0))
        } else {
            "n/a".to_owned()
        };

        let domains_str = info.domains.domains_display();
        let status = if info.is_dirty { "dirty" } else { "clean" };

        println!(
            "{:<30} {:<16} {:<16} {:<15} {:<8} {:<6} {:<10}",
            info.display_name, domains_str, info.access_str, info.git_ref, info.commit, status, size_str
        );
    }

    Ok(())
}

/// Handle clone command with streaming progress
pub async fn handle_clone(
    registry: &GenRegistryClient,
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
        println!("📦 Cloning model from: {repo_url}");

        if let Some(ref b) = branch {
            println!("   Branch: {b}");
        }

        if full {
            println!("   Mode: Full history");
        } else if depth > 0 {
            println!("   Depth: {depth} commits");
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
            .trim_end_matches(".git").to_owned();

        if extracted.is_empty() {
            anyhow::bail!(
                "Cannot derive model name from URL '{}'. Please provide --name.",
                repo_url
            );
        }
        extracted
    };

    // Determine clone parameters
    let shallow = !full;
    let clone_depth = if full { 0 } else { depth };

    // Try streaming clone with progress display
    let clone_result = clone_with_streaming(registry, repo_url, &model_name, shallow, clone_depth, branch.as_deref(), quiet, verbose).await;

    // If streaming fails, fall back to non-streaming clone
    if let Err(e) = clone_result {
        if verbose {
            warn!("Streaming clone failed, falling back to non-streaming: {}", e);
        }
        registry.clone(repo_url, &model_name, shallow, clone_depth, branch.as_deref().unwrap_or("")).await
            .map_err(|e| anyhow::anyhow!("Failed to clone model: {}", e))?;
    }

    // Get repo client for worktree creation
    let tracked = registry.get_by_name(&model_name).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    // Create default worktree so the model appears in list
    // Use requested branch or fall back to default branch (usually "main")
    let worktree_branch = if let Some(ref b) = branch {
        b.clone()
    } else {
        repo_client.get_head().await
            .unwrap_or_else(|_| "main".to_owned())
    };

    if !quiet {
        println!("   Creating worktree for branch: {worktree_branch}");
    }
    let worktree_path = repo_client.create_worktree("", &worktree_branch, false).await
        .map_err(|e| anyhow::anyhow!("Failed to create worktree: {}", e))?;

    if !quiet {
        println!("✅ Model '{}' cloned successfully!", model_name);
        println!("   Location: {}", worktree_path);
    }

    // Apply policy template if specified
    if let Some(ref template_name) = policy_template {
        apply_policy_template_to_model(registry, &model_name, template_name).await?;
    }

    Ok(())
}

/// Clone with streaming progress display.
///
/// Uses DH-authenticated streaming to receive clone progress updates.
async fn clone_with_streaming(
    registry: &GenRegistryClient,
    repo_url: &str,
    model_name: &str,
    shallow: bool,
    depth: u32,
    branch: Option<&str>,
    quiet: bool,
    verbose: bool,
) -> Result<()> {
    // Generate ephemeral keypair for DH key exchange
    let (client_secret, client_pubkey) = generate_ephemeral_keypair();
    let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

    // Call clone_stream to initiate streaming clone
    let stream_info = registry
        .clone_stream(repo_url, model_name, shallow, depth, branch.unwrap_or(""), client_pubkey_bytes)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start streaming clone: {}", e))?;

    if verbose {
        debug!(
            stream_id = %stream_info.stream_id,
            endpoint = %stream_info.endpoint,
            "Started streaming clone"
        );
    }

    // Create stream handle for receiving progress
    let ctx = global_context();
    let mut stream_handle = StreamHandle::new(
        &ctx,
        stream_info.stream_id.clone(),
        &stream_info.endpoint,
        &stream_info.server_pubkey,
        &client_secret,
        &client_pubkey_bytes,
    )?;

    if !quiet {
        print!("   Progress: ");
        io::stdout().flush()?;
    }

    // Receive and display progress
    loop {
        match stream_handle.recv_next()? {
            Some(StreamPayload::Data(data)) => {
                // Parse progress message (format: "stage:current:total")
                if let Ok(text) = String::from_utf8(data) {
                    if !quiet {
                        // Parse simple progress format
                        let parts: Vec<&str> = text.split(':').collect();
                        if parts.len() >= 3 {
                            let stage = parts[0];
                            if verbose {
                                println!("\r   {}: {}/{}", stage, parts[1], parts[2]);
                            } else {
                                print!(".");
                                io::stdout().flush()?;
                            }
                        } else {
                            // Plain message
                            if verbose {
                                println!("\r   {}", text);
                            }
                        }
                    }
                }
            }
            Some(StreamPayload::Complete(_metadata)) => {
                if !quiet {
                    println!(" done");
                }
                break;
            }
            Some(StreamPayload::Error(message)) => {
                if !quiet {
                    println!(" error");
                }
                return Err(anyhow::anyhow!("Clone stream error: {}", message));
            }
            None => {
                // Stream ended without completion
                if !quiet {
                    println!(" done");
                }
                break;
            }
        }
    }

    Ok(())
}

/// Handle info command
///
/// TODO: Adapter listing still uses local filesystem access. Consider moving to ModelService.
pub async fn handle_info(
    registry: &GenRegistryClient,
    model: &str,
    verbose: bool,
    adapters_only: bool,
) -> Result<()> {
    info!("Getting info for model {}", model);

    let model_ref = ModelRef::parse(model)?;

    // Get repository client from service layer
    let tracked_repo = registry.get_by_name(&model_ref.model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked_repo.id);

    // Get repository metadata via RepositoryClient
    let repo_metadata = {
        // Get remote URL
        let remotes = repo_client.list_remotes().await.ok();
        let url = remotes
            .as_ref()
            .and_then(|r| r.iter().find(|remote| remote.name == "origin"))
            .map(|remote| remote.url.clone())
            .unwrap_or_else(|| "unknown".to_owned());

        // Get current HEAD
        let current_oid = repo_client.get_head().await.ok();

        Some((
            Some(model_ref.model.clone()),
            url,
            model_ref.git_ref.to_string(),
            current_oid,
        ))
    };

    // Get model path from worktree info via service
    let worktrees = repo_client.list_worktrees().await
        .map_err(|e| anyhow::anyhow!("Failed to list worktrees: {}", e))?;

    // Find the worktree matching the requested branch/ref
    let branch_name = match &model_ref.git_ref {
        crate::storage::GitRef::DefaultBranch => {
            repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned())
        }
        crate::storage::GitRef::Branch(b) => b.clone(),
        _ => "main".to_owned(),
    };

    let model_path = worktrees.iter()
        .find(|wt| wt.branch_name == branch_name)
        .map(|wt| PathBuf::from(&wt.path))
        .unwrap_or_else(|| {
            // Fallback: use first available worktree
            worktrees.first()
                .map(|wt| PathBuf::from(&wt.path))
                .unwrap_or_else(|| PathBuf::from("."))
        });

    // If adapters_only is true, skip the general model info
    if !adapters_only {
        println!("Model: {}", model_ref.model);

        // Show git2db metadata if available
        if let Some((name, url, tracking_ref, current_oid)) = &repo_metadata {
            if let Some(n) = name {
                if n != &model_ref.model {
                    println!("  Registry name: {n}");
                }
            }
            println!("  Origin URL: {url}");
            println!("  Tracking ref: {tracking_ref}");
            if let Some(oid) = current_oid {
                println!("  Current OID: {}", &oid[..8.min(oid.len())]);
            }
        }

        // Get display ref using RepositoryClient
        let display_ref = match &model_ref.git_ref {
            crate::storage::GitRef::DefaultBranch => {
                repo_client.get_head().await.unwrap_or_else(|_| "unknown".to_owned())
            }
            _ => model_ref.git_ref.to_string(),
        };

        println!("Reference: {display_ref}");
        println!("Path: {}", model_path.display());

        // Detect and display archetypes
        let archetype_registry = crate::archetypes::global_registry();
        let detected = archetype_registry.detect(&model_path);

        if !detected.is_empty() {
            println!("\nArchetypes:");
            for archetype in &detected.archetypes {
                println!("  - {archetype}");
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

    // Get tags
    if let Ok(tags) = repo_client.list_tags().await {
        if !tags.is_empty() {
            println!("  Tags: {}", tags.join(", "));
        }
    }

    // Size calculation - this could be moved to RepositoryClient as get_repo_size()
    // Get bare repo path from TrackedRepository (already fetched above)
    let bare_repo_path = Some(PathBuf::from(&tracked_repo.worktree_path));

    if let Some(ref bare_repo_path) = bare_repo_path {
        if bare_repo_path.exists() {
            if let Ok(metadata) = std::fs::metadata(bare_repo_path) {
                if metadata.is_dir() {
                    let mut total_size = 0u64;
                    if let Ok(entries) = walkdir::WalkDir::new(bare_repo_path)
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
    }

    // Get git status via RepositoryClient
    println!("\nWorktree Status:");

    match repo_client.status().await {
        Ok(status) => {
            println!(
                "  Current branch/ref: {}",
                if status.branch.is_empty() { "detached" } else { &status.branch }
            );

            if !status.head_oid.is_empty() {
                println!("  HEAD commit: {}", &status.head_oid[..8.min(status.head_oid.len())]);
            }

            if !status.is_clean {
                println!("  Working tree: dirty");
                println!("  Modified files: {}", status.modified_files.len());
                if verbose {
                    // TODO: RepositoryStatus should include detailed file change types (A/M/D)
                    // For now we just show M for all modified files
                    for file in &status.modified_files {
                        println!("    M {}", file);
                    }
                }
            } else {
                println!("  Working tree: clean");
            }
        }
        Err(e) => {
            println!("  Unable to get status: {e}");
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
            println!("  Files: {file_count}");
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
                println!("\nAdapters: Error listing adapters: {e}");
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
    registry: &GenRegistryClient,
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

    // Get the policies directory - derive models_dir from TrackedRepository
    let tracked = registry.get_by_name(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository: {}", e))?;

    // TrackedRepository.worktree_path is bare repo: models/{name}/{name}.git
    // Navigate up 2 levels to get models_dir
    let bare_repo_path = PathBuf::from(&tracked.worktree_path);
    let models_dir = bare_repo_path
        .parent()  // models/{name}
        .and_then(|p| p.parent())  // models/
        .ok_or_else(|| anyhow::anyhow!("Invalid repository path structure"))?;

    let registry_path = models_dir.join(".registry");
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
        default_policy_header().to_owned()
    };

    // Check if template rules already exist
    let rules = template.expanded_rules();
    if existing_content.contains(rules.trim()) {
        println!("✓ Policy template '{template_name}' already applied");
        return Ok(());
    }

    // Append the template rules
    let new_content = format!("{}\n{}", existing_content.trim_end(), rules);

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
    let commit_msg = format!("policy: apply {template_name} template for model {model}");

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

    println!("✓ Applied policy template: {template_name}");
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
pub async fn handle_infer(
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
    signing_key: SigningKey,
) -> Result<()> {
    info!(
        "Running inference via ModelService: model={}, prompt_len={}",
        model_ref_str,
        prompt.len()
    );

    // Validate model reference format
    let _ = ModelRef::parse(model_ref_str)?;

    // ModelService is already running (started by main.rs in inproc mode, or by systemd in ipc-systemd mode).
    let model_client = ModelZmqClient::new(signing_key.clone(), RequestIdentity::local());

    // Apply chat template to the prompt via ModelService
    let messages = vec![ChatMessage {
        role: "user".to_owned(),
        content: Some(prompt.to_owned()),
        function_call: None,
        tool_calls: None,
        tool_call_id: None,
    }];

    let formatted_prompt = match model_client.apply_chat_template(model_ref_str, &messages, true, None).await {
        Ok(formatted) => formatted,
        Err(e) => {
            tracing::warn!("Could not apply chat template: {}. Using raw prompt.", e);
            crate::config::TemplatedPrompt::new(prompt.to_owned())
        }
    };

    // Build generation request with CLI overrides (ModelService applies model defaults)
    let mut request_builder = GenerationRequest::builder(formatted_prompt.into_inner())
        .max_tokens(max_tokens.unwrap_or(2048));

    // Apply CLI overrides only if specified
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
    if let Some(s) = seed {
        request_builder = request_builder.seed(Some(s as u64));
    }

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
        // Generate client ephemeral keypair for DH key exchange
        let (client_secret, client_pubkey) = generate_ephemeral_keypair();
        let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

        // Start stream with client ephemeral pubkey
        let stream_info = model_client.infer_stream(model_ref_str, &request, client_pubkey_bytes).await?;

        debug!(
            stream_id = %stream_info.stream_id,
            endpoint = %stream_info.endpoint,
            "Started inference stream"
        );

        // Create stream handle for receiving tokens (handles DH, subscription, verification)
        let ctx = global_context();
        let mut stream_handle = StreamHandle::new(
            &ctx,
            stream_info.stream_id.clone(),
            &stream_info.endpoint,
            &stream_info.server_pubkey,
            &client_secret,
            &client_pubkey_bytes,
        )?;

        println!();

        // Receive and print tokens
        loop {
            match stream_handle.recv_next()? {
                Some(StreamPayload::Data(data)) => {
                    // Token data is UTF-8 text
                    if let Ok(text) = String::from_utf8(data) {
                        print!("{text}");
                        io::stdout().flush()?;
                    }
                }
                Some(StreamPayload::Complete(_metadata)) => {
                    // Stream completed successfully
                    break;
                }
                Some(StreamPayload::Error(message)) => {
                    warn!("Stream error: {}", message);
                    bail!("Inference stream error: {}", message);
                }
                None => {
                    // Stream ended
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

/// Handle load command - pre-load a model with optional runtime config
pub async fn handle_load(
    model_ref_str: &str,
    max_context: Option<usize>,
    kv_quant: crate::runtime::kv_quant::KVQuantType,
    signing_key: SigningKey,
) -> Result<()> {
    use crate::services::model::ModelLoadConfig;

    info!("Loading model: {}", model_ref_str);

    // Validate model reference format
    let _ = ModelRef::parse(model_ref_str)?;

    let model_client = ModelZmqClient::new(signing_key, RequestIdentity::local());

    // Build config if any options specified
    let config = if max_context.is_some() || kv_quant != crate::runtime::kv_quant::KVQuantType::None {
        Some(ModelLoadConfig {
            max_context,
            kv_quant: if kv_quant == crate::runtime::kv_quant::KVQuantType::None {
                None
            } else {
                Some(kv_quant)
            },
        })
    } else {
        None
    };

    let endpoint = model_client.load(model_ref_str, config.as_ref()).await?;

    println!("✓ Model {} loaded", model_ref_str);
    println!("  Endpoint: {}", endpoint);
    if let Some(max_ctx) = max_context {
        println!("  Max context: {}", max_ctx);
    }
    if kv_quant != crate::runtime::kv_quant::KVQuantType::None {
        println!("  KV quantization: {:?}", kv_quant);
    }

    Ok(())
}

/// Handle unload command - unload a model from memory
pub async fn handle_unload(
    model_ref_str: &str,
    signing_key: SigningKey,
) -> Result<()> {
    info!("Unloading model: {}", model_ref_str);

    // Validate model reference format
    let _ = ModelRef::parse(model_ref_str)?;

    let model_client = ModelZmqClient::new(signing_key, RequestIdentity::local());

    model_client.unload(model_ref_str).await?;

    println!("✓ Model {} unloaded", model_ref_str);

    Ok(())
}

/// Handle push command
///
/// **EXPERIMENTAL**: This feature is behind the `experimental` flag.
#[cfg(feature = "experimental")]
pub async fn handle_push(
    registry: &GenRegistryClient,
    model: &str,
    remote: Option<String>,
    branch: Option<String>,
    set_upstream: bool,
    force: bool,
) -> Result<()> {
    info!("Pushing model {} to remote", model);

    let remote_name = remote.as_deref().unwrap_or("origin");
    let model_ref = ModelRef::new(model.to_string());

    // Get repository client from service layer
    let tracked = registry.get_by_name(&model_ref.model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    // Get current branch or specified branch
    let push_branch = if let Some(b) = &branch {
        b.clone()
    } else {
        // Get default branch from repository client
        repo_client.get_head().await?
    };

    // Build refspec
    let refspec = format!("refs/heads/{}:refs/heads/{}", push_branch, push_branch);

    // Push via RepositoryClient
    repo_client.push(remote_name, &refspec, force).await?;

    println!("✓ Pushed model {} to {}", model, remote_name);
    println!("  Branch: {}", push_branch);
    if force {
        println!("  ⚠️  Force push");
    }
    if set_upstream {
        println!("  Note: Upstream tracking is automatically configured by git");
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
    registry: &GenRegistryClient,
    model: &str,
    remote: Option<String>,
    branch: Option<String>,
    rebase: bool,
) -> Result<()> {
    info!("Pulling model {} from remote", model);

    let remote_name = remote.as_deref().unwrap_or("origin");
    let model_ref = ModelRef::new(model.to_owned());

    // Get repository client from service layer
    let tracked = registry.get_by_name(&model_ref.model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    // Build refspec for fetch
    let refspec = branch.as_ref().map(|branch_name| format!("refs/heads/{branch_name}"));

    // Use RepositoryClient.update() to fetch and merge
    // Note: This does a basic fetch+merge, doesn't expose merge analysis or fast-forward control
    repo_client.update(refspec.as_deref().unwrap_or("")).await?;

    println!("✓ Pulled latest changes for model {model}");
    println!("  Remote: {remote_name}");
    if let Some(ref b) = branch {
        println!("  Branch: {b}");
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
    registry: &GenRegistryClient,
    target: &str,
    source: &str,
    options: MergeOptions,
) -> Result<()> {
    // Handle conflict resolution modes first
    if options.abort || options.continue_merge || options.quit {
        return handle_merge_conflict_resolution(registry, target, options).await;
    }

    // Parse target ModelRef (e.g., "Qwen3-4B:branch3")
    let target_ref = ModelRef::parse(target)?;

    if !options.quiet {
        info!("Merging '{}' into '{}'", source, target_ref);
    }

    // Get repository client from service layer
    let tracked = registry.get_by_name(&target_ref.model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

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
/// **EXPERIMENTAL**: This is behind the `experimental` flag.
#[cfg(feature = "experimental")]
async fn handle_merge_conflict_resolution(
    registry: &GenRegistryClient,
    target: &str,
    options: MergeOptions,
) -> Result<()> {
    let target_ref = ModelRef::parse(target)?;

    // Get repository client from service layer
    let tracked = registry.get_by_name(&target_ref.model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    // Get target branch
    let target_branch = match &target_ref.git_ref {
        GitRef::Branch(b) => b.clone(),
        GitRef::DefaultBranch => repo_client.get_head().await?,
        _ => bail!("Target must be a branch reference"),
    };

    // Verify worktree exists
    let wts = repo_client.list_worktrees().await?;
    let worktree_path = wts.iter().find(|wt| wt.branch_name == target_branch).map(|wt| wt.path.clone())
        .ok_or_else(|| anyhow::anyhow!("Worktree not found for branch {}", target_branch))?;

    let worktree_path_buf = PathBuf::from(&worktree_path);
    if !worktree_path_buf.exists() {
        bail!("Worktree not found: {}", worktree_path);
    }

    if options.abort {
        // Abort merge: restore pre-merge state
        if !options.quiet {
            println!("→ Aborting merge...");
        }

        repo_client.abort_merge().await?;

        if !options.quiet {
            println!("✓ Merge aborted, restored pre-merge state");
        }
    } else if options.continue_merge {
        // Continue merge: check if conflicts are resolved and create merge commit
        if !options.quiet {
            println!("→ Continuing merge...");
        }

        let merge_oid = repo_client.continue_merge(options.message.as_deref()).await?;

        if !options.quiet {
            println!("✓ Merge completed successfully");
            if options.verbose {
                println!("  Commit: {}", &merge_oid[..8.min(merge_oid.len())]);
            }
        }
    } else if options.quit {
        // Quit merge: keep working tree changes but remove merge state
        if !options.quiet {
            println!("→ Quitting merge (keeping changes)...");
        }

        repo_client.quit_merge().await?;

        if !options.quiet {
            println!("✓ Merge state removed, changes retained");
            println!("  Use 'hyprstream status {}' to see modified files", target);
        }
    }

    Ok(())
}

/// Handle remove command
///
/// Removal is handled by the RegistryService which manages both
/// registry entries and file cleanup.
pub async fn handle_remove(
    registry: &GenRegistryClient,
    model: &str,
    force: bool,
    _registry_only: bool,
    _files_only: bool,
) -> Result<()> {
    info!("Removing model {}", model);

    // Parse model reference to handle model:branch format
    let model_ref = ModelRef::parse(model)?;

    // Check if a specific branch/worktree was specified
    let is_worktree_removal = !matches!(model_ref.git_ref, crate::storage::GitRef::DefaultBranch);

    if is_worktree_removal {
        // Removing a specific worktree, not the entire model
        let branch = model_ref.git_ref.display_name();
        info!("Removing worktree {} for model {}", branch, model_ref.model);

        let tracked = registry.get_by_name(&model_ref.model).await
            .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
        let repo_client = registry.repo(&tracked.id);

        // Check if the worktree exists via service
        let worktrees = repo_client.list_worktrees().await
            .map_err(|e| anyhow::anyhow!("Failed to list worktrees: {}", e))?;

        let wt = match worktrees.iter().find(|wt| wt.branch_name == branch) {
            Some(wt) => wt,
            None => {
                println!("❌ Worktree '{}' not found for model '{}'", branch, model_ref.model);
                return Ok(());
            }
        };

        // Show what will be removed
        println!("Worktree '{}:{}' removal plan:", model_ref.model, branch);
        println!("  📁 Remove worktree at: {}", wt.path);

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

        // Remove the worktree via service
        repo_client.remove_worktree(&wt.path, false).await
            .map_err(|e| anyhow::anyhow!("Failed to remove worktree: {}", e))?;

        println!("✓ Worktree '{}:{}' removed successfully", model_ref.model, branch);
        return Ok(());
    }

    // Removing the entire model (no branch specified)
    // Check if model exists in registry
    let tracked = match registry.get_by_name(&model_ref.model).await {
        Ok(t) => t,
        Err(e) => {
            let err_msg = e.to_string();
            if err_msg.contains("not found") || err_msg.contains("Not found") {
                println!("❌ Model '{}' not found in registry", model_ref.model);
                return Ok(());
            }
            return Err(anyhow::anyhow!("Failed to query registry: {}", e));
        }
    };

    // Show what will be removed
    println!("Model '{}' removal plan:", model_ref.model);
    println!("  🗂️  Remove from registry and all associated files");

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

    // Remove via service - service handles registry and file cleanup
    registry.remove(&tracked.id).await
        .map_err(|e| anyhow::anyhow!("Failed to remove model: {}", e))?;

    println!("✓ Model '{}' removed successfully", model_ref.model);
    Ok(())
}
