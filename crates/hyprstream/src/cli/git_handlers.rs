//! Handlers for git-style CLI commands
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use crate::runtime::GenerationRequest;
use crate::services::generated::inference_client::ChatMessage;
use crate::services::generated::model_client::ChatTemplateRequest;
use crate::services::generated::notification_client::{SubscribeRequest, UnsubscribeRequest};
use crate::services::generated::registry_client::{
    BranchRequest, CheckoutRequest, CloneRequest, CreateWorktreeRequest,
    RemoveWorktreeRequest, UpdateRequest,
};
use crate::services::RegistryClient;
use crate::services::generated::model_client::{ModelClient, LoadModelRequest, UnloadModelRequest, StatusRequest};
#[cfg(feature = "experimental")]
use crate::services::generated::registry_client::FileChangeType;
use crate::zmq::global_context;
use crate::storage::ModelRef;
#[cfg(feature = "experimental")]
use crate::storage::GitRef;
use anyhow::{bail, Result};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::streaming::StreamPayload;
use hyprstream_rpc::crypto::generate_ephemeral_keypair;
use std::io::{self, Write};
use std::path::PathBuf;
use tmq::SocketExt;
use tracing::{debug, info, warn};

/// Handle branch command
pub async fn handle_branch(
    registry: &RegistryClient,
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
    repo_client.create_branch(&BranchRequest {
        branch_name: branch_name.to_owned(),
        start_point: from_ref.as_deref().unwrap_or("").to_owned(),
    }).await
        .map_err(|e| anyhow::anyhow!("Failed to create branch '{}': {}", branch_name, e))?;

    println!("✓ Created branch {branch_name}");

    if let Some(ref from) = from_ref {
        println!("  Branch created from: {from}");
    }

    // Create worktree for the branch via service
    repo_client.create_worktree(&CreateWorktreeRequest {
        branch: branch_name.to_owned(),
    }).await
        .map_err(|e| anyhow::anyhow!("Failed to create worktree: {}", e))?;
    println!("✓ Created worktree for branch {branch_name}");

    // Apply policy template if specified
    if let Some(ref template_name) = policy_template {
        apply_policy_template_to_model(registry, model, template_name).await?;
    }

    // Show helpful next steps
    println!("\n→ Next steps:");
    println!("  hyprstream worktree info {model} {branch_name}");
    println!("  hyprstream status {model}:{branch_name}");
    println!("  hyprstream lt {model}:{branch_name} --adapter my-adapter");

    Ok(())
}

/// Handle checkout command
pub async fn handle_checkout(
    registry: &RegistryClient,
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
            repo_client.create_branch(&BranchRequest {
                branch_name: name.to_owned(),
                start_point: previous_oid.clone(),
            }).await
                .map_err(|e| anyhow::anyhow!("Failed to create branch: {}", e))?;
        }
    }

    // Checkout via worktree-scoped service layer
    // Determine which worktree to operate on
    let worktree_name = match &model_ref.git_ref {
        crate::storage::GitRef::Branch(name) => name.clone(),
        _ => tracked.tracking_ref.clone(),
    };
    repo_client.worktree(&worktree_name).checkout(&CheckoutRequest {
        ref_name: ref_spec.clone(),
        create_branch: false,
    }).await
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
    registry: &RegistryClient,
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
fn status_char(s: FileChangeType) -> char {
    match s {
        FileChangeType::None => ' ',
        FileChangeType::Added => 'A',
        FileChangeType::Modified => 'M',
        FileChangeType::Deleted => 'D',
        FileChangeType::Renamed => 'R',
        FileChangeType::Untracked => '?',
        FileChangeType::TypeChanged => 'T',
        FileChangeType::Conflicted => 'U',
    }
}

/// **EXPERIMENTAL**: This feature is behind the `experimental` flag.
#[cfg(feature = "experimental")]
pub async fn handle_commit(
    registry: &RegistryClient,
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
    if !wts.iter().any(|wt| wt.branch_name == branch_name) {
        anyhow::bail!(
            "Worktree '{}' does not exist for model '{}'.\n\nCreate it first with:\n  hyprstream branch {} {}",
            branch_name, model_ref.model, model_ref.model, branch_name
        );
    }

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
            for file in detailed_status.files.iter().filter(|f| !matches!(f.index_status, FileChangeType::None)) {
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

    // Use worktree-scoped client for all staging/commit operations
    let wt = repo_client.worktree(&branch_name);

    // Stage files based on mode
    if all_untracked {
        // Stage all files including untracked (git add -A)
        info!("Staging all files including untracked");
        wt.stage_all_including_untracked().await?;
    } else if all {
        // Stage all tracked files only (git add -u)
        info!("Staging all tracked files");
        wt.stage_all().await?;
    }

    // Perform commit operation
    let commit_oid = if amend {
        info!("Amending previous commit");
        wt.amend_commit(message).await?
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

        wt.commit_with_author(message, &name, &email).await?
    } else {
        // Simple commit
        wt.stage_all().await?;
        wt.commit(message).await?
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

/// Parse `--filter key=regex` strings into compiled `(key, Regex)` pairs.
///
/// Patterns are unanchored by default. Use `^`/`$` to anchor, `(?i)` for
/// case-insensitive, `|` for alternation. `status` is not a valid filter key —
/// use `--status`/`-s` instead.
///
/// Returns an error for unknown keys or invalid regex patterns.
pub fn parse_filters(raw: &[String]) -> Result<Vec<(String, regex::Regex)>> {
    const VALID_KEYS: &[&str] = &["name", "domains", "access", "ref", "commit", "size"];
    let mut out = Vec::with_capacity(raw.len());
    for s in raw {
        let (k, v) = s.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("--filter '{}': expected KEY=REGEX format", s)
        })?;
        let key = k.to_lowercase();
        if key == "status" {
            anyhow::bail!(
                "--filter '{}': 'status' cannot be used with --filter. Use --status/-s instead (e.g. -s loaded,loading)",
                s
            );
        }
        if !VALID_KEYS.contains(&key.as_str()) {
            anyhow::bail!(
                "--filter '{}': unknown column '{}'. Valid columns: {}",
                s, key, VALID_KEYS.join(", ")
            );
        }
        let re = regex::Regex::new(v).map_err(|e| {
            anyhow::anyhow!("--filter '{}': invalid regex pattern '{}': {}", s, v, e)
        })?;
        out.push((key, re));
    }
    Ok(out)
}

/// Parse `--status`/`-s` values into a `HashSet` of accepted status strings.
///
/// Each argument may be a single value or comma-separated list. Multiple `-s`
/// flags are OR'd together. Empty result means no status filter (all statuses shown).
///
/// Valid values: `loaded`, `loading`, `unloaded`.
pub fn parse_status_filter(raw: &[String]) -> Result<std::collections::HashSet<String>> {
    const VALID: &[&str] = &["loaded", "loading", "unloaded"];
    let mut out = std::collections::HashSet::new();
    for s in raw {
        for part in s.split(',') {
            let val = part.trim().to_lowercase();
            if val.is_empty() { continue; }
            if !VALID.contains(&val.as_str()) {
                anyhow::bail!(
                    "--status '{}': unknown status '{}'. Valid values: {}",
                    s, val, VALID.join(", ")
                );
            }
            out.insert(val);
        }
    }
    Ok(out)
}

pub async fn handle_list(
    registry: &RegistryClient,
    model_client: ModelClient,
    filters: &[(String, regex::Regex)],
    status_filter: &std::collections::HashSet<String>,
) -> Result<()> {
    info!("Listing models");

    // Fetch registry list and model service status in parallel
    let all_status_req = StatusRequest { model_ref: String::new() };
    let (repos_result, status_result) = tokio::join!(
        registry.list(),
        model_client.status(&all_status_req),
    );

    let repos = repos_result
        .map_err(|e| anyhow::anyhow!("Failed to list repositories: {}", e))?;

    if repos.is_empty() {
        println!("No models found.");
        println!("Try: hyprstream clone https://huggingface.co/Qwen/Qwen3-0.6B");
        return Ok(());
    }

    // Build status lookup: model_ref -> status string
    let status_map: std::collections::HashMap<String, String> = match status_result {
        Ok(entries) => entries.into_iter().map(|e| (e.model_ref, e.status)).collect(),
        Err(e) => {
            warn!("Model service unavailable for status: {}", e);
            std::collections::HashMap::new()
        }
    };

    struct RowInfo {
        display_name: String,
        domains_str: String,
        access_str: String,
        git_ref: String,
        commit: String,
        load_status: String,
        size_str: String,
    }

    let mut rows = Vec::new();

    for tracked in repos {
        if tracked.name.is_empty() { continue; }
        let name = &tracked.name;

        for wt in &tracked.worktrees {
            let branch_name = if wt.branch_name.is_empty() { "detached".to_owned() } else { wt.branch_name.clone() };
            let display_name = format!("{}:{}", name, branch_name);
            let commit = wt.head_oid.chars().take(7).collect::<String>();

            // Capabilities come from the server — no local filesystem access needed
            let domains_str = if wt.capabilities.is_empty() {
                "n/a".to_owned()
            } else {
                wt.capabilities.join(",")
            };
            // All capabilities returned are already policy-filtered by the server
            let access_str = domains_str.clone();

            // Load status: join with model service response
            let model_ref = format!("{}:{}", name, branch_name);
            let load_status = status_map
                .get(&model_ref)
                .cloned()
                .unwrap_or_else(|| "unloaded".to_owned());

            let size_str = "n/a".to_owned();

            rows.push(RowInfo {
                display_name,
                domains_str,
                access_str,
                git_ref: branch_name,
                commit,
                load_status,
                size_str,
            });
        }
    }

    // Apply --status filter (OR semantics: row must match one of the specified statuses)
    let rows: Vec<_> = if status_filter.is_empty() {
        rows
    } else {
        rows.into_iter()
            .filter(|r| status_filter.contains(r.load_status.as_str()))
            .collect()
    };

    // Apply --filter predicates (AND semantics, unanchored regex match)
    let rows: Vec<_> = rows.into_iter().filter(|r| {
        filters.iter().all(|(key, re)| {
            let haystack = match key.as_str() {
                "name"    => &r.display_name,
                "domains" => &r.domains_str,
                "access"  => &r.access_str,
                "ref"     => &r.git_ref,
                "commit"  => &r.commit,
                "size"    => &r.size_str,
                _         => return true,
            };
            re.is_match(haystack)
        })
    }).collect();

    if rows.is_empty() {
        println!("No models match the given filters.");
        return Ok(());
    }

    println!(
        "{:<35} {:<16} {:<15} {:<8} {:<10}",
        "MODEL NAME", "DOMAINS", "REF", "COMMIT", "STATUS"
    );
    println!("{}", "-".repeat(90));

    for r in &rows {
        println!(
            "{:<35} {:<16} {:<15} {:<8} {:<10}",
            r.display_name, r.domains_str, r.git_ref, r.commit, r.load_status
        );
    }

    Ok(())
}

/// Handle clone command with streaming progress
#[allow(clippy::too_many_arguments)]
pub async fn handle_clone(
    registry: &RegistryClient,
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
        registry.clone(&CloneRequest {
            url: repo_url.to_owned(),
            name: model_name.clone(),
            shallow,
            depth: clone_depth,
            branch: branch.as_deref().unwrap_or("").to_owned(),
        }).await
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
    } else if !tracked.tracking_ref.is_empty() {
        tracked.tracking_ref.clone()
    } else {
        "main".to_owned()
    };

    if !quiet {
        println!("   Creating worktree for branch: {worktree_branch}");
    }
    repo_client.create_worktree(&CreateWorktreeRequest {
        branch: worktree_branch.clone(),
    }).await
        .map_err(|e| anyhow::anyhow!("Failed to create worktree: {}", e))?;

    if !quiet {
        println!("✅ Model '{}' cloned successfully!", model_name);
        println!("✓ Worktree: {model_name}:{worktree_branch}");
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
#[allow(clippy::too_many_arguments)]
async fn clone_with_streaming(
    registry: &RegistryClient,
    repo_url: &str,
    model_name: &str,
    shallow: bool,
    depth: u32,
    branch: Option<&str>,
    quiet: bool,
    verbose: bool,
) -> Result<()> {
    use crate::services::generated::registry_client::RegistryRpc;
    let mut stream_handle = RegistryRpc::clone_stream(registry, &CloneRequest {
            url: repo_url.to_owned(),
            name: model_name.to_owned(),
            shallow,
            depth,
            branch: branch.unwrap_or("").to_owned(),
        })
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start streaming clone: {}", e))?;

    if !quiet {
        print!("   Progress: ");
        io::stdout().flush()?;
    }

    // Receive and display progress
    let mut current_stage = String::new();
    loop {
        match stream_handle.recv_next().await? {
            Some(StreamPayload::Data(data)) => {
                // Parse progress message (format: "stage:current:total")
                if let Ok(text) = String::from_utf8(data) {
                    if !quiet {
                        // Parse simple progress format
                        let parts: Vec<&str> = text.split(':').collect();
                        if parts.len() >= 3 {
                            let stage = parts[0];
                            let current = parts[1];
                            let total = parts[2];

                            // Detect stage transitions and print stage names
                            let is_transition = stage != current_stage;
                            if is_transition {
                                if !current_stage.is_empty() && !verbose {
                                    println!();
                                }
                                let stage_label = match stage {
                                    "fetch" => "Fetching objects",
                                    "indexing" => "Indexing",
                                    "smudge" => "Downloading model files",
                                    "lfs" => "Downloading LFS files",
                                    other => other,
                                };
                                current_stage = stage.to_owned();
                                if verbose {
                                    println!("   {stage_label}...");
                                } else {
                                    print!("   {stage_label}...");
                                    io::stdout().flush()?;
                                }
                            }

                            if verbose && !is_transition {
                                if total == "0" {
                                    // Indeterminate total (e.g. smudge)
                                    println!("   {}: {} files", stage, current);
                                } else {
                                    println!("   {}: {}/{}", stage, current, total);
                                }
                            } else if !verbose {
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
    registry: &RegistryClient,
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

    // Find the worktree matching the requested branch/ref
    let branch_name = match &model_ref.git_ref {
        crate::storage::GitRef::DefaultBranch => {
            repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned())
        }
        crate::storage::GitRef::Branch(b) => b.clone(),
        _ => "main".to_owned(),
    };

    // Resolve worktree path locally (not from RPC response)
    let storage_paths = crate::storage::StoragePaths::new()?;
    let model_path = storage_paths.worktree_path(&model_ref.model, &branch_name)
        .unwrap_or_else(|_| PathBuf::from("."));

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

    // Size calculation - derive bare repo path locally from models_dir
    let bare_repo_path = storage_paths.models_dir().ok()
        .map(|d| d.join(&model_ref.model).join(format!("{}.git", &model_ref.model)));

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
    registry: &RegistryClient,
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

    // Validate model exists in registry
    let _tracked = registry.get_by_name(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository: {}", e))?;

    // Derive models_dir from StoragePaths (not from RPC response)
    let storage_paths = crate::storage::StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;

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

    // Write the updated policy with restrictive permissions
    crate::auth::write_policy_file(&policy_path, &new_content).await
        .map_err(|e| anyhow::anyhow!("Failed to write policy: {}", e))?;

    // Validate the new policy
    let policy_manager = PolicyManager::new(&policies_dir)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to validate policy: {}", e))?;

    // Reload to validate
    if let Err(e) = policy_manager.reload().await {
        // Rollback on validation failure
        let _ = crate::auth::write_policy_file(&policy_path, &existing_content).await;
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
#[allow(clippy::too_many_arguments)]
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
    sync: bool,
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
    let model_client = ModelClient::new(signing_key.clone(), RequestIdentity::anonymous());

    // Apply chat template to the prompt via ModelService
    let messages = vec![ChatMessage {
        role: "user".to_owned(),
        content: prompt.to_owned(),
        tool_calls: vec![],
        tool_call_id: String::new(),
    }];

    let formatted_prompt = match model_client.infer(model_ref_str).apply_chat_template(&ChatTemplateRequest {
        messages: messages.clone(),
        add_generation_prompt: true,
        tools_json: Some(String::new()),
        max_tokens: None,
    }).await {
        Ok(prompt_str) => crate::config::TemplatedPrompt::new(prompt_str),
        Err(e) => {
            tracing::warn!("Could not apply chat template: {}. Using raw prompt.", e);
            crate::config::TemplatedPrompt::new(prompt.to_owned())
        }
    };

    // Build generation request with CLI overrides (ModelService applies model defaults)
    let mut request = GenerationRequest {
        prompt: formatted_prompt.into_inner(),
        max_tokens: Some(max_tokens.unwrap_or(2048) as u32),
        temperature,
        top_p,
        top_k: top_k.map(|v| v as u32),
        repeat_penalty,
        seed,
        ..Default::default()
    };

    // Add image path if provided (for multimodal models)
    if let Some(img_path) = image_path {
        info!("Using image: {}", img_path);
        let img_bytes = std::fs::read(&img_path)
            .map_err(|e| anyhow::anyhow!("Failed to read image {}: {}", img_path, e))?;
        request.images = Some(vec![img_bytes]);
    }

    info!(
        "Generating response: max_tokens={:?}, temperature={:?}, top_p={:?}, top_k={:?}, repeat_penalty={:?}",
        request.max_tokens, request.temperature, request.top_p, request.top_k, request.repeat_penalty
    );

    // Generate via ModelService (handles model loading, adapter loading, training collection, auth)
    if !sync {
        // Start stream with E2E authenticated handle (DH key exchange)
        use crate::services::generated::model_client::InferRpc;
        let mut stream_handle = InferRpc::generate_stream(&model_client.infer(model_ref_str), &request).await?;

        println!();

        // Receive and print tokens
        loop {
            match stream_handle.recv_next().await? {
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
        // Non-streaming: collect stream into full response
        use crate::services::generated::model_client::InferRpc;
        let mut handle = InferRpc::generate_stream(&model_client.infer(model_ref_str), &request).await?;

        let mut text = String::new();
        loop {
            match handle.recv_next().await? {
                Some(payload) => match payload {
                    StreamPayload::Data(data) => {
                        text.push_str(&String::from_utf8_lossy(&data));
                    }
                    StreamPayload::Complete(meta) => {
                        let stats: crate::services::rpc_types::InferenceComplete =
                            serde_json::from_slice(&meta).unwrap_or_else(|e| {
                                warn!("Failed to parse InferenceComplete: {e}");
                                crate::services::rpc_types::InferenceComplete::empty()
                            });
                        println!("\n{}", text);
                        info!(
                            "Generated {} tokens in {}ms ({:.2} tokens/sec overall)",
                            stats.tokens_generated, stats.generation_time_ms, stats.tokens_per_second
                        );
                        info!(
                            "  Prefill: {} tokens in {}ms ({:.2} tokens/sec)",
                            stats.prefill_tokens, stats.prefill_time_ms, stats.prefill_tokens_per_sec
                        );
                        info!(
                            "  Inference: {} tokens in {}ms ({:.2} tokens/sec)",
                            stats.inference_tokens, stats.inference_time_ms, stats.inference_tokens_per_sec
                        );
                        break;
                    }
                    StreamPayload::Error(msg) => {
                        bail!("Generation error: {msg}");
                    }
                },
                None => {
                    bail!("Stream ended without completion");
                }
            }
        }
    }

    Ok(())
}

/// Handle load command - pre-load a model with optional runtime config
///
/// `wait` is `None` for fire-and-forget, or `Some(timeout_secs)` to subscribe
/// to notification events and wait for model.loaded/model.failed.
pub async fn handle_load(
    model_ref_str: &str,
    max_context: Option<usize>,
    kv_quant: crate::runtime::KVQuantType,
    wait: Option<u64>,
    signing_key: SigningKey,
) -> Result<()> {
    info!("Loading model: {}", model_ref_str);

    // Validate model reference format
    let model_ref = ModelRef::parse(model_ref_str)?;

    let load_max_context = max_context.map(|v| v as u32);
    let load_kv_quant = if kv_quant == crate::runtime::KVQuantType::None {
        None
    } else {
        Some(kv_quant)
    };

    // If --wait, subscribe to notifications BEFORE issuing load request
    // (ensures no events are missed between load and subscribe)
    let notification_state = if let Some(timeout_secs) = wait {
        let scope_pattern = format!("serve:model:{}", model_ref.model);
        println!("Subscribing to model events: {}", scope_pattern);

        let (sub_secret, sub_pubkey) = generate_ephemeral_keypair();
        let sub_pub_bytes = sub_pubkey.to_bytes();

        let notif_client = crate::services::NotificationClient::new(
            signing_key.clone(),
            RequestIdentity::anonymous(),
        );

        let sub_resp = notif_client.subscribe(&SubscribeRequest {
            scope_pattern: scope_pattern.clone(),
            ephemeral_pubkey: sub_pub_bytes.to_vec(),
            ttl_seconds: timeout_secs.min(3600) as u32,
        }).await?;

        debug!(
            subscription_id = %sub_resp.subscription_id,
            topic = %sub_resp.assigned_topic,
            endpoint = %sub_resp.stream_endpoint,
            "Subscribed to notification stream"
        );

        // Connect SUB socket to StreamService XPUB
        let ctx = global_context();
        let subscriber = tmq::subscribe::subscribe(&ctx)
            .connect(&sub_resp.stream_endpoint)
            .map_err(|e| anyhow::anyhow!("SUB connect to {}: {}", sub_resp.stream_endpoint, e))?
            .subscribe(sub_resp.assigned_topic.as_bytes())
            .map_err(|e| anyhow::anyhow!("SUB subscribe to topic: {}", e))?;

        subscriber.set_linger(0)
            .map_err(|e| anyhow::anyhow!("Failed to set linger on SUB: {}", e))?;

        Some((subscriber, sub_secret, sub_pub_bytes, notif_client, sub_resp, timeout_secs))
    } else {
        None
    };

    // Await the load RPC directly — the model service returns "accepted" immediately
    // (Continuation pattern), so this completes in milliseconds.
    // fire-and-forget via tokio::spawn caused the task to never run: the CLI runtime
    // drops before the spawned task executes.
    let model_client = ModelClient::new(signing_key.clone(), RequestIdentity::anonymous());
    let model_ref_owned = model_ref_str.to_owned();
    match model_client.load(&LoadModelRequest {
        model_ref: model_ref_owned.clone(),
        max_context: load_max_context,
        kv_quant: load_kv_quant.filter(|q| *q != crate::runtime::KVQuantType::None),
    }).await {
        Ok(_) => debug!("Load RPC accepted for {}", model_ref_owned),
        Err(e) => warn!("Load RPC failed for {}: {}", model_ref_owned, e),
    }

    println!("Load initiated for model: {}", model_ref_str);
    if let Some(max_ctx) = max_context {
        println!("  Max context: {}", max_ctx);
    }
    if kv_quant != crate::runtime::KVQuantType::None {
        println!("  KV quantization: {:?}", kv_quant);
    }

    // If --wait, block on notification events until model.loaded or model.failed
    if let Some((subscriber, sub_secret, sub_pub_bytes, notif_client, sub_resp, timeout_secs)) = notification_state {
        println!("Waiting for model to load (timeout: {}s)...", timeout_secs);

        let wait_start = tokio::time::Instant::now();
        let deadline = wait_start + std::time::Duration::from_secs(timeout_secs);
        let decryptor = hyprstream_rpc::crypto::notification::BroadcastDecryptor::new(
            &sub_secret.scalar().to_bytes(),
            &sub_pub_bytes,
        );

        let result = wait_for_model_notification(
            subscriber,
            &decryptor,
            model_ref_str,
            deadline,
        ).await;

        // Clean up subscription
        let _ = notif_client.unsubscribe(&UnsubscribeRequest {
            subscription_id: sub_resp.subscription_id.clone(),
        }).await;

        match result {
            Ok(NotificationOutcome::Loaded { endpoint }) => {
                println!("✓ Model {} loaded", model_ref_str);
                println!("  Endpoint: {}", endpoint);
            }
            Ok(NotificationOutcome::Failed { error }) => {
                bail!("Model {} failed to load: {}", model_ref_str, error);
            }
            Err(_) => {
                bail!("Timeout waiting for model load notification ({}s elapsed)",
                    wait_start.elapsed().as_secs());
            }
        }
    }

    Ok(())
}

/// Outcome from waiting on a model notification.
enum NotificationOutcome {
    Loaded { endpoint: String },
    Failed { error: String },
}

/// Wait for a model.loaded or model.failed notification on the SUB socket.
async fn wait_for_model_notification(
    mut subscriber: tmq::subscribe::Subscribe,
    decryptor: &hyprstream_rpc::crypto::notification::BroadcastDecryptor,
    model_ref: &str,
    deadline: tokio::time::Instant,
) -> Result<NotificationOutcome> {
    use futures::StreamExt;
    use crate::events::{EventEnvelope, EventPayload};

    loop {
        let recv = tokio::time::timeout_at(deadline, subscriber.next()).await;

        let multipart = match recv {
            Ok(Some(Ok(msg))) => msg,
            Ok(Some(Err(e))) => {
                warn!("SUB receive error: {}", e);
                continue;
            }
            Ok(None) => bail!("Notification stream closed"),
            Err(_) => bail!("Timeout"),
        };

        // StreamService 3-frame: [topic, capnp_data, hmac]
        let frames: Vec<_> = multipart.iter().collect();
        if frames.len() < 2 {
            continue;
        }

        // Parse StreamBlock to extract the notification data
        let capnp_data = &frames[1];
        let payload_data = match parse_stream_block_data(capnp_data) {
            Ok(data) => data,
            Err(e) => {
                debug!("Failed to parse StreamBlock: {}", e);
                continue;
            }
        };

        // Parse NotificationBlock from the StreamPayload data
        let block = match parse_notification_block(&payload_data) {
            Ok(b) => b,
            Err(e) => {
                debug!("Failed to parse NotificationBlock: {}", e);
                continue;
            }
        };

        // Decrypt the notification payload
        let plaintext = match decryptor.decrypt(
            &block.publisher_pubkey,
            &block.blinding_scalar,
            &block.wrapped_key,
            &block.key_nonce,
            &block.encrypted_payload,
            &block.nonce,
            &block.publisher_mac,
            &block.intent_id,
            &block.scope,
        ) {
            Ok(pt) => pt,
            Err(e) => {
                debug!("Failed to decrypt notification: {}", e);
                continue;
            }
        };

        // Parse EventEnvelope from decrypted payload
        let event: EventEnvelope = match serde_json::from_slice(&plaintext) {
            Ok(e) => e,
            Err(e) => {
                debug!("Failed to parse event: {}", e);
                continue;
            }
        };

        // Check if this is our model
        match &event.payload {
            EventPayload::ModelLoaded { model_ref: ref mr, endpoint } if mr.contains(model_ref) || model_ref.contains(mr.as_str()) => {
                return Ok(NotificationOutcome::Loaded { endpoint: endpoint.clone() });
            }
            EventPayload::ModelFailed { model_ref: ref mr, error } if mr.contains(model_ref) || model_ref.contains(mr.as_str()) => {
                return Ok(NotificationOutcome::Failed { error: error.clone() });
            }
            _ => {
                debug!("Ignoring unrelated event: {:?}", event.topic);
                continue;
            }
        }
    }
}

/// Parsed notification block fields.
struct NotificationBlockFields {
    publisher_pubkey: [u8; 32],
    blinding_scalar: [u8; 32],
    wrapped_key: Vec<u8>,
    key_nonce: [u8; 12],
    encrypted_payload: Vec<u8>,
    nonce: [u8; 12],
    intent_id: String,
    scope: String,
    publisher_mac: [u8; 32],
}

/// Parse a StreamBlock's first data payload from Cap'n Proto bytes.
fn parse_stream_block_data(capnp_data: &[u8]) -> Result<Vec<u8>> {
    let reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(capnp_data),
        capnp::message::ReaderOptions::default(),
    )?;
    let block = reader.get_root::<crate::streaming_capnp::stream_block::Reader>()?;

    // StreamBlock has payloads: List(StreamPayload) — extract first Data variant
    let payloads = block.get_payloads()?;
    for i in 0..payloads.len() {
        let payload = payloads.get(i);
        if let crate::streaming_capnp::stream_payload::Which::Data(data) = payload.which()? {
            return Ok(data?.to_vec());
        }
    }

    bail!("StreamBlock contains no Data payload")
}

/// Parse a NotificationBlock from Cap'n Proto bytes.
fn parse_notification_block(data: &[u8]) -> Result<NotificationBlockFields> {
    let reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(data),
        capnp::message::ReaderOptions::default(),
    )?;
    let block = reader.get_root::<crate::notification_capnp::notification_block::Reader>()?;

    let publisher_pubkey: [u8; 32] = block.get_publisher_pubkey()?
        .try_into().map_err(|_| anyhow::anyhow!("publisher_pubkey not 32 bytes"))?;
    let blinding_scalar: [u8; 32] = block.get_blinding_scalar()?
        .try_into().map_err(|_| anyhow::anyhow!("blinding_scalar not 32 bytes"))?;
    let wrapped_key = block.get_wrapped_key()?.to_vec();
    let key_nonce: [u8; 12] = block.get_key_nonce()?
        .try_into().map_err(|_| anyhow::anyhow!("key_nonce not 12 bytes"))?;
    let encrypted_payload = block.get_encrypted_payload()?.to_vec();
    let nonce: [u8; 12] = block.get_nonce()?
        .try_into().map_err(|_| anyhow::anyhow!("nonce not 12 bytes"))?;
    let intent_id = block.get_intent_id()?.to_string()?;
    let scope = block.get_scope()?.to_string()?;
    let publisher_mac: [u8; 32] = block.get_publisher_mac()?
        .try_into().map_err(|_| anyhow::anyhow!("publisher_mac not 32 bytes"))?;

    Ok(NotificationBlockFields {
        publisher_pubkey,
        blinding_scalar,
        wrapped_key,
        key_nonce,
        encrypted_payload,
        nonce,
        intent_id,
        scope,
        publisher_mac,
    })
}

/// Handle the `notify` subcommands.
pub async fn handle_notify_command(
    command: crate::cli::quick::NotifyCommand,
    signing_key: SigningKey,
) -> Result<()> {
    use crate::cli::quick::NotifyCommand;

    match command {
        NotifyCommand::Subscribe { pattern, json, timeout, count } => {
            handle_notify_subscribe(&pattern, json, timeout, count, signing_key).await
        }
    }
}

/// Handle `notify subscribe` — listen for and print decrypted notification events.
async fn handle_notify_subscribe(
    pattern: &str,
    json_output: bool,
    timeout_secs: u64,
    max_count: u64,
    signing_key: SigningKey,
) -> Result<()> {
    use futures::StreamExt;
    use crate::events::EventEnvelope;

    let (sub_secret, sub_pubkey) = generate_ephemeral_keypair();
    let sub_pub_bytes = sub_pubkey.to_bytes();

    let notif_client = crate::services::NotificationClient::new(
        signing_key,
        RequestIdentity::anonymous(),
    );

    // Subscribe with maximum TTL for long-running listeners
    let ttl = if timeout_secs == 0 { 3600u32 } else { (timeout_secs as u32).min(3600) };
    let sub_resp = notif_client.subscribe(&SubscribeRequest {
        scope_pattern: pattern.to_owned(),
        ephemeral_pubkey: sub_pub_bytes.to_vec(),
        ttl_seconds: ttl,
    }).await?;

    eprintln!("Subscribed to: {}", pattern);
    eprintln!("  Subscription ID: {}", sub_resp.subscription_id);
    eprintln!("  Topic: {}", sub_resp.assigned_topic);

    // Connect SUB socket
    let ctx = global_context();
    let mut subscriber = tmq::subscribe::subscribe(&ctx)
        .connect(&sub_resp.stream_endpoint)
        .map_err(|e| anyhow::anyhow!("SUB connect: {}", e))?
        .subscribe(sub_resp.assigned_topic.as_bytes())
        .map_err(|e| anyhow::anyhow!("SUB subscribe: {}", e))?;

    subscriber.set_linger(0)?;

    let decryptor = hyprstream_rpc::crypto::notification::BroadcastDecryptor::new(
        &sub_secret.scalar().to_bytes(),
        &sub_pub_bytes,
    );

    let deadline = if timeout_secs > 0 {
        Some(tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs))
    } else {
        None
    };

    let mut received = 0u64;

    loop {
        // Check count limit
        if max_count > 0 && received >= max_count {
            break;
        }

        // Receive with optional timeout
        let multipart = if let Some(dl) = deadline {
            match tokio::time::timeout_at(dl, subscriber.next()).await {
                Ok(Some(Ok(msg))) => msg,
                Ok(Some(Err(e))) => { warn!("SUB error: {}", e); continue; }
                Ok(None) => break,
                Err(_) => { eprintln!("Timeout reached"); break; }
            }
        } else {
            match subscriber.next().await {
                Some(Ok(msg)) => msg,
                Some(Err(e)) => { warn!("SUB error: {}", e); continue; }
                None => break,
            }
        };

        let frames: Vec<_> = multipart.iter().collect();
        if frames.len() < 2 { continue; }

        let capnp_data = &frames[1];
        let payload_data = match parse_stream_block_data(capnp_data) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let block = match parse_notification_block(&payload_data) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let plaintext = match decryptor.decrypt(
            &block.publisher_pubkey,
            &block.blinding_scalar,
            &block.wrapped_key,
            &block.key_nonce,
            &block.encrypted_payload,
            &block.nonce,
            &block.publisher_mac,
            &block.intent_id,
            &block.scope,
        ) {
            Ok(pt) => pt,
            Err(e) => { debug!("Decrypt failed: {}", e); continue; }
        };

        if json_output {
            // Print raw JSON
            if let Ok(s) = String::from_utf8(plaintext.clone()) {
                println!("{}", s);
            }
        } else {
            // Human-readable output
            match serde_json::from_slice::<EventEnvelope>(&plaintext) {
                Ok(event) => {
                    println!("[{}] {} (source: {})",
                        event.timestamp.format("%H:%M:%S"),
                        event.topic,
                        event.source,
                    );
                    println!("  {:?}", event.payload);
                }
                Err(_) => {
                    // Not an EventEnvelope, print raw
                    if let Ok(s) = String::from_utf8(plaintext) {
                        println!("{}", s);
                    }
                }
            }
        }

        received += 1;
    }

    // Clean up
    let _ = notif_client.unsubscribe(&UnsubscribeRequest {
        subscription_id: sub_resp.subscription_id.clone(),
    }).await;
    eprintln!("Received {} events", received);

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

    let model_client = ModelClient::new(signing_key, RequestIdentity::anonymous());

    model_client.unload(&UnloadModelRequest { model_ref: model_ref_str.to_owned() }).await?;

    println!("✓ Model {} unloaded", model_ref_str);

    Ok(())
}

/// Handle push command
///
/// **EXPERIMENTAL**: This feature is behind the `experimental` flag.
#[cfg(feature = "experimental")]
pub async fn handle_push(
    registry: &RegistryClient,
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
    registry: &RegistryClient,
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
    repo_client.update(&UpdateRequest {
        refspec: refspec.as_deref().unwrap_or("").to_owned(),
    }).await?;

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
    registry: &RegistryClient,
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

    // Determine target branch for worktree-scoped merge
    let target_branch = match &target_ref.git_ref {
        crate::storage::GitRef::Branch(b) => b.clone(),
        crate::storage::GitRef::DefaultBranch => repo_client.get_head().await
            .unwrap_or_else(|_| "main".to_owned()),
        _ => anyhow::bail!("Merge target must be a branch reference"),
    };

    // Perform merge via worktree-scoped service layer
    let merge_result = repo_client.worktree(&target_branch).merge(source, message).await;

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
    registry: &RegistryClient,
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
    if !wts.iter().any(|wt| wt.branch_name == target_branch) {
        bail!("Worktree not found for branch {}", target_branch);
    }

    // Use worktree-scoped client for merge operations
    let wt = repo_client.worktree(&target_branch);

    if options.abort {
        // Abort merge: restore pre-merge state
        if !options.quiet {
            println!("→ Aborting merge...");
        }

        wt.abort_merge().await?;

        if !options.quiet {
            println!("✓ Merge aborted, restored pre-merge state");
        }
    } else if options.continue_merge {
        // Continue merge: check if conflicts are resolved and create merge commit
        if !options.quiet {
            println!("→ Continuing merge...");
        }

        let merge_oid = wt.continue_merge(options.message.as_deref()).await?;

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

        wt.quit_merge().await?;

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
    registry: &RegistryClient,
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
        println!("  Remove worktree for branch: {}", wt.branch_name);

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

        // Remove the worktree via service (pass branch name, not path)
        repo_client.remove_worktree(&RemoveWorktreeRequest {
            branch: wt.branch_name.clone(),
            force: false,
        }).await
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
