//! Handlers for policy management commands
//!
//! Cisco-inspired transactional UX for managing access control policies.
//!
//! Policy states:
//! - **Running** = Current active (HEAD of .registry/policies/)
//! - **Draft** = Uncommitted changes
//! - **History** = Previous versions (HEAD~n)

use crate::auth::{Operation, PolicyManager, TokenManager};
use anyhow::{Context, Result};
use chrono::Duration;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use tracing::info;

/// Handle `policy show` - Display the running policy
pub async fn handle_policy_show(
    policy_manager: &PolicyManager,
    raw: bool,
) -> Result<()> {
    let policy_path = policy_manager.policy_csv_path();

    if raw {
        // Show raw CSV content
        if policy_path.exists() {
            let content = tokio::fs::read_to_string(&policy_path).await?;
            println!("{}", content);
        } else {
            println!("# No policy file exists");
        }
    } else {
        // Show formatted table
        let policies = policy_manager.get_policy().await;
        let groupings = policy_manager.get_grouping_policy().await;

        if policies.is_empty() && groupings.is_empty() {
            println!("No policies defined.");
            println!("\nTo create policies, run:");
            println!("  hyprstream policy edit");
            return Ok(());
        }

        // Print policy rules as a table
        if !policies.is_empty() {
            println!("┌──────────────────────────────────────────────────────────────┐");
            println!("│ Policy Rules                                                 │");
            println!("├────────────────────┬────────────────────┬────────────────────┤");
            println!("│ Subject            │ Resource           │ Action             │");
            println!("├────────────────────┼────────────────────┼────────────────────┤");

            for p in &policies {
                let sub = p.get(0).map(|s| s.as_str()).unwrap_or("");
                let obj = p.get(1).map(|s| s.as_str()).unwrap_or("");
                let act = p.get(2).map(|s| s.as_str()).unwrap_or("");
                println!(
                    "│ {:18} │ {:18} │ {:18} │",
                    truncate_str(sub, 18),
                    truncate_str(obj, 18),
                    truncate_str(act, 18)
                );
            }
            println!("└────────────────────┴────────────────────┴────────────────────┘");
        }

        // Print role assignments
        if !groupings.is_empty() {
            println!();
            println!("┌────────────────────────────────────────────┐");
            println!("│ Role Assignments                           │");
            println!("├────────────────────┬───────────────────────┤");
            println!("│ User               │ Role                  │");
            println!("├────────────────────┼───────────────────────┤");

            for g in &groupings {
                let user = g.get(0).map(|s| s.as_str()).unwrap_or("");
                let role = g.get(1).map(|s| s.as_str()).unwrap_or("");
                println!(
                    "│ {:18} │ {:21} │",
                    truncate_str(user, 18),
                    truncate_str(role, 21)
                );
            }
            println!("└────────────────────┴───────────────────────┘");
        }
    }

    Ok(())
}

/// Handle `policy history` - Show policy commit history
pub async fn handle_policy_history(
    policy_manager: &PolicyManager,
    count: usize,
    oneline: bool,
) -> Result<()> {
    let policies_dir = policy_manager.policies_dir();

    // Find the .registry directory (parent of policies/)
    let registry_dir = policies_dir
        .parent()
        .context("Could not find .registry directory")?;

    // Run git log on the policies directory
    let mut cmd = Command::new("git");
    cmd.current_dir(registry_dir);
    cmd.args(["log", "-n", &count.to_string()]);

    if oneline {
        cmd.args(["--oneline"]);
    } else {
        cmd.args([
            "--pretty=format:%C(yellow)%h%Creset %C(green)%ad%Creset %s%n%C(dim)  by %an%Creset%n",
            "--date=relative",
        ]);
    }

    cmd.args(["--", "policies/"]);

    let output = cmd.output().context("Failed to run git log")?;

    if output.stdout.is_empty() {
        println!("No policy history found.");
        println!("\nPolicies are versioned using git commits in .registry/");
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.is_empty() {
            eprintln!("{}", stderr);
        }
    }

    Ok(())
}

/// Handle `policy edit` - Open policy in $VISUAL/$EDITOR
pub async fn handle_policy_edit(policy_manager: &PolicyManager) -> Result<()> {
    let policy_path = policy_manager.policy_csv_path();

    // Ensure policy file exists
    if !policy_path.exists() {
        info!("Creating policy file at {:?}", policy_path);
        tokio::fs::write(&policy_path, default_policy_template()).await?;
    }

    // Get editor from environment
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| "vi".to_string());

    println!("Opening {} in {}", policy_path.display(), editor);
    println!();
    println!("After editing, use:");
    println!("  hyprstream policy diff     # Preview changes");
    println!("  hyprstream policy apply    # Commit changes");
    println!();

    // Run editor
    let status = Command::new(&editor)
        .arg(&policy_path)
        .status()
        .context(format!("Failed to run editor: {}", editor))?;

    if !status.success() {
        anyhow::bail!("Editor exited with non-zero status");
    }

    // Check if there are uncommitted changes
    if has_uncommitted_changes(policy_manager.policies_dir())? {
        println!();
        println!("✏️  Draft changes detected. Run 'hyprstream policy diff' to review.");
    }

    Ok(())
}

/// Handle `policy diff` - Show diff between draft and running policy
pub async fn handle_policy_diff(
    policy_manager: &PolicyManager,
    against: Option<String>,
) -> Result<()> {
    let policies_dir = policy_manager.policies_dir();
    let registry_dir = policies_dir
        .parent()
        .context("Could not find .registry directory")?;

    let git_ref = against.as_deref().unwrap_or("HEAD");

    // Run git diff
    let output = Command::new("git")
        .current_dir(registry_dir)
        .args(["diff", "--color=always", git_ref, "--", "policies/"])
        .output()
        .context("Failed to run git diff")?;

    if output.stdout.is_empty() {
        println!("No changes from {} policy.", git_ref);
        return Ok(());
    }

    println!("Changes vs {}:\n", git_ref);
    print!("{}", String::from_utf8_lossy(&output.stdout));

    Ok(())
}

/// Handle `policy apply` - Commit draft changes to running policy
pub async fn handle_policy_apply(
    policy_manager: &PolicyManager,
    dry_run: bool,
    message: Option<String>,
) -> Result<()> {
    let policies_dir = policy_manager.policies_dir();
    let registry_dir = policies_dir
        .parent()
        .context("Could not find .registry directory")?;

    // Check for uncommitted changes
    if !has_uncommitted_changes(policies_dir)? {
        println!("No changes to apply.");
        return Ok(());
    }

    // Show what would be committed
    let diff_output = Command::new("git")
        .current_dir(registry_dir)
        .args(["diff", "--stat", "--", "policies/"])
        .output()
        .context("Failed to get diff")?;

    println!("Changes to be applied:");
    print!("{}", String::from_utf8_lossy(&diff_output.stdout));
    println!();

    if dry_run {
        println!("--dry-run specified, no changes committed.");
        return Ok(());
    }

    // Generate commit message
    let commit_msg = message.unwrap_or_else(|| {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        format!("policy: update access control rules ({})", timestamp)
    });

    // Validate the new policy before committing
    print!("Validating policy... ");
    io::stdout().flush()?;

    match policy_manager.reload().await {
        Ok(_) => println!("✓ valid"),
        Err(e) => {
            println!("✗ invalid");
            anyhow::bail!("Policy validation failed: {}. Fix errors before applying.", e);
        }
    }

    // Stage and commit
    Command::new("git")
        .current_dir(registry_dir)
        .args(["add", "policies/"])
        .output()
        .context("Failed to stage changes")?;

    let commit_result = Command::new("git")
        .current_dir(registry_dir)
        .args(["commit", "-m", &commit_msg])
        .output()
        .context("Failed to commit")?;

    if !commit_result.status.success() {
        let stderr = String::from_utf8_lossy(&commit_result.stderr);
        anyhow::bail!("Commit failed: {}", stderr);
    }

    // Reload the policy
    policy_manager.reload().await?;

    println!();
    println!("✓ Policy applied successfully.");
    println!("  {}", commit_msg);

    Ok(())
}

/// Handle `policy rollback [ref]` - Revert to a previous policy version
pub async fn handle_policy_rollback(
    policy_manager: &PolicyManager,
    git_ref: &str,
    dry_run: bool,
) -> Result<()> {
    let policies_dir = policy_manager.policies_dir();
    let registry_dir = policies_dir
        .parent()
        .context("Could not find .registry directory")?;

    // Show what we're rolling back to
    let log_output = Command::new("git")
        .current_dir(registry_dir)
        .args(["log", "-1", "--oneline", git_ref])
        .output()
        .context("Failed to get commit info")?;

    let target_commit = String::from_utf8_lossy(&log_output.stdout)
        .trim()
        .to_string();

    if target_commit.is_empty() {
        anyhow::bail!("Invalid git ref: {}", git_ref);
    }

    println!("Rolling back to: {}", target_commit);

    // Show diff
    let diff_output = Command::new("git")
        .current_dir(registry_dir)
        .args(["diff", "--stat", "HEAD", git_ref, "--", "policies/"])
        .output()
        .context("Failed to get diff")?;

    println!("\nChanges:");
    print!("{}", String::from_utf8_lossy(&diff_output.stdout));
    println!();

    if dry_run {
        println!("--dry-run specified, no changes applied.");
        return Ok(());
    }

    // Confirm
    print!("Apply rollback? [y/N] ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("y") {
        println!("Aborted.");
        return Ok(());
    }

    // Checkout the policy files from the target ref
    let checkout_result = Command::new("git")
        .current_dir(registry_dir)
        .args(["checkout", git_ref, "--", "policies/"])
        .output()
        .context("Failed to checkout files")?;

    if !checkout_result.status.success() {
        let stderr = String::from_utf8_lossy(&checkout_result.stderr);
        anyhow::bail!("Checkout failed: {}", stderr);
    }

    // Commit the rollback
    let commit_msg = format!("policy: rollback to {}", git_ref);

    Command::new("git")
        .current_dir(registry_dir)
        .args(["add", "policies/"])
        .output()
        .context("Failed to stage changes")?;

    let commit_result = Command::new("git")
        .current_dir(registry_dir)
        .args(["commit", "-m", &commit_msg])
        .output()
        .context("Failed to commit")?;

    if !commit_result.status.success() {
        let stderr = String::from_utf8_lossy(&commit_result.stderr);
        // If nothing to commit, that's okay
        if !stderr.contains("nothing to commit") {
            anyhow::bail!("Commit failed: {}", stderr);
        }
    }

    // Reload policy
    policy_manager.reload().await?;

    println!();
    println!("✓ Policy rolled back to {}", git_ref);

    Ok(())
}

/// Handle `policy check <user> <resource> <action>` - Test permission
pub async fn handle_policy_check(
    policy_manager: &PolicyManager,
    user: &str,
    resource: &str,
    action: &str,
) -> Result<()> {
    // Parse the action
    let operation = match action.to_lowercase().as_str() {
        "infer" | "i" => Operation::Infer,
        "train" | "t" => Operation::Train,
        "query" | "q" => Operation::Query,
        "write" | "w" => Operation::Write,
        "serve" | "s" => Operation::Serve,
        "manage" | "m" => Operation::Manage,
        _ => anyhow::bail!(
            "Unknown action: {}. Valid actions: infer, train, query, write, serve, manage",
            action
        ),
    };

    let allowed = policy_manager.check(user, resource, operation).await;

    println!("User:     {}", user);
    println!("Resource: {}", resource);
    println!("Action:   {}", action);
    println!();

    if allowed {
        println!("Result:   ✓ ALLOWED");
    } else {
        println!("Result:   ✗ DENIED");
    }

    // Show user's roles if any
    let roles = policy_manager.get_roles_for_user(user).await;
    if !roles.is_empty() {
        println!();
        println!("User roles: {}", roles.join(", "));
    }

    Ok(())
}

// === Token handlers ===

/// Handle `policy token create` - Create a new API token
pub async fn handle_token_create(
    token_manager: &mut TokenManager,
    user: &str,
    name: Option<String>,
    expires: &str,
    scopes: Vec<String>,
    admin: bool,
) -> Result<()> {
    // Parse expiration
    let expires_duration = parse_duration(expires)?;

    // Generate name if not provided
    let name = name.unwrap_or_else(|| {
        format!("token-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"))
    });

    // Create the token
    let token = token_manager
        .create_token(user, &name, expires_duration, scopes.clone(), admin)
        .await
        .context("Failed to create token")?;

    // Display the token (only shown once)
    println!();
    println!("Token created for user '{}':", user);
    println!();
    println!("  {}", token);
    println!();
    println!("  Name:    {}", name);

    match expires_duration {
        Some(d) => {
            let expires_at = chrono::Utc::now() + d;
            let days = d.num_days();
            println!("  Expires: {} ({} days)", expires_at.format("%Y-%m-%d"), days);
        }
        None => {
            println!("  Expires: never");
        }
    }

    if scopes.is_empty() || scopes.contains(&"*".to_string()) {
        println!("  Scopes:  * (all resources)");
    } else {
        println!("  Scopes:  {}", scopes.join(", "));
    }

    println!();
    println!("  \x1b[33m⚠️  Save this token now - it cannot be retrieved again.\x1b[0m");
    println!();
    println!("  Usage:");
    println!("    curl -H \"Authorization: Bearer {}...\" http://localhost:8080/v1/models", &token[..20]);
    println!();

    Ok(())
}

/// Handle `policy token list` - List all API tokens
pub async fn handle_token_list(
    token_manager: &TokenManager,
    user_filter: Option<String>,
) -> Result<()> {
    let tokens = match user_filter {
        Some(ref user) => token_manager.list_for_user(user),
        None => token_manager.list(),
    };

    if tokens.is_empty() {
        println!("No API tokens found.");
        println!();
        println!("Create a token with:");
        println!("  hyprstream policy token create <user>");
        return Ok(());
    }

    // Print header
    println!(
        "{:<12} {:<12} {:<20} {:<12} {}",
        "USER", "NAME", "CREATED", "EXPIRES", "SCOPES"
    );
    println!("{}", "-".repeat(70));

    for token in tokens {
        let created = token.created_at.format("%Y-%m-%d").to_string();
        let expires = match token.expires_at {
            Some(dt) => dt.format("%Y-%m-%d").to_string(),
            None => "never".to_string(),
        };
        let scopes = if token.scopes.is_empty() || token.scopes == vec!["*".to_string()] {
            "*".to_string()
        } else {
            token.scopes.join(", ")
        };

        println!(
            "{:<12} {:<12} {:<20} {:<12} {}",
            truncate_str(&token.user, 12),
            truncate_str(&token.name, 12),
            created,
            expires,
            truncate_str(&scopes, 30)
        );
    }

    Ok(())
}

/// Handle `policy token revoke` - Revoke an API token
pub async fn handle_token_revoke(
    token_manager: &mut TokenManager,
    token: Option<String>,
    name: Option<String>,
    revoke_user: Option<String>,
    force: bool,
) -> Result<()> {
    // Determine how to find the token
    let record = if let (Some(user), Some(name)) = (revoke_user, name) {
        // Revoke by user and name
        if !force {
            print!("Revoke token '{}' for user '{}'? [y/N] ", name, user);
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Aborted.");
                return Ok(());
            }
        }

        token_manager.revoke_by_name(&user, &name).await?
    } else if let Some(token_str) = token {
        // Revoke by token
        if !force {
            let display_token = if token_str.len() > 15 {
                format!("{}...", &token_str[..15])
            } else {
                token_str.clone()
            };
            print!("Revoke token '{}'? [y/N] ", display_token);
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Aborted.");
                return Ok(());
            }
        }

        token_manager.revoke(&token_str).await?
    } else {
        anyhow::bail!("Must provide either a token or --user and --name");
    };

    println!("✓ Token '{}' for user '{}' revoked", record.name, record.user);

    Ok(())
}

/// Parse duration string like "30d", "90d", "1y", "never"
fn parse_duration(s: &str) -> Result<Option<Duration>> {
    let s = s.trim().to_lowercase();

    if s == "never" || s.is_empty() {
        return Ok(None);
    }

    let (num_str, unit) = if s.ends_with('d') {
        (&s[..s.len() - 1], 'd')
    } else if s.ends_with('y') {
        (&s[..s.len() - 1], 'y')
    } else if s.ends_with('h') {
        (&s[..s.len() - 1], 'h')
    } else {
        // Assume days if no unit
        (s.as_str(), 'd')
    };

    let num: i64 = num_str
        .parse()
        .context(format!("Invalid duration number: {}", num_str))?;

    let duration = match unit {
        'd' => Duration::days(num),
        'y' => Duration::days(num * 365),
        'h' => Duration::hours(num),
        _ => anyhow::bail!("Unknown duration unit: {}", unit),
    };

    Ok(Some(duration))
}

// === Helper functions ===

/// Check if there are uncommitted changes in the policies directory
fn has_uncommitted_changes(policies_dir: &Path) -> Result<bool> {
    let registry_dir = policies_dir
        .parent()
        .context("Could not find .registry directory")?;

    let output = Command::new("git")
        .current_dir(registry_dir)
        .args(["status", "--porcelain", "--", "policies/"])
        .output()
        .context("Failed to check git status")?;

    Ok(!output.stdout.is_empty())
}

/// Truncate a string to max length, adding "..." if truncated
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s[..max_len].to_string()
    }
}

/// Default policy template for new policy files
fn default_policy_template() -> &'static str {
    r#"# Hyprstream Access Control Policy
# Format: p, subject, resource, action
#
# Subjects: user names or role names
# Resources: model:<name>, data:<name>, or * for all
# Actions: infer, train, query, write, serve, manage, or * for all
#
# Examples:
# p, admin, *, *                    # Admin can do anything
# p, trainer, model:*, infer        # Trainers can infer any model
# p, trainer, model:*, train        # Trainers can train any model
# p, analyst, data:*, query         # Analysts can query data
#
# Role assignments (g, user, role):
# g, alice, trainer                 # Alice has the trainer role
# g, bob, analyst                   # Bob has the analyst role

# Default permissive policy (remove for production)
p, *, *, *
"#
}
