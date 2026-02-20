//! Handlers for policy management commands
//!
//! Cisco-inspired transactional UX for managing access control policies.
//!
//! Policy states:
//! - **Running** = Current active (HEAD of .registry/policies/)
//! - **Draft** = Uncommitted changes
//! - **History** = Previous versions (HEAD~n)
//!
//! All commands route through PolicyService RPCs except `edit` which opens
//! a local editor (but still uses RPC for draft status detection afterward).
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use crate::auth::{jwt, Claims};
use crate::services::generated::policy_client::PolicyClient;
use anyhow::{Context, Result};
use chrono::Duration;
use ed25519_dalek::SigningKey;
use hyprstream_rpc::prelude::*;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use tracing::info;

/// Create a PolicyClient for RPC calls.
fn create_policy_client(signing_key: &SigningKey) -> PolicyClient {
    PolicyClient::new(signing_key.clone(), RequestIdentity::local())
}

/// Handle `policy show` - Display the running policy via RPC
pub async fn handle_policy_show(
    signing_key: &SigningKey,
    raw: bool,
) -> Result<()> {
    let client = create_policy_client(signing_key);
    let policy_info = client.get_policy().await
        .context("Failed to get policy from PolicyService. Are services running?")?;

    if raw {
        // Reconstruct CSV from structured data
        for rule in &policy_info.rules {
            println!("p, {}, {}, {}, {}, {}", rule.subject, rule.domain, rule.resource, rule.action, rule.effect);
        }
        for grouping in &policy_info.groupings {
            println!("g, {}, {}", grouping.user, grouping.role);
        }
    } else {
        if policy_info.rules.is_empty() && policy_info.groupings.is_empty() {
            println!("No policies defined.");
            println!("\nTo get started, apply a template:");
            println!("  hyprstream policy apply-template local    # local CLI full access");
            println!("\nOr edit manually:");
            println!("  hyprstream policy edit");
            return Ok(());
        }

        // Print policy rules as a table
        if !policy_info.rules.is_empty() {
            println!("┌──────────────────────────────────────────────────────────────────────────────┐");
            println!("│ Policy Rules                                                                 │");
            println!("├────────────────┬────────────────┬────────────────┬────────────────┬──────────┤");
            println!("│ Subject        │ Domain         │ Resource       │ Action         │ Effect   │");
            println!("├────────────────┼────────────────┼────────────────┼────────────────┼──────────┤");

            for rule in &policy_info.rules {
                println!(
                    "│ {:14} │ {:14} │ {:14} │ {:14} │ {:8} │",
                    truncate_str(&rule.subject, 14),
                    truncate_str(&rule.domain, 14),
                    truncate_str(&rule.resource, 14),
                    truncate_str(&rule.action, 14),
                    truncate_str(&rule.effect, 8)
                );
            }
            println!("└────────────────┴────────────────┴────────────────┴────────────────┴──────────┘");
        }

        // Print role assignments
        if !policy_info.groupings.is_empty() {
            println!();
            println!("┌────────────────────────────────────────────┐");
            println!("│ Role Assignments                           │");
            println!("├────────────────────┬───────────────────────┤");
            println!("│ User               │ Role                  │");
            println!("├────────────────────┼───────────────────────┤");

            for g in &policy_info.groupings {
                println!(
                    "│ {:18} │ {:21} │",
                    truncate_str(&g.user, 18),
                    truncate_str(&g.role, 21)
                );
            }
            println!("└────────────────────┴───────────────────────┘");
        }
    }

    Ok(())
}

/// Handle `policy history` - Show policy commit history via RPC
pub async fn handle_policy_history(
    signing_key: &SigningKey,
    count: usize,
    _oneline: bool,
) -> Result<()> {
    let client = create_policy_client(signing_key);
    let history = client.get_history(count as u32).await
        .context("Failed to get policy history from PolicyService. Are services running?")?;

    if history.entries.is_empty() {
        println!("No policy history found.");
        println!("\nPolicies are versioned using git commits in .registry/");
        return Ok(());
    }

    for entry in &history.entries {
        println!("\x1b[33m{}\x1b[0m \x1b[32m{}\x1b[0m {}", entry.hash, entry.date, entry.message);
    }

    Ok(())
}

/// Handle `policy edit` - Open policy in $VISUAL/$EDITOR
///
/// The editor itself must be local, but draft detection uses PolicyService RPC.
pub async fn handle_policy_edit(
    signing_key: &SigningKey,
    policy_csv_path: &Path,
) -> Result<()> {
    // Get editor from environment
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| "vi".to_owned());

    println!("Opening {} in {}", policy_csv_path.display(), editor);
    println!();
    println!("After editing, use:");
    println!("  hyprstream policy diff     # Preview changes");
    println!("  hyprstream policy apply    # Commit changes");
    println!();

    // Run editor
    let status = Command::new(&editor)
        .arg(policy_csv_path)
        .status()
        .context(format!("Failed to run editor: {editor}"))?;

    if !status.success() {
        anyhow::bail!("Editor exited with non-zero status");
    }

    // Check for draft changes via PolicyService RPC
    let client = create_policy_client(signing_key);
    match client.get_draft_status().await {
        Ok(draft) if draft.has_changes => {
            println!();
            println!("✏️  Draft changes detected ({}). Run 'hyprstream policy diff' to review.", draft.summary);
        }
        Ok(_) => {} // No changes
        Err(_) => {
            // Service may not be running; just suggest the commands
            println!();
            println!("Run 'hyprstream policy diff' to review changes.");
        }
    }

    Ok(())
}

/// Handle `policy diff` - Show diff between draft and running policy via RPC
pub async fn handle_policy_diff(
    signing_key: &SigningKey,
    against: Option<String>,
) -> Result<()> {
    let client = create_policy_client(signing_key);
    let git_ref = against.as_deref().unwrap_or("");

    let diff_text = client.get_diff(git_ref).await
        .context("Failed to get diff from PolicyService. Are services running?")?;

    if diff_text.is_empty() {
        let ref_display = if git_ref.is_empty() { "HEAD" } else { git_ref };
        println!("No changes from {ref_display} policy.");
        return Ok(());
    }

    let ref_display = if git_ref.is_empty() { "HEAD" } else { git_ref };
    println!("Changes vs {ref_display}:\n");

    // Colorize the diff output
    for line in diff_text.lines() {
        if let Some(rest) = line.strip_prefix('+') {
            println!("\x1b[32m+{rest}\x1b[0m");
        } else if let Some(rest) = line.strip_prefix('-') {
            println!("\x1b[31m-{rest}\x1b[0m");
        } else {
            println!("{line}");
        }
    }

    Ok(())
}

/// Handle `policy apply` - Commit draft changes via RPC
pub async fn handle_policy_apply(
    signing_key: &SigningKey,
    dry_run: bool,
    message: Option<String>,
) -> Result<()> {
    let client = create_policy_client(signing_key);

    // Check for uncommitted changes via RPC
    let draft = client.get_draft_status().await
        .context("Failed to check draft status from PolicyService. Are services running?")?;

    if !draft.has_changes {
        println!("No changes to apply.");
        return Ok(());
    }

    // Show what would be committed
    let diff_text = client.get_diff("").await
        .context("Failed to get diff from PolicyService.")?;

    println!("Changes to be applied ({}):", draft.summary);
    for line in diff_text.lines() {
        if let Some(rest) = line.strip_prefix('+') {
            println!("  \x1b[32m+{rest}\x1b[0m");
        } else if let Some(rest) = line.strip_prefix('-') {
            println!("  \x1b[31m-{rest}\x1b[0m");
        }
    }
    println!();

    if dry_run {
        println!("--dry-run specified, no changes committed.");
        return Ok(());
    }

    // Use PolicyService RPC to validate, stage, and commit
    let commit_msg = message.unwrap_or_else(|| {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        format!("policy: update access control rules ({timestamp})")
    });

    let result_msg = client.apply_draft(&commit_msg).await
        .context("Failed to apply draft via PolicyService.")?;

    println!("✓ Policy applied successfully.");
    println!("  {result_msg}");

    Ok(())
}

/// Handle `policy rollback [ref]` - Revert to a previous policy version via RPC
pub async fn handle_policy_rollback(
    signing_key: &SigningKey,
    git_ref: &str,
    dry_run: bool,
) -> Result<()> {
    let client = create_policy_client(signing_key);

    if dry_run {
        // Use history RPC to show what we'd be rolling back to
        let history = client.get_history(20).await
            .context("Failed to get history from PolicyService. Are services running?")?;

        // Find the matching entry
        let target = history.entries.iter().find(|e| e.hash.starts_with(git_ref) || git_ref.contains('~'));
        if let Some(entry) = target {
            println!("Rolling back to: {} {}", entry.hash, entry.message);
        } else {
            println!("Rolling back to: {git_ref}");
        }
        println!("\n--dry-run specified, no changes applied.");
        return Ok(());
    }

    // Confirm
    print!("Roll back policy to {git_ref}? [y/N] ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("y") {
        println!("Aborted.");
        return Ok(());
    }

    // Use PolicyService RPC
    let result_msg = client.rollback(git_ref).await
        .context("Failed to rollback via PolicyService. Are services running?")?;

    println!();
    println!("✓ Policy rolled back to {git_ref}");
    println!("  {result_msg}");

    Ok(())
}

/// Handle `policy check <user> <resource> <action>` - Test permission via RPC
pub async fn handle_policy_check(
    signing_key: &SigningKey,
    user: &str,
    resource: &str,
    action: &str,
) -> Result<()> {
    let client = create_policy_client(signing_key);

    let allowed = client.check(user, "*", resource, action).await
        .context("Failed to check policy via PolicyService. Are services running?")?;

    println!("User:     {user}");
    println!("Resource: {resource}");
    println!("Action:   {action}");
    println!();

    if allowed {
        println!("Result:   ✓ ALLOWED");
    } else {
        println!("Result:   ✗ DENIED");
    }

    Ok(())
}

// === Token handlers ===

/// Handle `policy token create` - Create a new JWT API token
///
/// Generates a JWT token signed with Ed25519. The token is self-contained
/// and can be validated statelessly by any server with the public key.
pub async fn handle_token_create(
    signing_key: &SigningKey,
    user: &str,
    name: Option<String>,
    expires: &str,
    scopes: Vec<String>,
) -> Result<()> {
    // Parse expiration
    let expires_duration = parse_duration(expires)?;

    // Default expiration is 90 days if not specified and not "never"
    let duration = expires_duration.unwrap_or_else(|| Duration::days(90));

    // Generate name for display (not stored in JWT, just for user reference)
    let name = name.unwrap_or_else(|| {
        format!("token-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"))
    });

    // Create JWT claims with bare username as subject.
    // Scopes are not embedded in JWT - Casbin enforces authorization server-side.
    let now = chrono::Utc::now().timestamp();
    let exp = (chrono::Utc::now() + duration).timestamp();
    let claims = Claims::new(user.to_owned(), now, exp);

    // Encode and sign the JWT
    let token = jwt::encode(&claims, signing_key);

    // Display the token (only shown once)
    println!();
    println!("JWT token created for subject '{user}':");
    println!();
    println!("  {token}");
    println!();
    println!("  Name:    {name} (reference only, not in token)");

    // Calculate expiration for display
    let expires_at = chrono::DateTime::from_timestamp(exp, 0);
    if let Some(expires_at) = expires_at {
        let days = duration.num_days();
        println!("  Expires: {} ({} days)", expires_at.format("%Y-%m-%d %H:%M UTC"), days);
    } else {
        println!("  Expires: never");
    }

    if scopes.is_empty() || scopes.contains(&"*".to_owned()) {
        println!("  Scopes:  * (all resources)");
    } else {
        println!("  Scopes:  {}", scopes.join(", "));
    }

    println!();
    println!("  \x1b[33m⚠️  Save this token now - it cannot be retrieved again.\x1b[0m");
    println!();
    println!("  Usage:");
    let display_len = std::cmp::min(30, token.len());
    println!("    curl -H \"Authorization: Bearer {}...\" http://localhost:8080/v1/models", &token[..display_len]);
    println!();

    Ok(())
}


/// Load or generate the signing key from .registry/keys/signing.key
pub async fn load_or_generate_signing_key(keys_dir: &Path) -> Result<SigningKey> {
    let key_path = keys_dir.join("signing.key");

    if key_path.exists() {
        // Load existing key
        let key_bytes = tokio::fs::read(&key_path).await?;
        if key_bytes.len() != 32 {
            anyhow::bail!("Invalid signing key file: expected 32 bytes, got {}", key_bytes.len());
        }
        let mut key_array = [0u8; 32];
        key_array.copy_from_slice(&key_bytes);
        let signing_key = SigningKey::from_bytes(&key_array);
        info!("Loaded signing key from {:?}", key_path);
        Ok(signing_key)
    } else {
        // Generate new key
        tokio::fs::create_dir_all(keys_dir).await?;
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        tokio::fs::write(&key_path, signing_key.to_bytes()).await?;

        // Set restrictive permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&key_path).await?.permissions();
            perms.set_mode(0o600);
            tokio::fs::set_permissions(&key_path, perms).await?;
        }

        info!("Generated new signing key at {:?}", key_path);
        Ok(signing_key)
    }
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
        .context(format!("Invalid duration number: {num_str}"))?;

    let duration = match unit {
        'd' => Duration::days(num),
        'y' => Duration::days(num * 365),
        'h' => Duration::hours(num),
        _ => anyhow::bail!("Unknown duration unit: {}", unit),
    };

    Ok(Some(duration))
}

// === Helper functions ===

/// Truncate a string to max length, adding "..." if truncated.
/// Uses char_indices for O(n) slicing without intermediate allocation.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_owned()
    } else if max_len > 3 {
        let end = s.char_indices()
            .nth(max_len - 3)
            .map_or(s.len(), |(i, _)| i);
        format!("{}...", &s[..end])
    } else {
        let end = s.char_indices()
            .nth(max_len)
            .map_or(s.len(), |(i, _)| i);
        s[..end].to_owned()
    }
}

// Re-export policy templates from shared location
pub use crate::auth::policy_templates::{PolicyTemplate, get_template, get_templates};

/// Handle `policy list-templates` - List available templates
pub async fn handle_policy_list_templates() -> Result<()> {
    let templates = get_templates();

    println!("Available policy templates:\n");
    println!("{:<20} DESCRIPTION", "NAME");
    println!("{}", "-".repeat(60));

    for template in templates {
        println!("{:<20} {}", template.name, template.description);
    }

    println!();
    println!("Apply with: hyprstream policy apply-template <name>");

    Ok(())
}

/// Handle `policy apply-template <name>` - Apply a built-in template via RPC
pub async fn handle_policy_apply_template(
    signing_key: &SigningKey,
    template_name: &str,
    dry_run: bool,
) -> Result<()> {
    let template = get_template(template_name)
        .ok_or_else(|| anyhow::anyhow!(
            "Unknown template: '{}'. Use 'hyprstream policy list-templates' to see available templates.",
            template_name
        ))?;

    let new_content = template.expanded_rules();

    println!("Applying template: {template_name}");
    println!("Description: {}", template.description);
    println!();
    println!("Rules:");
    for line in new_content.lines() {
        if !line.trim().is_empty() && !line.starts_with('#') {
            println!("  {line}");
        }
    }
    println!();

    if dry_run {
        println!("--dry-run specified, no changes applied.");
        return Ok(());
    }

    // Use PolicyService RPC to apply the template (writes file, validates, stages, commits)
    let client = create_policy_client(signing_key);
    let result_msg = client.apply_template(template_name).await
        .context("Failed to apply template via PolicyService. Are services running?")?;

    println!();
    println!("✓ Template '{template_name}' applied successfully.");
    println!("  {result_msg}");

    Ok(())
}
