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
use crate::services::generated::policy_client::{
    PolicyClient, GetHistory, GetDiff, ApplyDraft, RollbackPolicy, PolicyCheck, ApplyTemplate,
    AddGrouping, RemoveGrouping,
};
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
    PolicyClient::new(signing_key.clone(), RequestIdentity::anonymous())
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
            println!("  hyprstream quick policy apply-template local    # local CLI full access");
            println!("\nOr edit manually:");
            println!("  hyprstream quick policy edit");
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
    let history = client.get_history(&GetHistory { count: count as u32 }).await
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
    println!("  hyprstream quick policy diff     # Preview changes");
    println!("  hyprstream quick policy apply    # Commit changes");
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
            println!("✏️  Draft changes detected ({}). Run 'hyprstream quick policy diff' to review.", draft.summary);
        }
        Ok(_) => {} // No changes
        Err(_) => {
            // Service may not be running; just suggest the commands
            println!();
            println!("Run 'hyprstream quick policy diff' to review changes.");
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

    let diff_text = client.get_diff(&GetDiff { git_ref: Some(git_ref.to_owned()) }).await
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
    let diff_text = client.get_diff(&GetDiff { git_ref: Some(String::new()) }).await
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

    let result_msg = client.apply_draft(&ApplyDraft { message: Some(commit_msg) }).await
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
        let history = client.get_history(&GetHistory { count: 20 }).await
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
    let result_msg = client.rollback(&RollbackPolicy { git_ref: git_ref.to_owned() }).await
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

    let allowed = client.check(&PolicyCheck { subject: user.to_owned(), domain: "*".to_owned(), resource: resource.to_owned(), operation: action.to_owned() }).await
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

    // Mint JWT with proper iss/aud claims for OAI middleware compatibility
    let (token, exp) = mint_local_token(signing_key, user, duration);

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


/// Load or generate the Ed25519 signing key using the OS keyring exclusively.
///
/// Supported platforms:
/// - Linux: kernel keyutils (`linux-native` feature)
/// - macOS / iOS: Keychain (`apple-native` feature)
/// - Windows: Credential Manager (`windows-native` feature)
/// - FreeBSD / OpenBSD: DBus Secret Service (`sync-secret-service` + `crypto-rust` features)
///
/// For platforms without a supported keyring daemon, supply a client-provided
/// credential store via [`keyring::set_default_credential_builder`] before
/// calling this function.
///
/// The `_keys_dir` parameter is retained for API compatibility but unused;
/// keys are stored exclusively in the OS keyring.
pub async fn load_or_generate_signing_key(_keys_dir: &Path) -> Result<SigningKey> {
    // Check for test bypass via config before touching the OS keyring.
    // Set HYPRSTREAM__SIGNING_KEY=<hex32> to inject a pre-generated key.
    if let Ok(cfg) = crate::config::HyprConfig::load() {
        if let Some(ref hex_key) = cfg.signing_key {
            let bytes = hex::decode(hex_key)
                .map_err(|e| anyhow::anyhow!("HYPRSTREAM__SIGNING_KEY: invalid hex: {e}"))?;
            let arr: [u8; 32] = bytes.try_into()
                .map_err(|_| anyhow::anyhow!("HYPRSTREAM__SIGNING_KEY: expected 32 bytes"))?;
            tracing::info!("Using node signing key from config (test bypass)");
            return Ok(SigningKey::from_bytes(&arr));
        }
    }

    const SERVICE: &str = "hyprstream";
    const KEY_NAME: &str = "signing-key";

    let entry = keyring::Entry::new(SERVICE, KEY_NAME).or_else(|_| {
        // On Linux, the session keyring may be revoked (common in systemd user
        // sessions after the original login session ends). Try to join/create
        // a new session keyring and retry.
        #[cfg(target_os = "linux")]
        {
            tracing::debug!("session keyring unavailable, creating new session");
            // KEYCTL_JOIN_SESSION_KEYRING = 1, NULL name = create anonymous session
            let rc = unsafe { libc::syscall(libc::SYS_keyctl, 1i32, std::ptr::null::<u8>()) };
            if rc >= 0 {
                return keyring::Entry::new(SERVICE, KEY_NAME);
            }
        }
        keyring::Entry::new(SERVICE, KEY_NAME)
    }).map_err(|e| {
        anyhow::anyhow!(
            "Failed to access OS keyring (service={SERVICE}, key={KEY_NAME}): {e}.\n\
             On Linux, ensure kernel keyutils is available.\n\
             On FreeBSD/OpenBSD, ensure a Secret Service daemon (e.g. gnome-keyring \
             or kwallet) is running, or supply a custom credential store via \
             keyring::set_default_credential_builder()."
        )
    })?;

    match entry.get_secret() {
        Ok(bytes) if bytes.len() == 32 => {
            let arr: [u8; 32] = bytes.try_into().map_err(|_| anyhow::anyhow!("unreachable: length already checked"))?;
            info!("Loaded Ed25519 signing key from OS keyring");
            return Ok(SigningKey::from_bytes(&arr));
        }
        Ok(_) => {
            tracing::warn!("Keyring entry for signing key has wrong size; regenerating");
        }
        Err(keyring::Error::NoEntry) => {
            // No key stored yet; generate a new one below
        }
        Err(e) => {
            anyhow::bail!(
                "OS keyring error while loading signing key: {e}.\n\
                 Ensure the keyring daemon is accessible. On BSD without a \
                 Secret Service daemon, supply a custom credential store via \
                 keyring::set_default_credential_builder()."
            );
        }
    }

    // Generate new key and persist it in the keyring
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    entry.set_secret(&key.to_bytes()).map_err(|e| {
        anyhow::anyhow!(
            "Failed to store signing key in OS keyring: {e}.\n\
             Ensure the keyring daemon is accessible, or supply a custom \
             credential store via keyring::set_default_credential_builder()."
        )
    })?;
    info!("Generated and stored new Ed25519 signing key in OS keyring");
    Ok(key)
}

/// Mint a JWT locally with proper iss/aud claims.
///
/// Sets `iss` to the OAuth issuer URL and `aud` to the OAI resource URL,
/// matching the claim structure expected by the OAI HTTP middleware.
/// Does NOT require a running PolicyService — signs directly with the local key.
pub(crate) fn mint_local_token(
    signing_key: &SigningKey,
    subject: &str,
    duration: Duration,
) -> (String, i64) {
    let now = chrono::Utc::now().timestamp();
    let exp = now + duration.num_seconds();

    let config = crate::config::HyprConfig::load();
    let (issuer, audience) = match config {
        Ok(ref c) => (c.oauth.issuer_url(), c.oai.resource_url()),
        Err(_) => {
            // Fallback to defaults when config is not available
            ("http://localhost:6791".to_owned(), "http://localhost:6789".to_owned())
        }
    };

    let claims = Claims::new(subject.to_owned(), now, exp)
        .with_issuer(issuer)
        .with_audience(Some(audience));

    let token = jwt::encode(&claims, signing_key);
    (token, exp)
}

/// Keyring service name used for all hyprstream keyring entries.
pub(crate) const KEYRING_SERVICE: &str = "hyprstream";
/// Keyring key name for the user's Ed25519 signing key (for OAuth device flow challenges).
// The `user-signing-key` is separate from the node `signing-key`:
// - `signing-key`: server identity — signs JWTs and RPC envelopes
// - `user-signing-key`: client identity — signs OAuth device flow challenges
pub(crate) const KEYRING_USER_KEY_NAME: &str = "user-signing-key";

/// Ensure the local user has an Ed25519 identity keypair.
///
/// If a keypair already exists in the OS keyring, returns the existing
/// verifying key. Otherwise generates a new keypair, stores the private
/// key in the keyring, and returns the public key for UserStore registration.
///
/// This key is per-OS-user (not per-hyprstream-user). The wizard only
/// registers it for the local admin. Other users generate their own keys.
pub(crate) fn ensure_user_identity() -> Result<(SigningKey, ed25519_dalek::VerifyingKey)> {
    // Check for test bypass via config before touching the OS keyring.
    // Set HYPRSTREAM__OAUTH__USER_SIGNING_KEY=<hex32> to inject a pre-generated key.
    if let Ok(cfg) = crate::config::HyprConfig::load() {
        if let Some(ref hex_key) = cfg.oauth.user_signing_key {
            let bytes = hex::decode(hex_key)
                .map_err(|e| anyhow::anyhow!("HYPRSTREAM__OAUTH__USER_SIGNING_KEY: invalid hex: {e}"))?;
            let arr: [u8; 32] = bytes
                .try_into()
                .map_err(|_| anyhow::anyhow!("HYPRSTREAM__OAUTH__USER_SIGNING_KEY: expected 32 bytes"))?;
            let sk = SigningKey::from_bytes(&arr);
            info!("Using user signing key from config (test bypass)");
            return Ok((sk.clone(), sk.verifying_key()));
        }
    }

    const SERVICE: &str = KEYRING_SERVICE;
    const KEY_NAME: &str = KEYRING_USER_KEY_NAME;

    let entry = keyring::Entry::new(SERVICE, KEY_NAME).map_err(|e| {
        anyhow::anyhow!(
            "Failed to access OS keyring for user identity (service={SERVICE}, key={KEY_NAME}): {e}.\n\
             On Linux, ensure kernel keyutils is available.\n\
             On FreeBSD/OpenBSD, ensure a Secret Service daemon is running."
        )
    })?;

    match entry.get_secret() {
        Ok(bytes) if bytes.len() == 32 => {
            let arr: [u8; 32] = bytes.try_into()
                .map_err(|_| anyhow::anyhow!("unreachable: length already checked"))?;
            let sk = SigningKey::from_bytes(&arr);
            info!("Loaded user identity key from OS keyring");
            Ok((sk.clone(), sk.verifying_key()))
        }
        Ok(bad) => {
            anyhow::bail!(
                "Keyring entry '{KEY_NAME}' has wrong size ({} bytes, expected 32).\n\
                 This indicates corruption. Remove it manually and re-run:\n\
                 - Linux:   keyctl purge user hyprstream:{KEY_NAME}\n\
                 - macOS:   security delete-generic-password -s hyprstream -a {KEY_NAME}\n\
                 - Windows: cmdkey /delete:hyprstream:{KEY_NAME}",
                bad.len()
            );
        }
        Err(keyring::Error::NoEntry) => {
            let sk = SigningKey::generate(&mut rand::rngs::OsRng);
            entry.set_secret(&sk.to_bytes()).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to store user identity key in OS keyring: {e}.\n\
                     Ensure the keyring daemon is accessible."
                )
            })?;
            info!("Generated and stored new user identity key in OS keyring");
            Ok((sk.clone(), sk.verifying_key()))
        }
        Err(e) => anyhow::bail!("OS keyring error while loading user identity key: {e}"),
    }
}

/// Parse duration string like "30d", "90d", "1y", "never"
pub(crate) fn parse_duration(s: &str) -> Result<Option<Duration>> {
    let s = s.trim().to_lowercase();

    if s.is_empty() {
        return Ok(None);
    }

    if s == "never" {
        return Ok(Some(Duration::days(36500))); // ~100 years
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

// === Role handlers ===

/// Handle `policy role add <user> <role>` - Assign a role to a user via RPC
pub async fn handle_policy_role_add(
    signing_key: &SigningKey,
    user: &str,
    role: &str,
    dry_run: bool,
) -> Result<()> {
    if dry_run {
        println!("Would add: g, {user}, {role}");
        return Ok(());
    }

    let client = create_policy_client(signing_key);
    let sha = client.add_grouping(&AddGrouping { user: user.to_owned(), role: role.to_owned() }).await
        .context("Failed to add role assignment via PolicyService. Are services running?")?;

    println!("Role assigned. Commit: {}", &sha[..sha.len().min(8)]);
    Ok(())
}

/// Handle `policy role remove <user> <role>` - Remove a role from a user via RPC
pub async fn handle_policy_role_remove(
    signing_key: &SigningKey,
    user: &str,
    role: &str,
    force: bool,
) -> Result<()> {
    if !force {
        print!("Remove role '{role}' from '{user}'? [y/N] ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    let client = create_policy_client(signing_key);
    let sha = client.remove_grouping(&RemoveGrouping { user: user.to_owned(), role: role.to_owned() }).await
        .context("Failed to remove role assignment via PolicyService. Are services running?")?;

    println!("Role removed. Commit: {}", &sha[..sha.len().min(8)]);
    Ok(())
}

/// Handle `policy role list` - List role assignments (g-lines) via RPC
pub async fn handle_policy_role_list(
    signing_key: &SigningKey,
    user: Option<&str>,
    role: Option<&str>,
) -> Result<()> {
    let client = create_policy_client(signing_key);
    let policy_info = client.get_policy().await
        .context("Failed to get policy from PolicyService. Are services running?")?;

    let groupings: Vec<_> = policy_info.groupings.iter()
        .filter(|g| user.is_none_or(|u| g.user == u))
        .filter(|g| role.is_none_or(|r| g.role == r))
        .collect();

    if groupings.is_empty() {
        println!("No role assignments found.");
        return Ok(());
    }

    for g in groupings {
        println!("g, {}, {}", g.user, g.role);
    }

    Ok(())
}

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
    println!("Apply with: hyprstream quick policy apply-template <name>");

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
            "Unknown template: '{}'. Use 'hyprstream quick policy list-templates' to see available templates.",
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
    let result_msg = client.apply_template(&ApplyTemplate { name: template_name.to_owned() }).await
        .context("Failed to apply template via PolicyService. Are services running?")?;

    println!();
    println!("✓ Template '{template_name}' applied successfully.");
    println!("  {result_msg}");

    Ok(())
}
