//! Interactive setup wizard for bootstrapping hyprstream
//!
//! Guides new users through environment setup, policy configuration,
//! user/role creation, API token generation, and optional service startup.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use inquire::{Confirm, Select, Text};

use crate::auth::policy_templates::{get_templates, PolicyTemplate};
use crate::auth::{write_policy_file, LocalKeyStore, PolicyManager, UserStore};
use crate::cli::policy_handlers::{
    ensure_user_signing_key, load_or_generate_signing_key, mint_local_token, parse_duration,
};
use crate::cli::service_handlers::{
    build_version, format_size, handle_service_install, handle_service_start,
    is_binary_installed, print_check, run_repair_checks, CheckStatus, InstallPlan,
};

// ─────────────────────────────────────────────────────────────────────────────
// State tracking
// ─────────────────────────────────────────────────────────────────────────────

struct WizardState {
    models_dir: PathBuf,
    signing_key: Option<SigningKey>,
    templates_applied: Vec<String>,
    users_created: Vec<UserRecord>,
    tokens_generated: Vec<TokenRecord>,
}

struct UserRecord {
    username: String,
    role: String,
}

struct TokenRecord {
    username: String,
    token: String,
    expires: String,
}

impl WizardState {
    fn new(models_dir: PathBuf) -> Self {
        Self {
            models_dir,
            signing_key: None,
            templates_applied: Vec::new(),
            users_created: Vec::new(),
            tokens_generated: Vec::new(),
        }
    }

    fn policies_dir(&self) -> PathBuf {
        self.models_dir.join(".registry").join("policies")
    }

    fn keys_dir(&self) -> PathBuf {
        self.models_dir.join(".registry").join("keys")
    }

    fn credentials_dir(&self) -> PathBuf {
        crate::config::HyprConfig::load()
            .map(|c| c.config_dir().join("credentials"))
            .unwrap_or_else(|_|
                dirs::config_dir()
                    .unwrap_or_else(|| self.models_dir.clone())
                    .join("hyprstream")
                    .join("credentials")
            )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Predefined roles
// ─────────────────────────────────────────────────────────────────────────────

struct RoleDef {
    name: &'static str,
    description: &'static str,
    rules: &'static [(&'static str, &'static str)], // (resource, action)
}

const PREDEFINED_ROLES: &[RoleDef] = &[
    RoleDef {
        name: "admin",
        description: "Full access to everything",
        rules: &[("*", "*")],
    },
    RoleDef {
        name: "operator",
        description: "Infer + query + load/unload models",
        rules: &[
            ("model:*", "infer"),
            ("model:*", "query"),
            ("model:*", "serve"),
        ],
    },
    RoleDef {
        name: "viewer",
        description: "Read-only queries",
        rules: &[("model:*", "query"), ("registry:*", "query")],
    },
    RoleDef {
        name: "trainer",
        description: "Inference + training",
        rules: &[
            ("model:*", "infer"),
            ("model:*", "query"),
            ("model:*", "serve"),
            ("model:*", "train"),
        ],
    },
];

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `hyprstream wizard --tui` — TUI setup wizard with bootstrap manager.
///
/// Runs the TUI on a blocking thread so that `BootstrapManager`'s synchronous
/// trait methods can safely call `block_on` without panicking inside Tokio.
pub async fn handle_wizard_tui(models_dir: &Path) -> Result<()> {
    let rt = tokio::runtime::Handle::current();
    let models_dir = models_dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let backend =
            crate::cli::bootstrap_manager::BootstrapManager::new(rt, models_dir);
        let app = hyprstream_tui::wizard::WizardApp::new(backend);
        waxterm::run_sync(app, waxterm::TerminalConfig::new())
    })
    .await??;
    Ok(())
}

/// Handle `hyprstream wizard` — interactive setup wizard
pub async fn handle_wizard(
    models_dir: &Path,
    config_services: &[String],
    non_interactive: bool,
    start_services: bool,
) -> Result<()> {
    println!();
    println!("  Hyprstream Setup Wizard");
    println!("  {}", "=".repeat(40));
    println!();

    let mut state = WizardState::new(models_dir.to_path_buf());

    // Phase 1: Environment bootstrap
    phase_bootstrap(&mut state, non_interactive).await?;

    // Phase 2: Binary installation
    phase_binary_install(&mut state, non_interactive)?;

    // Phase 3: Policy template selection
    phase_policy_templates(&mut state, non_interactive).await?;

    // Phase 4: User/role creation
    phase_users(&mut state, non_interactive).await?;

    // Phase 5: Token generation
    phase_tokens(&mut state, non_interactive).await?;

    // Phase 6: Service startup
    phase_services(&state, config_services, non_interactive, start_services).await?;

    // Summary
    print_summary(&state);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

async fn phase_bootstrap(state: &mut WizardState, _non_interactive: bool) -> Result<()> {
    println!("  Phase 1: Environment Bootstrap");
    println!("  {}", "-".repeat(40));
    println!();

    // Reuse the repair checks from service install
    run_repair_checks(&state.models_dir, false).await?;

    // Load the signing key (repair checks ensure it exists)
    let signing_key = load_or_generate_signing_key(&state.keys_dir()).await?;
    state.signing_key = Some(signing_key);

    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Binary Installation
// ─────────────────────────────────────────────────────────────────────────────

fn phase_binary_install(_state: &mut WizardState, non_interactive: bool) -> Result<()> {
    println!("  Phase 2: Binary Installation");
    println!("  {}", "-".repeat(40));
    println!();

    // Check if already installed via current_exe() (kernel-verified path)
    if let Some(installed_path) = is_binary_installed() {
        let version = build_version();
        print_check(
            "Binary",
            CheckStatus::Ok,
            &format!("installed ({version}) at {}", installed_path.display()),
        );
        println!();
        return Ok(());
    }

    // Not installed — prepare install plan (detect, validate, check space)
    let plan = match InstallPlan::prepare() {
        Ok(p) => p,
        Err(e) => {
            print_check(
                "Binary",
                CheckStatus::Warn,
                &format!("cannot prepare: {e}"),
            );
            println!();
            return Ok(());
        }
    };

    // Display plan details
    println!(
        "    Source: {} ({}, {})",
        plan.source.display(),
        plan.type_label(),
        format_size(plan.source_size),
    );
    println!("    Target: {}/hyprstream", plan.bin_dir.display());

    if plan.available_space > 0 {
        println!("    Disk:   {} available", format_size(plan.available_space));
    }
    println!();

    // Check if there's enough space
    if !plan.has_sufficient_space() {
        print_check(
            "Binary",
            CheckStatus::Warn,
            &format!(
                "insufficient disk space ({} needed, {} available)",
                format_size(plan.source_size),
                format_size(plan.available_space),
            ),
        );
        println!();
        return Ok(());
    }

    // Prompt user (or auto-accept in non-interactive mode)
    let should_install = if non_interactive {
        true
    } else {
        Confirm::new("  Install hyprstream to your PATH?")
            .with_default(true)
            .prompt()
            .unwrap_or(false)
    };

    if !should_install {
        println!("    Skipping binary installation.");
        println!();
        return Ok(());
    }

    // Execute the plan
    match plan.execute() {
        Ok(result) => {
            print_check(
                "Binary",
                CheckStatus::Ok,
                &format!(
                    "installed ({}) to {}",
                    result.type_label(),
                    result.bin_dir.join("hyprstream").display()
                ),
            );
            println!("    Version store: {}", result.version_dir.display());
            if !result.updated_profiles.is_empty() {
                println!("    PATH updated:  {}", result.updated_profiles.join(", "));
            }
        }
        Err(e) => {
            print_check("Binary", CheckStatus::Fail, &format!("installation failed: {e}"));
        }
    }

    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Policy Templates
// ─────────────────────────────────────────────────────────────────────────────

async fn phase_policy_templates(state: &mut WizardState, non_interactive: bool) -> Result<()> {
    println!("  Phase 3: Policy Template");
    println!("  {}", "-".repeat(40));
    println!();

    // Check if policies already exist with rules
    let policy_csv = state.policies_dir().join("policy.csv");
    let has_existing_rules = if policy_csv.exists() {
        let content = tokio::fs::read_to_string(&policy_csv).await.unwrap_or_default();
        content.lines().any(|l| l.starts_with("p,") || l.starts_with("p "))
    } else {
        false
    };

    if has_existing_rules {
        print_check("Policy", CheckStatus::Ok, "existing policy rules found");

        if non_interactive {
            println!("    Keeping existing policy (non-interactive mode).");
            println!();
            return Ok(());
        }

        let overwrite = Confirm::new("  Existing policy rules found. Replace with a template?")
            .with_default(false)
            .prompt()
            .unwrap_or(false);

        if !overwrite {
            println!("    Keeping existing policy.");
            println!();
            return Ok(());
        }
    }

    let templates = get_templates();

    if non_interactive {
        // Apply "local" template by default
        apply_template(state, &templates[0]).await?;
        return Ok(());
    }

    // Build selection options
    let local_user = hyprstream_rpc::envelope::RequestIdentity::anonymous()
        .user()
        .to_owned();

    let mut options: Vec<String> = templates
        .iter()
        .map(|t| {
            if t.name == "local" {
                format!("local — Full access for {} (recommended)", local_user)
            } else {
                format!("{} — {}", t.name, t.description)
            }
        })
        .collect();
    options.push("None — skip template".to_owned());

    let selection = Select::new("  Select a policy template:", options)
        .prompt()
        .context("Template selection cancelled")?;

    if selection.starts_with("None") {
        println!("    Skipping template.");
        println!();
        return Ok(());
    }

    // Find selected template
    let template_name = selection.split(" —").next().unwrap_or("").trim();
    if let Some(template) = templates.iter().find(|t| t.name == template_name) {
        apply_template(state, template).await?;
    }

    Ok(())
}

async fn apply_template(state: &mut WizardState, template: &PolicyTemplate) -> Result<()> {
    let content = template.expanded_rules();
    let policy_csv = state.policies_dir().join("policy.csv");

    // Save current content for rollback
    let backup = if policy_csv.exists() {
        tokio::fs::read_to_string(&policy_csv).await.ok()
    } else {
        None
    };

    // Write new policy
    write_policy_file(&policy_csv, content.as_bytes()).await
        .context("Failed to write policy file")?;

    // Validate by loading
    match PolicyManager::new(&state.policies_dir()).await {
        Ok(_) => {
            print_check("Template", CheckStatus::Ok, &format!("applied '{}'", template.name));
            state.templates_applied.push(template.name.to_owned());
        }
        Err(e) => {
            // Rollback
            if let Some(backup_content) = backup {
                let _ = write_policy_file(&policy_csv, backup_content.as_bytes()).await;
            }
            print_check("Template", CheckStatus::Fail, &format!("validation failed: {e}"));
            return Err(e.into());
        }
    }

    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Users
// ─────────────────────────────────────────────────────────────────────────────

async fn phase_users(state: &mut WizardState, non_interactive: bool) -> Result<()> {
    println!("  Phase 4: Users & Roles");
    println!("  {}", "-".repeat(40));
    println!();

    let pm = PolicyManager::new(&state.policies_dir()).await
        .context("Failed to load PolicyManager")?;

    let credentials_dir = state.credentials_dir();
    let mut user_store = LocalKeyStore::load(&credentials_dir)
        .context("Failed to open credential store")?;

    // Separate Casbin subjects into local (bare names) vs federated/OIDC (contain "://")
    let existing_policies = pm.get_policy().await;
    let local_policy_users: std::collections::BTreeSet<String> = existing_policies.iter()
        .filter_map(|rule| rule.first())
        .filter(|s| *s != "*" && !s.contains("://"))
        .cloned()
        .collect();

    let federated_subjects: std::collections::BTreeSet<String> = existing_policies.iter()
        .filter_map(|rule| rule.first())
        .filter(|s| s.contains("://"))
        .cloned()
        .collect();

    let registered_users: std::collections::BTreeSet<String> =
        user_store.list_users().into_iter().collect();

    // Section 1: Local users (UserStore is authoritative)
    if !registered_users.is_empty() || !local_policy_users.is_empty() {
        println!("    Local users:");
        for user in &registered_users {
            let has_policy = local_policy_users.contains(user.as_str());
            if has_policy {
                print_check(user, CheckStatus::Ok, "identity + policy");
            } else {
                print_check(user, CheckStatus::Warn, "identity only — add policy rules");
            }
        }
        // Orphaned local policies: in Casbin but not in UserStore
        for user in local_policy_users.difference(&registered_users) {
            print_check(user, CheckStatus::Warn,
                "orphaned policy — no local identity (run 'hyprstream user register <username>')");
        }
        println!();
    }

    // Section 2: Federated/OIDC policy subjects (informational only)
    if !federated_subjects.is_empty() {
        println!("    Federated/OIDC policy subjects:");
        for subject in &federated_subjects {
            print_check(subject, CheckStatus::Info, "externally authenticated");
        }
        println!();
    }

    let local_user = hyprstream_rpc::envelope::RequestIdentity::anonymous()
        .user()
        .to_owned();

    // Non-interactive: auto-create admin if no local users exist in UserStore
    if non_interactive {
        if registered_users.is_empty() {
            // Register identity FIRST — UserStore is authoritative
            let (_sk, vk) = ensure_user_signing_key()?;
            user_store.register(&local_user, vk)
                .context("Failed to register identity")?;

            pm.add_policy_with_domain(&local_user, "*", "*", "*", "allow")
                .await
                .context("Failed to add admin policy")?;
            pm.save().await.context("Failed to save policies")?;

            print_check(&local_user, CheckStatus::Ok, "admin (identity + policy)");
            state.users_created.push(UserRecord {
                username: local_user,
                role: "admin".to_owned(),
            });
        } else {
            println!("    Local users already configured. Skipping.");
        }
        println!();
        return Ok(());
    }

    // Interactive: default YES only if no registered local users (UserStore is authoritative)
    let has_local_users = !registered_users.is_empty();
    let add_users = Confirm::new("  Add a local user?")
        .with_default(!has_local_users)
        .prompt()
        .unwrap_or(false);

    if !add_users {
        println!("    Skipping user creation.");
        println!();
        return Ok(());
    }

    loop {
        // Username
        let username = Text::new("  Username:")
            .prompt()
            .context("Username input cancelled")?;

        if username.trim().is_empty() {
            println!("    Username cannot be empty.");
            continue;
        }
        let username = username.trim().to_owned();

        if local_policy_users.contains(username.as_str()) {
            println!("    Note: '{}' already has policy rules. New permissions will be added.", username);
        }

        // Register local identity BEFORE adding any Casbin policies (Step 3)
        if username == local_user {
            if user_store.get_pubkey(&username)?.is_none() {
                let (_sk, vk) = ensure_user_signing_key()?;
                user_store.register(&username, vk)?;
                print_check(&username, CheckStatus::Ok, "identity registered (OAuth ready)");
            } else {
                print_check(&username, CheckStatus::Ok, "identity already registered");
            }
        } else {
            println!("    To enable OAuth for '{}': run 'hyprstream user register' on their machine.", username);
        }

        // Role selection
        let mut role_options: Vec<String> = PREDEFINED_ROLES
            .iter()
            .map(|r| format!("{} — {}", r.name, r.description))
            .collect();
        role_options.push("custom — Define custom permissions".to_owned());

        let role_selection = Select::new("  Role:", role_options)
            .prompt()
            .context("Role selection cancelled")?;

        let role_name = role_selection.split(" —").next().unwrap_or("").trim();

        if role_name == "custom" {
            // Custom role: ask for resource pattern and actions
            let resource = Text::new("  Resource pattern (e.g., model:*, registry:*):")
                .with_default("*")
                .prompt()
                .context("Resource input cancelled")?;

            let actions = &["infer", "train", "query", "write", "serve", "manage"];
            let action_options: Vec<String> = actions.iter().map(|a| (*a).to_owned()).collect();

            let selected_actions = inquire::MultiSelect::new(
                "  Actions to allow:",
                action_options,
            )
            .prompt()
            .context("Action selection cancelled")?;

            for action in &selected_actions {
                pm.add_policy_with_domain(&username, "*", &resource, action, "allow")
                    .await
                    .context("Failed to add policy")?;
            }

            let actions_str = selected_actions.join(",");
            print_check(
                &username,
                CheckStatus::Ok,
                &format!("custom ({actions_str} on {resource})"),
            );
            state.users_created.push(UserRecord {
                username: username.clone(),
                role: format!("custom({})", actions_str),
            });
        } else if let Some(role_def) = PREDEFINED_ROLES.iter().find(|r| r.name == role_name) {
            // Apply predefined role rules
            for (resource, action) in role_def.rules {
                pm.add_policy_with_domain(&username, "*", resource, action, "allow")
                    .await
                    .context("Failed to add policy")?;
            }

            print_check(&username, CheckStatus::Ok, role_def.name);
            state.users_created.push(UserRecord {
                username: username.clone(),
                role: role_name.to_owned(),
            });
        }

        // Save policies after each user's Casbin policies are added (Step 6)
        pm.save().await.context("Failed to save policies")?;

        // Ask to add another
        let add_another = Confirm::new("  Add another user?")
            .with_default(false)
            .prompt()
            .unwrap_or(false);

        if !add_another {
            break;
        }
    }

    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5: Tokens
// ─────────────────────────────────────────────────────────────────────────────

async fn phase_tokens(state: &mut WizardState, non_interactive: bool) -> Result<()> {
    println!("  Phase 5: API Tokens");
    println!("  {}", "-".repeat(40));
    println!();

    let signing_key = match state.signing_key.clone() {
        Some(key) => key,
        None => {
            println!("    No signing key available. Skipping token generation.");
            println!();
            return Ok(());
        }
    };

    // Collect users who need tokens
    let mut users_for_tokens: Vec<String> = state
        .users_created
        .iter()
        .map(|u| u.username.clone())
        .collect();

    // In non-interactive mode, generate a token for the local user
    if non_interactive {
        let local_user = hyprstream_rpc::envelope::RequestIdentity::anonymous()
            .user()
            .to_owned();

        generate_token(state, &signing_key, &local_user, "90d")?;
        println!();
        return Ok(());
    }

    // If no users created, ask if they want to create a token for the local user
    if users_for_tokens.is_empty() {
        let local_user = hyprstream_rpc::envelope::RequestIdentity::anonymous()
            .user()
            .to_owned();

        let create_local = Confirm::new(&format!(
            "  Generate API token for local user '{local_user}'?"
        ))
        .with_default(true)
        .prompt()
        .unwrap_or(false);

        if create_local {
            users_for_tokens.push(local_user);
        }
    }

    if users_for_tokens.is_empty() {
        println!("    No tokens to generate.");
        println!();
        return Ok(());
    }

    let expiration_options = vec![
        "30 days",
        "90 days (recommended)",
        "1 year",
        "never",
    ];

    for username in &users_for_tokens {
        let create = Confirm::new(&format!("  Generate API token for '{username}'?"))
            .with_default(true)
            .prompt()
            .unwrap_or(false);

        if !create {
            continue;
        }

        let expiry = Select::new("  Token expiration:", expiration_options.clone())
            .prompt()
            .context("Expiration selection cancelled")?;

        let duration_str = match expiry {
            "30 days" => "30d",
            "1 year" => "1y",
            "never" => "never",
            // "90 days (recommended)" and any other
            _ => "90d",
        };

        generate_token(state, &signing_key, username, duration_str)?;
    }

    println!();
    Ok(())
}

fn generate_token(
    state: &mut WizardState,
    signing_key: &SigningKey,
    username: &str,
    duration_str: &str,
) -> Result<()> {
    let duration = parse_duration(duration_str)?
        .unwrap_or_else(|| chrono::Duration::days(90));

    let (token, exp) = mint_local_token(signing_key, username, duration);

    let expires_display = if duration_str == "never" {
        "never".to_owned()
    } else {
        let expires_at = chrono::DateTime::from_timestamp(exp, 0);
        expires_at
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "unknown".to_owned())
    };

    println!();
    println!("    Token for '{username}':");
    println!("    {token}");
    println!();
    println!("    Expires: {expires_display}");
    let display_len = std::cmp::min(30, token.len());
    println!(
        "    Usage:   curl -H \"Authorization: Bearer {}...\" http://localhost:8080/v1/models",
        &token[..display_len]
    );
    println!();
    println!("    \x1b[33mSave this token now — it cannot be retrieved again.\x1b[0m");
    println!();

    state.tokens_generated.push(TokenRecord {
        username: username.to_owned(),
        token,
        expires: expires_display,
    });

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 6: Services
// ─────────────────────────────────────────────────────────────────────────────

async fn phase_services(
    state: &WizardState,
    config_services: &[String],
    non_interactive: bool,
    start_flag: bool,
) -> Result<()> {
    println!("  Phase 6: Services");
    println!("  {}", "-".repeat(40));
    println!();

    // Always install/update systemd units so they reflect the current binary and service list.
    if hyprstream_rpc::has_systemd() {
        handle_service_install(&state.models_dir, config_services, None, false, false).await?;
    }

    let should_start = if start_flag {
        true
    } else if non_interactive {
        false
    } else {
        Confirm::new("  Start services now?")
            .with_default(false)
            .prompt()
            .unwrap_or(false)
    };

    if should_start {
        handle_service_start(config_services, None, false).await?;
    } else {
        println!("    Services not started.");
        println!();
        println!("    To start later:");
        println!("      hyprstream service start");
        println!();
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────────────────────

fn print_summary(state: &WizardState) {
    println!("  {}", "=".repeat(40));
    println!("  Setup Complete");
    println!("  {}", "=".repeat(40));
    println!();

    // Templates
    if !state.templates_applied.is_empty() {
        println!("  Templates: {}", state.templates_applied.join(", "));
    }

    // Users
    if !state.users_created.is_empty() {
        println!("  Users:");
        for user in &state.users_created {
            println!("    {} ({})", user.username, user.role);
        }
    }

    // Tokens
    if !state.tokens_generated.is_empty() {
        println!("  Tokens:");
        for token in &state.tokens_generated {
            let preview = if token.token.len() > 20 {
                format!("{}...", &token.token[..20])
            } else {
                token.token.clone()
            };
            println!("    {} — {} (expires {})", token.username, preview, token.expires);
        }
    }

    if state.templates_applied.is_empty()
        && state.users_created.is_empty()
        && state.tokens_generated.is_empty()
    {
        println!("  Environment bootstrapped with default settings.");
    }

    println!();
    println!("  Next steps:");
    println!("    hyprstream service start          # Start services");
    println!("    hyprstream quick list              # List available models");
    println!("    hyprstream quick clone <model>     # Clone a model");
    println!("    hyprstream quick infer <model>     # Run inference");
    println!();
}
