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
use crate::auth::{jwt, write_policy_file, Claims, PolicyManager};
use crate::cli::policy_handlers::load_or_generate_signing_key;
use crate::cli::service_handlers::{
    handle_service_start, print_check, run_repair_checks, CheckStatus,
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

    // Phase 1: Bootstrap environment
    phase_bootstrap(&mut state, non_interactive).await?;

    // Phase 2: Policy template selection
    phase_policy_templates(&mut state, non_interactive).await?;

    // Phase 3: User/role creation
    phase_users(&mut state, non_interactive).await?;

    // Phase 4: Token generation
    phase_tokens(&mut state, non_interactive).await?;

    // Phase 5: Service startup
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
// Phase 2: Policy Templates
// ─────────────────────────────────────────────────────────────────────────────

async fn phase_policy_templates(state: &mut WizardState, non_interactive: bool) -> Result<()> {
    println!("  Phase 2: Policy Template");
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
    let local_user = hyprstream_rpc::envelope::RequestIdentity::local()
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
// Phase 3: Users
// ─────────────────────────────────────────────────────────────────────────────

async fn phase_users(state: &mut WizardState, non_interactive: bool) -> Result<()> {
    println!("  Phase 3: Users & Roles");
    println!("  {}", "-".repeat(40));
    println!();

    if non_interactive {
        println!("    Skipping user creation (non-interactive mode).");
        println!();
        return Ok(());
    }

    let add_users = Confirm::new("  Add a user?")
        .with_default(true)
        .prompt()
        .unwrap_or(false);

    if !add_users {
        println!("    Skipping user creation.");
        println!();
        return Ok(());
    }

    let pm = PolicyManager::new(&state.policies_dir()).await
        .context("Failed to load PolicyManager")?;

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
            let action_options: Vec<String> = actions.iter().map(|a| (*a).to_string()).collect();

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

        // Ask to add another
        let add_another = Confirm::new("  Add another user?")
            .with_default(false)
            .prompt()
            .unwrap_or(false);

        if !add_another {
            break;
        }
    }

    // Save policies to disk
    pm.save().await.context("Failed to save policies")?;
    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Tokens
// ─────────────────────────────────────────────────────────────────────────────

async fn phase_tokens(state: &mut WizardState, non_interactive: bool) -> Result<()> {
    println!("  Phase 4: API Tokens");
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
        let local_user = hyprstream_rpc::envelope::RequestIdentity::local()
            .user()
            .to_owned();

        generate_token(state, &signing_key, &local_user, "90d")?;
        println!();
        return Ok(());
    }

    // If no users created, ask if they want to create a token for the local user
    if users_for_tokens.is_empty() {
        let local_user = hyprstream_rpc::envelope::RequestIdentity::local()
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
    let duration = match duration_str {
        "30d" => chrono::Duration::days(30),
        "1y" => chrono::Duration::days(365),
        "never" => chrono::Duration::days(365 * 100),
        // "90d" and any other
        _ => chrono::Duration::days(90),
    };

    let now = chrono::Utc::now().timestamp();
    let exp = (chrono::Utc::now() + duration).timestamp();
    let claims = Claims::new(username.to_owned(), now, exp);
    let token = jwt::encode(&claims, signing_key);

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
// Phase 5: Services
// ─────────────────────────────────────────────────────────────────────────────

async fn phase_services(
    _state: &WizardState,
    config_services: &[String],
    non_interactive: bool,
    start_flag: bool,
) -> Result<()> {
    println!("  Phase 5: Services");
    println!("  {}", "-".repeat(40));
    println!();

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
