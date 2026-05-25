//! Interactive setup wizard for bootstrapping hyprstream
//!
//! Guides new users through environment setup, policy configuration,
//! user/role creation, API token generation, and optional service startup.
//!
//! Both CLI (inquire-based text prompts) and TUI (ratatui) wizards share
//! the same `WizardBackend` trait for business logic, avoiding duplication.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::Path;

use anyhow::Result;
use inquire::{Confirm, Select, Text};

use hyprstream_tui::wizard::backend::*;

use crate::cli::service_handlers::{
    build_version, format_size, handle_service_install,
    is_binary_installed, print_check, CheckStatus, InstallPlan,
};

// ─────────────────────────────────────────────────────────────────────────────
// Predefined roles (display labels for inquire prompts)
// ─────────────────────────────────────────────────────────────────────────────

struct RoleDef {
    name: &'static str,
    description: &'static str,
}

const PREDEFINED_ROLES: &[RoleDef] = &[
    RoleDef {
        name: "admin",
        description: "Full access to everything",
    },
    RoleDef {
        name: "operator",
        description: "Infer + query + load/unload models",
    },
    RoleDef {
        name: "viewer",
        description: "Read-only queries",
    },
    RoleDef {
        name: "trainer",
        description: "Inference + training",
    },
];

// ─────────────────────────────────────────────────────────────────────────────
// Display tracking (for end-of-wizard summary)
// ─────────────────────────────────────────────────────────────────────────────

struct TextWizardSummary {
    templates_applied: Vec<String>,
    users_created: Vec<(String, String)>,
    tokens_generated: Vec<(String, String, String)>,
}

impl TextWizardSummary {
    fn new() -> Self {
        Self {
            templates_applied: Vec::new(),
            users_created: Vec::new(),
            tokens_generated: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry points
// ─────────────────────────────────────────────────────────────────────────────

/// Handle `hyprstream wizard --tui` — TUI setup wizard with bootstrap manager.
pub async fn handle_wizard_tui(models_dir: &Path, config_services: &[String]) -> Result<()> {
    let rt = tokio::runtime::Handle::current();
    let models_dir = models_dir.to_path_buf();
    let config_services = config_services.to_vec();
    tokio::task::spawn_blocking(move || {
        let backend =
            crate::cli::bootstrap_manager::BootstrapManager::new(rt, models_dir, config_services);
        let app = hyprstream_tui::wizard::WizardApp::new(backend);
        waxterm::run_sync(app, waxterm::TerminalConfig::new())
    })
    .await??;
    Ok(())
}

/// Handle `hyprstream wizard` — interactive setup wizard.
///
/// When `bootstrap_only` is true, only phase 1 (trust-root setup) runs.
pub async fn handle_wizard(
    models_dir: &Path,
    config_services: &[String],
    non_interactive: bool,
    start_services: bool,
    bootstrap_only: bool,
    enable_federation: bool,
) -> Result<()> {
    // Install systemd units before entering spawn_blocking (async operation).
    if !bootstrap_only && hyprstream_rpc::has_systemd() {
        handle_service_install(models_dir, config_services, None, false, false).await?;
    }

    let rt = tokio::runtime::Handle::current();
    let models_dir = models_dir.to_path_buf();
    let config_services = config_services.to_vec();

    tokio::task::spawn_blocking(move || {
        let mut backend =
            crate::cli::bootstrap_manager::BootstrapManager::new(rt, models_dir, config_services.clone());
        run_text_wizard(
            &mut backend,
            non_interactive,
            bootstrap_only,
            start_services,
            enable_federation,
            &config_services,
        )
    })
    .await??;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Text-mode wizard driver (delegates to WizardBackend)
// ─────────────────────────────────────────────────────────────────────────────

fn run_text_wizard(
    backend: &mut impl WizardBackend,
    non_interactive: bool,
    bootstrap_only: bool,
    start_services_flag: bool,
    enable_federation: bool,
    config_services: &[String],
) -> Result<()> {
    println!();
    println!("  Hyprstream Setup Wizard");
    println!("  {}", "=".repeat(40));
    println!();

    let mut summary = TextWizardSummary::new();

    // Phase 1: Environment bootstrap
    text_phase_bootstrap(backend)?;

    if bootstrap_only {
        return Ok(());
    }

    // Phase 2: Binary installation (standalone — not a WizardBackend concern)
    text_phase_binary_install(non_interactive)?;

    // Phase 3: Policy template selection
    text_phase_templates(backend, non_interactive, enable_federation, &mut summary)?;

    // Phase 4: User/role creation
    text_phase_users(backend, non_interactive, &mut summary)?;

    // Phase 5: Token generation
    text_phase_tokens(backend, non_interactive, &mut summary)?;

    // Phase 6: Service startup
    text_phase_services(backend, config_services, non_interactive, start_services_flag)?;

    // Summary
    print_summary(&summary);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Bootstrap (via WizardBackend)
// ─────────────────────────────────────────────────────────────────────────────

fn text_phase_bootstrap(backend: &mut impl WizardBackend) -> Result<()> {
    println!("  Phase 1: Environment Bootstrap");
    println!("  {}", "-".repeat(40));
    println!();

    backend.start_bootstrap();

    loop {
        match backend.poll_bootstrap() {
            BootstrapPoll::InProgress(msg) => {
                println!("    {msg}");
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            BootstrapPoll::Done(steps) => {
                for step in &steps {
                    print_check(step, CheckStatus::Ok, "");
                }
                break;
            }
            BootstrapPoll::Failed(e) => {
                return Err(anyhow::anyhow!("Bootstrap failed: {e}"));
            }
        }
    }

    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Binary Installation (standalone — not through WizardBackend)
// ─────────────────────────────────────────────────────────────────────────────

fn text_phase_binary_install(non_interactive: bool) -> Result<()> {
    println!("  Phase 2: Binary Installation");
    println!("  {}", "-".repeat(40));
    println!();

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

    let plan = match InstallPlan::prepare() {
        Ok(p) => p,
        Err(e) => {
            print_check("Binary", CheckStatus::Warn, &format!("cannot prepare: {e}"));
            println!();
            return Ok(());
        }
    };

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
// Phase 3: Policy Templates (via WizardBackend)
// ─────────────────────────────────────────────────────────────────────────────

/// Template name for the opt-in open-federation grant. Kept here so the
/// wizard can show it as a separate confirm prompt (rather than
/// burying it in the main Select alongside server-access templates).
const FEDERATION_TEMPLATE: &str = "federation-open";

fn text_phase_templates(
    backend: &mut impl WizardBackend,
    non_interactive: bool,
    enable_federation: bool,
    summary: &mut TextWizardSummary,
) -> Result<()> {
    println!("  Phase 3: Policy Template");
    println!("  {}", "-".repeat(40));
    println!();

    let has_existing = backend.has_existing_policy();

    if has_existing {
        print_check("Policy", CheckStatus::Ok, "existing policy rules found");

        if non_interactive {
            println!("    Keeping existing policy (non-interactive mode).");
            // Federation is composable, so even with an existing policy
            // we honor --enable-federation by layering it on top.
            if enable_federation {
                apply_federation_template(backend, summary);
            }
            println!();
            return Ok(());
        }

        let overwrite = Confirm::new("  Existing policy rules found. Replace with a template?")
            .with_default(false)
            .prompt()
            .unwrap_or(false);

        if !overwrite {
            println!("    Keeping existing policy.");
            prompt_federation_interactive(backend, summary);
            println!();
            return Ok(());
        }
    }

    // Server-access templates only — federation handled separately.
    let templates: Vec<_> = backend
        .templates()
        .into_iter()
        .filter(|t| t.name != FEDERATION_TEMPLATE)
        .collect();

    if non_interactive {
        if let Some(first) = templates.first() {
            backend.apply_template(&first.name);
            print_check("Template", CheckStatus::Ok, &format!("applied '{}'", first.name));
            summary.templates_applied.push(first.name.clone());
        }
        if enable_federation {
            apply_federation_template(backend, summary);
        }
        println!();
        return Ok(());
    }

    let mut options: Vec<String> = templates
        .iter()
        .map(|t| format!("{} — {}", t.name, t.description))
        .collect();
    options.push("None — skip template".to_owned());

    let selection = Select::new("  Select a policy template:", options)
        .prompt()
        .map_err(|e| anyhow::anyhow!("Template selection cancelled: {e}"))?;

    if selection.starts_with("None") {
        println!("    Skipping template.");
    } else {
        let template_name = selection.split(" —").next().unwrap_or("").trim();
        if let Some(template) = templates.iter().find(|t| t.name == template_name) {
            backend.apply_template(&template.name);
            backend.save_policies();
            print_check("Template", CheckStatus::Ok, &format!("applied '{}'", template.name));
            summary.templates_applied.push(template.name.clone());
        }
    }

    prompt_federation_interactive(backend, summary);
    println!();
    Ok(())
}

/// Interactive prompt for the federation-open template. Default is N
/// — opening third-party client federation is opt-in.
fn prompt_federation_interactive(
    backend: &mut impl WizardBackend,
    summary: &mut TextWizardSummary,
) {
    let enable = Confirm::new(
        "  Enable open client federation? \
         Lets any third-party app connect using a published metadata URL (MCP-compatible).",
    )
    .with_default(false)
    .prompt()
    .unwrap_or(false);
    if enable {
        apply_federation_template(backend, summary);
    } else {
        println!("    Federation left disabled — operators can enable later with");
        println!("    `hyprstream quick policy apply-template {FEDERATION_TEMPLATE}`.");
    }
}

fn apply_federation_template(backend: &mut impl WizardBackend, summary: &mut TextWizardSummary) {
    backend.apply_template(FEDERATION_TEMPLATE);
    backend.save_policies();
    print_check(
        "Federation",
        CheckStatus::Ok,
        &format!("applied '{FEDERATION_TEMPLATE}'"),
    );
    summary.templates_applied.push(FEDERATION_TEMPLATE.to_owned());
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Users (via WizardBackend)
// ─────────────────────────────────────────────────────────────────────────────

fn text_phase_users(
    backend: &mut impl WizardBackend,
    non_interactive: bool,
    summary: &mut TextWizardSummary,
) -> Result<()> {
    println!("  Phase 4: Users & Roles");
    println!("  {}", "-".repeat(40));
    println!();

    if non_interactive {
        let local_user = backend.local_username();
        backend.add_user(&local_user, "admin");
        print_check(&local_user, CheckStatus::Ok, "admin");
        summary
            .users_created
            .push((local_user, "admin".to_owned()));
        println!();
        return Ok(());
    }

    let add_users = Confirm::new("  Add a local user?")
        .with_default(true)
        .prompt()
        .unwrap_or(false);

    if !add_users {
        println!("    Skipping user creation.");
        println!();
        return Ok(());
    }

    loop {
        let username = Text::new("  Username:")
            .prompt()
            .map_err(|e| anyhow::anyhow!("Username input cancelled: {e}"))?;

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
            .map_err(|e| anyhow::anyhow!("Role selection cancelled: {e}"))?;

        let role_name = role_selection.split(" —").next().unwrap_or("").trim();

        if role_name == "custom" {
            let resource = Text::new("  Resource pattern (e.g., model:*, registry:*):")
                .with_default("*")
                .prompt()
                .map_err(|e| anyhow::anyhow!("Resource input cancelled: {e}"))?;

            let actions = &["infer", "train", "query", "write", "serve", "manage"];
            let action_options: Vec<String> = actions.iter().map(|a| (*a).to_owned()).collect();

            let selected_actions = inquire::MultiSelect::new("  Actions to allow:", action_options)
                .prompt()
                .map_err(|e| anyhow::anyhow!("Action selection cancelled: {e}"))?;

            backend.add_user_custom(&username, &resource, &selected_actions);

            let actions_str = selected_actions.join(",");
            print_check(
                &username,
                CheckStatus::Ok,
                &format!("custom ({actions_str} on {resource})"),
            );
            summary
                .users_created
                .push((username.clone(), format!("custom({actions_str})")));
        } else {
            backend.add_user(&username, role_name);
            print_check(&username, CheckStatus::Ok, role_name);
            summary
                .users_created
                .push((username.clone(), role_name.to_owned()));
        }

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
// Phase 5: Tokens (via WizardBackend)
// ─────────────────────────────────────────────────────────────────────────────

fn text_phase_tokens(
    backend: &mut impl WizardBackend,
    non_interactive: bool,
    summary: &mut TextWizardSummary,
) -> Result<()> {
    println!("  Phase 5: API Tokens");
    println!("  {}", "-".repeat(40));
    println!();

    if non_interactive {
        let local_user = backend.local_username();
        let result = backend.generate_token(&local_user, "90d");
        print_token_result(&local_user, &result);
        summary.tokens_generated.push((
            local_user,
            token_preview(&result.token),
            result.expires,
        ));
        println!();
        return Ok(());
    }

    if summary.users_created.is_empty() {
        println!("    No users created — skipping token generation.");
        println!();
        return Ok(());
    }

    let expiration_options = vec!["30 days", "90 days (recommended)", "1 year", "never"];

    for (username, _role) in &summary.users_created {
        let create = Confirm::new(&format!("  Generate API token for '{username}'?"))
            .with_default(true)
            .prompt()
            .unwrap_or(false);

        if !create {
            continue;
        }

        let expiry = Select::new("  Token expiration:", expiration_options.clone())
            .prompt()
            .map_err(|e| anyhow::anyhow!("Expiration selection cancelled: {e}"))?;

        let duration_str = match expiry {
            "30 days" => "30d",
            "1 year" => "1y",
            "never" => "never",
            _ => "90d",
        };

        let result = backend.generate_token(username, duration_str);
        print_token_result(username, &result);
        summary.tokens_generated.push((
            username.clone(),
            token_preview(&result.token),
            result.expires.clone(),
        ));
    }

    println!();
    Ok(())
}

fn print_token_result(username: &str, result: &TokenResult) {
    println!();
    println!("    Token for '{username}':");
    println!("    {}", result.token);
    println!();
    println!("    Expires: {}", result.expires);
    let display_len = std::cmp::min(30, result.token.len());
    println!(
        "    Usage:   curl -H \"Authorization: Bearer {}...\" http://localhost:8080/v1/models",
        &result.token[..display_len]
    );
    println!();
    println!("    \x1b[33mSave this token now — it cannot be retrieved again.\x1b[0m");
    println!();
}

fn token_preview(token: &str) -> String {
    if token.len() > 20 {
        format!("{}...", &token[..20])
    } else {
        token.to_owned()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 6: Services (via WizardBackend)
// ─────────────────────────────────────────────────────────────────────────────

fn text_phase_services(
    backend: &mut impl WizardBackend,
    _config_services: &[String],
    non_interactive: bool,
    start_flag: bool,
) -> Result<()> {
    println!("  Phase 6: Services");
    println!("  {}", "-".repeat(40));
    println!();

    let should_start = if start_flag {
        true
    } else if non_interactive {
        false
    } else {
        Confirm::new("  Start services now?")
            .with_default(true)
            .prompt()
            .unwrap_or(false)
    };

    if should_start {
        backend.start_services();

        loop {
            match backend.poll_pending() {
                OpStatus::InProgress => {
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
                OpStatus::Done => {
                    print_check("Services", CheckStatus::Ok, "started");
                    break;
                }
                OpStatus::Failed(e) => {
                    print_check("Services", CheckStatus::Fail, &format!("failed: {e}"));
                    break;
                }
            }
        }
    } else {
        println!("    Services not started.");
        println!();
        println!("    To start later:");
        println!("      hyprstream service start");
    }

    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────────────────────

fn print_summary(summary: &TextWizardSummary) {
    println!("  {}", "=".repeat(40));
    println!("  Setup Complete");
    println!("  {}", "=".repeat(40));
    println!();

    if !summary.templates_applied.is_empty() {
        println!("  Templates: {}", summary.templates_applied.join(", "));
    }

    if !summary.users_created.is_empty() {
        println!("  Users:");
        for (username, role) in &summary.users_created {
            println!("    {username} ({role})");
        }
    }

    if !summary.tokens_generated.is_empty() {
        println!("  Tokens:");
        for (username, preview, expires) in &summary.tokens_generated {
            println!("    {username} — {preview} (expires {expires})");
        }
    }

    if summary.templates_applied.is_empty()
        && summary.users_created.is_empty()
        && summary.tokens_generated.is_empty()
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
