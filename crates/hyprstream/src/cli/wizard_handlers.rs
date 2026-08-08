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
    /// Grade line for the deployment root, when a ceremony ran.
    deployment_trust: Option<String>,
}

impl TextWizardSummary {
    fn new() -> Self {
        Self {
            templates_applied: Vec::new(),
            users_created: Vec::new(),
            tokens_generated: Vec::new(),
            deployment_trust: None,
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

/// Opt-in deployment-trust ceremony settings.
///
/// Deployment trust is a separate layer from node bootstrap: none of this runs
/// unless `enabled` is set, and the wizard's node-local behaviour is unchanged
/// when it is not.
#[derive(Clone, Debug, Default)]
pub struct DeploymentTrustOptions {
    /// Run the ceremony after node bootstrap.
    pub enabled: bool,
    /// Absolute ceremony working directory. Defaults under the models dir.
    pub dir: Option<std::path::PathBuf>,
    /// Mint a software root even if a token is attached.
    pub force_software: bool,
    /// Token serial to use when several are attached.
    pub serial: Option<String>,
    /// PIV slot for the Ed25519 leg on firmware that supports it.
    pub piv_slot: Option<String>,
    /// Accept an unencrypted break-glass identity. Throwaway roots only.
    pub allow_plaintext_break_glass: bool,
}

/// Everything `hyprstream wizard` was invoked with.
#[derive(Clone, Debug, Default)]
pub struct WizardOptions {
    /// Accept defaults without prompting.
    pub non_interactive: bool,
    /// Start services once setup finishes.
    pub start_services: bool,
    /// Run only phase 1 (node bootstrap).
    pub bootstrap_only: bool,
    /// Apply the federation-open policy template.
    pub enable_federation: bool,
    /// Role assigned to the local user under `--non-interactive`.
    pub initial_user_role: String,
    /// Deployment-trust ceremony settings.
    pub deployment_trust: DeploymentTrustOptions,
}

impl WizardOptions {
    /// Defaults for the first-run path: interactive, no federation, admin user.
    #[must_use]
    pub fn first_run() -> Self {
        Self {
            initial_user_role: "admin".to_owned(),
            ..Self::default()
        }
    }
}

/// Handle `hyprstream wizard` — interactive setup wizard.
///
/// When `bootstrap_only` is true, only phase 1 (trust-root setup) runs.
/// `initial_user_role` overrides the default "admin" role assigned to the
/// local user under `--non-interactive`; use "operator" or "viewer" for
/// least-privilege test setups (#184).
pub async fn handle_wizard(
    models_dir: &Path,
    config_services: &[String],
    options: WizardOptions,
) -> Result<()> {
    // Install systemd units before entering spawn_blocking (async operation).
    if !options.bootstrap_only && hyprstream_rpc::has_systemd() {
        handle_service_install(models_dir, config_services, None, false, false, hyprstream_service::ServiceTarget::User, false).await?;
    }

    let rt = tokio::runtime::Handle::current();
    let models_dir = models_dir.to_path_buf();
    let config_services = config_services.to_vec();

    tokio::task::spawn_blocking(move || {
        let mut backend =
            crate::cli::bootstrap_manager::BootstrapManager::new(rt, models_dir.clone(), config_services.clone());
        run_text_wizard(&mut backend, &options, &models_dir, &config_services)
    })
    .await??;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Text-mode wizard driver (delegates to WizardBackend)
// ─────────────────────────────────────────────────────────────────────────────

fn run_text_wizard(
    backend: &mut impl WizardBackend,
    options: &WizardOptions,
    models_dir: &Path,
    config_services: &[String],
) -> Result<()> {
    let non_interactive = options.non_interactive;

    println!();
    println!("  Hyprstream Setup Wizard");
    println!("  {}", "=".repeat(40));
    println!();

    let mut summary = TextWizardSummary::new();

    // Phase 1: Environment bootstrap
    text_phase_bootstrap(backend)?;

    if options.bootstrap_only {
        return Ok(());
    }

    // Phase 2: Binary installation (standalone — not a WizardBackend concern)
    text_phase_binary_install(non_interactive)?;

    // Phase 3: Policy template selection
    text_phase_templates(backend, non_interactive, options.enable_federation, &mut summary)?;

    // Phase 4: User/role creation
    text_phase_users(backend, non_interactive, &options.initial_user_role, &mut summary)?;

    // Phase 5: Token generation
    text_phase_tokens(backend, non_interactive, &mut summary)?;

    // Phase 6: Deployment trust (opt-in; separate from node-local trust)
    text_phase_deployment_trust(options, models_dir, &mut summary)?;

    // Phase 7: Service startup
    text_phase_services(backend, config_services, non_interactive, options.start_services)?;

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
        "  Enable open federation? \
         Accepts third-party apps and remote peer servers from any origin \
         (atproto-style; MCP/peer compatible).",
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
    initial_user_role: &str,
    summary: &mut TextWizardSummary,
) -> Result<()> {
    println!("  Phase 4: Users & Roles");
    println!("  {}", "-".repeat(40));
    println!();

    if non_interactive {
        let local_user = backend.local_username();
        backend.add_user(&local_user, initial_user_role);
        print_check(&local_user, CheckStatus::Ok, initial_user_role);
        summary
            .users_created
            .push((local_user, initial_user_role.to_owned()));
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
// Phase 6: Deployment trust (opt-in ceremony)
// ─────────────────────────────────────────────────────────────────────────────

/// Run the deployment-trust ceremony, if the operator asked for it.
///
/// Node bootstrap sets up node-local trust and stops there; this phase is the
/// separate, deliberately opt-in deployment layer. When it is not requested the
/// wizard behaves exactly as it did before, including under `--non-interactive`,
/// where the phase never reaches a prompt.
fn text_phase_deployment_trust(
    options: &WizardOptions,
    models_dir: &Path,
    summary: &mut TextWizardSummary,
) -> Result<()> {
    use crate::cli::trust_ceremony as ceremony;
    use crate::cli::trust_ceremony::{AgeYubikeyPlugin as _, TokenDetector as _};

    let settings = &options.deployment_trust;
    let request = ceremony::CeremonyRequest {
        enabled: settings.enabled,
        interactive: !options.non_interactive,
        force_software: settings.force_software,
        serial: settings.serial.clone(),
        piv_slot: settings
            .piv_slot
            .clone()
            .unwrap_or_else(|| ceremony::DEFAULT_PIV_SLOT.to_owned()),
    };

    if !request.enabled {
        return Ok(());
    }

    println!("  Phase 6: Deployment Trust");
    println!("  {}", "-".repeat(40));
    println!();

    println!("    Looking for a hardware token...");
    let detection = ceremony::SystemTokenDetector.detect();
    let mut plan = ceremony::plan_ceremony(&request, &detection);

    if let ceremony::CeremonyPlan::ChooseToken { tokens, rationale } = &plan {
        println!("    {rationale}.");
        let labels: Vec<String> = tokens.iter().map(ceremony::DetectedToken::label).collect();
        let chosen = Select::new("  Which token should hold the deployment root?", labels.clone())
            .prompt()
            .map_err(|e| anyhow::anyhow!("no token was chosen: {e}"))?;
        let index = labels
            .iter()
            .position(|label| *label == chosen)
            .ok_or_else(|| anyhow::anyhow!("token selection did not match an attached token"))?;
        let token = tokens
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("token selection did not match an attached token"))?
            .clone();
        plan = ceremony::select_mode_for_token(&request, &token);
    }

    let (mode, rationale) = match plan {
        ceremony::CeremonyPlan::Skipped { reason } => {
            println!("    {reason}.");
            println!();
            return Ok(());
        }
        ceremony::CeremonyPlan::Blocked { reason, remedy } => {
            print_check("Deployment trust", CheckStatus::Fail, &reason);
            println!("    {remedy}.");
            return Err(anyhow::anyhow!(
                "deployment trust ceremony cannot run: {reason} — {remedy}"
            ));
        }
        ceremony::CeremonyPlan::ChooseToken { .. } => {
            return Err(anyhow::anyhow!(
                "several tokens are attached and none was chosen; pass \
                 --deployment-trust-serial <SERIAL>"
            ));
        }
        ceremony::CeremonyPlan::Proceed { mode, rationale } => (mode, rationale),
    };

    println!("    {rationale}.");
    if mode.is_dev_grade() {
        println!();
        println!("    !! {}", ceremony::DEV_GRADE_LABEL);
        println!("    !! Anyone who can read the recipient files holds the deployment root.");
        println!("    !! Upgrade to hardware later with `hyprstream trust rotate-authority`.");
        println!();
    }

    let missing = ceremony::missing_tools(ceremony::required_tools(&mode), ceremony::tool_on_path);
    if !missing.is_empty() {
        return Err(anyhow::anyhow!(
            "the {} ceremony needs {} on PATH; install {} and re-run",
            mode.mode_id(),
            missing.join(", "),
            missing.join(" and ")
        ));
    }

    let dir = match settings.dir.clone() {
        Some(dir) => dir,
        None => models_dir.join("deployment-ceremony"),
    };
    let dir = if dir.is_absolute() {
        dir
    } else {
        std::env::current_dir()
            .map_err(|e| anyhow::anyhow!("resolve the ceremony directory: {e}"))?
            .join(dir)
    };
    let paths = ceremony::CeremonyPaths::new(dir)?;
    ceremony::create_ceremony_dir(paths.dir())?;
    println!("    Ceremony working directory: {}", paths.dir().display());

    // Primary recipient: the token's own age identity, or a software key.
    let (primary_recipient, unlock_identity) = match &mode {
        ceremony::CeremonyMode::HardwarePiv { token, .. }
        | ceremony::CeremonyMode::HardwareAgeRecipient { token } => {
            println!();
            println!("    Generating the token's age identity (PIN + touch required for every");
            println!("    use). Follow the prompts on your terminal and the token.");
            let plugin = ceremony::SystemAgeYubikeyPlugin;
            let recipient = plugin
                .generate_identity(&format!("hyprstream deployment root ({})", token.serial))
                .map_err(|e| anyhow::anyhow!("generate the token's age identity: {e}"))?;
            plugin
                .export_identity_file(&paths.yubikey_identity())
                .map_err(|e| anyhow::anyhow!("export the token's age identity: {e}"))?;
            (
                recipient,
                ceremony::UnlockIdentity::Yubikey(paths.yubikey_identity()),
            )
        }
        ceremony::CeremonyMode::SoftwareDevGrade => {
            let recipient = ceremony::generate_software_identity(&paths.primary_identity())?;
            (
                recipient,
                ceremony::UnlockIdentity::Age(paths.primary_identity()),
            )
        }
    };

    let break_glass = prompt_break_glass(&mode, &primary_recipient, &paths, options)?;
    println!("    Break-glass: {}", break_glass.kind_id());

    // The online signer is an autonomous key: no token, no root.
    let signer_recipient = ceremony::generate_software_identity(&paths.online_signer_identity())?;

    // The credential binds the registry service key node bootstrap just made.
    let credentials_dir = crate::auth::identity_store::credentials_dir()?;
    let pubkeys = crate::auth::identity_store::load_bootstrap_pubkeys(&credentials_dir)?;
    let registry_key = pubkeys.get("registry").ok_or_else(|| {
        anyhow::anyhow!(
            "no registry service key in {}; run node bootstrap before the ceremony",
            credentials_dir.display()
        )
    })?;
    ceremony::write_registry_public_key(&paths.registry_public_key(), registry_key.as_bytes())?;

    let inputs = ceremony::CeremonyInputs {
        mode: mode.clone(),
        primary_recipient,
        break_glass: break_glass.clone(),
        unlock_identity,
        signer_recipient,
        paths: paths.clone(),
    };

    println!();
    for (step, command) in ceremony::ceremony_commands(&inputs)?.into_iter().enumerate() {
        println!("    Ceremony step {}/4...", step + 1);
        crate::cli::trust::handle_trust_command(command)?;
    }

    let record = ceremony::mode_record(&mode, &break_glass, &rationale);
    std::fs::write(
        paths.mode_record(),
        serde_json::to_vec_pretty(&record)?,
    )?;

    print_check("Deployment trust", CheckStatus::Ok, mode.grade_label());
    println!();
    println!("    The public artifacts are NOT installed yet. Copy them to a deployed host:");
    println!("      {}", paths.public_ca().display());
    println!("      {}", paths.authority_log().display());
    println!("      {}", paths.authority_checkpoint().display());
    println!("    Keep {} offline — it never belongs on a deployed host.", paths.authority_key().display());
    println!();

    summary.deployment_trust = Some(format!("{} — {}", mode.mode_id(), mode.grade_label()));
    Ok(())
}

/// Walk the operator through a break-glass recipient the tooling can trust.
///
/// The trust CLI can only see that two recipients differ; whether the second is
/// protected is decided here, and an unprotected one is refused by default.
fn prompt_break_glass(
    mode: &crate::cli::trust_ceremony::CeremonyMode,
    primary_recipient: &str,
    paths: &crate::cli::trust_ceremony::CeremonyPaths,
    options: &WizardOptions,
) -> Result<crate::cli::trust_ceremony::BreakGlass> {
    use crate::cli::trust_ceremony as ceremony;

    const SECOND_TOKEN: &str = "A second hardware token (best — same protection as the primary)";
    const PASSPHRASE: &str = "A passphrase-encrypted identity file (age-keygen | age -p)";
    const PLAINTEXT: &str = "An unencrypted identity file (throwaway roots only)";

    let policy = if options.deployment_trust.allow_plaintext_break_glass {
        ceremony::PlaintextPolicy::AllowedByOverride
    } else if mode.is_dev_grade() {
        ceremony::PlaintextPolicy::AllowedForDevGradeRoot
    } else {
        ceremony::PlaintextPolicy::Refuse
    };

    // A scripted run cannot answer a prompt or type a passphrase. It only gets
    // here for a dev-grade root, where an unencrypted backup adds no exposure.
    let break_glass = if options.non_interactive {
        let recipient = ceremony::generate_software_identity(&paths.break_glass_identity())?;
        ceremony::BreakGlass::PlaintextIdentity {
            identity_file: paths.break_glass_identity(),
            recipient,
        }
    } else {
        println!();
        println!("    A deployment root needs a second recipient, or losing the primary loses");
        println!("    the root. The root is exactly as strong as its weakest recipient.");
        let choice = Select::new(
            "  How should the break-glass recipient be protected?",
            vec![SECOND_TOKEN, PASSPHRASE, PLAINTEXT],
        )
        .with_starting_cursor(if mode.is_dev_grade() { 1 } else { 0 })
        .prompt()
        .map_err(|e| anyhow::anyhow!("no break-glass recipient was chosen: {e}"))?;

        match choice {
            SECOND_TOKEN => {
                println!("    Prepare the second token elsewhere with:");
                println!(
                    "      age-plugin-yubikey --generate --name \"hyprstream break-glass\" \\"
                );
                println!("        --pin-policy always --touch-policy always");
                let recipient = Text::new("  Paste its age1yubikey1… recipient:")
                    .prompt()
                    .map_err(|e| anyhow::anyhow!("no break-glass recipient was entered: {e}"))?;
                ceremony::BreakGlass::SecondToken {
                    recipient: recipient.trim().to_owned(),
                }
            }
            PASSPHRASE => {
                let destination = Text::new("  Where should the encrypted backup identity go?")
                    .with_default(&paths.break_glass_identity().display().to_string())
                    .prompt()
                    .map_err(|e| anyhow::anyhow!("no backup destination was entered: {e}"))?;
                let destination = std::path::PathBuf::from(destination.trim());
                println!("    Choose a passphrase and store it apart from the file itself.");
                let recipient =
                    ceremony::generate_passphrase_encrypted_identity(&destination)?;
                match ceremony::classify_identity_file(
                    &std::fs::read_to_string(&destination).unwrap_or_default(),
                ) {
                    ceremony::IdentityFileKind::PassphraseEncrypted => {}
                    ceremony::IdentityFileKind::PlaintextAgeIdentity => {
                        return Err(anyhow::anyhow!(
                            "{} was written unencrypted; delete it and retry",
                            destination.display()
                        ));
                    }
                    ceremony::IdentityFileKind::Unconfirmed => {
                        println!(
                            "    Note: {} is encrypted, but its passphrase stanza could not be \
                             confirmed from here.",
                            destination.display()
                        );
                    }
                }
                ceremony::BreakGlass::PassphraseEncrypted {
                    identity_file: destination,
                    recipient,
                }
            }
            _ => {
                let recipient =
                    ceremony::generate_software_identity(&paths.break_glass_identity())?;
                ceremony::BreakGlass::PlaintextIdentity {
                    identity_file: paths.break_glass_identity(),
                    recipient,
                }
            }
        }
    };

    if let Err(reason) = ceremony::validate_break_glass(&break_glass, primary_recipient, policy) {
        return Err(anyhow::anyhow!("{reason}"));
    }
    Ok(break_glass)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 7: Services
// ─────────────────────────────────────────────────────────────────────────────

fn text_phase_services(
    backend: &mut impl WizardBackend,
    _config_services: &[String],
    non_interactive: bool,
    start_flag: bool,
) -> Result<()> {
    println!("  Phase 7: Services");
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

    if let Some(deployment_trust) = &summary.deployment_trust {
        println!("  Deployment trust: {deployment_trust}");
    }

    if summary.templates_applied.is_empty()
        && summary.users_created.is_empty()
        && summary.tokens_generated.is_empty()
        && summary.deployment_trust.is_none()
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
