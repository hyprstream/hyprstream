//! Wizard phase definitions and screen state machines.

use waxterm::widgets::{ConfirmDialog, MultiSelectList, SelectList, TextInput};

use super::backend::{EnvironmentInfo, InstallAction, LibtorchVariant, TemplateInfo};

/// Top-level wizard phase.
pub enum WizardPhase {
    Install(InstallScreen),
    Bootstrap(BootstrapScreen),
    PolicyTemplate(PolicyScreen),
    Users(UserScreen),
    Tokens(TokenScreen),
    Services(ServiceScreen),
    Summary(SummaryScreen),
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Install / Update
// ─────────────────────────────────────────────────────────────────────────────

pub enum InstallScreen {
    /// Auto-detecting environment.
    Detecting,
    /// Show findings and recommendation.
    ShowFindings {
        env: EnvironmentInfo,
        action: InstallAction,
    },
    /// User chose to pick variant manually.
    SelectVariant(SelectList<String>),
    /// Download + install in progress.
    Installing {
        variant: LibtorchVariant,
        progress_pct: u8,
        status_msg: String,
    },
    /// Completed.
    Done(String),
    /// User skipped.
    Skipped,
    /// Failed with option to retry.
    Failed(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

pub struct BootstrapScreen {
    pub started: bool,
    pub completed_steps: Vec<String>,
    pub current_step: String,
    pub done: bool,
    pub failed: Option<String>,
}

impl Default for BootstrapScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl BootstrapScreen {
    pub fn new() -> Self {
        Self {
            started: false,
            completed_steps: Vec::new(),
            current_step: String::new(),
            done: false,
            failed: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Policy Template
// ─────────────────────────────────────────────────────────────────────────────

pub enum PolicyScreen {
    /// Ask if user wants to replace existing policy.
    ConfirmReplace(ConfirmDialog),
    /// Select from template list.
    SelectTemplate(SelectList<String>),
    /// Template applied, showing result.
    Applied(String),
    /// User skipped or kept existing.
    Skipped,
}

impl PolicyScreen {
    /// Build the template selection list from backend data.
    pub fn select_from(templates: &[TemplateInfo]) -> Self {
        let mut options: Vec<String> = templates
            .iter()
            .map(|t| format!("{} — {}", t.name, t.description))
            .collect();
        options.push("None — skip template".to_string());
        PolicyScreen::SelectTemplate(SelectList::new("Select a policy template:", options))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Users & Roles
// ─────────────────────────────────────────────────────────────────────────────

pub enum UserScreen {
    /// Ask if user wants to add a user.
    AskAdd(ConfirmDialog),
    /// Enter username.
    EnterName(TextInput),
    /// Select role.
    SelectRole(SelectList<String>),
    /// For custom role: enter resource pattern.
    CustomResource(TextInput),
    /// For custom role: select actions.
    CustomActions(MultiSelectList<String>),
    /// Ask to add another user.
    AskAnother(ConfirmDialog),
    /// Done with user creation.
    Done,
}

/// Record of a created user.
#[derive(Debug, Clone)]
pub struct UserRecord {
    pub username: String,
    pub role: String,
}

/// Record of a generated token.
#[derive(Debug, Clone)]
pub struct TokenRecord {
    pub username: String,
    pub token: String,
    pub expires: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5: Tokens
// ─────────────────────────────────────────────────────────────────────────────

pub enum TokenScreen {
    /// Ask if user wants to generate a token.
    AskGenerate(ConfirmDialog, String),
    /// Select expiration.
    SelectExpiry(SelectList<String>, String),
    /// Show generated token.
    ShowToken(TokenRecord),
    /// Ask for next user's token.
    NextUser,
    /// Done with token generation.
    Done,
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 6: Services
// ─────────────────────────────────────────────────────────────────────────────

pub enum ServiceScreen {
    /// Ask if user wants to start services.
    AskStart(ConfirmDialog),
    /// Services starting.
    Starting,
    /// Done.
    Done(bool),
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────────────────────

pub struct SummaryScreen {
    pub install_result: Option<String>,
    pub templates_applied: Vec<String>,
    pub users_created: Vec<UserRecord>,
    pub tokens_generated: Vec<TokenRecord>,
    pub services_started: bool,
}
