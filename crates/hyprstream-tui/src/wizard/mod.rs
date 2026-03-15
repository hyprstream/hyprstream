//! Setup wizard TUI application.
//!
//! Implements `waxterm::TerminalApp` to drive the wizard through 6 phases:
//! Install, Bootstrap, Policy Templates, Users & Roles, API Tokens, Services.

pub mod backend;
pub mod phases;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use waxterm::input::KeyPress;
use waxterm::widgets::{ConfirmDialog, MultiSelectList, SelectList, TextInput, WidgetResult};

use backend::{
    BootstrapPoll, InstallAction, InstallPoll, LibtorchVariant, OpStatus, RunMode, WizardBackend,
};
use phases::*;

/// Wizard command type — just wraps KeyPress.
pub enum WizardCommand {
    Key(KeyPress),
}

impl From<KeyPress> for WizardCommand {
    fn from(k: KeyPress) -> Self {
        WizardCommand::Key(k)
    }
}

/// Predefined role definitions.
const ROLE_OPTIONS: &[(&str, &str)] = &[
    ("admin", "Full access to everything"),
    ("operator", "Infer + query + load/unload models"),
    ("viewer", "Read-only queries"),
    ("trainer", "Inference + training"),
    ("custom", "Define custom permissions"),
];

/// Action options for custom roles.
const ACTION_OPTIONS: &[&str] = &["infer", "train", "query", "write", "serve", "manage"];

/// Expiration options for tokens.
const EXPIRY_OPTIONS: &[&str] = &[
    "90 days (recommended)",
    "30 days",
    "1 year",
    "never",
];

/// The wizard TUI application.
pub struct WizardApp<B: WizardBackend> {
    backend: B,
    phase: WizardPhase,
    quit: bool,
    // Accumulated state across phases
    install_summary: Option<String>,
    templates_applied: Vec<String>,
    users_created: Vec<UserRecord>,
    tokens_generated: Vec<TokenRecord>,
    services_started: bool,
    // Transient state for user creation flow
    pending_username: String,
    pending_role: String,
    pending_resource: String,
    // Token generation queue
    token_queue: Vec<String>,
    token_queue_idx: usize,
    // Install phase transient state
    install_variant: Option<LibtorchVariant>,
}

impl<B: WizardBackend> WizardApp<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            phase: WizardPhase::Install(InstallScreen::Detecting),
            quit: false,
            install_summary: None,
            templates_applied: Vec::new(),
            users_created: Vec::new(),
            tokens_generated: Vec::new(),
            services_started: false,
            pending_username: String::new(),
            pending_role: String::new(),
            pending_resource: String::new(),
            token_queue: Vec::new(),
            token_queue_idx: 0,
            install_variant: None,
        }
    }

    fn phase_number(&self) -> u8 {
        match &self.phase {
            WizardPhase::Install(_) => 1,
            WizardPhase::Bootstrap(_) => 2,
            WizardPhase::PolicyTemplate(_) => 3,
            WizardPhase::Users(_) => 4,
            WizardPhase::Tokens(_) => 5,
            WizardPhase::Services(_) => 6,
            WizardPhase::Summary(_) => 6,
        }
    }

    fn phase_name(&self) -> &str {
        match &self.phase {
            WizardPhase::Install(_) => "Install / Update",
            WizardPhase::Bootstrap(_) => "Environment Bootstrap",
            WizardPhase::PolicyTemplate(_) => "Policy Template",
            WizardPhase::Users(_) => "Users & Roles",
            WizardPhase::Tokens(_) => "API Tokens",
            WizardPhase::Services(_) => "Services",
            WizardPhase::Summary(_) => "Setup Complete",
        }
    }

    fn advance_to_bootstrap(&mut self) {
        self.phase = WizardPhase::Bootstrap(BootstrapScreen::new());
    }

    fn advance_to_policy(&mut self) {
        let templates = self.backend.templates();
        if self.backend.has_existing_policy() {
            self.phase = WizardPhase::PolicyTemplate(PolicyScreen::ConfirmReplace(
                ConfirmDialog::new("Existing policy found. Replace with a template?")
                    .with_default(false),
            ));
        } else {
            self.phase =
                WizardPhase::PolicyTemplate(PolicyScreen::select_from(&templates));
        }
    }

    fn advance_to_users(&mut self) {
        self.phase = WizardPhase::Users(UserScreen::AskAdd(
            ConfirmDialog::new("Add a user?"),
        ));
    }

    fn advance_to_tokens(&mut self) {
        // Build queue of users to generate tokens for
        self.token_queue = self
            .users_created
            .iter()
            .map(|u| u.username.clone())
            .collect();

        if self.token_queue.is_empty() {
            self.token_queue.push(self.backend.local_username());
        }

        self.token_queue_idx = 0;
        self.advance_token_for_next_user();
    }

    fn advance_token_for_next_user(&mut self) {
        if self.token_queue_idx < self.token_queue.len() {
            let username = self.token_queue[self.token_queue_idx].clone();
            self.phase = WizardPhase::Tokens(TokenScreen::AskGenerate(
                ConfirmDialog::new(format!("Generate API token for '{username}'?")),
                username,
            ));
        } else {
            self.advance_to_services();
        }
    }

    fn advance_to_services(&mut self) {
        self.phase = WizardPhase::Services(ServiceScreen::AskStart(
            ConfirmDialog::new("Start services now?").with_default(false),
        ));
    }

    fn advance_to_summary(&mut self) {
        self.phase = WizardPhase::Summary(SummaryScreen {
            install_result: self.install_summary.clone(),
            templates_applied: self.templates_applied.clone(),
            users_created: self.users_created.clone(),
            tokens_generated: self.tokens_generated.clone(),
            services_started: self.services_started,
        });
    }

    fn make_variant_select() -> SelectList<String> {
        let options: Vec<String> = LibtorchVariant::all()
            .iter()
            .map(|v| format!("{} ({})", v.label(), v.id()))
            .collect();
        SelectList::new("Select variant:", options)
    }

    fn variant_from_selection(selection: &str) -> LibtorchVariant {
        if selection.contains("cuda128") {
            LibtorchVariant::Cuda128
        } else if selection.contains("cuda130") {
            LibtorchVariant::Cuda130
        } else if selection.contains("rocm71") {
            LibtorchVariant::Rocm71
        } else {
            LibtorchVariant::Cpu
        }
    }

    fn make_role_select() -> SelectList<String> {
        let options: Vec<String> = ROLE_OPTIONS
            .iter()
            .map(|(name, desc)| format!("{name} — {desc}"))
            .collect();
        SelectList::new("Role:", options)
    }

    fn make_expiry_select() -> SelectList<String> {
        let options: Vec<String> = EXPIRY_OPTIONS.iter().map(|s| (*s).to_string()).collect();
        SelectList::new("Token expiration:", options)
    }

    fn make_action_multiselect() -> MultiSelectList<String> {
        let options: Vec<String> = ACTION_OPTIONS.iter().map(|s| (*s).to_string()).collect();
        MultiSelectList::new("Actions to allow:", options)
    }

    fn expiry_to_duration(selection: &str) -> &str {
        if selection.starts_with("30") {
            "30d"
        } else if selection.starts_with("1 year") {
            "1y"
        } else if selection.starts_with("never") {
            "never"
        } else {
            "90d"
        }
    }
}

impl<B: WizardBackend> waxterm::TerminalApp for WizardApp<B> {
    type Command = WizardCommand;

    fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let chunks = Layout::vertical([
            Constraint::Length(3),  // header
            Constraint::Length(1),  // phase indicator
            Constraint::Length(1),  // separator
            Constraint::Min(10),   // content
            Constraint::Length(2),  // footer
        ])
        .split(area);

        render_header(frame, chunks[0]);
        render_phase_indicator(frame, chunks[1], self.phase_number(), self.phase_name());
        render_separator(frame, chunks[2]);
        self.render_content(frame, chunks[3]);
        render_footer(frame, chunks[4], &self.phase);
    }

    fn handle_input(&mut self, cmd: WizardCommand) -> bool {
        let WizardCommand::Key(key) = cmd;

        // Global quit on 'q' at summary, or Ctrl-C (Char(3))
        if let KeyPress::Char(b'q') = key {
            if matches!(&self.phase, WizardPhase::Summary(_)) {
                self.quit = true;
                return true;
            }
        }
        if let KeyPress::Char(3) = key {
            self.quit = true;
            return true;
        }

        // Dispatch based on current phase. We check discriminant without borrowing
        // to avoid double-mutable-borrow in the handler methods (which use mem::replace).
        let phase_num = self.phase_number();
        match phase_num {
            1 => self.handle_install_input(&key),
            2 => {
                // Bootstrap is tick-driven, no input needed
                false
            }
            3 => self.handle_policy_input(&key),
            4 => self.handle_users_input(&key),
            5 => self.handle_tokens_input(&key),
            6 => {
                if matches!(&self.phase, WizardPhase::Summary(_)) {
                    if key == KeyPress::Enter {
                        self.quit = true;
                    }
                    true
                } else {
                    self.handle_services_input(&key)
                }
            }
            _ => false,
        }
    }

    fn tick(&mut self, _delta_ms: u64) -> bool {
        match &mut self.phase {
            WizardPhase::Install(screen) => match screen {
                InstallScreen::Detecting => {
                    let env = self.backend.detect_environment();
                    let action = self.backend.recommend_action(&env);
                    // Auto-skip for AppImage and Development modes
                    match &action {
                        InstallAction::Skip { .. } => {
                            self.phase = WizardPhase::Install(InstallScreen::Skipped);
                            self.advance_to_bootstrap();
                            return true;
                        }
                        InstallAction::AlreadyCurrent => {
                            self.install_summary =
                                Some(format!("Already running optimal variant ({})", env.current_variant));
                            self.phase = WizardPhase::Install(InstallScreen::Done(
                                format!("Already running {} — no changes needed", env.current_variant),
                            ));
                            // Don't auto-advance; let user press Enter
                        }
                        InstallAction::UpgradeVariant(_) => {}
                    }
                    if !matches!(self.phase, WizardPhase::Install(InstallScreen::Done(_))) {
                        self.phase = WizardPhase::Install(InstallScreen::ShowFindings {
                            env,
                            action,
                        });
                    }
                    true
                }
                InstallScreen::Installing {
                    variant,
                    progress_pct,
                    status_msg,
                } => {
                    match self.backend.poll_install() {
                        InstallPoll::Downloading { item, pct } => {
                            *progress_pct = pct;
                            *status_msg = format!("Downloading {item}...");
                            true
                        }
                        InstallPoll::Extracting { item } => {
                            *status_msg = format!("Extracting {item}...");
                            true
                        }
                        InstallPoll::Configuring { step } => {
                            *status_msg = step;
                            *progress_pct = 100;
                            true
                        }
                        InstallPoll::Done { summary } => {
                            self.install_summary = Some(summary.clone());
                            let v = variant.clone();
                            self.phase = WizardPhase::Install(InstallScreen::Done(
                                format!("{} variant installed — {summary}", v.label()),
                            ));
                            true
                        }
                        InstallPoll::Failed(msg) => {
                            self.phase = WizardPhase::Install(InstallScreen::Failed(msg));
                            true
                        }
                        InstallPoll::Detecting => true,
                    }
                }
                _ => false,
            },
            WizardPhase::Bootstrap(screen) => {
                if !screen.started {
                    screen.started = true;
                    self.backend.start_bootstrap();
                }
                if screen.done {
                    return false;
                }
                match self.backend.poll_bootstrap() {
                    BootstrapPoll::InProgress(msg) => {
                        screen.current_step = msg;
                        true
                    }
                    BootstrapPoll::Done(steps) => {
                        screen.completed_steps = steps;
                        screen.current_step.clear();
                        screen.done = true;
                        // Auto-advance after bootstrap
                        self.advance_to_policy();
                        true
                    }
                    BootstrapPoll::Failed(msg) => {
                        screen.failed = Some(msg);
                        screen.done = true;
                        true
                    }
                }
            }
            WizardPhase::Services(ServiceScreen::Starting) => {
                match self.backend.poll_pending() {
                    OpStatus::InProgress => true,
                    OpStatus::Done => {
                        self.services_started = true;
                        self.phase = WizardPhase::Services(ServiceScreen::Done(true));
                        // Auto-advance to summary
                        self.advance_to_summary();
                        true
                    }
                    OpStatus::Failed(_msg) => {
                        self.phase = WizardPhase::Services(ServiceScreen::Done(false));
                        self.advance_to_summary();
                        true
                    }
                }
            }
            _ => false,
        }
    }

    fn should_quit(&self) -> bool {
        self.quit
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Input handling (split out to work around borrow checker)
// ─────────────────────────────────────────────────────────────────────────────

// These methods take &mut self but need separate borrowing for phase mutation.
// We use a pattern of taking the phase enum, matching, then reassigning.

impl<B: WizardBackend> WizardApp<B> {
    fn handle_install_input(&mut self, key: &KeyPress) -> bool {
        let phase = std::mem::replace(
            &mut self.phase,
            WizardPhase::Install(InstallScreen::Skipped),
        );

        let WizardPhase::Install(screen) = phase else {
            return false;
        };

        match screen {
            InstallScreen::Detecting => {
                // Tick-driven, no input needed
                self.phase = WizardPhase::Install(InstallScreen::Detecting);
                false
            }
            InstallScreen::ShowFindings { env, action } => {
                match key {
                    KeyPress::Enter => {
                        // Accept recommendation
                        if let InstallAction::UpgradeVariant(ref variant) = action {
                            let v = variant.clone();
                            self.install_variant = Some(v.clone());
                            self.backend.start_install(&v);
                            self.phase = WizardPhase::Install(InstallScreen::Installing {
                                variant: v,
                                progress_pct: 0,
                                status_msg: "Starting download...".to_string(),
                            });
                        } else {
                            // Skip/AlreadyCurrent — advance
                            self.advance_to_bootstrap();
                        }
                    }
                    KeyPress::Char(b's') => {
                        // Skip
                        self.phase = WizardPhase::Install(InstallScreen::Skipped);
                        self.advance_to_bootstrap();
                    }
                    KeyPress::Char(b'v') => {
                        // Choose variant manually
                        self.phase = WizardPhase::Install(InstallScreen::SelectVariant(
                            Self::make_variant_select(),
                        ));
                    }
                    _ => {
                        // Put it back
                        self.phase = WizardPhase::Install(InstallScreen::ShowFindings {
                            env,
                            action,
                        });
                        return false;
                    }
                }
                true
            }
            InstallScreen::SelectVariant(mut list) => {
                match list.handle_key(key) {
                    WidgetResult::Confirmed(selection) => {
                        let variant = Self::variant_from_selection(&selection);
                        self.install_variant = Some(variant.clone());
                        self.backend.start_install(&variant);
                        self.phase = WizardPhase::Install(InstallScreen::Installing {
                            variant,
                            progress_pct: 0,
                            status_msg: "Starting download...".to_string(),
                        });
                    }
                    WidgetResult::Cancelled => {
                        self.phase = WizardPhase::Install(InstallScreen::Skipped);
                        self.advance_to_bootstrap();
                    }
                    WidgetResult::Pending => {
                        self.phase =
                            WizardPhase::Install(InstallScreen::SelectVariant(list));
                    }
                }
                true
            }
            InstallScreen::Installing {
                variant,
                progress_pct,
                status_msg,
            } => {
                // Tick-driven, no user input
                self.phase = WizardPhase::Install(InstallScreen::Installing {
                    variant,
                    progress_pct,
                    status_msg,
                });
                false
            }
            InstallScreen::Done(_msg) => {
                if *key == KeyPress::Enter {
                    self.advance_to_bootstrap();
                    true
                } else {
                    self.phase = WizardPhase::Install(InstallScreen::Done(_msg));
                    false
                }
            }
            InstallScreen::Skipped => {
                self.advance_to_bootstrap();
                true
            }
            InstallScreen::Failed(msg) => {
                match key {
                    KeyPress::Char(b'r') => {
                        // Retry
                        if let Some(ref variant) = self.install_variant {
                            let v = variant.clone();
                            self.backend.start_install(&v);
                            self.phase = WizardPhase::Install(InstallScreen::Installing {
                                variant: v,
                                progress_pct: 0,
                                status_msg: "Retrying download...".to_string(),
                            });
                        } else {
                            self.phase = WizardPhase::Install(InstallScreen::Detecting);
                        }
                    }
                    KeyPress::Char(b's') => {
                        self.phase = WizardPhase::Install(InstallScreen::Skipped);
                        self.advance_to_bootstrap();
                    }
                    KeyPress::Char(b'q') => {
                        self.quit = true;
                    }
                    _ => {
                        self.phase = WizardPhase::Install(InstallScreen::Failed(msg));
                        return false;
                    }
                }
                true
            }
        }
    }

    fn handle_policy_input(&mut self, key: &KeyPress) -> bool {
        // Take the phase to work around the borrow
        let phase = std::mem::replace(
            &mut self.phase,
            WizardPhase::PolicyTemplate(PolicyScreen::Skipped),
        );

        let WizardPhase::PolicyTemplate(screen) = phase else {
            return false;
        };

        match screen {
            PolicyScreen::ConfirmReplace(mut dialog) => {
                match dialog.handle_key(key) {
                    WidgetResult::Confirmed(true) => {
                        let templates = self.backend.templates();
                        self.phase = WizardPhase::PolicyTemplate(
                            PolicyScreen::select_from(&templates),
                        );
                    }
                    WidgetResult::Confirmed(false) | WidgetResult::Cancelled => {
                        self.phase = WizardPhase::PolicyTemplate(PolicyScreen::Skipped);
                        self.advance_to_users();
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::PolicyTemplate(
                            PolicyScreen::ConfirmReplace(dialog),
                        );
                    }
                }
            }
            PolicyScreen::SelectTemplate(mut list) => {
                match list.handle_key(key) {
                    WidgetResult::Confirmed(selection) => {
                        if selection.starts_with("None") {
                            self.phase = WizardPhase::PolicyTemplate(PolicyScreen::Skipped);
                            self.advance_to_users();
                        } else {
                            let name = selection
                                .split(" —")
                                .next()
                                .unwrap_or("")
                                .trim()
                                .to_string();
                            self.backend.apply_template(&name);
                            self.templates_applied.push(name.clone());
                            self.phase = WizardPhase::PolicyTemplate(
                                PolicyScreen::Applied(name),
                            );
                            // Auto-advance after applying template
                            self.advance_to_users();
                        }
                    }
                    WidgetResult::Cancelled => {
                        self.advance_to_users();
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::PolicyTemplate(
                            PolicyScreen::SelectTemplate(list),
                        );
                    }
                }
            }
            PolicyScreen::Applied(_) | PolicyScreen::Skipped => {
                self.advance_to_users();
            }
        }
        true
    }

    fn handle_users_input(&mut self, key: &KeyPress) -> bool {
        let phase = std::mem::replace(
            &mut self.phase,
            WizardPhase::Users(UserScreen::Done),
        );

        let WizardPhase::Users(screen) = phase else {
            return false;
        };

        match screen {
            UserScreen::AskAdd(mut dialog) => {
                match dialog.handle_key(key) {
                    WidgetResult::Confirmed(true) => {
                        self.phase = WizardPhase::Users(UserScreen::EnterName(
                            TextInput::new("Username:"),
                        ));
                    }
                    WidgetResult::Confirmed(false) | WidgetResult::Cancelled => {
                        self.phase = WizardPhase::Users(UserScreen::Done);
                        self.advance_to_tokens();
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::Users(UserScreen::AskAdd(dialog));
                    }
                }
            }
            UserScreen::EnterName(mut input) => {
                match input.handle_key(key) {
                    WidgetResult::Confirmed(name) => {
                        if name.trim().is_empty() {
                            // Re-prompt
                            self.phase = WizardPhase::Users(UserScreen::EnterName(
                                TextInput::new("Username:"),
                            ));
                        } else {
                            self.pending_username = name.trim().to_string();
                            self.phase = WizardPhase::Users(UserScreen::SelectRole(
                                Self::make_role_select(),
                            ));
                        }
                    }
                    WidgetResult::Cancelled => {
                        self.phase = WizardPhase::Users(UserScreen::Done);
                        self.advance_to_tokens();
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::Users(UserScreen::EnterName(input));
                    }
                }
            }
            UserScreen::SelectRole(mut list) => {
                match list.handle_key(key) {
                    WidgetResult::Confirmed(selection) => {
                        let role_name = selection
                            .split(" —")
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_string();

                        if role_name == "custom" {
                            self.pending_role = "custom".to_string();
                            self.phase = WizardPhase::Users(UserScreen::CustomResource(
                                TextInput::new("Resource pattern (e.g., model:*, *):").with_default("*"),
                            ));
                        } else {
                            self.backend
                                .add_user(&self.pending_username, &role_name);
                            self.users_created.push(UserRecord {
                                username: self.pending_username.clone(),
                                role: role_name,
                            });
                            self.phase = WizardPhase::Users(UserScreen::AskAnother(
                                ConfirmDialog::new("Add another user?").with_default(false),
                            ));
                        }
                    }
                    WidgetResult::Cancelled => {
                        self.phase = WizardPhase::Users(UserScreen::AskAnother(
                            ConfirmDialog::new("Add another user?").with_default(false),
                        ));
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::Users(UserScreen::SelectRole(list));
                    }
                }
            }
            UserScreen::CustomResource(mut input) => {
                match input.handle_key(key) {
                    WidgetResult::Confirmed(resource) => {
                        self.pending_resource = resource;
                        self.phase = WizardPhase::Users(UserScreen::CustomActions(
                            Self::make_action_multiselect(),
                        ));
                    }
                    WidgetResult::Cancelled => {
                        self.phase = WizardPhase::Users(UserScreen::AskAnother(
                            ConfirmDialog::new("Add another user?").with_default(false),
                        ));
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::Users(UserScreen::CustomResource(input));
                    }
                }
            }
            UserScreen::CustomActions(mut list) => {
                match list.handle_key(key) {
                    WidgetResult::Confirmed(actions) => {
                        self.backend.add_user_custom(
                            &self.pending_username,
                            &self.pending_resource,
                            &actions,
                        );
                        let actions_str = actions.join(",");
                        self.users_created.push(UserRecord {
                            username: self.pending_username.clone(),
                            role: format!("custom({actions_str})"),
                        });
                        self.phase = WizardPhase::Users(UserScreen::AskAnother(
                            ConfirmDialog::new("Add another user?").with_default(false),
                        ));
                    }
                    WidgetResult::Cancelled => {
                        self.phase = WizardPhase::Users(UserScreen::AskAnother(
                            ConfirmDialog::new("Add another user?").with_default(false),
                        ));
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::Users(UserScreen::CustomActions(list));
                    }
                }
            }
            UserScreen::AskAnother(mut dialog) => {
                match dialog.handle_key(key) {
                    WidgetResult::Confirmed(true) => {
                        self.phase = WizardPhase::Users(UserScreen::EnterName(
                            TextInput::new("Username:"),
                        ));
                    }
                    WidgetResult::Confirmed(false) | WidgetResult::Cancelled => {
                        self.backend.save_policies();
                        self.phase = WizardPhase::Users(UserScreen::Done);
                        self.advance_to_tokens();
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::Users(UserScreen::AskAnother(dialog));
                    }
                }
            }
            UserScreen::Done => {
                self.advance_to_tokens();
            }
        }
        true
    }

    fn handle_tokens_input(&mut self, key: &KeyPress) -> bool {
        let phase = std::mem::replace(
            &mut self.phase,
            WizardPhase::Tokens(TokenScreen::Done),
        );

        let WizardPhase::Tokens(screen) = phase else {
            return false;
        };

        match screen {
            TokenScreen::AskGenerate(mut dialog, username) => {
                match dialog.handle_key(key) {
                    WidgetResult::Confirmed(true) => {
                        self.phase = WizardPhase::Tokens(TokenScreen::SelectExpiry(
                            Self::make_expiry_select(),
                            username,
                        ));
                    }
                    WidgetResult::Confirmed(false) | WidgetResult::Cancelled => {
                        self.token_queue_idx += 1;
                        self.advance_token_for_next_user();
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::Tokens(TokenScreen::AskGenerate(
                            dialog, username,
                        ));
                    }
                }
            }
            TokenScreen::SelectExpiry(mut list, username) => {
                match list.handle_key(key) {
                    WidgetResult::Confirmed(selection) => {
                        let duration = Self::expiry_to_duration(&selection);
                        let result =
                            self.backend.generate_token(&username, duration);
                        let record = TokenRecord {
                            username: username.clone(),
                            token: result.token,
                            expires: result.expires,
                        };
                        self.tokens_generated.push(record.clone());
                        self.phase = WizardPhase::Tokens(TokenScreen::ShowToken(record));
                    }
                    WidgetResult::Cancelled => {
                        self.token_queue_idx += 1;
                        self.advance_token_for_next_user();
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::Tokens(TokenScreen::SelectExpiry(
                            list, username,
                        ));
                    }
                }
            }
            TokenScreen::ShowToken(_record) => {
                // Any key advances
                self.token_queue_idx += 1;
                self.advance_token_for_next_user();
            }
            TokenScreen::NextUser | TokenScreen::Done => {
                self.advance_to_services();
            }
        }
        true
    }

    fn handle_services_input(&mut self, key: &KeyPress) -> bool {
        let phase = std::mem::replace(
            &mut self.phase,
            WizardPhase::Services(ServiceScreen::Done(false)),
        );

        let WizardPhase::Services(screen) = phase else {
            return false;
        };

        match screen {
            ServiceScreen::AskStart(mut dialog) => {
                match dialog.handle_key(key) {
                    WidgetResult::Confirmed(true) => {
                        self.backend.start_services();
                        self.phase = WizardPhase::Services(ServiceScreen::Starting);
                    }
                    WidgetResult::Confirmed(false) | WidgetResult::Cancelled => {
                        self.advance_to_summary();
                    }
                    WidgetResult::Pending => {
                        self.phase = WizardPhase::Services(ServiceScreen::AskStart(dialog));
                    }
                }
            }
            ServiceScreen::Starting => {
                // tick-driven
                self.phase = WizardPhase::Services(ServiceScreen::Starting);
            }
            ServiceScreen::Done(_) => {
                self.advance_to_summary();
            }
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

fn render_header(frame: &mut Frame, area: Rect) {
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Hyprstream Setup Wizard",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("  {}", "=".repeat(40)),
            Style::default().fg(Color::DarkGray),
        )),
    ];
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_phase_indicator(frame: &mut Frame, area: Rect, current: u8, name: &str) {
    let phases = ["Install", "Bootstrap", "Policy", "Users", "Tokens", "Services"];
    let mut spans = vec![Span::raw("  ")];
    for (i, label) in phases.iter().enumerate() {
        let num = (i + 1) as u8;
        let style = if num < current {
            Style::default().fg(Color::Green)
        } else if num == current {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let marker = if num < current { "v" } else if num == current { ">" } else { " " };
        spans.push(Span::styled(format!("{marker}{num}.{label}"), style));
        if i < phases.len() - 1 {
            spans.push(Span::raw(" "));
        }
    }
    spans.push(Span::raw(format!("  — {name}")));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_separator(frame: &mut Frame, area: Rect) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("  {}", "-".repeat(40)),
            Style::default().fg(Color::DarkGray),
        ))),
        area,
    );
}

fn render_footer(frame: &mut Frame, area: Rect, phase: &WizardPhase) {
    let hint = match phase {
        WizardPhase::Install(screen) => match screen {
            InstallScreen::Detecting => "  Detecting environment...",
            InstallScreen::ShowFindings { action, .. } => match action {
                InstallAction::UpgradeVariant(_) => {
                    "  [Enter] Install  [s] Skip  [v] Choose variant"
                }
                _ => "  [Enter] Continue",
            },
            InstallScreen::SelectVariant(_) => {
                "  Arrow keys to navigate, Enter to select, Esc to skip"
            }
            InstallScreen::Installing { .. } => "  Downloading and installing...",
            InstallScreen::Done(_) => "  [Enter] Continue",
            InstallScreen::Skipped => "  Skipping...",
            InstallScreen::Failed(_) => "  [r] Retry  [s] Skip  [q] Quit",
        },
        WizardPhase::Bootstrap(_) => "  Bootstrapping environment...",
        WizardPhase::PolicyTemplate(_) => "  Arrow keys to navigate, Enter to select, Esc to skip",
        WizardPhase::Users(_) => "  Arrow keys to navigate, Enter to confirm, Esc to skip",
        WizardPhase::Tokens(_) => "  Arrow keys to navigate, Enter to confirm",
        WizardPhase::Services(_) => "  y/n or arrow keys, Enter to confirm",
        WizardPhase::Summary(_) => "  Press Enter or q to exit",
    };
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(hint, Style::default().fg(Color::DarkGray))),
    ];
    frame.render_widget(Paragraph::new(lines), area);
}

impl<B: WizardBackend> WizardApp<B> {
    fn render_content(&self, frame: &mut Frame, area: Rect) {
        // Indent content area
        let content_area = Rect {
            x: area.x + 2,
            width: area.width.saturating_sub(4),
            ..area
        };

        match &self.phase {
            WizardPhase::Install(screen) => {
                render_install(frame, content_area, screen);
            }
            WizardPhase::Bootstrap(screen) => {
                render_bootstrap(frame, content_area, screen);
            }
            WizardPhase::PolicyTemplate(screen) => {
                render_policy(frame, content_area, screen);
            }
            WizardPhase::Users(screen) => {
                render_users(frame, content_area, screen, &self.users_created);
            }
            WizardPhase::Tokens(screen) => {
                render_tokens(frame, content_area, screen);
            }
            WizardPhase::Services(screen) => {
                render_services(frame, content_area, screen);
            }
            WizardPhase::Summary(screen) => {
                render_summary(frame, content_area, screen);
            }
        }
    }
}

fn render_install(frame: &mut Frame, area: Rect, screen: &InstallScreen) {
    use backend::{GpuKind, LIBTORCH_VERSION};

    match screen {
        InstallScreen::Detecting => {
            let lines = vec![
                Line::from(vec![
                    Span::styled("  > ", Style::default().fg(Color::Yellow)),
                    Span::styled("Detecting environment...", Style::default().fg(Color::Yellow)),
                ]),
            ];
            frame.render_widget(Paragraph::new(lines), area);
        }
        InstallScreen::ShowFindings { env, action } => {
            let mut lines = vec![
                Line::from(Span::styled(
                    "Environment Detection",
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    "-".repeat(30),
                    Style::default().fg(Color::DarkGray),
                )),
            ];

            let run_mode_str = match &env.run_mode {
                RunMode::AppImage => "AppImage",
                RunMode::BareBinary => "cargo install (bare binary)",
                RunMode::Development => "development (cargo)",
            };
            lines.push(Line::from(format!("Run mode:    {run_mode_str}")));

            let gpu_str = match &env.gpu {
                GpuKind::Nvidia {
                    driver_version,
                    cuda_compat,
                } => format!("NVIDIA (driver {driver_version}) -> {cuda_compat}"),
                GpuKind::Amd { rocm_version } => {
                    if let Some(ver) = rocm_version {
                        format!("AMD (ROCm {ver})")
                    } else {
                        "AMD (no ROCm)".to_string()
                    }
                }
                GpuKind::None => "None".to_string(),
            };
            lines.push(Line::from(format!("GPU:         {gpu_str}")));
            lines.push(Line::from(format!(
                "libtorch:    {LIBTORCH_VERSION}+{} (compiled)",
                env.current_variant.id()
            )));

            if let Some(ref installed) = env.installed_variant {
                lines.push(Line::from(format!("Installed:   {}", installed.label())));
            } else {
                lines.push(Line::from("Installed:   (none)"));
            }

            lines.push(Line::from(""));

            match action {
                InstallAction::UpgradeVariant(variant) => {
                    lines.push(Line::from(Span::styled(
                        "Recommendation",
                        Style::default().add_modifier(Modifier::BOLD),
                    )));
                    lines.push(Line::from(Span::styled(
                        "-".repeat(30),
                        Style::default().fg(Color::DarkGray),
                    )));
                    lines.push(Line::from(format!(
                        "Download GPU binary + libtorch ({}) (~2.5 GB)",
                        variant.label()
                    )));
                    lines.push(Line::from("Enables GPU-accelerated inference."));
                }
                InstallAction::AlreadyCurrent => {
                    lines.push(Line::from(Span::styled(
                        "Already running the optimal variant.",
                        Style::default().fg(Color::Green),
                    )));
                }
                InstallAction::Skip { reason } => {
                    lines.push(Line::from(Span::styled(
                        format!("Skipping: {reason}"),
                        Style::default().fg(Color::DarkGray),
                    )));
                }
            }

            frame.render_widget(Paragraph::new(lines), area);
        }
        InstallScreen::SelectVariant(list) => {
            list.render(frame, area);
        }
        InstallScreen::Installing {
            progress_pct,
            status_msg,
            ..
        } => {
            let bar_width = 40usize;
            let filled = ((*progress_pct as usize) * bar_width) / 100;
            let empty = bar_width - filled;
            let bar = format!(
                "[{}{}] {:>3}%",
                "#".repeat(filled),
                " ".repeat(empty),
                progress_pct,
            );

            let lines = vec![
                Line::from(Span::raw(format!("  {status_msg}"))),
                Line::from(Span::styled(
                    format!("  {bar}"),
                    Style::default().fg(Color::Cyan),
                )),
            ];
            frame.render_widget(Paragraph::new(lines), area);
        }
        InstallScreen::Done(msg) => {
            let lines = vec![
                Line::from(vec![
                    Span::styled("  v ", Style::default().fg(Color::Green)),
                    Span::raw(msg.as_str()),
                ]),
                Line::from(""),
                Line::from(Span::styled(
                    "  Press Enter to continue...",
                    Style::default().fg(Color::DarkGray),
                )),
            ];
            frame.render_widget(Paragraph::new(lines), area);
        }
        InstallScreen::Skipped => {
            let line = Line::from(Span::styled(
                "  Skipped",
                Style::default().fg(Color::DarkGray),
            ));
            frame.render_widget(Paragraph::new(line), area);
        }
        InstallScreen::Failed(msg) => {
            let lines = vec![
                Line::from(vec![
                    Span::styled("  x ", Style::default().fg(Color::Red)),
                    Span::styled(msg.as_str(), Style::default().fg(Color::Red)),
                ]),
                Line::from(""),
                Line::from(Span::styled(
                    "  [r] Retry  [s] Skip  [q] Quit",
                    Style::default().fg(Color::DarkGray),
                )),
            ];
            frame.render_widget(Paragraph::new(lines), area);
        }
    }
}

fn render_bootstrap(frame: &mut Frame, area: Rect, screen: &BootstrapScreen) {
    let mut lines: Vec<Line> = Vec::new();

    for step in &screen.completed_steps {
        lines.push(Line::from(vec![
            Span::styled("  v ", Style::default().fg(Color::Green)),
            Span::raw(step),
        ]));
    }

    if let Some(ref msg) = screen.failed {
        lines.push(Line::from(vec![
            Span::styled("  x ", Style::default().fg(Color::Red)),
            Span::styled(msg, Style::default().fg(Color::Red)),
        ]));
    } else if !screen.done {
        lines.push(Line::from(vec![
            Span::styled("  > ", Style::default().fg(Color::Yellow)),
            Span::styled(
                &screen.current_step,
                Style::default().fg(Color::Yellow),
            ),
        ]));
    }

    frame.render_widget(Paragraph::new(lines), area);
}

fn render_policy(frame: &mut Frame, area: Rect, screen: &PolicyScreen) {
    match screen {
        PolicyScreen::ConfirmReplace(dialog) => dialog.render(frame, area),
        PolicyScreen::SelectTemplate(list) => list.render(frame, area),
        PolicyScreen::Applied(name) => {
            let line = Line::from(vec![
                Span::styled("  v ", Style::default().fg(Color::Green)),
                Span::raw(format!("Applied template '{name}'")),
            ]);
            frame.render_widget(Paragraph::new(line), area);
        }
        PolicyScreen::Skipped => {
            let line = Line::from(Span::styled(
                "  Skipped",
                Style::default().fg(Color::DarkGray),
            ));
            frame.render_widget(Paragraph::new(line), area);
        }
    }
}

fn render_users(
    frame: &mut Frame,
    area: Rect,
    screen: &UserScreen,
    created: &[UserRecord],
) {
    let mut lines: Vec<Line> = Vec::new();

    // Show already-created users
    for user in created {
        lines.push(Line::from(vec![
            Span::styled("  v ", Style::default().fg(Color::Green)),
            Span::raw(format!("{} ({})", user.username, user.role)),
        ]));
    }

    if !created.is_empty() {
        lines.push(Line::from(""));
    }

    // The remaining area for the active widget
    let used_rows = lines.len() as u16;
    let widget_area = Rect {
        y: area.y + used_rows,
        height: area.height.saturating_sub(used_rows),
        ..area
    };

    frame.render_widget(Paragraph::new(lines), area);

    match screen {
        UserScreen::AskAdd(dialog) => dialog.render(frame, widget_area),
        UserScreen::EnterName(input) => input.render(frame, widget_area),
        UserScreen::SelectRole(list) => list.render(frame, widget_area),
        UserScreen::CustomResource(input) => input.render(frame, widget_area),
        UserScreen::CustomActions(list) => list.render(frame, widget_area),
        UserScreen::AskAnother(dialog) => dialog.render(frame, widget_area),
        UserScreen::Done => {}
    }
}

fn render_tokens(frame: &mut Frame, area: Rect, screen: &TokenScreen) {
    match screen {
        TokenScreen::AskGenerate(dialog, _) => dialog.render(frame, area),
        TokenScreen::SelectExpiry(list, username) => {
            let header = Line::from(Span::raw(format!("  Token for '{username}':")));
            let header_area = Rect {
                height: 1,
                ..area
            };
            frame.render_widget(Paragraph::new(header), header_area);
            let list_area = Rect {
                y: area.y + 2,
                height: area.height.saturating_sub(2),
                ..area
            };
            list.render(frame, list_area);
        }
        TokenScreen::ShowToken(record) => {
            let lines = vec![
                Line::from(vec![
                    Span::styled("  v ", Style::default().fg(Color::Green)),
                    Span::raw(format!("Token for '{}':", record.username)),
                ]),
                Line::from(""),
                Line::from(Span::styled(
                    format!("  {}", record.token),
                    Style::default().fg(Color::Yellow),
                )),
                Line::from(""),
                Line::from(Span::raw(format!("  Expires: {}", record.expires))),
                Line::from(""),
                Line::from(Span::styled(
                    "  Save this token now — it cannot be retrieved again.",
                    Style::default().fg(Color::Yellow),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "  Press any key to continue...",
                    Style::default().fg(Color::DarkGray),
                )),
            ];
            frame.render_widget(Paragraph::new(lines), area);
        }
        TokenScreen::NextUser | TokenScreen::Done => {}
    }
}

fn render_services(frame: &mut Frame, area: Rect, screen: &ServiceScreen) {
    match screen {
        ServiceScreen::AskStart(dialog) => dialog.render(frame, area),
        ServiceScreen::Starting => {
            let line = Line::from(vec![
                Span::styled("  > ", Style::default().fg(Color::Yellow)),
                Span::raw("Starting services..."),
            ]);
            frame.render_widget(Paragraph::new(line), area);
        }
        ServiceScreen::Done(success) => {
            let (marker, msg, color) = if *success {
                ("v", "Services started", Color::Green)
            } else {
                ("x", "Service startup failed", Color::Red)
            };
            let line = Line::from(vec![
                Span::styled(format!("  {marker} "), Style::default().fg(color)),
                Span::raw(msg),
            ]);
            frame.render_widget(Paragraph::new(line), area);
        }
    }
}

fn render_summary(frame: &mut Frame, area: Rect, screen: &SummaryScreen) {
    let mut lines = vec![
        Line::from(Span::styled(
            "Setup Complete",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
    ];

    if let Some(ref install) = screen.install_result {
        lines.push(Line::from(format!("Install: {install}")));
    }

    if !screen.templates_applied.is_empty() {
        lines.push(Line::from(format!(
            "Templates: {}",
            screen.templates_applied.join(", ")
        )));
    }

    if !screen.users_created.is_empty() {
        lines.push(Line::from("Users:"));
        for user in &screen.users_created {
            lines.push(Line::from(format!(
                "  {} ({})",
                user.username, user.role
            )));
        }
    }

    if !screen.tokens_generated.is_empty() {
        lines.push(Line::from("Tokens generated:"));
        for token in &screen.tokens_generated {
            let preview = if token.token.len() > 20 {
                format!("{}...", &token.token[..20])
            } else {
                token.token.clone()
            };
            lines.push(Line::from(format!(
                "  {} — {} (expires {})",
                token.username, preview, token.expires
            )));
        }
    }

    if screen.services_started {
        lines.push(Line::from(Span::styled(
            "Services: running",
            Style::default().fg(Color::Green),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from("Next steps:"));
    lines.push(Line::from("  hyprstream service start"));
    lines.push(Line::from("  hyprstream quick list"));
    lines.push(Line::from("  hyprstream quick clone <model>"));
    lines.push(Line::from("  hyprstream quick infer <model>"));

    frame.render_widget(Paragraph::new(lines), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::{GpuKind, MockWizardBackend, RunMode};
    use waxterm::TerminalApp;

    fn make_app() -> WizardApp<MockWizardBackend> {
        WizardApp::new(MockWizardBackend::new())
    }

    /// Advance past install phase (tick once to detect, then skip).
    fn skip_install(app: &mut WizardApp<MockWizardBackend>) {
        // First tick triggers detection → ShowFindings
        app.tick(33);
        assert_eq!(app.phase_number(), 1, "Should still be at Install after detection");
        // Press 's' to skip
        app.handle_input(WizardCommand::Key(KeyPress::Char(b's')));
        assert_eq!(app.phase_number(), 2, "Should be at Bootstrap after skip");
    }

    /// Advance past bootstrap phase (tick until done).
    fn complete_bootstrap(app: &mut WizardApp<MockWizardBackend>) {
        for _ in 0..100 {
            app.tick(33);
            if app.phase_number() > 2 {
                break;
            }
        }
        assert!(app.phase_number() >= 3, "Should have advanced past bootstrap");
    }

    // ─── Install phase tests ─────────────────────────────────────────────

    #[test]
    fn test_initial_phase_is_install() {
        let app = make_app();
        assert_eq!(app.phase_number(), 1);
        assert!(matches!(app.phase, WizardPhase::Install(InstallScreen::Detecting)));
        assert!(!app.should_quit());
    }

    #[test]
    fn test_install_detect_advances() {
        let mut app = make_app();
        // Tick triggers detection
        app.tick(33);
        assert!(matches!(
            app.phase,
            WizardPhase::Install(InstallScreen::ShowFindings { .. })
        ));
    }

    #[test]
    fn test_install_skip() {
        let mut app = make_app();
        app.tick(33); // detect
        // Press 's' to skip
        app.handle_input(WizardCommand::Key(KeyPress::Char(b's')));
        assert_eq!(app.phase_number(), 2, "Should advance to Bootstrap");
    }

    #[test]
    fn test_install_confirm() {
        let mut app = make_app();
        app.tick(33); // detect → ShowFindings with UpgradeVariant(Cuda130)

        // Press Enter to install
        app.handle_input(WizardCommand::Key(KeyPress::Enter));
        assert!(
            matches!(app.phase, WizardPhase::Install(InstallScreen::Installing { .. })),
            "Should be installing"
        );

        // Tick until done
        for _ in 0..50 {
            app.tick(33);
            if matches!(app.phase, WizardPhase::Install(InstallScreen::Done(_))) {
                break;
            }
        }
        assert!(matches!(app.phase, WizardPhase::Install(InstallScreen::Done(_))));

        // Press Enter to advance
        app.handle_input(WizardCommand::Key(KeyPress::Enter));
        assert_eq!(app.phase_number(), 2, "Should advance to Bootstrap");
        assert!(app.install_summary.is_some());
    }

    #[test]
    fn test_install_select_variant() {
        let mut app = make_app();
        app.tick(33); // detect

        // Press 'v' to pick variant
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'v')));
        assert!(matches!(
            app.phase,
            WizardPhase::Install(InstallScreen::SelectVariant(_))
        ));

        // Select first option (CPU)
        app.handle_input(WizardCommand::Key(KeyPress::Enter));
        assert!(matches!(
            app.phase,
            WizardPhase::Install(InstallScreen::Installing { .. })
        ));
    }

    #[test]
    fn test_install_appimage_auto_skip() {
        let backend = MockWizardBackend::new().with_run_mode(RunMode::AppImage);
        let mut app = WizardApp::new(backend);

        // Tick triggers detection → auto-skip for AppImage
        app.tick(33);
        assert_eq!(
            app.phase_number(),
            2,
            "AppImage should auto-skip to Bootstrap"
        );
    }

    #[test]
    fn test_install_already_current() {
        // No GPU → recommends CPU, compiled variant is CPU → AlreadyCurrent
        let backend = MockWizardBackend::new().with_gpu(GpuKind::None);
        let mut app = WizardApp::new(backend);

        app.tick(33); // detect → AlreadyCurrent → Done
        assert!(
            matches!(app.phase, WizardPhase::Install(InstallScreen::Done(_))),
            "Should show Done for already-current"
        );

        // Enter to continue
        app.handle_input(WizardCommand::Key(KeyPress::Enter));
        assert_eq!(app.phase_number(), 2);
    }

    #[test]
    fn test_install_development_auto_skip() {
        let backend = MockWizardBackend::new().with_run_mode(RunMode::Development);
        let mut app = WizardApp::new(backend);

        app.tick(33);
        assert_eq!(
            app.phase_number(),
            2,
            "Development mode should auto-skip to Bootstrap"
        );
    }

    // ─── Existing tests updated for new phase numbering ──────────────────

    #[test]
    fn test_bootstrap_auto_advances() {
        let mut app = make_app();
        skip_install(&mut app);
        complete_bootstrap(&mut app);
        assert!(app.phase_number() >= 3, "Should have advanced past bootstrap");
    }

    #[test]
    fn test_full_flow_skip_everything() {
        let mut app = make_app();

        // Install: skip
        skip_install(&mut app);

        // Bootstrap
        complete_bootstrap(&mut app);
        assert_eq!(app.phase_number(), 3);

        // Policy: select "None — skip template" (last option)
        for _ in 0..5 {
            app.handle_input(WizardCommand::Key(KeyPress::ArrowDown));
        }
        app.handle_input(WizardCommand::Key(KeyPress::Enter));
        assert_eq!(app.phase_number(), 4, "Should be at Users phase");

        // Users: decline to add user (press 'n')
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n')));
        assert_eq!(app.phase_number(), 5, "Should be at Tokens phase");

        // Tokens: decline to generate (press 'n')
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n')));
        assert_eq!(app.phase_number(), 6, "Should be at Services phase");

        // Services: decline to start (press 'n')
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n')));

        // Should be at summary
        assert!(matches!(app.phase, WizardPhase::Summary(_)));

        // Exit
        app.handle_input(WizardCommand::Key(KeyPress::Enter));
        assert!(app.should_quit());
    }

    #[test]
    fn test_full_flow_with_install() {
        let mut app = make_app();

        // Install: detect → confirm → tick through → done → advance
        app.tick(33);
        app.handle_input(WizardCommand::Key(KeyPress::Enter)); // install
        for _ in 0..50 {
            app.tick(33);
            if matches!(app.phase, WizardPhase::Install(InstallScreen::Done(_))) {
                break;
            }
        }
        app.handle_input(WizardCommand::Key(KeyPress::Enter)); // continue
        assert_eq!(app.phase_number(), 2);

        // Bootstrap
        complete_bootstrap(&mut app);

        // Skip remaining phases
        app.handle_input(WizardCommand::Key(KeyPress::Escape)); // policy
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n'))); // users
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n'))); // tokens
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n'))); // services

        assert!(matches!(app.phase, WizardPhase::Summary(_)));
        if let WizardPhase::Summary(ref s) = app.phase {
            assert!(s.install_result.is_some());
        }
    }

    #[test]
    fn test_select_template() {
        let mut app = make_app();
        skip_install(&mut app);
        complete_bootstrap(&mut app);

        // Select first template (local)
        app.handle_input(WizardCommand::Key(KeyPress::Enter));

        // Should have applied template and advanced
        assert!(!app.templates_applied.is_empty());
        assert_eq!(app.templates_applied[0], "local");
    }

    #[test]
    fn test_quit_at_summary() {
        let mut app = make_app();
        skip_install(&mut app);
        complete_bootstrap(&mut app);

        // Skip all phases
        app.handle_input(WizardCommand::Key(KeyPress::Escape)); // policy
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n'))); // users
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n'))); // tokens
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n'))); // services

        assert!(matches!(app.phase, WizardPhase::Summary(_)));
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'q')));
        assert!(app.should_quit());
    }
}
