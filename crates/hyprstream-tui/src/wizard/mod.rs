//! Setup wizard TUI application.
//!
//! Implements `waxterm::TerminalApp` to drive the wizard through 5 phases:
//! Bootstrap, Policy Templates, Users & Roles, API Tokens, Services.

pub mod backend;
pub mod phases;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use waxterm::input::KeyPress;
use waxterm::widgets::{ConfirmDialog, MultiSelectList, SelectList, TextInput, WidgetResult};

use backend::{BootstrapPoll, OpStatus, WizardBackend};
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
}

impl<B: WizardBackend> WizardApp<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            phase: WizardPhase::Bootstrap(BootstrapScreen::new()),
            quit: false,
            templates_applied: Vec::new(),
            users_created: Vec::new(),
            tokens_generated: Vec::new(),
            services_started: false,
            pending_username: String::new(),
            pending_role: String::new(),
            pending_resource: String::new(),
            token_queue: Vec::new(),
            token_queue_idx: 0,
        }
    }

    fn phase_number(&self) -> u8 {
        match &self.phase {
            WizardPhase::Bootstrap(_) => 1,
            WizardPhase::PolicyTemplate(_) => 2,
            WizardPhase::Users(_) => 3,
            WizardPhase::Tokens(_) => 4,
            WizardPhase::Services(_) => 5,
            WizardPhase::Summary(_) => 5,
        }
    }

    fn phase_name(&self) -> &str {
        match &self.phase {
            WizardPhase::Bootstrap(_) => "Environment Bootstrap",
            WizardPhase::PolicyTemplate(_) => "Policy Template",
            WizardPhase::Users(_) => "Users & Roles",
            WizardPhase::Tokens(_) => "API Tokens",
            WizardPhase::Services(_) => "Services",
            WizardPhase::Summary(_) => "Setup Complete",
        }
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
            templates_applied: self.templates_applied.clone(),
            users_created: self.users_created.clone(),
            tokens_generated: self.tokens_generated.clone(),
            services_started: self.services_started,
        });
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
            1 => {
                // Bootstrap is tick-driven, no input needed
                false
            }
            2 => self.handle_policy_input(&key),
            3 => self.handle_users_input(&key),
            4 => self.handle_tokens_input(&key),
            5 => {
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
    let phases = ["Bootstrap", "Policy", "Users", "Tokens", "Services"];
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
    use backend::MockWizardBackend;
    use waxterm::TerminalApp;

    fn make_app() -> WizardApp<MockWizardBackend> {
        WizardApp::new(MockWizardBackend::new())
    }

    #[test]
    fn test_initial_phase_is_bootstrap() {
        let app = make_app();
        assert_eq!(app.phase_number(), 1);
        assert!(!app.should_quit());
    }

    #[test]
    fn test_bootstrap_auto_advances() {
        let mut app = make_app();
        // Tick enough times for bootstrap to complete
        for _ in 0..100 {
            app.tick(33);
            if app.phase_number() > 1 {
                break;
            }
        }
        assert!(app.phase_number() >= 2, "Should have advanced past bootstrap");
    }

    #[test]
    fn test_full_flow_skip_everything() {
        let mut app = make_app();

        // Bootstrap
        for _ in 0..100 {
            app.tick(33);
            if app.phase_number() > 1 {
                break;
            }
        }
        assert_eq!(app.phase_number(), 2);

        // Policy: select "None — skip template" (last option)
        // Navigate down to last option
        for _ in 0..5 {
            app.handle_input(WizardCommand::Key(KeyPress::ArrowDown));
        }
        app.handle_input(WizardCommand::Key(KeyPress::Enter));
        assert_eq!(app.phase_number(), 3, "Should be at Users phase");

        // Users: decline to add user (press 'n')
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n')));
        assert_eq!(app.phase_number(), 4, "Should be at Tokens phase");

        // Tokens: decline to generate (press 'n')
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n')));
        assert_eq!(app.phase_number(), 5, "Should be at Services phase");

        // Services: decline to start (press 'n')
        app.handle_input(WizardCommand::Key(KeyPress::Char(b'n')));

        // Should be at summary
        assert!(matches!(app.phase, WizardPhase::Summary(_)));

        // Exit
        app.handle_input(WizardCommand::Key(KeyPress::Enter));
        assert!(app.should_quit());
    }

    #[test]
    fn test_select_template() {
        let mut app = make_app();

        // Bootstrap
        for _ in 0..100 {
            app.tick(33);
            if app.phase_number() > 1 {
                break;
            }
        }

        // Select first template (local)
        app.handle_input(WizardCommand::Key(KeyPress::Enter));

        // Should have applied template and advanced
        assert!(!app.templates_applied.is_empty());
        assert_eq!(app.templates_applied[0], "local");
    }

    #[test]
    fn test_quit_at_summary() {
        let mut app = make_app();

        // Fast-forward to summary
        for _ in 0..100 {
            app.tick(33);
            if app.phase_number() > 1 {
                break;
            }
        }
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
