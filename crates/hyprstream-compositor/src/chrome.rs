//! Shell chrome state machine — pure, no I/O, WASM-safe.
//!
//! `ShellChrome` holds all the TUI chrome state: mode, window list, model
//! list, background animation, and settings.  `handle_key` returns
//! `Vec<ChromeOutput>` describing what the event loop should do — no ZMQ,
//! no tokio, no side-effects.

use std::path::PathBuf;

use waxterm::input::KeyPress;
use waxterm::widgets::{SelectList, WidgetResult};

// ============================================================================
// Toast notification types
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub enum ToastLevel {
    Info,
    Warn,
    Error,
}

pub struct Toast {
    pub message: String,
    pub level:   ToastLevel,
    born:        std::time::Instant,
    ttl:         std::time::Duration,
}

use crate::background::{BackgroundState, BackgroundStyle, ALL_STYLES, PREVIEW_H, PREVIEW_W};

// ============================================================================
// Public types re-used by consumers
// ============================================================================

/// A model repo entry.
#[derive(Clone)]
pub struct ModelEntry {
    pub model_ref: String,
    pub path: PathBuf,
    pub loaded: bool,
    pub loading: bool,
}

impl std::fmt::Display for ModelEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.loading {
            write!(f, "{} [loading\u{2026}]", self.model_ref)
        } else if self.loaded {
            write!(f, "{} [loaded]", self.model_ref)
        } else {
            write!(f, "{}", self.model_ref)
        }
    }
}

/// Lightweight window descriptor from TuiService.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WindowSummary {
    pub id: u32,
    pub name: String,
    pub active_pane_id: u32,
    pub panes: Vec<PaneSummary>,
}

/// Lightweight pane descriptor.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaneSummary {
    pub id: u32,
    pub cols: u16,
    pub rows: u16,
    pub is_private: bool,
}

// ============================================================================
// ShellMode
// ============================================================================

pub enum ShellMode {
    /// Keys forwarded to the active shell via SendInput RPC.
    Normal,
    /// Model-list modal is open.
    ModelList,
    /// Settings modal is open.
    Settings,
    /// Start-menu popup (Ctrl+Space).
    StartMenu { selected: usize },
    /// Console log overlay (F9).
    Console,
}

/// Menu items: (label, chord hint shown in popup).
pub const MENU_ITEMS: &[(&str, &str)] = &[
    ("New",        "N"),
    ("Close",      "Q"),
    ("Cycle",      "Tab"),
    ("Models",     "M"),
    ("Settings",   "S"),
    ("Disconnect", "D"),
];

// ============================================================================
// RPC requests emitted by the chrome
// ============================================================================

/// Pure typed enum — no ZMQ types, no capnp types, WASM-safe.
#[derive(Debug, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RpcRequest {
    SendInput   { viewer_id: u32, data: Vec<u8> },
    CreateWindow { session_id: u32 },
    CloseWindow  { session_id: u32, window_id: u32 },
    FocusWindow  { session_id: u32, window_id: u32 },
    SpawnShell   { session_id: u32, pane_id: u32, cwd: String },
    LoadModel    { model_ref: String },
    UnloadModel  { model_ref: String },
    SpawnServerChat {
        session_id: u32,
        model_ref: String,
        cols: u16,
        rows: u16,
        pane_id: u32,
    },
    CreatePrivatePane {
        session_id: u32,
        cols: u16,
        rows: u16,
        name: String,
    },
    Quit,
}

/// Output from `ShellChrome::handle_key`.
pub enum ChromeOutput {
    Redraw,
    Rpc(RpcRequest),
    /// Key bytes that should be routed to a client-owned ChatApp task.
    RouteInput { app_id: u32, data: Vec<u8> },
}

// ============================================================================
// ShellChrome
// ============================================================================

pub struct ShellChrome {
    pub mode: ShellMode,
    pub windows: Vec<WindowSummary>,
    pub active_win: usize,
    pub model_list: SelectList<ModelEntry>,
    pub load_status: Option<String>,
    /// Background animation (full terminal area).
    pub bg: BackgroundState,
    /// Background animation for the settings preview box.
    pub preview_bg: BackgroundState,
    /// Settings drop-down.
    pub settings_list: SelectList<BackgroundStyle>,
    /// Saved background style — restored when settings is cancelled.
    saved_style: BackgroundStyle,
    /// pane IDs that are private (client-owned ChatApp); used for 🔒 indicator.
    pub private_panes: std::collections::HashSet<u32>,
    pub cols: u16,
    pub pane_rows: u16,
    pub session_id: u32,
    pub viewer_id: u32,
    /// Active toast notifications.
    pub toasts: std::collections::VecDeque<Toast>,
    /// Console log history (capped at 500 lines).
    pub console_log: std::collections::VecDeque<String>,
    /// Scroll offset for console modal.
    pub console_scroll: usize,
}

impl ShellChrome {
    pub fn new(
        cols: u16,
        pane_rows: u16,
        session_id: u32,
        viewer_id: u32,
        windows: Vec<WindowSummary>,
        models: Vec<ModelEntry>,
    ) -> Self {
        let model_list = SelectList::new("Models", models);
        let default_style = BackgroundStyle::Stars;
        let settings_list = SelectList::new("Background", ALL_STYLES.to_vec())
            .with_selected(1); // Stars is index 1

        let mut bg = BackgroundState::new(default_style);
        bg.tick(cols, pane_rows);

        Self {
            mode: ShellMode::Normal,
            windows,
            active_win: 0,
            model_list,
            load_status: None,
            bg,
            preview_bg: BackgroundState::new(default_style),
            settings_list,
            saved_style: default_style,
            private_panes: std::collections::HashSet::new(),
            cols,
            pane_rows,
            session_id,
            viewer_id,
            toasts: std::collections::VecDeque::new(),
            console_log: std::collections::VecDeque::new(),
            console_scroll: 0,
        }
    }

    /// Update model loaded status (called from background polling results).
    pub fn update_model_status(&mut self, model_ref: &str, loaded: bool) {
        for entry in self.model_list.items_mut() {
            if entry.model_ref == model_ref {
                entry.loaded = loaded;
                entry.loading = false;
            }
        }
        if self.load_status.as_ref().is_some_and(|s| s.contains(model_ref)) {
            self.load_status = None;
        }
    }

    /// Update window list (called when TuiService reports window changes).
    pub fn update_windows(&mut self, wins: Vec<WindowSummary>) -> bool {
        // Sync private_panes from the window list (server is authoritative).
        let new_private: std::collections::HashSet<u32> = wins.iter()
            .flat_map(|w| w.panes.iter())
            .filter(|p| p.is_private)
            .map(|p| p.id)
            .collect();
        let private_changed = new_private != self.private_panes;
        self.private_panes = new_private;

        // Also detect pane-list changes within existing windows (e.g. a private pane
        // added to an existing window without changing the window count/IDs).
        let changed = wins.len() != self.windows.len()
            || wins.iter().zip(&self.windows).any(|(a, b)| {
                a.id != b.id || a.panes.len() != b.panes.len()
            });
        if changed || private_changed {
            self.active_win = self.active_win.min(wins.len().saturating_sub(1));
            self.windows = wins;
        }
        changed || private_changed
    }

    /// Active pane ID in the currently focused window.
    /// Prefers a locally-owned private pane over the server-managed active pane,
    /// since the server intentionally keeps the non-private pane as active_pane_id
    /// to avoid publishing private content to viewers.
    pub fn active_pane_id(&self) -> u32 {
        let Some(win) = self.windows.get(self.active_win) else { return 0 };
        if let Some(p) = win.panes.iter().find(|p| self.private_panes.contains(&p.id)) {
            return p.id;
        }
        win.active_pane_id
    }

    /// Tick background animations. Returns true if animated (needs redraw).
    pub fn tick_bg(&mut self) -> bool {
        let show_bg = self.windows.is_empty() || matches!(self.mode, ShellMode::Settings);
        if show_bg {
            self.bg.tick(self.cols, self.pane_rows);
        }
        if matches!(self.mode, ShellMode::Settings) {
            self.preview_bg.tick(PREVIEW_W, PREVIEW_H);
        }
        show_bg && self.bg.is_animated()
    }

    /// Push a toast notification. Also persists the message to console_log.
    pub fn push_toast(&mut self, msg: impl Into<String>, level: ToastLevel) {
        let message: String = msg.into();
        self.log_line(message.clone());
        let ttl = match level {
            ToastLevel::Error => std::time::Duration::from_secs(5),
            ToastLevel::Warn  => std::time::Duration::from_secs(4),
            ToastLevel::Info  => std::time::Duration::from_secs(2),
        };
        self.toasts.push_back(Toast { message, level, born: std::time::Instant::now(), ttl });
        if self.toasts.len() > 5 {
            self.toasts.pop_front();
        }
    }

    /// Expire stale toasts. Returns true if any expired (caller should redraw).
    pub fn tick_toasts(&mut self) -> bool {
        let before = self.toasts.len();
        self.toasts.retain(|t| t.born.elapsed() < t.ttl);
        self.toasts.len() != before
    }

    /// Append a line to the console log (capped at 500 entries).
    pub fn log_line(&mut self, line: impl Into<String>) {
        self.console_log.push_back(line.into());
        if self.console_log.len() > 500 {
            self.console_log.pop_front();
        }
    }

    // =========================================================================
    // Key dispatch
    // =========================================================================

    pub fn handle_key(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        match self.mode {
            ShellMode::Normal                 => self.handle_normal(key),
            ShellMode::ModelList              => self.handle_model_list(key),
            ShellMode::Settings               => self.handle_settings(key),
            ShellMode::StartMenu { selected } => self.handle_start_menu(key, selected),
            ShellMode::Console                => self.handle_console(key),
        }
    }

    // ── Normal mode ──────────────────────────────────────────────────────────

    fn handle_normal(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        match key {
            KeyPress::F(12) => vec![ChromeOutput::Rpc(RpcRequest::Quit)],
            KeyPress::F(9) => {
                self.console_scroll = self.console_log.len().saturating_sub(1);
                self.mode = ShellMode::Console;
                vec![ChromeOutput::Redraw]
            }
            KeyPress::F(10) => {
                self.mode = ShellMode::ModelList;
                vec![ChromeOutput::Redraw]
            }
            KeyPress::F(11) => {
                self.saved_style = self.bg.style;
                let idx = ALL_STYLES.iter().position(|s| *s == self.bg.style).unwrap_or(0);
                self.settings_list = SelectList::new("Background", ALL_STYLES.to_vec())
                    .with_selected(idx);
                self.preview_bg.style = self.bg.style;
                self.mode = ShellMode::Settings;
                vec![ChromeOutput::Redraw]
            }
            KeyPress::F(7) => {
                let sid = self.session_id;
                vec![ChromeOutput::Rpc(RpcRequest::CreateWindow { session_id: sid })]
            }
            KeyPress::F(8) => {
                if let Some(win) = self.windows.get(self.active_win) {
                    let sid = self.session_id;
                    let wid = win.id;
                    return vec![ChromeOutput::Rpc(RpcRequest::CloseWindow {
                        session_id: sid,
                        window_id: wid,
                    })];
                }
                vec![]
            }
            KeyPress::CtrlSpace => {
                self.mode = ShellMode::StartMenu { selected: 0 };
                vec![ChromeOutput::Redraw]
            }
            KeyPress::Char(0x02) => {
                // Ctrl-B — handled at event loop level; chrome does nothing.
                vec![]
            }
            key => {
                let bytes = keypress_to_bytes(key);
                if bytes.is_empty() {
                    vec![]
                } else {
                    let active = self.active_pane_id();
                    if self.private_panes.contains(&active) {
                        // Route to client-owned ChatApp task instead of server
                        vec![ChromeOutput::RouteInput { app_id: active, data: bytes }]
                    } else {
                        let vid = self.viewer_id;
                        vec![ChromeOutput::Rpc(RpcRequest::SendInput { viewer_id: vid, data: bytes })]
                    }
                }
            }
        }
    }

    // ── ModelList mode ───────────────────────────────────────────────────────

    fn handle_model_list(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        // l — submit load request
        if matches!(key, KeyPress::Char(b'l' | b'L')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let model_ref = model.model_ref.clone();
                self.load_status = Some(format!("Loading {model_ref}\u{2026}"));
                for entry in self.model_list.items_mut() {
                    if entry.model_ref == model_ref {
                        entry.loading = true;
                    }
                }
                return vec![ChromeOutput::Rpc(RpcRequest::LoadModel { model_ref })];
            }
        }
        // u — unload
        if matches!(key, KeyPress::Char(b'u' | b'U')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let model_ref = model.model_ref.clone();
                return vec![ChromeOutput::Rpc(RpcRequest::UnloadModel { model_ref })];
            }
        }
        // c (lowercase) — private chat (client-owned, encrypted history)
        if matches!(key, KeyPress::Char(b'c')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let sid  = self.session_id;
                let cols = self.cols;
                let rows = self.pane_rows;
                let name = model.model_ref.clone();
                self.mode = ShellMode::Normal;
                return vec![ChromeOutput::Rpc(RpcRequest::CreatePrivatePane {
                    session_id: sid,
                    cols,
                    rows,
                    name,
                })];
            }
        }
        // C (shift) / Enter — server-side chat
        let is_server_chat = matches!(key, KeyPress::Enter | KeyPress::Char(b'C'));
        if is_server_chat {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let sid  = self.session_id;
                let cols = self.cols;
                let rows = self.pane_rows;
                let mref = model.model_ref.clone();
                self.mode = ShellMode::Normal;
                // SpawnServerChat handler creates its own window; no separate CreateWindow.
                return vec![ChromeOutput::Rpc(RpcRequest::SpawnServerChat {
                    session_id: sid,
                    model_ref: mref,
                    cols, rows,
                    pane_id: 0,
                })];
            }
        }
        // T — terminal in model worktree
        if matches!(key, KeyPress::Char(b't' | b'T')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let cwd = model.path.to_string_lossy().into_owned();
                let sid = self.session_id;
                self.mode = ShellMode::Normal;
                // SpawnShell handler creates its own window when pane_id == 0.
                return vec![ChromeOutput::Rpc(RpcRequest::SpawnShell {
                    session_id: sid,
                    pane_id: 0,
                    cwd,
                })];
            }
        }

        match self.model_list.handle_key(&key) {
            WidgetResult::Cancelled => {
                self.mode = ShellMode::Normal;
                vec![ChromeOutput::Redraw]
            }
            WidgetResult::Confirmed(model) => {
                let sid  = self.session_id;
                let cols = self.cols;
                let rows = self.pane_rows;
                let mref = model.model_ref.clone();
                self.mode = ShellMode::Normal;
                // SpawnServerChat handler creates its own window; no separate CreateWindow.
                vec![ChromeOutput::Rpc(RpcRequest::SpawnServerChat {
                    session_id: sid,
                    model_ref: mref,
                    cols, rows,
                    pane_id: 0,
                })]
            }
            WidgetResult::Pending => {
                if matches!(key, KeyPress::F(10)) {
                    self.mode = ShellMode::Normal;
                }
                vec![ChromeOutput::Redraw]
            }
        }
    }

    // ── StartMenu mode ───────────────────────────────────────────────────────

    fn handle_start_menu(&mut self, key: KeyPress, selected: usize) -> Vec<ChromeOutput> {
        let n = MENU_ITEMS.len();
        let chord: Option<usize> = match key {
            KeyPress::Char(b'n' | b'N') => Some(0),
            KeyPress::Char(b'q' | b'Q') => Some(1),
            KeyPress::Tab               => Some(2),
            KeyPress::Char(b'm' | b'M') => Some(3),
            KeyPress::Char(b's' | b'S') => Some(4),
            KeyPress::Char(b'd' | b'D') => Some(5),
            _ => None,
        };
        if let Some(idx) = chord {
            self.mode = ShellMode::Normal;
            return self.execute_menu_action(idx);
        }
        match key {
            KeyPress::Escape | KeyPress::CtrlSpace => {
                self.mode = ShellMode::Normal;
            }
            KeyPress::ArrowUp => {
                self.mode = ShellMode::StartMenu {
                    selected: if selected == 0 { n - 1 } else { selected - 1 },
                };
            }
            KeyPress::ArrowDown => {
                self.mode = ShellMode::StartMenu { selected: (selected + 1) % n };
            }
            KeyPress::Enter => {
                self.mode = ShellMode::Normal;
                return self.execute_menu_action(selected);
            }
            _ => {}
        }
        vec![ChromeOutput::Redraw]
    }

    fn execute_menu_action(&mut self, idx: usize) -> Vec<ChromeOutput> {
        let sid = self.session_id;
        match idx {
            0 => vec![ChromeOutput::Rpc(RpcRequest::CreateWindow { session_id: sid })],
            1 => {
                if let Some(win) = self.windows.get(self.active_win) {
                    let wid = win.id;
                    vec![ChromeOutput::Rpc(RpcRequest::CloseWindow {
                        session_id: sid, window_id: wid,
                    })]
                } else {
                    vec![]
                }
            }
            2 => {
                // Cycle windows — emit FocusWindow; active_win is updated in handle_rpc
                // after the RPC succeeds, so we don't diverge if the target was closed.
                if !self.windows.is_empty() {
                    let next = (self.active_win + 1) % self.windows.len();
                    if let Some(w) = self.windows.get(next) {
                        let wid = w.id;
                        return vec![ChromeOutput::Rpc(RpcRequest::FocusWindow {
                            session_id: sid, window_id: wid,
                        })];
                    }
                }
                vec![ChromeOutput::Redraw]
            }
            3 => {
                self.mode = ShellMode::ModelList;
                vec![ChromeOutput::Redraw]
            }
            4 => {
                self.saved_style = self.bg.style;
                let i = ALL_STYLES.iter().position(|s| *s == self.bg.style).unwrap_or(0);
                self.settings_list = SelectList::new("Background", ALL_STYLES.to_vec())
                    .with_selected(i);
                self.preview_bg.style = self.bg.style;
                self.mode = ShellMode::Settings;
                vec![ChromeOutput::Redraw]
            }
            5 => vec![ChromeOutput::Rpc(RpcRequest::Quit)],
            _ => vec![],
        }
    }

    // ── Console mode ─────────────────────────────────────────────────────────

    fn handle_console(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        match key {
            KeyPress::ArrowUp => {
                self.console_scroll = self.console_scroll.saturating_sub(1);
                vec![ChromeOutput::Redraw]
            }
            KeyPress::ArrowDown => {
                let max = self.console_log.len().saturating_sub(1);
                if self.console_scroll < max {
                    self.console_scroll += 1;
                }
                vec![ChromeOutput::Redraw]
            }
            KeyPress::Escape | KeyPress::F(9) => {
                self.mode = ShellMode::Normal;
                vec![ChromeOutput::Redraw]
            }
            _ => vec![],
        }
    }

    // ── Settings mode ────────────────────────────────────────────────────────

    fn handle_settings(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        match self.settings_list.handle_key(&key) {
            WidgetResult::Cancelled => {
                self.bg.style         = self.saved_style;
                self.preview_bg.style = self.saved_style;
                self.mode             = ShellMode::Normal;
                vec![ChromeOutput::Redraw]
            }
            WidgetResult::Confirmed(style) => {
                self.bg.style         = style;
                self.preview_bg.style = style;
                self.mode             = ShellMode::Normal;
                vec![ChromeOutput::Redraw]
            }
            WidgetResult::Pending => {
                if let Some(&style) = self.settings_list.selected_item() {
                    self.bg.style         = style;
                    self.preview_bg.style = style;
                }
                vec![ChromeOutput::Redraw]
            }
        }
    }
}

// ============================================================================
// KeyPress → raw bytes (for forwarding to PTY via sendInput)
// ============================================================================

pub fn keypress_to_bytes(key: KeyPress) -> Vec<u8> {
    match key {
        KeyPress::Char(b)    => vec![b],
        KeyPress::Enter      => vec![b'\r'],
        KeyPress::Backspace  => vec![0x7f],
        KeyPress::Tab        => vec![b'\t'],
        KeyPress::Escape     => vec![0x1b],
        KeyPress::ArrowUp    => b"\x1b[A".to_vec(),
        KeyPress::ArrowDown  => b"\x1b[B".to_vec(),
        KeyPress::ArrowRight => b"\x1b[C".to_vec(),
        KeyPress::ArrowLeft  => b"\x1b[D".to_vec(),
        KeyPress::F(1)  => b"\x1bOP".to_vec(),
        KeyPress::F(2)  => b"\x1bOQ".to_vec(),
        KeyPress::F(3)  => b"\x1bOR".to_vec(),
        KeyPress::F(4)  => b"\x1bOS".to_vec(),
        KeyPress::F(5)  => b"\x1b[15~".to_vec(),
        KeyPress::F(6)  => b"\x1b[17~".to_vec(),
        KeyPress::F(7)  => b"\x1b[18~".to_vec(),
        KeyPress::F(8)  => b"\x1b[19~".to_vec(),
        KeyPress::F(9)  => b"\x1b[20~".to_vec(),
        KeyPress::F(10) => b"\x1b[21~".to_vec(),
        KeyPress::F(11) => b"\x1b[23~".to_vec(),
        KeyPress::F(12) => b"\x1b[24~".to_vec(),
        KeyPress::F(_) | KeyPress::CtrlSpace => vec![],
    }
}
