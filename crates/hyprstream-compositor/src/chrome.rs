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

/// Bit-31 marks client-local IDs (windows/panes not on the server).
pub const LOCAL_ID_BIT: u32 = 0x8000_0000;

/// Returns true if the ID is client-local (bit 31 set).
pub fn is_local_id(id: u32) -> bool {
    id & LOCAL_ID_BIT != 0
}

use crate::background::{BackgroundState, BackgroundStyle, ALL_STYLES, PREVIEW_H, PREVIEW_W};

// ============================================================================
// Public types re-used by consumers
// ============================================================================

/// One row in the Service Manager modal.
#[derive(Clone, Debug)]
pub struct ServiceEntry {
    pub name: String,
    pub active: bool,
    pub mode: ServiceMode,
    pub pid: Option<u32>,
}

/// One row in the Images tab of the Worker Manager.
#[derive(Clone, Debug)]
pub struct ImageEntry {
    pub repo_tag: String,
    pub id: String,
    pub size_bytes: u64,
    pub created: String,
}

/// Tab within the Worker Manager modal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorkerTab {
    Sandboxes,
    Images,
}

/// Input dialog for text entry (pull image, create sandbox, create container).
#[derive(Debug)]
pub struct InputDialog {
    pub title: &'static str,
    pub fields: Vec<InputField>,
    pub focused: usize,
}

/// One field in an InputDialog.
#[derive(Debug)]
pub struct InputField {
    pub label: &'static str,
    pub buf: Vec<u8>,
    pub cursor: usize,
    /// If Some, field cycles through these choices instead of free text.
    pub choices: Option<Vec<String>>,
    pub choice_idx: usize,
    /// If true, field is a boolean toggle.
    pub is_toggle: bool,
    pub toggle_val: bool,
}

impl InputField {
    pub fn text(label: &'static str) -> Self {
        Self { label, buf: Vec::new(), cursor: 0, choices: None, choice_idx: 0, is_toggle: false, toggle_val: false }
    }
    pub fn text_with_default(label: &'static str, default: &str) -> Self {
        let buf = default.as_bytes().to_vec();
        let cursor = buf.len();
        Self { label, buf, cursor, choices: None, choice_idx: 0, is_toggle: false, toggle_val: false }
    }
    pub fn toggle(label: &'static str) -> Self {
        Self { label, buf: Vec::new(), cursor: 0, choices: None, choice_idx: 0, is_toggle: true, toggle_val: false }
    }
    pub fn choices(label: &'static str, options: Vec<String>) -> Self {
        Self { label, buf: Vec::new(), cursor: 0, choices: Some(options), choice_idx: 0, is_toggle: false, toggle_val: false }
    }
    pub fn value(&self) -> String {
        if self.is_toggle {
            if self.toggle_val { "true".to_owned() } else { "false".to_owned() }
        } else if let Some(ref opts) = self.choices {
            opts.get(self.choice_idx).cloned().unwrap_or_default()
        } else {
            String::from_utf8_lossy(&self.buf).to_string()
        }
    }
}

impl InputDialog {
    pub fn focused_field(&mut self) -> Option<&mut InputField> {
        self.fields.get_mut(self.focused)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ServiceMode {
    Systemd,
    Daemon,
    Both,
    Stopped,
}

/// One sandbox row in the Worker Manager modal.
#[derive(Clone, Debug)]
pub struct WorkerEntry {
    pub id: String,
    pub full_id: String,
    pub state: String,
    pub backend: String,
    pub cpu_pct: Option<u32>,
    pub mem_mb: Option<u64>,
    pub containers: Vec<ContainerEntry>,
}

#[derive(Clone, Debug)]
pub struct ContainerEntry {
    pub id: String,
    pub full_id: String,
    pub image: String,
    pub state: String,
    pub cpu_pct: Option<u32>,
    pub mem_mb: Option<u64>,
}

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

/// Entry in the conversation picker.
#[derive(Clone, Debug)]
pub struct ConversationPickerEntry {
    /// UUID as string (avoids uuid dependency in compositor).
    pub uuid: String,
    pub label: String,
    pub last_active: u64,
}

impl std::fmt::Display for ConversationPickerEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

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
    /// Fullscreen — all chrome hidden, pane fills terminal.
    Fullscreen,
    /// Conversation picker for a specific model.
    ConversationPicker {
        model_ref: String,
        list: waxterm::widgets::SelectList<ConversationPickerEntry>,
    },
    /// Service Manager modal (F5).
    ServiceManager { selected: usize },
    /// Worker / Sandbox Manager modal (F6).
    WorkerManager {
        sandbox_sel: usize,
        container_sel: usize,
        show_containers: bool,
        tab: WorkerTab,
        image_sel: usize,
        input_mode: Option<InputDialog>,
    },
}

/// Menu items: (label, chord hint shown in popup).
#[cfg(feature = "experimental")]
pub const MENU_ITEMS: &[(&str, &str)] = &[
    ("New",        "N"),
    ("Close",      "Q"),
    ("Cycle",      "Tab"),
    ("Models",     "M"),
    ("Log",        "L"),
    ("Workers",    "W"),
    ("Services",   "V"),
    ("Settings",   "S"),
    ("Fullscreen", "F"),
    ("Disconnect", "D"),
];

/// Menu items: (label, chord hint shown in popup).
#[cfg(not(feature = "experimental"))]
pub const MENU_ITEMS: &[(&str, &str)] = &[
    ("New",        "N"),
    ("Close",      "Q"),
    ("Cycle",      "Tab"),
    ("Models",     "M"),
    ("Log",        "L"),
    ("Workers",    "W"),
    ("Settings",   "S"),
    ("Fullscreen", "F"),
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
    /// Client-local private chat — no server round-trip required.
    /// If `resume_uuid` is Some, resumes an existing conversation.
    LocalPrivateChat {
        model_ref: String,
        cols: u16,
        rows: u16,
        resume_uuid: Option<String>,
    },
    /// Delete a private conversation (UUID as string).
    DeleteConversation {
        uuid: String,
        model_ref: String,
    },
    /// Request the conversation list for a model (shell handler populates it).
    ListConversations {
        model_ref: String,
    },
    ServiceStart { name: String },
    ServiceStop { name: String },
    ServiceRestart { name: String },
    ServiceInstall,
    ServiceStopAll,
    ServiceStartAll,
    WorkerDestroySandbox { sandbox_id: String },
    WorkerExecSync { sandbox_id: String, container_id: String, cmd: Vec<String> },
    WorkerImageList,
    WorkerImagePull { image: String },
    WorkerImageRemove { image: String },
    WorkerCreateSandbox { hostname: String, backend: String, gpu: bool },
    WorkerCreateContainer { sandbox_id: String, image: String, cmd: Vec<String>, tty: bool },
    WorkerStartContainer { container_id: String },
    WorkerStopContainer { container_id: String },
    WorkerRemoveContainer { container_id: String },
    AttachContainer { sandbox_id: String, container_id: String, cols: u16, rows: u16 },
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
    /// Service entries for the Service Manager modal.
    pub service_list: Vec<ServiceEntry>,
    /// Worker/sandbox entries for the Worker Manager modal.
    pub worker_list: Vec<WorkerEntry>,
    /// Pool summary line for the Worker Manager modal.
    pub worker_pool_summary: String,
    /// Image list for the Images tab.
    pub image_list: Vec<ImageEntry>,
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
            service_list: Vec::new(),
            worker_list: Vec::new(),
            worker_pool_summary: String::new(),
            image_list: Vec::new(),
        }
    }

    /// Update model loaded status (called from background polling results).
    pub fn update_model_status(&mut self, model_ref: &str, loaded: bool) {
        let mut was_loading = false;
        for entry in self.model_list.items_mut() {
            if entry.model_ref == model_ref {
                was_loading = was_loading || entry.loading;
                entry.loaded = loaded;
                entry.loading = false;
            }
        }
        if was_loading && loaded {
            self.push_toast(format!("Loaded {model_ref}"), ToastLevel::Info);
        }
    }

    /// Update window list (called when TuiService reports window changes).
    /// Merges server windows with existing client-local windows (bit-31 set).
    pub fn update_windows(&mut self, server_wins: Vec<WindowSummary>) -> bool {
        // Preserve client-local windows (bit-31 set).
        let local_wins: Vec<WindowSummary> = self.windows.drain(..)
            .filter(|w| is_local_id(w.id))
            .collect();

        // Sync private_panes: combine server-reported + locally-tracked.
        let mut new_private: std::collections::HashSet<u32> = server_wins.iter()
            .flat_map(|w| w.panes.iter())
            .filter(|p| p.is_private)
            .map(|p| p.id)
            .collect();
        for w in &local_wins {
            for p in &w.panes {
                if p.is_private { new_private.insert(p.id); }
            }
        }
        let private_changed = new_private != self.private_panes;
        self.private_panes = new_private;

        // Merge: server windows first, then local windows.
        let mut merged = server_wins;
        merged.extend(local_wins);

        let changed = merged.len() != self.windows.len()
            || merged.iter().zip(&self.windows).any(|(a, b)| {
                a.id != b.id || a.panes.len() != b.panes.len()
            });
        if changed || private_changed {
            self.active_win = self.active_win.min(merged.len().saturating_sub(1));
            self.windows = merged;
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

    pub fn update_service_list(&mut self, entries: Vec<ServiceEntry>) {
        self.service_list = entries;
    }

    pub fn update_worker_list(&mut self, sandboxes: Vec<WorkerEntry>, pool_summary: String) {
        self.worker_list = sandboxes;
        self.worker_pool_summary = pool_summary;
    }

    // =========================================================================
    // Key dispatch
    // =========================================================================

    pub fn handle_key(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        match self.mode {
            ShellMode::Normal                 => self.handle_normal(key),
            ShellMode::Fullscreen             => self.handle_fullscreen(key),
            ShellMode::ModelList              => self.handle_model_list(key),
            ShellMode::Settings               => self.handle_settings(key),
            ShellMode::StartMenu { selected } => self.handle_start_menu(key, selected),
            ShellMode::Console                => self.handle_console(key),
            ShellMode::ConversationPicker { .. } => self.handle_conversation_picker(key),
            ShellMode::ServiceManager { selected } => self.handle_service_manager(key, selected),
            ShellMode::WorkerManager { sandbox_sel, container_sel, show_containers, tab, image_sel, ref mut input_mode }
                => {
                    // Input dialog intercepts all keys when active.
                    if input_mode.is_some() {
                        return self.handle_input_dialog(key);
                    }
                    self.handle_worker_manager(key, sandbox_sel, container_sel, show_containers, tab, image_sel)
                }
        }
    }

    // ── Normal mode ──────────────────────────────────────────────────────────

    fn handle_normal(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        match key {
            KeyPress::CtrlSpace => {
                self.mode = ShellMode::StartMenu { selected: 0 };
                vec![ChromeOutput::Redraw]
            }
            KeyPress::Char(0x06) => {
                // Ctrl-F — toggle fullscreen (hide all chrome).
                self.mode = ShellMode::Fullscreen;
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
                self.push_toast(format!("Loading {model_ref}\u{2026}"), ToastLevel::Info);
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
                let model_ref = model.model_ref.clone();
                self.mode = ShellMode::Normal;
                return vec![ChromeOutput::Rpc(RpcRequest::ListConversations {
                    model_ref,
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
                vec![ChromeOutput::Redraw]
            }
        }
    }

    // ── StartMenu mode ───────────────────────────────────────────────────────

    fn handle_start_menu(&mut self, key: KeyPress, selected: usize) -> Vec<ChromeOutput> {
        let n = MENU_ITEMS.len();
        let chord: Option<usize> = match key {
            KeyPress::Char(b'n' | b'N') => Some(0),  // New
            KeyPress::Char(b'q' | b'Q') => Some(1),  // Close
            KeyPress::Tab               => Some(2),   // Cycle
            KeyPress::Char(b'm' | b'M') => Some(3),  // Models
            KeyPress::Char(b'l' | b'L') => Some(4),  // Log
            KeyPress::Char(b'w' | b'W') => Some(5),  // Workers
            #[cfg(feature = "experimental")]
            KeyPress::Char(b'v' | b'V') => Some(6),  // Services
            KeyPress::Char(b's' | b'S') => {
                #[cfg(feature = "experimental")]     { Some(7) }
                #[cfg(not(feature = "experimental"))] { Some(6) }
            }
            KeyPress::Char(b'f' | b'F') => {
                #[cfg(feature = "experimental")]     { Some(8) }
                #[cfg(not(feature = "experimental"))] { Some(7) }
            }
            KeyPress::Char(b'd' | b'D') => {
                #[cfg(feature = "experimental")]     { Some(9) }
                #[cfg(not(feature = "experimental"))] { Some(8) }
            }
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

        // Indices for items that shift when "Services" is present (experimental).
        #[cfg(feature = "experimental")]
        const SERVICES_IDX: usize = 6;
        #[cfg(feature = "experimental")]
        const SETTINGS_IDX: usize = 7;
        #[cfg(not(feature = "experimental"))]
        const SETTINGS_IDX: usize = 6;
        #[cfg(feature = "experimental")]
        const FULLSCREEN_IDX: usize = 8;
        #[cfg(not(feature = "experimental"))]
        const FULLSCREEN_IDX: usize = 7;
        #[cfg(feature = "experimental")]
        const DISCONNECT_IDX: usize = 9;
        #[cfg(not(feature = "experimental"))]
        const DISCONNECT_IDX: usize = 8;

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
                // Log (console)
                self.console_scroll = self.console_log.len().saturating_sub(1);
                self.mode = ShellMode::Console;
                vec![ChromeOutput::Redraw]
            }
            5 => {
                // Workers
                self.mode = ShellMode::WorkerManager {
                    sandbox_sel: 0, container_sel: 0, show_containers: false,
                    tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None,
                };
                vec![ChromeOutput::Redraw]
            }
            #[cfg(feature = "experimental")]
            SERVICES_IDX => {
                self.mode = ShellMode::ServiceManager { selected: 0 };
                vec![ChromeOutput::Redraw]
            }
            SETTINGS_IDX => {
                self.saved_style = self.bg.style;
                let i = ALL_STYLES.iter().position(|s| *s == self.bg.style).unwrap_or(0);
                self.settings_list = SelectList::new("Background", ALL_STYLES.to_vec())
                    .with_selected(i);
                self.preview_bg.style = self.bg.style;
                self.mode = ShellMode::Settings;
                vec![ChromeOutput::Redraw]
            }
            FULLSCREEN_IDX => {
                self.mode = ShellMode::Fullscreen;
                vec![ChromeOutput::Redraw]
            }
            DISCONNECT_IDX => vec![ChromeOutput::Rpc(RpcRequest::Quit)],
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
            KeyPress::Escape => {
                self.mode = ShellMode::Normal;
                vec![ChromeOutput::Redraw]
            }
            _ => vec![],
        }
    }

    // ── Fullscreen mode ───────────────────────────────────────────────────────

    fn handle_fullscreen(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        match key {
            KeyPress::Char(0x06) | KeyPress::Escape => {
                // Ctrl-F or Esc — exit fullscreen.
                self.mode = ShellMode::Normal;
                vec![ChromeOutput::Redraw]
            }
            key => {
                let bytes = keypress_to_bytes(key);
                if bytes.is_empty() {
                    vec![]
                } else {
                    let active = self.active_pane_id();
                    if self.private_panes.contains(&active) {
                        vec![ChromeOutput::RouteInput { app_id: active, data: bytes }]
                    } else {
                        let vid = self.viewer_id;
                        vec![ChromeOutput::Rpc(RpcRequest::SendInput { viewer_id: vid, data: bytes })]
                    }
                }
            }
        }
    }

    /// Open the conversation picker with the given entries.
    /// Called by the shell handler after loading the manifest.
    pub fn open_conversation_picker(
        &mut self,
        model_ref: String,
        conversations: Vec<ConversationPickerEntry>,
    ) {
        let mut items = vec![ConversationPickerEntry {
            uuid: "new".to_owned(),
            label: "[New conversation]".to_owned(),
            last_active: u64::MAX, // sort first
        }];
        items.extend(conversations);
        let list = SelectList::new(format!("{} — Conversations", model_ref), items);
        self.mode = ShellMode::ConversationPicker { model_ref, list };
    }

    // ── ConversationPicker mode ────────────────────────────────────────────

    fn handle_conversation_picker(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        let (model_ref, list) = match &mut self.mode {
            ShellMode::ConversationPicker { model_ref, list } => (model_ref.clone(), list),
            _ => return vec![],
        };

        // 'd' — delete selected conversation (stays in picker)
        if matches!(key, KeyPress::Char(b'd' | b'D')) {
            let idx = list.selected_index();
            if let Some(entry) = list.selected_item().cloned() {
                if entry.uuid != "new" {
                    let uuid = entry.uuid.clone();
                    let mr = model_ref.clone();
                    list.items_mut().remove(idx);
                    list.clamp_selected();
                    self.push_toast("Deleted conversation", ToastLevel::Info);
                    return vec![
                        ChromeOutput::Rpc(RpcRequest::DeleteConversation { uuid, model_ref: mr }),
                        ChromeOutput::Redraw,
                    ];
                }
            }
        }

        match list.handle_key(&key) {
            WidgetResult::Cancelled => {
                self.mode = ShellMode::Normal;
                vec![ChromeOutput::Redraw]
            }
            WidgetResult::Confirmed(entry) => {
                let resume = if entry.uuid == "new" { None } else { Some(entry.uuid) };
                let cols = self.cols;
                let rows = self.pane_rows;
                self.mode = ShellMode::Normal;
                vec![ChromeOutput::Rpc(RpcRequest::LocalPrivateChat {
                    model_ref,
                    cols,
                    rows,
                    resume_uuid: resume,
                })]
            }
            WidgetResult::Pending => {
                if matches!(key, KeyPress::Escape) {
                    self.mode = ShellMode::Normal;
                }
                vec![ChromeOutput::Redraw]
            }
        }
    }

    // ── ServiceManager mode ────────────────────────────────────────────────

    fn handle_service_manager(&mut self, key: KeyPress, selected: usize) -> Vec<ChromeOutput> {
        let n = self.service_list.len();
        match key {
            KeyPress::ArrowUp | KeyPress::Char(b'k') if n > 0 => {
                self.mode = ShellMode::ServiceManager {
                    selected: if selected == 0 { n - 1 } else { selected - 1 },
                };
                vec![ChromeOutput::Redraw]
            }
            KeyPress::ArrowDown | KeyPress::Char(b'j') if n > 0 => {
                self.mode = ShellMode::ServiceManager { selected: (selected + 1) % n };
                vec![ChromeOutput::Redraw]
            }
            KeyPress::Char(b's') => {
                if let Some(svc) = self.service_list.get(selected) {
                    let name = svc.name.clone();
                    return vec![ChromeOutput::Rpc(RpcRequest::ServiceStop { name })];
                }
                vec![]
            }
            KeyPress::Char(b't') => {
                if let Some(svc) = self.service_list.get(selected) {
                    let name = svc.name.clone();
                    return vec![ChromeOutput::Rpc(RpcRequest::ServiceStart { name })];
                }
                vec![]
            }
            KeyPress::Char(b'r') => {
                if let Some(svc) = self.service_list.get(selected) {
                    let name = svc.name.clone();
                    return vec![ChromeOutput::Rpc(RpcRequest::ServiceRestart { name })];
                }
                vec![]
            }
            KeyPress::Char(b'a') => {
                vec![ChromeOutput::Rpc(RpcRequest::ServiceStartAll)]
            }
            KeyPress::Char(b'S') => {
                vec![ChromeOutput::Rpc(RpcRequest::ServiceStopAll)]
            }
            KeyPress::Char(b'i') => {
                vec![ChromeOutput::Rpc(RpcRequest::ServiceInstall)]
            }
            KeyPress::Escape => {
                self.mode = ShellMode::Normal;
                vec![ChromeOutput::Redraw]
            }
            _ => vec![],
        }
    }

    // ── WorkerManager mode ──────────────────────────────────────────────────

    fn handle_worker_manager(
        &mut self, key: KeyPress,
        sandbox_sel: usize, container_sel: usize, show_containers: bool,
        tab: WorkerTab, image_sel: usize,
    ) -> Vec<ChromeOutput> {
        // Helper macro for constructing mode with current tab state.
        macro_rules! wm {
            ($ss:expr, $cs:expr, $sc:expr) => {
                ShellMode::WorkerManager {
                    sandbox_sel: $ss, container_sel: $cs, show_containers: $sc,
                    tab, image_sel, input_mode: None,
                }
            };
            ($ss:expr, $cs:expr, $sc:expr, tab: $t:expr) => {
                ShellMode::WorkerManager {
                    sandbox_sel: $ss, container_sel: $cs, show_containers: $sc,
                    tab: $t, image_sel, input_mode: None,
                }
            };
            ($ss:expr, $cs:expr, $sc:expr, tab: $t:expr, img: $is:expr) => {
                ShellMode::WorkerManager {
                    sandbox_sel: $ss, container_sel: $cs, show_containers: $sc,
                    tab: $t, image_sel: $is, input_mode: None,
                }
            };
        }

        // Tab key toggles between Sandboxes and Images
        if key == KeyPress::Tab {
            let new_tab = match tab {
                WorkerTab::Sandboxes => WorkerTab::Images,
                WorkerTab::Images => WorkerTab::Sandboxes,
            };
            self.mode = wm!(sandbox_sel, container_sel, false, tab: new_tab);
            // When switching to Images tab, request image list refresh
            if new_tab == WorkerTab::Images {
                return vec![ChromeOutput::Redraw, ChromeOutput::Rpc(RpcRequest::WorkerImageList)];
            }
            return vec![ChromeOutput::Redraw];
        }

        // Images tab handling
        if tab == WorkerTab::Images {
            let n = self.image_list.len();
            return match key {
                KeyPress::ArrowUp | KeyPress::Char(b'k') if n > 0 => {
                    let new_sel = if image_sel == 0 { n - 1 } else { image_sel - 1 };
                    self.mode = wm!(sandbox_sel, container_sel, false, tab: WorkerTab::Images, img: new_sel);
                    vec![ChromeOutput::Redraw]
                }
                KeyPress::ArrowDown | KeyPress::Char(b'j') if n > 0 => {
                    let new_sel = (image_sel + 1) % n;
                    self.mode = wm!(sandbox_sel, container_sel, false, tab: WorkerTab::Images, img: new_sel);
                    vec![ChromeOutput::Redraw]
                }
                // Pull image
                KeyPress::Char(b'p') => {
                    self.mode = ShellMode::WorkerManager {
                        sandbox_sel, container_sel, show_containers: false,
                        tab: WorkerTab::Images, image_sel,
                        input_mode: Some(InputDialog {
                            title: "Pull Image",
                            fields: vec![InputField::text("Image")],
                            focused: 0,
                        }),
                    };
                    vec![ChromeOutput::Redraw]
                }
                // Remove image
                KeyPress::Char(b'x') => {
                    if let Some(img) = self.image_list.get(image_sel) {
                        let image = img.repo_tag.clone();
                        return vec![ChromeOutput::Rpc(RpcRequest::WorkerImageRemove { image })];
                    }
                    vec![]
                }
                KeyPress::Escape => {
                    self.mode = ShellMode::Normal;
                    vec![ChromeOutput::Redraw]
                }
                _ => vec![],
            };
        }

        // Sandboxes tab
        let n = self.worker_list.len();
        match key {
            // Sandbox navigation (when not showing containers)
            KeyPress::ArrowUp | KeyPress::Char(b'k') if n > 0 && !show_containers => {
                self.mode = wm!(
                    if sandbox_sel == 0 { n - 1 } else { sandbox_sel - 1 },
                    0, false
                );
                vec![ChromeOutput::Redraw]
            }
            KeyPress::ArrowDown | KeyPress::Char(b'j') if n > 0 && !show_containers => {
                self.mode = wm!((sandbox_sel + 1) % n, 0, false);
                vec![ChromeOutput::Redraw]
            }
            // Enter container view
            KeyPress::Enter | KeyPress::ArrowRight if !show_containers => {
                if let Some(sb) = self.worker_list.get(sandbox_sel) {
                    if !sb.containers.is_empty() {
                        self.mode = wm!(sandbox_sel, 0, true);
                    }
                }
                vec![ChromeOutput::Redraw]
            }
            // Exit container view
            KeyPress::ArrowLeft if show_containers => {
                self.mode = wm!(sandbox_sel, 0, false);
                vec![ChromeOutput::Redraw]
            }
            // Container navigation
            KeyPress::ArrowUp | KeyPress::Char(b'k') if show_containers => {
                if let Some(sb) = self.worker_list.get(sandbox_sel) {
                    let cn = sb.containers.len();
                    let new_sel = if cn > 0 && container_sel > 0 { container_sel - 1 } else if cn > 0 { cn - 1 } else { 0 };
                    self.mode = wm!(sandbox_sel, new_sel, true);
                }
                vec![ChromeOutput::Redraw]
            }
            KeyPress::ArrowDown | KeyPress::Char(b'j') if show_containers => {
                if let Some(sb) = self.worker_list.get(sandbox_sel) {
                    let cn = sb.containers.len();
                    let new_sel = if cn > 0 { (container_sel + 1) % cn } else { 0 };
                    self.mode = wm!(sandbox_sel, new_sel, true);
                }
                vec![ChromeOutput::Redraw]
            }
            // Create sandbox
            KeyPress::Char(b'n') if !show_containers => {
                self.mode = ShellMode::WorkerManager {
                    sandbox_sel, container_sel, show_containers: false,
                    tab, image_sel,
                    input_mode: Some(InputDialog {
                        title: "Create Sandbox",
                        fields: vec![
                            InputField::text_with_default("Hostname", "sandbox"),
                            InputField::choices("Backend", vec!["kata".to_owned(), "runc".to_owned(), "gvisor".to_owned()]),
                            InputField::toggle("GPU"),
                        ],
                        focused: 0,
                    }),
                };
                vec![ChromeOutput::Redraw]
            }
            // Destroy sandbox
            KeyPress::Char(b'x') if !show_containers => {
                if let Some(sb) = self.worker_list.get(sandbox_sel) {
                    let sid = sb.full_id.clone();
                    return vec![ChromeOutput::Rpc(RpcRequest::WorkerDestroySandbox { sandbox_id: sid })];
                }
                vec![]
            }
            // Create container
            KeyPress::Char(b'n') if show_containers => {
                if self.worker_list.get(sandbox_sel).is_some() {
                    self.mode = ShellMode::WorkerManager {
                        sandbox_sel, container_sel, show_containers: true,
                        tab, image_sel,
                        input_mode: Some(InputDialog {
                            title: "Create Container",
                            fields: vec![
                                InputField::text_with_default("Image", "alpine:latest"),
                                InputField::text_with_default("Command", "/bin/sh"),
                                InputField::toggle("TTY"),
                            ],
                            focused: 0,
                        }),
                    };
                }
                vec![ChromeOutput::Redraw]
            }
            // Start container
            KeyPress::Char(b't') if show_containers => {
                if let Some(sb) = self.worker_list.get(sandbox_sel) {
                    if let Some(ctr) = sb.containers.get(container_sel) {
                        return vec![ChromeOutput::Rpc(RpcRequest::WorkerStartContainer {
                            container_id: ctr.full_id.clone(),
                        })];
                    }
                }
                vec![]
            }
            // Stop container
            KeyPress::Char(b's') if show_containers => {
                if let Some(sb) = self.worker_list.get(sandbox_sel) {
                    if let Some(ctr) = sb.containers.get(container_sel) {
                        return vec![ChromeOutput::Rpc(RpcRequest::WorkerStopContainer {
                            container_id: ctr.full_id.clone(),
                        })];
                    }
                }
                vec![]
            }
            // Remove container
            KeyPress::Char(b'x') if show_containers => {
                if let Some(sb) = self.worker_list.get(sandbox_sel) {
                    if let Some(ctr) = sb.containers.get(container_sel) {
                        return vec![ChromeOutput::Rpc(RpcRequest::WorkerRemoveContainer {
                            container_id: ctr.full_id.clone(),
                        })];
                    }
                }
                vec![]
            }
            // Attach to container
            KeyPress::Char(b'a') if show_containers => {
                if let Some(sb) = self.worker_list.get(sandbox_sel) {
                    if let Some(ctr) = sb.containers.get(container_sel) {
                        return vec![ChromeOutput::Rpc(RpcRequest::AttachContainer {
                            sandbox_id: sb.full_id.clone(),
                            container_id: ctr.full_id.clone(),
                            cols: self.cols,
                            rows: self.pane_rows,
                        })];
                    }
                }
                vec![]
            }
            // Exec in container
            KeyPress::Char(b'e') if show_containers => {
                if let Some(sb) = self.worker_list.get(sandbox_sel) {
                    if let Some(ctr) = sb.containers.get(container_sel) {
                        let sid = sb.full_id.clone();
                        let cid = ctr.full_id.clone();
                        return vec![ChromeOutput::Rpc(RpcRequest::WorkerExecSync {
                            sandbox_id: sid,
                            container_id: cid,
                            cmd: vec!["id".to_owned()],
                        })];
                    }
                }
                vec![]
            }
            KeyPress::Escape => {
                self.mode = ShellMode::Normal;
                vec![ChromeOutput::Redraw]
            }
            _ => vec![],
        }
    }

    // ── Input dialog handling ───────────────────────────────────────────────

    fn handle_input_dialog(&mut self, key: KeyPress) -> Vec<ChromeOutput> {
        // Extract current state to determine context.
        let (tab, sandbox_sel, container_sel, show_containers, image_sel) = match &self.mode {
            ShellMode::WorkerManager { tab, sandbox_sel, container_sel, show_containers, image_sel, .. } =>
                (*tab, *sandbox_sel, *container_sel, *show_containers, *image_sel),
            _ => return vec![],
        };

        match key {
            KeyPress::Escape => {
                // Cancel dialog
                self.mode = ShellMode::WorkerManager {
                    sandbox_sel, container_sel, show_containers,
                    tab, image_sel, input_mode: None,
                };
                vec![ChromeOutput::Redraw]
            }
            KeyPress::Enter => {
                // Submit dialog
                let dialog = match &mut self.mode {
                    ShellMode::WorkerManager { ref mut input_mode, .. } => input_mode.take(),
                    _ => None,
                };
                let Some(dialog) = dialog else { return vec![]; };
                let rpc = match dialog.title {
                    "Pull Image" => {
                        let image = dialog.fields[0].value();
                        if image.is_empty() { None } else { Some(RpcRequest::WorkerImagePull { image }) }
                    }
                    "Create Sandbox" => {
                        let hostname = dialog.fields[0].value();
                        let backend = dialog.fields[1].value();
                        let gpu = dialog.fields[2].toggle_val;
                        Some(RpcRequest::WorkerCreateSandbox { hostname, backend, gpu })
                    }
                    "Create Container" => {
                        let sandbox_id = self.worker_list.get(sandbox_sel)
                            .map(|sb| sb.full_id.clone())
                            .unwrap_or_default();
                        let image = dialog.fields[0].value();
                        let cmd_str = dialog.fields[1].value();
                        let cmd: Vec<String> = cmd_str.split_whitespace().map(String::from).collect();
                        let tty = dialog.fields[2].toggle_val;
                        Some(RpcRequest::WorkerCreateContainer { sandbox_id, image, cmd, tty })
                    }
                    _ => None,
                };
                self.mode = ShellMode::WorkerManager {
                    sandbox_sel, container_sel, show_containers,
                    tab, image_sel, input_mode: None,
                };
                if let Some(req) = rpc {
                    vec![ChromeOutput::Redraw, ChromeOutput::Rpc(req)]
                } else {
                    vec![ChromeOutput::Redraw]
                }
            }
            KeyPress::ArrowDown => {
                if let ShellMode::WorkerManager { input_mode: Some(ref mut d), .. } = self.mode {
                    if d.focused + 1 < d.fields.len() {
                        d.focused += 1;
                    }
                }
                vec![ChromeOutput::Redraw]
            }
            KeyPress::ArrowUp => {
                if let ShellMode::WorkerManager { input_mode: Some(ref mut d), .. } = self.mode {
                    if d.focused > 0 {
                        d.focused -= 1;
                    }
                }
                vec![ChromeOutput::Redraw]
            }
            KeyPress::Char(b' ') => {
                if let ShellMode::WorkerManager { input_mode: Some(ref mut d), .. } = self.mode {
                    if let Some(f) = d.fields.get_mut(d.focused) {
                        if f.is_toggle {
                            f.toggle_val = !f.toggle_val;
                        } else if let Some(ref opts) = f.choices {
                            if !opts.is_empty() {
                                f.choice_idx = (f.choice_idx + 1) % opts.len();
                            }
                        } else {
                            f.buf.insert(f.cursor, b' ');
                            f.cursor += 1;
                        }
                    }
                }
                vec![ChromeOutput::Redraw]
            }
            KeyPress::Backspace => {
                if let ShellMode::WorkerManager { input_mode: Some(ref mut d), .. } = self.mode {
                    if let Some(f) = d.fields.get_mut(d.focused) {
                        if !f.is_toggle && f.choices.is_none() && f.cursor > 0 {
                            f.cursor -= 1;
                            f.buf.remove(f.cursor);
                        }
                    }
                }
                vec![ChromeOutput::Redraw]
            }
            KeyPress::ArrowLeft => {
                if let ShellMode::WorkerManager { input_mode: Some(ref mut d), .. } = self.mode {
                    if let Some(f) = d.fields.get_mut(d.focused) {
                        if !f.is_toggle && f.choices.is_none() && f.cursor > 0 {
                            f.cursor -= 1;
                        }
                    }
                }
                vec![ChromeOutput::Redraw]
            }
            KeyPress::ArrowRight => {
                if let ShellMode::WorkerManager { input_mode: Some(ref mut d), .. } = self.mode {
                    if let Some(f) = d.fields.get_mut(d.focused) {
                        if !f.is_toggle && f.choices.is_none() && f.cursor < f.buf.len() {
                            f.cursor += 1;
                        }
                    }
                }
                vec![ChromeOutput::Redraw]
            }
            KeyPress::Char(c) => {
                if let ShellMode::WorkerManager { input_mode: Some(ref mut d), .. } = self.mode {
                    if let Some(f) = d.fields.get_mut(d.focused) {
                        if f.is_toggle {
                            f.toggle_val = !f.toggle_val;
                        } else if let Some(ref opts) = f.choices {
                            if !opts.is_empty() {
                                f.choice_idx = (f.choice_idx + 1) % opts.len();
                            }
                        } else {
                            f.buf.insert(f.cursor, c);
                            f.cursor += 1;
                        }
                    }
                }
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_chrome() -> ShellChrome {
        ShellChrome::new(120, 40, 1, 1, vec![], vec![])
    }

    #[cfg(feature = "experimental")]
    fn sample_services() -> Vec<ServiceEntry> {
        vec![
            ServiceEntry { name: "oai".into(), active: true, mode: ServiceMode::Systemd, pid: Some(1234) },
            ServiceEntry { name: "registry".into(), active: false, mode: ServiceMode::Stopped, pid: None },
            ServiceEntry { name: "model".into(), active: true, mode: ServiceMode::Daemon, pid: Some(5678) },
        ]
    }

    fn sample_workers() -> Vec<WorkerEntry> {
        vec![
            WorkerEntry {
                id: "abc123".into(), full_id: "abc12345-6789".into(),
                state: "Ready".into(), backend: "kata".into(),
                cpu_pct: Some(25), mem_mb: Some(512),
                containers: vec![
                    ContainerEntry {
                        id: "ctr1".into(), full_id: "ctr1-full".into(),
                        image: "hyprstream:latest".into(), state: "Running".into(),
                        cpu_pct: Some(10), mem_mb: Some(256),
                    },
                    ContainerEntry {
                        id: "ctr2".into(), full_id: "ctr2-full".into(),
                        image: "worker:v2".into(), state: "Stopped".into(),
                        cpu_pct: None, mem_mb: None,
                    },
                ],
            },
            WorkerEntry {
                id: "def456".into(), full_id: "def45678-abcd".into(),
                state: "Starting".into(), backend: "nspawn".into(),
                cpu_pct: None, mem_mb: None,
                containers: vec![],
            },
        ]
    }

    // ── F5 Service Manager ──────────────────────────────────────────────────

    #[cfg(feature = "experimental")]
    #[test]
    fn start_menu_v_opens_service_manager() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::StartMenu { selected: 0 };
        let out = chrome.handle_key(KeyPress::Char(b'v'));
        assert!(matches!(chrome.mode, ShellMode::ServiceManager { selected: 0 }));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Redraw)));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_escape_returns_to_normal() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };
        let out = chrome.handle_key(KeyPress::Escape);
        assert!(matches!(chrome.mode, ShellMode::Normal));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Redraw)));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_navigate_down() {
        let mut chrome = make_chrome();
        chrome.service_list = sample_services();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };

        chrome.handle_key(KeyPress::ArrowDown);
        assert!(matches!(chrome.mode, ShellMode::ServiceManager { selected: 1 }));

        chrome.handle_key(KeyPress::ArrowDown);
        assert!(matches!(chrome.mode, ShellMode::ServiceManager { selected: 2 }));

        // Wraps around
        chrome.handle_key(KeyPress::ArrowDown);
        assert!(matches!(chrome.mode, ShellMode::ServiceManager { selected: 0 }));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_navigate_up() {
        let mut chrome = make_chrome();
        chrome.service_list = sample_services();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };

        // Wraps to last
        chrome.handle_key(KeyPress::ArrowUp);
        assert!(matches!(chrome.mode, ShellMode::ServiceManager { selected: 2 }));

        chrome.handle_key(KeyPress::ArrowUp);
        assert!(matches!(chrome.mode, ShellMode::ServiceManager { selected: 1 }));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_vim_keys() {
        let mut chrome = make_chrome();
        chrome.service_list = sample_services();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };

        chrome.handle_key(KeyPress::Char(b'j'));
        assert!(matches!(chrome.mode, ShellMode::ServiceManager { selected: 1 }));

        chrome.handle_key(KeyPress::Char(b'k'));
        assert!(matches!(chrome.mode, ShellMode::ServiceManager { selected: 0 }));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_start_emits_rpc() {
        let mut chrome = make_chrome();
        chrome.service_list = sample_services();
        chrome.mode = ShellMode::ServiceManager { selected: 1 };

        let out = chrome.handle_key(KeyPress::Char(b't'));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Rpc(RpcRequest::ServiceStart { ref name }) if name == "registry")));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_stop_emits_rpc() {
        let mut chrome = make_chrome();
        chrome.service_list = sample_services();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };

        let out = chrome.handle_key(KeyPress::Char(b's'));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Rpc(RpcRequest::ServiceStop { ref name }) if name == "oai")));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_restart_emits_rpc() {
        let mut chrome = make_chrome();
        chrome.service_list = sample_services();
        chrome.mode = ShellMode::ServiceManager { selected: 2 };

        let out = chrome.handle_key(KeyPress::Char(b'r'));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Rpc(RpcRequest::ServiceRestart { ref name }) if name == "model")));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_start_all() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };

        let out = chrome.handle_key(KeyPress::Char(b'a'));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Rpc(RpcRequest::ServiceStartAll))));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_stop_all() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };

        let out = chrome.handle_key(KeyPress::Char(b'S'));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Rpc(RpcRequest::ServiceStopAll))));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_install() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };

        let out = chrome.handle_key(KeyPress::Char(b'i'));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Rpc(RpcRequest::ServiceInstall))));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_empty_list_no_crash() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };
        // Empty list — navigation should do nothing, no panic
        let out = chrome.handle_key(KeyPress::ArrowDown);
        assert!(out.is_empty());
        let out = chrome.handle_key(KeyPress::ArrowUp);
        assert!(out.is_empty());
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn service_manager_start_on_empty_list() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::ServiceManager { selected: 0 };
        // Start on empty list — no panic, no output
        let out = chrome.handle_key(KeyPress::Char(b't'));
        assert!(out.is_empty());
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn update_service_list_clamps_selection() {
        let mut chrome = make_chrome();
        chrome.service_list = sample_services();
        chrome.mode = ShellMode::ServiceManager { selected: 2 };

        // Shrink the list — selection should be clamped
        let short_list = vec![
            ServiceEntry { name: "oai".into(), active: true, mode: ServiceMode::Systemd, pid: None },
        ];
        chrome.update_service_list(short_list);
        // Selection clamping happens in Compositor::handle(ServiceList), not in chrome directly.
        // But verify the list was updated.
        assert_eq!(chrome.service_list.len(), 1);
    }

    // ── Worker Manager (via Ctrl-Space → W) ─────────────────────────────────

    #[test]
    fn start_menu_w_opens_worker_manager() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::StartMenu { selected: 0 };
        let out = chrome.handle_key(KeyPress::Char(b'w'));
        assert!(matches!(chrome.mode, ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: false, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None }));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Redraw)));
    }

    #[test]
    fn worker_manager_escape_returns_to_normal() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: false, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };
        chrome.handle_key(KeyPress::Escape);
        assert!(matches!(chrome.mode, ShellMode::Normal));
    }

    #[test]
    fn worker_manager_navigate_sandboxes() {
        let mut chrome = make_chrome();
        chrome.worker_list = sample_workers();
        chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: false, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };

        chrome.handle_key(KeyPress::ArrowDown);
        assert!(matches!(chrome.mode, ShellMode::WorkerManager { sandbox_sel: 1, .. }));

        chrome.handle_key(KeyPress::ArrowDown);
        assert!(matches!(chrome.mode, ShellMode::WorkerManager { sandbox_sel: 0, .. }));

        chrome.handle_key(KeyPress::ArrowUp);
        assert!(matches!(chrome.mode, ShellMode::WorkerManager { sandbox_sel: 1, .. }));
    }

    #[test]
    fn worker_manager_enter_container_view() {
        let mut chrome = make_chrome();
        chrome.worker_list = sample_workers();
        chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: false, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };

        // First sandbox has containers
        chrome.handle_key(KeyPress::Enter);
        assert!(matches!(chrome.mode, ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: true, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None }));
    }

    #[test]
    fn worker_manager_enter_no_containers() {
        let mut chrome = make_chrome();
        chrome.worker_list = sample_workers();
        chrome.mode = ShellMode::WorkerManager { sandbox_sel: 1, container_sel: 0, show_containers: false, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };

        // Second sandbox has no containers — should stay in sandbox view
        chrome.handle_key(KeyPress::Enter);
        assert!(matches!(chrome.mode, ShellMode::WorkerManager { show_containers: false, .. }));
    }

    #[test]
    fn worker_manager_navigate_containers() {
        let mut chrome = make_chrome();
        chrome.worker_list = sample_workers();
        chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: true, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };

        chrome.handle_key(KeyPress::ArrowDown);
        assert!(matches!(chrome.mode, ShellMode::WorkerManager { container_sel: 1, show_containers: true, .. }));

        // Wrap around
        chrome.handle_key(KeyPress::ArrowDown);
        assert!(matches!(chrome.mode, ShellMode::WorkerManager { container_sel: 0, show_containers: true, .. }));
    }

    #[test]
    fn worker_manager_back_from_containers() {
        let mut chrome = make_chrome();
        chrome.worker_list = sample_workers();
        chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 1, show_containers: true, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };

        chrome.handle_key(KeyPress::ArrowLeft);
        assert!(matches!(chrome.mode, ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: false, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None }));
    }

    #[test]
    fn worker_manager_destroy_sandbox() {
        let mut chrome = make_chrome();
        chrome.worker_list = sample_workers();
        chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: false, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };

        let out = chrome.handle_key(KeyPress::Char(b'x'));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Rpc(RpcRequest::WorkerDestroySandbox { ref sandbox_id }) if sandbox_id == "abc12345-6789")));
    }

    #[test]
    fn worker_manager_exec_in_container() {
        let mut chrome = make_chrome();
        chrome.worker_list = sample_workers();
        chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 1, show_containers: true, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };

        let out = chrome.handle_key(KeyPress::Char(b'e'));
        assert!(out.iter().any(|o| matches!(o,
            ChromeOutput::Rpc(RpcRequest::WorkerExecSync { ref sandbox_id, ref container_id, .. })
            if sandbox_id == "abc12345-6789" && container_id == "ctr2-full"
        )));
    }

    #[test]
    fn worker_manager_empty_list_no_crash() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: false, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };
        // Empty list — no panic
        chrome.handle_key(KeyPress::ArrowDown);
        chrome.handle_key(KeyPress::ArrowUp);
        chrome.handle_key(KeyPress::Enter);
        let out = chrome.handle_key(KeyPress::Char(b'x'));
        assert!(out.is_empty());
    }

    // ── Compositor ServiceList/WorkerList input ─────────────────────────────

    #[cfg(feature = "experimental")]
    #[test]
    fn compositor_service_list_clamps_selection() {
        let mut comp = crate::Compositor::new(120, 40, 1, 1, vec![], vec![]);
        comp.chrome.mode = ShellMode::ServiceManager { selected: 5 };

        let out = comp.handle(crate::CompositorInput::ServiceList(sample_services()));
        assert!(out.iter().any(|o| matches!(o, crate::CompositorOutput::Redraw)));
        if let ShellMode::ServiceManager { selected } = comp.chrome.mode {
            assert!(selected <= 2, "selection should be clamped to list len - 1");
        } else {
            panic!("mode should still be ServiceManager");
        }
    }

    #[test]
    fn compositor_worker_list_clamps_selection() {
        let mut comp = crate::Compositor::new(120, 40, 1, 1, vec![], vec![]);
        comp.chrome.mode = ShellMode::WorkerManager { sandbox_sel: 10, container_sel: 5, show_containers: true, tab: WorkerTab::Sandboxes, image_sel: 0, input_mode: None };

        let out = comp.handle(crate::CompositorInput::WorkerList {
            sandboxes: sample_workers(),
            pool_summary: "2 sandboxes".into(),
        });
        assert!(out.iter().any(|o| matches!(o, crate::CompositorOutput::Redraw)));
        if let ShellMode::WorkerManager { sandbox_sel, .. } = comp.chrome.mode {
            assert!(sandbox_sel <= 1, "sandbox_sel should be clamped");
        } else {
            panic!("mode should still be WorkerManager");
        }
    }

    // ── Start menu chord tests ───────────────────────────────────────────────

    #[test]
    fn start_menu_l_opens_console() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::StartMenu { selected: 0 };
        let out = chrome.handle_key(KeyPress::Char(b'l'));
        assert!(matches!(chrome.mode, ShellMode::Console));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Redraw)));
    }

    #[test]
    fn start_menu_f_enters_fullscreen() {
        let mut chrome = make_chrome();
        chrome.mode = ShellMode::StartMenu { selected: 0 };
        let out = chrome.handle_key(KeyPress::Char(b'f'));
        assert!(matches!(chrome.mode, ShellMode::Fullscreen));
        assert!(out.iter().any(|o| matches!(o, ChromeOutput::Redraw)));
    }

    #[test]
    fn f_keys_pass_through_in_normal() {
        let mut chrome = make_chrome();
        // F9 should pass through to the active pane as escape sequence bytes.
        let out = chrome.handle_key(KeyPress::F(9));
        assert!(
            out.iter().any(|o| matches!(o,
                ChromeOutput::Rpc(RpcRequest::SendInput { data, .. })
                    if *data == b"\x1b[20~"
            )),
            "F9 should produce SendInput with VT escape sequence"
        );
    }
}
