//! Client-side state for the ShellClient TUI.
//!
//! `ShellClientState` holds the `avt::Vt` that is fed ANSI frames from
//! TuiService, plus the window list, model list, settings, and keyboard-
//! dispatch logic.  The event loop in `shell_handlers.rs` owns this state and
//! calls `feed_frame` / `handle_key` on each tick.

use std::path::PathBuf;

use avt::Vt;
use waxterm::input::KeyPress;
use waxterm::widgets::{SelectList, WidgetResult};

use hyprstream_tui::background::{
    BackgroundState, BackgroundStyle, ALL_STYLES, PREVIEW_H, PREVIEW_W,
};

use crate::services::generated::tui_client::TuiClient;

// ============================================================================
// Window summary
// ============================================================================

/// Lightweight window descriptor from TuiService.
#[derive(Clone, Debug)]
pub struct WindowSummary {
    pub id: u32,
    pub name: String,
    pub active_pane_id: u32,
    pub panes: Vec<PaneSummary>,
}

/// Lightweight pane descriptor.
#[derive(Clone, Debug)]
pub struct PaneSummary {
    pub id: u32,
    pub cols: u16,
    pub rows: u16,
}

// ============================================================================
// Model entry
// ============================================================================

/// A model repo discovered via the registry RPC.
#[derive(Clone)]
pub struct ModelEntry {
    pub model_ref: String,  // "name:branch" — used as display label and load key
    pub path: PathBuf,
    pub loaded: bool,
}

impl std::fmt::Display for ModelEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.loaded {
            write!(f, "{} [loaded]", self.model_ref)
        } else {
            write!(f, "{}", self.model_ref)
        }
    }
}

// ============================================================================
// Mode + Action
// ============================================================================

pub enum ShellMode {
    /// Keys forwarded to the active shell via sendInput.
    Normal,
    /// Model-list modal is open.
    ModelList,
    /// Settings modal is open.
    Settings,
    /// Start-menu popup (Ctrl+Space).
    StartMenu { selected: usize },
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

/// Result of handling a key in the event loop.
pub enum ShellAction {
    /// Do nothing.
    None,
    /// Re-render the chrome.
    Redraw,
    /// Forward these raw bytes to TuiService via sendInput.
    Forward(Vec<u8>),
    /// Exit the ShellClient.
    Quit,
}

// ============================================================================
// ShellClientState
// ============================================================================

pub struct ShellClientState {
    pub mode: ShellMode,
    /// VTE state fed from TuiService ANSI frames.
    pub vt: Vt,
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
    pub cols: u16,
    pub pane_rows: u16,
    pub session_id: u32,
    pub viewer_id: u32,
    pub model_client: crate::services::ModelZmqClient,
    /// Send `(model_ref, loaded)` updates back to the event loop in
    /// `shell_handlers.rs` once a background polling task confirms load.
    pub model_status_tx: tokio::sync::mpsc::Sender<(String, bool)>,
}

impl ShellClientState {
    pub fn new(
        cols: u16,
        pane_rows: u16,
        session_id: u32,
        viewer_id: u32,
        windows: Vec<WindowSummary>,
        models: Vec<ModelEntry>,
        model_client: crate::services::ModelZmqClient,
        model_status_tx: tokio::sync::mpsc::Sender<(String, bool)>,
    ) -> Self {
        let model_list = SelectList::new("Models", models);
        let vt = Vt::builder()
            .size(cols as usize, pane_rows as usize)
            .build();

        let default_style = BackgroundStyle::Stars;
        let settings_list = SelectList::new("Background", ALL_STYLES.to_vec())
            .with_selected(1); // Stars is index 1

        let mut bg = BackgroundState::new(default_style);
        // Initialise immediately so the first rendered frame isn't blank.
        bg.tick(cols, pane_rows);

        Self {
            mode: ShellMode::Normal,
            vt,
            windows,
            active_win: 0,
            model_list,
            load_status: None,
            bg,
            preview_bg: BackgroundState::new(default_style),
            settings_list,
            saved_style: default_style,
            cols,
            pane_rows,
            session_id,
            viewer_id,
            model_client,
            model_status_tx,
        }
    }

    /// Feed ANSI bytes from TuiService into the local Vt.
    pub fn feed_frame(&mut self, ansi: &[u8]) {
        self.vt.feed_str(&String::from_utf8_lossy(ansi));
    }

    /// Tick background animations.  Returns true when a redraw is needed.
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

    /// Dispatch a keypress and return the resulting action.
    pub async fn handle_key(
        &mut self,
        key: KeyPress,
        client: &TuiClient,
    ) -> ShellAction {
        match self.mode {
            ShellMode::Normal                 => self.handle_normal(key, client).await,
            ShellMode::ModelList              => self.handle_model_list(key, client).await,
            ShellMode::Settings               => self.handle_settings(key),
            ShellMode::StartMenu { selected } => self.handle_start_menu(key, selected, client).await,
        }
    }

    // ── Normal mode ─────────────────────────────────────────────────────────

    async fn handle_normal(&mut self, key: KeyPress, client: &TuiClient) -> ShellAction {
        match key {
            KeyPress::F(12) => ShellAction::Quit,
            KeyPress::F(10) => {
                self.mode = ShellMode::ModelList;
                ShellAction::Redraw
            }
            KeyPress::F(11) => {
                self.saved_style = self.bg.style;
                let idx = ALL_STYLES
                    .iter()
                    .position(|s| *s == self.bg.style)
                    .unwrap_or(0);
                self.settings_list = SelectList::new("Background", ALL_STYLES.to_vec())
                    .with_selected(idx);
                self.preview_bg.style = self.bg.style;
                self.mode = ShellMode::Settings;
                ShellAction::Redraw
            }
            KeyPress::F(7) => {
                // Create new window + spawn shell, then focus it server-side.
                if let Ok(win_info) = client.create_window().await {
                    let pane_id = win_info.active_pane_id;
                    let _ = spawn_shell_rpc(client, self.session_id, pane_id, "").await;
                    let _ = focus_window_rpc(client, win_info.id).await;
                    self.windows.push(WindowSummary {
                        id: win_info.id,
                        name: win_info.name,
                        active_pane_id: win_info.active_pane_id,
                        panes: win_info
                            .panes
                            .iter()
                            .map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows })
                            .collect(),
                    });
                    self.active_win = self.windows.len() - 1;
                }
                ShellAction::Redraw
            }
            KeyPress::F(8) => {
                // Close the active window.
                if let Some(win) = self.windows.get(self.active_win) {
                    let win_id = win.id;
                    let _ = close_window_rpc(client, win_id).await;
                    self.windows.remove(self.active_win);
                    if self.windows.is_empty() {
                        // Clear stale VT content so the background shows through.
                        self.vt = Vt::builder()
                            .size(self.cols as usize, self.pane_rows as usize)
                            .build();
                    } else {
                        self.active_win = self.active_win.min(self.windows.len() - 1);
                        if let Some(w) = self.windows.get(self.active_win) {
                            let _ = focus_window_rpc(client, w.id).await;
                        }
                    }
                }
                ShellAction::Redraw
            }
            KeyPress::CtrlSpace => {
                self.mode = ShellMode::StartMenu { selected: 0 };
                ShellAction::Redraw
            }
            // Ctrl-B D — detach
            KeyPress::Char(0x02) => ShellAction::None,
            key => ShellAction::Forward(keypress_to_bytes(key)),
        }
    }

    // ── ModelList mode ───────────────────────────────────────────────────────

    async fn handle_model_list(&mut self, key: KeyPress, client: &TuiClient) -> ShellAction {
        // l — submit load request; poll status in background until confirmed
        if matches!(key, KeyPress::Char(b'l' | b'L')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let model_ref = model.model_ref.clone();
                // The ModelService returns "accepted" immediately (Continuation
                // pattern).  is_ok() only confirms the RPC was received, NOT
                // that the model is loaded.  We spawn a polling task instead.
                let _ = self.model_client.load(&model_ref, None).await;
                self.load_status = Some(format!("Loading {}…", model_ref));
                // Do NOT mark loaded=true yet.

                let poll_client = self.model_client.clone();
                let poll_mr    = model_ref.clone();
                let tx         = self.model_status_tx.clone();
                tokio::spawn(async move {
                    for _ in 0..60u32 {  // max ~2 minutes
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        match poll_client.status(&poll_mr).await {
                            Ok(entries) if entries.iter().any(|e| e.status == "loaded") => {
                                let _ = tx.send((poll_mr, true)).await;
                                return;
                            }
                            _ => {}
                        }
                    }
                    // Timed out — clear "Loading…" without marking loaded.
                    let _ = tx.send((poll_mr, false)).await;
                });

                return ShellAction::Redraw;
            }
        }
        // u — unload selected model
        if matches!(key, KeyPress::Char(b'u' | b'U')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let model_ref = model.model_ref.clone();
                let ok = self.model_client.unload(&model_ref).await.is_ok();
                if ok {
                    let idx = self.model_list.selected_index();
                    if let Some(entry) = self.model_list.items_mut().get_mut(idx) {
                        entry.loaded = false;
                    }
                }
                return ShellAction::Redraw;
            }
        }

        // Enter / C — spawn ChatApp inference pane for selected model (default action)
        let is_chat = matches!(key, KeyPress::Enter | KeyPress::Char(b'c' | b'C'));
        if is_chat {
            if let Some(model) = self.model_list.selected_item().cloned() {
                if let Ok(win_info) = client.create_window().await {
                    let pane_id = win_info.active_pane_id;
                    let _ = focus_window_rpc(client, win_info.id).await;
                    let _ = spawn_chat_app_rpc(
                        client,
                        self.session_id,
                        &model.model_ref,
                        self.cols,
                        self.pane_rows,
                        pane_id,
                    )
                    .await;
                    self.windows.push(WindowSummary {
                        id: win_info.id,
                        name: format!("{} chat", model.model_ref),
                        active_pane_id: win_info.active_pane_id,
                        panes: win_info
                            .panes
                            .iter()
                            .map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows })
                            .collect(),
                    });
                    self.active_win = self.windows.len() - 1;
                }
                self.mode = ShellMode::Normal;
                return ShellAction::Redraw;
            }
        }

        // T — open terminal scoped to the model's git worktree directory
        if matches!(key, KeyPress::Char(b't' | b'T')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let cwd = model.path.to_string_lossy().into_owned();
                let title = format!("{} shell", model.model_ref);
                if let Ok(win_info) = client.create_window().await {
                    let pane_id = win_info.active_pane_id;
                    let _ = focus_window_rpc(client, win_info.id).await;
                    let _ = spawn_shell_rpc(client, self.session_id, pane_id, &cwd).await;
                    self.windows.push(WindowSummary {
                        id: win_info.id,
                        name: title,
                        active_pane_id: win_info.active_pane_id,
                        panes: win_info
                            .panes
                            .iter()
                            .map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows })
                            .collect(),
                    });
                    self.active_win = self.windows.len() - 1;
                }
                self.mode = ShellMode::Normal;
                return ShellAction::Redraw;
            }
        }

        match self.model_list.handle_key(&key) {
            WidgetResult::Cancelled => {
                self.mode = ShellMode::Normal;
                ShellAction::Redraw
            }
            WidgetResult::Confirmed(model) => {
                // Enter on the select list widget → chat (same as is_chat above, but
                // handle_key consumes Enter before we reach the is_chat arm when focus
                // is inside the widget, so mirror the action here).
                if let Ok(win_info) = client.create_window().await {
                    let pane_id = win_info.active_pane_id;
                    let _ = focus_window_rpc(client, win_info.id).await;
                    let _ = spawn_chat_app_rpc(
                        client,
                        self.session_id,
                        &model.model_ref,
                        self.cols,
                        self.pane_rows,
                        pane_id,
                    )
                    .await;
                    self.windows.push(WindowSummary {
                        id: win_info.id,
                        name: format!("{} chat", model.model_ref),
                        active_pane_id: win_info.active_pane_id,
                        panes: win_info
                            .panes
                            .iter()
                            .map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows })
                            .collect(),
                    });
                    self.active_win = self.windows.len() - 1;
                }
                self.mode = ShellMode::Normal;
                ShellAction::Redraw
            }
            WidgetResult::Pending => {
                if matches!(key, KeyPress::F(10)) {
                    self.mode = ShellMode::Normal;
                }
                ShellAction::Redraw
            }
        }
    }

    // ── Start-menu mode ──────────────────────────────────────────────────────

    async fn handle_start_menu(
        &mut self,
        key: KeyPress,
        selected: usize,
        client: &TuiClient,
    ) -> ShellAction {
        let n = MENU_ITEMS.len();
        // Chord shortcuts execute immediately.
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
            return self.execute_menu_action(idx, client).await;
        }
        match key {
            KeyPress::Escape | KeyPress::CtrlSpace => { self.mode = ShellMode::Normal; }
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
                return self.execute_menu_action(selected, client).await;
            }
            _ => {}
        }
        ShellAction::Redraw
    }

    async fn execute_menu_action(&mut self, idx: usize, client: &TuiClient) -> ShellAction {
        match idx {
            0 => {
                // New window
                if let Ok(win_info) = client.create_window().await {
                    let pane_id = win_info.active_pane_id;
                    let _ = spawn_shell_rpc(client, self.session_id, pane_id, "").await;
                    let _ = focus_window_rpc(client, win_info.id).await;
                    self.windows.push(WindowSummary {
                        id: win_info.id, name: win_info.name,
                        active_pane_id: win_info.active_pane_id,
                        panes: win_info.panes.iter()
                            .map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows })
                            .collect(),
                    });
                    self.active_win = self.windows.len() - 1;
                }
            }
            1 => {
                // Close active window
                if let Some(win) = self.windows.get(self.active_win) {
                    let win_id = win.id;
                    let _ = close_window_rpc(client, win_id).await;
                    self.windows.remove(self.active_win);
                    if self.windows.is_empty() {
                        self.vt = avt::Vt::builder()
                            .size(self.cols as usize, self.pane_rows as usize)
                            .build();
                    } else {
                        self.active_win = self.active_win.min(self.windows.len() - 1);
                        if let Some(w) = self.windows.get(self.active_win) {
                            let _ = focus_window_rpc(client, w.id).await;
                        }
                    }
                }
            }
            2 => {
                // Cycle windows
                if !self.windows.is_empty() {
                    self.active_win = (self.active_win + 1) % self.windows.len();
                    if let Some(w) = self.windows.get(self.active_win) {
                        let _ = focus_window_rpc(client, w.id).await;
                    }
                }
            }
            3 => { self.mode = ShellMode::ModelList; }
            4 => {
                self.saved_style = self.bg.style;
                let idx = hyprstream_tui::background::ALL_STYLES
                    .iter().position(|s| *s == self.bg.style).unwrap_or(0);
                self.settings_list = waxterm::widgets::SelectList::new(
                    "Background",
                    hyprstream_tui::background::ALL_STYLES.to_vec(),
                ).with_selected(idx);
                self.preview_bg.style = self.bg.style;
                self.mode = ShellMode::Settings;
            }
            5 => return ShellAction::Quit,
            _ => {}
        }
        ShellAction::Redraw
    }

    // ── Settings mode ────────────────────────────────────────────────────────

    fn handle_settings(&mut self, key: KeyPress) -> ShellAction {
        match self.settings_list.handle_key(&key) {
            WidgetResult::Cancelled => {
                self.bg.style         = self.saved_style;
                self.preview_bg.style = self.saved_style;
                self.mode             = ShellMode::Normal;
                ShellAction::Redraw
            }
            WidgetResult::Confirmed(style) => {
                self.bg.style         = style;
                self.preview_bg.style = style;
                self.mode             = ShellMode::Normal;
                ShellAction::Redraw
            }
            WidgetResult::Pending => {
                // Live preview: follow the highlighted list entry.
                if let Some(&style) = self.settings_list.selected_item() {
                    self.bg.style         = style;
                    self.preview_bg.style = style;
                }
                ShellAction::Redraw
            }
        }
    }
}

// ============================================================================
// Manual RPC helpers (spawnShell / closePane)
// ============================================================================

/// Send a spawnShell RPC to TuiService.
pub async fn spawn_shell_rpc(
    client: &TuiClient,
    session_id: u32,
    pane_id: u32,
    cwd: &str,
) -> anyhow::Result<()> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        let mut spawn = req.init_spawn_shell();
        spawn.set_session_id(session_id);
        spawn.set_pane_id(pane_id);
        spawn.set_cwd(cwd);
    })?;
    client.call(payload).await?;
    Ok(())
}

/// Poll for stdin bytes queued for this viewer (pollStdin RPC).
pub async fn poll_stdin_rpc(client: &TuiClient, viewer_id: u32) -> anyhow::Result<Vec<u8>> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_poll_stdin(viewer_id);
    })?;
    let response_bytes = client.call(payload).await?;
    let reader = capnp::serialize::read_message_from_flat_slice(
        &mut &response_bytes[..],
        capnp::message::ReaderOptions::default(),
    )?;
    let response = reader.get_root::<crate::tui_capnp::tui_response::Reader<'_>>()?;
    match response.which()? {
        crate::tui_capnp::tui_response::Which::PollStdinResult(data) => Ok(data?.to_vec()),
        crate::tui_capnp::tui_response::Which::Error(e) => {
            let msg = e?.get_message()?.to_str().unwrap_or("unknown").to_owned();
            Err(anyhow::anyhow!("pollStdin error: {}", msg))
        }
        _ => Ok(vec![]),
    }
}

/// Tell TuiService which window is active for this session.
pub async fn focus_window_rpc(client: &TuiClient, window_id: u32) -> anyhow::Result<()> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_focus_window(window_id);
    })?;
    client.call(payload).await?;
    Ok(())
}

/// Send a spawnChatApp RPC to TuiService.
pub async fn spawn_chat_app_rpc(
    client: &TuiClient,
    session_id: u32,
    model_ref: &str,
    cols: u16,
    rows: u16,
    pane_id: u32,
) -> anyhow::Result<()> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        let mut chat = req.init_spawn_chat_app();
        chat.set_session_id(session_id);
        chat.set_model_ref(model_ref);
        chat.set_cols(cols);
        chat.set_rows(rows);
        chat.set_pane_id(pane_id);
    })?;
    client.call(payload).await?;
    Ok(())
}

/// Close a window by ID via RPC.
pub async fn close_window_rpc(client: &TuiClient, window_id: u32) -> anyhow::Result<()> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_close_window(window_id);
    })?;
    client.call(payload).await?;
    Ok(())
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
        KeyPress::F(_) | KeyPress::CtrlSpace  => vec![],  // consumed by chrome or unmapped; never forward to PTY
    }
}
