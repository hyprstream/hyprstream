//! Client-side state for the ShellClient TUI.
//!
//! `ShellClientState` holds the `avt::Vt` that is fed ANSI frames from
//! TuiService, plus the window list, model list, and keyboard-dispatch logic.
//! The event loop in `shell_handlers.rs` owns this state and calls
//! `feed_frame` / `handle_key` on each tick.

use std::path::PathBuf;

use avt::Vt;
use waxterm::input::KeyPress;
use waxterm::widgets::{SelectList, WidgetResult};

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
// Model entry (re-exported from hyprstream-tui for local use)
// ============================================================================

/// A model repo discovered in the registry directory.
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
}

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
    pub cols: u16,
    pub pane_rows: u16,
    pub session_id: u32,
    pub viewer_id: u32,
    pub model_client: crate::services::ModelZmqClient,
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
    ) -> Self {
        let model_list = SelectList::new("Models", models);
        let vt = Vt::builder()
            .size(cols as usize, pane_rows as usize)
            .build();
        Self {
            mode: ShellMode::Normal,
            vt,
            windows,
            active_win: 0,
            model_list,
            load_status: None,
            cols,
            pane_rows,
            session_id,
            viewer_id,
            model_client,
        }
    }

    /// Feed ANSI bytes from TuiService into the local Vt.
    pub fn feed_frame(&mut self, ansi: &[u8]) {
        self.vt.feed_str(&String::from_utf8_lossy(ansi));
    }

    /// Dispatch a keypress and return the resulting action.
    pub async fn handle_key(
        &mut self,
        key: KeyPress,
        client: &TuiClient,
    ) -> ShellAction {
        match self.mode {
            ShellMode::Normal => self.handle_normal(key, client).await,
            ShellMode::ModelList => self.handle_model_list(key, client).await,
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
                    if !self.windows.is_empty() {
                        self.active_win = self.active_win.min(self.windows.len() - 1);
                        // Focus the new active window server-side.
                        if let Some(w) = self.windows.get(self.active_win) {
                            let _ = focus_window_rpc(client, w.id).await;
                        }
                    }
                }
                ShellAction::Redraw
            }
            KeyPress::Tab => {
                // Cycle windows and tell the server which is active.
                if !self.windows.is_empty() {
                    self.active_win = (self.active_win + 1) % self.windows.len();
                    if let Some(w) = self.windows.get(self.active_win) {
                        let _ = focus_window_rpc(client, w.id).await;
                    }
                }
                ShellAction::Redraw
            }
            // Ctrl-B D — detach
            KeyPress::Char(0x02) => ShellAction::None,
            key => ShellAction::Forward(keypress_to_bytes(key)),
        }
    }

    // ── ModelList mode ───────────────────────────────────────────────────────

    async fn handle_model_list(&mut self, key: KeyPress, client: &TuiClient) -> ShellAction {
        // l — load selected model
        if matches!(key, KeyPress::Char(b'l' | b'L')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let model_ref = model.model_ref.clone();
                let loaded = self.model_client.load(&model_ref, None).await.is_ok();
                let idx = self.model_list.selected_index();
                if let Some(entry) = self.model_list.items_mut().get_mut(idx) {
                    entry.loaded = loaded;
                }
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

        // T / Enter — open terminal in selected model worktree
        let is_open = matches!(key, KeyPress::Char(b't' | b'T') | KeyPress::Enter);
        if is_open {
            if let Some(model) = self.model_list.selected_item().cloned() {
                let cwd = model.path.to_string_lossy().into_owned();
                let title = format!("{} shell", model.model_ref);
                if let Ok(win_info) = client.create_window().await {
                    let pane_id = win_info.active_pane_id;
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
                let cwd = model.path.to_string_lossy().into_owned();
                let title = format!("{} shell", model.model_ref);
                if let Ok(win_info) = client.create_window().await {
                    let pane_id = win_info.active_pane_id;
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
}

// ============================================================================
// Manual RPC helpers (spawnShell / closePane)
//
// The generated TuiClient won't have typed methods for the new schema variants
// until after `cargo build` regenerates CGR metadata. We build the Cap'n Proto
// messages manually here — same pattern as `handle_tui_list` / `sendInput`.
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
///
/// Returns all bytes enqueued since the last poll, concatenated.
/// Empty slice means nothing was queued.  Used instead of ZMQ SUB in embedded
/// mode to avoid a second ZMQ context and libzmq signaler assertion bugs.
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
        KeyPress::F(_)  => vec![],
    }
}
