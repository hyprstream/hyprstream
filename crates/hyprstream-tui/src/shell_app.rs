//! ShellApp — TUI shell with model list modal, settings, and background.
//!
//! Each "window" is a PTY-backed shell process. Output is VTE-parsed by
//! `avt::Vt`. The model list modal (F10) lets users open shells scoped to a
//! model worktree directory. The settings modal (F11) lets users choose a
//! background style. When no windows are open the animated background fills
//! the pane area.

#![cfg(not(target_os = "wasi"))]

use std::path::PathBuf;

use avt::Vt;
use waxterm::app::TerminalApp;
use waxterm::input::KeyPress;
use waxterm::widgets::{SelectList, WidgetResult};

use crate::background::{BackgroundState, BackgroundStyle, ALL_STYLES, PREVIEW_H, PREVIEW_W};
use crate::shell::{kill_shell, spawn_shell, ShellInput, ShellWindow};

type LoadFn = Box<dyn Fn(&str, ModelStatusSender) + Send>;

// ============================================================================
// Model discovery
// ============================================================================

/// A model repo discovered in the registry directory.
#[derive(Clone)]
pub struct ModelEntry {
    pub model_ref: String,  // "name:branch"
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

/// Scan `registry` for git repos (directories containing `.git` or `HEAD`).
pub fn discover_models(registry: &std::path::Path) -> Vec<ModelEntry> {
    let Ok(entries) = std::fs::read_dir(registry) else {
        return vec![];
    };
    let mut models: Vec<ModelEntry> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join(".git").exists() || e.path().join("HEAD").exists())
        .map(|e| ModelEntry {
            model_ref: e.file_name().to_string_lossy().into_owned(),
            path: e.path(),
            loaded: false,
        })
        .collect();
    models.sort_by(|a, b| a.model_ref.cmp(&b.model_ref));
    models
}

// ============================================================================
// Pane window
// ============================================================================

/// A running shell attached to an avt virtual terminal.
pub struct PaneWindow {
    pub title: String,
    /// VTE state (avt).
    pub vt: Vt,
    /// PTY channels.
    pub shell: ShellWindow,
}

impl PaneWindow {
    pub fn new(title: String, cwd: Option<PathBuf>, cols: u16, rows: u16) -> std::io::Result<Self> {
        let shell = spawn_shell(cwd, cols, rows)?;
        let vt = Vt::builder()
            .size(cols as usize, rows as usize)
            .build();
        Ok(Self { title, vt, shell })
    }

    /// Drain stdout_rx into the VTE parser.
    ///
    /// Returns `(got_data, is_dead)`:
    /// - `got_data` — true if any bytes arrived (caller should redraw)
    /// - `is_dead`  — true if the channel disconnected (PTY process exited)
    pub fn drain(&mut self) -> (bool, bool) {
        let mut got = false;
        loop {
            match self.shell.stdout_rx.try_recv() {
                Ok(data) => {
                    let s = String::from_utf8_lossy(&data);
                    self.vt.feed_str(&s);
                    got = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return (got, false),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return (got, true),
            }
        }
    }

    /// Send raw bytes to the shell stdin.
    pub fn send_bytes(&self, data: Vec<u8>) {
        let _ = self.shell.input_tx.send(ShellInput::Bytes(data));
    }

    /// Resize the PTY and VT.
    pub fn resize(&mut self, cols: u16, rows: u16) {
        self.vt.feed_str(&format!(
            "\x1b[8;{rows};{cols}t"
        ));
        let _ = self.shell.input_tx.send(ShellInput::Resize { cols, rows });
    }
}

impl Drop for PaneWindow {
    fn drop(&mut self) {
        kill_shell(self.shell.pid);
    }
}

// ============================================================================
// App mode
// ============================================================================

pub enum ShellMode {
    /// Keys forwarded directly to the active shell.
    Normal,
    /// Model list modal is open.
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

// ============================================================================
// Command
// ============================================================================

pub enum ShellCmd {
    Key(KeyPress),
}

impl From<KeyPress> for ShellCmd {
    fn from(k: KeyPress) -> Self {
        ShellCmd::Key(k)
    }
}

// ============================================================================
// ShellApp
// ============================================================================

/// Sender half for model status updates from background load-polling threads.
pub type ModelStatusSender = std::sync::mpsc::Sender<(String, bool)>;

pub struct ShellApp {
    pub mode: ShellMode,
    pub windows: Vec<PaneWindow>,
    pub active: usize,
    pub model_list: SelectList<ModelEntry>,
    /// Background animation state (full terminal area).
    pub bg: BackgroundState,
    /// Background animation state for the settings preview box.
    pub preview_bg: BackgroundState,
    /// Settings drop-down (one entry per BackgroundStyle).
    pub settings_list: SelectList<BackgroundStyle>,
    /// Background style saved when settings opens — restored on cancel.
    saved_style: BackgroundStyle,
    /// Shown in status bar while a model is loading.
    pub load_status: Option<String>,
    /// Receives `(model_ref, loaded)` updates from background load-polling threads.
    status_rx: std::sync::mpsc::Receiver<(String, bool)>,
    /// Cloned and passed into `load_fn` on each call.
    status_tx: ModelStatusSender,
    pub cols: u16,
    pub rows: u16,
    quit: bool,
    /// Ctrl-B prefix was pressed; next key selects action.
    ctrl_b_pending: bool,
    /// Submit a load request.  Receives a `ModelStatusSender` to push the final
    /// `(model_ref, true/false)` result once the model is actually loaded.
    load_fn: LoadFn,
    unload_fn: Box<dyn Fn(&str) -> bool + Send>,
}

impl ShellApp {
    pub fn new(
        models: Vec<ModelEntry>,
        cols: u16,
        rows: u16,
        load_fn: LoadFn,
        unload_fn: Box<dyn Fn(&str) -> bool + Send>,
    ) -> Self {
        let pane_rows = rows.saturating_sub(2);
        let initial = PaneWindow::new("shell".to_owned(), None, cols, pane_rows)
            .unwrap_or_else(|e| panic!("Failed to spawn shell: {e}"));

        let model_list = SelectList::new("Models", models);

        let default_style = BackgroundStyle::Stars;
        let settings_list = SelectList::new(
            "Background",
            ALL_STYLES.to_vec(),
        ).with_selected(1); // Stars is index 1

        let (status_tx, status_rx) = std::sync::mpsc::channel();

        Self {
            mode: ShellMode::Normal,
            windows: vec![initial],
            active: 0,
            model_list,
            bg: BackgroundState::new(default_style),
            preview_bg: BackgroundState::new(default_style),
            settings_list,
            saved_style: default_style,
            load_status: None,
            status_rx,
            status_tx,
            cols,
            rows,
            quit: false,
            ctrl_b_pending: false,
            load_fn,
            unload_fn,
        }
    }

    pub fn pane_rows(&self) -> u16 {
        self.rows.saturating_sub(2)
    }

    fn open_terminal(&mut self, cwd: Option<PathBuf>, title: String) {
        let rows = self.pane_rows();
        match PaneWindow::new(title, cwd, self.cols, rows) {
            Ok(win) => {
                self.windows.push(win);
                self.active = self.windows.len() - 1;
            }
            Err(e) => {
                eprintln!("hyprstream-tui: open_terminal failed: {e}");
            }
        }
    }

    fn close_active(&mut self) {
        if self.windows.is_empty() {
            return;
        }
        self.windows.remove(self.active);
        if !self.windows.is_empty() {
            self.active = self.active.min(self.windows.len() - 1);
        }
    }

    fn cycle_window(&mut self) {
        if !self.windows.is_empty() {
            self.active = (self.active + 1) % self.windows.len();
        }
    }

    fn handle_normal(&mut self, key: KeyPress) -> bool {
        // Ctrl-B prefix: next key selects action.
        if self.ctrl_b_pending {
            self.ctrl_b_pending = false;
            match key {
                KeyPress::Char(b'q') | KeyPress::Char(b'd') => {
                    self.quit = true;
                    return true;
                }
                _ => {
                    // Not a multiplexer key — forward Ctrl-B + this key to PTY.
                    if let Some(win) = self.windows.get(self.active) {
                        win.send_bytes(vec![0x02]);
                        let bytes = keypress_to_bytes(key);
                        if !bytes.is_empty() {
                            win.send_bytes(bytes);
                        }
                    }
                    return false;
                }
            }
        }
        match key {
            KeyPress::CtrlSpace => {
                self.mode = ShellMode::StartMenu { selected: 0 };
                true
            }
            KeyPress::Char(0x02) => {
                // Ctrl-B prefix — wait for next key.
                self.ctrl_b_pending = true;
                true
            }
            key => {
                let bytes = keypress_to_bytes(key);
                if let Some(win) = self.windows.get(self.active) {
                    if !bytes.is_empty() {
                        win.send_bytes(bytes);
                    }
                }
                false
            }
        }
    }

    fn handle_model_list(&mut self, key: KeyPress) -> bool {
        if matches!(key, KeyPress::Char(b'l' | b'L')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                // Submit the load request; the closure polls status and sends
                // (model_ref, true) back via status_tx when actually loaded.
                (self.load_fn)(&model.model_ref, self.status_tx.clone());
                self.load_status = Some(format!("Loading {}…", model.model_ref));
                // Do NOT mark loaded=true here — wait for status_rx confirmation.
                return true;
            }
        }
        if matches!(key, KeyPress::Char(b'u' | b'U')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                // Unload is synchronous — the RPC blocks until unloaded.
                let ok = (self.unload_fn)(&model.model_ref);
                if ok {
                    let idx = self.model_list.selected_index();
                    if let Some(entry) = self.model_list.items_mut().get_mut(idx) {
                        entry.loaded = false;
                    }
                }
                return true;
            }
        }

        // T / Enter open a terminal scoped to the model worktree.
        // Only available on platforms with filesystem sandbox support (Linux, OpenBSD).
        // On other platforms the key is a no-op; a Worker-based approach will be added later.
        #[cfg(any(target_os = "linux", target_os = "openbsd"))]
        if matches!(key, KeyPress::Char(b't' | b'T') | KeyPress::Enter) {
            if let Some(model) = self.model_list.selected_item().cloned() {
                self.open_terminal(Some(model.path), format!("{} shell", model.model_ref));
                self.mode = ShellMode::Normal;
                return true;
            }
        }

        match self.model_list.handle_key(&key) {
            WidgetResult::Cancelled => {
                self.mode = ShellMode::Normal;
                true
            }
            WidgetResult::Confirmed(model) => {
                // Scoped terminal: only on platforms with sandbox support.
                #[cfg(any(target_os = "linux", target_os = "openbsd"))]
                self.open_terminal(Some(model.path), format!("{} shell", model.model_ref));
                #[cfg(not(any(target_os = "linux", target_os = "openbsd")))]
                let _ = model; // worktree terminal not available on this platform
                self.mode = ShellMode::Normal;
                true
            }
            WidgetResult::Pending => true,
        }
    }

    fn handle_start_menu(&mut self, key: KeyPress, selected: usize) -> bool {
        let n = MENU_ITEMS.len();
        // Chord shortcuts execute immediately, closing the menu.
        let chord_action: Option<usize> = match key {
            KeyPress::Char(b'n' | b'N') => Some(0),
            KeyPress::Char(b'q' | b'Q') => Some(1),
            KeyPress::Tab               => Some(2),
            KeyPress::Char(b'm' | b'M') => Some(3),
            KeyPress::Char(b's' | b'S') => Some(4),
            KeyPress::Char(b'd' | b'D') => Some(5),
            _ => None,
        };
        if let Some(idx) = chord_action {
            self.mode = ShellMode::Normal;
            return self.execute_menu_action(idx);
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
                return self.execute_menu_action(selected);
            }
            _ => {}
        }
        true
    }

    fn execute_menu_action(&mut self, idx: usize) -> bool {
        match idx {
            0 => self.open_terminal(None, "shell".to_owned()),
            1 => self.close_active(),
            2 => self.cycle_window(),
            3 => { self.mode = ShellMode::ModelList; }
            4 => {
                self.saved_style = self.bg.style;
                let i = ALL_STYLES.iter().position(|s| *s == self.bg.style).unwrap_or(0);
                self.settings_list = SelectList::new("Background", ALL_STYLES.to_vec())
                    .with_selected(i);
                self.preview_bg.style = self.bg.style;
                self.mode = ShellMode::Settings;
            }
            5 => { self.quit = true; }
            _ => {}
        }
        true
    }

    fn handle_settings(&mut self, key: KeyPress) -> bool {
        match self.settings_list.handle_key(&key) {
            WidgetResult::Cancelled => {
                // Restore saved style on Esc.
                self.bg.style      = self.saved_style;
                self.preview_bg.style = self.saved_style;
                self.mode          = ShellMode::Normal;
                true
            }
            WidgetResult::Confirmed(style) => {
                self.bg.style      = style;
                self.preview_bg.style = style;
                self.mode          = ShellMode::Normal;
                true
            }
            WidgetResult::Pending => {
                // Live-preview: sync bg style to the highlighted list entry.
                if let Some(&style) = self.settings_list.selected_item() {
                    self.bg.style      = style;
                    self.preview_bg.style = style;
                }
                true
            }
        }
    }
}

impl TerminalApp for ShellApp {
    type Command = ShellCmd;

    fn render(&self, frame: &mut ratatui::Frame) {
        crate::shell_ui::draw(frame, self);
    }

    fn handle_input(&mut self, cmd: ShellCmd) -> bool {
        let ShellCmd::Key(key) = cmd;
        match self.mode {
            ShellMode::Normal                 => self.handle_normal(key),
            ShellMode::ModelList              => self.handle_model_list(key),
            ShellMode::Settings               => self.handle_settings(key),
            ShellMode::StartMenu { selected } => self.handle_start_menu(key, selected),
        }
    }

    fn tick(&mut self, _delta_ms: u64) -> bool {
        let mut redraw = false;
        let mut dead: Vec<usize> = Vec::new();
        for (i, win) in self.windows.iter_mut().enumerate() {
            let (got_data, is_dead) = win.drain();
            if got_data && i == self.active {
                redraw = true;
            }
            if is_dead {
                dead.push(i);
            }
        }
        // Remove dead windows in reverse order so earlier indices stay valid.
        for i in dead.into_iter().rev() {
            self.windows.remove(i);
            if !self.windows.is_empty() && self.active >= self.windows.len() {
                self.active = self.windows.len() - 1;
            }
            redraw = true;
        }

        // Drain model status updates from background load-polling threads.
        while let Ok((model_ref, loaded)) = self.status_rx.try_recv() {
            for entry in self.model_list.items_mut() {
                if entry.model_ref == model_ref {
                    entry.loaded = loaded;
                }
            }
            // Clear the "Loading…" status once the model responds.
            if self.load_status.as_ref()
                .is_some_and(|s| s.contains(&model_ref))
            {
                self.load_status = None;
            }
            redraw = true;
        }

        // Tick background animation when it is visible.
        let show_bg = self.windows.is_empty() || matches!(self.mode, ShellMode::Settings);
        if show_bg {
            self.bg.tick(self.cols, self.pane_rows());
            if self.bg.is_animated() {
                redraw = true;
            }
            // Also tick the settings preview at its fixed small dimensions.
            if matches!(self.mode, ShellMode::Settings) {
                self.preview_bg.tick(PREVIEW_W, PREVIEW_H);
            }
        }

        redraw
    }

    fn should_quit(&self) -> bool {
        self.quit
    }
}

// Re-export from compositor (canonical implementation).
pub use hyprstream_compositor::keypress_to_bytes;
