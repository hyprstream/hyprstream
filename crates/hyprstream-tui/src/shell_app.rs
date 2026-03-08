//! ShellApp — TUI shell with model list modal and basic pane management.
//!
//! Each "window" is a PTY-backed shell process. Output is VTE-parsed by
//! `avt::Vt`. The model list modal lets users open shells scoped to a
//! model worktree directory.

#![cfg(not(target_os = "wasi"))]

use std::path::PathBuf;

use avt::Vt;
use waxterm::app::TerminalApp;
use waxterm::input::KeyPress;
use waxterm::widgets::{SelectList, WidgetResult};

use crate::shell::{kill_shell, spawn_shell, ShellInput, ShellWindow};

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
}

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

pub struct ShellApp {
    pub mode: ShellMode,
    pub windows: Vec<PaneWindow>,
    pub active: usize,
    pub model_list: SelectList<ModelEntry>,
    pub cols: u16,
    pub rows: u16,
    quit: bool,
    /// Ctrl-B prefix was pressed; next key selects action.
    ctrl_b_pending: bool,
    load_fn: Box<dyn Fn(&str) -> bool + Send>,
    unload_fn: Box<dyn Fn(&str) -> bool + Send>,
}

impl ShellApp {
    pub fn new(
        models: Vec<ModelEntry>,
        cols: u16,
        rows: u16,
        load_fn: Box<dyn Fn(&str) -> bool + Send>,
        unload_fn: Box<dyn Fn(&str) -> bool + Send>,
    ) -> Self {
        let pane_rows = rows.saturating_sub(3);
        let initial = PaneWindow::new("shell".to_owned(), None, cols, pane_rows)
            .unwrap_or_else(|e| panic!("Failed to spawn shell: {e}"));

        let model_list = SelectList::new("Models", models);

        Self {
            mode: ShellMode::Normal,
            windows: vec![initial],
            active: 0,
            model_list,
            cols,
            rows,
            quit: false,
            ctrl_b_pending: false,
            load_fn,
            unload_fn,
        }
    }

    pub fn pane_rows(&self) -> u16 {
        self.rows.saturating_sub(3)
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
        // Ctrl-B prefix: next key selects multiplexer action.
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
            KeyPress::F(12) => {
                self.quit = true;
                true
            }
            KeyPress::F(10) => {
                self.mode = ShellMode::ModelList;
                true
            }
            KeyPress::F(7) => {
                self.open_terminal(None, "shell".to_owned());
                true
            }
            KeyPress::F(8) => {
                self.close_active();
                true
            }
            KeyPress::Tab => {
                self.cycle_window();
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
                let loaded = (self.load_fn)(&model.model_ref);
                let idx = self.model_list.selected_index();
                if let Some(entry) = self.model_list.items_mut().get_mut(idx) {
                    entry.loaded = loaded;
                }
                return true;
            }
        }
        if matches!(key, KeyPress::Char(b'u' | b'U')) {
            if let Some(model) = self.model_list.selected_item().cloned() {
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

        // T / Enter both open a terminal in the selected model worktree.
        let is_open = matches!(key, KeyPress::Char(b't' | b'T') | KeyPress::Enter);
        if is_open {
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
                // Enter also handled above, but handle_key may fire too.
                let path = model.path.clone();
                let title = format!("{} shell", model.model_ref);
                self.open_terminal(Some(path), title);
                self.mode = ShellMode::Normal;
                true
            }
            WidgetResult::Pending => {
                if matches!(key, KeyPress::F(10)) {
                    self.mode = ShellMode::Normal;
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
            ShellMode::Normal    => self.handle_normal(key),
            ShellMode::ModelList => self.handle_model_list(key),
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
        // Shell does not exit when all windows close (feature: pane closes, not shell).
        redraw
    }

    fn should_quit(&self) -> bool {
        self.quit
    }
}

// ============================================================================
// KeyPress → raw bytes (for forwarding to PTY)
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
