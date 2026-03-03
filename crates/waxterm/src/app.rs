use crate::input::{KeyPress, OscHandler};

/// The trait that application authors implement.
pub trait TerminalApp {
    /// App-specific command type. Must accept framework KeyPress via From.
    type Command: From<KeyPress>;

    /// Render the current frame.
    fn render(&self, frame: &mut ratatui::Frame);

    /// Handle a parsed input command. Return true if display needs redrawing.
    fn handle_input(&mut self, cmd: Self::Command) -> bool;

    /// Advance state by delta_ms. Return true if redraw needed.
    fn tick(&mut self, delta_ms: u64) -> bool;

    /// Whether the application should exit.
    fn should_quit(&self) -> bool;

    /// Override to register custom OSC input handlers.
    fn input_extensions(&self) -> Vec<OscHandler<Self::Command>> {
        vec![]
    }
}

/// Terminal configuration with builder pattern.
pub struct TerminalConfig {
    pub cols: u16,
    pub rows: u16,
    pub fps: u16,
    pub heartbeat: bool,
    pub buf_capacity: usize,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalConfig {
    pub fn new() -> Self {
        TerminalConfig {
            cols: 80,
            rows: 24,
            fps: 30,
            heartbeat: cfg!(target_os = "wasi"),
            buf_capacity: if cfg!(target_os = "wasi") { 1024 } else { 8192 },
        }
    }

    pub fn cols(mut self, cols: u16) -> Self {
        self.cols = cols;
        self
    }

    pub fn rows(mut self, rows: u16) -> Self {
        self.rows = rows;
        self
    }

    pub fn fps(mut self, fps: u16) -> Self {
        self.fps = fps;
        self
    }

    pub fn heartbeat(mut self, heartbeat: bool) -> Self {
        self.heartbeat = heartbeat;
        self
    }

    pub fn buf_capacity(mut self, capacity: usize) -> Self {
        self.buf_capacity = capacity;
        self
    }
}
