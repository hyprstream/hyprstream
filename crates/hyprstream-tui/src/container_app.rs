//! ContainerTermApp — read-only terminal for container stdout attach.
//!
//! Displays container output in a VT pane. Used by the AttachContainer
//! RPC handler (Phase W-III-a). Stdin forwarding (Phase W-III-b) is
//! stubbed via `VtPane::send_bytes`.

#![cfg(not(target_os = "wasi"))]

use ratatui::Frame;
use waxterm::app::TerminalApp;
use waxterm::input::KeyPress;

use crate::vt_pane::VtPane;

/// A terminal app that displays container stdout via a `VtPane`.
pub struct ContainerTermApp {
    pub pane: VtPane,
    pub container_id: String,
    pub sandbox_id: String,
    pub cols: u16,
    pub rows: u16,
    pub quit: bool,
    pub pending_toasts: Vec<String>,
}

impl ContainerTermApp {
    pub fn new(
        container_id: String,
        sandbox_id: String,
        cols: u16,
        rows: u16,
        stdout_rx: std::sync::mpsc::Receiver<Vec<u8>>,
        stdin_tx: Option<std::sync::mpsc::SyncSender<Vec<u8>>>,
    ) -> Self {
        Self {
            pane: VtPane::new(cols, rows, stdout_rx, stdin_tx),
            container_id,
            sandbox_id,
            cols,
            rows,
            quit: false,
            pending_toasts: Vec::new(),
        }
    }
}

impl TerminalApp for ContainerTermApp {
    type Command = KeyPress;

    fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        hyprstream_compositor::render::draw_vt_cells(frame, area, &self.pane.vt);
    }

    fn handle_input(&mut self, key: KeyPress) -> bool {
        // Phase W-III-b: forward key bytes via pane.send_bytes()
        let bytes = hyprstream_compositor::keypress_to_bytes(key);
        if !bytes.is_empty() {
            self.pane.send_bytes(bytes);
        }
        false
    }

    fn tick(&mut self, _delta_ms: u64) -> bool {
        let (got_data, is_dead) = self.pane.drain();
        if is_dead {
            self.quit = true;
        }
        got_data
    }

    fn should_quit(&self) -> bool {
        self.quit
    }
}
