//! ConsoleApp — floating log console backed by tui-logger.
//!
//! Holds mutable TuiWidgetState (scroll position, filters) and renders using
//! TuiLoggerWidget. Lives in shell_handlers.rs, NOT in the compositor (which
//! must remain WASM-safe).

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style};
use tui_logger::{TuiLoggerWidget, TuiWidgetState};
use hyprstream_compositor::theme;

pub struct ConsoleApp {
    pub state: TuiWidgetState,
}

impl ConsoleApp {
    pub fn new() -> Self {
        Self {
            state: TuiWidgetState::new(),
        }
    }

    /// Render the console content into `inner` (already inside the modal block frame).
    pub fn draw(&mut self, frame: &mut Frame, inner: Rect) {
        TuiLoggerWidget::default()
            .style_error(Style::default().fg(Color::Red))
            .style_warn(Style::default().fg(Color::Yellow))
            .style_info(Style::default().fg(Color::White))
            .style_debug(Style::default().fg(theme::DIM))
            .output_timestamp(Some("%H:%M:%S".into()))
            .output_target(true)
            .output_file(false)
            .output_line(false)
            .state(&self.state)
            .render(inner, frame.buffer_mut());
    }
}

impl Default for ConsoleApp {
    fn default() -> Self {
        Self::new()
    }
}
