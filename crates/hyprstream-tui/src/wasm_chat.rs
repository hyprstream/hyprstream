//! WASM-compatible private chat application (Phase W3).
//!
//! # Framed stdin protocol (9-byte header: type:u8, id:u32 LE, len:u32 LE)
//!
//! | Type | Payload |
//! |------|---------|
//! | 0x01 | Raw keyboard bytes (from RouteInput) |
//! | 0x02 | Resize: cols:u16 LE, rows:u16 LE, 4 bytes pad |
//! | 0x03 | Inference token (UTF-8) |
//! | 0x04 | Inference complete (empty) |
//! | 0x05 | Inference error (UTF-8 message) |
//!
//! # Stdout
//! - Raw ANSI (ratatui via AnsiBackend)
//! - OSC 0xFD inference request:
//!   `ESC ] 0xFD <len:u32 LE> <json> BEL`

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use hyprstream_compositor::theme;

// ============================================================================
// Chat state
// ============================================================================

pub struct WasmChat {
    pub model_name: String,
    /// (role: "user"|"assistant", content)
    pub history: Vec<(String, String)>,
    pub input_buf: String,
    pub streaming: bool,
    pub scroll_offset: usize,
    pub status: Option<String>,
    pub cols: u16,
    pub rows: u16,
    pub quit: bool,
}

impl WasmChat {
    pub fn new(model_name: String, cols: u16, rows: u16) -> Self {
        Self {
            model_name,
            history: Vec::new(),
            input_buf: String::new(),
            streaming: false,
            scroll_offset: 0,
            status: None,
            cols,
            rows,
            quit: false,
        }
    }

    /// Handle a keyboard key.
    ///
    /// Returns `Some(json_string)` when the user submits a message — the caller
    /// should write this as an OSC 0xFD inference request to stdout.
    pub fn handle_key(&mut self, key: waxterm::input::KeyPress) -> Option<String> {
        use waxterm::input::KeyPress;

        if self.streaming {
            match key {
                KeyPress::Escape | KeyPress::F(10) => { self.quit = true; }
                _ => {}
            }
            return None;
        }

        match key {
            KeyPress::Escape | KeyPress::F(10) => {
                self.quit = true;
                None
            }
            KeyPress::Enter => {
                if self.input_buf.is_empty() {
                    return None;
                }
                let user_msg = std::mem::take(&mut self.input_buf);
                self.history.push(("user".to_owned(), user_msg));
                self.history.push(("assistant".to_owned(), String::new()));
                self.streaming = true;
                self.status = Some("Generating\u{2026}".to_owned());
                self.scroll_offset = 0;
                Some(self.build_inference_request())
            }
            KeyPress::Backspace => {
                self.input_buf.pop();
                None
            }
            KeyPress::Char(b) if b.is_ascii_graphic() || b == b' ' => {
                self.input_buf.push(b as char);
                None
            }
            KeyPress::ArrowUp => {
                self.scroll_offset = self.scroll_offset.saturating_add(1);
                None
            }
            KeyPress::ArrowDown => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
                None
            }
            _ => None,
        }
    }

    pub fn on_token(&mut self, token: &str) {
        if let Some((_, content)) = self.history.last_mut() {
            content.push_str(token);
        }
    }

    pub fn on_complete(&mut self) {
        self.streaming = false;
        self.status = None;
    }

    pub fn on_error(&mut self, msg: &str) {
        if let Some((_, content)) = self.history.last_mut() {
            *content = format!("[Error: {msg}]");
        }
        self.streaming = false;
        self.status = None;
    }

    fn build_inference_request(&self) -> String {
        let messages: Vec<serde_json::Value> = self
            .history
            .iter()
            .filter(|(_, c)| !c.is_empty())
            .map(|(role, content)| {
                serde_json::json!({ "role": role, "content": content })
            })
            .collect();

        serde_json::json!({
            "model": self.model_name,
            "messages": messages,
            "stream": true,
        })
        .to_string()
    }
}

// ============================================================================
// Rendering
// ============================================================================

pub fn draw(frame: &mut Frame, app: &WasmChat) {
    let area = frame.area();
    let chunks = Layout::vertical([
        Constraint::Length(1), // status bar
        Constraint::Min(1),    // history scrollback
        Constraint::Length(1), // input line
        Constraint::Length(1), // F-key legend
    ])
    .split(area);

    draw_status_bar(frame, chunks[0], app);
    draw_history(frame, chunks[1], app);
    draw_input_line(frame, chunks[2], app);
    draw_fkey_bar(frame, chunks[3], app);
}

fn draw_status_bar(frame: &mut Frame, area: Rect, app: &WasmChat) {
    let mut spans = vec![
        Span::raw(" "),
        Span::styled("HYPRSTREAM", theme::title_style()),
        Span::raw("  "),
        Span::styled(app.model_name.clone(), theme::help_text()),
        Span::styled(" [\u{1f512} PRIVATE]", Style::default().fg(theme::DIM)),
    ];
    if let Some(status) = &app.status {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(status.clone(), theme::help_text()));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(theme::BG_PANEL)),
        area,
    );
}

fn draw_history(frame: &mut Frame, area: Rect, app: &WasmChat) {
    let width = area.width.max(1) as usize;
    let mut all_lines: Vec<Line<'static>> = Vec::new();

    for (role, content) in &app.history {
        let (prefix, prefix_style): (&str, Style) = if role == "user" {
            ("You: ", theme::tab_active())
        } else {
            ("AI:  ", Style::default().fg(Color::Cyan))
        };

        let full = format!("{}{}", prefix, content);
        if full.is_empty() {
            all_lines.push(Line::from(""));
            continue;
        }

        let mut remaining = full.as_str();
        let mut first = true;
        while !remaining.is_empty() {
            let take = remaining
                .char_indices()
                .take(width)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(remaining.len());
            let chunk = &remaining[..take];
            remaining = &remaining[take..];

            if first {
                let prefix_bytes = prefix.len().min(chunk.len());
                let line = if prefix_bytes > 0 && prefix_bytes <= chunk.len() {
                    Line::from(vec![
                        Span::styled(chunk[..prefix_bytes].to_owned(), prefix_style),
                        Span::raw(chunk[prefix_bytes..].to_owned()),
                    ])
                } else {
                    Line::from(Span::styled(chunk.to_owned(), prefix_style))
                };
                all_lines.push(line);
                first = false;
            } else {
                all_lines.push(Line::from(Span::raw(chunk.to_owned())));
            }
        }
    }

    let total   = all_lines.len();
    let visible = area.height as usize;
    let skip = if total <= visible {
        0
    } else {
        total.saturating_sub(visible).saturating_sub(app.scroll_offset)
    };

    let lines: Vec<Line<'static>> = all_lines.into_iter().skip(skip).take(visible).collect();
    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(theme::BG)),
        area,
    );
}

fn draw_input_line(frame: &mut Frame, area: Rect, app: &WasmChat) {
    let content = if app.streaming {
        "\u{2026} (generating)".to_owned()
    } else {
        format!("> {}", app.input_buf)
    };
    frame.render_widget(
        Paragraph::new(Line::from(content))
            .style(Style::default().bg(theme::BG_PANEL).fg(Color::White)),
        area,
    );
    if !app.streaming {
        let cx = (area.x + 2 + app.input_buf.len() as u16)
            .min(area.x + area.width.saturating_sub(1));
        frame.set_cursor_position((cx, area.y));
    }
}

fn draw_fkey_bar(frame: &mut Frame, area: Rect, app: &WasmChat) {
    let keys: &[(&str, &str)] = if app.streaming {
        &[("Esc", "Abort")]
    } else {
        &[("Enter", "Send"), ("\u{2191}\u{2193}", "Scroll"), ("Esc", "Close")]
    };
    let mut spans = Vec::new();
    for (key, label) in keys {
        spans.push(Span::styled(*key, theme::help_key()));
        spans.push(Span::raw(" "));
        spans.push(Span::styled(*label, theme::help_text()));
        spans.push(Span::raw("  "));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(theme::BG_PANEL)),
        area,
    );
}
