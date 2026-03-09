//! Ratatui rendering for [`ChatApp`].
//!
//! Layout:
//! ```text
//! ┌─ status bar (1 row) ─────────────────────────────┐
//! │  history scrollback (word-wrapped messages)       │
//! ├─ input line (1 row) ──────────────────────────────┤
//! └─ F-key legend (1 row) ────────────────────────────┘
//! ```

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::chat_app::{ChatApp, ChatMode, ChatRole};
use crate::theme;

// ============================================================================
// Top-level draw
// ============================================================================

pub fn draw(frame: &mut Frame, app: &ChatApp) {
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

// ============================================================================
// Status bar
// ============================================================================

fn draw_status_bar(frame: &mut Frame, area: Rect, app: &ChatApp) {
    let mut spans = vec![
        Span::raw(" "),
        Span::styled("HYPRSTREAM", theme::title_style()),
        Span::raw("  "),
        Span::styled(app.model_name.clone(), theme::help_text()),
    ];
    if let Some(status) = &app.status {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(status.clone(), theme::help_text()));
    }

    let para = Paragraph::new(Line::from(spans))
        .style(Style::default().bg(theme::BG_PANEL));
    frame.render_widget(para, area);
}

// ============================================================================
// History scrollback
// ============================================================================

fn draw_history(frame: &mut Frame, area: Rect, app: &ChatApp) {
    let width = area.width.max(1) as usize;

    // Collect all word-wrapped lines from history.
    let mut all_lines: Vec<Line<'static>> = Vec::new();

    for entry in &app.history {
        let (prefix, prefix_style): (&str, Style) = match entry.role {
            ChatRole::User => ("You: ", theme::tab_active()),
            ChatRole::Assistant => ("AI: ", Style::default().fg(Color::Cyan)),
        };

        let full = format!("{}{}", prefix, entry.content);

        if full.is_empty() {
            all_lines.push(Line::from(""));
            continue;
        }

        // Wrap at terminal width.
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
                // First line: colour the prefix portion.
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

    // Apply scroll_offset from tail (0 = show newest content at bottom).
    let total = all_lines.len();
    let visible = area.height as usize;
    let skip = if total <= visible {
        0
    } else {
        let tail_start = total.saturating_sub(visible);
        tail_start.saturating_sub(app.scroll_offset)
    };

    let display_lines: Vec<Line<'static>> = all_lines
        .into_iter()
        .skip(skip)
        .take(visible)
        .collect();

    let para = Paragraph::new(display_lines).style(Style::default().bg(theme::BG));
    frame.render_widget(para, area);
}

// ============================================================================
// Input line
// ============================================================================

fn draw_input_line(frame: &mut Frame, area: Rect, app: &ChatApp) {
    let content = match app.mode {
        ChatMode::Input => format!("> {}", app.input_buf),
        ChatMode::Streaming => "\u{2026} (generating)".to_owned(),
    };

    let para = Paragraph::new(Line::from(content))
        .style(Style::default().bg(theme::BG_PANEL).fg(Color::White));
    frame.render_widget(para, area);

    // Show cursor in input mode.
    if matches!(app.mode, ChatMode::Input) {
        let cursor_x = area.x + 2 + app.input_buf.len() as u16;
        let cx = cursor_x.min(area.x + area.width.saturating_sub(1));
        frame.set_cursor_position((cx, area.y));
    }
}

// ============================================================================
// F-key legend
// ============================================================================

fn draw_fkey_bar(frame: &mut Frame, area: Rect, app: &ChatApp) {
    let keys: &[(&str, &str)] = match app.mode {
        ChatMode::Input => &[
            ("Enter", "Send"),
            ("\u{2191}\u{2193}", "Scroll"),
            ("Esc", "Close"),
        ],
        ChatMode::Streaming => &[("Esc", "Abort")],
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
