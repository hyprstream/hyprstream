//! Minimal WASM-compatible rendering for [`ChatApp`].
//!
//! Replaces `wasm_chat.rs` rendering — uses `ChatHistoryEntry` with thinking
//! blocks instead of `(role, content)` pairs.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use hyprstream_compositor::theme;

use crate::chat_app::{ChatApp, ChatHistoryEntry, ChatMode, ChatRole};

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

fn draw_status_bar(frame: &mut Frame, area: Rect, app: &ChatApp) {
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

fn draw_history(frame: &mut Frame, area: Rect, app: &ChatApp) {
    let width = area.width.max(1) as usize;
    let mut all_lines: Vec<Line<'static>> = Vec::new();

    for entry in &app.history {
        let (prefix, prefix_style): (&str, Style) = match entry.role {
            ChatRole::User => ("You: ", theme::tab_active()),
            ChatRole::Assistant => ("AI:  ", Style::default().fg(Color::Cyan)),
        };

        let full = format!("{}{}", prefix, entry.content);
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

    let total = all_lines.len();
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

fn draw_input_line(frame: &mut Frame, area: Rect, app: &ChatApp) {
    let (content, show_cursor) = if matches!(app.mode, ChatMode::Streaming) {
        ("\u{2026} (generating)".to_owned(), false)
    } else {
        (format!("> {}", app.input_buf), true)
    };
    frame.render_widget(
        Paragraph::new(Line::from(content))
            .style(Style::default().bg(theme::BG_PANEL).fg(Color::White)),
        area,
    );
    if show_cursor {
        let cx = (area.x + 2 + app.input_buf.len() as u16)
            .min(area.x + area.width.saturating_sub(1));
        frame.set_cursor_position((cx, area.y));
    }
}

fn draw_fkey_bar(frame: &mut Frame, area: Rect, app: &ChatApp) {
    let keys: &[(&str, &str)] = if matches!(app.mode, ChatMode::Streaming) {
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
