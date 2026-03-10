//! Ratatui rendering for [`ChatApp`].
//!
//! Layout:
//! ```text
//! │  history scrollback (word-wrapped messages)       │
//! ├─ input line (1 row) ──────────────────────────────┤
//! └─ fkey legend / status (1 row) ────────────────────┘
//! ```
//!
//! The model name is shown in the compositor's window titlebar — no duplicate
//! top bar is rendered here. If a status message is active (error, generating),
//! the fkey legend row shows it instead of the key hints.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::chat_app::{ChatApp, ChatHistoryEntry, ChatMode, ChatRole};
use crate::theme;

// ============================================================================
// Top-level draw
// ============================================================================

pub fn draw(frame: &mut Frame, app: &ChatApp) {
    let area = frame.area();
    let chunks = Layout::vertical([
        Constraint::Min(1),    // history scrollback
        Constraint::Length(1), // input line
        Constraint::Length(1), // fkey legend / status
    ])
    .split(area);

    draw_history(frame, chunks[0], app);
    draw_input_line(frame, chunks[1], app);
    draw_fkey_bar(frame, chunks[2], app);
}

// ============================================================================
// History scrollback
// ============================================================================

/// Word-wrap a single logical line (no embedded newlines) into `width`-wide chunks,
/// emitting styled `Line` values. `first_indent` is prepended only to the first chunk.
fn wrap_line(text: &str, width: usize, style: Style) -> Vec<Line<'static>> {
    if text.is_empty() {
        return vec![Line::from(Span::raw(""))];
    }
    let mut lines = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        let take = remaining
            .char_indices()
            .take(width)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(remaining.len());
        let chunk = remaining[..take].to_owned();
        remaining = &remaining[take..];
        lines.push(Line::from(Span::styled(chunk, style)));
    }
    lines
}

/// Build all display lines for one history entry.
fn entry_lines(entry: &ChatHistoryEntry, width: usize, show_thinking: bool) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    // ── thinking block ────────────────────────────────────────────────────
    if !entry.thinking.is_empty() {
        if show_thinking {
            let think_header_style = Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC);
            lines.push(Line::from(Span::styled(
                "\u{256d} thinking".to_owned(),
                think_header_style,
            )));
            let think_style = Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC);
            for logical in entry.thinking.split('\n') {
                if logical.is_empty() {
                    lines.push(Line::from(Span::raw("")));
                } else {
                    lines.extend(wrap_line(logical, width.saturating_sub(2), think_style));
                }
            }
            lines.push(Line::from(Span::styled(
                "\u{2570}\u{2500}\u{2500}\u{2500}".to_owned(),
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            lines.push(Line::from(vec![
                Span::styled(
                    "\u{25b8} thinking  ".to_owned(),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    "[Ctrl-O to show]".to_owned(),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                ),
            ]));
        }
    }

    // ── content ───────────────────────────────────────────────────────────
    let (prefix, prefix_style): (&str, Style) = match entry.role {
        ChatRole::User => ("You: ", theme::tab_active()),
        ChatRole::Assistant => ("AI:  ", Style::default().fg(Color::Cyan)),
    };

    let content = &entry.content;
    if content.is_empty() && entry.thinking.is_empty() {
        // Blank placeholder while streaming starts.
        lines.push(Line::from(Span::styled(prefix.to_owned(), prefix_style)));
        return lines;
    }
    if content.is_empty() {
        return lines;
    }

    let logical_lines: Vec<&str> = content.split('\n').collect();
    for (li, logical) in logical_lines.iter().enumerate() {
        if li == 0 {
            // First logical line: prepend colored prefix.
            let first_text = format!("{}{}", prefix, logical);
            let prefix_bytes = prefix.len();
            if first_text.len() <= width {
                let line = Line::from(vec![
                    Span::styled(first_text[..prefix_bytes].to_owned(), prefix_style),
                    Span::raw(first_text[prefix_bytes..].to_owned()),
                ]);
                lines.push(line);
            } else {
                // Wrap the first logical line manually so prefix stays on first chunk.
                let mut remaining = first_text.as_str();
                let mut first_chunk = true;
                while !remaining.is_empty() {
                    let take = remaining
                        .char_indices()
                        .take(width)
                        .last()
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(remaining.len());
                    let chunk = &remaining[..take];
                    remaining = &remaining[take..];
                    if first_chunk {
                        let pb = prefix_bytes.min(chunk.len());
                        let line = Line::from(vec![
                            Span::styled(chunk[..pb].to_owned(), prefix_style),
                            Span::raw(chunk[pb..].to_owned()),
                        ]);
                        lines.push(line);
                        first_chunk = false;
                    } else {
                        lines.push(Line::from(Span::raw(chunk.to_owned())));
                    }
                }
            }
        } else if logical.is_empty() {
            lines.push(Line::from(Span::raw("")));
        } else {
            lines.extend(wrap_line(logical, width, Style::default()));
        }
    }

    lines
}

fn draw_history(frame: &mut Frame, area: Rect, app: &ChatApp) {
    let width = area.width.max(1) as usize;

    let mut all_lines: Vec<Line<'static>> = Vec::new();
    for entry in &app.history {
        all_lines.extend(entry_lines(entry, width, app.show_thinking));
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
    // When a status message is active in input mode, show it here instead of key hints.
    // "Generating…" is already communicated by the input line, so skip it.
    let show_status = matches!(app.mode, ChatMode::Input)
        && app.status.as_deref().map(|s| s != "Generating\u{2026}").unwrap_or(false);

    let line = if show_status {
        let msg = app.status.as_deref().unwrap_or("");
        Line::from(Span::styled(format!("  {msg}"), theme::help_text()))
    } else {
        let keys: &[(&str, &str)] = &[
            ("Enter", "Send"),
            ("\u{2191}\u{2193}", "Scroll"),
            ("Ctrl-O", "Thinking"),
            ("Esc", "Close"),
        ];
        let mut spans = Vec::new();
        spans.push(Span::raw("  "));
        for (key, label) in keys {
            spans.push(Span::styled(*key, theme::help_key()));
            spans.push(Span::raw(" "));
            spans.push(Span::styled(*label, theme::help_text()));
            spans.push(Span::raw("  "));
        }
        Line::from(spans)
    };

    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(theme::BG_PANEL)),
        area,
    );
}
