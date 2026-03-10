//! Ratatui rendering for [`ChatApp`].
//!
//! Layout:
//! ```text
//! │  history scrollback (word-wrapped messages)       │
//! ├─ textarea (multi-line input with border) ─────────┤
//! └─ fkey legend / status (1 row) ────────────────────┘
//! ```
//!
//! The model name is shown in the compositor's window titlebar — no duplicate
//! top bar is rendered here. If a status message is active (error, generating),
//! the fkey legend row shows it instead of the key hints.

use std::cell::RefCell;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::Frame;

use crate::chat_app::{ChatApp, ChatHistoryEntry, ChatMode, ChatRole};
use crate::theme;

// ============================================================================
// Thread-local EditorState for the full-screen editor overlay.
//
// edtui::EditorState is !Send (contains Rc), so we store it in a thread_local
// rather than in ChatApp (which must be Send for spawn_app_process).
// The editor is only used client-side where single-threaded rendering applies.
// ============================================================================

thread_local! {
    pub(crate) static EDITOR_STATE: RefCell<Option<edtui::EditorState>> = const { RefCell::new(None) };
}

/// Extract the current editor text from the thread_local state and clear it.
/// Called from ChatApp::handle_input on Ctrl-S to read back edited content.
pub fn take_editor_text() -> String {
    EDITOR_STATE.with(|cell| {
        let state = cell.borrow();
        if let Some(ref s) = *state {
            s.lines.to_string()
        } else {
            String::new()
        }
    })
}

// ============================================================================
// Top-level draw
// ============================================================================

pub fn draw(frame: &mut Frame, app: &ChatApp) {
    let area = frame.area();

    if matches!(app.mode, ChatMode::Editor) {
        draw_editor_overlay(frame, area, app);
        return;
    }

    // Clear the thread_local EditorState when not in editor mode so that
    // re-opening the editor starts fresh from the current textarea content.
    EDITOR_STATE.with(|cell| {
        *cell.borrow_mut() = None;
    });

    let input_height = (app.textarea.lines().len() as u16).max(1) + 2; // +2 for borders
    let [history, input_area, fkey] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(input_height),
        Constraint::Length(1),
    ])
    .areas(area);

    draw_history(frame, history, app);
    draw_input_area(frame, input_area, app);
    draw_fkey_bar(frame, fkey, app);
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
// Input area (textarea or streaming indicator)
// ============================================================================

fn draw_input_area(frame: &mut Frame, area: Rect, app: &ChatApp) {
    match app.mode {
        // Textarea is rendered in both Input and Streaming so the user can
        // compose and queue the next prompt while inference is in flight.
        ChatMode::Input | ChatMode::Streaming => {
            frame.render_widget(&app.textarea, area);
        }
        ChatMode::Editor => {
            // Editor mode is handled by draw_editor_overlay — should not reach here.
        }
    }
}

// ============================================================================
// F-key legend
// ============================================================================

fn draw_fkey_bar(frame: &mut Frame, area: Rect, app: &ChatApp) {
    // When a non-trivial status is active in Input mode, show it instead of hints.
    let show_status = matches!(app.mode, ChatMode::Input)
        && app.status.as_deref()
            .map(|s| s != "Generating\u{2026}" && s != "Cancelled")
            .unwrap_or(false);

    let line = if show_status {
        let msg = app.status.as_deref().unwrap_or("");
        Line::from(Span::styled(format!("  {msg}"), theme::help_text()))
    } else if matches!(app.mode, ChatMode::Streaming) {
        // Streaming mode: show cancel hint + queue depth.
        let mut spans = vec![
            Span::raw("  "),
            Span::styled("Esc", theme::help_key()),
            Span::raw(" "),
            Span::styled("Cancel", theme::help_text()),
            Span::raw("  "),
            Span::styled("Enter", theme::help_key()),
            Span::raw(" "),
            Span::styled("Queue", theme::help_text()),
            Span::raw("  "),
            Span::styled("Ctrl-O", theme::help_key()),
            Span::raw(" "),
            Span::styled("Thinking", theme::help_text()),
        ];
        if !app.pending_prompts.is_empty() {
            spans.push(Span::raw("  "));
            spans.push(Span::styled(
                format!("\u{23f3} {} queued", app.pending_prompts.len()),
                theme::help_text(),
            ));
        }
        Line::from(spans)
    } else {
        let keys: &[(&str, &str)] = &[
            ("Enter", "Send"),
            ("Ctrl-J", "Newline"),
            ("\u{2191}\u{2193}", "Scroll"),
            ("Ctrl-E", "Editor"),
            ("Ctrl-O", "Thinking"),
            ("Esc\u{00d7}2", "Close"),
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

// ============================================================================
// Editor overlay (edtui full-screen)
// ============================================================================

fn draw_editor_overlay(frame: &mut Frame, area: Rect, app: &ChatApp) {
    use ratatui::widgets::Clear;
    frame.render_widget(Clear, area);
    let block = hyprstream_compositor::theme::modal_block(
        Line::from(" Editor  (Ctrl-S save · Esc cancel) "),
    );
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Ensure the thread_local EditorState is initialised with the app's editor_text.
    EDITOR_STATE.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            *opt = Some(edtui::EditorState::new(
                edtui::Lines::from(app.editor_text.as_str()),
            ));
        }
        if let Some(ref mut state) = *opt {
            edtui::EditorView::new(state)
                .wrap(true)
                .render(inner, frame.buffer_mut());
        }
    });
}
