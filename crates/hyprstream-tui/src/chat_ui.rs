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

use crate::chat_app::{ChatApp, ChatHistoryEntry, ChatMode, ChatRole, ToolCallRecord};
use crate::theme;
use crate::theme::BG_PANEL;

// ============================================================================
// Thread-local EditorState for the full-screen editor overlay.
// ============================================================================

thread_local! {
    pub(crate) static EDITOR_STATE: RefCell<Option<edtui::EditorState>> = const { RefCell::new(None) };
}

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

    EDITOR_STATE.with(|cell| {
        *cell.borrow_mut() = None;
    });

    let input_height = (app.textarea.lines().len() as u16).max(1) + 2;
    let [history, input_area, fkey] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(input_height),
        Constraint::Length(1),
    ])
    .areas(area);

    draw_history(frame, history, app);
    draw_input_area(frame, input_area, app);
    draw_fkey_bar(frame, fkey, app);

    // Settings modal overlays the full screen.
    if matches!(app.mode, ChatMode::Settings { .. }) {
        draw_settings_modal(frame, area, app);
    }
}

// ============================================================================
// History scrollback
// ============================================================================

/// Word-wrap a single logical line into `width`-wide chunks.
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

// Spinner frames — cycles while a tool call is in-flight.
const SPINNER_FRAMES: &[char] = &['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'];

/// Render a single `ToolCallRecord` as 1-2 display lines.
fn tool_call_lines(tc: &ToolCallRecord, spinner_tick: u32) -> Vec<Line<'static>> {
    let dim = Style::default().fg(Color::DarkGray);
    let gear_style = Style::default().fg(Color::Yellow);
    let ok_style   = Style::default().fg(Color::Green);
    let err_style  = Style::default().fg(Color::Red);
    let spin_style = Style::default().fg(Color::Cyan);

    // First line: ⚙ description  {args (truncated)}
    let args_display = {
        let a = tc.arguments.trim();
        let end = a.char_indices().take(40).last().map(|(i, c)| i + c.len_utf8()).unwrap_or(a.len());
        if a.len() > end { format!("{}…", &a[..end]) } else { a.to_owned() }
    };
    let header = Line::from(vec![
        Span::raw("  "),
        Span::styled("\u{2699} ".to_owned(), gear_style),
        Span::styled(tc.description.clone(), dim.add_modifier(Modifier::ITALIC)),
        Span::raw("  "),
        Span::styled(args_display, dim),
    ]);

    // Second line: result or spinner.
    let status = match &tc.result {
        None => {
            let frame = SPINNER_FRAMES[(spinner_tick as usize) % SPINNER_FRAMES.len()];
            Line::from(vec![
                Span::raw("  "),
                Span::styled(frame.to_string(), spin_style),
                Span::styled(" running\u{2026}".to_owned(), dim),
            ])
        }
        Some(r) => {
            let (icon, style, text) = if r.starts_with("error:") {
                ("\u{2717}", err_style, r.as_str())
            } else {
                ("\u{2713}", ok_style, r.as_str())
            };
            let end = text.char_indices().take(80).last().map(|(i, c)| i + c.len_utf8()).unwrap_or(text.len());
            let display = if text.len() > end { format!("{}…", &text[..end]) } else { text.to_owned() };
            Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{icon} "), style),
                Span::styled(display, dim),
            ])
        }
    };

    vec![header, status]
}

/// Build all display lines for one history entry.
fn entry_lines(entry: &ChatHistoryEntry, width: usize, show_thinking: bool, spinner_tick: u32) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    // ── Tool message (injected result) ────────────────────────────────────
    if matches!(entry.role, ChatRole::Tool) {
        // Tool messages are displayed inline as part of the assistant turn
        // (the ToolCallRecord already shows the result). Skip rendering a
        // separate bubble to avoid duplication.
        return lines;
    }

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
        ChatRole::User      => ("You: ", theme::tab_active()),
        ChatRole::Assistant => ("AI:  ", Style::default().fg(Color::Cyan)),
        ChatRole::Tool      => unreachable!("Tool messages handled above"),
    };

    let content = &entry.content;
    if content.is_empty() && entry.thinking.is_empty() && entry.tool_calls.is_empty() {
        lines.push(Line::from(Span::styled(prefix.to_owned(), prefix_style)));
        return lines;
    }

    if !content.is_empty() {
        let logical_lines: Vec<&str> = content.split('\n').collect();
        for (li, logical) in logical_lines.iter().enumerate() {
            if li == 0 {
                let first_text = format!("{}{}", prefix, logical);
                let prefix_bytes = prefix.len();
                if first_text.len() <= width {
                    let line = Line::from(vec![
                        Span::styled(first_text[..prefix_bytes].to_owned(), prefix_style),
                        Span::raw(first_text[prefix_bytes..].to_owned()),
                    ]);
                    lines.push(line);
                } else {
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
    } else if entry.thinking.is_empty() && !entry.tool_calls.is_empty() {
        // Assistant with only tool calls and no text yet.
        lines.push(Line::from(Span::styled(prefix.to_owned(), prefix_style)));
    }

    // ── tool call records (inline after content) ───────────────────────────
    for tc in &entry.tool_calls {
        lines.extend(tool_call_lines(tc, spinner_tick));
    }

    lines
}

fn draw_history(frame: &mut Frame, area: Rect, app: &ChatApp) {
    let width = area.width.max(1) as usize;

    let mut all_lines: Vec<Line<'static>> = Vec::new();
    for entry in &app.history {
        all_lines.extend(entry_lines(entry, width, app.show_thinking, app.spinner_tick));
    }

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
// Input area
// ============================================================================

fn draw_input_area(frame: &mut Frame, area: Rect, app: &ChatApp) {
    match app.mode {
        ChatMode::Input | ChatMode::Streaming | ChatMode::Settings { .. } => {
            frame.render_widget(&app.textarea, area);
        }
        ChatMode::Editor => {}
    }
}

// ============================================================================
// F-key legend
// ============================================================================

fn draw_fkey_bar(frame: &mut Frame, area: Rect, app: &ChatApp) {
    // Settings mode: show navigation/save hints in the fkey bar.
    if matches!(app.mode, ChatMode::Settings { .. }) {
        let keys: &[(&str, &str)] = &[
            ("\u{2191}\u{2193}", "Select"),
            ("\u{2190}\u{2192}", "Adjust"),
            ("Enter", "Save"),
            ("Esc", "Cancel"),
        ];
        let mut spans = vec![Span::raw("  ")];
        for (key, label) in keys {
            spans.push(Span::styled(*key, theme::help_key()));
            spans.push(Span::raw(" "));
            spans.push(Span::styled(*label, theme::help_text()));
            spans.push(Span::raw("  "));
        }
        frame.render_widget(
            Paragraph::new(Line::from(spans)).style(Style::default().bg(BG_PANEL)),
            area,
        );
        return;
    }

    let show_status = matches!(app.mode, ChatMode::Input)
        && app.status.as_deref()
            .map(|s| s != "Generating\u{2026}" && s != "Cancelled")
            .unwrap_or(false);

    let line = if show_status {
        let msg = app.status.as_deref().unwrap_or("");
        Line::from(Span::styled(format!("  {msg}"), theme::help_text()))
    } else if matches!(app.mode, ChatMode::Streaming) {
        let mut spans = vec![Span::raw("  ")];
        // Suppress "Esc Cancel" while tool calls are in-flight — Esc cannot
        // interrupt tool executor threads, so showing it would be misleading.
        #[cfg(not(target_os = "wasi"))]
        if app.pending_tool_calls == 0 {
            spans.extend([
                Span::styled("Esc", theme::help_key()),
                Span::raw(" "),
                Span::styled("Cancel", theme::help_text()),
                Span::raw("  "),
            ]);
        }
        spans.extend([
            Span::styled("Enter", theme::help_key()),
            Span::raw(" "),
            Span::styled("Queue", theme::help_text()),
            Span::raw("  "),
            Span::styled("Ctrl-O", theme::help_key()),
            Span::raw(" "),
            Span::styled("Thinking", theme::help_text()),
        ]);
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
            ("s", "Settings"),
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
// Editor overlay
// ============================================================================

// ============================================================================
// Settings modal
// ============================================================================

fn draw_settings_modal(frame: &mut Frame, area: Rect, app: &ChatApp) {
    use ratatui::widgets::Clear;

    let selected_field = match app.mode {
        ChatMode::Settings { selected_field } => selected_field,
        _ => return,
    };

    // Modal dimensions: 5 fields + padding rows + 1 hint.
    let modal_w: u16 = 46;
    let modal_h: u16 = 11;
    let x = area.x + (area.width.saturating_sub(modal_w)) / 2;
    let y = area.y + (area.height.saturating_sub(modal_h)) / 2;
    let modal_area = Rect::new(x, y, modal_w.min(area.width), modal_h.min(area.height));

    frame.render_widget(Clear, modal_area);
    let block = hyprstream_compositor::theme::modal_block(
        Line::from(" Chat Settings  (Enter save \u{00b7} Esc cancel) "),
    );
    let inner = block.inner(modal_area);
    frame.render_widget(block, modal_area);

    let draft = &app.settings_draft;
    let field_defs: &[(&str, String)] = &[
        ("Max Tokens",      format!("{}", draft.max_tokens)),
        ("Temperature",     format!("{:.2}", draft.temperature)),
        ("Top-P",           format!("{:.2}", draft.top_p)),
        ("Top-K",           draft.top_k.map_or("off".to_owned(), |v| v.to_string())),
        ("Context Window",  draft.context_window.map_or(
            "model default".to_owned(),
            |v| format!("{v}"),
        )),
    ];

    let sel_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let dim_style = Style::default().fg(Color::DarkGray);
    let val_style = Style::default().fg(Color::White);
    let arrow_style = Style::default().fg(Color::Cyan);

    let mut lines: Vec<Line<'static>> = field_defs
        .iter()
        .enumerate()
        .map(|(i, (label, value))| {
            let is_sel = i == selected_field;
            let cursor = if is_sel { "\u{25b6} " } else { "  " };
            let lbl_style = if is_sel { sel_style } else { dim_style };
            let (al, ar) = if is_sel {
                (Span::styled("\u{2039} ", arrow_style), Span::styled(" \u{203a}", arrow_style))
            } else {
                (Span::raw("  "), Span::raw("  "))
            };
            Line::from(vec![
                Span::styled(cursor.to_owned(), if is_sel { sel_style } else { dim_style }),
                Span::styled(format!("{:<16}", label), lbl_style),
                al,
                Span::styled(format!("{:>10}", value), val_style),
                ar,
            ])
        })
        .collect();

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            "  \u{2191}\u{2193} select  \u{2190}\u{2192} adjust",
            dim_style,
        ),
    ]));

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(BG_PANEL)),
        inner,
    );
}

fn draw_editor_overlay(frame: &mut Frame, area: Rect, app: &ChatApp) {
    use ratatui::widgets::Clear;
    frame.render_widget(Clear, area);
    let block = hyprstream_compositor::theme::modal_block(
        Line::from(" Editor  (Ctrl-S save · Esc cancel) "),
    );
    let inner = block.inner(area);
    frame.render_widget(block, area);

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
