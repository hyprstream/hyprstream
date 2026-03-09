//! Ratatui chrome rendering for the ShellClient.
//!
//! Layout:
//! ```text
//! ┌─ status bar (1 row) ─────────────────────────────┐
//! │  pane content (Min rows) — avt::Vt or background  │
//! │  [model/settings modal overlay]                   │
//! ├─ window strip / taskbar (1 row) ─────────────────┤
//! └─ F-key legend (1 row) ────────────────────────────┘
//! ```
//!
//! Ported from `hyprstream-tui/src/shell_ui.rs`, adapted to work from
//! `ShellClientState` (avt::Vt fed by TuiService frames) rather than local
//! `PaneWindow` state.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Tabs};
use ratatui::Frame;
use waxterm::avt_style::avt_pen_to_style;

use hyprstream_tui::background::PREVIEW_W;
use hyprstream_tui::theme;

use super::shell_client::{ShellClientState, ShellMode, MENU_ITEMS};

// ============================================================================
// Top-level draw
// ============================================================================

pub fn draw(frame: &mut Frame, state: &ShellClientState) {
    let area = frame.area();
    let chunks = Layout::vertical([
        Constraint::Length(1), // status bar
        Constraint::Min(1),    // pane content / background
        Constraint::Length(1), // window strip
        Constraint::Length(1), // F-key legend
    ])
    .split(area);

    draw_status_bar(frame, chunks[0], state);
    draw_content(frame, chunks[1], state);
    draw_window_strip(frame, chunks[2], state);
    draw_fkey_bar(frame, chunks[3], &state.mode);

    match state.mode {
        ShellMode::ModelList              => draw_model_modal(frame, area, state),
        ShellMode::Settings               => draw_settings_modal(frame, area, state),
        ShellMode::StartMenu { selected } => draw_start_menu(frame, area, selected),
        ShellMode::Normal                 => {}
    }
}

// ============================================================================
// Status bar
// ============================================================================

fn draw_status_bar(frame: &mut Frame, area: Rect, state: &ShellClientState) {
    let clock = current_time_str();
    let clock_width = clock.len() as u16 + 1; // +1 for trailing space

    let mut left_spans = vec![
        Span::raw(" \u{1f319} "),                            // 🌙
        Span::styled("HYPRSTREAM", theme::title_style()),
    ];
    if let Some(status) = &state.load_status {
        left_spans.push(Span::raw("  "));
        left_spans.push(Span::styled(status.clone(), theme::help_text()));
    }

    let left_w = area.width.saturating_sub(clock_width);
    let left_area = Rect { width: left_w, ..area };
    let right_area = Rect { x: area.x + left_w, width: clock_width, ..area };

    let bar = Paragraph::new(Line::from(left_spans))
        .style(Style::default().bg(theme::BG_PANEL));
    frame.render_widget(bar, left_area);

    let clock_para = Paragraph::new(Line::from(vec![
        Span::styled(clock, theme::help_text()),
        Span::raw(" "),
    ]))
    .style(Style::default().bg(theme::BG_PANEL));
    frame.render_widget(clock_para, right_area);
}

fn current_time_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

// ============================================================================
// Pane content — VT cells or animated background
// ============================================================================

fn draw_content(frame: &mut Frame, area: Rect, state: &ShellClientState) {
    if state.windows.is_empty() {
        state.bg.render(frame, area);
    } else {
        draw_vt_cells(frame, area, state);
    }
}

/// Render the avt::Vt from `state` into `area`.
///
/// Also exported so `handle_tui_shell` can reuse this for the embedded mode.
pub fn draw_vt_cells(frame: &mut Frame, area: Rect, state: &ShellClientState) {
    let vt = &state.vt;
    let max_lines = area.height as usize;

    let lines: Vec<Line> = vt
        .view()
        .take(max_lines)
        .map(|line| {
            let cells = line.cells();
            if cells.is_empty() {
                return Line::from("");
            }
            let mut spans: Vec<Span> = Vec::new();
            let mut text = String::with_capacity(area.width as usize);
            let mut pen = *cells[0].pen();

            for cell in cells {
                let cp = *cell.pen();
                if cp != pen {
                    if !text.is_empty() {
                        spans.push(Span::styled(
                            std::mem::take(&mut text),
                            avt_pen_to_style(&pen),
                        ));
                    }
                    pen = cp;
                }
                if cell.width() > 0 {
                    text.push(cell.char());
                }
            }
            if !text.is_empty() {
                spans.push(Span::styled(text, avt_pen_to_style(&pen)));
            }
            Line::from(spans)
        })
        .collect();

    let para = Paragraph::new(lines).style(Style::default().bg(theme::BG));
    frame.render_widget(para, area);

    // Cursor
    let cursor = vt.cursor();
    if cursor.visible {
        let sx = area.x + (cursor.col as u16).min(area.width.saturating_sub(1));
        let sy = area.y + (cursor.row as u16).min(area.height.saturating_sub(1));
        frame.set_cursor_position((sx, sy));
    }
}

// ============================================================================
// Window strip
// ============================================================================

fn draw_window_strip(frame: &mut Frame, area: Rect, state: &ShellClientState) {
    if state.windows.is_empty() {
        frame.render_widget(
            Paragraph::new(" [no windows]")
                .style(Style::default().fg(theme::DIM).bg(theme::BG_PANEL)),
            area,
        );
        return;
    }

    let titles: Vec<Line> = state
        .windows
        .iter()
        .enumerate()
        .map(|(i, win)| {
            let label = format!(" {} ", win.name);
            if i == state.active_win {
                Line::from(Span::styled(label, theme::tab_active()))
            } else {
                Line::from(Span::styled(label, theme::tab_inactive()))
            }
        })
        .collect();

    let tabs = Tabs::new(titles)
        .select(state.active_win)
        .highlight_style(theme::tab_active())
        .divider(Span::styled("│", Style::default().fg(theme::DIM)))
        .style(Style::default().bg(theme::BG_PANEL));

    frame.render_widget(tabs, area);
}

// ============================================================================
// F-key legend
// ============================================================================

fn draw_fkey_bar(frame: &mut Frame, area: Rect, mode: &ShellMode) {
    let keys: &[(&str, &str)] = match mode {
        ShellMode::Normal | ShellMode::StartMenu { .. } => {
            frame.render_widget(
                Paragraph::new("").style(Style::default().bg(theme::BG_PANEL)),
                area,
            );
            return;
        }
        ShellMode::ModelList => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("Enter/C",          "Chat"),
            ("T",                "Terminal"),
            ("l/u",              "Load/Unload"),
            ("Esc",              "Close"),
        ],
        ShellMode::Settings => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("Enter",            "Select"),
            ("Esc",              "Cancel"),
        ],
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

// ============================================================================
// Model list modal
// ============================================================================

fn draw_model_modal(frame: &mut Frame, full_area: Rect, state: &ShellClientState) {
    let modal = centered_rect(60, 70, full_area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(" Models ", theme::title_style()))
        .borders(Borders::ALL)
        .border_style(theme::border_style())
        .style(Style::default().bg(theme::BG_PANEL));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);

    let hint_area = Rect { y: inner.y + inner.height.saturating_sub(1), height: 1, ..inner };
    let list_area = Rect { height: inner.height.saturating_sub(2), ..inner };

    let hint = Paragraph::new(Line::from(vec![
        Span::styled("Enter", theme::help_key()),
        Span::styled("/",     theme::help_text()),
        Span::styled("C",     theme::help_key()),
        Span::styled(" chat  ",     theme::help_text()),
        Span::styled("T",     theme::help_key()),
        Span::styled(" terminal  ", theme::help_text()),
        Span::styled("l",     theme::help_key()),
        Span::styled(" load  ",     theme::help_text()),
        Span::styled("u",     theme::help_key()),
        Span::styled(" unload  ",   theme::help_text()),
        Span::styled("Esc",   theme::help_key()),
        Span::styled(" close",      theme::help_text()),
    ]))
    .style(Style::default().bg(theme::BG_PANEL));
    frame.render_widget(hint, hint_area);

    state.model_list.render(frame, list_area);
}

// ============================================================================
// Settings modal
// ============================================================================

fn draw_settings_modal(frame: &mut Frame, full_area: Rect, state: &ShellClientState) {
    let modal = centered_rect(50, 65, full_area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(" Settings ", theme::title_style()))
        .borders(Borders::ALL)
        .border_style(theme::border_style())
        .style(Style::default().bg(theme::BG_PANEL));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);

    // Split: list on top, preview box below.
    let list_height = 5u16.min(inner.height / 2);
    let chunks = Layout::vertical([
        Constraint::Length(list_height),
        Constraint::Min(3),
    ])
    .split(inner);

    state.settings_list.render(frame, chunks[0]);

    // Preview box.
    let preview_block = Block::default()
        .title(Span::styled(" Preview ", theme::help_text()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::DIM))
        .style(Style::default().bg(Color::Black));

    let preview_outer = chunks[1];
    let preview_inner = preview_block.inner(preview_outer);
    frame.render_widget(preview_block, preview_outer);

    let pw = preview_inner.width.min(PREVIEW_W);
    let preview_render = Rect { width: pw, ..preview_inner };
    state.preview_bg.render(frame, preview_render);
}

// ============================================================================
// Start-menu popup
// ============================================================================

fn draw_start_menu(frame: &mut Frame, area: Rect, selected: usize) {
    let popup_w: u16 = 22;
    let popup_h: u16 = MENU_ITEMS.len() as u16 + 2;
    let popup_y = area.height.saturating_sub(popup_h + 2);
    let popup = Rect {
        x: 0,
        y: popup_y,
        width: popup_w.min(area.width),
        height: popup_h.min(area.height),
    };

    frame.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme::border_style())
        .style(Style::default().bg(theme::BG_PANEL));

    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let inner_w = inner.width as usize;
    for (i, (label, hint)) in MENU_ITEMS.iter().enumerate() {
        let row = Rect { y: inner.y + i as u16, height: 1, ..inner };
        let style = if i == selected { theme::tab_active() } else { theme::help_text() };
        let text = format!(" {:<lw$}{:>rw$}",
            label, hint,
            lw = inner_w.saturating_sub(hint.len() + 2),
            rw = hint.len() + 1);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(text, style)))
                .style(Style::default().bg(theme::BG_PANEL)),
            row,
        );
    }
}

// ============================================================================
// Utility
// ============================================================================

fn centered_rect(pct_x: u16, pct_y: u16, area: Rect) -> Rect {
    let margin_y = (100 - pct_y) / 2;
    let margin_x = (100 - pct_x) / 2;
    let vert = Layout::vertical([
        Constraint::Percentage(margin_y),
        Constraint::Percentage(pct_y),
        Constraint::Percentage(margin_y),
    ])
    .split(area);
    Layout::horizontal([
        Constraint::Percentage(margin_x),
        Constraint::Percentage(pct_x),
        Constraint::Percentage(margin_x),
    ])
    .split(vert[1])[1]
}
