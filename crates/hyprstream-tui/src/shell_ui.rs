//! Rendering for the TUI shell.
//!
//! Layout:
//! ```text
//! ┌─ status bar (1 row) ─────────────────────────────┐
//! │  pane content (Min rows)                          │
//! │  [modal overlay when ModelList mode]              │
//! ├─ window strip / taskbar (1 row) ─────────────────┤
//! └─ F-key legend (1 row) ────────────────────────────┘
//! ```

#![cfg(not(target_os = "wasi"))]

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Tabs};
use ratatui::Frame;
use waxterm::avt_style::avt_pen_to_style;

use crate::shell_app::{PaneWindow, ShellApp, ShellMode};
use crate::theme;

// ============================================================================
// Top-level draw
// ============================================================================

pub fn draw(frame: &mut Frame, app: &ShellApp) {
    let area = frame.area();
    let chunks = Layout::vertical([
        Constraint::Length(1), // status bar
        Constraint::Min(1),    // pane content
        Constraint::Length(1), // window strip
        Constraint::Length(1), // F-key legend
    ])
    .split(area);

    draw_status_bar(frame, chunks[0]);
    draw_content(frame, chunks[1], app);
    draw_window_strip(frame, chunks[2], app);
    draw_fkey_bar(frame, chunks[3], &app.mode);

    if matches!(app.mode, ShellMode::ModelList) {
        draw_model_modal(frame, area, app);
    }
}

// ============================================================================
// Status bar
// ============================================================================

fn draw_status_bar(frame: &mut Frame, area: Rect) {
    let clock = current_time_str();
    let clock_width = clock.len() as u16 + 1; // +1 for trailing space

    let left_w = area.width.saturating_sub(clock_width);
    let left_area = Rect { width: left_w, ..area };
    let right_area = Rect { x: area.x + left_w, width: clock_width, ..area };

    let bar = Paragraph::new(Line::from(vec![
        Span::raw(" "),
        Span::styled("HYPRSTREAM", theme::title_style()),
    ]))
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
// Pane content
// ============================================================================

fn draw_content(frame: &mut Frame, area: Rect, app: &ShellApp) {
    match app.windows.get(app.active) {
        Some(win) => draw_pane(frame, area, win),
        None => {
            let msg = Paragraph::new("No windows open.  F7 = new terminal  F10 = models")
                .style(Style::default().fg(theme::DIM).bg(theme::BG));
            frame.render_widget(msg, area);
        }
    }
}

fn draw_pane(frame: &mut Frame, area: Rect, win: &PaneWindow) {
    let max_lines = area.height as usize;

    let lines: Vec<Line> = win
        .vt
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
    let cursor = win.vt.cursor();
    if cursor.visible {
        let sx = area.x + (cursor.col as u16).min(area.width.saturating_sub(1));
        let sy = area.y + (cursor.row as u16).min(area.height.saturating_sub(1));
        frame.set_cursor_position((sx, sy));
    }
}

// ============================================================================
// Window strip
// ============================================================================

fn draw_window_strip(frame: &mut Frame, area: Rect, app: &ShellApp) {
    if app.windows.is_empty() {
        frame.render_widget(
            Paragraph::new(" [no windows]")
                .style(Style::default().fg(theme::DIM).bg(theme::BG_PANEL)),
            area,
        );
        return;
    }

    let titles: Vec<Line> = app
        .windows
        .iter()
        .enumerate()
        .map(|(i, win)| {
            let label = format!(" {} ", win.title);
            if i == app.active {
                Line::from(Span::styled(label, theme::tab_active()))
            } else {
                Line::from(Span::styled(label, theme::tab_inactive()))
            }
        })
        .collect();

    let tabs = Tabs::new(titles)
        .select(app.active)
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
        ShellMode::Normal => &[
            ("F7", "New"),
            ("F8", "Close"),
            ("Tab", "Cycle"),
            ("F10", "Models"),
            ("F12", "Quit"),
        ],
        ShellMode::ModelList => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("Enter/T", "Terminal"),
            ("l/u", "Load/Unload"),
            ("Esc", "Close"),
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
        Paragraph::new(Line::from(spans))
            .style(Style::default().bg(theme::BG_PANEL)),
        area,
    );
}

// ============================================================================
// Model list modal
// ============================================================================

fn draw_model_modal(frame: &mut Frame, full_area: Rect, app: &ShellApp) {
    let modal = centered_rect(60, 70, full_area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(" Models ", theme::title_style()))
        .borders(Borders::ALL)
        .border_style(theme::border_style())
        .style(Style::default().bg(theme::BG_PANEL));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);

    // Reserve bottom row for the action hint.
    let hint_area = Rect { y: inner.y + inner.height.saturating_sub(1), height: 1, ..inner };
    let list_area = Rect { height: inner.height.saturating_sub(2), ..inner };

    let hint = Paragraph::new(Line::from(vec![
        Span::styled("Enter", theme::help_key()),
        Span::styled("/", theme::help_text()),
        Span::styled("T", theme::help_key()),
        Span::styled(" terminal  ", theme::help_text()),
        Span::styled("l", theme::help_key()),
        Span::styled(" load  ", theme::help_text()),
        Span::styled("u", theme::help_key()),
        Span::styled(" unload  ", theme::help_text()),
        Span::styled("Esc", theme::help_key()),
        Span::styled(" close", theme::help_text()),
    ]))
    .style(Style::default().bg(theme::BG_PANEL));
    frame.render_widget(hint, hint_area);

    app.model_list.render(frame, list_area);
}

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
