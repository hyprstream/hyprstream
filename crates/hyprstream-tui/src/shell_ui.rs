//! Rendering for the TUI shell (standalone hyprstream-tui binary).
//!
//! Layout:
//! ```text
//! ┌─ status bar (1 row) ─────────────────────────────┐
//! │  pane content (Min rows) — VT cells or background │
//! │  [model/settings modal overlay]                   │
//! └─ window strip / taskbar (1 row) ─────────────────┘
//! ```

#![cfg(not(target_os = "wasi"))]

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Tabs};
use ratatui::Frame;

use crate::background::PREVIEW_W;
use crate::shell_app::{PaneWindow, ShellApp, ShellMode, MENU_ITEMS};
use crate::theme;

// ============================================================================
// Top-level draw
// ============================================================================

pub fn draw(frame: &mut Frame, app: &ShellApp) {
    let area = frame.area();
    let chunks = Layout::vertical([
        Constraint::Length(1), // status bar
        Constraint::Min(1),    // pane content / background
        Constraint::Length(1), // window strip
    ])
    .split(area);

    draw_status_bar(frame, chunks[0], app);
    draw_content(frame, chunks[1], app);
    draw_window_strip(frame, chunks[2], app);

    match app.mode {
        ShellMode::ModelList              => draw_model_modal(frame, area, app),
        ShellMode::Settings               => draw_settings_modal(frame, area, app),
        ShellMode::StartMenu { selected } => draw_start_menu(frame, area, selected),
        ShellMode::Normal                 => {}
    }
}

// ============================================================================
// Status bar
// ============================================================================

fn draw_status_bar(frame: &mut Frame, area: Rect, app: &ShellApp) {
    let clock = current_time_str();
    let clock_width = clock.len() as u16 + 1;

    let mut left_spans = vec![
        Span::raw(" \u{1f319} "),
        Span::styled("HYPRSTREAM", theme::title_style()),
    ];
    if let Some(status) = &app.load_status {
        left_spans.push(Span::raw("  "));
        left_spans.push(Span::styled(status.clone(), theme::help_text()));
    }

    let left_w = area.width.saturating_sub(clock_width);
    let left_area  = Rect { width: left_w, ..area };
    let right_area = Rect { x: area.x + left_w, width: clock_width, ..area };

    frame.render_widget(
        Paragraph::new(Line::from(left_spans)).style(Style::default().bg(theme::BG_PANEL)),
        left_area,
    );
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(clock, theme::help_text()),
            Span::raw(" "),
        ]))
        .style(Style::default().bg(theme::BG_PANEL)),
        right_area,
    );
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
// Pane content — VT or animated background
// ============================================================================

fn draw_content(frame: &mut Frame, area: Rect, app: &ShellApp) {
    match app.windows.get(app.active) {
        Some(win) => draw_pane(frame, area, win),
        None      => app.bg.render(frame, area),
    }
}

fn draw_pane(frame: &mut Frame, area: Rect, win: &PaneWindow) {
    hyprstream_compositor::render::draw_vt_cells(frame, area, &win.pane.vt);
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

    let hint_area = Rect { y: inner.y + inner.height.saturating_sub(1), height: 1, ..inner };
    let list_area = Rect { height: inner.height.saturating_sub(2), ..inner };

    let mut hint_spans = vec![
        Span::styled("c",     theme::help_key()),
        Span::styled(" private  ",  theme::help_text()),
        Span::styled("C",     theme::help_key()),
        Span::styled("/",     theme::help_text()),
        Span::styled("Enter", theme::help_key()),
        Span::styled(" server  ",   theme::help_text()),
    ];
    // Scoped terminal is only available on platforms with filesystem sandbox support.
    if cfg!(any(target_os = "linux", target_os = "openbsd")) {
        hint_spans.push(Span::styled("T",     theme::help_key()));
        hint_spans.push(Span::styled(" terminal  ", theme::help_text()));
    }
    hint_spans.extend([
        Span::styled("l",     theme::help_key()),
        Span::styled(" load  ",     theme::help_text()),
        Span::styled("u",     theme::help_key()),
        Span::styled(" unload  ",   theme::help_text()),
        Span::styled("Esc",   theme::help_key()),
        Span::styled(" close",      theme::help_text()),
    ]);
    let hint = Paragraph::new(Line::from(hint_spans))
    .style(Style::default().bg(theme::BG_PANEL));
    frame.render_widget(hint, hint_area);

    app.model_list.render(frame, list_area);
}

// ============================================================================
// Settings modal
// ============================================================================

fn draw_settings_modal(frame: &mut Frame, full_area: Rect, app: &ShellApp) {
    let modal = centered_rect(50, 65, full_area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(" Settings ", theme::title_style()))
        .borders(Borders::ALL)
        .border_style(theme::border_style())
        .style(Style::default().bg(theme::BG_PANEL));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);

    // Split vertically: list on top, preview in middle, hint at bottom.
    // List needs 1 (label) + 3 (items) + 1 (gap) = 5 rows minimum.
    let list_height = 5u16.min(inner.height.saturating_sub(1) / 2);
    let chunks = Layout::vertical([
        Constraint::Length(list_height),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .split(inner);

    app.settings_list.render(frame, chunks[0]);

    // Preview box.
    let preview_block = Block::default()
        .title(Span::styled(" Preview ", theme::help_text()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::DIM))
        .style(Style::default().bg(Color::Black));

    let preview_outer = chunks[1];
    let preview_inner = preview_block.inner(preview_outer);
    frame.render_widget(preview_block, preview_outer);

    // Clamp the preview render area to PREVIEW_W so matrix columns line up.
    let pw = preview_inner.width.min(PREVIEW_W);
    let preview_render = Rect { width: pw, ..preview_inner };
    app.preview_bg.render(frame, preview_render);

    // Hint bar.
    let hint_spans = vec![
        Span::styled("\u{2191}\u{2193}", theme::help_key()),
        Span::raw(" "),
        Span::styled("Navigate", theme::help_text()),
        Span::raw("  "),
        Span::styled("Enter",    theme::help_key()),
        Span::raw(" "),
        Span::styled("Select",   theme::help_text()),
        Span::raw("  "),
        Span::styled("Esc",      theme::help_key()),
        Span::raw(" "),
        Span::styled("Cancel",   theme::help_text()),
    ];
    frame.render_widget(
        Paragraph::new(Line::from(hint_spans)).style(Style::default().bg(theme::BG_PANEL)),
        chunks[2],
    );
}

// ============================================================================
// Start-menu popup
// ============================================================================

fn draw_start_menu(frame: &mut Frame, area: Rect, selected: usize) {
    let popup_w: u16 = 22;
    let popup_h: u16 = MENU_ITEMS.len() as u16 + 2;
    let popup_y = area.height.saturating_sub(popup_h + 1);
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
