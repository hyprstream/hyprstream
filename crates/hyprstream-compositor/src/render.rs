//! Ratatui chrome rendering for the compositor.
//!
//! `draw` is the unified renderer used by both the native event loop
//! (`terminal.draw(|f| render::draw(f, compositor))`) and the WASM path
//! (Phase W2, rendering into `AnsiBackend<Vec<u8>>`).
//!
//! Layout:
//! ```text
//! ┌─ status bar (1 row) ─────────────────────────────┐
//! │  pane content (Min rows) — VT cells or background │
//! │  [model/settings modal overlay]                   │
//! ├─ window strip / taskbar (1 row) ─────────────────┤
//! └─ F-key legend (1 row) ────────────────────────────┘
//! ```

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Tabs};
use ratatui::Frame;
use waxterm::avt_style::avt_pen_to_style;

use crate::background::PREVIEW_W;
use crate::chrome::{ShellChrome, ShellMode, ToastLevel, MENU_ITEMS};
use crate::layout::{CursorState, LayoutTree, PaneStorage};
use crate::theme;

// ============================================================================
// Top-level draw
// ============================================================================

pub fn draw(frame: &mut Frame, chrome: &ShellChrome, layout: &LayoutTree) {
    let area = frame.area();
    let chunks = Layout::vertical([
        Constraint::Length(1), // status bar
        Constraint::Length(1), // window titlebar
        Constraint::Min(1),    // pane content / background
        Constraint::Length(1), // window strip
        Constraint::Length(1), // F-key legend
    ])
    .split(area);

    draw_status_bar(frame, chunks[0], chrome);
    draw_window_titlebar(frame, chunks[1], chrome);
    draw_content(frame, chunks[2], chrome, layout);
    draw_window_strip(frame, chunks[3], chrome);
    draw_fkey_bar(frame, chunks[4], &chrome.mode);

    match chrome.mode {
        ShellMode::ModelList              => draw_model_modal(frame, area, chrome),
        ShellMode::Settings               => draw_settings_modal(frame, area, chrome),
        ShellMode::StartMenu { selected } => draw_start_menu(frame, area, selected),
        ShellMode::Console                => draw_console_modal(frame, area, chrome),
        ShellMode::Normal                 => {}
    }

    // Toasts are drawn on top of everything (except modals).
    if !chrome.toasts.is_empty() && !matches!(chrome.mode, ShellMode::Console) {
        draw_toasts(frame, chunks[1], chrome);
    }
}

// ============================================================================
// Status bar
// ============================================================================

fn draw_status_bar(frame: &mut Frame, area: Rect, _chrome: &ShellChrome) {
    let clock = current_time_str();
    let clock_width = clock.len() as u16 + 1;

    let left_spans = vec![
        Span::raw(" \u{1f319} "),                            // 🌙
        Span::styled("HYPRSTREAM", theme::title_style()),
    ];

    let left_w = area.width.saturating_sub(clock_width);
    let left_area  = Rect { width: left_w, ..area };
    let right_area = Rect { x: area.x + left_w, width: clock_width, ..area };

    frame.render_widget(
        Paragraph::new(Line::from(left_spans))
            .style(Style::default().bg(theme::BG_PANEL)),
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
// Window titlebar — one row below status bar, distinct color from status/fkeys
// ============================================================================

fn draw_window_titlebar(frame: &mut Frame, area: Rect, chrome: &ShellChrome) {
    let win = chrome.windows.get(chrome.active_win);
    let title_spans: Vec<Span> = if let Some(w) = win {
        let is_private = w.panes.iter().any(|p| chrome.private_panes.contains(&p.id));
        let lock = if is_private { " \u{1f512}" } else { "" };
        vec![
            Span::styled("  ", theme::titlebar_dim_style()),
            Span::styled(format!("{}{lock}", w.name), theme::titlebar_style()),
        ]
    } else {
        vec![Span::styled("  no windows", theme::titlebar_dim_style())]
    };

    if let Some(status) = &chrome.load_status {
        let right_text = format!("  {}  ", status);
        let right_w = right_text.chars().count() as u16;
        let left_w = area.width.saturating_sub(right_w);
        let left_area  = Rect { width: left_w, ..area };
        let right_area = Rect { x: area.x + left_w, width: right_w, ..area };
        frame.render_widget(
            Paragraph::new(Line::from(title_spans)).style(Style::default().bg(theme::BG_TITLEBAR)),
            left_area,
        );
        frame.render_widget(
            Paragraph::new(Span::styled(right_text, theme::titlebar_dim_style()))
                .style(Style::default().bg(theme::BG_TITLEBAR)),
            right_area,
        );
    } else {
        frame.render_widget(
            Paragraph::new(Line::from(title_spans)).style(Style::default().bg(theme::BG_TITLEBAR)),
            area,
        );
    }
}

// ============================================================================
// Pane content — VT cells or animated background
// ============================================================================

fn draw_content(frame: &mut Frame, area: Rect, chrome: &ShellChrome, layout: &LayoutTree) {
    if chrome.windows.is_empty() {
        chrome.bg.render(frame, area);
        return;
    }
    let pane_id = chrome.active_pane_id();
    if let Some(pane) = layout.get_pane(pane_id) {
        match &pane.storage {
            PaneStorage::ServerBuf { buf, cursor } => {
                draw_buffer_cells(frame, area, buf, cursor);
            }
            PaneStorage::ServerVt(vt) | PaneStorage::PrivateVt(vt) => {
                draw_vt_cells(frame, area, vt);
            }
        }
    } else {
        chrome.bg.render(frame, area);
    }
}

/// Render a ratatui Buffer into `area` by copying cells into the frame buffer.
fn draw_buffer_cells(
    frame: &mut Frame,
    area: Rect,
    src: &ratatui::buffer::Buffer,
    cursor: &CursorState,
) {
    // Fill the area with the unified pane background first so empty cells match VT rendering.
    frame.render_widget(Paragraph::new("").style(Style::default().bg(theme::BG)), area);
    let dst = frame.buffer_mut();
    let src_area = src.area;
    let dst_width = dst.area.width as usize;

    for y in 0..src_area.height.min(area.height) {
        for x in 0..src_area.width.min(area.width) {
            let src_idx = y as usize * src_area.width as usize + x as usize;
            let dst_x = area.x + x;
            let dst_y = area.y + y;
            let dst_idx = dst_y as usize * dst_width + dst_x as usize;
            if src_idx < src.content.len() && dst_idx < dst.content.len() {
                dst.content[dst_idx] = src.content[src_idx].clone();
            }
        }
    }

    if cursor.visible {
        let sx = area.x + cursor.x.min(area.width.saturating_sub(1));
        let sy = area.y + cursor.y.min(area.height.saturating_sub(1));
        frame.set_cursor_position((sx, sy));
    }
}

/// Render an `avt::Vt` buffer into `area`.
pub fn draw_vt_cells(frame: &mut Frame, area: Rect, vt: &avt::Vt) {
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

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(theme::BG)),
        area,
    );

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

fn draw_window_strip(frame: &mut Frame, area: Rect, chrome: &ShellChrome) {
    if chrome.windows.is_empty() {
        frame.render_widget(
            Paragraph::new(" [no windows]")
                .style(Style::default().fg(theme::DIM).bg(theme::BG_PANEL)),
            area,
        );
        return;
    }

    let titles: Vec<Line> = chrome
        .windows
        .iter()
        .enumerate()
        .map(|(i, win)| {
            let lock = if win.panes.iter().any(|p| chrome.private_panes.contains(&p.id)) {
                "\u{1f512} "  // 🔒
            } else {
                ""
            };
            let label = format!(" {lock}{} ", win.name);
            if i == chrome.active_win {
                Line::from(Span::styled(label, theme::tab_active()))
            } else {
                Line::from(Span::styled(label, theme::tab_inactive()))
            }
        })
        .collect();

    let tabs = Tabs::new(titles)
        .select(chrome.active_win)
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
        ShellMode::StartMenu { .. } => {
            frame.render_widget(
                Paragraph::new("").style(Style::default().bg(theme::BG_PANEL)),
                area,
            );
            return;
        }
        ShellMode::Normal => &[
            ("F7",  "new"),
            ("F8",  "close"),
            ("F9",  "log"),
            ("F10", "models"),
            ("F11", "settings"),
            ("F12", "quit"),
        ],
        ShellMode::Console => &[
            ("\u{2191}\u{2193}", "scroll"),
            ("F9/Esc",           "close"),
        ],
        ShellMode::ModelList => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("c",                "Private"),
            ("C/Enter",          "Server chat"),
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

fn draw_model_modal(frame: &mut Frame, full_area: Rect, chrome: &ShellChrome) {
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
        Span::styled("c",     theme::help_key()),
        Span::styled(" private  ",  theme::help_text()),
        Span::styled("C",     theme::help_key()),
        Span::styled("/",     theme::help_text()),
        Span::styled("Enter", theme::help_key()),
        Span::styled(" server  ",   theme::help_text()),
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

    chrome.model_list.render(frame, list_area);
}

// ============================================================================
// Settings modal
// ============================================================================

fn draw_settings_modal(frame: &mut Frame, full_area: Rect, chrome: &ShellChrome) {
    let modal = centered_rect(50, 65, full_area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(" Settings ", theme::title_style()))
        .borders(Borders::ALL)
        .border_style(theme::border_style())
        .style(Style::default().bg(theme::BG_PANEL));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);

    let list_height = 5u16.min(inner.height / 2);
    let chunks = Layout::vertical([
        Constraint::Length(list_height),
        Constraint::Min(3),
    ])
    .split(inner);

    chrome.settings_list.render(frame, chunks[0]);

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
    chrome.preview_bg.render(frame, preview_render);
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
        let text = format!(
            " {:<lw$}{:>rw$}",
            label, hint,
            lw = inner_w.saturating_sub(hint.len() + 2),
            rw = hint.len() + 1,
        );
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(text, style)))
                .style(Style::default().bg(theme::BG_PANEL)),
            row,
        );
    }
}

// ============================================================================
// Toast overlay
// ============================================================================

fn draw_toasts(frame: &mut Frame, area: Rect, chrome: &ShellChrome) {
    for (i, toast) in chrome.toasts.iter().enumerate() {
        let fg = match toast.level {
            ToastLevel::Error => Color::Red,
            ToastLevel::Warn  => Color::Yellow,
            ToastLevel::Info  => Color::from((150u8, 150u8, 150u8)),
        };
        let max_w = (area.width as usize).min(60);
        let msg = if toast.message.len() > max_w.saturating_sub(4) {
            format!("  {}…  ", &toast.message[..max_w.saturating_sub(5)])
        } else {
            format!("  {}  ", toast.message)
        };
        let w = msg.len() as u16;
        let x = area.x + area.width.saturating_sub(w);
        let y = area.y + i as u16;
        if y >= area.y + area.height {
            break;
        }
        let toast_area = Rect { x, y, width: w, height: 1 };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(msg, Style::default().fg(fg).bg(theme::BG_PANEL)))),
            toast_area,
        );
    }
}

// ============================================================================
// Console log modal
// ============================================================================

fn draw_console_modal(frame: &mut Frame, full_area: Rect, chrome: &ShellChrome) {
    let modal = centered_rect(85, 80, full_area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(" Console Log ", theme::title_style()))
        .borders(Borders::ALL)
        .border_style(theme::border_style())
        .style(Style::default().bg(theme::BG_PANEL));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);

    // Footer hint
    let footer_area = Rect { y: inner.y + inner.height.saturating_sub(1), height: 1, ..inner };
    let log_area = Rect { height: inner.height.saturating_sub(1), ..inner };

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("\u{2191}\u{2193}", theme::help_key()),
            Span::styled(" scroll  ", theme::help_text()),
            Span::styled("F9/Esc", theme::help_key()),
            Span::styled(" close", theme::help_text()),
        ]))
        .style(Style::default().bg(theme::BG_PANEL)),
        footer_area,
    );

    let visible = log_area.height as usize;
    let total = chrome.console_log.len();
    // scroll is an index into console_log; show lines starting from scroll.
    let start = chrome.console_scroll.min(total.saturating_sub(1));
    let recent_threshold = total.saturating_sub(10);

    let lines: Vec<Line> = chrome.console_log.iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(idx, line)| {
            let style = if idx >= recent_threshold {
                Style::default().fg(Color::White)
            } else {
                Style::default().fg(Color::from((120u8, 120u8, 120u8)))
            };
            Line::from(Span::styled(line.clone(), style))
        })
        .collect();

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(theme::BG_PANEL)),
        log_area,
    );
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
