//! Ratatui chrome rendering for the compositor.
//!
//! `draw` is the unified renderer used by both the native event loop
//! (`terminal.draw(|f| render::draw(f, compositor))`) and the WASM path
//! (Phase W2, rendering into `AnsiBackend<Vec<u8>>`).
//!
//! Layout:
//! ```text
//! ┌─ status bar (1 row) ─────────────────────────────┐
//! ╭─ 🔒 Name ──────────────────────────────────────╮  ← pane block top border (title)
//! │  pane content (Min rows) — VT cells or background │
//! ╰───────────────────────────────────────────────────╯  ← pane block bottom border
//! ├─ window strip / taskbar (1 row) ─────────────────┤
//! └─ F-key legend (1 row) ────────────────────────────┘
//! ```

use ratatui::layout::{Constraint, Flex, Layout, Rect};
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

    if matches!(chrome.mode, ShellMode::Fullscreen) {
        // Fullscreen: pane fills entire terminal, no chrome rows.
        draw_content(frame, area, chrome, layout);
        if !chrome.toasts.is_empty() {
            draw_toasts(frame, area, chrome);
        }
        return;
    }

    let [status, pane_block, strip, fkeys] = Layout::vertical([
        Constraint::Length(1), // global status bar
        Constraint::Min(1),    // pane block (border + content + bottom border)
        Constraint::Length(1), // window strip
        Constraint::Length(1), // F-key legend
    ])
    .areas(area);

    draw_status_bar(frame, status, chrome);
    draw_pane_block(frame, pane_block, chrome, layout);
    draw_window_strip(frame, strip, chrome);
    draw_fkey_bar(frame, fkeys, &chrome.mode);

    match chrome.mode {
        ShellMode::ModelList              => draw_model_modal(frame, area, chrome),
        ShellMode::Settings               => draw_settings_modal(frame, area, chrome),
        ShellMode::StartMenu { selected } => draw_start_menu(frame, area, selected),
        ShellMode::Console                => { /* drawn by shell_handlers (native) or wasm path */ }
        ShellMode::Normal | ShellMode::Fullscreen => {}
    }

    // Toasts drawn on top of pane area (not over modals).
    if !chrome.toasts.is_empty() && !matches!(chrome.mode, ShellMode::Console) {
        draw_toasts(frame, pane_block, chrome);
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
// Pane block — rounded border whose title row is the window titlebar
// ============================================================================

fn pane_title_line(chrome: &ShellChrome) -> Line<'static> {
    let win = chrome.windows.get(chrome.active_win);
    if let Some(w) = win {
        let is_private = w.panes.iter().any(|p| chrome.private_panes.contains(&p.id));
        let mut spans = vec![];
        if is_private {
            spans.push(Span::styled(" \u{1f512} ", Style::default().fg(theme::LOCK_COLOR)));
        } else {
            spans.push(Span::raw(" "));
        }
        spans.push(Span::styled(w.name.clone(), theme::titlebar_style()));
        if let Some(status) = &chrome.load_status {
            spans.push(Span::styled(
                format!("  {}  ", status),
                theme::titlebar_dim_style(),
            ));
        }
        Line::from(spans)
    } else {
        Line::from(Span::styled(" no windows", theme::titlebar_dim_style()))
    }
}

fn draw_pane_block(frame: &mut Frame, area: Rect, chrome: &ShellChrome, layout: &LayoutTree) {
    let title = pane_title_line(chrome);
    let block = theme::window_block(title, true);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    draw_content(frame, inner, chrome, layout);
}

/// Clear `area`, render `block` border frame, then call `f` with the inner area.
fn render_modal<F>(frame: &mut Frame, area: Rect, block: Block<'_>, f: F)
where
    F: FnOnce(&mut Frame, Rect),
{
    frame.render_widget(Clear, area);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    f(frame, inner);
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
        ShellMode::Fullscreen => &[
            ("Ctrl-F/Esc", "exit fullscreen"),
        ],
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
    render_modal(frame, modal, theme::modal_block(Line::from(" Models ")), |frame, inner| {
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
        .style(Style::default().bg(theme::BG_MODAL));
        frame.render_widget(hint, hint_area);

        chrome.model_list.render(frame, list_area);
    });
}

// ============================================================================
// Settings modal
// ============================================================================

fn draw_settings_modal(frame: &mut Frame, full_area: Rect, chrome: &ShellChrome) {
    let modal = centered_rect(50, 65, full_area);
    render_modal(frame, modal, theme::modal_block(Line::from(" Settings ")), |frame, inner| {
        let list_height = 5u16.min(inner.height / 2);
        let [list_area, preview_outer] = Layout::vertical([
            Constraint::Length(list_height),
            Constraint::Min(3),
        ])
        .areas(inner);

        chrome.settings_list.render(frame, list_area);

        let preview_block = Block::default()
            .title(Span::styled(" Preview ", theme::help_text()))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::DIM))
            .style(Style::default().bg(Color::Black));

        let preview_inner = preview_block.inner(preview_outer);
        frame.render_widget(preview_block, preview_outer);

        let pw = preview_inner.width.min(PREVIEW_W);
        let preview_render = Rect { width: pw, ..preview_inner };
        chrome.preview_bg.render(frame, preview_render);
    });
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

#[allow(dead_code)]
fn draw_console_modal(frame: &mut Frame, content: Rect, chrome: &ShellChrome) {
    let modal = centered_rect(85, 80, content);
    render_modal(frame, modal, theme::modal_block(Line::from(" Console Log ")), |frame, inner| {
        let footer_area = Rect { y: inner.y + inner.height.saturating_sub(1), height: 1, ..inner };
        let log_area = Rect { height: inner.height.saturating_sub(1), ..inner };

        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("\u{2191}\u{2193}", theme::help_key()),
                Span::styled(" scroll  ", theme::help_text()),
                Span::styled("F9/Esc", theme::help_key()),
                Span::styled(" close", theme::help_text()),
            ]))
            .style(Style::default().bg(theme::BG_MODAL)),
            footer_area,
        );

        let visible = log_area.height as usize;
        let total = chrome.console_log.len();
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
            Paragraph::new(lines).style(Style::default().bg(theme::BG_MODAL)),
            log_area,
        );
    });
}

// ============================================================================
// Utility
// ============================================================================

fn centered_rect(pct_w: u16, pct_h: u16, area: Rect) -> Rect {
    let [v] = Layout::vertical([Constraint::Percentage(pct_h)])
        .flex(Flex::Center)
        .areas(area);
    let [h] = Layout::horizontal([Constraint::Percentage(pct_w)])
        .flex(Flex::Center)
        .areas(v);
    h
}
