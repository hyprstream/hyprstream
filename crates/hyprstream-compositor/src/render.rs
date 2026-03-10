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
use crate::chrome::{ServiceMode, ShellChrome, ShellMode, ToastLevel, MENU_ITEMS};
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
        ShellMode::ConversationPicker { .. } => draw_conversation_picker(frame, area, chrome),
        ShellMode::ServiceManager { selected } => draw_service_modal(frame, area, chrome, selected),
        ShellMode::WorkerManager { sandbox_sel, container_sel, show_containers }
            => draw_worker_modal(frame, area, chrome, sandbox_sel, container_sel, show_containers),
        ShellMode::Normal | ShellMode::Fullscreen => {}
    }

    // Toasts drawn in the upper-right of the full chrome area (below status bar).
    if !chrome.toasts.is_empty() && !matches!(chrome.mode, ShellMode::Console) {
        let toast_area = Rect {
            x: area.x,
            y: area.y + 1, // below status bar
            width: area.width,
            height: area.height.saturating_sub(3), // above strip + fkeys
        };
        draw_toasts(frame, toast_area, chrome);
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
        ShellMode::Normal => {
            #[cfg(feature = "experimental")]
            {
                &[
                    ("F5",  "svc"),
                    ("F6",  "wrk"),
                    ("F7",  "new"),
                    ("F8",  "close"),
                    ("F9",  "log"),
                    ("F10", "models"),
                    ("F11", "settings"),
                    ("F12", "quit"),
                ]
            }
            #[cfg(not(feature = "experimental"))]
            {
                &[
                    ("F6",  "wrk"),
                    ("F7",  "new"),
                    ("F8",  "close"),
                    ("F9",  "log"),
                    ("F10", "models"),
                    ("F11", "settings"),
                    ("F12", "quit"),
                ]
            }
        }
        ShellMode::Console => &[
            ("\u{2191}\u{2193}", "scroll"),
            ("F9/Esc",           "close"),
        ],
        ShellMode::ModelList => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("c",                "Chat \u{1f512}"),
            ("C/Enter",          "Chat"),
            ("T",                "Terminal"),
            ("l/u",              "Load/Unload"),
            ("Esc",              "Close"),
        ],
        ShellMode::Settings => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("Enter",            "Select"),
            ("Esc",              "Cancel"),
        ],
        ShellMode::ConversationPicker { .. } => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("Enter",            "Select"),
            ("d",                "Delete"),
            ("Esc",              "Cancel"),
        ],
        ShellMode::ServiceManager { .. } => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("t",                "Start"),
            ("s",                "Stop"),
            ("r",                "Restart"),
            ("a/S",              "All start/stop"),
            ("i",                "Install"),
            ("Esc",              "Close"),
        ],
        ShellMode::WorkerManager { show_containers: false, .. } => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("Enter/\u{2192}",   "Containers"),
            ("x",                "Destroy"),
            ("Esc",              "Close"),
        ],
        ShellMode::WorkerManager { show_containers: true, .. } => &[
            ("\u{2191}\u{2193}", "Navigate"),
            ("\u{2190}",         "Back"),
            ("e",                "Exec"),
            ("Esc",              "Close"),
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
            Span::styled(" chat \u{1f512}  ",  theme::help_text()),
            Span::styled("C",     theme::help_key()),
            Span::styled("/",     theme::help_text()),
            Span::styled("Enter", theme::help_key()),
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
// Conversation picker modal
// ============================================================================

fn draw_conversation_picker(frame: &mut Frame, full_area: Rect, chrome: &ShellChrome) {
    let (model_ref, list) = match &chrome.mode {
        ShellMode::ConversationPicker { model_ref, list } => (model_ref, list),
        _ => return,
    };
    let modal = centered_rect(55, 60, full_area);
    let title = format!(" {} — Conversations ", model_ref);
    render_modal(frame, modal, theme::modal_block(Line::from(title)), |frame, inner| {
        let hint_area = Rect { y: inner.y + inner.height.saturating_sub(1), height: 1, ..inner };
        let list_area = Rect { height: inner.height.saturating_sub(2), ..inner };

        let hint = Paragraph::new(Line::from(vec![
            Span::styled("Enter", theme::help_key()),
            Span::styled(" select  ",  theme::help_text()),
            Span::styled("d",     theme::help_key()),
            Span::styled(" delete  ",  theme::help_text()),
            Span::styled("Esc",   theme::help_key()),
            Span::styled(" cancel",    theme::help_text()),
        ]))
        .style(Style::default().bg(theme::BG_MODAL));
        frame.render_widget(hint, hint_area);

        list.render(frame, list_area);
    });
}

// ============================================================================
// Service Manager modal
// ============================================================================

fn draw_service_modal(frame: &mut Frame, full_area: Rect, chrome: &ShellChrome, selected: usize) {
    let modal = centered_rect(72, 60, full_area);
    render_modal(frame, modal, theme::modal_block(Line::from(" Services ")), |frame, inner| {
        let hint_area = Rect { y: inner.y + inner.height.saturating_sub(1), height: 1, ..inner };
        let list_area = Rect { height: inner.height.saturating_sub(2), ..inner };

        // Hint bar
        let hint = Paragraph::new(Line::from(vec![
            Span::styled("t", theme::help_key()),
            Span::styled(" start  ", theme::help_text()),
            Span::styled("s", theme::help_key()),
            Span::styled(" stop  ", theme::help_text()),
            Span::styled("r", theme::help_key()),
            Span::styled(" restart  ", theme::help_text()),
            Span::styled("a", theme::help_key()),
            Span::styled(" all start  ", theme::help_text()),
            Span::styled("S", theme::help_key()),
            Span::styled(" all stop  ", theme::help_text()),
            Span::styled("i", theme::help_key()),
            Span::styled(" install  ", theme::help_text()),
            Span::styled("Esc", theme::help_key()),
            Span::styled(" close", theme::help_text()),
        ]))
        .style(Style::default().bg(theme::BG_MODAL));
        frame.render_widget(hint, hint_area);

        if chrome.service_list.is_empty() {
            frame.render_widget(
                Paragraph::new(" (no services)")
                    .style(Style::default().fg(theme::DIM).bg(theme::BG_MODAL)),
                list_area,
            );
            return;
        }

        // Header
        let header_area = Rect { height: 1, ..list_area };
        let header = format!(
            " {:<20} {:<10} {:<10} {:>8}",
            "SERVICE", "STATE", "MODE", "PID"
        );
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(header, Style::default().fg(theme::CYAN).add_modifier(ratatui::style::Modifier::BOLD))))
                .style(Style::default().bg(theme::BG_MODAL)),
            header_area,
        );

        let rows_area = Rect {
            y: list_area.y + 1,
            height: list_area.height.saturating_sub(1),
            ..list_area
        };
        let visible = rows_area.height as usize;
        let n = chrome.service_list.len();
        let scroll_offset = if selected >= visible { selected - visible + 1 } else { 0 };

        for (i, svc) in chrome.service_list.iter().enumerate().skip(scroll_offset).take(visible) {
            let row_y = rows_area.y + (i - scroll_offset) as u16;
            let row_area = Rect { y: row_y, height: 1, ..rows_area };

            let indicator = if svc.active { "\u{25cf}" } else { "\u{25cb}" }; // ● or ○
            let state_str = if svc.active { "active" } else { "inactive" };
            let mode_str = match &svc.mode {
                ServiceMode::Systemd => "systemd",
                ServiceMode::Daemon  => "daemon",
                ServiceMode::Both    => "both",
                ServiceMode::Stopped => "stopped",
            };
            let pid_str = match svc.pid {
                Some(p) => format!("{}", p),
                None    => "-".to_string(),
            };
            let prefix = if i == selected { "\u{25b8} " } else { "  " }; // ▸
            let line_text = format!(
                "{}{} {:<20} {:<10} {:<10} {:>8}",
                prefix, indicator, svc.name, state_str, mode_str, pid_str
            );

            let (fg, bg) = if i == selected {
                (theme::AMBER, theme::BG_PANEL)
            } else if svc.active {
                (Color::Green, theme::BG_MODAL)
            } else {
                (theme::DIM, theme::BG_MODAL)
            };

            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(line_text, Style::default().fg(fg))))
                    .style(Style::default().bg(bg)),
                row_area,
            );
        }

        // Fill remaining rows
        let rendered = n.saturating_sub(scroll_offset).min(visible);
        for r in rendered..visible {
            let row_area = Rect { y: rows_area.y + r as u16, height: 1, ..rows_area };
            frame.render_widget(
                Paragraph::new("").style(Style::default().bg(theme::BG_MODAL)),
                row_area,
            );
        }
    });
}

// ============================================================================
// Worker Manager modal
// ============================================================================

fn draw_worker_modal(
    frame: &mut Frame, full_area: Rect, chrome: &ShellChrome,
    sandbox_sel: usize, container_sel: usize, show_containers: bool,
) {
    let modal = centered_rect(78, 65, full_area);
    render_modal(frame, modal, theme::modal_block(Line::from(" Workers ")), |frame, inner| {
        let hint_area = Rect { y: inner.y + inner.height.saturating_sub(1), height: 1, ..inner };

        // Pool summary line
        let summary_area = Rect { height: 1, ..inner };
        let summary_text = if chrome.worker_pool_summary.is_empty() {
            " Pool: (none)".to_string()
        } else {
            format!(" {}", chrome.worker_pool_summary)
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(summary_text, Style::default().fg(theme::DIM))))
                .style(Style::default().bg(theme::BG_MODAL)),
            summary_area,
        );

        let list_area = Rect {
            y: inner.y + 1,
            height: inner.height.saturating_sub(3), // -1 summary, -1 gap, -1 hint
            ..inner
        };

        if !show_containers {
            // Sandbox view
            let hint = Paragraph::new(Line::from(vec![
                Span::styled("\u{2191}\u{2193}", theme::help_key()),
                Span::styled(" nav  ", theme::help_text()),
                Span::styled("Enter/\u{2192}", theme::help_key()),
                Span::styled(" containers  ", theme::help_text()),
                Span::styled("x", theme::help_key()),
                Span::styled(" destroy  ", theme::help_text()),
                Span::styled("Esc", theme::help_key()),
                Span::styled(" close", theme::help_text()),
            ]))
            .style(Style::default().bg(theme::BG_MODAL));
            frame.render_widget(hint, hint_area);

            if chrome.worker_list.is_empty() {
                frame.render_widget(
                    Paragraph::new(" (no sandboxes)")
                        .style(Style::default().fg(theme::DIM).bg(theme::BG_MODAL)),
                    list_area,
                );
                return;
            }

            // Header
            let header_area = Rect { height: 1, ..list_area };
            let header = format!(
                " {:<12} {:<10} {:<10} {:>4} {:>5} {:>8}",
                "ID", "STATE", "BACKEND", "CTRS", "CPU%", "MEM MB"
            );
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(header, Style::default().fg(theme::CYAN).add_modifier(ratatui::style::Modifier::BOLD))))
                    .style(Style::default().bg(theme::BG_MODAL)),
                header_area,
            );

            let rows_area = Rect {
                y: list_area.y + 1,
                height: list_area.height.saturating_sub(1),
                ..list_area
            };
            let visible = rows_area.height as usize;
            let n = chrome.worker_list.len();
            let scroll_offset = if sandbox_sel >= visible { sandbox_sel - visible + 1 } else { 0 };

            for (i, sb) in chrome.worker_list.iter().enumerate().skip(scroll_offset).take(visible) {
                let row_y = rows_area.y + (i - scroll_offset) as u16;
                let row_area = Rect { y: row_y, height: 1, ..rows_area };

                let cpu_str = match sb.cpu_pct { Some(c) => format!("{}%", c), None => "-".to_string() };
                let mem_str = match sb.mem_mb  { Some(m) => format!("{}", m),  None => "-".to_string() };
                let prefix = if i == sandbox_sel { "\u{25b8} " } else { "  " };
                let line_text = format!(
                    "{}{:<12} {:<10} {:<10} {:>4} {:>5} {:>8}",
                    prefix, sb.id, sb.state, sb.backend, sb.containers.len(), cpu_str, mem_str
                );

                let (fg, bg) = if i == sandbox_sel {
                    (theme::AMBER, theme::BG_PANEL)
                } else {
                    (theme::DIM, theme::BG_MODAL)
                };

                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(line_text, Style::default().fg(fg))))
                        .style(Style::default().bg(bg)),
                    row_area,
                );
            }

            let rendered = n.saturating_sub(scroll_offset).min(visible);
            for r in rendered..visible {
                let row_area = Rect { y: rows_area.y + r as u16, height: 1, ..rows_area };
                frame.render_widget(
                    Paragraph::new("").style(Style::default().bg(theme::BG_MODAL)),
                    row_area,
                );
            }
        } else {
            // Container view for selected sandbox
            let hint = Paragraph::new(Line::from(vec![
                Span::styled("\u{2191}\u{2193}", theme::help_key()),
                Span::styled(" nav  ", theme::help_text()),
                Span::styled("\u{2190}", theme::help_key()),
                Span::styled(" back  ", theme::help_text()),
                Span::styled("e", theme::help_key()),
                Span::styled(" exec  ", theme::help_text()),
                Span::styled("Esc", theme::help_key()),
                Span::styled(" close", theme::help_text()),
            ]))
            .style(Style::default().bg(theme::BG_MODAL));
            frame.render_widget(hint, hint_area);

            let sb = match chrome.worker_list.get(sandbox_sel) {
                Some(sb) => sb,
                None => return,
            };

            // Sub-header with sandbox ID
            let sub_area = Rect { height: 1, ..list_area };
            let sub_text = format!(" Sandbox: {} ({} containers)", sb.id, sb.containers.len());
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(sub_text, Style::default().fg(theme::CYAN))))
                    .style(Style::default().bg(theme::BG_MODAL)),
                sub_area,
            );

            // Column header
            let hdr_area = Rect { y: list_area.y + 1, height: 1, ..list_area };
            let header = format!(
                " {:<12} {:<20} {:<10} {:>5} {:>8}",
                "ID", "IMAGE", "STATE", "CPU%", "MEM MB"
            );
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(header, Style::default().fg(theme::CYAN).add_modifier(ratatui::style::Modifier::BOLD))))
                    .style(Style::default().bg(theme::BG_MODAL)),
                hdr_area,
            );

            let rows_area = Rect {
                y: list_area.y + 2,
                height: list_area.height.saturating_sub(2),
                ..list_area
            };
            let visible = rows_area.height as usize;
            let cn = sb.containers.len();
            let scroll_offset = if container_sel >= visible { container_sel - visible + 1 } else { 0 };

            if cn == 0 {
                frame.render_widget(
                    Paragraph::new(" (no containers)")
                        .style(Style::default().fg(theme::DIM).bg(theme::BG_MODAL)),
                    rows_area,
                );
                return;
            }

            for (i, ctr) in sb.containers.iter().enumerate().skip(scroll_offset).take(visible) {
                let row_y = rows_area.y + (i - scroll_offset) as u16;
                let row_area = Rect { y: row_y, height: 1, ..rows_area };

                let cpu_str = match ctr.cpu_pct { Some(c) => format!("{}%", c), None => "-".to_string() };
                let mem_str = match ctr.mem_mb  { Some(m) => format!("{}", m),  None => "-".to_string() };
                let img = if ctr.image.len() > 18 { &ctr.image[..18] } else { &ctr.image };
                let prefix = if i == container_sel { "\u{25b8} " } else { "  " };
                let line_text = format!(
                    "{}{:<12} {:<20} {:<10} {:>5} {:>8}",
                    prefix, ctr.id, img, ctr.state, cpu_str, mem_str
                );

                let (fg, bg) = if i == container_sel {
                    (theme::AMBER, theme::BG_PANEL)
                } else {
                    (theme::DIM, theme::BG_MODAL)
                };

                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(line_text, Style::default().fg(fg))))
                        .style(Style::default().bg(bg)),
                    row_area,
                );
            }

            let rendered = cn.saturating_sub(scroll_offset).min(visible);
            for r in rendered..visible {
                let row_area = Rect { y: rows_area.y + r as u16, height: 1, ..rows_area };
                frame.render_widget(
                    Paragraph::new("").style(Style::default().bg(theme::BG_MODAL)),
                    row_area,
                );
            }
        }
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

// ============================================================================
// Render smoke tests — verify no panics
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use crate::chrome::{
        ContainerEntry, ServiceEntry, ServiceMode, ShellMode, WorkerEntry,
    };
    use crate::Compositor;

    fn make_compositor() -> Compositor {
        Compositor::new(120, 40, 1, 1, vec![], vec![])
    }

    fn with_services(comp: &mut Compositor) {
        comp.chrome.service_list = vec![
            ServiceEntry { name: "oai".into(), active: true, mode: ServiceMode::Systemd, pid: Some(1234) },
            ServiceEntry { name: "registry".into(), active: false, mode: ServiceMode::Stopped, pid: None },
            ServiceEntry { name: "model".into(), active: true, mode: ServiceMode::Daemon, pid: Some(5678) },
        ];
    }

    fn with_workers(comp: &mut Compositor) {
        comp.chrome.worker_list = vec![WorkerEntry {
            id: "abc123".into(), full_id: "abc12345".into(),
            state: "Ready".into(), backend: "kata".into(),
            cpu_pct: Some(25), mem_mb: Some(512),
            containers: vec![
                ContainerEntry {
                    id: "ctr1".into(), full_id: "ctr1-full".into(),
                    image: "hyprstream:latest".into(), state: "Running".into(),
                    cpu_pct: Some(10), mem_mb: Some(256),
                },
            ],
        }];
        comp.chrome.worker_pool_summary = "1 sandbox, 1 container".into();
    }

    /// Render the compositor into a test terminal and return the buffer as a string.
    fn render_to_string(comp: &Compositor) -> String {
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|f| comp.render(f)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push(buf[(x, y)].symbol().chars().next().unwrap_or(' '));
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn render_normal_mode_no_panic() {
        let comp = make_compositor();
        let text = render_to_string(&comp);
        assert!(text.contains("HYPRSTREAM"));
    }

    #[test]
    fn render_service_modal_empty() {
        let mut comp = make_compositor();
        comp.chrome.mode = ShellMode::ServiceManager { selected: 0 };
        let text = render_to_string(&comp);
        assert!(text.contains("Services"));
        assert!(text.contains("no services"));
    }

    #[test]
    fn render_service_modal_with_entries() {
        let mut comp = make_compositor();
        with_services(&mut comp);
        comp.chrome.mode = ShellMode::ServiceManager { selected: 1 };
        let text = render_to_string(&comp);
        assert!(text.contains("Services"));
        assert!(text.contains("oai"));
        assert!(text.contains("registry"));
        assert!(text.contains("model"));
    }

    #[test]
    fn render_service_modal_selection_beyond_list() {
        let mut comp = make_compositor();
        with_services(&mut comp);
        // selected index out of range — should not panic
        comp.chrome.mode = ShellMode::ServiceManager { selected: 100 };
        let _text = render_to_string(&comp);
    }

    #[test]
    fn render_worker_modal_empty() {
        let mut comp = make_compositor();
        comp.chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: false };
        let text = render_to_string(&comp);
        assert!(text.contains("Workers"));
        assert!(text.contains("no sandboxes"));
    }

    #[test]
    fn render_worker_modal_sandbox_view() {
        let mut comp = make_compositor();
        with_workers(&mut comp);
        comp.chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: false };
        let text = render_to_string(&comp);
        assert!(text.contains("Workers"));
        assert!(text.contains("abc123"));
        assert!(text.contains("kata"));
    }

    #[test]
    fn render_worker_modal_container_view() {
        let mut comp = make_compositor();
        with_workers(&mut comp);
        comp.chrome.mode = ShellMode::WorkerManager { sandbox_sel: 0, container_sel: 0, show_containers: true };
        let text = render_to_string(&comp);
        assert!(text.contains("Workers"));
        assert!(text.contains("ctr1"));
        assert!(text.contains("hyprstream:latest"));
    }

    #[test]
    fn render_worker_modal_selection_beyond_list() {
        let mut comp = make_compositor();
        with_workers(&mut comp);
        comp.chrome.mode = ShellMode::WorkerManager { sandbox_sel: 99, container_sel: 99, show_containers: true };
        let _text = render_to_string(&comp);
    }

    #[test]
    fn render_small_terminal() {
        // Very small terminal — modals should not panic even if area is tiny
        let comp = Compositor::new(20, 5, 1, 1, vec![], vec![]);
        let backend = ratatui::backend::TestBackend::new(20, 5);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|f| comp.render(f)).unwrap();
    }

    #[test]
    fn render_service_modal_small_terminal() {
        let mut comp = Compositor::new(30, 10, 1, 1, vec![], vec![]);
        comp.chrome.service_list = vec![
            ServiceEntry { name: "oai".into(), active: true, mode: ServiceMode::Systemd, pid: Some(1) },
        ];
        comp.chrome.mode = ShellMode::ServiceManager { selected: 0 };
        let backend = ratatui::backend::TestBackend::new(30, 10);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|f| comp.render(f)).unwrap();
    }
}
