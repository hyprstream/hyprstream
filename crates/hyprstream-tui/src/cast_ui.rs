//! Rendering functions for the cast player UI.
//!
//! Draws the playback area (avt cells → ratatui spans), progress bar,
//! title bar, tab bar, and help line.

use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Tabs},
    Frame,
};

use crate::cast_app::CastPlayerApp;
use crate::theme;

/// Top-level draw function for the cast player.
pub fn draw(frame: &mut Frame, app: &CastPlayerApp) {
    let area = frame.area();

    let chunks = Layout::vertical([
        Constraint::Length(1), // title bar
        Constraint::Length(3), // tab bar
        Constraint::Min(8),   // playback area
        Constraint::Length(1), // progress bar
        Constraint::Length(1), // help line
    ])
    .split(area);

    draw_title(frame, chunks[0]);
    draw_tabs(frame, chunks[1], app);
    draw_playback(frame, chunks[2], app);
    draw_progress(frame, chunks[3], app);
    draw_help(frame, chunks[4], app);
}

fn draw_title(frame: &mut Frame, area: Rect) {
    let title = Paragraph::new(Line::from(vec![
        Span::styled(" HYPRSTREAM", theme::title_style()),
        Span::styled(" // ", Style::default().fg(theme::DIM)),
        Span::styled("CAST PLAYER", theme::title_style()),
    ]))
    .style(Style::default().bg(theme::BG_PANEL));

    frame.render_widget(title, area);
}

fn draw_tabs(frame: &mut Frame, area: Rect, app: &CastPlayerApp) {
    let titles: Vec<Line> = app
        .logs
        .iter()
        .enumerate()
        .map(|(i, log)| {
            let label = format!(" LOG {:03}  {} ", i + 1, log.name);
            if i == app.selected {
                Line::from(Span::styled(label, theme::tab_active()))
            } else {
                Line::from(Span::styled(label, theme::tab_inactive()))
            }
        })
        .collect();

    let tabs = Tabs::new(titles)
        .select(app.selected)
        .highlight_style(theme::tab_active())
        .divider(Span::styled(" | ", Style::default().fg(theme::DIM)))
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(theme::border_style()),
        );

    frame.render_widget(tabs, area);
}

fn draw_playback(frame: &mut Frame, area: Rect, app: &CastPlayerApp) {
    let block = Block::default()
        .title(Span::styled(
            " PLAYBACK ",
            Style::default()
                .fg(theme::CYAN)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(theme::border_style())
        .style(Style::default().bg(theme::BG));

    let inner = block.inner(area);
    frame.render_widget(block, area);
    hyprstream_compositor::render::draw_vt_cells(frame, inner, app.player.vt());
}

fn draw_progress(frame: &mut Frame, area: Rect, app: &CastPlayerApp) {
    let progress = app.player.progress();
    let elapsed = app.player.elapsed_display();
    let duration = app.player.duration_display();

    let mode_str = if app.player.paused {
        "PAUSED"
    } else {
        "PLAYING"
    };

    let label = format!("{} {} / {}", mode_str, elapsed, duration);

    let gauge = Gauge::default()
        .gauge_style(theme::progress_filled())
        .label(Span::styled(label, theme::status_style()))
        .ratio(progress)
        .use_unicode(true);

    frame.render_widget(gauge, area);
}

fn draw_help(frame: &mut Frame, area: Rect, app: &CastPlayerApp) {
    let log_name = app
        .logs
        .get(app.selected)
        .map(|l| l.name.as_str())
        .unwrap_or("—");

    let help = Line::from(vec![
        Span::styled(" [", theme::help_text()),
        Span::styled("Tab", theme::help_key()),
        Span::styled("] Switch  [", theme::help_text()),
        Span::styled("Space", theme::help_key()),
        Span::styled("] Pause  [", theme::help_text()),
        Span::styled("</>", theme::help_key()),
        Span::styled("] Seek  [", theme::help_text()),
        Span::styled("q", theme::help_key()),
        Span::styled("] Quit", theme::help_text()),
        Span::styled(
            format!("  // {}", log_name),
            Style::default().fg(theme::DIM),
        ),
    ]);

    let paragraph = Paragraph::new(help).style(Style::default().bg(theme::BG_PANEL));
    frame.render_widget(paragraph, area);
}
