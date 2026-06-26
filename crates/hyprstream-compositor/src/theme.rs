//! Color constants and reusable styles for hyprstream TUI applications.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders};

// ── Palette ──
pub const AMBER: Color = Color::Rgb(236, 201, 75);
pub const CYAN: Color = Color::Rgb(0, 232, 252);
pub const DIM: Color = Color::Rgb(112, 128, 160);
pub const BG: Color = Color::Rgb(10, 10, 20);
pub const BG_PANEL: Color = Color::Rgb(20, 20, 40);
/// Floating modal background — slightly lighter than BG for visual separation.
pub const BG_MODAL: Color = Color::Rgb(16, 16, 34);
/// Lock icon color in window titlebars.
pub const LOCK_COLOR: Color = Color::Rgb(136, 170, 255);

// ── Block factories ──

/// Width of the close button " x " rendered in block titlebars (including padding).
pub const CLOSE_BUTTON_WIDTH: u16 = 3;

fn close_button() -> Line<'static> {
    Line::from(Span::styled(" x ", Style::default().fg(DIM))).right_aligned()
}

/// Rounded border block for pane windows, with focus-sensitive border color.
pub fn window_block(title: Line<'_>, focused: bool) -> Block<'_> {
    Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(if focused { border_focused_style() } else { border_style() })
        .title(title)
        .title_top(close_button())
        .style(Style::default().bg(BG))
}

/// Rounded border block for floating modals.
pub fn modal_block(title: Line<'_>) -> Block<'_> {
    Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_focused_style())
        .title(title)
        .title_top(close_button())
        .style(Style::default().bg(BG_MODAL))
}

/// Rounded border block for the multi-line input box in ChatApp.
pub fn input_block() -> Block<'static> {
    Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(input_border_style())
        .style(Style::default().bg(BG_PANEL))
}

pub fn border_focused_style() -> Style {
    Style::default().fg(CYAN)
}

pub fn input_border_style() -> Style {
    Style::default().fg(Color::Rgb(68, 85, 102))
}

pub fn gutter_style() -> Style {
    Style::default().fg(DIM)
}

// ── Reusable styles ──

pub fn title_style() -> Style {
    Style::default().fg(AMBER).add_modifier(Modifier::BOLD)
}

pub fn tab_active() -> Style {
    Style::default()
        .fg(AMBER)
        .bg(BG_PANEL)
        .add_modifier(Modifier::BOLD)
}

pub fn tab_inactive() -> Style {
    Style::default().fg(DIM).bg(BG)
}

pub fn status_style() -> Style {
    Style::default().fg(DIM)
}

pub fn progress_filled() -> Style {
    Style::default().fg(AMBER).bg(BG_PANEL)
}

pub fn border_style() -> Style {
    Style::default().fg(DIM)
}

pub fn titlebar_style() -> Style {
    Style::default().fg(CYAN).add_modifier(Modifier::BOLD)
}

pub fn titlebar_dim_style() -> Style {
    Style::default().fg(DIM)
}

pub fn help_key() -> Style {
    Style::default().fg(AMBER).add_modifier(Modifier::BOLD)
}

pub fn help_text() -> Style {
    Style::default().fg(DIM)
}
