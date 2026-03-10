//! Color constants and reusable styles for hyprstream TUI applications.

use ratatui::style::{Color, Modifier, Style};

// ── Palette ──
pub const AMBER: Color = Color::Rgb(236, 201, 75);
pub const CYAN: Color = Color::Rgb(0, 232, 252);
pub const DIM: Color = Color::Rgb(112, 128, 160);
pub const BG: Color = Color::Rgb(10, 10, 20);
pub const BG_PANEL: Color = Color::Rgb(20, 20, 40);
/// Per-window titlebar background — distinct hue from BG_PANEL to visually
/// separate the window title row from the global status and F-key bars.
pub const BG_TITLEBAR: Color = Color::Rgb(10, 22, 48);

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
    Style::default().fg(CYAN).bg(BG_TITLEBAR).add_modifier(Modifier::BOLD)
}

pub fn titlebar_dim_style() -> Style {
    Style::default().fg(DIM).bg(BG_TITLEBAR)
}

pub fn help_key() -> Style {
    Style::default().fg(AMBER).add_modifier(Modifier::BOLD)
}

pub fn help_text() -> Style {
    Style::default().fg(DIM)
}
