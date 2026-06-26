//! Conversion utilities from `avt::Pen` / `avt::Color` to `ratatui::style`.

use ratatui::style::{Color, Modifier, Style};

/// Convert an `avt::Pen` (foreground, background, bold, etc.) to a `ratatui::Style`.
pub fn avt_pen_to_style(pen: &avt::Pen) -> Style {
    let mut style = Style::default();

    if let Some(fg) = pen.foreground() {
        style = style.fg(avt_color_to_ratatui(fg));
    }
    if let Some(bg) = pen.background() {
        style = style.bg(avt_color_to_ratatui(bg));
    }
    if pen.is_bold() {
        style = style.add_modifier(Modifier::BOLD);
    }
    if pen.is_italic() {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if pen.is_underline() {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    if pen.is_strikethrough() {
        style = style.add_modifier(Modifier::CROSSED_OUT);
    }
    if pen.is_inverse() {
        style = style.add_modifier(Modifier::REVERSED);
    }
    if pen.is_faint() {
        style = style.add_modifier(Modifier::DIM);
    }

    style
}

/// Convert an `avt::Color` to a `ratatui::style::Color`.
pub fn avt_color_to_ratatui(color: avt::Color) -> Color {
    match color {
        avt::Color::Indexed(i) => Color::Indexed(i),
        avt::Color::RGB(rgb) => Color::Rgb(rgb.r, rgb.g, rgb.b),
    }
}
