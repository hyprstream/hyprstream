//! SGR (Select Graphic Rendition) utility functions.
//!
//! Extracted from `backend.rs` for reuse by the TUI diff engine and
//! StructuredBackend. Provides combined SGR sequence emission and
//! individual color parameter formatting.

use std::io::{self, Write};

use ratatui::style::{Color, Modifier};

/// Emit a single combined SGR sequence: `\x1b[0;...m`
/// Resets all attributes then applies modifiers, fg, and bg in one escape.
pub fn write_combined_sgr<W: Write>(w: &mut W, fg: Color, bg: Color, mods: Modifier) -> io::Result<()> {
    // Start with reset
    write!(w, "\x1b[0")?;

    // Modifiers
    if mods.contains(Modifier::BOLD) {
        write!(w, ";1")?;
    }
    if mods.contains(Modifier::DIM) {
        write!(w, ";2")?;
    }
    if mods.contains(Modifier::ITALIC) {
        write!(w, ";3")?;
    }
    if mods.contains(Modifier::UNDERLINED) {
        write!(w, ";4")?;
    }
    if mods.contains(Modifier::REVERSED) {
        write!(w, ";7")?;
    }
    if mods.contains(Modifier::CROSSED_OUT) {
        write!(w, ";9")?;
    }

    // Foreground
    write_fg_params(w, fg)?;

    // Background
    write_bg_params(w, bg)?;

    // Close the sequence
    write!(w, "m")
}

/// Append foreground color parameters to an in-progress SGR sequence.
/// Color::Reset omitted (already handled by the leading `0` reset).
pub fn write_fg_params<W: Write>(w: &mut W, color: Color) -> io::Result<()> {
    match color {
        Color::Reset => {}
        Color::Black => write!(w, ";30")?,
        Color::Red => write!(w, ";31")?,
        Color::Green => write!(w, ";32")?,
        Color::Yellow => write!(w, ";33")?,
        Color::Blue => write!(w, ";34")?,
        Color::Magenta => write!(w, ";35")?,
        Color::Cyan => write!(w, ";36")?,
        Color::Gray => write!(w, ";37")?,
        Color::DarkGray => write!(w, ";90")?,
        Color::LightRed => write!(w, ";91")?,
        Color::LightGreen => write!(w, ";92")?,
        Color::LightYellow => write!(w, ";93")?,
        Color::LightBlue => write!(w, ";94")?,
        Color::LightMagenta => write!(w, ";95")?,
        Color::LightCyan => write!(w, ";96")?,
        Color::White => write!(w, ";97")?,
        Color::Rgb(r, g, b) => write!(w, ";38;2;{};{};{}", r, g, b)?,
        Color::Indexed(i) => write!(w, ";38;5;{}", i)?,
    }
    Ok(())
}

/// Append background color parameters to an in-progress SGR sequence.
pub fn write_bg_params<W: Write>(w: &mut W, color: Color) -> io::Result<()> {
    match color {
        Color::Reset => {}
        Color::Black => write!(w, ";40")?,
        Color::Red => write!(w, ";41")?,
        Color::Green => write!(w, ";42")?,
        Color::Yellow => write!(w, ";43")?,
        Color::Blue => write!(w, ";44")?,
        Color::Magenta => write!(w, ";45")?,
        Color::Cyan => write!(w, ";46")?,
        Color::Gray => write!(w, ";47")?,
        Color::DarkGray => write!(w, ";100")?,
        Color::LightRed => write!(w, ";101")?,
        Color::LightGreen => write!(w, ";102")?,
        Color::LightYellow => write!(w, ";103")?,
        Color::LightBlue => write!(w, ";104")?,
        Color::LightMagenta => write!(w, ";105")?,
        Color::LightCyan => write!(w, ";106")?,
        Color::White => write!(w, ";107")?,
        Color::Rgb(r, g, b) => write!(w, ";48;2;{};{};{}", r, g, b)?,
        Color::Indexed(i) => write!(w, ";48;5;{}", i)?,
    }
    Ok(())
}
