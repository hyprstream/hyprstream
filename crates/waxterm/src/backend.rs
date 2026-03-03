//! Minimal ANSI terminal backend for ratatui.
//! Works under WASI by writing escape sequences directly to stdout.
//! No platform-specific APIs needed.

use std::io::{self, Write};

use ratatui::{
    backend::{Backend, ClearType, WindowSize},
    buffer::Cell,
    layout::{Position, Size},
    style::{Color, Modifier},
};

/// A backend that writes ANSI escape codes to any `Write` sink.
pub struct AnsiBackend<W: Write> {
    writer: W,
    width: u16,
    height: u16,
    /// Optional Wanix pipe heartbeat bytes (60 cursor-move sequences).
    /// Built once in `new()` to avoid 60 format calls per frame.
    /// None on native terminals where heartbeat is unnecessary.
    heartbeat: Option<Box<[u8]>>,
}

impl<W: Write> AnsiBackend<W> {
    pub fn new(writer: W, width: u16, height: u16, heartbeat: bool) -> Self {
        let heartbeat = if heartbeat {
            Some(
                (1u16..=60)
                    .flat_map(|col| format!("\x1b[{};{}H", height, col).into_bytes())
                    .collect(),
            )
        } else {
            None
        };
        AnsiBackend {
            writer,
            width,
            height,
            heartbeat,
        }
    }

    fn write_csi(&mut self, args: &str) -> io::Result<()> {
        write!(self.writer, "\x1b[{}", args)
    }
}

impl<W: Write> Backend for AnsiBackend<W> {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> Result<(), Self::Error>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        let mut last_pos: Option<(u16, u16)> = None;
        let mut last_fg: Option<Color> = None;
        let mut last_bg: Option<Color> = None;
        let mut last_mods: Option<Modifier> = None;

        for (x, y, cell) in content {
            let need_move = match last_pos {
                Some((lx, ly)) => !(ly == y && lx + 1 == x),
                None => true,
            };

            if need_move {
                write!(self.writer, "\x1b[{};{}H", y + 1, x + 1)?;
            }

            // Emit a single combined SGR sequence when any style component changes,
            // instead of separate reset + modifiers + fg + bg sequences.
            let fg_changed = last_fg != Some(cell.fg);
            let bg_changed = last_bg != Some(cell.bg);
            let mods_changed = last_mods != Some(cell.modifier);

            if fg_changed || bg_changed || mods_changed {
                write_combined_sgr(&mut self.writer, cell.fg, cell.bg, cell.modifier)?;
                last_fg = Some(cell.fg);
                last_bg = Some(cell.bg);
                last_mods = Some(cell.modifier);
            }

            self.writer.write_all(cell.symbol().as_bytes())?;
            last_pos = Some((x, y));
        }

        self.write_csi("0m")?;

        // Wanix VFS pipe heartbeat: cursor moves on the last row.
        //
        // The Wanix pipe accumulates small writes and delivers them in 1024-byte
        // chunks to the JS ReadableStream reader. During idle periods (no cell
        // changes), each render frame produces only the 4-byte SGR reset above.
        // At 30fps this gives ~120 bytes/s — far below the 1024-byte delivery
        // threshold — causing multi-second write gaps.
        //
        // Cursor moves to different columns are each ~9 bytes and have no visual
        // effect when the cursor is hidden. Writing 60 of them adds ~540 bytes,
        // bringing idle frames closer to the 1024-byte threshold.
        if let Some(ref hb) = self.heartbeat {
            self.writer.write_all(hb)?;
        }

        self.writer.flush()
    }

    fn hide_cursor(&mut self) -> Result<(), Self::Error> {
        self.write_csi("?25l")?;
        self.writer.flush()
    }

    fn show_cursor(&mut self) -> Result<(), Self::Error> {
        self.write_csi("?25h")?;
        self.writer.flush()
    }

    fn get_cursor_position(&mut self) -> Result<Position, Self::Error> {
        Ok(Position::new(0, 0))
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> Result<(), Self::Error> {
        let pos = position.into();
        write!(self.writer, "\x1b[{};{}H", pos.y + 1, pos.x + 1)?;
        self.writer.flush()
    }

    fn clear(&mut self) -> Result<(), Self::Error> {
        self.write_csi("2J")?;
        self.write_csi("H")?;
        self.writer.flush()
    }

    fn clear_region(&mut self, clear_type: ClearType) -> Result<(), Self::Error> {
        match clear_type {
            ClearType::All => self.write_csi("2J")?,
            ClearType::AfterCursor => self.write_csi("0J")?,
            ClearType::BeforeCursor => self.write_csi("1J")?,
            ClearType::CurrentLine => self.write_csi("2K")?,
            ClearType::UntilNewLine => self.write_csi("0K")?,
        }
        self.writer.flush()
    }

    fn size(&self) -> Result<Size, Self::Error> {
        Ok(Size::new(self.width, self.height))
    }

    fn window_size(&mut self) -> Result<WindowSize, Self::Error> {
        Ok(WindowSize {
            columns_rows: Size::new(self.width, self.height),
            pixels: Size::new(0, 0),
        })
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.writer.flush()
    }
}

/// Emit a single combined SGR sequence: `\x1b[0;...m`
/// Resets all attributes then applies modifiers, fg, and bg in one escape.
fn write_combined_sgr<W: Write>(w: &mut W, fg: Color, bg: Color, mods: Modifier) -> io::Result<()> {
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
fn write_fg_params<W: Write>(w: &mut W, color: Color) -> io::Result<()> {
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
fn write_bg_params<W: Write>(w: &mut W, color: Color) -> io::Result<()> {
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
