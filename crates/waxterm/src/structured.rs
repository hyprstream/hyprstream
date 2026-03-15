//! Packed binary cell backend for ratatui.
//!
//! Implements `ratatui::backend::Backend` with a binary wire format instead of
//! ANSI escape sequences. Each cell is packed into ~14 bytes:
//!
//! ```text
//! [x: u16] [y: u16] [symbol_len: u8] [symbol: 1-4 bytes] [fg: u32] [bg: u32] [modifiers: u16]
//! ```
//!
//! The TUI display server deserializes directly into `PaneBuffer` cells,
//! bypassing VTE parsing (~30ns/cell vs ~120ns/cell for ANSI→VTE).

use std::io::{self, Write};

use ratatui::{
    backend::{Backend, ClearType, WindowSize},
    buffer::Cell,
    layout::{Position, Size},
    style::{Color, Modifier},
};

/// Magic header bytes identifying a structured frame.
pub const STRUCTURED_MAGIC: [u8; 4] = [0x54, 0x55, 0x49, 0x53]; // "TUIS"

/// A backend that writes packed binary cell data to a sink.
pub struct StructuredBackend<W: Write> {
    writer: W,
    width: u16,
    height: u16,
}

impl<W: Write> StructuredBackend<W> {
    pub fn new(writer: W, width: u16, height: u16) -> Self {
        Self { writer, width, height }
    }

    /// Get the underlying writer.
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Get a mutable reference to the underlying writer.
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }
}

impl<W: Write> Backend for StructuredBackend<W> {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> Result<(), Self::Error>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        // Write magic header
        self.writer.write_all(&STRUCTURED_MAGIC)?;

        // Write dimensions
        self.writer.write_all(&self.width.to_le_bytes())?;
        self.writer.write_all(&self.height.to_le_bytes())?;

        for (x, y, cell) in content {
            // Position
            self.writer.write_all(&x.to_le_bytes())?;
            self.writer.write_all(&y.to_le_bytes())?;

            // Symbol (length-prefixed, UTF-8)
            let symbol = cell.symbol().as_bytes();
            let len = symbol.len().min(255) as u8;
            self.writer.write_all(&[len])?;
            self.writer.write_all(&symbol[..len as usize])?;

            // Colors
            self.writer.write_all(&pack_color(cell.fg).to_le_bytes())?;
            self.writer.write_all(&pack_color(cell.bg).to_le_bytes())?;

            // Modifiers
            self.writer.write_all(&(cell.modifier.bits()).to_le_bytes())?;
        }

        self.writer.flush()
    }

    fn hide_cursor(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn show_cursor(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn get_cursor_position(&mut self) -> Result<Position, Self::Error> {
        Ok(Position::new(0, 0))
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, _position: P) -> Result<(), Self::Error> {
        Ok(())
    }

    fn clear(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn clear_region(&mut self, _clear_type: ClearType) -> Result<(), Self::Error> {
        Ok(())
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

/// Pack a ratatui Color into a u32 for wire format.
///
/// Encoding: `0x01_RR_GG_BB` for RGB, `0x02_00_00_II` for indexed, ordinal for named.
pub fn pack_color(color: Color) -> u32 {
    match color {
        Color::Reset => 0,
        Color::Black => 1,
        Color::Red => 2,
        Color::Green => 3,
        Color::Yellow => 4,
        Color::Blue => 5,
        Color::Magenta => 6,
        Color::Cyan => 7,
        Color::Gray => 8,
        Color::DarkGray => 9,
        Color::LightRed => 10,
        Color::LightGreen => 11,
        Color::LightYellow => 12,
        Color::LightBlue => 13,
        Color::LightMagenta => 14,
        Color::LightCyan => 15,
        Color::White => 16,
        Color::Rgb(r, g, b) => 0x0100_0000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32),
        Color::Indexed(i) => 0x0200_0000 | (i as u32),
    }
}

/// Unpack a u32 into a ratatui Color.
pub fn unpack_color(packed: u32) -> Color {
    if packed & 0x0200_0000 != 0 {
        Color::Indexed((packed & 0xFF) as u8)
    } else if packed & 0x0100_0000 != 0 {
        Color::Rgb(
            ((packed >> 16) & 0xFF) as u8,
            ((packed >> 8) & 0xFF) as u8,
            (packed & 0xFF) as u8,
        )
    } else {
        match packed {
            0 => Color::Reset,
            1 => Color::Black,
            2 => Color::Red,
            3 => Color::Green,
            4 => Color::Yellow,
            5 => Color::Blue,
            6 => Color::Magenta,
            7 => Color::Cyan,
            8 => Color::Gray,
            9 => Color::DarkGray,
            10 => Color::LightRed,
            11 => Color::LightGreen,
            12 => Color::LightYellow,
            13 => Color::LightBlue,
            14 => Color::LightMagenta,
            15 => Color::LightCyan,
            16 => Color::White,
            _ => Color::Reset,
        }
    }
}

/// Decode structured cells from a binary buffer.
///
/// Returns `(cols, rows, cells)` where cells is a vec of `(x, y, symbol, fg, bg, modifiers)`.
pub fn decode_structured(data: &[u8]) -> io::Result<(u16, u16, Vec<DecodedCell>)> {
    if data.len() < 8 || data[..4] != STRUCTURED_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid structured frame magic"));
    }

    let cols = u16::from_le_bytes([data[4], data[5]]);
    let rows = u16::from_le_bytes([data[6], data[7]]);
    let mut cells = Vec::new();
    let mut pos = 8;

    while pos < data.len() {
        if pos + 5 > data.len() {
            break;
        }
        let x = u16::from_le_bytes([data[pos], data[pos + 1]]);
        let y = u16::from_le_bytes([data[pos + 2], data[pos + 3]]);
        let sym_len = data[pos + 4] as usize;
        pos += 5;

        if pos + sym_len + 10 > data.len() {
            break;
        }
        let symbol = String::from_utf8_lossy(&data[pos..pos + sym_len]).to_string();
        pos += sym_len;

        let fg = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let bg = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let mods = u16::from_le_bytes([data[pos], data[pos + 1]]);
        pos += 2;

        cells.push(DecodedCell {
            x,
            y,
            symbol,
            fg: unpack_color(fg),
            bg: unpack_color(bg),
            modifiers: Modifier::from_bits_retain(mods),
        });
    }

    Ok((cols, rows, cells))
}

/// A decoded cell from structured binary data.
#[derive(Debug, Clone)]
pub struct DecodedCell {
    pub x: u16,
    pub y: u16,
    pub symbol: String,
    pub fg: Color,
    pub bg: Color,
    pub modifiers: Modifier,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::style::Style;

    #[test]
    fn test_binary_layout() {
        let mut buf = Vec::new();
        let mut backend = StructuredBackend::new(&mut buf, 2, 1);

        let rect = Rect::new(0, 0, 2, 1);
        let mut buffer = Buffer::empty(rect);
        buffer[(0, 0)].set_symbol("A");
        buffer[(1, 0)].set_symbol("B");

        let cells: Vec<_> = buffer.content().iter().enumerate().map(|(i, cell)| {
            let x = (i % 2) as u16;
            let y = (i / 2) as u16;
            (x, y, cell)
        }).collect();

        backend.draw(cells.into_iter()).unwrap();

        // Verify magic header
        assert_eq!(&buf[..4], &STRUCTURED_MAGIC);
        // Verify dimensions
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), 2);
        assert_eq!(u16::from_le_bytes([buf[6], buf[7]]), 1);
    }

    #[test]
    fn test_roundtrip() {
        let mut buf = Vec::new();
        let mut backend = StructuredBackend::new(&mut buf, 3, 1);

        let rect = Rect::new(0, 0, 3, 1);
        let mut buffer = Buffer::empty(rect);
        buffer[(0, 0)].set_symbol("X").set_style(Style::new().fg(Color::Red).bg(Color::Blue));
        buffer[(1, 0)].set_symbol("Y");
        buffer[(2, 0)].set_symbol("Z").set_style(Style::new().fg(Color::Rgb(255, 128, 0)));

        let cells: Vec<_> = buffer.content().iter().enumerate().map(|(i, cell)| {
            let x = (i % 3) as u16;
            let y = (i / 3) as u16;
            (x, y, cell)
        }).collect();

        backend.draw(cells.into_iter()).unwrap();

        let (cols, rows, decoded) = decode_structured(&buf).unwrap();
        assert_eq!(cols, 3);
        assert_eq!(rows, 1);
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].symbol, "X");
        assert_eq!(decoded[0].fg, Color::Red);
        assert_eq!(decoded[0].bg, Color::Blue);
        assert_eq!(decoded[2].fg, Color::Rgb(255, 128, 0));
    }

    #[test]
    fn test_color_roundtrip() {
        let colors = [
            Color::Reset, Color::Black, Color::Red, Color::White,
            Color::Rgb(255, 128, 0), Color::Indexed(42),
        ];
        for c in &colors {
            assert_eq!(unpack_color(pack_color(*c)), *c, "roundtrip failed for {:?}", c);
        }
    }

    #[test]
    fn test_size_returns_config() {
        let buf = Vec::new();
        let backend = StructuredBackend::new(buf, 120, 40);
        assert_eq!(backend.size().unwrap(), Size::new(120, 40));
    }
}
