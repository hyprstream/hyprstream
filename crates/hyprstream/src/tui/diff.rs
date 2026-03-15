//! Diff engine for TUI frame encoding.
//!
//! Computes incremental or full frame diffs from pane buffer state,
//! and encodes them as ANSI escape sequences or Cap'n Proto TuiFrame messages.

use ratatui::buffer::Cell;
use ratatui::style::{Color, Modifier};

use super::state::{CursorState, PaneBuffer, ScrollOp};

// ============================================================================
// Frame Diff Types
// ============================================================================

/// A cell change in a frame diff.
#[derive(Debug, Clone)]
pub struct CellDelta {
    pub x: u16,
    pub y: u16,
    pub symbol: String,
    pub fg: Color,
    pub bg: Color,
    pub modifiers: Modifier,
}

/// A computed frame diff.
#[derive(Debug)]
pub struct FrameDiff {
    /// Whether this is a full frame.
    pub is_full: bool,
    /// Terminal dimensions (for full frames).
    pub cols: u16,
    pub rows: u16,
    /// Scroll operations (for incremental frames).
    pub scrolls: Vec<ScrollOp>,
    /// Changed cells.
    pub deltas: Vec<CellDelta>,
    /// Cursor state.
    pub cursor: CursorState,
    /// Generation counter.
    pub generation: u64,
}

// ============================================================================
// Diff Computation
// ============================================================================

/// Compute a frame diff from a pane buffer.
///
/// If `force_full_frame` is set or >50% of rows are dirty, returns a full frame.
/// Otherwise returns an incremental frame with scroll ops and cell deltas.
pub fn compute_diff(
    pane: &PaneBuffer,
    prev: Option<&[Cell]>,
    generation: u64,
    cursor: CursorState,
) -> FrameDiff {
    let area = pane.buffer.area();
    let cols = area.width;
    let rows = area.height;

    let is_full = pane.force_full_frame
        || prev.is_none()
        || pane.dirty_count() > (rows as u32) / 2;

    if is_full {
        // Full frame: emit all cells
        let mut deltas = Vec::with_capacity(cols as usize * rows as usize);
        for y in 0..rows {
            for x in 0..cols {
                let cell = &pane.buffer[(x, y)];
                deltas.push(CellDelta {
                    x,
                    y,
                    symbol: cell.symbol().to_owned(),
                    fg: cell.fg,
                    bg: cell.bg,
                    modifiers: cell.modifier,
                });
            }
        }
        FrameDiff {
            is_full: true,
            cols,
            rows,
            scrolls: Vec::new(),
            deltas,
            cursor,
            generation,
        }
    } else {
        // Incremental: drain scroll log + compare dirty rows cell-by-cell
        let scrolls = pane.scroll_log.clone();
        let mut deltas = Vec::new();

        if let Some(prev_cells) = prev {
            for y in 0..rows {
                if !pane.is_row_dirty(y) {
                    continue;
                }
                for x in 0..cols {
                    let cell = &pane.buffer[(x, y)];
                    let prev_idx = y as usize * cols as usize + x as usize;
                    let changed = if prev_idx < prev_cells.len() {
                        let pc = &prev_cells[prev_idx];
                        cell.symbol() != pc.symbol()
                            || cell.fg != pc.fg
                            || cell.bg != pc.bg
                            || cell.modifier != pc.modifier
                    } else {
                        true
                    };
                    if changed {
                        deltas.push(CellDelta {
                            x,
                            y,
                            symbol: cell.symbol().to_owned(),
                            fg: cell.fg,
                            bg: cell.bg,
                            modifiers: cell.modifier,
                        });
                    }
                }
            }
        }

        FrameDiff {
            is_full: false,
            cols,
            rows,
            scrolls,
            deltas,
            cursor,
            generation,
        }
    }
}

// ============================================================================
// ANSI Encoding
// ============================================================================

/// Encode a frame diff as ANSI escape sequences.
///
/// Output is a byte buffer of VT100/ANSI sequences that, when written to
/// a terminal, reproduce the frame diff. Tracks previous SGR state to
/// minimize redundant escape sequences.
pub fn encode_ansi(diff: &FrameDiff) -> Vec<u8> {
    let mut out = Vec::with_capacity(diff.deltas.len() * 20);

    // Scroll operations
    for scroll in &diff.scrolls {
        // Set scroll region
        write_ansi!(out, "\x1b[{};{}r", scroll.top + 1, scroll.bottom + 1);
        if scroll.amount > 0 {
            // Scroll up
            write_ansi!(out, "\x1b[{}S", scroll.amount);
        } else if scroll.amount < 0 {
            // Scroll down
            write_ansi!(out, "\x1b[{}T", -scroll.amount);
        }
        // Reset scroll region
        write_ansi!(out, "\x1b[r");
    }

    // Cell deltas
    let mut last_x: Option<u16> = None;
    let mut last_y: Option<u16> = None;
    let mut last_fg: Option<Color> = None;
    let mut last_bg: Option<Color> = None;
    let mut last_mods: Option<Modifier> = None;

    for delta in &diff.deltas {
        // Cursor positioning
        let need_move = match (last_x, last_y) {
            (Some(lx), Some(ly)) => !(ly == delta.y && lx == delta.x),
            _ => true,
        };
        if need_move {
            write_ansi!(out, "\x1b[{};{}H", delta.y + 1, delta.x + 1);
        }

        // SGR (using waxterm's combined SGR approach)
        let fg_changed = last_fg != Some(delta.fg);
        let bg_changed = last_bg != Some(delta.bg);
        let mods_changed = last_mods != Some(delta.modifiers);

        if fg_changed || bg_changed || mods_changed {
            waxterm::sgr::write_combined_sgr(&mut out, delta.fg, delta.bg, delta.modifiers)
                .unwrap_or(());
            last_fg = Some(delta.fg);
            last_bg = Some(delta.bg);
            last_mods = Some(delta.modifiers);
        }

        // Symbol
        out.extend_from_slice(delta.symbol.as_bytes());

        // Track position for wide char awareness
        let char_width = if delta.symbol.len() > 1 {
            // Approximate: multi-byte chars are often wide
            unicode_width(&delta.symbol)
        } else {
            1
        };
        last_x = Some(delta.x + char_width as u16);
        last_y = Some(delta.y);
    }

    // Reset SGR
    out.extend_from_slice(b"\x1b[0m");

    // Cursor position + visibility
    if diff.cursor.visible {
        write_ansi!(out, "\x1b[?25h");
        write_ansi!(out, "\x1b[{};{}H", diff.cursor.y + 1, diff.cursor.x + 1);
    } else {
        write_ansi!(out, "\x1b[?25l");
    }

    out
}

/// Simple unicode width estimation (1 for ASCII, 2 for CJK, 1 for others).
fn unicode_width(s: &str) -> usize {
    s.chars().next().map_or(1, |c| {
        if c.is_ascii() {
            1
        } else {
            // Simplified: CJK ranges are typically double-width
            let cp = c as u32;
            if (0x1100..=0x115F).contains(&cp)
                || (0x2E80..=0x303E).contains(&cp)
                || (0x3040..=0x9FFF).contains(&cp)
                || (0xAC00..=0xD7AF).contains(&cp)
                || (0xF900..=0xFAFF).contains(&cp)
                || (0xFE10..=0xFE6F).contains(&cp)
                || (0xFF01..=0xFF60).contains(&cp)
                || (0xFFE0..=0xFFE6).contains(&cp)
                || (0x20000..=0x2FA1F).contains(&cp)
            {
                2
            } else {
                1
            }
        }
    })
}

/// Helper macro for writing ANSI sequences into a Vec<u8>.
macro_rules! write_ansi {
    ($buf:expr, $fmt:expr $(, $arg:expr)*) => {
        {
            use std::io::Write;
            let _ = write!($buf, $fmt $(, $arg)*);
        }
    };
}
use write_ansi;

// ============================================================================
// Cap'n Proto Encoding
// ============================================================================

/// Encode a frame diff as a Cap'n Proto TuiFrame message.
///
/// Returns serialized bytes ready for streaming via `StreamPublisher`.
/// `pane_id` is embedded in the TuiFrame so the client can route frames
/// to the correct pane buffer without relying on server-side active pane state.
pub fn encode_capnp(diff: &FrameDiff, pane_id: u32) -> Vec<u8> {
    use capnp::message::Builder;
    use capnp::serialize;

    let mut message = Builder::new_default();
    {
        let mut frame = message.init_root::<crate::tui_capnp::tui_frame::Builder<'_>>();
        frame.set_generation(diff.generation);
        frame.set_timestamp_ms(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );
        frame.set_pane_id(pane_id);

        // Cursor
        let mut cursor = frame.reborrow().init_cursor();
        cursor.set_x(diff.cursor.x);
        cursor.set_y(diff.cursor.y);
        cursor.set_visible(diff.cursor.visible);

        if diff.is_full {
            let mut full = frame.init_full();
            full.set_cols(diff.cols);
            full.set_rows(diff.rows);
            let mut cells = full.init_cells(diff.deltas.len() as u32);
            for (i, delta) in diff.deltas.iter().enumerate() {
                let mut cell = cells.reborrow().get(i as u32);
                cell.set_x(delta.x);
                cell.set_y(delta.y);
                cell.set_symbol(&delta.symbol);
                cell.set_fg(pack_color(delta.fg));
                cell.set_bg(pack_color(delta.bg));
                cell.set_modifiers(delta.modifiers.bits());
            }
        } else {
            let mut incr = frame.init_incremental();
            let mut scrolls = incr.reborrow().init_scrolls(diff.scrolls.len() as u32);
            for (i, scroll) in diff.scrolls.iter().enumerate() {
                let mut s = scrolls.reborrow().get(i as u32);
                s.set_top(scroll.top);
                s.set_bottom(scroll.bottom);
                s.set_amount(scroll.amount);
            }
            let mut deltas = incr.init_deltas(diff.deltas.len() as u32);
            for (i, delta) in diff.deltas.iter().enumerate() {
                let mut cell = deltas.reborrow().get(i as u32);
                cell.set_x(delta.x);
                cell.set_y(delta.y);
                cell.set_symbol(&delta.symbol);
                cell.set_fg(pack_color(delta.fg));
                cell.set_bg(pack_color(delta.bg));
                cell.set_modifiers(delta.modifiers.bits());
            }
        }
    }

    let mut buf = Vec::new();
    serialize::write_message(&mut buf, &message).unwrap_or(());
    buf
}

/// Pack a ratatui Color into a u32.
///
/// Encoding: RGB → 0x00RRGGBB, Indexed → 0x01_0000_II, Named → ordinal.
pub(crate) fn pack_color(color: Color) -> u32 {
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
        Color::Rgb(r, g, b) => ((r as u32) << 16) | ((g as u32) << 8) | (b as u32) | 0x0100_0000,
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_diff() {
        let buf = PaneBuffer::new(80, 24);
        let diff = compute_diff(&buf, None, 1, CursorState::default());
        assert!(diff.is_full);
        assert_eq!(diff.deltas.len(), 80 * 24);
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
    fn test_incremental_single_cell() {
        let mut buf = PaneBuffer::new(10, 5);
        buf.clear_damage();

        // Modify one cell
        buf.buffer[(3, 2)].set_symbol("X");
        buf.mark_dirty(2);

        // Create "previous" state
        let prev: Vec<Cell> = (0..50).map(|_| Cell::default()).collect();
        let diff = compute_diff(&buf, Some(&prev), 2, CursorState::default());

        assert!(!diff.is_full);
        assert!(!diff.deltas.is_empty());
    }

    #[test]
    fn test_full_frame_threshold() {
        let mut buf = PaneBuffer::new(10, 4);
        buf.clear_damage();

        // Mark >50% rows dirty (3 out of 4)
        buf.mark_dirty(0);
        buf.mark_dirty(1);
        buf.mark_dirty(2);

        let prev: Vec<Cell> = (0..40).map(|_| Cell::default()).collect();
        let diff = compute_diff(&buf, Some(&prev), 3, CursorState::default());

        assert!(diff.is_full);
    }

    #[test]
    fn test_encode_ansi_gap_cursor_position() {
        // Regression: encode_ansi had lx+1==delta.x condition which caused cells
        // with exactly one unchanged cell between them to be written at the wrong
        // position (off-by-one ghost bug).
        //
        // Pattern: changed at x=0, unchanged at x=1, changed at x=2.
        // The encoded ANSI must position the cursor at x=2 before writing it,
        // not assume the cursor is already there after x=0's write.
        let deltas = vec![
            CellDelta { x: 0, y: 0, symbol: "A".to_owned(), fg: Color::Reset, bg: Color::Reset, modifiers: Modifier::empty() },
            CellDelta { x: 2, y: 0, symbol: "B".to_owned(), fg: Color::Reset, bg: Color::Reset, modifiers: Modifier::empty() },
        ];
        let diff = FrameDiff {
            is_full: false,
            cols: 10,
            rows: 1,
            scrolls: vec![],
            deltas,
            cursor: CursorState::default(),
            generation: 1,
        };
        let ansi = encode_ansi(&diff);
        let s = String::from_utf8_lossy(&ansi);
        // Must contain a cursor position for x=2 (column 3 in 1-indexed)
        assert!(s.contains("\x1b[1;3H"), "Missing CUP for x=2: {:?}", s);
    }
}
