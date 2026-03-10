//! VTE parser for ANSI pane ingestion.
//!
//! Implements `vte::Perform` to translate ANSI escape sequences into
//! cell writes, cursor movement, scroll region operations, and screen
//! mode switches on a `TuiPane`.

use ratatui::buffer::Cell;
use ratatui::style::{Color, Modifier, Style};

use super::state::{CursorState, ScrollOp, TuiPane};

// ============================================================================
// SGR state
// ============================================================================

/// Tracked SGR (Select Graphic Rendition) state.
#[derive(Debug, Clone)]
struct SgrState {
    fg: Color,
    bg: Color,
    modifiers: Modifier,
}

impl Default for SgrState {
    fn default() -> Self {
        Self {
            fg: Color::Reset,
            bg: Color::Reset,
            modifiers: Modifier::empty(),
        }
    }
}

// ============================================================================
// PanePerformer
// ============================================================================

/// VTE performer that writes into a `TuiPane`.
///
/// Handles: print, execute (CR/LF/BS/HT), CSI dispatch (SGR, CUP, ED, EL,
/// SU, SD, DECSTBM, cursor movement, DECTCEM, DECSET 1049), OSC (title),
/// ESC (RIS, DECSC, DECRC).
pub struct PanePerformer<'a> {
    pane: &'a mut TuiPane,
    sgr: SgrState,
    /// Saved cursor position (DECSC/DECRC).
    saved_cursor: CursorState,
    /// Scroll region top (0-indexed, inclusive).
    scroll_top: u16,
    /// Scroll region bottom (0-indexed, inclusive).
    scroll_bottom: u16,
    /// DEC auto-wrap pending: set when a character is written at the last column.
    /// The actual wrap (and potential scroll) fires on the next printed character,
    /// not immediately — this prevents spurious scrolls when a ratatui draw
    /// writes the very last cell of the screen.
    pending_wrap: bool,
}

impl<'a> PanePerformer<'a> {
    /// Create a new performer for the given pane.
    pub fn new(pane: &'a mut TuiPane) -> Self {
        let (_, rows) = pane.size();
        Self {
            pane,
            sgr: SgrState::default(),
            saved_cursor: CursorState::default(),
            scroll_top: 0,
            scroll_bottom: rows.saturating_sub(1),
            pending_wrap: false,
        }
    }

    /// Feed raw bytes through the VTE parser into this pane.
    pub fn feed(&mut self, data: &[u8]) {
        let mut parser = vte::Parser::new();
        for &byte in data {
            parser.advance(self, byte);
        }
    }

    fn cols(&self) -> u16 {
        self.pane.active_buffer().buffer.area().width
    }

    fn rows(&self) -> u16 {
        self.pane.active_buffer().buffer.area().height
    }

    fn set_cell(&mut self, x: u16, y: u16, symbol: &str) {
        let buf = self.pane.active_buffer_mut();
        let area = buf.buffer.area();
        if x < area.width && y < area.height {
            buf.buffer[(x, y)].set_symbol(symbol);
            buf.buffer[(x, y)].set_style(Style::default()
                .fg(self.sgr.fg)
                .bg(self.sgr.bg)
                .add_modifier(self.sgr.modifiers));
            buf.mark_dirty(y);
        }
    }

    /// Scroll the region up by `n` lines.
    fn scroll_up(&mut self, n: u16) {
        let cols = self.cols();
        let top = self.scroll_top;
        let bottom = self.scroll_bottom;
        let blank = self.blank_cell();

        if n == 0 || top >= bottom {
            return;
        }

        // Save top lines to scrollback before overwriting
        if !self.pane.using_alternate && top == 0 {
            for row in 0..n.min(bottom - top + 1) {
                let mut line = Vec::with_capacity(cols as usize);
                let buf = self.pane.active_buffer();
                for x in 0..cols {
                    line.push(buf.buffer[(x, row)].clone());
                }
                self.pane.push_scrollback(line);
            }
        }

        // Move lines up within the scroll region
        let buf = self.pane.active_buffer_mut();
        for y in top..=bottom {
            let src_y = y + n;
            if src_y <= bottom {
                for x in 0..cols {
                    let cell = buf.buffer[(x, src_y)].clone();
                    buf.buffer[(x, y)] = cell;
                }
            } else {
                for x in 0..cols {
                    buf.buffer[(x, y)] = blank.clone();
                }
            }
            buf.mark_dirty(y);
        }

        buf.record_scroll(ScrollOp {
            top,
            bottom,
            amount: n as i16,
        });
    }

    /// Scroll the region down by `n` lines.
    fn scroll_down(&mut self, n: u16) {
        let cols = self.cols();
        let top = self.scroll_top;
        let bottom = self.scroll_bottom;
        let blank = self.blank_cell();

        if n == 0 || top >= bottom {
            return;
        }

        let buf = self.pane.active_buffer_mut();

        // Move lines down within the scroll region (iterate in reverse)
        for y in (top..=bottom).rev() {
            if y >= top + n {
                let src_y = y - n;
                for x in 0..cols {
                    let cell = buf.buffer[(x, src_y)].clone();
                    buf.buffer[(x, y)] = cell;
                }
            } else {
                // Clear new lines at top
                for x in 0..cols {
                    buf.buffer[(x, y)] = blank.clone();
                }
            }
            buf.mark_dirty(y);
        }

        buf.record_scroll(ScrollOp {
            top,
            bottom,
            amount: -(n as i16),
        });
    }

    /// Return a blank Cell using the current SGR background colour.
    ///
    /// Per the VT specification, erase operations (ED, EL, ECH, ICH clear,
    /// DCH clear, and scroll-cleared rows) fill cells with a space character
    /// styled with the current background colour rather than the terminal
    /// default.  Foreground and modifiers are reset to default.
    fn blank_cell(&self) -> Cell {
        let mut cell = Cell::default();
        cell.set_style(Style::default().fg(Color::Reset).bg(self.sgr.bg));
        cell
    }

    /// Erase cells in a region, filling them with the current background color.
    fn erase_region(&mut self, x1: u16, y1: u16, x2: u16, y2: u16) {
        let blank = self.blank_cell();
        let buf = self.pane.active_buffer_mut();
        let width = buf.buffer.area().width;
        let height = buf.buffer.area().height;
        for y in y1..=y2.min(height.saturating_sub(1)) {
            let start_x = if y == y1 { x1 } else { 0 };
            let end_x = if y == y2 { x2 } else { width.saturating_sub(1) };
            for x in start_x..=end_x.min(width.saturating_sub(1)) {
                buf.buffer[(x, y)] = blank.clone();
            }
            buf.mark_dirty(y);
        }
    }

    /// Parse SGR parameters and update state.
    fn process_sgr(&mut self, params: &vte::Params) {
        let mut iter = params.iter();

        while let Some(param) = iter.next() {
            let code = param[0];

            match code {
                0 => self.sgr = SgrState::default(),
                1 => self.sgr.modifiers |= Modifier::BOLD,
                2 => self.sgr.modifiers |= Modifier::DIM,
                3 => self.sgr.modifiers |= Modifier::ITALIC,
                4 => self.sgr.modifiers |= Modifier::UNDERLINED,
                5 => self.sgr.modifiers |= Modifier::SLOW_BLINK,
                7 => self.sgr.modifiers |= Modifier::REVERSED,
                8 => self.sgr.modifiers |= Modifier::HIDDEN,
                9 => self.sgr.modifiers |= Modifier::CROSSED_OUT,
                22 => self.sgr.modifiers -= Modifier::BOLD | Modifier::DIM,
                23 => self.sgr.modifiers -= Modifier::ITALIC,
                24 => self.sgr.modifiers -= Modifier::UNDERLINED,
                25 => self.sgr.modifiers -= Modifier::SLOW_BLINK,
                27 => self.sgr.modifiers -= Modifier::REVERSED,
                28 => self.sgr.modifiers -= Modifier::HIDDEN,
                29 => self.sgr.modifiers -= Modifier::CROSSED_OUT,
                // Foreground colors
                30 => self.sgr.fg = Color::Black,
                31 => self.sgr.fg = Color::Red,
                32 => self.sgr.fg = Color::Green,
                33 => self.sgr.fg = Color::Yellow,
                34 => self.sgr.fg = Color::Blue,
                35 => self.sgr.fg = Color::Magenta,
                36 => self.sgr.fg = Color::Cyan,
                37 => self.sgr.fg = Color::Gray,
                38 => {
                    self.sgr.fg = self.parse_extended_color(&mut iter);
                }
                39 => self.sgr.fg = Color::Reset,
                // Background colors
                40 => self.sgr.bg = Color::Black,
                41 => self.sgr.bg = Color::Red,
                42 => self.sgr.bg = Color::Green,
                43 => self.sgr.bg = Color::Yellow,
                44 => self.sgr.bg = Color::Blue,
                45 => self.sgr.bg = Color::Magenta,
                46 => self.sgr.bg = Color::Cyan,
                47 => self.sgr.bg = Color::Gray,
                48 => {
                    self.sgr.bg = self.parse_extended_color(&mut iter);
                }
                49 => self.sgr.bg = Color::Reset,
                // Bright foreground
                90 => self.sgr.fg = Color::DarkGray,
                91 => self.sgr.fg = Color::LightRed,
                92 => self.sgr.fg = Color::LightGreen,
                93 => self.sgr.fg = Color::LightYellow,
                94 => self.sgr.fg = Color::LightBlue,
                95 => self.sgr.fg = Color::LightMagenta,
                96 => self.sgr.fg = Color::LightCyan,
                97 => self.sgr.fg = Color::White,
                // Bright background
                100 => self.sgr.bg = Color::DarkGray,
                101 => self.sgr.bg = Color::LightRed,
                102 => self.sgr.bg = Color::LightGreen,
                103 => self.sgr.bg = Color::LightYellow,
                104 => self.sgr.bg = Color::LightBlue,
                105 => self.sgr.bg = Color::LightMagenta,
                106 => self.sgr.bg = Color::LightCyan,
                107 => self.sgr.bg = Color::White,
                _ => {} // Ignore unknown
            }
        }
    }

    /// Parse extended color (256-color or RGB) from SGR params.
    fn parse_extended_color<'b>(
        &self,
        iter: &mut impl Iterator<Item = &'b [u16]>,
    ) -> Color {
        match iter.next().map(|p| p[0]) {
            Some(5) => {
                // 256-color: ESC[38;5;Nm
                if let Some(idx) = iter.next().map(|p| p[0] as u8) {
                    Color::Indexed(idx)
                } else {
                    Color::Reset
                }
            }
            Some(2) => {
                // RGB: ESC[38;2;R;G;Bm
                let r = iter.next().map(|p| p[0] as u8).unwrap_or(0);
                let g = iter.next().map(|p| p[0] as u8).unwrap_or(0);
                let b = iter.next().map(|p| p[0] as u8).unwrap_or(0);
                Color::Rgb(r, g, b)
            }
            _ => Color::Reset,
        }
    }
}

impl<'a> vte::Perform for PanePerformer<'a> {
    fn print(&mut self, c: char) {
        // If a wrap was pending from the previous character at EOL, do it now.
        if self.pending_wrap {
            self.pending_wrap = false;
            self.pane.cursor.x = 0;
            self.pane.cursor.y += 1;
            if self.pane.cursor.y > self.scroll_bottom {
                self.pane.cursor.y = self.scroll_bottom;
                self.scroll_up(1);
            }
        }

        let x = self.pane.cursor.x;
        let y = self.pane.cursor.y;
        let cols = self.cols();

        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        self.set_cell(x, y, s);

        // Advance cursor (wide char detection).
        let width = if c.is_ascii() { 1 } else {
            let cp = c as u32;
            if (0x1100..=0x115F).contains(&cp)
                || (0x2E80..=0x303E).contains(&cp)
                || (0x3040..=0x9FFF).contains(&cp)
                || (0xAC00..=0xD7AF).contains(&cp)
                || (0xF900..=0xFAFF).contains(&cp)
                || (0x20000..=0x2FA1F).contains(&cp)
            {
                2
            } else {
                1
            }
        };

        let new_x = x + width;
        if new_x >= cols {
            // Defer the wrap: real wrap fires on the NEXT printed character.
            // This prevents a spurious scroll when the last cell of the screen
            // is written (e.g. by a ratatui full-redraw).
            self.pending_wrap = true;
            self.pane.cursor.x = cols - 1;
        } else {
            self.pane.cursor.x = new_x;
        }
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            // BS (backspace)
            0x08 => {
                self.pending_wrap = false;
                self.pane.cursor.x = self.pane.cursor.x.saturating_sub(1);
            }
            // HT (horizontal tab)
            0x09 => {
                self.pending_wrap = false;
                let next_tab = ((self.pane.cursor.x / 8) + 1) * 8;
                self.pane.cursor.x = next_tab.min(self.cols().saturating_sub(1));
            }
            // LF (line feed) / VT / FF
            0x0A..=0x0C => {
                self.pending_wrap = false;
                self.pane.cursor.y += 1;
                if self.pane.cursor.y > self.scroll_bottom {
                    self.pane.cursor.y = self.scroll_bottom;
                    self.scroll_up(1);
                }
            }
            // CR (carriage return)
            0x0D => {
                self.pending_wrap = false;
                self.pane.cursor.x = 0;
            }
            _ => {}
        }
    }

    fn csi_dispatch(
        &mut self,
        params: &vte::Params,
        _intermediates: &[u8],
        _ignore: bool,
        action: char,
    ) {
        // Any cursor-movement or screen-operation sequence cancels a pending wrap.
        self.pending_wrap = false;

        let p: Vec<u16> = params.iter().map(|s| s[0]).collect();
        let p0 = p.first().copied().unwrap_or(0);
        let p1 = p.get(1).copied().unwrap_or(0);

        match action {
            // CUP - Cursor Position
            'H' | 'f' => {
                let row = if p0 == 0 { 1 } else { p0 };
                let col = if p1 == 0 { 1 } else { p1 };
                self.pane.cursor.y = (row - 1).min(self.rows().saturating_sub(1));
                self.pane.cursor.x = (col - 1).min(self.cols().saturating_sub(1));
            }

            // CUU - Cursor Up
            'A' => {
                let n = if p0 == 0 { 1 } else { p0 };
                self.pane.cursor.y = self.pane.cursor.y.saturating_sub(n);
            }

            // CUD - Cursor Down
            'B' => {
                let n = if p0 == 0 { 1 } else { p0 };
                self.pane.cursor.y = (self.pane.cursor.y + n).min(self.rows().saturating_sub(1));
            }

            // CUF - Cursor Forward
            'C' => {
                let n = if p0 == 0 { 1 } else { p0 };
                self.pane.cursor.x = (self.pane.cursor.x + n).min(self.cols().saturating_sub(1));
            }

            // CUB - Cursor Backward
            'D' => {
                let n = if p0 == 0 { 1 } else { p0 };
                self.pane.cursor.x = self.pane.cursor.x.saturating_sub(n);
            }

            // CNL - Cursor Next Line
            'E' => {
                let n = if p0 == 0 { 1 } else { p0 };
                self.pane.cursor.x = 0;
                self.pane.cursor.y = (self.pane.cursor.y + n).min(self.rows().saturating_sub(1));
            }

            // CPL - Cursor Previous Line
            'F' => {
                let n = if p0 == 0 { 1 } else { p0 };
                self.pane.cursor.x = 0;
                self.pane.cursor.y = self.pane.cursor.y.saturating_sub(n);
            }

            // CHA - Cursor Horizontal Absolute
            'G' => {
                let col = if p0 == 0 { 1 } else { p0 };
                self.pane.cursor.x = (col - 1).min(self.cols().saturating_sub(1));
            }

            // ED - Erase in Display
            'J' => {
                let cx = self.pane.cursor.x;
                let cy = self.pane.cursor.y;
                let cols = self.cols();
                let rows = self.rows();
                match p0 {
                    0 => {
                        // Erase from cursor to end
                        self.erase_region(cx, cy, cols - 1, rows - 1);
                    }
                    1 => {
                        // Erase from start to cursor
                        self.erase_region(0, 0, cx, cy);
                    }
                    2 | 3 => {
                        // Erase entire display
                        self.erase_region(0, 0, cols - 1, rows - 1);
                    }
                    _ => {}
                }
            }

            // EL - Erase in Line
            'K' => {
                let cx = self.pane.cursor.x;
                let cy = self.pane.cursor.y;
                let cols = self.cols();
                match p0 {
                    0 => self.erase_region(cx, cy, cols - 1, cy),
                    1 => self.erase_region(0, cy, cx, cy),
                    2 => self.erase_region(0, cy, cols - 1, cy),
                    _ => {}
                }
            }

            // SU - Scroll Up
            'S' => {
                let n = if p0 == 0 { 1 } else { p0 };
                self.scroll_up(n);
            }

            // SD - Scroll Down
            'T' => {
                let n = if p0 == 0 { 1 } else { p0 };
                self.scroll_down(n);
            }

            // SGR - Select Graphic Rendition
            'm' => {
                self.process_sgr(params);
            }

            // DECSTBM - Set Top and Bottom Margins (scroll region)
            'r' => {
                let rows = self.rows();
                let top = if p0 == 0 { 1 } else { p0 };
                let bottom = if p1 == 0 { rows } else { p1 };
                self.scroll_top = (top - 1).min(rows.saturating_sub(1));
                self.scroll_bottom = (bottom - 1).min(rows.saturating_sub(1));
                // CUP to home on DECSTBM
                self.pane.cursor.x = 0;
                self.pane.cursor.y = 0;
            }

            // ICH - Insert Characters
            '@' => {
                let n = if p0 == 0 { 1 } else { p0 };
                let cols = self.cols();
                let cx = self.pane.cursor.x;
                let cy = self.pane.cursor.y;
                let blank = self.blank_cell();
                let buf = self.pane.active_buffer_mut();
                // Shift cells right
                for x in (cx..cols).rev() {
                    if x + n < cols {
                        let cell = buf.buffer[(x, cy)].clone();
                        buf.buffer[(x + n, cy)] = cell;
                    }
                }
                // Clear inserted cells with current background
                for x in cx..cx.saturating_add(n).min(cols) {
                    buf.buffer[(x, cy)] = blank.clone();
                }
                buf.mark_dirty(cy);
            }

            // DCH - Delete Characters
            'P' => {
                let n = if p0 == 0 { 1 } else { p0 };
                let cols = self.cols();
                let cx = self.pane.cursor.x;
                let cy = self.pane.cursor.y;
                let blank = self.blank_cell();
                let buf = self.pane.active_buffer_mut();
                // Shift cells left
                for x in cx..cols {
                    if x + n < cols {
                        let cell = buf.buffer[(x + n, cy)].clone();
                        buf.buffer[(x, cy)] = cell;
                    } else {
                        buf.buffer[(x, cy)] = blank.clone();
                    }
                }
                buf.mark_dirty(cy);
            }

            // ECH - Erase Characters
            'X' => {
                let n = if p0 == 0 { 1 } else { p0 };
                let cx = self.pane.cursor.x;
                let cy = self.pane.cursor.y;
                let cols = self.cols();
                self.erase_region(cx, cy, (cx + n - 1).min(cols - 1), cy);
            }

            // IL - Insert Lines
            'L' => {
                let n = if p0 == 0 { 1 } else { p0 };
                let cy = self.pane.cursor.y;
                let saved_top = self.scroll_top;
                self.scroll_top = cy;
                self.scroll_down(n);
                self.scroll_top = saved_top;
            }

            // DL - Delete Lines
            'M' => {
                let n = if p0 == 0 { 1 } else { p0 };
                let cy = self.pane.cursor.y;
                let saved_top = self.scroll_top;
                self.scroll_top = cy;
                self.scroll_up(n);
                self.scroll_top = saved_top;
            }

            // DECSET/DECRST (via intermediates '?')
            'h' | 'l' => {
                let enable = action == 'h';
                // Check if this is a DEC private mode (? prefix)
                if _intermediates.first() == Some(&b'?') {
                    for &code in &p {
                        match code {
                            // DECTCEM - cursor visibility
                            25 => {
                                self.pane.cursor.visible = enable;
                            }
                            // DECSET 1049 - alternate screen
                            1049 => {
                                if enable {
                                    self.pane.enter_alternate_screen();
                                } else {
                                    self.pane.leave_alternate_screen();
                                }
                                // Reset scroll region
                                self.scroll_top = 0;
                                self.scroll_bottom = self.rows().saturating_sub(1);
                            }
                            _ => {}
                        }
                    }
                }
            }

            // VPA - Vertical Line Position Absolute
            'd' => {
                let row = if p0 == 0 { 1 } else { p0 };
                self.pane.cursor.y = (row - 1).min(self.rows().saturating_sub(1));
            }

            // DECSC - Save Cursor (CSI s)
            's' => {
                self.saved_cursor = self.pane.cursor;
            }

            // DECRC - Restore Cursor (CSI u)
            'u' => {
                self.pane.cursor = self.saved_cursor;
            }

            _ => {} // Ignore unhandled CSI
        }
    }

    fn esc_dispatch(&mut self, _intermediates: &[u8], _ignore: bool, byte: u8) {
        match byte {
            // RIS - Full Reset
            b'c' => {
                self.sgr = SgrState::default();
                self.pane.cursor = CursorState::default();
                self.scroll_top = 0;
                self.scroll_bottom = self.rows().saturating_sub(1);
                let cols = self.cols();
                let rows = self.rows();
                self.erase_region(0, 0, cols - 1, rows - 1);
            }
            // DECSC - Save Cursor
            b'7' => {
                self.saved_cursor = self.pane.cursor;
            }
            // DECRC - Restore Cursor
            b'8' => {
                self.pane.cursor = self.saved_cursor;
            }
            _ => {}
        }
    }

    fn osc_dispatch(&mut self, params: &[&[u8]], _bell_terminated: bool) {
        if params.is_empty() {
            return;
        }
        // OSC 0 or 2: set title
        match params[0] {
            b"0" | b"2" => {
                if params.len() > 1 {
                    if let Ok(title) = std::str::from_utf8(params[1]) {
                        self.pane.title = title.to_owned();
                    }
                }
            }
            _ => {}
        }
    }

    fn hook(&mut self, _params: &vte::Params, _intermediates: &[u8], _ignore: bool, _action: char) {}
    fn put(&mut self, _byte: u8) {}
    fn unhook(&mut self) {}
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pane() -> TuiPane {
        TuiPane::new(1, 20, 10, 100)
    }

    #[test]
    fn test_print_basic() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"Hello");
        }
        assert_eq!(pane.cursor.x, 5);
        assert_eq!(pane.cursor.y, 0);
        assert_eq!(pane.primary.buffer[(0, 0)].symbol(), "H");
        assert_eq!(pane.primary.buffer[(4, 0)].symbol(), "o");
    }

    #[test]
    fn test_sgr_colors() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            // Red foreground, green background
            perf.feed(b"\x1b[31;42mX");
        }
        let cell = &pane.primary.buffer[(0, 0)];
        assert_eq!(cell.symbol(), "X");
        assert_eq!(cell.fg, Color::Red);
        assert_eq!(cell.bg, Color::Green);
    }

    #[test]
    fn test_sgr_256_color() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[38;5;42mA");
        }
        assert_eq!(pane.primary.buffer[(0, 0)].fg, Color::Indexed(42));
    }

    #[test]
    fn test_sgr_rgb() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[38;2;255;128;0mB");
        }
        assert_eq!(pane.primary.buffer[(0, 0)].fg, Color::Rgb(255, 128, 0));
    }

    #[test]
    fn test_cup_positioning() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            // Move to row 3, col 5 (1-based)
            perf.feed(b"\x1b[3;5HX");
        }
        assert_eq!(pane.primary.buffer[(4, 2)].symbol(), "X");
        assert_eq!(pane.cursor.x, 5);
        assert_eq!(pane.cursor.y, 2);
    }

    #[test]
    fn test_cursor_movement() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[5;5H"); // row 5, col 5
            perf.feed(b"\x1b[2A");   // up 2
            perf.feed(b"\x1b[3C");   // right 3
        }
        assert_eq!(pane.cursor.y, 2); // 5-1-2 = 2
        assert_eq!(pane.cursor.x, 7); // 5-1+3 = 7
    }

    #[test]
    fn test_ed_erase_display() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"ABCDE");
            perf.feed(b"\x1b[2J"); // Clear entire display
        }
        assert_eq!(pane.primary.buffer[(0, 0)].symbol(), " ");
        assert_eq!(pane.primary.buffer[(4, 0)].symbol(), " ");
    }

    #[test]
    fn test_el_erase_line() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"Hello World");
            perf.feed(b"\x1b[1;6H"); // Move to col 6
            perf.feed(b"\x1b[K");    // Erase to end of line
        }
        assert_eq!(pane.primary.buffer[(0, 0)].symbol(), "H");
        assert_eq!(pane.primary.buffer[(4, 0)].symbol(), "o");
        assert_eq!(pane.primary.buffer[(5, 0)].symbol(), " "); // Erased
    }

    #[test]
    fn test_scroll_region() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            // Set scroll region rows 2-5 (1-based)
            perf.feed(b"\x1b[2;5r");
            assert_eq!(perf.scroll_top, 1);
            assert_eq!(perf.scroll_bottom, 4);
        }
        // DECSTBM homes the cursor
        assert_eq!(pane.cursor.x, 0);
        assert_eq!(pane.cursor.y, 0);
    }

    #[test]
    fn test_alt_screen() {
        let mut pane = make_pane();
        assert!(!pane.using_alternate);
        {
            let mut perf = PanePerformer::new(&mut pane);
            // Enter alternate screen
            perf.feed(b"\x1b[?1049h");
        }
        assert!(pane.using_alternate);
        assert!(pane.alternate.is_some());
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[?1049l");
        }
        assert!(!pane.using_alternate);
    }

    #[test]
    fn test_cursor_visibility() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[?25l"); // Hide
        }
        assert!(!pane.cursor.visible);
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[?25h"); // Show
        }
        assert!(pane.cursor.visible);
    }

    #[test]
    fn test_cr_lf() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"AB\r\nCD");
        }
        assert_eq!(pane.primary.buffer[(0, 0)].symbol(), "A");
        assert_eq!(pane.primary.buffer[(1, 0)].symbol(), "B");
        assert_eq!(pane.primary.buffer[(0, 1)].symbol(), "C");
        assert_eq!(pane.primary.buffer[(1, 1)].symbol(), "D");
    }

    #[test]
    fn test_line_wrap_and_scroll() {
        let mut pane = TuiPane::new(1, 10, 3, 100); // 10 cols, 3 rows
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[1;1HAAAAA"); // Row 1
            perf.feed(b"\x1b[2;1HBBBBB"); // Row 2
            perf.feed(b"\x1b[3;1HCCCCC"); // Row 3
            // Now at row 3, col 5. LF should scroll.
            perf.feed(b"\r\nDDDDD");
        }
        // After scroll: row 0=B, row 1=C, row 2=D
        assert_eq!(pane.primary.buffer[(0, 0)].symbol(), "B");
        assert_eq!(pane.primary.buffer[(0, 1)].symbol(), "C");
        assert_eq!(pane.primary.buffer[(0, 2)].symbol(), "D");
    }

    /// Verifies that writing to the last column of the last row does NOT
    /// immediately scroll (pending_wrap deferred until next printed char).
    /// This prevents spurious scrolling when a ratatui full-redraw writes
    /// cell (cols-1, rows-1) as its final update.
    #[test]
    fn test_no_spurious_scroll_on_last_cell() {
        let mut pane = TuiPane::new(1, 5, 3, 100); // 5 cols, 3 rows
        {
            let mut perf = PanePerformer::new(&mut pane);
            // Fill all three rows (each CUP cancels pending_wrap).
            perf.feed(b"\x1b[1;1HAAAAA"); // Row 0: 5 chars → pending_wrap=true
            perf.feed(b"\x1b[2;1HBBBBB"); // Row 1: CUP cancels pending_wrap
            perf.feed(b"\x1b[3;1HCCCCC"); // Row 2: 5 chars → pending_wrap=true
            // No further input: wrap never fires, NO scroll happens.
        }
        // All three rows must retain their content.
        assert_eq!(pane.primary.buffer[(0, 0)].symbol(), "A");
        assert_eq!(pane.primary.buffer[(0, 1)].symbol(), "B");
        assert_eq!(pane.primary.buffer[(0, 2)].symbol(), "C");
    }

    /// Verifies that pending_wrap DOES fire when the next char arrives without
    /// a cursor-movement command in between.
    #[test]
    fn test_pending_wrap_fires_on_next_char() {
        let mut pane = TuiPane::new(1, 5, 3, 100); // 5 cols, 3 rows
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[1;1HAAAAA"); // Row 0
            perf.feed(b"\x1b[2;1HBBBBB"); // Row 1
            perf.feed(b"\x1b[3;1HCCCCC"); // Row 2: pending_wrap=true
            // Next char fires pending_wrap → wrap → scroll.
            perf.feed(b"D");
        }
        // After scroll: row0=B, row1=C, row2 starts with D.
        assert_eq!(pane.primary.buffer[(0, 0)].symbol(), "B");
        assert_eq!(pane.primary.buffer[(0, 1)].symbol(), "C");
        assert_eq!(pane.primary.buffer[(0, 2)].symbol(), "D");
    }

    #[test]
    fn test_damage_tracking() {
        let mut pane = make_pane();
        pane.primary.clear_damage();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[3;1HX"); // Write at row 3
        }
        assert!(pane.primary.is_row_dirty(2));
        assert!(!pane.primary.is_row_dirty(0));
    }

    #[test]
    fn test_osc_title() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b]0;My Title\x07");
        }
        assert_eq!(pane.title, "My Title");
    }

    #[test]
    fn test_save_restore_cursor() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[5;10H"); // row 5, col 10
            perf.feed(b"\x1b7");      // DECSC
            perf.feed(b"\x1b[1;1H");  // Home
            perf.feed(b"\x1b8");      // DECRC
        }
        assert_eq!(pane.cursor.y, 4);
        assert_eq!(pane.cursor.x, 9);
    }

    #[test]
    fn test_sgr_bold_italic() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[1;3mX"); // Bold + Italic
        }
        let cell = &pane.primary.buffer[(0, 0)];
        assert!(cell.modifier.contains(Modifier::BOLD));
        assert!(cell.modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn test_sgr_reset() {
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[31mR\x1b[0mN"); // Red then reset
        }
        assert_eq!(pane.primary.buffer[(0, 0)].fg, Color::Red);
        assert_eq!(pane.primary.buffer[(1, 0)].fg, Color::Reset);
    }

    #[test]
    fn test_erase_preserves_background_color() {
        // VT spec: EL (erase line) and ED (erase display) fill cells with
        // the current SGR background color, not the terminal default.
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            // Set dark-blue background (like btop header bars), write some text,
            // then erase to end of line — erased cells should keep the dark-blue bg.
            perf.feed(b"\x1b[48;2;20;20;40m"); // dark-blue bg
            perf.feed(b"ABC");
            perf.feed(b"\x1b[1;4H");            // cursor to col 4
            perf.feed(b"\x1b[K");               // EL: erase to EOL
        }
        // Cols 3..cols should have dark-blue bg (not Reset)
        for x in 3..10u16 {
            let cell = &pane.primary.buffer[(x, 0)];
            assert_eq!(
                cell.bg,
                Color::Rgb(20, 20, 40),
                "col {x} erase should preserve bg color, got {:?}",
                cell.bg
            );
        }
        // Written cells should also have the dark-blue bg
        assert_eq!(pane.primary.buffer[(0, 0)].bg, Color::Rgb(20, 20, 40));
    }

    #[test]
    fn test_scroll_up_preserves_background_color() {
        // When lines scroll up, newly cleared rows at the bottom should
        // use the current background color.  The pane is 20×10 so we need
        // 11+ LF characters to actually scroll past the bottom margin.
        let mut pane = make_pane();
        {
            let mut perf = PanePerformer::new(&mut pane);
            perf.feed(b"\x1b[48;2;10;10;20m"); // set bg
            // Move to last row then issue one more LF to trigger scroll
            perf.feed(b"\x1b[10;1H");           // cursor to row 10 (bottom)
            perf.feed(b"\n");                    // one LF → scroll up 1
        }
        // Bottom row (row 9, 0-indexed) should have been cleared with the
        // current background color, not Color::Reset.
        let last_row = (pane.primary.buffer.area().height - 1) as u16;
        let cell = &pane.primary.buffer[(0, last_row)];
        assert_eq!(
            cell.bg,
            Color::Rgb(10, 10, 20),
            "scroll-cleared row should preserve bg, got {:?}",
            cell.bg
        );
    }
}
