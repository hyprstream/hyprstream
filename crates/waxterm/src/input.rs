/// Standard key events (always parsed by the framework).
#[derive(Debug, PartialEq, Eq)]
pub enum KeyPress {
    Char(u8),
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Tab,
    Enter,
    Escape,
    Backspace,
}

/// Custom OSC handler: receives bytes after `ESC ]`, returns `(command, bytes_consumed)`.
/// `bytes_consumed` counts from the start of the OSC payload (after `ESC ]`).
pub type OscHandler<C> = Box<dyn Fn(&[u8]) -> Option<(C, usize)>>;

/// Extensible input parser.
///
/// Parses standard key sequences built-in, then dispatches `ESC ]` sequences
/// to app-registered OSC handlers.
pub struct InputParser<C> {
    osc_handlers: Vec<OscHandler<C>>,
}

impl<C: From<KeyPress>> InputParser<C> {
    pub fn new(osc_handlers: Vec<OscHandler<C>>) -> Self {
        InputParser { osc_handlers }
    }

    /// Parse a byte buffer into a sequence of commands.
    ///
    /// Protocol:
    /// - `ESC [ A/B/C/D` → ArrowUp/Down/Right/Left
    /// - `ESC ] ...` → dispatched to app-registered OSC handlers
    /// - `0x09` → Tab, `0x0D` → Enter, `0x1B` alone → Escape, `0x7F` → Backspace
    /// - `0x00` → skipped (JS stdin heartbeat)
    /// - anything else → `Char(byte)`
    pub fn parse(&self, data: &[u8]) -> Vec<C> {
        let mut cmds = Vec::new();
        let mut i = 0;

        while i < data.len() {
            // Skip NUL heartbeat bytes (JS sends 0x00 to keep stdin flowing)
            if data[i] == 0x00 {
                i += 1;
                continue;
            }

            if data[i] == 0x1B {
                // Need at least 1 more byte to determine escape type
                if i + 1 < data.len() {
                    match data[i + 1] {
                        // ESC [ → CSI keyboard sequences (3 bytes total)
                        0x5B => {
                            if i + 2 < data.len() {
                                match data[i + 2] {
                                    b'A' => cmds.push(C::from(KeyPress::ArrowUp)),
                                    b'B' => cmds.push(C::from(KeyPress::ArrowDown)),
                                    b'C' => cmds.push(C::from(KeyPress::ArrowRight)),
                                    b'D' => cmds.push(C::from(KeyPress::ArrowLeft)),
                                    _ => {} // ignore unknown CSI sequences
                                }
                                i += 3;
                            } else {
                                // Incomplete CSI, skip ESC
                                i += 1;
                            }
                        }
                        // ESC ] → OSC sequences, dispatched to app handlers
                        0x5D => {
                            let osc_start = i + 2;
                            let osc_data = &data[osc_start..];
                            let mut handled = false;
                            for handler in &self.osc_handlers {
                                if let Some((cmd, consumed)) = handler(osc_data) {
                                    cmds.push(cmd);
                                    i = osc_start + consumed;
                                    handled = true;
                                    break;
                                }
                            }
                            if !handled {
                                // Unknown OSC, skip ESC ]
                                i += 2;
                            }
                        }
                        _ => {
                            // Unknown escape, treat ESC as Escape key
                            cmds.push(C::from(KeyPress::Escape));
                            i += 1;
                        }
                    }
                } else {
                    // Trailing ESC with no following byte → Escape key
                    cmds.push(C::from(KeyPress::Escape));
                    i += 1;
                }
            } else {
                // Standard single-byte keys
                let key = match data[i] {
                    0x09 => KeyPress::Tab,
                    0x0D => KeyPress::Enter,
                    0x7F => KeyPress::Backspace,
                    b => KeyPress::Char(b),
                };
                cmds.push(C::from(key));
                i += 1;
            }
        }

        cmds
    }
}

/// WASI stdin poller (non-blocking read from VFS).
pub struct StdinPoller {
    buf: [u8; 256],
}

impl StdinPoller {
    pub fn new() -> Self {
        StdinPoller { buf: [0u8; 256] }
    }

    /// Read any available bytes from stdin.
    pub fn read(&mut self) -> &[u8] {
        use std::io::Read;
        match std::io::stdin().read(&mut self.buf) {
            Ok(n) if n > 0 => &self.buf[..n],
            _ => &[],
        }
    }
}

impl Default for StdinPoller {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> InputParser<KeyPress> {
        InputParser::new(vec![])
    }

    #[test]
    fn test_parse_single_keys() {
        let data = [b' ', b'q', 0x09, b'1'];
        let cmds = parser().parse(&data);
        assert_eq!(cmds.len(), 4);
        assert_eq!(cmds[0], KeyPress::Char(b' '));
        assert_eq!(cmds[1], KeyPress::Char(b'q'));
        assert_eq!(cmds[2], KeyPress::Tab);
        assert_eq!(cmds[3], KeyPress::Char(b'1'));
    }

    #[test]
    fn test_parse_arrows() {
        let data = [0x1B, 0x5B, b'C', 0x1B, 0x5B, b'D'];
        let cmds = parser().parse(&data);
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0], KeyPress::ArrowRight);
        assert_eq!(cmds[1], KeyPress::ArrowLeft);
    }

    #[test]
    fn test_parse_all_arrows() {
        let data = [0x1B, 0x5B, b'A', 0x1B, 0x5B, b'B', 0x1B, 0x5B, b'C', 0x1B, 0x5B, b'D'];
        let cmds = parser().parse(&data);
        assert_eq!(cmds.len(), 4);
        assert_eq!(cmds[0], KeyPress::ArrowUp);
        assert_eq!(cmds[1], KeyPress::ArrowDown);
        assert_eq!(cmds[2], KeyPress::ArrowRight);
        assert_eq!(cmds[3], KeyPress::ArrowLeft);
    }

    #[test]
    fn test_parse_osc_handlers() {
        // Register a custom OSC handler: ESC ] T <idx> → SelectTab
        #[derive(Debug, PartialEq)]
        enum Cmd {
            Key(KeyPress),
            SelectTab(u8),
        }
        impl From<KeyPress> for Cmd {
            fn from(k: KeyPress) -> Self {
                Cmd::Key(k)
            }
        }

        let parser = InputParser::new(vec![Box::new(|data: &[u8]| {
            if data.len() >= 2 && data[0] == b'T' {
                Some((Cmd::SelectTab(data[1]), 2))
            } else {
                None
            }
        })]);

        let data = [0x1B, 0x5D, b'T', 2, 0x1B, 0x5D, b'T', 0];
        let cmds = parser.parse(&data);
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0], Cmd::SelectTab(2));
        assert_eq!(cmds[1], Cmd::SelectTab(0));
    }

    #[test]
    fn test_parse_mixed() {
        let data = [b' ', 0x1B, 0x5B, b'C', b'q'];
        let cmds = parser().parse(&data);
        assert_eq!(cmds.len(), 3);
        assert_eq!(cmds[0], KeyPress::Char(b' '));
        assert_eq!(cmds[1], KeyPress::ArrowRight);
        assert_eq!(cmds[2], KeyPress::Char(b'q'));
    }

    #[test]
    fn test_nul_bytes_skipped() {
        let data = [0x00, b'a', 0x00, 0x00, b'b'];
        let cmds = parser().parse(&data);
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0], KeyPress::Char(b'a'));
        assert_eq!(cmds[1], KeyPress::Char(b'b'));
    }

    #[test]
    fn test_special_keys() {
        let data = [0x0D, 0x7F];
        let cmds = parser().parse(&data);
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0], KeyPress::Enter);
        assert_eq!(cmds[1], KeyPress::Backspace);
    }

    #[test]
    fn test_trailing_escape() {
        let data = [b'a', 0x1B];
        let cmds = parser().parse(&data);
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0], KeyPress::Char(b'a'));
        assert_eq!(cmds[1], KeyPress::Escape);
    }
}
