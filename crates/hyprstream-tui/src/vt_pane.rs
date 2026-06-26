//! Generic VT pane — avt virtual terminal + channel-based I/O.
//!
//! Extracted from `PaneWindow` to decouple VTE parsing from PTY lifecycle.
//! Used by both `PaneWindow` (PTY shells) and `ContainerTermApp` (container attach).

#![cfg(not(target_os = "wasi"))]

use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};

use avt::Vt;

/// A virtual terminal backed by byte channels.
///
/// Receives stdout bytes via `stdout_rx`, feeds them into the avt VT parser.
/// Optionally sends stdin bytes via `stdin_tx` (None for read-only streams).
pub struct VtPane {
    pub vt: Vt,
    pub stdout_rx: Receiver<Vec<u8>>,
    pub stdin_tx: Option<SyncSender<Vec<u8>>>,
}

impl VtPane {
    /// Create a new VT pane with the given dimensions and I/O channels.
    pub fn new(
        cols: u16,
        rows: u16,
        stdout_rx: Receiver<Vec<u8>>,
        stdin_tx: Option<SyncSender<Vec<u8>>>,
    ) -> Self {
        let vt = Vt::builder()
            .size(cols as usize, rows as usize)
            .build();
        Self { vt, stdout_rx, stdin_tx }
    }

    /// Drain stdout_rx into the VTE parser.
    ///
    /// Returns `(got_data, is_dead)`:
    /// - `got_data` — true if any bytes arrived (caller should redraw)
    /// - `is_dead`  — true if the channel disconnected (source exited)
    pub fn drain(&mut self) -> (bool, bool) {
        let mut got = false;
        loop {
            match self.stdout_rx.try_recv() {
                Ok(data) => {
                    let s = String::from_utf8_lossy(&data);
                    self.vt.feed_str(&s);
                    got = true;
                }
                Err(TryRecvError::Empty) => return (got, false),
                Err(TryRecvError::Disconnected) => return (got, true),
            }
        }
    }

    /// Send raw bytes to stdin (if a sender is configured).
    pub fn send_bytes(&self, data: Vec<u8>) {
        if let Some(ref tx) = self.stdin_tx {
            let _ = tx.send(data);
        }
    }

    /// Resize the VT to new dimensions (protocol-agnostic, no escape sequences).
    pub fn resize_vt(&mut self, cols: u16, rows: u16) {
        self.vt = Vt::builder()
            .size(cols as usize, rows as usize)
            .build();
    }
}
