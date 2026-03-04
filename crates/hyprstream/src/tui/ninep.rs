//! Plan 9-style virtual filesystem over TuiState.
//!
//! Provides a 9P-inspired namespace for programmatic access to the TUI multiplexer:
//!
//! ```text
//! /tui/ctl           — global control (write "new" to create session)
//! /tui/new           — walk atomically allocates window (Plan 9 /dev/new pattern)
//! /tui/event         — blocking event stream (broadcast::Receiver)
//! /tui/windows/      — directory listing
//! /tui/windows/0/    — window dir (ctl, cons, screen, winname, layout, panes/)
//! /tui/windows/0/panes/0/ — pane dir (cons, screen)
//! ```
//!
//! This is NOT the registry's ContainedFs — standalone, ephemeral.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Notify;

use super::state::TuiState;

// ============================================================================
// Node types
// ============================================================================

/// Virtual file system node identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TuiNodeId {
    Root,
    Ctl,
    New,
    Event,
    WindowsDir,
    WindowDir(u32),
    WindowCtl(u32),
    WindowCons(u32),
    WindowScreen(u32),
    WindowName(u32),
    WindowLayout(u32),
    PanesDir(u32),
    PaneDir(u32, u32),
    PaneCons(u32, u32),
    PaneScreen(u32, u32),
}

impl TuiNodeId {
    /// Encode as a qid_path: `(type_tag << 56) | (window_id << 32) | pane_id`
    pub fn qid_path(&self) -> u64 {
        match self {
            TuiNodeId::Root => 0,
            TuiNodeId::Ctl => 1 << 56,
            TuiNodeId::New => 2 << 56,
            TuiNodeId::Event => 3 << 56,
            TuiNodeId::WindowsDir => 4 << 56,
            TuiNodeId::WindowDir(w) => (5 << 56) | ((*w as u64) << 32),
            TuiNodeId::WindowCtl(w) => (6 << 56) | ((*w as u64) << 32),
            TuiNodeId::WindowCons(w) => (7 << 56) | ((*w as u64) << 32),
            TuiNodeId::WindowScreen(w) => (8 << 56) | ((*w as u64) << 32),
            TuiNodeId::WindowName(w) => (9 << 56) | ((*w as u64) << 32),
            TuiNodeId::WindowLayout(w) => (10 << 56) | ((*w as u64) << 32),
            TuiNodeId::PanesDir(w) => (11 << 56) | ((*w as u64) << 32),
            TuiNodeId::PaneDir(w, p) => (12 << 56) | ((*w as u64) << 32) | (*p as u64),
            TuiNodeId::PaneCons(w, p) => (13 << 56) | ((*w as u64) << 32) | (*p as u64),
            TuiNodeId::PaneScreen(w, p) => (14 << 56) | ((*w as u64) << 32) | (*p as u64),
        }
    }

    /// Whether this node is a directory.
    pub fn is_dir(&self) -> bool {
        matches!(
            self,
            TuiNodeId::Root
                | TuiNodeId::WindowsDir
                | TuiNodeId::WindowDir(_)
                | TuiNodeId::PanesDir(_)
                | TuiNodeId::PaneDir(_, _)
        )
    }
}

// ============================================================================
// Fid state
// ============================================================================

/// State associated with an open fid.
#[derive(Debug)]
pub enum TuiFidState {
    /// Walked to a node but not yet opened.
    Walked(TuiNodeId),
    /// Opened for read/write.
    Opened {
        node: TuiNodeId,
        /// Read cursor for cons files (monotonic offset into ring buffer).
        cons_cursor: u64,
        /// Whether opened for reading.
        reading: bool,
        /// Whether opened for writing.
        writing: bool,
    },
}

// ============================================================================
// Console ring buffer
// ============================================================================

/// Ring buffer for console I/O (`cons` files).
///
/// 256KB ring with monotonic write_offset. Readers track their own cursor.
/// If a reader falls too far behind (`write_offset - cursor > capacity`),
/// a gap marker is emitted.
pub struct ConsBuffer {
    /// Ring buffer data.
    data: Vec<u8>,
    /// Monotonic write offset (always increases, wraps in ring).
    write_offset: u64,
    /// Notify for blocking reads.
    notify: Arc<Notify>,
}

impl ConsBuffer {
    /// Create a new console buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0u8; capacity],
            write_offset: 0,
            notify: Arc::new(Notify::new()),
        }
    }

    /// Write data to the ring buffer.
    pub fn write(&mut self, data: &[u8]) {
        let cap = self.data.len();
        for &byte in data {
            let idx = (self.write_offset as usize) % cap;
            self.data[idx] = byte;
            self.write_offset += 1;
        }
        self.notify.notify_waiters();
    }

    /// Read data from the ring buffer starting at the given cursor.
    ///
    /// Returns `(bytes_read, new_cursor, gap_detected)`.
    /// If the cursor is too far behind, returns a gap marker and advances.
    pub fn read(&self, cursor: u64, buf: &mut [u8]) -> (usize, u64, bool) {
        let cap = self.data.len() as u64;

        // Gap detection
        if self.write_offset.saturating_sub(cursor) > cap {
            // Reader fell behind — data is overwritten
            let new_cursor = self.write_offset;
            return (0, new_cursor, true);
        }

        if cursor >= self.write_offset {
            return (0, cursor, false);
        }

        let available = (self.write_offset - cursor) as usize;
        let to_read = available.min(buf.len());

        for (i, byte) in buf[..to_read].iter_mut().enumerate() {
            let idx = ((cursor + i as u64) as usize) % cap as usize;
            *byte = self.data[idx];
        }

        (to_read, cursor + to_read as u64, false)
    }

    /// Current write offset.
    pub fn write_offset(&self) -> u64 {
        self.write_offset
    }

    /// Get a notifier for blocking reads.
    pub fn notifier(&self) -> Arc<Notify> {
        Arc::clone(&self.notify)
    }
}

// ============================================================================
// Fid table
// ============================================================================

/// Table mapping fids to their state.
pub struct TuiFidTable {
    fids: HashMap<u32, TuiFidState>,
    next_fid: u32,
}

impl TuiFidTable {
    pub fn new() -> Self {
        Self {
            fids: HashMap::new(),
            next_fid: 0,
        }
    }

    /// Allocate a new fid walked to the given node.
    pub fn walk(&mut self, node: TuiNodeId) -> u32 {
        let fid = self.next_fid;
        self.next_fid += 1;
        self.fids.insert(fid, TuiFidState::Walked(node));
        fid
    }

    /// Open a walked fid.
    pub fn open(&mut self, fid: u32, reading: bool, writing: bool, cons_cursor: u64) -> bool {
        if let Some(state) = self.fids.get_mut(&fid) {
            if let TuiFidState::Walked(node) = state {
                let node = *node;
                *state = TuiFidState::Opened {
                    node,
                    cons_cursor,
                    reading,
                    writing,
                };
                return true;
            }
        }
        false
    }

    /// Close/remove a fid.
    pub fn clunk(&mut self, fid: u32) -> Option<TuiFidState> {
        self.fids.remove(&fid)
    }

    /// Get a fid's state.
    pub fn get(&self, fid: u32) -> Option<&TuiFidState> {
        self.fids.get(&fid)
    }

    /// Get a fid's state mutably.
    pub fn get_mut(&mut self, fid: u32) -> Option<&mut TuiFidState> {
        self.fids.get_mut(&fid)
    }
}

impl Default for TuiFidTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Walk a path from a starting node to resolve a target node.
pub fn walk_path(state: &TuiState, from: TuiNodeId, name: &str) -> Option<TuiNodeId> {
    match (from, name) {
        (TuiNodeId::Root, "ctl") => Some(TuiNodeId::Ctl),
        (TuiNodeId::Root, "new") => Some(TuiNodeId::New),
        (TuiNodeId::Root, "event") => Some(TuiNodeId::Event),
        (TuiNodeId::Root, "windows") => Some(TuiNodeId::WindowsDir),
        (TuiNodeId::WindowsDir, id_str) => {
            let w: u32 = id_str.parse().ok()?;
            // Verify window exists in some session
            state.sessions.iter().any(|s| s.window(w).is_some()).then_some(TuiNodeId::WindowDir(w))
        }
        (TuiNodeId::WindowDir(w), "ctl") => Some(TuiNodeId::WindowCtl(w)),
        (TuiNodeId::WindowDir(w), "cons") => Some(TuiNodeId::WindowCons(w)),
        (TuiNodeId::WindowDir(w), "screen") => Some(TuiNodeId::WindowScreen(w)),
        (TuiNodeId::WindowDir(w), "winname") => Some(TuiNodeId::WindowName(w)),
        (TuiNodeId::WindowDir(w), "layout") => Some(TuiNodeId::WindowLayout(w)),
        (TuiNodeId::WindowDir(w), "panes") => Some(TuiNodeId::PanesDir(w)),
        (TuiNodeId::PanesDir(w), id_str) => {
            let p: u32 = id_str.parse().ok()?;
            // Verify pane exists
            state.sessions.iter()
                .flat_map(|s| s.windows.iter())
                .find(|win| win.id == w)
                .and_then(|win| win.pane(p))
                .map(|_| TuiNodeId::PaneDir(w, p))
        }
        (TuiNodeId::PaneDir(w, p), "cons") => Some(TuiNodeId::PaneCons(w, p)),
        (TuiNodeId::PaneDir(w, p), "screen") => Some(TuiNodeId::PaneScreen(w, p)),
        _ => None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_paths() {
        let mut state = TuiState::default();
        let sid = state.create_session();

        // The first session gets window_id=1, pane_id=1
        let session = state.session(sid).unwrap();
        let window = &session.windows[0];
        let wid = window.id;
        let pid = window.panes[0].id;

        assert_eq!(walk_path(&state, TuiNodeId::Root, "ctl"), Some(TuiNodeId::Ctl));
        assert_eq!(walk_path(&state, TuiNodeId::Root, "new"), Some(TuiNodeId::New));
        assert_eq!(walk_path(&state, TuiNodeId::Root, "windows"), Some(TuiNodeId::WindowsDir));
        assert_eq!(
            walk_path(&state, TuiNodeId::WindowsDir, &wid.to_string()),
            Some(TuiNodeId::WindowDir(wid))
        );
        assert_eq!(
            walk_path(&state, TuiNodeId::WindowDir(wid), "cons"),
            Some(TuiNodeId::WindowCons(wid))
        );
        assert_eq!(
            walk_path(&state, TuiNodeId::WindowDir(wid), "panes"),
            Some(TuiNodeId::PanesDir(wid))
        );
        assert_eq!(
            walk_path(&state, TuiNodeId::PanesDir(wid), &pid.to_string()),
            Some(TuiNodeId::PaneDir(wid, pid))
        );
    }

    #[test]
    fn test_cons_buffer_write_read() {
        let mut buf = ConsBuffer::new(16);
        buf.write(b"hello");

        let mut out = [0u8; 16];
        let (n, cursor, gap) = buf.read(0, &mut out);
        assert_eq!(n, 5);
        assert_eq!(cursor, 5);
        assert!(!gap);
        assert_eq!(&out[..n], b"hello");
    }

    #[test]
    fn test_cons_buffer_gap_detection() {
        let mut buf = ConsBuffer::new(8);
        // Write 20 bytes into 8-byte ring (overwrites)
        buf.write(&[b'A'; 20]);

        let mut out = [0u8; 8];
        let (n, cursor, gap) = buf.read(0, &mut out);
        assert!(gap);
        assert_eq!(n, 0);
        assert_eq!(cursor, 20); // Advanced to write_offset
    }

    #[test]
    fn test_qid_path_uniqueness() {
        use std::collections::HashSet;
        let nodes = [
            TuiNodeId::Root,
            TuiNodeId::Ctl,
            TuiNodeId::New,
            TuiNodeId::Event,
            TuiNodeId::WindowsDir,
            TuiNodeId::WindowDir(0),
            TuiNodeId::WindowCtl(0),
            TuiNodeId::WindowCons(0),
            TuiNodeId::WindowScreen(0),
            TuiNodeId::WindowName(0),
            TuiNodeId::WindowLayout(0),
            TuiNodeId::PanesDir(0),
            TuiNodeId::PaneDir(0, 0),
            TuiNodeId::PaneCons(0, 0),
            TuiNodeId::PaneScreen(0, 0),
        ];
        let paths: HashSet<u64> = nodes.iter().map(|n| n.qid_path()).collect();
        assert_eq!(paths.len(), nodes.len(), "qid_path collision detected");
    }

    #[test]
    fn test_fid_table() {
        let mut table = TuiFidTable::new();
        let fid = table.walk(TuiNodeId::Ctl);
        assert!(matches!(table.get(fid), Some(TuiFidState::Walked(TuiNodeId::Ctl))));

        assert!(table.open(fid, true, false, 0));
        assert!(matches!(table.get(fid), Some(TuiFidState::Opened { .. })));

        assert!(table.clunk(fid).is_some());
        assert!(table.get(fid).is_none());
    }
}
