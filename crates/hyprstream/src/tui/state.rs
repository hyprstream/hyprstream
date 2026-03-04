//! Core types for the TUI display server state.
//!
//! The state hierarchy is: `TuiState` → `TuiSession` → `TuiWindow` → `TuiPane`.
//! Each pane owns a ratatui `Buffer` for cell storage and tracks damage via bitmask.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use ratatui::buffer::{Buffer, Cell};
use tokio::sync::broadcast;

// ============================================================================
// Cursor
// ============================================================================

/// Cursor shape (matches DEC private mode sequences).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CursorShape {
    #[default]
    Block,
    Underline,
    Bar,
    BlinkingBlock,
    BlinkingUnderline,
    BlinkingBar,
}

/// Cursor state for a pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorState {
    pub x: u16,
    pub y: u16,
    pub visible: bool,
    pub shape: CursorShape,
}

impl Default for CursorState {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            visible: true,
            shape: CursorShape::default(),
        }
    }
}

// ============================================================================
// Scroll Operations
// ============================================================================

/// A scroll operation recorded for incremental diff encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollOp {
    /// Top row of scroll region (inclusive, 0-indexed).
    pub top: u16,
    /// Bottom row of scroll region (inclusive, 0-indexed).
    pub bottom: u16,
    /// Positive = scroll up, negative = scroll down.
    pub amount: i16,
}

// ============================================================================
// Pane Buffer
// ============================================================================

/// A pane's cell buffer with damage tracking.
///
/// Wraps a ratatui `Buffer` and tracks which rows have been modified since
/// the last frame was sent. Uses a `u128` bitmask for rows 0–127; if the
/// pane has more than 128 rows, `force_full_frame` is set instead.
pub struct PaneBuffer {
    /// The cell buffer.
    pub buffer: Buffer,
    /// Bitmask of dirty rows (bit N = row N is dirty). Rows ≥128 set `force_full_frame`.
    pub damage_rows: u128,
    /// Scroll operations since last frame (capped at 256).
    pub scroll_log: Vec<ScrollOp>,
    /// If true, next frame must be a full frame (resize, alt screen switch, etc.).
    pub force_full_frame: bool,
}

impl PaneBuffer {
    /// Create a new buffer with the given dimensions.
    pub fn new(cols: u16, rows: u16) -> Self {
        Self {
            buffer: Buffer::empty(ratatui::layout::Rect::new(0, 0, cols, rows)),
            damage_rows: 0,
            scroll_log: Vec::new(),
            force_full_frame: true, // First frame is always full
        }
    }

    /// Mark a row as dirty.
    #[inline]
    pub fn mark_dirty(&mut self, row: u16) {
        if row < 128 {
            self.damage_rows |= 1u128 << row;
        } else {
            self.force_full_frame = true;
        }
    }

    /// Mark all rows as dirty.
    pub fn mark_all_dirty(&mut self) {
        self.damage_rows = u128::MAX;
    }

    /// Check if a row is dirty.
    #[inline]
    pub fn is_row_dirty(&self, row: u16) -> bool {
        if row < 128 {
            (self.damage_rows >> row) & 1 != 0
        } else {
            self.force_full_frame
        }
    }

    /// Count dirty rows.
    pub fn dirty_count(&self) -> u32 {
        self.damage_rows.count_ones()
    }

    /// Clear damage tracking (called after frame is sent).
    pub fn clear_damage(&mut self) {
        self.damage_rows = 0;
        self.scroll_log.clear();
        self.force_full_frame = false;
    }

    /// Record a scroll operation.
    pub fn record_scroll(&mut self, op: ScrollOp) {
        if self.scroll_log.len() < 256 {
            self.scroll_log.push(op);
        } else {
            // Too many scrolls — force full frame
            self.force_full_frame = true;
        }
    }

    /// Resize the buffer. Sets `force_full_frame`.
    pub fn resize(&mut self, cols: u16, rows: u16) {
        self.buffer.resize(ratatui::layout::Rect::new(0, 0, cols, rows));
        self.force_full_frame = true;
        self.damage_rows = 0;
        self.scroll_log.clear();
    }

    /// Get a cell by position.
    pub fn cell(&self, x: u16, y: u16) -> Option<&Cell> {
        let area = self.buffer.area();
        if x < area.width && y < area.height {
            Some(&self.buffer[(x, y)])
        } else {
            None
        }
    }
}

// ============================================================================
// Ingestion Mode
// ============================================================================

/// How a pane receives content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IngestionMode {
    /// Direct ratatui Backend writes (TuiPaneBackend).
    #[default]
    Direct,
    /// ANSI escape sequences parsed by VTE.
    Ansi,
    /// Packed binary cells from StructuredBackend.
    Structured,
}

// ============================================================================
// TUI Pane
// ============================================================================

/// A single terminal pane with primary and alternate screen buffers.
pub struct TuiPane {
    /// Pane identifier (unique within a window).
    pub id: u32,
    /// Primary screen buffer.
    pub primary: PaneBuffer,
    /// Alternate screen buffer (lazy-allocated on `?1049h`).
    pub alternate: Option<PaneBuffer>,
    /// Which buffer is currently active (false = primary, true = alternate).
    pub using_alternate: bool,
    /// Scrollback history (lines that scrolled off the top of primary).
    pub scrollback: VecDeque<Vec<Cell>>,
    /// Maximum scrollback lines.
    pub max_scrollback: usize,
    /// Cursor state.
    pub cursor: CursorState,
    /// How this pane receives content.
    pub ingestion_mode: IngestionMode,
    /// Pane title (set via OSC 0/2).
    pub title: String,
}

impl TuiPane {
    /// Create a new pane.
    pub fn new(id: u32, cols: u16, rows: u16, max_scrollback: usize) -> Self {
        Self {
            id,
            primary: PaneBuffer::new(cols, rows),
            alternate: None,
            using_alternate: false,
            scrollback: VecDeque::new(),
            max_scrollback,
            cursor: CursorState::default(),
            ingestion_mode: IngestionMode::default(),
            title: String::new(),
        }
    }

    /// Get the active buffer.
    pub fn active_buffer(&self) -> &PaneBuffer {
        if self.using_alternate {
            self.alternate.as_ref().unwrap_or(&self.primary)
        } else {
            &self.primary
        }
    }

    /// Get the active buffer mutably.
    pub fn active_buffer_mut(&mut self) -> &mut PaneBuffer {
        if self.using_alternate {
            self.alternate.as_mut().unwrap_or(&mut self.primary)
        } else {
            &mut self.primary
        }
    }

    /// Get terminal dimensions (from active buffer).
    pub fn size(&self) -> (u16, u16) {
        let area = self.active_buffer().buffer.area();
        (area.width, area.height)
    }

    /// Switch to alternate screen buffer (`\x1b[?1049h`).
    pub fn enter_alternate_screen(&mut self) {
        if !self.using_alternate {
            let (cols, rows) = self.size();
            self.alternate.get_or_insert_with(|| PaneBuffer::new(cols, rows));
            self.using_alternate = true;
            // Force full frame on screen switch
            if let Some(alt) = &mut self.alternate {
                alt.force_full_frame = true;
            }
        }
    }

    /// Switch back to primary screen buffer (`\x1b[?1049l`).
    pub fn leave_alternate_screen(&mut self) {
        if self.using_alternate {
            self.using_alternate = false;
            self.primary.force_full_frame = true;
        }
    }

    /// Resize this pane.
    pub fn resize(&mut self, cols: u16, rows: u16) {
        self.primary.resize(cols, rows);
        if let Some(alt) = &mut self.alternate {
            alt.resize(cols, rows);
        }
    }

    /// Push a line to scrollback (when content scrolls off the top).
    pub fn push_scrollback(&mut self, line: Vec<Cell>) {
        if self.scrollback.len() >= self.max_scrollback {
            self.scrollback.pop_front();
        }
        self.scrollback.push_back(line);
    }
}

// ============================================================================
// Layout
// ============================================================================

/// Layout tree node for window pane arrangement.
#[derive(Debug, Clone)]
pub enum LayoutNode {
    /// A leaf pane.
    Leaf { pane_id: u32 },
    /// Horizontal split (top/bottom).
    HSplit {
        ratio: f32,
        first: Box<LayoutNode>,
        second: Box<LayoutNode>,
    },
    /// Vertical split (left/right).
    VSplit {
        ratio: f32,
        first: Box<LayoutNode>,
        second: Box<LayoutNode>,
    },
}

impl LayoutNode {
    /// Collect all pane IDs in this layout tree.
    pub fn pane_ids(&self) -> Vec<u32> {
        match self {
            LayoutNode::Leaf { pane_id } => vec![*pane_id],
            LayoutNode::HSplit { first, second, .. }
            | LayoutNode::VSplit { first, second, .. } => {
                let mut ids = first.pane_ids();
                ids.extend(second.pane_ids());
                ids
            }
        }
    }

    /// Remove a pane from the layout tree, collapsing its parent split.
    /// Returns the pruned tree, or None if the tree is now empty.
    pub fn prune(&self, target_pane_id: u32) -> Option<LayoutNode> {
        match self {
            LayoutNode::Leaf { pane_id } => {
                if *pane_id == target_pane_id {
                    None
                } else {
                    Some(self.clone())
                }
            }
            LayoutNode::HSplit { ratio, first, second } => {
                let f = first.prune(target_pane_id);
                let s = second.prune(target_pane_id);
                match (f, s) {
                    (Some(f), Some(s)) => Some(LayoutNode::HSplit {
                        ratio: *ratio,
                        first: Box::new(f),
                        second: Box::new(s),
                    }),
                    (Some(node), None) | (None, Some(node)) => Some(node),
                    (None, None) => None,
                }
            }
            LayoutNode::VSplit { ratio, first, second } => {
                let f = first.prune(target_pane_id);
                let s = second.prune(target_pane_id);
                match (f, s) {
                    (Some(f), Some(s)) => Some(LayoutNode::VSplit {
                        ratio: *ratio,
                        first: Box::new(f),
                        second: Box::new(s),
                    }),
                    (Some(node), None) | (None, Some(node)) => Some(node),
                    (None, None) => None,
                }
            }
        }
    }
}

// ============================================================================
// Window
// ============================================================================

/// A window containing one or more panes in a layout tree.
pub struct TuiWindow {
    /// Window identifier.
    pub id: u32,
    /// Window name/title.
    pub name: String,
    /// Layout tree.
    pub layout: LayoutNode,
    /// All panes in this window.
    pub panes: Vec<TuiPane>,
    /// Currently focused pane ID.
    pub active_pane_id: u32,
}

impl TuiWindow {
    /// Create a new window with a single pane.
    pub fn new(id: u32, pane_id: u32, cols: u16, rows: u16, max_scrollback: usize) -> Self {
        let pane = TuiPane::new(pane_id, cols, rows, max_scrollback);
        Self {
            id,
            name: format!("window-{}", id),
            layout: LayoutNode::Leaf { pane_id },
            panes: vec![pane],
            active_pane_id: pane_id,
        }
    }

    /// Find a pane by ID.
    pub fn pane(&self, pane_id: u32) -> Option<&TuiPane> {
        self.panes.iter().find(|p| p.id == pane_id)
    }

    /// Find a pane by ID mutably.
    pub fn pane_mut(&mut self, pane_id: u32) -> Option<&mut TuiPane> {
        self.panes.iter_mut().find(|p| p.id == pane_id)
    }

    /// Get the active pane.
    pub fn active_pane(&self) -> Option<&TuiPane> {
        self.pane(self.active_pane_id)
    }

    /// Get the active pane mutably.
    pub fn active_pane_mut(&mut self) -> Option<&mut TuiPane> {
        self.pane_mut(self.active_pane_id)
    }
}

// ============================================================================
// Session
// ============================================================================

/// A TUI session containing one or more windows.
pub struct TuiSession {
    /// Session identifier.
    pub id: u32,
    /// All windows in this session.
    pub windows: Vec<TuiWindow>,
    /// Currently active window ID.
    pub active_window_id: u32,
}

impl TuiSession {
    /// Create a new session with one window containing one pane.
    pub fn new(id: u32, window_id: u32, pane_id: u32, cols: u16, rows: u16, max_scrollback: usize) -> Self {
        let window = TuiWindow::new(window_id, pane_id, cols, rows, max_scrollback);
        Self {
            id,
            windows: vec![window],
            active_window_id: window_id,
        }
    }

    /// Find a window by ID.
    pub fn window(&self, window_id: u32) -> Option<&TuiWindow> {
        self.windows.iter().find(|w| w.id == window_id)
    }

    /// Find a window by ID mutably.
    pub fn window_mut(&mut self, window_id: u32) -> Option<&mut TuiWindow> {
        self.windows.iter_mut().find(|w| w.id == window_id)
    }

    /// Get the active window.
    pub fn active_window(&self) -> Option<&TuiWindow> {
        self.window(self.active_window_id)
    }

    /// Get the active window mutably.
    pub fn active_window_mut(&mut self) -> Option<&mut TuiWindow> {
        self.window_mut(self.active_window_id)
    }
}

// ============================================================================
// Events
// ============================================================================

/// Events broadcast from the TUI state to viewers and the 9P filesystem.
#[derive(Debug, Clone)]
pub enum TuiEvent {
    /// A pane's buffer was modified.
    Damage { session_id: u32, window_id: u32, pane_id: u32 },
    /// A window was created.
    WindowCreated { session_id: u32, window_id: u32 },
    /// A window was closed.
    WindowClosed { session_id: u32, window_id: u32 },
    /// A pane was created.
    PaneCreated { session_id: u32, window_id: u32, pane_id: u32 },
    /// A pane was closed.
    PaneClosed { session_id: u32, window_id: u32, pane_id: u32 },
    /// A session was created.
    SessionCreated { session_id: u32 },
    /// A session was destroyed.
    SessionDestroyed { session_id: u32 },
    /// Terminal resized.
    Resized { cols: u16, rows: u16 },
}

// ============================================================================
// TuiState (top-level)
// ============================================================================

/// Top-level TUI multiplexer state.
///
/// Owns all sessions and provides a broadcast channel for state change events.
/// The composite buffer and generation counter are used by the frame loop
/// to detect changes and encode diffs.
pub struct TuiState {
    /// All sessions.
    pub sessions: Vec<TuiSession>,
    /// Monotonically increasing generation counter.
    generation: AtomicU64,
    /// Event broadcast sender.
    pub event_tx: broadcast::Sender<TuiEvent>,
    /// Next ID counters.
    next_session_id: u32,
    next_window_id: u32,
    next_pane_id: u32,
    /// Default terminal dimensions.
    pub default_cols: u16,
    pub default_rows: u16,
    /// Default scrollback lines per pane.
    pub default_scrollback: usize,
}

impl TuiState {
    /// Create a new TUI state.
    pub fn new(cols: u16, rows: u16, scrollback: usize) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            sessions: Vec::new(),
            generation: AtomicU64::new(0),
            event_tx,
            next_session_id: 1,
            next_window_id: 1,
            next_pane_id: 1,
            default_cols: cols,
            default_rows: rows,
            default_scrollback: scrollback,
        }
    }

    /// Get the current generation.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Increment and return the new generation.
    pub fn next_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Create a new session.
    pub fn create_session(&mut self) -> u32 {
        let session_id = self.next_session_id;
        self.next_session_id += 1;
        let window_id = self.next_window_id;
        self.next_window_id += 1;
        let pane_id = self.next_pane_id;
        self.next_pane_id += 1;

        let session = TuiSession::new(
            session_id,
            window_id,
            pane_id,
            self.default_cols,
            self.default_rows,
            self.default_scrollback,
        );
        self.sessions.push(session);

        let _ = self.event_tx.send(TuiEvent::SessionCreated { session_id });
        session_id
    }

    /// Find a session by ID.
    pub fn session(&self, session_id: u32) -> Option<&TuiSession> {
        self.sessions.iter().find(|s| s.id == session_id)
    }

    /// Find a session by ID mutably.
    pub fn session_mut(&mut self, session_id: u32) -> Option<&mut TuiSession> {
        self.sessions.iter_mut().find(|s| s.id == session_id)
    }

    /// Create a new window in a session.
    pub fn create_window(&mut self, session_id: u32) -> Option<u32> {
        let window_id = self.next_window_id;
        self.next_window_id += 1;
        let pane_id = self.next_pane_id;
        self.next_pane_id += 1;

        let session = self.sessions.iter_mut().find(|s| s.id == session_id)?;
        let window = TuiWindow::new(
            window_id,
            pane_id,
            self.default_cols,
            self.default_rows,
            self.default_scrollback,
        );
        session.windows.push(window);

        let _ = self.event_tx.send(TuiEvent::WindowCreated { session_id, window_id });
        Some(window_id)
    }

    /// Allocate a new pane ID.
    pub fn next_pane_id(&mut self) -> u32 {
        let id = self.next_pane_id;
        self.next_pane_id += 1;
        id
    }

    /// Close a window in a session. Returns true if the window was found and removed.
    pub fn close_window(&mut self, session_id: u32, window_id: u32) -> bool {
        let session = match self.session_mut(session_id) {
            Some(s) => s,
            None => return false,
        };
        let pos = match session.windows.iter().position(|w| w.id == window_id) {
            Some(p) => p,
            None => return false,
        };
        session.windows.remove(pos);
        if session.active_window_id == window_id {
            session.active_window_id = session.windows.first().map(|w| w.id).unwrap_or(0);
        }
        let _ = self.event_tx.send(TuiEvent::WindowClosed { session_id, window_id });
        true
    }

    /// Focus a window in a session. Returns true if the window exists.
    pub fn focus_window(&mut self, session_id: u32, window_id: u32) -> bool {
        let session = match self.session_mut(session_id) {
            Some(s) => s,
            None => return false,
        };
        if session.window(window_id).is_none() {
            return false;
        }
        session.active_window_id = window_id;
        true
    }

    /// Split the active pane in the active window. Returns (new_pane_id, cols, rows) on success.
    pub fn split_pane(
        &mut self,
        session_id: u32,
        horizontal: bool,
        ratio: f32,
    ) -> Option<(u32, u16, u16)> {
        let new_pane_id = self.next_pane_id();
        let cols = self.default_cols;
        let rows = self.default_rows;
        let scrollback = self.default_scrollback;

        let session = self.session_mut(session_id)?;
        let window = session.active_window_mut()?;
        let _active_id = window.active_pane_id;

        let new_pane = TuiPane::new(new_pane_id, cols, rows, scrollback);
        window.panes.push(new_pane);

        let old_layout = window.layout.clone();
        let new_leaf = LayoutNode::Leaf { pane_id: new_pane_id };
        window.layout = if horizontal {
            LayoutNode::HSplit {
                ratio,
                first: Box::new(old_layout),
                second: Box::new(new_leaf),
            }
        } else {
            LayoutNode::VSplit {
                ratio,
                first: Box::new(old_layout),
                second: Box::new(new_leaf),
            }
        };

        let window_id = window.id;
        let _ = self.event_tx.send(TuiEvent::PaneCreated {
            session_id,
            window_id,
            pane_id: new_pane_id,
        });
        Some((new_pane_id, cols, rows))
    }

    /// Close a pane in the active window. Returns true if found and removed.
    pub fn close_pane(&mut self, session_id: u32, pane_id: u32) -> bool {
        let session = match self.session_mut(session_id) {
            Some(s) => s,
            None => return false,
        };
        let window = match session.active_window_mut() {
            Some(w) => w,
            None => return false,
        };

        // Don't close the last pane
        if window.panes.len() <= 1 {
            return false;
        }

        let pos = match window.panes.iter().position(|p| p.id == pane_id) {
            Some(p) => p,
            None => return false,
        };
        window.panes.remove(pos);

        // Prune from layout tree
        if let Some(pruned) = window.layout.prune(pane_id) {
            window.layout = pruned;
        }

        if window.active_pane_id == pane_id {
            window.active_pane_id = window.panes.first().map(|p| p.id).unwrap_or(0);
        }

        let window_id = window.id;
        let _ = self.event_tx.send(TuiEvent::PaneClosed {
            session_id,
            window_id,
            pane_id,
        });
        true
    }

    /// Focus a pane in the active window. Returns true if the pane exists.
    pub fn focus_pane(&mut self, session_id: u32, pane_id: u32) -> bool {
        let session = match self.session_mut(session_id) {
            Some(s) => s,
            None => return false,
        };
        let window = match session.active_window_mut() {
            Some(w) => w,
            None => return false,
        };
        if window.pane(pane_id).is_none() {
            return false;
        }
        window.active_pane_id = pane_id;
        true
    }

    /// Subscribe to TUI events.
    pub fn subscribe(&self) -> broadcast::Receiver<TuiEvent> {
        self.event_tx.subscribe()
    }

    /// Notify that a pane has been damaged (content changed).
    pub fn notify_damage(&self, session_id: u32, window_id: u32, pane_id: u32) {
        self.next_generation();
        let _ = self.event_tx.send(TuiEvent::Damage {
            session_id,
            window_id,
            pane_id,
        });
    }
}

impl Default for TuiState {
    fn default() -> Self {
        Self::new(80, 24, 2000)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damage_bitmask_boundaries() {
        let mut buf = PaneBuffer::new(80, 24);
        buf.clear_damage();

        // Row 0
        buf.mark_dirty(0);
        assert!(buf.is_row_dirty(0));
        assert!(!buf.is_row_dirty(1));

        // Row 63
        buf.mark_dirty(63);
        assert!(buf.is_row_dirty(63));

        // Row 127
        buf.mark_dirty(127);
        assert!(buf.is_row_dirty(127));
        assert!(!buf.force_full_frame);

        // Row 128+ triggers force_full_frame
        buf.mark_dirty(128);
        assert!(buf.force_full_frame);
    }

    #[test]
    fn test_scroll_log_overflow() {
        let mut buf = PaneBuffer::new(80, 24);
        buf.clear_damage();

        for i in 0..256 {
            buf.record_scroll(ScrollOp { top: 0, bottom: 23, amount: 1 });
            assert!(!buf.force_full_frame, "should not force full frame at scroll {}", i);
        }

        // 257th scroll should trigger force_full_frame
        buf.record_scroll(ScrollOp { top: 0, bottom: 23, amount: 1 });
        assert!(buf.force_full_frame);
    }

    #[test]
    fn test_resize_triggers_full_frame() {
        let mut buf = PaneBuffer::new(80, 24);
        buf.clear_damage();
        assert!(!buf.force_full_frame);

        buf.resize(120, 40);
        assert!(buf.force_full_frame);
    }

    #[test]
    fn test_session_creation() {
        let mut state = TuiState::default();
        let sid = state.create_session();
        assert_eq!(sid, 1);

        let session = state.session(sid).unwrap();
        assert_eq!(session.windows.len(), 1);
        assert_eq!(session.windows[0].panes.len(), 1);
    }

    #[test]
    fn test_window_creation() {
        let mut state = TuiState::default();
        let sid = state.create_session();
        let wid = state.create_window(sid).unwrap();

        let session = state.session(sid).unwrap();
        assert_eq!(session.windows.len(), 2);
        assert!(session.window(wid).is_some());
    }

    #[test]
    fn test_alternate_screen() {
        let mut pane = TuiPane::new(1, 80, 24, 2000);
        assert!(!pane.using_alternate);
        assert!(pane.alternate.is_none());

        pane.enter_alternate_screen();
        assert!(pane.using_alternate);
        assert!(pane.alternate.is_some());
        assert!(pane.alternate.as_ref().unwrap().force_full_frame);

        pane.leave_alternate_screen();
        assert!(!pane.using_alternate);
        assert!(pane.primary.force_full_frame);
    }

    #[test]
    fn test_generation_counter() {
        let state = TuiState::default();
        assert_eq!(state.generation(), 0);
        assert_eq!(state.next_generation(), 1);
        assert_eq!(state.next_generation(), 2);
        assert_eq!(state.generation(), 2);
    }
}
