//! Pane layout — per-pane buffer state.

use std::collections::HashMap;

pub type PaneId = u32;

// ============================================================================
// Frame update types (decoded TuiFrame, passed from shell_handlers → compositor)
// ============================================================================

/// Cursor state for a server-buffered pane.
#[derive(Clone, Debug, Default)]
pub struct CursorState {
    pub x: u16,
    pub y: u16,
    pub visible: bool,
}

/// A decoded cell from a TuiFrame.
#[derive(Debug, Clone)]
pub struct CellUpdate {
    pub x: u16,
    pub y: u16,
    pub symbol: String,
    pub fg: ratatui::style::Color,
    pub bg: ratatui::style::Color,
    pub modifiers: ratatui::style::Modifier,
}

/// A decoded scroll operation from a TuiFrame.
#[derive(Debug, Clone)]
pub struct ScrollUpdate {
    pub top: u16,
    pub bottom: u16,
    /// Positive = scroll up (content moves up, new rows at bottom).
    /// Negative = scroll down.
    pub amount: i16,
}

/// Decoded frame content (full or incremental).
#[derive(Debug)]
pub enum FrameContent {
    Full { cols: u16, rows: u16, cells: Vec<CellUpdate> },
    Incremental { scrolls: Vec<ScrollUpdate>, deltas: Vec<CellUpdate> },
}

/// A fully decoded TuiFrame ready for the compositor to apply.
///
/// Produced by `shell_handlers::decode_tui_frame` and delivered via
/// `CompositorInput::ServerFrameCapnp`.
#[derive(Debug)]
pub struct FrameUpdate {
    pub pane_id: u32,
    pub generation: u64,
    pub cursor: CursorState,
    pub content: FrameContent,
}

// ============================================================================
// Pane storage variants
// ============================================================================

/// Source of a pane's content.
pub enum PaneSource {
    /// Frame received from TuiService.
    Server { pane_id: u32 },
    /// Rendered ANSI bytes from a client-owned ChatApp (Phase 4+).
    Private { app_id: u32 },
}

/// How a pane's content is stored client-side.
pub enum PaneStorage {
    /// Server pane fed by decoded TuiFrame diffs (native CLI / Capnp path).
    ServerBuf {
        buf: ratatui::buffer::Buffer,
        cursor: CursorState,
    },
    /// Server pane fed by ANSI bytes via avt::Vt (WASM/browser/ANSI path).
    ServerVt(avt::Vt),
    /// Private (client-owned) pane fed by ANSI from ChatApp.render().
    PrivateVt(avt::Vt),
}

// ============================================================================
// PaneState
// ============================================================================

/// Per-pane state: storage buffer + source.
pub struct PaneState {
    pub storage: PaneStorage,
    pub source: PaneSource,
}

impl PaneState {
    /// Create a server pane backed by an ANSI VT emulator (legacy / WASM path).
    pub fn server(pane_id: u32, cols: u16, rows: u16) -> Self {
        Self {
            storage: PaneStorage::ServerVt(
                avt::Vt::builder()
                    .size(cols as usize, rows as usize)
                    .build(),
            ),
            source: PaneSource::Server { pane_id },
        }
    }

    /// Create a server pane backed by a ratatui Buffer (native Capnp path).
    pub fn server_buf(pane_id: u32, cols: u16, rows: u16) -> Self {
        Self {
            storage: PaneStorage::ServerBuf {
                buf: ratatui::buffer::Buffer::empty(
                    ratatui::layout::Rect::new(0, 0, cols, rows),
                ),
                cursor: CursorState::default(),
            },
            source: PaneSource::Server { pane_id },
        }
    }

    /// Create a private client-owned pane backed by an ANSI VT emulator.
    pub fn private(app_id: u32, cols: u16, rows: u16) -> Self {
        Self {
            storage: PaneStorage::PrivateVt(
                avt::Vt::builder()
                    .size(cols as usize, rows as usize)
                    .build(),
            ),
            source: PaneSource::Private { app_id },
        }
    }

    /// Feed ANSI bytes into the VTE emulator (VT-backed panes only).
    /// No-op for `ServerBuf` panes.
    pub fn feed(&mut self, ansi: &[u8]) {
        match &mut self.storage {
            PaneStorage::ServerVt(vt) | PaneStorage::PrivateVt(vt) => {
                vt.feed_str(&String::from_utf8_lossy(ansi));
            }
            PaneStorage::ServerBuf { .. } => {}
        }
    }
}

// ============================================================================
// LayoutTree
// ============================================================================

/// Tracks per-pane buffer state keyed by pane_id (or app_id for private panes).
pub struct LayoutTree {
    panes: HashMap<PaneId, PaneState>,
    cols: u16,
    rows: u16,
}

impl LayoutTree {
    pub fn new(cols: u16, rows: u16) -> Self {
        Self { panes: HashMap::new(), cols, rows }
    }

    /// Get or create a server pane backed by ANSI VT (WASM / ANSI mode).
    pub fn get_or_create_server(&mut self, pane_id: u32) -> &mut PaneState {
        let cols = self.cols;
        let rows = self.rows;
        self.panes
            .entry(pane_id)
            .or_insert_with(|| PaneState::server(pane_id, cols, rows))
    }

    /// Get or create a private client-owned pane.
    pub fn get_or_create_private(&mut self, app_id: u32) -> &mut PaneState {
        let cols = self.cols;
        let rows = self.rows;
        self.panes
            .entry(app_id)
            .or_insert_with(|| PaneState::private(app_id, cols, rows))
    }

    pub fn get_pane(&self, pane_id: u32) -> Option<&PaneState> {
        self.panes.get(&pane_id)
    }

    pub fn get_pane_mut(&mut self, pane_id: u32) -> Option<&mut PaneState> {
        self.panes.get_mut(&pane_id)
    }

    pub fn remove_pane(&mut self, pane_id: u32) {
        self.panes.remove(&pane_id);
    }

    /// Apply a decoded TuiFrame to a server pane buffer (native Capnp path).
    ///
    /// Creates the pane as `ServerBuf` if it doesn't exist yet.
    /// Upgrades a `ServerVt` pane to `ServerBuf` on first Capnp frame.
    pub fn apply_server_frame(&mut self, frame: &FrameUpdate) {
        let cols = self.cols;
        let rows = self.rows;
        let pane_id = frame.pane_id;

        let pane = self.panes
            .entry(pane_id)
            .or_insert_with(|| PaneState::server_buf(pane_id, cols, rows));

        // Upgrade ServerVt → ServerBuf if this pane was previously ANSI-fed.
        if matches!(pane.storage, PaneStorage::ServerVt(_)) {
            pane.storage = PaneStorage::ServerBuf {
                buf: ratatui::buffer::Buffer::empty(
                    ratatui::layout::Rect::new(0, 0, cols, rows),
                ),
                cursor: CursorState::default(),
            };
        }

        if let PaneStorage::ServerBuf { buf, cursor } = &mut pane.storage {
            apply_frame_to_buffer(buf, &frame.content);
            cursor.x       = frame.cursor.x;
            cursor.y       = frame.cursor.y;
            cursor.visible = frame.cursor.visible;
        }
    }

    pub fn resize(&mut self, cols: u16, rows: u16) {
        self.cols = cols;
        self.rows = rows;
        // Rebuild all pane buffers at the new size.
        // Content is discarded — the server sends a full frame after its own resize.
        for pane in self.panes.values_mut() {
            match &mut pane.storage {
                PaneStorage::ServerVt(vt) | PaneStorage::PrivateVt(vt) => {
                    *vt = avt::Vt::builder()
                        .size(cols as usize, rows as usize)
                        .build();
                }
                PaneStorage::ServerBuf { buf, .. } => {
                    *buf = ratatui::buffer::Buffer::empty(
                        ratatui::layout::Rect::new(0, 0, cols, rows),
                    );
                }
            }
        }
    }
}

// ============================================================================
// Buffer helpers
// ============================================================================

/// Apply decoded frame content to a ratatui Buffer.
fn apply_frame_to_buffer(buf: &mut ratatui::buffer::Buffer, content: &FrameContent) {
    match content {
        FrameContent::Full { cols, rows, cells } => {
            // Replace the buffer entirely.
            *buf = ratatui::buffer::Buffer::empty(
                ratatui::layout::Rect::new(0, 0, *cols, *rows),
            );
            let width = *cols as usize;
            for cell in cells {
                let idx = cell.y as usize * width + cell.x as usize;
                if idx < buf.content.len() {
                    buf.content[idx].set_symbol(&cell.symbol);
                    buf.content[idx].fg       = cell.fg;
                    buf.content[idx].bg       = cell.bg;
                    buf.content[idx].modifier = cell.modifiers;
                }
            }
        }
        FrameContent::Incremental { scrolls, deltas } => {
            for scroll in scrolls {
                apply_scroll(buf, scroll);
            }
            let width = buf.area.width as usize;
            for cell in deltas {
                let idx = cell.y as usize * width + cell.x as usize;
                if idx < buf.content.len() {
                    buf.content[idx].set_symbol(&cell.symbol);
                    buf.content[idx].fg       = cell.fg;
                    buf.content[idx].bg       = cell.bg;
                    buf.content[idx].modifier = cell.modifiers;
                }
            }
        }
    }
}

/// Apply a scroll operation to a ratatui Buffer in-place.
fn apply_scroll(buf: &mut ratatui::buffer::Buffer, scroll: &ScrollUpdate) {
    let width = buf.area.width as usize;
    let height = buf.area.height as usize;
    let top    = scroll.top as usize;
    let bottom = (scroll.bottom as usize).min(height.saturating_sub(1));

    if scroll.amount > 0 {
        // Scroll up: copy row (top + n..=bottom) → (top..=bottom - n), clear tail.
        let n = scroll.amount as usize;
        for dst_y in top..=bottom {
            let src_y = dst_y + n;
            if src_y <= bottom {
                let src = src_y * width;
                let dst = dst_y * width;
                // Can't do copy_within because Cell is not Copy; clone row.
                let row: Vec<_> = buf.content[src..src + width].to_vec();
                buf.content[dst..dst + width].clone_from_slice(&row);
            } else {
                let dst = dst_y * width;
                for x in 0..width {
                    buf.content[dst + x] = ratatui::buffer::Cell::default();
                }
            }
        }
    } else if scroll.amount < 0 {
        // Scroll down: copy row (top..=bottom - n) → (top + n..=bottom), clear head.
        let n = (-scroll.amount) as usize;
        let mut dst_y = bottom;
        loop {
            if dst_y >= top + n {
                let src_y = dst_y - n;
                let src = src_y * width;
                let dst = dst_y * width;
                let row: Vec<_> = buf.content[src..src + width].to_vec();
                buf.content[dst..dst + width].clone_from_slice(&row);
            } else {
                let dst = dst_y * width;
                for x in 0..width {
                    buf.content[dst + x] = ratatui::buffer::Cell::default();
                }
            }
            if dst_y == top { break; }
            dst_y -= 1;
        }
    }
}
