//! Pane layout — per-pane avt::Vt state.

use std::collections::HashMap;

pub type PaneId = u32;

/// Source of a pane's content.
pub enum PaneSource {
    /// Frame received from TuiService.
    Server { pane_id: u32 },
    /// Rendered ANSI bytes from a client-owned ChatApp (Phase 4+).
    Private { app_id: u32 },
}

/// Per-pane state: VTE buffer + source.
pub struct PaneState {
    pub vt: avt::Vt,
    pub source: PaneSource,
}

impl PaneState {
    pub fn server(pane_id: u32, cols: u16, rows: u16) -> Self {
        Self {
            vt: avt::Vt::builder()
                .size(cols as usize, rows as usize)
                .build(),
            source: PaneSource::Server { pane_id },
        }
    }

    pub fn private(app_id: u32, cols: u16, rows: u16) -> Self {
        Self {
            vt: avt::Vt::builder()
                .size(cols as usize, rows as usize)
                .build(),
            source: PaneSource::Private { app_id },
        }
    }

    /// Feed ANSI bytes into the VTE parser.
    pub fn feed(&mut self, ansi: &[u8]) {
        self.vt.feed_str(&String::from_utf8_lossy(ansi));
    }
}

/// Tracks per-pane VTE state keyed by pane_id (or app_id for private panes).
pub struct LayoutTree {
    panes: HashMap<PaneId, PaneState>,
    cols: u16,
    rows: u16,
}

impl LayoutTree {
    pub fn new(cols: u16, rows: u16) -> Self {
        Self { panes: HashMap::new(), cols, rows }
    }

    pub fn get_or_create_server(&mut self, pane_id: u32) -> &mut PaneState {
        let cols = self.cols;
        let rows = self.rows;
        self.panes
            .entry(pane_id)
            .or_insert_with(|| PaneState::server(pane_id, cols, rows))
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

    pub fn resize(&mut self, cols: u16, rows: u16) {
        self.cols = cols;
        self.rows = rows;
        // Existing panes keep their current size; new panes will use new dims.
    }
}
