//! TuiPaneBackend — native ratatui Backend writing directly to TuiPane.
//!
//! Double-buffer pattern: writes to a private staging buffer (no lock, ~8ns/cell),
//! then takes a brief write lock to swap into the pane buffer, sets damage,
//! and sends a damage notification.

use std::io;
use std::sync::Arc;

use ratatui::{
    backend::{Backend, ClearType, WindowSize},
    buffer::{Buffer, Cell},
    layout::{Position, Rect, Size},
};
use tokio::sync::RwLock;

use super::state::{TuiPane, TuiState};

/// A ratatui Backend that writes directly to a TuiPane's cell buffer.
///
/// Uses double-buffering: `draw()` writes to a private staging buffer,
/// then briefly locks the pane to swap the staging buffer in.
pub struct TuiPaneBackend {
    /// Staging buffer (private, no contention during writes).
    staging: Buffer,
    /// Shared TUI state for damage notification.
    state: Arc<RwLock<TuiState>>,
    /// Target session/window/pane IDs.
    session_id: u32,
    window_id: u32,
    pane_id: u32,
    /// Terminal dimensions.
    cols: u16,
    rows: u16,
}

impl TuiPaneBackend {
    /// Create a new TuiPaneBackend.
    pub fn new(
        state: Arc<RwLock<TuiState>>,
        session_id: u32,
        window_id: u32,
        pane_id: u32,
        cols: u16,
        rows: u16,
    ) -> Self {
        Self {
            staging: Buffer::empty(Rect::new(0, 0, cols, rows)),
            state,
            session_id,
            window_id,
            pane_id,
            cols,
            rows,
        }
    }

    /// Swap staging buffer into the pane (blocking, takes write lock).
    ///
    /// This is called after `draw()` completes. The brief write lock
    /// duration is proportional to a memcpy, not cell-by-cell iteration.
    pub fn flush_to_pane(&mut self) {
        let state = self.state.clone();
        let staging = std::mem::replace(
            &mut self.staging,
            Buffer::empty(Rect::new(0, 0, self.cols, self.rows)),
        );
        let session_id = self.session_id;
        let window_id = self.window_id;
        let pane_id = self.pane_id;

        // Use try_write to avoid blocking the current thread if possible.
        // In the Thread spawner context, this always succeeds.
        tokio::task::block_in_place(move || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                let mut state = state.write().await;
                if let Some(session) = state.session_mut(session_id) {
                    if let Some(window) = session.window_mut(window_id) {
                        if let Some(pane) = window.pane_mut(pane_id) {
                            let buf = pane.active_buffer_mut();
                            buf.buffer = staging;
                            buf.mark_all_dirty();
                        }
                    }
                }
                state.notify_damage(session_id, window_id, pane_id);
            });
        });
    }
}

impl Backend for TuiPaneBackend {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> Result<(), Self::Error>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        for (x, y, cell) in content {
            if x < self.cols && y < self.rows {
                self.staging[(x, y)] = cell.clone();
            }
        }
        // Flush staging to pane
        self.flush_to_pane();
        Ok(())
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
        self.staging.reset();
        Ok(())
    }

    fn clear_region(&mut self, _clear_type: ClearType) -> Result<(), Self::Error> {
        Ok(())
    }

    fn size(&self) -> Result<Size, Self::Error> {
        Ok(Size::new(self.cols, self.rows))
    }

    fn window_size(&mut self) -> Result<WindowSize, Self::Error> {
        Ok(WindowSize {
            columns_rows: Size::new(self.cols, self.rows),
            pixels: Size::new(0, 0),
        })
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Ingest structured binary data (from StructuredBackend) into a pane buffer.
///
/// Deserializes the packed cell tuples and writes them directly to the pane's buffer.
pub fn ingest_structured(pane: &mut TuiPane, data: &[u8]) -> io::Result<()> {
    let (_cols, _rows, cells) = waxterm::structured::decode_structured(data)?;

    let buf = pane.active_buffer_mut();
    for cell in cells {
        let area = buf.buffer.area();
        if cell.x < area.width && cell.y < area.height {
            let target = &mut buf.buffer[(cell.x, cell.y)];
            target.set_symbol(&cell.symbol);
            target.set_style(ratatui::style::Style::new()
                .fg(cell.fg)
                .bg(cell.bg)
                .add_modifier(cell.modifiers));
            buf.mark_dirty(cell.y);
        }
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backend_size() -> Result<(), Box<dyn std::error::Error>> {
        let state = Arc::new(RwLock::new(TuiState::default()));
        let backend = TuiPaneBackend::new(state, 1, 1, 1, 80, 24);
        assert_eq!(backend.size()?, Size::new(80, 24));
        Ok(())
    }
}
