//! hyprstream-compositor — pure Rust TUI compositor.
//!
//! WASM-safe: no I/O, no ZMQ, no tokio, no libc.
//! Compiles natively and to `wasm32-wasip1`.
//!
//! # Usage (native CLI)
//!
//! ```rust,ignore
//! let mut compositor = Compositor::new(cols, rows, session_id, viewer_id, windows, models);
//!
//! // In event loop:
//! let outputs = compositor.handle(CompositorInput::KeyPress(key));
//! for output in outputs {
//!     match output {
//!         CompositorOutput::Redraw => { terminal.draw(|f| compositor.render(f))?; }
//!         CompositorOutput::Rpc(req) => { rpc_adapter.dispatch(req, ...).await; }
//!         CompositorOutput::RouteInput { .. } => { /* Phase 4 */ }
//!         CompositorOutput::Quit => break,
//!     }
//! }
//! ```

pub mod background;
pub mod chrome;
pub mod layout;
pub mod render;
pub mod theme;

pub use background::{BackgroundState, BackgroundStyle, ALL_STYLES, PREVIEW_H, PREVIEW_W};
pub use chrome::{
    ChromeOutput, ContainerEntry, ConversationPickerEntry, ModelEntry, PaneSummary, RpcRequest,
    ServiceEntry, ServiceMode, ShellChrome, ShellMode, Toast, ToastLevel, WindowSummary,
    WorkerEntry, MENU_ITEMS, keypress_to_bytes, LOCAL_ID_BIT, is_local_id,
};
pub use layout::{
    CellUpdate, CursorState, FrameContent, FrameUpdate, LayoutTree, PaneId, PaneSource,
    PaneState, PaneStorage, ScrollUpdate,
};

// ============================================================================
// CompositorInput / CompositorOutput
// ============================================================================

/// Events delivered to the compositor from the event loop.
pub enum CompositorInput {
    /// ANSI frame bytes from TuiService for a server-managed pane (ANSI / WASM path).
    ServerFrame { pane_id: u32, ansi: Vec<u8> },
    /// Decoded TuiFrame from TuiService (Capnp display mode, native CLI path).
    ServerFrameCapnp { frame: FrameUpdate },
    /// Rendered ANSI bytes from a client-owned ChatApp (Phase 4).
    AppFrame { app_id: u32, ansi: Vec<u8> },
    /// Updated window list from TuiService.
    WindowList(Vec<WindowSummary>),
    /// A server-managed pane was closed.
    PaneClosed { pane_id: u32 },
    /// Keyboard input from the user.
    KeyPress(waxterm::input::KeyPress),
    /// Terminal resize.
    Resize(u16, u16),
    /// A client-owned ChatApp exited (Phase 4).
    AppExited { app_id: u32 },
    /// Updated service list from polling.
    ServiceList(Vec<crate::chrome::ServiceEntry>),
    /// Updated worker/sandbox list from polling.
    WorkerList {
        sandboxes: Vec<crate::chrome::WorkerEntry>,
        pool_summary: String,
    },
}

/// Actions returned by `Compositor::handle`.
pub enum CompositorOutput {
    /// The compositor state changed — caller should re-render.
    Redraw,
    /// A pure RPC request — dispatch to `ShellRpcAdapter` or encode as OSC IPC.
    Rpc(RpcRequest),
    /// Route key bytes to a client-owned ChatApp (Phase 4).
    RouteInput { app_id: u32, data: Vec<u8> },
    /// Session should exit.
    Quit,
}

// ============================================================================
// Compositor
// ============================================================================

/// The central compositor state machine.
pub struct Compositor {
    pub chrome: ShellChrome,
    pub layout: LayoutTree,
    cols: u16,
    rows: u16,
}

impl Compositor {
    pub fn new(
        cols: u16,
        rows: u16,
        session_id: u32,
        viewer_id: u32,
        windows: Vec<WindowSummary>,
        models: Vec<ModelEntry>,
    ) -> Self {
        let pane_rows = rows.saturating_sub(5);
        let pane_cols = cols.saturating_sub(2);
        Self {
            chrome: ShellChrome::new(pane_cols, pane_rows, session_id, viewer_id, windows, models),
            layout: LayoutTree::new(pane_cols, pane_rows),
            cols,
            rows,
        }
    }

    /// Returns the area occupied by the pane block (inside border) for the given frame area.
    /// Used by shell_handlers to position overlay widgets that aren't WASM-safe.
    pub fn pane_block_area(&self, frame_area: ratatui::layout::Rect) -> ratatui::layout::Rect {
        use ratatui::layout::{Constraint, Layout};
        if matches!(self.chrome.mode, ShellMode::Fullscreen) {
            return frame_area;
        }
        let [_, pane_block, _, _] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(frame_area);
        pane_block
    }

    /// The active pane ID (from the focused window's active pane).
    pub fn active_pane_id(&self) -> u32 {
        self.chrome.active_pane_id()
    }

    /// Render the full chrome + pane content into a ratatui frame.
    pub fn render(&self, frame: &mut ratatui::Frame) {
        render::draw(frame, &self.chrome, &self.layout);
    }

    /// Process a compositor input event, returning a list of actions for the
    /// event loop to dispatch.
    pub fn handle(&mut self, input: CompositorInput) -> Vec<CompositorOutput> {
        match input {
            CompositorInput::ServerFrame { pane_id, ansi } => {
                let pane = self.layout.get_or_create_server(pane_id);
                pane.feed(&ansi);
                vec![CompositorOutput::Redraw]
            }

            CompositorInput::ServerFrameCapnp { frame } => {
                self.layout.apply_server_frame(&frame);
                vec![CompositorOutput::Redraw]
            }

            CompositorInput::AppFrame { app_id, ansi } => {
                if let Some(pane) = self.layout.get_pane_mut(app_id) {
                    pane.feed(&ansi);
                    vec![CompositorOutput::Redraw]
                } else {
                    vec![]
                }
            }

            CompositorInput::WindowList(wins) => {
                if self.chrome.update_windows(wins) {
                    vec![CompositorOutput::Redraw]
                } else {
                    vec![]
                }
            }

            CompositorInput::PaneClosed { pane_id } => {
                self.layout.remove_pane(pane_id);
                vec![CompositorOutput::Redraw]
            }

            CompositorInput::KeyPress(key) => {
                let chrome_outputs = self.chrome.handle_key(key);
                chrome_outputs
                    .into_iter()
                    .map(|co| match co {
                        ChromeOutput::Redraw    => CompositorOutput::Redraw,
                        ChromeOutput::Rpc(req)  => match req {
                            RpcRequest::Quit    => CompositorOutput::Quit,
                            other               => CompositorOutput::Rpc(other),
                        },
                        ChromeOutput::RouteInput { app_id, data }
                            => CompositorOutput::RouteInput { app_id, data },
                    })
                    .collect()
            }

            CompositorInput::Resize(cols, rows) => {
                self.cols = cols;
                self.rows = rows;
                let pane_rows = rows.saturating_sub(5);
                let pane_cols = cols.saturating_sub(2);
                self.chrome.cols      = pane_cols;
                self.chrome.pane_rows = pane_rows;
                self.layout.resize(pane_cols, pane_rows);
                vec![CompositorOutput::Redraw]
            }

            CompositorInput::ServiceList(entries) => {
                self.chrome.update_service_list(entries);
                if let ShellMode::ServiceManager { ref mut selected } = self.chrome.mode {
                    *selected = (*selected).min(self.chrome.service_list.len().saturating_sub(1));
                }
                vec![CompositorOutput::Redraw]
            }

            CompositorInput::WorkerList { sandboxes, pool_summary } => {
                self.chrome.update_worker_list(sandboxes, pool_summary);
                if let ShellMode::WorkerManager { ref mut sandbox_sel, ref mut container_sel, .. } = self.chrome.mode {
                    *sandbox_sel = (*sandbox_sel).min(self.chrome.worker_list.len().saturating_sub(1));
                    if let Some(sb) = self.chrome.worker_list.get(*sandbox_sel) {
                        *container_sel = (*container_sel).min(sb.containers.len().saturating_sub(1));
                    } else {
                        *container_sel = 0;
                    }
                }
                vec![CompositorOutput::Redraw]
            }

            CompositorInput::AppExited { app_id } => {
                self.layout.remove_pane(app_id);
                self.chrome.private_panes.remove(&app_id);

                // Find and close the server-side window that contained this pane.
                let mut out = vec![CompositorOutput::Redraw];
                if let Some(win_idx) = self.chrome.windows.iter().position(|w| {
                    w.panes.iter().any(|p| p.id == app_id)
                }) {
                    let window_id = self.chrome.windows[win_idx].id;
                    self.chrome.windows.remove(win_idx);
                    if self.chrome.windows.is_empty() {
                        self.chrome.active_win = 0;
                    } else {
                        self.chrome.active_win =
                            self.chrome.active_win.min(self.chrome.windows.len() - 1);
                    }
                    out.push(CompositorOutput::Rpc(RpcRequest::CloseWindow {
                        session_id: self.chrome.session_id,
                        window_id,
                    }));
                }
                out
            }
        }
    }
}
