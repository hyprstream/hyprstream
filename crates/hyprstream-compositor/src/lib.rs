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
    ChromeOutput, ContainerEntry, ConversationPickerEntry, ImageEntry, InputDialog, InputField,
    ModelEntry, PaneSummary, RpcRequest,
    ServiceEntry, ServiceMode, ShellChrome, ShellMode, Toast, ToastLevel, WindowSummary,
    WorkerEntry, WorkerTab, MENU_ITEMS, keypress_to_bytes, LOCAL_ID_BIT, is_local_id,
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
    /// Mouse click at terminal (col, row) — 0-indexed.
    MouseClick { col: u16, row: u16 },
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
    /// Updated image list.
    WorkerImageList {
        images: Vec<crate::chrome::ImageEntry>,
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
        let pane_rows = rows.saturating_sub(4);
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
        let [_, pane_block, _] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
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
                let co = self.chrome.handle_key(key);
                self.dispatch_chrome(co)
            }

            CompositorInput::MouseClick { col, row } => {
                self.handle_mouse_click(col, row)
            }

            CompositorInput::Resize(cols, rows) => {
                self.cols = cols;
                self.rows = rows;
                let pane_rows = rows.saturating_sub(4);
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

            CompositorInput::WorkerImageList { images } => {
                self.chrome.image_list = images;
                if let ShellMode::WorkerManager { ref mut image_sel, .. } = self.chrome.mode {
                    *image_sel = (*image_sel).min(self.chrome.image_list.len().saturating_sub(1));
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

    fn dispatch_chrome(&mut self, chrome_outputs: Vec<ChromeOutput>) -> Vec<CompositorOutput> {
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

    /// Handle a mouse click at terminal position (col, row), 0-indexed.
    /// Checks for close-button clicks, menu item clicks, and modal list clicks.
    fn handle_mouse_click(&mut self, col: u16, row: u16) -> Vec<CompositorOutput> {
        use crate::render::{centered_rect, modal_percentages};
        use crate::theme::CLOSE_BUTTON_WIDTH;

        match self.chrome.mode {
            // Pane window close button: top border is row 1.
            ShellMode::Normal => {
                let pane_top_row = 1u16;
                if row == pane_top_row
                    && col >= self.cols.saturating_sub(1 + CLOSE_BUTTON_WIDTH)
                    && col < self.cols.saturating_sub(1)
                {
                    if let Some(win) = self.chrome.windows.get(self.chrome.active_win) {
                        let sid = self.chrome.session_id;
                        let wid = win.id;
                        return vec![CompositorOutput::Rpc(RpcRequest::CloseWindow {
                            session_id: sid, window_id: wid,
                        })];
                    }
                }
                vec![]
            }

            // Start menu: close button, item clicks, click-outside-to-dismiss.
            ShellMode::StartMenu { .. } => {
                let popup_w: u16 = 26;
                let popup_h: u16 = MENU_ITEMS.len() as u16 + 2;
                let popup_y = self.rows.saturating_sub(popup_h + 1);
                let inner_y = popup_y + 1;
                let inner_x = 1u16;
                let inner_right = popup_w.saturating_sub(1);

                // Close button on popup top border.
                if row == popup_y
                    && col >= popup_w.saturating_sub(1 + CLOSE_BUTTON_WIDTH)
                    && col < popup_w.saturating_sub(1)
                {
                    let out = self.chrome.handle_key(waxterm::input::KeyPress::Escape);
                    return self.dispatch_chrome(out);
                }

                // Click on a menu item row.
                if row >= inner_y
                    && row < inner_y + MENU_ITEMS.len() as u16
                    && col >= inner_x
                    && col < inner_right
                {
                    let item_idx = (row - inner_y) as usize;
                    self.chrome.mode = ShellMode::Normal;
                    let out = self.chrome.execute_menu_action(item_idx);
                    return self.dispatch_chrome(out);
                }

                // Click outside popup — dismiss (including bottom gap below popup).
                if col >= popup_w || row < popup_y || row >= popup_y + popup_h {
                    let out = self.chrome.handle_key(waxterm::input::KeyPress::Escape);
                    return self.dispatch_chrome(out);
                }

                vec![]
            }

            // Console is rendered by shell_handlers with different geometry.
            // No click handling (keyboard Escape only).
            ShellMode::Console => vec![],

            // Modals: close button, list item clicks, click-outside-to-dismiss.
            ShellMode::ModelList
            | ShellMode::Settings
            | ShellMode::ConversationPicker { .. }
            | ShellMode::ServiceManager { .. }
            | ShellMode::WorkerManager { .. } => {
                let (pct_w, pct_h) = match modal_percentages(&self.chrome.mode) {
                    Some(p) => p,
                    None => return vec![],
                };
                let area = ratatui::layout::Rect::new(0, 0, self.cols, self.rows);
                let modal = centered_rect(pct_w, pct_h, area);

                // Close button on modal top border.
                if row == modal.y
                    && col >= modal.x + modal.width.saturating_sub(1 + CLOSE_BUTTON_WIDTH)
                    && col < modal.x + modal.width.saturating_sub(1)
                {
                    let out = self.chrome.handle_key(waxterm::input::KeyPress::Escape);
                    return self.dispatch_chrome(out);
                }

                // Click outside modal — dismiss.
                if col < modal.x || col >= modal.x + modal.width
                    || row < modal.y || row >= modal.y + modal.height
                {
                    let out = self.chrome.handle_key(waxterm::input::KeyPress::Escape);
                    return self.dispatch_chrome(out);
                }

                // Click on a list item inside the modal.
                let first_item_row = modal.y + 2; // border(1) + label(1)
                let inside_cols = col > modal.x && col < modal.x + modal.width - 1;

                if inside_cols && row >= first_item_row {
                    let idx = (row - first_item_row) as usize;
                    match self.chrome.mode {
                        ShellMode::ModelList => {
                            if idx < self.chrome.model_list.items_mut().len() {
                                self.chrome.model_list.set_selected(idx);
                                let out = self.chrome.handle_key(waxterm::input::KeyPress::Enter);
                                return self.dispatch_chrome(out);
                            }
                        }
                        ShellMode::Settings => {
                            // Clamp to visible list area (Settings has a constrained list height).
                            let inner_h = modal.height.saturating_sub(2);
                            let list_height = 5u16.min(inner_h / 2);
                            let max_visible = list_height.saturating_sub(1) as usize;
                            if idx < self.chrome.settings_list.items_mut().len() && idx < max_visible {
                                self.chrome.settings_list.set_selected(idx);
                                let out = self.chrome.handle_key(waxterm::input::KeyPress::Enter);
                                return self.dispatch_chrome(out);
                            }
                        }
                        ShellMode::ConversationPicker { ref mut list, .. } => {
                            if idx < list.items_mut().len() {
                                list.set_selected(idx);
                                let out = self.chrome.handle_key(waxterm::input::KeyPress::Enter);
                                return self.dispatch_chrome(out);
                            }
                        }
                        ShellMode::ServiceManager { ref mut selected } => {
                            // ServiceManager is informational — click selects only, no confirm.
                            if idx < self.chrome.service_list.len() {
                                *selected = idx;
                                return vec![CompositorOutput::Redraw];
                            }
                        }
                        ShellMode::WorkerManager { ref mut sandbox_sel, show_containers: false, .. } => {
                            if idx < self.chrome.worker_list.len() {
                                *sandbox_sel = idx;
                                let out = self.chrome.handle_key(waxterm::input::KeyPress::Enter);
                                return self.dispatch_chrome(out);
                            }
                        }
                        _ => {}
                    }
                }

                vec![]
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_compositor() -> Compositor {
        Compositor::new(120, 40, 1, 1, vec![], vec![])
    }

    #[test]
    fn mouse_click_start_menu_item() {
        let mut comp = make_compositor();
        comp.chrome.mode = ShellMode::StartMenu { selected: 0 };
        // Menu items start at inner_y = popup_y + 1.
        // popup_h = MENU_ITEMS.len() + 2; popup_y = 40 - popup_h - 1
        let popup_h = MENU_ITEMS.len() as u16 + 2;
        let popup_y = 40u16.saturating_sub(popup_h + 1);
        let first_item_row = popup_y + 1;
        // Click on item 3 (Models).
        let out = comp.handle(CompositorInput::MouseClick { col: 5, row: first_item_row + 3 });
        // Models action opens ModelList mode.
        assert!(matches!(comp.chrome.mode, ShellMode::ModelList));
        assert!(out.iter().any(|o| matches!(o, CompositorOutput::Redraw)));
    }

    #[test]
    fn mouse_click_outside_start_menu_dismisses() {
        let mut comp = make_compositor();
        comp.chrome.mode = ShellMode::StartMenu { selected: 0 };
        // Click far right (outside the 26-wide popup).
        let out = comp.handle(CompositorInput::MouseClick { col: 80, row: 20 });
        assert!(matches!(comp.chrome.mode, ShellMode::Normal));
        assert!(out.iter().any(|o| matches!(o, CompositorOutput::Redraw)));
    }

    #[test]
    fn mouse_click_outside_modal_dismisses() {
        let mut comp = make_compositor();
        comp.chrome.mode = ShellMode::ModelList;
        // Click at (0, 0) — outside the centered modal.
        let out = comp.handle(CompositorInput::MouseClick { col: 0, row: 0 });
        assert!(matches!(comp.chrome.mode, ShellMode::Normal));
        assert!(out.iter().any(|o| matches!(o, CompositorOutput::Redraw)));
    }

    #[test]
    fn mouse_click_model_list_item() {
        use crate::render::{centered_rect, modal_percentages};
        let models = vec![
            ModelEntry { model_ref: "a:main".into(), path: "/tmp".into(), loaded: true, loading: false },
            ModelEntry { model_ref: "b:main".into(), path: "/tmp".into(), loaded: true, loading: false },
        ];
        let mut comp = Compositor::new(120, 40, 1, 1, vec![], models);
        comp.chrome.mode = ShellMode::ModelList;
        let area = ratatui::layout::Rect::new(0, 0, 120, 40);
        let (pw, ph) = modal_percentages(&ShellMode::ModelList).unwrap();
        let modal = centered_rect(pw, ph, area);
        let first_item_row = modal.y + 2; // border + label
        let item_col = modal.x + 2;
        // Click on second item (index 1 = "b:main").
        let out = comp.handle(CompositorInput::MouseClick { col: item_col, row: first_item_row + 1 });
        assert!(out.iter().any(|o| matches!(o,
            CompositorOutput::Rpc(RpcRequest::ListConversations { ref model_ref })
                if model_ref == "b:main"
        )));
    }

    #[test]
    fn mouse_click_close_button_on_pane() {
        use crate::theme::CLOSE_BUTTON_WIDTH;
        let wins = vec![WindowSummary {
            id: 1, name: "w".into(), active_pane_id: 1,
            panes: vec![PaneSummary { id: 1, cols: 80, rows: 24, is_private: false }],
        }];
        let mut comp = Compositor::new(120, 40, 1, 1, wins, vec![]);
        // Close button at row 1, cols [width-1-3 .. width-2] = [116, 117, 118]
        let out = comp.handle(CompositorInput::MouseClick {
            col: 120 - 1 - CLOSE_BUTTON_WIDTH + 1, // 117
            row: 1,
        });
        assert!(out.iter().any(|o| matches!(o, CompositorOutput::Rpc(RpcRequest::CloseWindow { .. }))));
    }

    #[test]
    fn mouse_click_close_button_on_modal() {
        use crate::render::{centered_rect, modal_percentages};
        use crate::theme::CLOSE_BUTTON_WIDTH;
        let mut comp = make_compositor();
        comp.chrome.mode = ShellMode::ModelList;
        let area = ratatui::layout::Rect::new(0, 0, 120, 40);
        let (pw, ph) = modal_percentages(&ShellMode::ModelList).unwrap();
        let modal = centered_rect(pw, ph, area);
        let out = comp.handle(CompositorInput::MouseClick {
            col: modal.x + modal.width - 1 - CLOSE_BUTTON_WIDTH + 1,
            row: modal.y,
        });
        assert!(matches!(comp.chrome.mode, ShellMode::Normal));
        assert!(out.iter().any(|o| matches!(o, CompositorOutput::Redraw)));
    }

    #[test]
    fn mouse_click_bottom_gap_dismisses_start_menu() {
        let mut comp = make_compositor();
        comp.chrome.mode = ShellMode::StartMenu { selected: 0 };
        // Click on the window strip row directly below the popup.
        let popup_h = MENU_ITEMS.len() as u16 + 2;
        let popup_y = 40u16.saturating_sub(popup_h + 1);
        let out = comp.handle(CompositorInput::MouseClick { col: 5, row: popup_y + popup_h });
        assert!(matches!(comp.chrome.mode, ShellMode::Normal));
        assert!(out.iter().any(|o| matches!(o, CompositorOutput::Redraw)));
    }
}
