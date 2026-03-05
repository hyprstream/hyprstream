//! TuiService — ZMQ RPC service for the TUI multiplexer.
//!
//! Implements `ZmqService` for the TUI display server. Manages sessions,
//! windows, and panes via RPC. Frame publishing is handled separately
//! by the frame loop task (not in the service struct).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use capnp::message::Builder;
use capnp::serialize;
use ratatui::buffer::Cell;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{info, debug, warn};

use hyprstream_rpc::prelude::*;
use hyprstream_rpc::streaming::{
    BatchingConfig, StreamChannel, StreamContext, StreamPublisher, StreamPublisherConfig,
};
use hyprstream_rpc::{EnvelopeContext, ZmqService, Continuation};

use crate::tui_capnp;
use super::state::{TuiState, TuiEvent};
use super::diff;
use super::vte_parser;

// ============================================================================
// Display Mode
// ============================================================================

/// How a viewer wants frames encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    Ansi,
    Capnp,
    Structured,
}

// ============================================================================
// Viewer Management
// ============================================================================

/// A connected viewer's handle — tracks its publisher and health.
///
/// This is `!Send` because `StreamPublisher` contains a `tmq::push::Push` socket.
/// Viewer handles live on the frame loop's local task (spawned via `spawn_local`).
struct ViewerHandle {
    /// Viewer ID.
    id: u32,
    /// FD 1 (stdout): dedicated publisher for this viewer's frame stream.
    publisher: StreamPublisher,
    /// FD 0 (stdin): publisher for relaying viewer input back to the CLI process.
    /// Only populated when a session has a remote process that needs input relay.
    stdin_publisher: Option<StreamPublisher>,
    /// How this viewer wants frames encoded.
    display_mode: DisplayMode,
    /// Consecutive frames that failed to send (HWM full).
    consecutive_skips: u32,
    /// Cancel token for this viewer.
    cancel: CancellationToken,
}

// Viewer handles live on local tasks (not Send) — no type alias needed.

/// Deferred viewer registration — sent from the RPC handler (Send) to the
/// frame loop's local task (!Send) via a channel.
pub(crate) struct PendingViewer {
    id: u32,
    #[allow(dead_code)]
    session_id: u32,
    display_mode: DisplayMode,
    cancel: CancellationToken,
    /// FD-indexed stream contexts: [0]=stdin (input relay), [1]=stdout (frames).
    stream_ctxs: Vec<StreamContext>,
}

// We need PendingViewer to be Send so it can cross from the RPC thread
// to the frame loop task. StreamContext is Send.
// Safety: PendingViewer contains only Send types.

/// Commands sent from RPC handlers to the frame loop.
#[allow(dead_code)]
pub(crate) enum FrameLoopCommand {
    /// Register a new viewer.
    RegisterViewer(PendingViewer),
    /// Forward input bytes to the active pane.
    SendInput { viewer_id: u32, data: Vec<u8> },
    /// Evict a viewer by ID (disconnect).
    EvictViewer { viewer_id: u32 },
    /// Attach a process to a pane.
    SpawnProcess { pane_id: u32, process: super::process::PaneProcess },
}

/// Channel for sending commands to the frame loop.
pub(crate) type CommandSender = tokio::sync::mpsc::UnboundedSender<FrameLoopCommand>;
pub(crate) type CommandReceiver = tokio::sync::mpsc::UnboundedReceiver<FrameLoopCommand>;

/// Registry of active sessions → their command channels + cancel tokens.
/// This is Send+Sync because it only contains channel senders, not !Send sockets.
type SessionRegistry = Arc<RwLock<HashMap<u32, (CommandSender, CancellationToken)>>>;

// ============================================================================
// TuiService
// ============================================================================

/// The TUI multiplexer service.
///
/// Handles RPC requests for session/window/pane management.
/// All mutable state is behind `Arc<RwLock<TuiState>>`, making this Send+Sync.
/// Viewer/frame management is handled by separate local tasks (spawned via
/// `spawn_local` in the continuation), keeping !Send types off the service struct.
pub struct TuiService {
    /// Shared TUI state (sessions, windows, panes).
    state: Arc<RwLock<TuiState>>,
    /// Next viewer ID counter (atomic for Send+Sync).
    next_viewer_id: AtomicU32,
    /// ZMQ context (required by ZmqService).
    context: Arc<zmq::Context>,
    /// Transport config (required by ZmqService).
    transport: TransportConfig,
    /// Signing key (required by ZmqService).
    signing_key: SigningKey,
    /// Per-session viewer registration channels + frame loop tokens.
    sessions: SessionRegistry,
}

impl TuiService {
    /// Create a new TUI service.
    pub fn new(
        state: Arc<RwLock<TuiState>>,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
    ) -> Self {
        Self {
            state,
            next_viewer_id: AtomicU32::new(1),
            context,
            transport,
            signing_key,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Handle a connect request.
    ///
    /// Creates a DH-derived stream context and sends the viewer registration
    /// to the frame loop via a channel. The frame loop creates the !Send
    /// publisher socket on its local task.
    async fn handle_connect(
        &self,
        request_id: u64,
        session_id: u32,
        display_mode: DisplayMode,
        cols: u16,
        rows: u16,
        ctx: &EnvelopeContext,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        let mut state = self.state.write().await;

        // Create or find session
        let sid = if session_id == 0 {
            if state.sessions.is_empty() {
                state.create_session()
            } else {
                state.sessions.last().map(|s| s.id).unwrap_or(1)
            }
        } else {
            if state.session(session_id).is_none() {
                return Ok((self.build_error_response(request_id, "session not found", "NOT_FOUND")?, None));
            }
            session_id
        };

        if cols > 0 && rows > 0 {
            state.default_cols = cols;
            state.default_rows = rows;
            // Resize active pane to viewer's size (tmux: last-attached wins)
            if let Some(session) = state.session_mut(sid) {
                if let Some(window) = session.active_window_mut() {
                    if let Some(pane) = window.active_pane_mut() {
                        let (cur_cols, cur_rows) = pane.size();
                        if cur_cols != cols || cur_rows != rows {
                            pane.resize(cols, rows);
                        }
                    }
                }
            }
        }

        let viewer_id = self.next_viewer_id.fetch_add(1, Ordering::Relaxed);

        let session = match state.session(sid) {
            Some(s) => s,
            None => return Ok((self.build_error_response(request_id, "session not found", "NOT_FOUND")?, None)),
        };
        let windows: Vec<_> = session.windows.iter().map(|w| {
            (w.id, w.name.clone(), w.panes.iter().map(|p| (p.id, p.size())).collect::<Vec<_>>(), w.active_pane_id)
        }).collect();

        drop(state);

        // Create FD-indexed stream contexts: [0]=stdin (input relay), [1]=stdout (frames).
        // DH key exchange derives context from client's ephemeral pubkey.
        // If no pubkey (local/test connections), generate random standalone contexts.
        let make_stream_ctx = |label: &str| -> Result<StreamContext> {
            match ctx.ephemeral_pubkey() {
                Some(pubkey) => StreamContext::from_dh(pubkey),
                None => {
                    use rand::RngCore;
                    let mut rng = rand::thread_rng();
                    let mut topic_bytes = [0u8; 32];
                    let mut mac_key = [0u8; 32];
                    rng.fill_bytes(&mut topic_bytes);
                    rng.fill_bytes(&mut mac_key);
                    Ok(StreamContext::new(
                        format!("tui-{}-{}", label, viewer_id),
                        hex::encode(topic_bytes),
                        mac_key,
                        [0u8; 32],
                    ))
                }
            }
        };

        let stdin_ctx = make_stream_ctx("stdin")?;
        let stdout_ctx = make_stream_ctx("stdout")?;

        let sub_endpoint = hyprstream_rpc::registry::global()
            .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
            .to_zmq_string();

        let streams: [(&str, &str, &[u8; 32]); 2] = [
            (stdin_ctx.topic(), &sub_endpoint, stdin_ctx.mac_key()),
            (stdout_ctx.topic(), &sub_endpoint, stdout_ctx.mac_key()),
        ];

        let response = self.build_connect_response(
            request_id, viewer_id, sid, &windows, &streams,
        )?;

        let viewer_cancel = CancellationToken::new();
        let pending = PendingViewer {
            id: viewer_id,
            session_id: sid,
            display_mode,
            cancel: viewer_cancel,
            stream_ctxs: vec![stdin_ctx, stdout_ctx],
        };

        // Build continuation: send viewer registration to frame loop
        let sessions_reg = Arc::clone(&self.sessions);
        let tui_state = Arc::clone(&self.state);
        let zmq_context = Arc::clone(&self.context);
        let sk = self.signing_key.clone();
        let continuation: Continuation = Box::pin(async move {
            let mut reg = sessions_reg.write().await;
            let (sender, _cancel) = reg.entry(sid).or_insert_with(|| {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<FrameLoopCommand>();
                let cancel = CancellationToken::new();
                // Spawn frame loop for this session
                let s = Arc::clone(&tui_state);
                let c = cancel.clone();
                let ctx = Arc::clone(&zmq_context);
                tokio::task::spawn_local(run_frame_loop(s, sid, rx, ctx, sk, c));
                info!(session_id = sid, "Frame loop spawned");
                (tx, cancel)
            });
            if let Err(e) = sender.send(FrameLoopCommand::RegisterViewer(pending)) {
                warn!(viewer_id, "Failed to register viewer: {}", e);
            } else {
                debug!(viewer_id, session_id = sid, "Viewer registration sent");
            }
        });

        Ok((response, Some(continuation)))
    }

    async fn handle_create_window(&self, request_id: u64) -> Result<Vec<u8>> {
        let mut state = self.state.write().await;
        let session = state.sessions.last().map(|s| s.id);
        if let Some(sid) = session {
            if let Some(wid) = state.create_window(sid) {
                let window = match state.session(sid).and_then(|s| s.window(wid)) {
                    Some(w) => w,
                    None => return self.build_error_response(request_id, "window not found", "NOT_FOUND"),
                };
                let panes: Vec<_> = window.panes.iter().map(|p| (p.id, p.size(), p.title.clone())).collect();
                return self.build_window_response(request_id, window.id, &window.name, &panes);
            }
        }
        self.build_error_response(request_id, "no active session", "NO_SESSION")
    }

    async fn handle_snapshot(&self, request_id: u64, session_id: u32) -> Result<Vec<u8>> {
        let state = self.state.read().await;
        let sid = if session_id == 0 {
            state.sessions.last().map(|s| s.id).unwrap_or(0)
        } else {
            session_id
        };

        if let Some(session) = state.session(sid) {
            let generation = state.generation();
            let active_window_id = session.active_window_id;
            let windows: Vec<_> = session.windows.iter().map(|w| {
                let panes: Vec<_> = w.panes.iter().map(|p| (p.id, p.size(), p.title.clone())).collect();
                (w.id, w.name.clone(), panes, w.active_pane_id)
            }).collect();

            if let Some(window) = session.active_window() {
                if let Some(pane) = window.active_pane() {
                    let buf = pane.active_buffer();
                    let frame_diff = diff::compute_diff(buf, None, generation, pane.cursor);
                    return self.build_snapshot_response(
                        request_id, sid, generation, active_window_id,
                        &frame_diff, &windows,
                    );
                }
            }
        }

        self.build_error_response(request_id, "session not found", "NOT_FOUND")
    }

    async fn handle_list_windows(&self, request_id: u64, session_id: u32) -> Result<Vec<u8>> {
        let state = self.state.read().await;
        let sid = if session_id == 0 {
            state.sessions.last().map(|s| s.id).unwrap_or(0)
        } else {
            session_id
        };

        if let Some(session) = state.session(sid) {
            let windows: Vec<_> = session.windows.iter().map(|w| {
                let panes: Vec<_> = w.panes.iter().map(|p| {
                    (p.id, p.size(), p.title.clone())
                }).collect();
                (w.id, w.name.clone(), panes, w.active_pane_id)
            }).collect();

            let mut msg = Builder::new_default();
            {
                let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
                resp.set_request_id(request_id);
                let wl = resp.init_list_windows_result();
                let mut win_list = wl.init_windows(windows.len() as u32);
                for (i, (wid, name, panes, active_pane)) in windows.iter().enumerate() {
                    let mut w = win_list.reborrow().get(i as u32);
                    w.set_id(*wid);
                    w.set_name(name);
                    w.set_active_pane_id(*active_pane);
                    let mut pane_list = w.init_panes(panes.len() as u32);
                    for (j, (pid, (cols, rows), title)) in panes.iter().enumerate() {
                        let mut p = pane_list.reborrow().get(j as u32);
                        p.set_id(*pid);
                        p.set_cols(*cols);
                        p.set_rows(*rows);
                        p.set_title(title);
                    }
                }
            }
            let mut buf = Vec::new();
            serialize::write_message(&mut buf, &msg)?;
            Ok(buf)
        } else {
            self.build_error_response(request_id, "session not found", "NOT_FOUND")
        }
    }

    async fn handle_disconnect(&self, request_id: u64, viewer_id: u32) -> Result<Vec<u8>> {
        // Send eviction command to all session frame loops
        let sessions = self.sessions.read().await;
        for (sender, _) in sessions.values() {
            let _ = sender.send(FrameLoopCommand::EvictViewer { viewer_id });
        }
        drop(sessions);

        let mut msg = Builder::new_default();
        {
            let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
            resp.set_request_id(request_id);
            resp.set_disconnect_result(());
        }
        let mut buf = Vec::new();
        serialize::write_message(&mut buf, &msg)?;
        Ok(buf)
    }

    async fn handle_send_input(&self, request_id: u64, viewer_id: u32, data: &[u8]) -> Result<Vec<u8>> {
        // Send input to the frame loop for the viewer's session
        let sessions = self.sessions.read().await;
        for (sender, _) in sessions.values() {
            let _ = sender.send(FrameLoopCommand::SendInput {
                viewer_id,
                data: data.to_vec(),
            });
        }
        drop(sessions);

        let mut msg = Builder::new_default();
        {
            let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
            resp.set_request_id(request_id);
            resp.set_send_input_result(());
        }
        let mut buf = Vec::new();
        serialize::write_message(&mut buf, &msg)?;
        Ok(buf)
    }

    async fn handle_close_window(&self, request_id: u64, window_id: u32) -> Result<Vec<u8>> {
        let mut state = self.state.write().await;
        let sid = state.sessions.last().map(|s| s.id).unwrap_or(0);
        if state.close_window(sid, window_id) {
            let mut msg = Builder::new_default();
            {
                let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
                resp.set_request_id(request_id);
                resp.set_close_window_result(());
            }
            let mut buf = Vec::new();
            serialize::write_message(&mut buf, &msg)?;
            Ok(buf)
        } else {
            self.build_error_response(request_id, "window not found", "NOT_FOUND")
        }
    }

    async fn handle_focus_window(&self, request_id: u64, window_id: u32) -> Result<Vec<u8>> {
        let mut state = self.state.write().await;
        let sid = state.sessions.last().map(|s| s.id).unwrap_or(0);
        if state.focus_window(sid, window_id) {
            let mut msg = Builder::new_default();
            {
                let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
                resp.set_request_id(request_id);
                resp.set_focus_window_result(());
            }
            let mut buf = Vec::new();
            serialize::write_message(&mut buf, &msg)?;
            Ok(buf)
        } else {
            self.build_error_response(request_id, "window not found", "NOT_FOUND")
        }
    }

    async fn handle_split_pane(&self, request_id: u64, horizontal: bool, ratio: f32) -> Result<Vec<u8>> {
        let mut state = self.state.write().await;
        let sid = state.sessions.last().map(|s| s.id).unwrap_or(0);
        let ratio = if ratio <= 0.0 || ratio >= 1.0 { 0.5 } else { ratio };

        if let Some((pane_id, cols, rows)) = state.split_pane(sid, horizontal, ratio) {
            let mut msg = Builder::new_default();
            {
                let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
                resp.set_request_id(request_id);
                let mut pane_info = resp.init_split_pane_result();
                pane_info.set_id(pane_id);
                pane_info.set_cols(cols);
                pane_info.set_rows(rows);
                pane_info.set_title("");
            }
            let mut buf = Vec::new();
            serialize::write_message(&mut buf, &msg)?;
            Ok(buf)
        } else {
            self.build_error_response(request_id, "no active session or window", "NO_SESSION")
        }
    }

    async fn handle_close_pane(&self, request_id: u64, pane_id: u32) -> Result<Vec<u8>> {
        let mut state = self.state.write().await;
        let sid = state.sessions.last().map(|s| s.id).unwrap_or(0);
        if state.close_pane(sid, pane_id) {
            let mut msg = Builder::new_default();
            {
                let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
                resp.set_request_id(request_id);
                resp.set_close_pane_result(());
            }
            let mut buf = Vec::new();
            serialize::write_message(&mut buf, &msg)?;
            Ok(buf)
        } else {
            self.build_error_response(request_id, "pane not found or is the last pane", "NOT_FOUND")
        }
    }

    async fn handle_focus_pane(&self, request_id: u64, pane_id: u32) -> Result<Vec<u8>> {
        let mut state = self.state.write().await;
        let sid = state.sessions.last().map(|s| s.id).unwrap_or(0);
        if state.focus_pane(sid, pane_id) {
            let mut msg = Builder::new_default();
            {
                let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
                resp.set_request_id(request_id);
                resp.set_focus_pane_result(());
            }
            let mut buf = Vec::new();
            serialize::write_message(&mut buf, &msg)?;
            Ok(buf)
        } else {
            self.build_error_response(request_id, "pane not found", "NOT_FOUND")
        }
    }

    async fn handle_resize(&self, request_id: u64, cols: u16, rows: u16) -> Result<Vec<u8>> {
        // Resize the active pane in the most recent session
        let mut state = self.state.write().await;
        if let Some(session) = state.sessions.last_mut() {
            if let Some(window) = session.active_window_mut() {
                if let Some(pane) = window.active_pane_mut() {
                    pane.resize(cols, rows);
                }
            }
        }
        let mut msg = Builder::new_default();
        {
            let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
            resp.set_request_id(request_id);
            resp.set_resize_result(());
        }
        let mut buf = Vec::new();
        serialize::write_message(&mut buf, &msg)?;
        Ok(buf)
    }

    // ========================================================================
    // Response builders
    // ========================================================================

    fn build_error_response(&self, request_id: u64, message: &str, code: &str) -> Result<Vec<u8>> {
        let mut msg = Builder::new_default();
        {
            let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
            resp.set_request_id(request_id);
            let mut err = resp.init_error();
            err.set_message(message);
            err.set_code(code);
        }
        let mut buf = Vec::new();
        serialize::write_message(&mut buf, &msg)?;
        Ok(buf)
    }

    fn build_connect_response(
        &self,
        request_id: u64,
        viewer_id: u32,
        session_id: u32,
        windows: &[(u32, String, Vec<(u32, (u16, u16))>, u32)],
        streams: &[(&str, &str, &[u8; 32])], // (topic, sub_endpoint, mac_key)
    ) -> Result<Vec<u8>> {
        let mut msg = Builder::new_default();
        {
            let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
            resp.set_request_id(request_id);
            let mut connect = resp.init_connect_result();
            connect.set_viewer_id(viewer_id);
            connect.set_session_id(session_id);

            // FD-indexed streams: [0]=stdin (input relay), [1]=stdout (frames)
            let mut stream_list = connect.reborrow().init_streams(streams.len() as u32);
            for (i, (topic, sub_endpoint, mac_key)) in streams.iter().enumerate() {
                let mut si = stream_list.reborrow().get(i as u32);
                si.set_topic(topic);
                si.set_sub_endpoint(sub_endpoint);
                si.set_mac_key(*mac_key);
            }

            let mut win_list = connect.init_windows(windows.len() as u32);
            for (i, (wid, name, panes, active_pane)) in windows.iter().enumerate() {
                let mut w = win_list.reborrow().get(i as u32);
                w.set_id(*wid);
                w.set_name(name);
                w.set_active_pane_id(*active_pane);
                let mut pane_list = w.init_panes(panes.len() as u32);
                for (j, (pid, (cols, rows))) in panes.iter().enumerate() {
                    let mut p = pane_list.reborrow().get(j as u32);
                    p.set_id(*pid);
                    p.set_cols(*cols);
                    p.set_rows(*rows);
                    p.set_title("");
                }
            }
        }
        let mut buf = Vec::new();
        serialize::write_message(&mut buf, &msg)?;
        Ok(buf)
    }

    fn build_window_response(
        &self,
        request_id: u64,
        window_id: u32,
        name: &str,
        panes: &[(u32, (u16, u16), String)],
    ) -> Result<Vec<u8>> {
        let mut msg = Builder::new_default();
        {
            let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
            resp.set_request_id(request_id);
            let mut win = resp.init_create_window_result();
            win.set_id(window_id);
            win.set_name(name);
            win.set_active_pane_id(panes.first().map(|(id, _, _)| *id).unwrap_or(0));
            let mut pane_list = win.init_panes(panes.len() as u32);
            for (i, (pid, (cols, rows), title)) in panes.iter().enumerate() {
                let mut p = pane_list.reborrow().get(i as u32);
                p.set_id(*pid);
                p.set_cols(*cols);
                p.set_rows(*rows);
                p.set_title(title);
            }
        }
        let mut buf = Vec::new();
        serialize::write_message(&mut buf, &msg)?;
        Ok(buf)
    }

    fn build_snapshot_response(
        &self,
        request_id: u64,
        session_id: u32,
        generation: u64,
        active_window_id: u32,
        frame_diff: &diff::FrameDiff,
        windows: &[(u32, String, Vec<(u32, (u16, u16), String)>, u32)],
    ) -> Result<Vec<u8>> {
        let mut msg = Builder::new_default();
        {
            let mut resp = msg.init_root::<tui_capnp::tui_response::Builder<'_>>();
            resp.set_request_id(request_id);
            let mut snap = resp.init_snapshot_result();
            snap.set_session_id(session_id);
            snap.set_generation(generation);
            snap.set_active_window_id(active_window_id);

            // Populate full frame from frame_diff
            let mut frame = snap.reborrow().init_frame();
            frame.set_cols(frame_diff.cols);
            frame.set_rows(frame_diff.rows);
            let mut cells = frame.init_cells(frame_diff.deltas.len() as u32);
            for (i, delta) in frame_diff.deltas.iter().enumerate() {
                let mut cell = cells.reborrow().get(i as u32);
                cell.set_x(delta.x);
                cell.set_y(delta.y);
                cell.set_symbol(&delta.symbol);
                cell.set_fg(diff::pack_color(delta.fg));
                cell.set_bg(diff::pack_color(delta.bg));
                cell.set_modifiers(delta.modifiers.bits());
            }

            // Populate windows list
            let mut win_list = snap.init_windows(windows.len() as u32);
            for (i, (wid, name, panes, active_pane)) in windows.iter().enumerate() {
                let mut w = win_list.reborrow().get(i as u32);
                w.set_id(*wid);
                w.set_name(name);
                w.set_active_pane_id(*active_pane);
                let mut pane_list = w.init_panes(panes.len() as u32);
                for (j, (pid, (cols, rows), title)) in panes.iter().enumerate() {
                    let mut p = pane_list.reborrow().get(j as u32);
                    p.set_id(*pid);
                    p.set_cols(*cols);
                    p.set_rows(*rows);
                    p.set_title(title);
                }
            }
        }
        let mut buf = Vec::new();
        serialize::write_message(&mut buf, &msg)?;
        Ok(buf)
    }
}

#[async_trait(?Send)]
impl ZmqService for TuiService {
    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        let reader = serialize::read_message_from_flat_slice(
            &mut &payload[..],
            capnp::message::ReaderOptions::default(),
        )?;
        let request = reader.get_root::<tui_capnp::tui_request::Reader<'_>>()?;
        let request_id = request.get_id();

        match request.which()? {
            tui_capnp::tui_request::Which::Connect(connect_reader) => {
                let connect = connect_reader?;
                let sid = connect.get_session_id();
                let cols = connect.get_cols();
                let rows = connect.get_rows();
                let display_mode = match connect.get_display_mode()? {
                    tui_capnp::DisplayMode::Capnp => DisplayMode::Capnp,
                    tui_capnp::DisplayMode::Structured => DisplayMode::Structured,
                    _ => DisplayMode::Ansi,
                };
                self.handle_connect(request_id, sid, display_mode, cols, rows, _ctx).await
            }
            tui_capnp::tui_request::Which::CreateWindow(()) => {
                Ok((self.handle_create_window(request_id).await?, None))
            }
            tui_capnp::tui_request::Which::Snapshot(sid) => {
                Ok((self.handle_snapshot(request_id, sid).await?, None))
            }
            tui_capnp::tui_request::Which::ListWindows(session_id) => {
                Ok((self.handle_list_windows(request_id, session_id).await?, None))
            }
            tui_capnp::tui_request::Which::Disconnect(viewer_id) => {
                Ok((self.handle_disconnect(request_id, viewer_id).await?, None))
            }
            tui_capnp::tui_request::Which::Resize(resize_reader) => {
                let resize = resize_reader?;
                let cols = resize.get_cols();
                let rows = resize.get_rows();
                Ok((self.handle_resize(request_id, cols, rows).await?, None))
            }
            tui_capnp::tui_request::Which::CloseWindow(window_id) => {
                Ok((self.handle_close_window(request_id, window_id).await?, None))
            }
            tui_capnp::tui_request::Which::FocusWindow(window_id) => {
                Ok((self.handle_focus_window(request_id, window_id).await?, None))
            }
            tui_capnp::tui_request::Which::SplitPane(split_reader) => {
                let split = split_reader?;
                let horizontal = match split.get_direction()? {
                    tui_capnp::SplitDirection::Horizontal => true,
                    tui_capnp::SplitDirection::Vertical => false,
                };
                let ratio = split.get_ratio();
                Ok((self.handle_split_pane(request_id, horizontal, ratio).await?, None))
            }
            tui_capnp::tui_request::Which::ClosePane(pane_id) => {
                Ok((self.handle_close_pane(request_id, pane_id).await?, None))
            }
            tui_capnp::tui_request::Which::FocusPane(pane_id) => {
                Ok((self.handle_focus_pane(request_id, pane_id).await?, None))
            }
            tui_capnp::tui_request::Which::SendInput(input_reader) => {
                let input = input_reader?;
                let viewer_id = input.get_viewer_id();
                let data = input.get_data()?;
                Ok((self.handle_send_input(request_id, viewer_id, data).await?, None))
            }
        }
    }

    fn name(&self) -> &str {
        "tui"
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        None
    }

    fn verify_claims(&self, _ctx: &EnvelopeContext) -> Result<()> {
        Ok(())
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        self.build_error_response(request_id, error, "INTERNAL")
            .unwrap_or_default()
    }
}

// ============================================================================
// Frame Loop
// ============================================================================

/// Maximum consecutive skips before a viewer is evicted (~1s at 30fps).
const MAX_CONSECUTIVE_SKIPS: u32 = 30;

/// Spawn a frame loop for a session.
///
/// The frame loop runs at ~30fps, encodes diffs based on damage notifications,
/// and publishes to all connected viewers. Slow viewers (>30 consecutive skips)
/// are evicted.
///
/// Runs on a local task (`spawn_local`) because `StreamPublisher` is `!Send`.
/// New viewers are registered via `viewer_rx` channel from the RPC handler.
pub(crate) async fn run_frame_loop(
    state: Arc<RwLock<TuiState>>,
    session_id: u32,
    mut cmd_rx: CommandReceiver,
    zmq_context: Arc<zmq::Context>,
    signing_key: SigningKey,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(33));
    let mut event_rx = {
        let s = state.read().await;
        s.subscribe()
    };

    // Viewer list lives here (local, !Send is OK)
    let mut viewers: Vec<ViewerHandle> = Vec::new();
    // Hosted processes attached to panes (pane_id → PaneProcess)
    let mut processes: HashMap<u32, super::process::PaneProcess> = HashMap::new();
    // Previous frame snapshot for incremental diffs (per-pane, keyed by pane_id)
    let mut prev_cells: HashMap<u32, Vec<Cell>> = HashMap::new();
    let mut has_damage = false;
    // Delayed initial frame: after a viewer registers, wait for the ZMQ SUB to connect
    // before publishing the first frame (avoids "slow joiner" race condition).
    let mut initial_frame_at: Option<tokio::time::Instant> = None;

    // StreamChannel for creating dedicated publisher sockets
    let stream_channel = StreamChannel::new(zmq_context, signing_key);

    info!(session_id, "Frame loop started");

    loop {
        tokio::select! {
            biased;

            _ = cancel.cancelled() => {
                info!(session_id, "Frame loop cancelled");
                break;
            }

            // Accept commands from RPC handlers
            Some(cmd) = cmd_rx.recv() => {
                match cmd {
                    FrameLoopCommand::RegisterViewer(pending) => {
                        // Pre-authorize stream topics with the StreamService proxy.
                        // Without this, the proxy rejects subscriptions.
                        let publisher_config = StreamPublisherConfig { sndhwm: 100, dedicated: true };
                        let batching = BatchingConfig {
                            min_batch_size: 1,
                            max_batch_size: 1,
                            max_block_bytes: 256 * 1024,
                            min_rate: 1.0,
                            max_rate: 30.0,
                        };
                        let exp = chrono::Utc::now().timestamp() + 86400; // 24h expiry

                        // FD 0 (stdin): input relay publisher — only for Capnp-mode viewers
                        // (i.e. the CLI cast player). Ansi-mode viewers don't need it;
                        // their keyboard input falls through to VTE when no producer exists.
                        let stdin_publisher = if pending.display_mode == DisplayMode::Capnp {
                            if let Some(stdin_ctx) = pending.stream_ctxs.get(0) {
                                let topic = stdin_ctx.topic().to_owned();
                                if let Err(e) = stream_channel.register_topic(&topic, exp, None).await {
                                    warn!(viewer_id = pending.id, error = %e, "Failed to register stdin topic");
                                }
                                match stream_channel.create_publisher_socket(&publisher_config) {
                                    Ok(socket) => Some(StreamPublisher::with_dedicated_socket(
                                        socket, stdin_ctx, batching.clone(),
                                    )),
                                    Err(e) => {
                                        warn!(viewer_id = pending.id, error = %e, "Failed to create stdin publisher");
                                        None
                                    }
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        // FD 1 (stdout): frame output publisher
                        let stdout_ctx = match pending.stream_ctxs.get(1) {
                            Some(ctx) => ctx,
                            None => {
                                warn!(viewer_id = pending.id, "Missing stdout stream context");
                                continue;
                            }
                        };
                        let stdout_topic = stdout_ctx.topic().to_owned();
                        if let Err(e) = stream_channel.register_topic(&stdout_topic, exp, None).await {
                            warn!(viewer_id = pending.id, error = %e, "Failed to register stdout topic");
                        } else {
                            debug!(viewer_id = pending.id, topic = %stdout_topic, "Stream topic registered with proxy");
                        }

                        match stream_channel.create_publisher_socket(&publisher_config) {
                            Ok(socket) => {
                                let publisher = StreamPublisher::with_dedicated_socket(
                                    socket, stdout_ctx, batching,
                                );
                                viewers.push(ViewerHandle {
                                    id: pending.id,
                                    publisher,
                                    stdin_publisher,
                                    display_mode: pending.display_mode,
                                    consecutive_skips: 0,
                                    cancel: pending.cancel,
                                });
                                // Schedule initial frame after a short delay to allow the
                                // WebTransport subscription bridge (ZMQ SUB) to connect.
                                initial_frame_at = Some(tokio::time::Instant::now()
                                    + std::time::Duration::from_millis(500));
                                debug!(viewer_id = pending.id, session_id, "Viewer registered in frame loop");
                            }
                            Err(e) => {
                                warn!(viewer_id = pending.id, error = %e, "Failed to create stdout publisher");
                            }
                        }
                    }
                    FrameLoopCommand::SendInput { viewer_id, data } => {
                        // Find the active pane ID to check for an attached process
                        let active_pane_id = {
                            let s = state.read().await;
                            s.session(session_id)
                                .and_then(|sess| sess.active_window())
                                .and_then(|win| win.active_pane())
                                .map(|p| p.id)
                        };

                        if let Some(pane_id) = active_pane_id {
                            if let Some(proc) = processes.get(&pane_id) {
                                // Route input to the local attached process
                                let _ = proc.input_tx.try_send(
                                    super::process::ProcessInput::Stdin(data),
                                );
                                continue;
                            }
                        }

                        // Determine if the sender is a producer (Capnp-mode viewer
                        // sending ANSI output, e.g. cast player) or a consumer
                        // (Ansi-mode viewer sending keyboard input, e.g. tui attach).
                        let sender_is_producer = viewers.iter()
                            .any(|v| v.id == viewer_id && v.display_mode == DisplayMode::Capnp);

                        if sender_is_producer {
                            // Producer output → VTE render into pane
                            let mut tui_state = state.write().await;
                            let damage_info = if let Some(session) = tui_state.session_mut(session_id) {
                                if let Some(window) = session.active_window_mut() {
                                    let window_id = window.id;
                                    if let Some(pane) = window.active_pane_mut() {
                                        let pane_id = pane.id;
                                        let mut performer = vte_parser::PanePerformer::new(pane);
                                        performer.feed(&data);
                                        Some((window_id, pane_id))
                                    } else { None }
                                } else { None }
                            } else { None };
                            drop(tui_state);

                            if let Some((window_id, pane_id)) = damage_info {
                                has_damage = true;
                                let s = state.read().await;
                                s.notify_damage(session_id, window_id, pane_id);
                            }
                        } else {
                            // Consumer input → relay via stdin stream (FD 0)
                            // to any remote process listening (e.g. CLI cast player).
                            let mut relayed = false;
                            for viewer in &mut viewers {
                                if let Some(ref mut stdin_pub) = viewer.stdin_publisher {
                                    if stdin_pub.publish_data(&data).await.is_ok() {
                                        relayed = true;
                                    }
                                }
                            }

                            if !relayed {
                                // No stdin subscribers — VTE fallback (direct write to pane)
                                let mut tui_state = state.write().await;
                                let damage_info = if let Some(session) = tui_state.session_mut(session_id) {
                                    if let Some(window) = session.active_window_mut() {
                                        let window_id = window.id;
                                        if let Some(pane) = window.active_pane_mut() {
                                            let pane_id = pane.id;
                                            let mut performer = vte_parser::PanePerformer::new(pane);
                                            performer.feed(&data);
                                            Some((window_id, pane_id))
                                        } else { None }
                                    } else { None }
                                } else { None };
                                drop(tui_state);

                                if let Some((window_id, pane_id)) = damage_info {
                                    // Set has_damage directly so the next interval tick
                                    // publishes the frame even if event_rx races ahead.
                                    has_damage = true;
                                    let s = state.read().await;
                                    s.notify_damage(session_id, window_id, pane_id);
                                }
                            }
                        }
                    }
                    FrameLoopCommand::EvictViewer { viewer_id } => {
                        if let Some(pos) = viewers.iter().position(|v| v.id == viewer_id) {
                            let v = viewers.remove(pos);
                            v.cancel.cancel();
                            debug!(viewer_id, session_id, "Viewer evicted by command");
                        }
                    }
                    FrameLoopCommand::SpawnProcess { pane_id, process } => {
                        debug!(pane_id, session_id, "Process attached to pane");
                        processes.insert(pane_id, process);
                    }
                }
            }

            Ok(event) = event_rx.recv() => {
                if let TuiEvent::Damage { session_id: sid, .. } = event {
                    if sid == session_id {
                        has_damage = true;
                    }
                }
            }

            _ = interval.tick() => {
                // Poll attached processes for output and feed through VTE parser
                {
                    let process_pane_ids: Vec<u32> = processes.keys().copied().collect();
                    for pane_id in process_pane_ids {
                        let proc = match processes.get_mut(&pane_id) {
                            Some(p) => p,
                            None => continue,
                        };
                        let mut got_output = false;
                        while let Ok(data) = proc.stdout_rx.try_recv() {
                            let mut tui_state = state.write().await;
                            if let Some(session) = tui_state.session_mut(session_id) {
                                // Search all windows for the pane
                                for window in &mut session.windows {
                                    if let Some(pane) = window.pane_mut(pane_id) {
                                        let mut performer = vte_parser::PanePerformer::new(pane);
                                        performer.feed(&data);
                                        got_output = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if got_output {
                            has_damage = true;
                        }
                    }
                }

                // Check if delayed initial frame timer has fired
                if let Some(t) = initial_frame_at {
                    if tokio::time::Instant::now() >= t {
                        has_damage = true;
                        initial_frame_at = None;
                        debug!(session_id, "Initial frame timer fired, sending first frame");
                    }
                }
                if !has_damage || viewers.is_empty() {
                    continue;
                }
                has_damage = false;

                // Read state and compute diff
                let mut tui_state = state.write().await;
                let generation = tui_state.generation();

                let (ansi_bytes, capnp_bytes) = {
                    let session = match tui_state.session(session_id) {
                        Some(s) => s,
                        None => continue,
                    };
                    let window = match session.active_window() {
                        Some(w) => w,
                        None => continue,
                    };
                    let pane = match window.active_pane() {
                        Some(p) => p,
                        None => continue,
                    };
                    let pane_id = pane.id;
                    let buf = pane.active_buffer();
                    let prev = prev_cells.get(&pane_id).map(|v| v.as_slice());
                    let frame_diff = diff::compute_diff(buf, prev, generation, pane.cursor);

                    // Encode for each display mode in use
                    let needs_ansi = viewers.iter().any(|v| v.display_mode == DisplayMode::Ansi);
                    let needs_capnp = viewers.iter().any(|v| v.display_mode == DisplayMode::Capnp);

                    let ansi = if needs_ansi { Some(diff::encode_ansi(&frame_diff)) } else { None };
                    let capnp = if needs_capnp { Some(diff::encode_capnp(&frame_diff)) } else { None };

                    // Snapshot cells for next incremental diff
                    let area = buf.buffer.area();
                    let cells: Vec<Cell> = (0..area.height)
                        .flat_map(|y| (0..area.width).map(move |x| (x, y)))
                        .map(|(x, y)| buf.buffer[(x, y)].clone())
                        .collect();
                    prev_cells.insert(pane_id, cells);

                    (ansi, capnp)
                };

                // Clear damage on the active pane
                if let Some(session) = tui_state.session_mut(session_id) {
                    if let Some(window) = session.active_window_mut() {
                        if let Some(pane) = window.active_pane_mut() {
                            pane.active_buffer_mut().clear_damage();
                        }
                    }
                }

                drop(tui_state);

                // Publish to each viewer
                let mut evict_ids = Vec::new();
                for viewer in viewers.iter_mut() {
                    let data = match viewer.display_mode {
                        DisplayMode::Ansi => ansi_bytes.as_deref(),
                        DisplayMode::Capnp => capnp_bytes.as_deref(),
                        DisplayMode::Structured => capnp_bytes.as_deref(), // fallback to capnp encoding
                    };
                    let data = match data {
                        Some(d) if !d.is_empty() => d,
                        _ => {
                            debug!(viewer_id = viewer.id, mode = ?viewer.display_mode, "No data for viewer (empty or None)");
                            continue;
                        }
                    };

                    debug!(viewer_id = viewer.id, data_len = data.len(), "Publishing frame to viewer");
                    let publish_result = tokio::time::timeout(
                        std::time::Duration::from_millis(100),
                        viewer.publisher.try_publish_data(data, 30.0),
                    ).await;
                    match publish_result {
                        Ok(Ok(true)) => {
                            viewer.consecutive_skips = 0;
                            debug!(viewer_id = viewer.id, "Frame published successfully");
                        }
                        Ok(Ok(false)) => {
                            viewer.consecutive_skips += 1;
                            if viewer.consecutive_skips >= MAX_CONSECUTIVE_SKIPS {
                                warn!(viewer_id = viewer.id, "Evicting slow viewer");
                                evict_ids.push(viewer.id);
                            }
                        }
                        Ok(Err(e)) => {
                            debug!(viewer_id = viewer.id, error = %e, "Viewer publish error");
                            evict_ids.push(viewer.id);
                        }
                        Err(_) => {
                            warn!(viewer_id = viewer.id, "Viewer publish timed out (100ms)");
                            viewer.consecutive_skips += 1;
                            if viewer.consecutive_skips >= MAX_CONSECUTIVE_SKIPS {
                                evict_ids.push(viewer.id);
                            }
                        }
                    }
                }

                // Evict slow/errored viewers
                for id in &evict_ids {
                    if let Some(pos) = viewers.iter().position(|v| v.id == *id) {
                        let v = viewers.remove(pos);
                        v.cancel.cancel();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_mode() {
        assert_ne!(DisplayMode::Ansi, DisplayMode::Capnp);
        assert_ne!(DisplayMode::Capnp, DisplayMode::Structured);
    }
}
