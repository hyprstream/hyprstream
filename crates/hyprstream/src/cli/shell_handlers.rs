//! Shell client handlers.
//!
//! Two modes:
//!
//! - `handle_shell_tui` — interactive local mode (`hyprstream` no args).
//!   Connects to TuiService, renders ratatui chrome via `AnsiBackend` to the
//!   local terminal, and routes input through the `Compositor` state machine.
//!
//! - `handle_tui_shell` — embedded mode (`hyprstream tui shell`).
//!   Sends `SpawnChromeShell` RPC to TuiService so ShellApp runs server-side
//!   as a `PaneProcess`.  No fork occurs in the client, avoiding the ZMQ
//!   signaler assertion that fired when forking after ZMQ initialisation.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashMap;

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use waxterm::backend::AnsiBackend;
use waxterm::input::InputParser;
use zeroize::Zeroizing;

use hyprstream_compositor::{
    Compositor, CompositorInput, CompositorOutput, ConversationPickerEntry, ModelEntry,
    PaneSummary, RpcRequest, ServiceEntry, ServiceMode, ToastLevel, WindowSummary,
};
use hyprstream_compositor::layout::{CellUpdate, CursorState, FrameContent, FrameUpdate, ScrollUpdate};
use hyprstream_tui::chat_app::{ChatApp, LoadHook, SaveHook};
use hyprstream_tui::console_app::ConsoleApp;
use hyprstream_tui::private_store::{FsBackend, StorageKey};
use waxterm::app::TerminalApp;

use crate::services::generated::tui_client::TuiClient;
use crate::tui::shell_client::{
    close_window_rpc, focus_window_rpc, spawn_chat_app_rpc,
    spawn_shell_rpc,
};

// ============================================================================
// handle_shell_tui — local interactive mode
// ============================================================================

/// Run ShellClient as a local interactive terminal using the compositor.
pub async fn handle_shell_tui(
    signing_key: &SigningKey,
    models_dir: &std::path::Path,
) -> Result<()> {
    use crate::cli::tui_handlers::{
        create_tui_client, enter_raw_mode, restore_terminal, terminal_size,
    };
    use hyprstream_rpc::streaming::{StreamPayload, StreamVerifier};

    let client = create_tui_client(signing_key);
    let (cols, rows) = terminal_size();
    let pane_rows = rows.saturating_sub(5); // status(1) + top border(1) + bottom border(1) + strip(1) + fkeys(1)
    let pane_cols = cols.saturating_sub(2); // left + right border

    // Connect to session 0 using Capnp display mode so the frame stream
    // carries structured TuiFrame messages with pane_id for correct routing.
    let result = client
        .connect(0, "capnp", pane_cols, pane_rows)
        .await
        .context("Failed to connect to TUI session. Is the TUI service running?")?;

    let session_id = result.session_id;
    let viewer_id  = result.viewer_id;

    // Auto-spawn shell in the first pane if there is one.
    if let Some(win) = result.windows.first() {
        if let Some(pane) = win.panes.first() {
            let _ = spawn_shell_rpc(&client, session_id, pane.id, "").await;
        }
    }

    // Build initial window list.
    let windows: Vec<WindowSummary> = result
        .windows
        .iter()
        .map(|w| WindowSummary {
            id: w.id,
            name: w.name.clone(),
            active_pane_id: w.active_pane_id,
            panes: w.panes.iter().map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows, is_private: p.is_private }).collect(),
        })
        .collect();

    // Fetch model list from registry.
    let models = fetch_models(signing_key, models_dir).await;

    // Model load-status channel (background polling → event loop).
    let model_client = {
        use hyprstream_rpc::envelope::RequestIdentity;
        crate::services::ModelZmqClient::new(signing_key.clone(), RequestIdentity::local())
    };
    let (model_status_tx, mut model_status_rx) =
        tokio::sync::mpsc::channel::<(String, bool)>(32);

    // Terminal resize channel — SIGWINCH handler → main event loop.
    let (resize_tx, mut resize_rx) = tokio::sync::mpsc::channel::<(u16, u16)>(4);

    // rows tracks the current terminal height and is updated on SIGWINCH.
    let mut rows = rows;

    // Build compositor.
    let mut compositor = Compositor::new(cols, rows, session_id, viewer_id, windows, models);

    // Subscribe to ANSI frame stream.
    let stdout_stream = result.streams.get(1)
        .context("TuiService did not return a frame stream")?;
    let mac_key: [u8; 32] = stdout_stream.mac_key.as_slice().try_into()
        .context("mac_key must be 32 bytes")?;
    let topic = stdout_stream.topic.clone();

    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    let sub_endpoint = hyprstream_rpc::registry::global()
        .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
        .to_zmq_string();
    let zmq_ctx = zmq::Context::new();
    let sub_socket = zmq_ctx.socket(zmq::SUB)?;
    sub_socket.connect(&sub_endpoint)?;
    sub_socket.set_subscribe(topic.as_bytes())?;
    sub_socket.set_linger(0)?;

    let _recv_handle = tokio::task::spawn_blocking(move || {
        let mut verifier = StreamVerifier::new(mac_key, topic);
        loop {
            let mut frames: Vec<Vec<u8>> = Vec::with_capacity(3);
            match sub_socket.recv_bytes(0) {
                Ok(f) => frames.push(f),
                Err(_) => break,
            }
            while sub_socket.get_rcvmore().unwrap_or(false) {
                match sub_socket.recv_bytes(0) {
                    Ok(f) => frames.push(f),
                    Err(_) => break,
                }
            }
            match verifier.verify(&frames) {
                Ok(payloads) => {
                    for p in payloads {
                        if let StreamPayload::Data(data) = p {
                            if frame_tx.blocking_send(data).is_err() { return; }
                        }
                    }
                }
                Err(e) => tracing::warn!("Frame stream verification failed: {e}"),
            }
        }
    });

    // Stdin reader thread.
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    std::thread::spawn(move || {
        use std::io::Read;
        let mut buf = [0u8; 256];
        loop {
            match std::io::stdin().lock().read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if stdin_tx.blocking_send(buf[..n].to_vec()).is_err() { break; }
                }
            }
        }
    });

    // Set up ratatui terminal.
    let writer = AnsiWriter(std::io::stdout());
    let backend = AnsiBackend::new(writer, cols, rows, false);
    let mut terminal = ratatui::Terminal::new(backend)
        .context("Failed to create ratatui terminal")?;
    let _ = terminal.hide_cursor();
    let _ = terminal.clear();

    let input_parser = InputParser::new(vec![]);

    // Redirect tracing to a log file so raw-mode rendering is not corrupted.
    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("hyprstream");
    std::fs::create_dir_all(&log_dir).ok();
    let file_appender = tracing_appender::rolling::never(&log_dir, "hyprstream-tui.log");
    let (non_blocking, _log_guard) = tracing_appender::non_blocking(file_appender);
    #[allow(unused_imports)]
    use tracing_subscriber::util::SubscriberInitExt as _;
    let _subscriber_guard = tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_ansi(false)
        .set_default();

    // Install tui-logger so the Console overlay can display structured logs.
    let _ = tui_logger::init_logger(log::LevelFilter::Debug);
    tui_logger::set_default_level(log::LevelFilter::Info);

    let orig = enter_raw_mode()?;

    // Enable SGR mouse tracking.
    {
        use std::io::Write;
        let _ = std::io::stdout().write_all(b"\x1b[?1002h\x1b[?1006h");
        let _ = std::io::stdout().flush();
    }

    // Derive storage key for private chat sessions.
    let storage_key = derive_storage_key(signing_key);

    // Console overlay for log viewing (F9 / Ctrl-L).
    let mut console_app = ConsoleApp::new();

    // Active client-owned ChatApps keyed by pane_id.
    let mut active_apps: HashMap<u32, ChatApp> = HashMap::new();
    // Local pane ID counter — high bit set to avoid collision with server IDs.
    let mut next_local_id: u32 = hyprstream_compositor::LOCAL_ID_BIT;

    let mut saw_ctrl_b = false;
    let mut win_refresh = tokio::time::interval(std::time::Duration::from_secs(2));
    win_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut bg_tick = tokio::time::interval(std::time::Duration::from_millis(50));
    bg_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Service/worker panel polling (5s, only active when modals are open).
    let mut panel_refresh = tokio::time::interval(std::time::Duration::from_secs(5));
    panel_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let (service_data_tx, mut service_data_rx) =
        tokio::sync::mpsc::channel::<Vec<ServiceEntry>>(4);

    // SIGWINCH — forward terminal resize to TuiService and compositor.
    let sigwinch_key = signing_key.clone();
    let resize_tx_sigwinch = resize_tx.clone();
    let _sigwinch_handle = tokio::spawn(async move {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigwinch = match signal(SignalKind::window_change()) {
            Ok(s) => s,
            Err(_) => return,
        };
        let resize_client = create_tui_client(&sigwinch_key);
        loop {
            sigwinch.recv().await;
            let (c, r) = terminal_size();
            // Send pane dimensions to server (full rows minus 5 chrome rows, cols minus 2 borders),
            // matching the initial connect() call which sends pane dimensions not terminal dimensions.
            let _ = resize_rpc(&resize_client, c.saturating_sub(2), r.saturating_sub(5)).await;
            // Compositor needs full terminal rows to recalculate layout.
            let _ = resize_tx_sigwinch.try_send((c, r));
        }
    });

    // Initial render.
    composite_draw(&mut terminal, &compositor, &mut console_app);

    let mut should_exit = false;

    loop {
        tokio::select! {
            Some(frame_bytes) = frame_rx.recv() => {
                // Decode structured TuiFrame (Capnp mode) and route by embedded pane_id.
                let outputs = match decode_tui_frame(&frame_bytes) {
                    Ok(mut frame) => {
                        if frame.pane_id == 0 {
                            frame.pane_id = compositor.active_pane_id();
                        }
                        compositor.handle(CompositorInput::ServerFrameCapnp { frame })
                    }
                    Err(_) => {
                        // Fallback: treat as raw ANSI (shouldn't happen in Capnp mode).
                        let pane_id = compositor.active_pane_id();
                        compositor.handle(CompositorInput::ServerFrame { pane_id, ansi: frame_bytes })
                    }
                };
                if dispatch_outputs(
                    &mut compositor, &client, &model_client,
                    &model_status_tx, &mut terminal, &mut console_app,
                    &mut active_apps, &mut next_local_id, &storage_key, signing_key, outputs,
                ).await { break; }
            }

            Some(raw) = stdin_rx.recv() => {
                // Ctrl-B D/Q — detach/quit at event loop level.
                for &b in &raw {
                    if saw_ctrl_b && (b == b'd' || b == b'q') {
                        should_exit = true;
                        break;
                    }
                    saw_ctrl_b = b == 0x02;
                }
                if should_exit { break; }

                // SGR mouse — window strip tab click.
                if let Some((col, row)) = parse_sgr_mouse_click(&raw) {
                    let win_strip_row = rows.saturating_sub(2);
                    if row == win_strip_row {
                        if let Some(idx) = tab_index_at_col(&compositor.chrome.windows, col) {
                            compositor.chrome.active_win = idx;
                            if let Some(w) = compositor.chrome.windows.get(idx) {
                                let wid = w.id;
                                let _ = focus_window_rpc(&client, wid).await;
                            }
                            composite_draw(&mut terminal, &compositor, &mut console_app);
                        }
                        continue;
                    }
                    let (pane_top, pane_rows) =
                        if matches!(compositor.chrome.mode, hyprstream_compositor::ShellMode::Fullscreen) {
                            (0u16, rows)
                        } else {
                            (2u16, rows.saturating_sub(5))
                        };
                    if row < pane_top || row >= pane_top + pane_rows {
                        continue;
                    }
                }

                let keys = input_parser.parse(&raw);
                for key in keys {
                    let outputs = compositor.handle(CompositorInput::KeyPress(key));
                    if dispatch_outputs(
                        &mut compositor, &client, &model_client,
                        &model_status_tx, &mut terminal, &mut console_app,
                        &mut active_apps, &mut next_local_id, &storage_key, signing_key, outputs,
                    ).await {
                        should_exit = true;
                        break;
                    }
                }
                if should_exit { break; }
            }

            _ = win_refresh.tick() => {
                if let Ok(wins) = list_windows_rpc(&client).await {
                    let outputs = compositor.handle(CompositorInput::WindowList(wins));
                    if dispatch_outputs(
                        &mut compositor, &client, &model_client,
                        &model_status_tx, &mut terminal, &mut console_app,
                        &mut active_apps, &mut next_local_id, &storage_key, signing_key, outputs,
                    ).await { break; }
                }
            }

            _ = bg_tick.tick() => {
                let bg_dirty = compositor.chrome.tick_bg();
                let toast_dirty = compositor.chrome.tick_toasts();
                if bg_dirty || toast_dirty {
                    composite_draw(&mut terminal, &compositor, &mut console_app);
                }
                // Tick active ChatApps and collect frames / quit IDs.
                // Two-pass to avoid holding &mut active_apps while dispatching.
                let mut dirty_frames: Vec<(u32, Vec<u8>)> = Vec::new();
                let mut quit_app_ids: Vec<u32> = Vec::new();
                for (&pane_id, app) in active_apps.iter_mut() {
                    let dirty = app.tick(50);
                    // Surface inference errors as compositor toasts.
                    for msg in app.pending_toasts.drain(..) {
                        compositor.chrome.push_toast(msg, ToastLevel::Error);
                    }
                    if app.should_quit() {
                        quit_app_ids.push(pane_id);
                        continue;
                    }
                    if dirty {
                        dirty_frames.push((pane_id, render_chat_app_to_ansi(app)));
                    }
                }
                // Feed dirty frames into compositor.
                for (pane_id, ansi) in dirty_frames {
                    let outputs = compositor.handle(CompositorInput::AppFrame { app_id: pane_id, ansi });
                    if dispatch_outputs(
                        &mut compositor, &client, &model_client,
                        &model_status_tx, &mut terminal, &mut console_app,
                        &mut active_apps, &mut next_local_id, &storage_key, signing_key, outputs,
                    ).await { should_exit = true; break; }
                }
                // Remove quitting apps.
                for id in quit_app_ids {
                    active_apps.remove(&id);
                    compositor.chrome.private_panes.remove(&id);
                    let outputs = compositor.handle(CompositorInput::AppExited { app_id: id });
                    dispatch_outputs(
                        &mut compositor, &client, &model_client,
                        &model_status_tx, &mut terminal, &mut console_app,
                        &mut active_apps, &mut next_local_id, &storage_key, signing_key, outputs,
                    ).await;
                }
                if should_exit { break; }
            }

            Some((model_ref, loaded)) = model_status_rx.recv() => {
                compositor.chrome.update_model_status(&model_ref, loaded);
                if !loaded {
                    compositor.chrome.push_toast(
                        format!("Model load timed out: {model_ref}"),
                        ToastLevel::Error,
                    );
                }
                composite_draw(&mut terminal, &compositor, &mut console_app);
            }

            Some((new_cols, new_rows)) = resize_rx.recv() => {
                rows = new_rows;
                // Resize the ratatui backend to the new dimensions.
                terminal.backend_mut().resize(new_cols, new_rows);
                let _ = terminal.clear();
                let outputs = compositor.handle(CompositorInput::Resize(new_cols, new_rows));
                if dispatch_outputs(
                    &mut compositor, &client, &model_client,
                    &model_status_tx, &mut terminal, &mut console_app,
                    &mut active_apps, &mut next_local_id, &storage_key, signing_key, outputs,
                ).await { break; }
                composite_draw(&mut terminal, &compositor, &mut console_app);
            }

            _ = panel_refresh.tick() => {
                let is_svc = matches!(compositor.chrome.mode, hyprstream_compositor::ShellMode::ServiceManager { .. });
                if is_svc {
                    let tx = service_data_tx.clone();
                    tokio::spawn(async move {
                        let entries = poll_service_status().await;
                        let _ = tx.try_send(entries);
                    });
                }
            }

            Some(entries) = service_data_rx.recv() => {
                let outputs = compositor.handle(CompositorInput::ServiceList(entries));
                if dispatch_outputs(
                    &mut compositor, &client, &model_client,
                    &model_status_tx, &mut terminal, &mut console_app,
                    &mut active_apps, &mut next_local_id, &storage_key, signing_key, outputs,
                ).await { break; }
            }

            else => break,
        }
    }

    // Disable mouse tracking.
    {
        use std::io::Write;
        let _ = std::io::stdout().write_all(b"\x1b[?1002l\x1b[?1006l");
        let _ = std::io::stdout().flush();
    }
    let _ = terminal.show_cursor();
    let _ = terminal.flush();
    restore_terminal(&orig);
    Ok(())
}

// ============================================================================
// dispatch_outputs — translate CompositorOutput to ZMQ calls
// ============================================================================

// ============================================================================
// composite_draw — render compositor + optional Console overlay
// ============================================================================

fn composite_draw(
    terminal: &mut ratatui::Terminal<AnsiBackend<AnsiWriter>>,
    compositor: &hyprstream_compositor::Compositor,
    console_app: &mut ConsoleApp,
) {
    let _ = terminal.draw(|f| {
        compositor.render(f);
        if matches!(compositor.chrome.mode, hyprstream_compositor::ShellMode::Console) {
            use ratatui::layout::{Constraint, Flex, Layout};
            use ratatui::widgets::Clear;
            let area = f.area();
            let pane_area = compositor.pane_block_area(area);
            let [v] = Layout::vertical([Constraint::Percentage(68)]).flex(Flex::Center).areas(pane_area);
            let [popup] = Layout::horizontal([Constraint::Percentage(82)]).flex(Flex::Center).areas(v);
            f.render_widget(Clear, popup);
            let block = hyprstream_compositor::theme::modal_block(ratatui::text::Line::from(" Console "));
            let inner = block.inner(popup);
            f.render_widget(block, popup);
            console_app.draw(f, inner);
        }
    });
}

/// Dispatch compositor outputs to the appropriate ZMQ RPC calls.
/// Returns `true` if the session should exit.
async fn dispatch_outputs(
    compositor: &mut Compositor,
    client: &TuiClient,
    model_client: &crate::services::ModelZmqClient,
    model_status_tx: &tokio::sync::mpsc::Sender<(String, bool)>,
    terminal: &mut ratatui::Terminal<AnsiBackend<AnsiWriter>>,
    console_app: &mut ConsoleApp,
    active_apps: &mut HashMap<u32, ChatApp>,
    next_local_id: &mut u32,
    storage_key: &StorageKey,
    signing_key: &SigningKey,
    outputs: Vec<CompositorOutput>,
) -> bool {
    for output in outputs {
        match output {
            CompositorOutput::Redraw => {
                // Trigger immediate poll on first open of service/worker panels.
                if matches!(compositor.chrome.mode, hyprstream_compositor::ShellMode::ServiceManager { .. })
                    && compositor.chrome.service_list.is_empty()
                {
                    let entries = poll_service_status().await;
                    compositor.chrome.update_service_list(entries);
                }
                composite_draw(terminal, compositor, console_app);
            }
            CompositorOutput::Quit => return true,
            CompositorOutput::Rpc(req) => {
                let feed_back = handle_rpc(
                    compositor, client, model_client, model_status_tx,
                    active_apps, next_local_id, storage_key, signing_key, req,
                ).await;
                for input in feed_back {
                    let follow = compositor.handle(input);
                    for fo in follow {
                        match fo {
                            CompositorOutput::Redraw => { composite_draw(terminal, compositor, console_app); }
                            CompositorOutput::Quit => return true,
                            _ => {}
                        }
                    }
                }
            }
            CompositorOutput::RouteInput { app_id, data } => {
                if let Some(app) = active_apps.get_mut(&app_id) {
                    let keys: Vec<waxterm::input::KeyPress> = InputParser::new(vec![]).parse(&data);
                    for key in keys {
                        if app.handle_input(key) {
                            let ansi = render_chat_app_to_ansi(app);
                            let outs = compositor.handle(CompositorInput::AppFrame { app_id, ansi });
                            for o in outs {
                                if let CompositorOutput::Redraw = o {
                                    composite_draw(terminal, compositor, console_app);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

/// Translate an `RpcRequest` into ZMQ calls and return `CompositorInput`
/// events to feed back (e.g. updated window list after create/close).
async fn handle_rpc(
    compositor: &mut Compositor,
    client: &TuiClient,
    model_client: &crate::services::ModelZmqClient,
    model_status_tx: &tokio::sync::mpsc::Sender<(String, bool)>,
    active_apps: &mut HashMap<u32, ChatApp>,
    next_local_id: &mut u32,
    storage_key: &StorageKey,
    signing_key: &SigningKey,
    req: RpcRequest,
) -> Vec<CompositorInput> {
    let session_id = compositor.chrome.session_id;
    match req {
        RpcRequest::SendInput { viewer_id, data } => {
            send_input_rpc(client, viewer_id, &data).await;
            vec![]
        }

        RpcRequest::CreateWindow { .. } => {
            // Create window + spawn shell, then advance active_win to the new window.
            if let Ok(win_info) = client.create_window().await {
                let new_win_id = win_info.id;
                let _ = spawn_shell_rpc(client, session_id, win_info.active_pane_id, "").await;
                let _ = focus_window_rpc(client, new_win_id).await;
                let wins = list_windows_rpc(client).await.unwrap_or_default();
                if let Some(idx) = wins.iter().position(|w| w.id == new_win_id) {
                    compositor.chrome.active_win = idx;
                }
                return vec![CompositorInput::WindowList(wins)];
            }
            vec![]
        }

        RpcRequest::CloseWindow { window_id, .. } => {
            // Collect pane IDs before the RPC removes the window from TuiState.
            let pane_ids: Vec<u32> = compositor.chrome.windows
                .iter()
                .find(|w| w.id == window_id)
                .map(|w| w.panes.iter().map(|p| p.id).collect())
                .unwrap_or_default();
            // Clean up client-owned ChatApps for any private panes in this window.
            for id in &pane_ids {
                active_apps.remove(id);
            }
            let _ = close_window_rpc(client, window_id).await;
            // Remove each pane's VT buffer from the client-side LayoutTree.
            for id in pane_ids {
                compositor.handle(CompositorInput::PaneClosed { pane_id: id });
            }
            refresh_windows(client).await
        }

        RpcRequest::FocusWindow { window_id, .. } => {
            if focus_window_rpc(client, window_id).await.is_ok() {
                if let Some(idx) = compositor.chrome.windows.iter().position(|w| w.id == window_id) {
                    compositor.chrome.active_win = idx;
                }
            }
            vec![]
        }

        RpcRequest::SpawnShell { pane_id, cwd, .. } => {
            if pane_id == 0 {
                // pane_id == 0 means "new window" — create one and spawn the shell
                // in its pane with the requested CWD (same pattern as SpawnServerChat).
                if let Ok(win_info) = client.create_window().await {
                    let new_win_id = win_info.id;
                    let _ = focus_window_rpc(client, new_win_id).await;
                    let _ = spawn_shell_rpc(client, session_id, win_info.active_pane_id, &cwd).await;
                    let wins = list_windows_rpc(client).await.unwrap_or_default();
                    if let Some(idx) = wins.iter().position(|w| w.id == new_win_id) {
                        compositor.chrome.active_win = idx;
                    }
                    return vec![CompositorInput::WindowList(wins)];
                }
                vec![]
            } else {
                // pane_id given — spawn shell in the existing pane.
                let _ = spawn_shell_rpc(client, session_id, pane_id, &cwd).await;
                refresh_windows(client).await
            }
        }

        RpcRequest::SpawnServerChat { model_ref, cols, rows, .. } => {
            // Create a window, spawn chat app in its pane, then advance active_win.
            if let Ok(win_info) = client.create_window().await {
                let new_win_id = win_info.id;
                let _ = focus_window_rpc(client, new_win_id).await;
                let _ = spawn_chat_app_rpc(client, session_id, &model_ref, cols, rows,
                                           win_info.active_pane_id).await;
                let wins = list_windows_rpc(client).await.unwrap_or_default();
                if let Some(idx) = wins.iter().position(|w| w.id == new_win_id) {
                    compositor.chrome.active_win = idx;
                }
                return vec![CompositorInput::WindowList(wins)];
            }
            vec![]
        }

        RpcRequest::LoadModel { model_ref } => {
            let _ = model_client.load(&model_ref, None).await;
            // Spawn background polling task.
            let poll_client = model_client.clone();
            let poll_mr    = model_ref.clone();
            let tx         = model_status_tx.clone();
            tokio::spawn(async move {
                for _ in 0..60u32 {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    match poll_client.status(&poll_mr).await {
                        Ok(entries) if entries.iter().any(|e| e.status == "loaded") => {
                            let _ = tx.send((poll_mr, true)).await;
                            return;
                        }
                        _ => {}
                    }
                }
                let _ = tx.send((poll_mr, false)).await;
            });
            vec![]
        }

        RpcRequest::UnloadModel { model_ref } => {
            let ok = model_client.unload(&model_ref).await.is_ok();
            if ok {
                compositor.chrome.update_model_status(&model_ref, false);
            }
            vec![]
        }

        RpcRequest::LocalPrivateChat { model_ref, cols, rows, resume_uuid } => {
            // Allocate a local pane ID in the high-bit range to avoid
            // collision with server-allocated IDs (which start at 1).
            let pane_id = *next_local_id;
            *next_local_id = next_local_id.saturating_add(1);

            let session_uuid = match &resume_uuid {
                Some(s) => uuid::Uuid::parse_str(s).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                None => uuid::Uuid::new_v4(),
            };
            let storage_dir = private_store_dir();
            let app = match FsBackend::new(storage_dir) {
                Ok(fs) => {
                    let key_bytes: [u8; 32] = **storage_key;
                    let fs_store = std::sync::Arc::new(
                        hyprstream_tui::private_store::PrivateStore::new(
                            fs,
                            Zeroizing::new(key_bytes),
                        )
                    );
                    // Register conversation in manifest (skip if resuming).
                    if resume_uuid.is_none() {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        let _ = fs_store.register_conversation(
                            hyprstream_tui::private_store::ConversationMeta {
                                uuid: session_uuid,
                                model_ref: model_ref.clone(),
                                created_at: now,
                                last_active: now,
                                label: Some(format!("Chat: {}", model_ref)),
                            },
                        );
                    }
                    let load_store = fs_store.clone();
                    let save_store = fs_store;
                    let uuid_for_load = session_uuid;
                    let uuid_for_save = session_uuid;
                    let load_hook: LoadHook = Box::new(move || {
                        load_store.load::<hyprstream_tui::chat_app::ChatHistoryEntry>(&uuid_for_load)
                            .ok()
                            .flatten()
                    });
                    let save_hook: SaveHook = Box::new(move |entries| {
                        let _ = save_store.save(&uuid_for_save, entries);
                    });
                    let spawner = make_chat_spawner(signing_key, &model_ref);
                    ChatApp::new_private(
                        model_ref.clone(), cols, rows, spawner,
                        session_uuid, load_hook, save_hook,
                    )
                }
                Err(e) => {
                    compositor.chrome.push_toast(
                        format!("Private store dir unavailable: {e}"),
                        ToastLevel::Warn,
                    );
                    ChatApp::new(model_ref.clone(), cols, rows, make_chat_spawner(signing_key, &model_ref))
                }
            };
            active_apps.insert(pane_id, app);
            compositor.chrome.private_panes.insert(pane_id);
            compositor.layout.get_or_create_private(pane_id);

            // Auto-number window names for duplicate models.
            let chat_prefix = format!("Chat: {}", model_ref);
            let existing_count = compositor.chrome.windows.iter()
                .filter(|w| w.name == chat_prefix || w.name.starts_with(&format!("{} (", chat_prefix)))
                .count();
            let win_name = if existing_count > 0 {
                format!("{} ({})", chat_prefix, existing_count + 1)
            } else {
                chat_prefix
            };

            // Insert a phantom local window into chrome (no server round-trip).
            let win_summary = WindowSummary {
                id: pane_id, // use pane_id as window_id for local windows
                name: win_name,
                active_pane_id: pane_id,
                panes: vec![PaneSummary {
                    id: pane_id,
                    cols,
                    rows,
                    is_private: true,
                }],
            };
            compositor.chrome.windows.push(win_summary);
            compositor.chrome.active_win = compositor.chrome.windows.len() - 1;

            // Render first frame.
            if let Some(a) = active_apps.get(&pane_id) {
                let ansi = render_chat_app_to_ansi(a);
                compositor.handle(CompositorInput::AppFrame { app_id: pane_id, ansi });
            }
            vec![]
        }

        RpcRequest::ListConversations { model_ref } => {
            let storage_dir = private_store_dir();
            let conversations = match FsBackend::new(storage_dir) {
                Ok(fs) => {
                    let key_bytes: [u8; 32] = **storage_key;
                    let store = hyprstream_tui::private_store::PrivateStore::new(
                        fs, Zeroizing::new(key_bytes),
                    );
                    store.list_conversations(&model_ref)
                        .into_iter()
                        .map(|meta| ConversationPickerEntry {
                            uuid: meta.uuid.to_string(),
                            label: meta.label.unwrap_or_else(|| format!("Chat: {}", meta.model_ref)),
                            last_active: meta.last_active,
                        })
                        .collect()
                }
                Err(_) => vec![],
            };
            compositor.chrome.open_conversation_picker(model_ref, conversations);
            vec![]
        }

        RpcRequest::DeleteConversation { uuid, model_ref } => {
            let storage_dir = private_store_dir();
            if let Ok(fs) = FsBackend::new(storage_dir) {
                let key_bytes: [u8; 32] = **storage_key;
                let store = hyprstream_tui::private_store::PrivateStore::new(
                    fs, Zeroizing::new(key_bytes),
                );
                if let Ok(uid) = uuid::Uuid::parse_str(&uuid) {
                    let _ = store.delete_conversation(&uid);
                }
            }
            compositor.chrome.push_toast(
                format!("Deleted conversation from {}", model_ref),
                ToastLevel::Info,
            );
            vec![]
        }

        RpcRequest::ServiceStart { name } => {
            match do_service_start(&name).await {
                Ok(()) => compositor.chrome.push_toast(
                    format!("Started {name}"), ToastLevel::Info,
                ),
                Err(e) => compositor.chrome.push_toast(
                    format!("Start {name}: {e}"), ToastLevel::Error,
                ),
            }
            let entries = poll_service_status().await;
            vec![CompositorInput::ServiceList(entries)]
        }
        RpcRequest::ServiceStop { name } => {
            match do_service_stop(&name).await {
                Ok(()) => compositor.chrome.push_toast(
                    format!("Stopped {name}"), ToastLevel::Info,
                ),
                Err(e) => compositor.chrome.push_toast(
                    format!("Stop {name}: {e}"), ToastLevel::Error,
                ),
            }
            let entries = poll_service_status().await;
            vec![CompositorInput::ServiceList(entries)]
        }
        RpcRequest::ServiceRestart { name } => {
            let _ = do_service_stop(&name).await;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            match do_service_start(&name).await {
                Ok(()) => compositor.chrome.push_toast(
                    format!("Restarted {name}"), ToastLevel::Info,
                ),
                Err(e) => compositor.chrome.push_toast(
                    format!("Restart {name}: {e}"), ToastLevel::Error,
                ),
            }
            let entries = poll_service_status().await;
            vec![CompositorInput::ServiceList(entries)]
        }
        RpcRequest::ServiceInstall => {
            compositor.chrome.push_toast("Installing services...".to_owned(), ToastLevel::Info);
            if hyprstream_rpc::has_systemd() {
                if let Ok(manager) = hyprstream_rpc::detect_service_manager().await {
                    for &svc in ALL_SERVICE_NAMES {
                        let _ = manager.install(svc).await;
                    }
                }
            }
            compositor.chrome.push_toast("Install complete".to_owned(), ToastLevel::Info);
            let entries = poll_service_status().await;
            vec![CompositorInput::ServiceList(entries)]
        }
        RpcRequest::ServiceStopAll => {
            for &svc in ALL_SERVICE_NAMES {
                let _ = do_service_stop(svc).await;
            }
            compositor.chrome.push_toast("All services stopped".to_owned(), ToastLevel::Warn);
            let entries = poll_service_status().await;
            vec![CompositorInput::ServiceList(entries)]
        }
        RpcRequest::ServiceStartAll => {
            for &svc in ALL_SERVICE_NAMES {
                let _ = do_service_start(svc).await;
            }
            compositor.chrome.push_toast("All services started".to_owned(), ToastLevel::Info);
            let entries = poll_service_status().await;
            vec![CompositorInput::ServiceList(entries)]
        }
        RpcRequest::WorkerDestroySandbox { sandbox_id } => {
            compositor.chrome.push_toast(
                format!("Destroying {}...", &sandbox_id[..sandbox_id.len().min(8)]),
                ToastLevel::Info,
            );
            // Worker RPC not yet wired — placeholder for Phase 6.
            vec![]
        }
        RpcRequest::WorkerExecSync { sandbox_id: _, container_id, cmd } => {
            compositor.chrome.push_toast(
                format!("Exec in {}: {:?}", &container_id[..container_id.len().min(8)], cmd),
                ToastLevel::Info,
            );
            // Worker RPC not yet wired — placeholder for Phase 6.
            vec![]
        }
        RpcRequest::Quit => vec![],
    }
}

/// Call listWindows and return a `WindowList` compositor input.
async fn refresh_windows(client: &TuiClient) -> Vec<CompositorInput> {
    match list_windows_rpc(client).await {
        Ok(wins) => vec![CompositorInput::WindowList(wins)],
        Err(_) => vec![],
    }
}

// ============================================================================
// handle_tui_shell — embedded mode (hyprstream tui shell)
// ============================================================================

/// Spawn ShellApp inside TuiService via SpawnChromeShell RPC.
///
/// The ShellApp runs server-side as a `PaneProcess` — no fork occurs in this
/// process after ZMQ is initialised.  Playwright tests use this path.
pub async fn handle_tui_shell(
    signing_key: &SigningKey,
    models_dir: &std::path::Path,
    session_id: u32,
) -> Result<()> {
    use crate::cli::tui_handlers::{create_tui_client, terminal_size};

    let (cols, rows) = terminal_size();
    let client = create_tui_client(signing_key);

    let registry_dir = models_dir.join(".registry").join("models");
    let registry_dir = if registry_dir.exists() { registry_dir } else { models_dir.to_path_buf() };
    let registry_str = registry_dir.to_string_lossy().into_owned();

    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        let mut spawn = req.init_spawn_chrome_shell();
        spawn.set_session_id(session_id);
        spawn.set_registry_dir(&registry_str);
        spawn.set_cols(cols);
        spawn.set_rows(rows);
        spawn.set_pane_id(0);
    })?;
    client.call(payload).await.context("spawnChromeShell RPC failed")?;

    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?
        .recv()
        .await;

    Ok(())
}

// ============================================================================
// Helpers — model fetch
// ============================================================================

async fn fetch_models(
    signing_key: &SigningKey,
    models_dir: &std::path::Path,
) -> Vec<ModelEntry> {
    use hyprstream_rpc::envelope::RequestIdentity;
    let registry_endpoint = hyprstream_rpc::registry::global()
        .endpoint("registry", hyprstream_rpc::registry::SocketKind::Rep)
        .to_zmq_string();
    let registry: crate::services::GenRegistryClient = crate::services::create_service_client(
        &registry_endpoint,
        signing_key.clone(),
        RequestIdentity::local(),
    );
    let model_client_for_status = crate::services::ModelZmqClient::new(
        signing_key.clone(),
        RequestIdentity::local(),
    );
    let registry_models_dir = models_dir.to_path_buf();
    let status_timeout = std::time::Duration::from_millis(500);

    let (repos_result, status_result) = tokio::join!(
        registry.list(),
        tokio::time::timeout(status_timeout, model_client_for_status.status("")),
    );

    let status_map: std::collections::HashMap<String, bool> = match status_result {
        Ok(Ok(entries)) => entries.into_iter()
            .map(|e| (e.model_ref, e.status == "loaded"))
            .collect(),
        _ => std::collections::HashMap::new(),
    };

    match repos_result {
        Ok(repos) => repos
            .into_iter()
            .filter(|r| !r.name.is_empty())
            .flat_map(|r| {
                let name = r.name.clone();
                let rmd  = registry_models_dir.clone();
                r.worktrees.into_iter().map(move |wt| {
                    let branch = if wt.branch_name.is_empty() { "main".to_owned() } else { wt.branch_name };
                    let model_ref = format!("{}:{}", name, branch);
                    let path = rmd.join(&name).join("worktrees").join(&branch);
                    ModelEntry { model_ref, path, loaded: false, loading: false }
                })
                .collect::<Vec<_>>()
            })
            .map(|mut entry| {
                entry.loaded = *status_map.get(&entry.model_ref).unwrap_or(&false);
                entry
            })
            .collect(),
        Err(_) => vec![],
    }
}

// ============================================================================
// ZMQ RPC helpers
// ============================================================================

async fn list_windows_rpc(client: &TuiClient) -> anyhow::Result<Vec<WindowSummary>> {
    use crate::services::generated::tui_client::TuiResponseVariant;
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_list_windows(0);
    })?;
    let response = client.call(payload).await?;
    match TuiClient::parse_response(&response)? {
        TuiResponseVariant::ListWindowsResult(list) => Ok(list
            .windows
            .iter()
            .map(|w| WindowSummary {
                id: w.id,
                name: w.name.clone(),
                active_pane_id: w.active_pane_id,
                panes: w.panes.iter().map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows, is_private: p.is_private }).collect(),
            })
            .collect()),
        _ => anyhow::bail!("unexpected response"),
    }
}

async fn resize_rpc(client: &TuiClient, cols: u16, rows: u16) -> anyhow::Result<()> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        let mut r = req.init_resize();
        r.set_cols(cols);
        r.set_rows(rows);
    })?;
    client.call(payload).await?;
    Ok(())
}

async fn send_input_rpc(client: &TuiClient, viewer_id: u32, data: &[u8]) {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        let mut send_input = req.init_send_input();
        send_input.set_viewer_id(viewer_id);
        send_input.set_data(data);
    });
    if let Ok(p) = payload {
        if let Err(e) = client.call(p).await {
            tracing::debug!("sendInput failed: {e}");
        }
    }
}

// ============================================================================
// Service & Worker polling
// ============================================================================

const ALL_SERVICE_NAMES: &[&str] = &[
    "oai", "registry", "model", "inference", "worker",
    "policy", "mcp", "streams", "event", "tui",
];

/// Poll systemd and PID files for service status.
async fn poll_service_status() -> Vec<ServiceEntry> {
    let manager = if hyprstream_rpc::has_systemd() {
        hyprstream_rpc::detect_service_manager().await.ok()
    } else {
        None
    };
    let runtime_dir = hyprstream_rpc::paths::runtime_dir();

    let mut entries = Vec::with_capacity(ALL_SERVICE_NAMES.len());
    for &name in ALL_SERVICE_NAMES {
        let systemd_active = if let Some(ref mgr) = manager {
            mgr.is_active(name).await.unwrap_or(false)
        } else {
            false
        };

        let (daemon_active, daemon_pid) = {
            let pid_file = runtime_dir.join(format!("{name}.pid"));
            if pid_file.exists() {
                if let Ok(s) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = s.trim().parse::<i32>() {
                        let alive = nix::sys::signal::kill(
                            nix::unistd::Pid::from_raw(pid), None,
                        ).is_ok();
                        (alive, if alive { Some(pid as u32) } else { None })
                    } else { (false, None) }
                } else { (false, None) }
            } else { (false, None) }
        };

        let (active, mode, pid) = match (systemd_active, daemon_active) {
            (true, true)   => (true,  ServiceMode::Both,    daemon_pid),
            (true, false)  => (true,  ServiceMode::Systemd, None),
            (false, true)  => (true,  ServiceMode::Daemon,  daemon_pid),
            (false, false) => (false, ServiceMode::Stopped, None),
        };

        entries.push(ServiceEntry { name: name.to_owned(), active, mode, pid });
    }
    entries
}

/// Start a single service (systemd or standalone fallback).
async fn do_service_start(name: &str) -> anyhow::Result<()> {
    if hyprstream_rpc::has_systemd() {
        let manager = hyprstream_rpc::detect_service_manager().await?;
        manager.start(name).await
    } else {
        let spawner = hyprstream_rpc::ProcessSpawner::standalone();
        let exe = hyprstream_rpc::paths::executable_path()?;
        let config = hyprstream_rpc::ProcessConfig::new(name, &exe)
            .args(["service", "start", name, "--foreground", "--ipc"]);
        spawner.spawn(config).await.map(|_| ()).map_err(Into::into)
    }
}

/// Stop a single service (systemd or SIGTERM via PID file).
async fn do_service_stop(name: &str) -> anyhow::Result<()> {
    if hyprstream_rpc::has_systemd() {
        let manager = hyprstream_rpc::detect_service_manager().await?;
        manager.stop(name).await
    } else {
        let runtime_dir = hyprstream_rpc::paths::runtime_dir();
        let pid_file = runtime_dir.join(format!("{name}.pid"));
        if let Ok(s) = std::fs::read_to_string(&pid_file) {
            if let Ok(pid) = s.trim().parse::<i32>() {
                let _ = nix::sys::signal::kill(
                    nix::unistd::Pid::from_raw(pid),
                    nix::sys::signal::Signal::SIGTERM,
                );
                let _ = std::fs::remove_file(&pid_file);
            }
        }
        Ok(())
    }
}

// ============================================================================
// AnsiWriter newtype
// ============================================================================

struct AnsiWriter(std::io::Stdout);

impl std::io::Write for AnsiWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> std::io::Result<()> { self.0.flush() }
}

// ============================================================================
// Mouse helpers
// ============================================================================

fn parse_sgr_mouse_click(data: &[u8]) -> Option<(u16, u16)> {
    let mut i = 0;
    while i + 5 < data.len() {
        if data[i] == 0x1b && data[i + 1] == b'[' && data[i + 2] == b'<' {
            let rest = &data[i + 3..];
            if let Some(end) = rest.iter().position(|&b| b == b'M' || b == b'm') {
                let term = rest[end];
                let params = std::str::from_utf8(&rest[..end]).ok()?;
                let parts: Vec<&str> = params.split(';').collect();
                if parts.len() == 3 && term == b'M' {
                    let col = parts[1].parse::<u16>().ok()?.saturating_sub(1);
                    let row = parts[2].parse::<u16>().ok()?.saturating_sub(1);
                    return Some((col, row));
                }
                i += 3 + end + 1;
                continue;
            }
        }
        i += 1;
    }
    None
}

fn tab_index_at_col(windows: &[WindowSummary], col: u16) -> Option<usize> {
    let mut x = 0u16;
    for (i, win) in windows.iter().enumerate() {
        let label_width = (win.name.len() as u16) + 2;
        let divider = if i + 1 < windows.len() { 1 } else { 0 };
        if col >= x && col < x + label_width {
            return Some(i);
        }
        x += label_width + divider;
    }
    None
}

// ============================================================================
// Private chat helpers
// ============================================================================

/// Build a `StreamSpawner` — delegates to `tui::zmq_transport`.
fn make_chat_spawner(
    signing_key: &SigningKey,
    model_ref: &str,
) -> hyprstream_tui::chat_app::StreamSpawner {
    crate::tui::zmq_transport::make_chat_spawner(signing_key, model_ref)
}

/// Derive a 32-byte AES-256 storage key from the Ed25519 signing key seed.
///
/// Uses Blake3 `derive_key` (same as `hyprstream_rpc::crypto::backend::derive_key`)
/// so the key is deterministic and session-independent (rotating the signing key
/// invalidates all history — documented V1 limitation).
fn derive_storage_key(signing_key: &SigningKey) -> StorageKey {
    let seed = signing_key.to_bytes(); // 32-byte seed (pre-SHA-512)
    let key = hyprstream_rpc::crypto::backend::derive_key(
        "hyprstream-storage-session-enc-v1",
        &seed,
    );
    Zeroizing::new(key)
}

/// Return the native path for private session storage files.
/// `$XDG_DATA_HOME/hyprstream/private/v1/` (or `~/.local/share/…` fallback).
fn private_store_dir() -> std::path::PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            dirs::home_dir().map(|h| h.join(".local").join("share"))
        })
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
    base.join("hyprstream").join("private").join("v1")
}

/// Render a `ChatApp` off-screen to an ANSI byte buffer.
fn render_chat_app_to_ansi(app: &ChatApp) -> Vec<u8> {
    let backend = AnsiBackend::new(Vec::<u8>::new(), app.cols, app.rows, false);
    let Ok(mut term) = ratatui::Terminal::new(backend) else {
        return Vec::new();
    };
    let _ = term.draw(|f| app.render(f));
    term.backend().writer().clone()
}

// ============================================================================
// TuiFrame decoder — Capnp → FrameUpdate
// ============================================================================

/// Decode a serialized Cap'n Proto TuiFrame into a compositor `FrameUpdate`.
///
/// Called from the frame_rx arm in the event loop when connected in Capnp mode.
/// The resulting `FrameUpdate` is passed to `CompositorInput::ServerFrameCapnp`.
fn decode_tui_frame(data: &[u8]) -> anyhow::Result<FrameUpdate> {
    use capnp::serialize;
    use crate::tui_capnp;

    let reader = serialize::read_message_from_flat_slice(
        &mut &data[..],
        capnp::message::ReaderOptions::default(),
    )?;
    let frame = reader.get_root::<tui_capnp::tui_frame::Reader<'_>>()?;

    let pane_id    = frame.get_pane_id();
    let generation = frame.get_generation();

    let cursor = {
        let c = frame.get_cursor()?;
        CursorState { x: c.get_x(), y: c.get_y(), visible: c.get_visible() }
    };

    let content = match frame.which()? {
        tui_capnp::tui_frame::Which::Full(full_r) => {
            let full = full_r?;
            let cols = full.get_cols();
            let rows = full.get_rows();
            let cells = full.get_cells()?.iter().map(|c| CellUpdate {
                x: c.get_x(),
                y: c.get_y(),
                symbol: c.get_symbol().ok().and_then(|r| r.to_str().ok()).unwrap_or("").to_owned(),
                fg:        unpack_color(c.get_fg()),
                bg:        unpack_color(c.get_bg()),
                modifiers: ratatui::style::Modifier::from_bits_truncate(c.get_modifiers()),
            }).collect();
            FrameContent::Full { cols, rows, cells }
        }
        tui_capnp::tui_frame::Which::Incremental(incr_r) => {
            let incr = incr_r?;
            let scrolls = incr.get_scrolls()?.iter().map(|s| ScrollUpdate {
                top:    s.get_top(),
                bottom: s.get_bottom(),
                amount: s.get_amount(),
            }).collect();
            let deltas = incr.get_deltas()?.iter().map(|c| CellUpdate {
                x: c.get_x(),
                y: c.get_y(),
                symbol: c.get_symbol().ok().and_then(|r| r.to_str().ok()).unwrap_or("").to_owned(),
                fg:        unpack_color(c.get_fg()),
                bg:        unpack_color(c.get_bg()),
                modifiers: ratatui::style::Modifier::from_bits_truncate(c.get_modifiers()),
            }).collect();
            FrameContent::Incremental { scrolls, deltas }
        }
    };

    Ok(FrameUpdate { pane_id, generation, cursor, content })
}

/// Unpack a u32 color (as encoded by `diff::pack_color`) into a ratatui Color.
fn unpack_color(packed: u32) -> ratatui::style::Color {
    use ratatui::style::Color;
    if packed & 0x0200_0000 != 0 {
        Color::Indexed((packed & 0xFF) as u8)
    } else if packed & 0x0100_0000 != 0 {
        Color::Rgb(
            ((packed >> 16) & 0xFF) as u8,
            ((packed >>  8) & 0xFF) as u8,
            ( packed        & 0xFF) as u8,
        )
    } else {
        match packed {
            1  => Color::Black,
            2  => Color::Red,
            3  => Color::Green,
            4  => Color::Yellow,
            5  => Color::Blue,
            6  => Color::Magenta,
            7  => Color::Cyan,
            8  => Color::Gray,
            9  => Color::DarkGray,
            10 => Color::LightRed,
            11 => Color::LightGreen,
            12 => Color::LightYellow,
            13 => Color::LightBlue,
            14 => Color::LightMagenta,
            15 => Color::LightCyan,
            16 => Color::White,
            _  => Color::Reset,
        }
    }
}
