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
    Compositor, CompositorInput, CompositorOutput, ModelEntry, PaneSummary, RpcRequest,
    WindowSummary,
};
use hyprstream_tui::chat_app::{ChatApp, LoadHook, SaveHook};
use hyprstream_tui::private_store::{FsBackend, StorageKey};
use waxterm::app::TerminalApp;

use crate::services::generated::tui_client::TuiClient;
use crate::tui::shell_client::{
    close_window_rpc, create_private_pane_rpc, focus_window_rpc, spawn_chat_app_rpc,
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
    let pane_rows = rows.saturating_sub(3);

    // Connect to session 0
    let result = client
        .connect(0, "ansi", cols, pane_rows)
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
            panes: w.panes.iter().map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows }).collect(),
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
    let orig = enter_raw_mode()?;

    // Enable SGR mouse tracking.
    {
        use std::io::Write;
        let _ = std::io::stdout().write_all(b"\x1b[?1002h\x1b[?1006h");
        let _ = std::io::stdout().flush();
    }

    // Derive storage key for private chat sessions.
    let storage_key = derive_storage_key(signing_key);

    // Active client-owned ChatApps keyed by pane_id.
    let mut active_apps: HashMap<u32, ChatApp> = HashMap::new();

    let mut saw_ctrl_b = false;
    let mut win_refresh = tokio::time::interval(std::time::Duration::from_secs(2));
    win_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut bg_tick = tokio::time::interval(std::time::Duration::from_millis(50));
    bg_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // SIGWINCH — forward terminal resize to TuiService.
    let sigwinch_key = signing_key.clone();
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
            let _ = resize_rpc(&resize_client, c, r).await;
        }
    });

    // Initial render.
    let _ = terminal.draw(|f| compositor.render(f));

    let mut should_exit = false;

    loop {
        tokio::select! {
            Some(ansi_data) = frame_rx.recv() => {
                // All frames go to the active pane (single-stream model for Phase 0).
                let pane_id = compositor.active_pane_id();
                let outputs = compositor.handle(CompositorInput::ServerFrame { pane_id, ansi: ansi_data });
                if dispatch_outputs(
                    &mut compositor, &client, &model_client,
                    &model_status_tx, &mut terminal,
                    &mut active_apps, &storage_key, signing_key, outputs,
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
                            let _ = terminal.draw(|f| compositor.render(f));
                        }
                        continue;
                    }
                    let pane_top = 1u16;
                    if row < pane_top || row >= pane_top + pane_rows {
                        continue;
                    }
                }

                let keys = input_parser.parse(&raw);
                for key in keys {
                    let outputs = compositor.handle(CompositorInput::KeyPress(key));
                    if dispatch_outputs(
                        &mut compositor, &client, &model_client,
                        &model_status_tx, &mut terminal,
                        &mut active_apps, &storage_key, signing_key, outputs,
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
                        &model_status_tx, &mut terminal,
                        &mut active_apps, &storage_key, signing_key, outputs,
                    ).await { break; }
                }
            }

            _ = bg_tick.tick() => {
                if compositor.chrome.tick_bg() {
                    let _ = terminal.draw(|f| compositor.render(f));
                }
                // Tick active ChatApps and collect frames / quit IDs.
                // Two-pass to avoid holding &mut active_apps while dispatching.
                let mut dirty_frames: Vec<(u32, Vec<u8>)> = Vec::new();
                let mut quit_app_ids: Vec<u32> = Vec::new();
                for (&pane_id, app) in active_apps.iter_mut() {
                    let dirty = app.tick(50);
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
                        &model_status_tx, &mut terminal,
                        &mut active_apps, &storage_key, signing_key, outputs,
                    ).await { should_exit = true; break; }
                }
                // Remove quitting apps.
                for id in quit_app_ids {
                    active_apps.remove(&id);
                    compositor.chrome.private_panes.remove(&id);
                    let outputs = compositor.handle(CompositorInput::AppExited { app_id: id });
                    dispatch_outputs(
                        &mut compositor, &client, &model_client,
                        &model_status_tx, &mut terminal,
                        &mut active_apps, &storage_key, signing_key, outputs,
                    ).await;
                }
                if should_exit { break; }
            }

            Some((model_ref, loaded)) = model_status_rx.recv() => {
                compositor.chrome.update_model_status(&model_ref, loaded);
                let _ = terminal.draw(|f| compositor.render(f));
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

/// Dispatch compositor outputs to the appropriate ZMQ RPC calls.
/// Returns `true` if the session should exit.
async fn dispatch_outputs(
    compositor: &mut Compositor,
    client: &TuiClient,
    model_client: &crate::services::ModelZmqClient,
    model_status_tx: &tokio::sync::mpsc::Sender<(String, bool)>,
    terminal: &mut ratatui::Terminal<AnsiBackend<AnsiWriter>>,
    active_apps: &mut HashMap<u32, ChatApp>,
    storage_key: &StorageKey,
    signing_key: &SigningKey,
    outputs: Vec<CompositorOutput>,
) -> bool {
    for output in outputs {
        match output {
            CompositorOutput::Redraw => {
                let _ = terminal.draw(|f| compositor.render(f));
            }
            CompositorOutput::Quit => return true,
            CompositorOutput::Rpc(req) => {
                let feed_back = handle_rpc(
                    compositor, client, model_client, model_status_tx,
                    active_apps, storage_key, signing_key, req,
                ).await;
                for input in feed_back {
                    let follow = compositor.handle(input);
                    for fo in follow {
                        match fo {
                            CompositorOutput::Redraw => { let _ = terminal.draw(|f| compositor.render(f)); }
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
                                    let _ = terminal.draw(|f| compositor.render(f));
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
            // Create window + spawn shell in its default pane.
            if let Ok(win_info) = client.create_window().await {
                let pane_id = win_info.active_pane_id;
                let _ = spawn_shell_rpc(client, session_id, pane_id, "").await;
                let _ = focus_window_rpc(client, win_info.id).await;
                // Refresh window list.
                return refresh_windows(client).await;
            }
            vec![]
        }

        RpcRequest::CloseWindow { window_id, .. } => {
            let _ = close_window_rpc(client, window_id).await;
            refresh_windows(client).await
        }

        RpcRequest::FocusWindow { window_id, .. } => {
            let _ = focus_window_rpc(client, window_id).await;
            vec![]
        }

        RpcRequest::SpawnShell { pane_id, cwd, .. } => {
            // Called after CreateWindow returns a new pane_id.
            let _ = spawn_shell_rpc(client, session_id, pane_id, &cwd).await;
            refresh_windows(client).await
        }

        RpcRequest::SpawnServerChat { model_ref, cols, rows, .. } => {
            // Create a window, spawn chat app in its pane.
            if let Ok(win_info) = client.create_window().await {
                let pane_id = win_info.active_pane_id;
                let _ = focus_window_rpc(client, win_info.id).await;
                let _ = spawn_chat_app_rpc(client, session_id, &model_ref, cols, rows, pane_id).await;
                return refresh_windows(client).await;
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

        RpcRequest::CreatePrivatePane { cols, rows, name, .. } => {
            let window_id = compositor.chrome.windows
                .get(compositor.chrome.active_win)
                .map(|w| w.id)
                .unwrap_or(0);
            match create_private_pane_rpc(
                client,
                compositor.chrome.session_id,
                window_id,
                cols,
                rows,
                &name,
            ).await {
                Ok(pane_id) => {
                    let session_uuid = uuid::Uuid::new_v4();
                    let storage_dir = private_store_dir();
                    let app = match FsBackend::new(storage_dir) {
                        Ok(fs) => {
                            // Derive a fresh copy of the storage key for this store instance.
                            let key_bytes: [u8; 32] = **storage_key;
                            let fs_store = std::sync::Arc::new(
                                hyprstream_tui::private_store::PrivateStore::new(
                                    fs,
                                    Zeroizing::new(key_bytes),
                                )
                            );
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
                            let spawner = make_chat_spawner(signing_key, &name);
                            ChatApp::new_private(
                                name.clone(), cols, rows, spawner,
                                session_uuid, load_hook, save_hook,
                            )
                        }
                        Err(e) => {
                            tracing::warn!("Private store dir unavailable: {e}");
                            ChatApp::new(name.clone(), cols, rows, make_chat_spawner(signing_key, &name))
                        }
                    };
                    active_apps.insert(pane_id, app);
                    compositor.chrome.private_panes.insert(pane_id);
                    // Render initial frame immediately.
                    if let Some(a) = active_apps.get(&pane_id) {
                        let ansi = render_chat_app_to_ansi(a);
                        return vec![CompositorInput::AppFrame { app_id: pane_id, ansi }];
                    }
                    refresh_windows(client).await
                }
                Err(e) => {
                    tracing::warn!("createPrivatePane RPC failed: {e}");
                    vec![]
                }
            }
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
    let registry_models_dir = models_dir.join(".registry").join("models");
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
                    ModelEntry { model_ref, path, loaded: false }
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
                panes: w.panes.iter().map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows }).collect(),
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

/// Build a `StreamSpawner` that drives inference via `ModelZmqClient`.
///
/// Matches the pattern in `service.rs::handle_spawn_chat_app`:
/// - Spawns a dedicated OS thread with a single-threaded Tokio runtime.
/// - Applies the chat template, starts an authenticated inference stream.
/// - Sends `ChatEvent::Token` / `StreamComplete` / `StreamError` to the app.
fn make_chat_spawner(
    signing_key: &SigningKey,
    model_ref: &str,
) -> hyprstream_tui::chat_app::StreamSpawner {
    use hyprstream_rpc::envelope::RequestIdentity;
    use hyprstream_rpc::streaming::StreamPayload;
    use hyprstream_rpc::crypto::generate_ephemeral_keypair;
    use crate::services::model::ModelZmqClient;
    use crate::services::rpc_types::StreamHandle;
    use crate::config::GenerationRequest;
    use crate::api::openai_compat::ChatMessage;
    use crate::zmq::global_context;
    use hyprstream_tui::chat_app::ChatEvent;

    let sk = signing_key.clone();
    let mr = model_ref.to_owned();

    Box::new(move |pairs, event_tx| {
        let sk_inner = sk.clone();
        let mr_inner = mr.clone();
        let tx = event_tx.clone();

        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(ChatEvent::StreamError(e.to_string()));
                    return;
                }
            };

            rt.block_on(async move {
                let model_client = ModelZmqClient::new(sk_inner.clone(), RequestIdentity::local());

                let messages: Vec<ChatMessage> = pairs
                    .into_iter()
                    .map(|(role, content)| ChatMessage {
                        role,
                        content: Some(content),
                        function_call: None,
                        tool_calls: None,
                        tool_call_id: None,
                    })
                    .collect();

                let prompt = match model_client
                    .apply_chat_template(&mr_inner, &messages, true, None)
                    .await
                {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = tx.send(ChatEvent::TemplateError(e.to_string()));
                        return;
                    }
                };

                let req = GenerationRequest {
                    prompt,
                    max_tokens: 2048,
                    temperature: 0.7,
                    ..Default::default()
                };

                let (client_secret, client_pubkey) = generate_ephemeral_keypair();
                let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

                let stream_info = match model_client
                    .infer_stream(&mr_inner, &req, client_pubkey_bytes)
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(ChatEvent::StreamError(e.to_string()));
                        return;
                    }
                };

                let mut handle = match StreamHandle::new(
                    &global_context(),
                    stream_info.stream_id,
                    &stream_info.endpoint,
                    &stream_info.server_pubkey,
                    &client_secret,
                    &client_pubkey_bytes,
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        let _ = tx.send(ChatEvent::StreamError(e.to_string()));
                        return;
                    }
                };

                loop {
                    match handle.recv_next().await {
                        Ok(Some(StreamPayload::Data(b))) => {
                            let _ = tx.send(ChatEvent::Token(String::from_utf8_lossy(&b).into_owned()));
                        }
                        Ok(Some(StreamPayload::Complete(_))) | Ok(None) => {
                            let _ = tx.send(ChatEvent::StreamComplete);
                            break;
                        }
                        Ok(Some(StreamPayload::Error(m))) => {
                            let _ = tx.send(ChatEvent::StreamError(m));
                            break;
                        }
                        Err(e) => {
                            let _ = tx.send(ChatEvent::StreamError(e.to_string()));
                            break;
                        }
                    }
                }
            });
        });
    })
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
