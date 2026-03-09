//! Shell client handlers.
//!
//! Two modes:
//!
//! - `handle_shell_tui` — interactive local mode (`hyprstream` no args).
//!   Connects to TuiService, renders ratatui chrome via `AnsiBackend` to the
//!   local terminal, and routes input through `ShellClientState`.
//!
//! - `handle_tui_shell` — embedded mode (`hyprstream tui shell`).
//!   Sends `SpawnChromeShell` RPC to TuiService so ShellApp runs server-side
//!   as a `PaneProcess`.  No fork occurs in the client, avoiding the ZMQ
//!   signaler assertion that fired when forking after ZMQ initialisation.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use waxterm::backend::AnsiBackend;
use waxterm::input::InputParser;

use crate::services::generated::tui_client::TuiClient;
use crate::tui::shell_client::{ShellAction, ShellClientState, ModelEntry, WindowSummary, PaneSummary, spawn_shell_rpc, focus_window_rpc};
use crate::tui::shell_ui;

// ============================================================================
// handle_shell_tui — local interactive mode
// ============================================================================

/// Run ShellClient as a local interactive terminal.
///
/// - Connects to TuiService (session 0).
/// - Subscribes to the ANSI frame stream.
/// - Spawns a shell in the first pane (spawnShell RPC).
/// - Renders chrome + Vt via `AnsiBackend` to stdout.
/// - Routes raw stdin through `ShellClientState::handle_key`.
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
    let pane_rows = rows.saturating_sub(3); // 3 chrome rows

    // Connect to session 0 (attach or create)
    let result = client
        .connect(0, "ansi", cols, pane_rows)
        .await
        .context("Failed to connect to TUI session. Is the TUI service running?")?;

    let session_id = result.session_id;
    let viewer_id = result.viewer_id;

    // Auto-spawn shell in the first pane if there is one
    if let Some(win) = result.windows.first() {
        if let Some(pane) = win.panes.first() {
            let _ = spawn_shell_rpc(&client, session_id, pane.id, "").await;
        }
    }

    // Build initial WindowSummary list from connect result
    let windows: Vec<WindowSummary> = result
        .windows
        .iter()
        .map(|w| WindowSummary {
            id: w.id,
            name: w.name.clone(),
            active_pane_id: w.active_pane_id,
            panes: w.panes.iter().map(|p| PaneSummary {
                id: p.id,
                cols: p.cols,
                rows: p.rows,
            }).collect(),
        })
        .collect();

    // Fetch model list from registry service (exclusive enumeration via RPC).
    let models = {
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
                    let rmd = registry_models_dir.clone();
                    r.worktrees.into_iter().map(|wt| {
                        let branch = if wt.branch_name.is_empty() { "main".to_owned() } else { wt.branch_name };
                        let model_ref = format!("{}:{}", name, branch);
                        let loaded = *status_map.get(&model_ref).unwrap_or(&false);
                        let path = rmd.join(&name).join("worktrees").join(&branch);
                        ModelEntry { model_ref, path, loaded }
                    }).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
            Err(_) => vec![],
        }
    };

    use hyprstream_rpc::envelope::RequestIdentity;
    let model_client = crate::services::ModelZmqClient::new(
        signing_key.clone(),
        RequestIdentity::local(),
    );

    // Channel for background load-polling tasks to report confirmed status.
    let (model_status_tx, mut model_status_rx) =
        tokio::sync::mpsc::channel::<(String, bool)>(32);

    let mut state = ShellClientState::new(
        cols, pane_rows, session_id, viewer_id, windows, models, model_client,
        model_status_tx,
    );

    // Subscribe to ANSI frame stream (FD 1 = stdout)
    let stdout_stream = result.streams.get(1)
        .context("TuiService did not return a frame stream")?;

    let mac_key: [u8; 32] = stdout_stream
        .mac_key
        .as_slice()
        .try_into()
        .context("mac_key must be 32 bytes")?;
    let topic = stdout_stream.topic.clone();

    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);

    // ZMQ SUB in a blocking thread
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
                            if frame_tx.blocking_send(data).is_err() {
                                return;
                            }
                        }
                    }
                }
                Err(e) => tracing::warn!("Frame stream verification failed: {}", e),
            }
        }
    });

    // Stdin reader thread
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    std::thread::spawn(move || {
        use std::io::Read;
        let mut buf = [0u8; 256];
        loop {
            match std::io::stdin().lock().read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if stdin_tx.blocking_send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    // Set up ratatui terminal with AnsiBackend writing to stdout
    let writer = AnsiWriter(std::io::stdout());
    let backend = AnsiBackend::new(writer, cols, rows, false);
    let mut terminal = ratatui::Terminal::new(backend)
        .context("Failed to create ratatui terminal")?;
    let _ = terminal.hide_cursor();
    let _ = terminal.clear();

    let input_parser = InputParser::new(vec![]);
    let orig = enter_raw_mode()?;

    // Enable SGR mouse tracking (button events on click) for chrome interaction.
    {
        use std::io::Write;
        let _ = std::io::stdout().write_all(b"\x1b[?1002h\x1b[?1006h");
        let _ = std::io::stdout().flush();
    }

    let mut saw_ctrl_b = false;
    let mut win_refresh = tokio::time::interval(std::time::Duration::from_secs(2));
    win_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Background animation: ~20fps tick used when no windows are open or
    // the settings modal is showing.
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

    // Initial render
    let _ = terminal.draw(|f| shell_ui::draw(f, &state));

    loop {
        tokio::select! {
            Some(ansi_data) = frame_rx.recv() => {
                state.feed_frame(&ansi_data);
                let _ = terminal.draw(|f| shell_ui::draw(f, &state));
            }
            Some(raw) = stdin_rx.recv() => {
                // Check for ctrl-b prefix sequences (d = detach, q = quit).
                let mut should_exit = false;
                for &b in &raw {
                    if saw_ctrl_b && (b == b'd' || b == b'q') {
                        should_exit = true;
                        break;
                    }
                    saw_ctrl_b = b == 0x02;
                }
                if should_exit { break; }

                // Handle SGR mouse events (ESC [ < btn ; col ; row M/m).
                if let Some((col, row)) = parse_sgr_mouse_click(&raw) {
                    let win_strip_row = rows.saturating_sub(2);
                    if row == win_strip_row {
                        // Click on window strip — select tab by column position.
                        if let Some(idx) = tab_index_at_col(&state.windows, col) {
                            state.active_win = idx;
                            if let Some(w) = state.windows.get(idx) {
                                let _ = focus_window_rpc(&client, w.id).await;
                            }
                        }
                        let _ = terminal.draw(|f| shell_ui::draw(f, &state));
                        continue;
                    }
                    // Clicks on other chrome rows are swallowed (not forwarded).
                    let pane_rows = rows.saturating_sub(3);
                    let pane_top = 1u16; // status bar is row 0
                    if row < pane_top || row >= pane_top + pane_rows {
                        continue;
                    }
                    // Click inside the pane — forward as mouse event bytes.
                }

                // Parse raw bytes into keypresses
                let mut quit_from_key = false;
                let keys = input_parser.parse(&raw);
                for key in keys {
                    match state.handle_key(key, &client).await {
                        ShellAction::Quit => { quit_from_key = true; break; }
                        ShellAction::Redraw => {
                            let _ = terminal.draw(|f| shell_ui::draw(f, &state));
                        }
                        ShellAction::Forward(bytes) => {
                            send_input_rpc(&client, viewer_id, &bytes).await;
                        }
                        ShellAction::None => {}
                    }
                }
                if quit_from_key { break; }
            }
            _ = win_refresh.tick() => {
                if let Ok(wins) = list_windows_rpc(&client).await {
                    let changed = wins.len() != state.windows.len()
                        || wins.iter().zip(&state.windows).any(|(a, b)| a.id != b.id);
                    if changed {
                        state.active_win = state.active_win.min(wins.len().saturating_sub(1));
                        state.windows = wins;
                        let _ = terminal.draw(|f| shell_ui::draw(f, &state));
                    }
                }
            }
            _ = bg_tick.tick() => {
                if state.tick_bg() {
                    let _ = terminal.draw(|f| shell_ui::draw(f, &state));
                }
            }
            Some((model_ref, loaded)) = model_status_rx.recv() => {
                // Background poll confirmed model load/timeout — update list entry.
                for entry in state.model_list.items_mut() {
                    if entry.model_ref == model_ref {
                        entry.loaded = loaded;
                    }
                }
                if state.load_status.as_ref()
                    .is_some_and(|s| s.contains(&model_ref))
                {
                    state.load_status = None;
                }
                let _ = terminal.draw(|f| shell_ui::draw(f, &state));
            }
            else => break,
        }
    }

    // Disable mouse tracking before restoring terminal.
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
// handle_tui_shell — embedded mode (hyprstream tui shell)
// ============================================================================

/// Spawn ShellApp inside TuiService via SpawnChromeShell RPC.
///
/// The ShellApp runs server-side as a `PaneProcess` — no fork occurs in this
/// process after ZMQ is initialised, eliminating the ZMQ signaler assertion.
/// This process simply kicks off the RPC and then waits for SIGTERM so the
/// Playwright fixture can observe it as a long-lived process.
pub async fn handle_tui_shell(
    signing_key: &SigningKey,
    models_dir: &std::path::Path,
    session_id: u32,
) -> Result<()> {
    use crate::cli::tui_handlers::{create_tui_client, terminal_size};

    let (cols, rows) = terminal_size();
    let client = create_tui_client(signing_key);

    // Resolve the registry directory.
    let registry_dir = models_dir.join(".registry").join("models");
    let registry_dir = if registry_dir.exists() { registry_dir } else { models_dir.to_path_buf() };
    let registry_str = registry_dir.to_string_lossy().into_owned();

    // Ask TuiService to spawn ShellApp server-side.
    // Pass session_id=0 and pane_id=0 so the service auto-resolves the most
    // recent session's active pane — no viewer registration required here,
    // which prevents zombie viewer accumulation across tests.
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        let mut spawn = req.init_spawn_chrome_shell();
        spawn.set_session_id(session_id); // 0 = most recent session
        spawn.set_registry_dir(&registry_str);
        spawn.set_cols(cols);
        spawn.set_rows(rows);
        spawn.set_pane_id(0); // 0 = active pane in active window
    })?;
    client.call(payload).await.context("spawnChromeShell RPC failed")?;

    // Stay alive until SIGTERM (sent by the test fixture or the user).
    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?
        .recv()
        .await;

    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

/// Query the window list from TuiService (for tab bar refresh).
async fn list_windows_rpc(client: &TuiClient) -> anyhow::Result<Vec<WindowSummary>> {
    use crate::services::generated::tui_client::TuiResponseVariant;
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_list_windows(0); // session_id 0 = all
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
                panes: w.panes.iter().map(|p| PaneSummary {
                    id: p.id,
                    cols: p.cols,
                    rows: p.rows,
                }).collect(),
            })
            .collect()),
        _ => anyhow::bail!("unexpected response"),
    }
}

/// Send a resize RPC to TuiService.
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

/// Send raw bytes to TuiService via sendInput RPC.
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
            tracing::debug!("sendInput failed: {}", e);
        }
    }
}

// Thin newtype so AnsiBackend (which needs `Write`) can write to stdout.
struct AnsiWriter(std::io::Stdout);

impl std::io::Write for AnsiWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

// ============================================================================
// Mouse helpers
// ============================================================================

/// Parse an SGR mouse press event: `ESC [ < btn ; col ; row M`.
/// Returns `(col, row)` (0-indexed) if the data contains a press event.
fn parse_sgr_mouse_click(data: &[u8]) -> Option<(u16, u16)> {
    // Look for ESC [ < ... M sequence
    let mut i = 0;
    while i + 5 < data.len() {
        if data[i] == 0x1b && data[i + 1] == b'[' && data[i + 2] == b'<' {
            let rest = &data[i + 3..];
            // Parse "btn;col;row M" or "btn;col;row m"
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

/// Given a list of window summaries (tab order) and a column position,
/// return the tab index that was clicked, if any.
fn tab_index_at_col(windows: &[crate::tui::shell_client::WindowSummary], col: u16) -> Option<usize> {
    let mut x = 0u16;
    for (i, win) in windows.iter().enumerate() {
        // Tab label: " {name} " + 1 for the "│" divider (except last)
        let label_width = (win.name.len() as u16) + 2; // " name "
        let divider = if i + 1 < windows.len() { 1 } else { 0 };
        if col >= x && col < x + label_width {
            return Some(i);
        }
        x += label_width + divider;
    }
    None
}
