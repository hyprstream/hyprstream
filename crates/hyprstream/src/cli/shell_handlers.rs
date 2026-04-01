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
    Compositor, CompositorInput, CompositorOutput, ConversationKind,
    ConversationPickerEntry, GenerationDefaults, ModelEntry, PaneSummary, RpcRequest,
    ToastLevel, WindowSummary,
};

/// Status update from the model polling background task.
struct ModelStatusUpdate {
    model_ref: String,
    loaded: bool,
    gen_defaults: GenerationDefaults,
}
use hyprstream_compositor::layout::{CellUpdate, CursorState, FrameContent, FrameUpdate, ScrollUpdate};
use hyprstream_tui::chat_app::{ChatApp, LoadHook, SaveHook};
use hyprstream_tui::console_app::ConsoleApp;
use hyprstream_tui::container_app::ContainerTermApp;
use hyprstream_tui::private_store::{FsBackend, StorageKey};
use waxterm::app::TerminalApp;
use waxterm::input::KeyPress;

// ============================================================================
// ActiveApp — client-owned app variants (ChatApp or ContainerTermApp)
// ============================================================================

enum ActiveApp {
    Chat(Box<ChatApp>),
    Container(Box<ContainerTermApp>),
}

impl ActiveApp {
    fn tick(&mut self, delta_ms: u64) -> bool {
        match self {
            Self::Chat(a) => a.tick(delta_ms),
            Self::Container(a) => a.tick(delta_ms),
        }
    }

    fn should_quit(&self) -> bool {
        match self {
            Self::Chat(a) => a.should_quit(),
            Self::Container(a) => a.should_quit(),
        }
    }

    fn pending_toasts(&mut self) -> Vec<String> {
        match self {
            Self::Chat(a) => a.pending_toasts.drain(..).collect(),
            Self::Container(a) => a.pending_toasts.drain(..).collect(),
        }
    }

    fn cols(&self) -> u16 {
        match self {
            Self::Chat(a) => a.cols,
            Self::Container(a) => a.cols,
        }
    }

    fn rows(&self) -> u16 {
        match self {
            Self::Chat(a) => a.rows,
            Self::Container(a) => a.rows,
        }
    }

    fn render(&self, frame: &mut ratatui::Frame) {
        match self {
            Self::Chat(a) => a.render(frame),
            Self::Container(a) => a.render(frame),
        }
    }

    fn handle_input(&mut self, key: KeyPress) -> bool {
        match self {
            Self::Chat(a) => a.handle_input(key),
            Self::Container(a) => a.handle_input(key),
        }
    }

    fn as_chat_mut(&mut self) -> Option<&mut ChatApp> {
        match self {
            Self::Chat(a) => Some(a),
            Self::Container(_) => None,
        }
    }

}

use crate::services::generated::tui_client::{TuiClient, ConnectRequest, DisplayMode, SpawnChromeShellRequest, ResizeRequest, SendInputRequest};
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
    let pane_rows = rows.saturating_sub(4); // status(1) + top border(1) + bottom border(1) + strip(1)
    let pane_cols = cols.saturating_sub(2); // left + right border

    // Connect to session 0 using Capnp display mode so the frame stream
    // carries structured TuiFrame messages with pane_id for correct routing.
    let result = client
        .connect(&ConnectRequest { session_id: 0, display_mode: DisplayMode::Capnp, cols: pane_cols, rows: pane_rows })
        .await
        .context("Failed to connect to TUI session. Is the TUI service running?")?;

    let session_id = result.session_id;
    let viewer_id  = result.viewer_id;

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
        crate::services::generated::model_client::ModelClient::new(signing_key.clone(), RequestIdentity::anonymous())
    };
    // Worker client for sandbox/container/image management.
    let worker_client = {
        use hyprstream_rpc::envelope::RequestIdentity;
        hyprstream_workers::runtime::WorkerClient::new(signing_key.clone(), RequestIdentity::anonymous())
    };
    let (model_status_tx, mut model_status_rx) =
        tokio::sync::mpsc::channel::<ModelStatusUpdate>(32);

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
    sub_socket.set_rcvtimeo(1000)?; // 1s timeout so recv unblocks periodically for shutdown

    let _recv_handle = tokio::task::spawn_blocking(move || {
        let mut verifier = StreamVerifier::new(mac_key, topic);
        loop {
            let mut frames: Vec<Vec<u8>> = Vec::with_capacity(3);
            match sub_socket.recv_bytes(0) {
                Ok(f) => frames.push(f),
                Err(zmq::Error::EAGAIN) => {
                    // Timeout — check if the receiver is still alive.
                    if frame_tx.is_closed() { return; }
                    continue;
                }
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

    // Derive VFS subject from the signing key's public key — real identity, not anonymous.
    let vfs_subject = {
        let vk = signing_key.verifying_key();
        let pubkey_hex = hex::encode(vk.as_bytes());
        hyprstream_rpc::Subject::new(pubkey_hex)
    };

    // Build VFS namespace for `/path` routing in ChatApps.
    #[allow(clippy::type_complexity)]
    let (vfs_ns, tcl_mount_rx): (std::sync::Arc<hyprstream_vfs::Namespace>, std::sync::Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<hyprstream_tcl::TclCommand>>>) = {
        use crate::services::fs::{SyntheticNode, SyntheticTree};

        let mut ns = hyprstream_vfs::Namespace::new();

        // Mount the model service's 9P filesystem via RPC proxy.
        // RemoteModelMount translates sync Mount trait calls to async ModelClient
        // RPC requests, so `/srv/model/{model_ref}/status` etc. are served by the
        // model service's SyntheticTree on the other end of the ZMQ socket.
        let remote_model_mount = crate::services::remote_mount::RemoteModelMount::new(
            model_client.clone(),
        );
        let _ = ns.mount("/srv/model",
            std::sync::Arc::new(remote_model_mount),
        );

        // Wrap in Arc, then build /bin/ with Weak captures to avoid circular ref.
        // Weak doesn't block Arc::get_mut, so we can still mount after.
        let mut ns = std::sync::Arc::new(ns);
        let weak = std::sync::Arc::downgrade(&ns);
        let subj = vfs_subject.clone();

        let bin_tree = {
            let mut children = std::collections::HashMap::new();

            // /bin/cat — write path(s), read file contents.
            let w = weak.clone();
            let s = subj.clone();
            children.insert("cat".to_owned(), SyntheticNode::CtlFile {
                handler: Box::new(move |data, _subject| {
                    let ns = w.upgrade().ok_or("namespace dropped")?;
                    let input = std::str::from_utf8(data).map_err(|e| e.to_string())?;
                    let handle = tokio::runtime::Handle::current();
                    let mut out = Vec::new();
                    for path in input.split_whitespace() {
                        out.extend_from_slice(
                            &tokio::task::block_in_place(|| handle.block_on(ns.cat(path, &s))).map_err(|e| e.to_string())?,
                        );
                    }
                    Ok(out)
                }),
            });

            // /bin/ls — write path, read directory listing.
            let w = weak.clone();
            let s = subj.clone();
            children.insert("ls".to_owned(), SyntheticNode::CtlFile {
                handler: Box::new(move |data, _subject| {
                    let ns = w.upgrade().ok_or("namespace dropped")?;
                    let input = std::str::from_utf8(data).map_err(|e| e.to_string())?;
                    let path = input.trim();
                    let path = if path.is_empty() { "/" } else { path };
                    let handle = tokio::runtime::Handle::current();
                    let entries = tokio::task::block_in_place(|| handle.block_on(ns.ls(path, &s))).map_err(|e| e.to_string())?;
                    let listing: Vec<String> = entries.iter().map(|e| {
                        if e.is_dir { format!("{}/", e.name) } else { e.name.clone() }
                    }).collect();
                    Ok(listing.join("\n").into_bytes())
                }),
            });

            // /bin/help — read-only, lists available commands.
            let w = weak.clone();
            let s = subj.clone();
            children.insert("help".to_owned(), SyntheticNode::CtlFile {
                handler: Box::new(move |_data, _subject| {
                    let mut out = String::from("VFS commands:\n");
                    out.push_str("  cat <path>           read file contents\n");
                    out.push_str("  ls [path]            list directory\n");
                    out.push_str("  echo <path> <data>   write to file\n");
                    out.push_str("  ctl <path> <cmd>     control file (write+read)\n");
                    out.push_str("  mount [prefix]       list mount points\n");
                    out.push_str("  help                 this message\n");
                    if let Some(ns) = w.upgrade() {
                        let handle = tokio::runtime::Handle::current();
                        if let Ok(entries) = tokio::task::block_in_place(|| handle.block_on(ns.ls("/bin", &s))) {
                            if !entries.is_empty() {
                                out.push_str("\nCommands (/bin/):\n");
                                for entry in &entries {
                                    out.push_str(&format!("  {}\n", entry.name));
                                }
                            }
                        }
                    }
                    Ok(out.into_bytes())
                }),
            });

            // /bin/mount — read-only, lists mount prefixes.
            let w = weak.clone();
            children.insert("mount".to_owned(), SyntheticNode::CtlFile {
                handler: Box::new(move |_data, _subject| {
                    let ns = w.upgrade().ok_or("namespace dropped")?;
                    Ok(ns.mount_prefixes().join("\n").into_bytes())
                }),
            });

            SyntheticTree::new(SyntheticNode::Dir { children })
        };

        // /env/ — session variables as files (Plan9 per-process /env/).
        let env_store: std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<String, String>>> =
            std::sync::Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new()));

        let env_tree = {
            let store_list = env_store.clone();
            let store_resolve = env_store.clone();

            SyntheticTree::new(SyntheticNode::DynamicDir {
                list: Box::new(move || {
                    store_list.read().keys().map(|k| hyprstream_vfs::DirEntry {
                        name: k.clone(),
                        is_dir: false,
                        size: 0,
                        stat: None,
                    }).collect()
                }),
                resolve: Box::new(move |name| {
                    let store = store_resolve.clone();
                    let key = name.to_owned();
                    Some(SyntheticNode::ReadFile(Box::new(move || {
                        store.read().get(&key).cloned().unwrap_or_default().into_bytes()
                    })))
                }),
            })
        };

        // Create /lang/tcl mount channel — the TclMount exposes interpreter state
        // as files. The receiver is polled by ChatApp::tick().
        let (tcl_mount_tx, tcl_mount_rx) = hyprstream_tcl::create_mount_channel();
        let tcl_mount = std::sync::Arc::new(hyprstream_tcl::TclMount::new(tcl_mount_tx));

        // Mount /bin/, /env/, /lang/tcl — Arc::get_mut works because only Weak refs exist (no strong clones).
        if let Some(ns_mut) = std::sync::Arc::get_mut(&mut ns) {
            let _ = ns_mut.mount("/bin",
                std::sync::Arc::new(bin_tree),
            );

            // Discover .tcl tool scripts and bind them into /bin/ (After = fallback).
            let tools_dir = std::env::var("XDG_CONFIG_HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| {
                    dirs::home_dir().unwrap_or_default().join(".config")
                })
                .join("hyprstream/tools");
            let tools = discover_tools(&tools_dir);
            if !tools.is_empty() {
                let tools_tree = SyntheticTree::new(SyntheticNode::Dir { children: tools });
                let _ = ns_mut.bind_mount("/bin",
                    std::sync::Arc::new(tools_tree),
                    hyprstream_vfs::BindFlag::After,
                );
            }

            let _ = ns_mut.mount("/env",
                std::sync::Arc::new(env_tree),
            );
            let _ = ns_mut.mount("/lang/tcl", tcl_mount);
        }

        (ns, std::sync::Arc::new(tokio::sync::Mutex::new(tcl_mount_rx)))
    };

    // Console overlay for log viewing (F9 / Ctrl-L).
    let mut console_app = ConsoleApp::new();

    // Active client-owned apps (ChatApp or ContainerTermApp) keyed by pane_id.
    let mut active_apps: HashMap<u32, ActiveApp> = HashMap::new();
    // Local pane ID counter — high bit set to avoid collision with server IDs.
    let mut next_local_id: u32 = hyprstream_compositor::LOCAL_ID_BIT;

    let mut saw_ctrl_b = false;
    let mut win_refresh = tokio::time::interval(std::time::Duration::from_secs(2));
    win_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut bg_tick = tokio::time::interval(std::time::Duration::from_millis(50));
    bg_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut worker_refresh = tokio::time::interval(std::time::Duration::from_secs(5));
    worker_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

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
            // Send pane dimensions to server (full rows minus 4 chrome rows, cols minus 2 borders),
            // matching the initial connect() call which sends pane dimensions not terminal dimensions.
            let _ = resize_rpc(&resize_client, c.saturating_sub(2), r.saturating_sub(4)).await;
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
                    &mut compositor, &client, &model_client, &worker_client,
                    &model_status_tx, &mut terminal, &mut console_app,
                    &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                    &vfs_ns, &vfs_subject, &tcl_mount_rx, outputs,
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

                // SGR mouse click handling.
                if let Some((col, row)) = parse_sgr_mouse_click(&raw) {
                    // Close button on windows/modals — handled by compositor.
                    let close_outputs = compositor.handle(CompositorInput::MouseClick { col, row });
                    if !close_outputs.is_empty() {
                        if dispatch_outputs(
                            &mut compositor, &client, &model_client, &worker_client,
                            &model_status_tx, &mut terminal, &mut console_app,
                            &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                            &vfs_ns, &vfs_subject, &tcl_mount_rx, close_outputs,
                        ).await { break; }
                        continue;
                    }

                    // Window strip: [Ctrl-Space] label + tab clicks.
                    let win_strip_row = rows.saturating_sub(1);
                    if row == win_strip_row {
                        const STRIP_LABEL_W: u16 = 14;
                        if col < STRIP_LABEL_W {
                            // Click on [Ctrl-Space] label — open start menu.
                            let outs = compositor.chrome.handle_key(waxterm::input::KeyPress::CtrlSpace);
                            let mut should_redraw = false;
                            for co in outs {
                                if let hyprstream_compositor::ChromeOutput::Redraw = co {
                                    should_redraw = true;
                                }
                            }
                            if should_redraw {
                                composite_draw(&mut terminal, &compositor, &mut console_app);
                            }
                        } else if let Some(idx) = tab_index_at_col(&compositor.chrome.windows, col - STRIP_LABEL_W) {
                            compositor.chrome.active_win = idx;
                            if let Some(w) = compositor.chrome.windows.get(idx) {
                                let wid = w.id;
                                let _ = focus_window_rpc(&client, wid).await;
                            }
                            composite_draw(&mut terminal, &compositor, &mut console_app);
                        }
                        continue;
                    }

                    // Pass-through clicks in the pane area to the terminal.
                    let (pane_top, pane_rows) =
                        if matches!(compositor.chrome.mode, hyprstream_compositor::ShellMode::Fullscreen) {
                            (0u16, rows)
                        } else {
                            (2u16, rows.saturating_sub(4))
                        };
                    if row < pane_top || row >= pane_top + pane_rows {
                        continue;
                    }
                }

                let keys = input_parser.parse(&raw);
                for key in keys {
                    let outputs = compositor.handle(CompositorInput::KeyPress(key));
                    if dispatch_outputs(
                        &mut compositor, &client, &model_client, &worker_client,
                        &model_status_tx, &mut terminal, &mut console_app,
                        &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                        &vfs_ns, &vfs_subject, &tcl_mount_rx, outputs,
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
                        &mut compositor, &client, &model_client, &worker_client,
                        &model_status_tx, &mut terminal, &mut console_app,
                        &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                        &vfs_ns, &vfs_subject, &tcl_mount_rx, outputs,
                    ).await { break; }
                }
            }

            _ = worker_refresh.tick() => {
                if matches!(compositor.chrome.mode, hyprstream_compositor::ShellMode::WorkerManager { .. }) {
                    
                    // Poll sandboxes + containers
                    if let Ok(sandbox_infos) = worker_client.sandbox().list(&crate::services::worker::PodSandboxFilter::default()).await {
                        let mut entries = Vec::new();
                        for sb in &sandbox_infos {
                            let short_id = if sb.id.len() > 8 { sb.id[..8].to_owned() } else { sb.id.clone() };
                            let containers = match worker_client.container().list(&crate::services::worker::ContainerFilter {
                                pod_sandbox_id: sb.id.clone(),
                                ..Default::default()
                            }).await {
                                Ok(cis) => cis.into_iter().map(|c| {
                                    let c_short = if c.id.len() > 8 { c.id[..8].to_owned() } else { c.id.clone() };
                                    hyprstream_compositor::ContainerEntry {
                                        id: c_short,
                                        full_id: c.id,
                                        image: c.image.image,
                                        state: format!("{:?}", c.state),
                                        cpu_pct: None,
                                        mem_mb: None,
                                    }
                                }).collect(),
                                Err(_) => vec![],
                            };
                            entries.push(hyprstream_compositor::WorkerEntry {
                                id: short_id,
                                full_id: sb.id.clone(),
                                state: format!("{:?}", sb.state),
                                backend: sb.runtime_handler.clone(),
                                cpu_pct: None,
                                mem_mb: None,
                                containers,
                            });
                        }
                        let pool_summary = format!("{} sandboxes", entries.len());
                        let outputs = compositor.handle(CompositorInput::WorkerList { sandboxes: entries, pool_summary });
                        if dispatch_outputs(
                            &mut compositor, &client, &model_client, &worker_client,
                            &model_status_tx, &mut terminal, &mut console_app,
                            &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                            &vfs_ns, &vfs_subject, &tcl_mount_rx, outputs,
                        ).await { break; }
                    }
                    // Poll images
                    if let Ok(images) = worker_client.image().list(&crate::services::worker::ImageFilter { image: crate::services::worker::ImageSpec::default() }).await {
                        let entries: Vec<hyprstream_compositor::ImageEntry> = images.into_iter().map(|img| {
                            let repo_tag = img.repo_tags.first().cloned().unwrap_or_else(|| "<none>".to_owned());
                            let id = if img.id.len() > 12 { img.id[..12].to_owned() } else { img.id.clone() };
                            hyprstream_compositor::ImageEntry {
                                repo_tag,
                                id,
                                size_bytes: img.size,
                                created: format!("{}s ago", std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
                                    .saturating_sub(img.uid as u64)),
                            }
                        }).collect();
                        let outputs = compositor.handle(CompositorInput::WorkerImageList { images: entries });
                        if dispatch_outputs(
                            &mut compositor, &client, &model_client, &worker_client,
                            &model_status_tx, &mut terminal, &mut console_app,
                            &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                            &vfs_ns, &vfs_subject, &tcl_mount_rx, outputs,
                        ).await { break; }
                    }
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
                let mut pending_ctx_reloads: Vec<(String, usize)> = Vec::new();
                for (&pane_id, app) in active_apps.iter_mut() {
                    let dirty = app.tick(50);
                    // Surface inference errors as compositor toasts.
                    for msg in app.pending_toasts() {
                        compositor.chrome.push_toast(msg, ToastLevel::Error);
                    }
                    if app.should_quit() {
                        quit_app_ids.push(pane_id);
                        continue;
                    }
                    // Collect context-window reload requests from settings modal.
                    if let Some(chat) = app.as_chat_mut() {
                        if let Some(ctx) = chat.requested_context_window.take() {
                            pending_ctx_reloads.push((chat.model_name.clone(), ctx));
                        }
                    }
                    if dirty {
                        dirty_frames.push((pane_id, render_app_to_ansi(app)));
                    }
                }
                // Apply context-window reloads (model will reload with new context size).
                for (model_ref, ctx_window) in pending_ctx_reloads {
                    let reload_max_context = if ctx_window == 0 { None } else { Some(ctx_window as u32) };
                    let ctx_label = if ctx_window == 0 {
                        "model default".to_owned()
                    } else {
                        ctx_window.to_string()
                    };
                    compositor.chrome.push_toast(
                        format!("Reloading {model_ref} with context {ctx_label}…"),
                        ToastLevel::Info,
                    );
                    let _ = model_client.load(&crate::services::generated::model_client::LoadModelRequest {
                        model_ref: model_ref.clone(),
                        max_context: reload_max_context,
                        kv_quant: None,
                    }).await;
                    spawn_model_load_poll(model_client.clone(), model_ref.clone(), model_status_tx.clone());
                }
                // Feed dirty frames into compositor.
                for (pane_id, ansi) in dirty_frames {
                    let outputs = compositor.handle(CompositorInput::AppFrame { app_id: pane_id, ansi });
                    if dispatch_outputs(
                        &mut compositor, &client, &model_client, &worker_client,
                        &model_status_tx, &mut terminal, &mut console_app,
                        &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                        &vfs_ns, &vfs_subject, &tcl_mount_rx, outputs,
                    ).await { should_exit = true; break; }
                }
                // Remove quitting apps.
                for id in quit_app_ids {
                    // Fire-and-forget detach for container apps.
                    if let Some(ActiveApp::Container(ref c)) = active_apps.get(&id) {
                        let cid = c.container_id.clone();
                        let wc = worker_client.clone();
                        std::thread::spawn(move || {
                            let rt = tokio::runtime::Builder::new_current_thread()
                                .enable_all()
                                .build();
                            if let Ok(rt) = rt {
                                let _ = rt.block_on(wc.container().detach(&cid));
                            }
                        });
                    }
                    active_apps.remove(&id);
                    compositor.chrome.private_panes.remove(&id);
                    let outputs = compositor.handle(CompositorInput::AppExited { app_id: id });
                    dispatch_outputs(
                        &mut compositor, &client, &model_client, &worker_client,
                        &model_status_tx, &mut terminal, &mut console_app,
                        &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                        &vfs_ns, &vfs_subject, &tcl_mount_rx, outputs,
                    ).await;
                }
                if should_exit { break; }
            }

            Some(update) = model_status_rx.recv() => {
                let ModelStatusUpdate { model_ref, loaded, gen_defaults } = update;
                // Update gen_defaults on the model entry.
                for entry in compositor.chrome.model_list.items_mut() {
                    if entry.model_ref == model_ref {
                        entry.gen_defaults = gen_defaults.clone();
                    }
                }
                let follow_up = compositor.chrome.update_model_status(&model_ref, loaded);
                if !loaded {
                    compositor.chrome.push_toast(
                        format!("Model load timed out: {model_ref}"),
                        ToastLevel::Error,
                    );
                }
                // Dispatch follow-up RPC (e.g., ListConversations when pending picker resolves).
                if let Some(rpc_req) = follow_up {
                    dispatch_outputs(
                        &mut compositor, &client, &model_client, &worker_client,
                        &model_status_tx, &mut terminal, &mut console_app,
                        &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                        &vfs_ns, &vfs_subject, &tcl_mount_rx,
                        vec![CompositorOutput::Rpc(rpc_req)],
                    ).await;
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
                    &mut compositor, &client, &model_client, &worker_client,
                    &model_status_tx, &mut terminal, &mut console_app,
                    &mut active_apps, &mut next_local_id, &storage_key, signing_key,
                    &vfs_ns, &vfs_subject, &tcl_mount_rx, outputs,
                ).await { break; }
                composite_draw(&mut terminal, &compositor, &mut console_app);
            }

            else => break,
        }
    }

    // Abort background tasks that would otherwise block shutdown.
    _sigwinch_handle.abort();

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
    model_client: &crate::services::generated::model_client::ModelClient,
    worker_client: &hyprstream_workers::runtime::WorkerClient,
    model_status_tx: &tokio::sync::mpsc::Sender<ModelStatusUpdate>,
    terminal: &mut ratatui::Terminal<AnsiBackend<AnsiWriter>>,
    console_app: &mut ConsoleApp,
    active_apps: &mut HashMap<u32, ActiveApp>,
    next_local_id: &mut u32,
    storage_key: &StorageKey,
    signing_key: &SigningKey,
    vfs_ns: &std::sync::Arc<hyprstream_vfs::Namespace>,
    vfs_subject: &hyprstream_rpc::Subject,
    tcl_mount_rx: &std::sync::Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<hyprstream_tcl::TclCommand>>>,
    outputs: Vec<CompositorOutput>,
) -> bool {
    for output in outputs {
        match output {
            CompositorOutput::Redraw => {
                composite_draw(terminal, compositor, console_app);
            }
            CompositorOutput::Quit => return true,
            CompositorOutput::Rpc(req) => {
                let feed_back = handle_rpc(
                    compositor, client, model_client, worker_client, model_status_tx,
                    active_apps, next_local_id, storage_key, signing_key,
                    vfs_ns, vfs_subject, tcl_mount_rx, req,
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
                            let ansi = render_app_to_ansi(app);
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
    model_client: &crate::services::generated::model_client::ModelClient,
    worker_client: &hyprstream_workers::runtime::WorkerClient,
    model_status_tx: &tokio::sync::mpsc::Sender<ModelStatusUpdate>,
    active_apps: &mut HashMap<u32, ActiveApp>,
    next_local_id: &mut u32,
    storage_key: &StorageKey,
    signing_key: &SigningKey,
    vfs_ns: &std::sync::Arc<hyprstream_vfs::Namespace>,
    vfs_subject: &hyprstream_rpc::Subject,
    tcl_mount_rx: &std::sync::Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<hyprstream_tcl::TclCommand>>>,
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
            // Collect pane IDs before removing the window.
            let pane_ids: Vec<u32> = compositor.chrome.windows
                .iter()
                .find(|w| w.id == window_id)
                .map(|w| w.panes.iter().map(|p| p.id).collect())
                .unwrap_or_default();
            // Clean up client-owned ChatApps for any private panes in this window.
            for id in &pane_ids {
                active_apps.remove(id);
                compositor.chrome.private_panes.remove(id);
            }
            // Remove the window from chrome immediately so the tab disappears
            // and the background animation resumes without waiting for server confirmation.
            if let Some(win_idx) = compositor.chrome.windows.iter().position(|w| w.id == window_id) {
                compositor.chrome.windows.remove(win_idx);
                if compositor.chrome.windows.is_empty() {
                    compositor.chrome.active_win = 0;
                } else {
                    compositor.chrome.active_win =
                        compositor.chrome.active_win.min(compositor.chrome.windows.len() - 1);
                }
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
            let _ = model_client.load(&crate::services::generated::model_client::LoadModelRequest {
                model_ref: model_ref.clone(),
                max_context: None,
                kv_quant: None,
            }).await;
            spawn_model_load_poll(model_client.clone(), model_ref, model_status_tx.clone());
            vec![]
        }

        RpcRequest::UnloadModel { model_ref } => {
            let ok = model_client.unload(&crate::services::generated::model_client::UnloadModelRequest { model_ref: model_ref.clone() }).await.is_ok();
            if ok {
                compositor.chrome.update_model_status(&model_ref, false);
            }
            vec![]
        }

        RpcRequest::LocalPrivateChat { model_ref, cols, rows, resume_uuid, gen_defaults } => {
            // Allocate a local pane ID in the high-bit range to avoid
            // collision with server-allocated IDs (which start at 1).
            let pane_id = *next_local_id;
            *next_local_id = next_local_id.saturating_add(1);

            let session_uuid = match &resume_uuid {
                Some(s) => uuid::Uuid::parse_str(s).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                None => uuid::Uuid::new_v4(),
            };
            // Fetch tool list once; used by both the spawner (for chat template)
            // and the ChatApp (for dispatch + descriptions).
            let (tool_caller, tool_descriptions, openai_tools) =
                crate::tui::zmq_transport::make_tool_caller(signing_key);

            // Shared gen_config Arc — passed to both the spawner and the ChatApp.
            // Use model-specific defaults from RPC if available, otherwise fall back.
            let chat_gen_cfg = match gen_defaults {
                Some(gd) => {
                    let d = hyprstream_tui::chat_app::ChatGenConfig::default();
                    hyprstream_tui::chat_app::ChatGenConfig {
                        max_tokens: gd.max_tokens.unwrap_or(d.max_tokens),
                        temperature: gd.temperature.unwrap_or(d.temperature),
                        top_p: gd.top_p.unwrap_or(d.top_p),
                        top_k: gd.top_k.or(d.top_k),
                        context_window: gd.context_window,
                    }
                }
                None => hyprstream_tui::chat_app::ChatGenConfig::default(),
            };
            let gen_config = std::sync::Arc::new(parking_lot::RwLock::new(chat_gen_cfg));

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
                    let spawner = crate::tui::zmq_transport::make_chat_spawner(
                        signing_key, &model_ref, Some(openai_tools.clone()),
                        gen_config.clone(),
                    );
                    ChatApp::new_private(
                        model_ref.clone(), cols, rows, spawner,
                        session_uuid, load_hook, save_hook,
                    )
                    .with_gen_config(gen_config)
                    .with_vfs(vfs_ns.clone(), vfs_subject.clone())
                    .with_tcl_mount_rx(tcl_mount_rx.clone())
                }
                Err(e) => {
                    compositor.chrome.push_toast(
                        format!("Private store dir unavailable: {e}"),
                        ToastLevel::Warn,
                    );
                    let spawner = crate::tui::zmq_transport::make_chat_spawner(
                        signing_key, &model_ref, Some(openai_tools.clone()),
                        gen_config.clone(),
                    );
                    ChatApp::new(model_ref.clone(), cols, rows, spawner)
                        .with_gen_config(gen_config)
                        .with_vfs(vfs_ns.clone(), vfs_subject.clone())
                    .with_tcl_mount_rx(tcl_mount_rx.clone())
                }
            };
            // Detect tool call format from the model reference name.
            let tool_format = {
                use crate::api::tools::ToolCallFormat;
                match ToolCallFormat::from_model_ref(&model_ref) {
                    ToolCallFormat::Qwen3Xml | ToolCallFormat::None => hyprstream_tui::chat_app::ToolCallFormat::Qwen3Xml,
                    ToolCallFormat::Qwen35XmlParam => hyprstream_tui::chat_app::ToolCallFormat::Qwen35XmlParam,
                    ToolCallFormat::LlamaJson     => hyprstream_tui::chat_app::ToolCallFormat::LlamaJson,
                    ToolCallFormat::MistralJson   => hyprstream_tui::chat_app::ToolCallFormat::MistralJson,
                }
            };
            let app = app.with_tool_caller(
                tool_caller,
                tool_descriptions,
                tool_format,
            );
            active_apps.insert(pane_id, ActiveApp::Chat(Box::new(app)));
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
                let ansi = render_app_to_ansi(a);
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
                            kind: ConversationKind::Resume { uuid: meta.uuid.to_string() },
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

        RpcRequest::ServiceStart { name: _ }
        | RpcRequest::ServiceStop { name: _ }
        | RpcRequest::ServiceRestart { name: _ }
        | RpcRequest::ServiceInstall
        | RpcRequest::ServiceStopAll
        | RpcRequest::ServiceStartAll => {
            compositor.chrome.push_toast(
                "Service management not yet available (requires ZMQ service RPC)".to_owned(),
                ToastLevel::Warn,
            );
            vec![]
        }
        RpcRequest::WorkerDestroySandbox { sandbox_id } => {
            let short = &sandbox_id[..sandbox_id.len().min(8)];
            compositor.chrome.push_toast(format!("Destroying {short}..."), ToastLevel::Info);
            match worker_client.sandbox().stop(&sandbox_id).await {
                Ok(()) => {
                    let _ = worker_client.sandbox().remove(&sandbox_id).await;
                    compositor.chrome.push_toast(format!("Destroyed {short}"), ToastLevel::Info);
                }
                Err(e) => {
                    compositor.chrome.push_toast(format!("Destroy failed: {e}"), ToastLevel::Error);
                }
            }
            vec![]
        }
        RpcRequest::WorkerExecSync { sandbox_id: _, container_id, cmd } => {
            let short = &container_id[..container_id.len().min(8)];
            compositor.chrome.push_toast(format!("Exec in {short}: {:?}", cmd), ToastLevel::Info);
            match worker_client.container().exec(&crate::services::worker::ExecSyncRequest {
                container_id: container_id.clone(),
                cmd: cmd.clone(),
                timeout: 30,
            }).await {
                Ok(result) => {
                    let stdout = String::from_utf8_lossy(&result.stdout);
                    let msg = if stdout.len() > 60 { format!("{}…", &stdout[..60]) } else { stdout.to_string() };
                    compositor.chrome.push_toast(format!("exec: {msg}"), ToastLevel::Info);
                }
                Err(e) => {
                    compositor.chrome.push_toast(format!("exec failed: {e}"), ToastLevel::Error);
                }
            }
            vec![]
        }
        RpcRequest::WorkerImageList => {
            match worker_client.image().list(&crate::services::worker::ImageFilter { image: crate::services::worker::ImageSpec::default() }).await {
                Ok(images) => {
                    let entries: Vec<hyprstream_compositor::ImageEntry> = images.into_iter().map(|img| {
                        let repo_tag = img.repo_tags.first().cloned().unwrap_or_else(|| "<none>".to_owned());
                        let id = if img.id.len() > 12 { img.id[..12].to_owned() } else { img.id.clone() };
                        hyprstream_compositor::ImageEntry {
                            repo_tag,
                            id,
                            size_bytes: img.size,
                            created: format!("{}s ago", std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
                                .saturating_sub(img.uid as u64)),
                        }
                    }).collect();
                    vec![CompositorInput::WorkerImageList { images: entries }]
                }
                Err(e) => {
                    compositor.chrome.push_toast(format!("Image list failed: {e}"), ToastLevel::Warn);
                    vec![]
                }
            }
        }
        RpcRequest::WorkerImagePull { image } => {
            use crate::services::worker::ImageSpec;
            compositor.chrome.push_toast(format!("Pulling {image}..."), ToastLevel::Info);
            let spec = ImageSpec { image: image.clone(), annotations: Default::default(), runtime_handler: Default::default() };
            match worker_client.image().pull(&crate::services::worker::PullImageRequest {
                image: spec,
                auth: crate::services::worker::AuthConfig::default(),
                sandbox_config: crate::services::worker::PodSandboxConfig::default(),
            }).await {
                Ok(ref_id) => {
                    compositor.chrome.push_toast(format!("Pulled: {}", &ref_id[..ref_id.len().min(20)]), ToastLevel::Info);
                }
                Err(e) => {
                    compositor.chrome.push_toast(format!("Pull failed: {e}"), ToastLevel::Error);
                }
            }
            vec![]
        }
        RpcRequest::WorkerImageRemove { image } => {
            use crate::services::worker::ImageSpec;
            let spec = ImageSpec { image: image.clone(), annotations: Default::default(), runtime_handler: Default::default() };
            match worker_client.image().remove(&spec).await {
                Ok(()) => {
                    compositor.chrome.push_toast(format!("Removed {image}"), ToastLevel::Info);
                }
                Err(e) => {
                    compositor.chrome.push_toast(format!("Remove failed: {e}"), ToastLevel::Error);
                }
            }
            vec![]
        }
        RpcRequest::WorkerCreateSandbox { hostname, backend, gpu } => {
            use crate::services::worker::{PodSandboxConfig, KeyValue};
            let mut annotations = vec![];
            annotations.push(KeyValue { key: "io.hyprstream/runtime-handler".to_owned(), value: backend });
            if gpu {
                annotations.push(KeyValue { key: "hyprstream.io/gpu".to_owned(), value: "true".to_owned() });
            }
            let config = PodSandboxConfig {
                hostname,
                annotations,
                ..Default::default()
            };
            match worker_client.sandbox().run(&config).await {
                Ok(sandbox_id) => {
                    let short = &sandbox_id[..sandbox_id.len().min(8)];
                    compositor.chrome.push_toast(format!("Created sandbox {short}"), ToastLevel::Info);
                }
                Err(e) => {
                    compositor.chrome.push_toast(format!("Create sandbox failed: {e}"), ToastLevel::Error);
                }
            }
            vec![]
        }
        RpcRequest::WorkerCreateContainer { sandbox_id, image, cmd, tty } => {
            use crate::services::worker::{ContainerConfig, ImageSpec, PodSandboxConfig};
            let short_sb = &sandbox_id[..sandbox_id.len().min(8)];
            // PodSandboxStatusResponse has no config field — use default
            let sandbox_config = PodSandboxConfig::default();
            let container_config = ContainerConfig {
                image: ImageSpec { image: image.clone(), annotations: Default::default(), runtime_handler: Default::default() },
                command: cmd,
                tty,
                ..Default::default()
            };
            match worker_client.container().create(&crate::services::worker::CreateContainerRequest {
                pod_sandbox_id: sandbox_id.clone(),
                config: container_config,
                sandbox_config,
            }).await {
                Ok(container_id) => {
                    let short = &container_id[..container_id.len().min(8)];
                    compositor.chrome.push_toast(format!("Created container {short} in {short_sb}"), ToastLevel::Info);
                    // Auto-start
                    if let Err(e) = worker_client.container().start(&container_id).await {
                        compositor.chrome.push_toast(format!("Start failed: {e}"), ToastLevel::Error);
                    }
                }
                Err(e) => {
                    compositor.chrome.push_toast(format!("Create container failed: {e}"), ToastLevel::Error);
                }
            }
            vec![]
        }
        RpcRequest::WorkerStartContainer { container_id } => {
            let short = &container_id[..container_id.len().min(8)];
            match worker_client.container().start(&container_id).await {
                Ok(()) => compositor.chrome.push_toast(format!("Started {short}"), ToastLevel::Info),
                Err(e) => compositor.chrome.push_toast(format!("Start failed: {e}"), ToastLevel::Error),
            }
            vec![]
        }
        RpcRequest::WorkerStopContainer { container_id } => {
            let short = &container_id[..container_id.len().min(8)];
            match worker_client.container().stop(&crate::services::worker::StopContainerRequest {
                container_id: container_id.clone(),
                timeout: 10,
            }).await {
                Ok(()) => compositor.chrome.push_toast(format!("Stopped {short}"), ToastLevel::Info),
                Err(e) => compositor.chrome.push_toast(format!("Stop failed: {e}"), ToastLevel::Error),
            }
            vec![]
        }
        RpcRequest::WorkerRemoveContainer { container_id } => {
            let short = &container_id[..container_id.len().min(8)];
            match worker_client.container().remove(&container_id).await {
                Ok(()) => compositor.chrome.push_toast(format!("Removed {short}"), ToastLevel::Info),
                Err(e) => compositor.chrome.push_toast(format!("Remove failed: {e}"), ToastLevel::Error),
            }
            vec![]
        }
        RpcRequest::AttachContainer { sandbox_id, container_id, cols, rows } => {
            use hyprstream_rpc::streaming::StreamPayload;

            let short_c = container_id[..container_id.len().min(8)].to_owned();
            let short_s = sandbox_id[..sandbox_id.len().min(8)].to_owned();

            // Attach RPC → returns ready-to-use StreamHandle (DH + SUB + HMAC managed internally).
            use hyprstream_workers::generated::worker_client::ContainerRpc;
            let mut stream_handle = match ContainerRpc::attach(&worker_client.container(), &crate::services::worker::AttachRequest {
                container_id: container_id.clone(),
                fds: vec![],
            }).await {
                Ok(h) => h,
                Err(e) => {
                    compositor.chrome.push_toast(
                        format!("Attach failed: {e}"),
                        ToastLevel::Error,
                    );
                    return vec![];
                }
            };

            // Channel: stream reader thread → ContainerTermApp VtPane.
            let (stdout_tx, stdout_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(64);

            // Spawn a thread with its own Tokio runtime to pump the stream.
            std::thread::spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(r) => r,
                    Err(_) => return,
                };
                rt.block_on(async move {
                    while let Ok(Some(StreamPayload::Data(bytes))) = stream_handle.recv_next().await {
                        if stdout_tx.send(bytes).is_err() {
                            break; // ContainerTermApp dropped
                        }
                    }
                });
            });

            // Create ContainerTermApp with the stdout channel.
            let app = ContainerTermApp::new(
                container_id.clone(),
                sandbox_id.clone(),
                cols,
                rows,
                stdout_rx,
                None, // stdin not wired yet (Phase W-III-b)
            );

            // Allocate local pane ID.
            let pane_id = *next_local_id;
            *next_local_id = next_local_id.saturating_add(1);

            active_apps.insert(pane_id, ActiveApp::Container(Box::new(app)));
            compositor.chrome.private_panes.insert(pane_id);
            compositor.layout.get_or_create_private(pane_id);

            // Insert phantom window for the container terminal.
            let win_name = format!("{short_s}:{short_c}");
            let win_summary = WindowSummary {
                id: pane_id,
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

            // Close the worker modal.
            compositor.chrome.mode = hyprstream_compositor::chrome::ShellMode::Normal;

            // Render initial frame.
            if let Some(a) = active_apps.get(&pane_id) {
                let ansi = render_app_to_ansi(a);
                compositor.handle(CompositorInput::AppFrame { app_id: pane_id, ansi });
            }

            compositor.chrome.push_toast(
                format!("Attached to {short_c}"),
                ToastLevel::Info,
            );
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

    client.spawn_chrome_shell(&SpawnChromeShellRequest {
        session_id,
        registry_dir: registry_str,
        cols,
        rows,
        pane_id: 0,
    }).await.context("spawnChromeShell RPC failed")?;

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
    let registry: crate::services::RegistryClient = crate::services::RegistryClient::with_endpoint(
        &registry_endpoint,
        signing_key.clone(),
        RequestIdentity::anonymous(),
    );
    let model_client_for_status = crate::services::generated::model_client::ModelClient::new(
        signing_key.clone(),
        RequestIdentity::anonymous(),
    );
    let registry_models_dir = models_dir.to_path_buf();
    let status_timeout = std::time::Duration::from_millis(500);

    let all_status_req = crate::services::generated::model_client::StatusRequest { model_ref: String::new() };
    let (repos_result, status_result) = tokio::join!(
        registry.list(),
        tokio::time::timeout(status_timeout, model_client_for_status.status(&all_status_req)),
    );

    let status_map: std::collections::HashMap<String, (bool, GenerationDefaults)> = match status_result {
        Ok(Ok(entries)) => entries.into_iter()
            .map(|e| {
                let loaded = e.status == "loaded";
                let gd = wire_gen_defaults_to_compositor(&e.generation_defaults);
                (e.model_ref, (loaded, gd))
            })
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
                    ModelEntry { model_ref, path, loaded: false, loading: false, gen_defaults: Default::default() }
                })
                .collect::<Vec<_>>()
            })
            .map(|mut entry| {
                if let Some((loaded, gd)) = status_map.get(&entry.model_ref) {
                    entry.loaded = *loaded;
                    entry.gen_defaults = gd.clone();
                }
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
    let list = client.list_windows(0).await?;
    Ok(list
        .windows
        .iter()
        .map(|w| WindowSummary {
            id: w.id,
            name: w.name.clone(),
            active_pane_id: w.active_pane_id,
            panes: w.panes.iter().map(|p| PaneSummary { id: p.id, cols: p.cols, rows: p.rows, is_private: p.is_private }).collect(),
        })
        .collect())
}

async fn resize_rpc(client: &TuiClient, cols: u16, rows: u16) -> anyhow::Result<()> {
    client.resize(&ResizeRequest { cols, rows }).await
}

async fn send_input_rpc(client: &TuiClient, viewer_id: u32, data: &[u8]) {
    if let Err(e) = client.send_input(&SendInputRequest {
        viewer_id,
        data: data.to_vec(),
    }).await {
        tracing::debug!("sendInput failed: {e}");
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



/// Convert the wire GenerationDefaults (7 fields) to the compositor's
/// GenerationDefaults (5 fields).
///
/// Intentionally dropped from the wire type:
///   - `repeat_penalty`: not exposed in the pre-chat settings modal
///   - `stop_tokens`: handled at the engine layer, not user-configurable
///   - `do_sample`: inferred from temperature > 0
///
/// `context_window` is set to `None` because it is a load-time parameter
/// configured via `LoadModelRequest.maxContext`, not a generation default.
fn wire_gen_defaults_to_compositor(
    wire: &crate::services::generated::model_client::GenerationDefaults,
) -> GenerationDefaults {
    GenerationDefaults {
        temperature: wire.temperature,
        top_p: wire.top_p,
        top_k: wire.top_k.map(|v| v as usize),
        max_tokens: wire.max_tokens.map(|v| v as usize),
        context_window: None,
    }
}

/// Spawn a background task that polls model status every 2s until loaded (or timeout).
/// Sends a `ModelStatusUpdate` through `tx` when the model loads or after 120s timeout.
fn spawn_model_load_poll(
    poll_client: crate::services::generated::model_client::ModelClient,
    model_ref: String,
    tx: tokio::sync::mpsc::Sender<ModelStatusUpdate>,
) {
    tokio::spawn(async move {
        for _ in 0..60u32 {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            if let Ok(entries) = poll_client.status(
                &crate::services::generated::model_client::StatusRequest {
                    model_ref: model_ref.clone(),
                },
            ).await {
                if let Some(entry) = entries.iter().find(|e| e.status == "loaded") {
                    let gd = wire_gen_defaults_to_compositor(&entry.generation_defaults);
                    let _ = tx.send(ModelStatusUpdate {
                        model_ref, loaded: true, gen_defaults: gd,
                    }).await;
                    return;
                }
            }
        }
        let _ = tx.send(ModelStatusUpdate {
            model_ref, loaded: false, gen_defaults: GenerationDefaults::default(),
        }).await;
    });
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

/// Render an `ActiveApp` off-screen to an ANSI byte buffer.
fn render_app_to_ansi(app: &ActiveApp) -> Vec<u8> {
    let backend = AnsiBackend::new(Vec::<u8>::new(), app.cols(), app.rows(), false);
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

// ─────────────────────────────────────────────────────────────────────────────
// Tool discovery from .tcl files
// ─────────────────────────────────────────────────────────────────────────────

/// Scan a directory for `.tcl` scripts and return SyntheticNode entries.
///
/// Each `.tcl` file becomes a CtlFile whose handler evaluates the script
/// contents with the written data available as `$args`. The file stem
/// (without `.tcl` extension) is used as the tool name.
///
/// Returns a HashMap suitable for constructing a `SyntheticNode::Dir`.
/// Returns an empty map if the directory doesn't exist or is unreadable.
fn discover_tools(dir: &std::path::Path) -> HashMap<String, crate::services::fs::SyntheticNode> {
    use crate::services::fs::SyntheticNode;

    let mut tools = HashMap::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return tools,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("tcl") {
            continue;
        }
        let name = match path.file_stem().and_then(|s| s.to_str()) {
            Some(n) => n.to_owned(),
            None => continue,
        };
        // Reject names with path separators or dots (safety).
        if name.contains('/') || name.contains("..") || name.is_empty() {
            continue;
        }

        let script_path = path.clone();
        tools.insert(name, SyntheticNode::CtlFile {
            handler: Box::new(move |data, _subject| {
                let script = std::fs::read_to_string(&script_path)
                    .map_err(|e| format!("failed to read tool script: {e}"))?;
                let args = String::from_utf8_lossy(data);
                // Wrap the script: set $args before executing, capture the result.
                let wrapped = format!("set args {{{}}}\n{}", args.trim(), script);
                // Return the wrapped script for evaluation by the caller's TclShell.
                // The ctl response is the script to eval — the /bin/ fallback in
                // TclShell.try_cmd_resolve() will pass this through eval.
                Ok(wrapped.into_bytes())
            }),
        });
    }

    tools
}
