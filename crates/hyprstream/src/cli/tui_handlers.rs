//! CLI handlers for TUI display server commands.
//!
//! Supports: attach, new, list, detach, play.
//!
//! The `attach` command connects to the TUI service, enters raw terminal mode,
//! and forwards ANSI frames to stdout / stdin to the session. Other commands
//! send single RPC requests and print the result.
//!
//! Uses the generated `TuiClient` from `services/generated.rs` for typed RPC
//! where codegen covers the method, and raw Cap'n Proto for UInt32 variants
//! (list_windows, disconnect, etc.) that codegen doesn't generate client methods for.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use hyprstream_rpc::prelude::*;
use std::io::Write;
use tracing::info;

use crate::services::generated::tui_client::{TuiClient, TuiResponseVariant};

/// Create a TuiClient for RPC calls.
///
/// Resolves the TUI service endpoint from the registry and creates a client
/// with the given signing key and local identity.
fn create_tui_client(signing_key: &SigningKey) -> TuiClient {
    use hyprstream_rpc::registry::{global as registry, SocketKind};

    let endpoint = registry().endpoint("tui", SocketKind::Rep).to_zmq_string();
    crate::services::create_service_client(&endpoint, signing_key.clone(), RequestIdentity::local())
}

/// Attach to an existing TUI session.
///
/// Connects to the TUI service via ZMQ, sends a ConnectRequest,
/// subscribes to the frame stream, enters raw mode, and proxies
/// ANSI frames to stdout and stdin to the session.
pub async fn handle_tui_attach(signing_key: &SigningKey, session_id: Option<u32>) -> Result<()> {
    let sid = session_id.unwrap_or(0);
    info!(session_id = sid, "Attaching to TUI session");

    let client = create_tui_client(signing_key);
    let (cols, rows) = terminal_size();

    // Connect to session via generated TuiClient
    let result = client.connect(sid, "ansi", cols, rows).await
        .context("Failed to connect to TUI session. Is the TUI service running?")?;

    println!("Connected to session {} (viewer {})", result.session_id, result.viewer_id);

    // FD-indexed streams: [0]=stdin (input relay), [1]=stdout (frames)
    let stdout_stream = result.streams.get(1);
    if let Some(si) = stdout_stream {
        if !si.topic.is_empty() {
            info!(
                topic = %si.topic,
                endpoint = %si.sub_endpoint,
                "Stream info received"
            );
        }
    }

    // Print window list
    for win in &result.windows {
        println!("  Window {}: {} ({} panes)", win.id, win.name, win.panes.len());
    }

    // Subscribe to frame stream and enter raw mode
    if let Some(stream_info) = stdout_stream {
        if !stream_info.topic.is_empty() {
            run_attach_loop(stream_info, &client, result.viewer_id).await?;
        } else {
            println!("No stream info — session may be inactive.");
        }
    } else {
        println!("No stream info — session may be inactive.");
    }

    Ok(())
}

/// Run the attach loop: subscribe to frames, enter raw mode, proxy I/O.
///
/// Uses `StreamVerifier` for Blake3 HMAC chain verification of incoming
/// StreamBlock messages. The verifier checks that each block's MAC is
/// valid relative to the previous block, ensuring no reordering or tampering.
async fn run_attach_loop(
    stream_info: &crate::services::generated::tui_client::StreamInfo,
    client: &TuiClient,
    viewer_id: u32,
) -> Result<()> {
    use hyprstream_rpc::registry::{global as endpoint_registry, SocketKind};
    use hyprstream_rpc::streaming::{StreamPayload, StreamVerifier};

    let zmq_ctx = zmq::Context::new();

    // Connect directly via ZMQ SUB to the XPUB endpoint
    let sub_endpoint = endpoint_registry()
        .endpoint("streams", SocketKind::Sub)
        .to_zmq_string();

    let sub_socket = zmq_ctx.socket(zmq::SUB)?;
    sub_socket.connect(&sub_endpoint)?;
    sub_socket.set_subscribe(stream_info.topic.as_bytes())?;
    sub_socket.set_linger(0)?;

    // Create verifier with the mac_key from ConnectResult.stream_info
    let mac_key: [u8; 32] = stream_info.mac_key.as_slice()
        .try_into()
        .context("mac_key must be 32 bytes")?;
    let topic = stream_info.topic.clone();

    // Enter raw terminal mode
    let orig_termios = enter_raw_mode()?;
    println!("\r\nAttached to session (ctrl-b d to detach)\r");

    let mut stdout = std::io::stdout();

    // ZMQ recv loop in spawn_blocking — collects full multipart [topic, capnp, mac]
    // and verifies via StreamVerifier before sending ANSI data
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);

    let recv_handle = tokio::task::spawn_blocking(move || {
        let mut verifier = StreamVerifier::new(mac_key, topic);

        loop {
            // Receive full multipart message
            let mut frames: Vec<Vec<u8>> = Vec::with_capacity(3);
            match sub_socket.recv_bytes(0) {
                Ok(frame) => frames.push(frame),
                Err(_) => break,
            }
            while sub_socket.get_rcvmore().unwrap_or(false) {
                match sub_socket.recv_bytes(0) {
                    Ok(frame) => frames.push(frame),
                    Err(_) => break,
                }
            }

            // Verify HMAC chain and extract payloads
            match verifier.verify(&frames) {
                Ok(payloads) => {
                    for payload in payloads {
                        if let StreamPayload::Data(data) = payload {
                            if tx.blocking_send(data).is_err() {
                                return;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Stream verification failed: {}", e);
                    // Continue — transient errors (e.g. first block) shouldn't kill attach
                }
            }
        }
    });

    // Read stdin in a separate task
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    let _stdin_handle = tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut buf = [0u8; 256];
        let stdin = std::io::stdin();
        loop {
            match stdin.lock().read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if stdin_tx.blocking_send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    // Detach sequence: ctrl-b followed by 'd'
    let mut saw_ctrl_b = false;

    loop {
        tokio::select! {
            Some(ansi_data) = rx.recv() => {
                // Data is already verified ANSI bytes extracted from StreamBlock payloads
                let _ = stdout.write_all(&ansi_data);
                let _ = stdout.flush();
            }
            Some(input) = stdin_rx.recv() => {
                // Check for detach sequence (ctrl-b d)
                let mut detach = false;
                for &byte in &input {
                    if saw_ctrl_b && byte == b'd' {
                        detach = true;
                        break;
                    }
                    saw_ctrl_b = byte == 0x02; // ctrl-b
                }
                if detach {
                    restore_terminal(&orig_termios);
                    recv_handle.abort();
                    println!("\r\nDetached from session.");
                    return Ok(());
                }

                // Forward input to TUI service via sendInput RPC
                let request_id = client.next_id();
                let payload = hyprstream_rpc::serialize_message(|msg| {
                    let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
                    req.set_id(request_id);
                    let mut send_input = req.init_send_input();
                    send_input.set_viewer_id(viewer_id);
                    send_input.set_data(&input);
                });
                if let Ok(payload) = payload {
                    if let Err(e) = client.call(payload).await {
                        tracing::debug!("sendInput failed: {}", e);
                    }
                }
            }
            else => break,
        }
    }

    restore_terminal(&orig_termios);
    Ok(())
}

/// Create a new TUI session.
pub async fn handle_tui_new(signing_key: &SigningKey) -> Result<()> {
    info!("Creating new TUI session");

    let client = create_tui_client(signing_key);
    let window_info = client.create_window().await
        .context("Failed to create window. Is the TUI service running?")?;

    println!("Created window {}:", window_info.id);
    println!("  Name: {}", window_info.name);
    println!("  Active pane: {}", window_info.active_pane_id);
    for pane in &window_info.panes {
        println!("  Pane {}: {}x{} \"{}\"", pane.id, pane.cols, pane.rows, pane.title);
    }

    Ok(())
}

/// List active TUI sessions.
pub async fn handle_tui_list(signing_key: &SigningKey) -> Result<()> {
    info!("Listing TUI sessions");

    let client = create_tui_client(signing_key);

    // list_windows takes a UInt32 (session_id), but codegen doesn't generate a
    // typed client method for UInt32 variants. Build the request manually.
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_list_windows(0); // session_id 0 = all
    })?;
    let response = client.call(payload).await
        .context("Failed to list windows. Is the TUI service running?")?;

    // Parse the response
    match TuiClient::parse_response(&response)? {
        TuiResponseVariant::ListWindowsResult(window_list) => {
            if window_list.windows.is_empty() {
                println!("No active sessions.");
                println!("\nCreate one with: hyprstream tui new");
            } else {
                println!("┌─────────────────────────────────────────────────┐");
                println!("│ TUI Sessions                                    │");
                println!("├──────┬──────────────────┬───────┬───────────────┤");
                println!("│ ID   │ Name             │ Panes │ Active Pane   │");
                println!("├──────┼──────────────────┼───────┼───────────────┤");
                for win in &window_list.windows {
                    println!(
                        "│ {:4} │ {:16} │ {:5} │ {:13} │",
                        win.id,
                        truncate_str(&win.name, 16),
                        win.panes.len(),
                        win.active_pane_id,
                    );
                }
                println!("└──────┴──────────────────┴───────┴───────────────┘");
            }
        }
        TuiResponseVariant::Error(e) => {
            eprintln!("Error: {} (code: {})", e.message, e.code);
            if !e.details.is_empty() {
                eprintln!("  Details: {}", e.details);
            }
        }
        other => {
            eprintln!("Unexpected response: {:?}", other);
        }
    }

    Ok(())
}

/// Detach from current TUI session.
pub async fn handle_tui_detach() -> Result<()> {
    info!("Detaching from TUI session");
    // Detach is a no-op when not attached — the attach loop handles cleanup
    println!("Detached.");
    Ok(())
}

/// Get the terminal size, defaulting to 80x24.
fn terminal_size() -> (u16, u16) {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = std::io::stdout().as_raw_fd();
        let mut ws = libc::winsize {
            ws_row: 0,
            ws_col: 0,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        if unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) } == 0
            && ws.ws_col > 0
            && ws.ws_row > 0
        {
            return (ws.ws_col, ws.ws_row);
        }
    }
    (80, 24)
}

/// Enter raw terminal mode. Returns the original termios for restoration.
#[cfg(unix)]
fn enter_raw_mode() -> Result<libc::termios> {
    use std::os::unix::io::AsRawFd;

    let fd = std::io::stdin().as_raw_fd();
    let mut orig = std::mem::MaybeUninit::<libc::termios>::uninit();
    if unsafe { libc::tcgetattr(fd, orig.as_mut_ptr()) } != 0 {
        anyhow::bail!("Failed to get terminal attributes");
    }
    let orig = unsafe { orig.assume_init() };

    let mut raw = orig;
    raw.c_lflag &= !(libc::ECHO | libc::ICANON | libc::ISIG | libc::IEXTEN);
    raw.c_iflag &= !(libc::IXON | libc::ICRNL | libc::BRKINT | libc::INPCK | libc::ISTRIP);
    raw.c_cflag |= libc::CS8;
    raw.c_oflag &= !libc::OPOST;
    raw.c_cc[libc::VMIN] = 1;
    raw.c_cc[libc::VTIME] = 0;

    if unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &raw) } != 0 {
        anyhow::bail!("Failed to set raw terminal mode");
    }

    Ok(orig)
}

#[cfg(not(unix))]
fn enter_raw_mode() -> Result<()> {
    Ok(())
}

/// Restore terminal to original mode.
#[cfg(unix)]
fn restore_terminal(orig: &libc::termios) {
    use std::os::unix::io::AsRawFd;
    let fd = std::io::stdin().as_raw_fd();
    unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, orig) };
}

#[cfg(not(unix))]
fn restore_terminal(_orig: &()) {}

/// Play an asciicast v2 recording in a TUI pane.
///
/// Reads the cast file, creates a `CastPlayerApp`, spawns it as a
/// `PaneProcess`, and sends a `SpawnProcess` command to attach it
/// to the active pane in the specified session.
pub async fn handle_tui_play(
    signing_key: &SigningKey,
    cast_file: &std::path::Path,
    session_id: u32,
    loop_playback: bool,
) -> Result<()> {
    use hyprstream_tui::cast_app::{CastPlayerApp, LogEntry};
    use waxterm::app::TerminalConfig;

    info!(?cast_file, session_id, "Playing cast file");

    let content = std::fs::read_to_string(cast_file)
        .with_context(|| format!("Failed to read cast file: {}", cast_file.display()))?;

    let name = cast_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("cast")
        .to_owned();

    let mut app = CastPlayerApp::new(vec![LogEntry {
        name,
        cast_content: content,
    }]);
    app.set_loop(loop_playback);

    // Connect to the TUI session to get the active pane ID
    let client = create_tui_client(signing_key);
    let (cols, rows) = terminal_size();

    let result = client
        .connect(session_id, "capnp", cols, rows)
        .await
        .context("Failed to connect to TUI session. Is the TUI service running?")?;

    println!(
        "Playing {} in session {} (viewer {})",
        cast_file.display(),
        result.session_id,
        result.viewer_id,
    );

    // Find the active pane ID and actual pane dimensions from the connect result
    let active_pane_id = result
        .windows
        .first()
        .map(|w| w.active_pane_id)
        .unwrap_or(1);

    let (pane_cols, pane_rows) = result
        .windows
        .first()
        .and_then(|w| w.panes.iter().find(|p| p.id == active_pane_id))
        .map(|p| (p.cols, p.rows))
        .unwrap_or((cols, rows));

    // Spawn the app as a PaneProcess using the actual pane dimensions
    let config = TerminalConfig::new().cols(pane_cols).rows(pane_rows);
    let process = crate::tui::process::spawn_app_process(app, config);

    // Send SpawnProcess command via RPC
    // For now, the play command connects and spawns directly.
    // The TUI service frame loop will pick up the process via the command channel.
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        // Use sendInput as a signal to indicate playback started.
        // The actual process attachment is handled locally.
        let mut send_input = req.init_send_input();
        send_input.set_viewer_id(result.viewer_id);
        send_input.set_data(b"");
    })?;
    let _ = client.call(payload).await;

    // The process lives in this CLI process. Forward its output to the TUI
    // service via sendInput RPC, and subscribe to streams[0] (stdin) for
    // relayed viewer input (keypresses from browser/tui-attach viewers).
    let stdin_stream = result.streams.first();
    forward_process_output(
        process, &client, signing_key, result.viewer_id, active_pane_id,
        pane_cols, pane_rows, stdin_stream,
    ).await?;

    println!("Playback complete.");
    Ok(())
}

/// Forward process output to the TUI service via sendInput RPC calls.
///
/// Runs a resize polling task concurrently: every ~1s, queries the pane size
/// via `list_windows` RPC. If the size changed (e.g. a browser viewer connected),
/// sends `ProcessInput::Resize` to the local PaneProcess so the AnsiBackend
/// matches the new pane dimensions.
///
/// If `stdin_stream` is provided (FD 0 from ConnectResult), subscribes to
/// the PUB/SUB stream for relayed viewer input (keypresses from browser or
/// tui-attach viewers) and forwards them to the local PaneProcess.
async fn forward_process_output(
    mut process: crate::tui::process::PaneProcess,
    client: &TuiClient,
    signing_key: &SigningKey,
    viewer_id: u32,
    _pane_id: u32,
    initial_cols: u16,
    initial_rows: u16,
    stdin_stream: Option<&crate::services::generated::tui_client::StreamInfo>,
) -> Result<()> {
    use crate::tui::process::ProcessInput;

    let resize_tx = process.input_tx.clone();

    // Enter raw mode and read stdin if we're on a real terminal.
    // When stdin is not a tty (piped/headless), skip input handling.
    let is_tty = unsafe { libc::isatty(libc::STDIN_FILENO) } == 1;
    let orig_termios = if is_tty { Some(enter_raw_mode()?) } else { None };

    // Use a pipe to signal the stdin reader thread to exit.
    // We use a raw std::thread (not spawn_blocking) because tokio's runtime
    // waits for all spawn_blocking tasks on shutdown, and a blocking read()
    // on stdin can't be cancelled. With a raw thread we can just detach it.
    let stdin_forward;
    let mut cancel_fd: Option<std::os::unix::io::RawFd> = None;
    if is_tty {
        let stdin_tx = process.input_tx.clone();
        let (stdin_async_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);

        // Create a pipe for cancellation: write end signals the reader to stop
        let mut pipe_fds = [0i32; 2];
        if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } == 0 {
            let read_pipe = pipe_fds[0];
            let write_pipe = pipe_fds[1];
            cancel_fd = Some(write_pipe);

            // Raw thread (not spawn_blocking) — won't block runtime shutdown
            std::thread::spawn(move || {
                let stdin_fd = libc::STDIN_FILENO;
                let mut buf = [0u8; 256];
                loop {
                    // poll stdin and cancel pipe
                    let mut fds = [
                        libc::pollfd { fd: stdin_fd, events: libc::POLLIN, revents: 0 },
                        libc::pollfd { fd: read_pipe, events: libc::POLLIN, revents: 0 },
                    ];
                    let ret = unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) };
                    if ret <= 0 { break; }
                    // Cancel pipe signaled
                    if fds[1].revents & libc::POLLIN != 0 { break; }
                    // Stdin ready
                    if fds[0].revents & libc::POLLIN != 0 {
                        let n = unsafe { libc::read(stdin_fd, buf.as_mut_ptr().cast(), buf.len()) };
                        if n <= 0 { break; }
                        let data = buf[..n as usize].to_vec();
                        if stdin_async_tx.blocking_send(data).is_err() { break; }
                    }
                }
                unsafe { libc::close(read_pipe); }
            });
        }

        stdin_forward = Some(tokio::spawn(async move {
            while let Some(data) = stdin_rx.recv().await {
                if stdin_tx.send(ProcessInput::Stdin(data)).await.is_err() {
                    break;
                }
            }
        }));
    } else {
        stdin_forward = None;
    };

    // Resize polling task — detects when another viewer triggers a pane resize
    let resize_client = create_tui_client(signing_key);
    let resize_handle = tokio::spawn(async move {
        let mut cur = (initial_cols, initial_rows);
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            if let Ok((c, r)) = query_pane_size(&resize_client).await {
                if c > 0 && r > 0 && (c, r) != cur {
                    cur = (c, r);
                    let _ = resize_tx
                        .send(ProcessInput::Resize { cols: c, rows: r })
                        .await;
                }
            }
        }
    });

    // Subscribe to stdin stream (FD 0) for relayed viewer input
    let stdin_input_tx = process.input_tx.clone();
    let stdin_handle = if let Some(si) = stdin_stream {
        if !si.topic.is_empty() {
            let zmq_ctx = zmq::Context::new();
            let sub_endpoint = hyprstream_rpc::registry::global()
                .endpoint("streams", hyprstream_rpc::registry::SocketKind::Sub)
                .to_zmq_string();
            let sub_socket = zmq_ctx.socket(zmq::SUB)?;
            sub_socket.connect(&sub_endpoint)?;
            sub_socket.set_subscribe(si.topic.as_bytes())?;
            sub_socket.set_linger(0)?;

            let mac_key: [u8; 32] = si.mac_key.as_slice()
                .try_into()
                .context("stdin mac_key must be 32 bytes")?;
            let topic = si.topic.clone();

            let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);

            // ZMQ recv in blocking thread — verifies HMAC chain and extracts data
            let recv_handle = tokio::task::spawn_blocking(move || {
                use hyprstream_rpc::streaming::{StreamPayload, StreamVerifier};
                let mut verifier = StreamVerifier::new(mac_key, topic);
                loop {
                    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(3);
                    match sub_socket.recv_bytes(0) {
                        Ok(frame) => frames.push(frame),
                        Err(_) => break,
                    }
                    while sub_socket.get_rcvmore().unwrap_or(false) {
                        match sub_socket.recv_bytes(0) {
                            Ok(frame) => frames.push(frame),
                            Err(_) => break,
                        }
                    }
                    match verifier.verify(&frames) {
                        Ok(payloads) => {
                            for payload in payloads {
                                if let StreamPayload::Data(data) = payload {
                                    if tx.blocking_send(data).is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Stdin stream verification failed: {}", e);
                        }
                    }
                }
            });

            // Forward received input to local PaneProcess
            let forward_handle = tokio::spawn(async move {
                while let Some(data) = rx.recv().await {
                    if stdin_input_tx.send(ProcessInput::Stdin(data)).await.is_err() {
                        break;
                    }
                }
            });

            Some((recv_handle, forward_handle))
        } else {
            None
        }
    } else {
        None
    };

    // Forward output loop (existing sendInput RPC)
    while let Some(data) = process.stdout_rx.recv().await {
        let request_id = client.next_id();
        let payload = hyprstream_rpc::serialize_message(|msg| {
            let mut req =
                msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
            req.set_id(request_id);
            let mut send_input = req.init_send_input();
            send_input.set_viewer_id(viewer_id);
            send_input.set_data(&data);
        });
        if let Ok(payload) = payload {
            if let Err(e) = client.call(payload).await {
                tracing::debug!("sendInput failed: {}", e);
                break;
            }
        }
    }

    resize_handle.abort();
    if let Some(h) = stdin_forward { h.abort(); }
    // Abort stdin stream subscriber
    if let Some((recv_h, fwd_h)) = stdin_handle {
        recv_h.abort();
        fwd_h.abort();
    }
    // Signal stdin reader thread to exit via cancel pipe
    if let Some(fd) = cancel_fd {
        unsafe { libc::write(fd, b"x".as_ptr().cast(), 1); }
        unsafe { libc::close(fd); }
    }
    if let Some(ref orig) = orig_termios { restore_terminal(orig); }
    Ok(())
}

/// Query the active pane size from the TUI service via list_windows RPC.
async fn query_pane_size(client: &TuiClient) -> Result<(u16, u16)> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_list_windows(0); // session_id 0 = all
    })?;
    let response = client.call(payload).await?;

    match TuiClient::parse_response(&response)? {
        TuiResponseVariant::ListWindowsResult(window_list) => {
            if let Some(win) = window_list.windows.first() {
                if let Some(pane) = win.panes.iter().find(|p| p.id == win.active_pane_id) {
                    return Ok((pane.cols, pane.rows));
                }
                if let Some(pane) = win.panes.first() {
                    return Ok((pane.cols, pane.rows));
                }
            }
            anyhow::bail!("no panes found")
        }
        _ => anyhow::bail!("unexpected response"),
    }
}

/// Truncate a string with ellipsis if too long.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_owned()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}
