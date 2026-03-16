// WASI binary entry point for the hyprstream TUI
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use hyprstream_tui::wizard::backend::MockWizardBackend;
use hyprstream_tui::wizard::WizardApp;

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        #[cfg(not(target_os = "wasi"))]
        Some("shell") => run_shell(args),
        Some("wizard") => run_wizard(),
        #[cfg(target_os = "wasi")]
        Some("chat") => run_chat_wasi_v2(),
        #[cfg(target_os = "wasi")]
        _ => run_compositor_wasi(),
        #[cfg(not(target_os = "wasi"))]
        _ => run_wizard(),
    }
}

fn run_wizard() {
    let backend = MockWizardBackend::new();
    let app = WizardApp::new(backend);
    let config = waxterm::TerminalConfig::new();
    if let Err(e) = waxterm::run_sync(app, config) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "wasi"))]
fn run_shell(mut args: impl Iterator<Item = String>) {
    use hyprstream_tui::shell_app::ShellApp;

    // --models-dir PATH  (default: $HOME/.hyprstream/models)
    let models_dir = if args.next().as_deref() == Some("--models-dir") {
        args.next()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(default_models_dir)
    } else {
        default_models_dir()
    };

    let (cols, rows) = terminal_size();
    let models = hyprstream_tui::shell_app::discover_models(
        &models_dir.join(".registry").join("models"),
    );
    let models = if models.is_empty() {
        hyprstream_tui::shell_app::discover_models(&models_dir)
    } else {
        models
    };
    let app = ShellApp::new(
        models,
        cols,
        rows,
        Box::new(|_, _| {}),  // no model service in standalone mode
        Box::new(|_| false),
    );
    let config = waxterm::TerminalConfig::new().cols(cols).rows(rows);

    if let Err(e) = waxterm::run_sync(app, config) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "wasi"))]
fn default_models_dir() -> std::path::PathBuf {
    std::env::var("HYPRSTREAM_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_next::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".hyprstream")
                .join("models")
        })
}

// ============================================================================
// WASI compositor entry point (Phase W2)
// ============================================================================

/// Reads capnp `CompositorIpcIn` messages from stdin, drives `Compositor`,
/// writes pure ANSI to stdout, and emits capnp `CompositorIpcOut` messages
/// to a Wanix named pipe at `rpc/data`.
///
/// Wire format (both directions):
///   [4B little-endian message length][capnp single-segment bytes, no segment table]
#[cfg(target_os = "wasi")]
fn run_compositor_wasi() {
    use hyprstream_compositor::{Compositor, CompositorInput, CompositorOutput, WindowSummary};
    use hyprstream_tui::compositor_ipc_capnp::compositor_ipc_in;
    use std::io::{Read, Write};
    use waxterm::backend::AnsiBackend;
    use waxterm::input::{InputParser, KeyPress};

    let (mut cols, mut rows) = (80u16, 24u16);
    let stdout = std::io::stdout();
    let backend = AnsiBackend::new(
        std::io::BufWriter::with_capacity(64 * 1024, stdout),
        cols,
        rows,
        true, // heartbeat keeps Wanix pipe flushed during idle frames
    );
    let Ok(mut terminal) = ratatui::Terminal::new(backend) else { return };

    let mut compositor = Compositor::new(cols, rows, 0, 0, vec![], vec![]);
    terminal.draw(|f| compositor.render(f)).ok();

    // Open IPC output pipe (Wanix named pipe created by host via `bind #| rpc`)
    let mut ipc_out: Option<std::fs::File> = std::fs::OpenOptions::new()
        .write(true)
        .open("rpc/data")
        .ok();

    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();

    loop {
        let data = match read_capnp_frame(&mut stdin) {
            Ok(d) => d,
            Err(_) => break,
        };

        let reader = match capnp_reader_from_segment(&data) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let Ok(msg) = reader.get_root::<compositor_ipc_in::Reader<'_>>() else { continue };
        let Ok(which) = msg.which() else { continue };

        let mut needs_redraw = false;

        match which {
            compositor_ipc_in::AnsiFrame(Ok(frame)) => {
                let pane_id = frame.get_id();
                let ansi = frame.get_data().unwrap_or_default().to_vec();
                let outs = compositor.handle(CompositorInput::ServerFrame { pane_id, ansi });
                let quit = dispatch_compositor_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw, &mut ipc_out);
                if quit { return }
            }
            compositor_ipc_in::Keyboard(Ok(data_bytes)) => {
                let parser: InputParser<KeyPress> = InputParser::new(vec![]);
                for key in parser.parse(data_bytes) {
                    let outs = compositor.handle(CompositorInput::KeyPress(key));
                    let quit = dispatch_compositor_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw, &mut ipc_out);
                    if quit { return }
                }
            }
            compositor_ipc_in::Resize(Ok(resize)) => {
                cols = resize.get_cols();
                rows = resize.get_rows();
                terminal.backend_mut().resize(cols, rows);
                let outs = compositor.handle(CompositorInput::Resize(cols, rows));
                dispatch_compositor_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw, &mut ipc_out);
            }
            compositor_ipc_in::AppFrame(Ok(frame)) => {
                let app_id = frame.get_id();
                let ansi = frame.get_data().unwrap_or_default().to_vec();
                let outs = compositor.handle(CompositorInput::AppFrame { app_id, ansi });
                dispatch_compositor_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw, &mut ipc_out);
            }
            compositor_ipc_in::WindowList(Ok(data_bytes)) => {
                if let Ok(wins) = serde_json::from_slice::<Vec<WindowSummary>>(data_bytes) {
                    let outs = compositor.handle(CompositorInput::WindowList(wins));
                    dispatch_compositor_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw, &mut ipc_out);
                }
            }
            compositor_ipc_in::PaneClosed(pane_id) => {
                let outs = compositor.handle(CompositorInput::PaneClosed { pane_id });
                dispatch_compositor_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw, &mut ipc_out);
            }
            compositor_ipc_in::SessionReset(()) => {
                compositor = Compositor::new(cols, rows, 0, 0, vec![], vec![]);
                needs_redraw = true;
            }
            _ => {}
        }

        if needs_redraw {
            terminal.draw(|f| compositor.render(f)).ok();
        }
    }
}

/// Dispatch compositor outputs: Redraw, RPC (via capnp pipe), RouteInput (via capnp pipe), Quit.
/// Returns `true` if the session should exit.
#[cfg(target_os = "wasi")]
fn dispatch_compositor_outputs(
    outs: Vec<hyprstream_compositor::CompositorOutput>,
    _compositor: &mut hyprstream_compositor::Compositor,
    terminal: &mut ratatui::Terminal<waxterm::backend::AnsiBackend<std::io::BufWriter<std::io::Stdout>>>,
    needs_redraw: &mut bool,
    ipc_out: &mut Option<std::fs::File>,
) -> bool {
    use hyprstream_compositor::CompositorOutput;
    use hyprstream_tui::compositor_ipc_capnp::compositor_ipc_out;

    for out in outs {
        match out {
            CompositorOutput::Redraw => *needs_redraw = true,
            CompositorOutput::Rpc(req) => {
                let json = serde_json::to_vec(&req).unwrap_or_default();
                let mut builder = capnp::message::Builder::new_default();
                builder.init_root::<compositor_ipc_out::Builder<'_>>()
                    .set_rpc_request(&json);
                if write_capnp_message(ipc_out, &builder).is_err() {
                    *ipc_out = None; // EPIPE — stop writing
                }
            }
            CompositorOutput::RouteInput { app_id, data } => {
                let mut builder = capnp::message::Builder::new_default();
                let mut root = builder.init_root::<compositor_ipc_out::Builder<'_>>();
                let mut route = root.init_route_input();
                route.set_app_id(app_id);
                route.set_data(&data);
                if write_capnp_message(ipc_out, &builder).is_err() {
                    *ipc_out = None; // EPIPE — stop writing
                }
            }
            CompositorOutput::Quit => {
                terminal.draw(|_| {}).ok(); // flush
                return true;
            }
        }
    }
    false
}

// ============================================================================
// WASI chat entry point (Phase W3)
// ============================================================================

/// WASM chat using ChatApp with capnp IPC.
///
/// Reads `ChatAppIpcIn` capnp messages from stdin, drives ChatApp,
/// writes pure ANSI to stdout, and emits `ChatAppIpcOut` messages
/// to a Wanix named pipe.
#[cfg(target_os = "wasi")]
fn run_chat_wasi_v2() {
    use hyprstream_tui::chat_app::ChatApp;
    use hyprstream_tui::chat_ui_wasm::draw as chat_draw;
    use hyprstream_tui::compositor_ipc_capnp::{chat_app_ipc_in, chat_app_ipc_out};
    use waxterm::backend::AnsiBackend;
    use waxterm::input::{InputParser, KeyPress};

    let model_name = std::env::args().nth(2).unwrap_or_else(|| "unknown".to_owned());
    let (mut cols, mut rows) = (80u16, 24u16);

    let stdout = std::io::stdout();
    let backend = AnsiBackend::new(
        std::io::BufWriter::with_capacity(32 * 1024, stdout),
        cols, rows,
        true,
    );
    let Ok(mut terminal) = ratatui::Terminal::new(backend) else { return };

    let mut app = ChatApp::new_wasm(model_name, cols, rows);
    terminal.draw(|f| chat_draw(f, &app)).ok();

    // Open IPC output pipe for inference requests
    let mut ipc_out: Option<std::fs::File> = std::fs::OpenOptions::new()
        .write(true)
        .open("rpc/data")
        .ok();

    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();

    loop {
        let data = match read_capnp_frame(&mut stdin) {
            Ok(d) => d,
            Err(_) => break,
        };

        let reader = match capnp_reader_from_segment(&data) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let Ok(msg) = reader.get_root::<chat_app_ipc_in::Reader<'_>>() else { continue };
        let Ok(which) = msg.which() else { continue };

        let mut needs_redraw = false;

        match which {
            chat_app_ipc_in::Keyboard(Ok(key_data)) => {
                let parser: InputParser<KeyPress> = InputParser::new(vec![]);
                for key in parser.parse(key_data) {
                    if let Some(req_json) = app.handle_key(key) {
                        // Write inference request via capnp IPC pipe
                        let mut builder = capnp::message::Builder::new_default();
                        builder.init_root::<chat_app_ipc_out::Builder<'_>>()
                            .set_inference_request(req_json.as_bytes());
                        if write_capnp_message(&mut ipc_out, &builder).is_err() {
                            ipc_out = None;
                        }
                    }
                    needs_redraw = true;
                    if app.quit { break }
                }
                if app.quit { break }
            }
            chat_app_ipc_in::Resize(Ok(resize)) => {
                cols = resize.get_cols();
                rows = resize.get_rows();
                app.cols = cols;
                app.rows = rows;
                terminal.backend_mut().resize(cols, rows);
                needs_redraw = true;
            }
            chat_app_ipc_in::InferenceToken(Ok(token)) => {
                app.on_token(token);
                needs_redraw = true;
            }
            chat_app_ipc_in::InferenceComplete(()) => {
                app.on_stream_complete();
                needs_redraw = true;
            }
            chat_app_ipc_in::InferenceError(Ok(err_msg)) => {
                app.on_stream_error(err_msg.to_owned());
                needs_redraw = true;
            }
            chat_app_ipc_in::InferenceCancel(()) => {
                app.on_stream_cancelled();
                needs_redraw = true;
            }
            _ => {}
        }

        if needs_redraw {
            terminal.draw(|f| chat_draw(f, &app)).ok();
        }
    }
}

// ============================================================================
// Capnp IPC helpers
// ============================================================================

/// Read a length-prefixed capnp frame: [4B LE length][raw segment bytes].
#[cfg(target_os = "wasi")]
fn read_capnp_frame(r: &mut impl std::io::Read) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    read_exact(r, &mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut data = vec![0u8; len];
    read_exact(r, &mut data)?;
    Ok(data)
}

/// Create a capnp message reader from raw single-segment bytes (no segment table).
///
/// Prepends the standard capnp segment table header (single segment) so that
/// `read_message_from_flat_slice` can parse it.
#[cfg(target_os = "wasi")]
fn capnp_reader_from_segment(
    raw: &[u8],
) -> capnp::Result<capnp::message::Reader<capnp::serialize::OwnedSegments>> {
    // Pad to word boundary if needed
    let padded_len = (raw.len() + 7) & !7;
    let word_count = (padded_len / 8) as u32;

    // Build segment table: [segment_count - 1 = 0][segment_0_word_count]
    let mut buf = Vec::with_capacity(8 + padded_len);
    buf.extend_from_slice(&0u32.to_le_bytes()); // 1 segment
    buf.extend_from_slice(&word_count.to_le_bytes());
    buf.extend_from_slice(raw);
    // Zero-pad to word boundary
    buf.resize(8 + padded_len, 0);

    capnp::serialize::read_message_from_flat_slice(
        &mut &buf[..],
        capnp::message::ReaderOptions::default(),
    )
}

/// Write a capnp message as [4B LE length][raw segment bytes] to a pipe.
/// Returns `Err` on EPIPE (pipe closed by reader).
#[cfg(target_os = "wasi")]
fn write_capnp_message(
    pipe: &mut Option<std::fs::File>,
    builder: &capnp::message::Builder<capnp::message::HeapAllocator>,
) -> std::io::Result<()> {
    use std::io::Write;
    let Some(ref mut f) = pipe else {
        return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
    };
    let segments = builder.get_segments_for_output();
    // Single-segment assumption (default HeapAllocator produces one segment)
    let segment = segments.first().ok_or(std::io::ErrorKind::InvalidData)?;
    let len = segment.len() as u32;
    f.write_all(&len.to_le_bytes())?;
    f.write_all(segment)?;
    f.flush()
}

/// Read exactly `buf.len()` bytes from `r`, retrying on short reads.
#[cfg(target_os = "wasi")]
fn read_exact(r: &mut impl std::io::Read, buf: &mut [u8]) -> std::io::Result<()> {
    let mut offset = 0;
    while offset < buf.len() {
        match r.read(&mut buf[offset..])? {
            0 => return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
            n => offset += n,
        }
    }
    Ok(())
}

#[cfg(not(target_os = "wasi"))]
fn terminal_size() -> (u16, u16) {
    // TIOCGWINSZ via libc
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
            (ws.ws_col, ws.ws_row)
        } else {
            (80, 24)
        }
    }
}
