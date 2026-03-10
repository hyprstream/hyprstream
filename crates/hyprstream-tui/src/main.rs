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

/// Reads the framed stdin protocol, drives `Compositor`, writes ANSI to stdout,
/// and emits OSC-framed IPC for RpcRequest and RouteInput.
///
/// Frame format (9-byte header, all integers little-endian):
///   [type:u8][id:u32][len:u32][data:len]
///
/// Outbound OSC:
///   RpcRequest:  ESC ] 0xFE <len:u32 LE> <json>   BEL
///   RouteInput:  ESC ] 0xFF <app_id:u32 LE> <len:u32 LE> <data>  BEL
#[cfg(target_os = "wasi")]
fn run_compositor_wasi() {
    use hyprstream_compositor::{Compositor, CompositorInput, CompositorOutput, WindowSummary};
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

    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();
    let mut header = [0u8; 9];

    loop {
        if read_exact(&mut stdin, &mut header).is_err() { break }

        let msg_type = header[0];
        let id       = u32::from_le_bytes(header[1..5].try_into().unwrap());
        let len      = u32::from_le_bytes(header[5..9].try_into().unwrap()) as usize;

        let mut data = vec![0u8; len];
        if read_exact(&mut stdin, &mut data).is_err() { break }

        let mut needs_redraw = false;

        match msg_type {
            // 0x01 — ANSI frame from TuiService pane
            0x01 => {
                let outs = compositor.handle(CompositorInput::ServerFrame { pane_id: id, ansi: data });
                dispatch_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw);
            }
            // 0x02 — raw keyboard bytes
            0x02 => {
                let parser: InputParser<KeyPress> = InputParser::new(vec![]);
                for key in parser.parse(&data) {
                    let outs = compositor.handle(CompositorInput::KeyPress(key));
                    let quit = dispatch_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw);
                    if quit { return }
                }
            }
            // 0x03 — resize: cols:u16 LE, rows:u16 LE, 4 bytes padding
            0x03 if data.len() >= 4 => {
                cols = u16::from_le_bytes([data[0], data[1]]);
                rows = u16::from_le_bytes([data[2], data[3]]);
                terminal.backend_mut().resize(cols, rows);
                let outs = compositor.handle(CompositorInput::Resize(cols, rows));
                dispatch_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw);
            }
            // 0x04 — ANSI frame from ChatApp (Phase W3)
            0x04 => {
                let outs = compositor.handle(CompositorInput::AppFrame { app_id: id, ansi: data });
                dispatch_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw);
            }
            // 0x05 — window list JSON
            0x05 => {
                if let Ok(wins) = serde_json::from_slice::<Vec<WindowSummary>>(&data) {
                    let outs = compositor.handle(CompositorInput::WindowList(wins));
                    dispatch_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw);
                }
            }
            // 0x06 — pane closed
            0x06 => {
                let outs = compositor.handle(CompositorInput::PaneClosed { pane_id: id });
                dispatch_outputs(outs, &mut compositor, &mut terminal, &mut needs_redraw);
            }
            // 0x07 — session reset
            0x07 => {
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

/// Returns `true` if the session should exit (Quit output).
#[cfg(target_os = "wasi")]
fn dispatch_outputs(
    outs: Vec<hyprstream_compositor::CompositorOutput>,
    _compositor: &mut hyprstream_compositor::Compositor,
    terminal: &mut ratatui::Terminal<waxterm::backend::AnsiBackend<std::io::BufWriter<std::io::Stdout>>>,
    needs_redraw: &mut bool,
) -> bool {
    use hyprstream_compositor::CompositorOutput;
    for out in outs {
        match out {
            CompositorOutput::Redraw => *needs_redraw = true,
            CompositorOutput::Rpc(req) => osc_write_rpc(&req),
            CompositorOutput::RouteInput { app_id, data } => osc_write_route(app_id, &data),
            CompositorOutput::Quit => {
                terminal.draw(|_| {}).ok(); // flush
                return true;
            }
        }
    }
    false
}

/// Encode `req` as  ESC ] 0xFE <len:u32 LE> <json> BEL  on stdout.
#[cfg(target_os = "wasi")]
fn osc_write_rpc(req: &hyprstream_compositor::RpcRequest) {
    use std::io::Write;
    let json = serde_json::to_vec(req).unwrap_or_default();
    let len  = json.len() as u32;
    let mut out = std::io::stdout();
    let _ = out.write_all(b"\x1b]");
    let _ = out.write_all(&[0xFEu8]);
    let _ = out.write_all(&len.to_le_bytes());
    let _ = out.write_all(&json);
    let _ = out.write_all(b"\x07");
    let _ = out.flush();
}

/// Encode route input as  ESC ] 0xFF <app_id:u32 LE> <len:u32 LE> <data> BEL  on stdout.
#[cfg(target_os = "wasi")]
fn osc_write_route(app_id: u32, data: &[u8]) {
    use std::io::Write;
    let len = data.len() as u32;
    let mut out = std::io::stdout();
    let _ = out.write_all(b"\x1b]");
    let _ = out.write_all(&[0xFFu8]);
    let _ = out.write_all(&app_id.to_le_bytes());
    let _ = out.write_all(&len.to_le_bytes());
    let _ = out.write_all(data);
    let _ = out.write_all(b"\x07");
    let _ = out.flush();
}

// ============================================================================
// WASI chat entry point (Phase W3)
// ============================================================================

/// WASM chat using ChatApp.
///
/// Uses ChatApp's push-based event methods (on_token, on_stream_complete, etc.)
/// and submit_message_payload() for inference requests.
#[cfg(target_os = "wasi")]
fn run_chat_wasi_v2() {
    use hyprstream_tui::chat_app::ChatApp;
    use hyprstream_tui::chat_ui_wasm::draw as chat_draw;
    use std::io::Write;
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

    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();
    let mut header = [0u8; 9];

    loop {
        if read_exact(&mut stdin, &mut header).is_err() { break }

        let msg_type = header[0];
        let len = u32::from_le_bytes(header[5..9].try_into().unwrap()) as usize;
        let mut data = vec![0u8; len];
        if read_exact(&mut stdin, &mut data).is_err() { break }

        let mut needs_redraw = false;

        match msg_type {
            0x01 => {
                // Keyboard bytes from compositor RouteInput
                let parser: InputParser<KeyPress> = InputParser::new(vec![]);
                for key in parser.parse(&data) {
                    if let Some(req_json) = app.handle_key(key) {
                        osc_write_inference(&req_json);
                    }
                    needs_redraw = true;
                    if app.quit { break }
                }
                if app.quit { break }
            }
            0x02 if data.len() >= 4 => {
                cols = u16::from_le_bytes([data[0], data[1]]);
                rows = u16::from_le_bytes([data[2], data[3]]);
                app.cols = cols;
                app.rows = rows;
                terminal.backend_mut().resize(cols, rows);
                needs_redraw = true;
            }
            0x03 => {
                // Inference token
                if let Ok(s) = std::str::from_utf8(&data) {
                    app.on_token(s);
                    needs_redraw = true;
                }
            }
            0x04 => {
                // Inference complete
                app.on_stream_complete();
                needs_redraw = true;
            }
            0x05 => {
                // Inference error
                let msg = std::str::from_utf8(&data).unwrap_or("unknown error");
                app.on_stream_error(msg.to_owned());
                needs_redraw = true;
            }
            0x06 => {
                // Inference cancelled
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

/// Write an inference request as  ESC ] 0xFD <len:u32 LE> <json> BEL  on stdout.
#[cfg(target_os = "wasi")]
fn osc_write_inference(json: &str) {
    use std::io::Write;
    let bytes = json.as_bytes();
    let len = bytes.len() as u32;
    let mut out = std::io::stdout();
    let _ = out.write_all(b"\x1b]");
    let _ = out.write_all(&[0xFDu8]);
    let _ = out.write_all(&len.to_le_bytes());
    let _ = out.write_all(bytes);
    let _ = out.write_all(b"\x07");
    let _ = out.flush();
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
