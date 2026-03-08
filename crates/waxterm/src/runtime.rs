use std::io::{self, BufWriter, Write};

use ratatui::Terminal;

use crate::app::{TerminalApp, TerminalConfig};
use crate::backend::AnsiBackend;
use crate::input::InputParser;
#[cfg(target_os = "wasi")]
use crate::input::StdinPoller;

/// Synchronous event loop. Works on both native and WASI.
pub fn run_sync<A: TerminalApp>(mut app: A, config: TerminalConfig) -> io::Result<()> {
    let stdout = io::stdout();
    let backend = AnsiBackend::new(
        BufWriter::with_capacity(config.buf_capacity, stdout.lock()),
        config.cols,
        config.rows,
        config.heartbeat,
    );
    let mut terminal = Terminal::new(backend)?;

    // Build input parser with app-provided OSC extensions
    let parser = InputParser::new(app.input_extensions());

    // Hide cursor + clear screen (no alternate screen — simpler for Wanix)
    let mut out = io::stdout();
    write!(out, "\x1b[?25l\x1b[2J\x1b[H")?;
    out.flush()?;

    #[cfg(not(target_os = "wasi"))]
    {
        run_native(&mut app, &mut terminal, &parser, &config)?;
    }

    #[cfg(target_os = "wasi")]
    {
        run_wasi(&mut app, &mut terminal, &parser, &config)?;
    }

    // Restore terminal
    let mut out = io::stdout();
    write!(out, "\x1b[?25h\x1b[?1049l")?;
    out.flush()?;

    Ok(())
}

/// Async wrapper: runs sync loop on blocking thread. Native only.
#[cfg(feature = "tokio")]
pub async fn run_async<A: TerminalApp + Send + 'static>(
    app: A,
    config: TerminalConfig,
) -> io::Result<()> {
    tokio::task::spawn_blocking(move || run_sync(app, config))
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
}

// --- Native event loop ---

#[cfg(not(target_os = "wasi"))]
fn run_native<A, W>(
    app: &mut A,
    terminal: &mut Terminal<AnsiBackend<W>>,
    parser: &InputParser<A::Command>,
    config: &TerminalConfig,
) -> io::Result<()>
where
    A: TerminalApp,
    W: Write,
{
    use std::time::{Duration, Instant};

    let original_termios = enable_raw_mode()?;
    let mut stdin = io::stdin();
    let mut buf = [0u8; 256];
    let mut needs_redraw = true;
    let mut last_tick = Instant::now();
    let frame_ms = 1000 / u64::from(config.fps);

    loop {
        if needs_redraw {
            terminal.draw(|frame| app.render(frame))?;
            needs_redraw = false;
        }

        std::thread::sleep(Duration::from_millis(frame_ms));

        let now = Instant::now();
        let delta_ms = now.duration_since(last_tick).as_millis() as u64;
        last_tick = now;

        // Non-blocking stdin read
        use std::io::Read;
        match stdin.read(&mut buf) {
            Ok(n) if n > 0 => {
                let cmds = parser.parse(&buf[..n]);
                for cmd in cmds {
                    if app.handle_input(cmd) {
                        needs_redraw = true;
                    }
                }
            }
            _ => {}
        }

        if app.tick(delta_ms) {
            needs_redraw = true;
        }

        if app.should_quit() {
            break;
        }
    }

    disable_raw_mode(original_termios);
    Ok(())
}

#[cfg(not(target_os = "wasi"))]
fn enable_raw_mode() -> io::Result<libc::termios> {
    unsafe {
        let mut original = std::mem::zeroed::<libc::termios>();
        if libc::tcgetattr(0, &mut original) != 0 {
            return Err(io::Error::last_os_error());
        }
        let mut raw = original;
        raw.c_lflag &= !(libc::ECHO | libc::ICANON);
        raw.c_iflag &= !libc::ICRNL; // CR → NL translation off; Enter delivers 0x0D
        raw.c_cc[libc::VMIN] = 0;
        raw.c_cc[libc::VTIME] = 0;
        if libc::tcsetattr(0, libc::TCSANOW, &raw) != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(original)
    }
}

#[cfg(not(target_os = "wasi"))]
fn disable_raw_mode(original: libc::termios) {
    unsafe {
        libc::tcsetattr(0, libc::TCSANOW, &original);
    }
}

// --- WASI event loop ---

#[cfg(target_os = "wasi")]
fn run_wasi<A, W>(
    app: &mut A,
    terminal: &mut Terminal<AnsiBackend<W>>,
    parser: &InputParser<A::Command>,
    config: &TerminalConfig,
) -> io::Result<()>
where
    A: TerminalApp,
    W: Write,
{
    use std::time::{Duration, Instant};

    let mut poller = StdinPoller::new();
    let frame_ms = (1000 / u64::from(config.fps)) as u64;
    let mut render_timer_ms: u64 = frame_ms; // draw immediately on first tick
    let mut force_redraw = false;
    let mut last_tick = Instant::now();

    loop {
        // Render at target fps, or immediately when input changes state
        if render_timer_ms >= frame_ms || force_redraw {
            terminal.draw(|frame| app.render(frame))?;
            render_timer_ms = 0;
            force_redraw = false;
        }

        // Short sleep so the JS reader can drain stdout between frames.
        std::thread::sleep(Duration::from_millis(5));

        // Clamped delta: fallback to 5ms if clock_time_get is unimplemented,
        // cap at 50ms to prevent catch-up after focus loss.
        let now = Instant::now();
        let delta_ms = now.duration_since(last_tick).as_millis().clamp(5, 50) as u64;
        last_tick = now;

        // Poll stdin (non-blocking)
        let data = poller.read();
        if !data.is_empty() {
            let cmds = parser.parse(data);
            for cmd in cmds {
                app.handle_input(cmd);
            }
            force_redraw = true;
        }

        // Advance app state (decoupled from render rate)
        if app.tick(delta_ms) {
            // tick may produce output worth rendering
        }
        render_timer_ms += delta_ms;

        if app.should_quit() {
            break;
        }
    }

    Ok(())
}
