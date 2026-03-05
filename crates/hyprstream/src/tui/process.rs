//! PaneProcess — connects byte-producing processes to multiplexer panes.
//!
//! Provides a `PaneProcess` handle that represents a running application
//! connected to a pane via channels. The frame loop polls `stdout_rx`
//! for output bytes and feeds them through `PanePerformer`, and routes
//! input bytes through `input_tx`.

use std::io::{self, Write};

use tokio::sync::mpsc;
use waxterm::app::{TerminalApp, TerminalConfig};
use waxterm::backend::AnsiBackend;
use waxterm::input::InputParser;

/// Commands from the multiplexer to a hosted process.
pub enum ProcessInput {
    /// Raw bytes to deliver as stdin.
    Stdin(Vec<u8>),
    /// Pane was resized.
    Resize { cols: u16, rows: u16 },
    /// Kill the process.
    Kill,
}

/// A byte source connected to a pane.
///
/// The frame loop polls `stdout_rx` for ANSI output bytes and feeds them
/// through `PanePerformer::feed()`. Input from the viewer is sent through
/// `input_tx`.
pub struct PaneProcess {
    pub stdout_rx: mpsc::Receiver<Vec<u8>>,
    pub input_tx: mpsc::Sender<ProcessInput>,
}

/// A `Write` sink that sends bytes through a sync mpsc channel.
struct ChannelWriter {
    tx: std::sync::mpsc::SyncSender<Vec<u8>>,
    buf: Vec<u8>,
}

impl ChannelWriter {
    fn new(tx: std::sync::mpsc::SyncSender<Vec<u8>>) -> Self {
        Self {
            tx,
            buf: Vec::with_capacity(4096),
        }
    }
}

impl Write for ChannelWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.buf.is_empty() {
            let data = std::mem::take(&mut self.buf);
            self.tx
                .send(data)
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "receiver dropped"))?;
        }
        Ok(())
    }
}

impl Drop for ChannelWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Spawn a `TerminalApp` in a background thread, returning a `PaneProcess`
/// handle that the frame loop can use to exchange bytes with the app.
///
/// The app renders into an `AnsiBackend` whose output is sent through
/// the stdout channel. Input is received through the input channel and
/// dispatched to the app's `handle_input`.
pub fn spawn_app_process<A>(app: A, config: TerminalConfig) -> PaneProcess
where
    A: TerminalApp + Send + 'static,
    A::Command: Send + 'static,
{
    let (stdout_sync_tx, stdout_sync_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(64);
    let (input_tx, input_rx) = mpsc::channel::<ProcessInput>(64);

    // Bridge from std sync channel to tokio async channel
    let (stdout_tx, stdout_rx) = mpsc::channel::<Vec<u8>>(64);

    std::thread::spawn({
        let stdout_tx = stdout_tx;
        move || {
            while let Ok(data) = stdout_sync_rx.recv() {
                if stdout_tx.blocking_send(data).is_err() {
                    break;
                }
            }
        }
    });

    std::thread::spawn(move || {
        run_app_loop(app, config, stdout_sync_tx, input_rx);
    });

    PaneProcess {
        stdout_rx,
        input_tx,
    }
}

/// Run the app event loop on the current thread.
fn run_app_loop<A>(
    mut app: A,
    config: TerminalConfig,
    stdout_tx: std::sync::mpsc::SyncSender<Vec<u8>>,
    mut input_rx: mpsc::Receiver<ProcessInput>,
) where
    A: TerminalApp,
{
    let writer = ChannelWriter::new(stdout_tx);
    let backend = AnsiBackend::new(writer, config.cols, config.rows, false);
    let mut terminal = match ratatui::Terminal::new(backend) {
        Ok(t) => t,
        Err(_) => return,
    };

    let parser = InputParser::new(app.input_extensions());

    // Initial hide cursor + clear screen
    let _ = terminal.hide_cursor();
    let _ = terminal.clear();

    let frame_ms = 1000u64 / u64::from(config.fps);
    let mut needs_redraw = true;
    let mut last_tick = std::time::Instant::now();

    loop {
        if needs_redraw {
            if terminal.draw(|frame| app.render(frame)).is_err() {
                break;
            }
            needs_redraw = false;
        }

        std::thread::sleep(std::time::Duration::from_millis(frame_ms));

        let now = std::time::Instant::now();
        let delta_ms = now.duration_since(last_tick).as_millis() as u64;
        last_tick = now;

        // Poll input (non-blocking)
        while let Ok(input) = input_rx.try_recv() {
            match input {
                ProcessInput::Stdin(data) => {
                    let cmds = parser.parse(&data);
                    for cmd in cmds {
                        if app.handle_input(cmd) {
                            needs_redraw = true;
                        }
                    }
                }
                ProcessInput::Resize { cols, rows } => {
                    terminal.backend_mut().resize(cols, rows);
                    let _ = terminal.resize(ratatui::layout::Rect::new(0, 0, cols, rows));
                    let _ = terminal.clear();
                    needs_redraw = true;
                }
                ProcessInput::Kill => return,
            }
        }

        let tick_changed = app.tick(delta_ms);
        if tick_changed {
            needs_redraw = true;
        }

        if app.should_quit() {
            break;
        }
    }

    let _ = terminal.show_cursor();
    let _ = terminal.flush();
}
