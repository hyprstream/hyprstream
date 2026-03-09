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
