// WASI binary entry point for the hyprstream TUI
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use hyprstream_tui::wizard::backend::MockWizardBackend;
use hyprstream_tui::wizard::WizardApp;

fn main() {
    let backend = MockWizardBackend::new();
    let app = WizardApp::new(backend);
    let config = waxterm::TerminalConfig::new();

    if let Err(e) = waxterm::run_sync(app, config) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
