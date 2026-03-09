// Re-export background and theme from the compositor crate so existing code
// that uses `hyprstream_tui::background` / `hyprstream_tui::theme` still compiles.
pub use hyprstream_compositor::background;
pub use hyprstream_compositor::theme;

pub mod private_store;

pub mod cast_app;
pub mod cast_player;
pub mod cast_ui;
pub mod wizard;

#[cfg(not(target_os = "wasi"))]
pub mod chat_app;
#[cfg(not(target_os = "wasi"))]
pub mod chat_ui;
#[cfg(not(target_os = "wasi"))]
pub mod shell;
#[cfg(not(target_os = "wasi"))]
pub mod shell_app;
#[cfg(not(target_os = "wasi"))]
pub mod shell_ui;
