// Re-export background and theme from the compositor crate so existing code
// that uses `hyprstream_tui::background` / `hyprstream_tui::theme` still compiles.
pub use hyprstream_compositor::background;
pub use hyprstream_compositor::theme;

/// Generated Cap'n Proto types for compositor/ChatApp IPC.
#[cfg(target_os = "wasi")]
pub mod compositor_ipc_capnp {
    #![allow(dead_code, unused_imports, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/compositor_ipc_capnp.rs"));
}

pub mod private_store;

pub mod chat_app;

#[cfg(target_os = "wasi")]
pub mod chat_ui_wasm;

pub mod cast_app;
pub mod cast_player;
pub mod cast_ui;
pub mod wizard;

#[cfg(not(target_os = "wasi"))]
pub mod console_app;
#[cfg(not(target_os = "wasi"))]
pub mod chat_ui;
#[cfg(not(target_os = "wasi"))]
pub mod vt_pane;
#[cfg(not(target_os = "wasi"))]
pub mod container_app;
#[cfg(not(target_os = "wasi"))]
pub mod shell;
#[cfg(not(target_os = "wasi"))]
pub mod shell_app;
#[cfg(not(target_os = "wasi"))]
pub mod shell_ui;
