//! RPC helpers for the ShellClient TUI.
//!
//! Thin wrappers around generated `TuiClient` typed methods for ergonomic
//! call-site usage: spawnShell, pollStdin, focusWindow, spawnChatApp, closeWindow.

use crate::services::generated::tui_client::{
    SpawnShellRequest, SpawnChatAppRequest, TuiClient,
};

/// Send a spawnShell RPC to TuiService.
pub async fn spawn_shell_rpc(
    client: &TuiClient,
    session_id: u32,
    pane_id: u32,
    cwd: &str,
) -> anyhow::Result<()> {
    let _ = client.spawn_shell(&SpawnShellRequest {
        session_id,
        pane_id,
        cwd: cwd.to_owned(),
    }).await?;
    Ok(())
}

/// Poll for stdin bytes queued for this viewer (pollStdin RPC).
pub async fn poll_stdin_rpc(client: &TuiClient, viewer_id: u32) -> anyhow::Result<Vec<u8>> {
    client.poll_stdin(viewer_id).await
}

/// Tell TuiService which window is active for this session.
pub async fn focus_window_rpc(client: &TuiClient, window_id: u32) -> anyhow::Result<()> {
    client.focus_window(window_id).await
}

/// Send a spawnChatApp RPC to TuiService.
pub async fn spawn_chat_app_rpc(
    client: &TuiClient,
    session_id: u32,
    model_ref: &str,
    cols: u16,
    rows: u16,
    pane_id: u32,
) -> anyhow::Result<()> {
    let _ = client.spawn_chat_app(&SpawnChatAppRequest {
        session_id,
        model_ref: model_ref.to_owned(),
        cols,
        rows,
        pane_id,
    }).await?;
    Ok(())
}

/// Close a window by ID via RPC.
pub async fn close_window_rpc(client: &TuiClient, window_id: u32) -> anyhow::Result<()> {
    client.close_window(window_id).await
}
