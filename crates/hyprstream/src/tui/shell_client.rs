//! RPC helpers for the ShellClient TUI.
//!
//! Functions for communicating with TuiService over Cap'n Proto ZMQ RPC:
//! spawnShell, pollStdin, focusWindow, spawnChatApp, closeWindow,
//! createPrivatePane.

use crate::services::generated::tui_client::TuiClient;

// ============================================================================
// Manual RPC helpers (spawnShell / closePane)
// ============================================================================

/// Send a spawnShell RPC to TuiService.
pub async fn spawn_shell_rpc(
    client: &TuiClient,
    session_id: u32,
    pane_id: u32,
    cwd: &str,
) -> anyhow::Result<()> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        let mut spawn = req.init_spawn_shell();
        spawn.set_session_id(session_id);
        spawn.set_pane_id(pane_id);
        spawn.set_cwd(cwd);
    })?;
    client.call(payload).await?;
    Ok(())
}

/// Poll for stdin bytes queued for this viewer (pollStdin RPC).
pub async fn poll_stdin_rpc(client: &TuiClient, viewer_id: u32) -> anyhow::Result<Vec<u8>> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_poll_stdin(viewer_id);
    })?;
    let response_bytes = client.call(payload).await?;
    let reader = capnp::serialize::read_message_from_flat_slice(
        &mut &response_bytes[..],
        capnp::message::ReaderOptions::default(),
    )?;
    let response = reader.get_root::<crate::tui_capnp::tui_response::Reader<'_>>()?;
    match response.which()? {
        crate::tui_capnp::tui_response::Which::PollStdinResult(data) => Ok(data?.to_vec()),
        crate::tui_capnp::tui_response::Which::Error(e) => {
            let msg = e?.get_message()?.to_str().unwrap_or("unknown").to_owned();
            Err(anyhow::anyhow!("pollStdin error: {}", msg))
        }
        _ => Ok(vec![]),
    }
}

/// Tell TuiService which window is active for this session.
pub async fn focus_window_rpc(client: &TuiClient, window_id: u32) -> anyhow::Result<()> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_focus_window(window_id);
    })?;
    client.call(payload).await?;
    Ok(())
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
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        let mut chat = req.init_spawn_chat_app();
        chat.set_session_id(session_id);
        chat.set_model_ref(model_ref);
        chat.set_cols(cols);
        chat.set_rows(rows);
        chat.set_pane_id(pane_id);
    })?;
    client.call(payload).await?;
    Ok(())
}

/// Close a window by ID via RPC.
pub async fn close_window_rpc(client: &TuiClient, window_id: u32) -> anyhow::Result<()> {
    let request_id = client.next_id();
    let payload = hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(request_id);
        req.set_close_window(window_id);
    })?;
    client.call(payload).await?;
    Ok(())
}

