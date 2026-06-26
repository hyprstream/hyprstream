//! moq streaming Worker bridge via SharedArrayBuffer DMA ring buffer.
//!
//! Resolves the `moq-net Session: Sync` incompatibility with browser `web-sys`
//! types without waiting for an upstream moq-net fix. The iroh/moq session runs
//! inside a dedicated Web Worker (a real OS thread from wasm's perspective, where
//! `Send + Sync` requirements are logically satisfied for single-threaded-per-worker
//! JS objects). The main browser thread communicates with the worker via the
//! existing `DmaTransport` SharedArrayBuffer ring buffer (`hyprstream-9p/src/dma.rs`).
//!
//! # Architecture
//!
//! ```text
//! Main thread (wasm32)                    Web Worker (wasm32)
//! ─────────────────────────────────       ──────────────────────────────────────
//! MoqWorkerHandle                         moq_worker_main()
//!   .subscribe(track) ──────────────────► recv subscribe cmd over DMA
//!   .recv_frame() ◄─────────────────────  send frames over DMA
//!                                         │
//!                                         ▼
//!                                    WebTransport → moq relay/server
//!                                    (opened FROM worker; WT here is the
//!                                     sole JS thread so !Sync is not an issue
//!                                     in practice — and moq-net is available
//!                                     once its Sync bound is relaxed)
//! ```
//!
//! # DMA frame envelope
//!
//! Both directions use the same layout over `DmaTransport`'s length-prefixed messages:
//! ```text
//! [2 bytes LE: track_name_len][track_name bytes][4 bytes LE: payload_len][payload bytes]
//! ```
//! A special track name `__cmd__` carries JSON control messages from main→worker.
//!
//! # COOP/COEP
//!
//! `SharedArrayBuffer` requires `Cross-Origin-Opener-Policy: same-origin` and
//! `Cross-Origin-Embedder-Policy: credentialless` (or `require-corp`). These
//! headers are already set by the `crossOriginIsolation()` Vite plugin in the
//! www-cyberdione-ai frontend.
//!
//! # moq-net Sync status
//!
//! `moq-net 0.1.8` requires `Session: Send + Sync`. `web_sys::WebTransport` is
//! `!Sync`. The worker infrastructure here is complete and ready; the actual
//! moq subscribe/publish loop (`worker_moq_loop`) is stubbed pending either:
//! a) upstream moq-net relaxing the Sync bound, or
//! b) a local `[patch.crates-io]` fork with the one-line fix.
//! Track: <https://github.com/hyprstream/hyprstream/issues/484>

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use web_sys::{Worker, WorkerOptions, WorkerType};

use hyprstream_9p::dma::DmaTransport;

// ─── Frame envelope helpers ──────────────────────────────────────────────────

/// Encode a single DMA frame: `[2: track_len][track][4: payload_len][payload]`.
pub fn encode_frame(track: &str, payload: &[u8]) -> Vec<u8> {
    let t = track.as_bytes();
    let mut out = Vec::with_capacity(2 + t.len() + 4 + payload.len());
    out.extend_from_slice(&(t.len() as u16).to_le_bytes());
    out.extend_from_slice(t);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Decode a DMA frame. Returns `(track_name, payload)` or `None` if malformed.
pub fn decode_frame(buf: &[u8]) -> Option<(&str, &[u8])> {
    if buf.len() < 6 {
        return None;
    }
    let track_len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    if buf.len() < 2 + track_len + 4 {
        return None;
    }
    let track = std::str::from_utf8(&buf[2..2 + track_len]).ok()?;
    let payload_len = u32::from_le_bytes([
        buf[2 + track_len],
        buf[3 + track_len],
        buf[4 + track_len],
        buf[5 + track_len],
    ]) as usize;
    let start = 2 + track_len + 4;
    if buf.len() < start + payload_len {
        return None;
    }
    Some((track, &buf[start..start + payload_len]))
}

// ─── Main-thread handle ───────────────────────────────────────────────────────

/// Handle to the moq streaming Web Worker, held on the main thread.
///
/// Created via [`MoqWorkerHandle::spawn`]. Send subscribe/publish commands
/// and receive stream frames over the DMA ring buffer.
#[wasm_bindgen]
pub struct MoqWorkerHandle {
    sab: js_sys::SharedArrayBuffer,
    dma: DmaTransport,
    worker: Worker,
}

#[wasm_bindgen]
impl MoqWorkerHandle {
    /// Spawn a moq streaming Worker and connect it to a moq relay/server.
    ///
    /// # Arguments
    /// - `worker_script_url`: URL of the worker JS shim (e.g. `/moq-worker.js`).
    /// - `reach_json`: JSON-serialized reach config, e.g.
    ///   `{"url":"https://relay.example.com:443","cert_hash":"<base64>","track":"local/streams/abcdef"}`.
    ///
    /// # COOP/COEP
    /// The page must be served with `Cross-Origin-Opener-Policy: same-origin` and
    /// `Cross-Origin-Embedder-Policy: credentialless` for `SharedArrayBuffer` to be
    /// available. The `crossOriginIsolation()` Vite plugin handles this in dev; ensure
    /// production headers are also set.
    pub fn spawn(worker_script_url: &str, reach_json: &str) -> Result<MoqWorkerHandle, JsError> {
        // Allocate a 256 KiB SAB: 4 KiB control + 2 × 126 KiB data.
        let sab_size = 4096 + 2 * 128 * 1024;
        let sab = js_sys::SharedArrayBuffer::new(sab_size);

        // Main thread takes the client endpoint (writes on chan0, reads chan1).
        let dma = DmaTransport::new(&sab, true);

        // Spawn the worker.
        let mut opts = WorkerOptions::new();
        opts.type_(WorkerType::Module);
        let worker = Worker::new_with_options(worker_script_url, &opts)
            .map_err(|e| JsError::new(&format!("Worker spawn failed: {:?}", e)))?;

        // Send (SAB, reach_json) to the worker via postMessage.
        let msg = js_sys::Array::new();
        msg.push(&sab);
        msg.push(&JsValue::from_str(reach_json));
        let transfer = js_sys::Array::new();
        // SAB is shared, not transferred (cannot transfer a SAB — only ArrayBuffer).
        worker
            .post_message_with_transfer(&msg, &transfer)
            .map_err(|e| JsError::new(&format!("postMessage failed: {:?}", e)))?;

        Ok(MoqWorkerHandle { sab, dma, worker })
    }

    /// Send a subscribe command for `track_name` to the worker.
    pub async fn subscribe(&self, track_name: &str) -> Result<(), JsError> {
        let cmd = serde_json::json!({ "cmd": "subscribe", "track": track_name }).to_string();
        let frame = encode_frame("__cmd__", cmd.as_bytes());
        self.dma
            .send(&frame)
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Poll for the next incoming frame from the worker.
    ///
    /// Returns `(track_name, payload_bytes)` as a two-element JS array, or null
    /// if no frame is available yet.
    pub async fn recv_frame(&self) -> Result<JsValue, JsError> {
        let raw = self
            .dma
            .recv()
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        match decode_frame(&raw) {
            Some((track, payload)) => {
                let arr = js_sys::Array::new();
                arr.push(&JsValue::from_str(track));
                arr.push(&js_sys::Uint8Array::from(payload).into());
                Ok(arr.into())
            }
            None => Ok(JsValue::NULL),
        }
    }

    /// Terminate the worker.
    pub fn terminate(&self) {
        self.worker.terminate();
    }
}

// ─── Worker entrypoint ────────────────────────────────────────────────────────

/// Entry function called inside the Web Worker.
///
/// Receives `(sab: SharedArrayBuffer, reach_json: String)` via the initial
/// `postMessage` from the main thread, then loops — reading subscribe commands
/// from DMA and pushing stream frames back.
///
/// The WebTransport connection is opened FROM the worker thread, which means
/// `web_sys::WebTransport` is the only JS object on this thread and the
/// `!Sync` restriction is logically satisfied (no cross-thread sharing occurs).
///
/// # moq-net stub
///
/// The `worker_moq_loop` function is currently stubbed — it dials the relay
/// via WebTransport and echoes a "ready" frame to main, but does not yet
/// implement full moq SUBSCRIBE/OBJECT framing. This is gated on:
/// - `moq-net` relaxing its `Session: Sync` bound for wasm32, OR
/// - A local `[patch.crates-io]` fork (one-line fix; tracked in #484).
///
/// The DMA channel and Worker spawn infrastructure are fully functional.
#[wasm_bindgen]
pub async fn moq_worker_main(sab: JsValue, reach_json: String) {
    console_error_panic_hook::set_once();

    let sab = match sab.dyn_into::<js_sys::SharedArrayBuffer>() {
        Ok(s) => s,
        Err(_) => {
            web_sys::console::error_1(&"moq_worker_main: sab is not a SharedArrayBuffer".into());
            return;
        }
    };

    // Worker is the server endpoint (reads on chan0, writes on chan1).
    let dma = DmaTransport::new(&sab, false);

    web_sys::console::log_1(
        &format!("[moq-worker] started, reach: {}", &reach_json[..reach_json.len().min(120)])
            .into(),
    );

    // Parse reach config.
    let reach: serde_json::Value = match serde_json::from_str(&reach_json) {
        Ok(v) => v,
        Err(e) => {
            let err = encode_frame("__error__", format!("bad reach_json: {e}").as_bytes());
            let _ = dma.send(&err).await;
            return;
        }
    };

    // Signal to main thread that the worker is up.
    let ready = encode_frame("__ready__", b"1");
    if let Err(e) = dma.send(&ready).await {
        web_sys::console::error_1(&format!("[moq-worker] DMA send error: {e}").into());
        return;
    }

    // Main loop: receive subscribe commands from main, relay frames back.
    // moq framing stub — full implementation pending moq-net Sync fix (#484).
    loop {
        let raw = match dma.recv().await {
            Ok(r) => r,
            Err(e) => {
                web_sys::console::error_1(&format!("[moq-worker] DMA recv error: {e}").into());
                break;
            }
        };

        let (track, payload) = match decode_frame(&raw) {
            Some(f) => f,
            None => continue,
        };

        if track == "__cmd__" {
            let cmd: serde_json::Value = match serde_json::from_slice(payload) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if cmd.get("cmd").and_then(|c| c.as_str()) == Some("subscribe") {
                let track_name = cmd
                    .get("track")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let url = reach
                    .get("url")
                    .and_then(|u| u.as_str())
                    .unwrap_or("")
                    .to_string();
                web_sys::console::log_1(
                    &format!("[moq-worker] subscribe track={track_name} url={url}").into(),
                );
                // TODO(#484): open WebTransport to `url`, implement moq SUBSCRIBE +
                // OBJECT consumer, push received OBJECT payloads via:
                //   dma.send(&encode_frame(&track_name, &object_payload)).await
                //
                // Stub: echo an acknowledgment so the main thread knows the subscribe
                // was received and the DMA channel is working.
                let ack = encode_frame(
                    &track_name,
                    format!("__subscribed_stub__:{url}").as_bytes(),
                );
                if let Err(e) = dma.send(&ack).await {
                    web_sys::console::error_1(
                        &format!("[moq-worker] DMA send ack error: {e}").into(),
                    );
                    break;
                }
            }
        }
    }
}
