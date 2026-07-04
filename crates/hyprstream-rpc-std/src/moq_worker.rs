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
//! # moq-net on wasm
//!
//! `moq-net` required `Session: Send + Sync` (rejecting a `!Sync`
//! `web_sys::WebTransport` Session) and used real-path `tokio::time` (panics on
//! wasm). Both are fixed on our fork (ewindisch/moq PR #1: Session cfg-split +
//! `tokio::time`→`web_async::time` + producer-on-wasm), sourced via
//! `[patch.crates-io]` until upstream merges. The subscribe loop
//! ([`run_subscribe`]) is implemented against it.
//! Track: <https://github.com/hyprstream/hyprstream/issues/719>

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use web_sys::{Worker, WorkerOptions, WorkerType};

use hyprstream_9p::client::P9Transport;
use hyprstream_9p::dma::DmaTransport;

use crate::moq_frame::{decode_frame, encode_frame, parse_reach};

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
    /// - `reach_json`: JSON reach config from the frontend's `selectReach`, e.g.
    ///   `{"url":"https://relay.example.com:443/moq","certHashes":["<base64 SHA-256>"]}`.
    ///   The broadcast path to subscribe arrives later in a `subscribe` command
    ///   (not here), since one worker can serve several subscriptions. See
    ///   [`crate::moq_frame::ReachConfig`].
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

// ─── moq subscribe loop ───────────────────────────────────────────────────────

/// Dial the moq plane over WebTransport, subscribe to the stream broadcast, and
/// pump each received frame to the main thread over DMA.
///
/// Mirrors the native subscriber (`hyprstream-rpc::dial`): open a
/// `web_transport_trait::Session` (here `web_sys::WebTransport` via
/// `web-transport-wasm`, adapted in `moq_wt_session`), run the moq handshake with
/// a consume-side origin, await the announced `broadcast_path`, subscribe to the
/// single `STREAM_TRACK` ("stream"), and forward every group's frames. Each frame
/// is a `capnp StreamBlock || mac[16]` the main thread parses/verifies/decrypts.
///
/// Returns when the track closes (stream complete) or on the first error.
async fn run_subscribe(
    dma: &DmaTransport,
    url: &str,
    broadcast_path: &str,
    cert_hashes: Vec<Vec<u8>>,
) -> Result<(), String> {
    use crate::moq_wt_session::WtSession;

    let parsed = url::Url::parse(url).map_err(|e| format!("bad reach url {url:?}: {e}"))?;

    // Cert-pin to the self-signed mesh leaf when hashes are advertised; otherwise
    // fall back to the browser's system roots (a real did:web PDS with a CA cert).
    let client = if cert_hashes.is_empty() {
        web_transport_wasm::ClientBuilder::new().with_system_roots()
    } else {
        web_transport_wasm::ClientBuilder::new().with_server_certificate_hashes(cert_hashes)
    };
    let wt = client
        .connect(parsed)
        .await
        .map_err(|e| format!("WebTransport connect: {e}"))?;
    let session = WtSession(wt);

    // moq handshake, consume side: received broadcasts are announced into
    // `client_consumer` (mirrors `dial.rs`'s loopback subscriber).
    let client_origin = moq_net::Origin::random().produce();
    let client_consumer = client_origin.consume();
    let moq_client = moq_net::Client::new().with_consume(client_origin);
    let _moq_session = moq_client
        .connect(session)
        .await
        .map_err(|e| format!("moq handshake: {e}"))?;

    let broadcast = client_consumer
        .announced_broadcast(broadcast_path)
        .await
        .ok_or_else(|| format!("broadcast {broadcast_path:?} was not announced"))?;
    let mut track = broadcast
        .subscribe_track(&moq_net::Track::new("stream"))
        .map_err(|e| format!("subscribe_track(stream): {e}"))?;

    // Consume groups in order; each group's frames are pushed to main over DMA.
    loop {
        let mut group = match track
            .next_group()
            .await
            .map_err(|e| format!("next_group: {e}"))?
        {
            Some(g) => g,
            None => return Ok(()), // producer closed the track — stream complete.
        };
        while let Some(frame) = group
            .read_frame()
            .await
            .map_err(|e| format!("read_frame: {e}"))?
        {
            dma.send(&encode_frame("stream", &frame))
                .await
                .map_err(|e| format!("DMA send frame: {e}"))?;
        }
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
/// moq-net's over-tight `SessionInner: Send + Sync` bound is relaxed for wasm by
/// the `third_party/moq-net` fork so this Session can drive the moq consumer.
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

    // Truncate on a char boundary, not a raw byte index — slicing mid-UTF-8
    // (an internationalized host or non-ASCII cert encoding) would panic and
    // kill the worker before the DMA channel is established.
    let reach_preview: String = reach_json.chars().take(120).collect();
    web_sys::console::log_1(&format!("[moq-worker] started, reach: {}", reach_preview).into());

    // Parse reach config (url + base64-decoded cert-hash pins).
    let reach = match parse_reach(&reach_json) {
        Ok(r) => r,
        Err(e) => {
            let err = encode_frame("__error__", e.as_bytes());
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

    // Main loop: receive subscribe commands from main, run the moq subscribe
    // (`run_subscribe`) for each, relaying OBJECT frames back over DMA.
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
                // `track` is the moq broadcast path (e.g. "local/streams/{topic_hex}");
                // the token stream itself lives on the single STREAM_TRACK within it.
                let broadcast_path = cmd
                    .get("track")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                // url + cert-hash pins come from the reach config (parsed once);
                // empty cert_hashes ⇒ use the browser's system roots.
                let url = reach.url.clone();
                let cert_hashes = reach.cert_hashes.clone();

                web_sys::console::log_1(
                    &format!("[moq-worker] subscribe broadcast={broadcast_path} url={url}").into(),
                );

                // Runs until the track closes (stream complete) or errors; then the
                // outer loop resumes waiting for the next command. One inference
                // stream per worker, so blocking the command loop here is intended.
                if let Err(e) =
                    run_subscribe(&dma, &url, &broadcast_path, cert_hashes).await
                {
                    web_sys::console::error_1(&format!("[moq-worker] subscribe: {e}").into());
                    let err = encode_frame("__error__", e.as_bytes());
                    let _ = dma.send(&err).await;
                }
            }
        }
    }
}
