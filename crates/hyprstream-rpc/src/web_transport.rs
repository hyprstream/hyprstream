//! WebTransport client for browser (wasm32) targets.
//!
//! Provides ZMTP-framed RPC over WebTransport bidirectional streams,
//! matching the server's `handle_wt_stream` handler in `zmtp_quic.rs`.
//!
//! Implements the `Transport` trait so `RpcClient<JsSigner, WtConnection>`
//! works identically to `RpcClient<LocalSigner, ZmqConnection>`.
//!
//! Requires `RUSTFLAGS='--cfg=web_sys_unstable_apis'` for WebTransport API.

#![cfg(target_arch = "wasm32")]

use std::pin::Pin;
use std::task::{Context, Poll};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::Future;
use futures::Stream;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::transport_traits::{PublishSink, Transport};
use crate::zmtp_framing;

/// Transport ceiling applied when the caller supplies no request deadline —
/// the same 30s family default `RpcClientImpl::calculate_timeout` returns
/// for calls without caller options (rpc_client.rs). A caller deadline
/// replaces it verbatim: a non-negative deadline is honored as-is (never
/// clamped to the default), and a negative value expires immediately rather
/// than being clamped up.
const DEFAULT_REQUEST_TIMEOUT_MS: i32 = 30_000;

/// Cleanup BUDGET, measured FROM the request deadline, shared by the whole
/// expired-request cleanup: the stream-reset observation and the
/// abandoned-read observation are both bounded by the SAME absolute instant
/// — deadline + `CLEANUP_GRACE_MS` — so the caller's error is allocated at
/// most `CLEANUP_GRACE_MS` of cleanup work past the deadline, never a
/// per-step sum. This bounds allocated cleanup work, not wall-clock
/// delivery: browser event-loop suspension (background-tab timer
/// throttling, a busy main thread) can stretch real time beyond it.
///
/// `doc(hidden)` pub so the browser behavioral tests derive their bound
/// from this constant instead of hardcoding it.
#[doc(hidden)]
pub const CLEANUP_GRACE_MS: i64 = 50;

/// Await `fut` bounded by `deadline` (a monotonic `web_time::Instant`).
///
/// `Err` means the deadline expired before the future resolved; `fut` is
/// dropped at that point. Every await inside [`WtConnection::request`] goes
/// through this so a stalled stream can never hang a browser RPC past its
/// deadline.
///
/// The timer is `web_async::time` — the browser timer via `wasmtimer` on
/// WASM, `tokio::time` on native — and it is DISPOSED when the future
/// completes first or the bound future is dropped: no retained
/// `setTimeout` callbacks outlive the await.
///
/// `pub` + `doc(hidden)` so the browser behavioral tests in
/// `tests/wasm_deadline_behavior.rs` drive the exact production function;
/// not part of the supported SDK contract.
#[doc(hidden)]
pub async fn bounded<F: Future>(fut: F, deadline: web_time::Instant) -> Result<F::Output> {
    let remaining = deadline.saturating_duration_since(web_time::Instant::now());
    web_async::time::timeout(remaining, fut)
        .await
        .map_err(|_| anyhow!("WebTransport RPC request deadline exceeded"))
}

/// Production cleanup for an expired read.
///
/// Requirement: the reset must happen while the reader is still associated
/// with the stream. This function therefore does exactly one thing to the
/// stream — `reader.cancel()` — before anything else touches or drops the
/// reader, and performs no lock release of its own.
///
/// Both observations — the stream reset and the abandoned read settlement —
/// are bounded by the SAME absolute instant, the request deadline plus
/// [`CLEANUP_GRACE_MS`]: the caller's error is allocated at most that one
/// shared cleanup budget past the deadline (allocated work, not a
/// wall-clock delivery guarantee), while still reporting whether the stream
/// actually reset. The browser behavioral tests in
/// `tests/wasm_deadline_behavior.rs` drive this exact function and assert
/// those outcomes in a real browser.
///
/// `doc(hidden)` pub for those tests; not part of the supported SDK
/// contract.
#[doc(hidden)]
pub async fn cancel_reader_on_expiry(
    reader: &web_sys::ReadableStreamDefaultReader,
    abandoned_read: Option<js_sys::Promise>,
    base: anyhow::Error,
    deadline: web_time::Instant,
) -> anyhow::Error {
    let cleanup_deadline = deadline + web_time::Duration::from_millis(CLEANUP_GRACE_MS as u64);
    let result = reset_with_grace(reader.cancel(), base, cleanup_deadline).await;
    if let Some(read) = abandoned_read {
        let _ = bounded(JsFuture::from(read), cleanup_deadline).await;
    }
    result
}

/// Observe a stream-reset promise under a short cleanup deadline and fold
/// any failure into the surfaced error — a reset that did not settle is
/// never silently swallowed. The cleanup budget can extend the OBSERVED
/// completion of a deadline-expired call past its deadline; the deadline
/// itself governs the request, the budget only bounds the reset
/// observation.
async fn reset_with_grace(
    reset: js_sys::Promise,
    base: anyhow::Error,
    deadline: web_time::Instant,
) -> anyhow::Error {
    match bounded(JsFuture::from(reset), deadline).await {
        Ok(Ok(_)) => base,
        Ok(Err(rejection)) => {
            anyhow::anyhow!(
                "{base}; stream reset rejected: {}",
                rejection
                    .as_string()
                    .unwrap_or_else(|| "unknown JS rejection".to_owned())
            )
        }
        Err(_) => anyhow::anyhow!(
            "{base}; stream reset did not settle within the {CLEANUP_GRACE_MS}ms cleanup budget"
        ),
    }
}

/// Browser-side WebTransport connection to a hyprstream server.
///
/// Renamed from `WtClient` for consistency with `ZmqConnection`.
pub struct WtConnection {
    wt: web_sys::WebTransport,
    tls_url: String,
    tls_hashes: Vec<[u8; 32]>,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WtConnection {}
unsafe impl Sync for WtConnection {}

impl WtConnection {
    /// Connect to a hyprstream WebTransport endpoint.
    pub async fn connect(url: &str, cert_hash: Option<&str>) -> Result<Self> {
        let cert_hashes = match cert_hash {
            Some(hash) => {
                vec![
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, hash)
                        .map_err(|e| anyhow!("invalid cert hash: {}", e))?
                        .try_into()
                        .map_err(|_| anyhow!("certificate hash must be 32 bytes"))?,
                ]
            }
            None => Vec::new(),
        };
        Self::connect_with_certificate_hashes(url, &cert_hashes).await
    }

    /// Connect using raw accepted-current certificate hashes.
    pub async fn connect_with_certificate_hashes(
        url: &str,
        cert_hashes: &[[u8; 32]],
    ) -> Result<Self> {
        let opts = web_sys::WebTransportOptions::new();
        if !cert_hashes.is_empty() {
            let mut js_hashes = Vec::with_capacity(cert_hashes.len());
            for hash_bytes in cert_hashes {
                let js_hash = web_sys::WebTransportHash::new();
                js_hash.set_algorithm("sha-256");
                let js_bytes = js_sys::Uint8Array::from(&hash_bytes[..]);
                js_hash.set_value(&js_bytes.buffer());
                js_hashes.push(js_hash);
            }

            opts.set_server_certificate_hashes(&js_hashes);
        }

        let wt = web_sys::WebTransport::new_with_options(url, &opts)
            .map_err(|e| anyhow!("WebTransport::new failed: {:?}", e))?;

        JsFuture::from(wt.ready())
            .await
            .map_err(|e| anyhow!("WebTransport connect failed: {:?}", e))?;

        Ok(Self {
            wt,
            tls_url: url.to_owned(),
            tls_hashes: cert_hashes.to_vec(),
        })
    }

    /// Effective TLS policy used by this connection. Browser provisioning
    /// validates it before dialing; stream consumers may reuse it only after
    /// matching the signed stream destination to this exact authority.
    pub fn tls_policy(&self) -> (&str, &[[u8; 32]]) {
        (&self.tls_url, &self.tls_hashes)
    }

    /// Send a REQ/REP request. Returns response payload bytes.
    ///
    /// `timeout_ms` bounds the whole request — stream setup, write, and the
    /// response read loop. `None` applies [`DEFAULT_REQUEST_TIMEOUT_MS`]; a
    /// caller deadline replaces the default verbatim (never clamped to it;
    /// a negative value expires immediately). On expiry the bidirectional
    /// stream is reset so the server observes the abort (`reader.cancel()`
    /// — a direct `ReadableStream::cancel` while the reader holds the lock
    /// rejects; asserted by the browser behavioral tests) and an error is
    /// returned: no await here can hang a browser RPC past its deadline.
    /// The one tail: the expiry cleanup is allocated a single shared
    /// [`CLEANUP_GRACE_MS`] budget past the deadline for observing the reset
    /// and the abandoned read (allocated cleanup work, not a wall-clock
    /// delivery guarantee).
    pub async fn request(&self, payload: &[u8], timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        let started = web_time::Instant::now();
        let deadline = started
            + web_time::Duration::from_millis(
                timeout_ms.unwrap_or(DEFAULT_REQUEST_TIMEOUT_MS).max(0) as u64,
            );
        let deadline_error = || {
            anyhow!(
                "WebTransport RPC request deadline exceeded after {}ms",
                (web_time::Instant::now() - started).as_millis()
            )
        };

        // Accepted gap: expiry during stream setup drops the just-created
        // bidirectional stream without an explicit reset — there is no
        // reader/writer yet to cancel it through, and the WebTransport
        // session reclaims abandoned streams when the connection closes.
        let (writable, readable) = match bounded(self.create_request_streams(), deadline).await {
            Ok(Ok(streams)) => streams,
            Ok(Err(error)) => return Err(error),
            Err(_) => return Err(deadline_error()),
        };

        // Build ZMTP command + multipart
        let cmd = zmtp_framing::encode_command("STREAM_TYPE", b"REQ");
        let msg = zmtp_framing::encode_multipart(&[payload]);
        let mut combined = Vec::with_capacity(cmd.len() + msg.len());
        combined.extend_from_slice(&cmd);
        combined.extend_from_slice(&msg);

        // Write
        let write = async {
            let writer = writable
                .get_writer()
                .map_err(|e| anyhow!("getWriter: {:?}", e))?;
            let js_data = js_sys::Uint8Array::from(&combined[..]);
            JsFuture::from(writer.write_with_chunk(&js_data))
                .await
                .map_err(|e| anyhow!("write: {:?}", e))?;
            JsFuture::from(writer.ready())
                .await
                .map_err(|e| anyhow!("writer.ready: {:?}", e))?;
            // Best-effort close: the request bytes are already written, and a
            // close failure does not invalidate the exchange — the read loop's
            // deadline still bounds the call.
            let _ = JsFuture::from(writer.close()).await;
            writer.release_lock();
            Ok::<(), anyhow::Error>(())
        };
        match bounded(write, deadline).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                // Reset the stream before surfacing the write failure. The
                // readable side is not locked in this phase (only the writer
                // holds the writable lock), so stream cancel reaches the
                // underlying source and aborts both directions.
                return Err(reset_with_grace(readable.cancel(), error, deadline).await);
            }
            Err(_) => {
                return Err(reset_with_grace(readable.cancel(), deadline_error(), deadline).await);
            }
        }

        // Read response
        let reader: web_sys::ReadableStreamDefaultReader = readable.get_reader().unchecked_into();
        let mut chunks: Vec<u8> = Vec::new();

        loop {
            // Keep the read promise: if the deadline wins, the cleanup
            // observes its settlement so it cannot become an unhandled
            // rejection.
            let read_promise = reader.read();
            let result = match bounded(JsFuture::from(read_promise.clone()), deadline).await {
                Ok(result) => result.map_err(|e| anyhow!("read: {:?}", e))?,
                Err(_) => {
                    return Err(cancel_reader_on_expiry(
                        &reader,
                        Some(read_promise),
                        deadline_error(),
                        deadline,
                    )
                    .await);
                }
            };
            let done = js_sys::Reflect::get(&result, &JsValue::from_str("done"))
                .unwrap_or(JsValue::TRUE)
                .as_bool()
                .unwrap_or(true);
            if done {
                break;
            }
            let value = js_sys::Reflect::get(&result, &JsValue::from_str("value"))
                .map_err(|_| anyhow!("no value"))?;
            let chunk = js_sys::Uint8Array::from(value);
            let mut buf = vec![0u8; chunk.length() as usize];
            chunk.copy_to(&mut buf);
            chunks.extend_from_slice(&buf);
        }
        reader.release_lock();

        // Decode ZMTP multipart — first frame is the response
        let (frames, _consumed) = zmtp_framing::decode_multipart(&chunks)?;
        if frames.is_empty() {
            return Err(anyhow!("empty ZMTP response"));
        }
        Ok(frames[0].clone())
    }

    /// Create the bidirectional stream pair for one REQ/REP exchange.
    async fn create_request_streams(
        &self,
    ) -> Result<(web_sys::WritableStream, web_sys::ReadableStream)> {
        let bidi = JsFuture::from(self.wt.create_bidirectional_stream())
            .await
            .map_err(|e| anyhow!("createBidirectionalStream: {:?}", e))?;
        let writable: web_sys::WritableStream =
            js_sys::Reflect::get(&bidi, &JsValue::from_str("writable"))
                .map_err(|_| anyhow!("no writable"))?
                .unchecked_into();
        let readable: web_sys::ReadableStream =
            js_sys::Reflect::get(&bidi, &JsValue::from_str("readable"))
                .map_err(|_| anyhow!("no readable"))?
                .unchecked_into();
        Ok((writable, readable))
    }

    /// Open a SUB stream (internal, returns WtSubscriber).
    async fn open_subscriber(&self, topic: &[u8]) -> Result<WtSubscriber> {
        let bidi = JsFuture::from(self.wt.create_bidirectional_stream())
            .await
            .map_err(|e| anyhow!("createBidirectionalStream: {:?}", e))?;

        let writable: web_sys::WritableStream =
            js_sys::Reflect::get(&bidi, &JsValue::from_str("writable"))
                .map_err(|_| anyhow!("no writable"))?
                .unchecked_into();
        let readable: web_sys::ReadableStream =
            js_sys::Reflect::get(&bidi, &JsValue::from_str("readable"))
                .map_err(|_| anyhow!("no readable"))?
                .unchecked_into();

        // Send STREAM_TYPE=SUB + topic
        let cmd = zmtp_framing::encode_command("STREAM_TYPE", b"SUB");
        let msg = zmtp_framing::encode_multipart(&[topic]);
        let mut combined = Vec::with_capacity(cmd.len() + msg.len());
        combined.extend_from_slice(&cmd);
        combined.extend_from_slice(&msg);

        let writer = writable
            .get_writer()
            .map_err(|e| anyhow!("getWriter: {:?}", e))?;
        let js_data = js_sys::Uint8Array::from(&combined[..]);
        JsFuture::from(writer.write_with_chunk(&js_data))
            .await
            .map_err(|e| anyhow!("write: {:?}", e))?;
        let _ = JsFuture::from(writer.close()).await;
        writer.release_lock();

        let reader: web_sys::ReadableStreamDefaultReader = readable.get_reader().unchecked_into();
        let (tx, rx) = futures::channel::mpsc::unbounded();

        // Spawn a reader task that feeds frames into the channel
        let inner = WtSubStreamInner {
            reader,
            buffer: Vec::new(),
        };
        wasm_bindgen_futures::spawn_local(async move {
            let mut inner = inner;
            loop {
                match inner.next_frames().await {
                    Ok(Some(frames)) => {
                        if tx.unbounded_send(Ok(frames)).is_err() {
                            break; // receiver dropped
                        }
                    }
                    Ok(None) => break, // stream ended
                    Err(e) => {
                        let _ = tx.unbounded_send(Err(e));
                        break;
                    }
                }
            }
        });

        Ok(WtSubscriber { rx })
    }

    /// Open a PUB stream (STREAM_TYPE=PUB) for control messages.
    async fn open_publisher(&self, topic: &[u8]) -> Result<WtPublisher> {
        let bidi = JsFuture::from(self.wt.create_bidirectional_stream())
            .await
            .map_err(|e| anyhow!("createBidirectionalStream: {:?}", e))?;

        let writable: web_sys::WritableStream =
            js_sys::Reflect::get(&bidi, &JsValue::from_str("writable"))
                .map_err(|_| anyhow!("no writable"))?
                .unchecked_into();

        // Send STREAM_TYPE=PUB + topic (keep writable open for subsequent messages)
        let cmd = zmtp_framing::encode_command("STREAM_TYPE", b"PUB");
        let msg = zmtp_framing::encode_multipart(&[topic]);
        let mut combined = Vec::with_capacity(cmd.len() + msg.len());
        combined.extend_from_slice(&cmd);
        combined.extend_from_slice(&msg);

        let writer = writable
            .get_writer()
            .map_err(|e| anyhow!("getWriter: {:?}", e))?;
        let js_data = js_sys::Uint8Array::from(&combined[..]);
        JsFuture::from(writer.write_with_chunk(&js_data))
            .await
            .map_err(|e| anyhow!("write: {:?}", e))?;
        // Do NOT close the writer — keep it open for sending ctrl messages
        writer.release_lock();

        Ok(WtPublisher { writable })
    }

    pub fn close(&self) {
        self.wt.close();
    }
}

// ============================================================================
// Transport impl
// ============================================================================

#[async_trait(?Send)]
impl Transport for WtConnection {
    type Sub = WtSubscriber;
    type Pub = WtPublisher;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        self.request(&payload, timeout_ms).await
    }

    fn forbids_cleartext_envelope(&self) -> bool {
        true
    }

    async fn subscribe(&self, topic: &[u8]) -> Result<WtSubscriber> {
        self.open_subscriber(topic).await
    }

    async fn publish(&self, topic: &[u8]) -> Result<WtPublisher> {
        self.open_publisher(topic).await
    }
}

// ============================================================================
// WtSubscriber — implements futures::Stream<Item = Result<Vec<Vec<u8>>>>
// ============================================================================

/// Internal reader that pulls ZMTP frames from a WebTransport readable stream.
struct WtSubStreamInner {
    reader: web_sys::ReadableStreamDefaultReader,
    buffer: Vec<u8>,
}

impl WtSubStreamInner {
    async fn next_frames(&mut self) -> Result<Option<Vec<Vec<u8>>>> {
        loop {
            if !self.buffer.is_empty() {
                if let Ok((frames, consumed)) = zmtp_framing::decode_multipart(&self.buffer) {
                    if !frames.is_empty() {
                        self.buffer = self.buffer[consumed..].to_vec();
                        return Ok(Some(frames));
                    }
                }
            }

            let result = JsFuture::from(self.reader.read())
                .await
                .map_err(|e| anyhow!("read: {:?}", e))?;
            let done = js_sys::Reflect::get(&result, &JsValue::from_str("done"))
                .unwrap_or(JsValue::TRUE)
                .as_bool()
                .unwrap_or(true);
            if done {
                return Ok(None);
            }

            let value = js_sys::Reflect::get(&result, &JsValue::from_str("value"))
                .map_err(|_| anyhow!("no value"))?;
            let chunk = js_sys::Uint8Array::from(value);
            let mut buf = vec![0u8; chunk.length() as usize];
            chunk.copy_to(&mut buf);
            self.buffer.extend_from_slice(&buf);
        }
    }
}

/// WebTransport subscriber that implements `futures::Stream`.
///
/// Backed by a spawned reader task that feeds frames through an unbounded channel.
pub struct WtSubscriber {
    rx: futures::channel::mpsc::UnboundedReceiver<Result<Vec<Vec<u8>>>>,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WtSubscriber {}

impl Stream for WtSubscriber {
    type Item = Result<Vec<Vec<u8>>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_next(cx)
    }
}

impl Unpin for WtSubscriber {}

// ============================================================================
// WtPublisher — sends ZMTP multipart on a PUB stream
// ============================================================================

/// WebTransport publisher for control channel (STREAM_TYPE=PUB).
///
/// Keeps the writable side open for sending `[ctrl_topic, capnp, mac]` messages.
pub struct WtPublisher {
    writable: web_sys::WritableStream,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WtPublisher {}
unsafe impl Sync for WtPublisher {}

#[async_trait(?Send)]
impl PublishSink for WtPublisher {
    async fn send_frames(&self, frames: &[&[u8]]) -> Result<()> {
        let encoded = zmtp_framing::encode_multipart(frames);
        let writer = self
            .writable
            .get_writer()
            .map_err(|e| anyhow!("getWriter: {:?}", e))?;
        let js_data = js_sys::Uint8Array::from(&encoded[..]);
        JsFuture::from(writer.write_with_chunk(&js_data))
            .await
            .map_err(|e| anyhow!("write: {:?}", e))?;
        writer.release_lock();
        Ok(())
    }
}

/// Legacy alias — use `WtConnection` instead.
pub type WtClient = WtConnection;
