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
use futures::Stream;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::transport_traits::{PublishSink, Transport};
use crate::zmtp_framing;

/// Browser-side WebTransport connection to a hyprstream server.
///
/// Renamed from `WtClient` for consistency with `ZmqConnection`.
pub struct WtConnection {
    wt: web_sys::WebTransport,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WtConnection {}
unsafe impl Sync for WtConnection {}

impl WtConnection {
    /// Connect to a hyprstream WebTransport endpoint.
    pub async fn connect(url: &str, cert_hash: Option<&str>) -> Result<Self> {
        let opts = web_sys::WebTransportOptions::new();

        if let Some(hash) = cert_hash {
            let hash_bytes = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD, hash
            ).map_err(|e| anyhow!("invalid cert hash: {}", e))?;

            let js_hash = web_sys::WebTransportHash::new();
            js_hash.set_algorithm("sha-256");
            let js_bytes = js_sys::Uint8Array::from(&hash_bytes[..]);
            js_hash.set_value(&js_bytes.buffer());

            opts.set_server_certificate_hashes(&[js_hash]);
        }

        let wt = web_sys::WebTransport::new_with_options(url, &opts)
            .map_err(|e| anyhow!("WebTransport::new failed: {:?}", e))?;

        JsFuture::from(wt.ready()).await
            .map_err(|e| anyhow!("WebTransport connect failed: {:?}", e))?;

        Ok(Self { wt })
    }

    /// Send a REQ/REP request. Returns response payload bytes.
    pub async fn request(&self, payload: &[u8]) -> Result<Vec<u8>> {
        let bidi = JsFuture::from(self.wt.create_bidirectional_stream()).await
            .map_err(|e| anyhow!("createBidirectionalStream: {:?}", e))?;

        let writable: web_sys::WritableStream = js_sys::Reflect::get(&bidi, &JsValue::from_str("writable"))
            .map_err(|_| anyhow!("no writable"))?
            .unchecked_into();
        let readable: web_sys::ReadableStream = js_sys::Reflect::get(&bidi, &JsValue::from_str("readable"))
            .map_err(|_| anyhow!("no readable"))?
            .unchecked_into();

        // Build ZMTP command + multipart
        let cmd = zmtp_framing::encode_command("STREAM_TYPE", b"REQ");
        let msg = zmtp_framing::encode_multipart(&[payload]);
        let mut combined = Vec::with_capacity(cmd.len() + msg.len());
        combined.extend_from_slice(&cmd);
        combined.extend_from_slice(&msg);

        // Write
        let writer = writable.get_writer()
            .map_err(|e| anyhow!("getWriter: {:?}", e))?;
        let js_data = js_sys::Uint8Array::from(&combined[..]);
        JsFuture::from(writer.write_with_chunk(&js_data)).await
            .map_err(|e| anyhow!("write: {:?}", e))?;
        JsFuture::from(writer.ready()).await
            .map_err(|e| anyhow!("writer.ready: {:?}", e))?;
        let _ = JsFuture::from(writer.close()).await;
        writer.release_lock();

        // Read response
        let reader: web_sys::ReadableStreamDefaultReader = readable.get_reader().unchecked_into();
        let mut chunks: Vec<u8> = Vec::new();

        loop {
            let result = JsFuture::from(reader.read()).await
                .map_err(|e| anyhow!("read: {:?}", e))?;
            let done = js_sys::Reflect::get(&result, &JsValue::from_str("done"))
                .unwrap_or(JsValue::TRUE).as_bool().unwrap_or(true);
            if done { break; }
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

    /// Open a SUB stream (internal, returns WtSubscriber).
    async fn open_subscriber(&self, topic: &[u8]) -> Result<WtSubscriber> {
        let bidi = JsFuture::from(self.wt.create_bidirectional_stream()).await
            .map_err(|e| anyhow!("createBidirectionalStream: {:?}", e))?;

        let writable: web_sys::WritableStream = js_sys::Reflect::get(&bidi, &JsValue::from_str("writable"))
            .map_err(|_| anyhow!("no writable"))?
            .unchecked_into();
        let readable: web_sys::ReadableStream = js_sys::Reflect::get(&bidi, &JsValue::from_str("readable"))
            .map_err(|_| anyhow!("no readable"))?
            .unchecked_into();

        // Send STREAM_TYPE=SUB + topic
        let cmd = zmtp_framing::encode_command("STREAM_TYPE", b"SUB");
        let msg = zmtp_framing::encode_multipart(&[topic]);
        let mut combined = Vec::with_capacity(cmd.len() + msg.len());
        combined.extend_from_slice(&cmd);
        combined.extend_from_slice(&msg);

        let writer = writable.get_writer()
            .map_err(|e| anyhow!("getWriter: {:?}", e))?;
        let js_data = js_sys::Uint8Array::from(&combined[..]);
        JsFuture::from(writer.write_with_chunk(&js_data)).await
            .map_err(|e| anyhow!("write: {:?}", e))?;
        let _ = JsFuture::from(writer.close()).await;
        writer.release_lock();

        let reader: web_sys::ReadableStreamDefaultReader = readable.get_reader().unchecked_into();
        let (tx, rx) = futures::channel::mpsc::unbounded();

        // Spawn a reader task that feeds frames into the channel
        let inner = WtSubStreamInner { reader, buffer: Vec::new() };
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
        let bidi = JsFuture::from(self.wt.create_bidirectional_stream()).await
            .map_err(|e| anyhow!("createBidirectionalStream: {:?}", e))?;

        let writable: web_sys::WritableStream = js_sys::Reflect::get(&bidi, &JsValue::from_str("writable"))
            .map_err(|_| anyhow!("no writable"))?
            .unchecked_into();

        // Send STREAM_TYPE=PUB + topic (keep writable open for subsequent messages)
        let cmd = zmtp_framing::encode_command("STREAM_TYPE", b"PUB");
        let msg = zmtp_framing::encode_multipart(&[topic]);
        let mut combined = Vec::with_capacity(cmd.len() + msg.len());
        combined.extend_from_slice(&cmd);
        combined.extend_from_slice(&msg);

        let writer = writable.get_writer()
            .map_err(|e| anyhow!("getWriter: {:?}", e))?;
        let js_data = js_sys::Uint8Array::from(&combined[..]);
        JsFuture::from(writer.write_with_chunk(&js_data)).await
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

    async fn send(&self, payload: Vec<u8>, _timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        self.request(&payload).await
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

            let result = JsFuture::from(self.reader.read()).await
                .map_err(|e| anyhow!("read: {:?}", e))?;
            let done = js_sys::Reflect::get(&result, &JsValue::from_str("done"))
                .unwrap_or(JsValue::TRUE).as_bool().unwrap_or(true);
            if done { return Ok(None); }

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
        let writer = self.writable.get_writer()
            .map_err(|e| anyhow!("getWriter: {:?}", e))?;
        let js_data = js_sys::Uint8Array::from(&encoded[..]);
        JsFuture::from(writer.write_with_chunk(&js_data)).await
            .map_err(|e| anyhow!("write: {:?}", e))?;
        writer.release_lock();
        Ok(())
    }
}

/// Legacy alias — use `WtConnection` instead.
pub type WtClient = WtConnection;
