//! WebTransport client for browser (wasm32) targets.
//!
//! Provides ZMTP-framed RPC over WebTransport bidirectional streams,
//! matching the server's `handle_wt_stream` handler in `zmtp_quic.rs`.
//!
//! Requires `RUSTFLAGS='--cfg=web_sys_unstable_apis'` for WebTransport API.

#![cfg(target_arch = "wasm32")]

use anyhow::{anyhow, Result};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::zmtp_framing;

/// Browser-side WebTransport connection to a hyprstream server.
pub struct WtClient {
    wt: web_sys::WebTransport,
}

impl WtClient {
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

            let hashes = js_sys::Array::new();
            hashes.push(&js_hash);
            opts.set_server_certificate_hashes(&hashes);
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

        // Get writable/readable via JS property access (web-sys may not have typed accessors)
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

    /// Open a SUB stream.
    pub async fn subscribe(&self, topic: &[u8]) -> Result<SubStream> {
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
        Ok(SubStream { reader, buffer: Vec::new() })
    }

    pub fn close(&self) {
        self.wt.close();
    }
}

/// Subscription stream yielding ZMTP multipart messages.
pub struct SubStream {
    reader: web_sys::ReadableStreamDefaultReader,
    buffer: Vec<u8>,
}

impl SubStream {
    /// Read next block. Returns data frames (topic stripped). None on stream end.
    pub async fn next(&mut self) -> Result<Option<Vec<Vec<u8>>>> {
        loop {
            if !self.buffer.is_empty() {
                if let Ok((frames, consumed)) = zmtp_framing::decode_multipart(&self.buffer) {
                    if !frames.is_empty() {
                        // Skip topic frame (index 0), return data frames
                        let data_frames = if frames.len() >= 3 {
                            frames[1..].to_vec()
                        } else {
                            frames
                        };
                        self.buffer = self.buffer[consumed..].to_vec();
                        return Ok(Some(data_frames));
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
