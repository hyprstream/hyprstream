//! RpcClient — wasm-bindgen export of `RpcClient<JsSigner, WtConnection>`.
//!
//! This is the JS-facing RPC client. Exported as `RpcClient` to JavaScript
//! (matching the Rust generic name). Internally wraps the concrete
//! `RpcClient<JsSigner, WtConnection>` parameterization.
//!
//! The old `RpcSession` in `wasm_exports.rs` is the legacy equivalent.
//! Once all consumers migrate, `RpcSession` will be deleted.

#![cfg(target_arch = "wasm32")]

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use hyprstream_rpc::crypto::VerifyingKey;
use hyprstream_rpc::rpc_client::{RpcClientImpl, RpcClient};
use hyprstream_rpc::signer::JsSigner;
use hyprstream_rpc::stream_consumer::{StreamHandle, StreamHandleImpl, StreamPayload};
use hyprstream_rpc::web_transport::WtConnection;

/// Unified RPC client exported to JavaScript as `RpcClient`.
///
/// Wraps `RpcClient<JsSigner, WtConnection>` — same envelope construction,
/// signing, and response verification as the native `RpcClient<LocalSigner, ZmqConnection>`.
///
/// TypeScript consumers use this via generated client classes that call
/// `client.call(payload)` with Cap'n Proto bytes.
#[wasm_bindgen(js_name = "RpcClient")]
pub struct WasmRpcClient {
    inner: RpcClientImpl<JsSigner, WtConnection>,
}

#[wasm_bindgen(js_class = "RpcClient")]
impl WasmRpcClient {
    /// Connect to a hyprstream WebTransport endpoint with an external signer.
    ///
    /// - `url`: WebTransport URL (e.g., `https://host:port/wt`)
    /// - `cert_hash`: Optional base64-encoded SHA-256 certificate hash for pinning
    /// - `signer_pubkey`: 32-byte Ed25519 public key for envelope signing
    /// - `sign_fn`: JavaScript async function `(canonicalBytes: Uint8Array) => Promise<Uint8Array>`
    /// - `server_verifying_key`: 32-byte Ed25519 public key for response verification
    #[wasm_bindgen(constructor)]
    pub async fn connect(
        url: &str,
        cert_hash: Option<String>,
        signer_pubkey: &[u8],
        sign_fn: js_sys::Function,
        server_verifying_key: &[u8],
    ) -> Result<WasmRpcClient, JsError> {
        let transport = WtConnection::connect(url, cert_hash.as_deref())
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let signer = JsSigner::new(signer_pubkey, sign_fn)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let server_key = VerifyingKey::from_bytes(
            server_verifying_key.try_into()
                .map_err(|_| JsError::new("server_verifying_key must be 32 bytes"))?
        ).map_err(|e| JsError::new(&format!("invalid server verifying key: {}", e)))?;

        Ok(Self {
            inner: RpcClientImpl::new(signer, transport, server_key),
        })
    }

    /// Set opaque JWT token for authenticated requests. Server decodes and verifies.
    #[wasm_bindgen(js_name = "setJwt")]
    pub fn set_jwt(&self, token: &str) {
        self.inner.set_jwt(if token.is_empty() {
            None
        } else {
            Some(token.to_string())
        });
    }

    /// Send a signed request and return the verified response payload (Cap'n Proto bytes).
    pub async fn call(&self, payload: &[u8]) -> Result<Vec<u8>, JsError> {
        self.inner.call(payload.to_vec())
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Send a streaming request with ephemeral DH pubkey.
    #[wasm_bindgen(js_name = "callStreaming")]
    pub async fn call_streaming(
        &self,
        payload: &[u8],
        ephemeral_pubkey: &[u8],
    ) -> Result<Vec<u8>, JsError> {
        let mut epk = [0u8; 32];
        if ephemeral_pubkey.len() != 32 {
            return Err(JsError::new("ephemeral_pubkey must be 32 bytes"));
        }
        epk.copy_from_slice(ephemeral_pubkey);

        self.inner.call_streaming(payload.to_vec(), epk)
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the next request ID.
    #[wasm_bindgen(js_name = "nextId")]
    pub fn next_id(&self) -> u64 {
        self.inner.next_id()
    }

    /// Close the WebTransport connection.
    pub fn close(&self) {
        self.inner.transport.close();
    }

    /// Open a verified streaming subscription.
    #[wasm_bindgen(js_name = "openStream")]
    pub async fn open_stream(&self, payload: &[u8]) -> Result<WasmStreamHandle, JsError> {
        let handle = self.inner.open_stream(payload.to_vec())
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmStreamHandle { inner: handle })
    }
}

/// Verified stream handle exported to JavaScript as `StreamHandle`.
///
/// Wraps `StreamHandleImpl<WtConnection>` — same HMAC verification, same
/// Cap'n Proto parsing as the native `StreamHandle<ZmqConnection>`.
#[wasm_bindgen(js_name = "StreamHandle")]
pub struct WasmStreamHandle {
    inner: StreamHandleImpl<WtConnection>,
}

#[wasm_bindgen(js_class = "StreamHandle")]
impl WasmStreamHandle {
    /// Get next verified payload as raw bytes.
    ///
    /// Returns the payload data bytes for Data/Complete variants,
    /// null on stream end, or throws on error.
    #[wasm_bindgen(js_name = "nextPayload")]
    pub async fn next_payload(&mut self) -> Result<JsValue, JsError> {
        match self.inner.next_payload().await {
            Ok(Some(StreamPayload::Data(data))) => {
                Ok(js_sys::Uint8Array::from(&data[..]).into())
            }
            Ok(Some(StreamPayload::Complete(meta))) => {
                // Return completion metadata; caller checks via is_completed()
                Ok(js_sys::Uint8Array::from(&meta[..]).into())
            }
            Ok(Some(StreamPayload::Error(msg))) => {
                Err(JsError::new(&msg))
            }
            Ok(Some(StreamPayload::Tagged { payload, .. })) => {
                Ok(js_sys::Uint8Array::from(&payload[..]).into())
            }
            Ok(None) => Ok(JsValue::NULL),
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }

    /// Cancel the stream via authenticated ctrl channel.
    pub async fn cancel(&self) -> Result<(), JsError> {
        self.inner.cancel().await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the stream ID.
    #[wasm_bindgen(js_name = "streamId")]
    pub fn stream_id(&self) -> String {
        self.inner.stream_id().to_owned()
    }

    /// Check if stream is completed.
    #[wasm_bindgen(js_name = "isCompleted")]
    pub fn is_completed(&self) -> bool {
        self.inner.is_completed()
    }
}
