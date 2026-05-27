//! RpcClient — wasm-bindgen export of `RpcClientImpl<JsSigner, WtConnection>`.
//!
//! This is the JS-facing RPC client. Exported as `RpcClient` to JavaScript
//! (matching the Rust generic name). Internally wraps the concrete
//! `RpcClient<JsSigner, WtConnection>` parameterization.
//!
//! The WASM shim mirrors the native Rust API:
//! - Layered construction: `new WtConnection()` + `new RpcClient(conn, ...)`
//! - Immutable JWT via `with_default_jwt()` builder (no `set_jwt` mutation)
//! - Per-call override via `call_with_options()`
//! - Streaming via `callStreaming()` / `openStream()`

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

use hyprstream_rpc::crypto::VerifyingKey;
use hyprstream_rpc::rpc_client::{CallOptions, RpcClientImpl};
use hyprstream_rpc::signer::JsSigner;
use hyprstream_rpc::stream_consumer::{StreamHandle, StreamHandleImpl, StreamPayload};
use hyprstream_rpc::web_transport::WtConnection;

// ============================================================================
// JsTokenProvider — Send+Sync wrapper for a JS token callback
// ============================================================================

/// Wraps a `js_sys::Function` so it satisfies `Send + Sync`.
///
/// SAFETY: WASM is single-threaded; there is no concurrent access possible.
struct JsTokenProvider(js_sys::Function);
unsafe impl Send for JsTokenProvider {}
unsafe impl Sync for JsTokenProvider {}

impl JsTokenProvider {
    fn call(&self) -> Option<String> {
        self.0.call0(&wasm_bindgen::JsValue::NULL).ok().and_then(|v| v.as_string())
    }
}

// ============================================================================
// WtConnection — standalone WebTransport connection (Step 1)
// ============================================================================

/// WebTransport connection exported to JavaScript.
///
/// Constructed separately from the RPC client, matching how native code
/// constructs `ZmqConnection::new(endpoint)` before building a client.
#[wasm_bindgen(js_name = "WtConnection")]
pub struct WasmWtConnection {
    inner: WtConnection,
}

#[wasm_bindgen(js_class = "WtConnection")]
impl WasmWtConnection {
    /// Connect to a hyprstream WebTransport endpoint.
    ///
    /// - `url`: WebTransport URL (e.g., `https://host:port/wt`)
    /// - `cert_hash`: Optional base64-encoded SHA-256 certificate hash for pinning
    #[wasm_bindgen(constructor)]
    pub async fn connect(url: &str, cert_hash: Option<String>) -> Result<WasmWtConnection, JsError> {
        let transport = WtConnection::connect(url, cert_hash.as_deref())
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner: transport })
    }
}

// ============================================================================
// RpcClient — unified RPC client (Steps 2-3)
// ============================================================================

/// Unified RPC client exported to JavaScript as `RpcClient`.
///
/// Wraps `RpcClientImpl<JsSigner, WtConnection>` — same envelope construction,
/// signing, and response verification as the native `RpcClientImpl<LocalSigner, ZmqConnection>`.
///
/// TypeScript consumers use this via generated client classes that call
/// `client.call(payload)` with Cap'n Proto bytes.
///
/// Construction is layered (matches Rust pattern):
/// ```js
/// const conn = await new WtConnection(url, certHash);
/// const client = new RpcClient(conn, signerPubkey, signFn, serverVerifyingKey)
///     .withDefaultJwt(token);
/// ```
#[wasm_bindgen(js_name = "RpcClient")]
pub struct WasmRpcClient {
    inner: RpcClientImpl<JsSigner, WtConnection>,
}

#[wasm_bindgen(js_class = "RpcClient")]
impl WasmRpcClient {
    /// Create a new RPC client with a pre-built transport connection.
    ///
    /// - `connection`: A `WtConnection` (constructed separately, consumed by this call)
    /// - `signer_pubkey`: 32-byte Ed25519 public key for envelope signing
    /// - `sign_fn`: JavaScript async function `(canonicalBytes: Uint8Array) -> Promise<Uint8Array>`
    /// - `server_verifying_key`: Optional 32-byte Ed25519 public key for response verification.
    ///   Pass `null`/`undefined` to skip response signature verification (TLS still protects the connection).
    #[wasm_bindgen(constructor)]
    pub fn new(
        connection: WasmWtConnection,
        signer_pubkey: &[u8],
        sign_fn: js_sys::Function,
        server_verifying_key: Option<Vec<u8>>,
    ) -> Result<WasmRpcClient, JsError> {
        let signer = JsSigner::new(signer_pubkey, sign_fn)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let server_key: Option<VerifyingKey> = match server_verifying_key {
            Some(bytes) => {
                let arr: [u8; 32] = bytes.try_into()
                    .map_err(|_| JsError::new("server_verifying_key must be 32 bytes"))?;
                Some(VerifyingKey::from_bytes(&arr)
                    .map_err(|e| JsError::new(&format!("invalid server verifying key: {}", e)))?)
            }
            None => None,
        };

        Ok(Self {
            inner: RpcClientImpl::new(signer, connection.inner, server_key),
        })
    }

    /// Convenience: one-step connect + construct (backward compat).
    ///
    /// Equivalent to `new WtConnection(url, certHash)` + `new RpcClient(conn, ...)`.
    #[wasm_bindgen(js_name = "connect")]
    pub async fn connect(
        url: &str,
        cert_hash: Option<String>,
        signer_pubkey: &[u8],
        sign_fn: js_sys::Function,
        server_verifying_key: Option<Vec<u8>>,
    ) -> Result<WasmRpcClient, JsError> {
        let conn = WasmWtConnection::connect(url, cert_hash).await?;
        // conn is moved into Self::new()
        Self::new(conn, signer_pubkey, sign_fn, server_verifying_key)
    }

    /// Builder: set a dynamic token provider called on every RPC request.
    ///
    /// `provider` is a JS `() => string | null` function invoked before each call.
    /// This keeps short-lived tokens (OAuth at+jwt, WIT) fresh without reconstructing
    /// the client. Consumes and returns a new client.
    #[wasm_bindgen(js_name = "withTokenProvider")]
    pub fn with_token_provider(self, provider: js_sys::Function) -> WasmRpcClient {
        let wrapped = JsTokenProvider(provider);
        WasmRpcClient {
            inner: self.inner.with_token_provider(move || wrapped.call()),
        }
    }

    /// Builder: set a static default JWT token for all calls.
    ///
    /// Sugar over `withTokenProvider`. Consumes and returns a new client.
    #[wasm_bindgen(js_name = "withDefaultJwt")]
    pub fn with_default_jwt(self, token: &str) -> WasmRpcClient {
        WasmRpcClient {
            inner: self.inner.with_default_jwt(token.to_string()),
        }
    }

    /// Send a signed request and return the verified response payload (Cap'n Proto bytes).
    ///
    /// Uses the client's default JWT (set via `withDefaultJwt()`).
    pub async fn call(&self, payload: &[u8]) -> Result<Vec<u8>, JsError> {
        self.inner.call(payload.to_vec())
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Send a request with per-call authentication options.
    ///
    /// - `payload`: Cap'n Proto request bytes
    /// - `jwt`: Optional per-call JWT override (takes precedence over default)
    /// - `delegated_bearer`: Optional bearer token to relay on behalf of a user
    #[wasm_bindgen(js_name = "callWithOptions")]
    pub async fn call_with_options(
        &self,
        payload: &[u8],
        jwt: Option<String>,
        delegated_bearer: Option<String>,
    ) -> Result<Vec<u8>, JsError> {
        let options = CallOptions::new();
        let options = match jwt {
            Some(t) => options.jwt(t),
            None => options,
        };
        let options = match delegated_bearer {
            Some(b) => options.delegated_bearer(b),
            None => options,
        };
        self.inner.call_with_options(payload.to_vec(), options)
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

// ============================================================================
// StreamHandle — verified stream subscription
// ============================================================================

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
