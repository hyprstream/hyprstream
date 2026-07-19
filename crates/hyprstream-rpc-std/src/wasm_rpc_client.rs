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

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use hyprstream_rpc::browser_provisioning::{
    fetch_browser_provisioning, BrowserCarrierProfile, BrowserProvisioningGuard,
    BrowserProvisioningRequest,
};

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
/// constructs a transport before building a client.
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
    pub async fn connect(
        _url: &str,
        _cert_hash: Option<String>,) -> Result<WasmWtConnection, JsError> {
        Err( JsError::new(
            "unprovisioned browser WebTransport dial is disabled; use RpcClient.connectResolved",))
    }
}

// ============================================================================
// RpcClient — unified RPC client (Steps 2-3)
// ============================================================================

/// Unified RPC client exported to JavaScript as `RpcClient`.
///
/// Wraps `RpcClientImpl<JsSigner, WtConnection>` — same envelope construction,
/// signing, and response verification as the native `RpcClientImpl<LocalSigner, LazyUdsTransport>`.
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

fn call_options(jwt: Option<String>, delegated_bearer: Option<String>) -> CallOptions {
    let options = CallOptions::new();
    let options = match jwt {
        Some(token) => options.jwt(token),
        None => options,
    };
    match delegated_bearer {
        Some(bearer) => options.delegated_bearer(bearer),
        None => options,
    }
}

fn fixed_ephemeral_pubkey(ephemeral_pubkey: &[u8]) -> Result<[u8; 32], JsError> {
    ephemeral_pubkey
        .try_into()
        .map_err(|_| JsError::new("ephemeral_pubkey must be 32 bytes"))
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
                    .map_err(|e| { JsError::new(&format!("invalid server verifying key: {}", e))
                    })?)
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

    /// Resolve accepted-current authority, then dial and seal only to the
    /// bound owned WebTransport reach. Every call re-fetches before sealing.
    #[wasm_bindgen(js_name = "connectResolved")]
    pub async fn connect_resolved(
        provisioning_origin: &str,
        service_name: &str,
        signer_pubkey: &[u8],
        sign_fn: js_sys::Function,
        signer_ml_dsa65_pubkey: &[u8],
        pq_sign_fn: js_sys::Function,
        jwt: Option<String>,
    ) -> Result<WasmRpcClient, JsError> {
        let expected = BrowserProvisioningRequest::new(
            service_name,
            "hyprstream-rpc/1",
            service_name,
            BrowserCarrierProfile::OwnedHybridWebTransport,
        )
        .map_err(|error| JsError::new(&error.to_string()))?;
        let provisioned = fetch_browser_provisioning(provisioning_origin, &expected)
            .await
            .map_err(|error| JsError::new(&error.to_string()))?;
        let transport = WtConnection::connect_with_certificate_hashes(
            provisioned.webtransport_url(),
            provisioned.certificate_hashes(),
        )
        .await
        .map_err(|error| JsError::new(&error.to_string()))?;
        let signer =
            JsSigner::new_hybrid(signer_pubkey, sign_fn, signer_ml_dsa65_pubkey, pq_sign_fn)
                .map_err(|error| JsError::new(&error.to_string()))?;
        let (kem, pq) = provisioned
            .crypto_stores()
            .map_err(|error| JsError::new(&error.to_string()))?;
        let guard =
            BrowserProvisioningGuard::new(provisioning_origin, expected, provisioned.clone());
        let binding = provisioned
            .request_binding()
            .map_err(|error| JsError::new(&error.to_string()))?;
        let client =
            RpcClientImpl::new(signer, transport, Some(provisioned.server_verifying_key()))
                .with_request_kem_store(kem)
                .with_response_pq_store(pq)
                .with_pre_seal_guard(Arc::new(guard))
                .with_browser_provisioning_binding(binding)
                .map_err(|error| JsError::new(&error.to_string()))?;
        let client = match jwt {
            Some(token) => client.with_default_jwt(token),
            None => client,
        };
        Ok(Self { inner: client })
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

    /// Send a generated request with explicit canonical service and method metadata.
    #[wasm_bindgen(js_name = "callForServiceWithMethod")]
    pub async fn call_for_service_with_method(
        &self,
        service_name: &str,
        method_discriminator: u16,
        payload: &[u8],
    ) -> Result<Vec<u8>, JsError> {
        self.inner
            .call_for_service_with_method(
                service_name,
                method_discriminator,
                payload.to_vec(),
            )
            .await
            .map_err(|error| JsError::new(&error.to_string()))
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
        let options = call_options(jwt, delegated_bearer);
        self.inner.call_with_options(payload.to_vec(), options)
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Send an option-bearing generated request with explicit service and method metadata.
    #[wasm_bindgen(js_name = "callWithOptionsForServiceWithMethod")]
    pub async fn call_with_options_for_service_with_method(
        &self,
        service_name: &str,
        method_discriminator: u16,
        payload: &[u8],
        jwt: Option<String>,
        delegated_bearer: Option<String>,
    ) -> Result<Vec<u8>, JsError> {
        let options = call_options(jwt, delegated_bearer);
        self.inner
            .call_with_options_for_service_with_method(
                service_name,
                method_discriminator,
                payload.to_vec(),
                options,
            )
            .await
            .map_err(|error| JsError::new(&error.to_string()))
    }

    /// Send a streaming request with ephemeral DH pubkey.
    #[wasm_bindgen(js_name = "callStreaming")]
    pub async fn call_streaming(
        &self,
        payload: &[u8],
        ephemeral_pubkey: &[u8],
    ) -> Result<Vec<u8>, JsError> {
        let epk = fixed_ephemeral_pubkey(ephemeral_pubkey)?;
        self.inner.call_streaming(payload.to_vec(), epk)
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Send a generated streaming request with explicit service and method metadata.
    #[wasm_bindgen(js_name = "callStreamingForServiceWithMethod")]
    pub async fn call_streaming_for_service_with_method(
        &self,
        service_name: &str,
        method_discriminator: u16,
        payload: &[u8],
        ephemeral_pubkey: &[u8],
    ) -> Result<Vec<u8>, JsError> {
        let epk = fixed_ephemeral_pubkey(ephemeral_pubkey)?;
        self.inner
            .call_streaming_for_service_with_method(
                service_name,
                method_discriminator,
                payload.to_vec(),
                epk,
            )
            .await
            .map_err(|error| JsError::new(&error.to_string()))
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

    /// Open a generated verified stream with explicit service and method metadata.
    #[wasm_bindgen(js_name = "openStreamForServiceWithMethod")]
    pub async fn open_stream_for_service_with_method(
        &self,
        service_name: &str,
        method_discriminator: u16,
        payload: &[u8],
    ) -> Result<WasmStreamHandle, JsError> {
        let handle = self.inner
            .open_stream_for_service_with_method(
                service_name,
                method_discriminator,
                payload.to_vec(),
            )
            .await
            .map_err(|error| JsError::new(&error.to_string()))?;
        Ok(WasmStreamHandle { inner: handle })
    }

    /// Open an option-bearing generated stream with explicit service and method metadata.
    #[wasm_bindgen(js_name = "openStreamWithOptionsForServiceWithMethod")]
    pub async fn open_stream_with_options_for_service_with_method(
        &self,
        service_name: &str,
        method_discriminator: u16,
        payload: &[u8],
        jwt: Option<String>,
        delegated_bearer: Option<String>,
    ) -> Result<WasmStreamHandle, JsError> {
        let handle = self.inner
            .open_stream_with_options_for_service_with_method(
                service_name,
                method_discriminator,
                payload.to_vec(),
                call_options(jwt, delegated_bearer),
            )
            .await
            .map_err(|error| JsError::new(&error.to_string()))?;
        Ok(WasmStreamHandle { inner: handle })
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
/// Cap'n Proto parsing as the native `StreamHandle<LazyUdsTransport>`.
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
            Ok(Some(StreamPayload::Data(data))) =>
                Ok(js_sys::Uint8Array::from(&data[..]).into()),
            Ok(Some(StreamPayload::Complete(meta))) => {
                Ok(js_sys::Uint8Array::from(&meta[..]).into())
            }
            Ok(Some(StreamPayload::Error(msg))) =>
                Err(JsError::new(&msg)),
            Ok(Some(StreamPayload::Tagged { payload, .. })) => {
                Ok(js_sys::Uint8Array::from(&payload[..]).into())
            }
            Ok(None) => Ok(JsValue::NULL),
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }

    /// Cancel the stream via authenticated ctrl channel.
    pub async fn cancel(&mut self) -> Result<(), JsError> {
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
