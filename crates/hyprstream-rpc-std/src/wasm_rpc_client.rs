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

use capnp::traits::HasTypeId;
use wasm_bindgen::prelude::*;

use hyprstream_rpc::browser_provisioning::{
    fetch_browser_provisioning, BrowserCarrierProfile, BrowserProvisioningGuard,
    BrowserProvisioningRequest,
};

use hyprstream_rpc::crypto::VerifyingKey;
use hyprstream_rpc::rpc_client::{CallOptions, RpcClientImpl};
use hyprstream_rpc::rpc_client::FederateProofProvider;
use hyprstream_rpc::proof::{
    build::{AuthenticatedRequestProofInput, build_authenticated_hybrid_request_proof_with_signer},
    recipient_binding::FederateRecipientBinding,
};
use hyprstream_rpc::envelope::RequestEnvelope;
use hyprstream_rpc::signer::JsSigner;
use hyprstream_rpc::transport_traits::Signer;
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

struct BrowserFederateProofProvider {
    signer: JsSigner,
    ed_kid: Vec<u8>,
    pq_kid: Vec<u8>,
    service_name: String,
    request_schema_id: u64,
}

#[async_trait::async_trait(?Send)]
impl FederateProofProvider for BrowserFederateProofProvider {
    async fn proof_for(&self, envelope: &RequestEnvelope, capnp_body: &[u8]) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(envelope.delegation_token.is_none(), "Federate proof cannot delegate a bearer");
        let service = envelope.service_domain.as_deref().ok_or_else(||
            anyhow::anyhow!("Federate proof requires a canonical service domain"))?;
        anyhow::ensure!(service == self.service_name, "Federate proof service mismatch");
        let credential = envelope.jwt_token().ok_or_else(||
            anyhow::anyhow!("Federate proof requires the host session credential"))?;
        let issued_at = u64::try_from(envelope.iat).map_err(|_| anyhow::anyhow!(
            "Federate proof requires a nonnegative request timestamp"))?;
        let binding = FederateRecipientBinding::from_recipients(
            envelope.response_kem_recipient.as_ref(),
            envelope.client_kem_public.as_ref(),
            envelope.client_dh_public,
        )?;
        let input = AuthenticatedRequestProofInput {
            service_domain: service,
            credential: credential.as_bytes(),
            issued_at,
            expires_at: issued_at.checked_add(30).ok_or_else(|| anyhow::anyhow!(
                "Federate proof timestamp overflow"))?,
            capnp_schema_id: self.request_schema_id,
            capnp_body,
            // v16 response-proof binding is a separate COSE response model.
            // Current network responses use the hybrid envelope KEM, which is
            // committed above by signed -70009; it is not an ML-KEM-only
            // response-proof recipient.
            response_binding: None,
            federate_recipient_binding: Some(&binding),
        };
        build_authenticated_hybrid_request_proof_with_signer(
            &input, &self.signer, &self.ed_kid, &self.pq_kid,
        ).await
    }
}

impl WasmRpcClient {
    /// Wrap an already-built resolved client (see [`build_resolved_client`]).
    /// Shared by [`WasmRpcClient::connect_resolved`] and by
    /// `browser_session::BrowserSession`, which constructs direct typed
    /// clients without a `VfsShell`.
    pub(crate) fn from_resolved(inner: RpcClientImpl<JsSigner, WtConnection>) -> Self {
        Self { inner }
    }
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

/// Resolve accepted-current browser provisioning for `service_name`, dial the
/// bound owned WebTransport reach, and build a fully-provisioned client (hybrid
/// `JsSigner`, KEM/PQ crypto stores, request-binding pre-seal guard, optional
/// default JWT).
///
/// Rust-internal counterpart of [`WasmRpcClient::connect_resolved`]: it returns
/// the concrete client so callers can either wrap it in `WasmRpcClient` (the JS
/// export) or erase it behind `Arc<dyn RpcClient>` for a VFS namespace mount
/// (see `VfsShell::connect`). This is the single working browser dial — the
/// `dial_wasm::dial` stub is deliberately disabled in favor of this path.
pub(crate) async fn build_resolved_client(
    provisioning_origin: &str,
    service_name: &str,
    signer_pubkey: &[u8],
    sign_fn: js_sys::Function,
    signer_ml_dsa65_pubkey: &[u8],
    pq_sign_fn: js_sys::Function,
    jwt: Option<String>,
) -> Result<RpcClientImpl<JsSigner, WtConnection>, JsError> {
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
    let signer = JsSigner::new_hybrid(signer_pubkey, sign_fn, signer_ml_dsa65_pubkey, pq_sign_fn)
        .map_err(|error| JsError::new(&error.to_string()))?;
    let (kem, pq) = provisioned
        .crypto_stores()
        .map_err(|error| JsError::new(&error.to_string()))?;
    let guard = BrowserProvisioningGuard::new(provisioning_origin, expected, provisioned.clone());
    let binding = provisioned
        .request_binding()
        .map_err(|error| JsError::new(&error.to_string()))?;
    let client = RpcClientImpl::new(signer, transport, Some(provisioned.server_verifying_key()))
        .with_request_kem_store(kem)
        .with_response_pq_store(pq)
        .with_pre_seal_guard(Arc::new(guard))
        .with_browser_provisioning_binding(binding)
        .map_err(|error| JsError::new(&error.to_string()))?;
    let client = match jwt {
        Some(token) => client.with_default_jwt(token),
        None => client,
    };
    Ok(client)
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
        let inner = build_resolved_client(
            provisioning_origin,
            service_name,
            signer_pubkey,
            sign_fn,
            signer_ml_dsa65_pubkey,
            pq_sign_fn,
            jwt,
        )
        .await?;
        Ok(Self::from_resolved(inner))
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

    /// Opt into per-request Federate COSE proofs for a single canonical
    /// Registry or Model client. These callbacks own the dedicated proof keys;
    /// envelope-signing callbacks configured at connect time stay separate.
    #[wasm_bindgen(js_name = "withFederateProofSigner")]
    pub fn with_federate_proof_signer(
        self,
        service_name: &str,
        ed_pubkey: &[u8],
        ed_sign_fn: js_sys::Function,
        pq_pubkey: &[u8],
        pq_sign_fn: js_sys::Function,
        ed_kid: &[u8],
        pq_kid: &[u8],
    ) -> Result<WasmRpcClient, JsError> {
        hyprstream_rpc::envelope::validate_service_domain(service_name)
            .map_err(|error| JsError::new(&error.to_string()))?;
        let request_schema_id = match service_name {
            "registry" => <crate::registry_capnp::registry_request::Reader<'static> as HasTypeId>::TYPE_ID,
            "model" => <crate::model_capnp::model_request::Reader<'static> as HasTypeId>::TYPE_ID,
            _ => return Err(JsError::new("Federate proof sender is limited to Registry and Model")),
        };
        if ed_pubkey == self.inner.signer.pubkey() {
            return Err(JsError::new("Federate proof Ed25519 key must differ from the envelope key"));
        }
        if self.inner.signer.pq_pubkey().as_deref() == Some(pq_pubkey) {
            return Err(JsError::new("Federate proof ML-DSA-65 key must differ from the envelope key"));
        }
        for (name, kid) in [("Ed25519", ed_kid), ("ML-DSA-65", pq_kid)] {
            if kid.is_empty() || kid.len() > hyprstream_rpc::proof::MAX_KID_BYTES {
                return Err(JsError::new(&format!("Federate proof {name} kid must be 1..64 bytes")));
            }
        }
        let signer = JsSigner::new_hybrid(ed_pubkey, ed_sign_fn, pq_pubkey, pq_sign_fn)
            .map_err(|error| JsError::new(&error.to_string()))?;
        let provider = BrowserFederateProofProvider {
            signer,
            ed_kid: ed_kid.to_vec(),
            pq_kid: pq_kid.to_vec(),
            service_name: service_name.to_owned(),
            request_schema_id,
        };
        Ok(WasmRpcClient {
            inner: self.inner.with_federate_proof_provider(Arc::new(provider)),
        })
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
