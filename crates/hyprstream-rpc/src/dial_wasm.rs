//! Browser (wasm32) counterpart to native `dial.rs`.
//!
//! # Why a separate module, not an arm in `dial()`
//!
//! The native [`crate::dial::dial`] factory is `#[cfg(not(target_arch =
//! "wasm32"))]`: its arms (Quic / Iroh / Ipc / SystemdFd) pull `quinn`, `iroh`,
//! `tokio`, and Unix-socket code that do not compile for
//! `wasm32-unknown-unknown`, and the capnp-over-`web_transport_trait::Session`
//! framing it would otherwise lean on ([`SessionRpcTransport`]) is itself
//! native-only — `web-transport-trait` is a native-only dependency and the
//! whole `transport::rpc_session` submodule is gated off on wasm.
//!
//! The browser therefore speaks a *different* RPC wire format than native:
//! ZMTP framing over a `web_sys::WebTransport` bidi stream (see
//! [`crate::web_transport::WtConnection`]), matched by the server's
//! `handle_wt_stream` handler in `transport::zmtp_quic.rs`. Constructing an
//! `Arc<dyn RpcClient>` for that wire format is what this module does, so the
//! browser goes through one `dial()`-style entrypoint per platform instead of
//! hand-assembling `RpcClientImpl` at every call site.
//!
//! # What this is (and is not)
//!
//! This is the Path-A gap closure from #409: a single factory that takes a
//! URL + signer + (optional) pin + (optional) JWT and returns a ready
//! `Arc<dyn RpcClient>`, mirroring native `dial()`'s contract. It hides the
//! concrete `WtConnection` transport behind the object-safe `RpcClient` trait
//! exactly as native `dial()` hides `LazyQuinnTransport` / `LazyIrohTransport`
//! / etc.
//!
//! It is **not** a swap-in replacement for `GenericServiceMount`:
//! `GenericServiceMount` remains the correct VFS mount for the browser. The
//! "real filesystem walk" alternative ([`RemoteModelMount`] /
//! [`RemoteRegistryMount`]) lives in the native-only `hyprstream` binary crate
//! and spins its own `tokio::runtime::Runtime`; it cannot be reached from the
//! wasm-shared `hyprstream-rpc-std` crate, and porting it across the crate
//! boundary would be high-risk churn for no functional gain since
//! `GenericServiceMount` already proxies the same service methods over the
//! same `Arc<dyn RpcClient>`. See #409 findings.

#![cfg(target_arch = "wasm32")]

use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::crypto::hybrid_kem::KemTrustStore;
use crate::crypto::VerifyingKey;
use crate::rpc_client::{RpcClient, RpcClientImpl};
use crate::signer::JsSigner;
use crate::web_transport::WtConnection;

/// Dial a hyprstream server from the browser over WebTransport, returning a
/// ready RPC client.
///
/// This is the wasm32 mirror of native [`crate::dial::dial`]. It performs the
/// WT handshake eagerly (unlike native's lazy transports), because the browser
/// WT API (`new WebTransport(url)`) is itself async and there is no benefit to
/// deferring it — the first `call()` would block on the same handshake anyway.
///
/// # Arguments
///
/// * `url` — WebTransport URL of the server (e.g. `https://host:port`).
/// * `cert_hash` — Optional base64-encoded SHA-256 of the server's leaf
///   certificate, for self-signed mesh pinning (mirrors native
///   `QuicServerAuth::pinned`). Omit for WebPKI-validated servers.
/// * `signer` — A [`JsSigner`] wrapping the caller's Ed25519 signing callback.
/// * `server_verifying_key` — Optional pinned server identity. `None` does NOT
///   disable signature verification (the response is still verified against the
///   key embedded in its envelope); it only declines to pin *which* identity
///   that key must be. Mirrors the `dial()` contract. For WebTransport the
///   channel is TLS-authenticated (WebPKI or cert-hash pin), so `None` is sound
///   in the same way it is for native `Quic` with WebPKI.
/// * `jwt` — Optional default JWT applied to every request (CA-signed trust
///   cert included in request envelopes).
///
/// # Errors
///
/// Returns an error if the WebTransport handshake fails.
pub async fn dial(
    _url: &str,
    _cert_hash: Option<&str>,
    _signer: JsSigner,
    _server_verifying_key: Option<VerifyingKey>,
    _jwt: Option<String>,
) -> Result<Arc<dyn RpcClient>> {
    Err(anyhow!(
        "unprovisioned browser WebTransport dial is disabled; resolve accepted-current browser provisioning first"))
}

/// Dial over WebTransport with an anchored `#mesh-kem` recipient trust store.
///
/// WebTransport is a cleartext-forbidding carrier
/// ([`WtConnection::forbids_cleartext_envelope`] → `true`), so its request
/// envelopes MUST be sealed to the server's anchored `#mesh-kem` recipient. This
/// mirrors native [`crate::dial::dial_with_kem_store`]: without a store, every
/// `call()` fails closed at sign time (no cleartext downgrade).
///
/// # Prerequisite
///
/// The browser must first resolve the server's `#mesh-kem` recipient key
/// out-of-band (DID `keyAgreement` / peer attestation) and provision a
/// [`KemTrustStore`]. That browser-side provisioning is tracked separately; the
/// bare [`dial`] entrypoint (store `None`) therefore cannot yet complete an
/// encrypted WebTransport request — use this entrypoint once a store exists.
pub async fn dial_with_kem_store(
    url: &str,
    cert_hash: Option<&str>,
    signer: JsSigner,
    server_verifying_key: Option<VerifyingKey>,
    jwt: Option<String>,
    request_kem_store: Option<Arc<dyn KemTrustStore>>,
) -> Result<Arc<dyn RpcClient>> {
    dial_with_crypto_stores(
        url,
        cert_hash,
        signer,
        server_verifying_key,
        jwt,
        request_kem_store,
        None,
    )
    .await
}

/// WebTransport dial seam provisioning both request KEM and response hybrid
/// signature anchors. The shared client then applies the same response
/// carrier-check → verify → HyKEM-open ordering as native clients.
pub async fn dial_with_crypto_stores(
    url: &str,
    cert_hash: Option<&str>,
    signer: JsSigner,
    server_verifying_key: Option<VerifyingKey>,
    jwt: Option<String>,
    request_kem_store: Option<Arc<dyn KemTrustStore>>,
    response_pq_store: Option<Arc<dyn crate::envelope::PqTrustStore>>,
) -> Result<Arc<dyn RpcClient>> {
    let transport = WtConnection::connect(url, cert_hash).await?;
    let client = RpcClientImpl::new(signer, transport, server_verifying_key);
    let client = match request_kem_store {
        Some(store) => client.with_request_kem_store(store),
        None => client,
    };
    let client = match response_pq_store {
        Some(store) => client.with_response_pq_store(store),
        None => client,
    };
    let client = match jwt {
        Some(t) => client.with_default_jwt(t),
        None => client,
    };
    Ok(Arc::new(client) as Arc<dyn RpcClient>)
}

/// Dial from raw URL + signer bytes, without a pre-built [`JsSigner`].
///
/// Convenience wrapper that constructs the [`JsSigner`] from the JS-facing
/// pubkey + sign-callback pair, then delegates to [`dial`]. This matches the
/// argument shape of `WasmRpcClient::connect` so callers migrating from
/// hand-assembled clients keep their call sites.
///
/// # Errors
///
/// Returns an error if `signer_pubkey` is not 32 bytes, the JS callback is
/// malformed, or the WebTransport handshake fails.
pub async fn dial_with_js_signer(
    url: &str,
    cert_hash: Option<&str>,
    signer_pubkey: &[u8],
    sign_fn: js_sys::Function,
    server_verifying_key: Option<VerifyingKey>,
    jwt: Option<String>,
) -> Result<Arc<dyn RpcClient>> {
    let signer = JsSigner::new(signer_pubkey, sign_fn)
        .map_err(|e| anyhow!("dial_with_js_signer: invalid signer: {e}"))?;
    dial(url, cert_hash, signer, server_verifying_key, jwt).await
}
