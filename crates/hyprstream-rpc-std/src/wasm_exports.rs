//! WASM-bindgen exports for browser clients.
//!
//! - `VfsShell` — Tcl + VFS namespace backed by RpcClient
//! - Re-exports of low-level WASM API (crypto primitives, ZMTP framing)
//!
//! The per-service RPC methods that were here (`registry_list`, `model_load`, etc.)
//! have been replaced by generated TypeScript clients that call `RpcClient.call()`
//! with Cap'n Proto bytes. See `wasm_rpc_client.rs` for the unified `RpcClient` export.

#![cfg(target_arch = "wasm32")]

use std::sync::Arc;

use js_sys::Function;
use serde::Serialize as _;
use wasm_bindgen::prelude::*;

// Re-export low-level WASM API (crypto primitives, ZMTP framing) from hyprstream-rpc.
pub use hyprstream_rpc::wasm_api::*;

/// Exchange an ATProto service-auth JWT plus DPoP proof for hyprstream's
/// opaque HttpOnly browser session. The service JWT must target
/// `ai.hyprstream.identity.exchangeSession`.
#[wasm_bindgen(js_name = "exchangeBrowserSession")]
pub async fn exchange_browser_session(
    exchange_endpoint: &str,
    service_auth_jwt: &str,
    dpop_proof: &str,
) -> Result<JsValue, JsError> {
    let context = hyprstream_rpc::wasm_token_exchange::fetch_session_exchange(
        exchange_endpoint,
        service_auth_jwt,
        dpop_proof,
    )
    .await
    .map_err(|error| JsError::new(&error.to_string()))?;
    context
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|error| JsError::new(&format!("session context serialization failed: {error}")))
}

/// Read the current server-derived viewer context from the HttpOnly session.
#[wasm_bindgen(js_name = "browserSessionWhoami")]
pub async fn browser_session_whoami(
    exchange_endpoint: &str,
) -> Result<JsValue, JsError> {
    let context =
        hyprstream_rpc::wasm_token_exchange::fetch_session_context(exchange_endpoint)
            .await
            .map_err(|error| JsError::new(&error.to_string()))?;
    context
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|error| JsError::new(&format!("session context serialization failed: {error}")))
}

// ============================================================================
// VFS Shell — Tcl + Namespace backed by RpcClient
// ============================================================================

/// Browser-side Tcl shell backed by a VFS namespace.
///
/// All I/O goes through `RpcClient<JsSigner, WtConnection>` → ZMTP/QUIC → server.
///
/// `connect` first exchanges `subject_token` for a short-lived at+jwt Bearer via
/// the #1314 token-exchange endpoint (`POST {exchangeEndpoint}/oauth/token`), then
/// dials the registry + model services through browser-provisioning (the only
/// working browser dial) and presents that Bearer as the default JWT on every
/// request. The shell's `Subject` is decoded from the Bearer's `sub` claim.
///
/// ```js
/// const shell = await VfsShell.connect(
///   registryOrigin, modelOrigin,            // https provisioning origins
///   ed25519Pubkey, ed25519SignFn,            // hybrid signer (Ed25519 arm)
///   mlDsa65Pubkey, mlDsa65SignFn,            // hybrid signer (PQ arm)
///   exchangeEndpoint,                        // https origin hosting POST /oauth/token
///   atprotoJwt, "urn:ietf:params:oauth:token-type:jwt",
/// );
/// const result = await shell.eval('ls /srv/registry');
/// ```
#[wasm_bindgen]
pub struct VfsShell {
    shell: std::cell::RefCell<hyprstream_workers_tcl::TclShell>,
}

#[wasm_bindgen]
impl VfsShell {
    /// Create a new VFS shell by authenticating + connecting to services.
    ///
    /// - `registry_origin` / `model_origin`: HTTPS origins serving the
    ///   `/.well-known/hyprstream/browser-provisioning/{registry,model}`
    ///   discovery documents. Cert pinning is server-supplied by provisioning
    ///   (there is no caller `cert_hash` on the resolved path — that is the
    ///   security model the `dial_wasm` stub deliberately forced us onto).
    /// - `signer_pubkey` / `sign_fn`: 32-byte Ed25519 pubkey + async JS callback
    ///   `(canonicalBytes: Uint8Array) => Promise<Uint8Array>` (hybrid Ed25519 arm).
    /// - `signer_ml_dsa65_pubkey` / `pq_sign_fn`: ML-DSA-65 pubkey + PQ sign
    ///   callback (hybrid PQ arm; required by the resolved path).
    /// - `exchange_endpoint`: HTTPS origin hosting the #1314 `POST /oauth/token`.
    /// - `subject_token` / `subject_token_type`: the credential to exchange
    ///   (e.g. an atproto JWT, `subject_token_type = urn:ietf:params:oauth:token-type:jwt`).
    pub async fn connect(
        registry_origin: &str,
        model_origin: &str,
        signer_pubkey: &[u8],
        sign_fn: Function,
        signer_ml_dsa65_pubkey: &[u8],
        pq_sign_fn: Function,
        exchange_endpoint: &str,
        subject_token: String,
        subject_token_type: String,
    ) -> Result<VfsShell, JsError> {
        console_error_panic_hook::set_once();

        // 1) Exchange the atproto/external JWT for a short-lived at+jwt Bearer (#1314).
        web_sys::console::log_1(&"[VfsShell] Exchanging subject token...".into());
        let exchanged = hyprstream_rpc::wasm_token_exchange::fetch_exchange_token(
            exchange_endpoint,
            &subject_token,
            &subject_token_type,
        )
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;

        // 2) Derive the display Subject from the freshly-minted Bearer's `sub`.
        //    The response does not echo `sub` (the mint stamps `sub = verified.sub`),
        //    so decode it client-side. Authority is re-verified by the server on
        //    every RPC; this decode is bookkeeping only.
        let subject = hyprstream_rpc::wasm_token_exchange::subject_from_access_token(
            &exchanged.access_token,
        )
        .map_err(|e| JsError::new(&e.to_string()))?;
        web_sys::console::log_1(&"[VfsShell] Subject resolved from exchanged token".into());

        // 3) Connect to registry + model through the resolved browser-provisioning
        //    path (the only working browser dial), presenting the Bearer as the
        //    default JWT on every request. The `dial_wasm::dial` stub is dropped.
        let bearer = Some(exchanged.access_token);
        web_sys::console::log_1(&"[VfsShell] Connecting to registry...".into());
        let reg_client: Arc<dyn hyprstream_rpc::rpc_client::RpcClient> = Arc::new(
            crate::wasm_rpc_client::build_resolved_client(
                registry_origin,
                "registry",
                signer_pubkey,
                sign_fn.clone(),
                signer_ml_dsa65_pubkey,
                pq_sign_fn.clone(),
                bearer.clone(),
            )
            .await?,
        );
        web_sys::console::log_1(&"[VfsShell] Registry connected".into());

        web_sys::console::log_1(&"[VfsShell] Connecting to model...".into());
        let model_client: Arc<dyn hyprstream_rpc::rpc_client::RpcClient> = Arc::new(
            crate::wasm_rpc_client::build_resolved_client(
                model_origin,
                "model",
                signer_pubkey,
                sign_fn,
                signer_ml_dsa65_pubkey,
                pq_sign_fn,
                bearer,
            )
            .await?,
        );
        web_sys::console::log_1(&"[VfsShell] Model connected".into());

        // 4) Build VFS namespace + Tcl shell with the authenticated subject.
        web_sys::console::log_1(&"[VfsShell] Building namespace...".into());
        let (ns, _stream_registry) =
            crate::vfs_mount::build_browser_namespace(reg_client, model_client);
        let ns = Arc::new(ns);
        web_sys::console::log_1(&"[VfsShell] Namespace built".into());

        let shell = hyprstream_workers_tcl::TclShell::new(subject, ns);

        Ok(VfsShell {
            shell: std::cell::RefCell::new(shell),
        })
    }

    /// Evaluate a Tcl script against the VFS namespace.
    pub async fn eval(&self, script: &str) -> Result<String, JsError> {
        let mut shell = self.shell.borrow_mut();
        shell.eval(script).await
            .map_err(|e| JsError::new(&e))
    }
}
