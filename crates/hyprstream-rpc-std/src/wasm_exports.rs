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
use wasm_bindgen::prelude::*;

// Re-export low-level WASM API (crypto primitives, ZMTP framing) from hyprstream-rpc.
pub use hyprstream_rpc::wasm_api::*;

// ============================================================================
// VFS Shell — Tcl + Namespace backed by RpcClient
// ============================================================================

/// Browser-side Tcl shell backed by a VFS namespace.
///
/// All I/O goes through `RpcClient<JsSigner, WtConnection>` → ZMTP/QUIC → server.
///
/// ```js
/// const shell = await VfsShell.connect(regUrl, modelUrl, certHash, signerPubkey, signFn);
/// const result = await shell.eval('ls /srv/registry');
/// ```
#[wasm_bindgen]
pub struct VfsShell {
    shell: std::cell::RefCell<hyprstream_tcl::TclShell>,
}

#[wasm_bindgen]
impl VfsShell {
    /// Create a new VFS shell by connecting to services.
    ///
    /// Creates its own `RpcClient` instances for the registry and model services.
    /// Both clients share the same `signer_pubkey` and `sign_fn`.
    ///
    /// - `signer_pubkey`: 32-byte Ed25519 public key for every envelope
    /// - `sign_fn`: async JS callback `(canonicalBytes: Uint8Array) => Promise<Uint8Array>`
    pub async fn connect(
        registry_url: &str,
        model_url: &str,
        cert_hash: Option<String>,
        signer_pubkey: &[u8],
        sign_fn: Function,
    ) -> Result<VfsShell, JsError> {
        console_error_panic_hook::set_once();

        use hyprstream_rpc::rpc_client::{RpcClientImpl, RpcClient};
        use hyprstream_rpc::signer::JsSigner;
        use hyprstream_rpc::web_transport::WtConnection;
        use hyprstream_rpc::crypto::VerifyingKey;

        // TODO: Get server verifying key from discovery endpoint
        let server_key = VerifyingKey::from_bytes(&[0u8; 32]).unwrap_or_else(|_| {
            VerifyingKey::from_bytes(&[
                0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7,
                0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07, 0x3a,
                0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa3, 0x23, 0x28,
                0x27, 0xbf, 0x5c, 0xdc, 0xb3, 0xa0, 0x35, 0x6c,
            ]).expect("fallback key")
        });

        // Connect to registry
        web_sys::console::log_1(&"[VfsShell] Connecting to registry...".into());
        let reg_transport = WtConnection::connect(registry_url, cert_hash.as_deref())
            .await.map_err(|e| JsError::new(&e.to_string()))?;
        let reg_signer = JsSigner::new(signer_pubkey, sign_fn.clone())
            .map_err(|e| JsError::new(&e.to_string()))?;
        let reg_client: Arc<dyn RpcClient> = Arc::new(
            RpcClientImpl::new(reg_signer, reg_transport, server_key.clone())
        );
        web_sys::console::log_1(&"[VfsShell] Registry connected".into());

        // Connect to model
        web_sys::console::log_1(&"[VfsShell] Connecting to model...".into());
        let model_transport = WtConnection::connect(model_url, cert_hash.as_deref())
            .await.map_err(|e| JsError::new(&e.to_string()))?;
        let model_signer = JsSigner::new(signer_pubkey, sign_fn)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let model_client: Arc<dyn RpcClient> = Arc::new(
            RpcClientImpl::new(model_signer, model_transport, server_key)
        );
        web_sys::console::log_1(&"[VfsShell] Model connected".into());

        // Build VFS namespace
        web_sys::console::log_1(&"[VfsShell] Building namespace...".into());
        let (ns, _stream_registry) = crate::vfs_mount::build_browser_namespace(reg_client, model_client);
        let ns = Arc::new(ns);
        web_sys::console::log_1(&"[VfsShell] Namespace built".into());

        // Create Tcl shell
        let subject = hyprstream_rpc::Subject::anonymous();
        let shell = hyprstream_tcl::TclShell::new(subject, ns);

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
