//! wasm-bindgen exports for iroh peer identity + pkarr discovery (Phase 2).
//!
//! Wraps `hyprstream_rpc::iroh_peer` with wasm-bindgen so JavaScript can:
//!
//! 1. Generate an ephemeral iroh identity (`IrohPeer::new()`).
//! 2. Get the browser's NodeId as transport-address bytes or z-base32.
//! 3. Resolve a peer's relay URL from the N0 pkarr relay
//!    (replaces the manual HTTP fetch + DNS-wire parse in `atproto.ts`).
//!
//! # Usage (TypeScript)
//!
//! ```ts
//! import { IrohPeer, irohResolvePkarrRelayUrl } from 'hyprstream-rpc-std';
//!
//! // Generate a fresh browser identity.
//! const peer = new IrohPeer();
//! console.log("NodeId (z32):", peer.nodeIdZ32());
//! // Resolve a peer's relay URL.
//! const relayUrl = await irohResolvePkarrRelayUrl(peer.nodeIdBytes());
//! ```

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

use hyprstream_rpc::iroh_peer::{BrowserIrohPeer, resolve_pkarr_relay_url};

// ============================================================================
// IrohPeer — ephemeral browser iroh identity
// ============================================================================

/// An ephemeral iroh peer identity for the browser.
///
/// Generates a fresh Ed25519 `SecretKey` on construction and derives the
/// corresponding `EndpointId` (NodeId). This gives the browser a stable iroh
/// identity for the session without binding a full `iroh::Endpoint`.
#[wasm_bindgen(js_name = "IrohPeer")]
pub struct WasmIrohPeer {
    inner: BrowserIrohPeer,
}

#[wasm_bindgen(js_class = "IrohPeer")]
impl WasmIrohPeer {
    /// Generate a fresh ephemeral iroh peer identity.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmIrohPeer {
        WasmIrohPeer {
            inner: BrowserIrohPeer::new(),
        }
    }

    /// The NodeId as a 32-byte `Uint8Array` (raw Ed25519 public key).
    #[wasm_bindgen(js_name = "nodeIdBytes")]
    pub fn node_id_bytes(&self) -> Vec<u8> {
        self.inner.node_id_bytes().to_vec()
    }

    /// The NodeId encoded as z-base32 (used in pkarr gateway URLs).
    #[wasm_bindgen(js_name = "nodeIdZ32")]
    pub fn node_id_z32(&self) -> String {
        self.inner.node_id_z32()
    }

}

impl Default for WasmIrohPeer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Pkarr relay lookup
// ============================================================================

/// Resolve a peer's relay URL from the N0 pkarr relay.
///
/// - `node_id`: 32-byte Ed25519 pubkey (`Uint8Array`).
/// - Returns the relay URL string, or `null` if the peer has no record.
/// - Throws on network or parse errors.
///
/// This replaces the manual HTTP fetch + DNS-wire parse in `atproto.ts`'s
/// `resolveDidKey()`. Both hit `https://dns.iroh.link/pkarr/<z32>`, but
/// this path uses iroh's signature-verifying parser.
#[wasm_bindgen(js_name = "irohResolvePkarrRelayUrl")]
pub async fn resolve_pkarr_relay_url_js(node_id: &[u8]) -> Result<JsValue, JsError> {
    let bytes: [u8; 32] = node_id
        .try_into()
        .map_err(|_| JsError::new("node_id must be exactly 32 bytes"))?;

    match resolve_pkarr_relay_url(&bytes).await {
        // Unwrap the unverified reach hint (D3: pkarr derives zero authority)
        // to the bare URL string the JS dial path consumes as a relay address.
        Ok(Some(hint)) => Ok(JsValue::from_str(hint.url())),
        Ok(None) => Ok(JsValue::NULL),
        Err(e) => Err(JsError::new(&e.to_string())),
    }
}
