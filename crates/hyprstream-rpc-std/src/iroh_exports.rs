//! wasm-bindgen exports for iroh peer identity + pkarr discovery (Phase 2).
//!
//! Wraps `hyprstream_rpc::iroh_peer` with wasm-bindgen so JavaScript can:
//!
//! 1. Generate an ephemeral iroh identity (`IrohPeer::new()`).
//! 2. Get the browser's NodeId as bytes or as a `did:key` DID.
//! 3. Resolve a peer's relay URL from the N0 pkarr relay
//!    (replaces the manual HTTP fetch + DNS-wire parse in `atproto.ts`).
//!
//! # Usage (TypeScript)
//!
//! ```ts
//! import { IrohPeer, irohNodeIdFromDidKey, irohDidKeyFromNodeId,
//!          irohResolvePkarrRelayUrl } from 'hyprstream-rpc-std';
//!
//! // Generate a fresh browser identity.
//! const peer = new IrohPeer();
//! console.log("NodeId (z32):", peer.nodeIdZ32());
//! console.log("did:key:", peer.didKey());
//!
//! // Convert a did:key to raw NodeId bytes (for pkarr lookup).
//! const nodeId = irohNodeIdFromDidKey("did:key:z6Mk...");
//!
//! // Resolve a peer's relay URL.
//! const relayUrl = await irohResolvePkarrRelayUrl(nodeId);
//! ```

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

use hyprstream_rpc::iroh_peer::{
    BrowserIrohPeer, did_key_from_node_id, node_id_from_did_key, resolve_pkarr_relay_url,
};

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

    /// This peer's identity as a W3C `did:key` DID.
    ///
    /// E.g. `did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK`
    #[wasm_bindgen(js_name = "didKey")]
    pub fn did_key(&self) -> String {
        self.inner.did_key()
    }
}

impl Default for WasmIrohPeer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// did:key ↔ NodeId helpers (free functions)
// ============================================================================

/// Convert a `did:key:z6Mk...` DID to a 32-byte NodeId (`Uint8Array`).
///
/// Throws if the DID is not a valid ed25519 `did:key`.
#[wasm_bindgen(js_name = "irohNodeIdFromDidKey")]
pub fn node_id_from_did_key_js(did: &str) -> Result<Vec<u8>, JsError> {
    node_id_from_did_key(did)
        .map(|b| b.to_vec())
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Convert a 32-byte NodeId (`Uint8Array`) to a `did:key:z6Mk...` DID.
///
/// Throws if `node_id` is not exactly 32 bytes.
#[wasm_bindgen(js_name = "irohDidKeyFromNodeId")]
pub fn did_key_from_node_id_js(node_id: &[u8]) -> Result<String, JsError> {
    let bytes: [u8; 32] = node_id
        .try_into()
        .map_err(|_| JsError::new("node_id must be exactly 32 bytes"))?;
    Ok(did_key_from_node_id(&bytes))
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
