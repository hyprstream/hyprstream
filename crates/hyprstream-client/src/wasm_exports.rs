//! wasm-bindgen exports for browser use.
//!
//! Thin wrappers around the generated service clients.
//! React/TypeScript calls these directly — no TS codegen needed.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

/// Placeholder — will be populated as we wire up each service client.
/// The generated clients need a transport + signing key to be constructed.
/// This module will provide:
///
/// ```ignore
/// #[wasm_bindgen]
/// pub async fn connect(url: &str, cert_hash: Option<String>) -> Result<(), JsError>
///
/// #[wasm_bindgen]
/// pub async fn registry_list() -> Result<JsValue, JsError>
///
/// #[wasm_bindgen]
/// pub async fn model_load(model_ref: &str) -> Result<JsValue, JsError>
/// ```
///
/// Each function: get-or-create client → call method → serialize result via serde-wasm-bindgen
#[wasm_bindgen]
pub fn client_version() -> String {
    "hyprstream-client 0.1.0".to_owned()
}
