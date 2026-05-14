//! GET /oauth/jwks — JSON Web Key Set (RFC 7517).
//!
//! Returns the node's signing keys: Ed25519 (OKP, RFC 8037) and optionally
//! RSA (for RS256 interop with enterprise RPs).

use axum::{extract::State, response::IntoResponse, Json};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use std::sync::Arc;
use super::state::OAuthState;

/// Compute a stable kid from raw key bytes (first 8 hex chars of SHA-256).
pub fn compute_kid(key_bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(key_bytes);
    hex::encode(&hash[..4])
}

/// GET /oauth/jwks
pub async fn jwks(State(state): State<Arc<OAuthState>>) -> impl IntoResponse {
    let mut keys: Vec<serde_json::Value> = Vec::new();

    // Serve all rotation slots (drain + active + lead) when the store is present.
    // WITs and rotating-issuance tokens use these.
    if let Some(ref store) = state.signing_key_store {
        for slot in store.all_slots_snapshot().await {
            let vk_bytes = slot.verifying_key_bytes();
            let kid = compute_kid(&vk_bytes);
            let x = URL_SAFE_NO_PAD.encode(vk_bytes);
            keys.push(serde_json::json!({
                "kty": "OKP",
                "crv": "Ed25519",
                "use": "sig",
                "alg": "EdDSA",
                "kid": kid,
                "x": x,
                "nbf": slot.nbf,
                "exp": slot.exp,
            }));
        }
    }

    // Always publish the PolicyService verifying key. at+JWTs are signed by
    // PolicyService (`policy_client.issue_token()`) regardless of whether
    // a rotation store is configured; clients fetching JWKS to verify those
    // tokens need this key. De-duplicate against rotation slots by kid.
    {
        let key_bytes = state.verifying_key_bytes;
        let eddsa_kid = compute_kid(&key_bytes);
        let already_present = keys
            .iter()
            .any(|k| k.get("kid").and_then(|v| v.as_str()) == Some(eddsa_kid.as_str()));
        if !already_present {
            let x = URL_SAFE_NO_PAD.encode(key_bytes);
            keys.push(serde_json::json!({
                "kty": "OKP",
                "crv": "Ed25519",
                "use": "sig",
                "alg": "EdDSA",
                "kid": eddsa_kid,
                "x": x,
                "nbf": state.jwt_key_nbf,
                "exp": state.jwt_key_exp,
            }));
        }
    }

    // Add RSA public key if available
    if let Some(ref rsa_jwk) = state.rsa_jwk {
        keys.push(rsa_jwk.clone());
    }

    Json(serde_json::json!({ "keys": keys }))
}

#[cfg(test)]
mod tests {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use sha2::{Digest, Sha256};

    #[test]
    fn test_kid_is_8_hex_chars() {
        let key_bytes = [0u8; 32];
        let hash = Sha256::digest(key_bytes);
        let kid = hex::encode(&hash[..4]);
        // 4 bytes * 2 hex chars per byte = 8 chars
        assert_eq!(kid.len(), 8);
        assert!(kid.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_x_is_base64url_no_pad() {
        let key_bytes = [1u8; 32];
        let x = URL_SAFE_NO_PAD.encode(key_bytes);
        // 32 bytes base64url no-pad = 43 chars
        assert_eq!(x.len(), 43);
        assert!(!x.contains('='));
    }
}
