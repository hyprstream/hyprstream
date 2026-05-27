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
    let key_bytes = state.verifying_key_bytes;
    let x = URL_SAFE_NO_PAD.encode(key_bytes);
    let eddsa_kid = compute_kid(&key_bytes);

    let mut keys = vec![serde_json::json!({
        "kty": "OKP",
        "crv": "Ed25519",
        "use": "sig",
        "alg": "EdDSA",
        "kid": eddsa_kid,
        "x": x,
    })];

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
