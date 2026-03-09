//! GET /oauth/jwks — JSON Web Key Set (RFC 7517) for the local signing key.
//!
//! Returns the node's Ed25519 verifying key as an OKP JWK (RFC 8037).
//! Federation peers use this to verify JWTs issued by this node.

use axum::{extract::State, response::IntoResponse, Json};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use std::sync::Arc;
use super::state::OAuthState;

/// GET /oauth/jwks
pub async fn jwks(State(state): State<Arc<OAuthState>>) -> impl IntoResponse {
    let key_bytes = state.verifying_key_bytes;
    let x = URL_SAFE_NO_PAD.encode(key_bytes);

    // Stable key ID: first 8 hex chars of SHA-256 of the raw key bytes
    let kid = {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(key_bytes);
        hex::encode(&hash[..4])
    };

    Json(serde_json::json!({
        "keys": [{
            "kty": "OKP",
            "crv": "Ed25519",
            "use": "sig",
            "kid": kid,
            "x": x,
        }]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
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
