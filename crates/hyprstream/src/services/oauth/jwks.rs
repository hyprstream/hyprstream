//! GET /oauth/jwks — JSON Web Key Set (RFC 7517).
//!
//! Returns the node's signing keys: Ed25519 (OKP, RFC 8037) and optionally
//! RSA (for RS256 interop with enterprise RPs).

use axum::{extract::State, response::IntoResponse, Json};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use hyprstream_rpc::auth::{JwkThumbprintInput, jwk_thumbprint};
use std::sync::Arc;
use super::state::OAuthState;

/// Compute the RFC 7638 JWK Thumbprint for an Ed25519 key (32-byte raw pubkey).
pub fn compute_kid(key_bytes: &[u8]) -> String {
    let bytes: [u8; 32] = match key_bytes.try_into() {
        Ok(b) => b,
        Err(_) => {
            tracing::error!("compute_kid called with {} bytes, expected 32", key_bytes.len());
            return String::new();
        }
    };
    jwk_thumbprint(&JwkThumbprintInput::Ed25519 { x: &bytes })
}

/// Compute the RFC 7638 JWK Thumbprint for an RSA key (n and e as base64url strings).
pub fn compute_rsa_kid(n: &str, e: &str) -> String {
    jwk_thumbprint(&JwkThumbprintInput::Rsa { n, e })
}

/// GET /oauth/jwks
pub async fn jwks(State(state): State<Arc<OAuthState>>) -> impl IntoResponse {
    Json(serde_json::json!({ "keys": jwks_json(&state).await }))
}

/// Build the public JWKS key array shared by `/oauth/jwks` and the SPIFFE
/// bundle endpoint.
pub async fn jwks_json(state: &OAuthState) -> Vec<serde_json::Value> {
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

    // Always publish the cluster CA verifying key (state.verifying_key_bytes).
    // This is the key PolicyService uses to sign at+JWTs — for single-process
    // mode it's the derived `hyprstream-jwt-v1` key set by
    // `generate_independent_service_keys` (factory.rs); for IPC mode it's the
    // loaded ca-pubkey credential. Clients fetching JWKS to verify at+JWTs
    // need this key regardless of whether a rotation store is present.
    // De-duplicate against rotation slots by kid.
    {
        let key_bytes = state.verifying_key_bytes;
        let eddsa_kid = compute_kid(&key_bytes);
        let already = keys
            .iter()
            .any(|k| k.get("kid").and_then(|v| v.as_str()) == Some(eddsa_kid.as_str()));
        if !already {
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

    // ES256 from rotation store — publish all slots (drain/active/lead)
    if let Some(ref store) = state.es256_key_store {
        for slot in store.all_slots_snapshot().await {
            let mut jwk = crate::auth::jwt::es256_jwk(&slot.key);
            if let Some(obj) = jwk.as_object_mut() {
                obj.insert("nbf".to_owned(), slot.nbf.into());
                obj.insert("exp".to_owned(), slot.exp.into());
            }
            keys.push(jwk);
        }
    }

    // ML-DSA-65 from rotation store — publish all slots + composite pairing
    if let Some(ref store) = state.ml_dsa_key_store {
        for slot in store.all_slots_snapshot().await {
            let vk = ml_dsa::Keypair::verifying_key(&*slot.key);
            let mut jwk = crate::auth::jwt::ml_dsa_65_jwk(&vk);
            if let Some(obj) = jwk.as_object_mut() {
                obj.insert("nbf".to_owned(), slot.nbf.into());
                obj.insert("exp".to_owned(), slot.exp.into());
            }
            keys.push(jwk);

            // Composite key pairing with active Ed25519 from rotation store
            if let Some(ref ed_store) = state.signing_key_store {
                if let Some(ed_key) = ed_store.active_key().await {
                    keys.push(crate::auth::jwt::composite_jwk(
                        &vk,
                        &ed_key.verifying_key(),
                    ));
                }
            }
        }
    }

    // Add RSA public key if available
    if let Some(ref rsa_jwk) = state.rsa_jwk {
        keys.push(rsa_jwk.clone());
    }

    keys
}

#[cfg(test)]
mod tests {
    use super::compute_kid;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;

    #[test]
    fn test_kid_is_rfc7638_thumbprint() {
        let key_bytes = [0u8; 32];
        let kid = compute_kid(&key_bytes);
        // RFC 7638 thumbprint: base64url(SHA-256(canonical_jwk)) = 43 chars
        assert_eq!(kid.len(), 43);
        assert!(!kid.contains('='));
    }

    #[test]
    fn test_kid_deterministic() {
        let key_bytes = [1u8; 32];
        assert_eq!(compute_kid(&key_bytes), compute_kid(&key_bytes));
    }

    #[test]
    fn test_kid_matches_jwt_module() {
        let key_bytes = [42u8; 32];
        let kid = compute_kid(&key_bytes);
        let jwt_kid = hyprstream_rpc::auth::jwk_thumbprint(
            &hyprstream_rpc::auth::JwkThumbprintInput::Ed25519 { x: &key_bytes },
        );
        assert_eq!(kid, jwt_kid);
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
