//! OpenID Federation 1.0 entity configuration endpoint.
//!
//! Publishes `GET /.well-known/openid-federation` — a signed JWT (entity-statement+jwt)
//! that makes this hyprstream node discoverable in OpenID Federation-aware networks.
//!
//! This is a purely additive step: existing `trusted_issuers`-based federation continues
//! to work unchanged. The entity configuration is a prerequisite for future trust-chain
//! resolution (Step 13, deferred).
//!
//! # Entity configuration JWT
//! - Header: `{"alg":"EdDSA","typ":"entity-statement+jwt"}`
//! - `iss` / `sub`: the node's issuer URL (self-issued — both equal)
//! - `iat` / `exp`: now / now + 24h
//! - `jwks`: the node's Ed25519 verifying key as OKP JWK
//! - `metadata.oauth_authorization_server`: RFC 8414 AS metadata fields
//! - `authority_hints`: Trust Anchor URLs from config (empty if none configured)

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::{SigningKey, VerifyingKey};
use ed25519_dalek::Signer;

use super::state::OAuthState;

/// GET /.well-known/openid-federation — OpenID Federation 1.0 entity configuration
pub async fn entity_configuration(
    State(state): State<Arc<OAuthState>>,
) -> Response {
    let Some(ref sk) = state.signing_key else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "signing key not configured for OpenID Federation",
        ).into_response();
    };
    let vk = sk.verifying_key();

    // Build AS metadata (mirrors authorization_server_metadata endpoint)
    let issuer = &state.issuer_url;
    let as_metadata = serde_json::json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{}/oauth/authorize", issuer),
        "token_endpoint": format!("{}/oauth/token", issuer),
        "registration_endpoint": format!("{}/oauth/register", issuer),
        "device_authorization_endpoint": format!("{}/oauth/device", issuer),
        "jwks_uri": format!("{}/oauth/jwks", issuer),
        "response_types_supported": ["code"],
        "grant_types_supported": [
            "authorization_code",
            "refresh_token",
            "urn:ietf:params:oauth:grant-type:device_code",
        ],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": state.default_scopes,
        "client_id_metadata_document_supported": true,
    });

    let jwt = build_entity_configuration(
        issuer,
        sk,
        &vk,
        as_metadata,
        &state.authority_hints,
    );

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/entity-statement+jwt")],
        jwt,
    ).into_response()
}

/// Build and sign an OpenID Federation 1.0 entity configuration JWT.
///
/// The JWT `typ` is `entity-statement+jwt` (not `JWT`), so we cannot use the
/// existing `jwt::encode()` helper which hardcodes `"typ":"JWT"`. We build it
/// directly using standard base64url encoding.
pub fn build_entity_configuration(
    issuer_url: &str,
    signing_key: &SigningKey,
    verifying_key: &VerifyingKey,
    as_metadata: serde_json::Value,
    authority_hints: &[String],
) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let exp = now + 86400; // 24 hours

    let header = base64url_json(serde_json::json!({
        "alg": "EdDSA",
        "typ": "entity-statement+jwt",
    }));

    let payload = base64url_json(serde_json::json!({
        "iss": issuer_url,
        "sub": issuer_url,
        "iat": now,
        "exp": exp,
        "jwks": {
            "keys": [verifying_key_as_okp_jwk(verifying_key)],
        },
        "metadata": {
            "oauth_authorization_server": as_metadata,
            "federation_entity": {
                "organization_name": "hyprstream",
            },
        },
        "authority_hints": authority_hints,
    }));

    let signing_input = format!("{header}.{payload}");
    let signature = signing_key.sign(signing_input.as_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());

    format!("{signing_input}.{sig_b64}")
}

/// Serialize a JSON value to base64url-encoded bytes (no padding).
fn base64url_json(value: serde_json::Value) -> String {
    // serde_json::to_vec on a Value never fails; unwrap_or_else avoids clippy::expect_used
    let json_bytes = serde_json::to_vec(&value).unwrap_or_else(|_| unreachable!("Value serialization cannot fail"));
    URL_SAFE_NO_PAD.encode(&json_bytes)
}

/// Represent an Ed25519 verifying key as an OKP JWK (RFC 8037).
fn verifying_key_as_okp_jwk(vk: &VerifyingKey) -> serde_json::Value {
    let x_b64 = URL_SAFE_NO_PAD.encode(vk.as_bytes());
    serde_json::json!({
        "kty": "OKP",
        "crv": "Ed25519",
        "x": x_b64,
        "use": "sig",
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn make_key() -> SigningKey {
        SigningKey::generate(&mut rand::rngs::OsRng)
    }

    #[test]
    fn test_entity_configuration_structure() {
        let sk = make_key();
        let vk = sk.verifying_key();
        let as_meta = serde_json::json!({"issuer": "http://localhost:6791"});
        let jwt = build_entity_configuration(
            "http://localhost:6791",
            &sk,
            &vk,
            as_meta,
            &[],
        );

        let parts: Vec<&str> = jwt.split('.').collect();
        assert_eq!(parts.len(), 3, "JWT must have 3 parts");

        // Decode header
        let header_json = URL_SAFE_NO_PAD.decode(parts[0]).expect("header is valid base64url");
        let header: serde_json::Value = serde_json::from_slice(&header_json).expect("header is valid JSON");
        assert_eq!(header["alg"], "EdDSA");
        assert_eq!(header["typ"], "entity-statement+jwt");

        // Decode payload
        let payload_json = URL_SAFE_NO_PAD.decode(parts[1]).expect("payload is valid base64url");
        let payload: serde_json::Value = serde_json::from_slice(&payload_json).expect("payload is valid JSON");
        assert_eq!(payload["iss"], "http://localhost:6791");
        assert_eq!(payload["sub"], payload["iss"]);
        assert!(payload["jwks"]["keys"].as_array().unwrap().len() == 1);
        assert_eq!(payload["jwks"]["keys"][0]["kty"], "OKP");
        assert_eq!(payload["jwks"]["keys"][0]["crv"], "Ed25519");
        assert!(payload["metadata"]["oauth_authorization_server"].is_object());
    }

    #[test]
    fn test_entity_configuration_signature_verifies() {
        let sk = make_key();
        let vk = sk.verifying_key();
        let as_meta = serde_json::json!({"issuer": "http://example.com"});
        let jwt = build_entity_configuration("http://example.com", &sk, &vk, as_meta, &[]);

        let parts: Vec<&str> = jwt.split('.').collect();
        assert_eq!(parts.len(), 3);

        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let sig_bytes = URL_SAFE_NO_PAD.decode(parts[2]).expect("signature is valid base64url");
        let sig_array: [u8; 64] = sig_bytes.try_into().expect("signature must be 64 bytes");
        let sig = ed25519_dalek::Signature::from_bytes(&sig_array);

        assert!(
            vk.verify_strict(signing_input.as_bytes(), &sig).is_ok(),
            "entity configuration JWT signature must verify"
        );
    }
}
