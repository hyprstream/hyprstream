//! did:web document endpoints (Phase 0c).
//!
//! Serves DID documents at:
//!   - `GET /.well-known/did.json` — root deployment DID (controller for all keys
//!     under this issuer's authority)
//!   - `GET /users/:username/did.json` — per-user DID document with that user's
//!     registered Ed25519 verification methods
//!   - `GET /clients/:client_id/did.json` — per-client DID document (for Tier 3
//!     confidential clients with registered keys)
//!
//! All documents follow the W3C DID Core 1.0 + did:web 1.0 conventions plus
//! the `Ed25519VerificationKey2020` verification-method type (Multibase z6Mk
//! encoding per the spec).
//!
//! The format produced here matches the architecture-doc Subject Identity
//! Format section and is consumed by:
//!   - Federation peers verifying did:web subjects in tokens
//!   - Clients that want to discover an issuer's signing keys without OAuth
//!     metadata fetch
//!   - The Phase 1c `private_key_jwt` flow's client-key lookup

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::VerifyingKey;
use serde_json::{json, Value};

use super::state::OAuthState;

/// Extract the authority component (host[:port]) from the OAuth issuer URL.
///
/// Used as the method-specific identifier in did:web — `did:web:{authority}:...`.
/// Mirrors the helper in `user_mapping.rs` but kept inline here to avoid
/// pulling in the user-mapping module from a different concern.
fn issuer_authority(issuer_url: &str) -> Option<String> {
    let after_scheme = issuer_url.split_once("://").map(|(_, rest)| rest)?;
    let authority = after_scheme.split('/').next().unwrap_or(after_scheme);
    if authority.is_empty() { None } else { Some(authority.to_owned()) }
}

/// Build the multibase z-encoded Ed25519 public-key string per the
/// `Ed25519VerificationKey2020` spec.
///
/// The multibase prefix `z` means base58btc; the actual key payload is
/// `0xed 0x01 || raw_pubkey_bytes` (multicodec ed25519-pub varint header).
fn ed25519_to_multibase(vk: &VerifyingKey) -> String {
    // 0xed01 is the multicodec prefix for ed25519-pub (RFC: did:key spec).
    let mut payload = Vec::with_capacity(34);
    payload.push(0xed);
    payload.push(0x01);
    payload.extend_from_slice(vk.as_bytes());
    format!("z{}", bs58::encode(payload).into_string())
}

/// Build the verification-method JSON for a single Ed25519 key under a
/// did:web subject.
fn ed25519_verification_method(did: &str, key_id: &str, vk: &VerifyingKey) -> Value {
    json!({
        "id": format!("{did}#{key_id}"),
        "type": "Ed25519VerificationKey2020",
        "controller": did,
        "publicKeyMultibase": ed25519_to_multibase(vk),
    })
}

/// Build a JWK fallback verification method (useful for consumers that
/// don't speak Multibase but understand JWK).
fn ed25519_verification_method_jwk(did: &str, key_id: &str, vk: &VerifyingKey) -> Value {
    json!({
        "id": format!("{did}#{key_id}-jwk"),
        "type": "JsonWebKey2020",
        "controller": did,
        "publicKeyJwk": {
            "kty": "OKP",
            "crv": "Ed25519",
            "x": URL_SAFE_NO_PAD.encode(vk.as_bytes()),
        },
    })
}

/// Construct the DID document for a given did:web subject.
///
/// `did` is the full did:web identifier; `keys` is the ordered list of
/// `(key_id, VerifyingKey)` pairs to include as verification methods.
fn build_did_document(
    did: &str,
    issuer_url: &str,
    keys: &[(String, VerifyingKey)],
) -> Value {
    let mut verification_methods = Vec::with_capacity(keys.len() * 2);
    let mut authentication_refs = Vec::with_capacity(keys.len() * 2);

    for (key_id, vk) in keys {
        let vm_id = format!("{did}#{key_id}");
        let vm_id_jwk = format!("{did}#{key_id}-jwk");
        verification_methods.push(ed25519_verification_method(did, key_id, vk));
        verification_methods.push(ed25519_verification_method_jwk(did, key_id, vk));
        authentication_refs.push(Value::String(vm_id));
        authentication_refs.push(Value::String(vm_id_jwk));
    }

    json!({
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://w3id.org/security/suites/ed25519-2020/v1",
            "https://w3id.org/security/suites/jws-2020/v1",
        ],
        "id": did,
        "verificationMethod": verification_methods,
        "authentication": authentication_refs,
        "assertionMethod": authentication_refs,
        "service": [{
            "id": format!("{did}#hyprstream"),
            "type": "HyprstreamService",
            "serviceEndpoint": issuer_url,
        }],
    })
}

/// `GET /.well-known/did.json` — root deployment DID document.
///
/// `id = did:web:{authority}`. Verification methods: the OAuth issuer's
/// current signing key (entity-signing key from OAuthState). Acts as the
/// trust anchor that controls user/client DIDs under this authority.
pub async fn root_did_document(
    State(state): State<Arc<OAuthState>>,
) -> Response {
    let authority = match issuer_authority(&state.issuer_url) {
        Some(a) => a,
        None => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "issuer URL has no authority",
        ).into_response(),
    };
    let did = format!("did:web:{authority}");

    // Use the OAuth signing key as the root verification method.
    let Some(ref sk) = state.signing_key else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "OAuth signing key not configured",
        ).into_response();
    };
    let vk = sk.verifying_key();

    let doc = build_did_document(&did, &state.issuer_url, &[("key-1".to_owned(), vk)]);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/did+json")],
        Json(doc),
    ).into_response()
}

/// `GET /users/:username/did.json` — per-user DID document.
///
/// `id = did:web:{authority}:users:{username}`. Verification methods:
/// every Ed25519 pubkey the user has registered via SCIM
/// (`list_pubkeys`). Empty `verificationMethod` array if the user has no
/// registered keys — that's a valid DID document; it just means no
/// authentication keys are bound.
pub async fn user_did_document(
    State(state): State<Arc<OAuthState>>,
    Path(username): Path<String>,
) -> Response {
    let authority = match issuer_authority(&state.issuer_url) {
        Some(a) => a,
        None => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "issuer URL has no authority",
        ).into_response(),
    };

    // Reject usernames containing characters that would break the did:web
    // path component or imply arbitrary path traversal.
    if username.contains(['/', '#', '?', ':']) || username.is_empty() {
        return (StatusCode::BAD_REQUEST, "invalid username").into_response();
    }
    let did = format!("did:web:{authority}:users:{username}");

    // Resolve user keys via UserStore.list_pubkeys. If the user store is
    // unavailable or the user has no profile, serve an empty DID document
    // (404 would be more strictly correct, but did:web consumers tend to
    // tolerate empty verificationMethod better than HTTP errors and we
    // want this endpoint to be usable as a presence check).
    let keys: Vec<(String, VerifyingKey)> = match state.user_service.as_ref() {
        Some(user_svc) => {
            match user_svc.store().list_pubkeys(&username).await {
                Ok(pubkeys) => pubkeys
                    .into_iter()
                    .map(|pk| (format!("key-{}", &pk.fingerprint[..8.min(pk.fingerprint.len())]), pk.pubkey))
                    .collect(),
                Err(_) => Vec::new(),
            }
        }
        None => Vec::new(),
    };

    let doc = build_did_document(&did, &state.issuer_url, &keys);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/did+json")],
        Json(doc),
    ).into_response()
}

/// `GET /clients/:client_id/did.json` — per-client DID document.
///
/// `id = did:web:{authority}:clients:{client_id}`. Verification methods:
/// JWKS keys registered for the client via dynamic-client-registration's
/// `jwks` field (Tier 3 confidential clients with `private_key_jwt`).
pub async fn client_did_document(
    State(state): State<Arc<OAuthState>>,
    Path(client_id): Path<String>,
) -> Response {
    let authority = match issuer_authority(&state.issuer_url) {
        Some(a) => a,
        None => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "issuer URL has no authority",
        ).into_response(),
    };
    if client_id.contains(['/', '#', '?', ':']) || client_id.is_empty() {
        return (StatusCode::BAD_REQUEST, "invalid client_id").into_response();
    }
    let did = format!("did:web:{authority}:clients:{client_id}");

    // Client JWKS storage will be wired up in Phase 1b (client-key
    // registration CLI + server-side JWKS persistence). For now,
    // return an empty verificationMethod list — the DID document is
    // structurally valid; consumers learn that no keys are bound yet.
    //
    // Once RegisteredClient grows a `jwks: Option<serde_json::Value>`
    // field, replace this with a `clients.read().await.get(client_id)`
    // lookup + extract_ed25519_keys_from_jwks call.
    let keys: Vec<(String, VerifyingKey)> = Vec::new();

    let doc = build_did_document(&did, &state.issuer_url, &keys);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/did+json")],
        Json(doc),
    ).into_response()
}

/// Extract Ed25519 VerifyingKeys from a JWKS JSON value.
///
/// Currently unused: client JWKS storage is a Phase 1b deliverable. This
/// helper is retained (and exercised by tests) so the call-site in
/// `client_did_document` is a one-line activation once `RegisteredClient`
/// grows a `jwks` field.
#[allow(dead_code)]
fn extract_ed25519_keys_from_jwks(jwks: &Option<Value>) -> Vec<(String, VerifyingKey)> {
    let Some(jwks) = jwks else { return Vec::new() };
    let Some(keys) = jwks.get("keys").and_then(|k| k.as_array()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (idx, key) in keys.iter().enumerate() {
        if key.get("kty").and_then(|v| v.as_str()) != Some("OKP")
            || key.get("crv").and_then(|v| v.as_str()) != Some("Ed25519")
        {
            continue;
        }
        let Some(x) = key.get("x").and_then(|v| v.as_str()) else { continue };
        let Ok(raw) = URL_SAFE_NO_PAD.decode(x) else { continue };
        let bytes: [u8; 32] = match raw.try_into() {
            Ok(b) => b,
            Err(_) => continue,
        };
        let Ok(vk) = VerifyingKey::from_bytes(&bytes) else { continue };
        let kid = key
            .get("kid")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("key-{idx}"));
        out.push((kid, vk));
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn issuer_authority_with_port() {
        assert_eq!(
            issuer_authority("http://127.0.0.1:6791").as_deref(),
            Some("127.0.0.1:6791"),
        );
    }

    #[test]
    fn issuer_authority_https_no_port() {
        assert_eq!(
            issuer_authority("https://hyprstream.example.com").as_deref(),
            Some("hyprstream.example.com"),
        );
    }

    #[test]
    fn issuer_authority_strips_path() {
        assert_eq!(
            issuer_authority("https://example.com/oauth/issuer").as_deref(),
            Some("example.com"),
        );
    }

    #[test]
    fn issuer_authority_rejects_no_scheme() {
        assert_eq!(issuer_authority("example.com"), None);
    }

    #[test]
    fn ed25519_multibase_format() {
        let sk = SigningKey::generate(&mut OsRng);
        let mb = ed25519_to_multibase(&sk.verifying_key());
        assert!(mb.starts_with('z'), "must use base58btc multibase prefix");
        // Decode + check the multicodec prefix is ed25519-pub (0xed 0x01).
        let decoded = bs58::decode(&mb[1..]).into_vec().unwrap();
        assert_eq!(decoded[0], 0xed);
        assert_eq!(decoded[1], 0x01);
        assert_eq!(&decoded[2..], sk.verifying_key().as_bytes());
    }

    #[test]
    fn build_did_doc_minimum_structure() {
        let sk = SigningKey::generate(&mut OsRng);
        let did = "did:web:hyprstream.example.com:users:alice";
        let doc = build_did_document(
            did,
            "https://hyprstream.example.com",
            &[("key-1".to_owned(), sk.verifying_key())],
        );
        assert_eq!(doc["id"].as_str().unwrap(), did);
        assert!(doc["@context"].is_array());
        assert_eq!(doc["verificationMethod"].as_array().unwrap().len(), 2); // multibase + jwk
        assert_eq!(doc["service"][0]["type"].as_str().unwrap(), "HyprstreamService");
        assert_eq!(
            doc["service"][0]["serviceEndpoint"].as_str().unwrap(),
            "https://hyprstream.example.com",
        );
    }

    #[test]
    fn build_did_doc_empty_keys_is_valid() {
        let did = "did:web:example.com:users:alice";
        let doc = build_did_document(did, "https://example.com", &[]);
        assert_eq!(doc["id"].as_str().unwrap(), did);
        assert_eq!(doc["verificationMethod"].as_array().unwrap().len(), 0);
        assert_eq!(doc["authentication"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn extract_jwks_filters_non_ed25519() {
        let jwks = json!({
            "keys": [
                {"kty": "OKP", "crv": "Ed25519", "x": URL_SAFE_NO_PAD.encode([1u8; 32]), "kid": "good"},
                {"kty": "EC", "crv": "P-256", "kid": "p256-not-ed25519"},
                {"kty": "OKP", "crv": "X25519", "kid": "wrong-curve"},
            ]
        });
        let keys = extract_ed25519_keys_from_jwks(&Some(jwks));
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].0, "good");
    }
}
