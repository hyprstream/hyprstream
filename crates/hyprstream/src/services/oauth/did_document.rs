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
//! All documents follow the W3C DID Core 1.0 + did:web 1.0 conventions. All
//! key-bearing verification methods use the `Multikey` type with a
//! multicodec-prefixed `publicKeyMultibase` (ed25519 z6Mk form; p256 / ML-DSA-65
//! likewise), matching the atproto `#atproto` VM and the `did:key` encoding.
//!
//! The **root** document is additionally **atproto-compatible** (#154): when an
//! ES256 key store is present it carries a P-256 `#atproto` `Multikey`, an
//! `#atproto_pds` service, and `alsoKnownAs at://{handle}` — the shape existing
//! atproto resolvers expect. Our Ed25519 mesh verification method and any typed
//! transport `service` entries (IrohTransport/QuicTransport/OnionTransport) are
//! **optional, additive**, and ignored by atproto (which matches by id/type).
//! P-256 satisfies atproto (it accepts p256/k256); k256 is not used.
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

/// Extract the origin (scheme://authority) from the OAuth issuer URL.
///
/// atproto's `#atproto_pds` serviceEndpoint must be an origin only — scheme,
/// host, optional port — with no path. (atproto/specs/did)
fn issuer_origin(issuer_url: &str) -> Option<String> {
    let (scheme, rest) = issuer_url.split_once("://")?;
    let authority = rest.split('/').next().unwrap_or(rest);
    if authority.is_empty() {
        None
    } else {
        Some(format!("{scheme}://{authority}"))
    }
}

/// Build the multibase z-encoded P-256 public key per the `Multikey` /
/// did:key conventions used by atproto.
///
/// Multibase prefix `z` (base58btc); payload is the multicodec `p256-pub`
/// unsigned-varint header (`0x1200` → bytes `0x80 0x24`) followed by the
/// **compressed** SEC1 point (33 bytes). atproto requires compressed pubkeys.
fn p256_to_multibase(vk: &p256::ecdsa::VerifyingKey) -> String {
    let point = vk.to_encoded_point(true); // compressed
    let compressed = point.as_bytes();
    let mut payload = Vec::with_capacity(2 + compressed.len());
    payload.push(0x80);
    payload.push(0x24);
    payload.extend_from_slice(compressed);
    format!("z{}", bs58::encode(payload).into_string())
}

/// multicodec `ml-dsa-65-pub` unsigned-varint prefix.
///
/// Code point `0x1211` (FIPS 204 / ML-DSA-65 public key) from the multiformats
/// multicodec registry (status: draft); see also draft-ietf-cose-mldsa. As an
/// unsigned varint `0x1211` encodes to the two bytes `0x91 0x24` — same shape as
/// `p256-pub` (`0x1200` → `0x80 0x24`).
const MULTICODEC_ML_DSA_65_PUB: [u8; 2] = [0x91, 0x24];

/// Build the multibase z-encoded ML-DSA-65 public key as a `Multikey`-compatible
/// `publicKeyMultibase` value.
///
/// Multibase prefix `z` (base58btc); payload is the multicodec `ml-dsa-65-pub`
/// unsigned-varint header (`0x1211` → bytes `0x91 0x24`) followed by the raw
/// ML-DSA-65 verifying-key bytes (1952 bytes, FIPS 204). The input is the raw
/// public-key bytes (e.g. from `Signer::pq_pubkey()` / `ml_dsa_vk_bytes`), so
/// this helper stays decoupled from the concrete `ml_dsa` key type and can be
/// called by #157 when it publishes the `#mesh` PQ key.
fn mldsa65_to_multibase(vk_bytes: &[u8]) -> String {
    let mut payload = Vec::with_capacity(MULTICODEC_ML_DSA_65_PUB.len() + vk_bytes.len());
    payload.extend_from_slice(&MULTICODEC_ML_DSA_65_PUB);
    payload.extend_from_slice(vk_bytes);
    format!("z{}", bs58::encode(payload).into_string())
}

/// The node's atproto-native identity to embed in the root DID document:
/// the P-256 signing key (published as the `#atproto` Multikey) plus the
/// account handle (published in `alsoKnownAs` as `at://{handle}`).
pub struct AtprotoIdentity<'a> {
    pub p256_vk: &'a p256::ecdsa::VerifyingKey,
    pub handle: &'a str,
}

/// An optional, additive transport endpoint advertised as a typed `service`
/// entry (DIDComm-style map). atproto resolvers ignore these; the mesh dial
/// resolves by `service.type`.
pub struct TransportEndpoint {
    /// Fragment id, e.g. `"iroh"` → `{did}#iroh`.
    pub fragment: String,
    /// Service `type`, e.g. `"IrohTransport"`.
    pub vm_type: String,
    /// The `serviceEndpoint` map (e.g. `{ "uri": ..., "accept": [...] }`).
    pub endpoint: Value,
}

/// Build the atproto `#atproto` verification method (P-256 `Multikey`).
fn atproto_verification_method(did: &str, vk: &p256::ecdsa::VerifyingKey) -> Value {
    json!({
        "id": format!("{did}#atproto"),
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": p256_to_multibase(vk),
    })
}

/// Build the verification-method JSON for a single Ed25519 key under a
/// did:web subject.
///
/// Emitted as `type: "Multikey"` (W3C / atproto canonical VM type): the
/// `publicKeyMultibase` value `ed25519_to_multibase` produces is already the
/// multicodec-prefixed (`0xed01`) base58btc form `Multikey` requires, so this
/// is the same shape used for `#atproto` (p256) and matches the `did:key`
/// encoding (multicodec/multibase) for cross-tool interop.
fn ed25519_verification_method(did: &str, key_id: &str, vk: &VerifyingKey) -> Value {
    json!({
        "id": format!("{did}#{key_id}"),
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": ed25519_to_multibase(vk),
    })
}

/// Build the verification-method JSON for an ML-DSA-65 (post-quantum) key under
/// a did:web subject, emitted as a `Multikey` with the registered
/// `ml-dsa-65-pub` multicodec (`0x1211`).
///
/// `vk_bytes` is the raw ML-DSA-65 verifying-key (1952 bytes). The DID document
/// builder does not yet publish a PQ key (that is #157's job — populating
/// `KeyedPqTrustStore` from the `#mesh` ML-DSA key); this helper exists so #157
/// can emit the VM with the settled multicodec encoding (e.g. as `#mesh-pq`).
#[allow(dead_code)]
fn mldsa65_verification_method(did: &str, key_id: &str, vk_bytes: &[u8]) -> Value {
    json!({
        "id": format!("{did}#{key_id}"),
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": mldsa65_to_multibase(vk_bytes),
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
/// `atproto` (when `Some`) makes this an atproto-compatible identity document:
/// the P-256 `#atproto` Multikey is listed FIRST in `verificationMethod`, an
/// `#atproto_pds` service is added FIRST in `service`, and `alsoKnownAs` carries
/// `at://{handle}`. The Ed25519 `keys` remain as additional (mesh) verification
/// methods, and `transports` are additional typed `service` entries — both are
/// ignored by atproto resolvers (which match by id/type) but used by our mesh.
fn build_did_document(
    did: &str,
    issuer_url: &str,
    keys: &[(String, VerifyingKey)],
    atproto: Option<&AtprotoIdentity<'_>>,
    transports: &[TransportEndpoint],
) -> Value {
    let mut verification_methods = Vec::with_capacity(keys.len() * 2 + 1);
    let mut authentication_refs = Vec::with_capacity(keys.len() * 2 + 1);
    let mut assertion_refs = Vec::with_capacity(keys.len() * 2 + 1);

    // atproto signing key FIRST (atproto takes the first matching entry).
    if let Some(at) = atproto {
        verification_methods.push(atproto_verification_method(did, at.p256_vk));
        let atproto_vm_id = format!("{did}#atproto");
        authentication_refs.push(Value::String(atproto_vm_id.clone()));
        assertion_refs.push(Value::String(atproto_vm_id));
    }

    // Ed25519 mesh / OAuth verification methods (multibase + JWK).
    for (key_id, vk) in keys {
        let vm_id = format!("{did}#{key_id}");
        let vm_id_jwk = format!("{did}#{key_id}-jwk");
        verification_methods.push(ed25519_verification_method(did, key_id, vk));
        verification_methods.push(ed25519_verification_method_jwk(did, key_id, vk));
        authentication_refs.push(Value::String(vm_id.clone()));
        authentication_refs.push(Value::String(vm_id_jwk.clone()));
        assertion_refs.push(Value::String(vm_id));
        assertion_refs.push(Value::String(vm_id_jwk));
    }

    // Services: atproto PDS first, then the legacy HyprstreamService, then any
    // typed transport endpoints (optional, atproto-ignored).
    let mut services = Vec::with_capacity(2 + transports.len());
    if atproto.is_some() {
        if let Some(origin) = issuer_origin(issuer_url) {
            services.push(json!({
                "id": format!("{did}#atproto_pds"),
                "type": "AtprotoPersonalDataServer",
                "serviceEndpoint": origin,
            }));
        }
    }
    services.push(json!({
        "id": format!("{did}#hyprstream"),
        "type": "HyprstreamService",
        "serviceEndpoint": issuer_url,
    }));
    for t in transports {
        services.push(json!({
            "id": format!("{did}#{}", t.fragment),
            "type": t.vm_type,
            "serviceEndpoint": t.endpoint,
        }));
    }

    let mut doc = json!({
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://w3id.org/security/multikey/v1",
            "https://w3id.org/security/suites/ed25519-2020/v1",
            "https://w3id.org/security/suites/jws-2020/v1",
        ],
        "id": did,
        "verificationMethod": verification_methods,
        "authentication": authentication_refs,
        "assertionMethod": assertion_refs,
        "service": services,
    });

    if let Some(at) = atproto {
        doc["alsoKnownAs"] = json!([format!("at://{}", at.handle)]);
    }

    doc
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

    // atproto-native identity: the active P-256 key from the ES256 rotation
    // store becomes the `#atproto` Multikey; the issuer authority is the handle.
    let atproto_sk = match state.es256_key_store.as_ref() {
        Some(store) => store.active_key().await,
        None => None,
    };
    // The atproto handle is a bare hostname (no port); strip any port the
    // authority carries (the DID identifier keeps the port, the handle does not).
    let handle = authority.split(':').next().unwrap_or(authority.as_str());
    let atproto_vk = atproto_sk.as_ref().map(|sk| sk.verifying_key());
    let atproto = atproto_vk.map(|vk| AtprotoIdentity {
        p256_vk: vk,
        handle,
    });

    // Transport `service` entries: populate QUIC entry when cert hash is available (#185).
    // The cert hash was set at OAuthService startup from the node's QUIC TLS cert,
    // closing the two-trust-roots gap: peers dialing by DID can now pin the cert
    // instead of accepting it on TOFU.
    let mut transports: Vec<TransportEndpoint> = Vec::new();
    if !state.quic_cert_hashes.is_empty() {
        if let Some(ref quic_uri) = state.quic_public_uri {
            let auth = match hyprstream_rpc::transport::QuicServerAuth::pinned(
                state.quic_cert_hashes.clone(),
            ) {
                Ok(auth) => auth,
                Err(_) => hyprstream_rpc::transport::QuicServerAuth::web_pki(),
            };
            transports.push(TransportEndpoint {
                fragment: "quic".to_owned(),
                vm_type: "QuicTransport".to_owned(),
                endpoint: hyprstream_rpc::service_entry::encode_quic(
                    quic_uri,
                    &auth,
                    &["hyprstream-rpc/1"],
                ),
            });
        }
    }

    let doc = build_did_document(
        &did,
        &state.issuer_url,
        &[("key-1".to_owned(), vk)],
        atproto.as_ref(),
        &transports,
    );
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

    let doc = build_did_document(&did, &state.issuer_url, &keys, None, &[]);
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

    let doc = build_did_document(&did, &state.issuer_url, &keys, None, &[]);
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
            None,
            &[],
        );
        assert_eq!(doc["id"].as_str().unwrap(), did);
        assert!(doc["@context"].is_array());
        assert_eq!(doc["verificationMethod"].as_array().unwrap().len(), 2); // multibase + jwk
        // The ed25519 VM is emitted as a Multikey (#280) with a multibase key.
        let vm = &doc["verificationMethod"][0];
        assert_eq!(vm["type"].as_str().unwrap(), "Multikey");
        assert!(vm["publicKeyMultibase"].as_str().unwrap().starts_with('z'));
        // The JWK fallback VM is retained.
        assert_eq!(doc["verificationMethod"][1]["type"].as_str().unwrap(), "JsonWebKey2020");
        assert_eq!(doc["service"][0]["type"].as_str().unwrap(), "HyprstreamService");
        assert_eq!(
            doc["service"][0]["serviceEndpoint"].as_str().unwrap(),
            "https://hyprstream.example.com",
        );
    }

    #[test]
    fn build_did_doc_empty_keys_is_valid() {
        let did = "did:web:example.com:users:alice";
        let doc = build_did_document(did, "https://example.com", &[], None, &[]);
        assert_eq!(doc["id"].as_str().unwrap(), did);
        assert_eq!(doc["verificationMethod"].as_array().unwrap().len(), 0);
        assert_eq!(doc["authentication"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn p256_multibase_format() {
        let sk = p256::ecdsa::SigningKey::random(&mut OsRng);
        let mb = p256_to_multibase(sk.verifying_key());
        assert!(mb.starts_with('z'), "must use base58btc multibase prefix");
        let decoded = bs58::decode(&mb[1..]).into_vec().unwrap();
        // multicodec p256-pub = 0x1200 → unsigned-varint [0x80, 0x24].
        assert_eq!(decoded[0], 0x80);
        assert_eq!(decoded[1], 0x24);
        // Compressed SEC1 point: 33 bytes, leading 0x02/0x03.
        assert_eq!(decoded.len(), 2 + 33);
        assert!(decoded[2] == 0x02 || decoded[2] == 0x03);
    }

    #[test]
    fn build_did_doc_atproto_identity() {
        let ed = SigningKey::generate(&mut OsRng);
        let p256_sk = p256::ecdsa::SigningKey::random(&mut OsRng);
        let p256_vk = p256_sk.verifying_key();
        let did = "did:web:hyprstream.example.com";
        let atproto = AtprotoIdentity { p256_vk, handle: "hyprstream.example.com" };
        let transports = [TransportEndpoint {
            fragment: "iroh".to_owned(),
            vm_type: "IrohTransport".to_owned(),
            endpoint: json!({ "uri": "iroh:abc", "accept": ["hyprstream-rpc/1", "moql"] }),
        }];
        let doc = build_did_document(
            did,
            // issuer URL with a path — origin must strip it for #atproto_pds.
            "https://hyprstream.example.com/oauth",
            &[("key-1".to_owned(), ed.verifying_key())],
            Some(&atproto),
            &transports,
        );

        // #atproto Multikey is FIRST and p256.
        let vms = doc["verificationMethod"].as_array().unwrap();
        assert_eq!(vms[0]["id"].as_str().unwrap(), format!("{did}#atproto"));
        assert_eq!(vms[0]["type"].as_str().unwrap(), "Multikey");
        assert!(vms[0]["publicKeyMultibase"].as_str().unwrap().starts_with('z'));
        // Ed25519 mesh VMs still present after it.
        assert_eq!(vms.len(), 1 + 2);
        // The ed25519 mesh VM is also a Multikey (#280), not the deprecated
        // Ed25519VerificationKey2020 type, with a multibase key; JWK fallback kept.
        assert_eq!(vms[1]["type"].as_str().unwrap(), "Multikey");
        assert!(vms[1]["publicKeyMultibase"].as_str().unwrap().starts_with('z'));
        assert_eq!(vms[2]["type"].as_str().unwrap(), "JsonWebKey2020");

        // #atproto_pds first, origin-only (no path), correct type.
        let svcs = doc["service"].as_array().unwrap();
        assert_eq!(svcs[0]["id"].as_str().unwrap(), format!("{did}#atproto_pds"));
        assert_eq!(svcs[0]["type"].as_str().unwrap(), "AtprotoPersonalDataServer");
        assert_eq!(
            svcs[0]["serviceEndpoint"].as_str().unwrap(),
            "https://hyprstream.example.com"
        );
        // transport entry present as a typed map service.
        let iroh = svcs.iter().find(|s| s["type"] == "IrohTransport").unwrap();
        assert_eq!(iroh["id"].as_str().unwrap(), format!("{did}#iroh"));
        assert!(iroh["serviceEndpoint"]["accept"].is_array());

        // alsoKnownAs handle.
        assert_eq!(doc["alsoKnownAs"][0].as_str().unwrap(), "at://hyprstream.example.com");
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

    #[test]
    fn mesh_ed25519_vm_is_multikey() {
        // The `#mesh` verification method must be emitted as a Multikey (#280),
        // not the deprecated Ed25519VerificationKey2020 type.
        let sk = SigningKey::generate(&mut OsRng);
        let did = "did:web:hyprstream.example.com";
        let vm = ed25519_verification_method(did, "mesh", &sk.verifying_key());
        assert_eq!(vm["id"].as_str().unwrap(), format!("{did}#mesh"));
        assert_eq!(vm["type"].as_str().unwrap(), "Multikey");
        assert_eq!(vm["controller"].as_str().unwrap(), did);
        let mb = vm["publicKeyMultibase"].as_str().unwrap();
        assert!(mb.starts_with('z'), "Multikey publicKeyMultibase must be base58btc multibase");
        // The encoded key carries the ed25519-pub multicodec prefix (0xed 0x01).
        let decoded = bs58::decode(&mb[1..]).into_vec().unwrap();
        assert_eq!(&decoded[..2], &[0xed, 0x01]);
        assert_eq!(&decoded[2..], sk.verifying_key().as_bytes());
    }

    #[test]
    fn mldsa65_multibase_multicodec_round_trip() {
        // ML-DSA-65 public keys are 1952 bytes (FIPS 204). Use a stand-in buffer
        // of the correct length to exercise the multicodec/multibase encoding.
        let vk_bytes = vec![0xABu8; 1952];
        let mb = mldsa65_to_multibase(&vk_bytes);
        assert!(mb.starts_with('z'), "must use base58btc multibase prefix");
        let decoded = bs58::decode(&mb[1..]).into_vec().unwrap();
        // multicodec ml-dsa-65-pub = 0x1211 → unsigned-varint [0x91, 0x24].
        assert_eq!(&decoded[..2], &[0x91, 0x24]);
        // Remaining bytes are the verifying-key payload, unmodified.
        assert_eq!(&decoded[2..], vk_bytes.as_slice());
        assert_eq!(decoded.len(), 2 + 1952);
    }

    #[test]
    fn mldsa65_verification_method_is_multikey() {
        let did = "did:web:hyprstream.example.com";
        let vk_bytes = vec![0x07u8; 1952];
        let vm = mldsa65_verification_method(did, "mesh-pq", &vk_bytes);
        assert_eq!(vm["id"].as_str().unwrap(), format!("{did}#mesh-pq"));
        assert_eq!(vm["type"].as_str().unwrap(), "Multikey");
        assert_eq!(vm["controller"].as_str().unwrap(), did);
        let mb = vm["publicKeyMultibase"].as_str().unwrap();
        let decoded = bs58::decode(&mb[1..]).into_vec().unwrap();
        assert_eq!(&decoded[..2], &[0x91, 0x24]); // ml-dsa-65-pub multicodec
        assert_eq!(&decoded[2..], vk_bytes.as_slice());
    }
}
