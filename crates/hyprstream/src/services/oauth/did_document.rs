//! did:web document endpoints (Phase 0c).
//!
//! Serves DID documents at:
//!   - `GET /.well-known/did.json` — root deployment DID (controller for all keys
//!     under this issuer's authority)
//!   - `GET /users/:username/did.json` — hard error while path-form account DID
//!     minting is frozen (#1159)
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
pub(crate) fn issuer_authority(issuer_url: &str) -> Option<String> {
    let after_scheme = issuer_url.split_once("://").map(|(_, rest)| rest)?;
    let authority = after_scheme.split('/').next().unwrap_or(after_scheme);
    if authority.is_empty() {
        None
    } else {
        Some(authority.to_owned())
    }
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

fn normalize_atproto_handle(handle: &str) -> Option<String> {
    let handle = handle.trim_end_matches('.').to_ascii_lowercase();
    if handle.len() > 253 || handle.split('.').count() < 2 {
        return None;
    }
    let valid = handle.split('.').all(|label| {
        !label.is_empty()
            && label.len() <= 63
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    });
    valid.then_some(handle)
}

fn configured_handle_host(issuer_url: &str) -> Option<String> {
    let url = url::Url::parse(issuer_url).ok()?;
    match url.host()? {
        url::Host::Domain(host) => normalize_atproto_handle(host),
        url::Host::Ipv4(_) | url::Host::Ipv6(_) => None,
    }
}

#[cfg(test)]
fn account_handle(username: &str, issuer_url: &str) -> Option<String> {
    let host = configured_handle_host(issuer_url)?;
    normalize_atproto_handle(&format!("{username}.{host}"))
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
/// the active P-256 signing key plus the account handle. Bounded overlap keys
/// use distinct fragments so stock atproto resolvers continue selecting only
/// the active `#atproto` method.
pub struct AtprotoIdentity<'a> {
    pub p256_vk: &'a p256::ecdsa::VerifyingKey,
    pub handle: &'a str,
    pub drain: Option<AtprotoOverlapKey<'a>>,
    pub lead: Option<AtprotoOverlapKey<'a>>,
}

/// A bounded verification-only overlap key. It is never selected for signing.
pub struct AtprotoOverlapKey<'a> {
    pub vk: &'a p256::ecdsa::VerifyingKey,
    pub nbf: i64,
    pub exp: i64,
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
pub(crate) fn atproto_verification_method(did: &str, vk: &p256::ecdsa::VerifyingKey) -> Value {
    json!({
        "id": format!("{did}#atproto"),
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": p256_to_multibase(vk),
    })
}

fn atproto_overlap_verification_method(
    did: &str,
    fragment: &str,
    key: &AtprotoOverlapKey<'_>,
) -> Value {
    json!({
        "id": format!("{did}#{fragment}"),
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": p256_to_multibase(key.vk),
        "nbf": key.nbf,
        "exp": key.exp,
    })
}

/// Build the active plus bounded drain/lead VMs used by the producer. The
/// auxiliary fragments are deliberately distinct: upstream resolvers select
/// the first exact `#atproto` fragment and ignore these entries.
pub fn atproto_verification_methods(
    did: &str,
    active: &p256::ecdsa::VerifyingKey,
    drain: Option<AtprotoOverlapKey<'_>>,
    lead: Option<AtprotoOverlapKey<'_>>,
) -> Vec<Value> {
    let mut methods = vec![atproto_verification_method(did, active)];
    if let Some(drain) = drain {
        methods.push(atproto_overlap_verification_method(
            did,
            "atproto_drain",
            &drain,
        ));
    }
    if let Some(lead) = lead {
        methods.push(atproto_overlap_verification_method(
            did,
            "atproto_lead",
            &lead,
        ));
    }
    methods
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
/// `vk_bytes` is the raw ML-DSA-65 verifying-key (1952 bytes). The root DID
/// document publishes the node's mesh ML-DSA-65 key as `#mesh-pq` (#157) so
/// peers can anchor it in their `KeyedPqTrustStore`.
fn mldsa65_verification_method(did: &str, key_id: &str, vk_bytes: &[u8]) -> Value {
    json!({
        "id": format!("{did}#{key_id}"),
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": mldsa65_to_multibase(vk_bytes),
    })
}

// ============================================================================
// #mesh-kem hybrid keyAgreement identity (S1 / #552, epic #550)
// ============================================================================

/// multicodec `x25519-pub` unsigned-varint prefix (`0xec` → bytes `0xec 0x01`).
const MULTICODEC_X25519_PUB: [u8; 2] = [0xec, 0x01];

/// multicodec `mlkem-768-pub` unsigned-varint prefix.
///
/// Code point `0x120c` (ML-KEM-768 / FIPS 203 encapsulation key) from the
/// multiformats multicodec registry (`table.csv`, status: draft) — confirmed
/// against the upstream registry (NOT `0x120b`, which is `mlkem-512-pub`; a
/// one-code-point mixup there would silently anchor peers to the wrong key
/// size). As an unsigned varint, `0x120c` encodes to the two bytes `0x8c 0x24`
/// (same shape as `ml-dsa-65-pub` `0x1211` → `0x91 0x24`).
const MULTICODEC_ML_KEM_768_PUB: [u8; 2] = [0x8c, 0x24];

/// Multibase z-encode (base58btc, multicodec-prefixed) a raw X25519 encapsulation
/// key (32 bytes) as a `Multikey` `publicKeyMultibase`.
fn x25519_to_multibase(ek_bytes: &[u8]) -> String {
    let mut payload = Vec::with_capacity(MULTICODEC_X25519_PUB.len() + ek_bytes.len());
    payload.extend_from_slice(&MULTICODEC_X25519_PUB);
    payload.extend_from_slice(ek_bytes);
    format!("z{}", bs58::encode(payload).into_string())
}

/// Multibase z-encode a raw ML-KEM-768 encapsulation key (1184 bytes) as a
/// `Multikey` `publicKeyMultibase`.
fn mlkem768_to_multibase(ek_bytes: &[u8]) -> String {
    let mut payload = Vec::with_capacity(MULTICODEC_ML_KEM_768_PUB.len() + ek_bytes.len());
    payload.extend_from_slice(&MULTICODEC_ML_KEM_768_PUB);
    payload.extend_from_slice(ek_bytes);
    format!("z{}", bs58::encode(payload).into_string())
}

/// Build the `#mesh-kem` **keyAgreement** verification methods for the node's
/// hybrid-KEM identity (S1 / #552): one `Multikey` per suite leg (X25519 +
/// ML-KEM-768). These belong in the DID document's `keyAgreement` relationship
/// (NOT `verificationMethod`/`assertionMethod` — they are key-agreement, not
/// signing, keys). Peers reconstruct the policy-pinned hybrid recipient
/// (`SuiteId::HyKemX25519MlKem768`) from the two legs and anchor it in their
/// `crate`-side `KemTrustStore`.
///
/// `x25519_ek` / `mlkem768_ek` are the raw encapsulation keys in suite-component
/// order, from `hyprstream_rpc::node_identity::derive_mesh_kem_recipient(..)
/// .public().eks`.
fn mesh_kem_key_agreement_methods(did: &str, x25519_ek: &[u8], mlkem768_ek: &[u8]) -> Vec<Value> {
    vec![
        json!({
            "id": format!("{did}#mesh-kem-x25519"),
            "type": "Multikey",
            "controller": did,
            "publicKeyMultibase": x25519_to_multibase(x25519_ek),
        }),
        json!({
            "id": format!("{did}#mesh-kem-mlkem768"),
            "type": "Multikey",
            "controller": did,
            "publicKeyMultibase": mlkem768_to_multibase(mlkem768_ek),
        }),
    ]
}

/// Build the `keyAgreement` entries for a node's `#mesh-kem` recipient public
/// material, embedding the VMs directly in the `keyAgreement` relationship
/// array (did:key convention — not cross-referenced via `verificationMethod`,
/// since these are key-agreement-only keys, never used for signing).
///
/// Validates the suite shape defensively (exactly the 2 components of
/// `SuiteId::HyKemX25519MlKem768`, X25519 then ML-KEM-768) before emitting —
/// `recipient_public` only ever comes from this node's own
/// `derive_mesh_kem_recipient`, but a shape mismatch must fail closed (publish
/// an empty `keyAgreement` relationship) rather than publish a malformed or
/// misordered key that a peer would silently mis-anchor.
fn mesh_kem_key_agreement(
    did: &str,
    recipient_public: Option<&hyprstream_rpc::crypto::hybrid_kem::RecipientPublic>,
) -> Vec<Value> {
    use hyprstream_rpc::crypto::hybrid_kem::SuiteId;
    let Some(pub_material) = recipient_public else {
        return Vec::new();
    };
    // `SuiteId::HyKemX25519MlKem768.components()` is `[X25519, MlKem768]` by
    // construction, so pinning the suite id here also pins the component
    // order below.
    if pub_material.suite_id != SuiteId::HyKemX25519MlKem768 {
        return Vec::new();
    }
    let [x25519_ek, mlkem768_ek] = &pub_material.eks[..] else {
        return Vec::new();
    };
    mesh_kem_key_agreement_methods(did, x25519_ek, mlkem768_ek)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod mesh_kem_did_tests {
    use super::*;

    #[test]
    fn mesh_kem_methods_shape_and_multicodec() {
        let did = "did:web:example.com";
        let x_ek = [0x11u8; 32];
        let m_ek = [0x22u8; 1184];
        let vms = mesh_kem_key_agreement_methods(did, &x_ek, &m_ek);
        assert_eq!(vms.len(), 2);
        assert_eq!(vms[0]["id"], format!("{did}#mesh-kem-x25519"));
        assert_eq!(vms[0]["type"], "Multikey");
        assert_eq!(vms[1]["id"], format!("{did}#mesh-kem-mlkem768"));
        assert_eq!(vms[1]["type"], "Multikey");

        // publicKeyMultibase = 'z' + base58btc(multicodec ‖ key); decode + check.
        let mb = vms[0]["publicKeyMultibase"].as_str().unwrap();
        assert!(mb.starts_with('z'));
        let decoded = bs58::decode(&mb[1..]).into_vec().unwrap();
        assert_eq!(&decoded[..2], &MULTICODEC_X25519_PUB);
        assert_eq!(decoded.len(), 2 + 32);

        let mb2 = vms[1]["publicKeyMultibase"].as_str().unwrap();
        let decoded2 = bs58::decode(&mb2[1..]).into_vec().unwrap();
        assert_eq!(&decoded2[..2], &MULTICODEC_ML_KEM_768_PUB);
        assert_eq!(decoded2.len(), 2 + 1184);
    }
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
///
/// `mesh_kem_public` (S1 / #552), when present, publishes the node's
/// `#mesh-kem` hybrid keyAgreement recipient (X25519 + ML-KEM-768) as
/// `keyAgreement` entries — additive and, like `mesh_pq_vk`, ignored by
/// atproto resolvers.
pub(crate) fn build_did_document(
    did: &str,
    issuer_url: &str,
    keys: &[(String, VerifyingKey)],
    atproto: Option<&AtprotoIdentity<'_>>,
    transports: &[TransportEndpoint],
    mesh_pq_vk: Option<&[u8]>,
    mesh_kem_public: Option<&hyprstream_rpc::crypto::hybrid_kem::RecipientPublic>,
) -> Value {
    let mut verification_methods = Vec::with_capacity(keys.len() * 2 + 4);
    let mut authentication_refs = Vec::with_capacity(keys.len() * 2 + 2);
    let mut assertion_refs = Vec::with_capacity(keys.len() * 2 + 2);

    // The active key stays FIRST. Upstream resolvers select this exact fragment;
    // bounded overlap keys use distinct fragments and are Hyprstream-only.
    if let Some(at) = atproto {
        verification_methods.extend(atproto_verification_methods(
            did,
            at.p256_vk,
            at.drain.as_ref().map(|key| AtprotoOverlapKey {
                vk: key.vk,
                nbf: key.nbf,
                exp: key.exp,
            }),
            at.lead.as_ref().map(|key| AtprotoOverlapKey {
                vk: key.vk,
                nbf: key.nbf,
                exp: key.exp,
            }),
        ));
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

    // Mesh post-quantum verification method (#157): the node's ML-DSA-65 mesh
    // key, published as `#mesh-pq` so peers can anchor it in their PQ trust
    // store. Additive and ignored by atproto resolvers (matched by id/type).
    if let Some(vk_bytes) = mesh_pq_vk {
        verification_methods.push(mldsa65_verification_method(did, "mesh-pq", vk_bytes));
        let vm_id = format!("{did}#mesh-pq");
        authentication_refs.push(Value::String(vm_id.clone()));
        assertion_refs.push(Value::String(vm_id));
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

    // `#mesh-kem` hybrid keyAgreement (S1 / #552): embedded directly in the
    // `keyAgreement` relationship (did:key convention) — these are
    // key-agreement-only keys, never cross-referenced from
    // `verificationMethod`/`authentication`/`assertionMethod`.
    let key_agreement = mesh_kem_key_agreement(did, mesh_kem_public);

    let mut doc = json!({
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://w3id.org/security/multikey/v1",
            "https://w3id.org/security/suites/jws-2020/v1",
        ],
        "id": did,
        "verificationMethod": verification_methods,
        "authentication": authentication_refs,
        "assertionMethod": assertion_refs,
        "keyAgreement": key_agreement,
        "service": services,
    });

    if let Some(at) = atproto {
        doc["alsoKnownAs"] = json!([format!("at://{}", at.handle)]);
    }

    doc
}

#[derive(Clone)]
struct RootIdentityMethod {
    fragment: String,
    vk: VerifyingKey,
    pq_vk: Vec<u8>,
    nbf: Option<i64>,
    exp: Option<i64>,
}

fn root_identity_methods(
    legacy_key: &ed25519_dalek::SigningKey,
    legacy_pq: &[u8],
    slots: Option<&crate::auth::key_rotation::KeySlots>,
    now: i64,
) -> Vec<RootIdentityMethod> {
    let Some(slots) = slots else {
        return vec![RootIdentityMethod {
            fragment: "key-1".to_owned(),
            vk: legacy_key.verifying_key(),
            pq_vk: legacy_pq.to_vec(),
            nbf: None,
            exp: None,
        }];
    };

    let legacy_bytes = legacy_key.verifying_key().to_bytes();
    let legacy_slot = [&slots.active, &slots.drain, &slots.lead]
        .into_iter()
        .flatten()
        .find(|slot| slot.verifying_key_bytes() == legacy_bytes && now < slot.exp);

    let mut methods = Vec::new();
    // Preserve the fleet's existing stable fragment during the bounded
    // compatibility window. It aliases the exact same key and PQ binding as
    // its named slot, so upgraded consumers can collapse it safely.
    if let Some(slot) = legacy_slot {
        methods.push(RootIdentityMethod {
            fragment: "key-1".to_owned(),
            vk: legacy_key.verifying_key(),
            pq_vk: legacy_pq.to_vec(),
            nbf: Some(slot.nbf),
            exp: Some(slot.exp),
        });
    }

    // Active is first for legacy order-based consumers after the compatibility
    // alias retires. Drain and lead remain separately named and bounded.
    for slot in [&slots.active, &slots.drain, &slots.lead]
        .into_iter()
        .flatten()
    {
        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&slot.key);
        methods.push(RootIdentityMethod {
            fragment: format!("mesh-{}", slot.kid()),
            vk: slot.key.verifying_key(),
            pq_vk: hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
            nbf: Some(slot.nbf),
            exp: Some(slot.exp),
        });
    }
    methods
}

fn append_root_identity_methods(doc: &mut Value, did: &str, methods: &[RootIdentityMethod]) {
    let Some(verification_methods) = doc
        .get_mut("verificationMethod")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    let mut relationship_ids = Vec::new();

    for method in methods {
        let mut ed = ed25519_verification_method(did, &method.fragment, &method.vk);
        let mut ed_jwk = ed25519_verification_method_jwk(did, &method.fragment, &method.vk);
        let pq_fragment = format!("{}-pq", method.fragment);
        let mut pq = mldsa65_verification_method(did, &pq_fragment, &method.pq_vk);
        for vm in [&mut ed, &mut ed_jwk, &mut pq] {
            if let Some(nbf) = method.nbf {
                vm["nbf"] = nbf.into();
            }
            if let Some(exp) = method.exp {
                vm["exp"] = exp.into();
            }
        }
        relationship_ids.extend([ed["id"].clone(), ed_jwk["id"].clone(), pq["id"].clone()]);
        verification_methods.extend([ed, ed_jwk, pq]);
    }

    // Retain the old `#mesh-pq` DID URL as an alias for the first candidate's
    // PQ half. Old consumers that pair by document order therefore preserve
    // hybrid assurance in both mixed-version directions.
    if let Some(first) = methods.first() {
        let mut legacy_pq = mldsa65_verification_method(did, "mesh-pq", &first.pq_vk);
        if let Some(nbf) = first.nbf {
            legacy_pq["nbf"] = nbf.into();
        }
        if let Some(exp) = first.exp {
            legacy_pq["exp"] = exp.into();
        }
        relationship_ids.push(legacy_pq["id"].clone());
        verification_methods.push(legacy_pq);
    }

    for relationship in ["authentication", "assertionMethod"] {
        if let Some(values) = doc.get_mut(relationship).and_then(Value::as_array_mut) {
            values.extend(relationship_ids.iter().cloned());
        }
    }
}

/// `GET /.well-known/did.json` — root deployment DID document.
///
/// `id = did:web:{authority}`. Verification methods: the OAuth issuer's
/// current signing key (entity-signing key from OAuthState). Acts as the
/// trust anchor that controls user/client DIDs under this authority.
pub async fn root_did_document(State(state): State<Arc<OAuthState>>) -> Response {
    let authority = match issuer_authority(&state.issuer_url) {
        Some(a) => a,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "issuer URL has no authority",
            )
                .into_response()
        }
    };
    let did = format!("did:web:{authority}");

    // Use the OAuth signing key as the root verification method.
    let Some(ref sk) = state.signing_key else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "OAuth signing key not configured",
        )
            .into_response();
    };
    let root_slots = if let Some(store) = &state.root_identity_key_store {
        Some(store.slots_snapshot().await)
    } else {
        None
    };
    let now = chrono::Utc::now().timestamp();
    let root_methods = root_identity_methods(
        sk,
        state.mesh_pq_verifying_key.as_deref().unwrap_or_default(),
        root_slots.as_ref(),
        now,
    );

    // The active key signs new commits. Bounded drain/lead slots are published
    // only for verification overlap; they are never signing candidates.
    let slots = state
        .es256_key_store
        .as_ref()
        .map(|store| store.slots_snapshot())
        .unwrap_or_default();
    let (active_slot, drain_slot, lead_slot) = (slots.active, slots.drain, slots.lead);
    let handle = configured_handle_host(&state.issuer_url);
    let atproto = active_slot
        .as_ref()
        .zip(handle.as_deref())
        .map(|(active, handle)| AtprotoIdentity {
            p256_vk: active.key.verifying_key(),
            handle,
            drain: drain_slot.as_ref().map(|slot| AtprotoOverlapKey {
                vk: slot.key.verifying_key(),
                nbf: slot.nbf,
                exp: slot.exp,
            }),
            lead: lead_slot.as_ref().map(|slot| AtprotoOverlapKey {
                vk: slot.key.verifying_key(),
                nbf: slot.nbf,
                exp: slot.exp,
            }),
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

    // #282: iroh transport entry — published ONLY when the iroh substrate is
    // bound (the daemon set `iroh_node_id`). Advertises only an IrohTransport
    // entry. NodeId is not a VM, JWKS key, or trust anchor.
    if let Some(node_id) = state.iroh_node_id {
        transports.push(TransportEndpoint {
            fragment: "iroh".to_owned(),
            vm_type: "IrohTransport".to_owned(),
            endpoint: hyprstream_rpc::service_entry::encode_iroh(
                &node_id,
                &state.iroh_relays,
                &["hyprstream-rpc/1", "moql"],
            ),
        });
    }

    let mut doc = build_did_document(
        &did,
        &state.issuer_url,
        &[],
        atproto.as_ref(),
        &transports,
        None,
        state.mesh_kem_public.as_ref(),
    );
    append_root_identity_methods(&mut doc, &did, &root_methods);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/did+json")],
        Json(doc),
    )
        .into_response()
}

/// `GET /.well-known/atproto-did` — atproto handle→DID resolution (HTTP method).
///
/// Per the atproto Handle spec (https://atproto.com/specs/handle), the HTTPS
/// well-known method returns the **bare DID string** as `text/plain` (no JSON
/// or wrapper) so this deployment's handle (its authority hostname) resolves to
/// its `did:web:{authority}`. Companion to `/.well-known/did.json`, which serves
/// the full W3C DID document for the same subject.
///
/// Consumed by the frontend handle resolver
/// (`www-cyberdione-ai/src/api/atproto.ts:resolveHandleToDid`). This is a
/// CORS-simple GET (no custom request headers → no preflight), so it needs only
/// cross-origin readability from the public CORS layer, not permissive headers.
pub async fn atproto_did(State(state): State<Arc<OAuthState>>) -> Response {
    let Some(authority) = issuer_authority(&state.issuer_url) else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "issuer URL has no authority",
        )
            .into_response();
    };
    let did = format!("did:web:{authority}");
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        did,
    )
        .into_response()
}

/// `GET /users/:username/did.json` — frozen path-form account DID endpoint.
///
/// This route previously synthesized `did:web:{authority}:users:{username}`.
/// That shape is outside the atproto did:web profile, so serving it would mint
/// another permanent invalid account identifier. Keep the route as an explicit
/// hard error until the separately designed host-form mint path lands (#1163).
pub async fn user_did_document(
    State(_state): State<Arc<OAuthState>>,
    Path(username): Path<String>,
) -> Response {
    tracing::debug!(%username, "deprecated path-form account DID route requested");
    path_form_account_did_disabled_response()
}

fn path_form_account_did_disabled_response() -> Response {
    (
        StatusCode::GONE,
        "did:web path-form account minting is disabled; host-form account minting is not available yet (#1159)",
    )
        .into_response()
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
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "issuer URL has no authority",
            )
                .into_response()
        }
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

    let doc = build_did_document(&did, &state.issuer_url, &keys, None, &[], None, None);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/did+json")],
        Json(doc),
    )
        .into_response()
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
        let Some(x) = key.get("x").and_then(|v| v.as_str()) else {
            continue;
        };
        let Ok(raw) = URL_SAFE_NO_PAD.decode(x) else {
            continue;
        };
        let bytes: [u8; 32] = match raw.try_into() {
            Ok(b) => b,
            Err(_) => continue,
        };
        let Ok(vk) = VerifyingKey::from_bytes(&bytes) else {
            continue;
        };
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

    /// `GET /users/:username/did.json` must be a deliberate 410 hard error,
    /// driven through the **production** OAuth router (`create_app`) — NOT by
    /// calling the `path_form_account_did_disabled_response()` helper directly.
    ///
    /// This is the regression guard for the #1159 path-form account freeze.
    /// The handler must return 410 *before* consulting any user store: a
    /// user-store spy whose `list_pubkeys` panics is wired into the state so
    /// that restoring the pre-PR document-serving body (which resolved user
    /// keys via `list_pubkeys`) fails this test loudly, instead of passing on
    /// a coincidental "no keys found → 200 with an empty document". Asserting
    /// only on the helper would pass either way — that tautology is what this
    /// test replaces.
    #[tokio::test]
    async fn path_form_account_document_endpoint_is_a_hard_error() {
        use crate::auth::user_store::{
            PubkeyEntry, UserFilter, UserProfile, UserProfilePatch, UserStore,
        };
        use crate::config::OAuthConfig;
        use crate::config::server::CorsConfig;
        use crate::services::oauth::create_app;
        use crate::services::{DiscoveryClient, PolicyClient};
        use async_trait::async_trait;
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;
        use tower::ServiceExt; // oneshot

        /// User-store spy whose every method panics. The frozen account-DID
        /// handler must return 410 without ever consulting the store; if the
        /// old key-lookup/serving code path is reintroduced it touches
        /// `list_pubkeys` and this test aborts instead of passing.
        struct UntouchedUserStore;
        #[async_trait]
        impl UserStore for UntouchedUserStore {
            async fn get_profile(&self, _: &str) -> anyhow::Result<Option<UserProfile>> {
                unreachable!("frozen account-DID handler must not read profiles")
            }
            async fn register(&self, _: &str) -> anyhow::Result<String> {
                unreachable!()
            }
            async fn set_profile(&self, _: &str, _: UserProfilePatch) -> anyhow::Result<()> {
                unreachable!()
            }
            async fn remove(&self, _: &str) -> anyhow::Result<bool> {
                unreachable!()
            }
            async fn list_users(&self) -> Vec<String> {
                unreachable!()
            }
            async fn search(&self, _: &UserFilter) -> anyhow::Result<Vec<(String, UserProfile)>> {
                unreachable!()
            }
            async fn set_active(&self, _: &str, _: bool) -> anyhow::Result<()> {
                unreachable!()
            }
            async fn list_pubkeys(&self, _: &str) -> anyhow::Result<Vec<PubkeyEntry>> {
                // This is the method the pre-PR `user_did_document` called to
                // resolve keys before minting the path-form DID. Reaching it
                // means the freeze was bypassed.
                unreachable!("frozen account-DID handler must not call list_pubkeys (#1159)")
            }
            async fn add_pubkey(
                &self,
                _: &str,
                _: VerifyingKey,
                _: Option<String>,
            ) -> anyhow::Result<String> {
                unreachable!()
            }
            async fn add_pubkey_hybrid(
                &self,
                _: &str,
                _: VerifyingKey,
                _: Vec<u8>,
                _: Option<String>,
            ) -> anyhow::Result<String> {
                unreachable!()
            }
            async fn remove_pubkey(&self, _: &str, _: &str) -> anyhow::Result<bool> {
                unreachable!()
            }
            async fn get_pubkey_user(&self, _: &str) -> anyhow::Result<Option<String>> {
                unreachable!()
            }
            async fn touch_pubkey(&self, _: &str, _: &str) -> anyhow::Result<()> {
                unreachable!()
            }
        }

        // Minimal OAuthState over a LazyUdsTransport pointed at /dev/null.
        // The frozen handler returns before opening it, so this keeps the test
        // hermetic while still exercising the real router + handler wiring.
        let key = SigningKey::from_bytes(&[0x76; 32]);
        let vk = SigningKey::from_bytes(&[0x73; 32]).verifying_key();
        let dummy = std::path::PathBuf::from("/dev/null/did-doc-test.sock");
        let mk_client = || {
            let rpc = RpcClientImpl::new(
                LocalSigner::new(key.clone()),
                LazyUdsTransport::new(dummy.clone()),
                Some(vk),
            )
            .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical);
            Arc::new(rpc)
        };
        let state = Arc::new(
            OAuthState::new(
                &OAuthConfig::default(),
                PolicyClient::new(mk_client()),
                DiscoveryClient::new(mk_client()),
                [0x76; 32],
            )
            .with_user_service(Arc::new(
                crate::services::oauth::user_service::UserService::new(Arc::new(
                    UntouchedUserStore,
                )),
            )),
        );

        let cors = CorsConfig {
            enabled: false,
            ..Default::default()
        };
        let app = create_app(state, &cors);

        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/users/alice/did.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::GONE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert!(
            std::str::from_utf8(&body)
                .unwrap()
                .contains("path-form account minting is disabled"),
            "body was: {}",
            std::str::from_utf8(&body).unwrap(),
        );
    }

    #[test]
    fn ipv6_issuer_does_not_synthesize_atproto_handle() {
        assert_eq!(configured_handle_host("https://[::1]:6791"), None);
        assert_eq!(account_handle("alice", "https://[::1]:6791"), None);
    }

    #[test]
    fn invalid_username_label_does_not_synthesize_atproto_handle() {
        assert_eq!(account_handle("alice_bad", "https://pds.example.com"), None);
        assert_eq!(
            account_handle("Alice", "https://PDS.Example.COM").as_deref(),
            Some("alice.pds.example.com")
        );
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
        let did = "did:web:alice.hyprstream.example.com";
        let doc = build_did_document(
            did,
            "https://hyprstream.example.com",
            &[("key-1".to_owned(), sk.verifying_key())],
            None,
            &[],
            None,
            None,
        );
        assert_eq!(doc["id"].as_str().unwrap(), did);
        assert!(doc["@context"].is_array());
        assert_eq!(doc["verificationMethod"].as_array().unwrap().len(), 2); // multibase + jwk
        // The ed25519 VM is emitted as a Multikey (#280) with a multibase key.
        let vm = &doc["verificationMethod"][0];
        assert_eq!(vm["type"].as_str().unwrap(), "Multikey");
        assert!(vm["publicKeyMultibase"].as_str().unwrap().starts_with('z'));
        // The JWK fallback VM is retained.
        assert_eq!(
            doc["verificationMethod"][1]["type"].as_str().unwrap(),
            "JsonWebKey2020"
        );
        assert_eq!(
            doc["service"][0]["type"].as_str().unwrap(),
            "HyprstreamService"
        );
        assert_eq!(
            doc["service"][0]["serviceEndpoint"].as_str().unwrap(),
            "https://hyprstream.example.com",
        );
    }

    #[test]
    fn build_did_doc_empty_keys_is_valid() {
        let did = "did:web:alice.example.com";
        let doc = build_did_document(did, "https://example.com", &[], None, &[], None, None);
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
        let atproto = AtprotoIdentity {
            p256_vk,
            handle: "hyprstream.example.com",
            drain: None,
            lead: None,
        };
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
            None,
            None,
        );

        // #atproto Multikey is FIRST and p256.
        let vms = doc["verificationMethod"].as_array().unwrap();
        assert_eq!(vms[0]["id"].as_str().unwrap(), format!("{did}#atproto"));
        assert_eq!(vms[0]["type"].as_str().unwrap(), "Multikey");
        assert!(
            vms[0]["publicKeyMultibase"]
                .as_str()
                .unwrap()
                .starts_with('z')
        );
        // Ed25519 mesh VMs still present after it.
        assert_eq!(vms.len(), 1 + 2);
        // The ed25519 mesh VM is also a Multikey (#280), not the deprecated
        // Ed25519VerificationKey2020 type, with a multibase key; JWK fallback kept.
        assert_eq!(vms[1]["type"].as_str().unwrap(), "Multikey");
        assert!(
            vms[1]["publicKeyMultibase"]
                .as_str()
                .unwrap()
                .starts_with('z')
        );
        assert_eq!(vms[2]["type"].as_str().unwrap(), "JsonWebKey2020");

        // #atproto_pds first, origin-only (no path), correct type.
        let svcs = doc["service"].as_array().unwrap();
        assert_eq!(
            svcs[0]["id"].as_str().unwrap(),
            format!("{did}#atproto_pds")
        );
        assert_eq!(
            svcs[0]["type"].as_str().unwrap(),
            "AtprotoPersonalDataServer"
        );
        assert_eq!(
            svcs[0]["serviceEndpoint"].as_str().unwrap(),
            "https://hyprstream.example.com"
        );
        // transport entry present as a typed map service.
        let iroh = svcs.iter().find(|s| s["type"] == "IrohTransport").unwrap();
        assert_eq!(iroh["id"].as_str().unwrap(), format!("{did}#iroh"));
        assert!(iroh["serviceEndpoint"]["accept"].is_array());

        // alsoKnownAs handle.
        assert_eq!(
            doc["alsoKnownAs"][0].as_str().unwrap(),
            "at://hyprstream.example.com"
        );
    }

    /// #1113 rev2 finding 2/7: the per-user atproto DID document (served at
    /// `/users/:u/did.json` for a `did:web:{authority}:users:{u}` token `sub`)
    /// carries an `AtprotoPersonalDataServer` service whose `serviceEndpoint`
    /// is THIS AS's origin — the round-trip the stock atproto client performs
    /// (resolve `sub` → PDS service → PDS metadata issuer == AS). Mirrors the
    /// `user_did_document` handler's construction exactly.
    #[test]
    fn per_user_atproto_doc_pds_service_points_at_issuer() {
        let p256_sk = p256::ecdsa::SigningKey::random(&mut OsRng);
        let p256_vk = p256_sk.verifying_key();
        let issuer = "https://pds.example.com";
        let did = "did:web:pds.example.com:users:alice";
        let handle = "alice.pds.example.com";
        let atproto = AtprotoIdentity {
            p256_vk,
            handle,
            drain: None,
            lead: None,
        };
        let doc = build_did_document(did, issuer, &[], Some(&atproto), &[], None, None);

        // The atproto PDS service is present and points at the issuer origin.
        let pds = doc["service"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["type"] == "AtprotoPersonalDataServer")
            .expect("AtprotoPersonalDataServer service required for the sub→PDS round-trip");
        assert_eq!(
            pds["serviceEndpoint"].as_str().unwrap(),
            issuer,
            "PDS serviceEndpoint MUST equal the AS issuer (PDS = its own AS)"
        );
        // The account handle alias is present.
        assert_eq!(
            doc["alsoKnownAs"][0].as_str().unwrap(),
            "at://alice.pds.example.com"
        );
        // The hosted-account #atproto VM (P-256 Multikey) is present.
        assert!(
            doc["verificationMethod"]
                .as_array()
                .unwrap()
                .iter()
                .any(|vm| vm["id"].as_str() == Some(&format!("{did}#atproto")))
        );
    }

    /// #918 producer side: active remains the exact `#atproto` VM, while drain
    /// and lead are bounded distinct-fragment overlap VMs.
    #[test]
    fn build_did_doc_publishes_bounded_overlap_slots() {
        let active_sk = p256::ecdsa::SigningKey::random(&mut OsRng);
        let active_vk = active_sk.verifying_key();
        let drain_sk = p256::ecdsa::SigningKey::random(&mut OsRng);
        let drain_vk = drain_sk.verifying_key();
        let lead_sk = p256::ecdsa::SigningKey::random(&mut OsRng);
        let lead_vk = lead_sk.verifying_key();
        let did = "did:web:hyprstream.example.com";
        let atproto = AtprotoIdentity {
            p256_vk: active_vk,
            handle: "hyprstream.example.com",
            drain: Some(AtprotoOverlapKey {
                vk: drain_vk,
                nbf: 1_000,
                exp: 2_000,
            }),
            lead: Some(AtprotoOverlapKey {
                vk: lead_vk,
                nbf: 1_500,
                exp: 3_000,
            }),
        };
        let doc = build_did_document(
            did,
            "https://hyprstream.example.com",
            &[],
            Some(&atproto),
            &[],
            None,
            None,
        );
        let vms = doc["verificationMethod"].as_array().unwrap();
        assert_eq!(vms[0]["id"].as_str().unwrap(), format!("{did}#atproto"));
        // Exactly one active #atproto: stock resolvers keep their existing
        // first-exact-fragment behavior.
        assert_eq!(
            vms.iter()
                .filter(|v| v["id"]
                    .as_str()
                    .map(|s| s.ends_with("#atproto"))
                    .unwrap_or(false))
                .count(),
            1,
            "exactly one active #atproto method"
        );
        let drain = vms
            .iter()
            .find(|v| v["id"] == format!("{did}#atproto_drain"))
            .unwrap();
        assert_eq!(drain["nbf"], 1_000);
        assert_eq!(drain["exp"], 2_000);
        let lead = vms
            .iter()
            .find(|v| v["id"] == format!("{did}#atproto_lead"))
            .unwrap();
        assert_eq!(lead["nbf"], 1_500);
        assert_eq!(lead["exp"], 3_000);

        let published = hyprstream_pds::commit::PublishedAtprotoKeys::from_did_document(&doc, did)
            .expect("published overlap document parses");
        assert_eq!(published.len(), 3);
        assert_eq!(published.live_keys(1_250).count(), 2, "active + drain");
        assert_eq!(published.live_keys(2_500).count(), 2, "active + lead");
    }

    #[test]
    fn root_identity_publisher_and_consumer_enforce_bounded_overlap() {
        use hyprstream_rpc::admission::AdmittedIdentity;
        use hyprstream_rpc::auth::AtprotoPerimeterGateway;
        use hyprstream_rpc::auth::mac::Assurance;
        use hyprstream_rpc::identity_resolver::{DidDocumentProvider, MethodDispatchResolver};

        #[derive(Clone)]
        struct Docs(Value);
        impl DidDocumentProvider for Docs {
            fn document(&self, _did: &str) -> anyhow::Result<Value> {
                Ok(self.0.clone())
            }
        }

        let old = SigningKey::generate(&mut OsRng);
        let new = SigningKey::generate(&mut OsRng);
        let lead = SigningKey::generate(&mut OsRng);
        let old_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&old);
        let old_pq = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&old_pq);
        let slots = crate::auth::key_rotation::KeySlots {
            drain: Some(crate::auth::key_rotation::KeySlot::new(
                old.clone(),
                100,
                200,
            )),
            active: Some(crate::auth::key_rotation::KeySlot::new(
                new.clone(),
                150,
                400,
            )),
            lead: Some(crate::auth::key_rotation::KeySlot::new(
                lead.clone(),
                350,
                550,
            )),
        };
        let methods = root_identity_methods(&old, &old_pq, Some(&slots), 175);
        let did = "did:web:peer.example";
        let mut doc = build_did_document(did, "https://peer.example", &[], None, &[], None, None);
        append_root_identity_methods(&mut doc, did, &methods);

        let ids: Vec<_> = doc["verificationMethod"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|vm| vm["id"].as_str())
            .collect();
        assert!(ids.contains(&format!("{did}#key-1").as_str()));
        assert!(ids.contains(&format!("{did}#key-1-pq").as_str()));
        assert!(ids.contains(&format!("{did}#mesh-pq").as_str()));
        assert!(
            ids.iter().any(|id| id.contains("#mesh-")),
            "named rotation slots must be published"
        );
        // Mixed-version producer direction: the pre-upgrade resolver chose the
        // first compatible Ed25519 and first compatible ML-DSA method. During
        // the compatibility window those are still the old exact pair.
        let vms = doc["verificationMethod"].as_array().unwrap();
        let first_ed = vms
            .iter()
            .filter_map(|vm| vm["publicKeyMultibase"].as_str())
            .find_map(|multibase| {
                hyprstream_rpc::did_web::decode_ed25519_multikey(multibase).ok()
            })
            .unwrap();
        let first_pq = vms
            .iter()
            .filter_map(|vm| vm["publicKeyMultibase"].as_str())
            .find_map(|multibase| {
                hyprstream_rpc::did_web::decode_multikey(
                    multibase,
                    &hyprstream_rpc::did_web::MULTICODEC_ML_DSA_65_PUB,
                )
                .ok()
            })
            .unwrap();
        assert_eq!(first_ed, old.verifying_key().to_bytes());
        assert_eq!(first_pq, old_pq);

        let now = Arc::new(std::sync::atomic::AtomicI64::new(175));
        let clock = Arc::clone(&now);
        let resolver = MethodDispatchResolver::new(Docs(doc))
            .with_clock(move || clock.load(std::sync::atomic::Ordering::SeqCst));
        let gateway = AtprotoPerimeterGateway::new(resolver);
        let admitted = |key: &SigningKey| AdmittedIdentity {
            origin: "https://peer.example".to_owned(),
            did: Some(did.to_owned()),
            key: key.verifying_key().to_bytes(),
        };

        let old_peer = gateway
            .enroll(&admitted(&old))
            .expect("drain signer accepted during overlap");
        let new_peer = gateway
            .enroll(&admitted(&new))
            .expect("active signer accepted during overlap");
        assert_eq!(old_peer.assurance, Assurance::PqHybrid);
        assert_eq!(new_peer.assurance, Assurance::PqHybrid);
        assert!(
            gateway.enroll(&admitted(&lead)).is_err(),
            "lead is published before use but not accepted before nbf"
        );

        now.store(200, std::sync::atomic::Ordering::SeqCst);
        assert!(
            gateway.enroll(&admitted(&old)).is_err(),
            "retired signer must fail exactly when its bound closes"
        );
        assert!(gateway.enroll(&admitted(&new)).is_ok());
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
        assert!(
            mb.starts_with('z'),
            "Multikey publicKeyMultibase must be base58btc multibase"
        );
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
    fn build_did_doc_publishes_mesh_pq_vm() {
        // #157: when a mesh ML-DSA-65 vk is provided, build_did_document emits a
        // `#mesh-pq` Multikey VM with the ml-dsa-65-pub multicodec, listed after
        // the Ed25519 VMs, and referenced in authentication/assertionMethod.
        let sk = SigningKey::generate(&mut OsRng);
        let did = "did:web:hyprstream.example.com";
        let vk_bytes = vec![0x42u8; 1952];
        let doc = build_did_document(
            did,
            "https://hyprstream.example.com",
            &[("key-1".to_owned(), sk.verifying_key())],
            None,
            &[],
            Some(&vk_bytes),
            None,
        );
        let vms = doc["verificationMethod"].as_array().unwrap();
        // key-1 multibase + key-1 jwk + mesh-pq = 3.
        assert_eq!(vms.len(), 3);
        let pq = vms
            .iter()
            .find(|v| v["id"] == format!("{did}#mesh-pq"))
            .unwrap();
        assert_eq!(pq["type"].as_str().unwrap(), "Multikey");
        let mb = pq["publicKeyMultibase"].as_str().unwrap();
        let decoded = bs58::decode(&mb[1..]).into_vec().unwrap();
        assert_eq!(&decoded[..2], &[0x91, 0x24]); // ml-dsa-65-pub multicodec
        assert_eq!(&decoded[2..], vk_bytes.as_slice());
        // Referenced as both an authentication and assertion method.
        let mesh_pq_id = format!("{did}#mesh-pq");
        assert!(
            doc["authentication"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| *v == mesh_pq_id)
        );
        assert!(
            doc["assertionMethod"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| *v == mesh_pq_id)
        );
    }

    #[test]
    fn mesh_pq_vm_matches_derived_signing_key() {
        // #157 scope #1/#2 consistency: the `#mesh-pq` VM published from the
        // OAuth signing key must equal the ML-DSA-65 verifying key the mesh
        // actually signs with (derive_mesh_mldsa_key over the same Ed25519 key).
        let sk = SigningKey::generate(&mut OsRng);
        let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&sk);
        let pq_vk_bytes = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk);

        let did = "did:web:hyprstream.example.com";
        let doc = build_did_document(
            did,
            "https://hyprstream.example.com",
            &[("key-1".to_owned(), sk.verifying_key())],
            None,
            &[],
            Some(&pq_vk_bytes),
            None,
        );
        let vms = doc["verificationMethod"].as_array().unwrap();
        let pq = vms
            .iter()
            .find(|v| v["id"] == format!("{did}#mesh-pq"))
            .unwrap();
        let mb = pq["publicKeyMultibase"].as_str().unwrap();
        let decoded = bs58::decode(&mb[1..]).into_vec().unwrap();
        assert_eq!(
            &decoded[2..],
            pq_vk_bytes.as_slice(),
            "published #mesh-pq key must equal the derived mesh signing key's public key"
        );
    }

    #[test]
    fn root_doc_advertises_iroh_only_as_transport_when_bound() {
        // #282: when the iroh substrate is bound, root_did_document adds the
        // IrohTransport service entry accepting both ALPNs, but no VM.
        let oauth_sk = SigningKey::generate(&mut OsRng);
        let node_sk = SigningKey::generate(&mut OsRng);
        let node_id = node_sk.verifying_key().to_bytes();
        let did = "did:web:hyprstream.example.com";

        let keys = vec![("key-1".to_owned(), oauth_sk.verifying_key())];
        let transports = [TransportEndpoint {
            fragment: "iroh".to_owned(),
            vm_type: "IrohTransport".to_owned(),
            endpoint: hyprstream_rpc::service_entry::encode_iroh(
                &node_id,
                &[],
                &["hyprstream-rpc/1", "moql"],
            ),
        }];
        let doc = build_did_document(
            did,
            "https://hyprstream.example.com",
            &keys,
            None,
            &transports,
            None,
            None,
        );

        // Equality to a valid Ed25519 key does not create identity authority.
        let vms = doc["verificationMethod"].as_array().unwrap();
        assert!(vms.iter().all(|v| v["id"] != format!("{did}#iroh")));

        // The IrohTransport service entry accepts BOTH ALPNs.
        let svcs = doc["service"].as_array().unwrap();
        let iroh_svc = svcs
            .iter()
            .find(|s| s["type"] == "IrohTransport")
            .expect("IrohTransport service entry must be advertised");
        assert_eq!(iroh_svc["id"].as_str().unwrap(), format!("{did}#iroh"));
        let accept = iroh_svc["serviceEndpoint"]["accept"].as_array().unwrap();
        let accept: Vec<&str> = accept.iter().filter_map(|v| v.as_str()).collect();
        assert!(accept.contains(&"hyprstream-rpc/1"));
        assert!(accept.contains(&"moql"));
    }

    #[test]
    fn root_doc_omits_iroh_when_not_bound() {
        // No iroh transport configured → no IrohTransport entry.
        let sk = SigningKey::generate(&mut OsRng);
        let did = "did:web:hyprstream.example.com";
        let doc = build_did_document(
            did,
            "https://hyprstream.example.com",
            &[("key-1".to_owned(), sk.verifying_key())],
            None,
            &[],
            None,
            None,
        );
        let vms = doc["verificationMethod"].as_array().unwrap();
        assert!(
            !vms.iter().any(|v| v["id"] == format!("{did}#iroh")),
            "carrier NodeId is never a DID verification method"
        );
        let svcs = doc["service"].as_array().unwrap();
        assert!(
            !svcs.iter().any(|s| s["type"] == "IrohTransport"),
            "no IrohTransport entry when iroh is not bound"
        );
    }

    #[test]
    fn build_did_doc_omits_key_agreement_when_mesh_kem_absent() {
        // No `#mesh-kem` public material provided → `keyAgreement` is present
        // but empty (a valid, empty DID-document relationship), never omitted
        // outright (omitting the key entirely would be a different, also-valid
        // shape, but an empty array is simpler for consumers to handle uniformly).
        let sk = SigningKey::generate(&mut OsRng);
        let did = "did:web:hyprstream.example.com";
        let doc = build_did_document(
            did,
            "https://hyprstream.example.com",
            &[("key-1".to_owned(), sk.verifying_key())],
            None,
            &[],
            None,
            None,
        );
        assert_eq!(doc["keyAgreement"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn build_did_doc_publishes_mesh_kem_key_agreement() {
        // S1 / #552: when a `#mesh-kem` recipient public is provided,
        // build_did_document emits both suite-component `keyAgreement` VMs
        // (X25519 + ML-KEM-768), embedded directly (not cross-referenced from
        // verificationMethod/authentication/assertionMethod — these are
        // key-agreement-only keys).
        let node_sk = SigningKey::generate(&mut OsRng);
        let kem_kp = hyprstream_rpc::node_identity::derive_mesh_kem_recipient(&node_sk)
            .expect("derive #mesh-kem recipient");
        let kem_pub = kem_kp.public();

        let did = "did:web:hyprstream.example.com";
        let doc = build_did_document(
            did,
            "https://hyprstream.example.com",
            &[("key-1".to_owned(), node_sk.verifying_key())],
            None,
            &[],
            None,
            Some(&kem_pub),
        );

        // keyAgreement carries exactly the two suite-component VMs, and they
        // are NOT duplicated into verificationMethod/authentication/assertionMethod.
        let kas = doc["keyAgreement"].as_array().unwrap();
        assert_eq!(kas.len(), 2);
        assert_eq!(
            kas[0]["id"].as_str().unwrap(),
            format!("{did}#mesh-kem-x25519")
        );
        assert_eq!(kas[0]["type"].as_str().unwrap(), "Multikey");
        assert_eq!(
            kas[1]["id"].as_str().unwrap(),
            format!("{did}#mesh-kem-mlkem768")
        );
        assert_eq!(kas[1]["type"].as_str().unwrap(), "Multikey");

        let vms = doc["verificationMethod"].as_array().unwrap();
        assert!(
            !vms.iter()
                .any(|v| v["id"] == format!("{did}#mesh-kem-x25519"))
        );
        assert!(
            !vms.iter()
                .any(|v| v["id"] == format!("{did}#mesh-kem-mlkem768"))
        );
        assert!(
            !doc["authentication"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_str().map(|s| s.contains("mesh-kem")).unwrap_or(false))
        );

        // The published X25519 leg round-trips to the exact bytes derived —
        // consistency between the DID doc and the key the node actually holds.
        let mb = kas[0]["publicKeyMultibase"].as_str().unwrap();
        let decoded = bs58::decode(&mb[1..]).into_vec().unwrap();
        assert_eq!(&decoded[..2], &MULTICODEC_X25519_PUB);
        assert_eq!(&decoded[2..], kem_pub.eks[0].as_slice());

        let mb2 = kas[1]["publicKeyMultibase"].as_str().unwrap();
        let decoded2 = bs58::decode(&mb2[1..]).into_vec().unwrap();
        assert_eq!(&decoded2[..2], &MULTICODEC_ML_KEM_768_PUB);
        assert_eq!(&decoded2[2..], kem_pub.eks[1].as_slice());
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
