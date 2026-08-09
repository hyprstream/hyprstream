//! JWT token implementation.
//!
//! Re-exports EdDSA signing from hyprstream-rpc and adds ES256 (P-256) signing.

pub use hyprstream_rpc::auth::{
    decode, decode_with_any_key, decode_with_any_key_lenient,
    decode_with_federation_candidates, decode_with_key, encode, encode_service_jwt, Claims,
    JwtError,
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use p256::ecdsa::{SigningKey as Es256SigningKey, signature::Signer};

/// Encode and sign an OAuth 2.0 access token with ES256 (P-256 ECDSA).
///
/// Produces a standard `at+jwt` with `alg: "ES256"` in the JOSE header.
/// The `kid` is the RFC 7638 JWK Thumbprint of the P-256 public key.
/// Automatically assigns a `jti` if not already set.
pub fn encode_es256(claims: &Claims, signing_key: &Es256SigningKey) -> String {
    let claims = if claims.jti.is_some() {
        std::borrow::Cow::Borrowed(claims)
    } else {
        std::borrow::Cow::Owned(claims.clone().with_jti())
    };
    let kid = es256_kid(signing_key);
    let header = format!(r#"{{"alg":"ES256","typ":"at+jwt","kid":"{}"}}"#, kid);
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());
    let payload_json = serde_json::to_string(claims.as_ref()).unwrap_or_else(|_e| {
        tracing::error!("JWT claims serialization failed: {}", _e);
        "{}".to_owned()
    });
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());
    let signing_input = format!("{header_b64}.{payload_b64}");
    let signature: p256::ecdsa::Signature = signing_key.sign(signing_input.as_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());
    format!("{signing_input}.{sig_b64}")
}

fn es256_coordinates(signing_key: &Es256SigningKey) -> ([u8; 32], [u8; 32]) {
    let vk = signing_key.verifying_key();
    let point = vk.to_encoded_point(false);
    // Uncompressed P-256 point always has x and y (32 bytes each).
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(point.x().map(AsRef::as_ref).unwrap_or(&[0u8; 32]));
    y.copy_from_slice(point.y().map(AsRef::as_ref).unwrap_or(&[0u8; 32]));
    (x, y)
}

/// Compute the RFC 7638 JWK Thumbprint for a P-256 signing key.
pub fn es256_kid(signing_key: &Es256SigningKey) -> String {
    let (x, y) = es256_coordinates(signing_key);
    hyprstream_rpc::auth::jwk_thumbprint(&hyprstream_rpc::auth::JwkThumbprintInput::Es256 { x: &x, y: &y })
}

/// Build a JWK (serde_json::Value) for JWKS publishing from a P-256 signing key.
pub fn es256_jwk(signing_key: &Es256SigningKey) -> serde_json::Value {
    let (x, y) = es256_coordinates(signing_key);
    let kid = es256_kid(signing_key);
    serde_json::json!({
        "kty": "EC",
        "crv": "P-256",
        "use": "sig",
        "alg": "ES256",
        "kid": kid,
        "x": URL_SAFE_NO_PAD.encode(x),
        "y": URL_SAFE_NO_PAD.encode(y),
    })
}

/// Generate a new random P-256 signing key.
pub fn generate_es256_key() -> Es256SigningKey {
    Es256SigningKey::random(&mut rand::rngs::OsRng)
}

// ── ML-DSA-65 JWT encoding (draft-ietf-cose-dilithium-11) ──────────────

/// Encode and sign a JWT with ML-DSA-65 (`alg: "ML-DSA-65"`, `kty: "AKP"`).
pub fn encode_ml_dsa_65(
    claims: &Claims,
    signing_key: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
) -> String {
    let claims = if claims.jti.is_some() {
        std::borrow::Cow::Borrowed(claims)
    } else {
        std::borrow::Cow::Owned(claims.clone().with_jti())
    };
    let vk = ml_dsa::Keypair::verifying_key(signing_key);
    let vk_bytes = hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&vk);
    let kid = hyprstream_rpc::auth::jwk_thumbprint(
        &hyprstream_rpc::auth::JwkThumbprintInput::Akp {
            alg: "ML-DSA-65",
            pub_bytes: &vk_bytes,
        },
    );
    let header = format!(r#"{{"alg":"ML-DSA-65","typ":"at+jwt","kid":"{}"}}"#, kid);
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());
    let payload_json = serde_json::to_string(claims.as_ref()).unwrap_or_else(|_e| {
        tracing::error!("JWT claims serialization failed: {}", _e);
        "{}".to_owned()
    });
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());
    let signing_input = format!("{header_b64}.{payload_b64}");
    let sig = hyprstream_rpc::crypto::pq::ml_dsa_sign(signing_key, signing_input.as_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(&sig);
    format!("{signing_input}.{sig_b64}")
}

/// Encode and sign a composite ML-DSA-65-Ed25519 JWT.
///
/// Signature = ML-DSA-65 sig (3309 bytes) ∥ Ed25519 sig (64 bytes).
/// Per draft-ietf-jose-pq-composite-sigs.
pub fn encode_composite_ml_dsa_65_ed25519(
    claims: &Claims,
    ml_dsa_key: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
    ed25519_key: &ed25519_dalek::SigningKey,
) -> String {
    use ed25519_dalek::Signer;

    let claims = if claims.jti.is_some() {
        std::borrow::Cow::Borrowed(claims)
    } else {
        std::borrow::Cow::Owned(claims.clone().with_jti())
    };
    let vk = ml_dsa::Keypair::verifying_key(ml_dsa_key);
    let kid = composite_kid(&vk, &ed25519_key.verifying_key());
    let header = format!(r#"{{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt","kid":"{}"}}"#, kid);
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());
    let payload_json = serde_json::to_string(claims.as_ref()).unwrap_or_else(|_e| {
        tracing::error!("JWT claims serialization failed: {}", _e);
        "{}".to_owned()
    });
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());
    let signing_input = format!("{header_b64}.{payload_b64}");
    let message = signing_input.as_bytes();

    let ml_dsa_sig = hyprstream_rpc::crypto::pq::ml_dsa_sign(ml_dsa_key, message);
    let ed25519_sig = ed25519_key.sign(message);

    let mut composite_sig = Vec::with_capacity(ml_dsa_sig.len() + 64);
    composite_sig.extend_from_slice(&ml_dsa_sig);
    composite_sig.extend_from_slice(&ed25519_sig.to_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(&composite_sig);
    format!("{signing_input}.{sig_b64}")
}

/// Encode a service WIT (`typ: "wit+jwt"`) with ML-DSA-65-Ed25519 composite signature.
pub fn encode_composite_service_jwt(
    claims: &Claims,
    ml_dsa_key: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
    ed25519_key: &ed25519_dalek::SigningKey,
) -> String {
    use ed25519_dalek::Signer;

    let claims = if claims.jti.is_some() {
        std::borrow::Cow::Borrowed(claims)
    } else {
        std::borrow::Cow::Owned(claims.clone().with_jti())
    };
    let vk = ml_dsa::Keypair::verifying_key(ml_dsa_key);
    let kid = composite_kid(&vk, &ed25519_key.verifying_key());
    let header = format!(r#"{{"alg":"ML-DSA-65-Ed25519","typ":"wit+jwt","kid":"{}"}}"#, kid);
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());
    let payload_json = serde_json::to_string(claims.as_ref()).unwrap_or_else(|_e| {
        tracing::error!("JWT claims serialization failed: {}", _e);
        "{}".to_owned()
    });
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());
    let signing_input = format!("{header_b64}.{payload_b64}");
    let message = signing_input.as_bytes();

    let ml_dsa_sig = hyprstream_rpc::crypto::pq::ml_dsa_sign(ml_dsa_key, message);
    let ed25519_sig = ed25519_key.sign(message);

    let mut composite_sig = Vec::with_capacity(ml_dsa_sig.len() + 64);
    composite_sig.extend_from_slice(&ml_dsa_sig);
    composite_sig.extend_from_slice(&ed25519_sig.to_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(&composite_sig);
    format!("{signing_input}.{sig_b64}")
}

/// Encode a service WIT with the mandatory hybrid suite for the dispatch plane.
///
/// When a composite signing authority is configured, mints with its active
/// OAuth pair through the authority barrier (rotation-aware, mirroring
/// PolicyService's token signing seam) — a mint-snapshot failure or a missing
/// active pair is an error, never a silent downgrade to another key: the
/// callers' `fallback_ed` is the rotating `active_jwt_signing_key()` slot, so
/// a derived pair minted behind a configured authority would carry a
/// composite kid nothing resolves while bypassing the staleness barrier.
///
/// Only when NO authority has been initialized (fresh install, before
/// rotation bootstrap) does it fall back to the self-contained CA JWT pair
/// `(derive_mesh_mldsa_key(fallback_ed), fallback_ed)`, whose composite kid
/// the dispatch key sources resolve offline from the `ca-mldsa-pubkey`
/// credential. A classical WIT is never minted: the dispatch plane's Hybrid
/// policy would reject it unconditionally.
pub fn encode_service_jwt_hybrid_via_authority(
    claims: &Claims,
    fallback_ed: &ed25519_dalek::SigningKey,
) -> anyhow::Result<String> {
    encode_service_jwt_hybrid_with_key_set(
        &hyprstream_rpc::auth::global_composite_key_set(),
        claims,
        fallback_ed,
    )
}

/// Key-set-parameterized body of [`encode_service_jwt_hybrid_via_authority`].
fn encode_service_jwt_hybrid_with_key_set(
    key_set: &hyprstream_rpc::auth::CompositeKeySet,
    claims: &Claims,
    fallback_ed: &ed25519_dalek::SigningKey,
) -> anyhow::Result<String> {
    if key_set.authority_configured() {
        let snapshot = key_set
            .mint_snapshot()
            .map_err(|error| anyhow::anyhow!("composite authority unavailable: {error}"))?;
        let (ml_key, ed_key) = snapshot
            .active_signing_pair(hyprstream_rpc::auth::CompositePairRole::OAuth)
            .and_then(hyprstream_rpc::auth::CompositeKeyPair::signing_keys)
            .ok_or_else(|| {
                anyhow::anyhow!("no active OAuth composite signing pair; refusing to mint")
            })?;
        return Ok(encode_composite_service_jwt(claims, &ml_key, &ed_key));
    }
    let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(fallback_ed);
    let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&pq);
    Ok(hyprstream_rpc::auth::jwt::encode_service_jwt_hybrid(
        claims,
        fallback_ed,
        &pq,
        &pq_vk,
    ))
}

/// Build a JWK for an ML-DSA-65 key (`kty: "AKP"`).
pub fn ml_dsa_65_jwk(
    vk: &hyprstream_rpc::crypto::pq::MlDsaVerifyingKey,
) -> serde_json::Value {
    let vk_bytes = hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(vk);
    let kid = hyprstream_rpc::auth::jwk_thumbprint(
        &hyprstream_rpc::auth::JwkThumbprintInput::Akp {
            alg: "ML-DSA-65",
            pub_bytes: &vk_bytes,
        },
    );
    serde_json::json!({
        "kty": "AKP",
        "alg": "ML-DSA-65",
        "use": "sig",
        "kid": kid,
        "pub": URL_SAFE_NO_PAD.encode(&vk_bytes),
    })
}

/// Compute the RFC 7638 JWK thumbprint for one exact composite key pair.
pub fn composite_kid(
    ml_dsa_vk: &hyprstream_rpc::crypto::pq::MlDsaVerifyingKey,
    ed25519_vk: &ed25519_dalek::VerifyingKey,
) -> String {
    hyprstream_rpc::auth::composite_kid(ml_dsa_vk, ed25519_vk)
}

/// Build a JWK for a composite ML-DSA-65-Ed25519 key (`kty: "AKP"`).
pub fn composite_jwk(
    ml_dsa_vk: &hyprstream_rpc::crypto::pq::MlDsaVerifyingKey,
    ed25519_vk: &ed25519_dalek::VerifyingKey,
) -> serde_json::Value {
    let kid = composite_kid(ml_dsa_vk, ed25519_vk);
    let ml_dsa_vk_bytes = hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(ml_dsa_vk);
    let ed25519_vk_bytes = ed25519_vk.to_bytes();
    let mut composite_pub = Vec::with_capacity(ml_dsa_vk_bytes.len() + 32);
    composite_pub.extend_from_slice(&ml_dsa_vk_bytes);
    composite_pub.extend_from_slice(&ed25519_vk_bytes);
    serde_json::json!({
        "kty": "AKP",
        "alg": "ML-DSA-65-Ed25519",
        "use": "sig",
        "kid": kid,
        "pub": URL_SAFE_NO_PAD.encode(&composite_pub),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// With NO composite authority initialized (fresh install), the hybrid
    /// WIT mint falls back to the self-contained pair derived from the passed
    /// Ed25519 key, and the token verifies against exactly that pair.
    #[test]
    fn hybrid_wit_falls_back_only_without_authority() {
        let key_set = hyprstream_rpc::auth::CompositeKeySet::default();
        let ed = ed25519_dalek::SigningKey::from_bytes(&[0x41; 32]);
        let now = chrono::Utc::now().timestamp();
        let claims = Claims::new("alice".to_owned(), now, now + 3600);

        let wit = encode_service_jwt_hybrid_with_key_set(&key_set, &claims, &ed).unwrap();

        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ed);
        let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&pq);
        let dispatch =
            hyprstream_rpc::auth::jwt::parse_composite_dispatch(&wit, &["wit+jwt"]).unwrap();
        assert_eq!(dispatch.kid(), composite_kid(&pq_vk, &ed.verifying_key()));
        let verified = hyprstream_rpc::auth::jwt::decode_composite(
            &wit,
            &pq_vk,
            &ed.verifying_key(),
            None,
            &dispatch,
        )
        .unwrap();
        assert_eq!(verified.sub, "alice");
    }

    /// With a CONFIGURED authority whose mint snapshot fails (stale / pending
    /// cutover / unreadable), the hybrid WIT mint propagates the error — it
    /// must never silently downgrade to a derived pair, whose kid nothing
    /// resolves and which would bypass the authority mint barrier.
    #[test]
    fn hybrid_wit_mint_failure_propagates_instead_of_falling_back() {
        let dir = tempfile::tempdir().unwrap();
        let key_set = hyprstream_rpc::auth::CompositeKeySet::default();
        // Configured, but no committed generation exists on disk, so
        // mint_snapshot fails closed.
        key_set.configure_authority(
            dir.path().join("ledger.json"),
            dir.path().join("committed"),
            dir.path().join("committed-ledger"),
            dir.path().join("ledger.lock"),
        );
        let ed = ed25519_dalek::SigningKey::from_bytes(&[0x42; 32]);
        let now = chrono::Utc::now().timestamp();
        let claims = Claims::new("alice".to_owned(), now, now + 3600);

        let result = encode_service_jwt_hybrid_with_key_set(&key_set, &claims, &ed);
        assert!(
            result.is_err(),
            "mint-snapshot failure must propagate, not fall back to a derived pair"
        );
    }

    /// With a healthy authority that has no ACTIVE OAuth signing pair, the
    /// hybrid WIT mint refuses rather than downgrading to a derived pair.
    #[test]
    fn hybrid_wit_refuses_without_active_oauth_pair() {
        use hyprstream_rpc::auth::{CompositeKeyPair, CompositePairRole, CompositePairState};
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let key_set = hyprstream_rpc::auth::CompositeKeySet::default();
        let generation = br#"{"version":1,"component_digest":"g1"}"#;
        std::fs::write(dir.path().join("ledger.json"), generation).unwrap();
        std::fs::write(dir.path().join("committed"), generation).unwrap();
        std::fs::write(dir.path().join("committed-ledger-1-g1.json"), generation).unwrap();
        key_set.configure_authority(
            dir.path().join("ledger.json"),
            dir.path().join("committed"),
            dir.path().join("committed-ledger"),
            dir.path().join("ledger.lock"),
        );
        // Publish only a Policy-role signing pair — no active OAuth pair.
        let (pq, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let pair_ed = Arc::new(ed25519_dalek::SigningKey::from_bytes(&[0x43; 32]));
        let kid = composite_kid(&pq_vk, &pair_ed.verifying_key());
        key_set
            .publish(
                1,
                "g1".to_owned(),
                vec![CompositeKeyPair::signing(
                    kid,
                    Arc::new(pq),
                    pair_ed,
                    CompositePairRole::Policy,
                    CompositePairState::Active,
                    0,
                    i64::MAX,
                )],
            )
            .unwrap();

        let ed = ed25519_dalek::SigningKey::from_bytes(&[0x44; 32]);
        let now = chrono::Utc::now().timestamp();
        let claims = Claims::new("alice".to_owned(), now, now + 3600);
        assert!(encode_service_jwt_hybrid_with_key_set(&key_set, &claims, &ed).is_err());
    }

    #[test]
    fn es256_roundtrip() {
        let key = generate_es256_key();
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        let token = encode_es256(&claims, &key);

        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);

        let header_bytes = URL_SAFE_NO_PAD.decode(parts[0]).unwrap();
        let header: serde_json::Value = serde_json::from_slice(&header_bytes).unwrap();
        assert_eq!(header["alg"], "ES256");
        assert_eq!(header["typ"], "at+jwt");
        assert!(header["kid"].as_str().unwrap().len() == 43);

        let payload_bytes = URL_SAFE_NO_PAD.decode(parts[1]).unwrap();
        let decoded: Claims = serde_json::from_slice(&payload_bytes).unwrap();
        assert_eq!(decoded.sub, "alice");
        assert!(decoded.jti.is_some());

        // Verify signature
        use p256::ecdsa::{Signature, signature::Verifier};
        let sig_bytes = URL_SAFE_NO_PAD.decode(parts[2]).unwrap();
        let signature = Signature::from_slice(&sig_bytes).unwrap();
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        key.verifying_key().verify(signing_input.as_bytes(), &signature).unwrap();
    }

    #[test]
    fn es256_kid_deterministic() {
        let key = generate_es256_key();
        assert_eq!(es256_kid(&key), es256_kid(&key));
        assert_eq!(es256_kid(&key).len(), 43);
    }

    #[test]
    fn es256_jwk_structure() {
        let key = generate_es256_key();
        let jwk = es256_jwk(&key);
        assert_eq!(jwk["kty"], "EC");
        assert_eq!(jwk["crv"], "P-256");
        assert_eq!(jwk["alg"], "ES256");
        assert_eq!(jwk["use"], "sig");
        assert!(jwk["kid"].as_str().unwrap().len() == 43);
        assert!(jwk["x"].as_str().is_some());
        assert!(jwk["y"].as_str().is_some());
    }

    #[test]
    fn es256_auto_assigns_jti() {
        let key = generate_es256_key();
        let claims = Claims::new("bob".to_owned(), 0, 9_999_999_999);
        assert!(claims.jti.is_none());
        let token = encode_es256(&claims, &key);

        let parts: Vec<&str> = token.split('.').collect();
        let payload_bytes = URL_SAFE_NO_PAD.decode(parts[1]).unwrap();
        let decoded: Claims = serde_json::from_slice(&payload_bytes).unwrap();
        assert!(decoded.jti.is_some());
    }

    #[test]
    fn ml_dsa_65_roundtrip() {
        let (sk, vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        let token = encode_ml_dsa_65(&claims, &sk);

        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);

        let header_bytes = URL_SAFE_NO_PAD.decode(parts[0]).unwrap();
        let header: serde_json::Value = serde_json::from_slice(&header_bytes).unwrap();
        assert_eq!(header["alg"], "ML-DSA-65");
        assert_eq!(header["typ"], "at+jwt");
        assert!(header["kid"].as_str().unwrap().len() == 43);

        let decoded = hyprstream_rpc::auth::jwt::decode_ml_dsa_65(&token, &vk, None).unwrap();
        assert_eq!(decoded.sub, "alice");
        assert!(decoded.jti.is_some());
    }

    #[test]
    fn ml_dsa_65_jwk_structure() {
        let (sk, _) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let vk = ml_dsa::Keypair::verifying_key(&sk);
        let jwk = ml_dsa_65_jwk(&vk);
        assert_eq!(jwk["kty"], "AKP");
        assert_eq!(jwk["alg"], "ML-DSA-65");
        assert_eq!(jwk["use"], "sig");
        assert!(jwk["kid"].as_str().unwrap().len() == 43);
        assert!(jwk["pub"].as_str().is_some());
    }

    #[test]
    fn composite_ml_dsa_65_ed25519_roundtrip() {
        let (ml_dsa_sk, ml_dsa_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let ed25519_sk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let ed25519_vk = ed25519_sk.verifying_key();

        let claims = Claims::new("bob".to_owned(), 0, 9_999_999_999);
        let token = encode_composite_ml_dsa_65_ed25519(&claims, &ml_dsa_sk, &ed25519_sk);

        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);

        let header_bytes = URL_SAFE_NO_PAD.decode(parts[0]).unwrap();
        let header: serde_json::Value = serde_json::from_slice(&header_bytes).unwrap();
        assert_eq!(header["alg"], "ML-DSA-65-Ed25519");

        let dispatch = hyprstream_rpc::auth::parse_composite_dispatch(&token, &["at+jwt"]).unwrap();
        let decoded = hyprstream_rpc::auth::jwt::decode_composite(
            &token, &ml_dsa_vk, &ed25519_vk, None, &dispatch,
        ).unwrap();
        assert_eq!(decoded.sub, "bob");
        assert!(decoded.jti.is_some());
    }

    #[test]
    fn composite_jwk_structure() {
        let (ml_dsa_sk, _) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let ml_dsa_vk = ml_dsa::Keypair::verifying_key(&ml_dsa_sk);
        let ed25519_sk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let ed25519_vk = ed25519_sk.verifying_key();

        let jwk = composite_jwk(&ml_dsa_vk, &ed25519_vk);
        assert_eq!(jwk["kty"], "AKP");
        assert_eq!(jwk["alg"], "ML-DSA-65-Ed25519");
        assert_eq!(jwk["use"], "sig");
        assert!(jwk["kid"].as_str().unwrap().len() == 43);
        assert!(jwk["pub"].as_str().is_some());
    }

    #[test]
    fn ml_dsa_65_wrong_key_rejects() {
        let (sk, _) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let (_, wrong_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        let token = encode_ml_dsa_65(&claims, &sk);
        assert!(hyprstream_rpc::auth::jwt::decode_ml_dsa_65(&token, &wrong_vk, None).is_err());
    }

    #[test]
    fn composite_wrong_ed25519_key_rejects() {
        let (ml_dsa_sk, ml_dsa_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let ed25519_sk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let wrong_ed25519_vk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();

        let claims = Claims::new("bob".to_owned(), 0, 9_999_999_999);
        let token = encode_composite_ml_dsa_65_ed25519(&claims, &ml_dsa_sk, &ed25519_sk);
        let dispatch = hyprstream_rpc::auth::parse_composite_dispatch(&token, &["at+jwt"]).unwrap();
        assert!(hyprstream_rpc::auth::jwt::decode_composite(
            &token, &ml_dsa_vk, &wrong_ed25519_vk, None, &dispatch,
        ).is_err());
    }

    #[test]
    fn ml_dsa_65_expired_token_rejected() {
        let (sk, vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let claims = Claims::new("alice".to_owned(), 0, 1);
        let token = encode_ml_dsa_65(&claims, &sk);
        let err = hyprstream_rpc::auth::jwt::decode_ml_dsa_65(&token, &vk, None).unwrap_err();
        assert!(matches!(err, hyprstream_rpc::auth::JwtError::Expired));
    }

    #[test]
    fn composite_expired_token_rejected() {
        let (ml_dsa_sk, ml_dsa_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let ed25519_sk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let ed25519_vk = ed25519_sk.verifying_key();
        let claims = Claims::new("bob".to_owned(), 0, 1);
        let token = encode_composite_ml_dsa_65_ed25519(&claims, &ml_dsa_sk, &ed25519_sk);
        let dispatch = hyprstream_rpc::auth::parse_composite_dispatch(&token, &["at+jwt"]).unwrap();
        let err = hyprstream_rpc::auth::jwt::decode_composite(
            &token, &ml_dsa_vk, &ed25519_vk, None, &dispatch,
        ).unwrap_err();
        assert!(matches!(err, hyprstream_rpc::auth::JwtError::Expired));
    }

    #[test]
    fn ml_dsa_65_rejects_eddsa_token() {
        let ed25519_sk = ed25519_dalek::SigningKey::from_bytes(&[42u8; 32]);
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        let token = hyprstream_rpc::auth::encode(&claims, &ed25519_sk);
        let (_, vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        assert!(hyprstream_rpc::auth::jwt::decode_ml_dsa_65(&token, &vk, None).is_err());
    }

    #[test]
    fn ml_dsa_65_lenient_audience() {
        let (sk, vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        // Token without aud should be accepted even when expected_aud is set
        let token = encode_ml_dsa_65(&claims, &sk);
        assert!(hyprstream_rpc::auth::jwt::decode_ml_dsa_65(
            &token, &vk, Some("https://example.com"),
        ).is_ok());
    }

    #[test]
    fn ml_dsa_65_wrong_audience_rejected() {
        let (sk, vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let mut claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        claims.aud = Some("https://wrong.example.com".to_owned());
        let token = encode_ml_dsa_65(&claims, &sk);
        let err = hyprstream_rpc::auth::jwt::decode_ml_dsa_65(
            &token, &vk, Some("https://example.com"),
        ).unwrap_err();
        assert!(matches!(err, hyprstream_rpc::auth::JwtError::InvalidAudience));
    }
}
