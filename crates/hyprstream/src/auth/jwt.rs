//! JWT token implementation.
//!
//! Re-exports EdDSA signing from hyprstream-rpc and adds ES256 (P-256) signing.

pub use hyprstream_rpc::auth::{
    decode, decode_id_token_unverified, decode_id_token_with_key, decode_with_any_key,
    decode_with_any_key_lenient, decode_with_expectation, decode_with_federation_candidates,
    decode_with_key, decode_with_key_expectation, encode, encode_service_jwt,
    AudienceExpectation, Claims, JwtError,
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

        let decoded = hyprstream_rpc::auth::jwt::decode_ml_dsa_65(
            &token,
            &vk,
            AudienceExpectation::ExplicitlyUnchecked {
                reason: "the round-trip test exercises ML-DSA signing, not audience",
            },
        )
        .unwrap();
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
            &token,
            &ml_dsa_vk,
            &ed25519_vk,
            AudienceExpectation::ExplicitlyUnchecked {
                reason: "the round-trip test exercises composite signing, not audience",
            },
            &dispatch,
        )
        .unwrap();
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
        assert!(hyprstream_rpc::auth::jwt::decode_ml_dsa_65(
            &token,
            &wrong_vk,
            AudienceExpectation::ExplicitlyUnchecked {
                reason: "the test exercises wrong-key rejection, not audience",
            },
        )
        .is_err());
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
            &token,
            &ml_dsa_vk,
            &wrong_ed25519_vk,
            AudienceExpectation::ExplicitlyUnchecked {
                reason: "the test exercises wrong-key rejection, not audience",
            },
            &dispatch,
        )
        .is_err());
    }

    #[test]
    fn ml_dsa_65_expired_token_rejected() {
        let (sk, vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let claims = Claims::new("alice".to_owned(), 0, 1);
        let token = encode_ml_dsa_65(&claims, &sk);
        let err = hyprstream_rpc::auth::jwt::decode_ml_dsa_65(
            &token,
            &vk,
            AudienceExpectation::ExplicitlyUnchecked {
                reason: "the test exercises expiry rejection, not audience",
            },
        )
        .unwrap_err();
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
            &token,
            &ml_dsa_vk,
            &ed25519_vk,
            AudienceExpectation::ExplicitlyUnchecked {
                reason: "the test exercises expiry rejection, not audience",
            },
            &dispatch,
        )
        .unwrap_err();
        assert!(matches!(err, hyprstream_rpc::auth::JwtError::Expired));
    }

    #[test]
    fn ml_dsa_65_rejects_eddsa_token() {
        let ed25519_sk = ed25519_dalek::SigningKey::from_bytes(&[42u8; 32]);
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        let token = hyprstream_rpc::auth::encode(&claims, &ed25519_sk);
        let (_, vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        assert!(hyprstream_rpc::auth::jwt::decode_ml_dsa_65(
            &token,
            &vk,
            AudienceExpectation::ExplicitlyUnchecked {
                reason: "the test exercises algorithm rejection, not audience",
            },
        )
        .is_err());
    }

    #[test]
    fn ml_dsa_65_exact_audience_rejects_absent_claim() {
        let (sk, vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        let token = encode_ml_dsa_65(&claims, &sk);
        assert!(matches!(
            hyprstream_rpc::auth::jwt::decode_ml_dsa_65(
                &token,
                &vk,
                AudienceExpectation::Exact("https://example.com"),
            ),
            Err(hyprstream_rpc::auth::JwtError::InvalidAudience)
        ));

        hyprstream_rpc::auth::jwt::decode_ml_dsa_65(
            &token,
            &vk,
            AudienceExpectation::ExactOrMissing {
                expected: "https://example.com",
                reason: "federated ML-DSA compatibility test",
            },
        )
        .expect("named compatibility posture permits an absent audience");
    }

    #[test]
    fn ml_dsa_65_wrong_audience_rejected() {
        let (sk, vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let mut claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        claims.aud = Some("https://wrong.example.com".to_owned());
        let token = encode_ml_dsa_65(&claims, &sk);
        let err = hyprstream_rpc::auth::jwt::decode_ml_dsa_65(
            &token,
            &vk,
            AudienceExpectation::Exact("https://example.com"),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            hyprstream_rpc::auth::JwtError::InvalidAudience
        ));
    }
}
