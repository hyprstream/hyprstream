//! JWT token implementation with Ed25519 (EdDSA) signatures.
//!
//! Implements a minimal JWT encoder/decoder for API authentication.
//! Uses Ed25519 signatures for stateless token validation.
//!
//! # Token Format
//!
//! Standard RFC 7519 JWT:
//! - Header: `{"alg":"EdDSA","typ":"JWT"}`
//! - Payload: Claims (sub, exp, iat)
//! - Signature: Ed25519 over `base64url(header).base64url(payload)`

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::Utc;
use ed25519_dalek::{Signature, Signer, Verifier, SigningKey, VerifyingKey};
use thiserror::Error;

use super::Claims;

/// Input for RFC 7638 JWK Thumbprint computation.
///
/// Each variant carries the required members for its key type's canonical
/// JSON representation (RFC 7638 §3.2: members sorted lexicographically).
pub enum JwkThumbprintInput<'a> {
    /// OKP / Ed25519: canonical `{"crv":"Ed25519","kty":"OKP","x":"..."}`
    Ed25519 { x: &'a [u8; 32] },
    /// EC / P-256 (ES256): canonical `{"crv":"P-256","kty":"EC","x":"...","y":"..."}`
    Es256 { x: &'a [u8; 32], y: &'a [u8; 32] },
    /// RSA: canonical `{"e":"...","kty":"RSA","n":"..."}` (n, e already base64url)
    Rsa { n: &'a str, e: &'a str },
    /// AKP / ML-DSA-65 (draft-ietf-cose-dilithium-11): canonical
    /// `{"alg":"ML-DSA-65","kty":"AKP","pub":"<base64url>"}`
    Akp { alg: &'a str, pub_bytes: &'a [u8] },
}

/// Compute the RFC 7638 JWK Thumbprint for any supported key type.
///
/// Returns `base64url(SHA-256(canonical_jwk_json))` — a 43-char string.
pub fn jwk_thumbprint(input: &JwkThumbprintInput<'_>) -> String {
    use sha2::{Digest, Sha256};
    let canonical = match input {
        JwkThumbprintInput::Ed25519 { x } => {
            let x_b64 = URL_SAFE_NO_PAD.encode(x);
            format!(r#"{{"crv":"Ed25519","kty":"OKP","x":"{}"}}"#, x_b64)
        }
        JwkThumbprintInput::Es256 { x, y } => {
            let x_b64 = URL_SAFE_NO_PAD.encode(x);
            let y_b64 = URL_SAFE_NO_PAD.encode(y);
            format!(r#"{{"crv":"P-256","kty":"EC","x":"{}","y":"{}"}}"#, x_b64, y_b64)
        }
        JwkThumbprintInput::Rsa { n, e } => {
            format!(r#"{{"e":"{}","kty":"RSA","n":"{}"}}"#, e, n)
        }
        JwkThumbprintInput::Akp { alg, pub_bytes } => {
            let pub_b64 = URL_SAFE_NO_PAD.encode(pub_bytes);
            format!(r#"{{"alg":"{}","kty":"AKP","pub":"{}"}}"#, alg, pub_b64)
        }
    };
    let hash = Sha256::digest(canonical.as_bytes());
    URL_SAFE_NO_PAD.encode(hash)
}

/// Compute the JWT `kid` for an Ed25519 signing key using RFC 7638 JWK Thumbprint.
pub fn kid_for_key(signing_key: &SigningKey) -> String {
    jwk_thumbprint(&JwkThumbprintInput::Ed25519 {
        x: signing_key.verifying_key().as_bytes(),
    })
}

/// Extract the `alg` from a JWT's JOSE header without verifying the signature.
///
/// Returns `Ok(None)` if the header has no `alg` field.
pub fn header_alg(token: &str) -> Result<Option<String>, JwtError> {
    let header_b64 = token.split('.').next().ok_or(JwtError::InvalidFormat)?;
    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .map_err(|_| JwtError::InvalidBase64)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| JwtError::InvalidJson(e.to_string()))?;
    Ok(header.get("alg").and_then(|v| v.as_str()).map(String::from))
}

/// Extract the `kid` from a JWT's JOSE header without verifying the signature.
///
/// Returns `Ok(None)` if the header has no `kid` field.
pub fn header_kid(token: &str) -> Result<Option<String>, JwtError> {
    let header_b64 = token.split('.').next().ok_or(JwtError::InvalidFormat)?;
    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .map_err(|_| JwtError::InvalidBase64)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| JwtError::InvalidJson(e.to_string()))?;
    Ok(header.get("kid").and_then(|v| v.as_str()).map(String::from))
}

/// Errors from JWT operations
#[derive(Error, Debug)]
pub enum JwtError {
    #[error("Invalid token format")]
    InvalidFormat,

    #[error("Invalid base64 encoding")]
    InvalidBase64,

    #[error("Invalid JSON: {0}")]
    InvalidJson(String),

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Token expired")]
    Expired,

    #[error("Token not yet valid")]
    NotYetValid,

    #[error("Missing required claim: {0}")]
    MissingClaim(String),

    #[error("Invalid audience")]
    InvalidAudience,

    #[error("Unsupported algorithm: {0}")]
    UnsupportedAlgorithm(String),
}

/// Encode and sign a JWT with a specific JOSE header.
fn encode_with_header(claims: &Claims, signing_key: &SigningKey, header_json: &str) -> String {
    let header_b64 = URL_SAFE_NO_PAD.encode(header_json);
    let payload_json = serde_json::to_string(claims).unwrap_or_else(|_e| {
        #[cfg(not(target_arch = "wasm32"))]
        tracing::error!("JWT claims serialization failed: {}", _e);
        "{}".to_owned()
    });
    let payload_b64 = URL_SAFE_NO_PAD.encode(&payload_json);
    let signing_input = format!("{header_b64}.{payload_b64}");
    let signature = signing_key.sign(signing_input.as_bytes());
    let signature_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());
    format!("{signing_input}.{signature_b64}")
}

/// Encode a user/client OAuth 2.0 access token (`typ: "at+jwt"`, RFC 9068).
/// Includes `kid` in the JOSE header for key rotation support.
/// Automatically assigns a `jti` if the claims don't already have one.
pub fn encode(claims: &Claims, signing_key: &SigningKey) -> String {
    let kid = kid_for_key(signing_key);
    let header = format!(r#"{{"alg":"EdDSA","typ":"at+jwt","kid":"{}"}}"#, kid);
    let claims = ensure_jti(claims);
    encode_with_header(&claims, signing_key, &header)
}

/// Encode a WIMSE Workload Identity Token (`typ: "wit+jwt"`) for service JWTs.
/// Includes `kid` in the JOSE header for key rotation support.
/// Automatically assigns a `jti` if the claims don't already have one.
pub fn encode_service_jwt(claims: &Claims, signing_key: &SigningKey) -> String {
    let kid = kid_for_key(signing_key);
    let header = format!(r#"{{"alg":"EdDSA","typ":"wit+jwt","kid":"{}"}}"#, kid);
    let claims = ensure_jti(claims);
    encode_with_header(&claims, signing_key, &header)
}

fn ensure_jti(claims: &Claims) -> std::borrow::Cow<'_, Claims> {
    if claims.jti.is_some() {
        std::borrow::Cow::Borrowed(claims)
    } else {
        std::borrow::Cow::Owned(claims.clone().with_jti())
    }
}

/// Encode and sign an OIDC ID Token (EdDSA with `kid` in header).
///
/// The header includes `kid` (SHA-256 of the public key, first 8 hex chars)
/// and `typ: "JWT"` per OIDC convention.
pub fn encode_id_token(claims: &super::IdTokenClaims, signing_key: &SigningKey) -> String {
    let kid = kid_for_key(signing_key);
    let header = serde_json::json!({
        "alg": "EdDSA",
        "typ": "JWT",
        "kid": kid,
    });
    let header_b64 = URL_SAFE_NO_PAD.encode(header.to_string().as_bytes());
    let payload_json = serde_json::to_string(claims).unwrap_or_else(|_e| {
        #[cfg(not(target_arch = "wasm32"))]
        tracing::error!("id_token claims serialization failed: {}", _e);
        "{}".to_owned()
    });
    let payload_b64 = URL_SAFE_NO_PAD.encode(&payload_json);
    let signing_input = format!("{header_b64}.{payload_b64}");
    let signature = signing_key.sign(signing_input.as_bytes());
    let signature_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());
    format!("{signing_input}.{signature_b64}")
}

/// Decode and verify a JWT token issued by the **local** node.
///
/// For tokens from foreign issuers (federation), use [`decode_with_key`]
/// with the key obtained from `FederationKeyResolver::get_key`.
///
/// Uses strict audience validation: if `expected_aud` is `Some`, the token
/// must have a matching `aud` claim (absent `aud` is rejected).
pub fn decode(token: &str, verifying_key: &VerifyingKey, expected_aud: Option<&str>) -> Result<Claims, JwtError> {
    decode_inner(token, verifying_key, expected_aud, false)
}

/// Decode a JWT using a caller-supplied verifying key (for multi-issuer support).
///
/// Uses lenient audience validation: if `expected_aud` is `Some`, a wrong `aud`
/// is rejected but an absent `aud` is accepted. This allows federated tokens
/// from issuers that don't set `aud` while still rejecting cross-node replay
/// attacks from issuers that do.
pub fn decode_with_key(
    token: &str,
    verifying_key: &VerifyingKey,
    expected_aud: Option<&str>,
) -> Result<Claims, JwtError> {
    decode_inner(token, verifying_key, expected_aud, true)
}

/// Internal decode implementation shared by `decode` and `decode_with_key`.
///
/// `lenient_aud`: when true, accepts tokens with no `aud` claim even when
/// `expected_aud` is `Some`. Wrong `aud` is always rejected.
fn decode_inner(token: &str, verifying_key: &VerifyingKey, expected_aud: Option<&str>, lenient_aud: bool) -> Result<Claims, JwtError> {
    // Split into parts
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(JwtError::InvalidFormat);
    }

    let header_b64 = parts[0];
    let payload_b64 = parts[1];
    let signature_b64 = parts[2];

    // Verify signature first
    let signing_input = format!("{header_b64}.{payload_b64}");
    let signature_bytes = URL_SAFE_NO_PAD
        .decode(signature_b64)
        .map_err(|_| JwtError::InvalidBase64)?;

    if signature_bytes.len() != 64 {
        return Err(JwtError::InvalidSignature);
    }

    let mut sig_array = [0u8; 64];
    sig_array.copy_from_slice(&signature_bytes);
    let signature = Signature::from_bytes(&sig_array);

    verifying_key
        .verify(signing_input.as_bytes(), &signature)
        .map_err(|_| JwtError::InvalidSignature)?;

    // Validate the JWT header `alg` field to prevent algorithm-agility attacks.
    // We only accept EdDSA tokens; any other alg value is rejected.
    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .map_err(|_| JwtError::InvalidBase64)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| JwtError::InvalidJson(e.to_string()))?;
    if header.get("alg").and_then(|v| v.as_str()) != Some("EdDSA") {
        return Err(JwtError::InvalidSignature);
    }

    // Decode payload
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|_| JwtError::InvalidBase64)?;

    let claims: Claims = serde_json::from_slice(&payload_bytes)
        .map_err(|e| JwtError::InvalidJson(e.to_string()))?;

    // Check expiration
    if claims.is_expired() {
        return Err(JwtError::Expired);
    }

    // Check iat is not in the future (with small tolerance)
    let now = Utc::now().timestamp();
    if claims.iat > now + 60 {
        return Err(JwtError::NotYetValid);
    }

    // Validate audience if expected (trailing-slash tolerant)
    if let Some(expected) = expected_aud {
        let expected_norm = expected.trim_end_matches('/');
        match &claims.aud {
            Some(aud) if aud.trim_end_matches('/') == expected_norm => {}
            None if lenient_aud => {}
            Some(_) | None => return Err(JwtError::InvalidAudience),
        }
    }

    Ok(claims)
}

/// Decode a JWT without verifying the signature.
///
/// Used by `verify_claims()` to peek at the issuer field before selecting
/// the appropriate verification key (local vs federated). The token is
/// always fully verified after issuer routing.
pub fn decode_unverified(token: &str) -> Result<Claims, JwtError> {
    // Split into parts
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(JwtError::InvalidFormat);
    }

    // Decode payload only
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|_| JwtError::InvalidBase64)?;

    serde_json::from_slice(&payload_bytes)
        .map_err(|e| JwtError::InvalidJson(e.to_string()))
}

/// Decode and verify a JWT signed with ML-DSA-65 (`alg: "ML-DSA-65"`).
///
/// Uses lenient audience validation (same as `decode_with_key`): if
/// `expected_aud` is `Some`, a wrong `aud` is rejected but an absent
/// `aud` is accepted.
pub fn decode_ml_dsa_65(
    token: &str,
    vk: &crate::crypto::pq::MlDsaVerifyingKey,
    expected_aud: Option<&str>,
) -> Result<Claims, JwtError> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(JwtError::InvalidFormat);
    }

    let signing_input = format!("{}.{}", parts[0], parts[1]);
    let sig_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|_| JwtError::InvalidBase64)?;

    crate::crypto::pq::ml_dsa_verify(vk, signing_input.as_bytes(), &sig_bytes)
        .map_err(|_| JwtError::InvalidSignature)?;

    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .map_err(|_| JwtError::InvalidBase64)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| JwtError::InvalidJson(e.to_string()))?;
    if header.get("alg").and_then(|v| v.as_str()) != Some("ML-DSA-65") {
        return Err(JwtError::UnsupportedAlgorithm(
            header.get("alg").and_then(|v| v.as_str()).unwrap_or("none").to_owned(),
        ));
    }

    let payload_bytes = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|_| JwtError::InvalidBase64)?;
    let claims: Claims = serde_json::from_slice(&payload_bytes)
        .map_err(|e| JwtError::InvalidJson(e.to_string()))?;

    if claims.is_expired() {
        return Err(JwtError::Expired);
    }
    let now = Utc::now().timestamp();
    if claims.iat > now + 60 {
        return Err(JwtError::NotYetValid);
    }

    if let Some(expected) = expected_aud {
        match &claims.aud {
            Some(aud) if aud == expected => {}
            None => {}                                    // lenient: absent aud accepted
            Some(_) => return Err(JwtError::InvalidAudience), // wrong aud rejected
        }
    }

    Ok(claims)
}

/// Decode and verify a composite ML-DSA-65-Ed25519 JWT.
///
/// Per draft-ietf-jose-pq-composite-sigs, the signature is
/// `ml_dsa_sig (3309 bytes) ∥ ed25519_sig (64 bytes)`.
pub fn decode_composite(
    token: &str,
    ml_dsa_vk: &crate::crypto::pq::MlDsaVerifyingKey,
    ed25519_vk: &VerifyingKey,
    expected_aud: Option<&str>,
) -> Result<Claims, JwtError> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(JwtError::InvalidFormat);
    }

    let signing_input = format!("{}.{}", parts[0], parts[1]);
    let sig_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|_| JwtError::InvalidBase64)?;

    // Composite signature: ML-DSA-65 (3309 bytes) + Ed25519 (64 bytes) = 3373 bytes
    if sig_bytes.len() != 3309 + 64 {
        return Err(JwtError::InvalidSignature);
    }

    let (ml_dsa_sig, ed25519_sig) = sig_bytes.split_at(3309);

    // Build message per draft-ietf-jose-pq-composite-sigs:
    // Hash(header.payload) used as message for both algorithms
    let message = signing_input.as_bytes();

    // Verify ML-DSA-65
    crate::crypto::pq::ml_dsa_verify(ml_dsa_vk, message, ml_dsa_sig)
        .map_err(|_| JwtError::InvalidSignature)?;

    // Verify Ed25519
    if ed25519_sig.len() != 64 {
        return Err(JwtError::InvalidSignature);
    }
    let mut ed_sig_arr = [0u8; 64];
    ed_sig_arr.copy_from_slice(ed25519_sig);
    let ed_signature = Signature::from_bytes(&ed_sig_arr);
    ed25519_vk
        .verify_strict(message, &ed_signature)
        .map_err(|_| JwtError::InvalidSignature)?;

    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .map_err(|_| JwtError::InvalidBase64)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| JwtError::InvalidJson(e.to_string()))?;
    if header.get("alg").and_then(|v| v.as_str()) != Some("ML-DSA-65-Ed25519") {
        return Err(JwtError::UnsupportedAlgorithm(
            header.get("alg").and_then(|v| v.as_str()).unwrap_or("none").to_owned(),
        ));
    }

    let payload_bytes = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|_| JwtError::InvalidBase64)?;
    let claims: Claims = serde_json::from_slice(&payload_bytes)
        .map_err(|e| JwtError::InvalidJson(e.to_string()))?;

    if claims.is_expired() {
        return Err(JwtError::Expired);
    }
    let now = Utc::now().timestamp();
    if claims.iat > now + 60 {
        return Err(JwtError::NotYetValid);
    }

    if let Some(expected) = expected_aud {
        match &claims.aud {
            Some(aud) if aud == expected => {}
            None => {}                                    // lenient: absent aud accepted
            Some(_) => return Err(JwtError::InvalidAudience), // wrong aud rejected
        }
    }

    Ok(claims)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::auth::Claims;
    use ed25519_dalek::SigningKey;

    fn make_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    #[test]
    fn test_decode_with_key_multi_issuer() {
        // Simulate two nodes with separate signing keys.
        let key_a = make_key(0xAA);
        let key_b = make_key(0xBB);
        let vk_a = key_a.verifying_key();

        // Node A issues a JWT with iss = "https://node-a"
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999)
            .with_issuer("https://node-a".to_owned());
        let token = encode(&claims, &key_a);

        // decode_with_key should accept the correct key
        let decoded = decode_with_key(&token, &vk_a, None)
            .expect("decode_with_key with correct key must succeed");
        assert_eq!(decoded.iss, "https://node-a");
        assert_eq!(decoded.sub, "alice");

        // decode_with_key must reject the wrong key
        let vk_b = key_b.verifying_key();
        let result = decode_with_key(&token, &vk_b, None);
        assert!(
            matches!(result, Err(JwtError::InvalidSignature)),
            "wrong key must yield InvalidSignature"
        );
    }

    #[test]
    fn test_decode_with_key_audience_validation() {
        let key = make_key(0x01);
        let vk = key.verifying_key();
        let claims = Claims::new("bob".to_owned(), 0, 9_999_999_999)
            .with_issuer("https://node-a".to_owned())
            .with_audience(Some("https://node-b".to_owned()));
        let token = encode(&claims, &key);

        // Correct audience
        decode_with_key(&token, &vk, Some("https://node-b"))
            .expect("correct audience must succeed");

        // Wrong audience
        let result = decode_with_key(&token, &vk, Some("https://wrong"));
        assert!(matches!(result, Err(JwtError::InvalidAudience)));
    }

    #[test]
    fn test_decode_and_decode_with_key_are_equivalent() {
        let key = make_key(0x42);
        let vk = key.verifying_key();
        let claims = Claims::new("carol".to_owned(), 0, 9_999_999_999);
        let token = encode(&claims, &key);

        let via_decode = decode(&token, &vk, None).unwrap();
        let via_decode_with_key = decode_with_key(&token, &vk, None).unwrap();
        assert_eq!(via_decode.sub, via_decode_with_key.sub);
        assert_eq!(via_decode.exp, via_decode_with_key.exp);
    }

    #[test]
    fn test_decode_strict_rejects_absent_audience() {
        // Local tokens (via decode) must have aud present when expected
        let key = make_key(0x10);
        let vk = key.verifying_key();
        let claims = Claims::new("dave".to_owned(), 0, 9_999_999_999);
        let token = encode(&claims, &key);

        let result = decode(&token, &vk, Some("http://localhost:6789"));
        assert!(
            matches!(result, Err(JwtError::InvalidAudience)),
            "strict mode must reject absent aud"
        );
    }

    #[test]
    fn test_decode_with_key_lenient_accepts_absent_audience() {
        // Federated tokens (via decode_with_key) accept absent aud
        let key = make_key(0x20);
        let vk = key.verifying_key();
        let claims = Claims::new("eve".to_owned(), 0, 9_999_999_999)
            .with_issuer("https://remote-node".to_owned());
        let token = encode(&claims, &key);

        // Lenient mode: absent aud is accepted
        decode_with_key(&token, &vk, Some("http://localhost:6789"))
            .expect("lenient mode must accept absent aud");
    }

    #[test]
    fn test_decode_with_key_lenient_rejects_wrong_audience() {
        // Federated tokens with wrong aud must still be rejected
        let key = make_key(0x30);
        let vk = key.verifying_key();
        let claims = Claims::new("frank".to_owned(), 0, 9_999_999_999)
            .with_issuer("https://node-a".to_owned())
            .with_audience(Some("https://node-b".to_owned()));
        let token = encode(&claims, &key);

        let result = decode_with_key(&token, &vk, Some("https://node-c"));
        assert!(
            matches!(result, Err(JwtError::InvalidAudience)),
            "lenient mode must reject wrong aud"
        );
    }

    #[test]
    fn test_jwk_thumbprint_ed25519() {
        let key = make_key(0x01);
        let kid = kid_for_key(&key);
        // RFC 7638: base64url(SHA-256(...)) = 43 chars
        assert_eq!(kid.len(), 43);
        assert!(!kid.contains('='));
        // Deterministic
        assert_eq!(kid, kid_for_key(&key));
    }

    #[test]
    fn test_jwk_thumbprint_rsa() {
        let kid = jwk_thumbprint(&JwkThumbprintInput::Rsa {
            n: "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
            e: "AQAB",
        });
        assert_eq!(kid.len(), 43);
        assert!(!kid.contains('='));
    }

    #[test]
    fn test_jwk_thumbprint_es256() {
        let x = [1u8; 32];
        let y = [2u8; 32];
        let kid = jwk_thumbprint(&JwkThumbprintInput::Es256 { x: &x, y: &y });
        assert_eq!(kid.len(), 43);
        assert!(!kid.contains('='));
    }

    #[test]
    fn test_jwk_thumbprint_different_algorithms_differ() {
        let bytes = [1u8; 32];
        let ed_kid = jwk_thumbprint(&JwkThumbprintInput::Ed25519 { x: &bytes });
        let es_kid = jwk_thumbprint(&JwkThumbprintInput::Es256 { x: &bytes, y: &[2u8; 32] });
        assert_ne!(ed_kid, es_kid);
    }

    #[test]
    fn test_header_kid_extraction() {
        let key = make_key(0x42);
        let claims = Claims::new("test".to_owned(), 0, 9_999_999_999);
        let token = encode(&claims, &key);

        let kid = header_kid(&token).unwrap();
        assert!(kid.is_some());
        assert_eq!(kid.unwrap(), kid_for_key(&key));
    }

    #[test]
    fn test_header_kid_missing() {
        // Manually craft a JWT with no kid in header
        let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"EdDSA","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(r#"{"sub":"test","exp":9999999999,"iat":0}"#);
        let token = format!("{}.{}.AAAA", header, payload);

        let kid = header_kid(&token).unwrap();
        assert!(kid.is_none());
    }
}
