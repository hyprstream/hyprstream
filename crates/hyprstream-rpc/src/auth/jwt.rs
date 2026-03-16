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

/// JWT header (static for EdDSA)
const JWT_HEADER: &str = r#"{"alg":"EdDSA","typ":"JWT"}"#;

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
}

/// Encode and sign a JWT token
///
/// Returns a standard RFC 7519 JWT (no prefix).
pub fn encode(claims: &Claims, signing_key: &SigningKey) -> String {
    // Encode header and payload
    let header_b64 = URL_SAFE_NO_PAD.encode(JWT_HEADER);
    let payload_json = serde_json::to_string(claims).unwrap_or_else(|_e| {
        #[cfg(not(target_arch = "wasm32"))]
        tracing::error!("JWT claims serialization failed: {}", _e);
        "{}".to_owned()
    });
    let payload_b64 = URL_SAFE_NO_PAD.encode(&payload_json);

    // Create signing input
    let signing_input = format!("{header_b64}.{payload_b64}");

    // Sign with Ed25519
    let signature = signing_key.sign(signing_input.as_bytes());
    let signature_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());

    // Combine into JWT
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

    // Validate audience if expected
    if let Some(expected) = expected_aud {
        match &claims.aud {
            Some(aud) if aud == expected => {}        // correct audience
            None if lenient_aud => {}                 // absent — accept in lenient mode
            Some(_) | None => return Err(JwtError::InvalidAudience), // wrong or absent audience
        }
    }

    Ok(claims)
}

/// Decode a JWT without verifying the signature (for introspection)
///
/// WARNING: Only use this for debugging or when signature has already been verified.
/// Restricted to test and debug builds to prevent misuse in production.
#[cfg(test)]
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
}
