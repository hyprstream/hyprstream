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
}

/// Encode and sign a JWT token
///
/// Returns a standard RFC 7519 JWT (no prefix).
pub fn encode(claims: &Claims, signing_key: &SigningKey) -> String {
    // Encode header and payload
    let header_b64 = URL_SAFE_NO_PAD.encode(JWT_HEADER);
    let payload_json = serde_json::to_string(claims).unwrap_or_else(|e| {
        tracing::error!("JWT claims serialization failed: {}", e);
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

/// Decode and verify a JWT token
///
/// Accepts a raw RFC 7519 JWT. Returns the claims if valid and not expired.
pub fn decode(token: &str, verifying_key: &VerifyingKey) -> Result<Claims, JwtError> {
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
