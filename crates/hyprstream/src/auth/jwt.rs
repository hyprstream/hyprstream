//! JWT token implementation with Ed25519 (EdDSA) signatures.
//!
//! Implements a minimal JWT encoder/decoder for API authentication.
//! Uses Ed25519 signatures for stateless token validation.
//!
//! # Token Format
//!
//! ```text
//! hypr_eyJ...  (standard token)
//! hypr_admin_eyJ...  (admin token)
//! ```
//!
//! The JWT itself follows RFC 7519:
//! - Header: `{"alg":"EdDSA","typ":"JWT"}`
//! - Payload: Claims (sub, exp, iat, scope, admin)
//! - Signature: Ed25519 over `base64url(header).base64url(payload)`

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{DateTime, Duration, Utc};
use ed25519_dalek::{Signature, Signer, Verifier, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Token prefix for standard API keys
pub const TOKEN_PREFIX: &str = "hypr_";

/// Token prefix for admin keys
pub const ADMIN_TOKEN_PREFIX: &str = "hypr_admin_";

/// JWT header (static for EdDSA)
const JWT_HEADER: &str = r#"{"alg":"EdDSA","typ":"JWT"}"#;

/// Errors from JWT operations
#[derive(Error, Debug)]
pub enum JwtError {
    #[error("Invalid token format")]
    InvalidFormat,

    #[error("Invalid token prefix")]
    InvalidPrefix,

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

/// JWT claims for API tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user identifier)
    pub sub: String,

    /// Expiration time (Unix timestamp)
    pub exp: i64,

    /// Issued at (Unix timestamp)
    pub iat: i64,

    /// Resource scopes (e.g., ["model:*", "data:analytics"])
    #[serde(default)]
    pub scope: Vec<String>,

    /// Admin flag
    #[serde(default)]
    pub admin: bool,
}

impl Claims {
    /// Create new claims for a user
    pub fn new(user: &str, expires_in: Duration, scopes: Vec<String>, admin: bool) -> Self {
        let now = Utc::now();
        Self {
            sub: user.to_string(),
            exp: (now + expires_in).timestamp(),
            iat: now.timestamp(),
            scope: scopes,
            admin,
        }
    }

    /// Check if the token has expired
    pub fn is_expired(&self) -> bool {
        Utc::now().timestamp() > self.exp
    }

    /// Get expiration as DateTime
    pub fn expires_at(&self) -> Option<DateTime<Utc>> {
        DateTime::from_timestamp(self.exp, 0)
    }

    /// Get issued at as DateTime
    pub fn issued_at(&self) -> Option<DateTime<Utc>> {
        DateTime::from_timestamp(self.iat, 0)
    }

    /// Check if token has access to a resource
    pub fn has_scope(&self, resource: &str) -> bool {
        if self.scope.is_empty() || self.scope.contains(&"*".to_string()) {
            return true;
        }
        self.scope.iter().any(|scope| {
            if scope.ends_with('*') {
                let prefix = &scope[..scope.len() - 1];
                resource.starts_with(prefix)
            } else {
                scope == resource
            }
        })
    }
}

/// Encode and sign a JWT token
///
/// Returns a token with the appropriate prefix (hypr_ or hypr_admin_)
pub fn encode(claims: &Claims, signing_key: &SigningKey) -> String {
    // Encode header and payload
    let header_b64 = URL_SAFE_NO_PAD.encode(JWT_HEADER);
    let payload_json = serde_json::to_string(claims).expect("Claims serialization cannot fail");
    let payload_b64 = URL_SAFE_NO_PAD.encode(&payload_json);

    // Create signing input
    let signing_input = format!("{}.{}", header_b64, payload_b64);

    // Sign with Ed25519
    let signature = signing_key.sign(signing_input.as_bytes());
    let signature_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());

    // Combine into JWT
    let jwt = format!("{}.{}", signing_input, signature_b64);

    // Add prefix
    let prefix = if claims.admin { ADMIN_TOKEN_PREFIX } else { TOKEN_PREFIX };
    format!("{}{}", prefix, jwt)
}

/// Decode and verify a JWT token
///
/// Returns the claims if the token is valid and not expired.
pub fn decode(token: &str, verifying_key: &VerifyingKey) -> Result<Claims, JwtError> {
    // Strip prefix
    let jwt = if let Some(rest) = token.strip_prefix(ADMIN_TOKEN_PREFIX) {
        rest
    } else if let Some(rest) = token.strip_prefix(TOKEN_PREFIX) {
        rest
    } else {
        return Err(JwtError::InvalidPrefix);
    };

    // Split into parts
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return Err(JwtError::InvalidFormat);
    }

    let header_b64 = parts[0];
    let payload_b64 = parts[1];
    let signature_b64 = parts[2];

    // Verify signature first
    let signing_input = format!("{}.{}", header_b64, payload_b64);
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
pub fn decode_unverified(token: &str) -> Result<Claims, JwtError> {
    // Strip prefix
    let jwt = if let Some(rest) = token.strip_prefix(ADMIN_TOKEN_PREFIX) {
        rest
    } else if let Some(rest) = token.strip_prefix(TOKEN_PREFIX) {
        rest
    } else {
        return Err(JwtError::InvalidPrefix);
    };

    // Split into parts
    let parts: Vec<&str> = jwt.split('.').collect();
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

/// Check if a token string has a valid prefix
pub fn has_valid_prefix(token: &str) -> bool {
    token.starts_with(TOKEN_PREFIX)
}

/// Check if a token is an admin token (by prefix)
pub fn is_admin_token(token: &str) -> bool {
    token.starts_with(ADMIN_TOKEN_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn generate_keypair() -> (SigningKey, VerifyingKey) {
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let verifying_key = signing_key.verifying_key();
        (signing_key, verifying_key)
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let (signing_key, verifying_key) = generate_keypair();

        let claims = Claims::new(
            "alice",
            Duration::hours(1),
            vec!["model:*".to_string()],
            false,
        );

        let token = encode(&claims, &signing_key);
        assert!(token.starts_with(TOKEN_PREFIX));

        let decoded = decode(&token, &verifying_key).unwrap();
        assert_eq!(decoded.sub, "alice");
        assert_eq!(decoded.scope, vec!["model:*"]);
        assert!(!decoded.admin);
    }

    #[test]
    fn test_admin_token_prefix() {
        let (signing_key, verifying_key) = generate_keypair();

        let claims = Claims::new("admin", Duration::hours(1), vec![], true);

        let token = encode(&claims, &signing_key);
        assert!(token.starts_with(ADMIN_TOKEN_PREFIX));
        assert!(is_admin_token(&token));

        let decoded = decode(&token, &verifying_key).unwrap();
        assert!(decoded.admin);
    }

    #[test]
    fn test_expired_token() {
        let (signing_key, verifying_key) = generate_keypair();

        let mut claims = Claims::new("alice", Duration::hours(1), vec![], false);
        claims.exp = Utc::now().timestamp() - 3600; // Expired 1 hour ago

        let token = encode(&claims, &signing_key);
        let result = decode(&token, &verifying_key);
        assert!(matches!(result, Err(JwtError::Expired)));
    }

    #[test]
    fn test_invalid_signature() {
        let (signing_key, _) = generate_keypair();
        let (_, wrong_verifying_key) = generate_keypair();

        let claims = Claims::new("alice", Duration::hours(1), vec![], false);
        let token = encode(&claims, &signing_key);

        let result = decode(&token, &wrong_verifying_key);
        assert!(matches!(result, Err(JwtError::InvalidSignature)));
    }

    #[test]
    fn test_tampered_payload() {
        let (signing_key, verifying_key) = generate_keypair();

        let claims = Claims::new("alice", Duration::hours(1), vec![], false);
        let token = encode(&claims, &signing_key);

        // Tamper with the token
        let mut parts: Vec<&str> = token.split('.').collect();
        let tampered_payload = URL_SAFE_NO_PAD.encode(r#"{"sub":"bob","exp":9999999999,"iat":0}"#);

        // Skip prefix when rebuilding
        let prefix = if token.starts_with(ADMIN_TOKEN_PREFIX) {
            ADMIN_TOKEN_PREFIX
        } else {
            TOKEN_PREFIX
        };
        let jwt_start = prefix.len();
        let jwt_parts: Vec<&str> = token[jwt_start..].split('.').collect();

        let tampered_token = format!(
            "{}{}.{}.{}",
            prefix, jwt_parts[0], tampered_payload, jwt_parts[2]
        );

        let result = decode(&tampered_token, &verifying_key);
        assert!(matches!(result, Err(JwtError::InvalidSignature)));
    }

    #[test]
    fn test_scope_matching() {
        let claims = Claims {
            sub: "alice".to_string(),
            exp: Utc::now().timestamp() + 3600,
            iat: Utc::now().timestamp(),
            scope: vec!["model:qwen*".to_string(), "data:analytics".to_string()],
            admin: false,
        };

        assert!(claims.has_scope("model:qwen-7b"));
        assert!(claims.has_scope("model:qwen-large"));
        assert!(claims.has_scope("data:analytics"));
        assert!(!claims.has_scope("model:llama"));
        assert!(!claims.has_scope("data:logs"));
    }

    #[test]
    fn test_wildcard_scope() {
        let claims = Claims {
            sub: "admin".to_string(),
            exp: Utc::now().timestamp() + 3600,
            iat: Utc::now().timestamp(),
            scope: vec!["*".to_string()],
            admin: true,
        };

        assert!(claims.has_scope("anything"));
        assert!(claims.has_scope("model:whatever"));
    }

    #[test]
    fn test_empty_scope_allows_all() {
        let claims = Claims {
            sub: "alice".to_string(),
            exp: Utc::now().timestamp() + 3600,
            iat: Utc::now().timestamp(),
            scope: vec![],
            admin: false,
        };

        assert!(claims.has_scope("anything"));
    }

    #[test]
    fn test_decode_unverified() {
        let (signing_key, _) = generate_keypair();

        let claims = Claims::new("alice", Duration::hours(1), vec!["test".to_string()], false);
        let token = encode(&claims, &signing_key);

        let decoded = decode_unverified(&token).unwrap();
        assert_eq!(decoded.sub, "alice");
    }
}
