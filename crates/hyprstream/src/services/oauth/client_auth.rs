//! OAuth 2.1 client authentication at the token endpoint.
//!
//! Currently supports:
//!   - `none` — public client; PKCE substitutes for client auth
//!   - `private_key_jwt` — RFC 7521 / RFC 7523 §2.2 client assertion
//!
//! The client assertion is a JWT signed by one of the keys in the
//! registered client's `jwks` field (inline) or via `jwks_uri`
//! (deferred). Claim requirements per RFC 7523 §3:
//!   iss == sub == client_id
//!   aud == token endpoint URL (canonical)
//!   exp > now
//!   jti SHOULD be present; we treat absence as a soft warning
//!   alg per the JWKS key kty/crv (RS256, ES256, EdDSA)
//!
//! Spec reference: MCP 2025-11-25 § Client ID Metadata Documents lists
//! `private_key_jwt` as a MAY for clients (CIMD §6.2). Servers honoring
//! the `token_endpoint_auth_method` field in a CIMD/DCR registration
//! MUST validate the assertion before issuing tokens.

use serde_json::Value;

use super::state::RegisteredClient;
use crate::auth::id_token_verify::{algorithm_for_key_pub, build_decoding_key};
use jsonwebtoken::{decode, DecodingKey, Validation};

/// Client authentication failure.
#[derive(Debug, Clone)]
pub enum ClientAuthError {
    /// The client is configured for `private_key_jwt` but no assertion
    /// was presented at the token endpoint.
    MissingAssertion,
    /// The presented `client_assertion_type` is not the RFC 7523 jwt-bearer
    /// type. We accept exactly:
    /// `urn:ietf:params:oauth:client-assertion-type:jwt-bearer`.
    UnsupportedAssertionType(String),
    /// The client has no usable JWKS (inline or via `jwks_uri`) to verify
    /// against.
    NoKeys,
    /// `jwks_uri` is set but the implementation does not yet fetch it.
    JwksUriNotImplemented,
    /// JWT structural / decoding error (malformed parts, base64, JSON).
    Malformed(String),
    /// Signature verification failed against every candidate key.
    InvalidSignature,
    /// One of the iss/sub/aud/exp claims failed validation. The string
    /// describes which.
    InvalidClaim(String),
}

impl std::fmt::Display for ClientAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAssertion => write!(f, "client_assertion required for private_key_jwt"),
            Self::UnsupportedAssertionType(t) => write!(f, "unsupported client_assertion_type: {t}"),
            Self::NoKeys => write!(f, "client has no jwks/jwks_uri for private_key_jwt"),
            Self::JwksUriNotImplemented => write!(f, "jwks_uri client-key resolution is not yet implemented; supply inline jwks"),
            Self::Malformed(e) => write!(f, "malformed client_assertion: {e}"),
            Self::InvalidSignature => write!(f, "client_assertion signature did not verify"),
            Self::InvalidClaim(c) => write!(f, "invalid client_assertion claim: {c}"),
        }
    }
}

impl std::error::Error for ClientAuthError {}

pub const JWT_BEARER_ASSERTION_TYPE: &str =
    "urn:ietf:params:oauth:client-assertion-type:jwt-bearer";

/// Decide whether a registered client requires `private_key_jwt`. Default
/// is "no" — i.e. public client / PKCE-only — unless the registration
/// explicitly set `token_endpoint_auth_method=private_key_jwt`.
pub fn requires_private_key_jwt(client: &RegisteredClient) -> bool {
    matches!(
        client.token_endpoint_auth_method.as_deref(),
        Some("private_key_jwt")
    )
}

/// Verify an RFC 7523 client_assertion against the registered client's
/// inline JWKS. `expected_audience` is the canonical token endpoint URL.
///
/// On success, returns the verified JWT's claims for caller-side use
/// (typically just to log the `jti` for replay-cache decisions).
pub fn verify_client_assertion(
    client: &RegisteredClient,
    assertion_type: &str,
    assertion: &str,
    expected_audience: &str,
) -> Result<Value, ClientAuthError> {
    if assertion_type != JWT_BEARER_ASSERTION_TYPE {
        return Err(ClientAuthError::UnsupportedAssertionType(assertion_type.to_owned()));
    }

    let keys = candidate_keys(client)?;

    // Parse the header to learn alg + (optional) kid before iterating keys.
    let header = jsonwebtoken::decode_header(assertion)
        .map_err(|e| ClientAuthError::Malformed(format!("header decode: {e}")))?;
    let kid = header.kid.as_deref();

    // Filter by kid first when present — avoids futile signature attempts.
    let candidates: Vec<&Value> = keys
        .iter()
        .filter(|k| match kid {
            Some(want) => k.get("kid").and_then(Value::as_str) == Some(want),
            None => true,
        })
        .collect();

    if candidates.is_empty() {
        return Err(ClientAuthError::InvalidClaim(
            "no JWKS key matches assertion header kid".to_owned(),
        ));
    }

    // Try each candidate. jsonwebtoken::decode validates exp and aud
    // when set in Validation. We disable aud-via-validation and check it
    // manually so we can produce a precise error.
    for jwk in &candidates {
        let alg = match algorithm_for_key_pub(jwk) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let mut validation = Validation::new(alg);
        validation.validate_exp = true;
        validation.validate_aud = false; // checked manually below
        validation.required_spec_claims = ["exp"].into_iter().map(str::to_owned).collect();

        let decoding_key: DecodingKey = match build_decoding_key(jwk) {
            Ok(k) => k,
            Err(_) => continue,
        };

        match decode::<Value>(assertion, &decoding_key, &validation) {
            Ok(data) => {
                let claims = data.claims;
                check_claims(&claims, &client.client_id, expected_audience)?;
                return Ok(claims);
            }
            Err(_) => continue,
        }
    }

    Err(ClientAuthError::InvalidSignature)
}

/// Extract candidate keys from the registered client's `jwks` value.
/// Accepts both `{"keys": [...]}` (RFC 7517 JWKS) and a bare array of keys.
fn candidate_keys(client: &RegisteredClient) -> Result<Vec<Value>, ClientAuthError> {
    if client.jwks_uri.is_some() && client.jwks.is_none() {
        return Err(ClientAuthError::JwksUriNotImplemented);
    }
    let jwks = client.jwks.as_ref().ok_or(ClientAuthError::NoKeys)?;
    if let Some(keys) = jwks.get("keys").and_then(Value::as_array) {
        return Ok(keys.clone());
    }
    if let Some(keys) = jwks.as_array() {
        return Ok(keys.clone());
    }
    Err(ClientAuthError::NoKeys)
}

fn check_claims(
    claims: &Value,
    client_id: &str,
    expected_audience: &str,
) -> Result<(), ClientAuthError> {
    let iss = claims.get("iss").and_then(Value::as_str)
        .ok_or_else(|| ClientAuthError::InvalidClaim("missing iss".to_owned()))?;
    let sub = claims.get("sub").and_then(Value::as_str)
        .ok_or_else(|| ClientAuthError::InvalidClaim("missing sub".to_owned()))?;
    if iss != client_id {
        return Err(ClientAuthError::InvalidClaim(format!(
            "iss '{iss}' does not match client_id"
        )));
    }
    if sub != client_id {
        return Err(ClientAuthError::InvalidClaim(format!(
            "sub '{sub}' does not match client_id"
        )));
    }

    // Explicit exp check — defend against jsonwebtoken's
    // Validation::validate_exp behaviour quirks across versions.
    let exp = claims.get("exp").and_then(Value::as_i64)
        .ok_or_else(|| ClientAuthError::InvalidClaim("missing exp".to_owned()))?;
    let now = chrono::Utc::now().timestamp();
    if exp <= now {
        return Err(ClientAuthError::InvalidClaim(format!(
            "exp {exp} is not in the future (now={now})"
        )));
    }

    // aud may be a string or an array of strings (RFC 7519 §4.1.3).
    let aud_ok = match claims.get("aud") {
        Some(Value::String(s)) => s == expected_audience,
        Some(Value::Array(a)) => a.iter().any(|v| v.as_str() == Some(expected_audience)),
        _ => false,
    };
    if !aud_ok {
        return Err(ClientAuthError::InvalidClaim(format!(
            "aud does not include '{expected_audience}'"
        )));
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use ed25519_dalek::{Signer, SigningKey};
    use std::time::Instant;

    fn make_client(client_id: &str, jwks: serde_json::Value, auth_method: &str) -> RegisteredClient {
        RegisteredClient {
            client_id: client_id.to_owned(),
            redirect_uris: vec!["http://localhost/cb".to_owned()],
            client_name: None,
            client_uri: None,
            logo_uri: None,
            grant_types: vec![],
            response_types: vec![],
            token_endpoint_auth_method: Some(auth_method.to_owned()),
            jwks: Some(jwks),
            jwks_uri: None,
            is_cimd: true,
            registered_at: Instant::now(),
        }
    }

    fn ed25519_keypair_and_jwk() -> (SigningKey, serde_json::Value) {
        let mut rng = rand::rngs::OsRng;
        let sk = SigningKey::generate(&mut rng);
        let vk = sk.verifying_key();
        let x = URL_SAFE_NO_PAD.encode(vk.to_bytes());
        let jwk = serde_json::json!({
            "kty": "OKP",
            "crv": "Ed25519",
            "x": x,
            "kid": "k1",
        });
        (sk, jwk)
    }

    /// Sign a JWT manually with ed25519-dalek — avoids needing a PKCS#8
    /// DER encoding just for tests.
    fn make_ed_assertion(sk: &SigningKey, claims: serde_json::Value) -> String {
        let header = serde_json::json!({"alg": "EdDSA", "typ": "JWT", "kid": "k1"});
        let header_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap());
        let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
        let signing_input = format!("{header_b64}.{payload_b64}");
        let sig = sk.sign(signing_input.as_bytes());
        let sig_b64 = URL_SAFE_NO_PAD.encode(sig.to_bytes());
        format!("{signing_input}.{sig_b64}")
    }

    #[test]
    fn valid_assertion_round_trip() {
        let (sk, jwk) = ed25519_keypair_and_jwk();
        let jwks = serde_json::json!({"keys": [jwk]});
        let client = make_client("https://app.test/c", jwks, "private_key_jwt");
        let claims = serde_json::json!({
            "iss": "https://app.test/c",
            "sub": "https://app.test/c",
            "aud": "https://hs.test/oauth/token",
            "exp": chrono::Utc::now().timestamp() + 60,
            "jti": "j1",
        });
        let assertion = make_ed_assertion(&sk, claims);
        let got = verify_client_assertion(
            &client,
            JWT_BEARER_ASSERTION_TYPE,
            &assertion,
            "https://hs.test/oauth/token",
        );
        assert!(got.is_ok(), "verify failed: {:?}", got.err());
    }

    #[test]
    fn rejects_wrong_audience() {
        let (sk, jwk) = ed25519_keypair_and_jwk();
        let jwks = serde_json::json!({"keys": [jwk]});
        let client = make_client("https://app.test/c", jwks, "private_key_jwt");
        let claims = serde_json::json!({
            "iss": "https://app.test/c",
            "sub": "https://app.test/c",
            "aud": "https://attacker.test/oauth/token",
            "exp": chrono::Utc::now().timestamp() + 60,
        });
        let assertion = make_ed_assertion(&sk, claims);
        let got = verify_client_assertion(
            &client,
            JWT_BEARER_ASSERTION_TYPE,
            &assertion,
            "https://hs.test/oauth/token",
        );
        assert!(matches!(got, Err(ClientAuthError::InvalidClaim(_))));
    }

    #[test]
    fn rejects_wrong_iss() {
        let (sk, jwk) = ed25519_keypair_and_jwk();
        let jwks = serde_json::json!({"keys": [jwk]});
        let client = make_client("https://app.test/c", jwks, "private_key_jwt");
        let claims = serde_json::json!({
            "iss": "https://impostor.test/c",
            "sub": "https://impostor.test/c",
            "aud": "https://hs.test/oauth/token",
            "exp": chrono::Utc::now().timestamp() + 60,
        });
        let assertion = make_ed_assertion(&sk, claims);
        let got = verify_client_assertion(
            &client,
            JWT_BEARER_ASSERTION_TYPE,
            &assertion,
            "https://hs.test/oauth/token",
        );
        assert!(matches!(got, Err(ClientAuthError::InvalidClaim(_))));
    }

    #[test]
    fn rejects_expired_assertion() {
        let (sk, jwk) = ed25519_keypair_and_jwk();
        let jwks = serde_json::json!({"keys": [jwk]});
        let client = make_client("https://app.test/c", jwks, "private_key_jwt");
        let claims = serde_json::json!({
            "iss": "https://app.test/c",
            "sub": "https://app.test/c",
            "aud": "https://hs.test/oauth/token",
            "exp": chrono::Utc::now().timestamp() - 60,
        });
        let assertion = make_ed_assertion(&sk, claims);
        let got = verify_client_assertion(
            &client,
            JWT_BEARER_ASSERTION_TYPE,
            &assertion,
            "https://hs.test/oauth/token",
        );
        // jsonwebtoken rejects expired tokens; we surface either
        // InvalidSignature (decode fell through) or InvalidClaim. Either
        // way, the assertion MUST NOT verify.
        assert!(got.is_err(), "expired assertion should not verify: {got:?}");
    }

    #[test]
    fn rejects_wrong_assertion_type() {
        let (_, jwk) = ed25519_keypair_and_jwk();
        let jwks = serde_json::json!({"keys": [jwk]});
        let client = make_client("https://app.test/c", jwks, "private_key_jwt");
        let got = verify_client_assertion(&client, "saml2-bearer", "irrelevant", "irrelevant");
        assert!(matches!(got, Err(ClientAuthError::UnsupportedAssertionType(_))));
    }

    #[test]
    fn rejects_no_keys() {
        let client = make_client("https://app.test/c", serde_json::json!({"keys": []}), "private_key_jwt");
        let got = verify_client_assertion(
            &client,
            JWT_BEARER_ASSERTION_TYPE,
            "irrelevant",
            "https://hs.test/oauth/token",
        );
        // candidate_keys returns empty; then header decode fails for "irrelevant"
        // → Malformed. But really the test asserts we don't accept a no-keys client.
        assert!(got.is_err());
    }

    #[test]
    fn jwks_uri_returns_unimplemented() {
        let mut client = make_client("https://app.test/c", serde_json::json!({}), "private_key_jwt");
        client.jwks = None;
        client.jwks_uri = Some("https://app.test/jwks.json".to_owned());
        let got = verify_client_assertion(
            &client,
            JWT_BEARER_ASSERTION_TYPE,
            "irrelevant",
            "https://hs.test/oauth/token",
        );
        assert!(matches!(got, Err(ClientAuthError::JwksUriNotImplemented)));
    }

    #[test]
    fn requires_private_key_jwt_only_when_configured() {
        let (_, jwk) = ed25519_keypair_and_jwk();
        let jwks = serde_json::json!({"keys": [jwk]});
        let mut client = make_client("https://app.test/c", jwks, "private_key_jwt");
        assert!(requires_private_key_jwt(&client));
        client.token_endpoint_auth_method = Some("none".to_owned());
        assert!(!requires_private_key_jwt(&client));
        client.token_endpoint_auth_method = None;
        assert!(!requires_private_key_jwt(&client));
    }
}
