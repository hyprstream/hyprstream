//! External IdP id_token verification with multi-algorithm JWKS support.
//!
//! Verifies JWT signatures from external OIDC providers (Keycloak, Auth0, etc.)
//! using keys fetched from the provider's JWKS endpoint. Supports RS256, ES256,
//! and EdDSA (Ed25519) algorithms.

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use jsonwebtoken::{DecodingKey, Validation, Algorithm};
use serde_json::Value;

/// Verification result for an external id_token.
#[derive(Debug)]
pub struct VerifiedIdToken {
    /// Decoded claims from the verified id_token.
    pub claims: Value,
}

/// Verify an external id_token against the provider's JWKS.
///
/// Fetches the JWKS from the given URI, selects the appropriate key (by `kid`
/// if present in the JWT header), and verifies the signature. Returns the
/// decoded claims on success.
///
/// # Arguments
///
/// * `token` — The raw JWT string (id_token from the external provider).
/// * `jwks_uri` — The provider's JWKS endpoint URL.
/// * `expected_issuer` — The expected `iss` claim value (provider's issuer URL).
/// * `expected_audience` — The expected `aud` claim value (client_id).
/// * `http` — HTTP client for fetching the JWKS.
pub async fn verify_id_token(
    token: &str,
    jwks_uri: &str,
    expected_issuer: &str,
    expected_audience: &str,
    http: &reqwest::Client,
) -> Result<VerifiedIdToken> {
    // Fetch JWKS (256KB response limit to prevent resource exhaustion)
    let jwks_response = http
        .get(jwks_uri)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .with_context(|| format!("Failed to fetch JWKS from {}", jwks_uri))?;

    if !jwks_response.status().is_success() {
        return Err(anyhow!(
            "JWKS endpoint returned {}: {}",
            jwks_response.status(),
            jwks_uri
        ));
    }

    let jwks_body = jwks_response
        .bytes()
        .await
        .context("Failed to read JWKS response body")?;
    if jwks_body.len() > 256 * 1024 {
        return Err(anyhow!("JWKS response too large: {} bytes (max 256KB)", jwks_body.len()));
    }

    let jwks_json: Value = serde_json::from_slice(&jwks_body)
        .context("Invalid JWKS JSON")?;

    let keys = jwks_json["keys"]
        .as_array()
        .ok_or_else(|| anyhow!("JWKS missing 'keys' array at {}", jwks_uri))?;

    // Extract the JWT header to get the `kid` for key selection
    let header = decode_jwt_header(token)?;

    let kid = header.get("kid").and_then(|v| v.as_str());

    // Find the matching key from JWKS
    let matching_key = find_matching_key(keys, kid, &header)?;

    // Build a jsonwebtoken::Validation with the correct algorithm
    let alg = determine_algorithm(matching_key, &header)?;
    let mut validation = Validation::new(alg);
    validation.set_issuer(&[expected_issuer]);
    validation.set_audience(&[expected_audience]);
    // Allow 60 seconds of clock skew (same as provider.clock_skew_seconds default)
    validation.leeway = 60;

    // Build the DecodingKey from the JWKS key
    let decoding_key = build_decoding_key(matching_key)?;

    // Decode and verify
    let decoded = jsonwebtoken::decode::<Value>(token, &decoding_key, &validation)
        .map_err(|e| anyhow!("id_token signature verification failed: {}", e))?;

    Ok(VerifiedIdToken {
        claims: decoded.claims,
    })
}

/// Decode only the JWT header (without verification) for key selection.
fn decode_jwt_header(token: &str) -> Result<Value> {
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err(anyhow!("Invalid JWT format"));
    }
    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .map_err(|e| anyhow!("Invalid base64 in JWT header: {e}"))?;
    serde_json::from_slice(&header_bytes)
        .map_err(|e| anyhow!("Invalid JSON in JWT header: {e}"))
}

/// Find the matching JWKS key by `kid` (if present) or by algorithm.
fn find_matching_key<'a>(keys: &'a [Value], kid: Option<&str>, header: &Value) -> Result<&'a Value> {
    // If kid is present, match by kid first
    if let Some(kid) = kid {
        for key in keys {
            if key.get("kid").and_then(|v| v.as_str()) == Some(kid) {
                return Ok(key);
            }
        }
        return Err(anyhow!("No JWKS key found with kid='{}'", kid));
    }

    // No kid — try to match by algorithm from the header
    let header_alg = header.get("alg").and_then(|v| v.as_str()).unwrap_or("");
    for key in keys {
        let key_kty = key.get("kty").and_then(|v| v.as_str()).unwrap_or("");
        let matches = match header_alg {
            "RS256" => key_kty == "RSA",
            "ES256" => key_kty == "EC",
            "EdDSA" => key_kty == "OKP",
            _ => false,
        };
        if matches {
            return Ok(key);
        }
    }

    // Fallback: return first key
    keys.first()
        .ok_or_else(|| anyhow!("JWKS keys array is empty"))
}

/// Determine the Algorithm from the JWKS key and JWT header.
fn determine_algorithm(jwks_key: &Value, header: &Value) -> Result<Algorithm> {
    let header_alg = header.get("alg").and_then(|v| v.as_str());

    // Trust the JWT header's alg if it's one we support
    if let Some(alg_str) = header_alg {
        let alg = match alg_str {
            "RS256" => Algorithm::RS256,
            "ES256" => Algorithm::ES256,
            "EdDSA" => Algorithm::EdDSA,
            other => return Err(anyhow!("Unsupported JWT algorithm: {}", other)),
        };
        // Verify the JWKS key type is consistent
        let kty = jwks_key.get("kty").and_then(|v| v.as_str()).unwrap_or("");
        let expected_kty = match alg {
            Algorithm::RS256 => "RSA",
            Algorithm::ES256 => "EC",
            Algorithm::EdDSA => "OKP",
            _ => unreachable!(),
        };
        if kty != expected_kty {
            return Err(anyhow!(
                "JWT algorithm {} expects kty='{}' but JWKS key has kty='{}'",
                alg_str,
                expected_kty,
                kty
            ));
        }
        return Ok(alg);
    }

    // Infer from JWKS key type
    algorithm_for_key_pub(jwks_key)
}

/// Public alias for the algorithm-from-JWK helper. Used by client_auth.
pub fn algorithm_for_key_pub(jwks_key: &Value) -> Result<Algorithm> {
    let kty = jwks_key.get("kty").and_then(|v| v.as_str()).unwrap_or("");
    match kty {
        "RSA" => Ok(Algorithm::RS256),
        "EC" => {
            let crv = jwks_key.get("crv").and_then(|v| v.as_str()).unwrap_or("");
            match crv {
                "P-256" => Ok(Algorithm::ES256),
                other => Err(anyhow!("Unsupported EC curve: {}", other)),
            }
        }
        "OKP" => {
            let crv = jwks_key.get("crv").and_then(|v| v.as_str()).unwrap_or("");
            match crv {
                "Ed25519" => Ok(Algorithm::EdDSA),
                other => Err(anyhow!("Unsupported OKP curve: {}", other)),
            }
        }
        other => Err(anyhow!("Unsupported JWKS key type: {}", other)),
    }
}

/// Build a jsonwebtoken::DecodingKey from a JWKS key object.
pub fn build_decoding_key(jwks_key: &Value) -> Result<DecodingKey> {
    let kty = jwks_key.get("kty").and_then(|v| v.as_str()).unwrap_or("");

    match kty {
        "RSA" => {
            // RSA keys use 'n' and 'e' components (base64url-encoded per RFC 7518)
            let n = jwks_key
                .get("n")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("RSA JWKS key missing 'n' field"))?;
            let e = jwks_key
                .get("e")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("RSA JWKS key missing 'e' field"))?;
            DecodingKey::from_rsa_components(n, e)
                .map_err(|e| anyhow!("Invalid RSA key components: {e}"))
        }
        "EC" => {
            // EC keys use 'x' and 'y' components (P-256)
            let x = jwks_key
                .get("x")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("EC JWKS key missing 'x' field"))?;
            let y = jwks_key
                .get("y")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("EC JWKS key missing 'y' field"))?;
            DecodingKey::from_ec_components(x, y)
                .map_err(|e| anyhow!("Invalid EC key components: {e}"))
        }
        "OKP" => {
            // Ed25519 keys use 'x' (the public key)
            let x = jwks_key
                .get("x")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("OKP JWKS key missing 'x' field"))?;
            let raw = URL_SAFE_NO_PAD
                .decode(x)
                .map_err(|e| anyhow!("Invalid base64 in OKP 'x': {e}"))?;
            // from_ed_der stores raw bytes — the name is misleading but
            // equivalent to from_ed_components (both store SecretOrDer).
            Ok(DecodingKey::from_ed_der(&raw))
        }
        other => Err(anyhow!("Unsupported JWKS key type: {}", other)),
    }
}

/// Decode an external id_token without signature verification.
///
/// Used as a fallback when JWKS verification is not available (e.g., no
/// configured trusted issuer). The caller must validate claims manually.
pub fn decode_unverified(token: &str) -> Result<Value> {
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err(anyhow!("Invalid JWT format"));
    }
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|e| anyhow!("Invalid base64 in id_token payload: {e}"))?;
    serde_json::from_slice(&payload_bytes)
        .map_err(|e| anyhow!("Invalid JSON in id_token payload: {e}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_jwt_header_eddsa() {
        let header = serde_json::json!({"alg": "EdDSA", "typ": "JWT"});
        let header_b64 = URL_SAFE_NO_PAD.encode(header.to_string().as_bytes());
        let token = format!("{header_b64}.e30.Zm9v");
        let decoded = decode_jwt_header(&token).unwrap();
        assert_eq!(decoded["alg"], "EdDSA");
    }

    #[test]
    fn test_decode_jwt_header_rs256() {
        let header = serde_json::json!({"alg": "RS256", "typ": "JWT", "kid": "key-1"});
        let header_b64 = URL_SAFE_NO_PAD.encode(header.to_string().as_bytes());
        let token = format!("{header_b64}.e30.Zm9v");
        let decoded = decode_jwt_header(&token).unwrap();
        assert_eq!(decoded["alg"], "RS256");
        assert_eq!(decoded["kid"], "key-1");
    }

    #[test]
    fn test_determine_algorithm_from_header() {
        let key = serde_json::json!({"kty": "RSA", "n": "abc", "e": "AQAB"});
        let header = serde_json::json!({"alg": "RS256"});
        assert_eq!(determine_algorithm(&key, &header).unwrap(), Algorithm::RS256);
    }

    #[test]
    fn test_determine_algorithm_kty_mismatch() {
        let key = serde_json::json!({"kty": "EC", "x": "abc", "y": "def"});
        let header = serde_json::json!({"alg": "RS256"});
        assert!(determine_algorithm(&key, &header).is_err());
    }

    #[test]
    fn test_determine_algorithm_infer_from_kty() {
        let rsa_key = serde_json::json!({"kty": "RSA"});
        let empty_header = serde_json::json!({});
        assert_eq!(
            determine_algorithm(&rsa_key, &empty_header).unwrap(),
            Algorithm::RS256
        );
    }

    #[test]
    fn test_find_matching_key_by_kid() {
        let keys = vec![
            serde_json::json!({"kty": "RSA", "kid": "key-1", "n": "abc", "e": "AQAB"}),
            serde_json::json!({"kty": "RSA", "kid": "key-2", "n": "def", "e": "AQAB"}),
        ];
        let header = serde_json::json!({"alg": "RS256", "kid": "key-2"});
        let found = find_matching_key(&keys, Some("key-2"), &header).unwrap();
        assert_eq!(found["kid"], "key-2");
    }

    #[test]
    fn test_find_matching_key_no_kid_matches_alg() {
        let keys = vec![
            serde_json::json!({"kty": "RSA", "n": "abc", "e": "AQAB"}),
        ];
        let header = serde_json::json!({"alg": "RS256"});
        let found = find_matching_key(&keys, None, &header).unwrap();
        assert_eq!(found["kty"], "RSA");
    }

    #[test]
    fn test_decode_unverified_valid() {
        let payload = serde_json::json!({"sub": "user123", "iss": "https://example.com"});
        let payload_b64 = URL_SAFE_NO_PAD.encode(payload.to_string().as_bytes());
        let token = format!("e30.{}.Zm9v", payload_b64);
        let claims = decode_unverified(&token).unwrap();
        assert_eq!(claims["sub"], "user123");
    }

    #[test]
    fn test_decode_unverified_invalid_format() {
        assert!(decode_unverified("not.a.jwt").is_err());
        assert!(decode_unverified("onlytwo").is_err());
    }
}
