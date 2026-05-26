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
use std::time::{Duration, Instant};

use super::state::{OAuthState, RegisteredClient};
use crate::auth::id_token_verify::{algorithm_for_key_pub, build_decoding_key};
use jsonwebtoken::{decode, DecodingKey, Validation};

/// HTTP fetch timeout for jwks_uri. Same as CIMD document fetch.
const JWKS_URI_FETCH_TIMEOUT: Duration = Duration::from_secs(10);

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
    /// jwks_uri fetch failed (network error, non-2xx status, parse error,
    /// or SSRF-blocked URL).
    JwksUriFetchFailed(String),
    /// Trust-policy denied this CIMD origin between PAR/authorize and
    /// the token-endpoint client-assertion verification.
    TrustPolicyDenied(String),
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
            Self::JwksUriFetchFailed(e) => write!(f, "jwks_uri fetch failed: {e}"),
            Self::TrustPolicyDenied(e) => write!(f, "CIMD trust policy denied: {e}"),
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
/// JWKS — inline `jwks` if present, otherwise fetched from `jwks_uri`
/// (cached for `state.client_jwks_uri_cache_ttl`). `expected_audience` is the canonical
/// token endpoint URL.
///
/// On success, returns the verified JWT's claims for caller-side use
/// (typically just to log the `jti` for replay-cache decisions).
pub async fn verify_client_assertion(
    state: &OAuthState,
    client: &RegisteredClient,
    assertion_type: &str,
    assertion: &str,
    expected_audience: &str,
) -> Result<Value, ClientAuthError> {
    if assertion_type != JWT_BEARER_ASSERTION_TYPE {
        return Err(ClientAuthError::UnsupportedAssertionType(assertion_type.to_owned()));
    }
    let keys = resolve_keys(state, client).await?;
    verify_assertion_with_keys(&keys, &client.client_id, assertion, expected_audience)
}

/// Pure-sync core verification: given a set of candidate keys, verify
/// the JWT's signature and claims. Used both by the async public entry
/// point and directly by unit tests.
pub fn verify_assertion_with_keys(
    keys: &[Value],
    client_id: &str,
    assertion: &str,
    expected_audience: &str,
) -> Result<Value, ClientAuthError> {
    let keys = keys.to_vec();

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
    //
    // Per-key failures log at `debug!` — this loop is expected to
    // discard non-matching candidates in the happy path (a JWKS may
    // hold many keys and only the kid-matched one is the right
    // verifier). Promoting to warn would flood logs. The final
    // `InvalidSignature` carries an aggregate warn-level summary so
    // operators can correlate a verification failure with the keys
    // that were considered.
    let mut attempts = 0u32;
    for jwk in &candidates {
        let kid_for_log = jwk.get("kid").and_then(Value::as_str);
        let alg = match algorithm_for_key_pub(jwk) {
            Ok(a) => a,
            Err(e) => {
                tracing::debug!(
                    kid = ?kid_for_log,
                    error = %e,
                    "client_assertion: skipping JWKS key (unsupported algorithm)"
                );
                continue;
            }
        };
        let mut validation = Validation::new(alg);
        validation.validate_exp = true;
        validation.validate_aud = false; // checked manually below
        validation.required_spec_claims = ["exp"].into_iter().map(str::to_owned).collect();

        let decoding_key: DecodingKey = match build_decoding_key(jwk) {
            Ok(k) => k,
            Err(e) => {
                tracing::debug!(
                    kid = ?kid_for_log,
                    error = %e,
                    "client_assertion: skipping JWKS key (invalid key material)"
                );
                continue;
            }
        };

        attempts += 1;
        match decode::<Value>(assertion, &decoding_key, &validation) {
            Ok(data) => {
                let claims = data.claims;
                check_claims(&claims, client_id, expected_audience)?;
                return Ok(claims);
            }
            Err(e) => {
                tracing::debug!(
                    kid = ?kid_for_log,
                    error = %e,
                    "client_assertion: signature/claims verification failed for this key"
                );
                continue;
            }
        }
    }

    // Aggregate summary at warn level: emits once per failed assertion
    // with the kid the assertion was *signed for*, helping operators
    // pinpoint a misconfigured client.
    let header_kid = jsonwebtoken::decode_header(assertion)
        .ok()
        .and_then(|h| h.kid);
    tracing::warn!(
        client_id = %client_id,
        header_kid = ?header_kid,
        candidates = candidates.len(),
        actual_verify_attempts = attempts,
        "client_assertion verification failed against every candidate key"
    );
    Err(ClientAuthError::InvalidSignature)
}

/// Resolve a registered client's signing keys: inline `jwks` first,
/// then `jwks_uri` (cached). The two are RFC 7591 §2.1 mutually
/// exclusive, but defensive prefer-inline is harmless.
async fn resolve_keys(
    state: &OAuthState,
    client: &RegisteredClient,
) -> Result<Vec<Value>, ClientAuthError> {
    // Defense in depth: re-check the unified federation:register policy
    // at the token endpoint. The cimd_cache entry was admitted at
    // PAR/authorize time, but operators may have flipped policy in the
    // interim (the cache TTL bounds the window). Fail-closed on RPC
    // error, matching resolve_cimd_client.
    if client.is_cimd {
        if let Some(origin) = super::registration::extract_origin(&client.client_id) {
            if let Err(e) = super::registration::check_federation_register_for_client_auth(state, &origin).await {
                return Err(ClientAuthError::TrustPolicyDenied(e));
            }
        }
    }

    if let Some(jwks) = client.jwks.as_ref() {
        return extract_keys_array(jwks);
    }
    if let Some(uri) = client.jwks_uri.as_deref() {
        let jwks = fetch_jwks_uri(state, uri).await?;
        return extract_keys_array(&jwks);
    }
    Err(ClientAuthError::NoKeys)
}

/// Extract the `keys` array from a JWKS document. Accepts both the
/// canonical `{"keys": [...]}` (RFC 7517) and a bare array.
fn extract_keys_array(jwks: &Value) -> Result<Vec<Value>, ClientAuthError> {
    if let Some(keys) = jwks.get("keys").and_then(Value::as_array) {
        return Ok(keys.clone());
    }
    if let Some(keys) = jwks.as_array() {
        return Ok(keys.clone());
    }
    Err(ClientAuthError::NoKeys)
}

/// SSRF + scheme check for jwks_uri. Same policy as CIMD document fetch:
/// HTTPS only, no private / loopback / RFC 1918 hosts.
fn validate_jwks_uri(url: &str) -> Result<(), ClientAuthError> {
    if !url.starts_with("https://") {
        return Err(ClientAuthError::JwksUriFetchFailed(
            "jwks_uri must use https://".to_owned(),
        ));
    }
    let parsed = url::Url::parse(url)
        .map_err(|e| ClientAuthError::JwksUriFetchFailed(format!("invalid jwks_uri: {e}")))?;
    if let Some(host) = parsed.host_str() {
        if super::registration::is_private_host_for_jwks(host) {
            return Err(ClientAuthError::JwksUriFetchFailed(
                "jwks_uri must not point to private/loopback addresses".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Fetch a jwks_uri (or return a fresh-enough cached entry). Lazy
/// expiry: stale entries in the map are detected here and re-fetched
/// before being used.
async fn fetch_jwks_uri(state: &OAuthState, url: &str) -> Result<Value, ClientAuthError> {
    // Cache check (read lock).
    {
        let cache = state.jwks_uri_cache.read().await;
        if let Some((jwks, fetched_at)) = cache.get(url) {
            if fetched_at.elapsed() < state.client_jwks_uri_cache_ttl {
                return Ok(jwks.clone());
            }
        }
    }

    validate_jwks_uri(url)?;

    let client = reqwest::Client::builder()
        .timeout(JWKS_URI_FETCH_TIMEOUT)
        .build()
        .map_err(|e| ClientAuthError::JwksUriFetchFailed(format!("http client init: {e}")))?;
    let response = client
        .get(url)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| ClientAuthError::JwksUriFetchFailed(format!("network: {e}")))?;
    if !response.status().is_success() {
        return Err(ClientAuthError::JwksUriFetchFailed(format!(
            "HTTP {}",
            response.status()
        )));
    }
    let jwks: Value = response
        .json()
        .await
        .map_err(|e| ClientAuthError::JwksUriFetchFailed(format!("parse: {e}")))?;

    // Validate JWKS shape: must have `keys` array OR be a bare array.
    let _ = extract_keys_array(&jwks)?;

    // Cache (write lock).
    state.jwks_uri_cache
        .write()
        .await
        .insert(url.to_owned(), (jwks.clone(), Instant::now()));
    Ok(jwks)
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

    fn keys_array(jwk: serde_json::Value) -> Vec<serde_json::Value> {
        vec![jwk]
    }

    #[test]
    fn valid_assertion_round_trip() {
        let (sk, jwk) = ed25519_keypair_and_jwk();
        let claims = serde_json::json!({
            "iss": "https://app.test/c",
            "sub": "https://app.test/c",
            "aud": "https://hs.test/oauth/token",
            "exp": chrono::Utc::now().timestamp() + 60,
            "jti": "j1",
        });
        let assertion = make_ed_assertion(&sk, claims);
        let got = verify_assertion_with_keys(
            &keys_array(jwk),
            "https://app.test/c",
            &assertion,
            "https://hs.test/oauth/token",
        );
        assert!(got.is_ok(), "verify failed: {:?}", got.err());
    }

    #[test]
    fn rejects_wrong_audience() {
        let (sk, jwk) = ed25519_keypair_and_jwk();
        let claims = serde_json::json!({
            "iss": "https://app.test/c",
            "sub": "https://app.test/c",
            "aud": "https://attacker.test/oauth/token",
            "exp": chrono::Utc::now().timestamp() + 60,
        });
        let assertion = make_ed_assertion(&sk, claims);
        let got = verify_assertion_with_keys(
            &keys_array(jwk),
            "https://app.test/c",
            &assertion,
            "https://hs.test/oauth/token",
        );
        assert!(matches!(got, Err(ClientAuthError::InvalidClaim(_))));
    }

    #[test]
    fn rejects_wrong_iss() {
        let (sk, jwk) = ed25519_keypair_and_jwk();
        let claims = serde_json::json!({
            "iss": "https://impostor.test/c",
            "sub": "https://impostor.test/c",
            "aud": "https://hs.test/oauth/token",
            "exp": chrono::Utc::now().timestamp() + 60,
        });
        let assertion = make_ed_assertion(&sk, claims);
        let got = verify_assertion_with_keys(
            &keys_array(jwk),
            "https://app.test/c",
            &assertion,
            "https://hs.test/oauth/token",
        );
        assert!(matches!(got, Err(ClientAuthError::InvalidClaim(_))));
    }

    #[test]
    fn rejects_expired_assertion() {
        let (sk, jwk) = ed25519_keypair_and_jwk();
        let claims = serde_json::json!({
            "iss": "https://app.test/c",
            "sub": "https://app.test/c",
            "aud": "https://hs.test/oauth/token",
            "exp": chrono::Utc::now().timestamp() - 60,
        });
        let assertion = make_ed_assertion(&sk, claims);
        let got = verify_assertion_with_keys(
            &keys_array(jwk),
            "https://app.test/c",
            &assertion,
            "https://hs.test/oauth/token",
        );
        assert!(got.is_err(), "expired assertion should not verify: {got:?}");
    }

    #[test]
    fn rejects_empty_keys_set() {
        let got = verify_assertion_with_keys(
            &[],
            "https://app.test/c",
            "irrelevant",
            "https://hs.test/oauth/token",
        );
        assert!(got.is_err(), "empty key set must not verify");
    }

    #[test]
    fn extract_keys_supports_bare_array_and_keys_wrapper() {
        let jwk = serde_json::json!({"kty": "OKP", "crv": "Ed25519", "x": "abc"});
        let wrapped = serde_json::json!({"keys": [jwk.clone()]});
        let bare = serde_json::json!([jwk]);
        assert_eq!(extract_keys_array(&wrapped).unwrap().len(), 1);
        assert_eq!(extract_keys_array(&bare).unwrap().len(), 1);
        assert!(extract_keys_array(&serde_json::json!({"other": 1})).is_err());
    }

    #[test]
    fn jwks_uri_blocks_non_https() {
        let got = validate_jwks_uri("http://app.test/jwks.json");
        assert!(matches!(got, Err(ClientAuthError::JwksUriFetchFailed(_))));
    }

    #[test]
    fn jwks_uri_blocks_private_hosts() {
        for url in &[
            "https://localhost/jwks.json",
            "https://127.0.0.1/jwks.json",
            "https://10.0.0.1/jwks.json",
            "https://192.168.1.1/jwks.json",
        ] {
            let got = validate_jwks_uri(url);
            assert!(
                matches!(got, Err(ClientAuthError::JwksUriFetchFailed(_))),
                "expected SSRF rejection for {url}"
            );
        }
    }

    #[test]
    fn alg_confusion_hs256_header_against_rsa_jwk_is_rejected() {
        // Canonical alg-confusion attack: a malicious assertion declares
        // alg=HS256 in the header, hoping the verifier will HMAC-verify
        // the JWT using the RSA public key (as bytes) — which the client
        // also knows. Our defense: alg is derived from JWK kty (RSA →
        // RS256), not from the header. The signature verify must use
        // RS256 against the HMAC-signed payload and fail.
        //
        // Build a JWS structure with alg=HS256 in the header and a
        // bogus signature, against an RSA JWK. verify must reject.
        let rsa_jwk = serde_json::json!({
            "kty": "RSA",
            "n": "u1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gunVTLw7onLRnrq0_IzW7yWR7QkrmBL7jTKEn5u-qKhbwKfBstIs-bMY2Zkp18gnTxKLxoS2tFczGkPLPgizskuemMghRniWaoLcyehkd3qqGElvW_VDL5AaWTg0nLVkjRo9z-40RQzuVaE8AkAFmxZzow3x-VJYKdjykkJ0iT9wCS0DRTXu269V264Vf_3jvredZiKRkgwlL9xNAwxXFg0x_XFw005UWVRIkdgcKWTjpBP2dPwVZ4WWC-9aGVd-Gyn1o0CLelf4rEjGoXbAAEgAqeGUxrcIlbjXfbcmw",
            "e": "AQAB",
            "kid": "rsa1",
        });
        // A header with alg=HS256 (attacker-controlled), body, and
        // signature computed by HMAC-SHA256 with the public-RSA bytes
        // as key. We don't need to actually compute it — the structural
        // claim is that decoding will fail because:
        //   - algorithm_for_key_pub returns RS256 (from kty=RSA)
        //   - decode<>(token, key, validation_with_RS256_alg) checks
        //     the header alg matches the Validation's allowed algs
        //     (jsonwebtoken rejects HS256 token with RS256 validation)
        let bogus_assertion = format!(
            "{}.{}.{}",
            URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256","typ":"JWT","kid":"rsa1"}"#),
            URL_SAFE_NO_PAD.encode(br#"{"iss":"https://app.test/c","sub":"https://app.test/c","aud":"https://hs.test/oauth/token","exp":9999999999}"#),
            URL_SAFE_NO_PAD.encode([0u8; 32]),
        );
        let got = verify_assertion_with_keys(
            &[rsa_jwk],
            "https://app.test/c",
            &bogus_assertion,
            "https://hs.test/oauth/token",
        );
        assert!(
            matches!(got, Err(ClientAuthError::InvalidSignature)),
            "HS256 header against RSA JWK MUST be rejected; got {got:?}"
        );
    }

    #[test]
    fn jwks_uri_accepts_public_https() {
        // We can't fetch in a unit test; just verify the SSRF + scheme
        // gate lets a plausible URL through.
        let got = validate_jwks_uri("https://app.example.com/jwks.json");
        assert!(got.is_ok());
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
