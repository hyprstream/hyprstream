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

    verify_id_token_with_keys(token, keys, expected_issuer, expected_audience)
}

/// Pure (HTTP-free) verification core: given the parsed JWKS `keys` array,
/// verify an id_token's signature and claims. Exposed for tests so the
/// overlap-rotation behaviour can be exercised without a mock server.
///
/// Selects every algorithm-compatible candidate and tries each until one
/// verifies — never collapsing the published set to a positional singleton
/// (#1183 / #1184).
pub(crate) fn verify_id_token_with_keys(
    token: &str,
    keys: &[Value],
    expected_issuer: &str,
    expected_audience: &str,
) -> Result<VerifiedIdToken> {
    // Extract the JWT header to get the `kid` for key selection
    let header = decode_jwt_header(token)?;

    let kid = header.get("kid").and_then(|v| v.as_str());

    // Collect every algorithm-compatible candidate. When the JWT carries a
    // `kid` the set is the kid-matched entries; when it does not, the set is
    // every key whose `kty`/`crv` matches the header `alg`. The verifier then
    // tries each until one verifies — never collapsing a published key SET to
    // a positional singleton, which would foreclose overlap rotation and
    // PQ-hybrid rollout (#1183 / #1184).
    let candidates = candidate_keys(keys, kid, &header);
    if candidates.is_empty() {
        return Err(anyhow!(
            "No JWKS key matches {} (kid={:?})",
            header.get("alg").and_then(|v| v.as_str()).unwrap_or(""),
            kid,
        ));
    }
    if kid.is_some() && candidates.len() != 1 {
        return Err(anyhow!(
            "JWKS kid {:?} is ambiguous for the token algorithm",
            kid
        ));
    }

    // Build a jsonwebtoken::Validation. Algorithm is taken from the JWT header
    // (each candidate shares the header `alg` because candidate_keys filtered
    // on kty/crv consistency).
    let alg = algorithm_from_header(&header)?;
    let mut validation = Validation::new(alg);
    validation.set_issuer(&[expected_issuer]);
    validation.set_audience(&[expected_audience]);
    // Allow 60 seconds of clock skew (same as provider.clock_skew_seconds default)
    validation.leeway = 60;

    // Try each candidate until one verifies. A JWKS may publish several keys
    // simultaneously (overlap rotation / PQ-hybrid); only one is the real
    // verifier and the rest are expected misses, logged at `debug`. An unknown
    // or unsupported entry is skipped rather than failing the whole set.
    let mut last_err: Option<anyhow::Error> = None;
    for jwk in &candidates {
        let decoding_key = match build_decoding_key(jwk) {
            Ok(k) => k,
            Err(e) => {
                tracing::debug!(error = %e, "id_token: skipping JWKS key (invalid key material)");
                last_err = Some(e);
                continue;
            }
        };
        match jsonwebtoken::decode::<Value>(token, &decoding_key, &validation) {
            Ok(decoded) => {
                return Ok(VerifiedIdToken {
                    claims: decoded.claims,
                });
            }
            Err(e) => {
                tracing::debug!(error = %e, "id_token: candidate key did not verify; trying next");
                last_err = Some(anyhow!("id_token signature verification failed: {}", e));
            }
        }
    }

    Err(last_err.unwrap_or_else(|| {
        anyhow!(
            "id_token verification failed: no candidate verified ({} candidates)",
            candidates.len()
        )
    }))
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

/// Collect every algorithm-compatible JWKS candidate for a JWT header.
///
/// When the JWT carries a `kid`, only entries with that exact `kid` are
/// returned (there may be several — e.g. an Ed25519 entry and a composite
/// entry published side-by-side under the same kid). When it carries no
/// `kid`, every entry whose `kty`/`crv` matches the header `alg` is
/// returned. Unknown / unsupported algorithms are skipped: the caller tries
/// each candidate and a published set is never collapsed to a positional
/// singleton (#1183 / #1184).
fn candidate_keys<'a>(keys: &'a [Value], kid: Option<&str>, header: &Value) -> Vec<&'a Value> {
    let header_alg = header.get("alg").and_then(|v| v.as_str()).unwrap_or("");

    keys.iter()
        .filter(|key| {
            let kid_matches = match kid {
                Some(want) => key.get("kid").and_then(|v| v.as_str()) == Some(want),
                None => true,
            };
            kid_matches && key_matches_algorithm(key, header_alg)
        })
        .collect()
}

/// Pin a JWK candidate to the JWT's declared algorithm. Key type, curve, and
/// an explicit JWK `alg` (when present) must all agree; candidate iteration is
/// never allowed to create an algorithm downgrade.
fn key_matches_algorithm(key: &Value, header_alg: &str) -> bool {
    if let Some(key_alg) = key.get("alg").and_then(|v| v.as_str()) {
        if key_alg != header_alg {
            return false;
        }
    }

    let kty = key.get("kty").and_then(|v| v.as_str()).unwrap_or("");
    let crv = key.get("crv").and_then(|v| v.as_str());
    match header_alg {
        "RS256" => kty == "RSA",
        "ES256" => kty == "EC" && crv == Some("P-256"),
        "EdDSA" => kty == "OKP" && crv == Some("Ed25519"),
        _ => false,
    }
}

/// Resolve the jsonwebtoken `Algorithm` from the JWT header `alg`, after
/// [`candidate_keys`] has already filtered the JWKS to kty/crv-consistent
/// entries.
fn algorithm_from_header(header: &Value) -> Result<Algorithm> {
    let alg_str = header
        .get("alg")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("JWT header missing 'alg'"))?;
    match alg_str {
        "RS256" => Ok(Algorithm::RS256),
        "ES256" => Ok(Algorithm::ES256),
        "EdDSA" => Ok(Algorithm::EdDSA),
        other => Err(anyhow!("Unsupported JWT algorithm: {}", other)),
    }
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
    fn test_algorithm_from_header_rs256() {
        let header = serde_json::json!({"alg": "RS256"});
        assert_eq!(algorithm_from_header(&header).unwrap(), Algorithm::RS256);
    }

    #[test]
    fn test_algorithm_from_header_missing() {
        let header = serde_json::json!({});
        assert!(algorithm_from_header(&header).is_err());
    }

    #[test]
    fn test_algorithm_from_header_unsupported() {
        let header = serde_json::json!({"alg": "RS512"});
        assert!(algorithm_from_header(&header).is_err());
    }

    #[test]
    fn test_candidate_keys_by_kid() {
        let keys = vec![
            serde_json::json!({"kty": "RSA", "kid": "key-1", "n": "abc", "e": "AQAB"}),
            serde_json::json!({"kty": "RSA", "kid": "key-2", "n": "def", "e": "AQAB"}),
        ];
        let header = serde_json::json!({"alg": "RS256", "kid": "key-2"});
        let found = candidate_keys(&keys, Some("key-2"), &header);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0]["kid"], "key-2");
    }

    #[test]
    fn test_candidate_keys_no_kid_returns_all_alg_compatible() {
        // Two simultaneously published RSA keys (overlap rotation). Without a
        // `kid`, the verifier MUST retain BOTH candidates so it can try each —
        // collapsing to "the first one" forecloses overlap rotation and
        // PQ-hybrid rollout (#1183 / #1184).
        let keys = vec![
            serde_json::json!({"kty": "RSA", "kid": "key-1", "n": "abc", "e": "AQAB"}),
            serde_json::json!({"kty": "RSA", "kid": "key-2", "n": "def", "e": "AQAB"}),
        ];
        let header = serde_json::json!({"alg": "RS256"});
        let found = candidate_keys(&keys, None, &header);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_candidate_keys_skips_unknown_alg() {
        // An unsupported/unknown algorithm entry MUST be skipped rather than
        // fail the whole set — a PQ-capable verifier ignores what it does not
        // recognize while a classical verifier keeps working against the
        // classical entry.
        let keys = vec![
            serde_json::json!({"kty": "AKP", "alg": "ML-DSA-65-Ed25519", "kid": "pq"}),
            serde_json::json!({"kty": "RSA", "kid": "classic", "n": "abc", "e": "AQAB"}),
        ];
        let header = serde_json::json!({"alg": "RS256"});
        let found = candidate_keys(&keys, None, &header);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0]["kid"], "classic");
    }

    #[test]
    fn test_candidate_keys_no_kid_no_match() {
        let keys = vec![
            serde_json::json!({"kty": "EC", "kid": "ec-1", "x": "a", "y": "b"}),
        ];
        let header = serde_json::json!({"alg": "RS256"});
        let found = candidate_keys(&keys, None, &header);
        assert!(found.is_empty());
    }

    /// Build an EdDSA JWT signed with `signing_key`. The header carries the
    /// given optional `kid`. Used by the overlap-rotation tests so they can
    /// sign with arbitrary Ed25519 keys without a PEM/DER round-trip.
    fn sign_eddsa_jwt(
        signing_key: &ed25519_dalek::SigningKey,
        claims: &serde_json::Value,
        kid: Option<&str>,
    ) -> String {
        use ed25519_dalek::Signer;
        let mut header = serde_json::json!({"alg": "EdDSA", "typ": "JWT"});
        if let Some(k) = kid {
            header["kid"] = serde_json::Value::String(k.to_owned());
        }
        let header_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap());
        let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(claims).unwrap());
        let signing_input = format!("{header_b64}.{payload_b64}");
        let sig = signing_key.sign(signing_input.as_bytes());
        let sig_b64 = URL_SAFE_NO_PAD.encode(sig.to_bytes());
        format!("{signing_input}.{sig_b64}")
    }

    fn jwks_ed25519(entries: &[(&str, &ed25519_dalek::VerifyingKey)]) -> Vec<Value> {
        entries
            .iter()
            .map(|(kid, vk)| {
                serde_json::json!({
                    "kty": "OKP", "crv": "Ed25519", "alg": "EdDSA",
                    "kid": kid,
                    "x": URL_SAFE_NO_PAD.encode(vk.as_bytes()),
                })
            })
            .collect()
    }

    /// #1184: a kid-less id_token signed by the SECOND of two simultaneously
    /// published Ed25519 keys must verify against the JWKS set. The previous
    /// implementation returned `keys.first()` and would fail for the non-first
    /// signer; reverting this hunk to "try only the first candidate" makes
    /// this test fail.
    #[test]
    fn id_token_kid_less_signer_is_non_first_published_key() -> anyhow::Result<()> {
        let sk_a = ed25519_dalek::SigningKey::from_bytes(&[0xA0; 32]);
        let sk_b = ed25519_dalek::SigningKey::from_bytes(&[0xB0; 32]);

        // Publish BOTH keys (overlap window): lead first, drain second.
        let keys = jwks_ed25519(&[("lead", &sk_a.verifying_key()), ("drain", &sk_b.verifying_key())]);

        // Sign with the SECOND (drain) key — no `kid` in the header.
        let claims = serde_json::json!({
            "iss": "https://idp.example.com",
            "aud": "client-42",
            "sub": "user-1",
            "exp": (jsonwebtoken::get_current_timestamp() + 300),
        });
        let token = sign_eddsa_jwt(&sk_b, &claims, None);

        let verified = verify_id_token_with_keys(
            &token,
            &keys,
            "https://idp.example.com",
            "client-42",
        )?;
        assert_eq!(verified.claims["sub"], "user-1");
        Ok(())
    }

    /// #1184 overlap: while both keys are published, a token signed by either
    /// verifies (first AND non-first), and a token signed by an UNKNOWN key
    /// that is not in the set is rejected.
    #[test]
    fn id_token_overlap_both_signers_accepted_unknown_rejected() -> anyhow::Result<()> {
        let sk_lead = ed25519_dalek::SigningKey::from_bytes(&[0x11; 32]);
        let sk_drain = ed25519_dalek::SigningKey::from_bytes(&[0x22; 32]);
        let sk_unknown = ed25519_dalek::SigningKey::from_bytes(&[0x99; 32]);

        let keys = jwks_ed25519(&[
            ("lead", &sk_lead.verifying_key()),
            ("drain", &sk_drain.verifying_key()),
        ]);

        let claims = serde_json::json!({
            "iss": "https://idp.example.com",
            "aud": "client-42",
            "sub": "user-1",
            "exp": (jsonwebtoken::get_current_timestamp() + 300),
        });

        // First (lead) signer — accepted.
        let token_lead = sign_eddsa_jwt(&sk_lead, &claims, None);
        let v = verify_id_token_with_keys(&token_lead, &keys, "https://idp.example.com", "client-42")?;
        assert_eq!(v.claims["sub"], "user-1");

        // Non-first (drain) signer — accepted.
        let token_drain = sign_eddsa_jwt(&sk_drain, &claims, None);
        let v = verify_id_token_with_keys(&token_drain, &keys, "https://idp.example.com", "client-42")?;
        assert_eq!(v.claims["sub"], "user-1");

        // Unknown signer (not in the published set) — rejected.
        let token_bad = sign_eddsa_jwt(&sk_unknown, &claims, None);
        let err = verify_id_token_with_keys(&token_bad, &keys, "https://idp.example.com", "client-42")
            .unwrap_err();
        assert!(
            err.to_string().contains("signature verification failed")
                || err.to_string().contains("no candidate verified"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    /// #1184: an unsupported/unknown algorithm entry in the JWKS MUST be
    /// skipped rather than fail the whole set — a PQ-capable verifier ignores
    /// what it does not recognize while a classical verifier keeps working
    /// against the classical entry.
    #[test]
    fn id_token_skips_unsupported_alg_and_accepts_classical() -> anyhow::Result<()> {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[0x33; 32]);
        let vk = sk.verifying_key();

        let keys = vec![
            // Unknown composite_alg entry — must be skipped, not fatal.
            serde_json::json!({"kty": "AKP", "alg": "ML-DSA-65-Ed25519", "kid": "pq"}),
            // Classical Ed25519 entry — the real verifier.
            serde_json::json!({
                "kty": "OKP", "crv": "Ed25519", "alg": "EdDSA", "kid": "classic",
                "x": URL_SAFE_NO_PAD.encode(vk.as_bytes()),
            }),
        ];

        let claims = serde_json::json!({
            "iss": "https://idp.example.com",
            "aud": "client-42",
            "sub": "user-1",
            "exp": (jsonwebtoken::get_current_timestamp() + 300),
        });
        let token = sign_eddsa_jwt(&sk, &claims, None);
        let v = verify_id_token_with_keys(&token, &keys, "https://idp.example.com", "client-42")?;
        assert_eq!(v.claims["sub"], "user-1");
        Ok(())
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
