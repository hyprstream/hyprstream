//! SPIFFE/JWT-SVID compatibility surface for the Kubernetes security spine.
//!
//! This module deliberately reuses the existing OAuth trust machinery:
//! external cluster/SPIRE issuers must already be present in
//! `OAuthState::trusted_issuers`, which is populated only after the normal
//! `federation:register` admission path. Accepted workload JWTs are re-issued
//! as local WITs; hyprstream-provided JWT-SVIDs are signed by the same active
//! JWT signing key published by the bundle endpoint.

use std::sync::Arc;

use axum::{
    Extension, Json,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use ed25519_dalek::{Signature, Signer as _, Verifier as _};
use serde::{Deserialize, Serialize};

use super::state::OAuthState;
use crate::server::middleware::AuthenticatedUser;

const K8S_WIT_TTL: i64 = 10 * 60;
const JWT_SVID_TTL: i64 = 10 * 60;

#[derive(Debug, Deserialize)]
pub struct WorkloadWitRequest {
    /// Projected Kubernetes ServiceAccount token or SPIRE JWT-SVID.
    pub subject_token: String,
    /// Expected audience of the presented token. Kubernetes projected tokens
    /// should request this audience explicitly; absent/wrong aud fails closed.
    pub audience: String,
    /// Optional base64url Ed25519 key to carry as `cnf.jwk` in the issued WIT.
    /// If omitted and a valid DPoP proof is supplied, the DPoP public key binds
    /// the WIT instead.
    #[serde(default)]
    pub pubkey: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WorkloadWitResponse {
    pub wit: String,
    pub token_type: &'static str,
    pub expires_in: i64,
    pub subject: String,
    pub source_issuer: String,
    pub source_subject: String,
}

#[derive(Debug, Deserialize)]
pub struct ServiceSvidRequest {
    /// Service name in `spiffe://<trust-domain>/service/<name>`.
    pub service: String,
    /// JWT-SVID audiences. At least one is required by SPIFFE JWT-SVID.
    pub audience: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ServiceSvidResponse {
    pub token: String,
    pub token_type: &'static str,
    pub expires_in: i64,
    pub spiffe_id: String,
    pub bundle_endpoint: String,
}

#[derive(Debug, Serialize)]
struct JwtSvidClaims {
    iss: String,
    sub: String,
    aud: Vec<String>,
    iat: i64,
    exp: i64,
}

#[derive(Debug, Deserialize)]
struct ExternalWorkloadClaims {
    iss: String,
    sub: String,
    exp: i64,
    #[serde(default)]
    nbf: Option<i64>,
    #[serde(default)]
    aud: serde_json::Value,
}

/// `POST /oauth/spiffe/wit` — exchange a cluster/SPIRE JWT for a local WIT.
pub async fn exchange_workload_wit(
    State(state): State<Arc<OAuthState>>,
    headers: HeaderMap,
    Json(body): Json<WorkloadWitRequest>,
) -> Response {
    if body.audience.trim().is_empty() {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("audience is required"),
        );
    }

    let source =
        match verify_external_workload_jwt(&state, &body.subject_token, &body.audience).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "SPIFFE/WIT exchange rejected");
                return oauth_error(StatusCode::UNAUTHORIZED, "invalid_grant", None);
            }
        };

    let endpoint = format!(
        "{}/oauth/spiffe/wit",
        state.issuer_url.trim_end_matches('/')
    );
    let bind_key = match binding_key(&headers, body.pubkey.as_deref(), &endpoint) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "SPIFFE/WIT exchange rejected: invalid sender binding");
            return oauth_error(StatusCode::BAD_REQUEST, "invalid_request", Some(e));
        }
    };

    let ca_key = match state.active_jwt_signing_key().await {
        Some(k) => k,
        None => {
            return oauth_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "temporarily_unavailable",
                Some("WIT issuance not available"),
            );
        }
    };

    let now = chrono::Utc::now().timestamp();
    let exp = now + K8S_WIT_TTL;
    let subject = map_workload_subject(&source.iss, &source.sub);
    let mut claims = hyprstream_rpc::auth::Claims::new(subject.clone(), now, exp)
        .with_issuer(state.issuer_url.clone())
        .with_audience(Some(body.audience));
    if let Some(key) = bind_key {
        claims = claims.with_cnf_jwk(&key);
    }
    let wit = hyprstream_rpc::auth::jwt::encode_service_jwt(&claims, &ca_key);

    tracing::info!(
        source_issuer = %source.iss,
        source_subject = %source.sub,
        subject = %subject,
        "SPIFFE/Kubernetes workload token exchanged for WIT"
    );

    (
        StatusCode::OK,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(WorkloadWitResponse {
            wit,
            token_type: "wit+jwt",
            expires_in: K8S_WIT_TTL,
            subject,
            source_issuer: source.iss,
            source_subject: source.sub,
        }),
    )
        .into_response()
}

/// `POST /oauth/spiffe/service-svid` — issue a JWT-SVID for a hyprstream service.
///
/// Protected by normal OAuth bearer auth. PolicyService remains the authority
/// behind issuance; this HTTP surface only shapes the token as JWT-SVID.
pub async fn issue_service_svid(
    State(state): State<Arc<OAuthState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(body): Json<ServiceSvidRequest>,
) -> Response {
    if !valid_service_name(&body.service) {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("service must contain only [A-Za-z0-9._-]"),
        );
    }
    if body.audience.is_empty() || body.audience.iter().any(|a| a.trim().is_empty()) {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("audience must contain at least one non-empty value"),
        );
    }

    let signing_key = match state.active_jwt_signing_key().await {
        Some(k) => k,
        None => {
            return oauth_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "temporarily_unavailable",
                Some("JWT-SVID issuance not available"),
            );
        }
    };

    let now = chrono::Utc::now().timestamp();
    let exp = now + JWT_SVID_TTL;
    let spiffe_id = service_spiffe_id(&state.issuer_url, &body.service);
    let claims = JwtSvidClaims {
        iss: state.issuer_url.clone(),
        sub: spiffe_id.clone(),
        aud: body.audience,
        iat: now,
        exp,
    };
    let token = encode_jwt_svid(&claims, &signing_key);

    tracing::info!(
        requester = %user.user,
        spiffe_id = %spiffe_id,
        "SPIFFE JWT-SVID issued for hyprstream service"
    );

    (
        StatusCode::OK,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(ServiceSvidResponse {
            token,
            token_type: "jwt-svid",
            expires_in: JWT_SVID_TTL,
            spiffe_id,
            bundle_endpoint: format!(
                "{}/.well-known/spiffe/bundle",
                state.issuer_url.trim_end_matches('/')
            ),
        }),
    )
        .into_response()
}

/// `GET /.well-known/spiffe/bundle` — SPIFFE trust-domain bundle.
pub async fn spiffe_bundle(State(state): State<Arc<OAuthState>>) -> Response {
    let keys = super::jwks::jwks_json(&state).await;
    Json(serde_json::json!({
        "spiffe_sequence": 1,
        "keys": keys,
    }))
    .into_response()
}

async fn verify_external_workload_jwt(
    state: &Arc<OAuthState>,
    token: &str,
    expected_audience: &str,
) -> Result<ExternalWorkloadClaims, String> {
    let unverified = decode_external_claims_unverified(token)
        .map_err(|e| format!("cannot parse workload jwt: {e}"))?;
    if unverified.iss.is_empty() {
        return Err("workload jwt missing iss".to_owned());
    }
    if unverified.sub.is_empty() {
        return Err("workload jwt missing sub".to_owned());
    }

    let cfg = state
        .trusted_issuers
        .get(&unverified.iss)
        .ok_or_else(|| format!("issuer is not registered/trusted: {}", unverified.iss))?
        .clone();
    let vk =
        super::jwt_bearer::resolve_federated_key(state, &unverified.iss, token, cfg.allow_http)
            .await?;
    verify_external_eddsa_jwt(token, &vk)?;
    let claims = decode_external_claims_unverified(token)?;
    validate_external_claims(&claims, expected_audience)?;
    Ok(claims)
}

fn decode_external_claims_unverified(token: &str) -> Result<ExternalWorkloadClaims, String> {
    let payload_b64 = token
        .split('.')
        .nth(1)
        .ok_or_else(|| "invalid jwt format".to_owned())?;
    let payload = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|_| "invalid jwt payload base64".to_owned())?;
    serde_json::from_slice(&payload).map_err(|e| format!("invalid jwt claims: {e}"))
}

fn verify_external_eddsa_jwt(token: &str, key: &ed25519_dalek::VerifyingKey) -> Result<(), String> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err("invalid jwt format".to_owned());
    }
    let header = URL_SAFE_NO_PAD
        .decode(parts[0])
        .map_err(|_| "invalid jwt header base64".to_owned())?;
    let header: serde_json::Value =
        serde_json::from_slice(&header).map_err(|e| format!("invalid jwt header: {e}"))?;
    if header.get("alg").and_then(|v| v.as_str()) != Some("EdDSA") {
        return Err("only EdDSA workload JWTs are supported".to_owned());
    }
    let sig_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|_| "invalid jwt signature base64".to_owned())?;
    let sig_bytes: [u8; 64] = sig_bytes
        .try_into()
        .map_err(|_| "invalid Ed25519 signature length".to_owned())?;
    let signing_input = format!("{}.{}", parts[0], parts[1]);
    key.verify(signing_input.as_bytes(), &Signature::from_bytes(&sig_bytes))
        .map_err(|_| "workload jwt signature verification failed".to_owned())
}

fn validate_external_claims(
    claims: &ExternalWorkloadClaims,
    expected_audience: &str,
) -> Result<(), String> {
    let now = chrono::Utc::now().timestamp();
    if claims.exp <= now {
        return Err("workload jwt expired".to_owned());
    }
    if claims.nbf.is_some_and(|nbf| nbf > now + 5) {
        return Err("workload jwt not yet valid".to_owned());
    }
    let audience_ok = match &claims.aud {
        serde_json::Value::String(aud) => aud == expected_audience,
        serde_json::Value::Array(auds) => auds
            .iter()
            .any(|aud| aud.as_str() == Some(expected_audience)),
        _ => false,
    };
    if !audience_ok {
        return Err("workload jwt audience mismatch".to_owned());
    }
    Ok(())
}

fn binding_key(
    headers: &HeaderMap,
    explicit_pubkey: Option<&str>,
    endpoint: &str,
) -> Result<Option<[u8; 32]>, &'static str> {
    if let Some(pubkey) = explicit_pubkey {
        return decode_ed25519_pubkey(pubkey).map(Some);
    }

    let Some(dpop) = headers.get("DPoP").and_then(|v| v.to_str().ok()) else {
        return Ok(None);
    };
    let proof = super::dpop::verify_dpop_proof(dpop, "POST", endpoint, None)
        .map_err(|_| "invalid DPoP proof")?;
    match proof.key {
        super::dpop::DpopKey::Ed25519 { bytes } => Ok(Some(bytes)),
        _ => Err("DPoP proof key must be Ed25519 for WIT cnf.jwk binding"),
    }
}

fn decode_ed25519_pubkey(value: &str) -> Result<[u8; 32], &'static str> {
    let bytes: [u8; 32] = URL_SAFE_NO_PAD
        .decode(value)
        .ok()
        .and_then(|b| b.try_into().ok())
        .ok_or("pubkey must be base64url-encoded 32-byte Ed25519 public key")?;
    ed25519_dalek::VerifyingKey::from_bytes(&bytes)
        .map_err(|_| "pubkey is not a valid Ed25519 public key")?;
    Ok(bytes)
}

fn map_workload_subject(issuer: &str, sub: &str) -> String {
    if let Some(rest) = sub.strip_prefix("system:serviceaccount:") {
        let mut parts = rest.splitn(2, ':');
        if let (Some(ns), Some(sa)) = (parts.next(), parts.next()) {
            return format!("k8s:{}:{}:{}", trust_domain_from_issuer(issuer), ns, sa);
        }
    }
    if let Some(rest) = sub.strip_prefix("spiffe://") {
        return format!("spiffe:{rest}");
    }
    format!("federated:{}:{}", trust_domain_from_issuer(issuer), sub)
}

fn service_spiffe_id(issuer_url: &str, service: &str) -> String {
    format!(
        "spiffe://{}/service/{service}",
        trust_domain_from_issuer(issuer_url)
    )
}

fn trust_domain_from_issuer(issuer_url: &str) -> String {
    issuer_url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .split('/')
        .next()
        .unwrap_or(issuer_url)
        .to_owned()
}

fn valid_service_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

fn encode_jwt_svid(claims: &JwtSvidClaims, signing_key: &ed25519_dalek::SigningKey) -> String {
    let kid = hyprstream_rpc::auth::jwt::kid_for_key(signing_key);
    let header = serde_json::json!({
        "alg": "EdDSA",
        "typ": "JWT",
        "kid": kid,
    });
    let header_b64 = URL_SAFE_NO_PAD.encode(header.to_string());
    let payload = serde_json::to_vec(claims).unwrap_or_default();
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload);
    let signing_input = format!("{header_b64}.{payload_b64}");
    let sig = signing_key.sign(signing_input.as_bytes());
    format!(
        "{}.{}",
        signing_input,
        URL_SAFE_NO_PAD.encode(sig.to_bytes())
    )
}

fn oauth_error(status: StatusCode, error: &str, description: Option<&str>) -> Response {
    let mut body = serde_json::Map::new();
    body.insert(
        "error".to_owned(),
        serde_json::Value::String(error.to_owned()),
    );
    if let Some(description) = description {
        body.insert(
            "error_description".to_owned(),
            serde_json::Value::String(description.to_owned()),
        );
    }
    (
        status,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(serde_json::Value::Object(body)),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn k8s_service_account_subject_maps_without_collision() {
        assert_eq!(
            map_workload_subject(
                "https://issuer.example.test",
                "system:serviceaccount:training:runner"
            ),
            "k8s:issuer.example.test:training:runner"
        );
    }

    #[test]
    fn service_spiffe_id_uses_issuer_host_as_trust_domain() {
        assert_eq!(
            service_spiffe_id("https://hypr.example.test/oauth", "model"),
            "spiffe://hypr.example.test/service/model"
        );
    }

    #[test]
    fn jwt_svid_uses_audience_array() {
        let key = ed25519_dalek::SigningKey::generate(&mut OsRng);
        let token = encode_jwt_svid(
            &JwtSvidClaims {
                iss: "https://hypr.example.test".to_owned(),
                sub: "spiffe://hypr.example.test/service/model".to_owned(),
                aud: vec!["9p".to_owned(), "rpc".to_owned()],
                iat: 1,
                exp: 2,
            },
            &key,
        );
        let Some(payload_b64) = token.split('.').nth(1) else {
            panic!("payload part");
        };
        let payload = match URL_SAFE_NO_PAD.decode(payload_b64) {
            Ok(payload) => payload,
            Err(e) => panic!("payload decodes: {e}"),
        };
        let value: serde_json::Value = match serde_json::from_slice(&payload) {
            Ok(value) => value,
            Err(e) => panic!("json payload: {e}"),
        };
        assert_eq!(value["aud"], serde_json::json!(["9p", "rpc"]));
        assert_eq!(value["sub"], "spiffe://hypr.example.test/service/model");
    }
}
