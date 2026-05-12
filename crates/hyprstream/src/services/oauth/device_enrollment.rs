//! Anonymous device identity enrollment — POST /api/device/challenge and /api/device/enroll.
//!
//! Unauthenticated endpoints. A client proves Ed25519 key possession without
//! an existing user session. Device-user association happens later, implicitly,
//! when the `__Secure-VaultDevice` cookie accompanies an OAuth token request.
//!
//! Rate limiting on these endpoints is mandatory before production use (RATE-1).

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::Utc;
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};

use crate::auth::{
    device_challenge::{
        compute_kid, derive_challenge_key, derive_cookie_key_seed,
        device_challenge, encode_vault_device_cookie, verify_device_enrollment,
    },
    user_store::DeviceRecord,
};
use crate::services::oauth::state::OAuthState;
use hyprstream_rpc::node_identity::derive_purpose_key;

#[derive(Debug, Deserialize)]
pub struct ChallengeRequest {
    pub pubkey: String,
}

#[derive(Debug, Serialize)]
pub struct ChallengeResponse {
    pub challenge: String,
}

#[derive(Debug, Deserialize)]
pub struct EnrollRequest {
    pub pubkey: String,
    pub signature: String,
}

#[derive(Debug, Serialize)]
pub struct EnrollResponse {
    pub adt: String,
    pub expires_in: u64,
}

fn decode_b64url(s: &str) -> Option<Vec<u8>> {
    URL_SAFE_NO_PAD.decode(s).ok()
}

fn json_error(status: StatusCode, msg: &str) -> Response {
    (status, Json(serde_json::json!({ "error": msg }))).into_response()
}

/// POST /api/device/challenge
///
/// Returns `HMAC(challenge_key, pubkey)` for the client to sign.
/// Stateless — no server storage.
pub async fn device_challenge_handler(
    State(state): State<Arc<OAuthState>>,
    Json(req): Json<ChallengeRequest>,
) -> Response {
    let pubkey_bytes = match decode_b64url(&req.pubkey) {
        Some(b) if b.len() == 32 => b,
        _ => return json_error(StatusCode::BAD_REQUEST, "pubkey must be 32 bytes base64url"),
    };
    let mut pubkey_arr = [0u8; 32];
    pubkey_arr.copy_from_slice(&pubkey_bytes);

    let Some(ref sk) = state.signing_key else {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "signing key not configured");
    };

    let sk_bytes = sk.to_bytes();
    let challenge_key = derive_challenge_key(&sk_bytes);
    let challenge = device_challenge(&challenge_key, &pubkey_arr);

    Json(ChallengeResponse {
        challenge: URL_SAFE_NO_PAD.encode(challenge),
    })
    .into_response()
}

/// POST /api/device/enroll
///
/// Verifies the Ed25519 signature, upserts a DeviceRecord, issues an ADT JWT,
/// and sets the `__Secure-VaultDevice` cookie.
pub async fn device_enroll_handler(
    State(state): State<Arc<OAuthState>>,
    Json(req): Json<EnrollRequest>,
) -> Response {
    let pubkey_bytes = match decode_b64url(&req.pubkey) {
        Some(b) if b.len() == 32 => b,
        _ => return json_error(StatusCode::BAD_REQUEST, "pubkey must be 32 bytes base64url"),
    };
    let sig_bytes = match decode_b64url(&req.signature) {
        Some(b) if b.len() == 64 => b,
        _ => return json_error(StatusCode::BAD_REQUEST, "signature must be 64 bytes base64url"),
    };

    let mut pubkey_arr = [0u8; 32];
    pubkey_arr.copy_from_slice(&pubkey_bytes);
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&sig_bytes);

    let Some(ref sk) = state.signing_key else {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "signing key not configured");
    };

    let sk_bytes = sk.to_bytes();
    let challenge_key = derive_challenge_key(&sk_bytes);

    if !verify_device_enrollment(&challenge_key, &pubkey_arr, &sig_arr) {
        return json_error(StatusCode::UNAUTHORIZED, "invalid signature");
    }

    // Compute base58 fingerprint
    let fingerprint = bs58::encode(&pubkey_arr).into_string();
    let now = Utc::now().timestamp();

    // Upsert DeviceRecord
    let record = DeviceRecord {
        pubkey: pubkey_arr,
        fingerprint: fingerprint.clone(),
        label: None,
        enrolled_at: now,
        last_seen_at: None,
        user_sub: None,
    };

    if let Some(ref ds) = state.device_store {
        if let Err(e) = ds.enroll_device(record).await {
            tracing::warn!("Failed to persist DeviceRecord: {e}");
        }
    }

    // Derive cookie signing key from signing key
    let cookie_seed = derive_cookie_key_seed(&sk_bytes);
    let cookie_sk = SigningKey::from_bytes(&cookie_seed);
    let cookie_kid = compute_kid(cookie_sk.verifying_key().as_bytes());
    let cookie_value = encode_vault_device_cookie(&cookie_sk, &cookie_kid, &pubkey_arr);

    // Issue ADT JWT
    let sub = format!("device:{fingerprint}");
    let pubkey_b64 = URL_SAFE_NO_PAD.encode(pubkey_arr);
    let adt_exp = now + 86400;

    // Build claims manually (Claims struct may not have all fields we need)
    let adt_payload = serde_json::json!({
        "iss": state.issuer_url,
        "sub": sub,
        "aud": "hyprstream",
        "cnf": { "jwk": { "kty": "OKP", "crv": "Ed25519", "x": pubkey_b64 } },
        "scope": "device:anonymous",
        "iat": now,
        "exp": adt_exp,
    });

    // Sign the ADT JWT using the JWT signing purpose key
    let jwt_key = derive_purpose_key(sk, "hyprstream-jwt-v1");
    let kid = hyprstream_rpc::auth::jwt::kid_for_key(&jwt_key);
    let header = serde_json::json!({ "alg": "EdDSA", "typ": "at+jwt", "kid": kid });
    let header_b64 = URL_SAFE_NO_PAD.encode(header.to_string());
    let payload_b64 = URL_SAFE_NO_PAD.encode(adt_payload.to_string());
    let signing_input = format!("{header_b64}.{payload_b64}");
    use ed25519_dalek::Signer;
    let sig = jwt_key.sign(signing_input.as_bytes());
    let adt = format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(sig.to_bytes()));

    // Cookie attributes: HttpOnly, Secure, SameSite=Strict, Max-Age=30 days
    let cookie_header = format!(
        "__Secure-VaultDevice={}; HttpOnly; Secure; SameSite=Strict; Max-Age=2592000; Path=/",
        cookie_value
    );

    (
        StatusCode::OK,
        [(header::SET_COOKIE, cookie_header)],
        Json(EnrollResponse { adt, expires_in: 86400 }),
    )
        .into_response()
}
