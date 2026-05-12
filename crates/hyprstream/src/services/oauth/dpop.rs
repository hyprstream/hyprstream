//! RFC 9449 DPoP (Demonstrating Proof-of-Possession) proof verification.
//!
//! Verifies `DPoP` JWT proofs presented at the token endpoint or as
//! `Authorization: DPoP` on resource requests.
//!
//! DPoP proofs carry:
//! - `jwk` in the header — the ephemeral client key (used to verify signature)
//! - `htm` — HTTP method
//! - `htu` — HTTP URI (scheme + host + path, no query)
//! - `iat` — issued-at (must be within ±60s of server clock)
//! - `jti` — unique identifier (replay prevention)
//! - `ath` — base64url(SHA-256(access_token)) — required when the proof
//!   accompanies an access token on a resource request

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::VerifyingKey;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// Parsed and verified DPoP proof.
#[derive(Debug, Clone)]
pub struct DpopProof {
    /// Raw Ed25519 public key bytes from the proof's `jwk` header.
    pub public_key_bytes: [u8; 32],
    /// RFC 7638 JWK thumbprint of the key (SHA-256, base64url).
    pub jkt: String,
    /// Unique token identifier for replay prevention.
    pub jti: String,
    /// HTTP method from the proof.
    pub htm: String,
    /// HTTP URI from the proof.
    pub htu: String,
    /// Issued-at timestamp (Unix seconds).
    pub iat: i64,
    /// `ath` — base64url(SHA-256(access_token)) when present.
    pub ath: Option<String>,
    /// Optional server-issued nonce.
    pub nonce: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum DpopError {
    #[error("DPoP proof header is not a valid JWT: {0}")]
    InvalidFormat(String),
    #[error("DPoP JWT typ must be 'dpop+jwt', got '{0}'")]
    WrongTyp(String),
    #[error("DPoP JWT must use Ed25519 (EdDSA), got '{0}'")]
    WrongAlg(String),
    #[error("DPoP JWT jwk header must be an OKP/Ed25519 JWK")]
    InvalidJwk,
    #[error("DPoP proof signature verification failed")]
    SignatureInvalid,
    #[error("DPoP proof missing claim '{0}'")]
    MissingClaim(&'static str),
    #[error("DPoP proof htm '{actual}' does not match expected '{expected}'")]
    HtmMismatch { expected: String, actual: String },
    #[error("DPoP proof htu '{actual}' does not match expected '{expected}'")]
    HtuMismatch { expected: String, actual: String },
    #[error("DPoP proof iat {iat} is outside the ±60s window (server time: {now})")]
    IatOutOfWindow { iat: i64, now: i64 },
    #[error("DPoP proof ath mismatch — access token binding failed")]
    AthMismatch,
    #[error("DPoP proof jti has been seen before (replay)")]
    JtiReplayed,
}

/// Verify a DPoP proof JWT string.
///
/// - `expected_htm`: uppercase HTTP method (e.g. "POST")
/// - `expected_htu`: scheme + host + path (no query string)
/// - `access_token`: when `Some`, the `ath` claim is required and verified
pub fn verify_dpop_proof(
    dpop_header: &str,
    expected_htm: &str,
    expected_htu: &str,
    access_token: Option<&str>,
) -> Result<DpopProof, DpopError> {
    // Split JWT into three parts.
    let parts: Vec<&str> = dpop_header.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err(DpopError::InvalidFormat("not a three-part JWT".to_owned()));
    }

    // Decode header.
    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .map_err(|e| DpopError::InvalidFormat(format!("header base64: {e}")))?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| DpopError::InvalidFormat(format!("header JSON: {e}")))?;

    // Validate typ.
    let typ = header["typ"].as_str().unwrap_or("");
    if !typ.eq_ignore_ascii_case("dpop+jwt") {
        return Err(DpopError::WrongTyp(typ.to_owned()));
    }

    // Validate alg.
    let alg = header["alg"].as_str().unwrap_or("");
    if !alg.eq_ignore_ascii_case("EdDSA") {
        return Err(DpopError::WrongAlg(alg.to_owned()));
    }

    // Extract public key from jwk header.
    let jwk = &header["jwk"];
    let key_bytes = extract_ed25519_key(jwk)?;
    let verifying_key = VerifyingKey::from_bytes(&key_bytes)
        .map_err(|_| DpopError::InvalidJwk)?;

    // Verify signature over header.payload.
    let signing_input = format!("{}.{}", parts[0], parts[1]);
    let sig_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|e| DpopError::InvalidFormat(format!("signature base64: {e}")))?;
    let sig_array: [u8; 64] = sig_bytes.try_into()
        .map_err(|_| DpopError::SignatureInvalid)?;
    let signature = ed25519_dalek::Signature::from_bytes(&sig_array);
    use ed25519_dalek::Verifier;
    verifying_key
        .verify(signing_input.as_bytes(), &signature)
        .map_err(|_| DpopError::SignatureInvalid)?;

    // Decode payload.
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|e| DpopError::InvalidFormat(format!("payload base64: {e}")))?;
    let payload: serde_json::Value = serde_json::from_slice(&payload_bytes)
        .map_err(|e| DpopError::InvalidFormat(format!("payload JSON: {e}")))?;

    // Required claims.
    let jti = payload["jti"].as_str()
        .ok_or(DpopError::MissingClaim("jti"))?.to_owned();
    let htm = payload["htm"].as_str()
        .ok_or(DpopError::MissingClaim("htm"))?.to_owned();
    let htu = payload["htu"].as_str()
        .ok_or(DpopError::MissingClaim("htu"))?.to_owned();
    let iat = payload["iat"].as_i64()
        .ok_or(DpopError::MissingClaim("iat"))?;

    // Validate htm and htu.
    if !htm.eq_ignore_ascii_case(expected_htm) {
        return Err(DpopError::HtmMismatch {
            expected: expected_htm.to_owned(),
            actual: htm,
        });
    }
    // htu comparison: strip trailing slash, ignore query/fragment.
    let htu_normalized = htu.split('?').next().unwrap_or(&htu).trim_end_matches('/');
    let expected_htu_norm = expected_htu.split('?').next().unwrap_or(expected_htu).trim_end_matches('/');
    if htu_normalized != expected_htu_norm {
        return Err(DpopError::HtuMismatch {
            expected: expected_htu.to_owned(),
            actual: htu.clone(),
        });
    }

    // Validate iat within ±60s.
    let now = chrono::Utc::now().timestamp();
    if (iat - now).abs() > 60 {
        return Err(DpopError::IatOutOfWindow { iat, now });
    }

    // Validate ath when access_token is provided.
    let ath = payload["ath"].as_str().map(str::to_owned);
    if let Some(token) = access_token {
        let expected_ath = URL_SAFE_NO_PAD.encode(Sha256::digest(token.as_bytes()));
        match &ath {
            Some(a) if a.as_bytes().ct_eq(expected_ath.as_bytes()).unwrap_u8() == 1 => {}
            _ => return Err(DpopError::AthMismatch),
        }
    }

    let nonce = payload["nonce"].as_str().map(str::to_owned);
    let jkt = hyprstream_rpc::auth::compute_jkt(&key_bytes);

    Ok(DpopProof {
        public_key_bytes: key_bytes,
        jkt,
        jti,
        htm,
        htu,
        iat,
        ath,
        nonce,
    })
}

fn extract_ed25519_key(jwk: &serde_json::Value) -> Result<[u8; 32], DpopError> {
    if jwk["kty"].as_str() != Some("OKP") || jwk["crv"].as_str() != Some("Ed25519") {
        return Err(DpopError::InvalidJwk);
    }
    let x = jwk["x"].as_str().ok_or(DpopError::InvalidJwk)?;
    let bytes = URL_SAFE_NO_PAD.decode(x).map_err(|_| DpopError::InvalidJwk)?;
    bytes.try_into().map_err(|_| DpopError::InvalidJwk)
}
