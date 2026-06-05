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
//!
//! Supported algorithms:
//! - EdDSA (Ed25519) — native hyprstream identity keys
//! - ES256 (ECDSA P-256 + SHA-256) — atproto interop (required by atproto OAuth)

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::VerifyingKey;
use hyprstream_rpc::auth::{JwkThumbprintInput, jwk_thumbprint};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// The public key from a verified DPoP proof, algorithm-polymorphic.
#[derive(Debug, Clone)]
pub enum DpopKey {
    Ed25519 { bytes: [u8; 32] },
    Es256 { x: [u8; 32], y: [u8; 32] },
    #[cfg(feature = "pq-hybrid")]
    MlDsa65 { pub_bytes: Vec<u8> },
    #[cfg(feature = "pq-hybrid")]
    MlDsa65Ed25519 { pub_bytes: Vec<u8> },
}

impl DpopKey {
    pub fn jkt(&self) -> String {
        match self {
            DpopKey::Ed25519 { bytes } => {
                jwk_thumbprint(&JwkThumbprintInput::Ed25519 { x: bytes })
            }
            DpopKey::Es256 { x, y } => {
                jwk_thumbprint(&JwkThumbprintInput::Es256 { x, y })
            }
            #[cfg(feature = "pq-hybrid")]
            DpopKey::MlDsa65 { pub_bytes } => {
                jwk_thumbprint(&JwkThumbprintInput::Akp { alg: "ML-DSA-65", pub_bytes })
            }
            #[cfg(feature = "pq-hybrid")]
            DpopKey::MlDsa65Ed25519 { pub_bytes } => {
                jwk_thumbprint(&JwkThumbprintInput::Akp { alg: "ML-DSA-65-Ed25519", pub_bytes })
            }
        }
    }

    /// Returns Ed25519 raw bytes if this is an Ed25519 key, None otherwise.
    pub fn ed25519_bytes(&self) -> Option<&[u8; 32]> {
        match self {
            DpopKey::Ed25519 { bytes } => Some(bytes),
            _ => None,
        }
    }
}

/// Parsed and verified DPoP proof.
#[derive(Debug, Clone)]
pub struct DpopProof {
    /// The verified public key from the proof's `jwk` header.
    pub key: DpopKey,
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

impl DpopProof {
    /// Backwards-compat: raw Ed25519 key bytes (panics if ES256).
    /// Prefer `key` field for new code.
    pub fn public_key_bytes(&self) -> Option<[u8; 32]> {
        self.key.ed25519_bytes().copied()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DpopError {
    #[error("DPoP proof header is not a valid JWT: {0}")]
    InvalidFormat(String),
    #[error("DPoP JWT typ must be 'dpop+jwt', got '{0}'")]
    WrongTyp(String),
    #[error("DPoP JWT must use EdDSA or ES256, got '{0}'")]
    WrongAlg(String),
    #[error("DPoP JWT jwk header must be an OKP/Ed25519 or EC/P-256 JWK")]
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

    // Validate alg + extract key + verify signature.
    let alg = header["alg"].as_str().unwrap_or("");
    let signing_input = format!("{}.{}", parts[0], parts[1]);
    let sig_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|e| DpopError::InvalidFormat(format!("signature base64: {e}")))?;

    let jwk = &header["jwk"];
    let key = if alg.eq_ignore_ascii_case("EdDSA") {
        let key_bytes = extract_ed25519_key(jwk)?;
        verify_ed25519(&key_bytes, signing_input.as_bytes(), &sig_bytes)?;
        DpopKey::Ed25519 { bytes: key_bytes }
    } else if alg.eq_ignore_ascii_case("ES256") {
        let (x, y) = extract_p256_key(jwk)?;
        verify_es256(&x, &y, signing_input.as_bytes(), &sig_bytes)?;
        DpopKey::Es256 { x, y }
    } else if cfg!(feature = "pq-hybrid") && alg == "ML-DSA-65" {
        #[cfg(feature = "pq-hybrid")]
        {
            let pub_bytes = extract_akp_pub(jwk)?;
            let vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&pub_bytes)
                .map_err(|_| DpopError::InvalidJwk)?;
            hyprstream_rpc::crypto::pq::ml_dsa_verify(&vk, signing_input.as_bytes(), &sig_bytes)
                .map_err(|_| DpopError::SignatureInvalid)?;
            DpopKey::MlDsa65 { pub_bytes }
        }
        #[cfg(not(feature = "pq-hybrid"))]
        return Err(DpopError::WrongAlg(alg.to_owned()))
    } else if cfg!(feature = "pq-hybrid") && alg == "ML-DSA-65-Ed25519" {
        #[cfg(feature = "pq-hybrid")]
        {
            let pub_bytes = extract_akp_pub(jwk)?;
            // Composite: ML-DSA-65 vk (1952) + Ed25519 vk (32) = 1984
            if pub_bytes.len() != 1952 + 32 {
                return Err(DpopError::InvalidJwk);
            }
            let (ml_dsa_pub, ed25519_pub) = pub_bytes.split_at(1952);
            let ml_dsa_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(ml_dsa_pub)
                .map_err(|_| DpopError::InvalidJwk)?;
            let ed25519_vk = ed25519_dalek::VerifyingKey::from_bytes(
                ed25519_pub.try_into().map_err(|_| DpopError::InvalidJwk)?
            ).map_err(|_| DpopError::InvalidJwk)?;
            // Composite sig: ML-DSA-65 (3309) + Ed25519 (64)
            if sig_bytes.len() != 3309 + 64 {
                return Err(DpopError::SignatureInvalid);
            }
            let (ml_sig, ed_sig) = sig_bytes.split_at(3309);
            hyprstream_rpc::crypto::pq::ml_dsa_verify(&ml_dsa_vk, signing_input.as_bytes(), ml_sig)
                .map_err(|_| DpopError::SignatureInvalid)?;
            let mut ed_sig_arr = [0u8; 64];
            ed_sig_arr.copy_from_slice(ed_sig);
            let ed_signature = ed25519_dalek::Signature::from_bytes(&ed_sig_arr);
            ed25519_vk.verify_strict(signing_input.as_bytes(), &ed_signature)
                .map_err(|_| DpopError::SignatureInvalid)?;
            DpopKey::MlDsa65Ed25519 { pub_bytes }
        }
        #[cfg(not(feature = "pq-hybrid"))]
        return Err(DpopError::WrongAlg(alg.to_owned()))
    } else {
        return Err(DpopError::WrongAlg(alg.to_owned()));
    };

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
    let jkt = key.jkt();

    Ok(DpopProof {
        key,
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

fn extract_p256_key(jwk: &serde_json::Value) -> Result<([u8; 32], [u8; 32]), DpopError> {
    if jwk["kty"].as_str() != Some("EC") || jwk["crv"].as_str() != Some("P-256") {
        return Err(DpopError::InvalidJwk);
    }
    let x_str = jwk["x"].as_str().ok_or(DpopError::InvalidJwk)?;
    let y_str = jwk["y"].as_str().ok_or(DpopError::InvalidJwk)?;
    let x_bytes: [u8; 32] = URL_SAFE_NO_PAD.decode(x_str)
        .map_err(|_| DpopError::InvalidJwk)?
        .try_into().map_err(|_| DpopError::InvalidJwk)?;
    let y_bytes: [u8; 32] = URL_SAFE_NO_PAD.decode(y_str)
        .map_err(|_| DpopError::InvalidJwk)?
        .try_into().map_err(|_| DpopError::InvalidJwk)?;
    Ok((x_bytes, y_bytes))
}

#[cfg(feature = "pq-hybrid")]
fn extract_akp_pub(jwk: &serde_json::Value) -> Result<Vec<u8>, DpopError> {
    if jwk["kty"].as_str() != Some("AKP") {
        return Err(DpopError::InvalidJwk);
    }
    let pub_str = jwk["pub"].as_str().ok_or(DpopError::InvalidJwk)?;
    URL_SAFE_NO_PAD.decode(pub_str).map_err(|_| DpopError::InvalidJwk)
}

fn verify_ed25519(key_bytes: &[u8; 32], msg: &[u8], sig_bytes: &[u8]) -> Result<(), DpopError> {
    let verifying_key = VerifyingKey::from_bytes(key_bytes)
        .map_err(|_| DpopError::InvalidJwk)?;
    let sig_array: [u8; 64] = sig_bytes.try_into()
        .map_err(|_| DpopError::SignatureInvalid)?;
    let signature = ed25519_dalek::Signature::from_bytes(&sig_array);
    use ed25519_dalek::Verifier;
    verifying_key.verify(msg, &signature)
        .map_err(|_| DpopError::SignatureInvalid)
}

fn verify_es256(x: &[u8; 32], y: &[u8; 32], msg: &[u8], sig_bytes: &[u8]) -> Result<(), DpopError> {
    use p256::ecdsa::{Signature, VerifyingKey, signature::Verifier};
    use p256::EncodedPoint;

    let point = EncodedPoint::from_affine_coordinates(x.into(), y.into(), false);
    let verifying_key = VerifyingKey::from_encoded_point(&point)
        .map_err(|_| DpopError::InvalidJwk)?;

    // JWS ES256 signatures are fixed-size r||s (64 bytes), not DER.
    let signature = Signature::from_slice(sig_bytes)
        .map_err(|_| DpopError::SignatureInvalid)?;

    verifying_key.verify(msg, &signature)
        .map_err(|_| DpopError::SignatureInvalid)
}

#[cfg(test)]
mod dpop_nonce_tests {
    //! Unit tests for the per-client nonce-enforcement bookkeeping logic.
    //! Building a full `OAuthState` requires a live `PolicyClient` (ZMQ),
    //! so we mirror the `HashMap<jkt, expiry>` shape used in `state.rs`
    //! and assert the invariants the verifier relies on.
    use std::collections::HashMap;
    use parking_lot::Mutex;

    struct NonceState {
        nonces: Mutex<HashMap<String, i64>>,
        clients: Mutex<HashMap<String, i64>>,
    }
    impl NonceState {
        fn new() -> Self {
            Self {
                nonces: Mutex::new(HashMap::new()),
                clients: Mutex::new(HashMap::new()),
            }
        }
        fn issue_nonce(&self) -> String {
            use base64::Engine as _;
            use rand::RngCore;
            let mut b = [0u8; 16];
            rand::rngs::OsRng.fill_bytes(&mut b);
            let n = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b);
            let exp = chrono::Utc::now().timestamp() + 300;
            #[allow(clippy::unwrap_used)]
            self.nonces.lock().insert(n.clone(), exp);
            n
        }
        fn verify_nonce(&self, n: &str) -> bool {
            let now = chrono::Utc::now().timestamp();
            #[allow(clippy::unwrap_used)]
            let g = self.nonces.lock();
            g.get(n).is_some_and(|&e| e > now)
        }
        fn client_requires_nonce(&self, jkt: &str) -> bool {
            let now = chrono::Utc::now().timestamp();
            #[allow(clippy::unwrap_used)]
            let mut g = self.clients.lock();
            g.retain(|_, e| *e > now);
            g.contains_key(jkt)
        }
        fn mark_nonced(&self, jkt: &str) {
            let exp = chrono::Utc::now().timestamp() + 300;
            #[allow(clippy::unwrap_used)]
            self.clients.lock().insert(jkt.to_owned(), exp);
        }
    }

    /// dpop_bootstrap_accepted_first_request:
    /// First proof from a jkt with no nonce is accepted as a bootstrap.
    #[test]
    fn dpop_bootstrap_accepted_first_request() {
        let s = NonceState::new();
        assert!(!s.client_requires_nonce("jkt-1"));
    }

    /// dpop_subsequent_without_nonce_rejected:
    /// After `mark_nonced`, the same jkt is required to present a nonce.
    #[test]
    fn dpop_subsequent_without_nonce_rejected() {
        let s = NonceState::new();
        let _n = s.issue_nonce();
        s.mark_nonced("jkt-1");
        assert!(s.client_requires_nonce("jkt-1"));
    }

    /// dpop_valid_nonce_accepted:
    /// A presented nonce that we issued and not yet expired verifies.
    #[test]
    fn dpop_valid_nonce_accepted() {
        let s = NonceState::new();
        let n = s.issue_nonce();
        s.mark_nonced("jkt-1");
        assert!(s.verify_nonce(&n));
    }

    /// dpop_unknown_nonce_rejected:
    /// A nonce we never issued is rejected.
    #[test]
    fn dpop_unknown_nonce_rejected() {
        let s = NonceState::new();
        s.mark_nonced("jkt-1");
        assert!(!s.verify_nonce("not-a-real-nonce"));
    }

    /// dpop_jkt_isolation:
    /// Different jkts have independent enforcement state.
    #[test]
    fn dpop_jkt_isolation() {
        let s = NonceState::new();
        s.mark_nonced("jkt-A");
        assert!(s.client_requires_nonce("jkt-A"));
        assert!(!s.client_requires_nonce("jkt-B"));
    }
}

/// Regression test: take a JWT-style DPoP proof signed with a composite alg,
/// strip the ML-DSA-65 sig half, confirm `verify_dpop_proof` rejects.
#[cfg(all(test, feature = "pq-hybrid"))]
mod dpop_stripping_tests {
    use super::*;
    use base64::Engine as _;
    use ed25519_dalek::Signer as _;

    #[test]
    fn dpop_composite_stripped_signature_rejected() {
        let (ml_sk, ml_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let ml_vk_bytes = hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&ml_vk);
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[42u8; 32]);
        let ed_vk_bytes = ed_sk.verifying_key().to_bytes();
        let mut composite_pub = Vec::with_capacity(ml_vk_bytes.len() + 32);
        composite_pub.extend_from_slice(&ml_vk_bytes);
        composite_pub.extend_from_slice(&ed_vk_bytes);
        let pub_b64 = URL_SAFE_NO_PAD.encode(&composite_pub);

        let header = serde_json::json!({
            "typ": "dpop+jwt",
            "alg": "ML-DSA-65-Ed25519",
            "jwk": { "kty": "AKP", "alg": "ML-DSA-65-Ed25519", "pub": pub_b64 },
        });
        let now = chrono::Utc::now().timestamp();
        let payload = serde_json::json!({
            "jti": "test-jti-1",
            "htm": "POST",
            "htu": "https://example.test/oauth/token",
            "iat": now,
        });
        let h_b64 = URL_SAFE_NO_PAD.encode(header.to_string());
        let p_b64 = URL_SAFE_NO_PAD.encode(payload.to_string());
        let signing_input = format!("{h_b64}.{p_b64}");
        let ml_sig = hyprstream_rpc::crypto::pq::ml_dsa_sign(&ml_sk, signing_input.as_bytes());
        let ed_sig = ed_sk.sign(signing_input.as_bytes());

        // Properly composed composite proof verifies.
        let mut composite = Vec::with_capacity(ml_sig.len() + 64);
        composite.extend_from_slice(&ml_sig);
        composite.extend_from_slice(&ed_sig.to_bytes());
        let good = format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(&composite));
        assert!(verify_dpop_proof(&good, "POST", "https://example.test/oauth/token", None).is_ok());

        // Stripped: only Ed25519 half (64 bytes) — composite-alg verifier
        // requires 3309 + 64 bytes. Stripping must fail.
        let stripped = format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(ed_sig.to_bytes()));
        let err = verify_dpop_proof(&stripped, "POST", "https://example.test/oauth/token", None).err();
        assert!(matches!(err, Some(DpopError::SignatureInvalid)),
            "expected SignatureInvalid, got {err:?}");
    }
}
