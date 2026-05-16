//! JWT token implementation.
//!
//! Re-exports EdDSA signing from hyprstream-rpc and adds ES256 (P-256) signing.

pub use hyprstream_rpc::auth::{
    decode, decode_with_key, encode, encode_service_jwt, Claims, JwtError,
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use p256::ecdsa::{SigningKey as Es256SigningKey, signature::Signer};

/// Encode and sign an OAuth 2.0 access token with ES256 (P-256 ECDSA).
///
/// Produces a standard `at+jwt` with `alg: "ES256"` in the JOSE header.
/// The `kid` is the RFC 7638 JWK Thumbprint of the P-256 public key.
/// Automatically assigns a `jti` if not already set.
pub fn encode_es256(claims: &Claims, signing_key: &Es256SigningKey) -> String {
    let claims = if claims.jti.is_some() {
        std::borrow::Cow::Borrowed(claims)
    } else {
        std::borrow::Cow::Owned(claims.clone().with_jti())
    };
    let kid = es256_kid(signing_key);
    let header = format!(r#"{{"alg":"ES256","typ":"at+jwt","kid":"{}"}}"#, kid);
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());
    let payload_json = serde_json::to_string(claims.as_ref()).unwrap_or_else(|_e| {
        tracing::error!("JWT claims serialization failed: {}", _e);
        "{}".to_owned()
    });
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());
    let signing_input = format!("{header_b64}.{payload_b64}");
    let signature: p256::ecdsa::Signature = signing_key.sign(signing_input.as_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());
    format!("{signing_input}.{sig_b64}")
}

fn es256_coordinates(signing_key: &Es256SigningKey) -> ([u8; 32], [u8; 32]) {
    let vk = signing_key.verifying_key();
    let point = vk.to_encoded_point(false);
    // Uncompressed P-256 point always has x and y (32 bytes each).
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(point.x().map(AsRef::as_ref).unwrap_or(&[0u8; 32]));
    y.copy_from_slice(point.y().map(AsRef::as_ref).unwrap_or(&[0u8; 32]));
    (x, y)
}

/// Compute the RFC 7638 JWK Thumbprint for a P-256 signing key.
pub fn es256_kid(signing_key: &Es256SigningKey) -> String {
    let (x, y) = es256_coordinates(signing_key);
    hyprstream_rpc::auth::jwk_thumbprint(&hyprstream_rpc::auth::JwkThumbprintInput::Es256 { x: &x, y: &y })
}

/// Build a JWK (serde_json::Value) for JWKS publishing from a P-256 signing key.
pub fn es256_jwk(signing_key: &Es256SigningKey) -> serde_json::Value {
    let (x, y) = es256_coordinates(signing_key);
    let kid = es256_kid(signing_key);
    serde_json::json!({
        "kty": "EC",
        "crv": "P-256",
        "use": "sig",
        "alg": "ES256",
        "kid": kid,
        "x": URL_SAFE_NO_PAD.encode(x),
        "y": URL_SAFE_NO_PAD.encode(y),
    })
}

/// Generate a new random P-256 signing key.
pub fn generate_es256_key() -> Es256SigningKey {
    Es256SigningKey::random(&mut rand::rngs::OsRng)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn es256_roundtrip() {
        let key = generate_es256_key();
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999);
        let token = encode_es256(&claims, &key);

        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);

        let header_bytes = URL_SAFE_NO_PAD.decode(parts[0]).unwrap();
        let header: serde_json::Value = serde_json::from_slice(&header_bytes).unwrap();
        assert_eq!(header["alg"], "ES256");
        assert_eq!(header["typ"], "at+jwt");
        assert!(header["kid"].as_str().unwrap().len() == 43);

        let payload_bytes = URL_SAFE_NO_PAD.decode(parts[1]).unwrap();
        let decoded: Claims = serde_json::from_slice(&payload_bytes).unwrap();
        assert_eq!(decoded.sub, "alice");
        assert!(decoded.jti.is_some());

        // Verify signature
        use p256::ecdsa::{Signature, signature::Verifier};
        let sig_bytes = URL_SAFE_NO_PAD.decode(parts[2]).unwrap();
        let signature = Signature::from_slice(&sig_bytes).unwrap();
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        key.verifying_key().verify(signing_input.as_bytes(), &signature).unwrap();
    }

    #[test]
    fn es256_kid_deterministic() {
        let key = generate_es256_key();
        assert_eq!(es256_kid(&key), es256_kid(&key));
        assert_eq!(es256_kid(&key).len(), 43);
    }

    #[test]
    fn es256_jwk_structure() {
        let key = generate_es256_key();
        let jwk = es256_jwk(&key);
        assert_eq!(jwk["kty"], "EC");
        assert_eq!(jwk["crv"], "P-256");
        assert_eq!(jwk["alg"], "ES256");
        assert_eq!(jwk["use"], "sig");
        assert!(jwk["kid"].as_str().unwrap().len() == 43);
        assert!(jwk["x"].as_str().is_some());
        assert!(jwk["y"].as_str().is_some());
    }

    #[test]
    fn es256_auto_assigns_jti() {
        let key = generate_es256_key();
        let claims = Claims::new("bob".to_owned(), 0, 9_999_999_999);
        assert!(claims.jti.is_none());
        let token = encode_es256(&claims, &key);

        let parts: Vec<&str> = token.split('.').collect();
        let payload_bytes = URL_SAFE_NO_PAD.decode(parts[1]).unwrap();
        let decoded: Claims = serde_json::from_slice(&payload_bytes).unwrap();
        assert!(decoded.jti.is_some());
    }
}
