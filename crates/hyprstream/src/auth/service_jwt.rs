//! Shared helper for service JWT issuance with load-or-renew semantics.
//!
//! Both the wizard and bootstrap manager use this function to avoid
//! duplicating the "load existing JWT, renew if within 7 days of expiry"
//! logic.

use std::path::Path;

use anyhow::Result;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{SigningKey, VerifyingKey};

const NEW_EXPIRY_TTL: i64 = 30 * 86_400;
const RENEW_THRESHOLD: i64 = 7 * 86_400;

/// Load an existing service JWT from disk, or sign a new one if absent or
/// within `RENEW_THRESHOLD` seconds of expiry.
///
/// Does NOT write to disk — callers are responsible for persisting the
/// returned JWT via `identity_store::write_service_jwt`.
pub fn issue_or_load_service_jwt(
    credentials_dir: &Path,
    service_name: &str,
    ca_jwt_key: &SigningKey,
    service_vk: &VerifyingKey,
    local_issuer_url: &str,
    now: i64,
) -> Result<String> {
    let existing = super::identity_store::load_service_jwt(credentials_dir, service_name)?;
    let needs_issue = match existing {
        None => true,
        Some(ref jwt) => {
            let exp = decode_jwt_exp(jwt).unwrap_or(0);
            // Re-issue if near expiry OR if the persisted token has no `iss`
            // (older bootstraps minted empty-issuer service JWTs, which the
            // #328 gate rejects on the IPC/AnySigner plane — see below).
            (exp - now) <= RENEW_THRESHOLD || decode_jwt_iss(jwt).is_none_or(|s| s.is_empty())
        }
    };

    if !needs_issue {
        if let Some(jwt) = existing {
            return Ok(jwt);
        }
    }

    let expiry = now + NEW_EXPIRY_TTL;
    // Set `iss` to the local issuer URL — NOT empty.
    //
    // Service JWTs are presented service->service over the local IPC (UDS)
    // plane, which the dispatch core verifies as `AnySigner`. The #328 gate
    // rejects an EMPTY `iss` from any `AnySigner` (non-`is_local_caller`)
    // caller, so an empty-issuer service JWT is rejected on exactly the plane
    // it is meant for, and (with #441's fail-closed registration) the service
    // refuses to start. Stamping the local issuer URL makes the token pass
    // `ClusterKeySource::is_trusted` (issuer == local_issuer_url -> CA key)
    // WITHOUT weakening #328: a networked peer still cannot present an
    // empty-`iss` token, and a non-local issuer is still rejected.
    let mut claims = hyprstream_rpc::auth::Claims::new(
        format!("service:{service_name}"),
        now,
        expiry,
    )
    .with_cnf_jwk(service_vk.as_bytes());
    if !local_issuer_url.is_empty() {
        claims = claims.with_issuer(local_issuer_url.to_owned());
    }

    Ok(hyprstream_rpc::auth::jwt::encode_service_jwt(&claims, ca_jwt_key))
}

fn decode_jwt_exp(jwt: &str) -> Option<i64> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    value.get("exp")?.as_i64()
}

/// Decode the `iss` claim without verifying the signature. Used only to detect
/// legacy empty-issuer service JWTs that must be re-minted (see the #328 note
/// in `issue_or_load_service_jwt`). Returns `None` if the token is malformed or
/// carries no `iss` claim.
fn decode_jwt_iss(jwt: &str) -> Option<String> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    value.get("iss")?.as_str().map(ToOwned::to_owned)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    /// A freshly issued service JWT carries the local issuer URL (NOT empty).
    /// Empty `iss` is rejected on the IPC/AnySigner plane by the #328 gate, so
    /// stamping the local issuer is what lets service-to-service registration
    /// succeed fail-closed (with #441).
    #[test]
    fn issued_service_jwt_carries_local_issuer() {
        let dir = tempfile::tempdir().unwrap();
        let ca = SigningKey::from_bytes(&[3u8; 32]);
        let svc = SigningKey::from_bytes(&[4u8; 32]);
        let issuer = "http://localhost:6791";

        let jwt = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, 1_000_000,
        )
        .unwrap();

        assert_eq!(decode_jwt_iss(&jwt).as_deref(), Some(issuer));
    }

    /// PolicyService gets a `service:policy` JWT just like every other service
    /// (#448). Bootstrap no longer skips it: it mints a CA-signed JWT whose
    /// `cnf` binds the root/CA verifying key — the same key recorded in
    /// `bootstrap_pubkeys["policy"]` and the key PolicyService actually signs
    /// RPC responses with. This test encodes that invariant directly so the
    /// "skip policy" regression cannot silently return: trust-store key ==
    /// JWT `cnf` == actual signer, all three equal to `root_key`'s verifying
    /// key.
    #[test]
    fn policy_service_jwt_is_symmetric_and_binds_root_key() {
        let dir = tempfile::tempdir().unwrap();
        // The CA: PolicyService's identity IS the root/CA key.
        let root_key = SigningKey::from_bytes(&[7u8; 32]);
        // The CA JWT signing key is purpose-derived from the root key, exactly
        // as `do_bootstrap` derives it before signing any service JWT.
        let ca_jwt_key =
            hyprstream_rpc::node_identity::derive_purpose_key(&root_key, "hyprstream-jwt-v1");
        let issuer = "http://localhost:6791";

        // Reproduce the policy branch of the bootstrap loop: service_key ==
        // root_key (NOT an independent per-service key), then issue + persist.
        let policy_key = root_key.clone();
        let policy_vk = policy_key.verifying_key();
        let jwt = issue_or_load_service_jwt(
            dir.path(),
            "policy",
            &ca_jwt_key,
            &policy_vk,
            issuer,
            1_000_000,
        )
        .unwrap();
        super::super::identity_store::write_service_jwt(dir.path(), "policy", &jwt).unwrap();

        // The JWT is persisted at policy/service-jwt (not skipped).
        let loaded =
            super::super::identity_store::load_service_jwt(dir.path(), "policy").unwrap();
        assert_eq!(loaded.as_deref(), Some(jwt.as_str()));

        // Subject is service:policy.
        let payload_b64 = jwt.split('.').nth(1).unwrap();
        let payload = URL_SAFE_NO_PAD.decode(payload_b64).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(value.get("sub").and_then(|v| v.as_str()), Some("service:policy"));

        // cnf.jwk binds the root verifying key — what the trust store records
        // for "policy" and what PolicyService signs with.
        let cnf = value.get("cnf").and_then(|v| v.get("jwk"));
        assert!(cnf.is_some(), "policy service JWT must carry a cnf.jwk");
        // The JWK thumbprint representation here is the raw 32-byte verifying
        // key (x); compare against root_key.verifying_key() bytes.
        let x = cnf
            .and_then(|v| v.get("x"))
            .and_then(|v| v.as_str())
            .unwrap();
        let x_bytes = URL_SAFE_NO_PAD.decode(x).unwrap();
        assert_eq!(
            x_bytes.as_slice(),
            root_key.verifying_key().as_bytes(),
            "cnf.jwk must bind root_key.verifying_key(), the key bootstrap registers for policy"
        );

        // And that equals what bootstrap would register in bootstrap_pubkeys.
        assert_eq!(
            root_key.verifying_key().as_bytes(),
            policy_vk.as_bytes()
        );
    }

    /// A persisted legacy empty-issuer token is re-minted with the issuer set,
    /// rather than being returned as-is — so an upgrade heals the #328 mismatch
    /// on the next wizard/bootstrap run without manual intervention.
    #[test]
    fn empty_issuer_legacy_jwt_is_reissued_with_issuer() {
        let dir = tempfile::tempdir().unwrap();
        let ca = SigningKey::from_bytes(&[3u8; 32]);
        let svc = SigningKey::from_bytes(&[4u8; 32]);

        // Mint a legacy token with NO issuer and a far-future expiry, persist it.
        let now = 1_000_000;
        let legacy_claims =
            hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 30 * 86_400)
                .with_cnf_jwk(svc.verifying_key().as_bytes());
        let legacy = hyprstream_rpc::auth::jwt::encode_service_jwt(&legacy_claims, &ca);
        assert!(decode_jwt_iss(&legacy).is_none_or(|s| s.is_empty()));
        super::super::identity_store::write_service_jwt(dir.path(), "model", &legacy).unwrap();

        // Even though it is far from expiry, the empty-iss legacy token forces a
        // re-issue carrying the issuer.
        let jwt = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), "http://localhost:6791", now,
        )
        .unwrap();
        assert_eq!(decode_jwt_iss(&jwt).as_deref(), Some("http://localhost:6791"));
    }
}
