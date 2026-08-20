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

/// Load an existing service JWT from disk, or sign a new one if absent,
/// within `RENEW_THRESHOLD` seconds of expiry, not binding the local issuer
/// URL in BOTH `iss` and `aud`, or carrying a classical (non-PQ-covering)
/// header alg — the last two are backfills that re-mint tokens older
/// bootstraps (or older renewal paths, which stamped `iss` but not `aud`)
/// produced in shapes the composite dispatch verification now rejects.
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
    target_clearance: Option<&hyprstream_rpc::auth::mac::SecurityLabel>,
) -> Result<String> {
    let existing = super::identity_store::load_service_jwt(credentials_dir, service_name)?;
    let needs_issue = match existing {
        None => true,
        Some(ref jwt) => {
            let exp = decode_jwt_exp(jwt).unwrap_or(0);
            // Re-issue if near expiry OR if the persisted token does not bind
            // the local issuer URL in BOTH `iss` and `aud` (older bootstraps
            // minted empty-issuer service JWTs, which the #328 gate rejects
            // on the IPC/AnySigner plane — see below — and older renewal
            // paths stamped `iss` without `aud`, which strict composite
            // audience validation rejects on every dispatch) OR if the
            // persisted token's header alg is not PQ-covering (older
            // bootstraps minted classical EdDSA service JWTs, which the
            // mandatory Hybrid dispatch policy rejects at registration — the
            // same backfill shape).
            (exp - now) <= RENEW_THRESHOLD
                || !claims_bind_local_issuer(jwt, local_issuer_url)
                || !header_alg_covers_pq(jwt)
                // Full binding validation before reuse: a corrupted or
                // foreign persisted token must force reissue here, not a
                // later boot failure — verify the CA composite signature,
                // exact subject, cnf key binding, and required jti.
                || !persisted_token_binds(jwt, ca_jwt_key, service_name, service_vk)
                // The persisted token's clearance must equal the enrollment
                // target's WIRE PROJECTION (level + compartments; assurance is
                // never on the wire): a clearance-less or stale-clearance token
                // is re-minted, never reused (v16 §11 — renewal cannot preserve
                // authority the enrollment no longer grants, and an enrolled
                // credential never goes clearance-less).
                || decode_jwt_clearance(jwt)
                    != target_clearance
                        .map(|c| hyprstream_rpc::auth::mac::CredentialClearance::from_label(*c))
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
        // Stamp `aud` alongside `iss`: composite verification is strict about
        // the audience (an absent `aud` is rejected when the verifier expects
        // one), and every service's expected audience is this same issuer URL.
        claims = claims
            .with_issuer(local_issuer_url.to_owned())
            .with_audience(Some(local_issuer_url.to_owned()));
    }
    // Stamp the enrollment target clearance (v16 §11): the bootstrap/wizard
    // mint is clearance-bearing when the deployment has an enrollment
    // manifest. `None` (legacy, no manifest) mints without the claim.
    if let Some(clearance) = target_clearance {
        claims = claims.with_clearance(*clearance);
    }

    // Mint the mandatory hybrid (ML-DSA-65-Ed25519) form: the dispatch plane
    // enforces a Hybrid crypto policy with no bypass, so a classical EdDSA
    // service JWT would be rejected at the very first registration and a
    // clean deployment could never start. The post-quantum half is derived
    // deterministically from the CA JWT signing key — the same self-contained
    // derivation `BootstrapPubkey::for_service_key` uses for service
    // identities — so nothing new is persisted, protected, or rotated.
    let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(ca_jwt_key);
    let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
    Ok(hyprstream_rpc::auth::jwt::encode_service_jwt_hybrid(
        &claims, ca_jwt_key, &ca_pq, &ca_pq_vk,
    ))
}

/// Whether the persisted token binds the local issuer URL in BOTH `iss` and
/// `aud` — the shape a fresh mint produces. An empty `iss` always forces a
/// re-issue (the #328 gate rejects it); when a local issuer URL is
/// configured, a mismatched `iss` or a missing/mismatched `aud` (older
/// renewal paths stamped `iss` without `aud`) forces one too, because strict
/// composite audience validation rejects such tokens on every dispatch.
fn claims_bind_local_issuer(jwt: &str, local_issuer_url: &str) -> bool {
    let iss = decode_jwt_iss(jwt);
    if iss.as_deref().is_none_or(str::is_empty) {
        return false;
    }
    if local_issuer_url.is_empty() {
        return true;
    }
    iss.as_deref() == Some(local_issuer_url)
        && decode_jwt_aud(jwt).as_deref() == Some(local_issuer_url)
}

/// Whether the persisted token's header `alg` carries a post-quantum
/// component. Classical (`EdDSA`) tokens fail the mandatory Hybrid dispatch
/// policy and must be re-minted.
fn header_alg_covers_pq(jwt: &str) -> bool {
    matches!(
        hyprstream_rpc::auth::jwt::header_alg(jwt)
            .ok()
            .flatten()
            .as_deref(),
        Some("ML-DSA-65" | "ML-DSA-65-Ed25519")
    )
}

/// Full binding validation of a persisted service JWT before reuse: verify
/// the CA's composite signature and require the exact subject, the cnf key
/// binding to this service's verifying key, and a present `jti`. Any parse,
/// signature, or binding failure returns false (forcing reissue) — reuse
/// decides from VERIFIED fields, never unsigned decodes alone.
fn persisted_token_binds(
    jwt: &str,
    ca_jwt_key: &SigningKey,
    service_name: &str,
    service_vk: &VerifyingKey,
) -> bool {
    let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(ca_jwt_key);
    let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
    let Ok(dispatch) = hyprstream_rpc::auth::jwt::parse_composite_dispatch(jwt, &["wit+jwt"])
    else {
        return false;
    };
    let Ok(claims) = hyprstream_rpc::auth::jwt::decode_composite(
        jwt,
        &ca_pq_vk,
        &ca_jwt_key.verifying_key(),
        None,
        &dispatch,
    ) else {
        return false;
    };
    claims.sub == format!("service:{service_name}")
        && claims.cnf_key_bytes() == Some(service_vk.to_bytes())
        && claims.jti.is_some()
}

/// Decode the two-axis `clearance` claim (`[level, [compartments]]`) without
/// verifying the signature — the reuse predicate's enrollment-currency check (a
/// mismatched or missing clearance forces re-issue). Returns `None` when the
/// token carries no clearance or the claim is not a well-formed two-axis wire
/// value. The assurance axis is never on the wire; comparison is against the
/// enrollment target's wire projection (level + compartments).
fn decode_jwt_clearance(
    jwt: &str,
) -> Option<hyprstream_rpc::auth::mac::CredentialClearance> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    serde_json::from_value(value.get("clearance")?.clone()).ok()
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

/// Decode the `aud` claim without verifying the signature — the `iss` twin
/// used by the aud-backfill check in `issue_or_load_service_jwt`.
fn decode_jwt_aud(jwt: &str) -> Option<String> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    value.get("aud")?.as_str().map(ToOwned::to_owned)
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
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, 1_000_000, None,
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
            None,
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

    /// A persisted classical (EdDSA) service JWT is re-minted as hybrid on
    /// load, even when far from expiry and carrying an issuer — classical
    /// tokens fail the mandatory Hybrid dispatch policy at registration, so
    /// an upgrade heals them the same way the empty-`iss` backfill does.
    #[test]
    fn classical_persisted_jwt_is_reissued_hybrid() {
        let dir = tempfile::tempdir().unwrap();
        let ca = SigningKey::from_bytes(&[3u8; 32]);
        let svc = SigningKey::from_bytes(&[4u8; 32]);
        let issuer = "http://localhost:6791";
        // Real wall-clock time: the re-minted token is verified through
        // decode_composite below, which enforces iat/exp against the clock.
        let now = chrono::Utc::now().timestamp();

        // A classical token with issuer + far-future expiry, persisted.
        let legacy_claims =
            hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 30 * 86_400)
                .with_issuer(issuer.to_owned())
                .with_cnf_jwk(svc.verifying_key().as_bytes());
        let legacy = hyprstream_rpc::auth::jwt::encode_service_jwt(&legacy_claims, &ca);
        super::super::identity_store::write_service_jwt(dir.path(), "model", &legacy).unwrap();

        let jwt = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, None,
        )
        .unwrap();
        assert_ne!(jwt, legacy, "classical token must not be returned as-is");
        assert_eq!(
            hyprstream_rpc::auth::jwt::header_alg(&jwt).unwrap().as_deref(),
            Some("ML-DSA-65-Ed25519"),
        );

        // The re-minted token verifies against the exact derived pair, via the
        // real composite dispatch path.
        let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca);
        let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
        let dispatch =
            hyprstream_rpc::auth::jwt::parse_composite_dispatch(&jwt, &["wit+jwt"]).unwrap();
        let verified = hyprstream_rpc::auth::jwt::decode_composite(
            &jwt,
            &ca_pq_vk,
            &ca.verifying_key(),
            None,
            &dispatch,
        )
        .unwrap();
        assert_eq!(verified.sub, "service:model");
        assert_eq!(verified.iss, issuer);
    }

    /// A persisted hybrid token that carries `iss` but NO `aud` (the shape an
    /// older renewal path produced) is re-issued with both bound to the local
    /// issuer URL — strict composite audience validation rejects an aud-less
    /// token on every dispatch, so it must not be treated as current.
    #[test]
    fn audless_persisted_hybrid_jwt_is_reissued_with_audience() {
        let dir = tempfile::tempdir().unwrap();
        let ca = SigningKey::from_bytes(&[3u8; 32]);
        let svc = SigningKey::from_bytes(&[4u8; 32]);
        let issuer = "http://localhost:6791";
        let now = chrono::Utc::now().timestamp();

        // Hybrid, far from expiry, correct issuer — but no aud.
        let legacy_claims =
            hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 30 * 86_400)
                .with_issuer(issuer.to_owned())
                .with_cnf_jwk(svc.verifying_key().as_bytes());
        let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca);
        let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
        let legacy = hyprstream_rpc::auth::jwt::encode_service_jwt_hybrid(
            &legacy_claims, &ca, &ca_pq, &ca_pq_vk,
        );
        assert!(decode_jwt_aud(&legacy).is_none());
        super::super::identity_store::write_service_jwt(dir.path(), "model", &legacy).unwrap();

        let jwt = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, None,
        )
        .unwrap();
        assert_ne!(jwt, legacy, "aud-less token must not be returned as-is");
        assert_eq!(decode_jwt_iss(&jwt).as_deref(), Some(issuer));
        assert_eq!(decode_jwt_aud(&jwt).as_deref(), Some(issuer));

        // The re-minted token passes strict composite audience validation.
        let dispatch =
            hyprstream_rpc::auth::jwt::parse_composite_dispatch(&jwt, &["wit+jwt"]).unwrap();
        hyprstream_rpc::auth::jwt::decode_composite(
            &jwt,
            &ca_pq_vk,
            &ca.verifying_key(),
            Some(issuer),
            &dispatch,
        )
        .unwrap();
    }

    /// A persisted hybrid token far from expiry (with issuer) is returned
    /// as-is — the alg backfill fires only for classical tokens.
    #[test]
    fn hybrid_persisted_jwt_is_returned_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let ca = SigningKey::from_bytes(&[3u8; 32]);
        let svc = SigningKey::from_bytes(&[4u8; 32]);
        let issuer = "http://localhost:6791";
        // Real wall-clock time: reuse validation verifies the persisted
        // token's composite signature, which enforces iat/exp against now.
        let now = chrono::Utc::now().timestamp();

        let first = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, None,
        )
        .unwrap();
        super::super::identity_store::write_service_jwt(dir.path(), "model", &first).unwrap();

        let second = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, None,
        )
        .unwrap();
        assert_eq!(first, second);
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
            dir.path(), "model", &ca, &svc.verifying_key(), "http://localhost:6791", now, None,
        )
        .unwrap();
        assert_eq!(decode_jwt_iss(&jwt).as_deref(), Some("http://localhost:6791"));
    }

    fn enrollment_label() -> hyprstream_rpc::auth::mac::SecurityLabel {
        hyprstream_rpc::auth::mac::SecurityLabel::new(
            hyprstream_rpc::auth::mac::Level::Internal,
            hyprstream_rpc::auth::mac::Assurance::Classical,
            hyprstream_rpc::auth::mac::CompartmentSet::EMPTY,
        )
    }

    /// A manifest-backed mint stamps the enrollment target clearance, and a
    /// persisted token whose clearance is MISSING or STALE is re-minted, not
    /// reused — renewal never preserves authority the enrollment no longer
    /// grants, and an enrolled credential never goes clearance-less.
    #[test]
    fn clearance_is_stamped_and_stale_or_missing_clearance_reissued() {
        let dir = tempfile::tempdir().unwrap();
        let ca = SigningKey::from_bytes(&[3u8; 32]);
        let svc = SigningKey::from_bytes(&[4u8; 32]);
        let issuer = "http://localhost:6791";
        // Real wall-clock time: reuse validation verifies the persisted token's
        // composite signature (`persisted_token_binds`), which enforces
        // iat/exp against `now` — a fixed epoch would leave every persisted
        // token expired and force a reissue on the same-target reuse path.
        let now = chrono::Utc::now().timestamp();
        let target = enrollment_label();

        // Fresh mint carries the target clearance.
        let jwt = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, Some(&target),
        )
        .unwrap();
        assert_eq!(
            decode_jwt_clearance(&jwt),
            Some(hyprstream_rpc::auth::mac::CredentialClearance::from_label(target))
        );
        super::super::identity_store::write_service_jwt(dir.path(), "model", &jwt).unwrap();

        // Same target: reused unchanged.
        let second = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, Some(&target),
        )
        .unwrap();
        assert_eq!(jwt, second);

        // Enrollment NARROWED (compartment added): the persisted token is
        // stale and must be re-minted.
        let narrowed = hyprstream_rpc::auth::mac::SecurityLabel::new(
            hyprstream_rpc::auth::mac::Level::Secret,
            hyprstream_rpc::auth::mac::Assurance::Classical,
            hyprstream_rpc::auth::mac::CompartmentSet::EMPTY,
        );
        let reminted = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, Some(&narrowed),
        )
        .unwrap();
        assert_ne!(reminted, jwt, "stale-clearance token must not be reused");
        assert_eq!(
            decode_jwt_clearance(&reminted),
            Some(hyprstream_rpc::auth::mac::CredentialClearance::from_label(narrowed))
        );

        // A clearance-less legacy token under an enrollment target is
        // re-minted with the clearance.
        let legacy_claims =
            hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 30 * 86_400)
                .with_issuer(issuer.to_owned())
                .with_audience(Some(issuer.to_owned()))
                .with_cnf_jwk(svc.verifying_key().as_bytes());
        let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca);
        let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
        let legacy = hyprstream_rpc::auth::jwt::encode_service_jwt_hybrid(
            &legacy_claims, &ca, &ca_pq, &ca_pq_vk,
        );
        super::super::identity_store::write_service_jwt(dir.path(), "model", &legacy).unwrap();
        let healed = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, Some(&target),
        )
        .unwrap();
        assert_ne!(healed, legacy, "clearance-less token must not be reused");
        assert_eq!(
            decode_jwt_clearance(&healed),
            Some(hyprstream_rpc::auth::mac::CredentialClearance::from_label(target))
        );
    }

    /// Reuse validates the persisted token's binding, not just its unsigned
    /// fields: a tampered signature, a foreign cnf key, or a foreign subject
    /// all force reissue.
    #[test]
    fn persisted_token_with_broken_binding_is_reissued() {
        let dir = tempfile::tempdir().unwrap();
        let ca = SigningKey::from_bytes(&[3u8; 32]);
        let svc = SigningKey::from_bytes(&[4u8; 32]);
        let issuer = "http://localhost:6791";
        let now = chrono::Utc::now().timestamp();

        let good = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, None,
        )
        .unwrap();
        super::super::identity_store::write_service_jwt(dir.path(), "model", &good).unwrap();

        // Sanity: the good token is reused.
        let reused = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, None,
        )
        .unwrap();
        assert_eq!(good, reused);

        // Tampered signature (payload untouched, signature flipped) → the
        // composite verification fails and the token is reissued.
        let parts: Vec<&str> = good.split('.').collect();
        let mut sig = URL_SAFE_NO_PAD.decode(parts[2]).unwrap();
        sig[0] ^= 0x01;
        let tampered = format!(
            "{}.{}.{}",
            parts[0],
            parts[1],
            URL_SAFE_NO_PAD.encode(sig)
        );
        super::super::identity_store::write_service_jwt(dir.path(), "model", &tampered)
            .unwrap();
        let reissued = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, None,
        )
        .unwrap();
        assert_ne!(reissued, tampered, "tampered-signature token must be reissued");

        // Foreign cnf (signed by the CA but bound to a DIFFERENT service key)
        // → reissue.
        let other_svc = SigningKey::from_bytes(&[9u8; 32]);
        let foreign_claims =
            hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 30 * 86_400)
                .with_issuer(issuer.to_owned())
                .with_audience(Some(issuer.to_owned()))
                .with_cnf_jwk(other_svc.verifying_key().as_bytes());
        let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca);
        let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
        let foreign = hyprstream_rpc::auth::jwt::encode_service_jwt_hybrid(
            &foreign_claims, &ca, &ca_pq, &ca_pq_vk,
        );
        super::super::identity_store::write_service_jwt(dir.path(), "model", &foreign).unwrap();
        let reissued = issue_or_load_service_jwt(
            dir.path(), "model", &ca, &svc.verifying_key(), issuer, now, None,
        )
        .unwrap();
        assert_ne!(reissued, foreign, "foreign-cnf token must be reissued");
        // And the reissue binds the correct service key.
        let dispatch =
            hyprstream_rpc::auth::jwt::parse_composite_dispatch(&reissued, &["wit+jwt"]).unwrap();
        let verified = hyprstream_rpc::auth::jwt::decode_composite(
            &reissued,
            &ca_pq_vk,
            &ca.verifying_key(),
            None,
            &dispatch,
        )
        .unwrap();
        assert_eq!(
            verified.cnf_key_bytes(),
            Some(svc.verifying_key().to_bytes())
        );
    }
}
