//! Internal execution work orders for the Model→Inference single-service
//! boundary (user-approved D5, PR1678).
//!
//! Model is the authorization authority for Model-routed operations. Its
//! pinned, allocated Inference instances are subprocessors: they execute work
//! only for their owning controller and never form an independent authority
//! domain. When Model forwards an already-authorized operation to one of its
//! pinned instances, it mints a short-lived, Model-signed **internal work
//! token** (`typ: "iw+jwt"`) instead of relaying the caller's credential:
//!
//! - the token is bound to the pinned controller key (`cnf.jwk`) and is
//!   verified against that pinned key — not against the CA key source;
//! - `aud` names exactly one allocated instance; `tenant`/`model` must match
//!   that instance's binding;
//! - `resource`/`operation` pin the exact downstream dispatch coordinate the
//!   controller already authorized at ingress;
//! - `exp` is capped by the caller's own expiry and a hard 120 s lifetime;
//!   `jti` is admitted once per worker process;
//! - the original caller's verified claims snapshot rides along so
//!   subject-keyed TTT state, stream ownership, and ledger accounting keep
//!   attributing to the ORIGINAL caller — never to Model.
//!
//! **This token is authority nowhere else.** The generic `verify_claims`
//! pipeline rejects `iw+jwt` at every service (the trait hook defaults to
//! deny); only a service operating a pinned execution boundary overrides the
//! hook. It is never a bearer for Policy, Registry, Discovery, OAI, or any
//! other control-plane surface, and it is never a substitute for a holder
//! credential crossing an EXTERNAL boundary.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::Result;
use ed25519_dalek::{Signer as _, Verifier as _, VerifyingKey};
use serde::{Deserialize, Serialize};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;

use super::claims::{Cnf, CnfJwk, Claims};
use super::jwt::{self, JwkThumbprintInput};

/// URL-safe base64 without padding — the JWT/JWK encoding used by the
/// existing `Claims`/`cnf` machinery.
fn b64(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

/// JOSE `typ` for internal execution work orders.
pub const INTERNAL_WORK_JWT_TYP: &str = "iw+jwt";

/// Informational issuer marker. Verification is pinned-key, not issuer-based;
/// a non-empty issuer deliberately avoids the #328 empty-iss in-process
/// shortcut so a future subprocess/UDS leg needs no locality exception.
pub const INTERNAL_WORK_ISSUER: &str = "hyprstream://model-internal-work";

/// Hard upper bound on one work order's lifetime. Model caps `exp` by the
/// caller's own expiry as well; this bound keeps a leaked token's window
/// short regardless of the caller credential.
pub const MAX_LIFETIME_SECS: i64 = 120;

/// Maximum accepted clock skew between mint (`iat`) and admission.
pub const MAX_IAT_SKEW_SECS: i64 = 30;

/// The exact downstream dispatch coordinate one work order authorizes.
///
/// Model mints one token per forwarded operation; the worker's per-method PEP
/// requires an exact match before executing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InternalWorkScope {
    pub resource: String,
    pub operation: String,
}

impl InternalWorkScope {
    pub fn new(resource: impl Into<String>, operation: impl Into<String>) -> Self {
        Self { resource: resource.into(), operation: operation.into() }
    }
}

/// Claims of one internal execution work order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InternalWorkClaims {
    pub iss: String,
    /// The ORIGINAL caller's verified subject (resolved at Model ingress).
    pub sub: String,
    /// The one allocated instance this work order is valid for.
    pub aud: String,
    /// The instance's authority-verified tenant.
    pub tenant: String,
    /// The instance's model reference.
    pub model: String,
    /// Exact downstream resource coordinate (e.g. `inference:generateStream`).
    pub resource: String,
    /// Exact downstream operation scope (e.g. `infer`).
    pub operation: String,
    pub iat: i64,
    pub exp: i64,
    pub jti: String,
    /// Holder binding: the pinned controller's Ed25519 key.
    pub cnf: Cnf,
    /// Original caller's authenticated pairwise DID (ledger attribution).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_did: Option<String>,
    /// The caller's verified claims snapshot from Model ingress. The raw
    /// bearer string is `#[serde(skip)]` on `Claims` and is never embedded.
    pub caller: Claims,
}

impl InternalWorkClaims {
    /// The confirmation JWK for a controller verifying key.
    #[must_use]
    pub fn controller_cnf(controller: &VerifyingKey) -> Cnf {
        Cnf {
            jwk: Some(CnfJwk {
                kty: "OKP".to_owned(),
                crv: "Ed25519".to_owned(),
                x: b64(controller.as_bytes()),
            }),
            jkt: None,
            hs_signer_suite: None,
        }
    }
}

/// The verified work order installed on an [`crate::service::EnvelopeContext`].
#[derive(Clone, Debug)]
pub struct InternalWorkContext {
    /// Original caller's subject — what subject-keyed state, streams, and
    /// accounting must attribute to.
    pub subject: crate::envelope::Subject,
    /// Original caller's authenticated pairwise DID, when the caller presented
    /// one (ledger owner attribution).
    pub owner_did: Option<String>,
    /// Bound downstream resource coordinate.
    pub resource: String,
    /// Bound downstream operation scope.
    pub operation: String,
    /// Instance tenant.
    pub tenant: String,
    /// Instance model reference.
    pub model: String,
    /// Single-use admission id.
    pub jti: String,
}

/// Encode and sign a work order (`typ: "iw+jwt"`, pinned Ed25519).
pub fn encode_internal_work(claims: &InternalWorkClaims, signing_key: &ed25519_dalek::SigningKey) -> String {
    let kid = jwt::kid_for_key(signing_key);
    let header = format!(
        r#"{{"alg":"EdDSA","typ":"{INTERNAL_WORK_JWT_TYP}","kid":"{kid}"}}"#
    );
    let header_b64 = b64(header.as_bytes());
    let payload_json = serde_json::to_string(claims).unwrap_or_else(|_err| {
        #[cfg(not(target_arch = "wasm32"))]
        tracing::error!("internal work claims serialization failed: {_err}");
        "{}".to_owned()
    });
    let payload_b64 = b64(payload_json.as_bytes());
    let signing_input = format!("{header_b64}.{payload_b64}");
    let signature = signing_key.sign(signing_input.as_bytes());
    format!("{signing_input}.{}", b64(signature.to_bytes().as_ref()))
}

/// Verify a work order against the PINNED controller key (never a CA key
/// source) and bind it to one expected instance audience.
///
/// Pinned-alg Ed25519 only; `typ` must be exactly [`INTERNAL_WORK_JWT_TYP`];
/// the declared `kid` must name the pinned key (invariant I2); freshness and
/// the hard lifetime bound are enforced here.
pub fn verify_internal_work(
    token: &str,
    controller: &VerifyingKey,
    expected_aud: &str,
    now: i64,
) -> Result<InternalWorkClaims> {
    let parts: Vec<&str> = token.split('.').collect();
    anyhow::ensure!(parts.len() == 3, "internal work token is malformed");
    let header = jwt::parse_protected_header(token)?;
    anyhow::ensure!(
        header.typ == INTERNAL_WORK_JWT_TYP,
        "internal work token has wrong typ {:?}",
        header.typ
    );
    anyhow::ensure!(
        header.alg == "EdDSA",
        "internal work token must be pinned EdDSA"
    );
    let expected_kid = jwt::jwk_thumbprint(&JwkThumbprintInput::Ed25519 {
        x: controller.as_bytes(),
    });
    anyhow::ensure!(
        header.kid == expected_kid,
        "internal work token kid does not name the pinned controller key"
    );
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|e| anyhow::anyhow!("internal work payload decode failed: {e}"))?;
    let claims: InternalWorkClaims = serde_json::from_slice(&payload)
        .map_err(|e| anyhow::anyhow!("internal work claims parse failed: {e}"))?;
    anyhow::ensure!(
        claims.iss == INTERNAL_WORK_ISSUER,
        "internal work token issuer is not the internal work authority"
    );
    anyhow::ensure!(
        claims.exp > now,
        "internal work token is expired"
    );
    anyhow::ensure!(
        (now - claims.iat).abs() <= MAX_IAT_SKEW_SECS,
        "internal work token iat is outside the admission skew window"
    );
    anyhow::ensure!(
        claims.exp - claims.iat <= MAX_LIFETIME_SECS,
        "internal work token exceeds the maximum internal lifetime"
    );
    anyhow::ensure!(
        claims.aud == expected_aud,
        "internal work token audience does not name this instance"
    );
    anyhow::ensure!(!claims.tenant.is_empty(), "internal work token carries no tenant");
    anyhow::ensure!(!claims.model.is_empty(), "internal work token carries no model");
    anyhow::ensure!(!claims.jti.is_empty(), "internal work token carries no jti");
    // Holder binding: the claims must name the pinned controller key itself.
    anyhow::ensure!(
        claims
            .cnf
            .jwk
            .as_ref()
            .is_some_and(|jwk| jwk.x == b64(controller.as_bytes())),
        "internal work token cnf does not bind the pinned controller key"
    );
    // Signature LAST: parse/structure failures are cheaper than verification.
    let signature: [u8; 64] = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|e| anyhow::anyhow!("internal work signature decode failed: {e}"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("internal work signature has wrong length"))?;
    let signing_input = format!("{}.{}", parts[0], parts[1]);
    controller
        .verify(signing_input.as_bytes(), &ed25519_dalek::Signature::from_bytes(&signature))
        .map_err(|_| anyhow::anyhow!("internal work token signature does not verify against the pinned controller key"))?;
    Ok(claims)
}

// ── bounded single-use jti admission ────────────────────────────────────────

const JTI_CACHE_CAPACITY: usize = 10_000;

fn jti_cache() -> &'static parking_lot::Mutex<HashMap<String, i64>> {
    static CACHE: OnceLock<parking_lot::Mutex<HashMap<String, i64>>> = OnceLock::new();
    CACHE.get_or_init(|| parking_lot::Mutex::new(HashMap::new()))
}

/// Admit a work-order `jti` exactly once. Entries expire at the token's own
/// `exp`; the cache is capacity-bounded and fails closed when full. This is
/// worker-local defense in depth ON TOP of the envelope replay gates, not a
/// replacement for them.
pub fn admit_internal_work_jti_once(jti: &str, exp: i64, now: i64) -> bool {
    let mut cache = jti_cache().lock();
    admit_once_in(&mut cache, jti, exp, now)
}

/// The admission decision over one bounded map (isolated for tests).
fn admit_once_in(
    cache: &mut HashMap<String, i64>,
    jti: &str,
    exp: i64,
    now: i64,
) -> bool {
    cache.retain(|_, entry_exp| *entry_exp > now);
    if cache.contains_key(jti) {
        return false;
    }
    if cache.len() >= JTI_CACHE_CAPACITY {
        return false;
    }
    cache.insert(jti.to_owned(), exp);
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn controller() -> SigningKey {
        SigningKey::from_bytes(&[0xA5; 32])
    }

    fn fixture_claims(signer: &SigningKey, now: i64) -> InternalWorkClaims {
        InternalWorkClaims {
            iss: INTERNAL_WORK_ISSUER.to_owned(),
            sub: "alice".to_owned(),
            aud: "inference-fixture01".to_owned(),
            tenant: "tenant-a".to_owned(),
            model: "qwen3:main".to_owned(),
            resource: "inference:generateStream".to_owned(),
            operation: "infer".to_owned(),
            iat: now,
            exp: now + 60,
            jti: "jti-1".to_owned(),
            cnf: InternalWorkClaims::controller_cnf(&signer.verifying_key()),
            owner_did: Some("did:key:z6Mk".to_owned()),
            caller: Claims::new("alice".to_owned(), now, now + 3600)
                .with_tenant("tenant-a".to_owned()),
        }
    }

    #[test]
    fn mint_verify_round_trip_binds_instance_and_controller() {
        let signer = controller();
        let now = chrono::Utc::now().timestamp();
        let claims = fixture_claims(&signer, now);
        let token = encode_internal_work(&claims, &signer);
        let verified = verify_internal_work(&token, &signer.verifying_key(), &claims.aud, now)
            .expect("a well-formed work order must verify against its pinned controller");
        assert_eq!(verified.sub, "alice");
        assert_eq!(verified.tenant, "tenant-a");
        assert_eq!(verified.model, "qwen3:main");
        assert_eq!(verified.resource, "inference:generateStream");
        assert_eq!(verified.operation, "infer");
        assert_eq!(verified.owner_did.as_deref(), Some("did:key:z6Mk"));
        assert_eq!(verified.caller.sub, "alice");
        assert!(
            verified.caller.token.is_none(),
            "the caller snapshot must never embed the raw bearer"
        );
    }

    #[test]
    fn foreign_signer_never_verifies() {
        let signer = controller();
        let attacker = SigningKey::from_bytes(&[0xA6; 32]);
        let now = chrono::Utc::now().timestamp();
        let token = encode_internal_work(&fixture_claims(&signer, now), &attacker);
        let err = verify_internal_work(&token, &signer.verifying_key(), "inference-fixture01", now)
            .expect_err("a foreign key cannot mint for a pinned controller");
        assert!(err.to_string().contains("pinned controller key"), "{err:#}");
    }

    #[test]
    fn wrong_audience_tenant_or_model_denies() {
        let signer = controller();
        let now = chrono::Utc::now().timestamp();
        let token = encode_internal_work(&fixture_claims(&signer, now), &signer);
        let err = verify_internal_work(&token, &signer.verifying_key(), "inference-other", now)
            .expect_err("a work order for another instance must deny");
        assert!(err.to_string().contains("audience"), "{err:#}");
        let claims = fixture_claims(&signer, now);
        let mut other_instance = claims.clone();
        other_instance.tenant = "tenant-b".to_owned();
        let token2 = encode_internal_work(&other_instance, &signer);
        // The tenant/model mismatch is adjudicated by the worker against its
        // own binding; verification returns the claims so the adapter can
        // fail closed. Assert the values arrive un-tampered (signature holds).
        let verified = verify_internal_work(&token2, &signer.verifying_key(), &claims.aud, now)
            .expect("signature remains valid for a different tenant binding");
        assert_eq!(verified.tenant, "tenant-b");
    }

    #[test]
    fn expired_future_iat_and_oversized_lifetime_denies() {
        let signer = controller();
        let now = chrono::Utc::now().timestamp();

        let mut expired = fixture_claims(&signer, now);
        expired.exp = now - 1;
        let token = encode_internal_work(&expired, &signer);
        let err = verify_internal_work(&token, &signer.verifying_key(), &expired.aud, now)
            .expect_err("expired work orders must deny");
        assert!(err.to_string().contains("expired"), "{err:#}");

        let mut future = fixture_claims(&signer, now);
        future.iat = now + MAX_IAT_SKEW_SECS + 1;
        let token = encode_internal_work(&future, &signer);
        let err = verify_internal_work(&token, &signer.verifying_key(), &future.aud, now)
            .expect_err("future-minted work orders must deny");
        assert!(err.to_string().contains("skew"), "{err:#}");

        let mut long = fixture_claims(&signer, now);
        long.exp = now + MAX_LIFETIME_SECS + 1;
        let token = encode_internal_work(&long, &signer);
        let err = verify_internal_work(&token, &signer.verifying_key(), &long.aud, now)
            .expect_err("oversized lifetimes must deny");
        assert!(err.to_string().contains("maximum internal lifetime"), "{err:#}");
    }

    #[test]
    fn wrong_typ_or_alg_denies() {
        let signer = controller();
        let now = chrono::Utc::now().timestamp();
        let claims = fixture_claims(&signer, now);
        let kid = jwt::kid_for_key(&signer);
        // at+jwt framing must never be admitted as internal work.
        let claims_json = serde_json::to_string(&claims).unwrap();
        let header = format!(r#"{{"alg":"EdDSA","typ":"at+jwt","kid":"{kid}"}}"#);
        let signing_input = format!("{}.{}", b64(header.as_bytes()), b64(claims_json.as_bytes()));
        let sig = signer.sign(signing_input.as_bytes());
        let at_jwt = format!("{signing_input}.{}", b64(sig.to_bytes().as_ref()));
        let err = verify_internal_work(&at_jwt, &signer.verifying_key(), &claims.aud, now)
            .expect_err("an at+jwt framing is not an internal work order");
        assert!(err.to_string().contains("typ"), "{err:#}");
    }

    #[test]
    fn jti_is_single_use_and_cache_is_bounded() {
        let now = chrono::Utc::now().timestamp();
        let id = format!("jti-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
        assert!(admit_internal_work_jti_once(&id, now + 60, now));
        assert!(!admit_internal_work_jti_once(&id, now + 60, now), "replay must deny");
        // Past its exp the entry is purged, so a fresh order with the same id
        // (impossible for a real mint, but proves the release) admits again.
        assert!(admit_internal_work_jti_once(&id, now + 120, now + 61));
        // Capacity bound fails closed — decided against an ISOLATED map so the
        // process-global admission cache is not poisoned for sibling tests.
        let mut isolated: HashMap<String, i64> = HashMap::new();
        let mut i = 0u64;
        while isolated.len() < JTI_CACHE_CAPACITY {
            isolated.insert(format!("fill-{i}"), now + 3600);
            i += 1;
        }
        assert!(!admit_once_in(&mut isolated, "overflow", now + 60, now));
    }
}
