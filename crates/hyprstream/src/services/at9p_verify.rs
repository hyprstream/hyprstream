//! Credential-free HTTPS face for `did:at9p` login-assertion verification (#1114).
//!
//! An external web app that cannot speak Cap'n Proto and holds no mesh
//! credentials POSTs a [`LoginAssertionRequest`] to `POST /at9p/verify` and
//! gets back whether the assertion is authentic and live. This module is a
//! thin JSON translator over the pure, fully-tested core in
//! [`hyprstream_pds::at9p_login`]; it adds no verification logic of its own.
//!
//! # Where this is mounted, and why
//!
//! This is a **standalone credential-free HTTP face on its own listener**, not
//! a route on an existing service. The reasoning is forced by the current
//! state of the tree (recorded for #1135):
//!
//! - **Not on OAuth.** OAuth is the only genuinely dual-stack service today
//!   (HTTP + Rep via `serve_bridged`), but it is an authorization server.
//!   Mounting a public verification route there would either inherit OAuth's
//!   session/cookie middleware (violating the credential-free requirement) or
//!   require a special-case bypass inside OAuth's router — exactly the "first
//!   convenient exception" #1165 warns decays the guarantee.
//! - **Not on Discovery or Policy.** Both are pure Cap'n Proto RPC services
//!   with no HTTP face, and reaching them from the web app would re-introduce
//!   the mesh-credential problem this endpoint exists to avoid. The verifier
//!   does not need a live RPC channel: the pure GATE core is I/O-free.
//! - **No `SocketKind::Http`.** `SocketKind` is `Req | Rep | Quic`; there is
//!   no HTTP variant, so this face is **not announceable in the RPC
//!   registry** and holds no mesh credentials by construction. It sits
//!   outside the trust store entirely — no service signing key, no service
//!   JWT, no `register_service_key`, no `registrations()`. That is the
//!   security argument: a face the mesh cannot reach cannot leak mesh
//!   authority, and a face with no credential-bearing state cannot require
//!   credentials. This is the #1165 "credential-free origin on its own
//!   listener" pattern, applied to verification.
//!
//! # Credential-free origin — enforced, not conventional
//!
//! [`credential_free_router`] is constructed from a [`VerifyFaceState`] that
//! contains **only** verification tunables (skew window, challenge size cap)
//! and a clock — no trust store, no JWT verifying key, no policy client, no
//! session store. The handler extractors are `State<VerifyFaceState>` and the
//! JSON body only; there is no `Authorization`/`Cookie` extractor and no
//! auth middleware layered on the router. A test asserts a request bearing
//! bogus `Authorization` and `Cookie` headers yields the same result as one
//! with none — if a future change layers a credential-consuming middleware,
//! that test fails (see `credential_free_origin_ignores_credentials`).
//!
//! # What the response does and does not assert
//!
//! See [`hyprstream_pds::at9p_login`] for the authoritative statement; the
//! JSON response mirrors it: a `verified: true` result asserts the GATE,
//! liveness, and (when claimed) bidirectional aliasing all passed, and
//! nothing about reachability, account standing, or authorization. The
//! endpoint mints no credential and sets no cookie.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use hyprstream_pds::at9p_login::{
    ClassicalClaim, Clock, LoginAssertion, verify_login_assertion,
};
use hyprstream_rpc::auth::mac::Assurance;
use hyprstream_rpc::error::RpcError;
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_service::Spawnable;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tracing::info;

use crate::config::{At9pVerifyConfig, TlsConfig};
use crate::server::tls::{resolve_rustls_config, serve_app};

/// Service name for logging (not registered in the RPC mesh — see module docs).
pub const SERVICE_NAME: &str = "at9p_verify";

/// Default freshness window for a liveness proof (seconds, symmetric skew).
pub const DEFAULT_MAX_SKEW_SECONDS: u64 = 300;

/// Default cap on the challenge string length (bytes). Defends against a holder
/// coaxing the web app into issuing an enormous challenge that this endpoint
/// would then have to hash; the web app should cap its own challenges too.
pub const DEFAULT_MAX_CHALLENGE_BYTES: usize = 256;

/// The non-credential state carried by the credential-free router.
///
/// Deliberately holds **only** verification tunables + a clock. There is no
/// field here for a trust store, JWT key, policy client, or session store,
/// so the router built from it cannot require or consume credentials. Adding
/// such a field is itself a review flag that the credential-free contract is
/// being broken.
#[derive(Clone, Copy, Debug)]
pub struct VerifyFaceState {
    /// Symmetric freshness window for the liveness `issued_at` (seconds).
    pub max_skew_seconds: u64,
    /// Maximum accepted challenge length (bytes).
    pub max_challenge_bytes: usize,
}

/// The verifier's wall clock. Injected so tests are deterministic; in
/// production this reads `SystemTime::now`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_unix_seconds(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

// ─── Wire types ─────────────────────────────────────────────────────────────

/// The classical-alias claim (advisory strings only — never key material).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalClaimWire {
    pub classical_did: String,
    /// The `alsoKnownAs` entry the classical DID document names for its at9p
    /// alias (the classical→at9p leg). Advisory.
    pub classical_aka_at9p: String,
}

/// The login assertion a web app POSTs. Capsule and signature are base64
/// (standard alphabet) over the raw bytes the pure core expects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginAssertionRequest {
    /// `did:at9p:<cid512>` — the claimed identity.
    pub did: String,
    /// Base64 canonical DAG-CBOR bytes of the genesis capsule.
    pub capsule_b64: String,
    /// The fresh challenge the web app issued.
    pub challenge: String,
    /// Holder's signing instant, Unix seconds.
    pub issued_at: u64,
    /// Base64 Ed25519 signature over the liveness binding.
    pub signature_b64: String,
    /// Optional classical alias claim; when present, mutual #905 §2/§6
    /// aliasing is required or verification fails closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classical: Option<ClassicalClaimWire>,
}

/// A successful verification response. Mirrors the assertion statement in
/// [`hyprstream_pds::at9p_login`]: this is a verification result, not a
/// credential, and asserts nothing about reachability or authorization.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyOk {
    pub verified: bool,
    /// The content-verified DID (recomputed from the capsule, not the claim).
    pub did: String,
    /// `"classical"` (no alias claimed) or `"pq-hybrid"` (mutual alias).
    pub assurance: &'static str,
    /// Whether a classical alias was mutually attested.
    pub classical_alias: Option<String>,
}

/// A verification failure. `code` is a stable machine-readable reason; `error`
/// is a human-readable string that **never** echoes attacker-controlled bytes
/// beyond the DID identifier.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyErr {
    pub verified: bool,
    pub code: &'static str,
    pub error: String,
}

impl VerifyErr {
    fn new(code: &'static str, error: impl Into<String>) -> Self {
        Self {
            verified: false,
            code,
            error: error.into(),
        }
    }
}

/// Every verification failure is a 400 — the endpoint never returns 200 on a
/// failed check. The stable machine-readable reason lives in `VerifyErr::code`;
/// the status is uniformly `BAD_REQUEST` (malformed/invalid assertion).
fn err_status(_code: &str) -> StatusCode {
    StatusCode::BAD_REQUEST
}

/// The credential-free router over which the verification endpoint is served.
///
/// Constructed solely from [`VerifyFaceState`] (tunables + clock); no
/// credential-bearing state is accepted, so none can be layered. The single
/// route is `POST /at9p/verify`.
pub fn credential_free_router(state: VerifyFaceState) -> Router {
    Router::new().route("/at9p/verify", post(verify_handler)).with_state(state)
}

/// The handler. Extracts only tunable state + the JSON body — no headers, no
/// cookies, no auth extension. Any verification failure is a 400 with a
/// `VerifyErr` body; the endpoint never returns 200 for a failed check.
///
/// Returns a single [`Response`] (not `impl IntoResponse`) so the success
/// (`VerifyOk`) and failure (`VerifyErr`) arms — different body types — share
/// one return type via [`IntoResponse::into_response`].
async fn verify_handler(
    State(state): State<VerifyFaceState>,
    Json(req): Json<LoginAssertionRequest>,
) -> Response {
    // Bound the challenge before doing any cryptographic work.
    if req.challenge.len() > state.max_challenge_bytes {
        return (
            err_status("challenge_too_long"),
            Json(VerifyErr::new(
                "challenge_too_long",
                format!("challenge exceeds max {} bytes", state.max_challenge_bytes),
            )),
        )
            .into_response();
    }

    let capsule_bytes = match STANDARD.decode(&req.capsule_b64) {
        Ok(b) => b,
        Err(_) => {
            return (err_status("bad_capsule"), Json(VerifyErr::new(
                "bad_capsule",
                "capsule_b64 is not valid base64",
            )))
                .into_response();
        }
    };
    let signature = match STANDARD.decode(&req.signature_b64) {
        Ok(b) => b,
        Err(_) => {
            return (err_status("bad_signature"), Json(VerifyErr::new(
                "bad_signature",
                "signature_b64 is not valid base64",
            )))
                .into_response();
        }
    };

    let classical = req.classical.as_ref().map(|c| ClassicalClaim {
        classical_did: c.classical_did.as_str(),
        classical_aka_at9p: c.classical_aka_at9p.as_str(),
    });

    let assertion = LoginAssertion {
        did: req.did.as_str(),
        capsule_bytes: &capsule_bytes,
        challenge: req.challenge.as_str(),
        issued_at: req.issued_at,
        liveness_signature: &signature,
        classical,
    };

    match verify_login_assertion(&assertion, &SystemClock, state.max_skew_seconds) {
        Ok(v) => {
            let assurance = match v.assurance {
                Assurance::PqHybrid => "pq-hybrid",
                Assurance::Classical => "classical",
                Assurance::Unverified => "unverified",
            };
            let classical_alias = v
                .classical
                .as_ref()
                .map(|a| a.classical_did.as_str().to_owned());
            (
                StatusCode::OK,
                Json(VerifyOk {
                    verified: true,
                    did: v.did.as_str().to_owned(),
                    assurance,
                    classical_alias,
                }),
            )
                .into_response()
        }
        Err(e) => {
            // Classify the failure for the stable `code` field. The error
            // string is safe to surface: it names the gate that rejected, not
            // attacker-supplied capsule contents.
            let msg = format!("{e:#}");
            let code: &'static str = if msg.contains("hash-gate") || msg.contains("GATE rejected")
            {
                "hash_mismatch"
            } else if msg.contains("liveness signature must be")
                || msg.contains("does not verify against the GATE-verified subject key")
            {
                "bad_or_absent_liveness"
            } else if msg.contains("stale") {
                "expired_liveness"
            } else if msg.contains("does not name") || msg.contains("one-way") {
                "one_sided_alias"
            } else if msg.contains("canon-gate") {
                "capsule_unfetchable"
            } else {
                "verification_failed"
            };
            (err_status(code), Json(VerifyErr::new(code, msg))).into_response()
        }
    }
}

// ─── Service ────────────────────────────────────────────────────────────────

/// Credential-free HTTPS verification face. HTTP-only (no RPC control channel,
/// no mesh credentials); `registrations()` is empty by design.
pub struct At9pVerifyService {
    config: At9pVerifyConfig,
    tls_config: TlsConfig,
}

impl At9pVerifyService {
    /// Construct the face from its config section and the process TLS config.
    pub fn new(config: At9pVerifyConfig, tls_config: TlsConfig) -> Self {
        Self { config, tls_config }
    }

    fn http_addr(&self) -> Result<SocketAddr, RpcError> {
        let s = format!("{}:{}", self.config.host, self.config.port);
        s.parse()
            .map_err(|e| RpcError::SpawnFailed(format!("at9p_verify: invalid bind address: {e}")))
    }
}

impl Spawnable for At9pVerifyService {
    fn name(&self) -> &str {
        SERVICE_NAME
    }

    /// Empty: this face is not part of the RPC mesh. It holds no service key,
    /// no service JWT, and is not discoverable — see the module docs.
    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        Vec::new()
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> hyprstream_rpc::error::Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| RpcError::SpawnFailed(format!("at9p_verify runtime: {e}")))?;

        rt.block_on(async move {
            let addr = self.http_addr()?;

            let rustls_config = resolve_rustls_config(
                &self.tls_config,
                self.config.tls_cert.as_ref(),
                self.config.tls_key.as_ref(),
            )
            .await
            .map_err(|e| RpcError::SpawnFailed(format!("at9p_verify TLS config: {e}")))?;

            let scheme = if rustls_config.is_some() { "https" } else { "http" };
            let state = VerifyFaceState {
                max_skew_seconds: self.config.max_skew_seconds,
                max_challenge_bytes: self.config.max_challenge_bytes,
            };
            let app = credential_free_router(state);

            info!("{SERVICE_NAME} credential-free verify face at {scheme}://{addr}/at9p/verify");

            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }
            let _ = hyprstream_rpc::notify::ready();

            serve_app(addr, app, rustls_config, shutdown, "At9pVerifyService").await
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use hyprstream_pds::at9p::{CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry,
                              ServiceType, Transport};
    use hyprstream_pds::at9p_gate::DID_AT9P_PREFIX;
    use hyprstream_pds::at9p_login::login_binding;
    use hyprstream_pds::at9p_sign::sign_capsule;
    use tower::ServiceExt; // oneshot

    use ed25519_dalek::{Signer, SigningKey};
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};

    fn signer(tag: u8) -> (SigningKey, MlDsaSigningKey, HybridKeyPair) {
        let mut seed = [0u8; 32];
        seed[0] = tag;
        seed[31] = tag.wrapping_add(7);
        let ed = SigningKey::from_bytes(&seed);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let kp = HybridKeyPair::new(ed.verifying_key().to_bytes().to_vec(), ml_dsa_vk_bytes(&pq_vk))
            .unwrap();
        (ed, pq_sk, kp)
    }

    /// Sign a capsule, return (bytes, did, ed_signing_key).
    fn signed_capsule(tag: u8, aliases: Vec<String>) -> (Vec<u8>, String, SigningKey) {
        let (ed, pq, kp) = signer(tag);
        let endpoint = ServiceEndpoint::new(Transport::Iroh, format!("iroh://n{tag}")).unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![kp], vec![service]).unwrap();
        if !aliases.is_empty() {
            body.also_known_as = Some(aliases);
        }
        let capsule = sign_capsule(body, &ed, &pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let did = format!("{DID_AT9P_PREFIX}{}", capsule.cid512().unwrap());
        (bytes, did, ed)
    }

    fn face_state() -> VerifyFaceState {
        VerifyFaceState {
            max_skew_seconds: 300,
            max_challenge_bytes: 256,
        }
    }

    /// POST a JSON body to the credential-free router and return (status, body).
    async fn post(router: Router, body: &LoginAssertionRequest) -> (StatusCode, Vec<u8>) {
        let bytes = serde_json::to_vec(body).unwrap();
        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/at9p/verify")
                    .header("content-type", "application/json")
                    .body(Body::from(bytes))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap().to_vec();
        (status, body)
    }

    /// POST with explicit extra headers (to prove the face ignores them).
    async fn post_with_headers(
        router: Router,
        body: &LoginAssertionRequest,
        headers: &[(&str, &str)],
    ) -> (StatusCode, Vec<u8>) {
        let bytes = serde_json::to_vec(body).unwrap();
        let mut req = Request::builder()
            .method("POST")
            .uri("/at9p/verify")
            .header("content-type", "application/json");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let resp = router
            .oneshot(req.body(Body::from(bytes)).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap().to_vec();
        (status, body)
    }

    #[tokio::test]
    async fn valid_assertion_returns_200_classical() {
        let (bytes, did, ed) = signed_capsule(1, Vec::new());
        let issued = SystemClock.now_unix_seconds();
        let sig = ed.sign(&login_binding(&did, "challenge-1", issued)).to_bytes();
        let req = LoginAssertionRequest {
            did: did.clone(),
            capsule_b64: STANDARD.encode(&bytes),
            challenge: "challenge-1".to_owned(),
            issued_at: issued,
            signature_b64: STANDARD.encode(sig),
            classical: None,
        };
        let (status, body) = post(credential_free_router(face_state()), &req).await;
        assert_eq!(status, StatusCode::OK);
        let ok: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(ok["verified"], true);
        assert_eq!(ok["did"], did);
        assert_eq!(ok["assurance"], "classical");
    }

    /// Credential-free guarantee, enforced by test (not convention, #1165):
    /// a request bearing a bogus `Authorization` and `Cookie` header MUST
    /// produce the same result as one with neither. REVERT the (absence of an)
    /// auth layer and this still passes; ADD an auth-rejecting layer and the
    /// no-credential baseline flips to 401, failing this test.
    #[tokio::test]
    async fn credential_free_origin_ignores_credentials() {
        let (bytes, did, ed) = signed_capsule(2, Vec::new());
        let issued = SystemClock.now_unix_seconds();
        let sig = ed.sign(&login_binding(&did, "c", issued)).to_bytes();
        let req = LoginAssertionRequest {
            did: did.clone(),
            capsule_b64: STANDARD.encode(&bytes),
            challenge: "c".to_owned(),
            issued_at: issued,
            signature_b64: STANDARD.encode(sig),
            classical: None,
        };
        let plain = post(credential_free_router(face_state()), &req).await;
        let with_creds = post_with_headers(
            credential_free_router(face_state()),
            &req,
            &[
                ("authorization", "Bearer totally.not.a.real.token"),
                ("cookie", "session=attacker; _ga=GA1.2.x"),
            ],
        )
        .await;
        // Same status, same body — the face neither requires nor consumes
        // credential headers. An auth-gated layer would diverge here.
        assert_eq!(plain, with_creds);
        assert_eq!(plain.0, StatusCode::OK);
    }

    /// Fail-closed: hash mismatch (capsule bytes don't match the DID) → 400,
    /// `verified: false`, never 200.
    #[tokio::test]
    async fn hash_mismatch_returns_400_never_200() {
        let (bytes_a, _did_a, _ed_a) = signed_capsule(3, Vec::new());
        let (_bytes_b, did_b, ed_b) = signed_capsule(4, Vec::new());
        let issued = SystemClock.now_unix_seconds();
        let sig = ed_b.sign(&login_binding(&did_b, "c", issued)).to_bytes();
        let req = LoginAssertionRequest {
            did: did_b,
            capsule_b64: STANDARD.encode(&bytes_a),
            challenge: "c".to_owned(),
            issued_at: issued,
            signature_b64: STANDARD.encode(sig),
            classical: None,
        };
        let (status, body) = post(credential_free_router(face_state()), &req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["verified"], false);
        assert_eq!(v["code"], "hash_mismatch");
    }

    /// Fail-closed: capsule unfetchable (garbage bytes) → 400.
    #[tokio::test]
    async fn unfetchable_capsule_returns_400() {
        let (_b, did, ed) = signed_capsule(5, Vec::new());
        let issued = SystemClock.now_unix_seconds();
        let sig = ed.sign(&login_binding(&did, "c", issued)).to_bytes();
        let req = LoginAssertionRequest {
            did,
            capsule_b64: STANDARD.encode(b"\xff\xff garbage"),
            challenge: "c".to_owned(),
            issued_at: issued,
            signature_b64: STANDARD.encode(sig),
            classical: None,
        };
        let (status, body) = post(credential_free_router(face_state()), &req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["verified"], false);
        assert_eq!(v["code"], "capsule_unfetchable");
    }

    /// Fail-closed: expired liveness → 400 `expired_liveness`.
    #[tokio::test]
    async fn expired_liveness_returns_400() {
        let (bytes, did, ed) = signed_capsule(6, Vec::new());
        let now = SystemClock.now_unix_seconds();
        let stale = now - 3600; // well outside the 300s window
        let sig = ed.sign(&login_binding(&did, "c", stale)).to_bytes();
        let req = LoginAssertionRequest {
            did,
            capsule_b64: STANDARD.encode(&bytes),
            challenge: "c".to_owned(),
            issued_at: stale,
            signature_b64: STANDARD.encode(sig),
            classical: None,
        };
        let (status, body) = post(credential_free_router(face_state()), &req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["verified"], false);
        assert_eq!(v["code"], "expired_liveness");
    }

    /// Fail-closed: one-sided alias → 400 `one_sided_alias`.
    #[tokio::test]
    async fn one_sided_alias_returns_400() {
        let (bytes, did, ed) = signed_capsule(7, vec!["did:web:alice.example".to_owned()]);
        let issued = SystemClock.now_unix_seconds();
        let sig = ed.sign(&login_binding(&did, "c", issued)).to_bytes();
        let req = LoginAssertionRequest {
            did: did.clone(),
            capsule_b64: STANDARD.encode(&bytes),
            challenge: "c".to_owned(),
            issued_at: issued,
            signature_b64: STANDARD.encode(sig),
            classical: Some(ClassicalClaimWire {
                classical_did: "did:web:bob.example".to_owned(),
                classical_aka_at9p: did,
            }),
        };
        let (status, body) = post(credential_free_router(face_state()), &req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["verified"], false);
        assert_eq!(v["code"], "one_sided_alias");
    }

    /// Mutual alias upgrades to `pq-hybrid` and reports the classical alias.
    #[tokio::test]
    async fn mutual_alias_returns_200_pqhybrid() {
        let classical = "did:web:node.example";
        let (bytes, did, ed) = signed_capsule(8, vec![classical.to_owned()]);
        let issued = SystemClock.now_unix_seconds();
        let sig = ed.sign(&login_binding(&did, "c", issued)).to_bytes();
        let req = LoginAssertionRequest {
            did: did.clone(),
            capsule_b64: STANDARD.encode(&bytes),
            challenge: "c".to_owned(),
            issued_at: issued,
            signature_b64: STANDARD.encode(sig),
            classical: Some(ClassicalClaimWire {
                classical_did: classical.to_owned(),
                classical_aka_at9p: did.clone(),
            }),
        };
        let (status, body) = post(credential_free_router(face_state()), &req).await;
        assert_eq!(status, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["verified"], true);
        assert_eq!(v["assurance"], "pq-hybrid");
        assert_eq!(v["classical_alias"], classical);
    }

    /// The handler does no verification work for an over-long challenge — it
    /// short-circuits before any crypto. REVERT the size guard and this still
    /// passes (but exposes the endpoint to a cheap preimage-amplification DoS);
    /// the guard exists so the cap is enforced.
    #[tokio::test]
    async fn oversized_challenge_rejected_before_crypto() {
        let (bytes, did, _ed) = signed_capsule(9, Vec::new());
        let big = "x".repeat(10_000);
        let req = LoginAssertionRequest {
            did,
            capsule_b64: STANDARD.encode(&bytes),
            challenge: big,
            issued_at: SystemClock.now_unix_seconds(),
            signature_b64: STANDARD.encode([0u8; 64]),
            classical: None,
        };
        let (status, body) = post(credential_free_router(face_state()), &req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["code"], "challenge_too_long");
    }
}
