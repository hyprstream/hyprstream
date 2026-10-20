//! Fixed Federate browser exchange. The OAuth runner installs the issuer only
//! in PostgreSQL builds after constructing its authenticated Policy adapter and
//! the currently committed composite signer; downstream Policy handlers still
//! deny until their own session-store runtime is fully configured.
#![allow(dead_code)] // The production adapter is a separate B1 activation packet.

#[cfg(feature = "postgres")]
use std::collections::BTreeMap;
use std::sync::Arc;
#[cfg(feature = "postgres")]
use std::time::Duration;

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
#[cfg(feature = "postgres")]
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine as _};
#[cfg(feature = "postgres")]
use rand::RngCore as _;
use serde::Serialize;
#[cfg(feature = "postgres")]
use tokio::sync::Mutex;

#[cfg(feature = "postgres")]
use super::{
    federate_exchange::{ChallengeRequest, ExchangeRequest},
    federate_source::{AuthorityBinding, Challenge, Source, VerifiedPossession, Verifier},
};
use super::{federate_source, state::OAuthState};

#[cfg(feature = "postgres")]
const MAX_PENDING: usize = 128;

#[cfg(feature = "postgres")]
use hyprstream_rpc::auth::Claims;
#[cfg(feature = "postgres")]
use hyprstream_session_store::Session;

/// Policy owns source-to-account mapping and the durable source/session commit.
/// Its production implementation must authenticate the OAuth service caller.
#[cfg(feature = "postgres")]
#[async_trait::async_trait]
pub(crate) trait FederateAdmission: Send + Sync {
    async fn prepare(
        &self,
        source: &Source,
        requested: &str,
        challenge_id: [u8; 32],
        created_at: u64,
    ) -> anyhow::Result<PreparedDecision>;
    async fn commit(
        &self,
        evidence: VerifiedPossession,
        policy_handle: Vec<u8>,
    ) -> anyhow::Result<Session>;
}

/// Policy's decision stays opaque to OAuth. Only the enrolled Policy adapter
/// can validate this handle at redeem; it is never sent to the browser.
#[cfg(feature = "postgres")]
pub(crate) struct PreparedDecision {
    pub binding: AuthorityBinding,
    pub policy_handle: Vec<u8>,
}

/// The signer uses the current OAuth access-token key, not a browser key.
/// It is called only after Policy returns the committed session record.
#[cfg(feature = "postgres")]
#[async_trait::async_trait]
pub(crate) trait FederateSigner: Send + Sync {
    async fn sign(&self, claims: &Claims) -> anyhow::Result<String>;
}

#[cfg(feature = "postgres")]
struct Pending {
    request: ChallengeRequest,
    challenge: Challenge,
    policy_handle: Vec<u8>,
}

/// The OAuthState slot defaults to None until the production OAuth factory
/// installs this with its Policy adapter and access-token signer. When absent,
/// both routes fail before source, challenge or token state changes.
#[cfg(feature = "postgres")]
pub(crate) struct FederateIssuer {
    verifier: Verifier,
    admission: Arc<dyn FederateAdmission>,
    signer: Arc<dyn FederateSigner>,
    pending: Mutex<BTreeMap<[u8; 32], Pending>>,
}

#[cfg(feature = "postgres")]
impl FederateIssuer {
    pub(crate) fn new(
        admission: Arc<dyn FederateAdmission>,
        signer: Arc<dyn FederateSigner>,
        host: federate_source::FederateHost,
        tui_client: Option<federate_source::ClientClass>,
    ) -> Result<Self, federate_source::Error> {
        let mut verifier = Verifier::for_host(host)?;
        if let Some(class) = tui_client {
            verifier = verifier.with_client_class(class);
        }
        Ok(Self {
            verifier,
            admission,
            signer,
            pending: Mutex::new(BTreeMap::new()),
        })
    }

    /// The configured TUI client id, for the `/oauth/token` federate form
    /// reservation (W1.3). `None` = staging-only routing, previous behavior.
    pub(crate) fn tui_client_id(&self) -> Option<&str> {
        self.verifier.tui_client_id()
    }

    async fn challenge(
        &self,
        headers: &HeaderMap,
        body: &[u8],
        now: u64,
    ) -> Result<ChallengeResponse, federate_source::Error> {
        let request = ChallengeRequest::parse_for_host(
            body,
            self.verifier.host()?,
            self.verifier.tui_client(),
        )?;
        if !origin_for_class(headers, request.client_id(), self.verifier.tui_client_id())? {
            return Err(federate_source::Error::Invalid);
        }
        let source = request.verify_source(&self.verifier, now).await?;
        let mut id = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut id);
        let prepared = tokio::time::timeout(
            Duration::from_secs(2),
            self.admission
                .prepare(&source, request.requested_scope(), id, now),
        )
        .await
        .map_err(|_| federate_source::Error::Unavailable)?;
        let prepared = prepared.map_err(|_| federate_source::Error::Unavailable)?;
        if prepared.policy_handle.is_empty() || prepared.policy_handle.len() > 4096 {
            return Err(federate_source::Error::Unavailable);
        }
        if prepared.binding.requested != request.requested_scope() {
            return Err(federate_source::Error::Unavailable);
        }
        let challenge = source.challenge(prepared.binding, id, now)?;
        let response = ChallengeResponse {
            challenge_id: B64.encode(challenge.challenge_id()),
            // The class's own profile string: a TUI-class challenge must
            // answer federate-tui-v1, not the www v1 pin.
            profile: super::federate_exchange::resolve_wire_class(
                request.client_id(),
                self.verifier.tui_client(),
            )?
            .profile(),
            expires_in: challenge.expires_at() - now,
            granted_scope: challenge.granted_scope().to_owned(),
            transcript_b64u: B64.encode(challenge.transcript()),
        };
        let mut pending = self.pending.lock().await;
        pending.retain(|_, entry| entry.challenge.expires_at() > now);
        if pending.len() >= MAX_PENDING || pending.contains_key(&id) {
            return Err(federate_source::Error::Unavailable);
        }
        pending.insert(
            id,
            Pending {
                request,
                challenge,
                policy_handle: prepared.policy_handle,
            },
        );
        Ok(response)
    }

    async fn exchange(
        &self,
        headers: &HeaderMap,
        body: &[u8],
        now: u64,
    ) -> Result<super::federate_exchange::ExchangeResponse, federate_source::Error> {
        let request = ExchangeRequest::parse_for_host(
            body,
            self.verifier.host()?,
            self.verifier.tui_client(),
        )?;
        if !origin_for_class(headers, request.client_id(), self.verifier.tui_client_id())? {
            return Err(federate_source::Error::Invalid);
        }
        let id = request.challenge_id()?;
        // Remove before network calls: a duplicate or uncertain retry cannot
        // race the same challenge to a second Policy commit.
        let pending = self
            .pending
            .lock()
            .await
            .remove(&id)
            .ok_or(federate_source::Error::Invalid)?;
        // The exchange must stay in the class the challenge was prepared
        // under: a submitted client_id/hs_profile pair naming the OTHER
        // class is cross-class substitution — it would apply the wrong
        // Origin context pin (already checked above for the submitted
        // class) while the proof, Policy commit, and mint all bind the
        // challenge's original class. The one-use handle is already
        // consumed above, so a mismatch denies the attempt entirely.
        if pending.request.client_id() != request.client_id()
            || pending.request.requested_scope() != request.requested_scope()
            || pending.request.subject_token() != request.subject_token()
        {
            return Err(federate_source::Error::Invalid);
        }
        let (expected, ed_key, pq_key) = pending.challenge.expected_session();
        let source = pending.request.verify_source(&self.verifier, now).await?;
        let (source_issuer, source_subject) = source.provenance();
        let source_issuer = source_issuer.to_owned();
        let source_subject = source_subject.to_owned();
        let (ed, pq) = request.signatures()?;
        let verified = pending.challenge.verify(source, &id, &ed, &pq, now)?;
        let committed = tokio::time::timeout(
            Duration::from_secs(2),
            self.admission.commit(verified, pending.policy_handle),
        )
        .await
        .map_err(|_| federate_source::Error::Unavailable)?;
        let committed = committed.map_err(|_| federate_source::Error::Unavailable)?;
        if committed.account_id != expected.account
            || committed.subject != expected.subject
            || committed.tenant != expected.tenant
            || committed.scopes.join(" ") != expected.granted
            || committed.grant_revision != expected.revision
            || committed.ed_public != ed_key
            || committed.pq_public != pq_key
        {
            return Err(federate_source::Error::Unavailable);
        }
        let now = i64::try_from(now).map_err(|_| federate_source::Error::Invalid)?;
        // TUI-class credentials take NORMAL JWT verification at dispatch,
        // where svc.rs (v16 §3.3) fails closed on an unknown `sid` for a
        // local sid-carrying token. Publish the committed session to the
        // session authority BEFORE minting, or every TUI credential would
        // deny at dispatch. Fail-closed: no registry or failed registration,
        // no token. The www class never takes the normal path (it dispatches
        // through the deferred branch) and is deliberately not registered.
        if Some(committed.client_id.as_str()) == self.verifier.tui_client_id() {
            let registry = hyprstream_rpc::auth::global_session_registry()
                .ok_or(federate_source::Error::Unavailable)?;
            register_tui_session(registry.as_ref(), &committed, now).await?;
        }
        let claims = super::federate_exchange::authorized_committed_claims_for_host(
            &committed,
            now,
            &source_issuer,
            &source_subject,
            crate::mac::exchange_enrollment_resolver().as_ref(),
            self.verifier.host()?,
            self.verifier.tui_client_id(),
        )?;
        let token = tokio::time::timeout(Duration::from_secs(2), self.signer.sign(&claims))
            .await
            .map_err(|_| federate_source::Error::Unavailable)?;
        let token = token.map_err(|_| federate_source::Error::Unavailable)?;
        let response_now = chrono::Utc::now().timestamp();
        super::federate_exchange::ExchangeResponse::committed_for_host(
            token,
            &committed,
            response_now,
            &source_issuer,
            &source_subject,
            self.verifier.host()?,
            self.verifier.tui_client_id(),
        )
    }
}

/// Publish a committed TUI-class session to the session authority (W1.3).
/// Interactive kind, active, expiring with the session; fail-closed on any
/// registration error (no session record, no minted credential).
#[cfg(feature = "postgres")]
async fn register_tui_session(
    registry: &dyn hyprstream_rpc::auth::SessionRegistry,
    committed: &Session,
    now: i64,
) -> Result<(), federate_source::Error> {
    registry
        .register_session(
            hyprstream_rpc::auth::SessionKey::oidc(committed.host.clone(), committed.sid.clone()),
            hyprstream_rpc::auth::SessionState {
                subject: committed.subject.clone(),
                tenant: committed.tenant.clone(),
                kind: hyprstream_rpc::auth::SessionKind::Interactive,
                created_at: now,
                expires_at: committed.expires_at,
                status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                clearance_epoch: 0,
            },
        )
        .await
        .map_err(|_| federate_source::Error::Unavailable)
}

#[derive(Serialize)]
struct ChallengeResponse {
    challenge_id: String,
    profile: &'static str,
    expires_in: u64,
    granted_scope: String,
    transcript_b64u: String,
}

/// Class-aware origin rule (W1.3 native-client HTTP contract).
///
/// * www staging class (browser): exactly one `Origin` header, byte-equal
///   to the pinned website origin — unchanged from the browser-only rule.
/// * TUI class (native client): the www `Origin` header must be ABSENT.
///   A native TUI client never sends a browser `Origin`; if any `Origin`
///   is present the request is denied, so a web page cannot pose as the
///   native TUI client (defense in depth under the mandatory holder
///   possession proof — the origin check is a context pin, not the auth).
/// * Any other client id: denied (no accepted class resolved).
fn origin_for_class(
    headers: &HeaderMap,
    client_id: &str,
    tui_client: Option<&str>,
) -> Result<bool, federate_source::Error> {
    let origin_count = headers.get_all(header::ORIGIN).iter().count();
    if client_id == federate_source::CLIENT {
        return Ok(origin_count == 1
            && headers
                .get(header::ORIGIN)
                .and_then(|value| value.to_str().ok())
                == Some(federate_source::WEBSITE));
    }
    if Some(client_id) == tui_client {
        return Ok(origin_count == 0);
    }
    Err(federate_source::Error::Invalid)
}

fn content_type(headers: &HeaderMap, expected: &str) -> bool {
    headers.get_all(header::CONTENT_TYPE).iter().count() == 1
        && headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            == Some(expected)
}

fn unavailable() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({"error":"temporarily_unavailable"})),
    )
        .into_response()
}

fn denied() -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error":"invalid_grant"})),
    )
        .into_response()
}

pub(crate) async fn challenge(
    State(state): State<Arc<OAuthState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    #[cfg(feature = "postgres")]
    if let Some(issuer) = &state.federate_issuer {
        // Content type stays a route-level format pin for both classes; the
        // class-aware origin rule is applied inside the issuer after the
        // wire client_id resolves to an accepted class.
        if !content_type(&headers, "application/json") {
            return denied();
        }
        let now = chrono::Utc::now().timestamp();
        let Ok(now) = u64::try_from(now) else {
            return denied();
        };
        return match issuer.challenge(&headers, &body, now).await {
            Ok(response) => (
                [
                    (header::CACHE_CONTROL, "no-store"),
                    (header::PRAGMA, "no-cache"),
                ],
                Json(response),
            )
                .into_response(),
            Err(federate_source::Error::Unavailable) => unavailable(),
            Err(_) => denied(),
        };
    }
    let _ = (state, headers, body);
    unavailable()
}

pub(crate) async fn exchange(state: &OAuthState, headers: &HeaderMap, body: &[u8]) -> Response {
    #[cfg(feature = "postgres")]
    if let Some(issuer) = &state.federate_issuer {
        // Content type stays a route-level format pin for both classes; the
        // class-aware origin rule is applied inside the issuer after the
        // wire client_id resolves to an accepted class.
        if !content_type(headers, "application/x-www-form-urlencoded") {
            return denied();
        }
        let now = chrono::Utc::now().timestamp();
        let Ok(now) = u64::try_from(now) else {
            return denied();
        };
        return match issuer.exchange(headers, body, now).await {
            Ok(response) => (
                [
                    (header::CACHE_CONTROL, "no-store"),
                    (header::PRAGMA, "no-cache"),
                ],
                Json(response),
            )
                .into_response(),
            Err(federate_source::Error::Unavailable) => unavailable(),
            Err(_) => denied(),
        };
    }
    let _ = (state, headers, body);
    unavailable()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const TUI_CLIENT: &str = "hyprstream-tui-staging";

    #[test]
    fn exact_origin_and_content_type() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::ORIGIN,
            axum::http::HeaderValue::from_static(federate_source::WEBSITE),
        );
        headers.insert(
            header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
        assert!(content_type(&headers, "application/json"));
        // www class: the exact single website origin, and nothing else.
        assert!(origin_for_class(&headers, federate_source::CLIENT, Some(TUI_CLIENT)).unwrap());
        headers.append(
            header::ORIGIN,
            axum::http::HeaderValue::from_static("https://attacker.invalid"),
        );
        assert!(!origin_for_class(&headers, federate_source::CLIENT, Some(TUI_CLIENT)).unwrap());
        // TUI class: no Origin header at all; any Origin (including the
        // website's own) is denied, and an unconfigured client is denied.
        let mut native = HeaderMap::new();
        assert!(origin_for_class(&native, TUI_CLIENT, Some(TUI_CLIENT)).unwrap());
        native.insert(
            header::ORIGIN,
            axum::http::HeaderValue::from_static(federate_source::WEBSITE),
        );
        assert!(!origin_for_class(&native, TUI_CLIENT, Some(TUI_CLIENT)).unwrap());
        native.clear();
        native.insert(
            header::ORIGIN,
            axum::http::HeaderValue::from_static("https://attacker.invalid"),
        );
        assert!(!origin_for_class(&native, TUI_CLIENT, Some(TUI_CLIENT)).unwrap());
        let none = HeaderMap::new();
        assert!(origin_for_class(&none, "unknown-client", Some(TUI_CLIENT)).is_err());
        assert!(origin_for_class(&none, TUI_CLIENT, None).is_err());
        // The www origin pin still holds with no TUI class configured.
        let mut www = HeaderMap::new();
        www.insert(
            header::ORIGIN,
            axum::http::HeaderValue::from_static(federate_source::WEBSITE),
        );
        assert!(origin_for_class(&www, federate_source::CLIENT, None).unwrap());
    }

    #[cfg(feature = "postgres")]
    #[tokio::test]
    async fn tui_session_registration_is_interactive_active_and_fail_closed() {
        use hyprstream_rpc::auth::SessionRegistry as _;
        let registry = hyprstream_rpc::auth::InMemorySessionRegistry::new();
        let now = chrono::Utc::now().timestamp();
        let session = hyprstream_session_store::Session {
            host: "https://tui-host.test".into(),
            sid: "tui-sid-1".into(),
            account_id: "account".into(),
            subject: "did:plc:abcdefghijklmnopqrstuvwx".into(),
            tenant: "tenant-a".into(),
            client_id: TUI_CLIENT.into(),
            resource: "https://tui-host.test".into(),
            scopes: vec!["connect:tui:session".into()],
            grant_revision: "revision-1".into(),
            ed_public: [3; 32],
            pq_public: vec![4; 1952],
            generation: [5; 32],
            collision_inventory_id: [5; 32],
            proof_epoch: 1,
            expires_at: now + 120,
        };
        register_tui_session(&registry, &session, now)
            .await
            .expect("TUI session registration");
        let key = hyprstream_rpc::auth::SessionKey::oidc("https://tui-host.test", "tui-sid-1");
        assert!(
            !registry.is_revoked(&key).await,
            "registered session is active"
        );
        let state = registry.session_state(&key).await.expect("session state");
        assert_eq!(state.subject, "did:plc:abcdefghijklmnopqrstuvwx");
        assert_eq!(state.tenant, "tenant-a");
        assert_eq!(state.kind, hyprstream_rpc::auth::SessionKind::Interactive);
        assert_eq!(state.expires_at, now + 120);
        // A duplicate sid registration fails closed (session identifiers are
        // never reassigned, v16 §3.3) and surfaces as Unavailable (no mint).
        assert!(matches!(
            register_tui_session(&registry, &session, now).await,
            Err(federate_source::Error::Unavailable)
        ));
    }
    // ── Finding-3 fixtures: the exchange stays in the challenge's class ────

    #[cfg(feature = "postgres")]
    mod exchange_class_binding {
        //! End-to-end through the real FederateIssuer: challenge in one
        //! class, exchange submitted as the other. Both directions must
        //! deny before source re-verification, possession, or Policy
        //! commit. Offline RSA/JWKS fixtures only (disposable test key).
        #![allow(clippy::unwrap_used, clippy::expect_used)]
        use super::*;
        use ed25519_dalek::Signer as _;
        use hyprstream_rpc::crypto::pq::{ml_dsa_sign, ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes};
        use serde_json::{json, Value};
        use sha2::Digest as _;

        const NOW: u64 = 1_800_000_000;
        const CODE: &str = "TEST-CODE-NOT-LIVE";
        const TUI_AUD: &str = "hyprstream-tui-aud";
        const SCOPE: &str = "connect:tui:session";
        const MODULUS: &str = "C97781230C20F14C2EB47FDD3AFBB4821EAC87003FF1A8240D1E840EE743E3EAF7A5F204503E15CFB42751AEDB97D76BE41DF559834957D283E4AC097283D18D19A12DCE56434FCA31BFC9635E721798A3D6AACC55FC8CAD559F74D66549EFC160AAD48DA9A2F860CCEA515DFEEEFA930E10B8BA5414CB26034B7352E53C7AAD5C4CE24528615705C737B7D4B554B1B4CE1A52F2D3439B42F67324B4F007691E46BE826EC6DC7CC65DD58FBF183B86DF537889EDBA5866BFD911A2AA27181C9EBD14C6DBACA28E3A37153BCD0018071101E6EA5A900812D194E6220A672EDB9FCA2C24C5486DCF981C3A2C9DC20B95E3D14875F9AF20993751BB8CBB6020678D";

        fn encoding_key() -> jsonwebtoken::EncodingKey {
            let pem: String = include_str!("federate_source/test-rsa.pem")
                .lines()
                .filter(|s| !s.starts_with('-'))
                .collect();
            jsonwebtoken::EncodingKey::from_rsa_der(
                &base64::engine::general_purpose::STANDARD
                    .decode(pem)
                    .unwrap(),
            )
        }

        fn jwks() -> Value {
            json!({"keys":[{"kty":"RSA","kid":"fixture","alg":"RS256","use":"sig","n":B64.encode(hex::decode(MODULUS).unwrap()),"e":"AQAB"}]})
        }

        fn token(claims: &Value) -> String {
            let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256);
            header.kid = Some("fixture".into());
            jsonwebtoken::encode(&header, claims, &encoding_key()).unwrap()
        }

        struct StubAdmission {
            commits: Arc<std::sync::atomic::AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl FederateAdmission for StubAdmission {
            async fn prepare(
                &self,
                _source: &Source,
                _requested: &str,
                _challenge_id: [u8; 32],
                _created_at: u64,
            ) -> anyhow::Result<PreparedDecision> {
                Ok(PreparedDecision {
                    binding: AuthorityBinding {
                        account: "account".into(),
                        subject: "did:plc:abcdefghijklmnopqrstuvwx".into(),
                        tenant: "tenant-a".into(),
                        requested: SCOPE.into(),
                        granted: SCOPE.into(),
                        revision: "rev-1".into(),
                    },
                    policy_handle: vec![1; 32],
                })
            }
            async fn commit(
                &self,
                _evidence: VerifiedPossession,
                _policy_handle: Vec<u8>,
            ) -> anyhow::Result<Session> {
                self.commits
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                anyhow::bail!("sentinel: reached Policy commit")
            }
        }

        struct StubSigner;
        #[async_trait::async_trait]
        impl FederateSigner for StubSigner {
            async fn sign(&self, _claims: &Claims) -> anyhow::Result<String> {
                anyhow::bail!("sentinel: reached mint")
            }
        }

        fn class_parts(tui: bool) -> (federate_source::ClientClass, &'static str, &'static str) {
            if tui {
                (
                    federate_source::ClientClass::new(TUI_CLIENT, TUI_AUD, None).unwrap(),
                    federate_source::TUI_PROFILE,
                    TUI_AUD,
                )
            } else {
                (
                    federate_source::ClientClass::staging(),
                    "federate-session-v1",
                    federate_source::CLIENT,
                )
            }
        }

        struct Prepared {
            issuer: FederateIssuer,
            commits: Arc<std::sync::atomic::AtomicUsize>,
            challenge_id: String,
            transcript: Vec<u8>,
            subject_token: String,
        }

        /// Run a full challenge in the given class against a real issuer
        /// (real Verifier + offline JWKS, stub admission/signer).
        async fn prepare_challenge(tui: bool) -> Prepared {
            let (class, profile, aud) = class_parts(tui);
            let host = federate_source::FederateHost::new(federate_source::HOST).unwrap();
            let commits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let admission = Arc::new(StubAdmission {
                commits: Arc::clone(&commits),
            });
            let mut issuer = FederateIssuer::new(
                admission,
                Arc::new(StubSigner),
                host.clone(),
                Some(federate_source::ClientClass::new(TUI_CLIENT, TUI_AUD, None).unwrap()),
            )
            .unwrap();
            issuer.verifier.set_test_responses(
                (0..4)
                    .map(|_| Ok(serde_json::to_vec(&jwks()).unwrap()))
                    .collect(),
            );

            let login_random = B64.encode([11; 32]);
            let state = B64.encode([12; 32]);
            let pkce = B64.encode([13; 32]);
            let ed = ed25519_dalek::SigningKey::from_bytes(&[7; 32]);
            let pq = ml_dsa_sk_from_seed(&[8; 32]);
            let ed_public = B64.encode(ed.verifying_key().as_bytes());
            let pq_public = B64.encode(ml_dsa_sk_to_vk_bytes(&pq));
            let commitment = federate_source::Commitment::for_class(
                host.clone(),
                &class,
                &login_random,
                &state,
                &pkce,
                &ed_public,
                &pq_public,
            )
            .unwrap();
            let claims = json!({
                "iss": federate_source::ISSUER,
                "aud": aud,
                "azp": class.client_id(),
                "sub": "opaque-sub",
                "jti": "source-id",
                "iat": NOW,
                "exp": NOW + 300,
                "nonce": commitment.nonce(),
                "c_hash": B64.encode(&sha2::Sha256::digest(CODE.as_bytes())[..16]),
                "federated_claims": {"connector_id":"atproto","user_id":"did:plc:abcdefghijklmnopqrstuvwx"},
            });
            let subject_token = token(&claims);
            let body = serde_json::to_vec(&json!({
                "profile": profile,
                "client_id": class.client_id(),
                "subject_token": subject_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
                "authorization_code": CODE,
                "resource": federate_source::HOST,
                "scope": SCOPE,
                "login_random": login_random,
                "state": state,
                "pkce_challenge": pkce,
                "ed25519_public": ed_public,
                "ml_dsa65_public": pq_public,
            }))
            .unwrap();
            let mut headers = HeaderMap::new();
            if !tui {
                headers.insert(
                    header::ORIGIN,
                    axum::http::HeaderValue::from_static(federate_source::WEBSITE),
                );
            }
            let response = issuer.challenge(&headers, &body, NOW).await.unwrap();
            Prepared {
                issuer,
                commits,
                challenge_id: response.challenge_id,
                transcript: B64.decode(response.transcript_b64u).unwrap(),
                subject_token,
            }
        }

        fn exchange_form(prepared: &Prepared, tui: bool, transcript: &[u8]) -> String {
            let (_class, profile, _aud) = class_parts(tui);
            let ed = ed25519_dalek::SigningKey::from_bytes(&[7; 32]);
            let pq = ml_dsa_sk_from_seed(&[8; 32]);
            let ed_sig = ed.sign(transcript);
            let pq_sig = ml_dsa_sign(&pq, transcript);
            url::form_urlencoded::Serializer::new(String::new())
                .append_pair(
                    "grant_type",
                    "urn:ietf:params:oauth:grant-type:token-exchange",
                )
                .append_pair("hs_profile", profile)
                .append_pair(
                    "client_id",
                    if tui {
                        TUI_CLIENT
                    } else {
                        federate_source::CLIENT
                    },
                )
                .append_pair(
                    "subject_token_type",
                    "urn:ietf:params:oauth:token-type:id_token",
                )
                .append_pair("subject_token", &prepared.subject_token)
                .append_pair(
                    "requested_token_type",
                    "urn:ietf:params:oauth:token-type:access_token",
                )
                .append_pair("resource", federate_source::HOST)
                .append_pair("scope", SCOPE)
                .append_pair("challenge_id", &prepared.challenge_id)
                .append_pair("ed25519_signature", &B64.encode(ed_sig.to_bytes()))
                .append_pair("ml_dsa65_signature", &B64.encode(pq_sig))
                .finish()
        }

        #[tokio::test]
        async fn exchange_stays_in_the_challenge_class_both_directions() {
            // www challenge redeemed as TUI (native, no Origin): denied
            // before Policy commit.
            let prepared = prepare_challenge(false).await;
            let form = exchange_form(&prepared, true, &prepared.transcript);
            let result = prepared
                .issuer
                .exchange(&HeaderMap::new(), form.as_bytes(), NOW)
                .await;
            assert!(matches!(result, Err(federate_source::Error::Invalid)));
            assert_eq!(
                prepared.commits.load(std::sync::atomic::Ordering::SeqCst),
                0
            );

            // TUI challenge redeemed as www (website Origin): denied before
            // Policy commit.
            let prepared = prepare_challenge(true).await;
            let form = exchange_form(&prepared, false, &prepared.transcript);
            let mut headers = HeaderMap::new();
            headers.insert(
                header::ORIGIN,
                axum::http::HeaderValue::from_static(federate_source::WEBSITE),
            );
            let result = prepared
                .issuer
                .exchange(&headers, form.as_bytes(), NOW)
                .await;
            assert!(matches!(result, Err(federate_source::Error::Invalid)));
            assert_eq!(
                prepared.commits.load(std::sync::atomic::Ordering::SeqCst),
                0
            );

            // Control: a same-class exchange passes the class binding and
            // reaches Policy commit (the stub's sentinel error).
            let prepared = prepare_challenge(true).await;
            let form = exchange_form(&prepared, true, &prepared.transcript);
            let result = prepared
                .issuer
                .exchange(&HeaderMap::new(), form.as_bytes(), NOW)
                .await;
            let reached = prepared.commits.load(std::sync::atomic::Ordering::SeqCst);
            assert_eq!(
                reached,
                1,
                "matched-class exchange must reach Policy commit: {:?}",
                result.as_ref().err()
            );
            assert!(matches!(result, Err(federate_source::Error::Unavailable)));
        }
    }
}
