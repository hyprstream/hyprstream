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
            profile: "federate-session-v1",
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
        if pending.request.requested_scope() != request.requested_scope()
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
#[allow(clippy::unwrap_used)]
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
}
