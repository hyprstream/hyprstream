//! Credential-free HTTP face for hosted-account identity artifacts (#1165).
//!
//! The router in this module is intentionally a complete, small Axum router
//! with no session, cookie, bearer, or authorization middleware. It serves
//! immutable bytes loaded from sealed publication artifacts on a separately
//! bound listener. The listener and TLS termination are deployment concerns;
//! this module owns the routing and credential-free security contract.

use std::{collections::BTreeMap, sync::Arc};

use anyhow::{ensure, Result};
use async_trait::async_trait;
use axum::{
    extract::State,
    http::{header, uri::Authority, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use hyprstream_pds::{AccountLabel, SealedHostedAccount};

/// Canonical hosted account zone used by the default deployment.
pub const DEFAULT_ACCOUNT_ZONE: &str = "tormentnexus.social";

/// A byte-stable set of artifacts for one hosted account.
///
/// Values are copied into this object once, at publication/index-load time.
/// Request handlers only clone the `Arc`-backed bytes; they never regenerate
/// or reserialize a DID document or operation log.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostedAccountHttpArtifacts {
    label: String,
    did: String,
    did_document: Arc<[u8]>,
    atproto_did: Arc<[u8]>,
    did_log: Arc<[u8]>,
}

impl HostedAccountHttpArtifacts {
    /// Construct an artifact bundle after the publisher has validated the
    /// sealed bytes. The constructor checks identity binding and rejects empty
    /// artifacts; it does not parse or rewrite any bytes.
    pub fn new(
        label: impl Into<String>,
        did: impl Into<String>,
        did_document: impl Into<Vec<u8>>,
        atproto_did: impl Into<Vec<u8>>,
        did_log: impl Into<Vec<u8>>,
    ) -> Result<Self> {
        let label = label.into();
        AccountLabel::parse(&label).map_err(|error| anyhow::anyhow!(error))?;
        let did = did.into();
        let host = did
            .strip_prefix("did:web:")
            .ok_or_else(|| anyhow::anyhow!("hosted account HTTP artifact DID is not did:web"))?;
        ensure!(
            !host.is_empty()
                && !host.contains([':', '/'])
                && host.split('.').count() >= 2
                && host.split('.').next() == Some(label.as_str()),
            "hosted account HTTP artifact DID does not match its label"
        );
        for component in host.split('.') {
            ensure!(
                !component.is_empty()
                    && component.len() <= 63
                    && component.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
                    })
                    && !component.starts_with('-')
                    && !component.ends_with('-'),
                "hosted account HTTP artifact DID contains an invalid DNS label"
            );
        }
        let did_document: Arc<[u8]> = did_document.into().into();
        let atproto_did: Arc<[u8]> = atproto_did.into().into();
        let did_log: Arc<[u8]> = did_log.into().into();
        ensure!(!did_document.is_empty(), "sealed DID document is empty");
        ensure!(
            !atproto_did.is_empty(),
            "sealed atproto DID artifact is empty"
        );
        ensure!(!did_log.is_empty(), "sealed DID operation log is empty");
        ensure!(
            atproto_did.as_ref() == did.as_bytes(),
            "atproto DID artifact does not match its sealed DID"
        );
        Ok(Self {
            label,
            did,
            did_document,
            atproto_did,
            did_log,
        })
    }

    /// Build the HTTP bundle from a sealed account publication.
    ///
    /// `SealedHostedAccount` has already bound the DID document and operation
    /// log into the signed genesis. This method only copies those exact bytes
    /// and creates the plaintext atproto-did artifact once.
    pub fn from_sealed(account: &SealedHostedAccount) -> Result<Self> {
        let label = account.record().name().label().to_owned();
        let did = account.record().name().did().to_owned();
        let document = account.did_document().as_bytes();
        let parsed = hyprstream_pds::SealedHostedDidDocument::from_canonical_json(document)?;
        ensure!(
            parsed.did() == did,
            "sealed DID document does not match account record"
        );
        Self::new(
            label,
            did.clone(),
            document.to_vec(),
            did.into_bytes(),
            account.genesis_bytes().to_vec(),
        )
    }

    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[must_use]
    pub fn did(&self) -> &str {
        &self.did
    }

    #[must_use]
    pub fn did_document(&self) -> &[u8] {
        &self.did_document
    }

    #[must_use]
    pub fn atproto_did(&self) -> &[u8] {
        &self.atproto_did
    }

    #[must_use]
    pub fn did_log(&self) -> &[u8] {
        &self.did_log
    }
}

/// Read-only account artifact index used by the credential-free router.
#[async_trait]
pub trait HostedAccountHttpDirectory: Send + Sync {
    /// Return the immutable artifact bundle for one canonical host label.
    async fn lookup(&self, label: &str) -> Result<Option<Arc<HostedAccountHttpArtifacts>>>;
}

/// Immutable in-memory index suitable for a startup-loaded sealed artifact
/// snapshot and for deterministic tests.
#[derive(Clone, Default)]
pub struct StaticHostedAccountHttpDirectory {
    entries: Arc<BTreeMap<String, Arc<HostedAccountHttpArtifacts>>>,
}

impl StaticHostedAccountHttpDirectory {
    pub fn new(artifacts: impl IntoIterator<Item = HostedAccountHttpArtifacts>) -> Result<Self> {
        let mut entries = BTreeMap::new();
        for artifact in artifacts {
            let label = artifact.label().to_owned();
            ensure!(
                entries.insert(label.clone(), Arc::new(artifact)).is_none(),
                "duplicate hosted account HTTP label {label:?}"
            );
        }
        Ok(Self {
            entries: Arc::new(entries),
        })
    }
}

#[async_trait]
impl HostedAccountHttpDirectory for StaticHostedAccountHttpDirectory {
    async fn lookup(&self, label: &str) -> Result<Option<Arc<HostedAccountHttpArtifacts>>> {
        Ok(self.entries.get(label).cloned())
    }
}

#[derive(Clone)]
struct AccountHttpState {
    zone: String,
    directory: Arc<dyn HostedAccountHttpDirectory>,
}

/// Build the separate credential-free account-artifact router.
///
/// The returned router has only three GET routes and no layers. Bind it to a
/// distinct listener from authenticated OAuth/XRPC routes. The caller remains
/// responsible for binding the wildcard certificate and for refusing to mount
/// this router into an authenticated listener.
pub fn router(
    zone: impl Into<String>,
    directory: Arc<dyn HostedAccountHttpDirectory>,
) -> Result<Router> {
    let zone = canonical_zone(&zone.into())?;
    let state = AccountHttpState { zone, directory };
    Ok(Router::new()
        .route("/.well-known/did.json", get(serve_did_document))
        .route("/.well-known/atproto-did", get(serve_atproto_did))
        .route("/.well-known/did-log.json", get(serve_did_log))
        .with_state(Arc::new(state)))
}

async fn serve_did_document(
    State(state): State<Arc<AccountHttpState>>,
    headers: HeaderMap,
) -> Response {
    serve_artifact(&state, &headers, ArtifactKind::DidDocument).await
}

async fn serve_atproto_did(
    State(state): State<Arc<AccountHttpState>>,
    headers: HeaderMap,
) -> Response {
    serve_artifact(&state, &headers, ArtifactKind::AtprotoDid).await
}

async fn serve_did_log(State(state): State<Arc<AccountHttpState>>, headers: HeaderMap) -> Response {
    serve_artifact(&state, &headers, ArtifactKind::DidLog).await
}

#[derive(Clone, Copy)]
enum ArtifactKind {
    DidDocument,
    AtprotoDid,
    DidLog,
}

async fn serve_artifact(
    state: &AccountHttpState,
    headers: &HeaderMap,
    kind: ArtifactKind,
) -> Response {
    // A credential-bearing request must never reach an artifact lookup. This
    // explicit rejection is the testable security boundary for the separate
    // origin; there is no cookie/session/bearer middleware to accidentally
    // inherit from the authenticated service.
    if headers.contains_key(header::AUTHORIZATION) || headers.contains_key(header::COOKIE) {
        return (StatusCode::BAD_REQUEST, "credentials are not accepted").into_response();
    }
    let Some(host) = headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
    else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let Some(label) = host_label(host, &state.zone) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let artifact = match state.directory.lookup(&label).await {
        Ok(Some(artifact)) => artifact,
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(_) => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };
    if artifact.did() != format!("did:web:{label}.{}", state.zone) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let (body, content_type): (&[u8], &'static str) = match kind {
        ArtifactKind::DidDocument => (artifact.did_document(), "application/did+json"),
        ArtifactKind::AtprotoDid => (artifact.atproto_did(), "text/plain; charset=utf-8"),
        // The current sealed operation-log artifact is a signed CBOR genesis
        // envelope even though the stable URL retains its historical .json
        // suffix. Never transcode it at the HTTP boundary.
        ArtifactKind::DidLog => (artifact.did_log(), "application/cbor"),
    };
    let mut response = body.to_vec().into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=300, immutable"),
    );
    response
}

fn canonical_zone(zone: &str) -> Result<String> {
    ensure!(
        !zone.is_empty() && zone.is_ascii(),
        "account zone must be ASCII"
    );
    let zone = zone.trim_end_matches('.');
    let labels: Vec<&str> = zone.split('.').collect();
    ensure!(labels.len() >= 2, "account zone must contain two labels");
    for label in labels {
        ensure!(
            !label.is_empty()
                && label.len() <= 63
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
                && !label.starts_with('-')
                && !label.ends_with('-'),
            "account zone contains an invalid DNS label"
        );
    }
    Ok(zone.to_owned())
}

fn host_label(host: &str, zone: &str) -> Option<String> {
    let authority: Authority = host.parse().ok()?;
    let hostname = authority.host().trim_end_matches('.');
    let suffix = format!(".{zone}");
    let label = hostname.strip_suffix(&suffix)?;
    if label.contains('.') {
        return None;
    }
    AccountLabel::parse(label).ok().map(|_| label.to_owned())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use axum::{body::to_bytes, http::Request};
    use tower::ServiceExt;

    fn directory() -> Arc<StaticHostedAccountHttpDirectory> {
        Arc::new(
            StaticHostedAccountHttpDirectory::new([HostedAccountHttpArtifacts::new(
                "alice",
                "did:web:alice.tormentnexus.social",
                b"sealed-did-document".to_vec(),
                b"did:web:alice.tormentnexus.social".to_vec(),
                b"sealed-did-log".to_vec(),
            )
            .unwrap()])
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn serves_exact_sealed_bytes_by_host_label() {
        let app = router("tormentnexus.social", directory()).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/.well-known/did.json")
                    .header(header::HOST, "alice.tormentnexus.social")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/did+json"
        );
        assert_eq!(
            to_bytes(response.into_body(), 1024).await.unwrap().as_ref(),
            b"sealed-did-document"
        );
    }

    #[tokio::test]
    async fn serves_atproto_did_and_did_log_without_transcoding() {
        let app = router("tormentnexus.social", directory()).unwrap();
        for (path, expected, content_type) in [
            (
                "/.well-known/atproto-did",
                &b"did:web:alice.tormentnexus.social"[..],
                "text/plain; charset=utf-8",
            ),
            (
                "/.well-known/did-log.json",
                &b"sealed-did-log"[..],
                "application/cbor",
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .header(header::HOST, "alice.tormentnexus.social:443")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()[header::CONTENT_TYPE], content_type);
            assert_eq!(
                to_bytes(response.into_body(), 1024).await.unwrap(),
                expected
            );
        }
    }

    #[tokio::test]
    async fn unknown_or_malformed_hosts_are_not_synthesized() {
        let app = router("tormentnexus.social", directory()).unwrap();
        for host in [
            "bob.tormentnexus.social",
            "alice.other.example",
            "nested.alice.tormentnexus.social",
            "-alice.tormentnexus.social",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/.well-known/did.json")
                        .header(header::HOST, host)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "host={host}");
        }
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/.well-known/did.json")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn credentials_are_rejected_before_lookup() {
        let app = router("tormentnexus.social", directory()).unwrap();
        for name in [header::AUTHORIZATION, header::COOKIE] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/.well-known/did.json")
                        .header(header::HOST, "alice.tormentnexus.social")
                        .header(name, "present")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[test]
    fn duplicate_labels_are_rejected() {
        let first = HostedAccountHttpArtifacts::new(
            "alice",
            "did:web:alice.tormentnexus.social",
            vec![1],
            b"did:web:alice.tormentnexus.social".to_vec(),
            vec![2],
        )
        .unwrap();
        let second = HostedAccountHttpArtifacts::new(
            "alice",
            "did:web:alice.tormentnexus.social",
            vec![3],
            b"did:web:alice.tormentnexus.social".to_vec(),
            vec![4],
        )
        .unwrap();
        assert!(StaticHostedAccountHttpDirectory::new([first, second]).is_err());
    }
}
