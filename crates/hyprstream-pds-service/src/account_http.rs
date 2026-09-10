//! Credential-free HTTP face for hosted-account identity artifacts (#1165).
//!
//! The router in this module is intentionally a complete, small Axum router
//! with no session, cookie, bearer, or authorization middleware. It serves
//! immutable bytes loaded from sealed publication artifacts on a separately
//! bound listener. The listener and TLS termination are deployment concerns;
//! this module owns the routing and credential-free security contract.

use std::{collections::BTreeMap, sync::Arc};

use anyhow::{Result, ensure};
use async_trait::async_trait;
use axum::{
    Router,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header, uri::Authority},
    response::{IntoResponse, Response},
    routing::get,
};
use hyprstream_pds::{
    AccountLabel, AccountRecord, GenesisDidOp, SealedHostedAccount, SealedHostedDidDocument,
};

use crate::{
    AccountRecordStore, OAUTH_ACCOUNT_RESOLVER_SUBJECT, PDS_ACCOUNT_DID_DOCUMENT_FILE,
    PDS_ACCOUNT_DID_LOG_FILE, PDS_ACCOUNT_RECORD_FILE,
};

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

/// Live adapter that loads the immutable artifact bundle from the authorized
/// PDS mount. The OAuth service identity is fixed at construction; callers
/// cannot substitute a tenant or subject through an HTTP request.
#[derive(Clone)]
pub struct MountedHostedAccountHttpDirectory {
    store: Arc<AccountRecordStore>,
    authority: hyprstream_rpc::Subject,
    zone: String,
}

impl MountedHostedAccountHttpDirectory {
    pub fn new(
        store: Arc<AccountRecordStore>,
        authority: hyprstream_rpc::Subject,
        zone: impl Into<String>,
    ) -> Result<Self> {
        ensure!(
            authority.name() == Some(OAUTH_ACCOUNT_RESOLVER_SUBJECT),
            "hosted HTTP directory requires the fixed OAuth resolver subject"
        );
        let directory = Self {
            store,
            authority,
            zone: canonical_zone(&zone.into())?,
        };
        directory
            .store
            .schedule_hosted_did_index_refresh(directory.authority.clone());
        Ok(directory)
    }
}

#[async_trait]
impl HostedAccountHttpDirectory for MountedHostedAccountHttpDirectory {
    async fn lookup(&self, label: &str) -> Result<Option<Arc<HostedAccountHttpArtifacts>>> {
        AccountLabel::parse(label).map_err(|error| anyhow::anyhow!(error))?;
        let did = format!("did:web:{label}.{}", self.zone);
        // The store keeps a refreshed immutable label/DID index, so each
        // request performs one bounded map lookup. Index refresh is serialized
        // independently and never holds the request lookup lock.
        let tenant = self
            .store
            .resolve_tenant_for_hosted_did(&self.authority, &did)
            .await?;
        let Some(tenant) = tenant else {
            return Ok(None);
        };
        let record_bytes = self
            .store
            .read_hosted_http_artifact(
                &self.authority,
                &tenant,
                label,
                PDS_ACCOUNT_RECORD_FILE,
                16 * 1024,
            )
            .await?;
        let record = AccountRecord::from_dag_cbor(&record_bytes)?;
        ensure!(
            record.name().label() == label && record.name().did() == did,
            "hosted account record does not match host"
        );
        let document = self
            .store
            .read_hosted_http_artifact(
                &self.authority,
                &tenant,
                label,
                PDS_ACCOUNT_DID_DOCUMENT_FILE,
                16 * 1024,
            )
            .await?;
        let parsed = SealedHostedDidDocument::from_canonical_json(&document)?;
        ensure!(
            parsed.did() == did && parsed.cid() == record.doc_cid(),
            "served DID document does not match host or account record"
        );
        let log = self
            .store
            .read_hosted_http_artifact(
                &self.authority,
                &tenant,
                label,
                PDS_ACCOUNT_DID_LOG_FILE,
                64 * 1024,
            )
            .await?;
        let genesis = GenesisDidOp::from_dag_cbor(&log)?;
        ensure!(
            genesis.cid()? == record.genesis_op(),
            "served genesis operation does not match account record"
        );
        ensure!(
            genesis.unsigned().did() == did,
            "served genesis operation does not match hosted DID"
        );
        ensure!(
            genesis.unsigned().doc_cid() == record.doc_cid(),
            "served genesis operation does not match account document"
        );
        Ok(Some(Arc::new(HostedAccountHttpArtifacts::new(
            label,
            did.clone(),
            document,
            did.into_bytes(),
            log,
        )?)))
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
    request: Request,
) -> Response {
    serve_artifact(&state, request, ArtifactKind::DidDocument).await
}

async fn serve_atproto_did(
    State(state): State<Arc<AccountHttpState>>,
    request: Request,
) -> Response {
    serve_artifact(&state, request, ArtifactKind::AtprotoDid).await
}

async fn serve_did_log(State(state): State<Arc<AccountHttpState>>, request: Request) -> Response {
    serve_artifact(&state, request, ArtifactKind::DidLog).await
}

#[derive(Clone, Copy)]
enum ArtifactKind {
    DidDocument,
    AtprotoDid,
    DidLog,
}

async fn serve_artifact(
    state: &AccountHttpState,
    request: Request,
    kind: ArtifactKind,
) -> Response {
    // A credential-bearing request must never reach an artifact lookup. This
    // explicit rejection is the testable security boundary for the separate
    // origin; there is no cookie/session/bearer middleware to accidentally
    // inherit from the authenticated service.
    let label = {
        let headers: &HeaderMap = request.headers();
        if headers.contains_key(header::AUTHORIZATION) || headers.contains_key(header::COOKIE) {
            return (StatusCode::BAD_REQUEST, "credentials are not accepted").into_response();
        }
        // HTTP/1.1 supplies Host, while HTTP/2 carries the origin in the
        // :authority pseudo-header, which `http` exposes as the URI authority.
        // When both are present, they must be byte-identical: accepting a
        // mismatched Host would let a proxy route one public account while the
        // request authority names another. Only an absent URI authority may
        // use Host as its target.
        let uri_authority = request.uri().authority().map(Authority::as_str);
        let host_header = headers
            .get(header::HOST)
            .map(|value| value.to_str().unwrap_or_default());
        if headers.contains_key(header::HOST) && host_header == Some("") {
            return StatusCode::BAD_REQUEST.into_response();
        }
        let host = match (uri_authority, host_header) {
            (Some(authority), Some(host))
                if !equivalent_authorities(authority, host, &state.zone) =>
            {
                return StatusCode::BAD_REQUEST.into_response();
            }
            (Some(authority), _) => authority,
            (None, Some(host)) => host,
            (None, None) => return StatusCode::BAD_REQUEST.into_response(),
        };
        if host.is_empty() {
            return negative_response(StatusCode::BAD_REQUEST);
        }
        let Some(label) = host_label(host, &state.zone) else {
            return negative_response(StatusCode::NOT_FOUND);
        };
        label
    };
    drop(request);
    let artifact = match state.directory.lookup(&label).await {
        Ok(Some(artifact)) => artifact,
        Ok(None) => return negative_response(StatusCode::NOT_FOUND),
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

fn negative_response(status: StatusCode) -> Response {
    let mut response = status.into_response();
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store"),
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
    if host.contains('@') {
        return None;
    }
    let authority: Authority = host.parse().ok()?;
    if host.contains(':') {
        let port = authority.port()?;
        if !port.as_str().bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
    }
    let hostname = authority.host().trim_end_matches('.').to_ascii_lowercase();
    let suffix = format!(".{zone}");
    let label = hostname.strip_suffix(&suffix)?;
    if label.contains('.') {
        return None;
    }
    AccountLabel::parse(label).ok().map(|_| label.to_owned())
}

fn equivalent_authorities(left: &str, right: &str, zone: &str) -> bool {
    let Some(left_label) = host_label(left, zone) else {
        return false;
    };
    let Some(right_label) = host_label(right, zone) else {
        return false;
    };
    if left_label != right_label {
        return false;
    }
    let Ok(left_authority) = left.parse::<Authority>() else {
        return false;
    };
    let Ok(right_authority) = right.parse::<Authority>() else {
        return false;
    };
    left_authority.port().map(|port| port.as_str().to_owned())
        == right_authority.port().map(|port| port.as_str().to_owned())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use axum::{body::to_bytes, http::Request};
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes};
    use hyprstream_pds::did_op::{
        GenesisRepoHead, GenesisRotationKeys, HostKeyEnrollment, HybridRotationKey,
        RecoveryKeyEnrollment, UserRotationKey, sign_genesis,
    };
    use hyprstream_pds::{
        AllocatedAccountName, Cid, DID_DOCUMENT_FILE, GENESIS_DID_OP_FILE, HostedAccountMint,
    };
    use hyprstream_pds::dag_cbor::DagCbor;
    use hyprstream_rpc::Subject;
    use hyprstream_rpc::auth::mac::{MacDecision, SecurityContext};
    use hyprstream_vfs::{SyntheticMount, SyntheticNode};
    use rand::rngs::OsRng;
    use tower::ServiceExt;

    use crate::{PDS_ACCOUNT_RECORD_FILE, PDS_ACCOUNTS_DIRECTORY};

    struct PermitReads;

    impl crate::AccountRecordReadAuthorizer for PermitReads {
        fn check_read(
            &self,
            _subject: &Subject,
            _verified_tenant: Option<&str>,
            _security_context: Option<&SecurityContext>,
            _object_id: &str,
        ) -> MacDecision {
            MacDecision::Permit
        }
    }

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

    fn mounted_directory() -> (
        Arc<MountedHostedAccountHttpDirectory>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
    ) {
        mounted_directory_with_files(None, None, None)
    }

    fn mounted_directory_with_files(
        record_override: Option<Vec<u8>>,
        document_override: Option<Vec<u8>>,
        log_override: Option<Vec<u8>>,
    ) -> (
        Arc<MountedHostedAccountHttpDirectory>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
    ) {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq, pq_vk) = ml_dsa_generate_keypair();
        let hybrid =
            HybridRotationKey::new(ed.verifying_key().to_bytes(), ml_dsa_vk_bytes(&pq_vk)).unwrap();
        let rotations = GenesisRotationKeys::new(
            UserRotationKey::new(hybrid),
            RecoveryKeyEnrollment::Declined,
            HostKeyEnrollment::Absent,
        )
        .unwrap();
        let name =
            AllocatedAccountName::new("alice", "did:web:alice.tormentnexus.social".to_owned())
                .unwrap();
        let mint = HostedAccountMint::begin(name, rotations).unwrap();
        let document = mint.seal_did_document("https://pds.example.com").unwrap();
        let pending = mint
            .prepare_genesis(document, GenesisRepoHead::EmptyRepo)
            .unwrap();
        let signature = sign_genesis(pending.unsigned_genesis(), &ed, &pq).unwrap();
        let account = pending.seal(signature).unwrap();
        let record_bytes = account.record_bytes().to_vec();
        let document_bytes = account.did_document().as_bytes().to_vec();
        let log_bytes = account.genesis_bytes().to_vec();
        let record_file = record_override.unwrap_or_else(|| record_bytes.clone());
        let document_file = document_override.unwrap_or_else(|| document_bytes.clone());
        let log_file = log_override.unwrap_or_else(|| log_bytes.clone());
        let root = SyntheticNode::dir().with_child(
            "tenant",
            SyntheticNode::dir().with_child(
                PDS_ACCOUNTS_DIRECTORY,
                SyntheticNode::dir().with_child(
                    "alice",
                    SyntheticNode::dir()
                        .with_child(
                            PDS_ACCOUNT_RECORD_FILE,
                            SyntheticNode::file(record_file),
                        )
                        .with_child(DID_DOCUMENT_FILE, SyntheticNode::file(document_file))
                        .with_child(GENESIS_DID_OP_FILE, SyntheticNode::file(log_file)),
                ),
            ),
        );
        let store = Arc::new(crate::AccountRecordStore::new(
            Arc::new(SyntheticMount::new(root)),
            Arc::new(PermitReads),
        ));
        let directory = Arc::new(
            MountedHostedAccountHttpDirectory::new(
                store,
                Subject::new(crate::OAUTH_ACCOUNT_RESOLVER_SUBJECT),
                DEFAULT_ACCOUNT_ZONE,
            )
            .unwrap(),
        );
        (directory, document_bytes, log_bytes, record_bytes)
    }

    fn alternate_genesis_same_did() -> Vec<u8> {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq, pq_vk) = ml_dsa_generate_keypair();
        let hybrid =
            HybridRotationKey::new(ed.verifying_key().to_bytes(), ml_dsa_vk_bytes(&pq_vk)).unwrap();
        let rotations = GenesisRotationKeys::new(
            UserRotationKey::new(hybrid),
            RecoveryKeyEnrollment::Declined,
            HostKeyEnrollment::Absent,
        )
        .unwrap();
        let name =
            AllocatedAccountName::new("alice", "did:web:alice.tormentnexus.social".to_owned())
                .unwrap();
        let mint = HostedAccountMint::begin(name, rotations).unwrap();
        let document = mint.seal_did_document("https://alternate.example.com").unwrap();
        let pending = mint
            .prepare_genesis(document, GenesisRepoHead::EmptyRepo)
            .unwrap();
        let signature = sign_genesis(pending.unsigned_genesis(), &ed, &pq).unwrap();
        pending.seal(signature).unwrap().genesis_bytes().to_vec()
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
    async fn authority_only_requests_use_http2_uri_authority() {
        let app = router("tormentnexus.social", directory()).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("https://alice.tormentnexus.social/.well-known/did.json")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            to_bytes(response.into_body(), 1024).await.unwrap().as_ref(),
            b"sealed-did-document"
        );
    }

    #[tokio::test]
    async fn mismatched_http2_authority_and_host_fail_closed() {
        let app = router("tormentnexus.social", directory()).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("https://bob.tormentnexus.social/.well-known/did.json")
                    .header(header::HOST, "alice.tormentnexus.social")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn equivalent_http2_authority_and_host_ignore_dns_case() {
        let app = router("tormentnexus.social", directory()).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("https://alice.tormentnexus.social/.well-known/did.json")
                    .header(header::HOST, "ALICE.TORMENTNEXUS.SOCIAL")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn authority_only_requests_still_reject_credentials() {
        let app = router("tormentnexus.social", directory()).unwrap();
        for name in [header::AUTHORIZATION, header::COOKIE] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("https://alice.tormentnexus.social/.well-known/did.json")
                        .header(name, "present")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn hostname_matching_is_case_insensitive() {
        let app = router("tormentnexus.social", directory()).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/.well-known/did.json")
                    .header(header::HOST, "ALICE.TORMENTNEXUS.SOCIAL")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unallocated_label_404_is_not_cached() {
        let app = router("tormentnexus.social", directory()).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/.well-known/did.json")
                    .header(header::HOST, "bob.tormentnexus.social")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
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
            "attacker@alice.tormentnexus.social",
            "alice.tormentnexus.social:bad",
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

    #[tokio::test]
    async fn mounted_directory_reads_only_authorized_sealed_files() {
        let (directory, document, log, _record) = mounted_directory();
        directory
            .store
            .refresh_hosted_did_index(&directory.authority)
            .await
            .unwrap();
        let app = router(DEFAULT_ACCOUNT_ZONE, directory).unwrap();
        let did_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/.well-known/did.json")
                    .header(header::HOST, "alice.tormentnexus.social")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(did_response.status(), StatusCode::OK);
        assert_eq!(
            to_bytes(did_response.into_body(), 16 * 1024)
                .await
                .unwrap()
                .as_ref(),
            document
        );
        let log_response = app
            .oneshot(
                Request::builder()
                    .uri("/.well-known/did-log.json")
                    .header(header::HOST, "alice.tormentnexus.social")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(log_response.status(), StatusCode::OK);
        assert_eq!(
            to_bytes(log_response.into_body(), 64 * 1024)
                .await
                .unwrap()
                .as_ref(),
            log
        );
    }

    #[tokio::test]
    async fn mounted_directory_unallocated_404_is_not_cached() {
        let (directory, _document, _log, _record) = mounted_directory();
        directory
            .store
            .refresh_hosted_did_index(&directory.authority)
            .await
            .unwrap();
        let response = router(DEFAULT_ACCOUNT_ZONE, directory)
            .unwrap()
            .oneshot(
                Request::builder()
                    .uri("/.well-known/did.json")
                    .header(header::HOST, "bob.tormentnexus.social")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
    }

    #[tokio::test]
    async fn mounted_directory_rejects_artifacts_rebound_from_account_record() {
        let (_directory, document, log, _record) = mounted_directory();

        let mut altered_document = String::from_utf8(document).unwrap();
        assert!(altered_document.contains("https://pds.example.com"));
        altered_document =
            altered_document.replace("https://pds.example.com", "https://other.example.com");
        let document_directory = mounted_directory_with_files(
            None,
            Some(altered_document.into_bytes()),
            Some(log.clone()),
        )
        .0;
        document_directory
            .store
            .refresh_hosted_did_index(&document_directory.authority)
            .await
            .unwrap();
        assert!(document_directory.lookup("alice").await.is_err());

        let mut altered_log = log;
        let last = altered_log.len() - 1;
        altered_log[last] ^= 1;
        let log_directory = mounted_directory_with_files(None, None, Some(altered_log)).0;
        log_directory
            .store
            .refresh_hosted_did_index(&log_directory.authority)
            .await
            .unwrap();
        assert!(log_directory.lookup("alice").await.is_err());
    }

    #[tokio::test]
    async fn mounted_directory_rejects_genesis_rebound_from_document() {
        let (_original, document, _log, record) = mounted_directory();
        let alternate_log = alternate_genesis_same_did();
        let alternate_cid = Cid::from_dag_cbor(&alternate_log);
        let mut record_value = DagCbor::decode(&record).unwrap();
        let fields = match &mut record_value {
            DagCbor::Map(fields) => fields,
            other => panic!("account record must be a map, got {other:?}"),
        };
        let mut replaced = false;
        for (key, value) in fields {
            if matches!(key, DagCbor::Text(name) if name == "genesis_op") {
                *value = DagCbor::Link(alternate_cid);
                replaced = true;
            }
        }
        assert!(replaced, "account record must contain genesis_op");
        let rebound_record = record_value.encode();
        let rebound = mounted_directory_with_files(
            Some(rebound_record),
            Some(document),
            Some(alternate_log),
        )
        .0;
        rebound
            .store
            .refresh_hosted_did_index(&rebound.authority)
            .await
            .unwrap();
        let error = rebound
            .lookup("alice")
            .await
            .expect_err("genesis for another document must be rejected");
        assert!(error.to_string().contains("account document"));
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
