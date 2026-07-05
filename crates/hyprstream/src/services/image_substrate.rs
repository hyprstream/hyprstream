//! HTTP image substrate for stock nydus-snapshotter consumers (#789).
//!
//! This is a thin read-only data plane over [`RafsStore`]. Ingest and dedup
//! remain owned by the worker image store; this module only projects already
//! pulled RAFS bootstraps and blob CAS objects over HTTP so cluster nodes can
//! fetch them without learning the host-local store layout.

use std::sync::Arc;

use axum::{
    extract::{Extension, Path, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use hyprstream_workers::image::RafsStore;

use crate::server::AuthenticatedUser;

/// Serving policy for the nydus image substrate.
#[derive(Clone, Debug, Default)]
pub struct ImageSubstratePolicy {
    /// Permit unauthenticated read-only blob/bootstrap fetches.
    ///
    /// This exists for nydus/containerd deployments where the snapshotter cannot
    /// yet forward WIT/ticket credentials. Keep it false for protected clusters.
    pub allow_anonymous_read: bool,
}

/// Shared axum state for image substrate routes.
#[derive(Clone)]
pub struct ImageSubstrateState {
    store: Arc<RafsStore>,
    policy: ImageSubstratePolicy,
}

impl ImageSubstrateState {
    pub fn new(store: Arc<RafsStore>, policy: ImageSubstratePolicy) -> Self {
        Self { store, policy }
    }
}

/// Build a read-only router exposing RAFS bootstraps and layer/config blobs.
///
/// Routes:
/// - `GET|HEAD /nydus/v1/bootstraps/:image_id`
/// - `GET|HEAD /nydus/v1/blobs/:algorithm/:hex`
/// - `GET|HEAD /v2/*path` for registry-compatible blob paths ending in
///   `/blobs/<algorithm>:<hex>`
pub fn create_image_substrate_router(
    store: Arc<RafsStore>,
    policy: ImageSubstratePolicy,
) -> Router {
    let state = ImageSubstrateState::new(store, policy);
    Router::new()
        .route(
            "/nydus/v1/bootstraps/:image_id",
            get(get_bootstrap).head(head_bootstrap),
        )
        .route(
            "/nydus/v1/blobs/:algorithm/:hex",
            get(get_blob_by_parts).head(head_blob_by_parts),
        )
        .route("/v2/*path", get(get_registry_blob).head(head_registry_blob))
        .with_state(state)
}

async fn get_bootstrap(
    State(state): State<ImageSubstrateState>,
    Path(image_id): Path<String>,
    user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
) -> Response {
    serve_bootstrap(state, image_id, user, headers, Method::GET).await
}

async fn head_bootstrap(
    State(state): State<ImageSubstrateState>,
    Path(image_id): Path<String>,
    user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
) -> Response {
    serve_bootstrap(state, image_id, user, headers, Method::HEAD).await
}

async fn get_blob_by_parts(
    State(state): State<ImageSubstrateState>,
    Path((algorithm, hex)): Path<(String, String)>,
    user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
) -> Response {
    serve_blob(
        state,
        format!("{algorithm}:{hex}"),
        user,
        headers,
        Method::GET,
    )
    .await
}

async fn head_blob_by_parts(
    State(state): State<ImageSubstrateState>,
    Path((algorithm, hex)): Path<(String, String)>,
    user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
) -> Response {
    serve_blob(
        state,
        format!("{algorithm}:{hex}"),
        user,
        headers,
        Method::HEAD,
    )
    .await
}

async fn get_registry_blob(
    State(state): State<ImageSubstrateState>,
    Path(path): Path<String>,
    user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
) -> Response {
    match digest_from_registry_path(&path) {
        Some(digest) => serve_blob(state, digest, user, headers, Method::GET).await,
        None => (StatusCode::NOT_FOUND, "registry path is not a blob").into_response(),
    }
}

async fn head_registry_blob(
    State(state): State<ImageSubstrateState>,
    Path(path): Path<String>,
    user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
) -> Response {
    match digest_from_registry_path(&path) {
        Some(digest) => serve_blob(state, digest, user, headers, Method::HEAD).await,
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn serve_bootstrap(
    state: ImageSubstrateState,
    image_id: String,
    user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
    method: Method,
) -> Response {
    if let Some(resp) = authorize(&state, user.as_ref()) {
        return resp;
    }
    if !is_sha256_digest(&image_id) {
        return (StatusCode::BAD_REQUEST, "image_id must be sha256:<hex>").into_response();
    }
    serve_file(
        state.store.bootstrap_path(&image_id),
        "application/vnd.nydus.rafs.bootstrap.v1",
        headers,
        method,
    )
    .await
}

async fn serve_blob(
    state: ImageSubstrateState,
    digest: String,
    user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
    method: Method,
) -> Response {
    if let Some(resp) = authorize(&state, user.as_ref()) {
        return resp;
    }
    if !is_sha256_digest(&digest) {
        return (StatusCode::BAD_REQUEST, "digest must be sha256:<hex>").into_response();
    }
    serve_file(
        state.store.blob_path(&digest),
        "application/octet-stream",
        headers,
        method,
    )
    .await
}

fn authorize(
    state: &ImageSubstrateState,
    user: Option<&Extension<AuthenticatedUser>>,
) -> Option<Response> {
    if state.policy.allow_anonymous_read || user.is_some() {
        None
    } else {
        let mut resp = (StatusCode::UNAUTHORIZED, "authentication required").into_response();
        resp.headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        Some(resp)
    }
}

async fn serve_file(
    path: std::path::PathBuf,
    content_type: &'static str,
    headers: HeaderMap,
    method: Method,
) -> Response {
    let bytes = match tokio::fs::read(&path).await {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return StatusCode::NOT_FOUND.into_response();
        }
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    let len = bytes.len() as u64;
    let range = parse_single_range(headers.get(header::RANGE), len);
    let mut status = StatusCode::OK;
    let mut start = 0usize;
    let mut end = bytes.len();
    if let Some(Ok((range_start, range_end))) = range {
        status = StatusCode::PARTIAL_CONTENT;
        start = range_start as usize;
        end = range_end as usize + 1;
    } else if matches!(range, Some(Err(()))) {
        let mut resp = StatusCode::RANGE_NOT_SATISFIABLE.into_response();
        resp.headers_mut().insert(
            header::CONTENT_RANGE,
            HeaderValue::from_str(&format!("bytes */{len}"))
                .unwrap_or_else(|_| HeaderValue::from_static("bytes */0")),
        );
        return resp;
    }

    let mut resp = if method == Method::HEAD {
        status.into_response()
    } else {
        (status, bytes[start..end].to_vec()).into_response()
    };
    let headers = resp.headers_mut();
    headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&(end - start).to_string())
            .unwrap_or_else(|_| HeaderValue::from_static("0")),
    );
    if status == StatusCode::PARTIAL_CONTENT {
        headers.insert(
            header::CONTENT_RANGE,
            HeaderValue::from_str(&format!("bytes {start}-{}/{len}", end - 1))
                .unwrap_or_else(|_| HeaderValue::from_static("bytes */0")),
        );
    }
    resp
}

fn digest_from_registry_path(path: &str) -> Option<String> {
    let (_repo, digest) = path.rsplit_once("/blobs/")?;
    Some(digest.to_owned())
}

fn is_sha256_digest(digest: &str) -> bool {
    let Some(hex) = digest.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())
}

fn parse_single_range(value: Option<&HeaderValue>, len: u64) -> Option<Result<(u64, u64), ()>> {
    let value = value?.to_str().ok()?;
    let spec = value.strip_prefix("bytes=")?;
    let (start, end) = spec.split_once('-')?;
    if len == 0 {
        return Some(Err(()));
    }
    if start.is_empty() {
        let suffix = end.parse::<u64>().ok()?;
        if suffix == 0 {
            return Some(Err(()));
        }
        let start = len.saturating_sub(suffix);
        return Some(Ok((start, len - 1)));
    }
    let start = start.parse::<u64>().ok()?;
    let end = if end.is_empty() {
        len - 1
    } else {
        end.parse::<u64>().ok()?
    };
    if start >= len || start > end {
        return Some(Err(()));
    }
    Some(Ok((start, end.min(len - 1))))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::http::Request;
    use hyprstream_workers::config::ImageConfig;
    use tower::ServiceExt;

    const DIGEST: &str = "sha256:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

    fn test_store(dir: &std::path::Path) -> Arc<RafsStore> {
        let config = ImageConfig {
            blobs_dir: dir.join("blobs"),
            bootstrap_dir: dir.join("bootstrap"),
            refs_dir: dir.join("refs"),
            cache_dir: dir.join("cache"),
            ..ImageConfig::default()
        };
        std::fs::create_dir_all(&config.blobs_dir).unwrap();
        std::fs::create_dir_all(&config.bootstrap_dir).unwrap();
        std::fs::create_dir_all(&config.refs_dir).unwrap();
        std::fs::create_dir_all(&config.cache_dir).unwrap();
        Arc::new(RafsStore::new(config).unwrap())
    }

    #[tokio::test]
    async fn serves_bootstrap_by_image_id() {
        let dir = tempfile::tempdir().unwrap();
        let store = test_store(dir.path());
        std::fs::write(store.bootstrap_path(DIGEST), b"bootstrap").unwrap();

        let app = create_image_substrate_router(
            store,
            ImageSubstratePolicy {
                allow_anonymous_read: true,
            },
        );
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/nydus/v1/bootstraps/{DIGEST}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"bootstrap");
    }

    #[tokio::test]
    async fn serves_registry_compatible_blob_path_with_ranges() {
        let dir = tempfile::tempdir().unwrap();
        let store = test_store(dir.path());
        std::fs::write(store.blob_path(DIGEST), b"0123456789").unwrap();

        let app = create_image_substrate_router(
            store,
            ImageSubstratePolicy {
                allow_anonymous_read: true,
            },
        );
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/v2/example/repo/blobs/{DIGEST}"))
                    .header(header::RANGE, "bytes=2-5")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            resp.headers().get(header::CONTENT_RANGE).unwrap(),
            "bytes 2-5/10"
        );
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"2345");
    }

    #[tokio::test]
    async fn anonymous_reads_are_opt_in() {
        let dir = tempfile::tempdir().unwrap();
        let store = test_store(dir.path());
        let app = create_image_substrate_router(store, ImageSubstratePolicy::default());

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/nydus/v1/blobs/sha256/{}",
                        &DIGEST["sha256:".len()..]
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
