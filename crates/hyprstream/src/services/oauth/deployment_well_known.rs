//! Deployment static well-known endpoints (#1137 / #1136 serving half).
//!
//! The DID-anchored trust bootstrap (#1143) fetches three public,
//! integrity-anchored documents from the deployment's did:web host:
//!
//! - `/.well-known/did.json` — the deployment DID document (alsoKnownAs →
//!   cluster `did:at9p`, exactly one Ed25519 Multikey deployment CA,
//!   Iroh/QUIC transport reach);
//! - `/.well-known/at9p/<cid>.cbor` — the cluster at9p genesis capsule
//!   (self-certifying: BLAKE3-512 over its canonical bytes IS the DID);
//! - `/.well-known/deployment/registry-service.jwt` — the current CA-signed
//!   registry deployment credential (one-hour freshness profile).
//!
//! None of these are secret: every one is verified by the fetcher (GATE
//! canon → hash → sig for the capsule; mutual alsoKnownAs attestation for
//! the document; CA signature + numeric-date profile for the credential).
//! Serving them is therefore plain static-file work — but it must happen
//! SOMEWHERE, and the OAuth service is the deliberate choice: it already
//! terminates the did:web document surface (`/.well-known/did.json`), and it
//! is the only genuinely dual-stack (HTTP + RPC) service in the mesh, so the
//! deployment's `did:web` host can terminate here directly (e.g. via the
//! stack's Caddy reverse proxy).
//!
//! The operator provisions a deployment well-known directory:
//!
//! ```text
//! <dir>/did.json
//! <dir>/at9p/<cid>.cbor
//! <dir>/deployment/registry-service.jwt   (refreshed hourly by the CA holder)
//! ```
//!
//! Files are re-read on EVERY request so the external credential-refresh
//! agent needs no service restart. Missing/unreadable files are 404 — the
//! bootstrap fails closed, never on stale in-memory copies.

use std::path::{Path, PathBuf};

use axum::{
    extract::{Path as AxumPath, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use tokio::io::AsyncReadExt;

/// Capsule cap, matching the client's `MAX_CAPSULE_BYTES`.
const MAX_CAPSULE_BYTES: u64 = 4 * 1024 * 1024;
/// Registry credential cap, matching the client's `MAX_CREDENTIAL_BYTES`.
const MAX_CREDENTIAL_BYTES: u64 = 64 * 1024;
/// DID document cap (a deployment document is a few KiB of JSON).
const MAX_DID_DOC_BYTES: u64 = 1024 * 1024;

/// Mount the deployment well-known routes. Always registered; each handler
/// 404s when no deployment directory is configured, so an OAuth instance
/// that is not the deployment terminator answers exactly like one where the
/// material is absent — no information about configuration leaks.
pub fn router<S>(deployment_dir: Option<PathBuf>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/.well-known/did.json", get(serve_did_document))
        .route("/.well-known/at9p/:cid.cbor", get(serve_capsule))
        .route(
            "/.well-known/deployment/registry-service.jwt",
            get(serve_registry_credential),
        )
        .with_state(deployment_dir)
}

/// Serve the deployment directory's `did.json`. Mounted only in deployment
/// mode (see `create_app`): a missing document 404s — deployment mode was
/// explicitly selected, so a missing document is an operator error, never a
/// reason to silently serve the dynamic node document the bootstrap would
/// rightly reject.
async fn serve_did_document(State(deployment_dir): State<Option<PathBuf>>) -> Response {
    let Some(ref dir) = deployment_dir else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(
        &dir.join("did.json"),
        "application/did+json",
        MAX_DID_DOC_BYTES,
    )
    .await
}

async fn serve_capsule(
    State(deployment_dir): State<Option<PathBuf>>,
    AxumPath(cid): AxumPath<String>,
) -> Response {
    let cid = match cid.strip_suffix(".cbor") {
        Some(cid) => cid,
        None => return StatusCode::NOT_FOUND.into_response(),
    };
    // The client constrains the CID to lowercase alphanumerics before it ever
    // builds this URL; mirror that here so the parameter can never become a
    // path-traversal primitive.
    if cid.is_empty()
        || !cid
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(ref dir) = deployment_dir else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(
        &dir.join("at9p").join(format!("{cid}.cbor")),
        "application/cbor",
        MAX_CAPSULE_BYTES,
    )
    .await
}

async fn serve_registry_credential(State(deployment_dir): State<Option<PathBuf>>) -> Response {
    let Some(ref dir) = deployment_dir else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(
        &dir.join("deployment").join("registry-service.jwt"),
        "application/jwt",
        MAX_CREDENTIAL_BYTES,
    )
    .await
}

/// Read and serve one file with a hard size cap. Any failure — missing,
/// unreadable, oversize — is a plain 404: these endpoints reveal nothing
/// about filesystem state beyond existence, and the bootstrap treats every
/// one of them as "no deployment material served."
async fn serve_file(path: &Path, content_type: &'static str, max_bytes: u64) -> Response {
    match read_capped(path, max_bytes).await {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, content_type),
                (header::CACHE_CONTROL, "no-store"),
            ],
            bytes,
        )
            .into_response(),
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "deployment well-known read failed");
            StatusCode::NOT_FOUND.into_response()
        }
    }
}

async fn read_capped(path: &Path, max_bytes: u64) -> anyhow::Result<Vec<u8>> {
    let file = tokio::fs::File::open(path).await?;
    let mut limited = file.take(max_bytes + 1);
    let mut bytes = Vec::new();
    limited.read_to_end(&mut bytes).await?;
    anyhow::ensure!(
        bytes.len() <= max_bytes as usize,
        "file exceeds {max_bytes}-byte limit"
    );
    Ok(bytes)
}
