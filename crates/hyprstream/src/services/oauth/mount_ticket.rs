//! 9P mount-ticket minting for K3a.
//!
//! A mount ticket is a short-lived JWT derived from an already verified
//! OAuth/WIT caller. It is bound to a specific 9P plane and namespace path,
//! then consumed exactly once at 9P attach by the route layer.

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use hyprstream_pds::repo_authority::is_path_form_did_web;
use serde::{Deserialize, Serialize};

use super::state::OAuthState;
use crate::server::middleware::AuthenticatedUser;

pub const MOUNT_TICKET_TTL: i64 = 5 * 60;
pub const MOUNT_CAP_PREFIX: &str = "mount@9p://";

#[derive(Debug, Deserialize)]
pub struct MountTicketRequest {
    /// 9P plane name, normally `ws` or `webtransport`.
    pub plane: String,
    /// Namespace path being mounted. The minted token is not valid for broader
    /// paths; callers should request the narrowest path they need.
    pub namespace_path: String,
    /// Reserved for future sender-bound mount tickets. Rejected until the 9P
    /// attach path verifies proof-of-possession for this key.
    #[serde(default)]
    pub pubkey: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MountTicketResponse {
    pub ticket: String,
    pub token_type: &'static str,
    pub expires_in: i64,
    pub audience: String,
    pub capability: String,
}

/// `POST /oauth/mount-ticket` — mint a 9P attach credential.
pub async fn issue_mount_ticket(
    State(state): State<Arc<OAuthState>>,
    Extension(user): Extension<AuthenticatedUser>,
    headers: HeaderMap,
    Json(body): Json<MountTicketRequest>,
) -> Response {
    if !valid_plane(&body.plane) {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("plane must contain only [A-Za-z0-9._-]"),
        );
    }
    if body.pubkey.is_some() || headers.contains_key("DPoP") {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some(
                "sender-bound mount tickets are not supported until 9P attach verifies proof-of-possession",
            ),
        );
    }
    let normalized_path = match normalize_namespace_path(&body.namespace_path) {
        Some(path) => path,
        None => {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                Some("namespace_path must be absolute and must not contain '..'"),
            );
        }
    };
    if normalized_path != "/" {
        return oauth_error(
            StatusCode::FORBIDDEN,
            "insufficient_scope",
            Some(
                "non-root namespace_path tickets require namespace authorization and are not enabled yet",
            ),
        );
    }

    if is_path_form_did_web(&user.user) {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            Some(
                "path-form did:web account subjects are frozen; host-form account minting is not available yet (#1159)",
            ),
        );
    }

    let key = match state.active_jwt_signing_key().await {
        Some(k) => k,
        None => {
            return oauth_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "temporarily_unavailable",
                Some("mount-ticket issuance not available"),
            );
        }
    };

    let now = chrono::Utc::now().timestamp();
    let exp = now + MOUNT_TICKET_TTL;
    let audience = state.issuer_url.clone();
    let capability = ticket_capability(&body.plane, &normalized_path);
    let domain = match user.authorization_domain() {
        Ok(domain) => domain,
        Err(_) => {
            return oauth_error(
                StatusCode::FORBIDDEN,
                "insufficient_scope",
                Some("Verified hosted-account tenant binding required"),
            );
        }
    };
    // `AuthenticatedUser.token` is populated only after the resource
    // middleware verifies the source JWT. Re-read its signed authorization
    // attributes and bind them to the already-verified subject/tenant; never
    // manufacture clearance from the Subject string.
    let source_claims = match user
        .token
        .as_deref()
        .and_then(|token| hyprstream_rpc::auth::decode_unverified(token).ok())
    {
        Some(claims)
            if claims.sub == user.user
                && claims.tenant.as_deref() == user.verified_tenant.as_deref() =>
        {
            claims
        }
        _ => {
            return oauth_error(
                StatusCode::FORBIDDEN,
                "insufficient_scope",
                Some("verified source Claims required for MAC-bound mount ticket"),
            );
        }
    };
    let Some(clearance) = source_claims.clearance else {
        return oauth_error(
            StatusCode::FORBIDDEN,
            "insufficient_scope",
            Some("source credential has no authority-stamped MAC clearance"),
        );
    };
    let mut claims = hyprstream_rpc::auth::Claims::new(user.user.clone(), now, exp)
        .with_jti()
        .with_issuer(state.issuer_url.clone())
        .with_audience(Some(audience.clone()))
        .with_cap(capability.clone())
        .with_credential_clearance(clearance);
    if domain != "*" {
        claims = claims.with_tenant(domain);
    }
    let ticket = hyprstream_rpc::auth::jwt::encode(&claims, &key);

    tracing::info!(
        sub = %user.user,
        plane = %body.plane,
        namespace_path = %normalized_path,
        "9P mount ticket issued"
    );

    (
        StatusCode::OK,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(MountTicketResponse {
            ticket,
            token_type: "mount-ticket+jwt",
            expires_in: MOUNT_TICKET_TTL,
            audience,
            capability,
        }),
    )
        .into_response()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod freeze_tests {
    use super::*;
    use crate::config::OAuthConfig;
    use crate::services::{DiscoveryClient, PolicyClient};
    use axum::extract::Extension;
    use hyprstream_rpc::rpc_client::RpcClientImpl;
    use hyprstream_rpc::signer::LocalSigner;
    use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

    fn test_state() -> Arc<OAuthState> {
        let key = ed25519_dalek::SigningKey::from_bytes(&[0x75; 32]);
        let dummy = std::path::PathBuf::from("/dev/null/mount-ticket-freeze-test.sock");
        let make_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(key.clone()),
                    LazyUdsTransport::new(dummy.clone()),
                    Some(key.verifying_key()),
                )
                .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical),
            )
        };
        Arc::new(
            OAuthState::new(
                &OAuthConfig::default(),
                PolicyClient::new(make_client()),
                DiscoveryClient::new(make_client()),
                key.verifying_key().to_bytes(),
            )
            .with_ca_jwt_key(key),
        )
    }

    fn request() -> MountTicketRequest {
        MountTicketRequest {
            plane: "webtransport".to_owned(),
            namespace_path: "/".to_owned(),
            pubkey: None,
        }
    }

    #[tokio::test]
    async fn mount_ticket_rejects_path_form_authenticated_user_before_signing() {
        let response = issue_mount_ticket(
            State(test_state()),
            Extension(AuthenticatedUser {
                user: "did:web:accounts.example:users:alice".to_owned(),
                verified_tenant: Some("tenant-a.example".to_owned()),
                token: None,
                exp: None,
            }),
            HeaderMap::new(),
            Json(request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn mount_ticket_allows_ordinary_authenticated_user() {
        let source = hyprstream_rpc::auth::Claims::new(
            "alice".to_owned(),
            chrono::Utc::now().timestamp(),
            chrono::Utc::now().timestamp() + 300,
        )
        .with_issuer(test_state().issuer_url.clone())
        .with_tenant("tenant-a.example".to_owned())
        .with_clearance(hyprstream_rpc::auth::mac::SecurityLabel::new(
            hyprstream_rpc::auth::mac::Level::Secret,
            hyprstream_rpc::auth::mac::Assurance::Classical,
            hyprstream_rpc::auth::mac::CompartmentSet::EMPTY,
        ));
        let source_token =
            crate::auth::jwt::encode(&source, &ed25519_dalek::SigningKey::from_bytes(&[0x75; 32]));
        let response = issue_mount_ticket(
            State(test_state()),
            Extension(AuthenticatedUser {
                user: "alice".to_owned(),
                verified_tenant: Some("tenant-a.example".to_owned()),
                token: Some(source_token),
                exp: None,
            }),
            HeaderMap::new(),
            Json(request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let claims = crate::auth::jwt::decode(
            json["ticket"].as_str().unwrap(),
            &ed25519_dalek::SigningKey::from_bytes(&[0x75; 32]).verifying_key(),
            None,
        )
        .unwrap();
        assert!(claims.jti.is_some(), "issued mount tickets must be one-use");
        assert_eq!(claims.tenant.as_deref(), Some("tenant-a.example"));
        assert_eq!(claims.clearance, source.clearance);
    }
}

pub fn ticket_capability(plane: &str, namespace_path: &str) -> String {
    format!("{MOUNT_CAP_PREFIX}{plane}{namespace_path}")
}

pub fn is_mount_ticket_claims(claims: &hyprstream_rpc::auth::Claims) -> bool {
    claims
        .cap
        .as_deref()
        .is_some_and(|cap| cap.starts_with(MOUNT_CAP_PREFIX))
}

pub fn is_mount_ticket_for(
    claims: &hyprstream_rpc::auth::Claims,
    plane: &str,
    namespace_path: &str,
) -> bool {
    claims
        .cap
        .as_deref()
        .is_some_and(|cap| cap == ticket_capability(plane, namespace_path))
}

fn valid_plane(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

pub(crate) fn normalize_namespace_path(value: &str) -> Option<String> {
    if !value.starts_with('/') {
        return None;
    }
    let mut out = Vec::new();
    for component in value.split('/') {
        if component.is_empty() || component == "." {
            continue;
        }
        if component == ".." {
            return None;
        }
        out.push(component);
    }
    Some(format!("/{}", out.join("/")))
}

fn oauth_error(status: StatusCode, error: &str, description: Option<&str>) -> Response {
    let mut body = serde_json::Map::new();
    body.insert(
        "error".to_owned(),
        serde_json::Value::String(error.to_owned()),
    );
    if let Some(description) = description {
        body.insert(
            "error_description".to_owned(),
            serde_json::Value::String(description.to_owned()),
        );
    }
    (
        status,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(serde_json::Value::Object(body)),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_path_must_be_absolute_and_contained() {
        assert_eq!(
            normalize_namespace_path("/tenant/./data"),
            Some("/tenant/data".to_owned())
        );
        assert_eq!(normalize_namespace_path("/"), Some("/".to_owned()));
        assert_eq!(normalize_namespace_path("tenant/data"), None);
        assert_eq!(normalize_namespace_path("/tenant/../root"), None);
    }

    #[test]
    fn mount_capability_is_plane_and_path_bound() {
        assert_eq!(
            ticket_capability("webtransport", "/tenant/a"),
            "mount@9p://webtransport/tenant/a"
        );
    }

    #[test]
    fn mount_capability_match_is_exact() {
        let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), 1, 2)
            .with_cap(ticket_capability("webtransport", "/"));
        assert!(is_mount_ticket_for(&claims, "webtransport", "/"));
        assert!(!is_mount_ticket_for(&claims, "ws", "/"));
        assert!(!is_mount_ticket_for(&claims, "webtransport", "/tenant/a"));
    }
}
