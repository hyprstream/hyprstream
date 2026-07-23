//! SCIM 2.0 HTTP handlers for user provisioning.
//!
//! Implements RFC 7644 endpoints under `/scim/v2/Users`.
//! Delegates to `UserService` for shared CRUD logic.

use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use hex;

use super::state::OAuthState;
use super::user_service;
use crate::auth::{UserFilter, decode_pubkey_base64};

use super::scim_types::*;

// ─── Query parameters ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ListParams {
    pub filter: Option<String>,
    #[serde(rename = "sortBy")]
    pub sort_by: Option<String>,
    #[serde(rename = "sortOrder")]
    pub sort_order: Option<String>,
    pub count: Option<u32>,
    #[serde(rename = "startIndex")]
    pub start_index: Option<u32>,
    pub attributes: Option<String>,
    #[serde(rename = "excludedAttributes")]
    pub excluded_attributes: Option<String>,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn scim_error(status: u16, scim_type: &str, detail: &str) -> axum::response::Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST),
        [(header::CONTENT_TYPE, "application/scim+json")],
        Json(ScimError {
            schemas: vec![SCIM_API_ERROR.to_owned()],
            detail: Some(detail.to_owned()),
            status: status.to_string(),
            scim_type: Some(scim_type.to_owned()),
        }),
    )
        .into_response()
}

fn scim_error_simple(status: u16, detail: &str) -> axum::response::Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST),
        [(header::CONTENT_TYPE, "application/scim+json")],
        Json(ScimError {
            schemas: vec![SCIM_API_ERROR.to_owned()],
            detail: Some(detail.to_owned()),
            status: status.to_string(),
            scim_type: None,
        }),
    )
        .into_response()
}

fn user_service(state: &Arc<OAuthState>) -> Result<&Arc<user_service::UserService>, Box<axum::response::Response>> {
    state
        .user_service
        .as_ref()
        .ok_or_else(|| Box::new(scim_error_simple(503, "User service not configured")))
}

fn user_to_scim(
    info: &user_service::UserInfo,
    base_url: &str,
    include_pubkey: bool,
) -> ScimUser {
    let etag = compute_etag(&info.sub, info.active);
    ScimUser {
        schemas: {
            let mut s = vec![SCIM_SCHEMA_USER.to_owned()];
            if include_pubkey {
                s.push(SCIM_SCHEMA_EXT_HYPRSTREAM.to_owned());
            }
            s
        },
        id: info.sub.clone(),
        external_id: info.external_id.clone(),
        user_name: info.username.clone(),
        display_name: info.name.clone(),
        active: Some(info.active),
        emails: info.email.as_ref().map(|email| {
            vec![ScimEmail {
                value: email.clone(),
                primary: Some(info.email_verified),
                email_type: None,
            }]
        }),
        meta: ScimMeta {
            resource_type: "User".to_owned(),
            created: None,
            last_modified: None,
            location: Some(format!("{base_url}/scim/v2/Users/{}", info.sub)),
            version: Some(etag),
        },
        hyprstream: if include_pubkey && !info.pubkey_base64.is_empty() {
            Some(ScimHyprstreamExtension {
                pubkey_base64: info.pubkey_base64.clone(),
            })
        } else {
            None
        },
    }
}

fn compute_etag(sub: &str, active: bool) -> String {
    let mut hasher = Sha256::new();
    hasher.update(sub.as_bytes());
    hasher.update([active as u8]);
    let hash = hasher.finalize();
    format!("W/\"{}\"", hex::encode(hash))
}

/// Check If-Match header against current ETag. Returns `true` if OK to proceed.
fn check_if_match(headers: &HeaderMap, current_etag: &str) -> Result<(), Box<axum::response::Response>> {
    if let Some(if_match) = headers.get(header::IF_MATCH) {
        let if_match_str = if_match.to_str().map_err(|_| {
            Box::new(scim_error_simple(400, "Invalid If-Match header"))
        })?;
        // Support both "W/\"hash\"" and bare etag
        if if_match_str != current_etag && if_match_str != "*" {
            return Err(Box::new(scim_error(412, "invalidValue", "Resource version mismatch (ETag)")));
        }
    }
    Ok(())
}

/// Apply attribute projection to a ScimUser.
fn apply_attributes(user: &mut ScimUser, attributes: Option<&str>, excluded: Option<&str>) {
    // Always-returned: id, userName, meta (returned: always)
    if let Some(attrs) = attributes {
        let requested: Vec<&str> = attrs.split(',').map(str::trim).collect();
        // Clear fields not in the requested set
        if !requested.iter().any(|a| *a == "displayName" || *a == "name") {
            user.display_name = None;
        }
        if !requested.contains(&"emails") {
            user.emails = None;
        }
        if !requested.contains(&"active") {
            user.active = None;
        }
        if !requested.contains(&"externalId") {
            user.external_id = None;
        }
    }
    if let Some(excluded) = excluded {
        let excluded_set: Vec<&str> = excluded.split(',').map(str::trim).collect();
        if excluded_set.iter().any(|a| *a == "displayName" || *a == "name") {
            user.display_name = None;
        }
        if excluded_set.contains(&"emails") {
            user.emails = None;
        }
        if excluded_set.contains(&"active") {
            user.active = None;
        }
        if excluded_set.contains(&"externalId") {
            user.external_id = None;
        }
    }
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

fn base_url() -> String {
    crate::config::HyprConfig::load().unwrap_or_default().oauth.issuer_url()
}

/// Find a user by id (sub) or userName.
async fn find_user(svc: &user_service::UserService, id: &str) -> Option<user_service::UserInfo> {
    if let Ok(Some(i)) = svc.get(id).await {
        return Some(i);
    }
    // Try by sub/id
    let filter = UserFilter {
        filter: Some(format!("id eq \"{}\"", id)),
        ..Default::default()
    };
    svc.list(&filter).await.ok().and_then(|l| l.users.into_iter().next())
}

/// Build a SCIM JSON response with standard headers.
fn scim_response(status: StatusCode, body: &ScimUser, etag: &str) -> axum::response::Response {
    let mut resp = axum::response::Response::new(
        serde_json::to_string(body).unwrap_or_default().into(),
    );
    *resp.status_mut() = status;
    let headers = resp.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/scim+json"));
    if let Ok(val) = HeaderValue::from_str(etag) {
        headers.insert(header::ETAG, val);
    }
    resp
}

/// GET /scim/v2/Users — List/search users (RFC 7644 §3.4.2).
pub async fn list_users(
    State(state): State<Arc<OAuthState>>,
    Query(params): Query<ListParams>,
) -> axum::response::Response {
    let svc = match user_service(&state) {
        Ok(s) => s,
        Err(e) => return *e,
    };

    // Parse SCIM filter if provided
    let filter = UserFilter {
        filter: params.filter.clone(),
        active_only: None,
        count: params.count.map(|c| c as usize),
        start_index: params.start_index.map(|s| s as usize),
        sort_by: params.sort_by.clone(),
        sort_order: params.sort_order.clone(),
    };

    let list = match svc.list(&filter).await {
        Ok(l) => l,
        Err(e) => return scim_error_simple(500, &e.to_string()),
    };

    let base_url = {
        let config = crate::config::HyprConfig::load().unwrap_or_default();
        config.oauth.issuer_url()
    };

    let resources: Vec<ScimUser> = list
        .users
        .iter()
        .map(|u| {
            let mut scim = user_to_scim(u, &base_url, true);
            apply_attributes(
                &mut scim,
                params.attributes.as_deref(),
                params.excluded_attributes.as_deref(),
            );
            scim
        })
        .collect();

    let response = ScimListResponse {
        schemas: vec![SCIM_SCHEMA_LIST.to_owned()],
        total_results: list.total_results as u32,
        items_per_page: params.count,
        start_index: params.start_index,
        resources: Some(resources),
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/scim+json")],
        Json(response),
    )
        .into_response()
}

/// POST /scim/v2/Users — Create user (RFC 7644 §3.3).
pub async fn create_user(
    State(state): State<Arc<OAuthState>>,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    let svc = match user_service(&state) {
        Ok(s) => s,
        Err(e) => return *e,
    };

    let user_name = match body.get("userName").and_then(|v| v.as_str()) {
        Some(n) => n.to_owned(),
        None => return scim_error(400, "invalidValue", "userName is required"),
    };

    // Check for duplicate userName
    if let Ok(Some(_)) = svc.get(&user_name).await {
        return scim_error(409, "uniqueness", &format!("User '{}' already exists", user_name));
    }

    let pubkey_base64 = body
        .get("urn:ietf:params:scim:schemas:extension:hyprstream:1.0")
        .and_then(|ext| ext.get("pubkey_base64"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();

    let display_name = body.get("displayName").and_then(|v| v.as_str()).map(std::borrow::ToOwned::to_owned);
    let email = body
        .get("emails")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|e| e.get("value"))
        .and_then(|v| v.as_str())
        .map(std::borrow::ToOwned::to_owned);
    let external_id = body.get("externalId").and_then(|v| v.as_str()).map(std::borrow::ToOwned::to_owned);

    let _info = match svc.register(&user_name, &pubkey_base64).await {
        Ok(i) => i,
        Err(e) => return scim_error_simple(400, &e.to_string()),
    };

    // Set optional profile fields
    if display_name.is_some() || email.is_some() || external_id.is_some() {
        let update = user_service::UserUpdate {
            name: display_name.map(Some),
            email: email.map(Some),
            external_id: external_id.map(Some),
            email_verified: None,
            atproto_did: None,
        };
        if let Err(e) = svc.update(&user_name, update).await {
            return scim_error_simple(500, &e.to_string());
        }
    }

    let info = match svc.get(&user_name).await {
        Ok(Some(i)) => i,
        _ => return scim_error_simple(500, "User not found after creation"),
    };

    let base_url = {
        let config = crate::config::HyprConfig::load().unwrap_or_default();
        config.oauth.issuer_url()
    };
    let scim = user_to_scim(&info, &base_url, true);
    let etag = scim.meta.version.clone().unwrap_or_default();

    let mut resp = axum::response::Response::new(
        serde_json::to_string(&scim).unwrap_or_default().into(),
    );
    *resp.status_mut() = StatusCode::CREATED;
    let headers = resp.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/scim+json"));
    if let Ok(val) = HeaderValue::from_str(&format!("{base_url}/scim/v2/Users/{}", info.sub)) {
        headers.insert(header::LOCATION, val);
    }
    if let Ok(val) = HeaderValue::from_str(&etag) {
        headers.insert(header::ETAG, val);
    }
    resp
}

/// GET /scim/v2/Users/:id — Get single user (RFC 7644 §3.4.1).
pub async fn get_user(
    State(state): State<Arc<OAuthState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    let svc = match user_service(&state) {
        Ok(s) => s,
        Err(e) => return *e,
    };

    let info = find_user(svc, &id).await;
    let info = match info {
        Some(i) => i,
        None => return scim_error_simple(404, "Resource not found"),
    };

    let base_url = base_url();
    let scim = user_to_scim(&info, &base_url, true);
    let etag = scim.meta.version.clone().unwrap_or_default();

    scim_response(StatusCode::OK, &scim, &etag)
}

/// PUT /scim/v2/Users/:id — Replace user (RFC 7644 §3.5.1).
pub async fn replace_user(
    State(state): State<Arc<OAuthState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    let svc = match user_service(&state) {
        Ok(s) => s,
        Err(e) => return *e,
    };

    let info = match find_user(svc, &id).await {
        Some(i) => i,
        None => return scim_error_simple(404, "Resource not found"),
    };

    let current_etag = compute_etag(&info.sub, info.active);
    if let Err(e) = check_if_match(&headers, &current_etag) {
        return *e;
    }

    let display_name = body.get("displayName").and_then(|v| v.as_str()).map(std::borrow::ToOwned::to_owned);
    let email = body
        .get("emails")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|e| e.get("value"))
        .and_then(|v| v.as_str())
        .map(std::borrow::ToOwned::to_owned);
    let external_id = body.get("externalId").and_then(|v| v.as_str()).map(std::borrow::ToOwned::to_owned);
    let active = body.get("active").and_then(serde_json::Value::as_bool);

    let update = user_service::UserUpdate {
        name: display_name.map(Some),
        email: email.map(Some),
        external_id: external_id.map(Some),
        email_verified: None,
            atproto_did: None,
    };

    match svc.update(&info.username, update).await {
        Ok(updated) => {
            if let Some(want_active) = active {
                if want_active && !info.active {
                    let _ = svc.resume(&info.username).await;
                } else if !want_active && info.active {
                    let _ = svc.suspend(&info.username).await;
                }
            }

            let info = svc.get(&info.username).await.ok().flatten().unwrap_or(updated);
            let url = base_url();
            let scim = user_to_scim(&info, &url, true);
            let etag = scim.meta.version.clone().unwrap_or_default();
            scim_response(StatusCode::OK, &scim, &etag)
        }
        Err(e) => scim_error_simple(500, &e.to_string()),
    }
}

/// DELETE /scim/v2/Users/:id — Delete user (soft delete: set active=false) (RFC 7644 §3.6).
pub async fn delete_user(
    State(state): State<Arc<OAuthState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> axum::response::Response {
    let svc = match user_service(&state) {
        Ok(s) => s,
        Err(e) => return *e,
    };

    let info = match find_user(svc, &id).await {
        Some(i) => i,
        None => return scim_error_simple(404, "Resource not found"),
    };

    let current_etag = compute_etag(&info.sub, info.active);
    if let Err(e) = check_if_match(&headers, &current_etag) {
        return *e;
    }

    // Soft delete: suspend (set active=false)
    if let Err(e) = svc.suspend(&info.username).await {
        return scim_error_simple(500, &e.to_string());
    }

    StatusCode::NO_CONTENT.into_response()
}

// ─── Key management endpoints ─────────────────────────────────────────────────

/// GET /scim/v2/Users/:id/keys — List public keys for a user.
pub async fn list_user_keys(
    State(state): State<Arc<OAuthState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    let svc = match user_service(&state) {
        Ok(s) => s,
        Err(e) => return *e,
    };

    let info = match find_user(svc, &id).await {
        Some(i) => i,
        None => return scim_error_simple(404, "Resource not found"),
    };

    let resources: Vec<serde_json::Value> = info.pubkeys.iter().map(|pk| {
        serde_json::json!({
            "fingerprint": pk.fingerprint,
            "pubkeyBase64": pk.pubkey_base64,
            "label": pk.label,
            "createdAt": pk.created_at,
            "lastUsedAt": pk.last_used_at,
        })
    }).collect();
    let total = resources.len();

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/scim+json")],
        Json(serde_json::json!({
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": total,
            "Resources": resources,
        })),
    )
        .into_response()
}

/// POST /scim/v2/Users/:id/keys — Add a public key for a user.
pub async fn add_user_key(
    State(state): State<Arc<OAuthState>>,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    let svc = match user_service(&state) {
        Ok(s) => s,
        Err(e) => return *e,
    };

    let info = match find_user(svc, &id).await {
        Some(i) => i,
        None => return scim_error_simple(404, "Resource not found"),
    };

    let pubkey_b64 = match body.get("pubkeyBase64").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return scim_error(400, "invalidValue", "pubkeyBase64 is required"),
    };
    let label = body.get("label").and_then(|v| v.as_str()).map(str::to_owned);

    let pubkey = match decode_pubkey_base64(pubkey_b64) {
        Ok(k) => k,
        Err(e) => return scim_error(400, "invalidValue", &e.to_string()),
    };

    match svc.add_pubkey(&info.username, pubkey, label).await {
        Ok(entry) => (
            StatusCode::CREATED,
            [(header::CONTENT_TYPE, "application/scim+json")],
            Json(serde_json::json!({
                "fingerprint": entry.fingerprint,
                "pubkeyBase64": entry.pubkey_base64,
                "label": entry.label,
                "createdAt": entry.created_at,
            })),
        )
            .into_response(),
        Err(e) => scim_error_simple(500, &e.to_string()),
    }
}

/// DELETE /scim/v2/Users/:id/keys/:fingerprint — Remove a public key.
pub async fn remove_user_key(
    State(state): State<Arc<OAuthState>>,
    Path((id, fingerprint)): Path<(String, String)>,
) -> axum::response::Response {
    let svc = match user_service(&state) {
        Ok(s) => s,
        Err(e) => return *e,
    };

    let info = match find_user(svc, &id).await {
        Some(i) => i,
        None => return scim_error_simple(404, "Resource not found"),
    };

    // Normalize: ensure SHA256: prefix is present regardless of whether client included it
    let owned;
    let fp = if fingerprint.starts_with("SHA256:") {
        fingerprint.as_str()
    } else {
        owned = format!("SHA256:{fingerprint}");
        &owned
    };
    match svc.remove_pubkey(&info.username, fp).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => scim_error_simple(404, "Key not found"),
        Err(e) => scim_error_simple(500, &e.to_string()),
    }
}

// ─── Discovery endpoints ─────────────────────────────────────────────────────

/// GET /scim/v2/ServiceProviderConfig (RFC 7644 §4).
pub async fn service_provider_config() -> axum::response::Response {
    let config = ScimServiceProviderConfig {
        schemas: vec![
            "urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig".to_owned(),
        ],
        patch: ScimFeature { supported: false },
        bulk: ScimFeatureWithMax {
            supported: false,
            max_operations: 0,
            max_payload_size: 0,
        },
        filter: ScimFilterFeature {
            supported: true,
            max_results: 100,
        },
        change_password: ScimFeature { supported: false },
        sort: ScimFeature { supported: true },
        etag: ScimFeature { supported: true },
        authentication_schemes: vec![ScimAuthScheme {
            name: "OAuth Bearer Token".to_owned(),
            description: "Authentication via OAuth 2.0 Bearer Token".to_owned(),
            spec_uri: "https://www.rfc-editor.org/rfc/rfc6750".to_owned(),
            auth_type: "oauthbearertoken".to_owned(),
            primary: true,
        }],
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/scim+json")],
        Json(config),
    )
        .into_response()
}

/// GET /scim/v2/Schemas (RFC 7644 §4).
pub async fn schemas() -> axum::response::Response {
    let user_schema = ScimSchema {
        schemas: vec!["urn:ietf:params:scim:schemas:core:2.0:Schema".to_owned()],
        id: SCIM_SCHEMA_USER.to_owned(),
        name: "User".to_owned(),
        description: "User Account".to_owned(),
        attributes: vec![
            ScimAttribute {
                name: "userName".to_owned(),
                attr_type: "string".to_owned(),
                multi_valued: false,
                required: true,
                case_exact: false,
                mutability: "readWrite".to_owned(),
                returned: "default".to_owned(),
                uniqueness: "server".to_owned(),
                sub_attributes: None,
            },
            ScimAttribute {
                name: "displayName".to_owned(),
                attr_type: "string".to_owned(),
                multi_valued: false,
                required: false,
                case_exact: false,
                mutability: "readWrite".to_owned(),
                returned: "default".to_owned(),
                uniqueness: "none".to_owned(),
                sub_attributes: None,
            },
            ScimAttribute {
                name: "active".to_owned(),
                attr_type: "boolean".to_owned(),
                multi_valued: false,
                required: false,
                case_exact: false,
                mutability: "readWrite".to_owned(),
                returned: "default".to_owned(),
                uniqueness: "none".to_owned(),
                sub_attributes: None,
            },
            ScimAttribute {
                name: "emails".to_owned(),
                attr_type: "complex".to_owned(),
                multi_valued: true,
                required: false,
                case_exact: false,
                mutability: "readWrite".to_owned(),
                returned: "default".to_owned(),
                uniqueness: "none".to_owned(),
                sub_attributes: Some(vec![
                    ScimAttribute {
                        name: "value".to_owned(),
                        attr_type: "string".to_owned(),
                        multi_valued: false,
                        required: false,
                        case_exact: false,
                        mutability: "readWrite".to_owned(),
                        returned: "default".to_owned(),
                        uniqueness: "none".to_owned(),
                        sub_attributes: None,
                    },
                    ScimAttribute {
                        name: "primary".to_owned(),
                        attr_type: "boolean".to_owned(),
                        multi_valued: false,
                        required: false,
                        case_exact: false,
                        mutability: "readWrite".to_owned(),
                        returned: "default".to_owned(),
                        uniqueness: "none".to_owned(),
                        sub_attributes: None,
                    },
                ]),
            },
            ScimAttribute {
                name: "externalId".to_owned(),
                attr_type: "string".to_owned(),
                multi_valued: false,
                required: false,
                case_exact: false,
                mutability: "readWrite".to_owned(),
                returned: "default".to_owned(),
                uniqueness: "none".to_owned(),
                sub_attributes: None,
            },
        ],
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/scim+json")],
        Json(vec![user_schema]),
    )
        .into_response()
}

/// GET /scim/v2/ResourceTypes (RFC 7644 §4).
pub async fn resource_types() -> axum::response::Response {
    let user_type = ScimResourceType {
        schemas: vec!["urn:ietf:params:scim:schemas:core:2.0:ResourceType".to_owned()],
        id: "User".to_owned(),
        name: "User".to_owned(),
        endpoint: "/scim/v2/Users".to_owned(),
        description: "User Account".to_owned(),
        schema_: SCIM_SCHEMA_USER.to_owned(),
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/scim+json")],
        Json(vec![user_type]),
    )
        .into_response()
}
