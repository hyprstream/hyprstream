use std::sync::Arc;

use axum::extract::{Extension, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use hyprstream_pds_service::federation_intake::IdentityInventoryReadModel;
use hyprstream_rpc::auth::mac::{CompartmentSet, Level, SecurityContext, VerifiedKeyMaterial};
use serde::Deserialize;

const MAX_FILTER_BYTES: usize = 256;
const MAX_CURSOR_BYTES: usize = 2048;
const DEFAULT_PAGE_SIZE: usize = 200;
const MAX_PAGE_SIZE: usize = 200;

#[derive(Clone)]
struct InventoryHttpState {
    inventory: Arc<dyn IdentityInventoryReadModel>,
}

/// Viewer clearance installed by trusted authentication/MAC middleware.
///
/// HTTP input cannot construct request extensions. If the extension is absent,
/// the handler deliberately applies the unauthenticated public floor.
#[derive(Clone, Debug)]
pub struct InventoryViewer(SecurityContext);

impl InventoryViewer {
    /// Wrap a clearance already derived from verified server-side claims.
    pub fn from_verified_clearance(clearance: SecurityContext) -> Self {
        Self(clearance)
    }

    fn unauthenticated_floor() -> Self {
        Self(SecurityContext::new(
            Level::Public,
            CompartmentSet::EMPTY,
            VerifiedKeyMaterial::Unverified,
        ))
    }
}

#[derive(Debug, Default, Deserialize)]
struct InventoryQuery {
    filter: Option<String>,
    after: Option<String>,
    limit: Option<usize>,
}

/// Build a mountable `GET /inventory?filter=...` AppView router.
///
/// The response is exactly an `InventoryEntry[]`; it has no envelope in which a
/// pre-filter total or hidden count could leak. Deployment middleware must
/// reject an invalid presented credential with 401; only true credential
/// absence may reach this router without an [`InventoryViewer`] extension.
pub fn inventory_router(inventory: Arc<dyn IdentityInventoryReadModel>) -> Router {
    Router::new()
        .route("/inventory", get(query_inventory))
        .with_state(InventoryHttpState { inventory })
}

async fn query_inventory(
    State(state): State<InventoryHttpState>,
    Query(query): Query<InventoryQuery>,
    viewer: Option<Extension<InventoryViewer>>,
) -> Response {
    if query
        .filter
        .as_deref()
        .is_some_and(|filter| filter.len() > MAX_FILTER_BYTES)
        || query
            .after
            .as_deref()
            .is_some_and(|after| after.len() > MAX_CURSOR_BYTES)
        || query.limit == Some(0)
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "invalid inventory query"})),
        )
            .into_response();
    }
    let limit = query.limit.unwrap_or(DEFAULT_PAGE_SIZE).min(MAX_PAGE_SIZE);
    let viewer = viewer
        .map(|Extension(viewer)| viewer)
        .unwrap_or_else(InventoryViewer::unauthenticated_floor);
    match state
        .inventory
        .query_page(
            &viewer.0,
            query.filter.as_deref(),
            query.after.as_deref(),
            limit,
        )
        .await
    {
        Ok(entries) => Json(entries).into_response(),
        Err(error) => {
            tracing::error!(error = %error, "inventory query failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "inventory unavailable"})),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use hyprstream_pds_service::federation_intake::{InMemoryIdentityInventory, InventoryEntry};
    use hyprstream_rpc::auth::mac::{Assurance, SecurityLabel};
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn get_inventory_returns_only_post_clearance_array() {
        let inventory = Arc::new(InMemoryIdentityInventory::default());
        inventory
            .upsert_derived(
                InventoryEntry::indexed_federated(
                    "did:web:public.example",
                    Some("alice.example".to_owned()),
                    None,
                )
                .unwrap(),
                SecurityLabel::bottom(),
            )
            .await
            .unwrap();
        inventory
            .upsert_derived(
                InventoryEntry::indexed_federated(
                    "did:web:hidden.example",
                    Some("hidden.example".to_owned()),
                    None,
                )
                .unwrap(),
                SecurityLabel::new(
                    Level::Secret,
                    Assurance::PqHybrid,
                    CompartmentSet::single(9),
                ),
            )
            .await
            .unwrap();
        let app = inventory_router(inventory);

        let response = app
            .clone()
            .oneshot(
                Request::get("/inventory?filter=example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        let entries = body.as_array().expect("response must be a bare array");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["did"], "did:web:public.example");
        assert!(body.get("total").is_none());
        assert!(body.get("hiddenCount").is_none());

        let response = app
            .clone()
            .oneshot(
                Request::get(format!(
                    "/inventory?filter={}",
                    "x".repeat(MAX_FILTER_BYTES + 1)
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let privileged = InventoryViewer::from_verified_clearance(SecurityContext::new(
            Level::Secret,
            CompartmentSet::single(9),
            VerifiedKeyMaterial::PqHybrid,
        ));
        let mut request = Request::get("/inventory?limit=1")
            .body(Body::empty())
            .unwrap();
        request.extensions_mut().insert(privileged.clone());
        let response = app.clone().oneshot(request).await.unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        let page = body.as_array().unwrap();
        assert_eq!(page.len(), 1);
        let after = page[0]["did"].as_str().unwrap();

        let mut request = Request::get(format!("/inventory?limit=1&after={after}"))
            .body(Body::empty())
            .unwrap();
        request.extensions_mut().insert(privileged);
        let response = app.oneshot(request).await.unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        assert_eq!(body.as_array().unwrap().len(), 1);
    }
}
