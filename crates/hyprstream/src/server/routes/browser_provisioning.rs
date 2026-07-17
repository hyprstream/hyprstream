//! Browser/WebTransport accepted-current provisioning boundary.

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use hyprstream_rpc::browser_provisioning::{BrowserCarrierProfile, BrowserProvisioningRequest};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserProvisioningQuery {
    capability: String,
    scope: String,
    carrier_profile: BrowserCarrierProfile,
}

/// Return a no-store public-key projection minted only by Discovery's
/// checkpoint-backed production resolver. The route is public because it
/// carries no credential or private material; authenticity comes from the
/// accepted-current resolver and the browser's authenticated HTTPS fetch.
pub async fn browser_provisioning(
    State(state): State<crate::server::state::ServerState>,
    Path(service): Path<String>,
    Query(query): Query<BrowserProvisioningQuery>,
) -> Response {
    let request = match BrowserProvisioningRequest::new(
        service,
        query.capability,
        query.scope,
        query.carrier_profile,
    ) {
        Ok(request) => request,
        Err(error) => {
            return bounded_json_response(
                StatusCode::BAD_REQUEST,
                &serde_json::json!({ "error": error.to_string() }),
            );
        }
    };
    let document = match hyprstream_discovery::production_browser_provisioning(request).await {
        Ok(document) => match document.sign_projection(&state.signing_key) {
            Ok(document) => document,
            Err(error) => {
                tracing::error!(error = %error, "browser projection signing failed closed");
                return bounded_json_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    &serde_json::json!({
                        "error": "accepted-current browser provisioning unavailable"
                    }),
                );
            }
        },
        Err(error) => {
            tracing::warn!(error = %error, "browser provisioning failed closed");
            return bounded_json_response(
                StatusCode::SERVICE_UNAVAILABLE,
                &serde_json::json!({
                    "error": "accepted-current browser provisioning unavailable"
                }),
            );
        }
    };
    projection_response(&document)
}

fn projection_response(
    document: &hyprstream_rpc::browser_provisioning::BrowserProvisioningDocument,
) -> Response {
    bounded_json_response(StatusCode::OK, document)
}

fn bounded_json_response<T: Serialize + ?Sized>(status: StatusCode, document: &T) -> Response {
    let body = match serde_json::to_vec(document) {
        Ok(body) => body,
        Err(error) => {
            tracing::error!(error = %error, "browser projection serialization failed closed");
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    };
    if body.len() > hyprstream_rpc::browser_provisioning::MAX_PROVISIONING_BYTES {
        tracing::error!(
            length = body.len(),
            "browser projection exceeded the serialization bound"
        );
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let Ok(content_length) = HeaderValue::from_str(&body.len().to_string()) else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let mut response = Response::new(axum::body::Body::from(body));
    *response.status_mut() = status;
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    headers.insert(header::CONTENT_LENGTH, content_length);
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    headers.insert(header::PRAGMA, HeaderValue::from_static("no-cache"));
    response
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_rpc::browser_provisioning::{
        BrowserProvisioningDocument, BrowserProvisioningMaterial, BrowserRouteRole,
        BrowserTransportSecurity,
    };

    #[test]
    fn invalid_query_context_is_rejected_before_resolver_use() {
        let result = BrowserProvisioningRequest::new(
            "Model",
            "hyprstream-rpc/1",
            "model",
            BrowserCarrierProfile::OwnedHybridWebTransport,
        );
        assert!(result.is_err());
    }

    #[test]
    fn oversized_serialized_projection_fails_closed() {
        let response = bounded_json_response(
            StatusCode::OK,
            &"x".repeat(hyprstream_rpc::browser_provisioning::MAX_PROVISIONING_BYTES + 1),
        );
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn deployed_projection_response_exposes_bounded_no_store_json_headers() {
        let signing = hyprstream_rpc::SigningKey::from_bytes(&[0x31; 32]);
        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&signing);
        let pq_vk = ml_dsa::Keypair::verifying_key(&pq);
        let kem = hyprstream_rpc::node_identity::derive_mesh_kem_recipient(&signing)
            .expect("derive KEM")
            .public();
        let document = BrowserProvisioningDocument::from_material(BrowserProvisioningMaterial {
            service_name: "model".to_owned(),
            service_did: "did:at9p:test".to_owned(),
            service_origin: "https://model.example/".to_owned(),
            webtransport_url: "https://model.example/".to_owned(),
            capability: "hyprstream-rpc/1".to_owned(),
            scope: "model".to_owned(),
            carrier_profile: BrowserCarrierProfile::OwnedHybridWebTransport,
            route_role: BrowserRouteRole::Origin,
            transport_security: BrowserTransportSecurity::OwnedHybridRequired,
            response_key_id: "did:at9p:test#response".to_owned(),
            response_ed25519: signing.verifying_key().to_bytes(),
            response_ml_dsa65: hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&pq_vk),
            request_kem_key_id: "did:at9p:test#mesh-kem".to_owned(),
            request_kem_recipient: kem,
            accepted_state_digest: [0x44; 64],
            accepted_state_epoch: 1,
            expires_at_unix_ms: 4_070_908_800_000,
            certificate_hashes: vec![[0x55; 32]],
            encrypted_objects_required: false,
        })
        .sign_projection(&signing)
        .expect("sign projection");
        let response = projection_response(&document);
        let headers = response.headers();
        assert_eq!(headers[header::CONTENT_TYPE], "application/json");
        assert!(headers[header::CACHE_CONTROL]
            .to_str()
            .expect("cache header")
            .contains("no-store"));
        let length = headers[header::CONTENT_LENGTH]
            .to_str()
            .expect("length header")
            .parse::<usize>()
            .expect("numeric length");
        assert!(length > 0);
        assert!(length <= hyprstream_rpc::browser_provisioning::MAX_PROVISIONING_BYTES);
    }
}
