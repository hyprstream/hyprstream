//! OAuth 2.1 / OpenID Connect discovery metadata.
//!
//! - `GET /.well-known/oauth-authorization-server` — RFC 8414
//! - `GET /.well-known/openid-configuration` — OIDC Discovery 1.0

use std::sync::Arc;

use axum::{extract::State, response::IntoResponse, Json};

use super::state::OAuthState;

/// The full set of scopes the AS advertises in `scopes_supported` (#1113 rev2 F4).
/// Extends the operator's default grant set with the atproto transition scopes,
/// which are supported-but-explicit (NOT silently granted on omitted scope).
fn advertised_scopes(default_scopes: &[String]) -> Vec<String> {
    let mut scopes = default_scopes.to_vec();
    for atproto_scope in &["atproto", "transition:generic"] {
        if !scopes.iter().any(|s| s == *atproto_scope) {
            scopes.push((*atproto_scope).to_owned());
        }
    }
    scopes
}

/// Base metadata fields shared between RFC 8414 and OIDC discovery.
fn base_metadata(issuer: &str, scopes: &[String], require_par: bool) -> serde_json::Value {
    let mut scopes_supported = advertised_scopes(scopes);
    if !scopes_supported.iter().any(|scope| scope == "pds:attach") {
        scopes_supported.push("pds:attach".to_owned());
    }
    // #1113 rev2 finding 5: canonicalize the issuer to an exact origin before
    // emitting it anywhere — a trailing slash or path in external_url would
    // produce double-slash endpoint URLs and a non-origin issuer the stock
    // resolver rejects against the discovery origin.
    let issuer = super::state::canonical_issuer_origin(issuer).unwrap_or_else(|| issuer.to_owned());
    serde_json::json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{}/oauth/authorize", issuer),
        "token_endpoint": format!("{}/oauth/token", issuer),
        "registration_endpoint": format!("{}/oauth/register", issuer),
        "device_authorization_endpoint": format!("{}/oauth/device", issuer),
        "pushed_authorization_request_endpoint": format!("{}/oauth/par", issuer),
        "require_pushed_authorization_requests": require_par,
        "jwks_uri": format!("{}/oauth/jwks", issuer),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token", "urn:ietf:params:oauth:grant-type:device_code"],
        "code_challenge_methods_supported": ["S256"],
        // #1113 rev2 finding 5: atproto profile metadata. The AS accepts both
        // public clients (PKCE) and `private_key_jwt` (RFC 7523), signs token
        // endpoint assertions with ES256, signals DPoP-ES256 support, and
        // emits `authorization_endpoint` `iss` query parameter on redirects.
        "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"],
        "token_endpoint_auth_signing_alg_values_supported": ["ES256"],
        "dpop_signing_alg_values_supported": ["ES256"],
        "authorization_response_iss_parameter_supported": true,
        "scopes_supported": scopes_supported,
        "client_id_metadata_document_supported": true,
    })
}

/// GET /.well-known/oauth-authorization-server (RFC 8414)
pub async fn authorization_server_metadata(
    State(state): State<Arc<OAuthState>>,
) -> impl IntoResponse {
    Json(base_metadata(
        &state.issuer_url,
        &state.default_scopes,
        state.require_pushed_authorization_requests,
    ))
}

/// GET /.well-known/openid-configuration (OIDC Discovery 1.0)
///
/// Superset of RFC 8414 metadata with OIDC-specific fields:
/// userinfo_endpoint, id_token_signing_alg_values_supported,
/// subject_types_supported, claims_supported.
pub async fn openid_configuration(
    State(state): State<Arc<OAuthState>>,
) -> impl IntoResponse {
    let issuer = &state.issuer_url;
    let mut meta = base_metadata(issuer, &state.default_scopes, state.require_pushed_authorization_requests);

    // OIDC-specific fields
    let obj = meta.as_object_mut().unwrap_or_else(|| unreachable!());
    obj.insert("userinfo_endpoint".into(),
        serde_json::Value::String(format!("{}/oauth/userinfo", issuer)));
    obj.insert("id_token_signing_alg_values_supported".into(),
        serde_json::json!(["EdDSA", "RS256"]));
    obj.insert("subject_types_supported".into(),
        serde_json::json!(["public"]));
    obj.insert("claims_supported".into(),
        serde_json::json!([
            "sub", "iss", "aud", "exp", "iat", "nonce", "auth_time",
            "name", "email", "email_verified", "preferred_username"
        ]));

    Json(meta)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// #1113: the atproto OAuth AS conformance scopes (`atproto`,
    /// `transition:generic`) are advertised in `scopes_supported` so stock
    /// atproto clients (e.g. `@atproto/oauth-client-browser`) discover them.
    #[test]
    fn authorization_server_metadata_advertises_atproto_scopes() {
        let scopes = vec![
            "read:*:*".to_owned(),
            "write:*:*".to_owned(),
            "atproto".to_owned(),
            "transition:generic".to_owned(),
        ];
        let meta = base_metadata("https://pds.example.com", &scopes, true);
        let advertised: Vec<&str> = meta["scopes_supported"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(advertised.contains(&"atproto"), "scopes_supported missing atproto: {advertised:?}");
        assert!(
            advertised.contains(&"transition:generic"),
            "scopes_supported missing transition:generic: {advertised:?}"
        );
    }

    /// #1113: PAR is mandatory for the atproto AS profile, and the AS metadata
    /// must reflect `require_pushed_authorization_requests: true`.
    #[test]
    fn authorization_server_metadata_requires_par_when_configured() {
        let meta = base_metadata("https://pds.example.com", &[], true);
        assert_eq!(meta["require_pushed_authorization_requests"].as_bool(), Some(true));
        assert_eq!(
            meta["pushed_authorization_request_endpoint"].as_str(),
            Some("https://pds.example.com/oauth/par")
        );
    }

    /// #1113 rev2 finding 5: the RFC 8414 document carries the full atproto
    /// profile metadata surface — private_key_jwt client auth, ES256 token
    /// signing alg, DPoP ES256, iss-parameter support, and a canonical
    /// origin issuer (no trailing slash/path).
    #[test]
    fn authorization_server_metadata_has_atproto_profile_fields() {
        let meta = base_metadata("https://pds.example.com/", &["atproto".to_owned()], true);

        // Issuer canonicalized to exact origin (trailing slash stripped).
        assert_eq!(meta["issuer"].as_str(), Some("https://pds.example.com"));
        assert_eq!(
            meta["token_endpoint"].as_str(),
            Some("https://pds.example.com/oauth/token"),
            "no double-slash from a trailing-slash issuer"
        );

        let auth_methods: Vec<&str> = meta["token_endpoint_auth_methods_supported"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(auth_methods.contains(&"none"), "public clients (PKCE) must be accepted");
        assert!(
            auth_methods.contains(&"private_key_jwt"),
            "private_key_jwt must be advertised"
        );

        let token_algs: Vec<&str> = meta["token_endpoint_auth_signing_alg_values_supported"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(token_algs.contains(&"ES256"));

        let dpop_algs: Vec<&str> = meta["dpop_signing_alg_values_supported"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(dpop_algs.contains(&"ES256"));

        assert_eq!(
            meta["authorization_response_iss_parameter_supported"].as_bool(),
            Some(true)
        );
    }
}
