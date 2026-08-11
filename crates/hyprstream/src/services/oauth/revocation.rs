//! OAuth 2.0 Token Revocation (RFC 7009).
//!
//! `POST /oauth/revoke` — revokes refresh tokens and access tokens.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Form;
use serde::Deserialize;


use super::state::OAuthState;

#[derive(Deserialize)]
pub struct RevocationRequest {
    /// The token to revoke.
    pub token: String,
    /// Optional hint: "refresh_token" or "access_token".
    #[serde(default)]
    pub token_type_hint: Option<String>,
}

/// POST /oauth/revoke (RFC 7009)
///
/// Revokes a refresh token or access token. Access tokens with a `jti` claim
/// are added to the blocklist (checked by `verify_claims` on every request).
/// Per RFC 7009 Section 2.1, the server MUST respond with 200 OK even if the
/// token is invalid or already revoked.
pub async fn revoke_token(
    State(state): State<Arc<OAuthState>>,
    Form(params): Form<RevocationRequest>,
) -> Response {
    let is_access_hint = params.token_type_hint.as_deref() == Some("access_token");

    if !is_access_hint {
        if let Err(e) = state.delete_refresh_token(&params.token).await {
            tracing::warn!(error = %e, "Refresh token store delete failed during revocation");
        } else {
            tracing::info!("Revoked refresh token");
        }
    }

    if is_access_hint || params.token_type_hint.is_none() {
        // Use the process-global credential-revocation store.
        if let Some(store) = hyprstream_rpc::auth::global_credential_revocation_store() {
            // Verify the token using the single verification stack
            // (typ, signature, local-issuer) with the relaxed audience
            // policy appropriate for revocation: any locally-issued
            // resource-audience token may be revoked. An unverified
            // token's iss/jti/exp are attacker-controlled; a forged
            // token must not revoke another credential. If verification
            // fails, return 200 per RFC 7009 but do not modify the store.
            match super::auth::verify_access_token(
                state.as_ref(),
                &params.token,
                super::auth::AudiencePolicy::AnyLocalIssuer,
            )
            .await
            {
                Ok(claims) => {
                    if let Some(ref jti) = claims.jti {
                        let cred_id = hyprstream_rpc::auth::CredentialId::jwt(&claims.iss, jti);
                        store.revoke_credential(cred_id, claims.exp);
                        tracing::info!(sub = %claims.sub, "Revoked access token via credential revocation store");
                    }
                }
                Err(_) => {
                    // Token verification failed — may be expired, wrong key,
                    // or not a JWT. RFC 7009: still 200.
                }
            }
        }
    }

    StatusCode::OK.into_response()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::OAuthConfig;
    use crate::services::{DiscoveryClient, PolicyClient};

    /// Build a minimal OAuthState with a known Ed25519 signing key.
    fn make_test_state(
        signing_key: &ed25519_dalek::SigningKey,
    ) -> (Arc<OAuthState>, String) {
        use hyprstream_rpc::crypto::CryptoPolicy;
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

        let remote_key = ed25519_dalek::SigningKey::from_bytes(&[0x99; 32]).verifying_key();
        let make_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(signing_key.clone()),
                    LazyUdsTransport::new("/dev/null/revocation-test.sock".into()),
                    Some(remote_key),
                )
                .with_response_verify_policy(CryptoPolicy::Classical),
            )
        };
        let mut config = OAuthConfig::default();
        config.external_url = Some("https://idp.test.example".to_owned());
        let issuer = config.issuer_url();
        let state = Arc::new(OAuthState::new(
            &config,
            PolicyClient::new(make_client()),
            DiscoveryClient::new(make_client()),
            signing_key.verifying_key().to_bytes(),
        ));
        (state, issuer)
    }

    /// Ensure the global credential-revocation store is published. If it's
    /// already set (from another test), use the existing one.
    fn ensure_global_store() {
        if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
            let store = Arc::new(
                hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
            );
            let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(store);
        }
    }

    /// A valid locally-signed RFC 9068 access token with a resource audience
    /// (not the OAuth issuer URL) is successfully revoked: its exact
    /// CredentialId appears in the global store after the handler returns.
    #[tokio::test]
    async fn resource_audience_token_is_revoked() {
        ensure_global_store();

        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x55; 32]);
        let (state, issuer) = make_test_state(&signing_key);
        let resource_aud = "http://localhost:6792"; // MCP/xet resource audience

        let now = chrono::Utc::now().timestamp();
        let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 3600)
            .with_issuer(issuer.clone())
            .with_audience(Some(resource_aud.to_owned()))
            .with_jti();
        let token = hyprstream_rpc::auth::jwt::encode(&claims, &signing_key);
        let actual_jti = claims.jti.as_deref().unwrap();

        let cred_id = hyprstream_rpc::auth::CredentialId::jwt(&issuer, actual_jti);
        let store = hyprstream_rpc::auth::global_credential_revocation_store().unwrap();
        assert!(
            !store.is_revoked(&cred_id),
            "precondition: credential not yet revoked"
        );

        // Call the actual revocation handler.
        let response = revoke_token(
            State(state),
            Form(RevocationRequest {
                token,
                token_type_hint: Some("access_token".to_owned()),
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK, "RFC 7009: always 200");
        assert!(
            store.is_revoked(&cred_id),
            "resource-audience token must be revoked in the store"
        );
    }

    /// A forged (wrong-key-signed) token returns 200 (RFC 7009) but does NOT
    /// publish to the store — attacker-controlled claims cannot revoke
    /// another credential.
    #[tokio::test]
    async fn forged_token_is_not_revoked() {
        ensure_global_store();

        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x66; 32]);
        let wrong_key = ed25519_dalek::SigningKey::from_bytes(&[0x77; 32]);
        let (state, issuer) = make_test_state(&signing_key);
        let resource_aud = "http://localhost:6792";

        let now = chrono::Utc::now().timestamp();
        let claims = hyprstream_rpc::auth::Claims::new("mallory".to_owned(), now, now + 3600)
            .with_issuer(issuer.clone())
            .with_audience(Some(resource_aud.to_owned()))
            .with_jti();
        // Sign with the WRONG key — the state expects signing_key's VK.
        let forged_token = hyprstream_rpc::auth::jwt::encode(&claims, &wrong_key);
        let actual_jti = claims.jti.as_deref().unwrap();

        let cred_id = hyprstream_rpc::auth::CredentialId::jwt(&issuer, actual_jti);
        let store = hyprstream_rpc::auth::global_credential_revocation_store().unwrap();
        assert!(!store.is_revoked(&cred_id), "precondition: not revoked");

        let response = revoke_token(
            State(state),
            Form(RevocationRequest {
                token: forged_token,
                token_type_hint: Some("access_token".to_owned()),
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK, "RFC 7009: always 200");
        assert!(
            !store.is_revoked(&cred_id),
            "forged token must NOT publish a revocation"
        );
    }
}
