//! ZMQ RPC handler for user CRUD operations.
//!
//! Implements the `OauthHandler` trait (generated from `oauth.capnp`)
//! and `ZmqService` for ZMQ transport. Delegates to `UserService` for
//! shared CRUD logic.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::service::{Continuation, EnvelopeContext, ZmqService};
use hyprstream_rpc::transport::TransportConfig;

use crate::auth::UserFilter;
use crate::services::generated::oauth_client::{
    dispatch_oauth, serialize_response, ErrorInfo, ListUsers, OauthHandler,
    OauthResponseVariant, RegisterUser, UpdateUser, UserInfo as RpcUserInfo,
    UserListResult,
};

use super::user_service::{self, UserUpdate};
use super::state::OAuthState;

/// ZMQ RPC handler for OAuth user management.
///
/// Wraps `UserService` and implements the generated `OauthHandler` trait
/// for Cap'n Proto serialization, and `ZmqService` for ZMQ transport.
pub struct OAuthZmqHandler {
    state: Arc<OAuthState>,
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,
}

impl OAuthZmqHandler {
    pub fn new(
        state: Arc<OAuthState>,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
    ) -> Self {
        Self {
            state,
            context,
            transport,
            signing_key,
        }
    }

    fn user_service(&self) -> Result<&Arc<user_service::UserService>> {
        self.state
            .user_service
            .as_ref()
            .ok_or_else(|| anyhow!("User service not configured"))
    }

    fn user_info_to_rpc(info: &user_service::UserInfo) -> RpcUserInfo {
        RpcUserInfo {
            username: info.username.clone(),
            sub: info.sub.clone(),
            pubkey_base64: info.pubkey_base64.clone(),
            name: info.name.clone().unwrap_or_default(),
            email: info.email.clone().unwrap_or_default(),
            email_verified: info.email_verified,
            active: info.active,
            external_id: info.external_id.clone().unwrap_or_default(),
        }
    }
}

#[async_trait(?Send)]
impl OauthHandler for OAuthZmqHandler {
    async fn authorize(
        &self,
        _ctx: &EnvelopeContext,
        _resource: &str,
        _operation: &str,
    ) -> Result<()> {
        // ZMQ RPC is internal — service-to-service. All requests are authorized.
        Ok(())
    }

    async fn handle_register_user(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterUser,
    ) -> Result<OauthResponseVariant> {
        let svc = self.user_service()?;
        let info = svc.register(&data.username, &data.pubkey_base64).await?;
        Ok(OauthResponseVariant::RegisterUserResult(Self::user_info_to_rpc(&info)))
    }

    async fn handle_get_user(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        username: &str,
    ) -> Result<OauthResponseVariant> {
        let svc = self.user_service()?;
        match svc.get(username).await? {
            Some(info) => Ok(OauthResponseVariant::GetUserResult(Self::user_info_to_rpc(&info))),
            None => Ok(OauthResponseVariant::Error(ErrorInfo {
                message: format!("User '{}' not found", username),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_list_users(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ListUsers,
    ) -> Result<OauthResponseVariant> {
        let svc = self.user_service()?;
        let filter = UserFilter {
            filter: if data.filter.is_empty() {
                None
            } else {
                Some(data.filter.clone())
            },
            active_only: if data.active_only {
                Some(true)
            } else {
                None
            },
            count: if data.count > 0 {
                Some(data.count as usize)
            } else {
                None
            },
            start_index: if data.start_index > 0 {
                Some(data.start_index as usize)
            } else {
                None
            },
            sort_by: None,
            sort_order: None,
        };
        let list = svc.list(&filter).await?;
        Ok(OauthResponseVariant::ListUsersResult(UserListResult {
            users: list.users.iter().map(Self::user_info_to_rpc).collect(),
            total_results: list.total_results as u32,
        }))
    }

    async fn handle_update_user(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &UpdateUser,
    ) -> Result<OauthResponseVariant> {
        let svc = self.user_service()?;
        let update = UserUpdate {
            name: data.name.as_ref().filter(|s| !s.is_empty()).map(|s| Some(s.clone())),
            email: data.email.as_ref().filter(|s| !s.is_empty()).map(|s| Some(s.clone())),
            external_id: data.external_id.as_ref().filter(|s| !s.is_empty()).map(|s| Some(s.clone())),
            email_verified: None,
        };
        match svc.update(&data.username, update).await {
            Ok(info) => Ok(OauthResponseVariant::UpdateUserResult(Self::user_info_to_rpc(&info))),
            Err(e) => Ok(OauthResponseVariant::Error(ErrorInfo {
                message: e.to_string(),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_suspend_user(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        username: &str,
    ) -> Result<OauthResponseVariant> {
        let svc = self.user_service()?;
        svc.suspend(username).await?;
        Ok(OauthResponseVariant::SuspendUserResult)
    }

    async fn handle_resume_user(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        username: &str,
    ) -> Result<OauthResponseVariant> {
        let svc = self.user_service()?;
        svc.resume(username).await?;
        Ok(OauthResponseVariant::ResumeUserResult)
    }

    async fn handle_remove_user(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        username: &str,
    ) -> Result<OauthResponseVariant> {
        let svc = self.user_service()?;
        svc.remove(username).await?;
        Ok(OauthResponseVariant::RemoveUserResult)
    }
}

#[async_trait(?Send)]
impl ZmqService for OAuthZmqHandler {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        dispatch_oauth(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        "oauth"
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = OauthResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}
