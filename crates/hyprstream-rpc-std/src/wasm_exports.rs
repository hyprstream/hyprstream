//! WASM-bindgen exports for browser RPC clients.
//!
//! Provides `RpcSession` — a single entry point for all service RPCs from the browser.
//! Handles WebTransport connection, Ed25519 envelope signing, Cap'n Proto serialization,
//! and response parsing. Returns `JsValue` (serde-serialized) to React hooks.
//!
//! # Usage from JavaScript
//!
//! ```js
//! const session = await RpcSession.connect(url, certHash, keySeed);
//! session.set_jwt(token);
//! const repos = await session.registry_list();
//! ```

#![cfg(target_arch = "wasm32")]

use std::cell::{Cell, RefCell};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsError;

use hyprstream_rpc::web_transport::WtClient;

// Re-export low-level WASM API (signing, crypto, ZMTP framing) from hyprstream-rpc
pub use hyprstream_rpc::wasm_api::*;

/// Browser RPC session — wraps WebTransport + signing state.
///
/// One session per server connection. All service methods are available
/// as async methods that return serde-serialized JsValues.
#[wasm_bindgen]
pub struct RpcSession {
    client: WtClient,
    seed: [u8; 32],
    claims: RefCell<Option<hyprstream_rpc::auth::Claims>>,
    next_id: Cell<u64>,
}

#[wasm_bindgen]
impl RpcSession {
    /// Connect to a hyprstream WebTransport endpoint.
    ///
    /// - `url`: WebTransport URL (e.g., `https://host:port/wt`)
    /// - `cert_hash`: Optional base64-encoded SHA-256 certificate hash for pinning
    /// - `key_seed`: 32-byte Ed25519 signing key seed
    #[wasm_bindgen(constructor)]
    pub async fn connect(
        url: &str,
        cert_hash: Option<String>,
        key_seed: &[u8],
    ) -> Result<RpcSession, JsError> {
        if key_seed.len() != 32 {
            return Err(JsError::new("key_seed must be 32 bytes"));
        }
        let mut seed = [0u8; 32];
        seed.copy_from_slice(key_seed);

        let client = WtClient::connect(url, cert_hash.as_deref())
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(RpcSession {
            client,
            seed,
            claims: RefCell::new(None),
            next_id: Cell::new(1),
        })
    }

    /// Set JWT token for authenticated requests.
    /// Takes &self (not &mut self) so it can be called while async ops are in flight.
    pub fn set_jwt(&self, token: &str) -> Result<(), JsError> {
        if token.is_empty() {
            *self.claims.borrow_mut() = None;
            return Ok(());
        }
        let claims = parse_jwt_claims(token)?;
        *self.claims.borrow_mut() = claims;
        Ok(())
    }

    /// Get the next request ID.
    fn next_id(&self) -> u64 {
        let id = self.next_id.get();
        self.next_id.set(id.wrapping_add(1));
        id
    }

    /// Close the WebTransport connection.
    pub fn close(&self) {
        self.client.close();
    }

    // ========================================================================
    // Generic call — sign + send + return raw response bytes
    // ========================================================================

    /// Send a raw signed RPC request. Returns response payload bytes.
    pub async fn call(&self, payload: &[u8]) -> Result<Vec<u8>, JsError> {
        let request_id = self.next_id();
        let signed = self.sign(payload, request_id, &[])?;
        self.client
            .request(&signed)
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Send a raw signed streaming RPC request with ephemeral DH pubkey.
    pub async fn call_streaming(
        &self,
        payload: &[u8],
        ephemeral_pubkey: &[u8],
    ) -> Result<Vec<u8>, JsError> {
        let request_id = self.next_id();
        let signed = self.sign(payload, request_id, ephemeral_pubkey)?;
        self.client
            .request(&signed)
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Open a SUB stream. Returns a SubStream that yields blocks via next_block().
    pub async fn subscribe_stream(&self, topic: &[u8]) -> Result<hyprstream_rpc::web_transport::SubStream, JsError> {
        self.client
            .subscribe(topic)
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    // ========================================================================
    // Registry service
    // ========================================================================

    /// List all tracked repositories.
    pub async fn registry_list(&self) -> Result<JsValue, JsError> {
        let payload = build_registry_request(self.next_id(), |mut req| {
            req.set_list(());
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::registry_client::RegistryResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get a repository by ID.
    pub async fn registry_get(&self, id: &str) -> Result<JsValue, JsError> {
        let payload = build_registry_request(self.next_id(), |mut req| {
            req.set_get(id);
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::registry_client::RegistryResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get a repository by name.
    pub async fn registry_get_by_name(&self, name: &str) -> Result<JsValue, JsError> {
        let payload = build_registry_request(self.next_id(), |mut req| {
            req.set_get_by_name(name);
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::registry_client::RegistryResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Clone a repository.
    pub async fn registry_clone(&self, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::CloneRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_registry_request(self.next_id(), |mut builder| {
            let mut inner = builder.init_clone();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::registry_client::RegistryResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Remove a repository.
    pub async fn registry_remove(&self, id: &str) -> Result<JsValue, JsError> {
        let payload = build_registry_request(self.next_id(), |mut req| {
            req.set_remove(id);
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::registry_client::RegistryResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Health check.
    pub async fn registry_health_check(&self) -> Result<JsValue, JsError> {
        let payload = build_registry_request(self.next_id(), |mut req| {
            req.set_health_check(());
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::registry_client::RegistryResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    // ========================================================================
    // Registry — repo-scoped operations
    // ========================================================================

    /// List branches for a repository.
    pub async fn registry_repo_list_branches(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut req| {
            req.set_list_branches(());
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// List worktrees for a repository.
    pub async fn registry_repo_list_worktrees(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut req| {
            req.set_list_worktrees(());
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Get repository status.
    pub async fn registry_repo_status(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut req| {
            req.set_status(());
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Get detailed repository status.
    pub async fn registry_repo_detailed_status(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut req| {
            req.set_detailed_status(());
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Get HEAD ref.
    pub async fn registry_repo_get_head(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut req| {
            req.set_get_head(());
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// List remotes.
    pub async fn registry_repo_list_remotes(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut req| {
            req.set_list_remotes(());
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Stage all modified files.
    pub async fn registry_repo_stage_all(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut req| {
            req.set_stage_all(());
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Commit staged changes.
    pub async fn registry_repo_commit(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::CommitWithAuthorRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_commit_with_author();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Checkout a branch.
    pub async fn registry_repo_checkout(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::CheckoutRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_checkout();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Create a branch.
    pub async fn registry_repo_create_branch(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::BranchRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_create_branch();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Merge a branch.
    pub async fn registry_repo_merge(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::MergeRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_merge();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Abort merge.
    pub async fn registry_repo_abort_merge(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut req| {
            req.set_abort_merge(());
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Continue merge.
    pub async fn registry_repo_continue_merge(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |req| {
            req.init_continue_merge();
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Quit merge.
    pub async fn registry_repo_quit_merge(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut req| {
            req.set_quit_merge(());
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Push to remote.
    pub async fn registry_repo_push(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::PushRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_push();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Add a remote.
    pub async fn registry_repo_add_remote(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::AddRemoteRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_add_remote();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Remove a remote.
    pub async fn registry_repo_remove_remote(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::RemoveRemoteRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_remove_remote();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Rename a remote.
    pub async fn registry_repo_rename_remote(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::RenameRemoteRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_rename_remote();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Set remote URL.
    pub async fn registry_repo_set_remote_url(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::SetRemoteUrlRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_set_remote_url();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Create a worktree.
    pub async fn registry_repo_create_worktree(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::CreateWorktreeRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |mut builder| {
            let mut inner = builder.init_create_worktree();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Remove a worktree.
    pub async fn registry_repo_remove_worktree(&self, repo_id: &str, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::registry_client::RemoveWorktreeRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |builder| {
            let mut inner = builder.init_remove_worktree();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    /// Unload (remove tracking via update with empty params).
    pub async fn registry_repo_unload(&self, repo_id: &str) -> Result<JsValue, JsError> {
        let payload = build_repo_scoped_request(self.next_id(), repo_id, |req| {
            req.init_update();
        })?;
        let response = self.send(&payload).await?;
        parse_repo_response(&response)
    }

    // ========================================================================
    // Model service
    // ========================================================================

    /// Load a model.
    pub async fn model_load(&self, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::model_client::LoadModelRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_model_request(self.next_id(), |mut builder| {
            let mut inner = builder.init_load();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::model_client::ModelResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Unload a model.
    pub async fn model_unload(&self, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::model_client::UnloadModelRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_model_request(self.next_id(), |mut builder| {
            let mut inner = builder.init_unload();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::model_client::ModelResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get model status.
    pub async fn model_status(&self, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::model_client::StatusRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_model_request(self.next_id(), |mut builder| {
            let mut inner = builder.init_status();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::model_client::ModelResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Model health check.
    pub async fn model_health_check(&self) -> Result<JsValue, JsError> {
        let payload = build_model_request(self.next_id(), |mut builder| {
            builder.set_health_check(());
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::model_client::ModelResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    // ========================================================================
    // Model — infer-scoped operations
    // ========================================================================

    /// Apply chat template (model.infer scope).
    pub async fn model_infer_apply_chat_template(
        &self,
        model_ref: &str,
        request: JsValue,
    ) -> Result<JsValue, JsError> {
        let req: crate::inference_client::ChatTemplateRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_model_infer_request(self.next_id(), model_ref, |mut builder| {
            let mut inner = builder.init_apply_chat_template();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        // Use scoped response parser — handles nested ModelResponse → InferResponse union
        let parsed = crate::model_client::InferClientResponseVariant::parse_scoped_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Start streaming inference (model.infer scope). Returns StreamInfo.
    pub async fn model_infer_generate_stream(
        &self,
        model_ref: &str,
        request: JsValue,
        ephemeral_pubkey: &[u8],
    ) -> Result<JsValue, JsError> {
        let req: crate::inference_client::GenerationRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let request_id = self.next_id();
        let payload = build_model_infer_request(request_id, model_ref, |mut builder| {
            let mut inner = builder.init_generate_stream();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let signed = self.sign(&payload, request_id, ephemeral_pubkey)?;
        let response_bytes = self.client
            .request(&signed)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let inner = unwrap_response_envelope(&response_bytes)?;
        // Use scoped response parser — handles nested ModelResponse → InferResponse union
        let parsed = crate::model_client::InferClientResponseVariant::parse_scoped_response(&inner)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    // ========================================================================
    // Policy service
    // ========================================================================

    /// Check a policy.
    pub async fn policy_check(&self, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::policy_client::PolicyCheck =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_policy_request(self.next_id(), |mut builder| {
            let mut inner = builder.init_check();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::policy_client::PolicyResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// List policy scopes.
    pub async fn policy_list_scopes(&self) -> Result<JsValue, JsError> {
        let payload = build_policy_request(self.next_id(), |mut builder| {
            builder.set_list_scopes(());
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::policy_client::PolicyResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get current policy.
    pub async fn policy_get_policy(&self) -> Result<JsValue, JsError> {
        let payload = build_policy_request(self.next_id(), |mut builder| {
            builder.set_get_policy(());
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::policy_client::PolicyResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    // ========================================================================
    // Inference service
    // ========================================================================

    /// Start streaming inference (returns StreamInfo for subscription).
    pub async fn inference_generate_stream(
        &self,
        request: JsValue,
        ephemeral_pubkey: &[u8],
    ) -> Result<JsValue, JsError> {
        let req: crate::inference_client::GenerationRequest =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let request_id = self.next_id();
        let payload = hyprstream_rpc::serialize_message(|msg| {
            let mut builder = msg.init_root::<crate::inference_capnp::inference_request::Builder>();
            builder.set_id(request_id);
            let mut inner = builder.init_generate_stream();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })
        .map_err(|e| JsError::new(&e.to_string()))?;

        let signed = self.sign(&payload, request_id, ephemeral_pubkey)?;
        let response = self.client
            .request(&signed)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let parsed = crate::inference_client::InferenceResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    // ========================================================================
    // MCP service
    // ========================================================================

    /// List MCP tools.
    pub async fn mcp_list_tools(&self) -> Result<JsValue, JsError> {
        let payload = build_mcp_request(self.next_id(), |mut builder| {
            builder.set_list_tools(());
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::mcp_client::McpResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Call an MCP tool.
    pub async fn mcp_call_tool(&self, request: JsValue) -> Result<JsValue, JsError> {
        let req: crate::mcp_client::CallTool =
            serde_wasm_bindgen::from_value(request).map_err(|e| JsError::new(&e.to_string()))?;
        let payload = build_mcp_request(self.next_id(), |mut builder| {
            let mut inner = builder.init_call_tool();
            hyprstream_rpc::ToCapnp::write_to(&req, &mut inner);
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::mcp_client::McpResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get MCP service status.
    pub async fn mcp_get_status(&self) -> Result<JsValue, JsError> {
        let payload = build_mcp_request(self.next_id(), |mut builder| {
            builder.set_get_status(());
        })?;
        let response = self.send(&payload).await?;
        let parsed = crate::mcp_client::McpResponseVariant::parse_response(&response)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
    }
}

// ============================================================================
// Private helpers
// ============================================================================

impl RpcSession {
    /// Sign a payload into a SignedEnvelope.
    fn sign(
        &self,
        payload: &[u8],
        request_id: u64,
        ephemeral_pubkey: &[u8],
    ) -> Result<Vec<u8>, JsError> {
        use hyprstream_rpc::crypto::signing::signing_key_from_bytes;
        use hyprstream_rpc::envelope::{RequestEnvelope, RequestIdentity, SignedEnvelope};
        use hyprstream_rpc::ToCapnp;

        let signing_key = signing_key_from_bytes(&self.seed);

        let ephemeral = if ephemeral_pubkey.is_empty() {
            None
        } else if ephemeral_pubkey.len() == 32 {
            let mut buf = [0u8; 32];
            buf.copy_from_slice(ephemeral_pubkey);
            Some(buf)
        } else {
            return Err(JsError::new("ephemeral_pubkey must be empty or 32 bytes"));
        };

        let claims_ref = self.claims.borrow();
        let identity = match claims_ref.as_ref() {
            Some(c) => RequestIdentity::api_token(&c.sub, "jwt"),
            None => RequestIdentity::Anonymous,
        };

        let mut envelope = RequestEnvelope {
            request_id,
            identity,
            timestamp: hyprstream_rpc::envelope::current_timestamp(),
            nonce: hyprstream_rpc::envelope::generate_nonce(),
            ephemeral_pubkey: ephemeral,
            payload: payload.to_vec(),
            claims: None,
        };
        envelope.claims = claims_ref.clone();
        drop(claims_ref);

        let signed = SignedEnvelope::new_signed(envelope, &signing_key);

        let mut message = capnp::message::Builder::new_default();
        let mut builder =
            message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>();
        signed.write_to(&mut builder);

        let mut bytes = Vec::new();
        capnp::serialize::write_message(&mut bytes, &message)
            .map_err(|e| JsError::new(&format!("serialization failed: {}", e)))?;

        Ok(bytes)
    }

    /// Sign payload, send via WebTransport REQ/REP, unwrap ResponseEnvelope.
    async fn send(&self, payload: &[u8]) -> Result<Vec<u8>, JsError> {
        let request_id = self.next_id();
        let signed = self.sign(payload, request_id, &[])?;
        let response_bytes = self.client
            .request(&signed)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        unwrap_response_envelope(&response_bytes)
    }
}

// ============================================================================
// Request builders (one per service schema)
// ============================================================================

fn build_registry_request<F>(request_id: u64, setup: F) -> Result<Vec<u8>, JsError>
where
    F: FnOnce(crate::registry_capnp::registry_request::Builder),
{
    hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::registry_capnp::registry_request::Builder>();
        req.set_id(request_id);
        setup(req.reborrow());
    })
    .map_err(|e| JsError::new(&e.to_string()))
}

fn build_model_request<F>(request_id: u64, setup: F) -> Result<Vec<u8>, JsError>
where
    F: FnOnce(crate::model_capnp::model_request::Builder),
{
    hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::model_capnp::model_request::Builder>();
        req.set_id(request_id);
        setup(req.reborrow());
    })
    .map_err(|e| JsError::new(&e.to_string()))
}

fn build_policy_request<F>(request_id: u64, setup: F) -> Result<Vec<u8>, JsError>
where
    F: FnOnce(crate::policy_capnp::policy_request::Builder),
{
    hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::policy_capnp::policy_request::Builder>();
        req.set_id(request_id);
        setup(req.reborrow());
    })
    .map_err(|e| JsError::new(&e.to_string()))
}

fn build_mcp_request<F>(request_id: u64, setup: F) -> Result<Vec<u8>, JsError>
where
    F: FnOnce(crate::mcp_capnp::mcp_request::Builder),
{
    hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::mcp_capnp::mcp_request::Builder>();
        req.set_id(request_id);
        setup(req.reborrow());
    })
    .map_err(|e| JsError::new(&e.to_string()))
}

fn build_repo_scoped_request<F>(request_id: u64, repo_id: &str, setup: F) -> Result<Vec<u8>, JsError>
where
    F: FnOnce(crate::registry_capnp::repository_request::Builder),
{
    hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::registry_capnp::registry_request::Builder>();
        req.set_id(request_id);
        let mut repo = req.init_repo();
        repo.set_repo_id(repo_id);
        setup(repo.reborrow());
    })
    .map_err(|e| JsError::new(&e.to_string()))
}

/// Parse a registry response that contains a RepoResult, then extract the inner variant.
fn parse_repo_response(bytes: &[u8]) -> Result<JsValue, JsError> {
    // Use the generated parse_response to get the outer variant
    let parsed = crate::registry_client::RegistryResponseVariant::parse_response(bytes)
        .map_err(|e| JsError::new(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsError::new(&e.to_string()))
}

fn build_model_infer_request<F>(request_id: u64, model_ref: &str, setup: F) -> Result<Vec<u8>, JsError>
where
    F: FnOnce(crate::model_capnp::infer_request::Builder),
{
    hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::model_capnp::model_request::Builder>();
        req.set_id(request_id);
        let mut infer = req.init_infer();
        infer.set_model_ref(model_ref);
        setup(infer.reborrow());
    })
    .map_err(|e| JsError::new(&e.to_string()))
}

fn build_inference_request<F>(request_id: u64, setup: F) -> Result<Vec<u8>, JsError>
where
    F: FnOnce(crate::inference_capnp::inference_request::Builder),
{
    hyprstream_rpc::serialize_message(|msg| {
        let mut req = msg.init_root::<crate::inference_capnp::inference_request::Builder>();
        req.set_id(request_id);
        setup(req.reborrow());
    })
    .map_err(|e| JsError::new(&e.to_string()))
}

// ============================================================================
// Response envelope unwrapping
// ============================================================================

/// Extract the inner payload from a ResponseEnvelope.
///
/// The server wraps all service responses in a ResponseEnvelope (capnp) with:
/// - request_id (u64)
/// - payload (Data — the actual service response bytes)
/// - signature (Data — Ed25519 signature)
/// - signer_pubkey (Data — 32 bytes)
///
/// This extracts the `payload` field so the caller can parse it as a
/// service-specific response message.
fn unwrap_response_envelope(bytes: &[u8]) -> Result<Vec<u8>, JsError> {
    let reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(bytes),
        capnp::message::ReaderOptions::new(),
    )
    .map_err(|e| JsError::new(&format!("failed to read response envelope: {}", e)))?;

    let envelope = reader
        .get_root::<hyprstream_rpc::common_capnp::response_envelope::Reader>()
        .map_err(|e| JsError::new(&format!("failed to parse response envelope: {}", e)))?;

    let payload = envelope
        .get_payload()
        .map_err(|e| JsError::new(&format!("failed to get payload from envelope: {}", e)))?;

    Ok(payload.to_vec())
}

// ============================================================================
// JWT parsing (client-side, no signature verification)
// ============================================================================

fn parse_jwt_claims(jwt_token: &str) -> Result<Option<hyprstream_rpc::auth::Claims>, JsError> {
    if jwt_token.is_empty() {
        return Ok(None);
    }

    let parts: Vec<&str> = jwt_token.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err(JsError::new("Invalid JWT format"));
    }

    use base64::Engine;
    let payload_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|e| JsError::new(&format!("JWT decode error: {}", e)))?;

    let claims: hyprstream_rpc::auth::Claims = serde_json::from_slice(&payload_bytes)
        .map_err(|e| JsError::new(&format!("JWT parse error: {}", e)))?;

    Ok(Some(claims))
}
