//! RPC types with derived Cap'n Proto serialization.
//!
//! This module contains types that use `#[derive(ToCapnp, FromCapnp)]` for
//! automatic serialization/deserialization with Cap'n Proto.
//!
//! # Usage
//!
//! ```ignore
//! use crate::services::rpc_types::*;
//!
//! // Build a health response using the helper
//! let response = RegistryResponse::health(request_id, &status);
//!
//! // Parse a response
//! let health = HealthStatus::read_from(reader)?;
//! ```

use anyhow::Result;
use capnp::message::Builder;
use capnp::serialize;
use hyprstream_rpc::{FromCapnp, ToCapnp};
use std::path::PathBuf;

use crate::registry_capnp;

// ============================================================================
// Registry Service Types
// ============================================================================

/// Health status for the registry service.
#[derive(Debug, Clone, FromCapnp, ToCapnp)]
#[capnp(registry_capnp::health_status)]
pub struct HealthStatus {
    pub status: String,
    pub repository_count: u32,
    pub worktree_count: u32,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Remote repository information.
#[derive(Debug, Clone, FromCapnp, ToCapnp)]
#[capnp(registry_capnp::remote_info)]
pub struct RemoteInfo {
    pub name: String,
    pub url: String,
    #[capnp(optional)]
    pub push_url: Option<String>,
}

/// Worktree information (internal representation).
#[derive(Debug, Clone)]
pub struct WorktreeData {
    pub path: PathBuf,
    pub branch_name: Option<String>,
    pub head_oid: String,
    pub is_locked: bool,
    pub is_dirty: bool,
}

// Manual FromCapnp for WorktreeData since it has PathBuf
impl FromCapnp for WorktreeData {
    type Reader<'a> = registry_capnp::worktree_info::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        Ok(Self {
            path: PathBuf::from(reader.get_path()?.to_str()?),
            branch_name: {
                let b = reader.get_branch_name()?.to_str()?;
                if b.is_empty() { None } else { Some(b.to_owned()) }
            },
            head_oid: reader.get_head_oid()?.to_str()?.to_owned(),
            is_locked: reader.get_is_locked(),
            is_dirty: reader.get_is_dirty(),
        })
    }
}

impl ToCapnp for WorktreeData {
    type Builder<'a> = registry_capnp::worktree_info::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_path(self.path.to_string_lossy());
        if let Some(ref branch) = self.branch_name {
            builder.set_branch_name(branch);
        }
        builder.set_head_oid(&self.head_oid);
        builder.set_is_locked(self.is_locked);
        builder.set_is_dirty(self.is_dirty);
    }
}

// ============================================================================
// Response Builder Helper
// ============================================================================

/// Helper for building registry service responses.
///
/// Eliminates boilerplate by providing typed response builders.
///
/// # Example
///
/// ```ignore
/// // Before (manual - 15 lines):
/// fn build_health_response(&self, id: u64, status: &HealthStatus) -> Vec<u8> {
///     let mut message = Builder::new_default();
///     let mut response = message.init_root::<registry_response::Builder>();
///     response.set_request_id(id);
///     let mut health = response.init_health();
///     health.set_status(&status.status);
///     // ... 10 more lines
/// }
///
/// // After (1 line):
/// RegistryResponse::health(id, &status)
/// ```
pub struct RegistryResponse;

impl RegistryResponse {
    /// Build an error response.
    pub fn error(request_id: u64, message: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut error_info = response.init_error();
        error_info.set_message(message);
        error_info.set_code("ERROR");
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build an unauthorized error response.
    ///
    /// Used when a policy check fails for the requested operation.
    pub fn unauthorized(request_id: u64, subject: &str, resource: &str, operation: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut error_info = response.init_error();
        error_info.set_message(format!(
            "unauthorized: {subject} cannot {operation} on {resource}"
        ));
        error_info.set_code("UNAUTHORIZED");
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a success response (void).
    pub fn success(request_id: u64) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);
        response.set_success(());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a health status response.
    pub fn health(request_id: u64, status: &HealthStatus) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut health = response.init_health();
        status.write_to(&mut health);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a remotes list response.
    pub fn remotes(request_id: u64, remotes: &[RemoteInfo]) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut remotes_list = response.init_remotes(remotes.len() as u32);
        for (i, remote) in remotes.iter().enumerate() {
            let mut r = remotes_list.reborrow().get(i as u32);
            remote.write_to(&mut r);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a worktrees list response.
    pub fn worktrees(request_id: u64, worktrees: &[WorktreeData]) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut wt_builder = response.init_worktrees(worktrees.len() as u32);
        for (i, wt) in worktrees.iter().enumerate() {
            let mut wt_entry = wt_builder.reborrow().get(i as u32);
            wt.write_to(&mut wt_entry);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a branches list response.
    pub fn branches(request_id: u64, branches: &[String]) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut branches_builder = response.init_branches(branches.len() as u32);
        for (i, branch) in branches.iter().enumerate() {
            branches_builder.set(i as u32, branch);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a commit OID response.
    pub fn commit_oid(request_id: u64, oid: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);
        response.set_commit_oid(oid);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a ref OID response (for getHead, getRef).
    pub fn ref_oid(request_id: u64, oid: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);
        response.set_ref_oid(oid);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a path response.
    pub fn path(request_id: u64, path: &std::path::Path) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);
        response.set_path(path.to_string_lossy());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a repository response.
    pub fn repository(request_id: u64, repo: &git2db::TrackedRepository) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut repo_builder = response.init_repository();
        repo_builder.set_id(repo.id.to_string());
        if let Some(ref name) = repo.name {
            repo_builder.set_name(name);
        }
        repo_builder.set_url(&repo.url);
        repo_builder.set_worktree_path(repo.worktree_path.to_string_lossy());
        repo_builder.set_tracking_ref(repo.tracking_ref.to_string());
        if let Some(ref oid) = repo.current_oid {
            repo_builder.set_current_oid(oid);
        }
        repo_builder.set_registered_at(repo.registered_at);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a repositories list response.
    pub fn repositories(request_id: u64, repos: &[git2db::TrackedRepository]) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut repos_builder = response.init_repositories(repos.len() as u32);
        for (i, repo) in repos.iter().enumerate() {
            let mut repo_builder = repos_builder.reborrow().get(i as u32);
            repo_builder.set_id(repo.id.to_string());
            if let Some(ref name) = repo.name {
                repo_builder.set_name(name);
            }
            repo_builder.set_url(&repo.url);
            repo_builder.set_worktree_path(repo.worktree_path.to_string_lossy());
            repo_builder.set_tracking_ref(repo.tracking_ref.to_string());
            if let Some(ref oid) = repo.current_oid {
                repo_builder.set_current_oid(oid);
            }
            repo_builder.set_registered_at(repo.registered_at);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a repository status response.
    pub fn repository_status(request_id: u64, status: &git2db::RepositoryStatus) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut status_builder = response.init_repository_status();

        // Set branch (optional)
        if let Some(ref branch) = status.branch {
            status_builder.set_branch(branch);
        }

        // Set head OID (optional)
        if let Some(ref oid) = status.head {
            status_builder.set_head_oid(oid.to_string());
        }

        // Set other fields
        status_builder.set_ahead(status.ahead as u32);
        status_builder.set_behind(status.behind as u32);
        status_builder.set_is_clean(status.is_clean);

        // Set modified files
        let mut files_builder = status_builder.init_modified_files(status.modified_files.len() as u32);
        for (i, file) in status.modified_files.iter().enumerate() {
            files_builder.set(i as u32, file.to_string_lossy());
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }
}

// ============================================================================
// Request Types (for client-side serialization)
// ============================================================================

/// Clone repository request.
#[derive(Debug, Clone, ToCapnp)]
#[capnp(registry_capnp::clone_request)]
pub struct CloneRequest {
    pub url: String,
    pub name: String,
    pub shallow: bool,
    pub depth: u32,
    pub branch: String,
}

/// Commit request.
#[derive(Debug, Clone, ToCapnp)]
#[capnp(registry_capnp::commit_request)]
pub struct CommitRequest {
    pub repo_id: String,
    pub message: String,
    pub author: String,
    pub email: String,
}

/// Add remote request.
#[derive(Debug, Clone, ToCapnp)]
#[capnp(registry_capnp::add_remote_request)]
pub struct AddRemoteRequest {
    pub repo_id: String,
    pub name: String,
    pub url: String,
}

// ============================================================================
// Inference Service Types
// ============================================================================

use crate::inference_capnp;
use crate::config::{FinishReason, GenerationResult, ModelInfo};
use crate::runtime::GenerationStats;

/// Stream started info returned by generate_stream.
///
/// Now includes server_pubkey for E2E authenticated streaming via DH.
#[derive(Debug, Clone)]
pub struct StreamStartedInfo {
    /// Stream ID for client display/logging
    pub stream_id: String,
    /// StreamService SUB endpoint
    pub endpoint: String,
    /// Server's ephemeral Ristretto255 public key for DH (32 bytes)
    pub server_pubkey: [u8; 32],
}

impl FromCapnp for StreamStartedInfo {
    type Reader<'a> = inference_capnp::stream_info::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let stream_id = reader.get_stream_id()?.to_str()?.to_owned();
        let endpoint = reader.get_endpoint()?.to_str()?.to_owned();

        let pubkey_data = reader.get_server_pubkey()?;
        if pubkey_data.len() != 32 {
            anyhow::bail!("Invalid server_pubkey length: {}", pubkey_data.len());
        }
        let mut server_pubkey = [0u8; 32];
        server_pubkey.copy_from_slice(pubkey_data);

        Ok(Self {
            stream_id,
            endpoint,
            server_pubkey,
        })
    }
}

// ============================================================================
// Inference Service Response Helper
// ============================================================================

/// Helper for building inference service responses.
pub struct InferenceResponse;

impl InferenceResponse {
    /// Build an error response.
    pub fn error(request_id: u64, message: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut error_info = response.init_error();
        error_info.set_message(message);
        error_info.set_code("ERROR");
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build an unauthorized error response.
    ///
    /// Used when a policy check fails for the requested operation.
    pub fn unauthorized(request_id: u64, subject: &str, resource: &str, operation: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut error_info = response.init_error();
        error_info.set_message(format!(
            "unauthorized: {subject} cannot {operation} on {resource}"
        ));
        error_info.set_code("UNAUTHORIZED");
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a success response (void).
    pub fn success(request_id: u64) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);
        response.set_success(());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a generation result response.
    pub fn generation_result(request_id: u64, result: &GenerationResult) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut gen_result = response.init_generation_result();
        gen_result.set_text(&result.text);
        gen_result.set_tokens_generated(result.tokens_generated as u32);
        gen_result.set_finish_reason(Self::finish_reason_to_capnp(&result.finish_reason));
        gen_result.set_generation_time_ms(result.generation_time_ms);
        gen_result.set_tokens_per_second(result.tokens_per_second);
        // Prefill metrics
        gen_result.set_prefill_tokens(result.prefill_tokens as u32);
        gen_result.set_prefill_time_ms(result.prefill_time_ms);
        gen_result.set_prefill_tokens_per_sec(result.prefill_tokens_per_sec);
        // Inference metrics
        gen_result.set_inference_tokens(result.inference_tokens as u32);
        gen_result.set_inference_time_ms(result.inference_time_ms);
        gen_result.set_inference_tokens_per_sec(result.inference_tokens_per_sec);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a stream started response.
    ///
    /// # Arguments
    /// * `request_id` - Correlation ID for the request
    /// * `stream_id` - Stream identifier (legacy, for client display)
    /// * `endpoint` - StreamService SUB endpoint for client subscription
    /// * `server_pubkey` - Server's ephemeral Ristretto255 public key for DH (32 bytes)
    pub fn stream_started(request_id: u64, stream_id: &str, endpoint: &str, server_pubkey: &[u8; 32]) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut stream_info = response.init_stream_started();
        stream_info.set_stream_id(stream_id);
        stream_info.set_endpoint(endpoint);
        stream_info.set_server_pubkey(server_pubkey);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a stream authorized response.
    ///
    /// Sent in response to StartStream to confirm stream subscription is authorized.
    /// Future: will include server's ephemeral public key for DH key exchange.
    pub fn stream_authorized(request_id: u64, stream_id: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut auth_response = response.init_stream_authorized();
        auth_response.set_stream_id(stream_id);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a model info response.
    pub fn model_info(request_id: u64, info: &ModelInfo, has_lora: bool) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut model_info = response.init_model_info();
        model_info.set_model_id(&info.name);
        model_info.set_architecture(&info.architecture);
        model_info.set_vocab_size(info.vocab_size as u32);
        model_info.set_hidden_size(info.hidden_size as u32);
        model_info.set_num_layers(info.num_hidden_layers.unwrap_or(0) as u32);
        model_info.set_num_heads(info.num_attention_heads.unwrap_or(0) as u32);
        model_info.set_max_sequence_length(info.context_length as u32);
        model_info.set_quantization(info.quantization.as_deref().unwrap_or("none"));
        model_info.set_has_vision(false);
        model_info.set_lora_loaded(has_lora);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a ready response.
    pub fn ready(request_id: u64, ready: bool) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);
        response.set_ready(ready);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a template result response.
    pub fn template_result(request_id: u64, result: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);
        response.set_template_result(result);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a has LoRA result response.
    pub fn has_lora(request_id: u64, has_lora: bool) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);
        response.set_has_lora_result(has_lora);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a health status response.
    pub fn health(request_id: u64, model_loaded: bool) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut health = response.init_health();
        health.set_status("healthy");
        health.set_model_loaded(model_loaded);
        health.set_kv_cache_usage_percent(0.0);
        health.set_gpu_memory_used_mb(0);
        health.set_gpu_memory_total_mb(0);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build an inference payload message with a token.
    ///
    /// The payload is serialized into streaming_capnp::StreamBlock.payloads.
    /// Ordering is handled by prevMac in the outer StreamBlock wrapper.
    pub fn inference_token(stream_id: &str, token: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut payload = msg.init_root::<inference_capnp::inference_payload::Builder>();
        payload.set_stream_id(stream_id);
        payload.set_token(token);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build an inference payload error message.
    pub fn inference_error(stream_id: &str, error: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut payload = msg.init_root::<inference_capnp::inference_payload::Builder>();
        payload.set_stream_id(stream_id);

        let mut error_info = payload.init_error();
        error_info.set_message(error);
        error_info.set_code("GENERATION_ERROR");
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build an inference payload complete message.
    pub fn inference_complete(stream_id: &str, stats: &GenerationStats) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut payload = msg.init_root::<inference_capnp::inference_payload::Builder>();
        payload.set_stream_id(stream_id);

        let mut complete = payload.init_complete();
        complete.set_tokens_generated(stats.tokens_generated as u32);
        // Use Stop as default if no finish_reason is set
        let finish_reason = stats.finish_reason.as_ref().unwrap_or(&FinishReason::Stop);
        complete.set_finish_reason(Self::finish_reason_to_capnp(finish_reason));
        complete.set_generation_time_ms(stats.generation_time_ms);
        complete.set_tokens_per_second(stats.tokens_per_second);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Convert FinishReason to Cap'n Proto enum.
    fn finish_reason_to_capnp(reason: &FinishReason) -> inference_capnp::FinishReason {
        match reason {
            FinishReason::MaxTokens => inference_capnp::FinishReason::MaxTokens,
            FinishReason::StopToken(_) => inference_capnp::FinishReason::StopToken,
            FinishReason::EndOfSequence => inference_capnp::FinishReason::EndOfSequence,
            FinishReason::Error(_) => inference_capnp::FinishReason::Error,
            FinishReason::Stop => inference_capnp::FinishReason::Stop,
        }
    }
}

// ============================================================================
// Authenticated Streaming Helpers
// ============================================================================

use hyprstream_rpc::common_capnp;
use hyprstream_rpc::streaming_capnp;
use hyprstream_rpc::envelope::{RequestEnvelope, SignedEnvelope};
use hyprstream_rpc::prelude::SigningKey;

/// Build a StreamRegister message wrapped in SignedEnvelope.
///
/// This replaces the legacy "AUTHORIZE|topic|exp" format with secure signed registration.
pub fn build_stream_register_envelope(
    topic: &str,
    exp: i64,
    signing_key: &SigningKey,
    claims: Option<hyprstream_rpc::auth::Claims>,
) -> Vec<u8> {
    use hyprstream_rpc::ToCapnp;

    // Build StreamRegister payload
    let mut register_msg = Builder::new_default();
    {
        let mut register = register_msg.init_root::<streaming_capnp::stream_register::Builder>();
        register.set_topic(topic);
        register.set_exp(exp);
    }

    let mut payload = Vec::new();
    serialize::write_message(&mut payload, &register_msg).unwrap_or_default();

    // Create request envelope
    let envelope = RequestEnvelope {
        request_id: 0, // Not used for stream registration
        identity: hyprstream_rpc::envelope::RequestIdentity::Local {
            user: "system".to_owned(),
        },
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64,
        nonce: rand::random(),
        payload,
        claims,
        ephemeral_pubkey: None, // TODO: Add DH for key derivation
    };

    // Sign and serialize
    let signed = SignedEnvelope::new_signed(envelope, signing_key);

    let mut out = Builder::new_default();
    let mut builder = out.init_root::<common_capnp::signed_envelope::Builder>();
    signed.write_to(&mut builder);

    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &out).unwrap_or_default();
    bytes
}
// ============================================================================
// StreamBlock Types - Re-exported from hyprstream-rpc
// ============================================================================

// Re-export generic streaming types from hyprstream-rpc
pub use hyprstream_rpc::streaming::{
    BatchingConfig,
    StreamBuilder,
    StreamFrames,
    StreamHandle,
    StreamHmacState,
    StreamPayload,
    StreamPayloadData,
    StreamVerifier,
};

/// Alias for backwards compatibility
pub type ParsedStreamPayload = StreamPayload;

// ============================================================================
// Inference-Specific Stream Types
// ============================================================================

/// Inference-specific completion metadata (serialized into StreamPayload.complete).
///
/// This is the application-layer completion data for inference streams.
/// Serialized as JSON for simplicity and debuggability.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceComplete {
    pub tokens_generated: usize,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
    pub finish_reason: String,
    // Prefill metrics
    pub prefill_tokens: usize,
    pub prefill_time_ms: u64,
    pub prefill_tokens_per_sec: f32,
    // Inference metrics
    pub inference_tokens: usize,
    pub inference_time_ms: u64,
    pub inference_tokens_per_sec: f32,
    pub inference_tokens_per_sec_ema: f32,
    // Optional quality metrics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perplexity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avg_entropy: Option<f32>,
}

impl InferenceComplete {
    /// Serialize to bytes for StreamPayload.complete.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize from StreamPayload.complete bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| anyhow::anyhow!("Failed to parse InferenceComplete: {}", e))
    }
}

impl From<&crate::runtime::GenerationStats> for InferenceComplete {
    fn from(stats: &crate::runtime::GenerationStats) -> Self {
        let finish_reason = match &stats.finish_reason {
            Some(crate::config::FinishReason::MaxTokens) => "length",
            Some(crate::config::FinishReason::EndOfSequence) => "eos",
            Some(crate::config::FinishReason::Stop) => "stop",
            Some(crate::config::FinishReason::StopToken(_)) => "stop",
            Some(crate::config::FinishReason::Error(_)) => "error",
            None => "unknown",
        };
        Self {
            tokens_generated: stats.tokens_generated,
            generation_time_ms: stats.generation_time_ms,
            tokens_per_second: stats.tokens_per_second,
            finish_reason: finish_reason.to_owned(),
            prefill_tokens: stats.prefill_tokens,
            prefill_time_ms: stats.prefill_time_ms,
            prefill_tokens_per_sec: stats.prefill_tokens_per_sec,
            inference_tokens: stats.inference_tokens,
            inference_time_ms: stats.inference_time_ms,
            inference_tokens_per_sec: stats.inference_tokens_per_sec,
            inference_tokens_per_sec_ema: stats.inference_tokens_per_sec_ema,
            perplexity: stats.quality_metrics.as_ref().map(|m| m.perplexity),
            avg_entropy: stats.quality_metrics.as_ref().map(|m| m.avg_entropy),
        }
    }
}

/// Inference-specific stream payload (parsed from generic StreamPayload).
///
/// Provides typed access to inference stream data:
/// - Token: UTF-8 text token
/// - Error: Error message
/// - Complete: Generation statistics
///
/// Note: Stream identity comes from the DH-derived topic, not from payload fields.
/// The topic cryptographically binds the stream to the DH key exchange.
#[derive(Debug, Clone)]
pub enum InferenceStreamPayload {
    /// UTF-8 text token
    Token(String),
    /// Error during streaming
    Error(String),
    /// Completion with generation statistics
    Complete(InferenceComplete),
}

/// Extension trait to convert generic StreamPayload to inference-specific payload.
pub trait StreamPayloadExt {
    /// Convert generic payload to inference-specific payload.
    ///
    /// Interprets Data as UTF-8 text tokens and Complete as InferenceComplete.
    fn to_inference(self) -> Result<InferenceStreamPayload>;
}

impl StreamPayloadExt for StreamPayload {
    fn to_inference(self) -> Result<InferenceStreamPayload> {
        match self {
            StreamPayload::Data(data) => {
                let text = String::from_utf8(data)
                    .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in token: {}", e))?;
                Ok(InferenceStreamPayload::Token(text))
            }
            StreamPayload::Error(message) => {
                Ok(InferenceStreamPayload::Error(message))
            }
            StreamPayload::Complete(data) => {
                let stats = InferenceComplete::from_bytes(&data)?;
                Ok(InferenceStreamPayload::Complete(stats))
            }
        }
    }
}


// ============================================================================
// Policy Service Types
// ============================================================================

use crate::policy_capnp;

/// Token information response.
#[derive(Debug, Clone, FromCapnp)]
#[capnp(policy_capnp::token_info)]
pub struct TokenInfo {
    pub token: String,
    pub expires_at: i64,
}

// ============================================================================
// Policy Service Response Helper
// ============================================================================

/// Helper for building policy service responses.
///
/// Eliminates boilerplate by providing typed response builders.
///
/// # Example
///
/// ```ignore
/// // Before (manual - 12 lines):
/// let mut msg = Builder::new_default();
/// let mut response = msg.init_root::<policy_response::Builder>();
/// response.set_request_id(id);
/// let mut token_info = response.init_success();
/// token_info.set_token(&token);
/// // ... 7 more lines
///
/// // After (1 line):
/// PolicyResponse::token_success(id, &token, expires_at)
/// ```
pub struct PolicyResponse;

impl PolicyResponse {
    /// Build an error response.
    pub fn error(request_id: u64, message: &str, code: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<policy_capnp::issue_token_response::Builder>();
        response.set_request_id(request_id);

        let mut error_info = response.init_error();
        error_info.set_message(message);
        error_info.set_code(code);
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build an unauthorized error response.
    pub fn unauthorized(request_id: u64, scope: &str) -> Vec<u8> {
        Self::error(
            request_id,
            &format!("Access denied for scope: {scope}"),
            "UNAUTHORIZED"
        )
    }

    /// Build a TTL exceeded error response.
    pub fn ttl_exceeded(request_id: u64, requested: u32, max: u32) -> Vec<u8> {
        Self::error(
            request_id,
            &format!("TTL exceeds maximum: {requested} > {max}"),
            "TTL_EXCEEDED"
        )
    }

    /// Build a token success response.
    pub fn token_success(request_id: u64, token: &str, expires_at: i64) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<policy_capnp::issue_token_response::Builder>();
        response.set_request_id(request_id);

        let mut token_info = response.init_success();
        token_info.set_token(token);
        token_info.set_expires_at(expires_at);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }
}
