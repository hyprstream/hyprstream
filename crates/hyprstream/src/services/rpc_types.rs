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
}

// Manual FromCapnp for WorktreeData since it has PathBuf
impl FromCapnp for WorktreeData {
    type Reader<'a> = registry_capnp::worktree_info::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        Ok(Self {
            path: PathBuf::from(reader.get_path()?.to_str()?),
            branch_name: {
                let b = reader.get_branch_name()?.to_str()?;
                if b.is_empty() { None } else { Some(b.to_string()) }
            },
            head_oid: reader.get_head_oid()?.to_str()?.to_string(),
            is_locked: reader.get_is_locked(),
        })
    }
}

impl ToCapnp for WorktreeData {
    type Builder<'a> = registry_capnp::worktree_info::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_path(&self.path.to_string_lossy());
        if let Some(ref branch) = self.branch_name {
            builder.set_branch_name(branch);
        }
        builder.set_head_oid(&self.head_oid);
        builder.set_is_locked(self.is_locked);
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
        error_info.set_message(&format!(
            "unauthorized: {} cannot {} on {}",
            subject, operation, resource
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
        response.set_path(&path.to_string_lossy());

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
        repo_builder.set_id(&repo.id.to_string());
        if let Some(ref name) = repo.name {
            repo_builder.set_name(name);
        }
        repo_builder.set_url(&repo.url);
        repo_builder.set_worktree_path(&repo.worktree_path.to_string_lossy());
        repo_builder.set_tracking_ref(&repo.tracking_ref.to_string());
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
            repo_builder.set_id(&repo.id.to_string());
            if let Some(ref name) = repo.name {
                repo_builder.set_name(name);
            }
            repo_builder.set_url(&repo.url);
            repo_builder.set_worktree_path(&repo.worktree_path.to_string_lossy());
            repo_builder.set_tracking_ref(&repo.tracking_ref.to_string());
            if let Some(ref oid) = repo.current_oid {
                repo_builder.set_current_oid(oid);
            }
            repo_builder.set_registered_at(repo.registered_at);
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
#[derive(Debug, Clone, FromCapnp)]
#[capnp(inference_capnp::stream_info)]
pub struct StreamStartedInfo {
    #[capnp(rename = "stream_id")]
    pub stream_id: String,
    pub endpoint: String,
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
        error_info.set_message(&format!(
            "unauthorized: {} cannot {} on {}",
            subject, operation, resource
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
    pub fn stream_started(request_id: u64, stream_id: &str, endpoint: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut response = msg.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut stream_info = response.init_stream_started();
        stream_info.set_stream_id(stream_id);
        stream_info.set_endpoint(endpoint);

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

    /// Build a stream chunk message.
    pub fn stream_chunk(stream_id: &str, seq_num: u32, text: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut chunk = msg.init_root::<inference_capnp::stream_chunk::Builder>();
        chunk.set_stream_id(stream_id);
        chunk.set_sequence_num(seq_num);
        chunk.set_text(text);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a stream error message.
    pub fn stream_error(stream_id: &str, seq_num: u32, error: &str) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut chunk = msg.init_root::<inference_capnp::stream_chunk::Builder>();
        chunk.set_stream_id(stream_id);
        chunk.set_sequence_num(seq_num);

        let mut error_info = chunk.init_error();
        error_info.set_message(error);
        error_info.set_code("GENERATION_ERROR");
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &msg).unwrap_or_default();
        bytes
    }

    /// Build a stream complete message.
    pub fn stream_complete(stream_id: &str, seq_num: u32, stats: &GenerationStats) -> Vec<u8> {
        let mut msg = Builder::new_default();
        let mut chunk = msg.init_root::<inference_capnp::stream_chunk::Builder>();
        chunk.set_stream_id(stream_id);
        chunk.set_sequence_num(seq_num);

        let mut complete = chunk.init_complete();
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
