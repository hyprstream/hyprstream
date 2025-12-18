//! Command-line interface module.
//!
//! This module provides the CLI functionality for:
//! - Server management
//! - Configuration handling
//! - Model management (HuggingFace, etc.)
//! - LoRA adapter creation and training

pub mod commands;
pub mod context;
pub mod git_handlers;
pub mod handlers;
pub mod policy_handlers;
pub mod worktree_handlers;

pub use context::AppContext;
pub use git_handlers::{
    handle_branch, handle_checkout, handle_clone, handle_commit, handle_infer, handle_info,
    handle_list, handle_lora_train, handle_merge, handle_pull, handle_push, handle_remove,
    handle_status, MergeOptions,
};
pub use handlers::{handle_config, handle_server, FlightServerConfig};
pub use worktree_handlers::{
    handle_worktree_info, handle_worktree_list, handle_worktree_remove,
};
pub use policy_handlers::{
    handle_policy_apply, handle_policy_check, handle_policy_diff, handle_policy_edit,
    handle_policy_history, handle_policy_rollback, handle_policy_show,
    handle_token_create, handle_token_list, handle_token_revoke,
};

/// Device preference strategy
#[derive(Debug, Clone, Copy)]
pub enum DevicePreference {
    /// Request GPU if available, fall back to CPU gracefully
    RequestGPU,
    /// Request CPU-only execution
    RequestCPU,
    /// Require GPU, fail if not available
    RequireGPU,
}

/// Device configuration for command execution
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Device preference strategy
    pub preference: DevicePreference,
    /// Specific CUDA device ID (None = auto-detect)
    pub cuda_device: Option<u32>,
    /// Specific ROCm device ID (None = auto-detect)
    pub rocm_device: Option<u32>,
}

impl DeviceConfig {
    /// Create a GPU request config with auto-detection
    pub fn request_gpu() -> Self {
        Self {
            preference: DevicePreference::RequestGPU,
            cuda_device: None,
            rocm_device: None,
        }
    }

    /// Create a CPU-only config
    pub fn request_cpu() -> Self {
        Self {
            preference: DevicePreference::RequestCPU,
            cuda_device: None,
            rocm_device: None,
        }
    }

    /// Create a GPU requirement config with auto-detection
    pub fn require_gpu() -> Self {
        Self {
            preference: DevicePreference::RequireGPU,
            cuda_device: None,
            rocm_device: None,
        }
    }

    /// Create a config with specific CUDA device
    pub fn cuda_device(device_id: u32) -> Self {
        Self {
            preference: DevicePreference::RequireGPU,
            cuda_device: Some(device_id),
            rocm_device: None,
        }
    }

    /// Create a config with specific ROCm device
    pub fn rocm_device(device_id: u32) -> Self {
        Self {
            preference: DevicePreference::RequireGPU,
            cuda_device: None,
            rocm_device: Some(device_id),
        }
    }
}

/// Runtime configuration for commands
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub device: DeviceConfig,
    pub multi_threaded: bool,
}
