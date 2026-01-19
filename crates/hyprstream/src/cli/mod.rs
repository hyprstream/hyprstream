//! Command-line interface module.
//!
//! This module provides the CLI functionality for:
//! - Server management
//! - Configuration handling
//! - Model management (HuggingFace, etc.)
//! - LoRA adapter creation and training

pub mod commands;
pub mod context;
pub mod daemon;
pub mod git_handlers;
pub mod handlers;
pub mod policy_handlers;
pub mod remote_handlers;
pub mod systemd_setup;
pub mod training_handlers;
pub mod worker_handlers;
pub mod worktree_handlers;

pub use context::AppContext;
pub use git_handlers::{
    apply_policy_template_to_model, handle_branch, handle_checkout, handle_clone,
    handle_infer, handle_info, handle_list, handle_pull, handle_remove, handle_status,
};

#[cfg(feature = "experimental")]
pub use git_handlers::{handle_commit, handle_merge, handle_push, MergeOptions};
pub use training_handlers::{
    handle_training_batch, handle_training_checkpoint, handle_training_infer, handle_training_init,
};
pub use handlers::handle_config;
pub use worktree_handlers::{
    handle_worktree_add, handle_worktree_info, handle_worktree_list, handle_worktree_remove,
};
pub use policy_handlers::{
    handle_policy_apply, handle_policy_apply_template, handle_policy_check, handle_policy_diff,
    handle_policy_edit, handle_policy_history, handle_policy_list_templates, handle_policy_rollback,
    handle_policy_show, handle_token_create,
    get_template, get_templates, load_or_generate_signing_key, PolicyTemplate,
};
pub use remote_handlers::{
    handle_remote_add, handle_remote_list, handle_remote_remove, handle_remote_rename,
    handle_remote_set_url,
};
pub use worker_handlers::{
    handle_images_df, handle_images_list, handle_images_pull, handle_images_rm,
    handle_worker_exec, handle_worker_list, handle_worker_restart, handle_worker_rm,
    handle_worker_run, handle_worker_start, handle_worker_stats, handle_worker_status,
    handle_worker_stop,
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
