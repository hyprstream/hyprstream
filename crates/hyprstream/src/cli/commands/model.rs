//! Model management CLI commands

use crate::cli::DeviceConfig;
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

/// Model management commands
#[derive(Args)]
pub struct ModelCommand {
    #[command(subcommand)]
    pub action: ModelAction,
}

/// Model management actions
#[derive(Subcommand)]
pub enum ModelAction {
    /// Pull a model from a registry
    Pull {
        /// Git repository URL (e.g., https://huggingface.co/microsoft/DialoGPT-medium)
        uri: String,

        /// Force re-download even if cached
        #[arg(long)]
        force: bool,

        /// Specific files to download
        #[arg(long, value_delimiter = ',')]
        files: Option<Vec<String>>,

        /// Preferred format (safetensors, pytorch)
        #[arg(long, default_value = "safetensors")]
        format: String,

        /// Auto-convert to SafeTensors if not available
        #[arg(long, default_value = "true")]
        auto_convert: bool,

        /// Show download progress
        #[arg(long, default_value = "true")]
        progress: bool,
    },

    /// List available models
    List {
        /// Filter by registry type (hf, custom)
        #[arg(long)]
        registry: Option<String>,

        /// Search query to filter models
        #[arg(long)]
        search: Option<String>,

        /// Include remote models (not just local cache)
        #[arg(long)]
        remote: bool,

        /// Output format (table, json)
        #[arg(long, default_value = "table")]
        format: String,

        /// Show git reference info (branch/tag/commit)
        #[arg(long, default_value = "true")]
        show_git_ref: bool,

        /// Show repository status (clean/dirty)
        #[arg(long)]
        show_status: bool,

        /// Filter by git branch
        #[arg(long)]
        branch: Option<String>,

        /// Filter by git tag
        #[arg(long)]
        tag: Option<String>,

        /// Show only models with uncommitted changes
        #[arg(long)]
        dirty_only: bool,
    },

    /// Get detailed information about a model
    Inspect {
        /// Model URI
        uri: String,

        /// Output format (json, yaml)
        #[arg(long, default_value = "json")]
        format: String,
    },

    // Share and Import commands removed - redundant with Clone/Push
    // P2P is handled transparently at git transport layer (GitTorrent)
    // Use: hyprstream clone <url> (works with gittorrent:// URLs)
    // Use: hyprstream push <remote> <branch>

    /// Remove a model from local cache
    Remove {
        /// Model name or reference (e.g., "gitignore", "qwen/qwen-2b")
        uri: String,

        /// Keep metadata but remove files
        #[arg(long)]
        keep_metadata: bool,

        /// Confirm removal without prompting
        #[arg(long)]
        yes: bool,
    },

    // Search functionality has been removed - use 'model list' with search filter instead
    /// Repair model metadata and fix inconsistencies
    Repair {
        /// Perform automatic repairs without confirmation
        #[arg(long)]
        yes: bool,

        /// Show detailed repair information
        #[arg(long)]
        verbose: bool,
    },

    /// Show cache status and statistics
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// Convert model between formats
    Convert {
        /// Source model path or URI
        source: String,

        /// Target format (safetensors)
        #[arg(long, default_value = "safetensors")]
        to: String,

        /// Output path (optional, defaults to same directory)
        #[arg(long)]
        output: Option<String>,

        /// Target precision (bf16, fp16, fp32)
        #[arg(long, default_value = "bf16")]
        precision: String,

        /// Verify conversion accuracy
        #[arg(long)]
        verify: bool,
    },

    /// List available registries
    Registries,

    /// Run pure base model inference without any LoRA adapters
    Infer {
        /// Model reference (e.g., "Qwen3-4B", "qwen/qwen-2b", "model:branch")
        model: String,

        /// Prompt text
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate (overrides model default)
        #[arg(short = 'm', long)]
        max_tokens: Option<usize>,

        /// Temperature for sampling (overrides model default)
        #[arg(short = 't', long)]
        temperature: Option<f32>,

        /// Top-p (nucleus) sampling (overrides model default)
        #[arg(long)]
        top_p: Option<f32>,

        /// Top-k sampling (overrides model default)
        #[arg(long)]
        top_k: Option<usize>,

        /// Stream output tokens as they're generated
        #[arg(short = 's', long)]
        stream: bool,

        /// Force re-download even if cached
        #[arg(long)]
        force_download: bool,
    },
}

/// Cache management actions
#[derive(Subcommand)]
pub enum CacheAction {
    /// Show cache status
    Status,

    /// Clean up old cached models
    Cleanup {
        /// Confirm cleanup without prompting
        #[arg(long)]
        yes: bool,
    },

    /// Verify integrity of cached models
    Verify {
        /// Model URI to verify (optional, verifies all if not specified)
        uri: Option<String>,
    },
}

impl ModelAction {
    /// Returns the device configuration for this action
    pub fn device_config(&self) -> crate::cli::DeviceConfig {
        match self {
            ModelAction::Infer { .. } => DeviceConfig::request_gpu(),
            _ => DeviceConfig::request_cpu(),
        }
    }
}

/// Model pull configuration
#[derive(Debug, Clone)]
pub struct PullConfig {
    pub uri: String,
    pub force: bool,
    pub files: Option<Vec<String>>,
    pub show_progress: bool,
}

/// Model list configuration
#[derive(Debug, Clone)]
pub struct ListConfig {
    pub registry: Option<String>,
    pub search: Option<String>,
    pub include_remote: bool,
    pub format: OutputFormat,
    pub show_git_ref: bool,
    pub show_status: bool,
    pub branch_filter: Option<String>,
    pub tag_filter: Option<String>,
    pub dirty_only: bool,
}

/// Output format options
#[derive(Debug, Clone)]
pub enum OutputFormat {
    Table,
    Json,
    Yaml,
}

impl From<&str> for OutputFormat {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            _ => Self::Table,
        }
    }
}

/// Git information for a model repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitInfo {
    pub current_ref: Option<String>,
    pub ref_type: RefType,
    pub commit: Option<String>,
    pub short_commit: Option<String>,
    pub is_dirty: bool,
    pub last_commit_date: Option<String>,
}

/// Type of git reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefType {
    Branch,
    Tag,
    Commit,
    Detached,
}

/// Model information for display
#[derive(Debug, Clone)]
pub struct ModelDisplayInfo {
    pub uri: String,
    pub registry: String,
    pub org: String,
    pub name: String,
    pub size_mb: Option<f64>,
    pub cached: bool,
    pub files_count: usize,
    pub last_accessed: Option<String>,
    pub parameters: Option<String>,
    pub model_type: String,
    pub git_info: Option<GitInfo>,
}

impl ModelDisplayInfo {
    /// Format size for display
    pub fn format_size(&self) -> String {
        match self.size_mb {
            Some(size) => {
                if size < 1024.0 {
                    format!("{:.1} MB", size)
                } else {
                    format!("{:.1} GB", size / 1024.0)
                }
            }
            None => "Unknown".to_string(),
        }
    }

    /// Format parameters for display
    pub fn format_parameters(&self) -> String {
        match &self.parameters {
            Some(params) => params.clone(),
            None => "Unknown".to_string(),
        }
    }

    /// Format last accessed time
    pub fn format_last_accessed(&self) -> String {
        match &self.last_accessed {
            Some(time) => time.clone(),
            None => "Never".to_string(),
        }
    }

    /// Format git reference for display
    pub fn format_git_ref(&self) -> String {
        match &self.git_info {
            Some(git) => match &git.current_ref {
                Some(ref_name) => ref_name.clone(),
                None => git.short_commit.clone().unwrap_or("unknown".to_string()),
            },
            None => "n/a".to_string(),
        }
    }

    /// Format git status for display
    pub fn format_git_status(&self) -> String {
        match &self.git_info {
            Some(git) => {
                if git.is_dirty {
                    "dirty".to_string()
                } else {
                    "clean".to_string()
                }
            }
            None => "n/a".to_string(),
        }
    }

    /// Get git commit for display
    pub fn format_git_commit(&self) -> String {
        match &self.git_info {
            Some(git) => git.short_commit.clone().unwrap_or("unknown".to_string()),
            None => "n/a".to_string(),
        }
    }
}

impl GitInfo {
    /// Create GitInfo from a git repository path
    pub fn from_repo_path(repo_path: &std::path::Path) -> Option<Self> {
        let repo = crate::git::get_repository(repo_path).ok()?;

        // Get current reference
        let (current_ref, ref_type) = match repo.head() {
            Ok(head) => {
                if head.is_branch() {
                    let name = head.shorthand().unwrap_or("unknown").to_string();
                    (Some(name), RefType::Branch)
                } else if head.is_tag() {
                    let name = head.shorthand().unwrap_or("unknown").to_string();
                    (Some(name), RefType::Tag)
                } else {
                    (None, RefType::Detached)
                }
            }
            Err(_) => (None, RefType::Detached),
        };

        // Get commit information
        let (commit, short_commit) = match repo.head().ok()?.peel_to_commit() {
            Ok(commit) => {
                let full = commit.id().to_string();
                let short = if full.len() > 7 {
                    full[..7].to_string()
                } else {
                    full.clone()
                };
                (Some(full), Some(short))
            }
            Err(_) => (None, None),
        };

        // Check if repository is dirty
        let is_dirty = repo
            .statuses(None)
            .map(|statuses| !statuses.is_empty())
            .unwrap_or(false);

        // Get last commit date
        let last_commit_date = repo
            .head()
            .ok()
            .and_then(|head| head.peel_to_commit().ok())
            .map(|commit| {
                let seconds = commit.time().seconds();
                chrono::DateTime::from_timestamp(seconds, 0)
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            });

        Some(GitInfo {
            current_ref,
            ref_type,
            commit,
            short_commit,
            is_dirty,
            last_commit_date,
        })
    }

    /// Create GitInfo from a git2db RepositoryStatus object
    pub fn from_status(status: &git2db::RepositoryStatus) -> Self {
        // Get branch information
        let (current_ref, ref_type) = if let Some(ref branch) = &status.branch {
            (Some(branch.clone()), RefType::Branch)
        } else {
            // Detached HEAD or other state
            (None, RefType::Detached)
        };

        // Get commit information from head Oid
        let (commit, short_commit) = if let Some(oid) = &status.head {
            let full_id = oid.to_string();
            // Take first 7 characters for short commit (standard git convention)
            let short_id = if full_id.len() >= 7 {
                full_id[..7].to_string()
            } else {
                full_id.clone()
            };
            (Some(full_id), Some(short_id))
        } else {
            (None, None)
        };

        // Note: RepositoryStatus doesn't provide commit timestamps
        let last_commit_date = None;

        // Check if repository is dirty (has uncommitted changes)
        let is_dirty = !status.is_clean;

        GitInfo {
            current_ref,
            ref_type,
            commit,
            short_commit,
            is_dirty,
            last_commit_date,
        }
    }

    /// Create GitInfo from a bare repository path
    /// This works with bare repos (unlike status which requires worktrees)
    pub fn from_bare_repo(repo_path: &std::path::Path) -> Option<Self> {
        // Try to open as a bare repository
        let repo = crate::git::get_repository(repo_path).ok()?;

        // Get HEAD reference - this works even in bare repos
        let (current_ref, ref_type, head_oid) = match repo.head() {
            Ok(head) => {
                let oid = head.target();
                if head.is_branch() {
                    let name = head.shorthand().unwrap_or("unknown").to_string();
                    (Some(name), RefType::Branch, oid)
                } else if head.is_tag() {
                    let name = head.shorthand().unwrap_or("unknown").to_string();
                    (Some(name), RefType::Tag, oid)
                } else {
                    (None, RefType::Detached, oid)
                }
            }
            Err(_) => (None, RefType::Detached, None),
        };

        // Get commit information
        let (commit, short_commit, last_commit_date) = if let Some(oid) = head_oid {
            let full_id = oid.to_string();
            let short_id = if full_id.len() >= 7 {
                full_id[..7].to_string()
            } else {
                full_id.clone()
            };

            // Try to get commit time
            let date = repo.find_commit(oid).ok()
                .map(|commit| {
                    let seconds = commit.time().seconds();
                    chrono::DateTime::from_timestamp(seconds, 0)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                });

            (Some(full_id), Some(short_id), date)
        } else {
            (None, None, None)
        };

        // For bare repos, we can't check if it's dirty (no working tree)
        // So we'll assume clean for display purposes
        let is_dirty = false;

        Some(GitInfo {
            current_ref,
            ref_type,
            commit,
            short_commit,
            is_dirty,
            last_commit_date,
        })
    }

    /// Check if this git info matches a branch filter
    pub fn matches_branch(&self, branch_filter: &str) -> bool {
        match (&self.ref_type, &self.current_ref) {
            (RefType::Branch, Some(current_ref)) => current_ref == branch_filter,
            _ => false,
        }
    }

    /// Check if this git info matches a tag filter
    pub fn matches_tag(&self, tag_filter: &str) -> bool {
        match (&self.ref_type, &self.current_ref) {
            (RefType::Tag, Some(current_ref)) => current_ref == tag_filter,
            _ => false,
        }
    }
}
