pub mod chat;
pub mod config;
pub mod flight;
pub mod git;
pub mod policy;
pub mod server;
pub mod training;
pub mod worker;

pub use flight::FlightArgs;
pub use git::{GitAction, GitCommand};
pub use policy::{PolicyCommand, TokenCommand};
pub use server::{ServerCliArgs, ServerCommand};
pub use training::{TrainingAction, TrainingCommand};
pub use worker::{ImageCommand, WorkerAction, WorkerCommand};

use clap::{Subcommand, ValueEnum};

use crate::runtime::kv_quant::KVQuantType;

/// KV cache quantization type for inference
///
/// Quantization reduces GPU memory usage at a slight quality cost:
/// - `none`: Full precision (default)
/// - `int8`: 50% memory savings, minimal quality loss
/// - `nf4`: 75% memory savings, best quality for 4-bit
/// - `fp4`: 75% memory savings, standard 4-bit quantization
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KVQuantArg {
    /// No quantization (full precision FP16/BF16)
    #[default]
    None,
    /// 8-bit integer quantization (~50% memory savings)
    Int8,
    /// 4-bit NormalFloat quantization (~75% memory savings)
    Nf4,
    /// 4-bit FloatingPoint quantization (~75% memory savings)
    Fp4,
}

impl From<KVQuantArg> for KVQuantType {
    fn from(arg: KVQuantArg) -> Self {
        match arg {
            KVQuantArg::None => KVQuantType::None,
            KVQuantArg::Int8 => KVQuantType::Int8,
            KVQuantArg::Nf4 => KVQuantType::Nf4,
            KVQuantArg::Fp4 => KVQuantType::Fp4,
        }
    }
}

/// Overall execution mode for the hyprstream CLI and services
///
/// This determines how services are spawned and managed:
/// - **Inproc**: Single process, all services in-process, inproc:// ZMQ transport
/// - **IpcStandalone**: Multiple processes spawned directly, ipc:// ZMQ transport
/// - **IpcSystemd**: Multiple systemd services with socket activation, ipc:// ZMQ transport
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionMode {
    /// Standalone mode with in-process services (default for development)
    #[default]
    Inproc,

    /// Standalone mode with forked processes (no systemd)
    /// Services spawned via ProcessSpawner with StandaloneBackend
    IpcStandalone,

    /// Systemd mode with socket-activated services
    /// Services managed by systemd, started on-demand via socket activation
    IpcSystemd,
}

impl ExecutionMode {
    /// Detect best execution mode based on system capabilities
    pub fn detect() -> Self {
        #[cfg(feature = "systemd")]
        {
            if hyprstream_rpc::has_systemd() {
                return ExecutionMode::IpcSystemd;
            }
        }
        ExecutionMode::Inproc
    }

    /// Get the EndpointMode for this execution mode
    pub fn endpoint_mode(&self) -> hyprstream_rpc::registry::EndpointMode {
        match self {
            ExecutionMode::Inproc => hyprstream_rpc::registry::EndpointMode::Inproc,
            ExecutionMode::IpcStandalone | ExecutionMode::IpcSystemd => {
                hyprstream_rpc::registry::EndpointMode::Ipc
            }
        }
    }

    /// Whether this mode uses IPC sockets
    pub fn uses_ipc(&self) -> bool {
        matches!(self, ExecutionMode::IpcStandalone | ExecutionMode::IpcSystemd)
    }

    /// Whether this mode uses systemd
    pub fn uses_systemd(&self) -> bool {
        matches!(self, ExecutionMode::IpcSystemd)
    }
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionMode::Inproc => write!(f, "inproc"),
            ExecutionMode::IpcStandalone => write!(f, "ipc-standalone"),
            ExecutionMode::IpcSystemd => write!(f, "ipc-systemd"),
        }
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the Hyprstream server
    Server(ServerCommand),

    /// Flight SQL client to query datasets
    Flight(FlightArgs),

    // Phase 1: Git-style commands at top level
    /// Create a new branch
    Branch {
        /// Model name
        model: String,
        /// Branch name
        name: String,
        /// Create from specific ref
        #[arg(long)]
        from: Option<String>,
        /// Apply a policy template to the new branch (local, public-inference, public-read)
        #[arg(long)]
        policy: Option<String>,
    },

    /// Switch branches or checkout specific commit/tag
    Checkout {
        /// Model name or model reference (e.g., "llama3" or "llama3:main")
        model: String,
        /// Git reference (branch/tag/commit) to checkout (optional)
        git_ref: Option<String>,
        /// Create new branch if it doesn't exist
        #[arg(short = 'b')]
        create_branch: bool,
        /// Force checkout
        #[arg(long)]
        force: bool,
    },

    /// Show working tree status
    Status {
        /// Model reference (e.g., "llama3", "llama3:main", "llama3:v2.0")
        model: Option<String>,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Record changes to the repository
    Commit {
        /// Model reference (e.g., "model:branch")
        model: String,

        /// Commit message
        #[arg(short, long)]
        message: String,

        /// Stage all tracked files
        #[arg(short = 'a', long)]
        all: bool,

        /// Stage all files including untracked
        #[arg(short = 'A', long)]
        all_untracked: bool,

        /// Amend the previous commit
        #[arg(long)]
        amend: bool,

        /// Override commit author (format: "Name <email>")
        #[arg(long)]
        author: Option<String>,

        /// Override author name (use with --author-email)
        #[arg(long)]
        author_name: Option<String>,

        /// Override author email (use with --author-name)
        #[arg(long)]
        author_email: Option<String>,

        /// Allow empty commits (no changes)
        #[arg(long)]
        allow_empty: bool,

        /// Show what would be committed without committing
        #[arg(long)]
        dry_run: bool,

        /// Show diff of changes being committed
        #[arg(short = 'v', long)]
        verbose: bool,
    },

    /// Training management commands
    ///
    /// Commands for TTT (Test-Time Training) and LoRA adapter management:
    /// - `training init` - Initialize adapter for training
    /// - `training infer` - Inference with TTT (dirty writes to adapters)
    /// - `training batch` - Batch training with checkpoints
    /// - `training checkpoint` - Commit dirty adapter changes
    Training(TrainingCommand),

    /// Run inference with a model (read-only, no training)
    Infer {
        /// Model reference (e.g., "Qwen3-4B", "qwen/qwen-2b", "model:branch")
        model: String,

        /// Prompt text (reads from stdin if not provided)
        #[arg(short, long)]
        prompt: Option<String>,

        /// Image file path for multimodal models (e.g., --image /path/to/image.jpg)
        #[arg(short = 'i', long)]
        image: Option<String>,

        /// Maximum tokens to generate
        #[arg(short = 'm', long)]
        max_tokens: Option<usize>,

        /// Temperature for sampling (0.0 = deterministic)
        #[arg(short = 't', long)]
        temperature: Option<f32>,

        /// Top-p (nucleus) sampling
        #[arg(long)]
        top_p: Option<f32>,

        /// Top-k sampling
        #[arg(long)]
        top_k: Option<usize>,

        /// Repetition penalty (1.0 = no penalty, >1.0 = penalize)
        #[arg(short = 'r', long)]
        repeat_penalty: Option<f32>,

        /// Random seed for deterministic generation (for debugging)
        #[arg(long)]
        seed: Option<u32>,

        /// Stream output tokens as they're generated
        #[arg(short = 's', long)]
        stream: bool,

        /// Force re-download even if cached
        #[arg(long)]
        force_download: bool,

        /// Maximum context length for KV cache allocation.
        /// Overrides model's max_position_embeddings to reduce GPU memory.
        #[arg(long, env = "HYPRSTREAM_MAX_CONTEXT")]
        max_context: Option<usize>,

        /// KV cache quantization type for reduced GPU memory usage.
        /// Reduces GPU memory by 50-75% at slight quality cost.
        #[arg(long, value_enum, default_value = "none", env = "HYPRSTREAM_KV_QUANT")]
        kv_quant: KVQuantArg,
    },

    /// List available models
    List,

    /// Get detailed information about a model
    Inspect {
        /// Model name or reference
        model: String,
        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,
        /// Show only adapter information
        #[arg(long)]
        adapters_only: bool,
    },

    /// Clone a model repository
    Clone {
        /// Git repository URL
        repo_url: String,

        /// Local name for the model
        #[arg(long)]
        name: Option<String>,

        /// Branch, tag, or commit to clone
        #[arg(long, short = 'b')]
        branch: Option<String>,

        /// Clone depth (number of commits). Default is 1 for shallow clone.
        /// Use 0 or --full for complete history
        #[arg(long, default_value = "1")]
        depth: u32,

        /// Clone with full history (overrides --depth)
        #[arg(long)]
        full: bool,

        /// Suppress progress output
        #[arg(long, short = 'q')]
        quiet: bool,

        /// Verbose output
        #[arg(long, short = 'v')]
        verbose: bool,

        /// Apply a policy template to the cloned model (local, public-inference, public-read)
        #[arg(long)]
        policy: Option<String>,
    },

    /// Push changes to remote
    Push {
        /// Model name
        model: String,
        /// Remote name (default: origin)
        remote: Option<String>,
        /// Branch to push
        branch: Option<String>,
        /// Set upstream
        #[arg(short = 'u', long)]
        set_upstream: bool,
        /// Force push
        #[arg(long)]
        force: bool,
    },

    /// Pull changes from remote
    Pull {
        /// Model name
        model: String,
        /// Remote name (default: origin)
        remote: Option<String>,
        /// Branch to pull
        branch: Option<String>,
        /// Rebase instead of merge
        #[arg(long)]
        rebase: bool,
    },

    /// Merge branches
    Merge {
        /// Target model and branch (e.g., "model:branch")
        target: String,

        /// Branch/ref to merge into target
        source: String,

        // Fast-forward control
        /// Fast-forward if possible (default)
        #[arg(long, conflicts_with = "no_ff", conflicts_with = "ff_only")]
        ff: bool,

        /// Always create merge commit
        #[arg(long, conflicts_with = "ff", conflicts_with = "ff_only")]
        no_ff: bool,

        /// Only allow fast-forward merges
        #[arg(long, conflicts_with = "ff", conflicts_with = "no_ff")]
        ff_only: bool,

        // Commit control
        /// Stage changes but don't commit
        #[arg(long)]
        no_commit: bool,

        /// Squash commits into single commit
        #[arg(long)]
        squash: bool,

        // Message control
        /// Custom merge commit message
        #[arg(short, long)]
        message: Option<String>,

        // Conflict handling
        /// Abort merge and restore pre-merge state
        #[arg(long, conflicts_with_all = ["continue_merge", "quit"])]
        abort: bool,

        /// Continue merge after resolving conflicts
        #[arg(long = "continue", conflicts_with_all = ["abort", "quit"])]
        continue_merge: bool,

        /// Quit merge but keep changes
        #[arg(long, conflicts_with_all = ["abort", "continue_merge"])]
        quit: bool,

        // Output control
        /// Suppress diffstat
        #[arg(long)]
        no_stat: bool,

        /// Suppress output except errors
        #[arg(short = 'q', long)]
        quiet: bool,

        /// Verbose output
        #[arg(short = 'v', long)]
        verbose: bool,

        // Advanced
        /// Merge strategy (resolve, recursive, ours, theirs, subtree)
        #[arg(long, short = 's')]
        strategy: Option<String>,

        /// Strategy-specific options (can be specified multiple times)
        #[arg(long, short = 'X')]
        strategy_option: Vec<String>,

        /// Allow merging unrelated histories
        #[arg(long)]
        allow_unrelated_histories: bool,

        /// Skip pre-merge hooks
        #[arg(long)]
        no_verify: bool,
    },

    /// Remove a model from the system
    Remove {
        /// Model name to remove (e.g., "Qwen3-4B")
        model: String,
        /// Force removal without confirmation
        #[arg(short, long)]
        force: bool,
        /// Remove only from registry, keep files
        #[arg(long)]
        registry_only: bool,
        /// Remove files only, keep registry entry
        #[arg(long)]
        files_only: bool,
    },

    /// Worktree management commands
    Worktree {
        #[command(subcommand)]
        command: WorktreeCommand,
    },

    /// Policy management commands for RBAC/ABAC access control
    Policy {
        #[command(subcommand)]
        command: PolicyCommand,
    },

    /// Remote management commands
    Remote {
        #[command(subcommand)]
        command: RemoteCommand,
    },

    /// Worker (sandbox/container) management commands
    ///
    /// Commands for managing Kata VMs (sandboxes) and OCI containers:
    /// - `worker list` - List sandboxes and containers
    /// - `worker run` - Run a container in a sandbox
    /// - `worker stop` - Stop sandbox or container
    /// - `worker rm` - Remove sandbox or container
    /// - `worker images` - Image management
    Worker(WorkerCommand),

    /// Run a single service (for systemd socket activation or callback mode)
    ///
    /// This command is used internally by systemd to run individual services.
    /// Systemd socket activation spawns these when a connection arrives.
    ///
    /// Examples:
    ///   hyprstream service event --ipc    # Run event service with IPC sockets
    ///   hyprstream service worker --ipc   # Run worker service with IPC sockets
    ///   hyprstream service registry --ipc # Run registry service with IPC sockets
    ///   hyprstream service inference@abc123 --callback ipc:///run/hyprstream/callback.sock
    Service {
        /// Service name: event, worker, registry, policy, or inference@{id} for callback mode
        name: String,

        /// Use IPC sockets for distributed mode (required for systemd socket activation)
        #[arg(long, default_value = "false")]
        ipc: bool,

        /// Callback endpoint for callback mode (inference service only)
        ///
        /// When specified, the inference service runs in callback mode:
        /// 1. Connects DEALER to this ROUTER endpoint
        /// 2. Sends Register message with its stream endpoint
        /// 3. Waits for LoadModel command
        /// 4. Handles Infer/Shutdown commands
        #[arg(long)]
        callback: Option<String>,
    },
}

/// Worktree subcommands
#[derive(Subcommand)]
pub enum WorktreeCommand {
    /// Create a worktree from an existing branch
    Add {
        /// Model name
        model: String,
        /// Branch name (must exist in bare repo)
        branch: String,
        /// Apply a policy template to the worktree (local, public-inference, public-read)
        #[arg(long)]
        policy: Option<String>,
    },

    /// List all worktrees for a model
    List {
        /// Model name
        model: String,
        /// Show all branches (including those without worktrees)
        #[arg(long)]
        all: bool,
    },

    /// Show detailed information about a worktree
    Info {
        /// Model name
        model: String,
        /// Branch/worktree name
        branch: String,
    },

    /// Remove a worktree
    Remove {
        /// Model name
        model: String,
        /// Branch/worktree name
        branch: String,
        /// Force removal without confirmation
        #[arg(short, long)]
        force: bool,
    },
}

/// Remote subcommands
#[derive(Subcommand)]
pub enum RemoteCommand {
    /// Add a new remote
    Add {
        /// Model name
        model: String,
        /// Remote name (e.g., "private", "upstream")
        name: String,
        /// Remote URL (e.g., git@github.com:user/repo.git)
        url: String,
    },

    /// List all remotes
    List {
        /// Model name
        model: String,
        /// Show verbose output (fetch/push URLs)
        #[arg(short, long)]
        verbose: bool,
    },

    /// Remove a remote
    Remove {
        /// Model name
        model: String,
        /// Remote name to remove
        name: String,
    },

    /// Change a remote's URL
    SetUrl {
        /// Model name
        model: String,
        /// Remote name
        name: String,
        /// New URL
        url: String,
    },

    /// Rename a remote
    Rename {
        /// Model name
        model: String,
        /// Current remote name
        old_name: String,
        /// New remote name
        new_name: String,
    },
}
