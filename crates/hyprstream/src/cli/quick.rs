//! Quick subcommand — orchestrated multi-step workflows.
//!
//! These are hand-coded CLI commands that involve multiple RPC calls,
//! interactive prompts, progress bars, local filesystem access, or
//! other complex orchestration that cannot be expressed as a single
//! schema-driven RPC call.

use clap::Subcommand;

use super::commands::policy::PolicyCommand;
use super::commands::training::TrainingCommand;
use super::commands::worker::WorkerAction;
use super::commands::KVQuantArg;

/// Quick workflows — multi-step convenience commands
#[derive(Subcommand)]
pub enum QuickCommand {
    /// Run inference with a model (streaming, stdin support)
    Infer {
        /// Model reference (e.g., "Qwen3-4B", "qwen/qwen-2b", "model:branch")
        model: String,

        /// Prompt text (reads from stdin if not provided)
        #[arg(short, long)]
        prompt: Option<String>,

        /// Image file path for multimodal models
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

        /// Random seed for deterministic generation
        #[arg(long)]
        seed: Option<u32>,

        /// Collect full response before printing (default: stream tokens live)
        #[arg(long)]
        sync: bool,
    },

    /// Clone a model repository (with progress, policy support)
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

        /// Apply a policy template to the cloned model
        #[arg(long)]
        policy: Option<String>,
    },

    /// List available models with capability detection
    List,

    /// Get detailed information about a model
    Info {
        /// Model name or reference
        model: String,
        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,
        /// Show only adapter information
        #[arg(long)]
        adapters_only: bool,
    },

    /// Show working tree status
    Status {
        /// Model reference
        model: Option<String>,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Create a new branch (with worktree + policy setup)
    Branch {
        /// Model name
        model: String,
        /// Branch name
        name: String,
        /// Create from specific ref
        #[arg(long)]
        from: Option<String>,
        /// Apply a policy template to the new branch
        #[arg(long)]
        policy: Option<String>,
    },

    /// Switch branches or checkout specific commit/tag
    Checkout {
        /// Model name or model reference
        model: String,
        /// Git reference to checkout
        git_ref: Option<String>,
        /// Create new branch if it doesn't exist
        #[arg(short = 'b')]
        create_branch: bool,
        /// Force checkout
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

    /// Remove a model from the system (interactive confirmation)
    Remove {
        /// Model name to remove
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

    /// Load a model into memory with runtime configuration
    Load {
        /// Model reference
        model: String,
        /// Maximum context length for KV cache allocation
        #[arg(long, env = "HYPRSTREAM_MAX_CONTEXT")]
        max_context: Option<usize>,
        /// KV cache quantization type
        #[arg(long, value_enum, default_value = "none", env = "HYPRSTREAM_KV_QUANT")]
        kv_quant: KVQuantArg,
    },

    /// Unload a model from memory
    Unload {
        /// Model reference
        model: String,
    },

    /// Worktree management
    Worktree {
        #[command(subcommand)]
        command: WorktreeQuickCommand,
    },

    /// Record changes to the repository
    #[cfg(feature = "experimental")]
    Commit {
        /// Model reference
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
        /// Override commit author
        #[arg(long)]
        author: Option<String>,
        /// Override author name
        #[arg(long)]
        author_name: Option<String>,
        /// Override author email
        #[arg(long)]
        author_email: Option<String>,
        /// Allow empty commits
        #[arg(long)]
        allow_empty: bool,
        /// Show what would be committed without committing
        #[arg(long)]
        dry_run: bool,
        /// Show diff of changes being committed
        #[arg(short = 'v', long)]
        verbose: bool,
    },

    /// Push changes to remote
    #[cfg(feature = "experimental")]
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

    /// Merge branches
    #[cfg(feature = "experimental")]
    Merge {
        /// Target model and branch
        target: String,
        /// Branch/ref to merge into target
        source: String,
        #[arg(long, conflicts_with = "no_ff", conflicts_with = "ff_only")]
        ff: bool,
        #[arg(long, conflicts_with = "ff", conflicts_with = "ff_only")]
        no_ff: bool,
        #[arg(long, conflicts_with = "ff", conflicts_with = "no_ff")]
        ff_only: bool,
        #[arg(long)]
        no_commit: bool,
        #[arg(long)]
        squash: bool,
        #[arg(short, long)]
        message: Option<String>,
        #[arg(long, conflicts_with_all = ["continue_merge", "quit"])]
        abort: bool,
        #[arg(long = "continue", conflicts_with_all = ["abort", "quit"])]
        continue_merge: bool,
        #[arg(long, conflicts_with_all = ["abort", "continue_merge"])]
        quit: bool,
        #[arg(long)]
        no_stat: bool,
        #[arg(short = 'q', long)]
        quiet: bool,
        #[arg(short = 'v', long)]
        verbose: bool,
        #[arg(long, short = 's')]
        strategy: Option<String>,
        #[arg(long, short = 'X')]
        strategy_option: Vec<String>,
        #[arg(long)]
        allow_unrelated_histories: bool,
        #[arg(long)]
        no_verify: bool,
    },

    /// Worker (sandbox/container) management commands
    ///
    /// Multi-step orchestrated workflows for Kata VMs and OCI containers.
    /// Commands involve multiple RPC calls, service bootstrapping, and
    /// interactive terminal I/O.
    Worker {
        #[command(subcommand)]
        command: WorkerAction,
    },

    /// Training workflows (multi-step, GPU, streaming)
    Training(TrainingCommand),

    /// Policy management commands (interactive, multi-step)
    Policy {
        #[command(subcommand)]
        command: PolicyCommand,
    },

    /// Remote management commands
    Remote {
        #[command(subcommand)]
        command: RemoteQuickCommand,
    },
}

/// Worktree subcommands for quick workflows
#[derive(Subcommand)]
pub enum WorktreeQuickCommand {
    /// Create a worktree from an existing branch
    Add {
        /// Model name
        model: String,
        /// Branch name
        branch: String,
        /// Apply a policy template
        #[arg(long)]
        policy: Option<String>,
    },

    /// List all worktrees for a model
    List {
        /// Model name
        model: String,
        /// Show all branches
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

/// Remote management subcommands for quick workflows
#[derive(Subcommand)]
pub enum RemoteQuickCommand {
    /// Add a new remote
    Add {
        /// Model name
        model: String,
        /// Remote name
        name: String,
        /// Remote URL
        url: String,
    },

    /// List all remotes
    List {
        /// Model name
        model: String,
        /// Show verbose output
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
