//! Training CLI commands
//!
//! Provides training-specific commands that modify adapter weights:
//! - `training init` - Initialize adapter for training
//! - `training infer` - Inference with TTT (dirty writes)
//! - `training batch` - Batch training with checkpoints
//! - `training checkpoint` - Commit dirty adapter changes

use clap::{Args, Subcommand};
use std::path::PathBuf;

use super::KVQuantArg;

/// Training management commands
#[derive(Args)]
pub struct TrainingCommand {
    #[command(subcommand)]
    pub action: TrainingAction,
}

/// Training actions
#[derive(Subcommand)]
pub enum TrainingAction {
    /// Initialize adapter for training
    ///
    /// Creates a new LoRA adapter and configures the model for TTT training.
    /// Optionally creates a new branch/worktree for isolated training.
    Init {
        /// Model reference (e.g., "qwen3-small", "qwen3-small:main")
        model: String,

        /// Create new branch for isolated training
        #[arg(long, short = 'B')]
        branch: Option<String>,

        /// Adapter name (will be auto-prefixed with index)
        #[arg(long)]
        adapter: Option<String>,

        /// Explicit index for adapter (auto-increments if not specified)
        #[arg(long)]
        index: Option<u32>,

        /// LoRA rank (default: 16)
        #[arg(long, short = 'r', default_value = "16")]
        rank: u32,

        /// LoRA alpha (default: 32)
        #[arg(long, default_value = "32")]
        alpha: u32,

        /// Training mode: ttt (default), supervised
        #[arg(long, default_value = "ttt")]
        mode: String,

        /// Learning rate
        #[arg(long, short = 'l', default_value = "0.0003")]
        learning_rate: f32,
    },

    /// Inference with TTT (dirty writes to adapters)
    ///
    /// Runs inference while applying Test-Time Training. This modifies
    /// the adapter weights during inference. Use `training checkpoint`
    /// to commit changes when ready.
    Infer {
        /// Model reference (e.g., "qwen3-small:experiment")
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

        /// Collect full response before printing (default: stream tokens live)
        #[arg(long)]
        sync: bool,

        /// Maximum context length for KV cache allocation
        #[arg(long, env = "HYPRSTREAM_MAX_CONTEXT")]
        max_context: Option<usize>,

        /// KV cache quantization type
        #[arg(long, value_enum, default_value = "none", env = "HYPRSTREAM_KV_QUANT")]
        kv_quant: KVQuantArg,
    },

    /// Batch training with checkpoints
    ///
    /// Loads model once and processes all input files, applying TTT
    /// to each. Auto-checkpoints at specified intervals.
    Batch {
        /// Model reference (e.g., "qwen3-small:experiment")
        model: String,

        /// Input file(s) - can specify multiple times
        #[arg(short, long)]
        input: Vec<String>,

        /// Input directory to scan for files
        #[arg(long)]
        input_dir: Option<PathBuf>,

        /// File pattern for --input-dir (default: "*.txt")
        #[arg(long, default_value = "*.txt")]
        pattern: String,

        /// Input format: "text" (raw files) or "jsonl" (structured)
        #[arg(long, default_value = "text")]
        format: String,

        /// Max tokens to generate per file (for text format)
        #[arg(long, default_value = "100")]
        max_tokens: usize,

        /// Characters to read from each file
        #[arg(long, default_value = "4000")]
        chunk_size: usize,

        /// Skip first N files (for resume)
        #[arg(long, default_value = "0")]
        skip: usize,

        /// Process only N files
        #[arg(long)]
        limit: Option<usize>,

        /// Show progress every N files
        #[arg(long, default_value = "10")]
        progress_interval: usize,

        /// Checkpoint (commit) every N files
        #[arg(long, default_value = "100")]
        checkpoint_interval: usize,

        /// Test set files for validation (glob pattern)
        #[arg(long)]
        test_set: Option<String>,

        /// Maximum context length for KV cache allocation
        #[arg(long, env = "HYPRSTREAM_MAX_CONTEXT")]
        max_context: Option<usize>,

        /// KV cache quantization type
        #[arg(long, value_enum, default_value = "none", env = "HYPRSTREAM_KV_QUANT")]
        kv_quant: KVQuantArg,
    },

    /// Commit dirty adapter changes to git
    ///
    /// Stages and commits modified adapter safetensor files.
    /// Use after `training infer` or `training batch` to persist changes.
    Checkpoint {
        /// Model reference (e.g., "qwen3-small:experiment")
        model: String,

        /// Commit message (auto-generated if not provided)
        #[arg(short, long)]
        message: Option<String>,

        /// Push to remote after commit
        #[arg(long)]
        push: bool,

        /// Remote name for push (default: origin)
        #[arg(long, default_value = "origin")]
        remote: String,
    },
}
