//! Inference-facing configuration types.
//!
//! Carved from the main crate's `config` module (Wave B, `hyprstream-inference`
//! extraction): the runtime/model/generation/training configuration surface the
//! inference engine and training pipeline own. The main crate re-exports these
//! from `crate::config` so existing paths keep resolving.
//!
//! AGPL-3.0-only. This crate is part of hyprstream's combined AGPL distribution.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::runtime::generation_metrics::GenerationQualityMetrics;

/// Model loading and identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to model file
    pub path: PathBuf,
    /// Model identifier (e.g., "qwen2-1.5b")
    pub name: String,
    /// Architecture type ("llama", "qwen", etc.)
    pub architecture: String,
    /// Expected parameter count
    pub parameters: Option<u64>,
    /// QUIC/WebTransport port for model service. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,
}


impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            name: String::new(),
            architecture: String::new(),
            parameters: None,
            quic_port: None,
        }
    }
}


/// Runtime execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Context window size
    pub context_length: usize,
    /// Maximum context length override for KV cache allocation.
    /// None = use model's max_position_embeddings (can be very large, e.g., 40K tokens)
    /// Some(n) = cap KV cache at n tokens (significantly reduces GPU memory)
    pub max_context: Option<u32>,
    /// KV cache quantization type (None, INT8, NF4, FP4).
    /// Reduces GPU memory by 50-75% at slight quality cost.
    #[serde(default)]
    pub kv_quant_type: hyprstream_rpc_std::model_client::KVQuantType,
    /// Batch processing size
    pub batch_size: usize,
    /// CPU threads (None = auto-detect)
    pub cpu_threads: Option<usize>,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// GPU device ID (None = auto-detect, typically device 0).
    ///
    /// Legacy single-GPU selector. For multi-GPU, prefer [`Self::devices`];
    /// this field remains the back-compat fallback when `devices` is empty.
    pub gpu_device_id: Option<usize>,
    /// Explicit set of GPU device indices for multi-GPU (#313, epic #310).
    ///
    /// Empty = unset → fall back to the single [`Self::gpu_device_id`] /
    /// `HYPRSTREAM_GPU_DEVICE` (existing single-GPU behavior is unchanged).
    /// Parsed from `HYPRSTREAM_GPU_DEVICES` (comma-separated, e.g. `0,1`).
    /// Resolution + validation lives in [`Self::resolve_device_indices`] and is
    /// consumed by `runtime::DevicePool`.
    #[serde(default)]
    pub devices: Vec<usize>,
    /// Fail fast when a *requested* GPU is unavailable instead of silently
    /// downgrading to CPU (#315, epic #310).
    ///
    /// A process told to run on GPU 3 that silently lands on CPU tanks a pipeline
    /// split, so strictness is the safe default for the multi-GPU path. This only
    /// affects the case where a GPU was *explicitly* requested (`use_gpu` with an
    /// explicit `gpu_device_id`/`devices`); pure auto-detect (`use_gpu` with no
    /// device requested) still falls back to CPU so the legacy single-GPU
    /// "use a GPU if there is one" behavior is unchanged.
    /// Defaults to `true`; override with `HYPRSTREAM_STRICT_DEVICE=0`.
    #[serde(default = "default_strict_device")]
    pub strict_device: bool,
    /// GPU layers to offload (None = auto)
    pub gpu_layers: Option<usize>,
    /// Use memory mapping for model files
    pub mmap: bool,
    /// KV cache size in MB
    pub kv_cache_size_mb: usize,
    /// Precision mode (BF16/FP16/FP32/FP8)
    pub precision_mode: Option<String>,
    // NEW: Concurrency and timeout settings
    pub max_concurrent_loads: usize,
    pub max_concurrent_generations: usize,
    pub default_generation_timeout_ms: u64,
    pub default_model_load_timeout_ms: u64,

    /// Continuous / in-flight batching (#329, epic #310). **Default: off.**
    ///
    /// When enabled, the inference scheduler groups concurrent decode steps of
    /// same-tenant-delta sequences into a single batched forward (Llama only).
    /// When off, each stream runs the unchanged batch=1 decode path. Override
    /// with `HYPRSTREAM_CONTINUOUS_BATCH` (truthy = on). Off by default while the
    /// scheduler wiring lands incrementally — the batched kernel is correctness-
    /// gated by `batched_ragged_decode_matches_serial`.
    #[serde(default = "default_continuous_batching")]
    pub continuous_batching: bool,
    /// Max sequences fused into one batched decode step when
    /// [`Self::continuous_batching`] is on (spike default 16). Tunable via
    /// `HYPRSTREAM_CONTINUOUS_BATCH_MAX`.
    #[serde(default = "default_continuous_batch_max")]
    pub continuous_batch_max: usize,
    /// FP8 GEMM via torch `_scaled_mm` for FP8 (e4m3) weight projections.
    /// **Default: off.**
    ///
    /// When enabled, `LinearProjection::apply` computes FP8-weight matmuls
    /// with `at::_scaled_mm` instead of the lazy BF16 dequant-then-matmul,
    /// picking the first recipe the device supports (torch 2.11, verified
    /// against release/2.11 ScaledBlas.cpp / RowwiseScaledMM.cu):
    ///
    ///  1. v2 blockwise (1x128 activation × 128x128 weight blocks, the
    ///     checkpoint's native scales): **NVIDIA Hopper (SM90) + cuBLASLt ≥
    ///     12.9 only** — `_check_deepseek_support` hard-errors on SM120,
    ///     SM100, SM89, and there is no CPU kernel.
    ///  2. v1 rowwise (per-token × per-output-channel scales): **SM90+,
    ///     including SM120/Blackwell** — cuBLASLt rowwise at cuBLAS ≥ 12.9,
    ///     otherwise the CUTLASS `f8f8bf16_rowwise` SM120 kernel. Requires a
    ///     load-time requantization of the weight to per-output-channel
    ///     scales (coarser than 128x128 blocks; +1x FP8 weight memory while
    ///     the flag is on).
    ///
    /// Each recipe latches off per device after its first kernel error and the
    /// runtime falls back to lazy dequant, so enabling this on unsupported
    /// hardware costs one warning per device per recipe, not a failure.
    ///
    /// Precedence note: like `mmap`, this field is currently **env-only in
    /// effect** — model construction has no `RuntimeConfig` plumbing, so the
    /// runtime reads `HYPRSTREAM_FP8_GEMM` directly (cached per process);
    /// setting the field programmatically does not reach the model path.
    #[serde(default = "default_fp8_gemm")]
    pub fp8_gemm: bool,


    /// Self-speculative decoding via the Qwen3.5 MTP draft head. **Default: off.**
    ///
    /// When enabled (and the loaded model is a dense Qwen3.5 checkpoint with an
    /// MTP head), each decode step drafts 1 token with the MTP head and verifies
    /// it in the next main-model forward: a greedy exact-match accept emits 2
    /// tokens, a reject emits the verifier's token and rewinds KV/SSM state.
    /// v1 restrictions: batch=1 streams only, greedy sampling only
    /// (temperature ≤ 0.01), no tenant LoRA delta, no quantized KV cache
    /// (`truncate_to` is a full clear for quantized storage, which would silently
    /// break the reject rewind), dense (non-MoE) MTP blocks only. Override with
    /// `HYPRSTREAM_SPECULATIVE_DECODE` (truthy = on). Correctness is gated by
    /// `mtp_decode_matches_serial`.
    ///
    /// # Performance envelope (measured)
    ///
    /// **Experimental — measured regression, keep default-off.** GPU validation
    /// (RTX 5090 / SM120, Qwen3.5-4B BF16) found the path functionally correct
    /// but 0.237x serial throughput (48.84% draft acceptance). The naive cost
    /// model — accept = 1 forward per 2 tokens, reject = 2 forwards per 1
    /// token, breakeven at α ≈ 0.5 acceptance with speedup ≈ (1+α)/(2−α) —
    /// holds only BEFORE structural per-round overheads:
    ///
    /// - a full GDN conv/rec SSM-state deep copy (`snapshot_ssm_states`) every
    ///   round — fixed cost per round, suspected dominant (profile this first
    ///   if this is ever optimized);
    /// - the reject-path single-token re-forward that re-syncs SSM state;
    /// - the per-round MTP draft forward itself.
    ///
    /// At 4B with k=1 these overheads dominate the verify savings; break-even
    /// needs the verification cost amortized (larger backbone) or a cheaper
    /// draft path. Enable only per workload after measuring acceptance via the
    /// `inference_speculative_tokens_total` counter (kind=accepted|rejected);
    /// if a workload shows <~60–70% acceptance this flag is a pessimization.
    #[serde(default = "default_speculative_decoding")]
    pub speculative_decoding: bool,
    /// Materialize FP8 weights as BF16 once at load time (applying the
    /// block-wise `_scale_inv` scales during load) and drop the FP8+scale
    /// tensors, instead of dequantizing inside every matmul on the hot path.
    /// Trades ~2x weight VRAM for zero per-matmul dequant work. Tunable via
    /// `HYPRSTREAM_FP8_DEQUANT_LOAD` (truthy = on). Off by default — off keeps
    /// the FP8-in-VRAM behavior, which is required when the BF16 equivalent
    /// would exceed VRAM. Ignored (with a warning) for multi-device pipelines:
    /// weights are loaded onto the pool's primary device before layers are
    /// distributed, so full-model BF16 materialization there could OOM.
    #[serde(default = "default_fp8_dequant_load")]
    pub fp8_dequant_load: bool,
}

/// Default for [`RuntimeConfig::continuous_batching`]: off unless
/// `HYPRSTREAM_CONTINUOUS_BATCH` is set truthy (#329). Off is the safe default —
/// the batch=1 path is the verified reference.
fn default_continuous_batching() -> bool {
    std::env::var("HYPRSTREAM_CONTINUOUS_BATCH")
        .map(|v| {
            matches!(
                v.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Default for [`RuntimeConfig::continuous_batch_max`] (#329): 16 (spike rec),
/// overridable via `HYPRSTREAM_CONTINUOUS_BATCH_MAX`.
fn default_continuous_batch_max() -> usize {
    std::env::var("HYPRSTREAM_CONTINUOUS_BATCH_MAX")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(16)
}

/// Default for [`RuntimeConfig::fp8_gemm`]: off unless `HYPRSTREAM_FP8_GEMM`
/// is set truthy. Off is the safe default — the lazy BF16 dequant matmul is
/// the verified reference, and the `_scaled_mm_v2` blockwise path requires
/// NVIDIA Hopper (SM90) + cuBLASLt ≥ 12.9 (no CPU/ROCm kernel in torch 2.11).
pub(crate) fn default_fp8_gemm() -> bool {
    std::env::var("HYPRSTREAM_FP8_GEMM")
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Default for [`RuntimeConfig::speculative_decoding`]: off unless
/// `HYPRSTREAM_SPECULATIVE_DECODE` is set truthy. Off is the safe default — the
/// serial batch=1 path is the verified reference; speculation is a v1 perf
/// experiment (greedy-only, batch=1-only, dense Qwen3.5 MTP only).
fn default_speculative_decoding() -> bool {
    std::env::var("HYPRSTREAM_SPECULATIVE_DECODE")
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Default for [`RuntimeConfig::fp8_dequant_load`]: off unless
/// `HYPRSTREAM_FP8_DEQUANT_LOAD` is set truthy. Off is the safe default — it
/// keeps FP8 weights at FP8 size in VRAM with lazy per-matmul dequantization.
pub(crate) fn default_fp8_dequant_load() -> bool {
    std::env::var("HYPRSTREAM_FP8_DEQUANT_LOAD")
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Default for [`RuntimeConfig::strict_device`]: strict (fail-fast) unless
/// `HYPRSTREAM_STRICT_DEVICE` is set to a falsy value. Strictness is the safe
/// default for the multi-GPU path (#315).
fn default_strict_device() -> bool {
    std::env::var("HYPRSTREAM_STRICT_DEVICE")
        .map(|v| {
            !matches!(
                v.trim().to_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        // Check environment variables for runtime configuration
        // Precedence: CLI args > env vars (read here) > hardcoded defaults.
        // Environment variables set initial defaults; CLI args may override them later.
        let gpu_device_id = std::env::var("HYPRSTREAM_GPU_DEVICE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        // `devices` is populated leniently here (Default cannot return errors);
        // the authoritative strict parse + validation lives in
        // `RuntimeConfig::resolve_device_indices`, which re-reads the env var and
        // turns parse errors into hard errors.
        let devices = std::env::var("HYPRSTREAM_GPU_DEVICES")
            .ok()
            .and_then(|s| Self::parse_device_list(&s).ok())
            .unwrap_or_default();

        let max_context = std::env::var("HYPRSTREAM_MAX_CONTEXT")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());

        let kv_quant_type = std::env::var("HYPRSTREAM_KV_QUANT")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "int8" => Some(hyprstream_rpc_std::model_client::KVQuantType::Int8),
                "nf4" => Some(hyprstream_rpc_std::model_client::KVQuantType::Nf4),
                "fp4" => Some(hyprstream_rpc_std::model_client::KVQuantType::Fp4),
                "none" | "" => Some(hyprstream_rpc_std::model_client::KVQuantType::None),
                _ => None,
            })
            .unwrap_or(hyprstream_rpc_std::model_client::KVQuantType::None);

        Self {
            context_length: 4096,
            max_context,
            kv_quant_type,
            batch_size: 512,
            cpu_threads: None,
            use_gpu: true,
            gpu_device_id, // From env or None (auto-detect device 0)
            devices,       // From HYPRSTREAM_GPU_DEVICES or empty (→ fall back to gpu_device_id)
            strict_device: default_strict_device(),
            gpu_layers: None,
            mmap: true,
            kv_cache_size_mb: 2048,
            precision_mode: Some("auto".to_owned()),
            max_concurrent_loads: 2,
            max_concurrent_generations: 10,
            default_generation_timeout_ms: 120000, // 2 minutes
            default_model_load_timeout_ms: 300000, // 5 minutes
            continuous_batching: default_continuous_batching(),
            continuous_batch_max: default_continuous_batch_max(),
            fp8_gemm: default_fp8_gemm(),
            speculative_decoding: default_speculative_decoding(),
            fp8_dequant_load: default_fp8_dequant_load(),
        }
    }
}

impl RuntimeConfig {
    /// Parse a comma-separated GPU device list (e.g. `"0,1"`), strictly.
    ///
    /// Whitespace around entries is trimmed. Any non-numeric entry, or a
    /// trailing/empty field (e.g. `"0,"` or `"0,,1"`), is a hard error — there
    /// is no silent default. Returns the parsed indices (possibly with
    /// duplicates; dedup/validation is the caller's job).
    fn parse_device_list(raw: &str) -> anyhow::Result<Vec<usize>> {
        raw.split(',')
            .map(|part| {
                let trimmed = part.trim();
                trimmed.parse::<usize>().map_err(|e| {
                    anyhow::anyhow!(
                        "invalid GPU device index {trimmed:?} in HYPRSTREAM_GPU_DEVICES={raw:?}: {e}"
                    )
                })
            })
            .collect()
    }

    /// Resolve the explicitly-requested *multi-GPU* device set, fail-fast.
    ///
    /// This is the seam the multi-GPU foundation uses to decide whether to engage
    /// the new `DevicePool` path. It considers **only** the explicit multi-GPU
    /// inputs and deliberately excludes the legacy single [`Self::gpu_device_id`]
    /// so that existing single-GPU behavior is left entirely on its old code
    /// path (#313 introduces the pool without changing single-GPU runtime
    /// behavior). Precedence:
    ///
    /// 1. `HYPRSTREAM_GPU_DEVICES` env var (re-parsed here strictly, so a
    ///    malformed value is a hard error rather than a silent fallback).
    /// 2. The [`Self::devices`] field (e.g. from a config file / CLI).
    ///
    /// Returns `Ok(None)` when no explicit multi-GPU set was requested,
    /// `Ok(Some(indices))` (non-empty, duplicate-free) otherwise, and `Err` on
    /// parse errors or duplicate indices.
    pub fn resolve_explicit_multi_device_indices(&self) -> anyhow::Result<Option<Vec<usize>>> {
        // (1) env var wins and is parsed strictly here. An absent or
        //     empty/whitespace-only value is treated as "unset" so resolution
        //     falls through to (2) the struct field (config file / CLI).
        let env_raw = std::env::var("HYPRSTREAM_GPU_DEVICES").ok();
        let env_trimmed = env_raw.as_deref().map(str::trim).filter(|s| !s.is_empty());
        let indices = match env_trimmed {
            Some(raw) => Self::parse_device_list(raw)?,
            None => self.devices.clone(),
        };

        Self::validate_index_set(indices)
    }

    /// Resolve the GPU device indices to use, fail-fast, including the legacy
    /// single-GPU fallback.
    ///
    /// This is the full-precedence resolver consumed by
    /// `runtime::DevicePool::from_config`. It is
    /// [`Self::resolve_explicit_multi_device_indices`] plus a final fallback to
    /// the legacy single [`Self::gpu_device_id`] (mapped to a one-element set),
    /// preserving back-compat for callers that build a pool directly from config.
    ///
    /// `Ok(None)` means nothing was requested (caller should auto-detect).
    pub fn resolve_device_indices(&self) -> anyhow::Result<Option<Vec<usize>>> {
        if let Some(indices) = self.resolve_explicit_multi_device_indices()? {
            return Ok(Some(indices));
        }
        // Legacy single-GPU selector as the final fallback.
        match self.gpu_device_id {
            Some(id) => Self::validate_index_set(vec![id]),
            None => Ok(None),
        }
    }

    /// Shared post-processing for a resolved index list: empty → `None`, reject
    /// duplicates, otherwise `Some(indices)`.
    fn validate_index_set(indices: Vec<usize>) -> anyhow::Result<Option<Vec<usize>>> {
        if indices.is_empty() {
            return Ok(None);
        }
        let mut seen = std::collections::HashSet::with_capacity(indices.len());
        for &idx in &indices {
            if !seen.insert(idx) {
                return Err(anyhow::anyhow!(
                    "duplicate GPU device index {idx} in requested set {indices:?}"
                ));
            }
        }
        Ok(Some(indices))
    }
}

/// Text generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0-2.0)
    pub temperature: f32,
    /// Nucleus sampling threshold
    pub top_p: f32,
    /// Top-k sampling limit
    pub top_k: Option<usize>,
    /// Repetition penalty
    pub repeat_penalty: f32,
    /// Stop sequences
    pub stop_tokens: Vec<String>,
    /// Random seed for reproducible generation
    pub seed: Option<u32>,
    /// Enable streaming output
    pub stream: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            repeat_penalty: 1.1,
            stop_tokens: vec!["</s>".to_owned(), "<|endoftext|>".to_owned()],
            seed: None,
            stream: false,
        }
    }
}


/// Generation result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: usize,
    pub finish_reason: FinishReason,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
    /// Quality metrics for self-supervised training
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_metrics: Option<GenerationQualityMetrics>,

    // Prefill metrics (processing the prompt)
    #[serde(default)]
    pub prefill_tokens: usize,
    #[serde(default)]
    pub prefill_time_ms: u64,
    #[serde(default)]
    pub prefill_tokens_per_sec: f32,

    // Inference metrics (generating new tokens, excluding prefill)
    #[serde(default)]
    pub inference_tokens: usize,
    #[serde(default)]
    pub inference_time_ms: u64,
    #[serde(default)]
    pub inference_tokens_per_sec: f32,

    /// Online training (TTT) adaptation metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttt_metrics: Option<TTTMetrics>,
}

/// TTT adaptation metrics (mirrors training::ttt::TTTResult)
///
/// Exposed as "Online Training" metrics in user-facing APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTTMetrics {
    pub avg_loss: f32,
    pub loss_improvement: f32,
    pub steps_performed: usize,
    pub adaptation_time_ms: u64,
    pub skipped: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,

    // Advanced metrics (expert recommendation)
    pub avg_grad_norm: f32,
    pub max_grad_norm: f32,
    pub gradient_clipped: bool,
    pub tokens_used: usize,
    pub tokens_provided: usize,
    pub was_truncated: bool,

    // Tenant-aware TTT metrics
    /// Initial perplexity before adaptation
    #[serde(default)]
    pub initial_perplexity: f32,
    /// Final perplexity after adaptation
    #[serde(default)]
    pub final_perplexity: f32,
    /// Server's recommendation: true = commit, false = rollback
    #[serde(default)]
    pub recommendation: bool,
    /// Number of steps determined by perplexity gating
    #[serde(default)]
    pub gated_steps: usize,
    /// Whether adaptation is pending client commit/rollback
    #[serde(default)]
    pub pending: bool,
}

impl From<crate::training::ttt::TTTResult> for TTTMetrics {
    fn from(r: crate::training::ttt::TTTResult) -> Self {
        Self {
            avg_loss: r.avg_loss,
            loss_improvement: r.loss_improvement,
            steps_performed: r.steps_performed,
            adaptation_time_ms: r.adaptation_time_ms,
            skipped: r.skipped,
            skip_reason: r.skip_reason,
            avg_grad_norm: r.avg_grad_norm,
            max_grad_norm: r.max_grad_norm,
            gradient_clipped: r.gradient_clipped,
            tokens_used: r.tokens_used,
            tokens_provided: r.tokens_provided,
            was_truncated: r.was_truncated,
            initial_perplexity: r.initial_perplexity,
            final_perplexity: r.final_perplexity,
            recommendation: r.recommendation,
            gated_steps: r.gated_steps,
            pending: r.pending,
        }
    }
}

/// Why generation stopped
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    MaxTokens,
    StopToken(String),
    EndOfSequence,
    Error(String),
    Stop,
}

// =============================================================================
// Training Mode Configuration (Phase D)
// =============================================================================

/// Model-level training mode configuration (embedded in config.json under "hyprstream_training")
///
/// This allows inference to automatically adapt models when enabled.
/// The training mode is set via `hyprstream training set test_time_training`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyprstreamTrainingConfig {
    /// Training mode: disabled, test_time_training, supervised
    #[serde(default)]
    pub mode: TrainingMode,

    /// Target adapter to train (e.g., "01_coding")
    pub target_adapter: Option<String>,

    /// Learning rate for training
    #[serde(default = "default_training_learning_rate")]
    pub learning_rate: f64,

    /// Batch size for training (used by supervised mode)
    #[serde(default = "default_training_batch_size")]
    pub batch_size: usize,

    /// Training steps per cycle (used by supervised mode)
    #[serde(default = "default_training_steps_per_cycle")]
    pub steps_per_cycle: usize,

    /// Minimum quality score to keep examples (0.0-1.0)
    #[serde(default = "default_training_min_quality")]
    pub min_quality_threshold: f32,

    /// Enable training on base model weights (vs LoRA only)
    #[serde(default)]
    pub train_base_model: bool,

    /// TTT-specific configuration (for TestTimeTraining mode)
    #[serde(default)]
    pub ttt: TTTTrainingConfig,

    /// LoRA rank for TTT delta (default: 8)
    #[serde(default = "default_lora_rank")]
    pub lora_rank: usize,

    /// LoRA alpha scaling factor (default: None, which means alpha = rank)
    #[serde(default)]
    pub lora_alpha: Option<f32>,

    /// Target modules for LoRA adaptation (default: ["q_proj", "v_proj"])
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,
}

/// TTT-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTTTrainingConfig {
    /// Learning rate for TTT adaptation (higher than fine-tuning)
    #[serde(default = "default_ttt_learning_rate")]
    pub learning_rate: f64,

    /// Number of gradient steps per input
    #[serde(default = "default_ttt_gradient_steps")]
    pub gradient_steps: u32,

    /// Maximum gradient norm for clipping
    #[serde(default = "default_ttt_max_grad_norm")]
    pub max_grad_norm: f64,

    /// Minimum input length (tokens) to trigger TTT
    #[serde(default = "default_ttt_min_input_length")]
    pub min_input_length: u32,

    /// Maximum input length to process for TTT
    #[serde(default = "default_ttt_max_context")]
    pub max_ttt_context: u32,

    /// Rank oracle configuration (optional — omit to disable runtime rank adaptation)
    #[serde(default)]
    pub rank_oracle: Option<crate::training::RankOracleConfig>,

    /// Per-layer gradient gating (optional — enabled by default)
    #[serde(default)]
    pub gradient_gating: Option<crate::training::GradientGatingConfig>,
}

fn default_ttt_learning_rate() -> f64 {
    3e-4
}
fn default_ttt_gradient_steps() -> u32 {
    3
}
fn default_ttt_max_grad_norm() -> f64 {
    1.0
}
fn default_ttt_min_input_length() -> u32 {
    32
}
fn default_ttt_max_context() -> u32 {
    512
}

impl Default for TTTTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_ttt_learning_rate(),
            gradient_steps: default_ttt_gradient_steps(),
            max_grad_norm: default_ttt_max_grad_norm(),
            min_input_length: default_ttt_min_input_length(),
            max_ttt_context: default_ttt_max_context(),
            rank_oracle: None,
            gradient_gating: None,
        }
    }
}

impl HyprstreamTrainingConfig {
    /// Check if training is enabled (mode != Disabled)
    pub fn is_enabled(&self) -> bool {
        self.mode != TrainingMode::Disabled
    }
}

impl Default for HyprstreamTrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::default(),
            target_adapter: None,
            learning_rate: default_training_learning_rate(),
            batch_size: default_training_batch_size(),
            steps_per_cycle: default_training_steps_per_cycle(),
            min_quality_threshold: default_training_min_quality(),
            train_base_model: false,
            ttt: TTTTrainingConfig::default(),
            lora_rank: default_lora_rank(),
            lora_alpha: None,
            target_modules: default_target_modules(),
        }
    }
}

/// Training mode configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    /// Training disabled (default)
    #[default]
    Disabled,
    /// Test-Time Training: adapts to input context before generation
    /// Research-valid approach based on TTT-E2E
    TestTimeTraining,
    /// Supervised training with explicit training data
    Supervised,
}

// Default functions for HyprstreamTrainingConfig
pub fn default_lora_rank() -> usize {
    8
}
pub fn default_target_modules() -> Vec<String> {
    vec!["q_proj".to_owned(), "v_proj".to_owned()]
}
fn default_training_learning_rate() -> f64 {
    1e-5
}
fn default_training_batch_size() -> usize {
    4
}
fn default_training_steps_per_cycle() -> usize {
    10
}
fn default_training_min_quality() -> f32 {
    0.3
}



/// Multi-GPU device resolution tests (#313) for the moved `RuntimeConfig`.
#[cfg(test)]
mod runtime_config_tests {
#![allow(clippy::unwrap_used)]

    use super::*;

    /// RAII guard to set/unset a process env var for the duration of a test
    /// and restore the previous value. Local copy of the helper that stays
    /// with the main crate's config tests; test scaffolding only.
    struct EnvVarGuard {
        key: String,
        prev: Option<String>,
    }
    impl EnvVarGuard {
        fn set(key: &str, val: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, val);
            Self { key: key.to_owned(), prev }
        }
        fn unset(key: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key: key.to_owned(), prev }
        }
    }
    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var(&self.key, v),
                None => std::env::remove_var(&self.key),
            }
        }
    }

#[test]
#[allow(clippy::unwrap_used)]
fn parse_device_list_basic() {

    assert_eq!(RuntimeConfig::parse_device_list("0,1").unwrap(), vec![0, 1]);

    // Whitespace around entries is tolerated.

    assert_eq!(

        RuntimeConfig::parse_device_list(" 0 , 2 ").unwrap(),

        vec![0, 2]

    );

    assert_eq!(RuntimeConfig::parse_device_list("3").unwrap(), vec![3]);

}

#[test]
fn parse_device_list_rejects_garbage() {

    // Non-numeric, empty fields, and negatives are hard errors (no silent default).

    assert!(RuntimeConfig::parse_device_list("0,foo").is_err());

    assert!(RuntimeConfig::parse_device_list("0,").is_err());

    assert!(RuntimeConfig::parse_device_list("0,,1").is_err());

    assert!(RuntimeConfig::parse_device_list("-1").is_err());

}

#[test]
#[allow(clippy::unwrap_used)]
fn validate_index_set_dedup_and_empty() {

    // Empty → None (auto-detect).

    assert_eq!(RuntimeConfig::validate_index_set(vec![]).unwrap(), None);

    // Duplicates → error.

    assert!(RuntimeConfig::validate_index_set(vec![0, 0]).is_err());

    // Distinct → Some, order preserved.

    assert_eq!(

        RuntimeConfig::validate_index_set(vec![2, 0, 1]).unwrap(),

        Some(vec![2, 0, 1])

    );

}

/// Serializes the two tests that mutate the shared `HYPRSTREAM_GPU_DEVICES`
/// process env var so they don't race under the parallel test runner.
static GPU_DEVICES_ENV_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

#[test]
#[allow(clippy::unwrap_used)]
fn resolve_uses_devices_field_and_legacy_fallback() {

    let _serial = GPU_DEVICES_ENV_LOCK.lock();

    // Guard against the env var leaking from the ambient environment so the

    // struct-field/legacy precedence is exercised deterministically.

    let _guard = EnvVarGuard::unset("HYPRSTREAM_GPU_DEVICES");

    // Explicit multi-device field wins.

    let mut cfg = RuntimeConfig::default();

    cfg.devices = vec![0, 1];

    cfg.gpu_device_id = Some(7);

    assert_eq!(

        cfg.resolve_explicit_multi_device_indices().unwrap(),

        Some(vec![0, 1])

    );

    assert_eq!(cfg.resolve_device_indices().unwrap(), Some(vec![0, 1]));

    // No explicit multi-device set → explicit resolver is None, but the full

    // resolver falls back to the legacy single gpu_device_id.

    let mut legacy = RuntimeConfig::default();

    legacy.devices = vec![];

    legacy.gpu_device_id = Some(3);

    assert_eq!(

        legacy.resolve_explicit_multi_device_indices().unwrap(),

        None

    );

    assert_eq!(legacy.resolve_device_indices().unwrap(), Some(vec![3]));

    // Nothing requested anywhere → None (auto-detect path preserved).

    let mut none = RuntimeConfig::default();

    none.devices = vec![];

    none.gpu_device_id = None;

    assert_eq!(none.resolve_device_indices().unwrap(), None);

}

#[test]
#[allow(clippy::unwrap_used)]
fn resolve_env_var_overrides_and_is_strict() {

    let _serial = GPU_DEVICES_ENV_LOCK.lock();

    let mut cfg = RuntimeConfig::default();

    cfg.devices = vec![5];

    cfg.gpu_device_id = Some(9);

    {

        let _g = EnvVarGuard::set("HYPRSTREAM_GPU_DEVICES", "0,1,2");

        assert_eq!(

            cfg.resolve_explicit_multi_device_indices().unwrap(),

            Some(vec![0, 1, 2]),

            "env var must override the devices field"

        );

    }

    {

        // Malformed env var is a hard error (not a silent fallback to field).

        let _g = EnvVarGuard::set("HYPRSTREAM_GPU_DEVICES", "0,nope");

        assert!(cfg.resolve_explicit_multi_device_indices().is_err());

    }

    {

        // Duplicate in env var → error.

        let _g = EnvVarGuard::set("HYPRSTREAM_GPU_DEVICES", "1,1");

        assert!(cfg.resolve_explicit_multi_device_indices().is_err());

    }

    {

        // Explicitly-empty env var is treated as unset (falls back to field).

        let _g = EnvVarGuard::set("HYPRSTREAM_GPU_DEVICES", "  ");

        assert_eq!(

            cfg.resolve_explicit_multi_device_indices().unwrap(),

            Some(vec![5])

        );

    }

}

/// Serializes tests mutating the shared `HYPRSTREAM_STRICT_DEVICE` env var.
static STRICT_DEVICE_ENV_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// #315: strict_device defaults to fail-fast, and can be opted out via env.
#[test]
fn strict_device_defaults_to_true_and_respects_env() {

    let _serial = STRICT_DEVICE_ENV_LOCK.lock();

    {

        let _g = EnvVarGuard::unset("HYPRSTREAM_STRICT_DEVICE");

        assert!(

            RuntimeConfig::default().strict_device,

            "strict_device must default to true (safe default for multi-GPU)"

        );

    }

    for falsy in ["0", "false", "no", "off"] {

        let _g = EnvVarGuard::set("HYPRSTREAM_STRICT_DEVICE", falsy);

        assert!(

            !RuntimeConfig::default().strict_device,

            "HYPRSTREAM_STRICT_DEVICE={falsy} must disable strict_device"

        );

    }

    {

        let _g = EnvVarGuard::set("HYPRSTREAM_STRICT_DEVICE", "1");

        assert!(RuntimeConfig::default().strict_device);

    }

}
}
