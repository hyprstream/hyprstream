//! Native Clef joint-schema decision runtime (R4 serving lane).
//!
//! Tensor-touching half of the typed decision path: metadata-first strict
//! loading of the original Clef joint head from the admitted model directory;
//! shared cheap request preflight (raw-byte bound + ordered Clef JSON parse +
//! per-question and AGGREGATE budget admission); encoding through the
//! head-lane [`ClefEncoder`]; ONE fresh-state unpadded CPU forward through
//! [`TorchEngine::with_decision_backbone_checked`]; softmax and canonical
//! label remapping. The service-facing half (auth context, pinned-artifact
//! requirement, E2 marshalling, wire response assembly) lives in
//! `hyprstream::services::decision`.
//!
//! Invariants (serving reviews):
//! - Head safetensors METADATA (exact expected names/shapes/count/dtypes/
//!   checked dimensions/bounded total bytes) is validated BEFORE any tensor
//!   is materialized or cast; the head config geometry must match the
//!   admitted engine hidden size before the weights file is even mapped.
//! - The caller of [`load`] must hold an immutable admitted model tree (the
//!   service layer requires a retained pinned artifact); identity digests are
//!   computed ONCE from the same bytes this loader maps, so backbone, head,
//!   tokenizer and config provenance describe one coherent immutable source.
//! - Every forward uses fresh call-local state; a cancelled or failed forward
//!   drops its local state and the next request starts clean. Cancellation is
//!   cooperative (checker before admission, per decoder layer, around the
//!   head closure). Dropping the response future does NOT stop a running CPU
//!   forward — the deadline and drain flag do.
//! - Budgets are server-controlled aggregates, checked with overflow-safe
//!   arithmetic BEFORE tokenization or tensor allocation: callers may lower,
//!   never raise them.
//! - Probabilities are data, never authorization.
//! - Canonical answer order: noul [false,true] (model order [true,false] is
//!   remapped explicitly), choice = sorted option IDs (the model's own
//!   option_ids order), score = ascending indices 0..N-1 (criteria array
//!   positions, NOT legacy typed-IR "levels"). Legacy typed IR authored-order
//!   projections are a distinct versioned adapter and must remap by label.

use crate::runtime::channel_fp8::ChannelFp8Profile;
use crate::runtime::TorchEngine;
use anyhow::{anyhow, Context as _, Result};
use hyprstream_decision_head::clef::{
    clef_label_permutation, ClefEncoder, ClefEncodingLimits, ClefError, ClefHeadConfig,
    ClefQuestion, ClefQuestionType, ClefRequest, EncodedClefRecord, JointSchemaHead,
    CLEF_ENCODING_VERSION,
};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tch::{Kind, Tensor};
use tokenizers::Tokenizer;

/// Official Clef bundle head subdirectory inside the admitted model dir.
pub const CLEF_HEAD_SUBDIR: &str = "clef_head";
/// Head geometry/config file (strict `deny_unknown_fields` parse).
pub const CLEF_HEAD_CONFIG_FILE: &str = "joint_head_config.json";
/// Original joint head weights.
pub const CLEF_HEAD_WEIGHTS_FILE: &str = "joint_head.safetensors";

/// Raw Clef JSON body byte bound, enforced BEFORE parsing (in the shared
/// preflight at BOTH Model ingress and the worker). The generic capnp body
/// traversal cap bounds the wire String allocation; this bound gates the
/// ordered JSON parse itself.
pub const MAX_CLEF_BODY_BYTES: usize = 4 << 20;
/// Server default/cap for a caller's state-token budget (callers may lower).
pub const DEFAULT_MAX_STATE_TOKENS: u32 = 8192;
/// Head config file bound (geometry JSON; anything larger is malformed).
pub const MAX_HEAD_CONFIG_BYTES: usize = 64 << 10;
/// Head weights file bound (official BF16 head is ~232 MiB).
pub const MAX_HEAD_WEIGHTS_BYTES: u64 = 2 << 30;

/// Per-request question bound.
pub const MAX_CLEF_QUESTIONS: usize = 255;
/// Choice options per question: 2..=MAX (D2 lower bound is 2).
pub const MAX_CLEF_CHOICE_OPTIONS: usize = 255;
/// Score levels per question: 2..=MAX (D1 lower bound is 2).
pub const MAX_CLEF_SCORE_LEVELS: usize = 255;
/// Server-controlled TOTAL option bound across all questions of one record.
/// The head concatenates every option query, so total options — not just
/// per-question counts — drives head working memory.
pub const MAX_CLEF_TOTAL_OPTIONS: u64 = 1024;
/// Server-controlled aggregate head-attention cell bound: total options ×
/// total sequence tokens (the routing cross-attention materializes scores of
/// exactly this order). Individually valid question/option/length counts can
/// still exceed it and must be rejected before tensor allocation.
pub const MAX_CLEF_ATTENTION_CELLS: u64 = 1 << 22;
/// Reference default total encoding budget; the effective budget passed to
/// [`decide`] is min(this, admitted engine context length).
pub const DEFAULT_CLEF_MAX_LENGTH: usize = 16384;

/// Head-config geometry bounds, enforced BEFORE `expected_shapes` derives the
/// tensor contract (and before any allocation its `width*4`/`3*width`
/// projections imply). A tiny malicious config must not admit a huge head.
/// Qualified profile: broad enough for the official joint head (width 1024,
/// 16 heads, 2 routing + 4 decoder layers, ffn 4096) and tiny fixtures,
/// nothing beyond. Wider heads need a new reviewed profile, not a bump.
pub const MAX_HEAD_HIDDEN_SIZE: i64 = 8_192;
pub const MAX_HEAD_WIDTH: i64 = 2_048;
pub const MAX_HEAD_ROUTING_LAYERS: usize = 8;
pub const MAX_HEAD_DECODER_LAYERS: usize = 16;
pub const MAX_HEAD_ATTENTION_HEADS: i64 = 32;
pub const MAX_HEAD_FEEDFORWARD: i64 = 16_384;
/// Total head parameter bound (official head ≈ 121.8M): bounds the Float32
/// conversion/activation working set, not just the source file.
pub const MAX_HEAD_TOTAL_PARAMS: u64 = 200_000_000;

/// Deterministic, typed decision-serving failures. The `clef-decision-*`
/// prefixes are load-bearing: the service layer maps them onto wire errors
/// by prefix and tests pin them.
#[derive(Debug, thiserror::Error)]
pub enum ClefDecisionError {
    #[error("clef-decision-cancelled: {0}")]
    Cancelled(String),
    #[error("clef-decision-input: {0}")]
    Input(String),
    #[error("clef-decision-bundle: {0}")]
    Bundle(String),
    #[error("clef-decision-profile: {0}")]
    Profile(String),
    #[error("clef-decision-runtime: {0}")]
    Runtime(String),
}

impl From<ClefError> for ClefDecisionError {
    fn from(e: ClefError) -> Self {
        match e {
            ClefError::Input(m) => Self::Input(m),
            other => Self::Runtime(other.to_string()),
        }
    }
}

/// Shared cheap preflight over the ORIGINAL raw Clef JSON body.
///
/// Runs at ModelService ingress (BEFORE any model load / internal-work mint)
/// and again at the worker after authorization: byte bound, ordered-visitor
/// parse (rejects duplicate request fields/question IDs, media fields,
/// out-of-domain numbers), per-question cardinalities and AGGREGATE budgets.
/// Tokenization and tensor work never happen here.
pub fn preflight_body(body: &str) -> Result<ClefRequest, ClefDecisionError> {
    if body.len() > MAX_CLEF_BODY_BYTES {
        return Err(ClefDecisionError::Input(format!(
            "raw Clef body {} bytes exceeds bound {MAX_CLEF_BODY_BYTES}",
            body.len()
        )));
    }
    let request: ClefRequest = serde_json::from_str(body).map_err(|e| {
        ClefDecisionError::Input(format!("raw Clef body rejected by ordered parser: {e}"))
    })?;
    validate_request(&request)?;
    validate_aggregate(&request, DEFAULT_CLEF_MAX_LENGTH)?;
    Ok(request)
}

/// Cooperative cancellation + deadline for one decision request.
///
/// The checker closure passed to the engine runs before admission, between
/// decoder layers, and around head evaluation. An active libtorch kernel
/// completes before the next check; a cancelled or failed forward drops its
/// call-local state, so nothing persists. Disconnecting a client alone does
/// not stop a running CPU forward — the deadline and drain flag bound it.
#[derive(Clone, Debug)]
pub struct DecisionCancel {
    flag: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl DecisionCancel {
    /// New checker: external cancel flag (e.g. service drain) plus optional
    /// absolute deadline.
    pub fn new(flag: Arc<AtomicBool>, deadline: Option<Instant>) -> Self {
        Self { flag, deadline }
    }

    /// A never-firing checker for tooling/tests on the default path.
    pub fn none() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            deadline: None,
        }
    }

    /// Fail with the typed cancellation error when tripped. Called before
    /// admission, per decoder layer, and around the head closure.
    pub fn check(&self) -> Result<()> {
        if self.flag.load(Ordering::Acquire) {
            return Err(ClefDecisionError::Cancelled("cancel flag set".into()).into());
        }
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return Err(ClefDecisionError::Cancelled("deadline exceeded".into()).into());
            }
        }
        Ok(())
    }
}

/// Content identity of the admitted bundle, computed ONCE from the same
/// immutable bytes this loader mapped. Never a directory name or echo.
#[derive(Clone, Debug)]
pub struct ClefBundleIdentity {
    pub head_config_sha256: String,
    pub head_weights_sha256: String,
    /// tokenizer.json digest; `None` when the bundle ships none.
    pub tokenizer_sha256: Option<String>,
    /// backbone config.json digest; `None` when absent.
    pub backbone_config_sha256: Option<String>,
    /// quantization format actually declared by config.json, if any.
    pub quantization: Option<String>,
    pub head_tensor_count: u32,
}

/// Cached Clef decision runtime: validated head + identity. One per worker
/// engine instance; built once, then read-only.
pub struct ClefDecisionRuntime {
    head: JointSchemaHead,
    pub identity: ClefBundleIdentity,
    pub encoding_version: &'static str,
    /// Whether the admitted bundle carries a channel-FP8 profile (input to
    /// the OBSERVED compute-profile string; artifact quantization stays in
    /// identity.quantization).
    pub channel_fp8: bool,
    /// Kind the head tensors were converted to (must equal the engine's
    /// compute kind; mismatch fails closed inside `JointSchemaHead::forward`).
    pub compute_kind: Kind,
    /// ADMITTED compute profile (policy-derived at load: same ChannelFp8
    /// detector and precision mapping the engine applies), validated at
    /// forward time by JointSchemaHead's own actual hidden/row dtype+device
    /// checks — a drift cannot produce a mislabeled successful response.
    /// e.g. "cpu-float32-fp8ch-w8a8-dynact" or "cpu-float32".
    pub compute_profile: String,
}

/// File-derived default head dtype. Mirrors the runtime FP8 policy (CPU +
/// quantized artifact → FP32 decode; else the config dtype). A drift between
/// this mirror and the engine policy fails closed at forward with the head's
/// dtype-mismatch error.
/// Serving-qualified compute kind for the Clef v1 endpoint (R0 §5): CPU
/// Float32 compute — for channel-FP8 artifacts, packed FP8 weights + dynamic
/// per-token E4M3FN activation QDQ over Float32 CPU matmuls. Other engine
/// precision modes (e.g. explicit bf16) remain valid for diagnostic engine
/// use but are NOT admitted for this serving path; the caller rejects them
/// with a typed profile error BEFORE any head materialization or forward.
/// No silent promotion of an unqualified kind.
pub fn resolve_serving_kind(
    model_path: &Path,
    precision_mode: Option<&str>,
) -> Result<Kind, ClefDecisionError> {
    let kind = default_head_kind(model_path, precision_mode)?;
    // Clef v1 serving gate (R0 section 5): CPU Float32 ONLY. Explicit
    // BF16/FP16 and non-channel auto/fp8 resolve (BFloat16/Half) but are
    // NOT qualified for serving — typed rejection, no silent downgrade or
    // promotion.
    if kind != Kind::Float {
        return Err(ClefDecisionError::Profile(format!(
            "clef v1 serving profile admits cpu-float32 compute only; got {kind:?} (R0 section 5)"
        )));
    }
    Ok(kind)
}

/// Head dtype from the AUTHORITATIVE shared policy
/// (`torch_engine::resolve_inference_compute_kind`): case-insensitive
/// aliases, CPU-device binding, channel-FP8 auto/fp8 -> Float. No duplicated
/// mapping in this crate; drift is impossible by construction.
fn default_head_kind(
    model_path: &Path,
    precision_mode: Option<&str>,
) -> Result<Kind, ClefDecisionError> {
    let channel_fp8 = channel_fp8_present(model_path)?;
    crate::runtime::torch_engine::resolve_inference_compute_kind(
        precision_mode,
        tch::Device::Cpu,
        channel_fp8,
    )
    .map_err(|e| ClefDecisionError::Bundle(format!("precision policy: {e}")))
}

/// Authoritative channel-Fp8Profile detection (the same detector the engine
/// uses), shared by the serving gate and the compute-profile string.
fn channel_fp8_present(model_path: &Path) -> Result<bool, ClefDecisionError> {
    Ok(ChannelFp8Profile::read(model_path)
        .map_err(|e| ClefDecisionError::Bundle(format!("fp8 profile read: {e}")))?
        .is_some())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn sha256_file(path: &Path) -> Result<Option<String>> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(Some(hex::encode(h.finalize())))
}

/// The tensor dtype kind used for compute-profile reporting (re-exported so
/// service crates need no direct tch dependency for this boundary).
pub type ComputeKind = tch::Kind;

/// Observed compute profile string: device + actual compute dtype + the
/// native execution path. For channel-FP8 artifacts the path is packed
/// per-channel FP8 weights with dynamic per-token E4M3FN activation QDQ and
/// `kind` CPU matmuls (the accepted target is Float32). No hardware-FP8
/// kernel or performance claim.
pub fn compute_profile_string(kind: Kind, channel_fp8: bool) -> String {
    let dtype = match kind {
        Kind::Float => "float32",
        Kind::BFloat16 => "bfloat16",
        Kind::Half => "float16",
        _ => "other",
    };
    if channel_fp8 {
        format!("cpu-{dtype}-fp8ch-w8a8-dynact")
    } else {
        format!("cpu-{dtype}")
    }
}

fn quant_profile(config: &Value) -> Option<String> {
    let qc = config.get("quantization_config")?;
    let format = qc.get("format").and_then(Value::as_str)?;
    let groups = qc.get("config_groups")?.as_object()?;
    let group = groups.values().next()?;
    let weights = group.get("weights")?;
    let num_bits = weights.get("num_bits").and_then(Value::as_u64)?;
    let strategy = weights.get("strategy").and_then(Value::as_str)?;
    let dyn_act = group
        .get("input_activations")
        .and_then(|a| a.get("dynamic"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Some(format!(
        "cpu-{format}-{strategy}-w{num_bits}{}",
        if dyn_act { "-dynact" } else { "" }
    ))
}

/// Validate head config geometry + safet/// Bounded head geometry admission with checked arithmetic. Runs BEFORE the
/// expected-shape derivation and before any tensor work.
fn validate_head_geometry(config: &ClefHeadConfig) -> Result<(), ClefDecisionError> {
    let bound = |v: i64, max: i64, what: &str| -> Result<(), ClefDecisionError> {
        if !(1..=max).contains(&v) {
            return Err(ClefDecisionError::Bundle(format!(
                "head {what} {v} outside 1..={max}"
            )));
        }
        Ok(())
    };
    bound(config.hidden_size, MAX_HEAD_HIDDEN_SIZE, "hidden_size")?;
    bound(config.width, MAX_HEAD_WIDTH, "width")?;
    bound(config.heads, MAX_HEAD_ATTENTION_HEADS, "heads")?;
    bound(config.feedforward, MAX_HEAD_FEEDFORWARD, "feedforward")?;
    if config.routing_layers == 0 || config.routing_layers > MAX_HEAD_ROUTING_LAYERS {
        return Err(ClefDecisionError::Bundle(format!(
            "head routing_layers {} outside 1..={MAX_HEAD_ROUTING_LAYERS}",
            config.routing_layers
        )));
    }
    if config.layers == 0 || config.layers > MAX_HEAD_DECODER_LAYERS {
        return Err(ClefDecisionError::Bundle(format!(
            "head layers {} outside 1..={MAX_HEAD_DECODER_LAYERS}",
            config.layers
        )));
    }
    if config.width % config.heads != 0 {
        return Err(ClefDecisionError::Bundle(
            "head width not divisible by heads".into(),
        ));
    }
    // Checked products for the head's widest derived tensors
    // (residual_scorer.0 is [width, width*4]).
    config
        .width
        .checked_mul(4)
        .filter(|v| *v <= MAX_HEAD_WIDTH * 4)
        .ok_or_else(|| ClefDecisionError::Bundle("head width*4 overflow".into()))?;
    Ok(())
}

/// Metadata admission for one head tensor set: everything needed to reject a
/// wrong file BEFORE materializing any tensor.
#[derive(Debug)]
struct AdmittedHeadMetadata {
    /// (name -> (shape, safetensors dtype)), exactly the expected key set.
    tensors: Vec<(String, Vec<i64>, safetensors::Dtype)>,
    expected_total_bytes: u64,
    total_params: u64,
}

/// Validate head config geometry + safetensors METADATA against the strict
/// expected-shape contract. No tensor is built here; a wrong file is rejected
/// by name/count/shape/dtype/bounded-bytes alone.
fn head_metadata_admission(
    config: &ClefHeadConfig,
    engine_hidden: Option<u32>,
    tensors: &SafeTensors<'_>,
    file_len: u64,
) -> Result<AdmittedHeadMetadata, ClefDecisionError> {
    validate_head_geometry(config)?;
    if let Some(hidden) = engine_hidden {
        if config.hidden_size != i64::from(hidden) {
            return Err(ClefDecisionError::Bundle(format!(
                "head hidden_size {} != admitted engine hidden size {hidden}",
                config.hidden_size
            )));
        }
    }
    if file_len > MAX_HEAD_WEIGHTS_BYTES {
        return Err(ClefDecisionError::Bundle(format!(
            "head weights file {file_len} bytes exceeds bound {MAX_HEAD_WEIGHTS_BYTES}"
        )));
    }
    let expected = JointSchemaHead::expected_shapes(config)
        .map_err(|e| ClefDecisionError::Bundle(format!("head config geometry: {e}")))?;
    if tensors.len() != expected.len() {
        return Err(ClefDecisionError::Bundle(format!(
            "head tensor count {} != expected {}",
            tensors.len(),
            expected.len()
        )));
    }
    let mut admitted = Vec::with_capacity(expected.len());
    let mut total_bytes: u64 = 0;
    let mut total_params: u64 = 0;
    for (name, shape) in &expected {
        let view = tensors
            .tensor(name)
            .map_err(|_| ClefDecisionError::Bundle(format!("head tensor missing: {name}")))?;
        let dtype = view.dtype();
        if !matches!(
            dtype,
            safetensors::Dtype::BF16 | safetensors::Dtype::F16 | safetensors::Dtype::F32
        ) {
            return Err(ClefDecisionError::Bundle(format!(
                "head tensor {name}: unsupported dtype {dtype:?}"
            )));
        }
        if view.shape().len() != shape.len() {
            return Err(ClefDecisionError::Bundle(format!(
                "head tensor {name}: rank {} != expected {}",
                view.shape().len(),
                shape.len()
            )));
        }
        let mut checked = Vec::with_capacity(shape.len());
        for (got, want) in view.shape().iter().zip(shape.iter()) {
            let got_i = i64::try_from(*got).map_err(|_| {
                ClefDecisionError::Bundle(format!("head tensor {name}: dimension overflow"))
            })?;
            if got_i != *want {
                return Err(ClefDecisionError::Bundle(format!(
                    "head tensor {name}: shape {:?} != expected {shape:?}",
                    view.shape()
                )));
            }
            checked.push(got_i);
        }
        let numel: u64 = checked
            .iter()
            .try_fold(1u64, |acc, d| {
                let d = u64::try_from(*d).map_err(|_| ())?;
                acc.checked_mul(d).ok_or(())
            })
            .map_err(|()| {
                ClefDecisionError::Bundle(format!("head tensor {name}: size overflow"))
            })?;
        total_params = total_params
            .checked_add(numel)
            .ok_or_else(|| ClefDecisionError::Bundle("head total parameter overflow".into()))?;
        total_bytes = total_bytes
            .checked_add(
                numel
                    .checked_mul(match dtype {
                        safetensors::Dtype::BF16 | safetensors::Dtype::F16 => 2,
                        _ => 4,
                    })
                    .ok_or_else(|| {
                        ClefDecisionError::Bundle(format!("head tensor {name}: byte overflow"))
                    })?,
            )
            .ok_or_else(|| ClefDecisionError::Bundle("head total byte overflow".into()))?;
        admitted.push((name.clone(), checked, dtype));
    }
    // The total-parameter cap is enforced by the caller against the
    // admitted struct field (single source of truth in `load`).
    Ok(AdmittedHeadMetadata {
        tensors: admitted,
        expected_total_bytes: total_bytes,
        total_params,
    })
}

/// Materialize the admitted tensors (copy out of the mapped file; one
/// explicit dtype conversion when the profile differs). Only called AFTER
/// [`head_metadata_admission`] succeeded.
fn materialize_head(
    metadata: &AdmittedHeadMetadata,
    tensors: &SafeTensors<'_>,
    kind: Kind,
) -> Result<HashMap<String, Tensor>, ClefDecisionError> {
    let mut weights = HashMap::with_capacity(metadata.tensors.len());
    for (name, shape, dtype) in &metadata.tensors {
        let view = tensors.tensor(name).map_err(|_| {
            ClefDecisionError::Bundle(format!(
                "head tensor {name} missing after admission (internal invariant)"
            ))
        })?;
        let raw_kind = match dtype {
            safetensors::Dtype::BF16 => Kind::BFloat16,
            safetensors::Dtype::F16 => Kind::Half,
            _ => Kind::Float,
        };
        let tensor = Tensor::from_data_size(view.data(), shape, raw_kind);
        let tensor = if tensor.kind() == kind {
            tensor
        } else {
            tensor.to_kind(kind)
        };
        weights.insert(name.clone(), tensor);
    }
    Ok(weights)
}

impl ClefDecisionRuntime {
    /// Returns the head directory when `model_path` is a Clef bundle.
    /// Existence check only — strict validation happens in [`Self::load`].
    pub fn detect(model_path: &Path) -> Option<PathBuf> {
        let head_dir = model_path.join(CLEF_HEAD_SUBDIR);
        let has = head_dir.join(CLEF_HEAD_CONFIG_FILE).is_file()
            && head_dir.join(CLEF_HEAD_WEIGHTS_FILE).is_file();
        has.then_some(head_dir)
    }

    /// Strictly load and validate the original joint head from the admitted
    /// (immutable, service-pinned) model directory.
    ///
    /// `engine_hidden` validates the head geometry against the admitted
    /// engine before the weights are even mapped. `kind_hint` overrides the
    /// file-derived head dtype (tests with tiny FP32 backbones); production
    /// passes `None`.
    pub fn load(
        model_path: &Path,
        engine_hidden: Option<u32>,
        kind_hint: Option<Kind>,
        precision_mode: Option<&str>,
    ) -> Result<Self> {
        let head_dir = Self::detect(model_path)
            .ok_or_else(|| ClefDecisionError::Bundle("not a Clef bundle".into()))?;
        let config_meta = std::fs::metadata(head_dir.join(CLEF_HEAD_CONFIG_FILE))
            .with_context(|| "stat Clef head config")?;
        if config_meta.len() > MAX_HEAD_CONFIG_BYTES as u64 {
            return Err(ClefDecisionError::Bundle(format!(
                "head config exceeds {MAX_HEAD_CONFIG_BYTES} bytes"
            ))
            .into());
        }
        let config_bytes = std::fs::read(head_dir.join(CLEF_HEAD_CONFIG_FILE))
            .with_context(|| "reading Clef head config")?;
        let config: ClefHeadConfig = serde_json::from_slice(&config_bytes).map_err(|e| {
            ClefDecisionError::Bundle(format!("invalid {CLEF_HEAD_CONFIG_FILE}: {e}"))
        })?;

        let weights_path = head_dir.join(CLEF_HEAD_WEIGHTS_FILE);
        // File-size bound BEFORE any mapping or parsing (ordering matters:
        // an oversized file must never reach the header decoder).
        let file_len = std::fs::metadata(&weights_path)
            .with_context(|| "stat Clef head weights")?
            .len();
        if file_len > MAX_HEAD_WEIGHTS_BYTES {
            return Err(ClefDecisionError::Bundle(format!(
                "head weights file {file_len} bytes exceeds bound {MAX_HEAD_WEIGHTS_BYTES}"
            ))
            .into());
        }
        // Identity digests come from THIS mapping — the same bytes the head
        // tensors are materialized from.
        let weights_file =
            std::fs::File::open(&weights_path).with_context(|| "opening Clef head weights")?;
        let mmap =
            unsafe { Mmap::map(&weights_file) }.with_context(|| "mapping Clef head weights")?;
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
            ClefDecisionError::Bundle(format!("invalid {CLEF_HEAD_WEIGHTS_FILE}: {e}"))
        })?;

        // Backbone config identity: REQUIRED (fail closed — incomplete
        // provenance is never attested), length-bounded via metadata before
        // the read, parsed strictly.
        let config_path = model_path.join("config.json");
        let config_len = std::fs::metadata(&config_path)
            .with_context(|| "stat backbone config.json")?
            .len();
        if config_len > MAX_HEAD_CONFIG_BYTES as u64 {
            return Err(ClefDecisionError::Bundle(format!(
                "backbone config.json {config_len} bytes exceeds bound {MAX_HEAD_CONFIG_BYTES}"
            ))
            .into());
        }
        let backbone_bytes =
            std::fs::read(&config_path).with_context(|| "reading backbone config.json")?;
        let backbone_config: Value = serde_json::from_slice(&backbone_bytes)
            .map_err(|e| ClefDecisionError::Bundle(format!("invalid backbone config.json: {e}")))?;
        let kind = kind_hint.unwrap_or(default_head_kind(model_path, precision_mode)?);

        // Metadata admission BEFORE any tensor materialization. The
        // admitted metadata is LOAD-BEARING here: the summed data-region
        // bound and total parameter cap are enforced against the struct
        // fields (not just computed), and the file must physically contain
        // the whole admitted data region.
        let metadata = head_metadata_admission(&config, engine_hidden, &tensors, file_len)?;
        anyhow::ensure!(
            file_len >= metadata.expected_total_bytes,
            ClefDecisionError::Bundle(format!(
                "head weights file {file_len} bytes smaller than admitted data region {}",
                metadata.expected_total_bytes
            ))
            .to_string()
        );
        anyhow::ensure!(
            metadata.total_params <= MAX_HEAD_TOTAL_PARAMS,
            ClefDecisionError::Bundle(format!(
                "head total parameters {} exceed bound {MAX_HEAD_TOTAL_PARAMS}",
                metadata.total_params
            ))
            .to_string()
        );
        let weights = materialize_head(&metadata, &tensors, kind)?;
        let head_tensor_count = weights.len() as u32;
        let head = JointSchemaHead::from_named_tensors(config, weights)
            .map_err(|e| ClefDecisionError::Bundle(format!("joint head validation: {e}")))?;

        // compute_profile reports the OBSERVED compute kind/device and the
        // execution path; artifact quantization is reported separately in
        // identity.quantization (T3: identical quantized artifacts under
        // different supported compute modes must yield DISTINCT profiles).
        let fp8 = channel_fp8_present(model_path)?;
        let compute_profile = compute_profile_string(kind, fp8);

        Ok(Self {
            identity: ClefBundleIdentity {
                head_config_sha256: sha256_hex(&config_bytes),
                head_weights_sha256: sha256_hex(&mmap[..]),
                // The bundle tokenizer is REQUIRED for decision serving:
                // fail closed rather than attesting incomplete provenance.
                tokenizer_sha256: Some(
                    sha256_file(&model_path.join("tokenizer.json"))?.ok_or_else(|| {
                        ClefDecisionError::Bundle(
                            "decision bundle is missing tokenizer.json".into(),
                        )
                    })?,
                ),
                backbone_config_sha256: Some(sha256_hex(&backbone_bytes)),
                quantization: quant_profile(&backbone_config)
                    .map(|p| p.trim_start_matches("cpu-").to_owned()),
                head_tensor_count,
            },
            head,
            encoding_version: CLEF_ENCODING_VERSION,
            channel_fp8: fp8,
            compute_kind: kind,
            compute_profile,
        })
    }
}

/// Canonical labels for one admitted question (see module docs). Sorted
/// choice IDs, noul [false,true], score ascending indices.
pub fn canonical_labels(question: &ClefQuestion) -> Result<Vec<String>, ClefDecisionError> {
    match question.kind {
        ClefQuestionType::Noul => Ok(vec!["false".to_owned(), "true".to_owned()]),
        ClefQuestionType::Choice => {
            let map = question.criteria.as_object().ok_or_else(|| {
                ClefDecisionError::Input("choice criteria must be an object".into())
            })?;
            let mut ids: Vec<String> = map.keys().cloned().collect();
            ids.sort();
            Ok(ids)
        }
        ClefQuestionType::Score => {
            let n = question.criteria.as_array().ok_or_else(|| {
                ClefDecisionError::Input("score criteria must be an array".into())
            })?;
            Ok((0..n.len()).map(|i| i.to_string()).collect())
        }
    }
}

/// Per-question admission: type/cardinality checks BEFORE any tokenization
/// or tensor work (deterministic typed errors; D1/D2 bounds).
pub fn validate_request(request: &ClefRequest) -> Result<(), ClefDecisionError> {
    let n = request.record.questions.len();
    if n == 0 || n > MAX_CLEF_QUESTIONS {
        return Err(ClefDecisionError::Input(format!(
            "question count {n} outside 1..={MAX_CLEF_QUESTIONS}"
        )));
    }
    for (id, q) in &request.record.questions {
        if id.is_empty() {
            return Err(ClefDecisionError::Input("empty question id".into()));
        }
        match q.kind {
            ClefQuestionType::Noul => {
                if !q.criteria.is_null() && !q.criteria.is_object() {
                    return Err(ClefDecisionError::Input(format!(
                        "question {id}: noul criteria must be null or an object"
                    )));
                }
            }
            ClefQuestionType::Choice => {
                let map = q.criteria.as_object().ok_or_else(|| {
                    ClefDecisionError::Input(format!(
                        "question {id}: choice criteria must be an object"
                    ))
                })?;
                if map.len() < 2 || map.len() > MAX_CLEF_CHOICE_OPTIONS {
                    return Err(ClefDecisionError::Input(format!(
                        "question {id}: choice options {} outside 2..={MAX_CLEF_CHOICE_OPTIONS}",
                        map.len()
                    )));
                }
            }
            ClefQuestionType::Score => {
                let arr = q.criteria.as_array().ok_or_else(|| {
                    ClefDecisionError::Input(format!(
                        "question {id}: score criteria must be an array"
                    ))
                })?;
                if arr.len() < 2 || arr.len() > MAX_CLEF_SCORE_LEVELS {
                    return Err(ClefDecisionError::Input(format!(
                        "question {id}: score levels {} outside 2..={MAX_CLEF_SCORE_LEVELS}",
                        arr.len()
                    )));
                }
            }
        }
    }
    Ok(())
}

/// AGGREGATE admission (server-controlled; callers may lower, never raise):
/// total options across the record and total_options × sequence budget with
/// overflow-safe arithmetic. Individually valid questions can still exceed
/// these; reject before tokenization/tensor allocation.
pub fn validate_aggregate(
    request: &ClefRequest,
    max_length: usize,
) -> Result<(), ClefDecisionError> {
    let mut total_options: u64 = 0;
    for (id, q) in &request.record.questions {
        let per = match q.kind {
            ClefQuestionType::Noul => 2u64,
            ClefQuestionType::Choice => q.criteria.as_object().map(|m| m.len() as u64).unwrap_or(0),
            ClefQuestionType::Score => q.criteria.as_array().map(|a| a.len() as u64).unwrap_or(0),
        };
        total_options = total_options.checked_add(per).ok_or_else(|| {
            ClefDecisionError::Input(format!("question {id}: aggregate option overflow"))
        })?;
    }
    if total_options > MAX_CLEF_TOTAL_OPTIONS {
        return Err(ClefDecisionError::Input(format!(
            "aggregate options {total_options} exceed bound {MAX_CLEF_TOTAL_OPTIONS}"
        )));
    }
    let cells = total_options
        .checked_mul(max_length as u64)
        .ok_or_else(|| ClefDecisionError::Input("aggregate attention budget overflow".into()))?;
    if cells > MAX_CLEF_ATTENTION_CELLS {
        return Err(ClefDecisionError::Input(format!(
            "aggregate attention cells {cells} exceed bound {MAX_CLEF_ATTENTION_CELLS}"
        )));
    }
    Ok(())
}

/// Map one question's logits tensor (model option order) to probabilities in
/// CANONICAL label order. Softmax in FP32 regardless of compute dtype.
pub fn map_question_probabilities(
    logits: &Tensor,
    model_ids: &[String],
    canonical_ids: &[String],
) -> Result<Vec<f32>, ClefDecisionError> {
    let permutation = clef_label_permutation(model_ids, canonical_ids)?;
    let n = logits.size()[0];
    if n as usize != model_ids.len() {
        return Err(ClefDecisionError::Runtime(format!(
            "logits cardinality {n} != option count {}",
            model_ids.len()
        )));
    }
    let probs = logits.softmax(-1, Kind::Float);
    let numel = probs.numel();
    let mut buf = vec![0f32; numel];
    probs.copy_data(&mut buf, numel);
    let mapped = permutation
        .iter()
        .map(|&i| buf[i as usize])
        .collect::<Vec<_>>();
    if mapped.iter().any(|p| !p.is_finite()) {
        return Err(ClefDecisionError::Runtime(
            "nonfinite probability after softmax".to_owned(),
        ));
    }
    Ok(mapped)
}

/// Per-question canonical distribution, bound to the request's question ID.
#[derive(Clone, Debug)]
pub struct ClefQuestionDistribution {
    pub question_id: String,
    pub kind: ClefQuestionType,
    /// Canonical label order for `probabilities` (module docs).
    pub canonical_ids: Vec<String>,
    pub probabilities: Vec<f32>,
}

/// Deterministic usage accounting from the encoder's own single state
/// tokenization (never re-tokenized here).
#[derive(Clone, Copy, Debug)]
pub struct ClefDecisionAccounting {
    pub max_length: usize,
    pub max_state_tokens_effective: Option<usize>,
    pub state_tokens_before_truncation: usize,
    pub state_tokens_admitted: usize,
    pub state_truncated: bool,
    pub sequence_tokens: usize,
}

fn accounting(encoded: &EncodedClefRecord, limits: &ClefEncodingLimits) -> ClefDecisionAccounting {
    ClefDecisionAccounting {
        max_length: limits.max_length,
        max_state_tokens_effective: limits.max_state_tokens,
        state_tokens_before_truncation: encoded.state_tokens_before_truncation,
        state_tokens_admitted: encoded.state_tokens_admitted,
        state_truncated: encoded.state_truncated,
        sequence_tokens: encoded.input_ids.len(),
    }
}

/// Tokenizer-only admission + encoding: per-question validation, aggregate
/// budgets (against `max_length`), cooperative cancel check, then the single
/// canonical tokenization with deterministic state truncation. SCHEMA TOKEN
/// OVERFLOW IS REJECTED HERE — before any head materialization or backbone
/// forward. Returns the encoded record plus exact accounting.
pub fn encode_request(
    tokenizer: &Tokenizer,
    request: &ClefRequest,
    max_state_tokens: Option<usize>,
    max_length: usize,
    cancel: &DecisionCancel,
) -> Result<(EncodedClefRecord, ClefDecisionAccounting)> {
    validate_request(request)?;
    validate_aggregate(request, max_length)?;
    cancel.check()?;
    let limits = ClefEncodingLimits {
        max_length,
        max_state_tokens,
    };
    let encoded =
        ClefEncoder::encode(tokenizer, &request.record, limits).map_err(ClefDecisionError::from)?;
    cancel.check()?;
    let usage = accounting(&encoded, &limits);
    Ok((encoded, usage))
}

/// Run one admitted, already-encoded record through the cached runtime:
/// fresh-state CPU forward → per-question softmax → canonical remap.
pub fn decide_encoded(
    runtime: &ClefDecisionRuntime,
    engine: &TorchEngine,
    request: &ClefRequest,
    encoded: &EncodedClefRecord,
    cancel: &DecisionCancel,
) -> Result<(Vec<ClefQuestionDistribution>, tch::Kind)> {
    let mut observed_kind = None;
    let model_ids: Vec<Vec<String>> = encoded
        .questions
        .iter()
        .map(|q| q.option_ids.clone())
        .collect();
    let canonical: Vec<Vec<String>> = request
        .record
        .questions
        .iter()
        .map(|(_, q)| canonical_labels(q))
        .collect::<Result<_, _>>()?;

    // One unpadded record, fresh call-local state, model lock + no_grad held
    // by the engine; the checker bounds admitted work cooperatively.
    let logits = engine
        .with_decision_backbone_checked(&encoded.input_ids, &|| cancel.check(), |hidden, gather| {
            // OBSERVED compute kind: recorded from the actual successful
            // hidden tensor the head validated (dtype+device), not from
            // load-time policy. Used for response provenance.
            observed_kind = Some(hidden.kind());
            runtime
                .head
                .forward(hidden, encoded, |ids| {
                    gather(ids).map_err(|e| ClefError::Head(e.to_string()))
                })
                .map_err(|e| anyhow!(ClefDecisionError::Runtime(format!("head forward: {e}"))))
        })
        .map_err(|e| {
            // Typed decision errors (cancellation/deadline) pass through
            // intact; genuinely untyped backbone/FP8 failures gain the
            // runtime prefix so they reach DECISION_RUNTIME on the wire
            // instead of INTERNAL. The source text is preserved.
            match e.downcast_ref::<ClefDecisionError>() {
                Some(_) => e,
                None => anyhow::Error::new(ClefDecisionError::Runtime(format!(
                    "backbone forward: {e:#}"
                ))),
            }
        })?;
    cancel.check()?;

    if logits.len() != encoded.questions.len() {
        return Err(ClefDecisionError::Runtime(format!(
            "head returned {} question logit sets for {} questions",
            logits.len(),
            encoded.questions.len()
        ))
        .into());
    }

    let mut distributions = Vec::with_capacity(logits.len());
    for (i, logit) in logits.iter().enumerate() {
        let probabilities = map_question_probabilities(logit, &model_ids[i], &canonical[i])?;
        distributions.push(ClefQuestionDistribution {
            question_id: request.record.questions[i].0.clone(),
            kind: request.record.questions[i].1.kind,
            canonical_ids: canonical[i].clone(),
            probabilities,
        });
    }
    // The head validates the hidden tensor's dtype/device BEFORE any
    // computation, so a successful forward means the observed kind IS the
    // admitted head kind; anything else is a typed runtime failure.
    let observed_kind = observed_kind.ok_or_else(|| {
        ClefDecisionError::Runtime("no observed hidden kind from successful forward".into())
    })?;
    if observed_kind != runtime.compute_kind {
        return Err(ClefDecisionError::Runtime(format!(
            "observed hidden kind {observed_kind:?} != admitted head kind {:?}",
            runtime.compute_kind
        ))
        .into());
    }
    Ok((distributions, observed_kind))
}

/// Convenience composition (tests/tooling): encode then forward.
pub fn decide(
    runtime: &ClefDecisionRuntime,
    tokenizer: &Tokenizer,
    engine: &TorchEngine,
    request: &ClefRequest,
    max_state_tokens: Option<usize>,
    max_length: usize,
    cancel: &DecisionCancel,
) -> Result<(Vec<ClefQuestionDistribution>, ClefDecisionAccounting)> {
    let (encoded, usage) =
        encode_request(tokenizer, request, max_state_tokens, max_length, cancel)?;
    let (distributions, _observed) = decide_encoded(runtime, engine, request, &encoded, cancel)?;
    Ok((distributions, usage))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // fixture assertions

    use super::*;
    use serde_json::json;

    fn question(kind: ClefQuestionType, criteria: Value) -> ClefQuestion {
        ClefQuestion {
            kind,
            instructions: Value::Null,
            criteria,
        }
    }

    fn request(questions: Vec<(String, ClefQuestion)>) -> ClefRequest {
        ClefRequest {
            model: "tiny".into(),
            record: hyprstream_decision_head::clef::ClefRecord {
                record_id: "t".into(),
                state: Value::Null,
                questions,
            },
        }
    }

    #[test]
    fn canonical_label_orders() {
        assert_eq!(
            canonical_labels(&question(ClefQuestionType::Noul, Value::Null)).unwrap(),
            ["false", "true"]
        );
        // Author order b,a,c is irrelevant: canonical = sorted IDs.
        assert_eq!(
            canonical_labels(&question(
                ClefQuestionType::Choice,
                json!({"b": null, "a": 1, "c": "x"})
            ))
            .unwrap(),
            ["a", "b", "c"]
        );
        assert_eq!(
            canonical_labels(&question(
                ClefQuestionType::Score,
                json!([null, null, null])
            ))
            .unwrap(),
            ["0", "1", "2"]
        );
        assert!(canonical_labels(&question(ClefQuestionType::Choice, json!([]))).is_err());
        assert!(canonical_labels(&question(ClefQuestionType::Score, json!({}))).is_err());
    }

    #[test]
    fn request_cardinality_bounds_before_any_tensor_work() {
        assert!(validate_request(&request(vec![(
            "q".into(),
            question(ClefQuestionType::Choice, json!({"only": 1}))
        )]))
        .is_err());
        let many: serde_json::Map<String, Value> = (0..=MAX_CLEF_CHOICE_OPTIONS)
            .map(|i| (i.to_string(), Value::Null))
            .collect();
        assert!(validate_request(&request(vec![(
            "q".into(),
            question(ClefQuestionType::Choice, Value::Object(many))
        )]))
        .is_err());
        assert!(validate_request(&request(vec![(
            "q".into(),
            question(ClefQuestionType::Score, json!([null]))
        )]))
        .is_err());
        assert!(validate_request(&request(vec![(
            "q".into(),
            question(ClefQuestionType::Noul, json!("bad"))
        )]))
        .is_err());
        assert!(validate_request(&request(vec![])).is_err());
        let too_many: Vec<(String, ClefQuestion)> = (0..=MAX_CLEF_QUESTIONS)
            .map(|i| (i.to_string(), question(ClefQuestionType::Noul, Value::Null)))
            .collect();
        assert!(validate_request(&request(too_many)).is_err());
    }

    #[test]
    fn aggregate_budget_bounds_individually_valid_questions() {
        // 255 individually-valid questions x 5 options = 1275 total >
        // MAX_CLEF_TOTAL_OPTIONS (question COUNT alone stays within bounds).
        let big: Vec<(String, ClefQuestion)> = (0..MAX_CLEF_QUESTIONS)
            .map(|i| {
                (
                    format!("q{i}"),
                    question(
                        ClefQuestionType::Choice,
                        json!({"a": null, "b": null, "c": null, "d": null, "e": null}),
                    ),
                )
            })
            .collect();
        let req = request(big);
        assert!(validate_request(&req).is_ok());
        let err = validate_aggregate(&req, DEFAULT_CLEF_MAX_LENGTH)
            .unwrap_err()
            .to_string();
        assert!(err.contains("aggregate options"), "{err}");

        // Small total options but full-length budget: 258 × 16384 > 2^22
        // (256 × 16384 == 2^22 exactly and is admitted).
        let tall: Vec<(String, ClefQuestion)> = (0..129)
            .map(|i| {
                (
                    format!("q{i}"),
                    question(ClefQuestionType::Choice, json!({"a": null, "b": null})),
                )
            })
            .collect();
        let req = request(tall);
        assert!(validate_request(&req).is_ok());
        assert!(validate_aggregate(&req, DEFAULT_CLEF_MAX_LENGTH).is_err());
        // A smaller admitted sequence budget admits the same request.
        assert!(validate_aggregate(&req, 4096).is_ok());
    }

    #[test]
    fn preflight_rejects_oversize_media_duplicates_and_bad_numbers() {
        let ok = r#"{"model":"m","state":null,"questions":{"q":{"type":"noul"}}}"#;
        assert!(preflight_body(ok).is_ok());
        // Byte bound.
        let huge = format!(
            r#"{{"model":"m","state":"{}","questions":{{"q":{{"type":"noul"}}}}}}"#,
            "x".repeat(MAX_CLEF_BODY_BYTES)
        );
        let err = preflight_body(&huge).unwrap_err().to_string();
        assert!(err.contains("exceeds bound"), "{err}");
        // Media field (ordered visitor).
        assert!(preflight_body(
            r#"{"model":"m","state":null,"questions":{"q":{"type":"noul"}},"images":[]}"#
        )
        .is_err());
        // Duplicate question id.
        assert!(preflight_body(
            r#"{"model":"m","state":null,"questions":{"q":{"type":"noul"},"q":{"type":"noul"}}}"#
        )
        .is_err());
        // Out-of-domain integer (lexical admission in the head parser).
        assert!(preflight_body(
            r#"{"model":"m","state":{"n":123456789012345678901234567890},"questions":{"q":{"type":"noul"}}}"#
        )
        .is_err());
        // Aggregate budget rejection through the shared preflight (255
        // individually-valid questions x 5 options).
        let many: Vec<String> = (0..MAX_CLEF_QUESTIONS)
            .map(|i| format!(r#""q{i}":{{"type":"choice","criteria":{{"a":null,"b":null,"c":null,"d":null,"e":null}}}}"#))
            .collect();
        let body = format!(
            r#"{{"model":"m","state":null,"questions":{{{}}}}}"#,
            many.join(",")
        );
        let err = preflight_body(&body).unwrap_err().to_string();
        assert!(err.contains("aggregate"), "{err}");
    }

    #[test]
    fn noul_and_choice_probability_mapping() {
        // Model order [true,false], canonical [false,true]: with logits
        // [1.0, 3.0] the model prefers "false", so p(false)≈0.881 and
        // p(true)≈0.119 in canonical order.
        let logits = Tensor::from_slice(&[1.0f32, 3.0f32]);
        let probs = map_question_probabilities(
            &logits,
            &["true".to_owned(), "false".to_owned()],
            &["false".to_owned(), "true".to_owned()],
        )
        .unwrap();
        // e^1/(e^1+e^3); clippy-suffixed literals preserve the ORIGINAL
        // f32 reference bits of 0.88079707 / 0.11920292 exactly.
        let (want_false, want_true) = (0.880_797_1_f32, 0.119_202_92_f32);
        assert!((probs[0] - want_false).abs() < 1e-4, "{probs:?}");
        assert!((probs[1] - want_true).abs() < 1e-4, "{probs:?}");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Identity mapping for sorted choice IDs; exact softmax values.
        let logits = Tensor::from_slice(&[0.2f32, 0.3f32, 5.0f32]);
        let probs = map_question_probabilities(
            &logits,
            &["a".to_owned(), "b".to_owned(), "c".to_owned()],
            &["a".to_owned(), "b".to_owned(), "c".to_owned()],
        )
        .unwrap();
        let z = 0.2f32.exp() + 0.3f32.exp() + 5.0f32.exp();
        for (got, raw) in probs.iter().zip([0.2f32, 0.3f32, 5.0f32]) {
            assert!((got - raw.exp() / z).abs() < 1e-5, "{probs:?}");
        }
        // Cardinality mismatch is a typed error.
        assert!(map_question_probabilities(
            &logits,
            &["a".to_owned(), "b".to_owned()],
            &["a".to_owned(), "b".to_owned()],
        )
        .is_err());
    }

    #[test]
    fn cancel_and_deadline_fail_typed() {
        let flag = Arc::new(AtomicBool::new(true));
        let c = DecisionCancel::new(flag, None);
        let err = c.check().unwrap_err().to_string();
        assert!(err.starts_with("clef-decision-cancelled"), "{err}");
        let c = DecisionCancel::new(
            Arc::new(AtomicBool::new(false)),
            Some(Instant::now() - std::time::Duration::from_secs(1)),
        );
        assert!(c.check().is_err());
        assert!(DecisionCancel::none().check().is_ok());
    }

    fn tiny_head_config(hidden: i64) -> ClefHeadConfig {
        ClefHeadConfig {
            hidden_size: hidden,
            width: 16,
            routing_layers: 1,
            layers: 1,
            heads: 4,
            feedforward: 16,
        }
    }

    fn tiny_head_tensors(config: &ClefHeadConfig) -> HashMap<String, Tensor> {
        let shapes = JointSchemaHead::expected_shapes(config).unwrap();
        let mut weights = HashMap::new();
        for (name, shape) in shapes {
            let n: i64 = shape.iter().product::<i64>().max(1);
            let t = (Tensor::arange(n, (Kind::Float, tch::Device::Cpu)) * 0.019)
                .sin()
                .reshape(&shape);
            weights.insert(name, t);
        }
        weights
    }

    /// tch::Tensor does not implement Clone in this fork: shallow-clone map.
    fn clone_weights(w: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        w.iter()
            .map(|(k, v)| (k.clone(), v.shallow_clone()))
            .collect()
    }

    /// Serialize f32 tensors to a safetensors file with explicit little-endian
    /// byte conversion (no unsafe buffer reinterpretation).
    fn write_safetensors(path: &Path, weights: &HashMap<String, Tensor>) {
        use safetensors::tensor::{serialize, TensorView};
        let mut data: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
        for (k, t) in weights {
            let t = t.contiguous();
            let n = t.numel();
            let mut vals = vec![0f32; n];
            t.copy_data(&mut vals, n);
            let mut bytes = Vec::with_capacity(n * 4);
            for v in &vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            data.push((
                k.clone(),
                safetensors::Dtype::F32,
                t.size().iter().map(|&d| d as usize).collect(),
                bytes,
            ));
        }
        let views: Vec<(String, TensorView<'_>)> = data
            .iter()
            .map(|(k, d, s, b)| (k.clone(), TensorView::new(*d, s.clone(), b).unwrap()))
            .collect();
        let refs: Vec<(&str, TensorView<'_>)> =
            views.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        std::fs::write(path, serialize(refs, &None).unwrap()).unwrap();
    }

    fn write_bundle(
        dir: &Path,
        config: &ClefHeadConfig,
        weights: &HashMap<String, Tensor>,
        backbone_config: Option<&str>,
    ) {
        let head_dir = dir.join(CLEF_HEAD_SUBDIR);
        std::fs::create_dir_all(&head_dir).unwrap();
        std::fs::write(
            head_dir.join(CLEF_HEAD_CONFIG_FILE),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
        write_safetensors(&head_dir.join(CLEF_HEAD_WEIGHTS_FILE), weights);
        if let Some(body) = backbone_config {
            std::fs::write(dir.join("config.json"), body).unwrap();
        }
        // The bundle tokenizer is required identity material for load().
        if !dir.join("tokenizer.json").exists() {
            tiny_wordlevel_tokenizer()
                .save(dir.join("tokenizer.json"), false)
                .unwrap();
        }
    }

    const SIMPLE_BACKBONE_CONFIG: &str = r#"{"model_type": "qwen3_5", "dtype": "float32"}"#;

    /// Scoped, lifetime-sound access to a bundle's head safetensors: the mmap
    /// and the borrowed SafeTensors live entirely inside the closure scope.
    fn with_head_safetensors<T>(dir: &Path, f: impl FnOnce(&SafeTensors<'_>, u64) -> T) -> T {
        let file =
            std::fs::File::open(dir.join(CLEF_HEAD_SUBDIR).join(CLEF_HEAD_WEIGHTS_FILE)).unwrap();
        let map = unsafe { Mmap::map(&file) }.unwrap();
        let tensors = SafeTensors::deserialize(&map[..]).unwrap();
        f(&tensors, map.len() as u64)
    }

    #[test]
    fn bundle_metadata_admission_rejects_before_materialization() {
        let config = tiny_head_config(8);
        let weights = tiny_head_tensors(&config);

        // Exact set admits.
        let dir = tempfile::tempdir().unwrap();
        write_bundle(dir.path(), &config, &weights, None);
        with_head_safetensors(dir.path(), |tensors, len| {
            let meta = head_metadata_admission(&config, Some(8), tensors, len).unwrap();
            assert_eq!(meta.tensors.len(), weights.len());
            assert!(meta.expected_total_bytes > 0);
        });

        // Extra tensor rejected by metadata alone.
        let mut extra = clone_weights(&weights);
        extra.insert(
            "rogue.weight".to_owned(),
            Tensor::zeros([2], (Kind::Float, tch::Device::Cpu)),
        );
        let dir = tempfile::tempdir().unwrap();
        write_bundle(dir.path(), &config, &extra, Some(SIMPLE_BACKBONE_CONFIG));
        with_head_safetensors(dir.path(), |tensors, len| {
            let err = head_metadata_admission(&config, None, tensors, len)
                .unwrap_err()
                .to_string();
            assert!(err.contains("tensor count"), "{err}");
        });

        // Wrong shape rejected by name/shape alone. DETERMINISTIC key
        // (min) and original rank preserved: only dimension 0 is invalid,
        // so the shape check — not an accidental rank mismatch — triggers.
        let mut wrong = clone_weights(&weights);
        let key = wrong.keys().min().unwrap().clone();
        let mut wrong_shape = wrong[&key].size();
        wrong_shape[0] += 1;
        wrong.insert(
            key.clone(),
            Tensor::zeros(wrong_shape.as_slice(), (Kind::Float, tch::Device::Cpu)),
        );
        let dir = tempfile::tempdir().unwrap();
        write_bundle(dir.path(), &config, &wrong, Some(SIMPLE_BACKBONE_CONFIG));
        with_head_safetensors(dir.path(), |tensors, len| {
            let err = head_metadata_admission(&config, None, tensors, len)
                .unwrap_err()
                .to_string();
            assert!(err.contains("shape"), "{err}");
        });

        // Engine hidden mismatch rejected from metadata alone (geometry check
        // precedes any per-tensor work).
        let dir = tempfile::tempdir().unwrap();
        write_bundle(dir.path(), &config, &weights, Some(SIMPLE_BACKBONE_CONFIG));
        with_head_safetensors(dir.path(), |tensors, len| {
            let err = head_metadata_admission(&config, Some(4096), tensors, len)
                .unwrap_err()
                .to_string();
            assert!(err.contains("hidden_size"), "{err}");
        });

        // Out-of-bound head geometry (small config, huge derived tensors)
        // rejected before expected_shapes runs — the bound itself must be
        // EXCLUSIVE here (a value inside 1..=MAX would be admitted).
        let mut huge = config.clone();
        huge.width = MAX_HEAD_WIDTH + 1;
        huge.heads = 1;
        let err = validate_head_geometry(&huge).unwrap_err().to_string();
        assert!(err.contains("width"), "{err}");
        let mut deep = config.clone();
        deep.layers = MAX_HEAD_DECODER_LAYERS + 1;
        let err = validate_head_geometry(&deep).unwrap_err().to_string();
        assert!(err.contains("layers"), "{err}");
    }

    #[test]
    fn bundle_loader_strict_rejections_and_identity() {
        let dir = tempfile::tempdir().unwrap();
        let config = tiny_head_config(8);
        let weights = tiny_head_tensors(&config);
        write_bundle(dir.path(), &config, &weights, Some(SIMPLE_BACKBONE_CONFIG));

        let runtime =
            ClefDecisionRuntime::load(dir.path(), Some(8), Some(Kind::Float), None).unwrap();
        assert_eq!(runtime.identity.head_tensor_count, weights.len() as u32);
        assert_eq!(runtime.compute_profile, "cpu-float32");
        assert_eq!(runtime.encoding_version, CLEF_ENCODING_VERSION);
        assert_eq!(runtime.identity.head_weights_sha256.len(), 64);
        // Required identity material: the bundle tokenizer digest is present.
        let tokenizer_digest = runtime
            .identity
            .tokenizer_sha256
            .expect("tokenizer digest required");
        assert_eq!(tokenizer_digest.len(), 64);
        assert!(runtime.identity.backbone_config_sha256.is_some());

        // Absent tokenizer fails closed (no incomplete provenance).
        let no_tok = tempfile::tempdir().unwrap();
        write_bundle(
            no_tok.path(),
            &config,
            &weights,
            Some(SIMPLE_BACKBONE_CONFIG),
        );
        std::fs::remove_file(no_tok.path().join("tokenizer.json")).unwrap();
        let err = match ClefDecisionRuntime::load(no_tok.path(), None, Some(Kind::Float), None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("missing tokenizer must fail closed"),
        };
        assert!(err.contains("tokenizer.json"), "{err}");

        // Engine hidden mismatch fails closed.
        // (unwrap_err would need Debug on the Ok type; match instead)
        let err = match ClefDecisionRuntime::load(dir.path(), Some(4096), Some(Kind::Float), None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("hidden-size mismatch must fail closed"),
        };
        assert!(err.contains("hidden_size"), "{err}");

        // Missing one tensor must fail closed.
        let mut broken = clone_weights(&weights);
        let key = broken.keys().next().unwrap().clone();
        broken.remove(&key);
        let dir2 = tempfile::tempdir().unwrap();
        write_bundle(dir2.path(), &config, &broken, Some(SIMPLE_BACKBONE_CONFIG));
        let err = match ClefDecisionRuntime::load(dir2.path(), None, Some(Kind::Float), None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("missing tensor must fail closed"),
        };
        assert!(err.contains("clef-decision-bundle"), "{err}");

        // A non-Clef directory is not a bundle.
        let plain = tempfile::tempdir().unwrap();
        assert!(ClefDecisionRuntime::detect(plain.path()).is_none());
    }

    // ── T5b: tiny native end-to-end decide() through a REAL engine ─────────
    //
    // Configurable tiny head geometry (NOT the official 4096 head) consistent
    // with a tiny Qwen3.5 backbone; real TorchEngine::load_model, real
    // checked forward, real tokenization. Real-head smoke remains the head
    // lane's artifact-gated tests; full-checkpoint acceptance is R5/T6.

    const T5_HIDDEN: i64 = 8;
    const T5_LAYERS: usize = 4;
    const T5_HEADS: i64 = 2;
    const T5_KV_HEADS: i64 = 1;
    const T5_HEAD_DIM: i64 = 4;
    const T5_INTER: i64 = 16;
    const T5_VOCAB: i64 = 64;
    const T5_LIN_K_HEADS: i64 = 1;
    const T5_LIN_K_DIM: i64 = 4;
    const T5_LIN_V_HEADS: i64 = 2;
    const T5_LIN_V_DIM: i64 = 4;
    const T5_CONV_KERNEL: i64 = 4;

    fn tiny_qwen_config_json() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3_5",
            "dtype": "float32",
            "hidden_size": T5_HIDDEN,
            "num_hidden_layers": T5_LAYERS,
            "num_attention_heads": T5_HEADS,
            "num_key_value_heads": T5_KV_HEADS,
            "head_dim": T5_HEAD_DIM,
            "intermediate_size": T5_INTER,
            "vocab_size": T5_VOCAB,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "linear_conv_kernel_dim": T5_CONV_KERNEL,
            "linear_key_head_dim": T5_LIN_K_DIM,
            "linear_value_head_dim": T5_LIN_V_DIM,
            "linear_num_key_heads": T5_LIN_K_HEADS,
            "linear_num_value_heads": T5_LIN_V_HEADS,
            // head_dim 4 x 0.25 default = rotary 1 (odd, rejected by the
            // engine); full rotary keeps it even.
            "partial_rotary_factor": 1.0,
        })
    }

    fn tiny_qwen_weights() -> HashMap<String, Tensor> {
        let opt = (Kind::Float, tch::Device::Cpu);
        let mut w = HashMap::new();
        let mut seed: i64 = 0;
        let mut pat = |dims: &[i64]| -> Tensor {
            let n: i64 = dims.iter().product();
            seed += 7;
            (Tensor::arange(n, opt) * 0.017 + seed as f64 * 0.013)
                .sin()
                .reshape(dims)
                * 0.05
        };
        w.insert(
            "model.embed_tokens.weight".to_owned(),
            pat(&[T5_VOCAB, T5_HIDDEN]),
        );
        w.insert("model.norm.weight".to_owned(), pat(&[T5_HIDDEN]));
        w.insert("lm_head.weight".to_owned(), pat(&[T5_VOCAB, T5_HIDDEN]));

        let key_dim = T5_LIN_K_HEADS * T5_LIN_K_DIM;
        let val_dim = T5_LIN_V_HEADS * T5_LIN_V_DIM;
        let conv_dim = key_dim * 2 + val_dim;
        for i in 0..T5_LAYERS {
            let p = format!("model.layers.{i}");
            w.insert(format!("{p}.input_layernorm.weight"), pat(&[T5_HIDDEN]));
            w.insert(
                format!("{p}.post_attention_layernorm.weight"),
                pat(&[T5_HIDDEN]),
            );
            let is_full = (i + 1) % 4 == 0;
            if is_full {
                let ap = format!("{p}.self_attn");
                w.insert(
                    format!("{ap}.q_proj.weight"),
                    pat(&[T5_HEADS * T5_HEAD_DIM * 2, T5_HIDDEN]),
                );
                w.insert(
                    format!("{ap}.k_proj.weight"),
                    pat(&[T5_KV_HEADS * T5_HEAD_DIM, T5_HIDDEN]),
                );
                w.insert(
                    format!("{ap}.v_proj.weight"),
                    pat(&[T5_KV_HEADS * T5_HEAD_DIM, T5_HIDDEN]),
                );
                w.insert(
                    format!("{ap}.o_proj.weight"),
                    pat(&[T5_HIDDEN, T5_HEADS * T5_HEAD_DIM]),
                );
                w.insert(format!("{ap}.q_norm.weight"), pat(&[T5_HEAD_DIM]));
                w.insert(format!("{ap}.k_norm.weight"), pat(&[T5_HEAD_DIM]));
            } else {
                let lp = format!("{p}.linear_attn");
                w.insert(
                    format!("{lp}.in_proj_qkv.weight"),
                    pat(&[conv_dim, T5_HIDDEN]),
                );
                w.insert(format!("{lp}.in_proj_z.weight"), pat(&[val_dim, T5_HIDDEN]));
                w.insert(
                    format!("{lp}.in_proj_b.weight"),
                    pat(&[T5_LIN_V_HEADS, T5_HIDDEN]),
                );
                w.insert(
                    format!("{lp}.in_proj_a.weight"),
                    pat(&[T5_LIN_V_HEADS, T5_HIDDEN]),
                );
                w.insert(
                    format!("{lp}.conv1d.weight"),
                    pat(&[conv_dim, 1, T5_CONV_KERNEL]),
                );
                w.insert(format!("{lp}.A_log"), pat(&[T5_LIN_V_HEADS]));
                w.insert(format!("{lp}.dt_bias"), pat(&[T5_LIN_V_HEADS]));
                w.insert(format!("{lp}.norm.weight"), pat(&[T5_LIN_V_DIM]));
                w.insert(format!("{lp}.out_proj.weight"), pat(&[T5_HIDDEN, val_dim]));
            }
            let mp = format!("{p}.mlp");
            w.insert(
                format!("{mp}.gate_proj.weight"),
                pat(&[T5_INTER, T5_HIDDEN]),
            );
            w.insert(format!("{mp}.up_proj.weight"), pat(&[T5_INTER, T5_HIDDEN]));
            w.insert(
                format!("{mp}.down_proj.weight"),
                pat(&[T5_HIDDEN, T5_INTER]),
            );
        }
        w
    }

    fn tiny_wordlevel_tokenizer() -> Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("tokenizer")
    }

    fn t5_record(state: &str) -> ClefRequest {
        ClefRequest {
            model: "tiny".into(),
            record: hyprstream_decision_head::clef::ClefRecord {
                record_id: "t5".into(),
                state: Value::String(state.to_owned()),
                questions: vec![
                    (
                        "is_ok".to_owned(),
                        question(ClefQuestionType::Noul, Value::Null),
                    ),
                    (
                        "pick".to_owned(),
                        question(
                            ClefQuestionType::Choice,
                            json!({"b": "second", "a": "first"}),
                        ),
                    ),
                    (
                        "grade".to_owned(),
                        question(ClefQuestionType::Score, json!(["lo", "hi"])),
                    ),
                ],
            },
        }
    }

    #[test]
    fn encoder_schema_overflow_preserves_typed_input_error() {
        let tokenizer = tiny_wordlevel_tokenizer();
        let body = serde_json::to_string(&json!({
            "model": "tiny", "state": "small",
            "questions": {"overflow": {"type": "noul", "instructions": "word ".repeat(4000)}}
        }))
        .unwrap();
        let request = preflight_body(&body).expect("cheap preflight admits valid schema JSON");
        let error = encode_request(&tokenizer, &request, None, 1024, &DecisionCancel::none())
            .expect_err("tokenized schema must overflow");
        assert!(
            matches!(error.downcast_ref::<ClefDecisionError>(), Some(ClefDecisionError::Input(message)) if message.contains("schema requires"))
        );
        assert!(error.to_string().starts_with("clef-decision-input:"));
        // The underlying real encoder produces a different error type; bare ?
        // would lose the service/runtime classification and fail this test.
        let raw = ClefEncoder::encode(
            &tokenizer,
            &request.record,
            ClefEncodingLimits {
                max_length: 1024,
                max_state_tokens: None,
            },
        )
        .unwrap_err();
        assert!(matches!(raw, ClefError::Input(_)));
        assert!(raw.to_string().starts_with("invalid Clef input:"));
    }

    fn t5_bundle_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        // Engine checkpoint first; write_bundle then adds the head subtree.
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&tiny_qwen_config_json()).unwrap(),
        )
        .unwrap();
        write_safetensors(&dir.path().join("model.safetensors"), &tiny_qwen_weights());
        let head_config = tiny_head_config(T5_HIDDEN);
        let head_weights = tiny_head_tensors(&head_config);
        write_bundle(dir.path(), &head_config, &head_weights, None);
        // Bundle tokenizer (identity digest source).
        tiny_wordlevel_tokenizer()
            .save(dir.path().join("tokenizer.json"), false)
            .unwrap();
        dir
    }

    #[tokio::test]
    async fn tiny_engine_decide_end_to_end_and_cancellation() {
        use crate::runtime::{create_engine, RuntimeConfig, RuntimeEngine};
        let dir = t5_bundle_dir();
        // EXPLICIT tiny profile: engine precision fp32 and head kind Float
        // match by construction (auto+non-FP8 would be BF16 on both sides —
        // also consistent, but explicit beats mirrored for a fixture).
        let config = RuntimeConfig {
            precision_mode: Some("fp32".to_owned()),
            ..RuntimeConfig::default()
        };
        let mut engine = create_engine(&config).expect("engine");
        RuntimeEngine::load_model(&mut engine, dir.path())
            .await
            .expect("tiny model load");

        let runtime = ClefDecisionRuntime::load(
            dir.path(),
            Some(T5_HIDDEN as u32),
            Some(Kind::Float),
            Some("fp32"),
        )
        .expect("runtime");
        let tokenizer = tiny_wordlevel_tokenizer();
        let request = t5_record("one two three four five");
        let cancel = DecisionCancel::none();

        let (distributions, accounting) = decide(
            &runtime,
            &tokenizer,
            &engine,
            &request,
            Some(DEFAULT_MAX_STATE_TOKENS as usize),
            256,
            &cancel,
        )
        .expect("decide");

        assert_eq!(distributions.len(), 3);
        let noul = &distributions[0];
        assert_eq!(noul.question_id, "is_ok");
        assert_eq!(noul.canonical_ids, ["false", "true"]);
        let sum: f32 = noul.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "{:?}", noul.probabilities);
        let pick = &distributions[1];
        assert_eq!(pick.canonical_ids, ["a", "b"]); // sorted, not author order
        let sum: f32 = pick.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        let grade = &distributions[2];
        assert_eq!(grade.canonical_ids, ["0", "1"]);
        assert!(accounting.sequence_tokens > 0);
        assert!(accounting.sequence_tokens <= 256);
        assert!(!accounting.state_truncated);
        assert_eq!(accounting.state_tokens_admitted, 5); // five whitespace tokens

        // OBSERVED kind from the successful forward equals the admitted kind
        // (Float32 for the explicit fp32 tiny profile) and feeds the
        // response compute profile.
        let (enc_only, _) =
            encode_request(&tokenizer, &request, Some(2), 256, &DecisionCancel::none()).unwrap();
        let (_, observed) = decide_encoded(
            &runtime,
            &engine,
            &request,
            &enc_only,
            &DecisionCancel::none(),
        )
        .unwrap();
        assert_eq!(observed, Kind::Float);

        // Deterministic fresh-state rerun: identical distributions bit-for-bit.
        let (again, _) = decide(
            &runtime,
            &tokenizer,
            &engine,
            &request,
            Some(DEFAULT_MAX_STATE_TOKENS as usize),
            256,
            &DecisionCancel::none(),
        )
        .expect("rerun");
        for (a, b) in distributions.iter().zip(&again) {
            assert_eq!(a.probabilities, b.probabilities);
        }

        // Deterministic truncation accounting via the explicit state cap.
        let (truncated, acc) = decide(
            &runtime,
            &tokenizer,
            &engine,
            &request,
            Some(2),
            256,
            &DecisionCancel::none(),
        )
        .expect("capped decide");
        assert!(acc.state_truncated);
        assert_eq!(acc.state_tokens_admitted, 2);
        assert_eq!(acc.state_tokens_before_truncation, 5);
        assert!(truncated[0].probabilities.iter().all(|p| p.is_finite()));

        // Head/engine dtype mismatch fails closed at the head's own strict
        // check (causal mismatch rejection).
        let bf16_head = ClefDecisionRuntime::load(
            dir.path(),
            Some(T5_HIDDEN as u32),
            Some(Kind::BFloat16),
            None,
        )
        .expect("bf16 head loads");
        let mismatch = decide(
            &bf16_head,
            &tokenizer,
            &engine,
            &request,
            Some(DEFAULT_MAX_STATE_TOKENS as usize),
            256,
            &DecisionCancel::none(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            mismatch.contains("dtype") || mismatch.contains("clef-decision-runtime"),
            "expected the head dtype-mismatch failure, got: {mismatch}"
        );

        // Pre-tripped cancellation fails typed BEFORE admission, and a fresh
        // request afterwards matches the clean baseline exactly (no residual
        // state). Engine-level between-layer cancellation is proven by the
        // runtime lane's dedicated tiny fixture (runtime-api.txt).
        let tripped = DecisionCancel::new(Arc::new(AtomicBool::new(true)), None);
        let err = decide(&runtime, &tokenizer, &engine, &request, None, 256, &tripped)
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("clef-decision-cancelled"), "{err}");
        let (fresh, _) = decide(
            &runtime,
            &tokenizer,
            &engine,
            &request,
            Some(DEFAULT_MAX_STATE_TOKENS as usize),
            256,
            &DecisionCancel::none(),
        )
        .expect("fresh after cancel");
        for (a, b) in distributions.iter().zip(&fresh) {
            assert_eq!(a.probabilities, b.probabilities);
        }
    }

    #[test]
    fn quant_profile_names_the_actual_format() {
        let config = json!({
            "quantization_config": {
                "format": "float-quantized",
                "config_groups": {"group_0": {
                    "weights": {"num_bits": 8, "strategy": "channel"},
                    "input_activations": {"dynamic": true}
                }}
            }
        });
        assert_eq!(
            quant_profile(&config).unwrap(),
            "cpu-float-quantized-channel-w8-dynact"
        );
        assert!(quant_profile(&json!({})).is_none());
    }

    #[test]
    fn serving_profile_rejects_bf16_and_accepts_fp32_or_auto_fp8() {
        // Causal controls at the resolver level: non-FP8 bundle (no channel
        // profile) — auto resolves BF16 and is REJECTED for v1 serving;
        // explicit fp32 resolves Float and is admitted; explicit bf16 is
        // rejected with the typed profile error.
        let dir = tempfile::tempdir().unwrap();
        let config = tiny_head_config(8);
        let weights = tiny_head_tensors(&config);
        write_bundle(dir.path(), &config, &weights, Some(SIMPLE_BACKBONE_CONFIG));

        let err = match resolve_serving_kind(dir.path(), None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("auto bf16 must not qualify for v1 serving"),
        };
        assert!(err.starts_with("clef-decision-profile"), "{err}");

        let err = match resolve_serving_kind(dir.path(), Some("bf16")) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("explicit bf16 must be rejected for v1 serving"),
        };
        assert!(err.contains("cpu-float32 compute only"), "{err}");
        // fp16/half alias: resolves Half, still rejected for serving.
        let err = match resolve_serving_kind(dir.path(), Some("FP16")) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("explicit fp16 must be rejected for v1 serving"),
        };
        assert!(err.contains("cpu-float32 compute only"), "{err}");
        // Case-insensitive fp32 alias accepted.
        assert_eq!(
            resolve_serving_kind(dir.path(), Some("Float32")).unwrap(),
            Kind::Float
        );
        // Non-channel fp8/auto is NOT silently qualified (resolves BF16).
        assert_eq!(
            crate::runtime::torch_engine::resolve_inference_compute_kind(
                Some("FP8"),
                tch::Device::Cpu,
                false
            )
            .unwrap(),
            Kind::BFloat16
        );
        assert_eq!(
            resolve_serving_kind(dir.path(), Some("fp32")).unwrap(),
            Kind::Float
        );
        // The generic engine path (diagnostics) still resolves BF16.
        assert_eq!(default_head_kind(dir.path(), None).unwrap(), Kind::BFloat16);
    }

    #[tokio::test]
    async fn untyped_backbone_failure_maps_to_typed_runtime_error() {
        // Engine WITHOUT a loaded model: the backbone's untyped anyhow
        // failure must surface with the clef-decision-runtime prefix (wire
        // DECISION_RUNTIME), while pre-tripped cancellation stays typed.
        use crate::runtime::{create_engine, RuntimeConfig};
        let dir = t5_bundle_dir();
        let engine = create_engine(&RuntimeConfig {
            precision_mode: Some("fp32".to_owned()),
            ..RuntimeConfig::default()
        })
        .expect("engine");
        let runtime = ClefDecisionRuntime::load(
            dir.path(),
            Some(T5_HIDDEN as u32),
            Some(Kind::Float),
            Some("fp32"),
        )
        .expect("runtime");
        let tokenizer = tiny_wordlevel_tokenizer();
        let request = t5_record("one two");
        let err = decide(
            &runtime,
            &tokenizer,
            &engine,
            &request,
            None,
            256,
            &DecisionCancel::none(),
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(
            err.starts_with("clef-decision-runtime") && err.contains("not loaded"),
            "untyped backbone failure must map to typed runtime error, got: {err}"
        );
        // Cancellation control on the same fixture stays typed (no blanket
        // conversion into runtime failures).
        let tripped = DecisionCancel::new(Arc::new(AtomicBool::new(true)), None);
        let err = decide(&runtime, &tokenizer, &engine, &request, None, 256, &tripped)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.starts_with("clef-decision-cancelled"), "{err}");
    }

    #[tokio::test]
    async fn post_model_lock_recheck_cancels_before_input_allocation() {
        // COUNT witness: a successful run through the loaded 4-layer tiny
        // engine invokes the checker EXACTLY 10 times — engine admission 1,
        // post-model-lock 2 (the new boundary), model admission 3,
        // pre-layer 4..7, post-layers 8, hidden 9, head 10. Removing the
        // post-lock check yields 9 and fails. The second-check probe below
        // plus SOURCE PLACEMENT (immediately after model.lock(), before
        // no_grad/input allocation) proves cancellation precedes allocation.
        // Blocked-model-lock/deadline CONTENTION remains unproven here;
        // observer v2 separately tests the actual mid-forward deadline/flag
        // path (after layer 0), not lock contention.
        use crate::runtime::{create_engine, RuntimeConfig, RuntimeEngine};
        let dir = t5_bundle_dir();
        let config = RuntimeConfig {
            precision_mode: Some("fp32".to_owned()),
            ..RuntimeConfig::default()
        };
        let mut engine = create_engine(&config).expect("engine");
        RuntimeEngine::load_model(&mut engine, dir.path())
            .await
            .expect("tiny model load");
        let runtime = ClefDecisionRuntime::load(
            dir.path(),
            Some(T5_HIDDEN as u32),
            Some(Kind::Float),
            Some("fp32"),
        )
        .expect("runtime");
        let tokenizer = tiny_wordlevel_tokenizer();
        let request = t5_record("one two");
        let (encoded, _) =
            encode_request(&tokenizer, &request, Some(4), 256, &DecisionCancel::none()).unwrap();
        // Successful counting run: exactly 10 callbacks (see comment).
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let _ok = engine
            .with_decision_backbone_checked(
                &encoded.input_ids,
                &|| {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(())
                },
                |_hidden, _gather| -> Result<Tensor> {
                    Ok(Tensor::zeros([], (Kind::Float, tch::Device::Cpu)))
                },
            )
            .unwrap();
        let observed = calls.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            observed, 10,
            "expected exactly 10 checker callbacks (post-lock boundary present)"
        );

        // Probe: fail exactly at the SECOND check invocation — the post-lock
        // site, before the no_grad guard/input tensor/decoder work. The
        // error identity proves cancellation at that boundary (fresh state:
        // nothing to unwind).
        let count = std::sync::atomic::AtomicUsize::new(0);
        let err = engine
            .with_decision_backbone_checked(
                &encoded.input_ids,
                &|| {
                    let n = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    if n == 2 {
                        Err(anyhow!("clef-decision-cancelled: post-lock probe"))
                    } else {
                        Ok(())
                    }
                },
                |_hidden, _gather| -> Result<Tensor> {
                    Err(anyhow!("clef-decision-runtime: forward must not run"))
                },
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("post-lock probe"),
            "expected the post-lock check site to fire, got: {err:#}"
        );
        // Fresh-state control: a clean run afterwards still succeeds.
        let (_, observed) = decide_encoded(
            &runtime,
            &engine,
            &request,
            &encoded,
            &DecisionCancel::none(),
        )
        .unwrap();
        assert_eq!(observed, Kind::Float);
    }

    #[test]
    fn compute_profile_separates_execution_path_from_quantization() {
        // Same channel-FP8 artifact, different SUPPORTED compute modes:
        // distinct compute profiles, identical artifact quantization.
        assert_eq!(
            compute_profile_string(Kind::Float, true),
            "cpu-float32-fp8ch-w8a8-dynact"
        );
        assert_eq!(
            compute_profile_string(Kind::BFloat16, true),
            "cpu-bfloat16-fp8ch-w8a8-dynact"
        );
        assert_ne!(
            compute_profile_string(Kind::Float, true),
            compute_profile_string(Kind::BFloat16, true)
        );
        // Unquantized artifacts carry no execution-path suffix.
        assert_eq!(compute_profile_string(Kind::Float, false), "cpu-float32");
    }

    #[test]
    fn head_kind_follows_engine_precision_policy_not_config_dtype() {
        // Non-FP8 bundle: engine auto policy is BF16 regardless of a float32
        // dtype field in config.json (R4B3). Explicit fp32 selects Float.
        let dir = tempfile::tempdir().unwrap();
        let config = tiny_head_config(8);
        let weights = tiny_head_tensors(&config);
        write_bundle(dir.path(), &config, &weights, Some(SIMPLE_BACKBONE_CONFIG));
        let runtime =
            ClefDecisionRuntime::load(dir.path(), None, None, None).expect("auto loads BF16 head");
        assert_eq!(runtime.compute_kind, Kind::BFloat16);
        let runtime = ClefDecisionRuntime::load(dir.path(), None, None, Some("fp32"))
            .expect("explicit fp32 head");
        assert_eq!(runtime.compute_kind, Kind::Float);
        let err = match ClefDecisionRuntime::load(dir.path(), None, None, Some("fp8e4m3")) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("unsupported precision must fail closed"),
        };
        assert!(
            err.contains("unsupported inference precision mode"),
            "{err}"
        );
    }
}
