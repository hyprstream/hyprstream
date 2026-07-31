//! KV-cache compatibility fingerprint and rejection policy.
//!
//! *Epic [#1276](https://github.com/hyprstream/hyprstream/issues/1276),
//! issue [#1277](https://github.com/hyprstream/hyprstream/issues/1277).*
//!
//! KV states are a pure function of (a) the weights that produced them and
//! (b) every model/runtime parameter that shapes the attention computation.
//! Reusing KV produced under one configuration inside a request running under
//! another is a silent correctness bug — the cached keys/values belong to a
//! different function. This module defines the **authoritative identity** of
//! that function as a versioned [`KvCompatDescriptor`], reduces it to a stable
//! content-addressed [`KvCompatFingerprint`], and exposes a [`check_compatibility`]
//! rejection policy: a cache whose descriptor differs in *any* authoritative
//! field is rejected (its KV is discarded and recomputed).
//!
//! This is the foundational lane item — every later reusable/durable lookup
//! (cross-session prefix index, snapshot codec, durable store) consumes the
//! fingerprint defined here. The module deliberately owns **no** `tch`/Cap'n
//! Proto dependency: it is pure data + hashing, so it is fully unit-testable
//! without a loaded model. Boundary types (`KVQuantType`, `ModelConfig`) are
//! converted into the self-contained enums below at the engine seam.
//!
//! ## What participates (authoritative inputs)
//!
//! Every field of [`KvCompatDescriptor`] is derived from the model/runtime state
//! that actually produced the KV tensors — never from a human label:
//!
//! - **Effective weights** — base-weight **content digest** + adapter/delta
//!   generation (the live `lora_generation` actually applied to the request).
//! - **Model identity** — name, architecture (`model_type`), config version.
//! - **Attention geometry** — layers, KV heads, attention heads, head dim,
//!   hidden size (determines K/V tensor *shape*).
//! - **Effective attention behavior** — QK-norm, partial-rotary dimension,
//!   per-layer local/global layout (`layer_types`), and hybrid/linear-attention
//!   dimensions (change K/V *values*, not just shape).
//! - **Position encoding** — RoPE theta, RoPE scaling, max position embeddings
//!   (determines K *values* post-RoPE).
//! - **Compute dtype** — the dtype the model computes/stores K/V in.
//! - **KV storage encoding** — quantization mode in effect.
//! - **Block geometry** — `BLOCK_SIZE` for paged caches.
//! - **Context cap** — the effective max context actually bounding the cache.
//! - **Tokenizer identity** — vocab size + a digest of the tokenizer bytes
//!   (cached token ids are only meaningful relative to one tokenizer).
//! - **Format version** — the wire/layout version of the fingerprint itself.
//!
//! ## Authority and fail-closed reuse
//!
//! A descriptor is only usable to bless a cache when every authoritative axis is
//! populated ([`KvCompatDescriptor::is_authoritatively_complete`]): the
//! base-weight content digest and the tokenizer hash. If the loader cannot
//! obtain the base digest, or tokenizer hashing fails, the engine declines to
//! mint an expected identity and the cache boundary **fails closed** — KV is
//! recomputed rather than reused under a guess ([`decide_cache_reuse`]). An
//! unstamped but populated cache is likewise discarded, never silently adopted.
//!
//! ## Relation to TTT delta invalidation
//!
//! The registry's existing push-based [`invalidate_for_tenant`][crate::runtime::kv_cache::KVCacheRegistry::invalidate_for_tenant]
//! clears a tenant's caches when that tenant's LoRA delta changes. The
//! fingerprint is the complementary **pull-based** guard checked at reuse time;
//! it does not replace the push path. The descriptor carries an
//! `adapter_generation` slot that a live counter (issue #1260) will feed so the
//! pull path also covers per-request delta changes; until then, delta staleness
//! is handled by the push path, and the fingerprint covers every *other* axis.

use sha2::{Digest, Sha256};

/// Wire/layout format version of the compatibility fingerprint.
///
/// Bumped whenever the set of fields that participate in the fingerprint, or
/// their canonical encoding, changes in a backward-incompatible way. A cache
/// produced (or persisted) under an older format version is **never** reused —
/// the rejection policy treats a format-version mismatch as a hard reject.
///
/// v3 adds the remaining behavior-affecting effective model configuration:
/// RMS/LayerNorm epsilon, activation/embedding/attention scaling, MoE routing,
/// and local-RoPE/sliding-window settings. Two runs with identical weight bytes
/// and coarse geometry can still produce different KV when any of these differ,
/// so they are authoritative for reuse.
///
/// v2 added `use_qk_norm`, partial rotary dimension, attention layer layout,
/// and linear-attention dimensions (K/V-value-affecting fields below coarse
/// geometry).
pub const KV_COMPAT_FORMAT_VERSION: u32 = 3;

/// Default tokens-per-block for paged KV storage.
///
/// This mirrors [`crate::runtime::kv_cache::BLOCK_SIZE`]; the two are asserted
/// equal by a unit test so the const lives here without pulling the `tch`-backed
/// `kv_cache` module into this module's dependency graph.
pub const KV_BLOCK_SIZE_DEFAULT: usize = 256;

// ===========================================================================
// Effective-weights identity
// ===========================================================================

/// Authoritative identity of the effective weights (base + adapter/delta) that
/// produced a cache's KV.
///
/// Two caches are reuse-compatible only if their KV was produced under the same
/// effective weights. This is split from the structural descriptor so a weight
/// change (e.g. loading/unloading an adapter) is a first-class mismatch reason
/// distinct from a geometry or dtype change.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct WeightIdentity {
    /// Authoritative identity of the base model weights — a content digest of
    /// the weight shards actually resolved by the loader (today a SHA-256 over
    /// the sorted `.safetensors` shard bytes; a git2db content OID when one is
    /// associated with the loaded worktree, issue #1285). This is **never** a
    /// human/path label: two snapshots sharing a basename but differing in
    /// content diverge here. Empty when no authoritative digest could be
    /// obtained, in which case the descriptor is treated as non-authoritative
    /// and reuse fails closed (see [`KvCompatDescriptor::is_authoritatively_complete`]).
    pub base_revision: String,
    /// Adapter/delta generation active when the KV was produced. `0` means base
    /// weights only (no adapter). Populated at the per-request reuse boundary
    /// from the live `lora_generation` counter actually applied to the request,
    /// so a re-adapted model is detected as incompatible (issue #1260).
    pub adapter_generation: u64,
}

impl WeightIdentity {
    /// Base weights only, no adapter (generation 0).
    pub fn base_only(base_revision: impl Into<String>) -> Self {
        Self {
            base_revision: base_revision.into(),
            adapter_generation: 0,
        }
    }

    /// Effective weights = `base` plus an adapter/delta at `generation`.
    pub fn with_adapter(mut self, generation: u64) -> Self {
        self.adapter_generation = generation;
        self
    }
}

/// Mixture-of-experts routing configuration. Different routing changes which
/// expert FFN each token traverses, and therefore the residual — and thus later
/// K/V — feeding downstream attention. A dense model (the default) is all-zero.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct MoeRouting {
    /// Whether this is a MoE model (e.g. `model_type` contains `"moe"`).
    pub is_moe: bool,
    /// Total number of experts.
    pub num_experts: usize,
    /// Experts activated per token (top-k routing).
    pub num_experts_per_tok: usize,
    /// Per-expert FFN intermediate size.
    pub moe_intermediate_size: usize,
    /// Shared-expert intermediate size (Qwen3.5 MoE), if any.
    pub shared_expert_intermediate_size: usize,
}

// ===========================================================================
// Scalar identity enums (own types, decoupled from codegen/tch)
// ===========================================================================

/// Compute dtype the model stores K/V in.
///
/// Owned by this module (rather than reusing `tch::Kind`) so the fingerprint's
/// stability does not depend on the `tch` version, and so the module stays
/// `tch`-free. Unrecognized dtypes fall through to [`KvDtype::Other`], keyed by
/// a canonicalized name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvDtype {
    Float64,
    Float32,
    Float16,
    #[default]
    BFloat16,
    /// Any other dtype, identified by its canonicalized (trimmed, lowercased) name.
    Other(String),
}

impl KvDtype {
    /// Parse a dtype from a HuggingFace `dtype`/`torch_dtype` string, case- and
    /// alias-insensitive.
    pub fn from_name(s: &str) -> Self {
        let n = s.trim().to_ascii_lowercase();
        match n.as_str() {
            "float64" | "double" | "f64" => KvDtype::Float64,
            "float32" | "float" | "f32" => KvDtype::Float32,
            "float16" | "half" | "fp16" | "f16" => KvDtype::Float16,
            "bfloat16" | "bf16" => KvDtype::BFloat16,
            other => KvDtype::Other(other.to_owned()),
        }
    }

    /// Canonical lowercase name used both for [`Display`](std::fmt::Display) and
    /// as the hashed representation.
    pub fn as_canonical_str(&self) -> &str {
        match self {
            KvDtype::Float64 => "float64",
            KvDtype::Float32 => "float32",
            KvDtype::Float16 => "float16",
            KvDtype::BFloat16 => "bfloat16",
            KvDtype::Other(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for KvDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_canonical_str())
    }
}

/// KV-cache storage encoding / quantization mode.
///
/// Mirrors the Cap'n Proto `KVQuantType` (`none`/`int8`/`nf4`/`fp4`) but is
/// owned here so the fingerprint is independent of schema codegen. Convert at
/// the engine boundary with a `match`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvQuantMode {
    /// Full precision (FP16/BF16).
    #[default]
    None,
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit NormalFloat.
    Nf4,
    /// 4-bit FloatingPoint.
    Fp4,
}

impl KvQuantMode {
    /// Canonical lowercase name used for display and hashing.
    pub fn as_canonical_str(&self) -> &'static str {
        match self {
            KvQuantMode::None => "none",
            KvQuantMode::Int8 => "int8",
            KvQuantMode::Nf4 => "nf4",
            KvQuantMode::Fp4 => "fp4",
        }
    }
}

impl std::fmt::Display for KvQuantMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_canonical_str())
    }
}

/// RoPE scaling config, with the factor canonicalized to its IEEE-754 bit
/// pattern so the descriptor is byte-stable (no `f32` equality pitfalls).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RopeScalingFp {
    /// Canonicalized (trimmed, lowercased) RoPE type, e.g. `"linear"`.
    pub rope_type: String,
    /// `f32::to_bits` of the scaling factor.
    pub factor_bits: u32,
}

impl RopeScalingFp {
    /// Construct from a RoPE type string and a scaling factor.
    pub fn new(rope_type: &str, factor: f32) -> Self {
        Self {
            rope_type: rope_type.trim().to_ascii_lowercase(),
            factor_bits: factor.to_bits(),
        }
    }

    /// The scaling factor, recovered from its bit pattern.
    pub fn factor(&self) -> f32 {
        f32::from_bits(self.factor_bits)
    }
}

// ===========================================================================
// Descriptor
// ===========================================================================

/// The complete set of authoritative inputs that determine whether KV states
/// computed under one configuration can be reused under another.
///
/// Two caches are compatible **iff** their descriptors are equal in every
/// field. [`KvCompatFingerprint`] is the content-addressed digest of a canonical
/// serialization, so equality of fingerprints implies equality of descriptors
/// (modulo the astronomically unlikely SHA-256 collision). The descriptor is
/// `Clone + Eq + Hash` and small; the engine retains one (the "expected"
/// identity of the loaded model) and caches store a [`KvCompatFingerprint`]
/// stamped at first use.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct KvCompatDescriptor {
    /// Fingerprint format/layout version ([`KV_COMPAT_FORMAT_VERSION`]).
    pub format_version: u32,

    /// Effective weights (base + adapter/delta) identity.
    pub weights: WeightIdentity,

    /// Loaded model identity (path stem / name).
    pub model_name: String,
    /// HuggingFace `model_type` (e.g. `"llama"`, `"qwen2"`).
    pub architecture: String,
    /// Parsed config `version`.
    pub model_version: u32,

    // --- attention geometry (determines K/V tensor shape) ---
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,

    // --- effective attention behavior (changes K/V *values*, not just shape) ---
    /// QK-norm applied to queries/keys (Gemma3, Qwen3 families). Off vs. on
    /// normalizes K and changes every downstream value.
    pub use_qk_norm: bool,
    /// `f32::to_bits` of the partial-rotary factor. Determines `rotary_dim`
    /// (`head_dim * factor`), i.e. how much of each key is rotated by RoPE —
    /// a different factor rotates a different slice of K. `None` = full rotary.
    pub partial_rotary_factor_bits: Option<u32>,
    /// Per-layer attention type in layer order (e.g. `"global"`/`"local"`,
    /// `"full_attention"`/`"linear_attention"`). The local/global layout and any
    /// local-RoPE behavior change which tokens each layer attends to and thus its
    /// K/V. Stored lowercased; canonicalized at hash time.
    pub layer_types: Vec<String>,
    /// Linear/hybrid-attention state dimensions (Qwen3.5 GatedDeltaNet etc.).
    /// Non-zero on hybrid models; a different dimension changes the per-layer
    /// recurrent state that feeds later K/V.
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_conv_kernel_dim: usize,

    // --- effective model config (changes hidden states / downstream K/V) ---
    /// `f32::to_bits` of `rms_norm_eps`. The RMS epsilon changes the normalized
    /// inputs to Q/K and every layer's residual, so it changes all downstream
    /// K/V. Stored as bits for byte-stable hashing (no `f32` equality pitfalls).
    pub rms_norm_eps_bits: u32,
    /// `f32::to_bits` of `layer_norm_eps` when the architecture uses LayerNorm
    /// instead of RMSNorm. `None` when LayerNorm is not in effect.
    pub layer_norm_eps_bits: Option<u32>,
    /// Activation function (e.g. `"silu"`, `"gelu_pytorch_tanh"`). A different
    /// activation changes every FFN output and thus the residual feeding later
    /// K/V. Canonicalized (trimmed, lowercased) at construction.
    pub hidden_activation: String,
    /// Whether embeddings are scaled before entering the stack (changes all
    /// layer inputs and therefore all K/V).
    pub scale_embeddings: bool,
    /// `f32::to_bits` of `query_pre_attn_scalar` (Gemma attention logit scale),
    /// when set. Changes the attention scores and thus the per-layer output.
    pub query_pre_attn_scalar_bits: Option<u32>,
    /// MoE routing config. `is_moe == false` and all-zero for a dense model.
    pub moe: MoeRouting,
    /// Sliding-window size for local-attention layers (Gemma3), when set.
    /// Changes which tokens each local layer attends to and thus its K/V.
    pub sliding_window: Option<usize>,
    /// `f32::to_bits` of `rope_local_base_freq` (the RoPE theta used in local
    /// layers, Gemma3), when set. Changes local-layer post-RoPE key values.
    pub rope_local_base_freq_bits: Option<u32>,

    // --- position encoding (determines K values post-RoPE) ---
    /// `f32::to_bits` of `rope_theta`.
    pub rope_theta_bits: u32,
    /// RoPE scaling, if any.
    pub rope_scaling: Option<RopeScalingFp>,
    /// Model's `max_position_embeddings`.
    pub max_position_embeddings: usize,

    // --- dtype + KV storage encoding ---
    /// Compute dtype the model stores K/V in.
    pub dtype: KvDtype,
    /// KV storage encoding/quantization mode.
    pub kv_quant: KvQuantMode,
    /// Tokens per paged block (must match for paged caches).
    pub block_size: usize,
    /// Effective runtime context cap actually bounding the cache.
    pub max_context: usize,

    // --- tokenizer identity (cached token ids are per-tokenizer) ---
    pub vocab_size: usize,
    /// Hex digest of the tokenizer bytes, when known.
    pub tokenizer_hash: Option<String>,
}

impl KvCompatDescriptor {
    /// The RoPE theta, recovered from its bit pattern.
    pub fn rope_theta(&self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }

    /// Set the RoPE theta from an `f32` (canonicalized to bits).
    pub fn set_rope_theta(&mut self, theta: f32) {
        self.rope_theta_bits = theta.to_bits();
    }

    /// The partial-rotary factor, recovered from its bit pattern (`None` = full rotary).
    pub fn partial_rotary_factor(&self) -> Option<f32> {
        self.partial_rotary_factor_bits.map(f32::from_bits)
    }

    /// Set the partial-rotary factor from an `f32` (canonicalized to bits).
    pub fn set_partial_rotary_factor(&mut self, factor: Option<f32>) {
        self.partial_rotary_factor_bits = factor.map(f32::to_bits);
    }

    /// Set the RMS norm epsilon from an `f32` (canonicalized to bits).
    pub fn set_rms_norm_eps(&mut self, eps: f32) {
        self.rms_norm_eps_bits = eps.to_bits();
    }

    /// Set the LayerNorm epsilon from an `f32`, or `None`.
    pub fn set_layer_norm_eps(&mut self, eps: Option<f32>) {
        self.layer_norm_eps_bits = eps.map(f32::to_bits);
    }

    /// Set the query pre-attention scalar from an `f32`, or `None`.
    pub fn set_query_pre_attn_scalar(&mut self, scalar: Option<f32>) {
        self.query_pre_attn_scalar_bits = scalar.map(f32::to_bits);
    }

    /// Set the local-RoPE base frequency from an `f32`, or `None`.
    pub fn set_rope_local_base_freq(&mut self, freq: Option<f32>) {
        self.rope_local_base_freq_bits = freq.map(f32::to_bits);
    }

    /// True iff every authoritative identity axis is populated.
    ///
    /// A descriptor missing the base-weight digest or the tokenizer identity was
    /// built from a non-authoritative source and must **not** be used to bless a
    /// cache for reuse: the engine treats a non-authoritative expected identity
    /// as fail-closed (discard + recompute) rather than stamping a guess.
    pub fn is_authoritatively_complete(&self) -> bool {
        !self.weights.base_revision.is_empty() && self.tokenizer_hash.is_some()
    }

    /// Record the tokenizer identity (vocab size + digest of its bytes).
    ///
    /// Called by the engine once the tokenizer is loaded; the descriptor is
    /// finalized before any generation reads its fingerprint.
    pub fn set_tokenizer(&mut self, vocab_size: usize, hash: impl Into<Option<String>>) {
        self.vocab_size = vocab_size;
        self.tokenizer_hash = hash.into();
    }

    /// Content-addressed fingerprint of this descriptor.
    pub fn fingerprint(&self) -> KvCompatFingerprint {
        KvCompatFingerprint::of(self)
    }

    /// Rejection policy against an observed (cache-side) descriptor.
    ///
    /// Returns the first differing field in a fixed order, or `Ok(())` if the
    /// descriptors are fully compatible. See [`check_compatibility`].
    pub fn check_against(&self, observed: &KvCompatDescriptor) -> Result<(), KvCompatMismatch> {
        check_compatibility(self, observed)
    }

    /// Feed a canonical, byte-stable serialization of the descriptor into `hash`.
    ///
    /// Every variable-length field is length-prefixed and every `Option` is
    /// tag-prefixed so distinct descriptors never collide (e.g. `"ab"+"c"` is
    /// distinguishable from `"a"+"bc"`). Little-endian fixed widths for scalars.
    /// This encoding is what makes persisted fingerprints portable across builds.
    fn write_canonical(&self, hash: &mut Sha256) {
        put_u32(hash, self.format_version);

        put_str(hash, &self.weights.base_revision);
        put_u64(hash, self.weights.adapter_generation);

        put_str(hash, &self.model_name);
        put_str(hash, &self.architecture);
        put_u32(hash, self.model_version);

        put_usize(hash, self.num_hidden_layers);
        put_usize(hash, self.num_attention_heads);
        put_usize(hash, self.num_key_value_heads);
        put_usize(hash, self.head_dim);
        put_usize(hash, self.hidden_size);

        // Effective attention behavior.
        put_bool(hash, self.use_qk_norm);
        put_opt_u32(hash, self.partial_rotary_factor_bits);
        put_str_vec(hash, &self.layer_types);
        put_usize(hash, self.linear_key_head_dim);
        put_usize(hash, self.linear_value_head_dim);
        put_usize(hash, self.linear_num_key_heads);
        put_usize(hash, self.linear_num_value_heads);
        put_usize(hash, self.linear_conv_kernel_dim);

        // Effective model config (changes hidden states / downstream K/V).
        put_u32(hash, self.rms_norm_eps_bits);
        put_opt_u32(hash, self.layer_norm_eps_bits);
        put_str(hash, &canon_str(&self.hidden_activation));
        put_bool(hash, self.scale_embeddings);
        put_opt_u32(hash, self.query_pre_attn_scalar_bits);
        put_bool(hash, self.moe.is_moe);
        put_usize(hash, self.moe.num_experts);
        put_usize(hash, self.moe.num_experts_per_tok);
        put_usize(hash, self.moe.moe_intermediate_size);
        put_usize(hash, self.moe.shared_expert_intermediate_size);
        put_opt_usize(hash, self.sliding_window);
        put_opt_u32(hash, self.rope_local_base_freq_bits);

        put_u32(hash, self.rope_theta_bits);
        match &self.rope_scaling {
            Some(rs) => {
                hash.update([1u8]);
                put_str(hash, &rs.rope_type);
                put_u32(hash, rs.factor_bits);
            }
            None => hash.update([0u8]),
        }
        put_usize(hash, self.max_position_embeddings);

        put_str(hash, self.dtype.as_canonical_str());
        put_str(hash, self.kv_quant.as_canonical_str());
        put_usize(hash, self.block_size);
        put_usize(hash, self.max_context);

        put_usize(hash, self.vocab_size);
        put_opt_str(hash, self.tokenizer_hash.as_deref());
    }
}

// ===========================================================================
// Fingerprint
// ===========================================================================

/// Content-addressed digest (SHA-256) of a [`KvCompatDescriptor`].
///
/// This is what gets stamped on a cache and compared at reuse time — the hot
/// path checks `cache_fp == expected_fp`, a 32-byte compare. Two caches are
/// reuse-compatible iff their fingerprints are equal.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KvCompatFingerprint([u8; 32]);

impl KvCompatFingerprint {
    /// Compute the fingerprint of a descriptor.
    pub fn of(desc: &KvCompatDescriptor) -> Self {
        let mut hash = Sha256::new();
        desc.write_canonical(&mut hash);
        let digest = hash.finalize();
        // Copy rather than `.into()` so this is stable across generic-array
        // versions (digest output is always exactly 32 bytes for SHA-256).
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        Self(out)
    }

    /// The raw 32-byte digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex encoding of the digest.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// True iff a cache carrying this fingerprint is reuse-compatible with a
    /// request whose expected fingerprint is `expected`.
    pub fn is_compatible_with(&self, expected: &KvCompatFingerprint) -> bool {
        self.0 == expected.0
    }
}

impl std::fmt::Display for KvCompatFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl std::fmt::Debug for KvCompatFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KvCompatFingerprint({})", self.to_hex())
    }
}

// ===========================================================================
// Rejection policy
// ===========================================================================

/// Why a cache was rejected for reuse, naming the first differing authoritative
/// field in a fixed comparison order. Produced by [`check_compatibility`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvCompatMismatch {
    FormatVersion {
        expected: u32,
        observed: u32,
    },
    Weights {
        expected: WeightIdentity,
        observed: WeightIdentity,
    },
    ModelName {
        expected: String,
        observed: String,
    },
    Architecture {
        expected: String,
        observed: String,
    },
    ModelVersion {
        expected: u32,
        observed: u32,
    },
    Geometry {
        field: &'static str,
        expected: usize,
        observed: usize,
    },
    UseQkNorm {
        expected: bool,
        observed: bool,
    },
    PartialRotaryFactor {
        expected_bits: Option<u32>,
        observed_bits: Option<u32>,
    },
    LayerTypes {
        expected: Vec<String>,
        observed: Vec<String>,
    },
    /// Effective model config (changes hidden states / downstream K/V).
    RmsNormEps {
        expected_bits: u32,
        observed_bits: u32,
    },
    LayerNormEps {
        expected_bits: Option<u32>,
        observed_bits: Option<u32>,
    },
    HiddenActivation {
        expected: String,
        observed: String,
    },
    ScaleEmbeddings {
        expected: bool,
        observed: bool,
    },
    QueryPreAttnScalar {
        expected_bits: Option<u32>,
        observed_bits: Option<u32>,
    },
    MoeRouting {
        expected: MoeRouting,
        observed: MoeRouting,
    },
    SlidingWindow {
        expected: Option<usize>,
        observed: Option<usize>,
    },
    RopeLocalBaseFreq {
        expected_bits: Option<u32>,
        observed_bits: Option<u32>,
    },
    RopeTheta {
        expected_bits: u32,
        observed_bits: u32,
    },
    RopeScaling {
        expected: Option<RopeScalingFp>,
        observed: Option<RopeScalingFp>,
    },
    MaxPositionEmbeddings {
        expected: usize,
        observed: usize,
    },
    Dtype {
        expected: KvDtype,
        observed: KvDtype,
    },
    KvQuant {
        expected: KvQuantMode,
        observed: KvQuantMode,
    },
    BlockSize {
        expected: usize,
        observed: usize,
    },
    MaxContext {
        expected: usize,
        observed: usize,
    },
    VocabSize {
        expected: usize,
        observed: usize,
    },
    TokenizerHash {
        expected: Option<String>,
        observed: Option<String>,
    },
}

impl std::fmt::Display for KvCompatMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCompatMismatch::FormatVersion { expected, observed } => write!(
                f,
                "KV compat mismatch: format_version (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::Weights { expected, observed } => write!(
                f,
                "KV compat mismatch: effective weights (expected {expected:?}, observed {observed:?})"
            ),
            KvCompatMismatch::ModelName { expected, observed } => write!(
                f,
                "KV compat mismatch: model name (expected {expected:?}, observed {observed:?})"
            ),
            KvCompatMismatch::Architecture { expected, observed } => write!(
                f,
                "KV compat mismatch: architecture (expected {expected:?}, observed {observed:?})"
            ),
            KvCompatMismatch::ModelVersion { expected, observed } => write!(
                f,
                "KV compat mismatch: model version (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::Geometry {
                field,
                expected,
                observed,
            } => write!(
                f,
                "KV compat mismatch: geometry field {field:?} (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::UseQkNorm { expected, observed } => write!(
                f,
                "KV compat mismatch: use_qk_norm (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::PartialRotaryFactor {
                expected_bits,
                observed_bits,
            } => write!(
                f,
                "KV compat mismatch: partial_rotary_factor (expected bits {expected_bits:?}, observed bits {observed_bits:?})"
            ),
            KvCompatMismatch::LayerTypes { expected, observed } => write!(
                f,
                "KV compat mismatch: layer_types (expected {expected:?}, observed {observed:?})"
            ),
            KvCompatMismatch::RmsNormEps {
                expected_bits,
                observed_bits,
            } => write!(
                f,
                "KV compat mismatch: rms_norm_eps (expected bits {expected_bits:#x}, observed bits {observed_bits:#x})"
            ),
            KvCompatMismatch::LayerNormEps {
                expected_bits,
                observed_bits,
            } => write!(
                f,
                "KV compat mismatch: layer_norm_eps (expected bits {expected_bits:?}, observed bits {observed_bits:?})"
            ),
            KvCompatMismatch::HiddenActivation { expected, observed } => write!(
                f,
                "KV compat mismatch: hidden_activation (expected {expected:?}, observed {observed:?})"
            ),
            KvCompatMismatch::ScaleEmbeddings { expected, observed } => write!(
                f,
                "KV compat mismatch: scale_embeddings (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::QueryPreAttnScalar {
                expected_bits,
                observed_bits,
            } => write!(
                f,
                "KV compat mismatch: query_pre_attn_scalar (expected bits {expected_bits:?}, observed bits {observed_bits:?})"
            ),
            KvCompatMismatch::MoeRouting { expected, observed } => write!(
                f,
                "KV compat mismatch: moe routing (expected {expected:?}, observed {observed:?})"
            ),
            KvCompatMismatch::SlidingWindow { expected, observed } => write!(
                f,
                "KV compat mismatch: sliding_window (expected {expected:?}, observed {observed:?})"
            ),
            KvCompatMismatch::RopeLocalBaseFreq {
                expected_bits,
                observed_bits,
            } => write!(
                f,
                "KV compat mismatch: rope_local_base_freq (expected bits {expected_bits:?}, observed bits {observed_bits:?})"
            ),
            KvCompatMismatch::RopeTheta {
                expected_bits,
                observed_bits,
            } => write!(
                f,
                "KV compat mismatch: rope_theta (expected bits {expected_bits:#x}, observed {observed_bits:#x})"
            ),
            KvCompatMismatch::RopeScaling { expected, observed } => write!(
                f,
                "KV compat mismatch: rope_scaling (expected {expected:?}, observed {observed:?})"
            ),
            KvCompatMismatch::MaxPositionEmbeddings { expected, observed } => write!(
                f,
                "KV compat mismatch: max_position_embeddings (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::Dtype { expected, observed } => write!(
                f,
                "KV compat mismatch: dtype (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::KvQuant { expected, observed } => write!(
                f,
                "KV compat mismatch: kv quant mode (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::BlockSize { expected, observed } => write!(
                f,
                "KV compat mismatch: block_size (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::MaxContext { expected, observed } => write!(
                f,
                "KV compat mismatch: max_context (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::VocabSize { expected, observed } => write!(
                f,
                "KV compat mismatch: vocab_size (expected {expected}, observed {observed})"
            ),
            KvCompatMismatch::TokenizerHash { expected, observed } => write!(
                f,
                "KV compat mismatch: tokenizer_hash (expected {expected:?}, observed {observed:?})"
            ),
        }
    }
}

impl std::error::Error for KvCompatMismatch {}

/// **Rejection policy**: a cache whose descriptor differs in *any* authoritative
/// field is rejected.
///
/// Compares `expected` (the current model/runtime identity) against `observed`
/// (the descriptor under which a cache's KV was produced), field by field in a
/// fixed order, returning the first mismatch. Returns `Ok(())` only when the
/// descriptors are fully compatible.
///
/// At runtime the engine usually compares [`KvCompatFingerprint`]s (a 32-byte
/// equality) rather than calling this; this function exists to produce a
/// structured, human-readable rejection reason (for logs, metrics, and tests)
/// and to pin the exact comparison contract the fingerprint must honor.
pub fn check_compatibility(
    expected: &KvCompatDescriptor,
    observed: &KvCompatDescriptor,
) -> Result<(), KvCompatMismatch> {
    if expected.format_version != observed.format_version {
        return Err(KvCompatMismatch::FormatVersion {
            expected: expected.format_version,
            observed: observed.format_version,
        });
    }
    if expected.weights != observed.weights {
        return Err(KvCompatMismatch::Weights {
            expected: expected.weights.clone(),
            observed: observed.weights.clone(),
        });
    }
    if expected.model_name != observed.model_name {
        return Err(KvCompatMismatch::ModelName {
            expected: expected.model_name.clone(),
            observed: observed.model_name.clone(),
        });
    }
    if expected.architecture != observed.architecture {
        return Err(KvCompatMismatch::Architecture {
            expected: expected.architecture.clone(),
            observed: observed.architecture.clone(),
        });
    }
    if expected.model_version != observed.model_version {
        return Err(KvCompatMismatch::ModelVersion {
            expected: expected.model_version,
            observed: observed.model_version,
        });
    }
    // Geometry: report the first diverging dimension.
    if expected.num_hidden_layers != observed.num_hidden_layers {
        return Err(geometry(
            "num_hidden_layers",
            expected.num_hidden_layers,
            observed.num_hidden_layers,
        ));
    }
    if expected.num_attention_heads != observed.num_attention_heads {
        return Err(geometry(
            "num_attention_heads",
            expected.num_attention_heads,
            observed.num_attention_heads,
        ));
    }
    if expected.num_key_value_heads != observed.num_key_value_heads {
        return Err(geometry(
            "num_key_value_heads",
            expected.num_key_value_heads,
            observed.num_key_value_heads,
        ));
    }
    if expected.head_dim != observed.head_dim {
        return Err(geometry("head_dim", expected.head_dim, observed.head_dim));
    }
    if expected.hidden_size != observed.hidden_size {
        return Err(geometry(
            "hidden_size",
            expected.hidden_size,
            observed.hidden_size,
        ));
    }
    // Effective attention behavior (changes K/V *values*, not just shape).
    if expected.use_qk_norm != observed.use_qk_norm {
        return Err(KvCompatMismatch::UseQkNorm {
            expected: expected.use_qk_norm,
            observed: observed.use_qk_norm,
        });
    }
    if expected.partial_rotary_factor_bits != observed.partial_rotary_factor_bits {
        return Err(KvCompatMismatch::PartialRotaryFactor {
            expected_bits: expected.partial_rotary_factor_bits,
            observed_bits: observed.partial_rotary_factor_bits,
        });
    }
    // `layer_types` is canonicalized (trimmed, lowercased) before hashing
    // (`put_str_vec`), so compare the canonical forms here too — aliases like
    // "GLOBAL"/"global" must be compatible in both the hot-path fingerprint and
    // this structured policy (#1277 review, hot-path/policy equivalence).
    {
        let exp_lt = canonical_layer_types(&expected.layer_types);
        let obs_lt = canonical_layer_types(&observed.layer_types);
        if exp_lt != obs_lt {
            return Err(KvCompatMismatch::LayerTypes {
                expected: exp_lt,
                observed: obs_lt,
            });
        }
    }
    if expected.linear_key_head_dim != observed.linear_key_head_dim {
        return Err(geometry(
            "linear_key_head_dim",
            expected.linear_key_head_dim,
            observed.linear_key_head_dim,
        ));
    }
    if expected.linear_value_head_dim != observed.linear_value_head_dim {
        return Err(geometry(
            "linear_value_head_dim",
            expected.linear_value_head_dim,
            observed.linear_value_head_dim,
        ));
    }
    if expected.linear_num_key_heads != observed.linear_num_key_heads {
        return Err(geometry(
            "linear_num_key_heads",
            expected.linear_num_key_heads,
            observed.linear_num_key_heads,
        ));
    }
    if expected.linear_num_value_heads != observed.linear_num_value_heads {
        return Err(geometry(
            "linear_num_value_heads",
            expected.linear_num_value_heads,
            observed.linear_num_value_heads,
        ));
    }
    if expected.linear_conv_kernel_dim != observed.linear_conv_kernel_dim {
        return Err(geometry(
            "linear_conv_kernel_dim",
            expected.linear_conv_kernel_dim,
            observed.linear_conv_kernel_dim,
        ));
    }
    // Effective model config — changes hidden states / downstream K/V even when
    // coarse geometry is identical (#1277 review finding F3).
    if expected.rms_norm_eps_bits != observed.rms_norm_eps_bits {
        return Err(KvCompatMismatch::RmsNormEps {
            expected_bits: expected.rms_norm_eps_bits,
            observed_bits: observed.rms_norm_eps_bits,
        });
    }
    if expected.layer_norm_eps_bits != observed.layer_norm_eps_bits {
        return Err(KvCompatMismatch::LayerNormEps {
            expected_bits: expected.layer_norm_eps_bits,
            observed_bits: observed.layer_norm_eps_bits,
        });
    }
    if canon_str(&expected.hidden_activation) != canon_str(&observed.hidden_activation) {
        return Err(KvCompatMismatch::HiddenActivation {
            expected: canon_str(&expected.hidden_activation),
            observed: canon_str(&observed.hidden_activation),
        });
    }
    if expected.scale_embeddings != observed.scale_embeddings {
        return Err(KvCompatMismatch::ScaleEmbeddings {
            expected: expected.scale_embeddings,
            observed: observed.scale_embeddings,
        });
    }
    if expected.query_pre_attn_scalar_bits != observed.query_pre_attn_scalar_bits {
        return Err(KvCompatMismatch::QueryPreAttnScalar {
            expected_bits: expected.query_pre_attn_scalar_bits,
            observed_bits: observed.query_pre_attn_scalar_bits,
        });
    }
    if expected.moe != observed.moe {
        return Err(KvCompatMismatch::MoeRouting {
            expected: expected.moe.clone(),
            observed: observed.moe.clone(),
        });
    }
    if expected.sliding_window != observed.sliding_window {
        return Err(KvCompatMismatch::SlidingWindow {
            expected: expected.sliding_window,
            observed: observed.sliding_window,
        });
    }
    if expected.rope_local_base_freq_bits != observed.rope_local_base_freq_bits {
        return Err(KvCompatMismatch::RopeLocalBaseFreq {
            expected_bits: expected.rope_local_base_freq_bits,
            observed_bits: observed.rope_local_base_freq_bits,
        });
    }
    if expected.rope_theta_bits != observed.rope_theta_bits {
        return Err(KvCompatMismatch::RopeTheta {
            expected_bits: expected.rope_theta_bits,
            observed_bits: observed.rope_theta_bits,
        });
    }
    if expected.rope_scaling != observed.rope_scaling {
        return Err(KvCompatMismatch::RopeScaling {
            expected: expected.rope_scaling.clone(),
            observed: observed.rope_scaling.clone(),
        });
    }
    if expected.max_position_embeddings != observed.max_position_embeddings {
        return Err(KvCompatMismatch::MaxPositionEmbeddings {
            expected: expected.max_position_embeddings,
            observed: observed.max_position_embeddings,
        });
    }
    if expected.dtype != observed.dtype {
        return Err(KvCompatMismatch::Dtype {
            expected: expected.dtype.clone(),
            observed: observed.dtype.clone(),
        });
    }
    if expected.kv_quant != observed.kv_quant {
        return Err(KvCompatMismatch::KvQuant {
            expected: expected.kv_quant,
            observed: observed.kv_quant,
        });
    }
    if expected.block_size != observed.block_size {
        return Err(KvCompatMismatch::BlockSize {
            expected: expected.block_size,
            observed: observed.block_size,
        });
    }
    if expected.max_context != observed.max_context {
        return Err(KvCompatMismatch::MaxContext {
            expected: expected.max_context,
            observed: observed.max_context,
        });
    }
    if expected.vocab_size != observed.vocab_size {
        return Err(KvCompatMismatch::VocabSize {
            expected: expected.vocab_size,
            observed: observed.vocab_size,
        });
    }
    if expected.tokenizer_hash != observed.tokenizer_hash {
        return Err(KvCompatMismatch::TokenizerHash {
            expected: expected.tokenizer_hash.clone(),
            observed: observed.tokenizer_hash.clone(),
        });
    }
    Ok(())
}

fn geometry(field: &'static str, expected: usize, observed: usize) -> KvCompatMismatch {
    KvCompatMismatch::Geometry {
        field,
        expected,
        observed,
    }
}

// ===========================================================================
// Reuse decision (fail-closed policy at the cache boundary)
// ===========================================================================

/// What to do with a cache encountered at the reuse boundary, produced by
/// [`decide_cache_reuse`]. The policy is **fail-closed**: it never blesses a
/// cache whose KV cannot be proven compatible with an authoritative identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheReuseDecision {
    /// Cache KV is provably compatible with an authoritative expected identity —
    /// reuse it as-is.
    Reuse,
    /// Cache is empty and carries no stamp, and an authoritative expected
    /// identity is available. Stamp it and keep the (empty) cache; prefill will
    /// populate it under that identity.
    StampFresh,
    /// The cache's KV cannot be proven compatible. This covers a stamp mismatch,
    /// an **unstamped but populated** cache (produced under an unknown identity),
    /// and the absence of any authoritative expected identity. Discard the KV and
    /// recompute; stamp with the authoritative identity if one is available.
    DiscardAndRecompute,
}

/// Decide whether a cache may be reused, given the authoritative expected
/// identity (if any), the stamp currently on the cache (if any), and whether the
/// cache provably carries no KV.
///
/// Decision matrix (every non-`Reuse` outcome discards KV — fail-closed):
///
/// | expected | stored  | empty | decision              |
/// |----------|---------|-------|-----------------------|
/// | Some(e)  | Some(s) | *     | `e == s` ? Reuse : DiscardAndRecompute |
/// | Some(_)  | None    | true  | StampFresh            |
/// | Some(_)  | None    | false | DiscardAndRecompute   |
/// | None     | *       | *     | DiscardAndRecompute   |
///
/// The unstamped-but-populated and no-authoritative-identity rows are the
/// fail-closed guards: a cache we cannot authoritatively attribute is never
/// reused, only recomputed.
pub fn decide_cache_reuse(
    expected: Option<&KvCompatFingerprint>,
    stored: Option<&KvCompatFingerprint>,
    cache_is_empty: bool,
) -> CacheReuseDecision {
    match (expected, stored) {
        (Some(exp), Some(stored)) if exp == stored => CacheReuseDecision::Reuse,
        (Some(_), None) if cache_is_empty => CacheReuseDecision::StampFresh,
        _ => CacheReuseDecision::DiscardAndRecompute,
    }
}

// ===========================================================================
// Canonical byte framing for stable hashing
// ===========================================================================

#[inline]
fn put_u32(hash: &mut Sha256, v: u32) {
    hash.update(v.to_le_bytes());
}

#[inline]
fn put_opt_u32(hash: &mut Sha256, v: Option<u32>) {
    match v {
        Some(x) => {
            hash.update([1u8]);
            put_u32(hash, x);
        }
        None => hash.update([0u8]),
    }
}

#[inline]
fn put_bool(hash: &mut Sha256, v: bool) {
    hash.update([if v { 1u8 } else { 0u8 }]);
}

#[inline]
fn put_u64(hash: &mut Sha256, v: u64) {
    hash.update(v.to_le_bytes());
}

#[inline]
fn put_usize(hash: &mut Sha256, v: usize) {
    // usize width is platform-dependent; hash it as a fixed u64 so fingerprints
    // are stable across 32-/64-bit builds.
    hash.update((v as u64).to_le_bytes());
}

#[inline]
fn put_opt_usize(hash: &mut Sha256, v: Option<usize>) {
    match v {
        Some(x) => {
            hash.update([1u8]);
            put_usize(hash, x);
        }
        None => hash.update([0u8]),
    }
}

#[inline]
fn put_str(hash: &mut Sha256, s: &str) {
    put_u64(hash, s.len() as u64);
    hash.update(s.as_bytes());
}

#[inline]
fn put_opt_str(hash: &mut Sha256, s: Option<&str>) {
    match s {
        Some(s) => {
            hash.update([1u8]);
            put_str(hash, s);
        }
        None => hash.update([0u8]),
    }
}

/// Canonicalize a string identifier: trim and ASCII-lowercase. Used everywhere
/// a string participates in the fingerprint or the structured comparison so
/// aliases ("GLOBAL"/"global") cannot disagree between the two (#1277 review).
#[inline]
fn canon_str(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}

/// Canonicalized (trimmed, lowercased) copy of a `layer_types` sequence, in
/// order. Used by both the hash ([`put_str_vec`]) and the structured comparison
/// ([`check_compatibility`]) so they agree on casing/whitespace aliases.
fn canonical_layer_types(v: &[String]) -> Vec<String> {
    v.iter().map(|s| canon_str(s)).collect()
}

/// Hash a `Vec<String>` as a length-prefixed sequence of canonicalized
/// (trimmed, lowercased) strings so layout casing/order changes are caught.
#[inline]
fn put_str_vec(hash: &mut Sha256, v: &[String]) {
    put_u64(hash, v.len() as u64);
    for s in v {
        put_str(hash, &canon_str(s));
    }
}

// ===========================================================================
// Authoritative base-weight content digest (#1277, finding F1)
// ===========================================================================
//
// The base-weight identity is a content digest over the *resolved* weight shards
// the loader actually builds tensors from — never a directory listing or a path
// label, and never a pointer stub. Two snapshots sharing a basename but
// differing in content diverge here. These pure helpers own the canonical
// framing; the engine seam (`TorchEngine::weights_content_digest`) resolves each
// shard's bytes (transparently resolving LFS/XET pointers) and feeds them here.

/// SHA-256 of a single shard's resolved content bytes.
pub fn shard_content_digest(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(h.finalize().as_slice());
    out
}

/// Canonical base-weight digest over a shard set: the shard count, then for each
/// shard a length-prefixed filename and that shard's content digest. Emitting
/// the count and length-prefixing every name makes distinct shard segmentations
/// collide-resistant (shard A+B split at byte N cannot masquerade as a single
/// shard, nor can two shard sets of equal total bytes share a digest). Shards
/// must be supplied sorted by name for determinism.
pub(crate) fn base_weight_digest_from_shards(shards: &[(String, [u8; 32])]) -> String {
    let mut h = Sha256::new();
    put_u64(&mut h, shards.len() as u64);
    for (name, digest) in shards {
        put_str(&mut h, name);
        h.update(digest);
    }
    hex::encode(h.finalize())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A representative, fully-populated descriptor used as the compatibility
    /// baseline in the tests below.
    fn baseline() -> KvCompatDescriptor {
        KvCompatDescriptor {
            format_version: KV_COMPAT_FORMAT_VERSION,
            weights: WeightIdentity::base_only("sha256:my-model-weights"),
            model_name: "my-model".to_owned(),
            architecture: "llama".to_owned(),
            model_version: 1,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            hidden_size: 4096,
            // Plain Llama: no QK-norm, full rotary, uniform layers, no linear attn.
            use_qk_norm: false,
            partial_rotary_factor_bits: None,
            layer_types: Vec::new(),
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_conv_kernel_dim: 0,
            // Effective model config (plain-Llama defaults; non-default below
            // diverge the fingerprint in every_authoritative_field_participates).
            rms_norm_eps_bits: 1e-5f32.to_bits(),
            layer_norm_eps_bits: None,
            hidden_activation: "silu".to_owned(),
            scale_embeddings: false,
            query_pre_attn_scalar_bits: None,
            moe: MoeRouting::default(),
            sliding_window: None,
            rope_local_base_freq_bits: None,
            rope_theta_bits: 10_000.0f32.to_bits(),
            rope_scaling: None,
            max_position_embeddings: 4096,
            dtype: KvDtype::BFloat16,
            kv_quant: KvQuantMode::None,
            block_size: KV_BLOCK_SIZE_DEFAULT,
            max_context: 4096,
            vocab_size: 32_000,
            tokenizer_hash: Some("deadbeef".to_owned()),
        }
    }

    /// Test helper: set one of the linear-attention dimensions by name.
    fn set_linear_dim(d: &mut KvCompatDescriptor, field: &str, val: usize) {
        match field {
            "linear_key_head_dim" => d.linear_key_head_dim = val,
            "linear_value_head_dim" => d.linear_value_head_dim = val,
            "linear_num_key_heads" => d.linear_num_key_heads = val,
            "linear_num_value_heads" => d.linear_num_value_heads = val,
            "linear_conv_kernel_dim" => d.linear_conv_kernel_dim = val,
            _ => panic!("unknown linear dim field {field}"),
        }
    }

    #[test]
    fn default_block_size_tracks_kv_cache_block_size() {
        // Keeps this module's const in sync with the paged-KV block geometry
        // without importing the tch-backed kv_cache module here.
        assert_eq!(KV_BLOCK_SIZE_DEFAULT, crate::runtime::kv_cache::BLOCK_SIZE);
    }

    #[test]
    fn fingerprint_is_deterministic() {
        let a = baseline().fingerprint();
        let b = baseline().fingerprint();
        assert_eq!(a, b, "identical descriptors must hash identically");
        assert_eq!(a.to_hex(), b.to_hex());
        assert_eq!(a.as_bytes().len(), 32);
    }

    #[test]
    fn fingerprint_is_clone_stable() {
        let d = baseline();
        assert_eq!(d.fingerprint(), d.clone().fingerprint());
    }

    /// Asserts that `mutator` produces a descriptor whose fingerprint differs
    /// from the baseline, and (when given) the expected mismatch variant.
    fn assert_field_diverges(
        label: &str,
        mutator: impl FnOnce(&mut KvCompatDescriptor),
        expected_mismatch: Option<KvCompatMismatch>,
    ) {
        let base_fp = baseline().fingerprint();
        let mut d = baseline();
        mutator(&mut d);
        let divergent_fp = d.fingerprint();
        assert_ne!(
            base_fp, divergent_fp,
            "{label}: flipping the field must change the fingerprint"
        );
        if let Some(expected) = expected_mismatch {
            let observed = baseline();
            let reason = check_compatibility(&d, &observed).err().unwrap_or_else(|| {
                panic!("{label}: expected a mismatch but descriptors compared equal")
            });
            assert_eq!(
                reason, expected,
                "{label}: mismatch variant/reason did not match expectation"
            );
        }
    }

    #[test]
    fn every_authoritative_field_participates() {
        // Weight identity.
        assert_field_diverges(
            "weights.base_revision",
            |d| d.weights.base_revision = "other-model".into(),
            Some(KvCompatMismatch::Weights {
                expected: WeightIdentity::base_only("other-model"),
                observed: WeightIdentity::base_only("sha256:my-model-weights"),
            }),
        );
        assert_field_diverges(
            "weights.adapter_generation",
            |d| d.weights.adapter_generation = 7,
            Some(KvCompatMismatch::Weights {
                expected: WeightIdentity::base_only("sha256:my-model-weights").with_adapter(7),
                observed: WeightIdentity::base_only("sha256:my-model-weights"),
            }),
        );
        // Model identity.
        assert_field_diverges("model_name", |d| d.model_name = "other".into(), None);
        assert_field_diverges("architecture", |d| d.architecture = "qwen2".into(), None);
        assert_field_diverges("model_version", |d| d.model_version = 2, None);
        // Geometry.
        assert_field_diverges(
            "num_hidden_layers",
            |d| d.num_hidden_layers = 24,
            Some(geometry("num_hidden_layers", 24, 32)),
        );
        assert_field_diverges(
            "num_attention_heads",
            |d| d.num_attention_heads = 16,
            Some(geometry("num_attention_heads", 16, 32)),
        );
        assert_field_diverges(
            "num_key_value_heads",
            |d| d.num_key_value_heads = 4,
            Some(geometry("num_key_value_heads", 4, 8)),
        );
        assert_field_diverges(
            "head_dim",
            |d| d.head_dim = 64,
            Some(geometry("head_dim", 64, 128)),
        );
        assert_field_diverges(
            "hidden_size",
            |d| d.hidden_size = 2048,
            Some(geometry("hidden_size", 2048, 4096)),
        );
        // Effective attention behavior (changes K/V *values*, not just shape).
        assert_field_diverges(
            "use_qk_norm",
            |d| d.use_qk_norm = true,
            Some(KvCompatMismatch::UseQkNorm {
                expected: true,
                observed: false,
            }),
        );
        assert_field_diverges(
            "partial_rotary_factor",
            |d| d.set_partial_rotary_factor(Some(0.25)),
            Some(KvCompatMismatch::PartialRotaryFactor {
                expected_bits: Some(0.25f32.to_bits()),
                observed_bits: None,
            }),
        );
        assert_field_diverges(
            "partial_rotary_factor None->Some spelling",
            |d| d.set_partial_rotary_factor(Some(1.0)),
            Some(KvCompatMismatch::PartialRotaryFactor {
                expected_bits: Some(1.0f32.to_bits()),
                observed_bits: None,
            }),
        );
        assert_field_diverges(
            "layer_types",
            |d| d.layer_types = vec!["global".to_owned(), "local".to_owned()],
            Some(KvCompatMismatch::LayerTypes {
                expected: vec!["global".to_owned(), "local".to_owned()],
                observed: Vec::new(),
            }),
        );
        // layer_types must be compared case-insensitively (canonicalized) — in
        // BOTH the fingerprint hash and the structured policy (#1277 MINOR).
        {
            let mut d = baseline();
            let fp0 = d.fingerprint();
            d.layer_types = vec!["GLOBAL".to_owned(), "Local".to_owned()];
            let mut d2 = baseline();
            d2.layer_types = vec!["global".to_owned(), "local".to_owned()];
            assert_eq!(
                d.fingerprint(),
                d2.fingerprint(),
                "layer_types casing must canonicalize before hashing"
            );
            // Equal fingerprints must imply a compatible structured policy:
            // casing/whitespace aliases must NOT report a LayerTypes mismatch.
            assert!(
                check_compatibility(&d, &d2).is_ok(),
                "layer_types casing alias must be policy-compatible, not just hash-equal"
            );
            assert_ne!(fp0, d.fingerprint(), "non-empty layer layout must diverge");
        }
        for (field, val) in [
            ("linear_key_head_dim", 512usize),
            ("linear_value_head_dim", 256usize),
            ("linear_num_key_heads", 8usize),
            ("linear_num_value_heads", 8usize),
            ("linear_conv_kernel_dim", 4usize),
        ] {
            let mut d = baseline();
            set_linear_dim(&mut d, field, val);
            let divergent_fp = d.fingerprint();
            assert_ne!(
                baseline().fingerprint(),
                divergent_fp,
                "{field}: changing the linear dim must change the fingerprint"
            );
            assert_eq!(
                check_compatibility(&d, &baseline()).unwrap_err(),
                geometry(field, val, 0),
                "{field}: mismatch variant"
            );
        }
        // Effective model config — each demonstrably changes hidden states /
        // downstream K/V without changing coarse geometry (#1277 F3).
        assert_field_diverges(
            "rms_norm_eps",
            |d| d.set_rms_norm_eps(1e-6),
            Some(KvCompatMismatch::RmsNormEps {
                expected_bits: 1e-6f32.to_bits(),
                observed_bits: 1e-5f32.to_bits(),
            }),
        );
        assert_field_diverges(
            "layer_norm_eps",
            |d| d.set_layer_norm_eps(Some(1e-5)),
            Some(KvCompatMismatch::LayerNormEps {
                expected_bits: Some(1e-5f32.to_bits()),
                observed_bits: None,
            }),
        );
        assert_field_diverges(
            "hidden_activation",
            |d| d.hidden_activation = "gelu_pytorch_tanh".into(),
            Some(KvCompatMismatch::HiddenActivation {
                expected: "gelu_pytorch_tanh".into(),
                observed: "silu".into(),
            }),
        );
        // hidden_activation must canonicalize (casing/whitespace) in both the
        // hash and the policy — an alias must be compatible, a real change must
        // not.
        {
            let mut d = baseline();
            d.hidden_activation = "  SiLU ".into();
            assert_eq!(
                d.fingerprint(),
                baseline().fingerprint(),
                "hidden_activation casing/whitespace must canonicalize"
            );
            assert!(check_compatibility(&d, &baseline()).is_ok());
        }
        assert_field_diverges(
            "scale_embeddings",
            |d| d.scale_embeddings = true,
            Some(KvCompatMismatch::ScaleEmbeddings {
                expected: true,
                observed: false,
            }),
        );
        assert_field_diverges(
            "query_pre_attn_scalar",
            |d| d.set_query_pre_attn_scalar(Some(256.0)),
            Some(KvCompatMismatch::QueryPreAttnScalar {
                expected_bits: Some(256.0f32.to_bits()),
                observed_bits: None,
            }),
        );
        // MoE routing — each field individually diverges. `is_moe` (bool) and the
        // four usize routing fields are exercised separately.
        {
            let mut d = baseline();
            d.moe.is_moe = true;
            assert_ne!(
                baseline().fingerprint(),
                d.fingerprint(),
                "moe.is_moe: setting MoE must change the fingerprint"
            );
            assert_eq!(
                check_compatibility(&d, &baseline()).unwrap_err(),
                KvCompatMismatch::MoeRouting {
                    expected: d.moe.clone(),
                    observed: baseline().moe,
                },
                "moe.is_moe: mismatch variant"
            );
        }
        for (field, val) in [
            ("num_experts", 64usize),
            ("num_experts_per_tok", 8usize),
            ("moe_intermediate_size", 2048usize),
            ("shared_expert_intermediate_size", 1024usize),
        ] {
            let mut d = baseline();
            match field {
                "num_experts" => d.moe.num_experts = val,
                "num_experts_per_tok" => d.moe.num_experts_per_tok = val,
                "moe_intermediate_size" => d.moe.moe_intermediate_size = val,
                "shared_expert_intermediate_size" => d.moe.shared_expert_intermediate_size = val,
                _ => unreachable!(),
            }
            let divergent_fp = d.fingerprint();
            assert_ne!(
                baseline().fingerprint(),
                divergent_fp,
                "{field}: MoE routing change must change the fingerprint"
            );
            assert_eq!(
                check_compatibility(&d, &baseline()).unwrap_err(),
                KvCompatMismatch::MoeRouting {
                    expected: d.moe.clone(),
                    observed: baseline().moe,
                },
                "{field}: MoE mismatch variant"
            );
        }
        assert_field_diverges(
            "sliding_window",
            |d| d.sliding_window = Some(1024),
            Some(KvCompatMismatch::SlidingWindow {
                expected: Some(1024),
                observed: None,
            }),
        );
        assert_field_diverges(
            "rope_local_base_freq",
            |d| d.set_rope_local_base_freq(Some(10_000.0)),
            Some(KvCompatMismatch::RopeLocalBaseFreq {
                expected_bits: Some(10_000.0f32.to_bits()),
                observed_bits: None,
            }),
        );
        // Position encoding.
        assert_field_diverges(
            "rope_theta",
            |d| d.set_rope_theta(1_000_000.0),
            Some(KvCompatMismatch::RopeTheta {
                expected_bits: 1_000_000.0f32.to_bits(),
                observed_bits: 10_000.0f32.to_bits(),
            }),
        );
        assert_field_diverges(
            "rope_scaling presence",
            |d| d.rope_scaling = Some(RopeScalingFp::new("linear", 2.0)),
            None,
        );
        assert_field_diverges(
            "rope_scaling factor",
            |d| {
                d.rope_scaling = Some(RopeScalingFp::new("linear", 2.0));
                d.rope_scaling = Some(RopeScalingFp::new("linear", 4.0));
            },
            None,
        );
        assert_field_diverges(
            "max_position_embeddings",
            |d| d.max_position_embeddings = 8192,
            None,
        );
        // Dtype + encoding.
        assert_field_diverges(
            "dtype",
            |d| d.dtype = KvDtype::Float16,
            Some(KvCompatMismatch::Dtype {
                expected: KvDtype::Float16,
                observed: KvDtype::BFloat16,
            }),
        );
        assert_field_diverges(
            "kv_quant",
            |d| d.kv_quant = KvQuantMode::Int8,
            Some(KvCompatMismatch::KvQuant {
                expected: KvQuantMode::Int8,
                observed: KvQuantMode::None,
            }),
        );
        assert_field_diverges(
            "block_size",
            |d| d.block_size = 128,
            Some(KvCompatMismatch::BlockSize {
                expected: 128,
                observed: 256,
            }),
        );
        assert_field_diverges(
            "max_context",
            |d| d.max_context = 2048,
            Some(KvCompatMismatch::MaxContext {
                expected: 2048,
                observed: 4096,
            }),
        );
        // Tokenizer.
        assert_field_diverges(
            "vocab_size",
            |d| d.vocab_size = 64_000,
            Some(KvCompatMismatch::VocabSize {
                expected: 64_000,
                observed: 32_000,
            }),
        );
        assert_field_diverges(
            "tokenizer_hash",
            |d| d.tokenizer_hash = Some("cafebabe".into()),
            Some(KvCompatMismatch::TokenizerHash {
                expected: Some("cafebabe".into()),
                observed: Some("deadbeef".into()),
            }),
        );
        assert_field_diverges("tokenizer_hash absent", |d| d.tokenizer_hash = None, None);
    }

    #[test]
    fn format_version_bump_is_hard_reject() {
        let mut observed = baseline();
        observed.format_version = 0; // an older/persisted format
        assert_eq!(
            check_compatibility(&baseline(), &observed),
            Err(KvCompatMismatch::FormatVersion {
                expected: KV_COMPAT_FORMAT_VERSION,
                observed: 0,
            })
        );
    }

    #[test]
    fn equal_descriptors_are_compatible() {
        assert!(check_compatibility(&baseline(), &baseline()).is_ok());
        assert!(baseline().check_against(&baseline()).is_ok());
    }

    #[test]
    fn fingerprint_equality_implies_descriptor_compatibility() {
        // The hot path compares fingerprints; assert it agrees with the
        // structured policy for both compatible and divergent cases.
        let a = baseline();
        let b = baseline();
        assert!(a.fingerprint().is_compatible_with(&b.fingerprint()));

        let mut c = baseline();
        c.kv_quant = KvQuantMode::Nf4;
        assert!(!a.fingerprint().is_compatible_with(&c.fingerprint()));
        assert!(check_compatibility(&a, &c).is_err());
    }

    #[test]
    fn kvdtype_is_case_and_alias_insensitive() {
        assert_eq!(KvDtype::from_name("BFLOAT16"), KvDtype::BFloat16);
        assert_eq!(KvDtype::from_name("  bf16 "), KvDtype::BFloat16);
        assert_eq!(KvDtype::from_name("float16"), KvDtype::Float16);
        assert_eq!(KvDtype::from_name("FP16"), KvDtype::Float16);
        assert_eq!(KvDtype::from_name("float32"), KvDtype::Float32);
        // Two descriptors differing only by dtype *spelling* must hash equally.
        let mut d1 = baseline();
        let mut d2 = baseline();
        d1.dtype = KvDtype::from_name("BFLOAT16");
        d2.dtype = KvDtype::from_name("bfloat16");
        assert_eq!(d1.fingerprint(), d2.fingerprint());
    }

    #[test]
    fn rope_scaling_canonicalizes_case() {
        assert_eq!(
            RopeScalingFp::new("Linear", 2.0),
            RopeScalingFp::new("linear", 2.0)
        );
        // Different factor -> different identity.
        assert_ne!(
            RopeScalingFp::new("linear", 2.0),
            RopeScalingFp::new("linear", 4.0)
        );
    }

    #[test]
    fn kvquant_canonical_strs_are_distinct() {
        let modes = [
            KvQuantMode::None,
            KvQuantMode::Int8,
            KvQuantMode::Nf4,
            KvQuantMode::Fp4,
        ];
        let strs: Vec<&str> = modes.iter().map(KvQuantMode::as_canonical_str).collect();
        let unique: std::collections::HashSet<&str> = strs.iter().copied().collect();
        assert_eq!(unique.len(), modes.len(), "quant mode names must be unique");
    }

    #[test]
    fn canonical_framing_disambiguates_concatenations() {
        // Length-prefixing must keep "ab"+"c" distinct from "a"+"bc".
        let mut h1 = Sha256::new();
        put_str(&mut h1, "ab");
        put_str(&mut h1, "c");
        let mut h2 = Sha256::new();
        put_str(&mut h2, "a");
        put_str(&mut h2, "bc");
        assert_ne!(
            h1.finalize().as_slice(),
            h2.finalize().as_slice(),
            "framing must prevent concatenation collisions"
        );
    }

    #[test]
    fn mismatch_reasons_are_displayable() {
        let cases = [
            KvCompatMismatch::FormatVersion {
                expected: 1,
                observed: 0,
            },
            KvCompatMismatch::Weights {
                expected: WeightIdentity::base_only("a"),
                observed: WeightIdentity::base_only("b"),
            },
            KvCompatMismatch::Geometry {
                field: "head_dim",
                expected: 128,
                observed: 64,
            },
            KvCompatMismatch::Dtype {
                expected: KvDtype::BFloat16,
                observed: KvDtype::Float16,
            },
            KvCompatMismatch::KvQuant {
                expected: KvQuantMode::None,
                observed: KvQuantMode::Int8,
            },
        ];
        for c in cases {
            let s = format!("{c}");
            assert!(
                s.contains("KV compat mismatch"),
                "display missing prefix: {s}"
            );
        }
    }

    #[test]
    fn weight_identity_with_adapter() {
        let w = WeightIdentity::base_only("m").with_adapter(3);
        assert_eq!(w.adapter_generation, 3);
        assert_eq!(w.base_revision, "m");
        // Adapter generation changes the fingerprint.
        let mut d = baseline();
        let fp0 = d.fingerprint();
        d.weights = WeightIdentity::base_only("sha256:my-model-weights").with_adapter(1);
        assert_ne!(fp0, d.fingerprint());
    }

    /// Golden digest: pins the exact canonical bytes of a fully-populated
    /// descriptor across builds/refactors. If this changes, it is an intentional
    /// format bump (`KV_COMPAT_FORMAT_VERSION`) — never a silent drift.
    #[test]
    fn golden_digest_is_pinned() {
        let fp = baseline().fingerprint();
        let hex = fp.to_hex();
        let expected = "2c1658e12b2935e6444e0503c35b9e36b3f2e941614d9078b445776eae5cc55c";
        assert_eq!(
            hex, expected,
            "KV-compat golden digest drifted; computed = {hex}. \
             If the canonical encoding changed intentionally, bump \
             KV_COMPAT_FORMAT_VERSION and update this pinned value."
        );
    }

    /// Two weight snapshots sharing a basename but differing in *content* must
    /// diverge in base identity (the #1277 "authoritative, not a label" rule).
    #[test]
    fn base_revision_content_diverges() {
        let mut a = baseline();
        let mut b = baseline();
        a.weights.base_revision = "sha256:snapshot-A".into();
        b.weights.base_revision = "sha256:snapshot-B".into();
        assert_ne!(
            a.fingerprint(),
            b.fingerprint(),
            "different base-weight content must produce different fingerprints"
        );
        // Same content → same fingerprint (stable).
        let mut c = baseline();
        c.weights.base_revision = "sha256:snapshot-A".into();
        assert_eq!(a.fingerprint(), c.fingerprint());
    }

    // ---- base-weight content digest producer (#1277, finding F1) ----
    //
    // These exercise the pure framing/digest helpers that the engine seam feeds
    // with the *resolved* shard bytes. The fail-closed enumeration + pointer
    // resolution is covered by `torch_engine::tests::weights_content_digest_*`.

    /// Helper: build a digested shard set (name + per-shard content digest) from
    /// raw bytes, mirroring what `TorchEngine::weights_content_digest` produces.
    fn digested(shards: &[(&str, &[u8])]) -> Vec<(String, [u8; 32])> {
        shards
            .iter()
            .map(|(n, b)| ((*n).to_owned(), shard_content_digest(b)))
            .collect()
    }

    #[test]
    fn shard_content_digest_is_deterministic_and_distinct() {
        let a = b"hello weights";
        let b = b"hello weights!";
        // Deterministic.
        assert_eq!(shard_content_digest(a), shard_content_digest(a));
        // Distinct content → distinct digest.
        assert_ne!(shard_content_digest(a), shard_content_digest(b));
        // 32 bytes.
        assert_eq!(shard_content_digest(a).len(), 32);
    }

    #[test]
    fn base_weight_digest_is_stable() {
        let s = digested(&[("model-00001-of-00002.safetensors", b"AAA"), ("model-00002-of-00002.safetensors", b"BBB")]);
        // Same sorted input → identical digest.
        assert_eq!(
            base_weight_digest_from_shards(&s),
            base_weight_digest_from_shards(&s)
        );
    }

    /// Same basenames, different content → different base-weight digest (the
    /// "authoritative, not a label" rule; two snapshots sharing a path stem but
    /// differing in bytes must diverge).
    #[test]
    fn base_weight_digest_same_basename_different_bytes_diverge() {
        let names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"];
        let a = digested(&[(names[0], b"content-A"), (names[1], b"content-B")]);
        let b = digested(&[(names[0], b"content-A!"), (names[1], b"content-B")]);
        assert_ne!(
            base_weight_digest_from_shards(&a),
            base_weight_digest_from_shards(&b),
            "different shard content must produce a different digest"
        );
    }

    /// Canonical framing must disambiguate shard boundaries: a single shard
    /// whose bytes are the concatenation of two others must NOT collide with the
    /// two-shard set, even though total bytes are equal.
    #[test]
    fn base_weight_digest_shard_boundary_disambiguates() {
        let two = digested(&[
            ("model-00001-of-00002.safetensors", b"PART-ONE-"),
            ("model-00002-of-00002.safetensors", b"PART-TWO"),
        ]);
        let one = digested(&[("model.safetensors", b"PART-ONE-PART-TWO")]);
        assert_ne!(
            base_weight_digest_from_shards(&two),
            base_weight_digest_from_shards(&one),
            "shard segmentation must not be ambiguous in the digest"
        );
    }

    /// A different shard count alone (same content, split differently) must
    /// diverge — the count is part of the canonical frame.
    #[test]
    fn base_weight_digest_shard_count_participates() {
        // Two shards each carrying the same bytes vs one shard carrying them.
        let two = digested(&[
            ("a-00001.safetensors", b"X"),
            ("a-00002.safetensors", b"X"),
        ]);
        let one = digested(&[("a.safetensors", b"X")]);
        assert_ne!(
            base_weight_digest_from_shards(&two),
            base_weight_digest_from_shards(&one)
        );
    }

    /// Renaming a shard (same content, different filename) must diverge — the
    /// length-prefixed name is part of the frame.
    #[test]
    fn base_weight_digest_filename_participates() {
        let a = digested(&[("model-00001.safetensors", b"X")]);
        let b = digested(&[("model-00002.safetensors", b"X")]);
        assert_ne!(
            base_weight_digest_from_shards(&a),
            base_weight_digest_from_shards(&b),
            "shard filename must participate in the digest"
        );
    }

    #[test]
    fn authority_completeness_gate() {
        // Baseline is fully authoritative.
        assert!(baseline().is_authoritatively_complete());

        // Missing base-weight digest → not authoritative (fail-closed).
        let mut no_base = baseline();
        no_base.weights.base_revision.clear();
        assert!(!no_base.is_authoritatively_complete());

        // Missing tokenizer identity → not authoritative (fail-closed).
        let mut no_tok = baseline();
        no_tok.tokenizer_hash = None;
        assert!(!no_tok.is_authoritatively_complete());
    }

    #[test]
    fn decide_cache_reuse_is_fail_closed() {
        use CacheReuseDecision::*;
        let exp = baseline().fingerprint();
        let other = {
            let mut d = baseline();
            d.kv_quant = KvQuantMode::Int8;
            d.fingerprint()
        };

        // Authoritative + matching stamp → reuse.
        assert_eq!(decide_cache_reuse(Some(&exp), Some(&exp), false), Reuse);
        assert_eq!(decide_cache_reuse(Some(&exp), Some(&exp), true), Reuse);

        // Authoritative + mismatching stamp → discard.
        assert_eq!(
            decide_cache_reuse(Some(&exp), Some(&other), false),
            DiscardAndRecompute
        );

        // Authoritative + empty unstamped → stamp fresh and reuse (empty) cache.
        assert_eq!(decide_cache_reuse(Some(&exp), None, true), StampFresh);

        // Authoritative + POPULATED unstamped → fail-closed discard (the bug fix).
        assert_eq!(
            decide_cache_reuse(Some(&exp), None, false),
            DiscardAndRecompute
        );

        // No authoritative identity at all → always discard, even on an empty
        // unstamped cache (nothing to bless) and even on a stamped cache.
        assert_eq!(decide_cache_reuse(None, None, true), DiscardAndRecompute);
        assert_eq!(
            decide_cache_reuse(None, Some(&exp), false),
            DiscardAndRecompute
        );
        assert_eq!(decide_cache_reuse(None, Some(&exp), true), DiscardAndRecompute);
    }
}
