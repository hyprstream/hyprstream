//! TTN (Test-Time Naturalization) analysis pipeline for model layer profiling.
//!
//! Provides per-layer LoRA rank allocation. Uses a tiered approach:
//!
//! - **Tier 1**: Embedded profiles for known models (zero-cost const lookup) —
//!   ranks from ablation perplexity delta (the gold standard).
//! - **Tier 2**: Cached profiles from disk (~1ms file read).
//! - **Tier 3**: Computed profiles — layer type detected from weight-key
//!   geometry, every layer assigned a flat unvalidated rank (see
//!   [`UNVALIDATED_RANK_CAP`] / [`GDN_LORA_RANK`]) that the runtime
//!   utilization oracle narrows from. There is no per-layer spectral analysis;
//!   a data-calibrated allocator is gated on the rank-proxy spike (#842).
//!
//! Ported from `tnn-transformer-1/src/entropy.py` and `attention_analysis.py`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tch::{Kind, Tensor};
use tracing::{info, warn};

use crate::runtime::model_config::ModelConfig;

// ============================================================================
// Core types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayerType {
    FullAttention,
    GatedDeltaNet,
    StandardAttention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAnalysis {
    pub layer_idx: usize,
    pub layer_type: LayerType,
    /// Perplexity delta if ablation data available, else None
    pub perplexity_delta: Option<f64>,
    pub recommended_rank: usize,
    pub target_modules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProfile {
    pub model_id: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub baseline_perplexity: Option<f64>,
    pub layers: Vec<LayerAnalysis>,
    /// ISO timestamp; None for embedded profiles
    pub computed_at: Option<String>,
}

// ============================================================================
// Tier 1: Embedded profiles (known models — zero cost)
// ============================================================================

/// Embedded Tier-1 layer profile for the single ablated hybrid (full-attention
/// + GatedDeltaNet) reference model.
///
/// Data from `tnn-transformer-1/results/linearization_ablation.json`.
/// Rank assignments based on perplexity delta (the ablation gold standard).
pub fn qwen3_5_0_8b_profile() -> LayerProfile {
    // Full-attention layers: 3, 7, 11, 15, 19, 23
    // GDN layers: 0-2, 4-6, 8-10, 12-14, 16-18, 20-22
    // (layer_idx, perplexity_delta, rank). Rank comes from perplexity delta;
    // no per-projection spectral data is carried.
    let attn_layers = [
        (23usize, Some(8.40f64), 32usize),
        (3usize,  Some(5.62f64), 24usize),
        (19usize, Some(4.39f64), 16usize),
        (15usize, Some(3.22f64), 16usize),
        (7usize,  Some(2.16f64), 8usize),
        (11usize, Some(1.13f64), 8usize),
    ];
    let attn_set: std::collections::HashSet<usize> =
        attn_layers.iter().map(|(i, ..)| *i).collect();

    let mut layers: Vec<LayerAnalysis> = Vec::with_capacity(24);

    // Build full-attention entries
    for (idx, ppl_delta, rank) in &attn_layers {
        layers.push(LayerAnalysis {
            layer_idx: *idx,
            layer_type: LayerType::FullAttention,
            perplexity_delta: *ppl_delta,
            recommended_rank: *rank,
            target_modules: vec![
                "q_proj".to_owned(),
                "v_proj".to_owned(),
                "o_proj".to_owned(),
            ],
        });
    }

    // Build GDN entries (layers 0-23 not in attn_set)
    for idx in 0..24usize {
        if !attn_set.contains(&idx) {
            layers.push(LayerAnalysis {
                layer_idx: idx,
                layer_type: LayerType::GatedDeltaNet,
                perplexity_delta: None,
                recommended_rank: GDN_LORA_RANK,
                target_modules: vec!["o_proj".to_owned()],
            });
        }
    }

    layers.sort_by_key(|l| l.layer_idx);

    LayerProfile {
        model_id: "Qwen/Qwen3.5-0.8B".to_owned(),
        num_layers: 24,
        hidden_size: 1024,
        baseline_perplexity: Some(29.54),
        layers,
        computed_at: None,
    }
}

/// Registry lookup for embedded profiles. Returns `None` for unknown models.
///
/// A known model family with unexpected geometry (e.g. a larger variant of a
/// profiled family) also returns `None`, but loudly: it falls through to
/// unvalidated allocation, and silence here previously masked that.
pub fn find_embedded_profile(
    model_type: &str,
    num_layers: usize,
    hidden_size: usize,
) -> Option<LayerProfile> {
    let normalized = model_type.to_lowercase();
    let normalized = normalized.as_str();
    let known_family = matches!(normalized, "qwen3_5" | "qwen3.5" | "qwen3_5_text");
    match (normalized, num_layers, hidden_size) {
        ("qwen3_5" | "qwen3.5" | "qwen3_5_text", 24, 1024) => Some(qwen3_5_0_8b_profile()),
        _ => {
            if known_family {
                warn!(
                    model_type,
                    num_layers,
                    hidden_size,
                    "Known model family but unexpected geometry — no embedded \
                     rank profile matches; falling through to unvalidated \
                     allocation"
                );
            }
            None
        }
    }
}

// ============================================================================
// Tier 2: Cached profiles (disk cache)
// ============================================================================

const ANALYSIS_DIR: &str = ".analysis";
const PROFILE_FILE: &str = "layer_profile.json";

/// Load cached profile from model directory.
pub fn load_cached_profile(model_path: &Path) -> Option<LayerProfile> {
    let path = model_path.join(ANALYSIS_DIR).join(PROFILE_FILE);
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Save computed profile to model directory (atomic: tmpfile + rename).
pub fn save_cached_profile(model_path: &Path, profile: &LayerProfile) -> Result<()> {
    let dir = model_path.join(ANALYSIS_DIR);
    std::fs::create_dir_all(&dir)?;
    let content = serde_json::to_string_pretty(profile)?;
    let tmp = dir.join(format!("{PROFILE_FILE}.tmp.{}", std::process::id()));
    std::fs::write(&tmp, &content)?;
    std::fs::rename(&tmp, dir.join(PROFILE_FILE))?; // atomic on Linux same-FS
    Ok(())
}

// ============================================================================
// Tier 3: Computed profiles (weight-key geometry → flat rank)
// ============================================================================

/// Shannon entropy of a non-negative tensor treated as unnormalized probabilities.
///
/// Used by `delta_rank_utilization` (applied to gram eigenvalues, which are
/// already σᵢ²) to measure how spread the learned LoRA update is across rank.
fn entropy_of_nonneg(s: &Tensor) -> f64 {
    let total: f64 = s.sum(Kind::Double).double_value(&[]);
    if total < 1e-30 {
        return 0.0;
    }
    let p = s / total;
    let mask = p.gt(1e-30);
    let p_valid = p.masked_select(&mask);
    -(&p_valid * p_valid.log()).sum(Kind::Double).double_value(&[])
}

/// Rank ceiling for any profile derived without ablation ground truth
/// (computed Tier-3 profiles, the uniform fallback, and cached copies of
/// them). Without a validated per-layer allocator, every unprofiled layer gets
/// this flat value, which the runtime utilization oracle then narrows from.
const UNVALIDATED_RANK_CAP: usize = 8;

/// LoRA rank for GatedDeltaNet layers, as an explicit constant.
///
/// GDN capacity lives in the recurrent/conv/gating parameters; this value
/// matches the embedded GDN rank and sits under [`UNVALIDATED_RANK_CAP`].
/// Revisit when GDN ablation ground truth exists.
const GDN_LORA_RANK: usize = 4;

/// Detect the layer type for a given layer index.
///
/// Priority:
/// 1. `config.layer_types` if non-empty
/// 2. Weight key presence
/// 3. Default: `StandardAttention`
fn detect_layer_type(
    config: &ModelConfig,
    layer_idx: usize,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> LayerType {
    // Tier 1: use config.layer_types
    if let Some(lt) = config.layer_types.get(layer_idx) {
        return match lt.as_str() {
            "linear_attention" => LayerType::GatedDeltaNet,
            "full_attention" | "global" | "sliding_window" => LayerType::FullAttention,
            _ => LayerType::StandardAttention,
        };
    }

    // Tier 2: weight key presence
    let gdn_key = format!("{prefix}.linear_attn.out_proj.weight");
    if weights.contains_key(&gdn_key) {
        return LayerType::GatedDeltaNet;
    }
    let attn_key = format!("{prefix}.self_attn.q_proj.weight");
    if weights.contains_key(&attn_key) {
        return LayerType::FullAttention;
    }

    LayerType::StandardAttention
}

/// Compute a layer profile from weight-key geometry (Tier 3).
///
/// Detects each layer's type from weight-key presence and assigns a flat
/// unvalidated rank (see [`UNVALIDATED_RANK_CAP`] / [`GDN_LORA_RANK`]); there
/// is no per-layer spectral analysis. Results are cached to
/// `{model_path}/.analysis/layer_profile.json`.
pub fn compute_layer_profile(
    weights: &HashMap<String, Tensor>,
    config: &ModelConfig,
    model_path: &Path,
) -> Result<LayerProfile> {
    let _guard = tch::no_grad_guard();
    info!(
        "Computing TTN layer profile for {} ({} layers)...",
        config.model_type, config.num_hidden_layers
    );

    // Detect key prefix: base models use "model.layers.*",
    // Instruct models may use "model.language_model.layers.*"
    let prefix_base = if weights.keys().any(|k| k.starts_with("model.language_model.")) {
        "model.language_model.layers"
    } else {
        "model.layers"
    };

    let mut layers = Vec::new();

    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("{prefix_base}.{layer_idx}");

        let layer_type = detect_layer_type(config, layer_idx, weights, &prefix);

        // No validated per-layer rank signal exists for unprofiled models: the
        // spectral-entropy → rank mapping that lived here was unvalidated (its
        // sign inverted across attention projections on the only ablation
        // data) and has been removed. Every layer gets a flat prior — the cap
        // for attention/standard layers, the explicit GDN constant for GDN —
        // and the runtime utilization oracle narrows from there. A
        // data-calibrated allocator is gated on the rank-proxy validation
        // spike (#842); until then a flat prior is more honest than an
        // unvalidated guess.
        let recommended_rank = if layer_type == LayerType::GatedDeltaNet {
            GDN_LORA_RANK
        } else {
            UNVALIDATED_RANK_CAP
        };

        // Target modules: default ["q_proj", "v_proj"] for standard attention
        let target_modules = match layer_type {
            LayerType::FullAttention => {
                vec!["q_proj".to_owned(), "v_proj".to_owned(), "o_proj".to_owned()]
            }
            LayerType::GatedDeltaNet => vec!["o_proj".to_owned()],
            LayerType::StandardAttention => {
                vec!["q_proj".to_owned(), "v_proj".to_owned()]
            }
        };

        layers.push(LayerAnalysis {
            layer_idx,
            layer_type,
            perplexity_delta: None,
            recommended_rank,
            target_modules,
        });
    }

    let profile = LayerProfile {
        model_id: config.model_type.clone(),
        num_layers: config.num_hidden_layers,
        hidden_size: config.hidden_size,
        baseline_perplexity: None,
        layers,
        computed_at: Some(chrono::Utc::now().to_rfc3339()),
    };

    // Atomic cache write (tmpfile + rename)
    if let Err(e) = save_cached_profile(model_path, &profile) {
        warn!("Failed to cache layer profile: {e}");
    }

    info!(
        "TTN profile computed: {} layers analyzed",
        profile.layers.len()
    );
    Ok(profile)
}

/// Fallback uniform profile when no weights are available for analysis.
///
/// Uses `["o_proj"]` as the sole target module because it is present in all known
/// architectures (full-attention and GDN/SSM layers alike). Using `["q_proj", "v_proj"]`
/// would crash delta pool creation for hybrid models whose GDN layers have no q/v projections.
fn uniform_fallback_profile(config: &ModelConfig) -> LayerProfile {
    let layers = (0..config.num_hidden_layers)
        .map(|layer_idx| LayerAnalysis {
            layer_idx,
            layer_type: LayerType::StandardAttention,
            perplexity_delta: None,
            recommended_rank: UNVALIDATED_RANK_CAP, // uninformed uniform default
            target_modules: vec!["o_proj".to_owned()],
        })
        .collect();

    LayerProfile {
        model_id: config.model_type.clone(),
        num_layers: config.num_hidden_layers,
        hidden_size: config.hidden_size,
        baseline_perplexity: None,
        layers,
        computed_at: None,
    }
}

// ============================================================================
// Unified public API
// ============================================================================

/// Get or compute the layer profile for a model.
///
/// Tiered: embedded → cached → computed.
///
/// # Arguments
/// * `model_path` - Path to model directory (for cache read/write)
/// * `config` - Model configuration
/// * `weights` - Optional loaded weights (required for Tier 3 computation)
pub fn get_layer_profile(
    model_path: &Path,
    config: &ModelConfig,
    weights: Option<&HashMap<String, Tensor>>,
) -> Result<LayerProfile> {
    // Tier 1: Embedded profile for known models (zero-cost)
    if let Some(profile) =
        find_embedded_profile(&config.model_type, config.num_hidden_layers, config.hidden_size)
    {
        info!(
            "Using embedded TTN profile for {} ({} layers)",
            config.model_type, config.num_hidden_layers
        );
        return Ok(profile);
    }

    // Tier 2: Cached profile from disk (~1ms)
    if let Some(mut profile) = load_cached_profile(model_path) {
        info!(
            "Loaded cached TTN profile from {}",
            model_path.display()
        );
        // A cached profile for a model with no embedded ablation data is a
        // persisted Tier-3 computation and gets the same cap. Older caches
        // predate the cap; hand-edited ones must not bypass it either.
        for layer in &mut profile.layers {
            if layer.recommended_rank > UNVALIDATED_RANK_CAP {
                warn!(
                    layer_idx = layer.layer_idx,
                    cached_rank = layer.recommended_rank,
                    "Cached profile exceeds the unvalidated-rank cap; clamping"
                );
                layer.recommended_rank = UNVALIDATED_RANK_CAP;
            }
        }
        return Ok(profile);
    }

    // Tier 3: Compute from weights (~2-5s)
    if let Some(weights) = weights {
        warn!(
            model_type = %config.model_type,
            rank_cap = UNVALIDATED_RANK_CAP,
            "Unknown model — Tier 3 entropy-based rank allocation is unvalidated \
             (no embedded ablation data). A conservative rank cap is applied until \
             the mapping is validated against ablation ground truth. Consider \
             contributing an embedded profile."
        );
        info!("Computing TTN profile for new model (one-time analysis)...");
        return compute_layer_profile(weights, config, model_path);
    }

    // Fallback: uniform profile
    warn!("No weights available for TTN analysis, using uniform profile");
    Ok(uniform_fallback_profile(config))
}

// ============================================================================
// Runtime observability
// ============================================================================

/// Compute delta rank utilization: how much of allocated rank is actually used.
///
/// Returns value in [0, 1]: 0 = rank-1 (collapsed), 1 = fully utilized.
///
/// Uses SVD of `B^T @ B` which is `[rank, rank]` — much cheaper than `B @ A`
/// which is `[out, in]`. B accumulates the learned update (initialized to zeros), so its
/// singular value distribution captures how much of the allocated rank is actually being
/// used. When B is zero (fresh delta), returns 0.0.
///
/// lora_a: [rank, in_features], lora_b: [out_features, rank]
///
/// Called periodically during TTT (every N steps), not on every forward pass.
pub fn delta_rank_utilization(lora_a: &Tensor, lora_b: &Tensor) -> f64 {
    let _guard = tch::no_grad_guard();
    let _ = lora_a; // A is kaiming-initialized and not informative for rank utilization
    // Gram trick: B^T @ B = [rank, rank]; eigenvalues = σᵢ² of B.
    // O(rank³) — much cheaper than SVD of [out, rank] directly (O(out·rank²)).
    let gram = lora_b.tr().matmul(lora_b);
    let (_, s, _) = gram.svd(true, false); // s = σᵢ² of B (eigenvalues of PSD gram)
    let max_entropy = (s.size()[0] as f64).ln();
    if max_entropy < 1e-10 {
        return 0.0;
    }
    // s is already σᵢ² — pass directly to shared helper (no double-squaring)
    entropy_of_nonneg(&s) / max_entropy
}

// ============================================================================
// Attention analysis functions (offline / developer tooling)
// ============================================================================

/// Compute Shannon entropy of attention distributions.
///
/// Input: `[B, H, N, N]` attention weights → Output: `[B, H, N]` per-token entropy.
///
/// Ported from `tnn-transformer-1/src/attention_analysis.py:compute_attention_entropy`.
pub fn attention_entropy(attn_weights: &Tensor) -> Tensor {
    let _guard = tch::no_grad_guard();
    let eps = 1e-10f64;
    let p = attn_weights.clamp_min(eps);
    -(&p * p.log()).sum_dim_intlist(&[-1i64][..], false, None)
}

/// Compute top-k attention mass fraction (sparsity measure).
///
/// Input: `[B, H, N, N]` attention weights → Output: `[B, H, N]` per-token sparsity.
///
/// Ported from `tnn-transformer-1/src/attention_analysis.py:compute_attention_sparsity`.
pub fn attention_sparsity(attn_weights: &Tensor, top_k: i64) -> Tensor {
    let _guard = tch::no_grad_guard();
    let (top_vals, _) = attn_weights.topk(top_k, -1, true, false);
    top_vals.sum_dim_intlist(&[-1i64][..], false, None)
}

// ============================================================================
// Rank Utilization Tracker
// ============================================================================

/// Rolling-window rank utilization tracker.
///
/// Records per-key utilization values and produces adaptation signals
/// (increase/decrease/hold) based on configurable thresholds.
#[derive(Debug, Clone)]
pub struct RankUtilizationTracker {
    window_size: usize,
    history: HashMap<String, std::collections::VecDeque<f64>>,
}

/// Signal for rank adaptation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankSignal {
    Increase,
    Decrease,
    Hold,
}

/// Summary statistics for a key's utilization history.
#[derive(Debug, Clone)]
pub struct UtilizationSummary {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

impl RankUtilizationTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            history: HashMap::new(),
        }
    }

    pub fn record(&mut self, key: &str, utilization: f64) {
        let entry = self
            .history
            .entry(key.to_owned())
            .or_default();
        entry.push_back(utilization);
        while entry.len() > self.window_size {
            entry.pop_front();
        }
    }

    pub fn summary(&self, key: &str) -> Option<UtilizationSummary> {
        let h = self.history.get(key)?;
        if h.is_empty() {
            return None;
        }
        let count = h.len();
        let sum: f64 = h.iter().sum();
        let min = h.iter().copied().fold(f64::MAX, f64::min);
        let max = h.iter().copied().fold(f64::MIN, f64::max);
        Some(UtilizationSummary {
            mean: sum / count as f64,
            min,
            max,
            count,
        })
    }

    pub fn rank_adaptation_signals(
        &self,
        low_threshold: f64,
        high_threshold: f64,
    ) -> HashMap<String, RankSignal> {
        self.history
            .keys()
            .filter_map(|key| {
                let s = self.summary(key)?;
                let signal = if s.mean < low_threshold {
                    RankSignal::Decrease
                } else if s.mean > high_threshold {
                    RankSignal::Increase
                } else {
                    RankSignal::Hold
                };
                Some((key.clone(), signal))
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_5_profile_structure() {
        let profile = qwen3_5_0_8b_profile();
        assert_eq!(profile.num_layers, 24);
        assert_eq!(profile.layers.len(), 24);

        // Check full-attention layers have correct ranks
        let attn_23 = profile.layers.iter().find(|l| l.layer_idx == 23).expect("layer 23 should exist");
        assert_eq!(attn_23.recommended_rank, 32);
        assert_eq!(attn_23.layer_type, LayerType::FullAttention);
        assert_eq!(attn_23.perplexity_delta, Some(8.40));

        let gdn_0 = profile.layers.iter().find(|l| l.layer_idx == 0).expect("layer 0 should exist");
        assert_eq!(gdn_0.recommended_rank, GDN_LORA_RANK);
        assert_eq!(gdn_0.layer_type, LayerType::GatedDeltaNet);
    }

    #[test]
    fn test_find_embedded_profile() {
        assert!(find_embedded_profile("qwen3_5", 24, 1024).is_some());
        assert!(find_embedded_profile("qwen3.5", 24, 1024).is_some());
        assert!(find_embedded_profile("llama", 32, 4096).is_none());
        assert!(find_embedded_profile("qwen3_5", 32, 4096).is_none()); // different dims
    }

    #[test]
    fn test_tier3_synthetic_profile_golden() {
        let _guard = tch::no_grad_guard();
        tch::manual_seed(42);
        let dir = tempfile::tempdir().expect("tempdir");

        let config = ModelConfig {
            model_type: "synthetic_test".to_owned(),
            num_hidden_layers: 2,
            hidden_size: 64,
            ..Default::default()
        };

        // Layer 0: GDN (detected by weight-key presence). Layer 1: attention.
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert(
            "model.layers.0.linear_attn.out_proj.weight".to_owned(),
            Tensor::randn([64, 64], (Kind::Float, tch::Device::Cpu)),
        );
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            weights.insert(
                format!("model.layers.1.self_attn.{proj}.weight"),
                Tensor::randn([64, 64], (Kind::Float, tch::Device::Cpu)),
            );
        }

        let profile =
            compute_layer_profile(&weights, &config, dir.path()).expect("profile");
        assert_eq!(profile.layers.len(), 2);

        let gdn = &profile.layers[0];
        assert_eq!(gdn.layer_type, LayerType::GatedDeltaNet);
        assert_eq!(gdn.recommended_rank, GDN_LORA_RANK);

        let attn = &profile.layers[1];
        assert_eq!(attn.layer_type, LayerType::FullAttention);
        assert_eq!(
            attn.recommended_rank, UNVALIDATED_RANK_CAP,
            "unprofiled attention layers get the flat unvalidated cap"
        );

        // The computation persisted itself to the Tier-2 cache.
        assert!(load_cached_profile(dir.path()).is_some());
    }

    #[test]
    fn test_cached_profile_rank_is_capped_on_load() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ModelConfig {
            model_type: "synthetic_test".to_owned(),
            num_hidden_layers: 1,
            hidden_size: 64,
            ..Default::default()
        };

        // A pre-cap (or hand-edited) cache claiming rank 32.
        let mut profile = uniform_fallback_profile(&config);
        profile.layers[0].recommended_rank = 32;
        save_cached_profile(dir.path(), &profile).expect("save");

        // No embedded profile for this model → Tier 2 load must clamp.
        let loaded = get_layer_profile(dir.path(), &config, None).expect("load");
        assert_eq!(loaded.layers[0].recommended_rank, UNVALIDATED_RANK_CAP);
    }

    #[test]
    fn test_delta_rank_utilization() {
        let _guard = tch::no_grad_guard();
        let rank = 8i64;
        let in_f = 64i64;
        let out_f = 64i64;

        let a = Tensor::randn([rank, in_f], (Kind::Float, tch::Device::Cpu));
        let b = Tensor::zeros([out_f, rank], (Kind::Float, tch::Device::Cpu));
        // B = 0 → product is 0 → utilization should be 0
        let util = delta_rank_utilization(&a, &b);
        assert_eq!(util, 0.0);
    }

    // --- Rank Utilization Tracker Tests ---

    #[test]
    fn test_utilization_tracker_records() {
        let mut tracker = RankUtilizationTracker::new(5);
        tracker.record("0.q_proj", 0.3);
        tracker.record("0.q_proj", 0.5);
        let summary = tracker.summary("0.q_proj");
        assert!(summary.is_some());
        let s = summary.unwrap();
        assert_eq!(s.count, 2);
        assert!((s.mean - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_utilization_tracker_window() {
        let mut tracker = RankUtilizationTracker::new(3);
        tracker.record("k", 0.1);
        tracker.record("k", 0.2);
        tracker.record("k", 0.3);
        tracker.record("k", 0.9); // pushes out 0.1
        let s = tracker.summary("k").unwrap();
        assert_eq!(s.count, 3);
        // mean of [0.2, 0.3, 0.9] = 0.4667
        assert!((s.mean - 0.4667).abs() < 0.01);
    }

    #[test]
    fn test_utilization_tracker_rank_signals() {
        let mut tracker = RankUtilizationTracker::new(10);
        for _ in 0..5 {
            tracker.record("low", 0.15);
        }
        for _ in 0..5 {
            tracker.record("high", 0.95);
        }
        for _ in 0..5 {
            tracker.record("ok", 0.5);
        }

        let signals = tracker.rank_adaptation_signals(0.25, 0.85);
        assert_eq!(signals.get("low"), Some(&RankSignal::Decrease));
        assert_eq!(signals.get("high"), Some(&RankSignal::Increase));
        assert_eq!(signals.get("ok"), Some(&RankSignal::Hold));
    }

    #[test]
    fn test_utilization_tracker_empty_key() {
        let tracker = RankUtilizationTracker::new(5);
        assert!(tracker.summary("nonexistent").is_none());
    }

}
