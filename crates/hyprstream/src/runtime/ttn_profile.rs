//! TTN (Test-Time Naturalization) analysis pipeline for model layer profiling.
//!
//! Provides adaptive analysis for determining optimal LoRA rank allocation
//! per layer. Uses a tiered approach:
//!
//! - **Tier 1**: Embedded profiles for known models (zero-cost const lookup)
//! - **Tier 2**: Cached profiles from disk (~1ms file read)
//! - **Tier 3**: Computed profiles via bond entropy analysis (~2-5s, cached)
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
    /// Weight name → bond entropy in nats
    pub bond_entropy: HashMap<String, f64>,
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

/// Pre-computed layer analysis for Qwen3.5-0.8B (Tier 1).
///
/// Data from `tnn-transformer-1/results/linearization_ablation.json`.
/// Rank assignments based on perplexity delta (ablation gold standard).
pub fn qwen3_5_0_8b_profile() -> LayerProfile {
    // Full-attention layers: 3, 7, 11, 15, 19, 23
    // GDN layers: 0-2, 4-6, 8-10, 12-14, 16-18, 20-22
    let attn_layers = [
        // (layer_idx, perplexity_delta, rank, bond_entropies: [(name, k_entropy, v_entropy)])
        (23usize, Some(8.40f64), 32usize, vec![
            ("q_proj", 5.33f64), ("k_proj", 5.28f64), ("v_proj", 5.28f64),
        ]),
        (3usize,  Some(5.62f64), 24usize, vec![
            ("q_proj", 5.75f64), ("k_proj", 5.85f64), ("v_proj", 5.85f64),
        ]),
        (19usize, Some(4.39f64), 16usize, vec![
            ("q_proj", 5.15f64), ("k_proj", 5.56f64), ("v_proj", 5.56f64),
        ]),
        (15usize, Some(3.22f64), 16usize, vec![
            ("q_proj", 5.49f64), ("k_proj", 5.83f64), ("v_proj", 5.83f64),
        ]),
        (7usize,  Some(2.16f64), 8usize, vec![
            ("q_proj", 5.65f64), ("k_proj", 5.82f64), ("v_proj", 5.82f64),
        ]),
        (11usize, Some(1.13f64), 8usize, vec![
            ("q_proj", 5.42f64), ("k_proj", 5.77f64), ("v_proj", 5.77f64),
        ]),
    ];
    let attn_set: std::collections::HashSet<usize> =
        attn_layers.iter().map(|(i, ..)| *i).collect();

    let mut layers: Vec<LayerAnalysis> = Vec::with_capacity(24);

    // Build full-attention entries
    for (idx, ppl_delta, rank, entropies) in &attn_layers {
        let mut entropy_map = HashMap::new();
        for (name, e) in entropies {
            entropy_map.insert((*name).to_owned(), *e);
        }
        layers.push(LayerAnalysis {
            layer_idx: *idx,
            layer_type: LayerType::FullAttention,
            bond_entropy: entropy_map,
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
                bond_entropy: {
                    let mut m = HashMap::new();
                    m.insert("o_proj".to_owned(), 6.2f64); // representative value
                    m
                },
                perplexity_delta: None,
                recommended_rank: 4,
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
pub fn find_embedded_profile(
    model_type: &str,
    num_layers: usize,
    hidden_size: usize,
) -> Option<LayerProfile> {
    let normalized = model_type.to_lowercase();
    let normalized = normalized.as_str();
    match (normalized, num_layers, hidden_size) {
        ("qwen3_5" | "qwen3.5" | "qwen3_5_text", 24, 1024) => Some(qwen3_5_0_8b_profile()),
        _ => None,
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
// Tier 3: Computed profiles (bond entropy analysis)
// ============================================================================

/// Shannon entropy of a non-negative tensor treated as unnormalized probabilities.
///
/// Shared by `bond_entropy` (applied to squared singular values) and
/// `delta_rank_utilization` (applied to gram eigenvalues, which are already σᵢ²).
/// Keeping this separate preserves the distinct SVD strategies in each caller.
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

/// Compute bond entropy of a weight matrix.
///
/// Returns `(entropy_nats, num_singular_values)`.
/// The caller normalizes by `ln(num_sv)` for per-matrix normalization (I-1 fix).
///
/// Ported from `tnn-transformer-1/src/entropy.py:bond_entropy_from_singular_values`.
pub fn bond_entropy(matrix: &Tensor) -> (f64, usize) {
    let _guard = tch::no_grad_guard();
    // Reshape to 2D if needed (FP8 may have been dequantized already)
    let m = if matrix.dim() == 2 {
        matrix.to_kind(Kind::Float)
    } else {
        let shape = matrix.size();
        let rows = shape[0];
        let cols: i64 = shape[1..].iter().product();
        matrix.reshape([rows, cols]).to_kind(Kind::Float)
    };

    // Use Tensor::svd (economy, compute_uv=false) — compatible with both CPU and CUDA.
    // linalg_svdvals requires the driver argument which fails on CPU without cuSOLVER.
    let (_, s, _) = m.svd(true, false);
    let num_sv = s.size()[0] as usize;
    let s_sq = &s * &s;
    (entropy_of_nonneg(&s_sq), num_sv)
}

/// Map normalized entropy [0,1] to LoRA rank.
/// Lower normalized entropy → more structured → higher rank.
///
/// Calibrated approximately against Qwen3.5-0.8B ablation data.
/// Note: Tier 3 ranks are approximate — embedded profiles (Tier 1) use
/// ablation-derived perplexity delta as the gold standard.
fn entropy_to_rank(normalized: f64) -> usize {
    match normalized {
        x if x < 0.80 => 32, // Very structured (critical layers)
        x if x < 0.85 => 16, // Moderate structure
        x if x < 0.92 => 8,  // Low structure
        _ => 4,               // Near-uniform
    }
}

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

/// Compute layer profile by analyzing weight matrices (Tier 3).
///
/// Runs SVD on Q/K/V/O projection weights per layer to determine bond entropy.
/// Results are cached to `{model_path}/.analysis/layer_profile.json`.
///
/// # Complexity
/// O(layers × projs × min(rows,cols)²) SVD. Typically 2–5s for 0.8B models.
pub fn compute_weight_entropy_profile(
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

        // Architecture-aware key resolution (R2-C1 fix)
        let (subpath, projs): (&str, &[&str]) = match layer_type {
            LayerType::FullAttention | LayerType::StandardAttention => {
                ("self_attn", &["q_proj", "k_proj", "v_proj", "o_proj"])
            }
            LayerType::GatedDeltaNet => ("linear_attn", &["out_proj"]),
        };

        let mut entropies = HashMap::new();
        let mut entropy_normalized_sum = 0.0f64;
        let mut entropy_count = 0usize;

        for proj in projs {
            let key = format!("{prefix}.{subpath}.{proj}.weight");
            if let Some(w) = weights.get(&key) {
                let (entropy_nats, num_sv) = bond_entropy(w);
                entropies.insert((*proj).to_owned(), entropy_nats);

                // Per-matrix normalization (R2-I1 fix): divide by ln(num_sv)
                let max_entropy = (num_sv as f64).ln();
                if max_entropy > 1e-10 {
                    entropy_normalized_sum += entropy_nats / max_entropy;
                    entropy_count += 1;
                }
            }
        }

        let avg_normalized = if entropy_count > 0 {
            entropy_normalized_sum / entropy_count as f64
        } else {
            0.95 // Default: near-uniform → low rank
        };

        let recommended_rank = entropy_to_rank(avg_normalized);

        // Target modules: default ["q_proj", "v_proj"] for standard (R2-I4 fix)
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
            bond_entropy: entropies,
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

    // Atomic cache write (R2-I2 fix)
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
            bond_entropy: HashMap::new(),
            perplexity_delta: None,
            recommended_rank: 8, // uniform default
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
    if let Some(profile) = load_cached_profile(model_path) {
        info!(
            "Loaded cached TTN profile from {}",
            model_path.display()
        );
        return Ok(profile);
    }

    // Tier 3: Compute from weights (~2-5s)
    if let Some(weights) = weights {
        info!("Computing TTN profile for new model (one-time analysis)...");
        return compute_weight_entropy_profile(weights, config, model_path);
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
/// Uses SVD of `B^T @ B` which is `[rank, rank]` (I2 fix) — much cheaper than `B @ A`
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

        let gdn_0 = profile.layers.iter().find(|l| l.layer_idx == 0).expect("layer 0 should exist");
        assert_eq!(gdn_0.recommended_rank, 4);
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
    fn test_entropy_to_rank() {
        assert_eq!(entropy_to_rank(0.70), 32);
        assert_eq!(entropy_to_rank(0.82), 16);
        assert_eq!(entropy_to_rank(0.90), 8);
        assert_eq!(entropy_to_rank(0.95), 4);
    }

    #[test]
    fn test_bond_entropy_random() {
        let _guard = tch::no_grad_guard();
        let m = Tensor::randn([64, 64], (Kind::Float, tch::Device::Cpu));
        let (entropy, num_sv) = bond_entropy(&m);
        assert_eq!(num_sv, 64);
        assert!(entropy > 0.0);
        // Random matrix should have near-uniform distribution, so high entropy
        let max_entropy = (64f64).ln();
        assert!(entropy < max_entropy + 1e-6);
    }

    #[test]
    fn test_bond_entropy_rank1() {
        let _guard = tch::no_grad_guard();
        // Rank-1 matrix: outer product of two vectors
        let u = Tensor::randn([64, 1], (Kind::Float, tch::Device::Cpu));
        let v = Tensor::randn([1, 32], (Kind::Float, tch::Device::Cpu));
        let m = u.matmul(&v);
        let (entropy, _) = bond_entropy(&m);
        // Rank-1 → single nonzero singular value → entropy = 0
        assert!(entropy < 0.1, "Rank-1 matrix entropy should be near 0, got {entropy}");
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
}
