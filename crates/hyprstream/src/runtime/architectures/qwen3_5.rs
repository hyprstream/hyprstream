//! Qwen3.5 hybrid SSM/full-attention model implementation
#![allow(dead_code, unused_variables, clippy::redundant_closure, clippy::assign_op_pattern, clippy::needless_borrows_for_generic_args, clippy::redundant_closure_for_method_calls)]
//!
//! Qwen3.5 interleaves Gated DeltaNet (linear SSM attention) and standard GQA layers.
//! Every 4th layer (by default) is full attention; the rest are GatedDeltaNet.
//! Supports both dense (qwen3_5_text) and MoE (qwen3_5_moe) variants.

use super::config::ArchitectureConfig;
use super::llama::{LlamaMLP, LinearProjection};
use super::qwen3_5_vision::{Qwen3_5VisionConfig, Qwen3_5VisionEncoder};
use super::{ModelArchitecture, ModelOperations};
use crate::runtime::device_pool::LayerDeviceMap;
use crate::runtime::kv_cache::KVCacheManager;
use crate::runtime::KVQuantType;
use crate::runtime::model_config::ModelConfig;
use crate::runtime::rope::RoPE;
use crate::runtime::tensor_helpers::{dims3, dims4};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};
use tracing::info;

// ============================================================================
// Config
// ============================================================================

/// Qwen3.5 text backbone configuration
pub struct Qwen3_5TextConfig {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Fraction of head_dim to rotate (0.25 → rotary_dim = head_dim/4)
    pub partial_rotary_factor: f32,
    /// Derived: head_dim * partial_rotary_factor
    pub rotary_dim: usize,
    /// Per-layer type: "linear_attention" or "full_attention"
    pub layer_types: Vec<String>,
    // Linear attention (GatedDeltaNet) dimensions
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    // MoE
    pub is_moe: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    // Vision
    pub has_vision: bool,
    pub vision_out_hidden_size: usize,
}

impl ArchitectureConfig for Qwen3_5TextConfig {
    fn num_attention_heads(&self) -> usize { self.num_attention_heads as usize }
    fn num_key_value_heads(&self) -> usize { self.num_key_value_heads as usize }
    fn hidden_size(&self) -> usize { self.hidden_size as usize }
    fn intermediate_size(&self) -> usize { self.intermediate_size as usize }
    fn head_dim(&self) -> usize { self.head_dim as usize }
    fn vocab_size(&self) -> usize { self.vocab_size as usize }
    fn max_position_embeddings(&self) -> usize { self.max_position_embeddings as usize }
    fn rope_theta(&self) -> Option<f32> { Some(self.rope_theta) }
    fn rope_dim(&self) -> Option<usize> { Some(self.rotary_dim) }
    fn layer_norm_eps(&self) -> f32 { self.rms_norm_eps }
    fn use_rms_norm(&self) -> bool { true }
}

impl Qwen3_5TextConfig {
    pub fn from_model_config(cfg: &ModelConfig, max_pos: usize) -> Self {
        let partial_rotary_factor = cfg.partial_rotary_factor.unwrap_or(0.25);
        let rotary_dim = ((cfg.head_dim as f32) * partial_rotary_factor) as usize;

        // Derive layer_types if not present (text-only checkpoint may lack them)
        let layer_types = if cfg.layer_types.is_empty() {
            (0..cfg.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % 4 == 0 {
                        "full_attention".to_owned()
                    } else {
                        "linear_attention".to_owned()
                    }
                })
                .collect()
        } else {
            cfg.layer_types.clone()
        };

        // MoE: use config.moe_intermediate_size if set, else fall back to intermediate_size
        let moe_int_size = if cfg.moe_intermediate_size > 0 {
            cfg.moe_intermediate_size
        } else {
            cfg.intermediate_size
        };
        let shared_int_size = if cfg.shared_expert_intermediate_size > 0 {
            cfg.shared_expert_intermediate_size
        } else {
            cfg.intermediate_size
        };

        Self {
            hidden_size: cfg.hidden_size as u32,
            num_hidden_layers: cfg.num_hidden_layers as u32,
            num_attention_heads: cfg.num_attention_heads as u32,
            num_key_value_heads: cfg.num_key_value_heads as u32,
            head_dim: cfg.head_dim as u32,
            intermediate_size: cfg.intermediate_size as u32,
            vocab_size: cfg.vocab_size as u32,
            max_position_embeddings: max_pos as u32,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            partial_rotary_factor,
            rotary_dim,
            layer_types,
            linear_conv_kernel_dim: if cfg.linear_conv_kernel_dim > 0 { cfg.linear_conv_kernel_dim } else { 4 },
            linear_key_head_dim: if cfg.linear_key_head_dim > 0 { cfg.linear_key_head_dim } else { 128 },
            linear_value_head_dim: if cfg.linear_value_head_dim > 0 { cfg.linear_value_head_dim } else { 128 },
            linear_num_key_heads: if cfg.linear_num_key_heads > 0 { cfg.linear_num_key_heads } else { 16 },
            linear_num_value_heads: if cfg.linear_num_value_heads > 0 { cfg.linear_num_value_heads } else { 32 },
            is_moe: cfg.is_moe,
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            moe_intermediate_size: moe_int_size,
            shared_expert_intermediate_size: shared_int_size,
            has_vision: cfg.has_vision,
            vision_out_hidden_size: cfg.vision_out_hidden_size,
        }
    }
}

// ============================================================================
// FP8 helpers
// ============================================================================

/// Fuse multiple `LinearProjection`s into one by concatenating along the output dimension (dim 1).
///
/// If all projections are the same dtype (all FP8 or all BF16), the weight and scale tensors
/// are concatenated directly — no BF16 conversion at load time.
///
/// If dtypes are mixed (e.g. FP8 + BF16 in the same layer), FP8 weights are dequantized to
/// BF16 first so `cat` can proceed. This only occurs for small projection types in hybrid layers.
fn cat_projs(projs: Vec<LinearProjection>) -> LinearProjection {
    #[inline]
    fn is_fp8(k: tch::Kind) -> bool {
        matches!(k, tch::Kind::Float8e4m3fn | tch::Kind::Float8e5m2)
    }

    let all_fp8 = projs.iter().all(|p| is_fp8(p.weight.kind()));
    let any_fp8 = projs.iter().any(|p| is_fp8(p.weight.kind()));

    if all_fp8 {
        // All FP8: cat weights + scales directly, lazy dequant via apply().
        let weight_refs: Vec<&Tensor> = projs.iter().map(|p| &p.weight).collect();
        let fused_weight = Tensor::cat(&weight_refs, 1);
        let has_scale = projs.iter().any(|p| p.scale.is_some());
        let scale = if has_scale {
            #[allow(clippy::expect_used)] // all_fp8 invariant: every proj has a scale
            let scale_refs: Vec<&Tensor> = projs.iter()
                .map(|p| p.scale.as_ref().expect("FP8 projection missing scale"))
                .collect();
            Some(Tensor::cat(&scale_refs, 1))
        } else {
            None
        };
        LinearProjection { weight: fused_weight, bias: None, scale }
    } else if any_fp8 {
        // Mixed FP8/BF16: dequantize FP8 projections to BF16 before catting.
        // Occurs for small gating projections in hybrid layers; memory impact is minor.
        let weights: Vec<Tensor> = projs.into_iter().map(|p| {
            if is_fp8(p.weight.kind()) {
                let w_bf = p.weight.to_kind(tch::Kind::BFloat16);
                if let Some(s) = p.scale {
                    let ws = w_bf.size();
                    let ss = s.size();
                    let br = ws[0] / ss[0];
                    let bc = ws[1] / ss[1];
                    let w_4d = w_bf.view([ss[0], br, ss[1], bc]);
                    let s_4d = s.to_kind(tch::Kind::BFloat16).view([ss[0], 1, ss[1], 1]);
                    (w_4d * s_4d).reshape([ws[0], ws[1]])
                } else {
                    w_bf
                }
            } else {
                p.weight
            }
        }).collect();
        let weight_refs: Vec<&Tensor> = weights.iter().collect();
        LinearProjection::new(Tensor::cat(&weight_refs, 1))
    } else {
        // All non-FP8: cat directly.
        let weight_refs: Vec<&Tensor> = projs.iter().map(|p| &p.weight).collect();
        LinearProjection::new(Tensor::cat(&weight_refs, 1))
    }
}

/// Dequantize a batched FP8 tensor `[k, in, out]` with optional scale `[k, in/bs, out/bs]`.
/// Used after `index_select` on stacked MoE expert weights.
/// Non-FP8 tensors are returned as-is (shallow clone).
fn dequant_batched(w: &Tensor, scale: Option<&Tensor>) -> Tensor {
    match w.kind() {
        tch::Kind::Float8e4m3fn | tch::Kind::Float8e5m2 => {
            let w_bf = w.to_kind(tch::Kind::BFloat16);
            if let Some(s) = scale {
                let ws = w_bf.size(); // [k, in, out]
                let ss = s.size();   // [k, in/bs, out/bs]
                let br = ws[1] / ss[1];
                let bc = ws[2] / ss[2];
                let w_5d = w_bf.view([ss[0], ss[1], br, ss[2], bc]);
                let s_5d = s.to_kind(tch::Kind::BFloat16).view([ss[0], ss[1], 1, ss[2], 1]);
                (&w_5d * &s_5d).reshape([ws[0], ws[1], ws[2]])
            } else {
                w_bf
            }
        }
        _ => w.shallow_clone(),
    }
}

// ============================================================================
// Normalization helpers
// ============================================================================

/// Qwen3.5 RMSNorm: weight initialized to 0, formula norm(x_f32) * (1 + weight)
struct Qwen3_5RMSNorm {
    weight: Tensor,
    /// Precomputed weight cast to float — avoids one dtype conversion per forward call.
    weight_f32: Tensor,
    eps: f32,
}

impl Qwen3_5RMSNorm {
    fn new(weight: Tensor, eps: f32) -> Self {
        let weight_f32 = weight.to_kind(Kind::Float);
        Self { weight, weight_f32, eps }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig = x.kind();
        let x_f = x.to_kind(Kind::Float);
        let mean_sq = (&x_f * &x_f).mean_dim(&[-1i64][..], true, Kind::Float);
        let rrms = (mean_sq + self.eps as f64).rsqrt();
        let normed = &x_f * rrms;
        // Checkpoint weights are near 0 (mean≈0.028) — model uses (1+w)*normed formula
        let one_plus_w = &self.weight_f32 + 1.0f64;
        Ok((normed * one_plus_w).to_kind(orig))
    }

    /// Move both weight copies to `device` (#314 pipeline placement). No-op when
    /// the tensors already live on `device`.
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self {
            weight: self.weight.to_device(device),
            weight_f32: self.weight_f32.to_device(device),
            eps: self.eps,
        }
    }
}

unsafe impl Send for Qwen3_5RMSNorm {}
unsafe impl Sync for Qwen3_5RMSNorm {}

/// Gated RMSNorm used for GatedDeltaNet output: norm(x) * weight * silu(gate)
struct RMSNormGated {
    weight: Tensor,
    /// Precomputed `weight.to_kind(Float)` — avoids 1 cast kernel per forward call.
    weight_f32: Tensor,
    eps: f32,
}

impl RMSNormGated {
    fn new(weight: Tensor, eps: f32) -> Self {
        let weight_f32 = weight.to_kind(Kind::Float);
        Self { weight, weight_f32, eps }
    }

    fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let orig = x.kind();
        let x_f = if x.kind() == Kind::Float { x.shallow_clone() } else { x.to_kind(Kind::Float) };
        let mean_sq = (&x_f * &x_f).mean_dim(&[-1i64][..], true, Kind::Float);
        let rrms = (mean_sq + self.eps as f64).rsqrt();
        let normed = &x_f * rrms;
        let g = gate.to_kind(Kind::Float).silu();
        Ok((normed * &self.weight_f32 * g).to_kind(orig))
    }

    /// Move both weight copies to `device` (#314 pipeline placement).
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self {
            weight: self.weight.to_device(device),
            weight_f32: self.weight_f32.to_device(device),
            eps: self.eps,
        }
    }
}

unsafe impl Send for RMSNormGated {}
unsafe impl Sync for RMSNormGated {}

/// L2 normalization along last dim (used on Q and K before delta rule)
#[inline]
fn l2_normalize(x: &Tensor) -> Tensor {
    // norm_scalaropt_dim = 2 GPU ops (vs 5 with manual x*x+sum+sqrt+div)
    let norm = x.norm_scalaropt_dim(2.0, &[-1i64][..], true);
    x / norm.clamp_min(1e-3)
}

/// Chunked GatedDeltaNet prefill (translation of `torch_chunk_gated_delta_rule`).
///
/// All inputs in [batch, num_v_heads, seq, head_dim] layout (heads second, already in f32).
/// `q` is already L2-normalized and scaled by 1/sqrt(head_k_dim).
/// Returns (output [batch, num_v_heads, seq, head_v_dim], final state [batch, nv, hk, hv]).
fn chunked_delta_rule(
    q: &Tensor,       // [B, nv, T, hk]  L2-normed, scaled
    k: &Tensor,       // [B, nv, T, hk]  L2-normed
    v: &Tensor,       // [B, nv, T, hv]
    g: &Tensor,       // [B, nv, T]      log-decay (negative)
    beta: &Tensor,    // [B, nv, T]      in (0,1)
    initial_state: Option<&Tensor>,
    device: tch::Device,
) -> Result<(Tensor, Tensor)> {
    const CHUNK_SIZE: i64 = 64;

    let batch = q.size()[0];
    let nv    = q.size()[1];
    let seq   = q.size()[2];
    let hk    = q.size()[3];
    let hv    = v.size()[3];

    // Pad sequence length to multiple of CHUNK_SIZE
    let pad = (CHUNK_SIZE - seq % CHUNK_SIZE) % CHUNK_SIZE;
    let seq_p = seq + pad;

    let pad_tensor = |t: &Tensor, extra_dims: i64| -> Tensor {
        if pad == 0 { return t.shallow_clone(); }
        let z = match extra_dims {
            0 => Tensor::zeros([batch, nv, pad], (Kind::Float, device)),
            _ => Tensor::zeros([batch, nv, pad, extra_dims], (Kind::Float, device)),
        };
        Tensor::cat(&[t, &z], 2)
    };

    let q  = pad_tensor(q, hk);
    let k  = pad_tensor(k, hk);
    let v  = pad_tensor(v, hv);
    let g  = pad_tensor(g, 0);
    let beta = pad_tensor(beta, 0);

    let v_beta = &v * beta.unsqueeze(-1);
    let k_beta = &k * beta.unsqueeze(-1);

    let nc = seq_p / CHUNK_SIZE;

    // Reshape to [B, nv, num_chunks, chunk_size, D]
    let reshape5 = |t: &Tensor, d: i64| t.reshape([batch, nv, nc, CHUNK_SIZE, d]);
    let q      = reshape5(&q,      hk);
    let k      = reshape5(&k,      hk);
    let v      = reshape5(&v,      hv);
    let k_beta = reshape5(&k_beta, hk);
    let v_beta = reshape5(&v_beta, hv);
    // g: [B, nv, num_chunks, chunk_size]
    let g = g.reshape([batch, nv, nc, CHUNK_SIZE]);

    // Cumulative decay within each chunk
    let g = g.cumsum(-1, Kind::Float);

    // decay_mask[t, s] = exp(g[t] - g[s]) for t >= s, 0 for t < s
    // Must exp first, then tril — otherwise upper tri becomes exp(0)=1 instead of 0
    let g_t = g.unsqueeze(-1);  // [..., C, 1]
    let g_s = g.unsqueeze(-2);  // [..., 1, C]
    let decay_mask = (g_t - g_s).exp().tril(0); // [B, nv, nc, C, C]

    // Upper triangular mask (diagonal=0 inclusive) for masking future positions
    let mask0 = Tensor::ones([CHUNK_SIZE, CHUNK_SIZE], (Kind::Bool, device)).triu(0);
    // Upper triangular excluding diagonal (diagonal=1) for within-chunk attn
    let mask1 = Tensor::ones([CHUNK_SIZE, CHUNK_SIZE], (Kind::Bool, device)).triu(1);

    // attn = -(k_beta @ k^T * decay_mask).masked_fill(mask0, 0)
    // [B, nv, nc, C, C]
    let attn0 = -(k_beta.matmul(&k.transpose(-1, -2)) * &decay_mask)
        .masked_fill(&mask0, 0.0f64);

    // Recursive delta-rule correction (forward substitution over chunk positions).
    //
    // Row `i` of the corrected lower-triangular matrix is
    //   T[i, :i] = A[i, :i] + A[i, :i] @ T[:i, :i]
    // i.e. each row depends on the ALREADY-corrected rows above it. The original
    // implementation built this with an in-place `attn[..i].copy_(row)`, which is
    // unsafe for autograd (it mutates a tensor saved for backward → "variable
    // needed for gradient computation modified by an inplace operation"). On the
    // inference path this ran under `no_grad`; on the TTT-on-split training path
    // (#316) the graph is live, so we build the rows out-of-place and stack them.
    // Numerically identical to the in-place form; only the graph differs.
    let c = CHUNK_SIZE;
    // Each entry is a full-width row [B, nv, nc, C]; row 0 is unchanged.
    let mut rows: Vec<Tensor> = Vec::with_capacity(c as usize);
    rows.push(attn0.narrow(-2, 0, 1).squeeze_dim(-2)); // [B, nv, nc, C]
    for i in 1..c {
        // Original row i, columns [0, i) (columns >= i are 0 from mask0).
        let a_row = attn0.narrow(-2, i, 1).narrow(-1, 0, i).squeeze_dim(-2); // [B, nv, nc, i]
        // T[:i, :i] — stack the already-corrected rows, take their first i cols.
        let sub = Tensor::stack(&rows, -2).narrow(-1, 0, i); // [B, nv, nc, i, i]
        let correction = a_row.unsqueeze(-2).matmul(&sub).squeeze_dim(-2); // [B, nv, nc, i]
        let corrected = &a_row + &correction; // [B, nv, nc, i]
        // Re-pad to full width C (columns [i, C) are 0, matching mask0).
        let pad_cols = c - i;
        let zeros = Tensor::zeros(
            [batch, nv, nc, pad_cols],
            (Kind::Float, device),
        );
        rows.push(Tensor::cat(&[corrected, zeros], -1)); // [B, nv, nc, C]
    }
    let attn = Tensor::stack(&rows, -2); // [B, nv, nc, C, C]

    // Add identity
    let eye = Tensor::eye(CHUNK_SIZE, (Kind::Float, device))
        .unsqueeze(0).unsqueeze(0).unsqueeze(0); // [1, 1, 1, C, C]
    let attn = attn + eye;

    // value_transformed = attn @ v_beta  [B, nv, nc, C, hv]
    let value_t = attn.matmul(&v_beta);
    // k_cumdecay = attn @ (k_beta * g.exp()) [B, nv, nc, C, hk]
    let k_cumdecay = attn.matmul(&(k_beta * g.exp().unsqueeze(-1)));

    let mut state = initial_state.map(|s| s.to_kind(Kind::Float)).unwrap_or_else(|| {
        Tensor::zeros([batch, nv, hk, hv], (Kind::Float, device))
    });
    let mut chunks_out = Vec::with_capacity(nc as usize);

    for i in 0..nc {
        let q_i  = q.select(2, i);           // [B, nv, C, hk]
        let k_i  = k.select(2, i);           // [B, nv, C, hk]
        let v_i  = value_t.select(2, i);     // [B, nv, C, hv]
        let g_i  = g.select(2, i);           // [B, nv, C]
        let dm_i = decay_mask.select(2, i);  // [B, nv, C, C]
        let kcd_i = k_cumdecay.select(2, i); // [B, nv, C, hk]

        // Within-chunk attention (causal, exclude future)
        let attn_inner = (q_i.matmul(&k_i.transpose(-1, -2)) * &dm_i).masked_fill(&mask1, 0.0f64);

        // Cross-chunk contribution from previous state
        let v_prime = kcd_i.matmul(&state);           // [B, nv, C, hv]
        let v_new   = v_i - v_prime;

        // Inter-chunk: q scaled by cumulative g × state
        let attn_inter = (q_i * g_i.exp().unsqueeze(-1)).matmul(&state); // [B, nv, C, hv]

        let chunk_out = attn_inter + attn_inner.matmul(&v_new);
        chunks_out.push(chunk_out);

        // Update state for next chunk
        let g_last = g_i.narrow(-1, CHUNK_SIZE - 1, 1); // [B, nv, 1]
        let decay = g_last.exp().unsqueeze(-1);          // [B, nv, 1, 1]
        let g_diff = (g_last - &g_i).exp().unsqueeze(-1); // [B, nv, C, 1]
        let update = (k_i * g_diff).transpose(-1, -2).matmul(&v_new); // [B, nv, hk, hv]
        state = &state * decay + update;
    }

    // Stack chunks: [B, nv, nc, C, hv] → flatten → trim padding → contiguous
    // The .contiguous() after narrow is critical: without it, the resulting tensor has
    // padded strides (stride[0] = nv*seq_p*hv instead of nv*seq*hv) which causes
    // libtorch to miscalculate buffer sizes during subsequent permute+reshape operations.
    let out = Tensor::stack(&chunks_out, 2) // [B, nv, nc, C, hv]
        .reshape([batch, nv, seq_p, hv])
        .narrow(2, 0, seq)
        .contiguous();

    Ok((out, state))
}

// ============================================================================
// GatedDeltaNet (linear attention / SSM layer)
// ============================================================================

struct GatedDeltaNetLayer {
    // Fused input projection: hidden → (conv_dim + value_dim + 2*num_v_heads)
    // Single matmul instead of 4 — reduces kernel launches by 75% for this step.
    in_proj_all: LinearProjection,
    // Split offsets (conv_dim, value_dim, num_v_heads, num_v_heads)
    proj_split: [i64; 4],
    // Depthwise conv1d on mixed QKV
    conv1d_weight: Tensor,         // [conv_dim, 1, kernel_size]
    // Decay and dt bias (stored in Float to avoid per-step casting)
    a_log: Tensor,                 // [num_v_heads] original dtype
    dt_bias: Tensor,               // [num_v_heads] original dtype
    neg_a_exp_f32: Tensor,          // [1, 1, num_v_heads] precomputed -exp(a_log) in Float
    dt_bias_f32: Tensor,           // [1, 1, num_v_heads] precomputed in Float
    // Output norm and projection
    norm: RMSNormGated,
    out_proj: LinearProjection,    // value_dim → hidden
    // Dimensions
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,   // num_k_heads * head_k_dim
    value_dim: usize, // num_v_heads * head_v_dim
    conv_dim: usize,  // key_dim*2 + value_dim
    kernel_size: usize,
    layer_idx: usize,
}

unsafe impl Send for GatedDeltaNetLayer {}
unsafe impl Sync for GatedDeltaNetLayer {}

impl GatedDeltaNetLayer {
    fn load(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        cfg: &Qwen3_5TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let num_k_heads = cfg.linear_num_key_heads;
        let num_v_heads = cfg.linear_num_value_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let kernel_size = cfg.linear_conv_kernel_dim;

        let mut get = |name: &str| -> Result<Tensor> {
            let full = format!("{prefix}.{name}");
            weights.remove(&full).ok_or_else(|| anyhow!("Missing weight: {}", full))
        };

        let conv1d_weight = get("conv1d.weight")?;
        let a_log = get("A_log")?;
        let dt_bias = get("dt_bias")?;
        let norm_weight = get("norm.weight")?;

        let a_log_sq = a_log.squeeze();
        let dt_bias_sq = dt_bias.squeeze();
        // Precompute float versions with broadcast shape [1, 1, nv] for decode path
        let neg_a_exp_f32 = -(a_log_sq.to_kind(Kind::Float).exp()
            .unsqueeze(0).unsqueeze(0));   // [1, 1, nv] already negated
        let dt_bias_f32 = dt_bias_sq.to_kind(Kind::Float)
            .unsqueeze(0).unsqueeze(0);   // [1, 1, nv]

        // Fuse 4 input projections into 1 matmul.
        // Cat along output dim (dim 1, weight is [in, out] after take()), preserving FP8+scale.
        // LinearProjection::apply() dequantizes lazily at forward time — no BF16 copy at load.
        let w_qkv = LinearProjection::take(weights, &format!("{prefix}.in_proj_qkv.weight"))?;
        let w_z   = LinearProjection::take(weights, &format!("{prefix}.in_proj_z.weight"))?;
        let w_b   = LinearProjection::take(weights, &format!("{prefix}.in_proj_b.weight"))?;
        let w_a   = LinearProjection::take(weights, &format!("{prefix}.in_proj_a.weight"))?;
        let in_proj_all = cat_projs(vec![w_qkv, w_z, w_b, w_a]);
        let proj_split = [
            conv_dim as i64,
            value_dim as i64,
            num_v_heads as i64,
            num_v_heads as i64,
        ];

        Ok(Self {
            in_proj_all,
            proj_split,
            conv1d_weight,
            a_log: a_log_sq,
            dt_bias: dt_bias_sq,
            neg_a_exp_f32,
            dt_bias_f32,
            norm: RMSNormGated::new(norm_weight, 1e-6),
            out_proj: LinearProjection::take(weights, &format!("{prefix}.out_proj.weight"))?,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            kernel_size,
            layer_idx,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        conv_state: &mut Option<Tensor>,
        rec_state: &mut Option<Tensor>,
        delta: Option<(&crate::training::TenantDelta, usize)>,
    ) -> Result<Tensor> {
        let (batch, seq, _) = dims3(hidden)?;
        let device = hidden.device();
        let dtype = hidden.kind();

        // Fused input projection: 1 matmul for all 4 projections.
        let h2 = hidden.reshape([batch * seq, hidden.size()[2]]);
        let all_proj = self.in_proj_all.apply(&h2); // [batch*seq, conv_dim+val_dim+2*nv]
        let [cd, vd, nv_d, _] = self.proj_split;
        let mut off = 0i64;
        let mixed_qkv = all_proj.narrow(1, off, cd).contiguous().reshape([batch, seq, cd]); off += cd;
        let z         = all_proj.narrow(1, off, vd).contiguous().reshape([batch, seq, vd]); off += vd;
        let b         = all_proj.narrow(1, off, nv_d).contiguous().reshape([batch, seq, nv_d]); off += nv_d;
        let a_in      = all_proj.narrow(1, off, nv_d).contiguous().reshape([batch, seq, nv_d]);

        // Depthwise causal conv1d on mixed_qkv
        // mixed_qkv: [batch, seq, conv_dim] → need [batch, conv_dim, seq] for conv1d
        let conv_dim = self.conv_dim as i64;
        let ks = self.kernel_size as i64;

        // Maintain conv state: [batch, conv_dim, kernel_size-1]
        let x_t = mixed_qkv.permute([0, 2, 1]); // [batch, conv_dim, seq]

        let (x_for_conv, new_conv_state) = match conv_state {
            Some(ref cs) => {
                // Prepend previous state for causal context
                let padded = Tensor::cat(&[cs, &x_t], 2); // [batch, conv_dim, prev+seq]
                // Keep last kernel_size-1 frames as new state
                let total = padded.size()[2];
                let new_state = padded.narrow(2, total - (ks - 1), ks - 1).contiguous();
                (padded, new_state)
            }
            None => {
                // First call: zero-pad left by kernel_size-1
                let pad = Tensor::zeros([batch, conv_dim, ks - 1], (dtype, device));
                let padded = Tensor::cat(&[&pad, &x_t], 2);
                let new_state = x_t.narrow(2, (seq - (ks - 1)).max(0), (ks - 1).min(seq))
                    .contiguous();
                (padded, new_state)
            }
        };
        *conv_state = Some(new_conv_state);

        // conv1d with groups=conv_dim (depthwise), no padding (already padded manually)
        let conv_out = x_for_conv.conv1d(
            &self.conv1d_weight,
            None::<&Tensor>,
            &[1i64],      // stride=1
            &[0i64],      // padding=0 (handled manually)
            &[1i64],      // dilation=1
            conv_dim,     // groups
        ); // [batch, conv_dim, seq]
        let conv_out = conv_out.silu().permute([0, 2, 1]); // [batch, seq, conv_dim]

        // Split conv_out into q, k, v
        let key_dim = self.key_dim as i64;
        let val_dim = self.value_dim as i64;
        let q_raw = conv_out.narrow(2, 0, key_dim);
        let k_raw = conv_out.narrow(2, key_dim, key_dim);
        let v_raw = conv_out.narrow(2, key_dim * 2, val_dim);

        // Reshape to [batch, seq, num_heads, head_dim]
        let q = q_raw.reshape([batch, seq, self.num_k_heads as i64, self.head_k_dim as i64]);
        let k = k_raw.reshape([batch, seq, self.num_k_heads as i64, self.head_k_dim as i64]);
        let v = v_raw.reshape([batch, seq, self.num_v_heads as i64, self.head_v_dim as i64]);

        // L2 normalize Q and K before delta rule
        let q = l2_normalize(&q.to_kind(Kind::Float));
        let k = l2_normalize(&k.to_kind(Kind::Float));
        let scale = 1.0 / (self.head_k_dim as f64).sqrt();
        let q = q * scale;

        // Compute decay g = A * dt where A = -exp(A_log), dt = softplus(a_in + dt_bias)
        // A_log stores log(-A) so A = -exp(A_log) is negative; g = A * dt is negative log-decay
        // Use precomputed float tensors — avoids to_kind + exp + unsqueeze per step
        let a_in_f = a_in.to_kind(Kind::Float); // [batch, seq, num_v_heads]
        let softplus_input = a_in_f + &self.dt_bias_f32; // [1,1,nv] broadcasts
        let g = &self.neg_a_exp_f32 * softplus_input.softplus(); // [batch, seq, nv]

        // Recurrent delta rule
        // state shape: [batch, num_v_heads, head_k_dim, head_v_dim]
        let nv = self.num_v_heads as i64;
        let hk = self.head_k_dim as i64;
        let hv = self.head_v_dim as i64;

        let state_init = rec_state.take().unwrap_or_else(|| {
            Tensor::zeros([batch, nv, hk, hv], (Kind::Float, device))
        });

        let b_f = b.to_kind(Kind::Float).sigmoid(); // [batch, seq, num_v_heads]
        let v_f = v.to_kind(Kind::Float); // [batch, seq, num_v_heads, head_v_dim]

        // For num_k_heads != num_v_heads, each k head is shared by
        // (num_v_heads / num_k_heads) consecutive v heads (GQA-style interleave).
        // Must use repeat_interleave (not repeat/tile) so head i maps to v heads
        // [i*r, ..., i*r + r-1] rather than [i, i+num_k, ...].
        let k_for_state = if self.num_k_heads == self.num_v_heads {
            k.to_kind(Kind::Float)
        } else {
            let repeats = (self.num_v_heads / self.num_k_heads) as i64;
            k.repeat_interleave_self_int(repeats, 2, None).to_kind(Kind::Float)
        };
        // q similarly expanded with repeat_interleave
        let q_for_state = if self.num_k_heads == self.num_v_heads {
            q.shallow_clone()
        } else {
            let repeats = (self.num_v_heads / self.num_k_heads) as i64;
            q.repeat_interleave_self_int(repeats, 2, None)
        };

        // Choose recurrent path based on sequence length:
        // - seq == 1  (decode): token-by-token recurrent loop
        // - seq > 1   (prefill): chunked algorithm for O(seq · chunk) vs O(seq²)
        let out = if seq == 1 {
            // --- Decode path: single-token recurrent step ---
            let g_t = g.narrow(1, 0, 1).squeeze_dim(1);
            let b_t = b_f.narrow(1, 0, 1).squeeze_dim(1);
            let k_t = k_for_state.narrow(1, 0, 1).squeeze_dim(1);
            let v_t = v_f.narrow(1, 0, 1).squeeze_dim(1);
            let q_t = q_for_state.narrow(1, 0, 1).squeeze_dim(1);

            // state: [B, nv, hk, hv]; k_t, q_t: [B, nv, hk]; v_t, b_t: [B, nv, hv]
            // In-place state decay (Step 9)
            let decay = g_t.exp().reshape([batch, nv, 1, 1]);
            let state = &state_init * &decay;
            // BMM-based readout: state[B*nv, hk, hv] × k_t[B*nv, 1, hk]^T → [B*nv, 1, hv]
            let state_3d = state.reshape([batch * nv, hk, hv]);
            let k_col = k_t.reshape([batch * nv, 1, hk]);
            let kv_mem = k_col.bmm(&state_3d).reshape([batch, nv, hv]);
            let delta = (v_t - kv_mem) * b_t.unsqueeze(-1);
            // outer product update: state[h,k,v] += k[h,k] * delta[h,v]
            let state = state.reshape([batch, nv, hk, hv]) + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
            // BMM-based output readout
            let state_3d = state.reshape([batch * nv, hk, hv]);
            let q_col = q_t.reshape([batch * nv, 1, hk]);
            let out_t = q_col.bmm(&state_3d).reshape([batch, nv, hv]);

            *rec_state = Some(state);
            // out_t: [batch, nv, hv] → [batch, 1, nv*hv]
            out_t.unsqueeze(1).reshape([batch, 1, -1])
        } else {
            // --- Prefill path: chunked GatedDeltaNet ---
            // Reorder to [batch, nv, seq, dim] for chunked_delta_rule
            let q_h = q_for_state.permute([0, 2, 1, 3]);  // [B, nv, T, hk]
            let k_h = k_for_state.permute([0, 2, 1, 3]);
            let v_h = v_f.permute([0, 2, 1, 3]);           // [B, nv, T, hv]
            let g_h = g.permute([0, 2, 1]);                 // [B, nv, T]
            let b_h = b_f.permute([0, 2, 1]);               // [B, nv, T]

            let init = if rec_state.is_some() { rec_state.as_ref() } else { None };
            let (out_h, new_state) = chunked_delta_rule(&q_h, &k_h, &v_h, &g_h, &b_h, init, device)?;
            // out_h: [B, nv, T, hv]
            *rec_state = Some(new_state);

            // Transpose back to [B, T, nv, hv] and flatten to [B, T, nv*hv]
            // Flatten [B, nv, T, hv] → [B, T, nv*hv]
            out_h.permute([0, 2, 1, 3]).contiguous().reshape([batch, seq, -1])
        };

        // Gated norm and output projection
        // RMSNormGated weight is [head_v_dim]; reshape to per-head before norm.
        // Derive actual value dim from out tensor (may differ from config for small models).
        let actual_vd = out.size()[2];
        let actual_hv = self.norm.weight.size()[0]; // norm weight shape = [head_v_dim]
        let actual_nv = actual_vd / actual_hv;
        let z_f = z.narrow(2, 0, actual_vd).contiguous(); // trim z to match actual value dim

        let out_4d = out.reshape([batch, seq, actual_nv, actual_hv]);
        let z_4d = z_f.reshape([batch, seq, actual_nv, actual_hv]);
        let normed = self.norm.forward(&out_4d, &z_4d)?;  // out_4d may be F32 from recurrent step; RMSNormGated handles both
        // Ensure output matches model dtype for out_proj matmul (weight is BF16)
        let normed = if normed.kind() != dtype { normed.to_kind(dtype) } else { normed };
        let flat = normed.reshape([batch * seq, actual_vd]);

        let projected = self.out_proj.apply(&flat);
        // Standard LoRA: add correction in the OUTPUT space of out_proj (y = Wx + BAx).
        // Inject after out_proj so correction shape [B*seq, hidden_size] matches projected,
        // regardless of whether val_dim == hidden_size. (Issue 6 + standard LoRA semantics)
        let projected = if let Some((delta, layer_idx)) = delta {
            if delta.has_module("o_proj", layer_idx) {
                // forward_2d(flat): [B*seq, val_dim] → [B*seq, hidden_size]
                &projected + delta.forward_2d(&flat, "o_proj", layer_idx)?
            } else {
                projected
            }
        } else {
            projected
        };
        Ok(projected.reshape([batch, seq, self.hidden_size_from_out_proj()]))
    }

    fn hidden_size_from_out_proj(&self) -> i64 {
        // weight is stored [in_features, out_features] after transpose at load; size()[1] = out_features = hidden_size
        self.out_proj.weight.size()[1]
    }

    /// Move every owned tensor to `device` (#314 pipeline placement). The SSM
    /// runtime state (`conv`/`rec`) is NOT part of the layer — it is stage-local
    /// and threaded separately — so only the static weights move here.
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self {
            in_proj_all: self.in_proj_all.into_device(device),
            conv1d_weight: self.conv1d_weight.to_device(device),
            a_log: self.a_log.to_device(device),
            dt_bias: self.dt_bias.to_device(device),
            neg_a_exp_f32: self.neg_a_exp_f32.to_device(device),
            dt_bias_f32: self.dt_bias_f32.to_device(device),
            norm: self.norm.into_device(device),
            out_proj: self.out_proj.into_device(device),
            ..self
        }
    }
}

// ============================================================================
// Full attention layer (GQA + output gate + partial RoPE)
// ============================================================================

struct Qwen3_5FullAttention {
    qkv_proj: LinearProjection, // fused: hidden → (q_out + k_out + v_out)
    qkv_split: [i64; 3],       // [q_dim, k_dim, v_dim] for narrow splits
    o_proj: LinearProjection,   // in: num_heads * head_dim
    q_norm: Qwen3_5RMSNorm,    // per-head, weight dim = head_dim
    k_norm: Qwen3_5RMSNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    max_pos: usize,
    layer_idx: usize,
}

unsafe impl Send for Qwen3_5FullAttention {}
unsafe impl Sync for Qwen3_5FullAttention {}

impl Qwen3_5FullAttention {
    fn load(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        cfg: &Qwen3_5TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let q_norm_weight = weights
            .remove(&format!("{prefix}.q_norm.weight"))
            .ok_or_else(|| anyhow!("Missing {prefix}.q_norm.weight"))?;
        let k_norm_weight = weights
            .remove(&format!("{prefix}.k_norm.weight"))
            .ok_or_else(|| anyhow!("Missing {prefix}.k_norm.weight"))?;

        // Fuse Q/K/V projections into single matmul, preserving FP8+scale.
        // LinearProjection::apply() dequantizes lazily at forward time — no BF16 copy at load.
        let q_proj = LinearProjection::take(weights, &format!("{prefix}.q_proj.weight"))?;
        let k_proj = LinearProjection::take(weights, &format!("{prefix}.k_proj.weight"))?;
        let v_proj = LinearProjection::take(weights, &format!("{prefix}.v_proj.weight"))?;
        let q_dim = q_proj.weight.size()[1];
        let k_dim = k_proj.weight.size()[1];
        let v_dim = v_proj.weight.size()[1];
        let fused = cat_projs(vec![q_proj, k_proj, v_proj]);

        Ok(Self {
            qkv_proj: fused,
            qkv_split: [q_dim, k_dim, v_dim],
            o_proj: LinearProjection::take(weights, &format!("{prefix}.o_proj.weight"))?,
            q_norm: Qwen3_5RMSNorm::new(q_norm_weight, cfg.rms_norm_eps),
            k_norm: Qwen3_5RMSNorm::new(k_norm_weight, cfg.rms_norm_eps),
            num_heads: cfg.num_attention_heads as usize,
            num_kv_heads: cfg.num_key_value_heads as usize,
            head_dim: cfg.head_dim as usize,
            rotary_dim: cfg.rotary_dim,
            rope_theta: cfg.rope_theta,
            max_pos: cfg.max_position_embeddings as usize,
            layer_idx,
        })
    }

    /// Move all projection + QK-norm weights to `device` (#314 pipeline placement).
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self {
            qkv_proj: self.qkv_proj.into_device(device),
            o_proj: self.o_proj.into_device(device),
            q_norm: self.q_norm.into_device(device),
            k_norm: self.k_norm.into_device(device),
            ..self
        }
    }

    fn forward(
        &self,
        hidden: &Tensor,
        position_ids: Option<&Tensor>,
        kv_cache: Option<&mut crate::runtime::kv_cache::LayerKVCache>,
        start_pos: usize,
        delta: Option<(&crate::training::TenantDelta, usize)>,
    ) -> Result<Tensor> {
        let (batch, seq, _) = dims3(hidden)?;
        let device = hidden.device();
        let dtype = hidden.kind();
        let h2 = hidden.reshape([batch * seq, hidden.size()[2]]);

        // Fused QKV projection: single matmul, then split
        let qkv = self.qkv_proj.apply(&h2);
        let [q_dim, k_dim, v_dim] = self.qkv_split;

        // Delta injection on q_proj and v_proj (out-of-place to avoid aliasing, C2 fix)
        // q_out layout: [B*seq, q_dim] where q_dim = num_heads * head_dim * 2
        //   first half = Q, second half = gate
        // Must apply correction to Q-half only, then reassemble with gate half
        let (q_out, v_out) = if let Some((d, layer_idx)) = delta {
            let q_raw = qkv.narrow(1, 0, q_dim);
            let v_raw = qkv.narrow(1, q_dim + k_dim, v_dim);

            let q_out = if d.has_module("q_proj", layer_idx) {
                // q_dim = num_heads * head_dim * 2; split into Q and gate halves
                let q_half = q_dim / 2;
                let q_part = q_raw.narrow(1, 0, q_half);
                let gate_part = q_raw.narrow(1, q_half, q_dim - q_half);
                let q_correction = d.forward_2d(&h2, "q_proj", layer_idx)?;
                // q_correction shape: [B*seq, q_half] (delta target is the pre-gate Q only)
                // If correction has full q_dim, only apply to q-half
                let q_correction = if q_correction.size()[1] == q_half {
                    q_correction
                } else {
                    q_correction.narrow(1, 0, q_half)
                };
                let corrected_q = &q_part + &q_correction;
                Tensor::cat(&[corrected_q, gate_part], 1)
            } else {
                q_raw
            };

            let v_out = if d.has_module("v_proj", layer_idx) {
                &v_raw + d.forward_2d(&h2, "v_proj", layer_idx)?
            } else {
                v_raw
            };
            (q_out, v_out)
        } else {
            (qkv.narrow(1, 0, q_dim), qkv.narrow(1, q_dim + k_dim, v_dim))
        };
        let k_out = qkv.narrow(1, q_dim, k_dim);

        let nh = self.num_heads as i64;
        let nkv = self.num_kv_heads as i64;
        let hd = self.head_dim as i64;

        // Split query and gate (from doubled q_proj output)
        // Reference: gate.reshape(batch, seq, num_heads * head_dim), then * sigmoid(gate) before o_proj
        let q_full = q_out.reshape([batch, seq, nh, hd * 2]);
        let query = q_full.narrow(3, 0, hd);  // [batch, seq, nh, head_dim]
        let gate = q_full.narrow(3, hd, hd);  // [batch, seq, nh, head_dim]
        // gate will be applied as sigmoid(gate.reshape(batch, seq, nh*hd)) * attn_output BEFORE o_proj

        let mut k = k_out.reshape([batch, seq, nkv, hd]);
        let v = v_out.reshape([batch, seq, nkv, hd]);

        // Per-head QK norm
        let query = self.apply_qknorm(&query, &self.q_norm, self.num_heads)?;
        k = self.apply_qknorm(&k, &self.k_norm, self.num_kv_heads)?;

        // Partial RoPE: rotate first rotary_dim dims, pass through rest
        let rd = self.rotary_dim as i64;
        let (query, k) = if rd > 0 {
            let query_rot = self.apply_rope(&query.narrow(3, 0, rd), position_ids, start_pos, device, dtype)?;
            let query_pass = query.narrow(3, rd, hd - rd);
            let query = Tensor::cat(&[query_rot, query_pass], 3);

            let k_rot = self.apply_rope(&k.narrow(3, 0, rd), position_ids, start_pos, device, dtype)?;
            let k_pass = k.narrow(3, rd, hd - rd);
            let k = Tensor::cat(&[k_rot, k_pass], 3);
            (query, k)
        } else {
            (query, k)
        };

        // KV cache
        let (k_full, v_full) = if let Some(cache) = kv_cache {
            cache.update(&k, &v, start_pos)?;
            cache.get()?
        } else {
            (k.shallow_clone(), v.shallow_clone())
        };

        // Expand KV for GQA
        let groups = self.num_heads / self.num_kv_heads;
        let k_exp = k_full.repeat_interleave_self_int(groups as i64, 2, None);
        let v_exp = v_full.repeat_interleave_self_int(groups as i64, 2, None);

        // Manual attention in f32 for numerical stability. For long prefills the
        // query axis is processed in chunks so the [q_seq, kv_seq] FP32 score and
        // softmax tensors are never materialized at full size — peak VRAM stays
        // O(ATTN_CHUNK * kv_seq) rather than O(q_seq * kv_seq). An 8K-token prompt
        // otherwise pushes these full-attn layers past 13 GB, and the CUDA caching
        // allocator retains that peak, OOMing the next request. The decode path
        // (q_seq == 1) and short prompts take the single-shot branch unchanged.
        const ATTN_CHUNK: i64 = 1024;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_p = query.permute([0, 2, 1, 3]).to_kind(Kind::Float).contiguous();
        let k_p = k_exp.permute([0, 2, 3, 1]).to_kind(Kind::Float).contiguous();
        let v_p = v_exp.permute([0, 2, 1, 3]).to_kind(Kind::Float).contiguous();

        let q_len = q_p.size()[2];
        let kv_len = k_p.size()[3];
        let ctx = if q_len <= ATTN_CHUNK {
            let mut scores = q_p.matmul(&k_p) * scale; // [batch, heads, q_seq, kv_seq]
            if q_len > 1 {
                let mask = Tensor::ones([q_len, kv_len], (Kind::Float, device)).tril(0);
                let mask = mask.unsqueeze(0).unsqueeze(0).expand_as(&scores);
                scores = scores.masked_fill(&mask.eq(0.0), -10000.0f64);
            }
            let attn = scores.softmax(-1, Kind::Float).to_kind(dtype);
            attn.matmul(&v_p.to_kind(dtype)) // [batch, heads, q_seq, head_dim]
        } else {
            // Chunk over the query axis. Per-chunk causal mask `tril(start)` is the
            // exact restriction of the full `tril(0)` mask to rows [start, start+cur).
            let v_d = v_p.to_kind(dtype);
            let mut outs: Vec<Tensor> = Vec::new();
            let mut start = 0i64;
            while start < q_len {
                let cur = (q_len - start).min(ATTN_CHUNK);
                let q_chunk = q_p.narrow(2, start, cur); // [batch, heads, cur, head_dim]
                let mut scores = q_chunk.matmul(&k_p) * scale; // [batch, heads, cur, kv_seq]
                let mask = Tensor::ones([cur, kv_len], (Kind::Float, device)).tril(start);
                let mask = mask.unsqueeze(0).unsqueeze(0).expand_as(&scores);
                scores = scores.masked_fill(&mask.eq(0.0), -10000.0f64);
                let attn = scores.softmax(-1, Kind::Float).to_kind(dtype);
                outs.push(attn.matmul(&v_d)); // [batch, heads, cur, head_dim]
                start += cur;
            }
            Tensor::cat(&outs, 2) // [batch, heads, q_seq, head_dim]
        };
        let ctx = ctx.permute([0, 2, 1, 3]).contiguous(); // [batch, seq, heads, head_dim]

        // Apply output gate: gate_flat[batch*seq, nh*hd] * sigmoid(gate) BEFORE o_proj
        let ctx_flat = ctx.reshape([batch * seq, nh * hd]);
        let gate_flat = gate.reshape([batch * seq, nh * hd]).sigmoid();
        let ctx_gated = ctx_flat * gate_flat;

        // Delta injection on o_proj (after gate, before linear projection)
        let ctx_gated = if let Some((d, layer_idx)) = delta {
            if d.has_module("o_proj", layer_idx) {
                &ctx_gated + d.forward_2d(&ctx_gated, "o_proj", layer_idx)?
            } else {
                ctx_gated
            }
        } else {
            ctx_gated
        };

        let out = self.o_proj.apply(&ctx_gated);
        Ok(out.reshape([batch, seq, -1]))
    }

    fn apply_qknorm(&self, x: &Tensor, norm: &Qwen3_5RMSNorm, _nheads: usize) -> Result<Tensor> {
        // x: [batch, seq, heads, head_dim]
        let orig_shape = x.size();
        let (batch, seq, heads, hd) = dims4(x)?;
        let flat = x.reshape([batch * seq * heads, hd]);
        let orig = flat.kind();
        let flat_f = flat.to_kind(Kind::Float);
        let mean_sq = (&flat_f * &flat_f).mean_dim(&[-1i64][..], true, Kind::Float);
        let rrms = (mean_sq + norm.eps as f64).rsqrt();
        let normed = &flat_f * rrms;
        // q_norm/k_norm weights are near-0 init — use (1+w)*normed formula (same as Qwen3_5RMSNorm)
        let one_plus_w = norm.weight_f32.reshape([1, hd]) + 1.0f64;
        let normed_scaled = normed * one_plus_w;
        Ok(normed_scaled.to_kind(orig).reshape(orig_shape))
    }

    fn apply_rope(
        &self,
        x: &Tensor,
        position_ids: Option<&Tensor>,
        start_pos: usize,
        device: Device,
        dtype: Kind,
    ) -> Result<Tensor> {
        // Thread-local RoPE cache: reuses sin/cos tables across decode steps.
        // Saves ~9 GPU kernels per call (generate_embeddings) after the first token.
        // Same pattern as LlamaModel::apply_rope.
        use std::cell::RefCell;
        // Cache key includes the DEVICE: the cached RoPE holds sin/cos tables on
        // a specific device. Under a pipeline split (#314) the same `layer_idx`
        // could legitimately run on different devices on one thread, so omitting
        // device would hand back tables on the wrong device. We encode the device
        // as an i64 (`-1` = CPU, otherwise the CUDA ordinal) since tch::Device is
        // not Hash. On the single-device path this just adds one constant key dim.
        thread_local! {
            static ROPE_CACHE: RefCell<HashMap<(usize, u32, i64), RoPE>> =
                RefCell::new(HashMap::new());
        }
        let device_key: i64 = match device {
            Device::Cuda(i) => i as i64,
            _ => -1,
        };

        let rd = self.rotary_dim as i64;
        let seq = x.size()[1];
        let pos_ids = position_ids.map(|p| p.shallow_clone()).unwrap_or_else(|| {
            Tensor::arange_start(start_pos as i64, start_pos as i64 + seq, (Kind::Int64, device))
        });

        ROPE_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let key = (self.layer_idx, self.rope_theta.to_bits(), device_key);
            let rope = match cache.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(RoPE::new_with_dtype(rd, self.rope_theta as f64, self.max_pos as i64, device, dtype)?)
                }
            };
            rope.forward(x, Some(&pos_ids))
        })
    }
}

// ============================================================================
// MoE (Sparse MLP)
// ============================================================================

struct Qwen3_5Expert {
    gate_proj: LinearProjection,
    up_proj: LinearProjection,
    down_proj: LinearProjection,
}

impl Qwen3_5Expert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.apply(x).silu();
        let up = self.up_proj.apply(x);
        Ok(self.down_proj.apply(&(gate * up)))
    }

    /// Move all projection weights to `device` (#314 pipeline placement).
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self {
            gate_proj: self.gate_proj.into_device(device),
            up_proj: self.up_proj.into_device(device),
            down_proj: self.down_proj.into_device(device),
        }
    }
}

unsafe impl Send for Qwen3_5Expert {}
unsafe impl Sync for Qwen3_5Expert {}

struct Qwen3_5SparseMoE {
    gate: LinearProjection,     // router: [hidden, num_experts]
    // Stacked expert weights — [num_experts, in_dim, out_dim], kept in FP8 to save VRAM.
    // Companion scales — [num_experts, in/128, out/128], BF16. None for non-FP8 weights.
    // Dequantized lazily after index_select (only k=8 experts per forward, not all 256).
    expert_gate_w: Tensor,               // [num_experts, hidden, moe_int]  FP8 or BF16
    expert_gate_scale: Option<Tensor>,   // [num_experts, hidden/128, moe_int/128]
    expert_up_w: Tensor,                 // [num_experts, hidden, moe_int]
    expert_up_scale: Option<Tensor>,
    expert_down_w: Tensor,               // [num_experts, moe_int, hidden]
    expert_down_scale: Option<Tensor>,
    shared_expert: Qwen3_5Expert,
    /// Optional gating weight for shared expert: [hidden, 1] (transposed from [1, hidden]).
    /// When present: shared_out *= sigmoid(x @ shared_expert_gate)
    shared_expert_gate: Option<LinearProjection>,
    num_experts_per_tok: usize,
}

impl Qwen3_5SparseMoE {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, hidden]
        let batch = x.size()[0];
        let hidden = x.size()[1];
        let k = self.num_experts_per_tok as i64;
        let dtype = x.kind();

        // Router — topK on raw logits, then softmax over selected experts only.
        // Avoids softmax over all 256 experts + renormalize div+sum.
        let logits = self.gate.apply(x);  // [batch, num_experts]
        let (top_logits, top_ids) = logits.topk(k, -1, true, true);
        let top_weights = top_logits.softmax(-1, dtype);  // softmax over k=8 only

        // Gather expert weights on GPU — no CPU sync.
        // top_ids: [batch, k] → [batch*k]
        let top_ids_flat = top_ids.reshape([-1]);

        // Select k experts and dequantize FP8 → BF16 (only k=8 of 256, not all at once).
        let gate_sel = dequant_batched(
            &self.expert_gate_w.index_select(0, &top_ids_flat),
            self.expert_gate_scale.as_ref().map(|s| s.index_select(0, &top_ids_flat)).as_ref(),
        );  // [batch*k, hidden, moe_int] BF16
        let up_sel = dequant_batched(
            &self.expert_up_w.index_select(0, &top_ids_flat),
            self.expert_up_scale.as_ref().map(|s| s.index_select(0, &top_ids_flat)).as_ref(),
        );

        // Repeat input k times: [batch*k, hidden]
        let x_rep = if x.kind() != tch::Kind::BFloat16 {
            x.repeat_interleave_self_int(k, 0, None).to_kind(tch::Kind::BFloat16)
        } else {
            x.repeat_interleave_self_int(k, 0, None)
        };

        // Fused gate+up BMM: cat weights along output dim, single BMM, then split
        let moe_int = gate_sel.size()[2];
        let gate_up_sel = Tensor::cat(&[&gate_sel, &up_sel], 2); // [batch*k, hidden, 2*moe_int]
        let x_3d = x_rep.unsqueeze(1); // [batch*k, 1, hidden]
        let gate_up = x_3d.bmm(&gate_up_sel).squeeze_dim(1); // [batch*k, 2*moe_int]
        let gate_out = gate_up.narrow(1, 0, moe_int).silu();
        let up_out = gate_up.narrow(1, moe_int, moe_int);

        // [batch*k, 1, moe_int] × [batch*k, moe_int, hidden] → [batch*k, hidden]
        let down_sel = dequant_batched(
            &self.expert_down_w.index_select(0, &top_ids_flat),
            self.expert_down_scale.as_ref().map(|s| s.index_select(0, &top_ids_flat)).as_ref(),
        );  // [batch*k, moe_int, hidden] BF16
        let expert_out = (gate_out * up_out).unsqueeze(1).bmm(&down_sel).squeeze_dim(1);

        // Weighted sum: [batch, k, hidden], then reduce over k.
        let expert_out = expert_out.reshape([batch, k, hidden]);
        let output = (expert_out * top_weights.unsqueeze(-1).to_kind(tch::Kind::BFloat16))
            .sum_dim_intlist(&[1i64][..], false, None);
        let output = if dtype != tch::Kind::BFloat16 { output.to_kind(dtype) } else { output };

        let shared_out = self.shared_expert.forward(x)?;
        let shared_out = if let Some(gate_proj) = &self.shared_expert_gate {
            // sigmoid(x @ gate_proj): [batch, 1], broadcast over hidden dim
            let gate = gate_proj.apply(x).sigmoid();
            shared_out * gate
        } else {
            shared_out
        };

        Ok(output + shared_out)
    }

    /// Move the router, stacked expert weights/scales, and shared expert to
    /// `device` (#314 pipeline placement).
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self {
            gate: self.gate.into_device(device),
            expert_gate_w: self.expert_gate_w.to_device(device),
            expert_gate_scale: self.expert_gate_scale.map(|s| s.to_device(device)),
            expert_up_w: self.expert_up_w.to_device(device),
            expert_up_scale: self.expert_up_scale.map(|s| s.to_device(device)),
            expert_down_w: self.expert_down_w.to_device(device),
            expert_down_scale: self.expert_down_scale.map(|s| s.to_device(device)),
            shared_expert: self.shared_expert.into_device(device),
            shared_expert_gate: self.shared_expert_gate.map(|g| g.into_device(device)),
            num_experts_per_tok: self.num_experts_per_tok,
        }
    }
}

unsafe impl Send for Qwen3_5SparseMoE {}
unsafe impl Sync for Qwen3_5SparseMoE {}

// ============================================================================
// MLP enum (Dense or Sparse)
// ============================================================================

enum Qwen3_5Mlp {
    Dense(LlamaMLP),
    Sparse(Qwen3_5SparseMoE),
}

impl Qwen3_5Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Qwen3_5Mlp::Dense(mlp) => mlp.forward(x, None),
            Qwen3_5Mlp::Sparse(moe) => {
                let orig = x.size();
                let batch = orig[0];
                let seq = orig[1];
                let h = orig[2];
                let flat = x.reshape([batch * seq, h]);
                let out = moe.forward(&flat)?;
                Ok(out.reshape(orig))
            }
        }
    }

    /// Move the dense or sparse MLP weights to `device` (#314 pipeline placement).
    #[inline]
    fn into_device(self, device: Device) -> Self {
        match self {
            Qwen3_5Mlp::Dense(mlp) => Qwen3_5Mlp::Dense(mlp.into_device(device)),
            Qwen3_5Mlp::Sparse(moe) => Qwen3_5Mlp::Sparse(moe.into_device(device)),
        }
    }
}

unsafe impl Send for Qwen3_5Mlp {}
unsafe impl Sync for Qwen3_5Mlp {}

// ============================================================================
// Transformer layer
// ============================================================================

enum LayerMixer {
    LinearAttn(GatedDeltaNetLayer),
    FullAttn(Qwen3_5FullAttention),
}

impl LayerMixer {
    /// Move the mixer's weights to `device` (#314 pipeline placement).
    #[inline]
    fn into_device(self, device: Device) -> Self {
        match self {
            LayerMixer::LinearAttn(gdn) => LayerMixer::LinearAttn(gdn.into_device(device)),
            LayerMixer::FullAttn(attn) => LayerMixer::FullAttn(attn.into_device(device)),
        }
    }
}

unsafe impl Send for LayerMixer {}
unsafe impl Sync for LayerMixer {}

struct Qwen3_5Layer {
    mixer: LayerMixer,
    mlp: Qwen3_5Mlp,
    input_layernorm: Qwen3_5RMSNorm,
    post_attention_layernorm: Qwen3_5RMSNorm,
}

impl Qwen3_5Layer {
    /// Move every weight in this layer (mixer + MLP + both norms) to `device`
    /// (#314 pipeline placement). A no-op per tensor already resident on `device`.
    #[inline]
    fn into_device(self, device: Device) -> Self {
        Self {
            mixer: self.mixer.into_device(device),
            mlp: self.mlp.into_device(device),
            input_layernorm: self.input_layernorm.into_device(device),
            post_attention_layernorm: self.post_attention_layernorm.into_device(device),
        }
    }
}

unsafe impl Send for Qwen3_5Layer {}
unsafe impl Sync for Qwen3_5Layer {}

// ============================================================================
// Top-level model
// ============================================================================

pub struct Qwen3_5Model {
    config: Qwen3_5TextConfig,
    device: Device,
    dtype: Kind,

    /// Per-global-layer device assignment (#314, 2b pipeline). For the unsplit
    /// fast path this is a single-device map of length `num_hidden_layers`
    /// (`is_single_device() == true`); for a pipeline shard it is the *global*
    /// map (still length `num_hidden_layers`) and `layer_offset` selects the owned
    /// window. Drives both per-layer placement at construction and the lone
    /// boundary `to_device` in `forward_layers`. Mirrors `LlamaModel`.
    device_map: LayerDeviceMap,

    /// Global index of `self.layers[0]`. `0` for a whole model; `a` for a shard
    /// owning global layers `[a..a+self.layers.len())`. Used to remap a global
    /// layer index to its local slot in `self.layers` / KV / conv-rec state, and
    /// to pick the correct per-global-index `layer_types` entry at load.
    layer_offset: usize,

    embed_tokens: Tensor,
    layers: Vec<Qwen3_5Layer>,
    norm: Qwen3_5RMSNorm,
    lm_head: Option<Tensor>,             // None = tied to embed_tokens
    lm_head_transposed: Option<Tensor>,  // pre-transposed tied weights
    // Interior-mutable SSM state (required because forward_with_cache takes &self).
    // Sized to the OWNED layer count (== self.layers.len()), indexed by LOCAL
    // layer index. Stage-local: never transferred across a pipeline boundary.
    conv_states: Arc<parking_lot::Mutex<Vec<Option<Tensor>>>>,
    rec_states: Arc<parking_lot::Mutex<Vec<Option<Tensor>>>>,
    kv_cache: Option<Arc<parking_lot::Mutex<KVCacheManager>>>,
    // Vision encoder (optional — None for text-only weights)
    vision_encoder: Option<Qwen3_5VisionEncoder>,
    // Linear projection: vision out_hidden_size → text hidden_size
    vision_projector: Option<LinearProjection>,
}

unsafe impl Send for Qwen3_5Model {}
unsafe impl Sync for Qwen3_5Model {}

impl Qwen3_5Model {
    pub fn from_weights(
        weights: &mut HashMap<String, Tensor>,
        cfg: Qwen3_5TextConfig,
        vision_cfg: Option<Qwen3_5VisionConfig>,
        device: &Device,
        dtype: Kind,
        _kv_quant_type: KVQuantType,
    ) -> Result<Self> {
        // Normalize weight key prefixes:
        // Qwen3.5 Instruct weights use "model.language_model." and "model.visual." prefixes
        // Strip these so the rest of the loader sees the expected flat names
        let needs_remap = weights.keys().any(|k| k.starts_with("model.language_model."));
        if needs_remap {
            info!("Remapping Qwen3.5 weight prefixes (model.language_model.* → *, model.visual.* → visual.*)");
            let remapped: HashMap<String, Tensor> = weights
                .drain()
                .map(|(k, v)| {
                    // model.language_model.X → model.X  (restores expected prefix)
                    // model.visual.X        → visual.X  (handled by vision loader)
                    let new_key = if let Some(rest) = k.strip_prefix("model.language_model.") {
                        format!("model.{rest}")
                    } else if let Some(rest) = k.strip_prefix("model.visual.") {
                        format!("visual.{rest}")
                    } else {
                        k
                    };
                    (new_key, v)
                })
                .collect();
            *weights = remapped;
        }

        // Load vision encoder if visual weights are present
        let has_vision = weights.keys().any(|k| k.starts_with("visual."));
        let (vision_encoder, vision_projector) = if has_vision {
            if let Some(vcfg) = vision_cfg {
                info!("Loading Qwen3.5 vision encoder (out_hidden_size={})", vcfg.out_hidden_size);
                let out_hidden = vcfg.out_hidden_size;
                let text_hidden = cfg.hidden_size;
                // Best-effort: vision key naming varies across model sizes; skip on error
                match Qwen3_5VisionEncoder::from_weights(weights, vcfg) {
                    Ok(enc) => {
                        let proj_key = if weights.contains_key("visual_projector.weight") {
                            Some("visual_projector.weight")
                        } else if weights.contains_key("vision_projector.weight") {
                            Some("vision_projector.weight")
                        } else {
                            None
                        };
                        let projector = if let Some(key) = proj_key {
                            Some(LinearProjection::take(weights, key)?)
                        } else {
                            info!("No vision_projector weights found; vision output will be zeros");
                            let w = Tensor::zeros([out_hidden as i64, text_hidden as i64], (dtype, *device));
                            Some(LinearProjection::new(w))
                        };
                        (Some(enc), projector)
                    }
                    Err(e) => {
                        info!("Vision encoder load skipped (key naming mismatch, text-only mode): {e}");
                        weights.retain(|k, _| !k.starts_with("visual."));
                        (None, None)
                    }
                }
            } else {
                // No vision config provided — discard visual weights
                weights.retain(|k, _| !k.starts_with("visual."));
                (None, None)
            }
        } else {
            (None, None)
        };

        // Remove MTP (multi-token prediction) weights silently
        weights.retain(|k, _| !k.starts_with("mtp."));

        // #405: place top-level weights on `device`, mirroring the
        // `into_device` move that `stage_from_weights_with_config` applies.
        // Without this, constructing from CPU weights with device=Cuda(0)
        // leaves embed/norm/lm_head on CPU — a mixed-device model.
        let embed = weights.remove("model.embed_tokens.weight")
            .ok_or_else(|| anyhow!("Missing model.embed_tokens.weight"))?
            .to_device(*device);
        let norm_w = weights.remove("model.norm.weight")
            .ok_or_else(|| anyhow!("Missing model.norm.weight"))?
            .to_device(*device);
        let lm_head_w = if let Some(w) = weights.remove("lm_head.weight") {
            // Cast FP8 lm_head to BF16, applying the companion block scale if present.
            // Without this the raw FP8 values (~448) are used instead of the true weights (~0.09),
            // which gives completely wrong logit distributions.
            match w.kind() {
                tch::Kind::Float8e4m3fn | tch::Kind::Float8e5m2 => {
                    let w_bf = w.to_kind(tch::Kind::BFloat16);
                    if let Some(s) = weights.remove("lm_head.weight_scale_inv") {
                        // scale stored as [vocab/128, hidden/128]; apply block-wise
                        let ws = w_bf.size();  // [vocab, hidden]
                        let ss = s.size();     // [vocab/128, hidden/128]
                        let br = ws[0] / ss[0];
                        let bc = ws[1] / ss[1];
                        let w_4d = w_bf.view([ss[0], br, ss[1], bc]);
                        let s_4d = s.to_kind(tch::Kind::BFloat16).view([ss[0], 1i64, ss[1], 1i64]);
                        Some((w_4d * s_4d).reshape([ws[0], ws[1]]).to_device(*device))
                    } else {
                        Some(w_bf.to_device(*device))
                    }
                }
                _ => Some(w.to_device(*device)),
            }
        } else {
            None
        };

        let num_layers = cfg.num_hidden_layers as usize;
        let mut layers = Vec::with_capacity(num_layers);

        for idx in 0..num_layers {
            layers.push(Self::build_layer(weights, &cfg, idx)?);
        }

        let lm_head_transposed = if lm_head_w.is_none() {
            // embed may be FP8; cast to BF16 for matmul
            let embed_bf16 = match embed.kind() {
                tch::Kind::Float8e4m3fn | tch::Kind::Float8e5m2 => embed.to_kind(tch::Kind::BFloat16),
                _ => embed.shallow_clone(),
            };
            Some(embed_bf16.tr().contiguous())
        } else {
            None
        };

        let conv_states = Arc::new(parking_lot::Mutex::new(
            (0..num_layers).map(|_| None::<Tensor>).collect::<Vec<_>>(),
        ));
        let rec_states = Arc::new(parking_lot::Mutex::new(
            (0..num_layers).map(|_| None::<Tensor>).collect::<Vec<_>>(),
        ));

        // Create KV cache for full-attention layers (same pattern as LlamaModel)
        let kv_cache = Some(Arc::new(parking_lot::Mutex::new(
            crate::runtime::kv_cache::KVCacheManager::new(
                num_layers,
                cfg.max_position_embeddings as usize,
                _kv_quant_type,
            ),
        )));

        // Unsplit fast path: every layer on the one device. This keeps the
        // whole-model forward byte-identical — `forward_layers` over this
        // single-device map performs zero cross-device copies (#314).
        let device_map = LayerDeviceMap::single(*device, num_layers.max(1))?;

        Ok(Self {
            config: cfg,
            device: *device,
            dtype,
            device_map,
            layer_offset: 0,
            embed_tokens: embed,
            layers,
            norm: Qwen3_5RMSNorm::new(norm_w, 1e-6),
            lm_head: lm_head_w,
            lm_head_transposed,
            conv_states,
            rec_states,
            kv_cache,
            vision_encoder,
            vision_projector,
        })
    }

    fn load_mlp(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        cfg: &Qwen3_5TextConfig,
        layer_idx: usize,
    ) -> Result<Qwen3_5Mlp> {
        if cfg.is_moe {
            // gate: [num_experts, hidden] → [hidden, num_experts]
            let gate_proj = LinearProjection::take(weights, &format!("{prefix}.gate.weight"))?;

            // Stack expert weights as [num_experts, in_dim, out_dim] for GPU-native BMM dispatch.
            // Keep FP8 weights as-is; stack companion scales as [num_experts, in/128, out/128].
            // Dequantization happens lazily after index_select in forward() — only k=8 experts,
            // not all 256, keeping peak VRAM at FP8 size (~35 GB) instead of BF16 (~70 GB).
            let n = cfg.num_experts;
            let mut gate_w_vecs: Vec<Tensor> = Vec::with_capacity(n);
            let mut gate_s_vecs: Vec<Tensor> = Vec::with_capacity(n);
            let mut up_w_vecs: Vec<Tensor> = Vec::with_capacity(n);
            let mut up_s_vecs: Vec<Tensor> = Vec::with_capacity(n);
            let mut down_w_vecs: Vec<Tensor> = Vec::with_capacity(n);
            let mut down_s_vecs: Vec<Tensor> = Vec::with_capacity(n);
            for e in 0..n {
                let ep = format!("{prefix}.experts.{e}");
                let g = LinearProjection::take(weights, &format!("{ep}.gate_proj.weight"))?;
                let u = LinearProjection::take(weights, &format!("{ep}.up_proj.weight"))?;
                let d = LinearProjection::take(weights, &format!("{ep}.down_proj.weight"))?;
                gate_w_vecs.push(g.weight.unsqueeze(0));
                if let Some(s) = g.scale { gate_s_vecs.push(s.unsqueeze(0)); }
                up_w_vecs.push(u.weight.unsqueeze(0));
                if let Some(s) = u.scale { up_s_vecs.push(s.unsqueeze(0)); }
                down_w_vecs.push(d.weight.unsqueeze(0));
                if let Some(s) = d.scale { down_s_vecs.push(s.unsqueeze(0)); }
            }
            // Stack: [num_experts, in, out] (FP8) + optional [num_experts, in/128, out/128] scale
            let expert_gate_w     = Tensor::cat(&gate_w_vecs.iter().collect::<Vec<_>>(), 0);
            let expert_gate_scale = (!gate_s_vecs.is_empty()).then(|| Tensor::cat(&gate_s_vecs.iter().collect::<Vec<_>>(), 0));
            let expert_up_w       = Tensor::cat(&up_w_vecs.iter().collect::<Vec<_>>(), 0);
            let expert_up_scale   = (!up_s_vecs.is_empty()).then(|| Tensor::cat(&up_s_vecs.iter().collect::<Vec<_>>(), 0));
            let expert_down_w     = Tensor::cat(&down_w_vecs.iter().collect::<Vec<_>>(), 0);
            let expert_down_scale = (!down_s_vecs.is_empty()).then(|| Tensor::cat(&down_s_vecs.iter().collect::<Vec<_>>(), 0));

            let sp = format!("{prefix}.shared_expert");
            let shared = Qwen3_5Expert {
                gate_proj: LinearProjection::take(weights, &format!("{sp}.gate_proj.weight"))?,
                up_proj: LinearProjection::take(weights, &format!("{sp}.up_proj.weight"))?,
                down_proj: LinearProjection::take(weights, &format!("{sp}.down_proj.weight"))?,
            };
            // Optional gating scalar for shared expert (present in 35B FP8, not all models).
            let shared_expert_gate = LinearProjection::take(
                weights, &format!("{prefix}.shared_expert_gate.weight")
            ).ok();

            Ok(Qwen3_5Mlp::Sparse(Qwen3_5SparseMoE {
                gate: gate_proj,
                expert_gate_w,
                expert_gate_scale,
                expert_up_w,
                expert_up_scale,
                expert_down_w,
                expert_down_scale,
                shared_expert: shared,
                shared_expert_gate,
                num_experts_per_tok: cfg.num_experts_per_tok,
            }))
        } else {
            Ok(Qwen3_5Mlp::Dense(LlamaMLP {
                gate_proj: LinearProjection::take(weights, &format!("{prefix}.gate_proj.weight"))?,
                up_proj: LinearProjection::take(weights, &format!("{prefix}.up_proj.weight"))?,
                down_proj: LinearProjection::take(weights, &format!("{prefix}.down_proj.weight"))?,
                activation: "silu".to_owned(),
                layer_idx,
            }))
        }
    }

    /// Build a single decoder layer for **global** index `g`, selecting the
    /// hybrid mixer (GatedDeltaNet vs full attention) by the *global* per-layer
    /// type (`cfg.layer_types[g]`). Shared by the whole-model loader and the
    /// pipeline stage loader — the latter passes the global index so a shard that
    /// owns `[a..b)` still picks the correct type for each layer it owns.
    fn build_layer(
        weights: &mut HashMap<String, Tensor>,
        cfg: &Qwen3_5TextConfig,
        g: usize,
    ) -> Result<Qwen3_5Layer> {
        let layer_prefix = format!("model.layers.{g}");

        let input_norm_w = weights
            .remove(&format!("{layer_prefix}.input_layernorm.weight"))
            .ok_or_else(|| anyhow!("Missing {layer_prefix}.input_layernorm.weight"))?;
        let post_norm_w = weights
            .remove(&format!("{layer_prefix}.post_attention_layernorm.weight"))
            .ok_or_else(|| anyhow!("Missing {layer_prefix}.post_attention_layernorm.weight"))?;

        // Hybrid layer type is selected by GLOBAL index: every Nth layer is full
        // attention, the rest GatedDeltaNet. A stage owning [a..b) must consult
        // the global index `g`, not a local one, to load the right mixer.
        let layer_type = cfg
            .layer_types
            .get(g)
            .map(|s| s.as_str())
            .unwrap_or("linear_attention");
        info!("Loading layer {g}: {layer_type}");

        let mixer = if layer_type == "full_attention" {
            let attn_prefix = format!("{layer_prefix}.self_attn");
            LayerMixer::FullAttn(Qwen3_5FullAttention::load(weights, &attn_prefix, cfg, g)?)
        } else {
            let lin_prefix = format!("{layer_prefix}.linear_attn");
            LayerMixer::LinearAttn(GatedDeltaNetLayer::load(weights, &lin_prefix, cfg, g)?)
        };

        let mlp = Self::load_mlp(weights, &format!("{layer_prefix}.mlp"), cfg, g)?;

        Ok(Qwen3_5Layer {
            mixer,
            mlp,
            input_layernorm: Qwen3_5RMSNorm::new(input_norm_w, cfg.rms_norm_eps),
            post_attention_layernorm: Qwen3_5RMSNorm::new(post_norm_w, cfg.rms_norm_eps),
        })
    }

    /// Build a **single pipeline stage** of a Qwen3.5 model (#314, 2b layer-split).
    ///
    /// Mirrors `LlamaModel::stage_from_weights_with_config`, plus Qwen3.5's
    /// specifics:
    /// - **Hybrid layer type** is chosen by the *global* layer index inside
    ///   [`Self::build_layer`] (`cfg.layer_types[g]`), so a shard owning `[a..b)`
    ///   loads the correct GDN/full-attn mixer per global layer.
    /// - **SSM state** (`conv_states`/`rec_states`) is sized to the OWNED layer
    ///   count (`layer_range.len()`), indexed by the LOCAL layer index, and is
    ///   stage-local — never transferred across a pipeline boundary (M-LOAD seam
    ///   #2). Only `hidden` + `start_pos` cross a boundary.
    ///
    /// Non-layer weights are gated by stage position (M-LOAD seam #1):
    /// - `is_first` (`layer_range.start == 0`)  → keep `embed_tokens`.
    /// - `is_last`  (`layer_range.end == num_hidden_layers`) → keep `norm` +
    ///   `lm_head` (or the tied transpose).
    ///
    /// Each owned layer is placed on its mapped device (`devices.device_for(g)`)
    /// via the `into_device` builders — the only placement cost; the forward path
    /// then performs zero intra-stage copies. Vision weights are out of scope for
    /// a pipeline stage (text backbone split only).
    #[allow(clippy::too_many_arguments)]
    pub fn stage_from_weights_with_config(
        weights: &mut HashMap<String, Tensor>,
        cfg: Qwen3_5TextConfig,
        devices: &LayerDeviceMap,
        layer_range: std::ops::Range<usize>,
        dtype: Kind,
        kv_quant_type: KVQuantType,
    ) -> Result<Self> {
        let num_global = cfg.num_hidden_layers as usize;
        if devices.len() != num_global {
            return Err(anyhow!(
                "stage_from_weights: device map covers {} layers but config has {} \
                 (num_hidden_layers)",
                devices.len(),
                num_global
            ));
        }
        if layer_range.end > num_global || layer_range.start >= layer_range.end {
            return Err(anyhow!(
                "stage_from_weights: invalid layer range {:?} for {} layers",
                layer_range,
                num_global
            ));
        }

        let is_first = layer_range.start == 0;
        let is_last = layer_range.end == num_global;
        let layer_offset = layer_range.start;
        // Stage device = device of this stage's first owned layer (its inputs and
        // KV/SSM state live here). `embed_tokens`, when present, lives on the first
        // stage's first device, which is exactly this when is_first.
        let stage_device = devices.device_for(layer_range.start);

        // --- Non-layer weights, gated by stage position (M-LOAD seam #1) ---
        let embed = if is_first {
            weights
                .remove("model.embed_tokens.weight")
                .map(|w| w.to_device(stage_device))
                .ok_or_else(|| {
                    anyhow!("stage_from_weights: first stage requires model.embed_tokens.weight")
                })?
        } else {
            // Middle/last stage owns no embedding; a placeholder is never read
            // (embed_tokens()/forward_inner from-ids paths run only on stage 0).
            Tensor::zeros([1, cfg.hidden_size as i64], (dtype, stage_device))
        };

        let (norm, lm_head_w) = if is_last {
            let norm_w = weights
                .remove("model.norm.weight")
                .map(|w| w.to_device(stage_device))
                .ok_or_else(|| anyhow!("stage_from_weights: last stage requires model.norm.weight"))?;
            let lm_head_w = weights.remove("lm_head.weight").map(|w| {
                // Cast FP8 lm_head to BF16 (applying companion scale), mirroring
                // from_weights. On the test path (plain f32) this is a no-op clone.
                match w.kind() {
                    tch::Kind::Float8e4m3fn | tch::Kind::Float8e5m2 => {
                        let w_bf = w.to_kind(tch::Kind::BFloat16);
                        if let Some(s) = weights.remove("lm_head.weight_scale_inv") {
                            let ws = w_bf.size();
                            let ss = s.size();
                            let br = ws[0] / ss[0];
                            let bc = ws[1] / ss[1];
                            let w_4d = w_bf.view([ss[0], br, ss[1], bc]);
                            let s_4d = s.to_kind(tch::Kind::BFloat16).view([ss[0], 1i64, ss[1], 1i64]);
                            (w_4d * s_4d).reshape([ws[0], ws[1]]).to_device(stage_device)
                        } else {
                            w_bf.to_device(stage_device)
                        }
                    }
                    _ => w.to_device(stage_device),
                }
            });
            (Some(Qwen3_5RMSNorm::new(norm_w, 1e-6)), lm_head_w)
        } else {
            (None, None)
        };

        // Tied lm_head: only on the last stage, and only if it also holds the
        // embedding (single-stage model). A multi-stage last shard must ship an
        // explicit lm_head since it does not own embed_tokens.
        let lm_head_transposed = if is_last && lm_head_w.is_none() {
            if !is_first {
                return Err(anyhow!(
                    "stage_from_weights: last stage requires lm_head.weight (no tied embedding present)"
                ));
            }
            let embed_bf16 = match embed.kind() {
                tch::Kind::Float8e4m3fn | tch::Kind::Float8e5m2 => embed.to_kind(tch::Kind::BFloat16),
                _ => embed.shallow_clone(),
            };
            Some(embed_bf16.tr().contiguous())
        } else {
            None
        };

        // --- Owned decoder layers, each on its mapped device (per-layer .to) ---
        let mut layers = Vec::with_capacity(layer_range.len());
        for g in layer_range.clone() {
            let target = devices.device_for(g);
            let layer = Self::build_layer(weights, &cfg, g)?;
            layers.push(layer.into_device(target));
        }

        // SSM + KV state sized to the OWNED layer count (M-LOAD seam #2); indexed
        // LOCALLY in the forward loop. Stage-local — never crosses a boundary.
        let owned = layers.len();
        let conv_states = Arc::new(parking_lot::Mutex::new(
            (0..owned).map(|_| None::<Tensor>).collect::<Vec<_>>(),
        ));
        let rec_states = Arc::new(parking_lot::Mutex::new(
            (0..owned).map(|_| None::<Tensor>).collect::<Vec<_>>(),
        ));
        let kv_cache = Some(Arc::new(parking_lot::Mutex::new(
            crate::runtime::kv_cache::KVCacheManager::new(
                owned,
                cfg.max_position_embeddings as usize,
                kv_quant_type,
            ),
        )));

        let lm_head_norm = norm.unwrap_or_else(|| {
            // Middle/first stages have no final norm. A placeholder is never
            // invoked (apply_final_norm runs only on the last stage); sized [1].
            Qwen3_5RMSNorm::new(Tensor::zeros([1], (dtype, stage_device)), cfg.rms_norm_eps)
        });

        let device_map = devices.clone();

        Ok(Self {
            config: cfg,
            device: stage_device,
            dtype,
            device_map,
            layer_offset,
            embed_tokens: embed,
            layers,
            norm: lm_head_norm,
            lm_head: lm_head_w,
            lm_head_transposed,
            conv_states,
            rec_states,
            kv_cache,
            vision_encoder: None,
            vision_projector: None,
        })
    }

    /// Unified forward inner: replaces the old `forward_inner` to avoid
    /// a deadlock that would occur if a no-delta wrapper acquired conv/rec
    /// locks and then delegated to a delta version that also tried to acquire them.
    /// (C5 deadlock fix: single implementation, delta=None for inference path)
    fn forward_inner(
        &self,
        input_ids: Option<&Tensor>,
        embeddings: Option<&Tensor>,
        start_pos: usize,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let mut hidden = if let Some(embeds) = embeddings {
            embeds.shallow_clone()
        } else {
            let ids = input_ids.ok_or_else(|| anyhow!("No input"))?;
            let flat = ids.flatten(0, -1);
            let emb = self.embed_tokens.index_select(0, &flat);
            let emb_shape = emb.size();
            let hidden_sz = emb_shape[emb_shape.len() - 1];
            let id_shape = ids.size();
            let batch = id_shape[0];
            let seq = id_shape[1];
            emb.reshape([batch, seq, hidden_sz])
        };

        let (_, seq) = (hidden.size()[0], hidden.size()[1]);

        // Position IDs for full-attention RoPE. Build on the HIDDEN-state device,
        // not `self.device` — under a pipeline split (#314) the first owned layer
        // may have been moved off `self.device`, and a mismatched position_ids
        // device would fault at SDPA. (llama already does this; see
        // forward_with_cache_inner.) On the single-device path this is identical.
        let position_ids = Tensor::arange_start(
            start_pos as i64,
            start_pos as i64 + seq,
            (Kind::Int64, hidden.device()),
        );

        let mut conv_guard = self.conv_states.lock();
        let mut rec_guard = self.rec_states.lock();

        for (idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden.shallow_clone();

            hidden = layer.input_layernorm.forward(&hidden)?;

            let delta_arg = delta.map(|d| (d, idx));

            let mixer_out = match &layer.mixer {
                LayerMixer::LinearAttn(gdn) => {
                    gdn.forward(&hidden, &mut conv_guard[idx], &mut rec_guard[idx], delta_arg)?
                }
                LayerMixer::FullAttn(attn) => {
                    if let Some(ref cache_arc) = self.kv_cache {
                        let kv = cache_arc.lock();
                        kv.with_layer_cache(idx, |lc| {
                            attn.forward(&hidden, Some(&position_ids), Some(lc), start_pos, delta_arg)
                        })
                        .ok_or_else(|| anyhow!("No KV cache for layer {idx}"))??
                    } else {
                        attn.forward(&hidden, Some(&position_ids), None, start_pos, delta_arg)?
                    }
                }
            };

            hidden = residual + &mixer_out;

            let residual2 = hidden.shallow_clone();
            hidden = layer.post_attention_layernorm.forward(&hidden)?;
            hidden = layer.mlp.forward(&hidden)?;
            hidden = residual2 + &hidden;
        }
        Ok(hidden)
    }

    /// Run global decoder layers `[range.start..range.end)` — the 2b pipeline
    /// layer-range runner (#314). See the trait docs for the stage contract.
    ///
    /// Mirrors `LlamaModel::forward_layers_inner`, plus Qwen3.5's hybrid mixers
    /// and second (SSM) state vector:
    /// - Global layer `g` is remapped to its local slot `g - layer_offset` for
    ///   `self.layers`, the KV cache, and the `conv`/`rec` SSM state (all sized to
    ///   the owned layer count).
    /// - Per-layer GDN-vs-full-attention dispatch is intrinsic to the loaded
    ///   `LayerMixer`, so no global `layer_types` lookup is needed at runtime.
    /// - The lone cross-device copy is `hidden.to_device(next)` (carrying
    ///   `position_ids` with it), inserted only when the next layer's mapped device
    ///   differs from where `hidden` currently lives. SSM `conv`/`rec` state is
    ///   stage-local and never transferred across a boundary.
    fn forward_layers_inner(
        &self,
        hidden: &Tensor,
        range: std::ops::Range<usize>,
        start_pos: usize,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let num_global = self.config.num_hidden_layers as usize;
        if range.end > num_global || range.start >= range.end {
            return Err(anyhow!(
                "forward_layers: invalid global range {range:?} for {num_global} layers"
            ));
        }
        // The owned window must contain the requested range.
        let owned = self.layer_offset..self.layer_offset + self.layers.len();
        if range.start < owned.start || range.end > owned.end {
            return Err(anyhow!(
                "forward_layers: range {range:?} not within this stage's owned layers {owned:?}"
            ));
        }

        let mut hidden = hidden.shallow_clone();

        // position_ids recomputed from start_pos + seq (never carried across a
        // boundary as state). Built on the *input* device (the first owned
        // layer's), then moved alongside `hidden` at a device boundary so RoPE /
        // SDPA always see a matching device.
        let seq = hidden.size()[1];
        let mut position_ids = Tensor::arange_start(
            start_pos as i64,
            start_pos as i64 + seq,
            (Kind::Int64, hidden.device()),
        );

        let mut conv_guard = self.conv_states.lock();
        let mut rec_guard = self.rec_states.lock();

        for g in range {
            let local_idx = g - self.layer_offset;

            // The single cross-device transfer: only when this layer's device
            // differs from where `hidden` currently is (zero-copy otherwise).
            let target = self.device_map.device_for(g);
            if hidden.device() != target {
                hidden = hidden.to_device(target);
                position_ids = position_ids.to_device(target);
            }

            let layer = &self.layers[local_idx];
            let residual = hidden.shallow_clone();
            hidden = layer.input_layernorm.forward(&hidden)?;

            // Delta module lookup uses the GLOBAL layer index (per-layer LoRA is
            // keyed by global layer, matching whole-model inference).
            let delta_arg = delta.map(|d| (d, g));

            let mixer_out = match &layer.mixer {
                LayerMixer::LinearAttn(gdn) => {
                    // SSM conv/rec state indexed LOCALLY; stage-local, not moved.
                    gdn.forward(
                        &hidden,
                        &mut conv_guard[local_idx],
                        &mut rec_guard[local_idx],
                        delta_arg,
                    )?
                }
                LayerMixer::FullAttn(attn) => {
                    if let Some(ref cache_arc) = self.kv_cache {
                        let kv = cache_arc.lock();
                        kv.with_layer_cache(local_idx, |lc| {
                            attn.forward(&hidden, Some(&position_ids), Some(lc), start_pos, delta_arg)
                        })
                        .ok_or_else(|| anyhow!("No KV cache for local layer {local_idx}"))??
                    } else {
                        attn.forward(&hidden, Some(&position_ids), None, start_pos, delta_arg)?
                    }
                }
            };

            hidden = residual + &mixer_out;

            let residual2 = hidden.shallow_clone();
            hidden = layer.post_attention_layernorm.forward(&hidden)?;
            hidden = layer.mlp.forward(&hidden)?;
            hidden = residual2 + &hidden;
        }
        Ok(hidden)
    }

    /// Training-path sibling of [`Self::forward_layers_inner`] — the cross-device
    /// autograd primitive for TTT-on-split (#316). See the trait
    /// `forward_layers_train` docs for the contract.
    ///
    /// Differs from the inference runner in three deliberate ways:
    /// - **no KV cache** — full-attention layers run with full causal attention
    ///   over the entire context (`start_pos = 0`, no cache);
    /// - **`start_pos = 0`** — `position_ids = 0..seq`;
    /// - **fresh, call-local recurrent (SSM) state** — `conv`/`rec` start at
    ///   `None` for every owned layer and are never written back to
    ///   `self.conv_states` / `self.rec_states`. This keeps a TTT step from
    ///   polluting the persistent inference recurrent state, and makes the split
    ///   numerically identical to the whole-model training forward (per-layer
    ///   recurrent state never crosses a layer boundary, so partitioning the layer
    ///   range is exact).
    ///
    /// The lone cross-device `hidden.to_device(next)` (carrying `position_ids`) is
    /// autograd-transparent, so `loss.backward()` materializes grads on each
    /// parameter's own device.
    fn forward_layers_train_inner(
        &self,
        hidden: &Tensor,
        range: std::ops::Range<usize>,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let num_global = self.config.num_hidden_layers as usize;
        if range.end > num_global || range.start >= range.end {
            return Err(anyhow!(
                "forward_layers_train: invalid global range {range:?} for {num_global} layers"
            ));
        }
        let owned = self.layer_offset..self.layer_offset + self.layers.len();
        if range.start < owned.start || range.end > owned.end {
            return Err(anyhow!(
                "forward_layers_train: range {range:?} not within this stage's owned layers {owned:?}"
            ));
        }

        let mut hidden = hidden.shallow_clone();

        // Training path: start_pos is always 0 → position_ids = 0..seq, built on
        // the input device and moved alongside `hidden` at each device boundary.
        let seq = hidden.size()[1];
        let mut position_ids =
            Tensor::arange(seq, (Kind::Int64, hidden.device()));

        // Fresh, call-local SSM state — never touches the persistent inference
        // `conv_states`/`rec_states`. Sized to the OWNED layer count and indexed
        // locally, mirroring the persistent vectors' sizing.
        let mut conv_local: Vec<Option<Tensor>> =
            (0..self.layers.len()).map(|_| None).collect();
        let mut rec_local: Vec<Option<Tensor>> =
            (0..self.layers.len()).map(|_| None).collect();

        for g in range {
            let local_idx = g - self.layer_offset;

            // The single cross-device transfer (autograd-transparent): only when
            // this layer's mapped device differs from where `hidden` currently is.
            let target = self.device_map.device_for(g);
            if hidden.device() != target {
                hidden = hidden.to_device(target);
                position_ids = position_ids.to_device(target);
            }

            let layer = &self.layers[local_idx];
            let residual = hidden.shallow_clone();
            hidden = layer.input_layernorm.forward(&hidden)?;

            // Delta module lookup uses the GLOBAL layer index (per-layer LoRA is
            // keyed by global layer, matching whole-model training).
            let delta_arg = delta.map(|d| (d, g));

            let mixer_out = match &layer.mixer {
                LayerMixer::LinearAttn(gdn) => gdn.forward(
                    &hidden,
                    &mut conv_local[local_idx],
                    &mut rec_local[local_idx],
                    delta_arg,
                )?,
                LayerMixer::FullAttn(attn) => {
                    // No KV cache (training): full causal attention, start_pos = 0.
                    attn.forward(&hidden, Some(&position_ids), None, 0, delta_arg)?
                }
            };

            hidden = residual + &mixer_out;

            let residual2 = hidden.shallow_clone();
            hidden = layer.post_attention_layernorm.forward(&hidden)?;
            hidden = layer.mlp.forward(&hidden)?;
            hidden = residual2 + &hidden;
        }
        Ok(hidden)
    }

    /// Encode images through the vision encoder and project to text hidden space.
    ///
    /// pixel_values: [B, C, H, W]
    /// Returns: [total_image_patches, hidden_size]
    fn encode_vision_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let enc = self.vision_encoder.as_ref()
            .ok_or_else(|| anyhow!("No vision encoder loaded"))?;
        let proj = self.vision_projector.as_ref()
            .ok_or_else(|| anyhow!("No vision projector loaded"))?;

        // [B * out_patches, out_hidden_size]
        let features = enc.forward(pixel_values)?;
        // Project to text hidden size: [B * out_patches, hidden_size]
        Ok(proj.apply(&features))
    }

    /// Build combined embeddings by replacing image placeholder tokens with vision features.
    ///
    /// image_token_id identifies placeholder positions in input_ids.
    /// pixel_values provides image patches for each placeholder sequence.
    fn prepare_inputs_embeds(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        images_seq_mask: Option<&Tensor>,
        _images_emb_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Get text embeddings for all tokens
        let flat = input_ids.flatten(0, -1);
        let text_emb = self.embed_tokens.index_select(0, &flat);
        let ids_shape = input_ids.size();
        let hidden_sz = text_emb.size().last().copied().unwrap_or(self.config.hidden_size as i64);
        let combined = text_emb.reshape([ids_shape[0], ids_shape[1], hidden_sz]);

        // If we have pixel_values and a sequence mask, inject vision embeddings
        if let (Some(pv), Some(mask)) = (pixel_values, images_seq_mask) {
            let vision_feats = self.encode_vision_features(pv)?; // [N_img, hidden]
            // mask: [batch, seq] bool tensor marking where to inject
            // Flatten and scatter vision features into combined at masked positions
            let (batch, seq, hs) = dims3(&combined)?;
            for b in 0..batch {
                let mask_b = mask.select(0, b).to_kind(Kind::Bool); // [seq]
                let positions: Vec<i64> = (0..seq)
                    .filter(|&t| mask_b.double_value(&[t]) != 0.0)
                    .collect();
                if positions.is_empty() { continue; }
                // Slice vision_feats to match the number of masked positions
                let n_img = positions.len() as i64;
                let vf_slice = vision_feats.narrow(0, 0, n_img.min(vision_feats.size()[0]));
                for (img_idx, &pos) in positions.iter().enumerate() {
                    if (img_idx as i64) < vf_slice.size()[0] {
                        let vf = vf_slice.select(0, img_idx as i64); // [hidden]
                        combined.select(0, b).select(0, pos).copy_(&vf.to_kind(self.dtype));
                    }
                }
            }
        }

        Ok(combined)
    }

    /// Expose the Qwen3.5 text config for external callers (e.g., get_per_layer_lora_dims).
    pub fn text_config(&self) -> &Qwen3_5TextConfig {
        &self.config
    }

    /// Deep-copy current SSM states for TTT snapshot/restore (C4 fix: must use .copy(), not .shallow_clone()).
    ///
    /// Returns (conv_snapshot, rec_snapshot). Each tensor is deep-copied so that
    /// subsequent SSM mutations during TTT adaptation do not corrupt the snapshot.
    pub fn snapshot_ssm_states(&self) -> (Vec<Option<Tensor>>, Vec<Option<Tensor>>) {
        let conv_snapshot = {
            let guard = self.conv_states.lock();
            guard.iter().map(|opt| opt.as_ref().map(|t| t.copy())).collect()
        };
        let rec_snapshot = {
            let guard = self.rec_states.lock();
            guard.iter().map(|opt| opt.as_ref().map(|t| t.copy())).collect()
        };
        (conv_snapshot, rec_snapshot)
    }

    /// Restore SSM states from a snapshot produced by `snapshot_ssm_states`.
    pub fn restore_ssm_states(
        &self,
        conv_snapshot: Vec<Option<Tensor>>,
        rec_snapshot: Vec<Option<Tensor>>,
    ) {
        *self.conv_states.lock() = conv_snapshot;
        *self.rec_states.lock() = rec_snapshot;
    }

    /// Shared implementation for `decode_layer` and `decode_layer_with_delta`.
    ///
    /// **Training path only.** Full-attention layers run with `start_pos=0` and no KV
    /// cache, which is correct for the TTT gradient loop (full causal mask over the entire
    /// context). Do not use this method for incremental autoregressive inference — use
    /// `forward_with_cache_and_delta` / `forward_with_cache` instead.
    ///
    /// GDN layers acquire `conv_states` / `rec_states` locks per call (not held across
    /// layers), which is safe because `forward_with_delta` in TorchEngine never holds
    /// those locks itself.
    fn decode_layer_impl(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        position_ids: Option<&Tensor>,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let layer = self.layers.get(layer_idx)
            .ok_or_else(|| anyhow!("Layer {layer_idx} out of range"))?;
        let residual = hidden_states.shallow_clone();
        let normed = layer.input_layernorm.forward(hidden_states)?;

        let delta_arg = delta.map(|d| (d, layer_idx));

        let mixer_out = match &layer.mixer {
            LayerMixer::LinearAttn(gdn) => {
                let mut conv = self.conv_states.lock();
                let mut rec = self.rec_states.lock();
                gdn.forward(&normed, &mut conv[layer_idx], &mut rec[layer_idx], delta_arg)?
            }
            LayerMixer::FullAttn(attn) => {
                attn.forward(&normed, position_ids, None, 0, delta_arg)?
            }
        };

        let hidden = &residual + &mixer_out;
        let residual2 = hidden.shallow_clone();
        let normed2 = layer.post_attention_layernorm.forward(&hidden)?;
        let ffn_out = layer.mlp.forward(&normed2)?;
        Ok((&residual2 + &ffn_out, None))
    }

    fn lm_head_apply(&self, hidden: &Tensor) -> Result<Tensor> {
        let (batch, seq, hs) = dims3(hidden)?;
        let flat = hidden.reshape([batch * seq, hs]);
        let logits = if let Some(ref lm_head) = self.lm_head {
            // lm_head stored as [vocab, hidden], need to transpose for matmul
            flat.matmul(&lm_head.tr())
        } else if let Some(ref lm_t) = self.lm_head_transposed {
            flat.matmul(lm_t)
        } else {
            return Err(anyhow!("No lm_head weight"));
        };

        Ok(logits.reshape([batch, seq, -1]))
    }
}

// ============================================================================
// ModelOperations impl
// ============================================================================

impl ModelOperations for Qwen3_5Model {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Qwen3_5
    }

    fn config(&self) -> &dyn ArchitectureConfig {
        &self.config
    }

    fn forward(&self, input: &Tensor, _past_kv: Option<&Tensor>) -> Result<Tensor> {
        let hidden = self.forward_inner(Some(input), None, 0, None)?;
        let normed = self.norm.forward(&hidden)?;
        self.lm_head_apply(&normed)
    }

    fn forward_with_cache(&self, input: &Tensor, start_pos: usize) -> Result<Tensor> {
        let hidden = self.forward_inner(Some(input), None, start_pos, None)?;
        let seq_len = hidden.size()[1];
        anyhow::ensure!(seq_len > 0, "forward_with_cache: empty token sequence (seq_len=0)");
        // Prefill optimization (#201): only run the expensive LM-head matmul over the
        // last position — the caller always discards all but the last token's logits.
        let last = hidden.narrow(1, seq_len - 1, 1);
        let normed = self.norm.forward(&last)?;
        self.lm_head_apply(&normed)
    }

    fn forward_from_embeddings(&self, embeddings: &Tensor, start_pos: usize) -> Result<Tensor> {
        let hidden = self.forward_inner(None, Some(embeddings), start_pos, None)?;
        let seq_len = hidden.size()[1];
        anyhow::ensure!(seq_len > 0, "forward_from_embeddings: empty embedding sequence (seq_len=0)");
        let last = hidden.narrow(1, seq_len - 1, 1);
        let normed = self.norm.forward(&last)?;
        self.lm_head_apply(&normed)
    }

    fn forward_with_cache_and_delta(
        &self,
        input: &Tensor,
        start_pos: usize,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let hidden = self.forward_inner(Some(input), None, start_pos, delta)?;
        let seq_len = hidden.size()[1];
        anyhow::ensure!(seq_len > 0, "forward_with_cache_and_delta: empty token sequence (seq_len=0)");
        let last = hidden.narrow(1, seq_len - 1, 1);
        let normed = self.norm.forward(&last)?;
        self.lm_head_apply(&normed)
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let flat = input_ids.flatten(0, -1);
        let emb = self.embed_tokens.index_select(0, &flat);
        let emb_shape = emb.size();
        let hidden_sz = emb_shape[emb_shape.len() - 1];
        let ids_shape = input_ids.size();
        Ok(emb.reshape([ids_shape[0], ids_shape[1], hidden_sz]))
    }

    fn apply_final_norm(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.norm.forward(hidden_states)
    }

    fn lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.lm_head_apply(hidden_states)
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers as usize
    }

    fn forward_layers(
        &self,
        hidden: &Tensor,
        range: std::ops::Range<usize>,
        start_pos: usize,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        self.forward_layers_inner(hidden, range, start_pos, delta)
    }

    fn forward_layers_train(
        &self,
        hidden: &Tensor,
        range: std::ops::Range<usize>,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        self.forward_layers_train_inner(hidden, range, delta)
    }

    fn decode_layer(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        _attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        _past_kv: Option<&crate::runtime::kv_cache::LayerKVCache>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        self.decode_layer_impl(layer_idx, hidden_states, position_ids, None)
    }

    fn decode_layer_with_delta(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        _attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        _past_kv: Option<&crate::runtime::kv_cache::LayerKVCache>,
        delta: &crate::training::TenantDelta,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        self.decode_layer_impl(layer_idx, hidden_states, position_ids, Some(delta))
    }

    fn clear_kv_cache(&self) {
        let mut conv = self.conv_states.lock();
        let mut rec = self.rec_states.lock();
        for s in conv.iter_mut() { *s = None; }
        for s in rec.iter_mut() { *s = None; }
        if let Some(ref cache) = self.kv_cache {
            cache.lock().clear_all();
        }
    }

    fn set_kv_cache(&mut self, cache: Arc<parking_lot::Mutex<KVCacheManager>>) {
        self.kv_cache = Some(cache);
    }

    fn get_kv_cache(&self) -> Option<Arc<parking_lot::Mutex<KVCacheManager>>> {
        self.kv_cache.clone()
    }

    fn take_kv_cache(&mut self) -> Option<Arc<parking_lot::Mutex<KVCacheManager>>> {
        self.kv_cache.take()
    }

    fn reshape_for_attention(&self, tensor: &Tensor, _is_key_value: bool) -> Result<Tensor> {
        Ok(tensor.shallow_clone())
    }

    fn apply_rope(&self, tensor: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let rd = self.config.rotary_dim as i64;
        let device = tensor.device();
        let dtype = tensor.kind();
        let mut rope = RoPE::new_with_dtype(rd, self.config.rope_theta as f64, self.config.max_position_embeddings as i64, device, dtype)?;
        rope.forward(tensor, Some(position_ids))
    }

    fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        self.norm.forward(tensor)
    }

    fn get_attention_mask(&self, seq_len: usize, _past_kv_len: usize) -> Result<Tensor> {
        let sq = seq_len as i64;
        Ok(Tensor::ones([sq, sq], (Kind::Float, self.device)).tril(0))
    }

    fn is_multimodal(&self) -> bool {
        self.vision_encoder.is_some()
    }

    fn prepare_multimodal_inputs(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        images_seq_mask: Option<&Tensor>,
        images_emb_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.prepare_inputs_embeds(input_ids, pixel_values, images_seq_mask, images_emb_mask)
    }

    fn encode_vision(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.encode_vision_features(pixel_values)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod pipeline_tests {
    //! CPU equivalence tests for the 2b pipeline layer-split (#314).
    //!
    //! These prove that running qwen3_5's hybrid (GatedDeltaNet + full-attention)
    //! decoder stack via `forward_layers` — whole-range AND split into two stages
    //! over an all-CPU [`LayerDeviceMap`] — reproduces the whole-model forward to
    //! float-reassociation tolerance. The split must hold even though each GDN
    //! layer carries its own stage-local SSM (`conv`/`rec`) state: if the SSM
    //! state threading were wrong, the two-stage result would diverge.
    use super::*;
    use crate::runtime::device_pool::LayerDeviceMap;

    const HIDDEN: i64 = 16;
    const HEADS: i64 = 2;
    const KV_HEADS: i64 = 2;
    const HEAD_DIM: i64 = 8;
    const INTER: i64 = 32;
    const VOCAB: i64 = 48;
    const LAYERS: usize = 4;
    // GatedDeltaNet dims (kept tiny + symmetric: num_k_heads == num_v_heads).
    const LIN_K_HEADS: usize = 1;
    const LIN_V_HEADS: usize = 1;
    const LIN_K_DIM: usize = 4;
    const LIN_V_DIM: usize = 4;
    const CONV_KERNEL: usize = 4;

    fn tiny_config() -> Qwen3_5TextConfig {
        // Default hybrid pattern: layer (i+1)%4==0 is full attention, rest GDN.
        // With LAYERS=4 → [GDN, GDN, GDN, FullAttn]; a split at 2 puts the full-
        // attention layer in the second stage, exercising both mixer kinds across
        // the stage boundary.
        let layer_types = (0..LAYERS)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "full_attention".to_owned()
                } else {
                    "linear_attention".to_owned()
                }
            })
            .collect();
        let rotary_dim = ((HEAD_DIM as f32) * 0.25) as usize; // 2
        Qwen3_5TextConfig {
            hidden_size: HIDDEN as u32,
            num_hidden_layers: LAYERS as u32,
            num_attention_heads: HEADS as u32,
            num_key_value_heads: KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            intermediate_size: INTER as u32,
            vocab_size: VOCAB as u32,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.25,
            rotary_dim,
            layer_types,
            linear_conv_kernel_dim: CONV_KERNEL,
            linear_key_head_dim: LIN_K_DIM,
            linear_value_head_dim: LIN_V_DIM,
            linear_num_key_heads: LIN_K_HEADS,
            linear_num_value_heads: LIN_V_HEADS,
            is_moe: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            has_vision: false,
            vision_out_hidden_size: 0,
        }
    }

    /// Deterministic, RNG-free weights on CPU/f32. tch's global RNG is shared
    /// across parallel tests, so a `randn`-based build could interleave and
    /// diverge between the whole-model and staged paths; a fixed sin-of-index
    /// pattern is byte-identical regardless of scheduling. HF stores projections
    /// as `[out, in]` (LinearProjection::take transposes to `[in, out]`).
    fn tiny_weights() -> HashMap<String, Tensor> {
        let opt = (Kind::Float, Device::Cpu);
        let mut w = HashMap::new();
        // Bounded small values in roughly [-0.05, 0.05], offset per-tensor so
        // different weights are not identical.
        let mut seed: i64 = 0;
        let mut pat = |dims: &[i64]| -> Tensor {
            let n: i64 = dims.iter().product();
            seed += 7;
            (Tensor::arange(n, opt) * 0.017 + seed as f64 * 0.013)
                .sin()
                .reshape(dims)
                * 0.05
        };

        w.insert("model.embed_tokens.weight".to_owned(), pat(&[VOCAB, HIDDEN]));
        w.insert("model.norm.weight".to_owned(), pat(&[HIDDEN]));
        w.insert("lm_head.weight".to_owned(), pat(&[VOCAB, HIDDEN]));

        let key_dim = (LIN_K_HEADS * LIN_K_DIM) as i64;
        let val_dim = (LIN_V_HEADS * LIN_V_DIM) as i64;
        let conv_dim = key_dim * 2 + val_dim;
        let nv = LIN_V_HEADS as i64;

        for i in 0..LAYERS {
            let p = format!("model.layers.{i}");
            w.insert(format!("{p}.input_layernorm.weight"), pat(&[HIDDEN]));
            w.insert(format!("{p}.post_attention_layernorm.weight"), pat(&[HIDDEN]));

            let is_full = (i + 1) % 4 == 0;
            if is_full {
                // Full attention: q_proj out = nh*hd*2 (q + gate), k/v out = nkv*hd.
                let ap = format!("{p}.self_attn");
                w.insert(format!("{ap}.q_proj.weight"), pat(&[HEADS * HEAD_DIM * 2, HIDDEN]));
                w.insert(format!("{ap}.k_proj.weight"), pat(&[KV_HEADS * HEAD_DIM, HIDDEN]));
                w.insert(format!("{ap}.v_proj.weight"), pat(&[KV_HEADS * HEAD_DIM, HIDDEN]));
                w.insert(format!("{ap}.o_proj.weight"), pat(&[HIDDEN, HEADS * HEAD_DIM]));
                w.insert(format!("{ap}.q_norm.weight"), pat(&[HEAD_DIM]));
                w.insert(format!("{ap}.k_norm.weight"), pat(&[HEAD_DIM]));
            } else {
                // GatedDeltaNet: 4 input projections fused at load.
                let lp = format!("{p}.linear_attn");
                w.insert(format!("{lp}.in_proj_qkv.weight"), pat(&[conv_dim, HIDDEN]));
                w.insert(format!("{lp}.in_proj_z.weight"), pat(&[val_dim, HIDDEN]));
                w.insert(format!("{lp}.in_proj_b.weight"), pat(&[nv, HIDDEN]));
                w.insert(format!("{lp}.in_proj_a.weight"), pat(&[nv, HIDDEN]));
                // conv1d_weight stays raw [conv_dim, 1, kernel_size] (not transposed).
                w.insert(format!("{lp}.conv1d.weight"), pat(&[conv_dim, 1, CONV_KERNEL as i64]));
                w.insert(format!("{lp}.A_log"), pat(&[nv]));
                w.insert(format!("{lp}.dt_bias"), pat(&[nv]));
                w.insert(format!("{lp}.norm.weight"), pat(&[LIN_V_DIM as i64]));
                w.insert(format!("{lp}.out_proj.weight"), pat(&[HIDDEN, val_dim]));
            }

            // Dense MLP.
            let mp = format!("{p}.mlp");
            w.insert(format!("{mp}.gate_proj.weight"), pat(&[INTER, HIDDEN]));
            w.insert(format!("{mp}.up_proj.weight"), pat(&[INTER, HIDDEN]));
            w.insert(format!("{mp}.down_proj.weight"), pat(&[HIDDEN, INTER]));
        }
        w
    }

    fn whole_model() -> Qwen3_5Model {
        let mut w = tiny_weights();
        Qwen3_5Model::from_weights(
            &mut w, tiny_config(), None, &Device::Cpu, Kind::Float, KVQuantType::None,
        )
        .unwrap()
    }

    /// Drive the pipeline path on one model the way an engine would:
    /// embed → forward_layers(0..N) → final norm → lm_head.
    fn orchestrated_logits(m: &Qwen3_5Model, input: &Tensor) -> Tensor {
        let emb = m.embed_tokens(input).unwrap();
        let h = m.forward_layers(&emb, 0..m.num_layers(), 0, None).unwrap();
        let h = m.apply_final_norm(&h).unwrap();
        m.lm_head(&h).unwrap()
    }

    #[test]
    fn forward_layers_full_range_matches_whole_model_forward() {
        // An all-CPU LayerDeviceMap means forward_layers(0..N) must equal the
        // whole-model single-device forward (the byte-identity equivalence test).
        let input = Tensor::from_slice(&[1i64, 5, 9, 2]).reshape([1, 4]);

        // Reference: whole-model forward over the full sequence (NOT
        // forward_with_cache, which only returns the last position's logits).
        let reference = whole_model();
        let ref_logits = reference.forward(&input, None).unwrap();

        // Pipeline path on a fresh model (independent SSM/KV state), full range
        // over the implicit single-device map.
        let piped = whole_model();
        let pipe_logits = orchestrated_logits(&piped, &input);

        let max_diff = (&ref_logits - &pipe_logits).abs().max().double_value(&[]);
        assert!(
            ref_logits.allclose(&pipe_logits, 1e-4, 1e-4, false),
            "forward_layers(0..N) over a single-device map must equal whole-model forward \
             (max_diff={max_diff}, shape={:?})",
            ref_logits.size()
        );
        assert!(piped.device_map.is_single_device());
        assert_eq!(piped.layer_offset, 0);
    }

    #[test]
    fn staged_split_equals_whole_model() {
        // Build TWO stages over an all-CPU 2-way split and run them as a pipeline:
        // stage0: embed → forward_layers(0..k); stage1: forward_layers(k..N) →
        // norm → lm_head. The split point (k=2) puts the full-attention layer in
        // stage1, so this exercises the global↔local remap, is_first/is_last
        // gating, per-global-index hybrid layer selection, and — critically —
        // that each stage's GDN SSM state is sized to its OWNED layer count and
        // indexed locally. Result must equal the whole-model forward.
        let input = Tensor::from_slice(&[3i64, 7, 1, 4]).reshape([1, 4]);

        let reference = whole_model();
        let ref_logits = reference.forward(&input, None).unwrap();

        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let split = LAYERS / 2;

        let mut w0 = tiny_weights();
        let stage0 = Qwen3_5Model::stage_from_weights_with_config(
            &mut w0, tiny_config(), &map, 0..split, Kind::Float, KVQuantType::None,
        )
        .unwrap();
        assert_eq!(stage0.layer_offset, 0);
        assert_eq!(stage0.layers.len(), split, "stage0 owns exactly its layers");
        assert!(stage0.lm_head.is_none(), "first (non-last) stage has no head");
        // SSM state sized to OWNED count, not global.
        assert_eq!(stage0.conv_states.lock().len(), split);
        assert_eq!(stage0.rec_states.lock().len(), split);

        let mut w1 = tiny_weights();
        let stage1 = Qwen3_5Model::stage_from_weights_with_config(
            &mut w1, tiny_config(), &map, split..LAYERS, Kind::Float, KVQuantType::None,
        )
        .unwrap();
        assert_eq!(stage1.layer_offset, split);
        assert_eq!(stage1.layers.len(), LAYERS - split);
        assert_eq!(stage1.conv_states.lock().len(), LAYERS - split);

        // Drive the pipeline. Only `hidden` crosses the (same-device) boundary;
        // each stage's SSM state stays local.
        let emb = stage0.embed_tokens(&input).unwrap();
        let h0 = stage0.forward_layers(&emb, 0..split, 0, None).unwrap();
        let h1 = stage1.forward_layers(&h0, split..LAYERS, 0, None).unwrap();
        let h1 = stage1.apply_final_norm(&h1).unwrap();
        let logits = stage1.lm_head(&h1).unwrap();

        let max_diff = (&ref_logits - &logits).abs().max().double_value(&[]);
        assert!(
            ref_logits.allclose(&logits, 1e-4, 1e-4, false),
            "two-stage split (same device) must equal whole-model forward \
             (max_diff={max_diff}); a mismatch here means the stage-local SSM state \
             threading is wrong"
        );
    }

    #[test]
    fn forward_layers_rejects_out_of_window_range() {
        // A stage that owns [2..4) must reject a range outside its window.
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let mut w = tiny_weights();
        let stage = Qwen3_5Model::stage_from_weights_with_config(
            &mut w, tiny_config(), &map, 2..LAYERS, Kind::Float, KVQuantType::None,
        )
        .unwrap();
        let emb = Tensor::randn([1, 3, HIDDEN], (Kind::Float, Device::Cpu));
        assert!(stage.forward_layers(&emb, 0..2, 0, None).is_err(), "range below window");
        assert!(stage.forward_layers(&emb, 2..LAYERS, 0, None).is_ok(), "owned range ok");
    }

    #[test]
    fn stage_loader_rejects_missing_required_weights() {
        // Last stage with no norm weight must error (M-LOAD seam #1 gating).
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let mut w = tiny_weights();
        w.remove("model.norm.weight");
        let result = Qwen3_5Model::stage_from_weights_with_config(
            &mut w, tiny_config(), &map, 0..LAYERS, Kind::Float, KVQuantType::None,
        );
        match result {
            Ok(_) => panic!("last stage without norm weight must error"),
            Err(e) => assert!(e.to_string().contains("norm"), "got: {e}"),
        }
    }

    // ========================================================================
    // #316 — TTT-on-split cross-device autograd equivalence (CPU-verifiable),
    // exercising the hybrid GDN + full-attention stack across a stage boundary.
    // ========================================================================

    use crate::training::tenant_delta::{TenantDelta, TenantDeltaConfig};

    /// Per-layer o_proj LoRA delta on CPU. o_proj is the one module BOTH mixer
    /// kinds inject (GDN out_proj: in=val_dim; full-attn: in=nh*hd), so its dims
    /// differ per layer type — supplied via per_layer_dims (mirrors
    /// `TorchEngine::get_per_layer_lora_dims`). Seeds deterministic non-zero A and
    /// B so gradients w.r.t. both are non-trivial.
    fn tiny_delta() -> TenantDelta {
        let hidden = HIDDEN as usize;
        let gdn_in = LIN_V_HEADS * LIN_V_DIM; // out_proj input dim for GDN layers
        let full_in = (HEADS * HEAD_DIM) as usize; // o_proj input dim for full-attn layers
        let mut per_layer: std::collections::HashMap<usize, std::collections::HashMap<String, (usize, usize)>> =
            std::collections::HashMap::new();
        for i in 0..LAYERS {
            let is_full = (i + 1) % 4 == 0;
            let mut m = std::collections::HashMap::new();
            let in_dim = if is_full { full_in } else { gdn_in };
            m.insert("o_proj".to_owned(), (in_dim, hidden));
            per_layer.insert(i, m);
        }
        let cfg = TenantDeltaConfig {
            rank: 2,
            alpha: 2.0,
            target_modules: vec!["o_proj".to_owned()],
            ..Default::default()
        };
        let flat: std::collections::HashMap<String, (usize, usize)> = std::collections::HashMap::new();
        let delta = TenantDelta::new_with_per_layer_dims(
            &cfg, &flat, Device::Cpu, LAYERS, Some(&per_layer),
        ).unwrap();

        // Seed by a STABLE per-key offset (HashMap iteration order is not
        // deterministic, so independent builds must agree key-for-key).
        let key_offset = |key: &str| -> i64 {
            key.bytes().map(|b| b as i64).sum::<i64>() % 17
        };
        let _g = tch::no_grad_guard();
        for (k, a) in delta.lora_a.iter() {
            let n: i64 = a.size().iter().product();
            let vals = (Tensor::arange(n, (Kind::Float, Device::Cpu)) + key_offset(k)).sin() * 0.1;
            // copy_ needs &mut; shallow_clone shares storage so the delta param is mutated.
            a.shallow_clone().copy_(&vals.reshape(a.size()));
        }
        for (k, b) in delta.lora_b.iter() {
            let n: i64 = b.size().iter().product();
            let vals = (Tensor::arange(n, (Kind::Float, Device::Cpu)) + key_offset(k) + 7).cos() * 0.1;
            b.shallow_clone().copy_(&vals.reshape(b.size()));
        }
        delta
    }

    fn grad_snapshot(delta: &TenantDelta) -> std::collections::HashMap<String, f64> {
        let mut out = std::collections::HashMap::new();
        for (k, a) in &delta.lora_a {
            assert!(a.grad().defined(), "A grad undefined for {k}");
            out.insert(format!("A:{k}"), a.grad().norm().double_value(&[]));
        }
        for (k, b) in &delta.lora_b {
            assert!(b.grad().defined(), "B grad undefined for {k}");
            out.insert(format!("B:{k}"), b.grad().norm().double_value(&[]));
        }
        out
    }

    /// The #316 correctness guardrail for the hybrid architecture: a TTT/training
    /// step (forward_layers_train → NTP loss → backward) over a two-stage CPU split
    /// must produce the SAME delta gradients as the whole-model training forward,
    /// within ~1e-4. The split point (k=2) keeps the full-attention layer in stage1
    /// and GDN layers in stage0, so this proves cross-device autograd is correct
    /// across both mixer kinds AND the (fresh, stage-local) SSM state threading.
    #[test]
    fn ttt_split_autograd_matches_whole_model_grads() {
        use crate::training::pipeline::{compute_ntp_loss_split, TrainStage};

        let input = Tensor::from_slice(&[3i64, 7, 1, 4, 9, 2]).reshape([1, 6]);

        // (a) whole-model training forward over a single-device map.
        let whole = whole_model();
        let whole_delta = tiny_delta();
        let whole_stage = [TrainStage { model: &whole, range: 0..LAYERS }];
        let loss_whole = compute_ntp_loss_split(&whole_stage, &input, Some(&whole_delta)).unwrap();
        loss_whole.backward();
        let whole_grads = grad_snapshot(&whole_delta);
        whole_delta.zero_grad();

        // (b) two-stage split over an all-CPU map.
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let split = LAYERS / 2;
        let mut w0 = tiny_weights();
        let stage0 = Qwen3_5Model::stage_from_weights_with_config(
            &mut w0, tiny_config(), &map, 0..split, Kind::Float, KVQuantType::None,
        ).unwrap();
        let mut w1 = tiny_weights();
        let stage1 = Qwen3_5Model::stage_from_weights_with_config(
            &mut w1, tiny_config(), &map, split..LAYERS, Kind::Float, KVQuantType::None,
        ).unwrap();
        let split_delta = tiny_delta();
        let stages = [
            TrainStage { model: &stage0, range: 0..split },
            TrainStage { model: &stage1, range: split..LAYERS },
        ];
        let loss_split = compute_ntp_loss_split(&stages, &input, Some(&split_delta)).unwrap();
        loss_split.backward();
        let split_grads = grad_snapshot(&split_delta);

        let lw = loss_whole.double_value(&[]);
        let ls = loss_split.double_value(&[]);
        assert!((lw - ls).abs() < 1e-4, "loss diverged: whole={lw} split={ls}");

        assert_eq!(whole_grads.len(), split_grads.len());
        for (k, &gw) in &whole_grads {
            let gs = *split_grads.get(k).unwrap_or_else(|| panic!("missing grad {k}"));
            assert!(
                (gw - gs).abs() <= 1e-4 + 1e-4 * gw.abs(),
                "grad norm diverged for {k}: whole={gw} split={gs}"
            );
            assert!(gw > 0.0, "grad for {k} is zero — autograd path not exercised");
        }
    }

    #[test]
    fn forward_layers_train_rejects_out_of_window_range() {
        let map = LayerDeviceMap::single(Device::Cpu, LAYERS).unwrap();
        let mut w = tiny_weights();
        let stage = Qwen3_5Model::stage_from_weights_with_config(
            &mut w, tiny_config(), &map, 2..LAYERS, Kind::Float, KVQuantType::None,
        ).unwrap();
        let emb = Tensor::randn([1, 3, HIDDEN], (Kind::Float, Device::Cpu));
        assert!(stage.forward_layers_train(&emb, 0..2, None).is_err(), "range below window");
        assert!(stage.forward_layers_train(&emb, 2..LAYERS, None).is_ok(), "owned range ok");
    }
}
