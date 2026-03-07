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
use crate::runtime::kv_cache::KVCacheManager;
use crate::runtime::kv_quant::KVQuantType;
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
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
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
    fn num_attention_heads(&self) -> usize { self.num_attention_heads }
    fn num_key_value_heads(&self) -> usize { self.num_key_value_heads }
    fn hidden_size(&self) -> usize { self.hidden_size }
    fn intermediate_size(&self) -> usize { self.intermediate_size }
    fn head_dim(&self) -> usize { self.head_dim }
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn max_position_embeddings(&self) -> usize { self.max_position_embeddings }
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
            hidden_size: cfg.hidden_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            max_position_embeddings: max_pos,
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
                    let s_exp = s.to_kind(tch::Kind::BFloat16)
                        .view([ss[0], 1, ss[1], 1])
                        .expand([ss[0], br, ss[1], bc], false)
                        .reshape([ws[0], ws[1]]);
                    w_bf * s_exp
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
                let s_exp = s
                    .to_kind(tch::Kind::BFloat16)
                    .view([ss[0], ss[1], 1, ss[2], 1])
                    .expand([ss[0], ss[1], br, ss[2], bc], false)
                    .reshape([ws[0], ws[1], ws[2]]);
                &w_bf * &s_exp
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
}

unsafe impl Send for RMSNormGated {}
unsafe impl Sync for RMSNormGated {}

/// L2 normalization along last dim (used on Q and K before delta rule)
#[inline]
fn l2_normalize(x: &Tensor) -> Tensor {
    // norm_scalaropt_dim = 2 GPU ops (vs 5 with manual x*x+sum+sqrt+div)
    let norm = x.norm_scalaropt_dim(2.0, &[-1i64][..], true);
    x / norm.clamp_min(1e-6)
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
    let attn = -(k_beta.matmul(&k.transpose(-1, -2)) * &decay_mask)
        .masked_fill(&mask0, 0.0f64);

    // Recursive delta-rule correction (loop over chunk_size positions)
    for i in 1..CHUNK_SIZE as usize {
        let i = i as i64;
        // row = attn[..., i, :i]
        let row = attn.narrow(-2, i, 1).narrow(-1, 0, i).squeeze_dim(-2); // [B, nv, nc, i]
        // sub = attn[..., :i, :i]
        let sub = attn.narrow(-2, 0, i).narrow(-1, 0, i); // [B, nv, nc, i, i]
        // correction = (row @ sub)  [B, nv, nc, i]
        let correction = row.unsqueeze(-2).matmul(&sub).squeeze_dim(-2);
        let new_row = (&row + &correction).unsqueeze(-2); // [B, nv, nc, 1, i]
        attn.narrow(-2, i, 1).narrow(-1, 0, i).copy_(&new_row);
    }

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

    // Stack chunks: [B, nv, nc, C, hv] → flatten → trim padding
    let out = Tensor::stack(&chunks_out, 2) // [B, nv, nc, C, hv]
        .reshape([batch, nv, seq_p, hv])
        .narrow(2, 0, seq);

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
    ) -> Result<Tensor> {
        let (batch, seq, _) = dims3(hidden)?;
        let device = hidden.device();
        let dtype = hidden.kind();

        // Fused input projection: 1 matmul for all 4 projections.
        let h2 = hidden.reshape([batch * seq, hidden.size()[2]]);
        let all_proj = self.in_proj_all.apply(&h2); // [batch*seq, conv_dim+val_dim+2*nv]
        let [cd, vd, nv_d, _] = self.proj_split;
        let mut off = 0i64;
        let mixed_qkv = all_proj.narrow(1, off, cd).reshape([batch, seq, cd]); off += cd;
        let z         = all_proj.narrow(1, off, vd).reshape([batch, seq, vd]); off += vd;
        let b         = all_proj.narrow(1, off, nv_d).reshape([batch, seq, nv_d]); off += nv_d;
        let a_in      = all_proj.narrow(1, off, nv_d).reshape([batch, seq, nv_d]);

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
            // out_t: [batch, nv, hv] → [batch, 1, nv, hv] → [batch, 1, val_dim]
            out_t.unsqueeze(1).reshape([batch, 1, val_dim])
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

            // Transpose back to [B, T, nv, hv] and flatten
            out_h.permute([0, 2, 1, 3]).reshape([batch, seq, val_dim])
        };

        // Gated norm and output projection
        // RMSNormGated weight is [head_v_dim]; reshape to per-head before norm
        let z_f = z; // [batch, seq, value_dim]

        let out_4d = out.reshape([batch, seq, nv, hv]);
        let z_4d = z_f.reshape([batch, seq, nv, hv]);
        let normed = self.norm.forward(&out_4d, &z_4d)?;  // out_4d may be F32 from recurrent step; RMSNormGated handles both
        // Ensure output matches model dtype for out_proj matmul (weight is BF16)
        let normed = if normed.kind() != dtype { normed.to_kind(dtype) } else { normed };
        let flat = normed.reshape([batch * seq, val_dim]);
        let projected = self.out_proj.apply(&flat);
        Ok(projected.reshape([batch, seq, self.hidden_size_from_out_proj()]))
    }

    fn hidden_size_from_out_proj(&self) -> i64 {
        self.out_proj.weight.size()[1]
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
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rotary_dim,
            rope_theta: cfg.rope_theta,
            max_pos: cfg.max_position_embeddings,
            layer_idx,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        position_ids: Option<&Tensor>,
        kv_cache: Option<&mut crate::runtime::kv_cache::LayerKVCache>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let (batch, seq, _) = dims3(hidden)?;
        let device = hidden.device();
        let dtype = hidden.kind();
        let h2 = hidden.reshape([batch * seq, hidden.size()[2]]);

        // Fused QKV projection: single matmul, then split
        let qkv = self.qkv_proj.apply(&h2);
        let [q_dim, k_dim, v_dim] = self.qkv_split;
        let q_out = qkv.narrow(1, 0, q_dim);
        let k_out = qkv.narrow(1, q_dim, k_dim);
        let v_out = qkv.narrow(1, q_dim + k_dim, v_dim);

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

        // Manual attention in f32 for numerical stability
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_p = query.permute([0, 2, 1, 3]).to_kind(Kind::Float).contiguous();
        let k_p = k_exp.permute([0, 2, 3, 1]).to_kind(Kind::Float).contiguous();
        let v_p = v_exp.permute([0, 2, 1, 3]).to_kind(Kind::Float).contiguous();

        let mut scores = q_p.matmul(&k_p) * scale; // [batch, heads, q_seq, kv_seq]
        let q_len = scores.size()[2];
        let kv_len = scores.size()[3];
        if q_len > 1 {
            let mask = Tensor::ones([q_len, kv_len], (Kind::Float, device)).tril(0);
            let mask = mask.unsqueeze(0).unsqueeze(0).expand_as(&scores);
            scores = scores.masked_fill(&mask.eq(0.0), -10000.0f64);
        }
        let attn = scores.softmax(-1, Kind::Float).to_kind(dtype);
        let ctx = attn.matmul(&v_p.to_kind(dtype)); // [batch, heads, seq, head_dim]
        let ctx = ctx.permute([0, 2, 1, 3]).contiguous(); // [batch, seq, heads, head_dim]

        // Apply output gate: gate_flat[batch*seq, nh*hd] * sigmoid(gate) BEFORE o_proj
        let ctx_flat = ctx.reshape([batch * seq, nh * hd]);
        let gate_flat = gate.reshape([batch * seq, nh * hd]).sigmoid();
        let ctx_gated = ctx_flat * gate_flat;
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
        thread_local! {
            static ROPE_CACHE: RefCell<HashMap<(usize, u32), RoPE>> =
                RefCell::new(HashMap::new());
        }

        let rd = self.rotary_dim as i64;
        let seq = x.size()[1];
        let pos_ids = position_ids.map(|p| p.shallow_clone()).unwrap_or_else(|| {
            Tensor::arange_start(start_pos as i64, start_pos as i64 + seq, (Kind::Int64, device))
        });

        ROPE_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let key = (self.layer_idx, (self.rope_theta * 1000.0) as u32);
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

unsafe impl Send for LayerMixer {}
unsafe impl Sync for LayerMixer {}

struct Qwen3_5Layer {
    mixer: LayerMixer,
    mlp: Qwen3_5Mlp,
    input_layernorm: Qwen3_5RMSNorm,
    post_attention_layernorm: Qwen3_5RMSNorm,
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
    embed_tokens: Tensor,
    layers: Vec<Qwen3_5Layer>,
    norm: Qwen3_5RMSNorm,
    lm_head: Option<Tensor>,             // None = tied to embed_tokens
    lm_head_transposed: Option<Tensor>,  // pre-transposed tied weights
    // Interior-mutable SSM state (required because forward_with_cache takes &self)
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

        let embed = weights.remove("model.embed_tokens.weight")
            .ok_or_else(|| anyhow!("Missing model.embed_tokens.weight"))?;
        let norm_w = weights.remove("model.norm.weight")
            .ok_or_else(|| anyhow!("Missing model.norm.weight"))?;
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
                        let s_exp = s.to_kind(tch::Kind::BFloat16)
                            .view([ss[0], 1i64, ss[1], 1i64])
                            .expand([ss[0], br, ss[1], bc], false)
                            .reshape([ws[0], ws[1]]);
                        Some(w_bf * s_exp)
                    } else {
                        Some(w_bf)
                    }
                }
                _ => Some(w),
            }
        } else {
            None
        };

        let num_layers = cfg.num_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers);

        for idx in 0..num_layers {
            let layer_prefix = format!("model.layers.{idx}");
            let ln_prefix = &layer_prefix;

            let input_norm_w = weights.remove(&format!("{ln_prefix}.input_layernorm.weight"))
                .ok_or_else(|| anyhow!("Missing {ln_prefix}.input_layernorm.weight"))?;
            let post_norm_w = weights.remove(&format!("{ln_prefix}.post_attention_layernorm.weight"))
                .ok_or_else(|| anyhow!("Missing {ln_prefix}.post_attention_layernorm.weight"))?;

            let layer_type = cfg.layer_types.get(idx).map(|s| s.as_str()).unwrap_or("linear_attention");
            info!("Loading layer {idx}: {layer_type}");

            let mixer = if layer_type == "full_attention" {
                let attn_prefix = format!("{layer_prefix}.self_attn");
                LayerMixer::FullAttn(Qwen3_5FullAttention::load(weights, &attn_prefix, &cfg, idx)?)
            } else {
                let lin_prefix = format!("{layer_prefix}.linear_attn");
                LayerMixer::LinearAttn(GatedDeltaNetLayer::load(weights, &lin_prefix, &cfg, idx)?)
            };

            let mlp = Self::load_mlp(weights, &format!("{layer_prefix}.mlp"), &cfg, idx)?;

            layers.push(Qwen3_5Layer {
                mixer,
                mlp,
                input_layernorm: Qwen3_5RMSNorm::new(input_norm_w, cfg.rms_norm_eps),
                post_attention_layernorm: Qwen3_5RMSNorm::new(post_norm_w, cfg.rms_norm_eps),
            });
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
                cfg.max_position_embeddings,
                _kv_quant_type,
            ),
        )));

        Ok(Self {
            config: cfg,
            device: *device,
            dtype,
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

    fn forward_inner(
        &self,
        input_ids: Option<&Tensor>,
        embeddings: Option<&Tensor>,
        start_pos: usize,
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

        let (batch, seq) = (hidden.size()[0], hidden.size()[1]);
        let device = self.device;

        // Position IDs for full-attention RoPE
        let position_ids = Tensor::arange_start(
            start_pos as i64,
            start_pos as i64 + seq,
            (Kind::Int64, device),
        );

        let mut conv_guard = self.conv_states.lock();
        let mut rec_guard = self.rec_states.lock();
        // kv_arc dropped per-layer to avoid holding lock across SSM layers

        for (idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden.shallow_clone();

            hidden = layer.input_layernorm.forward(&hidden)?;

            let mixer_out = match &layer.mixer {
                LayerMixer::LinearAttn(gdn) => {
                    gdn.forward(&hidden, &mut conv_guard[idx], &mut rec_guard[idx])?
                }
                LayerMixer::FullAttn(attn) => {
                    if let Some(ref cache_arc) = self.kv_cache {
                        let kv = cache_arc.lock();
                        kv.with_layer_cache(idx, |lc| {
                            attn.forward(&hidden, Some(&position_ids), Some(lc), start_pos)
                        })
                        .ok_or_else(|| anyhow!("No KV cache for layer {idx}"))??
                    } else {
                        attn.forward(&hidden, Some(&position_ids), None, start_pos)?
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
        let hidden = self.forward_inner(Some(input), None, 0)?;
        let normed = self.norm.forward(&hidden)?;
        self.lm_head_apply(&normed)
    }

    fn forward_with_cache(&self, input: &Tensor, start_pos: usize) -> Result<Tensor> {
        let hidden = self.forward_inner(Some(input), None, start_pos)?;
        let normed = self.norm.forward(&hidden)?;
        self.lm_head_apply(&normed)
    }

    fn forward_from_embeddings(&self, embeddings: &Tensor, start_pos: usize) -> Result<Tensor> {
        let hidden = self.forward_inner(None, Some(embeddings), start_pos)?;
        let normed = self.norm.forward(&hidden)?;
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
        self.config.num_hidden_layers
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
