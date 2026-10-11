//! Torch-native GatedDeltaNet: the chunked gated delta rule as pure tensor ops
//! (the S6c feasibility pin — no Triton, no fla kernels, autograd-safe).
//!
//! This is an independent encoder-side implementation of the public
//! `torch_chunk_gated_delta_rule` algorithm (Yang et al., "Gated Delta Networks",
//! and the Qwen3-Next/Qwen3.5 hybrid reference). Only the full-sequence
//! (training/prefill) path exists here: the decision model always scores complete
//! serialized specs, so there is no KV cache, no recurrent state threading, and no
//! incremental decode — and no LM head downstream.

use tch::{Kind, Tensor};

use crate::Error;

/// Chunk size for the blocked algorithm: O(seq · chunk) memory/compute trade-off,
/// matching the reference implementation.
const CHUNK_SIZE: i64 = 64;

/// L2-normalize along the last dim (applied to Q and K before the delta rule).
fn l2_normalize(x: &Tensor) -> Tensor {
    let norm = x.norm_scalaropt_dim(2.0, &[-1i64][..], true);
    x / norm.clamp_min(1e-3)
}

/// Chunked gated delta rule over a full sequence. All inputs are
/// `[batch, heads, seq, dim]` (heads second), already in `Float`; `q` is
/// L2-normalized and pre-scaled by `1/sqrt(head_k_dim)`. `g` is the per-position
/// log-decay (negative), `beta` the per-position write gate in (0, 1).
///
/// Returns the per-position outputs `[batch, heads, seq, head_v_dim]`. The
/// within-chunk delta-rule correction is built out-of-place (row stacking, no
/// in-place writes to tensors saved for backward) so the graph stays live for
/// training — the same numerics as the in-place reference form.
pub(crate) fn chunked_gated_delta_rule(
    q: &Tensor,    // [B, H, T, hk]
    k: &Tensor,    // [B, H, T, hk]
    v: &Tensor,    // [B, H, T, hv]
    g: &Tensor,    // [B, H, T]
    beta: &Tensor, // [B, H, T]
) -> Result<Tensor, Error> {
    let size = q.size();
    let (batch, heads, seq, hk) = (size[0], size[1], size[2], size[3]);
    let hv = v.size()[3];
    let device = q.device();

    // Pad the sequence to a multiple of CHUNK_SIZE; padded positions carry zero
    // decay/gate contributions and are trimmed at the end.
    let pad = (CHUNK_SIZE - seq % CHUNK_SIZE) % CHUNK_SIZE;
    let seq_p = seq + pad;
    let pad_right = |t: &Tensor, last: i64| -> Tensor {
        if pad == 0 {
            return t.shallow_clone();
        }
        let zeros = if last == 0 {
            Tensor::zeros([batch, heads, pad], (Kind::Float, device))
        } else {
            Tensor::zeros([batch, heads, pad, last], (Kind::Float, device))
        };
        Tensor::cat(&[t, &zeros], 2)
    };

    let q = pad_right(q, hk);
    let k = pad_right(k, hk);
    let v = pad_right(v, hv);
    let g = pad_right(g, 0);
    let beta = pad_right(beta, 0);

    let v_beta = &v * beta.unsqueeze(-1);
    let k_beta = &k * beta.unsqueeze(-1);

    let chunks = seq_p / CHUNK_SIZE;
    let to_chunks = |t: &Tensor, d: i64| t.reshape([batch, heads, chunks, CHUNK_SIZE, d]);
    let q = to_chunks(&q, hk);
    let k = to_chunks(&k, hk);
    let k_beta = to_chunks(&k_beta, hk);
    let v_beta = to_chunks(&v_beta, hv);
    // Cumulative intra-chunk decay; g[..., t] is the total decay from the chunk start.
    let g = g
        .reshape([batch, heads, chunks, CHUNK_SIZE])
        .cumsum(-1, Kind::Float);

    // decay[t, s] = exp(g[t] - g[s]) for t >= s (exp first, then tril — otherwise the
    // upper triangle would read exp(0) = 1 instead of 0).
    let decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).exp().tril(0); // [B, H, C, cs, cs]

    let diag_and_above = Tensor::ones([CHUNK_SIZE, CHUNK_SIZE], (Kind::Bool, device)).triu(0);
    let strictly_above = Tensor::ones([CHUNK_SIZE, CHUNK_SIZE], (Kind::Bool, device)).triu(1);

    // A = -(k_beta @ k^T * decay), future columns zeroed.
    let a =
        -(k_beta.matmul(&k.transpose(-1, -2)) * &decay).f_masked_fill(&diag_and_above, 0.0f64)?;

    // Forward substitution for T = (I - A_strict)^{-1}-style correction: row i is
    // T[i, :i] = A[i, :i] + A[i, :i] @ T[:i, :i], depending only on corrected rows
    // above it. Out-of-place stacking keeps autograd safe.
    let cs = CHUNK_SIZE;
    let mut rows: Vec<Tensor> = Vec::with_capacity(cs as usize);
    rows.push(a.narrow(-2, 0, 1).squeeze_dim(-2));
    for i in 1..cs {
        let a_row = a.narrow(-2, i, 1).narrow(-1, 0, i).squeeze_dim(-2); // [B, H, C, i]
        let prev = Tensor::stack(&rows, -2).narrow(-1, 0, i); // [B, H, C, i, i]
        let corrected = &a_row + a_row.unsqueeze(-2).matmul(&prev).squeeze_dim(-2);
        let zeros = Tensor::zeros([batch, heads, chunks, cs - i], (Kind::Float, device));
        rows.push(Tensor::cat(&[&corrected, &zeros], -1));
    }
    let eye = Tensor::eye(cs, (Kind::Float, device)).view([1, 1, 1, cs, cs]);
    let correction = Tensor::stack(&rows, -2) + eye; // [B, H, C, cs, cs]

    let value_t = correction.matmul(&v_beta); // [B, H, C, cs, hv]
    let k_decayed = correction.matmul(&(&k_beta * g.exp().unsqueeze(-1))); // [B, H, C, cs, hk]

    // Chunk scan: carry the [hk, hv] recurrent state across chunks.
    let mut state = Tensor::zeros([batch, heads, hk, hv], (Kind::Float, device));
    let mut outputs = Vec::with_capacity(chunks as usize);
    for c in 0..chunks {
        let q_c = q.select(2, c); // [B, H, cs, hk]
        let k_c = k.select(2, c);
        let v_c = value_t.select(2, c); // [B, H, cs, hv]
        let g_c = g.select(2, c); // [B, H, cs]
        let decay_c = decay.select(2, c); // [B, H, cs, cs]
        let kd_c = k_decayed.select(2, c); // [B, H, cs, hk]

        // Within-chunk (causal, strictly-above-future masked) + cross-chunk state read.
        let attn_inner = (q_c.matmul(&k_c.transpose(-1, -2)) * &decay_c)
            .f_masked_fill(&strictly_above, 0.0f64)?;
        let v_new = &v_c - kd_c.matmul(&state);
        let attn_inter = (&q_c * g_c.exp().unsqueeze(-1)).matmul(&state);
        outputs.push(attn_inter + attn_inner.matmul(&v_new));

        // State update: decay the whole state by the chunk's total decay, then add the
        // per-position writes reweighted by their remaining decay to the chunk end.
        let g_last = g_c.narrow(-1, cs - 1, 1); // [B, H, 1]
        let state_decay = g_last.exp().unsqueeze(-1); // [B, H, 1, 1]
        let remaining = (g_last - &g_c).exp().unsqueeze(-1); // [B, H, cs, 1]
        let writes = (k_c * remaining).transpose(-1, -2).matmul(&v_new);
        state = state * state_decay + writes;
    }

    let out = Tensor::stack(&outputs, 2) // [B, H, C, cs, hv]
        .reshape([batch, heads, seq_p, hv])
        .narrow(2, 0, seq)
        .contiguous();
    Ok(out)
}

/// Per-layer GatedDeltaNet mixer (Qwen3.5 hybrid layout).
///
/// Fused projections under one `nn::Path` with the checkpoint's names:
/// `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, depthwise causal
/// `conv1d`, decay parameters `A_log`/`dt_bias`, gated output `norm`, `out_proj`.
pub(crate) struct GdnMixer {
    in_proj_qkv: tch::nn::Linear,
    in_proj_z: tch::nn::Linear,
    in_proj_b: tch::nn::Linear,
    in_proj_a: tch::nn::Linear,
    conv1d_weight: Tensor, // [conv_dim, 1, kernel]
    a_log: Tensor,         // [num_v_heads]
    dt_bias: Tensor,       // [num_v_heads]
    norm_weight: Tensor,   // [head_v_dim]
    out_proj: tch::nn::Linear,
    num_k_heads: i64,
    num_v_heads: i64,
    head_k_dim: i64,
    head_v_dim: i64,
    kernel_size: i64,
    norm_eps: f64,
}

/// Dims that define a GDN layer, derived from the model config.
pub(crate) struct GdnDims {
    pub hidden: i64,
    pub num_k_heads: i64,
    pub num_v_heads: i64,
    pub head_k_dim: i64,
    pub head_v_dim: i64,
    pub kernel_size: i64,
    pub norm_eps: f64,
}

impl GdnMixer {
    pub(crate) fn new(path: &tch::nn::Path, dims: &GdnDims) -> Self {
        let key_dim = dims.num_k_heads * dims.head_k_dim;
        let value_dim = dims.num_v_heads * dims.head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let no_bias = tch::nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        Self {
            in_proj_qkv: tch::nn::linear(path.sub("in_proj_qkv"), dims.hidden, conv_dim, no_bias),
            in_proj_z: tch::nn::linear(path.sub("in_proj_z"), dims.hidden, value_dim, no_bias),
            in_proj_b: tch::nn::linear(
                path.sub("in_proj_b"),
                dims.hidden,
                dims.num_v_heads,
                no_bias,
            ),
            in_proj_a: tch::nn::linear(
                path.sub("in_proj_a"),
                dims.hidden,
                dims.num_v_heads,
                no_bias,
            ),
            conv1d_weight: path.sub("conv1d").var(
                "weight",
                &[conv_dim, 1, dims.kernel_size],
                tch::nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            ),
            a_log: path.var("A_log", &[dims.num_v_heads], tch::nn::Init::Const(0.0)),
            dt_bias: path.var("dt_bias", &[dims.num_v_heads], tch::nn::Init::Const(1.0)),
            norm_weight: path.sub("norm").var(
                "weight",
                &[dims.head_v_dim],
                tch::nn::Init::Const(1.0),
            ),
            out_proj: tch::nn::linear(path.sub("out_proj"), value_dim, dims.hidden, no_bias),
            num_k_heads: dims.num_k_heads,
            num_v_heads: dims.num_v_heads,
            head_k_dim: dims.head_k_dim,
            head_v_dim: dims.head_v_dim,
            kernel_size: dims.kernel_size,
            norm_eps: dims.norm_eps,
        }
    }

    /// Full-sequence forward: `[batch, seq, hidden]` → `[batch, seq, hidden]`.
    /// Causal by construction — output position *t* depends only on inputs ≤ *t*
    /// (the conversion leaves GDN layers untouched; bidirectionality enters only
    /// through the softmax layers).
    pub(crate) fn forward(&self, hidden: &Tensor) -> Result<Tensor, Error> {
        let size = hidden.size();
        let (batch, seq, _) = (size[0], size[1], size[2]);
        let dtype = hidden.kind();
        let device = hidden.device();
        let key_dim = self.num_k_heads * self.head_k_dim;
        let value_dim = self.num_v_heads * self.head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;

        let flat = hidden.reshape([batch * seq, size[2]]);
        let mixed_qkv = flat.apply(&self.in_proj_qkv).view([batch, seq, conv_dim]);
        let z = flat.apply(&self.in_proj_z).view([batch, seq, value_dim]);
        let b = flat
            .apply(&self.in_proj_b)
            .view([batch, seq, self.num_v_heads])
            .to_kind(Kind::Float)
            .sigmoid(); // [B, T, nv]
        let a = flat
            .apply(&self.in_proj_a)
            .view([batch, seq, self.num_v_heads])
            .to_kind(Kind::Float); // [B, T, nv]

        // Depthwise causal conv1d over mixed QKV: left-pad by kernel-1, no right
        // context, silu activation.
        let x_t = mixed_qkv.permute([0, 2, 1]); // [B, conv_dim, T]
        let pad = Tensor::zeros([batch, conv_dim, self.kernel_size - 1], (dtype, device));
        let padded = Tensor::cat(&[&pad, &x_t], 2);
        let conv_out = padded
            .conv1d(
                &self.conv1d_weight,
                None::<&Tensor>,
                &[1i64][..],
                &[0i64][..],
                &[1i64][..],
                conv_dim,
            )
            .silu()
            .permute([0, 2, 1]); // [B, T, conv_dim]

        let q = conv_out
            .narrow(2, 0, key_dim)
            .reshape([batch, seq, self.num_k_heads, self.head_k_dim])
            .to_kind(Kind::Float);
        let k = conv_out
            .narrow(2, key_dim, key_dim)
            .reshape([batch, seq, self.num_k_heads, self.head_k_dim])
            .to_kind(Kind::Float);
        let v = conv_out
            .narrow(2, key_dim * 2, value_dim)
            .reshape([batch, seq, self.num_v_heads, self.head_v_dim])
            .to_kind(Kind::Float);

        let scale = 1.0 / (self.head_k_dim as f64).sqrt();
        let q = l2_normalize(&q) * scale;
        let k = l2_normalize(&k);

        // GQA-style head sharing: each key head serves (nv / nk) consecutive value
        // heads — repeat_interleave, not tile.
        let expand = |t: &Tensor| -> Tensor {
            if self.num_k_heads == self.num_v_heads {
                t.shallow_clone()
            } else {
                t.repeat_interleave_self_int(self.num_v_heads / self.num_k_heads, 2, None)
            }
        };
        let q = expand(&q);
        let k = expand(&k);

        // Log-decay g = -exp(A_log) * softplus(a + dt_bias) (negative).
        let neg_a = self.a_log.to_kind(Kind::Float).exp().neg(); // [nv]
        let g = (a + self.dt_bias.to_kind(Kind::Float)).softplus() * neg_a; // [B, T, nv]

        let out = chunked_gated_delta_rule(
            &q.permute([0, 2, 1, 3]),
            &k.permute([0, 2, 1, 3]),
            &v.permute([0, 2, 1, 3]),
            &g.permute([0, 2, 1]),
            &b.permute([0, 2, 1]),
        )?; // [B, nv, T, hv]

        let out = out.permute([0, 2, 1, 3]).contiguous().view([
            batch,
            seq,
            self.num_v_heads,
            self.head_v_dim,
        ]);
        let z = z
            .view([batch, seq, self.num_v_heads, self.head_v_dim])
            .to_kind(Kind::Float);

        // Gated RMSNorm over the head dim: norm(x) * weight * silu(gate).
        let mean_sq = (&out * &out).mean_dim(&[-1i64][..], true, Kind::Float);
        let normed = &out * (mean_sq + self.norm_eps).rsqrt();
        let gated = normed * self.norm_weight.to_kind(Kind::Float) * z.silu();
        let gated = gated.to_kind(dtype).reshape([batch * seq, value_dim]);

        Ok(gated.apply(&self.out_proj).view([batch, seq, size[2]]))
    }
}
