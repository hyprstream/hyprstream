//! The converted Qwen3.5-0.8B hybrid encoder (LLM2Vec-style, partial).
//!
//! Architecture: 24 layers for the 0.8B hybrid — every 4th layer is full softmax
//! attention (6 total), the other 18 are GatedDeltaNet. The conversion:
//!
//! - softmax layers run under [`MaskPolicy::Bidirectional`] (the LLM2Vec move);
//! - GDN layers are untouched (causal chunked delta rule, [`crate::gdn`]);
//! - no LM head, no MTP block, no KV cache — output is the post-final-norm hidden
//!   state consumed by the shared scorer.
//!
//! Parameter names follow the checkpoint layout (`model.layers.{i}.self_attn.*`,
//! `model.layers.{i}.linear_attn.*`, `model.embed_tokens`, `model.norm`) so a
//! converted `VarStore` loads the HF-format safetensors directly after the
//! `lm_head.*` tensors are dropped ([`Qwen35Encoder::amputated_lm_head_names`] —
//! that list *is* the amputation: the encoder never allocates them).

use serde::Deserialize;
use tch::nn::{self, Init};
use tch::{Device, Kind, Tensor};

use crate::backbone::{Backbone, MaskPolicy};
use crate::gdn::{GdnDims, GdnMixer};
use crate::Error;

/// Which mixer a layer carries, per the checkpoint's `layer_types` (or the
/// derived every-4th-full-attention default when the checkpoint omits it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    LinearAttention,
    FullAttention,
}

/// Qwen3.5 hybrid text config (HF `config.json` shape, serde-compatible).
///
/// The 0.8B hybrid values are read from the checkpoint's own `config.json` at
/// conversion time; [`Qwen35Config::test_tiny`] provides a miniature for CPU tests.
/// The serde defaults (linear head dim 128, 16 k-heads / 32 v-heads, rope theta
/// 10M, partial rotary 0.25) are **placeholders** for when a text-only checkpoint
/// omits the fields — they are plausible, not verified against a real 0.8B
/// config.json (none exists in-repo yet).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub head_dim: i64,
    pub intermediate_size: i64,
    pub vocab_size: i64,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: i64,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Fraction of `head_dim` that is rotated (partial RoPE).
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
    /// Per-layer `"linear_attention"` / `"full_attention"`; empty derives every-4th.
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: i64,
    #[serde(default = "default_linear_head_dim")]
    pub linear_key_head_dim: i64,
    #[serde(default = "default_linear_head_dim")]
    pub linear_value_head_dim: i64,
    #[serde(default = "default_linear_k_heads")]
    pub linear_num_key_heads: i64,
    #[serde(default = "default_linear_v_heads")]
    pub linear_num_value_heads: i64,
}

fn default_max_positions() -> i64 {
    32768
}
fn default_rms_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10_000_000.0
}
fn default_partial_rotary() -> f64 {
    0.25
}
fn default_conv_kernel() -> i64 {
    4
}
fn default_linear_head_dim() -> i64 {
    128
}
fn default_linear_k_heads() -> i64 {
    16
}
fn default_linear_v_heads() -> i64 {
    32
}

impl Qwen35Config {
    /// Miniature config for CPU tests and the smoke test: same hybrid shape
    /// (GDN + every-4th full attention), tiny dims.
    pub fn test_tiny() -> Self {
        Self {
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            layer_types: Vec::new(),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
        }
    }

    fn layer_kinds(&self) -> Result<Vec<LayerKind>, Error> {
        if self.layer_types.is_empty() {
            return Ok((0..self.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % 4 == 0 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
                })
                .collect());
        }
        if self.layer_types.len() != self.num_hidden_layers as usize {
            return Err(Error::Config(format!(
                "layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        self.layer_types
            .iter()
            .map(|t| match t.as_str() {
                "linear_attention" => Ok(LayerKind::LinearAttention),
                "full_attention" => Ok(LayerKind::FullAttention),
                other => Err(Error::Config(format!("unknown layer_type {other:?}"))),
            })
            .collect()
    }

    fn rotary_dim(&self) -> i64 {
        ((self.head_dim as f64) * self.partial_rotary_factor) as i64
    }
}

/// Qwen3.5 RMSNorm: `norm(x_f32) * (1 + weight)` — checkpoint weights are trained
/// around zero, so the effective gain is `1 + w`. This zero-centered convention is
/// pinned to the Qwen3-Next-style checkpoint family; if the 0.8B hybrid checkpoint
/// ships unit-centered weights instead, this norm is where the numerics would
/// diverge (verify at conversion time, P1.4).
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(path: &nn::Path, dim: i64, eps: f64) -> Self {
        Self {
            weight: path.var("weight", &[dim], Init::Const(0.0)),
            eps,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let dtype = x.kind();
        let x_f = x.to_kind(Kind::Float);
        let mean_sq = (&x_f * &x_f).mean_dim(&[-1i64][..], true, Kind::Float);
        let normed = x_f * (mean_sq + self.eps).rsqrt();
        (normed * (self.weight.to_kind(Kind::Float) + 1.0)).to_kind(dtype)
    }
}

/// Rotary embedding tables for the softmax layers (partial RoPE: only the first
/// `rotary_dim` of each head rotates).
struct RotaryEmbedding {
    cos: Tensor, // [max_pos, rotary_dim]
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(rotary_dim: i64, theta: f64, max_pos: i64, device: Device) -> Result<Self, Error> {
        if rotary_dim % 2 != 0 {
            return Err(Error::Config(format!(
                "rotary_dim {rotary_dim} must be even (head_dim * partial_rotary_factor)"
            )));
        }
        let half = rotary_dim / 2;
        let inv_freq = Tensor::arange(half, (Kind::Float, device))
            .to_kind(Kind::Float)
            .neg()
            * (2.0 / rotary_dim as f64)
            * theta.ln();
        let inv_freq = inv_freq.exp();
        let t = Tensor::arange(max_pos, (Kind::Float, device));
        let freqs = t.unsqueeze(-1).matmul(&inv_freq.unsqueeze(0)); // [pos, half]
        let emb = Tensor::cat(&[&freqs, &freqs], -1); // [pos, rotary_dim]
        Ok(Self {
            cos: emb.cos(),
            sin: emb.sin(),
        })
    }

    /// Apply rotation to `[batch, heads, seq, head_dim]`; only the leading
    /// `rotary_dim` slice rotates, the tail passes through (partial RoPE).
    fn apply(&self, x: &Tensor) -> Tensor {
        let size = x.size();
        let seq = size[2];
        let rotary = self.cos.size()[1];
        let cos = self
            .cos
            .narrow(0, 0, seq)
            .view([1, 1, seq, rotary])
            .to_kind(x.kind());
        let sin = self
            .sin
            .narrow(0, 0, seq)
            .view([1, 1, seq, rotary])
            .to_kind(x.kind());
        let rot = x.narrow(-1, 0, rotary);
        let half = rotary / 2;
        let x1 = rot.narrow(-1, 0, half);
        let x2 = rot.narrow(-1, half, half);
        let rotated = Tensor::cat(&[&x2.neg(), &x1], -1);
        let out = &rot * &cos + rotated * &sin;
        Tensor::cat(&[&out, &x.narrow(-1, rotary, size[3] - rotary)], -1)
    }
}

/// Softmax attention layer: GQA + per-head Q/K RMSNorm + partial RoPE, under the
/// conversion's mask policy. Bias-free projections, checkpoint names.
struct FullAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
}

impl FullAttention {
    fn new(path: &nn::Path, cfg: &Qwen35Config) -> Self {
        let no_bias = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        Self {
            q_proj: nn::linear(
                path.sub("q_proj"),
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                no_bias,
            ),
            k_proj: nn::linear(
                path.sub("k_proj"),
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                no_bias,
            ),
            v_proj: nn::linear(
                path.sub("v_proj"),
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                no_bias,
            ),
            o_proj: nn::linear(
                path.sub("o_proj"),
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                no_bias,
            ),
            q_norm: RmsNorm::new(&path.sub("q_norm"), cfg.head_dim, cfg.rms_norm_eps),
            k_norm: RmsNorm::new(&path.sub("k_norm"), cfg.head_dim, cfg.rms_norm_eps),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        }
    }

    fn forward(&self, hidden: &Tensor, rope: &RotaryEmbedding, mask: &Tensor) -> Tensor {
        let size = hidden.size();
        let (batch, seq, _) = (size[0], size[1], size[2]);
        let flat = hidden.reshape([batch * seq, size[2]]);
        let split = |t: Tensor, heads: i64| {
            t.view([batch, seq, heads, self.head_dim])
                .permute([0, 2, 1, 3])
        };
        let q = split(flat.apply(&self.q_proj), self.num_heads);
        let k = split(flat.apply(&self.k_proj), self.num_kv_heads);
        let v = split(flat.apply(&self.v_proj), self.num_kv_heads);

        // Per-head Q/K norm, then partial RoPE.
        let q = rope.apply(&self.q_norm.forward(&q));
        let k = rope.apply(&self.k_norm.forward(&k));

        let repeat = self.num_heads / self.num_kv_heads;
        let k = k.repeat_interleave_self_int(repeat, 1, None);
        let v = v.repeat_interleave_self_int(repeat, 1, None);

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores =
            q.matmul(&k.transpose(-1, -2)).to_kind(Kind::Float) * scale + mask.to_kind(Kind::Float);
        let probs = scores.softmax(-1, Kind::Float).to_kind(hidden.kind());
        probs
            .matmul(&v)
            .permute([0, 2, 1, 3])
            .contiguous()
            .view([batch * seq, self.num_heads * self.head_dim])
            .apply(&self.o_proj)
            .view([batch, seq, size[2]])
    }
}

/// SwiGLU MLP (gate/up/down, bias-free).
struct Mlp {
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_proj: nn::Linear,
}

impl Mlp {
    fn new(path: &nn::Path, cfg: &Qwen35Config) -> Self {
        let no_bias = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        Self {
            gate_proj: nn::linear(
                path.sub("gate_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
                no_bias,
            ),
            up_proj: nn::linear(
                path.sub("up_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
                no_bias,
            ),
            down_proj: nn::linear(
                path.sub("down_proj"),
                cfg.intermediate_size,
                cfg.hidden_size,
                no_bias,
            ),
        }
    }

    fn forward(&self, hidden: &Tensor) -> Tensor {
        let size = hidden.size();
        let flat = hidden.reshape([size[0] * size[1], size[2]]);
        (flat.apply(&self.gate_proj).silu() * flat.apply(&self.up_proj))
            .apply(&self.down_proj)
            .view([size[0], size[1], size[2]])
    }
}

enum Mixer {
    Gdn(GdnMixer),
    Attn(FullAttention),
}

struct Layer {
    input_norm: RmsNorm,
    mixer: Mixer,
    post_norm: RmsNorm,
    mlp: Mlp,
}

/// The converted Qwen3.5 hybrid encoder. Construct under a `nn::Path` scoped to
/// the model root (the embedding/norm variables sit directly under it, layers
/// under `layers.{i}`), with checkpoint-compatible names.
pub struct Qwen35Encoder {
    embeddings: nn::Embedding,
    layers: Vec<Layer>,
    final_norm: RmsNorm,
    rope: RotaryEmbedding,
    mask_policy: MaskPolicy,
    hidden_dim: i64,
}

impl Qwen35Encoder {
    /// Build the encoder under `path`. `mask_policy` is the conversion switch:
    /// [`MaskPolicy::Bidirectional`] for the converted decision model (the default
    /// arm), [`MaskPolicy::Causal`] to reproduce the unconverted decoder's masking
    /// (ablation/comparison only — P1.5 territory). GDN layers ignore the policy:
    /// the delta rule is causal by construction.
    pub fn new(
        path: &nn::Path,
        cfg: &Qwen35Config,
        mask_policy: MaskPolicy,
    ) -> Result<Self, Error> {
        let kinds = cfg.layer_kinds()?;
        let embeddings = nn::embedding(
            path.sub("embed_tokens"),
            cfg.vocab_size,
            cfg.hidden_size,
            Default::default(),
        );
        let mut layers = Vec::with_capacity(kinds.len());
        for (i, kind) in kinds.iter().enumerate() {
            let lp = path.sub("layers").sub(i.to_string());
            let mixer = match kind {
                LayerKind::LinearAttention => Mixer::Gdn(GdnMixer::new(
                    &lp.sub("linear_attn"),
                    &GdnDims {
                        hidden: cfg.hidden_size,
                        num_k_heads: cfg.linear_num_key_heads,
                        num_v_heads: cfg.linear_num_value_heads,
                        head_k_dim: cfg.linear_key_head_dim,
                        head_v_dim: cfg.linear_value_head_dim,
                        kernel_size: cfg.linear_conv_kernel_dim,
                        norm_eps: cfg.rms_norm_eps,
                    },
                )),
                LayerKind::FullAttention => {
                    Mixer::Attn(FullAttention::new(&lp.sub("self_attn"), cfg))
                }
            };
            layers.push(Layer {
                input_norm: RmsNorm::new(
                    &lp.sub("input_layernorm"),
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                ),
                mixer,
                post_norm: RmsNorm::new(
                    &lp.sub("post_attention_layernorm"),
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                ),
                mlp: Mlp::new(&lp.sub("mlp"), cfg),
            });
        }
        Ok(Self {
            embeddings,
            layers,
            final_norm: RmsNorm::new(&path.sub("norm"), cfg.hidden_size, cfg.rms_norm_eps),
            rope: RotaryEmbedding::new(
                cfg.rotary_dim(),
                cfg.rope_theta,
                cfg.max_position_embeddings,
                path.device(),
            )?,
            mask_policy,
            hidden_dim: cfg.hidden_size,
        })
    }

    /// The parameter name prefixes a source checkpoint carries that this encoder
    /// deliberately does not allocate — the LM head amputation. Drop every tensor
    /// under these prefixes when converting a generative checkpoint into the
    /// decision model's `VarStore` (extra tensors would otherwise fail the load).
    pub fn amputated_lm_head_names() -> &'static [&'static str] {
        &["lm_head.", "embed_out.", "mtp."]
    }
}

#[allow(clippy::assign_op_pattern)]
impl Backbone for Qwen35Encoder {
    fn forward(&self, token_ids: &Tensor) -> Result<Tensor, Error> {
        let size = token_ids.size();
        if size.len() != 2 {
            return Err(Error::Shape(format!(
                "token_ids must be [batch, seq], got {size:?}"
            )));
        }
        let seq = size[1];
        let mut hidden = token_ids.apply(&self.embeddings);
        let mask = self
            .mask_policy
            .additive_mask(seq, hidden.kind(), hidden.device())?;
        for layer in &self.layers {
            let normed = layer.input_norm.forward(&hidden);
            let mixed = match &layer.mixer {
                Mixer::Gdn(gdn) => gdn.forward(&normed)?,
                Mixer::Attn(attn) => attn.forward(&normed, &self.rope, &mask),
            };
            hidden = hidden + mixed;
            hidden = &hidden + layer.mlp.forward(&layer.post_norm.forward(&hidden));
        }
        Ok(self.final_norm.forward(&hidden))
    }

    fn hidden_dim(&self) -> i64 {
        self.hidden_dim
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn derived_layer_kinds_are_every_fourth_full_attention() {
        let mut cfg = Qwen35Config::test_tiny();
        cfg.num_hidden_layers = 8;
        let kinds = cfg.layer_kinds().unwrap();
        let full: Vec<usize> = kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == LayerKind::FullAttention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            full,
            vec![3, 7],
            "6 softmax layers per 24 at the 0.8B ratio"
        );
    }

    #[test]
    fn forward_shape_and_final_norm() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = Qwen35Config::test_tiny();
        let model =
            Qwen35Encoder::new(&vs.root().sub("model"), &cfg, MaskPolicy::Bidirectional).unwrap();
        let ids = Tensor::randint(255, [2, 32], (Kind::Int64, Device::Cpu));
        let out = model.forward(&ids).unwrap();
        assert_eq!(out.size(), [2, 32, cfg.hidden_size]);
        assert_eq!(model.hidden_dim(), cfg.hidden_size);
        assert!(
            out.isfinite().all().int64_value(&[]) == 1,
            "no NaN/Inf in hidden states"
        );
    }

    #[test]
    fn gdn_layers_stay_causal_under_the_bidirectional_conversion() {
        // The conversion pin: only softmax layers flip. Perturbing a future token
        // must not change any GDN-layer-dominated position before it — check the
        // strongest form available end-to-end: with a pure-GDN model (layers 1-3 of
        // the 4-layer tiny config feed the one softmax layer, so test positions
        // upstream of the FIRST full-attention layer via a 1-layer GDN model).
        let mut cfg = Qwen35Config::test_tiny();
        cfg.num_hidden_layers = 1;
        cfg.layer_types = vec!["linear_attention".to_owned()];
        let vs = nn::VarStore::new(Device::Cpu);
        let model =
            Qwen35Encoder::new(&vs.root().sub("model"), &cfg, MaskPolicy::Bidirectional).unwrap();
        let ids_a = Tensor::randint(255, [1, 24], (Kind::Int64, Device::Cpu));
        let mut ids_b = Vec::<i64>::try_from(ids_a.view(-1)).unwrap();
        let n = ids_b.len();
        ids_b[n - 1] = (ids_b[n - 1] + 1) % 255; // perturb only the last token
        let ids_b = Tensor::from_slice(&ids_b).view([1, 24]);
        let out_a = model.forward(&ids_a).unwrap();
        let out_b = model.forward(&ids_b).unwrap();
        let prefix_a = out_a.narrow(1, 0, 23);
        let prefix_b = out_b.narrow(1, 0, 23);
        let max_diff = (prefix_a - prefix_b).abs().max().double_value(&[]);
        assert_eq!(
            max_diff, 0.0,
            "GDN positions are invariant to future tokens"
        );
    }

    #[test]
    fn softmax_layers_attend_bidirectionally_after_conversion() {
        // A single bidirectional full-attention layer: changing the LAST token must
        // change EARLIER positions' outputs (impossible under a causal mask).
        let mut cfg = Qwen35Config::test_tiny();
        cfg.num_hidden_layers = 1;
        cfg.layer_types = vec!["full_attention".to_owned()];
        let vs = nn::VarStore::new(Device::Cpu);
        let model =
            Qwen35Encoder::new(&vs.root().sub("model"), &cfg, MaskPolicy::Bidirectional).unwrap();
        let ids_a = Tensor::randint(255, [1, 16], (Kind::Int64, Device::Cpu));
        let mut ids_b = Vec::<i64>::try_from(ids_a.view(-1)).unwrap();
        let n = ids_b.len();
        ids_b[n - 1] = (ids_b[n - 1] + 1) % 255;
        let ids_b = Tensor::from_slice(&ids_b).view([1, 16]);
        let out_a = model.forward(&ids_a).unwrap();
        let out_b = model.forward(&ids_b).unwrap();
        let early_diff = (out_a.get(0).get(0) - out_b.get(0).get(0))
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            early_diff > 0.0,
            "bidirectional layer: future tokens affect earlier states"
        );

        // Same layer under the causal policy: earlier positions must NOT change.
        let vs_c = nn::VarStore::new(Device::Cpu);
        let causal =
            Qwen35Encoder::new(&vs_c.root().sub("model"), &cfg, MaskPolicy::Causal).unwrap();
        let out_a = causal.forward(&ids_a).unwrap();
        let out_b = causal.forward(&ids_b).unwrap();
        let early_diff = (out_a.get(0).get(0) - out_b.get(0).get(0))
            .abs()
            .max()
            .double_value(&[]);
        assert_eq!(
            early_diff, 0.0,
            "causal policy leaves earlier states untouched"
        );
    }

    #[test]
    fn forward_backward_populates_grads() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = Qwen35Config::test_tiny();
        let model =
            Qwen35Encoder::new(&vs.root().sub("model"), &cfg, MaskPolicy::Bidirectional).unwrap();
        let ids = Tensor::randint(255, [1, 16], (Kind::Int64, Device::Cpu));
        let loss = model.forward(&ids).unwrap().sum(Kind::Float);
        loss.backward();
        let grads: usize = vs
            .variables()
            .iter()
            .filter(|(_, t)| t.grad().isfinite().any().int64_value(&[]) == 1)
            .count();
        assert!(grads > 0, "loss backprops into backbone weights");
    }
}
