//! ModernBERT-large baseline arm (S3 clean-slate comparison, P1.4 gate (b)).
//!
//! Encoder-only implementation of the ModernBERT architecture (Answer.AI 2024):
//! alternating local/global attention (local sliding window on 2 of every 3
//! layers), GeGLU MLP, pre-norm blocks, RoPE with separate thetas for local and
//! global layers, final norm, **no pooler and no MLM head** — the shared scorer
//! reads anchor states exactly as on the converted hybrid, so gate (b) compares
//! both arms through an identical harness on identical versioned splits.
//!
//! Parameter names follow the HF layout (`model.layers.{i}.attn.Wqkv.weight`, ...)
//! so a ModernBERT-large checkpoint loads after dropping its head tensors
//! ([`ModernBertEncoder::amputated_head_names`]). Checkpoint parity validation
//! against the reference implementation is a P1.4-time task (it needs the real
//! weights); the architecture here is the encoder-relevant subset only.

use serde::Deserialize;
use tch::nn::{self, Init};
use tch::{Kind, Tensor};

use crate::backbone::Backbone;
use crate::Error;

/// ModernBERT config (HF `config.json` shape, serde-compatible).
#[derive(Debug, Clone, Deserialize)]
pub struct ModernBertConfig {
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub vocab_size: i64,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: i64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    /// RoPE theta for global layers.
    #[serde(default = "default_global_theta")]
    pub global_rope_theta: f64,
    /// RoPE theta for local layers.
    #[serde(default = "default_local_theta")]
    pub local_rope_theta: f64,
    /// Local sliding-window size (per side) for local layers.
    #[serde(default = "default_local_window")]
    pub local_attention: i64,
    /// Every Nth layer is global attention.
    #[serde(default = "default_global_every")]
    pub global_attn_every_n_layers: i64,
}

fn default_max_positions() -> i64 {
    8192
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_global_theta() -> f64 {
    160_000.0
}
fn default_local_theta() -> f64 {
    10_000.0
}
fn default_local_window() -> i64 {
    128
}
fn default_global_every() -> i64 {
    3
}

impl ModernBertConfig {
    /// The ModernBERT-large reference shape (the S3 baseline arm).
    pub fn large() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 22,
            num_attention_heads: 12,
            intermediate_size: 1152,
            vocab_size: 50368,
            max_position_embeddings: 8192,
            norm_eps: 1e-5,
            global_rope_theta: 160_000.0,
            local_rope_theta: 10_000.0,
            local_attention: 128,
            global_attn_every_n_layers: 3,
        }
    }

    /// Miniature config for CPU tests.
    pub fn test_tiny() -> Self {
        Self {
            hidden_size: 48,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            intermediate_size: 96,
            vocab_size: 256,
            max_position_embeddings: 512,
            norm_eps: 1e-5,
            global_rope_theta: 10_000.0,
            local_rope_theta: 10_000.0,
            local_attention: 8,
            global_attn_every_n_layers: 3,
        }
    }

    fn head_dim(&self) -> i64 {
        self.hidden_size / self.num_attention_heads
    }

    fn is_global(&self, layer_idx: usize) -> bool {
        layer_idx.is_multiple_of(self.global_attn_every_n_layers as usize)
    }
}

/// LayerNorm with weight + bias (ModernBERT norms are biased, eps 1e-5).
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(path: &nn::Path, dim: i64, eps: f64) -> Self {
        Self {
            weight: path.var("weight", &[dim], Init::Const(1.0)),
            bias: path.var("bias", &[dim], Init::Const(0.0)),
            eps,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let dtype = x.kind();
        let x_f = x.to_kind(Kind::Float);
        let mean = x_f.mean_dim(&[-1i64][..], true, Kind::Float);
        let centered = &x_f - &mean;
        let var = (&centered * &centered).mean_dim(&[-1i64][..], true, Kind::Float);
        let normed = centered * (var + self.eps).rsqrt();
        (normed * self.weight.to_kind(Kind::Float) + self.bias.to_kind(Kind::Float)).to_kind(dtype)
    }
}

/// RoPE tables for one theta.
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(head_dim: i64, theta: f64, max_pos: i64, device: tch::Device) -> Self {
        let half = head_dim / 2;
        let inv_freq = (Tensor::arange(half, (Kind::Float, device)).neg()
            * (2.0 / head_dim as f64)
            * theta.ln())
        .exp();
        let t = Tensor::arange(max_pos, (Kind::Float, device));
        let freqs = t.unsqueeze(-1).matmul(&inv_freq.unsqueeze(0));
        let emb = Tensor::cat(&[&freqs, &freqs], -1);
        Self {
            cos: emb.cos(),
            sin: emb.sin(),
        }
    }

    /// Full-head rotation on `[batch, heads, seq, head_dim]`.
    fn apply(&self, x: &Tensor) -> Tensor {
        let size = x.size();
        let seq = size[2];
        let cos = self
            .cos
            .narrow(0, 0, seq)
            .view([1, 1, seq, size[3]])
            .to_kind(x.kind());
        let sin = self
            .sin
            .narrow(0, 0, seq)
            .view([1, 1, seq, size[3]])
            .to_kind(x.kind());
        let half = size[3] / 2;
        let x1 = x.narrow(-1, 0, half);
        let x2 = x.narrow(-1, half, half);
        let rotated = Tensor::cat(&[&x2.neg(), &x1], -1);
        x * &cos + rotated * &sin
    }
}

struct Attention {
    wqkv: nn::Linear,
    wo: nn::Linear,
    num_heads: i64,
    head_dim: i64,
}

impl Attention {
    fn forward(&self, hidden: &Tensor, rope: &Rotary, additive_mask: &Tensor) -> Tensor {
        let size = hidden.size();
        let (batch, seq, _) = (size[0], size[1], size[2]);
        let flat = hidden.reshape([batch * seq, size[2]]);
        let qkv = flat
            .apply(&self.wqkv)
            .view([batch, seq, 3, self.num_heads, self.head_dim])
            .permute([2, 0, 3, 1, 4]); // [3, B, H, T, d]
        let q = rope.apply(&qkv.get(0));
        let k = rope.apply(&qkv.get(1));
        let v = qkv.get(2);
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(-1, -2)).to_kind(Kind::Float) * scale
            + additive_mask.to_kind(Kind::Float);
        let probs = scores.softmax(-1, Kind::Float).to_kind(hidden.kind());
        probs
            .matmul(&v)
            .permute([0, 2, 1, 3])
            .contiguous()
            .view([batch * seq, self.num_heads * self.head_dim])
            .apply(&self.wo)
            .view([batch, seq, size[2]])
    }
}

/// GeGLU MLP: `Wi` projects to `2 * intermediate` (value | gate), gelu(gate) * value,
/// `Wo` projects back. Checkpoint fuses value/gate into one `Wi` weight.
struct GeGluMlp {
    wi: nn::Linear,
    wo: nn::Linear,
    intermediate: i64,
}

impl GeGluMlp {
    fn forward(&self, hidden: &Tensor) -> Tensor {
        let size = hidden.size();
        let flat = hidden.reshape([size[0] * size[1], size[2]]);
        let wi = flat.apply(&self.wi);
        let value = wi.narrow(-1, 0, self.intermediate);
        let gate = wi
            .narrow(-1, self.intermediate, self.intermediate)
            .gelu("none");
        (value * gate)
            .apply(&self.wo)
            .view([size[0], size[1], size[2]])
    }
}

struct Layer {
    attn_norm: Option<LayerNorm>, // first block has an identity norm in the reference
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: GeGluMlp,
    global: bool,
}

/// The ModernBERT encoder arm. Construct under a `nn::Path` scoped to the model
/// root (`embeddings`, `layers.{i}`, `final_norm`).
pub struct ModernBertEncoder {
    embeddings: nn::Embedding,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    rope_global: Rotary,
    rope_local: Rotary,
    local_window: i64,
    hidden_dim: i64,
}

impl ModernBertEncoder {
    pub fn new(path: &nn::Path, cfg: &ModernBertConfig) -> Result<Self, Error> {
        let head_dim = cfg.head_dim();
        if cfg.hidden_size % cfg.num_attention_heads != 0 || head_dim % 2 != 0 {
            return Err(Error::Config(format!(
                "hidden_size {} not divisible into even head dims over {} heads",
                cfg.hidden_size, cfg.num_attention_heads
            )));
        }
        let no_bias = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        let embeddings = nn::embedding(
            path.sub("embeddings").sub("tok_embeddings"),
            cfg.vocab_size,
            cfg.hidden_size,
            Default::default(),
        );
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers as usize {
            let lp = path.sub("layers").sub(i.to_string());
            layers.push(Layer {
                // The reference makes the FIRST layer's attention norm an identity;
                // mirror that so checkpoint tensors line up (it owns no such weight).
                attn_norm: if i == 0 {
                    None
                } else {
                    Some(LayerNorm::new(
                        &lp.sub("attn_norm"),
                        cfg.hidden_size,
                        cfg.norm_eps,
                    ))
                },
                attn: Attention {
                    wqkv: nn::linear(
                        lp.sub("attn").sub("Wqkv"),
                        cfg.hidden_size,
                        3 * cfg.hidden_size,
                        no_bias,
                    ),
                    wo: nn::linear(
                        lp.sub("attn").sub("Wo"),
                        cfg.hidden_size,
                        cfg.hidden_size,
                        no_bias,
                    ),
                    num_heads: cfg.num_attention_heads,
                    head_dim,
                },
                mlp_norm: LayerNorm::new(&lp.sub("mlp_norm"), cfg.hidden_size, cfg.norm_eps),
                mlp: GeGluMlp {
                    wi: nn::linear(
                        lp.sub("mlp").sub("Wi"),
                        cfg.hidden_size,
                        2 * cfg.intermediate_size,
                        no_bias,
                    ),
                    wo: nn::linear(
                        lp.sub("mlp").sub("Wo"),
                        cfg.intermediate_size,
                        cfg.hidden_size,
                        no_bias,
                    ),
                    intermediate: cfg.intermediate_size,
                },
                global: cfg.is_global(i),
            });
        }
        Ok(Self {
            embeddings,
            layers,
            final_norm: LayerNorm::new(&path.sub("final_norm"), cfg.hidden_size, cfg.norm_eps),
            rope_global: Rotary::new(
                head_dim,
                cfg.global_rope_theta,
                cfg.max_position_embeddings,
                path.device(),
            ),
            rope_local: Rotary::new(
                head_dim,
                cfg.local_rope_theta,
                cfg.max_position_embeddings,
                path.device(),
            ),
            local_window: cfg.local_attention,
            hidden_dim: cfg.hidden_size,
        })
    }

    /// Head prefixes a ModernBERT checkpoint carries that this arm never allocates
    /// (MLM decoder head + pooler) — drop them when converting the checkpoint.
    pub fn amputated_head_names() -> &'static [&'static str] {
        &["head.", "decoder.", "pooler."]
    }

    /// Additive `[1, 1, seq, seq]` mask: global layers attend everywhere (this arm
    /// is an encoder — bidirectional is its native mode); local layers add the
    /// sliding-window band.
    fn additive_mask(
        &self,
        seq: i64,
        global: bool,
        kind: Kind,
        device: tch::Device,
    ) -> Result<Tensor, Error> {
        let mask = Tensor::zeros([1, 1, seq, seq], (kind, device));
        if global {
            return Ok(mask);
        }
        let idx = Tensor::arange(seq, (Kind::Int64, device));
        let dist = idx.unsqueeze(0) - idx.unsqueeze(1); // [seq, seq] = q_pos - k_pos
        let outside = dist.abs().f_greater(self.local_window)?;
        Ok(mask.f_masked_fill(&outside.view([1, 1, seq, seq]), f64::NEG_INFINITY)?)
    }
}

impl Backbone for ModernBertEncoder {
    fn forward(&self, token_ids: &Tensor) -> Result<Tensor, Error> {
        let size = token_ids.size();
        if size.len() != 2 {
            return Err(Error::Shape(format!(
                "token_ids must be [batch, seq], got {size:?}"
            )));
        }
        let seq = size[1];
        let mut hidden = token_ids.apply(&self.embeddings);
        for layer in &self.layers {
            let normed = match &layer.attn_norm {
                Some(norm) => norm.forward(&hidden),
                None => hidden.shallow_clone(),
            };
            let rope = if layer.global {
                &self.rope_global
            } else {
                &self.rope_local
            };
            let mask = self.additive_mask(seq, layer.global, hidden.kind(), hidden.device())?;
            hidden = &hidden + layer.attn.forward(&normed, rope, &mask);
            hidden = &hidden + layer.mlp.forward(&layer.mlp_norm.forward(&hidden));
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
    use tch::Device;

    #[test]
    fn large_config_is_the_s3_reference_shape() {
        let cfg = ModernBertConfig::large();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 22);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn forward_shape_and_finiteness() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = ModernBertConfig::test_tiny();
        let model = ModernBertEncoder::new(&vs.root().sub("model"), &cfg).unwrap();
        let ids = Tensor::randint(255, [2, 20], (Kind::Int64, Device::Cpu));
        let out = model.forward(&ids).unwrap();
        assert_eq!(out.size(), [2, 20, cfg.hidden_size]);
        assert!(out.isfinite().all().int64_value(&[]) == 1);
    }

    #[test]
    fn local_attention_masks_outside_the_window() {
        // Layer 0 is global by design (HF ModernBERT: layer_idx % every == 0, and
        // 0 % n == 0), so window invariance is checked on the attention sublayer
        // itself: position 0 must not see a token 19 positions away with window 8.
        let cfg = ModernBertConfig::test_tiny();
        let vs = nn::VarStore::new(Device::Cpu);
        let enc = ModernBertEncoder::new(&vs.root().sub("model"), &cfg).unwrap();
        let seq = 20i64;
        let mask = enc
            .additive_mask(seq, false, Kind::Float, Device::Cpu)
            .unwrap();
        assert_eq!(mask.double_value(&[0, 0, 0, 19]), f64::NEG_INFINITY);
        assert_eq!(mask.double_value(&[0, 0, 0, 9]), f64::NEG_INFINITY);
        assert_eq!(mask.double_value(&[0, 0, 0, 8]), 0.0);

        // The local layer (index 1 in the tiny config) applied directly.
        let local = &enc.layers[1];
        assert!(!local.global, "layer 1 is local in the tiny config");
        let hidden_a = Tensor::randn([1, seq, cfg.hidden_size], (Kind::Float, Device::Cpu));
        let mut b = Vec::<f64>::try_from(hidden_a.view(-1)).unwrap();
        let width = cfg.hidden_size as usize;
        let n = b.len();
        b[n - width] += 1.0; // perturb the last token only
        let hidden_b = Tensor::from_slice(&b)
            .to_kind(Kind::Float)
            .view([1, seq, cfg.hidden_size]);
        let rope = &enc.rope_local;
        let out_a = local.attn.forward(&hidden_a, rope, &mask);
        let out_b = local.attn.forward(&hidden_b, rope, &mask);
        for pos in 0..(seq - cfg.local_attention - 1) {
            let diff = (out_a.get(0).get(pos) - out_b.get(0).get(pos))
                .abs()
                .max()
                .double_value(&[]);
            assert_eq!(
                diff, 0.0,
                "position {pos} never sees a token outside its window"
            );
        }
        // A global layer's mask is all-zero (encoder: natively bidirectional).
        let global_mask = enc
            .additive_mask(seq, true, Kind::Float, Device::Cpu)
            .unwrap();
        assert_eq!(global_mask.abs().max().double_value(&[]), 0.0);
    }

    #[test]
    fn forward_backward_populates_grads() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = ModernBertConfig::test_tiny();
        let model = ModernBertEncoder::new(&vs.root().sub("model"), &cfg).unwrap();
        let ids = Tensor::randint(255, [1, 12], (Kind::Int64, Device::Cpu));
        let loss = model.forward(&ids).unwrap().sum(Kind::Float);
        loss.backward();
        let grads = vs
            .variables()
            .iter()
            .filter(|(_, t)| t.grad().isfinite().any().int64_value(&[]) == 1)
            .count();
        assert!(grads > 0);
    }
}
