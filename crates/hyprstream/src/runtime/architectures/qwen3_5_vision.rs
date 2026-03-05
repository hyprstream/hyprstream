//! Qwen3.5 vision encoder
#![allow(dead_code, unused_variables, clippy::redundant_closure, clippy::needless_borrows_for_generic_args, clippy::redundant_closure_for_method_calls)]
//!
//! Architecture: Conv3d patch embedding → 27 transformer blocks (2D RoPE on Q/K) →
//! VisionPatchMerger (2×2 spatial merge) → out_hidden_size projection.
//!
//! Output: [total_image_patches, out_hidden_size]

use super::llama::{LlamaMLP, LinearProjection, RMSNorm};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tch::{Kind, Tensor};

// ============================================================================
// Vision config
// ============================================================================

pub struct Qwen3_5VisionConfig {
    pub depth: usize,              // 27
    pub hidden_size: usize,        // 1152
    pub intermediate_size: usize,  // 4304
    pub num_heads: usize,          // 16
    pub head_dim: usize,           // hidden_size / num_heads = 72
    pub patch_size: usize,         // 16
    pub temporal_patch_size: usize,// 2 (image-only: use 1)
    pub spatial_merge_size: usize, // 2
    pub out_hidden_size: usize,    // 3584 (9B) or 2048 (35B)
    pub rms_norm_eps: f32,         // 1e-6
}

impl Qwen3_5VisionConfig {
    pub fn from_json(json: &serde_json::Value) -> Self {
        let hidden_size = json["hidden_size"].as_u64().unwrap_or(1152) as usize;
        let num_heads = json["num_heads"].as_u64().unwrap_or(16) as usize;
        Self {
            depth: json["depth"].as_u64().unwrap_or(27) as usize,
            hidden_size,
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(4304) as usize,
            num_heads,
            head_dim: hidden_size / num_heads,
            patch_size: json["patch_size"].as_u64().unwrap_or(16) as usize,
            temporal_patch_size: json["temporal_patch_size"].as_u64().unwrap_or(2) as usize,
            spatial_merge_size: json["spatial_merge_size"].as_u64().unwrap_or(2) as usize,
            out_hidden_size: json["out_hidden_size"].as_u64().unwrap_or(3584) as usize,
            rms_norm_eps: 1e-6,
        }
    }
}

// ============================================================================
// Patch embedding (Conv3d)
// ============================================================================

struct VisionPatchEmbed {
    proj_weight: Tensor, // [hidden, 3, temporal_patch_size, patch_size, patch_size]
    proj_bias: Option<Tensor>,
    patch_size: usize,
    temporal_patch_size: usize,
}

impl VisionPatchEmbed {
    fn load(weights: &mut HashMap<String, Tensor>, prefix: &str, cfg: &Qwen3_5VisionConfig) -> Result<Self> {
        let pw = weights.remove(&format!("{prefix}.patch_embed.proj.weight"))
            .ok_or_else(|| anyhow!("Missing {prefix}.patch_embed.proj.weight"))?;
        let pb = weights.remove(&format!("{prefix}.patch_embed.proj.bias"));
        Ok(Self {
            proj_weight: pw,
            proj_bias: pb,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
        })
    }

    /// pixel_values: [B, C, H, W] for images
    /// Returns [B, num_patches, hidden]
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (batch, _c, h, w) = {
            let s = pixel_values.size();
            (s[0], s[1], s[2], s[3])
        };
        let ps = self.patch_size as i64;
        let tp = self.temporal_patch_size as i64;

        // Reshape to 5D: [B, C, T=1, H, W] for Conv3d
        let x5d = pixel_values.unsqueeze(2);

        // Conv3d: kernel = [tp, ps, ps], stride = same
        let out = x5d.conv3d(
            &self.proj_weight,
            self.proj_bias.as_ref(),
            &[tp, ps, ps],   // stride
            &[0i64, 0, 0],   // padding
            &[1i64, 1, 1],   // dilation
            1,               // groups
        );

        // out: [B, hidden, T_out, H_out, W_out]
        let out_shape = out.size();
        let hidden = out_shape[1];
        let total_patches = out_shape[2] * out_shape[3] * out_shape[4];

        // Flatten to [B, num_patches, hidden]
        let out = out.permute([0, 2, 3, 4, 1]).reshape([batch, total_patches, hidden]);
        Ok(out)
    }
}

unsafe impl Send for VisionPatchEmbed {}
unsafe impl Sync for VisionPatchEmbed {}

// ============================================================================
// 2D Rotary Embeddings for vision
// ============================================================================

/// Compute 2D RoPE (cos, sin) for a patch grid
/// Returns (cos, sin) each of shape [num_patches, head_dim]
fn vision_rope_2d(
    grid_h: i64,
    grid_w: i64,
    head_dim: i64,
    device: tch::Device,
) -> (Tensor, Tensor) {
    // half of head_dim goes to rows, half to cols
    let half = head_dim / 2;
    let inv_freq = {
        let pos = Tensor::arange(half / 2, (Kind::Float, device));
        let exp = &pos * 2.0 / (half as f64);
        Tensor::full([half / 2], 10000.0, (Kind::Float, device)).pow(&(-exp))
    };

    // Row positions: [grid_h, 1] × inv_freq → [grid_h, half/2]
    let rows = Tensor::arange(grid_h, (Kind::Float, device)).unsqueeze(1);
    let row_angles = rows.matmul(&inv_freq.unsqueeze(0)); // [grid_h, half/2]

    // Col positions: [grid_w, 1] × inv_freq → [grid_w, half/2]
    let cols = Tensor::arange(grid_w, (Kind::Float, device)).unsqueeze(1);
    let col_angles = cols.matmul(&inv_freq.unsqueeze(0)); // [grid_w, half/2]

    // For each patch (r, c), angle = cat(row_angle, col_angle) of total dim head_dim/2
    // Expand: [grid_h, grid_w, half/2] for each
    let row_exp = row_angles.unsqueeze(1).expand([grid_h, grid_w, half / 2], false);
    let col_exp = col_angles.unsqueeze(0).expand([grid_h, grid_w, half / 2], false);

    // Full angle for each patch: [grid_h*grid_w, half]
    let angles = Tensor::cat(&[
        row_exp.reshape([grid_h * grid_w, half / 2]),
        col_exp.reshape([grid_h * grid_w, half / 2]),
    ], 1);

    // Duplicate for cos/sin halves: [num_patches, head_dim]
    let angles_full = Tensor::cat(&[&angles, &angles], 1);
    let cos = angles_full.cos();
    let sin = angles_full.sin();
    (cos, sin)
}

/// Apply 2D RoPE to q or k: [seq, heads, head_dim]
fn apply_rope_2d(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    // x: [seq, heads, head_dim]
    // cos, sin: [seq, head_dim]
    let cos = cos.unsqueeze(1).expand_as(x); // [seq, heads, head_dim]
    let sin = sin.unsqueeze(1).expand_as(x);

    let hd = x.size()[2];
    let half = hd / 2;
    let x1 = x.narrow(2, 0, half);
    let x2 = x.narrow(2, half, half);
    let x1_rot = &x1 * &cos.narrow(2, 0, half) - &x2 * &sin.narrow(2, 0, half);
    let x2_rot = &x1 * &sin.narrow(2, half, half) + &x2 * &cos.narrow(2, half, half);
    Tensor::cat(&[x1_rot, x2_rot], 2)
}

// ============================================================================
// Vision attention block
// ============================================================================

struct VisionAttention {
    qkv: LinearProjection,  // fused [3*hidden, hidden]
    proj: LinearProjection, // [hidden, hidden]
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (seq, hidden) = {
            let s = x.size();
            (s[0], s[1])
        };
        let nh = self.num_heads as i64;
        let hd = self.head_dim as i64;

        let qkv = self.qkv.apply(x).reshape([seq, 3, nh, hd]);
        let q = qkv.select(1, 0); // [seq, nh, hd]
        let k = qkv.select(1, 1);
        let v = qkv.select(1, 2);

        // Apply 2D RoPE
        let q = apply_rope_2d(&q, cos, sin).to_kind(Kind::Float);
        let k = apply_rope_2d(&k, cos, sin).to_kind(Kind::Float);
        let v = v.to_kind(Kind::Float);

        // Attention [seq, nh, seq]
        let scale = 1.0 / (hd as f64).sqrt();
        let q_p = q.permute([1, 0, 2]); // [nh, seq, hd]
        let k_p = k.permute([1, 2, 0]); // [nh, hd, seq]
        let v_p = v.permute([1, 0, 2]); // [nh, seq, hd]

        let scores = q_p.matmul(&k_p) * scale; // [nh, seq, seq]
        let attn = scores.softmax(-1, Kind::Float);
        let ctx = attn.matmul(&v_p); // [nh, seq, hd]
        let ctx = ctx.permute([1, 0, 2]).contiguous().reshape([seq, hidden]);

        let out = self.proj.apply(&ctx.to_kind(x.kind()));
        Ok(out)
    }
}

unsafe impl Send for VisionAttention {}
unsafe impl Sync for VisionAttention {}

// ============================================================================
// Vision transformer block
// ============================================================================

struct VisionBlock {
    norm1: RMSNorm,
    attn: VisionAttention,
    norm2: RMSNorm,
    mlp: LlamaMLP,
}

impl VisionBlock {
    fn load(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        cfg: &Qwen3_5VisionConfig,
        block_idx: usize,
    ) -> Result<Self> {
        let norm1_w = weights.remove(&format!("{prefix}.norm1.weight"))
            .ok_or_else(|| anyhow!("Missing {prefix}.norm1.weight"))?;
        let norm2_w = weights.remove(&format!("{prefix}.norm2.weight"))
            .ok_or_else(|| anyhow!("Missing {prefix}.norm2.weight"))?;

        Ok(Self {
            norm1: RMSNorm { weight: norm1_w, eps: cfg.rms_norm_eps },
            attn: VisionAttention {
                qkv: LinearProjection::take(weights, &format!("{prefix}.attn.qkv.weight"))?,
                proj: LinearProjection::take(weights, &format!("{prefix}.attn.proj.weight"))?,
                num_heads: cfg.num_heads,
                head_dim: cfg.head_dim,
            },
            norm2: RMSNorm { weight: norm2_w, eps: cfg.rms_norm_eps },
            mlp: LlamaMLP {
                gate_proj: LinearProjection::take(weights, &format!("{prefix}.mlp.gate_proj.weight"))?,
                up_proj: LinearProjection::take(weights, &format!("{prefix}.mlp.up_proj.weight"))?,
                down_proj: LinearProjection::take(weights, &format!("{prefix}.mlp.down_proj.weight"))?,
                activation: "gelu_pytorch_tanh".to_owned(),
                layer_idx: block_idx,
            },
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let attn_out = self.attn.forward(&self.norm1.forward(x)?, cos, sin)?;
        let x = x + &attn_out;
        let mlp_out = self.mlp.forward(&self.norm2.forward(&x)?, None)?;
        Ok(&x + &mlp_out)
    }
}

unsafe impl Send for VisionBlock {}
unsafe impl Sync for VisionBlock {}

// ============================================================================
// Patch merger (spatial 2×2 → out_hidden_size)
// ============================================================================

struct VisionPatchMerger {
    ln_q: RMSNorm,
    mlp0: LinearProjection,  // visual.merger.mlp.0  [(hidden*merge^2), out_hidden_size]
    mlp2: LinearProjection,  // visual.merger.mlp.2  [out_hidden_size, out_hidden_size]
    spatial_merge_size: usize,
    hidden_size: usize,
    out_hidden_size: usize,
}

impl VisionPatchMerger {
    fn load(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        cfg: &Qwen3_5VisionConfig,
    ) -> Result<Self> {
        let ln_q_w = weights.remove(&format!("{prefix}.ln_q.weight"))
            .ok_or_else(|| anyhow!("Missing {prefix}.ln_q.weight"))?;

        Ok(Self {
            ln_q: RMSNorm { weight: ln_q_w, eps: cfg.rms_norm_eps },
            mlp0: LinearProjection::take(weights, &format!("{prefix}.mlp.0.weight"))?,
            mlp2: LinearProjection::take(weights, &format!("{prefix}.mlp.2.weight"))?,
            spatial_merge_size: cfg.spatial_merge_size,
            hidden_size: cfg.hidden_size,
            out_hidden_size: cfg.out_hidden_size,
        })
    }

    /// x: [num_patches, hidden_size], grid_hw = (H_patches, W_patches)
    /// Returns [num_out_patches, out_hidden_size]
    fn forward(&self, x: &Tensor, grid_h: i64, grid_w: i64) -> Result<Tensor> {
        let x = self.ln_q.forward(x)?;
        let s = self.spatial_merge_size as i64;
        let h = self.hidden_size as i64;

        // Reshape patches to [grid_h, grid_w, hidden], then merge s×s spatially
        let x = x.reshape([grid_h, grid_w, h]);

        // Only merge if grid dimensions are divisible by s
        let out_h = grid_h / s;
        let out_w = grid_w / s;

        // Reshape: [out_h, s, out_w, s, h] → [out_h*out_w, s*s*h]
        let merged = x.reshape([out_h, s, out_w, s, h])
            .permute([0, 2, 1, 3, 4])
            .reshape([out_h * out_w, s * s * h]);

        // Two-layer MLP with gelu activation
        let out = self.mlp0.apply(&merged).gelu("none");
        let out = self.mlp2.apply(&out);
        Ok(out)
    }
}

unsafe impl Send for VisionPatchMerger {}
unsafe impl Sync for VisionPatchMerger {}

// ============================================================================
// Full vision encoder
// ============================================================================

pub struct Qwen3_5VisionEncoder {
    config: Qwen3_5VisionConfig,
    patch_embed: VisionPatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: VisionPatchMerger,
}

impl Qwen3_5VisionEncoder {
    /// Load encoder from weight map. All vision weights are under the "visual." prefix.
    pub fn from_weights(
        weights: &mut HashMap<String, Tensor>,
        cfg: Qwen3_5VisionConfig,
    ) -> Result<Self> {
        let prefix = "visual";

        let patch_embed = VisionPatchEmbed::load(weights, prefix, &cfg)?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let block_prefix = format!("{prefix}.blocks.{i}");
            blocks.push(VisionBlock::load(weights, &block_prefix, &cfg, i)?);
        }

        let merger_prefix = format!("{prefix}.merger");
        let merger = VisionPatchMerger::load(weights, &merger_prefix, &cfg)?;

        Ok(Self { config: cfg, patch_embed, blocks, merger })
    }

    /// Encode pixel values to vision features.
    ///
    /// pixel_values: [B, C, H, W]
    /// Returns: [B * num_out_patches, out_hidden_size]
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let device = pixel_values.device();
        let dtype = pixel_values.kind();

        // Patch embedding: [B, num_patches, hidden]
        let patches = self.patch_embed.forward(pixel_values)?;

        let batch = patches.size()[0];
        let num_patches = patches.size()[1];

        // Grid dimensions (square images assumed)
        let ps = self.config.patch_size;
        let h_img = pixel_values.size()[2] as usize;
        let w_img = pixel_values.size()[3] as usize;
        let grid_h = (h_img / ps) as i64;
        let grid_w = (w_img / ps) as i64;

        // 2D RoPE for vision patches: [num_patches, head_dim]
        let (cos, sin) = vision_rope_2d(grid_h, grid_w, self.config.head_dim as i64, device);
        let cos = cos.to_kind(dtype);
        let sin = sin.to_kind(dtype);

        // Process each image separately through transformer blocks
        let mut all_outputs = Vec::with_capacity(batch as usize);

        for b in 0..batch {
            let x = patches.select(0, b); // [num_patches, hidden]
            let mut x = x;
            for block in &self.blocks {
                x = block.forward(&x, &cos, &sin)?;
            }
            // Merge spatially: [out_patches, out_hidden]
            let merged = self.merger.forward(&x, grid_h, grid_w)?;
            all_outputs.push(merged);
        }

        // Cat along batch: [B * out_patches, out_hidden]
        Ok(Tensor::cat(&all_outputs, 0))
    }

    /// Returns the output hidden size (dimension of encoder output)
    pub fn out_hidden_size(&self) -> usize {
        self.config.out_hidden_size
    }
}

unsafe impl Send for Qwen3_5VisionEncoder {}
unsafe impl Sync for Qwen3_5VisionEncoder {}
