# Qwen3.5 Architecture Implementation

## Overview

This document describes the implementation plan for Qwen3.5 model support in hyprstream. Qwen3.5 is a hybrid SSM/attention architecture from Alibaba that interleaves **Gated DeltaNet** (linear attention / SSM) and standard **full attention** layers.

Two model families are supported under a single implementation:

| Model | Type | Layers | Experts | Vision |
|-------|------|--------|---------|--------|
| Qwen3.5-9B | Dense | 32 | — | Optional |
| Qwen3.5-35B-A3B | MoE (35B total, 3B active) | 40 | 256 (8/tok) | Yes |

---

## Architecture

### Hybrid Layer Pattern

Every 4th layer is `full_attention`; the remaining 3 are `linear_attention` (GatedDeltaNet). This is encoded in the `layer_types` array in the config.

```
Layer 0: linear_attention  (GatedDeltaNet)
Layer 1: linear_attention
Layer 2: linear_attention
Layer 3: full_attention    (standard GQA)
Layer 4: linear_attention
...
```

### GatedDeltaNet (Linear Attention Layers)

A recurrent SSM with a delta rule update. Key parameters (from `text_config`):

| Field | Value |
|-------|-------|
| `linear_conv_kernel_dim` | 4 (depthwise conv kernel size) |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 32 |

**State tensors** (per layer, interior-mutable via `Arc<parking_lot::Mutex<...>>`):
- `conv_state`: `[batch, conv_dim, kernel_size]` — ring buffer of last K input frames
- `recurrent_state`: `[batch, num_v_heads, head_k_dim, head_v_dim]` — delta rule accumulator

**Forward pass (decode, seq_len=1)**:
1. Project input: `in_proj_qkv` → mixed QKV of dim `key_dim*2 + value_dim`; `in_proj_z` → gate; `in_proj_a` → decay param; `in_proj_b` → beta
2. Depthwise conv1d (causal, kernel_size=4) on mixed QKV
3. Split mixed QKV → Q `[b, num_k_heads, head_k_dim]`, K, V
4. L2-normalize Q and K (NOT RMS norm); scale Q by `1/sqrt(head_k_dim)`
5. Compute decay `g = -A_log.exp() * softplus(a + dt_bias)` in f32
6. Recurrent delta rule update:
   ```
   state = state * exp(g)
   kv_mem = einsum(state, k) → [b, num_v_heads, head_v_dim]
   delta = (v - kv_mem) * sigmoid(b)
   state = state + outer(k, delta)
   out = einsum(state, q)
   ```
7. `RMSNormGated(out, z)` — `norm(out) * weight * silu(z)`
8. `out_proj`

**Forward pass (prefill, seq_len>1)**:
Chunked algorithm (chunk_size=64), translating `torch_chunk_gated_delta_rule`. All intermediate computation in f32. Stores final recurrent and conv states for subsequent decode.

**L2 normalization** (NOT RMSNorm):
```rust
fn l2_normalize(x: &Tensor) -> Tensor {
    let norm_sq = (x * x).sum_dim_intlist(&[-1i64][..], true, None);
    x / (norm_sq + 1e-6f64).sqrt()
}
```

### Full Attention Layers

Standard GQA with an output gate and partial RoPE.

| Field | Value (9B) |
|-------|-----------|
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 |
| `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 → `rotary_dim = 64` |

**Q projection quirk**: `q_proj` outputs `heads * head_dim * 2`. The second half is the output gate:
```rust
// q_proj output: [b, seq, heads * head_dim * 2]
let qg = q_proj_out.reshape([b, seq, heads, head_dim * 2]);
let (query, gate) = qg.chunk(2, -1);  // each [b, seq, heads, head_dim]
```

**Per-head QK-norm** via `Qwen3_5RMSNorm` (weight initialized to 0, formula `norm(x_f32) * (1 + weight)`).

**Partial RoPE**: apply RoPE only to first `rotary_dim=64` dims; passthrough remaining `head_dim - rotary_dim = 192` dims.

**Output gate**: `attn_output * gate.sigmoid()` before `o_proj`.

### Normalization Variants

Three distinct norm types in Qwen3.5:

| Type | Formula | Used For |
|------|---------|---------|
| `Qwen3_5RMSNorm` | `norm(x_f32) * (1 + weight)` | Layer norms, QK-norm |
| `RMSNormGated` | `norm(x) * weight * silu(gate)` | GatedDeltaNet output |
| L2 normalize | `x / sqrt(sum(x²) + eps)` | Q/K before delta rule |

### MLP (Dense vs MoE)

**Dense** (`model_type = "qwen3_5_text"`): standard SwiGLU MLP — reuses `LlamaMLP` (made `pub(crate)`).

**MoE** (`model_type = "qwen3_5_moe"`): sparse routing MoE for ALL layer FFNs.

| MoE Field | Value (35B) |
|-----------|------------|
| `num_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 512 (per expert) |
| `shared_expert_intermediate_size` | 512 |

Router: linear `hidden → num_experts`, softmax + top-k, normalize weights. Dispatch to selected experts, sum weighted outputs. Add always-active shared expert.

Polymorphic MLP:
```rust
enum Qwen3_5Mlp {
    Dense(LlamaMLP),
    Sparse(Qwen3_5SparseMoE),
}
```

---

## Config Parsing

The config JSON uses a **nested structure** — text parameters live under `text_config`, not at the root:

```json
{
  "model_type": "qwen3_5",
  "text_config": {
    "hidden_size": 4096,
    "rope_parameters": {
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25
    },
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]
  },
  "vision_config": { ... }
}
```

Detection in `detect_architecture_from_json()`:
- `model_type == "qwen3_5"` or `"qwen3_5_text"` → `ModelArchitecture::Qwen3_5`
- `model_type == "qwen3_5_moe"` → `ModelArchitecture::Qwen3_5` (with `is_moe=true`)
- architectures array containing `"Qwen3_5ForConditionalGeneration"` or `"Qwen3_5MoeForConditionalGeneration"`

Weight-based detection (fallback): `model.layers.0.linear_attn.in_proj_qkv.weight` present.

New `ModelConfig` fields:
```rust
pub partial_rotary_factor: Option<f32>,
pub layer_types: Vec<String>,
pub linear_conv_kernel_dim: usize,
pub linear_key_head_dim: usize,
pub linear_value_head_dim: usize,
pub linear_num_key_heads: usize,
pub linear_num_value_heads: usize,
pub is_moe: bool,
pub num_experts: usize,
pub num_experts_per_tok: usize,
pub moe_intermediate_size: usize,
pub shared_expert_intermediate_size: usize,
pub vision_out_hidden_size: usize,
```

---

## Vision / Multimodal

### Vision Encoder (`qwen3_5_vision.rs`)

Conv3d patch embedding → 27 transformer blocks with 2D rotary embeddings → spatial merge → linear projection.

| Config Field | Value |
|-------------|-------|
| `depth` | 27 blocks |
| `hidden_size` | 1152 |
| `num_heads` | 16 (head_dim = 72) |
| `patch_size` | 16 |
| `spatial_merge_size` | 2 (2×2 merge) |
| `out_hidden_size` | 3584 (9B) / 2048 (35B) |

**Components**:
- `VisionPatchEmbed`: Conv3d `[hidden, 3, temporal_patch_size, patch_size, patch_size]`, flatten → `[B, num_patches, hidden]`
- `VisionAttention`: fused QKV projection + 2D RoPE on Q/K
- `VisionMLP`: reuses `LlamaMLP` (pub(crate))
- `VisionBlock`: pre-norm transformer (norm1 → attn → norm2 → mlp, residual)
- `VisionPatchMerger`: `ln_q` norm → spatial 2×2 reshape → linear `(hidden * merge²) → out_hidden_size`

Output: `[total_patches, out_hidden_size]`

### Language–Vision Integration

Optional `vision_encoder: Option<Qwen3_5VisionEncoder>` and `vision_projector: Option<LinearProjection>` fields on `Qwen3_5Model`.

Pipeline:
1. `vision_encoder.forward(pixel_values, grid_thw)` → `[patches, out_hidden_size]`
2. `vision_projector.apply(features)` → `[patches, hidden_size]`
3. Replace image placeholder tokens in text embedding sequence with vision embeddings
4. Pass merged embeddings through text backbone via `forward_from_embeddings()`

Vision detection: `visual.patch_embed.proj.weight` key present in weights.

### mRoPE (Multimodal RoPE)

For image+text inputs, the 35B-A3B model uses mRoPE with sections `[11, 11, 10]` (temporal, height, width) over `rotary_dim=64` dimensions. For text-only inference, standard sequential position IDs remain valid. mRoPE position ID generation is deferred to a second implementation pass.

---

## Weight Key Reference

### Text Backbone

```
model.embed_tokens.weight
model.norm.weight
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight
model.layers.N.mlp.{gate,up,down}_proj.weight          (dense)
model.layers.N.mlp.gate.weight                         (MoE router)
model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight (MoE experts)
model.layers.N.mlp.shared_expert.{gate,up,down}_proj.weight
lm_head.weight
```

### Linear Attention Layers

```
model.layers.N.linear_attn.in_proj_qkv.weight
model.layers.N.linear_attn.in_proj_z.weight
model.layers.N.linear_attn.in_proj_b.weight
model.layers.N.linear_attn.in_proj_a.weight
model.layers.N.linear_attn.conv1d.weight               [conv_dim, 1, kernel_size]
model.layers.N.linear_attn.A_log                       [num_v_heads]
model.layers.N.linear_attn.dt_bias                     [num_v_heads]
model.layers.N.linear_attn.norm.weight                 [head_v_dim]
model.layers.N.linear_attn.out_proj.weight
```

### Full Attention Layers

```
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.self_attn.q_norm.weight                 [head_dim]
model.layers.N.self_attn.k_norm.weight                 [head_dim]
```

### Vision Encoder

```
visual.patch_embed.proj.weight                         [hidden, 3, tp, p, p]
visual.blocks.N.norm1.weight
visual.blocks.N.norm2.weight
visual.blocks.N.attn.qkv.weight                        [3*hidden, hidden]
visual.blocks.N.attn.proj.weight
visual.blocks.N.mlp.gate_proj.weight
visual.blocks.N.mlp.up_proj.weight
visual.blocks.N.mlp.down_proj.weight
visual.merger.ln_q.weight
visual.merger.mlp.0.weight                             [(hidden*merge²), out_hidden_size]
visual.merger.mlp.2.weight                             [out_hidden_size, out_hidden_size]
```

Weights with prefix `mtp.*` are silently skipped.

---

## Files Changed

| File | Change |
|------|--------|
| `crates/hyprstream/src/runtime/architectures/llama.rs` | Make `LinearProjection`, `LlamaMLP`, `RMSNorm` `pub(crate)` |
| `crates/hyprstream/src/runtime/architectures/mod.rs` | Add `pub mod qwen3_5`, `pub mod qwen3_5_vision`; add `Qwen3_5` enum variant; tokenizer routing |
| `crates/hyprstream/src/runtime/model_config.rs` | Add `Qwen3_5` variant; nested `text_config` parsing; new fields |
| `crates/hyprstream/src/runtime/model_factory.rs` | Add `Qwen3_5` arm → `create_qwen3_5_model()` |
| `crates/hyprstream/src/runtime/architectures/qwen3_5.rs` | **New**: full hybrid SSM/attention model |
| `crates/hyprstream/src/runtime/architectures/qwen3_5_vision.rs` | **New**: vision encoder pipeline |

---

## Critical Implementation Notes

1. **SSM state mutability**: `forward_with_cache` takes `&self`. Conv and recurrent states use `Arc<parking_lot::Mutex<Vec<Option<Tensor>>>>`.

2. **RoPE computation always in FP32**: cast to model dtype after `sin`/`cos`. Matches HuggingFace/vLLM behavior. See `rope.rs`.

3. **Conv1d depthwise**: weight shape `[conv_dim, 1, kernel_size]`, groups = conv_dim:
   ```rust
   x.conv1d(&weight, None::<&Tensor>, &[1i64], &[0i64], &[1i64], conv_dim as i64)
   ```

4. **L2 normalize, not RMSNorm**: Q and K in GatedDeltaNet use L2 norm before delta rule.

5. **QK-norm formula**: `Qwen3_5RMSNorm` uses weight initialized to 0 → formula is `norm(x_f32) * (1 + weight)`, not `norm(x_f32) * weight`.

6. **Partial RoPE**: `rotary_dim = (head_dim * partial_rotary_factor) as usize`. Only the first `rotary_dim` dimensions are rotated; remaining dims pass through unchanged.

7. **Q gate split**: `q_proj` output dim is `num_heads * head_dim * 2`. Reshape and split along last dim to get query and gate tensors.

8. **MoE routing normalization**: top-k weights are normalized post-softmax (divide by sum of selected weights). Shared expert output is added unconditionally.

---

## Verification

```bash
# 1. Build
cargo check -p hyprstream
cargo clippy -p hyprstream

# 2. AppImage
./appimage/build-appimage.sh build cpu

# 3. Install services
./appimage/output/hyprstream-dev-cpu-x86_64.AppImage service install --start
journalctl --user -u hyprstream-inference -f

# 4. Register and infer
hyprstream quick clone <local-or-remote-path> --name qwen3-5
hyprstream quick infer qwen3-5 --prompt "Hello, world"
```

Expected: model loads without panic, at least one token generated, `layer_types` length matches `num_hidden_layers`.
