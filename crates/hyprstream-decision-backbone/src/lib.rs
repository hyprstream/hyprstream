//! # hyprstream-decision-backbone — System One P1.2
//!
//! Backbone conversion for the System One decision model, plus the baseline arm.
//! Everything here composes the P1.1 pieces: the
//! [`DecisionEncoder`](hyprstream_decision_head::DecisionEncoder) produces token ids
//! plus anchor maps; a [`Backbone`] implementation turns token ids into hidden states;
//! the [`SharedScorer`](hyprstream_decision_head::SharedScorer) reads one hidden state
//! per anchor position. [`DecisionModel`] wires the three into a single
//! forward pass inside one `VarStore`.
//!
//! ## The conversion (LLM2Vec-style, partial — pinned by the plan)
//!
//! [`qwen35::Qwen35Encoder`] converts the Qwen3.5-0.8B hybrid decoder into an
//! encoder for decision scoring:
//!
//! - **Softmax (full-attention) layers go bidirectional.** LLM2Vec (BehnamGhader
//!   et al. 2024) showed a decoder's causal mask can be dropped for embedding/scoring
//!   tasks; here it is dropped only on the 6 softmax attention layers
//!   ([`MaskPolicy::Bidirectional`]), the cheap, partial precedent the plan costs.
//! - **The 18 GatedDeltaNet layers stay causal.** Linear attention has documented
//!   associative-recall deficits and no established bidirectional form; the chunked
//!   gated delta rule is inherently causal. Bidirectionality enters only through the
//!   interleaved softmax layers.
//! - **The LM head is amputated** — the encoder owns no `lm_head`/`embed_out`
//!   parameters; output is the final-norm hidden state the scorer consumes.
//! - **The shared scorer attaches** ([`DecisionModel`]) under its own `nn::Path`
//!   scope, so head and backbone share one checkpoint without name collisions.
//!
//! Anchor tokens are learned special tokens (P1.1 assigned the id); their embedding
//! rows start at the converted checkpoint's values and train with the model. GDN
//! causality plus adjacent-after anchor placement (S6b1) means an anchor state sees
//! its own option's rubric text — exactly the serialization P0.1a froze.
//!
//! ## Torch-native GDN path (S6c feasibility verdict)
//!
//! The GatedDeltaNet forward here is a pure-tensor-op implementation of the public
//! chunked gated delta rule (`torch_chunk_gated_delta_rule` algorithm) with an
//! autograd-safe within-chunk correction (out-of-place row stacking, no in-place
//! mutation of saved tensors), so the same code path trains. **fla Triton kernels
//! are NOT load-bearing**: they may only be adopted behind a Triton ≥ 3.7 gate
//! (gfx90a chunk-kernel crash, vllm#44973, fixed upstream). Fallback ranking
//! (S6c): native path → fla on Triton ≥ 3.7 → cloud MI300X rental → 5090-only
//! iteration.
//!
//! ## Dual-backend pin
//!
//! One source tree, two libtorch builds, zero code divergence: tch-rs links whichever
//! libtorch `LIBTORCH` points at, and both CUDA and ROCm builds surface devices
//! through tch's `Device::Cuda` API (ROCm masquerades as CUDA inside torch).
//! **CUDA build → RTX 5090 interactive dev; ROCm build → MI210 volume training**
//! (S6c compute floor: 8× MI210 as 2×4 xGMI groups). [`backend`] records the pin and
//! probes devices. No cross-group tensor parallelism (PCIe-only links); cross-group
//! training is data-parallel only.
//!
//! ## Training-framework / DDP decision (P1.2 deliverable)
//!
//! hyprstream has **no multi-GPU DDP today (no RCCL)** — S6c inventory flag. Decision:
//!
//! 1. **Single-process single-GPU tch-rs first** (this crate's training path, P1.4):
//!    gate iterations on 1–2M-example subsets fit within one MI210's 64 GB at 0.8B bf16
//!    with gradient checkpointing — no new framework to land the first corpus runs.
//! 2. **Data-parallel via libtorch distributed (NCCL/RCCL through tch's `distributed`
//!    bindings) when volume training starts**, within one 4-GPU xGMI group only;
//!    cross-group scaling is embarrassingly parallel (separate runs / corpus shards),
//!    matching the PCIe-only interconnect.
//! 3. Rejected: adding a Python framework dependency (breaks the single-fork tch-rs
//!    weight story and the Apache-2.0 boundary story); FSDP/TP (unneeded at 0.8B).
//!
//! ## Smoke test (P1.2 deliverable)
//!
//! `cargo run -p hyprstream-decision-backbone --bin hyprstream-backbone-smoke` runs
//! the pinned 4-step check: (1) `rocm-smi`/`nvidia-smi` device probe, (2) libtorch FFI
//! probe (device tensor alloc + kernel), (3) one forward/backward optimizer step at
//! 2k context on the converted architecture (miniature config) with step-time and
//! tokens/sec reported for the MFU sizing sanity check (S6c: 0.8B bf16, 30–40% MFU →
//! ~3.8–5.1 days per 10M×2k-token epoch on one group), (4) the optional fla leg —
//! reported as gated-skipped (Triton ≥ 3.7 gate; not load-bearing). Steps degrade
//! gracefully on GPU-less hosts (the current inventory flag: both reachable hosts
//! report no GPUs) so CI can run steps 1–3 on CPU.
//!
//! ## Baseline arm + pre-priced hybrid fallback
//!
//! [`modernbert::ModernBertEncoder`] implements the ModernBERT-large clean-slate
//! baseline arm behind the same [`Backbone`] trait, so P1.4's gate (b) compares both
//! arms on identical versioned splits with no harness divergence.
//!
//! The **hybrid fallback** (kill-switch, promoted if P1.4 leg (e) collapses) is
//! pre-priced here in design terms: the generalist scorer ships as-is; per-family
//! calibration anchoring is pure P2.2 post-hoc work (per-primitive temperature +
//! per-family parameters persisted as P0.3 rows) — no backbone change, no new
//! training loop, cost ≈ the P2.2 fitting pipeline run per pinned family plus one
//! calibration-param read at serving time. That is why "promote the fallback" is a
//! configuration event, not a re-architecture.

pub mod backbone;
pub mod backend;
pub mod error;
pub mod gdn;
pub mod model;
pub mod modernbert;
pub mod qwen35;

pub use backbone::{Backbone, MaskPolicy};
pub use error::Error;
pub use model::DecisionModel;
pub use modernbert::{ModernBertConfig, ModernBertEncoder};
pub use qwen35::{Qwen35Config, Qwen35Encoder};
