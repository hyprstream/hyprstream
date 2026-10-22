//! Torch inference engine and training pipeline for hyprstream.
//!
//! Carved out of the `hyprstream` application crate (Wave B decomposition) so
//! that libtorch links only into binaries that actually run inference or
//! training. Everything here lives behind the engine trait
//! [`runtime::RuntimeEngine`], whose surface carries only
//! `hyprstream-rpc-std` types — no `tch` types leak across the crate boundary,
//! and callers below this crate consume the engine through that trait.
//!
//! - `runtime` — PyTorch engine (`torch_engine`), model architectures, model
//!   factory, KV cache, sampling, chat templates, multi-GPU device pool
//! - `training` — Test-Time Training (TTT), per-tenant LoRA deltas
//!   (`tenant_delta`, `delta_pool`), merge strategies, quality filter,
//!   checkpoints
//! - `config` — the inference-facing configuration surface (`RuntimeConfig`,
//!   `ModelConfig`, generation/training configs); re-exported by the main
//!   crate's `config` module so existing paths keep resolving
//! - `worktree_ext` — 9P worktree-client convenience helpers and the
//!   COW-aware checkpoint copy used by the training pipeline
//!
//! `tch` tensors are `!Send`; engine services that hold them spawn through the
//! `hyprstream-service` Thread spawner (the existing `Spawnable` !Send impl
//! path). This crate never forces `Send` across the inference boundary.
//!
//! License: AGPL-3.0-only. This crate is carved from the AGPL `hyprstream`
//! application and is part of its combined AGPL distribution.

pub mod config;
pub mod runtime;
pub mod training;
pub mod worktree_ext;
