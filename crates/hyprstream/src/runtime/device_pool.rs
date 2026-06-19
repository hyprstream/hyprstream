//! Multi-GPU device abstraction (`DevicePool`).
//!
//! This module introduces the device-strategy seam for the multi-GPU epic (#310,
//! part of #313). It resolves a set of configured device indices
//! (`RuntimeConfig.devices`) into concrete [`tch::Device`] values and exposes a
//! small, stable API that the later phases build on:
//!
//! - **2a replication** (#313): N `TorchEngine`s, one per local GPU, requests
//!   routed across [`DevicePool::devices`].
//! - **2b intra-host pipeline** (#314): a layer→device `device_map` layered on
//!   top of the same pool (see [`DevicePool`] note on extensibility).
//!
//! # Send / !Send boundary
//!
//! `tch-rs` tensors are `!Send`: a `Tensor` must never cross a thread boundary,
//! and engines that hold tensors stay pinned to one thread (as
//! `services/inference.rs` already does — one thread per engine). However, a
//! *single* thread may legitimately own tensors on `Cuda(0)` **and** `Cuda(1)`
//! at once: libtorch dispatches each op to the correct device via a
//! `DeviceGuard`, so multi-device-on-one-thread is fine; only multi-*thread*
//! tensor moves are forbidden.
//!
//! `DevicePool` deliberately holds **only** device indices / [`tch::Device`]
//! values (which are `Copy` and contain no tensor state). It is therefore
//! `Send + Sync` and can be shared across threads (e.g. handed to each
//! engine-owning thread so each can construct its own thread-pinned engine).
//! **Do not** add a `Tensor`, `VarStore`, or any `!Send` field to this type — it
//! would silently make the pool `!Send` and break the replication design.

use anyhow::{anyhow, Result};
use tch::Device;

/// A resolved, validated set of compute devices a single inference service owns.
///
/// Constructed from [`crate::config::RuntimeConfig`] via [`DevicePool::from_config`].
/// The pool performs **fail-fast** device resolution: when devices are explicitly
/// requested it never silently downgrades to CPU (see the plan's
/// "no-fragile-fallbacks", `docs/plans/2026-06-18-multi-gpu-multi-host-inference-spike.md`
/// §B / rust-M2). A process told to use GPU 1 that would land on CPU is an error,
/// not a warning.
///
/// # Send / Sync
///
/// `DevicePool` is `Send + Sync` because it holds only [`Device`] values (no
/// tensors). See the module-level docs for the `!Send` boundary — keep this type
/// tensor-free.
///
/// # Extensibility (#314)
///
/// The pipeline phase needs a layer→device map. That belongs *on* this type
/// (e.g. a `fn device_for_layer(&self, layer_idx, num_layers) -> Device` or an
/// explicit `device_map`), reusing the already-resolved [`Self::devices`]. New
/// strategy methods can be added without changing the existing accessors below,
/// so current callers (which only need [`Self::primary`]) keep working.
#[derive(Debug, Clone)]
pub struct DevicePool {
    /// Resolved devices, in the order they were requested. Always non-empty.
    devices: Vec<Device>,
}

impl DevicePool {
    /// Build a pool for the given GPU indices, validating availability fail-fast.
    ///
    /// `indices` must be non-empty and free of duplicates (the caller —
    /// [`crate::config::RuntimeConfig::resolve_device_indices`] — guarantees
    /// this, but it is re-checked here so the type's invariant holds regardless
    /// of construction path).
    ///
    /// Every index is validated against the live CUDA/HIP device count. If CUDA
    /// is unavailable, or any requested index is out of range, this returns an
    /// error rather than degrading to CPU — devices were *explicitly* requested.
    ///
    /// Note: ROCm presents as [`Device::Cuda`] via HIP, so this path is
    /// vendor-agnostic and contains no vendor-special casing.
    pub fn from_cuda_indices(indices: &[usize]) -> Result<Self> {
        if indices.is_empty() {
            return Err(anyhow!(
                "DevicePool requires at least one device index (got none)"
            ));
        }

        // Reject duplicates defensively (config layer also enforces this).
        let mut seen = std::collections::HashSet::with_capacity(indices.len());
        for &idx in indices {
            if !seen.insert(idx) {
                return Err(anyhow!(
                    "DevicePool: duplicate device index {idx} in requested set {indices:?}"
                ));
            }
        }

        // Fail-fast: devices were explicitly requested, so a CPU downgrade is an
        // error, not a fallback (no `Device::cuda_if_available()` here).
        if !tch::Cuda::is_available() {
            return Err(anyhow!(
                "DevicePool: GPU devices {indices:?} requested but no CUDA/ROCm device is \
                 available (refusing to silently fall back to CPU)"
            ));
        }

        let available = tch::Cuda::device_count();
        if available < 0 {
            return Err(anyhow!(
                "DevicePool: CUDA reported a negative device count ({available})"
            ));
        }
        let available = available as usize;

        let devices = indices
            .iter()
            .map(|&idx| {
                if idx >= available {
                    Err(anyhow!(
                        "DevicePool: GPU device index {idx} is out of range \
                         (only {available} CUDA/ROCm device(s) present)"
                    ))
                } else {
                    Ok(Device::Cuda(idx))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { devices })
    }

    /// Construct a pool directly from already-resolved [`Device`] values.
    ///
    /// Intended for tests and for callers that have an explicit device list
    /// (e.g. CPU-only setups). Validates only the structural invariants
    /// (non-empty, no duplicates); it does **not** probe CUDA, so prefer
    /// [`Self::from_cuda_indices`] / [`Self::from_config`] for GPU resolution.
    pub fn from_devices(devices: Vec<Device>) -> Result<Self> {
        if devices.is_empty() {
            return Err(anyhow!("DevicePool requires at least one device (got none)"));
        }
        let mut seen = std::collections::HashSet::with_capacity(devices.len());
        for dev in &devices {
            if !seen.insert(*dev) {
                return Err(anyhow!("DevicePool: duplicate device {dev:?} in {devices:?}"));
            }
        }
        Ok(Self { devices })
    }

    /// Resolve a [`DevicePool`] from runtime configuration.
    ///
    /// Behavior:
    /// - `use_gpu == false` → a single-entry CPU pool ([`Device::Cpu`]).
    /// - `use_gpu == true` → resolve the configured indices
    ///   ([`crate::config::RuntimeConfig::resolve_device_indices`], which honors
    ///   `devices` then falls back to the legacy single `gpu_device_id`) and
    ///   validate them fail-fast via [`Self::from_cuda_indices`].
    ///
    /// When `use_gpu` is true but no index is configured, this falls back to
    /// `cuda_if_available()`-style auto-detection of device 0 (preserving the
    /// pre-existing single-GPU default). That auto-detect path is the *only*
    /// place a CPU result is allowed when GPU is on, and only because nothing
    /// was explicitly requested.
    pub fn from_config(config: &crate::config::RuntimeConfig) -> Result<Self> {
        if !config.use_gpu {
            return Self::from_devices(vec![Device::Cpu]);
        }

        match config.resolve_device_indices()? {
            // Explicit device(s) requested → strict, fail-fast resolution.
            Some(indices) => Self::from_cuda_indices(&indices),
            // Nothing explicitly requested → preserve legacy auto-detect (may
            // be CPU if no GPU is present; this is NOT a silent downgrade because
            // the user requested no specific device).
            None => Self::from_devices(vec![Device::cuda_if_available()]),
        }
    }

    /// All resolved devices, in requested order. Always non-empty.
    ///
    /// The primary seam for #313/2a replication (one engine per device).
    #[must_use]
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// The primary device — the first configured device.
    ///
    /// Single-device callers (e.g. today's `TorchEngine`) use this to keep
    /// existing behavior while the multi-engine routing lands in a later PR.
    #[must_use]
    pub fn primary(&self) -> Device {
        self.devices[0]
    }

    /// Number of devices in the pool (always >= 1).
    ///
    /// No `is_empty()` companion exists: a `DevicePool` is never empty by
    /// construction. Use [`Self::is_single`] for the meaningful predicate.
    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    /// Whether the pool holds a single device. (`is_empty` is meaningless here —
    /// the pool is never empty — so this single-device predicate is provided
    /// instead; clippy's `len`/`is_empty` lint is allowed below.)
    #[must_use]
    pub fn is_single(&self) -> bool {
        self.devices.len() == 1
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn from_devices_rejects_empty() {
        let err = DevicePool::from_devices(vec![]).unwrap_err();
        assert!(err.to_string().contains("at least one device"));
    }

    #[test]
    fn from_devices_rejects_duplicates() {
        let err = DevicePool::from_devices(vec![Device::Cpu, Device::Cpu]).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn from_devices_preserves_order_and_primary() {
        let pool =
            DevicePool::from_devices(vec![Device::Cuda(2), Device::Cuda(0)]).unwrap();
        assert_eq!(pool.devices(), &[Device::Cuda(2), Device::Cuda(0)]);
        assert_eq!(pool.primary(), Device::Cuda(2));
        assert_eq!(pool.len(), 2);
        assert!(!pool.is_single());
    }

    #[test]
    fn from_devices_single() {
        let pool = DevicePool::from_devices(vec![Device::Cpu]).unwrap();
        assert_eq!(pool.primary(), Device::Cpu);
        assert!(pool.is_single());
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn from_cuda_indices_rejects_empty() {
        let err = DevicePool::from_cuda_indices(&[]).unwrap_err();
        assert!(err.to_string().contains("at least one device index"));
    }

    #[test]
    fn from_cuda_indices_rejects_duplicates_before_probing() {
        // Duplicate detection must happen before any CUDA probe so it is
        // deterministic on CPU-only CI.
        let err = DevicePool::from_cuda_indices(&[1, 1]).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn from_cuda_indices_no_gpu_is_fail_fast() {
        // On CPU-only CI, requesting an explicit GPU index must error rather
        // than degrade to CPU. (If CUDA happens to be present, this asserts the
        // out-of-range guard instead via a deliberately huge index.)
        if tch::Cuda::is_available() {
            let err = DevicePool::from_cuda_indices(&[usize::MAX]).unwrap_err();
            assert!(err.to_string().contains("out of range"));
        } else {
            let err = DevicePool::from_cuda_indices(&[0]).unwrap_err();
            assert!(err.to_string().contains("no CUDA/ROCm device is available"));
        }
    }

    #[test]
    fn pool_is_send_sync() {
        // Compile-time proof that the !Send boundary is upheld: DevicePool holds
        // no tensors, so it must be Send + Sync for the replication design.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DevicePool>();
    }
}
