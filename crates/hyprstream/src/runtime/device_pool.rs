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
        // error, not a fallback (no `Device::cuda_if_available()` here). Surface
        // the actual cause: a bare "not available" hid a libcuda.so.1 / wrong-
        // backend problem on the live cuda130 node and made it undiagnosable.
        if !tch::Cuda::is_available() {
            let has_cuda = tch::utils::has_cuda();
            let has_hip = tch::utils::has_hip();
            let device_count = tch::Cuda::device_count();
            let ld_path =
                std::env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| "<unset>".to_owned());
            return Err(anyhow!(
                "DevicePool: GPU devices {indices:?} requested but no CUDA/ROCm device is \
                 available (refusing to silently fall back to CPU). \
                 Diagnostics: has_cuda={has_cuda}, has_hip={has_hip}, \
                 device_count={device_count}, LD_LIBRARY_PATH={ld_path}. If has_cuda=true \
                 but device_count=0, the host driver lib (libcuda.so.1) is not loadable — \
                 add the NVIDIA driver lib dir to LD_LIBRARY_PATH. If has_cuda=false, a \
                 CPU-only libtorch/backend is installed."
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

/// A per-layer device assignment for **2b intra-host pipeline** (#314).
///
/// This is the layer→device `device_map` the [`DevicePool`] doc-comment
/// anticipates. It is a thin, validated newtype over the per-global-layer
/// `Vec<Device>` derived from a [`DevicePool`]: entry `g` is the device that
/// owns global decoder layer `g`. A *stage* is a contiguous run of layers that
/// map to the same device; the only cross-device copy in the forward pass is a
/// single `hidden.to_device(next)` at a boundary where consecutive layers map to
/// different devices (see [`LayerDeviceMap::is_boundary`]).
///
/// # Why a map over `num_hidden_layers`, not the stage-local count
///
/// The map is **global** — indexed by global layer index, length
/// `num_hidden_layers` — so a single source of truth drives both shard-aware
/// construction (per-layer placement + per-layer type/RoPE selection, which need
/// the global index) and the forward loop. A stage that owns layers `[a..b)`
/// queries the same global map; the architecture is responsible for the
/// global↔local remap of its own `self.layers` / KV / SSM vectors via the
/// `layer_offset = a` it was built with.
///
/// # Send / Sync
///
/// Holds only [`Device`] values (`Copy`, no tensors), so it is `Send + Sync` and
/// may be handed to an engine-owning thread. Like [`DevicePool`], **never** add a
/// `Tensor`/`VarStore` field — see the module-level `!Send` boundary note.
#[derive(Debug, Clone)]
pub struct LayerDeviceMap {
    /// One device per global layer index. Always non-empty; `len()` equals the
    /// model's `num_hidden_layers`.
    per_layer: Vec<Device>,
}

impl LayerDeviceMap {
    /// All layers on a single device — the unsplit fast path.
    ///
    /// Used both by single-GPU inference and by the CPU-only equivalence tests
    /// (`forward_layers(0..N)` over an all-CPU map must equal the whole-model
    /// forward). `num_layers` must be non-zero.
    pub fn single(device: Device, num_layers: usize) -> Result<Self> {
        if num_layers == 0 {
            return Err(anyhow!("LayerDeviceMap requires num_layers >= 1 (got 0)"));
        }
        Ok(Self {
            per_layer: vec![device; num_layers],
        })
    }

    /// Build an explicit per-layer map, validating it is non-empty.
    ///
    /// `per_layer[g]` is the device that owns global layer `g`.
    pub fn from_per_layer(per_layer: Vec<Device>) -> Result<Self> {
        if per_layer.is_empty() {
            return Err(anyhow!("LayerDeviceMap requires at least one layer (got none)"));
        }
        Ok(Self { per_layer })
    }

    /// Spread `num_layers` contiguously across a [`DevicePool`]'s devices in
    /// requested order, balanced as evenly as possible (capacity-symmetric).
    ///
    /// Earlier stages get the extra layer when `num_layers` is not divisible by
    /// the device count, matching the natural prefix-heavy split. This is the
    /// parameter-balanced planner; the capacity-aware (asymmetric-VRAM) variant
    /// is a later epic concern (plan §H) and would replace this constructor
    /// without touching the forward path.
    ///
    /// Layers are assigned in **contiguous runs** so each device owns a single
    /// pipeline stage `[a..b)` — never interleaved — which keeps boundary copies
    /// to exactly `(num_stages - 1)` per forward.
    pub fn even_split(pool: &DevicePool, num_layers: usize) -> Result<Self> {
        if num_layers == 0 {
            return Err(anyhow!("LayerDeviceMap requires num_layers >= 1 (got 0)"));
        }
        let devices = pool.devices();
        let n_dev = devices.len();
        let base = num_layers / n_dev;
        let rem = num_layers % n_dev;

        let mut per_layer = Vec::with_capacity(num_layers);
        for (d, &dev) in devices.iter().enumerate() {
            // First `rem` devices get one extra layer (prefix-heavy split).
            let count = base + usize::from(d < rem);
            for _ in 0..count {
                per_layer.push(dev);
            }
        }
        debug_assert_eq!(per_layer.len(), num_layers);
        Ok(Self { per_layer })
    }

    /// Device owning global layer `global_layer_idx`.
    ///
    /// # Panics
    /// Panics if `global_layer_idx >= len()` — a programming error (the caller
    /// iterates a known global range). Use [`Self::try_device_for`] for a checked
    /// variant.
    #[must_use]
    pub fn device_for(&self, global_layer_idx: usize) -> Device {
        self.per_layer[global_layer_idx]
    }

    /// Checked [`Self::device_for`].
    pub fn try_device_for(&self, global_layer_idx: usize) -> Result<Device> {
        self.per_layer
            .get(global_layer_idx)
            .copied()
            .ok_or_else(|| {
                anyhow!(
                    "LayerDeviceMap: global layer index {global_layer_idx} out of range \
                     (map covers {} layers)",
                    self.per_layer.len()
                )
            })
    }

    /// Whether a cross-device boundary copy is needed *before* running layer
    /// `global_layer_idx`, given the device the previous layer's output is on.
    ///
    /// Returns `false` for the first layer of a stage when its device equals
    /// `prev_device` (zero-copy within a stage and across same-device stages), and
    /// `true` only when the device actually changes. This is the single point
    /// that gates the lone `hidden.to_device()` in the forward loop.
    #[must_use]
    pub fn is_boundary(&self, prev_device: Device, global_layer_idx: usize) -> bool {
        self.device_for(global_layer_idx) != prev_device
    }

    /// Number of global layers covered (equals the model's `num_hidden_layers`).
    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.per_layer.len()
    }

    /// Whether every layer maps to the same device (the unsplit fast path).
    ///
    /// When true, `forward_layers` performs zero cross-device copies and is
    /// numerically identical to the single-device whole-model forward.
    #[must_use]
    pub fn is_single_device(&self) -> bool {
        self.per_layer
            .first()
            .is_some_and(|&first| self.per_layer.iter().all(|&d| d == first))
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
        // The layer→device map crosses to the engine thread too, so it must also
        // be Send + Sync (it holds only Device values).
        assert_send_sync::<LayerDeviceMap>();
    }

    #[test]
    fn layer_map_single_is_zero_copy() {
        let map = LayerDeviceMap::single(Device::Cpu, 4).unwrap();
        assert_eq!(map.len(), 4);
        assert!(map.is_single_device());
        for g in 0..4 {
            assert_eq!(map.device_for(g), Device::Cpu);
        }
        // No boundary anywhere when every layer is the same device.
        for g in 1..4 {
            assert!(!map.is_boundary(Device::Cpu, g));
        }
    }

    #[test]
    fn layer_map_single_rejects_zero() {
        assert!(LayerDeviceMap::single(Device::Cpu, 0).is_err());
        assert!(LayerDeviceMap::from_per_layer(vec![]).is_err());
        let pool = DevicePool::from_devices(vec![Device::Cpu]).unwrap();
        assert!(LayerDeviceMap::even_split(&pool, 0).is_err());
    }

    #[test]
    fn even_split_is_contiguous_and_prefix_heavy() {
        // 5 layers over 2 devices → [d0,d0,d0, d1,d1] (prefix gets the extra).
        let pool =
            DevicePool::from_devices(vec![Device::Cpu, Device::Cuda(0)]).unwrap();
        let map = LayerDeviceMap::even_split(&pool, 5).unwrap();
        assert_eq!(map.len(), 5);
        assert_eq!(map.device_for(0), Device::Cpu);
        assert_eq!(map.device_for(1), Device::Cpu);
        assert_eq!(map.device_for(2), Device::Cpu);
        assert_eq!(map.device_for(3), Device::Cuda(0));
        assert_eq!(map.device_for(4), Device::Cuda(0));
        assert!(!map.is_single_device());

        // Exactly one boundary, at the device change (layer 3).
        let mut boundaries = 0;
        let mut prev = map.device_for(0);
        for g in 1..map.len() {
            if map.is_boundary(prev, g) {
                boundaries += 1;
            }
            prev = map.device_for(g);
        }
        assert_eq!(boundaries, 1, "contiguous 2-device split has exactly one boundary");
    }

    #[test]
    fn even_split_balanced_when_divisible() {
        let pool = DevicePool::from_devices(vec![
            Device::Cpu,
            Device::Cuda(0),
            Device::Cuda(1),
        ])
        .unwrap();
        let map = LayerDeviceMap::even_split(&pool, 6).unwrap();
        assert_eq!(
            (0..6).map(|g| map.device_for(g)).collect::<Vec<_>>(),
            vec![
                Device::Cpu,
                Device::Cpu,
                Device::Cuda(0),
                Device::Cuda(0),
                Device::Cuda(1),
                Device::Cuda(1),
            ]
        );
    }

    #[test]
    fn try_device_for_is_checked() {
        let map = LayerDeviceMap::single(Device::Cpu, 2).unwrap();
        assert_eq!(map.try_device_for(1).unwrap(), Device::Cpu);
        assert!(map.try_device_for(2).unwrap_err().to_string().contains("out of range"));
    }
}
