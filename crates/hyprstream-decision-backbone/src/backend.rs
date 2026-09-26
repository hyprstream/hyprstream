//! Dual-backend pin: one source tree, CUDA (RTX 5090 dev) and ROCm (MI210 volume)
//! libtorch builds, zero code divergence.
//!
//! tch-rs links whichever libtorch `LIBTORCH` points at; both CUDA and ROCm builds
//! surface devices through tch's `Device::Cuda` API (ROCm masquerades as CUDA
//! inside torch — kernels, streams, and `Device::Cuda(i)` addressing are identical
//! from Rust). Backend selection is therefore a **build-time** decision, not a
//! code path:
//!
//! - **CUDA build → RTX 5090**: interactive dev + the P3.7 latency reference rig.
//! - **ROCm build → 8× MI210 (2×4 xGMI groups)**: volume training. S6c confirmed
//!   the torch-native GDN path trains on gfx90a through this exact stack.
//!
//! There is no cross-group tensor parallelism (PCIe-only links); cross-group
//! training is data-parallel only (see the crate-level DDP decision).

use tch::Device;

/// The pinned backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPin {
    /// RTX 5090 interactive dev (CUDA libtorch build).
    CudaDev,
    /// MI210 volume training (ROCm libtorch build; gfx90a).
    RocmTraining,
}

/// Probe result for one device: ordinal, whether a kernel actually ran, and the
/// device-reported name when available.
#[derive(Debug, Clone)]
pub struct DeviceProbe {
    /// Device ordinal within the current libtorch build.
    pub ordinal: usize,
    /// True when a tensor alloc + elementwise kernel succeeded on the device.
    pub functional: bool,
}

/// Select the compute device for the current process: `Device::Cuda(0)` when the
/// linked libtorch has a working accelerator (either pinned backend — the API is
/// shared), CPU otherwise. Callers log the choice; nothing here hides a GPU-less
/// host (the S6c inventory flag).
pub fn select_device() -> Device {
    if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

/// Probe every accelerator the linked libtorch can see: for each ordinal, allocate
/// a tensor and run one elementwise kernel. Returns an empty vec on a GPU-less
/// host (CPU-only callers still work; the smoke test reports the flag).
pub fn probe_devices() -> Vec<DeviceProbe> {
    let count = if tch::Cuda::is_available() {
        tch::Cuda::device_count()
    } else {
        0
    };
    (0..count as usize)
        .map(|ordinal| {
            let functional = std::panic::catch_unwind(|| {
                let t = tch::Tensor::ones([8, 8], (tch::Kind::Float, Device::Cuda(ordinal)));
                let out = t.sin().sum(tch::Kind::Float);
                out.double_value(&[]).is_finite()
            })
            .unwrap_or(false);
            DeviceProbe {
                ordinal,
                functional,
            }
        })
        .collect()
}
