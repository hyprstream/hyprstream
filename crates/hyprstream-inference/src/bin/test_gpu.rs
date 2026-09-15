// Test binary intentionally prints diagnostic output
#![allow(clippy::print_stdout, clippy::print_stderr)]

use tch::Device;

fn main() {
    println!("ROCm GPU Detection Test");
    println!("=======================");

    // Set environment for ROCm — use /opt/rocm (standard install path) and let
    // the runtime auto-detect the GPU ISA. HSA_OVERRIDE_GFX_VERSION=9.0.0 was
    // removed: it forced MI210/gfx90a kernels on all GPUs, breaking gfx1151
    // (Strix Halo / Radeon 8060S) by loading the wrong ISA and silently
    // falling back to CPU (#228).
    std::env::set_var("ROCM_PATH", "/opt/rocm");
    std::env::set_var("HIP_VISIBLE_DEVICES", "0");

    println!("Environment set for ROCm (auto-detect GPU architecture)");

    // Check device
    let device = Device::cuda_if_available();

    match device {
        Device::Cpu => {
            println!("Result: CPU mode");
            println!("\nDebug: Trying to force GPU...");

            // Try to force GPU allocation
            match std::panic::catch_unwind(|| {
                let _tensor = tch::Tensor::zeros([1], (tch::Kind::Float, Device::Cuda(0)));
                println!("Force GPU: Success!");
            }) {
                Ok(_) => {}
                Err(_) => println!("Force GPU: Failed - GPU not accessible"),
            }
        }
        Device::Cuda(n) => {
            println!("Result: ✅ GPU {n} detected!");
            // Detection alone is not validation: a libtorch without kernels for
            // the detected ISA (e.g. gfx1151 / Strix Halo) allocates fine but
            // errors or silently falls back on first compute. Exercise real
            // matmul/softmax/reduction on-device and compare against CPU.
            if let Err(err) = compute_smoke(Device::Cuda(n)) {
                eprintln!("❌ GPU compute smoke FAILED: {err:#}");
                std::process::exit(1);
            }
        }
        _ => {
            println!("Result: Other device type detected");
        }
    }
}

/// Run a small end-to-end compute check on `device`: y = softmax(x @ w + b),
/// compared elementwise against the same computation on CPU. Catches
/// missing-ISA kernels, mislinked HIP libraries, and silent CPU fallback.
fn compute_smoke(device: Device) -> anyhow::Result<()> {
    use tch::Kind;

    let shape: [i64; 2] = [256, 256];
    let x = tch::Tensor::f_randn(shape, (Kind::Float, device))?;
    let w = tch::Tensor::f_randn(shape, (Kind::Float, device))?;
    let b = tch::Tensor::f_randn([shape[1]], (Kind::Float, device))?;

    let y_gpu = tch::Tensor::f_softmax(&x.f_matmul(&w)?.f_add(&b)?, -1, Kind::Float)?;

    // Same math on CPU for the reference result (reuse the exact same tensors).
    let (x_c, w_c, b_c) = (
        x.f_to_device(Device::Cpu)?,
        w.f_to_device(Device::Cpu)?,
        b.f_to_device(Device::Cpu)?,
    );
    let y_cpu = tch::Tensor::f_softmax(&x_c.f_matmul(&w_c)?.f_add(&b_c)?, -1, Kind::Float)?;

    let max_diff = y_gpu.f_to_device(Device::Cpu)?.f_sub(&y_cpu)?.f_abs()?.f_max()?;
    let max_diff = f32::try_from(&max_diff)?;
    println!(
        "GPU compute smoke: softmax(matmul+bias) on {device:?} — max |GPU−CPU| = {max_diff:.2e}"
    );
    if !max_diff.is_finite() || max_diff > 1e-4 {
        anyhow::bail!("GPU result diverges from CPU reference by {max_diff}");
    }

    // fp16 path too — HIP matmul requires matching dtypes (see torch_engine
    // TTT notes) and inference weights are commonly half precision. Tolerance
    // is loose: fp16 input rounding on ±N(0,16) dot products, not exact math.
    let y_h = x.f_to_kind(Kind::Half)?.f_matmul(&w.f_to_kind(Kind::Half)?)?;
    let h_diff = y_h
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Float)?
        .f_sub(&x_c.f_matmul(&w_c)?)?
        .f_abs()?
        .f_max()?;
    let h_diff = f32::try_from(&h_diff)?;
    println!("GPU fp16 matmul smoke: max |fp16−fp32| = {h_diff:.2e} (tolerance 1.0)");
    if !h_diff.is_finite() || h_diff > 1.0 {
        anyhow::bail!("fp16 GPU matmul diverges unexpectedly from fp32 ({h_diff})");
    }

    println!("✅ GPU compute validation passed");
    Ok(())
}
