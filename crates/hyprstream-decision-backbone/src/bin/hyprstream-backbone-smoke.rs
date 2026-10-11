//! P1.2 smoke test (pinned 4-step): verifies the training stack end-to-end on the
//! converted architecture and reports the numbers the MFU sizing sanity check needs.
//!
//! Steps:
//! 1. **Device probe** — `rocm-smi` / `nvidia-smi` enumeration (degrades to a
//!    "no GPUs attached" report on the current GPU-less hosts — the S6c inventory
//!    flag, not a failure).
//! 2. **FFI probe** — libtorch device tensor alloc + kernel through tch (proves the
//!    linked CUDA/ROCm build matches the driver).
//! 3. **One fwd/bwd optimizer step at 2k context** on the converted hybrid
//!    (miniature config: same GDN + bidirectional softmax structure) with step time
//!    and tokens/sec, plus the S6c sizing reference (0.8B bf16, 30–40% MFU →
//!    ~3.8–5.1 days per 10M×2k-token epoch on one 4×MI210 group).
//! 4. **Optional fla leg** — gated-skipped by design: fla Triton kernels are NOT
//!    load-bearing and may only be adopted behind a Triton ≥ 3.7 gate (gfx90a
//!    chunk-kernel crash, fixed upstream). The leg reports the gate state.
//!
//! Exit code 0 when steps 1–3 complete (on any device); step 4 is informational.

// The step report lines are the machine-read contract of the smoke test: stdout is
// the deliverable (CI logs and the MFU sizing check parse them), so printing is
// intentional here rather than tracing.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::process::Command;
use std::time::Instant;

use hyprstream_decision_backbone::{DecisionModel, MaskPolicy, Qwen35Config, Qwen35Encoder};
use tch::nn::{OptimizerConfig, VarStore};
use tch::{Kind, Tensor};

const SMOKE_CONTEXT: i64 = 2048;

fn main() {
    let mut failures = 0u32;

    // --- Step 1: device probe -------------------------------------------------
    println!("== step 1/4: device probe (rocm-smi / nvidia-smi)");
    let rocm = Command::new("rocm-smi").arg("--showproductname").output();
    match &rocm {
        Ok(out) if out.status.success() => {
            println!("rocm-smi: {}", String::from_utf8_lossy(&out.stdout).trim());
        }
        _ => println!("rocm-smi: not available"),
    }
    let nvidia = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output();
    match &nvidia {
        Ok(out) if out.status.success() => {
            println!(
                "nvidia-smi: {}",
                String::from_utf8_lossy(&out.stdout).trim()
            );
        }
        _ => println!("nvidia-smi: not available"),
    }
    if rocm.as_ref().map(|o| !o.status.success()).unwrap_or(true)
        && nvidia.as_ref().map(|o| !o.status.success()).unwrap_or(true)
    {
        println!("note: no GPUs attached (S6c inventory flag) — steps 2-3 will run on CPU");
    }

    // --- Step 2: FFI probe -----------------------------------------------------
    println!("== step 2/4: libtorch FFI probe");
    let probes = hyprstream_decision_backbone::backend::probe_devices();
    let device = hyprstream_decision_backbone::backend::select_device();
    if probes.is_empty() {
        println!("libtorch sees no accelerators; CPU fallback selected");
    } else {
        for probe in &probes {
            println!(
                "device {}: {}",
                probe.ordinal,
                if probe.functional {
                    "functional"
                } else {
                    "ALLOC/KERNEL FAILED"
                }
            );
            if !probe.functional {
                failures += 1;
            }
        }
    }
    println!("selected device: {device:?}");

    // --- Step 3: one fwd/bwd step at 2k context --------------------------------
    println!("== step 3/4: fwd/bwd optimizer step at {SMOKE_CONTEXT} tokens (converted hybrid, tiny config)");
    let cfg = Qwen35Config::test_tiny();
    let vs = VarStore::new(device);
    let encoder = match Qwen35Encoder::new(&vs.root().sub("model"), &cfg, MaskPolicy::Bidirectional)
    {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FAILED to build converted backbone: {e}");
            std::process::exit(1);
        }
    };
    let model = DecisionModel::new(encoder, &vs.root().sub("scorer"));
    let mut opt = match tch::nn::AdamW::default().build(&vs, 1e-4) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("FAILED to build optimizer: {e}");
            std::process::exit(1);
        }
    };
    let ids = Tensor::randint(
        cfg.vocab_size - 1,
        [1, SMOKE_CONTEXT],
        (Kind::Int64, device),
    );
    let anchor_positions: Vec<u32> = (0..4).map(|i| (SMOKE_CONTEXT as u32) - 4 + i).collect();
    let batch = vec![vec![hyprstream_decision_head::QuestionAnchors {
        question_id: "smoke".to_owned(),
        kind: hyprstream_decision::QuestionKind::Choice,
        anchor_positions,
    }]];
    let start = Instant::now();
    let out = match model.forward(&ids, &batch) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("FAILED forward at 2k ctx: {e}");
            std::process::exit(1);
        }
    };
    let target = Tensor::from_slice(&[0.25f32, 0.25, 0.25, 0.25]).to_device(device);
    let loss = hyprstream_decision_head::soft_target_kl(&out[0][0], &target);
    opt.backward_step(&loss);
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    println!(
        "step ok: loss {:.4}, {:.1} ms, {:.0} tokens/s on {device:?}",
        loss.double_value(&[]),
        secs * 1e3,
        SMOKE_CONTEXT as f64 / secs
    );
    println!(
        "sizing reference (S6c): 0.8B bf16, 30-40% MFU, 10M x 2k-token epoch ≈ 3.8-5.1 days on one 4xMI210 group"
    );
    let _ = model;

    // --- Step 4: optional fla leg (gated) --------------------------------------
    println!("== step 4/4: optional fla leg");
    println!(
        "SKIPPED by design: fla Triton kernels are not load-bearing; adoption is gated on Triton >= 3.7 \
         (gfx90a chunk-kernel crash, fixed upstream). The torch-native GDN path above is the training path."
    );

    if failures > 0 {
        eprintln!("smoke test: {failures} device probe failure(s)");
        std::process::exit(1);
    }
    println!("smoke test: PASS (steps 1-3; step 4 gated-skipped)");
}
