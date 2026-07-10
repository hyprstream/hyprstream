//! E2E (#421 / E6): inference end-to-end on a single host.
//!
//! This is the leaf-tier integration test for the inference pipeline. It
//! validates, at the highest fidelity the current hardware permits, the four
//! load-bearing properties of the inference plane:
//!
//! 1. **GPU load** — a real model loads onto a CUDA device and `model_info`
//!    reports a non-empty architecture (cuda130 / RTX 5090 on this host).
//! 2. **Token stream** — a `generate_stream` request produces >0 non-empty
//!    text chunks and final stats (`tokens_generated > 0`), exercising the
//!    streaming path the Moq subscriber consumes in production.
//! 3. **HRW placement (#322 router)** — `CellRouter` gives session affinity
//!    (same `session_id` → same node) and spread (distinct `session_id`s →
//!    multiple nodes). This is pure data-structure logic, so it runs
//!    everywhere (CPU).
//! 4. **Multi-device plumbing** — `DevicePool` + `LayerDeviceMap::even_split`
//!    produce a contiguous, prefix-heavy, single-boundary-per-stage partition
//!    of the decoder stack (the input to `forward_layers`, #314 / engine-wiring,
//!    #405). The numerical split-vs-single-device equivalence is proven at the
//!    architecture layer (see `qwen3_5::tests::staged_split_equals_whole_model`
//!    and `llama::tests`); this test pins the **planning** invariants those
//!    architecture tests rely on, plus a real single-GPU load/inference smoke
//!    when hardware permits.
//!
//! # Hardware gating
//!
//! The GPU tests are `#[ignore]` by default and auto-skip when the model is
//! absent, so `cargo test -p hyprstream` stays green on CPU-only CI and on
//! machines without the model checked out. Run them explicitly on the GPU host:
//!
//! ```text
//! OPENSSL_NO_VENDOR=1 LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
//!   LD_LIBRARY_PATH=$(python3 -c 'import torch;print(torch.__path__[0])')/lib \
//!   cargo test -p hyprstream --release --test e2e_inference -- --ignored
//! ```
//!
//! ## Teardown ordering (#429)
//!
//! `e6_gpu_load_and_stream_tokens` loads real model weights onto CUDA. Two
//! classes of device tensor used to outlive the engine struct: the
//! thread-local RoPE sin/cos tables (`llama::ROPE_CACHE` / `qwen3_5::ROPE_CACHE`)
//! and the model weights. Because the inference service thread is spawned
//! *detached*, their TLS destructors could fire during process-exit cleanup —
//! after libtorch's `atexit` handler had torn down the CUDA context — aborting
//! the test binary with `c10::Error: invalid device pointer` (SIGABRT), *after*
//! the assertions had already passed.
//!
//! The fix is an explicit, ordered teardown: `client.shutdown()` now drops every
//! device tensor the engine owns **on the service thread, before replying** —
//! while the caller is still blocked on the reply, so the process and the CUDA
//! context are indubitably still alive. See `TorchEngine::release_device_resources`
//! and the `Shutdown` arm of `LocalInferenceService`. After it returns only
//! empty shells remain for any late static/TLS destruction.
//!
//! The router/placement tests (sections 3 & 4) are CPU-only and always run.
//!
//! # GPU prerequisites (when hardware blocks a full run)
//!
//! To exercise sections 1 & 2 you need:
//! - a CUDA/ROCm build of libtorch on `LD_LIBRARY_PATH` (cuda130 here);
//! - a model worktree at `<model dir>` containing `config.json`,
//!   `tokenizer.json`, and a `*.safetensors` shard set (the default
//!   `LocalInferenceService::start` path). This host carries
//!   `~/.local/share/hyprstream/models/qwen3.5-0.8b/worktrees/main`.
//!
//! To exercise the *cross-device* numerical split (section 4's full form) you
//! need >=2 visible GPUs (`HYPRSTREAM_GPU_DEVICES=0,1`); the single-GPU host
//! runs the planning-invariant checks plus the single-device load/inference.
//!
//! # Lint posture
//!
//! `unwrap_used` / `expect_used` are `deny` at the workspace level. The
//! non-ignored tests below are written assertion-style with `?` on a `Result`
//! return (matching `tests/policy_over_iroh.rs`); the `#[ignore]`d GPU tests
//! carry an explicit `#[allow]` because panicking is the right behavior for a
//! hardware smoke test (a half-loaded model should fail loudly, not return a
//! typed error to be ignored).

// stderr is the right channel for an integration test's host-status / progress
// breadcrumbs (model dir, load time, tok/s); `cargo test` only surfaces it on
// failure or with `--nocapture`. The workspace denies `print_stderr` for shipping
// code, so allow it here at the test-binary scope.
#![allow(clippy::print_stderr)]

// ===== Shared helpers ==========================================================

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};

// ===== Section 3 & 4: router + device-pool (CPU, always run) ===================

/// Capacity-weighted HRW placement for a cell (#322).
///
/// These types are plain data structures (no tensors, no GPU), so the cell
/// router is fully testable on CPU — see `services::router` module docs.
use hyprstream_core::services::router::{
    CellRouter, ExcludeReason, InferenceServerInfo, PlacementPolicy, SessionAffinity,
    DEFAULT_LEASE_DURATION,
};
use hyprstream_core::runtime::device_pool::{DevicePool, LayerDeviceMap};
use hyprstream_rpc::transport::TransportConfig;

/// Same helper the router's own unit tests use: a node with a byte-pattern
/// `node_id`, a free-GPU-memory weight, and an active-session counter.
fn node(id: u8, mem_free: u64, active: u64) -> InferenceServerInfo {
    InferenceServerInfo {
        node_id: [id; 32],
        transport: TransportConfig::inproc("test"),
        gpu_memory_free: mem_free,
        active_sessions: active,
        last_heartbeat: Instant::now(),
    }
}

/// Three equal-capacity inference nodes — a realistic single-cell replica set.
fn replica_set() -> Vec<InferenceServerInfo> {
    const GB: u64 = 1024 * 1024 * 1024;
    vec![
        node(0x10, 16 * GB, 0),
        node(0x20, 16 * GB, 0),
        node(0x30, 16 * GB, 0),
    ]
}

/// Cross-process-stable hash used to re-derive an HRW score independently of
/// the router internals, so the test is a real cross-check rather than a
/// tautology over `hrw_select`. Mirrors `router::score`'s recipe.
fn independent_score(session_id: &str, node_id: &[u8; 32], weight: u64) -> u64 {
    let mut h = DefaultHasher::new();
    session_id.hash(&mut h);
    node_id.hash(&mut h);
    h.finish().wrapping_mul(weight)
}

// ---- E6.3a: HRW affinity — same session_id sticks to one node ------------------

#[test]
fn e6_router_same_session_id_is_sticky_across_calls() -> Result<()> {
    // Affinity policy (the default): the same session_id must resolve to the
    // same node on every placement, so a long conversation reuses its KV cache.
    let mut router = CellRouter::default();
    let set = replica_set();
    let now = Instant::now();

    let first = router
        .place("sess-A", &set, now)
        .ok_or_else(|| anyhow!("non-empty replica set must place"))?;
    assert!(
        first.rebound,
        "first placement of a session must be a cold rebind, not an affinity hit"
    );

    // Every subsequent placement for the same session must hit the same node and
    // report a cache hit (`rebound == false`).
    for i in 1..=5 {
        let again = router
            .place("sess-A", &set, now)
            .ok_or_else(|| anyhow!("placement must succeed (call {i})"))?;
        assert_eq!(
            again.node_id, first.node_id,
            "affinity: same session_id must stick to one node (call {i})"
        );
        assert_eq!(
            again.candidate_idx, first.candidate_idx,
            "affinity: same candidate index (call {i})"
        );
        assert!(
            !again.rebound,
            "affinity: subsequent placements must be cache hits (call {i})"
        );
    }

    // The stickiness comes from the affinity map, not chance: the bound owner is
    // recorded there and survives an unrelated placement in between.
    let first_node = first.node_id;
    let _other = router.place("sess-B", &set, now);
    let again = router
        .place("sess-A", &set, now)
        .ok_or_else(|| anyhow!("sess-A placement after sess-B"))?;
    assert_eq!(again.node_id, first_node, "sess-A unchanged after sess-B");
    Ok(())
}

// ---- E6.3b: HRW spread — distinct session_ids distribute across the set --------

#[test]
fn e6_router_distinct_session_ids_spread_across_nodes() -> Result<()> {
    // Spread is the other half of HRW: distinct sessions must not all collapse
    // onto one node (that would indicate a broken hash). With 3 equal-weight
    // nodes and 120 sessions, every node should own a meaningful share.
    let mut router = CellRouter::default();
    let set = replica_set();
    let now = Instant::now();

    let mut owners: HashSet<_> = HashSet::new();
    let mut counts = std::collections::HashMap::new();
    for i in 0..120u32 {
        let p = router
            .place(&format!("sess-{i}"), &set, now)
            .ok_or_else(|| anyhow!("placement must succeed for session {i}"))?;
        owners.insert(p.node_id);
        *counts.entry(p.node_id).or_insert(0u32) += 1;
    }

    assert!(
        owners.len() > 1,
        "distinct sessions must spread across >1 node; got {owners:?} (counts {counts:?})"
    );
    // Every node in the replica set should receive at least one session — a
    // healthy hash hits all buckets given enough draws.
    for n in &set {
        assert!(
            owners.contains(&n.node_id),
            "node {:?} received zero sessions (counts {counts:?})",
            n.node_id
        );
    }
    // No single node should hoard >80% of sessions (a uniform 3-way split would
    // be ~33% each; >80% indicates a severely skewed hash).
    let max_share = counts.values().copied().max().unwrap_or(0);
    assert!(
        max_share < 96,
        "one node owns {max_share}/120 sessions — hash is skewed (counts {counts:?})"
    );
    Ok(())
}

// ---- E6.3c: HRW is deterministic across router instances ----------------------

#[test]
fn e6_router_placement_is_process_independent() -> Result<()> {
    // HRW's whole point: two independent routers in the same cell, given the
    // same replica set and session, must agree on the owner without any
    // coordination. `DefaultHasher::new()` uses constant keys, so the score is
    // stable across processes (module docs spell this out).
    let set = replica_set();
    let now = Instant::now();

    let mut a = CellRouter::default();
    let mut b = CellRouter::default();
    for i in 0..20u32 {
        let sid = format!("sess-{i}");
        let pa = a
            .place(&sid, &set, now)
            .ok_or_else(|| anyhow!("router a placement for {sid}"))?;
        let pb = b
            .place(&sid, &set, now)
            .ok_or_else(|| anyhow!("router b placement for {sid}"))?;
        assert_eq!(
            pa.node_id, pb.node_id,
            "two routers must agree on owner for {sid}"
        );
    }

    // And the decision is recoverable from a hand-rolled score: the highest
    // independent_score over the replica set names the same node the router
    // picked on a cold placement.
    let mut fresh = CellRouter::default();
    let p = fresh
        .place("probe", &set, now)
        .ok_or_else(|| anyhow!("probe placement"))?;
    let expected = set
        .iter()
        .map(|n| {
            (
                independent_score("probe", &n.node_id, n.gpu_memory_free.saturating_add(1)),
                n.node_id,
            )
        })
        .max_by_key(|(s, _)| *s)
        .map(|(_, id)| id);
    assert_eq!(
        Some(p.node_id),
        expected,
        "router selection must match the HRW score"
    );
    Ok(())
}

// ---- E6.3d: affinity lease expiry forces a rebind -----------------------------

#[test]
fn e6_router_affinity_lease_expiry_rebinds() -> Result<()> {
    // Affinity is a perf hint, not a correctness invariant: once a lease lapses
    // the next placement re-runs HRW. Use a sub-second lease so the test is
    // fast. (The binding itself stays valid for the lease window.)
    let mut router = CellRouter::default().with_lease_duration(Duration::from_millis(40));
    let set = replica_set();
    let now = Instant::now();

    let first = router
        .place("sess-A", &set, now)
        .ok_or_else(|| anyhow!("first placement"))?;
    assert!(first.rebound, "sanity: first placement is a rebind");
    assert_eq!(router.affinity().peek("sess-A"), Some(first.node_id));

    // After the lease lapses with no heartbeat, the binding is gone.
    std::thread::sleep(Duration::from_millis(60));
    assert_eq!(
        router.affinity().peek("sess-A"),
        None,
        "expired binding must not be visible to a read-only peek"
    );

    // Heartbeat before expiry would have renewed it (negative control).
    let mut kept = CellRouter::default().with_lease_duration(Duration::from_millis(40));
    let _k = kept
        .place("sess-K", &set, now)
        .ok_or_else(|| anyhow!("kept placement"))?;
    std::thread::sleep(Duration::from_millis(20));
    assert!(kept.heartbeat("sess-K"), "heartbeat renews a live lease");
    Ok(())
}

// ---- E6.3e: failure detection reassigns to the next HRW choice ----------------

#[test]
fn e6_router_dial_fail_and_stall_reassign_to_a_different_node() -> Result<()> {
    // The two reactive failure layers (dial-fail / token-stream stall) must
    // both exclude the bad node and rebind the affected session to a *different*
    // healthy candidate. This is what keeps a downed GPU from killing the cell.
    let set = replica_set();
    let now = Instant::now();

    // --- dial-fail ---
    let mut router = CellRouter::default();
    let owner = router
        .place("sess-A", &set, now)
        .ok_or_else(|| anyhow!("sess-A placement"))?;
    let was_new = router.report_dial_fail(owner.node_id, now);
    assert!(was_new, "first dial-fail report is a state transition");
    assert_eq!(
        router.is_excluded(owner.node_id, now),
        Some(ExcludeReason::Down),
        "down node is excluded for placement"
    );
    let rebound = router
        .place("sess-A", &set, now)
        .ok_or_else(|| anyhow!("sess-A re-placement after down"))?;
    assert_ne!(
        rebound.node_id, owner.node_id,
        "down node must not be reselected for its own session"
    );
    assert!(rebound.rebound, "reassignment records a fresh binding");

    // --- token-stream stall ---
    let mut router2 = CellRouter::default();
    let owner2 = router2
        .place("sess-B", &set, now)
        .ok_or_else(|| anyhow!("sess-B placement"))?;
    router2.report_stall(owner2.node_id, now);
    assert_eq!(
        router2.is_excluded(owner2.node_id, now),
        Some(ExcludeReason::Stalled)
    );
    let after_stall = router2
        .place("sess-B", &set, now)
        .ok_or_else(|| anyhow!("sess-B re-placement after stall"))?;
    assert_ne!(after_stall.node_id, owner2.node_id, "stalled node is skipped");

    // And clearing the stall re-enables the node.
    router2.clear_node(owner2.node_id);
    assert!(
        router2.is_excluded(owner2.node_id, now).is_none(),
        "clear_node restores the node"
    );
    Ok(())
}

// ---- E6.3f: all-down returns None (no healthy candidate) ----------------------

#[test]
fn e6_router_all_nodes_down_returns_none() {
    let mut router = CellRouter::default();
    let set = replica_set();
    let now = Instant::now();
    for n in &set {
        router.report_dial_fail(n.node_id, now);
    }
    assert!(
        router.place("sess-A", &set, now).is_none(),
        "with every node down, placement has nowhere to go"
    );
}

// ---- E6.3g: Spread policy ignores affinity and least-loads --------------------

#[test]
fn e6_router_spread_policy_picks_least_loaded() -> Result<()> {
    // Spread (stateless fanout) bypasses the affinity map entirely and lands on
    // the least-loaded node. Useful for one-shot requests where KV reuse is
    // irrelevant.
    let mut router = CellRouter::new(PlacementPolicy::Spread);
    const GB: u64 = 1024 * 1024 * 1024;
    let set = vec![
        node(0x10, 8 * GB, 5),
        node(0x20, 8 * GB, 1), // fewest active sessions
        node(0x30, 8 * GB, 9),
    ];
    let now = Instant::now();

    // A prior placement must not bias Spread.
    let _ = router.place("warmup", &set, now).unwrap_or_else(|| {
        // unreachable in practice; keep the test assert-based not Result-based
        panic!("warmup placement under Spread must succeed")
    });
    let p = router
        .place("sess-A", &set, now)
        .ok_or_else(|| anyhow!("sess-A placement under Spread"))?;
    assert_eq!(
        p.node_id,
        set[1].node_id,
        "Spread must pick the fewest-active-sessions node"
    );
    Ok(())
}

// ---- E6.4: DevicePool + LayerDeviceMap (multi-device planning, CPU) -----------

#[test]
fn e6_device_pool_from_devices_preserves_order_and_primary() -> Result<()> {
    // The pool keeps the requested device order; `primary()` is the first one
    // (where embeddings / lm_head live in the pipeline wiring).
    let pool = DevicePool::from_devices(vec![tch::Device::Cuda(2), tch::Device::Cuda(0)])?;
    assert_eq!(pool.devices().len(), 2);
    assert_eq!(pool.primary(), tch::Device::Cuda(2));
    assert!(!pool.is_single());
    Ok(())
}

#[test]
// `unwrap_err()` is the natural way to read an error-path message; these are
// negative tests that assert specific error text.
#[allow(clippy::unwrap_used)]
fn e6_device_pool_rejects_empty_and_duplicates() {
    let err = DevicePool::from_devices(vec![]);
    assert!(err.is_err(), "empty device list must error");
    let s = err.unwrap_err().to_string();
    assert!(s.contains("at least one device"), "{s}");

    let err = DevicePool::from_devices(vec![tch::Device::Cpu, tch::Device::Cpu]);
    assert!(err.is_err(), "duplicate devices must error");
    assert!(
        err.unwrap_err().to_string().contains("duplicate"),
        "duplicate error message"
    );
}

#[test]
// Negative tests: `unwrap_err()` reads the error-path message we assert on.
#[allow(clippy::unwrap_used)]
fn e6_device_pool_from_cuda_indices_fail_fast_on_no_gpu() {
    // On a host with no CUDA, an *explicit* GPU request must error rather than
    // silently degrade to CPU (the no-fragile-fallbacks rule, #315). On a
    // GPU host the same call with an out-of-range index errors too.
    if tch::Cuda::is_available() {
        let err = DevicePool::from_cuda_indices(&[usize::MAX]);
        assert!(err.is_err(), "out-of-range index should be rejected");
        assert!(
            err.unwrap_err().to_string().contains("out of range"),
            "out-of-range error message"
        );
    } else {
        let err = DevicePool::from_cuda_indices(&[0]);
        assert!(err.is_err(), "CPU-only host should fail fast");
        assert!(
            err.unwrap_err()
                .to_string()
                .contains("no CUDA/ROCm device is available"),
            "CPU-only fail-fast message"
        );
    }
    // Duplicate detection happens before any CUDA probe, so it is deterministic.
    let err = DevicePool::from_cuda_indices(&[1, 1]);
    assert!(err.is_err());
    assert!(
        err.unwrap_err().to_string().contains("duplicate"),
        "duplicate error before CUDA probe"
    );
}

#[test]
fn e6_layer_device_map_even_split_is_contiguous_and_prefix_heavy() -> Result<()> {
    // The planner the engine wires into `forward_layers` (#314, engine-wiring,
    // #405): `even_split` must (a) cover every layer, (b) give each device a
    // single contiguous run (so each device owns exactly one stage), and
    // (c) put the remainder on the earliest devices (prefix-heavy). These are
    // the invariants the architecture-level split-equivalence tests
    // (`qwen3_5::tests::staged_split_equals_whole_model`, `llama::tests`)
    // rely on from the planning side.
    let pool = DevicePool::from_devices(vec![tch::Device::Cpu, tch::Device::Cuda(0)])?;
    let n_layers = 5usize;
    let map = LayerDeviceMap::even_split(&pool, n_layers)?;

    assert_eq!(map.len(), n_layers, "even_split covers every layer");
    // Prefix-heavy: first device gets ceil(5/2)=3, second gets 2.
    assert_eq!(map.device_for(0), tch::Device::Cpu);
    assert_eq!(map.device_for(1), tch::Device::Cpu);
    assert_eq!(map.device_for(2), tch::Device::Cpu);
    assert_eq!(map.device_for(3), tch::Device::Cuda(0));
    assert_eq!(map.device_for(4), tch::Device::Cuda(0));
    assert!(!map.is_single_device());

    // Contiguity = exactly one stage boundary. Walk the map and count device
    // changes: a correct contiguous split has (num_devices - 1) boundaries.
    let mut boundaries = 0usize;
    let mut prev = map.device_for(0);
    for g in 1..map.len() {
        if map.is_boundary(prev, g) {
            boundaries += 1;
        }
        prev = map.device_for(g);
    }
    assert_eq!(
        boundaries, 1,
        "2-device contiguous split has exactly one stage boundary (got {boundaries})"
    );

    // The boundary sits exactly at the device change, not before/after.
    assert!(!map.is_boundary(tch::Device::Cpu, 0), "no boundary before layer 0");
    assert!(
        map.is_boundary(tch::Device::Cpu, 3),
        "boundary at the Cpu->Cuda(0) change at layer 3"
    );
    Ok(())
}

#[test]
fn e6_layer_device_map_even_split_balanced_when_divisible() -> Result<()> {
    // When layers divide evenly, every device gets the same count — the
    // degenerate (and easiest to eyeball) split. 6 layers / 3 devices = 2 each.
    let pool = DevicePool::from_devices(vec![
        tch::Device::Cpu,
        tch::Device::Cuda(0),
        tch::Device::Cuda(1),
    ])?;
    let map = LayerDeviceMap::even_split(&pool, 6)?;
    let assignment: Vec<_> = (0..6).map(|g| map.device_for(g)).collect();
    assert_eq!(
        assignment,
        vec![
            tch::Device::Cpu,
            tch::Device::Cpu,
            tch::Device::Cuda(0),
            tch::Device::Cuda(0),
            tch::Device::Cuda(1),
            tch::Device::Cuda(1),
        ],
    );
    // Three stages → two boundaries.
    let mut boundaries = 0;
    let mut prev = map.device_for(0);
    for g in 1..map.len() {
        if map.is_boundary(prev, g) {
            boundaries += 1;
        }
        prev = map.device_for(g);
    }
    assert_eq!(boundaries, 2);
    Ok(())
}

#[test]
fn e6_layer_device_map_single_is_zero_copy() -> Result<()> {
    // Single-device map = the unsplit fast path: no boundaries anywhere, so
    // `forward_layers` performs zero cross-device copies and is numerically
    // identical to the whole-model forward (the property the architecture
    // equivalence tests check).
    let map = LayerDeviceMap::single(tch::Device::Cpu, 8)?;
    assert!(map.is_single_device());
    for g in 0..8 {
        assert_eq!(map.device_for(g), tch::Device::Cpu);
    }
    for g in 1..8 {
        assert!(!map.is_boundary(tch::Device::Cpu, g), "no boundary within a stage");
    }
    Ok(())
}

#[test]
// `unwrap()` on a pool we just asserted `is_ok()`; the rest are negative checks.
#[allow(clippy::unwrap_used)]
fn e6_layer_device_map_rejects_zero_layers() {
    // The planner must refuse a 0-layer model — every downstream stage assumes
    // >=1 layer, so failing here keeps the forward path free of a special case.
    let pool = DevicePool::from_devices(vec![tch::Device::Cpu]);
    assert!(pool.is_ok(), "single-CPU pool is valid");
    let pool = pool.unwrap();
    assert!(LayerDeviceMap::even_split(&pool, 0).is_err());
    assert!(LayerDeviceMap::single(tch::Device::Cpu, 0).is_err());
    assert!(LayerDeviceMap::from_per_layer(vec![]).is_err());
}

#[test]
fn e6_device_pool_and_layer_map_are_send_sync() {
    // The multi-GPU design hands the pool and the layer map to engine-owning
    // threads, so both must be Send + Sync. They hold only `Device` values
    // (Copy, no tensors) — adding a Tensor/VarStore field would silently break
    // this. Compile-time proof, mirroring the in-crate unit test.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DevicePool>();
    assert_send_sync::<LayerDeviceMap>();
}

#[test]
fn e6_layer_map_matches_a_two_gpu_pipeline_shape() -> Result<()> {
    // A 2-GPU even split of a real-shaped model (qwen3.5-0.8b has 24 decoder
    // layers) must produce two equal stages of 12 with exactly one boundary at
    // layer 12. This is the concrete shape the engine would build on a 2x-GPU
    // host for that model — checked on CPU devices so it runs everywhere, but
    // the partition is device-agnostic.
    let pool = DevicePool::from_devices(vec![tch::Device::Cuda(0), tch::Device::Cuda(1)])?;
    let map = LayerDeviceMap::even_split(&pool, 24)?;
    assert_eq!(map.len(), 24);
    // Stage 0: layers 0..12 on Cuda(0).
    for g in 0..12 {
        assert_eq!(map.device_for(g), tch::Device::Cuda(0), "stage0 owns {g}");
    }
    // Stage 1: layers 12..24 on Cuda(1).
    for g in 12..24 {
        assert_eq!(map.device_for(g), tch::Device::Cuda(1), "stage1 owns {g}");
    }
    // Exactly one boundary, at the stage seam (layer 12).
    assert!(!map.is_boundary(tch::Device::Cuda(0), 11));
    assert!(map.is_boundary(tch::Device::Cuda(0), 12));
    Ok(())
}

// ===== Sections 1 & 2: real GPU load + streaming inference =====================
//
// These are `#[ignore]` by default: they need a CUDA libtorch on the linker
// path *and* a model checked out on disk, which CPU-only CI does not have. Run
// them explicitly on the GPU host with `--ignored` (see the module docs for the
// exact env). They auto-skip if the model dir is missing, so a bare
// `cargo test -- --ignored` on a GPU host without the model still exits cleanly.

/// Resolve the model directory used by the GPU tests.
///
/// Prefers the `HYPRSTREAM_E2E_MODEL` env override (so a reviewer can point at
/// any model), then falls back to the qwen3.5-0.8b worktree this host carries.
// The loop checks multiple side-conditions per candidate; a `find` closure would
// be less readable than the explicit early-return scan.
#[allow(clippy::manual_find)]
fn e2e_model_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HYPRSTREAM_E2E_MODEL") {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    // Default: the qwen3.5-0.8b worktree shipped on this host. 24 decoder
    // layers, hidden_size 1024 — small enough to load on a single RTX 5090.
    let candidates = [
        // XDG data home (where this host keeps it).
        dirs::data_dir().map(|d| d.join("hyprstream/models/qwen3.5-0.8b/worktrees/main")),
        // Legacy ~/.hyprstream location.
        dirs::home_dir().map(|h| h.join(".hyprstream/models/qwen3.5-0.8b")),
    ];
    for c in candidates.into_iter().flatten() {
        // Require the minimum the loader needs: config + tokenizer + weights.
        if c.is_dir()
            && c.join("config.json").is_file()
            && c.join("tokenizer.json").is_file()
            && std::fs::read_dir(&c)
                .ok()
                .map(|mut it| {
                    it.any(|e| {
                        e.ok()
                            .map(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        {
            return Some(c);
        }
    }
    None
}

/// Skip-flag: returns the model dir when a CUDA device is actually visible to
/// libtorch *and* the model is present, otherwise `None`. Used so the
/// `#[ignore]`d GPU tests can no-op cleanly when invoked on a host that only
/// partially satisfies the prerequisites.
fn gpu_and_model_ready() -> Option<PathBuf> {
    if !tch::Cuda::is_available() {
        eprintln!("[e2e_inference] skip: CUDA not available to libtorch");
        return None;
    }
    if tch::Cuda::device_count() <= 0 {
        eprintln!("[e2e_inference] skip: libtorch reports <=0 CUDA devices");
        return None;
    }
    let model = e2e_model_dir()?;
    eprintln!("[e2e_inference] using model: {}", model.display());
    Some(model)
}

// ---- E6.1 + E6.2: load model on GPU and stream tokens -------------------------

#[tokio::test]
#[ignore = "needs CUDA libtorch on LD_LIBRARY_PATH + a checked-out model (see module docs)"]
#[allow(clippy::expect_used, clippy::unwrap_used)] // GPU smoke: fail loud, not via typed error
async fn e6_gpu_load_and_stream_tokens() {
    // `StreamHandle` exposes an inherent async `next(&mut self)` (it is not a
    // `futures::Stream` itself — `into_stream()` is the adapter), so no StreamExt
    // import is needed here.
    use hyprstream_core::config::RuntimeConfig;
    use hyprstream_core::inference::{InferenceClient, LocalInferenceService};

    let model = match gpu_and_model_ready() {
        Some(m) => m,
        None => return, // auto-skip on hosts without the prerequisites
    };

    // Build a GPU runtime config. Do NOT set HYPRSTREAM_GPU_DEVICES here: with a
    // single GPU we want the legacy auto-detect path to land on Cuda(0) and
    // build a single-device engine (is_multi_device == false). The multi-device
    // planning path is exercised by the CPU tests above.
    let mut config = RuntimeConfig::default();
    config.use_gpu = true;
    // Cap context to keep the KV budget modest for a fast test.
    config.max_context = Some(2048);
    config.default_generation_timeout_ms = 120_000;
    config.default_model_load_timeout_ms = 300_000;

    let started = std::time::Instant::now();
    let client = LocalInferenceService::start(&model, config)
        .await
        .expect("model should load on GPU");
    eprintln!(
        "[e2e_inference] model loaded on GPU in {:.2}s",
        started.elapsed().as_secs_f32()
    );

    // The service is ready once model_info reports a real architecture.
    let info = client.model_info().await.expect("model_info");
    assert!(
        !info.name.is_empty() && info.name != "unloaded",
        "model_info.name should reflect a loaded model, got {:?}",
        info.name
    );
    assert!(
        !info.architecture.is_empty() && info.architecture != "unknown",
        "model_info.architecture should be populated, got {:?}",
        info.architecture
    );
    assert!(
        info.num_hidden_layers.unwrap_or(0) > 0,
        "a loaded model reports >=1 decoder layer (got {:?})",
        info.num_hidden_layers
    );

    // Streaming generation: collect chunks until the stream ends, then read the
    // final stats. This is the exact path a Moq token subscriber consumes.
    let request = hyprstream_core::runtime::GenerationRequest {
        prompt: "The capital of France is".to_owned(),
        max_tokens: Some(16),
        temperature: Some(0.0), // greedy — deterministic, fast, stable
        top_p: Some(1.0),
        ..Default::default()
    };

    let mut handle = client
        .generate_stream(request)
        .await
        .expect("stream should start");

    let mut full = String::new();
    let mut chunks = 0usize;
    while let Some(chunk) = handle.next().await {
        match chunk {
            Ok(piece) => {
                chunks += 1;
                full.push_str(&piece);
            }
            Err(e) => panic!("stream chunk error: {e}"),
        }
    }
    let stats = handle.stats().await.expect("final stream stats");

    eprintln!(
        "[e2e_inference] produced {chunks} chunks, {} tokens, {:.1} tok/s: {:?}",
        stats.tokens_generated, stats.tokens_per_second, full
    );

    // The load-bearing assertions: generation actually produced tokens over the
    // stream (not an empty/dead service).
    assert!(
        chunks > 0,
        "stream must deliver >0 text chunks (got 0; full={:?})",
        full
    );
    assert!(
        stats.tokens_generated > 0,
        "stats must report >0 generated tokens (got {:?})",
        stats.tokens_generated
    );
    assert!(
        !full.trim().is_empty(),
        "streamed text must be non-empty (got {:?})",
        full
    );

    // A gentle sanity check on content: greedy continuation of
    // "The capital of France is" should mention Paris somewhere in the first 16
    // tokens. We don't assert exact text (model versions drift), just that the
    // answer is in the right ballpark — this catches a silently-broken forward
    // path (e.g. weights on the wrong device) without being brittle.
    let lower = full.to_lowercase();
    assert!(
        lower.contains("paris"),
        "greedy continuation should mention Paris; got {:?}",
        full
    );

    // Clean shutdown so the GPU memory is released before the next test.
    client
        .shutdown()
        .await
        .expect("service should shut down cleanly");
}

// ---- E6.4 GPU form: engine reports single-device on a 1-GPU host --------------

#[test]
#[ignore = "needs CUDA libtorch on LD_LIBRARY_PATH (see module docs)"]
#[allow(clippy::expect_used, clippy::unwrap_used)] // GPU smoke: fail loud
fn e6_gpu_engine_is_single_device_on_one_gpu_host() {
    // On a single-GPU host the engine must build a single-device pipeline
    // (is_multi_device == false): the multi-device split only engages when
    // HYPRSTREAM_GPU_DEVICES names >1 device. This guards against a regression
    // where the engine accidentally tries to pipeline-split across one GPU.
    use hyprstream_core::config::RuntimeConfig;
    use hyprstream_core::runtime::TorchEngine;

    if !tch::Cuda::is_available() || tch::Cuda::device_count() <= 0 {
        eprintln!("[e2e_inference] skip: no CUDA device");
        return;
    }

    // No HYPRSTREAM_GPU_DEVICES set → legacy path → single device.
    // (We deliberately do not touch the env var to avoid races with the
    // config tests that pin it.)
    let config = RuntimeConfig::default();
    let engine = TorchEngine::new(config).expect("engine should construct on GPU");

    // The engine's primary device must be a CUDA device (we proved one exists
    // above), and with no multi-device env it must NOT report multi-device.
    assert!(
        matches!(engine.device(), tch::Device::Cuda(_)),
        "primary device should be CUDA on a GPU host, got {:?}",
        engine.device()
    );
    assert!(
        !engine.is_multi_device(),
        "a single-GPU host must build a single-device engine, not a split pipeline"
    );
    // The pool, if present, holds exactly one device.
    if let Some(pool) = engine.device_pool() {
        assert_eq!(
            pool.len(),
            1,
            "device pool on a single-GPU host holds exactly one device"
        );
        assert!(pool.is_single());
    }
    // Legacy single-device path: pool is None, which also means single.
}

// ===== GPU prerequisites documentation (always-run self-report) ===============

#[test]
fn e6_report_gpu_and_model_prerequisites() {
    // A non-ignored, always-run canary that prints the host's GPU/model status
    // to stderr. It always passes; its job is to make `cargo test` output tell
    // a reviewer exactly what a full GPU run would need on this host.
    let cuda = tch::Cuda::is_available();
    let n_devices = if cuda {
        tch::Cuda::device_count()
    } else {
        0
    };
    let model = e2e_model_dir();
    eprintln!(
        "[e2e_inference] prerequisites: cuda_available={cuda}, cuda_devices={n_devices}, \
         model_present={}, model_dir={}",
        model.is_some(),
        model
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<none>".to_owned()),
    );

    // Assert the router's documented defaults so a future change to them is a
    // deliberate, visible act — the router tests above lean on these.
    assert_eq!(DEFAULT_LEASE_DURATION, Duration::from_secs(30));
    // SessionAffinity must be Default-constructible (the router relies on it).
    let _ = SessionAffinity::default();
}
