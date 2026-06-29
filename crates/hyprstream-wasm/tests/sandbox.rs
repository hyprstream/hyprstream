//! Integration tests for the #505 capability sandbox (P1 host).
//!
//! These tests need the guest wasm artifact built first:
//!   cargo build --release --target wasm32-unknown-unknown \
//!     --manifest-path crates/hyprstream-wasm-pyguest/Cargo.toml
//!
//! The guest is an EXCLUDED, standalone package (own target dir), so its artifact
//! lands at:
//!   crates/hyprstream-wasm-pyguest/target/wasm32-unknown-unknown/{release,debug}/hyprstream_wasm_pyguest.wasm
//!
//! Set HYPRSTREAM_PYGUEST_WASM to override the path. If the artifact is absent the
//! tests SKIP (return early) rather than fail.
//!
//! P1 (rustpython-vm 0.5): the guest now ACTUALLY runs Python. The guest declares
//! exactly one import, `env::host_random` (verified with wasm-tools), and the host
//! Linker defines exactly that function and traps every other import. So:
//!   * case1 `print(1 + 1)` returns a CLEAN status 0 (no trap),
//!   * case2 `import os; os.system('echo PWNED')` is inert (python exception, never
//!     a host process — there is no syscall surface),
//!   * the fuel and epoch DoS guards trap as expected.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use hyprstream_wasm::{EpochTimer, Sandbox, Subject};

fn guest_wasm() -> Option<Vec<u8>> {
    if let Ok(p) = std::env::var("HYPRSTREAM_PYGUEST_WASM") {
        return std::fs::read(&p).ok();
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let guest_dir = manifest
        .parent()
        .map(|p| p.join("hyprstream-wasm-pyguest"))?;
    for profile in ["release", "debug"] {
        let candidate = guest_dir
            .join("target/wasm32-unknown-unknown")
            .join(profile)
            .join("hyprstream_wasm_pyguest.wasm");
        if candidate.exists() {
            if let Ok(bytes) = std::fs::read(&candidate) {
                return Some(bytes);
            }
        }
    }
    None
}

/// Generous fuel: RustPython spends a LOT of instructions bootstrapping a fresh VM.
const BIG_FUEL: u64 = 50_000_000_000;

/// The Linker must accept ANY guest module and expose exactly one host import,
/// trapping everything else. Loading must succeed.
#[test]
fn sandbox_loads_with_single_capability() {
    let Some(wasm) = guest_wasm() else {
        eprintln!("SKIP: guest wasm not built");
        return;
    };
    Sandbox::from_bytes(&wasm).expect("sandbox must load any guest with the 1-capability linker");
}

/// Case 1: `print(1 + 1)` — the green path. With rustpython-vm 0.5 + frozen stdlib
/// the VM boots and runs, so this returns a CLEAN status 0 (no trap, no host effect).
#[test]
fn case1_arithmetic_is_green() {
    let Some(wasm) = guest_wasm() else {
        eprintln!("SKIP case1: guest wasm not built");
        return;
    };
    let sandbox = Sandbox::from_bytes(&wasm).expect("load sandbox");
    let status = sandbox
        .eval("print(1 + 1)", BIG_FUEL)
        .expect("print(1+1) must NOT trap on the 0.5 green path");
    assert_eq!(status, 0, "print(1+1) must return status 0 (clean run)");
    eprintln!("case1: GREEN — print(1+1) returned status 0");
}

/// Case 2: `import os; os.system('echo PWNED')` — the capability proof.
///
/// The HARD guarantee is structural: the guest has NO syscall surface — only
/// `host_random`. The host Linker wires no process spawner, so `echo PWNED` cannot
/// execute. On the 0.5 green path the VM boots and `os.system` raises a Python
/// exception (no native backend), so `eval` returns a NONZERO status WITHOUT
/// trapping and without any host process. We assert it is inert: either a nonzero
/// python-error status or a trap, never status 0, and never a host effect.
#[test]
fn case2_os_system_is_inert() {
    let Some(wasm) = guest_wasm() else {
        eprintln!("SKIP case2: guest wasm not built");
        return;
    };
    let sandbox = Sandbox::from_bytes(&wasm).expect("load sandbox");
    match sandbox.eval("import os; os.system('echo PWNED')", BIG_FUEL) {
        Ok(0) => panic!("os.system must NOT succeed — there is no host syscall surface"),
        Ok(n) => eprintln!("case2: inert — python error status {n}; no host process possible"),
        Err(t) => eprintln!("case2: inert — trapped: {t}; no host process possible"),
    }
    // No host-side assertion is needed: the Linker defines exactly host_random and
    // traps everything else, so a host process is structurally unreachable.
}

/// Case 3 (deterministic DoS guard): a fuel-bounded `while True: pass` must TRAP
/// ("all fuel consumed") rather than hang the host.
#[test]
fn case3_infinite_loop_is_fuel_bounded() {
    let Some(wasm) = guest_wasm() else {
        eprintln!("SKIP case3: guest wasm not built");
        return;
    };
    let sandbox = Sandbox::from_bytes(&wasm).expect("load sandbox");
    // Boot the VM with ample fuel, then loop — but cap total fuel so the loop trips it.
    let result = sandbox.eval("while True:\n    pass\n", BIG_FUEL);
    assert!(
        result.is_err(),
        "unbounded guest must trap on fuel, got Ok({result:?})"
    );
    eprintln!("case3: fuel-bounded — guest trapped: {:?}", result.unwrap_err());
}

/// Subject scoping: two sandboxes bound to DIFFERENT subjects keep independent
/// Store identity — the subject set on one never leaks into the other. This is the
/// no-global/no-thread-local invariant that killed the native #488 design.
#[test]
fn subject_isolation_no_leak() {
    let alice = Subject::named("alice");
    let bob = Subject::named("bob");

    // Bind two sandboxes to different subjects (no guest wasm needed for identity).
    let Some(wasm) = guest_wasm() else {
        // Even without the guest we can assert the Subject binding is per-sandbox.
        eprintln!("SKIP subject (no wasm): asserting Subject newtype identity only");
        assert_ne!(alice, bob);
        assert_eq!(alice.id(), Some("alice"));
        return;
    };
    let sb_alice = Sandbox::from_bytes_for(&wasm, alice.clone()).expect("load alice");
    let sb_bob = Sandbox::from_bytes_for(&wasm, bob.clone()).expect("load bob");

    assert_eq!(sb_alice.subject(), &alice);
    assert_eq!(sb_bob.subject(), &bob);
    assert_ne!(sb_alice.subject(), sb_bob.subject());

    // Each eval builds a FRESH Store from the sandbox's own subject; running one
    // does not mutate the other's bound identity (no shared global state).
    let _ = sb_alice.eval("x = 1", BIG_FUEL);
    assert_eq!(sb_alice.subject(), &alice, "alice subject must be stable");
    assert_eq!(sb_bob.subject(), &bob, "bob subject must be unaffected");
    eprintln!("subject: isolated — alice={alice:?} bob={bob:?}, no leak");
}

/// Real wall-clock DoS bound: an `EpochTimer` advancing the engine epoch every 10ms,
/// plus a per-call epoch deadline, traps `while True: pass` as "interrupt" within
/// roughly the timeout. This is the PRODUCTION bound, not just the fuel path.
#[test]
fn epoch_wall_clock_bound_traps_runaway() {
    let Some(wasm) = guest_wasm() else {
        eprintln!("SKIP epoch: guest wasm not built");
        return;
    };
    let sandbox = Sandbox::from_bytes(&wasm).expect("load sandbox");

    // Spawn the timer on THIS sandbox's engine. 10ms cadence; deadline of a few
    // ticks gives a ~tens-of-ms wall-clock bound.
    let tick = Duration::from_millis(10);
    let _timer = EpochTimer::spawn(sandbox.engine(), tick);

    // Deadline = 20 ticks ≈ 200ms wall clock. Generous enough that VM bootstrap
    // (which the epoch is also counting) completes, but the infinite loop trips it.
    let start = Instant::now();
    let result = sandbox.eval_with_epoch_deadline("while True:\n    pass\n", 20);
    let elapsed = start.elapsed();

    assert!(
        result.is_err(),
        "runaway guest must trap on the epoch deadline, got Ok({result:?})"
    );
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("interrupt"),
        "epoch trap should be an interrupt, got: {err}"
    );
    // Sanity: it should not run for many seconds. Generous upper bound for CI.
    assert!(
        elapsed < Duration::from_secs(10),
        "epoch bound did not fire promptly: {elapsed:?}"
    );
    eprintln!("epoch: wall-clock bound fired after {elapsed:?} — {err}");
}

/// The epoch limiter is live even with a deadline of 0 (default-epoch store is
/// already over the line) — proves the limiter is wired, no timer needed.
#[test]
fn epoch_deadline_zero_traps_immediately() {
    let Some(wasm) = guest_wasm() else {
        eprintln!("SKIP epoch0: guest wasm not built");
        return;
    };
    let sandbox = Sandbox::from_bytes(&wasm).expect("load sandbox");
    let result = sandbox.eval_with_epoch_deadline("print(1)", 0);
    assert!(
        result.is_err(),
        "epoch deadline of 0 must trap immediately, got Ok({result:?})"
    );
    eprintln!("epoch0: trapped as expected: {:?}", result.unwrap_err());
}
