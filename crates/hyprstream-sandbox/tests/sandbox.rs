//! Integration tests for the Profile A capability sandbox (python guest).
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
//! tests SKIP locally, but FAIL under CI (`CI` env var set) — the security/DoS
//! guarantees must never go silently untested in the pipeline.
//!
//! The guest declares exactly one import, `env::host_random` (verified with
//! wasm-tools), and the host Linker defines exactly that function and traps every
//! other import. So:
//!   * case1 `print(1 + 1)` returns a CLEAN status 0 (no trap),
//!   * case2 `import os; os.system('echo PWNED')` is inert (python exception, never
//!     a host process — there is no syscall surface),
//!   * the fuel and epoch DoS guards trap as expected.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use hyprstream_sandbox::python::PyResult;
use hyprstream_sandbox::{EpochTimer, Sandbox, Subject};

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

/// CI guard: in CI the guest artifact MUST be present so the security/DoS guarantees
/// are actually exercised — a missing artifact is a hard failure, never a silent skip.
/// Locally (no `CI` env var) the test still skips when the artifact is absent.
///
/// Returns `Some(wasm)` when the artifact is available, or `None` to signal the
/// caller to `return` early (local skip). Panics under CI when the artifact is
/// missing so the pipeline cannot let the guarantees go untested.
fn guest_wasm_or_ci_fail(label: &str) -> Option<Vec<u8>> {
    match guest_wasm() {
        Some(wasm) => Some(wasm),
        None => {
            assert!(
                std::env::var("CI").is_err(),
                "{label}: guest wasm not built but running under CI — build the pyguest \
                 (cargo build --release --target wasm32-unknown-unknown --manifest-path \
                 crates/hyprstream-wasm-pyguest/Cargo.toml) and set HYPRSTREAM_PYGUEST_WASM"
            );
            eprintln!("SKIP {label}: guest wasm not built (set CI=1 to make this a failure)");
            None
        }
    }
}

/// Generous fuel: RustPython spends a LOT of instructions bootstrapping a fresh VM.
const BIG_FUEL: u64 = 50_000_000_000;

/// The Linker must accept ANY guest module and expose exactly one host import,
/// trapping everything else. Loading must succeed.
#[test]
fn sandbox_loads_with_single_capability() {
    let Some(wasm) = guest_wasm_or_ci_fail("sandbox_loads") else {
        return;
    };
    Sandbox::from_bytes(&wasm).expect("sandbox must load any guest with the 1-capability linker");
}

/// Case 1: `print(1 + 1)` — the green path. With rustpython-vm 0.5 + frozen stdlib
/// the VM boots and runs, so this returns a CLEAN status 0 (no trap, no host effect).
#[test]
fn case1_arithmetic_is_green() {
    let Some(wasm) = guest_wasm_or_ci_fail("case1") else {
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
    let Some(wasm) = guest_wasm_or_ci_fail("case2") else {
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
    let Some(wasm) = guest_wasm_or_ci_fail("case3") else {
        return;
    };
    let sandbox = Sandbox::from_bytes(&wasm).expect("load sandbox");
    // Boot the VM with ample fuel, then loop — but cap total fuel so the loop trips it.
    let result = sandbox.eval("while True:\n    pass\n", BIG_FUEL);
    assert!(
        result.is_err(),
        "unbounded guest must trap on fuel, got Ok({result:?})"
    );
    eprintln!(
        "case3: fuel-bounded — guest trapped: {:?}",
        result.unwrap_err()
    );
}

/// Subject scoping: two sandboxes bound to DIFFERENT subjects keep independent
/// Store identity — the subject set on one never leaks into the other. This is the
/// no-global/no-thread-local invariant that killed the native #488 design.
#[test]
fn subject_isolation_no_leak() {
    let alice = Subject::new("alice");
    let bob = Subject::new("bob");

    // Bind two sandboxes to different subjects (no guest wasm needed for identity).
    let Some(wasm) = guest_wasm_or_ci_fail("subject_isolation") else {
        // Even without the guest we can assert the Subject binding is per-sandbox.
        assert_ne!(alice, bob);
        assert_eq!(alice.name(), Some("alice"));
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
    let Some(wasm) = guest_wasm_or_ci_fail("epoch") else {
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
    let Some(wasm) = guest_wasm_or_ci_fail("epoch0") else {
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

// ─────────────────────────────────────────────────────────────────────────────
// #483 P2 — /lang/python semantics over the persistent guest shell.
// ─────────────────────────────────────────────────────────────────────────────

/// Generous per-call fuel for the persistent shell. The interpreter is built once
/// (on the first op), so subsequent ops are cheaper, but bootstrap is expensive.
const SHELL_FUEL: u64 = 50_000_000_000;

/// eval: an expression returns its repr; exec: statements capture stdout; and the
/// interpreter scope PERSISTS across calls (a var set by exec is visible later).
#[test]
fn pyshell_eval_exec_and_persistent_scope() {
    let Some(wasm) = guest_wasm_or_ci_fail("pyshell") else {
        return;
    };
    let sandbox = Sandbox::from_bytes(&wasm).expect("load sandbox");
    let mut shell = sandbox.open_shell(SHELL_FUEL).expect("open shell");

    // eval an expression -> repr.
    let r = shell.eval("2 + 3").expect("eval 2+3");
    assert_eq!(r, PyResult::Ok("5".to_owned()), "eval should repr to 5");

    // exec stdout capture (pure-Python __Capture surrogate).
    let r = shell.exec("print('hello')").expect("exec print");
    assert_eq!(
        r,
        PyResult::Ok("hello\n".to_owned()),
        "exec should capture stdout"
    );

    // PERSISTENT scope: a var set in one exec is visible in a later eval.
    let _ = shell.exec("x = 41").expect("exec assign");
    let r = shell.eval("x + 1").expect("eval persisted var");
    assert_eq!(
        r,
        PyResult::Ok("42".to_owned()),
        "interpreter scope must persist across calls"
    );
    eprintln!("pyshell: eval/exec/persistent-scope all green");
}

/// vars/ enumerates non-dunder globals; defs/ enumerates callables only; get_var /
/// get_def return the repr of one named global (None if absent).
#[test]
fn pyshell_vars_and_defs() {
    let Some(wasm) = guest_wasm_or_ci_fail("pyshell_vars_defs") else {
        return;
    };
    let sandbox = Sandbox::from_bytes(&wasm).expect("load sandbox");
    let mut shell = sandbox.open_shell(SHELL_FUEL).expect("open shell");

    // Define a var and a function.
    let _ = shell.exec("answer = 42").expect("set var");
    let _ = shell
        .exec("def greet(n):\n    return 'hi ' + n\n")
        .expect("def func");

    // vars/ lists both names (no dunders).
    let vars = shell.list_vars().expect("list vars");
    assert!(vars.contains(&"answer".to_owned()), "vars: {vars:?}");
    assert!(vars.contains(&"greet".to_owned()), "vars: {vars:?}");
    assert!(
        !vars.iter().any(|v| v.starts_with("__")),
        "vars must exclude dunders: {vars:?}"
    );

    // defs/ lists ONLY the callable.
    let defs = shell.list_defs().expect("list defs");
    assert!(defs.contains(&"greet".to_owned()), "defs: {defs:?}");
    assert!(
        !defs.contains(&"answer".to_owned()),
        "defs must exclude non-callables: {defs:?}"
    );

    // get_var / get_def repr.
    assert_eq!(
        shell.get_var("answer").expect("get_var"),
        PyResult::Ok("42".to_owned())
    );
    assert!(matches!(
        shell.get_def("greet").expect("get_def"),
        PyResult::Ok(_)
    ));
    // Absent name -> None.
    assert_eq!(
        shell.get_var("nope").expect("get_var absent"),
        PyResult::None
    );
    // A non-callable is not a def.
    assert_eq!(
        shell.get_def("answer").expect("get_def non-callable"),
        PyResult::None
    );
    eprintln!("pyshell: vars/defs/get_* all green");
}

// ─────────────────────────────────────────────────────────────────────────────
// #483 P2 — REAL VFS capability: guest -> host vfs_* -> Subject-scoped proxy ->
// in-memory Namespace. Deliverable (1): the proof the capability is real.
// ─────────────────────────────────────────────────────────────────────────────

mod vfs_e2e {
    use super::*;
    use hyprstream_vfs::proxy::spawn_vfs_proxy;
    use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Namespace, Stat};
    use std::collections::HashMap;
    use std::sync::Arc;

    /// In-memory mount; a `denied` Subject is refused at every op.
    struct MemMount {
        files: parking_lot::Mutex<HashMap<String, Vec<u8>>>,
    }
    struct MemFid {
        path: String,
        wbuf: parking_lot::Mutex<Vec<u8>>,
    }
    impl MemMount {
        fn new(files: Vec<(&str, &[u8])>) -> Self {
            Self {
                files: parking_lot::Mutex::new(
                    files
                        .into_iter()
                        .map(|(k, v)| (k.to_owned(), v.to_vec()))
                        .collect(),
                ),
            }
        }
        fn check(c: &Subject) -> Result<(), MountError> {
            if c.name() == Some("denied") {
                Err(MountError::PermissionDenied("denied".into()))
            } else {
                Ok(())
            }
        }
    }
    #[async_trait::async_trait]
    impl Mount for MemMount {
        async fn walk(&self, comps: &[&str], c: &Subject) -> Result<Fid, MountError> {
            Self::check(c)?;
            Ok(Fid::new(MemFid {
                path: comps.join("/"),
                wbuf: parking_lot::Mutex::new(Vec::new()),
            }))
        }
        async fn open(&self, _f: &mut Fid, _m: u8, c: &Subject) -> Result<(), MountError> {
            Self::check(c)
        }
        async fn read(
            &self,
            f: &Fid,
            off: u64,
            _n: u32,
            c: &Subject,
        ) -> Result<Vec<u8>, MountError> {
            Self::check(c)?;
            let i = f.downcast_ref::<MemFid>().unwrap();
            let d = self
                .files
                .lock()
                .get(&i.path)
                .cloned()
                .ok_or_else(|| MountError::NotFound(i.path.clone()))?;
            let s = (off as usize).min(d.len());
            Ok(d[s..].to_vec())
        }
        async fn write(
            &self,
            f: &Fid,
            _o: u64,
            data: &[u8],
            c: &Subject,
        ) -> Result<u32, MountError> {
            Self::check(c)?;
            let i = f.downcast_ref::<MemFid>().unwrap();
            i.wbuf.lock().extend_from_slice(data);
            self.files
                .lock()
                .insert(i.path.clone(), i.wbuf.lock().clone());
            Ok(data.len() as u32)
        }
        async fn readdir(&self, f: &Fid, c: &Subject) -> Result<Vec<DirEntry>, MountError> {
            Self::check(c)?;
            let i = f.downcast_ref::<MemFid>().unwrap();
            let prefix = if i.path.is_empty() {
                String::new()
            } else {
                format!("{}/", i.path)
            };
            let mut out = Vec::new();
            for k in self.files.lock().keys() {
                if let Some(rest) = k.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        out.push(DirEntry {
                            name: rest.to_owned(),
                            is_dir: false,
                            size: 0,
                            stat: None,
                        });
                    }
                }
            }
            Ok(out)
        }
        async fn stat(&self, f: &Fid, c: &Subject) -> Result<Stat, MountError> {
            Self::check(c)?;
            let i = f.downcast_ref::<MemFid>().unwrap();
            Ok(Stat {
                qtype: 0,
                size: 0,
                name: i.path.clone(),
                mtime: 0,
            })
        }
        async fn clunk(&self, _f: Fid, _c: &Subject) {}
    }

    fn make_ns() -> Arc<Namespace> {
        let mut ns = Namespace::new();
        ns.mount("/config", Arc::new(MemMount::new(vec![("temp", b"0.7")])))
            .unwrap();
        Arc::new(ns)
    }

    /// A GUEST `vfs_probe` call (op=cat/echo) goes through the host `env::vfs_*` fn,
    /// the Subject-scoped proxy, and the in-memory Mount. An allowed Subject can
    /// read/write; a `denied` Subject is refused — proving Subject-scoping is
    /// enforced at the backend and the guest cannot forge identity.
    #[test]
    fn guest_vfs_is_subject_scoped() {
        let Some(wasm) = guest_wasm_or_ci_fail("guest_vfs") else {
            return;
        };
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let ns = make_ns();
        let alice = Subject::new("alice");
        let denied = Subject::new("denied");

        let alice_tx = rt.block_on(async { spawn_vfs_proxy(Arc::clone(&ns), alice.clone()) });
        let denied_tx = rt.block_on(async { spawn_vfs_proxy(Arc::clone(&ns), denied.clone()) });

        let alice_h = hyprstream_sandbox::vfs::VfsProxyHandle::new(alice_tx, alice.clone());
        let denied_h = hyprstream_sandbox::vfs::VfsProxyHandle::new(denied_tx, denied.clone());

        let sb_alice = Sandbox::from_bytes_for(&wasm, alice)
            .expect("load alice")
            .with_vfs(alice_h);
        let sb_denied = Sandbox::from_bytes_for(&wasm, denied)
            .expect("load denied")
            .with_vfs(denied_h);

        // blocking_send must NOT run on a tokio worker — drive from a plain thread.
        let h = std::thread::spawn(move || {
            // op 0 = cat. Allowed subject reads an existing file THROUGH THE GUEST.
            let r = sb_alice
                .probe_vfs(0, "/config/temp", b"")
                .expect("alice cat");
            assert!(
                r.ok,
                "alice cat should succeed: {:?}",
                String::from_utf8_lossy(&r.body)
            );
            assert_eq!(r.body, b"0.7");

            // op 2 = echo, then cat back: write persists through the Mount.
            let w = sb_alice
                .probe_vfs(2, "/config/temp", b"0.9")
                .expect("alice echo");
            assert!(w.ok, "alice echo should succeed");
            let r2 = sb_alice
                .probe_vfs(0, "/config/temp", b"")
                .expect("alice cat2");
            assert!(r2.ok);
            assert_eq!(r2.body, b"0.9");

            // Denied subject is refused at the Mount — Subject reached the backend.
            let d = sb_denied
                .probe_vfs(0, "/config/temp", b"")
                .expect("denied cat");
            assert!(!d.ok, "denied subject must be refused");
            assert!(
                String::from_utf8_lossy(&d.body).contains("permission denied"),
                "got: {}",
                String::from_utf8_lossy(&d.body)
            );
            eprintln!(
                "guest_vfs: guest->host->proxy->Mount is Subject-scoped (alice ok, denied refused)"
            );
        });
        h.join().expect("vfs thread");
    }

    /// A sandbox with NO vfs capability granted: the guest `vfs_*` call returns the
    /// deny-by-default reply (no capability), never reaching any Namespace.
    #[test]
    fn guest_vfs_deny_by_default() {
        let Some(wasm) = guest_wasm_or_ci_fail("guest_vfs_deny") else {
            return;
        };
        // No .with_vfs() -> no capability.
        let sb = Sandbox::from_bytes_for(&wasm, Subject::new("alice")).expect("load");
        let h = std::thread::spawn(move || {
            let r = sb.probe_vfs(0, "/config/temp", b"").expect("probe");
            assert!(!r.ok, "no-capability sandbox must deny");
            assert!(
                String::from_utf8_lossy(&r.body).contains("no capability"),
                "got: {}",
                String::from_utf8_lossy(&r.body)
            );
        });
        h.join().expect("deny thread");
    }
}
