//! Integration tests for the persistent [`PythonShell`] over the wasm python guest.
//!
//! These mirror the `/lang/python` semantics (eval/exec/persistent-scope, vars/defs)
//! that previously lived in `hyprstream-workers-wasmtime`'s `tests/sandbox.rs` before
//! the python shell was extracted into this crate.
//!
//! The guest wasm artifact must be built first:
//!   cargo build --release --target wasm32-unknown-unknown \
//!     --manifest-path crates/hyprstream-workers-python-guest/Cargo.toml
//!
//! Set HYPRSTREAM_PYGUEST_WASM to override the path. If the artifact is absent the
//! tests SKIP locally, but FAIL under CI (`CI` env var set).

use std::path::PathBuf;

use hyprstream_workers_python::{PyResult, PythonShell};
use hyprstream_workers_wasmtime::Sandbox;

/// Generous per-call fuel for the persistent shell. The interpreter is built once
/// (on the first op), so subsequent ops are cheaper, but bootstrap is expensive.
const SHELL_FUEL: u64 = 50_000_000_000;

fn guest_wasm() -> Option<Vec<u8>> {
    if let Ok(p) = std::env::var("HYPRSTREAM_PYGUEST_WASM") {
        return std::fs::read(&p).ok();
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let guest_dir = manifest
        .parent()
        .map(|p| p.join("hyprstream-workers-python-guest"))?;
    for profile in ["release", "debug"] {
        let candidate = guest_dir
            .join("target/wasm32-unknown-unknown")
            .join(profile)
            .join("hyprstream_workers_python_guest.wasm");
        if candidate.exists() {
            if let Ok(bytes) = std::fs::read(&candidate) {
                return Some(bytes);
            }
        }
    }
    None
}

/// CI guard: under CI a missing pyguest artifact is a hard failure (the shell
/// guarantees must be exercised); locally it still skips.
fn guest_wasm_or_ci_fail(label: &str) -> Option<Vec<u8>> {
    match guest_wasm() {
        Some(wasm) => Some(wasm),
        None => {
            assert!(
                std::env::var("CI").is_err(),
                "{label}: pyguest wasm not built but running under CI — build the pyguest \
                 (cargo build --release --target wasm32-unknown-unknown --manifest-path \
                 crates/hyprstream-workers-python-guest/Cargo.toml) and set HYPRSTREAM_PYGUEST_WASM"
            );
            eprintln!("SKIP {label}: pyguest wasm not built (set CI=1 to make this a failure)");
            None
        }
    }
}

/// eval: an expression returns its repr; exec: statements capture stdout; and the
/// interpreter scope PERSISTS across calls (a var set by exec is visible later).
#[test]
fn pyshell_eval_exec_and_persistent_scope() {
    let Some(wasm) = guest_wasm_or_ci_fail("pyshell") else {
        return;
    };
    let sandbox = Sandbox::from_bytes(&wasm).expect("load sandbox");
    let mut shell = PythonShell::open(&sandbox, SHELL_FUEL).expect("open shell");

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
    let mut shell = PythonShell::open(&sandbox, SHELL_FUEL).expect("open shell");

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
