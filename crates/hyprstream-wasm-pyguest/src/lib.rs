//! RustPython guest for the #505/#483 capability sandbox.
//!
//! Compiled to `wasm32-unknown-unknown` (NOT wasi). The only host capabilities the
//! guest can reach are the imports the wasmtime `Linker` defines under `env`:
//!   * `host_random(ptr,len)`           — entropy (the #505 P1 capability)
//!   * `vfs_cat/vfs_ls/vfs_echo/vfs_ctl` — Subject-scoped VFS access (#483 P2)
//! Anything else (e.g. `import os; os.system(...)`) has nothing to call — inert by
//! construction.
//!
//! Build (MUST run from THIS crate's directory):
//!   (cd crates/hyprstream-wasm-pyguest && cargo build --release --target wasm32-unknown-unknown)
//!
//! The `getrandom_backend="custom"` rustflag lives in this crate's `.cargo/config.toml`,
//! and Cargo only discovers `.cargo/config.toml` from the CWD (walking up), NOT from
//! `--manifest-path`. Building with `--manifest-path …pyguest/Cargo.toml` from the repo
//! root drops the cfg and the build FAILS on getrandom/wasm32 (enable wasm_js error).
//!
//! ## ABI
//!
//! Exports for the host:
//!   - `alloc(len) -> *mut u8`  : host allocates guest memory to write input into
//!   - `dealloc(ptr, len)`      : free guest memory
//!   - `eval(ptr, len) -> i32`  : LEGACY #505 entry — run UTF-8 source in a FRESH
//!     interpreter, 0 = ok, nonzero = error. Kept so the #505 host tests still pass.
//!   - `py_op(op, ptr, len) -> i64` : #483 entry — run an operation against the
//!     PERSISTENT interpreter, returning a packed `(out_ptr<<32)|out_len` into guest
//!     memory (the host reads it, then calls `dealloc(out_ptr, out_len)`). The first
//!     byte of the output is a STATUS tag (`OK`/`ERR`/`NONE`), the rest is the payload
//!     (UTF-8). See [`Op`].
//!
//! ## #483 semantics (ported from the native `/lang/python` mount, re-expressed in 0.5)
//!
//!   * Op::Eval  — evaluate an expression -> `repr(result)` (captured stdout is
//!                 returned too via the EVAL_STDOUT side channel; see `py_op`).
//!   * Op::Exec  — execute statements -> captured stdout.
//!   * Op::ListVars / Op::ListDefs — newline-joined non-dunder global names
//!                 (defs = callables only).
//!   * Op::GetVar / Op::GetDef     — `repr` of one named global (NONE if absent).
//!
//! The interpreter + its globals dict are PERSISTENT across `py_op` calls (held in a
//! thread-local `Rc`), so user state survives — exactly like the native shell's
//! persistent `globals`. wasm32 is single-threaded, so the thread-local is the whole
//! guest's state.

use rustpython_vm as vm;
use std::cell::RefCell;
use vm::builtins::{PyDictRef, PyStr};
use vm::scope::Scope;
use vm::{AsObject, Interpreter};

// ---------------------------------------------------------------------------
// Host imports (resolved by the wasmtime Linker in the host crate).
// ---------------------------------------------------------------------------
extern "C" {
    /// Entropy: fill `len` bytes at guest `ptr` with random data.
    fn host_random(ptr: *mut u8, len: usize);

    // #483 VFS capability host fns. Each takes a path (and, for writes, a body) in
    // guest memory and writes a reply into a host-allocated guest buffer, returning
    // a packed `(reply_ptr<<32)|reply_len`. The reply's first byte is a status tag
    // (0 = ok, 1 = err), the rest is UTF-8. The HOST scopes every call to the bound
    // Subject — the guest cannot supply or forge an identity.
    fn vfs_cat(path_ptr: *const u8, path_len: usize) -> i64;
    fn vfs_ls(path_ptr: *const u8, path_len: usize) -> i64;
    fn vfs_echo(path_ptr: *const u8, path_len: usize, data_ptr: *const u8, data_len: usize) -> i64;
    fn vfs_ctl(path_ptr: *const u8, path_len: usize, cmd_ptr: *const u8, cmd_len: usize) -> i64;
}

/// getrandom 0.3 custom backend (selected by `--cfg getrandom_backend="custom"`).
///
/// # Safety
/// getrandom 0.3 passes a valid writable `dest`/`len`.
#[no_mangle]
unsafe fn __getrandom_v03_custom(dest: *mut u8, len: usize) -> Result<(), getrandom::Error> {
    let buf = std::slice::from_raw_parts_mut(dest, len);
    host_random(buf.as_mut_ptr(), buf.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// Memory ABI helpers.
// ---------------------------------------------------------------------------

/// Allocate `len` bytes in guest memory and return the pointer.
///
/// # Safety
/// The host must later call `dealloc` with the same `ptr`/`len`.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Free memory previously returned by [`alloc`].
///
/// # Safety
/// `ptr`/`len` must come from a prior `alloc` call.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len != 0 {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

/// Pack a freshly-allocated `(ptr, len)` into the `i64` ABI the host expects.
/// The host reads `len` bytes at `ptr`, then calls `dealloc(ptr, len)`.
fn pack_owned(bytes: Vec<u8>) -> i64 {
    let len = bytes.len();
    // Round-trip through `alloc` so the host's `dealloc(ptr,len)` matches the
    // allocation (Vec::with_capacity(len) backing, like `alloc`).
    let ptr = alloc(len);
    // SAFETY: `ptr` has capacity `len`; copy the payload in.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len);
    }
    ((ptr as u64) << 32 | len as u64) as i64
}

/// Read a `(ptr,len)` UTF-8 string supplied by the host (lossy).
///
/// # Safety
/// `ptr`..`ptr+len` must be a valid readable region.
unsafe fn read_str(ptr: *const u8, len: usize) -> String {
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned()
}

// ---------------------------------------------------------------------------
// Status tags for the py_op reply payload.
// ---------------------------------------------------------------------------
const TAG_OK: u8 = 0;
const TAG_ERR: u8 = 1;
const TAG_NONE: u8 = 2;

fn reply(tag: u8, body: impl AsRef<[u8]>) -> i64 {
    let mut out = Vec::with_capacity(1 + body.as_ref().len());
    out.push(tag);
    out.extend_from_slice(body.as_ref());
    pack_owned(out)
}

// ---------------------------------------------------------------------------
// Op codes for py_op.
// ---------------------------------------------------------------------------

/// Operations the host can drive against the persistent interpreter.
#[repr(i32)]
enum Op {
    Eval = 0,
    Exec = 1,
    ListVars = 2,
    GetVar = 3,
    ListDefs = 4,
    GetDef = 5,
}

impl Op {
    fn from_i32(v: i32) -> Option<Self> {
        Some(match v {
            0 => Op::Eval,
            1 => Op::Exec,
            2 => Op::ListVars,
            3 => Op::GetVar,
            4 => Op::ListDefs,
            5 => Op::GetDef,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// Persistent interpreter state (single-threaded wasm => thread_local is global).
// ---------------------------------------------------------------------------

struct Shell {
    interp: Interpreter,
    /// Persistent globals reused across every call so user state (vars, funcs)
    /// survives — the same design as the native shell's `globals` dict.
    globals: PyDictRef,
}

thread_local! {
    static SHELL: RefCell<Option<Shell>> = const { RefCell::new(None) };
}

/// Build (or reuse) the persistent shell.
fn with_shell<R>(f: impl FnOnce(&Shell) -> R) -> R {
    SHELL.with(|cell| {
        {
            let mut b = cell.borrow_mut();
            if b.is_none() {
                let interp = vm::Interpreter::builder(vm::Settings::default())
                    .add_frozen_modules(rustpython_pylib::FROZEN_STDLIB)
                    .build();
                let globals = interp.enter(|vm| {
                    // Install the safe sink stdout baseline (never None, never the
                    // host process stdout) — same posture as the native shell.
                    let scope = vm.new_scope_with_builtins();
                    let _ = vm.run_string(scope, SINK_SETUP, "<sink>".to_owned());
                    vm.ctx.new_dict()
                });
                *b = Some(Shell { interp, globals });
            }
        }
        let b = cell.borrow();
        // SAFETY: just ensured Some above; no re-entrant borrow_mut while held.
        f(b.as_ref().expect("shell initialised"))
    })
}

impl Shell {
    /// A scope over the persistent globals for one execution.
    fn scope(&self, vm: &vm::VirtualMachine) -> Scope {
        Scope::with_builtins(None, self.globals.clone(), vm)
    }
}

// ---------------------------------------------------------------------------
// Pure-Python stdout capture (ported from native mount.rs; no io.StringIO so we
// avoid extra frozen-module init under the sandbox stdio path).
// ---------------------------------------------------------------------------

const CAPTURE_SETUP: &str = r"
import sys as _sys
class __Capture:
    def __init__(self): self.__buf = []
    def write(self, s): self.__buf.append(str(s))
    def flush(self): pass
    def getvalue(self): return ''.join(self.__buf)
_sys.stdout = __Capture()
del _sys
";

const CAPTURE_DRAIN: &str = r"
import sys as _sys
__v = _sys.stdout.getvalue() if hasattr(_sys.stdout, 'getvalue') else ''
class __Sink:
    def write(self, s): pass
    def flush(self): pass
_sys.stdout = __Sink()
del _sys
__v
";

const SINK_SETUP: &str = r"
import sys as _sys
class __Sink:
    def write(self, s): pass
    def flush(self): pass
_sys.stdout = __Sink()
del _sys
";

/// Redirect `sys.stdout` to a pure-Python capture object. Returns whether it took.
fn setup_capture(vm: &vm::VirtualMachine) -> bool {
    let scope = vm.new_scope_with_builtins();
    let ok = vm
        .run_string(scope, CAPTURE_SETUP, "<capture>".to_owned())
        .is_ok();
    if !ok {
        let scope = vm.new_scope_with_builtins();
        let _ = vm.run_string(scope, SINK_SETUP, "<sink>".to_owned());
    }
    ok
}

/// Read captured stdout and restore a safe sink. Empty if capture wasn't active.
fn drain_capture(vm: &vm::VirtualMachine, active: bool) -> String {
    if !active {
        return String::new();
    }
    let scope = vm.new_scope_with_builtins();
    vm.run_block_expr(scope, CAPTURE_DRAIN)
        .ok()
        .and_then(|obj| obj.str(vm).ok().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Operation impls (ported behaviour from native lib.rs, re-expressed on 0.5).
// ---------------------------------------------------------------------------

/// Eval an expression. Reply OK body = `repr` then (if any) `\n---\n<stdout>`.
fn op_eval(code: &str) -> i64 {
    with_shell(|shell| {
        shell.interp.enter(|vm| {
            let active = setup_capture(vm);
            let scope = shell.scope(vm);
            let run = vm.run_block_expr(scope, code);
            let stdout = drain_capture(vm, active);
            match run {
                Ok(obj) => {
                    let repr = if vm.is_none(&obj) {
                        String::new()
                    } else {
                        obj.repr(vm)
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|_| "<repr error>".to_owned())
                    };
                    let body = if stdout.is_empty() {
                        repr
                    } else if repr.is_empty() {
                        stdout
                    } else {
                        format!("{repr}\n---\n{stdout}")
                    };
                    reply(TAG_OK, body)
                }
                Err(exc) => reply(TAG_ERR, exc_message(vm, &exc)),
            }
        })
    })
}

/// Exec statements. Reply OK body = captured stdout.
fn op_exec(code: &str) -> i64 {
    with_shell(|shell| {
        shell.interp.enter(|vm| {
            let active = setup_capture(vm);
            let scope = shell.scope(vm);
            let run = vm.run_string(scope, code, "<guest>".to_owned());
            let stdout = drain_capture(vm, active);
            match run {
                Ok(_) => reply(TAG_OK, stdout),
                Err(exc) => reply(TAG_ERR, exc_message(vm, &exc)),
            }
        })
    })
}

fn exc_message(vm: &vm::VirtualMachine, exc: &vm::PyRef<vm::builtins::PyBaseException>) -> String {
    exc.as_object()
        .to_owned()
        .str(vm)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "<exception>".to_owned())
}

fn is_dunder(name: &str) -> bool {
    name.starts_with("__") && name.ends_with("__")
}

/// Newline-joined non-dunder globals (callables only if `callables_only`).
fn op_list(callables_only: bool) -> i64 {
    with_shell(|shell| {
        shell.interp.enter(|_vm| {
            let names: Vec<String> = shell
                .globals
                .clone()
                .into_iter()
                .filter_map(|(k, v)| {
                    let name = k.downcast_ref::<PyStr>()?.to_str()?.to_owned();
                    if is_dunder(&name) {
                        return None;
                    }
                    if callables_only && !v.is_callable() {
                        return None;
                    }
                    Some(name)
                })
                .collect();
            reply(TAG_OK, names.join("\n"))
        })
    })
}

/// `repr` of one named global. NONE if absent (or not callable, for defs).
fn op_get(name: &str, callable_only: bool) -> i64 {
    with_shell(|shell| {
        shell.interp.enter(|vm| {
            match shell.globals.get_item(name, vm) {
                Ok(obj) => {
                    if callable_only && !obj.is_callable() {
                        return reply(TAG_NONE, "");
                    }
                    match obj.repr(vm) {
                        Ok(s) => reply(TAG_OK, s.to_string_lossy().into_owned()),
                        Err(_) => reply(TAG_NONE, ""),
                    }
                }
                Err(_) => reply(TAG_NONE, ""),
            }
        })
    })
}

// ---------------------------------------------------------------------------
// Exports.
// ---------------------------------------------------------------------------

/// #483 entry: run `op` against the persistent interpreter.
///
/// `ptr`/`len` is the UTF-8 argument (source for eval/exec, a name for get_*,
/// ignored for list_*). Returns a packed `(out_ptr<<32)|out_len`; the host reads
/// it then `dealloc`s. Output byte[0] is a status tag.
///
/// # Safety
/// `ptr`..`ptr+len` must be a valid readable region of guest memory.
#[no_mangle]
pub unsafe extern "C" fn py_op(op: i32, ptr: *const u8, len: usize) -> i64 {
    let arg = read_str(ptr, len);
    match Op::from_i32(op) {
        Some(Op::Eval) => op_eval(&arg),
        Some(Op::Exec) => op_exec(&arg),
        Some(Op::ListVars) => op_list(false),
        Some(Op::GetVar) => op_get(&arg, false),
        Some(Op::ListDefs) => op_list(true),
        Some(Op::GetDef) => op_get(&arg, true),
        None => reply(TAG_ERR, "unknown op"),
    }
}

/// LEGACY #505 entry: interpret UTF-8 source in a FRESH interpreter (no persistent
/// state). Returns 0 on success, nonzero on error. Kept so the original #505 host
/// tests (`case1_arithmetic`, `case2_os_system_is_inert`, …) still pass unchanged.
///
/// # Safety
/// `ptr`..`ptr+len` must be a valid readable region holding UTF-8 source.
#[no_mangle]
pub unsafe extern "C" fn eval(ptr: *const u8, len: usize) -> i32 {
    let src = match std::str::from_utf8(std::slice::from_raw_parts(ptr, len)) {
        Ok(s) => s,
        Err(_) => return 2,
    };
    let interp = vm::Interpreter::builder(vm::Settings::default())
        .add_frozen_modules(rustpython_pylib::FROZEN_STDLIB)
        .build();
    interp.enter(|vm| {
        let scope = vm.new_scope_with_builtins();
        let code = match vm.compile(src, vm::compiler::Mode::Exec, "<guest>".to_owned()) {
            Ok(c) => c,
            Err(_) => return 3,
        };
        match vm.run_code_obj(code, scope) {
            Ok(_) => 0,
            Err(_) => 1,
        }
    })
}

// ---------------------------------------------------------------------------
// VFS builtins (#483): expose the host vfs_* capabilities to guest Python.
//
// These are thin wrappers so guest scripts can do `cat("/config/x")` etc. The
// HOST scopes every call to the bound Subject; the guest never names an identity.
// Reply ABI: first byte status (0 ok / 1 err), rest UTF-8.
// ---------------------------------------------------------------------------

/// Read a host vfs_* packed reply `(ptr<<32)|len` back into `(status, body)`.
///
/// # Safety
/// `packed` must be a value returned by one of the `vfs_*` host imports; the host
/// allocated the buffer via the guest `alloc`, so we own and `dealloc` it here.
unsafe fn take_reply(packed: i64) -> (u8, String) {
    let p = packed as u64;
    let ptr = (p >> 32) as usize as *mut u8;
    let len = (p & 0xffff_ffff) as usize;
    if ptr.is_null() || len == 0 {
        return (TAG_ERR, String::new());
    }
    let slice = std::slice::from_raw_parts(ptr, len);
    let tag = slice[0];
    let body = String::from_utf8_lossy(&slice[1..]).into_owned();
    dealloc(ptr, len);
    (tag, body)
}

/// #483 capability probe export: drive a VFS op THROUGH the guest so the
/// `env::vfs_*` host imports are reachable (not dead-code-eliminated) and the
/// host's Subject-scoping test can exercise a real guest->host->Namespace path.
///
/// `op`: 0=cat, 1=ls, 2=echo, 3=ctl. `ptr/len` = path. `dptr/dlen` = body (echo/ctl;
/// pass 0/0 for cat/ls). Returns the host's packed reply `(ptr<<32)|len` verbatim,
/// so the HOST reads the status byte + payload and `dealloc`s it. The guest does NOT
/// own the buffer here (the host allocated it and reads it back).
///
/// This is what makes deliverable (1) end-to-end real: a guest call lands in
/// `vfs_cat`/`vfs_echo`/… which the host resolves against the Subject-scoped proxy.
///
/// # Safety
/// `ptr`/`dptr` regions must be valid readable guest memory of the given lengths.
#[no_mangle]
pub unsafe extern "C" fn vfs_probe(
    op: i32,
    ptr: *const u8,
    len: usize,
    dptr: *const u8,
    dlen: usize,
) -> i64 {
    match op {
        0 => vfs_cat(ptr, len),
        1 => vfs_ls(ptr, len),
        2 => vfs_echo(ptr, len, dptr, dlen),
        3 => vfs_ctl(ptr, len, dptr, dlen),
        _ => 0,
    }
}

/// Helper the guest-side native builtins would call. Exposed for completeness; the
/// minimal P2 guest drives VFS via these free functions rather than registering
/// native Python builtins (which needs `host_env`/extra wiring we deliberately omit).
/// A follow-up can register these as `cat`/`ls`/`write`/`ctl` Python builtins.
pub fn guest_cat(path: &str) -> Result<String, String> {
    // SAFETY: calling the host import with a valid path slice; reply is owned.
    let (tag, body) = unsafe { take_reply(vfs_cat(path.as_ptr(), path.len())) };
    if tag == TAG_OK {
        Ok(body)
    } else {
        Err(body)
    }
}

pub fn guest_ls(path: &str) -> Result<String, String> {
    let (tag, body) = unsafe { take_reply(vfs_ls(path.as_ptr(), path.len())) };
    if tag == TAG_OK {
        Ok(body)
    } else {
        Err(body)
    }
}

pub fn guest_echo(path: &str, data: &str) -> Result<(), String> {
    let (tag, body) =
        unsafe { take_reply(vfs_echo(path.as_ptr(), path.len(), data.as_ptr(), data.len())) };
    if tag == TAG_OK {
        Ok(())
    } else {
        Err(body)
    }
}

pub fn guest_ctl(path: &str, cmd: &str) -> Result<String, String> {
    let (tag, body) =
        unsafe { take_reply(vfs_ctl(path.as_ptr(), path.len(), cmd.as_ptr(), cmd.len())) };
    if tag == TAG_OK {
        Ok(body)
    } else {
        Err(body)
    }
}
