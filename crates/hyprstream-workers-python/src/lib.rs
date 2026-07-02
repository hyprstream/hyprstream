//! RustPython engine + `/lang/python` 9P shell for the hyprstream VFS namespace.
//!
//! This crate is the ONE python-aware layer over the generic, language-agnostic
//! wasm sandbox engine ([`hyprstream_workers_wasmtime`]). It runs untrusted Python on
//! the RustPython guest (`hyprstream-workers-python-guest`, compiled to
//! `wasm32-unknown-unknown`) inside a Profile-A
//! [`Sandbox`](hyprstream_workers_wasmtime::Sandbox). The engine knows nothing about
//! Python; everything python-specific (the `eval`/`py_op` ABI, the result decoding,
//! the `/lang/python` mount) lives here.
//!
//! Module shape mirrors `hyprstream-workers-tcl`:
//! - `lib.rs` — the [`PythonShell`] + re-exports of the mount items.
//! - `mount.rs` — [`PythonMount`] (`impl Mount`), [`PyCommand`], [`PythonMount::spawn`].
//!
//! ## Guest ABI used here
//!
//! * `eval(ptr,len) -> i32` — run UTF-8 source in a FRESH interpreter; 0 = ok,
//!   nonzero = a Python error. Driven via
//!   [`Sandbox::call_export`](hyprstream_workers_wasmtime::Sandbox::call_export).
//! * `py_op(op, ptr, len) -> i64` — run an operation against the PERSISTENT
//!   interpreter, returning a packed `(out_ptr<<32)|out_len` reply whose first byte is
//!   a status tag. Driven by [`PythonShell`] over the generic
//!   [`PersistentInstance`](hyprstream_workers_wasmtime::PersistentInstance).
//!
//! ## Security
//!
//! The guest only ever reaches the host capabilities the engine's `Linker` grants
//! (`env::host_random` + the Subject-scoped `env::vfs_*`); everything else traps. DoS
//! bounds (fuel / epoch) are enforced by the engine. `import os; os.system(...)` is
//! inert — there is no syscall surface.

use hyprstream_workers_wasmtime::{PersistentInstance, Sandbox, TAG_OK};
use wasmtime::Result;

pub mod mount;

pub use mount::{PyCommand, PythonMount};

/// The packed-op export name on the python guest (the PERSISTENT interpreter ABI).
const PY_OP_EXPORT: &str = "py_op";

/// Status tag for "name absent" in a `py_op` reply (`TAG_NONE=2`), matching the
/// guest. `TAG_OK=0` is the shared engine ABI tag
/// ([`hyprstream_workers_wasmtime::TAG_OK`]); any other tag (i.e. ERR=1) decodes to
/// [`PyResult::Err`].
const TAG_NONE: u8 = 2;

/// Run Python `source` in a FRESH interpreter with the given fuel budget.
///
/// Returns the guest `eval` status (0 = ok, nonzero = python error) on normal
/// completion, or `Err` if the guest TRAPPED (e.g. ran out of fuel, or reached a
/// trap-wired import). Deterministic; preferred for tests. Built on the generic
/// [`Sandbox::call_export`](hyprstream_workers_wasmtime::Sandbox::call_export) over
/// the `"eval"` export.
pub fn eval(sandbox: &Sandbox, source: &str, fuel: u64) -> Result<i32> {
    sandbox.call_export("eval", source.as_bytes(), fuel)
}

/// Run Python `source` (FRESH interpreter) with a WALL-CLOCK epoch deadline of
/// `ticks` epoch increments.
///
/// This is the PRODUCTION DoS bound: paired with an
/// [`EpochTimer`](hyprstream_workers_wasmtime::EpochTimer) spawned on the sandbox's
/// engine ticking every `t`, the guest traps with "interrupt" after roughly
/// `ticks * t` of wall time. The caller owns the timer; this fn does not start one.
pub fn eval_with_epoch_deadline(sandbox: &Sandbox, source: &str, ticks: u64) -> Result<i32> {
    sandbox.call_export_with_epoch("eval", source.as_bytes(), ticks)
}

/// The op codes the guest's `py_op` export understands (must match the guest's
/// `Op` enum byte-for-byte).
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
enum PyOp {
    Eval = 0,
    Exec = 1,
    ListVars = 2,
    GetVar = 3,
    ListDefs = 4,
    GetDef = 5,
}

/// The outcome of a `py_op`: a status tag plus the UTF-8 payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PyResult {
    /// Success — the payload (repr / stdout / newline-joined names).
    Ok(String),
    /// A Python-level error — the message.
    Err(String),
    /// The requested name was absent (get_var / get_def).
    None,
}

impl PyResult {
    /// The success payload, or `None` for Err/None.
    pub fn ok(&self) -> Option<&str> {
        match self {
            PyResult::Ok(s) => Some(s),
            _ => None,
        }
    }
}

/// A persistent python shell over a single wasm guest instance.
///
/// Holds the engine's generic
/// [`PersistentInstance`](hyprstream_workers_wasmtime::PersistentInstance) so the
/// guest interpreter's state survives across calls (the `/lang/python/vars/` +
/// `/defs/` semantics depend on this). Each method drives the guest's `py_op` ABI via
/// the generic [`PersistentInstance::call_op`] and decodes the `(tag, payload)` reply
/// into a [`PyResult`].
///
/// A `PythonShell` must be driven from a NON-async thread if its sandbox holds a VFS
/// capability (the `vfs_*` host fns `blocking_send`).
pub struct PythonShell {
    instance: PersistentInstance,
}

impl PythonShell {
    /// Open a persistent python shell over `sandbox`.
    ///
    /// `per_call_fuel` is the fuel budget per `eval`/`exec` (a fresh budget each call;
    /// the underlying store is reused so the interpreter scope persists).
    pub fn open(sandbox: &Sandbox, per_call_fuel: u64) -> Result<Self> {
        Ok(Self {
            instance: sandbox.open_instance(per_call_fuel)?,
        })
    }

    /// Run one op with `arg` as the UTF-8 argument; decode the reply.
    fn call(&mut self, op: PyOp, arg: &str) -> Result<PyResult> {
        let (tag, body) = self.instance.call_op(PY_OP_EXPORT, op as i32, arg.as_bytes())?;
        let body = String::from_utf8_lossy(&body).into_owned();
        Ok(match tag {
            TAG_OK => PyResult::Ok(body),
            TAG_NONE => PyResult::None,
            _ => PyResult::Err(body),
        })
    }

    /// Evaluate an expression -> `repr(result)` (with `\n---\n<stdout>` appended if
    /// the expression printed anything), mirroring the `/lang/python/eval` file.
    pub fn eval(&mut self, expr: &str) -> Result<PyResult> {
        self.call(PyOp::Eval, expr)
    }

    /// Execute statements -> captured stdout, mirroring the `/lang/python/stdout` file.
    pub fn exec(&mut self, src: &str) -> Result<PyResult> {
        self.call(PyOp::Exec, src)
    }

    /// Newline-joined non-dunder global variable names (`vars/`).
    pub fn list_vars(&mut self) -> Result<Vec<String>> {
        Ok(split_names(self.call(PyOp::ListVars, "")?))
    }

    /// `repr` of one global variable (`vars/<name>`), or `None` if absent.
    pub fn get_var(&mut self, name: &str) -> Result<PyResult> {
        self.call(PyOp::GetVar, name)
    }

    /// Newline-joined callable global names (`defs/`).
    pub fn list_defs(&mut self) -> Result<Vec<String>> {
        Ok(split_names(self.call(PyOp::ListDefs, "")?))
    }

    /// `repr` of one callable global (`defs/<name>`), or `None` if absent/not callable.
    pub fn get_def(&mut self, name: &str) -> Result<PyResult> {
        self.call(PyOp::GetDef, name)
    }

    /// Serve one [`PyCommand`] against this shell (runs on the shell-owning thread).
    ///
    /// Mirrors `TclShell::process_command`: the [`PythonMount`] forwards each request
    /// over the mount channel; the owner drains the channel and calls this.
    pub fn process_command(&mut self, cmd: PyCommand) {
        match cmd {
            PyCommand::Eval { code, resp } => {
                let _ = resp.send(self.eval(&code).unwrap_or_else(err_result));
            }
            PyCommand::Exec { code, resp } => {
                let _ = resp.send(self.exec(&code).unwrap_or_else(err_result));
            }
            PyCommand::ListVars { resp } => {
                let _ = resp.send(self.list_vars().unwrap_or_default());
            }
            PyCommand::GetVar { name, resp } => {
                let _ = resp.send(self.get_var(&name).unwrap_or_else(err_result));
            }
            PyCommand::ListDefs { resp } => {
                let _ = resp.send(self.list_defs().unwrap_or_default());
            }
            PyCommand::GetDef { name, resp } => {
                let _ = resp.send(self.get_def(&name).unwrap_or_else(err_result));
            }
        }
    }
}

/// A trapped/host-side error surfaces to the guest as a Python-level error string.
fn err_result(e: wasmtime::Error) -> PyResult {
    PyResult::Err(format!("sandbox: {e}"))
}

/// Split a newline-joined name list reply into a `Vec<String>` (empty if not Ok).
fn split_names(r: PyResult) -> Vec<String> {
    match r {
        PyResult::Ok(s) if !s.is_empty() => s.lines().map(|l| l.to_owned()).collect(),
        _ => Vec::new(),
    }
}
