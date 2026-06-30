//! The Python profile: run untrusted Python on the RustPython guest
//! (`hyprstream-wasm-pyguest`, compiled to `wasm32-unknown-unknown`) inside the
//! generic Profile-A [`Sandbox`](crate::Sandbox).
//!
//! This module is the ONE python-aware layer in the crate; the core
//! ([`crate::Sandbox`]) knows nothing about Python. It builds on the generic
//! [`Sandbox::call_export`](crate::Sandbox::call_export) (for the one-shot `eval`
//! entry) and on the guest's `py_op` ABI (for the persistent [`PyShell`]). The guest
//! still only ever reaches the host capabilities the core's `Linker` grants
//! (`env::host_random` + the Subject-scoped `env::vfs_*`); everything else traps.
//!
//! ## Guest ABI used here
//!
//! * `eval(ptr,len) -> i32` — run UTF-8 source in a FRESH interpreter; 0 = ok,
//!   nonzero = a Python error. Driven via [`Sandbox::call_export`](crate::Sandbox::call_export).
//! * `py_op(op, ptr, len) -> i64` — run an operation against the PERSISTENT
//!   interpreter, returning a packed `(out_ptr<<32)|out_len` reply whose first byte
//!   is a status tag. Driven by [`PyShell`].

use wasmtime::error::Context as _;
use wasmtime::{bail, Memory, Result, Store};

use crate::{Sandbox, SandboxState, TAG_OK};

pub mod mount;

/// Status tag for "name absent" in a `py_op` reply (`TAG_NONE=2`), matching the
/// guest. `TAG_OK=0` is shared with the core ([`crate::TAG_OK`]); any other tag
/// (i.e. ERR=1) decodes to [`PyResult::Err`].
const TAG_NONE: u8 = 2;

impl Sandbox {
    /// Run Python `source` in a FRESH interpreter with the given fuel budget.
    ///
    /// Returns the guest `eval` status (0 = ok, nonzero = python error) on normal
    /// completion, or `Err` if the guest TRAPPED (e.g. ran out of fuel, or reached a
    /// trap-wired import). Deterministic; preferred for tests. Built on the generic
    /// [`Sandbox::call_export`](crate::Sandbox::call_export) over the `"eval"` export.
    pub fn eval(&self, source: &str, fuel: u64) -> Result<i32> {
        self.call_export("eval", source.as_bytes(), fuel)
    }

    /// Run Python `source` (FRESH interpreter) with a WALL-CLOCK epoch deadline of
    /// `ticks` epoch increments.
    ///
    /// This is the PRODUCTION DoS bound: paired with an [`EpochTimer`](crate::EpochTimer)
    /// spawned on [`Sandbox::engine`](crate::Sandbox::engine) ticking every `t`, the
    /// guest traps with "interrupt" after roughly `ticks * t` of wall time. Fuel is
    /// set effectively infinite so the epoch (not fuel) is the limiter. The caller
    /// owns the [`EpochTimer`](crate::EpochTimer); this method does not start one.
    pub fn eval_with_epoch_deadline(&self, source: &str, ticks: u64) -> Result<i32> {
        let mut store = self.new_store();
        // Give plenty of fuel so the EPOCH deadline (not fuel) is the limiter here.
        store.set_fuel(u64::MAX).context("set fuel")?;
        // DoS guard: epoch deadline. Trap once the engine epoch advances `ticks`.
        store.set_epoch_deadline(ticks);
        self.run_export(store, "eval", source.as_bytes())
    }

    /// Open a PERSISTENT python shell over this sandbox.
    ///
    /// Unlike [`Sandbox::eval`] (fresh interpreter per call), the returned [`PyShell`]
    /// holds ONE long-lived `Store` + `Instance`, so the guest's persistent
    /// interpreter + globals survive across `eval`/`exec` calls — the
    /// `/lang/python/vars/` + `/defs/` semantics depend on this.
    ///
    /// `per_call_fuel` is set per `py_op` call (a fresh budget each invocation); the
    /// store itself is reused. The epoch is pushed out (fuel is the limiter for the
    /// shell path).
    pub fn open_shell(&self, per_call_fuel: u64) -> Result<PyShell> {
        let mut store = self.new_store();
        store.set_fuel(per_call_fuel).context("set fuel")?;
        store.set_epoch_deadline(u64::MAX);
        let instance = self
            .linker()
            .instantiate(&mut store, self.module())
            .context("instantiate guest")?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| wasmtime::Error::msg("guest memory export"))?;
        let alloc = instance.get_typed_func::<i32, i32>(&mut store, "alloc")?;
        let dealloc = instance.get_typed_func::<(i32, i32), ()>(&mut store, "dealloc")?;
        let py_op = instance.get_typed_func::<(i32, i32, i32), i64>(&mut store, "py_op")?;
        Ok(PyShell {
            store,
            memory,
            alloc,
            dealloc,
            py_op,
            per_call_fuel,
        })
    }
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

/// A persistent python shell over a single guest instance.
///
/// Holds the long-lived `Store`/`Instance` so the guest interpreter's state
/// survives across calls. Drives the guest's `py_op` ABI: ship the argument into
/// guest memory, call `py_op(op, ptr, len)`, decode the packed `(out_ptr<<32)|len`
/// reply, then `dealloc` the reply buffer.
///
/// `!Send`-free at the type level (wasmtime types are `Send`), but a `PyShell` must
/// be driven from a NON-async thread if its sandbox holds a VFS capability (the
/// `vfs_*` host fns `blocking_send`).
pub struct PyShell {
    store: Store<SandboxState>,
    memory: Memory,
    alloc: wasmtime::TypedFunc<i32, i32>,
    dealloc: wasmtime::TypedFunc<(i32, i32), ()>,
    py_op: wasmtime::TypedFunc<(i32, i32, i32), i64>,
    per_call_fuel: u64,
}

impl PyShell {
    /// Run one op with `arg` as the UTF-8 argument; decode the reply.
    fn call(&mut self, op: PyOp, arg: &str) -> Result<PyResult> {
        // Fresh fuel budget per call (the store is reused, fuel is not).
        self.store
            .set_fuel(self.per_call_fuel)
            .context("set fuel")?;

        let bytes = arg.as_bytes();
        let len = bytes.len() as i32;
        let ptr = if len > 0 {
            let p = self.alloc.call(&mut self.store, len)?;
            self.memory
                .write(&mut self.store, p as usize, bytes)
                .context("write py_op arg")?;
            p
        } else {
            0
        };

        let packed = self
            .py_op
            .call(&mut self.store, (op as i32, ptr, len))
            .context("guest py_op trapped")?;
        if len > 0 {
            let _ = self.dealloc.call(&mut self.store, (ptr, len));
        }

        // Decode (out_ptr<<32)|out_len, read [tag][payload], then dealloc it.
        let p = packed as u64;
        let out_ptr = (p >> 32) as usize;
        let out_len = (p & 0xffff_ffff) as usize;
        if out_ptr == 0 || out_len == 0 {
            return Ok(PyResult::Err("empty reply".to_owned()));
        }
        let data = self.memory.data(&self.store);
        if out_ptr + out_len > data.len() {
            bail!("py_op reply out of bounds");
        }
        let tag = data[out_ptr];
        let body = String::from_utf8_lossy(&data[out_ptr + 1..out_ptr + out_len]).into_owned();
        let _ = self
            .dealloc
            .call(&mut self.store, (out_ptr as i32, out_len as i32));

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
}

/// Split a newline-joined name list reply into a `Vec<String>` (empty if not Ok).
fn split_names(r: PyResult) -> Vec<String> {
    match r {
        PyResult::Ok(s) if !s.is_empty() => s.lines().map(|l| l.to_owned()).collect(),
        _ => Vec::new(),
    }
}
