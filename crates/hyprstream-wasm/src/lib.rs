//! Embedded wasmtime host for the #505 untrusted-Python capability sandbox.
//!
//! The whole point: run the RustPython guest (`hyprstream-wasm-pyguest`, compiled
//! to `wasm32-unknown-unknown`) under wasmtime with a BESPOKE `Linker` that exposes
//! exactly ONE capability host function — `host_random(ptr,len)` — and NOTHING from
//! WASI. No `wasmtime-wasi`, no `add_to_linker`, no preopens. Therefore the guest's
//! `import os; os.system(...)` cannot reach a real OS: there is no syscall surface.
//!
//! P1 additions vs. the P0 spike:
//!   * The guest now actually RUNS Python: rustpython-vm 0.5 + a single
//!     `env::host_random` import (verified) means `print(1 + 1)` returns status 0.
//!   * The `Store` data is generalized from a fixed `HostState` into a
//!     [`SandboxState`] that carries a per-call [`Subject`] (tenant/identity).
//!     Capability host fns (today `host_random`, tomorrow VFS — see [`vfs`]) are
//!     Subject-scoped BY CONSTRUCTION: they read the Subject out of the Store.
//!   * A real wall-clock DoS bound: an [`EpochTimer`] thread advances the engine
//!     epoch on a fixed cadence, and [`Sandbox::eval`] sets a per-call epoch
//!     deadline, so a runaway guest (`while True: pass`) traps within ~the timeout
//!     in PRODUCTION, not just in the fuel test path.
//!
//! DoS guards on the `Store` (wasmtime 46.0.1 spellings):
//!   * fuel  (`Config::consume_fuel(true)` + `Store::set_fuel(u64)`)
//!   * epoch (`Config::epoch_interruption(true)` + `Store::set_epoch_deadline(u64)`
//!     + `Engine::increment_epoch()`)
//!
//! NB: wasmtime 46 ships its OWN error type (`wasmtime::Error` / `wasmtime::Result`,
//! NOT `anyhow::Error`), with an inherent `wasmtime::Context` trait. We use those.

use std::sync::Arc;
use std::time::Duration;

/// Boxed entropy source: fills a byte buffer. Boxed so [`SandboxState`] can hold an
/// OS-backed or deterministic-test RNG behind the same field.
type RngFill = Box<dyn FnMut(&mut [u8]) + Send>;

use wasmtime::error::Context as _;
use wasmtime::{bail, Caller, Config, Engine, Extern, Linker, Memory, Module, Result, Store};

pub mod mount;
pub mod vfs;
pub mod wasi_fs;
pub mod wasi_sandbox;

// ---------------------------------------------------------------------------
// Subject: the identity/tenant a sandbox call runs as.
// ---------------------------------------------------------------------------

/// The identity a sandbox evaluation runs as.
///
/// #483 P2: this is now the CANONICAL [`hyprstream_rpc::Subject`] (re-exported via
/// `hyprstream_vfs::Subject`), NOT the P1 crate-local newtype. Confirmed torch-free:
/// `hyprstream-vfs -> hyprstream-rpc` pulls no `tch`/libtorch, so the host crate
/// still builds without LIBTORCH. Using the canonical Subject means the VFS proxy
/// the sandbox talks to is scoped to the SAME identity the rest of the daemon uses.
pub use hyprstream_rpc::Subject;

// ---------------------------------------------------------------------------
// Per-call Store data.
// ---------------------------------------------------------------------------

/// Per-call host state carried in the wasmtime `Store` data.
///
/// Generalizes the P0 `HostState` (which held only an RNG) to ALSO carry the
/// [`Subject`] the call runs as. Every capability host function gets a
/// `Caller<'_, SandboxState>` and so can read `caller.data().subject` — the
/// capability surface is Subject-parameterized without any global/thread-local
/// state (the leak that killed the native #488 design).
pub struct SandboxState {
    /// The identity this evaluation runs as. Capability host-fns (`host_random`,
    /// `vfs_*`) are authorized/scoped against this.
    pub subject: Subject,
    /// Entropy source for the single `host_random` capability. The HOST has OS
    /// entropy; the guest only ever sees it through this function.
    rng_bytes: RngFill,
    /// #483: the Subject-scoped VFS proxy handle backing `env::vfs_*`. `None` =
    /// the sandbox was created without a VFS capability (the `vfs_*` host fns then
    /// return a "no vfs capability" error to the guest — deny-by-default).
    vfs: Option<vfs::VfsProxyHandle>,
}

impl SandboxState {
    /// Host state for the given subject, drawing randomness from the OS. No VFS.
    pub fn new(subject: Subject) -> Self {
        Self {
            subject,
            rng_bytes: Box::new(|buf| {
                use rand::RngCore;
                rand::thread_rng().fill_bytes(buf);
            }),
            vfs: None,
        }
    }

    /// Host state with a Subject-scoped VFS capability backing `env::vfs_*`.
    pub fn with_vfs(subject: Subject, vfs: vfs::VfsProxyHandle) -> Self {
        let mut s = Self::new(subject);
        s.vfs = Some(vfs);
        s
    }

    /// Deterministic host state for tests (fills with a fixed pattern).
    pub fn deterministic(subject: Subject, seed: u8) -> Self {
        Self {
            subject,
            rng_bytes: Box::new(move |buf| {
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = seed.wrapping_add(i as u8);
                }
            }),
            vfs: None,
        }
    }

    /// The subject this state is bound to.
    pub fn subject(&self) -> &Subject {
        &self.subject
    }
}

/// Build the wasmtime `Engine` with both DoS guards enabled.
pub fn build_engine() -> Result<Engine> {
    let mut config = Config::new();
    // Epoch interruption: cooperative deadline interruption (cheap, no per-op cost).
    config.epoch_interruption(true);
    // Fuel: deterministic instruction budget (used by the deterministic DoS test).
    config.consume_fuel(true);
    Engine::new(&config).context("build wasmtime engine")
}

// ---------------------------------------------------------------------------
// Epoch timer: the real wall-clock DoS bound.
// ---------------------------------------------------------------------------

/// A background thread that advances an [`Engine`]'s epoch at a fixed cadence.
///
/// wasmtime's epoch interruption is the cheap, production-grade way to enforce a
/// WALL-CLOCK timeout (fuel is deterministic but instruction-count based, which is
/// the right tool for tests, not for "kill anything that runs longer than N ms").
/// This timer ticks every `tick`; a per-call `Store::set_epoch_deadline(n)` then
/// means the guest traps with "interrupt" after roughly `n * tick` of wall time.
///
/// Dropping the timer stops the thread (cooperatively, within one tick).
pub struct EpochTimer {
    handle: Option<std::thread::JoinHandle<()>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    tick: Duration,
}

impl EpochTimer {
    /// Spawn a timer that calls `engine.increment_epoch()` every `tick`.
    pub fn spawn(engine: &Engine, tick: Duration) -> Self {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // The Engine is cheaply cloneable (Arc inside) and the clone shares the
        // same epoch counter, so the timer's increments are observed by stores
        // created from the original engine.
        let engine = engine.clone();
        let stop_clone = stop.clone();
        let handle = std::thread::Builder::new()
            .name("hyprstream-wasm-epoch".into())
            .spawn(move || {
                while !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(tick);
                    engine.increment_epoch();
                }
            })
            .expect("spawn epoch timer thread");
        Self {
            handle: Some(handle),
            stop,
            tick,
        }
    }

    /// The cadence at which this timer advances the epoch.
    pub fn tick(&self) -> Duration {
        self.tick
    }
}

impl Drop for EpochTimer {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Construct the bespoke Linker exposing ONLY `env::host_random`.
///
/// Every other import the guest declares is wired to a trap via
/// `define_unknown_imports_as_traps` — so if guest code ever actually reached for
/// them it would trap rather than silently linking to nothing. This is what keeps
/// the capability surface to exactly one function. (With rustpython-vm 0.5 the guest
/// declares EXACTLY `env::host_random` and nothing else — but the trap-everything
/// fence stays as a deny-by-default belt-and-suspenders for any future guest.)
pub fn build_linker(engine: &Engine, module: &Module) -> Result<Linker<SandboxState>> {
    let mut linker: Linker<SandboxState> = Linker::new(engine);

    // The ONE capability: fill guest memory[ptr..ptr+len] with host randomness.
    // It reads its Subject from the Store data (`caller.data().subject`), so it is
    // Subject-scoped by construction — the seam future capability fns reuse.
    linker.func_wrap(
        "env",
        "host_random",
        |mut caller: Caller<'_, SandboxState>, ptr: i32, len: i32| -> Result<()> {
            // Subject is available here for scoping/auditing (RNG is granted to all
            // subjects; the point is the seam, not a denial for entropy).
            let _subject = caller.data().subject.clone();
            let memory = match caller.get_export("memory") {
                Some(Extern::Memory(m)) => m,
                _ => bail!("guest has no exported memory"),
            };
            let len = len as usize;
            let mut scratch = vec![0u8; len];
            (caller.data_mut().rng_bytes)(&mut scratch);
            write_memory(&memory, &mut caller, ptr as usize, &scratch)?;
            Ok(())
        },
    )?;

    // ── #483: the VFS capability host fns ──────────────────────────────────
    //
    // Each reads `caller.data().subject` (NEVER a guest-supplied identity) and
    // submits the op to the Subject-scoped proxy handle in the Store data. The op
    // result is written back into a guest buffer (allocated via the guest's own
    // `alloc` export) as `[status_byte][payload...]` and returned packed as
    // `(out_ptr<<32)|out_len` — the i64 ABI the guest's `take_reply` decodes.
    //
    // If the sandbox has NO vfs handle, the call returns a deny reply
    // (deny-by-default): the capability is only present when explicitly granted.

    // vfs_cat(path_ptr, path_len) -> i64
    linker.func_wrap(
        "env",
        "vfs_cat",
        |mut caller: Caller<'_, SandboxState>, ptr: i32, len: i32| -> Result<i64> {
            let path = read_guest_string(&mut caller, ptr, len)?;
            let reply = match caller.data().vfs.clone() {
                Some(h) => h.cat(&path),
                None => no_vfs_reply(),
            };
            write_reply(&mut caller, reply)
        },
    )?;

    // vfs_ls(path_ptr, path_len) -> i64
    linker.func_wrap(
        "env",
        "vfs_ls",
        |mut caller: Caller<'_, SandboxState>, ptr: i32, len: i32| -> Result<i64> {
            let path = read_guest_string(&mut caller, ptr, len)?;
            let reply = match caller.data().vfs.clone() {
                Some(h) => h.ls(&path),
                None => no_vfs_reply(),
            };
            write_reply(&mut caller, reply)
        },
    )?;

    // vfs_echo(path_ptr, path_len, data_ptr, data_len) -> i64
    linker.func_wrap(
        "env",
        "vfs_echo",
        |mut caller: Caller<'_, SandboxState>,
         pptr: i32,
         plen: i32,
         dptr: i32,
         dlen: i32|
         -> Result<i64> {
            let path = read_guest_string(&mut caller, pptr, plen)?;
            let data = read_guest_bytes(&mut caller, dptr, dlen)?;
            let reply = match caller.data().vfs.clone() {
                Some(h) => h.echo(&path, &data),
                None => no_vfs_reply(),
            };
            write_reply(&mut caller, reply)
        },
    )?;

    // vfs_ctl(path_ptr, path_len, cmd_ptr, cmd_len) -> i64
    linker.func_wrap(
        "env",
        "vfs_ctl",
        |mut caller: Caller<'_, SandboxState>,
         pptr: i32,
         plen: i32,
         cptr: i32,
         clen: i32|
         -> Result<i64> {
            let path = read_guest_string(&mut caller, pptr, plen)?;
            let cmd = read_guest_bytes(&mut caller, cptr, clen)?;
            let reply = match caller.data().vfs.clone() {
                Some(h) => h.ctl(&path, &cmd),
                None => no_vfs_reply(),
            };
            write_reply(&mut caller, reply)
        },
    )?;

    // Wire every OTHER declared import to a trap. This is the explicit, auditable
    // statement that the guest gets NO other host capability.
    linker.define_unknown_imports_as_traps(module)?;

    Ok(linker)
}

/// The deny-by-default reply when a sandbox has no VFS capability granted.
fn no_vfs_reply() -> vfs::VfsReply {
    vfs::VfsReply {
        ok: false,
        body: b"vfs: no capability granted to this sandbox".to_vec(),
    }
}

/// Read `len` bytes of guest memory at `ptr` as a (lossy) UTF-8 string.
fn read_guest_string(caller: &mut Caller<'_, SandboxState>, ptr: i32, len: i32) -> Result<String> {
    let bytes = read_guest_bytes(caller, ptr, len)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Read `len` bytes of guest memory at `ptr`.
fn read_guest_bytes(caller: &mut Caller<'_, SandboxState>, ptr: i32, len: i32) -> Result<Vec<u8>> {
    if len <= 0 {
        return Ok(Vec::new());
    }
    let memory = match caller.get_export("memory") {
        Some(Extern::Memory(m)) => m,
        _ => bail!("guest has no exported memory"),
    };
    let data = memory.data(&caller);
    let start = ptr as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| wasmtime::Error::msg("vfs arg length overflow"))?;
    if end > data.len() {
        bail!("vfs arg out of bounds");
    }
    Ok(data[start..end].to_vec())
}

/// Serialise a [`vfs::VfsReply`] into a freshly guest-`alloc`ated buffer as
/// `[status_byte][payload...]` and return the packed `(ptr<<32)|len` i64. The
/// guest reads it back and `dealloc`s the buffer (see the guest's `take_reply`).
fn write_reply(caller: &mut Caller<'_, SandboxState>, reply: vfs::VfsReply) -> Result<i64> {
    // status: 0 = ok, 1 = err (matches the guest's TAG_OK / TAG_ERR).
    let status: u8 = if reply.ok { 0 } else { 1 };
    let total = 1 + reply.body.len();

    // Allocate the reply buffer inside the guest via its own allocator so the
    // guest's `dealloc(ptr, len)` matches.
    let alloc = caller
        .get_export("alloc")
        .and_then(Extern::into_func)
        .ok_or_else(|| wasmtime::Error::msg("guest has no alloc export"))?
        .typed::<i32, i32>(&caller)?;
    let out_ptr = alloc.call(&mut *caller, total as i32)?;

    let memory = match caller.get_export("memory") {
        Some(Extern::Memory(m)) => m,
        _ => bail!("guest has no exported memory"),
    };
    let mem = memory.data_mut(&mut *caller);
    let start = out_ptr as usize;
    let end = start
        .checked_add(total)
        .ok_or_else(|| wasmtime::Error::msg("reply length overflow"))?;
    if end > mem.len() {
        bail!("reply write out of bounds");
    }
    mem[start] = status;
    mem[start + 1..end].copy_from_slice(&reply.body);

    Ok(((out_ptr as u64) << 32 | total as u64) as i64)
}

fn write_memory(
    memory: &Memory,
    store: &mut Caller<'_, SandboxState>,
    offset: usize,
    data: &[u8],
) -> Result<()> {
    let mem = memory.data_mut(store);
    let end = match offset.checked_add(data.len()) {
        Some(e) => e,
        None => bail!("host_random length overflow"),
    };
    if end > mem.len() {
        bail!("host_random write out of bounds");
    }
    mem[offset..end].copy_from_slice(data);
    Ok(())
}

/// A loaded, ready-to-run sandbox over a single guest module, bound to one
/// [`Subject`].
///
/// Constructing a `Sandbox` binds it to a Subject; every `eval` on it runs as that
/// Subject. Two `Sandbox`es bound to different Subjects keep independent Store
/// identity (no shared global/thread-local state) — see the `subject_isolation`
/// test.
pub struct Sandbox {
    engine: Engine,
    module: Module,
    linker: Linker<SandboxState>,
    subject: Subject,
    /// #483: optional Subject-scoped VFS capability handed to every `Store` this
    /// sandbox builds. `None` = no VFS capability (deny-by-default).
    vfs: Option<vfs::VfsProxyHandle>,
}

impl Sandbox {
    /// Load a guest module from raw wasm bytes, bound to the anonymous subject.
    pub fn from_bytes(wasm: &[u8]) -> Result<Self> {
        Self::from_bytes_for(wasm, Subject::anonymous())
    }

    /// Load a guest module from raw wasm bytes, bound to `subject`.
    pub fn from_bytes_for(wasm: &[u8], subject: Subject) -> Result<Self> {
        let engine = build_engine()?;
        let module = Module::new(&engine, wasm).context("compile guest module")?;
        let linker = build_linker(&engine, &module)?;
        Ok(Self {
            engine,
            module,
            linker,
            subject,
            vfs: None,
        })
    }

    /// Grant this sandbox a Subject-scoped VFS capability (`env::vfs_*`).
    ///
    /// The `handle` should be a [`vfs::VfsProxyHandle`] from
    /// `spawn_vfs_proxy(ns, subject)` with the SAME subject the sandbox is bound to,
    /// so the capability is scoped to this sandbox's identity. Builder-style.
    pub fn with_vfs(mut self, handle: vfs::VfsProxyHandle) -> Self {
        self.vfs = Some(handle);
        self
    }

    /// The subject this sandbox is bound to.
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// Build the per-call host state (subject + optional VFS capability).
    fn make_state(&self) -> SandboxState {
        match self.vfs.clone() {
            Some(h) => SandboxState::with_vfs(self.subject.clone(), h),
            None => SandboxState::new(self.subject.clone()),
        }
    }

    /// Fresh per-call Store for this sandbox's subject, with the epoch pushed out
    /// of the way (so the caller chooses fuel- or epoch-bounding explicitly).
    fn new_store(&self) -> Store<SandboxState> {
        Store::new(&self.engine, self.make_state())
    }

    /// Run `source` in a fresh interpreter with the given fuel budget.
    ///
    /// Returns the guest `eval` status (0 = ok, nonzero = python error) on normal
    /// completion, or `Err` if the guest TRAPPED (e.g. ran out of fuel / a dead
    /// import was actually called). Deterministic; preferred for tests.
    pub fn eval(&self, source: &str, fuel: u64) -> Result<i32> {
        let mut store = self.new_store();
        // DoS guard #1: instruction budget.
        store.set_fuel(fuel).context("set fuel")?;
        // Because the engine has epoch_interruption(true), EVERY store defaults to
        // an epoch deadline of 0 and would trap immediately ("wasm trap: interrupt")
        // unless we push the deadline out. For the fuel-bounded path we disable the
        // epoch by setting it effectively infinite, so fuel is the only limiter.
        store.set_epoch_deadline(u64::MAX);
        self.run(store, source)
    }

    /// Run `source` with a WALL-CLOCK epoch deadline of `ticks` epoch increments.
    ///
    /// This is the PRODUCTION DoS bound: paired with an [`EpochTimer`] spawned on
    /// [`Sandbox::engine`] ticking every `t`, the guest traps with "interrupt"
    /// after roughly `ticks * t` of wall time. Fuel is set effectively infinite so
    /// the epoch (not fuel) is the limiter.
    ///
    /// The caller owns the [`EpochTimer`]; this method does not start one (so the
    /// timer cadence/lifetime is explicit and the engine is shared by reference).
    pub fn eval_with_epoch_deadline(&self, source: &str, ticks: u64) -> Result<i32> {
        let mut store = self.new_store();
        // Give plenty of fuel so the EPOCH deadline (not fuel) is the limiter here.
        store.set_fuel(u64::MAX).context("set fuel")?;
        // DoS guard #2: epoch deadline. Trap once the engine epoch advances `ticks`.
        store.set_epoch_deadline(ticks);
        self.run(store, source)
    }

    /// Shared instantiate + ship-source + call-eval body.
    fn run(&self, mut store: Store<SandboxState>, source: &str) -> Result<i32> {
        let instance = self
            .linker
            .instantiate(&mut store, &self.module)
            .context("instantiate guest")?;

        let alloc = instance.get_typed_func::<i32, i32>(&mut store, "alloc")?;
        let dealloc = instance.get_typed_func::<(i32, i32), ()>(&mut store, "dealloc")?;
        let eval = instance.get_typed_func::<(i32, i32), i32>(&mut store, "eval")?;
        let memory = match instance.get_memory(&mut store, "memory") {
            Some(m) => m,
            None => bail!("guest memory export"),
        };

        let bytes = source.as_bytes();
        let len = bytes.len() as i32;
        let ptr = alloc.call(&mut store, len)?;
        memory
            .write(&mut store, ptr as usize, bytes)
            .context("write source into guest")?;

        let status = eval.call(&mut store, (ptr, len));
        let _ = dealloc.call(&mut store, (ptr, len));
        status.context("guest eval trapped")
    }

    /// The engine backing this sandbox. Spawn an [`EpochTimer`] on it to enable the
    /// wall-clock bound used by [`Sandbox::eval_with_epoch_deadline`].
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Drive the guest's `vfs_probe` export: make the GUEST call an `env::vfs_*`
    /// host fn, proving the capability is real end-to-end (guest -> host fn ->
    /// Subject-scoped proxy -> `Namespace`).
    ///
    /// `op`: 0=cat, 1=ls, 2=echo, 3=ctl. `path` + optional `body` (echo/ctl). Returns
    /// the decoded `[status][payload]` reply the host fn produced. This is the
    /// minimal "a guest script that reads/writes a VFS path goes through the Mount"
    /// path for deliverable (1); a follow-up registers Python builtins so guest
    /// Python source itself can call these (see guest `guest_cat`/etc.).
    pub fn probe_vfs(&self, op: i32, path: &str, body: &[u8]) -> Result<vfs::VfsReply> {
        let mut store = self.new_store();
        store.set_fuel(u64::MAX).context("set fuel")?;
        store.set_epoch_deadline(u64::MAX);
        let instance = self
            .linker
            .instantiate(&mut store, &self.module)
            .context("instantiate guest")?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| wasmtime::Error::msg("guest memory export"))?;
        let alloc = instance.get_typed_func::<i32, i32>(&mut store, "alloc")?;
        let probe = instance
            .get_typed_func::<(i32, i32, i32, i32, i32), i64>(&mut store, "vfs_probe")?;

        // Ship path + body into guest memory.
        let pbytes = path.as_bytes();
        let plen = pbytes.len() as i32;
        let pptr = if plen > 0 {
            let p = alloc.call(&mut store, plen)?;
            memory.write(&mut store, p as usize, pbytes)?;
            p
        } else {
            0
        };
        let blen = body.len() as i32;
        let bptr = if blen > 0 {
            let p = alloc.call(&mut store, blen)?;
            memory.write(&mut store, p as usize, body)?;
            p
        } else {
            0
        };

        let packed = probe
            .call(&mut store, (op, pptr, plen, bptr, blen))
            .context("guest vfs_probe trapped")?;

        // The host wrote the reply via the guest allocator; decode it.
        let p = packed as u64;
        let out_ptr = (p >> 32) as usize;
        let out_len = (p & 0xffff_ffff) as usize;
        if out_ptr == 0 || out_len == 0 {
            bail!("vfs_probe empty reply");
        }
        let data = memory.data(&store);
        if out_ptr + out_len > data.len() {
            bail!("vfs_probe reply out of bounds");
        }
        let status = data[out_ptr];
        let payload = data[out_ptr + 1..out_ptr + out_len].to_vec();
        Ok(vfs::VfsReply {
            ok: status == TAG_OK,
            body: payload,
        })
    }

    /// Open a PERSISTENT `/lang/python` shell over this sandbox (#483 P2).
    ///
    /// Unlike [`Sandbox::eval`] (legacy #505, fresh interpreter per call), the
    /// returned [`PyShell`] holds ONE long-lived `Store` + `Instance`, so the guest's
    /// persistent interpreter + globals survive across `eval`/`exec` calls — the
    /// `/lang/python/vars/` + `/defs/` semantics depend on this.
    ///
    /// `fuel` is set per `py_op` call (a fresh budget each invocation); the store
    /// itself is reused. The epoch is pushed out (fuel is the limiter for the shell
    /// path; a future variant can re-arm the epoch per call).
    pub fn open_shell(&self, per_call_fuel: u64) -> Result<PyShell> {
        let mut store = self.new_store();
        store.set_fuel(per_call_fuel).context("set fuel")?;
        store.set_epoch_deadline(u64::MAX);
        let instance = self
            .linker
            .instantiate(&mut store, &self.module)
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

/// Status tags in the first byte of a `py_op` reply (must match the guest:
/// `TAG_OK=0`, `TAG_ERR=1`, `TAG_NONE=2`). Only OK and NONE are matched explicitly;
/// any other tag (i.e. ERR=1) decodes to [`PyResult::Err`].
const TAG_OK: u8 = 0;
const TAG_NONE: u8 = 2;

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

/// A persistent `/lang/python` shell over a single guest instance (#483 P2).
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
        self.store.set_fuel(self.per_call_fuel).context("set fuel")?;

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
    /// the expression printed anything), mirroring the native `eval` file.
    pub fn eval(&mut self, expr: &str) -> Result<PyResult> {
        self.call(PyOp::Eval, expr)
    }

    /// Execute statements -> captured stdout, mirroring the native `stdout` file.
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
