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

pub mod vfs;

// ---------------------------------------------------------------------------
// Subject: the identity/tenant a sandbox call runs as.
// ---------------------------------------------------------------------------

/// The identity a sandbox evaluation runs as.
///
/// Modelled as a LOCAL newtype for now (P1): deliberately NOT a dependency on
/// `hyprstream-rpc`/`hyprstream-vfs` yet — we are only defining the seam so that
/// future capability host-fns are Subject-scoped by construction. When #483 / the
/// native MAC authz epic re-cuts this, swap the inner representation for the real
/// `RequestIdentity` / VFS namespace key without touching the Linker wiring.
///
/// `None` is the anonymous subject (no bound identity).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Subject(pub Option<String>);

impl Subject {
    /// An anonymous subject (no bound identity).
    pub fn anonymous() -> Self {
        Self(None)
    }

    /// A subject bound to a named identity/tenant.
    pub fn named(id: impl Into<String>) -> Self {
        Self(Some(id.into()))
    }

    /// The bound identity, if any.
    pub fn id(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

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
    /// The identity this evaluation runs as. Future capability host-fns
    /// (`vfs_*`, etc.) are authorized/scoped against this.
    pub subject: Subject,
    /// Entropy source for the single `host_random` capability. The HOST has OS
    /// entropy; the guest only ever sees it through this function.
    rng_bytes: RngFill,
}

impl SandboxState {
    /// Host state for the given subject, drawing randomness from the OS.
    pub fn new(subject: Subject) -> Self {
        Self {
            subject,
            rng_bytes: Box::new(|buf| {
                use rand::RngCore;
                rand::thread_rng().fill_bytes(buf);
            }),
        }
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

    // Wire every OTHER declared import to a trap. This is the explicit, auditable
    // statement that the guest gets NO other host capability.
    linker.define_unknown_imports_as_traps(module)?;

    Ok(linker)
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
        })
    }

    /// The subject this sandbox is bound to.
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// Fresh per-call Store for this sandbox's subject, with the epoch pushed out
    /// of the way (so the caller chooses fuel- or epoch-bounding explicitly).
    fn new_store(&self) -> Store<SandboxState> {
        Store::new(&self.engine, SandboxState::new(self.subject.clone()))
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
}
