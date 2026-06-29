//! Embedded wasmtime host for the #505 untrusted-Python capability sandbox spike.
//!
//! The whole point: run the RustPython guest (`hyprstream-wasm-pyguest`, compiled
//! to `wasm32-unknown-unknown`) under wasmtime with a BESPOKE `Linker` that exposes
//! exactly ONE capability host function — `host_random(ptr,len)` — and NOTHING from
//! WASI. No `wasmtime-wasi`, no `add_to_linker`, no preopens. Therefore the guest's
//! `import os; os.system(...)` cannot reach a real OS: there is no syscall surface.
//!
//! DoS guards available on the `Store` (wasmtime 46.0.1 spellings):
//!   * fuel  (`Config::consume_fuel(true)` + `Store::set_fuel(u64)`)
//!   * epoch (`Config::epoch_interruption(true)` + `Store::set_epoch_deadline(u64)` +
//!            `Engine::increment_epoch()`)
//!
//! NB: wasmtime 46 ships its OWN error type (`wasmtime::Error` / `wasmtime::Result`,
//! NOT `anyhow::Error`), with an inherent `wasmtime::Context` trait. We use those.

use wasmtime::error::Context as _;
use wasmtime::{bail, Caller, Config, Engine, Extern, Linker, Memory, Module, Result, Store};

/// Per-instance host state. Holds an RNG used by the single capability function.
pub struct HostState {
    rng_bytes: Box<dyn FnMut(&mut [u8]) + Send>,
}

impl HostState {
    /// Default host state: fills randomness from the OS via `rand`/`getrandom`.
    /// (The HOST has OS entropy; the guest only sees it through `host_random`.)
    pub fn new() -> Self {
        Self {
            rng_bytes: Box::new(|buf| {
                use rand::RngCore;
                rand::thread_rng().fill_bytes(buf);
            }),
        }
    }

    /// Deterministic host state for tests (fills with a fixed pattern).
    pub fn deterministic(seed: u8) -> Self {
        Self {
            rng_bytes: Box::new(move |buf| {
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = seed.wrapping_add(i as u8);
                }
            }),
        }
    }
}

impl Default for HostState {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the wasmtime `Engine` with both DoS guards enabled.
pub fn build_engine() -> Result<Engine> {
    let mut config = Config::new();
    // Epoch interruption: cooperative deadline interruption (cheap, no per-op cost).
    config.epoch_interruption(true);
    // Fuel: deterministic instruction budget (used by the DoS test).
    config.consume_fuel(true);
    Engine::new(&config).context("build wasmtime engine")
}

/// Construct the bespoke Linker exposing ONLY `env::host_random`.
///
/// Every other import the guest declares (the dead chrono/js-sys/getrandom-js
/// `__wbindgen_placeholder__` stubs that RustPython drags in on the wasm target)
/// is wired to a trap via `define_unknown_imports_as_traps` — so if guest code ever
/// actually reached for them it would trap rather than silently linking to nothing.
/// This is what keeps the capability surface to exactly one function.
pub fn build_linker(engine: &Engine, module: &Module) -> Result<Linker<HostState>> {
    let mut linker: Linker<HostState> = Linker::new(engine);

    // The ONE capability: fill guest memory[ptr..ptr+len] with host randomness.
    linker.func_wrap(
        "env",
        "host_random",
        |mut caller: Caller<'_, HostState>, ptr: i32, len: i32| -> Result<()> {
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
    // statement that the guest gets NO other host capability — any dead JS/WASI-ish
    // import becomes a trap-if-called rather than an instantiation failure.
    linker.define_unknown_imports_as_traps(module)?;

    Ok(linker)
}

fn write_memory(
    memory: &Memory,
    store: &mut Caller<'_, HostState>,
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

/// A loaded, ready-to-run sandbox over a single guest module.
pub struct Sandbox {
    engine: Engine,
    module: Module,
    linker: Linker<HostState>,
}

impl Sandbox {
    /// Load a guest module from raw wasm bytes.
    pub fn from_bytes(wasm: &[u8]) -> Result<Self> {
        let engine = build_engine()?;
        let module = Module::new(&engine, wasm).context("compile guest module")?;
        let linker = build_linker(&engine, &module)?;
        Ok(Self {
            engine,
            module,
            linker,
        })
    }

    /// Run `source` in a fresh interpreter with the given fuel budget.
    ///
    /// Returns the guest `eval` status (0 = ok, nonzero = python error) on normal
    /// completion, or `Err` if the guest TRAPPED (e.g. ran out of fuel / hit the
    /// epoch deadline / a dead import was actually called).
    pub fn eval(&self, source: &str, fuel: u64) -> Result<i32> {
        let mut store = Store::new(&self.engine, HostState::new());
        // DoS guard #1: instruction budget.
        store.set_fuel(fuel).context("set fuel")?;
        // Because the engine has epoch_interruption(true), EVERY store defaults to
        // an epoch deadline of 0 and would trap immediately ("wasm trap: interrupt")
        // unless we push the deadline out. For the fuel-bounded path we disable the
        // epoch by setting it effectively infinite, so fuel is the only limiter here.
        store.set_epoch_deadline(u64::MAX);

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

    /// Run with an EPOCH deadline instead of fuel (demonstrates the epoch path).
    ///
    /// The caller is responsible for calling `engine().increment_epoch()` from
    /// another thread/timer to actually fire the deadline. This method exists to
    /// prove the epoch API compiles and is wired; the fuel test is the deterministic
    /// DoS assertion.
    pub fn eval_with_epoch_deadline(&self, source: &str, ticks: u64) -> Result<i32> {
        let mut store = Store::new(&self.engine, HostState::new());
        // Give plenty of fuel so the EPOCH deadline (not fuel) is the limiter here.
        store.set_fuel(u64::MAX).context("set fuel")?;
        // DoS guard #2: epoch deadline. Trap once the engine epoch advances `ticks`.
        store.set_epoch_deadline(ticks);

        let instance = self.linker.instantiate(&mut store, &self.module)?;
        let alloc = instance.get_typed_func::<i32, i32>(&mut store, "alloc")?;
        let eval = instance.get_typed_func::<(i32, i32), i32>(&mut store, "eval")?;
        let memory = match instance.get_memory(&mut store, "memory") {
            Some(m) => m,
            None => bail!("guest memory export"),
        };

        let bytes = source.as_bytes();
        let len = bytes.len() as i32;
        let ptr = alloc.call(&mut store, len)?;
        memory.write(&mut store, ptr as usize, bytes)?;
        eval.call(&mut store, (ptr, len)).context("guest eval trapped")
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}
