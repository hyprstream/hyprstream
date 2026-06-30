//! Profile B (#506): a WASI (`wasm32-wasip1`) sandbox whose ENTIRE filesystem is a
//! Subject-scoped [`hyprstream_vfs::Mount`] — and nothing else.
//!
//! Contrast with Profile A ([`crate::Sandbox`]): Profile A is a bespoke capability
//! Linker granting ONLY `env::host_random` + `env::vfs_*`, zero WASI. Profile B
//! grants a REAL WASI preview1 surface (so an off-the-shelf `wasm32-wasip1` guest
//! that does `open`/`read`/`write`/`readdir` runs unmodified) — but every WASI
//! filesystem op is routed to a [`MountDir`](crate::wasi_fs::MountDir) backed by the
//! VFS Mount. The make-or-break property: **the VFS is the guest's ONLY filesystem.
//! There is no host preopen.**
//!
//! ## The hard caveat — capability withholding
//!
//! A single host preopen or any ambient WASI capability that touches the host
//! reopens the sandbox escape. So the Profile-B [`wasi_common::WasiCtx`] is built
//! by hand (NOT via `sync::WasiCtxBuilder`, which only does `cap_std::fs::Dir` host
//! preopens):
//!
//! - **Preopen set:** EXACTLY ONE — the VFS [`MountDir`] at guest path `/`. No
//!   `preopened_dir(host_dir)`, ever.
//! - **Clocks:** withheld. `WasiClocks::new()` leaves both system + monotonic
//!   `None`; any `clock_time_get`/`clock_res_get` returns `badf` (a future revision
//!   can VFS-mediate a clock file). The [`crate::EpochTimer`] is the DoS bound, not
//!   a guest-visible clock.
//! - **Randomness:** withheld from the host OS. The ctx random source is
//!   `wasi_common::random::Deterministic` (NOT OS entropy); `random_get` yields a
//!   deterministic stream, not host CSPRNG output.
//! - **Scheduling:** [`DeniedSched`] — `poll_oneoff`/`sched_yield`/`sleep` all
//!   error (deny-by-default).
//! - **stdio:** the [`WasiCtx::new`] default — stdin = empty, stdout/stderr = sink.
//!   No host fd inheritance.
//! - **`wasi:sockets`, environ, args:** none. preview1 has no sockets; environ/args
//!   are left empty.
//!
//! The bound [`Subject`] is fixed at construction and threaded by [`MountDir`] into
//! every Mount call — the Mount is the ONE policy enforcement point.

use std::sync::Arc;

use hyprstream_vfs::{Mount, Subject};
use wasi_common::clocks::WasiClocks;
use wasi_common::random::Deterministic;
use wasi_common::sched::{Poll, WasiSched};
use wasi_common::{Error as WasiError, ErrorExt, Table, WasiCtx};
use wasmtime::error::Context as _;
use wasmtime::{Engine, Linker, Module, Result, Store};

use crate::wasi_fs::MountDir;

/// A `WasiSched` that denies all scheduling — the Profile-B deny-by-default sched.
///
/// `poll_oneoff`/`sched_yield`/`sleep` are not part of the minimal fs-only surface
/// and `sleep` in particular is a (weak) DoS lever, so we withhold all three. The
/// [`crate::EpochTimer`] remains the real wall-clock bound.
struct DeniedSched;

#[async_trait::async_trait]
impl WasiSched for DeniedSched {
    async fn poll_oneoff<'a>(&self, _poll: &mut Poll<'a>) -> std::result::Result<(), WasiError> {
        Err(WasiError::not_supported().context("poll_oneoff withheld in Profile B"))
    }
    async fn sched_yield(&self) -> std::result::Result<(), WasiError> {
        Err(WasiError::not_supported().context("sched_yield withheld in Profile B"))
    }
    async fn sleep(
        &self,
        _duration: wasi_common::sched::Duration,
    ) -> std::result::Result<(), WasiError> {
        Err(WasiError::not_supported().context("sleep withheld in Profile B"))
    }
}

/// Per-call host state for a Profile-B Store: the bound [`Subject`] + the WASI ctx
/// (whose only preopen is the VFS [`MountDir`]).
pub struct WasiState {
    pub subject: Subject,
    pub wasi: WasiCtx,
}

impl WasiState {
    pub fn subject(&self) -> &Subject {
        &self.subject
    }
}

/// A Profile-B WASI sandbox: a `wasm32-wasip1` guest whose filesystem is a
/// Subject-scoped VFS [`Mount`].
pub struct WasiSandbox {
    engine: Engine,
    module: Module,
    linker: Linker<WasiState>,
    mount: Arc<dyn Mount>,
    subject: Subject,
    /// Dedicated current-thread runtime used by the `MountDir`/`MountFile` adapters
    /// to drive the async Mount synchronously under the wiggle `block_on` executor.
    rt: Arc<tokio::runtime::Runtime>,
}

impl WasiSandbox {
    /// Build a Profile-B sandbox over `wasm` (a `wasm32-wasip1` module), bound to
    /// `subject`, whose ONLY filesystem is `mount` (preopened at guest path `/`).
    ///
    /// No host preopen, no sockets, clocks+random withheld/VFS-mediated. The
    /// returned sandbox MUST be driven from a plain (non-async) thread — the Mount
    /// adapters `block_on` an inner runtime, which would panic on a tokio worker.
    pub fn wasi_for(wasm: &[u8], subject: Subject, mount: Arc<dyn Mount>) -> Result<Self> {
        let engine = crate::build_engine()?;
        let module = Module::new(&engine, wasm).context("compile wasip1 guest module")?;

        // Link the preview1 WASI surface (sync wiggle = `block_on` executor). The
        // ctx getter hands out `&mut state.wasi` — that is the WASI surface the
        // guest sees, and its ONLY filesystem is the VFS preopen built in `run`.
        let mut linker: Linker<WasiState> = Linker::new(&engine);
        wasi_common::sync::add_to_linker(&mut linker, |s: &mut WasiState| &mut s.wasi)
            .context("add wasi preview1 to linker")?;

        // A current-thread runtime is enough: every Mount op is `block_on`'d to
        // completion synchronously (the adapters never spawn).
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("build mount-bridge runtime")?;

        Ok(Self {
            engine,
            module,
            linker,
            mount,
            subject,
            rt: Arc::new(rt),
        })
    }

    /// The subject this sandbox runs as.
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// The engine (spawn a [`crate::EpochTimer`] on it for the wall-clock bound).
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Build the hand-rolled, capability-withholding [`WasiCtx`]: deterministic
    /// random, NO clocks, denied sched, sink stdio, and EXACTLY one preopen — the
    /// VFS [`MountDir`] at guest path `/`. No host preopen.
    fn make_ctx(&self) -> Result<WasiCtx> {
        // Random: deterministic, NOT host OS entropy (withheld). The byte cycle is
        // arbitrary; the point is "no host CSPRNG".
        let random: Box<dyn wasi_rand::Rng + Send + Sync> =
            Box::new(Deterministic::new(vec![0xA5, 0x5A, 0x3C, 0xC3]));
        // Clocks: both None -> clock_*_get returns badf (withheld).
        let clocks = WasiClocks::new();
        let sched = Box::new(DeniedSched);
        let ctx = WasiCtx::new(random, clocks, sched, Table::new());

        // The ONLY preopen: the Subject-scoped VFS Mount, at guest path "/".
        let dir = MountDir::new(
            Arc::clone(&self.mount),
            self.subject.clone(),
            Arc::clone(&self.rt),
        );
        ctx.push_preopened_dir(Box::new(dir), "/")
            .context("push VFS preopen")?;
        Ok(ctx)
    }

    /// Fresh per-call Store bound to this sandbox's Subject + a freshly built ctx.
    fn new_store(&self) -> Result<Store<WasiState>> {
        let ctx = self.make_ctx()?;
        Ok(Store::new(
            &self.engine,
            WasiState {
                subject: self.subject.clone(),
                wasi: ctx,
            },
        ))
    }

    /// Run the guest's `_start` (a wasip1 *command*) under a fuel budget.
    ///
    /// Returns `Ok(())` on clean exit (or `I32Exit(0)`); a nonzero `proc_exit`
    /// surfaces as an error carrying the exit code. The guest's filesystem side
    /// effects land in the VFS Mount.
    pub fn run_start(&self, fuel: u64) -> Result<()> {
        let mut store = self.new_store()?;
        store.set_fuel(fuel).context("set fuel")?;
        // Epoch defaults to deadline 0 with epoch_interruption(true); push it out so
        // fuel is the limiter on this path (mirrors Profile A `eval`).
        store.set_epoch_deadline(u64::MAX);

        let instance = self
            .linker
            .instantiate(&mut store, &self.module)
            .context("instantiate wasip1 guest")?;
        let start = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .context("guest has no _start (not a wasip1 command?)")?;

        match start.call(&mut store, ()) {
            Ok(()) => Ok(()),
            Err(e) => {
                // A clean `exit(0)` shows up as an I32Exit(0) trap — treat as success.
                if let Some(exit) = e.downcast_ref::<wasi_common::I32Exit>() {
                    if exit.0 == 0 {
                        return Ok(());
                    }
                }
                Err(e).context("guest _start trapped")
            }
        }
    }

    /// Run `_start` under a WALL-CLOCK epoch deadline of `ticks` (paired with an
    /// [`crate::EpochTimer`] on [`Self::engine`]). Fuel is set effectively infinite
    /// so the epoch is the limiter — the production DoS bound for Profile B.
    pub fn run_start_with_epoch_deadline(&self, ticks: u64) -> Result<()> {
        let mut store = self.new_store()?;
        store.set_fuel(u64::MAX).context("set fuel")?;
        store.set_epoch_deadline(ticks);
        let instance = self
            .linker
            .instantiate(&mut store, &self.module)
            .context("instantiate wasip1 guest")?;
        let start = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .context("guest has no _start")?;
        match start.call(&mut store, ()) {
            Ok(()) => Ok(()),
            Err(e) => {
                if let Some(exit) = e.downcast_ref::<wasi_common::I32Exit>() {
                    if exit.0 == 0 {
                        return Ok(());
                    }
                }
                Err(e).context("guest _start trapped")
            }
        }
    }
}
