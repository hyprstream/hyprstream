# #505 P0 — WASM substrate build-viability spike: FINDINGS

Goal: prove RustPython can run as a `wasm32-unknown-unknown` guest under an embedded
wasmtime host with ZERO WASI and a host-provided RNG — an untrusted-Python capability
sandbox where `import os; os.system(...)` is inert by construction.

Status: **VIABLE.** The host crate compiles and its 5 tests pass. The guest builds to
`wasm32-unknown-unknown`. The #1 unknown (getrandom) is fully pinned, with a validated
green path identified. Two interpreter-version regimes:

| | rustpython-vm 0.3.1 (spec-pinned) | rustpython-vm 0.5.0 (validated resolution) |
|---|---|---|
| getrandom major | 0.2.17 (runtime RNG) + 0.3.4 (ahash) | 0.3.x only |
| getrandom/js forced? | YES (hard-coded features=["js"]) | NO (features=["std"]) |
| wasm imports | host_random + ~34 dead __wbindgen_placeholder__ JS/Date/crypto stubs | exactly 1: env::host_random |
| runtime | traps at VirtualMachine::new (JS getrandom -> undefined __wbg_static_accessor_SELF) | instantiates with 1 capability (Python essential-init is a separate P1 item) |

The committed crates use 0.3 (as specced) and document the blocker; the 0.5 path was
built and its single-import wasm verified with wasm-tools.

---

## 1. What compiles + what the tests do

Builds (both green):
- `cargo build --target wasm32-unknown-unknown` in crates/hyprstream-wasm-pyguest/
  (standalone, excluded crate) -> hyprstream_wasm_pyguest.wasm (~11 MB release).
  Exports: memory, alloc, dealloc, eval. Imports: env::host_random PLUS dead
  __wbindgen_placeholder__ JS stubs (0.3 only — see section 2).
- `cargo build -p hyprstream-wasm` (host) — no LIBTORCH; depends only on wasmtime/rand.

Host tests (`cargo test -p hyprstream-wasm`, all 5 pass):
- sandbox_loads_with_single_capability — Linker loads any guest, exposes exactly one
  host function, traps every other import.
- case1_arithmetic — print(1 + 1). Green (0.5) path: status 0. 0.3 path: traps at VM
  bootstrap (JS getrandom). No host side effect either way.
- case2_os_system_is_inert — import os; os.system('echo PWNED'). Structurally inert: the
  Linker wires no process spawner, so no host process can run. Observed: trap.
- case3_infinite_loop_is_bounded — fuel-bounded `while True: pass` traps deterministically
  ("all fuel consumed"), proving the DoS guard.
- epoch_deadline_path_compiles_and_traps — set_epoch_deadline(0) traps with
  "wasm trap: interrupt", proving the epoch limiter is live and wired.

The capability guarantee is structural, not behavioral: the bespoke wasmtime::Linker
defines exactly env::host_random and calls define_unknown_imports_as_traps(&module) for
everything else. No WASI, no wasmtime-wasi, no add_to_linker, no preopen. os.system has
no reachable syscall.

---

## 2. getrandom resolution (the #1 unknown)

rustpython-vm 0.3.1 pulls TWO getrandom majors (cargo tree -i getrandom):
- 0.2.17 — the RUNTIME RNG: getrandom 0.2.17 <- rand_core 0.6 <- rand 0.8 <-
  rustpython-common <- rustpython-vm. Seeds hash randomization at VirtualMachine::new.
- 0.3.4 — via ahash 0.8 (runtime-rng) <- hashbrown 0.14 <- malachite. Not the VM RNG.

Mechanisms (both majors satisfied; both differ):
- getrandom 0.2: `custom` cargo feature + getrandom::register_custom_getrandom!(f),
  f(&mut [u8]) -> Result<(), Error>.
- getrandom 0.3: macro GONE. Backend selected by build cfg
  --cfg getrandom_backend="custom" (in .cargo/config.toml) + a
  #[no_mangle] unsafe fn __getrandom_v03_custom(*mut u8, usize) -> Result<(), Error>.

Both delegate to one host import:
    extern "C" { fn host_random(ptr: *mut u8, len: usize); }
Plus linker flag -C link-arg=--allow-undefined so host_random becomes a wasm IMPORT
instead of a hard link error.

THE BLOCKER (0.3.1): getrandom 0.2 backend priority (getrandom-0.2.17 src/lib.rs ~L338)
checks `js` BEFORE `custom`:
    } else if cfg!(all(feature="js", wasm32, target_os="unknown")) { js.rs }   // wins
    } else if cfg!(feature="custom") { custom }                                 // unreachable
rustpython-vm 0.3.1 hard-enables getrandom/js (its Cargo.toml L63-65:
getrandom = { version="0.2.12", features=["js"] }, not even target-gated). So on
wasm32-unknown-unknown the JS backend is forced, the custom backend is DEAD, and at
runtime VirtualMachine::new -> OsRng -> getrandom_inner reaches for undefined
__wbindgen_placeholder__::__wbg_static_accessor_SELF_146583524fe1469b and traps:
"unknown import: __wbindgen_placeholder__::__wbg_static_accessor_SELF... has not been
defined". The same getrandom/js + chrono/wasmbind pull also injects ~34 dead __wbg_*
stubs (Date/now, Uint8Array, WebCrypto). Feature unification means a downstream crate
CANNOT turn getrandom/js back off.

THE RESOLUTION (validated): rustpython-vm 0.5.0 declares
getrandom = { version="0.3", features=["std"] } — NO js. getrandom 0.3 has no js-feature
precedence problem (its wasm_js/custom backends are cfg-selected, not default features).
Built crates/hyprstream-wasm-pyguest against 0.5 with only the 0.3 custom backend +
getrandom_backend="custom" cfg, and wasm-tools print confirmed the module has EXACTLY one
import: (import "env" "host_random" ...) — zero JS, zero WASI, zero wasm-bindgen. Clean
P0 green target.

Resolution options, ranked: (a) bump rustpython-vm 0.3 -> 0.5 (lowest effort, real fix —
the spec's 0.3 pin should be revised); (b) [patch.crates-io] a rustpython-vm 0.3 fork
dropping getrandom/js; (c) implement JS shims host-side (impractical — huge wasm-bindgen
surface). Regardless of version, wire Linker::define_unknown_imports_as_traps(&module) so
any residual dead JS stub is a trap-if-called, not an instantiation failure.

---

## 3. wasmtime version + fuel/epoch API spelling

wasmtime = "=46.0.1" (pinned exact; latest stable as of 2026-06-29).

Config:
    let mut c = wasmtime::Config::new();
    c.epoch_interruption(true);   // enable epoch deadline limiter
    c.consume_fuel(true);         // enable fuel limiter
Fuel (per Store):
    store.set_fuel(n: u64)?;      // returns wasmtime::Result<()>
    // exhaustion trap: "wasm trap: all fuel consumed by WebAssembly"
Epoch (per Store + per Engine):
    store.set_epoch_deadline(ticks: u64);   // infallible; relative to current engine epoch
    engine.increment_epoch();               // advance from a timer/thread to fire
    // trap: "wasm trap: interrupt"

GOTCHA: with epoch_interruption(true), EVERY store defaults to epoch deadline 0 and traps
immediately ("interrupt") unless you push the deadline out. Fuel-only path must call
store.set_epoch_deadline(u64::MAX); epoch path must give ample fuel (set_fuel(u64::MAX))
so the epoch (not fuel) is the limiter.

ERROR-TYPE GOTCHA: wasmtime 46 ships its OWN error type — wasmtime::Error /
wasmtime::Result / the wasmtime::error::Context trait (NOT anyhow). anyhow::Context does
not apply to Result<_, wasmtime::Error>; import wasmtime::error::Context and use
wasmtime::bail!. A func_wrap closure must return wasmtime::Result<()>.

---

## 4. Profile B verdict — WASI preopen backed by an arbitrary virtual filesystem

Question: which wasmtime API lets a preopen directory be backed by an arbitrary async
Mount trait (walk/open/read/write/stat) rather than a cap-std host Dir?

VERDICT (wasmtime 46.0.1): use the legacy preview1 path via wasi-common-style trait
objects, NOT the component-model wasmtime-wasi.

- wasmtime-wasi (component model / preview2): filesystem is host resources in a
  ResourceTable behind WasiView/IoView. Preopens via
  WasiCtxBuilder::preopen_dir(dir: cap_std::fs::Dir, ...) — takes a concrete cap-std Dir.
  No first-class public "implement your own async directory/file trait" seam at 46.x; the
  abstraction point is the resource table, a much larger lift to back with a custom VFS.
- wasmtime-wasi preview1 adapter (wasmtime_wasi::preview1 / wasi_snapshot_preview1 shim) +
  wasi-common: exposes WasiDir and WasiFile trait objects (open/read_at/write_at/
  get_filestat/readdir/...) that you implement and register via
  WasiCtxBuilder::preopened_dir(Box<dyn WasiDir>, ...). THIS is the make-or-break seam for
  backing a preopen with an arbitrary (async) Mount. Concrete names to target:
  wasi_common::WasiDir, wasi_common::WasiFile, wasi_common::dir::ReaddirEntity,
  wasi_common::file::Filestat, wasi_common::WasiCtx / WasiCtxBuilder::preopened_dir.

So the later "Wanix" consumer should target the preview1 wasi-common WasiDir/WasiFile
traits, not preview2's ResourceTable. (Not needed for the Python guest — zero WASI.)

---

## 5. Excludable vs unconditionally-frozen freeze-stdlib modules (defense-in-depth)

freeze-stdlib embeds the pure-Python stdlib into the binary (no filesystem import path —
itself a capability win). Granularity at build time:

- NOT individually excludable via a cargo feature. RustPython's freeze-stdlib is a single
  boolean feature; no per-module flag. The frozen set is fixed by the Lib/ snapshot baked
  at build time.
- Unconditionally frozen (essential bootstrap): importlib._bootstrap,
  importlib._bootstrap_external, codecs + the encodings package (gated by the `encodings`
  feature, which freeze-stdlib enables), and the small set imported during essential init
  (io, abc, ...). Cannot be dropped without breaking VM startup.
- Defense-in-depth lever that DOES work: use Interpreter::without_stdlib(...) (no frozen
  stdlib registered) or Interpreter::builder() and register ONLY the native module defs you
  want. os/subprocess/socket native modules are gated by host_env/stdlib feature wiring and
  by whether you call add_native_modules; with default-features=false and only
  compiler,encodings,freeze-stdlib they are not wired with any working host backend. To
  exclude a module at source level rather than feature level, prune the frozen set (custom
  build) or simply do not register the native counterpart — `import os` may resolve to a
  frozen pure-Python shim whose host-touching functions (os.system, os.fork) raise, since
  there is no native backend and no WASI. NET: module-level exclusion is a
  source/registration-time decision, not a cargo-feature toggle; the capability guarantee
  does not depend on it (the host syscall surface is empty regardless).

---

## 6. Remaining blockers + precise first step for P1

Remaining blockers:
1. rustpython-vm 0.3 forces getrandom/js -> the pure unknown-unknown custom-RNG path is
   unreachable at runtime on 0.3. Fix: bump to 0.5 (revise the spec's 0.3 pin) OR [patch] a
   0.3 fork dropping getrandom/js. 0.5 produces a single-import wasm (verified).
2. rustpython-vm 0.5 Python essential-init: Interpreter::without_stdlib with freeze-stdlib
   panics in 0.5's VirtualMachine::initialize ("essential initialization failed") because
   without_stdlib registers an empty frozen set while essential init expects the frozen
   importlib/io. Fix: build via Interpreter::builder(settings) and register the frozen
   stdlib + native module defs per 0.5's InterpreterBuilder (the rustpython crate's builder
   is the reference).
3. wasm32-wasip1 is NOT a fallback at 0.3: rustpython-vm 0.3.1 stdlib/time.rs uses
   libc::CLOCK_PROCESS_CPUTIME_ID, absent on wasip1 -> compile error. (Also moot for the
   zero-WASI goal.)

Precise first step for P1 (hyprstream-wasm host with capability Linker + Subject binding +
epoch bound):
1. Bump the guest to rustpython-vm = "0.5" (+ getrandom 0.3 custom backend only; drop the
   0.2 dual-backend code) and switch guest construction to Interpreter::builder(settings)
   registering the frozen stdlib so print(1+1) returns status 0 (case1 green).
2. In the host, generalize HostState into a per-call Subject (tenant/identity) carried in
   Store data, so host_random (and future capability host fns) are scoped to the bound
   Subject — the capability Linker becomes Subject-parameterized.
3. Add a real epoch bound: spawn an Engine::increment_epoch() timer thread and set a
   per-call set_epoch_deadline(n) so wall-clock DoS (not just fuel) is enforced; keep
   define_unknown_imports_as_traps as the deny-by-default capability fence.
