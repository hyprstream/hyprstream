# #505 P0 — WASM substrate build-viability spike: FINDINGS

> **P1 UPDATE (host crate, branch `ewindisch/505-wasm-p1-host`).** The P0 blockers
> below are RESOLVED. The guest now actually runs Python: `Sandbox::eval("print(1 + 1)")`
> returns status **0** (no trap), and `wasm-tools print` confirms the guest imports
> EXACTLY `{ env::host_random }` (zero JS/WASI/wasm-bindgen). See the appended
> **"P1 RESOLUTIONS"** section at the end of this file for the exact APIs used; the P0
> body below is preserved for historical context.

Goal: prove RustPython can run as a `wasm32-unknown-unknown` guest under an embedded
wasmtime host with ZERO WASI and a host-provided RNG — an untrusted-Python capability
sandbox where `import os; os.system(...)` is inert by construction.

Status: **VIABLE.** The host crate compiles and its 5 tests pass. The guest builds to
`wasm32-unknown-unknown`. The #1 unknown (getrandom) is fully pinned, with a validated
green path identified. Two interpreter-version regimes:

| | rustpython-vm 0.3.1 (spec-pinned) | rustpython-vm 0.5.0 (validated resolution) |
|---|---|---|
| getrandom major | 0.2.17 (runtime RNG) + 0.3.4 (ahash) | 0.3.x only **compiled for wasm** [1] |
| getrandom/js forced? | YES (hard-coded features=["js"]) | NO (features=["std"]) |
| wasm imports | host_random + ~34 dead __wbindgen_placeholder__ JS/Date/crypto stubs | exactly 1: env::host_random |
| runtime | traps at VirtualMachine::new (JS getrandom -> undefined __wbg_static_accessor_SELF) | instantiates with 1 capability (Python essential-init is a separate P1 item) |

[1] The pyguest `Cargo.lock` (0.5 path) still lists getrandom 0.2.17, but it is a HOST
build-dependency of rustpython's codegen tooling only — pulled via
`unicode_names2_generator` `[build-dependencies]` (rand 0.8 -> rand_core 0.6 ->
getrandom 0.2.17), which runs on the host during the build. It is NEVER compiled for the
wasm32-unknown-unknown target. Harmless; the wasm guest uses only getrandom 0.3 with the
custom backend.

The committed crates use 0.3 (as specced) and document the blocker; the 0.5 path was
built and its single-import wasm verified with wasm-tools.

---

## 1. What compiles + what the tests do

Builds (both green):
- `cargo build --target wasm32-unknown-unknown` run FROM INSIDE crates/hyprstream-wasm-pyguest/
  i.e. `(cd crates/hyprstream-wasm-pyguest && cargo build --release --target wasm32-unknown-unknown)`.
  MUST build from the crate dir, NOT with `--manifest-path …pyguest/Cargo.toml` from the repo
  root: the `getrandom_backend="custom"` rustflag lives in that crate's `.cargo/config.toml`,
  and Cargo only discovers `.cargo/config.toml` from the CWD (walking up), not from
  `--manifest-path`. A root `--manifest-path` build drops the cfg and FAILS on getrandom/wasm32.
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

---

## P1 RESOLUTIONS (branch `ewindisch/505-wasm-p1-host`)

All four P1 deliverables landed; `cargo test -p hyprstream-wasm` is 9/9 green and
`cargo clippy -p hyprstream-wasm --tests` is warning-clean.

### 1. Green path — `print(1 + 1)` returns status 0; guest imports = `{ env::host_random }`

- **Guest bump 0.3 -> 0.5.** `rustpython-vm = "0.5"` declares getrandom 0.3 (NO `js`),
  so the custom getrandom-0.3 backend is finally reachable. Dropped the getrandom-0.2
  dual backend entirely; the guest now keeps ONLY:
  - `.cargo/config.toml`: `--cfg getrandom_backend="custom"` (+ `-C link-arg=--allow-undefined`)
  - `lib.rs`: `#[no_mangle] unsafe fn __getrandom_v03_custom(*mut u8, usize) -> Result<(), getrandom::Error>`
    delegating to the host import `host_random`.
- **Frozen-stdlib construction (the exact 0.5 API).** `Interpreter::without_stdlib` +
  `freeze-stdlib` panics in `VirtualMachine::initialize` ("essential initialization
  failed") because it registers an EMPTY frozen set and essential init then fails to
  `import encodings`. The supported embedded path is the BUILDER, registering the
  frozen stdlib from the **`rustpython-pylib`** crate:

  ```rust
  use rustpython_vm as vm;
  vm::Interpreter::builder(vm::Settings::default())
      .add_frozen_modules(rustpython_pylib::FROZEN_STDLIB)
      .build()
  ```

  KEY GOTCHA: on rustpython-vm 0.5 the `freeze-stdlib` cargo feature is merely an
  alias for `encodings` — it does NOT itself embed the stdlib bytecode. You MUST add
  `rustpython-pylib = { version = "0.5", features = ["freeze-stdlib"] }` as a direct
  dependency and pass `rustpython_pylib::FROZEN_STDLIB` (a `&FrozenLib`, which iterates
  as `(&'static str, FrozenModule)` — exactly what `add_frozen_modules` wants).
- **`print` needs the `stdio` feature.** With `default-features=false` and no `stdio`,
  `sys.stdout` is `None` and `print(...)` raises (eval returned status 1). Enabling the
  `stdio` feature WITHOUT `host_env` selects the VM's SANDBOX stdio path
  (`stdlib::sys::SandboxStdio` -> `std::io::stdout()`), which on
  wasm32-unknown-unknown discards output (no real fd) — a capability WIN: `print`
  succeeds (status 0) with ZERO host effect. Final guest feature set:
  `["compiler", "encodings", "freeze-stdlib", "stdio"]`. We deliberately OMIT
  `host_env` (would wire native `posix`/`os`), `wasmbind` (pulls wasm-bindgen/js — the
  exact thing we eliminated), `gc`, `jit`, `threading`.
- **Verified imports.** `wasm-tools print … | grep '(import '` on the release guest
  shows EXACTLY `(import "env" "host_random" …)` — nothing else.
- **`import os; os.system('echo PWNED')` is inert.** With `host_env` off there is no
  native posix backend, so `os.system` raises a Python exception (eval status 1, NOT a
  trap, NOT a host process). `case2_os_system_is_inert` asserts it never returns 0.

### 2. Subject-scoped Store

- New LOCAL newtype `Subject(pub Option<String>)` in `hyprstream-wasm` (no dep on
  hyprstream-rpc/hyprstream-vfs yet — just the seam). `Subject::anonymous()` /
  `Subject::named(id)` / `id()`.
- P0's fixed `HostState` is generalized to `SandboxState { subject: Subject, rng_bytes }`,
  carried as the wasmtime `Store` data. `host_random` reads `caller.data().subject` —
  capability host fns are Subject-scoped by construction.
- `Sandbox` is bound to one `Subject` at construction (`from_bytes_for(wasm, subject)`;
  `from_bytes` = anonymous). Each `eval` builds a FRESH `Store` from the sandbox's own
  subject — no global/thread-local state (the leak that killed native #488).
- `subject_isolation_no_leak` test: two sandboxes bound to `alice` / `bob` keep
  independent subjects across evals — PASSES.

### 3. Real wall-clock DoS bound

- New `EpochTimer::spawn(engine, tick)` — a background thread calling
  `engine.increment_epoch()` every `tick` (default test cadence 10ms); `Drop` stops it.
  The `Engine` is cheaply cloned (shared epoch counter) into the thread.
- `Sandbox::eval_with_epoch_deadline(src, ticks)` sets `store.set_fuel(u64::MAX)` so the
  EPOCH (not fuel) is the limiter, then `store.set_epoch_deadline(ticks)`. The P0 gotcha
  holds: with `epoch_interruption(true)` every store defaults to deadline 0 and traps
  immediately unless pushed out — the fuel path sets `set_epoch_deadline(u64::MAX)`.
- `epoch_wall_clock_bound_traps_runaway` test: timer at 10ms + deadline 20 ticks traps
  `while True: pass` with "wasm trap: interrupt" within ~hundreds of ms — PASSES. Fuel
  remains the deterministic test option (`eval(src, fuel)`,
  `case3_infinite_loop_is_fuel_bounded`).

### 4. VFS host-fn seam (sketch for #483)

- New module `hyprstream_wasm::vfs`. Trait `VfsCapability` with the Profile-A surface:
  `vfs_walk / vfs_open / vfs_read / vfs_write / vfs_stat / vfs_ls / vfs_create`, each
  taking `&Subject` (matching `hyprstream_vfs::Mount`'s methods, which already take
  `&Subject`). `UnimplementedVfs` is the `Err(VfsError::Unimplemented)` default; an
  in-memory test stub proves the Subject threads through and scopes (a "denied" subject
  is refused). A `// P1b/#483:` comment documents that the real backing is
  `hyprstream_vfs::spawn_vfs_proxy(ns, subject) -> Sender<VfsRequest>` held in Store
  data, with sync wasm host fns submitting `VfsRequest { op, reply }` over it (the
  proxy's `VfsOp` deliberately excludes mount/bind/unmount). `hyprstream-vfs` is NOT
  pulled in (would drag the RPC/tokio stack into this minimal crate).

### Remaining notes / next step

- `import io` (the pure-Python frozen `io` module) still returns status 1 — it is not
  needed for `print` (which uses native `_io` + sandbox stdio directly) and no test
  depends on it. If a future guest needs `io`, investigate the frozen `io` module's
  init under the sandbox stdio path. NOT a blocker for P1.
- Next step (P1b/#483): replace `vfs::UnimplementedVfs` with the real
  `spawn_vfs_proxy`-backed impl held in `SandboxState`, and add the `env::vfs_*` host
  fns to `build_linker` (sync shims over the proxy `Sender`), reading
  `caller.data().subject` for scoping.

---

## P2 RESOLUTIONS (branch `ewindisch/483-python-profile-a`) — #483 `/lang/python`

All three P2 deliverables landed. `cargo test -p hyprstream-wasm` is GREEN (1 lib +
11 integration + 1 mount unit test); the guest builds to `wasm32-unknown-unknown`
importing EXACTLY `{ env::host_random, env::vfs_cat, env::vfs_ls, env::vfs_echo,
env::vfs_ctl }` (verified with `wasm-tools print`).

### 0. hyprstream-vfs torch-free verdict: CONFIRMED

`cargo build -p hyprstream-wasm` with `hyprstream-vfs` + `hyprstream-rpc` added as
deps builds with NO LIBTORCH. Neither crate references `tch`/`torch` — libtorch is
linked ONLY by the top-level `hyprstream` crate. So the canonical VFS seam is usable
directly from the wasm host. The P1 crate-local `Subject` newtype is REPLACED by
`pub use hyprstream_rpc::Subject` (re-exported as `hyprstream_vfs::Subject`).

Canonical seam shapes used (from `crates/hyprstream-vfs/src/proxy.rs`):
- `spawn_vfs_proxy(ns: Arc<Namespace>, subject: Subject) -> tokio::sync::mpsc::Sender<VfsRequest>`
  — the proxy is PINNED to one Subject at spawn time (identity is implicit, not
  per-request — the guest cannot change identity).
- `struct VfsRequest { op: VfsOp, reply: std::sync::mpsc::SyncSender<Result<Vec<u8>, String>> }`
- `enum VfsOp { Cat{path}, Ls{path}, Echo{path,data}, Ctl{path,cmd}, MountPrefixes }`
  — PATH-based, and DELIBERATELY excludes mount/bind/unmount (its security
  invariant). We backed the capability against this real seam rather than the P1
  fid-based sketch (`Mount::walk/open/read/...`), so the #483 capability is exactly
  the proxy's bounded surface.

### 1. Real VFS host-fn backing (the capability is real)

- New `vfs::VfsProxyHandle { tx: Sender<VfsRequest>, subject }` stored in
  `SandboxState` (`Option` — `None` = deny-by-default). `submit(op)` is the
  sync-over-async bridge: `tx.blocking_send(VfsRequest { op, reply })` then block on a
  `std::sync::mpsc::sync_channel(1)` reply receiver. (blocking_send => the driver must
  be a plain thread, NOT a tokio worker.)
- `build_linker` now wires `env::vfs_cat/vfs_ls/vfs_echo/vfs_ctl`. Each reads
  `caller.data().subject` for audit and submits via `caller.data().vfs` — NEVER a
  guest-supplied identity. The reply is serialised into a guest buffer (allocated via
  the guest's own `alloc` export) as `[status_byte][payload]` and returned packed as
  `(out_ptr<<32)|out_len` (the i64 ABI the guest's `take_reply` decodes).
- `define_unknown_imports_as_traps` still fences everything else (deny-by-default).
- TEST `vfs_e2e::guest_vfs_is_subject_scoped` (PASSES): a GUEST `vfs_probe` call
  (cat/echo) goes guest -> `env::vfs_*` -> Subject-scoped proxy -> in-memory `Mount`.
  `alice` reads `/config/temp`, writes `0.9`, reads it back; a `denied` Subject is
  refused at the `Mount` (`permission denied`) — proving the Subject reaches the
  backend. `guest_vfs_deny_by_default` (PASSES): a sandbox WITHOUT `.with_vfs(...)`
  gets the "no capability" reply, never touching a Namespace.

### 2. Guest `/lang/python` semantics (re-expressed on RustPython 0.5)

- Guest gained a PERSISTENT interpreter: a `thread_local Shell { interp, globals }`
  built once (wasm32 is single-threaded, so the thread-local is the whole guest
  state). `globals` is a persistent `PyDictRef` reused across calls via
  `Scope::with_builtins(None, globals.clone(), vm)` — user state survives between
  calls (the native shell's design).
- New guest export `py_op(op, ptr, len) -> i64` (packed reply ABI):
  - eval  -> `repr(result)` (+ `\n---\n<stdout>` if it printed)
  - exec  -> captured stdout
  - list_vars / list_defs -> newline-joined non-dunder names (defs = callables only)
  - get_var / get_def     -> repr of one named global (NONE tag if absent)
- STDOUT CAPTURE on 0.5: the pure-Python `__Capture` surrogate is PORTED verbatim
  (`sys.stdout = __Capture()` with a list buffer + `getvalue()`; drained back to a
  no-op `__Sink`). No `io.StringIO` (avoids extra frozen-module init under the
  sandbox-stdio path). A `__Sink` baseline is installed at shell construction so
  stray `print()` never crashes on a `None` stdout. 0.5 API notes: `run_code_string`
  -> `run_string`; `PyStr::as_str()` is gone — use `to_str()/to_string_lossy()`.
- Host driver `PyShell` (in lib.rs): holds ONE long-lived `Store`+`Instance` so guest
  persistence is real, exposes `eval/exec/list_vars/get_var/list_defs/get_def`
  returning `PyResult::{Ok,Err,None}`. The legacy `eval(ptr,len)->i32` export
  (fresh-interpreter #505 path) is KEPT so all original #505 host tests still pass.
- TESTS `pyshell_eval_exec_and_persistent_scope` + `pyshell_vars_and_defs` (PASS):
  `2+3 -> "5"`, `print('hello') -> "hello\n"`, a var set by exec is visible in a later
  eval, vars/defs enumeration + dunder exclusion + callable-only defs all verified.

### 3. `/lang/python` Mount (torch-free placement)

- New module `hyprstream_wasm::mount` (INSIDE `hyprstream-wasm` => stays
  scoped-buildable; no new crate needed). `PythonMount` implements the canonical
  `hyprstream_vfs::Mount` with the EXACT native layout: `eval` (ctl: expr->repr),
  `stdout` (ctl: stmts->captured stdout), `vars/` (dir of non-dunder globals),
  `defs/` (dir of callables). `walk/open/read/write/readdir/stat/clunk` ported.
- Because `PyShell` must run off the async runtime (its `vfs_*` host fns
  `blocking_send`), `PythonMount` holds a `tokio::mpsc::Sender<PyCommand>` and
  `PythonMount::spawn(sandbox, fuel)` starts a dedicated OS thread that owns the
  `PyShell` and serves commands via `rx.blocking_recv()` — the same channel pattern
  the native mount used for its `!Send` interpreter. `spawn` returns a single
  `PythonMount` that OWNS the driver `JoinHandle`; on `Drop` it drops the command
  sender FIRST (so the driver's `blocking_recv` returns `None` and the loop exits)
  THEN joins — folding both into one type avoids a drop-ordering deadlock where a
  separate guard would join while the sender is still alive.
- TEST `mount::tests::mount_eval_vars_defs_over_guest` (PASSES) against an in-memory
  VFS: write `2 + 3` to `eval` -> read `5`; exec `x = 42` via `stdout` -> read `42`
  under `vars/x`; `def f()` -> `f` appears in `defs/` readdir.

### Stubbed / deferred + precise next step

- DAEMON NAMESPACE WIRING (out of scope, documented): registering this mount into the
  running `/lang/python` namespace lives in the torch-bound `hyprstream` crate. A
  `// #483 daemon wiring:` block in `mount.rs` gives the exact snippet
  (`Sandbox::from_bytes_for(..).with_vfs(VfsProxyHandle::new(spawn_vfs_proxy(ns,subj),
  subj))` -> `PythonMount::spawn` -> `ns.mount("/lang/python", Arc::new(mount))`).
  NEXT STEP: in the daemon's per-session Namespace builder (where `/lang/tcl` is
  mounted), embed the guest `.wasm` (build artifact / `include_bytes!`) and call that
  snippet with the session's verified Subject + the session Namespace.
- GUEST PYTHON BUILTINS for VFS: the guest exposes `vfs_*` end-to-end via the
  `vfs_probe` export and Rust helpers (`guest_cat`/`guest_ls`/`guest_echo`/
  `guest_ctl`), but does NOT yet register them as Python builtins (`cat`/`ls`/`write`/
  `ctl`) callable from guest source — that needs native-module registration which the
  current feature set (no `host_env`) omits. Follow-up: register them as native
  builtins so `cat("/config/x")` works from Python. The capability itself is already
  real and tested through the host fns.
- No blockers.
