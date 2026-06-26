# RustPython VFS shell — WASM capability sandbox (re-architecture of #488)

**Status:** spike / design. Supersedes the native-interpreter sandbox posture of
PR #488 (`crates/hyprstream-python`, branch `ewindisch/python-shell-483`).

## Thesis

The correct sandbox boundary for untrusted Python is the **WASM module boundary**,
not Python-level `__import__` / `importlib` removal. The current #488 crate runs
RustPython **natively in-process** (it depends on `hyprstream-vfs` directly and
bridges VFS calls onto a tokio runtime). In that model, stripping `builtins.__import__`
is best-effort and unwinnable: `import os; os.system(...)` reaches the host because the
interpreter shares the host process's authority. The #488 review confirmed this and
the docs were softened to "best-effort, not a hard sandbox."

Under WASM the property is structural: a WebAssembly instance has **zero ambient
authority** — no filesystem, no network, no clock, no syscalls — unless the host
explicitly grants them as imports. So `import os` either fails to import or imports
but is inert: there is no host function for it to call. The sandbox is real **by
construction**, and the only capability we inject is the hyprstream VFS, scoped to the
actor's atproto identity via the existing per-`Subject` / Casbin boundary.

## The two hard caveats (the design is only safe if BOTH hold)

1. **Zero WASI grants.** Run the guest with **no WASI context** — no `preopen`
   directories, no `wasi:filesystem`, `wasi:sockets`, `wasi:clocks/random` access.
   The instance's import surface must be *exactly* `{ hyprstream VFS host functions }`
   and nothing else. Handing the guest a single WASI preopen or socket re-opens the
   escape. (This is why we do **not** use `wasmtime-wasi`'s default linker; we build a
   bespoke `Linker` exposing only our host fns.)
2. **Fuel/epoch CPU bound.** WASM does not stop `while True: pass`; it still pins the
   executing thread. The runtime must bound it. Use wasmtime **epoch interruption**
   (cheap; a timer thread increments the epoch and the store traps on its deadline) as
   the production DoS guard, with **fuel** as a deterministic option for tests/repro.
   This *replaces* the Python `check_signals` wall-clock watchdog the #488 fix added.

## Findings (the 6 questions)

### 1. RustPython on wasm32 — viable, with one flagged risk
- `rustpython-vm` already targets wasm; the crate enables `freeze-stdlib` (Cargo.toml),
  which embeds the stdlib in the module so there is no on-disk stdlib to mount. The
  upstream project ships wasm examples and is distributed as a wasm module (wasmer.io).
- Server target: build the guest for `wasm32-unknown-unknown` (no WASI) or
  `wasm32-wasip1` and then **withhold the WASI imports** at instantiation. Preferred:
  `wasm32-unknown-unknown` so there is no WASI surface to accidentally wire.
- **getrandom:** in a no-WASI guest, `getrandom` must be satisfied by a host import or
  the `getrandom` *custom* backend (`register_custom_getrandom!` / `custom` feature),
  fed by a host function. The current crate's wasm32 build fails on `getrandom`
  expecting the browser `wasm_js` backend — for the server guest we instead register a
  host-provided RNG. **OPEN QUESTION:** confirm the exact getrandom backend wiring for
  `rustpython-vm`'s transitive `getrandom` at our pinned version.
- **OPEN QUESTION / risk:** historical reports (RustPython#3052) of standalone
  `wasm32-wasi` RustPython binaries not running cleanly under wasmtime/wasmer. We do
  **not** run the standalone WASI binary — we run RustPython as a *guest module under
  our own embedded wasmtime with our own imports* — which sidesteps the standalone-WASI
  path, but the build must be validated end-to-end (compile → instantiate → eval).

### 2. Server-side wasm host — does not exist yet; small addition
- `grep -rnE "wasmtime|wasmer|wasi" crates/ --include=*.toml` finds **no** server-side
  wasm runtime. The only hits are `cfg(not(target_os = "wasi"))` guards in
  `hyprstream-tui` / `waxterm` (i.e. *they* may be built as wasi guests), not an
  embedder. The browser wasm (`hyprstream-rpc` wasm32) is a different thing — it is
  *our code compiled to wasm to run in a browser*, not a host running guests.
- Smallest addition: add `wasmtime` (with `default-features`, plus `Config::epoch_interruption(true)`)
  to a new `crates/hyprstream-python-host` (or behind a feature in the daemon). Build a
  custom `wasmtime::Linker` exposing only the VFS host functions (§3) and the host RNG;
  **no `wasmtime_wasi::add_to_linker`.** Instantiate the frozen RustPython guest module
  per shell session.

### 3. VFS as the sole capability — the seam already exists
- `hyprstream_vfs::Mount` (`crates/hyprstream-vfs/src/mount.rs:113`, `Send + Sync`,
  async) threads `caller: &Subject` through `walk/open/read/write/stat/create`. The
  `Subject` is documented as "the Casbin subject string" (`hyprstream-rpc/src/envelope.rs`:
  `pub struct Subject(Option<String>)`; `Subject::new(name)` / `anonymous()`).
- Per-tenant policy is the *declared* boundary: `fuse_adapter.rs:22,443` carries the
  Subject "for the policy/audit boundary (#353/#319/#328)" but several mount impls still
  `_caller` it — i.e. Casbin **enforcement** at the mount layer is scaffolded and lands
  with the #353/#319/#328 authz-reconcile work. The wasm shell does not need to
  re-implement authz: it threads the bound Subject and inherits enforcement when the VFS
  enforces it (same as the native shell, same as 9P/FUSE callers).
- **Host-function interface** (the only imports the guest gets), each enqueuing onto a
  Subject-bound channel (§4) and blocking the guest on the reply:
  - `vfs_walk(path_ptr,len) -> fid` · `vfs_open(fid,mode)` · `vfs_read(fid,off,count) -> bytes`
  - `vfs_write(fid,off,data) -> n` · `vfs_stat(fid) -> stat` · `vfs_ls(path) -> entries`
  - `vfs_create(path,mode)` · `host_random(buf,len)` (RNG only; no clock/net/env)
  These map 1:1 to `Mount` / `Namespace` methods. The guest never names a Subject.

### 4. Identity binding — reuse `spawn_vfs_proxy`, delete SHELL_CTX
- The #488 review flagged a real cross-tenant bug: `SHELL_CTX` is a **thread-local**
  (`lib.rs`) overwritten on every `PythonShell::new(subject, namespace)`, so two shells
  on one thread collide and one runs under the other's Subject.
- The fix already exists in the VFS: `proxy::spawn_vfs_proxy(ns: Arc<Namespace>, subject: Subject)
  -> Sender<VfsRequest>` (`crates/hyprstream-vfs/src/proxy.rs`) captures **one Subject in
  one task** and applies it to every op. That is exactly "one instance = one Subject."
- Design: at session creation, resolve the actor's **atproto identity → `Subject::new(did)`**,
  call `spawn_vfs_proxy(ns, subject)` once, and give the resulting `Sender` to *this wasm
  instance's* host-function state (`Store` data). The Subject is bound in the proxy task,
  not in any thread-local and not passed by the guest → no spoofing, no cross-tenant leak.
  `SHELL_CTX` is removed entirely.
- Identity provenance: the atproto DID/Subject arrives the same way it does for other
  authenticated surfaces (the RPC envelope `Subject` / session identity); it is set by the
  host at instantiation, never by the guest.

### 5. CPU / DoS — wasmtime epoch (replaces the Python watchdog)
- `Config::epoch_interruption(true)`; per call `Store::set_epoch_deadline(n)`; a single
  background timer thread calls `Engine::increment_epoch()` at a fixed cadence. On
  deadline the store **traps**; the host catches the trap and returns a clean
  "execution time exceeded" to the caller. Epoch is ~2–3x cheaper than fuel and matches
  the wall-clock intent of the current watchdog.
- Offer **fuel** (`Config::consume_fuel(true)` + `Store::set_fuel`) as a deterministic
  alternative for tests / reproducible bounds (same input + same fuel → same trap point).
- **OPEN QUESTION:** exact wasmtime API names/signatures drift across versions — pin a
  wasmtime version and confirm `set_epoch_deadline` / `set_fuel` spelling at integration.

### 6. What `import os` does under zero-WASI
- With `freeze-stdlib`, `os` is importable as a *module object*, but its host-touching
  functions (`os.system`, `os.open`, `os.environ` population, `subprocess`, `socket`,
  `ctypes`) have nothing to call: there is no WASI/host import backing them, so they
  raise (e.g. `OSError`/`unsupported`) or return empty — **inert**, not an escape.
  `importlib` reaching `os` therefore buys nothing, which is precisely why the native
  `__import__`-removal approach was futile and the WASM approach is not.
- Hardening (defense in depth, not the primary control): freeze a **minimal stdlib**
  excluding `socket`, `subprocess`, `ctypes`, `mmap`, `select`, `_thread` where feasible,
  so they are not even importable. Primary control remains the empty capability surface.
- **OPEN QUESTION:** confirm which frozen-stdlib modules RustPython lets us exclude at
  build time vs. which are unconditionally frozen.

## Phased plan

- **P0 — build viability spike.** Compile a trivial `rustpython-vm` eval to
  `wasm32-unknown-unknown`, instantiate under an embedded wasmtime with an empty linker +
  host RNG, run `print(1+1)` and `import os; os.system('x')` (expect inert). Resolves the
  getrandom backend + the standalone-WASI risk. Gate everything else on this.
- **P1 — VFS host imports.** Implement the `vfs_*` host fns over a
  `spawn_vfs_proxy(ns, subject)` `Sender` held in `Store` data; wire `eval/exec` to call
  the guest. Delete `SHELL_CTX`.
- **P2 — identity + epoch.** Bind atproto `Subject` at instantiation; add epoch
  interruption + deadline + timer thread; surface "time exceeded".
- **P3 — mount + parity.** Re-expose the `/lang/python` mount (`mount.rs`) over the wasm
  shell; port the persistent-scope / stdout-capture behavior into the guest; minimal
  frozen stdlib.
- **P4 — authz convergence.** Rides #353/#319/#328 VFS Casbin enforcement; no shell-side
  authz code.

## Verdict

**The WASM-capability approach is viable in our tree** — RustPython compiles to wasm,
the VFS `Mount`/`Subject` seam and `spawn_vfs_proxy` Subject-binding already exist, and
wasmtime epoch/fuel cover the DoS gap. The blockers are a contained build spike (P0:
getrandom backend + confirm the embedded-guest path avoids the standalone-WASI quirk),
not architectural unknowns.

**Recommendation: (a) redirect #488.** Do not merge the native crate as the security
story — its sandbox is unfixable-in-kind. Options:
- Land current #488 **only** if feature-gated **off by default** and documented as a
  non-isolated dev convenience (stopgap), while the wasm host is built; OR
- Close #488 and reopen as `hyprstream-python` + `hyprstream-python-host` (wasm) per this
  doc.
Preferred: **redirect** — keep the crate, swap the execution substrate to the wasm host;
the VFS-facing surface (`vfs_*`, the `/lang/python` mount, persistent scope) is reusable.

## Open questions (verify at integration; do not assume)
- getrandom backend wiring for `rustpython-vm`'s transitive `getrandom` on the server guest.
- The RustPython#3052 standalone-WASI quirk does not affect the embedded-guest path (validate in P0).
- Exact wasmtime version + `set_epoch_deadline` / `set_fuel` API spelling.
- Which frozen-stdlib modules are excludable vs. unconditionally frozen.
- Timing of #353/#319/#328 so VFS Casbin enforcement is actually live when this ships
  (until then the Subject is threaded/audited but not enforced — same caveat as every
  other VFS caller today).
