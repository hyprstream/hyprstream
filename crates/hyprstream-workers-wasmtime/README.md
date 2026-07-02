# hyprstream-workers-wasmtime

Embedded [wasmtime] host for sandboxing untrusted wasm guests behind capability
profiles. The crate loads an arbitrary guest module and runs it with a
deny-by-default capability surface plus DoS bounds. It is **torch-free** (does not
link libtorch), so it builds without `LIBTORCH`.

## Profiles

- **Profile A** — [`Sandbox`] (`src/lib.rs`): a bespoke `Linker` that exposes exactly
  the host functions we choose (`env::host_random` + the Subject-scoped `env::vfs_*`
  capabilities) and wires every other declared import to a trap via
  `define_unknown_imports_as_traps`. No `wasmtime-wasi`, no preopens, no syscall
  surface. `Sandbox::call_export(name, input, fuel)` is the generic entry point: it
  ships a byte payload to a named guest export over the guest's `alloc`/`memory` ABI.
  The [`python`] profile module builds its `eval`/shell on top of `call_export`.
- **Profile B** — [`wasi_sandbox::WasiSandbox`] (`src/wasi_sandbox.rs`,
  `src/wasi_fs.rs`): a real WASI preview1 surface whose ONLY filesystem is a
  Subject-scoped [`hyprstream_vfs::Mount`]. No host preopen; clocks, randomness,
  sockets, and scheduling are withheld.

Capabilities are Subject-scoped **by construction**: the per-call `SandboxState`
carries the `Subject` the call runs as, and host functions read it from the `Store`
rather than trusting any guest-supplied identity. There is no global/thread-local
identity state, and the Mount is the single policy enforcement point.

## Building the guests

The guest crates are excluded workspace members compiled to wasm targets:

- `hyprstream-workers-python-guest` — RustPython, `wasm32-unknown-unknown` (Profile A).
- `hyprstream-workers-wasmtime-fsguest` — `wasm32-wasip1` command (Profile B).

> **Footgun:** the pyguest's entropy backend is selected by a build cfg in its own
> `.cargo/config.toml` (`--cfg getrandom_backend="custom"` + `-C
> link-arg=--allow-undefined`). Cargo only reads `.cargo/config.toml` from the
> current working directory and its ancestors — **not** from the directory named by
> `--manifest-path`. Build the pyguest **from its crate directory** so the cfg
> applies; otherwise the custom getrandom backend is silently not selected.

```sh
(cd crates/hyprstream-workers-python-guest && cargo build --release --target wasm32-unknown-unknown)
(cd crates/hyprstream-workers-wasmtime-fsguest && cargo build --release --target wasm32-wasip1)
```

Point the tests at the artifacts (otherwise they skip locally, and **fail under CI**
when `CI` is set):

```sh
export HYPRSTREAM_PYGUEST_WASM=crates/hyprstream-workers-python-guest/target/wasm32-unknown-unknown/release/hyprstream_workers_python_guest.wasm
export HYPRSTREAM_FSGUEST_WASM=crates/hyprstream-workers-wasmtime-fsguest/target/wasm32-wasip1/release/hyprstream-workers-wasmtime-fsguest.wasm
cargo test -p hyprstream-workers-wasmtime
```

## wasmtime 46 notes (durable gotchas)

Pinned `wasmtime = "=46.0.1"`. The DoS-guard API:

- **Fuel** (deterministic instruction budget): `Config::consume_fuel(true)` then
  `store.set_fuel(n)`. Exhaustion traps with `"wasm trap: all fuel consumed"`.
- **Epoch** (wall-clock bound): `Config::epoch_interruption(true)`,
  `store.set_epoch_deadline(ticks)`, and a thread calling `engine.increment_epoch()`
  on a cadence (`EpochTimer`). Firing traps with `"wasm trap: interrupt"`.
- **Gotcha:** with `epoch_interruption(true)`, every store defaults to epoch deadline
  `0` and traps immediately unless the deadline is pushed out. The fuel-only path must
  `set_epoch_deadline(u64::MAX)`; the epoch path must give ample fuel
  (`set_fuel(u64::MAX)`) so the epoch (not fuel) is the limiter.
- **Error type:** wasmtime 46 ships its own `wasmtime::Error` / `wasmtime::Result`
  with the `wasmtime::error::Context` trait (re-exported from `anyhow`). Use those and
  `wasmtime::bail!`; a `func_wrap` closure must return `wasmtime::Result<_>`.

## preview1 vs preview2 (Profile B verdict)

To back a WASI preopen with an arbitrary async filesystem (a `Mount` trait, not a
cap-std host `Dir`), use the **legacy preview1** path via `wasi-common`'s `WasiDir` /
`WasiFile` trait objects, registered with `WasiCtx::push_preopened_dir`. The
component-model `wasmtime-wasi` (preview2) exposes the filesystem as host resources in
a `ResourceTable` and its preopen takes a concrete `cap_std::fs::Dir`; at 46.x there is
no first-class public "implement your own async directory/file trait" seam, making a
custom VFS a much larger lift. Profile B therefore builds the `WasiCtx` by hand (not
`sync::WasiCtxBuilder`) and pushes a single `Box<dyn WasiDir>` VFS preopen.

[wasmtime]: https://wasmtime.dev/
