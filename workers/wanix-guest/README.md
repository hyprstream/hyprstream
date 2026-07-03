# wanix-guest

A thin **native Go** guest binary that embeds [Wanix](https://github.com/tractordev/wanix)
as a *library* and mounts a hyprstream **9P** export as the **root of its Wanix
namespace** — using Wanix's own mechanisms, with no bespoke verb and no Rust shim.

Part of hyprstream issue **#506** (deliverable 2). It is a foreign-toolchain
(Go) guest artifact and lives **outside the Cargo workspace** on purpose: it is
not a crate and is never built by `cargo`.

## What it is

Given a Unix socket where hyprstream exposes a 9P2000.L server, `wanix-guest`:

1. dials the socket → `net.Conn`;
2. `p9kit.ClientFS(conn, aname)` adapts that connection into a Wanix `fs.FS`;
3. `wanix.NewRoot()` builds a root `Task` carrying its own namespace (`NS`);
4. `root.Register("exec", &native.ExecDriver{})` registers the host-process
   task driver;
5. `root.NS().Bind(hyprfs, ".", ".", fs.BindReplace)` makes the remote 9P tree
   the **authoritative root** of the Wanix namespace.

### This uses Wanix's native bind mechanism — not a verb

The mount is performed with Wanix's own `p9kit.ClientFS` + `NS().Bind`. There is
**no** custom hyprstream verb, no forked Wanix, and no Rust component involved in
the mount. `wanix-guest` is just a small `main` that wires public Wanix APIs
together. This was proven end-to-end by a feasibility spike before productionizing.

## Run modes

```
wanix-guest --sock /run/hyprstream/9p.sock [--aname ""] [--cmd "sh"]
```
Connect to a real hyprstream 9P server and bind it as the namespace root, then:
- with `--cmd`: run the command as a Wanix task (bounded wait for its exit); or
- without `--cmd`: **serve the namespace** — block until `SIGINT`/`SIGTERM`,
  periodically probing the remote 9P root and reconnecting+rebinding if the
  server drops/restarts.

```
wanix-guest --self-test
```
Spin up a throwaway in-process 9P server over a temp socket, bind it as the
namespace root, and list `/`. Exercises the full 9P attach/walk/read + bind
wiring with no external server. Used as the build/CI smoke test.

Flags / env:

| flag | env | meaning |
|------|-----|---------|
| `--sock` | `HYPRSTREAM_9P_SOCK` | path to hyprstream's 9P Unix socket |
| `--aname` | | 9P attach name (empty = default tree) |
| `--cmd` | | command to run as a Wanix task; empty = serve the namespace |
| `--self-test` | | run against a throwaway in-process 9P server and exit |

Connection lifecycle is handled: the single live `net.Conn` is owned by the
guest and never leaked; the serve loop health-probes the remote root and
reconnects with bounded backoff, failing the guest with a clear error only if
reconnection is exhausted.

## Known semantic gap (be honest)

`native.ExecDriver` runs a task's command as a **host OS process** (`os/exec`).
That process is **NOT** rooted/chrooted into the mounted 9P namespace — the bound
tree is the **Wanix-level `fs.FS` view**, reachable through Wanix's namespace API,
not through the host process's POSIX root. POSIX-rooting host exec into the
mounted tree is the **sandbox's mount concern**, handled separately in **PR-C /
`hyprstream-workers`**, not by this binary.

There is a second, related limitation this binary surfaces honestly: in this
*bare* embed, a host-exec task cannot even fully launch, because `ExecDriver`
opens the task's stdio at `#task/<id>/fd/N`, which the full Wanix runtime
provides by binding a console/pipe capability — the minimal embed does not wire
those caps. So `--cmd` allocates + starts the task via the correct native
mechanism, but the started process has no stdio and never reports an exit; the
guest bounds the wait (default 10s) and prints a clear diagnostic pointing at the
deferred sandbox wiring rather than hanging. **The reliable, fully-working mode
today is the default "serve the namespace" mode** (and `--self-test` for the
mount+bind wiring). Wiring task stdio + POSIX-rooting is PR-C's job.

## Building

Not built by cargo. Use the repo script (produces a static, CGO-disabled binary):

```
scripts/build-wanix-guest.sh [output-path]     # default: target/wanix-guest
```

or directly:

```
cd workers/wanix-guest
CGO_ENABLED=0 go build -o wanix-guest .
go vet ./...
go run . --self-test
```

CI builds/vets/self-tests it via `.github/workflows/wanix-guest.yml` (a
standalone Go job, independent of the Rust workflows).

## Dependency pinning

- **Wanix** is pinned in `go.mod` to commit
  `3811507904441f1298ea62df9061083f1d47a799`
  (`tractor.dev/wanix v0.0.0-20260703022758-381150790444`).
- **Mandatory `p9` fork replace** — copied verbatim from Wanix's own `go.mod`:

  ```
  replace github.com/hugelgupf/p9 => github.com/progrium/p9 v0.0.0-20260529042029-b49ec572080f
  ```

  Wanix's `p9kit` imports the path `github.com/hugelgupf/p9` but requires the
  **progrium fork**. Go does **not** apply a dependency's `replace` directives
  transitively, so this line **must** be duplicated in *this* module's `go.mod`
  or the build fails. (Wanix's other replaces — `golang.org/x/sys` → a wasm fork,
  `cbor`, `r2fs` — are wasm/build-specific and are *not* needed by this native
  guest's import subset, so they are intentionally omitted.)
