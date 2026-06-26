# WASM/WASI worker sandbox — convergence spike

Spike for #483 (WASM-Python) extended to a **unified WASM/WASI execution substrate** that
converges three threads currently designed in isolation:

1. **#483** — WASM-sandboxed RustPython shell at `/lang/python/` (embedded wasmtime, VFS-only).
2. **CRI/OCI worker backends** — #346/#484 (podman/OCI), #348/#486 (autoselect), #341 epic.
3. **Wanix in Workers** — run Wanix (a WASI Plan9 OS) server-side as a worker workload.

**Verdict: viable and worth converging now.** Add **one** embedded `wasmtime` substrate, used by
*both* the interactive `/lang/*` shells (#483) *and* a new `BackendType::Wasm` worker backend; make
its WASI filesystem **backed by the Subject-scoped hyprstream 9P/VFS**, so Python, Wanix, and
waxterm TUIs all see the VFS as their only filesystem. Do **not** start from runwasi (see §3).

---

## 0. Ground truth (verified in-tree)

- **No server-side wasm runtime exists.** `Cargo.lock` has only the `wasm-bindgen` family (browser
  build-target tooling) + `wasmparser`/`wasm-encoder` (transitive parsers). No `wasmtime`/`wasmer`/
  `runwasi`/`wasi-common`/`cap-std`. → wasmtime is a **new dependency** regardless.
- **Wanix integration already exists — browser-side.** `crates/waxterm` = "ratatui apps native and
  in the browser **via WASI**" (`cfg(target_os = "wasi")`), with a "Wanix VFS pipe heartbeat" and a
  "WASI stdin poller (non-blocking read from VFS)". `crates/hyprstream-9p` (`wanix_mount.rs`,
  `dma.rs`, `client.rs`) is the browser 9P client over SAB DMA. Wanix refs also in
  `hyprstream-rpc/src/wasm_api.rs`, `hyprstream-tui`, `chat-core`. **The new ask — Wanix *server-side
  in Workers* — does not exist yet.**
- **`SandboxBackend` is a clean seam** (`crates/hyprstream-workers/src/runtime/backend.rs`):
  `backend_type / is_available / initialize / start / stop / destroy / reset / get_pids /
  supports_exec / exec_sync`. `BackendType` = `Kata | Nspawn` only. A `Wasm` variant fits.
- **9P is the convergence hinge.** Wanix speaks 9P; our VFS exposes 9P (#391 translator, merged
  #412). So Wanix's namespace *can be* our VFS namespace with no impedance.
- **#346 explicitly leaves the OCI-runtime door open** — "Decision D2: podman/docker shell-out vs
  embedded OCI runtime (youki/crun)". The wasm path is a natural third option in the same slot.

---

## 1. The core design problem: WASI vs no-WASI in one substrate

The two consumers have *opposite* capability needs:

| Consumer | WASI? | Filesystem |
|---|---|---|
| **#483 Python shell** | **No WASI** (capability sandbox) | VFS host-fns only; zero ambient authority |
| **Wanix / waxterm** | **Yes, WASI** (it's a WASI Plan9 OS / WASI TUI) | expects a POSIX-ish/9P filesystem |

Resolved by a **configurable wasmtime `Linker`, two profiles**, sharing one runtime:

- **Profile A — capability (no WASI):** Linker exposes *only* the VFS host functions
  (`vfs_walk/open/read/write/stat/ls/create`) + a host RNG. This is #483 verbatim. `import os` is
  inert by construction.
- **Profile B — WASI, filesystem backed by the VFS:** a WASI context whose **filesystem is the
  hyprstream 9P/VFS Mount**, not the host. Wanix/waxterm get a real WASI fs, but every path resolves
  to the **Subject-scoped VFS** — so even the WASI guest cannot touch the host; its "OS filesystem"
  *is* the VFS. Network/clock/random capabilities stay withheld or VFS-mediated.

**Mechanism (the key open question):** wasmtime's WASI fs must be virtualized over our `Mount`.
Two candidate APIs — `wasi-common`'s `WasiDir`/`WasiFile` traits (implement a virtual dir over the
9P/VFS Mount; preview1, most direct) vs the component-model `wasmtime-wasi` filesystem
(`wasmtime-wasi` ResourceTable + a custom host). **Open:** confirm which wasmtime WASI layer lets a
preopen be backed by an arbitrary `Mount` rather than a `cap-std` host `Dir`. This is the single
make-or-break unknown for the Wanix half; P0 must validate it.

This is what makes **"the VFS is the OS filesystem" literally true** for a Plan9 OS running in a
worker — the convergence payoff.

---

## 2. `BackendType::Wasm` worker backend

Add a wasm variant to the existing `SandboxBackend` seam:

- `backend_type() = "wasm"`, `is_available()` = wasmtime present (always, once vendored).
- `start()` instantiates a guest module under the shared wasmtime host with Profile A or B per the
  workload's declared capability class (RuntimeClass-style selection alongside `Kata`/`Nspawn`/OCI).
- `exec_sync()` maps to invoking a guest export; `reset()` = drop+reinstantiate (cheap vs Kata VM
  reboot); `get_pids()` = N/A (in-process) → return empty or a logical id.
- Isolation/DoS via wasmtime **epoch interruption** (timer thread + `set_epoch_deadline`); fuel for
  deterministic tests. (Same mechanism #483 needs.)

**Two consumers, one runtime:** the *same* `hyprstream-wasm` host crate powers (i) the interactive
`/lang/python/` + future `/lang/*` shells and Wanix-as-interactive-OS (embedded, in the daemon's VFS
namespace), and (ii) `BackendType::Wasm` for **batch wasm/Wanix jobs** under the worker pool. One
wasmtime integration, two entry points — not two isolation stacks.

---

## 3. Embedded wasmtime vs runwasi (the CRI/OCI relationship)

| Approach | Fit | Verdict |
|---|---|---|
| **Embedded wasmtime** (in-daemon host) | Full control of the Linker → can back WASI fs with our 9P/VFS Mount; in-process, sub-ms for interactive shells; Subject binding trivial | **Recommended substrate.** Required for the VFS-backed-fs thesis. |
| **runwasi** (containerd wasm shim, wasm-as-CRI-workload) | Rides the CRI/OCI plane + RuntimeClass; but the guest gets containerd's WASI (host fs / standard preopens) → backing fs with our VFS is fought, not free | Only if you want pure containerd-managed wasm jobs with host fs — **misaligned** with VFS-everything. Defer. |

So wasm is a **sibling native `BackendType`**, not a CRI-ridden workload. This keeps it orthogonal to
the OciBackend/CriBackend decision: Kata (VM), OCI/podman or CriBackend (containers), **Wasm
(in-process VFS-scoped)** — three peers under one `SandboxBackend` seam, selected by RuntimeClass.

---

## 4. Wanix-in-Workers path

Wanix already runs in the **browser** (waxterm WASI + WanixMount over SAB DMA). Bringing it
server-side:

1. Build Wanix (or a Wanix-compatible WASI image) as a guest module runnable under the embedded host
   (Profile B). **Open:** Wanix's exact wasm/WASI ABI + whether it expects wasip1 or component model
   (drives §1's API choice).
2. Mount its 9P namespace onto the hyprstream VFS via the existing 9P seam (#391/#412) — Wanix's
   filesystem ops become VFS ops, Subject-scoped. `crates/hyprstream-9p` (today browser-only,
   `wanix_mount.rs`) is the reusable client; server-side it binds to the in-process VFS `Mount`.
3. Bind the actor's atproto DID → `Subject` at instantiation (same as #483). One instance = one
   Subject; the Wanix "machine" runs entirely inside the actor's authorization scope.

Result: a Plan9 OS as a worker whose entire world is the Subject-scoped VFS — the cleanest possible
realization of the namespace-as-OS model, and shared infra with #495/#496 (vfsd) and #465 (browser
WanixMount).

---

## 5. CRI/OCI status (verified GH state, 2026-06-26)

| # | Title | State | Relation to wasm convergence |
|---|---|---|---|
| **#341** | Epic: Worker GA (Kata 4.0 + casual OCI) | open epic | parent; wasm backend = a 3rd track |
| **#346** | T2-A rootless podman/docker (or **youki/crun**) OCI SandboxBackend | open | PR #484; D2 (shell-out vs embedded OCI) still open — wasm is a sibling slot |
| **#484** | podman/docker OCI SandboxBackend | **open PR** | drops CRI `PodSandboxConfig`, docker-rootless misdetect, dead code (prior review) |
| **#348 / #486** | availability-based backend auto-selection | open / open PR | `resolve()` would add `Wasm` to the selectable set |
| **#347 / #491** | feature-gate kata/nydus/nspawn/podman backends | open PR | wasm becomes another gated feature |
| **#349 / #487** | workflow scan_repo + routing | open PR | orthogonal (workflow svc), not backend |
| **#344 / #342 / #350** | kata guest-agent vsock / runtime-rs 4.0 / ZMQ→moq | open | kata-track; wasm sidesteps the guest-agent gap entirely (in-process) |
| **#353** | X-AUTHZ: reconcile worker/sandbox policy vocab | open | wasm backend must land under the same Subject/Casbin vocab |
| **#495 / #496** | vfsd virtio-fs host mount + DAX | open / open PR | shares the VFS-into-guest goal; wasm = the in-process peer of vfsd's VM path |
| **#465** | Phase 3.7 WanixMount wasm32 dep (browser) | open | browser counterpart of server-side Wanix here |
| **#170** | zb2 fd/9P container I/O as duplex Pipe | open | relevant to 9P-as-guest-I/O plumbing |
| **CRI-client spike** | `docs/plans/2026-06-26-cri-client-vfs-injection-spike.md` (branch `ewindisch/cri-client-spike`) | doc only | CriBackend = the *container* peer; wasm = the *in-process* peer |

**Not filed:** no `OciBackend`-rework ticket (the CRI→oci-spec / CriBackend rework is still a
pending-decision proposal, not an issue). No `BackendType::Wasm` ticket. Both should be filed if this
direction is accepted.

---

## 6. Unified-substrate thesis + phased plan

**Thesis:** one `hyprstream-wasm` host crate (embedded wasmtime + configurable Linker) is the single
wasm/WASI substrate for: `{#483 Python shell, BackendType::Wasm batch jobs, Wanix-as-OS}` — each with
its filesystem = the Subject-scoped hyprstream 9P/VFS, all under the worker plane's `SandboxBackend`
seam beside Kata/OCI/CRI (RuntimeClass selection), all bound to the actor's atproto Subject.

- **P0 — substrate spike:** vendor wasmtime; run a trivial wasm32 guest under an embedded host; prove
  epoch interruption. **Validate the §1 unknown**: back a single WASI preopen with a virtual
  `Mount`-over-VFS (`wasi-common WasiDir` vs component-model). Make-or-break for Profile B.
- **P1 — `hyprstream-wasm` host crate:** the two Linker profiles (A capability / B WASI-over-VFS);
  Subject binding at instantiation; epoch/fuel CPU bound.
- **P2 — #483 rides P1:** RustPython→wasm32 under Profile A (VFS-only). Closes the #483 redirect.
- **P3 — `BackendType::Wasm`:** wire into `SandboxBackend` + `resolve()`/feature-gate (#348/#347);
  land under the #353 Subject/Casbin vocab.
- **P4 — Wanix-in-Workers:** Wanix guest under Profile B; 9P namespace bound to the VFS; reuse
  `hyprstream-9p`. The namespace-as-OS milestone.

**Hard caveats (carry from #483):** (1) Profile A must grant *zero* WASI; (2) Profile B's WASI fs
must resolve *only* to the VFS (no host preopen); (3) epoch/fuel bound is mandatory — WASM alone does
not stop infinite loops.

**Open questions (verify, do not assume):** wasmtime WASI-fs-over-custom-Mount API (wasi-common
`WasiDir` vs component model); Wanix's exact wasm/WASI ABI (wasip1 vs component); whether Wanix needs
sockets/threads beyond fs (drives the Profile B capability set); runwasi maturity if the CRI-ridden
path is ever revisited.

---

## Recommendation

Accept the convergence. File two issues: **(a) `hyprstream-wasm` embedded substrate + `BackendType::Wasm`**
(P0–P3, parent #341), and **(b) Wanix-in-Workers** (P4, depends on (a) + the 9P seam). Redirect #483
to consume (a) rather than standing up its own one-off host. Keep wasm as a native sibling backend —
not a runwasi/CRI workload — so the VFS-backed-filesystem (the whole point) stays in our control.
