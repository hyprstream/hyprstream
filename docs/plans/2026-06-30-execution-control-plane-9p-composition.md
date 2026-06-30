# Execution Control Plane & 9P-Native Composition

**Status:** accepted (operator, 2026-06-30) · supersedes the `hyprstream-sandbox` direction in PR #596

## 1. Context
The wasm/python work drifted from hyprstream conventions (a `hyprstream-sandbox` crate overloading the
reserved "sandbox" vocabulary, a `/lang/python` shell bundled into it and wired into no namespace, wasm
treated as a one-off). This note defines the model the codebase is already bending toward (#523/#525/#526
scheduling, #539/#543 9P projection, the spawner, `SandboxBackend`, the Plan9 VFS), and re-grounds the
correction.

## 2. One kind of mountable thing: a 9P file service
- Everything addressable is a **9P file service**; its contract is `nine.capnp` (`Np*`⇄`R*`). `#[service_factory]`
  services already serve it (`registry.capnp` imports the `Np*` types).
- `hyprstream_vfs::Mount` is the **local face**; remote services are reached by a proxy `Mount`
  (`RemoteRegistryMount`/`RemoteModelMount` via `NinePBridge` → `nine.capnp`). In-proc `Mount` and remote
  `nine.capnp` are two transports for one 9P contract. "capnp service" and "9P file service" are the same
  thing; `nine.capnp` is the bridge. #539 = projection; #543 = service→namespace auto-mount.

## 3. Three orthogonal facets (do not conflate)
A managed execution unit has three independent facets:
- **Role / lifecycle:** a **worker workload** (run-to-completion, e.g. a workflow `run:` step) OR a long-lived **service**.
- **Exposure:** a **9P file service** (`/lang/<lang>`) when exposed.
- **Isolation:** an **`ExecutionBackend`** (inproc → kata).

A Tcl/Python interpreter is an **engine** that appears *as a worker* AND *as a 9P service*, on *any* backend.
"Worker" is a role (offloaded execution at any isolation level, incl. in-proc / not-a-VM), not "VM."

## 4. The execution control plane (`/exec`) — WorkersService as process/service manager
`WorkersService` is the control plane for managed execution units (services AND jobs): lazy-spawn, warm-pool,
scale, place, isolate, resolve. Exposed as a 9P tree:
```
/exec/
  backends/{inproc,subprocess,systemd,nspawn,wasm,kata}/  caps  ctl
  classes/{wasm, wasm-in-kata, ...}     # RuntimeClass = namespace recipe (data) + approved fallbacks
  pool/                                 # warm contexts, keyed by (class, clearance-domain)
  instances/<id>/  ctl  status  ns/     # running units — the /proc model
  sched/  ctl  query
```
Spawn = write spec to `/exec/classes/<class>/clone`. Reach = scheduler resolves/mounts an instance; caller
calls it **directly** (control-plane resolves, data-plane direct). `ModelService→InferenceService` = ask
`/exec` → lazy-spawn/place → direct call.

## 5. `ExecutionBackend` seam (unifies spawner + sandbox)
`SandboxBackend` generalizes to **`ExecutionBackend`**; "sandbox" = the isolated subset (wasm/nspawn/kata).
Spawner modes fold in (inproc/subprocess/systemd as backends). `provides`/`requires` capability descriptors
on the inventory `BackendRegistration`:

| backend | requires | provides | isolation (fault / authority) |
|---|---|---|---|
| inproc | daemon runtime | — (top only) | none / ambient |
| subprocess·systemd | OS process | a process (+cgroups) | process / ambient |
| nspawn | linux+systemd | namespaced procs | container / ambient |
| wasm | a process | wasm ctx | in-proc addr / **zero ambient** |
| kata | KVM+CH | guest kernel+procs | microVM / ambient-in-VM |

`start(spec, host_ctx) -> Handle`; spec is service-shape or job-shape; a controller layer manages shape.
Select by **RuntimeClass/trust**, not a linear scalar (keep `auto_selectable`/fail-closed).

## 6. Composition & nesting (Plan9-native)
Composition-of-backends = composition-of-namespaces. `B`-in-`A` legal iff `A.provides ⊇ B.requires`.
- **RuntimeClass = namespace recipe** (Plan9 `namespace(6)`): mount/bind ops + approved fallback list. Data, not code.
- **Algebra enforced at the mount = the MAC PEP (#547).** Composition-legality + authorization unify in one
  `walk`/`mount` decision. No separate validator.
- **Compose via host-context parameterization**, not decomposition: `WasmBackend.start(spec, host_ctx=KataInstance)`
  runs wasmtime in the VM. Whole-backend granularity; don't tear kata apart.
- **Local == remote** (Plan9 cpu/exportfs over 9P-on-QUIC/Iroh).
- Default shallow (host→isolation→runtime). Algebra expresses depth for free; don't build a recursion engine.

## 7. Lifetimes, pooling, placement
- Per-layer lifecycle; **containment `inner.lifetime ⊆ outer.lifetime`** (ref-count inner under outer).
- Warm-pool the expensive outer layers (`SandboxPool` is the outer half); inner ephemeral.
- **Pool sharing partitioned by MAC clearance domain** — key by `(RuntimeClass, clearance-domain)`; no
  cross-IFC co-tenancy in a shared context. (Composition meets #547 here, day one.)
- "Serverless" for stateful GPU = warm-pool + affinity (place where model resident; HRW #526) + lazy-spawn-on-miss.
- Bootstrap: eager control-plane core; lazy leaves.

## 8. Invariants (non-negotiable)
1. **Spawn-a-stack = ONE atomic, authorized, tear-down-on-failure `ctl` op** executed by the backend (recipe = data; execution = atomic). Not client-driven N mounts.
2. **Control-plane resolves; data-plane is direct.** Files are control-only; no per-call file op / Workers proxy.
3. **Fail-closed.** A layer that can't materialize → mount fails → spawn fails → error. Only "downgrade" = explicitly approved fallback. No silent layer-drop (ZSP).
4. **Sandbox = isolation only.** inproc is not a sandbox; the seam is `ExecutionBackend`.
5. **Placement takes a `SecurityContext`** (clearance + quotas).

## 9. Crate map & naming (DECIDED — see [[worker-engine-crate-naming]])
```
hyprstream-workers                 # control plane: ExecutionBackend seam + scheduler + /exec + pool
hyprstream-workers-wasmtime        # embedded wasmtime engine (Profile A cap / B WASI, Sandbox, fuel/epoch)
hyprstream-workers-wasmtime-fsguest# WASI test guest
hyprstream-workers-tcl             # molt engine + /lang/tcl (9P service)
hyprstream-workers-python          # RustPython engine + /lang/python (9P service; runs on -wasmtime)
hyprstream-workers-python-guest    # RustPython → wasm32 guest
hyprstream-vfs                     # Mount/Namespace/Subject/nine.capnp client
```
The engines are **worker execution engines under the WorkersService umbrella** — NOT `hyprstream-<lang>`
(reads as an SDK), NOT `hyprstream-sandbox` ("sandbox" reserved for isolation). They are simultaneously
9P file services and worker workloads; engine-named, not role/SDK-named.

## 10. Decisions (locked)
- /lang/python isolation → **wasm-backed** (on `hyprstream-workers-wasmtime`).
- Composition dynamism → **(a) static curated recipes**, validated/authorized at the mount.
- Service-ness → **9P file services**; interim direct `Mount` OK, converge to `#[service_factory]`/auto-mount (#543).
- Spawn → **backend executes recipe atomically**.
- PR #596 → **re-cut** onto the §9 crate names.

## 11. Staged plan (strangler)
1. **Correction (now):** re-cut PR #596 onto §9 names; build `hyprstream-workers-python` mirroring `hyprstream-workers-tcl`; wire `/lang/python`.
2. **Seam re-seat:** `ExecutionBackend` (+`caps`,`host_ctx`,service-shape); adapter the spawner under it (no behavior change).
3. **`/exec` projection + RuntimeClass recipes** (static) + mount-as-PEP; ship `wasm-in-kata`.
4. **Pilot:** `ModelService→InferenceService` via `/exec` (warm-pool + affinity + lazy-spawn).
5. **Converge** remaining services onto `/exec`; tie in scheduler/discovery/routing (#525/#524/#526).

## 12. Disposition
- #597 (`WasmBackend` as a backend) — keep (correct pattern); dep → `hyprstream-workers-wasmtime`; `auto_selectable:false`.
- #596 — re-cut per §9/§11.
- #488 (closed native python) — harvest its structure as the `hyprstream-workers-tcl`-shaped template.
