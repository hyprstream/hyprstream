# Worker GA Epic (#341) — preliminary-research spike findings (2026-06-19)

Three read-only spikes for the preliminary-research phase. Maps onto #342/#343 (Kata),
#350 (ZMQ→moq), and a new filesystem/VFS-share area (proposed). Provenance flagged per claim.

## Spike 1 — Kata 3.25.0 → 4.0 / runtime-rs GA (#342/#343)
- **No `4.0.0` tag exists yet** — 4.0 is preview/final-prep. "runtime-rs GA + default" = a *packaging*
  change (flips the shim default Go→Rust); **NOT a new crate-API generation**. The runtime-rs crates
  we embed have been the active codebase since 3.x.
- **API deltas effectively EMPTY:** every symbol `kata_backend.rs` uses (the `Hypervisor` trait
  `prepare_vm`/`start_vm`/`stop_vm`/`cleanup`/`get_pids`, `CloudHypervisor::set_hypervisor_config`,
  `kata_types::config::hypervisor::{Hypervisor, RootlessUser}`, `kata_types::rootless`, `ch-config`)
  is unchanged at `main`. Real cost = transitive dep / `Cargo.lock` churn ("make it build"), not
  source edits. **#342 plan:** bump the 4 kata-* pins (`Cargo.toml:87-90`) 3.25.0 → latest stable
  3.3x (~3.31.0) now; move to the 4.0 branch/tag when it lands.
- **#343 GATING decision — integration surface:** Kata does NOT formally bless the `hypervisor` crate
  as a stable public API (no semver contract), but embedding isn't forbidden and is the only
  Rust-native non-containerd path. Options: (a) keep embedding [REC — low cost], (b) pivot to
  `containerd-shim-kata-v2` [disproportionate: process+OCI+agent], (c) runtime-rs library API
  [doesn't exist]. → keep (a); use Kata-team cooperation to get it blessed. **Open Qs for Kata team:**
  semver-stabilize/publish the hypervisor embedding API? a runtime-rs library API? which tag to pin
  for 4.0? embedder churn/deprecation policy?
- Side deps: host `cloud-hypervisor` binary must upgrade in lockstep (~CH v50); guest-agent exec = a
  NEW ttRPC/vsock surface (separate item, #344); nydus pin (v2.4.0 git, published-API mismatch)
  DECOUPLED from the kata tag.

## Spike 2 — workers ZMQ→moq (#350)
- **Caveat:** spike read the STALE local `main` (pre-moq-epic `2a463263d`); real `origin/main`
  (`f6615fc20`) HAS the moq plane (`dial.rs`/`moq_stream.rs`/moq-net). Re-validate the migration
  TARGET against origin/main, not stale main. Findings below still hold:
- **Already migrated:** workers RPC services are QUIC-enabled (`into_spawnable_quic`,
  `factories.rs:573`); the `*Zmq` clients are gone — now generated transport-generic `RpcClient`
  clients (`WorkerClient` etc.). 
- **Latent ZMQ = the entire `events/` module** (publisher/subscriber/secure pub-sub/XPUB-XSUB
  proxy/endpoints/sockopt) + the `zmq::Context` plumbing it forces into the service structs. The
  secure-event crypto (`event_crypto`, AES-256-GCM) is transport-agnostic — only the carrier changes.
- **CROSS-CRATE:** the main `hyprstream` binary + the inference `StreamService` share this same ZMQ
  event bus (`main.rs:2005/2035`, `factories.rs:253/406`), so it can't be migrated workers-locally.
  Plan as a hyprstream-rpc + hyprstream + workers migration. Gated: shared `ZmqService::context()`
  accessor, a net-new QUIC/moq event broker, `SocketKind` discovery contracts.

## Spike 3 — filesystem / VFS-into-guest (proposed new tickets; user asked for "9P")
- **VERDICT: "9P into the Kata guest" is NOT viable on the current VMM.** Worker sandboxes run
  **Cloud Hypervisor, which supports only virtio-fs (vhost-user-fs), not virtio-9p** (verified in
  Kata's CH backend `ch/inner_device.rs:133` — rejects non-virtio-fs shares; CH docs confirm). In
  runtime-rs, virtio-9p is QEMU/Dragonball-only. Even on QEMU, virtio-9p is server-in-VMM over a host
  dir and CANNOT forward to an external 9P server (proxy backend removed) → real 9P needs
  9p-over-vsock to a host server, bypassing the VMM fs device.
- **RECOMMENDED primary path: a custom vhost-user-fs (FUSE-over-vhost-user) daemon backed by
  hyprstream-vfs**, attached via Kata's existing CH `ShareFs` device. Achieves the goal (map VFS
  read-write into the guest, good POSIX/perf) on the current VMM. NOT 9P. Keep RAFS for read-only
  image rootfs + add the VFS share as a second mount. **9P stays a fallback** only if we add the QEMU
  VMM. → **USER DECISION:** vhost-user-fs-on-CH (recommended) vs add QEMU for real 9P.
- **2nd blocker:** the existing virtiofs/RAFS path is HALF-BUILT — `nydusd` spawned but never attached
  to the VM (`kata_backend.rs:279` `prepare_vm(...,None,...,None)`, no `add_device`/`ShareFs`), and no
  RAFS bootstrap is generated (`image/store.rs` downloads raw OCI blobs, never runs `nydus-image`;
  `.meta` path at `virtiofs.rs:98` doesn't exist, failure swallowed). No working shared-fs baseline
  to extend yet.
- **VFS foundation EXISTS:** `hyprstream-vfs` = Plan9-style namespace multiplexer — async `Mount`
  trait (`walk/open/read/write/readdir/stat/clunk`, each takes `caller:&Subject`), `Namespace`
  `fork()`/`unmount()`/longest-prefix bind mounts, `/stream/` mount. `hyprstream-9p` has a 9P2000.L
  client + codec (client-direction only). Net-new: the protocol SERVER direction (fid/qid/inode
  table, R-encoders), and a `Mount` trait extension for `create/remove/rename/wstat` (writes target
  pre-existing files only today).
- **Spike-split recommendation (new sub-tickets under #341):** Sub-0 (prereq) finish virtio-fs/RAFS
  attach + bootstrap; Sub-A transport-into-guest + VMM decision (vhost-user-fs-on-CH vs QEMU 9p vs
  9p-over-vsock) — resolve FIRST; Sub-B VFS protocol-server impl (+ `Mount` create/remove/rename);
  Sub-C mapping + tenant-isolation security (per-session forked namespace, `Subject` authz,
  path-traversal, writable-prefix allow-list). A+B parallel after A's VMM decision; C depends on both.

## Cross-cutting decisions for the epic
- D-Kata (#343): integration surface — keep embedding, get Kata-team blessing.
- D-FS: 9P-as-asked is blocked on CH → choose vhost-user-fs-on-CH (recommended) vs add QEMU. **User.**
- D-FS-prereq: finish the half-built RAFS/virtiofs attach before/with any VFS share.
- ZMQ→moq is cross-crate (not workers-only); re-scope against origin/main's moq plane.
- All capnp/RPC-surface + auth changes remain human-gated; worker tenant-isolation reconciles with
  #310's #319/#328 (X-AUTHZ #353).
