# Container-rootfs-into-VFS composition — spike (Worker GA FS track)

**Date:** 2026-06-19 · read-only spike · feasibility for #341 FS-A..D (#362-365). Verdict: **feasible;
VFS-as-single-composition-layer matches the existing design grain and subsumes the direct-RAFS attach.**

## Ground truth (code)
- **`hyprstream-vfs` = in-process mount table, NO protocol server.** `Mount` trait (`mount.rs:113`) is
  9P2000-shaped + **read/write-ONLY**: walk/open/read/write/readdir/stat/clunk — **no create/remove/rename**
  (only OTRUNC/ORCLOSE open-mode bits). Subject-per-call security is REAL + enforced (SyntheticTree
  fid-ownership, `services/fs.rs`). `Namespace` already does union/overlay (`BindFlag::{Replace,Before,After}`,
  dedup-merged readdir) + per-task `fork()` — exactly the composition layer wanted. Mount mutation
  (mount/bind/unmount) deliberately NOT exposed to untrusted callers (good for isolation). Most Mount impls +
  the `/stream/` mount are **wasm32-only** → need native counterparts.
- **`hyprstream-9p` = client + codec only, NO server** (doc-comment "server" is a misnomer). `msg.rs` codec
  reusable; `wanix_mount` is the inverse (9p-client-AS-Mount, wasm32). Server loop + fid table = net-new.
- **OCI/Nydus RAFS = half-built on BOTH ends:** `image/store.rs pull()` writes a JSON metadata, NOT a RAFS
  bootstrap (`.json` written vs `.meta` expected by `bootstrap_path()` — mismatch); `runtime/virtiofs.rs`
  spawns external nydusd but never touches hyprstream-vfs; `kata_backend.rs:276` passes the virtiofs socket
  into `create_hypervisor` which **only logs it** — no ShareFs/add_device, daemon never attached to the VM.
  No injected paths (models/deltas/stream) on the native worker path today.

## Answers
- **Feasible + subsumes direct-RAFS:** RAFS becomes a backing Mount (FS-B) behind ONE vhost-user-fs bridge
  (FS-A) → one bridge into the guest, not two. Nothing to throw away (direct path is unfinished).
- **(a) Perf:** recommend **in-process RAFS** (crate already deps `nydus_storage` + `fuse-backend-rs 0.12`)
  = ONE hop (vhost-user-fs server → RAFS read), preserves lazy chunk-CAS, no external nydusd, no double FUSE.
  External-nydusd shape = two hops (avoid).
- **(b) Overlay/CoW:** fits the union model (`BindFlag::Before` writable-upper over RO RAFS lower; union
  walk/readdir already prefer upper) — BUT the **Mount trait is read/write-only**, so copy-up/whiteout/
  rename/unlink/mkdir/chmod/truncate are **not expressible** → **FS-C must EXTEND the Mount trait** (richer
  `FsMount` superset) + implement CoW + whiteout + upper-layer persistence/eviction. **Largest net-new.**
- **(c) async-Mount↔FUSE bridge:** bridges (native `Mount: Send+Sync` + async_trait), but real work: FUSE/
  vhost-user-fs is request/response over a vring with its own fid/inode space + full op set; the 7-verb Mount
  surface must be mapped+extended; wasm Mounts' `unsafe impl Send/Sync` "single-threaded" assumptions DON'T
  hold native (SyntheticTree already genuinely Send+Sync). Server loop + fid/inode table = net-new.
- **Tenant isolation:** Subject-per-call exists+enforced; bind each guest's vhost-user-fs connection to the
  sandbox `Subject` + its own **forked Namespace** (image + injected paths only). This IS the #353/#319/#328
  intersection — the Subject at the Mount boundary is the Casbin/per-host policy principal; one authz
  vocabulary, checked at the Mount boundary, not a second FS ACL system.

## Refined decomposition (vs filed #362-365) + NEW ticket
- **FS-B0 (NEW, blocking precursor — FILE IT):** generate the RAFS bootstrap from pulled OCI layers (close
  the `store.rs` `.json` vs `.meta` gap). FS-B depends on it.
- **FS-A (#362):** split into (i) native **VFS→vhost-user-fs server** (fid/inode table, op→Mount mapping,
  multi-thread-safe) and (ii) the smaller, isolatable **CH ShareFs/device attach** (close the
  `kata_backend.rs:276`/`create_hypervisor` gap).
- **FS-B (#363):** RootfsMount over RAFS, **in-process** (nydus_storage/fuse-backend-rs), one hop. Depends FS-B0.
- **FS-C (#364):** **extend the Mount trait** (create/remove/rename/whiteout/attr) + CoW upper + persistence/
  eviction. **Heaviest; biggest under-scoped risk.**
- **FS-D (#365):** per-connection **forked Namespace bound to sandbox Subject** + reconcile #353/#319/#328;
  ALSO port the wasm32-only injected Mounts (`/stream/`, models, deltas) to native (isolation must span them).

## Risks
FS-C (trait is read/write-only today) and the unticketed RAFS-bootstrap-build gap (now FS-B0) are the two
biggest. runtime-rs 4.0 flux (#342/#343) affects the exact CH ShareFs API.
