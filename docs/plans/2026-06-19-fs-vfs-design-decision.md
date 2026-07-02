# FS/VFS architecture — converged design (decision)

**Date:** 2026-06-19 · Worker GA FS track (#341, FS-A #362 / FS-B #363 / FS-C #364 / FS-D #365 / FS-B0
#366 + native-overlay backlog). Supersedes the open questions in `2026-06-19-container-into-vfs-spike.md`.

## The stack (user-directed)
```
Cloud Hypervisor
  → fuse-backend-rs            (vhost-user-fs SERVER TRANSPORT — reused, not reimplemented)
    → hyprstream-vfs           (9p-inspired COMPOSITION + Subject-per-call; OWNS the overlay TRAIT)
      → mounts { RAFS (rootfs lower), OverlayFs-wrap (rootfs), RPC-fs, injected: /stream, models, deltas }
```
`fuse-backend-rs`'s server dispatches FUSE ops to a `FileSystem` impl → `hyprstream-vfs`'s `Namespace`
IS that `FileSystem` (down-adapter). Anything already a `fuse_backend_rs::FileSystem` (RAFS via
`nydus-rafs`, `OverlayFs`, `PseudoFs`) plugs in as a Mount (up-adapter). **Bidirectional
`FileSystem ↔ Mount` bridge.** Every guest op flows through a Mount call carrying the `Subject` →
**uniform Subject-per-call over the rootfs too** (#353/#319/#328 boundary).

## Overlay/CoW decision = OPTION B (own the interface now, native impl next)
**hyprstream is an OS → it OWNS the overlay/CoW primitive long-term.** Staged:
- **Own the ABSTRACTION now:** FS-C extends `Mount` → an `FsMount` supertrait (create/remove/rename/
  whiteout/setattr/symlink) — hyprstream's overlay interface + policy + Subject-per-call. Supertrait
  (not a `Mount` change) to limit blast radius: existing read/write Mounts stay on `Mount`; only
  full-FS mounts (rootfs, overlay) implement `FsMount`.
- **v1 backend = wrap `fuse_backend_rs::OverlayFs`** behind that trait → correct from day one (Kata's
  production overlay), Worker GA unblocked, consumers code against hyprstream's interface.
- **vNext = NATIVE hyprstream CoW** (backlog ticket) implementing the same trait → the strategic OS
  payoff `OverlayFs` structurally CANNOT give: **per-tenant writable CoW layers over shared RO model
  weights**, **TTT/LoRA-delta snapshots**, **sandbox snapshot/restore**. Built deliberately, behind the
  owned interface, swappable with zero consumer churn.

**Wanix is NOT the overlay** (checked): its union is "later-binds-win" (NOT CoW — no copy-up/whiteouts)
AND it's browser/WASM-only (our integration `hyprstream-9p`/`wanix_mount` is all `cfg(wasm32)`). Wanix
stays the **browser-side** namespace (`/wanix/`, compositor/chat-core wasm tasks) — same side as the
streaming relay (browsers can't iroh). Native sandbox rootfs overlay ≠ Wanix's domain.

## FS track (updated)
- **FS-B0 #366** — build the RAFS bootstrap from pulled OCI layers (store.rs `.json`→`.meta` gap). Precursor to FS-B. **UNBLOCKED.**
- **FS-C #364** (reframed) — extend `Mount`→`FsMount` supertrait + bidirectional `FileSystem↔Mount` adapter + **wrap `fuse_backend_rs::OverlayFs` as the v1 overlay backend**. (Native CoW → backlog.) **UNBLOCKED** (foundational; gates FS-A/FS-B). Unit-test with passthrough layers.
- **FS-A #362** — `hyprstream-vfs-server` crate: `fuse_backend_rs` `Server` + `Namespace→FileSystem` down-adapter; CH ShareFs attach via the **embedded** hypervisor crate (#343=embed). Depends FS-C trait.
- **FS-B #363** — RootfsMount = `OverlayFs(RAFS-lower in-process + writable-upper)` wrapped as a mount. Depends FS-B0 + FS-C adapter.
- **FS-D #365** — per-sandbox `Namespace`+`Subject` isolation + port wasm-only injected mounts (`/stream`, models, deltas) native. Depends FS-A.
- **NEW backlog** — native hyprstream CoW/overlay (vNext, behind `FsMount`; unlocks model-weight CoW + delta/sandbox snapshots).

## Sequencing
**Now (parallel, disjoint):** FS-B0 (image/store.rs) ∥ FS-C (hyprstream-vfs trait + adapter + OverlayFs wrap).
Then FS-A (after FS-C trait) → FS-B (after FS-B0 + FS-C adapter) → FS-D (after FS-A). Native overlay = backlog.
All gated work (CH-attach) uses #343=embed.
