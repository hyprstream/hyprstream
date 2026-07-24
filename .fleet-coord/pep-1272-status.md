# PEP #1272 — VFS direct-API + exec/ctl MAC PEP

**Branch:** `feat/mac-vfs-pep` → PR (refs #1272, epic #1267 T3)
**Status:** structure complete, fail-closed, release-validated; **dormant** (activation is the #1267 B-lane).
**Package focus:** `hyprstream-vfs`; secondary `hyprstream-workers` (exec/ctl).

## What landed
- **`hyprstream-vfs::mac_pep`** — the plane-specific PEP (`NamespacePep`, `NamespaceAccessDecider`, `SubjectContextResolver`, `NamespaceAction`, deny-all defaults) consumes the canonical `MacDecision` / `MacDenyReason` / `RpcObjectLabelResolver` contract. `MacDispatchPep` remains RPC-only as #1288 requires.
- **`Namespace`** — `pep: Option<Arc<NamespacePep>>` field; `cat`/`read_one`/`echo`/`create`/`ctl`/`ls` consult it; subject-less `mount`/`bind_mount`/`unmount` deny when armed; `_as` variants mediate. `resolve_targets` requires a caller and authorizes the write-capable raw handle before returning it. `fork` inherits the PEP.
- **`hyprstream-workers::ExecMount`** — `LifecyclePolicy` seam (fail-closed `DenyAllLifecycle`); `write` threads `caller` into `apply_verb`; stop/kill/destroy gated when armed.
- **`hyprstream::mac::pep`** — `VfsAccessDecider` (audited, reuses WAL + `can_access` + write-direction pause) + `production_vfs_pep(...)` assembly + `DenyUnenrolledSubjects` stub. Missing-clearance, missing-label, floor, write-direction, and subject-less-mutation denials all cross the WAL sink before returning. Extends the shared contract; does NOT reinvent label resolution/clearance/`can_access`.

## Activation posture
- `Namespace::new()` / `ExecMount::new()` = **unenforced** (the documented dormant status quo — CLAUDE.md "MAC current status").
- `set_pep` / `new_with_lifecycle` = **armed**, fail-closed by construction (no permissive path inside the PEP; #547).
- Flipping construction sites to armed = the separately-gated **#1267 B-lane**, NOT this PR.

## Dependencies flagged (fail-closed stubs in place)
- **#698** (production clearance provenance): `DenyUnenrolledSubjects` resolves no `Subject` → armed PEP denies every op until a real `SubjectContextResolver` lands. NO permissive default.
- **#1288 canonical contract** is consumed directly: VFS passes normalized paths through `RpcObjectLabelResolver` with `method=None` and returns canonical `MacDecision` values. A missing `NamespacePep` is dormant/pass-through; once installed, missing clearance, missing labels, and policy denial fail closed.
- **#1196 `verified_tenant`** is inherited from `origin/main`. This PR adds no `EnvelopeContext` literals; every existing literal initializes the field, and global/node-global subsystem boundaries remain unchanged.
- **Write-direction (IFC)**: VFS write/create/mount/raw-handle-class operations deny pending the VFS IFC decision.

## Validation
- `cargo metadata --all-features` → exit 0.
- `buildq -- cargo nextest run --release --profile ci -p hyprstream -p hyprstream-rpc -p hyprstream-vfs --no-fail-fast` → 2305/2305 pass, 0 failed, 9 skipped.
- `buildq -- cargo nextest run --release --profile ci -p hyprstream-vfs -p hyprstream --no-fail-fast` → 1407/1407 pass, 0 failed, 6 skipped (rebased onto `origin/main` at `9b269564d`).
- `buildq -- cargo nextest run --release --profile ci -p hyprstream-vfs -p hyprstream -p hyprstream-rpc --no-fail-fast` → exit 0, 0 failed (rebased onto `origin/main` at `36043b917`).
- New tests: 3 PEP-contract unit tests + 8 `Namespace` integration tests (un-armed non-regression, armed deny-all, clearance-dominates, write-pause, unlabeled-deny, subject-less-mutation-block, fork-inherits-pep, raw-handle resolution denies missing clearance) + audited VFS fail-closed denial regression.

## Final gate
kimi-k3 review (do NOT self-merge / touch the merge queue).
