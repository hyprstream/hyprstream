# PEP #1269 — 9P MAC activation status

**Lane:** T3 / #1269 (epic #1267)
**Branch:** feat/mac-9p-activate
**Status:** STRUCTURE INSTALLED, fail-closed. Ready for kimi-k3 review.

## What landed

- **Write-direction IFC** (`hyprstream-rpc`): `SecurityLabel::can_write_to` /
  `SecurityContext::can_write_to` — Bell–LaPadula *-property (no-write-down).
  Assurance axis is a crypto floor in both directions (does not flip).
- **ReferenceMonitor write-direction** (`hyprstream-9p`): `authorize` selects
  `can_write_to` for `Action::Write`, `can_access` for reads.
- **SessionContext::from_verified_clearance** (`hyprstream-9p`): verified
  clearance but no S6 token — denies at the token gate. Structural constructor
  for the #698-not-wired case.
- **SHARED MAC PEP contract** (`hyprstream/src/mac/pep.rs`):
  - `NinePClearanceSource` trait — clean clearance-input seam (#698).
  - `EnrollmentClearanceSource` — production impl via enrollment resolver
    (fail-closed when no policy / unenrolled).
  - `ClearanceAttachAuthenticator<C>` — `AttachAuthenticator` wrapping a
    clearance source.
  - `production_ninep_reference_monitor` / `enrollment_ninep_reference_monitor`
    — monitor assembler.
  - `NinePAccessDecider::check` — writes resolved via `can_write_to` (was
    blanket-deny `WriteDirectionUndecided`).
- **4 production constructors wired**:
  - WS route (`ninep.rs:288`) ✅
  - WT route (`ninep.rs:571`) ✅
  - worker UDS (`inject_9p_socket` / `prepare_wanix_workload`) ✅
  - kata/vsock (`serve_tenant_vfs_9p`) ✅
  - `BackendCtx.ninep_monitor` threaded through worker layer.
  - `serve_mount_uds` / `serve_mount_vsock[_raw]` accept
    `Option<Arc<ReferenceMonitor>>`.

## What is NOT wired (gated dependencies)

- **#698** — production clearance issuance: `Claims.clearance` field not present.
  `EnrollmentClearanceSource` resolves via the compiled policy enrollment table;
  no enrollment table populated in production → all subjects resolve to `None`
  → deny.
- **S6 sender-bound token** — not wired into 9P `Tattach`. Sessions constructed
  via `from_verified_clearance` (token = None) → token gate denies every op.
- **#699 / name↔object TOCTOU** — `ReferenceMonitor::authorize` resolves labels
  from `ObjectRef::Path` (the cached walked name), not the backend's reached
  object. The `..` / bind / symlink gap is unchanged; activation does not close
  it. Flagged in the `mac_seam.rs` module docs.

## Fleet-coordination notes

- **#1268 contract** (`mac-pep-contract.md`) was NOT published at start. Built
  plane-specific label resolution (genesis resolver) + a clean clearance-input
  seam (`NinePClearanceSource`) as directed by the fallback clause. The
  `NinePClearanceSource` trait is the seam #1268 can consume/extend when it
  publishes the shared claims→clearance contract.
- **`anonymous_floor()` reachability**: not reachable from any production
  constructor (all 4 install `Some(monitor)`). Still used by tests.
