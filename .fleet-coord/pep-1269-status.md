# PEP #1269 — 9P MAC activation status

**Lane:** T3 / #1269 (epic #1267)
**Branch:** feat/mac-9p-activate
**Status:** REBASED ON #1288/#1196; STRUCTURE INSTALLED, fail-closed.

## What landed

- **Write-direction IFC** (`hyprstream-rpc`): `SecurityLabel::can_write_to` /
  `SecurityContext::can_write_to` — Bell–LaPadula *-property (no-write-down).
  Assurance axis is a crypto floor in both directions (does not flip).
- **ReferenceMonitor write-direction** (`hyprstream-9p`): `authorize` selects
  `can_write_to` for `Action::Write`, `can_access` for reads.
- **SessionContext::from_verified_clearance** (`hyprstream-9p`): verified
  clearance but no S6 token — denies at the token gate. Structural constructor
  for the #698-not-wired case.
- **Audited early denials** (`hyprstream-9p` + production PEP): label
  resolution, token-gate, and IFC-floor denials call the decider's required
  audit hook; `NinePAccessDecider` writes those records through the same
  tamper-evident WAL sink used for final policy decisions.
- **Tenant-bound verified attach identity** (`hyprstream-9p`):
  `VerifiedAttachIdentity` carries subject + tenant from the same verified
  credential. Deny-only sessions carry no verified identity.
- **Plane-specific 9P PEP** (`hyprstream/src/mac/pep.rs`):
  - `NinePClearanceSource` trait — clean clearance-input seam (#698).
  - `EnrollmentClearanceSource` — production impl via enrollment resolver
    (fail-closed when no policy / unenrolled).
  - `VerifiedClearanceSessionFactory<C>` — consumes only an already verified,
    tenant-bound identity; it cannot accept raw `Tattach` fields.
  - `production_ninep_reference_monitor` / `enrollment_ninep_reference_monitor`
    — monitor assembler. The current production helper explicitly installs
    `AnonymousAuthenticator`, because no verified attach credential reaches the
    seam yet; raw `uname` is never promoted to identity.
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

- **#698** — production 9P clearance issuance: although RPC `Claims` can carry
  clearance, the 9P attach path does not yet carry verified identity + tenant +
  clearance/token material. The production authenticator therefore returns a
  deny-only session with no verified identity.
- **S6 sender-bound token** — not wired into 9P `Tattach`. Sessions constructed
  via `from_verified_clearance` (token = None) → token gate denies every op.
- **#699 / name↔object TOCTOU** — `ReferenceMonitor::authorize` resolves labels
  from `ObjectRef::Path` (the cached walked name), not the backend's reached
  object. The `..` / bind / symlink gap is unchanged; activation does not close
  it. Flagged in the `mac_seam.rs` module docs.

## Canonical contract reconciliation

- Rebased onto the merged #1288 contract and retained its canonical
  `MacDispatchPep` / `MacDecision` / `MacDenyReason` /
  `RpcObjectLabelResolver` definitions unchanged. In particular, an
  uninstalled RPC dispatch PEP remains dormant and returns `Permit`; only an
  installed PEP is fail-closed.
- Per `.fleet-coord/mac-pep-contract.md`, 9P does not implement
  `MacDispatchPep` or depend on `EnvelopeContext`; it retains the existing
  plane-specific `NinePAccessDecider`, trusted path/CID label resolution, and
  verified-attach clearance seam.
- #1196's `EnvelopeContext::verified_tenant` initializers arrive from main.
  This PR adds no `EnvelopeContext` initializer. The 9P genesis/content-truth
  resolver is node-global, so it deliberately remains in the global domain;
  tenant isolation continues at the Subject-scoped export boundary.
- **`anonymous_floor()` semantics**: all 4 production constructors install
  `Some(monitor)`. Until verified attach credentials are wired, their
  authenticator returns a deny-only session at the anonymous diagnostic floor
  with no token and no verified identity; every attempted op denies and audits.
