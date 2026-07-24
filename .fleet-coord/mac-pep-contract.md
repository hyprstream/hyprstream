# Shared MAC PEP Contract (epic #1267, lane T3 = #1268)

**Status:** LIVE — consume this interface, do not reinvent it.
**Owner:** #1268 (RPC dispatch PEP). Other lanes: consume the types below.

## Canonical types (all in `hyprstream_rpc::auth::mac`)

### Decision types — consumed by ALL lanes

```rust
/// The MAC decision returned by any PEP.
pub enum MacDecision { Permit, Deny(MacDenyReason) }

/// Why a MAC decision denied. Auditable and testable.
pub enum MacDenyReason {
    NoPepInstalled,     // no PEP installed process-globally — fail-closed default
    NoClearance,        // subject has no derivable clearance (unlabeled subject)
    UnlabeledObject,    // object has no trusted label
    FloorDeny,          // clearance does not dominate the object label (lattice floor)
    StaleAuthority,     // RESERVED — streaming continuation re-check (DEFERRED, see below)
}
```

### Object-label resolution — consumed by ALL lanes

```rust
/// Resolve the trusted SecurityLabel for the concrete object being acted on.
/// NONE ⇒ unlabeled ⇒ deny (D2/D3 — objects deny/clamp, never default-allow).
pub trait RpcObjectLabelResolver: Send + Sync {
    fn resolve(&self, service_domain: &str, method: Option<u16>) -> Option<SecurityLabel>;
}
```

**`method: Option<u16>` note:** only populated on browser carriers (the sealed
transcript's method discriminator). Non-browser RPC callers pass `None` —
labels are at service-granularity for non-browser requests.

### The RPC dispatch PEP — RPC lane ONLY

```rust
/// The mandatory, unavoidable RPC dispatch PEP.
/// Takes &EnvelopeContext — an RPC-transport type the 9P/CAS/events/VFS
/// lanes do NOT have. Those lanes reuse MacDecision/MacDenyReason/
/// RpcObjectLabelResolver but implement their own plane-specific PEP.
pub trait MacDispatchPep: Send + Sync {
    fn check(&self, ctx: &EnvelopeContext, service_domain: &str, method: Option<u16>)
        -> MacDecision;
}
```

**Other lanes (9P/CAS/events/VFS):** do NOT depend on `MacDispatchPep` or
`EnvelopeContext`. Build your own PEP struct that produces `MacDecision` using
your plane's subject context + `RpcObjectLabelResolver` + `SecurityContext::
can_access`. The decision types and the object-label resolver are the shared
seam; the PEP trait is RPC-plane-specific.

### Clearance derivation (no separate trait needed)

The RPC lane derives subject clearance via the existing
`EnvelopeContext::security_context()` (Claims × VerifiedKeyMaterial, S1
invariant). Other planes derive their own `SecurityContext` from their
plane-specific verified identity (9P `Tattach`, CAS manifest writer, etc.).

There is **no `ClearanceProvenance` trait** — each plane derives
`SecurityContext` from whatever verified identity it already has at its
boundary.

## Process-global installation (RPC lane)

```rust
/// Install the RPC dispatch PEP. RwLock-backed (swappable for tests).
/// Until installed, process_request denies every request (NoPepInstalled).
pub fn install_mac_dispatch_pep(pep: Arc<dyn MacDispatchPep>);

/// The installed PEP, if any. None ⇒ deny (NoPepInstalled).
pub fn global_mac_dispatch_pep() -> Option<Arc<dyn MacDispatchPep>>;

/// Convenience: read global PEP, check, or deny if None.
pub fn check_dispatch_mac(ctx, service_domain, method) -> MacDecision;

/// Test helper: install a DormantMacPep (permits everything). Idempotent.
pub fn ensure_dormant_mac_pep();
```

## Integration point in `process_request`

```text
verify_claims(ctx)                      ← existing (line 207)
  ↓
check_dispatch_mac(ctx, domain, method) ← MANDATORY gate (#1268)
  ↓ Permit only
handle_request(ctx, payload)            ← existing (line 228)
```

**Fail-closed:** no PEP installed ⇒ `Deny(NoPepInstalled)`. There is NO
pass-through / dormant path in dispatch. Integration tests install a
`DormantMacPep` via `ensure_dormant_mac_pep()`.

## Streaming continuations — DEFERRED

`MacDenyReason::StaleAuthority` is a **reserved variant** for future
streaming-continuation re-check (when authority may be revoked between
dispatch and continuation execution). It is **NOT used today** — the PEP
check at dispatch time is the sole gate. Continuation re-check is filed as
a follow-up issue ([#1291](https://github.com/hyprstream/hyprstream/issues/1291)).

## What each lane consumes

| Lane | Issue | Consumes |
|------|-------|----------|
| T3 RPC dispatch | #1268 | OWNS this contract; `MacDispatchPep`, `check_dispatch_mac` |
| CAS PEP | #1269 | `MacDecision`, `MacDenyReason`, `RpcObjectLabelResolver` |
| MoQ/event PEP | #1271 | `MacDecision`, `MacDenyReason`, `RpcObjectLabelResolver` |
| OAuth grant PEP | sibling | `SecurityContext`, `can_access` |
| 9P PEP | existing | `NinePAccessDecider` (already ships) |

## Rules (from #547, non-negotiable)

1. **No permissive default.** Missing PEP, missing clearance, missing label ⇒ Deny.
2. **MAC context is never a plaintext contract field** — derive from verified identity.
3. **Method-level Casbin/TE is discretionary** — the MAC PEP is the mandatory floor.
4. **Streaming continuations re-check** — DEFERRED (filed as follow-up; `StaleAuthority` reserved).
