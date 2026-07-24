# Shared MAC PEP Contract (epic #1267, lane T3 = #1268)

**Status:** LIVE — consume this interface, do not reinvent it.
**Owner:** #1268 (RPC dispatch PEP). Other lanes: consume the types below.

## Canonical types (all in `hyprstream_rpc::auth::mac`)

### Subject side (already S1, re-exported)

```rust
/// Verified subject clearance. Built from Claims × VerifiedKeyMaterial (S1).
/// NONE ⇒ deny (no default clearance, per #547).
use hyprstream_rpc::auth::mac::SecurityContext;
```

### Object side — RPC-plane label resolution

```rust
/// Resolve the trusted SecurityLabel for the concrete object a dispatching
/// RPC acts on. NONE ⇒ unlabeled ⇒ deny (D2/D3 — objects deny/clamp, never
/// default-allow).
///
/// `service_domain` — the canonical service name ("model", "registry", …).
/// `method` — the browser method discriminator if available (u16), else None.
pub trait RpcObjectLabelResolver: Send + Sync {
    fn resolve(&self, service_domain: &str, method: Option<u16>) -> Option<SecurityLabel>;
}
```

### Clearance provenance seam (#698 dependency)

```rust
/// How the PEP obtains a verified subject's clearance from the dispatch
/// context. Production: derive from Claims × VerifiedKeyMaterial.
/// Until #698 wires the clearance field, resolves to None (deny).
pub trait ClearanceProvenance: Send + Sync {
    fn resolve(&self, ctx: &EnvelopeContext) -> Option<SecurityContext>;
}
```

### The mandatory dispatch PEP

```rust
/// The MAC decision returned by the PEP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacDecision {
    /// Access permitted — proceed to handler.
    Permit,
    /// Access denied — handler MUST NOT be called.
    Deny(MacDenyReason),
}

/// Why a MAC decision denied. Audited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacDenyReason {
    /// No PEP installed — fail-closed default.
    NoPepInstalled,
    /// Subject has no derivable clearance.
    NoClearance,
    /// Object has no trusted label.
    UnlabeledObject,
    /// Clearance does not dominate the object label.
    FloorDeny,
    /// Stale continuation authority (streaming re-check).
    StaleAuthority,
    /// Audit failure — decision could not be durably recorded.
    AuditFailClosed,
}

/// The mandatory, unavoidable RPC dispatch PEP.
/// Called by `process_request` between verify_claims and handle_request.
/// If no PEP is installed globally, the default is Deny(NoPepInstalled).
pub trait MacDispatchPep: Send + Sync {
    fn check(
        &self,
        ctx: &EnvelopeContext,
        service_domain: &str,
        method: Option<u16>,
    ) -> MacDecision;
}
```

## Process-global installation (mirror PQ trust store pattern)

```rust
/// Install the node's MAC dispatch PEP. Write-once per process.
/// Until installed, `process_request` denies every request (NoPepInstalled).
pub fn install_mac_dispatch_pep(pep: Arc<dyn MacDispatchPep>) -> bool;

/// The installed PEP, if any. None ⇒ deny-all default.
pub fn global_mac_dispatch_pep() -> Option<Arc<dyn MacDispatchPep>>;
```

## Integration point in `process_request`

```text
verify_claims(ctx)           ← line 207 (existing)
  ↓
mac_pep.check(ctx, domain, method)   ← NEW: mandatory gate (#1268)
  ↓ Permit only
handle_request(ctx, payload)  ← line 228 (existing)
```

## What each lane consumes

| Lane | Issue | Consumes |
|------|-------|----------|
| T3 RPC dispatch | #1268 | OWNS this contract |
| CAS PEP | #1269 | `MacDecision`, `MacDenyReason`, clearance seam |
| MoQ/event PEP | #1271 | `MacDispatchPep` pattern, `RpcObjectLabelResolver` |
| OAuth grant PEP | sibling | `ClearanceProvenance`, `SecurityContext` |
| 9P PEP | existing | `NinePAccessDecider` (already ships) |

## Rules (from #547, non-negotiable)

1. **No permissive default.** Missing PEP, missing clearance, missing label ⇒ Deny.
2. **MAC context is never a plaintext contract field** — derive from verified identity (rule #1).
3. **Method-level Casbin/TE is discretionary** — the MAC PEP is the mandatory floor.
4. **Streaming continuations are re-checked** — stale authority ⇒ Deny.
