# Clearance Provenance — Resolver Interface Contract (#698)

**Status:** ACTIVE — the interface the RPC-PEP (#1268) and 9P-activate (#1269) lanes consume.
**Branch:** `feat/mac-clearance-provenance`
**Issues:** [#698](https://github.com/hyprstream/hyprstream/issues/698) · [#1267](https://github.com/hyprstream/hyprstream/issues/1267)

## The contract in one sentence

A verified subject identity (atproto DID resolved through the token-exchange path)
maps to an `Option<SecurityLabel>` clearance — `Some` for an enrolled DID, `None`
(fail-closed deny) for an unenrolled or unverified identity.

## Interface signature

```rust
/// Resolve a verified subject identity → MAC clearance context.
///
/// `audience_did` is the DID that was cryptographically verified (UCAN chain
/// validation, OAuth token-exchange signature verification, or 9P Tattach
/// credential). The resolver maps it to a `SecurityContext` carrying the
/// authority-assigned clearance, or returns `None` → the PDP denies.
///
/// SECURITY: clearance is authority-owned (the enrollment table inside a
/// signed `CompiledPolicy`). A principal never authors its own clearance.
/// Assurance is always clamped DOWN to what the verified key material proves
/// (Decision D, #548) — a classical-DID peer cannot claim PqHybrid.
pub trait SubjectContextResolver: Send + Sync {
    fn resolve(&self, audience_did: &str) -> Option<SecurityContext>;
}
```

**Crate:** `hyprstream` → `services::oauth::token_exchange::SubjectContextResolver`
**Canonical types:** `SecurityContext`, `SecurityLabel`, `VerifiedKeyMaterial` — all
from `hyprstream_rpc::auth::mac`.

## The two resolvers

### 1. EnrollmentSubjectContextResolver (the production DID→clearance resolver)

Used by the UCAN grant path (both mint and refresh). Backed by the signed
`CompiledPolicy`'s enrollment table (`BTreeMap<String, SecurityLabel>`).

- **Input:** the audience DID from the verified UCAN grant.
- **Lookup:** `CompiledPolicy::clearance_for(did)`.
- **Assurance:** ALWAYS `VerifiedKeyMaterial::Classical` (Decision D, #698 — a
  delegated actor proves only DPoP possession of a classical ephemeral key).
- **`None` case:** DID not in the enrollment table → `None` → deny.

**Accessed via:** `crate::mac::exchange_enrollment_resolver()` — returns this
resolver when a compiled policy is installed, or `DenyUnlabeledResolver`
(fail-closed) when none is installed.

### 2. ClaimsSubjectContextResolver (the token→clearance resolver)

Used downstream (PEP, RPC handlers) to extract the clearance from a **verified
JWT** the issuer minted. This is the resolver the #1268/#1269 PEP lanes consume
at per-op enforcement time.

- **Input:** verified `Claims` (signed by the issuing node) + `VerifiedKeyMaterial`.
- **Clearance:** read from `Claims.clearance` (authority-asserted, signed).
- **Assurance:** derived from the verified crypto, clamped DOWN via
  `SecurityContext::from_clearance` — never from the claim.
- **`None` case:** `Claims.clearance` absent → `None` → deny.

## The issuer path (what this PR adds)

Before this PR, no production mint path called `Claims::with_clearance`. The
enrollment resolver resolved the DID→clearance for the S6 gate evaluation, then
threw it away — the minted token carried no clearance, so every downstream
consumer saw an unlabeled subject.

**This PR threads the resolved clearance into `mint_grant_token`:**

```
exchange_ucan_grant
  → resolve_grant_subject(grant, exchange_enrollment_resolver())
    → subject_ctx: Option<SecurityContext>           // the enrollment-table clearance
  → audited_evaluate_grant(…, subject_ctx, …)         // S6 gate (unchanged)
  → mint_grant_token(…, subject_ctx.clearance())      // NEW: stamp on Claims
    → claims = claims.with_clearance(clearance)        // the issuer path
```

The same threading happens in `exchange_ucan_grant_refresh` (B1/#673 — refresh
must not be more permissive than mint).

## Fail-closed contract (MUST hold)

| Condition | Result |
|-----------|--------|
| No compiled policy installed | `DenyUnlabeledResolver` → `None` → deny |
| Compiled policy installed, DID not enrolled | `EnrollmentSubjectContextResolver` → `None` → deny |
| DID enrolled, but token not yet minted with clearance | `Claims.clearance = None` → `ClaimsSubjectContextResolver` → `None` → deny |
| DID enrolled, token minted with clearance (this PR) | `Claims.clearance = Some(label)` → resolver returns `SecurityContext` |
| Classical key, PqHybrid clearance in table | Assurance clamped to `Classical` by `from_clearance` (Decision D, #548) |

**No row in this table produces a permissive default.** The only path to a real
`SecurityContext` is: compiled policy installed → DID enrolled → token minted
with `Claims.clearance` (this PR) → Claims verified downstream.

## How #1268 (RPC PEP) consumes

The RPC PEP receives an `EnvelopeContext` from the verified signed envelope. It
constructs a `ClaimsSubjectContextResolver` from:

```rust
let key_material = envelope_ctx.verified_key_material(); // Classical or PqHybrid
let resolver = ClaimsSubjectContextResolver::new(
    &claims.sub,
    claims,
    key_material,
);
let subject_ctx = resolver.resolve(&claims.sub); // Option<SecurityContext>
```

Then passes `subject_ctx` to the MAC PDP (`LatticeTeEvaluator`) along with the
object label and action. `None` → deny (the floor).

## How #1269 (9P activate) consumes

At 9P `Tattach`, the server presents a verified credential once. The server
constructs the same `ClaimsSubjectContextResolver` from the verified JWT,
caches the `SubjectCtx` connection-scoped (the `mac::avc` amortization model),
and enforces via the AVC on every subsequent operation.

Per the MAC interface policy (epic #547, ratified): the clearance is **derived**
from verified credentials, never passed as a plaintext parameter. Labels in wire
schemas are hints (D1), never authoritative PDP inputs.

## Upgrade path (#718)

Raising a specific enrolled actor above `Classical` assurance requires
enrollment-key registration (#718) — binding a PQ verification method to the
enrolled DID so the resolver can return `VerifiedKeyMaterial::PqHybrid`. That is
NOT this PR; this PR's resolver floors at `Classical` unconditionally (Decision D).
