# PEP-698 Status — Subject Clearance Provenance

**Branch:** `feat/mac-clearance-provenance`
**PR:** (pending push)
**Issues:** [#698](https://github.com/hyprstream/hyprstream/issues/698) · [#1267](https://github.com/hyprstream/hyprstream/issues/1267)
**Reviewer:** kimi-k3 (FINAL — do NOT self-merge)

## What landed

### The issuer path (the missing wire)

**Before:** The UCAN grant mint path resolved a DID→clearance via
`EnrollmentSubjectContextResolver` for the S6 gate evaluation, then threw it
away. The minted JWT carried no `clearance` claim, so every downstream consumer
(PEP, RPC handlers, 9P translator) saw an unlabeled subject → MAC deny.

**After:** `mint_grant_token` now accepts
`subject_clearance: Option<SecurityLabel>` and stamps it on Claims via
`claims.with_clearance(clearance)` when `Some`. Both the initial mint
(`exchange_ucan_grant`) and the refresh (`exchange_ucan_grant_refresh`) thread
the enrollment-resolved clearance into the mint, so the token carries the
authority-asserted label under its hybrid signature.

### What did NOT change (intentionally)

- **DenyUnlabeledResolver** stays as the fail-closed fallback when no compiled
  policy is installed. It is NOT removed.
- **Decision D** (#698): assurance is clamped to `Classical` unconditionally
  in `EnrollmentSubjectContextResolver`. Raising above Classical is #718.
- **`exchange_enrollment_resolver()`** wiring is unchanged — it already returns
  `EnrollmentSubjectContextResolver` when a policy is installed.
- **Token-exchange path** (`exchange_token_exchange`): goes through PolicyService
  RPC (`issue_token`), which mints remotely. Clearance stamping there requires a
  schema change and is out of scope for this PR.

## Fail-closed contract (verified by tests)

| Condition | Result | Test |
|-----------|--------|------|
| No compiled policy | `DenyUnlabeledResolver` → None → deny | `baseline_boot_flips_seam_on_but_still_denies_every_did` |
| Policy installed, DID enrolled, token minted (this PR) | `Claims.clearance = Some` → PEP resolves | `issuer_path_stamps_clearance_readable_by_pep_resolver` |
| Policy installed, DID NOT enrolled | resolver → None → no clearance stamped → PEP denies | `issuer_path_unenrolled_did_produces_no_clearance_pep_denies` |
| Delegated grant | met clearance stamped (not delegator's higher) | `issuer_path_stamps_met_clearance_not_delegator_higher` |
| Classical key, PqHybrid table entry | assurance clamped to Classical | `classical_key_clamps_pqhybrid_clearance_down` |

## How #1268 (RPC PEP) consumes

```rust
// In the RPC handler, after envelope verification:
let key_material = envelope_ctx.verified_key_material();
let resolver = ClaimsSubjectContextResolver::new(
    &claims.sub,
    claims,        // carries clearance stamped by this PR
    key_material,
);
let subject_ctx = resolver.resolve(&claims.sub);
// Pass to LatticeTeEvaluator: None → deny (floor)
```

## How #1269 (9P activate) consumes

At `Tattach`, construct `ClaimsSubjectContextResolver` from the verified JWT,
cache the `SubjectCtx` connection-scoped, enforce via AVC on every op.

## Files changed

- `crates/hyprstream/src/services/oauth/token_exchange.rs` — issuer path +
  tests
- `.fleet-coord/clearance-provenance.md` — resolver interface contract
- `.fleet-coord/pep-698-status.md` — this file

## Validation

```
cargo nextest run --release --profile ci --workspace
```

**Result: 3604 tests passed, 0 failed, 10 skipped.** ✅
