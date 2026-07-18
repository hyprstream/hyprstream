# Resource state machine and recovery specification

**Status:** pre-construction (2026-07-17). Conformance closure view; the durable registrar saga, fencing, quarantine, outbox, and reconciler are owned by #1069. This document is the normative state-machine and recovery contract the implementation must satisfy.

## Reference happy-path lifecycle

```
Requested -> MacAuthorized -> LedgerReserved -> Materialized -> LedgerPosted -> Finalized
```

- `Requested`: a deterministic operation ID is assigned; the canonical `ResourceIntent` is fixed.
- `MacAuthorized`: the MAC title/control attestation over the canonical digest is verified.
- `LedgerReserved`: the ledger economic attestation reserves the entitlement.
- `Materialized`: the content-addressed blob is sealed; content CID is bound.
- `LedgerPosted`: the reservation posts durably.
- `Finalized`: the registrar commits the single canonical successor by compare-and-swap with a fencing token.

No transition to `Finalized` is defined from a state that lacks both attestations. No transition to ordinary namespace visibility is defined from a provisional state. Unknown states, versions, profile kinds, or transition events reject.

## Reference failure and recovery lifecycle

```
Voiding -> Voided
Quarantined -> Reconciled | ManualReview
```

- A reservation eventually posts, voids, or enters explicit quarantine or manual review.
- A crash or restart must not repeat a finalized transition or mint a second successor (vector: `crash-repeats-finalized`).
- A crash after durable spend/commit may burn the operation but cannot repeat authorization or finalize twice.

## Fencing and concurrency

At most one finalized successor may advance a resource version. Every transition carries an expected predecessor and version, finalized by compare-and-swap guarded by a fencing token. Two concurrent transitions for one predecessor or version must not both finalize (vector: `concurrent-successors`). A stale, replayed, or crossed fencing token must not advance a resource version (vectors: `stale-predecessor-version`, `crossed-fencing-token`, `missing-fencing-token`).

## Idempotency

Every request and external effect is idempotent by deterministic operation ID. A registrar must not finalize two successors for one operation ID (vectors: `replayed-operation-id`, `replay-across-resource`).

## Operational runbooks

Recovery procedures are recorded in `docs/standards/runbooks/`:

- `crash-recovery.md` — crash/restart reconciliation, double-finalize prevention.
- `quarantine-manual-review.md` — quarantine entry, manual-review escalation.
- `reconciliation.md` — ledger/registrar reconciliation and outbox drainage.
- `key-rotation.md` — policy/MAC key rotation and key-history verification.

## Implementation status

The durable registrar, fencing, outbox, and reconciler are pending #1069. The state-machine contract above is the conformance target; the boundary checker enforces the transition rules that are structurally checkable today and refuses a structurally valid fixture with `construction-incomplete`.
