# Runbook: resource registrar crash recovery

**Status:** pre-construction (2026-07-17). Operational target for #1069. The boundary checker enforces the structural invariants today; this runbook records the operator procedure for the durable registrar once it lands.

## Invariant

A crash or restart must not repeat a finalized transition or mint a second successor. A crash after durable spend/commit may burn the operation but cannot repeat authorization. Every transition is fencing-guarded compare-and-swap; every operation ID is idempotent.

## Recovery procedure

1. **Replay the outbox.** On restart, the registrar replays its durable outbox in fencing-generation order. Each entry is idempotent by operation ID.
2. **Re-check fencing.** A transition whose fencing token is stale or from another registrar generation is rejected; it does not advance a version.
3. **Detect partial finalization.** A transition recorded as `spend-committed` but not `finalized` is re-evaluated against the **complete finalization precondition set**, not just the attestations: (a) both attestations hold over the identical canonical `ResourceIntent` digest; (b) the blob is durably `Materialized` and the sealed bytes match the manifest `content_cid`; (c) the ledger effect is durably recorded (`LedgerPosted`); (d) the expected predecessor and version still match the accepted-current registrar state; and (e) the fencing token is from the current registrar generation and not already consumed. Only if all preconditions hold, finalize once; if any fails, void or quarantine — never finalize a resource whose blob or ledger effect was not durably completed.
4. **No double-finalize.** If a successor for the (resource, predecessor, version) triple already exists, the replay is a no-op for that operation ID.
5. **Record.** Every recovery decision is written to the tamper-evident audit journal (`mac::audit::WalAuditStore`) with the operation ID, fencing generation, and decision.

## Negative controls (structural)

Vector `crash-repeats-finalized` rejects a resubmitted finalized operation; vector `missing-fencing-token` rejects finalization without a fencing token; vector `concurrent-successors` rejects two successors for one predecessor.
