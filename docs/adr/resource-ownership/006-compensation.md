# ADR RO-006: Compensation preserves history

**Status:** Accepted for review (#1067)

## Decision

A reservation that cannot complete is voided. A posted charge that cannot finalize is corrected by a new compensating ledger transfer. Original MAC decisions, ledger postings, manifests, receipts, and checkpoints are immutable facts. Timeouts with unknown external outcome enter quarantine rather than guessing success or failure.

## Consequences

Reconciliation is auditable and idempotent. Delete, correction, revocation, issuer default, and title transfer create successor/tombstone/compensation facts rather than editing history. Compensation restores economics but does not confer or erase title.
