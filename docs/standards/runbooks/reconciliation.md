# Runbook: ledger and registrar reconciliation

**Status:** pre-construction (2026-07-17). Operational target for #1069/#1072.

## Goal

Keep the registrar's finalized state and the ledger's posted entitlements consistent. The ledger attests economic entitlement; the registrar joins it with MAC title. They must agree on what was reserved, consumed, released, and compensated.

## Reconciliation loop

1. **Enumerate.** For each registrar generation, enumerate `LedgerReserved` operations older than the settlement window that are not yet `LedgerPosted`, `Voided`, or `Quarantined`.
2. **Compare.** For each, compare the registrar's recorded ledger attestation to the ledger's posted state for the operation ID.
3. **Settle.**
   - Posted in the ledger → advance registrar to `LedgerPosted`/`Finalized` if both attestations hold.
   - Voided in the ledger → advance registrar to `Voiding`/`Voided` with compensation.
   - Indeterminate → `Quarantined` (see quarantine runbook).
4. **Drain outbox.** Emit durable effects (namespace updates, proof-plane manifests/receipts) via the outbox exactly once per operation ID.
5. **Checkpoint.** Anchor the reconciled state at a tamper-evident checkpoint signed with the hybrid-PQC composite.

## Invariants

- Reconciliation is idempotent by operation ID.
- A reconciled transition never raises assurance or fabricates identity.
- No ledger or proof-plane I/O is placed on ordinary reads or fine-grained 9P writes.
