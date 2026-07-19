# Runbook: quarantine and manual review

**Status:** pre-construction (2026-07-17). Operational target for #1069/#1072.

## When to enter quarantine

A reservation enters `Quarantined` when it cannot immediately post or void and cannot be safely retried: a ledger attestation is ambiguous, a MAC attestation is policy-ambiguous, a content seal is contested, or a concurrent-transition winner is undecided after the fencing grace window.

Every reservation eventually posts, voids, or enters quarantine or manual review. Nothing stays in an indeterminate provisional state indefinitely.

## Procedure

1. **Freeze the operation.** Mark the operation ID `Quarantined`; provisional material remains unavailable through the normal namespace.
2. **Hold entitlement.** The reserved entitlement remains held until reconciliation; it is neither spent nor released.
3. **Triage.** An operator (or automated reconciler) inspects the tamper-evident audit entry for the operation. Decisions are recorded, not silent.
4. **Resolve.** Move to `Reconciled` (post or void with compensation) or escalate to `ManualReview`.
5. **ManualReview.** A human decision is recorded in the audit journal; the operation then moves to `Reconciled` per that decision.

## Safety

Quarantine never confers title, never exposes provisional bytes as finalized, and never mints a DID for an anonymous principal. The registrar state gates visibility throughout.
