# ADR 0005 — Reconciliation and compensation for reservations

**Status:** ratified (2026-07-17, epic #1064). **Normative source:** `draft-hyprstream-resource-attestation-00` §transition, §state-machine.

## Context

A reservation may hold resources (entitlement, quota, bytes) and then fail: the MAC attestation may not verify, the blob may fail to seal, a crash may interrupt, or a concurrent transition may win. A reservation that neither posts, voids, nor is reconciled leaks entitlement and breaks the "every reservation eventually settles" invariant.

## Decision

Every reservation eventually **posts, voids, or enters explicit quarantine or manual review**. The registrar (#1069) drives a durable saga with fencing-guarded compare-and-swap, an outbox, and a reconciler. Quarantined records move to `Reconciled` or `ManualReview`. Compensation voids a reservation and releases its held resources.

## Consequences

- A crash or restart must not repeat a finalized transition or mint a second successor (RA-REQ-012).
- At most one finalized successor per predecessor/version (RA-REQ-008/009/010).
- The failure lifecycle `Voiding→Voided` and `Quarantined→Reconciled|ManualReview` is normative.

## Alternatives considered

- **Best-effort reservations with no quarantine:** rejected — leaks entitlement and violates the settlement invariant.
- **Synchronous lock-based finalization:** rejected — the proof plane is not a synchronous filesystem lock service.
