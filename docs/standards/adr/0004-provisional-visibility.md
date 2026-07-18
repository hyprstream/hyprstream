# ADR 0004 — Provisional visibility is a property of registrar state

**Status:** ratified (2026-07-17, epic #1064). **Normative source:** `draft-hyprstream-resource-attestation-00` §visibility.

## Context

If provisional material (bytes sealed before finalization, or either attestation alone) were reachable through the normal namespace, then a reader could observe or act on non-final state — leaking in-progress operations and letting unjoined material be used as if finalized.

## Decision

Visibility of provisional material is a property of **registrar state**, not merely of byte existence in the resource store or proof plane. Provisional material is unavailable through the normal namespace until the registrar reaches `Finalized`. Ordinary reads remain MAC-only; ledger and proof-plane I/O is not placed on ordinary reads or fine-grained 9P writes — bounded local leases settle at defined commit points only.

## Consequences

- A registrar-state check gates namespace visibility (#1069/#1071).
- Reads stay on the MAC-only hot path; ledger/proof I/O is amortized at commit points (the `mac::avc` model).
- Vector `cas-exposes-provisional` rejects a finalized claim over provisional bytes (RA-REQ-026).

## Alternatives considered

- **Byte-existence visibility:** rejected — provisional bytes would be reachable before finalization.
- **Per-read ledger checks:** rejected — violates the zero-ledger-I/O hot-path invariant.
