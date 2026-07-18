# ADR 0001 — Dual attestation for resource ownership

**Status:** ratified (2026-07-17, epic #1064). **Normative source:** `draft-hyprstream-resource-attestation-00` §dual-attestation.

## Context

A resource manifest confers title and control. A single source of authority (MAC alone, or payment alone) is unsafe: payment is not title, and MAC authorization is not proof of payment. Mixing the two in one attestation lets one authority's failure forge the other's claim.

## Decision

Require two typed attestations over the **identical canonical `ResourceIntent` digest**: a MAC title/control attestation (#1068) and a ledger economic attestation (#1072). The registrar (#1069) joins them and finalizes one canonical successor. Neither alone is a final manifest; provisional material from either alone is unavailable through the normal namespace.

## Consequences

- A registrar rejects a manifest whose two attestations are not over byte-identical canonical digests (RA-REQ-005).
- A MAC-only or ledger-only attestation is never a final manifest (RA-REQ-006).
- The canonical `ResourceIntent` encoding is the single joining point; it must be deterministic (DAG-CBOR, pending #1065).
- Anonymous profiles remain possible: the origin learns an authorized operation without a stable holder DID.

## Alternatives considered

- **Single combined attestation:** rejected — collapses the typed-role boundary and lets one authority's compromise forge the other's claim.
- **Either-attestation finalization:** rejected — payment would confer title, or MAC would prove payment.
