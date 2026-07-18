# ADR 0003 — Distinct typed roles: owner, controller, payer, issuer, ledger operator, registrar

**Status:** ratified (2026-07-17, epic #1064). **Normative source:** `draft-hyprstream-resource-attestation-00` §authority.

## Context

Conflating owner, controller, and payer (or issuer, ledger operator, and registrar) produces a confused-deputy attack surface: a payer could claim title, or a MAC authority could "prove" payment, or the registrar could invent title while claiming to only join attestations.

## Decision

Owner, controller, payer, issuer, ledger operator, and registrar are **distinct typed roles**. Payment does not confer title; MAC authorization does not prove payment. The registrar joins attestations; it does not independently invent title. The resource store and proof plane hold bytes and manifests; they do not confer title.

## Consequences

- Each role maps to a typed boundary in `resource-authority-boundaries.md`.
- Anonymous authorization never fabricates a DID; an anonymous controller/payer is a typed principal, not a fabricated identity (RA-REQ-018/019).
- The assurance floor is the minimum across all composed assurances (RA-REQ-016/017).

## Alternatives considered

- **Single "principal" role:** rejected — removes the boundary that prevents payment→title and MAC→payment confusion.
- **Registrar as title authority:** rejected — the registrar is a serialized state machine, not an authority.
