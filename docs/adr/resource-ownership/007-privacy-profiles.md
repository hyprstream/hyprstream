# ADR RO-007: Explicit privacy profiles

**Status:** Accepted for review (#1067)

## Decision

Identified, pairwise-DID, committed-owner, anonymous-controller, and anonymous-payer are separate profiles with typed fields and separate negative controls. Public artifacts use commitments and minimum routing/verifiability data; detailed receipts are encrypted or selectively disclosed. Anonymous and committed commitment constructions remain opaque and production-gated on #1059–#1062 **and #1065**, which owns their canonical construction/schema.

## Consequences

No fallback converts anonymous to identified or unauthenticated. Origin storage/audit excludes raw tokens, holder/root DIDs, handles, stable client keys, and linkable entitlement IDs. Pairwise identifiers reduce trivial joins but do not justify a collusion-resistance claim; timing, amount, issuer, and operation leakage remain documented.
