# ADR RO-002: Role and identity separation

**Status:** Accepted for review (#1067)

## Decision

Owner, controller, payer, issuer, ledger operator, registrar operator, custodian, namespace operator, and proof publisher are distinct roles. Owner supports identified, pairwise, and committed references; controller and payer additionally support typed anonymous commitments. Anonymous authority never fabricates a DID. Delegation preserves delegator and actor.

## Consequences

Schemas use separate typed fields rather than a generic `subject`. One principal may occupy multiple roles in a deployment, but signatures, audit, policy, and privacy projections preserve the roles. No legal-identity field exists; KYC remains outside the protocol at an optional fiat boundary.

**Vocabulary reconciliation (F6):** the epic #1064 text lists only `Identified`/`Anonymous*` role variants, while this crate and architecture §9 intentionally include `Pairwise(Did)` for owner, controller, and payer. The `hyprstream-resource` vocabulary is authoritative: #1065 freezes the wire schema from it, and the epic text should be amended to match rather than the pairwise variants dropped.
