# ADR 0006 — Distinct privacy profiles; anonymous authorization never fabricates a DID

**Status:** ratified (2026-07-17, epic #1064). **Normative source:** `draft-hyprstream-resource-attestation-00` §profiles, §receipt.

## Context

A single "anonymous" profile that sometimes emits a stable holder DID would link redemptions across resources and origins. Merging identified and anonymous profile kinds would let an anonymous principal be treated as an identified one (or vice versa). Public receipts carrying raw tokens, holder DIDs, or linkable entitlement IDs would undo the privacy properties.

## Decision

Seven profiles remain distinct: `identified-owner`, `pairwise-owner`, `committed-owner`, `identified-controller`, `anonymous-capability-controller`, `identified-payer`, `anonymous-entitlement-payer`. A profile that accepts an anonymous controller or payer must not emit a stable holder DID, ATProto handle, or linkable entitlement identifier into an origin audit record, public receipt, or trust store. Public proof records use commitments; detailed receipts are selectively disclosed or encrypted to authorized parties.

## Consequences

- Vectors `anonymous-fabricates-did` and `public-receipt-reveals-holder` reject leakage (RA-REQ-019/020/022).
- The origin may learn an authorized operation without a stable holder DID.
- Issuance-side accounting may remain identified; redemption-side records commitments/nullifiers and aggregate issuer liabilities.
- Unknown or crossed profile kinds reject (RA-REQ-025).

## Alternatives considered

- **A single anonymous profile that may carry a DID:** rejected — links redemptions.
- **Public detailed receipts:** rejected — leaks holder and entitlement linkage.
