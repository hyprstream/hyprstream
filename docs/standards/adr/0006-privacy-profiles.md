# ADR 0006 — Distinct privacy profiles; anonymous authorization never fabricates a DID

**Status:** ratified (2026-07-17, epic #1064). **Normative source:** `draft-hyprstream-resource-attestation-00` §profiles, §receipt.

## Context

A single "anonymous" profile that sometimes emits a stable holder DID would link redemptions across resources and origins. Merging identified and anonymous profile kinds would let an anonymous principal be treated as an identified one (or vice versa). Public receipts carrying raw tokens, holder DIDs, or linkable entitlement IDs would undo the privacy properties.

## Decision

Seven profile values remain distinct: `identified-owner`, `pairwise-owner`, `committed-owner`, `identified-controller`, `anonymous-capability-controller`, `identified-payer`, `anonymous-entitlement-payer`. These are **per-role privacy kinds, not complete intent tuples**: every ResourceIntent carries three typed references (`owner_ref`, `controller_ref`, `payer_ref`), each constrained to its own per-role union, and the declared profile MUST match the kind of the typed reference it governs. The three typed references are the authoritative composite; a crossed profile/reference combination or a reference kind outside its per-role union rejects (vectors `profile-ref-kind-mismatch`, `crossed-ref-kind`). A profile that accepts an anonymous controller or payer must not emit a stable holder DID, ATProto handle, or linkable entitlement identifier into an origin audit record, public receipt, or trust store. Public proof records use commitments; detailed receipts are selectively disclosed or encrypted to authorized parties.

## Consequences

- Vectors `anonymous-fabricates-did` and `public-receipt-reveals-holder` reject leakage (RA-REQ-019/020/022).
- The origin may learn an authorized operation without a stable holder DID.
- Issuance-side accounting may remain identified; redemption-side records commitments/nullifiers and aggregate issuer liabilities.
- Unknown or crossed profile kinds reject (RA-REQ-025).

## Alternatives considered

- **A single anonymous profile that may carry a DID:** rejected — links redemptions.
- **Public detailed receipts:** rejected — leaks holder and entitlement linkage.
