# Privacy and collusion analysis

**Status:** pre-construction (2026-07-17). Conformance closure view; the design privacy analysis is owned by #1067, with pseudonymity and selective-disclosure constraints from #928. This document records the data-flow limits and collusion model the boundary checker and vectors enforce.

## Principal kinds and what each party may learn

| Principal kind | Origin may learn | Issuer/ledger may learn | Public receipt may reveal |
|---|---|---|---|
| identified | exact DID | exact DID and entitlement | only what the profile permits; selective disclosure or encryption |
| pairwise | a pairwise DID scoped to the origin | identified issuance-side accounting only | commitment, never the pairwise DID |
| committed-owner | an authorized operation, not a stable holder DID | liability aggregate | owner commitment only |
| anonymous-capability controller | an authorized control operation | no stable holder DID | no holder linkage |
| anonymous-entitlement payer | an authorized payment operation | commitment + nullifier; aggregate liability | entitlement commitment only |

The origin may learn an exact authorized operation without learning a stable holder DID. Issuance-side accounting may remain identified while redemption-side accounting records commitments, nullifiers, and aggregate issuer liabilities.

## Forbidden data flows (enforced by vectors)

A profile that accepts an anonymous controller or anonymous payer must not emit a stable holder DID, ATProto handle, or linkable entitlement identifier into an origin audit record, a public receipt, or a trust store. The vectors `anonymous-fabricates-did` and `public-receipt-reveals-holder` exercise these rejections.

Raw anonymous tokens, holder DIDs, stable client keys, and linkable entitlement identifiers do not enter origin audit, public receipts, or trust stores.

## Collusion model

- **Issuer + origin:** may attempt to link a redemption to an issuance. Defense: redemption records commitments/nullifiers; issuance-side and redemption-side accounting are separable.
- **Origin + ledger:** may attempt to correlate an anonymous payer across resources. Defense: per-resource entitlement commitments and aggregate-only issuer liabilities.
- **Ledger + storage + cross-cell:** may attempt cross-resource correlation by content or namespace. Defense: content-bound labels (#699) and compartment separation; no cross-cell linkage is claimed beyond MAC and compartment boundaries.
- **Global observer / relay:** may observe timing, size, cache behavior. No traffic-analysis resistance is claimed (mirrors the #1058/#1059 stock-relay boundary).

## Selective disclosure and encrypted receipts

Public proof records use commitments; detailed receipts are selectively disclosed or encrypted to authorized parties. An encrypted receipt is addressed only to a party authorized by the active profile and is bound to the same canonical `ResourceIntent` digest as the manifest it accompanies. The encrypted-receipt profile is specified in `resource-encrypted-receipt.md`; its implementation is pending #1072/#928.

## Out of scope

No issuer-origin-ledger unlinkability proof, no traffic-analysis resistance, and no cross-cell correlation resistance beyond MAC/compartment boundaries. These remain residual risks gated on the sibling implementations and independent privacy review.
