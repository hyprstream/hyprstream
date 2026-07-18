# Resource ownership privacy analysis

## Goals

- Never put legal identity in protocol records.
- Prevent a single origin, relay, storage operator, public observer, or unrelated cell from learning a stable holder identity solely from resource authorization/accounting.
- Keep owner, controller, and payer separable.
- Publish commitments/checkpoints rather than detailed private receipts.
- State unavoidable leakage and avoid claiming collusion resistance v1 cannot provide.

## Profile data flow and leakage

| Profile | Private inputs that must not leave verifier boundary | Public/proof projection | Unavoidable leakage |
|---|---|---|---|
| Identified | private keys, full capability chain where disclosure unnecessary | role DIDs, intent/receipt commitment | DIDs, operation, unit/amount/timing to counterparties |
| Pairwise DID | root DID and pairwise mapping | pairwise DID/commitments | per-cell activity; issuer can join its mappings; timing/volume cross-cell |
| Committed owner | opening secret and hidden owner identity | owner commitment | long-lived resource commitment can itself be linkable across versions |
| Anonymous controller | raw token, holder DID/handle, issuance transcript, stable client key | capability commitment/nullifier, assurance | origin sees exact authorized operation, redemption timing, carrier and object |
| Anonymous payer | raw token, holder/allocation ID, issuer linkage map | entitlement commitment/nullifier, aggregate liability/checkpoint | issuer/unit, amount bucket/ceiling, timing, origin ledger context |

Committed-owner concealment is not equivalent to anonymous control. A stable owner commitment may intentionally link a resource's versions while hiding the identity; its reveal/transfer construction is owned by #1065 and its cryptographic profile. This issue selects no commitment primitive.

## Observer analysis

The frozen interface separates the private finalized record from the public namespace projection by construction. The registrar's finalized output and the stored attestations (which include exact amount, issuer, unit, payer commitment, epochs, and capability binding) are authorized-party evidence; their CIDs resolve only inside access-gated evidence domains. Publication consumes only `PublicResourceProjection` — resource ID, version, manifest CID, optional content CID, and an opaque public evidence commitment. A namespace, PDS, or storage observer therefore cannot obtain exact amount/issuer/unit or attestation detail from anything the publisher emits; that detail stays scoped to origin, issuer, and ledger.

- **Origin:** must know exact operation, resource, policy/label, and enough entitlement ceiling to enforce it. Anonymous profiles remove stable holder identity, not transaction observability.
- **Issuer:** knows issuance and underwritten/prepaid relationship. Blind issuance aims to hide redemption; timing/denomination can still correlate.
- **Ledger:** knows issuer unit, amount, cell, transfer state, and either payer pseudonym or commitment/nullifier. It must not receive owner identity unless the selected profile requires/coincides with it.
- **Registrar:** knows the full role-separated intent needed for finalization, but stores opaque commitments rather than raw tokens or linkage maps.
- **Storage:** learns byte length, timing, dedup behavior, and access pattern unless end-to-end encrypted/padded. It does not receive payer evidence.
- **PDS/public observer:** sees public commitments, CIDs, checkpoint cadence, and publication sizes. Detailed receipts are encrypted/selective; anchors contain checkpoint digests only.
- **Relays:** see routing metadata, ciphertext size/timing, and endpoints; no plaintext, capability clearance, payer linkage, or key material.
- **Cross-cell counterparty:** sees tranche/settlement bilateral metadata and valid receipt evidence needed for dispute; not unrelated inventory.

## Correlation controls

Pairwise per-cell identifiers; allocation/entitlement-keyed rather than subject-keyed nullifiers; coarse/fixed denominations; reservation ceilings with private actual detail where possible; batching and delayed/fixed-cadence publication; aggregate issuer liabilities; encrypted detailed receipts; group-keyed disclosure; Tor/Snowflake-compatible transport; domain-scoped dedup responses; no activity-triggered public anchors.

These are mitigations, not proofs. V1 explicitly does not claim unlinkability against issuer-origin-ledger collusion or a global passive adversary. Pairwise IDs make joining nontrivial but timing and amount can dominate.

## Retention and logging

Origin audit/trust/observability stores exclude raw anonymous token bytes, holder/root DID, ATProto handle, stable PoP/client key, issuance transcript, holder-stable capability-chain roots, and directly linkable allocation identifier. In anonymous profiles, any capability value that leaves the verifier boundary — in an attestation or any public projection — is a blinded, operation-scoped commitment, never the holder's capability-chain root; transfer IDs are per-operation derivatives of the operation ID. Logs use operation-scoped opaque IDs and coarse error classes. Private registrar recovery data and detailed receipts have explicit retention/access policy; public commitments may be permanent and therefore should reveal the minimum.

## KYC boundary

No schema, lexicon, extension, receipt, grant, bond, or discovery record has a legal-identity field. Optional KYC belongs only to an external fiat on/off-ramp and is the converter's regulatory responsibility. A fully KYC-free deployment remains valid.

## Required privacy evidence

Per-profile positive and fail-closed negative tests; schema scans for forbidden fields; log capture tests; amount/timing correlation measurement; issuer-origin-ledger collusion experiments; proof that anonymous retry uses the same opaque outcome without storing raw token; dedup possession-oracle tests; public-record inspection, including a dedicated negative control proving the public projection resolves to no exact amount/issuer/unit, no attestation CID, and no holder-linkable capability value (#1070); independent privacy review before enablement.
