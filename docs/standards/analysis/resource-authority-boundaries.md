# Resource authority boundaries

**Status:** pre-construction (2026-07-17). This document is the conformance and documentation closure view of the authority model owned by #1067. It records the typed boundaries the registrar enforces; it does not select a signature suite, canonical encoding, or anonymous construction. The normative source of truth for the MUSTs cited here is `docs/standards/rfc/draft-hyprstream-resource-attestation-00.xml`.

## Scope split with #1067

#1067 owns the design analysis: the authority model, the full threat/privacy analysis, and the standards disposition decision. #1070 owns this conformance closure view: how each typed boundary maps to a normative requirement, a test, or an explicit specification-only obligation with a blocking owner. Where the two disagree, #1067 is the design authority and this document's MUSTs are the conformance contract that the implementation must satisfy.

## Typed authority boundaries

| Boundary | Decides | Must NOT decide | Owning issue |
|---|---|---|---|
| MAC authority | whether a principal may create, control, transfer, mutate, or delete a resource under current policy and the content-bound label | payment, economic entitlement, ledger state | #1068 |
| Ledger authority | whether the identified or anonymous payer has a valid entitlement; what was reserved, consumed, released, or compensated | title, control, MAC clearance | #1072 |
| Resource registrar | joins both attestations; selects the single canonical successor manifest | inventing title, fabricating identity, raising assurance | #1069 |
| Resource store / namespace | bytes and namespace projection | independently inventing title, exposing provisional material | #1066 / #1071 |
| Proof plane (PDS) | manifests, receipts, checkpoints, selective-disclosure material | synchronous filesystem locking, authoritative title | #1065 |

Payment does not confer title. MAC authorization does not prove payment. Owner, controller, payer, issuer, ledger operator, and registrar are distinct typed roles. A relay route, NodeId, URL, namespace, topic, content CID, or outer token is never title or authority to release.

## Non-negotiable invariants (conformance view)

Each invariant below is a MUST or MUST NOT in the RFCXML and is traced in `resource-attestation-obligations.json`:

1. No final manifest exists without both attestations over the identical canonical `ResourceIntent` digest.
2. At most one finalized successor per predecessor/version, finalized by compare-and-swap with a fencing token.
3. Stable resource identity, content CID, manifest CID, and operation ID are independent; no cyclic hash or signature construction.
4. Every request and external effect is idempotent by deterministic operation ID; every reservation eventually posts, voids, or enters quarantine.
5. Provisional material is unavailable through the normal namespace; visibility is a property of registrar state.
6. Ordinary reads remain MAC-only; ledger/proof-plane I/O is not on the read or fine-grained 9P-write hot path.
7. Identified, pairwise, committed-owner, identified/anonymous-capability controller, and identified/anonymous-entitlement payer profiles remain distinct; anonymous authorization never fabricates a DID.
8. Effective assurance is the minimum across entitlement, issuer token, binding, MAC decision, ledger attestation, and key release; a classical outer admission cannot raise an inner resource operation to post-quantum-hybrid.

## Construction status

The sibling implementation issues (#1065 canonical ResourceIntent, #1066 CAS integration, #1068 MAC title attestation, #1069 registrar saga, #1071 9P ownership, #1072 ledger redemption) are not complete. The boundary checker therefore terminates a structurally valid fixture with the `construction-incomplete` refusal. No production manifest is finalized until both reviewed attestations exist over the identical canonical digest and the crash, concurrency, privacy, and interoperability suites pass.
