# Residual-risk analysis

**Status:** pre-construction (2026-07-17). Conformance closure view. This document records the risks that remain after the boundary checker, vectors, and sibling implementations land, and the mitigations required before production enablement.

## Residual risks

| Risk | Why it remains | Mitigation / gate |
|---|---|---|
| Cryptographic construction unselected | No reviewed MAC title suite or ledger redemption construction is selected | #1068/#1072 selection + independent cryptographic review |
| Canonical production encoding pending | DAG-CBOR production encoding of `ResourceIntent` is not implemented | #1065 |
| Registrar not durable | The saga, fencing, outbox, and reconciler are not implemented | #1069 + crash/concurrency suites |
| Namespace visibility gating absent | Registrar-state-driven provisional visibility is not wired | #1069/#1071 |
| Read hot-path not MAC-enforced | The MAC PEP is not wired to production read paths | #547/#568 |
| Key history single-classical | Hybrid-PQC key-rotation history not implemented | #1068/#1072 |
| Selective disclosure / encrypted receipts | Proof-plane path not implemented | #1072/#928 |
| Collusion beyond data minimization | No unlinkability proof against issuer-origin-ledger collusion | independent privacy review + accepted-current design |
| Traffic analysis | No resistance claimed against a global observer | out of scope; record only |
| Compromised TCB/registrar host | A compromised host can forge state | out of scope for this profile; recorded |

## Fail-closed posture

Every unimplemented path fails closed: a structurally valid fixture terminates with `construction-incomplete`; a missing, crossed, downgraded, replayed, or concurrent transition rejects. No production manifest is finalized until both reviewed attestations exist over the identical canonical digest and the crash, concurrency, privacy, and interoperability suites pass.

## Production gates (acceptance criteria)

Before production enablement, the following must close:

- [ ] Every normative MUST/MUST NOT maps to an automated test or is explicitly specification-only with rationale.
- [ ] Vectors cover canonicalization, mutation, replay, crossing, downgrade, privacy leakage, crash recovery, and concurrency fencing.
- [ ] Independent security, privacy, cryptographic, protocol, and standards reviews recorded; blocking findings resolved.
- [ ] At least one identified and one privacy-preserving end-to-end resource flow has reproducible implementation evidence.
- [ ] CAS/blob and 9P interoperability reports demonstrate the layers exercised.
- [ ] Each draft disposition recorded.

## Status of this slice (2026-07-17)

This slice lands the pre-construction scaffold: RFCXML source, registries, CDDL, vectors, analysis, ADRs, runbooks, and the boundary checker. It records dispositions, traces every normative MUST to a test or spec-only obligation, and refuses production finalization. It does not claim endorsement, adoption, allocation, or interoperability not demonstrated by evidence.
