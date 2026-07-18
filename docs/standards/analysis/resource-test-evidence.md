# Resource test-evidence scaffold

**Status:** pre-construction (2026-07-17). This document records the property, crash, concurrency, privacy, and performance test plan and the evidence the sibling implementations must produce. The deterministic boundary checker and vectors land now; the end-to-end suites are gated on #1065–#1072.

## Layered test model

| Layer | What it proves | Status | Owner |
|---|---|---|---|
| Boundary checker + deterministic vectors | structural rejection of every prohibited transition | **landed in this slice** (#1070) | #1070 |
| Property tests | invariants hold over generated manifests | planned (property definitions below) | #1065/#1069 |
| Crash tests | crash/restart cannot double-finalize or repeat authorization | planned | #1069 |
| Concurrency tests | two concurrent transitions cannot both finalize | planned | #1069 |
| Privacy tests | no forbidden holder/entitlement linkage in public artifacts | planned (structural subset landed) | #1072/#928 |
| Performance tests | read hot-path is MAC-only and ledger/proof-free | planned | #1069/#1071 |
| End-to-end | one identified and one privacy-preserving resource flow | planned | #1065–#1072 |

## Property definitions (to be implemented)

- `no_final_without_both_attestations`: for all generated manifests, finalization requires both attestations over byte-identical canonical digests.
- `single_successor`: for all (resource, predecessor, version) pairs, at most one finalized successor exists.
- `idempotent_operation`: for all operation IDs, replay does not produce a second effect.
- `assurance_minimum`: effective assurance equals the minimum of composed assurances.
- `provisional_invisible`: provisional material is unreachable through the normal namespace.
- `anonymous_no_did`: no anonymous profile emits a stable holder DID into audit/receipt/trust.

## Vector category coverage (landed)

`vectors/resource-intent-canonicalization-v1.json` `mutation_categories` covers: canonicalization, mutation, replay, crossing, downgrade, privacy-leakage, crash-recovery, concurrency-fencing, provisional-visibility, state-and-profile. This satisfies the vector-coverage acceptance criterion at the structural level.

## Acceptance criteria status

- [x] Vectors cover canonicalization, mutation, replay, crossing, downgrade, privacy leakage, crash recovery, and concurrency fencing (structural level).
- [x] Every normative MUST/MUST NOT maps to an automated test or is explicitly specification-only with rationale (`resource-attestation-obligations.json`).
- [ ] Independent security, privacy, cryptographic, protocol, and standards reviews recorded; blocking findings resolved (requires the review process).
- [ ] At least one identified and one privacy-preserving end-to-end resource flow has reproducible implementation evidence (requires #1065–#1072).
- [ ] CAS/blob and 9P interoperability reports demonstrate the layers exercised (plans landed; execution gated on #1066/#1071).
- [x] Each draft disposition recorded (`resource-standards-gap.md`).
- [x] Generated XML/text/HTML is reproducible in CI and checked for stale output (CI job + `tools/check_resource_attestation.py`).

## CI integration

The `Standards artifacts` workflow runs `tools/check_resource_attestation.py`, which validates the registries, the deterministic vector (baseline canonical digest, both-attestations-over-identical-digest, production refusal, and every mutation's expected rejection), the obligation manifest's bidirectional coverage of RFCXML normative statements, and the stale-output check for the new draft's generated text and HTML.

## Reports (to be appended as the suites land)

Subsequent slices append concrete crash/concurrency/privacy/performance reports here, or under sibling issues, once the implementations exist. This slice records the plan and the structural evidence; it makes no completion claim for the end-to-end suites.
