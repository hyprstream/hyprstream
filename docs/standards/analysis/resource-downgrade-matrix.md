# Assurance and downgrade matrix

**Status:** pre-construction (2026-07-17). Conformance closure view. This matrix records the outer/inner assurance separation and the negative controls the boundary checker enforces.

## Assurance composition rule

The effective assurance of a resource operation is the **minimum** of:

1. entitlement assurance,
2. issuer token assurance,
3. binding assurance,
4. MAC decision assurance,
5. ledger attestation assurance,
6. key-release assurance.

Effective assurance must not exceed the assurance of any attestation it composes. A classical outer admission result must not raise an inner resource operation to a post-quantum-hybrid assurance level.

## Negative-control matrix

| Control | Input | Required decision | Vector |
|---|---|---|---|
| classical-claims-pq-hybrid | a classical outer admission presented as PqHybrid authority | reject | `downgraded-assurance` |
| stale accepted state | accepted_state_epoch below current | reject | `stale-accepted-state` |
| policy rollback | policy_generation below current | reject | `policy-rollback` |
| mac-only attestation | manifest missing ledger attestation | reject (not final) | `mac-only-attestation` |
| ledger-only attestation | manifest missing MAC attestation | reject (not final) | `ledger-only-attestation` |
| crossed digest | MAC and ledger over different digests | reject | `crossed-resourceintent-digest` |
| crossed fencing token | fencing token from a stale registrar generation | reject | `crossed-fencing-token` |
| crossed profile | merged identified/anonymous profile kind | reject | `crossed-profile-kind` |
| unknown profile | profile not in the closed registry | reject | `unknown-profile-kind` |
| expired attestation | expiry before now | reject | `expired-attestation` |
| crossed expiry | manifest expiry not bound to attestation | reject | `crossed-expiry` |

## Outer/inner separation (carrier evidence boundary)

Stock-relay and stock-carrier evidence shared with the #1058/#1059 draft family remains owned by those drafts. This profile records only the resource-specific layers exercised: the registrar boundary (both attestations over the identical canonical digest, fencing-guarded compare-and-swap) and the content/CAS boundary (write-then-seal before title). No carrier-transport claim is made here.

A structurally valid fixture terminates with `construction-incomplete`; the downgrade controls above are structural rejections exercised by the boundary checker, not cryptographic proofs of PQ assurance.
