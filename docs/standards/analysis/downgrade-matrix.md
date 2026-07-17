# Downgrade and substitution matrix

The matrix covers the profile boundary, not a claim that the listed mechanisms are already implemented. Each row becomes an implementation/vector obligation when its owning issue selects the construction.

| Attempt | Boundary that rejects it | Required fail-closed rule | Owner | Evidence status |
|---|---|---|---|---|
| Replace inner capability with outer classical token | origin authorization | Outer admission can only route/rate-limit ciphertext; it cannot mint `AnonymousCapability`, MAC clearance, or a key | #1062/#1063/#726 | specification-only until inner construction |
| Strip PQ issuance leg | issuer/origin verifier | `PqHybrid` requires every exact reviewed classical and PQ leg; missing, duplicate, reordered, unknown, or substituted legs reject | #1060 | blocked on construction selection |
| Treat hybrid signature wrapper as blind issuance | issuer/origin verifier | Ordinary visible signing is not anonymous issuance and cannot satisfy the issuance property | #1060 | specification-only |
| Use classical entitlement to raise assurance | capability derivation | Effective assurance is the minimum of verified legs; a classical attestation stays Classical | #1060/#1062 | specification-only |
| Substitute issuer/origin/controller key | accepted-current resolver | Purpose-separated key IDs and current accepted state must match the exact role | #1039/#1060 | prerequisite landed; profile wiring pending |
| Serve stale/revoked/forked state | client/issuer/origin | Reject before issuance, redemption, dispatch, CONNECT, or key release | #1039/#1051/#726 | profile/test pending |
| Substitute service/method/resource | inner binding verifier | Exact canonical transcript commitment must match requested action | #1061/#1063 | blocked on binding |
| Substitute MOQT track/group/epoch | inner binding/key release | Exact carrier, operation, resource, track/group, direction, epoch, and recipient must match | #1061/#726/#554/#555 | blocked/pending |
| Change carrier profile after resolution | client/origin | `owned-hybrid-transport` and `standard-public-relay` are explicit; negotiation failure cannot silently switch profiles | #1051/#557/#726 | pending |
| Replay token or binding proof | origin spend store | One use by default; record consume atomically before dispatch/key release | #1061/#1062 | blocked on construction |
| Reuse PoP or HyKEM recipient | client/origin | Fresh one-use keys only; erase after spend/cancel/failure; key mismatch rejects | #1061/#1051 | pending |
| URL/query or proprietary CONNECT credential | client/relay | Use standard MOQT authorization-token surfaces where the relay participates; no long-lived replayable URL/query credential | #726 | specification-only |
| Relay supplies authority from route/topic/NodeId | origin/PEP | Routing inputs are never authorization; require a current admitted result for release | #726 | production gap |
| Relay gets plaintext/key by cache path | encryption/key release | Relay receives ordinary encrypted Object bytes only; no epoch/traffic key release | #554/#555/#726 | interop plan ready; implementation pending |
| Reuse immutable Object identity for changed ciphertext | producer epoch lifecycle | Rekey creates epoch-bound group/namespace; cached Object identities remain immutable | #554/#555 | pending |
| Cross-protocol/domain transcript reuse | registry/verifier | Each protocol step uses an exact registry label and field set; unknown label/version rejects | #1059/#1060/#1061 | registry scaffold ready |
| Missing/unknown content label interpreted as public | MAC PEP | Unknown/untrusted/missing labels deny; `Public` is a lattice floor, not bearer access | #699/#1062 | pending |
| Audit/log records raw anonymous material | audit boundary | Record only minimum opaque decision evidence; redaction failure denies restricted release | #1062/#1051 | pending |

## Test classification

`specification-only` means the skeleton makes a normative/profile decision but there is no selected wire construction to test. It is not an implementation claim. The checked manifest (`../registry/obligations.json`) assigns an owner and blocker to each such obligation. Local negative controls will mutate repository-owned fixtures only: remove, replace, replay, reorder, expire, or cross-bind a required field and assert rejection before handler/key release.
