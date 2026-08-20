# v16 standards profile — RPC request proof and credential (gate-2 inputs)

Status: **draft, not operator-approved.** This directory holds the checked-in
artifacts the MAC dispatch profile freeze requires *as inputs* to its operator
gate, not as outputs promised after it. Nothing here authorizes an
implementation, allocates an external codepoint, or claims IETF endorsement.

The controlling design lives outside this repository in the fleet design
record; the artifacts here restate its normative content in a reviewable,
machine-checkable form, and every value they add beyond it is marked
**PROPOSED**.

## Contents

| Artifact | What it fixes |
|---|---|
| [`hyprstream-proof-cwt.cddl`](hyprstream-proof-cwt.cddl) | Normative CDDL: proof CWT claims sets, response binding, signature plan, unattributed key set, private-use header parameters, exact `crit` sets, and the proof-v1 parser caps |
| [`private-label-registry.md`](private-label-registry.md) | The checked private-use label registry (claims −70001…−70004, headers −70100…−70103) with presence/null rules and a collision review against the IANA COSE/CWT registries |
| [`canonical-vectors.md`](canonical-vectors.md) | Human index of the positive and negative vectors, with the deny rule each negative exercises |
| [`vectors/`](vectors) | Machine-readable vectors: 5 positive, 30 negative, plus the seeded test keys |
| [`credential-profile.md`](credential-profile.md) | JWT/CWT credential claims table, credential/session identifier rules, one-shot versus reusable profiles, revocation semantics |
| [`tools/`](tools) | `gen_proof_vectors.py` (reproducible generator) and `check_proof_vectors.py` (verifier for the checked-in files) |

```sh
python3 docs/standards/v16/tools/gen_proof_vectors.py    # regenerate, byte-identical
python3 docs/standards/v16/tools/check_proof_vectors.py  # check the checked-in vectors
```

## Relationship to the sibling profiles

This directory does not overlap the Privacy Pass PQ-hybrid profile or the
resource-attestation profile in this tree: those profile the anonymous
authorization and resource-ownership surfaces, while this one profiles the RPC
request proof and the credential it binds to.

## Open items before the freeze

1. IANA vendor-tree registration of `application/vnd.hyprstream.proof+cwt` and
   `application/vnd.hyprstream.response-proof+cwt`.
2. Mechanical CDDL validation (no validator is available in the current
   environment; the schema was reviewed by hand).
3. Operator disposition of every value marked PROPOSED in these artifacts.
4. Re-verification of each recorded watch-item draft
   (`draft-ietf-jose-pq-composite-sigs`, `draft-ietf-cose-hpke`,
   `draft-ietf-cose-hpke-pq-pqt`) at freeze time. Nothing here depends on them.
