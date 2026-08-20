# v16 standards profile — RPC request proof and credential

Status: **FROZEN by the accepted Gate-2 vote** (v16 §19, operator disposition
2026-08-19). This directory holds the checked-in, mechanically-validated
artifacts of the MAC dispatch profile freeze. It is **not production-closed**:
the two `vnd.hyprstream` media types still require IANA vendor-tree registration
before production freeze, and the private-use ML-KEM-768 identifier −70200 stays
project-private. Nothing here authorizes an implementation beyond the frozen
wire contract, allocates an external codepoint, or claims IETF endorsement.

The controlling design lives outside this repository in the fleet design
record; the artifacts here restate its normative content in a reviewable,
machine-checkable form. Every value is frozen by the Gate-2 vote; where the vote
amended an originally-proposed value, the amendment (Gate-2 §19) is cited in the
artifact. No value here is PROPOSED any longer.

## Contents

| Artifact | What it fixes |
|---|---|
| [`hyprstream-proof-cwt.cddl`](hyprstream-proof-cwt.cddl) | Normative CDDL: proof CWT claims sets, response binding, signature plan, unattributed key set, private-use header parameters, exact `crit` sets, and the proof-v1 parser caps |
| [`private-label-registry.md`](private-label-registry.md) | The checked private-use label registry (claims −70001…−70004, headers −70100…−70103) with presence/null rules and a collision review against the IANA COSE/CWT registries |
| [`canonical-vectors.md`](canonical-vectors.md) | Human index of the positive and negative vectors, with the deny rule each negative exercises |
| [`vectors/`](vectors) | Machine-readable vectors: 8 positive, 53 negative, the seeded test keys, and the replay-namespace thumbprint vectors (`proof-v1-thumbprints.json`) |
| [`credential-profile.md`](credential-profile.md) | JWT/CWT credential claims table, credential/session identifier rules, Reusable-only credential use (OneShotTransaction deferred), revocation semantics |
| [`tools/`](tools) | `validate_profile.py` (the mechanical validation gate), `gen_proof_vectors.py` (reproducible generator), `check_proof_vectors.py` (vector verifier), and `requirements.txt` (pinned deps) |

```sh
# Install the pinned CDDL validator (prebuilt wheel; no Rust toolchain needed):
python3 -m pip install -r docs/standards/v16/tools/requirements.txt

python3 docs/standards/v16/tools/validate_profile.py    # the full gate (CDDL + registry + fixtures)
python3 docs/standards/v16/tools/gen_proof_vectors.py    # regenerate, byte-identical
python3 docs/standards/v16/tools/check_proof_vectors.py  # check the checked-in vectors
```

The gate uses a **real CDDL validator** — `pycddl` (the Rust `cddl` crate behind
a pinned prebuilt wheel) — to validate every fixture against the normative CDDL,
alongside the exact private values, the frozen caps, the closed response map,
the orthogonal enum axes, the recipient/encryption relation, the collision
review, and a byte-identical regeneration of the fixtures. Two pinned-version
limitations are documented in `validate_profile.py` and worked around by
stronger Python-side checks: byte-length `.size` **ranges** are re-checked in
Python, and the validator **panics** on the AKP/ML-DSA-65 `COSE_Key` in the
unattributed key set, so that key set is validated by the gate's B4
correspondence + embedded-key verification (exact 1:1 plan correspondence,
closed key shape, and per-signature verification against the embedded key)
rather than by the pycddl pass. The normative CDDL is unchanged.

## Relationship to the sibling profiles

This directory does not overlap the Privacy Pass PQ-hybrid profile or the
resource-attestation profile in this tree: those profile the anonymous
authorization and resource-ownership surfaces, while this one profiles the RPC
request proof and the credential it binds to.

## Open items before production close

The Gate-2 vote froze the wire/profile choices; the profile is **not** yet
production-closed. Remaining conditions:

1. **IANA vendor-tree registration** of `application/vnd.hyprstream.proof+cwt`
   and `application/vnd.hyprstream.response-proof+cwt`. The literal strings are
   already normative; registration is REQUIRED before production freeze.
2. **`−70200` stays project-private** until an incompatible profile revision
   adopts a registered COSE PQ/T KEM identifier
   (`draft-ietf-cose-hpke-pq-pqt`).
3. **Watch-item re-verification** at production-close time
   (`draft-ietf-jose-pq-composite-sigs`, `draft-ietf-cose-hpke`,
   `draft-ietf-cose-hpke-pq-pqt`). Nothing here depends on them.

Resolved by this freeze (previously open): operator disposition of every
PROPOSED value (Gate-2 §19, 2026-08-19); mechanical CDDL validation
(`tools/validate_profile.py` with a real, pinned CDDL validator); and the
**credential use-profile question — v16 credentials are Reusable-only**
(operator decision 2026-08-20, `DECISION-defer-oneshot-credentials`): there is
no `credential_use_profile` claim, `-70008` is not allocated, and
`OneShotTransaction`/consume-once semantics are deferred to a future amendment.
A fully `.size`-range-conformant reference validator (e.g. the Ruby `cddl` gem)
is an optional additional CI layer; the pinned gate re-checks those size ranges
in Python, so nothing in the freeze depends on it.
