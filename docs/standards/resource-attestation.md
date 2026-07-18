# Resource attestation standards profile (pre-construction)

This directory slice is the **#1070 conformance and documentation closure** for the dual-attested resource-ownership epic #1064. It does **not** select a signature suite, a canonical production encoding, an anonymous construction, or a Privacy Pass token type, allocate any IANA/9P/MoQ codepoint, or claim IETF endorsement or adoption. Its RFCXML skeleton is a **pre-I-D work in progress** named `draft-hyprstream-resource-attestation-00`.

The generated text/HTML intentionally exercise xml2rfc's standard Internet-Draft boilerplate. The source has not been submitted, and the rendered boilerplate is not evidence of submission, adoption, endorsement, or an allocation.

## Ownership boundary with #1058/#1059

#1070 owns `draft-hyprstream-resource-attestation-00`, its resource-specific CDDL/vectors/traceability, and the CAS/9P implementation reports. #1058/#1059 retain ownership of `draft-hyprstream-privacypass-*` and `draft-hyprstream-moq-*` sources, the threat/property vocabulary, and stock-relay evidence. #1070 supplies reviewed resource-intent requirements, vectors, and proposed application-context changes to those documents; it does not fork or independently publish overlapping Privacy Pass/MoQ drafts.

## Contents

- `rfc/draft-hyprstream-resource-attestation-00.xml` and generated `.txt`/`.html` — reproducible RFCXML v3 scaffold.
- `cddl/resource-attestation.cddl` — canonical CDDL field structure (production DAG-CBOR pending #1065).
- `registry/resource-vocabulary.json` — bounded roles, principal kinds, profiles, identifiers, lifecycle, bounds, and transition invariants.
- `registry/resource-intent.json` — canonical `ResourceIntent` fields, labels, bounds, canonicalization, and construction status.
- `registry/resource-attestation-obligations.json` — machine-readable mapping for every RFCXML MUST/MUST NOT to an implemented test or a specification-only obligation with owner and blocker.
- `vectors/resource-intent-canonicalization-v1.json` — deterministic non-cryptographic baseline and 30 mutation-effective negative controls.
- `analysis/resource-authority-boundaries.md` — typed authority boundaries and invariants.
- `analysis/resource-threat-model.md` — adversaries, assets, properties the vectors exercise.
- `analysis/resource-privacy-collusion.md` — data-flow limits and collusion model.
- `analysis/resource-downgrade-matrix.md` — assurance floor and negative-control matrix.
- `analysis/resource-state-machine-recovery.md` — reference lifecycle, fencing, recovery.
- `analysis/resource-key-history.md` — key-history verification (no single classical signature).
- `analysis/resource-residual-risk.md` — residual risks and production gates.
- `analysis/resource-standards-gap.md` — standards-gap matrix, explicit non-allocations, draft disposition.
- `analysis/resource-interop-cas-9p.md` — CAS/blob and 9P interoperability evidence plans.
- `analysis/resource-encrypted-receipt.md` — encrypted receipt / selective-disclosure profile.
- `analysis/resource-test-evidence.md` — test-evidence scaffold and acceptance-criteria status.
- `adr/0001..0006` — dual attestation, stable identity, role separation, provisional visibility, reconciliation/compensation, privacy profiles.
- `runbooks/` — crash recovery, quarantine/manual review, reconciliation, key rotation.

## Vector coverage

`vectors/resource-intent-canonicalization-v1.json` covers, by category: canonicalization, mutation, replay, crossing, downgrade, privacy leakage, crash recovery, concurrency fencing, provisional visibility, and state/profile. A structurally valid baseline still terminates with the `construction-incomplete` refusal.

## Generation and validation

Generated text and HTML are deterministic outputs of the fully pinned xml2rfc 3.34.0 environment shared with the #1059 scaffold:

```bash
python3 tools/generate_resource_attestation.py
python3 tools/check_resource_attestation.py
python3 tools/check_resource_attestation.py --check-generated --xml2rfc xml2rfc
```

The `Standards artifacts` GitHub Actions workflow installs the complete pinned toolchain, validates the resource registries, vocabulary, and obligation manifest, runs every #1070 structural mutation, ensures every normative RFCXML sentence is manifested, and fails if regenerated text or HTML differs byte-for-byte.

## Draft disposition (recorded per acceptance criterion)

| Draft | Disposition |
|---|---|
| `draft-hyprstream-resource-attestation-00` | retained as an implementation profile (pre-submission; not submitted) |
| Privacy Pass extension (new token wire semantics) | not created (no new wire semantics; requirements supplied to #1058/#1059) |
| 9P ownership Internet-Draft | not created (standards-gap review concludes external interoperability is not warranted) |

## Production gates remaining

This slice lands the pre-construction scaffold and refuses production finalization (`construction-incomplete`). The end-to-end resource flows, independent security/privacy/cryptographic/protocol/standards reviews, and the CAS/9P interoperability execution are gated on the sibling issues #1065–#1072 and on those reviews. See `analysis/resource-residual-risk.md` and `analysis/resource-test-evidence.md`.
