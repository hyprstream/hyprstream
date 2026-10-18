# hyprstream-synthesis

Teacher-agnostic synthetic corpus pipeline — System One program, DAG node **P1.3**.

Generates `(state, question spec, answer)` triples for decision-model
distillation: procedurally synthesized jev-1 question specs across workflow
families (`triage`, `compliance`, `extraction`, `approvals`) with
rubric/option paraphrase diversity, cardinality variation, and noul/choice/
score primitive mixes, answered by a pluggable teacher ensemble. **No human
labeling** — answers are raw per-teacher probability vectors, persisted
verbatim so corrections (fit by P0.5) can be applied at training time (P1.4).

## Guarantees

- **Mandatory augmentations (S6b1)**: full cyclic option-order closure on
  choice items + rubric/instruction paraphrase variants (always on;
  `paraphrase_variants >= 2` enforced).
- **Firewalls**: the frozen vob-1.1 benchmark manifest is a consumed,
  pin-verified artifact (release id + manifest/DISCLOSURE blake3 checked at
  load). Gate families are never synthesized; item-hash collisions fail the
  run closed.
- **Provenance + distributability**: every row records its answering
  teachers' roster pins (id, version, ToS class). Rows with any encumbered
  (`api-prohibited`) teacher are `distributable = false` — never published,
  never in Apache/MIT artifacts (`Corpus::publishable`).
- **Label-distribution control + dedup**: optional balancing over the
  corrected-ensemble argmax; canonical-hash dedup.
- **Deterministic**: same config + roster ⇒ bit-identical corpus.

## CLI

```text
hyprstream-synthesis pins                          # the consumed frozen pins
hyprstream-synthesis run --roster roster.json [--manifest path] \
    [--config config.json] [--out corpus.jsonl] [--publishable-only]
hyprstream-synthesis roster-md --roster roster.json  # DISCLOSURE table
```

The CLI drives deterministic `HashTeacher` stand-ins (roster versions pinned
`simulated:*`). Real teacher adapters implement `teacher::Teacher` in the
P0.4 eval harness / P1.4 training loop; the roster JSON is the integration
contract. The teacher roster itself is authorized by humans out-of-band and
recorded with ToS classes in `DISCLOSURE.md`.

## Licensing

Apache-2.0. This crate is a `permissive_roots` member
(`.github/license-boundary.toml`): no dependency path to any AGPL crate.
Corpus distributability additionally depends on the teacher roster's ToS
classes — see `DISCLOSURE.md`.

### Required content admission (source revision)

The frozen `syn-item/1` and `vob-item/1` hashes still identify artifacts;
contamination checks now use the additive shared typed `decision-content/1`
projection. It includes state, question kind, instructions and ordered criteria,
with explicit types/presence and length framing. It excludes opaque question IDs,
truth, family, seed and augmentation/provenance coordinates. It detects exact
typed content copies; it does not claim semantic/paraphrase equivalence.

`Firewall::load` still verifies the original frozen manifest and DISCLOSURE pins,
but does not authorize synthesis by itself. Attach a COMPLETE benchmark index
with `with_attested_items(&[hyprstream_bench::Item])` or
`with_content_index_json`. Every supplied item must match a distinct pinned
manifest ID and its unchanged artifact hash. Missing, incomplete, duplicate,
unknown or corrupt content fails closed. Pipeline preflight requires this index,
and every generated item is content-checked before teacher inference. No pin or
release override is exposed; the tiny test constructor exists only under cfg(test).

The CLI requires `--content-index index.json` before it constructs stand-in
teachers. The caller-supplied JSON schema is:

```json
{"version":"decision-content/1","items":[{"id":"...","family":"arith","stratum":"clean","group":"...","seed":7,"truth":1,"state":"...","question":{"instructions":"...","body":{"type":"noul","criteria":null}}}]}
```

Choice bodies carry `options: [{"name":"...","rubric":null}]`; score bodies
carry `levels: [null, "..."]`; noul criteria, when present, carry optional
`on_false` and `on_true` typed entries. Strata are `clean`, `nearmiss`, or
`perm-N`. Caller-supplied hashes are never trusted in place of actual content.
The full production index has NOT been generated or published by this revision.
Until an independently verified complete index is supplied, production corpus
runs are deliberately refused. Existing frozen artifacts/pins remain unchanged.

Raw teacher vectors and imported ensemble rows now reject empty/duplicate or
unversioned teacher rosters and nonfinite, out-of-range, wrong-width or
unnormalized probabilities. `spread` and `drop_worst_teachers` now return Result
so malformed imported rows cannot silently enter fallback calculations.

Whole-group balancing/quota semantics remain a separate pending owner decision;
this source slice does not fix or approve the existing row-wise balanced selector.
No corpus generation, external teacher inference or training is part of this change.

Index attestation additionally enforces the source-qualified frozen profile
`vob-1.1-text/1`: state is Str, instructions are Some(Str), noul criteria are
None, all choice rubrics are None, and all score level rubrics are None, with
matching kind/body. This is the actual vob1.1 generator profile. Old artifact
canonical_text hashes cannot distinguish some type/presence substitutions and
omit noul/score rubrics; such alternatives are rejected BEFORE hash acceptance.
The shared fingerprint itself remains generic and typed for future profiles,
which would require independent attestation. No frozen pins are overridden.
