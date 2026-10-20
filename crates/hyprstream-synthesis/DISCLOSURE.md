# Pre-committed disclosure — P1.3 synthetic training corpus

This text is the **pre-committed disclosure** that must accompany every
published corpus export and every trained-artifact claim derived from the
`hyprstream-synthesis` pipeline. It mirrors, and does not replace, the frozen
benchmark disclosure (`crates/hyprstream-bench/DISCLOSURE.md`, blake3
`9bf26d86de584f07b1e4138dfcf197ff00c9563089365e861ffbabe084cb533d`, pinned in
the consumed vob-1.1 manifest).

---

## No human labeling; calibration-to-teacher ≠ calibration-to-reality

Every corpus row is a procedurally synthesized (state, question spec) pair
answered by a **teacher ensemble** — no row is human-labeled. The raw
per-teacher probability vectors are persisted verbatim; any per-teacher
correction (fit on verifiable-outcome splits) is applied at training time and
recorded with the artifact that applied it.

Agreement with the teacher ensemble bounds consistency with **that ensemble**,
not correctness in the world. Verifiable truth anchors live in the vob-1.1
benchmark only; synthesized rows carry teacher distributions, and claims made
from them are labeled as teacher-agreement claims.

## Firewalls

- The held-out benchmark gate families (`temporal`, `syllogism`) were
  designated at the vob-1.1 freeze, before any synthesis ran, and are
  **never synthesized into training data** (enforced structurally — synthesis
  families share no ids with benchmark families — and re-checked at run time
  against the consumed frozen manifest, blake3
  `a19596ff13fbf7c09c259d2ed2ec5c8aafdca49a23616099e5b17f84a0a5be8f`).
- No synthesized item may hash-collide with a frozen benchmark item
  (canonical-hash check against the same manifest).

## Teacher roster and ToS classes (distributability)

The teacher roster is authorized by humans **out-of-band**; this table
mirrors that decision record. ToS classes per the S2 spike: `open-weights`
(self-hosted open models — distillation unconstrained),
`api-distillation-permitted` (API terms permit distillation, e.g. DeepSeek),
`api-prohibited` (API terms prohibit distillation — OpenAI / Anthropic /
Google; such teachers serve evaluation only).

| teacher | version | ToS class | distributable |
|---|---|---|---|
| *(roster pending the out-of-band owner decision — record here at authorization time; every corpus row carries its answering teachers' pins and a derived `distributable` flag)* | | | |

Rows answered by any **encumbered** (`api-prohibited`) teacher are flagged
`distributable = false`: they are never published and never included in
Apache/MIT-licensed artifacts (dataset exports or trained model releases).
Under the program licensing policy (model artifacts MIT), a flagship model
artifact may only be trained on distributable rows.

## Augmentation

Option-order permutation (full cyclic closure on choice items) and
rubric/instruction paraphrase variants are **mandatory** augmentations,
always on in this pipeline (S6b1: reordering alone swings accuracy 13–75 pp;
the debiasing is data-side). Augmented variants are linked by the row's
`group` field with their `(rotation, paraphrase)` coordinates recorded.
