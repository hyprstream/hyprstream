# Pre-committed disclosure — vob-1.0 benchmark results

This text is **frozen at manifest freeze time** (its blake3 digest is pinned
inside `manifest/vob-1.0.manifest.json`). Every published result measured
against the vob-1.0 benchmark must ship with this disclosure verbatim.

---

Results reported against vob-1.0 are claims about **mechanically checkable
question families only** (arithmetic word problems, JSON well-formedness,
unit conversion, plus the held-out gate families temporal reasoning and
set-logic syllogisms). Truth on these items is computed, not judged.

Calibration and accuracy claims are additionally scoped to:

- **teacher-agreement splits** — where truth is teacher-ensemble agreement,
  it is labeled as such and never conflated with verifiable truth; and
- **realized production telemetry from the stated date onward** — where
  reported, the collection window and version triple are named.

**Calibration-to-teacher ≠ calibration-to-reality.** Agreement with a teacher
ensemble bounds consistency with that ensemble, not correctness in the world.
The verifiable-outcome strata and the held-out gate families exist precisely
to anchor the claims that teacher agreement cannot.

The held-out gate families were designated at benchmark freeze, before any
training synthesis ran, and were never synthesized into training data. The
frozen manifest (per-item hashes) is the audit artifact for both firewalls.
