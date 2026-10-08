# hyprstream-eval

Workflow eval harness — System One DAG node **P0.4** (Wave 2). Apache-2.0.

Runs **any jev-1-shaped subject** — our model, frontier LLMs through the
TypeSafe adapter, or Jev itself — over declared question sets and
verifiable-outcome items, scores the results under the single pinned
measurement protocol (`hyprstream-calibration`, P0.2), and persists
Arrow-native batches through the P0.3 metrics API.

## Pieces

- `subject` — the `Subject` trait (one seam, every arm). `HttpSubject` speaks
  the jev-1 `POST /v1/systemone` wire (stub, adapter, or Jev — the harness
  cannot tell the difference). `HashSubject` (deterministic in-process
  fixture) and `TruthSubject` (one-hot at verifiable truth) anchor offline
  runs and the scoring sanity tests.
- `teacher` — the teacher-ensemble reference client: roster with per-teacher
  **ToS class** provenance, **raw per-teacher probability vectors** (P0.5
  corrections are fit later, never baked in), average distribution
  (corrected-ensemble stand-in), argmax agreement.
- `run` — `EvalSet` (one question set × many state rows → one Arrow batch)
  and `EvalItem` (the one-question bench shape; `hyprstream-bench` items
  convert directly). `Harness` produces the observation stream plus the
  `hyprstream-decision` `DecisionSchema` batch.
- `score` — per-field ECE/Brier/NLL, SCE/ACE + RPS for the ordinal `score`
  primitive, protocol macro rule with bootstrap CIs (`GateReport`),
  per-family/per-stratum breakdowns, abstention rate, option-order flip rate
  over cyclic-permutation groups (S6b1).
- `persist` — `BatchSink` over any `hyprstream-metrics-api` `StorageBackend`;
  this crate never links the AGPL engine.

## Benches

Criterion scaffold (P0.7 wire floor + harness baselines; feeds P3.7):

```bash
cargo bench -p hyprstream-eval --bench wire_floor
cargo bench -p hyprstream-eval --bench harness_baseline
```

Every published number ships with the pre-committed disclosure language
(`hyprstream-calibration`'s `disclosure`): calibration-to-teacher ≠
calibration-to-reality.
