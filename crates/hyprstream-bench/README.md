# hyprstream-bench — verifiable-outcome calibration benchmark (vob-1.0)

The System One program's truth anchor: score-bearing decision items with
**mechanically verifiable outcomes**, emitted as jev-1 question specs
(`hyprstream-decision`). All items are **procedurally generated** from the
published generators in `src/gen/` and pinned seeds — immune to
teacher-pretraining contamination by construction.

- **Strata**: `clean`, `nearmiss` (near-miss JSON, near-miss arithmetic,
  close distractors), and `perm-N` — a cyclic-permutation stratum
  (CircularEval precedent) measuring option-order robustness on every choice
  item. Score levels are never permuted (level order is the semantics).
- **Family holdout firewall**: `temporal` and `syllogism` are designated
  **gate** families, frozen at release time and never synthesized into
  training data; the zero-shot transfer gate measures the model on them.
- **Frozen manifest**: `manifest/vob-1.0.manifest.json` holds a blake3 hash
  of every item, frozen before any synthesis run. It is the audit artifact
  for both the item-level contamination firewall and the family firewall.

## CLI

```text
hyprstream-bench generate <outdir>   # items.jsonl + vob-1.0.manifest.json
hyprstream-bench verify [manifest]   # regenerate + compare (default: committed)
hyprstream-bench families            # print family designations (the firewall)
```

`verify` regenerates the full benchmark from the pinned config and compares
every item hash against the manifest — the freeze is checked in CI by the
`release.rs` integration tests.

## Licensing

Harness: **Apache-2.0**. Generated items: **CC-BY-4.0**. All published
results ship with the pre-committed disclosure in
[`DISCLOSURE.md`](DISCLOSURE.md) (its hash is pinned in the manifest).
