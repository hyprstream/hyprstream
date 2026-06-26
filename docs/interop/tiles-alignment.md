# Tiles interop: MIR (Machine Intelligence Resource) alignment

This document records how hyprstream adopts the **MIR** model-naming/classification
schema defined by Tiles ([tiles.run](https://www.tiles.run)) and how MIR maps to
our internal model identifiers. Implements GitHub #283 (epic #131).

Source of truth for the grammar/taxonomy: the MIR section of
<https://www.tiles.run/llms-full.txt> (fetched 2026-06), cross-checked against
issue #283. Implementation: `crates/hyprstream/src/storage/mir.rs`.

## Grammar

```
mir:<domain>.<arch>.<variant>:<series>
```

Example: `mir:model.transformer.clip-l:stable-diffusion-xl`

| Field     | Meaning                                              | Example               |
|-----------|-----------------------------------------------------|-----------------------|
| `domain`  | broadest classification axis                        | `model`               |
| `arch`    | architecture taxonomy label                         | `transformer`         |
| `variant` | free-form normalized sub-classification             | `clip-l`              |
| `series`  | normalized model series                             | `stable-diffusion-xl` |

The Tiles prose is loose about whether the third dot-field is the "variant" or
"series"; issue #283's authoritative grammar names the third dot-field `variant`
and the post-colon field `series`. We implement that. The canonical example
parses under this grammar.

### Domains

`dev`, `model`, `ops`, `info`, and `arch` (Tiles also writes the last as
`architecture`; we accept both). Ordered most-specific to most-general.

### Architecture taxonomy

Known labels (transcribed from the Tiles taxonomy; canonical emission is
lowercase): `gru`, `rbm`, `tae`, `vae`, `lstm`, `resnet`, `cnn`, `rcnn`, `rnn`,
`brnn`, `gan`, `ssm`, `detr`, `vit`, `moe`, `aet`, `stst`, `art`, `lora`,
`controlnet`, `unclassified`, plus `transformer` (used by the canonical example).

**Superset behaviour:** an architecture label that is *not* in the taxonomy is
accepted as an extension (`Arch::Extension`) rather than rejected, so a label
Tiles adds later — or a project-local one — parses instead of hard-failing.
Canonical emission stays stable (always lowercase).

### Series normalization

Rules (from the Tiles spec):

- Lowercase, hyphen-only separators.
- Strip org/library prefix (drop everything before the last `/`).
- Remove parameter-size tokens (`7b`, `0.6b`, `13b`, ...).
- Collapse non-breaking semantic versions to the breaking/major component
  (`v1.2` -> `v1`).
- Strip known library-name suffixes (`-diffusers`, `-transformers`, ...).

Spec examples:

| Input                                          | Output            |
|------------------------------------------------|-------------------|
| `tencent-hunyuan/hunyuandiT-v1.2-diffusers`    | `hunyuandit-v1`   |
| `black-forest-labs/FLUX.1-dev`                 | `flux1dev`        |

**Documented spec ambiguity:** the two examples disagree on separators — hunyuan
keeps the `-` before `v1`, flux drops all separators. The distinguishing feature
is that flux contains a *dotted bare-number* token (`FLUX.1`) that merges a digit
into the name. We reconcile both deterministically:

- If any surviving token is a dotted bare-number (e.g. `flux.1`): flatten the
  whole series to a single lowercase alnum run -> `flux1dev`.
- Otherwise: keep `-` separators, collapse `vN.M` -> `vN`, drop stray `.` ->
  `hunyuandit-v1`, `stable-diffusion-xl`, `qwen3`, `llama-2`.

## Internal model identity

Our internal identity is `ModelRef { model, git_ref }`
(`crates/hyprstream/src/storage/model_ref.rs`), where `model` is a registry name
that is typically a HuggingFace-style id (`org/repo`, e.g. `Qwen/Qwen3-0.6B`) or
a bare local name (e.g. `llama3`), and `git_ref` selects a revision. Models are
tracked repositories in the git-native registry (`git2db`), cloned from URLs like
`https://huggingface.co/Qwen/Qwen3-0.6B`. A model is resolved by name via
`registry.get_by_name(name)`.

## Internal id <-> MIR mapping

| Direction          | Function                                | Lossy?                                   |
|--------------------|-----------------------------------------|------------------------------------------|
| internal -> MIR    | `mir_from_internal_name(name)`          | **Yes** — see below                      |
| internal -> MIR    | `mir_from_internal_name_with(name, arch, variant)` | partial — domain still defaults to `model` |
| MIR -> internal    | `internal_series_from_mir(mir)`         | **Yes** — recovers series only           |
| name match key     | `series_key_for_name(name)`             | normalization key for resolution         |

### internal -> MIR (lossy)

A HF repo id / local name carries no architecture, domain, or variant
information, so `mir_from_internal_name` defaults:

- `domain = model`
- `arch = unclassified`
- `variant = base`
- `series = normalize_series(name)`

e.g. `Qwen/Qwen3-0.6B` -> `mir:model.unclassified.base:qwen3`. Callers that know
the architecture/variant should use `mir_from_internal_name_with(...)` or
`Mir::new(...)`.

### MIR -> internal (lossy, partial inverse)

`internal_series_from_mir` recovers the normalized **series** only. The
org/library prefix, the original casing, and the parameter-size of the source HF
id are not recoverable. The series is therefore used as a *match key* against the
registry rather than to reconstruct a canonical HF id.

### No exact bijection

Because both directions lose information there is no exact round-trip between a
HF repo id and a MIR. The round-trip that **is** defined and tested:
series normalization is idempotent, so `internal -> MIR -> series` is stable once
normalized (`series_key_for_name(name) == internal_series_from_mir(mir_from_internal_name(name))`).

## Registry integration

`ModelService` (`crates/hyprstream/src/services/model.rs`) resolves a model
identifier through `resolve_model_ref`, which accepts:

- a registry name / HF repo id (`Qwen3-0.6B`, optionally `name:ref`), and
- a MIR string (`mir:...`).

For a MIR string it lists registered models and matches the MIR `series` against
each model's `series_key_for_name(name)` (via
`mir::resolve_mir_against_names`). A unique match resolves to that model's
`ModelRef` (default branch — MIR carries no revision); zero matches or an
ambiguous (>1) match is an error. The HF-repo-id and `name:ref` paths are
unchanged.

`ModelService::mir_for_model(name)` emits a (lossy) MIR for a registered model.

## Tests

`crates/hyprstream/src/storage/mir.rs` `#[cfg(test)]` covers: parse of the
canonical example; all domains; every known arch label; unknown-arch extension;
malformed rejection (missing scheme/series/domain/arch, extra colon, 4 dot
fields); `Display` round-trip + idempotency; case normalization; series
normalization (hunyuan, flux, param-size); internal<->MIR mapping round-trip;
and registry resolution by MIR (match, not-found, ambiguous).
