# Clef native inference contract v1 (`clef-native-v1`)

**Status:** draft R0 artifact/encoding contract (source under review; not merged)
**Artifact pin:** `crates/hyprstream-decision-head/fixtures/clef-artifact-manifest.json` (schema_version 1)
**Base revision authored against:** `origin/main` 67eae00fdcd532f95a4fb9a79b042715dbde7bdd

Authority to execute, deploy, or relax this contract belongs to the active
request and its reviewers — not to this document. This file describes the
draft artifact contract only.

## 1. Scope

This contract fixes what the native (Rust, tch-rs) Clef decision path consumes and
produces at milestone R0: the pinned published FP8 artifact, its quantization and
loading profile, the record encoding rules it inherits from the upstream schema
source, the label/distribution mapping to the typed IR, and the evidence taxonomy
that separates provenance, parity, drift, accuracy, and performance claims.

It does **not** specify runtime wiring (R4), serving auth (R4/E2), GPU paths (R8),
training, or calibration fitting. It defines **no** approved decision thresholds:
none exist yet, and none may be inferred from this document.

Non-goals, explicit: no original high-precision (BF16 shard) execution yet; no
Python runtime or hosted-service dependency anywhere in the native path; no model
conversion; no media support in v1 (section 7).

## 2. Pinned artifact and provenance

| Role | Value |
|---|---|
| Preferred FP8 producer | `kurcontko/clef-flash-FP8-Dynamic` @ `7b43c2d799e7b5e9fef96b6137840088d75af0d6` |
| Upstream original | `Cloudflare/clef-flash` @ `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` (frozen main HEAD; first observed 2026-10-17 and unchanged since) |
| Head weights | `clef_head/joint_head.safetensors`, bit-identical (sha256) to the upstream original release head |
| Tokenizer | `tokenizer.json`, bit-identical (sha256) to the upstream original release tokenizer |
| Schema/encoding source | `joint_schema_model.py`, bit-identical (git blob sha1) to upstream; **evidence only — never executed, never copied into this repository** |

The producer self-describes (README) as an **unofficial** `llm-compressor`
FP8_DYNAMIC quantization of Cloudflare/clef-flash and is not affiliated with or
endorsed by Cloudflare. The producer's published aggregate evaluation numbers
(98.8% top-1 agreement with BF16 across 1,400 records) are **attributable
producer claims**, not our parity evidence (section 9).

Every file's durable pin is its SHA256 in the fixture manifest. Provenance
evidence, recorded per file: Hugging Face LFS oids (published content SHA256) for
LFS files and git blob SHA-1 for regular files, both at the pinned producer
revision; files additionally bit-identical to the upstream original are marked
with the upstream hash. No absolute host paths appear in the manifest; the
artifact is addressed by its upstream URL and content hashes only.

What this provenance **does not** establish: numerical parity of the FP8 port
against upstream. Hashes pin bytes, not behavior (section 9).

## 3. Encoding contract (inherited from the upstream schema source)

Native inference consumes typed records and reproduces the upstream token
encoding exactly. The normative behaviors below are read from the pinned
`joint_schema_model.py` and are binding on the native encoder (R2):

- **Media policy.** The encoding source supports optional `images`/`videos`
  through a processor; the native v1 path rejects records carrying `images`,
  `videos`, or any media-adjacent field with an explicit typed error (section 7).
- **Canonical rendering.** A top-level string `state` or `instructions` renders
  as the bare string itself — no JSON quoting. Every nested or non-string value
  renders through `json.dumps` with `sort_keys=true`, `ensure_ascii=false`,
  compact separators `,`/`:` (sorted keys, no spaces, non-ASCII preserved). The
  native serializer must byte-match this rendering for identical inputs.
- **Cloudflare-ordered fields.** Question order is the record's `questions` map
  order as authored (Cloudflare ordered-fields semantics); the native path must
  preserve authored order and must not reorder or deduplicate question IDs.
- **Option enumeration per type** (`QUESTION_TYPES`: `noul=0`, `choice=1`,
  `score=2`):
  - `noul` — exactly two options, **model order `true`, then `false`**, with
    fixed upstream descriptions unless the question overrides `criteria`.
  - `choice` — options **sorted by option ID** (lexicographic byte order of the
    ID string), each rendered as `{"option_id": ..., "description": ...}`. The
    canonical raw output order is this sorted-ID order; consumers expecting the
    legacy typed-IR **authored** choice order remap by label (never by position).
  - `score` — options enumerated by list position (`"0"`, `"1"`, ...), order =
    authored `criteria` order (NOT sorted). Raw criteria entries are
    **descriptions keyed by indices** `0..N-1`; numeric-looking criteria values
    do NOT automatically become legacy IR score levels — mapping raw indices to
    IR score levels requires a distinct, explicit normalization adapter.
  Answers normalize back **by option ID / label**, never by vector position.
- **Label-order mapping (noul).** The model emits the two-way softmax in model
  order `true`,`false`; the typed IR/Arrow distribution order is `[P(false), P(true)]`
  (existing IR contract; see `hyprstream-decision-head` lib doc). The native path
  performs this single explicit remap. Score levels and ties are preserved.
- **Integer domain and `-0` (normative; implementation future).** Raw-JSON
  integers are admitted in the i64/u64 domain; integers outside it must be
  rejected with a typed error rather than silently converted, and integer `-0`
  must canonicalize to `0` while floating `-0.0` is preserved. This PR pins the
  required behavior and its tests-to-be; the implementing canonicalization
  belongs to the native `ClefRequest` encoder (`clef_render` in
  `crates/hyprstream-decision-head/src/clef/encoding.rs`) and lands with that
  PR — it is absent at this head and no implementation is claimed here.
  Typed-IR values normalize to f64 as a **distinct** domain — raw-JSON domain
  rules must not leak into, or be inferred from, IR-side f64 normalization.
- **Structure.** Fixed chat prefix/suffix frames (system prompt, `STATE:`,
  `\n\nSCHEMA FIELDS:\n`, per-field `FIELD n / ID / TYPE / INSTRUCTION /
  ALLOWED OPTIONS / OPTION n / END FIELD` blocks, assistant frame with empty
  `think` block and `JOINT SCHEMA DECISIONS:` terminal), per-option token spans,
  per-question spans; question spans shift by the state length. Empty records or
  zero questions are errors upstream and remain errors natively.
- **Token budget.** Reference budget is a 16384-token maximum. The schema
  (everything except state tokens) must fit first — upstream raises when the
  schema alone exceeds the budget; the native path reports a typed schema-
  overflow error and never truncates schema. State truncates to fit, and applied
  limits/truncation are reported. Native may run a smaller explicit cap when CPU
  capacity demands; any reduction is reported per request.

## 4. Quantization profile (published, pinned)

`config.json` `quantization_config` (compressed-tensors, format `float-quantized`):

- **Weights** — FP8 E4M3 (`F8_E4M3` in the safetensors header), symmetric,
  **per-output-channel static scales**, observer `memoryless_minmax`, stored as
  200 `<layer>.weight_scale` companions (BF16 in file) for 200 quantized Linear
  weights. The `.weight_scale` (not `.weight_scale_inv`) companion form is
  normative for this artifact.
- **Input activations** — FP8, symmetric, **dynamic per-token scales**
  (`strategy: token`, `dynamic: true`): each forward quantizes activations
  per token row (QDQ) against that row's max-abs value.
- **Ignored floating layers** (159 entries, kept BF16): all `model.visual.*`
  Linear layers and the merger (110); `linear_attn.in_proj_a`/`in_proj_b` in
  all 24 linear-attention (GDN) language layers of 32 total — the 8
  full-attention layers are quantized (48 tensors); and `lm_head` (1) — the
  output-embedding table, whose normalized rows feed the head's lexical priors
  and which must therefore be preserved and addressable. The vision tower is
  present in the checkpoint but untested upstream; the producer's own server
  disables it.
- `kv_cache_scheme` is null; the decision path is a single unpadded forward with
  no cache: one fresh record per forward, no inherited state.

**Runtime profile admission is pinned to the producer's toolchain version**:
compressed-tensors `0.19.0` @ `525a7a7b84ebccdbc0db07956338b6f8c851d58f`.
Normative semantics read from that pin:
[`calculate_qparams`](https://github.com/vllm-project/compressed-tensors/blob/525a7a7b84ebccdbc0db07956338b6f8c851d58f/src/compressed_tensors/quantization/utils/helpers.py#L80)
computes `scale = amax / (bit_range/2)` and replaces **only** `scale == 0` with
the dtype epsilon (zero-only epsilon; no small-amax clamp in this pinned path);
[`_quantize_dequantize`](https://github.com/vllm-project/compressed-tensors/blob/525a7a7b84ebccdbc0db07956338b6f8c851d58f/src/compressed_tensors/quantization/lifecycle/forward_helpers.py#L168)
divides by the scale, clamps/rounds, and multiplies back. Other profile versions
require explicit audit before admission. CPU Float32 is an explicit **compute
profile**: it does not claim bitwise BF16 producer-arithmetic parity nor
upstream full-model port parity.

The native loader must validate this profile before allocating: scale presence
and geometry per quantized tensor, dtype identities, ignore-list agreement, and
the E4M3FN interpretation (never FNUZ, never integer reinterpretation). Missing,
extra, nonfinite, or geometry-mismatched tensors are typed load errors. Strict
finite checks are normative at load: any NaN/Inf weight, scale, or norm
parameter fails the load. The original finite-value audit covers only the 200
weight-scale tensors (1,425,408 values: zero 0, negative 0, nonfinite 0; see
`fp8-scale-audit-receipt.json` beside the artifact manifest). The manifest's head
content hashes establish artifact identity, not finiteness. Separate archived
development load-time evidence now binds the checked native constructor,
executed source/artifact receipts, and all 122 tensors of the pinned head:
[`native-head-finite-load-receipt.json`](../../crates/hyprstream-decision-head/fixtures/native-head-finite-load-receipt.json).
All nine pinned inputs and a fresh-checkout command are included in the
[archived evidence bundle](../evidence/clef-head-finite-load-20261018/README.md).
Its reproducible public-receipt derivation performs no tensor execution. This is
not a standalone head scan or current-main/new-candidate qualification;
whole-backbone weight finite-load verification and an implementation PASS are
not claimed here.

## 5. Compute profile (CPU v1)

- **CPU Float32 compute with packed FP8 weights**: quantized tensors stay packed
  E4M3 at rest; matmuls decode tiles to Float32 (or BF16-upcast Float32 where
  equivalence is evidenced) with scale application at Float32 precision. Scale
  values loaded from BF16 are upcast exactly (BF16 -> F32 is exact); no re-rounding.
- **Faithful W8A8 semantics**: dynamic per-token activation quantize/dequantize
  follows the published compressed-tensors scheme (symmetric, per-token max-abs
  scaling, E4M3 clamp/round). CPU has no hardware FP8 `scaled_mm` in this stack;
  the QDQ emulation **is** the correctness route. A weight-only float-activation
  run is a separately labelled ablation/diagnostic and must never be presented as
  candidate parity.
- No full-model FP32 expansion or cached decoded copy merely to pass a call;
  decode workspace is bounded (plan RQ).
- No GPU device selection, initialization, or test in the CPU milestone; device
  policy is CPU-explicit with typed failure on any CUDA/ROCm request.

## 6. Output contract

One logit per option, concatenated over questions in encoded order; the head
consumes final normalized hidden states plus **output-embedding rows** (lexical
priors) and must not materialize full token-by-vocabulary logits. Per question:
softmax over that question's option logits; answers keyed by option ID/label;
noul maps to the IR `[P(false), P(true)]` pair (section 3). Score questions
carry index-ordered raw outputs with ties preserved; the index-to-level mapping
is the explicit adapter of section 3, not an identity transform.

**Calibration is null**: raw inference only, provenance marked uncalibrated.
Peaked distributions are not calibration evidence. No conformal sets are
synthesized. **No decision thresholds are approved**; any thresholding consumer
is out of scope for this contract until separately decided.

## 7. Unsupported media and failure behavior

Records with `images`, `videos`, or media processor fields produce an explicit
typed unsupported-media error in v1. Media is a later milestone with its own
processor/parity contract. Strict finite load,
typed schema-overflow, budget/truncation reporting, and label-space validation
errors are normative (sections 3-4). There is **no hidden fallback**: no hosted
service, no remote model, no Python runtime, no silent higher-precision or
lower-precision substitution. A higher-precision diagnostic mode, when built
(RQ), is explicit per request — never automatic.

## 8. What is and is not claimed

- **Claimed here:** provenance/pinning of the preferred artifact and upstream
  original; encoding/label semantics; quantization and compute profile; loader
  validation requirements.
- **Not claimed:** CPU full-model PASS; numerical parity with Cloudflare's
  original; quantization drift bounds; label accuracy; performance envelopes.
  None of these exist yet for the native path.

## 9. Evidence taxonomy (binding vocabulary for later milestones)

1. **Trusted upstream parity** — native outputs match pinned *upstream-produced*
   expected tensors within declared tolerances. Status: **not established from
   the inspected pinned trees**: no numerical fixture outputs (per-tensor
   logits/hidden states) were found in the pinned revisions of either
   repository; the producer's `scripts/build_data.py` references public
   **input** datasets only, and its README tables are aggregate producer claims.
   This absence claim is scoped to those two pinned trees — external fixtures
   may exist elsewhere; none are pinned or admitted by this contract. Producing
   an independent oracle is a separate authorized task; until then, nothing may
   cite upstream parity.
2. **Quantization drift** — native FP8 vs native higher-precision (same code,
   same inputs). Measures our quantization emulation only; never cites upstream.
3. **Label accuracy** — scored against gold labels on evaluation datasets.
   Independent of parity and drift.
4. **Performance** — measured load/warm/cold time, peak RSS, threads, dtype.
   Informational until thresholds are set; never a correctness proxy.

Any report must name which of these it is; conflating them is a review defect.

## 10. Licensing and third-party provenance

| Component | License | Source/pin |
|---|---|---|
| Cloudflare/clef-flash (weights, tokenizer, schema source) | Apache-2.0 | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` |
| kurcontko/clef-flash-FP8-Dynamic (FP8 artifact) | Apache-2.0 (upstream terms retained) | `7b43c2d799e7b5e9fef96b6137840088d75af0d6` |
| llm-compressor / compressed-tensors (producer toolchain; format semantics referenced) | Apache-2.0 | pinned `compressed-tensors` `0.19.0` @ `525a7a7b84ebccdbc0db07956338b6f8c851d58f` (section 4); artifact `config.json`/`recipe.yaml` record the applied scheme |
| Qwen3.5 tokenizer lineage (tokenizer.json) | Apache-2.0 (via Cloudflare release) | bit-identical to upstream `tokenizer.json` |
| This repository's Clef contract/fixture files | Apache-2.0 (permissive leaf: `hyprstream-decision-head`) | repo license map |

Model/weights remain Apache-2.0 under upstream terms; the artifact is not
relabelled to satisfy other policy preferences. The schema
source is referenced as pinned evidence and is **not** vendored; the native
implementation is written from this contract, with the source consulted as
documentation. `clef-artifact-manifest.json` pins contents, not code.

## 11. Verification recipe (reproducible)

Given a transferred artifact tree, verification is: (a) sha256 every file against
`clef-artifact-manifest.json`; (b) for LFS files, confirm the sha256 equals the
producer revision's HF LFS oid; (c) for regular files, recompute git blob SHA-1
and compare to the revision's tree oids; (d) for upstream-identical files,
compare against the frozen Cloudflare revision's hashes. All steps are offline
hashing; none execute model code.

## 12. Revision history

- rev 1 (R0 draft, 2026-10-18): initial contract; artifact pinned at
  `7b43c2d7…`, upstream frozen at `8b2e5fd1…`; profile, mapping, and evidence
  taxonomy fixed; parity/accuracy/performance explicitly unclaimed.
- rev 2 (2026-10-18, root review C1-C7): removed execution-authority/planning-
  file wording; canonical rendering split (bare top-level strings vs nested
  JSON) and number-domain rules marked implemented; per-file provenance
  algorithm corrected (LFS sha256 vs git blob sha1; head-config path remap
  verified; 10/21 identical, no root-config equality claim); producer-specific
  classification replaces "no upstream counterpart" where a counterpart exists;
  compressed-tensors 0.19.0 @ 525a7a7b… pinned with zero-only-epsilon and FP32
  compute-profile semantics; choice sorted-ID output vs authored-IR remap and
  score index-vs-level adapter made explicit; fixture-absence claim scoped to
  the pinned trees; scale finite-value audit attached.
- rev 3 (2026-10-18, review correction): removed the unsupported head-finiteness
  claim. The attached audit covers scales only; head content hashes do not
  prove finiteness. The strict runtime finite-load requirement is unchanged.
- rev 4 (2026-10-18, archived evidence binding): attached a separate reproducible
  receipt for the previously executed f2a82911/e90affaa development native load
  of all 122 pinned head tensors. The original scale audit remains scale-only;
  no standalone head scan or current-main/new-candidate qualification is claimed.
