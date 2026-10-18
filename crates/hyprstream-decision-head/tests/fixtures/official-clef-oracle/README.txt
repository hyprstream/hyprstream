OFFICIAL OFFLINE CLEF ENCODER / JOINT-HEAD ORACLE
This fixture tooling is for offline verification only. The product serves through
native Rust/libtorch; it never imports or executes these Python programs.

Pinned source and original artifact: Cloudflare/clef-flash revision
8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3. manifest.json pins the verified source,
head/config/tokenizer and embedding shard identities, corpus/generator hashes,
external tensor bundle SHA256, and fixed pre-comparison numerical tolerances.
records.json contains complete upstream-generated input token IDs, spans, question
and option order, plus state admission accounting for five public synthetic
records. expected.json contains the upstream joint head's 11 question outputs.

The 47MiB tensors.safetensors stays outside Git. It contains deterministic saved
synthetic hidden inputs and selected ACTUAL original LM output-embedding rows,
plus official head-normalized hidden states/logits/probabilities. There is no
mirrored native implementation in the oracle: generate_head_v1.py imports the
unmodified pinned official class and encode_record function. A sparse indexing
adapter supplies exactly the original rows requested by that class, avoiding a
multi-GB vocabulary table. Strict meta-device assignment avoids a duplicate head
allocation. Source-head BF16 values are represented exactly in Float32 compute.

Original head fixture generator SHA256:
51974abb9ba4d9a24f6d18b81139d51821da2cd9b3f2c70598d9bf60aef7b0ba
The generator is preserved as generate_head_v1.py. Its default corpus is corpus.json.
requirements-head-cpu.lock preserves its initial 37-package environment.
generate.py is the full-oracle-capable successor that records observed attention
configuration. requirements-cpu.lock additionally pins matching CPU torchvision,
which the official full AutoProcessor requires even for text-only records.
JENSEN-RUNBOOK.txt describes the separately admitted full reference profile.

Generate the original head-only fixture in a new output directory:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python generate_head_v1.py \
  --model-dir /absolute/original-pinned-model \
  --artifact-manifest /absolute/verified-bf16-artifact-manifest.json \
  --output /absolute/new/head-f32-v1 --mode head --dtype float32 --device cpu --threads 2
Core pins: torch2.11.0+cpu, transformers5.10.2, tokenizers0.22.2, safetensors0.7.0.
Package bootstrap is task-local; do not install into a global Python environment.

Run native parity through the CPU helper / BuildQ:
HYPRSTREAM_LIBTORCH_VARIANT=cpu \
CLEF_OFFICIAL_ORACLE_DIR=/absolute/head-f32-v1 \
CLEF_HEAD_DIRECTORY=/absolute/original-pinned-model \
CLEF_NATIVE_ORACLE_OUTPUT_DIR=/absolute/new/native-head-f32-v1 \
  /absolute/hyprstream-build.sh run -- cargo test -p hyprstream-decision-head \
  clef_official_encoder_head_parity --lib -- --ignored
The ignored test compares exact encoding and committed expected vectors, exports
native tensors and per-question metrics, and uses limits fixed BEFORE comparison:
logit absolute error <=1e-4; probability absolute error <=1e-5; relative tolerance0.
Use compare_native.py with matching explicit bounds for a separate hashed report.

Observed first comparison: all exact token/span/order/accounting checks passed;
max logit error2.384185791015625e-7, probability error5.960464477539063e-8,
head-normalization error0. Total Python process peak RSS1,268,780KiB (includes
Torch/Transformers/tokenizer overhead); head weights alone are about465MiB.

This proves encoder/head port parity on these saved synthetic hidden inputs.
It does NOT prove the backbone, FP8 quantization profile, calibration, performance,
full CPU serving, GPU qualification, or R5 acceptance. Full original-Float32
backbone and candidate-FP8 comparisons remain separate gates.

Saved native backbone export (no full model load)
------------------------------------------------
The separate ignored clef_export_native_head test accepts:
  CLEF_OFFICIAL_ORACLE_DIR: original full Float32 CPU oracle records/provenance
  CLEF_HEAD_DIRECTORY: pinned original tokenizer and original joint head
  CLEF_NATIVE_INPUT_TENSORS: saved native hidden and actual LM row safetensors
  CLEF_NATIVE_INPUT_PROFILE: original-f32 or fp8-w8a8
  CLEF_NATIVE_ORACLE_OUTPUT_DIR: distinct task-owned output directory
Run via the canonical CPU build helper:
  cargo test -p hyprstream-decision-head clef_export_native_head --lib -- --ignored
It checks exact official encoding and exports the actual native hidden/rows,
head-normalized hidden, logits, probabilities and labels. Export success is
not a numerical parity assertion. Apply the predeclared original full-port
comparison separately; native-to-native FP8 development drift is a distinct
gate. A head-only second export tests head repeatability only, not backbone
process restart or RPC determinism.
