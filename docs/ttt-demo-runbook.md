# Test-Time Training (TTT) Demo Runbook

**Audience:** Engineers, ML practitioners, and prospective users demonstrating Hyprstream's online learning capabilities.

**What you'll demonstrate:** A model adapting to domain-specific text in real-time, with measurable perplexity reduction, per-tenant isolation, and versioned adapter export.

**Time:** ~10 minutes

## Prerequisites

- Hyprstream installed and running (`hyprstream service install --start`)
- A model registered (e.g., `qwen3-4b-2507` from HuggingFace)
- MCP connection active (Claude Code: `/mcp`, or `HYPRSTREAM_TOKEN` env var for CLI)

## Step 1: Load the Model

Load the model on a dedicated branch (worktree). This isolates TTT experiments from the base model.

```
model.load(model_ref="qwen3-4b-2507:hn-demo", max_context=2048)
```

Verify it loaded:

```
model.status(model_ref="qwen3-4b-2507:hn-demo")
```

Expected: `loaded: true`, `online_training_config.enabled: true`

## Step 2: Configure TTT

Write TTT training parameters to the model's `config.json`:

```
ttt.configure(
  model_ref = "qwen3-4b-2507:hn-demo",
  learning_rate = 0.0003,
  gradient_steps = 3,
  max_grad_norm = 1.0,
  min_input_length = 32,
  max_ttt_context = 512,
  lora_rank = 8,
  lora_alpha = 16,
  target_modules = ["q_proj", "v_proj"],
  auto_reload = true
)
```

Then initialize the LoRA infrastructure (creates the delta pool and optimizer):

```
ttt.init(
  model_ref = "qwen3-4b-2507:hn-demo",
  rank = 8,
  alpha = 16,
  target_modules = ["q_proj", "v_proj"],
  learning_rate = 0.0003
)
```

This creates 72 trainable LoRA modules (36 layers x q_proj + v_proj).

## Step 3: Establish a Baseline

Run inference WITHOUT TTT to show the model's default behavior:

```
infer.generate(
  model_ref = "qwen3-4b-2507:hn-demo",
  prompt = "What is test-time training?",
  max_tokens = 100,
  temperature = 0.3,
  ttt_enabled = false
)
```

**Expected:** Generic, possibly repetitive response with no domain-specific knowledge.

Save this output for comparison.

## Step 4: Train on Domain Text

Feed domain-specific text to the model. TTT uses next-token prediction loss to adapt the LoRA parameters:

```
ttt.trainStream(
  model_ref = "qwen3-4b-2507:hn-demo",
  input = "Test-time training (TTT) is an emerging paradigm in machine learning where model parameters are adapted at inference time using the input itself as a self-supervised signal. Unlike traditional fine-tuning which requires labeled data and offline training, TTT leverages next-token prediction loss on the input context to create temporary parameter adjustments. This approach is particularly effective for domain adaptation scenarios where the test distribution differs significantly from the training distribution. The key insight is that even a few gradient steps on the input context can substantially improve the model's predictions for that specific input, without requiring any labeled examples.",
  gradient_steps = 3,
  auto_commit = true
)
```

**Key metrics to highlight:**

| Metric | What it means | Good values |
|--------|--------------|-------------|
| `grad_norm` | Gradient magnitude (>0 = autograd working) | 0.5 - 5.0 |
| `loss_improvement` | Initial loss minus final loss | > 0.1 |
| `initial_perplexity` | How surprised the model is by the text | Higher = more novel |
| `final_perplexity` | After adaptation | Should be significantly lower |
| `recommendation` | Server's commit recommendation | `true` = beneficial |
| `gradient_clipped` | Whether max_grad_norm was hit | Normal on first steps |

**Expected:** loss_improvement > 0.5, perplexity reduction of 30-50%, recommendation = true.

## Step 5: Verify Adaptation

Run the same inference query again:

```
infer.generate(
  model_ref = "qwen3-4b-2507:hn-demo",
  prompt = "What is test-time training?",
  max_tokens = 100,
  temperature = 0.3,
  ttt_enabled = false
)
```

**Expected:** More coherent, domain-aware response compared to Step 3. The model now has the training text "in its weights" via the LoRA delta.

## Step 6: Train More (Optional)

Add additional domain text. Each training round accumulates into the same delta:

```
ttt.trainStream(
  model_ref = "qwen3-4b-2507:hn-demo",
  input = "Hyprstream implements test-time training using per-tenant LoRA deltas. Each tenant gets an isolated set of low-rank adaptation matrices that are trained using next-token prediction loss on the input context. The system supports both inference-time adaptation and explicit training via the trainStream API.",
  gradient_steps = 3,
  auto_commit = true
)
```

Check accumulated state:

```
ttt.status(model_ref="qwen3-4b-2507:hn-demo")
```

Look for `accumulated_steps` increasing and `delta_norm_ratios` showing non-zero values across all modules.

## Step 7: TTT-Enabled Inference

The most impressive demo — the model adapts to the prompt DURING inference, then generates:

```
infer.generate(
  model_ref = "qwen3-4b-2507:hn-demo",
  prompt = "Explain how hyprstream implements test-time training with per-tenant LoRA deltas.",
  max_tokens = 200,
  temperature = 0.3,
  ttt_enabled = true,
  ttt_gradient_steps = 2,
  auto_commit = true
)
```

The response includes `ttt_metrics` showing the inline adaptation that happened before generation. Highlight the `adaptation_time_ms` — typically 1-2 seconds for 2 gradient steps.

## Step 8: Export the Adapter

Save the accumulated delta as a PEFT-compatible adapter (interoperable with HuggingFace):

```
ttt.export(
  model_ref = "qwen3-4b-2507:hn-demo",
  name = "my-demo-adapter",
  commit_message = "Demo adapter from TTT session"
)
```

This creates `adapters/my-demo-adapter/` with:
- `adapter_config.json` — PEFT metadata (rank, alpha, target modules)
- `adapter_model.safetensors` — trained weights

The adapter is committed to the model's git repository for version control.

## Step 9: Reset and Compare

Reset the delta to show the before/after difference:

```
ttt.reset(model_ref="qwen3-4b-2507:hn-demo")
```

Verify all norms are back to zero:

```
ttt.status(model_ref="qwen3-4b-2507:hn-demo")
```

Run inference again — the response should revert to the generic baseline from Step 3, proving the adaptation was real.

## Step 10: Check Logs (Optional)

View the server-side training logs:

```bash
journalctl --user -u hyprstream-model --since "15 min ago" | grep TTT
```

Key log lines to look for:
- `[TTT] Layer 0 q_proj: correction_norm=X.XX` — shows LoRA corrections growing per step
- `PREFILL: N tokens in Xms [delta-aware]` — confirms delta injection during inference
- `Delta pool initialized: rank=8, alpha=16.0` — confirms LoRA setup

## Talking Points

**For engineers:**
- "Models are git repositories — every adapter export is a versioned commit"
- "Per-tenant isolation means multiple users can train simultaneously without interference"
- "The LoRA rank-8 delta is only 11MB — 0.15% of the base model's parameters"
- "Gradient clipping and AdamW optimizer ensure stable training even with aggressive learning rates"

**For ML practitioners:**
- "TTT uses NTP loss on the INPUT context — no labeled data needed"
- "Perplexity gating automatically adjusts step count based on how novel the input is"
- "The commit/rollback mechanism lets you A/B test adaptations before persisting"
- "Exported adapters are standard PEFT format — load them in HuggingFace transformers"

**For product people:**
- "The model learns from each conversation in real-time"
- "Each customer's adaptations are isolated and versioned"
- "Adaptation adds 1-3 seconds to first response, then inference runs at normal speed"
- "No retraining, no GPU cluster, no data pipeline — just feed it text and it learns"

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `grad_norm = 0` | GradMode leaked from panicked inference | Fixed in v0.3.0 — `tch::with_grad()` wrapper ensures training always has gradients |
| `Token expired` | Default TTL was 5 minutes | Fixed in v0.3.0 — now 48 hours |
| `Input too short` | Prompt < 32 tokens | Use longer input text or lower `min_input_length` |
| `Delta at capacity` | Hit `max_accumulated_steps` (300) | Export/save the adapter, then reset |
| `No pending adaptation` | Auto-rollback after 60 seconds | Use `auto_commit=true` or commit within 60s |
| `HeaderTooLarge` on adapter.load | Known issue in safetensors deserialization | Export works — reload path has a pre-existing bug |
