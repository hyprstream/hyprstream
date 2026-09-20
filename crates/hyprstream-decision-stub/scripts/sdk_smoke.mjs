// P0.7 portability proof, JS arm: run the STOCK `@typesafe-ai/sdk` client against the
// stub facade via TYPESAFE_BASE_URL (S6a §4).
//
// Requires: TYPESAFE_BASE_URL (facade address), TYPESAFE_API_KEY (any value).
// Exit 0 on success; assertions throw otherwise.

import assert from "node:assert/strict";
import { TypeSafeClient, UnprocessableEntityError, choice, noul, score } from "@typesafe-ai/sdk";

const STUB_VERSION = "jev-stub-1.0.0";
const client = new TypeSafeClient();

const models = await client.models.list();
assert.ok(models.some((m) => m.name === STUB_VERSION), `versioned id listed: ${models.map((m) => m.name)}`);

const result = await client.systemOne({
  state: "The refund arrived two weeks late and the box was crushed.",
  model: "jev-latest",
  questions: {
    is_refund: noul("The customer wants money back.", { true: "Explicit refund request" }),
    tone: choice("Classify the tone.", { angry: "Hostile message", calm: null, pleading: "Begging" }),
    severity: score("How severe?", ["cosmetic", "usable", "unusable"]),
  },
});

assert.equal(result.model, STUB_VERSION, "alias resolves to the versioned id");

const refund = result.answers.is_refund;
assert.equal(refund.type, "noul");
assert.ok(refund.noul >= 0 && refund.noul <= 1);

const tone = result.answers.tone;
assert.equal(tone.type, "choice");
const toneSum = Object.values(tone.probabilities).reduce((a, b) => a + b, 0);
assert.ok(Math.abs(toneSum - 1) <= 1e-2, `probabilities sum: ${toneSum}`);
const argmax = Object.entries(tone.probabilities).sort((a, b) => b[1] - a[1])[0][0];
assert.equal(tone.choice, argmax, "choice == argmax");
assert.ok(tone.confidence >= 0 && tone.confidence <= 1);

const severity = result.answers.severity;
assert.equal(severity.type, "score");
const expected = Object.entries(severity.probabilities).reduce((acc, [i, p]) => acc + Number(i) * p, 0);
assert.ok(Math.abs(severity.score - expected) < 1e-4, `score == Σ i·pᵢ: ${severity.score} vs ${expected}`);
assert.equal(severity.legend[0], "cosmetic", "legend echoes criteria");
assert.ok(severity.confidence >= 0 && severity.confidence <= 1);

assert.ok(result.usage.input_tokens > 0 && result.usage.output_tokens > 0);

// Server-side 422 surfaces as the SDK's typed error.
await assert.rejects(
  client.systemOne({ state: "x", questions: { bad: { type: "noul", criteria: { yes: "nope" } } } }, { retry: { maxRetries: 0 } }),
  (error) => error instanceof UnprocessableEntityError,
  "invalid noul criteria must raise UnprocessableEntityError",
);

console.log(`JS SDK SMOKE OK against ${client.baseURL} (model=${result.model})`);
