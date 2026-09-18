//! # hyprstream-decision — jev-1 decision/question schema IR, authoring, and Arrow emission
//!
//! System One program, DAG node **P0.1a** (Wave 0, critical path). This crate owns the
//! contract half of the decision-model program: the intermediate representation for
//! runtime-declared typed questions, a language-agnostic authoring format (YAML/JSON),
//! the pinned canonical serialization consumed by the question encoder (P1.1), and the
//! Arrow schema emission consumed by the ADBC/Flight SQL operator (P3.5) and the stub
//! service (P0.7). It contains **no model code and no opinions about serving policy** —
//! primitives, not opinions.
//!
//! ## jev-1 profile (clean-room)
//!
//! The question/answer semantics implement the *jev-1 compatibility profile*: a clean-room
//! re-expression of the published wire shape (S6a spike: unauthenticated OpenAPI snapshot
//! `sha256:a191f8a7df6bd6fe…` pinned 2026-09-18, cross-anchored against the MIT SDKs and
//! the MIT adapter — all facts re-expressed here in our own prose). Three question
//! primitives:
//!
//! - **noul** — a statement judged against the state, answered with `P(true) ∈ [0,1]`.
//! - **choice** — one option chosen from 2–255 pre-enumerated options, each with an
//!   optional rubric; answered with a probability per option.
//! - **score** — an ordered rubric of levels (index 0-based); answered with a probability
//!   per level. The probability-weighted value `Σᵢ i·pᵢ` is derivable
//!   ([`confidence::expected_score`]) and is *not* stored — the raw distribution is the
//!   trust contract.
//!
//! Profile v2 question types (`span`, `derived`) are **reserved** in [`QuestionKind`]:
//! the authoring layer recognizes the tags and rejects them with a structured error, so a
//! v2 document fails loudly instead of being silently misread as something else.
//!
//! ## Profile decisions (S6a ambiguity register, the six P0.1a resolutions)
//!
//! The upstream wire format has registered ambiguities (S6a §5). This crate pins six of
//! them as normative profile decisions; each is enforced here:
//!
//! - **D1 (A2) — score requires ≥ 2 levels.** A single-level score is degenerate
//!   (constant value, vacuous confidence). Rejected at authoring.
//! - **D2 (A5) — choice requires ≥ 2 options; ≤ 255 options (A4) is the pinned profile
//!   limit.** Above 255 options the two-stage recipe (score candidates, then choose among
//!   top scorers — a client-side pattern, not a wire feature) is mandatory; see the
//!   serialization budget below.
//! - **D3 (A6) — `null` score levels are allowed and mean "undescribed level"**,
//!   consistent with `null` rubrics on choice options. The level index is its identity.
//! - **D4 (A7) — `instructions` is optional and nullable** on every question type
//!   (the machine-readable schema trumps prose renderings).
//! - **D5 (A8) — probability-sum tolerance:** producers MUST emit distributions with
//!   `|Σp − 1| ≤ 1e-6` ([`confidence::PRODUCER_SUM_TOLERANCE`]); consumers MUST accept
//!   `|Σp − 1| ≤ 1e-2` ([`confidence::CONSUMER_SUM_TOLERANCE`]). Batch construction in
//!   this crate enforces the producer bound.
//! - **D6 (A9) — argmax ties break toward the earliest option in authoring (insertion)
//!   order.** Question-spec order is canonical and is preserved from the authored
//!   document through the IR and into Arrow field metadata.
//!
//! Unresolved-by-design register entries: confidence-formula identity and the 255/10
//! enforcement behavior are live-check items for P0.5/P0.7 (A1/A3/A4); non-422 error body
//! shapes, the `usage` tokenizer identity, and request fan-out limits are profile
//! footnotes (A10–A12) — out of scope for a schema crate.
//!
//! ## Abstention
//!
//! `null` is the abstention value everywhere: a [`answer::QuestionAnswer`] with
//! `value: None` is an *abstained* answer; in Arrow emission the probabilities and label
//! columns for that question are null on that row. Abstention is data, not an error —
//! selective prediction is built on it downstream.
//!
//! ## Version triple
//!
//! Every emitted batch carries the batch-level version triple as three columns:
//! `schema_version` (question-set schema), `model_version` (resolved model id — alias
//! resolution is part of the serving contract), and `calib_version` (nullable; null =
//! uncalibrated raw distribution). Calibration *parameters* are never in the schema —
//! they live as rows in the P0.3 metrics tables.
//!
//! ## Canonical serialization contract (pinned per S6b1)
//!
//! [`serialize`] is the single frozen text form fed to the encoder. Learned special
//! **anchor token placeholder immediately AFTER each option's rubric text** (option →
//! anchor, never a trailing anchor section): marker-adjacent readout is the proven
//! pattern, and our backbone's 18/24 causal linear-attention layers cannot bind a
//! trailing anchor section to its options across 255 positions. Format changes swing
//! frozen-LLM accuracy by up to 76 pp (Sclar et al.), so there is exactly ONE canonical
//! serialization and it does not change without a schema version bump. Question ids are
//! caller-facing only and are never serialized to the model.
//!
//! **Acceptance criterion — token budget:** a serialized request (state + all questions)
//! must fit the **2048-token budget at 2k ctx** ([`serialize::MAX_SERIALIZED_SPEC_TOKENS`]).
//! When a question cannot fit its options within the budget — the canonical case being
//! cardinality above 255 — the two-stage score-then-choice recipe becomes mandatory.
//! Token counting is tokenizer-bound (P1.1); this crate exposes the byte length and the
//! budget constant as the contract surface.
//!
//! ## Confidence-from-peakedness (derived, normative formulas)
//!
//! Raw distributions are primary. Jev-style confidence is *derivable* from a distribution
//! via the normative S6a formulas ([`confidence`]):
//!
//! - choice: `(p_max − 1/n) / (1 − 1/n)` (n = 1 → 1.0)
//! - score: `max(0, 1 − MAD_mode / MAD_uniform)`
//!
//! These exist so ecosystem thresholds port; they are conveniences, not stored contract.
//!
//! ## Arrow emission contract
//!
//! [`arrow::DecisionSchema`] emits, per question, a nullable
//! `fixed_size_list<f32>` **probabilities** column (width = cardinality; noul is the
//! 2-wide `[P(false), P(true)]` distribution), a nullable utf8 **label** column
//! (null = abstained), and — while v1 reserves it — a nullable `list<utf8>`
//! **conformal prediction-set** column that may be entirely unpopulated. The label list
//! for a question lives in the probabilities column's **Arrow field metadata**
//! (`jev.labels`, `jev.kind`) — **never in schema metadata**. Label-set evolution is
//! schema evolution: [`arrow::DecisionSchema::check_evolution`] fails loudly when an
//! existing question's kind or label set changes; evolving a label set means minting a
//! new schema version.

pub mod answer;
pub mod arrow;
pub mod author;
pub mod confidence;
pub mod entry;
pub mod error;
pub mod serialize;
pub mod spec;

pub use answer::{AnswerValue, QuestionAnswer, VersionTriple};
pub use arrow::{
    AnswerRow, ArrowSchemaError, BatchError, DecisionSchema, EvolutionReport, LabelChange,
    LabelEvolutionError,
};
pub use author::{parse_json, parse_yaml, MAX_CHOICE_OPTIONS};
pub use confidence::DistributionError;
pub use entry::Entry;
pub use error::{ErrorPath, SpecError, SpecErrorKind};
pub use spec::{ChoiceOption, NoulCriteria, QuestionBody, QuestionKind, QuestionSet, QuestionSpec};
