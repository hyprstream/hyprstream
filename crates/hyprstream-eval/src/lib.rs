//! # hyprstream-eval — workflow eval harness (System One DAG node P0.4)
//!
//! The program's falsifiability surface: a public harness that runs **any
//! jev-1-shaped subject** — our model, frontier LLMs through the typed adapter,
//! or Jev itself — over declared question sets and verifiable-outcome items,
//! then scores the results under the single pinned measurement protocol
//! ([`hyprstream_calibration`], P0.2) so numbers are comparable across arms,
//! splits, and releases.
//!
//! ## Pieces
//!
//! - **Subjects** ([`subject`]): the [`Subject`] trait is the one seam every
//!   arm implements. [`subject::HttpSubject`] speaks the jev-1 `POST
//!   /v1/systemone` wire — point it at the P0.7 stub, at the TypeSafe adapter
//!   fronting a frontier LLM, or at Jev; the harness cannot tell the
//!   difference. [`subject::HashSubject`] is a deterministic in-process
//!   reference subject (hash-drawn distributions) for offline development and
//!   the baseline benches; [`subject::TruthSubject`] answers one-hot at the
//!   item's verifiable truth and anchors the scoring sanity tests.
//! - **Teacher ensemble** ([`teacher`]): a roster of [`teacher::Teacher`]
//!   subjects, each carrying its **ToS class** ([`teacher::TosClass`]) — the
//!   provenance the P1.3 distributability flags and DISCLOSURE pins are built
//!   from. The ensemble persists **raw per-teacher probability vectors**
//!   (P0.5's corrections are fit later and applied at training time), reports
//!   the corrected-ensemble stand-in (the plain average — corrections land in
//!   P0.5), and measures argmax **agreement** against it.
//! - **Runs** ([`run`]): an [`run::EvalSet`] is one question set applied to
//!   many states — the workflow shape. [`run::Harness`] executes a set (or
//!   loose one-question [`run::EvalItem`]s, which is what P0.6 bench items
//!   convert into) against a subject, producing a [`run::RunOutput`]: the raw
//!   [`Observation`](run::Observation) stream plus, for question-set runs, the
//!   `hyprstream-decision` Arrow batch.
//! - **Scoring** ([`score`]): per-field calibration (ECE/Brier/NLL; SCE/ACE +
//!   RPS for the ordinal `score` primitive), the protocol macro rule with
//!   bootstrap CIs ([`hyprstream_calibration::GateReport`]), per-family and
//!   per-stratum breakdowns, abstention rate, and the **option-order flip
//!   rate** over cyclic-permutation groups (S6b1 — transfer that reorders
//!   answers under permutation is not transfer).
//! - **Persistence** ([`persist`]): Arrow-native batches into the P0.3
//!   [`hyprstream_metrics_api::DecisionBatchStore`] over any
//!   [`hyprstream_metrics_api::StorageBackend`] — the harness links only the
//!   Apache API crate; the engine lives in the serving process.
//!
//! Every published number from this harness ships with the pre-committed
//! disclosure language ([`hyprstream_calibrated disclosure`]):
//! calibration-to-teacher ≠ calibration-to-reality.
//!
//! [`hyprstream_calibrated disclosure`]: hyprstream_calibration::disclosure

pub mod error;
pub mod persist;
pub mod run;
pub mod score;
pub mod subject;
pub mod teacher;

pub use error::EvalError;
pub use persist::BatchSink;
pub use run::{EvalItem, EvalRow, EvalSet, Harness, Observation, RunOutput};
pub use score::{
    agreement_summary, flip_rate_with_labels, score_bench_run, score_run, BreakdownMetrics,
    EnsembleAgreement, FlipRate, ItemScore, ScoreConfig, ScoreReport,
};
pub use subject::{HashSubject, HttpSubject, Subject, TruthSubject};
pub use teacher::{EnsembleOutput, Teacher, TeacherAnswer, TeacherEnsemble, TosClass};
