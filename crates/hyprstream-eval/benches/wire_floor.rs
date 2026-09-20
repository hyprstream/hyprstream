//! Wire-floor bench (P0.7 floor, feeds P3.7): the irreducible jev-1 wire cost.
//!
//! Measures, as a function of question count and cardinality:
//! - request serialization bytes (`{state, model, questions}` envelope),
//! - response bytes for the same envelope rendered from deterministic
//!   distributions (the P0.7 mock answers, so sizes are byte-reproducible),
//! - encode/decode CPU per request.
//!
//! These floors are the "you cannot do better than this" numbers P3.7's
//! latency budget subtracts before blaming serving overhead. Run:
//! `cargo bench -p hyprstream-eval --bench wire_floor`.

#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hyprstream_decision::author;
use hyprstream_decision::entry::Entry;
use hyprstream_decision_stub::mock::MockDecisionModel;
use hyprstream_decision_stub::render;
use hyprstream_eval::subject::request_to_json;

/// A question set of `n` questions cycling through the three primitives, with
/// choice cardinality `k`.
fn question_set(n: usize, k: usize) -> hyprstream_decision::QuestionSet {
    let mut doc = String::from("state: \"The crate arrived damaged.\"\nquestions:\n");
    for i in 0..n {
        match i % 3 {
            0 => {
                doc.push_str(&format!("  q{i}:\n    type: noul\n    instructions: \"Is a refund owed for case {i}?\"\n"));
            }
            1 => {
                doc.push_str(&format!("  q{i}:\n    type: choice\n    instructions: \"Pick the disposition for case {i}.\"\n    criteria:\n"));
                for j in 0..k {
                    doc.push_str(&format!("      opt{j}: \"Disposition option number {j}\"\n"));
                }
            }
            _ => {
                doc.push_str(&format!("  q{i}:\n    type: score\n    instructions: \"Rate the damage severity for case {i}.\"\n    criteria:\n"));
                for j in 0..k {
                    doc.push_str(&format!("      - \"Severity level {j} of the rubric\"\n"));
                }
            }
        }
    }
    author::parse_yaml(&doc).unwrap_or_else(|error| panic!("generated doc parses: {error}"))
}

fn wire_bytes(set: &hyprstream_decision::QuestionSet) -> (usize, usize) {
    let request = request_to_json(
        set,
        &Entry::Str("The crate arrived damaged.".into()),
        "jev-stub-latest",
    );
    let request_bytes = serde_json::to_vec(&request).unwrap_or_else(|e| panic!("json: {e}")).len();
    let row = MockDecisionModel.answer_row(set, "The crate arrived damaged.", 0);
    let response = serde_json::json!({
        "model": "jev-stub-1.0.0",
        "answers": render::render_answers(set, &row),
        "usage": {"input_tokens": 0, "output_tokens": 0},
    });
    let response_bytes = serde_json::to_vec(&response).unwrap_or_else(|e| panic!("json: {e}")).len();
    (request_bytes, response_bytes)
}

fn bench_wire_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("wire_floor");
    for (questions, cardinality) in [(1, 4), (5, 4), (10, 8), (25, 8)] {
        let set = question_set(questions, cardinality);
        // The byte floors themselves, as a benchmarked value (the numbers P3.7
        // subtracts before blaming serving overhead).
        group.bench_with_input(
            BenchmarkId::new("wire_bytes", format!("q{questions}-k{cardinality}")),
            &set,
            |b, set| b.iter(|| black_box(wire_bytes(set))),
        );
        group.bench_with_input(
            BenchmarkId::new("request_encode", format!("q{questions}-k{cardinality}")),
            &set,
            |b, set| {
                b.iter(|| {
                    black_box(serde_json::to_vec(&request_to_json(
                        set,
                        &Entry::Str("The crate arrived damaged.".into()),
                        "jev-stub-latest",
                    )))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("response_render", format!("q{questions}-k{cardinality}")),
            &set,
            |b, set| {
                b.iter(|| {
                    let row = MockDecisionModel.answer_row(set, "The crate arrived damaged.", 0);
                    black_box(serde_json::to_vec(&render::render_answers(set, &row)))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("request_parse", format!("q{questions}-k{cardinality}")),
            &set,
            |b, set| {
                let body = serde_json::to_vec(&request_to_json(
                    set,
                    &Entry::Str("The crate arrived damaged.".into()),
                    "jev-stub-latest",
                ))
                .unwrap_or_else(|e| panic!("json: {e}"));
                b.iter(|| black_box(hyprstream_decision_stub::wire::parse_request(&body)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_wire_sizes);
criterion_main!(benches);
