#![allow(clippy::print_stdout)] // the wire-floor report IS the deliverable

//! Wire-overhead floor (P0.7 → feeds P3.7's latency/throughput budget).
//!
//! The stub is deterministic, so wire sizes are exact and pinnable. This test measures:
//!
//! 1. **JSON facade**: request/response bytes for golden question sets of 1 and 3
//!    questions over real HTTP — envelope overhead and per-question marginal cost.
//! 2. **Arrow/Flight path**: IPC stream bytes for a `DecisionSchema` batch at 1 and 1000
//!    rows — schema overhead and per-row marginal cost.
//!
//! The exact byte counts below are the measured floor: they are *ceilings* for
//! regression purposes (a change that makes the wire fatter fails here) and the baseline
//! P3.7 must beat or explain. Run with `--nocapture` to see the report.

use std::io::Cursor;
use std::net::TcpListener;

use arrow_ipc::writer::StreamWriter;
use hyprstream_decision::answer::VersionTriple;
use hyprstream_decision::arrow::DecisionSchema;
use hyprstream_decision::author;
use hyprstream_decision_stub::mock::{MockDecisionModel, STUB_MODEL_VERSION};

/// Golden facade request bodies (deterministic sizes).
const ONE_QUESTION: &str = r#"{"state": "The refund arrived two weeks late and the box was crushed.", "model": "jev-latest", "questions": {"is_refund": {"type": "noul", "instructions": "The customer wants money back."}}}"#;
const THREE_QUESTIONS: &str = r#"{"state": "The refund arrived two weeks late and the box was crushed.", "model": "jev-latest", "questions": {"is_refund": {"type": "noul", "instructions": "The customer wants money back."}, "tone": {"type": "choice", "criteria": {"angry": "Hostile message", "calm": null, "pleading": "Begging for help"}}, "severity": {"type": "score", "criteria": ["cosmetic", "usable", "unusable"]}}}"#;

/// Measured 2026-09-20 on this stub; regression ceilings for the facade envelope.
const ONE_QUESTION_RESPONSE_CEILING: usize = 150;
const THREE_QUESTION_RESPONSE_CEILING: usize = 800;

#[tokio::test]
async fn facade_wire_floor() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|error| panic!("bind: {error}"));
    let (address, server) = hyprstream_decision_stub::facade::serve(listener)
        .await
        .unwrap_or_else(|error| panic!("serve: {error}"));
    let client = reqwest::Client::new();

    let mut report = Vec::new();
    for (name, body) in [("1-question", ONE_QUESTION), ("3-question", THREE_QUESTIONS)] {
        let response = client
            .post(format!("http://{address}/v1/systemone"))
            .header("authorization", "Bearer wire-floor")
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .unwrap_or_else(|error| panic!("request: {error}"));
        assert_eq!(response.status(), 200);
        let bytes = response.bytes().await.unwrap_or_else(|error| panic!("body: {error}"));
        report.push((name, body.len(), bytes.len()));
    }
    server.abort();

    let (one_req, one_res) = (report[0].1, report[0].2);
    let (three_req, three_res) = (report[1].1, report[1].2);
    println!("WIRE FLOOR — jev-1 JSON facade");
    println!("  1-question:  request {one_req} B, response {one_res} B");
    println!("  3-question:  request {three_req} B, response {three_res} B");
    println!(
        "  marginal per-question: request {} B, response {} B",
        (three_req - one_req) / 2,
        (three_res - one_res) / 2
    );

    assert!(
        one_res <= ONE_QUESTION_RESPONSE_CEILING,
        "1-question response {one_res} B exceeds pinned ceiling {ONE_QUESTION_RESPONSE_CEILING} B"
    );
    assert!(
        three_res <= THREE_QUESTION_RESPONSE_CEILING,
        "3-question response {three_res} B exceeds pinned ceiling {THREE_QUESTION_RESPONSE_CEILING} B"
    );
    assert!(three_res > one_res, "answers grow with questions");
}

/// Arrow path: measured per-row marginal cost is the number P3.7's batch budget uses.
#[test]
fn arrow_wire_floor() {
    let set = author::parse_yaml(
        r#"
state: "The refund arrived two weeks late and the box was crushed."
questions:
  is_refund:
    type: noul
    instructions: "The customer wants money back."
  tone:
    type: choice
    criteria:
      angry: "Hostile message"
      calm: ~
      pleading: "Begging for help"
  severity:
    type: score
    criteria: ["cosmetic", "usable", "unusable"]
"#,
    )
    .unwrap_or_else(|error| panic!("fixture parses: {error}"));
    let schema = DecisionSchema::from_question_set(&set).unwrap_or_else(|error| panic!("schema: {error}"));
    let triple = VersionTriple {
        schema: "qs-v1".to_owned(),
        model: STUB_MODEL_VERSION.to_owned(),
        calib: None,
    };
    let mock = MockDecisionModel;
    let state = "The refund arrived two weeks late and the box was crushed.";

    let encode = |rows: usize| -> usize {
        let answers: Vec<_> = (0..rows)
            .map(|row| mock.answer_row(&set, state, row))
            .collect();
        let batch = schema.build_batch(&triple, &answers).unwrap_or_else(|error| panic!("batch: {error}"));
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema()).unwrap_or_else(|error| panic!("writer: {error}"));
            writer.write(&batch).unwrap_or_else(|error| panic!("write: {error}"));
            writer.finish().unwrap_or_else(|error| panic!("finish: {error}"));
        }
        buffer.into_inner().len()
    };

    let one_row = encode(1);
    let thousand_rows = encode(1000);
    let per_row_milli = (thousand_rows - one_row) * 1000 / 999;
    println!("WIRE FLOOR — Arrow IPC (decision batch, 3 questions: noul+choice(3)+score(3))");
    println!("  1 row:    {one_row} B (schema + batch overhead)");
    println!("  1000 rows: {thousand_rows} B");
    println!("  marginal per row: {}.{:03} B", per_row_milli / 1000, per_row_milli % 1000);

    // Regression ceilings (measured 2026-09-20: 4616 B / 99.235 B per row): if the
    // Arrow contract gets fatter, fail.
    assert!(one_row <= 5000, "1-row batch {one_row} B exceeds pinned ceiling");
    assert!(per_row_milli <= 110_000, "per-row cost {per_row_milli} mB exceeds pinned ceiling");
    assert!(thousand_rows > one_row, "batches grow with rows");
}
