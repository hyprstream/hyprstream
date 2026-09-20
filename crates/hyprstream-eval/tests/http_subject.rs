//! The HTTP reference client against the P0.7 stub facade: the same wire the
//! adapter-fronted LLMs and Jev speak, proven end-to-end over a loopback
//! socket.

#![allow(clippy::unwrap_used)]

use std::net::SocketAddr;

use hyprstream_decision::author;
use hyprstream_decision::entry::Entry;
use hyprstream_eval::{EvalItem, Harness, HttpSubject, Subject};

fn fixture() -> hyprstream_decision::QuestionSet {
    author::parse_yaml(
        r#"
state: "The box was crushed."
questions:
  is_refund:
    type: noul
    instructions: "The customer wants money back."
  tone:
    type: choice
    criteria:
      angry: "Hostile message"
      calm: ~
  severity:
    type: score
    criteria: ["cosmetic", "usable", "unusable"]
"#,
    )
    .unwrap()
}

async fn serve_stub() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = std::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
    hyprstream_decision_stub::facade::serve(listener).await.unwrap()
}

#[tokio::test]
async fn http_subject_round_trips_against_stub_facade() {
    let (address, server) = serve_stub().await;
    let subject = HttpSubject::new(
        format!("http://{address}"),
        "jev-stub-latest",
        "eval-harness",
    );
    let set = fixture();
    let state = Entry::Str("The box was crushed.".into());
    let row = subject.decide(&set, &state, 0).await.unwrap();

    for question in &set.questions {
        let answer = row
            .answers
            .get(&question.id)
            .and_then(|a| a.value.as_ref())
            .unwrap();
        assert_eq!(answer.probabilities().len(), question.cardinality());
        let sum: f32 = answer.probabilities().iter().sum();
        assert!((sum - 1.0).abs() <= 1e-2, "{}: sum {sum}", question.id);
    }

    // Determinism: the same request twice yields identical distributions.
    let again = subject.decide(&set, &state, 0).await.unwrap();
    assert_eq!(row, again);
    server.abort();
}

#[tokio::test]
async fn http_subject_drives_a_full_bench_slice() {
    let (address, server) = serve_stub().await;
    let subject = HttpSubject::new(format!("http://{address}"), "jev-stub-latest", "t");
    let items: Vec<EvalItem> = hyprstream_bench::generate_all(&hyprstream_bench::BenchConfig {
        seeds_per_stratum: 1,
        seed_base: 0x06,
    })
    .iter()
    .take(12)
    .map(EvalItem::from)
    .collect();
    let output = Harness.run_items(&items, &subject).await.unwrap();
    assert_eq!(output.observations.len(), items.len());
    assert!(
        output
            .observations
            .iter()
            .all(|obs| obs.probabilities.is_some())
    );
    server.abort();
}

#[tokio::test]
async fn http_subject_surfaces_http_errors() {
    let (address, server) = serve_stub().await;
    let subject = HttpSubject::new(format!("http://{address}"), "jev-stub-latest", "");
    let set = fixture();
    // Empty bearer token → the facade 401s; the subject must surface it.
    let result = subject.decide(&set, &Entry::Null, 0).await;
    let error = result.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::Http(_)),
        "expected Http error, got {error:?}"
    );
    server.abort();
}
