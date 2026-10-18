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
    hyprstream_decision_stub::facade::serve(listener)
        .await
        .unwrap()
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
    assert!(output
        .observations
        .iter()
        .all(|obs| obs.probabilities.is_some()));
    server.abort();
}

#[tokio::test]
async fn http_subject_surfaces_http_errors() {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    // Valid local configuration can still receive a provider auth rejection.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 8192];
        let mut filled = 0;
        while !request[..filled]
            .windows(4)
            .any(|bytes| bytes == b"\r\n\r\n")
        {
            assert!(
                filled < request.len(),
                "provider fixture request header too large"
            );
            let count = socket.read(&mut request[filled..]).await.unwrap();
            assert!(count > 0, "provider fixture request ended before headers");
            filled += count;
        }
        socket
            .write_all(
                b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
            )
            .await
            .unwrap();
    });
    let subject = HttpSubject::new(
        format!("http://{address}"),
        "jev-stub-latest",
        "provider-rejected-token",
    );
    let set = fixture();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        subject.decide(&set, &Entry::Str("valid state".into()), 0),
    )
    .await
    .unwrap();
    let error = result.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::Http(ref message) if message.contains("status 401")),
        "expected provider401 Http error, got {error:?}"
    );
    server.await.unwrap();
}

#[tokio::test]
async fn http_subject_records_the_resolved_model_id() {
    let (address, server) = serve_stub().await;
    let subject = HttpSubject::new(
        format!("http://{address}"),
        "jev-stub-latest",
        "eval-harness",
    );
    // Before the first call the requested string is the best known id.
    assert_eq!(subject.requested_model(), "jev-stub-latest");
    assert_eq!(subject.resolved_model_id(), "jev-stub-latest");
    subject
        .decide(&fixture(), &Entry::Str("The box was crushed.".into()), 0)
        .await
        .unwrap();
    // The alias resolved to the stub's versioned id, and that is what runs
    // must record.
    assert_eq!(subject.requested_model(), "jev-stub-latest");
    assert_eq!(
        subject.resolved_model_id(),
        hyprstream_decision_stub::mock::STUB_MODEL_VERSION
    );
    server.abort();
}

#[tokio::test]
async fn decide_with_version_binds_the_version_to_its_own_response() {
    // Runs stamp one version on every row; the version must come from the
    // same exchange as the answers, not from a shared mutable slot a
    // concurrent run could overwrite in between.
    let (address, server) = serve_stub().await;
    let subject = HttpSubject::new(
        format!("http://{address}"),
        "jev-stub-latest",
        "eval-harness",
    );
    let (answers, resolved) = subject
        .decide_with_version(&fixture(), &Entry::Str("The box was crushed.".into()), 0)
        .await
        .unwrap();
    assert_eq!(resolved, hyprstream_decision_stub::mock::STUB_MODEL_VERSION);
    assert_eq!(answers.answers.len(), fixture().questions.len());
    server.abort();
}

#[tokio::test]
async fn wire_preserves_declared_option_order_end_to_end() {
    // The mock's distributions are a hash of the canonical question
    // serialization, which is option-order sensitive. If the harness's
    // request body reordered options (e.g. serde_json's sorted map keys),
    // the stub would hash a different serialization than the declared set
    // and the distributions would diverge from the in-process reference.
    let (address, server) = serve_stub().await;
    let subject = HttpSubject::new(format!("http://{address}"), "jev-stub-latest", "t");
    let set = fixture();
    let state = Entry::Str("The box was crushed.".into());
    let row = subject.decide(&set, &state, 0).await.unwrap();
    let reference = hyprstream_decision_stub::mock::MockDecisionModel.answer_row(
        &set,
        "The box was crushed.",
        0,
    );
    assert_eq!(
        row, reference,
        "the wire must carry the declared option order byte-for-byte"
    );

    // Control: the same questions with permuted option order hash
    // differently — order genuinely flows through the wire.
    let permuted = author::parse_yaml(
        r#"
state: "The box was crushed."
questions:
  is_refund:
    type: noul
    instructions: "The customer wants money back."
  tone:
    type: choice
    criteria:
      calm: ~
      angry: "Hostile message"
  severity:
    type: score
    criteria: ["unusable", "usable", "cosmetic"]
"#,
    )
    .unwrap();
    let permuted_reference = hyprstream_decision_stub::mock::MockDecisionModel.answer_row(
        &permuted,
        "The box was crushed.",
        0,
    );
    assert_ne!(
        reference.answers.get("tone").and_then(|a| a.value.as_ref()),
        permuted_reference
            .answers
            .get("tone")
            .and_then(|a| a.value.as_ref()),
        "option order must change the mock's distribution"
    );
    server.abort();
}

#[tokio::test]
async fn teacher_answers_record_the_resolved_model_id() {
    let (address, server) = serve_stub().await;
    let ensemble = hyprstream_eval::TeacherEnsemble::new(vec![hyprstream_eval::Teacher {
        id: "stub-teacher".into(),
        tos_class: hyprstream_eval::TosClass::Distributable,
        subject: Box::new(HttpSubject::new(
            format!("http://{address}"),
            "jev-stub-latest",
            "t",
        )),
    }])
    .unwrap();
    let out = ensemble
        .decide(
            &fixture(),
            &Entry::Str("The box was crushed.".into()),
            0,
            "item",
        )
        .await
        .unwrap();
    assert_eq!(
        out.teacher_answers[0].model_id,
        hyprstream_decision_stub::mock::STUB_MODEL_VERSION,
        "TeacherAnswer.model_id is the resolved version, not the alias"
    );
    server.abort();
}

#[test]
fn http_subject_debug_redacts_the_bearer_token() {
    let subject = HttpSubject::new("http://example.invalid", "m", "super-secret-token");
    let debug = format!("{subject:?}");
    assert!(
        !debug.contains("super-secret-token"),
        "Debug must never print the bearer token: {debug}"
    );
    assert!(debug.contains("<redacted>"));
}

#[tokio::test]
async fn http_subject_rejects_a_response_without_the_resolved_model() {
    // The serving contract requires `response.model` to name the resolved
    // version; accepting its absence would attribute answers to the
    // requested alias or a version cached from an unrelated earlier
    // response.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 65536];
            let _ = socket.read(&mut buffer).await;
            let body = r#"{"answers":{"is_refund":{"type":"noul","noul":0.5},"tone":{"type":"choice","probabilities":{"angry":0.5,"calm":0.5}},"severity":{"type":"score","probabilities":{"0":0.34,"1":0.33,"2":0.33}}}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });
    let subject = HttpSubject::new(format!("http://{address}"), "some-model", "t");
    let error = subject
        .decide_with_version(&fixture(), &Entry::Str("The box was crushed.".into()), 0)
        .await
        .err()
        .unwrap();
    match error {
        hyprstream_eval::EvalError::Http(message) => {
            assert!(
                message.contains("model"),
                "the error must name the missing field: {message}"
            );
        }
        other => panic!("a model-less response must be an Http error, got {other:?}"),
    }
    server.abort();
}

#[tokio::test]
async fn http_subject_rejects_an_empty_resolved_model() {
    // `"model":""` is the same contract violation as a missing field:
    // accepting it would produce runs and Arrow version triples with no
    // model provenance (set runs only fail later, in BatchSink::persist_run).
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 65536];
            let _ = socket.read(&mut buffer).await;
            let body = r#"{"model":"","answers":{"is_refund":{"type":"noul","noul":0.5},"tone":{"type":"choice","probabilities":{"angry":0.5,"calm":0.5}},"severity":{"type":"score","probabilities":{"0":0.34,"1":0.33,"2":0.33}}}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });
    let subject = HttpSubject::new(format!("http://{address}"), "some-model", "t");
    let error = subject
        .decide_with_version(&fixture(), &Entry::Str("The box was crushed.".into()), 0)
        .await
        .err()
        .unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::Http(_)),
        "an empty model must be an Http error, got {error:?}"
    );
    server.abort();
}

#[tokio::test]
async fn rejected_http_answers_never_change_the_resolved_model_cache() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
        for (model, valid) in [
            ("accepted-1", true),
            ("rejected", false),
            ("accepted-2", true),
        ] {
            let (socket, _) = listener.accept().await.unwrap();
            let mut reader = tokio::io::BufReader::new(socket);
            let mut content_length = None;
            loop {
                let mut line = String::new();
                assert!(reader.read_line(&mut line).await.unwrap() > 0);
                if line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") {
                        content_length = Some(value.trim().parse::<usize>().unwrap());
                    }
                }
            }
            let length = content_length.unwrap();
            assert!(length <= 65536);
            let mut request = vec![0; length];
            reader.read_exact(&mut request).await.unwrap();
            let request: serde_json::Value = serde_json::from_slice(&request).unwrap();
            assert_eq!(request["model"], "requested");
            let mut socket = reader.into_inner();
            let answers = if valid {
                serde_json::json!({"q": {"type":"noul", "noul":0.5}})
            } else {
                serde_json::json!({})
            };
            let body = serde_json::json!({"model":model, "answers":answers}).to_string();
            socket.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", body.len(), body).as_bytes()).await.unwrap();
        }
    });
    let subject = HttpSubject::new(format!("http://{address}"), "requested", "t");
    let set =
        hyprstream_decision::author::parse_yaml("questions:\n  q:\n    type: noul\n").unwrap();
    subject
        .decide_with_version(&set, &Entry::Str("valid state".into()), 0)
        .await
        .unwrap();
    assert_eq!(subject.resolved_model_id(), "accepted-1");
    assert!(subject
        .decide_with_version(&set, &Entry::Str("valid state".into()), 1)
        .await
        .is_err());
    assert_eq!(
        subject.resolved_model_id(),
        "accepted-1",
        "rejected exchange cannot poison provenance"
    );
    subject
        .decide_with_version(&set, &Entry::Str("valid state".into()), 2)
        .await
        .unwrap();
    assert_eq!(subject.resolved_model_id(), "accepted-2");
    server.await.unwrap();
}

#[tokio::test]
async fn legacy_http_null_states_fail_before_any_request_or_teacher_query() {
    use hyprstream_eval::{
        EvalError, EvalRow, EvalSet, Teacher, TeacherEnsemble, TosClass, TruthSubject,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::Duration;

    struct CountNative(Arc<AtomicUsize>);
    #[async_trait::async_trait]
    impl Subject for CountNative {
        fn model_id(&self) -> &str {
            "native-null-control"
        }
        async fn decide(
            &self,
            set: &hyprstream_decision::QuestionSet,
            state: &Entry,
            row: usize,
        ) -> Result<hyprstream_decision::arrow::AnswerRow, EvalError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            TruthSubject::new(self.model_id())
                .decide(set, state, row)
                .await
        }
    }
    // A real listening socket records any attempted connection. A regression
    // cannot hang waiting for a response: each run has a bounded timeout.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let http = HttpSubject::new(&endpoint, "alias", "token");
    let question = fixture().questions.remove(0);
    let item = EvalItem {
        id: "first".into(),
        family: None,
        stratum: None,
        group: None,
        state: Entry::Str("valid state".into()),
        question,
        truth: None,
    };
    let null_item = EvalItem {
        id: "second".into(),
        state: Entry::Null,
        ..item.clone()
    };
    let items = vec![item.clone(), null_item.clone()];
    let result = tokio::time::timeout(Duration::from_secs(1), Harness.run_items(&items, &http))
        .await
        .unwrap();
    assert!(matches!(result, Err(EvalError::InvalidInput(_))));
    let row = EvalRow {
        id: "first".into(),
        family: None,
        stratum: None,
        group: None,
        state: item.state.clone(),
        truth: std::collections::BTreeMap::new(),
    };
    let set = EvalSet {
        name: "null-transport".into(),
        schema_version: "v1".into(),
        questions: vec![item.question.clone()],
        rows: vec![
            row.clone(),
            EvalRow {
                id: "second".into(),
                state: Entry::Null,
                ..row
            },
        ],
    };
    let result = tokio::time::timeout(Duration::from_secs(1), Harness.run_set(&set, &http))
        .await
        .unwrap();
    assert!(matches!(result, Err(EvalError::InvalidInput(_))));
    let question_set = set.question_set();
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        http.decide(&question_set, &Entry::Null, 0),
    )
    .await
    .unwrap();
    assert!(matches!(result, Err(EvalError::InvalidInput(_))));

    let calls = Arc::new(AtomicUsize::new(0));
    let ensemble = TeacherEnsemble::new(vec![
        Teacher {
            id: "native-first".into(),
            tos_class: TosClass::Unknown,
            subject: Box::new(CountNative(calls.clone())),
        },
        Teacher {
            id: "http-second".into(),
            tos_class: TosClass::Unknown,
            subject: Box::new(HttpSubject::new(&endpoint, "alias", "token")),
        },
    ])
    .unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        ensemble.decide(&question_set, &Entry::Null, 0, "null"),
    )
    .await
    .unwrap();
    assert!(matches!(result, Err(EvalError::InvalidInput(_))));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "preflight all teachers before querying any"
    );
    assert!(
        matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
        "no rejected run may open even one HTTP connection"
    );

    // The default hook preserves explicit null for native/in-process subjects.
    let native = CountNative(calls.clone());
    let output = Harness.run_items(&[null_item], &native).await.unwrap();
    assert_eq!(output.observations.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let local_set = EvalSet {
        rows: vec![EvalRow {
            id: "null".into(),
            family: None,
            stratum: None,
            group: None,
            state: Entry::Null,
            truth: std::collections::BTreeMap::new(),
        }],
        ..set
    };
    assert_eq!(
        Harness
            .run_set(&local_set, &native)
            .await
            .unwrap()
            .observations
            .len(),
        1
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn invalid_http_configuration_preflights_all_teachers_before_any_query() {
    use hyprstream_eval::{EvalError, Teacher, TeacherEnsemble, TosClass, TruthSubject};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::Duration;

    struct CountNative(Arc<AtomicUsize>);
    #[async_trait::async_trait]
    impl Subject for CountNative {
        fn model_id(&self) -> &str {
            "native-configuration-control"
        }
        async fn decide(
            &self,
            set: &hyprstream_decision::QuestionSet,
            state: &Entry,
            row: usize,
        ) -> Result<hyprstream_decision::arrow::AnswerRow, EvalError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            TruthSubject::new(self.model_id())
                .decide(set, state, row)
                .await
        }
    }
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let set = fixture();
    let state = Entry::Str("valid non-null state".into());
    for base_url in [endpoint.as_str(), "https://example.invalid"] {
        assert!(HttpSubject::new(base_url, "alias", "valid-token")
            .preflight(&set, &state)
            .is_ok());
    }
    let unsupported = endpoint.replacen("http://", "ftp://", 1);
    for (base_url, model, token) in [
        (endpoint.as_str(), "", "token"),
        (endpoint.as_str(), "alias", ""),
        (endpoint.as_str(), "alias", " \t "),
        ("", "alias", "token"),
        (":invalid-url", "alias", "token"),
        (unsupported.as_str(), "alias", "token"),
        (endpoint.as_str(), "alias", "secret\r\ninvalid-token"),
        (endpoint.as_str(), "alias", "secret\0invalid-token"),
    ] {
        let http = HttpSubject::new(base_url, model, token);
        let direct = tokio::time::timeout(Duration::from_secs(1), http.decide(&set, &state, 0))
            .await
            .unwrap();
        let error = direct.unwrap_err();
        assert!(matches!(error, EvalError::InvalidInput(_)));
        assert!(
            !error.to_string().contains("secret"),
            "preflight must redact credential errors"
        );
        let item = EvalItem {
            id: "configuration".into(),
            question: set.questions[0].clone(),
            state: state.clone(),
            family: None,
            stratum: None,
            group: None,
            truth: None,
        };
        let items = tokio::time::timeout(Duration::from_secs(1), Harness.run_items(&[item], &http))
            .await
            .unwrap();
        assert!(matches!(items, Err(EvalError::InvalidInput(_))));
        let calls = Arc::new(AtomicUsize::new(0));
        let teachers = TeacherEnsemble::new(vec![
            Teacher {
                id: "native-first".into(),
                tos_class: TosClass::Unknown,
                subject: Box::new(CountNative(calls.clone())),
            },
            Teacher {
                id: "http-later".into(),
                tos_class: TosClass::Unknown,
                subject: Box::new(HttpSubject::new(base_url, model, token)),
            },
        ])
        .unwrap();
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            teachers.decide(&set, &state, 0, "configuration"),
        )
        .await
        .unwrap();
        assert!(matches!(result, Err(EvalError::InvalidInput(_))));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "later deterministic error must not bill earlier teacher"
        );
        assert!(
            matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
        );
    }
}
