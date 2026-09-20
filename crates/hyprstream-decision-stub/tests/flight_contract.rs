#![allow(clippy::unwrap_used, clippy::expect_used)] // panicking is correct in unit tests
//! Contract test: the P3.5 `decide` prepared-statement flow over a real Flight SQL
//! client (`arrow-flight`'s `FlightSqlServiceClient` — the same shape stock ADBC drivers
//! consume).

use std::net::TcpListener;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StringArray};
use arrow_flight::sql::client::FlightSqlServiceClient;
use arrow_schema::{DataType, Schema};
use futures::TryStreamExt;
use hyprstream_decision::arrow::{
    CALIB_VERSION_COLUMN, FIELD_META_KIND, FIELD_META_LABELS, MODEL_VERSION_COLUMN,
    SCHEMA_VERSION_COLUMN,
};
use hyprstream_decision::confidence;
use hyprstream_decision_stub::flight::{decide_document, serve, STATE_PARAMETER};
use hyprstream_decision_stub::mock::STUB_MODEL_VERSION;
use tonic::transport::Endpoint;

const QUESTIONS: &str = r#"{
    "is_refund": {"type": "noul", "instructions": "The customer wants money back."},
    "tone": {"type": "choice", "criteria": {"angry": "Hostile message", "calm": null, "pleading": "Begging for help"}},
    "severity": {"type": "score", "criteria": ["cosmetic", "usable", "unusable"]}
}"#;

async fn start() -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|error| panic!("bind: {error}"));
    serve(listener).await.unwrap_or_else(|error| panic!("serve: {error}"))
}

async fn client(addr: std::net::SocketAddr) -> FlightSqlServiceClient<tonic::transport::Channel> {
    let endpoint = Endpoint::new(format!("http://{addr}")).unwrap_or_else(|error| panic!("endpoint: {error}"));
    let channel = endpoint.connect().await.unwrap_or_else(|error| panic!("connect: {error}"));
    FlightSqlServiceClient::new(channel)
}

fn parameters(states: &[&str]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(hyprstream_decision_stub::flight::DecideFlightSqlService::parameter_schema()),
        vec![Arc::new(StringArray::from(
            states.iter().map(|s| Some(*s)).collect::<Vec<_>>(),
        ))],
    )
    .unwrap_or_else(|error| panic!("parameter batch: {error}"))
}

#[tokio::test]
async fn decide_prepared_statement_round_trip() {
    let (addr, server) = start().await;
    let mut client = client(addr).await;

    let mut statement = client
        .prepare(decide_document(QUESTIONS, "qs-v1"), None)
        .await
        .unwrap_or_else(|error| panic!("prepare: {error}"));

    // Parameter schema: one non-null utf8 `state` column.
    let parameter_schema = statement.parameter_schema().unwrap_or_else(|error| panic!("parameter schema: {error}"));
    assert_eq!(parameter_schema.fields().len(), 1);
    assert_eq!(parameter_schema.field(0).name(), STATE_PARAMETER);
    assert!(!parameter_schema.field(0).is_nullable());

    // Dataset schema: exactly the P0.1a DecisionSchema contract, labels in field metadata.
    let dataset_schema: Schema = statement.dataset_schema().unwrap_or_else(|error| panic!("dataset schema: {error}")).clone();
    assert!(dataset_schema.metadata().is_empty(), "labels never in schema metadata");
    let probabilities = dataset_schema.field_with_name("tone.probabilities").unwrap_or_else(|error| panic!("column: {error}"));
    assert_eq!(
        probabilities.metadata().get(FIELD_META_KIND).map(String::as_str),
        Some("choice")
    );
    assert_eq!(
        probabilities.metadata().get(FIELD_META_LABELS).map(String::as_str),
        Some(r#"["angry","calm","pleading"]"#)
    );
    match probabilities.data_type() {
        DataType::FixedSizeList(_, width) => assert_eq!(usize::try_from(*width).unwrap_or_else(|error| panic!("width: {error}")), 3),
        other => panic!("tone.probabilities must be fixed_size_list, got {other}"),
    }
    assert!(dataset_schema.field_with_name(SCHEMA_VERSION_COLUMN).is_ok());
    assert!(dataset_schema.field_with_name(MODEL_VERSION_COLUMN).is_ok());
    assert!(dataset_schema.field_with_name(CALIB_VERSION_COLUMN).is_ok());
    assert!(dataset_schema.field_with_name("severity.conformal_set").is_ok());

    // Bind 4 states, execute, validate the batch against DecisionSchema semantics.
    statement
        .set_parameters(parameters(&[
            "The refund arrived two weeks late and the box was crushed.",
            "Great product, works perfectly.",
            "I am furious about the double charge.",
            "Where is my order?",
        ]))
        .unwrap_or_else(|error| panic!("bind parameters: {error}"));
    let info = statement.execute().await.unwrap_or_else(|error| panic!("execute: {error}"));
    let mut batches = Vec::new();
    for endpoint in &info.endpoint {
        let ticket = endpoint.ticket.clone().unwrap_or_else(|| panic!("ticket"));
        let mut stream = client.do_get(ticket).await.unwrap_or_else(|error| panic!("do_get: {error}"));
        while let Some(batch) = stream.try_next().await.unwrap_or_else(|error| panic!("batch: {error}")) {
            batches.push(batch);
        }
    }
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.schema().as_ref(), &dataset_schema, "streamed schema == prepared dataset schema");

    let labels = ["angry", "calm", "pleading"];
    let tone_probs = batch
        .column_by_name("tone.probabilities")
        .unwrap_or_else(|| panic!("column"))
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeListArray>()
        .unwrap_or_else(|| panic!("fixed size list"));
    let tone_labels = batch
        .column_by_name("tone.label")
        .unwrap_or_else(|| panic!("column"))
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| panic!("utf8"));
    for row in 0..4 {
        assert!(tone_probs.is_valid(row));
        let values = tone_probs
            .value(row)
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .unwrap_or_else(|| panic!("f32"))
            .values()
            .to_vec();
        confidence::check_distribution(&values, confidence::CONSUMER_SUM_TOLERANCE)
            .unwrap_or_else(|error| panic!("distribution: {error}"));
        let argmax = confidence::argmax_index(&values).unwrap_or_else(|| panic!("nonempty"));
        assert_eq!(
            tone_labels.value(row),
            labels[argmax],
            "label column is the D6 argmax label"
        );
    }

    // Version triple: schema pinned at prepare, model resolved, calib null.
    let schema_col = batch
        .column_by_name(SCHEMA_VERSION_COLUMN)
        .unwrap_or_else(|| panic!("column"))
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| panic!("utf8"));
    let model_col = batch
        .column_by_name(MODEL_VERSION_COLUMN)
        .unwrap_or_else(|| panic!("column"))
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| panic!("utf8"));
    let calib_col = batch
        .column_by_name(CALIB_VERSION_COLUMN)
        .unwrap_or_else(|| panic!("column"))
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| panic!("utf8"));
    for row in 0..4 {
        assert_eq!(schema_col.value(row), "qs-v1");
        assert_eq!(model_col.value(row), STUB_MODEL_VERSION);
        assert!(calib_col.is_null(row), "uncalibrated triple");
    }

    statement.close().await.unwrap_or_else(|error| panic!("close: {error}"));
    server.abort();
}

#[tokio::test]
async fn unknown_handle_fails_loudly() {
    use arrow_flight::sql::{CommandPreparedStatementQuery, ProstMessageExt};
    use arrow_flight::Ticket;
    use prost::Message;

    let (addr, server) = start().await;
    let mut client = client(addr).await;
    // A handle that was never prepared (stale, drifted, or forged) must not silently
    // resolve — P3.5's drift semantics. Craft the prepared-statement ticket directly.
    let command = CommandPreparedStatementQuery {
        prepared_statement_handle: b"never-prepared".to_vec().into(),
    };
    let ticket = Ticket {
        ticket: command.as_any().encode_to_vec().into(),
    };
    let error = client
        .do_get(ticket)
        .await
        .expect_err("unknown handle must fail");
    let message = format!("{error:?}");
    assert!(
        message.contains("unknown prepared statement handle"),
        "fails loudly: {message}"
    );
    server.abort();
}

#[tokio::test]
async fn invalid_decide_documents_fail_at_create() {
    let (addr, server) = start().await;
    let mut client = client(addr).await;

    // Missing schema_version: the triple must be pinned at prepare time.
    let error = client
        .prepare(r#"{"questions": {"q": {"type": "noul"}}}"#.to_owned(), None)
        .await
        .expect_err("schema_version is required");
    assert!(format!("{error}").contains("schema_version"), "{error}");

    // Question ids that cannot back Arrow columns are rejected at create time.
    let error = client
        .prepare(
            decide_document(r#"{"not a column!": {"type": "noul"}}"#, "qs-v1"),
            None,
        )
        .await
        .expect_err("identifier-unsafe ids are rejected");
    assert!(format!("{error}").contains("identifier-safe"), "{error}");

    // A 256-option choice violates the pinned 255 limit.
    let options = (0..256)
        .map(|i| format!(r#""opt{i}": null"#))
        .collect::<Vec<_>>()
        .join(",");
    let error = client
        .prepare(
            decide_document(&format!(r#"{{"q": {{"type": "choice", "criteria": {{{options}}}}}}}"#), "qs-v1"),
            None,
        )
        .await
        .expect_err("256 options rejected");
    assert!(format!("{error}").contains("255"), "{error}");
    server.abort();
}

#[tokio::test]
async fn same_document_shares_handle_and_is_deterministic() {
    let (addr, server) = start().await;
    let mut client = client(addr).await;
    let mut first = client
        .prepare(decide_document(QUESTIONS, "qs-v1"), None)
        .await
        .unwrap_or_else(|error| panic!("prepare 1: {error}"));
    first
        .set_parameters(parameters(&["same state", "same state"]))
        .unwrap_or_else(|error| panic!("bind: {error}"));
    let info = first.execute().await.unwrap_or_else(|error| panic!("execute: {error}"));
    let ticket = info.endpoint[0].ticket.clone().unwrap_or_else(|| panic!("ticket"));
    let batches: Vec<RecordBatch> = client
        .do_get(ticket)
        .await
        .unwrap_or_else(|error| panic!("do_get: {error}"))
        .try_collect()
        .await
        .unwrap_or_else(|error| panic!("collect: {error}"));

    let mut second = client
        .prepare(decide_document(QUESTIONS, "qs-v1"), None)
        .await
        .unwrap_or_else(|error| panic!("prepare 2: {error}"));
    second
        .set_parameters(parameters(&["same state", "same state"]))
        .unwrap_or_else(|error| panic!("bind: {error}"));
    let info = second.execute().await.unwrap_or_else(|error| panic!("execute: {error}"));
    let ticket = info.endpoint[0].ticket.clone().unwrap_or_else(|| panic!("ticket"));
    let batches2: Vec<RecordBatch> = client
        .do_get(ticket)
        .await
        .unwrap_or_else(|error| panic!("do_get: {error}"))
        .try_collect()
        .await
        .unwrap_or_else(|error| panic!("collect: {error}"));
    assert_eq!(batches, batches2, "the stub is deterministic on the wire");
    server.abort();
}
