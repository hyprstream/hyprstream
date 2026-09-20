//! The P3.5 `decide` contract, stubbed: Flight SQL prepared statements that pin a
//! question set plus the (schema, model, calib) version triple, bind batches of states,
//! and return `DecisionSchema`-conformant Arrow batches.
//!
//! Contract shape (what P3.5 will implement for real on `hyprstream-flight`):
//!
//! - **Create**: `CreatePreparedStatement` whose query text is a JSON *decide document*:
//!   `{"questions": {…jev-1 question map…}, "schema_version": "…", "calib_version": …,
//!   "model": "…"}`. `schema_version` is required (the triple is pinned at prepare time,
//!   per P3.5); `model` defaults to the stub alias and resolves to the versioned id;
//!   `calib_version` may be null (uncalibrated). The returned handle is the blake3 hash
//!   of the canonical decide document, so identical documents share a statement.
//! - **Parameter schema**: a single non-null `state: utf8` column — `decide(schema,
//!   batch)` where the batch is one state per row.
//! - **Dataset schema**: the P0.1a `DecisionSchema` Arrow schema — per-question
//!   `fixed_size_list<f32>` probabilities with `jev.kind`/`jev.labels` field metadata,
//!   label, reserved conformal-set column, and the version-triple columns.
//! - **Drift fails loudly**: an unknown (or closed) handle is `not_found`; a decide
//!   document whose question ids are not identifier-safe for the columnar surface is
//!   `invalid_argument` at create time (P0.1a's Arrow contract).
//! - **Execute**: bound states are mock-answered row by row and emitted as one
//!   `RecordBatch` via `DecisionSchema::build_batch` — producer tolerance (D5) enforced
//!   by construction.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use arrow_array::{ArrayRef, StringArray};
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::sql::server::{FlightSqlService, PeekableFlightDataStream};
use arrow_flight::sql::{
    ActionClosePreparedStatementRequest, ActionCreatePreparedStatementRequest,
    ActionCreatePreparedStatementResult, CommandPreparedStatementQuery,
    DoPutPreparedStatementResult, ProstMessageExt, SqlInfo,
};
use arrow_flight::utils::flight_data_to_arrow_batch;
use arrow_flight::{
    Action, FlightDescriptor, FlightEndpoint, FlightInfo, IpcMessage, SchemaAsIpc, Ticket,
};
use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use arrow_array::Array as _;
use futures::{StreamExt, TryStreamExt};
use hyprstream_decision::answer::VersionTriple;
use hyprstream_decision::arrow::DecisionSchema;
use hyprstream_decision::author;
use hyprstream_decision::entry::Entry;
use hyprstream_decision::spec::QuestionSet;
use prost::Message;
use tonic::{Request, Response, Status};

use crate::mock::{MockDecisionModel, STUB_MODEL_VERSION};

/// Name of the single parameter column: one state per row of the bound batch.
pub const STATE_PARAMETER: &str = "state";

/// One pinned prepared `decide` statement.
struct PreparedDecide {
    set: QuestionSet,
    decision_schema: DecisionSchema,
    triple: VersionTriple,
    /// States bound via DoPut, one per parameter row.
    states: Vec<String>,
}

/// The stub Flight SQL service. Stateless apart from the prepared-statement table.
#[derive(Default)]
pub struct DecideFlightSqlService {
    statements: Mutex<HashMap<Vec<u8>, PreparedDecide>>,
}

impl DecideFlightSqlService {
    /// A tonic service wrapping this stub, for `Server::builder().add_service(…)`.
    pub fn into_server(
        self,
    ) -> arrow_flight::flight_service_server::FlightServiceServer<Self> {
        arrow_flight::flight_service_server::FlightServiceServer::new(self)
    }

    /// Parse and validate a decide document, returning the pinned statement parts.
    // tonic::Status is the trait-mandated Err; boxing it would fight the Flight SQL API.
    #[allow(clippy::result_large_err)]
    fn prepare(query: &str) -> Result<(QuestionSet, DecisionSchema, VersionTriple), Status> {
        let entry: Entry = serde_json::from_str(query)
            .map_err(|error| Status::invalid_argument(format!("decide document is not valid JSON: {error}")))?;
        let Entry::Map(fields) = &entry else {
            return Err(Status::invalid_argument(
                "decide document must be a JSON object: {questions, schema_version, calib_version?, model?}",
            ));
        };
        let mut questions = None;
        let mut schema_version = None;
        let mut calib_version = None;
        let mut model = None;
        for (key, value) in fields {
            match key.as_str() {
                "questions" => questions = Some(value.clone()),
                "schema_version" => match value {
                    Entry::Str(version) => schema_version = Some(version.clone()),
                    _ => return Err(Status::invalid_argument("`schema_version` must be a string")),
                },
                "calib_version" => match value {
                    Entry::Null => calib_version = Some(None),
                    Entry::Str(version) => calib_version = Some(Some(version.clone())),
                    _ => return Err(Status::invalid_argument("`calib_version` must be a string or null")),
                },
                "model" => match value {
                    Entry::Str(name) => model = Some(name.clone()),
                    _ => return Err(Status::invalid_argument("`model` must be a string")),
                },
                other => {
                    return Err(Status::invalid_argument(format!(
                        "unknown decide-document field `{other}`; expected `questions`, `schema_version`, `calib_version`, `model`"
                    )));
                }
            }
        }
        let questions = questions.ok_or_else(|| {
            Status::invalid_argument("decide document requires a `questions` map")
        })?;
        let schema_version = schema_version.ok_or_else(|| {
            Status::invalid_argument(
                "decide document requires `schema_version`: prepared statements pin the (schema, model, calib) version triple",
            )
        })?;
        let _requested = model; // every alias resolves to the stub's versioned id

        let document = Entry::Map(vec![("questions".to_owned(), questions)]);
        let set = author::parse_json(&document.canonical_text())
            .map_err(|error| Status::invalid_argument(format!("invalid question set: {error}")))?;
        let decision_schema = DecisionSchema::from_question_set(&set).map_err(|error| {
            Status::invalid_argument(format!(
                "question set is not usable on the columnar surface: {error}"
            ))
        })?;
        let triple = VersionTriple {
            schema: schema_version,
            model: STUB_MODEL_VERSION.to_owned(),
            calib: calib_version.flatten(),
        };
        Ok((set, decision_schema, triple))
    }

    /// The `decide` parameter schema: one non-null `state` utf8 column.
    pub fn parameter_schema() -> Schema {
        Schema::new(vec![Field::new(STATE_PARAMETER, DataType::Utf8, false)])
    }

    #[allow(clippy::result_large_err)] // tonic::Status is the trait-mandated Err
    fn statement(&self, handle: &[u8]) -> Result<(), Status> {
        let statements = self.statements.lock();
        if statements.contains_key(handle) {
            Ok(())
        } else {
            Err(Status::not_found(
                "unknown prepared statement handle (stale or drifted; re-prepare)",
            ))
        }
    }
}

/// IPC-encapsulated schema bytes for prepared-statement results.
fn schema_ipc(schema: &Schema) -> bytes::Bytes {
    let IpcMessage(bytes) = SchemaAsIpc::new(schema, &IpcWriteOptions::default())
        .try_into()
        .unwrap_or_else(|error| panic!("schema IPC encoding is infallible for in-memory schemas: {error}"));
    bytes
}

#[tonic::async_trait]
impl FlightSqlService for DecideFlightSqlService {
    type FlightService = Self;

    async fn register_sql_info(&self, _id: i32, _info: &SqlInfo) {}

    async fn do_action_create_prepared_statement(
        &self,
        query: ActionCreatePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<ActionCreatePreparedStatementResult, Status> {
        let (set, decision_schema, triple) = Self::prepare(&query.query)?;
        let handle = blake3::hash(query.query.as_bytes()).as_bytes().to_vec();
        let dataset_schema = decision_schema.arrow_schema();
        let parameter_schema = Self::parameter_schema();
        let prepared = PreparedDecide {
            set,
            decision_schema,
            triple,
            states: Vec::new(),
        };
        self.statements
            .lock()
            .insert(handle.clone(), prepared);
        Ok(ActionCreatePreparedStatementResult {
            prepared_statement_handle: handle.into(),
            dataset_schema: schema_ipc(&dataset_schema),
            parameter_schema: schema_ipc(&parameter_schema),
        })
    }

    async fn do_action_close_prepared_statement(
        &self,
        query: ActionClosePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<(), Status> {
        self.statement(&query.prepared_statement_handle)?;
        self.statements
            .lock()
            .remove(&query.prepared_statement_handle[..]);
        Ok(())
    }

    async fn get_flight_info_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let statements = self.statements.lock();
        let prepared = statements
            .get(&query.prepared_statement_handle[..])
            .ok_or_else(|| {
                Status::not_found("unknown prepared statement handle (stale or drifted; re-prepare)")
            })?;
        let schema = prepared.decision_schema.arrow_schema();
        // The Flight SQL do_get dispatcher routes on the ticket's Any-encoded command,
        // so the ticket carries the handle wrapped in CommandPreparedStatementQuery.
        let command = CommandPreparedStatementQuery {
            prepared_statement_handle: query.prepared_statement_handle.clone(),
        };
        let ticket = Ticket {
            ticket: command.as_any().encode_to_vec().into(),
        };
        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|error| Status::internal(format!("schema encode failed: {error}")))?
            .with_endpoint(FlightEndpoint::new().with_ticket(ticket))
            .with_total_records(-1)
            .with_ordered(false);
        Ok(Response::new(info))
    }

    async fn do_put_prepared_statement_query(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<DoPutPreparedStatementResult, Status> {
        self.statement(&query.prepared_statement_handle)?;
        let parameter_schema = Arc::new(Self::parameter_schema());
        let mut stream = request.into_inner();
        let dictionaries: HashMap<i64, ArrayRef> = HashMap::new();
        let mut states = Vec::new();
        while let Some(data) = stream.next().await {
            let data = data?;
            // The DoPut stream opens with a Schema message; only record-batch messages
            // carry parameter rows.
            let header_type = arrow_ipc::root_as_message(&data.data_header[..])
                .map_err(|error| Status::invalid_argument(format!("parameter message: {error}")))?
                .header_type();
            if header_type != arrow_ipc::MessageHeader::RecordBatch {
                continue;
            }
            let batch = flight_data_to_arrow_batch(&data, parameter_schema.clone(), &dictionaries)
                .map_err(|error| Status::invalid_argument(format!("parameter batch: {error}")))?;
            let column = batch
                .column_by_name(STATE_PARAMETER)
                .and_then(|column| column.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| {
                    Status::invalid_argument("parameter batch must carry a non-null utf8 `state` column")
                })?;
            for row in 0..column.len() {
                if column.is_null(row) {
                    return Err(Status::invalid_argument(format!(
                        "parameter row {row}: state must not be null"
                    )));
                }
                states.push(column.value(row).to_owned());
            }
        }
        let mut statements = self.statements.lock();
        let prepared = statements
            .get_mut(&query.prepared_statement_handle[..])
            .ok_or_else(|| Status::not_found("statement closed while binding"))?;
        prepared.states.extend(states);
        Ok(DoPutPreparedStatementResult {
            prepared_statement_handle: None,
        })
    }

    async fn do_get_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<<Self as arrow_flight::flight_service_server::FlightService>::DoGetStream>, Status> {
        let (set, schema, triple, states) = {
            let statements = self.statements.lock();
            let prepared = statements
                .get(&query.prepared_statement_handle[..])
                .ok_or_else(|| {
                    Status::not_found("unknown prepared statement handle (stale or drifted; re-prepare)")
                })?;
            (
                prepared.set.clone(),
                prepared.decision_schema.clone(),
                prepared.triple.clone(),
                prepared.states.clone(),
            )
        };
        let mock = MockDecisionModel;
        let rows: Vec<_> = states
            .iter()
            .enumerate()
            .map(|(row, state)| mock.answer_row(&set, state, row))
            .collect();
        let batch = schema
            .build_batch(&triple, &rows)
            .map_err(|error| Status::internal(format!("decision batch failed: {error}")))?;
        let schema_ref: SchemaRef = Arc::new(schema.arrow_schema());
        let stream = futures::stream::once(async move { Ok(batch) });
        let encoded = FlightDataEncoderBuilder::new()
            .with_schema(schema_ref)
            .build(stream)
            .map_err(Status::from);
        Ok(Response::new(Box::pin(encoded)))
    }
}

/// Serve the stub Flight SQL endpoint on `listener` in the background.
pub async fn serve(
    listener: std::net::TcpListener,
) -> std::io::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    listener.set_nonblocking(true)?;
    let address = listener.local_addr()?;
    let handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap_or_else(|error| panic!("tokio listener: {error}"));
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        tonic::transport::Server::builder()
            .add_service(DecideFlightSqlService::default().into_server())
            .serve_with_incoming(incoming)
            .await
            .unwrap_or_else(|error| panic!("flight server: {error}"));
    });
    Ok((address, handle))
}

/// Standalone decide-document builder used by tests and the SDK-less contract check.
pub fn decide_document(questions_json: &str, schema_version: &str) -> String {
    format!(r#"{{"schema_version": "{schema_version}", "questions": {questions_json}}}"#)
}
