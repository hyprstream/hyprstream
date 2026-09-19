//! Arrow-native decision-batch storage (System One P0.3).
//!
//! Decision model outputs are stored as opaque Arrow IPC stream payloads keyed
//! by the producing decision-schema fingerprint, so eval/training reads back
//! the exact `RecordBatch` the model emitted — no row-wise SQL materialization
//! and no lossy re-typing. Each row also carries the `(schema, model, calib)`
//! version triple for joins against `calibration_params`.

// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

use crate::storage::StorageBackend;
use arrow::array::{
    Array, ArrayRef, BinaryArray, Int64Array, LargeBinaryArray, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use std::sync::Arc;
use std::time::SystemTime;
use tonic::Status;

/// Decision batch table name.
pub const DECISION_BATCH_TABLE: &str = "decision_batches";

/// Payload format tag for Arrow IPC stream bytes.
pub const PAYLOAD_FORMAT_IPC_STREAM: &str = "arrow-ipc-stream";

/// Mint a new batch id.
pub fn mint_batch_id() -> String {
    format!("batch-{}", uuid::Uuid::new_v4().simple())
}

fn now_unix_ms() -> i64 {
    let ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    i64::try_from(ms).unwrap_or(i64::MAX)
}

/// Row metadata for one stored decision batch (everything but the payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionBatchMeta {
    /// Unique batch id (see [`mint_batch_id`]).
    pub batch_id: String,
    /// Decision-schema fingerprint of the payload schema (P0.1a IR identity).
    pub schema_fingerprint: String,
    /// Model id/version that produced the batch.
    pub model_id: String,
    /// Calibration version applied at emission; `None` for uncalibrated/raw output.
    pub calib_version: Option<String>,
    /// Creation time, Unix epoch milliseconds.
    pub created_at: i64,
    /// Number of rows in the payload batch.
    pub row_count: i64,
}

/// Arrow schema of the `decision_batches` table.
pub fn decision_batch_schema() -> Schema {
    Schema::new(vec![
        Field::new("batch_id", DataType::Utf8, false),
        Field::new("schema_fingerprint", DataType::Utf8, false),
        Field::new("model_id", DataType::Utf8, false),
        Field::new("calib_version", DataType::Utf8, true),
        Field::new("created_at", DataType::Int64, false),
        Field::new("row_count", DataType::Int64, false),
        Field::new("payload_format", DataType::Utf8, false),
        Field::new("payload", DataType::Binary, false),
    ])
}

/// Serialize a batch as a self-describing Arrow IPC stream.
pub fn encode_batch_ipc(batch: &RecordBatch) -> Result<Vec<u8>, Status> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())
            .map_err(|e| Status::internal(format!("IPC writer init: {e}")))?;
        writer
            .write(batch)
            .map_err(|e| Status::internal(format!("IPC write: {e}")))?;
        writer
            .finish()
            .map_err(|e| Status::internal(format!("IPC finish: {e}")))?;
    }
    Ok(buf)
}

/// Decode an Arrow IPC stream payload back into a single batch
/// (multi-batch streams are concatenated).
pub fn decode_batch_ipc(bytes: &[u8]) -> Result<RecordBatch, Status> {
    let reader = StreamReader::try_new(std::io::Cursor::new(bytes), None)
        .map_err(|e| Status::internal(format!("IPC reader init: {e}")))?;
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|e| Status::internal(format!("IPC read: {e}")))?);
    }
    match batches.len() {
        0 => Err(Status::internal("IPC payload contained no batches")),
        1 => Ok(batches.remove(0)),
        _ => {
            let schema = batches[0].schema();
            arrow::compute::concat_batches(&schema, &batches)
                .map_err(|e| Status::internal(format!("IPC concat: {e}")))
        }
    }
}

fn meta_from_row(batch: &RecordBatch, row: usize) -> Result<DecisionBatchMeta, Status> {
    let string_at = |name: &str| -> Result<String, Status> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .map(|arr| arr.value(row).to_owned())
            .ok_or_else(|| Status::internal(format!("invalid {name} column")))
    };
    let i64_at = |name: &str| -> Result<i64, Status> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .map(|arr| arr.value(row))
            .ok_or_else(|| Status::internal(format!("invalid {name} column")))
    };
    let calib_version = batch
        .column_by_name("calib_version")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .and_then(|arr| {
            if arr.is_null(row) || arr.value(row).is_empty() {
                None
            } else {
                Some(arr.value(row).to_owned())
            }
        });
    Ok(DecisionBatchMeta {
        batch_id: string_at("batch_id")?,
        schema_fingerprint: string_at("schema_fingerprint")?,
        model_id: string_at("model_id")?,
        calib_version,
        created_at: i64_at("created_at")?,
        row_count: i64_at("row_count")?,
    })
}

fn payload_from_row(batch: &RecordBatch, row: usize) -> Result<RecordBatch, Status> {
    let col = batch
        .column_by_name("payload")
        .ok_or_else(|| Status::internal("missing payload column"))?;
    let bytes: Vec<u8> = if let Some(arr) = col.as_any().downcast_ref::<BinaryArray>() {
        arr.value(row).to_vec()
    } else if let Some(arr) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        arr.value(row).to_vec()
    } else {
        return Err(Status::internal(format!(
            "unexpected payload column type: {:?}",
            col.data_type()
        )));
    };
    decode_batch_ipc(&bytes)
}

/// Build the one-row RecordBatch for a decision-batch insert.
fn meta_to_record_batch(
    meta: &DecisionBatchMeta,
    payload_format: &str,
    payload: &[u8],
) -> Result<RecordBatch, Status> {
    let schema = Arc::new(decision_batch_schema());
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![meta.batch_id.as_str()])),
        Arc::new(StringArray::from(vec![meta.schema_fingerprint.as_str()])),
        Arc::new(StringArray::from(vec![meta.model_id.as_str()])),
        Arc::new(StringArray::from(vec![meta.calib_version.as_deref()])),
        Arc::new(Int64Array::from(vec![meta.created_at])),
        Arc::new(Int64Array::from(vec![meta.row_count])),
        Arc::new(StringArray::from(vec![payload_format])),
        Arc::new(BinaryArray::from(vec![payload])),
    ];
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| Status::internal(format!("Failed to create decision batch row: {e}")))
}

fn escape_sql(value: &str) -> String {
    value.replace('\'', "''")
}

/// Decision-batch store over any [`StorageBackend`]: the write/read API for
/// Arrow-native model output (eval harness, distillation corpora, re-fit
/// input).
pub struct DecisionBatchStore<B: StorageBackend> {
    backend: Arc<B>,
}

impl<B: StorageBackend> DecisionBatchStore<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Create the `decision_batches` table.
    pub async fn init(&self) -> Result<(), Status> {
        self.backend
            .create_table(DECISION_BATCH_TABLE, &decision_batch_schema())
            .await
    }

    /// Store one model-emitted batch. Returns the minted batch id.
    pub async fn put(
        &self,
        schema_fingerprint: &str,
        model_id: &str,
        calib_version: Option<&str>,
        batch: &RecordBatch,
    ) -> Result<String, Status> {
        if schema_fingerprint.is_empty() {
            return Err(Status::invalid_argument("schema_fingerprint must not be empty"));
        }
        if model_id.is_empty() {
            return Err(Status::invalid_argument("model_id must not be empty"));
        }
        let meta = DecisionBatchMeta {
            batch_id: mint_batch_id(),
            schema_fingerprint: schema_fingerprint.to_owned(),
            model_id: model_id.to_owned(),
            calib_version: calib_version.map(str::to_owned),
            created_at: now_unix_ms(),
            row_count: i64::try_from(batch.num_rows())
                .map_err(|_| Status::invalid_argument("batch row count overflows i64"))?,
        };
        let payload = encode_batch_ipc(batch)?;
        let row = meta_to_record_batch(&meta, PAYLOAD_FORMAT_IPC_STREAM, &payload)?;
        self.backend.insert_into_table(DECISION_BATCH_TABLE, row).await?;
        Ok(meta.batch_id)
    }

    /// Fetch one batch by id, decoding the payload back to its RecordBatch.
    pub async fn get(
        &self,
        batch_id: &str,
    ) -> Result<Option<(DecisionBatchMeta, RecordBatch)>, Status> {
        let sql = format!(
            "SELECT * FROM {DECISION_BATCH_TABLE} WHERE batch_id = '{}'",
            escape_sql(batch_id)
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let meta = meta_from_row(&batch, 0)?;
        let payload = payload_from_row(&batch, 0)?;
        Ok(Some((meta, payload)))
    }

    /// All batches for a decision-schema fingerprint, oldest first.
    pub async fn fetch_by_schema(
        &self,
        schema_fingerprint: &str,
    ) -> Result<Vec<(DecisionBatchMeta, RecordBatch)>, Status> {
        let sql = format!(
            "SELECT * FROM {DECISION_BATCH_TABLE} WHERE schema_fingerprint = '{}' \
             ORDER BY created_at ASC",
            escape_sql(schema_fingerprint)
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        let mut out = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            out.push((meta_from_row(&batch, row)?, payload_from_row(&batch, row)?));
        }
        Ok(out)
    }

    /// List batch metadata, optionally filtered by schema fingerprint.
    pub async fn list(
        &self,
        schema_fingerprint: Option<&str>,
    ) -> Result<Vec<DecisionBatchMeta>, Status> {
        let mut sql = format!(
            "SELECT batch_id, schema_fingerprint, model_id, calib_version, created_at, row_count \
             FROM {DECISION_BATCH_TABLE}"
        );
        if let Some(fingerprint) = schema_fingerprint {
            sql.push_str(&format!(
                " WHERE schema_fingerprint = '{}'",
                escape_sql(fingerprint)
            ));
        }
        sql.push_str(" ORDER BY created_at ASC");
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        (0..batch.num_rows())
            .map(|row| meta_from_row(&batch, row))
            .collect()
    }
}
