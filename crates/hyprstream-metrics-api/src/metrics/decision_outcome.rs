//! Durable decision-row and realized-outcome storage (System One R6).
//!
//! The P0.3 [`DecisionBatchStore`](super::decision_batch::DecisionBatchStore)
//! keeps model output losslessly but is anonymous: no tenant, no
//! encoding/quantization/compute pins, no per-row identity, and no link from
//! a stored distribution to the realized outcome later observed for it. This
//! module adds the durable append-only path on the same
//! [`StorageBackend`](crate::storage::StorageBackend) seam:
//!
//! - `decision_rows` — one immutable row per stored decision batch. The
//!   payload is a self-describing Arrow IPC stream (same lossless encoding
//!   as `decision_batches`) that additionally carries a `decision_row_id`
//!   column: one stable id per input row, minted by the store or preserved
//!   verbatim from the caller, so outcomes can link the EXACT row forever.
//! - `decision_outcomes` — one immutable row per realized outcome, linking
//!   `(tenant, decision_id, decision_row_id, question_id)` to the observed
//!   label/value with provenance. Corrections are append-only via
//!   `supersedes_outcome_id`; nothing is ever updated in place.
//!
//! Trust model: every operation takes a [`TrustedTenant`] constructed by the
//! embedding service AFTER it verified the caller (verified envelope/session
//! context). This library validates the tenant is a bounded domain and
//! filters every read/write/join by it; the authentication decision itself
//! belongs to the embedding service and is never inferred from stored data.

// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

use crate::storage::StorageBackend;
use arrow::array::FixedSizeListArray;
use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use hyprstream_decision::confidence::{check_distribution, CONSUMER_SUM_TOLERANCE};
use hyprstream_decision::spec::QuestionKind;
use std::sync::Arc;
use std::time::SystemTime;
use tonic::Status;

use super::decision_batch::{
    decode_batch_ipc_bounded, encode_batch_ipc, PAYLOAD_FORMAT_IPC_STREAM,
};

/// Decision-row table name (append-only; one row per stored decision batch).
pub const DECISION_ROWS_TABLE: &str = "decision_rows";

/// Realized-outcome table name (append-only; one row per observed outcome).
pub const DECISION_OUTCOMES_TABLE: &str = "decision_outcomes";

/// Storage schema version of the two tables this module owns. Bump (and
/// migrate via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, the metrics-v2
/// pattern) when adding columns; never rewrite existing rows.
pub const DECISION_OUTCOME_SCHEMA_VERSION: u32 = 1;

/// Payload column carrying the stable per-input-row decision id.
pub const DECISION_ROW_ID_COLUMN: &str = "decision_row_id";

/// Field-metadata key carrying the question kind — the canonical constant
/// from the jev Arrow emission contract (`hyprstream-decision::arrow`).
pub use hyprstream_decision::arrow::FIELD_META_KIND;
/// Field-metadata key carrying the declared label list (canonical order).
pub use hyprstream_decision::arrow::FIELD_META_LABELS;

/// Suffix of the per-question probabilities column (`{qid}.probabilities`).
pub const PROBABILITIES_COLUMN_SUFFIX: &str = ".probabilities";

/// Maximum input rows carried by one `decision_rows` payload.
pub const MAX_DECISION_ROWS_PER_BATCH: usize = 4096;

/// Maximum encoded payload size accepted for durable storage and bounded IPC
/// decode (bytes). Single source of truth lives in `decision_batch`.
pub use super::decision_batch::MAX_PAYLOAD_BYTES;

/// Default row cap applied to join/list queries when the caller passes 0.
pub const DEFAULT_QUERY_LIMIT: i64 = 1000;

/// Hard cap on explicit query limits (bounded requests).
pub const MAX_QUERY_LIMIT: i64 = 10_000;

const MAX_TENANT_LEN: usize = 128;
const MAX_PIN_LEN: usize = 256;
const MAX_CALIB_LEN: usize = 128;
const MAX_QUESTION_LEN: usize = 128;
const MAX_STORED_ID_LEN: usize = 64;
const MAX_LABEL_LEN: usize = 256;
const MAX_SOURCE_LEN: usize = 64;

fn now_unix_ms() -> i64 {
    let ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    i64::try_from(ms).unwrap_or(i64::MAX)
}

fn escape_sql(value: &str) -> String {
    value.replace('\'', "''")
}

fn invalid(msg: impl Into<String>) -> Status {
    Status::invalid_argument(msg.into())
}

/// A tenant domain the embedding service vouches for.
///
/// Construction is the trust boundary: `new` validates only that the value is
/// a bounded, non-wildcard domain string. The verification decision (which
/// caller maps to which tenant) belongs to the embedding service, derived
/// from verified credentials — never from request fields or stored data.
/// Every store read/write/join requires it, so a caller cannot touch another
/// tenant's rows without a vouched tenant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrustedTenant(String);

impl TrustedTenant {
    /// Wrap a tenant domain resolved from VERIFIED caller context.
    ///
    /// Validates the bounded domain shape only; the caller MUST have derived
    /// the value from verified credentials (e.g. `EnvelopeContext::domain()`).
    pub fn new(domain: impl Into<String>) -> Result<Self, Status> {
        let domain = domain.into();
        if domain.is_empty() {
            return Err(invalid("tenant domain must not be empty"));
        }
        if domain == "*" {
            return Err(invalid("tenant domain must not be the global wildcard"));
        }
        if domain.len() > MAX_TENANT_LEN {
            return Err(invalid(format!(
                "tenant domain exceeds {MAX_TENANT_LEN} characters"
            )));
        }
        if domain.chars().any(char::is_control) {
            return Err(invalid("tenant domain must not contain control characters"));
        }
        if domain.trim() != domain {
            return Err(invalid(
                "tenant domain must not have leading/trailing whitespace",
            ));
        }
        Ok(Self(domain))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for TrustedTenant {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// The complete identity pins stamped on every decision row.
///
/// `calib_version: None` marks RAW (uncalibrated) inference and is preserved
/// end to end; it is never defaulted or inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionPins {
    /// Decision-schema fingerprint of the payload (P0.1a IR identity).
    pub schema_fingerprint: String,
    /// Resolved model id/version that produced the batch.
    pub model_id: String,
    /// Clef encoding contract id (text/JSON encoding identity).
    pub encoding_id: String,
    /// Quantization profile pin (e.g. `"fp8-e4m3-dynamic"`, `"none"`).
    pub quantization: String,
    /// Compute profile pin (e.g. `"cpu-f32"`).
    pub compute_profile: String,
    /// Calibration version applied at emission; `None` = raw output.
    pub calib_version: Option<String>,
}

impl DecisionPins {
    /// Structural validation: every pin non-empty and bounded; control
    /// characters rejected so stored rows stay single-line queryable.
    pub fn validate(&self) -> Result<(), Status> {
        for (name, value) in [
            ("schema_fingerprint", &self.schema_fingerprint),
            ("model_id", &self.model_id),
            ("encoding_id", &self.encoding_id),
            ("quantization", &self.quantization),
            ("compute_profile", &self.compute_profile),
        ] {
            validate_pin_field(name, value)?;
        }
        if let Some(calib) = &self.calib_version {
            validate_pin_field("calib_version", calib)?;
            if calib.len() > MAX_CALIB_LEN {
                return Err(invalid(format!(
                    "calib_version exceeds {MAX_CALIB_LEN} characters"
                )));
            }
        }
        Ok(())
    }
}

fn validate_pin_field(name: &str, value: &str) -> Result<(), Status> {
    if value.is_empty() {
        return Err(invalid(format!("{name} must not be empty")));
    }
    if value.len() > MAX_PIN_LEN {
        return Err(invalid(format!("{name} exceeds {MAX_PIN_LEN} characters")));
    }
    if value.chars().any(char::is_control) {
        return Err(invalid(format!(
            "{name} must not contain control characters"
        )));
    }
    Ok(())
}

/// Validate a caller/supplier-supplied stable stored id (`drow-…`, `dec-…`).
fn validate_stored_id(kind: &str, id: &str) -> Result<(), Status> {
    if id.is_empty() {
        return Err(invalid(format!("{kind} must not be empty")));
    }
    if id.len() > MAX_STORED_ID_LEN {
        return Err(invalid(format!(
            "{kind} exceeds {MAX_STORED_ID_LEN} characters"
        )));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(invalid(format!("{kind} may only contain [A-Za-z0-9._:/-]")));
    }
    Ok(())
}

/// Validate a question id: identifier-safe, matching the jev Arrow surface
/// constraint (columns compose unquoted in SQL).
fn validate_question_id(question_id: &str) -> Result<(), Status> {
    if question_id.is_empty() || question_id.len() > MAX_QUESTION_LEN {
        return Err(invalid(format!(
            "question_id must be 1..={MAX_QUESTION_LEN} characters"
        )));
    }
    let valid = question_id.chars().enumerate().all(|(i, c)| {
        if i == 0 {
            c.is_ascii_alphabetic() || c == '_'
        } else {
            c.is_ascii_alphanumeric() || c == '_'
        }
    });
    if !valid {
        return Err(invalid(
            "question_id must be identifier-safe ([A-Za-z_][A-Za-z0-9_]*)",
        ));
    }
    Ok(())
}

/// Mint a decision-batch id (stable primary key of `decision_rows`).
pub fn mint_decision_id() -> String {
    format!("dec-{}", uuid::Uuid::new_v4().simple())
}

/// Mint a per-input-row decision id (stable for outcome linkage).
pub fn mint_decision_row_id() -> String {
    format!("drow-{}", uuid::Uuid::new_v4().simple())
}

/// Mint an outcome id (stable primary key of `decision_outcomes`).
pub fn mint_outcome_id() -> String {
    format!("outc-{}", uuid::Uuid::new_v4().simple())
}

/// Arrow schema of the `decision_rows` table.
pub fn decision_rows_schema() -> Schema {
    Schema::new(vec![
        Field::new("decision_id", DataType::Utf8, false),
        Field::new("tenant_id", DataType::Utf8, false),
        Field::new("schema_fingerprint", DataType::Utf8, false),
        Field::new("model_id", DataType::Utf8, false),
        Field::new("encoding_id", DataType::Utf8, false),
        Field::new("quantization", DataType::Utf8, false),
        Field::new("compute_profile", DataType::Utf8, false),
        Field::new("calib_version", DataType::Utf8, true),
        Field::new("created_at", DataType::Int64, false),
        Field::new("row_count", DataType::Int64, false),
        Field::new("payload_format", DataType::Utf8, false),
        Field::new("payload", DataType::Binary, false),
    ])
}

/// Arrow schema of the `decision_outcomes` table.
pub fn decision_outcomes_schema() -> Schema {
    Schema::new(vec![
        Field::new("outcome_id", DataType::Utf8, false),
        Field::new("tenant_id", DataType::Utf8, false),
        Field::new("decision_id", DataType::Utf8, false),
        Field::new("decision_row_id", DataType::Utf8, false),
        Field::new("question_id", DataType::Utf8, false),
        Field::new("outcome_label", DataType::Utf8, true),
        Field::new("outcome_value", DataType::Float64, true),
        Field::new("outcome_source", DataType::Utf8, false),
        Field::new("supersedes_outcome_id", DataType::Utf8, true),
        Field::new("created_at", DataType::Int64, false),
    ])
}

/// Create both R6 tables (idempotent `CREATE TABLE IF NOT EXISTS`).
pub async fn create_decision_outcome_tables(backend: &dyn StorageBackend) -> Result<(), Status> {
    backend
        .create_table(DECISION_ROWS_TABLE, &decision_rows_schema())
        .await?;
    backend
        .create_table(DECISION_OUTCOMES_TABLE, &decision_outcomes_schema())
        .await
}

/// Metadata of one stored decision batch (everything but the payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRowMeta {
    /// Stable decision-batch id ([`mint_decision_id`]).
    pub decision_id: String,
    /// Tenant the row belongs to (from the write's [`TrustedTenant`]).
    pub tenant_id: String,
    /// Complete identity pins (schema/model/encoding/quantization/compute/calib).
    pub pins: DecisionPins,
    /// Creation time, Unix epoch milliseconds.
    pub created_at: i64,
    /// Number of input rows in the payload.
    pub row_count: i64,
    /// Payload format tag (always `arrow-ipc-stream`).
    pub payload_format: String,
}

/// Result of a successful decision-row write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredDecision {
    /// Minted decision-batch id.
    pub decision_id: String,
    /// Stable per-input-row ids, in payload row order (caller-supplied ids
    /// are preserved verbatim).
    pub row_ids: Vec<String>,
}

fn decision_row_meta_from_row(batch: &RecordBatch, row: usize) -> Result<DecisionRowMeta, Status> {
    let string_at = |name: &str| -> Result<String, Status> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .map(|arr| arr.value(row).to_owned())
            .ok_or_else(|| Status::internal(format!("decision_rows: invalid {name} column")))
    };
    let i64_at = |name: &str| -> Result<i64, Status> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .map(|arr| arr.value(row))
            .ok_or_else(|| Status::internal(format!("decision_rows: invalid {name} column")))
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
    Ok(DecisionRowMeta {
        decision_id: string_at("decision_id")?,
        tenant_id: string_at("tenant_id")?,
        pins: DecisionPins {
            schema_fingerprint: string_at("schema_fingerprint")?,
            model_id: string_at("model_id")?,
            encoding_id: string_at("encoding_id")?,
            quantization: string_at("quantization")?,
            compute_profile: string_at("compute_profile")?,
            calib_version,
        },
        created_at: i64_at("created_at")?,
        row_count: i64_at("row_count")?,
        payload_format: string_at("payload_format")?,
    })
}

fn decision_row_meta_row_batch(
    meta: &DecisionRowMeta,
    payload: &[u8],
) -> Result<RecordBatch, Status> {
    let schema = Arc::new(decision_rows_schema());
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![meta.decision_id.as_str()])),
        Arc::new(StringArray::from(vec![meta.tenant_id.as_str()])),
        Arc::new(StringArray::from(vec![meta
            .pins
            .schema_fingerprint
            .as_str()])),
        Arc::new(StringArray::from(vec![meta.pins.model_id.as_str()])),
        Arc::new(StringArray::from(vec![meta.pins.encoding_id.as_str()])),
        Arc::new(StringArray::from(vec![meta.pins.quantization.as_str()])),
        Arc::new(StringArray::from(vec![meta.pins.compute_profile.as_str()])),
        Arc::new(StringArray::from(vec![meta.pins.calib_version.as_deref()])),
        Arc::new(Int64Array::from(vec![meta.created_at])),
        Arc::new(Int64Array::from(vec![meta.row_count])),
        Arc::new(StringArray::from(vec![meta.payload_format.as_str()])),
        Arc::new(arrow::array::BinaryArray::from(vec![payload])),
    ];
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| Status::internal(format!("Failed to create decision-row record: {e}")))
}

/// Append the store-owned [`DECISION_ROW_ID_COLUMN`] with freshly minted ids.
fn batch_with_minted_row_ids(batch: &RecordBatch) -> Result<(RecordBatch, Vec<String>), Status> {
    let row_ids: Vec<String> = (0..batch.num_rows())
        .map(|_| mint_decision_row_id())
        .collect();
    let mut fields: Vec<FieldRef> = batch.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(
        DECISION_ROW_ID_COLUMN,
        DataType::Utf8,
        false,
    )));
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(Arc::new(StringArray::from(row_ids.clone())));
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| Status::internal(format!("Failed to attach decision row ids: {e}")))?;
    Ok((batch, row_ids))
}

/// Extract and validate the stable row-id column of a payload batch.
///
/// Caller-supplied ids are preserved verbatim (R6C): validated for shape and
/// uniqueness, never rewritten. Absent column -> mints one id per row.
fn payload_row_ids(batch: &RecordBatch) -> Result<(Option<RecordBatch>, Vec<String>), Status> {
    match batch.column_by_name(DECISION_ROW_ID_COLUMN) {
        Some(col) => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| invalid(format!("{DECISION_ROW_ID_COLUMN} column must be Utf8")))?;
            let mut ids = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    return Err(invalid(format!(
                        "{DECISION_ROW_ID_COLUMN} must not contain nulls"
                    )));
                }
                let id = arr.value(i);
                validate_stored_id("decision_row_id", id)?;
                ids.push(id.to_owned());
            }
            let unique = ids.iter().collect::<std::collections::HashSet<_>>();
            if unique.len() != ids.len() {
                return Err(invalid(
                    "decision_row_id values must be unique within a batch",
                ));
            }
            Ok((None, ids))
        }
        None => {
            let (batch, ids) = batch_with_minted_row_ids(batch)?;
            Ok((Some(batch), ids))
        }
    }
}

/// One declared question domain, read from a payload schema at admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QuestionDomain {
    pub kind: QuestionKind,
    pub labels: Vec<String>,
}

/// Validate a declared label domain against the jev profile decisions.
///
/// - noul: exactly `["false", "true"]` (canonical order),
/// - choice: 2..=255 unique non-empty labels (D2/A4),
/// - score: canonical level indices `"0".."N-1"` (>= 2 levels, D1),
/// - every label bounded and distinct.
fn validate_declared_labels(
    question_id: &str,
    kind: QuestionKind,
    labels: &[String],
) -> Result<(), Status> {
    let bounded = labels.len() <= 255
        && labels
            .iter()
            .all(|l| !l.is_empty() && l.len() <= MAX_LABEL_LEN && !l.chars().any(char::is_control));
    if !bounded {
        return Err(invalid(format!(
            "question {question_id:?}: declared labels must be 1..=255 non-empty bounded strings"
        )));
    }
    let distinct = labels
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        == labels.len();
    if !distinct {
        return Err(invalid(format!(
            "question {question_id:?}: declared labels must be distinct"
        )));
    }
    match kind {
        QuestionKind::Noul => {
            if labels != ["false".to_owned(), "true".to_owned()] {
                return Err(invalid(format!(
                    "question {question_id:?}: noul labels must be the canonical [false, true]"
                )));
            }
        }
        QuestionKind::Choice => {
            if labels.len() < 2 {
                return Err(invalid(format!(
                    "question {question_id:?}: choice needs >= 2 options"
                )));
            }
        }
        QuestionKind::Score => {
            if labels.len() < 2 {
                return Err(invalid(format!(
                    "question {question_id:?}: score needs >= 2 levels"
                )));
            }
            let canonical: Vec<String> = (0..labels.len()).map(|level| level.to_string()).collect();
            if labels != canonical {
                return Err(invalid(format!(
                    "question {question_id:?}: score labels must be the canonical levels 0..{}",
                    labels.len() - 1
                )));
            }
        }
        QuestionKind::Span | QuestionKind::Derived => {
            return Err(invalid(format!(
                "question {question_id:?}: reserved profile-v2 kind {} is not storable",
                kind.as_str()
            )));
        }
    }
    Ok(())
}

/// Read and fully validate the jev question domain for a `{qid}.probabilities`
/// field: identifier-safe question id, present `FIELD_META_KIND` /
/// `FIELD_META_LABELS` metadata (profile-v1 kind), canonical declared label
/// domain, and a `fixed_size_list<float32; N>` distribution whose width
/// equals the label count (R6A, enforced at admission AND at outcome write).
pub(crate) fn question_domain_of(
    field: &Field,
    question_id: &str,
) -> Result<QuestionDomain, Status> {
    validate_question_id(question_id)?;
    let metadata = field.metadata();
    let kind_tag = metadata.get(FIELD_META_KIND).ok_or_else(|| {
        invalid(format!(
            "question {question_id:?}: field carries no {FIELD_META_KIND} metadata"
        ))
    })?;
    let kind = QuestionKind::from_tag(kind_tag).ok_or_else(|| {
        invalid(format!(
            "question {question_id:?}: unknown {FIELD_META_KIND} {kind_tag:?}"
        ))
    })?;
    if !kind.is_v1() {
        return Err(invalid(format!(
            "question {question_id:?}: reserved profile-v2 kind {kind_tag:?} is not storable"
        )));
    }
    let labels_json = metadata.get(FIELD_META_LABELS).ok_or_else(|| {
        invalid(format!(
            "question {question_id:?}: field carries no {FIELD_META_LABELS} metadata"
        ))
    })?;
    let labels: Vec<String> = serde_json::from_str(labels_json).map_err(|e| {
        invalid(format!(
            "question {question_id:?}: invalid {FIELD_META_LABELS} JSON: {e}"
        ))
    })?;
    validate_declared_labels(question_id, kind, &labels)?;
    match field.data_type() {
        DataType::FixedSizeList(values, size) => {
            if values.data_type() != &arrow::datatypes::DataType::Float32 {
                return Err(invalid(format!(
                    "question {question_id:?}: distribution values must be float32, got {:?}",
                    values.data_type()
                )));
            }
            if (*size as usize) != labels.len() {
                return Err(invalid(format!(
                    "question {question_id:?}: declared {} labels but distribution width is {size}",
                    labels.len()
                )));
            }
        }
        other => {
            return Err(invalid(format!(
                "question {question_id:?}: distribution must be fixed_size_list<f32; N>, got {other:?}"
            )));
        }
    }
    Ok(QuestionDomain { kind, labels })
}

/// Read the declared domain of one question from a stored payload schema
/// (R6A): the `{qid}.probabilities` field must exist and validate fully.
fn question_domain(payload_schema: &Schema, question_id: &str) -> Result<QuestionDomain, Status> {
    let field_name = format!("{question_id}{PROBABILITIES_COLUMN_SUFFIX}");
    let field = payload_schema.field_with_name(&field_name).map_err(|_| {
        invalid(format!(
            "unknown question {question_id:?}: payload schema has no {field_name} column"
        ))
    })?;
    question_domain_of(field, question_id)
}

/// Validate a decision payload schema at ADMISSION (`put`).
///
/// Rejects: duplicate field names (Arrow tolerates them; name-keyed question
/// lookup must stay unambiguous), zero `{qid}.probabilities` question fields,
/// and any question field that fails the full jev domain check
/// ([`question_domain_of`]). Non-question columns are not constrained beyond
/// uniqueness; the reserved stable-id column is handled by the caller.
pub(crate) fn validate_payload_schema(schema: &Schema) -> Result<(), Status> {
    let mut seen = std::collections::HashSet::new();
    let mut questions = 0usize;
    for field in schema.fields() {
        if !seen.insert(field.name().clone()) {
            return Err(invalid(format!(
                "payload schema has duplicate field name {:?}",
                field.name()
            )));
        }
        if let Some(question_id) = field.name().strip_suffix(PROBABILITIES_COLUMN_SUFFIX) {
            question_domain_of(field, question_id)?;
            questions += 1;
        }
    }
    if questions == 0 {
        return Err(invalid(
            "decision payload needs at least one `{qid}.probabilities` question field",
        ));
    }
    Ok(())
}

/// Reserved version-triple column names the jev emitter appends to every
/// batch. When present, their ROW VALUES are authoritative provenance and are
/// cross-checked against the declared pins before a durable write.
pub(crate) const MODEL_VERSION_COLUMN: &str = "model_version";
/// See [`MODEL_VERSION_COLUMN`] — null on every row = raw (uncalibrated).
pub(crate) const CALIB_VERSION_COLUMN: &str = "calib_version";

/// Cross-check the reserved version-triple columns, when present, against the
/// declared pins (review admission gap 2): `model_version` rows must all equal
/// `pins.model_id`; `calib_version` rows must be all-NULL for a raw pin and
/// all equal to the pin otherwise. `schema_version` is deliberately NOT
/// compared to the Arrow fingerprint — it is the question-set schema version,
/// a different identity.
pub(crate) fn validate_reserved_columns(
    batch: &RecordBatch,
    pins: &DecisionPins,
) -> Result<(), Status> {
    let schema = batch.schema();
    if let Ok(field) = schema.field_with_name(MODEL_VERSION_COLUMN) {
        if field.data_type() != &DataType::Utf8 {
            return Err(invalid(format!(
                "{MODEL_VERSION_COLUMN} column must be Utf8, got {:?}",
                field.data_type()
            )));
        }
        let values = batch
            .column_by_name(MODEL_VERSION_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| invalid(format!("invalid {MODEL_VERSION_COLUMN} column")))?;
        for row in 0..values.len() {
            if values.is_null(row) {
                return Err(invalid(format!(
                    "{MODEL_VERSION_COLUMN} must not be null when the column is present"
                )));
            }
            if values.value(row) != pins.model_id {
                return Err(invalid(format!(
                    "{MODEL_VERSION_COLUMN} row value {:?} contradicts the declared model pin {:?}",
                    values.value(row),
                    pins.model_id
                )));
            }
        }
    }
    if let Ok(field) = schema.field_with_name(CALIB_VERSION_COLUMN) {
        if field.data_type() != &DataType::Utf8 {
            return Err(invalid(format!(
                "{CALIB_VERSION_COLUMN} column must be Utf8, got {:?}",
                field.data_type()
            )));
        }
        let values = batch
            .column_by_name(CALIB_VERSION_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| invalid(format!("invalid {CALIB_VERSION_COLUMN} column")))?;
        for row in 0..values.len() {
            match (&pins.calib_version, values.is_null(row)) {
                (None, false) => {
                    return Err(invalid(format!(
                        "{CALIB_VERSION_COLUMN} row value {:?} contradicts the RAW pin (calib_version = None)",
                        values.value(row)
                    )));
                }
                (Some(want), true) => {
                    return Err(invalid(format!(
                        "null {CALIB_VERSION_COLUMN} row contradicts the calibrated pin {want:?}"
                    )));
                }
                (Some(want), false) => {
                    if values.value(row) != want {
                        return Err(invalid(format!(
                            "{CALIB_VERSION_COLUMN} row value {:?} contradicts the declared calibration pin {want:?}",
                            values.value(row)
                        )));
                    }
                }
                (None, true) => {}
            }
        }
    }
    Ok(())
}

/// Validate distribution VALUES of every question column (review admission
/// gap 1): a whole-list NULL row is preserved abstention; a PRESENT
/// distribution must have no null components and must satisfy the jev D5
/// contract at this boundary's READER tolerance — every component finite in
/// [0, 1] and the sum within [`CONSUMER_SUM_TOLERANCE`] (1e-2) of 1. Durable
/// ingest is a reader of a supplied batch: distributions inside the consumer
/// tolerance (but outside the tighter 1e-6 producer bound) are accepted
/// unchanged, never normalized; beyond-consumer sums are rejected.
pub(crate) fn validate_payload_values(batch: &RecordBatch) -> Result<(), Status> {
    for field in batch.schema().fields() {
        let Some(question_id) = field.name().strip_suffix(PROBABILITIES_COLUMN_SUFFIX) else {
            continue;
        };
        let column = batch
            .column_by_name(field.name())
            .ok_or_else(|| invalid("payload column vanished between schema and data"))?;
        let list = column
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                invalid(format!(
                    "question {question_id:?}: distribution must be fixed_size_list<f32; N>"
                ))
            })?;
        for row in 0..list.len() {
            if list.is_null(row) {
                continue; // whole-list null = abstained, preserved as data
            }
            let row_values = list.value(row);
            let values = row_values
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .ok_or_else(|| {
                    invalid(format!(
                        "question {question_id:?}: distribution values must be float32"
                    ))
                })?;
            if values.null_count() > 0 {
                return Err(invalid(format!(
                    "question {question_id:?}: null component inside a present distribution \
                     (whole-row null is the only abstention form)"
                )));
            }
            let slice: Vec<f32> = values.values().to_vec();
            check_distribution(&slice, CONSUMER_SUM_TOLERANCE)
                .map_err(|e| invalid(format!("question {question_id:?} row {row}: {e}")))?;
        }
    }
    Ok(())
}

/// Validate a realized outcome against the declared question domain (R6A).
///
/// - noul/choice: realized LABEL only, and it must be an exact member of the
///   declared label set — no numeric value channel.
/// - score: EITHER the realized level label (exact canonical `"i"` member of
///   the declared domain — `"01"`/`"+1"` spellings are not members) OR the
///   realized numeric value in `[0, N-1]`. Providing both is ambiguous and
///   rejected; this keeps the numeric-score outcome semantics explicit.
/// - An abstained decision never forces a gold: at least one of label/value
///   must be present, and labels stay nullable in storage.
pub(crate) fn validate_outcome_against_domain(
    domain: &QuestionDomain,
    question_id: &str,
    label: &Option<String>,
    value: &Option<f64>,
) -> Result<(), Status> {
    if let Some(label) = label {
        if label.is_empty() || label.len() > MAX_LABEL_LEN {
            return Err(invalid(format!(
                "outcome label for {question_id:?} must be 1..={MAX_LABEL_LEN} characters"
            )));
        }
        if !domain.labels.iter().any(|l| l == label) {
            return Err(invalid(format!(
                "outcome label {label:?} is not in the declared domain of question {question_id:?}"
            )));
        }
        if value.is_some() {
            return Err(invalid(format!(
                "question {question_id:?} is {}: realized label and value are mutually exclusive; \
                 supply the level label or the numeric value, not both",
                domain.kind
            )));
        }
    }
    if let Some(value) = value {
        if !value.is_finite() {
            return Err(invalid(format!(
                "outcome value for {question_id:?} must be finite"
            )));
        }
        if domain.kind != QuestionKind::Score {
            return Err(invalid(format!(
                "question {question_id:?} is {}: realized values are score-only; label the outcome instead",
                domain.kind
            )));
        }
        let max_level = (domain.labels.len() - 1) as f64;
        if !(0.0..=max_level).contains(value) {
            return Err(invalid(format!(
                "outcome value {value} is outside the declared score domain [0, {max_level}] of question {question_id:?}"
            )));
        }
    }
    if label.is_none() && value.is_none() {
        return Err(invalid(format!(
            "outcome for {question_id:?} needs a realized label or value (abstention is a property of the decision, never a fabricated gold)"
        )));
    }
    Ok(())
}

/// Calibration expectation for joins and outcome assertions.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CalibFilter {
    /// No calibration constraint.
    #[default]
    All,
    /// Only raw (uncalibrated) decision rows — `calib_version IS NULL`.
    RawOnly,
    /// Only calibrated decision rows — `calib_version IS NOT NULL`.
    CalibratedOnly,
    /// Only rows pinned to this exact calibration version.
    Version(String),
}

impl CalibFilter {
    fn sql_predicate(&self, column: &str) -> Result<String, Status> {
        Ok(match self {
            Self::All => String::new(),
            Self::RawOnly => format!(" AND {column} IS NULL"),
            Self::CalibratedOnly => format!(" AND {column} IS NOT NULL"),
            Self::Version(v) => {
                if v.is_empty() {
                    return Err(invalid("calibration version filter must not be empty"));
                }
                format!(" AND {column} = '{}'", escape_sql(v))
            }
        })
    }
}

/// Calibration expectation asserted on an outcome write (R6B).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CalibExpectation {
    /// No calibration assertion.
    #[default]
    Any,
    /// The linked decision row must be raw (`calib_version` NULL).
    Raw,
    /// The linked decision row must carry this exact calibration version.
    Version(String),
}

/// Optional complete-identity assertions checked on an outcome write (R6B).
///
/// Every `Some` field is compared exactly against the linked decision row's
/// pin; `None` fields are not asserted.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExpectedPins {
    pub schema_fingerprint: Option<String>,
    pub model_id: Option<String>,
    pub encoding_id: Option<String>,
    pub quantization: Option<String>,
    pub compute_profile: Option<String>,
}

impl ExpectedPins {
    fn check(&self, pins: &DecisionPins) -> Result<(), Status> {
        let mismatch = |field: &str, expected: &str, actual: &str| {
            Status::failed_precondition(format!(
                "pin mismatch on {field}: expected {expected:?}, decision row has {actual:?}"
            ))
        };
        if let Some(want) = &self.schema_fingerprint {
            if want != &pins.schema_fingerprint {
                return Err(mismatch(
                    "schema_fingerprint",
                    want,
                    &pins.schema_fingerprint,
                ));
            }
        }
        if let Some(want) = &self.model_id {
            if want != &pins.model_id {
                return Err(mismatch("model_id", want, &pins.model_id));
            }
        }
        if let Some(want) = &self.encoding_id {
            if want != &pins.encoding_id {
                return Err(mismatch("encoding_id", want, &pins.encoding_id));
            }
        }
        if let Some(want) = &self.quantization {
            if want != &pins.quantization {
                return Err(mismatch("quantization", want, &pins.quantization));
            }
        }
        if let Some(want) = &self.compute_profile {
            if want != &pins.compute_profile {
                return Err(mismatch("compute_profile", want, &pins.compute_profile));
            }
        }
        Ok(())
    }
}

/// A new realized outcome to append (R6 write request).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NewOutcome {
    /// Exact decision batch containing the observed row.
    pub decision_id: String,
    /// Exact input-row id inside that batch's payload.
    pub decision_row_id: String,
    /// Question the outcome answers (must exist in the stored schema).
    pub question_id: String,
    /// Optional complete pin assertions (R6B).
    pub expected_pins: Option<ExpectedPins>,
    /// Optional calibration assertion (R6B).
    pub expected_calib: CalibExpectation,
    /// Realized label (declared domain checked against the stored schema).
    pub outcome_label: Option<String>,
    /// Realized numeric value (score questions).
    pub outcome_value: Option<f64>,
    /// Provenance of the observation (e.g. "human-label", "ground-truth").
    pub outcome_source: String,
    /// When this observation CORRECTS an earlier one: the outcome id it
    /// supersedes (same tenant + same decision row + same question). The
    /// superseded row stays (append-only) but is excluded from default joins.
    pub supersedes_outcome_id: Option<String>,
}

/// One stored realized outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeRecord {
    pub outcome_id: String,
    pub tenant_id: String,
    pub decision_id: String,
    pub decision_row_id: String,
    pub question_id: String,
    pub outcome_label: Option<String>,
    pub outcome_value: Option<f64>,
    pub outcome_source: String,
    pub supersedes_outcome_id: Option<String>,
    pub created_at: i64,
}

fn outcome_record_from_row(batch: &RecordBatch, row: usize) -> Result<OutcomeRecord, Status> {
    let string_at = |name: &str| -> Result<String, Status> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .map(|arr| arr.value(row).to_owned())
            .ok_or_else(|| Status::internal(format!("decision_outcomes: invalid {name} column")))
    };
    let opt_string_at = |name: &str| -> Result<Option<String>, Status> {
        Ok(batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .and_then(|arr| {
                if arr.is_null(row) {
                    None
                } else {
                    Some(arr.value(row).to_owned())
                }
            }))
    };
    let value_at = batch
        .column_by_name("outcome_value")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        .and_then(|arr| {
            if arr.is_null(row) {
                None
            } else {
                Some(arr.value(row))
            }
        });
    let created_at = batch
        .column_by_name("created_at")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .map(|arr| arr.value(row))
        .ok_or_else(|| Status::internal("decision_outcomes: invalid created_at column"))?;
    Ok(OutcomeRecord {
        outcome_id: string_at("outcome_id")?,
        tenant_id: string_at("tenant_id")?,
        decision_id: string_at("decision_id")?,
        decision_row_id: string_at("decision_row_id")?,
        question_id: string_at("question_id")?,
        outcome_label: opt_string_at("outcome_label")?,
        outcome_value: value_at,
        outcome_source: string_at("outcome_source")?,
        supersedes_outcome_id: opt_string_at("supersedes_outcome_id")?,
        created_at,
    })
}

fn outcome_record_row_batch(record: &OutcomeRecord) -> Result<RecordBatch, Status> {
    let schema = Arc::new(decision_outcomes_schema());
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![record.outcome_id.as_str()])),
        Arc::new(StringArray::from(vec![record.tenant_id.as_str()])),
        Arc::new(StringArray::from(vec![record.decision_id.as_str()])),
        Arc::new(StringArray::from(vec![record.decision_row_id.as_str()])),
        Arc::new(StringArray::from(vec![record.question_id.as_str()])),
        Arc::new(StringArray::from(vec![record.outcome_label.as_deref()])),
        Arc::new(Float64Array::from(vec![record.outcome_value])),
        Arc::new(StringArray::from(vec![record.outcome_source.as_str()])),
        Arc::new(StringArray::from(vec![record
            .supersedes_outcome_id
            .as_deref()])),
        Arc::new(Int64Array::from(vec![record.created_at])),
    ];
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| Status::internal(format!("Failed to create outcome record: {e}")))
}

/// Durable decision-row store over any [`StorageBackend`].
///
/// `B` may be `?Sized` so `Arc<dyn StorageBackend>` works where the serving
/// process wires the engine backend behind the trait object.
pub struct DecisionRowStore<B: StorageBackend + ?Sized> {
    backend: Arc<B>,
}

/// Filters for [`DecisionRowStore::fetch_meta`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DecisionRowQuery {
    pub schema_fingerprint: Option<String>,
    pub model_id: Option<String>,
    pub calib: CalibFilter,
    /// 0 = [`DEFAULT_QUERY_LIMIT`]; hard-capped at [`MAX_QUERY_LIMIT`].
    pub limit: u32,
}

impl<B: StorageBackend + ?Sized> DecisionRowStore<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Create the `decision_rows` table.
    pub async fn init(&self) -> Result<(), Status> {
        self.backend
            .create_table(DECISION_ROWS_TABLE, &decision_rows_schema())
            .await
    }

    /// Store one decision batch durably (append-only).
    ///
    /// Admission gates, all before any durable write: pin shape; payload
    /// schema (duplicate fields, jev question domains); reserved
    /// `model_version`/`calib_version` row values against the declared pins
    /// (raw/calibrated and model provenance are checked, never guessed);
    /// distribution values (finite, [0, 1], sum within the D5 consumer
    /// tolerance — ingest is a READER boundary; native producers stay bound
    /// by the tighter 1e-6 producer tolerance upstream; whole-list null is
    /// the only abstention form). The IPC payload itself is decoded
    /// BOUNDED: framing, declared row counts, and the ACTUAL decompressed
    /// expansion of every compressed buffer are verified within
    /// MAX_PAYLOAD_BYTES before the Arrow decoder allocates. Invalid input is
    /// rejected — nothing is normalized and nothing is inserted.
    ///
    /// Idempotency/retry semantics: each call appends a NEW decision record.
    /// This is explicitly NOT an atomic or concurrency-safe exactly-once
    /// guarantee — a retried request that already landed (or races a
    /// concurrent identical request) stores a second record; append
    /// duplicates remain possible by design. Callers needing deduplication
    /// supply their own stable `decision_row_id` column (preserved verbatim)
    /// and resolve duplicates at read time. Payload columns must NOT collide
    /// with the reserved `decision_row_id` name unless they ARE the stable id
    /// column.
    pub async fn put(
        &self,
        tenant: &TrustedTenant,
        pins: &DecisionPins,
        batch: &RecordBatch,
    ) -> Result<StoredDecision, Status> {
        pins.validate()?;
        // R6A admission gate: duplicate field names, question-field dtype and
        // declared label domains are validated on write, before any row
        // accounting, so an invalid schema never lands durably.
        validate_payload_schema(&batch.schema())?;
        // Reserved provenance columns contradicting the declared pins are
        // rejected: raw/calibrated and model identity come from the pins AND
        // the payload together, never from one alone.
        validate_reserved_columns(batch, pins)?;
        // Distribution VALUES satisfy the jev D5 contract at the READER
        // tolerance (consumer 1e-2); invalid vectors are rejected, never
        // normalized. Native producers remain bound by the tighter 1e-6
        // producer tolerance upstream.
        validate_payload_values(batch)?;
        if batch.num_rows() == 0 {
            return Err(invalid("decision payload must carry at least one row"));
        }
        if batch.num_rows() > MAX_DECISION_ROWS_PER_BATCH {
            return Err(invalid(format!(
                "decision payload exceeds {MAX_DECISION_ROWS_PER_BATCH} rows"
            )));
        }
        let (payload_batch, row_ids) = match payload_row_ids(batch)? {
            (Some(batch), ids) => (batch, ids),
            (None, ids) => (batch.clone(), ids),
        };
        let payload = encode_batch_ipc(&payload_batch)?;
        if payload.len() > MAX_PAYLOAD_BYTES {
            return Err(invalid(format!(
                "encoded decision payload exceeds {MAX_PAYLOAD_BYTES} bytes"
            )));
        }
        let meta = DecisionRowMeta {
            decision_id: mint_decision_id(),
            tenant_id: tenant.as_str().to_owned(),
            pins: pins.clone(),
            created_at: now_unix_ms(),
            row_count: i64::try_from(payload_batch.num_rows())
                .map_err(|_| Status::internal("row count overflow"))?,
            payload_format: PAYLOAD_FORMAT_IPC_STREAM.to_owned(),
        };
        let row = decision_row_meta_row_batch(&meta, &payload)?;
        self.backend
            .insert_into_table(DECISION_ROWS_TABLE, row)
            .await?;
        Ok(StoredDecision {
            decision_id: meta.decision_id,
            row_ids,
        })
    }

    /// Fetch one decision batch's metadata by id, tenant-scoped.
    pub async fn get_decision(
        &self,
        tenant: &TrustedTenant,
        decision_id: &str,
    ) -> Result<Option<DecisionRowMeta>, Status> {
        validate_stored_id("decision_id", decision_id)?;
        let sql = format!(
            "SELECT decision_id, tenant_id, schema_fingerprint, model_id, encoding_id, \
             quantization, compute_profile, calib_version, created_at, row_count, payload_format \
             FROM {DECISION_ROWS_TABLE} \
             WHERE decision_id = '{}' AND tenant_id = '{}' LIMIT 2",
            escape_sql(decision_id),
            escape_sql(tenant.as_str()),
        );
        let batch = self.query(&sql).await?;
        match batch.num_rows() {
            0 => Ok(None),
            1 => Ok(Some(decision_row_meta_from_row(&batch, 0)?)),
            _ => Err(Status::internal("duplicate decision_id for tenant")),
        }
    }

    /// List decision-batch metadata under the caller's tenant, oldest first.
    pub async fn fetch_meta(
        &self,
        tenant: &TrustedTenant,
        query: &DecisionRowQuery,
    ) -> Result<Vec<DecisionRowMeta>, Status> {
        let mut sql = format!(
            "SELECT decision_id, tenant_id, schema_fingerprint, model_id, encoding_id, \
             quantization, compute_profile, calib_version, created_at, row_count, payload_format \
             FROM {DECISION_ROWS_TABLE} WHERE tenant_id = '{}'",
            escape_sql(tenant.as_str()),
        );
        if let Some(fingerprint) = &query.schema_fingerprint {
            validate_pin_field("schema_fingerprint", fingerprint)?;
            sql.push_str(&format!(
                " AND schema_fingerprint = '{}'",
                escape_sql(fingerprint)
            ));
        }
        if let Some(model) = &query.model_id {
            validate_pin_field("model_id", model)?;
            sql.push_str(&format!(" AND model_id = '{}'", escape_sql(model)));
        }
        sql.push_str(&query.calib.sql_predicate("calib_version")?);
        // Deterministic oldest-first order with a total tie-break: created_at
        // is millisecond wall time (ties happen), decision_id is unique.
        sql.push_str(" ORDER BY created_at ASC, decision_id ASC");
        sql.push_str(&limit_clause(query.limit)?);
        let batch = self.query(&sql).await?;
        (0..batch.num_rows())
            .map(|row| decision_row_meta_from_row(&batch, row))
            .collect()
    }

    /// Decode the typed distribution row for one exact input row.
    ///
    /// Returns the decision's pins plus a single-row slice of the payload —
    /// the lossless Arrow distributions the producer emitted, addressed by
    /// stable row id across reloads.
    pub async fn fetch_distribution(
        &self,
        tenant: &TrustedTenant,
        decision_id: &str,
        decision_row_id: &str,
    ) -> Result<Option<(DecisionPins, RecordBatch)>, Status> {
        validate_stored_id("decision_id", decision_id)?;
        validate_stored_id("decision_row_id", decision_row_id)?;
        let (pins, payload) = match self.load_payload(tenant, decision_id).await? {
            Some(found) => found,
            None => return Ok(None),
        };
        let batch = decode_payload_bounded(&payload)?;
        let ids = payload_row_id_column(&batch)?;
        let mut offset = None;
        for i in 0..ids.len() {
            if ids.value(i) == decision_row_id {
                offset = Some(i);
                break;
            }
        }
        match offset {
            None => Ok(None),
            Some(i) => Ok(Some((pins, batch.slice(i, 1)))),
        }
    }

    /// Load (pins, payload bytes) for one tenant-scoped decision row.
    async fn load_payload(
        &self,
        tenant: &TrustedTenant,
        decision_id: &str,
    ) -> Result<Option<(DecisionPins, Vec<u8>)>, Status> {
        let sql = format!(
            "SELECT decision_id, tenant_id, schema_fingerprint, model_id, encoding_id, \
             quantization, compute_profile, calib_version, created_at, row_count, \
             payload_format, payload FROM {DECISION_ROWS_TABLE} \
             WHERE decision_id = '{}' AND tenant_id = '{}' LIMIT 2",
            escape_sql(decision_id),
            escape_sql(tenant.as_str()),
        );
        let batch = self.query(&sql).await?;
        match batch.num_rows() {
            0 => return Ok(None),
            1 => {}
            _ => return Err(Status::internal("duplicate decision_id for tenant")),
        }
        let meta = decision_row_meta_from_row(&batch, 0)?;
        let payload_col = batch
            .column_by_name("payload")
            .ok_or_else(|| Status::internal("decision_rows: missing payload column"))?;
        let bytes: Vec<u8> = if let Some(arr) = payload_col
            .as_any()
            .downcast_ref::<arrow::array::BinaryArray>()
        {
            arr.value(0).to_vec()
        } else if let Some(arr) = payload_col
            .as_any()
            .downcast_ref::<arrow::array::LargeBinaryArray>()
        {
            arr.value(0).to_vec()
        } else {
            return Err(Status::internal(format!(
                "decision_rows: unexpected payload column type {:?}",
                payload_col.data_type()
            )));
        };
        Ok(Some((meta.pins, bytes)))
    }

    async fn query(&self, sql: &str) -> Result<RecordBatch, Status> {
        let handle = self.backend.prepare_sql(sql).await?;
        self.backend.query_sql(&handle).await
    }
}

/// The stable row-id column of a decoded payload batch.
fn payload_row_id_column(batch: &RecordBatch) -> Result<&StringArray, Status> {
    batch
        .column_by_name(DECISION_ROW_ID_COLUMN)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            Status::internal(format!(
                "stored payload lost its {DECISION_ROW_ID_COLUMN} column"
            ))
        })
}

/// Decode an IPC payload with bounds enforced BEFORE decoder allocation: the
/// encoded size and every declared (decompressed) buffer length are verified
/// within MAX_PAYLOAD_BYTES first; valid compression is preserved.
fn decode_payload_bounded(bytes: &[u8]) -> Result<RecordBatch, Status> {
    if bytes.len() > MAX_PAYLOAD_BYTES {
        return Err(Status::internal(
            "stored decision payload exceeds the durable size bound",
        ));
    }
    decode_batch_ipc_bounded(bytes, MAX_PAYLOAD_BYTES)
}

fn effective_query_limit(limit: u32) -> i64 {
    if limit == 0 {
        DEFAULT_QUERY_LIMIT
    } else {
        i64::from(limit).min(MAX_QUERY_LIMIT)
    }
}

fn limit_clause(limit: u32) -> Result<String, Status> {
    Ok(format!(" LIMIT {}", effective_query_limit(limit)))
}

/// Realized-outcome store over any [`StorageBackend`] (`B` may be unsized).
pub struct OutcomeStore<B: StorageBackend + ?Sized> {
    backend: Arc<B>,
}

/// Filters for [`OutcomeStore::join_outcomes`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DecisionOutcomeQuery {
    pub schema_fingerprint: Option<String>,
    pub model_id: Option<String>,
    pub question_id: Option<String>,
    pub calib: CalibFilter,
    /// Include outcomes that a later correction superseded (default: no).
    pub include_superseded: bool,
    /// 0 = [`DEFAULT_QUERY_LIMIT`]; hard-capped at [`MAX_QUERY_LIMIT`].
    pub limit: u32,
}

/// Validated ASC `(created_at, outcome_id)` continuation key.
///
/// A cursor is a position, never tenant authority or a snapshot guarantee.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutcomeCursor {
    created_at: i64,
    outcome_id: String,
}

impl OutcomeCursor {
    pub fn new(created_at: i64, outcome_id: String) -> Result<Self, Status> {
        if created_at < 0 {
            return Err(invalid("afterCreatedAt must not be negative"));
        }
        validate_stored_id("afterOutcomeId", &outcome_id)?;
        Ok(Self {
            created_at,
            outcome_id,
        })
    }

    /// Decode the append-only wire fields, including the legacy first page.
    pub fn from_wire(created_at: i64, outcome_id: &str) -> Result<Option<Self>, Status> {
        if created_at == 0 && outcome_id.is_empty() {
            return Ok(None);
        }
        Self::new(created_at, outcome_id.to_owned()).map(Some)
    }

    pub fn created_at(&self) -> i64 {
        self.created_at
    }
    pub fn outcome_id(&self) -> &str {
        &self.outcome_id
    }
}

/// One bounded page; a continuation exists only when a later row was seen.
#[derive(Debug)]
pub struct OutcomePage {
    pub rows: RecordBatch,
    pub next_cursor: Option<OutcomeCursor>,
}

impl<B: StorageBackend + ?Sized> OutcomeStore<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Create the `decision_outcomes` table.
    pub async fn init(&self) -> Result<(), Status> {
        self.backend
            .create_table(DECISION_OUTCOMES_TABLE, &decision_outcomes_schema())
            .await
    }

    /// Append one realized outcome (append-only).
    ///
    /// Validates, in order: request shape; the linked decision EXISTS under
    /// the caller's tenant (cross-tenant linkage is a not-found, never a
    /// leak); asserted pins/calibration match the decision row exactly (R6B);
    /// the question exists in the STORED Arrow schema with a declared domain
    /// that admits the realized label/value (R6A); the exact decision row id
    /// is present in the payload (R6 exact-row linkage); and a superseded
    /// target, when given, is a same-observation outcome of the same tenant.
    pub async fn record(
        &self,
        tenant: &TrustedTenant,
        outcome: &NewOutcome,
    ) -> Result<OutcomeRecord, Status> {
        validate_stored_id("decision_id", &outcome.decision_id)?;
        validate_stored_id("decision_row_id", &outcome.decision_row_id)?;
        validate_question_id(&outcome.question_id)?;
        if outcome.outcome_source.is_empty() || outcome.outcome_source.len() > MAX_SOURCE_LEN {
            return Err(invalid(format!(
                "outcome_source must be 1..={MAX_SOURCE_LEN} characters"
            )));
        }
        if outcome.outcome_source.chars().any(char::is_control) {
            return Err(invalid(
                "outcome_source must not contain control characters",
            ));
        }
        if let Some(prev) = &outcome.supersedes_outcome_id {
            validate_stored_id("supersedes_outcome_id", prev)?;
        }

        // Tenant-scoped decision lookup: foreign tenants read as not-found.
        let sql = format!(
            "SELECT decision_id, tenant_id, schema_fingerprint, model_id, encoding_id, \
             quantization, compute_profile, calib_version, created_at, row_count, \
             payload_format, payload FROM {DECISION_ROWS_TABLE} \
             WHERE decision_id = '{}' AND tenant_id = '{}' LIMIT 2",
            escape_sql(&outcome.decision_id),
            escape_sql(tenant.as_str()),
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        if batch.num_rows() == 0 {
            return Err(Status::not_found(format!(
                "decision {} not found in tenant {}",
                outcome.decision_id,
                tenant.as_str()
            )));
        }
        if batch.num_rows() != 1 {
            return Err(Status::internal("duplicate decision_id for tenant"));
        }
        let meta = decision_row_meta_from_row(&batch, 0)?;

        // R6B: exact complete-identity comparison for every asserted pin.
        if let Some(expected) = &outcome.expected_pins {
            expected.check(&meta.pins)?;
        }
        match &outcome.expected_calib {
            CalibExpectation::Any => {}
            CalibExpectation::Raw => {
                if let Some(calib) = &meta.pins.calib_version {
                    return Err(Status::failed_precondition(format!(
                        "calibration mismatch: expected raw decision, row is pinned to {calib:?}"
                    )));
                }
            }
            CalibExpectation::Version(want) => {
                if meta.pins.calib_version.as_deref() != Some(want.as_str()) {
                    return Err(Status::failed_precondition(format!(
                        "calibration mismatch: expected {want:?}, decision row has {:?}",
                        meta.pins.calib_version
                    )));
                }
            }
        }

        // R6A: the question must exist in the STORED payload schema.
        let payload_col = batch
            .column_by_name("payload")
            .ok_or_else(|| Status::internal("decision_rows: missing payload column"))?;
        let payload_bytes: Vec<u8> = if let Some(arr) = payload_col
            .as_any()
            .downcast_ref::<arrow::array::BinaryArray>()
        {
            arr.value(0).to_vec()
        } else if let Some(arr) = payload_col
            .as_any()
            .downcast_ref::<arrow::array::LargeBinaryArray>()
        {
            arr.value(0).to_vec()
        } else {
            return Err(Status::internal("decision_rows: unexpected payload type"));
        };
        let payload = decode_payload_bounded(&payload_bytes)?;
        let domain = question_domain(&payload.schema(), &outcome.question_id)?;
        validate_outcome_against_domain(
            &domain,
            &outcome.question_id,
            &outcome.outcome_label,
            &outcome.outcome_value,
        )?;

        // Exact-row linkage: the id must address one row of this payload.
        let ids = payload_row_id_column(&payload)?;
        let occurrences = (0..ids.len())
            .filter(|i| ids.value(*i) == outcome.decision_row_id)
            .count();
        match occurrences {
            0 => {
                return Err(invalid(format!(
                    "decision_row_id {:?} is not a row of decision {}",
                    outcome.decision_row_id, outcome.decision_id
                )));
            }
            1 => {}
            _ => return Err(Status::internal("duplicate row id inside stored payload")),
        }

        // Supersession must correct the SAME observation, same tenant.
        if let Some(prev_id) = &outcome.supersedes_outcome_id {
            let prev = self.fetch_outcome(tenant, prev_id).await?.ok_or_else(|| {
                Status::not_found(format!(
                    "superseded outcome {prev_id} not found in tenant {}",
                    tenant.as_str()
                ))
            })?;
            if prev.decision_id != outcome.decision_id
                || prev.decision_row_id != outcome.decision_row_id
                || prev.question_id != outcome.question_id
            {
                return Err(invalid(
                    "superseded outcome must address the same decision row and question",
                ));
            }
        }

        let record = OutcomeRecord {
            outcome_id: mint_outcome_id(),
            tenant_id: tenant.as_str().to_owned(),
            decision_id: outcome.decision_id.clone(),
            decision_row_id: outcome.decision_row_id.clone(),
            question_id: outcome.question_id.clone(),
            outcome_label: outcome.outcome_label.clone(),
            outcome_value: outcome.outcome_value,
            outcome_source: outcome.outcome_source.clone(),
            supersedes_outcome_id: outcome.supersedes_outcome_id.clone(),
            created_at: now_unix_ms(),
        };
        let row = outcome_record_row_batch(&record)?;
        self.backend
            .insert_into_table(DECISION_OUTCOMES_TABLE, row)
            .await?;
        Ok(record)
    }

    /// Fetch one outcome by id, tenant-scoped.
    pub async fn fetch_outcome(
        &self,
        tenant: &TrustedTenant,
        outcome_id: &str,
    ) -> Result<Option<OutcomeRecord>, Status> {
        validate_stored_id("outcome_id", outcome_id)?;
        let sql = format!(
            "SELECT outcome_id, tenant_id, decision_id, decision_row_id, question_id, \
             outcome_label, outcome_value, outcome_source, supersedes_outcome_id, created_at \
             FROM {DECISION_OUTCOMES_TABLE} \
             WHERE outcome_id = '{}' AND tenant_id = '{}' LIMIT 2",
            escape_sql(outcome_id),
            escape_sql(tenant.as_str()),
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        match batch.num_rows() {
            0 => Ok(None),
            1 => Ok(Some(outcome_record_from_row(&batch, 0)?)),
            _ => Err(Status::internal("duplicate outcome_id for tenant")),
        }
    }

    /// Decision-to-outcome join for the caller's tenant, bounded in SQL.
    ///
    /// One row per realized outcome with the decision's complete pins. Rows
    /// a later correction superseded are excluded unless
    /// [`DecisionOutcomeQuery::include_superseded`] is set. Superseded
    /// exclusion uses a correlated NOT EXISTS — still tenant-predicated.
    ///
    /// Result columns (in order): `outcome_id, tenant_id, decision_id,
    /// decision_row_id, question_id, outcome_label, outcome_value,
    /// outcome_source, supersedes_outcome_id, created_at,
    /// schema_fingerprint, model_id, encoding_id, quantization,
    /// compute_profile, calib_version, decision_created_at`.
    pub async fn join_outcomes(
        &self,
        tenant: &TrustedTenant,
        query: &DecisionOutcomeQuery,
    ) -> Result<RecordBatch, Status> {
        Ok(self.join_outcomes_page(tenant, query, None).await?.rows)
    }

    /// Continue the existing ascending join with a strict keyset cursor.
    /// Every page re-applies all tenant/filter predicates. This is not a
    /// snapshot across concurrent appends, corrections or clock changes.
    pub async fn join_outcomes_page(
        &self,
        tenant: &TrustedTenant,
        query: &DecisionOutcomeQuery,
        cursor: Option<&OutcomeCursor>,
    ) -> Result<OutcomePage, Status> {
        let mut sql = format!(
            "SELECT o.outcome_id, o.tenant_id, o.decision_id, o.decision_row_id, o.question_id, \
             o.outcome_label, o.outcome_value, o.outcome_source, o.supersedes_outcome_id, \
             o.created_at, d.schema_fingerprint, d.model_id, d.encoding_id, d.quantization, \
             d.compute_profile, d.calib_version, d.created_at AS decision_created_at \
             FROM {DECISION_OUTCOMES_TABLE} AS o \
             JOIN {DECISION_ROWS_TABLE} AS d \
               ON d.decision_id = o.decision_id AND d.tenant_id = o.tenant_id \
             WHERE o.tenant_id = '{}'",
            escape_sql(tenant.as_str()),
        );
        if let Some(fingerprint) = &query.schema_fingerprint {
            validate_pin_field("schema_fingerprint", fingerprint)?;
            sql.push_str(&format!(
                " AND d.schema_fingerprint = '{}'",
                escape_sql(fingerprint)
            ));
        }
        if let Some(model) = &query.model_id {
            validate_pin_field("model_id", model)?;
            sql.push_str(&format!(" AND d.model_id = '{}'", escape_sql(model)));
        }
        if let Some(question) = &query.question_id {
            validate_question_id(question)?;
            sql.push_str(&format!(" AND o.question_id = '{}'", escape_sql(question)));
        }
        sql.push_str(&query.calib.sql_predicate("d.calib_version")?);
        if !query.include_superseded {
            sql.push_str(&format!(
                " AND NOT EXISTS (SELECT 1 FROM {DECISION_OUTCOMES_TABLE} AS s \
                 WHERE s.tenant_id = o.tenant_id AND s.supersedes_outcome_id = o.outcome_id)"
            ));
        }
        if let Some(cursor) = cursor {
            sql.push_str(&format!(
                " AND (o.created_at > {} OR (o.created_at = {} AND o.outcome_id > '{}'))",
                cursor.created_at,
                cursor.created_at,
                escape_sql(&cursor.outcome_id)
            ));
        }
        sql.push_str(" ORDER BY o.created_at ASC, o.outcome_id ASC");
        let limit = usize::try_from(effective_query_limit(query.limit))
            .map_err(|_| Status::internal("invalid effective outcome query limit"))?;
        // The one look-ahead row is internal, never an increased public cap.
        sql.push_str(&format!(" LIMIT {}", limit + 1));
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        let next_cursor = if batch.num_rows() > limit {
            let ids = batch
                .column_by_name("outcome_id")
                .and_then(|v| v.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Status::internal("outcome page missing typed outcome_id"))?;
            let times = batch
                .column_by_name("created_at")
                .and_then(|v| v.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| Status::internal("outcome page missing typed created_at"))?;
            let last = limit - 1;
            if ids.is_null(last) || times.is_null(last) {
                return Err(Status::internal("outcome page has null continuation key"));
            }
            Some(
                OutcomeCursor::new(times.value(last), ids.value(last).to_owned()).map_err(
                    |_| Status::internal("outcome page has invalid stored continuation key"),
                )?,
            )
        } else {
            None
        };
        Ok(OutcomePage {
            rows: batch.slice(0, batch.num_rows().min(limit)),
            next_cursor,
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use arrow::array::BinaryArray;

    fn pins() -> DecisionPins {
        DecisionPins {
            schema_fingerprint: "jev1:route:v3".to_owned(),
            model_id: "clef-flash@8b2e5fd1".to_owned(),
            encoding_id: "clef-text-json-v1".to_owned(),
            quantization: "fp8-e4m3-dynamic".to_owned(),
            compute_profile: "cpu-f32".to_owned(),
            calib_version: None,
        }
    }

    #[test]
    fn trusted_tenant_validates_bounded_domain() {
        assert!(TrustedTenant::new("tenant-a").is_ok());
        assert!(TrustedTenant::new("").is_err());
        assert!(TrustedTenant::new("*").is_err());
        assert!(TrustedTenant::new(" a").is_err());
        assert!(TrustedTenant::new("a\nb").is_err());
        assert!(TrustedTenant::new("x".repeat(MAX_TENANT_LEN + 1)).is_err());
    }

    #[test]
    fn pins_validate_required_fields() {
        let mut p = pins();
        assert!(p.validate().is_ok());
        p.quantization = String::new();
        assert!(p.validate().is_err());
        let mut p = pins();
        p.calib_version = Some(String::new());
        assert!(p.validate().is_err());
    }

    #[test]
    fn stored_id_validation() {
        assert!(validate_stored_id("decision_id", "dec-abc123").is_ok());
        assert!(validate_stored_id("decision_id", "").is_err());
        assert!(validate_stored_id("decision_id", "has space").is_err());
        assert!(validate_stored_id("decision_id", "it's").is_err());
    }

    #[test]
    fn question_id_must_be_identifier_safe() {
        assert!(validate_question_id("route_choice").is_ok());
        assert!(validate_question_id("_q1").is_ok());
        assert!(validate_question_id("1q").is_err());
        assert!(validate_question_id("has-dash").is_err());
        assert!(validate_question_id("").is_err());
    }

    #[test]
    fn outcome_needs_label_or_value_and_domain_membership() {
        let domain = QuestionDomain {
            kind: QuestionKind::Choice,
            labels: vec!["a".to_owned(), "b".to_owned()],
        };
        assert!(validate_outcome_against_domain(&domain, "q", &Some("a".into()), &None).is_ok());
        assert!(validate_outcome_against_domain(&domain, "q", &Some("z".into()), &None).is_err());
        assert!(validate_outcome_against_domain(&domain, "q", &None, &None).is_err());
        assert!(validate_outcome_against_domain(&domain, "q", &None, &Some(0.5)).is_err());
        assert!(
            validate_outcome_against_domain(&domain, "q", &Some("a".into()), &Some(0.5)).is_err()
        );

        let score = QuestionDomain {
            kind: QuestionKind::Score,
            labels: vec!["0".into(), "1".into(), "2".into()],
        };
        assert!(validate_outcome_against_domain(&score, "s", &Some("2".into()), &None).is_ok());
        assert!(validate_outcome_against_domain(&score, "s", &None, &Some(1.5)).is_ok());
        assert!(validate_outcome_against_domain(&score, "s", &Some("3".into()), &None).is_err());
        // Non-canonical level spellings are not declared members.
        assert!(validate_outcome_against_domain(&score, "s", &Some("01".into()), &None).is_err());
        assert!(validate_outcome_against_domain(&score, "s", &Some("+1".into()), &None).is_err());
        assert!(validate_outcome_against_domain(&score, "s", &None, &Some(2.5)).is_err());
        assert!(validate_outcome_against_domain(&score, "s", &None, &Some(f64::NAN)).is_err());
        // Label+value on a score is ambiguous -> rejected (review followup).
        assert!(
            validate_outcome_against_domain(&score, "s", &Some("0".into()), &Some(2.0)).is_err()
        );
        // A fabricated empty label is rejected; null abstention stays null.
        assert!(validate_outcome_against_domain(&score, "s", &Some(String::new()), &None).is_err());
    }

    #[test]
    fn declared_label_domains_are_canonical() {
        assert!(validate_declared_labels(
            "n",
            QuestionKind::Noul,
            &["false".into(), "true".into()]
        )
        .is_ok());
        assert!(validate_declared_labels(
            "n",
            QuestionKind::Noul,
            &["true".into(), "false".into()]
        )
        .is_err());
        assert!(
            validate_declared_labels("c", QuestionKind::Choice, &["a".into(), "b".into()]).is_ok()
        );
        assert!(validate_declared_labels("c", QuestionKind::Choice, &["a".into()]).is_err());
        assert!(
            validate_declared_labels("c", QuestionKind::Choice, &["a".into(), "a".into()]).is_err()
        );
        assert!(validate_declared_labels(
            "s",
            QuestionKind::Score,
            &["0".into(), "1".into(), "2".into()]
        )
        .is_ok());
        // Non-canonical score spellings rejected at admission.
        assert!(validate_declared_labels(
            "s",
            QuestionKind::Score,
            &["00".into(), "1".into(), "2".into()]
        )
        .is_err());
        assert!(validate_declared_labels("s", QuestionKind::Score, &["0".into()]).is_err());
        // Reserved v2 kinds are not storable.
        assert!(
            validate_declared_labels("d", QuestionKind::Derived, &["0".into(), "1".into()])
                .is_err()
        );
    }

    fn question_field(
        name: &str,
        kind: &str,
        labels: &str,
        width: i32,
    ) -> arrow::datatypes::FieldRef {
        use std::collections::HashMap;
        let mut meta = HashMap::new();
        meta.insert(FIELD_META_KIND.to_owned(), kind.to_owned());
        meta.insert(FIELD_META_LABELS.to_owned(), labels.to_owned());
        Arc::new(
            Field::new(
                name,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    width,
                ),
                true,
            )
            .with_metadata(meta),
        )
    }

    #[test]
    fn score_domain_cardinality_rejects_before_range_formatting() {
        // Zero-width FixedSizeList and empty label metadata are structurally
        // valid Arrow schema input. Domain admission must return an error
        // without evaluating an underflowing last-level range expression.
        for (labels, width) in [("[]", 0), (r#"["0"]"#, 1)] {
            let schema = Schema::new(vec![question_field(
                "score.probabilities",
                "score",
                labels,
                width,
            )]);
            let Err(error) = validate_payload_schema(&schema) else {
                panic!("score domains require at least two levels");
            };
            assert_eq!(error.code(), tonic::Code::InvalidArgument);
            assert_eq!(
                error.message(),
                r#"question "score": score needs >= 2 levels"#
            );
        }
        // The minimum valid canonical domain and non-canonical rejection
        // retain their existing behavior at the same schema boundary.
        let valid = Schema::new(vec![question_field(
            "score.probabilities",
            "score",
            r#"["0","1"]"#,
            2,
        )]);
        assert!(validate_payload_schema(&valid).is_ok());
        let malformed = Schema::new(vec![question_field(
            "score.probabilities",
            "score",
            r#"["0","01"]"#,
            2,
        )]);
        let Err(error) = validate_payload_schema(&malformed) else {
            panic!("score labels must remain canonical");
        };
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(error.message().contains("canonical levels 0..1"));
    }

    #[test]
    fn payload_schema_admission_rejects_bad_questions() {
        // Duplicate field names are rejected.
        let dup = Schema::new(vec![
            question_field("q.probabilities", "choice", r#"["a","b"]"#, 2),
            question_field("q.probabilities", "choice", r#"["a","b"]"#, 2),
        ]);
        let err = validate_payload_schema(&dup).unwrap_err();
        assert!(err.message().contains("duplicate field name"));

        // Zero question fields rejected.
        let empty = Schema::new(vec![Field::new("note", DataType::Utf8, false)]);
        let err = validate_payload_schema(&empty).unwrap_err();
        assert!(err.message().contains("at least one"));

        // Wrong inner dtype rejected (float64 list is not the contract).
        let mut f64_meta = std::collections::HashMap::new();
        f64_meta.insert(FIELD_META_KIND.to_owned(), "choice".to_owned());
        f64_meta.insert(FIELD_META_LABELS.to_owned(), r#"["a","b"]"#.to_owned());
        let wrong_dtype = Schema::new(vec![Arc::new(
            Field::new(
                "q.probabilities",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 2),
                true,
            )
            .with_metadata(f64_meta),
        )]);
        let err = validate_payload_schema(&wrong_dtype).unwrap_err();
        assert!(err.message().contains("float32"));

        // Cardinality mismatch rejected.
        let bad_width = Schema::new(vec![question_field(
            "q.probabilities",
            "choice",
            r#"["a","b","c"]"#,
            2,
        )]);
        let err = validate_payload_schema(&bad_width).unwrap_err();
        assert!(err.message().contains("distribution width"));

        // Non-canonical declared labels rejected (score "00").
        let bad_labels = Schema::new(vec![question_field(
            "q.probabilities",
            "score",
            r#"["00","1","2"]"#,
            3,
        )]);
        let err = validate_payload_schema(&bad_labels).unwrap_err();
        assert!(err.message().contains("canonical levels"));

        // A valid schema passes.
        let ok = Schema::new(vec![
            question_field("route.probabilities", "choice", r#"["a","b","c"]"#, 3),
            question_field("score_level.probabilities", "score", r#"["0","1","2"]"#, 3),
            Arc::new(Field::new("model_version", DataType::Utf8, false)),
        ]);
        assert!(validate_payload_schema(&ok).is_ok());
    }

    #[test]
    fn calib_filter_sql_predicates() {
        assert_eq!(CalibFilter::All.sql_predicate("c").unwrap(), "");
        assert_eq!(
            CalibFilter::RawOnly.sql_predicate("c").unwrap(),
            " AND c IS NULL"
        );
        assert_eq!(
            CalibFilter::CalibratedOnly.sql_predicate("c").unwrap(),
            " AND c IS NOT NULL"
        );
        assert_eq!(
            CalibFilter::Version("cal-1".into())
                .sql_predicate("c")
                .unwrap(),
            " AND c = 'cal-1'"
        );
        assert!(CalibFilter::Version(String::new())
            .sql_predicate("c")
            .is_err());
    }

    #[test]
    fn expected_pins_compare_exactly() {
        let p = pins();
        let mut expected = ExpectedPins::default();
        assert!(expected.check(&p).is_ok());
        expected.quantization = Some("none".into());
        let err = expected.check(&p).unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(err.message().contains("quantization"));
        // A raw row never satisfies a version expectation.
        let mismatch = CalibExpectation::Version("cal-1".into());
        assert_ne!(mismatch, CalibExpectation::Raw);
    }

    #[test]
    fn minted_ids_are_unique_and_prefixed() {
        assert_ne!(mint_decision_id(), mint_decision_id());
        assert_ne!(mint_decision_row_id(), mint_decision_row_id());
        assert_ne!(mint_outcome_id(), mint_outcome_id());
        assert!(mint_decision_id().starts_with("dec-"));
        assert!(mint_decision_row_id().starts_with("drow-"));
        assert!(mint_outcome_id().starts_with("outc-"));
    }

    #[test]
    fn limit_clause_bounds_queries() {
        assert_eq!(
            limit_clause(0).unwrap(),
            format!(" LIMIT {DEFAULT_QUERY_LIMIT}")
        );
        assert_eq!(limit_clause(5).unwrap(), " LIMIT 5");
        assert_eq!(
            limit_clause(u32::MAX).unwrap(),
            format!(" LIMIT {MAX_QUERY_LIMIT}")
        );
    }

    #[test]
    fn binary_payload_column_type_is_declared() {
        let schema = decision_rows_schema();
        assert_eq!(
            schema.field_with_name("payload").unwrap().data_type(),
            &DataType::Binary
        );
        let outcomes = decision_outcomes_schema();
        assert_eq!(
            outcomes
                .field_with_name("outcome_value")
                .unwrap()
                .data_type(),
            &DataType::Float64
        );
        assert!(outcomes
            .field_with_name("outcome_label")
            .unwrap()
            .is_nullable());
        assert!(!outcomes
            .field_with_name("question_id")
            .unwrap()
            .is_nullable());
    }

    #[test]
    fn minted_row_id_column_is_appended_not_duplicated() -> Result<(), Status> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "q.probabilities",
            DataType::Binary,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BinaryArray::from(vec![b"d".as_slice()])) as ArrayRef],
        )
        .map_err(|e| Status::internal(e.to_string()))?;
        let (with_ids, ids) = payload_row_ids(&batch)?;
        assert_eq!(ids.len(), 1);
        let batch = with_ids.expect("minted column produces a new batch");
        assert!(batch.column_by_name(DECISION_ROW_ID_COLUMN).is_some());
        // Caller-supplied column is preserved verbatim (no new batch).
        let (none, ids2) = payload_row_ids(&batch)?;
        assert!(none.is_none());
        assert_eq!(ids2, ids);
        Ok(())
    }
    /// Reserved provenance columns: model pin and raw/calibrated pin are
    /// cross-checked row-by-row against the declared pins (utf8 enforced).
    #[test]
    fn reserved_columns_cross_check_against_pins() -> Result<(), Status> {
        let batch_with = |model: &str, calib: Option<&str>| {
            let values: Vec<Option<Vec<Option<f32>>>> = vec![Some(vec![Some(0.6), Some(0.4)])];
            let list = FixedSizeListArray::from_iter_primitive::<arrow::datatypes::Float32Type, _, _>(
                values, 2,
            );
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("q.probabilities", list.data_type().clone(), true),
                    Field::new("model_version", DataType::Utf8, false),
                    Field::new("calib_version", DataType::Utf8, true),
                ])),
                vec![
                    Arc::new(list) as ArrayRef,
                    Arc::new(StringArray::from(vec![model])) as ArrayRef,
                    Arc::new(StringArray::from(vec![calib])) as ArrayRef,
                ],
            )
            .map_err(|e| Status::internal(e.to_string()))
        };
        let p = pins();

        // Consistent pairs pass.
        validate_reserved_columns(&batch_with("clef-flash@8b2e5fd1", None)?, &p)?;
        validate_reserved_columns(
            &batch_with("clef-flash@8b2e5fd1", Some("cal-1"))?,
            &DecisionPins {
                calib_version: Some("cal-1".into()),
                ..p.clone()
            },
        )?;

        // Wrong model rejected.
        let err = validate_reserved_columns(&batch_with("other@v9", None)?, &p).unwrap_err();
        assert!(err.message().contains("contradicts the declared model pin"));

        // Calibrated payload under raw pin (and vice versa) rejected.
        let err = validate_reserved_columns(&batch_with("clef-flash@8b2e5fd1", Some("cal-1"))?, &p)
            .unwrap_err();
        assert!(err.message().contains("contradicts the RAW pin"));
        let err = validate_reserved_columns(
            &batch_with("clef-flash@8b2e5fd1", None)?,
            &DecisionPins {
                calib_version: Some("cal-1".into()),
                ..p.clone()
            },
        )
        .unwrap_err();
        assert!(err.message().contains("contradicts the calibrated pin"));
        Ok(())
    }

    /// Distribution VALUES: whole-list null preserved; finite/[0,1]/sum
    /// enforced at the READER (consumer) tolerance; null components inside a
    /// present distribution rejected; within-consumer/outside-producer sums
    /// accepted unchanged.
    #[test]
    fn payload_values_reader_tolerance_and_null_abstention() -> Result<(), Status> {
        let build = |rows: Vec<Option<Vec<Option<f32>>>>| {
            let list = FixedSizeListArray::from_iter_primitive::<arrow::datatypes::Float32Type, _, _>(
                rows, 2,
            );
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "q.probabilities",
                    list.data_type().clone(),
                    true,
                )])),
                vec![Arc::new(list) as ArrayRef],
            )
            .map_err(|e| Status::internal(e.to_string()))
        };

        // Abstained whole-list null + present valid distribution.
        validate_payload_values(&build(vec![None, Some(vec![Some(0.6), Some(0.4)])])?)?;

        // NaN / negative / >1 rejected with the jev DistributionError text.
        let err = validate_payload_values(&build(vec![Some(vec![Some(f32::NAN), Some(0.4)])])?)
            .unwrap_err();
        assert!(err.message().contains("not a finite probability"), "{err}");
        let err =
            validate_payload_values(&build(vec![Some(vec![Some(-0.2), Some(0.4)])])?).unwrap_err();
        assert!(err.message().contains("in [0, 1]"), "{err}");

        // Beyond-consumer sum rejected.
        let err =
            validate_payload_values(&build(vec![Some(vec![Some(0.9), Some(0.4)])])?).unwrap_err();
        assert!(err.message().contains("sums to"), "{err}");

        // Null component inside a present distribution rejected.
        let err = validate_payload_values(&build(vec![Some(vec![None, Some(0.4)])])?).unwrap_err();
        assert!(err.message().contains("null component"), "{err}");

        // Within consumer (1e-2) but outside producer (1e-6): 0.5+0.495.
        // Deviation 5e-3 — reader accepts unchanged; store boundary test
        // proves the stored bytes are the caller's.
        validate_payload_values(&build(vec![Some(vec![Some(0.5), Some(0.495)])])?)?;
        Ok(())
    }
    #[test]
    fn outcome_cursor_validates_pair_epoch_and_bounded_stored_id() {
        assert_eq!(OutcomeCursor::from_wire(0, "").unwrap(), None);
        assert_eq!(
            OutcomeCursor::from_wire(0, "outc-zero")
                .unwrap()
                .unwrap()
                .created_at(),
            0
        );
        for (time, id) in [(1, ""), (-1, ""), (-1, "outc-x"), (0, "bad' OR 1=1")] {
            assert_eq!(
                OutcomeCursor::from_wire(time, id).unwrap_err().code(),
                tonic::Code::InvalidArgument
            );
        }
        assert!(OutcomeCursor::from_wire(0, &"x".repeat(MAX_STORED_ID_LEN)).is_ok());
        assert!(OutcomeCursor::from_wire(0, &"x".repeat(MAX_STORED_ID_LEN + 1)).is_err());
        assert_eq!(effective_query_limit(0), 1000);
        assert_eq!(effective_query_limit(1), 1);
        assert_eq!(effective_query_limit(10000), 10000);
        assert_eq!(effective_query_limit(u32::MAX), 10000);
    }
}
