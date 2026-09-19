//! Calibration parameter tables for decision models (System One P0.3/P2.2).
//!
//! One row holds the post-hoc calibration parameters for a single decision
//! primitive (`noul`/`choice`/`score`) of one decision schema on one model,
//! keyed by the `(schema_id, model_id, calib_version)` version triple:
//!
//! - per-primitive temperature,
//! - null-state (content-free) bias vector,
//! - optional Dirichlet-calibration params for high-cardinality choice,
//! - conformal quantiles per primitive — score uses ordinal
//!   contiguous-interval params, noul/choice use APS/RAPS-style quantiles.
//!
//! Rows are append-only: a calibration re-fit mints a NEW `calib_version` row
//! and never updates in place, so every past fit stays queryable and serving
//! can pin an exact triple.

// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

use crate::storage::StorageBackend;
use duckdb::arrow::array::{
    Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray,
};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tonic::Status;

/// Calibration parameter table name.
pub const CALIBRATION_TABLE: &str = "calibration_params";

/// Mint a new calibration version id. Re-fits call this (or supply their own)
/// and insert a new row; existing rows are never mutated.
pub fn mint_calib_version() -> String {
    format!("cal-{}", uuid::Uuid::new_v4().simple())
}

/// Decision primitive a calibration row applies to (jev-1 profile superset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionPrimitive {
    /// Statement → P(true).
    Noul,
    /// One of ≤255 enumerated options.
    Choice,
    /// Ordered levels → distribution + probability-weighted value.
    Score,
}

impl DecisionPrimitive {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }

    pub fn parse(s: &str) -> Result<Self, Status> {
        match s {
            "noul" => Ok(Self::Noul),
            "choice" => Ok(Self::Choice),
            "score" => Ok(Self::Score),
            other => Err(Status::invalid_argument(format!(
                "unknown decision primitive: {other:?}"
            ))),
        }
    }
}

impl std::fmt::Display for DecisionPrimitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The (schema, model, calib) version triple keying every calibration row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionTriple {
    /// Decision-schema fingerprint (P0.1a IR identity).
    pub schema_id: String,
    /// Model id/version the parameters were fit for.
    pub model_id: String,
    /// Calibration version; minted fresh by every re-fit (see
    /// [`mint_calib_version`]).
    pub calib_version: String,
}

/// Dirichlet-calibration parameters (Kull et al. 2019) for high-cardinality
/// choice questions, where scalar temperature underfits: calibrated log-probs
/// are `W · ln(p) + b` with `weights` row-major K×K and `bias` of length K.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirichletParams {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

/// One coverage level → nonconformity quantile (APS/RAPS-style).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConformalQuantile {
    /// Target coverage level in (0, 1).
    pub level: f64,
    /// Nonconformity-score quantile at that level.
    pub quantile: f64,
}

/// One coverage level → ordinal contiguous interval half-widths, in level
/// units, applied around the predicted level (min-CPS / RPS-based
/// nonconformity for the score primitive).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrdinalInterval {
    /// Target coverage level in (0, 1).
    pub level: f64,
    /// Levels below the prediction included at this coverage.
    pub lower: f64,
    /// Levels above the prediction included at this coverage.
    pub upper: f64,
}

/// Conformal parameters for one primitive. Score questions are ordinal, so
/// their prediction sets are contiguous intervals; noul/choice use
/// APS/RAPS-style quantiles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConformalParams {
    /// APS/RAPS: nonconformity quantile per coverage level (noul/choice).
    Quantiles { quantiles: Vec<ConformalQuantile> },
    /// Ordinal contiguous intervals per coverage level (score).
    OrdinalIntervals { intervals: Vec<OrdinalInterval> },
}

/// One calibration parameter row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationParams {
    /// (schema, model, calib) version triple — the primary key.
    pub version: VersionTriple,
    /// Primitive these parameters apply to.
    pub primitive: DecisionPrimitive,
    /// Per-primitive temperature (softmax scale). `None` = identity.
    pub temperature: Option<f64>,
    /// Null-state (content-free input) bias vector, one entry per option/level.
    pub null_bias: Option<Vec<f64>>,
    /// Dirichlet-calibration params (choice only, high-cardinality).
    pub dirichlet: Option<DirichletParams>,
    /// Conformal parameters; the variant must match `primitive`.
    pub conformal: Option<ConformalParams>,
    /// Option/level cardinality K at fit time.
    pub cardinality: Option<i64>,
    /// Id of the calibration set the parameters were fit on.
    pub calib_set_id: String,
    /// Number of examples in the calibration set.
    pub calib_size: Option<i64>,
    /// Fitting objective, free-form (e.g. "nll", "rps", "brier").
    pub objective: Option<String>,
    /// Fit time, Unix epoch milliseconds.
    pub fitted_at: i64,
}

impl CalibrationParams {
    /// Structural validation applied on every write.
    pub fn validate(&self) -> Result<(), Status> {
        let invalid = |msg: &str| Status::invalid_argument(msg.to_owned());
        if self.version.schema_id.is_empty() {
            return Err(invalid("schema_id must not be empty"));
        }
        if self.version.model_id.is_empty() {
            return Err(invalid("model_id must not be empty"));
        }
        if self.version.calib_version.is_empty() {
            return Err(invalid("calib_version must not be empty"));
        }
        if self.calib_set_id.is_empty() {
            return Err(invalid("calib_set_id must not be empty"));
        }
        if let Some(t) = self.temperature {
            if !t.is_finite() || t <= 0.0 {
                return Err(invalid("temperature must be finite and > 0"));
            }
        }
        if let Some(bias) = &self.null_bias {
            if bias.iter().any(|v| !v.is_finite()) {
                return Err(invalid("null_bias entries must be finite"));
            }
        }
        if let Some(dirichlet) = &self.dirichlet {
            if self.primitive != DecisionPrimitive::Choice {
                return Err(invalid(
                    "dirichlet params are only valid for the choice primitive",
                ));
            }
            let k = dirichlet.weights.len();
            if k == 0
                || dirichlet.weights.iter().any(|row| row.len() != k)
                || dirichlet.bias.len() != k
            {
                return Err(invalid(
                    "dirichlet weights must be square KxK with bias of length K",
                ));
            }
            let finite = dirichlet
                .weights
                .iter()
                .flatten()
                .chain(dirichlet.bias.iter())
                .all(|v| v.is_finite());
            if !finite {
                return Err(invalid("dirichlet params must be finite"));
            }
            if let Some(card) = self.cardinality {
                if card >= 0 && k as i64 != card {
                    return Err(invalid(
                        "dirichlet dimension must equal cardinality",
                    ));
                }
            }
        }
        match (&self.conformal, self.primitive) {
            (Some(ConformalParams::Quantiles { .. }), DecisionPrimitive::Score) => {
                return Err(invalid(
                    "score conformal params must be ordinal intervals, not quantiles",
                ));
            }
            (Some(ConformalParams::OrdinalIntervals { intervals }), DecisionPrimitive::Score) => {
                for e in intervals {
                    if !(0.0..1.0).contains(&e.level) {
                        return Err(invalid("conformal levels must be in (0, 1)"));
                    }
                    if !e.lower.is_finite() || !e.upper.is_finite() || e.lower < 0.0 || e.upper < 0.0
                    {
                        return Err(invalid(
                            "ordinal interval offsets must be finite and >= 0",
                        ));
                    }
                }
            }
            (Some(ConformalParams::OrdinalIntervals { .. }), _) => {
                return Err(invalid(
                    "ordinal interval conformal params are only valid for the score primitive",
                ));
            }
            (Some(ConformalParams::Quantiles { quantiles }), _) => {
                for e in quantiles {
                    if !(0.0..1.0).contains(&e.level) || !e.quantile.is_finite() {
                        return Err(invalid(
                            "conformal levels must be in (0, 1) with finite quantiles",
                        ));
                    }
                }
            }
            (None, _) => {}
        }
        if let Some(card) = self.cardinality {
            if card <= 0 {
                return Err(invalid("cardinality must be > 0"));
            }
            if let Some(bias) = &self.null_bias {
                if bias.len() as i64 != card {
                    return Err(invalid("null_bias length must equal cardinality"));
                }
            }
        }
        if let Some(size) = self.calib_size {
            if size < 0 {
                return Err(invalid("calib_size must be >= 0"));
            }
        }
        Ok(())
    }
}

/// Arrow schema of the `calibration_params` table. Vector/composite params
/// are stored as JSON in nullable VARCHAR columns (the crate's existing
/// convention for composite values, queryable via DuckDB JSON functions).
pub fn calibration_params_schema() -> Schema {
    Schema::new(vec![
        Field::new("schema_id", DataType::Utf8, false),
        Field::new("model_id", DataType::Utf8, false),
        Field::new("calib_version", DataType::Utf8, false),
        Field::new("primitive", DataType::Utf8, false),
        Field::new("temperature", DataType::Float64, true),
        Field::new("null_bias", DataType::Utf8, true),
        Field::new("dirichlet_params", DataType::Utf8, true),
        Field::new("conformal_params", DataType::Utf8, true),
        Field::new("cardinality", DataType::Int64, true),
        Field::new("calib_set_id", DataType::Utf8, false),
        Field::new("calib_size", DataType::Int64, true),
        Field::new("objective", DataType::Utf8, true),
        Field::new("fitted_at", DataType::Int64, false),
    ])
}

fn json_or_none<T: Serialize>(value: &Option<T>) -> Result<Option<String>, Status> {
    value
        .as_ref()
        .map(|v| {
            serde_json::to_string(v)
                .map_err(|e| Status::internal(format!("calibration params JSON encode: {e}")))
        })
        .transpose()
}

/// Build the one-row RecordBatch for a calibration insert.
pub fn params_to_record_batch(params: &CalibrationParams) -> Result<RecordBatch, Status> {
    params.validate()?;
    let schema = Arc::new(calibration_params_schema());
    let null_bias = json_or_none(&params.null_bias)?;
    let dirichlet = json_or_none(&params.dirichlet)?;
    let conformal = json_or_none(&params.conformal)?;

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![params.version.schema_id.as_str()])),
        Arc::new(StringArray::from(vec![params.version.model_id.as_str()])),
        Arc::new(StringArray::from(vec![params.version.calib_version.as_str()])),
        Arc::new(StringArray::from(vec![params.primitive.as_str()])),
        Arc::new(Float64Array::from(vec![params.temperature])),
        Arc::new(StringArray::from(vec![null_bias.as_deref()])),
        Arc::new(StringArray::from(vec![dirichlet.as_deref()])),
        Arc::new(StringArray::from(vec![conformal.as_deref()])),
        Arc::new(Int64Array::from(vec![params.cardinality])),
        Arc::new(StringArray::from(vec![params.calib_set_id.as_str()])),
        Arc::new(Int64Array::from(vec![params.calib_size])),
        Arc::new(StringArray::from(vec![params.objective.as_deref()])),
        Arc::new(Int64Array::from(vec![params.fitted_at])),
    ];
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| Status::internal(format!("Failed to create calibration batch: {e}")))
}

fn opt_string(col: Option<&ArrayRef>, row: usize) -> Result<Option<String>, Status> {
    let col = col.ok_or_else(|| Status::internal("missing column in calibration row"))?;
    let arr = col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| Status::internal("invalid Utf8 column in calibration row"))?;
    Ok(if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row).to_owned())
    })
}

fn req_string(col: Option<&ArrayRef>, row: usize, name: &str) -> Result<String, Status> {
    opt_string(col, row)?.ok_or_else(|| Status::internal(format!("NULL {name} in calibration row")))
}

fn opt_f64(col: Option<&ArrayRef>, row: usize) -> Result<Option<f64>, Status> {
    let col = col.ok_or_else(|| Status::internal("missing column in calibration row"))?;
    let arr = col
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| Status::internal("invalid Float64 column in calibration row"))?;
    Ok(if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row))
    })
}

fn opt_i64(col: Option<&ArrayRef>, row: usize) -> Result<Option<i64>, Status> {
    let col = col.ok_or_else(|| Status::internal("missing column in calibration row"))?;
    let arr = col
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| Status::internal("invalid Int64 column in calibration row"))?;
    Ok(if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row))
    })
}

fn decode_json<T: serde::de::DeserializeOwned>(
    raw: Option<String>,
    column: &str,
) -> Result<Option<T>, Status> {
    raw.map(|s| {
        serde_json::from_str(&s)
            .map_err(|e| Status::internal(format!("invalid {column} JSON: {e}")))
    })
    .transpose()
}

/// Decode one calibration row from a query result batch.
pub fn params_from_row(batch: &RecordBatch, row: usize) -> Result<CalibrationParams, Status> {
    let col = |name: &str| batch.column_by_name(name);
    let primitive = DecisionPrimitive::parse(&req_string(col("primitive"), row, "primitive")?)?;
    let null_bias: Option<Vec<f64>> = decode_json(opt_string(col("null_bias"), row)?, "null_bias")?;
    let dirichlet: Option<DirichletParams> =
        decode_json(opt_string(col("dirichlet_params"), row)?, "dirichlet_params")?;
    let conformal: Option<ConformalParams> =
        decode_json(opt_string(col("conformal_params"), row)?, "conformal_params")?;
    Ok(CalibrationParams {
        version: VersionTriple {
            schema_id: req_string(col("schema_id"), row, "schema_id")?,
            model_id: req_string(col("model_id"), row, "model_id")?,
            calib_version: req_string(col("calib_version"), row, "calib_version")?,
        },
        primitive,
        temperature: opt_f64(col("temperature"), row)?,
        null_bias,
        dirichlet,
        conformal,
        cardinality: opt_i64(col("cardinality"), row)?,
        calib_set_id: req_string(col("calib_set_id"), row, "calib_set_id")?,
        calib_size: opt_i64(col("calib_size"), row)?,
        objective: opt_string(col("objective"), row)?,
        fitted_at: opt_i64(col("fitted_at"), row)?
            .ok_or_else(|| Status::internal("NULL fitted_at in calibration row"))?,
    })
}

fn escape_sql(value: &str) -> String {
    value.replace('\'', "''")
}

/// Calibration parameter store over any [`StorageBackend`]: the writer API for
/// the P2.2 re-fit loop and the reader API for serving.
pub struct CalibrationStore<B: StorageBackend> {
    backend: Arc<B>,
}

impl<B: StorageBackend> CalibrationStore<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Create the `calibration_params` table.
    pub async fn init(&self) -> Result<(), Status> {
        self.backend
            .create_table(CALIBRATION_TABLE, &calibration_params_schema())
            .await
    }

    /// Insert one validated parameter row. The (primitive, version triple) key
    /// must be new — re-fits mint a fresh `calib_version`; rows are never
    /// updated in place.
    pub async fn insert(&self, params: &CalibrationParams) -> Result<(), Status> {
        params.validate()?;
        let v = &params.version;
        let count_sql = format!(
            "SELECT COUNT(*) AS cnt FROM {CALIBRATION_TABLE} \
             WHERE schema_id = '{}' AND model_id = '{}' AND calib_version = '{}' AND primitive = '{}'",
            escape_sql(&v.schema_id),
            escape_sql(&v.model_id),
            escape_sql(&v.calib_version),
            params.primitive.as_str(),
        );
        let handle = self.backend.prepare_sql(&count_sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        if count_rows(&batch)? > 0 {
            return Err(Status::already_exists(format!(
                "calibration row for ({}, {}, {}, {}) already exists; re-fit must mint a new calib_version",
                params.primitive.as_str(), v.schema_id, v.model_id, v.calib_version,
            )));
        }
        let batch = params_to_record_batch(params)?;
        self.backend.insert_into_table(CALIBRATION_TABLE, batch).await
    }

    /// The newest parameter row for a (schema, model, primitive), by fit time.
    pub async fn latest(
        &self,
        schema_id: &str,
        model_id: &str,
        primitive: DecisionPrimitive,
    ) -> Result<Option<CalibrationParams>, Status> {
        let history = self.history(schema_id, model_id, primitive).await?;
        Ok(history.into_iter().next())
    }

    /// All parameter versions for a (schema, model, primitive), newest first.
    pub async fn history(
        &self,
        schema_id: &str,
        model_id: &str,
        primitive: DecisionPrimitive,
    ) -> Result<Vec<CalibrationParams>, Status> {
        let sql = format!(
            "SELECT * FROM {CALIBRATION_TABLE} \
             WHERE schema_id = '{}' AND model_id = '{}' AND primitive = '{}' \
             ORDER BY fitted_at DESC",
            escape_sql(schema_id),
            escape_sql(model_id),
            primitive.as_str(),
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        (0..batch.num_rows())
            .map(|row| params_from_row(&batch, row))
            .collect()
    }
}

/// Extract a COUNT(*) result that DuckDB may type as Int64 or UInt64.
fn count_rows(batch: &RecordBatch) -> Result<u64, Status> {
    use duckdb::arrow::array::UInt64Array;
    let col = batch
        .column_by_name("cnt")
        .ok_or_else(|| Status::internal("COUNT(*) result has no cnt column"))?;
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return Ok(arr.value(0).max(0) as u64);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        return Ok(arr.value(0));
    }
    Err(Status::internal(format!(
        "unexpected COUNT(*) column type: {:?}",
        col.data_type()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::duckdb::DuckDbBackend;

    fn params(primitive: DecisionPrimitive, calib_version: &str, fitted_at: i64) -> CalibrationParams {
        CalibrationParams {
            version: VersionTriple {
                schema_id: "jev1:q1".to_owned(),
                model_id: "qwen3.5-0.8b@main".to_owned(),
                calib_version: calib_version.to_owned(),
            },
            primitive,
            temperature: Some(1.25),
            null_bias: Some(vec![0.1, -0.2, 0.05]),
            dirichlet: None,
            conformal: None,
            cardinality: Some(3),
            calib_set_id: "calibset-2026-09".to_owned(),
            calib_size: Some(10_000),
            objective: Some("nll".to_owned()),
            fitted_at,
        }
    }

    #[test]
    fn test_validate_rejects_bad_rows() {
        let mut p = params(DecisionPrimitive::Choice, "cal-v1", 1);
        p.temperature = Some(0.0);
        assert!(p.validate().is_err());

        let mut p = params(DecisionPrimitive::Noul, "cal-v1", 1);
        p.dirichlet = Some(DirichletParams {
            weights: vec![vec![1.0]],
            bias: vec![0.0],
        });
        assert!(p.validate().is_err(), "dirichlet is choice-only");

        let mut p = params(DecisionPrimitive::Score, "cal-v1", 1);
        p.conformal = Some(ConformalParams::Quantiles {
            quantiles: vec![ConformalQuantile {
                level: 0.9,
                quantile: 0.5,
            }],
        });
        assert!(p.validate().is_err(), "score needs ordinal intervals");

        let mut p = params(DecisionPrimitive::Choice, "cal-v1", 1);
        p.conformal = Some(ConformalParams::OrdinalIntervals {
            intervals: vec![OrdinalInterval {
                level: 0.9,
                lower: 1.0,
                upper: 1.0,
            }],
        });
        assert!(p.validate().is_err(), "choice needs quantiles");

        let mut p = params(DecisionPrimitive::Choice, "cal-v1", 1);
        p.version.calib_version = String::new();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_minted_versions_are_unique() {
        assert_ne!(mint_calib_version(), mint_calib_version());
    }

    #[tokio::test]
    async fn test_insert_and_latest_round_trip() -> Result<(), Status> {
        let backend = Arc::new(DuckDbBackend::new_in_memory()?);
        let store = CalibrationStore::new(backend);
        store.init().await?;

        let mut older = params(DecisionPrimitive::Choice, "cal-v1", 1_000);
        older.dirichlet = Some(DirichletParams {
            weights: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
            bias: vec![0.1, 0.0, -0.1],
        });
        older.conformal = Some(ConformalParams::Quantiles {
            quantiles: vec![
                ConformalQuantile { level: 0.8, quantile: 0.31 },
                ConformalQuantile { level: 0.9, quantile: 0.47 },
            ],
        });
        store.insert(&older).await?;

        let mut newer = params(DecisionPrimitive::Choice, "cal-v2", 2_000);
        newer.temperature = Some(0.9);
        store.insert(&newer).await?;

        let latest = store
            .latest("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Choice)
            .await?
            .ok_or_else(|| Status::internal("latest row missing"))?;
        assert_eq!(latest.version.calib_version, "cal-v2");
        assert_eq!(latest.temperature, Some(0.9));

        let history = store
            .history("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Choice)
            .await?;
        assert_eq!(history.len(), 2);
        assert_eq!(history[1], older, "older row keeps its exact params");

        // Score rows are separate per primitive.
        let mut score = params(DecisionPrimitive::Score, "cal-v1", 1_500);
        score.conformal = Some(ConformalParams::OrdinalIntervals {
            intervals: vec![OrdinalInterval {
                level: 0.9,
                lower: 1.0,
                upper: 2.0,
            }],
        });
        store.insert(&score).await?;
        let score_latest = store
            .latest("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Score)
            .await?
            .ok_or_else(|| Status::internal("score row missing"))?;
        assert_eq!(
            score_latest.conformal,
            Some(ConformalParams::OrdinalIntervals {
                intervals: vec![OrdinalInterval {
                    level: 0.9,
                    lower: 1.0,
                    upper: 2.0,
                }],
            })
        );

        // Unknown (schema, model, primitive) reads as empty.
        assert!(store
            .latest("jev1:other", "qwen3.5-0.8b@main", DecisionPrimitive::Noul)
            .await?
            .is_none());
        Ok(())
    }

    /// Regression: nullable Int64/Float64 columns must persist as SQL NULL,
    /// not the zeroed buffer slot of `array.value()` on a null.
    #[tokio::test]
    async fn test_none_numeric_params_round_trip_as_null() -> Result<(), Status> {
        let backend = Arc::new(DuckDbBackend::new_in_memory()?);
        let store = CalibrationStore::new(backend);
        store.init().await?;

        let mut row = params(DecisionPrimitive::Choice, "cal-v1", 1_000);
        row.temperature = None; // None = identity
        row.cardinality = None;
        row.calib_size = None;
        row.validate()?;
        store.insert(&row).await?;

        let back = store
            .latest("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Choice)
            .await?
            .ok_or_else(|| Status::internal("row missing"))?;
        assert_eq!(back.temperature, None, "identity temperature must stay NULL");
        assert_eq!(back.cardinality, None);
        assert_eq!(back.calib_size, None);
        assert_eq!(back.null_bias, row.null_bias, "set fields still round-trip");
        // The store must not persist rows its own validation would reject.
        back.validate()
    }

    #[tokio::test]
    async fn test_insert_rejects_duplicate_version_triple() -> Result<(), Status> {
        let backend = Arc::new(DuckDbBackend::new_in_memory()?);
        let store = CalibrationStore::new(backend);
        store.init().await?;

        let row = params(DecisionPrimitive::Noul, "cal-v1", 1_000);
        store.insert(&row).await?;
        let err = match store.insert(&row).await {
            Err(err) => err,
            Ok(()) => {
                return Err(Status::internal("duplicate version triple must be rejected"));
            }
        };
        assert_eq!(err.code(), tonic::Code::AlreadyExists);

        let mut refit = row.clone();
        refit.calib_size = Some(20_000);
        refit.version.calib_version = mint_calib_version();
        store.insert(&refit).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_insert_rejects_invalid_params() -> Result<(), Status> {
        let backend = Arc::new(DuckDbBackend::new_in_memory()?);
        let store = CalibrationStore::new(backend);
        store.init().await?;

        let mut bad = params(DecisionPrimitive::Score, "cal-v1", 1_000);
        bad.temperature = Some(-1.0);
        assert!(store.insert(&bad).await.is_err());
        // Nothing was written.
        assert!(store
            .latest("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Score)
            .await?
            .is_none());
        Ok(())
    }
}
