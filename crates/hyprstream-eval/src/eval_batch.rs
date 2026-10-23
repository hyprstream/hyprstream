//! Persisted evaluation Arrow contract. Decision/serving batches remain Arrow v1;
//! this eval-owned v2 adds row identity, tags, and canonical per-question truth.
//! Field order is row_id/family/stratum/group, then each question's truth followed
//! by its unchanged v1 answer fields, then the unchanged version triple.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use hyprstream_decision::answer::VersionTriple;
use hyprstream_decision::arrow::{AnswerRow, DecisionSchema};
use hyprstream_metrics_api::arrow::array::{ArrayRef, RecordBatch, StringArray};
use hyprstream_metrics_api::arrow::datatypes::{DataType, Field, Schema};

use crate::error::EvalError;
use crate::run::EvalBatchRow;

pub(crate) const CONTRACT_KEY: &str = "hyprstream.batch_contract";
pub(crate) const CONTRACT_V2: &str = "eval-v2";
const DECISION_FINGERPRINT_KEY: &str = "hyprstream.decision_schema_fingerprint";

/// Evaluation-specific schema layered over the unchanged serving DecisionSchema v1.
#[derive(Debug, Clone)]
pub(crate) struct EvalBatchSchema {
    decision: DecisionSchema,
    schema: Arc<Schema>,
}

impl EvalBatchSchema {
    pub(crate) fn new(decision: DecisionSchema) -> Result<Self, EvalError> {
        let decision_schema = decision.arrow_schema();
        let decision_fields: HashMap<&str, _> = decision_schema
            .fields()
            .iter()
            .map(|field| (field.name().as_str(), field))
            .collect();
        let mut fields =
            Vec::with_capacity(decision_schema.fields().len() + 4 + decision.questions().len());
        fields.push(Arc::new(Field::new("row_id", DataType::Utf8, false)));
        fields.push(Arc::new(Field::new("family", DataType::Utf8, true)));
        fields.push(Arc::new(Field::new("stratum", DataType::Utf8, true)));
        fields.push(Arc::new(Field::new("group", DataType::Utf8, true)));
        for question in decision.questions() {
            fields.push(Arc::new(Field::new(
                format!("{}.truth", question.id),
                DataType::Utf8,
                true,
            )));
            for suffix in ["probabilities", "label", "conformal_set"] {
                let name = format!("{}.{}", question.id, suffix);
                let Some(field) = decision_fields.get(name.as_str()) else {
                    if suffix == "conformal_set" {
                        continue;
                    }
                    return Err(invalid(
                        "canonical decision schema is missing an answer field",
                    ));
                };
                fields.push(Arc::clone(*field));
            }
        }
        for name in ["schema_version", "model_version", "calib_version"] {
            let field = decision_fields
                .get(name)
                .ok_or_else(|| invalid("canonical decision schema is missing a version field"))?;
            fields.push(Arc::clone(*field));
        }
        let mut metadata = HashMap::new();
        metadata.insert(CONTRACT_KEY.to_owned(), CONTRACT_V2.to_owned());
        metadata.insert(DECISION_FINGERPRINT_KEY.to_owned(), decision.fingerprint());
        Ok(Self {
            decision,
            schema: Arc::new(Schema::new(fields).with_metadata(metadata)),
        })
    }

    /// Physical eval-v2 fingerprint. The domain, schema metadata, every field's
    /// name/type/nullability/metadata participate; v1's decision fingerprint is
    /// separately embedded and keeps its original algorithm unchanged.
    pub(crate) fn fingerprint(&self) -> String {
        physical_fingerprint(&self.schema)
    }

    pub(crate) fn build_batch(
        &self,
        version: &VersionTriple,
        rows: &[EvalBatchRow],
    ) -> Result<RecordBatch, EvalError> {
        let mut ids = HashSet::with_capacity(rows.len());
        let questions_by_id: HashMap<&str, _> = self
            .decision
            .questions()
            .iter()
            .map(|question| (question.id.as_str(), question))
            .collect();
        let question_indexes: HashMap<&str, usize> = self
            .decision
            .questions()
            .iter()
            .enumerate()
            .map(|(index, question)| (question.id.as_str(), index))
            .collect();
        let mut truths_by_row = vec![vec![None; self.decision.questions().len()]; rows.len()];
        for (row_index, row) in rows.iter().enumerate() {
            if row.id.is_empty() {
                return Err(invalid("evaluation row ID must be nonempty"));
            }
            if !ids.insert(row.id.as_str()) {
                return Err(invalid("evaluation row IDs must be unique"));
            }
            for (question_id, &truth) in &row.truth {
                let question = questions_by_id
                    .get(question_id.as_str())
                    .ok_or_else(|| invalid("evaluation truth references an unknown question"))?;
                if truth >= question.cardinality() {
                    return Err(invalid("evaluation truth is outside canonical labels"));
                }
                let question_index = question_indexes[question_id.as_str()];
                truths_by_row[row_index][question_index] = Some(truth);
            }
        }

        let answer_rows = rows
            .iter()
            .map(|row| AnswerRow {
                answers: row.answers.clone(),
            })
            .collect::<Vec<_>>();
        let answer_batch = self.decision.build_batch(version, &answer_rows)?;
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.schema.fields().len());
        columns.push(Arc::new(StringArray::from(
            rows.iter()
                .map(|row| Some(row.id.as_str()))
                .collect::<Vec<_>>(),
        )));
        columns.push(Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.family.as_deref())
                .collect::<Vec<_>>(),
        )));
        columns.push(Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.stratum.as_deref())
                .collect::<Vec<_>>(),
        )));
        columns.push(Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.group.as_deref())
                .collect::<Vec<_>>(),
        )));

        let has_conformal = self
            .decision
            .arrow_schema()
            .fields()
            .iter()
            .any(|field| field.name().ends_with(".conformal_set"));
        let mut answer_column = 0;
        for (question_index, question) in self.decision.questions().iter().enumerate() {
            let labels = question.labels();
            let truth = truths_by_row
                .iter()
                .map(|row| row[question_index].map(|index| labels[index].as_str()))
                .collect::<Vec<_>>();
            columns.push(Arc::new(StringArray::from(truth)));
            for _ in 0..if has_conformal { 3 } else { 2 } {
                columns.push(answer_batch.column(answer_column).clone());
                answer_column += 1;
            }
        }
        while answer_column < answer_batch.num_columns() {
            columns.push(answer_batch.column(answer_column).clone());
            answer_column += 1;
        }
        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|error| invalid(&format!("eval-v2 Arrow batch assembly failed: {error}")))
    }
}

pub(crate) fn physical_fingerprint(schema: &Schema) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"hyprstream evaluation Arrow contract v2\0");
    for (key, value) in sorted_metadata(schema.metadata()) {
        hash_part(&mut hasher, key);
        hash_part(&mut hasher, value);
    }
    hasher.update(b"\xff");
    for field in schema.fields() {
        hash_part(&mut hasher, field.name());
        hash_part(&mut hasher, &format!("{:?}", field.data_type()));
        hasher.update(&[u8::from(field.is_nullable())]);
        for (key, value) in sorted_metadata(field.metadata()) {
            hash_part(&mut hasher, key);
            hash_part(&mut hasher, value);
        }
        hasher.update(b"\xfe");
    }
    hasher.finalize().to_hex().to_string()
}

fn sorted_metadata(metadata: &HashMap<String, String>) -> Vec<(&str, &str)> {
    let mut entries = metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect::<Vec<_>>();
    entries.sort_unstable();
    entries
}

fn hash_part(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn invalid(message: &str) -> EvalError {
    EvalError::InvalidInput(message.to_owned())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use hyprstream_decision::answer::{AnswerValue, QuestionAnswer};
    use hyprstream_decision::spec::{QuestionKind, QuestionSpec};

    fn noul_question(id: String) -> QuestionSpec {
        crate::run::question_from_labels(
            &id,
            QuestionKind::Noul,
            &["false".to_owned(), "true".to_owned()],
        )
        .expect("valid two-label noul question")
    }

    fn answer_rows(question_ids: &[String], count: usize) -> Vec<EvalBatchRow> {
        (0..count)
            .map(|row| {
                let answers = question_ids
                    .iter()
                    .map(|id| {
                        (
                            id.clone(),
                            QuestionAnswer::answered(AnswerValue::Noul {
                                p_true: (row % 2) as f32,
                            }),
                        )
                    })
                    .collect();
                let truth = question_ids
                    .iter()
                    .map(|id| (id.clone(), row % 2))
                    .collect();
                EvalBatchRow {
                    id: format!("row-{row}"),
                    family: None,
                    stratum: None,
                    group: None,
                    answers,
                    truth,
                }
            })
            .collect()
    }

    #[test]
    fn dense_wide_truth_batch_builds_with_expected_canonical_values() {
        // A deterministic wide/dense fixture guards the lookup path's behavior
        // without relying on wall-clock thresholds that vary across machines.
        let question_ids = (0..192)
            .map(|index| format!("q{index:03}"))
            .collect::<Vec<_>>();
        let schema = EvalBatchSchema::new(
            DecisionSchema::new(question_ids.iter().cloned().map(noul_question).collect()).unwrap(),
        )
        .unwrap();
        let rows = answer_rows(&question_ids, 48);
        let batch = schema
            .build_batch(
                &VersionTriple {
                    schema: "dense-test-v1".into(),
                    model: "dense-test-model".into(),
                    calib: None,
                },
                &rows,
            )
            .unwrap();

        assert_eq!(batch.num_rows(), 48);
        assert_eq!(batch.num_columns(), 4 + 192 * 4 + 3);
        for q in ["q000", "q095", "q191"] {
            let truth = batch
                .column_by_name(&format!("{q}.truth"))
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            assert_eq!(truth.value(0), "false");
            assert_eq!(truth.value(1), "true");
            assert_eq!(truth.value(47), "true");
        }
    }

    #[test]
    fn optional_conformal_schema_remains_supported_in_eval_v2() {
        let question = noul_question("binary_decision".into());
        let decision = DecisionSchema::new(vec![question])
            .unwrap()
            .without_conformal_set();
        let schema = EvalBatchSchema::new(decision).unwrap();
        assert!(schema
            .schema
            .fields()
            .iter()
            .all(|field| !field.name().ends_with(".conformal_set")));

        let questions = vec!["binary_decision".to_owned()];
        let batch = schema
            .build_batch(
                &VersionTriple {
                    schema: "optional-conformal-v1".into(),
                    model: "optional-conformal-model".into(),
                    calib: None,
                },
                &answer_rows(&questions, 1),
            )
            .unwrap();
        assert_eq!(batch.schema().as_ref(), schema.schema.as_ref());
    }
}
