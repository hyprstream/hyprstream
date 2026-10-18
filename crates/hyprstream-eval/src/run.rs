//! Running evals: eval sets/items in, observation streams and Arrow batches out.
//!
//! Two input shapes:
//!
//! - [`EvalSet`] — the **workflow** shape: one declared question set applied
//!   to many state rows. This is what P0.5/P1.4 run; the run's Arrow batch is
//!   a single `hyprstream-decision` `DecisionSchema` batch, persistable as one
//!   P0.3 `decision_batches` row.
//! - [`EvalItem`] — the **one-question** shape: a question, its state, and its
//!   verifiable truth. [`hyprstream_bench::Item`] converts into this; scoring
//!   groups the resulting observations into fields (family × kind ×
//!   cardinality) because calibration metrics need probability vectors and
//!   truth indices, not shared label sets.

use hyprstream_decision::answer::VersionTriple;
use hyprstream_decision::arrow::DecisionSchema;
use hyprstream_decision::entry::Entry;
use hyprstream_decision::spec::{QuestionKind, QuestionSet, QuestionSpec};
use hyprstream_metrics_api::arrow::array::RecordBatch;

use crate::error::EvalError;
use crate::subject::Subject;

/// One row of an [`EvalSet`]: a state plus per-question truth (label index in
/// canonical order) where the outcome is verifiable.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalRow {
    /// Stable row id (flows into observations and reports).
    pub id: String,
    /// Workflow family this row belongs to (P0.2 shift-split / macro-rule
    /// grouping); free-form for non-bench evals.
    pub family: Option<String>,
    /// Difficulty/robustness stratum label (`clean`, `nearmiss`, `perm-k`, …).
    pub stratum: Option<String>,
    /// Permutation group linking cyclic rotations of one item (S6b1 flip-rate
    /// input); rows not in a permutation group leave this `None`.
    pub group: Option<String>,
    /// The state the questions are judged against.
    pub state: Entry,
    /// Verifiable truth per question id (label index); absent = unlabeled row.
    pub truth: std::collections::BTreeMap<String, usize>,
}

/// A declared question set evaluated over many rows — one persisted batch.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalSet {
    /// Human/report name of the set (e.g. a workflow or release id).
    pub name: String,
    /// Question-set schema version (version-triple column; bump on any
    /// label-set evolution — the Arrow contract fails loudly otherwise).
    pub schema_version: String,
    /// The declared questions.
    pub questions: Vec<QuestionSpec>,
    /// The rows.
    pub rows: Vec<EvalRow>,
}

impl EvalSet {
    /// The question-set view (no shared state; states are per-row).
    pub fn question_set(&self) -> QuestionSet {
        QuestionSet {
            state: None,
            questions: self.questions.clone(),
        }
    }
}

/// A one-question eval item: question, state, verifiable truth. This is the
/// shape [`hyprstream_bench::Item`] converts into; runs over items produce
/// observations (scoring input) but no single shared-schema batch — items
/// can carry different label sets, so metadata binds to each (item id,
/// question id) pair rather than a single shared schema.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalItem {
    /// Stable item id.
    pub id: String,
    /// Family label (grouping/reporting).
    pub family: Option<String>,
    /// Stratum label (grouping/reporting).
    pub stratum: Option<String>,
    /// Permutation group (flip-rate input).
    pub group: Option<String>,
    /// The state.
    pub state: Entry,
    /// The question.
    pub question: QuestionSpec,
    /// Verifiable truth: label index in canonical order.
    pub truth: Option<usize>,
}

impl From<&hyprstream_bench::Item> for EvalItem {
    fn from(item: &hyprstream_bench::Item) -> Self {
        Self {
            id: item.id.clone(),
            family: Some(item.family.as_str().to_owned()),
            stratum: Some(item.stratum.as_str()),
            group: Some(item.group.clone()),
            state: item.state.clone(),
            question: item.question.clone(),
            truth: Some(item.truth),
        }
    }
}

/// One scored unit: a subject's distribution over one question on one row,
/// with the tags reports group by.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Item/row id.
    pub id: String,
    /// Question id.
    pub question_id: String,
    /// Question primitive.
    pub kind: QuestionKind,
    /// Family tag (if any).
    pub family: Option<String>,
    /// Stratum tag (if any).
    pub stratum: Option<String>,
    /// Permutation group (if any).
    pub group: Option<String>,
    /// The subject's distribution in canonical label order; `None` = the
    /// subject abstained.
    pub probabilities: Option<Vec<f32>>,
    /// Verifiable truth label index, where known.
    pub truth: Option<usize>,
}

impl Observation {
    /// The canonical labels this distribution is over (needs the question;
    /// kept on the observation as denormalized report input).
    pub fn argmax_label_index(&self) -> Option<usize> {
        self.probabilities
            .as_deref()
            .and_then(hyprstream_decision::confidence::argmax_index)
    }
}

/// What a run produced: the observation stream, the version triple, and — for
/// [`EvalSet`] runs — the Arrow batch under the set's decision schema.
#[derive(Debug)]
pub struct RunOutput {
    /// The subject's resolved model id.
    pub model_id: String,
    /// The batch version triple used for persistence.
    pub version: VersionTriple,
    /// One observation per (row, question), in run order.
    pub observations: Vec<Observation>,
    /// The emitted batch (`None` for loose item runs).
    pub batch: Option<RecordBatch>,
    /// The decision-schema fingerprint of the batch, when present.
    pub schema_fingerprint: Option<String>,
}

impl RunOutput {
    /// Both public model coordinates name the same resolved producer. Check
    /// them again at consuming boundaries because callers may mutate a run.
    pub(crate) fn validate_model_provenance(&self) -> Result<(), EvalError> {
        if self.model_id.is_empty() || self.model_id != self.version.model {
            return Err(EvalError::InvalidInput(
                "run requires one nonempty resolved model id matching its version triple"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// The harness. Stateless; every method is a pure function of its inputs plus
/// the subject's behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct Harness;

impl Harness {
    /// Run an [`EvalSet`] against a subject: one `decide` per row, each row's
    /// answers validated immediately after its call (presence, kind,
    /// cardinality, producer tolerance, abstention/conformal-set invariants —
    /// the same rules the Arrow batch builder enforces), plus the observation
    /// stream. The decision schema and every row's truth keys are validated
    /// **before** the first subject call (an HTTP-backed model bills per
    /// row). The version triple records the subject's **resolved** model id
    /// from the first response (HTTP alias resolution); a response resolving
    /// to a different version mid-run (rolling deploy) fails the run.
    pub async fn run_set(
        &self,
        set: &EvalSet,
        subject: &dyn Subject,
    ) -> Result<RunOutput, EvalError> {
        if set.schema_version.is_empty() {
            return Err(EvalError::InvalidInput(
                "an eval set requires a nonempty schema version".to_owned(),
            ));
        }
        let question_set = set.question_set();
        // Validate the decision schema BEFORE any row reaches the subject:
        // with an HTTP-backed model every row is a billed call, and a schema
        // error (duplicate/reserved/identifier-unsafe question ids) is
        // deterministic input rejection, not a runtime failure.
        let schema = DecisionSchema::new(set.questions.clone())?;
        // The schema check covers Arrow column ids only; the full jev-1 v1
        // question profile (v1 kinds, kind/body agreement, cardinalities,
        // unique nonempty labels) is validated here too, before any billed
        // subject call.
        for question in &set.questions {
            validate_question_profile(question).map_err(EvalError::InvalidInput)?;
        }
        // A truth entry keyed to an undeclared question would otherwise be
        // silently ignored (the row scored as unlabeled), biasing every
        // labeled denominator — fail loudly instead. A truth index outside
        // the question's cardinality is likewise an input error, caught here
        // rather than at scoring time (after every billed call).
        let mut row_ids = std::collections::HashSet::with_capacity(set.rows.len());
        for row in &set.rows {
            validate_entry(&row.state).map_err(EvalError::InvalidInput)?;
            if !row_ids.insert(&row.id) {
                return Err(EvalError::InvalidInput(format!(
                    "duplicate eval row id `{}`",
                    row.id
                )));
            }
            for (question_id, &label) in &row.truth {
                let Some(question) = question_set.question(question_id) else {
                    return Err(EvalError::InvalidInput(format!(
                        "row `{}` has a truth entry for undeclared question `{question_id}`",
                        row.id
                    )));
                };
                if label >= question.cardinality() {
                    return Err(EvalError::InvalidInput(format!(
                        "row `{}` truth index {label} is outside question `{question_id}`'s cardinality {}",
                        row.id,
                        question.cardinality()
                    )));
                }
            }
        }
        // Subject-specific checks must also complete for ALL requests before
        // any decide call (e.g. the legacy HTTP endpoint rejects null state).
        for row in &set.rows {
            subject.preflight(&question_set, &row.state)?;
        }
        let mut rows = Vec::with_capacity(set.rows.len());
        let mut observations = Vec::new();
        // Pin the resolved model id from the FIRST response: an HTTP subject
        // re-resolves its alias per response, and a rolling deploy mid-run
        // would otherwise stamp every row with the LAST resolved version,
        // mislabeling the earlier rows. The version is read from the same
        // exchange as the answers (`decide_with_version`) so a concurrent
        // run sharing the subject cannot overwrite it in between.
        let mut pinned_model: Option<String> = None;
        for (row_index, row) in set.rows.iter().enumerate() {
            let (answers, resolved) = subject
                .decide_with_version(&question_set, &row.state, row_index)
                .await
                .map_err(|error| EvalError::Subject {
                    model: subject.model_id().to_owned(),
                    item_id: row.id.clone(),
                    message: error.to_string(),
                })?;
            // Validate THIS row's answers before requesting the next one: a
            // malformed answer is a deterministic failure, and with an
            // HTTP-backed subject every subsequent row is a billed call
            // (build_batch would only report it after the whole run).
            validate_set_answer_row(&question_set, &answers).map_err(|message| {
                EvalError::InvalidAnswer {
                    question_id: row.id.clone(),
                    message,
                }
            })?;
            // An empty resolved id is no provenance at all: a set run would
            // only fail later in BatchSink::persist_run, and item reports
            // would carry the blank id. Reject it at capture, whatever the
            // subject implementation (HTTP responses are already required
            // to carry a nonempty `model`).
            if resolved.is_empty() {
                return Err(EvalError::InvalidInput(format!(
                    "subject `{}` resolved an empty model id on row `{}` — a run requires model provenance",
                    subject.model_id(),
                    row.id
                )));
            }
            match &pinned_model {
                None => pinned_model = Some(resolved),
                Some(pinned) if *pinned != resolved => {
                    return Err(EvalError::InvalidInput(format!(
                        "model version changed mid-run: `{pinned}` then `{resolved}` (rolling deploy?) — rerun against a stable deployment"
                    )));
                }
                Some(_) => {}
            }
            for question in &set.questions {
                let probabilities = answers
                    .answers
                    .get(&question.id)
                    .and_then(|answer| answer.value.as_ref())
                    .map(hyprstream_decision::answer::AnswerValue::probabilities);
                observations.push(Observation {
                    id: row.id.clone(),
                    question_id: question.id.clone(),
                    kind: question.kind,
                    family: row.family.clone(),
                    stratum: row.stratum.clone(),
                    group: row.group.clone(),
                    probabilities,
                    truth: row.truth.get(&question.id).copied(),
                });
            }
            rows.push(answers);
        }
        let version = VersionTriple {
            schema: set.schema_version.clone(),
            model: final_model_id(pinned_model, subject)?,
            calib: None,
        };
        let batch = schema.build_batch(&version, &rows)?;
        Ok(RunOutput {
            model_id: version.model.clone(),
            schema_fingerprint: Some(schema.fingerprint()),
            batch: Some(batch),
            version,
            observations,
        })
    }

    /// Run loose [`EvalItem`]s against a subject (the bench shape). Each item
    /// is answered as its own one-question request; observations carry the
    /// item's family/stratum/group tags through to scoring. Answers are
    /// validated with the same rules the Arrow batch builder enforces for
    /// [`Harness::run_set`] (presence, kind, cardinality, producer tolerance,
    /// abstention/conformal-set invariants) — a malformed subject fails
    /// loudly here too.
    pub async fn run_items(
        &self,
        items: &[EvalItem],
        subject: &dyn Subject,
    ) -> Result<RunOutput, EvalError> {
        // Preflight EVERY item's truth index before the first subject call:
        // item N's invalid truth must not surface only after the preceding
        // N−1 items have invoked (and possibly billed) the subject.
        let mut identities = std::collections::HashSet::with_capacity(items.len());
        for item in items {
            validate_entry(&item.state).map_err(EvalError::InvalidInput)?;
            if !identities.insert((&item.id, &item.question.id)) {
                return Err(EvalError::InvalidInput(format!(
                    "duplicate eval item identity ({:?}, {:?})",
                    item.id, item.question.id
                )));
            }
            validate_question_profile(&item.question).map_err(EvalError::InvalidInput)?;
            if let Some(truth) = item.truth {
                if truth >= item.question.cardinality() {
                    return Err(EvalError::InvalidInput(format!(
                        "item `{}` truth index {truth} is outside question `{}`'s cardinality {}",
                        item.id,
                        item.question.id,
                        item.question.cardinality()
                    )));
                }
            }
        }
        for item in items {
            let set = QuestionSet {
                state: Some(item.state.clone()),
                questions: vec![item.question.clone()],
            };
            subject.preflight(&set, &item.state)?;
        }
        let mut observations = Vec::with_capacity(items.len());
        let mut pinned_model: Option<String> = None;
        for (row_index, item) in items.iter().enumerate() {
            let set = QuestionSet {
                state: Some(item.state.clone()),
                questions: vec![item.question.clone()],
            };
            let (answers, resolved) = subject
                .decide_with_version(&set, &item.state, row_index)
                .await
                .map_err(|error| EvalError::Subject {
                    model: subject.model_id().to_owned(),
                    item_id: item.id.clone(),
                    message: error.to_string(),
                })?;
            if resolved.is_empty() {
                return Err(EvalError::InvalidInput(format!(
                    "subject `{}` resolved an empty model id on item `{}` — a run requires model provenance",
                    subject.model_id(),
                    item.id
                )));
            }
            match &pinned_model {
                None => pinned_model = Some(resolved),
                Some(pinned) if *pinned != resolved => {
                    return Err(EvalError::InvalidInput(format!(
                        "model version changed mid-run: `{pinned}` then `{resolved}` (rolling deploy?) — rerun against a stable deployment"
                    )));
                }
                Some(_) => {}
            }
            validate_answer_row(&item.question, &answers).map_err(|message| {
                EvalError::InvalidAnswer {
                    question_id: item.question.id.clone(),
                    message,
                }
            })?;
            let probabilities = answers
                .answers
                .get(&item.question.id)
                .and_then(|answer| answer.value.as_ref())
                .map(hyprstream_decision::answer::AnswerValue::probabilities);
            observations.push(Observation {
                id: item.id.clone(),
                question_id: item.question.id.clone(),
                kind: item.question.kind,
                family: item.family.clone(),
                stratum: item.stratum.clone(),
                group: item.group.clone(),
                probabilities,
                truth: item.truth,
            });
        }
        let version = VersionTriple {
            schema: "items".to_owned(),
            model: final_model_id(pinned_model, subject)?,
            calib: None,
        };
        Ok(RunOutput {
            model_id: version.model.clone(),
            version,
            observations,
            batch: None,
            schema_fingerprint: None,
        })
    }
}

/// Empty runs have no response capture, but still require usable provenance.
fn final_model_id(pinned: Option<String>, subject: &dyn Subject) -> Result<String, EvalError> {
    let model = pinned.unwrap_or_else(|| subject.resolved_model_id());
    if model.is_empty() {
        return Err(EvalError::InvalidInput(format!(
            "subject `{}` resolved an empty model id — a run requires model provenance",
            subject.model_id()
        )));
    }
    Ok(model)
}

/// Programmatic entries bypass the parser's finite-number invariant. Check
/// recursively during all-input preflight, before any subject can be billed.
pub(crate) fn validate_entry(entry: &Entry) -> Result<(), String> {
    match entry {
        Entry::Number(value) if !value.is_finite() => {
            Err("non-finite Entry::Number in eval input".to_owned())
        }
        Entry::Seq(entries) => entries.iter().try_for_each(validate_entry),
        Entry::Map(entries) => entries
            .iter()
            .try_for_each(|(_, value)| validate_entry(value)),
        _ => Ok(()),
    }
}

/// The jev-1 v1 question-profile rules the authoring layer enforces at
/// parse time, applied to programmatically constructed questions (which
/// bypass parsing): v1 kinds only, kind/body agreement, choice cardinality
/// 2–255 with unique nonempty option names, score cardinality ≥ 2.
/// `DecisionSchema::new` only validates Arrow column ids, so this runs in
/// both run preflights, before the first subject call.
pub(crate) fn validate_question_profile(question: &QuestionSpec) -> Result<(), String> {
    use hyprstream_decision::spec::{QuestionBody, QuestionKind};

    if let Some(instructions) = &question.instructions {
        validate_entry(instructions)?;
    }
    let id = &question.id;
    if question.id.is_empty() {
        return Err("a question id must be nonempty".to_owned());
    }
    if !question.kind.is_v1() {
        return Err(format!(
            "question `{id}` uses `{}`, reserved for profile v2",
            question.kind
        ));
    }
    let body_kind = match &question.body {
        QuestionBody::Noul { .. } => QuestionKind::Noul,
        QuestionBody::Choice { .. } => QuestionKind::Choice,
        QuestionBody::Score { .. } => QuestionKind::Score,
    };
    if body_kind != question.kind {
        return Err(format!(
            "question `{id}` declares kind {} but carries a {body_kind} body",
            question.kind
        ));
    }
    match &question.body {
        QuestionBody::Choice { options } => {
            if options.len() < 2 {
                return Err(format!(
                    "choice question `{id}` needs at least 2 options, got {}",
                    options.len()
                ));
            }
            if options.len() > hyprstream_decision::MAX_CHOICE_OPTIONS {
                return Err(format!(
                    "choice question `{id}` supports at most {} options, got {}",
                    hyprstream_decision::MAX_CHOICE_OPTIONS,
                    options.len()
                ));
            }
            let mut names = std::collections::HashSet::with_capacity(options.len());
            for option in options {
                if let Some(rubric) = &option.rubric {
                    validate_entry(rubric)?;
                }
                if option.name.is_empty() {
                    return Err(format!("choice question `{id}` has an empty option name"));
                }
                if !names.insert(&option.name) {
                    return Err(format!(
                        "choice question `{id}` repeats option `{}`",
                        option.name
                    ));
                }
            }
        }
        QuestionBody::Score { levels } => {
            for rubric in levels.iter().flatten() {
                validate_entry(rubric)?;
            }
            if levels.len() < 2 {
                return Err(format!(
                    "score question `{id}` needs at least 2 levels, got {}",
                    levels.len()
                ));
            }
        }
        QuestionBody::Noul { criteria } => {
            if let Some(criteria) = criteria {
                for rubric in [&criteria.on_true, &criteria.on_false]
                    .into_iter()
                    .flatten()
                {
                    validate_entry(rubric)?;
                }
            }
        }
    }
    Ok(())
}

/// The same row-level answer rules `DecisionSchema::build_batch` enforces,
/// for one-question item runs (bench question ids are not identifier-safe —
/// they contain `-` — so they cannot go through the Arrow validator itself).
pub(crate) fn validate_answer_row(
    question: &QuestionSpec,
    row: &hyprstream_decision::arrow::AnswerRow,
) -> Result<(), String> {
    let id = &question.id;
    for answered_id in row.answers.keys() {
        if answered_id != id {
            return Err(format!("row answers unknown question `{answered_id}`"));
        }
    }
    validate_one_answer(question, row)
}

/// Index declared IDs once without changing their original order. Direct
/// programmatic inputs must not collapse duplicate declarations into one map key.
pub(crate) fn question_ids(set: &QuestionSet) -> Result<std::collections::HashSet<&str>, String> {
    if set.questions.is_empty() {
        return Err("a question set requires at least one question".to_owned());
    }
    let mut ids = std::collections::HashSet::with_capacity(set.questions.len());
    for question in &set.questions {
        if !ids.insert(question.id.as_str()) {
            return Err(format!("duplicate question id `{}`", question.id));
        }
    }
    Ok(ids)
}

/// The same rules for a full row over a question set (the teacher ensemble's
/// shape): every answered key must be declared, and every declared question
/// must be answered per the per-answer rules.
pub(crate) fn validate_set_answer_row(
    set: &hyprstream_decision::spec::QuestionSet,
    row: &hyprstream_decision::arrow::AnswerRow,
) -> Result<(), String> {
    let ids = question_ids(set)?;
    for answered_id in row.answers.keys() {
        if !ids.contains(answered_id.as_str()) {
            return Err(format!("row answers unknown question `{answered_id}`"));
        }
    }
    for question in &set.questions {
        validate_one_answer(question, row)?;
    }
    Ok(())
}

/// Per-question answer rules: presence, abstention shape, kind match,
/// cardinality, producer tolerance, conformal labels.
fn validate_one_answer(
    question: &QuestionSpec,
    row: &hyprstream_decision::arrow::AnswerRow,
) -> Result<(), String> {
    use hyprstream_decision::answer::AnswerValue;
    use hyprstream_decision::confidence::{check_distribution, PRODUCER_SUM_TOLERANCE};
    use hyprstream_decision::spec::QuestionKind;

    let id = &question.id;
    let answer = row.answers.get(id).ok_or_else(|| {
        format!("row is missing an answer for question `{id}` (abstain explicitly instead)")
    })?;
    let Some(value) = &answer.value else {
        if answer.conformal_set.is_some() {
            return Err("an abstained answer cannot carry a conformal set".to_owned());
        }
        return Ok(());
    };
    // Match against the DECLARED kind, not the body variant: a
    // programmatically built question whose public `kind` and `body`
    // disagree is scored under `kind`, so the answer must agree with `kind`
    // too (the same rule `DecisionSchema` enforces).
    let answer_kind = match value {
        AnswerValue::Noul { .. } => QuestionKind::Noul,
        AnswerValue::Choice { .. } => QuestionKind::Choice,
        AnswerValue::Score { .. } => QuestionKind::Score,
    };
    if answer_kind != question.kind {
        return Err(format!(
            "answer kind {answer_kind} does not match question kind {}",
            question.kind
        ));
    }
    let probabilities = value.probabilities();
    if probabilities.len() != question.cardinality() {
        return Err(format!(
            "{} probabilities for a cardinality-{} question",
            probabilities.len(),
            question.cardinality()
        ));
    }
    check_distribution(&probabilities, PRODUCER_SUM_TOLERANCE)
        .map_err(|error| format!("distribution violates producer tolerance (D5): {error}"))?;
    if let Some(set) = &answer.conformal_set {
        let labels = question.labels();
        for label in set {
            if !labels.contains(label) {
                return Err(format!(
                    "conformal set member `{label}` is not one of the question's labels"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod row_identity_tests {
    use super::*;
    use hyprstream_decision::answer::QuestionAnswer;
    use hyprstream_decision::arrow::AnswerRow;
    use hyprstream_decision::author;

    #[test]
    fn row_validation_rejects_duplicate_declarations_and_exact_key_substitution(
    ) -> Result<(), String> {
        let set = author::parse_yaml("questions:\n  left: {type: noul}\n  right: {type: noul}\n")
            .map_err(|error| error.to_string())?;
        let valid = AnswerRow {
            answers: set
                .questions
                .iter()
                .map(|question| (question.id.clone(), QuestionAnswer::abstained()))
                .collect(),
        };
        validate_set_answer_row(&set, &valid)?;
        let mut duplicate = set.clone();
        duplicate.questions[1] = duplicate.questions[0].clone();
        let collapsed = AnswerRow {
            answers: [("left".to_owned(), QuestionAnswer::abstained())].into(),
        };
        assert!(
            matches!(validate_set_answer_row(&duplicate, &collapsed), Err(ref message) if message.contains("duplicate question id"))
        );
        let mut replaced = valid.clone();
        replaced.answers.remove("right");
        replaced
            .answers
            .insert("outside".into(), QuestionAnswer::abstained());
        assert!(
            matches!(validate_set_answer_row(&set, &replaced), Err(ref message) if message == "row answers unknown question `outside`")
        );
        let mut missing = valid;
        missing.answers.remove("right");
        assert!(
            matches!(validate_set_answer_row(&set, &missing), Err(ref message) if message.contains("missing an answer for question `right`"))
        );
        Ok(())
    }
}
