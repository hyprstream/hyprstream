//! Shared typed content identity for exact cross-source contamination checks.
//!
//! This is ADDITIVE: frozen `vob-item/1` artifact hashes are unchanged. The
//! projection omits provenance, truth and opaque question IDs, preserves Entry
//! types/presence and authored option/map order, and does not equate paraphrases
//! or other semantically equivalent but structurally different requests.
use hyprstream_decision::{Entry, QuestionBody, QuestionKind, QuestionSpec};

/// Domain/version of the shared benchmark/synthesis content projection.
pub const CONTENT_FINGERPRINT_VERSION: &str = "decision-content/1";

/// Malformed typed content cannot be attested as a valid fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentFingerprintError {
    /// Constructed Entry contained a nonfinite number.
    NonfiniteNumber,
    /// The question's declared kind disagrees with its body.
    KindMismatch,
}
impl std::fmt::Display for ContentFingerprintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NonfiniteNumber => "nonfinite number in content fingerprint",
            Self::KindMismatch => "question kind/body mismatch in content fingerprint",
        })
    }
}
impl std::error::Error for ContentFingerprintError {}

fn count(out: &mut Vec<u8>, n: usize) {
    out.extend_from_slice(&(n as u64).to_le_bytes());
}
fn text(out: &mut Vec<u8>, value: &str) {
    count(out, value.len());
    out.extend_from_slice(value.as_bytes());
}
fn optional(out: &mut Vec<u8>, value: Option<&Entry>) -> Result<(), ContentFingerprintError> {
    match value {
        None => out.push(0),
        Some(v) => {
            out.push(1);
            entry(out, v)?;
        }
    }
    Ok(())
}
fn entry(out: &mut Vec<u8>, value: &Entry) -> Result<(), ContentFingerprintError> {
    match value {
        Entry::Null => out.push(0),
        Entry::Bool(v) => {
            out.push(1);
            out.push(u8::from(*v));
        }
        Entry::Number(v) => {
            if !v.is_finite() {
                return Err(ContentFingerprintError::NonfiniteNumber);
            }
            out.push(2);
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        Entry::Str(v) => {
            out.push(3);
            text(out, v);
        }
        Entry::Seq(v) => {
            out.push(4);
            count(out, v.len());
            for x in v {
                entry(out, x)?;
            }
        }
        Entry::Map(v) => {
            out.push(5);
            count(out, v.len());
            for (k, x) in v {
                text(out, k);
                entry(out, x)?;
            }
        }
    }
    Ok(())
}

/// Exact typed state/question content identity, excluding opaque question ID.
/// Both benchmark and synthesis MUST use this same versioned function. An
/// artifact hash cannot be converted into this digest without its content.
pub fn content_fingerprint(
    state: &Entry,
    question: &QuestionSpec,
) -> Result<String, ContentFingerprintError> {
    let expected = match &question.body {
        QuestionBody::Noul { .. } => QuestionKind::Noul,
        QuestionBody::Choice { .. } => QuestionKind::Choice,
        QuestionBody::Score { .. } => QuestionKind::Score,
    };
    if question.kind != expected {
        return Err(ContentFingerprintError::KindMismatch);
    }
    let mut out = Vec::new();
    text(&mut out, CONTENT_FINGERPRINT_VERSION);
    entry(&mut out, state)?;
    optional(&mut out, question.instructions.as_ref())?;
    match &question.body {
        QuestionBody::Noul { criteria } => {
            out.push(0);
            match criteria {
                None => out.push(0),
                Some(c) => {
                    out.push(1);
                    optional(&mut out, c.on_false.as_ref())?;
                    optional(&mut out, c.on_true.as_ref())?;
                }
            }
        }
        QuestionBody::Choice { options } => {
            out.push(1);
            count(&mut out, options.len());
            for option in options {
                text(&mut out, &option.name);
                optional(&mut out, option.rubric.as_ref())?;
            }
        }
        QuestionBody::Score { levels } => {
            out.push(2);
            count(&mut out, levels.len());
            for level in levels {
                optional(&mut out, level.as_ref())?;
            }
        }
    }
    Ok(blake3::hash(&out).to_hex().to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use hyprstream_decision::ChoiceOption;
    fn question() -> QuestionSpec {
        QuestionSpec {
            id: "source_id".into(),
            kind: QuestionKind::Choice,
            instructions: Some(Entry::Str("κ?\0=".into())),
            body: QuestionBody::Choice {
                options: vec![
                    ChoiceOption {
                        name: "a=b".into(),
                        rubric: Some(Entry::Str("c".into())),
                    },
                    ChoiceOption {
                        name: "x".into(),
                        rubric: None,
                    },
                ],
            },
        }
    }
    #[test]
    fn identity_omits_id_preserves_typed_content_order_and_presence() {
        let q = question();
        let state = Entry::Str("true".into());
        let original = content_fingerprint(&state, &q).unwrap();
        let mut changed = q.clone();
        changed.id = "another_source".into();
        assert_eq!(original, content_fingerprint(&state, &changed).unwrap());
        assert_ne!(
            original,
            content_fingerprint(&Entry::Bool(true), &q).unwrap()
        );
        let mut changed = q.clone();
        if let QuestionBody::Choice { options } = &mut changed.body {
            options.reverse();
        }
        assert_ne!(original, content_fingerprint(&state, &changed).unwrap());
        let mut changed = q.clone();
        if let QuestionBody::Choice { options } = &mut changed.body {
            options[0].name = "a".into();
            options[0].rubric = Some(Entry::Str("b=c".into()));
        }
        assert_ne!(original, content_fingerprint(&state, &changed).unwrap());
        let mut changed = q.clone();
        changed.instructions = None;
        let absent = content_fingerprint(&state, &changed).unwrap();
        changed.instructions = Some(Entry::Null);
        assert_ne!(absent, content_fingerprint(&state, &changed).unwrap());
        let x = Entry::Map(vec![
            ("b".into(), Entry::Null),
            ("a".into(), Entry::Seq(vec![Entry::Number(-0.0)])),
        ]);
        let y = Entry::Map(vec![
            ("a".into(), Entry::Seq(vec![Entry::Number(-0.0)])),
            ("b".into(), Entry::Null),
        ]);
        assert_ne!(
            content_fingerprint(&x, &q).unwrap(),
            content_fingerprint(&y, &q).unwrap()
        );
    }
    #[test]
    fn invalid_constructed_content_fails_closed() {
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                content_fingerprint(&Entry::Seq(vec![Entry::Number(v)]), &question()),
                Err(ContentFingerprintError::NonfiniteNumber)
            );
        }
        let mut q = question();
        q.kind = QuestionKind::Noul;
        assert_eq!(
            content_fingerprint(&Entry::Null, &q),
            Err(ContentFingerprintError::KindMismatch)
        );
    }
}
