//! The shared option-scorer: ONE linear projection over anchor-token hidden states,
//! with per-primitive readouts. This is the generalist head (P1.1) — no per-question
//! parameters, no cross-question conditioning: each question is scored exclusively
//! from its own anchor positions, so runtime-declared questions work zero-shot and the
//! 255-option cardinality ceiling costs nothing in parameters.

use hyprstream_decision::QuestionKind;
use tch::nn;
use tch::{Kind, Tensor};

use crate::encoder::QuestionAnchors;
use crate::Error;

/// The scored output for one question: the raw per-anchor logits plus the primitive's
/// distribution readout, derived on demand.
#[derive(Debug)]
pub struct QuestionScore {
    /// Caller-facing question id (carried through from the anchor map).
    pub question_id: String,
    /// The primitive type.
    pub kind: QuestionKind,
    /// Raw scorer logits: 1 for noul (the sigmoid logit), one per option for choice,
    /// one per level for score.
    pub logits: Tensor,
}

impl QuestionScore {
    /// The distribution over the question's labels, in canonical order:
    /// sigmoid → `[P(false), P(true)]` for noul; softmax for choice; binned softmax
    /// over levels for score.
    pub fn probabilities(&self) -> Tensor {
        match self.kind {
            QuestionKind::Noul => {
                let p_true = self.logits.view(-1).sigmoid();
                let p_false = p_true.neg() + 1.0;
                Tensor::cat(&[p_false, p_true], 0)
            }
            QuestionKind::Choice | QuestionKind::Score => {
                self.logits.log_softmax(0, Kind::Float).exp()
            }
            // Reserved v2 kinds are rejected at authoring; the scorer never sees them.
            other => unreachable!("reserved question kind {other} cannot reach the scorer"),
        }
    }

    /// The probability-weighted value `Σᵢ i·pᵢ` over score levels (the expectation
    /// readout pinned by S6b2). `None` for non-score primitives — their trust object is
    /// the distribution itself.
    pub fn expected_value(&self) -> Option<Tensor> {
        if self.kind != QuestionKind::Score {
            return None;
        }
        let levels = self.logits.size()[0];
        let indices = Tensor::arange(levels, (Kind::Float, self.logits.device()));
        Some((self.probabilities() * indices).sum(Kind::Float))
    }
}

/// KL divergence `KL(target ‖ p)` between a target distribution (the corrected
/// teacher-average histogram in P1.4 distillation) and the scorer's readout for one
/// question. Distribution targets are the native input of the binned-softmax
/// parameterization (S6b2); the target must be non-negative and sum to 1 over the
/// question's label arity (2 for noul: `[P(false), P(true)]`).
///
/// Computed in log space (`Σ t·(log t − log p)`) with the target clamped away from
/// exact zero, so zero-probability target bins contribute nothing instead of NaNs.
pub fn soft_target_kl(score: &QuestionScore, target: &Tensor) -> Tensor {
    let log_p = match score.kind {
        QuestionKind::Noul => {
            let logit = score.logits.view(-1);
            let log_p_true = logit.log_sigmoid();
            let log_p_false = logit.neg().log_sigmoid();
            Tensor::cat(&[log_p_false, log_p_true], 0)
        }
        QuestionKind::Choice | QuestionKind::Score => score.logits.log_softmax(0, Kind::Float),
        other => unreachable!("reserved question kind {other} cannot reach the scorer"),
    };
    let log_target = target.clamp_min(1e-12).log();
    (target * (log_target - log_p)).sum(Kind::Float)
}

/// One shared linear scorer over anchor-token hidden states: a single
/// `d_model → 1` projection applied to every anchor position, of every option, of
/// every question. Trained weights live in a [`nn::VarStore`] so checkpoints attach to
/// the converted backbone in P1.2 without a format boundary.
pub struct SharedScorer {
    projection: nn::Linear,
}

impl SharedScorer {
    /// Create the scorer under `path` with the backbone's hidden size. The caller
    /// scopes the path (e.g. `vs.root().sub("scorer")` on the combined model
    /// `VarStore`) so the projection never collides with backbone parameter names
    /// when head and backbone share one checkpoint.
    pub fn new(path: &nn::Path, hidden_dim: i64) -> Self {
        Self {
            projection: nn::linear(path, hidden_dim, 1, Default::default()),
        }
    }

    /// Score every question of every sequence in the batch.
    ///
    /// `hidden` is the backbone's final hidden states, `[batch, tokens, d_model]`;
    /// `batch_anchors` is one anchor map list per batch row (from
    /// [`crate::encoder::DecisionEncoder::encode`]). Each question's logits are the
    /// projection of its own anchor states — structurally identical per option, which
    /// is what makes option-order permutation a data-side concern (S6b1) rather than
    /// an architectural one.
    pub fn forward(
        &self,
        hidden: &Tensor,
        batch_anchors: &[Vec<QuestionAnchors>],
    ) -> Result<Vec<Vec<QuestionScore>>, Error> {
        let size = hidden.size();
        let batch = size[0] as usize;
        if batch != batch_anchors.len() {
            return Err(Error::BatchMismatch {
                batch,
                maps: batch_anchors.len(),
            });
        }
        let device = hidden.device();
        let mut output = Vec::with_capacity(batch);
        for (row, anchors) in batch_anchors.iter().enumerate() {
            let states = hidden.get(row as i64);
            let mut questions = Vec::with_capacity(anchors.len());
            for question in anchors {
                let positions: Vec<i64> = question
                    .anchor_positions
                    .iter()
                    .map(|position| i64::from(*position))
                    .collect();
                let positions = Tensor::from_slice(&positions).to_device(device);
                let gathered = states.index_select(0, &positions);
                let logits = gathered.apply(&self.projection).squeeze_dim(1);
                questions.push(QuestionScore {
                    question_id: question.question_id.clone(),
                    kind: question.kind,
                    logits,
                });
            }
            output.push(questions);
        }
        Ok(output)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::encoder::{DecisionEncoder, QuestionAnchors};
    use hyprstream_decision::parse_yaml;
    use tch::nn::VarStore;
    use tch::Device;

    fn anchors(id: &str, kind: QuestionKind, positions: &[u32]) -> QuestionAnchors {
        QuestionAnchors {
            question_id: id.to_owned(),
            kind,
            anchor_positions: positions.to_vec(),
        }
    }

    fn probs_vec(score: &QuestionScore) -> Vec<f64> {
        Vec::<f64>::try_from(score.probabilities().to_kind(Kind::Double)).unwrap()
    }

    #[test]
    fn choice_and_score_readouts_are_distributions() {
        let vs = VarStore::new(Device::Cpu);
        let scorer = SharedScorer::new(&vs.root(), 8);
        let hidden = Tensor::randn([1, 6, 8], (Kind::Float, Device::Cpu));
        let batch = vec![vec![
            anchors("c", QuestionKind::Choice, &[0, 1, 2]),
            anchors("s", QuestionKind::Score, &[3, 4, 5]),
        ]];
        let output = scorer.forward(&hidden, &batch).unwrap();
        assert_eq!(output.len(), 1);
        let questions = &output[0];
        let choice_probs = probs_vec(&questions[0]);
        assert_eq!(choice_probs.len(), 3);
        let sum: f64 = choice_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "choice softmax sums to 1: {sum}");
        let score_probs = probs_vec(&questions[1]);
        let sum: f64 = score_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "score softmax sums to 1: {sum}");
    }

    #[test]
    fn noul_readout_is_sigmoid_in_label_order() {
        let vs = VarStore::new(Device::Cpu);
        let scorer = SharedScorer::new(&vs.root(), 8);
        let hidden = Tensor::randn([1, 1, 8], (Kind::Float, Device::Cpu));
        let batch = vec![vec![anchors("n", QuestionKind::Noul, &[0])]];
        let output = scorer.forward(&hidden, &batch).unwrap();
        let score = &output[0][0];
        let logit = score.logits.double_value(&[0]);
        let probs = probs_vec(score);
        assert_eq!(probs.len(), 2);
        let expected_p_true = 1.0 / (1.0 + (-logit).exp());
        assert!(
            (probs[1] - expected_p_true).abs() < 1e-5,
            "P(true) is sigmoid(logit): {} vs {expected_p_true}",
            probs[1]
        );
        assert!(
            (probs[0] + probs[1] - 1.0).abs() < 1e-5,
            "[P(false), P(true)] sums to 1"
        );
    }

    #[test]
    fn score_expected_value_is_the_expectation_over_levels() {
        let vs = VarStore::new(Device::Cpu);
        let scorer = SharedScorer::new(&vs.root(), 8);
        let hidden = Tensor::randn([1, 3, 8], (Kind::Float, Device::Cpu));
        let batch = vec![vec![anchors("s", QuestionKind::Score, &[0, 1, 2])]];
        let output = scorer.forward(&hidden, &batch).unwrap();
        let score = &output[0][0];
        let probs = probs_vec(score);
        let expected = score
            .expected_value()
            .expect("score carries an expected value")
            .double_value(&[]);
        let manual: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
        assert!((expected - manual).abs() < 1e-5, "{expected} vs {manual}");
        assert!(
            (0.0..=2.0).contains(&expected),
            "expectation within level range"
        );
        let c = &output[0][0];
        assert_eq!(c.kind, QuestionKind::Score);
    }

    #[test]
    fn scorer_is_shared_and_permutation_equivariant() {
        // The same projection applies to every anchor: permuting a question's option
        // order permutes its logits identically (the architectural half of order
        // robustness; the data half is P1.3's mandatory permutation augmentation).
        let vs = VarStore::new(Device::Cpu);
        let scorer = SharedScorer::new(&vs.root(), 8);
        let hidden = Tensor::randn([1, 4, 8], (Kind::Float, Device::Cpu));
        let forward = |positions: &[u32]| {
            let batch = vec![vec![anchors("c", QuestionKind::Choice, positions)]];
            let output = scorer.forward(&hidden, &batch).unwrap();
            Vec::<f64>::try_from(output[0][0].logits.to_kind(Kind::Double)).unwrap()
        };
        let base = forward(&[0, 1, 2]);
        let permuted = forward(&[2, 0, 1]);
        assert_eq!(base.len(), permuted.len());
        assert!((permuted[0] - base[2]).abs() < 1e-5);
        assert!((permuted[1] - base[0]).abs() < 1e-5);
        assert!((permuted[2] - base[1]).abs() < 1e-5);
    }

    #[test]
    fn distillation_loss_backprops_into_the_shared_projection() {
        let vs = VarStore::new(Device::Cpu);
        let scorer = SharedScorer::new(&vs.root(), 8);
        let hidden = Tensor::randn([1, 2, 8], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let batch = vec![vec![anchors("c", QuestionKind::Choice, &[0, 1])]];
        let output = scorer.forward(&hidden, &batch).unwrap();
        let target = Tensor::from_slice(&[0.25, 0.75]).to_kind(Kind::Float);
        let loss = soft_target_kl(&output[0][0], &target);
        assert!(loss.double_value(&[]) >= 0.0);
        loss.backward();
        let root = vs.root();
        let weight = root.get("weight").expect("projection weight exists");
        let grad = weight.grad();
        let grad_norm = grad.norm().double_value(&[]);
        assert!(grad_norm > 0.0, "KL loss backprops into the scorer");
    }

    #[test]
    fn soft_target_kl_ignores_zero_mass_bins() {
        let vs = VarStore::new(Device::Cpu);
        let scorer = SharedScorer::new(&vs.root(), 8);
        let hidden = Tensor::randn([1, 3, 8], (Kind::Float, Device::Cpu));
        let batch = vec![vec![anchors("s", QuestionKind::Score, &[0, 1, 2])]];
        let output = scorer.forward(&hidden, &batch).unwrap();
        let target = Tensor::from_slice(&[0.0, 1.0, 0.0]).to_kind(Kind::Float);
        let loss = soft_target_kl(&output[0][0], &target).double_value(&[]);
        assert!(
            loss.is_finite(),
            "zero target bins contribute nothing: {loss}"
        );
    }

    #[test]
    fn varstore_checkpoint_roundtrip_preserves_logits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scorer.ot");
        let vs = VarStore::new(Device::Cpu);
        let scorer = SharedScorer::new(&vs.root(), 8);
        let hidden = Tensor::randn([1, 2, 8], (Kind::Float, Device::Cpu));
        let batch = vec![vec![anchors("c", QuestionKind::Choice, &[0, 1])]];
        let before = scorer.forward(&hidden, &batch).unwrap();
        let before = Vec::<f64>::try_from(before[0][0].logits.to_kind(Kind::Double)).unwrap();
        vs.save(&path).unwrap();

        let mut vs2 = VarStore::new(Device::Cpu);
        let scorer2 = SharedScorer::new(&vs2.root(), 8);
        vs2.load(&path).unwrap();
        let after = scorer2.forward(&hidden, &batch).unwrap();
        let after = Vec::<f64>::try_from(after[0][0].logits.to_kind(Kind::Double)).unwrap();
        assert_eq!(before, after, "checkpoint roundtrip preserves the scorer");
    }

    #[test]
    fn batch_mismatch_fails_loudly() {
        let vs = VarStore::new(Device::Cpu);
        let scorer = SharedScorer::new(&vs.root(), 8);
        let hidden = Tensor::randn([2, 2, 8], (Kind::Float, Device::Cpu));
        let batch = vec![vec![anchors("n", QuestionKind::Noul, &[0])]];
        let error = scorer.forward(&hidden, &batch).unwrap_err();
        assert!(matches!(error, Error::BatchMismatch { batch: 2, maps: 1 }));
    }

    #[test]
    fn encode_then_score_end_to_end() {
        // The full contract path: authored spec → canonical serialization → token ids +
        // anchor map → hidden states → typed distributions.
        const TOK_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#;
        let tokenizer = tokenizers::Tokenizer::from_bytes(TOK_JSON.as_bytes()).unwrap();
        let encoder = DecisionEncoder::new(tokenizer).unwrap();
        let set = parse_yaml(
            r#"
state: "The box was crushed."
questions:
  tone:
    type: choice
    criteria:
      positive: "approval"
      negative: "disapproval"
  severe:
    type: score
    criteria: ["trivial", "critical"]
  is_refund:
    type: noul
"#,
        )
        .unwrap();
        let encoded = encoder.encode(&set).unwrap();
        let tokens = encoded.token_count() as i64;
        let vs = VarStore::new(Device::Cpu);
        let scorer = SharedScorer::new(&vs.root(), 8);
        let hidden = Tensor::randn([1, tokens, 8], (Kind::Float, Device::Cpu));
        let output = scorer.forward(&hidden, &[encoded.questions]).unwrap();
        let questions = &output[0];
        assert_eq!(questions.len(), 3);
        assert_eq!(probs_vec(&questions[0]).len(), 2, "choice over 2 options");
        assert_eq!(probs_vec(&questions[1]).len(), 2, "score over 2 levels");
        assert!(questions[1].expected_value().is_some());
        assert_eq!(
            probs_vec(&questions[2]).len(),
            2,
            "noul [P(false), P(true)]"
        );
        assert!(questions[2].expected_value().is_none());
    }
}
