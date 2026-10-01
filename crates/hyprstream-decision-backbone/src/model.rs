//! [`DecisionModel`]: backbone + shared scorer in one `VarStore`, one forward pass.
//!
//! This is the P1.1 → P1.2 integration point: the scorer is created under its own
//! `nn::Path` scope (the review-driven fix — `SharedScorer::new` takes a path, not
//! `vs.root()`, so the projection never collides with backbone parameter names),
//! and the whole model checkpoints as one file.

use hyprstream_decision_head::{QuestionAnchors, QuestionScore, SharedScorer};
use tch::nn;
use tch::Tensor;

use crate::backbone::Backbone;
use crate::Error;

/// The converted decision model: a [`Backbone`] producing hidden states plus the
/// P1.1 shared scorer reading anchor positions. Generic over the backbone so the
/// converted Qwen3.5 hybrid and the ModernBERT baseline arm share the harness —
/// P1.4's gate (b) swaps `B` and nothing else.
pub struct DecisionModel<B: Backbone> {
    backbone: B,
    scorer: SharedScorer,
}

impl<B: Backbone> DecisionModel<B> {
    /// Attach the scorer under `scorer_path` (e.g. `vs.root().sub("scorer")`) to a
    /// backbone already constructed under the same `VarStore`.
    pub fn new(backbone: B, scorer_path: &nn::Path) -> Self {
        let scorer = SharedScorer::new(scorer_path, backbone.hidden_dim());
        Self { backbone, scorer }
    }

    /// The backbone half (for backbone-specific inspection/ablations).
    pub fn backbone(&self) -> &B {
        &self.backbone
    }

    /// The scorer half (P1.4 needs it for the distillation loss,
    /// [`hyprstream_decision_head::soft_target_kl`]).
    pub fn scorer(&self) -> &SharedScorer {
        &self.scorer
    }

    /// Score every question of every sequence in the batch: token ids `[batch, seq]`
    /// plus one anchor-map list per batch row (from
    /// [`hyprstream_decision_head::DecisionEncoder::encode`]) → typed per-question
    /// distributions, one forward pass of the backbone.
    pub fn forward(
        &self,
        token_ids: &Tensor,
        batch_anchors: &[Vec<QuestionAnchors>],
    ) -> Result<Vec<Vec<QuestionScore>>, Error> {
        let hidden = self.backbone.forward(token_ids)?;
        Ok(self.scorer.forward(&hidden, batch_anchors)?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_decision::QuestionKind;
    use tch::{Device, Kind};

    use crate::modernbert::{ModernBertConfig, ModernBertEncoder};
    use crate::qwen35::{Qwen35Config, Qwen35Encoder};
    use crate::MaskPolicy;

    fn anchors(id: &str, kind: QuestionKind, positions: &[u32]) -> QuestionAnchors {
        QuestionAnchors {
            question_id: id.to_owned(),
            kind,
            anchor_positions: positions.to_vec(),
        }
    }

    #[test]
    fn qwen_arm_end_to_end_distributions() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = Qwen35Config::test_tiny();
        let encoder =
            Qwen35Encoder::new(&vs.root().sub("model"), &cfg, MaskPolicy::Bidirectional).unwrap();
        let model = DecisionModel::new(encoder, &vs.root().sub("scorer"));
        let ids = Tensor::randint(255, [1, 24], (Kind::Int64, Device::Cpu));
        let batch = vec![vec![
            anchors("c", QuestionKind::Choice, &[0, 1, 2]),
            anchors("n", QuestionKind::Noul, &[5]),
            anchors("s", QuestionKind::Score, &[10, 11, 12, 13]),
        ]];
        let out = model.forward(&ids, &batch).unwrap();
        let questions = &out[0];
        assert_eq!(questions.len(), 3);
        for q in questions {
            let probs = Vec::<f64>::try_from(q.probabilities().to_kind(Kind::Double)).unwrap();
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "distribution sums to 1: {sum}");
        }
        assert_eq!(questions[0].probabilities().size(), [3]);
        assert_eq!(questions[1].probabilities().size(), [2]);
        assert!(questions[2].expected_value().is_some());
    }

    #[test]
    fn modernbert_arm_end_to_end_distributions() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = ModernBertConfig::test_tiny();
        let encoder = ModernBertEncoder::new(&vs.root().sub("model"), &cfg).unwrap();
        let model = DecisionModel::new(encoder, &vs.root().sub("scorer"));
        let ids = Tensor::randint(255, [1, 16], (Kind::Int64, Device::Cpu));
        let batch = vec![vec![anchors("c", QuestionKind::Choice, &[0, 1, 2, 3])]];
        let out = model.forward(&ids, &batch).unwrap();
        assert_eq!(out[0][0].probabilities().size(), [4]);
    }

    #[test]
    fn scorer_params_do_not_collide_with_backbone_params() {
        // The nn::Path-scoping fix: backbone and scorer share one VarStore; names
        // must be disjoint and the checkpoint must round-trip both.
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = Qwen35Config::test_tiny();
        let encoder =
            Qwen35Encoder::new(&vs.root().sub("model"), &cfg, MaskPolicy::Bidirectional).unwrap();
        let model = DecisionModel::new(encoder, &vs.root().sub("scorer"));
        let names: Vec<String> = vs.variables().into_keys().collect();
        let scorer_vars: Vec<&String> = names.iter().filter(|n| n.starts_with("scorer")).collect();
        assert_eq!(
            scorer_vars.len(),
            2,
            "scorer owns weight + bias: {scorer_vars:?}"
        );
        assert!(
            names
                .iter()
                .all(|n| n.starts_with("scorer") || n.starts_with("model")),
            "backbone and scorer namespaces are disjoint: {names:?}"
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("decision_model.ot");
        let ids = Tensor::randint(255, [1, 12], (Kind::Int64, Device::Cpu));
        let batch = vec![vec![anchors("c", QuestionKind::Choice, &[0, 1])]];
        let before = model.forward(&ids, &batch).unwrap();
        let before = Vec::<f64>::try_from(before[0][0].logits.to_kind(Kind::Double)).unwrap();
        vs.save(&path).unwrap();

        let mut vs2 = nn::VarStore::new(Device::Cpu);
        let encoder2 =
            Qwen35Encoder::new(&vs2.root().sub("model"), &cfg, MaskPolicy::Bidirectional).unwrap();
        let model2 = DecisionModel::new(encoder2, &vs2.root().sub("scorer"));
        vs2.load(&path).unwrap();
        let after = model2.forward(&ids, &batch).unwrap();
        let after = Vec::<f64>::try_from(after[0][0].logits.to_kind(Kind::Double)).unwrap();
        assert_eq!(
            before, after,
            "joint checkpoint round-trips backbone + scorer"
        );
    }

    #[test]
    fn distillation_loss_trains_backbone_and_scorer_together() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = Qwen35Config::test_tiny();
        let encoder =
            Qwen35Encoder::new(&vs.root().sub("model"), &cfg, MaskPolicy::Bidirectional).unwrap();
        let model = DecisionModel::new(encoder, &vs.root().sub("scorer"));
        let ids = Tensor::randint(255, [1, 12], (Kind::Int64, Device::Cpu));
        let batch = vec![vec![anchors("c", QuestionKind::Choice, &[0, 1])]];
        let out = model.forward(&ids, &batch).unwrap();
        let target = Tensor::from_slice(&[0.3, 0.7]);
        let loss = hyprstream_decision_head::soft_target_kl(&out[0][0], &target);
        loss.backward();
        let backbone_grad = vs
            .variables()
            .iter()
            .filter(|(n, _)| n.starts_with("model"))
            .any(|(_, t)| t.grad().isfinite().any().int64_value(&[]) == 1);
        let scorer_grad = vs
            .variables()
            .iter()
            .filter(|(n, _)| n.starts_with("scorer"))
            .all(|(_, t)| t.grad().isfinite().any().int64_value(&[]) == 1);
        assert!(
            backbone_grad,
            "KL flows into the converted backbone (P1.4 path)"
        );
        assert!(scorer_grad, "KL flows into the shared scorer");
    }
}
