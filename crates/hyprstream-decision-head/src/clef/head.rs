use super::{ClefError, EncodedClefRecord};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use tch::{Kind, Tensor};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClefHeadConfig {
    pub hidden_size: i64,
    pub width: i64,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: i64,
    pub feedforward: i64,
}
/// Eval-only faithful joint head. Tensor names follow upstream state_dict exactly.
pub struct JointSchemaHead {
    config: ClefHeadConfig,
    weights: HashMap<String, Tensor>,
}
impl JointSchemaHead {
    pub fn expected_shapes(c: &ClefHeadConfig) -> Result<BTreeMap<String, Vec<i64>>, ClefError> {
        if c.hidden_size <= 0
            || c.width <= 0
            || c.heads <= 0
            || c.feedforward <= 0
            || c.width % c.heads != 0
        {
            return Err(ClefError::Head(
                "invalid dimensions/head divisibility".into(),
            ));
        }
        let mut s = BTreeMap::new();
        fn norm(s: &mut BTreeMap<String, Vec<i64>>, p: &str, n: i64) {
            s.insert(format!("{p}.weight"), vec![n]);
            s.insert(format!("{p}.bias"), vec![n]);
        }
        fn linear(s: &mut BTreeMap<String, Vec<i64>>, p: &str, o: i64, i: i64, b: bool) {
            s.insert(format!("{p}.weight"), vec![o, i]);
            if b {
                s.insert(format!("{p}.bias"), vec![o]);
            }
        }
        fn attn(s: &mut BTreeMap<String, Vec<i64>>, p: &str, w: i64) {
            s.insert(format!("{p}.in_proj_weight"), vec![3 * w, w]);
            s.insert(format!("{p}.in_proj_bias"), vec![3 * w]);
            linear(s, &format!("{p}.out_proj"), w, w, true);
        }
        norm(&mut s, "hidden_norm", c.hidden_size);
        for p in [
            "memory_projection",
            "question_projection",
            "option_question_projection",
            "global_projection",
            "option_context_projection",
            "option_lexical_projection",
        ] {
            linear(&mut s, p, c.width, c.hidden_size, false);
        }
        s.insert("type_embedding.weight".into(), vec![3, c.width]);
        for i in 0..c.routing_layers {
            let p = format!("evidence_layers.{i}");
            for n in ["query_norm", "memory_norm", "feedforward_norm"] {
                norm(&mut s, &format!("{p}.{n}"), c.width);
            }
            attn(&mut s, &format!("{p}.attention"), c.width);
            linear(
                &mut s,
                &format!("{p}.feedforward.0"),
                c.feedforward,
                c.width,
                true,
            );
            linear(
                &mut s,
                &format!("{p}.feedforward.3"),
                c.width,
                c.feedforward,
                true,
            );
        }
        norm(&mut s, "option_summary_norm", c.width);
        for i in 0..c.layers {
            let p = format!("layers.{i}");
            for n in ["norm1", "norm2", "norm3"] {
                norm(&mut s, &format!("{p}.{n}"), c.width);
            }
            for a in ["self_attn", "multihead_attn"] {
                attn(&mut s, &format!("{p}.{a}"), c.width);
            }
            linear(
                &mut s,
                &format!("{p}.linear1"),
                c.feedforward,
                c.width,
                true,
            );
            linear(
                &mut s,
                &format!("{p}.linear2"),
                c.width,
                c.feedforward,
                true,
            );
        }
        norm(&mut s, "field_norm", c.width);
        norm(&mut s, "option_norm", c.width);
        linear(&mut s, "residual_scorer.0", c.width, c.width * 4, true);
        linear(&mut s, "residual_scorer.3", 1, c.width, true);
        for p in ["prior_logit_scale", "joint_logit_scale", "residual_gate"] {
            s.insert(p.into(), vec![]);
        }
        Ok(s)
    }
    pub fn from_named_tensors(
        config: ClefHeadConfig,
        weights: HashMap<String, Tensor>,
    ) -> Result<Self, ClefError> {
        Self::from_named_tensors_checked(config, weights, &mut |_| Ok(()))
    }

    /// Cancellation-aware constructor with the SAME validation contract as
    /// [`Self::from_named_tensors`]. `check` runs BEFORE each expected
    /// tensor's admission — including its finite-value scan — with the
    /// tensor's index in the strict expected order; an `Err` aborts
    /// validation with that error. Checks are per-tensor only, never
    /// per-element: one in-flight finite-value scan (a single libtorch
    /// reduction over one tensor) completes before the next check, so a
    /// cancelled caller bounds its remaining validation work to ONE scan
    /// plus the already-in-flight staging work.
    pub fn from_named_tensors_checked<F>(
        config: ClefHeadConfig,
        weights: HashMap<String, Tensor>,
        check: &mut F,
    ) -> Result<Self, ClefError>
    where
        F: FnMut(usize) -> Result<(), ClefError>,
    {
        let expected = Self::expected_shapes(&config)?;
        if expected.len() != weights.len() {
            return Err(ClefError::Head(format!(
                "expected {} tensors, got {}",
                expected.len(),
                weights.len()
            )));
        }
        let mut profile = None;
        for (index, (name, shape)) in expected.into_iter().enumerate() {
            check(index)?;
            let t = weights
                .get(&name)
                .ok_or_else(|| ClefError::Head(format!("missing {name}")))?;
            if t.size() != shape {
                return Err(ClefError::Head(format!(
                    "{name}: expected {shape:?}, got {:?}",
                    t.size()
                )));
            }
            if !matches!(
                t.kind(),
                Kind::Float | Kind::Double | Kind::Half | Kind::BFloat16
            ) {
                return Err(ClefError::Head(format!(
                    "{name}: unsupported dtype {:?}",
                    t.kind()
                )));
            }
            let current = (t.device(), t.kind());
            if let Some(p) = profile {
                if p != current {
                    return Err(ClefError::Head("head device/dtype mismatch".into()));
                }
            } else {
                profile = Some(current);
            }
            if t.isfinite().all().int64_value(&[]) != 1 {
                return Err(ClefError::Head(format!("{name}: nonfinite values")));
            }
        }
        Ok(Self { config, weights })
    }
    fn w(&self, p: &str) -> &Tensor {
        &self.weights[p]
    }
    fn norm(&self, x: &Tensor, p: &str) -> Tensor {
        x.layer_norm(
            [x.size()[x.dim() - 1]],
            Some(self.w(&format!("{p}.weight"))),
            Some(self.w(&format!("{p}.bias"))),
            1e-5,
            true,
        )
    }
    fn linear(&self, x: &Tensor, p: &str) -> Tensor {
        let bias = self.weights.get(&format!("{p}.bias"));
        x.linear(self.w(&format!("{p}.weight")), bias)
    }
    // Equivalent to MultiheadAttention(batch_first=True, need_weights=False).
    // Inputs are unbatched [tokens,width]; there is no causal mask in the head.
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, p: &str) -> Tensor {
        let w = self.config.width;
        let h = self.config.heads;
        let d = w / h;
        let weight = self.w(&format!("{p}.in_proj_weight"));
        let bias = self.w(&format!("{p}.in_proj_bias"));
        let project = |x: &Tensor, i: i64| {
            x.linear(&weight.narrow(0, i * w, w), Some(bias.narrow(0, i * w, w)))
                .view([-1, h, d])
                .transpose(0, 1)
        };
        let q = project(q, 0);
        let k = project(k, 1);
        let v = project(v, 2);
        // Upstream MultiheadAttention(need_weights=False) dispatches SDPA.
        // Use that same primitive so BF16 softmax accumulation is not rounded
        // through a separately materialized low-precision probability tensor.
        let routed = Tensor::scaled_dot_product_attention(
            &q.unsqueeze(0),
            &k.unsqueeze(0),
            &v.unsqueeze(0),
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )
        .squeeze_dim(0)
        .transpose(0, 1)
        .contiguous()
        .view([-1, w]);
        self.linear(&routed, &format!("{p}.out_proj"))
    }
    fn unit(x: &Tensor, eps: f64) -> Tensor {
        x / x.norm_scalaropt_dim(2.0, [-1i64], true).clamp_min(eps)
    }
    fn mean(x: &Tensor, span: (usize, usize)) -> Tensor {
        x.narrow(0, span.0 as i64, (span.1 - span.0) as i64)
            .mean_dim(&[0i64][..], false, x.kind())
    }
    /// All final backbone token states, [sequence_length,hidden_size]. The gather
    /// callback reads original LM-output embedding rows without projecting vocabulary.
    pub fn forward<F>(
        &self,
        hidden: &Tensor,
        record: &EncodedClefRecord,
        mut gather_rows: F,
    ) -> Result<Vec<Tensor>, ClefError>
    where
        F: FnMut(&[i64]) -> Result<Tensor, ClefError>,
    {
        if hidden.size() != [record.input_ids.len() as i64, self.config.hidden_size]
            || record.questions.is_empty()
        {
            return Err(ClefError::Input("hidden shape/questions mismatch".into()));
        }
        let profile = self.w("hidden_norm.weight");
        if hidden.kind() != profile.kind() || hidden.device() != profile.device() {
            return Err(ClefError::Input(
                "hidden/head dtype or device mismatch".into(),
            ));
        }
        for q in &record.questions {
            for (a, b) in std::iter::once(&q.question_span).chain(q.option_spans.iter()) {
                if a >= b || *b > record.input_ids.len() {
                    return Err(ClefError::Input("empty/out-of-range span".into()));
                }
            }
            if q.option_spans.is_empty() || q.option_spans.len() != q.option_ids.len() {
                return Err(ClefError::Input("invalid option spans/ids".into()));
            }
        }
        let hidden = self.norm(hidden, "hidden_norm");
        let memory = self.linear(&hidden, "memory_projection");
        let global = hidden.get(-1);
        let questions = Tensor::stack(
            &record
                .questions
                .iter()
                .map(|q| Self::mean(&hidden, q.question_span))
                .collect::<Vec<_>>(),
            0,
        );
        let mut lexical = Vec::new();
        let mut queries = Vec::new();
        for (i, q) in record.questions.iter().enumerate() {
            let contexts = Tensor::stack(
                &q.option_spans
                    .iter()
                    .map(|&s| Self::mean(&hidden, s))
                    .collect::<Vec<_>>(),
                0,
            );
            let mut lex = Vec::new();
            for &(a, b) in &q.option_spans {
                let rows = gather_rows(&record.input_ids[a..b])?;
                if rows.size() != [(b - a) as i64, self.config.hidden_size]
                    || rows.kind() != hidden.kind()
                    || rows.device() != hidden.device()
                {
                    return Err(ClefError::Input(
                        "lexical row shape/device/dtype mismatch".into(),
                    ));
                }
                lex.push(rows.mean_dim(&[0i64][..], false, rows.kind()));
            }
            let lex = Tensor::stack(&lex, 0);
            queries.push(
                self.linear(&contexts, "option_context_projection")
                    + self.linear(&lex, "option_lexical_projection")
                    + self
                        .linear(&questions.get(i as i64), "option_question_projection")
                        .unsqueeze(0),
            );
            lexical.push(lex);
        }
        let mut options = Tensor::cat(&queries, 0);
        for i in 0..self.config.routing_layers {
            let p = format!("evidence_layers.{i}");
            let q = self.norm(&options, &format!("{p}.query_norm"));
            let m = self.norm(&memory, &format!("{p}.memory_norm"));
            options = &options + self.attention(&q, &m, &m, &format!("{p}.attention"));
            let ff = self.norm(&options, &format!("{p}.feedforward_norm"));
            options = &options
                + self.linear(
                    &self.linear(&ff, &format!("{p}.feedforward.0")).gelu("none"),
                    &format!("{p}.feedforward.3"),
                );
        }
        let base = self.linear(&questions, "question_projection");
        let mut split = Vec::new();
        let mut summaries = Vec::new();
        let mut offset = 0;
        for (i, q) in record.questions.iter().enumerate() {
            let option = options.narrow(0, offset, q.option_spans.len() as i64);
            offset += q.option_spans.len() as i64;
            let weights = (option.matmul(&base.get(i as i64)) / (self.config.width as f64).sqrt())
                .softmax(0, option.kind());
            summaries.push((weights.unsqueeze(-1) * &option).sum_dim_intlist(
                &[0i64][..],
                false,
                option.kind(),
            ));
            split.push(option);
        }
        let type_ids = Tensor::from_slice(
            &record
                .questions
                .iter()
                .map(|q| q.kind.type_id())
                .collect::<Vec<_>>(),
        )
        .to_device(hidden.device());
        let mut fields = base
            + self.norm(&Tensor::stack(&summaries, 0), "option_summary_norm")
            + self.linear(&global, "global_projection").unsqueeze(0)
            + self.w("type_embedding.weight").index_select(0, &type_ids);
        for i in 0..self.config.layers {
            let p = format!("layers.{i}");
            let n = self.norm(&fields, &format!("{p}.norm1"));
            fields = &fields + self.attention(&n, &n, &n, &format!("{p}.self_attn"));
            let n = self.norm(&fields, &format!("{p}.norm2"));
            fields = &fields + self.attention(&n, &memory, &memory, &format!("{p}.multihead_attn"));
            let n = self.norm(&fields, &format!("{p}.norm3"));
            fields = &fields
                + self.linear(
                    &self.linear(&n, &format!("{p}.linear1")).gelu("none"),
                    &format!("{p}.linear2"),
                );
        }
        fields = self.norm(&fields, "field_norm");
        let mut logits = Vec::new();
        for (i, (lex, routed)) in lexical.iter().zip(&split).enumerate() {
            let anchor = Self::unit(&(questions.get(i as i64) + &global), 1e-12);
            let prior = Self::unit(lex, 1e-12).matmul(&anchor)
                * self.w("prior_logit_scale").clamp_max(100f64.ln()).exp();
            let option = self.norm(routed, "option_norm");
            let field = fields.get(i as i64).unsqueeze(0).expand_as(&option);
            let cosine = Tensor::cosine_similarity(&field, &option, -1, 1e-8);
            let features = Tensor::cat(
                &[
                    &field,
                    &option,
                    &(&field * &option),
                    &(&field - &option).abs(),
                ],
                -1,
            );
            let residual = self
                .linear(
                    &self.linear(&features, "residual_scorer.0").gelu("none"),
                    "residual_scorer.3",
                )
                .squeeze_dim(-1);
            let joint =
                cosine * self.w("joint_logit_scale").clamp_max(100f64.ln()).exp() + residual;
            let logit = prior + self.w("residual_gate").sigmoid() * joint;
            if logit.isfinite().all().int64_value(&[]) != 1 {
                return Err(ClefError::Head("nonfinite joint logits".into()));
            }
            logits.push(logit);
        }
        Ok(logits)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::clef::{ClefQuestionType, EncodedClefQuestion};
    use tch::Device;
    fn config() -> ClefHeadConfig {
        ClefHeadConfig {
            hidden_size: 2,
            width: 2,
            routing_layers: 1,
            layers: 1,
            heads: 1,
            feedforward: 3,
        }
    }
    fn weights(c: &ClefHeadConfig) -> HashMap<String, Tensor> {
        JointSchemaHead::expected_shapes(c)
            .expect("shapes")
            .into_iter()
            .map(|(name, shape)| {
                let value = if name.ends_with("norm.weight")
                    || name.contains(".norm1.weight")
                    || name.contains(".norm2.weight")
                    || name.contains(".norm3.weight")
                {
                    Tensor::ones(&shape, (Kind::Double, Device::Cpu))
                } else {
                    Tensor::zeros(&shape, (Kind::Double, Device::Cpu))
                };
                (name, value)
            })
            .collect()
    }
    #[test]
    fn clef_head_full_path_lexical_reference() {
        // Source-derived fixture: all learned projections zero; residual zero;
        // scales exp(0)=1; lexical cosine is ±1. Routers and decoder still run.
        let c = config();
        let head = JointSchemaHead::from_named_tensors(c.clone(), weights(&c)).expect("head");
        let record = EncodedClefRecord {
            state_tokens_before_truncation: 0,
            state_tokens_admitted: 0,
            state_truncated: false,
            record_id: "r".into(),
            input_ids: vec![0, 1, 2, 0],
            questions: vec![EncodedClefQuestion {
                question_id: "q".into(),
                kind: ClefQuestionType::Noul,
                question_span: (0, 1),
                option_spans: vec![(1, 2), (2, 3)],
                option_ids: vec!["true".into(), "false".into()],
            }],
        };
        let hidden = Tensor::from_slice(&[1f64, -1., -1., 1., 1., -1., 1., -1.]).view([4, 2]);
        let logits = head
            .forward(&hidden, &record, |ids| {
                Ok(Tensor::stack(
                    &ids.iter()
                        .map(|&id| {
                            if id == 1 {
                                Tensor::from_slice(&[1f64, -1.])
                            } else {
                                Tensor::from_slice(&[-1f64, 1.])
                            }
                        })
                        .collect::<Vec<_>>(),
                    0,
                ))
            })
            .expect("forward");
        assert!((logits[0].double_value(&[0]) - 1.).abs() < 1e-12);
        assert!((logits[0].double_value(&[1]) + 1.).abs() < 1e-12);
        let p = logits[0].softmax(0, Kind::Double);
        assert!((p.double_value(&[0]) - 0.8807970779778823).abs() < 1e-12);
    }
    #[test]
    fn clef_attention_matches_two_key_equation() {
        let c = config();
        let mut ws = weights(&c);
        let identity = Tensor::eye(2, (Kind::Double, Device::Cpu));
        ws.insert(
            "evidence_layers.0.attention.in_proj_weight".into(),
            Tensor::cat(&[&identity, &identity, &identity], 0),
        );
        ws.insert(
            "evidence_layers.0.attention.out_proj.weight".into(),
            identity,
        );
        let head = JointSchemaHead::from_named_tensors(c, ws).expect("head");
        let q = Tensor::from_slice(&[1f64, 0.]).view([1, 2]);
        let memory = Tensor::from_slice(&[1f64, 0., 0., 1.]).view([2, 2]);
        let result = head.attention(&q, &memory, &memory, "evidence_layers.0.attention");
        let expected = (1f64 / 2f64.sqrt()).exp();
        let expected = expected / (expected + 1.);
        assert!((result.double_value(&[0, 0]) - expected).abs() < 1e-12);
        assert!((result.double_value(&[0, 1]) - (1. - expected)).abs() < 1e-12);
    }
    #[test]
    fn clef_head_uses_other_question_evidence() {
        let c = ClefHeadConfig {
            hidden_size: 4,
            width: 4,
            routing_layers: 1,
            layers: 1,
            heads: 2,
            feedforward: 7,
        };
        let ws = JointSchemaHead::expected_shapes(&c)
            .expect("shapes")
            .into_iter()
            .enumerate()
            .map(|(i, (name, shape))| {
                let count = shape.iter().product::<i64>();
                let value = if name.ends_with("norm.weight")
                    || name.contains(".norm1.weight")
                    || name.contains(".norm2.weight")
                    || name.contains(".norm3.weight")
                {
                    Tensor::ones(&shape, (Kind::Double, Device::Cpu))
                } else {
                    ((Tensor::arange(count, (Kind::Double, Device::Cpu)) + i as f64).sin() * 0.2)
                        .view(shape.as_slice())
                };
                (name, value)
            })
            .collect();
        let head = JointSchemaHead::from_named_tensors(c, ws).expect("head");
        let record = EncodedClefRecord {
            state_tokens_before_truncation: 0,
            state_tokens_admitted: 0,
            state_truncated: false,
            record_id: "joint".into(),
            input_ids: (0..8).collect(),
            questions: vec![
                EncodedClefQuestion {
                    question_id: "a".into(),
                    kind: ClefQuestionType::Choice,
                    question_span: (0, 1),
                    option_spans: vec![(1, 2), (2, 3)],
                    option_ids: vec!["a".into(), "b".into()],
                },
                EncodedClefQuestion {
                    question_id: "b".into(),
                    kind: ClefQuestionType::Score,
                    question_span: (3, 4),
                    option_spans: vec![(4, 5), (5, 6)],
                    option_ids: vec!["0".into(), "1".into()],
                },
            ],
        };
        let hidden = Tensor::arange(32, (Kind::Double, Device::Cpu))
            .sin()
            .view([8, 4]);
        let gather = |ids: &[i64]| {
            Ok((Tensor::from_slice(ids).to_kind(Kind::Double).unsqueeze(-1)
                + Tensor::arange(4, (Kind::Double, Device::Cpu)))
            .cos())
        };
        let before = head.forward(&hidden, &record, gather).expect("before");
        let changed = hidden.copy();
        let _ = changed.get(3).fill_(0.);
        let _ = changed.get(3).get(0).fill_(4.);
        let after = head.forward(&changed, &record, gather).expect("after");
        assert!(
            (&before[0] - &after[0]).abs().max().double_value(&[]) > 1e-8,
            "joint head must use other-field evidence"
        );
    }
    #[test]
    #[ignore = "requires CLEF_HEAD_DIRECTORY pointing to pinned clef_head artifact"]
    fn clef_pinned_head_load_and_forward() {
        tch::set_num_threads(2);
        let path =
            std::path::PathBuf::from(std::env::var("CLEF_HEAD_DIRECTORY").expect("head directory"));
        let config: ClefHeadConfig = serde_json::from_slice(
            &std::fs::read(path.join("joint_head_config.json")).expect("config file"),
        )
        .expect("config");
        let hidden_size = config.hidden_size;
        let tensors = Tensor::read_safetensors(path.join("joint_head.safetensors"))
            .expect("safetensors")
            .into_iter()
            .map(|(k, v)| (k, v.to_kind(Kind::Float)))
            .collect();
        let head =
            JointSchemaHead::from_named_tensors(config, tensors).expect("strict pinned head");
        let record = EncodedClefRecord {
            state_tokens_before_truncation: 0,
            state_tokens_admitted: 0,
            state_truncated: false,
            record_id: "head-smoke".into(),
            input_ids: vec![1, 2, 3, 4],
            questions: vec![EncodedClefQuestion {
                question_id: "q".into(),
                kind: ClefQuestionType::Noul,
                question_span: (0, 1),
                option_spans: vec![(1, 2), (2, 3)],
                option_ids: vec!["true".into(), "false".into()],
            }],
        };
        let hidden = (Tensor::arange(4 * hidden_size, (Kind::Float, Device::Cpu)) * 0.001)
            .sin()
            .view([4, hidden_size]);
        let logits = tch::no_grad(|| {
            head.forward(&hidden, &record, |ids| {
                Ok((Tensor::from_slice(ids).to_kind(Kind::Float).unsqueeze(-1)
                    + Tensor::arange(hidden_size, (Kind::Float, Device::Cpu)) * 0.01)
                    .cos())
            })
        })
        .expect("native head");
        assert_eq!(logits[0].size(), [2]);
        assert_eq!(logits[0].isfinite().all().int64_value(&[]), 1);
    }
    #[test]
    fn clef_head_rejects_missing_and_nonfinite() {
        let c = config();
        let mut ws = weights(&c);
        ws.remove("residual_gate");
        assert!(JointSchemaHead::from_named_tensors(c.clone(), ws).is_err());
        let mut ws = weights(&c);
        ws.insert("residual_gate".into(), Tensor::from(f64::NAN));
        assert!(JointSchemaHead::from_named_tensors(c, ws).is_err());
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
#[path = "oracle_tests.rs"]
mod oracle_tests;
