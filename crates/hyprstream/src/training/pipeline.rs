//! Pipeline-split (2b) **training** orchestration for TTT-on-split (#316).
//!
//! This is the training counterpart to the inference `forward_layers` pipeline
//! (#314): it drives a TTT/training forward across a set of per-device *stage*
//! models, building a single autograd graph over the
//! [`crate::runtime::device_pool::LayerDeviceMap`] so that `loss.backward()`
//! traverses the same partition inference uses (M-TRAIN-COUPLING). The only
//! cross-device copies are the stage-boundary `hidden.to_device(next)` transfers
//! inside each stage's `forward_layers_train`, which tch makes
//! autograd-transparent — gradients flow back across device boundaries to each
//! parameter's own device.
//!
//! # Stage contract (mirrors the inference pipeline)
//! Stages are contiguous global layer ranges `[a..b)` derived from the device
//! map. The first stage owns `embed_tokens`, the last stage owns the final norm
//! and LM head:
//! - **stage 0** : `embed_tokens` → `forward_layers_train(0..b)`
//! - **middle**  : `forward_layers_train(a..b)`
//! - **last**    : `forward_layers_train(a..N)` → `apply_final_norm` → `lm_head`
//!
//! Each stage's training runner uses **no KV cache, full causal attention,
//! `start_pos = 0`, and fresh call-local recurrent (SSM) state** — see
//! [`crate::runtime::architectures::ModelOperations::forward_layers_train`].
//!
//! # Send / !Send
//! Like the inference pipeline, the stage models hold `!Send` tch tensors and
//! must all live on the **one** engine-owning thread; this module never crosses a
//! thread boundary. Multi-device-on-one-thread is fine (libtorch `DeviceGuard`).
//! The single-stage case (`stages.len() == 1` over a single-device map) is the
//! unsplit fast path and is byte-identical to whole-model training.

use anyhow::{anyhow, Result};
use tch::Tensor;

use crate::runtime::architectures::ModelOperations;
use crate::runtime::device_pool::LayerDeviceMap;
use super::tenant_delta::TenantDelta;

/// A single pipeline stage: the stage model plus the global layer range it owns.
///
/// `range` is in **global** layer indices `[a..b)`; the stage model remaps to its
/// local `self.layers` via the `layer_offset` it was constructed with. Built from
/// `stage_from_weights_with_config` (#314).
pub struct TrainStage<'a> {
    /// The stage's model (owns embeddings iff `range.start == 0`; owns final
    /// norm + LM head iff `range.end == num_layers`).
    pub model: &'a dyn ModelOperations,
    /// Global layer range this stage owns.
    pub range: std::ops::Range<usize>,
}

/// Derive contiguous stage ranges from a [`LayerDeviceMap`].
///
/// Each maximal run of consecutive layers mapped to the same device becomes one
/// stage `[a..b)`. A single-device map yields exactly one stage `0..N`.
#[must_use]
pub fn stage_ranges(map: &LayerDeviceMap) -> Vec<std::ops::Range<usize>> {
    let n = map.len();
    let mut ranges = Vec::new();
    if n == 0 {
        return ranges;
    }
    let mut start = 0usize;
    let mut prev = map.device_for(0);
    for g in 1..n {
        let dev = map.device_for(g);
        if dev != prev {
            ranges.push(start..g);
            start = g;
            prev = dev;
        }
    }
    ranges.push(start..n);
    ranges
}

/// Run the TTT/training forward across pipeline stages and return logits with the
/// autograd graph intact across device boundaries.
///
/// `input_ids` is `[batch, seq]` and must already live on the first stage's
/// device. Stages must be contiguous, gap-free, and cover `0..num_layers` in
/// order (validated). The caller is responsible for wrapping this in
/// `tch::with_grad` and computing/backpropagating the loss.
pub fn forward_train_pipeline(
    stages: &[TrainStage<'_>],
    input_ids: &Tensor,
    delta: Option<&TenantDelta>,
) -> Result<Tensor> {
    if stages.is_empty() {
        return Err(anyhow!("forward_train_pipeline: no stages provided"));
    }
    // Validate contiguity and full coverage (the first stage must start at 0;
    // each subsequent stage must start where the previous ended).
    if stages[0].range.start != 0 {
        return Err(anyhow!(
            "forward_train_pipeline: first stage must start at layer 0 (got {:?})",
            stages[0].range
        ));
    }
    for w in stages.windows(2) {
        if w[0].range.end != w[1].range.start {
            return Err(anyhow!(
                "forward_train_pipeline: stages must be contiguous ({:?} then {:?})",
                w[0].range,
                w[1].range
            ));
        }
    }

    let first = &stages[0];
    // Stage 0 owns the embedding; subsequent stages consume the prior hidden.
    let mut hidden = first.model.embed_tokens(input_ids)?;
    hidden = first
        .model
        .forward_layers_train(&hidden, first.range.clone(), delta)?;

    for stage in &stages[1..] {
        hidden = stage
            .model
            .forward_layers_train(&hidden, stage.range.clone(), delta)?;
    }

    // The last stage owns the final norm + LM head.
    let last = stages
        .last()
        .ok_or_else(|| anyhow!("forward_train_pipeline: missing last stage"))?;
    hidden = last.model.apply_final_norm(&hidden)?;
    last.model.lm_head(&hidden)
}

/// Compute the next-token-prediction (NTP) cross-entropy loss for a TTT step over
/// a pipeline split, with the autograd graph intact across device boundaries.
///
/// Mirrors `TestTimeTrainer::compute_ntp_loss_with_delta` (the single-device
/// path) but drives the forward across `stages`. The returned loss tensor's
/// backward pass materializes gradients on each parameter's own device.
///
/// `input_ids` is `[1, seq]` on the first stage's device; the loss is computed on
/// that same device (the last stage's logits are produced on the last stage's
/// device, and the shifted targets are taken from `input_ids`, moved to the
/// logits' device).
pub fn compute_ntp_loss_split(
    stages: &[TrainStage<'_>],
    input_ids: &Tensor,
    delta: Option<&TenantDelta>,
) -> Result<Tensor> {
    tch::with_grad(|| {
        let logits = forward_train_pipeline(stages, input_ids, delta)?;

        let seq_len = input_ids.size()[1];
        if seq_len < 2 {
            return Err(anyhow!(
                "compute_ntp_loss_split: need at least 2 tokens for NTP loss (got {seq_len})"
            ));
        }
        let vocab_size = logits.size()[2];

        // Shift for next-token prediction. Targets live on `input_ids`' device;
        // move them to the logits' device (the last stage's) for cross_entropy.
        let pred_logits = logits.narrow(1, 0, seq_len - 1).reshape([-1, vocab_size]);
        let target_ids = input_ids
            .narrow(1, 1, seq_len - 1)
            .reshape([-1])
            .to_device(pred_logits.device());

        let loss = pred_logits.cross_entropy_loss::<Tensor>(
            &target_ids,
            None,
            tch::Reduction::Mean,
            -100,
            0.0,
        );
        Ok(loss)
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn stage_ranges_single_device_is_one_stage() {
        let map = LayerDeviceMap::single(Device::Cpu, 4).expect("map");
        assert_eq!(stage_ranges(&map), vec![0..4]);
    }

    #[test]
    fn stage_ranges_two_device_split() {
        let map = LayerDeviceMap::from_per_layer(vec![
            Device::Cpu,
            Device::Cpu,
            Device::Cuda(0),
            Device::Cuda(0),
        ])
        .expect("map");
        assert_eq!(stage_ranges(&map), vec![0..2, 2..4]);
    }

    #[test]
    fn stage_ranges_three_stages() {
        let map = LayerDeviceMap::from_per_layer(vec![
            Device::Cpu,
            Device::Cuda(0),
            Device::Cuda(0),
            Device::Cuda(1),
        ])
        .expect("map");
        assert_eq!(stage_ranges(&map), vec![0..1, 1..3, 3..4]);
    }
}
