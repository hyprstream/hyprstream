//! Thread-local execution boundary for one contiguous decoder stage.
//!
//! A stage owns exactly one global layer range. The first stage receives token
//! IDs and creates embeddings, middle stages receive an activation tensor, and
//! the last stage turns its activation into logits. Tensors therefore stay on
//! the engine-owning thread; a future transport boundary deals only in encoded
//! activations and never moves a `Tensor` across threads.

use std::ops::Range;
use std::path::Path;

use anyhow::{bail, Result};
use tch::{Device, Kind, Tensor};

use crate::runtime::architectures::ModelOperations;
use crate::runtime::model_factory::{ModelFactory, ModelStageRequest};
use crate::runtime::KVQuantType;

/// The range and endpoint responsibilities of one loaded decoder stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageContract {
    layer_range: Range<usize>,
    total_layers: usize,
}

impl StageContract {
    /// Validate a contiguous global layer range against the model layer count.
    pub fn new(layer_range: Range<usize>, total_layers: usize) -> Result<Self> {
        if total_layers == 0 {
            bail!("stage contract requires a model with at least one decoder layer");
        }
        if layer_range.start >= layer_range.end {
            bail!(
                "stage contract requires a non-empty layer range, got {:?}",
                layer_range
            );
        }
        if layer_range.end > total_layers {
            bail!(
                "stage contract range {:?} exceeds model layer count {}",
                layer_range,
                total_layers
            );
        }
        Ok(Self {
            layer_range,
            total_layers,
        })
    }

    /// Global decoder layers owned by this stage.
    #[must_use]
    pub fn layer_range(&self) -> Range<usize> {
        self.layer_range.clone()
    }

    /// Whether this stage owns the embedding table and accepts token IDs.
    #[must_use]
    pub fn is_first(&self) -> bool {
        self.layer_range.start == 0
    }

    /// Whether this stage owns final normalization and the language-model head.
    #[must_use]
    pub fn is_last(&self) -> bool {
        self.layer_range.end == self.total_layers
    }
}

/// Input admitted at a stage boundary.
pub enum StageInput<'a> {
    /// Token IDs, accepted only by the first stage.
    TokenIds(&'a Tensor),
    /// Hidden activation from the preceding stage, accepted only by a non-first stage.
    Hidden(&'a Tensor),
}

/// Result emitted at a stage boundary.
pub enum StageOutput {
    /// Activation for the next stage.
    Hidden(Tensor),
    /// Final vocabulary logits emitted by the last stage.
    Logits(Tensor),
}

/// A loaded, same-thread decoder stage.
///
/// Construction is the production consumer of [`ModelFactory::create_stage`].
/// The object intentionally has no transport implementation: callers must keep
/// it on their engine thread and serialize activation values at a process or
/// host boundary.
pub struct InferenceStage {
    model: Box<dyn ModelOperations>,
    contract: StageContract,
}

impl InferenceStage {
    /// Load exactly the weights required for one contiguous stage.
    pub async fn load(
        model_path: &Path,
        device: &Device,
        dtype: Kind,
        max_context: Option<usize>,
        kv_quant_type: KVQuantType,
        request: ModelStageRequest,
    ) -> Result<Self> {
        let layer_range = request.layer_range.clone();
        let model = ModelFactory::create_stage(
            model_path,
            device,
            dtype,
            max_context,
            kv_quant_type,
            None,
            request,
        )
        .await?;
        let contract = StageContract::new(layer_range, model.num_layers())?;
        Ok(Self { model, contract })
    }

    /// Immutable stage role and range contract.
    #[must_use]
    pub fn contract(&self) -> &StageContract {
        &self.contract
    }

    /// Execute one prefill or decode step for this stage.
    ///
    /// `start_pos` is the request-local cache position. The loaded model keeps
    /// its per-layer KV (and architecture-specific recurrent) state locally;
    /// only the returned hidden activation may cross to a later stage.
    pub fn execute(&self, input: StageInput<'_>, start_pos: usize) -> Result<StageOutput> {
        let hidden = match input {
            StageInput::TokenIds(token_ids) if self.contract.is_first() => {
                self.model.embed_tokens(token_ids)?
            }
            StageInput::TokenIds(_) => {
                bail!("token IDs are accepted only by the first decoder stage");
            }
            StageInput::Hidden(hidden) if !self.contract.is_first() => hidden.shallow_clone(),
            StageInput::Hidden(_) => {
                bail!("the first decoder stage requires token IDs, not a hidden activation");
            }
        };

        let hidden =
            self.model
                .forward_layers(&hidden, self.contract.layer_range(), start_pos, None)?;
        if self.contract.is_last() {
            let normalized = self.model.apply_final_norm(&hidden)?;
            Ok(StageOutput::Logits(self.model.lm_head(&normalized)?))
        } else {
            Ok(StageOutput::Hidden(hidden))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StageContract;

    fn valid_contract(range: std::ops::Range<usize>, total_layers: usize) -> StageContract {
        match StageContract::new(range, total_layers) {
            Ok(contract) => contract,
            Err(error) => panic!("test contract must be valid: {error}"),
        }
    }

    #[test]
    fn stage_contract_assigns_endpoint_roles_from_global_range() {
        let first = valid_contract(0..2, 6);
        assert!(first.is_first());
        assert!(!first.is_last());

        let middle = valid_contract(2..4, 6);
        assert!(!middle.is_first());
        assert!(!middle.is_last());

        let last = valid_contract(4..6, 6);
        assert!(!last.is_first());
        assert!(last.is_last());
    }

    #[test]
    fn stage_contract_rejects_empty_and_out_of_bounds_ranges() {
        assert!(StageContract::new(2..2, 6).is_err());
        assert!(StageContract::new(4..7, 6).is_err());
        assert!(StageContract::new(0..1, 0).is_err());
    }
}
