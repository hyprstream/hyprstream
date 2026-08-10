//! Thread-local execution boundary for one contiguous decoder stage.
//!
//! A stage owns exactly one global layer range. The first stage receives token
//! IDs and creates embeddings, middle stages receive an activation tensor, and
//! the last stage turns its activation into logits. The stage object and its
//! typed output are deliberately !Send; a future transport boundary must encode
//! activations instead of moving this execution object between threads.

use std::marker::PhantomData;
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, bail, Result};
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
    Hidden(StageTensor),
    /// Final vocabulary logits emitted by the last stage.
    Logits(StageTensor),
}

/// A stage result that remains attached to the engine-owning thread.
pub struct StageTensor {
    tensor: Tensor,
    _not_send: PhantomData<Rc<()>>,
}

impl StageTensor {
    fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            _not_send: PhantomData,
        }
    }

    /// Borrow the tensor on the same engine-owning thread for encoding or the
    /// next local stage call.
    pub fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }
}

/// The single cache timeline owned by one loaded stage.
///
/// A stage instance admits one sequence only. This prevents two request streams
/// from accidentally sharing the model-local KV/SSM state while the lifecycle
/// service has not yet installed per-request cache ownership.
#[derive(Debug)]
pub struct StageSequence {
    id: u64,
    next_start_pos: usize,
}

impl StageSequence {
    /// Start a fresh stage-local cache timeline at position zero.
    #[must_use]
    pub fn new() -> Self {
        static NEXT_STAGE_SEQUENCE_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT_STAGE_SEQUENCE_ID.fetch_add(1, Ordering::Relaxed),
            next_start_pos: 0,
        }
    }

    fn require_start_pos(&self, start_pos: usize) -> Result<()> {
        if start_pos != self.next_start_pos {
            bail!(
                "stage sequence expected start_pos {}, got {}",
                self.next_start_pos,
                start_pos
            );
        }
        Ok(())
    }

    fn advance(&mut self, tokens: usize) -> Result<()> {
        self.next_start_pos = self
            .next_start_pos
            .checked_add(tokens)
            .ok_or_else(|| anyhow!("stage sequence position overflow"))?;
        Ok(())
    }
}

impl Default for StageSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// A loaded, same-thread decoder stage.
///
/// Construction is the production consumer of [`ModelFactory::create_stage`].
/// One instance is deliberately bound to one [`StageSequence`] and therefore
/// one model-local cache timeline. The object intentionally has no transport
/// implementation: callers encode activations at a process or host boundary.
pub struct InferenceStage {
    model: Box<dyn ModelOperations>,
    contract: StageContract,
    active_sequence: Option<u64>,
    _not_send: PhantomData<Rc<()>>,
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
        Ok(Self::from_loaded_model(model, contract))
    }

    fn from_loaded_model(model: Box<dyn ModelOperations>, contract: StageContract) -> Self {
        Self {
            model,
            contract,
            active_sequence: None,
            _not_send: PhantomData,
        }
    }

    /// Immutable stage role and range contract.
    #[must_use]
    pub fn contract(&self) -> &StageContract {
        &self.contract
    }

    /// Execute one prefill or decode step for this stage.
    ///
    /// `sequence` is the sole cache timeline admitted by this stage instance.
    /// `start_pos` must advance monotonically from zero for that sequence. The
    /// loaded model keeps its per-layer KV (and architecture-specific recurrent)
    /// state locally; the typed result stays on the engine-owning thread.
    pub fn execute(
        &mut self,
        sequence: &mut StageSequence,
        input: StageInput<'_>,
        start_pos: usize,
    ) -> Result<StageOutput> {
        sequence.require_start_pos(start_pos)?;

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
        match self.active_sequence {
            Some(active) if active != sequence.id => {
                bail!("a loaded stage instance admits exactly one sequence");
            }
            Some(_) => {}
            None => self.active_sequence = Some(sequence.id),
        }

        let hidden =
            self.model
                .forward_layers(&hidden, self.contract.layer_range(), start_pos, None)?;
        let token_count = hidden
            .size()
            .get(1)
            .copied()
            .ok_or_else(|| anyhow!("stage output must have a sequence dimension"))?;
        let token_count = usize::try_from(token_count)
            .map_err(|_| anyhow!("stage output sequence dimension must be non-negative"))?;
        sequence.advance(token_count)?;
        if self.contract.is_last() {
            let normalized = self.model.apply_final_norm(&hidden)?;
            Ok(StageOutput::Logits(StageTensor::new(
                self.model.lm_head(&normalized)?,
            )))
        } else {
            Ok(StageOutput::Hidden(StageTensor::new(hidden)))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;

    use anyhow::{anyhow, Result};
    use parking_lot::Mutex;
    use tch::{Device, Kind, Tensor};

    use super::{InferenceStage, StageContract, StageInput, StageOutput, StageSequence};
    use crate::runtime::architectures::config::ArchitectureConfig;
    use crate::runtime::architectures::{ModelArchitecture, ModelOperations};

    fn valid_contract(range: std::ops::Range<usize>, total_layers: usize) -> StageContract {
        match StageContract::new(range, total_layers) {
            Ok(contract) => contract,
            Err(error) => panic!("test contract must be valid: {error}"),
        }
    }

    #[derive(Clone)]
    struct TestConfig;

    impl ArchitectureConfig for TestConfig {
        fn num_attention_heads(&self) -> usize {
            1
        }

        fn num_key_value_heads(&self) -> usize {
            1
        }

        fn hidden_size(&self) -> usize {
            4
        }

        fn intermediate_size(&self) -> usize {
            8
        }

        fn vocab_size(&self) -> usize {
            16
        }

        fn max_position_embeddings(&self) -> usize {
            16
        }

        fn rope_theta(&self) -> Option<f32> {
            None
        }

        fn rope_dim(&self) -> Option<usize> {
            None
        }

        fn layer_norm_eps(&self) -> f32 {
            1e-5
        }

        fn use_rms_norm(&self) -> bool {
            true
        }
    }

    struct TestModel {
        config: TestConfig,
        total_layers: usize,
        observed_ranges: Arc<Mutex<Vec<Range<usize>>>>,
    }

    impl ModelOperations for TestModel {
        fn architecture(&self) -> ModelArchitecture {
            ModelArchitecture::Llama { version: 3 }
        }

        fn config(&self) -> &dyn ArchitectureConfig {
            &self.config
        }

        fn forward(&self, _input: &Tensor, _past_kv: Option<&Tensor>) -> Result<Tensor> {
            Err(anyhow!("whole-model forward is not used by a stage"))
        }

        fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
            Ok(Tensor::zeros(
                [input_ids.size()[0], input_ids.size()[1], 4],
                (Kind::Float, input_ids.device()),
            ))
        }

        fn forward_layers(
            &self,
            hidden: &Tensor,
            range: Range<usize>,
            _start_pos: usize,
            _delta: Option<&crate::training::TenantDelta>,
        ) -> Result<Tensor> {
            self.observed_ranges.lock().push(range);
            Ok(hidden.shallow_clone())
        }

        fn apply_final_norm(&self, hidden: &Tensor) -> Result<Tensor> {
            Ok(hidden.shallow_clone())
        }

        fn lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
            Ok(hidden.shallow_clone())
        }

        fn num_layers(&self) -> usize {
            self.total_layers
        }

        fn reshape_for_attention(&self, tensor: &Tensor, _is_key_value: bool) -> Result<Tensor> {
            Ok(tensor.shallow_clone())
        }

        fn apply_rope(&self, tensor: &Tensor, _position_ids: &Tensor) -> Result<Tensor> {
            Ok(tensor.shallow_clone())
        }

        fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
            Ok(tensor.shallow_clone())
        }

        fn get_attention_mask(&self, _seq_len: usize, _past_kv_len: usize) -> Result<Tensor> {
            Ok(Tensor::zeros([1], (Kind::Float, Device::Cpu)))
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn test_stage(
        range: Range<usize>,
        total_layers: usize,
    ) -> (InferenceStage, Arc<Mutex<Vec<Range<usize>>>>) {
        let observed_ranges = Arc::new(Mutex::new(Vec::new()));
        let model = TestModel {
            config: TestConfig,
            total_layers,
            observed_ranges: Arc::clone(&observed_ranges),
        };
        let stage =
            InferenceStage::from_loaded_model(Box::new(model), valid_contract(range, total_layers));
        (stage, observed_ranges)
    }

    fn token_ids() -> Tensor {
        Tensor::zeros([1, 2], (Kind::Int64, Device::Cpu))
    }

    fn hidden() -> Tensor {
        Tensor::zeros([1, 2, 4], (Kind::Float, Device::Cpu))
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

    #[test]
    fn stage_execute_enforces_endpoint_inputs_and_forwards_its_global_range() {
        let (mut first, first_ranges) = test_stage(0..2, 6);
        let mut first_sequence = StageSequence::new();
        let first_output =
            match first.execute(&mut first_sequence, StageInput::TokenIds(&token_ids()), 0) {
                Ok(output) => output,
                Err(error) => panic!("first stage must execute: {error}"),
            };
        match first_output {
            StageOutput::Hidden(hidden) => assert_eq!(hidden.as_tensor().size(), [1, 2, 4]),
            StageOutput::Logits(_) => panic!("a non-final stage must emit hidden activation"),
        }
        assert_eq!(*first_ranges.lock(), vec![0..2]);

        let (mut middle, _) = test_stage(2..4, 6);
        let mut middle_sequence = StageSequence::new();
        assert!(middle
            .execute(&mut middle_sequence, StageInput::TokenIds(&token_ids()), 0)
            .is_err());

        let (mut final_stage, final_ranges) = test_stage(4..6, 6);
        let mut final_sequence = StageSequence::new();
        let final_output =
            match final_stage.execute(&mut final_sequence, StageInput::Hidden(&hidden()), 0) {
                Ok(output) => output,
                Err(error) => panic!("final stage must execute: {error}"),
            };
        match final_output {
            StageOutput::Logits(logits) => assert_eq!(logits.as_tensor().size(), [1, 2, 4]),
            StageOutput::Hidden(_) => panic!("the final stage must emit logits"),
        }
        assert_eq!(*final_ranges.lock(), vec![4..6]);
    }

    #[test]
    fn stage_execute_rejects_wrong_input_and_a_second_cache_timeline() {
        let (mut first, _) = test_stage(0..2, 6);
        let mut sequence = StageSequence::new();
        assert!(first
            .execute(&mut sequence, StageInput::Hidden(&hidden()), 0)
            .is_err());

        let mut first_sequence = StageSequence::new();
        if let Err(error) =
            first.execute(&mut first_sequence, StageInput::TokenIds(&token_ids()), 0)
        {
            panic!("valid first-stage request must execute: {error}");
        }
        let mut second_sequence = StageSequence::new();
        assert!(first
            .execute(&mut second_sequence, StageInput::TokenIds(&token_ids()), 0)
            .is_err());
        assert!(first
            .execute(&mut first_sequence, StageInput::TokenIds(&token_ids()), 0)
            .is_err());
    }
}
