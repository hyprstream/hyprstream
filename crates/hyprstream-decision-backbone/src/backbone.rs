//! The backbone trait shared by the converted Qwen3.5 hybrid and the
//! ModernBERT-large baseline arm, plus the attention-mask policy that encodes the
//! LLM2Vec conversion pin.

use tch::Tensor;

use crate::Error;

/// Attention-mask policy for a softmax (full-attention) layer.
///
/// The LLM2Vec-style conversion flips softmax layers to [`Bidirectional`](Self::Bidirectional);
/// GatedDeltaNet layers have no mask policy — the chunked delta rule is causal by
/// construction and stays causal (the partial precedent the plan pins: 18 of 24
/// layers unchanged).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaskPolicy {
    /// Standard decoder self-attention: position *t* attends to positions ≤ *t*.
    #[default]
    Causal,
    /// LLM2Vec conversion: every position attends to every position. Applied only
    /// to softmax attention layers of the converted backbone.
    Bidirectional,
}

impl MaskPolicy {
    /// Build the additive attention mask `[1, 1, seq, seq]` for this policy: `0.0`
    /// where attention is allowed, `-inf` where it is not.
    pub(crate) fn additive_mask(
        self,
        seq: i64,
        kind: tch::Kind,
        device: tch::Device,
    ) -> Result<Tensor, tch::TchError> {
        match self {
            // A zero mask keeps softmax numerics identical to the unmasked case
            // while making the policy explicit at the call site.
            MaskPolicy::Bidirectional => Ok(Tensor::zeros([1, 1, seq, seq], (kind, device))),
            MaskPolicy::Causal => {
                let future = Tensor::ones([seq, seq], (tch::Kind::Bool, device)).triu(1);
                Ok(Tensor::zeros([1, 1, seq, seq], (kind, device))
                    .f_masked_fill(&future.view([1, 1, seq, seq]), f64::NEG_INFINITY)?)
            }
        }
    }
}

/// An encoder backbone: token ids in, final hidden states out. The shared scorer
/// consumes the hidden states at anchor positions; P1.4 trains through the same
/// trait on both arms (converted hybrid and ModernBERT baseline).
pub trait Backbone {
    /// Forward pass: `token_ids` is `[batch, seq]` of `Int64`; returns
    /// `[batch, seq, hidden_dim]` final hidden states (post final norm).
    fn forward(&self, token_ids: &Tensor) -> Result<Tensor, Error>;

    /// The hidden size of the returned states (the scorer's input dim).
    fn hidden_dim(&self) -> i64;
}
