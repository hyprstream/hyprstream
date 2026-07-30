//! Pure-data activation envelope for inter-host pipeline stages (#324/N1).
//!
//! This module deliberately contains no transport, chunk splitting, reassembly,
//! credit, ordering, or tensor-execution code. It is crate-internal until one of
//! those reviewed consumers needs the contract.

use std::io::Cursor;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const ACTIVATION_ENVELOPE_VERSION: u16 = 1;
const MAX_ENCODED_ENVELOPE_BYTES: usize = 16 * 1024;
const MAX_TENSOR_RANK: usize = 16;
const MAX_ERROR_MESSAGE_BYTES: usize = 1024;

macro_rules! fixed_bytes_type {
    ($name:ident, $len:literal) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(transparent)]
        pub(crate) struct $name(pub(crate) [u8; $len]);

        impl $name {
            fn validate_nonzero(&self, label: &'static str) -> Result<(), ActivationEnvelopeError> {
                validate_nonzero(&self.0, label)
            }
        }
    };
}

fixed_bytes_type!(PlanDigest, 32);
fixed_bytes_type!(RequestId, 16);
fixed_bytes_type!(SessionId, 16);
fixed_bytes_type!(ChunkDigest, 32);
fixed_bytes_type!(WholeActivationDigest, 32);
fixed_bytes_type!(ProducerIdentity, 32);

/// Git object identity of the immutable model snapshot.
///
/// Both repository object formats are explicit on the wire; textual aliases or
/// abbreviated object IDs cannot enter the activation contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelOid {
    Sha1([u8; 20]),
    Sha256([u8; 32]),
}

impl ModelOid {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        match self {
            Self::Sha1(bytes) => validate_nonzero(bytes, "model OID"),
            Self::Sha256(bytes) => validate_nonzero(bytes, "model OID"),
        }
    }
}

/// The stage's position in the complete ordered pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StageRole {
    First,
    Interior,
    Last,
    Only,
}

/// Half-open model-layer interval `[start_layer, end_layer)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StageBoundary {
    pub(crate) start_layer: u32,
    pub(crate) end_layer: u32,
    pub(crate) role: StageRole,
}

impl StageBoundary {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        if self.start_layer >= self.end_layer {
            return Err(ActivationEnvelopeError::Invalid(
                "stage boundary must be a non-empty half-open interval",
            ));
        }
        match self.role {
            StageRole::First | StageRole::Only if self.start_layer != 0 => Err(
                ActivationEnvelopeError::Invalid("first/only stage must begin at layer zero"),
            ),
            StageRole::Interior | StageRole::Last if self.start_layer == 0 => Err(
                ActivationEnvelopeError::Invalid("interior/last stage cannot begin at layer zero"),
            ),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ActivationPhase {
    Prefill,
    Decode,
}

/// Stable scalar types accepted by activation metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ActivationDtype {
    Bool,
    Uint8,
    Int8,
    Float8E4m3Fn,
    Float8E5m2,
    Uint16,
    Int16,
    Float16,
    Bfloat16,
    Uint32,
    Int32,
    Float32,
    Uint64,
    Int64,
    Float64,
}

impl ActivationDtype {
    const fn byte_width(self) -> u64 {
        match self {
            Self::Bool | Self::Uint8 | Self::Int8 | Self::Float8E4m3Fn | Self::Float8E5m2 => 1,
            Self::Uint16 | Self::Int16 | Self::Float16 | Self::Bfloat16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorMetadata {
    pub(crate) dtype: ActivationDtype,
    pub(crate) shape: Vec<u64>,
    pub(crate) byte_length: u64,
}

impl TensorMetadata {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        if self.shape.is_empty() || self.shape.len() > MAX_TENSOR_RANK {
            return Err(ActivationEnvelopeError::Invalid(
                "tensor rank must be within 1..=16",
            ));
        }

        let elements = self.shape.iter().try_fold(1_u64, |product, dimension| {
            if *dimension == 0 {
                return None;
            }
            product.checked_mul(*dimension)
        });
        let expected_length = elements.and_then(|count| count.checked_mul(self.dtype.byte_width()));
        if expected_length != Some(self.byte_length) {
            return Err(ActivationEnvelopeError::Invalid(
                "tensor byte length does not match dtype and shape",
            ));
        }
        Ok(())
    }
}

/// Zero-based coordinate of one chunk in a complete activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChunkCoordinate {
    pub(crate) index: u32,
    pub(crate) total: u32,
}

impl ChunkCoordinate {
    fn validate(self) -> Result<(), ActivationEnvelopeError> {
        if self.total == 0 || self.index >= self.total {
            return Err(ActivationEnvelopeError::Invalid(
                "chunk coordinate requires total > 0 and index < total",
            ));
        }
        Ok(())
    }
}

/// Claimed producing identity and the key epoch under which it will be bound.
///
/// N1 carries this data only. Cryptographic host binding is explicitly N5.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ActivationProvenance {
    pub(crate) producer: ProducerIdentity,
    pub(crate) key_epoch: u64,
}

impl ActivationProvenance {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        self.producer.validate_nonzero("producer identity")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ActivationChunk {
    pub(crate) tensor: TensorMetadata,
    pub(crate) coordinate: ChunkCoordinate,
    pub(crate) chunk_digest: ChunkDigest,
    pub(crate) whole_activation_digest: WholeActivationDigest,
}

impl ActivationChunk {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        self.tensor.validate()?;
        self.coordinate.validate()?;
        self.chunk_digest.validate_nonzero("chunk digest")?;
        self.whole_activation_digest
            .validate_nonzero("whole activation digest")
    }
}

/// Successful terminal marker following the final activation chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ActivationTerminal {
    pub(crate) chunk_count: u32,
    pub(crate) whole_activation_digest: WholeActivationDigest,
}

impl ActivationTerminal {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        if self.chunk_count == 0 {
            return Err(ActivationEnvelopeError::Invalid(
                "terminal chunk count must be non-zero",
            ));
        }
        self.whole_activation_digest
            .validate_nonzero("whole activation digest")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ActivationErrorCode {
    InvalidRequest,
    StaleEpoch,
    OutOfOrder,
    Integrity,
    Cancelled,
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ActivationError {
    pub(crate) code: ActivationErrorCode,
    pub(crate) message: String,
    pub(crate) retryable: bool,
}

impl ActivationError {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        let bytes = self.message.as_bytes();
        if bytes.is_empty()
            || bytes.len() > MAX_ERROR_MESSAGE_BYTES
            || self.message.chars().any(char::is_control)
        {
            return Err(ActivationEnvelopeError::Invalid(
                "error message must contain 1..=1024 non-control bytes",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ActivationEnvelopeBody {
    Chunk(ActivationChunk),
    Terminal(ActivationTerminal),
    Error(ActivationError),
}

impl ActivationEnvelopeBody {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        match self {
            Self::Chunk(chunk) => chunk.validate(),
            Self::Terminal(terminal) => terminal.validate(),
            Self::Error(error) => error.validate(),
        }
    }
}

/// Complete identity, pipeline position, tensor, integrity, and terminal data
/// required to carry one activation protocol record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ActivationEnvelope {
    pub(crate) version: u16,
    pub(crate) model_oid: ModelOid,
    pub(crate) plan_digest: PlanDigest,
    pub(crate) request_id: RequestId,
    pub(crate) session_id: SessionId,
    pub(crate) pipeline_epoch: u64,
    pub(crate) microbatch_index: u32,
    pub(crate) sequence_index: u32,
    pub(crate) stage: StageBoundary,
    pub(crate) phase: ActivationPhase,
    pub(crate) start_position: u64,
    pub(crate) accepted_token_index: Option<u64>,
    pub(crate) provenance: ActivationProvenance,
    pub(crate) body: ActivationEnvelopeBody,
}

impl ActivationEnvelope {
    /// Serialize one validated envelope as CBOR.
    pub(crate) fn to_cbor(&self) -> Result<Vec<u8>, ActivationEnvelopeError> {
        self.validate()?;
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(self, &mut bytes)
            .map_err(|error| ActivationEnvelopeError::Encode(error.to_string()))?;
        if bytes.len() > MAX_ENCODED_ENVELOPE_BYTES {
            return Err(ActivationEnvelopeError::TooLarge(bytes.len()));
        }
        Ok(bytes)
    }

    /// Decode exactly one bounded CBOR envelope and validate all cross-fields.
    pub(crate) fn from_cbor(bytes: &[u8]) -> Result<Self, ActivationEnvelopeError> {
        if bytes.len() > MAX_ENCODED_ENVELOPE_BYTES {
            return Err(ActivationEnvelopeError::TooLarge(bytes.len()));
        }

        let mut reader = Cursor::new(bytes);
        let envelope: Self = ciborium::de::from_reader(&mut reader)
            .map_err(|error| ActivationEnvelopeError::Decode(error.to_string()))?;
        if reader.position() != bytes.len() as u64 {
            return Err(ActivationEnvelopeError::TrailingData);
        }
        envelope.validate()?;
        Ok(envelope)
    }

    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        if self.version != ACTIVATION_ENVELOPE_VERSION {
            return Err(ActivationEnvelopeError::Invalid(
                "unsupported activation envelope version",
            ));
        }
        self.model_oid.validate()?;
        self.plan_digest.validate_nonzero("plan digest")?;
        self.request_id.validate_nonzero("request identity")?;
        self.session_id.validate_nonzero("session identity")?;
        self.stage.validate()?;
        match (self.phase, self.accepted_token_index) {
            (ActivationPhase::Prefill, Some(_)) => {
                return Err(ActivationEnvelopeError::Invalid(
                    "prefill envelope cannot carry an accepted-token index",
                ));
            }
            (ActivationPhase::Decode, None) => {
                return Err(ActivationEnvelopeError::Invalid(
                    "decode envelope requires an accepted-token index",
                ));
            }
            _ => {}
        }
        self.provenance.validate()?;
        self.body.validate()
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum ActivationEnvelopeError {
    #[error("activation envelope CBOR encode failed: {0}")]
    Encode(String),
    #[error("activation envelope CBOR decode failed: {0}")]
    Decode(String),
    #[error("activation envelope is invalid: {0}")]
    Invalid(&'static str),
    #[error("activation envelope exceeds the {MAX_ENCODED_ENVELOPE_BYTES}-byte metadata cap: {0}")]
    TooLarge(usize),
    #[error("activation envelope contains trailing data")]
    TrailingData,
}

fn validate_nonzero(bytes: &[u8], label: &'static str) -> Result<(), ActivationEnvelopeError> {
    if bytes.iter().all(|byte| *byte == 0) {
        return Err(ActivationEnvelopeError::Invalid(label));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn chunk_envelope() -> ActivationEnvelope {
        ActivationEnvelope {
            version: ACTIVATION_ENVELOPE_VERSION,
            model_oid: ModelOid::Sha1([1; 20]),
            plan_digest: PlanDigest([2; 32]),
            request_id: RequestId([3; 16]),
            session_id: SessionId([4; 16]),
            pipeline_epoch: 7,
            microbatch_index: 11,
            sequence_index: 13,
            stage: StageBoundary {
                start_layer: 8,
                end_layer: 16,
                role: StageRole::Interior,
            },
            phase: ActivationPhase::Decode,
            start_position: 21,
            accepted_token_index: Some(20),
            provenance: ActivationProvenance {
                producer: ProducerIdentity([5; 32]),
                key_epoch: 9,
            },
            body: ActivationEnvelopeBody::Chunk(ActivationChunk {
                tensor: TensorMetadata {
                    dtype: ActivationDtype::Bfloat16,
                    shape: vec![2, 4, 8],
                    byte_length: 128,
                },
                coordinate: ChunkCoordinate { index: 1, total: 3 },
                chunk_digest: ChunkDigest([6; 32]),
                whole_activation_digest: WholeActivationDigest([7; 32]),
            }),
        }
    }

    fn unchecked_cbor(envelope: &ActivationEnvelope) -> Vec<u8> {
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(envelope, &mut bytes).unwrap();
        bytes
    }

    fn assert_decode_invalid(envelope: &ActivationEnvelope, expected: &'static str) {
        let error = ActivationEnvelope::from_cbor(&unchecked_cbor(envelope)).unwrap_err();
        assert_eq!(error, ActivationEnvelopeError::Invalid(expected));
    }

    #[test]
    fn chunk_round_trip_preserves_every_contract_field() {
        let envelope = chunk_envelope();
        let encoded = envelope.to_cbor().unwrap();
        let decoded = ActivationEnvelope::from_cbor(&encoded).unwrap();
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn terminal_and_error_variants_round_trip() {
        let mut terminal = chunk_envelope();
        terminal.phase = ActivationPhase::Prefill;
        terminal.start_position = 0;
        terminal.accepted_token_index = None;
        terminal.stage = StageBoundary {
            start_layer: 0,
            end_layer: 32,
            role: StageRole::Only,
        };
        terminal.body = ActivationEnvelopeBody::Terminal(ActivationTerminal {
            chunk_count: 3,
            whole_activation_digest: WholeActivationDigest([8; 32]),
        });

        let mut error = terminal.clone();
        error.body = ActivationEnvelopeBody::Error(ActivationError {
            code: ActivationErrorCode::Integrity,
            message: "whole activation digest mismatch".to_owned(),
            retryable: false,
        });

        for envelope in [terminal, error] {
            let encoded = envelope.to_cbor().unwrap();
            assert_eq!(ActivationEnvelope::from_cbor(&encoded).unwrap(), envelope);
        }
    }

    #[test]
    fn rejects_invalid_stage_boundary() {
        let mut envelope = chunk_envelope();
        envelope.stage.end_layer = envelope.stage.start_layer;
        assert_decode_invalid(
            &envelope,
            "stage boundary must be a non-empty half-open interval",
        );
    }

    #[test]
    fn rejects_phase_token_causality_mismatch() {
        let mut envelope = chunk_envelope();
        envelope.accepted_token_index = None;
        assert_decode_invalid(
            &envelope,
            "decode envelope requires an accepted-token index",
        );

        envelope.phase = ActivationPhase::Prefill;
        envelope.accepted_token_index = Some(0);
        assert_decode_invalid(
            &envelope,
            "prefill envelope cannot carry an accepted-token index",
        );
    }

    #[test]
    fn rejects_tensor_length_mismatch_and_overflow() {
        let mut envelope = chunk_envelope();
        if let ActivationEnvelopeBody::Chunk(chunk) = &mut envelope.body {
            chunk.tensor.byte_length = 127;
        } else {
            panic!("fixture is a chunk");
        }
        assert_decode_invalid(
            &envelope,
            "tensor byte length does not match dtype and shape",
        );

        if let ActivationEnvelopeBody::Chunk(chunk) = &mut envelope.body {
            chunk.tensor.shape = vec![u64::MAX, 2];
        } else {
            panic!("fixture is a chunk");
        }
        assert_decode_invalid(
            &envelope,
            "tensor byte length does not match dtype and shape",
        );
    }

    #[test]
    fn rejects_out_of_range_chunk_coordinate() {
        let mut envelope = chunk_envelope();
        let ActivationEnvelopeBody::Chunk(chunk) = &mut envelope.body else {
            panic!("fixture is a chunk");
        };
        chunk.coordinate = ChunkCoordinate { index: 3, total: 3 };
        assert_decode_invalid(
            &envelope,
            "chunk coordinate requires total > 0 and index < total",
        );
    }

    #[test]
    fn rejects_zero_integrity_and_identity_fields() {
        let mut envelope = chunk_envelope();
        envelope.plan_digest = PlanDigest([0; 32]);
        assert_decode_invalid(&envelope, "plan digest");

        envelope.plan_digest = PlanDigest([2; 32]);
        envelope.provenance.producer = ProducerIdentity([0; 32]);
        assert_decode_invalid(&envelope, "producer identity");
    }

    #[test]
    fn rejects_trailing_and_oversized_input() {
        let envelope = chunk_envelope();
        let mut encoded = envelope.to_cbor().unwrap();
        encoded.push(0);
        assert_eq!(
            ActivationEnvelope::from_cbor(&encoded).unwrap_err(),
            ActivationEnvelopeError::TrailingData
        );

        let oversized = vec![0; MAX_ENCODED_ENVELOPE_BYTES + 1];
        assert_eq!(
            ActivationEnvelope::from_cbor(&oversized).unwrap_err(),
            ActivationEnvelopeError::TooLarge(MAX_ENCODED_ENVELOPE_BYTES + 1)
        );
    }
}
