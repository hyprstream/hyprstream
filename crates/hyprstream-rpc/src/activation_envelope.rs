//! Pure-data activation envelope for inter-host pipeline stages (#324/N1).
//!
//! This module deliberately contains no transport, chunk splitting, reassembly,
//! credit, execution-ordering, cancellation, or tensor-execution code. It is
//! crate-internal until one of those reviewed consumers needs the contract.
//!
//! # Deterministic wire profile
//!
//! The wire representation is a fixed, numeric-key CBOR map. Keys are exactly
//! `0..=18`, in RFC 8949 deterministic order; every nested record is a fixed
//! array. Integers and container lengths use shortest-form encodings, maps and
//! arrays are definite-length, and byte/text fields are bounded before
//! allocation. Decode also re-encodes and compares byte-for-byte, so alternate
//! spellings never silently normalize into the same security identity.

use thiserror::Error;

const ACTIVATION_ENVELOPE_VERSION: u16 = 1;
const ENVELOPE_FIELDS: usize = 19;
const MAX_ENCODED_ENVELOPE_BYTES: usize = 16 * 1024;
const MAX_TENANT_ID_BYTES: usize = 512;
const MAX_ERROR_MESSAGE_BYTES: usize = 1024;

/// All v1 digests are BLAKE3-256. Later consumers must prepend the matching
/// domain before hashing; the algorithm/domain are not negotiated on the wire.
pub(crate) const DIGEST_ALGORITHM: &str = "BLAKE3-256";
pub(crate) const PLAN_DIGEST_DOMAIN: &str = "hyprstream activation plan digest v1";
pub(crate) const CHUNK_DIGEST_DOMAIN: &str = "hyprstream activation chunk digest v1";
pub(crate) const WHOLE_ACTIVATION_DIGEST_DOMAIN: &str = "hyprstream whole activation digest v1";

macro_rules! fixed_bytes_type {
    ($name:ident, $len:literal) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
fixed_bytes_type!(DeltaDigest, 32);
fixed_bytes_type!(ChunkDigest, 32);
fixed_bytes_type!(WholeActivationDigest, 32);
fixed_bytes_type!(ProducerIdentity, 32);

/// Git object identity of the immutable model snapshot.
///
/// Both repository object formats are explicit on the wire; abbreviated or
/// textual object IDs cannot enter the activation identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Exact tenant identity used for authorization and state isolation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TenantId(pub(crate) String);

impl TenantId {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        validate_visible_text(
            &self.0,
            MAX_TENANT_ID_BYTES,
            "tenant identity must contain 1..=512 non-whitespace bytes",
        )
    }
}

/// Exact delta identity admitted for this invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeltaRevision {
    BaseModel,
    Digest(DeltaDigest),
}

impl DeltaRevision {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        match self {
            Self::BaseModel => Ok(()),
            Self::Digest(digest) => digest.validate_nonzero("delta revision digest"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StageId(pub(crate) u32);

/// The source stage's position in the complete ordered pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StageRole {
    First,
    Interior,
    Last,
    Only,
}

/// Source stage's half-open global model-layer interval `[start_layer, end_layer)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StageBoundary {
    pub(crate) start_layer: u32,
    pub(crate) end_layer: u32,
    pub(crate) role: StageRole,
}

impl StageBoundary {
    fn validate(
        &self,
        source_stage_id: StageId,
        destination_stage_id: StageId,
    ) -> Result<(), ActivationEnvelopeError> {
        if self.start_layer >= self.end_layer {
            return Err(ActivationEnvelopeError::Invalid(
                "stage boundary must be a non-empty half-open interval",
            ));
        }

        match self.role {
            StageRole::First if source_stage_id.0 != 0 || self.start_layer != 0 => {
                return Err(ActivationEnvelopeError::Invalid(
                    "first stage must have identity and start layer zero",
                ));
            }
            StageRole::Interior if source_stage_id.0 == 0 || self.start_layer == 0 => {
                return Err(ActivationEnvelopeError::Invalid(
                    "interior stage must have non-zero identity and start layer",
                ));
            }
            StageRole::Last | StageRole::Only => {
                return Err(ActivationEnvelopeError::Invalid(
                    "last/only stage does not emit a next-stage activation",
                ));
            }
            _ => {}
        }

        let expected_destination =
            source_stage_id
                .0
                .checked_add(1)
                .ok_or(ActivationEnvelopeError::Invalid(
                    "source stage identity cannot advance without overflow",
                ))?;
        if destination_stage_id.0 != expected_destination {
            return Err(ActivationEnvelopeError::Invalid(
                "destination stage must immediately follow source stage",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActivationPhase {
    Prefill,
    Decode,
}

/// Scalar types admitted by the dense activation ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActivationDtype {
    Float16,
    Bfloat16,
    Float32,
}

impl ActivationDtype {
    const fn byte_width(self) -> u64 {
        match self {
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Float32 => 4,
        }
    }
}

/// The sole v1 activation codec: dense row-major `[B,Q,H]`, with each scalar
/// encoded in the admitted dtype's IEEE/raw little-endian representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActivationCodec {
    RawLittleEndian,
}

/// Authoritative dense hidden-state ABI. Rank is structurally fixed at three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TensorMetadata {
    pub(crate) dtype: ActivationDtype,
    pub(crate) shape: [u64; 3],
    pub(crate) byte_length: u64,
    pub(crate) codec: ActivationCodec,
}

impl TensorMetadata {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        let [batch, query, hidden] = self.shape;
        if batch == 0 || query == 0 || hidden == 0 {
            return Err(ActivationEnvelopeError::Invalid(
                "activation shape [B,Q,H] requires three non-zero dimensions",
            ));
        }

        let expected_length = batch
            .checked_mul(query)
            .and_then(|elements| elements.checked_mul(hidden))
            .and_then(|elements| elements.checked_mul(self.dtype.byte_width()));
        if expected_length != Some(self.byte_length) {
            return Err(ActivationEnvelopeError::Invalid(
                "activation byte length does not match dense [B,Q,H] dtype",
            ));
        }
        Ok(())
    }
}

/// Zero-based coordinate and exact logical byte span of one chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChunkCoordinate {
    pub(crate) index: u32,
    pub(crate) count: u32,
    pub(crate) offset: u64,
    pub(crate) length: u64,
}

impl ChunkCoordinate {
    fn validate(self, logical_length: u64) -> Result<(), ActivationEnvelopeError> {
        if self.count == 0 || self.index >= self.count {
            return Err(ActivationEnvelopeError::Invalid(
                "chunk coordinate requires count > 0 and index < count",
            ));
        }
        if self.length == 0 {
            return Err(ActivationEnvelopeError::Invalid(
                "chunk length must be non-zero",
            ));
        }
        let end = self
            .offset
            .checked_add(self.length)
            .ok_or(ActivationEnvelopeError::Invalid(
                "chunk byte span overflows",
            ))?;
        if end > logical_length {
            return Err(ActivationEnvelopeError::Invalid(
                "chunk byte span exceeds logical activation length",
            ));
        }
        if self.index == 0 && self.offset != 0 {
            return Err(ActivationEnvelopeError::Invalid(
                "first chunk must begin at logical byte zero",
            ));
        }
        if self.index + 1 == self.count && end != logical_length {
            return Err(ActivationEnvelopeError::Invalid(
                "last chunk must end at the logical activation length",
            ));
        }
        Ok(())
    }
}

/// Claimed producing identity and the key epoch under which it will be bound.
///
/// N1 carries this data only. Cryptographic host binding is explicitly N5.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActivationProvenance {
    pub(crate) producer: ProducerIdentity,
    pub(crate) key_epoch: u64,
}

impl ActivationProvenance {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        self.producer.validate_nonzero("producer identity")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActivationChunk {
    pub(crate) tensor: TensorMetadata,
    pub(crate) coordinate: ChunkCoordinate,
    pub(crate) chunk_digest: ChunkDigest,
    pub(crate) whole_activation_digest: WholeActivationDigest,
}

impl ActivationChunk {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        self.tensor.validate()?;
        self.coordinate.validate(self.tensor.byte_length)?;
        self.chunk_digest.validate_nonzero("chunk digest")?;
        self.whole_activation_digest
            .validate_nonzero("whole activation digest")
    }
}

/// Successful terminal marker following the final activation chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActivationErrorCode {
    InvalidRequest,
    StaleEpoch,
    OutOfOrder,
    Integrity,
    Cancelled,
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ActivationError {
    pub(crate) code: ActivationErrorCode,
    pub(crate) message: String,
    pub(crate) retryable: bool,
}

impl ActivationError {
    fn validate(&self) -> Result<(), ActivationEnvelopeError> {
        validate_visible_text(
            &self.message,
            MAX_ERROR_MESSAGE_BYTES,
            "error message must contain 1..=1024 non-whitespace bytes",
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
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

/// Complete identity, ordering position, direction, tensor, integrity, and
/// terminal data required for one activation protocol record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ActivationEnvelope {
    pub(crate) version: u16,
    pub(crate) model_oid: ModelOid,
    pub(crate) plan_digest: PlanDigest,
    pub(crate) request_id: RequestId,
    pub(crate) session_id: SessionId,
    pub(crate) tenant_id: TenantId,
    pub(crate) delta_revision: DeltaRevision,
    pub(crate) pipeline_epoch: u64,
    pub(crate) microbatch_id: u64,
    pub(crate) sequence_id: u64,
    pub(crate) activation_seq: u64,
    pub(crate) source_stage_id: StageId,
    pub(crate) destination_stage_id: StageId,
    pub(crate) source_boundary: StageBoundary,
    pub(crate) phase: ActivationPhase,
    pub(crate) start_position: u64,
    pub(crate) accepted_token_index: Option<u64>,
    pub(crate) provenance: ActivationProvenance,
    pub(crate) body: ActivationEnvelopeBody,
}

impl ActivationEnvelope {
    /// Serialize one semantically valid envelope in the sole canonical v1 form.
    pub(crate) fn to_cbor(&self) -> Result<Vec<u8>, ActivationEnvelopeError> {
        self.validate()?;
        let bytes = self.encode_canonical();
        if bytes.len() > MAX_ENCODED_ENVELOPE_BYTES {
            return Err(ActivationEnvelopeError::TooLarge(bytes.len()));
        }
        Ok(bytes)
    }

    /// Preflight, decode, validate, and re-encode exactly one canonical envelope.
    pub(crate) fn from_cbor(bytes: &[u8]) -> Result<Self, ActivationEnvelopeError> {
        if bytes.len() > MAX_ENCODED_ENVELOPE_BYTES {
            return Err(ActivationEnvelopeError::TooLarge(bytes.len()));
        }

        let envelope = Self::decode_preflight(bytes)?;
        envelope.validate()?;
        if envelope.encode_canonical() != bytes {
            return Err(ActivationEnvelopeError::NonCanonical);
        }
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
        self.tenant_id.validate()?;
        self.delta_revision.validate()?;
        self.source_boundary
            .validate(self.source_stage_id, self.destination_stage_id)?;

        match (self.phase, self.activation_seq, self.accepted_token_index) {
            (ActivationPhase::Prefill, 0, None) => {}
            (ActivationPhase::Prefill, _, _) => {
                return Err(ActivationEnvelopeError::Invalid(
                    "prefill requires activation sequence zero and no accepted-token index",
                ));
            }
            (ActivationPhase::Decode, activation_seq, Some(token_index))
                if token_index
                    .checked_add(1)
                    .is_some_and(|expected| expected == activation_seq) => {}
            (ActivationPhase::Decode, _, _) => {
                return Err(ActivationEnvelopeError::Invalid(
                    "decode activation sequence must equal accepted-token index plus one",
                ));
            }
        }

        self.provenance.validate()?;
        self.body.validate()
    }

    fn encode_canonical(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(512);
        put_map(&mut out, ENVELOPE_FIELDS);
        put_field(&mut out, 0, |out| put_uint(out, u64::from(self.version)));
        put_field(&mut out, 1, |out| put_model_oid(out, self.model_oid));
        put_field(&mut out, 2, |out| put_bytes(out, &self.plan_digest.0));
        put_field(&mut out, 3, |out| put_bytes(out, &self.request_id.0));
        put_field(&mut out, 4, |out| put_bytes(out, &self.session_id.0));
        put_field(&mut out, 5, |out| put_text(out, &self.tenant_id.0));
        put_field(&mut out, 6, |out| {
            put_delta_revision(out, self.delta_revision);
        });
        put_field(&mut out, 7, |out| put_uint(out, self.pipeline_epoch));
        put_field(&mut out, 8, |out| put_uint(out, self.microbatch_id));
        put_field(&mut out, 9, |out| put_uint(out, self.sequence_id));
        put_field(&mut out, 10, |out| put_uint(out, self.activation_seq));
        put_field(&mut out, 11, |out| {
            put_uint(out, u64::from(self.source_stage_id.0));
        });
        put_field(&mut out, 12, |out| {
            put_uint(out, u64::from(self.destination_stage_id.0));
        });
        put_field(&mut out, 13, |out| {
            put_stage_boundary(out, self.source_boundary);
        });
        put_field(&mut out, 14, |out| {
            put_uint(out, activation_phase_code(self.phase));
        });
        put_field(&mut out, 15, |out| put_uint(out, self.start_position));
        put_field(&mut out, 16, |out| match self.accepted_token_index {
            Some(index) => put_uint(out, index),
            None => put_null(out),
        });
        put_field(&mut out, 17, |out| {
            put_provenance(out, self.provenance);
        });
        put_field(&mut out, 18, |out| put_body(out, &self.body));
        out
    }

    fn decode_preflight(bytes: &[u8]) -> Result<Self, ActivationEnvelopeError> {
        let mut input = CanonicalDecoder::new(bytes);
        input.map_exact(ENVELOPE_FIELDS)?;

        input.field(0)?;
        let version = input.u16()?;
        input.field(1)?;
        let model_oid = input.model_oid()?;
        input.field(2)?;
        let plan_digest = PlanDigest(input.fixed_bytes()?);
        input.field(3)?;
        let request_id = RequestId(input.fixed_bytes()?);
        input.field(4)?;
        let session_id = SessionId(input.fixed_bytes()?);
        input.field(5)?;
        let tenant_id = TenantId(input.text(MAX_TENANT_ID_BYTES)?);
        input.field(6)?;
        let delta_revision = input.delta_revision()?;
        input.field(7)?;
        let pipeline_epoch = input.uint()?;
        input.field(8)?;
        let microbatch_id = input.uint()?;
        input.field(9)?;
        let sequence_id = input.uint()?;
        input.field(10)?;
        let activation_seq = input.uint()?;
        input.field(11)?;
        let source_stage_id = StageId(input.u32()?);
        input.field(12)?;
        let destination_stage_id = StageId(input.u32()?);
        input.field(13)?;
        let source_boundary = input.stage_boundary()?;
        input.field(14)?;
        let phase = input.activation_phase()?;
        input.field(15)?;
        let start_position = input.uint()?;
        input.field(16)?;
        let accepted_token_index = input.optional_uint()?;
        input.field(17)?;
        let provenance = input.provenance()?;
        input.field(18)?;
        let body = input.body()?;
        input.finish()?;

        Ok(Self {
            version,
            model_oid,
            plan_digest,
            request_id,
            session_id,
            tenant_id,
            delta_revision,
            pipeline_epoch,
            microbatch_id,
            sequence_id,
            activation_seq,
            source_stage_id,
            destination_stage_id,
            source_boundary,
            phase,
            start_position,
            accepted_token_index,
            provenance,
            body,
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum ActivationEnvelopeError {
    #[error("activation envelope CBOR decode failed: {0}")]
    Decode(String),
    #[error("activation envelope is invalid: {0}")]
    Invalid(&'static str),
    #[error("activation envelope exceeds the {MAX_ENCODED_ENVELOPE_BYTES}-byte metadata cap: {0}")]
    TooLarge(usize),
    #[error("activation envelope is not deterministic canonical CBOR")]
    NonCanonical,
}

fn validate_nonzero(bytes: &[u8], label: &'static str) -> Result<(), ActivationEnvelopeError> {
    if bytes.iter().all(|byte| *byte == 0) {
        return Err(ActivationEnvelopeError::Invalid(label));
    }
    Ok(())
}

fn validate_visible_text(
    text: &str,
    max_bytes: usize,
    error: &'static str,
) -> Result<(), ActivationEnvelopeError> {
    if text.is_empty()
        || text.len() > max_bytes
        || text
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
    {
        return Err(ActivationEnvelopeError::Invalid(error));
    }
    Ok(())
}

fn put_field(out: &mut Vec<u8>, key: u64, value: impl FnOnce(&mut Vec<u8>)) {
    put_uint(out, key);
    value(out);
}

fn put_head(out: &mut Vec<u8>, major: u8, argument: u64) {
    let prefix = major << 5;
    match argument {
        0..=23 => out.push(prefix | argument as u8),
        24..=0xff => {
            out.push(prefix | 24);
            out.push(argument as u8);
        }
        0x100..=0xffff => {
            out.push(prefix | 25);
            out.extend_from_slice(&(argument as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            out.push(prefix | 26);
            out.extend_from_slice(&(argument as u32).to_be_bytes());
        }
        _ => {
            out.push(prefix | 27);
            out.extend_from_slice(&argument.to_be_bytes());
        }
    }
}

fn put_uint(out: &mut Vec<u8>, value: u64) {
    put_head(out, 0, value);
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_head(out, 2, value.len() as u64);
    out.extend_from_slice(value);
}

fn put_text(out: &mut Vec<u8>, value: &str) {
    put_head(out, 3, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn put_array(out: &mut Vec<u8>, length: usize) {
    put_head(out, 4, length as u64);
}

fn put_map(out: &mut Vec<u8>, length: usize) {
    put_head(out, 5, length as u64);
}

fn put_bool(out: &mut Vec<u8>, value: bool) {
    out.push(if value { 0xf5 } else { 0xf4 });
}

fn put_null(out: &mut Vec<u8>) {
    out.push(0xf6);
}

fn put_model_oid(out: &mut Vec<u8>, oid: ModelOid) {
    put_array(out, 2);
    match oid {
        ModelOid::Sha1(bytes) => {
            put_uint(out, 0);
            put_bytes(out, &bytes);
        }
        ModelOid::Sha256(bytes) => {
            put_uint(out, 1);
            put_bytes(out, &bytes);
        }
    }
}

fn put_delta_revision(out: &mut Vec<u8>, revision: DeltaRevision) {
    match revision {
        DeltaRevision::BaseModel => {
            put_array(out, 1);
            put_uint(out, 0);
        }
        DeltaRevision::Digest(digest) => {
            put_array(out, 2);
            put_uint(out, 1);
            put_bytes(out, &digest.0);
        }
    }
}

fn stage_role_code(role: StageRole) -> u64 {
    match role {
        StageRole::First => 0,
        StageRole::Interior => 1,
        StageRole::Last => 2,
        StageRole::Only => 3,
    }
}

fn put_stage_boundary(out: &mut Vec<u8>, boundary: StageBoundary) {
    put_array(out, 3);
    put_uint(out, u64::from(boundary.start_layer));
    put_uint(out, u64::from(boundary.end_layer));
    put_uint(out, stage_role_code(boundary.role));
}

fn activation_phase_code(phase: ActivationPhase) -> u64 {
    match phase {
        ActivationPhase::Prefill => 0,
        ActivationPhase::Decode => 1,
    }
}

fn dtype_code(dtype: ActivationDtype) -> u64 {
    match dtype {
        ActivationDtype::Float16 => 0,
        ActivationDtype::Bfloat16 => 1,
        ActivationDtype::Float32 => 2,
    }
}

fn codec_code(codec: ActivationCodec) -> u64 {
    match codec {
        ActivationCodec::RawLittleEndian => 0,
    }
}

fn put_tensor(out: &mut Vec<u8>, tensor: TensorMetadata) {
    put_array(out, 6);
    put_uint(out, dtype_code(tensor.dtype));
    for dimension in tensor.shape {
        put_uint(out, dimension);
    }
    put_uint(out, tensor.byte_length);
    put_uint(out, codec_code(tensor.codec));
}

fn put_coordinate(out: &mut Vec<u8>, coordinate: ChunkCoordinate) {
    put_array(out, 4);
    put_uint(out, u64::from(coordinate.index));
    put_uint(out, u64::from(coordinate.count));
    put_uint(out, coordinate.offset);
    put_uint(out, coordinate.length);
}

fn put_provenance(out: &mut Vec<u8>, provenance: ActivationProvenance) {
    put_array(out, 2);
    put_bytes(out, &provenance.producer.0);
    put_uint(out, provenance.key_epoch);
}

fn error_code(error: ActivationErrorCode) -> u64 {
    match error {
        ActivationErrorCode::InvalidRequest => 0,
        ActivationErrorCode::StaleEpoch => 1,
        ActivationErrorCode::OutOfOrder => 2,
        ActivationErrorCode::Integrity => 3,
        ActivationErrorCode::Cancelled => 4,
        ActivationErrorCode::Internal => 5,
    }
}

fn put_body(out: &mut Vec<u8>, body: &ActivationEnvelopeBody) {
    match body {
        ActivationEnvelopeBody::Chunk(chunk) => {
            put_array(out, 5);
            put_uint(out, 0);
            put_tensor(out, chunk.tensor);
            put_coordinate(out, chunk.coordinate);
            put_bytes(out, &chunk.chunk_digest.0);
            put_bytes(out, &chunk.whole_activation_digest.0);
        }
        ActivationEnvelopeBody::Terminal(terminal) => {
            put_array(out, 3);
            put_uint(out, 1);
            put_uint(out, u64::from(terminal.chunk_count));
            put_bytes(out, &terminal.whole_activation_digest.0);
        }
        ActivationEnvelopeBody::Error(error) => {
            put_array(out, 4);
            put_uint(out, 2);
            put_uint(out, error_code(error.code));
            put_text(out, &error.message);
            put_bool(out, error.retryable);
        }
    }
}

/// Bounded semantic decoder for the exact deterministic CBOR subset above.
struct CanonicalDecoder<'a> {
    input: &'a [u8],
    offset: usize,
    items: usize,
}

impl<'a> CanonicalDecoder<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            offset: 0,
            items: 0,
        }
    }

    fn error(message: impl Into<String>) -> ActivationEnvelopeError {
        ActivationEnvelopeError::Decode(message.into())
    }

    fn finish(self) -> Result<(), ActivationEnvelopeError> {
        if self.offset != self.input.len() {
            return Err(Self::error("CBOR contains trailing data"));
        }
        Ok(())
    }

    fn read_byte(&mut self) -> Result<u8, ActivationEnvelopeError> {
        let byte = self
            .input
            .get(self.offset)
            .copied()
            .ok_or_else(|| Self::error("truncated CBOR item"))?;
        self.offset += 1;
        Ok(byte)
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N], ActivationEnvelopeError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or_else(|| Self::error("CBOR length overflow"))?;
        let bytes = self
            .input
            .get(self.offset..end)
            .ok_or_else(|| Self::error("truncated CBOR argument"))?;
        self.offset = end;
        bytes
            .try_into()
            .map_err(|_| Self::error("invalid CBOR argument width"))
    }

    fn head(&mut self) -> Result<(u8, u64), ActivationEnvelopeError> {
        self.items = self
            .items
            .checked_add(1)
            .ok_or_else(|| Self::error("CBOR item counter overflow"))?;
        if self.items > 128 {
            return Err(Self::error("CBOR item count exceeds envelope profile"));
        }

        let initial = self.read_byte()?;
        let major = initial >> 5;
        let additional = initial & 0x1f;
        let argument = match additional {
            value @ 0..=23 => u64::from(value),
            24 => {
                let value = u64::from(self.read_byte()?);
                if value < 24 {
                    return Err(Self::error("CBOR argument is not shortest-form encoded"));
                }
                value
            }
            25 => {
                let value = u64::from(u16::from_be_bytes(self.read_exact()?));
                if value <= u64::from(u8::MAX) {
                    return Err(Self::error("CBOR argument is not shortest-form encoded"));
                }
                value
            }
            26 => {
                let value = u64::from(u32::from_be_bytes(self.read_exact()?));
                if value <= u64::from(u16::MAX) {
                    return Err(Self::error("CBOR argument is not shortest-form encoded"));
                }
                value
            }
            27 => {
                let value = u64::from_be_bytes(self.read_exact()?);
                if value <= u64::from(u32::MAX) {
                    return Err(Self::error("CBOR argument is not shortest-form encoded"));
                }
                value
            }
            31 => return Err(Self::error("indefinite-length CBOR is forbidden")),
            _ => {
                return Err(Self::error("reserved CBOR additional-information value"));
            }
        };
        Ok((major, argument))
    }

    fn length(argument: u64) -> Result<usize, ActivationEnvelopeError> {
        usize::try_from(argument).map_err(|_| Self::error("CBOR length does not fit address space"))
    }

    fn container_exact(
        &mut self,
        expected_major: u8,
        expected_length: usize,
    ) -> Result<(), ActivationEnvelopeError> {
        let (major, argument) = self.head()?;
        if major != expected_major {
            return Err(Self::error(format!(
                "unexpected CBOR major type {major}, expected {expected_major}"
            )));
        }
        let length = Self::length(argument)?;
        if length != expected_length {
            return Err(Self::error(format!(
                "CBOR container has {length} entries, expected {expected_length}"
            )));
        }
        Ok(())
    }

    fn array_exact(&mut self, length: usize) -> Result<(), ActivationEnvelopeError> {
        self.container_exact(4, length)
    }

    fn map_exact(&mut self, length: usize) -> Result<(), ActivationEnvelopeError> {
        self.container_exact(5, length)
    }

    fn uint(&mut self) -> Result<u64, ActivationEnvelopeError> {
        let (major, argument) = self.head()?;
        if major != 0 {
            return Err(Self::error("expected unsigned CBOR integer"));
        }
        Ok(argument)
    }

    fn u16(&mut self) -> Result<u16, ActivationEnvelopeError> {
        u16::try_from(self.uint()?).map_err(|_| Self::error("integer does not fit u16"))
    }

    fn u32(&mut self) -> Result<u32, ActivationEnvelopeError> {
        u32::try_from(self.uint()?).map_err(|_| Self::error("integer does not fit u32"))
    }

    fn field(&mut self, expected: u64) -> Result<(), ActivationEnvelopeError> {
        let actual = self.uint()?;
        if actual != expected {
            return Err(Self::error(format!(
                "activation map field is missing, duplicate, unknown, or reordered: expected {expected}, got {actual}"
            )));
        }
        Ok(())
    }

    fn bytes(&mut self, limit: usize) -> Result<&'a [u8], ActivationEnvelopeError> {
        let (major, argument) = self.head()?;
        if major != 2 {
            return Err(Self::error("expected CBOR byte string"));
        }
        let length = Self::length(argument)?;
        if length > limit {
            return Err(Self::error(format!(
                "CBOR byte string length {length} exceeds {limit}"
            )));
        }
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| Self::error("CBOR byte string length overflow"))?;
        let bytes = self
            .input
            .get(self.offset..end)
            .ok_or_else(|| Self::error("truncated CBOR byte string"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn fixed_bytes<const N: usize>(&mut self) -> Result<[u8; N], ActivationEnvelopeError> {
        let bytes = self.bytes(N)?;
        if bytes.len() != N {
            return Err(Self::error(format!(
                "CBOR byte string has {} bytes, expected {N}",
                bytes.len()
            )));
        }
        bytes
            .try_into()
            .map_err(|_| Self::error("invalid fixed byte string width"))
    }

    fn text(&mut self, limit: usize) -> Result<String, ActivationEnvelopeError> {
        let (major, argument) = self.head()?;
        if major != 3 {
            return Err(Self::error("expected CBOR text string"));
        }
        let length = Self::length(argument)?;
        if length > limit {
            return Err(Self::error(format!(
                "CBOR text length {length} exceeds {limit}"
            )));
        }
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| Self::error("CBOR text length overflow"))?;
        let bytes = self
            .input
            .get(self.offset..end)
            .ok_or_else(|| Self::error("truncated CBOR text"))?;
        self.offset = end;
        let text =
            std::str::from_utf8(bytes).map_err(|_| Self::error("CBOR text is not valid UTF-8"))?;
        Ok(text.to_owned())
    }

    fn optional_uint(&mut self) -> Result<Option<u64>, ActivationEnvelopeError> {
        match self.input.get(self.offset).copied() {
            Some(0xf6) => {
                self.offset += 1;
                self.items += 1;
                Ok(None)
            }
            Some(_) => self.uint().map(Some),
            None => Err(Self::error("truncated optional integer")),
        }
    }

    fn boolean(&mut self) -> Result<bool, ActivationEnvelopeError> {
        let byte = self.read_byte()?;
        self.items += 1;
        match byte {
            0xf4 => Ok(false),
            0xf5 => Ok(true),
            _ => Err(Self::error("expected canonical CBOR boolean")),
        }
    }

    fn model_oid(&mut self) -> Result<ModelOid, ActivationEnvelopeError> {
        self.array_exact(2)?;
        match self.uint()? {
            0 => Ok(ModelOid::Sha1(self.fixed_bytes()?)),
            1 => Ok(ModelOid::Sha256(self.fixed_bytes()?)),
            value => Err(Self::error(format!("unknown model OID algorithm {value}"))),
        }
    }

    fn delta_revision(&mut self) -> Result<DeltaRevision, ActivationEnvelopeError> {
        let start = self.offset;
        let (major, length) = self.head()?;
        if major != 4 {
            return Err(Self::error("delta revision must be a CBOR array"));
        }
        match (length, self.uint()?) {
            (1, 0) => Ok(DeltaRevision::BaseModel),
            (2, 1) => Ok(DeltaRevision::Digest(DeltaDigest(self.fixed_bytes()?))),
            _ => {
                self.offset = start;
                Err(Self::error("unknown or malformed delta revision"))
            }
        }
    }

    fn stage_boundary(&mut self) -> Result<StageBoundary, ActivationEnvelopeError> {
        self.array_exact(3)?;
        let start_layer = self.u32()?;
        let end_layer = self.u32()?;
        let role = match self.uint()? {
            0 => StageRole::First,
            1 => StageRole::Interior,
            2 => StageRole::Last,
            3 => StageRole::Only,
            value => return Err(Self::error(format!("unknown stage role {value}"))),
        };
        Ok(StageBoundary {
            start_layer,
            end_layer,
            role,
        })
    }

    fn activation_phase(&mut self) -> Result<ActivationPhase, ActivationEnvelopeError> {
        match self.uint()? {
            0 => Ok(ActivationPhase::Prefill),
            1 => Ok(ActivationPhase::Decode),
            value => Err(Self::error(format!("unknown activation phase {value}"))),
        }
    }

    fn provenance(&mut self) -> Result<ActivationProvenance, ActivationEnvelopeError> {
        self.array_exact(2)?;
        Ok(ActivationProvenance {
            producer: ProducerIdentity(self.fixed_bytes()?),
            key_epoch: self.uint()?,
        })
    }

    fn tensor(&mut self) -> Result<TensorMetadata, ActivationEnvelopeError> {
        self.array_exact(6)?;
        let dtype = match self.uint()? {
            0 => ActivationDtype::Float16,
            1 => ActivationDtype::Bfloat16,
            2 => ActivationDtype::Float32,
            value => return Err(Self::error(format!("unknown activation dtype {value}"))),
        };
        let shape = [self.uint()?, self.uint()?, self.uint()?];
        let byte_length = self.uint()?;
        let codec = match self.uint()? {
            0 => ActivationCodec::RawLittleEndian,
            value => return Err(Self::error(format!("unknown activation codec {value}"))),
        };
        Ok(TensorMetadata {
            dtype,
            shape,
            byte_length,
            codec,
        })
    }

    fn coordinate(&mut self) -> Result<ChunkCoordinate, ActivationEnvelopeError> {
        self.array_exact(4)?;
        Ok(ChunkCoordinate {
            index: self.u32()?,
            count: self.u32()?,
            offset: self.uint()?,
            length: self.uint()?,
        })
    }

    fn body(&mut self) -> Result<ActivationEnvelopeBody, ActivationEnvelopeError> {
        let start = self.offset;
        let (major, length) = self.head()?;
        if major != 4 {
            return Err(Self::error("activation body must be a CBOR array"));
        }
        let kind = self.uint()?;
        match (kind, length) {
            (0, 5) => Ok(ActivationEnvelopeBody::Chunk(ActivationChunk {
                tensor: self.tensor()?,
                coordinate: self.coordinate()?,
                chunk_digest: ChunkDigest(self.fixed_bytes()?),
                whole_activation_digest: WholeActivationDigest(self.fixed_bytes()?),
            })),
            (1, 3) => Ok(ActivationEnvelopeBody::Terminal(ActivationTerminal {
                chunk_count: self.u32()?,
                whole_activation_digest: WholeActivationDigest(self.fixed_bytes()?),
            })),
            (2, 4) => {
                let code = match self.uint()? {
                    0 => ActivationErrorCode::InvalidRequest,
                    1 => ActivationErrorCode::StaleEpoch,
                    2 => ActivationErrorCode::OutOfOrder,
                    3 => ActivationErrorCode::Integrity,
                    4 => ActivationErrorCode::Cancelled,
                    5 => ActivationErrorCode::Internal,
                    value => {
                        return Err(Self::error(format!(
                            "unknown activation error code {value}"
                        )));
                    }
                };
                Ok(ActivationEnvelopeBody::Error(ActivationError {
                    code,
                    message: self.text(MAX_ERROR_MESSAGE_BYTES)?,
                    retryable: self.boolean()?,
                }))
            }
            _ => {
                self.offset = start;
                Err(Self::error("unknown or malformed activation body"))
            }
        }
    }
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
            tenant_id: TenantId("did:web:tenant.example".to_owned()),
            delta_revision: DeltaRevision::Digest(DeltaDigest([5; 32])),
            pipeline_epoch: 7,
            microbatch_id: 11,
            sequence_id: 13,
            activation_seq: 21,
            source_stage_id: StageId(1),
            destination_stage_id: StageId(2),
            source_boundary: StageBoundary {
                start_layer: 8,
                end_layer: 16,
                role: StageRole::Interior,
            },
            phase: ActivationPhase::Decode,
            start_position: 21,
            accepted_token_index: Some(20),
            provenance: ActivationProvenance {
                producer: ProducerIdentity([6; 32]),
                key_epoch: 9,
            },
            body: ActivationEnvelopeBody::Chunk(ActivationChunk {
                tensor: TensorMetadata {
                    dtype: ActivationDtype::Bfloat16,
                    shape: [2, 4, 8],
                    byte_length: 128,
                    codec: ActivationCodec::RawLittleEndian,
                },
                coordinate: ChunkCoordinate {
                    index: 1,
                    count: 3,
                    offset: 32,
                    length: 32,
                },
                chunk_digest: ChunkDigest([7; 32]),
                whole_activation_digest: WholeActivationDigest([8; 32]),
            }),
        }
    }

    fn prefill_envelope() -> ActivationEnvelope {
        let mut envelope = chunk_envelope();
        envelope.delta_revision = DeltaRevision::BaseModel;
        envelope.activation_seq = 0;
        envelope.source_stage_id = StageId(0);
        envelope.destination_stage_id = StageId(1);
        envelope.source_boundary = StageBoundary {
            start_layer: 0,
            end_layer: 8,
            role: StageRole::First,
        };
        envelope.phase = ActivationPhase::Prefill;
        envelope.start_position = 0;
        envelope.accepted_token_index = None;
        envelope
    }

    fn chunk_mut(envelope: &mut ActivationEnvelope) -> &mut ActivationChunk {
        let ActivationEnvelopeBody::Chunk(chunk) = &mut envelope.body else {
            panic!("fixture is a chunk");
        };
        chunk
    }

    fn unchecked_canonical(envelope: &ActivationEnvelope) -> Vec<u8> {
        envelope.encode_canonical()
    }

    fn unique_subslice(haystack: &[u8], needle: &[u8]) -> usize {
        let matches: Vec<_> = haystack
            .windows(needle.len())
            .enumerate()
            .filter_map(|(index, window)| (window == needle).then_some(index))
            .collect();
        assert_eq!(matches.len(), 1, "wire marker must be unique");
        matches[0]
    }

    fn assert_sender_and_receiver_invalid(envelope: &ActivationEnvelope, expected: &'static str) {
        assert_eq!(
            envelope.to_cbor().unwrap_err(),
            ActivationEnvelopeError::Invalid(expected),
            "sender guard did not fail causally"
        );
        assert_eq!(
            ActivationEnvelope::from_cbor(&unchecked_canonical(envelope)).unwrap_err(),
            ActivationEnvelopeError::Invalid(expected),
            "receiver guard did not fail causally"
        );
    }

    #[test]
    fn chunk_round_trip_is_deterministic_and_preserves_every_contract_field() {
        let envelope = chunk_envelope();
        let first = envelope.to_cbor().unwrap();
        let second = envelope.to_cbor().unwrap();
        assert_eq!(first, second);
        assert_eq!(ActivationEnvelope::from_cbor(&first).unwrap(), envelope);
        assert_eq!(first[0], 0xb3, "top-level wire item must be a 19-field map");
    }

    #[test]
    fn terminal_and_error_variants_round_trip() {
        let mut terminal = prefill_envelope();
        terminal.model_oid = ModelOid::Sha256([10; 32]);
        terminal.body = ActivationEnvelopeBody::Terminal(ActivationTerminal {
            chunk_count: 3,
            whole_activation_digest: WholeActivationDigest([9; 32]),
        });

        let mut error = terminal.clone();
        error.body = ActivationEnvelopeBody::Error(ActivationError {
            code: ActivationErrorCode::Integrity,
            message: "whole-activation-digest-mismatch".to_owned(),
            retryable: false,
        });

        for envelope in [terminal, error] {
            let encoded = envelope.to_cbor().unwrap();
            assert_eq!(ActivationEnvelope::from_cbor(&encoded).unwrap(), envelope);
        }
    }

    #[test]
    fn sender_and_receiver_reject_identity_guard_inversions() {
        let mut envelope = chunk_envelope();
        envelope.version = 2;
        assert_sender_and_receiver_invalid(&envelope, "unsupported activation envelope version");

        envelope = chunk_envelope();
        envelope.model_oid = ModelOid::Sha1([0; 20]);
        assert_sender_and_receiver_invalid(&envelope, "model OID");

        envelope = chunk_envelope();
        envelope.plan_digest = PlanDigest([0; 32]);
        assert_sender_and_receiver_invalid(&envelope, "plan digest");

        envelope = chunk_envelope();
        envelope.request_id = RequestId([0; 16]);
        assert_sender_and_receiver_invalid(&envelope, "request identity");

        envelope = chunk_envelope();
        envelope.session_id = SessionId([0; 16]);
        assert_sender_and_receiver_invalid(&envelope, "session identity");

        envelope = chunk_envelope();
        envelope.tenant_id = TenantId(String::new());
        assert_sender_and_receiver_invalid(
            &envelope,
            "tenant identity must contain 1..=512 non-whitespace bytes",
        );

        envelope = chunk_envelope();
        envelope.delta_revision = DeltaRevision::Digest(DeltaDigest([0; 32]));
        assert_sender_and_receiver_invalid(&envelope, "delta revision digest");

        envelope = chunk_envelope();
        envelope.provenance.producer = ProducerIdentity([0; 32]);
        assert_sender_and_receiver_invalid(&envelope, "producer identity");
    }

    #[test]
    fn sender_and_receiver_enforce_phase_activation_sequence_causality() {
        let mut envelope = prefill_envelope();
        envelope.activation_seq = 1;
        assert_sender_and_receiver_invalid(
            &envelope,
            "prefill requires activation sequence zero and no accepted-token index",
        );

        envelope = prefill_envelope();
        envelope.accepted_token_index = Some(0);
        assert_sender_and_receiver_invalid(
            &envelope,
            "prefill requires activation sequence zero and no accepted-token index",
        );

        envelope = chunk_envelope();
        envelope.activation_seq = 20;
        assert_sender_and_receiver_invalid(
            &envelope,
            "decode activation sequence must equal accepted-token index plus one",
        );

        envelope = chunk_envelope();
        envelope.accepted_token_index = None;
        assert_sender_and_receiver_invalid(
            &envelope,
            "decode activation sequence must equal accepted-token index plus one",
        );

        envelope = chunk_envelope();
        envelope.accepted_token_index = Some(u64::MAX);
        assert_sender_and_receiver_invalid(
            &envelope,
            "decode activation sequence must equal accepted-token index plus one",
        );
    }

    #[test]
    fn sender_and_receiver_enforce_stage_role_and_direction() {
        let mut envelope = prefill_envelope();
        envelope.source_stage_id = StageId(1);
        assert_sender_and_receiver_invalid(
            &envelope,
            "first stage must have identity and start layer zero",
        );

        envelope = chunk_envelope();
        envelope.source_stage_id = StageId(0);
        envelope.destination_stage_id = StageId(1);
        assert_sender_and_receiver_invalid(
            &envelope,
            "interior stage must have non-zero identity and start layer",
        );

        envelope = chunk_envelope();
        envelope.destination_stage_id = StageId(3);
        assert_sender_and_receiver_invalid(
            &envelope,
            "destination stage must immediately follow source stage",
        );

        envelope = chunk_envelope();
        envelope.source_stage_id = StageId(u32::MAX);
        assert_sender_and_receiver_invalid(
            &envelope,
            "source stage identity cannot advance without overflow",
        );

        for role in [StageRole::Last, StageRole::Only] {
            envelope = chunk_envelope();
            envelope.source_boundary.role = role;
            assert_sender_and_receiver_invalid(
                &envelope,
                "last/only stage does not emit a next-stage activation",
            );
        }

        envelope = chunk_envelope();
        envelope.source_boundary.end_layer = envelope.source_boundary.start_layer;
        assert_sender_and_receiver_invalid(
            &envelope,
            "stage boundary must be a non-empty half-open interval",
        );
    }

    #[test]
    fn sender_and_receiver_enforce_dense_activation_abi() {
        let mut envelope = chunk_envelope();
        chunk_mut(&mut envelope).tensor.shape = [0, 4, 8];
        assert_sender_and_receiver_invalid(
            &envelope,
            "activation shape [B,Q,H] requires three non-zero dimensions",
        );

        envelope = chunk_envelope();
        chunk_mut(&mut envelope).tensor.byte_length = 127;
        assert_sender_and_receiver_invalid(
            &envelope,
            "activation byte length does not match dense [B,Q,H] dtype",
        );

        envelope = chunk_envelope();
        chunk_mut(&mut envelope).tensor.shape = [u64::MAX, 2, 2];
        assert_sender_and_receiver_invalid(
            &envelope,
            "activation byte length does not match dense [B,Q,H] dtype",
        );
    }

    #[test]
    fn sender_and_receiver_enforce_chunk_span_and_integrity() {
        let mut envelope = chunk_envelope();
        chunk_mut(&mut envelope).coordinate.count = 0;
        assert_sender_and_receiver_invalid(
            &envelope,
            "chunk coordinate requires count > 0 and index < count",
        );

        envelope = chunk_envelope();
        chunk_mut(&mut envelope).coordinate.length = 0;
        assert_sender_and_receiver_invalid(&envelope, "chunk length must be non-zero");

        envelope = chunk_envelope();
        chunk_mut(&mut envelope).coordinate = ChunkCoordinate {
            index: 0,
            count: 3,
            offset: 1,
            length: 31,
        };
        assert_sender_and_receiver_invalid(
            &envelope,
            "first chunk must begin at logical byte zero",
        );

        envelope = chunk_envelope();
        chunk_mut(&mut envelope).coordinate = ChunkCoordinate {
            index: 2,
            count: 3,
            offset: 64,
            length: 63,
        };
        assert_sender_and_receiver_invalid(
            &envelope,
            "last chunk must end at the logical activation length",
        );

        envelope = chunk_envelope();
        chunk_mut(&mut envelope).coordinate = ChunkCoordinate {
            index: 2,
            count: 3,
            offset: u64::MAX,
            length: 2,
        };
        assert_sender_and_receiver_invalid(&envelope, "chunk byte span overflows");

        envelope = chunk_envelope();
        chunk_mut(&mut envelope).coordinate = ChunkCoordinate {
            index: 1,
            count: 3,
            offset: 120,
            length: 9,
        };
        assert_sender_and_receiver_invalid(
            &envelope,
            "chunk byte span exceeds logical activation length",
        );

        envelope = chunk_envelope();
        chunk_mut(&mut envelope).chunk_digest = ChunkDigest([0; 32]);
        assert_sender_and_receiver_invalid(&envelope, "chunk digest");

        envelope = chunk_envelope();
        chunk_mut(&mut envelope).whole_activation_digest = WholeActivationDigest([0; 32]);
        assert_sender_and_receiver_invalid(&envelope, "whole activation digest");
    }

    #[test]
    fn sender_and_receiver_enforce_terminal_and_error_guards() {
        let mut envelope = prefill_envelope();
        envelope.body = ActivationEnvelopeBody::Terminal(ActivationTerminal {
            chunk_count: 0,
            whole_activation_digest: WholeActivationDigest([9; 32]),
        });
        assert_sender_and_receiver_invalid(&envelope, "terminal chunk count must be non-zero");

        envelope.body = ActivationEnvelopeBody::Terminal(ActivationTerminal {
            chunk_count: 1,
            whole_activation_digest: WholeActivationDigest([0; 32]),
        });
        assert_sender_and_receiver_invalid(&envelope, "whole activation digest");

        envelope.body = ActivationEnvelopeBody::Error(ActivationError {
            code: ActivationErrorCode::Internal,
            message: String::new(),
            retryable: false,
        });
        assert_sender_and_receiver_invalid(
            &envelope,
            "error message must contain 1..=1024 non-whitespace bytes",
        );

        let mut oversized = "x".repeat(MAX_ERROR_MESSAGE_BYTES);
        oversized.push('x');
        envelope.body = ActivationEnvelopeBody::Error(ActivationError {
            code: ActivationErrorCode::Internal,
            message: oversized,
            retryable: false,
        });
        assert_eq!(
            envelope.to_cbor().unwrap_err(),
            ActivationEnvelopeError::Invalid(
                "error message must contain 1..=1024 non-whitespace bytes"
            )
        );
        assert!(matches!(
            ActivationEnvelope::from_cbor(&unchecked_canonical(&envelope)),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("text length 1025 exceeds 1024")
        ));
    }

    #[test]
    fn receiver_rejects_noncanonical_duplicate_reordered_missing_and_unknown_fields() {
        let canonical = chunk_envelope().to_cbor().unwrap();
        assert_eq!(canonical[0], 0xb3);
        assert_eq!(canonical[1], 0);
        assert_eq!(canonical[2], 1);
        assert_eq!(canonical[3], 1);

        let mut non_shortest = canonical.clone();
        non_shortest.splice(1..2, [0x18, 0x00]);
        assert!(matches!(
            ActivationEnvelope::from_cbor(&non_shortest),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("shortest-form")
        ));

        let mut duplicate = canonical.clone();
        duplicate[3] = 0;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&duplicate),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("duplicate")
        ));

        let mut reordered = canonical.clone();
        reordered[1] = 1;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&reordered),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("reordered")
        ));

        let mut missing = canonical.clone();
        missing[0] = 0xb2;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&missing),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("expected 19")
        ));

        let mut unknown = canonical.clone();
        unknown[1] = 23;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&unknown),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("unknown")
        ));

        let mut indefinite = canonical.clone();
        indefinite[0] = 0xbf;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&indefinite),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("indefinite")
        ));
    }

    #[test]
    fn receiver_rejects_unknown_admitted_wire_values_and_trailing_data() {
        let canonical = chunk_envelope().to_cbor().unwrap();

        let mut unknown_model_algorithm = canonical.clone();
        // Top map, key 0, version 1, key 1, model array(2), algorithm.
        assert_eq!(&unknown_model_algorithm[..6], &[0xb3, 0, 1, 1, 0x82, 0]);
        unknown_model_algorithm[5] = 2;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&unknown_model_algorithm),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("unknown model OID algorithm")
        ));

        let tensor_marker = [0x86, 1, 2, 4, 8, 0x18, 0x80, 0];
        let tensor_start = unique_subslice(&canonical, &tensor_marker);

        let mut wrong_rank = canonical.clone();
        wrong_rank[tensor_start] = 0x85;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&wrong_rank),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("expected 6")
        ));

        let mut unknown_dtype = canonical.clone();
        unknown_dtype[tensor_start + 1] = 3;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&unknown_dtype),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("unknown activation dtype")
        ));

        let mut unknown_codec = canonical.clone();
        unknown_codec[tensor_start + tensor_marker.len() - 1] = 1;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&unknown_codec),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("unknown activation codec")
        ));

        let body_marker = [0x85, 0, 0x86];
        let body_start = unique_subslice(&canonical, &body_marker);
        let mut unknown_body = canonical.clone();
        unknown_body[body_start + 1] = 3;
        assert!(matches!(
            ActivationEnvelope::from_cbor(&unknown_body),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("unknown or malformed activation body")
        ));

        let mut trailing = canonical;
        trailing.push(0);
        assert!(matches!(
            ActivationEnvelope::from_cbor(&trailing),
            Err(ActivationEnvelopeError::Decode(message))
                if message.contains("trailing data")
        ));

        let oversized = vec![0; MAX_ENCODED_ENVELOPE_BYTES + 1];
        assert_eq!(
            ActivationEnvelope::from_cbor(&oversized).unwrap_err(),
            ActivationEnvelopeError::TooLarge(MAX_ENCODED_ENVELOPE_BYTES + 1)
        );
    }
}
