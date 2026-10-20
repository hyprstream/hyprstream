//! Arrow-native decision-batch storage (System One P0.3).
//!
//! Decision model outputs are stored as opaque Arrow IPC stream payloads keyed
//! by the producing decision-schema fingerprint, so eval/training reads back
//! the exact `RecordBatch` the model emitted — no row-wise SQL materialization
//! and no lossy re-typing. Each row also carries the `(schema, model, calib)`
//! version triple for joins against `calibration_params`.

// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

use crate::storage::StorageBackend;
use arrow::array::{
    Array, ArrayRef, BinaryArray, Int64Array, LargeBinaryArray, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use std::sync::Arc;
use std::time::SystemTime;
use tonic::Status;

/// Decision batch table name.
pub const DECISION_BATCH_TABLE: &str = "decision_batches";

/// Payload format tag for Arrow IPC stream bytes.
pub const PAYLOAD_FORMAT_IPC_STREAM: &str = "arrow-ipc-stream";

/// Maximum encoded IPC payload accepted for bounded decode (bytes).
pub const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

/// Mint a new batch id.
pub fn mint_batch_id() -> String {
    format!("batch-{}", uuid::Uuid::new_v4().simple())
}

fn now_unix_ms() -> i64 {
    let ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    i64::try_from(ms).unwrap_or(i64::MAX)
}

/// Row metadata for one stored decision batch (everything but the payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionBatchMeta {
    /// Unique batch id (see [`mint_batch_id`]).
    pub batch_id: String,
    /// Decision-schema fingerprint of the payload schema (P0.1a IR identity).
    pub schema_fingerprint: String,
    /// Model id/version that produced the batch.
    pub model_id: String,
    /// Calibration version applied at emission; `None` for uncalibrated/raw output.
    pub calib_version: Option<String>,
    /// Creation time, Unix epoch milliseconds.
    pub created_at: i64,
    /// Number of rows in the payload batch.
    pub row_count: i64,
}

/// Arrow schema of the `decision_batches` table.
pub fn decision_batch_schema() -> Schema {
    Schema::new(vec![
        Field::new("batch_id", DataType::Utf8, false),
        Field::new("schema_fingerprint", DataType::Utf8, false),
        Field::new("model_id", DataType::Utf8, false),
        Field::new("calib_version", DataType::Utf8, true),
        Field::new("created_at", DataType::Int64, false),
        Field::new("row_count", DataType::Int64, false),
        Field::new("payload_format", DataType::Utf8, false),
        Field::new("payload", DataType::Binary, false),
    ])
}

/// Serialize a batch as a self-describing Arrow IPC stream.
pub fn encode_batch_ipc(batch: &RecordBatch) -> Result<Vec<u8>, Status> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())
            .map_err(|e| Status::internal(format!("IPC writer init: {e}")))?;
        writer
            .write(batch)
            .map_err(|e| Status::internal(format!("IPC write: {e}")))?;
        writer
            .finish()
            .map_err(|e| Status::internal(format!("IPC finish: {e}")))?;
    }
    Ok(buf)
}

/// Decoder-native ZSTD window limit for compressed decision payloads
/// (log_distance): the real decoder may use at most 2^24 = 16 MiB of
/// back-reference window, exactly MAX_PAYLOAD_BYTES.
const ZSTD_WINDOW_LOG_MAX_LOG: u32 = 24;

/// Maximum accepted IPC metadata (flatbuffer) message size. Schema and batch
/// metadata are small by construction; anything larger is rejected before the
/// verified flatbuffer parse.
const MAX_IPC_METADATA_BYTES: usize = MAX_PAYLOAD_BYTES;

use crate::metrics::decision_outcome::MAX_DECISION_ROWS_PER_BATCH;
/// Arrow IPC stream framing as emitted by `StreamWriter` (post-0.15):
/// messages start at byte 0 as
/// `[0xFFFFFFFF u32][metadata_size u32 LE][metadata flatbuffer][body..]`,
/// terminated by an end-of-stream marker (`metadata_size == 0`, with or
/// without the continuation prefix). The `ARROW1` magic belongs to the FILE
/// format and is not written by the stream writer.
///
/// Message layout (normative Message.fbs, mirrored by compression.rs): the
/// body of a compressed record/dictionary batch is a sequence of buffers; each
/// buffer's data is `[8-byte LE i64 uncompressed length][codec stream]`, with
/// `-1` marking "not compressed" and `0` marking empty. The metadata
/// `Buffer.length` for such buffers is the (compressed) body-slice length, so
/// the DECOMPRESSED size is only knowable from the prefix — and the codec
/// decoders allocate `Vec::with_capacity(advertised)` followed by an
/// UNBOUNDED `read_to_end`. A malicious payload can therefore claim a small
/// advertised length while the stream expands far beyond it. The bounded
/// verification below measures the ACTUAL expansion with a capped streaming
/// read and requires exact agreement with the advertised length BEFORE the
/// real Arrow decoder is invoked.
use arrow::ipc::gen::Message::CompressionType;
use arrow::ipc::{root_as_message, MessageHeader};

const LENGTH_OF_PREFIX_DATA: i64 = 8;
const LENGTH_NO_COMPRESSED_DATA: i64 = -1;

fn invalid_ipc(msg: impl Into<String>) -> Status {
    Status::invalid_argument(format!("decision payload IPC: {}", msg.into()))
}

/// Checked little-endian u32 read used by the stream framing walk.
fn read_u32_le(b: &[u8], pos: usize) -> Result<u32, Status> {
    let s = b
        .get(pos..pos + 4)
        .ok_or_else(|| invalid_ipc("truncated IPC framing (u32)"))?;
    let arr: [u8; 4] = s
        .try_into()
        .map_err(|_| invalid_ipc("truncated IPC framing (u32)"))?;
    Ok(u32::from_le_bytes(arr))
}

/// Per-message framing bounds (byte ranges into the scanned payload) and the
/// per-buffer verification result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcMessageBounds {
    pub header_type: MessageHeader,
    /// (start, end) of the verified metadata flatbuffer.
    pub metadata: (usize, usize),
    /// (start, end) of the message body.
    pub body: (usize, usize),
    /// Number of body buffers whose actual decompressed size was verified.
    pub verified_buffers: usize,
    /// Sum of verified decompressed sizes for this message.
    pub verified_bytes: u64,
}

/// Streaming bounded verification of ONE compressed buffer:
/// `[i64 advertised][codec stream]`. Codec-internal memory is bounded by
/// DECODER-NATIVE constraints (see the match arms); at most `advertised + 1`
/// output bytes are pulled in bounded chunks, rejecting any expansion beyond
/// `advertised` and requiring the ACTUAL decompressed size to equal the
/// advertised length exactly. Returns the verified size (which the caller
/// aggregates against `max_bytes`).
fn verify_bounded_decompressed(
    codec: CompressionType,
    input: &[u8],
    max_bytes: usize,
    total: &mut u64,
) -> Result<usize, Status> {
    if input.is_empty() {
        // Arrow56 reader parity: empty metadata buffers carry no prefix and
        // decode as empty regardless of codec.
        return Ok(0);
    }
    if input.len() < LENGTH_OF_PREFIX_DATA as usize {
        return Err(invalid_ipc(
            "compressed buffer shorter than its length prefix",
        ));
    }
    let advertised = i64::from_le_bytes(
        input[..LENGTH_OF_PREFIX_DATA as usize]
            .try_into()
            .map_err(|_| invalid_ipc("unreadable compressed length prefix"))?,
    );
    if advertised == 0 {
        // Arrow56 reader parity: an advertised 0 decodes as an EMPTY buffer;
        // any trailing bytes are ignored downstream, so this is not an
        // expansion vector and nothing is read.
        return Ok(0);
    }
    if advertised == LENGTH_NO_COMPRESSED_DATA {
        // Data that follows is not compressed: a plain slice of the already
        // bounded payload — no decoder, no expansion.
        let raw = input.len() - LENGTH_OF_PREFIX_DATA as usize;
        *total = total
            .checked_add(raw as u64)
            .ok_or_else(|| invalid_ipc("IPC buffer lengths overflow the payload bound"))?;
        if *total > max_bytes as u64 {
            return Err(invalid_ipc(format!(
                "IPC stream declares more than {max_bytes} uncompressed buffer bytes in total"
            )));
        }
        return Ok(raw);
    }
    let advertised = usize::try_from(advertised)
        .map_err(|_| invalid_ipc("negative IPC compressed length prefix"))?;
    let advertised_u64 = advertised as u64;
    if advertised > max_bytes {
        return Err(invalid_ipc(format!(
            "compressed buffer advertises {advertised} uncompressed bytes, above the {max_bytes}-byte decision payload bound"
        )));
    }
    let data = &input[LENGTH_OF_PREFIX_DATA as usize..];
    let mut actual: u64 = 0;
    let mut scratch = [0u8; 64 * 1024];
    use std::io::Read;
    // The codec decoder is constructed ONCE (frame state must persist across
    // chunked reads) after its frame header passed the internal bounds check;
    // the output side is hard-capped at advertised + 1 probe bytes in 64 KiB
    // chunks — actual expansion beyond the advertised length is rejected
    // before the Arrow decoder ever runs.
    let decoder: Box<dyn std::io::Read> = match codec {
        CompressionType::LZ4_FRAME => {
            // lz4_flex 0.11.6 (frame/decompress.rs 144-160): the FrameDecoder
            // allocates from the frame descriptor's Block_Maximum_Size enum —
            // a fixed 4 MiB block buffer, plus a destination of up to
            // 2 x 4 MiB + a 64 KiB window for linked frames: a FIXED internal
            // capacity of ~12 MiB + 64 KiB that does not scale with attacker
            // bytes. The codec itself validates the frame header and
            // dictionary enums; no bespoke header parsing. The output is
            // additionally hard-capped below at advertised + 1.
            Box::new(lz4_flex::frame::FrameDecoder::new(data))
        }
        CompressionType::ZSTD => {
            // The SAME decoder type/constructor Arrow56 uses
            // (compression.rs decompress_zstd: zstd::Decoder::with_buffer),
            // on the same immutable bytes, with the DECODER-NATIVE window
            // limit configured BEFORE the first read (zstd 0.13.3
            // stream/mod.rs decoder_parameters: actual max back-reference
            // distance = 2^log_distance). Default frame handling consumes
            // ALL concatenated frames to EOF — identical accepted-bytes and
            // EOF semantics to Arrow's read_to_end. A hostile frame
            // declaring a window above 2^24 (16 MiB == MAX_PAYLOAD_BYTES)
            // fails at frame-header time inside the real decoder.
            let mut zstd_decoder = zstd::stream::read::Decoder::with_buffer(data)
                .map_err(|e| invalid_ipc(format!("zstd IPC stream: {e}")))?;
            zstd_decoder
                .window_log_max(ZSTD_WINDOW_LOG_MAX_LOG)
                .map_err(|e| invalid_ipc(format!("zstd window limit: {e}")))?;
            Box::new(zstd_decoder)
        }
        other => {
            return Err(invalid_ipc(format!(
                "unsupported IPC compression codec {other:?}"
            )));
        }
    };
    let mut capped = decoder.take(
        advertised_u64
            .checked_add(1)
            .ok_or_else(|| invalid_ipc("IPC expansion bound overflow"))?,
    );
    loop {
        let want = cap_page(advertised_u64, actual, scratch.len());
        if want == 0 {
            break;
        }
        let n = capped
            .read(&mut scratch[..want])
            .map_err(|e| invalid_ipc(format!("IPC codec stream: {e}")))?;
        if n == 0 {
            break;
        }
        actual = actual.saturating_add(n as u64);
        if actual > advertised_u64 {
            return Err(invalid_ipc(format!(
                "compressed IPC buffer expands beyond its advertised length {advertised_u64}"
            )));
        }
    }
    if actual != advertised_u64 {
        return Err(invalid_ipc(format!(
            "compressed IPC buffer expanded to {actual} bytes, not the advertised {advertised_u64}"
        )));
    }
    *total = total
        .checked_add(actual)
        .ok_or_else(|| invalid_ipc("IPC buffer lengths overflow the payload bound"))?;
    if *total > max_bytes as u64 {
        return Err(invalid_ipc(format!(
            "IPC stream declares more than {max_bytes} uncompressed buffer bytes in total"
        )));
    }
    usize::try_from(actual).map_err(|_| invalid_ipc("IPC decoded buffer length overflow"))
}

/// Remaining probe budget for this buffer, clamped to the scratch chunk.
fn cap_page(advertised: u64, actual: u64, chunk: usize) -> usize {
    advertised
        .checked_add(1)
        .and_then(|cap| cap.checked_sub(actual))
        .and_then(|rest| usize::try_from(rest).ok())
        .map(|rest| rest.min(chunk))
        .unwrap_or(0)
}

/// Verify ONE record-batch table (flatbuffer-verified via
/// `root_as_message`/`header_as_record_batch`): every buffer's body slice must
/// sit inside the message body, and every compressed buffer must pass the
/// bounded actual-expansion verification. Aggregates into `total`.
/// Verify every buffer of one record-batch table (flatbuffer-verified type).
/// Real buffer bytes — compressed-verified or plain slices — aggregate into
/// `total` across the ENTIRE stream against `max_bytes`. Record ROW COUNTS
/// are not checked here: the walk aggregates them across ALL record messages
/// against the whole-stream cap before any decode.
fn verify_record_batch_buffers(
    rb: &arrow::ipc::gen::Message::RecordBatch<'_>,
    body: &[u8],
    max_bytes: usize,
    total: &mut u64,
    verified: &mut usize,
) -> Result<Vec<usize>, Status> {
    // Signed validity guard for BOTH record and dictionary data: a negative
    // declared length is rejected before the decoder. (Dictionary lengths are
    // NOT counted against the 4096 record-row cap — that aggregate lives in
    // the caller and covers record batches only.)
    if rb.length() < 0 {
        return Err(invalid_ipc("negative IPC record batch length"));
    }
    let buffers = rb
        .buffers()
        .ok_or_else(|| invalid_ipc("record batch without buffers"))?;
    // MetadataBudget::record separately charged this temporary usize vector
    // BEFORE any allocation, in addition to the Arrow Buffer descriptor stages.
    let mut lengths = Vec::with_capacity(buffers.len());
    for i in 0..buffers.len() {
        let buffer = buffers.get(i);
        let (offset, length) = (buffer.offset(), buffer.length());
        if length < 0 || offset < 0 {
            return Err(invalid_ipc("negative IPC buffer offset/length"));
        }
        let (offset, length) = (offset as u64, length as u64);
        let end = offset
            .checked_add(length)
            .ok_or_else(|| invalid_ipc("IPC buffer range overflow"))?;
        if end > body.len() as u64 {
            return Err(invalid_ipc("IPC buffer range outside the message body"));
        }
        let slice = &body[offset as usize..end as usize];
        // Arrow56 reader parity: a length-0 metadata buffer is an EMPTY
        // buffer with no 8-byte prefix — read_buffer early-returns the empty
        // slice regardless of codec, so verification must too.
        if slice.is_empty() {
            lengths.push(0);
            continue;
        }
        match rb.compression() {
            Some(compression) => {
                let actual =
                    verify_bounded_decompressed(compression.codec(), slice, max_bytes, total)?;
                lengths.push(actual);
                *verified += 1;
            }
            None => {
                lengths.push(length as usize);
                // Uncompressed batch buffers are plain slices of the already
                // bounded body — no decoder allocation — but their real bytes
                // still count toward the whole-stream aggregate.
                *total = total
                    .checked_add(length)
                    .ok_or_else(|| invalid_ipc("IPC buffer lengths overflow the payload bound"))?;
                if *total > max_bytes as u64 {
                    return Err(invalid_ipc(format!(
                        "IPC stream declares more than {max_bytes} uncompressed buffer bytes in total"
                    )));
                }
            }
        }
    }
    Ok(lengths)
}

/// Walk an encapsulated Arrow IPC stream with the generated, verified
/// flatbuffer parser (`root_as_message`): framing is checked, each message is
/// verified, and every record/dictionary batch has its buffer expansion
/// bounded and aggregated within `max_bytes` — all BEFORE the real Arrow
/// decoder runs. Valid compression is preserved (verified, then decoded a
/// second time from the immutable bytes by the standard reader).
fn walk_ipc_bounds(bytes: &[u8], max_bytes: usize) -> Result<Vec<IpcMessageBounds>, Status> {
    walk_ipc_bounds_with_budget(bytes, max_bytes, |_| {})
}
// Production only supplies the no-op above. A cfg(test)-only method on the
// private budget permits tiny safe amplification fixtures without changing
// input/schema bounds or exposing a caller-selectable production allowance.
pub(super) fn walk_ipc_bounds_with_budget(
    bytes: &[u8],
    max_bytes: usize,
    configure: impl FnOnce(&mut super::decision_ipc_metadata::MetadataBudget<'_>),
) -> Result<Vec<IpcMessageBounds>, Status> {
    if bytes.len() > max_bytes {
        return Err(invalid_ipc(format!(
            "decision payload is {} bytes, above the {max_bytes}-byte bound",
            bytes.len()
        )));
    }
    // Arrow StreamWriter emits continuation+metadata_size framing at byte 0
    // (the ARROW1 magic belongs to the FILE format and is not written here).
    let mut pos = 0usize;
    let mut walked: Vec<IpcMessageBounds> = Vec::new();
    // Whole-stream aggregate of REAL payload bytes (verified decompressed
    // sizes for compressed buffers, plain body bytes for uncompressed ones)
    // across record AND dictionary batches — bounded by max_bytes, not merely
    // per buffer or per message.
    let mut total: u64 = 0;
    // RECORD rows aggregate across ALL record-batch messages: the canonical
    // decoder concatenates every message, so the MERGED batch must respect
    // MAX_DECISION_ROWS_PER_BATCH — a per-message cap alone would let many
    // individually-valid messages exceed it after concat. Dictionary batches
    // are excluded (their VALUES legitimately exceed the row count).
    let mut record_rows: u64 = 0;
    let mut allocation = super::decision_ipc_metadata::MetadataBudget::new(max_bytes);
    configure(&mut allocation);
    while pos < bytes.len() {
        let mut size_buf = [0u8; 4];
        size_buf.copy_from_slice(
            bytes
                .get(pos..pos + 4)
                .ok_or_else(|| invalid_ipc("truncated IPC framing"))?,
        );
        let first = u32::from_le_bytes(size_buf);
        if first == 0 {
            break; // end-of-stream marker
        }
        let metadata_size = if first == u32::MAX {
            let size = read_u32_le(bytes, pos + 4)?;
            pos += 8;
            size
        } else {
            pos += 4;
            first
        };
        if metadata_size == 0 {
            break; // end-of-stream marker
        }
        let metadata_size = metadata_size as usize;
        if metadata_size > MAX_IPC_METADATA_BYTES {
            return Err(invalid_ipc(format!(
                "IPC metadata message of {metadata_size} bytes exceeds the {MAX_IPC_METADATA_BYTES}-byte bound"
            )));
        }
        let metadata_start = pos;
        let metadata_end = pos
            .checked_add(metadata_size)
            .ok_or_else(|| invalid_ipc("IPC metadata size overflow"))?;
        if metadata_end > bytes.len() {
            return Err(invalid_ipc("truncated IPC metadata message"));
        }
        pos = metadata_end;

        // Generated flatbuffers verifier: rejects malformed metadata before
        // any header access.
        let message = root_as_message(&bytes[metadata_start..metadata_end])
            .map_err(|e| invalid_ipc(format!("invalid IPC metadata: {e}")))?;
        allocation.message()?;
        let body_length = message.bodyLength();
        if body_length < 0 {
            return Err(invalid_ipc("negative IPC message body length"));
        }
        let body_end = pos
            .checked_add(body_length as usize)
            .ok_or_else(|| invalid_ipc("IPC body size overflow"))?;
        if body_end > bytes.len() {
            return Err(invalid_ipc("truncated IPC message body"));
        }
        let body = &bytes[pos..body_end];

        let mut verified_buffers = 0usize;
        let mut verified_bytes = 0u64;
        let header_type = message.header_type();
        match header_type {
            MessageHeader::RecordBatch => {
                let rb = message
                    .header_as_record_batch()
                    .ok_or_else(|| invalid_ipc("record batch message without header"))?;
                // Signed -> checked aggregate BEFORE decode: the canonical
                // decoder concatenates all record messages into one batch.
                let rows = rb.length();
                if rows < 0 {
                    return Err(invalid_ipc("negative IPC record batch row count"));
                }
                record_rows = record_rows
                    .checked_add(rows as u64)
                    .ok_or_else(|| invalid_ipc("IPC record row count overflow"))?;
                if record_rows > MAX_DECISION_ROWS_PER_BATCH as u64 {
                    return Err(invalid_ipc(format!(
                        "IPC stream declares {record_rows} record rows across all batches, above the {MAX_DECISION_ROWS_PER_BATCH}-row decision payload cap"
                    )));
                }
                allocation.record(&rb, None)?;
                let lengths = verify_record_batch_buffers(
                    &rb,
                    body,
                    max_bytes,
                    &mut total,
                    &mut verified_buffers,
                )?;
                allocation.layout(rb, None, message.version(), &lengths)?;
                verified_bytes = total;
            }
            MessageHeader::DictionaryBatch => {
                let dictionary = message
                    .header_as_dictionary_batch()
                    .ok_or_else(|| invalid_ipc("dictionary batch message without header"))?;
                let data = dictionary
                    .data()
                    .ok_or_else(|| invalid_ipc("dictionary batch without record data"))?;
                // Dictionary VALUES legitimately exceed the record row cap:
                // dictionary batches are never row-counted (root directive).
                allocation.record(&data, Some(dictionary.id()))?;
                let lengths = verify_record_batch_buffers(
                    &data,
                    body,
                    max_bytes,
                    &mut total,
                    &mut verified_buffers,
                )?;
                allocation.layout(
                    data,
                    Some((dictionary.id(), dictionary.isDelta())),
                    message.version(),
                    &lengths,
                )?;
                verified_bytes = total;
            }
            MessageHeader::Schema => {
                let schema = message
                    .header_as_schema()
                    .ok_or_else(|| invalid_ipc("schema message without header"))?;
                allocation.schema(schema)?;
            }
            other => {
                return Err(invalid_ipc(format!(
                    "unsupported IPC message header {other:?} in decision payload"
                )));
            }
        }
        walked.push(IpcMessageBounds {
            header_type,
            metadata: (metadata_start, metadata_end),
            body: (pos, body_end),
            verified_buffers,
            verified_bytes,
        });
        pos = body_end;
    }
    if walked.is_empty() {
        return Err(invalid_ipc("IPC payload contains no messages"));
    }
    allocation.finish()?;
    Ok(walked)
}

/// Verify a stream's framing, metadata and DECLARED/ACTUAL decompressed
/// expansion stay within `max_bytes` (see [`walk_ipc_bounds`]).
pub fn scan_ipc_bounds(bytes: &[u8], max_bytes: usize) -> Result<(), Status> {
    walk_ipc_bounds(bytes, max_bytes).map(|_| ())
}

/// The durable decode entry: bounded verification (framing, metadata, row
/// caps, actual decompressed expansion, whole-stream aggregate) runs FIRST,
/// then the canonical Arrow decoder runs on the immutable verified bytes.
/// Every durable record/write admission path MUST decode through this
/// function — never through the unbounded [`decode_batch_ipc`].
pub fn decode_batch_ipc_bounded(bytes: &[u8], max_bytes: usize) -> Result<RecordBatch, Status> {
    scan_ipc_bounds(bytes, max_bytes)?;
    decode_batch_ipc(bytes)
}

/// Decode an Arrow IPC stream payload back into a single batch
/// (multi-batch streams are concatenated).
pub fn decode_batch_ipc(bytes: &[u8]) -> Result<RecordBatch, Status> {
    let reader = StreamReader::try_new(std::io::Cursor::new(bytes), None)
        .map_err(|e| Status::internal(format!("IPC reader init: {e}")))?;
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|e| Status::internal(format!("IPC read: {e}")))?);
    }
    match batches.len() {
        0 => Err(Status::internal("IPC payload contained no batches")),
        1 => Ok(batches.remove(0)),
        _ => {
            let schema = batches[0].schema();
            arrow::compute::concat_batches(&schema, &batches)
                .map_err(|e| Status::internal(format!("IPC concat: {e}")))
        }
    }
}

fn meta_from_row(batch: &RecordBatch, row: usize) -> Result<DecisionBatchMeta, Status> {
    let string_at = |name: &str| -> Result<String, Status> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .map(|arr| arr.value(row).to_owned())
            .ok_or_else(|| Status::internal(format!("invalid {name} column")))
    };
    let i64_at = |name: &str| -> Result<i64, Status> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .map(|arr| arr.value(row))
            .ok_or_else(|| Status::internal(format!("invalid {name} column")))
    };
    let calib_version = batch
        .column_by_name("calib_version")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .and_then(|arr| {
            if arr.is_null(row) || arr.value(row).is_empty() {
                None
            } else {
                Some(arr.value(row).to_owned())
            }
        });
    Ok(DecisionBatchMeta {
        batch_id: string_at("batch_id")?,
        schema_fingerprint: string_at("schema_fingerprint")?,
        model_id: string_at("model_id")?,
        calib_version,
        created_at: i64_at("created_at")?,
        row_count: i64_at("row_count")?,
    })
}

fn payload_from_row(batch: &RecordBatch, row: usize) -> Result<RecordBatch, Status> {
    let col = batch
        .column_by_name("payload")
        .ok_or_else(|| Status::internal("missing payload column"))?;
    let bytes: Vec<u8> = if let Some(arr) = col.as_any().downcast_ref::<BinaryArray>() {
        arr.value(row).to_vec()
    } else if let Some(arr) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        arr.value(row).to_vec()
    } else {
        return Err(Status::internal(format!(
            "unexpected payload column type: {:?}",
            col.data_type()
        )));
    };
    decode_batch_ipc(&bytes)
}

/// Build the one-row RecordBatch for a decision-batch insert.
fn meta_to_record_batch(
    meta: &DecisionBatchMeta,
    payload_format: &str,
    payload: &[u8],
) -> Result<RecordBatch, Status> {
    let schema = Arc::new(decision_batch_schema());
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![meta.batch_id.as_str()])),
        Arc::new(StringArray::from(vec![meta.schema_fingerprint.as_str()])),
        Arc::new(StringArray::from(vec![meta.model_id.as_str()])),
        Arc::new(StringArray::from(vec![meta.calib_version.as_deref()])),
        Arc::new(Int64Array::from(vec![meta.created_at])),
        Arc::new(Int64Array::from(vec![meta.row_count])),
        Arc::new(StringArray::from(vec![payload_format])),
        Arc::new(BinaryArray::from(vec![payload])),
    ];
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| Status::internal(format!("Failed to create decision batch row: {e}")))
}

fn escape_sql(value: &str) -> String {
    value.replace('\'', "''")
}

/// Decision-batch store over any [`StorageBackend`]: the write/read API for
/// Arrow-native model output (eval harness, distillation corpora, re-fit
/// input).
pub struct DecisionBatchStore<B: StorageBackend> {
    backend: Arc<B>,
}

impl<B: StorageBackend> DecisionBatchStore<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Create the `decision_batches` table.
    pub async fn init(&self) -> Result<(), Status> {
        self.backend
            .create_table(DECISION_BATCH_TABLE, &decision_batch_schema())
            .await
    }

    /// Store one model-emitted batch. Returns the minted batch id.
    pub async fn put(
        &self,
        schema_fingerprint: &str,
        model_id: &str,
        calib_version: Option<&str>,
        batch: &RecordBatch,
    ) -> Result<String, Status> {
        if schema_fingerprint.is_empty() {
            return Err(Status::invalid_argument(
                "schema_fingerprint must not be empty",
            ));
        }
        if model_id.is_empty() {
            return Err(Status::invalid_argument("model_id must not be empty"));
        }
        let meta = DecisionBatchMeta {
            batch_id: mint_batch_id(),
            schema_fingerprint: schema_fingerprint.to_owned(),
            model_id: model_id.to_owned(),
            calib_version: calib_version.map(str::to_owned),
            created_at: now_unix_ms(),
            row_count: i64::try_from(batch.num_rows())
                .map_err(|_| Status::invalid_argument("batch row count overflows i64"))?,
        };
        let payload = encode_batch_ipc(batch)?;
        let row = meta_to_record_batch(&meta, PAYLOAD_FORMAT_IPC_STREAM, &payload)?;
        self.backend
            .insert_into_table(DECISION_BATCH_TABLE, row)
            .await?;
        Ok(meta.batch_id)
    }

    /// Fetch one batch by id, decoding the payload back to its RecordBatch.
    pub async fn get(
        &self,
        batch_id: &str,
    ) -> Result<Option<(DecisionBatchMeta, RecordBatch)>, Status> {
        let sql = format!(
            "SELECT * FROM {DECISION_BATCH_TABLE} WHERE batch_id = '{}'",
            escape_sql(batch_id)
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let meta = meta_from_row(&batch, 0)?;
        let payload = payload_from_row(&batch, 0)?;
        Ok(Some((meta, payload)))
    }

    /// All batches for a decision-schema fingerprint, oldest first.
    pub async fn fetch_by_schema(
        &self,
        schema_fingerprint: &str,
    ) -> Result<Vec<(DecisionBatchMeta, RecordBatch)>, Status> {
        let sql = format!(
            "SELECT * FROM {DECISION_BATCH_TABLE} WHERE schema_fingerprint = '{}' \
             ORDER BY created_at ASC",
            escape_sql(schema_fingerprint)
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        let mut out = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            out.push((meta_from_row(&batch, row)?, payload_from_row(&batch, row)?));
        }
        Ok(out)
    }

    /// List batch metadata, optionally filtered by schema fingerprint.
    pub async fn list(
        &self,
        schema_fingerprint: Option<&str>,
    ) -> Result<Vec<DecisionBatchMeta>, Status> {
        let mut sql = format!(
            "SELECT batch_id, schema_fingerprint, model_id, calib_version, created_at, row_count \
             FROM {DECISION_BATCH_TABLE}"
        );
        if let Some(fingerprint) = schema_fingerprint {
            sql.push_str(&format!(
                " WHERE schema_fingerprint = '{}'",
                escape_sql(fingerprint)
            ));
        }
        sql.push_str(" ORDER BY created_at ASC");
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;
        (0..batch.num_rows())
            .map(|row| meta_from_row(&batch, row))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod bounded_decode_tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::IpcWriteOptions;
    use std::sync::Arc;

    fn sample_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("n", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            ],
        )
        .unwrap()
    }

    // Arrow stores incompressible buffers verbatim with prefix -1. Use a
    // small repetitive batch so descriptor/expansion negatives mutate a real
    // codec frame, rather than treating a tiny raw buffer as compressed data.
    fn compressible_batch() -> RecordBatch {
        let schema = sample_batch().schema();
        let text = "repeated compression fixture ".repeat(16);
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![text.as_str(); 256])) as ArrayRef,
                Arc::new(Int64Array::from(vec![0; 256])) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn encode_with(batch: &RecordBatch, codec: Option<&str>) -> Vec<u8> {
        let compression = codec.map(|c| match c {
            "zstd" => arrow::ipc::CompressionType::ZSTD,
            _ => arrow::ipc::CompressionType::LZ4_FRAME,
        });
        let options = IpcWriteOptions::default()
            .try_with_compression(compression)
            .unwrap();
        let mut buf = Vec::new();
        {
            let mut writer =
                StreamWriter::try_new_with_options(&mut buf, &batch.schema(), options).unwrap();
            writer.write(batch).unwrap();
            writer.finish().unwrap();
        }
        buf
    }

    #[test]
    fn bounded_decode_roundtrip_plain() {
        let bytes = encode_batch_ipc(&sample_batch()).unwrap();
        scan_ipc_bounds(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        let decoded = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        assert_eq!(decoded.num_rows(), 3);
    }

    #[test]
    fn bounded_decode_preserves_valid_compression() {
        for codec in ["zstd", "lz4"] {
            let bytes = encode_with(&sample_batch(), Some(codec));
            scan_ipc_bounds(&bytes, MAX_PAYLOAD_BYTES).unwrap();
            let decoded = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap();
            assert_eq!(decoded.num_rows(), 3, "{codec}");
            assert_eq!(
                decoded
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value(2),
                "gamma",
                "{codec}: lossless through verified decompression"
            );
        }
    }

    #[test]
    fn bounded_decode_walks_dictionary_batches() {
        use arrow::array::DictionaryArray;
        let dict: DictionaryArray<arrow::datatypes::Int32Type> =
            vec!["alpha", "beta", "alpha"].into_iter().collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "tag",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(dict) as ArrayRef]).unwrap();
        let bytes = encode_batch_ipc(&batch).unwrap();
        scan_ipc_bounds(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        let decoded = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        assert_eq!(decoded.num_rows(), 3);
    }

    /// The actual compression-bomb shape: advertise a tiny length, supply a
    /// stream that decompresses far beyond it. The bounded verification must
    /// reject BEFORE Arrow allocates, using a hard-capped streaming read.
    #[test]
    fn bounded_decode_rejects_actual_expansion_beyond_advertised() {
        let batch = compressible_batch();
        let bytes = encode_with(&batch, Some("lz4"));
        assert_eq!(
            decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap(),
            batch
        );
        let (prefix_pos, _, advertised) =
            genuinely_compressed_buffer(&bytes, [0x04, 0x22, 0x4D, 0x18]);
        assert!(
            advertised > 8,
            "fixture must expand beyond the forged prefix"
        );
        let mut bytes = bytes.clone();
        // The bomb: claim 8 bytes, the stream actually expands to `advertised`.
        bytes[prefix_pos..prefix_pos + 8].copy_from_slice(&8i64.to_le_bytes());
        let err = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err}");
        assert!(
            err.message().contains("beyond its advertised length"),
            "{err}"
        );
    }

    /// Framing walk to the FIRST record-batch message:
    /// (metadata_start, metadata_end, body_start). StreamWriter framing
    /// starts at byte 0 (no magic).
    fn record_batch_message_range(bytes: &[u8]) -> (usize, usize, usize) {
        let mut pos = 0usize;
        loop {
            let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let meta_start = pos + 8;
            let meta_end = meta_start + size;
            let body_start = meta_end;
            let message = root_as_message(&bytes[meta_start..meta_end]).unwrap();
            if message.header_as_record_batch().is_some() {
                return (meta_start, meta_end, body_start);
            }
            pos = body_start + message.bodyLength() as usize;
        }
    }

    // Follow real message metadata from byte0 and select a positive length
    // prefix, excluding Arrow's 0/empty and -1/verbatim buffer representations.
    // Return (prefix, codec frame, advertised decoded bytes) for one mutation.
    fn genuinely_compressed_buffer(bytes: &[u8], magic: [u8; 4]) -> (usize, usize, i64) {
        let (meta_start, meta_end, body_start) = record_batch_message_range(bytes);
        let message = root_as_message(&bytes[meta_start..meta_end]).unwrap();
        let rb = message.header_as_record_batch().unwrap();
        assert!(rb.compression().is_some());
        let buffers = rb.buffers().unwrap();
        let mut chosen = None;
        for i in 0..buffers.len() {
            let buffer = buffers.get(i);
            if buffer.length() < 12 {
                continue;
            }
            let prefix = body_start + usize::try_from(buffer.offset()).unwrap();
            let advertised = i64::from_le_bytes(bytes[prefix..prefix + 8].try_into().unwrap());
            if advertised > 8 && chosen.is_none_or(|(_, _, prior)| advertised > prior) {
                chosen = Some((prefix, prefix + 8, advertised));
            }
        }
        let (prefix, frame, advertised) = chosen.expect("a genuine compressed codec frame");
        assert_eq!(&bytes[frame..frame + 4], magic.as_slice());
        (prefix, frame, advertised)
    }

    #[test]
    fn bounded_decode_rejects_advertised_mismatch() {
        // Take a real lz4 stream, locate the advertised prefix (first buffer
        // data starts at body offset given by metadata) and inflate the
        // prefix: actual then disagrees with the advertised size.
        let batch = compressible_batch();
        let bytes = encode_with(&batch, Some("lz4"));
        assert_eq!(
            decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap(),
            batch
        );
        let (prefix_pos, _, advertised) =
            genuinely_compressed_buffer(&bytes, [0x04, 0x22, 0x4D, 0x18]);
        let mut bytes = bytes.clone();
        assert!(advertised > 0, "test targets a genuinely compressed buffer");
        // Inflate: claim far more than the stream expands to.
        bytes[prefix_pos..prefix_pos + 8].copy_from_slice(&((advertised * 8).to_le_bytes()));
        let err = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err}");
        assert!(
            err.message().contains("advertised") || err.message().contains("expands"),
            "{err}"
        );
    }

    #[test]
    fn bounded_decode_rejects_bad_framing() {
        // Huge metadata size in framing.
        let bytes = encode_batch_ipc(&sample_batch()).unwrap();
        let mut bytes = bytes.clone();
        bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap_err();
        assert!(err.message().contains("metadata"), "{err}");

        // Truncated metadata/body.
        let bytes = encode_batch_ipc(&sample_batch()).unwrap();
        let err =
            decode_batch_ipc_bounded(&bytes[..bytes.len() / 2], MAX_PAYLOAD_BYTES).unwrap_err();
        assert!(
            err.message().contains("truncated") || err.message().contains("no messages"),
            "{err}"
        );
    }

    #[test]
    fn bounded_decode_rejects_oversize_encoded_input() {
        let bytes = encode_batch_ipc(&sample_batch()).unwrap();
        let mut padded = bytes.clone();
        padded.extend(std::iter::repeat_n(0u8, MAX_PAYLOAD_BYTES + 1));
        let err = decode_batch_ipc_bounded(&padded, MAX_PAYLOAD_BYTES).unwrap_err();
        assert!(err.message().contains("above the"), "{err}");
    }

    #[test]
    fn scan_rejects_record_batch_row_cap_before_decode() {
        // 4097 rows: one above the MAX_DECISION_ROWS_PER_BATCH cap — rejected
        // at the scan, before any Arrow array allocation.
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![
                0i64;
                MAX_DECISION_ROWS_PER_BATCH + 1
            ])) as ArrayRef],
        )
        .unwrap();
        let bytes = encode_batch_ipc(&batch).unwrap();
        let err = scan_ipc_bounds(&bytes, MAX_PAYLOAD_BYTES).unwrap_err();
        assert!(
            err.message()
                .contains("4097 record rows across all batches"),
            "{err}"
        );
        assert!(err.message().contains("decision payload cap"), "{err}");

        // Boundary: exactly the cap passes the scan and decodes.
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![0i64; MAX_DECISION_ROWS_PER_BATCH])) as ArrayRef],
        )
        .unwrap();
        let bytes = encode_batch_ipc(&batch).unwrap();
        scan_ipc_bounds(&bytes, MAX_PAYLOAD_BYTES).unwrap();
    }

    #[test]
    fn dictionary_length_independent_of_row_cap() {
        // 4097 distinct dictionary VALUES against 3 record rows: dictionary
        // batches legitimately exceed the row cap and must NOT be rejected.
        use arrow::array::DictionaryArray;
        let values: Vec<String> = (0..MAX_DECISION_ROWS_PER_BATCH + 1)
            .map(|i| format!("v{i}"))
            .collect();
        let keys = arrow::array::Int32Array::from(vec![0i32, 1, 2]);
        let values_arr = StringArray::from(values.iter().map(String::as_str).collect::<Vec<_>>());
        let dict = DictionaryArray::<arrow::datatypes::Int32Type>::new(keys, Arc::new(values_arr));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "tag",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(dict) as ArrayRef]).unwrap();
        let bytes = encode_batch_ipc(&batch).unwrap();
        scan_ipc_bounds(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        let decoded = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        assert_eq!(decoded.num_rows(), 3);
    }

    /// Wide-binary aggregate fixture: <= 4096 record rows whose BUFFER bytes
    /// alone cross the 16 MiB whole-stream bound (ZSTD keeps the encoded
    /// input tiny). Over-limit must fail with the AGGREGATE cause; the
    /// under-limit twin must decode.
    fn wide_batch(columns: usize, rows: usize, value_bytes: usize) -> RecordBatch {
        let value = vec![0u8; value_bytes];
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        for c in 0..columns {
            fields.push(Field::new(format!("wide{c}"), DataType::Utf8, false));
            arrays.push(Arc::new(StringArray::from(vec![
                String::from_utf8_lossy(&value)
                    .into_owned();
                rows
            ])) as ArrayRef);
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    #[test]
    fn aggregate_spans_whole_stream_including_dictionaries() {
        // 5 columns x 4 rows x 1 MiB = ~20 MiB of REAL buffer bytes across
        // several compressed buffers (encoded input stays tiny; each single
        // buffer is only ~4 MiB, below the per-buffer bound); the whole-
        // stream aggregate crosses the 16 MiB bound.
        let six = encode_with(&wide_batch(5, 4, 1024 * 1024), Some("zstd"));
        assert!(
            six.len() < 1024 * 1024,
            "test premise: zeros compress far below the bound"
        );
        let err = decode_batch_ipc_bounded(&six, MAX_PAYLOAD_BYTES).unwrap_err();
        assert!(
            err.message().contains("in total"),
            "expected the aggregate cause, got: {err}"
        );
        assert!(
            !err.message().contains("row cap"),
            "aggregate failure must not be misattributed to the row cap"
        );

        // Under-limit twin: 3 columns (~12 MiB aggregate) decode fully.
        let two = encode_with(&wide_batch(3, 4, 1024 * 1024), Some("zstd"));
        let decoded = decode_batch_ipc_bounded(&two, MAX_PAYLOAD_BYTES).unwrap();
        assert_eq!(decoded.num_rows(), 4);
        let col = decoded
            .column_by_name("wide0")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(col.value(0).len(), 1024 * 1024);
    }

    /// Arrow56 empty-buffer parity: an all-null column's validity buffer is
    /// EMPTY (length 0, no prefix) in a compressed stream and must decode.
    #[test]
    fn bounded_decode_accepts_empty_compressed_buffers() {
        let schema = Arc::new(Schema::new(vec![Field::new("maybe", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![None::<&str>, None, None])) as ArrayRef],
        )
        .unwrap();
        let bytes = encode_with(&batch, Some("zstd"));
        scan_ipc_bounds(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        let decoded = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        assert_eq!(decoded.num_rows(), 3);
    }

    /// Scanner-level Arrow56 parity: an advertised prefix of 0 means "empty
    /// buffer" — no decompression, no expansion, so the SCAN accepts. For a
    /// semantically NECESSARY buffer (the Int64 values buffer of a 3-row
    /// batch) the malformed record must then be rejected by the CANONICAL
    /// decoder, and the bounded decoder must AGREE (same rejection) — a
    /// codec-produced empty buffer does not make the record valid.
    #[test]
    fn zero_prefix_scan_accepts_then_canonical_and_bounded_agree_on_rejection() {
        let bytes = encode_with(&sample_batch(), Some("zstd"));
        let (meta_start, _meta_end, body_start) = record_batch_message_range(&bytes);
        let message = root_as_message(&bytes[meta_start..]).unwrap();
        let rb = message.header_as_record_batch().unwrap();
        let buffers = rb.buffers().unwrap();
        // The semantically necessary buffer: the largest advertised one (the
        // Int64 values buffer of the 3-row batch).
        let mut chosen: Option<usize> = None;
        let mut chosen_len = 0i64;
        for i in 0..buffers.len() {
            let b = buffers.get(i);
            if b.length() > chosen_len {
                chosen_len = b.length();
                chosen = Some(body_start + b.offset() as usize);
            }
        }
        let prefix_pos = chosen.expect("a non-empty compressed buffer");
        let mut bytes = bytes.clone();
        bytes[prefix_pos..prefix_pos + 8].copy_from_slice(&0i64.to_le_bytes());

        // 1) Scanner parity: advertised 0 is not an expansion — accepted.
        scan_ipc_bounds(&bytes, MAX_PAYLOAD_BYTES).unwrap();

        // 2) The canonical decoder rejects the now-malformed record...
        let canonical_err = decode_batch_ipc(&bytes)
            .expect_err("erasing a necessary buffer must fail canonical decode");

        // 3) ...and the bounded decoder agrees on the same malformed bytes
        //    (its extra gate adds no divergence here).
        let bounded_err = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES)
            .expect_err("bounded decode must agree with canonical rejection");
        assert_eq!(canonical_err.message(), bounded_err.message());
        assert_eq!(canonical_err.code(), bounded_err.code());
    }

    /// ZSTD decoder-internal window: a hostile frame header declaring a huge
    /// back-reference window is rejected by the REAL decoder's native
    /// window_log_max(24) constraint BEFORE its unbounded read_to_end path
    /// can run (the output take() alone would not bound this allocation).
    #[test]
    fn bounded_decode_rejects_high_zstd_window() {
        let batch = compressible_batch();
        let bytes = encode_with(&batch, Some("zstd"));
        assert_eq!(
            decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap(),
            batch
        );
        let (_, frame_pos, _) = genuinely_compressed_buffer(&bytes, [0x28, 0xB5, 0x2F, 0xFD]);
        let mut bytes = bytes.clone();
        // Window descriptor follows the frame-header descriptor byte.
        let wd_pos = frame_pos + 5;
        assert!(
            (bytes[wd_pos - 1] >> 5) & 1 == 0,
            "writer emits a windowed frame"
        );
        bytes[wd_pos] = 0x7F; // windowLog 25 -> ~60 MiB window, far above 2^24
        let err = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err}");
        assert!(
            err.message()
                .contains("Frame requires too much memory for decoding"),
            "expected the native window-limit rejection, got: {err}"
        );
    }

    /// LZ4 frame descriptor: reserved bits are rejected before decode; the
    /// declared block-maximum allocation is descriptor-bounded (<= 4 MiB).
    #[test]
    fn bounded_decode_rejects_invalid_lz4_descriptor() {
        let batch = compressible_batch();
        let bytes = encode_with(&batch, Some("lz4"));
        assert_eq!(
            decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap(),
            batch
        );
        let (_, frame_pos, _) = genuinely_compressed_buffer(&bytes, [0x04, 0x22, 0x4D, 0x18]);
        let mut bytes = bytes.clone();
        // Frame magic is4 bytes, then FLG at+4 and BD at+5. The previous
        // +6 mutation targeted a checksum/optional header byte and OR could
        // be a no-op. Set FLG's decoder-enforced reserved bit instead.
        let flg_pos = frame_pos + 4;
        let original_flg = bytes[flg_pos];
        assert_eq!(original_flg & 0b0000_0010, 0, "valid FLG control");
        bytes[flg_pos] |= 0b0000_0010;
        assert_ne!(bytes[flg_pos], original_flg, "negative must change FLG");
        let err = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap_err();
        assert!(
            err.message().contains("ReservedBitsSet"),
            "expected a descriptor rejection, got: {err}"
        );
    }

    /// RECORD rows aggregate across ALL record messages: three individually
    /// valid 2000-row messages concatenate to 6000 rows — rejected. The exact
    /// 4096-row boundary across two messages is accepted.
    #[test]
    fn scan_aggregates_record_rows_across_messages() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let batch_of = |rows: usize| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(vec![0i64; rows])) as ArrayRef],
            )
            .unwrap()
        };
        let multi = |counts: &[usize]| {
            let mut buf = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buf, &schema).unwrap();
                for &c in counts {
                    writer.write(&batch_of(c)).unwrap();
                }
                writer.finish().unwrap();
            }
            buf
        };

        // Over the aggregate: 3 x 2000 = 6000 > 4096.
        let err = scan_ipc_bounds(&multi(&[2000, 2000, 2000]), MAX_PAYLOAD_BYTES).unwrap_err();
        assert!(
            err.message()
                .contains("6000 record rows across all batches"),
            "{err}"
        );

        // Exact boundary: 2 x 2048 = 4096 — accepted (and decodes).
        let bytes = multi(&[2048, 2048]);
        scan_ipc_bounds(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        let decoded = decode_batch_ipc_bounded(&bytes, MAX_PAYLOAD_BYTES).unwrap();
        assert_eq!(decoded.num_rows(), 4096);
    }
}
