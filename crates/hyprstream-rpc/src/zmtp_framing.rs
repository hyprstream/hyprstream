//! Pure ZMTP 3.1 framing — zero I/O, zero async, wasm32-safe.
//!
//! This module provides synchronous byte-manipulation functions for ZMTP 3.1
//! frame encoding/decoding, greeting construction/validation, and command parsing.
//! It is compiled on **all** targets (including wasm32) and serves as the single
//! source of truth for ZMTP wire format logic.
//!
//! Async I/O wrappers live in `transport::zmtp_quic::ZmtpStream`.

use anyhow::{anyhow, ensure, Result};

// ============================================================================
// Constants
// ============================================================================

/// ZMTP protocol version.
pub const ZMTP_VERSION_MAJOR: u8 = 3;
pub const ZMTP_VERSION_MINOR: u8 = 1;

/// ZMTP greeting size (always 64 bytes).
pub const GREETING_SIZE: usize = 64;

/// ZMTP mechanism name for NULL (no encryption at ZMTP layer).
pub const MECHANISM_NULL: &[u8; 4] = b"NULL";

/// Maximum allowed frame size (64 MiB) — prevents unbounded allocation from malicious frames.
pub const MAX_FRAME_SIZE: usize = 64 * 1024 * 1024;

// ============================================================================
// Types
// ============================================================================

/// ZMQ socket types for ZMTP handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZmqSocketType {
    Req,
    Rep,
    Pub,
    Sub,
    XPub,
    XSub,
    Push,
    Pull,
}

impl ZmqSocketType {
    /// Get the ZMTP socket type string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Req => "REQ",
            Self::Rep => "REP",
            Self::Pub => "PUB",
            Self::Sub => "SUB",
            Self::XPub => "XPUB",
            Self::XSub => "XSUB",
            Self::Push => "PUSH",
            Self::Pull => "PULL",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "REQ" => Ok(Self::Req),
            "REP" => Ok(Self::Rep),
            "PUB" => Ok(Self::Pub),
            "SUB" => Ok(Self::Sub),
            "XPUB" => Ok(Self::XPub),
            "XSUB" => Ok(Self::XSub),
            "PUSH" => Ok(Self::Push),
            "PULL" => Ok(Self::Pull),
            _ => Err(anyhow!("unknown socket type: {}", s)),
        }
    }
}

/// A parsed ZMTP frame (owned, no `Bytes` dependency).
#[derive(Debug, Clone)]
pub struct ZmtpFrame {
    /// MORE flag: true if more frames follow in this message.
    pub more: bool,
    /// COMMAND flag: true if this is a command frame, false for message frame.
    pub command: bool,
    /// Frame body data.
    pub data: Vec<u8>,
}

/// A parsed ZMTP command.
#[derive(Debug, Clone)]
pub struct ZmtpCommand {
    /// Command name (e.g., "READY", "SUBSCRIBE", "CANCEL").
    pub name: String,
    /// Command-specific body.
    pub body: Vec<u8>,
}

impl ZmtpCommand {
    /// Parse a command from frame body data.
    ///
    /// Format: `<name-size (1 byte)><name><body>`
    pub fn parse(data: &[u8]) -> Result<Self> {
        ensure!(!data.is_empty(), "Command frame body is empty");

        let name_len = data[0] as usize;
        ensure!(
            1 + name_len <= data.len(),
            "Command name truncated: need {} bytes, have {}",
            1 + name_len,
            data.len()
        );

        let name = std::str::from_utf8(&data[1..1 + name_len])?.to_owned();
        let body = data[1 + name_len..].to_vec();

        Ok(Self { name, body })
    }
}

// ============================================================================
// Greeting
// ============================================================================

/// Build the 64-byte ZMTP greeting.
///
/// Format:
/// ```text
/// Bytes  0–9:  Signature: 0xFF + 8×0x00 + 0x7F
/// Byte    10:  Version major: 0x03
/// Byte    11:  Version minor: 0x01
/// Bytes 12–31: Mechanism: "NULL" + 16×0x00 (padded to 20 bytes)
/// Byte    32:  as-server: 0x00 (MUST be zero for NULL mechanism)
/// Bytes 33–63: Reserved: 31×0x00
/// ```
pub fn build_greeting() -> [u8; GREETING_SIZE] {
    let mut greeting = [0u8; GREETING_SIZE];

    // Signature: 0xFF + 8 zeros + 0x7F
    greeting[0] = 0xFF;
    greeting[9] = 0x7F;

    // Version: 3.1
    greeting[10] = ZMTP_VERSION_MAJOR;
    greeting[11] = ZMTP_VERSION_MINOR;

    // Mechanism: "NULL" padded to 20 bytes
    greeting[12..16].copy_from_slice(MECHANISM_NULL);

    greeting
}

/// Validate a peer's ZMTP greeting.
pub fn validate_greeting(greeting: &[u8]) -> Result<()> {
    ensure!(greeting.len() >= GREETING_SIZE, "Greeting too short: {} bytes", greeting.len());

    ensure!(
        greeting[0] == 0xFF,
        "Invalid greeting signature byte 0: expected 0xFF"
    );
    ensure!(
        greeting[1..9].iter().all(|&b| b == 0),
        "Invalid greeting signature bytes 1-8: expected zeros"
    );
    ensure!(
        greeting[9] == 0x7F,
        "Invalid greeting signature byte 9: expected 0x7F"
    );

    // Check version (major must match)
    ensure!(
        greeting[10] == ZMTP_VERSION_MAJOR,
        "ZMTP version mismatch: expected major {}, got {}",
        ZMTP_VERSION_MAJOR,
        greeting[10]
    );

    // Check minor version
    ensure!(
        greeting[11] >= ZMTP_VERSION_MINOR,
        "ZMTP version mismatch: expected minor >= {}, got {}",
        ZMTP_VERSION_MINOR,
        greeting[11]
    );

    // Check mechanism is "NULL"
    let mechanism_name = &greeting[12..16];
    ensure!(
        mechanism_name == MECHANISM_NULL,
        "ZMTP mechanism mismatch: expected NULL, got {:?}",
        std::str::from_utf8(mechanism_name).unwrap_or("<invalid>")
    );

    // Check mechanism padding (bytes 16..32 must be zero)
    ensure!(
        greeting[16..32].iter().all(|&b| b == 0),
        "ZMTP greeting has non-zero mechanism padding bytes"
    );

    // For NULL mechanism, as-server MUST be 0x00 (RFC 37)
    ensure!(
        greeting[32] == 0x00,
        "NULL mechanism requires as-server=0x00, got 0x{:02X}",
        greeting[32]
    );

    Ok(())
}

// ============================================================================
// READY metadata
// ============================================================================

/// Build READY metadata with size-prefixed property encoding (RFC 37).
///
/// Format: `<name-length (1 byte)><name><value-length (4 bytes BE)><value>`
pub fn build_ready_metadata(socket_type: ZmqSocketType) -> Vec<u8> {
    let name = b"Socket-Type";
    let value = socket_type.as_str().as_bytes();

    let mut buf = Vec::with_capacity(1 + name.len() + 4 + value.len());

    buf.push(name.len() as u8);
    buf.extend_from_slice(name);
    buf.extend_from_slice(&(value.len() as u32).to_be_bytes());
    buf.extend_from_slice(value);

    buf
}

/// Validate a READY command.
pub fn validate_ready_command(cmd: &ZmtpCommand) -> Result<()> {
    ensure!(
        cmd.name == "READY",
        "Expected READY command, got {}",
        cmd.name
    );

    let mut pos = 0;
    let body = &cmd.body;

    while pos < body.len() {
        let name_len = body[pos] as usize;
        pos += 1;

        ensure!(
            pos + name_len <= body.len(),
            "READY metadata truncated at name"
        );
        let _name = std::str::from_utf8(&body[pos..pos + name_len])?;
        pos += name_len;

        ensure!(
            pos + 4 <= body.len(),
            "READY metadata truncated at value length"
        );
        let value_len =
            u32::from_be_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        pos += 4;

        ensure!(
            pos + value_len <= body.len(),
            "READY metadata truncated at value"
        );
        let _value = std::str::from_utf8(&body[pos..pos + value_len])?;
        pos += value_len;
    }

    Ok(())
}

// ============================================================================
// Frame encoding / decoding
// ============================================================================

/// Encode a single ZMTP frame to bytes.
pub fn encode_frame(more: bool, command: bool, data: &[u8]) -> Vec<u8> {
    let long = data.len() > 255;
    let flags: u8 = (more as u8) | ((long as u8) << 1) | ((command as u8) << 2);

    let header_len = 1 + if long { 8 } else { 1 };
    let mut buf = Vec::with_capacity(header_len + data.len());
    buf.push(flags);

    if long {
        buf.extend_from_slice(&(data.len() as u64).to_be_bytes());
    } else {
        buf.push(data.len() as u8);
    }

    buf.extend_from_slice(data);
    buf
}

/// Decode a single ZMTP frame from a buffer.
///
/// Returns `(frame, bytes_consumed)`.
pub fn decode_frame(buf: &[u8]) -> Result<(ZmtpFrame, usize)> {
    ensure!(!buf.is_empty(), "buffer empty");

    let flags = buf[0];
    let more = (flags & 0x01) != 0;
    let long = (flags & 0x02) != 0;
    let command = (flags & 0x04) != 0;

    ensure!(flags & 0xF8 == 0, "ZMTP frame has reserved bits set: 0x{:02X}", flags);

    let (size, header_len): (usize, usize) = if long {
        ensure!(buf.len() >= 9, "buffer too short for long frame header");
        let raw_size = u64::from_be_bytes([
            buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8],
        ]);
        ensure!(
            raw_size <= MAX_FRAME_SIZE as u64,
            "frame size {} exceeds maximum {}",
            raw_size,
            MAX_FRAME_SIZE
        );
        (raw_size as usize, 9)
    } else {
        ensure!(buf.len() >= 2, "buffer too short for short frame header");
        (buf[1] as usize, 2)
    };

    let total = header_len
        .checked_add(size)
        .ok_or_else(|| anyhow!("frame size overflow: header_len={} + size={}", header_len, size))?;
    ensure!(
        buf.len() >= total,
        "buffer too short for frame body: need {}, have {}",
        total,
        buf.len()
    );

    let data = buf[header_len..header_len + size].to_vec();
    Ok((ZmtpFrame { more, command, data }, header_len + size))
}

/// Encode a ZMTP command frame.
///
/// Command frame body format: `<name-size (1 byte)><name><body>`
pub fn encode_command(name: &str, body: &[u8]) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let mut cmd_body = Vec::with_capacity(1 + name_bytes.len() + body.len());
    cmd_body.push(name_bytes.len() as u8);
    cmd_body.extend_from_slice(name_bytes);
    cmd_body.extend_from_slice(body);
    encode_frame(false, true, &cmd_body)
}

/// Encode a multipart message as ZMTP frames.
///
/// Each part becomes one message frame (COMMAND=0). The last frame has MORE=0.
pub fn encode_multipart(parts: &[&[u8]]) -> Vec<u8> {
    let mut buf = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        let more = i < parts.len() - 1;
        buf.extend_from_slice(&encode_frame(more, false, part));
    }
    buf
}

/// Decode a multipart message from a buffer of concatenated ZMTP frames.
///
/// Returns `(parts, total_bytes_consumed)`.
pub fn decode_multipart(buf: &[u8]) -> Result<(Vec<Vec<u8>>, usize)> {
    let mut parts = Vec::new();
    let mut offset = 0;

    loop {
        ensure!(offset < buf.len(), "unexpected end of multipart message");
        let (frame, consumed) = decode_frame(&buf[offset..])?;

        ensure!(!frame.command, "unexpected command frame in multipart message");

        let more = frame.more;
        parts.push(frame.data);
        offset += consumed;

        if !more {
            break;
        }
    }

    Ok((parts, offset))
}

/// Encode multipart with flat encoding for WASM boundary.
///
/// Input: `[4B-len, data, 4B-len, data, ...]`
/// Output: ZMTP-framed bytes.
pub fn encode_multipart_flat(flat: &[u8]) -> Result<Vec<u8>> {
    let parts = decode_flat_parts(flat)?;
    let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
    Ok(encode_multipart(&refs))
}

/// Decode ZMTP frames to flat encoding for WASM boundary.
///
/// Input: concatenated ZMTP frames.
/// Output: `[4B-len, data, 4B-len, data, ...]`
pub fn decode_frames_to_flat(buf: &[u8]) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let mut offset = 0;

    while offset < buf.len() {
        let (frame, consumed) = decode_frame(&buf[offset..])?;

        // Skip command frames — only include message data frames
        if frame.command {
            offset += consumed;
            continue;
        }

        result.extend_from_slice(&(frame.data.len() as u32).to_be_bytes());
        result.extend_from_slice(&frame.data);
        offset += consumed;

        if !frame.more {
            break;
        }
    }

    Ok(result)
}

/// Decode flat-encoded parts: `[4B-len, data, 4B-len, data, ...]`
pub fn decode_flat_parts(flat: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut parts = Vec::new();
    let mut offset = 0;

    while offset < flat.len() {
        ensure!(
            offset + 4 <= flat.len(),
            "flat encoding truncated at length field"
        );
        let len =
            u32::from_be_bytes([flat[offset], flat[offset + 1], flat[offset + 2], flat[offset + 3]])
                as usize;
        offset += 4;

        ensure!(
            offset + len <= flat.len(),
            "flat encoding truncated at data"
        );
        parts.push(flat[offset..offset + len].to_vec());
        offset += len;
    }

    Ok(parts)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_roundtrip() {
        let greeting = build_greeting();
        assert_eq!(greeting.len(), GREETING_SIZE);
        validate_greeting(&greeting).unwrap();
    }

    #[test]
    fn greeting_rejects_bad_signature() {
        let mut greeting = build_greeting();
        greeting[0] = 0x00;
        assert!(validate_greeting(&greeting).is_err());
    }

    #[test]
    fn greeting_rejects_wrong_version() {
        let mut greeting = build_greeting();
        greeting[10] = 2; // Wrong major
        assert!(validate_greeting(&greeting).is_err());
    }

    #[test]
    fn frame_short_roundtrip() {
        let data = vec![0x42u8; 100];
        let encoded = encode_frame(false, false, &data);
        let (decoded, consumed) = decode_frame(&encoded).unwrap();

        assert_eq!(consumed, encoded.len());
        assert!(!decoded.more);
        assert!(!decoded.command);
        assert_eq!(decoded.data, data);
    }

    #[test]
    fn frame_long_roundtrip() {
        let data = vec![0xABu8; 500];
        let encoded = encode_frame(false, false, &data);
        let (decoded, consumed) = decode_frame(&encoded).unwrap();

        assert_eq!(consumed, encoded.len());
        assert!(!decoded.more);
        assert!(!decoded.command);
        assert_eq!(decoded.data, data);
        assert_eq!(encoded[0] & 0x02, 0x02); // LONG bit set
    }

    #[test]
    fn frame_more_flag() {
        let data = b"test";
        let encoded_more = encode_frame(true, false, data);
        let encoded_last = encode_frame(false, false, data);

        let (decoded_more, _) = decode_frame(&encoded_more).unwrap();
        let (decoded_last, _) = decode_frame(&encoded_last).unwrap();

        assert!(decoded_more.more);
        assert!(!decoded_last.more);
    }

    #[test]
    fn command_frame_roundtrip() {
        let encoded = encode_command("READY", b"metadata");
        let (frame, _) = decode_frame(&encoded).unwrap();

        assert!(frame.command);
        assert!(!frame.more);

        let cmd = ZmtpCommand::parse(&frame.data).unwrap();
        assert_eq!(cmd.name, "READY");
        assert_eq!(cmd.body, b"metadata");
    }

    #[test]
    fn multipart_roundtrip() {
        let parts: Vec<&[u8]> = vec![b"identity", b"delimiter", b"payload"];
        let encoded = encode_multipart(&parts);
        let (decoded, consumed) = decode_multipart(&encoded).unwrap();

        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0], b"identity");
        assert_eq!(decoded[1], b"delimiter");
        assert_eq!(decoded[2], b"payload");
    }

    #[test]
    fn ready_metadata_roundtrip() {
        let metadata = build_ready_metadata(ZmqSocketType::Rep);
        let cmd_frame = encode_command("READY", &metadata);
        let (frame, _) = decode_frame(&cmd_frame).unwrap();
        let cmd = ZmtpCommand::parse(&frame.data).unwrap();
        validate_ready_command(&cmd).unwrap();
    }

    #[test]
    fn flat_encoding_roundtrip() {
        let parts: Vec<&[u8]> = vec![b"hello", b"world"];
        let encoded = encode_multipart(&parts);
        let flat = decode_frames_to_flat(&encoded).unwrap();

        // Flat format: [4B-len, data, 4B-len, data]
        let decoded_parts = decode_flat_parts(&flat).unwrap();
        assert_eq!(decoded_parts.len(), 2);
        assert_eq!(decoded_parts[0], b"hello");
        assert_eq!(decoded_parts[1], b"world");

        // Re-encode from flat
        let re_encoded = encode_multipart_flat(&flat).unwrap();
        let (re_decoded, _) = decode_multipart(&re_encoded).unwrap();
        assert_eq!(re_decoded[0], b"hello");
        assert_eq!(re_decoded[1], b"world");
    }

    #[test]
    fn decode_frame_rejects_oversized() {
        // Construct a long frame header claiming MAX_FRAME_SIZE + 1 bytes
        let mut buf = vec![0u8; 9];
        buf[0] = 0x02; // LONG flag
        let oversized = (MAX_FRAME_SIZE as u64) + 1;
        buf[1..9].copy_from_slice(&oversized.to_be_bytes());
        assert!(decode_frame(&buf).is_err());
    }

    #[test]
    fn decode_frame_rejects_overflow() {
        // Construct a long frame header with u64::MAX to trigger overflow
        let mut buf = vec![0u8; 9];
        buf[0] = 0x02; // LONG flag
        buf[1..9].copy_from_slice(&u64::MAX.to_be_bytes());
        assert!(decode_frame(&buf).is_err());
    }

    #[test]
    fn greeting_rejects_mechanism_padding() {
        let mut greeting = build_greeting();
        greeting[20] = 0xFF; // Non-zero in mechanism padding area (bytes 16..32)
        assert!(validate_greeting(&greeting).is_err());
    }

    #[test]
    fn greeting_rejects_old_minor_version() {
        let mut greeting = build_greeting();
        greeting[11] = 0; // Minor version 0 < required 1
        assert!(validate_greeting(&greeting).is_err());
    }

    #[test]
    fn decode_frames_to_flat_skips_commands() {
        // Build: command frame + message frame
        let cmd_frame = encode_command("READY", b"metadata");
        let msg_frame = encode_frame(false, false, b"hello");
        let mut buf = Vec::new();
        // Set MORE on command frame so parsing continues
        buf.extend_from_slice(&cmd_frame);
        // Manually set MORE flag on the command frame
        buf[0] |= 0x01; // MORE=1
        buf.extend_from_slice(&msg_frame);

        let flat = decode_frames_to_flat(&buf).unwrap();
        let parts = decode_flat_parts(&flat).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], b"hello");
    }

    #[test]
    fn socket_type_roundtrip() {
        for st in [
            ZmqSocketType::Req,
            ZmqSocketType::Rep,
            ZmqSocketType::Pub,
            ZmqSocketType::Sub,
            ZmqSocketType::XPub,
            ZmqSocketType::XSub,
            ZmqSocketType::Push,
            ZmqSocketType::Pull,
        ] {
            let s = st.as_str();
            let parsed = ZmqSocketType::from_str(s).unwrap();
            assert_eq!(parsed, st);
        }
    }
}
