//! Raw-byte CBOR deterministic-encoding audit.
//!
//! Validates RFC 8949 core deterministic encoding constraints at the byte
//! level BEFORE `ciborium` deserializes (which resolves indefinite lengths
//! and non-minimal integers transparently, destroying the evidence).
//!
//! Enforces:
//! - No indefinite-length maps (0xBF), arrays (0x9F), or byte/text strings
//!   (0x7B, 0x5F, 0x3F)
//! - No CBOR tags (major type 6: 0xC0–0xDB)
//! - No floating-point values (major type 7 simple: 0xF9, 0xFA, 0xFB for
//!   half/single/double)
//! - Definite-length integers are minimal (no non-minimal encodings)
//! - No trailing bytes (for top-level items)
//!
//! Returns `Ok(())` if the bytes conform to RFC 8949 core deterministic
//! encoding. Returns `Err` with a description of the violation otherwise.

use anyhow::{bail, Result};

// Keep raw-audit recursion aligned with the bounded Cap'n Proto decoders used
// by the proof path. The audit runs before ciborium builds its value tree, so
// this bound rejects deeply nested attacker input before either decoder can
// consume unbounded stack.
const MAX_CBOR_AUDIT_DEPTH: usize = 64;

/// Audit a complete CBOR byte string for deterministic-encoding compliance.
/// Rejects trailing data.
pub fn audit_deterministic(bytes: &[u8]) -> Result<()> {
    let mut pos = 0;
    let consumed = audit_one(bytes, &mut pos, 0)?;
    if consumed != bytes.len() {
        bail!(
            "deterministic CBOR: trailing data: consumed {consumed} of {} bytes",
            bytes.len()
        );
    }
    Ok(())
}

/// Audit one CBOR data item starting at `*pos`. Returns the end position.
fn audit_one(bytes: &[u8], pos: &mut usize, depth: usize) -> Result<usize> {
    if depth > MAX_CBOR_AUDIT_DEPTH {
        bail!("deterministic CBOR: nesting exceeds depth {MAX_CBOR_AUDIT_DEPTH}");
    }

    let start = *pos;
    if *pos >= bytes.len() {
        bail!("deterministic CBOR: unexpected end of input");
    }

    let initial = bytes[*pos];
    let major = initial >> 5;
    let info = initial & 0x1f;

    // Reject indefinite-length encodings (info == 31) for all major types
    // except break (which is 0xFF and handled by the caller of arrays/maps).
    if info == 31 && major != 7 {
        bail!(
            "deterministic CBOR: indefinite-length encoding (0x{:02x}) at byte {start} denied",
            initial
        );
    }

    // Reject tags (major type 6).
    if major == 6 {
        bail!(
            "deterministic CBOR: tag (0x{:02x}) at byte {start} denied",
            initial
        );
    }

    // Reject floating-point (major type 7, info 25/26/27 = half/single/double).
    if major == 7 && (info == 25 || info == 26 || info == 27) {
        bail!(
            "deterministic CBOR: floating-point value (0x{:02x}) at byte {start} denied",
            initial
        );
    }

    // Advance past the initial byte.
    *pos += 1;

    match major {
        0 | 1 => {
            // Unsigned / negative integer.
            let arg = read_arg(bytes, pos, info, initial)?;
            // Check minimality: the argument encoding must be the shortest
            // possible. For info 0..23 the value is inline (always minimal).
            // For info 24 (1-byte), the value MUST be ≥ 24.
            // For info 25 (2-byte), the value MUST be ≥ 256.
            // For info 26 (4-byte), the value MUST be ≥ 65536.
            // For info 27 (8-byte), the value MUST be ≥ 4294967296.
            if info >= 24 {
                check_int_minimal(info, arg, start, bytes)?;
            }
        }
        2 | 3 => {
            // Byte string / text string.
            let len = read_usize_arg(bytes, pos, info, initial, "string length")?;
            let end = pos.checked_add(len).ok_or_else(|| {
                anyhow::anyhow!(
                    "deterministic CBOR: {} string length {len} overflows input position at byte {start}",
                    if major == 2 { "byte" } else { "text" }
                )
            })?;
            if end > bytes.len() {
                bail!(
                    "deterministic CBOR: {} string length {len} exceeds remaining input at byte {start}",
                    if major == 2 { "byte" } else { "text" }
                );
            }
            *pos = end;
        }
        4 => {
            // Array.
            let count = read_usize_arg(bytes, pos, info, initial, "array length")?;
            for _ in 0..count {
                audit_one(bytes, pos, depth + 1)?;
            }
        }
        5 => {
            // Map.
            let count = read_usize_arg(bytes, pos, info, initial, "map length")?;
            let mut prev_key: Option<Vec<u8>> = None;
            for _ in 0..count {
                // Record the key bytes for canonical ordering check.
                let key_start = *pos;
                audit_one(bytes, pos, depth + 1)?;
                let key_bytes = bytes[key_start..*pos].to_vec();

                // Check canonical key ordering.
                if let Some(ref pk) = prev_key {
                    if key_bytes.as_slice() <= pk.as_slice() {
                        if key_bytes == *pk {
                            bail!("deterministic CBOR: duplicate map key at byte {key_start}");
                        }
                        bail!("deterministic CBOR: map keys not in canonical order at byte {key_start}");
                    }
                }
                prev_key = Some(key_bytes);

                // Audit the value.
                audit_one(bytes, pos, depth + 1)?;
            }
        }
        7 => {
            // Major type 7: simple values, float (rejected above), break.
            if info == 31 {
                // Break code (0xFF) should only appear inside indefinite-length
                // containers, which we already rejected. If we reach here it's
                // a bare break code — invalid.
                bail!("deterministic CBOR: unexpected break code (0xFF) at byte {start}");
            }
            // Simple values 0..23 are inline. 24 means next byte is the value.
            if info == 24 {
                if *pos >= bytes.len() {
                    bail!("deterministic CBOR: truncated simple value at byte {start}");
                }
                // RFC 8949: the value in the following byte MUST be ≥ 32 to
                // be canonical (values 0..23 use inline encoding). But we
                // don't enforce that here since it's not a security concern
                // for this profile (simple values are not used).
                *pos += 1;
            }
            // info 25/26/27 (floats) already rejected above.
            // info 20 (False), 21 (True), 22 (Null), 23 (Undefined) are inline.
        }
        _ => {
            // Major type 6 (tags) already rejected above.
            bail!("deterministic CBOR: unexpected major type {major} at byte {start}");
        }
    }

    Ok(*pos)
}

/// Read a CBOR argument that is used as an in-memory length or item count.
///
/// CBOR arguments are `u64`, while slices and loop bounds use `usize`.
/// Refuse values that cannot be represented on this platform rather than
/// truncating them before checking the remaining input.
fn read_usize_arg(
    bytes: &[u8],
    pos: &mut usize,
    info: u8,
    initial: u8,
    description: &str,
) -> Result<usize> {
    let value = read_arg(bytes, pos, info, initial)?;
    usize::try_from(value).map_err(|_| {
        anyhow::anyhow!(
            "deterministic CBOR: {description} {value} does not fit this platform's address space"
        )
    })
}

/// Read the argument value for a CBOR head byte.
fn read_arg(bytes: &[u8], pos: &mut usize, info: u8, initial: u8) -> Result<u64> {
    match info {
        0..=23 => Ok(info as u64),
        24 => {
            if *pos >= bytes.len() {
                bail!("deterministic CBOR: truncated 1-byte argument at initial 0x{initial:02x}");
            }
            let v = bytes[*pos] as u64;
            *pos += 1;
            Ok(v)
        }
        25 => {
            if *pos + 2 > bytes.len() {
                bail!("deterministic CBOR: truncated 2-byte argument at initial 0x{initial:02x}");
            }
            let v = u16::from_be_bytes([bytes[*pos], bytes[*pos + 1]]) as u64;
            *pos += 2;
            Ok(v)
        }
        26 => {
            if *pos + 4 > bytes.len() {
                bail!("deterministic CBOR: truncated 4-byte argument at initial 0x{initial:02x}");
            }
            let v = u32::from_be_bytes([
                bytes[*pos],
                bytes[*pos + 1],
                bytes[*pos + 2],
                bytes[*pos + 3],
            ]) as u64;
            *pos += 4;
            Ok(v)
        }
        27 => {
            if *pos + 8 > bytes.len() {
                bail!("deterministic CBOR: truncated 8-byte argument at initial 0x{initial:02x}");
            }
            let v = u64::from_be_bytes([
                bytes[*pos],
                bytes[*pos + 1],
                bytes[*pos + 2],
                bytes[*pos + 3],
                bytes[*pos + 4],
                bytes[*pos + 5],
                bytes[*pos + 6],
                bytes[*pos + 7],
            ]);
            *pos += 8;
            Ok(v)
        }
        28..=30 => bail!("deterministic CBOR: reserved info {info} at initial 0x{initial:02x}"),
        31 => bail!("deterministic CBOR: indefinite-length argument in definite context"),
        _ => bail!("deterministic CBOR: invalid info {info} (impossible: info is 5-bit)"),
    }
}

/// Check that a multi-byte integer encoding is minimal (RFC 8949 §3.1).
fn check_int_minimal(info: u8, arg: u64, start: usize, _bytes: &[u8]) -> Result<()> {
    match info {
        24 => {
            if arg < 24 {
                bail!(
                    "deterministic CBOR: non-minimal 1-byte integer {arg} (< 24) at byte {start}"
                );
            }
        }
        25 => {
            if arg < 256 {
                bail!(
                    "deterministic CBOR: non-minimal 2-byte integer {arg} (< 256) at byte {start}"
                );
            }
        }
        26 => {
            if arg < 65536 {
                bail!(
                    "deterministic CBOR: non-minimal 4-byte integer {arg} (< 65536) at byte {start}"
                );
            }
        }
        27 if arg < 4294967296 => {
            bail!(
                "deterministic CBOR: non-minimal 8-byte integer {arg} (< 4294967296) at byte {start}"
            );
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{audit_deterministic, MAX_CBOR_AUDIT_DEPTH};

    fn nested_arrays(depth: usize) -> Vec<u8> {
        let mut bytes = vec![0x81; depth];
        bytes.push(0x00);
        bytes
    }

    #[test]
    fn cbor_audit_enforces_a_bounded_nesting_depth() {
        // The configured boundary remains valid for legitimate deeply nested
        // CBOR, while one more container is rejected before ciborium runs.
        assert!(audit_deterministic(&nested_arrays(MAX_CBOR_AUDIT_DEPTH)).is_ok());

        assert!(audit_deterministic(&nested_arrays(MAX_CBOR_AUDIT_DEPTH + 1)).is_err());
    }

    #[test]
    fn cbor_audit_rejects_unrepresentable_string_lengths() {
        // Small definite strings remain valid controls.
        assert!(audit_deterministic(&[0x42, 0xaa, 0xbb]).is_ok());
        assert!(audit_deterministic(&[0x62, b'o', b'k']).is_ok());

        // A declared u64::MAX length must fail without wrapping either the
        // usize conversion (on narrower targets) or the input-position add.
        for initial in [0x5b, 0x7b] {
            let mut malformed = vec![initial];
            malformed.extend_from_slice(&u64::MAX.to_be_bytes());
            assert!(audit_deterministic(&malformed).is_err());
        }
    }
}
