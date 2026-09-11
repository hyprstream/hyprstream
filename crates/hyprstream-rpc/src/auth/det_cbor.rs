//! The ONE deterministic-CBOR (RFC 8949) byte encoder for v16 content
//! thumbprints. A direct port of WS-A's frozen `gen_proof_vectors.py::enc`
//! (at `af08825528627069b3cfbdd763948c4a24689cb5`), restricted to the value
//! subset the v16 content thumbprints use.
//!
//! Both the mint side (WS-B `signer_suite_thumbprint`) and the verify side
//! (WS-C signer-suite/cnf resolution AND the replay-namespace preimages, which
//! reuse the identical `enc` primitive over different array shapes) build their
//! preimages with THIS encoder — there is exactly one det-CBOR implementation
//! in the crate, so mint and verify are provably byte-identical.

/// A value in the RFC 8949 deterministic-CBOR subset the v16 content
/// thumbprints use: unsigned integers, byte strings, text strings, and
/// (possibly nested) definite-length arrays. This is all A's `enc` needs for
/// the signer-suite and replay-namespace preimages.
#[derive(Clone, Debug)]
pub enum DetCborValue<'a> {
    /// CBOR major type 0 (unsigned integer) — e.g. an enrollment epoch.
    Uint(u64),
    /// CBOR major type 2 (byte string) — a raw public key.
    Bytes(&'a [u8]),
    /// CBOR major type 3 (text string) — a suite ID / domain separator.
    Text(&'a str),
    /// CBOR major type 4 (definite-length array).
    Array(Vec<DetCborValue<'a>>),
}

/// Append the RFC 8949 deterministic head for major type `major` and
/// length/value `n` (shortest form).
fn head(out: &mut Vec<u8>, major: u8, n: u64) {
    let mt = major << 5;
    if n < 24 {
        out.push(mt | (n as u8));
    } else if n < 0x100 {
        out.push(mt | 24);
        out.push(n as u8);
    } else if n < 0x1_0000 {
        out.push(mt | 25);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else if n < 0x1_0000_0000 {
        out.push(mt | 26);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    } else {
        out.push(mt | 27);
        out.extend_from_slice(&n.to_be_bytes());
    }
}

fn append(out: &mut Vec<u8>, value: &DetCborValue) {
    match value {
        DetCborValue::Uint(n) => head(out, 0, *n),
        DetCborValue::Bytes(b) => {
            head(out, 2, b.len() as u64);
            out.extend_from_slice(b);
        }
        DetCborValue::Text(s) => {
            let b = s.as_bytes();
            head(out, 3, b.len() as u64);
            out.extend_from_slice(b);
        }
        DetCborValue::Array(items) => {
            head(out, 4, items.len() as u64);
            for item in items {
                append(out, item);
            }
        }
    }
}

/// Deterministically CBOR-encode `value` to its canonical byte string.
#[must_use]
pub fn det_cbor(value: &DetCborValue) -> Vec<u8> {
    let mut out = Vec::new();
    append(&mut out, value);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_use_shortest_form() {
        // 0..23 encode in one byte; 24 spills to a 1-byte length.
        assert_eq!(det_cbor(&DetCborValue::Uint(0)), vec![0x00]);
        assert_eq!(det_cbor(&DetCborValue::Uint(23)), vec![0x17]);
        assert_eq!(det_cbor(&DetCborValue::Uint(24)), vec![0x18, 24]);
        assert_eq!(det_cbor(&DetCborValue::Uint(255)), vec![0x18, 255]);
        assert_eq!(det_cbor(&DetCborValue::Uint(256)), vec![0x19, 0x01, 0x00]);
    }

    #[test]
    fn text_and_array_shapes() {
        // ["a", []] → 82 61 61 80
        let v = DetCborValue::Array(vec![
            DetCborValue::Text("a"),
            DetCborValue::Array(vec![]),
        ]);
        assert_eq!(det_cbor(&v), vec![0x82, 0x61, 0x61, 0x80]);
    }
}
