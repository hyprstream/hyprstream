//! atproto TID (Timestamp Identifier) — lexicographically-sortable record keys.
//!
//! A TID is a 13-character base32-sortable integer. A timestamp constructor
//! packs 53 microsecond-timestamp bits above a 10-bit per-actor clock id.
//! The sorted alphabet makes string order agree with numeric order.
//!
//! # Format
//!
//! The alphabet is `234567abcdefghijklmnopqrstuvwxyz` (32 symbols). The raw
//! 64-bit integer is represented by 13 radix-32 digits, with a zero 65th bit
//! at the LEFT. Thus the first digit is restricted to `2..7` or `a..j`, while
//! the final digit carries all five low bits. This is integer encoding, not
//! byte-oriented base32 with right-side padding bits.

use std::fmt;

use anyhow::{bail, Result};

/// The 32-symbol TID base32 alphabet (sorted: '2' < ... < 'z').
const TID_ALPHABET: &[u8; 32] = b"234567abcdefghijklmnopqrstuvwxyz";
const TID_LEN: usize = 13;

/// A 64-bit atproto TID.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Tid(u64);

impl Tid {
    /// Construct a TID from raw 64-bit integer bits.
    pub const fn from_raw(bits: u64) -> Self {
        Tid(bits)
    }

    /// The raw 64-bit integer value of this TID.
    ///
    /// Used by callers that need to advance a revision monotonically (the
    /// atproto commit `rev` must strictly increase), e.g. bumping past a
    /// previous commit's rev when the wall clock has not moved.
    pub const fn to_raw(self) -> u64 {
        self.0
    }

    /// Build a TID from a microsecond timestamp (high 53 bits) and a 10-bit
    /// clock id (low 10 bits).
    ///
    /// `clock_id` is masked to 10 bits. For tie-breaking across records written
    /// in the same microsecond, callers should use a random per-actor value.
    pub fn from_micros(micros: u64, clock_id: u16) -> Self {
        let ts = micros & ((1u64 << 53) - 1);
        let clk = (clock_id as u64) & 0x3ff;
        Tid((ts << 10) | clk)
    }

    /// Current TID from the system clock, with a fixed clock id of 0.
    ///
    /// Production callers should pass a per-actor random clock id via
    /// [`Tid::from_micros`]; this convenience is for tests/initial state.
    pub fn now() -> Self {
        let micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        Tid::from_micros(micros, 0)
    }

    /// Encode to the canonical 13-character base32 string.
    ///
    /// (Named `encode` rather than `to_string` to avoid shadowing the
    /// `ToString` blanket impl that `Display` would otherwise recurse through.)
    pub fn encode(self) -> String {
        // Zero-extend the raw integer on the left: digits carry bits
        // [64..60], [59..55], ..., [4..0]. Never shift the value left.
        let mut out = vec![TID_ALPHABET[0]; TID_LEN];
        for (i, slot) in out.iter_mut().enumerate() {
            // Symbol i carries bits [64-5i .. 60-5i] of the 65-bit field.
            // Shift the group's least significant bit down to bit zero.
            let shift = 60 - 5 * i;
            let idx = ((self.0 >> shift) & 0x1f) as usize;
            *slot = TID_ALPHABET[idx];
        }
        // TID_ALPHABET is ASCII, so from_utf8 is infallible in practice; use
        // from_utf8_unchecked's safe equivalent (unwrap_or with a fallback) to
        // satisfy the workspace's `expect_used = "deny"` lint without panicking.
        String::from_utf8(out).unwrap_or_else(|_| String::new())
    }

    /// Parse a 13-character base32 TID string.
    ///
    /// (Named `parse` rather than `from_str` to avoid confusion with the
    /// `std::str::FromStr` trait, which we intentionally do not implement
    /// because it would require a blanket `Sized` bound we don't want here.)
    pub fn parse(s: &str) -> Result<Self> {
        if s.len() != TID_LEN {
            bail!("TID must be {TID_LEN} chars, got {}", s.len());
        }
        let mut value = 0u64;
        for (position, &byte) in s.as_bytes().iter().enumerate() {
            let digit = match TID_ALPHABET.iter().position(|&a| a == byte) {
                Some(i) => i as u64,
                None => bail!("invalid TID char {byte:?}"),
            };
            if position == 0 && digit >= 16 {
                bail!("TID leading digit exceeds the 64-bit range");
            }
            value = value
                .checked_mul(32)
                .and_then(|value| value.checked_add(digit))
                .ok_or_else(|| anyhow::anyhow!("TID overflow"))?;
        }
        Ok(Tid(value))
    }
}

impl fmt::Display for Tid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.encode())
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;

    #[test]
    fn tid_raw_integer_vectors_cover_low_bits_and_full_range() {
        for (raw, text) in [
            (0, "2222222222222"),
            (1, "2222222222223"),
            (31, "222222222222z"),
            (32, "2222222222232"),
            (1u64 << 60, "3222222222222"),
            ((1u64 << 63) - 1, "bzzzzzzzzzzzz"),
            (1u64 << 63, "c222222222222"),
            (u64::MAX, "jzzzzzzzzzzzz"),
        ] {
            assert_eq!(Tid::from_raw(raw).encode(), text);
            assert_eq!(Tid::parse(text).unwrap().to_raw(), raw);
        }
        let maximum_timestamp = Tid::from_micros((1u64 << 53) - 1, 1023);
        assert_eq!(maximum_timestamp.to_raw(), (1u64 << 63) - 1);
        assert_eq!(maximum_timestamp.encode(), "bzzzzzzzzzzzz");
    }

    #[test]
    fn tid_preserves_every_final_digit_and_valid_odd_low_bit() {
        let even = Tid::parse("3jzfcijpj2z2a").unwrap();
        let odd = Tid::parse("3jzfcijpj2z2b").unwrap();
        assert_eq!(odd.to_raw(), even.to_raw() + 1);
        assert_eq!(odd.encode(), "3jzfcijpj2z2b");
        for &digit in TID_ALPHABET {
            let text = format!("3jzfcijpj2z2{}", digit as char);
            assert_eq!(Tid::parse(&text).unwrap().encode(), text);
        }
    }

    #[test]
    fn tid_rejects_out_of_range_leading_digits_and_noncanonical_text() {
        for (index, &digit) in TID_ALPHABET.iter().enumerate() {
            let text = format!("{}jzfcijpj2z2a", digit as char);
            assert_eq!(Tid::parse(&text).is_ok(), index < 16, "{text}");
        }
        for text in [
            "kjzfcijpj2z2a",
            "KJZFCIJPJ2Z2A",
            "3JZFCIJPJ2Z2B",
            "3jzfcijpj2z2b=",
            "3jzfcijpj2z2",
            "3jzfcijpj2z21",
        ] {
            assert!(Tid::parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn tid_string_round_trip() {
        for raw in [0u64, 1, 0x1234_5678, 0x7fff_ffff_ffff, 1u64 << 63, u64::MAX] {
            let tid = Tid::from_raw(raw);
            let s = tid.encode();
            assert_eq!(s.len(), TID_LEN, "tid {raw}");
            let back = Tid::parse(&s).expect("round-trip");
            assert_eq!(tid, back, "tid {raw}: {s}");
        }
    }

    #[test]
    fn tid_lexicographic_matches_chrono() {
        // String sort == integer sort == chronological sort.
        let earlier = Tid::from_micros(1_000_000, 0); // 1s
        let later = Tid::from_micros(2_000_000, 0); // 2s
        assert!(earlier < later);
        assert!(
            earlier.encode() < later.encode(),
            "string order must match chrono"
        );
    }

    #[test]
    fn tid_clock_id_breaks_ties() {
        let a = Tid::from_micros(5_000_000, 1);
        let b = Tid::from_micros(5_000_000, 2);
        assert!(a < b, "clock id must break microsecond ties");
    }

    #[test]
    fn tid_rejects_bad_length() {
        assert!(Tid::parse("too-short").is_err());
        assert!(Tid::parse("0uppercase!").is_err());
    }
}
