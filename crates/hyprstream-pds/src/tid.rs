//! atproto TID (Timestamp Identifier) — lexicographically-sortable record keys.
//!
//! A TID is a 13-character base32-sorted string encoding a 64-bit integer whose
//! high 53 bits are a microsecond timestamp and low 10 bits are a per-actor
//! "clock id" (random, to break ties within the same microsecond). Because the
//! base32 alphabet is sorted (`2..7`, then `a..z`) and the timestamp occupies
//! the high bits, string comparison of TIDs matches chronological order — which
//! is exactly what the MST relies on to keep record keys in order.
//!
//! # Format
//!
//! The 64-bit integer is big-endian base32-encoded with the 13-symbol alphabet
//! `234567abcdefghijklmnopqrstuvwxyz` (RFC 4648 base32 without padding, but
//! using lowercase). The leading bit is always 0 (the timestamp is capped at
//! 2^53), giving 53 bits of timestamp headroom until ~2242 CE.

use std::fmt;

use anyhow::{bail, Result};

/// The 13-symbol TID base32 alphabet (sorted: '2' < ... < 'z').
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
        // 13 base32 symbols carry 65 bits. The 64-bit TID value occupies the
        // low 64 bits of that 65-bit space; the top (65th) bit is always 0
        // (the timestamp is capped at 2^53, so bit 63 is 0 in practice too).
        // We emit big-endian, 5 bits per symbol, MSB first: symbol 0 carries
        // bits [64..60], symbol 1 carries bits [59..55], …, symbol 12 carries
        // bits [4..0].
        //
        // Concretely: shift the value left by 1 to place it in a 65-bit field,
        // then extract 5-bit groups from the top.
        let val65 = (self.0 as u128) << 1; // 65-bit quantity (top bit 0)
        let mut out = vec![TID_ALPHABET[0]; TID_LEN];
        for (i, slot) in out.iter_mut().enumerate() {
            // Symbol i carries bits [64-5i .. 60-5i] of the 65-bit field.
            // Extract by shifting the *upper* bit position down to 0.
            let shift = 60 - 5 * i;
            let idx = ((val65 >> shift) & 0x1f) as usize;
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
        // Inverse of encode: accumulate 13 symbols × 5 bits = 65 bits into a
        // u128 (big-endian), then drop the top padding bit and take the low 64.
        let mut val65: u128 = 0;
        for &byte in s.as_bytes().iter() {
            let idx = match TID_ALPHABET.iter().position(|&a| a == byte) {
                Some(i) => i as u128,
                None => bail!("invalid TID char {byte:?}"),
            };
            val65 = val65
                .checked_shl(5)
                .ok_or_else(|| anyhow::anyhow!("TID overflow"))?;
            val65 |= idx;
        }
        Ok(Tid((val65 >> 1) as u64))
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
    fn tid_string_round_trip() {
        for raw in [0u64, 1, 0x1234_5678, 0x7fff_ffff_ffff] {
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
