//! The pinned benchmark PRNG.
//!
//! Item generation must be bit-reproducible forever: the frozen manifest
//! (hashes) is the contract that P1.3's contamination firewall and the P1.4
//! gates consume, so the random stream is part of the released artifact. We
//! therefore pin our own generator instead of depending on an external `rand`
//! crate whose stream semantics may change across versions.
//!
//! Algorithm: **splitmix64**, counter-based. `FROZEN: changing anything in
//! this file invalidates every published item hash.`

/// A splitmix64 stream. State advances by the golden-ratio constant on every
/// draw; the output is the standard splitmix64 finalizer.
#[derive(Debug, Clone)]
pub struct BenchRng {
    state: u64,
}

impl BenchRng {
    /// Start a stream from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Derive an independent stream for (seed, stream index) pairs.
    pub fn keyed(seed: u64, key: u64) -> Self {
        let mut root = Self::new(seed);
        let mixed = root.next_u64() ^ key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        Self::new(mixed)
    }

    /// The next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n` (rejection-free multiply-high; bias is
    /// negligible at our n ≤ 255 and identical on every platform).
    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Uniform integer in `lo..=hi`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        debug_assert!(lo <= hi);
        lo + self.below((hi - lo + 1) as u64) as i64
    }

    /// Fisher–Yates shuffle of `0..n` drawn from this stream.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i as u64 + 1) as usize;
            order.swap(i, j);
        }
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_is_deterministic() {
        let mut a = BenchRng::new(42);
        let mut b = BenchRng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn pinned_golden_values() {
        // Regression lock: if these change, every frozen item hash changes.
        let mut rng = BenchRng::new(0);
        let values: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_eq!(
            values,
            vec![
                0xE220_A839_7B1D_CDAF,
                0x6E78_9E6A_A1B9_65F4,
                0x06C4_5D18_8009_454F,
                0xF88B_B8A8_724C_81EC,
            ]
        );
    }
}
