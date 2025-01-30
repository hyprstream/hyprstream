// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! A collection of useful functions, constants which are in use across the whole crate

pub const POW10: [u64; 20] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
    10_000_000_000_000,
    100_000_000_000_000,
    1_000_000_000_000_000,
    10_000_000_000_000_000,
    100_000_000_000_000_000,
    1_000_000_000_000_000_000,
    10_000_000_000_000_000_000,
];

pub trait FloorLog10 {
    fn floor_log10(&self) -> u32;
}

pub trait FloorLog2 {
    fn floor_log2(&self) -> u32;
}

impl FloorLog10 for u64 {
    // This solution is based on a post https://stackoverflow.com/a/25934909 by Stephen Canon
    #[inline]
    fn floor_log10(&self) -> u32 {
        const GUESS_TABLE: [u32; 64] = [
            0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8,
            8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15,
            15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 18,
        ];
        let estimate = GUESS_TABLE[self.floor_log2() as usize];
        let over_estimate = estimate + 1;
        if *self >= POW10[over_estimate as usize] {
            over_estimate
        } else {
            estimate
        }
    }
}

impl FloorLog2 for u64 {
    #[inline]
    fn floor_log2(&self) -> u32 {
        63u32
            .checked_sub(self.leading_zeros())
            .expect("Logarithm of zero is undefined")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_log2_for_u64_at_critical_margins() {
        for i in 0..63 {
            let pow = 1u64 << i;
            assert_eq!(pow.floor_log2(), i);
            if i > 0 {
                assert_eq!(pow.saturating_sub(1).floor_log2(), i.saturating_sub(1));
                assert_eq!(pow.saturating_add(1).floor_log2(), i);
            }
        }
    }
    #[test]
    fn test_floor_log2_for_u64_when_u64_max() {
        assert_eq!(u64::MAX.floor_log2(), 63);
    }

    #[test]
    #[should_panic = "Logarithm of zero is undefined"]
    fn test_floor_log2_for_u64_when_zero() {
        0u64.floor_log2();
    }

    #[test]
    fn test_floor_log10_for_u64_at_critical_margins() {
        for i in 0..19u32 {
            let pow = POW10[i as usize];
            assert_eq!(pow.floor_log10(), i);
            if i != 0 {
                assert_eq!(pow.saturating_sub(1).floor_log10(), i.saturating_sub(1));
            }
            assert_eq!(pow.saturating_add(1).floor_log10(), i);
        }
    }

    #[test]
    fn test_floor_log10_for_u64_when_u64_max() {
        assert_eq!(u64::MAX.floor_log10(), 19);
    }

    #[test]
    #[should_panic = "Logarithm of zero is undefined"]
    fn test_floor_log10_for_u64_when_zero() {
        0u64.floor_log10();
    }
}
