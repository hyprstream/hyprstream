#![allow(non_upper_case_globals)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::common::{divmod, lookup, write4, write4_pad};
use core::ptr;

#[repr(align(16))]
struct Aligned<T>(T);

impl<T> std::ops::Deref for Aligned<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

const kDiv10000: u32 = 0xd1b71759;
const kDivPowersVector: Aligned<[u16; 8]> =
    Aligned([8389, 5243, 13108, 32768, 8389, 5243, 13108, 32768]);
const kShiftPowersVector: Aligned<[u16; 8]> = Aligned([
    1 << (16 - (23 + 2 - 16)),
    1 << (16 - (19 + 2 - 16)),
    1 << (16 - 1 - 2),
    1 << (15),
    1 << (16 - (23 + 2 - 16)),
    1 << (16 - (19 + 2 - 16)),
    1 << (16 - 1 - 2),
    1 << (15),
]);

#[inline]
unsafe fn convert_8digits_sse2(value: u32) -> __m128i {
    debug_assert!(value <= 99999999);

    // abcd, efgh = abcdefgh divmod 10000
    let abcdefgh = _mm_cvtsi32_si128(value as i32);
    let abcd = _mm_srli_epi64(
        _mm_mul_epu32(abcdefgh, _mm_set1_epi32(kDiv10000 as i32)),
        45,
    );
    let efgh = _mm_sub_epi32(abcdefgh, _mm_mul_epu32(abcd, _mm_set1_epi32(10000)));

    // v1 = [ abcd, efgh, 0, 0, 0, 0, 0, 0 ]
    let v1 = _mm_unpacklo_epi16(abcd, efgh);

    // v1a = v1 * 4 = [ abcd*4, efgh*4, 0, 0, 0, 0, 0, 0 ]
    let v1a = _mm_slli_epi64(v1, 2);

    // v2 = [abcd*4, abcd*4, abcd*4, abcd*4, efgh*4, efgh*4, efgh*4, efgh*4]
    let v2a = _mm_unpacklo_epi16(v1a, v1a);
    let v2 = _mm_unpacklo_epi32(v2a, v2a);

    // v4 = v2 div 10^3, 10^2, 10^1, 10^0 = [ a, ab, abc, abcd, e, ef, efg, efgh ]
    let v3 = _mm_mulhi_epu16(
        v2,
        _mm_load_si128(kDivPowersVector.as_ptr() as *const __m128i),
    );
    let v4 = _mm_mulhi_epu16(
        v3,
        _mm_load_si128(kShiftPowersVector.as_ptr() as *const __m128i),
    );

    // v5 = v4 * 10 = [ a0, ab0, abc0, abcd0, e0, ef0, efg0, efgh0 ]
    let v5 = _mm_mullo_epi16(v4, _mm_set1_epi16(10));

    // v6 = v5 << 16 = [ 0, a0, ab0, abc0, 0, e0, ef0, efg0 ]
    let v6 = _mm_slli_epi64(v5, 16);

    // v4 - v6 = { a, b, c, d, e, f, g, h }
    _mm_sub_epi16(v4, v6)
}

pub unsafe fn write_u32(n: u32, buf: *mut u8) -> usize {
    if n < 10000 {
        write4(n, buf)
    } else if n < 100_000_000 {
        let (n1, n2) = divmod(n, 10000);

        let l = write4(n1, buf);
        write4_pad(n2, buf.add(l));
        l + 4
    } else {
        let (n1, n2) = divmod(n, 100_000_000);

        let l = if n1 >= 10 {
            ptr::copy_nonoverlapping(lookup(n1), buf, 2);
            2
        } else {
            *buf = n1 as u8 + 0x30;
            1
        };

        let b = convert_8digits_sse2(n2);
        let ba = _mm_add_epi8(
            _mm_packus_epi16(_mm_setzero_si128(), b),
            _mm_set1_epi8(b'0' as i8),
        );
        let result = _mm_srli_si128(ba, 8);
        _mm_storel_epi64(buf.add(l) as *mut __m128i, result);

        l + 8
    }
}

pub unsafe fn write_u64(n: u64, buf: *mut u8) -> usize {
    if n < 10000 {
        write4(n as u32, buf)
    } else if n < 100_000_000 {
        let (n1, n2) = divmod(n as u32, 10000);

        let l = write4(n1, buf);
        write4_pad(n2, buf.add(l));
        l + 4
    } else if n < 10_000_000_000_000_000 {
        let (n1, n2) = divmod(n, 100_000_000);
        let (n1, n2) = (n1 as u32, n2 as u32);

        let l = if n1 < 10000 {
            write4(n1, buf)
        } else {
            let (n11, n12) = divmod(n1, 10000);
            let l = write4(n11, buf);
            write4_pad(n12, buf.add(l));
            l + 4
        };

        let b = convert_8digits_sse2(n2);
        let ba = _mm_add_epi8(
            _mm_packus_epi16(_mm_setzero_si128(), b),
            _mm_set1_epi8(b'0' as i8),
        );
        let result = _mm_srli_si128(ba, 8);
        _mm_storel_epi64(buf.add(l) as *mut __m128i, result);

        l + 8
    } else {
        let (n1, n2) = divmod(n, 10_000_000_000_000_000);
        let l = write4(n1 as u32, buf);

        let (n21, n22) = divmod(n2, 100_000_000);

        let a0 = convert_8digits_sse2(n21 as u32);
        let a1 = convert_8digits_sse2(n22 as u32);

        // Convert to bytes, add '0'
        let va = _mm_add_epi8(_mm_packus_epi16(a0, a1), _mm_set1_epi8(b'0' as i8));
        _mm_storeu_si128(buf.add(l) as *mut __m128i, va);

        l + 16
    }
}
