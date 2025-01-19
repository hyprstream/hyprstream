use argminmax::NaNArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

use argminmax::dtype_strategy::FloatReturnNaN;
use argminmax::scalar::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::simd::{SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(target_arch = "aarch64")]
use argminmax::simd::{SIMDArgMinMax, NEON};

// _rn stands for "return nan"

fn argminmax_rn_f64_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[f64] = &utils::SampleUniformFullRange::get_random_array(n);
    c.bench_function("scalar_f64_argminmax_rn", |b| {
        b.iter(|| SCALAR::<FloatReturnNaN>::argminmax(black_box(data)))
    });
    c.bench_function("scalar_f64_argmin_rn", |b| {
        b.iter(|| SCALAR::<FloatReturnNaN>::argmin(black_box(data)))
    });
    c.bench_function("scalar_f64_argmax_rn", |b| {
        b.iter(|| SCALAR::<FloatReturnNaN>::argmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_f64_argminmax_rn", |b| {
            b.iter(|| unsafe { SSE::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_f64_argmin_rn", |b| {
            b.iter(|| unsafe { SSE::<FloatReturnNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_f64_argmax_rn", |b| {
            b.iter(|| unsafe { SSE::<FloatReturnNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f64_argminmax_rn", |b| {
            b.iter(|| unsafe { AVX2::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f64_argmin_rn", |b| {
            b.iter(|| unsafe { AVX2::<FloatReturnNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f64_argmax_rn", |b| {
            b.iter(|| unsafe { AVX2::<FloatReturnNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_f64_argminmax_rn", |b| {
            b.iter(|| unsafe { AVX512::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_f64_argmin_rn", |b| {
            b.iter(|| unsafe { AVX512::<FloatReturnNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_f64_argmax_rn", |b| {
            b.iter(|| unsafe { AVX512::<FloatReturnNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f64_argminmax_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f64_argmin_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f64_argmax_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argmax(black_box(data)) })
        });
    }
    c.bench_function("impl_f64_argminmax_rn", |b| {
        b.iter(|| black_box(data.nanargminmax()))
    });
    c.bench_function("impl_f64_argmin_rn", |b| {
        b.iter(|| black_box(data.nanargmin()))
    });
    c.bench_function("impl_f64_argmax_rn", |b| {
        b.iter(|| black_box(data.nanargmax()))
    });
}

criterion_group!(benches, argminmax_rn_f64_random_array_long,);
criterion_main!(benches);
